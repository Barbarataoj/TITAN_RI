#!/usr/bin/env python
"""
Phase 7 cash-flow layer: price the activity a Phase 7 arm generates.

Reads the pooled contrast table written by scripts/phase7_contrast.py
(--out <csv>; one row per arm x outcome with baseline_mean, arm_mean, diff_mean,
diff_se, diff_mean_statewide, horizon_years) and the unit-cost table
docs/phase7_unit_costs.yml, and writes three ledgers per arm, statewide and per
year: the clinic (revenue minus programme cost), the payer (what the activity
costs it, minus HIV treatment cost averted) and society (health outcomes valued).
Also the break-even reach: the share of eligible patients the programme must
reach for the clinic's annual net to cross zero, from the model's activity per
reached patient-month.

Every unit cost in the yml has a value, a range and a source; entries still at
null are PLACEHOLDERS and the script refuses to price a line until they are
filled (it lists them instead). 2026-10-02.

USAGE
    python scripts/phase7_cashflow.py --contrasts <contrasts_2023_2031.csv> \\
        --costs docs/phase7_unit_costs.yml --out <ledgers.csv> [--arm pc4_chia_central]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
ACTIVITY = ("pc_visits", "hiv_tests", "prep_person_months", "prep_monitoring_visits")
PERSPECTIVES = ("clinic", "payer", "societal")


def _value(entry):
    """A unit-cost entry's value, or None if it is still a placeholder."""
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def weighted_unit(block, payer_mix):
    """Payer-mix-weighted unit value from a {medicaid, commercial, medicare} block; None if any is unset."""
    total = 0.0
    for payer, share in payer_mix.items():
        v = _value(block.get(payer))
        if v is None:
            return None
        total += share * float(v)
    return total


def prep_month_unit(block):
    mix = block.get("product_mix") or {}
    total = 0.0
    for product, share in mix.items():
        v = _value(block.get(product))
        if v is None:
            return None
        total += share * float(v)
    return total


def unit_table(costs):
    """Clinic revenue and payer cost per activity unit; records placeholders."""
    units = {"clinic": {}, "payer": {}}
    missing = []
    rev = costs.get("clinic_revenue", {})
    mix = costs["payer_mix"]
    for act in ACTIVITY:
        block = rev.get(act, {})
        u = prep_month_unit(block) if act == "prep_person_months" else weighted_unit(block, mix)
        if u is None:
            missing.append(f"clinic_revenue.{act}")
        units["clinic"][act] = u
        pv = _value(costs.get("payer_cost", {}).get(act))
        if pv is None:
            missing.append(f"payer_cost.{act}")
        units["payer"][act] = pv
    return units, missing


def program_cost_per_year(costs, reach_visits, reached_pm, invitations):
    pc = costs["program_costs"]
    fixed = _value(pc.get("chia_platform_per_site_year"))
    ma = _value(pc.get("ma_wage_per_hour")); prov = _value(pc.get("provider_wage_per_hour"))
    missing = [k for k, v in (("chia_platform_per_site_year", fixed), ("ma_wage_per_hour", ma),
                              ("provider_wage_per_hour", prov)) if v is None]
    if missing:
        return None, missing
    variable = (invitations * _value(pc["ma_minutes_per_invitation"]) / 60.0 * ma
                + reach_visits * _value(pc["provider_minutes_per_reach_visit"]) / 60.0 * prov)
    return fixed + variable, []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrasts", required=True, type=Path)
    ap.add_argument("--costs", type=Path, default=REPO / "docs" / "phase7_unit_costs.yml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", default=None, help="price one arm only")
    ap.add_argument("--sites", type=int, default=1, help="number of clinics the platform cost is counted for")
    args = ap.parse_args()

    costs = yaml.safe_load(open(args.costs))
    table = pd.read_csv(args.contrasts)
    units, missing = unit_table(costs)
    rows = []
    for arm in sorted(set(table["arm"])):
        if args.arm and arm != args.arm:
            continue
        t = table[table["arm"] == arm].set_index("outcome")
        years = float(t["horizon_years"].iloc[0])

        def diff_statewide(oc):
            return float(t.loc[oc, "diff_mean_statewide"]) if oc in t.index else 0.0

        activity = {act: diff_statewide(act) / years for act in ACTIVITY}       # per year, statewide
        reach_visits = diff_statewide("reach_visits") / years
        reached_pm = diff_statewide("reached_person_months") / years
        infections = -diff_statewide("hiv_new_pwid") / years                     # averted per year
        ledger = {"arm": arm, "horizon_years": years, "infections_averted_per_year": infections,
                  **{f"activity_{k}": v for k, v in activity.items()},
                  "reach_visits_per_year": reach_visits, "reached_person_months_per_year": reached_pm}
        # clinic
        rev = sum(units["clinic"][a] * activity[a] for a in ACTIVITY if units["clinic"][a] is not None)
        prog, pmissing = program_cost_per_year(costs, reach_visits, reached_pm, reach_visits)
        ledger["clinic_revenue_per_year"] = rev
        ledger["program_cost_per_year"] = (prog * args.sites) if prog is not None else None
        ledger["clinic_net_per_year"] = rev - prog * args.sites if prog is not None else None
        # payer
        pay = sum(units["payer"][a] * activity[a] for a in ACTIVITY if units["payer"][a] is not None)
        hiv_cost = _value(costs["health_values"].get("lifetime_hiv_treatment_cost"))
        ledger["payer_activity_cost_per_year"] = pay
        ledger["hiv_treatment_cost_averted_per_year"] = infections * hiv_cost if hiv_cost is not None else None
        ledger["payer_net_per_year"] = (infections * hiv_cost - pay) if hiv_cost is not None else None
        # societal
        qaly = _value(costs["health_values"].get("qaly_per_infection_averted"))
        ledger["qalys_gained_per_year"] = infections * qaly if qaly is not None else None
        ledger["cost_per_infection_averted_clinic_and_payer"] = (
            ((prog or 0.0) * args.sites + pay) / infections if infections > 0 else None)
        # break-even reach: scale the programme's activity linearly with reach
        if prog is not None and reached_pm > 0:
            net_per_reached_pm = (rev - (prog - _value(costs["program_costs"]["chia_platform_per_site_year"]) * 0)) / reached_pm
            fixed = _value(costs["program_costs"]["chia_platform_per_site_year"]) * args.sites
            ledger["breakeven_reached_person_months_per_year"] = fixed / net_per_reached_pm if net_per_reached_pm > 0 else None
        rows.append(ledger)
        ledger["missing_unit_costs"] = ";".join(missing + pmissing)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    if missing:
        print("\nPLACEHOLDER unit costs still unset (lines priced as 0):", ", ".join(sorted(set(missing))))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
