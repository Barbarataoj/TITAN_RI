#!/usr/bin/env python
"""
Phase 7 cash-flow layer: price the activity a Phase 7 arm generates.

Reads the pooled contrast table written by scripts/phase7_contrast.py
(--out <csv>; one row per arm x outcome with baseline_mean, arm_mean, diff_mean,
diff_se, diff_mean_statewide, horizon_years) and the SERVICE INVENTORY in
docs/phase7_unit_costs.yml (every billable service an arm induces: the model
outcome that induces it, the probability per event, codes, and the unit
revenue to the clinic and cost to the payer by payer type), and writes three
ledgers per arm, statewide and per year: the clinic (revenue minus programme cost), the payer (what the activity
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
PERSPECTIVES = ("clinic", "payer", "societal")


def _value(entry):
    """A unit-cost entry's value, or None if it is still a placeholder."""
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def weighted_rate(block, payer_mix):
    """Payer-mix-weighted unit value from a {medicaid, commercial, medicare} block; None if any is unset."""
    if not isinstance(block, dict):
        return None
    total = 0.0
    for payer, share in payer_mix.items():
        v = block.get(payer)
        if v is None:
            return None
        total += share * float(v)
    return total


def product_rate(block, mix, payer_mix):
    """Product-mix-weighted unit value; each product is a number or a payer block."""
    total = 0.0
    for product, share in (mix or {}).items():
        v = (block or {}).get(product)
        v = weighted_rate(v, payer_mix) if isinstance(v, dict) else v
        if v is None:
            return None
        total += share * float(v)
    return total


def service_table(costs):
    """
    Per service: the model outcome that induces it, the probability per event, the
    optional non-billable adjustment, and the clinic revenue / payer cost per unit
    (payer-mix weighted; product-mix weighted for drugs; 340B gross spreads scaled by
    the net share kept). Records placeholders.
    """
    mix = costs["payer_mix"]
    net = costs.get("net_share_340b", {})
    out, missing = {}, []
    for name, s in (costs.get("services") or {}).items():
        prob = s.get("probability", 1.0)
        if prob is None:
            missing.append(f"services.{name}.probability (decision)"); prob = 0.0
        share = float(net.get(s["net_share"], 1.0)) if s.get("net_share") else 1.0
        if "clinic_revenue_gross" in s:
            block, scale = s["clinic_revenue_gross"], share
        else:
            block, scale = s.get("clinic_revenue"), 1.0
        if "product_mix" in s:
            rev = product_rate(block, s["product_mix"], mix)
            pay = product_rate(s.get("payer_cost"), s["product_mix"], mix)
        else:
            rev = weighted_rate(block, mix)
            pay = weighted_rate(s.get("payer_cost"), mix)
        if rev is None:
            missing.append(f"services.{name}.clinic_revenue")
        else:
            rev *= scale
        if pay is None:
            missing.append(f"services.{name}.payer_cost")
        out[name] = {"induced_by": s["induced_by"], "probability": float(prob),
                     "nonbillable_from": s.get("nonbillable_from"),
                     "nonbillable_share": float(s.get("nonbillable_share", 0.0)),
                     "clinic_revenue": rev, "payer_cost": pay}
    return out, missing


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
    missing = []
    rows = []
    for arm in sorted(set(table["arm"])):
        if args.arm and arm != args.arm:
            continue
        t = table[table["arm"] == arm].set_index("outcome")
        years = float(t["horizon_years"].iloc[0])

        def diff_statewide(oc):
            return float(t.loc[oc, "diff_mean_statewide"]) if oc in t.index else 0.0

        services, smissing = service_table(costs)
        counts = {}
        for name, s in services.items():
            n = diff_statewide(s["induced_by"])
            if s["nonbillable_from"]:
                n -= s["nonbillable_share"] * diff_statewide(s["nonbillable_from"])
            counts[name] = n / years * s["probability"]                              # per year, statewide
        reach_visits = diff_statewide("reach_visits") / years
        reached_pm = diff_statewide("reached_person_months") / years
        infections = -diff_statewide("hiv_new_pwid") / years                     # averted per year
        ledger = {"arm": arm, "horizon_years": years, "infections_averted_per_year": infections,
                  **{f"n_{k}": v for k, v in counts.items()},
                  "reach_visits_per_year": reach_visits, "reached_person_months_per_year": reached_pm}
        # clinic
        rev = sum(services[k]["clinic_revenue"] * counts[k] for k in services if services[k]["clinic_revenue"] is not None)
        prog, pmissing = program_cost_per_year(costs, reach_visits, reached_pm, reach_visits)
        ledger["clinic_revenue_per_year"] = rev
        ledger["program_cost_per_year"] = (prog * args.sites) if prog is not None else None
        ledger["clinic_net_per_year"] = rev - prog * args.sites if prog is not None else None
        # payer
        pay = sum(services[k]["payer_cost"] * counts[k] for k in services if services[k]["payer_cost"] is not None)
        hv = costs["health_values"].get("lifetime_hiv_treatment_cost") or {}
        hiv_cost = _value(hv)
        if hiv_cost is not None:
            cpi = costs.get("cpi_medical", {}) or {}
            base_year, target = hv.get("year"), costs.get("analysis_year")
            if base_year in cpi and target in cpi:
                hiv_cost = hiv_cost * float(cpi[target]) / float(cpi[base_year])   # inflate to the analysis year
        ledger["payer_activity_cost_per_year"] = pay
        ledger["hiv_treatment_cost_averted_per_year"] = infections * hiv_cost if hiv_cost is not None else None
        ledger["payer_net_per_year"] = (infections * hiv_cost - pay) if hiv_cost is not None else None
        # societal
        qaly = _value(costs["health_values"].get("qaly_per_infection_averted"))
        ledger["qalys_gained_per_year"] = infections * qaly if qaly is not None else None
        ledger["cost_per_infection_averted_clinic_and_payer"] = (
            ((prog or 0.0) * args.sites + pay) / infections if infections > 0 else None)
        # break-even reach: activity (and so revenue and variable cost) scales with
        # reached person-months; the platform cost is fixed per site
        if prog is not None and reached_pm > 0:
            fixed = _value(costs["program_costs"]["chia_platform_per_site_year"]) * args.sites
            variable = prog * args.sites - fixed
            net_per_reached_pm = (rev - variable) / reached_pm
            ledger["breakeven_reached_person_months_per_year"] = (
                fixed / net_per_reached_pm if net_per_reached_pm > 0 else None)
            ledger["net_revenue_per_reached_person_month"] = net_per_reached_pm
        rows.append(ledger)
        ledger["missing_unit_costs"] = ";".join(sorted(set(smissing + pmissing)))
        missing = sorted(set(missing + smissing + pmissing))
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
