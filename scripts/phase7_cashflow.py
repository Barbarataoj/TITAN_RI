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


def navigation_cost_per_year(costs, reach_visits, reached_pm):
    """Peer / patient navigation (2026-10-13): effort-based (navigator time per active
    programme visit and per enrolled month) and caseload-based (enrolled person-years over
    an active caseload per FTE), both with supervision overhead. Returns (effort cost,
    caseload cost, effort FTE, caseload FTE, missing)."""
    pc = costs["program_costs"]
    keys = ("navigator_wage_per_hour", "navigator_productive_hours_per_fte", "navigator_supervision_overhead",
            "navigator_minutes_per_active_visit", "navigator_minutes_per_enrolled_month", "navigator_active_caseload_per_fte")
    vals = {k: _value(pc.get(k)) for k in keys}
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        return None, None, None, None, missing
    hours = reach_visits * vals["navigator_minutes_per_active_visit"] / 60.0 + reached_pm * vals["navigator_minutes_per_enrolled_month"] / 60.0
    fte_effort = hours / vals["navigator_productive_hours_per_fte"]
    fte_caseload = (reached_pm / 12.0) / vals["navigator_active_caseload_per_fte"]
    per_fte = vals["navigator_wage_per_hour"] * 2080.0 * (1.0 + vals["navigator_supervision_overhead"])
    return fte_effort * per_fte, fte_caseload * per_fte, fte_effort, fte_caseload, []


def program_cost_per_year(costs, reach_visits, reached_pm, invitations, price_per_chat=None):
    """CHIA: returns (fixed platform cost PER SITE-YEAR, variable cost STATEWIDE, missing).
    The variable part is the usage charge (a flat price per completed chat, one chat per
    active CHIA visit; 2026-10-13: no set-up or licence fee, the clinic pays by use) plus the
    clinic's own staff time, all derived from statewide reach visits and invitations, so only
    the fixed part is multiplied by the number of sites (``--sites``). ``price_per_chat``
    overrides the cost file's ``chia_price_per_chat`` (low / central / high scenarios)."""
    pc = costs["program_costs"]
    fixed = _value(pc.get("chia_platform_per_site_year"))
    chat = price_per_chat if price_per_chat is not None else _value(pc.get("chia_price_per_chat"))
    ma = _value(pc.get("ma_wage_per_hour")); prov = _value(pc.get("provider_wage_per_hour"))
    missing = [k for k, v in (("chia_platform_per_site_year", fixed), ("chia_price_per_chat", chat),
                              ("ma_wage_per_hour", ma), ("provider_wage_per_hour", prov)) if v is None]
    if missing:
        return None, None, missing
    variable = (reach_visits * chat
                + invitations * _value(pc["ma_minutes_per_invitation"]) / 60.0 * ma
                + reach_visits * _value(pc["provider_minutes_per_reach_visit"]) / 60.0 * prov)
    return fixed, variable, []


def chain_clinical_time_cost(costs, extra_tests, extra_discussions, extra_dx):
    """Clinician / MA time on the chain an arm sets off (2026-10-14): result delivery on extra
    tests, the PrEP conversation on extra test-and-offer discussions, post-test counselling and
    linkage on extra new diagnoses. Returns (cost per year, missing)."""
    pc = costs["program_costs"]
    ma = _value(pc.get("ma_wage_per_hour")); prov = _value(pc.get("provider_wage_per_hour"))
    neg = _value(pc.get("ma_minutes_per_negative_result")); disc = _value(pc.get("prep_discussion_minutes_clinician"))
    link = _value(pc.get("linkage_minutes_per_new_diagnosis"))
    missing = [k for k, v in (("ma_wage_per_hour", ma), ("provider_wage_per_hour", prov),
                              ("ma_minutes_per_negative_result", neg), ("prep_discussion_minutes_clinician", disc),
                              ("linkage_minutes_per_new_diagnosis", link)) if v is None]
    if missing:
        return None, missing
    return (max(extra_tests, 0.0) * neg / 60.0 * ma + max(extra_discussions, 0.0) * disc / 60.0 * prov
            + max(extra_dx, 0.0) * link / 60.0 * prov), []


def injectable_admin_cost(costs, extra_injections):
    """Clinic administrative time per extra long-acting injection (prior authorisation,
    benefits, scheduling; Fenway 2023); returns (cost per year, missing)."""
    pc = costs["program_costs"]
    mins = _value(pc.get("injectable_admin_minutes_per_injection")); prov = _value(pc.get("provider_wage_per_hour"))
    missing = [k for k, v in (("injectable_admin_minutes_per_injection", mins), ("provider_wage_per_hour", prov)) if v is None]
    if missing:
        return None, missing
    return max(extra_injections, 0.0) * mins / 60.0 * prov, []


def ehr_program_cost_per_year(costs, extra_offers, years, sites=1):
    """pc2: (fixed per year = build amortised over the horizon x EHR instances + maintenance,
    variable = staff time on the extra opt-out offers, missing)."""
    pc = costs["program_costs"]
    build = _value(pc.get("ehr_build_one_time")); maint = _value(pc.get("ehr_maintenance_per_year"))
    ma_min = _value(pc.get("ehr_ma_minutes_per_offer")); prov_min = _value(pc.get("ehr_provider_minutes_per_offer"))
    ma = _value(pc.get("ma_wage_per_hour")); prov = _value(pc.get("provider_wage_per_hour"))
    missing = [k for k, v in (("ehr_build_one_time", build), ("ehr_maintenance_per_year", maint),
                              ("ehr_ma_minutes_per_offer", ma_min), ("ehr_provider_minutes_per_offer", prov_min),
                              ("ma_wage_per_hour", ma), ("provider_wage_per_hour", prov)) if v is None]
    if missing:
        return None, None, missing
    fixed = (build / years + maint) * sites
    variable = extra_offers * (ma_min / 60.0 * ma + prov_min / 60.0 * prov)
    return fixed, variable, []


def training_program_cost_per_year(costs, years):
    """pc1: detailing every year plus one-time training amortised over the horizon, per clinician,
    times the statewide clinician count; returns (cost per year, clinicians, missing)."""
    pc = costs["program_costs"]
    det = _value(pc.get("detailing_cost_per_clinician_year")); tr = _value(pc.get("training_cost_per_clinician"))
    n = _value(pc.get("clinicians_statewide"))
    missing = [k for k, v in (("detailing_cost_per_clinician_year", det), ("training_cost_per_clinician", tr),
                              ("clinicians_statewide", n)) if v is None]
    if missing:
        return None, None, missing
    return n * (det + tr / years), n, []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrasts", required=True, type=Path)
    ap.add_argument("--costs", type=Path, default=REPO / "docs" / "phase7_unit_costs.yml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", default=None, help="price one arm only")
    ap.add_argument("--sites", type=int, default=1, help="number of clinics the platform cost is counted for")
    ap.add_argument("--chia-price-per-chat", type=float, default=None,
                    help="override the cost file's chia_price_per_chat (USD per completed chat)")
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

        # Payer-billed activity is the INSURED agents' activity when the contrast carries it (insurance
        # as agent state, 2026-10-24, work record §19.34): phase7_contrast writes `<outcome>_insured`
        # for every priced activity from the insured rows of a stratified round. Programme time
        # (chain_clinical_time_cost, the navigator, the chatbot) stays on ALL agents: staff spend it
        # on uninsured patients too. An outcome without an insured column falls back to all agents and
        # is listed in `billed_all_agents` so the ledger says so.
        billed_insured, billed_all = [], []

        def billed(oc):
            if f"{oc}_insured" in t.index:
                billed_insured.append(oc)
                return diff_statewide(f"{oc}_insured")
            billed_all.append(oc)
            return diff_statewide(oc)

        services, smissing = service_table(costs)
        counts = {}
        for name, s in services.items():
            n = billed(s["induced_by"])
            if s["nonbillable_from"]:
                n -= s["nonbillable_share"] * billed(s["nonbillable_from"])
            counts[name] = n / years * s["probability"]                              # per year, statewide
        reach_visits = diff_statewide("reach_visits") / years
        reached_pm = diff_statewide("reached_person_months") / years
        # Infections averted per year, from the expected-infection (hazard)
        # estimator when the ensemble carries it (2026-10-09): same expectation
        # as the counts, without the coin-flip noise that made the counts
        # unusable at this incidence (SE 1-3 % of the status quo vs 0.3-0.8 %).
        estimator = "hiv_hazard_pwid" if "hiv_hazard_pwid" in t.index else "hiv_new_pwid"
        infections = -diff_statewide(estimator) / years
        ledger = {"arm": arm, "horizon_years": years, "infections_averted_per_year": infections,
                  "infections_estimator": estimator,
                  **{f"n_{k}": v for k, v in counts.items()},
                  "reach_visits_per_year": reach_visits, "reached_person_months_per_year": reached_pm}
        # clinic
        rev = sum(services[k]["clinic_revenue"] * counts[k] for k in services if services[k]["clinic_revenue"] is not None)
        # programme cost is STATEWIDE per year; ``--sites`` scales only the fixed per-site
        # platform cost of CHIA (navigation and the variable staff costs are already statewide)
        fixed_statewide = 0.0
        # 2026-10-15 (Jun: the same test for every arm): the clinician-delivered steps of the
        # chain every arm sets off are priced on the extra events in EVERY arm — result delivery,
        # the PrEP conversation, post-test counselling and linkage — plus the administrative time
        # of extra long-acting injections. For navigation and CHIA this is conservative: part of
        # the conversation is the navigator's / the chatbot's, already costed, and the clinician
        # still has to prescribe.
        extra_tests = diff_statewide("hiv_tests") / years        # programme time: every patient, insured or not
        extra_disc = diff_statewide("prep_bundle_offers") / years
        # linkage time is priced on every extra diagnosis in the cohort; contrast files written before
        # 2026-10-21 carry only the injecting subset, which is then used and flagged in `linkage_basis`
        dx_basis = "hiv_dx_new" if "hiv_dx_new" in t.index else "hiv_dx_new_pwid"
        extra_dx = diff_statewide(dx_basis) / years
        ledger["linkage_basis"] = dx_basis
        extra_inj = diff_statewide("prep_injections") / years
        chain_cost, cmissing = chain_clinical_time_cost(costs, extra_tests, extra_disc, extra_dx)
        inj_cost, imissing = injectable_admin_cost(costs, extra_inj)
        ledger["extra_tests_per_year"] = extra_tests
        ledger["billed_insured_only"] = ";".join(sorted(set(billed_insured)))
        ledger["billed_all_agents"] = ";".join(sorted(set(billed_all)))
        ledger["extra_prep_discussions_per_year"] = extra_disc
        ledger["extra_new_diagnoses_per_year"] = extra_dx
        ledger["extra_injections_per_year"] = extra_inj
        ledger["chain_clinical_time_cost_per_year"] = chain_cost
        ledger["injectable_admin_cost_per_year"] = inj_cost
        common = (chain_cost or 0.0) + (inj_cost or 0.0)
        # programme cost by strategy; a combined arm (pc24: EHR + CHIA) carries every component its
        # name matches (2026-10-20 code review: it had been priced as CHIA alone), so the branches
        # accumulate into `parts` rather than choosing one
        parts, pmissing = [], []
        if "navigation" in arm:
            prog_n, prog_caseload, fte_e, fte_c, pm_n = navigation_cost_per_year(costs, reach_visits, reached_pm)
            ledger["navigator_fte_effort"] = fte_e; ledger["navigator_fte_caseload"] = fte_c
            # the caseload (upper-bound) costing carries the same chain and injection time as the effort costing
            ledger["program_cost_caseload_per_year"] = prog_caseload + common if prog_caseload is not None else None
            parts.append(prog_n); pmissing += pm_n
        if "chia" in arm:
            fixed_site, variable, pmissing_c = program_cost_per_year(costs, reach_visits, reached_pm, reach_visits,
                                                                      price_per_chat=args.chia_price_per_chat)
            chat_price = (args.chia_price_per_chat if args.chia_price_per_chat is not None
                          else _value(costs["program_costs"].get("chia_price_per_chat")))
            ledger["chia_chats_per_year"] = reach_visits
            ledger["chia_price_per_chat"] = chat_price
            ledger["chia_usage_cost_per_year"] = reach_visits * chat_price if chat_price is not None else None
            if fixed_site is None:
                parts.append(None)
            else:
                fixed_statewide += fixed_site * args.sites
                parts.append(fixed_site * args.sites + variable)
            pmissing += pmissing_c
        if "ehr" in arm:
            extra_offers = diff_statewide("pc_test_offers") / years
            fixed_e, variable_e, pm_e = ehr_program_cost_per_year(costs, extra_offers, years, args.sites)
            ledger["ehr_extra_offers_per_year"] = extra_offers
            ledger["ehr_fixed_cost_per_year"] = fixed_e
            parts.append(None if fixed_e is None else fixed_e + variable_e)
            fixed_statewide += fixed_e or 0.0
            pmissing += pm_e
        if "training" in arm:
            prog_t, n_clin, pm_t = training_program_cost_per_year(costs, years)
            ledger["clinicians_statewide"] = n_clin
            parts.append(prog_t); pmissing += pm_t
        # injectable-first (no branch): a prescribing rule; only the chain and injection time below
        prog = None if any(x is None for x in parts) else sum(parts) + common
        pmissing = pmissing + cmissing + imissing
        ledger["clinic_revenue_per_year"] = rev
        ledger["program_cost_per_year"] = prog
        ledger["clinic_net_per_year"] = rev - prog if prog is not None else None
        # payer
        pay = sum(services[k]["payer_cost"] * counts[k] for k in services if services[k]["payer_cost"] is not None)
        hv = costs["health_values"].get("lifetime_hiv_treatment_cost") or {}
        hiv_cost = _value(hv)
        if hiv_cost is not None:
            cpi = costs.get("cpi_medical", {}) or {}
            base_year, target = hv.get("year"), costs.get("analysis_year")
            if base_year in cpi and target in cpi:
                hiv_cost = hiv_cost * float(cpi[target]) / float(cpi[base_year])   # inflate to the analysis year
        # present value over the horizon at the cost file's discount rate (flat per-year flows, so the
        # cost per infection averted is unchanged by discounting; the totals are what a budget sees)
        r = float(costs.get("discount_rate", 0.0) or 0.0)
        pv = sum((1.0 + r) ** -y for y in range(int(round(years))))
        ledger["pv_factor"] = pv
        ledger["program_cost_pv"] = prog * pv if prog is not None else None
        ledger["infections_averted_pv"] = infections * pv
        ledger["payer_activity_cost_per_year"] = pay
        ledger["hiv_treatment_cost_averted_per_year"] = infections * hiv_cost if hiv_cost is not None else None
        ledger["payer_net_per_year"] = (infections * hiv_cost - pay) if hiv_cost is not None else None
        ledger["payer_net_pv"] = ledger["payer_net_per_year"] * pv if ledger["payer_net_per_year"] is not None else None
        # societal
        qaly = _value(costs["health_values"].get("qaly_per_infection_averted"))
        ledger["qalys_gained_per_year"] = infections * qaly if qaly is not None else None
        # gross: programme cost plus payer activity cost per PWID infection averted; the
        # HIV-treatment offset and clinic revenue are reported separately, not netted here
        ledger["cost_per_infection_averted_clinic_and_payer"] = (
            ((prog or 0.0) + pay) / infections if infections > 0 else None)
        # break-even reach: activity (and so revenue and variable cost) scales with
        # reached person-months; the platform cost is fixed per site
        if prog is not None and reached_pm > 0 and "chia" in arm:
            fixed = fixed_statewide
            variable = prog - fixed
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
