#!/usr/bin/env python
"""
Generate the Phase 7 intervention scenarios from the baseline and a spec.

WHY GENERATED. Every value the five templates under scenarios/*.template ask
Jun to "FILL IN" is either a literature constant already recorded in
docs/calibration_log/phase_7_intervention_evidence_2026-08-17.md, or an
arithmetic consequence of one fitted number, `moud.reinit_prob`, which does not
exist until the refit lands. Hand-filling them means copying a 105 KB baseline
five times and doing the division by hand after every refit; generating them
means the arms are the baseline plus a stated delta, the arithmetic is printed
where the methods section can quote it, and `--check` catches drift, the same
contract as scripts/make_scenario.py and scripts/make_phase7_baseline.py.

WHAT NEEDS NO ONE'S DATA PULL. Every constant in ARMS below has a source in the
evidence note; none depends on the APCD. The only refit-dependent inputs are
`moud.reinit_prob` and `moud.discontinue_per_month`, read from the baseline.

THE DESIGN CHOICES, which are Jun's and are recorded here rather than in five
YAML comments:

  * Scenario 2 runs ED-initiated buprenorphine (D'Onofrio 2015, 78% engaged at
    30 days vs 37% referral alone) as the primary arm and a universal-linkage
    ceiling as a labelled upper bound. Peer navigation (AnchorED, ~21%) is NOT
    run as an arm: the controlled evidence finds little or no linkage effect
    (Beaudoin 2022 RCT in RI EDs, 32% vs 30%; NJ Medicaid cohort 2024, +3.4
    points), and with `moud.post_overdose_rr` fixed at 1.0 the baseline's own
    re-engagement rate for previously treated survivors is the fitted
    reinit_prob per month (~0.36 at the 2026-09-29 warm starts), already above
    RI's observed 13-20% within 30 days, so a 0.21 target would sit below the
    baseline.
  * Scenario 3 is framed as a treatment-INITIATION intervention (x2.11 on the
    per-step initiation rate, D'Onofrio 78/37; the size of Vermont's
    hub-and-spoke doubling 2014-2017), not as the X-waiver, whose measured
    effect on initiation was null (Chua, NEJM 2024). The coverage consequence
    is printed, because doubling a rate does not double coverage.
  * Scenario 5 is an opt-out co-located testing programme, anchored on the
    ever-OAT pooled OR of 3.82 (Ferraro 2021) for the stable multiplier, with
    the early:stable ratio preserved.
  * Scenario 6 holds the post-overdose LINKAGE PROBABILITY at scenario 2's
    target against scenario 3's raised reinit_prob (the "common linkage"
    design), so its sub-additivity is epidemiological rather than the moud.py
    clamp. The "each at its own setting" alternative is one flag away.
  * Scenario 4 saturates REACH only (slots >> N); the within-SSP sharing risk
    stays at its NHBS anchor.

USAGE
    python scripts/make_phase7_scenarios.py                 # write scenarios 2-6
    python scripts/make_phase7_scenarios.py --check         # fail if any is stale
    python scripts/make_phase7_scenarios.py --own-settings  # scenario 6 variant

Reads scenarios/scenario_1_baseline.yml (regenerate it first with
make_phase7_baseline.py after every refit) and writes
scenarios/scenario_{2..6}_*.yml. A YAML build; fine on a login node.
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from titan.calibration.optuna_calibration import load_yaml, set_yaml_value  # noqa: E402

BASELINE = REPO / "scenarios" / "scenario_1_baseline.yml"
OUT_DIR = REPO / "scenarios"

# The declared maximum of moud.post_overdose_rr (titan/params/moud.yml).
POST_OVERDOSE_RR_MAX = 20.0

# ---------------------------------------------------------------------------
# The evidence, as constants. Sources in phase_7_intervention_evidence_2026-08-17.md.
# ---------------------------------------------------------------------------
ED_BUP_LINKAGE_30D = 0.78      # D'Onofrio et al., JAMA 2015: ED-initiated buprenorphine, 78% engaged at 30 d
INITIATION_MULTIPLIER = 2.11   # D'Onofrio 78% / 37% (referral alone = usual care; brief intervention was 45%)
TESTING_STABLE_MULT = 3.8      # Ferraro et al., Addiction 2021: ever-OAT pooled OR 3.82 (2.96-4.95); Metsch 2012
                               #   RCT: on-site rapid testing 81-86% tested vs 24% by referral (RR 4.52 for results)
TESTING_EARLY_MULT = 6.0       # keeps the baseline's early:stable ratio (3.0 : 1.5)
SSP_SLOTS = 100000             # >> any N: unconditional access
# Full rationale, alternatives and what would change each:
# docs/calibration_log/phase_7_arm_design_rationale_2026-09-29.md


def fitted(baseline, path):
    node = baseline
    for part in path.split("."):
        node = node[part]
    return float(node)


def build_arms(baseline, own_settings=False):
    """Return {filename: (scenario dict, worked arithmetic lines)}.

    Pure: takes the baseline dict, returns new dicts. The arithmetic lines are
    what the methods section quotes, and what --check prints on drift.
    """
    reinit = fitted(baseline, "moud.reinit_prob")
    discontinue = fitted(baseline, "moud.discontinue_per_month")
    early = fitted(baseline, "hiv_testing.moud_early_mult")
    stable = fitted(baseline, "hiv_testing.moud_stable_mult")
    slots = fitted(baseline, "syringe_services.timeline.ssp_open.num_slots_start")

    def equilibrium(r, d):
        return r / (r + d)

    arms = {}

    # ---- Scenario 2: ED-initiated buprenorphine after a nonfatal overdose ----
    rr_ed = min(ED_BUP_LINKAGE_30D / reinit, POST_OVERDOSE_RR_MAX)
    notes2 = [
        f"baseline: post_overdose_rr 1.0, reinit_prob {reinit:.5f}/step -> a previously treated",
        f"  survivor restarts within one step with probability {reinit:.3f} (RI observed 0.13-0.20 at 30 days)",
        f"arm: target linkage {ED_BUP_LINKAGE_30D} (D'Onofrio 2015) -> post_overdose_rr = "
        f"{ED_BUP_LINKAGE_30D} / {reinit:.5f} = {ED_BUP_LINKAGE_30D / reinit:.3f}"
        + (f", capped at {POST_OVERDOSE_RR_MAX}" if ED_BUP_LINKAGE_30D / reinit > POST_OVERDOSE_RR_MAX else ""),
        f"  linkage probability in the arm: min(1, {reinit:.5f} x {rr_ed:.3f}) = {min(1.0, reinit * rr_ed):.3f}",
    ]
    if rr_ed < 1.0:
        notes2.append("  WARNING: rr < 1 -- the fitted reinit_prob already exceeds the target; the arm is the baseline")
        rr_ed = 1.0
    s2 = copy.deepcopy(baseline)
    set_yaml_value(s2, "moud.post_overdose_rr", rr_ed)
    arms["scenario_2_postod_linkage.yml"] = (s2, notes2)

    # ---- Scenario 2b: universal linkage, a labelled ceiling ----
    rr_ceiling = min(1.0 / reinit, POST_OVERDOSE_RR_MAX)
    notes2b = [
        f"ceiling: post_overdose_rr = 1 / {reinit:.5f} = {1.0 / reinit:.3f}"
        + (f", capped at {POST_OVERDOSE_RR_MAX}" if 1.0 / reinit > POST_OVERDOSE_RR_MAX else ""),
        f"  linkage probability {min(1.0, reinit * rr_ceiling):.3f} -- report as an upper bound, not a policy",
    ]
    s2b = copy.deepcopy(baseline)
    set_yaml_value(s2b, "moud.post_overdose_rr", rr_ceiling)
    arms["scenario_2b_universal_linkage.yml"] = (s2b, notes2b)

    # ---- Scenario 3: treatment initiation x2.11 ----
    reinit3 = min(INITIATION_MULTIPLIER * reinit, 1.0)
    notes3 = [
        f"reinit_prob {reinit:.5f} x {INITIATION_MULTIPLIER} = {INITIATION_MULTIPLIER * reinit:.5f}"
        + (" (clamped to 1.0)" if INITIATION_MULTIPLIER * reinit > 1.0 else ""),
        f"equilibrium coverage among ever-treated: {equilibrium(reinit, discontinue):.3f} -> "
        f"{equilibrium(reinit3, discontinue):.3f} "
        f"(+{100 * (equilibrium(reinit3, discontinue) - equilibrium(reinit, discontinue)):.1f} points), "
        f"at discontinue_per_month {discontinue:.4f}",
    ]
    s3 = copy.deepcopy(baseline)
    set_yaml_value(s3, "moud.reinit_prob", reinit3)
    arms["scenario_3_bup_expansion.yml"] = (s3, notes3)

    # ---- Scenario 4: SSP saturation (reach only) ----
    notes4 = [
        f"num_slots {slots:g} -> {SSP_SLOTS} (unconditional access); risk unchanged at its NHBS anchor",
        "check the baseline's PWID count against the old slot count before quoting the reach gained",
    ]
    s4 = copy.deepcopy(baseline)
    set_yaml_value(s4, "syringe_services.timeline.ssp_open.num_slots_start", SSP_SLOTS, is_int=True)
    set_yaml_value(s4, "syringe_services.timeline.ssp_open.num_slots_stop", SSP_SLOTS, is_int=True)
    arms["scenario_4_ssp_saturation.yml"] = (s4, notes4)

    # ---- Scenario 5: co-located opt-out HIV testing on MOUD ----
    notes5 = [
        f"moud_stable_mult {stable:g} -> {TESTING_STABLE_MULT} (Ferraro 2021 ever-OAT OR 3.82, CI 2.96-4.95)",
        f"moud_early_mult {early:g} -> {TESTING_EARLY_MULT} (early:stable ratio {early / stable:.2f} -> "
        f"{TESTING_EARLY_MULT / TESTING_STABLE_MULT:.2f})",
        "OR read as a multiplier on a small per-step probability; state the approximation",
    ]
    s5 = copy.deepcopy(baseline)
    set_yaml_value(s5, "hiv_testing.moud_early_mult", TESTING_EARLY_MULT)
    set_yaml_value(s5, "hiv_testing.moud_stable_mult", TESTING_STABLE_MULT)
    arms["scenario_5_moud_hiv_testing.yml"] = (s5, notes5)

    # ---- Scenario 6: 2 + 3 + 5 ----
    if own_settings:
        rr6 = rr_ed
        design = "each policy at its own setting"
    else:
        rr6 = min(ED_BUP_LINKAGE_30D / reinit3, POST_OVERDOSE_RR_MAX)
        design = "common linkage: post_overdose_rr recomputed against scenario 3's reinit_prob"
    notes6 = [
        f"design: {design}",
        f"reinit_prob {reinit3:.5f} (scenario 3), post_overdose_rr {rr6:.3f} -> post-overdose linkage "
        f"min(1, {reinit3:.5f} x {rr6:.3f}) = {min(1.0, reinit3 * rr6):.3f}"
        + ("  (CLAMPED -- name this in the methods)" if reinit3 * rr6 > 1.0 else ""),
        f"hiv_testing multipliers {TESTING_EARLY_MULT} / {TESTING_STABLE_MULT} (scenario 5)",
    ]
    s6 = copy.deepcopy(baseline)
    set_yaml_value(s6, "moud.reinit_prob", reinit3)
    set_yaml_value(s6, "moud.post_overdose_rr", rr6)
    set_yaml_value(s6, "hiv_testing.moud_early_mult", TESTING_EARLY_MULT)
    set_yaml_value(s6, "hiv_testing.moud_stable_mult", TESTING_STABLE_MULT)
    arms["scenario_6_combination.yml"] = (s6, notes6)

    return arms


def write_scenario(path, scenario, notes, baseline_path):
    header = [
        "# GENERATED by scripts/make_phase7_scenarios.py -- do not edit by hand.",
        f"# Baseline: {baseline_path.name}. Regenerate after every refit; --check guards drift.",
        "# Worked arithmetic:",
    ] + [f"#   {line}" for line in notes] + [""]
    with open(path, "w") as fh:
        fh.write("\n".join(header))
        yaml.safe_dump(scenario, fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--own-settings", action="store_true",
                    help="scenario 6 keeps scenario 2's multiplier instead of holding the linkage probability")
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on drift")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    baseline = load_yaml(baseline_path)
    out_dir = Path(args.out_dir)
    arms = build_arms(baseline, own_settings=args.own_settings)

    stale = []
    for name, (scenario, notes) in arms.items():
        path = out_dir / name
        print(f"{name}")
        for line in notes:
            print(f"    {line}")
        if args.check:
            if not path.exists() or load_yaml(path) != scenario:
                stale.append(name)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        write_scenario(path, scenario, notes, baseline_path)
        print(f"    wrote {path}")

    if args.check:
        if stale:
            print(f"\nSTALE: {', '.join(stale)} -- regenerate with scripts/make_phase7_scenarios.py")
            sys.exit(1)
        print("\nOK: every scenario matches the baseline plus its spec")


if __name__ == "__main__":
    main()
