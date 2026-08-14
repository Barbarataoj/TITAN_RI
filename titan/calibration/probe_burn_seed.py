"""
Resolve the seed/burn mismatch left open by step 5 of the 2026-08-12 plan.

THE PROBLEM. `INJ_HIV_INIT` (0.045) and `NONINJ_HIV_INIT_HET` (0.020) were
measured on 2026-08-07 as "where the model's own long burn converges" -- but at
trial 290's transmission level. Steps 3 and 4 then moved that level: the MSM
per-act probability fell 34% (0.010 -> 0.00745) and the injection per-act
probability was pinned at the published 0.0063, above the entire box every trial
had ever searched. The seeds no longer sit at the equilibrium of the model they
are seeding, and the 10-step-per-year, 120-step burn is not long enough to
absorb the gap. Measured at the corrected burn (job 4946300), PWID prevalence
drifts across the scored window by -19.8%, +36.8%, -11.9%, +4.7% and -7.3%
depending on the arm, against the 0.10 tolerance
`validate_candidates.VALIDATION` now applies to `hiv_prev_drift_pwid`.

Section 7 of the step 4 log set out three ways forward:

  (a) fit the two seeds -- what they were before 2026-08-07, and what upstream
      TITAN's "hiv.init is a data slot" contract argues against;
  (b) iterate -- refit, re-measure the equilibrium, re-seed, refit once more;
  (c) lengthen the burn, rejected in step 1 at a cost of 84% of compute, but now
      affordable to test since a burn-120 trial is 3.4x cheaper than the burn-720
      trials that had actually been running.

This probe measures what is needed to choose between (b) and (c) instead of
spending a 26-hour refit finding out.

WHY A TRANSMISSION-LEVEL AXIS. Drift is not a property of the seed alone. The
seed is one number; the level it must match is set per trial by two fitted
parameters, `calibration.sex.act` (box [0.38765, 2.53519]) and
`calibration.injection.act` (box [0.05, 1.5]). A seed that equilibrates the
middle of the box can still leave the edges drifting, so a single-level
measurement would answer the wrong question. Three levels spanning the boxes are
run, and the decision rule is stated over all three.

ROUND 1 -- where does the model actually equilibrate now?
  3 levels x burn {120, 240, 720}, current seeds.
  The burn-720 rows give the equilibrium prevalence per level, by compartment
  and (for NonInj) by sex type, which is the seed re-derivation option (b) needs.
  The burn-120 and burn-240 rows say how much of the current mismatch each burn
  length absorbs, which is option (c).

ROUND 2 -- does re-seeding fix it at a short burn?
  Seeds are set to round 1's burn-720 equilibrium, per level, and burn {120,
  240} is re-run. If drift comes inside +/-0.10 at burn 120, option (b) works and
  the burn stays cheap. If it only comes inside at 240, that is the price of (c).
  If neither, the seeds have to be fitted after all, i.e. (a).

The MSM:het seed ratio is re-derived rather than assumed. `NONINJ_HIV_INIT_HET`
is applied at 3.4x on MSM/MSMW strata (`NONINJ_HIV_MSM_TO_HET_RATIO`), a value
carried over from the prevalent-pool argument in probe_seed_burn.py. Round 1
reports the ratio the model's own equilibrium produces, so round 2 can seed with
the measured one and the assumption stops being load-bearing.

ALSO RE-MEASURED HERE. `inj_share_active_oud` is reported for every arm. The
0.363 quoted in section 1 of the step 5 log -- the figure that made
`inj_share_err` look already-passing against the retargeted 0.32 -- was taken
before `set_population_window` existed, so it is a burn-720 number. The target's
derivation from RIHRSS does not depend on it, but the claim that the objective
already passes does.

Run on a compute node (both rounds, ~1-2h at the defaults):

    sbatch --cpus-per-task=4 --time=06:00:00 scripts/run_on_node.slurm \\
      python -m titan.calibration.probe_burn_seed --reps 20
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    HIV_DX_INIT,
    INJ_HIV_INIT,
    MSM_SEX_TYPES,
    NONINJ_HIV_INIT_HET,
    NONINJ_HIV_MSM_TO_HET_RATIO,
    STEPS_PER_YEAR,
    build_targets,
    hiv_null_value,
    load_yaml,
    parse_basic_report,
    ridoh_hiv_rmse,
    run_titan,
    set_hiv_initial_conditions,
    set_nested,
    set_noninj_hiv_init,
    set_oud_initial_conditions,
    set_per_act_transmission,
    set_population_window,
    write_yaml,
)

BASE_SCENARIO = Path("/oscar/data/jtao9/TITAN_RI/calibrated_scenario_best.yml")
OUTPUT_DIR = Path("/oscar/scratch/jtao9/titan_calibration/burn_seed_probe")

POP_SEEDS = [12345, 20260720]

# Tolerance validate_candidates.VALIDATION applies to hiv_prev_drift_pwid and
# hiv_prev_drift_noninj. Repeated here so the probe's verdict is the gate's.
DRIFT_TOL = 0.10

# (label, calibration.sex.act, calibration.injection.act)
#
# Chosen to span both fitted boxes rather than to be individually plausible: the
# question is whether one seed can hold the window still across the search
# space, so the edges matter more than the centre. ridoh_hiv_err is reported per
# arm so a level that no refit would ever visit can be discounted by eye.
LEVELS = [
    ("cold", 0.60, 0.15),
    ("mid", 1.20, 0.50),
    ("hot", 2.00, 1.10),
]

R1_BURNS = [120, 240, 720]
R2_BURNS = [120, 240]


def apply_arm(scenario, calib, sex_act, injection_act, burn,
              inj_seed=None, het_seed=None, msm_seed=None):
    """
    Every current anchor, then this arm's level, burn and seeds.

    Order matters. set_hiv_initial_conditions writes the module-constant seeds
    and set_population_window writes the targets file's burn, so both overrides
    have to come after them -- the first run of the step 4 probe reported itself
    as burn 120 while every arm ran the base scenario's 720 for exactly this
    reason.
    """
    set_hiv_initial_conditions(scenario)
    set_oud_initial_conditions(scenario, calib)
    set_per_act_transmission(scenario)
    set_population_window(scenario, calib)

    # set_nested, not set_yaml_value: `calibration.injection` is absent from the
    # base scenario -- calibration.injection.act was only added to the search on
    # 2026-08-13 -- and set_yaml_value refuses to create intermediate keys. This
    # is the same call objective() makes for every `calibration.*` parameter.
    calib_block = scenario.setdefault("calibration", {})
    set_nested(calib_block, "sex.act", float(sex_act))
    set_nested(calib_block, "injection.act", float(injection_act))

    scenario["model"]["time"]["burn_steps"] = int(burn)

    if het_seed is not None:
        msm = msm_seed if msm_seed is not None else het_seed * NONINJ_HIV_MSM_TO_HET_RATIO
        set_noninj_hiv_init(scenario, het=float(het_seed), msm=float(msm),
                            dx_init=HIV_DX_INIT)
    if inj_seed is not None:
        for race in scenario.get("demographics", {}).values():
            for sx in (race or {}).get("sex_type", {}).values():
                inj = (sx or {}).get("drug_type", {}).get("Inj")
                if inj is not None:
                    inj.setdefault("hiv", {})["init"] = float(inj_seed)


def window_prevalence(report, steps_per_year=STEPS_PER_YEAR):
    """
    Prevalence at the open and close of the scored window, by compartment, plus
    the NonInj split by sex type.

    The sex-type split is the point: the seed convention forces MSM = 3.4 x het,
    and nothing has ever checked that against the ratio the model settles at.
    """
    df = pd.read_csv(report, sep="\t")
    scored = df[df["t"] >= 1]
    if scored.empty:
        return None

    first, last = scored["t"].min(), scored["t"].max()
    open_win = scored[scored["t"] < first + steps_per_year]
    close_win = scored[scored["t"] >= last - steps_per_year + 1]

    def prev(sub, drug_type, sex_types=None):
        s = sub[sub["drug_type"] == drug_type]
        if sex_types is not None:
            s = s[s["sex_type"].isin(sex_types)]
        agents = s["agents"].sum()
        return float(s["hiv"].sum() / agents) if agents else float("nan")

    all_sex = sorted(scored["sex_type"].unique())
    msm = [s for s in all_sex if s in MSM_SEX_TYPES]
    het = [s for s in all_sex if s not in MSM_SEX_TYPES]

    out = {
        "pwid_open": prev(open_win, "Inj"),
        "pwid_close": prev(close_win, "Inj"),
        "noninj_open": prev(open_win, "NonInj"),
        "noninj_close": prev(close_win, "NonInj"),
        "noninj_het_open": prev(open_win, "NonInj", het),
        "noninj_het_close": prev(close_win, "NonInj", het),
        "noninj_msm_open": prev(open_win, "NonInj", msm),
        "noninj_msm_close": prev(close_win, "NonInj", msm),
    }
    for comp in ("pwid", "noninj", "noninj_het", "noninj_msm"):
        o, c = out[f"{comp}_open"], out[f"{comp}_close"]
        out[f"{comp}_drift"] = float(c / o - 1.0) if o and np.isfinite(o) else float("nan")
    return out


def summarize(outdir, targets):
    report = outdir / "basicReport.txt"
    if not report.exists():
        return None

    out = window_prevalence(report)
    if out is None:
        return None

    stats = parse_basic_report(report)
    out["ridoh_hiv_err"] = float(
        ridoh_hiv_rmse(
            stats.get("idu_dx_counts_per_year"),
            stats.get("pwid_dx_counts_per_year"),
            targets["first_year"],
            targets,
        )
    )
    out["inj_share_active_oud"] = float(stats.get("inj_share_active_oud", float("nan")))
    return out


def run_arms(arms, calib, targets, base_scenario, reps, pop_seeds, tag_prefix):
    """arms: (label, sex_act, injection_act, burn, inj_seed, het_seed, msm_seed)."""
    rows = []
    for label, sex_act, inj_act, burn, inj_seed, het_seed, msm_seed in arms:
        for pop_seed in pop_seeds:
            scenario = copy.deepcopy(base_scenario)
            apply_arm(scenario, calib, sex_act, inj_act, burn,
                      inj_seed=inj_seed, het_seed=het_seed, msm_seed=msm_seed)
            scenario["model"].setdefault("seed", {})["ppl"] = int(pop_seed)

            tag = f"{tag_prefix}_{label}_pop{pop_seed}"
            yml = OUTPUT_DIR / f"{tag}.yml"
            outdir = OUTPUT_DIR / tag
            outdir.mkdir(parents=True, exist_ok=True)
            write_yaml(yml, scenario)

            print(f"[{tag}] sex.act={sex_act:.2f} inj.act={inj_act:.2f} burn={burn} "
                  f"inj_seed={inj_seed} het_seed={het_seed}", flush=True)
            run_titan(yml, outdir, n_reps=reps)

            summary = summarize(outdir, targets)
            if summary is None:
                print(f"[{tag}] no basicReport", flush=True)
                continue
            summary.update(arm=label, pop_seed=pop_seed, sex_act=sex_act,
                           injection_act=inj_act, burn=burn,
                           inj_seed=inj_seed if inj_seed is not None else INJ_HIV_INIT,
                           het_seed=het_seed if het_seed is not None else NONINJ_HIV_INIT_HET)
            rows.append(summary)
            print(f"[{tag}] pwid_drift={summary['pwid_drift']:+.1%} "
                  f"noninj_drift={summary['noninj_drift']:+.1%} "
                  f"ridoh_hiv_err={summary['ridoh_hiv_err']:.3f} "
                  f"inj_share={summary['inj_share_active_oud']:.3f}", flush=True)
    return pd.DataFrame(rows)


def drift_table(df, title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'arm':22s} {'PWID open':>9s} {'close':>8s} {'drift':>8s} "
          f"{'NInj open':>9s} {'close':>8s} {'drift':>8s} {'hiv_err':>8s} {'inj_shr':>8s}")
    agg = df.groupby("arm", sort=False).mean(numeric_only=True)
    for arm in agg.index:
        r = agg.loc[arm]
        flag = "" if abs(r.pwid_drift) <= DRIFT_TOL and abs(r.noninj_drift) <= DRIFT_TOL else "  FAIL"
        print(f"{arm:22s} {r.pwid_open:9.4f} {r.pwid_close:8.4f} {r.pwid_drift:+7.1%} "
              f"{r.noninj_open:9.4f} {r.noninj_close:8.4f} {r.noninj_drift:+7.1%} "
              f"{r.ridoh_hiv_err:8.3f} {r.inj_share_active_oud:8.3f}{flag}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--pop-seeds", type=int, default=2)
    ap.add_argument("--rounds", type=str, default="1,2")
    args = ap.parse_args()

    rounds = {int(r) for r in args.rounds.split(",") if r.strip()}
    pop_seeds = POP_SEEDS[: args.pop_seeds]

    calib = load_yaml(CALIB_YAML)
    targets = build_targets(calib["calibration"])
    with open(BASE_SCENARIO) as fh:
        base_scenario = yaml.safe_load(fh)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"current seeds: INJ_HIV_INIT={INJ_HIV_INIT} "
          f"NONINJ_HIV_INIT_HET={NONINJ_HIV_INIT_HET} "
          f"(MSM = {NONINJ_HIV_MSM_TO_HET_RATIO}x het)")
    print(f"targets-file burn_steps: {calib['calibration']['population']['burn_steps']}")
    print(f"HIV null (zero-transmission floor): {hiv_null_value(calib):.4f}")
    print(f"drift tolerance: +/-{DRIFT_TOL:.2f}\n", flush=True)

    r1 = pd.DataFrame()
    if 1 in rounds:
        arms = [
            (f"{lvl}_burn{burn}", sex_act, inj_act, burn, None, None, None)
            for lvl, sex_act, inj_act in LEVELS
            for burn in R1_BURNS
        ]
        r1 = run_arms(arms, calib, targets, base_scenario, args.reps, pop_seeds, "r1")
        if r1.empty:
            print("round 1 produced no results")
            return 1
        r1.to_csv(OUTPUT_DIR / "burn_seed_summary_r1.csv", index=False)
        drift_table(r1, "ROUND 1 -- CURRENT SEEDS, drift = close/open - 1")

    if 2 not in rounds:
        return 0

    if r1.empty:
        r1 = pd.read_csv(OUTPUT_DIR / "burn_seed_summary_r1.csv")

    # The equilibrium is read from the burn-720 rows' OPEN of the scored window:
    # after 60 years of burn the seeds are washed out, so what the window opens
    # at is where the model settles at that level.
    eq = r1[r1["burn"] == max(R1_BURNS)].groupby("arm").mean(numeric_only=True)
    print("\n" + "=" * 96)
    print(f"EQUILIBRIUM AT BURN {max(R1_BURNS)} -- the re-derived seeds")
    print("=" * 96)
    print(f"{'level':10s} {'Inj':>8s} {'NonInj het':>11s} {'NonInj MSM':>11s} "
          f"{'MSM:het':>8s}   (assumed ratio {NONINJ_HIV_MSM_TO_HET_RATIO})")

    derived = {}
    for lvl, sex_act, inj_act in LEVELS:
        key = f"{lvl}_burn{max(R1_BURNS)}"
        if key not in eq.index:
            continue
        r = eq.loc[key]
        ratio = (r.noninj_msm_open / r.noninj_het_open) if r.noninj_het_open else float("nan")
        derived[lvl] = (float(r.pwid_open), float(r.noninj_het_open),
                        float(r.noninj_msm_open))
        print(f"{lvl:10s} {r.pwid_open:8.4f} {r.noninj_het_open:11.4f} "
              f"{r.noninj_msm_open:11.4f} {ratio:8.2f}")

    arms = []
    for lvl, sex_act, inj_act in LEVELS:
        if lvl not in derived:
            continue
        inj_seed, het_seed, msm_seed = derived[lvl]
        for burn in R2_BURNS:
            arms.append((f"{lvl}_reseed_burn{burn}", sex_act, inj_act, burn,
                         inj_seed, het_seed, msm_seed))

    if not arms:
        print("\nno round-1 equilibria to re-seed from")
        return 1

    r2 = run_arms(arms, calib, targets, base_scenario, args.reps, pop_seeds, "r2")
    if r2.empty:
        print("round 2 produced no results")
        return 1
    r2.to_csv(OUTPUT_DIR / "burn_seed_summary_r2.csv", index=False)
    agg2 = drift_table(r2, "ROUND 2 -- RE-DERIVED SEEDS, drift = close/open - 1")

    # The decision rule from section 7 of the step 4 log, applied to the numbers
    # rather than argued: the shortest burn that holds every level inside the
    # gate is the one to set.
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    chosen = None
    for burn in R2_BURNS:
        rows = agg2[agg2["burn"] == burn]
        worst_pwid = rows["pwid_drift"].abs().max()
        worst_noninj = rows["noninj_drift"].abs().max()
        ok = worst_pwid <= DRIFT_TOL and worst_noninj <= DRIFT_TOL
        print(f"burn {burn:4d}: worst |PWID drift| {worst_pwid:.1%}, "
              f"worst |NonInj drift| {worst_noninj:.1%}  ->  "
              f"{'INSIDE the gate' if ok else 'outside the gate'}")
        if ok and chosen is None:
            chosen = burn

    if chosen is not None:
        print(f"\nOption (b) works at burn {chosen}: set calibration.population.burn_steps "
              f"= {chosen} and re-derive the seed constants from the table above "
              f"(use the 'mid' level, and record the measured MSM:het ratio).")
    else:
        print("\nNeither burn holds the window inside the gate with a re-derived seed. "
              "That is the case for option (a) -- put the two seeds back in the "
              "search -- or for a burn longer than the ones tested here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
