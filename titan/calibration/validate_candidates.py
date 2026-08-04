"""
Multi-seed re-evaluation of Phase 4 Pareto candidates (no re-optimization).

Phase 4 run 2 (job 4127761) left two members missing exactly one objective by
<10% at 10 reps (trial 54: OUD 1.07x; trial 172: HIV prevalence 1.09x). At that
rep count the miss is not distinguishable from Monte Carlo noise, so selection
cannot proceed on those numbers alone.

This re-runs each candidate's saved trial_<N>.yml under several independent
population seeds, R reps each, and reports mean / spread / per-seed pass rate for
all five objectives. Population seed is varied because the calibration run held
model.seed.ppl fixed at 12345 (only model.seed.run was random per rep), so the
10-rep errors carry no between-population variance at all.

Usage (env-configurable):
    TITAN_VAL_TRIALS=54,172,111 TITAN_VAL_SEEDS=5 TITAN_VAL_REPS=10 \
        python -m titan.calibration.validate_candidates
"""

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    COMPOSITE_CLIP_QUANTILE,
    SQUARED_OBJECTIVES,
    load_yaml,
    multi_error,
    run_titan,
    write_yaml,
)

# Objective order must match multi_error()'s return tuple.
OBJECTIVES = [
    ("OUD distribution", 0.05),
    # Scale changed 2026-08-03 with the objective itself: hiv_traj_rmse is now
    # RMSE over three period means, not eleven annual points. Reference values on
    # the new scale: zero-transmission 0.00684, flat-at-mean predictor 0.00121,
    # the 20 candidates validated at 5 seeds x 10 reps 0.00258-0.00505.
    #
    # History: 0.012 through 2026-08-02 did not bind at all (the annual-scale
    # zero-transmission model scored 0.00745, so the gate accepted a null model
    # outright); 0.005 on the annual scale bound correctly but is meaningless on
    # this one.
    #
    # Set to 0.0012, just under the flat-at-mean predictor's 0.00121, for the
    # reason the 0.012 gate failed: a threshold above that benchmark accepts
    # models that are beaten by a CONSTANT, so it cannot certify that the fit
    # reproduces the trajectory's SHAPE at all. Bisecting the current front (the
    # earlier provisional 0.0035) would have re-inherited exactly that defect --
    # every one of the 20 validated candidates scored 0.00258-0.00505, i.e. all
    # of them lose to a constant, so any gate that passes some of them is
    # certifying nothing.
    #
    # This gate is DELIBERATELY not known to be reachable. Probe 4578413's best
    # single configuration hit 0.00067 and its best per-seed mean was 0.00240,
    # so the model can reach it but not reliably at the current parameterisation.
    # If the refit produces no candidate under 0.0012, that is a real result --
    # the model cannot reproduce the trajectory shape -- and the response is to
    # report it, not to raise the gate until something passes.
    #
    # Reference values on the period scale: zero-transmission 0.00684,
    # flat-at-mean 0.00121, pre-refit candidates 0.00258-0.00505.
    ("HIV trajectory", 0.0012),
    # Squared error on PWID prevalence against a 0.062 target. 0.001 accepts
    # 0.030-0.094 -- a 3x range, wider than the quantity being fitted. 0.0003
    # accepts 0.045-0.079, about 2.5 SE at 50 runs, so it binds without
    # rejecting on noise.
    ("HIV prevalence", 0.0003),
    ("Overdose", 0.005),
    ("MOUD trajectory", 0.10),
    # Squared error on the injection share among active OUD. Threshold 0.0025 =
    # 0.05 absolute, i.e. accept 0.40-0.50 against the 0.45 target. Added
    # 2026-07-29 with the objective itself (EPIDEMIC_DYNAMICS_MAP_2026-07-29.md P2).
    ("Injection share", 0.0025),
]

TRIAL_DIR = Path(
    os.environ.get(
        "TITAN_VAL_TRIAL_DIR",
        "/oscar/scratch/{u}/titan_calibration/optuna_multi_runs".format(
            u=os.environ["USER"]
        ),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "TITAN_VAL_OUT",
        "/oscar/scratch/{u}/titan_calibration/validation".format(u=os.environ["USER"]),
    )
)

# Fixed seed list so a re-run of this script reproduces exactly. 12345 is the
# calibration's own population seed and is kept first as the reference point.
POP_SEEDS = [12345, 20260720, 8675309, 314159, 271828, 1618033, 42424242, 99991]


def load_targets():
    calib = load_yaml(CALIB_YAML)["calibration"]
    return {
        "oud": calib["oud_baseline"],
        "hiv_prev": calib["hiv"]["baseline_prevalence_pct"],
        "hiv_inc_series": calib["hiv"]["incidence_series_per_pwid_yr"],
        "hiv_inc_years": calib["hiv"]["incidence_series_years"],
        "hiv_inc_periods": calib["hiv"]["incidence_periods"],
        "od_fatal": calib["overdose"]["fatal_rate_annual"],
        "od_nonfatal_inj": calib["overdose"]["nonfatal_rate_inj_annual"],
        "od_nonfatal_noninj": calib["overdose"]["nonfatal_rate_noninj_annual"],
        "moud_series": calib["moud"]["uptake_series_per_yr"],
        # Added 2026-08-03. multi_error() grew inj_share_err as a 6th objective on
        # 2026-07-29 and reads targets["inj_share"] unconditionally, but this
        # loader was not updated with it, so every invocation since died with
        # KeyError: 'inj_share' on the first trial. The OBJECTIVES list here was
        # updated at the time, which hid the omission -- the script looked current.
        "inj_share": calib["injecting"]["injecting_pct"],
    }


# Data-anchored inputs, reported but never scored. (stat key, label, target,
# tolerance) -- targets mirror calibration_targets.yml `validation_targets`.
# A miss here does NOT disqualify a trial; it means an input has drifted from the
# RIDOC / CDC-PWID observation it was set from, which is a parameterisation bug
# rather than a fit failure. See EPIDEMIC_DYNAMICS_AUDIT_2026-07-29b.md.
VALIDATION = [
    ("incar_share",         "incarcerated share",   0.0552, 0.015),
    ("art_among_dx",        "on ART | diagnosed",   0.780,  0.06),
    ("suppressed_among_dx", "suppressed | diagnosed", 0.572, 0.08),
]


def _report_validation(sub):
    """Print the unscored validation checks for one trial's seed group."""
    have = [v for v in VALIDATION if v[0] in sub and not sub[v[0]].isna().all()]
    if not have:
        return
    print("  validation (NOT scored -- data-anchored inputs):")
    for key, label, tgt, tol in have:
        mean = float(np.nanmean(sub[key].to_numpy()))
        flag = "ok" if abs(mean - tgt) <= tol else "DRIFT"
        print(f"    {label:<24} {mean:.4f}  target {tgt:.4f} +/-{tol:.3f}  {flag}")
    if "dx_among_hiv" in sub and not sub["dx_among_hiv"].isna().all():
        # No target: the model saturates this against a real-world ~0.85-0.90.
        # Known open defect in hiv_testing (audit finding F7), reported for
        # visibility so it is not mistaken for a fitted quantity.
        print(f"    {'diagnosed | HIV+':<24} "
              f"{float(np.nanmean(sub['dx_among_hiv'].to_numpy())):.4f}  "
              f"(no target; saturates -- see audit F7)")


def _report_noninj_seed(trial):
    """
    Print the trial's fitted NonInj HIV seed, unscored.

    Added 2026-08-03 with the calibration.noninj_hiv_init_* search dimensions.
    These are fitted against wide priors, not set from data, so they need a human
    check against APCD (RI HIV prevalence among non-injecting people with OUD,
    and separately among MSM) before any fit that leans on them is trusted.
    Silent on trial YAMLs written before the parameter existed.
    """
    src = TRIAL_DIR / f"trial_{trial}.yml"
    if not src.exists():
        return
    seeds = {}
    for race in (load_yaml(src).get("demographics") or {}).values():
        for sx_name, sx in (race or {}).get("sex_type", {}).items():
            hiv = ((sx or {}).get("drug_type", {}).get("NonInj") or {}).get("hiv")
            if hiv is not None:
                seeds.setdefault(sx_name, set()).add(float(hiv.get("init", 0.0)))
    if not seeds:
        return
    shown = ", ".join(
        f"{sx} {'/'.join(f'{v:.4f}' for v in sorted(vals))}"
        for sx, vals in sorted(seeds.items())
    )
    print(f"  NonInj HIV seed (fitted, NOT data-anchored): {shown}")


def main():
    trials = [t.strip() for t in os.environ.get("TITAN_VAL_TRIALS", "54,172,111").split(",")]
    n_seeds = int(os.environ.get("TITAN_VAL_SEEDS", 5))
    n_reps = int(os.environ.get("TITAN_VAL_REPS", 10))
    seeds = POP_SEEDS[:n_seeds]

    targets = load_targets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Validating trials {trials} over {len(seeds)} population seeds x {n_reps} reps")
    print(f"Trial scenarios from: {TRIAL_DIR}")
    print(f"Outputs to:           {OUT_DIR}\n", flush=True)

    rows = []
    for trial in trials:
        src = TRIAL_DIR / f"trial_{trial}.yml"
        if not src.exists():
            print(f"SKIP trial {trial}: {src} not found")
            continue
        base = load_yaml(src)

        for seed in seeds:
            scenario = copy.deepcopy(base)
            scenario["model"]["seed"]["ppl"] = int(seed)
            scenario["model"]["seed"]["run"] = 0  # keep per-rep run seeds random

            tag = f"trial{trial}_seed{seed}"
            yml = OUT_DIR / f"{tag}.yml"
            outdir = OUT_DIR / tag
            outdir.mkdir(exist_ok=True)
            write_yaml(yml, scenario)

            print(f"--- {tag}", flush=True)
            stats = run_titan(yml, outdir, n_reps=n_reps)
            errs = multi_error(stats, targets)

            row = {"trial": trial, "pop_seed": seed}
            row.update({f"err_{i}": e for i, e in enumerate(errs)})
            # Raw pooled infection count, not just the RMSE: a trial can score well
            # on HIV trajectory with 0-1 stochastic infections (see
            # docs/calibration_log/phase_4_ridoh_idu_validation_2026-07-21.md).
            row["hiv_new_pooled"] = stats.get("hiv_new_pooled", 0)
            # Validation quantities -- carried through but NOT scored, see
            # calibration_targets.yml `validation_targets`.
            for k in ("incar_share", "art_among_dx", "suppressed_among_dx",
                      "dx_among_hiv"):
                row[k] = stats.get(k, float("nan"))
            rows.append(row)
            print(
                "    "
                + "  ".join(
                    f"{name}={e:.4f}({e / tgt:.2f}x)"
                    for (name, tgt), e in zip(OBJECTIVES, errs)
                )
                + f"  hiv_new_pooled={row['hiv_new_pooled']}",
                flush=True,
            )

    if not rows:
        print("No trials evaluated.")
        return

    df = pd.DataFrame(rows)
    csv = OUT_DIR / "validation_summary.csv"
    df.to_csv(csv, index=False)
    print(f"\nPer-seed results: {csv}\n")

    print("=" * 78)
    print("SUMMARY (mean +/- sd over population seeds; pass = mean below target)")
    print("=" * 78)
    for trial in df["trial"].unique():
        sub = df[df["trial"] == trial]
        print(f"\nTrial {trial}  ({len(sub)} seeds)")
        n_pass = 0
        for i, (name, tgt) in enumerate(OBJECTIVES):
            vals = sub[f"err_{i}"].to_numpy()
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            seed_pass = int((vals <= tgt).sum())
            ok = mean <= tgt
            n_pass += ok
            print(
                f"  {name:<18} target<{tgt:<7.4g} mean={mean:.4f} sd={sd:.4f} "
                f"({mean / tgt:.2f}x)  seeds passing: {seed_pass}/{len(vals)}  "
                f"{'PASS' if ok else 'MISS'}"
            )
        print(f"  --> {n_pass}/{len(OBJECTIVES)} objectives pass on the seed mean")
        _report_noninj_seed(trial)
        _report_validation(sub)
        if "hiv_new_pooled" in sub:
            hiv_counts = sub["hiv_new_pooled"].to_numpy()
            print(
                f"  HIV transmission: pooled hiv_new per seed = {list(hiv_counts)}, "
                f"mean={float(np.mean(hiv_counts)):.1f}, "
                f"seeds with 0-1 infections: {int((hiv_counts <= 1).sum())}/{len(hiv_counts)}"
            )

    print(
        "\nRead: an objective whose mean sits inside target and whose sd spans the "
        "10-rep value was a noise miss; one that misses on every seed is structural."
    )

    _rank_candidates(df)


def _rank_candidates(df):
    """
    Rank the validated trials by composite error on their seed-mean objectives.

    Without this the script only reports pass/fail per objective, which does not
    decide anything once more than one candidate passes all six -- the gap that
    forced the manual trial 5 -> 101 swap in Phase 4. The composite is the same
    one select_canonical() uses, but computed on means over 5 population seeds x
    10 reps instead of the single 10-rep value NSGA-II sorted on: square-root the
    SQUARED_OBJECTIVES slots so all six enter on the scale of the quantity they
    measure, normalize each onto [min, COMPOSITE_CLIP_QUANTILE] of the pool with
    anything above the anchor clipped to 1.0, then sum. Both halves matter -- see
    select_canonical for the job-4586902 measurement that motivated them.

    Normalization is relative to the pool being ranked, so these scores are NOT
    comparable to the composite printed by the calibration run, which normalizes
    over the eligible members of the whole front.
    """
    trials = list(df["trial"].unique())
    if len(trials) < 2:
        print("\nOnly one trial evaluated -- nothing to rank.")
        return

    means = np.array([
        [float(np.mean(df[df["trial"] == t][f"err_{i}"].to_numpy()))
         for i in range(len(OBJECTIVES))]
        for t in trials
    ])
    scaled = means.copy()
    for i in SQUARED_OBJECTIVES:
        scaled[:, i] = np.sqrt(scaled[:, i])
    lo = scaled.min(axis=0)
    hi = np.quantile(scaled, COMPOSITE_CLIP_QUANTILE, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    composite = np.where(hi > lo, np.minimum(1.0, (scaled - lo) / span), 0.0).sum(axis=1)

    n_pass = [(means[k] <= [tgt for _, tgt in OBJECTIVES]).sum() for k in range(len(trials))]
    hiv_pooled = [float(np.mean(df[df["trial"] == t]["hiv_new_pooled"].to_numpy()))
                  for t in trials]

    rank = pd.DataFrame({
        "trial": trials,
        "composite": composite,
        "objectives_passed": n_pass,
        "hiv_new_pooled_mean": hiv_pooled,
    }).sort_values(["objectives_passed", "composite"], ascending=[False, True])

    csv = OUT_DIR / "validation_ranking.csv"
    rank.to_csv(csv, index=False)

    print("\n" + "=" * 78)
    print("RANKING (objectives passed desc, then composite asc)")
    print("=" * 78)
    print(rank.to_string(index=False))
    print(f"\nRanking written to: {csv}")

    top = rank.iloc[0]
    if top["objectives_passed"] < len(OBJECTIVES):
        print(
            f"\nNO CLEAN WINNER: best trial {top['trial']} passes only "
            f"{int(top['objectives_passed'])}/{len(OBJECTIVES)} objectives on the seed "
            "mean. Ranking it first is a least-bad ordering, not a canonical pick."
        )
    else:
        ties = rank[(rank["objectives_passed"] == len(OBJECTIVES))]
        print(
            f"\nCanonical candidate: trial {top['trial']} "
            f"(composite={top['composite']:.4f}, {len(ties)} trials pass all "
            f"{len(OBJECTIVES)} objectives)"
        )


if __name__ == "__main__":
    main()
