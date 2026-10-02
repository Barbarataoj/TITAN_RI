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
    OBJECTIVE_GATES,
    SQUARED_OBJECTIVES,
    build_targets,
    load_yaml,
    multi_error,
    run_titan,
    write_yaml,
)

# Objective order must match multi_error()'s return tuple.
#
# THE NUMBERS COME FROM optuna_calibration.OBJECTIVE_GATES as of 2026-09-01, and
# the reasoning for each one stays here. The search reads the same tuple as
# NSGA-II constraints (D2), so a gate stated in two files would be a gate that
# drifts: the run would be selecting against one threshold and validated against
# another, and nothing would report the difference. Only the labels and the
# derivations live here now.
OBJECTIVES = [
    ("OUD distribution", OBJECTIVE_GATES[0]),
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
    #
    # SUPERSEDED 2026-08-04. The objective this gate scored is gone; slot 1 is now
    # ridoh_idu_err. The reasoning above is kept because it applies verbatim to
    # the replacement: do not raise a gate until something passes.
    #
    # The new gate is in statewide diagnoses per year among people who inject
    # drugs, RMSE across four RIDOH pooled windows: injection-attributed against
    # the IDU category (1.8 and 2.5 per year) and all-route against IDU + MSM/IDU
    # (5.4 and 8.7 per year), plus the one-sided pre-2015 bound.
    #
    # WIDENED 2026-08-04 from IDU-only to both series. The IDU-only version left
    # the sexual and background routes unconstrained while
    # calibration.acquisition scales all three at once, so a candidate could
    # score well by inerting the injection route. See ridoh_hiv_rmse.
    #
    # Reference values on this scale, all computed rather than estimated:
    #
    #   zero transmission (null)                     4.78
    #   best pre-refit candidate (trial 1283)        ~4.27
    #   injection route inerted, all-route on target  1.73
    #   BEST POSSIBLE FLAT series                     1.07
    #   flat series 20% high                          1.40
    #
    # The gate is 1.1, just above that flat-predictor floor, so passing means the
    # LEVEL is right on both series. It cannot be set below 1.07 without
    # demanding the model also reproduce RIDOH's rise (1.8 -> 2.5 and 5.4 -> 8.7),
    # and a shape-demanding HIV gate is exactly what produced 0/8 passes and
    # weeks of dead refits through July. Shape is reported separately instead.
    #
    # Nothing passes at 1.1 today either -- the best candidate is at 4.27, barely
    # better than a null model. That is the finding, not a reason to move the
    # gate.
    #
    # Note what the 1.73 row buys: a candidate that zeroes injection transmission
    # while matching all-route diagnoses scores WORSE than a real fit, so the
    # degenerate solution the IDU-only objective invited is now penalised.
    ("RIDOH PWID diagnoses", OBJECTIVE_GATES[1]),
    # HIV prevalence was dropped as an objective on 2026-08-04 -- APCD #25 has the
    # same any-transmission-route problem as the incidence series it was paired
    # with, and RIDOH publishes no PLWH breakdown by transmission category to
    # replace it (checked across the 2019, 2022 and 2024 surveillance reports).
    # It is still reported per-trial through the VALIDATION list below as an
    # unscored diagnostic; it is simply no longer gated.
    # RESCALED 2026-08-04 with the objective. od_err was a sum of ABSOLUTE
    # squared errors, so each component was weighted by the square of its own
    # scale and the smallest one drifted unpenalised: the canonical trial sat 52%
    # BELOW the fatal-overdose target and 44% below the non-injection nonfatal
    # target, scored 0.000598, and passed this gate at 0.12x. Every trial on the
    # 2026-08-04 front was 2-5x low on fatal overdose and every one passed.
    #
    # It is now the mean squared PROPORTIONAL error, so 0.01 means "within 10% on
    # all three components" -- the gate below is that, in the objective's squared
    # units. The canonical trial scores 0.1609, i.e. an RMS proportional miss of
    # 40%.
    ("Overdose", OBJECTIVE_GATES[2]),
    ("MOUD trajectory", OBJECTIVE_GATES[3]),
    # "Injection share", 0.0025 was the fifth objective from 2026-07-29
    # (EPIDEMIC_DYNAMICS_MAP_2026-07-29.md P2) until 2026-08-17, when it was
    # demoted to the VALIDATION list below. See multi_error() for the four
    # reasons. The quantity is still reported per trial; it is no longer scored.
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
    """
    Delegates to optuna_calibration.build_targets so this script and the
    calibration cannot score against different target dicts.

    This used to be a second hand-maintained copy. When inj_share_err was added
    as an objective on 2026-07-29 only the calibration copy was updated, and
    every run of this script from then until 2026-08-03 died with
    KeyError: 'inj_share' on its first trial -- while the OBJECTIVES list below
    had been updated, so the script looked current.
    """
    return build_targets(load_yaml(CALIB_YAML)["calibration"])


# The three rates od_err averages over: (stat key, label, key in targets dict).
# Names must match the stats keys set in optuna_calibration and the target keys
# built by build_targets.
OD_COMPONENT_TARGETS = [
    ("fatal_od_rate",           "fatal OD rate",       "od_fatal"),
    ("nonfatal_od_rate_inj",    "nonfatal OD (Inj)",   "od_nonfatal_inj"),
    ("nonfatal_od_rate_noninj", "nonfatal OD (NonInj)", "od_nonfatal_noninj"),
]
OD_COMPONENTS = [k for k, _, _ in OD_COMPONENT_TARGETS]

# Data-anchored inputs, reported but never scored. (stat key, label, target,
# tolerance) -- targets mirror calibration_targets.yml `validation_targets`.
# A miss here does NOT disqualify a trial; it means an input has drifted from the
# RIDOC / CDC-PWID observation it was set from, which is a parameterisation bug
# rather than a fit failure. See EPIDEMIC_DYNAMICS_AUDIT_2026-07-29b.md.
VALIDATION = [
    ("incar_share",         "incarcerated share",   0.0552, 0.015),
    ("art_among_dx",        "on ART | diagnosed",   0.780,  0.06),
    ("suppressed_among_dx", "suppressed | diagnosed", 0.572, 0.08),
    # Added 2026-08-04, when PWID HIV prevalence stopped being an objective.
    # Dropping it from the score should not drop it from view: it is the quantity
    # that exposed the APCD targets as jointly infeasible, and a refit that sends
    # it somewhere implausible needs to be visible even though nothing penalises
    # it. The 0.062 reference is APCD #25, retained as a comparison point and NOT
    # as a target -- it measures a claims-identified OUD population on any
    # transmission route. Tolerance is wide to signal that.
    ("hiv_prevalence_pwid", "PWID HIV prev (unscored)", 0.062, 0.04),
    # Stationarity of the scored window, added 2026-08-13. Target 0.0 = HIV
    # prevalence ends the window where it started; tolerance 0.10 = a 10% move
    # across eleven years, which is about the run-to-run spread at 40 reps.
    #
    # A fit measured across a window the epidemic is still traversing is valid
    # only for that window: change burn_steps or num_steps and it stops holding,
    # and a scenario run on a longer horizon shows the drift climbing as though
    # the intervention caused it. This went unnoticed for months and was
    # misattributed twice (to burn length, then to transmission level) before
    # step 3 traced it to a per-act value running 34% hot.
    #
    # Reported, not scored, for the same reason the others here are: drift is a
    # property of the fitted transmission level, so penalising it inside the
    # objective would trade it against HIV fit rather than surface it.
    ("hiv_prev_drift_pwid",   "PWID prev drift (unscored)",   0.0, 0.10),
    ("hiv_prev_drift_noninj", "NonInj prev drift (unscored)", 0.0, 0.10),
    # Demoted from the objective list 2026-08-17. Target 0.32 is RIHRSS
    # (PMC10872563: 125 of 393 people who used illicit drugs in the prior 30 days
    # reported injection), unchanged by the demotion -- it is also the value
    # demographics.*.drug_type.Inj.ppl is derived from, so a miss here means the
    # initial condition and the steady state implied by INJECT_START/INJECT_STOP
    # have pulled apart.
    #
    # Tolerance 0.05 preserves the old gate's width (0.0025 squared = 0.05
    # absolute) so the number stays comparable across the demotion. Expect it to
    # FAIL at burn 120: job 4963250 measured 0.262. That is the point of
    # reporting it -- the miss is a burn-length and initial-condition question
    # for Stage C, not something the optimizer should be trading HIV fit against.
    ("inj_share_active_oud",  "injection share | active OUD",  0.32, 0.05),
    # Added 2026-09-29 with the pinning of hiv_testing.base_prob_annual at
    # 0.155. NHBS-PWID's 45% tested in the past year, converted to tests per
    # PWID person-year (0.60 = -ln(1 - 0.45)); see calibration_targets.yml
    # validation_targets.hiv_testing for the band.
    ("hiv_tests_per_pwid_yr", "HIV tests / PWID-yr (NHBS 0.45)", 0.60, 0.18),
    # The fatal overdose rate left od_err the same day, when overdose.fatal_frac
    # became a derived quantity. Checked here so a miss on the nonfatal rates
    # still shows up on the fatal side, which is the Phase 7 headline outcome.
    ("fatal_od_rate", "fatal OD rate (unscored)", 0.015, 0.003),
    # Added 2026-09-29 after the HIV grid (job 6817937) found the model reaching
    # RIDOH's all-route PWID level with 0.4-0.8 HIV prevalence among non-injecting
    # MSM. RI GBMSM overall are ~0.07 (RIDOH 2022); MSM with OUD plausibly
    # 0.15-0.25. Target 0.20 +/- 0.10: a fit above 0.30 reached the HIV target
    # through the wrong reservoir.
    # Re-sourced 2026-09-29: 0.15 +/- 0.10, i.e. 0.05-0.25. Floor: ~7% of RI
    # GBMSM living with HIV (RIDOH 2022). Ceiling: 25% among MSM who INJECT
    # (NHBS-PWID 2018, CDC MMWR 2021;70(42)), which non-injecting MSM with OUD
    # should not exceed.
    ("hiv_prevalence_noninj_msm", "NonInj MSM HIV prev (unscored)", 0.15, 0.10),
    # APCD job 2b, 2018-2022: any MOUD within 30 days of the index overdose,
    # 13.2-18.7% (mean 15.4%), survivors already on MOUD included. Added
    # 2026-10-01 with the od_survivors_* / moud_start_postod_1step columns.
    ("survivor_any_moud_30d_share", "survivors w/ MOUD at 30 d (job 2b)", 0.154, 0.04),
]


def _report_overdose(sub, targets):
    """
    Decompose od_err into the three rates it averages over.

    od_err is the mean of three squared proportional errors, so the aggregate
    alone cannot distinguish "one component badly off" from "all three modestly
    off" -- and those have different fixes. Per-seed spread on each component is
    also what says whether the 0.01 gate (within 10% on all three at once) sits
    above or below this model's Monte Carlo floor at the current rep count.
    """
    have = [(k, label, targets[t]) for k, label, t in OD_COMPONENT_TARGETS
            if k in sub and not sub[k].isna().all()]
    if not have:
        return
    print("  overdose components (od_err = mean of these three squared prop. errors):")
    for key, label, tgt in have:
        vals = sub[key].to_numpy(dtype=float)
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        prop = (mean - tgt) / tgt
        # Share of od_err this component would contribute at the seed mean.
        print(
            f"    {label:<24} {mean:.5f} sd={sd:.5f}  target {tgt:.5f}  "
            f"{prop:+.0%}  (sq prop err {prop ** 2:.4f})"
        )


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


def build_row(trial, seed, stats, targets, errs=None):
    """
    One validation_summary.csv row for a (trial, population seed) pair.

    Split out of main() on 2026-10-01 so the row can be rebuilt from a saved
    basicReport.txt without re-simulating: eight ensemble shards writing the
    same OUT_DIR each overwrote validation_summary.csv, and the only
    full-fidelity record left was the per-batch report
    (scripts/rebuild_validation_summary.py).
    """
    if errs is None:
        errs = multi_error(stats, targets)
    row = {"trial": trial, "pop_seed": seed}
    row.update({f"err_{i}": e for i, e in enumerate(errs)})
    # Raw pooled infection count, not just the RMSE: a trial can score well
    # on HIV trajectory with 0-1 stochastic infections (see
    # docs/calibration_log/phase_4_ridoh_idu_validation_2026-07-21.md).
    row["hiv_new_pooled"] = stats.get("hiv_new_pooled", 0)
    # Injection-attributed diagnoses on RIDOH's statewide footing, which
    # is what objective slot 1 actually compares. Carried as a raw
    # quantity for the same reason hiv_new_pooled is: the RMSE alone does
    # not say whether a good score came from matching RIDOH or from
    # simulating almost no injection transmission at all.
    # Both series the objective compares, so a good RMSE can be read
    # against RIDOH's own numbers: IDU 1.8-2.5/yr, IDU + MSM/IDU
    # 5.4-8.7/yr. Their ratio is the model's injection-attributed share
    # of PWID diagnoses, which RIDOH puts at 29-40%.
    for key, col in (
        ("idu_dx_counts_per_year", "idu_dx_per_year_ri"),
        ("pwid_dx_counts_per_year", "pwid_dx_per_year_ri"),
    ):
        series = stats.get(key)
        row[col] = (
            float(np.mean(series)) * targets["pop_scale"]
            if series is not None and len(series)
            else float("nan")
        )
    # Validation quantities -- carried through but NOT scored, see
    # calibration_targets.yml `validation_targets`.
    # Every band in VALIDATION, so _report_validation() can print each one.
    # Until 2026-10-01 only five keys were carried, and the drift, injection
    # share, testing-rate and MSM-prevalence bands were defined but never
    # shown; adhoc item2_checks.py recovered them from the saved reports.
    for k in [v[0] for v in VALIDATION] + ["dx_among_hiv"]:
        row[k] = stats.get(k, float("nan"))
    # The three rates od_err is the mean squared proportional error OF.
    # Added 2026-08-12: od_err is a scalar mean over three components, so
    # a 2x miss on the aggregate does not say whether one component is
    # far off or all three are moderately off -- and the Optuna journal
    # stores only hiv_new_pooled in user_attrs, so the decomposition is
    # not recoverable from a finished study. Carrying the raw rates here
    # makes it recoverable for anything this script re-runs.
    for k in OD_COMPONENTS:
        row[k] = stats.get(k, float("nan"))
    return row


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

            row = build_row(trial, seed, stats, targets, errs)
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
    # Shards launched by make_phase7_ensemble.py share one OUT_DIR, so the
    # last one to finish overwrites validation_summary.csv. Keep a per-job
    # copy as well; the rebuild script merges them.
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        df.to_csv(OUT_DIR / f"validation_summary_{job_id}.csv", index=False)
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
        _report_overdose(sub, targets)
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
