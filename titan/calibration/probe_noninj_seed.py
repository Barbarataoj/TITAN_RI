"""
Diagnostic: can the model produce a DECLINING PWID HIV incidence trajectory?

Measured 2026-08-03 (job 4576015, 20 Pareto members at 5 pop seeds x 10 reps):
the model emits essentially flat incidence, ~0.0035/PWID-yr for ten of eleven
scored years, while the target falls 36% from T1 to T2. No candidate beat a
flat-at-mean predictor, and aggregating the objective to period means did not
change that (discrimination 1.17 -> 1.19). So the failure is not the objective.

Hypothesis under test: `hiv.init` is set to 0.075 on drug_type Inj ONLY. Every
NonInj agent -- 75% of each sex_type stratum, including all MSM -- starts
HIV-negative, and the model grows that reservoir from zero to ~0.060 prevalence
across the run. That is a RISING force of infection through exactly the window
where the data declines, which would hold the trajectory flat regardless of how
the transmission parameters are fitted.

This sweeps the NonInj seed and reports the trajectory SHAPE, not an objective
value. The question is whether the T1->T2 ratio moves toward the target's -36%,
not whether any particular seed value is right -- the true value is a
parameterisation question needing a data anchor (RI HIV prevalence among
non-injecting people with OUD, and separately among MSM), which this does not
attempt to answer.

dx.init is seeded alongside at the same 0.9 the Inj strata use: seeding
prevalence without diagnosis would put the whole new reservoir outside the ART
cascade and inflate transmission for a reason unrelated to the hypothesis.

Usage:
    python -m titan.calibration.probe_noninj_seed --trials 2945,2401 \
        --seeds 3 --inits 0.0,0.01,0.03,0.06
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    load_yaml,
    parse_basic_report,
    run_titan,
    write_yaml,
)

POP_SEEDS = [12345, 20260720, 8675309, 314159, 271828]


def set_noninj_hiv(scenario, init, dx_init=0.9):
    """Set hiv.init/hiv.dx.init on every NonInj stratum, leaving Inj untouched."""
    n = 0
    for race in scenario.get("demographics", {}).values():
        for sx in (race or {}).get("sex_type", {}).values():
            nd = (sx or {}).get("drug_type", {}).get("NonInj")
            if nd is None:
                continue
            nd["hiv"] = {"init": float(init), "dx": {"init": float(dx_init)}}
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="2945,2401")
    ap.add_argument("--inits", default="0.0,0.01,0.03,0.06")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--trial-dir", type=Path,
                    default=Path(f"/oscar/scratch/{os.environ['USER']}/titan_calibration/optuna_multi_runs"))
    ap.add_argument("--out", type=Path,
                    default=Path(f"/oscar/scratch/{os.environ['USER']}/titan_calibration/noninj_probe"))
    args = ap.parse_args()

    hiv = load_yaml(CALIB_YAML)["calibration"]["hiv"]
    S = np.array(hiv["incidence_series_per_pwid_yr"], dtype=float)
    Y = np.array(hiv["incidence_series_years"])
    P = hiv["incidence_periods"]
    groups = [np.where((Y >= a) & (Y <= b))[0] for a, b in groups_of(P)]
    tgt = np.array([S[g].mean() for g in groups])

    trials = [t.strip() for t in args.trials.split(",")]
    inits = [float(x) for x in args.inits.split(",")]
    seeds = POP_SEEDS[:args.seeds]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"target periods: T1={tgt[0]:.4f} T2={tgt[1]:.4f} T3={tgt[2]:.4f}  "
          f"(T1->T2 {tgt[1]/tgt[0]-1:+.0%})")
    print(f"trials={trials} inits={inits} seeds={len(seeds)} reps={args.reps}\n", flush=True)

    rows = []
    for trial in trials:
        src = args.trial_dir / f"trial_{trial}.yml"
        if not src.exists():
            print(f"SKIP trial {trial}: {src} missing")
            continue
        base = load_yaml(src)
        for init in inits:
            for seed in seeds:
                sc = copy.deepcopy(base)
                n = set_noninj_hiv(sc, init)
                sc["model"]["seed"]["ppl"] = int(seed)
                sc["model"]["seed"]["run"] = 0

                tag = f"t{trial}_i{init}_s{seed}"
                yml = args.out / f"{tag}.yml"
                od = args.out / tag
                od.mkdir(exist_ok=True)
                write_yaml(yml, sc)
                stats = run_titan(yml, od, n_reps=args.reps)

                ser = np.asarray(stats.get("hiv_incidence_series_pwid"), dtype=float)
                ser = ser[-len(S):]
                per = np.array([ser[g].mean() for g in groups])
                rows.append({
                    "trial": trial, "noninj_init": init, "pop_seed": seed,
                    "strata_set": n,
                    "T1": per[0], "T2": per[1], "T3": per[2],
                    "t1_t2_change": per[1] / per[0] - 1 if per[0] > 0 else np.nan,
                    "pwid_prev": stats.get("hiv_prevalence_pwid", np.nan),
                    "hiv_new_pooled": stats.get("hiv_new_pooled", 0),
                })
                print(f"  {tag}: T1={per[0]:.4f} T2={per[1]:.4f} T3={per[2]:.4f} "
                      f"T1->T2={rows[-1]['t1_t2_change']:+.0%} "
                      f"prev={rows[-1]['pwid_prev']:.4f} new={rows[-1]['hiv_new_pooled']}",
                      flush=True)

    df = pd.DataFrame(rows)
    csv = args.out / "noninj_probe.csv"
    df.to_csv(csv, index=False)

    print("\n" + "=" * 72)
    print("TRAJECTORY SHAPE vs NonInj HIV seed (mean over trials x pop seeds)")
    print("=" * 72)
    g = df.groupby("noninj_init").agg(
        T1=("T1", "mean"), T2=("T2", "mean"), T3=("T3", "mean"),
        t1_t2=("t1_t2_change", "mean"), prev=("pwid_prev", "mean"),
        new=("hiv_new_pooled", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\ntarget:  T1={tgt[0]:.4f} T2={tgt[1]:.4f} T3={tgt[2]:.4f}  "
          f"T1->T2={tgt[1]/tgt[0]-1:+.0%}")
    print(f"\nWritten: {csv}")
    print(
        "\nRead: if t1_t2 moves toward the target's -36% as the seed rises, the flat "
        "trajectory is an initial-condition artifact and calibration is viable. If it "
        "stays near 0 at every seed, the model has no mechanism that suppresses "
        "transmission over time and needs one -- that is model development, not fitting."
    )


def groups_of(periods):
    """Accept [[a,b],...] from YAML (loads as lists, not tuples)."""
    return [(int(a), int(b)) for a, b in periods]


if __name__ == "__main__":
    main()
