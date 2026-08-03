"""
Can NSGA-II actually SELECT on each objective, or is it sorting noise?

This is a go/no-go check on the refit, not a fit. A 4000-trial NSGA-II run scores
every trial on ONE 10-rep evaluation. For an objective to drive selection, the
spread it takes across different parameter vectors must exceed the run-to-run
noise in any single vector's score. When it doesn't, the front's ranking on that
objective is a random draw, and 6 h of compute buys a front that cannot be
trusted on exactly the axis it was run for.

That is what happened before. Job 4576015 measured, across the 640-member front
at 5 populations x 10 reps:

    HIV trajectory   between-trial SD 0.00045 / SE 0.00039  ->  1.15
    HIV prevalence   between-trial SD 0.00049 / SE 0.00032  ->  1.50

Both under the ~2 needed to rank reliably. The cause was that every trial emitted
essentially the same flat trajectory -- with NonInj HIV unseeded the model had no
parameter that could change the trajectory's shape, so the between-trial spread
collapsed and noise was all that was left.

calibration.noninj_hiv_init_{het,msm} (added 2026-08-03) should widen that
spread. This measures whether it does, BEFORE the refit spends the 6 h.

Method matters in one respect: this pins model.seed.ppl at the calibration value
and replicates over run seed only. That is deliberate -- it reproduces the noise
the SAMPLER sees. Population variance is excluded from the fit by the pinned
seed and only reappears at validation, so folding it in here would measure a
quantity NSGA-II never encounters and understate discrimination.

    ratio > 3  objective ranks reliably
    ratio > 2  usable
    ratio < 2  selection on this objective is mostly noise

Usage:
    python -m titan.calibration.probe_discrimination --trials 2566,2945,2812 --batches 3
"""

import argparse
import copy
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    load_yaml,
    multi_error,
    run_titan,
    set_nested,
    set_noninj_hiv_init,
    write_yaml,
)
from titan.calibration.validate_candidates import OBJECTIVES, load_targets

CALIB_POP_SEED = 12345  # must match BASE_SCENARIO model.seed.ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="2566,2945,2812",
                    help="trial YAMLs used as base parameter vectors")
    ap.add_argument("--het", default="0.000,0.012,0.025",
                    help="calibration.noninj_hiv_init_het values to span")
    ap.add_argument("--msm", default="0.00,0.12",
                    help="calibration.noninj_hiv_init_msm values to span")
    # Swept since 2026-08-03. calibration.sex.act was a dead search dimension
    # from the day it was added (2026-07-29) until the fix in objective(): the
    # base scenario has no `calibration.sex` sub-dict, so set_yaml_value raised
    # and the value went to the engine's unused-parameter bin. Trial YAMLs on
    # disk therefore all carry the engine default. It is the lever that fits the
    # network sexual route and calibration_targets.yml records it as violently
    # sensitive, so any discrimination measured without it understates the real
    # between-trial spread on the HIV objectives.
    ap.add_argument("--sexact", default="0.5,1.5,2.5",
                    help="calibration.sex.act values to span (bounds 0.25-3.0)")
    ap.add_argument("--batches", type=int, default=3,
                    help="independent 10-rep evaluations per config (>=2 to get an SE)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--trial-dir", type=Path,
                    default=Path(f"/oscar/scratch/{os.environ['USER']}/titan_calibration/optuna_multi_runs"))
    ap.add_argument("--out", type=Path,
                    default=Path(f"/oscar/scratch/{os.environ['USER']}/titan_calibration/discrimination"))
    args = ap.parse_args()

    targets = load_targets()
    trials = [t.strip() for t in args.trials.split(",")]
    hets = [float(x) for x in args.het.split(",")]
    msms = [float(x) for x in args.msm.split(",")]
    sexacts = [float(x) for x in args.sexact.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    configs = [c for c in itertools.product(trials, hets, msms, sexacts)]
    print(f"{len(configs)} configs x {args.batches} batches x {args.reps} reps "
          f"= {len(configs) * args.batches} runs")
    print(f"population seed pinned at {CALIB_POP_SEED} (replicating over run seed only)\n",
          flush=True)

    rows = []
    for trial, het, msm, sexact in configs:
        src = args.trial_dir / f"trial_{trial}.yml"
        if not src.exists():
            print(f"SKIP {src} missing")
            continue
        base = load_yaml(src)
        cfg = f"t{trial}_h{het}_m{msm}_s{sexact}"
        for b in range(args.batches):
            sc = copy.deepcopy(base)
            set_noninj_hiv_init(sc, het=het, msm=msm)
            sc.setdefault("calibration", {})
            set_nested(sc["calibration"], "sex.act", sexact)
            sc["model"]["seed"]["ppl"] = CALIB_POP_SEED
            # 0 => the engine draws a fresh run seed per rep, so batches of the
            # same config are independent draws of exactly the quantity the
            # sampler sees.
            sc["model"]["seed"]["run"] = 0

            tag = f"{cfg}_b{b}"
            yml = args.out / f"{tag}.yml"
            od = args.out / tag
            od.mkdir(exist_ok=True)
            write_yaml(yml, sc)
            errs = multi_error(run_titan(yml, od, n_reps=args.reps), targets)
            rows.append({"config": cfg, "trial": trial, "het": het, "msm": msm,
                         "sexact": sexact, "batch": b,
                         **{f"err_{i}": e for i, e in enumerate(errs)}})
            print(f"  {tag}: " + " ".join(f"{e:.5f}" for e in errs), flush=True)

    df = pd.DataFrame(rows)
    csv = args.out / "discrimination.csv"
    df.to_csv(csv, index=False)

    print("\n" + "=" * 82)
    print("DISCRIMINATION  (between-config SD of config means / SE of one 10-rep score)")
    print("=" * 82)
    print(f"{'objective':<20} {'betw SD':>9} {'within SE':>10} {'ratio':>7} {'range across configs':>26}")
    verdicts = {}
    for i, (name, gate) in enumerate(OBJECTIVES):
        col = f"err_{i}"
        means = df.groupby("config")[col].mean()
        # Pooled within-config SD across batches, then SE of a single evaluation.
        within = df.groupby("config")[col].std(ddof=1)
        se = float(np.sqrt(np.nanmean(within.to_numpy() ** 2)))
        sd = float(means.std(ddof=1))
        ratio = sd / se if se > 0 else np.inf
        verdicts[name] = ratio
        print(f"{name:<20} {sd:9.5f} {se:10.5f} {ratio:7.2f} "
              f"{means.min():11.5f} - {means.max():.5f}")

    print("\n" + "=" * 82)
    print("VERDICT")
    print("=" * 82)
    for name, r in verdicts.items():
        tag = "RELIABLE" if r >= 3 else ("USABLE" if r >= 2 else "NOISE-DOMINATED")
        print(f"  {name:<20} {r:5.2f}  {tag}")
    bad = [n for n, r in verdicts.items() if r < 2]
    if bad:
        print(f"\n  {len(bad)} objective(s) below 2: {', '.join(bad)}")
        print("  NSGA-II will rank these mostly at random. Raising --reps is the direct")
        print("  fix (noise falls as 1/sqrt(reps)); it costs run time linearly, so the")
        print("  trial budget has to come down to match.")
    else:
        print("\n  All objectives at or above 2 -- the refit will select on signal.")
    print(f"\nWritten: {csv}")


if __name__ == "__main__":
    main()
