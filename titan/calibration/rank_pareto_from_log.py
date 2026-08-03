"""
Recover the Pareto front from an Optuna calibration log and rank it by composite.

The calibration study is created with `optuna.create_study(...)` and no storage
(optuna_calibration.py main()), so the front exists only in-process during the
run. Everything needed to rebuild it is nonetheless in the SLURM stdout: optuna
logs `Trial <n> finished with values: [...]` for every completed trial.

This re-derives the non-dominated front from those lines and ranks it with the
SAME composite that select_canonical() uses (min-max normalization over the
front, summed across objectives), so the ordering here is consistent with the
canonical pick the calibration itself printed.

`hiv_new_pooled` is a user_attr and is NOT logged, so the transmission-activity
screen cannot be applied from the log alone. It is instead recomputed for the
top --rescan candidates by re-parsing each trial's saved basicReport.txt, which
is the same quantity parse_basic_report() produced during the run.

Usage:
    python -m titan.calibration.rank_pareto_from_log \
        --log /oscar/scratch/$USER/titan_calibration/calib_4436043.out \
        --top 20
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    load_yaml,
    min_transmission_events,
    parse_basic_report,
)

TRIAL_LINE = re.compile(r"Trial (\d+) finished with values: \[([^\]]*)\]")


def parse_log(path):
    """Return {trial_number: (v0..vn)} for every completed trial in the log."""
    trials = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = TRIAL_LINE.search(line)
            if not m:
                continue
            num = int(m.group(1))
            vals = tuple(float(x) for x in m.group(2).split(","))
            trials[num] = vals
    return trials


def pareto_front(trials):
    """
    Non-dominated subset under minimization on all objectives.

    O(n^2) on 4000 trials is ~8M comparisons, vectorized per-candidate against
    the full matrix -- a few seconds, not worth a smarter algorithm.
    """
    nums = np.array(sorted(trials))
    mat = np.array([trials[n] for n in nums])
    keep = []
    for i in range(len(nums)):
        # j dominates i iff j is <= on every objective and < on at least one
        le = (mat <= mat[i]).all(axis=1)
        lt = (mat < mat[i]).any(axis=1)
        if not (le & lt).any():
            keep.append(i)
    return nums[keep], mat[keep]


def composite_scores(mat):
    """select_canonical()'s composite: min-max normalize per objective, then sum."""
    lo, hi = mat.min(axis=0), mat.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = np.where(hi > lo, (mat - lo) / span, 0.0)
    return norm.sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--trial-dir", type=Path,
                    default=Path("/oscar/scratch/jtao9/titan_calibration/optuna_multi_runs"))
    ap.add_argument("--top", type=int, default=20,
                    help="how many candidates to emit for validation")
    ap.add_argument("--rescan", type=int, default=60,
                    help="how many top-composite trials to re-parse for hiv_new_pooled")
    ap.add_argument("--out", type=Path, default=Path("pareto_ranked.csv"))
    args = ap.parse_args()

    trials = parse_log(args.log)
    print(f"Parsed {len(trials)} completed trials from {args.log}")

    nums, mat = pareto_front(trials)
    print(f"Pareto front: {len(nums)} parameter sets")

    comp = composite_scores(mat)
    order = np.argsort(comp)

    min_events = min_transmission_events(load_yaml(CALIB_YAML))
    print(f"Transmission-activity screen: >= {min_events} pooled PWID infections")

    rows = []
    for rank, i in enumerate(order):
        n = int(nums[i])
        # Only the head of the front is re-parsed: basicReport.txt is ~6 MB per
        # trial, and anything below the top --rescan by composite cannot enter a
        # top-`top` list even if every screened candidate above it is dropped.
        pooled = np.nan
        if rank < args.rescan:
            report = args.trial_dir / f"trial_{n}" / "basicReport.txt"
            if report.exists():
                pooled = parse_basic_report(report).get("hiv_new_pooled", 0)
        rows.append({
            "trial": n,
            "composite": float(comp[i]),
            **{f"obj_{j}": float(mat[i][j]) for j in range(mat.shape[1])},
            "hiv_new_pooled": pooled,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Ranked front written to {args.out}")

    eligible = df[df["hiv_new_pooled"] >= min_events]
    dropped = df.head(args.rescan).shape[0] - eligible.shape[0]
    print(f"{len(eligible)}/{args.rescan} rescanned trials clear the transmission screen "
          f"({dropped} dropped)")

    picks = eligible.head(args.top)
    print(f"\nTop {len(picks)} by composite (transmission-screened):")
    print(picks[["trial", "composite", "hiv_new_pooled"]].to_string(index=False))
    print("\nTITAN_VAL_TRIALS=" + ",".join(str(int(t)) for t in picks["trial"]))


if __name__ == "__main__":
    main()
