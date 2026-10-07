#!/usr/bin/env python
"""
Re-score pooled validation reports against the current targets, no re-simulation.

validate_candidates leaves ``trial<trial>_pooled_basicReport.txt`` beside its
outputs so a trial's pooled (gate) score can be re-read without re-running
anything. This script does that for every pooled report under the given
directories, with the targets, objective and scoring window as they stand
now, and writes one CSV per directory with the same columns as
``validation_pooled.csv`` (build_row + raw_<i> + pass_<i> + pass4) plus a
``draw`` label.

Why: the HIV objective's scored windows changed on 2026-10-07 and the gates
of 240-step runs are now read over the calibration window; the reports on
disk are the record, the scores are recomputed from them.

Usage (compute node; each pooled report is 400-600 MB):
    python scripts/rescore_pooled_reports.py --out <csv_dir> \
        <label>=<dir_with_pooled_reports> [<label>=<dir> ...] [--workers 4]
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.calibration.optuna_calibration import (  # noqa: E402
    CALIB_YAML, load_yaml, multi_error, parse_basic_report, scoring_t_max,
)
from titan.calibration.validate_candidates import OBJECTIVES, build_row, load_targets  # noqa: E402

_TARGETS = None
_T_MAX = None


def _init(targets, t_max):
    global _TARGETS, _T_MAX
    _TARGETS, _T_MAX = targets, t_max


def _score(args):
    label, path = args
    trial = path.name.split("_pooled_basicReport")[0].replace("trial", "", 1)
    stats = parse_basic_report(path, score_t_max=_T_MAX)
    errs = tuple(float(e) for e in multi_error(stats, _TARGETS))
    raw = tuple(float(e) for e in multi_error(stats, _TARGETS, bias_correct=False))
    row = build_row(trial, "pooled", stats, _TARGETS, errs)
    for i, e in enumerate(raw):
        row[f"raw_{i}"] = e
    for i, (_, gate) in enumerate(OBJECTIVES):
        row[f"pass_{i}"] = bool(errs[i] <= gate)
    row["pass4"] = all(row[f"pass_{i}"] for i in range(len(OBJECTIVES)))
    row["n_reps"] = int(stats.get("n_reps", 0))
    row["draw"] = label
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="<label>=<directory holding trial*_pooled_basicReport.txt>")
    ap.add_argument("--out", type=Path, required=True, help="directory for the per-set CSVs")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = ap.parse_args()

    targets = load_targets()
    t_max = scoring_t_max(load_yaml(CALIB_YAML))
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for spec in args.sets:
        label, folder = spec.split("=", 1)
        reports = sorted(Path(folder).glob("trial*_pooled_basicReport.txt"))
        if not reports:
            print(f"{label}: no pooled reports under {folder}")
        tasks += [(label, p) for p in reports]
    with Pool(args.workers, initializer=_init, initargs=(targets, t_max)) as pool:
        rows = pool.map(_score, tasks)
    df = pd.DataFrame(rows)
    for label, sub in df.groupby("draw"):
        dest = args.out / f"validation_pooled_rescored_{label}.csv"
        sub.to_csv(dest, index=False)
        print(f"{label}: {len(sub)} trials -> {dest}")
    cols = ["draw", "trial"] + [f"err_{i}" for i in range(len(OBJECTIVES))] + ["pass4"]
    pd.set_option("display.width", 200)
    print(df[cols].sort_values(["trial", "draw"]).to_string(index=False))


if __name__ == "__main__":
    main()
