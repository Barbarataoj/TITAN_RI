"""
Rebuild validation_summary.csv from saved basicReport.txt files, no re-simulation.

Why this exists
---------------
make_phase7_ensemble.py launches several validate_candidates shards that share
one TITAN_VAL_OUT. Each shard writes validation_summary.csv at the end, so the
last one to finish overwrites every other shard's rows (observed 2026-10-01 on
jobs 6919501-6919508: the surviving CSV held only shard 0's 30 trials). The
per-batch directories are untouched, and basicReport.txt in each one is the
complete record the shard scored from, so the summary is reconstructible.

Each <out_dir>/trial<trial>_seed<seed>/basicReport.txt is parsed with the same
parse_basic_report / multi_error / build_row chain validate_candidates uses, so
the rebuilt rows are byte-for-byte what the shard would have written.

Usage
-----
    python scripts/rebuild_validation_summary.py \
        --out-dir /oscar/scratch/$USER/titan_phase7/ensemble/out \
        [--csv <path>] [--reps 40] [--workers 16] [--full]

Incremental by default: a (trial, pop_seed) already in --csv is not re-parsed,
so re-running after the last shard lands only scores the new directories.
Directories whose report holds fewer than --reps replicates are still being
written by a running shard and are skipped (reported, not scored).

Run on a compute node: ~1900 reports of ~50 MB each.
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.calibration.optuna_calibration import (  # noqa: E402
    CALIB_YAML, load_yaml, multi_error, parse_basic_report, scoring_t_max,
)
from titan.calibration.validate_candidates import OBJECTIVES, build_row, load_targets  # noqa: E402

# Gates read the calibration window only (steps <= 132), whatever the run length.
SCORE_T_MAX = scoring_t_max(load_yaml(CALIB_YAML))

DIR_RE = re.compile(r"^trial(?P<trial>.+)_seed(?P<seed>\d+)$")

_TARGETS = None


def _init(targets):
    global _TARGETS
    _TARGETS = targets


def _score(args):
    """Worker: parse one batch directory into a summary row (or a skip reason)."""
    path, trial, seed, min_reps = args
    t0 = time.time()
    try:
        stats = parse_basic_report(path / "basicReport.txt", score_t_max=SCORE_T_MAX)
    except Exception as exc:  # unreadable or truncated report
        return {"trial": trial, "pop_seed": seed, "_skip": f"parse error: {exc}"}
    n_reps = int(stats.get("n_reps", 0))
    if n_reps < min_reps:
        return {"trial": trial, "pop_seed": seed, "_skip": f"incomplete: {n_reps}/{min_reps} reps"}
    errs = multi_error(stats, _TARGETS)
    row = build_row(trial, seed, stats, _TARGETS, errs)
    row["n_reps"] = n_reps
    row["_secs"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="TITAN_VAL_OUT of the shards")
    ap.add_argument("--csv", type=Path, default=None,
                    help="output CSV (default: <out-dir>/../validation_summary_rebuilt.csv, "
                         "outside out/ so no shard can overwrite it)")
    ap.add_argument("--reps", type=int, default=40, help="replicates a complete batch must hold")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--full", action="store_true", help="re-score everything, ignore the existing CSV")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    csv = args.csv or (out_dir.parent / "validation_summary_rebuilt.csv")

    done = set()
    prior = None
    if csv.exists() and not args.full:
        prior = pd.read_csv(csv)
        prior["trial"] = prior["trial"].astype(str)
        done = set(zip(prior["trial"], prior["pop_seed"].astype(int)))
        print(f"existing {csv}: {len(prior)} rows, will skip those")

    todo = []
    for d in sorted(out_dir.iterdir()):
        m = DIR_RE.match(d.name)
        if not (d.is_dir() and m and (d / "basicReport.txt").exists()):
            continue
        trial, seed = m["trial"], int(m["seed"])
        if (trial, seed) in done:
            continue
        todo.append((d, trial, seed, args.reps))
    print(f"{len(todo)} directories to score under {out_dir} with {args.workers} workers", flush=True)
    if not todo:
        print("nothing to do")
        return

    targets = load_targets()
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(targets,)) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(_score, todo), 1):
            results.append(r)
            if "_skip" in r:
                print(f"  skip {r['trial']} seed {r['pop_seed']}: {r['_skip']}", flush=True)
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} done, {time.time() - t0:.0f}s elapsed", flush=True)

    rows = [r for r in results if "_skip" not in r]
    skipped = [r for r in results if "_skip" in r]
    if not rows and prior is None:
        print("no complete batches found; nothing written")
        return

    new = pd.DataFrame(rows).drop(columns=["_secs"], errors="ignore")
    df = pd.concat([prior, new], ignore_index=True) if prior is not None else new
    df["trial"] = df["trial"].astype(str)
    df = df.sort_values(["trial", "pop_seed"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}: {len(df)} rows ({len(new)} new, {len(skipped)} skipped), "
          f"{df['trial'].nunique()} trials")

    # Per-trial pass count on the seed mean, same rule as validate_candidates.
    gates = [tgt for _, tgt in OBJECTIVES]
    g = df.groupby("trial")[[f"err_{i}" for i in range(len(gates))]].mean()
    n_seeds = df.groupby("trial")["pop_seed"].count()
    g["n_seeds"] = n_seeds
    g["passed"] = sum((g[f"err_{i}"] <= gates[i]).astype(int) for i in range(len(gates)))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print("\nseed-mean objectives per trial (passed = count of err_i <= gate):")
    print(g.round(4).to_string())
    short = g[g["n_seeds"] < g["n_seeds"].max()]
    if len(short):
        print(f"\n{len(short)} trials have fewer seeds than the rest (shards still running?); re-run later.")


if __name__ == "__main__":
    main()
