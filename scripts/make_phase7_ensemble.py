#!/usr/bin/env python3
"""
Build and submit the Phase 7 ensemble: every scenario arm for every parameter
set that passed all four gates at 8 x 40.

WHY. A single calibrated vector carries no parameter uncertainty, and at
N = 500 the HIV outcome -- the primary outcome -- is Monte Carlo noise at one
vector's 8 x 40 (seed-difference sd 0.12-0.33 infections per 500 over the
horizon against effects of about 0.1). Running every arm over every 8 x 40
pass does both jobs at once: the spread across members is the parameter
interval, and the pooled contrast over (member x seed) has roughly 30 times
the replicates of a single vector (section 10.3 item 5 of
phase_6_pins_and_noise_correction_2026-09-29.md).

WHAT IT DOES. For each member it builds the Phase 7 baseline exactly as
scripts/make_phase7_baseline.py does (every anchor re-applied, including the
anchored-parameters table, population seed derived, horizon set) and the arms
exactly as scripts/make_phase7_scenarios.py does, then stages them as
``trial_<member>__<arm>.yml`` for validate_candidates and submits them in
batches. scripts/phase7_contrast.py --split-members reads the result.

Usage (needs titan_env; a YAML build, fine on a login node, but submit from
the repo):
    python scripts/make_phase7_ensemble.py --out-dir /oscar/scratch/$USER/titan_phase7/ensemble \
        [--summary ensemble_8x40_summary.csv] [--steps 240] [--arms all|list]
        [--seeds 8] [--reps 40] [--batches 8] [--submit]
"""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from titan.calibration.optuna_calibration import CALIB_YAML, load_yaml  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


baseline_mod = _load("make_phase7_baseline")
arms_mod = None  # chosen in main() by --arms-module

SCRATCH = Path(f"/oscar/scratch/{os.environ.get('USER', 'jtao9')}/titan_calibration")
DEFAULT_SUMMARY = SCRATCH / "ensemble_8x40_summary.csv"


TRIALS_DIR = None  # set by --trials-dir in main()


def member_yaml(name):
    """Where a pass-list member's full scenario lives."""
    if TRIALS_DIR is not None:
        return Path(TRIALS_DIR) / f"trial_{name}.yml"
    if name.startswith("282g"):
        return SCRATCH / "val_282_neighbors" / "trials" / f"trial_{name}.yml"
    return SCRATCH / "optuna_multi_runs" / f"trial_{name}.yml"


def members_from_summary(path):
    with open(path) as fh:
        return sorted(r["trial"] for r in csv.DictReader(fh) if r.get("pass4") == "True")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--members", default=None, help="comma-separated member names instead of the summary's pass list")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--arms", default="all", help="'all' or comma-separated scenario stems, e.g. scenario_1_baseline,scenario_6_combination")
    ap.add_argument("--trials-dir", default=None,
                    help="directory holding trial_<member>.yml for every member (e.g. the copy a re-score was run from); "
                         "default: optuna_multi_runs/, which is overwritten per study (2026-10-02)")
    ap.add_argument("--arms-module", default="make_phase7_scenarios",
                    help="scripts/<name>.py providing build_arms(baseline) -> {filename: (scenario, notes)}; "
                         "make_phase7_pc_arms for the primary-care strategy arms (2026-10-02)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    calib = load_yaml(CALIB_YAML)
    global arms_mod, TRIALS_DIR
    arms_mod = _load(args.arms_module)
    TRIALS_DIR = args.trials_dir
    members = args.members.split(",") if args.members else members_from_summary(args.summary)
    trials_dir = args.out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    arithmetic = []
    names = []
    for m in members:
        src = member_yaml(m)
        if not src.exists():
            print(f"SKIP {m}: {src} not found")
            continue
        base = baseline_mod.build(load_yaml(src), calib, steps=args.steps)
        arms = {"scenario_1_baseline.yml": (base, ["baseline"])}
        arms.update(arms_mod.build_arms(base))
        for fname, (scenario, notes) in arms.items():
            stem = fname[:-4]
            if args.arms != "all" and stem not in args.arms.split(","):
                continue
            tag = f"{m}__{stem}"
            with open(trials_dir / f"trial_{tag}.yml", "w") as fh:
                yaml.safe_dump(scenario, fh, default_flow_style=False, sort_keys=False)
            names.append(tag)
            arithmetic.append(f"{tag}: " + " | ".join(notes))
    (args.out_dir / "members.txt").write_text("\n".join(members) + "\n")
    (args.out_dir / "arithmetic.txt").write_text("\n".join(arithmetic) + "\n")
    print(f"{len(members)} members x arms -> {len(names)} scenarios in {trials_dir}")

    if not args.submit:
        (args.out_dir / "trials.txt").write_text(",".join(names) + "\n")
        print("not submitted (--submit); trial list in trials.txt")
        return 0
    per = (len(names) + args.batches - 1) // args.batches
    env = dict(os.environ, TITAN_VAL_SEEDS=str(args.seeds), TITAN_VAL_REPS=str(args.reps),
               TITAN_VAL_TRIAL_DIR=str(trials_dir), TITAN_VAL_OUT=str(args.out_dir / "out"))
    jobs = []
    for b in range(args.batches):
        chunk = names[b * per:(b + 1) * per]
        if not chunk:
            continue
        env["TITAN_VAL_TRIALS"] = ",".join(chunk)   # list-valued: through the environment, --export=ALL
        out = subprocess.run(
            ["sbatch", "--parsable", f"--job-name=ens7_{b}", "--cpus-per-task=16",
             f"--time={args.hours}:00:00", "--export=ALL", "run_validate_candidates.slurm"],
            cwd=REPO, env=env, capture_output=True, text=True, check=True)
        jobs.append(out.stdout.strip())
        print(f"batch {b}: job {jobs[-1]}: {len(chunk)} scenarios")
    (args.out_dir / "jobs.txt").write_text("\n".join(jobs) + "\n")
    print(f"outputs: {args.out_dir}/out/trial<member>__<arm>_seed<seed>/basicReport.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
