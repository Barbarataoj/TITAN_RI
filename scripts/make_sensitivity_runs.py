#!/usr/bin/env python3
"""
Emit the Phase 6 sensitivity variants of a calibrated scenario (section 10.1
item 4 of docs/calibration_log/phase_6_pins_and_noise_correction_2026-09-29.md).

Each variant is the canonical trial YAML with ONE anchored input moved to the
end of its evidence range, so that re-scoring every variant with
validate_candidates (8 populations x 40 replicates, the same fixed seeds as the
pass list) shows how far each pinned number can move the four objectives. The
unmodified scenario is emitted too, as ``sens_base``, so the contrast is taken
within the same code state and seed set.

Axes and ranges, as stated in section 10.1 item 4:

  inject_stop_rate_per_year   0.041 / 0.326   (INJECT_STOP_RATE, pinned 0.10)
  overdose.fatal_frac         fatal-rate target 0.012 / 0.027 -> fatal_frac via
                              derived_fatal_frac's formula (about 0.17 / 0.31)
  moud.overdose_scale.active  0.21 / 0.41
  moud.discontinue_per_month  0.069 / 0.124
  overdose.post_moud_stop_rr  2.4 / 3.3
  overdose.high_risk_mult     1.2 / 2.5
  MSM background prevalence   0.065 / 0.25   (MSM and MSMW partner_hiv_prevalence)
  MSM background rate         4.0            (0.5 is the base value)
  NonInj MSM HIV seed         the model's own long-run NonInj MSM prevalence,
                              passed with --msm-seed (skipped when not given)

Usage:
    python scripts/make_sensitivity_runs.py TRIAL.yml --out-dir DIR [--msm-seed 0.24]

Then:
    TITAN_VAL_TRIALS=$(cat DIR/trials.txt) TITAN_VAL_TRIAL_DIR=DIR \
    TITAN_VAL_OUT=DIR/out TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=40 \
    sbatch --cpus-per-task=16 --time=04:00:00 --export=ALL run_validate_candidates.slurm

Pass list-valued variables through the shell environment and --export=ALL:
sbatch splits the value of --export=...,VAR=a,b on the commas (2026-09-30).
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CALIB_YAML = ROOT / "titan" / "calibration" / "calibration_targets.yml"

SEX_TYPES_WITH_MSM_BACKGROUND = ("MSM", "MSMW")
RACES = ("white_nonhispanic", "white_hispanic", "black_nonhispanic",
         "black_hispanic", "other_nonhispanic", "other_hispanic")


def get_path(d, path):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"{path}: '{key}' not found in the scenario")
        cur = cur[key]
    return cur


def set_path(d, path, value):
    keys = path.split(".")
    cur = d
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"{path}: '{key}' not found in the scenario")
        cur = cur[key]
    if keys[-1] not in cur:
        raise KeyError(f"{path}: '{keys[-1]}' not found in the scenario")
    cur[keys[-1]] = value


def fatal_frac_for(fatal_rate_annual):
    """derived_fatal_frac's formula with the fatal-rate target replaced."""
    calib = yaml.safe_load(open(CALIB_YAML))["calibration"]
    od = calib["overdose"]
    share = float(calib["injecting"]["injecting_pct"])
    nonfatal = (share * float(od["nonfatal_rate_inj_annual"])
                + (1.0 - share) * float(od["nonfatal_rate_noninj_annual"]))
    return fatal_rate_annual / (fatal_rate_annual + nonfatal)


def variants(base, msm_seed=None):
    """Yield (tag, scenario) pairs; every scenario is a deep copy of base."""
    yield "sens_base", copy.deepcopy(base)

    single = [
        ("oud_state.transitions.inject_stop_rate_per_year", "stop", [0.041, 0.326]),
        ("moud.overdose_scale.active", "odscale", [0.21, 0.41]),
        ("moud.discontinue_per_month", "discont", [0.069, 0.124]),
        ("overdose.post_moud_stop_rr", "poststop", [2.4, 3.3]),
        ("overdose.high_risk_mult", "hirisk", [1.2, 2.5]),
    ]
    for path, short, values in single:
        get_path(base, path)  # fail loudly on a renamed key
        for v in values:
            s = copy.deepcopy(base)
            set_path(s, path, v)
            yield f"sens_{short}{v}", s

    for fatal in (0.012, 0.027):
        s = copy.deepcopy(base)
        set_path(s, "overdose.fatal_frac", round(fatal_frac_for(fatal), 4))
        yield f"sens_fatal{fatal}", s

    for prev in (0.065, 0.25):
        s = copy.deepcopy(base)
        for st in SEX_TYPES_WITH_MSM_BACKGROUND:
            set_path(s, f"hiv.background_sex.by_sex_type.{st}.partner_hiv_prevalence", prev)
        yield f"sens_msmprev{prev}", s

    s = copy.deepcopy(base)
    for st in SEX_TYPES_WITH_MSM_BACKGROUND:
        set_path(s, f"hiv.background_sex.by_sex_type.{st}.rate_per_timestep", 4.0)
    yield "sens_msmrate4.0", s

    if msm_seed is not None:
        s = copy.deepcopy(base)
        for race in RACES:
            for st in SEX_TYPES_WITH_MSM_BACKGROUND:
                set_path(s, f"demographics.{race}.sex_type.{st}.drug_type.NonInj.hiv.init", msm_seed)
        yield f"sens_msmseed{msm_seed}", s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_yml", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--msm-seed", type=float, default=None,
                    help="NonInj MSM/MSMW hiv.init for the 'model's own value' axis")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.trial_yml))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tags = []
    for tag, scenario in variants(base, args.msm_seed):
        out = args.out_dir / f"trial_{tag}.yml"
        with open(out, "w") as fh:
            yaml.safe_dump(scenario, fh, default_flow_style=False, sort_keys=False)
        tags.append(tag)
        print(f"wrote {out}")
    (args.out_dir / "trials.txt").write_text(",".join(tags) + "\n")
    print(f"\n{len(tags)} scenarios from {args.trial_yml}; TITAN_VAL_TRIALS list in {args.out_dir / 'trials.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
