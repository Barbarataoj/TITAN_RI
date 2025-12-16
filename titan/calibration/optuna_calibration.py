
# titan/calibration/optuna_calibration_multi.py

import optuna
import yaml
import numpy as np
import subprocess
import copy
from pathlib import Path

CALIB_YAML = "titan/calibration/calibration_targets.yml"
BASE_SCENARIO = "tests/params/new_race_ri.yml"
OUTPUT_DIR = "optuna_multi_runs"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utility functions
# -----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def run_titan(params_file, output_dir):
    """Run TITAN and return parsed basic report."""
    cmd = ["python", "-m", "titan.run_titan", "-p", params_file, "-o", output_dir]
    subprocess.run(cmd, check=True)

    report = Path(output_dir) / "basicReport.txt"
    if not report.exists():
        raise RuntimeError(f"basicReport missing – TITAN run failed: {output_dir}")

    return parse_basic_report(report)


def parse_basic_report(path):
    """Extract percentages needed for calibration."""
    stats = {}

    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                stats[k.strip()] = float(v.strip())
            except:
                pass

    total = stats.get("num agents", None)

    # Normalize OUD categories
    if total:
        for k in ["oud_active", "oud_remission", "oud_none"]:
            if k in stats:
                stats[k] /= total

    # HIV prevalence
    if "hiv_prev" in stats and total:
        stats["hiv_prevalence"] = stats["hiv_prev"] / total

    # HIV incidence per 100 person-years
    if "hiv_incidence" in stats:
        stats["hiv_incidence"] = stats["hiv_incidence"]

    # Overdose counts → annualized rates
    if "fatal_overdose" in stats and total:
        stats["fatal_od_rate"] = stats["fatal_overdose"] / total

    if "nonfatal_overdose" in stats and total:
        stats["nonfatal_od_rate"] = stats["nonfatal_overdose"] / total

    # MOUD uptake percentage
    if "moud_current" in stats and total:
        stats["moud_uptake"] = stats["moud_current"] / total

    return stats


def set_yaml_value(scenario, path, val):
    """Set nested YAML value a.b.c = val"""
    keys = path.split(".")
    ref = scenario
    for k in keys[:-1]:
        ref = ref[k]
    ref[keys[-1]] = float(val)


# -----------------------------------------------
# Multi-objective error function
# Returns 5 separate objective values
# -----------------------------------------------
def multi_error(stats, targets):
    """
    Returns these independent objective values:
    1. OUD RMSE
    2. HIV prevalence SE
    3. HIV incidence SE
    4. Overdose SE (fatal + nonfatal)
    5. MOUD uptake SE
    """

    # 1. OUD RMSE
    oud_err = 0
    for k in ["oud_active", "oud_remission", "oud_none"]:
        if k in stats:
            oud_err += (stats[k] - targets["oud"][k]) ** 2
    oud_rmse = np.sqrt(oud_err / 3)

    # 2. HIV prevalence
    hiv_prev_err = (stats.get("hiv_prevalence", 0) - targets["hiv_prev"]) ** 2

    # 3. HIV incidence
    hiv_inc_err = (stats.get("hiv_incidence", 0) - targets["hiv_inc"]) ** 2

    # 4. Overdose
    fatal_err = (stats.get("fatal_od_rate", 0) - targets["od_fatal"]) ** 2
    nonfatal_err = (stats.get("nonfatal_od_rate", 0) - targets["od_nonfatal"]) ** 2
    od_combined = fatal_err + nonfatal_err

    # 5. MOUD uptake
    moud_err = (stats.get("moud_uptake", 0) - targets["moud"]) ** 2

    return oud_rmse, hiv_prev_err, hiv_inc_err, od_combined, moud_err


# -----------------------------------------------
# The Optuna objective (NSGA-II multi-objective)
# -----------------------------------------------
def objective(trial):
    calib = load_yaml(CALIB_YAML)
    base = load_yaml(BASE_SCENARIO)

    params_to_opt = calib["calibration"]["parameters_to_optimize"]

    targets = {
        "oud": {
            "oud_active": calib["calibration"]["oud_baseline"]["active_pct"],
            "oud_remission": calib["calibration"]["oud_baseline"]["remission_pct"],
            "oud_none": calib["calibration"]["oud_baseline"]["none_pct"],
        },
        "hiv_prev": calib["calibration"]["hiv"]["baseline_prevalence_pct"],
        "hiv_inc": calib["calibration"]["hiv"]["incidence_per_100py"],
        "od_fatal": calib["calibration"]["overdose"]["fatal_rate_annual"],
        "od_nonfatal": calib["calibration"]["overdose"]["nonfatal_rate_annual"],
        "moud": calib["calibration"]["moud"]["uptake_pct"],
    }

    scenario = copy.deepcopy(base)

    # Suggest values
    for p in params_to_opt:
        lo, hi = p["lower"], p["upper"]
        val = trial.suggest_float(p["path"], lo, hi)
        set_yaml_value(scenario, p["path"], val)

    # prepare trial run directory
    temp_yaml = f"{OUTPUT_DIR}/temp_trial_{trial.number}.yml"
    write_yaml(temp_yaml, scenario)

    outdir = f"{OUTPUT_DIR}/trial_{trial.number}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    stats = run_titan(temp_yaml, outdir)

    return multi_error(stats, targets)


# -----------------------------------------------
# Main optimizer
# -----------------------------------------------

import multiprocessing

def main():

    n_workers = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core free
    print(f"Running NSGA-II with {n_workers} parallel workers")

    study = optuna.create_study(
        directions=["minimize"] * 5,
        sampler=optuna.samplers.NSGAIISampler()
    )

    # Multi-processing parallel execution:
    with multiprocessing.Pool(n_workers) as pool:
        study.optimize(
            objective,
            n_trials=120,
            n_jobs=n_workers,   # <---- parallel here
            show_progress_bar=True
        )

    print("\n=== PARETO FRONT ===")
    for t in study.best_trials:
        print("\nTRIAL", t.number)
        print("Objectives:", t.values)
        print("Params:", t.params)

    # Save calibrated YAML from best Pareto point
    best = study.best_trials[0]
    calibrated = load_yaml(BASE_SCENARIO)
    for k, v in best.params.items():
        set_yaml_value(calibrated, k, v)

    write_yaml("calibrated_scenario_multi_5obj.yml", calibrated)
    print("\nSaved calibrated_scenario_multi_5obj.yml")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    main()


