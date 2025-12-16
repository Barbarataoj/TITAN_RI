import yaml
import numpy as np
import subprocess
import copy
from pathlib import Path

CALIB_YAML = "titan/calibration/calibration_targets.yml"
BASE_SCENARIO = "tests/params/new_race_ri.yml"
OUTPUT_DIR = "calibration_runs"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def run_titan(params_file, output_dir):
    """Runs TITAN and returns parsed basic report."""
    cmd = ["python", "-m", "titan.run_titan", "-p", params_file, "-o", output_dir]
    subprocess.run(cmd, check=True)

    report = Path(output_dir) / "basicReport.txt"
    if not report.exists():
        raise RuntimeError(f"basicReport missing in {output_dir}")

    return parse_basic_report(report)


def parse_basic_report(path):
    """Extract percentages from basicReport."""
    stats = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            try:
                stats[key] = float(value.strip())
            except:
                pass

    # convert counts → percentages
    total = stats.get("num agents", None)
    if total:
        for k in ["oud_active", "oud_remission", "oud_none"]:
            if k in stats:
                stats[k] = stats[k] / total

    if "hiv_prev" in stats and total:
        stats["hiv_prevalence"] = stats["hiv_prev"] / total

    return stats


def error_function(targets, stats):
    """Squared-error objective."""
    err = 0.0
    for k, target_val in targets.items():
        if k not in stats:
            continue
        err += (stats[k] - target_val) ** 2
    return err


def set_yaml_value(scenario, path, val):
    """Set nested YAML value like a.b.c = val"""
    keys = path.split(".")
    ref = scenario
    for k in keys[:-1]:
        ref = ref[k]
    ref[keys[-1]] = float(val)


def calibrate():
    calib = load_yaml(CALIB_YAML)
    base = load_yaml(BASE_SCENARIO)

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    params_to_opt = calib["calibration"]["parameters_to_optimize"]

    targets = {
        "oud_active": calib["calibration"]["oud_baseline"]["active_pct"],
        "oud_remission": calib["calibration"]["oud_baseline"]["remission_pct"],
        "oud_none": calib["calibration"]["oud_baseline"]["none_pct"],
        "hiv_prevalence": calib["calibration"]["hiv"]["baseline_prevalence_pct"],
    }

    n_runs = calib["calibration"]["n_runs"]

    best = {}
    for p in params_to_opt:
        path = p["path"]
        low, high = p["lower"], p["upper"]
        test_vals = np.linspace(low, high, 7)

        best_err = 1e9
        best_val = None

        for val in test_vals:
            errs = []

            for r in range(n_runs):
                scenario = copy.deepcopy(base)
                set_yaml_value(scenario, path, val)

                temp_yaml = f"{OUTPUT_DIR}/temp_{Path(path).name}_{val:.4f}.yml"
                write_yaml(temp_yaml, scenario)

                outdir = f"{OUTPUT_DIR}/run_{Path(path).name}_{val:.4f}_r{r}"
                Path(outdir).mkdir(parents=True, exist_ok=True)

                stats = run_titan(temp_yaml, outdir)
                errs.append(error_function(targets, stats))

            avg_err = np.mean(errs)
            print(f"[{path}] val={val:.4f}  avg_error={avg_err:.6f}")

            if avg_err < best_err:
                best_err = avg_err
                best_val = val

        best[path] = best_val

    print("\n===== FINAL CALIBRATED PARAMETERS =====")
    for k, v in best.items():
        print(f"{k} = {v:.5f}")

    # write calibrated scenario
    final = copy.deepcopy(base)
    for p in params_to_opt:
        set_yaml_value(final, p["path"], best[p["path"]])

    write_yaml("calibrated_scenario.yml", final)
    print("\nSaved calibrated YAML → calibrated_scenario.yml")


if __name__ == "__main__":
    calibrate()
