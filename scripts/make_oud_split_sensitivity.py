#!/usr/bin/env python
"""
Sensitivity of the fit and of the primary-care strategy effects to the OUD
active / remission split.

WHY. `oud_baseline` (0.55 active / 0.45 remission) is one of the four scored
targets and the only one that is an assumption: the cohort literature puts about
40 % of people with OUD in active use (38-50 %), and 0.55 was set for a
persistent-OUD cohort (docs/calibration_log/phase_4_oud_addendum_2026-07-20.md;
titan/calibration/target_provenance.yml, kind: assumption).

DESIGN. The canonical pick, three splits, no refit:

    label 55  active 0.55   the fit as it stands; a fresh control draw
    label 45  active 0.45   upper-middle of the literature range
    label 40  active 0.40   literature centre

For each split the targets are changed in memory (never on disk), so that the
same anchors the calibration applies follow the split: the initial OUD states
and the injecting population share (active x injecting_pct). The one fitted
parameter that sets the split is moved with it: the relapse rate is scaled by
the ratio of the odds, odds(split) / odds(0.55), which keeps the fitted model's
offset from its target; remission stays at its fitted value (it sits at the top
of its search box). Everything else is the fitted vector.

Each split then gets the status-quo baseline and the four primary-care arms
(scripts/make_phase7_pc_arms.py), 240 steps, written as
trial_<label>__<arm>.yml for titan.calibration.validate_candidates;
scripts/phase7_contrast.py --split-members contrasts each arm with the baseline
of its own split.

WHAT TO READ. (1) The lever is live: the active share lands near each split.
(2) Fit against split-specific targets: overdose and MOUD targets are per
active-OUD person and unchanged; statewide HIV diagnoses scale by
13,704 / (split x 500). (3) Strategy effects as percentage changes, by split.

Usage (YAML build; run on a compute node):
    python scripts/make_oud_split_sensitivity.py --out-dir <dir> [--steps 240]
"""
import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from titan.calibration.optuna_calibration import CALIB_YAML, load_yaml  # noqa: E402

SPLITS = {55: 0.55, 45: 0.45, 40: 0.40}
REFERENCE = 0.55


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def odds(p):
    return p / (1.0 - p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=REPO / "calibrated_scenario_best.yml")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    baseline_mod = _load("make_phase7_baseline")
    arms_mod = _load("make_phase7_pc_arms")
    calib = load_yaml(CALIB_YAML)
    calibrated = load_yaml(args.input)
    assert abs(float(calib["calibration"]["oud_baseline"]["oud_active"]) - REFERENCE) < 1e-9

    trials = args.out_dir / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    names, manifest = [], []
    for label, active in SPLITS.items():
        calib_alt = copy.deepcopy(calib)
        calib_alt["calibration"]["oud_baseline"] = {
            "oud_active": active, "oud_remission": round(1.0 - active, 10), "oud_none": 0.0}
        scen = copy.deepcopy(calibrated)
        tr = scen["oud_state"]["transitions"]
        relapse_fit = float(tr["relapse_rate_per_year"])
        tr["relapse_rate_per_year"] = relapse_fit * odds(active) / odds(REFERENCE)
        base = baseline_mod.build(scen, calib_alt, steps=args.steps)
        arms = {"scenario_1_baseline.yml": (base, ["baseline"])}
        arms.update(arms_mod.build_arms(base))
        for fname, (scenario, _notes) in arms.items():
            tag = f"{label}__{fname[:-4]}"
            with open(trials / f"trial_{tag}.yml", "w") as fh:
                yaml.safe_dump(scenario, fh, default_flow_style=False, sort_keys=False)
            names.append(tag)
        inj = {sx["drug_type"]["Inj"]["ppl"]
               for race in base["demographics"].values()
               for sx in race["sex_type"].values() if "Inj" in (sx or {}).get("drug_type", {})}
        manifest.append(
            f"{label}: target active {active:.2f}; relapse {relapse_fit:.4f} -> "
            f"{base['oud_state']['transitions']['relapse_rate_per_year']:.4f}; remission "
            f"{base['oud_state']['transitions']['remission_rate_per_year']:.4f}; two-state share "
            f"{base['oud_state']['transitions']['relapse_rate_per_year'] / (base['oud_state']['transitions']['relapse_rate_per_year'] + base['oud_state']['transitions']['remission_rate_per_year']):.3f}; "
            f"init_active {base['oud_state']['init_active']}; Inj.ppl {sorted(inj)}; "
            f"statewide scale {13704 / (active * 500):.1f}; "
            f"lethality entries {sum(k.startswith('lethality_') for k in base['timeline_scaling']['timeline'])}")
    (args.out_dir / "trials.txt").write_text("\n".join(names) + "\n")
    (args.out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print("\n".join(manifest))
    print(f"{len(names)} scenarios in {trials}")


if __name__ == "__main__":
    main()
