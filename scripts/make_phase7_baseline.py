"""
Generate the Phase 7 baseline scenario from the calibration's canonical pick.

WHY THIS IS GENERATED RATHER THAN EDITED. `scenarios/scenario_1_baseline.yml` is
the file every intervention template tells you to copy, so any drift in it is
inherited by all five scenario contrasts at once. The version in the repo on
2026-08-13 had drifted about as far as it could:

  * sex_type shares HF 0.20 / HM 0.30 / MSM 0.30 / MTF 0.10 / WSW 0.10 -- the
    stock TITAN demographics, not Rhode Island. The calibration base runs
    HF 0.31 / HM 0.64 / MSM 0.03 / MSMW 0.02, so MSM was over-weighted TENFOLD
    in the group that carries most sexual transmission.
  * no MSMW stratum at all, and MTF/WSW strata the calibration does not have.
  * no `classes.sex_types` block, so nothing defined who partners with whom.
  * no `haart` or `incar` blocks.
  * `moud.post_overdose_rr` 4.976 and `partnership.injection.transmission.base`
    0.005498, both from the trial-137 lineage that was invalidated on
    2026-07-28.

Hand-patching that would have produced a file matching the calibration on the
fields someone happened to check. Generating it makes the guarantee structural:
the Phase 7 baseline IS the calibrated model, plus the scenario horizon, and
nothing else.

USAGE

    sbatch scripts/run_on_node.slurm python scripts/make_phase7_baseline.py

Reads `calibrated_scenario_best.yml` (the canonical pick written by
optuna_calibration.main) and writes `scenarios/scenario_1_baseline.yml`. Run it
after every refit -- the baseline is only as current as the fit behind it.

    --input PATH    take a different calibrated scenario
    --steps N       scenario horizon in months (default: the calibration's own)
    --check         compare only, write nothing, exit 1 on mismatch
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.calibration.optuna_calibration import (  # noqa: E402
    CALIB_YAML,
    load_yaml,
    set_hiv_initial_conditions,
    set_oud_initial_conditions,
    set_per_act_transmission,
    set_population_window,
    write_yaml,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "calibrated_scenario_best.yml"
OUTPUT = REPO / "scenarios" / "scenario_1_baseline.yml"

# Fields whose agreement with the calibration base is the whole point of this
# script. Checked after generation so a silent structural change fails loudly.
def composition(scenario):
    demo = scenario.get("demographics") or {}
    race = sorted(demo)[0] if demo else None
    sex_types = (demo.get(race) or {}).get("sex_type") or {}
    return {
        "sex_type_ppl": {k: v.get("ppl") for k, v in sorted(sex_types.items())},
        "has_classes_sex_types": "sex_types" in (scenario.get("classes") or {}),
        "drug_type_ppl": {
            k: (v or {}).get("ppl")
            for k, v in sorted(
                ((next(iter(sex_types.values())) or {}).get("drug_type") or {}).items()
            )
        },
        "oud_init": {
            k: v for k, v in (scenario.get("oud_state") or {}).items()
            if k.startswith("init")
        },
    }


def build(calibrated, calib, steps=None):
    """The calibrated model, re-anchored, at the scenario horizon."""
    scenario = copy.deepcopy(calibrated)

    # Re-apply every anchor rather than trusting the input file. The canonical
    # pick is written by main() with these applied, but this script also has to
    # be safe to point at an older scenario.
    set_hiv_initial_conditions(scenario)
    set_oud_initial_conditions(scenario, calib)
    set_per_act_transmission(scenario)
    set_population_window(scenario, calib)

    if steps is not None:
        scenario["model"]["time"]["num_steps"] = int(steps)

    return scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    calib = load_yaml(CALIB_YAML)
    calibrated = load_yaml(args.input)
    scenario = build(calibrated, calib, steps=args.steps)

    # Reference is the calibration base WITH the anchors applied in memory, not
    # the file as it sits on disk. The file lags whenever a target changes -- it
    # only catches up the next time something writes it -- and comparing against
    # a lagging file would report the generated baseline as wrong for being
    # right. The targets file is the authority.
    want = composition(build(load_yaml(REPO / "calibrated_scenario_multi_5obj.yml"), calib))
    got = composition(scenario)
    if want != got:
        print("generated baseline does not match the calibration base:")
        print(f"  calibration base: {want}")
        print(f"  generated:        {got}")
        return 1

    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} does not exist")
            return 1
        current = composition(load_yaml(OUTPUT))
        if current != want:
            print("scenarios/scenario_1_baseline.yml is STALE:")
            print(f"  calibration base: {want}")
            print(f"  baseline file:    {current}")
            return 1
        print("scenarios/scenario_1_baseline.yml matches the calibration base")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(OUTPUT, scenario)
    print(f"wrote {OUTPUT} from {args.input}")
    print(f"  population:   {got['sex_type_ppl']}")
    print(f"  drug_type:    {got['drug_type_ppl']}")
    print(f"  time:         {scenario['model']['time']}")
    print("Remember: re-run this after every refit, then rebuild scenarios 2-6 from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
