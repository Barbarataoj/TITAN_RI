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

THE POPULATION SEED IS PART OF THE CONTRAST DESIGN. Added 2026-08-17.

A Phase 7 result is a DIFFERENCE between a baseline run and an intervention run,
so the population it is measured on matters twice over:

  * PAIRED. Baseline and intervention must run on the SAME population, or the
    contrast carries between-population variance it does not need. This is free
    variance reduction and `run_scenarios.slurm` already gets it right, by
    re-loading one saved population through `--poppath` for every scenario.
  * PLURAL. That is also its defect. One population means the estimated effect of
    every intervention is conditioned on a single network realization, exactly as
    the four refits were (completion plan section 2.8), and there is nothing in
    the output to say how much of the contrast is the intervention and how much is
    the draw. Job 4963250 measured a draw as worth about one gate-width on
    hiv_prev_drift_pwid.

The design Phase 7 needs is therefore paired ACROSS several populations: for each
population seed in `calibration.population.population_seeds`, run every scenario
on that population, then contrast within seed and report the spread across seeds.
`run_scenarios.slurm` currently arrays over scenarios alone and has to array over
(scenario x population). See the note at the top of that file.

`model.seed.ppl` here is derived from `calibration.population.population_seeds[0]`
rather than inherited from the input file, so it cannot silently disagree with the
calibration. It is the seed for the baseline's own run; the plural design is the
runner's job, not this script's.

TWO THINGS TO CHECK BEFORE PHASE 7 RUNS, neither of which this script can fix:

  * `results_RI_pop/pop/*_pop.tar.gz` -- the population `run_scenarios.slurm`
    loads -- was saved on 2026-07-24 by job 4277443 from the trial-137 lineage,
    which was invalidated on 2026-07-28. It must be regenerated from the refit's
    canonical pick, or every contrast runs on a population from a discarded fit.
  * The baseline's PARAMETER VALUES are only as current as the fit behind them.
    Regenerate after every refit.

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
    set_anchored_parameters,
    set_hiv_initial_conditions,
    set_oud_initial_conditions,
    population_seeds,
    set_per_act_transmission,
    set_population_seed,
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


SCENARIO_SOURCE = REPO / "titan" / "calibration" / "scenario_source.yml"


def overlay_care_layer(scenario):
    """
    Carry the care layer's non-fitted scalars from today's scenario source
    into a calibrated pick that predates them (2026-10-02): PrEP monitoring
    (`prep.monitoring_*`) and any care_seeking setting field the pick lacks
    (reach_*, acceptance_mult_*, pool_refresh_time, visit_prob_after_overdose).
    Fitted values and fields the pick already carries are left as they are.
    """
    full = load_yaml(SCENARIO_SOURCE)
    src = full.get("scenario", {})
    prep_src = src.get("prep", {}) or {}
    prep = scenario.setdefault("prep", {})
    for k, v in prep_src.items():
        if k.startswith("monitoring_") or k.startswith("discontinue_late_") or k in ("use_products", "products"):
            prep[k] = v
    # PrEP adherence (sex-type constants) and discontinuation (sex-type constant
    # with a drug-type override), data decisions of 2026-10-02 that a calibrated
    # pick fitted before them does not carry
    levels = full.get("demographics", {}) or {}
    sex_prep = ((levels.get("sex_type", {}) or {}).get("constants", {}) or {}).get("prep", {}) or {}
    drug_var = [v for v in ((levels.get("drug_type", {}) or {}).get("varying", []) or [])
                if str(v.get("path", "")).startswith("prep.")]
    for race_block in (scenario.get("demographics", {}) or {}).values():
        for sex_block in (race_block.get("sex_type", {}) or {}).values():
            sp = sex_block.setdefault("prep", {})
            for k in ("adherence", "discontinue"):
                if k in sex_prep:
                    sp[k] = sex_prep[k]
            for drug, drug_block in (sex_block.get("drug_type", {}) or {}).items():
                for entry in drug_var:
                    field = entry["path"].split(".", 1)[1]
                    node = entry["values"]
                    for axis in entry["by"]:
                        node = node.get(drug) if axis == "drug" else node
                    if node is None or node == "__absent__":
                        continue
                    drug_block.setdefault("prep", {})[field] = node
    settings_src = (src.get("care_seeking", {}) or {}).get("setting", {}) or {}
    settings = scenario.setdefault("care_seeking", {}).setdefault("setting", {})
    for name, fields in settings_src.items():
        target = settings.setdefault(name, {})
        for k, v in fields.items():
            if k not in target:
                target[k] = v
    return scenario


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
    # The data-anchored table in calibration_targets.yml (`anchored_parameters`:
    # testing base rate, the post-overdose multiplier, ...). The refit applies it
    # after the suggest loop; the baseline has to apply it too, or a value that
    # changed after the pick was written -- moud.post_overdose_rr 1.0 -> 0.42 on
    # 2026-10-01 -- would be inherited by every arm from a stale file.
    set_anchored_parameters(scenario, calib)
    overlay_care_layer(scenario)

    # Derived, not inherited: the input file carries whatever seed the calibration
    # happened to write, and a Phase 7 baseline that disagrees with the
    # calibration's sampling design is a contrast measured on a different
    # population than the fit. See the docstring -- the plural design is the
    # runner's job, but the single source of truth is here.
    set_population_seed(scenario, population_seeds(calib)[0])

    if steps is not None:
        scenario["model"]["time"]["num_steps"] = int(steps)

    return scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the baseline here instead of scenarios/scenario_1_baseline.yml "
                         "(ensemble members, 2026-10-01); --check still compares the default path")
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
    out_path = args.output or OUTPUT
    write_yaml(out_path, scenario)
    print(f"wrote {out_path} from {args.input}")
    print(f"  population:   {got['sex_type_ppl']}")
    print(f"  drug_type:    {got['drug_type_ppl']}")
    print(f"  time:         {scenario['model']['time']}")
    print("Remember: re-run this after every refit, then rebuild scenarios 2-6 from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
