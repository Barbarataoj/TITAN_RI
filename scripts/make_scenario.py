#!/usr/bin/env python
"""
Generate the calibration scenario YAML from a compact source file.

WHY THIS EXISTS
---------------
`calibrated_scenario_multi_5obj.yml` is 2,774 leaf values, 95% of them in
`demographics`. Measured against the materialized `params.yml`, 111 of its 124
demographic leaf paths -- 3,246 of 3,828 values -- vary along nothing at all: they
are one Rhode Island number written once per demographic cell, 48 times over.

Storing one value in 48 places is how values drift apart, and it has cost this
project real time:

  * `set_population_window()` was silently inert for two days (2026-08-13) and
    invalidated a completed job, because a value living in 48 places cannot be
    checked by eye.
  * `model.seed.ppl: 12345` sat unread across four refits, so every trial of every
    refit was fitted against a single network realization.
  * `demographics.*.drug_type.Inj.ppl` and `calibration.injecting.injecting_pct`
    are the same quantity stored twice and had drifted apart by 2026-08-13.

`CLAUDE.md` already states the rule -- "values that pair must be derived from each
other rather than stored twice... both pairs have drifted apart in the past and
cost a refit each time." This script makes that rule enforceable for demographics
by giving every value exactly one home.

The generated file is a fully materialized scenario, identical in shape to the
hand-maintained one, so nothing downstream changes. What changes is that a human
never edits it again.

This follows `scripts/make_phase7_baseline.py`, which did the same for the Phase 7
baseline on 2026-08-14.

NOT A THIN OVERRIDE FILE
------------------------
An earlier attempt deleted the 111 constant paths on the theory that paraml would
fill them from `titan/params/*.yml`. It will not: those values are RI overrides
that happen to be equal in every cell, not upstream defaults
(`haart.enroll.enroll_0.prob` 0.802 against an upstream 0.0, `hiv.dx.init` 0.932
against 0.0, `injection.unsafe_prob` 0.38 against 0.0, `num_partners.*.dist_type`
`negative_binomial` against `set_value`). Materializing the thin file produced
1,704 differing values.

Nor is a thin form possible from the parameter side at all. `create_params` merges
parameter paths with a recursive merge over *literal dict keys*, and only then does
`parse_params` expand `sub-dict` blocks over `pops`. The slot where one value can
cover all cells exists only in the definitions, not in any parameter or settings
layer. So the values must be written out -- the fix is to write them once here and
expand them mechanically.

USAGE
-----
    python scripts/make_scenario.py                     # regenerate the scenario
    python scripts/make_scenario.py --check             # fail if it is stale
    python scripts/make_scenario.py --extract-from X    # rebuild the source from X

`--check` is the guard against a hand-edit: it regenerates into memory and diffs
against the committed artifact.

`--extract-from` is the migration tool. It decomposes an existing scenario into the
source form by finding, for each demographic leaf path, the smallest set of axes on
which its value is a well-defined function. It was used once to create
`scenario_source.yml` and is kept so the derivation stays reproducible.

VERIFICATION
------------
The acceptance test is not that the two YAML files match textually -- key order and
formatting are free. It is that the *materialized* parameters match:

    python scripts/make_scenario.py --check

compares the generated scenario to the committed one leaf by leaf. For the
stronger check, materialize both through `titan.parse_params.create_params` and
diff the resulting `params.yml`; if those agree the engine cannot tell the two
files apart. That is the standard the 2026-08-10 injection-recency change was held
to.
"""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "titan" / "calibration" / "scenario_source.yml"
DEFAULT_OUT = REPO_ROOT / "calibrated_scenario_multi_5obj.yml"

# The three demographic axes, in the order they nest in the scenario. Level names
# are the keys the scenario uses; axis names are what `by:` refers to in the
# source.
LEVELS = [
    ("race", ("race",)),
    ("sex_type", ("race", "sex")),
    ("drug_type", ("race", "sex", "drug")),
]


# ---------------------------------------------------------------- helpers


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=True, default_flow_style=False)


def flatten(obj, prefix=()):
    """Nested dict -> {path tuple: leaf}. Lists are leaves."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, prefix + (k,)))
        return out
    return {prefix: obj}


def unflatten(items):
    """{path tuple: leaf} -> nested dict."""
    out = {}
    for path, val in items.items():
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = val
    return out


# Sentinel for "this path does not exist in this cell".
#
# The demographic cross-product is ragged, and the raggedness is meaningful: a
# NonInj cell carries only the `Sex` bond under `num_partners`, while an Inj cell
# also carries `Inj` and `SexInj`. A non-injector has no injection partnerships,
# so writing those blocks into NonInj cells would not be a formatting change --
# it would add 240 parameters the model does not currently have.
#
# Presence is therefore treated as part of a path's value, so `minimal_axes` finds
# the axis that governs it (here, drug type) exactly as it does for any other
# varying value. The round-trip check caught this; without the sentinel the
# extractor read those paths as constants and expanded them everywhere.
MISSING = "__absent__"


def dotted(path):
    return ".".join(str(p) for p in path)


def undotted(s):
    """Dotted path back to a tuple, restoring integer keys.

    Bin blocks are keyed by integer (`num_partners.Sex.vars.1.value`), and YAML
    would otherwise round-trip them as strings, which paraml's `get_bins` rejects.
    """
    out = []
    for part in s.split("."):
        try:
            out.append(int(part))
        except ValueError:
            out.append(part)
    return tuple(out)


# ---------------------------------------------------------------- expansion


def expand_level(spec, axes, keys):
    """Build one demographic level's dict for a given (race, sex, drug) key tuple.

    `spec` is a source level: {"constants": {...}, "varying": [{path, by, values}]}.
    `keys` maps axis name -> value for the cell being built.
    """
    leaves = dict(flatten(spec.get("constants") or {}))

    for entry in spec.get("varying") or []:
        node = entry["values"]
        for axis in entry["by"]:
            node = node[keys[axis]]
        if node == MISSING:
            continue
        leaves[undotted(entry["path"])] = node

    return unflatten(leaves)


def build_demographics(source):
    axes = source["axes"]
    races, sexes, drugs = axes["races"], axes["sex_types"], axes["drug_types"]
    levels = source["demographics"]

    out = {}
    for race in races:
        race_block = expand_level(levels["race"], axes, {"race": race})
        race_block["sex_type"] = {}
        for sex in sexes:
            sex_block = expand_level(
                levels["sex_type"], axes, {"race": race, "sex": sex}
            )
            sex_block["drug_type"] = {}
            for drug in drugs:
                sex_block["drug_type"][drug] = expand_level(
                    levels["drug_type"], axes,
                    {"race": race, "sex": sex, "drug": drug},
                )
            race_block["sex_type"][sex] = sex_block
        out[race] = race_block
    return out


def build_scenario(source):
    """Source -> full scenario dict."""
    scenario = copy.deepcopy(source["scenario"])
    scenario["demographics"] = build_demographics(source)

    # The sampling block is stated once here and written into `model`, rather than
    # being three unrelated literals buried at different depths of a 4,300-line
    # file. `model.seed.ppl` in particular was invisible for four refits; see
    # docs/calibration_log/phase_6_completion_plan_2026-08-14.md section 2.8.
    sampling = source["sampling"]
    model = scenario.setdefault("model", {})
    model["num_pop"] = sampling["population_size"]
    model.setdefault("time", {})["burn_steps"] = sampling["burn_steps"]
    model["time"]["num_steps"] = sampling["scored_steps"]
    model.setdefault("seed", {})["ppl"] = sampling["population_seed"]

    return scenario


# ---------------------------------------------------------------- extraction


def minimal_axes(observations, axis_names):
    """Smallest subset of axes on which the value is a well-defined function.

    `observations` maps a full key tuple (aligned with `axis_names`) to a value.
    Returns the axis-name tuple, smallest first. A path whose value never changes
    returns the empty tuple and is therefore a constant.
    """
    for size in range(len(axis_names) + 1):
        for subset in itertools.combinations(range(len(axis_names)), size):
            seen = {}
            consistent = True
            for key, value in observations.items():
                reduced = tuple(key[i] for i in subset)
                encoded = json.dumps(value, sort_keys=True)
                if seen.get(reduced, encoded) != encoded:
                    consistent = False
                    break
                seen[reduced] = encoded
            if consistent:
                return tuple(axis_names[i] for i in subset)
    raise AssertionError("no axis subset explains this path")  # unreachable


def nest_values(observations, axis_names, by):
    """{full key: value} -> nested dict keyed by the `by` axes, in order."""
    idx = [axis_names.index(a) for a in by]
    out = {}
    for key, value in observations.items():
        node = out
        for i in idx[:-1]:
            node = node.setdefault(key[i], {})
        node[key[idx[-1]]] = value
    return out


def extract(scenario_path):
    scenario = load_yaml(scenario_path)
    dem = scenario["demographics"]

    races = list(dem)
    sexes = list(next(iter(dem.values()))["sex_type"])
    drugs = list(next(iter(next(iter(dem.values()))["sex_type"].values()))["drug_type"])

    # Observations per level: {leaf path: {key tuple: value}}
    obs = {name: {} for name, _ in LEVELS}
    cells = {name: [] for name, _ in LEVELS}
    for race, race_block in dem.items():
        cells["race"].append((race,))
        above = {k: v for k, v in race_block.items() if k != "sex_type"}
        for path, val in flatten(above).items():
            obs["race"].setdefault(path, {})[(race,)] = val
        for sex, sex_block in race_block["sex_type"].items():
            cells["sex_type"].append((race, sex))
            above = {k: v for k, v in sex_block.items() if k != "drug_type"}
            for path, val in flatten(above).items():
                obs["sex_type"].setdefault(path, {})[(race, sex)] = val
            for drug, drug_block in sex_block["drug_type"].items():
                cells["drug_type"].append((race, sex, drug))
                for path, val in flatten(drug_block).items():
                    obs["drug_type"].setdefault(path, {})[(race, sex, drug)] = val

    # Fill the ragged cross-product before looking for axes, so that a path's
    # absence is itself an observation rather than a silently dropped cell.
    for name, _ in LEVELS:
        for observations in obs[name].values():
            for cell in cells[name]:
                observations.setdefault(cell, MISSING)

    levels = {}
    for name, axis_names in LEVELS:
        constants, varying = {}, []
        for path, observations in sorted(obs[name].items()):
            by = minimal_axes(observations, axis_names)
            if not by:
                constants[path] = next(iter(observations.values()))
            else:
                varying.append({
                    "path": dotted(path),
                    "by": list(by),
                    "values": nest_values(observations, axis_names, by),
                })
        levels[name] = {"constants": unflatten(constants), "varying": varying}

    rest = {k: v for k, v in scenario.items() if k != "demographics"}
    model = rest.get("model", {})
    sampling = {
        "population_size": model.get("num_pop"),
        "burn_steps": model.get("time", {}).get("burn_steps"),
        "scored_steps": model.get("time", {}).get("num_steps"),
        "population_seed": model.get("seed", {}).get("ppl"),
    }

    return {
        "axes": {"races": races, "sex_types": sexes, "drug_types": drugs},
        "sampling": sampling,
        "demographics": levels,
        "scenario": rest,
    }


# ---------------------------------------------------------------- diffing


def diff_leaves(a, b):
    """Leaf-level diff of two scenario dicts. Returns (only_a, only_b, changed)."""
    fa, fb = flatten(a), flatten(b)
    only_a = sorted(dotted(k) for k in set(fa) - set(fb))
    only_b = sorted(dotted(k) for k in set(fb) - set(fa))
    changed = sorted(
        (dotted(k), fa[k], fb[k]) for k in set(fa) & set(fb) if fa[k] != fb[k]
    )
    return only_a, only_b, changed


def report_diff(only_a, only_b, changed, label_a, label_b, limit=20):
    print(f"  only in {label_a}: {len(only_a)}")
    for k in only_a[:limit]:
        print(f"    - {k}")
    print(f"  only in {label_b}: {len(only_b)}")
    for k in only_b[:limit]:
        print(f"    + {k}")
    print(f"  differing values: {len(changed)}")
    for k, x, y in changed[:limit]:
        print(f"    ~ {k}: {x!r} -> {y!r}")
    total = len(only_a) + len(only_b) + len(changed)
    if total > limit:
        print(f"  ... {total} differences in all")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed scenario differs from the generated one")
    ap.add_argument("--extract-from", default=None,
                    help="rebuild the source file from an existing scenario")
    args = ap.parse_args()

    if args.extract_from:
        source = extract(args.extract_from)
        write_yaml(args.source, source)
        n_const = sum(len(flatten(lv["constants"]))
                      for lv in source["demographics"].values())
        n_vary = sum(len(lv["varying"]) for lv in source["demographics"].values())
        print(f"Extracted {args.extract_from}")
        print(f"  -> {args.source}")
        print(f"  {n_const} constant demographic paths, {n_vary} varying")
        # Round-trip immediately: an extraction that does not rebuild its input is
        # a silent data loss, and this is the only place it can be caught.
        rebuilt = build_scenario(source)
        original = load_yaml(args.extract_from)
        only_a, only_b, changed = diff_leaves(original, rebuilt)
        if only_a or only_b or changed:
            print("\nROUND-TRIP FAILED -- extraction is lossy:")
            report_diff(only_a, only_b, changed, "original", "rebuilt")
            return 1
        print("  round-trip verified: rebuilt scenario is leaf-identical")
        return 0

    source = load_yaml(args.source)
    scenario = build_scenario(source)

    if args.check:
        committed = load_yaml(args.out)
        only_a, only_b, changed = diff_leaves(committed, scenario)
        if only_a or only_b or changed:
            print(f"STALE: {args.out} differs from what {args.source} generates.")
            report_diff(only_a, only_b, changed, "committed", "generated")
            print("\nRegenerate with: python scripts/make_scenario.py")
            return 1
        print(f"OK: {args.out} matches {args.source} ({len(flatten(scenario))} leaves)")
        return 0

    write_yaml(args.out, scenario)
    print(f"Wrote {args.out} ({len(flatten(scenario))} leaves) from {args.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
