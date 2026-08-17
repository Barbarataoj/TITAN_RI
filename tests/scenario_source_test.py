"""
Regression tests for the generated calibration scenario.

`calibrated_scenario_multi_5obj.yml` stopped being hand-maintained on 2026-08-17.
It is generated from `titan/calibration/scenario_source.yml`, which states each
demographic value once on the axis it actually varies along, because 111 of the
124 demographic leaf paths held one Rhode Island number written 48 times over.
See docs/calibration_log/phase_6_scenario_format_2026-08-17.md.

Two things need guarding, and they fail in different ways:

  * The committed artifact going stale, because someone edited it by hand. The
    edit would survive until the next regeneration and then vanish silently.
  * The expansion becoming lossy, because a new demographic parameter varies along
    an axis the source cannot express. That fails loudly here rather than quietly
    in a calibration run.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_scenario.py"
SOURCE = REPO_ROOT / "titan" / "calibration" / "scenario_source.yml"
SCENARIO = REPO_ROOT / "calibrated_scenario_multi_5obj.yml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_scenario import (  # noqa: E402
    MISSING,
    build_scenario,
    diff_leaves,
    extract,
    flatten,
)


@pytest.fixture(scope="module")
def source():
    with open(SOURCE) as f:
        return yaml.safe_load(f)


@pytest.mark.unit
def test_committed_scenario_is_not_stale():
    """`--check` is the guard against a hand-edit, so it has to be exercised."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_extraction_round_trips(source):
    """Extract-then-expand must reproduce the scenario leaf for leaf.

    This is the property that makes the migration safe. If it holds, the source
    carries everything the scenario did and the generator is not inventing values.
    """
    rebuilt = build_scenario(extract(str(SCENARIO)))
    with open(SCENARIO) as f:
        committed = yaml.safe_load(f)

    only_committed, only_rebuilt, changed = diff_leaves(committed, rebuilt)
    assert not only_committed, f"expansion drops paths: {only_committed[:5]}"
    assert not only_rebuilt, f"expansion invents paths: {only_rebuilt[:5]}"
    assert not changed, f"expansion changes values: {changed[:5]}"


@pytest.mark.unit
def test_source_is_much_smaller_than_the_artifact(source):
    """The point of the exercise, asserted so a regression is visible.

    Not a style preference: every duplicated value is a place a value can drift
    from its siblings, which is how `demographics.*.drug_type.Inj.ppl` and
    `calibration.injecting.injecting_pct` came apart before 2026-08-13.
    """
    generated = build_scenario(source)
    assert len(flatten(generated)) > 2000
    n_stated = sum(
        len(flatten(level["constants"])) + len(level["varying"])
        for level in source["demographics"].values()
    )
    assert n_stated < 100, f"{n_stated} demographic paths stated; expected < 100"


@pytest.mark.unit
def test_noninj_cells_have_no_injection_partnerships(source):
    """The demographic cross-product is ragged, and the raggedness is meaningful.

    A NonInj agent carries only the `Sex` bond; `Inj` and `SexInj` belong to
    injectors alone. The first extractor read those paths as constants and wrote
    them into all 48 cells, which would have added 240 parameters the model does
    not have. The MISSING sentinel is what prevents that, so assert the outcome.
    """
    generated = build_scenario(source)
    cell = generated["demographics"]["white_nonhispanic"]["sex_type"]["HF"]
    assert set(cell["drug_type"]["Inj"]["num_partners"]) == {"Inj", "Sex", "SexInj"}
    assert set(cell["drug_type"]["NonInj"]["num_partners"]) == {"Sex"}


@pytest.mark.unit
def test_bin_blocks_carry_every_field(source):
    """paraml's bin defaults define only bin 1, and its value_type default is int.

    Omitting `value_type` on bin 1 does not raise -- it silently coerces partner
    counts to integers -- and omitting anything on bin 2 raises inside get_bins.
    Both were hit while prototyping. Every bin must be emitted in full.
    """
    generated = build_scenario(source)
    for race, race_block in generated["demographics"].items():
        for sex, sex_block in race_block["sex_type"].items():
            for drug, cell in sex_block["drug_type"].items():
                for bond, spec in cell.get("num_partners", {}).items():
                    for index, var in spec["vars"].items():
                        where = f"{race}.{sex}.{drug}.{bond}.vars.{index}"
                        assert "value" in var, f"{where} missing value"
                        assert "value_type" in var, f"{where} missing value_type"


@pytest.mark.unit
def test_sampling_block_reaches_the_model(source):
    """The sampling knobs are stated once in the source and written into `model`.

    `model.seed.ppl` sat unread at line 4311 of the old artifact across four
    refits, which is how every trial came to be fitted against one network
    realization. It is a named field now; keep it wired.
    """
    generated = build_scenario(source)
    sampling = source["sampling"]
    assert generated["model"]["num_pop"] == sampling["population_size"]
    assert generated["model"]["time"]["burn_steps"] == sampling["burn_steps"]
    assert generated["model"]["time"]["num_steps"] == sampling["scored_steps"]
    assert generated["model"]["seed"]["ppl"] == sampling["population_seed"]


@pytest.mark.unit
def test_varying_entries_cover_every_axis_combination(source):
    """A `values` tree with a hole would raise KeyError mid-expansion.

    Cheaper to catch here than in trial 400 of a refit.
    """
    axes = source["axes"]
    axis_values = {
        "race": axes["races"],
        "sex": axes["sex_types"],
        "drug": axes["drug_types"],
    }
    for level_name, level in source["demographics"].items():
        for entry in level["varying"]:
            node_paths = [([], entry["values"])]
            for axis in entry["by"]:
                nxt = []
                for keys, node in node_paths:
                    missing = set(axis_values[axis]) - set(node)
                    assert not missing, (
                        f"{level_name}.{entry['path']} missing {axis}={sorted(missing)}"
                    )
                    nxt.extend((keys + [k], node[k]) for k in axis_values[axis])
                node_paths = nxt
            for keys, leaf in node_paths:
                assert leaf is not None, f"{entry['path']} at {keys} is null"


@pytest.mark.unit
def test_missing_sentinel_is_not_a_plausible_value(source):
    """MISSING is a string sentinel, so it must not collide with a real value.

    Compared element-wise rather than through a set: `flatten` treats lists as
    leaves (`classes.sex_types.HF.sleeps_with`), and those are unhashable.
    """
    generated = build_scenario(source)
    for path, value in flatten(generated).items():
        assert value != MISSING, f"sentinel leaked into {path}"
        if isinstance(value, list):
            assert MISSING not in value, f"sentinel leaked into {path}"
