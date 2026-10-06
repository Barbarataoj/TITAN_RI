"""
Guards for the working rules (docs/WORKING_RULES.md, 2026-10-06).

Each test encodes a mistake that was made once and found late:
- a scored target resting on a superseded data extract with nothing confirming it;
- a script carrying its own copy of the objective gates, which went stale;
- an analysis reading a data file the folder's README marks "do not use".
"""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from titan.calibration.optuna_calibration import CALIB_YAML, OBJECTIVE_GATES, load_yaml

REPO = Path(__file__).resolve().parents[1]
PROVENANCE = REPO / "titan" / "calibration" / "target_provenance.yml"

# Every quantity the objectives score or scale by. Adding a target to
# calibration_targets.yml means adding it here and to the provenance file.
REQUIRED = [
    "oud_baseline",
    "overdose.nonfatal_rate_inj_annual",
    "overdose.nonfatal_rate_noninj_annual",
    "overdose.fatal_rate_annual",
    "overdose.fatal_rate_annual_t3",
    "overdose.lethality_index",
    "moud.uptake_series_per_yr",
    "injecting.injecting_pct",
    "hiv.ridoh_idu_dx",
    "hiv.ridoh_pwid_dx_all_routes",
    "hiv.population_scale.ri_active_oud",
]


def _get(d, dotted):
    for key in dotted.split("."):
        d = d[key]
    return d


@pytest.mark.unit
def test_every_scored_target_has_provenance():
    prov = yaml.safe_load(PROVENANCE.read_text())
    calib = load_yaml(CALIB_YAML)["calibration"]
    for name in REQUIRED:
        _get(calib, name)  # the target itself still exists under this name
        entry = prov["targets"].get(name)
        assert entry, f"{name}: no entry in target_provenance.yml"
        assert str(entry.get("source", "")).strip(), f"{name}: no source"
        assert entry.get("checked"), f"{name}: no date checked"
        assert entry.get("kind") in ("data", "derived", "assumption"), f"{name}: kind"


@pytest.mark.unit
def test_superseded_data_needs_independent_confirmation():
    prov = yaml.safe_load(PROVENANCE.read_text())
    superseded = set(prov["superseded_vintages"])
    for name, entry in prov["targets"].items():
        if entry.get("data_vintage") in superseded:
            assert str(entry.get("confirmed_by", "")).strip(), (
                f"{name} rests on {entry['data_vintage']} and nothing confirms it: "
                "re-derive it on current data or cite an independent source"
            )


@pytest.mark.unit
def test_anchored_parameters_state_their_source():
    calib = load_yaml(CALIB_YAML)["calibration"]
    for entry in calib.get("anchored_parameters") or []:
        assert str(entry.get("source", "")).strip(), f"{entry['path']}: anchored without a source"


@pytest.mark.unit
def test_gates_have_one_owner():
    spec = importlib.util.spec_from_file_location("jf", REPO / "scripts" / "journal_feasible.py")
    jf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jf)
    assert tuple(jf.GATES) == tuple(OBJECTIVE_GATES), (
        "scripts/journal_feasible.py carries gates that differ from OBJECTIVE_GATES"
    )


@pytest.mark.unit
def test_no_code_reads_a_stale_data_file():
    prov = yaml.safe_load(PROVENANCE.read_text())
    reads = re.compile(r"open\(|read_csv|read_excel|ZipFile|load_workbook|Path\(|ExcelFile")
    offenders = []
    for folder in ("titan", "scripts", "analysis"):
        for path in (REPO / folder).rglob("*.py"):
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for stale in prov["stale_files"]:
                    if stale in line and reads.search(line) and not line.lstrip().startswith("#"):
                        offenders.append(f"{path.relative_to(REPO)}:{n}")
    assert not offenders, f"code reads a file marked stale: {offenders}"
