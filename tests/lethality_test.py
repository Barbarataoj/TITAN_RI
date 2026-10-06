"""
The drug-supply lethality index (titan/calibration/lethality.py, 2026-10-06).

The index must raise deaths by k(y) while leaving nonfatal overdoses where they
are, carry the projection the targets file names, and be safe to apply twice.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

from titan.calibration import lethality as L
from titan.calibration.optuna_calibration import CALIB_YAML, load_yaml

REPO = Path(__file__).resolve().parents[1]
F0 = 0.2007


def _scalars(entries, year):
    return {name.rsplit("_", 1)[1]: e["scalar"] for name, e in entries.items()
            if name.startswith(f"{L.PREFIX}{year}_")}


@pytest.mark.unit
@pytest.mark.parametrize("projection", ["hold2024", "hold2024_all", "peak", "pre2020"])
def test_deaths_scale_by_k_and_nonfatal_by_one(projection):
    entries = L.lethality_entries(F0, projection)
    for year, k in L.k_path(projection).items():
        if abs(k - 1.0) < 1e-12:
            assert not _scalars(entries, year)
            continue
        s = _scalars(entries, year)
        assert s["bpi"] == s["bpn"]
        deaths = s["bpi"] * s["ff"]
        nonfatal = s["bpi"] * (1 - F0 * s["ff"]) / (1 - F0)
        assert deaths == pytest.approx(k, rel=1e-6)
        assert nonfatal == pytest.approx(1.0, rel=1e-6)


@pytest.mark.unit
def test_windows_are_contiguous_and_last_is_held():
    entries = L.lethality_entries(F0, "hold2024")
    ff = sorted((e["start_time"], e["stop_time"]) for n, e in entries.items() if n.endswith("_ff"))
    assert ff[0][0] == L.start_step(2020) == 97
    for (a, b), (c, _) in zip(ff, ff[1:]):
        assert b == c
    assert ff[-1] == (L.start_step(2024), 10000)


@pytest.mark.unit
def test_pre2020_returns_to_one_in_2023():
    entries = L.lethality_entries(F0, "pre2020")
    ff = [e for n, e in entries.items() if n.endswith("_ff")]
    assert max(e["stop_time"] for e in ff) == L.start_step(2023)


@pytest.mark.unit
def test_apply_is_idempotent_and_none_removes():
    scen = {"overdose": {"fatal_frac": F0}, "features": {},
            "timeline_scaling": {"timeline": {"moud_acceptance_ratio_0": {"parameter": "moud|acceptance",
                                                                          "start_time": 49, "stop_time": 10000,
                                                                          "scalar": 1.1}}}}
    once = L.apply_lethality(copy.deepcopy(scen), "hold2024")
    twice = L.apply_lethality(copy.deepcopy(once), "hold2024")
    assert once == twice
    assert len([k for k in once["timeline_scaling"]["timeline"] if k.startswith(L.PREFIX)]) == 15
    peak = L.apply_lethality(copy.deepcopy(once), "peak")
    assert len([k for k in peak["timeline_scaling"]["timeline"] if k.startswith(L.PREFIX)]) == 12
    none = L.apply_lethality(copy.deepcopy(once), "none")
    assert list(none["timeline_scaling"]["timeline"]) == ["moud_acceptance_ratio_0"]


@pytest.mark.unit
def test_projection_from_calib():
    assert L.projection_from_calib({}) == "none"
    assert L.projection_from_calib({"calibration": {"overdose": {"lethality_index": "peak"}}}) == "peak"
    with pytest.raises(ValueError):
        L.projection_from_calib({"calibration": {"overdose": {"lethality_index": "bogus"}}})
    assert L.projection_from_calib(load_yaml(CALIB_YAML)) == "hold2024"


@pytest.mark.unit
def test_phase7_baseline_carries_the_index():
    spec = importlib.util.spec_from_file_location("mpb", REPO / "scripts" / "make_phase7_baseline.py")
    mpb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mpb)
    calib = load_yaml(CALIB_YAML)
    calibrated = load_yaml(REPO / "calibrated_scenario_best.yml")
    base = mpb.build(calibrated, calib, steps=240)
    tl = base["timeline_scaling"]["timeline"]
    assert len([k for k in tl if k.startswith(L.PREFIX)]) == 15
    assert any(k.startswith("moud_acceptance_ratio_") for k in tl)  # the fitted trend survives
    off = mpb.build(calibrated, calib, steps=240, lethality_projection="none")
    assert not [k for k in off["timeline_scaling"]["timeline"] if k.startswith(L.PREFIX)]
