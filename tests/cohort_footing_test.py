"""The cohort footing has one owner (calibration_targets.yml `calibration.cohort_footing`) and the scorer's
denominators follow it: scale = 13,704 / population size, MOUD coverage over all agents, overdose rates per
agent-year, fatal fraction with the injecting share taken per cohort member (work record §19.31, 2026-10-23)."""
import copy
import os

import pytest

from titan.calibration import optuna_calibration as oc


@pytest.fixture
def calib():
    return oc.load_yaml(oc.CALIB_YAML)["calibration"]


def test_default_footing_is_all(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    assert calib["cohort_footing"] == "all"
    assert oc.cohort_footing() == "all"


def test_scale_is_cohort_over_population(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    for n in (500, 2000, 13704):
        c = copy.deepcopy(calib); c["population"]["size"] = n
        assert oc.build_targets(c)["pop_scale"] == pytest.approx(13704 / n, rel=1e-9)
    monkeypatch.setenv("TITAN_COHORT_FOOTING", "active")
    c = copy.deepcopy(calib); c["population"]["size"] = 500
    assert oc.build_targets(c)["pop_scale"] == pytest.approx(13704 / (0.55 * 500), rel=1e-6)


def test_fatal_fraction_uses_cohort_injecting_share(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    od, inj, act = calib["overdose"], calib["injecting"]["injecting_pct"], calib["oud_baseline"]["oud_active"]
    share = inj * act
    expect = od["fatal_rate_annual"] / (od["fatal_rate_annual"] + share * od["nonfatal_rate_inj_annual"] + (1 - share) * od["nonfatal_rate_noninj_annual"])
    assert oc.derived_fatal_frac({"calibration": calib}) == pytest.approx(expect, rel=1e-9)


def test_invalid_footing_rejected(monkeypatch):
    monkeypatch.setenv("TITAN_COHORT_FOOTING", "half")
    with pytest.raises(ValueError):
        oc.cohort_footing()
