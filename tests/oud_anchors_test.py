"""
Regression tests for the OUD-side data anchors (step 2 of the 2026-08-12 plan).

Two failure modes are covered, both of which have real precedent in this repo.

The first is the pairing between `demographics.*.drug_type.Inj.ppl` and the
`injecting_pct` / `oud_active` targets. Inj.ppl is a POPULATION share that the
population builder rescales by 1 / (1 - init_remission), so the share of active
OUD agents who inject is Inj.ppl / oud_active. Those numbers lived in two files
with nothing checking them, and drifted: 0.25 against the 0.2475 the targets
imply. That is the same shape as the HIV seed / burn_steps drift.

The second is an anchored path finding its way back into parameters_to_optimize.
set_oud_initial_conditions runs after objective()'s suggest loop, so a path in
both places is overwritten after Optuna has recorded the suggested value -- the
study would report a fitted rate the simulation never used.
"""

import copy

import pytest

from titan.calibration.optuna_calibration import (
    ANCHORED_PATHS,
    INJECT_START_RATE,
    INJECT_STOP_RATE,
    assert_not_fitted,
    set_oud_initial_conditions,
)


def calib(active=0.55, remission=0.45, injecting_pct=0.45, fitted_paths=()):
    return {
        "calibration": {
            "oud_baseline": {
                "oud_active": active,
                "oud_remission": remission,
                "oud_none": 0.0,
            },
            "injecting": {"injecting_pct": injecting_pct},
            "parameters_to_optimize": [{"path": p} for p in fitted_paths],
        }
    }


def scenario():
    stratum = {
        "ppl": 0.5,
        "drug_type": {"Inj": {"ppl": 0.25}, "NonInj": {"ppl": 0.75}},
    }
    return {
        "oud_state": {
            "init_active": 0.9,
            "init_remission": 0.1,
            "init_none": 0.0,
            "transitions": {
                "inject_start_rate_per_year": 0.0735,
                "inject_stop_rate_per_year": 0.2320,
                "relapse_to_inj_prob": 0.5,
            },
        },
        "demographics": {
            "white_nonhispanic": {
                "sex_type": {"HM": copy.deepcopy(stratum), "HF": copy.deepcopy(stratum)}
            },
            "black_nonhispanic": {"sex_type": {"HM": copy.deepcopy(stratum)}},
        },
    }


def test_injection_share_is_derived_from_the_targets():
    sc = scenario()
    n, inj_ppl = set_oud_initial_conditions(sc, calib())

    assert n == 3
    assert inj_ppl == pytest.approx(0.55 * 0.45)
    for race in sc["demographics"].values():
        for sx in race["sex_type"].values():
            dt = sx["drug_type"]
            assert dt["Inj"]["ppl"] == pytest.approx(0.2475)
            assert dt["NonInj"]["ppl"] == pytest.approx(0.7525)
            # what the population builder realizes among active OUD agents
            assert dt["Inj"]["ppl"] / sc["oud_state"]["init_active"] == pytest.approx(0.45)


def test_injection_share_tracks_a_changed_target():
    """The point of deriving it: move the target, the initial condition follows."""
    sc = scenario()
    _, inj_ppl = set_oud_initial_conditions(sc, calib(active=0.60, remission=0.40,
                                                      injecting_pct=0.30))
    assert inj_ppl == pytest.approx(0.18)
    assert sc["oud_state"]["init_active"] == 0.60
    dt = sc["demographics"]["white_nonhispanic"]["sex_type"]["HM"]["drug_type"]
    assert dt["Inj"]["ppl"] / 0.60 == pytest.approx(0.30)


def test_baseline_split_overwrites_whatever_the_scenario_carried():
    sc = scenario()
    set_oud_initial_conditions(sc, calib())
    assert sc["oud_state"]["init_active"] == 0.55
    assert sc["oud_state"]["init_remission"] == 0.45
    assert sc["oud_state"]["init_none"] == 0.0


def test_injection_rates_are_pinned_and_relapse_to_inj_is_left_alone():
    sc = scenario()
    set_oud_initial_conditions(sc, calib())
    tr = sc["oud_state"]["transitions"]
    assert tr["inject_start_rate_per_year"] == INJECT_START_RATE
    assert tr["inject_stop_rate_per_year"] == INJECT_STOP_RATE
    # Not this function's to write: relapse_to_inj_prob is fixed by
    # set_anchored_parameters() (since 2026-09-29), and the OUD anchor must
    # leave it untouched rather than own it twice.
    assert tr["relapse_to_inj_prob"] == 0.5


def test_strata_without_a_drug_type_block_are_skipped():
    sc = scenario()
    sc["demographics"]["black_nonhispanic"]["sex_type"]["MSM"] = {"ppl": 0.03}
    n, _ = set_oud_initial_conditions(sc, calib())
    assert n == 3


@pytest.mark.parametrize("path", ANCHORED_PATHS)
def test_guard_fires_when_an_anchored_path_is_fitted(path):
    with pytest.raises(RuntimeError, match="data-anchored"):
        assert_not_fitted(calib(fitted_paths=[path]))


def test_guard_passes_on_a_search_of_only_free_parameters():
    # relapse_to_inj_prob left this list on 2026-09-29, when it was fixed at 0.5
    # under anchored_parameters; the two below are still searched.
    assert_not_fitted(calib(fitted_paths=[
        "oud_state.transitions.relapse_rate_per_year",
        "oud_state.transitions.remission_rate_per_year",
    ]))
