"""
Regression tests for the data-anchored parameters added 2026-09-29.

Seven parameters left the search that day: six are stated with their sources in
calibration_targets.yml `anchored_parameters`, and overdose.fatal_frac is
derived from the overdose targets. These tests guard the same failure modes the
OUD anchors' tests do -- a pin that is written twice, a pin that finds its way
back into parameters_to_optimize, and a warm start that still carries a
coordinate the search no longer suggests -- plus the derivation itself.
"""

import copy
from pathlib import Path

import pytest

from titan.calibration.optuna_calibration import (
    ANCHORED_PATHS,
    BASE_SCENARIO,
    CALIB_YAML,
    anchored_parameters,
    assert_not_fitted,
    derived_fatal_frac,
    load_yaml,
    set_anchored_parameters,
    warm_start_trials,
)


@pytest.fixture(scope="module")
def calib():
    return load_yaml(CALIB_YAML)


@pytest.mark.unit
def test_fatal_frac_is_the_fatal_share_the_targets_imply(calib):
    c = calib["calibration"]
    od = c["overdose"]
    share = c["injecting"]["injecting_pct"]
    nonfatal = (share * od["nonfatal_rate_inj_annual"]
                + (1 - share) * od["nonfatal_rate_noninj_annual"])
    expected = od["fatal_rate_annual"] / (od["fatal_rate_annual"] + nonfatal)
    assert derived_fatal_frac(calib) == pytest.approx(expected)
    # The number the 2026-08-11 box [0.1755, 0.2102] and the audit's 0.15-0.18
    # both sat below; if it moves out of this band the targets changed.
    assert 0.19 < derived_fatal_frac(calib) < 0.21


@pytest.mark.unit
def test_fatal_frac_tracks_the_targets():
    calib = {"calibration": {
        "injecting": {"injecting_pct": 0.5},
        "overdose": {"fatal_rate_annual": 0.02, "nonfatal_rate_inj_annual": 0.10,
                     "nonfatal_rate_noninj_annual": 0.06},
    }}
    assert derived_fatal_frac(calib) == pytest.approx(0.02 / (0.02 + 0.08))


@pytest.mark.unit
def test_every_yaml_anchor_is_in_the_guard_list(calib):
    """A path pinned in YAML but absent from ANCHORED_PATHS would be guarded only
    by the dynamic check; keep the two lists in step so the parametrized guard
    tests cover it too."""
    assert set(anchored_parameters(calib)) <= set(ANCHORED_PATHS)


@pytest.mark.unit
def test_live_config_has_nine_free_and_no_overlap(calib):
    assert_not_fitted(calib)
    fitted = {p["path"] for p in calib["calibration"]["parameters_to_optimize"]}
    assert not fitted & set(anchored_parameters(calib))
    # 9 until 2026-10-01; moud.reinit_prob left (anchored) and moud.acceptance +
    # hiv_testing.nonclinical_annual joined when features.care_seeking went on.
    assert len(fitted) == 12   # 10 + the two MOUD trend ratios (2026-10-05)


@pytest.mark.unit
def test_guard_fires_when_a_yaml_anchor_is_fitted(calib):
    bad = copy.deepcopy(calib)
    bad["calibration"]["parameters_to_optimize"].append(
        {"path": "moud.discontinue_per_month", "lower": 0.03, "upper": 0.2})
    with pytest.raises(RuntimeError, match="data-anchored"):
        assert_not_fitted(bad)


@pytest.mark.unit
def test_duplicate_yaml_anchor_is_an_error(calib):
    bad = copy.deepcopy(calib)
    entries = bad["calibration"]["anchored_parameters"]
    entries.append(dict(entries[0]))
    with pytest.raises(RuntimeError, match="twice"):
        anchored_parameters(bad)


@pytest.mark.unit
def test_set_anchored_parameters_writes_every_value_into_the_base(calib):
    scenario = load_yaml(BASE_SCENARIO)
    written = set_anchored_parameters(scenario, calib)
    assert set(written) == set(anchored_parameters(calib))
    for path, value in written.items():
        node = scenario
        for part in path.split("."):
            node = node[part]
        assert node == value
    assert scenario["overdose"]["fatal_frac"] == pytest.approx(derived_fatal_frac(calib))
    # The pin's value is the file's business (0.155 -> 0.08 on 2026-09-30); the
    # test checks that what the file states is what the scenario gets.
    assert scenario["hiv_testing"]["base_prob_annual"] == anchored_parameters(calib)["hiv_testing.base_prob_annual"]
    assert scenario["moud"]["overdose_scale"]["active"] == 0.30


@pytest.mark.unit
def test_warm_starts_carry_no_anchored_coordinate(calib):
    anchored = set(anchored_parameters(calib))
    entries = warm_start_trials(calib)
    assert entries, "the warm-start file should still enqueue something"
    for label, params in entries:
        assert not set(params) & anchored, label


@pytest.mark.unit
def test_apply_trends_writes_geometric_yearly_steps_reaching_the_ratio():
    from titan.calibration.optuna_calibration import apply_trends, TREND_STEPS
    s = {"features": {}, "moud": {"acceptance": 0.5, "discontinue_per_month": 0.078}}
    apply_trends(s, {"trend.moud_acceptance_ratio": 2.0, "trend.moud_discontinue_ratio": 1.0})
    tl = s["timeline_scaling"]["timeline"]
    acc = [v for k, v in tl.items() if k.startswith("moud_acceptance_ratio")]
    assert len(acc) == len(TREND_STEPS) and s["features"]["timeline_scaling"] is True
    prod = 1.0
    for e in acc:
        assert e["parameter"] == "moud|acceptance" and e["stop_time"] == 10000
        prod *= e["scalar"]
    assert prod == pytest.approx(2.0, rel=1e-6)
    assert [e["start_time"] for e in acc] == list(TREND_STEPS)
    assert not any(k.startswith("moud_discontinue_ratio") for k in tl)   # ratio 1 adds nothing


@pytest.mark.unit
def test_weighted_traj_rmse_and_flat_line_fails_the_moud_gate():
    import yaml
    from titan.calibration.optuna_calibration import _traj_rmse, OBJECTIVE_GATES, CALIB_YAML
    m = yaml.safe_load(open(CALIB_YAML))["calibration"]["moud"]
    tgt, w = m["uptake_series_per_yr"], m["uptake_series_weights"]
    assert _traj_rmse(tgt, tgt, w) == pytest.approx(0.0)
    best_flat = min(_traj_rmse([x / 1000] * len(tgt), tgt, w) for x in range(600, 850))
    assert best_flat > OBJECTIVE_GATES[3]            # a flat line cannot pass any more
    assert _traj_rmse([0.68] * len(tgt), tgt) < 0.10  # ... though it passed the old gate unweighted
