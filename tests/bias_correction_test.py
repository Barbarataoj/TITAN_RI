"""
Regression tests for the noise-corrected objectives (2026-09-29).

Both squared-error objectives carry their own Monte Carlo sampling variance as
an upward bias, E[(x + e)^2] = x^2 + var(e). probe_split_bias measured it on
od_err (0.0201 at 5 replicates against 0.0037 at 40 for one configuration), and
the arithmetic is worse on ridoh_hiv_err, whose windows hold single-digit
simulated diagnoses: a Poisson simulation of a PERFECT model at 40 runs averages
0.87 against a 1.1 gate. Each objective now subtracts the estimated variance of
the statistic it squares. These tests pin the estimator's inputs, its direction,
and its expectation.
"""

import numpy as np
import pytest

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    OD_SCORING_YEARS,
    STEPS_PER_YEAR,
    _annual_counts_by_run,
    _window_variances,
    build_targets,
    hiv_null_value,
    load_yaml,
    multi_error,
    od_error,
    parse_basic_report,
    ridoh_hiv_rmse,
)

COLUMNS = [
    "run_id", "t", "drug_type", "sex_type", "race", "agents", "hiv", "hiv_dx",
    "haart", "hiv_new", "hiv_dx_new", "hiv_dx_new_inj", "hiv_tests", "incar",
    "oud_active", "oud_remission", "oud_none", "moud", "moud_active",
    "overdose_events", "overdose_deaths",
]


def write_report(path, rows):
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, 0)) for c in COLUMNS) + "\n")


def replicate_report(path, n_runs, n_steps=132, events_inj_per_step=None,
                     events_noninj_per_step=None, dx_inj_per_step=None,
                     tests_per_step=2, rng=None):
    """n_runs replicates of a two-stratum population.

    Per-step counts can be constants or callables of (run, t) so a test can put
    noise, or a step change, exactly where it wants it.
    """
    def value(spec, run, t, default):
        if spec is None:
            return default
        return spec(run, t) if callable(spec) else spec

    rows = []
    for run in range(n_runs):
        for t in range(1, n_steps + 1):
            e_inj = value(events_inj_per_step, run, t, 1)
            rows.append({"run_id": f"r{run}", "t": t, "drug_type": "Inj",
                         "sex_type": "HM", "race": "w", "agents": 88, "hiv": 5,
                         "oud_active": 88, "overdose_events": e_inj,
                         "overdose_deaths": 0,
                         "hiv_dx_new_inj": value(dx_inj_per_step, run, t, 0),
                         "hiv_dx_new": value(dx_inj_per_step, run, t, 0),
                         "hiv_tests": tests_per_step})
            rows.append({"run_id": f"r{run}", "t": t, "drug_type": "NonInj",
                         "sex_type": "HM", "race": "w", "agents": 412, "hiv": 8,
                         "oud_active": 187,
                         "overdose_events": value(events_noninj_per_step, run, t, 1),
                         "overdose_deaths": 0})
    write_report(path, rows)


@pytest.fixture(scope="module")
def targets():
    return build_targets(load_yaml(CALIB_YAML)["calibration"])


# ---------------------------------------------------------------- parser inputs

@pytest.mark.unit
def test_per_run_counts_average_to_the_pooled_series(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "r.txt"
    replicate_report(path, n_runs=6, dx_inj_per_step=lambda r, t: int(rng.poisson(0.3)))
    stats = parse_basic_report(path)
    by_run = stats["idu_dx_counts_by_run"]
    assert by_run.shape == (6, 11)
    np.testing.assert_allclose(by_run.mean(axis=0), stats["idu_dx_counts_per_year"])
    np.testing.assert_allclose(stats["pwid_dx_counts_by_run"].mean(axis=0),
                               stats["pwid_dx_counts_per_year"])


@pytest.mark.unit
def test_overdose_rates_pool_the_last_two_years(tmp_path):
    """T3 is 2021-2022, so the scored rate is the mean over the last two model
    years -- not the final year alone, as it was until 2026-09-29."""
    assert OD_SCORING_YEARS == 2
    last_year_start = 132 - STEPS_PER_YEAR + 1
    path = tmp_path / "r.txt"
    # 1 event/step in every year but the last, 3/step in the last year.
    replicate_report(path, n_runs=2,
                     events_inj_per_step=lambda r, t: 3 if t >= last_year_start else 1)
    stats = parse_basic_report(path)
    # Person-years per stratum per run-year = 88; events over 2 years = 12 + 36.
    expected = (12 + 36) / (2 * 88)
    assert stats["nonfatal_od_rate_inj"] == pytest.approx(expected)
    by_run = stats["nonfatal_od_rate_inj_by_run"]
    assert by_run.shape == (2,)
    np.testing.assert_allclose(by_run, expected)


@pytest.mark.unit
def test_per_run_rates_match_the_pooled_rate_when_runs_agree(tmp_path):
    path = tmp_path / "r.txt"
    replicate_report(path, n_runs=4, events_inj_per_step=2, events_noninj_per_step=3)
    stats = parse_basic_report(path)
    np.testing.assert_allclose(stats["nonfatal_od_rate_inj_by_run"],
                               stats["nonfatal_od_rate_inj"])
    np.testing.assert_allclose(stats["nonfatal_od_rate_noninj_by_run"],
                               stats["nonfatal_od_rate_noninj"])


@pytest.mark.unit
def test_single_run_reports_carry_no_replicate_arrays(tmp_path):
    path = tmp_path / "r.txt"
    replicate_report(path, n_runs=1)
    stats = parse_basic_report(path)
    assert stats["idu_dx_counts_by_run"] is None
    assert "nonfatal_od_rate_inj_by_run" not in stats


@pytest.mark.unit
def test_hiv_tests_per_pwid_year_is_a_rate(tmp_path):
    path = tmp_path / "r.txt"
    replicate_report(path, n_runs=3, tests_per_step=4)
    stats = parse_basic_report(path)
    # 4 tests/step over 88 PWID: 48 tests per 88 person-years.
    assert stats["hiv_tests_per_pwid_yr"] == pytest.approx(48 / 88)


# ---------------------------------------------------------------- the estimator

@pytest.mark.unit
def test_window_variance_is_the_variance_of_the_mean():
    by_run = np.array([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0], [2.0, 2.0, 2.0], [0.0, 4.0, 2.0]])
    windows = [{"years": [2000, 2001], "cases_per_year": 0.0},
               {"years": [2002, 2002], "cases_per_year": 0.0}]
    v = _window_variances(by_run, 2000, 1.0, windows)
    # Window 1 means per run are all 2.0 -> zero variance; window 2 is constant too.
    assert v == [0.0, 0.0]
    by_run[:, 2] = [1.0, 2.0, 3.0, 4.0]
    v = _window_variances(by_run, 2000, 2.0, windows)
    assert v[1] == pytest.approx(np.var([2.0, 4.0, 6.0, 8.0], ddof=1) / 4)
    assert _window_variances(by_run[:1], 2000, 1.0, windows) == []
    assert _window_variances(None, 2000, 1.0, windows) == []


@pytest.mark.unit
def test_od_error_has_two_components_and_ignores_the_fatal_rate(targets):
    stats = {"nonfatal_od_rate_inj": targets["od_nonfatal_inj"],
             "nonfatal_od_rate_noninj": targets["od_nonfatal_noninj"],
             "fatal_od_rate": 10 * targets["od_fatal"]}
    assert od_error(stats, targets) == 0.0
    stats["nonfatal_od_rate_inj"] *= 1.1
    assert od_error(stats, targets) == pytest.approx(0.01 / 2)


@pytest.mark.unit
def test_corrected_never_exceeds_raw_and_zero_spread_changes_nothing(targets):
    rng = np.random.default_rng(7)
    inj, non = targets["od_nonfatal_inj"], targets["od_nonfatal_noninj"]
    for _ in range(50):
        by_inj = inj * (1 + 0.15 * rng.standard_normal(8))
        by_non = non * (1 + 0.15 * rng.standard_normal(8))
        stats = {"nonfatal_od_rate_inj": by_inj.mean(),
                 "nonfatal_od_rate_noninj": by_non.mean(),
                 "nonfatal_od_rate_inj_by_run": by_inj,
                 "nonfatal_od_rate_noninj_by_run": by_non}
        assert od_error(stats, targets) <= od_error(stats, targets, bias_correct=False)
    flat = {"nonfatal_od_rate_inj": inj * 1.2, "nonfatal_od_rate_noninj": non,
            "nonfatal_od_rate_inj_by_run": np.full(8, inj * 1.2),
            "nonfatal_od_rate_noninj_by_run": np.full(8, non)}
    assert od_error(flat, targets) == od_error(flat, targets, bias_correct=False)


@pytest.mark.unit
def test_od_correction_removes_the_bias_in_expectation(targets):
    """A perfect model scored on noisy replicate rates: raw od_err averages
    var/target^2, corrected averages ~0."""
    rng = np.random.default_rng(11)
    inj, non = targets["od_nonfatal_inj"], targets["od_nonfatal_noninj"]
    R, cv = 8, 0.2
    raw, corrected, subtracted = [], [], []
    for _ in range(400):
        by_inj = inj * (1 + cv * rng.standard_normal(R))
        by_non = non * (1 + cv * rng.standard_normal(R))
        stats = {"nonfatal_od_rate_inj": by_inj.mean(),
                 "nonfatal_od_rate_noninj": by_non.mean(),
                 "nonfatal_od_rate_inj_by_run": by_inj,
                 "nonfatal_od_rate_noninj_by_run": by_non}
        raw.append(od_error(stats, targets, bias_correct=False))
        corrected.append(od_error(stats, targets))
        subtracted.append(np.mean([by_inj.var(ddof=1) / R / inj ** 2,
                                   by_non.var(ddof=1) / R / non ** 2]))
    # The raw score IS the sampling variance, and the estimator subtracts an
    # unbiased estimate of it: before the clamp at zero the two agree in mean.
    assert np.mean(raw) == pytest.approx(cv ** 2 / R, rel=0.25)
    assert np.mean(subtracted) == pytest.approx(np.mean(raw), rel=0.15)
    # The clamp keeps the corrected mean positive (about 0.4x the raw bias for
    # a symmetric estimation error); it must sit well below the raw score.
    assert np.mean(corrected) < 0.6 * np.mean(raw)


@pytest.mark.unit
def test_hiv_correction_removes_the_bias_in_expectation(targets):
    """Poisson counts at exactly the RIDOH levels, 40 replicates, statewide
    scale ~50: raw ridoh_hiv_err sits near the 1.1 gate, corrected well below."""
    rng = np.random.default_rng(3)
    scale = targets["pop_scale"]
    first_year = targets["first_year"]
    years = first_year + np.arange(11)
    # Per-year expected statewide counts that hit both windows of each series
    # exactly: level a through 2017, b from 2018, so the 2018-2022 window is b
    # and the 2015-2019 window is (3a + 2b) / 5. Read from the targets rather
    # than hard-coded, because the all-route windows are derived (2026-09-29).
    # Since 2026-10-07 only the 2018-2022 windows are scored; the 2015-2019
    # windows ride along as `*_unscored`, and the simulated series still hits
    # both so the test is indifferent to which are scored.
    def levels(windows):
        early = next(w["cases_per_year"] for w in windows if w["years"][0] == 2015)
        late = next(w["cases_per_year"] for w in windows if w["years"][0] == 2018)
        a = (5 * early - 2 * late) / 3
        return np.where(years <= 2017, a, late)
    idu = levels(targets["ridoh_windows"] + targets.get("ridoh_windows_unscored", []))
    pwid = levels(targets["ridoh_pwid_windows"] + targets.get("ridoh_pwid_windows_unscored", []))
    assert np.all(pwid >= idu) and np.all(idu >= 0)
    R = 40
    raw, corrected = [], []
    for _ in range(300):
        idu_runs = rng.poisson(idu / scale, size=(R, 11)).astype(float)
        pwid_runs = idu_runs + rng.poisson((pwid - idu) / scale, size=(R, 11))
        args = (idu_runs.mean(axis=0), pwid_runs.mean(axis=0), first_year, targets)
        raw.append(ridoh_hiv_rmse(*args))
        corrected.append(ridoh_hiv_rmse(*args, idu_by_run=idu_runs, pwid_by_run=pwid_runs))
    raw, corrected = np.array(raw), np.array(corrected)
    assert np.all(corrected <= raw + 1e-12)
    assert raw.mean() > 0.6                     # the uncorrected floor is real
    assert corrected.mean() < 0.5 * raw.mean()  # and the correction removes most of it
    # The correction fixes the EXPECTATION, not the spread: with windows of
    # ~7 events the variance estimate is itself noisy, so a perfect model still
    # fails the gate on some draws. Measured here at 40 runs: raw passes ~76%,
    # corrected ~84%. The 8 x 40 validation stage is what settles a candidate;
    # the search only has to rank on an unbiased score.
    assert (corrected <= 1.1).mean() > (raw <= 1.1).mean() + 0.03
    assert (corrected <= 1.1).mean() > 0.8


@pytest.mark.unit
def test_null_value_and_uncorrected_paths_are_unchanged(targets):
    calib = load_yaml(CALIB_YAML)
    # 4.782 against the unadjusted RIDOH windows; 3.752 once the all-route
    # windows became opioid-adjusted (2026-09-29); since 2026-10-07 only the
    # 2018-2022 windows are scored, so the null is the RMS of their two levels
    # and the (unpenalised) pre-2015 bound: sqrt((2.5^2 + 6.59^2 + 0) / 3) = 4.07.
    scored = [w["cases_per_year"] for w in targets["ridoh_windows"] + targets["ridoh_pwid_windows"]]
    expected = float(np.sqrt(np.mean([c ** 2 for c in scored] + [0.0])))
    assert expected == pytest.approx(4.07, abs=0.01)
    assert hiv_null_value(calib) == pytest.approx(expected, abs=1e-6)
    zeros = np.zeros(11)
    assert ridoh_hiv_rmse(zeros, zeros, targets["first_year"], targets,
                          idu_by_run=np.zeros((5, 11)), pwid_by_run=np.zeros((5, 11))) \
        == pytest.approx(hiv_null_value(calib))


@pytest.mark.unit
def test_multi_error_exposes_both_scores(tmp_path, targets):
    rng = np.random.default_rng(5)
    path = tmp_path / "r.txt"
    replicate_report(path, n_runs=8,
                     events_inj_per_step=lambda r, t: int(rng.poisson(0.7)),
                     events_noninj_per_step=lambda r, t: int(rng.poisson(0.7)),
                     dx_inj_per_step=lambda r, t: int(rng.poisson(0.05)))
    stats = parse_basic_report(path)
    corrected = multi_error(stats, targets)
    raw = multi_error(stats, targets, bias_correct=False)
    assert len(corrected) == len(raw) == 4
    assert corrected[1] <= raw[1] and corrected[2] <= raw[2]
    assert corrected[0] == raw[0] and corrected[3] == raw[3]


@pytest.mark.unit
def test_all_route_windows_are_derived_from_their_parts():
    """IDU + (1 - s) x MSM/IDU (2026-09-29); a block without parts is unchanged."""
    from titan.calibration.optuna_calibration import all_route_windows  # noqa: PLC0415
    block = {"stimulant_share_of_msm_idu": 0.34,
             "parts": [{"years": [2015, 2019], "idu": 1.8, "msm_idu": 3.6},
                       {"years": [2018, 2022], "idu": 2.5, "msm_idu": 6.2}],
             "windows": [{"years": [2015, 2019], "cases_per_year": 5.4},
                         {"years": [2018, 2022], "cases_per_year": 8.7}]}
    derived = all_route_windows(block)
    assert [w["years"] for w in derived] == [[2015, 2019], [2018, 2022]]
    assert derived[0]["cases_per_year"] == pytest.approx(1.8 + 0.66 * 3.6)
    assert derived[1]["cases_per_year"] == pytest.approx(2.5 + 0.66 * 6.2)
    block["stimulant_share_of_msm_idu"] = 0.0
    assert [w["cases_per_year"] for w in all_route_windows(block)] == pytest.approx([5.4, 8.7])
    assert all_route_windows({"windows": block["windows"]}) == block["windows"]
    block["stimulant_share_of_msm_idu"] = 1.5
    with pytest.raises(ValueError):
        all_route_windows(block)


# ------------------------------------------------- scoring window (2026-10-07)

@pytest.mark.unit
def test_score_t_max_makes_a_long_run_score_like_the_calibration_run(tmp_path):
    """A 240-step Phase 7 report scored with score_t_max=132 must give the same
    overdose rates, OUD shares and series lengths as the 132-step run it
    extends; without it the end-anchored OD window reads the last two years."""
    from titan.calibration.optuna_calibration import scoring_t_max
    assert scoring_t_max(load_yaml(CALIB_YAML)) == 132
    short, long_ = tmp_path / "short.txt", tmp_path / "long.txt"
    replicate_report(short, n_runs=3, n_steps=132, events_inj_per_step=1)
    replicate_report(long_, n_runs=3, n_steps=240,
                     events_inj_per_step=lambda r, t: 1 if t <= 132 else 5)
    s = parse_basic_report(short)
    l_cut = parse_basic_report(long_, score_t_max=132)
    l_full = parse_basic_report(long_)
    for key in ("nonfatal_od_rate_inj", "nonfatal_od_rate_noninj", "oud_active"):
        assert l_cut[key] == pytest.approx(s[key])
    assert len(l_cut["idu_dx_counts_per_year"]) == len(s["idu_dx_counts_per_year"]) == 11
    assert l_full["nonfatal_od_rate_inj"] > 3 * s["nonfatal_od_rate_inj"]
    assert len(l_full["idu_dx_counts_per_year"]) == 20


@pytest.mark.unit
def test_only_scored_windows_enter_the_hiv_objective(targets):
    """With the 2018-2022 windows alone scored, a stationary model at their
    levels scores zero: the two-window floor is gone. None/empty scores all."""
    from titan.calibration.optuna_calibration import split_scored_windows
    windows = [{"years": [2015, 2019], "cases_per_year": 1.8},
               {"years": [2018, 2022], "cases_per_year": 2.5}]
    assert split_scored_windows(windows, None) == (windows, [])
    scored, unscored = split_scored_windows(windows, [2018])
    assert [w["years"] for w in scored] == [[2018, 2022]]
    assert [w["years"] for w in unscored] == [[2015, 2019]]
    with pytest.raises(ValueError):
        split_scored_windows(windows, [1999])
    scale = targets["pop_scale"]
    idu = np.full(11, 2.5 / scale)
    pwid = np.full(11, targets["ridoh_pwid_windows"][0]["cases_per_year"] / scale)
    assert ridoh_hiv_rmse(idu, pwid, targets["first_year"], targets) == pytest.approx(0.0, abs=1e-9)
