"""
Regression tests for the HIV prevalence drift metric and the trial journal.

Both landed 2026-08-17 as Stage A of
docs/calibration_log/phase_6_completion_plan_2026-08-14.md.

The drift metric was a ratio of a 12-month opening window to a 12-month closing
window. It discarded ~80% of the scored steps and then squared what remained into
a ratio of two small numbers, over a PWID stratum of ~88 agents at 2-13%
prevalence -- single-digit HIV-positive counts at each end. Job 4963250 measured
the result: the same configuration under a different population seed moved the
metric by a median of 0.12 against a +/-0.10 tolerance, so it could not tell drift
from the population draw. It is now an OLS slope over every scored step.

The journal previously stored one user_attr, `hiv_new_pooled`. On 2026-08-11 the
decisive question about od_err -- which of its three components was missing --
could not be answered from job 4874631's journal, and that analysis was abandoned
rather than re-run.
"""

import numpy as np
import pytest

from titan.calibration.optuna_calibration import (
    OBJECTIVE_NAMES,
    STEPS_PER_YEAR,
    TRIAL_DIAGNOSTICS,
    parse_basic_report,
    record_trial_diagnostics,
)

COLUMNS = [
    "t", "drug_type", "sex_type", "race", "agents", "hiv", "hiv_dx", "haart",
    "hiv_new", "hiv_dx_new", "hiv_dx_new_inj", "incar", "oud_active",
    "oud_remission", "oud_none", "moud", "overdose_events", "overdose_deaths",
    "sti", "vaccine", "prep", "deaths",
]


def write_report(path, rows):
    """Minimal basicReport.txt: TSV, one row per timestep x stratum."""
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, 0)) for c in COLUMNS) + "\n")


def constant_prevalence_report(path, n_steps=132, prev_inj=0.10, prev_noninj=0.05,
                               agents_inj=88, agents_noninj=412):
    """Flat prevalence, with an active-OUD denominator and overdose events.

    The overdose columns matter: the three od rates divide by active-OUD
    person-years, so a fixture without them produces no od stats at all and the
    journal test would pass vacuously.
    """
    rows = []
    for t in range(1, n_steps + 1):
        rows.append({"t": t, "drug_type": "Inj", "sex_type": "HM", "race": "w",
                     "agents": agents_inj, "hiv": round(agents_inj * prev_inj),
                     "oud_active": agents_inj, "overdose_events": 1,
                     "overdose_deaths": 0})
        rows.append({"t": t, "drug_type": "NonInj", "sex_type": "HM", "race": "w",
                     "agents": agents_noninj,
                     "hiv": round(agents_noninj * prev_noninj),
                     "oud_active": agents_noninj, "overdose_events": 2,
                     "overdose_deaths": 1})
    write_report(path, rows)


@pytest.mark.unit
def test_flat_prevalence_has_zero_drift(tmp_path):
    report = tmp_path / "basicReport.txt"
    constant_prevalence_report(report)
    stats = parse_basic_report(report)
    assert abs(stats["hiv_prev_drift_pwid"]) < 1e-9
    assert abs(stats["hiv_prev_drift_noninj"]) < 1e-9


@pytest.mark.unit
def test_linear_growth_reports_its_slope_per_decade(tmp_path):
    """A doubling over eleven years is +0.91 per decade on a mean-relative scale.

    Prevalence runs 0.05 -> 0.10 linearly across 132 steps. The mean is 0.075, the
    total rise is 0.05 over 11 years, so per decade that is 0.04545, and relative
    to the mean 0.606. Asserted against the closed form rather than a recorded
    number so the test states the definition rather than echoing the code.
    """
    report = tmp_path / "basicReport.txt"
    n_steps, agents = 132, 10000
    rows = []
    for t in range(1, n_steps + 1):
        prev = 0.05 + 0.05 * (t - 1) / (n_steps - 1)
        rows.append({"t": t, "drug_type": "Inj", "sex_type": "HM", "race": "w",
                     "agents": agents, "hiv": round(agents * prev)})
    write_report(report, rows)

    stats = parse_basic_report(report)

    slope_per_step = 0.05 / (n_steps - 1)
    expected = slope_per_step * STEPS_PER_YEAR * 10.0 / 0.075
    assert stats["hiv_prev_drift_pwid"] == pytest.approx(expected, rel=1e-3)
    assert stats["hiv_prev_slope_pwid"] == pytest.approx(
        slope_per_step * STEPS_PER_YEAR * 10.0, rel=1e-3
    )
    assert stats["hiv_prev_mean_pwid"] == pytest.approx(0.075, rel=1e-3)


@pytest.mark.unit
def test_slope_uses_the_whole_window_not_just_the_ends(tmp_path):
    """The defect the OLS slope fixes, stated as a test.

    Two trajectories with identical opening and closing years: one flat, one that
    rises and falls back. The old open-vs-close ratio scored both at 0. A slope
    over all 132 steps also scores both near 0 -- but the point is that it is
    reading 132 observations rather than 24, so its estimate of "near 0" carries
    the variance of the whole window. Assert the estimator's stability directly:
    perturbing only the first and last months moves the ratio a great deal and the
    slope very little.
    """
    agents, n_steps = 88, 132

    def build(path, bump_ends):
        rows = []
        for t in range(1, n_steps + 1):
            hiv = 9
            if bump_ends and t <= STEPS_PER_YEAR:
                hiv = 6           # a plausible single-population draw at n=88
            elif bump_ends and t > n_steps - STEPS_PER_YEAR:
                hiv = 12
            rows.append({"t": t, "drug_type": "Inj", "sex_type": "HM", "race": "w",
                         "agents": agents, "hiv": hiv})
        write_report(path, rows)

    flat, bumped = tmp_path / "a.txt", tmp_path / "b.txt"
    build(flat, False)
    build(bumped, True)

    flat_drift = parse_basic_report(flat)["hiv_prev_drift_pwid"]
    bumped_drift = parse_basic_report(bumped)["hiv_prev_drift_pwid"]

    # The old metric would have read (12/6 - 1) = +1.00 on the bumped series, ten
    # times its own tolerance, purely from the two end windows.
    assert abs(flat_drift) < 1e-9
    assert abs(bumped_drift) < 0.35


@pytest.mark.unit
def test_drift_is_skipped_rather_than_nan_when_a_stratum_is_empty(tmp_path):
    """An empty compartment must not put NaN into a scored dict."""
    report = tmp_path / "basicReport.txt"
    rows = [{"t": t, "drug_type": "Inj", "sex_type": "HM", "race": "w",
             "agents": 0, "hiv": 0} for t in range(1, 20)]
    rows += [{"t": t, "drug_type": "NonInj", "sex_type": "HM", "race": "w",
              "agents": 400, "hiv": 20} for t in range(1, 20)]
    write_report(report, rows)

    stats = parse_basic_report(report)
    assert "hiv_prev_drift_pwid" not in stats
    assert np.isfinite(stats["hiv_prev_drift_noninj"])


class FakeTrial:
    """Stand-in for an optuna Trial; only set_user_attr is exercised."""

    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


@pytest.mark.unit
def test_journal_records_every_objective_component(tmp_path):
    """Naming the components is what makes a study readable after the fact.

    Positional order has changed four times since 2026-07-29, so the journal keys
    are names rather than indices.
    """
    report = tmp_path / "basicReport.txt"
    constant_prevalence_report(report)
    stats = parse_basic_report(report)

    from titan.calibration.optuna_calibration import CALIB_YAML, build_targets, load_yaml

    targets = build_targets(load_yaml(CALIB_YAML)["calibration"])
    trial = FakeTrial()
    record_trial_diagnostics(trial, stats, targets)

    for name in OBJECTIVE_NAMES:
        assert f"err_{name}" in trial.user_attrs
        assert np.isfinite(trial.user_attrs[f"err_{name}"])

    # The three od components, without which the 2026-08-11 diagnosis could not
    # be run and was abandoned.
    for key in ("fatal_od_rate", "nonfatal_od_rate_inj", "nonfatal_od_rate_noninj"):
        assert key in trial.user_attrs

    assert "hiv_prev_drift_pwid" in trial.user_attrs
    assert "inj_share_active_oud" in trial.user_attrs


@pytest.mark.unit
def test_journal_prefix_namespaces_per_seed_records(tmp_path):
    """Stage A1 needs the spread across population seeds, not only the mean."""
    report = tmp_path / "basicReport.txt"
    constant_prevalence_report(report)
    stats = parse_basic_report(report)

    from titan.calibration.optuna_calibration import CALIB_YAML, build_targets, load_yaml

    targets = build_targets(load_yaml(CALIB_YAML)["calibration"])
    trial = FakeTrial()
    record_trial_diagnostics(trial, stats, targets, prefix="seed12345_")

    assert "seed12345_err_od_err" in trial.user_attrs
    assert "err_od_err" not in trial.user_attrs


@pytest.mark.unit
def test_journal_skips_non_scalar_and_non_finite_values(tmp_path):
    """Series and NaNs must not reach the journal; optuna stores scalars."""
    trial = FakeTrial()
    stats = {
        "fatal_od_rate": float("nan"),
        "moud_uptake_series": np.array([0.1, 0.2]),
        "oud_active": 0.55,
    }
    from titan.calibration.optuna_calibration import CALIB_YAML, build_targets, load_yaml

    targets = build_targets(load_yaml(CALIB_YAML)["calibration"])
    record_trial_diagnostics(trial, stats, targets)

    assert "fatal_od_rate" not in trial.user_attrs
    assert "moud_uptake_series" not in trial.user_attrs
    assert trial.user_attrs["oud_active"] == 0.55
    assert all(isinstance(v, float) for v in trial.user_attrs.values())


@pytest.mark.unit
def test_diagnostics_table_has_no_duplicate_attr_names():
    names = [attr for _, attr in TRIAL_DIAGNOSTICS]
    assert len(names) == len(set(names))
