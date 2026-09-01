"""
Regression tests for the Stage D search design: gates, constraints, warm starts
and the multi-population scoring.

These guard three things that are silent when they break:

  * A gate stated in two places. The search selects against
    optuna_calibration.OBJECTIVE_GATES and validate_candidates reports against
    its own list; if those ever diverge, a run would be optimising for one
    threshold and judged against another, and nothing would print a warning.
  * A warm start that no longer matches the search space. Optuna drops enqueued
    keys it was not asked to suggest and clamps out-of-range ones, so a stale
    vector runs as a DIFFERENT point than the one recorded, with no error.
  * The per-population report join. A trial is scored on the eight populations'
    reports pooled into one, because scoring them separately and averaging the
    scores biases the squared objective upward (measured 5.4x at 5 replicates by
    analysis/probe_split_bias.py). The join has to keep replicate ids distinct or
    every count series comes out eight times too high.

See docs/calibration_log/phase_6_stage_d_search_rebuild_2026-09-01.md.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from titan.calibration.optuna_calibration import (
    OBJECTIVE_GATES,
    OBJECTIVE_NAMES,
    UNKNOWN_CONSTRAINT,
    WARM_START_YAML,
    constraints_func,
    concat_reports,
    load_yaml,
    trial_constraints,
    warm_start_trials,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_YAML = REPO_ROOT / "titan" / "calibration" / "calibration_targets.yml"


@pytest.fixture(scope="module")
def calib():
    return load_yaml(CALIB_YAML)


class _FakeTrial:
    """Just enough of optuna.Trial for constraints_func: a user_attrs mapping."""

    def __init__(self, user_attrs=None):
        self.user_attrs = user_attrs or {}


@pytest.mark.unit
def test_gates_are_stated_once():
    """validate_candidates must read the same numbers the search selects on."""
    from titan.calibration.validate_candidates import OBJECTIVES  # noqa: PLC0415

    assert len(OBJECTIVES) == len(OBJECTIVE_GATES) == len(OBJECTIVE_NAMES)
    assert [gate for _, gate in OBJECTIVES] == list(OBJECTIVE_GATES)


@pytest.mark.unit
def test_constraint_sign_convention():
    """Optuna reads <= 0 as satisfied, so it must be (error - gate), not the reverse.

    Getting this backwards is not a crash: the sampler would prefer trials that
    MISS every gate, and the run would look like an ordinary bad search.
    """
    inside = trial_constraints([g * 0.5 for g in OBJECTIVE_GATES])
    assert all(c < 0 for c in inside)

    outside = trial_constraints([g * 2.0 for g in OBJECTIVE_GATES])
    assert all(c > 0 for c in outside)

    exact = trial_constraints(OBJECTIVE_GATES)
    assert all(c == pytest.approx(0.0) for c in exact)


@pytest.mark.unit
def test_unknown_trials_are_treated_as_infeasible():
    """A trial with no recorded constraints predates this design or failed mid-run."""
    assert constraints_func(_FakeTrial()) == [UNKNOWN_CONSTRAINT] * len(OBJECTIVE_GATES)

    recorded = [-0.1, 0.2, -0.3, 0.4]
    assert constraints_func(_FakeTrial({"constraints": recorded})) == recorded


@pytest.mark.unit
def test_warm_starts_are_complete_points_in_todays_box(calib):
    """Every committed warm start must be enqueueable exactly as written."""
    entries = warm_start_trials(calib)
    assert entries, f"{WARM_START_YAML} has no usable entries"

    spec = {p["path"]: p for p in calib["calibration"]["parameters_to_optimize"]}
    labels = [label for label, _ in entries]
    assert len(set(labels)) == len(labels)

    for label, params in entries:
        assert set(params) == set(spec), label
        for name, value in params.items():
            lower, upper = spec[name]["lower"], spec[name]["upper"]
            assert lower <= value <= upper, f"{label}: {name}={value}"
            if spec[name].get("type") == "int":
                assert isinstance(value, int), f"{label}: {name} must stay an int"


@pytest.mark.unit
def test_missing_warm_start_file_is_a_cold_start(calib, tmp_path):
    """A cold start is a legitimate design and must not require a file."""
    assert warm_start_trials(calib, path=tmp_path / "absent.yml") == []


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["drop", "unknown", "out_of_box"])
def test_stale_warm_starts_fail_at_startup(calib, tmp_path, mutation):
    """Each of these is a way an old trial vector silently stops being the point.

    A dropped key is a parameter that joined the search since; an unknown key is
    one that left it; an out-of-box value is a bound that moved. Optuna reports
    none of the three at enqueue time.
    """
    label, params = warm_start_trials(calib)[0]
    params = dict(params)
    spec = {p["path"]: p for p in calib["calibration"]["parameters_to_optimize"]}

    if mutation == "drop":
        params.pop(next(iter(params)))
    elif mutation == "unknown":
        params["overdose.no_such_parameter"] = 1.0
    else:
        name = next(iter(params))
        params[name] = spec[name]["upper"] * 10

    path = tmp_path / "warm_start.yml"
    with open(path, "w") as f:
        yaml.safe_dump({"warm_start": [{"label": label, "params": params}]}, f)

    with pytest.raises(RuntimeError):
        warm_start_trials(calib, path=path)


@pytest.mark.unit
def test_joined_report_keeps_replicates_distinct(tmp_path):
    """Replicate ids restart per TITAN run, so a plain concatenation miscounts them.

    parse_basic_report divides its annual COUNT series by `run_id.nunique()`. If
    eight per-population reports each carrying run_ids 0-4 were joined unchanged,
    that count would read 5 instead of 40 and every count series would come out
    eight times too high -- which would look like a model transmitting eight times
    as much HIV as it does.
    """
    import pandas as pd  # noqa: PLC0415

    paths = []
    for block in range(3):
        frame = pd.DataFrame({
            "t": [1, 2], "run_id": [0, 1], "agents": [500, 500], "hiv": [10, 11],
        })
        path = tmp_path / f"report_{block}.txt"
        frame.to_csv(path, sep="\t", index=False)
        paths.append(path)

    dest = tmp_path / "pooled.txt"
    captured = {}

    def fake_parse(path, **kwargs):
        captured["frame"] = pd.read_csv(path, sep="\t")
        return {"parsed": True}

    import titan.calibration.optuna_calibration as oc  # noqa: PLC0415

    real_parse = oc.parse_basic_report
    oc.parse_basic_report = fake_parse
    try:
        stats = oc.concat_reports(paths, dest)
    finally:
        oc.parse_basic_report = real_parse

    assert stats == {"parsed": True}
    joined = captured["frame"]
    assert len(joined) == 6
    # Two replicates per file, three files, and no id shared between blocks.
    assert joined["run_id"].nunique() == 6
    assert dest.exists()


class _FrontTrial:
    """A trial stand-in for the front helpers: values plus a constraints attr."""

    def __init__(self, values, constraints=None):
        self.values = values
        self.user_attrs = {} if constraints is None else {"constraints": constraints}


@pytest.mark.unit
def test_unconstrained_front_keeps_only_non_dominated_trials():
    """The fallback when every trial misses a gate and best_trials comes back empty.

    An all-infeasible run is the likeliest outcome of the first constrained refit,
    and the first smoke run of it died on `pareto[0]` -- so the front this returns
    is what keeps the canonical pick and the front CSV from being lost with it.
    """
    from titan.calibration.optuna_calibration import (  # noqa: PLC0415
        unconstrained_pareto_front,
    )

    best = _FrontTrial([1.0, 1.0, 1.0, 1.0])
    trade = _FrontTrial([0.5, 3.0, 1.0, 1.0])
    dominated = _FrontTrial([2.0, 2.0, 2.0, 2.0])

    front = unconstrained_pareto_front([best, trade, dominated])
    assert best in front and trade in front
    assert dominated not in front
    assert unconstrained_pareto_front([]) == []


@pytest.mark.unit
def test_violation_is_zero_only_when_every_gate_is_met():
    from titan.calibration.optuna_calibration import (  # noqa: PLC0415
        constraint_violation,
    )

    passing = _FrontTrial([0.0] * 4, constraints=[-0.1, -0.2, -0.3, -0.4])
    assert constraint_violation(passing) == 0.0

    # Only the misses count: a large margin on one gate must not pay for a miss on
    # another, or "least infeasible" would rank a trial that fails badly above one
    # that fails narrowly.
    mixed = _FrontTrial([0.0] * 4, constraints=[-5.0, 0.25, -0.3, 0.75])
    assert constraint_violation(mixed) == pytest.approx(1.0)


@pytest.mark.unit
def test_sampling_design_is_the_costed_one(calib):
    """8 populations x 5 replicates: A1's seed count at job 4874631's cost.

    The two numbers are a pair. Raising the replicates without cutting the seed
    list multiplies the refit's walltime, which is the mistake this asserts
    against rather than the values themselves being sacred.
    """
    from titan.calibration.optuna_calibration import (  # noqa: PLC0415
        population_seeds,
        reps_per_seed,
    )

    seeds = population_seeds(calib)
    assert len(seeds) == 8
    assert len(seeds) * reps_per_seed(calib) == 40
