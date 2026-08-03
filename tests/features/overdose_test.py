"""
Tests for Overdose build a minimal, hand-rolled harness (ObjMap params,
SimpleNamespace agent/model), the same pattern `care_seeking_test.py` uses -
see that file's module docstring for why.

Overdose reads only `params.overdose`, `params.model.time.steps_per_year` and a
few plain attributes off `agent`/`model`, so the harness keeps these tests
independent of the demographics/classes machinery the shared fixtures build.
"""

from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.features import Overdose


DEFAULT_OVERDOSE_PARAMS = {
    "base_prob_inj": 0.5,
    "base_prob_noninj": 0.25,
    "fatal_frac": 0.0,
    "high_risk_mult": 1.0,
    "ssp_rr": 1.0,
    "post_relapse_od_rr": 1.0,
    "post_overdose_window": 6,
}


def make_params(**overrides):
    overdose = dict(DEFAULT_OVERDOSE_PARAMS)
    overdose.update(overrides)
    return SimpleNamespace(
        overdose=SimpleNamespace(**overdose),
        model=SimpleNamespace(time=SimpleNamespace(steps_per_year=12)),
    )


def make_fake_agent(params, drug_type="Inj", has_oud=True, race="black", sex_type="MSM"):
    fake_agent = SimpleNamespace(
        race=race,
        sex_type=sex_type,
        drug_type=drug_type,
        has_oud=has_oud,
        location=SimpleNamespace(params=params),
    )
    fake_agent.overdose = Overdose(fake_agent)
    return fake_agent


def make_fake_model(params, run_random, agents):
    return SimpleNamespace(
        params=params,
        run_random=run_random,
        pop=SimpleNamespace(all_agents=agents),
    )


@pytest.fixture(autouse=True)
def _reset_counts():
    Overdose.event_counts = {"black": {"MSM": 0}}
    Overdose.death_counts = {"black": {"MSM": 0}}
    yield


@pytest.mark.unit
def test_init_agent():
    params = make_params()
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)

    assert a.overdose.active
    assert a.recent_overdose is False
    assert a.recent_overdose_timer == 0


@pytest.mark.unit
def test_nonfatal_event():
    params = make_params(fatal_frac=0.0)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)

    model = make_fake_model(params, FakeRandom(0.0), [a])
    a.overdose.update_agent(model)

    assert a.overdose.events_this_step == 1
    assert a.overdose.deaths_this_step == 0
    assert a.recent_overdose is True
    assert a.recent_overdose_timer == 0
    assert getattr(a, "exit_class", None) != "death"


@pytest.mark.unit
def test_fatal_event_sets_exit():
    params = make_params(fatal_frac=1.0)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)

    model = make_fake_model(params, FakeRandom(0.0), [a])
    a.overdose.update_agent(model)

    assert a.overdose.deaths_this_step == 1
    assert a.exit_class == "death"
    assert a.death_reason == "overdose"


@pytest.mark.unit
def test_ineligible_agent_no_event():
    params = make_params()
    a = make_fake_agent(params, has_oud=False)
    a.overdose.init_agent(None, 0)

    model = make_fake_model(params, FakeRandom(0.0), [a])
    a.overdose.update_agent(model)

    assert a.overdose.events_this_step == 0


@pytest.mark.unit
def test_recent_overdose_window_boundary():
    """
    A nonfatal overdose sets `recent_overdose` for exactly
    `post_overdose_window` subsequent `update_pop` calls, then clears it -
    not window + 1. `Overdose.update_pop` is where the window is now
    advanced/expired (moved out of this feature's own `update_agent`), so
    that `moud`/`hiv_testing` - declared earlier in
    `titan/features/__init__.py` and so reading this flag before `overdose`
    runs within a timestep - see this step's expiry rather than a stale
    value carried over from the step before.
    """
    window = 6
    params = make_params(post_overdose_window=window)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)

    a.recent_overdose = True
    a.recent_overdose_timer = 0

    model = make_fake_model(params, FakeRandom(1.1), [a])
    for step in range(1, window + 1):
        Overdose.update_pop(model)
        assert a.recent_overdose is True, f"expected True at step {step} (window={window})"

    Overdose.update_pop(model)
    assert a.recent_overdose is False
    assert a.recent_overdose_timer == 0


@pytest.mark.unit
def test_update_pop_skips_inactive_and_unflagged_agents():
    params = make_params()
    inactive = make_fake_agent(params)
    inactive.overdose.active = False
    inactive.recent_overdose = True
    inactive.recent_overdose_timer = 999

    unflagged = make_fake_agent(params)
    unflagged.overdose.init_agent(None, 0)  # recent_overdose False

    model = make_fake_model(params, FakeRandom(1.1), [inactive, unflagged])
    Overdose.update_pop(model)  # must not raise, must not touch either agent

    assert inactive.recent_overdose_timer == 999
    assert unflagged.recent_overdose is False


@pytest.mark.unit
def test_set_stats():
    params = make_params()
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)
    a.overdose.events_this_step = 2
    a.overdose.deaths_this_step = 1

    stats = {stat: 0 for stat in Overdose.stats}
    a.overdose.set_stats(stats, 0)

    assert stats["overdose_events"] == 2
    assert stats["overdose_deaths"] == 1
