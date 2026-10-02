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


@pytest.mark.unit
def test_post_moud_stop_multiplier_applies_only_off_moud():
    """
    The loss-of-tolerance term (added 2026-09-29): overdose.post_moud_stop_rr
    multiplies the per-step probability while `recent_moud_stop` is set and the
    agent is off MOUD, and never while on MOUD, whose own scale applies instead.

    base_prob_inj 0.12/yr is 0.01059 per step; x3 is 0.0318. A draw of 0.02
    fires only with the multiplier.
    """
    params = make_params(base_prob_inj=0.12, post_moud_stop_rr=3.0)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)
    assert a.recent_moud_stop is False

    model = make_fake_model(params, FakeRandom(0.02), [a])
    a.overdose.update_agent(model)
    assert a.overdose.events_this_step == 0

    a.recent_moud_stop = True
    a.on_moud = False
    a.overdose.update_agent(model)
    assert a.overdose.events_this_step == 1

    a.on_moud = True  # re-initiated: the flag is ignored while on treatment
    a.overdose.update_agent(model)
    assert a.overdose.events_this_step == 0


@pytest.mark.unit
def test_post_moud_stop_window_counts_the_stopping_step():
    """
    A window of 1 is the month of stopping only: the flag is raised by the
    moud feature within the step (before overdose runs) and the next step's
    update_pop clears it. A window of n stays up for n - 1 further steps.
    """
    params = make_params(post_moud_stop_window=1)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)
    a.recent_moud_stop = True
    a.recent_moud_stop_timer = 0

    model = make_fake_model(params, FakeRandom(1.1), [a])
    Overdose.update_pop(model)
    assert a.recent_moud_stop is False
    assert a.recent_moud_stop_timer == 0

    params = make_params(post_moud_stop_window=3)
    b = make_fake_agent(params)
    b.overdose.init_agent(None, 0)
    b.recent_moud_stop = True

    model = make_fake_model(params, FakeRandom(1.1), [b])
    Overdose.update_pop(model)
    assert b.recent_moud_stop is True
    Overdose.update_pop(model)
    assert b.recent_moud_stop is True
    Overdose.update_pop(model)
    assert b.recent_moud_stop is False

    # The two windows are independent: an overdose flag on the same agent is
    # advanced by the same call and expires on its own schedule.
    c = make_fake_agent(make_params(post_overdose_window=1, post_moud_stop_window=2))
    c.overdose.init_agent(None, 0)
    c.recent_overdose = True
    c.recent_moud_stop = True
    model = make_fake_model(c.location.params, FakeRandom(1.1), [c])
    Overdose.update_pop(model)
    assert c.recent_overdose is True and c.recent_moud_stop is True
    Overdose.update_pop(model)
    assert c.recent_overdose is False and c.recent_moud_stop is False


# ---- survivor MOUD status at the event (2026-10-01, post-overdose linkage diagnostic) ----

@pytest.mark.unit
def test_nonfatal_event_records_survivor_moud_status():
    params = make_params(fatal_frac=0.0)
    on = make_fake_agent(params)
    on.overdose.init_agent(None, 0)
    on.on_moud = True
    off = make_fake_agent(params)
    off.overdose.init_agent(None, 0)
    off.on_moud = False
    model = make_fake_model(params, FakeRandom(0.0), [on, off])
    model.time = 7
    on.overdose.update_agent(model)
    off.overdose.update_agent(model)
    assert (on.overdose.survivors_on_moud_this_step, on.overdose.survivors_off_moud_this_step) == (1, 0)
    assert (off.overdose.survivors_on_moud_this_step, off.overdose.survivors_off_moud_this_step) == (0, 1)
    assert on.last_overdose_t == 7 and off.last_overdose_t == 7
    stats = {k: 0 for k in Overdose.stats}
    on.overdose.set_stats(stats, 7)
    off.overdose.set_stats(stats, 7)
    assert stats == {"overdose_events": 2, "overdose_deaths": 0,
                     "od_survivors_on_moud": 1, "od_survivors_off_moud": 1}


@pytest.mark.unit
def test_fatal_event_is_not_a_survivor():
    params = make_params(fatal_frac=1.0)
    a = make_fake_agent(params)
    a.overdose.init_agent(None, 0)
    a.on_moud = True
    model = make_fake_model(params, FakeRandom(0.0), [a])
    model.time = 3
    a.overdose.update_agent(model)
    assert a.overdose.deaths_this_step == 1
    assert (a.overdose.survivors_on_moud_this_step, a.overdose.survivors_off_moud_this_step) == (0, 0)
    stats = {k: 0 for k in Overdose.stats}
    a.overdose.set_stats(stats, 3)
    assert stats["od_survivors_on_moud"] == 0 and stats["od_survivors_off_moud"] == 0
