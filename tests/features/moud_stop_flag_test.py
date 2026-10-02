"""
The moud feature's side of the post-stop loss-of-tolerance window (2026-09-29).

Stopping MOUD in STEP 5 raises `agent.recent_moud_stop`; re-initiating clears it.
The overdose feature reads and expires the flag (see overdose_test.py). Same
hand-rolled harness as moud_test.py.
"""

from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.features import MOUD


def _make_agent():
    enroll = {"w": SimpleNamespace(start=0, stop=999, prob=0.1)}
    demographic_moud = SimpleNamespace(enroll=enroll, discontinue=0.0)
    demographics = {
        "black": SimpleNamespace(
            sex_type={"MSM": SimpleNamespace(drug_type={"Inj": SimpleNamespace(moud=demographic_moud)})}
        )
    }
    agent = SimpleNamespace(
        race="black",
        sex_type="MSM",
        drug_type="Inj",
        has_oud=True,
        is_in_remission=False,
        recent_relapse=False,
        recent_overdose=False,
        location=SimpleNamespace(params=SimpleNamespace(demographics=demographics)),
    )
    agent.moud = MOUD(agent)
    return agent, demographic_moud


def _make_model(discontinue_per_month, draw):
    return SimpleNamespace(
        time=0,
        params=SimpleNamespace(
            moud=SimpleNamespace(
                start_time=0,
                use_reinit=True,
                discontinue_per_month=discontinue_per_month,
                reinit_prob=0.2,
            ),
            features=SimpleNamespace(care_seeking=False),
        ),
        run_random=FakeRandom(draw),
    )


@pytest.fixture(autouse=True)
def _reset_counts():
    MOUD.counts = {"black": {"MSM": 0}}
    yield


@pytest.mark.unit
def test_stopping_raises_the_flag_and_restarting_clears_it():
    agent, demographic_moud = _make_agent()
    agent.moud.initiate(FakeRandom(0.0), demographic_moud, "init")
    assert agent.on_moud is True
    assert agent.recent_moud_stop is False

    # draw 0.14 < 0.5: the agent stops this step
    agent.moud.update_agent(_make_model(discontinue_per_month=0.5, draw=0.14))
    assert agent.on_moud is False
    assert agent.moud.active is False
    assert agent.recent_moud_stop is True
    assert agent.recent_moud_stop_timer == 0

    agent.moud.initiate(FakeRandom(0.0), demographic_moud, "prob")
    assert agent.on_moud is True
    assert agent.recent_moud_stop is False


@pytest.mark.unit
def test_staying_on_moud_leaves_the_flag_down():
    agent, demographic_moud = _make_agent()
    agent.moud.initiate(FakeRandom(0.0), demographic_moud, "init")

    # draw 0.14 >= 0.1: the agent stays on
    agent.moud.update_agent(_make_model(discontinue_per_month=0.1, draw=0.14))
    assert agent.on_moud is True
    assert agent.recent_moud_stop is False


# ---- restart within one step of an overdose (2026-10-01, post-overdose linkage diagnostic) ----

def _stopped_survivor(last_overdose_t):
    agent, demographic_moud = _make_agent()
    agent.moud.initiate(FakeRandom(0.0), demographic_moud, "init")      # ever on MOUD
    agent.moud.update_agent(_make_model(discontinue_per_month=0.5, draw=0.14))  # stops
    assert agent.on_moud is False
    agent.recent_overdose = True
    agent.last_overdose_t = last_overdose_t
    return agent


@pytest.mark.unit
def test_restart_within_one_step_of_an_overdose_is_counted_once():
    agent = _stopped_survivor(last_overdose_t=0)
    model = _make_model(discontinue_per_month=0.0, draw=0.1)   # reinit_prob 0.2 > 0.1: restarts
    model.time = 1
    agent.moud.update_agent(model)
    assert agent.on_moud is True
    assert agent.moud.started_postod_1step is True
    stats = {k: 0 for k in MOUD.stats}
    agent.moud.set_stats(stats, 1)
    assert stats["moud_start_postod_1step"] == 1
    assert agent.moud.started_postod_1step is False
    later = {k: 0 for k in MOUD.stats}
    agent.moud.set_stats(later, 2)
    assert later["moud_start_postod_1step"] == 0


@pytest.mark.unit
def test_restart_later_in_the_window_is_not_thirty_day_linkage():
    agent = _stopped_survivor(last_overdose_t=0)
    model = _make_model(discontinue_per_month=0.0, draw=0.1)
    model.time = 3
    agent.moud.update_agent(model)
    assert agent.on_moud is True
    assert agent.moud.started_postod_1step is False
    stats = {k: 0 for k in MOUD.stats}
    agent.moud.set_stats(stats, 3)
    assert stats["moud_start_postod_1step"] == 0


@pytest.mark.unit
def test_no_restart_no_count():
    agent = _stopped_survivor(last_overdose_t=0)
    model = _make_model(discontinue_per_month=0.0, draw=0.9)   # 0.9 > reinit_prob 0.2: stays off
    model.time = 1
    agent.moud.update_agent(model)
    assert agent.on_moud is False
    assert agent.moud.started_postod_1step is False
