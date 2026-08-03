"""
Hand-rolled SimpleNamespace agent/model harness, same reason as
tests/features/overdose_test.py's module docstring: MOUD reads only its own
params block plus a few plain agent attributes, so the harness keeps these tests
independent of the demographics/classes machinery the shared fixtures build.
"""

from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.features import MOUD


def _make_agent(post_relapse_rr_demographic=None):
    enroll = {"w": SimpleNamespace(start=0, stop=999, prob=0.1)}
    demographic_moud = SimpleNamespace(enroll=enroll)
    if post_relapse_rr_demographic is not None:
        demographic_moud.post_relapse_rr = post_relapse_rr_demographic

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
        recent_relapse=True,
        location=SimpleNamespace(params=SimpleNamespace(demographics=demographics)),
    )
    agent.moud = MOUD(agent)
    return agent


def _make_model(moud_params):
    return SimpleNamespace(
        time=0,
        params=SimpleNamespace(
            moud=moud_params,
            features=SimpleNamespace(care_seeking=False),
        ),
        run_random=FakeRandom(0.14),
    )


@pytest.fixture(autouse=True)
def _reset_counts():
    MOUD.counts = {"black": {"MSM": 0}}
    yield


@pytest.mark.unit
def test_post_relapse_rr_top_level_override_applied():
    """
    moud.post_relapse_rr must take the same top-level-override-first
    precedence as post_overdose_rr/reinit_prob/discontinue_per_month.
    Previously it read only the demographic-level fallback, which
    demographics.yml never defines - a permanent no-op regardless of the
    configured value (CODEX_DEBUG_HANDOFF.md Priority 6 / MOUD cleanup).

    Baseline enrollment prob is 0.1 (from the enroll window); a 1.5x
    post-relapse multiplier pushes it to 0.15, which the fixed draw of
    0.14 clears - proving the top-level 1.5 was actually applied, not
    silently dropped to a 1.0 no-op.
    """
    agent = _make_agent()
    model = _make_model(
        SimpleNamespace(start_time=0, use_reinit=True, post_relapse_rr=1.5)
    )

    agent.moud.update_agent(model)

    assert agent.moud.active is True


@pytest.mark.unit
def test_post_relapse_rr_falls_back_to_demographic_when_no_top_level():
    """When the top-level override isn't present, the demographic-level
    value (or its 1.0 getattr default) is used instead - the baseline 0.1
    stays unmultiplied, so the same 0.14 draw does NOT clear it."""
    agent = _make_agent()  # no post_relapse_rr on the demographic stub either
    model = _make_model(SimpleNamespace(start_time=0, use_reinit=True))

    agent.moud.update_agent(model)

    assert agent.moud.active is False


# ---------------------------------------------------------------------------
# init_at_equilibrium (EPIDEMIC_DYNAMICS_MAP_2026-07-29.md, P6)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "reinit, discontinue, expected",
    [
        (0.286107204685432, 0.10379237530992133, 0.7338),  # the fitted pair
        (0.10, 0.10, 0.5),
        (0.27, 0.06, 0.8182),
    ],
)
def test_equilibrium_coverage_matches_reinit_discontinue_ratio(
    reinit, discontinue, expected
):
    """
    Coverage settles at reinit/(reinit+discontinue) regardless of the t=0
    placement, so `init_coverage` could not influence any objective scored after
    burn-in: a fitted 0.502 produced an observed 0.749 against this formula's
    0.734. init_at_equilibrium starts the run there instead.
    """
    moud_global = SimpleNamespace(
        init_at_equilibrium=True,
        use_reinit=True,
        reinit_prob=reinit,
        discontinue_per_month=discontinue,
    )
    got = MOUD._equilibrium_coverage(moud_global)
    assert got == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "moud_global",
    [
        None,
        SimpleNamespace(init_at_equilibrium=False, use_reinit=True,
                        reinit_prob=0.2, discontinue_per_month=0.1),
        # opting in without reinit enabled: there is no equilibrium to derive
        SimpleNamespace(init_at_equilibrium=True, use_reinit=False,
                        reinit_prob=0.2, discontinue_per_month=0.1),
        # the base calibration template leaves reinit_prob for Optuna to inject
        SimpleNamespace(init_at_equilibrium=True, use_reinit=True,
                        reinit_prob=0.0, discontinue_per_month=0.078),
        SimpleNamespace(init_at_equilibrium=True, use_reinit=True,
                        reinit_prob=0.2, discontinue_per_month=0.0),
    ],
)
def test_equilibrium_coverage_falls_back_when_not_derivable(moud_global):
    """Returns None so init_agent drops through to init_coverage / demographic
    moud.init, rather than silently placing nobody or everybody."""
    assert MOUD._equilibrium_coverage(moud_global) is None
