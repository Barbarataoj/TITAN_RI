"""
Hand-rolled SimpleNamespace agent/model harness, same reason as
tests/features/overdose_test.py's module docstring: OUDState reads only its own
params block plus a few plain agent attributes, so the harness keeps these tests
independent of the demographics/classes machinery the shared fixtures build.
"""

from types import SimpleNamespace

import pytest

from titan.features import OUDState


def _make_agent(window_months):
    agent = SimpleNamespace(
        location=SimpleNamespace(
            params=SimpleNamespace(
                oud_state=SimpleNamespace(
                    transitions=SimpleNamespace(
                        relapse_risk_window_months=window_months
                    )
                )
            )
        )
    )
    agent.oud_state = OUDState(agent)
    return agent


def _make_model(agents, steps_per_year=12):
    return SimpleNamespace(
        pop=SimpleNamespace(all_agents=agents),
        params=SimpleNamespace(model=SimpleNamespace(time=SimpleNamespace(steps_per_year=steps_per_year))),
    )


@pytest.mark.unit
def test_recent_relapse_window_boundary():
    """
    A relapse sets `recent_relapse` for exactly as many `update_pop` calls as
    `relapse_risk_window_months` converts to in timesteps, then clears it - not
    window + 1.
    `OUDState.update_pop` is where the window is now advanced/expired
    (moved out of this feature's own `update_agent`), so that
    `moud`/`hiv_testing` - declared earlier in
    `titan/features/__init__.py` and so reading this flag before
    `oud_state` runs within a timestep - see this step's expiry rather
    than a stale value carried over from the step before
    (CODEX_DEBUG_HANDOFF.md Priority 5).
    """
    # 8 months at 12 steps/yr == 8 timesteps
    a = _make_agent(8.0)
    a.recent_relapse = True
    a.relapse_timer = 0

    model = _make_model([a], steps_per_year=12)
    for step in range(1, 8 + 1):
        OUDState.update_pop(model)
        assert a.recent_relapse is True, f"expected True at step {step} (window=8)"

    OUDState.update_pop(model)
    assert a.recent_relapse is False
    assert a.relapse_timer == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "window_months, steps_per_year, expected_steps",
    [
        (2.0, 12, 2),    # ~8 weeks on monthly steps - the intended default
        (2.0, 52, 8),    # same 2 months on weekly steps is 8.67 -> expires after 8
        (6.0, 12, 6),
    ],
)
def test_relapse_risk_window_is_months_not_timesteps(
    window_months, steps_per_year, expected_steps
):
    """
    The window is denominated in MONTHS and converted with steps_per_year.

    It used to be read as raw timesteps while defaulting to 8 - a value chosen
    for weekly steps - so on this model's monthly steps the tolerance-loss
    window ran ~4x too long (8 months rather than ~8 weeks). The feature's own
    fallback was `steps_per_year // 6`, disagreeing with that default by the
    same factor, which is what gave the units bug away.
    See EPIDEMIC_DYNAMICS_MAP_2026-07-29.md finding P7.
    """
    a = _make_agent(window_months)
    a.recent_relapse = True
    a.relapse_timer = 0

    model = _make_model([a], steps_per_year=steps_per_year)
    for step in range(1, expected_steps + 1):
        OUDState.update_pop(model)
        assert a.recent_relapse is True, (
            f"expected still flagged at step {step} "
            f"({window_months} months at {steps_per_year} steps/yr)"
        )

    OUDState.update_pop(model)
    assert a.recent_relapse is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "legacy_key, replacement",
    [
        ("early_relapse_window", "early_relapse_window_months"),
        ("relapse_risk_window", "relapse_risk_window_months"),
    ],
)
def test_deprecated_raw_timestep_windows_raise(legacy_key, replacement):
    """
    Both windows were given weekly-step values on a monthly clock, so a params
    file still setting the raw-timestep key would silently apply a ~4x-too-long
    window. Refuse to run instead, same as hiv_testing.moud_early_window.
    """
    params = SimpleNamespace(
        oud_state=SimpleNamespace(transitions=SimpleNamespace(**{legacy_key: 26})),
        classes=SimpleNamespace(bond_types={}),
        demographics={},
    )
    with pytest.raises(ValueError, match=f"{legacy_key} is deprecated"):
        OUDState.init_class(params)


@pytest.mark.unit
def test_months_windows_do_not_raise():
    """The replacement keys are accepted, and the -1 legacy sentinel is ignored."""
    params = SimpleNamespace(
        oud_state=SimpleNamespace(
            transitions=SimpleNamespace(
                early_relapse_window_months=6.0,
                early_relapse_window=-1,
                relapse_risk_window_months=2.0,
                relapse_risk_window=-1,
            )
        ),
        classes=SimpleNamespace(bond_types={}),
        demographics={},
    )
    OUDState.init_class(params)  # must not raise


@pytest.mark.unit
def test_update_pop_skips_inactive_and_unflagged_agents():
    window = 8
    inactive = _make_agent(window)
    inactive.oud_state.active = False
    inactive.recent_relapse = True
    inactive.relapse_timer = 999

    unflagged = _make_agent(window)
    unflagged.recent_relapse = False
    unflagged.relapse_timer = 0

    model = _make_model([inactive, unflagged])
    OUDState.update_pop(model)  # must not raise, must not touch either agent

    assert inactive.relapse_timer == 999
    assert unflagged.recent_relapse is False


def _make_lookback_agent(drug_type="Inj", steps_per_year=12):
    agent = SimpleNamespace(
        drug_type=drug_type,
        has_oud=True,
        is_in_remission=False,
        location=SimpleNamespace(
            params=SimpleNamespace(
                model=SimpleNamespace(
                    time=SimpleNamespace(steps_per_year=steps_per_year)
                )
            )
        ),
    )
    agent.hiv = SimpleNamespace(active=False)
    agent.oud_state = OUDState(agent)
    return agent


def _empty_stats():
    return {k: 0 for k in OUDState.stats}


@pytest.mark.unit
def test_pwid_lookback_counts_recent_injector_after_remission():
    """
    `pwid_recent` is the claims-year PWID denominator: an agent stays in it for
    steps_per_year steps after they last injected, even though oud_state flips
    drug_type to NonInj the moment they enter remission.

    This is the denominator the APCD prevalence/incidence targets are actually
    built over, so it is reported alongside the instantaneous drug_type == "Inj"
    quantity the objective scores. See OUDState.set_stats.
    """
    agent = _make_lookback_agent()
    agent.last_injection_time = 0
    agent.drug_type = "NonInj"  # entered remission at t=1

    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 6)
    assert stats["pwid_recent"] == 1, "6 months after last injection: still in"

    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 11)
    assert stats["pwid_recent"] == 1, "11 months: last step inside the window"

    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 12)
    assert stats["pwid_recent"] == 0, "12 months: window has closed"


@pytest.mark.unit
def test_pwid_lookback_never_injected_excluded():
    """An agent with no injection history is never in the look-back denominator."""
    agent = _make_lookback_agent(drug_type="NonInj")
    agent.last_injection_time = None

    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 3)
    assert stats["pwid_recent"] == 0
    assert stats["pwid_recent_hiv"] == 0


@pytest.mark.unit
def test_pwid_lookback_hiv_subcount():
    """`pwid_recent_hiv` is the HIV+ numerator over the same look-back denominator."""
    agent = _make_lookback_agent()
    agent.last_injection_time = 2
    agent.hiv.active = True

    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 5)
    assert stats["pwid_recent"] == 1
    assert stats["pwid_recent_hiv"] == 1

    agent.hiv.active = False
    stats = _empty_stats()
    agent.oud_state.set_stats(stats, 5)
    assert stats["pwid_recent"] == 1
    assert stats["pwid_recent_hiv"] == 0
