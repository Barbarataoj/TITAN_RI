"""
Tests for CareSeeking build a minimal, hand-rolled harness (ObjMap params,
SimpleNamespace agent/model) instead of the shared `params`/`make_agent`/
`make_model`/`make_population` fixtures. CareSeeking only ever reads
`params.care_seeking`, `params.model.num_pop`, `params.model.time.steps_per_year`,
and a handful of plain attributes off `agent`/`model` - none of the demographics/
classes machinery those shared fixtures pull in - so this sidesteps the fixtures
cleanly without needing anything to be reset if it changes.
"""

from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.features import CareSeeking
from titan.parse_params import ObjMap


DEFAULT_SETTING = {
    "visit_rate_annual": 0.0,
    "n_providers_ri": 0,
    "panel_skew": 1.0,
    "visit_heterogeneity_shape": 0.0,
    "attrition_annual": 0.0,
    "p_willing_hiv_test": 0.0,
    "p_willing_prep": 0.0,
    "p_willing_moud": 0.0,
    "p_willing_counseling": 0.0,
    "intensity_hiv_test": 0.0,
    "intensity_prep": 0.0,
    "intensity_moud": 0.0,
    "intensity_counseling": 0.0,
    "visit_mult_moud": 1.0,
    "visit_mult_inj": 1.0,
    "visit_mult_hiv_positive": 1.0,
}


def make_setting(**overrides):
    setting = dict(DEFAULT_SETTING)
    setting.update(overrides)
    return ObjMap(setting)


def make_params(
    settings,
    n_ri_oud=1,
    num_pop=1,
    shared_trait_prob=0.0,
    reassign_on_attrition=True,
    steps_per_year=12,
):
    return ObjMap(
        {
            "care_seeking": {
                "n_ri_oud": n_ri_oud,
                "shared_trait_prob": shared_trait_prob,
                "reassign_on_attrition": reassign_on_attrition,
                "setting": {name: dict(setting) for name, setting in settings.items()},
            },
            "model": {"num_pop": num_pop, "time": {"steps_per_year": steps_per_year}},
        }
    )


def make_fake_agent(params, drug_type="None", hiv_active=False, on_moud=False):
    fake_agent = SimpleNamespace(
        drug_type=drug_type,
        location=SimpleNamespace(params=params),
        hiv=SimpleNamespace(active=hiv_active),
        on_moud=on_moud,
    )
    fake_agent.care_seeking = CareSeeking(fake_agent)
    return fake_agent


class _AgentList(list):
    """
    Minimal stand-in for titan's AgentSet: iterable, plus the num_members()
    accessor that CareSeeking.ensure_initialized uses to size the provider pool.
    """

    def num_members(self):
        return len(self)


def make_fake_model(params, run_random, np_random, agents):
    return SimpleNamespace(
        params=params,
        run_random=run_random,
        np_random=np_random,
        time=0,
        pop=SimpleNamespace(
            all_agents=_AgentList(agents),
            pop_random=run_random,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_care_seeking_pool():
    CareSeeking.providers = {}
    CareSeeking.provider_weights = {}
    CareSeeking.provider_shares = {}
    CareSeeking.pool_draw_p = {}
    yield
    CareSeeking.providers = {}
    CareSeeking.provider_weights = {}
    CareSeeking.provider_shares = {}
    CareSeeking.pool_draw_p = {}


@pytest.mark.unit
def test_update_pop_initializes_agents_that_never_ran_init_agent():
    """
    A population restored from disk is assembled with Population.add_agent, which
    never calls init_agent -- so the provider pool stayed empty, every agent
    stayed inactive, and because all three consumers read offers through
    getattr(..., {}) the feature silently produced no PrEP, no MOUD and no HIV
    testing while the report still looked plausible.
    """
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_hiv_test=1.0,
                intensity_hiv_test=1.0,
            )
        }
    )
    a = make_fake_agent(params)
    # deliberately do NOT call init_agent -- this is the loaded-population state
    assert not a.care_seeking.active
    assert not CareSeeking.providers

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.update_pop(model)

    assert a.care_seeking.active, "agent was never attached to a provider"
    assert sum(len(v) for v in CareSeeking.providers.values()) > 0
    assert a.care_offer["hiv_test"] is True


@pytest.mark.unit
def test_build_pool_uses_explicit_num_pop_over_stale_params():
    """
    population_io sets model.num_pop to 0 before building a loaded Population and
    that 0 is deep-copied into every Location's params, so scaling the provider
    pool off params.model.num_pop collapsed every pool to empty.
    """
    params = make_params(
        {"primary_care": make_setting(visit_rate_annual=12.0, n_providers_ri=1000)}
    )
    params.model.num_pop = 0  # the stale value population_io leaves behind

    CareSeeking.build_pool(params, FakeRandom(0.0), num_pop=5000)

    assert len(CareSeeking.providers["primary_care"]) > 0


@pytest.mark.unit
def test_current_step_offer_consumption():
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_hiv_test=1.0,
                intensity_hiv_test=1.0,
            )
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    assert a.care_seeking.active

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.update_pop(model)
    assert a.care_offer["hiv_test"] is True
    assert a.care_encounter["primary_care"] == 1

    # a later step with no offer must overwrite, not retain, the prior step's
    # data - proving update_pop refreshes care_offer every step, so a
    # downstream feature never reads a stale value from before care_seeking ran
    model.run_random = FakeRandom(1.1)
    CareSeeking.update_pop(model)
    assert a.care_offer["hiv_test"] is False


@pytest.mark.unit
def test_persistent_provider_attachment():
    params = make_params(
        {"primary_care": make_setting(visit_rate_annual=12.0, n_providers_ri=5)}
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.6)), 0)
    provider_idx = a.care_seeking.provider_id["primary_care"]

    model = make_fake_model(params, FakeRandom(1.1), FakeRandom(1.0), [a])
    for _ in range(3):
        CareSeeking.resolve_all_visits(params.care_seeking, model)
        assert a.care_seeking.provider_id["primary_care"] == provider_idx


@pytest.mark.unit
def test_setting_provenance():
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_moud=1.0,
                intensity_moud=1.0,
            ),
            "emergency_department": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_hiv_test=1.0,
                intensity_hiv_test=1.0,
            ),
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.resolve_all_visits(params.care_seeking, model)

    assert a.care_offer_by_setting["primary_care"]["moud"] is True
    assert a.care_offer_by_setting["primary_care"]["hiv_test"] is False
    assert a.care_offer_by_setting["emergency_department"]["hiv_test"] is True
    assert a.care_offer_by_setting["emergency_department"]["moud"] is False
    # the collapsed view still reflects both settings' offers
    assert a.care_offer["moud"] is True
    assert a.care_offer["hiv_test"] is True


@pytest.mark.unit
def test_pcp_and_ed_moud_availability():
    """MOUD (via buprenorphine) is available from primary care and the ED, not
    only substance use treatment."""
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_moud=1.0,
                intensity_moud=1.0,
            ),
            "emergency_department": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_moud=1.0,
                intensity_moud=1.0,
            ),
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.resolve_all_visits(params.care_seeking, model)

    assert a.care_offer_by_setting["primary_care"]["moud"] is True
    assert a.care_offer_by_setting["emergency_department"]["moud"] is True


@pytest.mark.unit
def test_stratified_visit_rate_multiplier():
    setting = make_setting(
        visit_rate_annual=10.0,
        visit_mult_moud=2.0,
        visit_mult_inj=3.0,
        visit_mult_hiv_positive=5.0,
    )
    params = make_params({"primary_care": setting})

    baseline = make_fake_agent(params, drug_type="None", hiv_active=False, on_moud=False)
    assert baseline.care_seeking.stratified_multiplier(setting) == pytest.approx(1.0)

    on_moud = make_fake_agent(params, drug_type="None", hiv_active=False, on_moud=True)
    assert on_moud.care_seeking.stratified_multiplier(setting) == pytest.approx(2.0)

    injecting = make_fake_agent(params, drug_type="Inj", hiv_active=False, on_moud=False)
    assert injecting.care_seeking.stratified_multiplier(setting) == pytest.approx(3.0)

    all_three = make_fake_agent(params, drug_type="Inj", hiv_active=True, on_moud=True)
    assert all_three.care_seeking.stratified_multiplier(setting) == pytest.approx(
        2.0 * 3.0 * 5.0
    )


@pytest.mark.unit
def test_visit_heterogeneity_disabled_by_default():
    setting = make_setting(visit_rate_annual=10.0)
    params = make_params({"primary_care": setting})
    a = make_fake_agent(params)

    # visit_heterogeneity_shape defaults to 0.0: frailty is pinned to 1.0, so
    # the visit rate reduces to the homogeneous-Poisson case unchanged
    assert CareSeeking.draw_frailty(setting, FakeRandom(0.5)) == 1.0
    a.care_seeking.visit_frailty["primary_care"] = 1.0
    assert a.care_seeking.visit_rate(setting, "primary_care", 12) == pytest.approx(
        10.0 / 12
    )


@pytest.mark.unit
def test_independent_counseling_and_testing():
    params = make_params({"primary_care": make_setting()})
    a = make_fake_agent(params)
    a.care_seeking.provider_id["primary_care"] = 0
    a.care_offer = {"hiv_test": False, "prep": False, "moud": False, "counseling": False}
    a.care_offer_by_setting = {}
    CareSeeking.providers["primary_care"] = [
        {
            "hiv_test": {"willing": True, "intensity": 0.9},
            "prep": {"willing": False, "intensity": 0.0},
            "moud": {"willing": False, "intensity": 0.0},
            "counseling": {"willing": True, "intensity": 0.1},
        }
    ]

    model = SimpleNamespace(run_random=FakeRandom(0.5))
    a.care_seeking.resolve_offers("primary_care", 0, model)

    # draw (0.5) is below the hiv_test intensity (0.9, offered) but above the
    # counseling intensity (0.1, not offered) - each service resolves off its
    # own intensity, with no cross term between them
    assert a.care_offer["hiv_test"] is True
    assert a.care_offer["counseling"] is False


@pytest.mark.unit
def test_attrition_reassignment_on():
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=0.0, n_providers_ri=2, attrition_annual=0.5
            )
        },
        reassign_on_attrition=True,
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    old_idx = a.care_seeking.provider_id["primary_care"]
    old_provider = CareSeeking.providers["primary_care"][old_idx]

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.process_attrition(params.care_seeking, model)

    assert CareSeeking.providers["primary_care"][old_idx] is not old_provider
    assert a.care_seeking.provider_id["primary_care"] == 0


@pytest.mark.unit
def test_attrition_no_reassignment_keeps_slot():
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=0.0, n_providers_ri=2, attrition_annual=0.5
            )
        },
        reassign_on_attrition=False,
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    old_idx = a.care_seeking.provider_id["primary_care"]
    old_provider = CareSeeking.providers["primary_care"][old_idx]

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.process_attrition(params.care_seeking, model)

    # the slot's provider record changed (the clinician retired) but the
    # agent stays attached to the same index (the practice continues)
    assert CareSeeking.providers["primary_care"][old_idx] is not old_provider
    assert a.care_seeking.provider_id["primary_care"] == old_idx


@pytest.mark.unit
def test_pool_scaling_stochastic_rounding():
    # expected = 1 * 0.3 = 0.3 providers
    assert CareSeeking.scaled_provider_count(1, 0.3, FakeRandom(0.5)) == 0
    assert CareSeeking.scaled_provider_count(1, 0.3, FakeRandom(0.1)) == 1
    # expected exactly 2 - never rounds down or up regardless of the draw
    assert CareSeeking.scaled_provider_count(2, 1.0, FakeRandom(0.999)) == 2


@pytest.mark.unit
def test_disabled_path_equivalence():
    """With no providers anywhere, the feature must behave as if it were
    disabled: never active, never publishing an encounter, and downstream
    features' getattr(agent, "care_offer", {}).get(x, False) default holds."""
    params = make_params(
        {"primary_care": make_setting(visit_rate_annual=12.0, n_providers_ri=0)}
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)

    assert a.care_seeking.active is False
    assert a.care_offer == {
        "hiv_test": False,
        "prep": False,
        "moud": False,
        "counseling": False,
    }

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.resolve_all_visits(params.care_seeking, model)

    assert a.care_encounter == {}
    assert a.care_offer.get("hiv_test", False) is False


@pytest.mark.unit
def test_set_stats_reports_by_setting_and_service():
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=1,
                p_willing_hiv_test=1.0,
                intensity_hiv_test=1.0,
            )
        }
    )
    CareSeeking.init_class(params)

    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)

    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.resolve_all_visits(params.care_seeking, model)

    stats = {stat: 0 for stat in CareSeeking.stats}
    a.care_seeking.set_stats(stats, 0)

    assert stats["care_visits"] == 1
    assert stats["care_encounters_primary_care"] == 1
    assert stats["care_offer_primary_care_hiv_test"] == 1
    assert stats["care_offer_primary_care_counseling"] == 0


@pytest.mark.unit
def test_panel_concentration():
    CareSeeking.provider_shares["primary_care"] = [0.5, 0.5]
    assert CareSeeking.panel_concentration("primary_care") == pytest.approx(0.5)

    CareSeeking.provider_shares["primary_care"] = [1.0]
    assert CareSeeking.panel_concentration("primary_care") == pytest.approx(1.0)

    assert CareSeeking.panel_concentration("unknown_setting") is None


@pytest.mark.unit
def test_visit_after_overdose_links_one_ed_encounter():
    """
    `visit_prob_after_overdose` (2026-10-02): an agent whose nonfatal overdose
    happened last step gets one extra visit to the setting, with the provider's
    offers resolved on it, even when the routine Poisson draw gives no visit.
    The same overdose never yields a second extra visit, and an agent with no
    overdose gets nothing. This is the only link between an overdose and an
    emergency encounter while the care layer bypasses moud.post_overdose_rr.
    """
    params = make_params(
        {
            "ED": make_setting(
                visit_rate_annual=1.2,  # 0.1 per step -> FakeRandom.poisson rounds to 0
                n_providers_ri=1,
                p_willing_moud=1.0,
                intensity_moud=1.0,
                visit_prob_after_overdose=1.0,
            )
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(0.0), [a])

    # no overdose on record: no visit
    model.time = 5
    CareSeeking.update_pop(model)
    assert "ED" not in a.care_encounter
    assert a.care_offer["moud"] is False

    # overdose last step: exactly one visit, MOUD offered on it
    a.last_overdose_t = 5
    model.time = 6
    CareSeeking.update_pop(model)
    assert a.care_encounter["ED"] == 1
    assert a.care_offer["moud"] is True
    assert a.care_offer_by_setting["ED"]["moud"] is True

    # the same overdose is not rolled again, inside or outside the window
    model.time = 7
    CareSeeking.update_pop(model)
    assert "ED" not in a.care_encounter
    model.time = 6
    CareSeeking.update_pop(model)
    assert "ED" not in a.care_encounter

    # a new overdose links again
    a.last_overdose_t = 9
    model.time = 10
    CareSeeking.update_pop(model)
    assert a.care_encounter["ED"] == 1


@pytest.mark.unit
def test_visit_after_overdose_off_by_default():
    params = make_params(
        {"ED": make_setting(visit_rate_annual=1.2, n_providers_ri=1)}
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(0.0), [a])
    a.last_overdose_t = 5
    model.time = 6
    CareSeeking.update_pop(model)
    assert "ED" not in a.care_encounter


@pytest.mark.unit
def test_visit_after_overdose_survives_zero_routine_rate():
    """
    The post-overdose visit is rolled before the routine-rate guard, so a
    scenario that zeroes a setting's routine visits (a Phase 7 arm, say) keeps
    the overdose-to-emergency link.
    """
    params = make_params(
        {
            "ED": make_setting(
                visit_rate_annual=0.0,
                n_providers_ri=1,
                p_willing_moud=1.0,
                intensity_moud=1.0,
                visit_prob_after_overdose=1.0,
            )
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(0.0), [a])
    model.time = 3
    CareSeeking.update_pop(model)
    assert "ED" not in a.care_encounter
    a.last_overdose_t = 3
    model.time = 4
    CareSeeking.update_pop(model)
    assert a.care_encounter["ED"] == 1
    assert a.care_offer["moud"] is True


@pytest.mark.unit
def test_pool_refresh_applies_new_willingness_and_intensity_at_the_step():
    """
    `pool_refresh_time` (2026-10-02): provider records freeze willing/intensity
    at draw time, so a dated arm raises the parameters (timeline_scaling) and
    refreshes the pool at its start step. With p_willing 0 -> 1 every provider
    becomes willing; intensity follows the current parameter; nothing happens on
    other steps.
    """
    params = make_params(
        {
            "primary_care": make_setting(
                visit_rate_annual=12.0,
                n_providers_ri=10,
                p_willing_hiv_test=0.0,
                intensity_hiv_test=0.1,
                pool_refresh_time=5,
            )
        },
        n_ri_oud=10,
        num_pop=10,
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    pool = CareSeeking.providers["primary_care"]
    assert pool and not any(p["hiv_test"]["willing"] for p in pool)
    assert CareSeeking.pool_draw_p["primary_care"]["hiv_test"] == 0.0

    # the arm raises the parameters at its start step (what timeline_scaling does)
    params.care_seeking.setting.primary_care["p_willing_hiv_test"] = 1.0
    params.care_seeking.setting.primary_care["intensity_hiv_test"] = 0.9
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(1.0), [a])

    model.time = 4
    CareSeeking.update_pop(model)
    assert not any(p["hiv_test"]["willing"] for p in pool), "refresh must wait for its step"

    model.time = 5
    CareSeeking.update_pop(model)
    assert all(p["hiv_test"]["willing"] for p in pool)
    assert all(p["hiv_test"]["intensity"] == 0.9 for p in pool)
    assert CareSeeking.pool_draw_p["primary_care"]["hiv_test"] == 1.0
    # the agent's provider offered on this step's visit (rate 12/yr -> 1 visit)
    assert a.care_offer["hiv_test"] is True


@pytest.mark.unit
def test_pool_refresh_trims_willingness_when_the_parameter_falls():
    params = make_params(
        {"primary_care": make_setting(visit_rate_annual=12.0, n_providers_ri=10,
                                      p_willing_prep=1.0, intensity_prep=0.5,
                                      pool_refresh_time=3)},
        n_ri_oud=10, num_pop=10,
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    pool = CareSeeking.providers["primary_care"]
    assert all(p["prep"]["willing"] for p in pool)
    params.care_seeking.setting.primary_care["p_willing_prep"] = 0.0
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(1.0), [a])
    model.time = 3
    CareSeeking.update_pop(model)
    assert not any(p["prep"]["willing"] for p in pool)


@pytest.mark.unit
def test_acceptance_multiplier_travels_with_the_offer_and_takes_the_max():
    """
    `acceptance_mult_<service>` (2026-10-02): a patient-facing strategy in one
    setting raises acceptance only for offers made there; when two settings
    offer the same service in a step the larger multiplier wins; a setting
    without the field contributes 1.0.
    """
    params = make_params(
        {
            "primary_care": make_setting(visit_rate_annual=12.0, n_providers_ri=1,
                                         p_willing_hiv_test=1.0, intensity_hiv_test=1.0,
                                         acceptance_mult_hiv_test=1.5),
            "ED": make_setting(visit_rate_annual=12.0, n_providers_ri=1,
                               p_willing_hiv_test=1.0, intensity_hiv_test=1.0,
                               acceptance_mult_hiv_test=1.2, acceptance_mult_prep=2.0,
                               p_willing_prep=1.0, intensity_prep=1.0),
        }
    )
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.update_pop(model)
    assert a.care_offer["hiv_test"] is True and a.care_offer["prep"] is True
    assert a.care_offer_mult["hiv_test"] == pytest.approx(1.5)   # max(1.5 PC, 1.2 ED)
    assert a.care_offer_mult["prep"] == pytest.approx(2.0)       # offered only by the ED
    assert a.care_offer_mult["moud"] == 1.0
    # a step with no offer resets the multiplier
    model.run_random = FakeRandom(1.1)
    CareSeeking.update_pop(model)
    assert a.care_offer_mult["hiv_test"] == 1.0



@pytest.mark.unit
def test_roll_visits_publishes_active_visits_and_same_step_reach_effects(make_model, make_agent):
    """2026-10-13: care_active[setting] says whether this step's visit was an active
    programme visit; care_reach carries the programme's multipliers from the reaching step."""
    from titan.features import CareSeeking
    from conftest import FakeRandom
    model = make_model()
    model.time = 140
    model.params.features.care_seeking = False
    a = make_agent(DU="Inj")
    model.params.features.care_seeking = True
    fields = {"visit_rate_annual": 12.0, "n_providers_ri": 10, "panel_skew": 1.0, "visit_heterogeneity_shape": 0.0,
              "attrition_annual": 0.0, "p_willing_hiv_test": 0.0, "p_willing_prep": 0.0, "p_willing_moud": 0.0,
              "p_willing_counseling": 0.0, "intensity_hiv_test": 0.0, "intensity_prep": 0.0, "intensity_moud": 0.0,
              "intensity_counseling": 0.0, "visit_prob_after_overdose": 0.0, "pool_refresh_time": -9999,
              "reach_prob": 1.0, "reach_retry_prob": 0.0, "reach_visit_prob": 1.0, "reach_start_time": 100,
              "reach_prep_adherence_mult": 1.5, "reach_prep_discontinue_mult": 1.0, "reach_prep_restart_prob": 0.0,
              "reach_prep_monitoring_mult": 1.0}
    care = ObjMap({"shared_trait_prob": 0.0, "setting": {"primary_care": fields}, "n_ri_oud": 500})
    model.params.care_seeking = care
    a.care_seeking.provider_id = {"primary_care": 0}
    a.care_seeking.reached = {}; a.care_seeking.reach_rolled = {}
    CareSeeking.providers = {"primary_care": [{s: {"willing": False, "intensity": 0.0} for s in ("hiv_test", "prep", "moud", "counseling")}]}
    model.run_random = FakeRandom(0.0)
    class NP:
        def poisson(self, rate): return 1
    model.np_random = NP()
    a.care_seeking.roll_visits(care, 12, model)
    assert a.care_active == {"primary_care": True}
    assert a.care_seeking.reached == {"primary_care": True}
    assert a.care_reach["prep_adherence_mult"] == 1.5      # same step, not the next
    model.params.features.care_seeking = False
