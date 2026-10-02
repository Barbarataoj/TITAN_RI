"""
The patient-facing programme fields of the care layer (`reach_*`, 2026-10-02:
a pre-visit chatbot such as CHIA) and the PrEP hooks that consume them. Same
hand-rolled harness as care_seeking_test.py.
"""
from types import SimpleNamespace

import pytest
from conftest import FakeRandom

from titan.features import CareSeeking
from titan.parse_params import ObjMap


DEFAULT_SETTING = {
    "visit_rate_annual": 0.0, "n_providers_ri": 0, "panel_skew": 1.0, "visit_heterogeneity_shape": 0.0,
    "attrition_annual": 0.0, "p_willing_hiv_test": 0.0, "p_willing_prep": 0.0, "p_willing_moud": 0.0,
    "p_willing_counseling": 0.0, "intensity_hiv_test": 0.0, "intensity_prep": 0.0, "intensity_moud": 0.0,
    "intensity_counseling": 0.0, "visit_mult_moud": 1.0, "visit_mult_inj": 1.0, "visit_mult_hiv_positive": 1.0,
}


def make_setting(**overrides):
    setting = dict(DEFAULT_SETTING)
    setting.update(overrides)
    return ObjMap(setting)


def make_params(settings, n_ri_oud=1, num_pop=1, shared_trait_prob=0.0, reassign_on_attrition=True, steps_per_year=12):
    return ObjMap({
        "care_seeking": {"n_ri_oud": n_ri_oud, "shared_trait_prob": shared_trait_prob,
                         "reassign_on_attrition": reassign_on_attrition,
                         "setting": {name: dict(setting) for name, setting in settings.items()}},
        "model": {"num_pop": num_pop, "time": {"steps_per_year": steps_per_year}},
    })


def make_fake_agent(params, drug_type="None", hiv_active=False, on_moud=False):
    fake_agent = SimpleNamespace(drug_type=drug_type, location=SimpleNamespace(params=params),
                                 hiv=SimpleNamespace(active=hiv_active), on_moud=on_moud)
    fake_agent.care_seeking = CareSeeking(fake_agent)
    return fake_agent


class _AgentList(list):
    def num_members(self):
        return len(self)


def make_fake_model(params, run_random, np_random, agents):
    return SimpleNamespace(params=params, run_random=run_random, np_random=np_random, time=0,
                           pop=SimpleNamespace(all_agents=_AgentList(agents), pop_random=run_random))


@pytest.fixture(autouse=True)
def _reset_pool():
    CareSeeking.providers = {}
    CareSeeking.provider_weights = {}
    CareSeeking.provider_shares = {}
    CareSeeking.pool_draw_p = {}
    yield
    CareSeeking.providers = {}
    CareSeeking.provider_weights = {}
    CareSeeking.provider_shares = {}
    CareSeeking.pool_draw_p = {}


def _pc(**kw):
    base = dict(visit_rate_annual=12.0, n_providers_ri=1, p_willing_hiv_test=0.0,
                intensity_hiv_test=0.0, p_willing_prep=1.0, intensity_prep=0.2)
    base.update(kw)
    return make_setting(**base)


@pytest.mark.unit
def test_reach_is_drawn_at_first_visit_after_start_and_visits_become_active():
    params = make_params({"primary_care": _pc(reach_prob=1.0, reach_start_time=10,
                                              reach_visit_prob=1.0, reach_offer_prob_hiv_test=1.0)})
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(1.0), [a])
    model.time = 9
    CareSeeking.update_pop(model)
    assert not a.care_seeking.reached.get("primary_care", False)
    assert a.care_offer["hiv_test"] is False          # provider never offers (W 0)
    model.time = 10
    CareSeeking.update_pop(model)
    assert a.care_seeking.reached["primary_care"] is True
    assert a.care_reach_visits == 1
    assert a.care_offer["hiv_test"] is True           # the programme's offer override
    assert a.care_offer_by_setting["primary_care"]["hiv_test"] is True


@pytest.mark.unit
def test_unreached_agent_is_not_retried_unless_retry_prob_set():
    params = make_params({"primary_care": _pc(reach_prob=0.3, reach_retry_prob=0.0, reach_offer_prob_hiv_test=1.0)})
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(1.0), [a])   # 0.5 >= 0.3: not reached
    for t in range(3):
        model.time = t
        CareSeeking.update_pop(model)
        assert not a.care_seeking.reached.get("primary_care", False)
        assert a.care_offer["hiv_test"] is False
    params.care_seeking.setting.primary_care["reach_retry_prob"] = 0.9
    model.time = 3
    CareSeeking.update_pop(model)
    assert a.care_seeking.reached["primary_care"] is True


@pytest.mark.unit
def test_active_visit_scales_intensity_and_acceptance_multiplier():
    params = make_params({"primary_care": _pc(reach_prob=1.0, reach_intensity_mult_prep=3.0,
                                              reach_acceptance_mult_prep=1.5, acceptance_mult_prep=1.2)})
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    # draw 0.5: provider intensity 0.2 alone would not offer; x3 = 0.6 does
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(1.0), [a])
    CareSeeking.update_pop(model)
    assert a.care_offer["prep"] is True
    assert a.care_offer_mult["prep"] == pytest.approx(1.2 * 1.5)
    # an unreached agent at the same draw gets no offer
    params2 = make_params({"primary_care": _pc(reach_prob=0.0, reach_intensity_mult_prep=3.0)})
    b = make_fake_agent(params2)
    CareSeeking.providers = {}
    b.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model2 = make_fake_model(params2, FakeRandom(0.5), FakeRandom(1.0), [b])
    CareSeeking.update_pop(model2)
    assert b.care_offer["prep"] is False


@pytest.mark.unit
def test_reach_effects_published_for_prep():
    params = make_params({"primary_care": _pc(reach_prob=1.0, reach_prep_discontinue_mult=0.6,
                                              reach_prep_adherence_mult=1.2, reach_prep_restart_prob=0.02,
                                              reach_prep_monitoring_mult=1.3)})
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.0), FakeRandom(1.0), [a])
    CareSeeking.update_pop(model)                      # reach drawn this step
    assert a.care_reach == {"prep_discontinue_mult": 1.0, "prep_adherence_mult": 1.0,
                            "prep_restart_prob": 0.0, "prep_monitoring_mult": 1.0}
    CareSeeking.update_pop(model)                      # effects visible from the next step
    assert a.care_reach["prep_discontinue_mult"] == pytest.approx(0.6)
    assert a.care_reach["prep_adherence_mult"] == pytest.approx(1.2)
    assert a.care_reach["prep_restart_prob"] == pytest.approx(0.02)
    assert a.care_reach["prep_monitoring_mult"] == pytest.approx(1.3)


@pytest.mark.unit
def test_prep_monitoring_visit_is_scheduled_with_a_guaranteed_test():
    params = make_params({"primary_care": _pc(visit_rate_annual=1.2)})   # routine visits round to 0
    params["prep"] = ObjMap({"monitoring_interval_steps": 3, "monitoring_attendance": 1.0,
                             "monitoring_setting": "primary_care"})
    a = make_fake_agent(params)
    a.prep = SimpleNamespace(active=True, type="Oral", time=4)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.5), FakeRandom(0.0), [a])
    model.time = 6                                      # 2 steps after start: not due
    CareSeeking.update_pop(model)
    assert "primary_care" not in a.care_encounter
    model.time = 7                                      # 3 steps after start: due
    CareSeeking.update_pop(model)
    assert a.care_encounter["primary_care"] == 1
    assert a.prep_monitoring_visit is True
    assert a.care_offer["hiv_test"] is True
    assert a.care_offer_mult["hiv_test"] >= 1e6        # accepted with probability 1
    model.time = 10
    a.prep.active = False
    CareSeeking.update_pop(model)
    assert "primary_care" not in a.care_encounter      # off PrEP: no monitoring


@pytest.mark.unit
def test_prep_offer_floor_applies_to_non_willing_providers_at_active_visits():
    """A prepared patient's request gets a non-prescribing provider to raise PrEP (reach_offer_prob_prep)."""
    params = make_params({"primary_care": _pc(p_willing_prep=0.0, intensity_prep=0.0,
                                              reach_prob=1.0, reach_offer_prob_prep=0.3)})
    a = make_fake_agent(params)
    a.care_seeking.init_agent(SimpleNamespace(pop_random=FakeRandom(0.0)), 0)
    model = make_fake_model(params, FakeRandom(0.2), FakeRandom(1.0), [a])   # 0.2 < 0.3
    CareSeeking.update_pop(model)
    assert a.care_offer["prep"] is True
    model.run_random = FakeRandom(0.5)                                        # 0.5 >= 0.3 (and reach already drawn)
    CareSeeking.update_pop(model)
    assert a.care_offer["prep"] is False
