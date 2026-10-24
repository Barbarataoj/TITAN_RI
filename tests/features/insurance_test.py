"""
Insurance as agent state (2026-10-24, work record §19.34). Hand-rolled harness as in moud_test.py: a
SimpleNamespace agent, FakeRandom for the draws.
"""
from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.features import Insurance, CareSeeking, base_feature


def _params(**ins):
    base = dict(init_uninsured_prob_inj=0.0, init_uninsured_prob_noninj=0.0, gain_prob_per_month=0.0,
                moud_acceptance_mult=1.0, prep_acceptance_mult=1.0)
    base.update(ins)
    return SimpleNamespace(insurance=SimpleNamespace(**base), features=SimpleNamespace(insurance=True))


def _agent(drug_type="Inj", params=None):
    a = SimpleNamespace(drug_type=drug_type, location=SimpleNamespace(params=params or _params()))
    a.insurance = Insurance(a)
    return a


def test_default_is_insured_and_both_attributes_move_together():
    a = _agent()
    assert a.insured is True and a.insured_state == "insured"      # set in __init__, before any init_agent
    a.insurance.set_status(False)
    assert a.insured is False and a.insured_state == "uninsured"
    # reported through the insured_states output class, never as insured-only copies of other stats
    assert Insurance.stats == []


def test_hooks_are_neutral_when_the_feature_is_off():
    a = _agent()
    off = SimpleNamespace(features=SimpleNamespace(insurance=False), insurance=SimpleNamespace(prep_acceptance_mult=0.3))
    a.insurance.set_status(False)
    assert Insurance.acceptance_mult(a, off, "prep") == 1.0


def test_init_draw_by_drug_type():
    p = _params(init_uninsured_prob_inj=0.12, init_uninsured_prob_noninj=0.06)
    lo, hi = SimpleNamespace(pop_random=FakeRandom(0.05)), SimpleNamespace(pop_random=FakeRandom(0.5))
    a = _agent("Inj", p); a.insurance.init_agent(lo, 0); assert a.insured is False and a.insured_state == "uninsured" and a.insurance.active
    a = _agent("Inj", p); a.insurance.init_agent(hi, 0); assert a.insured is True
    a = _agent("NonInj", p); a.insurance.init_agent(lo, 0); assert a.insured is False      # 0.05 < 0.06
    a = _agent("NonInj", p); a.insurance.init_agent(SimpleNamespace(pop_random=FakeRandom(0.07)), 0); assert a.insured is True


def test_loss_probability_is_derived_from_gain_and_share():
    assert Insurance.lose_prob(0.06, 0.12) == pytest.approx(0.06 * 0.12 / 0.88)
    assert Insurance.lose_prob(0.06, 0.0) == 0.0 and Insurance.lose_prob(0.0, 0.12) == 0.0
    assert Insurance.lose_prob(0.06, 1.0) == 1.0
    for u in (0.06, 0.12, 0.24):        # stationarity: expected flow out of coverage equals the flow back in
        assert (1 - u) * Insurance.lose_prob(0.06, u) == pytest.approx(u * 0.06)


def test_churn_runs_for_every_agent_in_update_pop_and_precedes_the_visit_roll():
    p = _params(init_uninsured_prob_inj=0.12, init_uninsured_prob_noninj=0.06, gain_prob_per_month=0.06)
    a, b, c = _agent("Inj", p), _agent("NonInj", p), _agent("Inj", p)
    c.insurance.set_status(False)
    model = SimpleNamespace(params=p, run_random=FakeRandom(0.005), pop=SimpleNamespace(all_agents=[a, b, c]))
    Insurance.update_pop(model)
    assert a.insured is False       # 0.005 < 0.06 x 0.12 / 0.88 = 0.0082: the insured injector loses coverage
    assert b.insured is True        # 0.005 >= 0.06 x 0.06 / 0.94 = 0.0038: the non-injector keeps it
    assert c.insured is True        # 0.005 < 0.06: the uninsured one regains it
    # the agent-level hook does nothing: churn is population-level so that it precedes CareSeeking.update_pop
    a.insurance.update_agent(model)
    assert a.insured is False
    subs = base_feature.BaseFeature.__subclasses__()
    assert subs.index(Insurance) < subs.index(CareSeeking)


def test_no_draws_when_every_probability_is_zero():
    """The identity run (feature on, nobody uninsured) must not consume a random number."""
    class Counting:
        n = 0
        def random(self):
            self.n += 1
            return 0.5
    p = _params()
    a = _agent("Inj", p)
    rng = Counting()
    a.insurance.init_agent(SimpleNamespace(pop_random=rng), 0)
    Insurance.update_pop(SimpleNamespace(params=p, run_random=rng, pop=SimpleNamespace(all_agents=[a])))
    assert rng.n == 0 and a.insured is True


def test_acceptance_mult_only_for_uninsured():
    p = _params(moud_acceptance_mult=0.5, prep_acceptance_mult=0.3)
    a = _agent("Inj", p)
    assert Insurance.acceptance_mult(a, p, "moud") == 1.0 and Insurance.acceptance_mult(a, p, "prep") == 1.0
    a.insurance.set_status(False)
    assert Insurance.acceptance_mult(a, p, "moud") == 0.5 and Insurance.acceptance_mult(a, p, "prep") == 0.3


def test_visit_rate_uninsured_multiplier():
    setting = SimpleNamespace(visit_rate_annual=12.0, visit_mult_uninsured=0.35, visit_mult_moud=1.0,
                              visit_mult_inj=1.0, visit_mult_hiv_positive=1.0)
    a = SimpleNamespace(drug_type="NonInj", insured=True, hiv=SimpleNamespace(active=False), moud=SimpleNamespace(active=False),
                        location=SimpleNamespace(params=SimpleNamespace(care_seeking=SimpleNamespace(setting={"primary_care": setting}))))
    cs = CareSeeking(a); cs.visit_frailty = {"primary_care": 1.0}
    insured_rate = cs.visit_rate(setting, "primary_care", 12)
    a.insured = False
    assert cs.visit_rate(setting, "primary_care", 12) == pytest.approx(0.35 * insured_rate)


def test_art_multipliers_are_relative_and_mean_preserving(monkeypatch):
    """ART hooks (Jun, 2026-10-24): the uninsured start ART less and lapse more, with the average among the
    diagnosed (the agents the hooks act on) kept."""
    p = _params(init_uninsured_prob_inj=0.1)
    p.insurance.art_start_mult = 0.8
    p.insurance.art_nonadherence_mult = 1.15
    monkeypatch.setattr(Insurance, "uninsured_share_dx", 0.1)
    ins, un = _agent("Inj", p), _agent("Inj", p)
    un.insurance.set_status(False)
    norm = 0.9 + 0.1 * 0.8
    assert Insurance.art_mult(ins, p, "art_start") == pytest.approx(1 / norm)
    assert Insurance.art_mult(un, p, "art_start") == pytest.approx(0.8 / norm)
    assert 0.9 * Insurance.art_mult(ins, p, "art_start") + 0.1 * Insurance.art_mult(un, p, "art_start") == pytest.approx(1.0)
    assert Insurance.art_mult(un, p, "art_nonadherence") == pytest.approx(1.15 / (0.9 + 0.1 * 1.15))
    assert Insurance.art_mult(un, SimpleNamespace(features=SimpleNamespace(insurance=False), insurance=p.insurance), "art_start") == 1.0
