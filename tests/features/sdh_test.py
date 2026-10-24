"""SDH as agent state (2026-10-24, work record §19.39): hardship index 0-3 on one latent propensity, sex work,
and the hooks the other features read. Hand-rolled harness as in insurance_test.py."""
import random
from types import SimpleNamespace

import pytest

from titan.features import SDH, CareSeeking, Insurance, base_feature


def _params(rho=0.0, q=0.0, inj=None, noninj=None, sw=None, mults=None, partner_mult=1.0, on=True):
    inj, noninj, sw, mults = inj or {}, noninj or {}, sw or {}, mults or {}
    comp = lambda d: SimpleNamespace(housing=d.get("housing", 0.0), food=d.get("food", 0.0), employment=d.get("employment", 0.0))
    hard = SimpleNamespace(rho=rho, episode_end_prob_per_month=q, prob_inj=comp(inj), prob_noninj=comp(noninj),
                           mult_per_point=SimpleNamespace(**{k: mults.get(k, 1.0) for k in
                                                             ("moud_discontinue", "prep_discontinue", "art_nonadherence", "art_readherence")}))
    prob = {st: SimpleNamespace(inj=sw.get(st, (0.0, 0.0))[0], noninj=sw.get(st, (0.0, 0.0))[1]) for st in ("HF", "HM", "MSM", "MSMW")}
    bonds = {"Sex": SimpleNamespace(acts_allowed=["sex"]), "Inj": SimpleNamespace(acts_allowed=["injection"])}
    return SimpleNamespace(sdh=SimpleNamespace(hardship=hard, sex_work=SimpleNamespace(prob=prob, partner_mult=partner_mult)),
                           features=SimpleNamespace(sdh=on), classes=SimpleNamespace(bond_types=bonds))


def _agent(drug_type="Inj", sex_type="HF", params=None, partners=2):
    inj = 1 if drug_type == "Inj" else 0
    a = SimpleNamespace(drug_type=drug_type, sex_type=sex_type, location=SimpleNamespace(params=params or _params()),
                        mean_num_partners={"Sex": partners, "Inj": inj}, target_partners={"Sex": partners, "Inj": inj})
    a.sdh = SDH(a)
    return a


class Pop:
    def __init__(self, seed):
        self.pop_random = random.Random(seed)


def test_defaults_are_neutral_and_draw_nothing():
    class Counting(random.Random):
        n = 0
        def gauss(self, *a):
            self.n += 1; return 0.0
        def random(self):
            self.n += 1; return 0.5
    a = _agent(); rng = Counting(1)
    a.sdh.init_agent(SimpleNamespace(pop_random=rng), 0)
    assert a.hardship == 0 and a.sex_work is False and rng.n == 0 and a.sdh.active
    SDH.update_pop(SimpleNamespace(params=_params(q=1 / 12), run_random=rng, pop=SimpleNamespace(all_agents=[a])))
    assert rng.n == 0
    assert SDH.mult(a, a.location.params, "moud_discontinue") == 1.0


def test_marginals_are_reproduced_and_components_correlate():
    p = _params(rho=0.35, inj={"housing": 0.69, "food": 0.60, "employment": 0.52}, sw={"HF": (0.31, 0.19)})
    pop = Pop(7); agents = [_agent("Inj", "HF", p) for _ in range(20000)]
    for a in agents:
        a.sdh.init_agent(pop, 0)
    share = lambda c: sum(a.sdh.components[c] for a in agents) / len(agents)
    assert share("housing") == pytest.approx(0.69, abs=0.012)
    assert share("food") == pytest.approx(0.60, abs=0.012)
    assert share("employment") == pytest.approx(0.52, abs=0.012)
    assert sum(a.sex_work for a in agents) / len(agents) == pytest.approx(0.31, abs=0.012)
    # the components correlate: unemployment is more common among the unstably housed (NHBS 2018: 57 % vs 34 %)
    h = [a for a in agents if a.sdh.components["housing"]]; nh = [a for a in agents if not a.sdh.components["housing"]]
    assert sum(a.sdh.components["employment"] for a in h) / len(h) > sum(a.sdh.components["employment"] for a in nh) / len(nh) + 0.1
    levels = [a.hardship for a in agents]
    assert set(levels) <= {0, 1, 2, 3} and sum(levels) / len(levels) == pytest.approx(0.69 + 0.60 + 0.52, abs=0.03)


def test_marginals_follow_drug_type_and_sex_type():
    p = _params(inj={"housing": 0.69}, noninj={"housing": 0.25}, sw={"HF": (0.31, 0.19), "HM": (0.08, 0.06)})
    assert SDH.marginals(_agent("Inj", "HF", p), p) == {"housing": 0.69, "food": 0.0, "employment": 0.0, "sex_work": 0.31}
    assert SDH.marginals(_agent("NonInj", "HM", p), p) == {"housing": 0.25, "food": 0.0, "employment": 0.0, "sex_work": 0.06}


def test_churn_moves_states_but_keeps_the_marginals():
    p = _params(rho=0.35, q=1 / 12, inj={"housing": 0.69, "food": 0.60, "employment": 0.52}, sw={"HF": (0.31, 0.19)})
    pop = Pop(3); agents = [_agent("Inj", "HF", p) for _ in range(20000)]
    for a in agents:
        a.sdh.init_agent(pop, 0)
    before = [a.hardship for a in agents]
    model = SimpleNamespace(params=p, run_random=random.Random(11), pop=SimpleNamespace(all_agents=agents))
    for _ in range(24):
        SDH.update_pop(model)
    after = [a.hardship for a in agents]
    assert sum(1 for b, c in zip(before, after) if b != c) > 0.3 * len(agents)
    assert sum(a.sdh.components["housing"] for a in agents) / len(agents) == pytest.approx(0.69, abs=0.015)
    assert sum(a.sex_work for a in agents) / len(agents) == pytest.approx(0.31, abs=0.015)


def test_multipliers_compound_per_point_and_vanish_when_off():
    p = _params(mults={"moud_discontinue": 1.2, "art_readherence": 0.85})
    a = _agent(params=p); a.hardship = 2
    assert SDH.mult(a, p, "moud_discontinue") == pytest.approx(1.44)
    assert SDH.mult(a, p, "art_readherence") == pytest.approx(0.85 ** 2)
    assert SDH.mult(a, p, "prep_discontinue") == 1.0
    assert SDH.mult(a, _params(mults={"moud_discontinue": 1.2}, on=False), "moud_discontinue") == 1.0


def test_sex_work_scales_sexual_partners_once_at_entry():
    p = _params(sw={"HF": (1.0, 1.0)}, partner_mult=1.5)
    a = _agent("Inj", "HF", p, partners=2); a.sdh.init_agent(Pop(1), 0)
    assert a.sex_work and a.mean_num_partners["Sex"] == 3 and a.target_partners["Sex"] == 3 and a.mean_num_partners["Inj"] == 1
    b = _agent("Inj", "HF", p, partners=0); b.sdh.init_agent(Pop(1), 0)
    assert b.sex_work and b.mean_num_partners["Sex"] == 0


def test_visit_rate_hardship_multiplier():
    setting = SimpleNamespace(visit_rate_annual=12.0, visit_mult_hardship_per_point=0.85, visit_mult_uninsured=1.0,
                              visit_mult_moud=1.0, visit_mult_inj=1.0, visit_mult_hiv_positive=1.0)
    a = SimpleNamespace(drug_type="NonInj", insured=True, hardship=0, hiv=SimpleNamespace(active=False), moud=SimpleNamespace(active=False),
                        location=SimpleNamespace(params=SimpleNamespace(care_seeking=SimpleNamespace(setting={"primary_care": setting}))))
    cs = CareSeeking(a); cs.visit_frailty = {"primary_care": 1.0}
    base = cs.visit_rate(setting, "primary_care", 12)
    a.hardship = 3
    assert cs.visit_rate(setting, "primary_care", 12) == pytest.approx(0.85 ** 3 * base)


def test_stats_count_levels_and_sex_work():
    a = _agent(); a.hardship = 2; a.sex_work = True
    stats = {k: 0 for k in SDH.stats}; a.sdh.set_stats(stats, 1)
    assert stats == {"hardship_0": 0, "hardship_1": 0, "hardship_2": 1, "hardship_3": 0, "sex_work": 1}


def test_registered_after_insurance_and_before_the_care_layer():
    subs = base_feature.BaseFeature.__subclasses__()
    assert subs.index(Insurance) < subs.index(SDH) < subs.index(CareSeeking)


def test_risk_multipliers_are_relative_and_mean_preserving():
    """Risk hooks (Jun, 2026-10-24): m^k over the population mean of m^k, so the average behaviour is unchanged."""
    p = _params(inj={"housing": 0.69})
    p.sdh.hardship.risk_mult_per_point = SimpleNamespace(sharing=1.15, condomless_sex=1.0, inject_start=1.2, overdose=1.25)
    p.sdh.hardship.mean_preserving = True
    agents = [_agent(params=p) for _ in range(10)]
    for a, k in zip(agents, [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]):
        a.hardship = k
    SDH.refresh_norms(agents, p)
    mults = [SDH.risk_mult(a, p, "sharing") for a in agents]
    assert sum(mults) / len(mults) == pytest.approx(1.0)
    assert mults[9] / mults[0] == pytest.approx(1.15 ** 3)
    assert SDH.risk_mult(agents[9], p, "condomless_sex") == 1.0           # multiplier 1: no effect, no normalisation
    p.sdh.hardship.mean_preserving = False
    assert SDH.risk_mult(agents[9], p, "overdose") == pytest.approx(1.25 ** 3)
    assert SDH.risk_mult(agents[9], _params(on=False), "overdose") == 1.0
    # nobody in hardship: every multiplier is exactly 1 (the identity run)
    for a in agents:
        a.hardship = 0
    SDH.refresh_norms(agents, p)
    assert all(SDH.risk_mult(a, p, h) == 1.0 for a in agents for h in ("sharing", "inject_start", "overdose"))
