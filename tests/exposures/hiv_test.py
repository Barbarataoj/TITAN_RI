import math
from types import SimpleNamespace

import pytest

from conftest import FakeRandom

from titan.exposures import HIV
from titan.agent import Relationship


# The four tests below hand-roll a minimal SimpleNamespace agent/model rather
# than using the `make_agent`/`make_model` fixtures, to sidestep a
# pre-existing, unrelated break: tests/params/basic.yml still allocates
# population share to a "None" drug type that titan/params/classes.yml's
# `drug_types` no longer lists, so `create_params`'s ppl-sums-to-1 check
# fails before any fixture-based test body runs. See the note in
# tests/features/overdose_test.py's module docstring for the same issue.


class _HashableNamespace:
    """Like SimpleNamespace, but hashable by identity (HIV.agents is a set,
    and SimpleNamespace's __eq__ override makes plain instances unhashable)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_hiv_agent(race="black_nonhispanic", sex_type="MSM"):
    agent = _HashableNamespace(race=race, sex_type=sex_type)
    agent.haart = SimpleNamespace(aids_scale=lambda: 0.0)
    agent.hiv = HIV(agent)
    agent.location = SimpleNamespace(
        params=SimpleNamespace(
            hiv=SimpleNamespace(
                start_time=0,
                dx=SimpleNamespace(prob=1.0),  # top-level override: always diagnoses if reached
                aids=SimpleNamespace(prob=0.0),
            )
        )
    )
    return agent


@pytest.mark.unit
def test_legacy_diagnosis_gated_off_when_hiv_testing_enabled():
    """
    HIV.update_agent()'s implicit per-timestep diagnosis hazard is the
    legacy stand-in for a testing process, from before HIVTesting existed.
    It must step aside once features.hiv_testing is on, so an infected
    agent isn't diagnosed twice, independently, every timestep (see
    CODEX_DEBUG_HANDOFF.md Priority 1).
    """
    agent = _make_hiv_agent()
    agent.hiv.active = True
    agent.hiv.dx = False
    HIV.dx_counts = {"black_nonhispanic": {"MSM": 0}}
    HIV.agents = set()

    model = SimpleNamespace(
        time=1,
        params=SimpleNamespace(
            hiv=agent.location.params.hiv,
            features=SimpleNamespace(hiv_testing=True),
        ),
        calibration=SimpleNamespace(test_frequency=1.0),
        run_random=FakeRandom(0.0),
    )

    agent.hiv.update_agent(model)

    assert agent.hiv.dx is False


@pytest.mark.unit
def test_legacy_diagnosis_still_fires_when_hiv_testing_disabled():
    """The legacy pathway is kept for scenarios that don't enable hiv_testing."""
    agent = _make_hiv_agent()
    agent.hiv.active = True
    agent.hiv.dx = False
    HIV.dx_counts = {"black_nonhispanic": {"MSM": 0}}
    HIV.agents = set()

    model = SimpleNamespace(
        time=1,
        params=SimpleNamespace(
            hiv=agent.location.params.hiv,
            features=SimpleNamespace(hiv_testing=False),
        ),
        calibration=SimpleNamespace(test_frequency=1.0),
        run_random=FakeRandom(0.0),
    )

    agent.hiv.update_agent(model)

    assert agent.hiv.dx is True


@pytest.mark.unit
def test_background_foi_applies_feature_acquisition_multiplier():
    """
    HIV.get_background_foi_probability() must run susceptible agents'
    acquisition-risk multipliers (PrEP, vaccine, ...) the same way the
    relationship pathway does, so PrEP/vaccine protection isn't silently
    skipped for the background-FOI pathway (CODEX_DEBUG_HANDOFF.md
    Priority 3). Also asserts the multiplier is called with model.time, not
    self.time - self.time is None here since this (HIV-negative) agent has
    never converted.
    """
    calls = []

    class FakePrep:
        name = "prep"

        def get_acquisition_risk_multiplier(self, time, interaction_type):
            calls.append((time, interaction_type))
            return 0.1

    agent = SimpleNamespace(drug_type="NonInj", sex_type="HF", sex_role="receptive")
    agent.hiv = HIV(agent)
    agent.location = SimpleNamespace(
        params=SimpleNamespace(
            hiv=SimpleNamespace(
                background_sex=SimpleNamespace(
                    rate_nonoud_per_timestep=10.0,
                    partner_hiv_prevalence_nonoud=1.0,
                    inj_multiplier=1.0,
                )
            ),
            # The per-act probability now comes from the same stratified table the
            # network route reads, not a standalone background_sex scalar.
            partnership=SimpleNamespace(
                sex=SimpleNamespace(
                    acquisition={"HF": {"receptive": 0.5, "insertive": 0.25,
                                        "versatile": 0.375}}
                )
            ),
        )
    )
    agent.prep = FakePrep()

    model = SimpleNamespace(
        time=7,
        features=[FakePrep],
        calibration=SimpleNamespace(acquisition=1.0),
    )

    p_with_prep = agent.hiv.get_background_foi_probability(model)
    assert calls == [(7, "sex")]

    agent.prep = SimpleNamespace(get_acquisition_risk_multiplier=lambda t, i: 1.0)
    p_without_prep = agent.hiv.get_background_foi_probability(model)

    assert p_with_prep == pytest.approx(p_without_prep * 0.1, rel=1e-6)
    assert p_with_prep > 0.0


@pytest.mark.unit
def test_transmission_probability_dx_risk_reduction_applied_once():
    """
    hiv.dx.risk_reduction.injection reduces the per-act transmission
    probability here, in HIV.get_transmission_probability - and only here;
    Injection.get_num_acts() no longer separately shrinks the act count with
    the same parameter (CODEX_DEBUG_HANDOFF.md Priority 4). Confirms the
    single application is exact: p_dx == p_no_dx * (1 - risk_reduction).
    """
    source = SimpleNamespace()
    source.location = SimpleNamespace(
        params=SimpleNamespace(
            hiv=SimpleNamespace(dx=SimpleNamespace(risk_reduction={"injection": 0.4}))
        )
    )
    source.hiv = HIV(source)
    source.hiv.dx = True

    partner = SimpleNamespace(
        race="black_nonhispanic",
        location=SimpleNamespace(
            params=SimpleNamespace(
                demographics={"black_nonhispanic": SimpleNamespace(hiv=SimpleNamespace(transmission=1.0))}
            )
        ),
    )

    model = SimpleNamespace(
        time=0,
        params=SimpleNamespace(
            partnership=SimpleNamespace(
                injection=SimpleNamespace(transmission=SimpleNamespace(base=0.5))
            )
        ),
        features=[],
        calibration=SimpleNamespace(acquisition=1.0),
    )

    p_dx = source.hiv.get_transmission_probability(model, "injection", partner, 1)

    source.hiv.dx = False
    p_no_dx = source.hiv.get_transmission_probability(model, "injection", partner, 1)

    assert p_dx == pytest.approx(p_no_dx * 0.6, rel=1e-6)


@pytest.mark.unit
def test_hiv_expose(make_model, make_agent):
    model = make_model()
    model.run_random = FakeRandom(0.0)  # always less than param
    a = make_agent()
    p = make_agent()
    a.partners["Sex"] = set()
    p.partners["Sex"] = set()
    rel = Relationship(a, p, 10, bond_type="Sex")

    HIV.expose(model, "sex", rel, 10)

    assert a.hiv.active is False
    assert p.hiv.active is False

    a.hiv.active = True
    p.hiv.active = True

    # test nothing happens
    HIV.expose(model, "sex", rel, 10)

    assert a.hiv.active
    assert p.hiv.active

    p.hiv.active = False

    # test conversion happens
    HIV.expose(model, "sex", rel, 10)

    assert a.hiv.active
    assert p.hiv.active


@pytest.mark.unit
def test_hiv_init(make_population, make_agent):
    pop = make_population()
    pop.pop_random = FakeRandom(0.0)
    a = make_agent()

    time = pop.params.hiv.start_time - 1

    a.hiv.init_agent(pop, time)

    assert a.hiv.active is False

    time = pop.params.hiv.start_time

    a.hiv.init_agent(pop, time)

    assert a.hiv.active
    assert a.hiv.time == time - pop.params.hiv.max_init_time
    assert a.hiv.dx
    assert a.hiv.dx_time == a.hiv.time
    assert a.hiv.aids
    assert a in HIV.agents
    assert HIV.dx_counts[a.race][a.sex_type] == 1


@pytest.mark.unit
def test_hiv_convert(make_model, make_agent):
    model = make_model()
    a = make_agent()
    a.prep.active = True

    model.run_random = FakeRandom(-0.1)

    a.hiv.convert(model)

    assert a.hiv.active
    assert a.hiv.time == model.time
    assert a in HIV.agents
    assert a.prep.active is False


@pytest.mark.unit
def test_diagnose_hiv(make_model, make_agent):
    model = make_model()
    model.params.partner_tracing.prob = 1.0
    model.time = 1
    a = make_agent()
    p = make_agent()
    p.hiv.active = True
    a.partners["Sex"].add(p)

    model.run_random = FakeRandom(-0.1)  # always less than param
    a.hiv.diagnose(model)

    assert a.hiv.dx
    assert a.hiv.dx_time == model.time


@pytest.mark.unit
def test_get_transmission_probability(make_model, make_agent):
    model = make_model()
    a = make_agent(race="white_nonhispanic", SO="MSM")
    a.sex_role = "versatile"

    p = make_agent(race="white_nonhispanic", SO="MSM")
    p.sex_role = "versatile"

    # test versatile-versatile relationship
    p_needle = model.params.partnership.injection.transmission.base
    p_sex = model.params.partnership.sex.acquisition.MSM.versatile
    scale = model.params.calibration.acquisition

    assert (
        a.hiv.get_transmission_probability(model, "injection", p, 1) == p_needle * scale
    )
    assert a.hiv.get_transmission_probability(model, "sex", p, 1) == p_sex * scale

    # test one vers agent, one receptive agent
    a.sex_role = "receptive"
    p_sex_ins = model.params.partnership.sex.acquisition.MSM.insertive
    p_sex_rec = model.params.partnership.sex.acquisition.MSM.receptive

    assert a.hiv.get_transmission_probability(model, "sex", p, 1) == p_sex_ins * scale
    assert p.hiv.get_transmission_probability(model, "sex", a, 1) == p_sex_rec * scale


@pytest.mark.unit
def test_get_acute_status(make_agent, make_model):
    model = make_model()
    a = make_agent()  # no HIV on init
    assert a.hiv.get_acute_status(model.time + 2) is False
    a.hiv.convert(model)
    assert a.hiv.get_acute_status(model.time + 2) is True


# ---------------------------------------------------------------------------
# Degenerate sex-act frequency guard (EPIDEMIC_DYNAMICS_MAP_2026-07-29.md, P1)
# ---------------------------------------------------------------------------
def _freq_params(sex_freq_by_bond, bond_acts):
    """Minimal params carrying just what _check_sex_acts_reachable reads."""
    return SimpleNamespace(
        classes=SimpleNamespace(
            sex_types={},
            races=[],
            bond_types={
                bond: SimpleNamespace(acts_allowed=acts)
                for bond, acts in bond_acts.items()
            },
        ),
        partnership=SimpleNamespace(
            sex=SimpleNamespace(frequency=sex_freq_by_bond)
        ),
    )


def _bins(*specs):
    return SimpleNamespace(
        type="bins",
        bins={
            i + 1: SimpleNamespace(prob=p, min=lo, max=hi)
            for i, (p, lo, hi) in enumerate(specs)
        },
    )


def _dist(*values):
    return SimpleNamespace(
        type="distribution",
        distribution=SimpleNamespace(
            dist_type="poisson",
            vars={
                i + 1: SimpleNamespace(value=v, value_type="float")
                for i, v in enumerate(values)
            },
        ),
    )


@pytest.mark.unit
def test_default_sex_frequency_bin_is_rejected():
    """
    The shipped default is a single bin of min:0/max:1, and safe_random_int
    returns on the half-open [min, max), so that bin yields zero acts forever.
    BaseInteraction.interact then returns before ever calling expose and the
    whole sexual transmission pathway vanishes with no error and no warning --
    which is exactly what every RI parameter file did: Sex.get_num_acts
    returned >=1 act on 0 of 369,609 calls.
    """
    params = _freq_params(
        {"Sex": _bins((1.0, 0, 1)), "SexInj": _bins((1.0, 0, 1))},
        {"Sex": ["sex"], "SexInj": ["sex", "injection"]},
    )
    with pytest.raises(ValueError, match="never produce a sex act"):
        HIV.init_class(params)


@pytest.mark.unit
def test_reachable_sex_frequency_bins_accepted():
    """A bin with max > 1 can return >= 1 act, so it is fine."""
    params = _freq_params(
        {"Sex": _bins((0.5, 0, 1), (1.0, 37, 38)), "SexInj": _bins((1.0, 0, 1))},
        {"Sex": ["sex"], "SexInj": ["sex", "injection"]},
    )
    HIV.init_class(params)  # must not raise


@pytest.mark.unit
def test_poisson_sex_frequency_accepted():
    """The distribution form the RI scenarios now use."""
    params = _freq_params(
        {"Sex": _dist(4.0), "SexInj": _dist(4.0)},
        {"Sex": ["sex"], "SexInj": ["sex", "injection"]},
    )
    HIV.init_class(params)


@pytest.mark.unit
def test_explicit_zero_distribution_is_allowed():
    """
    An injection-only epidemic is a legitimate modelling choice -- it just has to
    be written down rather than arrived at by leaving a key out. A zero-valued
    distribution says so explicitly and is accepted.
    """
    params = _freq_params(
        {"Sex": _dist(0.0), "SexInj": _dist(0.0)},
        {"Sex": ["sex"], "SexInj": ["sex", "injection"]},
    )
    HIV.init_class(params)


@pytest.mark.unit
def test_non_sex_bonds_are_not_checked():
    """Injection- and pca-only bonds carry no sex acts, so their frequency is
    irrelevant and their default bin must not trip the guard."""
    params = _freq_params(
        {
            "Sex": _dist(4.0),
            "Inj": _bins((1.0, 0, 1)),
            "Social": _bins((1.0, 0, 1)),
        },
        {"Sex": ["sex"], "Inj": ["injection"], "Social": ["pca"]},
    )
    HIV.init_class(params)


@pytest.mark.unit
def test_background_foi_uses_stratified_per_act_probability():
    """
    The background force-of-infection reads its per-act transmission probability
    from partnership.sex.acquisition[sex_type][sex_role] -- the same table the
    network route uses -- rather than a single blended scalar.

    Until 2026-07-30 it used hiv.background_sex.per_act_transmission_prob, one
    number applied to every sex type. That reproduced finding F1 (MSM anal-sex
    values applied to everyone) inside the background route after F1 had been
    fixed in the network route. Two agents identical except for sex type must
    now get different background risk.
    """
    acquisition = {
        "HF": {"receptive": 0.0008, "insertive": 0.0004, "versatile": 0.0006},
        "MSM": {"receptive": 0.015, "insertive": 0.005, "versatile": 0.01},
    }

    def make(sex_type):
        agent = SimpleNamespace(
            drug_type="NonInj", sex_type=sex_type, sex_role="receptive"
        )
        agent.hiv = HIV(agent)
        agent.location = SimpleNamespace(
            params=SimpleNamespace(
                hiv=SimpleNamespace(
                    background_sex=SimpleNamespace(
                        rate_nonoud_per_timestep=1.0,
                        partner_hiv_prevalence_nonoud=1.0,
                        inj_multiplier=1.0,
                    )
                ),
                partnership=SimpleNamespace(
                    sex=SimpleNamespace(acquisition=acquisition)
                ),
            )
        )
        return agent

    model = SimpleNamespace(
        time=1, features=[], calibration=SimpleNamespace(acquisition=1.0)
    )

    p_hf = make("HF").hiv.get_background_foi_probability(model)
    p_msm = make("MSM").hiv.get_background_foi_probability(model)

    # 1 - exp(-p) is monotone in p, so the ordering of the table survives
    assert p_msm > p_hf
    assert p_hf == pytest.approx(1 - math.exp(-0.0008), rel=1e-9)
    assert p_msm == pytest.approx(1 - math.exp(-0.015), rel=1e-9)


@pytest.mark.unit
def test_background_foi_sex_type_override_replaces_the_scalars():
    """
    hiv.background_sex.by_sex_type[sex_type] (2026-09-29) overrides the partner
    prevalence and the act rate for that sex type; a zero in either field, or
    no entry at all, falls back to the scalars. Same per-act table for both
    agents here, so the ratio of the two probabilities is the ratio of
    rate x prevalence.
    """
    acquisition = {"MSM": {"receptive": 0.01, "insertive": 0.01, "versatile": 0.01},
                   "HM": {"receptive": 0.01, "insertive": 0.01, "versatile": 0.01}}

    def make(sex_type, by_sex_type):
        agent = SimpleNamespace(drug_type="NonInj", sex_type=sex_type, sex_role="versatile")
        agent.hiv = HIV(agent)
        agent.location = SimpleNamespace(
            params=SimpleNamespace(
                hiv=SimpleNamespace(
                    background_sex=SimpleNamespace(
                        rate_nonoud_per_timestep=0.5,
                        partner_hiv_prevalence_nonoud=0.0034,
                        inj_multiplier=1.0,
                        by_sex_type=by_sex_type,
                    )
                ),
                partnership=SimpleNamespace(sex=SimpleNamespace(acquisition=acquisition)),
            )
        )
        return agent

    model = SimpleNamespace(time=1, features=[], calibration=SimpleNamespace(acquisition=1.0))
    overrides = {
        "MSM": SimpleNamespace(partner_hiv_prevalence=0.10, rate_per_timestep=2.0),
        "HM": SimpleNamespace(partner_hiv_prevalence=0.0, rate_per_timestep=0.0),
    }

    p_msm = make("MSM", overrides).hiv.get_background_foi_probability(model)
    p_hm = make("HM", overrides).hiv.get_background_foi_probability(model)
    p_hm_no_block = make("HM", None).hiv.get_background_foi_probability(model)

    assert p_msm == pytest.approx(1 - math.exp(-2.0 * 0.10 * 0.01), rel=1e-9)
    assert p_hm == pytest.approx(1 - math.exp(-0.5 * 0.0034 * 0.01), rel=1e-9)
    assert p_hm == pytest.approx(p_hm_no_block, rel=1e-12)


@pytest.mark.unit
def test_background_foi_msm_inj_multiplier_applies_only_to_msm_who_inject():
    """
    calibration.background.msm_inj_mult (2026-09-29) replaces the all-agent
    inj_multiplier for MSM/MSMW who inject; -1 or absence means "use
    inj_multiplier". Heterosexual injectors and non-injecting MSM are untouched.
    """
    acquisition = {st: {"receptive": 0.01, "insertive": 0.01, "versatile": 0.01}
                   for st in ("MSM", "MSMW", "HM")}

    def make(sex_type, drug_type):
        agent = SimpleNamespace(drug_type=drug_type, sex_type=sex_type, sex_role="versatile")
        agent.hiv = HIV(agent)
        agent.location = SimpleNamespace(
            params=SimpleNamespace(
                hiv=SimpleNamespace(
                    background_sex=SimpleNamespace(
                        rate_nonoud_per_timestep=0.5,
                        partner_hiv_prevalence_nonoud=0.1,
                        inj_multiplier=1.5,
                        by_sex_type=None,
                    )
                ),
                partnership=SimpleNamespace(sex=SimpleNamespace(acquisition=acquisition)),
            )
        )
        return agent

    def model(mult):
        cal = SimpleNamespace(acquisition=1.0)
        if mult is not None:
            cal.background = SimpleNamespace(msm_inj_mult=mult)
        return SimpleNamespace(time=1, features=[], calibration=cal)

    def p(lam):
        return 1 - math.exp(-lam)

    base = 0.5 * 0.1 * 0.01
    assert make("MSM", "Inj").hiv.get_background_foi_probability(model(6.0)) == pytest.approx(p(base * 6.0))
    assert make("MSMW", "Inj").hiv.get_background_foi_probability(model(6.0)) == pytest.approx(p(base * 6.0))
    assert make("HM", "Inj").hiv.get_background_foi_probability(model(6.0)) == pytest.approx(p(base * 1.5))
    assert make("MSM", "NonInj").hiv.get_background_foi_probability(model(6.0)) == pytest.approx(p(base))
    # -1 and an absent block both fall back to inj_multiplier
    assert make("MSM", "Inj").hiv.get_background_foi_probability(model(-1.0)) == pytest.approx(p(base * 1.5))
    assert make("MSM", "Inj").hiv.get_background_foi_probability(model(None)) == pytest.approx(p(base * 1.5))


@pytest.mark.unit
def test_background_foi_role_matters():
    """Receptive acquisition risk exceeds insertive for the same sex type."""
    acquisition = {"MSM": {"receptive": 0.015, "insertive": 0.005, "versatile": 0.01}}

    def make(role):
        agent = SimpleNamespace(drug_type="NonInj", sex_type="MSM", sex_role=role)
        agent.hiv = HIV(agent)
        agent.location = SimpleNamespace(
            params=SimpleNamespace(
                hiv=SimpleNamespace(
                    background_sex=SimpleNamespace(
                        rate_nonoud_per_timestep=1.0,
                        partner_hiv_prevalence_nonoud=1.0,
                        inj_multiplier=1.0,
                    )
                ),
                partnership=SimpleNamespace(
                    sex=SimpleNamespace(acquisition=acquisition)
                ),
            )
        )
        return agent

    model = SimpleNamespace(
        time=1, features=[], calibration=SimpleNamespace(acquisition=1.0)
    )
    rec = make("receptive").hiv.get_background_foi_probability(model)
    ins = make("insertive").hiv.get_background_foi_probability(model)
    ver = make("versatile").hiv.get_background_foi_probability(model)
    assert rec > ver > ins


@pytest.mark.unit
def test_deprecated_background_per_act_key_raises():
    """
    A params file still setting hiv.background_sex.per_act_transmission_prob must
    fail loudly: the value is no longer read, and it used to be a fitted,
    sex-type-blind duplicate of the network route's per-act table (audit F9).
    """
    params = SimpleNamespace(
        hiv=SimpleNamespace(
            background_sex=SimpleNamespace(per_act_transmission_prob=0.002805)
        )
    )
    with pytest.raises(ValueError, match="per_act_transmission_prob is deprecated"):
        HIV._check_deprecated_background_keys(params)


@pytest.mark.unit
def test_deprecated_background_per_act_key_sentinel_passes():
    """The -1.0 sentinel default means 'not set' and must not raise."""
    params = SimpleNamespace(
        hiv=SimpleNamespace(
            background_sex=SimpleNamespace(per_act_transmission_prob=-1.0)
        )
    )
    HIV._check_deprecated_background_keys(params)


# ------------------------------------------- expected-infection hazard (2026-10-09)

@pytest.mark.unit
def test_expose_records_the_hazard_and_what_prep_averted(make_model, make_agent):
    """The hazard recorded equals the probability the engine rolls; with the
    susceptible partner on adherent oral PrEP, averted = p without PrEP - p
    with it; nothing is recorded for an uninfected pair; the roll is unchanged."""
    model = make_model()
    model.run_random = FakeRandom(1.1)        # never converts
    model.time = 7
    a = make_agent(); p = make_agent()
    a.partners["Sex"] = set(); p.partners["Sex"] = set()
    rel = Relationship(a, p, 10, bond_type="Sex")
    HIV.expose(model, "sex", rel, 10)
    assert p.hiv._hazard == {} and a.hiv._hazard == {}       # neither infected
    a.hiv.active = True
    HIV.expose(model, "sex", rel, 10)
    p_roll = a.hiv.get_transmission_probability(model, "sex", p, 10)
    assert p.hiv._hazard["hiv_hazard"] == pytest.approx(p_roll)
    assert p.hiv._hazard["hiv_hazard_sex"] == pytest.approx(p_roll)
    assert "hiv_hazard_averted_prep" not in p.hiv._hazard
    assert p.hiv.active is False
    # same partner on adherent oral PrEP
    p.prep.active = True; p.prep.type = "Oral"; p.prep.adherent = True; p.prep.last_dose_time = 7
    model.time = 8
    HIV.expose(model, "sex", rel, 10)
    with_prep = a.hiv.get_transmission_probability(model, "sex", p, 10)
    without = a.hiv.get_transmission_probability(model, "sex", p, 10, skip_prep=True)
    assert without > with_prep > 0
    assert p.hiv._hazard_time == 8                              # reset on the new step
    assert p.hiv._hazard["hiv_hazard"] == pytest.approx(with_prep)
    assert p.hiv._hazard["hiv_hazard_averted_prep"] == pytest.approx(without - with_prep)
    # set_stats emits this step's hazard once, for the current step only
    stats = {k: 0 for k in HIV.stats}
    p.hiv.set_stats(stats, 8)
    assert stats["hiv_hazard"] == pytest.approx(with_prep)
    assert stats["hiv_hazard_averted_prep"] == pytest.approx(without - with_prep)
    stats = {k: 0 for k in HIV.stats}
    p.hiv.set_stats(stats, 9)
    assert stats["hiv_hazard"] == 0


@pytest.mark.unit
def test_background_exposure_records_its_hazard(make_model, make_agent):
    model = make_model()
    model.run_random = FakeRandom(1.1)
    model.time = 3
    a = make_agent()
    p_bg = a.hiv.get_background_foi_probability(model)
    a.hiv.update_agent(model)
    assert a.hiv._hazard.get("hiv_hazard_background", 0.0) == pytest.approx(p_bg)
    assert a.hiv._hazard.get("hiv_hazard", 0.0) == pytest.approx(p_bg)


@pytest.mark.unit
def test_expose_attributes_the_hazard_to_the_source_care_status(make_model, make_agent):
    """The hazard is attributed to the HIV-positive partner's care status (undiagnosed,
    diagnosed untreated, ART non-adherent, ART adherent) and, for the three unsuppressed
    states, the part the source's viral suppression would remove is recorded (2026-10-13).
    The roll is unchanged."""
    model = make_model()
    model.run_random = FakeRandom(1.1)        # never converts
    model.time = 7
    a = make_agent(); p = make_agent()
    a.partners["Sex"] = set(); p.partners["Sex"] = set()
    rel = Relationship(a, p, 10, bond_type="Sex")
    a.hiv.active = True
    # undiagnosed source
    a.hiv.dx = False; a.haart.active = False; a.haart.adherent = False
    HIV.expose(model, "sex", rel, 10)
    p_roll = a.hiv.get_transmission_probability(model, "sex", p, 10)
    p_sup = a.hiv.get_transmission_probability(model, "sex", p, 10, force_suppressed=True)
    assert p_sup < p_roll
    assert p.hiv._hazard["hiv_hazard_src_undx"] == pytest.approx(p_roll)
    assert p.hiv._hazard["hiv_hazard_avert_sup_undx"] == pytest.approx(p_roll - p_sup)
    assert "hiv_hazard_src_art_adh" not in p.hiv._hazard
    # diagnosed, on ART, non-adherent: full hazard, suppression would remove most of it
    model.time = 8
    a.hiv.dx = True; a.haart.active = True; a.haart.adherent = False
    HIV.expose(model, "sex", rel, 10)
    p_roll = a.hiv.get_transmission_probability(model, "sex", p, 10)
    p_sup = a.hiv.get_transmission_probability(model, "sex", p, 10, force_suppressed=True)
    assert p.hiv._hazard["hiv_hazard"] == pytest.approx(p_roll)
    assert p.hiv._hazard["hiv_hazard_src_art_nonadh"] == pytest.approx(p_roll)
    assert p.hiv._hazard["hiv_hazard_avert_sup_art_nonadh"] == pytest.approx(p_roll - p_sup)
    # adherent: attributed, nothing avertable by suppression; the forced counterfactual equals the roll
    model.time = 9
    a.haart.adherent = True
    HIV.expose(model, "sex", rel, 10)
    p_roll = a.hiv.get_transmission_probability(model, "sex", p, 10)
    assert a.hiv.get_transmission_probability(model, "sex", p, 10, force_suppressed=True) == pytest.approx(p_roll)
    assert p.hiv._hazard["hiv_hazard_src_art_adh"] == pytest.approx(p_roll)
    assert "hiv_hazard_avert_sup_art_adh" not in p.hiv._hazard
    # diagnosed, not on ART
    model.time = 10
    a.haart.active = False; a.haart.adherent = False
    HIV.expose(model, "sex", rel, 10)
    assert p.hiv._hazard["hiv_hazard_src_dx_noart"] == pytest.approx(p.hiv._hazard["hiv_hazard"])
    from collections import defaultdict
    stats = defaultdict(int); p.hiv.set_stats(stats, 10)
    assert stats["hiv_hazard_src_dx_noart"] == pytest.approx(stats["hiv_hazard"])
    assert stats["hiv_hazard_avert_sup_dx_noart"] > 0
