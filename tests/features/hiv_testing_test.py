import logging

import pytest

from titan.features import HIVTesting


@pytest.mark.unit
def test_legacy_base_prob_is_rejected(params):
    """
    base_prob was silently redefined from per-timestep to annual. A params file
    still using the old key must fail loudly rather than quietly applying a
    stale fitted value under the new meaning.
    """
    params.hiv_testing.base_prob = 0.21104210818614516

    with pytest.raises(ValueError, match="base_prob_annual"):
        HIVTesting.init_class(params)


@pytest.mark.unit
def test_base_prob_annual_is_accepted(params):
    params.hiv_testing.base_prob = -1.0
    params.hiv_testing.base_prob_annual = 0.3

    HIVTesting.init_class(params)  # must not raise


@pytest.mark.unit
def test_start_time_after_burn_in_warns(params, caplog):
    """
    Testing that cannot run during burn-in leaves everyone infected during
    burn-in permanently undiagnosed and off HAART.
    """
    params.hiv_testing.base_prob = -1.0
    params.model.time.burn_steps = 36
    params.hiv_testing.start_time = 0

    with caplog.at_level(logging.WARNING):
        HIVTesting.init_class(params)

    assert "burn-in" in caplog.text


@pytest.mark.unit
def test_default_start_time_allows_burn_in_testing(params, caplog):
    params.hiv_testing.base_prob = -1.0
    params.model.time.burn_steps = 36
    params.hiv_testing.start_time = -9999

    with caplog.at_level(logging.WARNING):
        HIVTesting.init_class(params)

    assert "burn-in" not in caplog.text


@pytest.mark.unit
def test_negative_start_time_is_schema_valid(params):
    """
    Burn-in runs at negative time steps, so start_time must be allowed to go
    negative -- the schema previously pinned min: 0, making the burn-in gap
    unfixable by configuration.
    """
    assert params.hiv_testing.start_time <= 0


# ------------------------------------------------- test-and-offer bundle (2026-10-09)

def _bundle_setup(make_model, make_agent, willing, prev=None, **setting):
    """prev: the model of an earlier setup in the same test; its care-layer flag
    is switched off first, because make_model/make_agent share one params object
    and TITAN() with the layer on would build a provider pool from the minimal
    setting table."""
    from types import SimpleNamespace
    from titan.parse_params import ObjMap
    from titan.features import CareSeeking
    from conftest import FakeRandom
    if prev is not None:
        prev.params.features.care_seeking = False
    model = make_model()
    model.time = 140
    # the agent is built BEFORE the care layer is switched on: the shared params
    # fixture persists across tests, and create_agent would otherwise build a
    # provider pool from the minimal setting table below
    model.params.features.care_seeking = False
    a = make_agent(DU="Inj")
    model.params.features.care_seeking = True
    fields = {"prep_bundle_prob": 0.0, "prep_same_day_prob": 1.0, "prep_deferred_return_prob": 1.0,
              "prep_referral_prob": 0.0, "prep_referral_return_prob": 0.0,
              "prep_bundle_prob_lever": -1.0, "prep_referral_prob_lever": -1.0, "prep_bundle_lever_start_step": -999,
              "acceptance_mult_prep": 1.0, "reach_acceptance_mult_prep": 1.0,
              "reach_prep_bundle_prob": -1.0, "reach_prep_deferred_return_prob": -1.0, "reach_prep_same_day_prob": -1.0,
              "reach_prep_referral_prob": -1.0, "reach_prep_referral_return_prob": -1.0}
    reached = setting.pop("reached", False)
    active = setting.pop("active", reached)
    fields.update(setting)
    model.params.care_seeking = ObjMap({"setting": {"primary_care": fields}})
    # per-setting reach (care_seeking.reached) and the per-step active-visit flag
    # (care_active), which is what the bundle gates on since 2026-10-13;
    # care_reach is the neutral multiplier dict every agent carries
    a.care_seeking.reached = {"primary_care": True} if reached else {}
    a.care_active = {"primary_care": bool(active)}
    a.care_reach = {"prep_discontinue_mult": 1.0, "prep_adherence_mult": 1.0, "prep_restart_prob": 0.0, "prep_monitoring_mult": 1.0}
    a.location.params.prep.target_model = ["pwid"]
    a.location.params.prep.start_time = 0
    a.location.params.prep.acceptance = 1.0
    a.location.params.hiv_testing.acceptance = 1.0
    a.care_offer = {"hiv_test": True, "prep": False}
    a.care_offer_mult = {"hiv_test": 1.0, "prep": 1.0}
    a.care_offer_by_setting = {"primary_care": {"hiv_test": True, "prep": False}}
    a.care_seeking.provider_id = {"primary_care": 0}
    CareSeeking.providers = {"primary_care": [{"prep": {"willing": willing, "intensity": 0.1},
                                               "hiv_test": {"willing": True, "intensity": 0.5}}]}
    a.hiv_testing.active = True
    a.hiv_testing.tests_this_step = 0
    HIVTesting.test_counts = {a.race: {a.sex_type: 0}}
    from titan.features import Prep
    Prep.counts[a.race] = 0
    model.run_random = FakeRandom(0.0)
    return model, a


@pytest.mark.unit
def test_negative_test_at_a_prescribing_provider_raises_prep(make_model, make_agent):
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0)
    a.hiv_testing.update_agent(model)
    assert a.hiv_testing.tests_this_step == 1
    assert a.prep.active and a.prep.started_bundled_this_step == "same_day"
    model.params.features.care_seeking = False



@pytest.mark.unit
def test_bundle_needs_a_negative_test_and_a_willing_provider_unless_reached(make_model, make_agent):
    # HIV-positive: diagnosed, no PrEP
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0)
    a.hiv.active = True
    a.hiv_testing.update_agent(model)
    assert a.hiv.dx and not a.prep.active
    # unwilling provider, no floor: no PrEP
    model, b = _bundle_setup(make_model, make_agent, willing=False, prev=model, prep_bundle_prob=1.0)
    b.hiv_testing.update_agent(model)
    assert b.hiv_testing.tests_this_step == 1 and not b.prep.active and not b.prep.bundle_offered_this_step
    # unwilling provider, reached by a programme whose only floor is the bundle
    # floor: still no PrEP (a non-prescriber cannot prescribe, 2026-10-12)
    model, c = _bundle_setup(make_model, make_agent, willing=False, prev=model, prep_bundle_prob=1.0,
                             reach_prep_bundle_prob=1.0, reached=True)
    c.hiv_testing.update_agent(model)
    assert not c.prep.active and not c.prep.bundle_offered_this_step
    # unwilling provider, reached by a programme with a referral floor and a
    # warm handoff: a referral, never same-day, started next step
    model, e = _bundle_setup(make_model, make_agent, willing=False, prev=model, prep_bundle_prob=1.0,
                             prep_same_day_prob=1.0, reach_prep_referral_prob=1.0, reach_prep_referral_return_prob=1.0, reached=True)
    e.hiv_testing.update_agent(model)
    assert e.prep.bundle_offered_this_step and not e.prep.active
    assert e.prep.pending_start == 141 and e.prep.pending_kind == "referred" and e.prep.pending_return_prob == 1.0
    model.time = 141
    e.prep.update_agent(model)
    assert e.prep.active and e.prep.started_bundled_this_step == "referred" and e.prep.started_bundled_setting == "primary_care"
    model.time = 140
    # dated lever: base 0 until the lever starts, then the lever value
    model, d = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=0.0,
                             prep_bundle_prob_lever=1.0, prep_bundle_lever_start_step=200)
    d.hiv_testing.update_agent(model)
    assert not d.prep.active and not d.prep.bundle_offered_this_step
    model.time = 200
    d.hiv_testing.tests_this_step = 0
    d.hiv_testing.update_agent(model)
    assert d.prep.active and d.prep.started_bundled_this_step == "same_day"
    model.time = 140
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_bundle_floors_apply_only_to_an_agent_reached_in_that_setting(make_model, make_agent):
    """The 2026-10-09 code gated on bool(care_reach), which is True for every agent
    (reach_effects returns neutral values), so programme floors applied to everyone
    (ensemble_bundle_2026-10-09 placebo failure). Gate on care_seeking.reached[setting]."""
    # unreached, neutral care_reach, setting carries a floor: base values apply (0 -> nothing)
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=0.0, reach_prep_bundle_prob=1.0)
    a.hiv_testing.update_agent(model)
    assert a.hiv_testing.tests_this_step == 1 and not a.prep.bundle_offered_this_step and not a.prep.active
    # reached in another setting only: still the base values here
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=0.0, reach_prep_bundle_prob=1.0)
    b.care_seeking.reached = {"ED": True}
    b.hiv_testing.update_agent(model)
    assert not b.prep.bundle_offered_this_step
    # reached in this setting but not an active visit this step: base values
    model, d = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=0.0, reach_prep_bundle_prob=1.0, reached=True, active=False)
    d.hiv_testing.update_agent(model)
    assert not d.prep.bundle_offered_this_step
    # reached and an active programme visit this step: the floor applies
    model, c = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=0.0, reach_prep_bundle_prob=1.0, reached=True)
    c.hiv_testing.update_agent(model)
    assert c.prep.active and c.prep.started_bundled_this_step == "same_day"
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_bundle_makes_no_offer_for_an_agent_on_prep_or_who_stopped_this_step(make_model, make_agent):
    """2026-10-13: an agent already on PrEP, or who discontinued PrEP this step (the
    monitoring-visit test is ordered for the user it is monitoring), gets no bundle and
    no bundle draw; a second setting's test can still raise PrEP after a miss elsewhere."""
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0)
    a.prep.active = True
    a.hiv_testing.update_agent(model)
    assert a.hiv_testing.tests_this_step == 1 and not a.prep.bundle_offered_this_step
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0)
    b.prep.stopped_this_step = True
    b.hiv_testing.update_agent(model)
    assert b.hiv_testing.tests_this_step == 1 and not b.prep.bundle_offered_this_step and not b.prep.active
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_bundle_acceptance_uses_the_settings_multiplier(make_model, make_agent):
    """acceptance_mult_prep and, when reached, reach_acceptance_mult_prep scale the
    bundled offer's acceptance (2026-10-12; before, only a regular offer's multiplier did)."""
    from conftest import FakeRandom
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0, acceptance_mult_prep=1.0)
    a.location.params.prep.acceptance = 0.5
    model.run_random = FakeRandom(0.3)          # test taken (0.3 < 1), bundle raised, declines (0.3 >= 0.5? no: 0.3 < 0.5 accepts)
    model.run_random = FakeRandom(0.6)          # 0.6 < 1 test; 0.6 < 1 raised; 0.6 >= 0.5 declines
    a.hiv_testing.update_agent(model)
    assert a.prep.bundle_offered_this_step and not a.prep.active
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0, acceptance_mult_prep=1.5)
    b.location.params.prep.acceptance = 0.5
    model.run_random = FakeRandom(0.6)          # 0.6 < 0.75 accepts
    b.hiv_testing.update_agent(model)
    assert b.prep.active
    # the programme multiplier only when reached in the setting
    model, c = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0, reach_acceptance_mult_prep=1.5)
    c.location.params.prep.acceptance = 0.5
    model.run_random = FakeRandom(0.6)
    c.hiv_testing.update_agent(model)
    assert not c.prep.active
    model, d = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0, reach_acceptance_mult_prep=1.5, reached=True)
    d.location.params.prep.acceptance = 0.5
    model.run_random = FakeRandom(0.6)
    d.hiv_testing.update_agent(model)
    assert d.prep.active
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_referral_at_a_non_prescriber_is_never_same_day(make_model, make_agent):
    model, a = _bundle_setup(make_model, make_agent, willing=False, prep_bundle_prob=1.0, prep_same_day_prob=1.0,
                             prep_referral_prob=1.0, prep_referral_return_prob=0.4)
    a.hiv_testing.update_agent(model)
    assert a.prep.bundle_offered_this_step and not a.prep.active
    assert a.prep.pending_start == 141 and a.prep.pending_kind == "referred" and a.prep.pending_return_prob == 0.4
    from conftest import FakeRandom
    from collections import defaultdict
    model.time = 141
    model.run_random = FakeRandom(0.2)           # attends (0.2 < 0.4)
    a.prep.update_agent(model)
    assert a.prep.active and a.prep.started_bundled_this_step == "referred"
    stats = defaultdict(int); a.prep.set_stats(stats, 141)
    assert stats["prep_new"] == 1 and stats["prep_new_bundled"] == 1 and stats["prep_new_referred"] == 1
    assert stats["prep_new_bundled_primary_care"] == 1 and stats["prep_new_same_day"] == 0 and stats["prep_new_deferred"] == 0
    model.time = 140
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_nonclinical_testing_rate_by_drug_type(make_model, make_agent):
    """People who do not inject get nonclinical_annual_noninj when it is set (>= 0),
    the PWID rate otherwise; people who inject always get nonclinical_annual (2026-10-12)."""
    model = make_model()
    inj = make_agent(DU="Inj"); non = make_agent(DU="NonInj")
    for ag in (inj, non):
        ag.care_offer = {"hiv_test": False}
    tp = inj.location.params.hiv_testing
    tp.nonclinical_annual = 0.38
    tp.nonclinical_annual_noninj = -1.0
    steps = model.params.model.time.steps_per_year
    assert inj.hiv_testing.encounter_prob(tp, steps) == pytest.approx(inj.hiv_testing.annual_to_step(0.38, steps))
    assert non.hiv_testing.encounter_prob(tp, steps) == pytest.approx(inj.hiv_testing.annual_to_step(0.38, steps))
    tp.nonclinical_annual_noninj = 0.20
    assert inj.hiv_testing.encounter_prob(tp, steps) == pytest.approx(inj.hiv_testing.annual_to_step(0.38, steps))
    assert non.hiv_testing.encounter_prob(tp, steps) == pytest.approx(non.hiv_testing.annual_to_step(0.20, steps))
    tp.nonclinical_annual_noninj = 0.0
    assert non.hiv_testing.encounter_prob(tp, steps) == 0.0
    tp.nonclinical_annual_noninj = -1.0



@pytest.mark.unit
def test_bundle_deferred_delivery_uses_the_programme_return_when_reached(make_model, make_agent):
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, prep_deferred_return_prob=0.3, reach_prep_deferred_return_prob=0.8)
    a.hiv_testing.update_agent(model)
    assert not a.prep.active and a.prep.pending_start == 141 and a.prep.pending_return_prob == 0.3
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, prep_deferred_return_prob=0.3, reach_prep_deferred_return_prob=0.8, reached=True)
    b.hiv_testing.update_agent(model)
    assert b.prep.pending_return_prob == 0.8 and b.prep.pending_kind == "deferred"
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_bundle_same_day_share_uses_the_programme_value_when_reached(make_model, make_agent):
    """Setting: deferred (same-day 0); the navigator programme makes it same-day for reached agents."""
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, reach_prep_same_day_prob=1.0)
    a.hiv_testing.update_agent(model)
    assert not a.prep.active and a.prep.pending_start == 141
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, reach_prep_same_day_prob=1.0, reached=True)
    b.hiv_testing.update_agent(model)
    assert b.prep.active and b.prep.started_bundled_this_step == "same_day"
    model.params.features.care_seeking = False



def test_people_tested_this_year_counts_an_agent_once_per_calendar_year():
    """hiv_tested_this_year (2026-10-24, reporting only): an agent with a test in the calendar year counts in
    every later step of that year and not in the next (t = 1 is January; 12 steps a year)."""
    from types import SimpleNamespace
    agent = SimpleNamespace(location=SimpleNamespace(params=SimpleNamespace(model=SimpleNamespace(time=SimpleNamespace(steps_per_year=12)))))
    t = HIVTesting(agent)
    stats = {k: 0 for k in HIVTesting.stats}
    t.set_stats(stats, 3)
    assert sum(stats.values()) == 0
    t.tests_this_step = 2                        # two tests this step, none at a care encounter
    t.set_stats(stats, 5)
    assert stats["hiv_tests"] == 2 and stats["hiv_tests_nonclinical"] == 2 and stats["hiv_tests_clinical"] == 0
    assert stats["hiv_tested_this_year"] == 1 and stats["hiv_tested_this_year_clinical"] == 0 and t.tests_this_step == 0
    t.set_stats(stats, 12)                       # December of the same year: still counted, no new test
    assert stats["hiv_tested_this_year"] == 2 and stats["hiv_tests"] == 2
    t.set_stats(stats, 13)                       # January of the next year: not counted
    assert stats["hiv_tested_this_year"] == 2
    t.tests_this_step = t.clinical_tests_this_step = 1      # a clinic-billed test in the new year
    t.last_clinical_test_time = 14
    t.set_stats(stats, 14)
    assert stats["hiv_tests_clinical"] == 1 and stats["hiv_tested_this_year_clinical"] == 1 and stats["hiv_tested_this_year"] == 3
