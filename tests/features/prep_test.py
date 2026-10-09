import pytest

from conftest import FakeRandom

from titan.features import Prep
from titan.agent import Relationship


@pytest.mark.unit
def test_init_agent(make_agent, make_population, make_model):
    model = make_model()
    pop = make_population(n=1)

    # test allcomers (no prep)
    pop.pop_random = FakeRandom(0.9)
    a1 = make_agent(race="black_nonhispanic")
    a2 = make_agent(race="white_nonhispanic")
    assert not a1.prep.active and not a2.prep.active

    # test allcomers
    pop.pop_random = FakeRandom(0.4)
    a1.prep.init_agent(pop, 10)
    a2.prep.init_agent(pop, 10)
    assert a1.prep.active
    assert a2.prep.active

    # reset prep, change to racial model between black and white init prob
    a1.prep.active = a2.prep.active = False
    a1.location.params.prep.target_model = a2.location.params.prep.target_model = [
        "Racial"
    ]
    a1.prep.init_agent(pop, 10)
    a2.prep.init_agent(pop, 10)
    assert a1.prep.active
    assert not a2.prep.active


@pytest.mark.unit
def test_discontinue_prep(make_agent):
    a = make_agent()

    # set up so the agent appears to be on PrEP
    a.prep.active = True
    a.prep.time = 0
    a.prep.last_dose_time = 0
    num_prep = Prep.counts[a.race]
    Prep.counts[a.race] += 1

    a.prep.discontinue()

    assert a.prep.active is False
    assert a.prep.time is None
    assert a.prep.last_dose_time is None
    assert num_prep == Prep.counts[a.race]


@pytest.mark.unit
def test_update_agent_prep_decrement_time(make_model, make_agent):
    model = make_model()
    a = make_agent()

    model.run_random = FakeRandom(1.1)
    model.time = model.params.prep.start_time

    # set up so the agent appears to be on PrEP
    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.last_dose_time = model.time - 1
    num_prep = Prep.counts[a.race]
    Prep.counts[a.race] += 1

    a.prep.update_agent(model)

    assert a.prep.active
    assert a.prep.last_dose_time == model.time
    assert num_prep + 1 == Prep.counts[a.race]


@pytest.mark.unit
def test_update_agent_prep_decrement_end(make_model, make_agent):
    model = make_model()
    a = make_agent(race="white_nonhispanic")

    model.run_random = FakeRandom(-0.1)
    model.time = model.params.prep.start_time

    # set up so the agent appears to be on PrEP
    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.last_dose_time = model.time - 1
    num_prep = Prep.counts[a.race]
    Prep.counts[a.race] += 1

    a.prep.update_agent(model)

    assert a.prep.active is False
    assert a.prep.type == ""
    assert a.prep.last_dose_time is None
    assert num_prep == Prep.counts[a.race]


@pytest.mark.unit
def test_update_agent_prep_decrement_not_end(make_model, make_agent):
    model = make_model()
    a = make_agent()

    model.run_random = FakeRandom(1.1)
    model.time = model.params.prep.start_time

    # set up so the agent appears to be on PrEP
    a.prep.active = True
    a.prep.last_dose_time = model.time
    a.prep.type = "Inj"
    num_prep = Prep.counts[a.race]
    Prep.counts[a.race] += 1

    a.prep.update_agent(model)

    assert a.prep.active
    assert a.prep.last_dose_time == model.time
    assert num_prep + 1 == Prep.counts[a.race]


@pytest.mark.unit
def test_update_agent_prep_decrement_inj_end(make_model, make_agent):
    model = make_model()
    a = make_agent()

    model.run_random = FakeRandom(1.1)
    model.time = model.params.prep.start_time

    # set up so the agent appears to be on PrEP
    a.prep.active = True
    a.prep.last_dose_time = (
        model.time - 12
    )  # last step before hitting year mark and discontinuing
    a.prep.type = "Inj"
    num_prep = Prep.counts[a.race]
    Prep.counts[a.race] += 1

    a.prep.update_agent(model)

    assert a.prep.active is False
    assert (
        a.prep.last_dose_time is None
    )  # 3 -> -1 -> +1 == 0 # Inj no longer in PrEP types
    assert num_prep == Prep.counts[a.race]


@pytest.mark.unit
def test_initiate_prep_assertions(make_model, make_agent):
    model = make_model()
    a = make_agent()

    # no PreP if already PreP
    a.prep.active = True
    assert a.prep.initiate(model) is None

    # no PrEP if already HIV
    a.prep.active = False
    a.hiv.active = True
    assert a.prep.initiate(model) is None


@pytest.mark.unit
def test_initiate_prep_force_adh(make_model, make_agent):
    model = make_model()
    a = make_agent()

    # forcing, adherant, inj
    model.run_random = FakeRandom(-0.1)
    a.prep.initiate(model, True)
    assert a.prep.active
    assert a.prep.adherent is True
    assert a.prep.last_dose_time == model.time


@pytest.mark.unit
def test_initiate_prep_force_non_adh(make_model, make_agent):
    model = make_model()
    a = make_agent()
    # forcing, non-adherant, inj
    model.run_random = FakeRandom(1.0)
    a.prep.initiate(model, True)
    assert a.prep.active
    assert a.prep.adherent is False
    assert a.prep.last_dose_time == model.time


@pytest.mark.unit
def test_initiate_prep_eligible(make_model, make_agent):
    model = make_model()

    # make sure there's room to add more prep agents
    a = make_agent(SO="HF")  # model is "CDCwomen"
    p = make_agent(DU="Inj")
    a.partners["Sex"] = set()
    p.partners["Sex"] = set()
    p.hiv.dx = True
    p.external_exposure.active = True
    model.time = 10
    a.location.params.prep.cap = 1.0
    a.location.params.prep.target_model = ["cdc_women"]
    rel = Relationship(a, p, 10, bond_type="Sex")
    # non-forcing, adherant, inj
    model.run_random = FakeRandom(-0.1)
    a.prep.initiate(model)
    assert a.prep.active
    assert a.prep.adherent is True
    assert a.prep.last_dose_time == 10
    assert a.prep.time == 10


@pytest.mark.unit
def test_initiate_prep_eligible_racial(make_model, make_agent):
    model = make_model()

    # make sure there's room to add more prep agents
    a = make_agent(SO="HF")  # model is "CDCwomen"
    p = make_agent(DU="Inj")
    a.partners["Sex"] = set()
    p.partners["Sex"] = set()
    p.hiv.dx = True
    p.external_exposure.active = True
    model.time = 10
    a.location.params.prep.cap = 1.0
    a.location.params.prep.target_model = ["Racial"]
    rel = Relationship(a, p, 10, bond_type="Sex")
    # non-forcing, adherant, inj
    model.run_random = FakeRandom(-0.1)
    a.prep.initiate(model)
    assert a.prep.active
    assert a.prep.adherent is True
    assert a.prep.last_dose_time == 10
    assert a.prep.time == 10


@pytest.mark.unit
def test_cdc_eligible(make_agent, make_relationship):
    # test MSM
    a = make_agent()
    p = make_agent()
    r = make_relationship(a, p)
    assert a.prep.cdc_eligible()

    # test WSW fail
    a = make_agent(SO="WSW")
    assert not a.is_msm()
    assert not a.prep.cdc_eligible()

    # relationship not eligible
    p = make_agent(SO="HM")
    r = make_relationship(a, p)
    assert not a.prep.cdc_eligible()

    # relationship eligible
    p.hiv.dx = True
    assert a.prep.cdc_eligible()

    # ongoing duration fail
    a.location.params.partnership.ongoing_duration = 10
    assert not a.prep.cdc_eligible()


@pytest.mark.unit
def test_prep_eligible(make_agent, make_relationship):
    a = make_agent(SO="HF")
    p = make_agent(SO="HM")

    # test no model
    a.location.params.prep.target_model = ["cdc_women"]
    assert not a.prep.eligible(10)

    # test cdc_women
    a.location.params.prep.target_model.append("cdc_women")
    assert not a.prep.eligible(10)
    r = make_relationship(a, p)
    assert not p.is_msm()
    assert not a.prep.eligible(10)
    p.drug_type = "Inj"
    assert a.prep.eligible(10)

    # test Allcomers and Racial
    a.location.params.prep.target_model.append("Allcomers")
    assert a.prep.eligible(10)
    a.location.params.prep.target_model = ["Racial"]
    assert a.prep.eligible(10)

    # test cdc_msm
    a.location.params.prep.target_model = ["cdc_msm"]
    assert not a.prep.eligible(10)
    msm_agent = make_agent()
    msm_agent.location.params.prep.target_model = ["cdc_msm"]
    make_relationship(msm_agent, p)
    assert msm_agent.is_msm()
    assert msm_agent.prep.eligible(10)

    # test pwid sex
    a.location.params.prep.target_model = ["pwid_sex"]
    p.location.params.prep.target_model = ["pwid_sex"]
    assert not a.prep.eligible(10)
    assert p.prep.eligible(10)

    # test pwid
    a.location.params.prep.target_model = ["pwid"]
    p.location.params.prep.target_model = ["pwid"]
    assert not a.prep.eligible(10)
    assert p.prep.eligible(10)
    p = make_agent(DU="Inj", SO="HM")
    p.location.params.prep.target_model = ["pwid"]
    assert p.prep.eligible(10)  # still eligible without partners

    # test ssp
    a.location.params.prep.target_model = ["ssp"]
    p.location.params.prep.target_model = ["ssp"]
    assert not a.prep.eligible(10)
    assert not p.prep.eligible(10)
    p.syringe_services.active = True
    assert p.prep.eligible(10)

    p.location.params.prep.target_model = ["ssp_sex"]
    assert not p.prep.eligible(10)
    make_relationship(p, msm_agent)
    assert p.prep.eligible(10)

    b = make_agent()
    b.location.params.prep.target_model = ["Allcomers"]
    assert b.prep.eligible(10)
    b.prep.active = True
    assert not b.prep.eligible(10)
    b.prep.active = False
    b.hiv.active = True
    assert not b.prep.eligible(10)
    b.hiv.active = False
    b.location.params.features.random_trial = True
    assert not b.prep.eligible(10)


@pytest.mark.unit
def test_enroll_prep_choice(make_agent, params):
    rand_gen = FakeRandom(-0.1)
    a = make_agent()
    a.location.params.prep.type = ["Oral", "Inj"]

    a.prep.enroll(rand_gen, 0)

    assert a.prep.active
    assert a.prep.last_dose_time == 0
    assert a.prep.adherent is True
    assert a.prep.type == "Inj"


@pytest.mark.unit
def test_enroll_prep_one(make_agent, params):
    rand_gen = FakeRandom(1.1)
    a = make_agent()
    a.location.params.prep.type = ["Oral"]

    a.prep.enroll(rand_gen, 0)

    assert a.prep.active
    assert a.prep.last_dose_time == 0
    assert a.prep.adherent is False
    assert a.prep.type == "Oral"


@pytest.mark.unit
def test_progress_inj_prep(make_agent, params, make_model):
    a = make_agent()
    a.location.params.prep.type = ["Inj"]
    assert a.prep.last_dose_time is None

    model = make_model()

    rand_gen = FakeRandom(1.1)
    a.prep.enroll(rand_gen, 0)
    assert a.prep.last_dose_time == 0

    # make time pass
    model.time = 12
    a.prep.progress(model)
    assert a.prep.last_dose_time is None


@pytest.mark.unit
def test_cap_as_prob(make_agent, make_model, params):
    model = make_model(params)
    a = make_agent()
    a.location.params.prep.cap_as_prob = True

    model.run_random = FakeRandom(0.2)
    a.prep.initiate(model)
    assert not a.prep.active

    model.run_random = FakeRandom(0.0)
    a = make_agent()
    a.prep.initiate(model)
    assert a.prep.active

    model.run_random = FakeRandom(0.4)
    a.location.params.prep.target_model = ["Racial"]
    a.prep.active = False
    a.prep.initiate(model)
    assert not a.prep.active

    model.run_random = FakeRandom(0.3)
    a.prep.initiate(model)
    assert a.prep.active


@pytest.mark.unit
def test_get_acquisition_risk_multiplier(make_agent):
    a = make_agent()
    t = 0

    assert a.prep.get_acquisition_risk_multiplier(t, "sex") == 1.0

    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.adherent = True
    a.prep.last_dose_time = -2

    assert a.prep.get_acquisition_risk_multiplier(t, "sex") == (
        1.0 - a.location.params.prep.efficacy.adherent
    )

    a.prep.type = "Inj"
    assert 0 < a.prep.get_acquisition_risk_multiplier(t, "sex") < 1.0


@pytest.mark.unit
def test_reach_multipliers_scale_prep_discontinuation_and_adherence(make_agent, make_model):
    """
    care_seeking's reach_* effects (2026-10-02): a reached agent's per-step PrEP
    discontinuation is multiplied by prep_discontinue_mult and the adherence
    draw by prep_adherence_mult; a drug-type-level `prep.discontinue` is read in
    preference to the sex-type level when present.
    """
    model = make_model()
    a = make_agent()
    sex = a.location.params.demographics[a.race].sex_type[a.sex_type]
    sex.prep.discontinue = 0.10
    sex.prep.adherence = 0.50
    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.time = 0
    a.prep.last_dose_time = 0
    Prep.counts[a.race] = 1

    # unreached: draw 0.05 < 0.10 -> discontinues
    a.care_reach = {}
    model.run_random = FakeRandom(0.05)
    a.prep.progress(model)
    assert a.prep.active is False

    # reached with mult 0.3: 0.03 -> draw 0.05 keeps the agent on PrEP
    a.prep.active = True
    a.prep.type = "Oral"
    Prep.counts[a.race] = 1
    a.care_reach = {"prep_discontinue_mult": 0.3}
    a.prep.progress(model)
    assert a.prep.active is True

    # adherence: draw 0.55 is above 0.50 (not adherent) but below 0.50 x 1.2
    a.prep.active = False
    Prep.counts[a.race] = 0
    a.care_reach = {"prep_adherence_mult": 1.2}
    a.prep.enroll(FakeRandom(0.55), 5)
    assert a.prep.adherent is True
    assert a.prep.ever is True

    # drug-type-level discontinuation wins when present
    sex.drug_type[a.drug_type].prep = {"discontinue": 0.9, "adherence": None}
    assert a.prep.demographic_value("discontinue") == pytest.approx(0.9)
    assert a.prep.demographic_value("adherence") == pytest.approx(0.50)


@pytest.mark.unit
def test_reached_agent_restarts_prep_without_an_offer(make_agent, make_model):
    model = make_model()
    model.params.features.care_seeking = True
    a = make_agent()
    a.location.params.features.care_seeking = True
    a.prep.ever = True
    a.prep.active = False
    a.care_offer = {"prep": False}
    a.care_reach = {"prep_restart_prob": 0.5}
    model.run_random = FakeRandom(0.3)
    a.prep.initiate(model)
    assert a.prep.active is True
    # never-enrolled agents do not restart
    b = make_agent()
    b.care_offer = {"prep": False}
    b.care_reach = {"prep_restart_prob": 0.5}
    b.prep.initiate(model)
    assert b.prep.active is False


@pytest.mark.unit
def test_late_phase_discontinuation_multiplier(make_agent, make_model):
    """
    prep.discontinue_late_after_steps / discontinue_late_mult (2026-10-02):
    after N steps on PrEP the per-step discontinuation hazard is scaled.
    """
    model = make_model()
    a = make_agent()
    sex = a.location.params.demographics[a.race].sex_type[a.sex_type]
    sex.prep.discontinue = 0.10
    a.location.params.prep.discontinue_late_after_steps = 6
    a.location.params.prep.discontinue_late_mult = 0.4
    a.care_reach = {}
    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.time = 0
    a.prep.last_dose_time = 0
    Prep.counts[a.race] = 1
    model.run_random = FakeRandom(0.06)      # 0.06 < 0.10 early -> stops; 0.06 >= 0.04 late -> stays
    model.time = 5
    a.prep.progress(model)
    assert a.prep.active is False
    a.prep.active = True; a.prep.type = "Oral"; a.prep.time = 0; Prep.counts[a.race] = 1
    model.time = 6
    a.prep.progress(model)
    assert a.prep.active is True


def _use_products(a, products):
    from titan.parse_params import ObjMap
    a.location.params.prep.use_products = True
    a.location.params.prep.products = ObjMap(products)
    a.location.params.prep.monitoring_setting = "primary_care"


CAB = {"type": "Inj", "share_of_starts": 1.0, "initiation_injections": 2, "interval_steps": 2,
       "attendance": 0.8, "return_prob": 0.5, "max_gap_steps": 2, "efficacy": 0.9,
       "tail_half_life_steps": 2.0, "test_at_injection": True}
LEN = {"type": "Inj", "share_of_starts": 1.0, "initiation_injections": 1, "interval_steps": 6,
       "attendance": 0.9, "return_prob": 0.0, "max_gap_steps": 1, "efficacy": 0.99,
       "tail_half_life_steps": 3.0, "test_at_injection": True}


@pytest.mark.unit
def test_cabotegravir_initiation_schedule_and_injection_visit(make_agent, make_model):
    """Two initiation injections one step apart, then every 2 steps; each scheduled injection is a PC visit with an HIV test."""
    model = make_model()
    a = make_agent()
    _use_products(a, {"cab_la": CAB})
    a.care_reach = {}
    a.prep.enroll(FakeRandom(0.0), 10)
    assert a.prep.product == "cab_la" and a.prep.type == "Inj"
    assert a.prep.injections == 1 and a.prep.next_due == 11
    a.care_encounter = {}; a.care_offer = {"hiv_test": False}; a.care_offer_mult = {"hiv_test": 1.0}
    model.time = 11
    model.run_random = FakeRandom(0.5)                 # 0.5 < attendance 0.8 -> attends
    a.prep.progress(model)
    assert a.prep.injections == 2 and a.prep.next_due == 13
    assert a.care_encounter["primary_care"] == 1
    assert a.care_offer["hiv_test"] is True and a.care_offer_mult["hiv_test"] >= 1e6
    # protection: on schedule 1 - 0.9; two steps past the due date with half-life 2 -> half the effect
    assert a.prep.get_acquisition_risk_multiplier(12, "sex") == pytest.approx(0.1)
    assert a.prep.get_acquisition_risk_multiplier(15, "sex") == pytest.approx(1 - 0.9 * 0.5)


@pytest.mark.unit
def test_missed_injection_returns_or_stops_after_max_gap(make_agent, make_model):
    model = make_model()
    a = make_agent()
    _use_products(a, {"cab_la": CAB})
    a.care_reach = {}
    a.prep.enroll(FakeRandom(0.0), 0)
    Prep.counts[a.race] = 1
    model.run_random = FakeRandom(0.95)                # misses (>= 0.8) and does not return (>= 0.5)
    model.time = 1
    a.prep.progress(model)
    assert a.prep.active and a.prep.overdue
    model.time = 2
    a.prep.progress(model)
    assert a.prep.active                              # 1 step late, max gap 2
    model.time = 3
    a.prep.progress(model)
    assert a.prep.active is False                     # 2 steps late -> stopped
    # a late return resets the schedule
    a.prep.enroll(FakeRandom(0.0), 20)
    Prep.counts[a.race] = 1
    model.run_random = FakeRandom(0.95); model.time = 21
    a.prep.progress(model)
    model.run_random = FakeRandom(0.3); model.time = 22        # 0.3 < return_prob 0.5
    a.prep.progress(model)
    assert a.prep.active and not a.prep.overdue and a.prep.next_due == 24


@pytest.mark.unit
def test_lenacapavir_six_month_interval_and_reach_attendance_multiplier(make_agent, make_model):
    model = make_model()
    a = make_agent()
    _use_products(a, {"lenacapavir": LEN})
    a.prep.enroll(FakeRandom(0.0), 0)
    assert a.prep.next_due == 6
    model.time = 5
    model.run_random = FakeRandom(0.0)
    a.prep.progress(model)
    assert a.prep.injections == 1                     # not yet due
    # 0.95 >= attendance 0.9 would miss, but a reached agent's multiplier 1.1 -> 0.99
    a.care_reach = {"prep_monitoring_mult": 1.1}
    model.time = 6
    model.run_random = FakeRandom(0.95)
    a.prep.progress(model)
    assert a.prep.injections == 2 and a.prep.next_due == 12


@pytest.mark.unit
def test_product_draw_by_share_and_oral_keeps_legacy_path(make_agent, make_model):
    model = make_model()
    a = make_agent()
    _use_products(a, {"oral": {"type": "Oral", "share_of_starts": 0.6},
                      "cab_la": dict(CAB, share_of_starts=0.3), "lenacapavir": dict(LEN, share_of_starts=0.1)})
    assert Prep.draw_product(a.location.params.prep.products, FakeRandom(0.5)) == "oral"
    assert Prep.draw_product(a.location.params.prep.products, FakeRandom(0.7)) == "cab_la"
    assert Prep.draw_product(a.location.params.prep.products, FakeRandom(0.95)) == "lenacapavir"
    a.prep.enroll(FakeRandom(0.5), 3)
    assert a.prep.type == "Oral" and a.prep.product == "oral" and a.prep.injections == 0


@pytest.mark.unit
def test_product_share_availability_ramp_and_remainder(make_agent):
    from titan.parse_params import ObjMap
    products = ObjMap({
        "oral": {"type": "Oral", "share_of_starts": -1.0},
        "len": dict(LEN, share_of_starts=0.02, available_from_step=162, share_ramp_start_step=162,
                    share_ramp_end_step=229, share_end=0.1875),
    })
    assert Prep.product_share(products["len"], 161) == 0.0
    assert Prep.product_share(products["len"], 162) == pytest.approx(0.02)
    assert Prep.product_share(products["len"], 229) == pytest.approx(0.1875)
    assert Prep.product_share(products["len"], 300) == pytest.approx(0.1875)
    # oral takes the remainder: at 229 a draw of 0.80 is above 0.8125? no -> oral; 0.90 -> len
    assert Prep.draw_product(products, FakeRandom(0.80), 229) == "oral"
    assert Prep.draw_product(products, FakeRandom(0.90), 229) == "len"
    assert Prep.draw_product(products, FakeRandom(0.99), 100) == "oral"       # len unavailable


@pytest.mark.unit
def test_full_protection_window_and_injection_exposure_efficacy(make_agent):
    a = make_agent()
    _use_products(a, {"cab_la": dict(CAB, efficacy=0.93, efficacy_injection=0.5, tail_full_steps=1,
                                     tail_half_life_steps=3.0)})
    a.prep.enroll(FakeRandom(0.0), 0)          # next due at 1 (second initiation injection)
    assert a.prep.get_acquisition_risk_multiplier(1, "sex") == pytest.approx(0.07)
    assert a.prep.get_acquisition_risk_multiplier(2, "sex") == pytest.approx(0.07)        # 1 step late: still full
    assert a.prep.get_acquisition_risk_multiplier(5, "sex") == pytest.approx(1 - 0.93 * 0.5)  # 3 steps beyond the window
    assert a.prep.get_acquisition_risk_multiplier(1, "injection") == pytest.approx(0.5)


@pytest.mark.unit
def test_set_stats_counts_oral_product_once(make_agent):
    """prep_oral was incremented by the Oral type and again by the product named
    oral (2026-10-06); the cash-flow layer prices oral PrEP from it."""
    from collections import defaultdict
    a = make_agent()
    a.prep.active = True
    a.prep.type = "Oral"
    a.prep.product = "oral"
    a.prep.time = 5
    stats = defaultdict(int)
    a.prep.set_stats(stats, 5)
    assert stats["prep"] == 1 and stats["prep_new"] == 1
    assert stats["prep_oral"] == 1
    assert stats["prep_injectable"] == 0

    a.prep.type = "Inj"
    a.prep.product = "cab_la"
    stats = defaultdict(int)
    a.prep.set_stats(stats, 6)
    assert stats["prep_injectable"] == 1 and stats["prep_cab_la"] == 1
    assert stats["prep_oral"] == 0


# ---------------------------------------------- 2026-10-07 clinical review changes

@pytest.mark.unit
def test_indicated_pwid_sharing_and_oud_active(make_agent):
    """pwid_sharing: currently injecting with a partnership that allows injection
    acts; oud_active: agent.has_oud. Neither is met by a sexual partnership or by
    remission."""
    a = make_agent(DU="Inj")
    a.location.params.prep.target_model = ["pwid_sharing"]
    assert not a.prep.indicated()                       # no partners
    sex = make_agent(DU="NonInj")
    Relationship(a, sex, 10, bond_type="Sex")
    assert not a.prep.indicated()                       # a sexual partner is not sharing
    inj = make_agent(DU="Inj")
    Relationship(a, inj, 10, bond_type="Inj")
    assert a.prep.indicated()
    b = make_agent(DU="NonInj")
    Relationship(b, inj, 10, bond_type="Inj")
    assert not b.prep.indicated()                       # not currently injecting

    c = make_agent()
    c.location.params.prep.target_model = ["oud_active"]
    c.has_oud = True
    assert c.prep.indicated()
    c.has_oud = False                                   # remission
    assert not c.prep.indicated()


@pytest.mark.unit
def test_eligible_needs_indication_and_hiv_negative(make_agent):
    a = make_agent(DU="Inj")
    a.location.params.prep.target_model = ["pwid"]
    a.location.params.prep.start_time = 0
    assert a.prep.eligible(5)
    a.hiv.active = True
    assert a.prep.indicated() and not a.prep.eligible(5)
    b = make_agent(DU="NonInj")
    b.location.params.prep.target_model = ["pwid"]
    assert not b.prep.eligible(5)


@pytest.mark.unit
def test_prep_indicated_stat_counts_hiv_negative_indicated_agents(make_agent):
    from collections import defaultdict
    a = make_agent(DU="Inj")
    a.location.params.prep.target_model = ["pwid"]
    stats = defaultdict(int)
    a.prep.set_stats(stats, 3)
    assert stats["prep_indicated"] == 1
    a.prep.active = True; a.prep.type = "Oral"; a.prep.time = 1
    stats = defaultdict(int)
    a.prep.set_stats(stats, 3)
    assert stats["prep_indicated"] == 1 and stats["prep"] == 1   # on PrEP still counts
    a.hiv.active = True
    stats = defaultdict(int)
    a.prep.set_stats(stats, 3)
    assert stats["prep_indicated"] == 0
    b = make_agent(DU="NonInj")
    b.location.params.prep.target_model = ["pwid"]
    stats = defaultdict(int)
    b.prep.set_stats(stats, 3)
    assert stats["prep_indicated"] == 0


@pytest.mark.unit
def test_oral_prep_injection_route_efficacy(make_agent):
    """efficacy_injection applies to injection exposure only; -1 means the
    sexual value (the engine default, so earlier scenarios are unchanged)."""
    a = make_agent()
    a.prep.active = True; a.prep.type = "Oral"; a.prep.last_dose_time = 0
    prep = a.location.params.prep
    prep.efficacy["adherent"] = 0.99
    prep.efficacy["non_adherent"] = 0.0
    a.prep.adherent = True
    assert a.prep.get_acquisition_risk_multiplier(1, "injection") == pytest.approx(0.01)
    prep.efficacy_injection["adherent"] = 0.70
    prep.efficacy_injection["non_adherent"] = 0.0
    assert a.prep.get_acquisition_risk_multiplier(1, "injection") == pytest.approx(0.30)
    assert a.prep.get_acquisition_risk_multiplier(1, "sex") == pytest.approx(0.01)
    a.prep.adherent = False
    assert a.prep.get_acquisition_risk_multiplier(1, "injection") == pytest.approx(1.0)
    assert a.prep.get_acquisition_risk_multiplier(1, "sex") == pytest.approx(1.0)


@pytest.mark.unit
def test_reengagement_raises_return_after_a_missed_injection_from_its_start(make_agent, make_model):
    """return_prob 0.5 x mult 1.6 = 0.8: a draw of 0.7 returns only once the
    programme has started; off (start -1) behaves exactly as before."""
    model = make_model()
    for start, expect_return in ((-1, False), (50, False), (2, True)):
        a = make_agent()
        _use_products(a, {"cab_la": CAB})
        a.care_reach = {}
        a.location.params.prep.reengagement_start_step = start
        a.location.params.prep.reengagement_return_mult = 1.6
        a.prep.enroll(FakeRandom(0.0), 0)
        Prep.counts[a.race] = 1
        model.run_random = FakeRandom(0.95); model.time = 1   # misses the due injection
        a.prep.progress(model)
        assert a.prep.overdue
        model.run_random = FakeRandom(0.7); model.time = 2    # 0.7 >= 0.5, < 0.8
        a.prep.progress(model)
        assert (not a.prep.overdue) is expect_return, (start, a.prep.overdue)


# ---------------------------------------- test-and-offer bundle, injectable-first (2026-10-09)

@pytest.mark.unit
def test_offer_after_negative_test_same_day_and_deferred(make_model, make_agent):
    model = make_model()
    model.time = 20
    a = make_agent(DU="Inj")
    a.location.params.prep.target_model = ["pwid"]
    a.location.params.prep.start_time = 0
    a.location.params.prep.acceptance = 0.5
    a.care_offer_mult = {"prep": 1.0}
    Prep.counts[a.race] = 0
    # declines (0.7 >= acceptance 0.5): offered, not started
    model.run_random = FakeRandom(0.7)
    a.prep.offer_after_negative_test(model, same_day_prob=1.0, return_prob=1.0)
    assert a.prep.bundle_offered_this_step and not a.prep.active
    # accepts, same day
    model.run_random = FakeRandom(0.1)
    a.prep.offer_after_negative_test(model, same_day_prob=1.0, return_prob=1.0)
    assert a.prep.active and a.prep.time == 20 and a.prep.started_bundled_this_step == "same_day"
    # deferred: accepted (0.1 < 0.5) but 0.1 >= same_day 0.0 -> pending next step
    b = make_agent(DU="Inj")
    b.location.params.prep.target_model = ["pwid"]; b.location.params.prep.start_time = 0
    b.location.params.prep.acceptance = 0.5; b.care_offer_mult = {"prep": 1.0}
    b.prep.offer_after_negative_test(model, same_day_prob=0.0, return_prob=0.4)
    assert not b.prep.active and b.prep.pending_start == 21 and b.prep.pending_return_prob == 0.4
    model.time = 21
    model.run_random = FakeRandom(0.6)            # does not return (0.6 >= 0.4)
    b.prep.update_agent(model)
    assert not b.prep.active and b.prep.pending_start is None
    c = make_agent(DU="Inj")
    c.location.params.prep.target_model = ["pwid"]; c.location.params.prep.start_time = 0
    c.location.params.prep.acceptance = 0.5; c.care_offer_mult = {"prep": 1.0}
    model.time = 30; model.run_random = FakeRandom(0.1)
    c.prep.offer_after_negative_test(model, same_day_prob=0.0, return_prob=0.4)
    model.time = 31; model.run_random = FakeRandom(0.2)   # returns (0.2 < 0.4)
    c.prep.update_agent(model)
    assert c.prep.active and c.prep.time == 31 and c.prep.started_bundled_this_step == "deferred"
    from collections import defaultdict
    stats = defaultdict(int); c.prep.set_stats(stats, 31)
    assert stats["prep_new_bundled"] == 1 and stats["prep_new_deferred"] == 1 and stats["prep_new_same_day"] == 0
    # no indication -> no offer
    d = make_agent(DU="NonInj")
    d.location.params.prep.target_model = ["pwid"]; d.location.params.prep.start_time = 0
    d.prep.offer_after_negative_test(model, 1.0, 1.0)
    assert not d.prep.bundle_offered_this_step and not d.prep.active


@pytest.mark.unit
def test_injectable_first_for_people_who_inject(make_agent):
    from titan.parse_params import ObjMap
    products = ObjMap({"oral": {"type": "Oral", "share_of_starts": -1.0},
                       "cab_la": {"type": "Inj", "share_of_starts": 0.1},
                       "lenacapavir": {"type": "Inj", "share_of_starts": 0.2}})
    pp = ObjMap({"injectable_first_pwid_share": 0.8, "injectable_first_start_step": 133})
    # off: draw 0.5 falls in oral's 0.7 remainder
    assert Prep.draw_product(products, FakeRandom(0.5), 200) == "oral"
    # on for people who inject from step 133: injectables take 0.8 (cab 0.8/3 = 0.267, len 0.533),
    # oral 0.2; cumulated in product order: oral [0, 0.2), cab [0.2, 0.467), len [0.467, 1]
    assert Prep.draw_product(products, FakeRandom(0.1), 200, drug_type="Inj", prep_params=pp) == "oral"
    assert Prep.draw_product(products, FakeRandom(0.3), 200, drug_type="Inj", prep_params=pp) == "cab_la"
    assert Prep.draw_product(products, FakeRandom(0.5), 200, drug_type="Inj", prep_params=pp) == "lenacapavir"
    assert Prep.draw_product(products, FakeRandom(0.9), 200, drug_type="Inj", prep_params=pp) == "lenacapavir"
    # not yet started, or not injecting: unchanged
    assert Prep.draw_product(products, FakeRandom(0.5), 100, drug_type="Inj", prep_params=pp) == "oral"
    assert Prep.draw_product(products, FakeRandom(0.5), 200, drug_type="NonInj", prep_params=pp) == "oral"
