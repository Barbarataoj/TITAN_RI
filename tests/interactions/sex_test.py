import pytest

from titan.interactions import Sex
from titan.agent import Relationship

from conftest import FakeRandom
from titan.parse_params import ObjMap


@pytest.mark.unit
def test_sex_transmission(make_model, make_agent):
    model = make_model()
    model.time = model.params.hiv.start_time
    a = make_agent()
    a.sex_role = "insertive"
    p = make_agent()
    p.sex_role = "receptive"
    a.partners["Sex"] = set()
    p.partners["Sex"] = set()
    rel = Relationship(a, p, 10, bond_type="Sex")

    a.hiv.active = True
    a.hiv.time = model.time  # acute

    rel.total_sex_acts = 0
    model.params.calibration.acquisition = 10

    model.params.calibration.acquisition = 5
    model.params.calibration.sex.act = 10
    model.run_random = FakeRandom(0.6)
    a.location.params.partnership.sex.frequency = ObjMap(
        {"Sex": {"type": "bins", "bins": {1: {"prob": 1.0, "min": 10, "max": 37}}}}
    )
    p.location.params.partnership.sex.frequency = ObjMap(
        {"Sex": {"type": "bins", "bins": {1: {"prob": 1.0, "min": 10, "max": 37}}}}
    )
    # test partner becomes
    Sex.interact(model, rel)
    assert p.hiv.active

    p.hiv.active = False

    a.location.params.partnership.sex.frequency = (
        p.location.params.partnership.sex.frequency
    ) = ObjMap(
        {
            "Sex": {
                "type": "distribution",
                "distribution": {
                    "dist_type": "set_value",
                    "vars": {1: {"value": 1, "value_type": "int"}},
                },
            }
        }
    )
    Sex.interact(model, rel)
    assert p.hiv.active
    # before hiv start time
    p.hiv.active = False
    model.time = model.params.hiv.start_time - 1
    Sex.interact(model, rel)
    assert not p.hiv.active


@pytest.mark.unit
def test_sex_num_acts(make_model, make_agent, make_relationship, params):
    params.hiv.dx.risk_reduction.sex = 1.0
    model = make_model()
    model.time = model.params.hiv.start_time
    model.np_random = FakeRandom(1.0)
    a = make_agent()
    p = make_agent()
    a.partners["Sex"] = set()
    p.partners["Sex"] = set()
    rel_Sex = Relationship(a, p, 10, bond_type="Sex")

    assert Sex.get_num_acts(model, rel_Sex) > 0

    a.hiv.active = True
    a.hiv.dx = True

    # test nothing happens
    assert Sex.get_num_acts(model, rel_Sex) == 0

    a.location.params.partnership.sex.frequency = (
        p.location.params.partnership.sex.frequency
    ) = ObjMap(
        {
            "Sex": {
                "type": "distribution",
                "distribution": {
                    "dist_type": "set_value",
                    "vars": {1: {"value": 0, "value_type": "int"}},
                },
            }
        }
    )
    assert Sex.get_num_acts(model, rel_Sex) == 0


@pytest.mark.unit
def test_msm_act_scalar_replaces_the_heterosexual_one(make_model, make_agent, params):
    """
    calibration.sex.act_msm (2026-09-29) scales partnerships with an MSM/MSMW
    agent in place of calibration.sex.act; -1 means "use act". The fitted
    heterosexual scalar must not drive the male-male route, whose per-act
    probability is ~12x higher.
    """
    params.hiv.dx.risk_reduction.sex = 1.0
    model = make_model()
    model.time = model.params.hiv.start_time
    model.np_random = FakeRandom(1.0)   # poisson -> round(mean)
    model.run_random = FakeRandom(1.0)  # never < p_safe_sex -> no condom reduction

    freq = ObjMap({"Sex": {"type": "distribution", "distribution": {
        "dist_type": "set_value", "vars": {1: {"value": 10, "value_type": "int"}}}}})

    def rel_between(sex_a, sex_b):
        a, p = make_agent(), make_agent()
        a.sex_type, p.sex_type = sex_a, sex_b
        a.partners["Sex"] = set()
        p.partners["Sex"] = set()
        a.location.params.partnership.sex.frequency = freq
        p.location.params.partnership.sex.frequency = freq
        a.location.params.demographics[a.race].sex_type[a.sex_type].safe_sex["Sex"].prob = 0.0
        return Relationship(a, p, 10, bond_type="Sex")

    het = rel_between("HM", "HF")
    msm = rel_between("MSM", "MSM")

    model.params.calibration.sex.act = 2.0
    model.params.calibration.sex.act_msm = -1.0
    assert Sex.get_num_acts(model, het) == 20
    assert Sex.get_num_acts(model, msm) == 20

    model.params.calibration.sex.act_msm = 0.5
    assert Sex.get_num_acts(model, het) == 20
    assert Sex.get_num_acts(model, msm) == 5
    # MSMW on either side counts as male-male
    mixed = rel_between("HM", "MSMW")
    assert Sex.get_num_acts(model, mixed) == 5
