import pytest

from conftest import FakeRandom


@pytest.mark.unit
def test_update_haart_t1(make_model, make_agent):
    model = make_model()
    model.time = 1
    a = make_agent(race="white_nonhispanic")

    a.hiv.active = True

    # nothing happens, not tested
    a.haart.update_agent(model)
    assert a.haart.adherent is False
    assert a.haart.active is False

    # t0 agent initialized HAART
    a.hiv.dx = True
    a.hiv.dx_time = model.time

    # go on haart
    model.run_random = FakeRandom(
        -0.1
    )  # means this will always be less than params even though not possible in reality
    a.haart.update_agent(model)

    assert a.haart.adherent is True
    assert a.haart.active

    # go off haart
    a.haart.update_agent(model)

    assert a.haart.adherent is False
    assert a.haart.active is False

    # Try haart cap with low cap, one agent on haart and one diagnosed agent. Nothing happens
    a.location.params.haart.use_cap = True
    a.haart.counts[a.race][a.sex_type] = 1
    a.hiv.dx_counts[a.race][a.sex_type] = 5
    a.location.params.demographics[a.race].sex_type[a.sex_type].drug_type[
        a.drug_type
    ].haart.prob = 0.1
    a.haart.update_agent(model)
    assert a.haart.active is False
    assert a.haart.adherent is False

    # Increase cap. Agent goes on haart
    a.location.params.demographics[a.race].sex_type[a.sex_type].drug_type[
        a.drug_type
    ].haart.cap = 2.0
    a.haart.update_agent(model)
    assert a.haart.active
    assert a.haart.adherent is True

    # falls off adherence
    model.run_random = FakeRandom(1.0)
    a.location.params.demographics[a.race].sex_type[a.sex_type].drug_type[
        a.drug_type
    ].haart.adherence.discontinue = 5.0
    a.haart.update_agent(model)
    assert a.haart.active
    assert a.haart.adherent is False


def test_update_haart_dx_time(make_model, make_agent):
    model = make_model()
    model.time = 1
    a = make_agent(race="white_nonhispanic")
    model.run_random = FakeRandom(0.5)

    a.hiv.active = True
    a.hiv.dx = True
    a.hiv.dx_time = model.time  # agent dx duration is 0
    a.haart.update_agent(model)
    assert not a.haart.active

    a.hiv.dx_time -= 10
    a.haart.update_agent(model)
    assert a.haart.active


def test_update_haart_reinit(make_model, make_agent):
    model = make_model()
    model.time = 1
    a = make_agent(race="white_nonhispanic")

    a.hiv.active = True
    a.hiv.dx = True
    a.hiv.dx_time = model.time - 1  # make duration 1, would pass if not reinit
    a.haart.ever = True  # uses reinit prob
    a.haart.update_agent(model)
    assert not a.haart.active

    a.location.params.demographics[a.race].sex_type[a.sex_type].drug_type[
        a.drug_type
    ].haart.reinit.prob = 1.0
    a.haart.update_agent(model)
    assert a.haart.active


@pytest.mark.unit
def test_first_month_art_window_reaches_a_testing_diagnosis(make_model, make_agent):
    """HAART updates before hiv_testing in a step, so a diagnosis is first seen by HAART
    at time-since-diagnosis 1. The scenario's first-month window must cover duration 1,
    or the first-month ART probability never applies (defect found 2026-10-13)."""
    import yaml
    from pathlib import Path
    from titan import features
    from titan.parse_params import ObjMap
    order = [f.name for f in features.BaseFeature.__subclasses__()]
    assert order.index("haart") < order.index("hiv_testing")
    src = yaml.safe_load(Path(__file__).resolve().parents[2].joinpath("titan", "calibration", "scenario_source.yml").read_text())
    enroll = src["demographics"]["drug_type"]["constants"]["haart"]["enroll"]
    first = enroll["enroll_0"]
    assert first["start"] <= 1 < first["stop"], first
    model = make_model()
    a = make_agent(race="white_nonhispanic")
    hp = a.location.params.demographics[a.race].sex_type[a.sex_type].drug_type[a.drug_type].haart
    hp.enroll = ObjMap(enroll)
    model.time = 5
    a.hiv.active = True
    a.hiv.diagnose(model)            # hiv_testing at step 5, after HAART ran at 5
    model.time = 6
    model.run_random = FakeRandom(0.5 * (first["prob"] + enroll["enroll_1"]["prob"]))
    a.haart.update_agent(model)
    assert a.haart.active            # the first-month probability applied
