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
    fields = {"prep_bundle_prob": 0.0, "prep_bundle_prob_unwilling": 0.0, "prep_bundle_start_step": -999,
              "prep_same_day_prob": 1.0, "prep_deferred_return_prob": 1.0,
              "reach_prep_bundle_prob": -1.0, "reach_prep_deferred_return_prob": -1.0, "reach_prep_same_day_prob": -1.0}
    fields.update(setting)
    model.params.care_seeking = ObjMap({"setting": {"primary_care": fields}})
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
    # unwilling provider, reached by a programme with a floor: PrEP
    model, c = _bundle_setup(make_model, make_agent, willing=False, prev=model, prep_bundle_prob=1.0, reach_prep_bundle_prob=1.0)
    c.care_reach = {"prep_adherence_mult": 1.0}
    c.hiv_testing.update_agent(model)
    assert c.prep.active
    # not yet started (dated arm): no PrEP
    model, d = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0, prep_bundle_start_step=200)
    d.hiv_testing.update_agent(model)
    assert not d.prep.active and not d.prep.bundle_offered_this_step
    model.params.features.care_seeking = False



@pytest.mark.unit
def test_bundle_deferred_delivery_uses_the_programme_return_when_reached(make_model, make_agent):
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, prep_deferred_return_prob=0.3, reach_prep_deferred_return_prob=0.8)
    a.hiv_testing.update_agent(model)
    assert not a.prep.active and a.prep.pending_start == 141 and a.prep.pending_return_prob == 0.3
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, prep_deferred_return_prob=0.3, reach_prep_deferred_return_prob=0.8)
    b.care_reach = {"prep_adherence_mult": 1.0}
    b.hiv_testing.update_agent(model)
    assert b.prep.pending_return_prob == 0.8
    model.params.features.care_seeking = False


@pytest.mark.unit
def test_bundle_same_day_share_uses_the_programme_value_when_reached(make_model, make_agent):
    """Setting: deferred (same-day 0); the navigator programme makes it same-day for reached agents."""
    model, a = _bundle_setup(make_model, make_agent, willing=True, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, reach_prep_same_day_prob=1.0)
    a.hiv_testing.update_agent(model)
    assert not a.prep.active and a.prep.pending_start == 141
    model, b = _bundle_setup(make_model, make_agent, willing=True, prev=model, prep_bundle_prob=1.0,
                             prep_same_day_prob=0.0, reach_prep_same_day_prob=1.0)
    b.care_reach = {"prep_adherence_mult": 1.0}
    b.hiv_testing.update_agent(model)
    assert b.prep.active and b.prep.started_bundled_this_step == "same_day"
    model.params.features.care_seeking = False
