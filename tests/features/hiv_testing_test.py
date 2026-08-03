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
