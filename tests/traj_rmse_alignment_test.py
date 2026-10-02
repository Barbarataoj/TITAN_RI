"""_traj_rmse must align the model series to the target by calendar year (head), not by tail."""
import numpy as np

from titan.calibration.optuna_calibration import _traj_rmse


def test_equal_length_is_plain_rmse():
    model = np.array([0.1, 0.2, 0.3])
    target = np.array([0.1, 0.2, 0.4])
    assert np.isclose(_traj_rmse(model, target), np.sqrt(0.01 / 3))


def test_longer_model_series_is_scored_on_its_first_years():
    # A 20-year run scored against an 11-year target: years 1-11 match exactly,
    # years 12-20 drift. Head alignment scores 0; tail alignment would not.
    target = np.linspace(0.5, 0.8, 11)
    model = np.concatenate([target, np.full(9, 0.95)])
    assert _traj_rmse(model, target) == 0.0
    assert _traj_rmse(model[::-1], target) > 0.0


def test_shorter_model_series_uses_available_years():
    target = np.array([0.1, 0.2, 0.3, 0.4])
    model = np.array([0.1, 0.2])
    assert _traj_rmse(model, target) == 0.0


def test_empty_model_series_is_max_penalty():
    target = np.array([0.3, 0.4])
    assert np.isclose(_traj_rmse(None, target), np.sqrt(np.mean(target ** 2)))
    assert np.isclose(_traj_rmse(np.array([]), target), np.sqrt(np.mean(target ** 2)))
