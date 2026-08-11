"""
Regression tests for the canonical Pareto pick.

select_canonical has silently chosen a bad calibration three times -- job 4586902
(PWID HIV prevalence 2.5x target), job 4743939 (HIV weighted at 0.085 of the
composite), and job 4830724 (a pick at ridoh_hiv_err 4.02 against a null value of
4.78). Each time the cause was the same: an objective whose front distribution is
heavy-tailed sets a wide normalizer, which mutes that objective for every trial in
the feasible region. Nothing in the suite covered this function until 2026-08-11.

Objective order is positional: (oud_rmse, ridoh_hiv_err, od_err, moud_traj_rmse,
inj_share_err).
"""

from types import SimpleNamespace

import pytest

from titan.calibration.optuna_calibration import select_canonical


def trial(number, values, pooled=1000):
    """Minimal stand-in for an optuna FrozenTrial.

    select_canonical reads only .values and .user_attrs; building real trials
    would need a storage backend and would not exercise anything extra.
    """
    return SimpleNamespace(
        number=number,
        values=list(values),
        user_attrs={"hiv_new_pooled": pooled},
    )


# Real members of job 4830724's Pareto front, objective values verbatim, trial
# numbers preserved. A synthetic front is a poor fixture here: the failure needs a
# near-null trial that is genuinely competitive on the other four objectives, and
# if the feasible trials are simply made worse everywhere else then the near-null
# trial SHOULD win and no screen ought to save it.
#
# The two that matter are 239 and 438. They are near-identical on oud, od, moud and
# inj; 239 fits HIV 2.2x better. The hot tail (113, 162, 204) pushes the
# COMPOSITE_CLIP_QUANTILE anchor to ~46, which flattens a 1.84-vs-4.02 difference
# into 0.05 of composite, and 438 takes it on rounding elsewhere. Removing the
# worse-than-null members pulls the anchor down to ~3 and hands the decision back
# to HIV.
NULL = 4.782049769711729

FRONT_4830724 = [
    #        oud     hiv       od      moud    inj
    (639, [0.1566, 0.3592, 0.0497, 0.3330, 0.1040]),
    (427, [0.0705, 0.6500, 0.0373, 0.0782, 0.0443]),
    (834, [0.0254, 1.0610, 0.0404, 0.0691, 0.0135]),
    (239, [0.0011, 1.8401, 0.0435, 0.0691, 0.0006]),
    (544, [0.0543, 2.0408, 0.0305, 0.0497, 0.0067]),
    (331, [0.0337, 2.2635, 0.0082, 0.1386, 0.0003]),
    (285, [0.0439, 2.7977, 0.0123, 0.0926, 0.0001]),
    (438, [0.0011, 4.0203, 0.0372, 0.0437, 0.0010]),
    (814, [0.0124, 5.3120, 0.0863, 0.0541, 0.0002]),
    (844, [0.0492, 8.3611, 0.0030, 0.0473, 0.0028]),
    (317, [0.0335, 113.3794, 0.0141, 0.0424, 0.0002]),
    (852, [0.0384, 161.9742, 0.0129, 0.0803, 0.0003]),
    (681, [0.0729, 204.1090, 0.4057, 0.0406, 0.0001]),
]


def heavy_tailed_front():
    return [trial(n, v) for n, v in FRONT_4830724]


@pytest.mark.unit
def test_heavy_tail_mutes_hiv_without_the_null_screen():
    """Documents the defect, so the fix below is demonstrably load-bearing."""
    best, _, _ = select_canonical(heavy_tailed_front(), min_events=1)
    assert best.number == 438
    # 84% of the way to a model that transmits nothing at all.
    assert best.values[1] > 0.8 * NULL


@pytest.mark.unit
def test_null_screen_rescales_and_hands_the_decision_back_to_hiv():
    """The screen removes the tail that was muting HIV, improving the winner.

    Asserted as an invariant rather than a trial number. Which member wins depends
    on the pool's quantiles, so it is not stable under subsampling: this 13-member
    slice picks 331 (hiv 2.26), while the full 174-member front picks 239 (hiv
    1.84). Both are the fix working. What is stable, and what the defect violated,
    is that screening out the worse-than-null members must not leave the pick with
    worse HIV than it had before.
    """
    front = heavy_tailed_front()
    unscreened, _, _ = select_canonical(front, min_events=1)
    screened, _, _ = select_canonical(front, min_events=1, hiv_null=NULL)

    assert screened.number != unscreened.number
    assert screened.values[1] < unscreened.values[1]
    assert screened.values[1] < NULL


@pytest.mark.unit
def test_null_screen_is_exclusive_at_the_boundary():
    """Exactly-null means zero transmission, which is not a calibration candidate."""
    front = [
        trial(1, [0.001, NULL, 0.001, 0.001, 0.001]),  # best everywhere but null HIV
        trial(2, [0.30, 4.00, 0.30, 0.30, 0.30]),
    ]
    best, _, _ = select_canonical(front, min_events=1, hiv_null=NULL)
    assert best.number == 2


@pytest.mark.unit
def test_falls_back_when_nothing_beats_null(capsys):
    """A run where no trial transmits must warn, not crash or silently pick."""
    front = [
        trial(1, [0.01, 5.0, 0.01, 0.01, 0.01]),
        trial(2, [0.02, 9.0, 0.02, 0.02, 0.02]),
    ]
    best, _, _ = select_canonical(front, min_events=1, hiv_null=NULL)
    assert best.number == 1
    assert "no eligible Pareto trial beat the zero-transmission null" in capsys.readouterr().out


@pytest.mark.unit
def test_cold_screen_still_applies_and_is_reported():
    """The two screens are independent: cold-side filtering survives the change."""
    front = [
        trial(1, [0.001, 0.5, 0.001, 0.001, 0.001], pooled=2),  # best, but inactive
        trial(2, [0.10, 1.0, 0.10, 0.10, 0.10], pooled=500),
    ]
    best, _, used_filter = select_canonical(front, min_events=100, hiv_null=NULL)
    assert used_filter
    assert best.number == 2


@pytest.mark.unit
def test_hiv_null_none_preserves_previous_behaviour():
    """Callers that do not pass hiv_null get exactly the pre-2026-08-11 result."""
    front = heavy_tailed_front()
    assert select_canonical(front, min_events=1)[0].number == (
        select_canonical(front, min_events=1, hiv_null=None)[0].number
    )
