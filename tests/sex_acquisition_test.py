"""
Which cells of `partnership.sex.acquisition` the model can actually reach.

The per-act table is indexed `[HIV- partner's sex_type][HIV- partner's sex role]`
(titan/exposures/hiv.py:429). Two things narrow it sharply in TITAN_RI:

  * `sex_role.init` is a point mass in every stratum -- HF receptive 1.0, HM
    insertive 1.0, MSM and MSMW versatile 1.0 -- so an agent's role never varies.
  * `classes.sex_types[*].sleeps_with` governs which pairs can bond at all, and
    HM only sleeps with HF.

A versatile agent takes the opposite role to its partner, so the reachable set is
much smaller than the 12 cells the table defines. Cells that cannot be reached
are dead config: they look authoritative in the YAML and change nothing, which is
how the MSM insertive value sat 4.5x off the published figure without effect.

This test pins the reachable set. If `sex_role.init` stops being degenerate, or a
`sleeps_with` entry changes, it fails and the per-act values have to be revisited
-- particularly MSMW insertive, which is only correct as a VAGINAL figure because
the sole partner type that can put an MSMW in the insertive role is HF.
"""

import itertools

import pytest
import yaml

SCENARIO = "calibrated_scenario_multi_5obj.yml"


@pytest.fixture(scope="module")
def scenario():
    with open(SCENARIO) as f:
        return yaml.safe_load(f)


def sex_roles(scenario):
    """The single role each sex_type can hold, from the point-mass init."""
    roles = {}
    for race in scenario["demographics"].values():
        for sx_name, sx in (race.get("sex_type") or {}).items():
            init = (sx.get("sex_role") or {}).get("init")
            if init is None:
                continue
            held = [r for r, p in init.items() if p > 0]
            roles.setdefault(sx_name, set()).update(held)
    return roles


def resolve(agent_role, partner_role):
    """titan/exposures/hiv.py:418-426 -- versatile takes the opposite position."""
    if partner_role != "versatile":
        return partner_role
    if agent_role == "insertive":
        return "receptive"
    if agent_role == "receptive":
        return "insertive"
    return "versatile"


def reachable_cells(scenario):
    """(sex_type, role) pairs an HIV- agent can actually present to the table."""
    roles = sex_roles(scenario)
    sleeps_with = {k: set(v["sleeps_with"]) for k, v in scenario["classes"]["sex_types"].items()}
    cells = set()
    for pos, neg in itertools.product(sleeps_with, repeat=2):
        # a bond needs both directions to permit it
        if neg not in sleeps_with[pos] or pos not in sleeps_with[neg]:
            continue
        for pos_role, neg_role in itertools.product(roles[pos], roles[neg]):
            cells.add((neg, resolve(pos_role, neg_role)))
    return cells


def test_every_sex_role_is_a_point_mass(scenario):
    """The premise of this whole test module."""
    for sx_type, held in sex_roles(scenario).items():
        assert len(held) == 1, f"{sx_type} can hold more than one role: {held}"


def test_reachable_cells_are_exactly_these(scenario):
    assert reachable_cells(scenario) == {
        ("HF", "receptive"),      # from HM, and from MSMW resolved insertive
        ("HM", "insertive"),      # from HF
        ("MSM", "versatile"),     # from MSM and MSMW, both versatile
        ("MSMW", "versatile"),    # from MSM and MSMW
        ("MSMW", "insertive"),    # ONLY from HF -- so this cell is vaginal, not anal
    }


def test_reachable_values_are_the_published_per_act_figures(scenario):
    """
    Patel et al. / CDC per-act estimates, as probabilities:

        receptive anal    0.0138      insertive anal     0.0011
        receptive vaginal 0.0008      insertive vaginal  0.0004

    Versatile is the mean of the two anal figures, 0.00745, since a versatile
    agent alternates position by act.
    """
    acq = scenario["partnership"]["sex"]["acquisition"]
    expected = {
        ("HF", "receptive"): 0.0008,
        ("HM", "insertive"): 0.0004,
        ("MSM", "versatile"): 0.00745,
        ("MSMW", "versatile"): 0.00745,
        ("MSMW", "insertive"): 0.0004,
    }
    for (sx_type, role), want in expected.items():
        assert acq[sx_type][role] == pytest.approx(want), f"{sx_type}/{role}"


def test_unreachable_cells_still_carry_the_published_values(scenario):
    """
    Dead config today, but it stops being dead the moment sex_role.init changes,
    and a wrong value there is invisible until then.

    MSMW insertive is deliberately absent: as a reachable cell it means vaginal
    insertive (0.0004), while a non-degenerate sex_role would also route
    male-partner anal insertive (0.0011) through it. That collision is the reason
    test_every_sex_role_is_a_point_mass exists.
    """
    acq = scenario["partnership"]["sex"]["acquisition"]
    assert acq["MSM"]["receptive"] == pytest.approx(0.0138)
    assert acq["MSM"]["insertive"] == pytest.approx(0.0011)
    assert acq["MSMW"]["receptive"] == pytest.approx(0.0138)
    assert acq["HF"]["insertive"] == pytest.approx(0.0004)
    assert acq["HM"]["receptive"] == pytest.approx(0.0008)
