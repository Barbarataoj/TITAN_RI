"""
The Phase 7 arm generator (scripts/make_phase7_scenarios.py, 2026-09-29): the
arithmetic each template asked Jun to do by hand, pinned so a refit cannot
silently change what an arm means.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "make_phase7_scenarios", REPO / "scripts" / "make_phase7_scenarios.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


def baseline(reinit=0.36, discontinue=0.078):
    return {
        "moud": {"reinit_prob": reinit, "discontinue_per_month": discontinue,
                 "post_overdose_rr": 1.0},
        "hiv_testing": {"moud_early_mult": 3.0, "moud_stable_mult": 1.5},
        "syringe_services": {"timeline": {"ssp_open": {"num_slots_start": 71,
                                                       "num_slots_stop": 71,
                                                       "risk": 0.16}}},
        "untouched": {"value": 1},
    }


@pytest.mark.unit
def test_scenario_2_targets_the_ed_buprenorphine_linkage():
    arms = gen.build_arms(baseline(reinit=0.36))
    s2, _ = arms["scenario_2_postod_linkage.yml"]
    assert s2["moud"]["post_overdose_rr"] == pytest.approx(0.78 / 0.36)
    assert s2["moud"]["reinit_prob"] == 0.36          # scenario 2 moves only the multiplier
    assert s2["untouched"] == {"value": 1}


@pytest.mark.unit
def test_scenario_2_floors_at_one_when_the_baseline_already_exceeds_the_target():
    arms = gen.build_arms(baseline(reinit=0.9))
    s2, notes = arms["scenario_2_postod_linkage.yml"]
    assert s2["moud"]["post_overdose_rr"] == 1.0
    assert any("WARNING" in n for n in notes)


@pytest.mark.unit
def test_ceiling_arm_saturates_and_respects_the_declared_max():
    s2b, _ = gen.build_arms(baseline(reinit=0.36))["scenario_2b_universal_linkage.yml"]
    assert s2b["moud"]["post_overdose_rr"] == pytest.approx(1 / 0.36)
    s2b_low, _ = gen.build_arms(baseline(reinit=0.02))["scenario_2b_universal_linkage.yml"]
    assert s2b_low["moud"]["post_overdose_rr"] == gen.POST_OVERDOSE_RR_MAX


@pytest.mark.unit
def test_scenario_3_doubles_the_initiation_rate_not_the_coverage():
    s3, notes = gen.build_arms(baseline(reinit=0.36, discontinue=0.078))["scenario_3_bup_expansion.yml"]
    assert s3["moud"]["reinit_prob"] == pytest.approx(2.11 * 0.36)
    # 0.36/(0.36+0.078) = 0.822 -> 0.7596/(0.7596+0.078) = 0.907
    assert any("0.822 -> 0.907" in n for n in notes)
    s3_hi, _ = gen.build_arms(baseline(reinit=0.6))["scenario_3_bup_expansion.yml"]
    assert s3_hi["moud"]["reinit_prob"] == 1.0


@pytest.mark.unit
def test_scenario_4_moves_reach_only():
    s4, _ = gen.build_arms(baseline())["scenario_4_ssp_saturation.yml"]
    ssp = s4["syringe_services"]["timeline"]["ssp_open"]
    assert ssp["num_slots_start"] == ssp["num_slots_stop"] == gen.SSP_SLOTS
    assert isinstance(ssp["num_slots_start"], int)
    assert ssp["risk"] == 0.16


@pytest.mark.unit
def test_scenario_5_sets_both_testing_multipliers():
    s5, _ = gen.build_arms(baseline())["scenario_5_moud_hiv_testing.yml"]
    assert s5["hiv_testing"]["moud_early_mult"] == 6.0
    assert s5["hiv_testing"]["moud_stable_mult"] == 3.8


@pytest.mark.unit
def test_scenario_6_holds_the_linkage_probability_by_default():
    arms = gen.build_arms(baseline(reinit=0.36))
    s6, notes = arms["scenario_6_combination.yml"]
    reinit3 = 2.11 * 0.36
    assert s6["moud"]["reinit_prob"] == pytest.approx(reinit3)
    assert s6["moud"]["post_overdose_rr"] == pytest.approx(0.78 / reinit3)
    assert s6["hiv_testing"]["moud_stable_mult"] == 3.8
    assert not any("CLAMPED" in n for n in notes)

    own = gen.build_arms(baseline(reinit=0.36), own_settings=True)["scenario_6_combination.yml"][0]
    assert own["moud"]["post_overdose_rr"] == pytest.approx(0.78 / 0.36)


@pytest.mark.unit
def test_written_files_round_trip(tmp_path):
    base = baseline()
    arms = gen.build_arms(base)
    for name, (scenario, notes) in arms.items():
        path = tmp_path / name
        gen.write_scenario(path, scenario, notes, Path("scenario_1_baseline.yml"))
        assert gen.load_yaml(path) == scenario
        assert path.read_text().startswith("# GENERATED")
