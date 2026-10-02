"""
The primary-care arm generator (scripts/make_phase7_pc_arms.py, 2026-10-02):
each arm changes only care_seeking.setting.primary_care (and, for patient-side
components, the acceptance scalars); dated arms carry timeline entries plus a
pool refresh; the arithmetic is pinned.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("make_phase7_pc_arms", REPO / "scripts" / "make_phase7_pc_arms.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


def setting(**kw):
    s = {"visit_rate_annual": 5.8, "p_willing_hiv_test": 0.246, "p_willing_prep": 0.079,
         "p_willing_moud": 1.0, "intensity_hiv_test": 0.108, "intensity_prep": 0.0052,
         "intensity_moud": 0.75, "pool_refresh_time": -9999}
    s.update(kw)
    return s


def baseline():
    return {
        "features": {"care_seeking": True},
        "care_seeking": {"setting": {"primary_care": setting(),
                                     "ED": setting(visit_rate_annual=2.0, p_willing_moud=0.07),
                                     "sud_treatment": setting(), "mental_health": setting()}},
        "hiv_testing": {"acceptance": 0.55},
        "prep": {"acceptance": 0.60},
        "moud": {"acceptance": 0.75},
        "untouched": {"value": 1},
    }


@pytest.mark.unit
def test_static_training_arm_moves_only_primary_care_willingness():
    arms = gen.build_arms(baseline(), static=True)
    s, notes = arms["pc1_training_detailing_central_static.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    assert pc["p_willing_hiv_test"] == pytest.approx(0.246 + 0.5 * 0.3 * (1 - 0.246))
    assert pc["intensity_hiv_test"] == pytest.approx(0.108 * 1.2)
    assert pc["p_willing_prep"] == pytest.approx(0.079 + 0.5 * 0.3 * (1 - 0.079))
    assert pc["intensity_prep"] == pytest.approx(0.0052)
    for other in ("ED", "sud_treatment", "mental_health"):
        assert s["care_seeking"]["setting"][other] == baseline()["care_seeking"]["setting"][other]
    assert s["hiv_testing"]["acceptance"] == 0.55 and s["prep"]["acceptance"] == 0.60
    assert s["untouched"] == {"value": 1}
    assert "timeline_scaling" not in s


@pytest.mark.unit
def test_dated_ehr_arm_uses_timeline_scalars_and_pool_refresh():
    arms = gen.build_arms(baseline(), start_step=133)
    s, notes = arms["pc2_ehr_opt_out_central.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    # parameters themselves untouched in a dated arm ...
    assert pc["p_willing_hiv_test"] == 0.246 and pc["intensity_hiv_test"] == 0.108
    # ... the timeline multiplies them at the start step ...
    tl = s["timeline_scaling"]["timeline"]
    w1 = 0.246 + 1.0 * 0.67 * (1 - 0.246)
    assert tl["pc_hiv_test_willing"]["scalar"] == pytest.approx(w1 / 0.246, abs=1e-5)
    assert tl["pc_hiv_test_willing"]["parameter"] == "care_seeking|setting|primary_care|p_willing_hiv_test"
    assert tl["pc_hiv_test_willing"]["start_time"] == 133
    assert tl["pc_hiv_test_intensity"]["scalar"] == pytest.approx(1.5)
    assert tl["pc_prep_intensity"]["scalar"] == pytest.approx(2.6)
    assert "pc_prep_willing" not in tl           # efficacy 0 for PrEP willingness in this arm
    assert tl["pc_hiv_test_acceptance"]["parameter"] == "care_seeking|setting|primary_care|acceptance_mult_hiv_test"
    assert tl["pc_hiv_test_acceptance"]["scalar"] == pytest.approx(0.65 / 0.55)
    assert s["hiv_testing"]["acceptance"] == 0.55            # the global scalar is never touched
    # ... and the pool is refreshed on that step
    assert pc["pool_refresh_time"] == 133
    assert s["features"]["timeline_scaling"] is True
    assert s["care_seeking"]["setting"]["ED"]["pool_refresh_time"] == -9999


@pytest.mark.unit
def test_navigation_arm_is_patient_side_only():
    arms = gen.build_arms(baseline(), static=True)
    s, notes = arms["pc3_navigation_same_day_central_static.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    assert pc["acceptance_mult_hiv_test"] == pytest.approx(0.83 / 0.55)
    assert pc["acceptance_mult_prep"] == pytest.approx(0.75 / 0.60)
    for other in ("ED", "sud_treatment", "mental_health"):
        assert s["care_seeking"]["setting"][other] == baseline()["care_seeking"]["setting"][other]
    assert s["hiv_testing"]["acceptance"] == 0.55 and s["prep"]["acceptance"] == 0.60
    assert s["moud"]["acceptance"] == 0.75


@pytest.mark.unit
def test_intensity_multiplier_is_capped_at_one():
    b = baseline()
    b["care_seeking"]["setting"]["primary_care"]["intensity_hiv_test"] = 0.8
    arms = gen.build_arms(b, static=True, variant="high")
    s, notes = arms["pc2_ehr_opt_out_high_static.yml"]
    assert s["care_seeking"]["setting"]["primary_care"]["intensity_hiv_test"] == 1.0
    assert any("capped at 1" in n for n in notes)


@pytest.mark.unit
def test_variants_are_ordered_low_central_high():
    vals = {}
    for v in ("low", "central", "high"):
        s, _ = gen.build_arms(baseline(), static=True, variant=v)[f"pc1_training_detailing_{v}_static.yml"]
        vals[v] = s["care_seeking"]["setting"]["primary_care"]["p_willing_hiv_test"]
    assert vals["low"] < vals["central"] < vals["high"] <= 1.0
