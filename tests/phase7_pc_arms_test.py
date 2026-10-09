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
    # since 2026-10-09 the PrEP lever of this arm is the test-and-offer bundle,
    # not an independent per-visit multiplier
    assert "pc_prep_intensity" not in tl
    assert pc["prep_bundle_prob"] == 0.8 and pc["prep_bundle_start_step"] == 133
    assert "pc_prep_willing" not in tl           # efficacy 0 for PrEP willingness in this arm
    assert tl["pc_hiv_test_acceptance"]["parameter"] == "care_seeking|setting|primary_care|acceptance_mult_hiv_test"
    assert tl["pc_hiv_test_acceptance"]["scalar"] == pytest.approx(0.65 / 0.55)
    assert s["hiv_testing"]["acceptance"] == 0.55            # the global scalar is never touched
    # ... and the pool is refreshed on that step
    assert pc["pool_refresh_time"] == 133
    assert s["features"]["timeline_scaling"] is True
    assert s["care_seeking"]["setting"]["ED"]["pool_refresh_time"] == -9999


@pytest.mark.unit
def test_navigation_arm_is_a_staffed_programme_in_primary_care():
    """Since 2026-10-09: reach, same-day, warm handoff, adherence support and
    re-engagement, all in primary care; acceptance as reach multipliers on the
    baseline values; re-engagement scaled by reach (mean-field)."""
    b = baseline()
    b.setdefault("prep", {})["reengagement_start_step"] = -1
    b["prep"]["reengagement_return_mult"] = 1.0
    arms = gen.build_arms(b)
    s, notes = arms["pc3_navigation_same_day_central.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    assert pc["reach_prob"] == 0.25 and pc["reach_start_time"] == gen.DEFAULT_START_STEP
    assert pc["reach_acceptance_mult_hiv_test"] == pytest.approx(0.83 / 0.55)
    assert pc["reach_acceptance_mult_prep"] == pytest.approx(0.75 / b["prep"]["acceptance"])
    assert pc["reach_prep_same_day_prob"] == 1.0 and pc["reach_prep_deferred_return_prob"] == 0.8
    assert pc["reach_prep_adherence_mult"] == pytest.approx(1.26)
    assert s["prep"]["reengagement_return_mult"] == pytest.approx(1 + 1.9 * 0.25)
    assert s["prep"]["reengagement_start_step"] == gen.DEFAULT_START_STEP
    for other in ("ED", "sud_treatment", "mental_health"):
        assert s["care_seeking"]["setting"][other] == b["care_seeking"]["setting"][other]
    assert pc["p_willing_prep"] == b["care_seeking"]["setting"]["primary_care"]["p_willing_prep"]
    assert s["hiv_testing"]["acceptance"] == b["hiv_testing"]["acceptance"] and s["prep"]["acceptance"] == b["prep"]["acceptance"]
    assert b["prep"]["reengagement_return_mult"] == 1.0          # baseline not mutated
    static, _ = gen.build_arms(b, static=True)["pc3_navigation_same_day_central_static.yml"]
    assert static["care_seeking"]["setting"]["primary_care"]["reach_start_time"] == -9999


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


@pytest.mark.unit
def test_chia_programme_arm_sets_reach_fields_only_in_primary_care():
    b = baseline()
    arms = gen.build_arms(b, start_step=133)
    s, notes = arms["pc4_chia_central.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    assert pc["reach_start_time"] == 133
    assert pc["reach_prob"] == pytest.approx(0.15)
    assert pc["reach_visit_prob"] == pytest.approx(0.65)
    assert pc["reach_offer_prob_hiv_test"] == pytest.approx(0.027 + 0.65 * (0.50 - 0.027), abs=1e-3)
    assert pc["reach_offer_prob_prep"] == pytest.approx(0.35 * 0.35 * 0.78, abs=1e-3)
    assert pc["reach_intensity_mult_prep"] == pytest.approx((1 + 0.65 * 2.15) / 0.65, abs=1e-3)
    assert pc["reach_prep_discontinue_mult"] == pytest.approx(1 - 0.65 * 0.5, abs=1e-3)
    # W, I and acceptance of primary care untouched; other settings untouched; no timeline
    for k in ("p_willing_hiv_test", "p_willing_prep", "intensity_hiv_test", "intensity_prep"):
        assert pc[k] == b["care_seeking"]["setting"]["primary_care"][k]
    for other in ("ED", "sud_treatment", "mental_health"):
        assert s["care_seeking"]["setting"][other] == b["care_seeking"]["setting"][other]
    assert "timeline_scaling" not in s
    assert s["hiv_testing"]["acceptance"] == 0.55 and s["prep"]["acceptance"] == 0.60
    s_static, _ = gen.build_arms(b, static=True)["pc4_chia_central_static.yml"]
    assert s_static["care_seeking"]["setting"]["primary_care"]["reach_start_time"] == -9999
    hi, _ = gen.build_arms(b, variant="high")["pc4_chia_high.yml"]
    assert hi["care_seeking"]["setting"]["primary_care"]["reach_intensity_mult_prep"] == pytest.approx(3.15 / 0.65, abs=1e-3)


@pytest.mark.unit
def test_ehr_arm_sets_the_bundle_and_injectable_first_is_a_prep_arm():
    b = baseline()
    arms = gen.build_arms(b)
    s, notes = arms["pc2_ehr_opt_out_central.yml"]
    pc = s["care_seeking"]["setting"]["primary_care"]
    assert pc["prep_bundle_prob"] == 0.8 and pc["prep_bundle_prob_unwilling"] == 0.15
    assert pc["prep_bundle_start_step"] == gen.DEFAULT_START_STEP
    assert "pc_prep_intensity" not in (s.get("timeline_scaling") or {}).get("timeline", {})   # the CDS lever is the bundle now
    static, _ = gen.build_arms(b, static=True)["pc2_ehr_opt_out_central_static.yml"]
    assert static["care_seeking"]["setting"]["primary_care"]["prep_bundle_start_step"] == -999
    inj, notes = arms["pc6_injectable_first_pwid_central.yml"]
    assert inj["prep"]["injectable_first_pwid_share"] == 0.4
    assert inj["prep"]["injectable_first_start_step"] == gen.DEFAULT_START_STEP
    assert inj["care_seeking"] == b["care_seeking"]
    for v, share in (("low", 0.2), ("high", 0.8)):
        assert gen.build_arms(b, variant=v)[f"pc6_injectable_first_pwid_{v}.yml"][0]["prep"]["injectable_first_pwid_share"] == share
    chia, _ = arms["pc4_chia_central.yml"]
    cpc = chia["care_seeking"]["setting"]["primary_care"]
    assert cpc["reach_prep_bundle_prob"] == pytest.approx(0.5 + 0.65 * 0.3)
    assert cpc["reach_prep_adherence_mult"] == pytest.approx(1 + 0.65 * 0.26)
