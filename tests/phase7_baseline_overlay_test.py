"""
make_phase7_baseline.overlay_care_layer (2026-10-02): a calibrated pick that
predates today's care-layer scalars and PrEP persistence decisions gets them
from the scenario source; fitted values stay.
"""
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("make_phase7_baseline", REPO / "scripts" / "make_phase7_baseline.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


@pytest.mark.unit
def test_overlay_carries_prep_persistence_and_monitoring_into_an_old_pick():
    src = yaml.safe_load(open(REPO / "titan" / "calibration" / "scenario_source.yml"))
    pick = {
        "prep": {"acceptance": 0.6, "cap": 0.015},
        "care_seeking": {"setting": {"primary_care": {"visit_rate_annual": 5.0, "p_willing_prep": 0.079}}},
        "demographics": {"white_nonhispanic": {"sex_type": {"HM": {
            "prep": {"adherence": 0.1, "discontinue": 0.1},
            "drug_type": {"Inj": {"prep": {}}, "NonInj": {"prep": {}}}}}}},
    }
    out = mod.overlay_care_layer(pick)
    assert out["prep"]["monitoring_interval_steps"] == src["scenario"]["prep"]["monitoring_interval_steps"]
    assert out["prep"]["discontinue_late_mult"] == src["scenario"]["prep"]["discontinue_late_mult"]
    assert out["prep"]["acceptance"] == 0.6                              # fitted / untouched
    hm = out["demographics"]["white_nonhispanic"]["sex_type"]["HM"]
    assert hm["prep"]["adherence"] == 0.5
    assert hm["prep"]["discontinue"] == 0.09
    assert hm["drug_type"]["Inj"]["prep"]["discontinue"] == 0.24
    assert hm["drug_type"]["NonInj"]["prep"]["discontinue"] == 0.09
    pc = out["care_seeking"]["setting"]["primary_care"]
    assert pc["p_willing_prep"] == 0.079                                 # the pick's own value kept
    # fields the source writes and the pick lacks are filled; schema defaults
    # (reach_*, acceptance_mult_*) are not in the source and are left to paraml
    assert pc["visit_prob_after_overdose"] == 0.0 and "intensity_hiv_test" in pc
    assert "reach_prob" not in pc
