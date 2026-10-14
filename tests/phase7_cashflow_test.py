"""
Phase 7 cash flow (scripts/phase7_cashflow.py): programme costs are statewide per
year; ``--sites`` scales only the fixed per-site CHIA platform cost (2026-10-13 fix:
it used to multiply the navigation cost and CHIA's variable staff cost as well).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "phase7_cashflow.py"
SPEC = importlib.util.spec_from_file_location("phase7_cashflow", SCRIPT)
cf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cf)


@pytest.fixture(scope="module")
def costs():
    return yaml.safe_load(open(REPO / "docs" / "phase7_unit_costs.yml"))


def test_chia_programme_cost_splits_fixed_per_site_from_variable_statewide(costs):
    c = yaml.safe_load(yaml.safe_dump(costs))
    c["program_costs"]["chia_platform_per_site_year"] = {"value": 10000.0}
    fixed, variable, missing = cf.program_cost_per_year(c, reach_visits=1200.0, reached_pm=5000.0, invitations=1200.0)
    pc = c["program_costs"]
    expected_variable = (1200.0 * cf._value(pc["chia_price_per_chat"])
                         + 1200.0 * cf._value(pc["ma_minutes_per_invitation"]) / 60.0 * cf._value(pc["ma_wage_per_hour"])
                         + 1200.0 * cf._value(pc["provider_minutes_per_reach_visit"]) / 60.0 * cf._value(pc["provider_wage_per_hour"]))
    assert missing == []
    assert fixed == 10000.0
    assert variable == pytest.approx(expected_variable)


def test_chia_is_priced_per_chat_with_no_set_up_fee(costs):
    # Jun 2026-10-13: no set-up, licence or integration fee; a flat price per completed chat
    pc = costs["program_costs"]
    assert cf._value(pc["chia_platform_per_site_year"]) == 0.0
    assert cf._value(pc["ehr_integration_one_time"]) == 0.0
    assert cf._value(pc["training_one_time"]) == 0.0
    assert cf._value(pc["chia_price_per_chat"]) == pytest.approx(10.0)
    assert pc["chia_price_per_chat"]["range"] == [4.50, 29.39]
    fixed, variable, missing = cf.program_cost_per_year(costs, reach_visits=100.0, reached_pm=400.0, invitations=100.0)
    assert missing == [] and fixed == 0.0
    # the override replaces the file's price and only the usage line moves
    fixed_hi, variable_hi, _ = cf.program_cost_per_year(costs, 100.0, 400.0, 100.0, price_per_chat=29.39)
    assert variable_hi - variable == pytest.approx(100.0 * (29.39 - 10.0))
    # a missing per-chat price is reported, not priced as zero
    c = yaml.safe_load(yaml.safe_dump(costs))
    c["program_costs"]["chia_price_per_chat"] = {"value": None}
    fixed, variable, missing = cf.program_cost_per_year(c, 1.0, 1.0, 1.0)
    assert (fixed, variable) == (None, None) and "chia_price_per_chat" in missing


def test_navigation_cost_effort_and_caseload(costs):
    pc = costs["program_costs"]
    effort, caseload, fte_e, fte_c, missing = cf.navigation_cost_per_year(costs, reach_visits=20000.0, reached_pm=60000.0)
    hours = 20000.0 * cf._value(pc["navigator_minutes_per_active_visit"]) / 60.0 + 60000.0 * cf._value(pc["navigator_minutes_per_enrolled_month"]) / 60.0
    per_fte = cf._value(pc["navigator_wage_per_hour"]) * 2080.0 * (1.0 + cf._value(pc["navigator_supervision_overhead"]))
    assert missing == []
    assert fte_e == pytest.approx(hours / cf._value(pc["navigator_productive_hours_per_fte"]))
    assert fte_c == pytest.approx(60000.0 / 12.0 / cf._value(pc["navigator_active_caseload_per_fte"]))
    assert effort == pytest.approx(fte_e * per_fte)
    assert caseload == pytest.approx(fte_c * per_fte)


def _contrasts(path):
    rows = []
    for arm, visits, pm, hazard in (("pc3_navigation_test", 9000.0, 27000.0, -0.9),
                                    ("pc4_chia_test", 3600.0, 18000.0, -1.8)):
        for outcome, v in (("reach_visits", visits), ("reached_person_months", pm), ("hiv_hazard_pwid", hazard)):
            rows.append({"arm": arm, "outcome": outcome, "diff_mean_statewide": v, "horizon_years": 9.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(contrasts, costs_path, out, sites, extra=()):
    subprocess.run([sys.executable, str(SCRIPT), "--contrasts", str(contrasts), "--costs", str(costs_path),
                    "--out", str(out), "--sites", str(sites), *extra], check=True, capture_output=True)
    return pd.read_csv(out).set_index("arm")


def test_chia_price_override_moves_only_the_usage_cost(tmp_path, costs):
    costs_path = tmp_path / "costs.yml"
    yaml.safe_dump(costs, open(costs_path, "w"))
    contrasts = tmp_path / "contrasts.csv"
    _contrasts(contrasts)
    central = _run(contrasts, costs_path, tmp_path / "c.csv", 1)
    high = _run(contrasts, costs_path, tmp_path / "h.csv", 1, ("--chia-price-per-chat", "29.39"))
    chia, nav = "pc4_chia_test", "pc3_navigation_test"
    chats = 3600.0 / 9.0
    assert central.loc[chia, "chia_chats_per_year"] == pytest.approx(chats)
    assert central.loc[chia, "chia_usage_cost_per_year"] == pytest.approx(chats * 10.0)
    assert high.loc[chia, "chia_usage_cost_per_year"] == pytest.approx(chats * 29.39)
    assert high.loc[chia, "program_cost_per_year"] - central.loc[chia, "program_cost_per_year"] == pytest.approx(chats * 19.39)
    assert pd.isna(central.loc[nav, "chia_chats_per_year"])
    assert high.loc[nav, "program_cost_per_year"] == pytest.approx(central.loc[nav, "program_cost_per_year"])


def test_sites_scales_only_the_fixed_chia_platform_cost(tmp_path, costs):
    c = yaml.safe_load(yaml.safe_dump(costs))
    c["program_costs"]["chia_platform_per_site_year"] = {"value": 10000.0}
    costs_path = tmp_path / "costs.yml"
    yaml.safe_dump(c, open(costs_path, "w"))
    contrasts = tmp_path / "contrasts.csv"
    _contrasts(contrasts)
    s1 = _run(contrasts, costs_path, tmp_path / "s1.csv", 1)
    s3 = _run(contrasts, costs_path, tmp_path / "s3.csv", 3)
    nav, chia = "pc3_navigation_test", "pc4_chia_test"
    # navigation: statewide effort cost, untouched by the number of sites
    for col in ("program_cost_per_year", "clinic_net_per_year", "cost_per_infection_averted_clinic_and_payer"):
        assert s1.loc[nav, col] == pytest.approx(s3.loc[nav, col])
    # CHIA: two more sites add exactly two platform licences, nothing else
    assert s3.loc[chia, "program_cost_per_year"] - s1.loc[chia, "program_cost_per_year"] == pytest.approx(20000.0)
    assert s1.loc[chia, "clinic_net_per_year"] - s3.loc[chia, "clinic_net_per_year"] == pytest.approx(20000.0)
    averted = 1.8 / 9.0
    assert (s3.loc[chia, "cost_per_infection_averted_clinic_and_payer"]
            - s1.loc[chia, "cost_per_infection_averted_clinic_and_payer"]) == pytest.approx(20000.0 / averted)
    # cost per infection averted is gross: programme cost plus payer activity cost
    assert s1.loc[chia, "cost_per_infection_averted_clinic_and_payer"] == pytest.approx(
        (s1.loc[chia, "program_cost_per_year"] + s1.loc[chia, "payer_activity_cost_per_year"]) / averted)
