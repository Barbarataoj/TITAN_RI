"""The cohort footing has one owner (calibration_targets.yml `calibration.cohort_footing`) and the scorer's
denominators follow it: scale = 13,704 / population size, MOUD coverage over all agents, overdose rates per
agent-year, fatal fraction with the injecting share taken per cohort member (work record §19.31, 2026-10-23)."""
import copy
import os

import pytest

from titan.calibration import optuna_calibration as oc


@pytest.fixture
def calib():
    return oc.load_yaml(oc.CALIB_YAML)["calibration"]


def test_default_footing_is_all(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    assert calib["cohort_footing"] == "all"
    assert oc.cohort_footing() == "all"


def test_scale_is_cohort_over_population(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    for n in (500, 2000, 13704):
        c = copy.deepcopy(calib); c["population"]["size"] = n
        assert oc.build_targets(c)["pop_scale"] == pytest.approx(13704 / n, rel=1e-9)
    monkeypatch.setenv("TITAN_COHORT_FOOTING", "active")
    c = copy.deepcopy(calib); c["population"]["size"] = 500
    assert oc.build_targets(c)["pop_scale"] == pytest.approx(13704 / (0.55 * 500), rel=1e-6)


def test_fatal_fraction_uses_cohort_injecting_share(calib, monkeypatch):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    od, inj, act = calib["overdose"], calib["injecting"]["injecting_pct"], calib["oud_baseline"]["oud_active"]
    share = inj * act
    expect = od["fatal_rate_annual"] / (od["fatal_rate_annual"] + share * od["nonfatal_rate_inj_annual"] + (1 - share) * od["nonfatal_rate_noninj_annual"])
    assert oc.derived_fatal_frac({"calibration": calib}) == pytest.approx(expect, rel=1e-9)


def test_invalid_footing_rejected(monkeypatch):
    monkeypatch.setenv("TITAN_COHORT_FOOTING", "half")
    with pytest.raises(ValueError):
        oc.cohort_footing()


# ---------------------------------------------------------------------------
# Insured footing (2026-10-24, work record §19.34). A run with the insurance feature on is stratified by
# insured_state; the claims-footed quantities are measured on the insured rows and the statewide scale is
# 13,704 / (insured share x N).
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

COLUMNS = [
    "t", "run_id", "drug_type", "sex_type", "race", "insured_state", "agents", "hiv", "hiv_dx", "haart",
    "hiv_new", "hiv_dx_new", "hiv_dx_new_inj", "incar", "oud_active", "oud_remission", "oud_none",
    "moud", "moud_active", "overdose_events", "overdose_deaths", "prep", "deaths",
]
# per step and drug type: agents, of them uninsured; remission among the insured; MOUD (all / insured);
# overdose events (all / insured); deaths (all / insured)
STRATA = {
    "Inj":    dict(agents=100, unins=20, rem_i=5, moud=30, moud_i=24, ev=10, ev_i=8,  d=2, d_i=1),
    "NonInj": dict(agents=400, unins=40, rem_i=5, moud=50, moud_i=40, ev=20, ev_i=16, d=2, d_i=1),
}


def _stratified_report(path, n_steps, n_runs=1, everyone_insured=False):
    rows = []
    for run in range(n_runs):
        for t in range(1, n_steps + 1):
            for drug, s in STRATA.items():
                unins = 0 if everyone_insured else s["unins"]
                insured = s["agents"] - unins
                moud_i = s["moud"] if everyone_insured else s["moud_i"]
                ev_i = s["ev"] if everyone_insured else s["ev_i"]
                d_i = s["d"] if everyone_insured else s["d_i"]
                base = {"t": t, "run_id": run, "drug_type": drug, "sex_type": "HM", "race": "w"}
                rows.append({**base, "insured_state": "insured", "agents": insured, "hiv": 5,
                             "oud_active": insured - s["rem_i"], "oud_remission": s["rem_i"],
                             "moud": moud_i, "moud_active": moud_i, "overdose_events": ev_i, "overdose_deaths": d_i})
                rows.append({**base, "insured_state": "uninsured", "agents": unins, "oud_active": unins,
                             "moud": s["moud"] - moud_i, "moud_active": s["moud"] - moud_i,
                             "overdose_events": s["ev"] - ev_i, "overdose_deaths": s["d"] - d_i})
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, 0)) for c in COLUMNS) + "\n")


def _collapse(stratified, path):
    """The same report as the engine would write WITHOUT the insured_states class: strata summed."""
    df = pd.read_csv(stratified, sep="\t")
    keys = ["t", "run_id", "drug_type", "sex_type", "race"]
    df.drop(columns=["insured_state"]).groupby(keys, as_index=False).sum().to_csv(path, sep="\t", index=False)


@pytest.fixture
def n_steps():
    return 12 * max(2, int(oc.OD_SCORING_YEARS))


def test_insured_footing_is_detected_from_the_strata_column():
    assert oc.insured_footing(pd.DataFrame({"agents": [5], "insured_state": ["insured"]}))
    assert not oc.insured_footing(pd.DataFrame({"agents": [5]}))


def test_run_scale_divides_by_the_insured_share():
    assert oc.run_pop_scale(13704 / 500, {}) == pytest.approx(13704 / 500)
    assert oc.run_pop_scale(13704 / 500, {"insured_frac": 0.88}) == pytest.approx(13704 / (0.88 * 500))
    with pytest.raises(ValueError):
        oc.run_pop_scale(1.0, {"insured_frac": 0.0})
    targets = {"pop_scale": 2.0, "other": 1}
    assert oc.targets_for_run(targets, {}) is targets
    scaled = oc.targets_for_run(targets, {"insured_frac": 0.5})
    assert scaled["pop_scale"] == pytest.approx(4.0) and targets["pop_scale"] == 2.0 and scaled["other"] == 1


def test_claims_footed_quantities_are_measured_on_the_insured_rows(tmp_path, monkeypatch, n_steps):
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    report = tmp_path / "basicReport.txt"
    _stratified_report(report, n_steps, n_runs=2)
    st = oc.parse_basic_report(report)
    insured = sum(s["agents"] - s["unins"] for s in STRATA.values())          # 440
    agents = sum(s["agents"] for s in STRATA.values())                       # 500
    assert st["insured_footing"] is True
    assert st["insured_frac"] == pytest.approx(insured / agents)
    rem_i = sum(s["rem_i"] for s in STRATA.values())
    assert st["oud_active"] == pytest.approx((insured - rem_i) / insured)
    assert st["oud_remission"] == pytest.approx(rem_i / insured)
    assert st["oud_none"] == pytest.approx(0.0)
    # MOUD: insured on MOUD over insured with OUD (active + remission == every insured agent here)
    moud_i = sum(s["moud_i"] for s in STRATA.values())
    assert st["moud_uptake_series"] == pytest.approx(np.full(n_steps // 12, moud_i / insured))
    # overdose: insured events less insured deaths (apportioned by event share) per insured agent-year
    ev_i = {k: 12 * s["ev_i"] for k, s in STRATA.items()}
    d_i = 12 * sum(s["d_i"] for s in STRATA.values())
    py = {k: s["agents"] - s["unins"] for k, s in STRATA.items()}
    share = {k: ev_i[k] / sum(ev_i.values()) for k in STRATA}
    assert st["nonfatal_od_rate_inj"] == pytest.approx((ev_i["Inj"] - d_i * share["Inj"]) / py["Inj"])
    assert st["nonfatal_od_rate_noninj"] == pytest.approx((ev_i["NonInj"] - d_i * share["NonInj"]) / py["NonInj"])
    # fatal check: EVERY death (RIDOH's statewide numerator) over insured agent-years (the 13,704 footing)
    d_all = 12 * sum(s["d"] for s in STRATA.values())
    assert st["fatal_od_rate"] == pytest.approx(d_all / insured)
    # per-run arithmetic agrees with the pooled one (two identical runs)
    assert st["nonfatal_od_rate_inj_by_run"] == pytest.approx([st["nonfatal_od_rate_inj"]] * 2)
    assert st["fatal_od_rate_by_run"] == pytest.approx([st["fatal_od_rate"]] * 2)
    # and the scorer scales this run by its insured share, in the objective and in the all-windows safeguard
    calib = oc.load_yaml(oc.CALIB_YAML)["calibration"]
    targets = oc.build_targets(calib)
    assert oc.targets_for_run(targets, st)["pop_scale"] == pytest.approx(targets["pop_scale"] * agents / insured)


def test_feature_on_with_everyone_insured_scores_like_feature_off(tmp_path, monkeypatch, n_steps):
    """The identity the lever runs check at scale: empty uninsured rows must reproduce the whole-cohort footing."""
    monkeypatch.delenv("TITAN_COHORT_FOOTING", raising=False)
    on, off = tmp_path / "on.txt", tmp_path / "off.txt"
    _stratified_report(on, n_steps, n_runs=2, everyone_insured=True)
    _collapse(on, off)
    st_on, st_off = oc.parse_basic_report(on), oc.parse_basic_report(off)
    assert st_on["insured_footing"] and not st_off["insured_footing"]
    assert st_on["insured_frac"] == pytest.approx(1.0) and "insured_frac" not in st_off
    for key in ("oud_active", "oud_remission", "oud_none", "nonfatal_od_rate_inj", "nonfatal_od_rate_noninj", "fatal_od_rate"):
        assert st_on[key] == pytest.approx(st_off[key]), key
    assert st_on["moud_uptake_series"] == pytest.approx(st_off["moud_uptake_series"])
    calib = oc.load_yaml(oc.CALIB_YAML)["calibration"]
    targets = oc.build_targets(calib)
    assert oc.multi_error(st_on, targets) == pytest.approx(oc.multi_error(st_off, targets))
    assert oc.hiv_err_all_windows(st_on, targets) == pytest.approx(oc.hiv_err_all_windows(st_off, targets))


def test_stratified_report_is_refused_on_the_archived_footing(tmp_path, monkeypatch, n_steps):
    monkeypatch.setenv("TITAN_COHORT_FOOTING", "active")
    report = tmp_path / "basicReport.txt"
    _stratified_report(report, n_steps)
    with pytest.raises(RuntimeError):
        oc.parse_basic_report(report)
