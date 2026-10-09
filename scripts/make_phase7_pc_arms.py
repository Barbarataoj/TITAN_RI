#!/usr/bin/env python
"""
Generate the Phase 7 primary-care implementation-strategy arms.

WHY THIS EXISTS
---------------
The provider review packet ("HIV Prevention in Primary Care, Implementation
Strategies", 2026-10) specifies three strategies with Proctor's seven questions
(actor, action, target, temporality, dose, outcome, justification). In the
model a primary-care strategy is a change to the care-seeking layer's
primary-care setting only -- `care_seeking.setting.primary_care.*` -- plus,
for patient-facing components, the patient acceptance scalars. Every other
setting (ED, SUD treatment, mental health / OTP) stays at its calibrated
value. This script turns a strategy specification into scenario YAMLs and
prints the arithmetic, so the methods section and the arm files cannot drift
apart (`--check`).

THE THREE LEVERS
----------------
  W  `p_willing_<service>`   share of providers who deliver the service at all.
                             A strategy with reach f and efficacy e moves it to
                                 W' = W0 + f * e * (1 - W0)
                             (reach = share of providers exposed; efficacy =
                             share of exposed non-deliverers who start).
  I  `intensity_<service>`   per-visit offer probability among willing
                             providers; moved by a multiplier, capped at 1.
  A  `hiv_testing.acceptance`, `prep.acceptance`, `moud.acceptance`
                             patient acceptance given an offer; set to a level.

TEMPORALITY
-----------
Two ways to run an arm:
  --static       the changed values are in the scenario from the start,
                 burn-in included (comparative statics, the design the 2026-10-01
                 ensemble used). No engine support needed.
  dated (default, --start-step 133 = 2023-01)
                 the scenario carries timeline_scaling entries that multiply the
                 parameters at the start step, and
                 `care_seeking.setting.primary_care.pool_refresh_time` set to the
                 same step so the provider pool is re-drawn against the new
                 values (provider records freeze willing/intensity at draw time;
                 see CareSeeking.refresh_pools). Acceptance scalars are read per
                 step, so timeline scaling alone moves them.

THE STRATEGIES
--------------
Three defaults stand in until the packet's three are lifted into the same
slots (docs/calibration_log/phase_7_pc_strategy_extraction_2026-10-02.md).
Magnitudes and ranges come from the literature pass in
docs/calibration_log/phase_7_pc_strategy_effects_2026-10-02.md; PMIDs are on
each number below. `--variant low|high` writes the sensitivity ends.

USAGE
-----
    python scripts/make_phase7_pc_arms.py --baseline <scenario.yml> --out-dir scenarios/pc
    python scripts/make_phase7_pc_arms.py ... --static
    python scripts/make_phase7_pc_arms.py ... --variant high
    python scripts/make_phase7_pc_arms.py ... --check
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from titan.calibration.optuna_calibration import load_yaml  # noqa: E402

SETTING = "primary_care"
SERVICES = ("hiv_test", "prep", "moud")
ACCEPTANCE_PATH = {"hiv_test": ("hiv_testing", "acceptance"),
                   "prep": ("prep", "acceptance"),
                   "moud": ("moud", "acceptance")}
# A patient-side component is expressed as the setting's acceptance multiplier
# (care_seeking.setting.primary_care.acceptance_mult_<service>), so the global
# acceptance scalar -- and every other setting's offers -- stay as calibrated.
DEFAULT_START_STEP = 133      # 2023-01: calibration = steps 1-132 = 2012-2022
TIMELINE_STOP = 10_000        # beyond any horizon (stop_time is exclusive)

# Each strategy: per service {reach, willing_efficacy, intensity_mult}, acceptance
# levels, and the low/high sensitivity ends. Sources in the effects note.
# Programme-type strategy (a patient-facing tool offered in the setting, e.g. the
# CHIA pre-visit chatbot): expressed through the setting's reach_* fields rather
# than W/I/acceptance. Values: docs/calibration_log/phase_7_chia_workflow_cashflow_2026-10-02.md
# section 7.5 (reference AHF messenger PMID 41792418 with an OUD discount d on the
# excess effect: central d = 0.65, low d = 0.5, high d = 1).
def _chia(d, reach_sms, ask=0.35, eff_req=0.35, oud_req=0.78):
    rr = 1.0 + d * (3.15 - 1.0)              # PrEP initiation among reached, AHF RR 3.15
    return {
        "reach_prob": reach_sms,
        "reach_retry_prob": 0.0,
        "reach_visit_prob": 0.65,
        "reach_offer_prob_hiv_test": round(0.027 + d * (0.50 - 0.027), 4),
        # a prepared patient raises PrEP (share who ask 0.35) and a non-prescribing
        # provider acts on the request (efficacy 0.35 x OUD factor 0.78); the
        # AHF initiation effect above covers willing providers, this covers the rest
        "reach_offer_prob_prep": round(ask * eff_req * oud_req, 4),
        "reach_acceptance_mult_hiv_test": round((0.55 + d * 0.30) / 0.55, 4),
        "reach_intensity_mult_prep": round(rr / 0.65, 4),      # per-visit multiplier so the per-patient-year effect is rr
        "reach_acceptance_mult_prep": 1.0,
        "reach_prep_discontinue_mult": round(1.0 - d * (1.0 - 0.50), 4),   # AHF refill 57.0 vs 32.4 % -> h 0.50
        # adherence messaging (2026-10-09): a chatbot can do what PrEPmate's
        # bidirectional texting did, protective drug levels 72 vs 57 % (x1.26,
        # PMID 30239620; multi-component feedback RR 1.83, PMID 38814406),
        # OUD-discounted by d; was x1.10
        "reach_prep_adherence_mult": round(1.0 + d * 0.26, 4),
        "reach_prep_restart_prob": 0.015,
        "reach_prep_monitoring_mult": round(1.0 + d * 0.22, 4),           # AHF attendance 66.1 vs 53.9 %
        # the prepared patient at the visit where the test comes back negative
        # (2026-10-09 bundle): PrEP raised with at least this probability,
        # whatever the provider's habit; status quo 0.5 at prescribers, 0 at others
        "reach_prep_bundle_prob": round(0.5 + d * 0.3, 4),
    }


# Field re-engagement of missed injectable PrEP doses (outreach worker). Not a
# primary-care lever: it multiplies an overdue agent's per-step return
# probability (titan/features/prep.py progress_injectable) from the start step.
# Suggested by the clinical reviewer, 2026-10-07. STAND-IN effect size: no
# published return-after-miss effect for PrEP outreach was found (search
# 2026-10-08). Status quo per the reviewer: 10-25 % come back after a miss
# (cabotegravir 0.175 per step over a 1-step window; lenacapavir 0.09 per step
# over 2 steps). Central x2.9 -> ~50 % come back (cab 0.51; len 1-(1-0.26)^2 =
# 45 %); low x2.0 -> ~35 %; high x5.1 -> ~90 % for cab, the ceiling seen where
# outreach is intensive (San Francisco drop-in clinic giving long-acting HIV
# treatment to people experiencing homelessness, 19 % of injections given
# outside the clinic: 3 % of 637 injections more than 14 days late; JAMA Netw
# Open 2026, doi 10.1001/jamanetworkopen.2026.20348).
REENGAGEMENT_MULT = {"central": 2.9, "low": 2.0, "high": 5.1}


STRATEGIES = {
    "pc4_chia": {
        "label": "CHIA pre-visit chatbot + provider summary (patient-facing programme in primary care)",
        "program": True,
        "central": {"program": _chia(0.65, 0.15)},
        "low": {"program": dict(_chia(0.50, 0.08, ask=0.30, eff_req=0.20, oud_req=0.70), reach_visit_prob=0.50, reach_prep_restart_prob=0.01)},
        "high": {"program": dict(_chia(1.0, 0.23, ask=0.50, eff_req=0.50, oud_req=0.85), reach_visit_prob=0.80, reach_prep_restart_prob=0.03)},
        "sources": [
            "reference: AHF conversational messenger, 155,217 adults (PMID 41792418): engaged 23.4%; PrEP initiation 19.3 vs 6.1% (RR 3.15); refill after initiation 57.0 vs 32.4%; attendance 66.1 vs 53.9%",
            "OUD discount d on the excess effect (0.65; 0.5-1): phone/portal ceiling 36575145, 41617402; PrEP acceptance in an OUD bridge clinic 36383918; PWID discontinuation aHR 4.8 31436610; null digital support in stimulant users 36659873",
            "testing links: patient-screen CDS acknowledged 82% / ordered 47% (39572251); computer counseling + rapid test 97% tested (23837807); engagement priors 42081747, 41219954",
            "PrEP offer floor: 47% of chatbot users initiated the prevention conversation (42712342) -> ask 0.35 (0.30-0.50); patient request converts non-prescribers e 0.35 (0.20-0.50): Kravitz 15855433, patient-initiated CME 31% (33735209), 94% of VA PrEP conversations patient-initiated (31705471); OUD factor 0.78 (27896552, 36964426)",
        ],
    },
    "pc1_training_detailing": {
        "label": "Provider training + academic detailing, ongoing programme (W lever)",
        "central": {
            "hiv_test": {"reach": 0.5, "willing_efficacy": 0.3, "intensity_mult": 1.2},
            "prep": {"reach": 0.5, "willing_efficacy": 0.3, "intensity_mult": 1.0},
            "acceptance": {},
        },
        "low": {
            "hiv_test": {"reach": 0.25, "willing_efficacy": 0.1, "intensity_mult": 1.0},
            "prep": {"reach": 0.25, "willing_efficacy": 0.1, "intensity_mult": 1.0},
            "acceptance": {},
        },
        "high": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.5, "intensity_mult": 1.35},
            "prep": {"reach": 1.0, "willing_efficacy": 0.5, "intensity_mult": 1.0},
            "acceptance": {},
        },
        "sources": [
            "testing: training/detailing alone I x1.0-1.35 (PMIDs 24143815, 30067778, 41338820); W +0-0.2 (26418260)",
            "PrEP: detailing W +0.12 per campaign (34686296); ECHO + EHR lists W +0.35 (37376664); training alone ~0 (37401992)",
            "generic: academic detailing +5.6 pp (17943742), training +4-7 pp (34523128), practice facilitation OR 2.76 on adoption (22230833)",
        ],
    },
    "pc2_ehr_opt_out": {
        "label": "EHR reminder + opt-out default + MA/nurse standing orders (I lever, W to the reach)",
        # PrEP since 2026-10-09: the clinical-decision-support prompt acts at the
        # visit where the test comes back negative (the test-and-offer bundle),
        # not as an independent per-visit multiplier. Prescribing providers raise
        # PrEP more often (0.5 -> 0.8); the prompt gets some non-prescribers to
        # act (0 -> 0.15). CDS x1.3 overall / x2.6 among willing (38412047), x4.5
        # from a near-zero base (40376113), passive BPA null (35225082).
        "central": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.67, "intensity_mult": 1.5},
            "prep": {"reach": 1.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.65},
            "bundle": {"prep_bundle_prob": 0.8, "prep_bundle_prob_unwilling": 0.15},
        },
        "low": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.35, "intensity_mult": 1.2},
            "prep": {"reach": 1.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.60},
            "bundle": {"prep_bundle_prob": 0.6, "prep_bundle_prob_unwilling": 0.05},
        },
        "high": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.9, "intensity_mult": 2.0},
            "prep": {"reach": 1.0, "willing_efficacy": 0.2, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.85},
            "bundle": {"prep_bundle_prob": 0.9, "prep_bundle_prob_unwilling": 0.30},
        },
        "sources": [
            "testing: EHR reminder x4 on receipt sustained 2 yr (24363958); BPA + incentive x5.1 (39214701); passive alert x1.9 (27378268); 70-80% offered under facilitation (23605307)",
            "testing acceptance: opt-out 0.60-0.66 vs opt-in 0.38 (26786744, 31365316); nurse standing orders 0.85 (18421508)",
            "PrEP: CDS I x1.3 overall, x2.6 among willing providers (38412047); x4.5 from a near-zero base (40376113); passive BPA null (35225082)",
        ],
    },
    "pc3_navigation_same_day": {
        # Since 2026-10-09 a staffed patient-facing programme in primary care (the
        # same machinery as CHIA), not an acceptance multiplier alone: a peer
        # navigator enrols a share of patients with OUD at their visits (reach,
        # capacity-limited), accompanies same-day testing and PrEP starts, hands
        # deferred starts off warmly, supports adherence and follow-up, and
        # re-engages missed injectable doses (the 2026-10-08 outreach arm, folded
        # in here at the reviewer's request). Workload is read from the model:
        # reached person-months, tests, starts, injection visits, re-engagements.
        "label": "Peer navigation: same-day testing / PrEP start, warm handoff, adherence support, re-engagement (patient-facing programme)",
        "program": True,
        "central": {
            "program": {"reach_prob": 0.25, "reach_retry_prob": 0.0, "reach_visit_prob": 1.0,
                        "reach_prep_bundle_prob": 0.8, "reach_prep_same_day_prob": 1.0,
                        "reach_prep_deferred_return_prob": 0.8, "reach_prep_adherence_mult": 1.26,
                        "reach_prep_monitoring_mult": 1.21, "reach_prep_restart_prob": 0.015},
            "acceptance": {"hiv_test": 0.83, "prep": 0.75},
            "prep_params": {"reengagement_return_mult": 2.9},
        },
        "low": {
            "program": {"reach_prob": 0.15, "reach_retry_prob": 0.0, "reach_visit_prob": 1.0,
                        "reach_prep_bundle_prob": 0.6, "reach_prep_same_day_prob": 0.8,
                        "reach_prep_deferred_return_prob": 0.6, "reach_prep_adherence_mult": 1.0,
                        "reach_prep_monitoring_mult": 1.1, "reach_prep_restart_prob": 0.01},
            "acceptance": {"hiv_test": 0.66, "prep": 0.55},
            "prep_params": {"reengagement_return_mult": 2.0},
        },
        "high": {
            "program": {"reach_prob": 0.40, "reach_retry_prob": 0.0, "reach_visit_prob": 1.0,
                        "reach_prep_bundle_prob": 0.9, "reach_prep_same_day_prob": 1.0,
                        "reach_prep_deferred_return_prob": 0.9, "reach_prep_adherence_mult": 1.6,
                        "reach_prep_monitoring_mult": 1.3, "reach_prep_restart_prob": 0.03},
            "acceptance": {"hiv_test": 0.85, "prep": 0.77},
            "prep_params": {"reengagement_return_mult": 5.1},
        },
        "sources": [
            "testing: navigation RR 1.5-1.65 on screening uptake (32700218), cap 0.85 (18421508 nurse-initiated); rapid POC +0.05 (18421508)",
            "PrEP: same-day / on-site offer to PWID accepted 0.55-0.77 (42201314, 33148998, 31944854); referral-out 0.10-0.15 (39886694)",
            "warm handoff: 81 % attended a first PrEP visit vs 36 % by EHR message and 27 % with a card (34448603)",
            "adherence support: PrEPmate protective drug levels 72 vs 57 % (x1.26, 30239620); retention 86 vs 71 % (x1.21); multi-component feedback RR 1.83 (38814406)",
            "re-engagement of missed injectable doses: STAND-IN x2.9 (2.0-5.1), scaled by reach as a mean-field approximation; ceiling SF drop-in LA-ART 3 % >14 d late (doi 10.1001/jamanetworkopen.2026.20348)",
            "reach 0.25 (0.15-0.40): ASSUMPTION pending navigator caseload data (Jun / Phil, 2026-10-09)",
        ],
    },
    "pc6_injectable_first_pwid": {
        # The clinical reviewer's question (2026-10-07): injectable PrEP for people
        # who inject. Clinics default to a long-acting injectable for patients who
        # currently inject; everyone else follows the product trend. Acceptability
        # of 3-monthly injectable PrEP among people who inject 38 % (1,127-person
        # survey, three US cities 2019-20; aOR 1.7 vs non-injectors; PMID 36104676).
        "label": "Injectable-first PrEP for people who inject (share of their starts long-acting)",
        "prep": True,
        "prep_start_fields": ["injectable_first_start_step"],
        "central": {"prep_params": {"injectable_first_pwid_share": 0.4}},
        "low": {"prep_params": {"injectable_first_pwid_share": 0.2}},
        "high": {"prep_params": {"injectable_first_pwid_share": 0.8}},
        "sources": ["acceptability 38 % among people who inject (PMID 36104676); 0.8 = the design ceiling"],
    },
}


def w_after(w0, reach, efficacy):
    return w0 + reach * efficacy * (1.0 - w0)


def _setting(scenario):
    return scenario["care_seeking"]["setting"][SETTING]


def build_arms(baseline, start_step=DEFAULT_START_STEP, static=False, variant="central",
               strategies=None):
    """Return {filename: (scenario dict, arithmetic lines)}. Pure."""
    out = {}
    for key, spec in (strategies or STRATEGIES).items():
        levers = spec[variant]
        scenario = copy.deepcopy(baseline)
        base = _setting(baseline)
        if spec.get("prep"):
            # a prep-level arm (injectable-first): dated by its own start-step fields
            prep = scenario.setdefault("prep", {})
            notes = [f"{spec['label']} [{variant}]",
                     ("static: on from the start" if static else f"dated: from step {start_step}")]
            for k, v in levers["prep_params"].items():
                prep[k] = v
                notes.append(f"  prep.{k} = {v}")
            for k in spec.get("prep_start_fields", []):
                prep[k] = -999 if static else int(start_step)
            notes.extend(f"  source: {x}" for x in spec["sources"])
            out[f"{key}_{variant}{'_static' if static else ''}.yml"] = (scenario, notes)
            continue
        if spec.get("program"):
            # programme fields are read per visit from params, so a dated arm
            # only needs reach_start_time; no timeline entry, no pool refresh
            fields = dict(levers["program"])
            fields["reach_start_time"] = -9999 if static else int(start_step)
            # patient-side acceptance at the programme's visits, as a multiplier
            # on the baseline acceptance (hiv_testing.acceptance, prep.acceptance)
            for service, a1 in levers.get("acceptance", {}).items():
                top, field = ACCEPTANCE_PATH[service]
                a0 = float(baseline[top][field])
                if a0 > 0:
                    fields[f"reach_acceptance_mult_{service}"] = round(a1 / a0, 6)
            notes = [f"{spec['label']} [{variant}]",
                     ("static: programme on from the start, burn-in included" if static
                      else f"dated: programme starts at step {start_step} (reach_start_time)")]
            for k, v in fields.items():
                _setting(scenario)[k] = v
                notes.append(f"  {SETTING}.{k} = {v}")
            # prep-level companions of a programme (re-engagement of missed
            # injectable doses by the navigator), scaled by reach as a mean-field
            # approximation: the engine's lever is population-wide
            prep_params = dict(levers.get("prep_params", {}))
            if prep_params:
                prep = scenario.setdefault("prep", {})
                reach = float(fields.get("reach_prob", 1.0))
                if "reengagement_return_mult" in prep_params:
                    full = float(prep_params["reengagement_return_mult"])
                    prep["reengagement_return_mult"] = round(1.0 + (full - 1.0) * reach, 4)
                    prep["reengagement_start_step"] = 0 if static else int(start_step)
                    notes.append(f"  prep.reengagement_return_mult = 1 + ({full} - 1) x reach {reach} = {prep['reengagement_return_mult']} (mean-field)")
                for k, v in prep_params.items():
                    if k != "reengagement_return_mult":
                        prep[k] = v
                        notes.append(f"  prep.{k} = {v}")
            notes.append("  other settings (ED, sud_treatment, mental_health) unchanged; W/I of primary care unchanged")
            notes.extend(f"  source: {s}" for s in spec["sources"])
            out[f"{key}_{variant}{'_static' if static else ''}.yml"] = (scenario, notes)
            continue
        notes = [f"{spec['label']} [{variant}]",
                 ("static: values in force from the start, burn-in included" if static
                  else f"dated: timeline_scaling at step {start_step} + provider pool refresh at that step")]
        timeline = {}
        changed = False
        for service in SERVICES:
            lv = levers.get(service)
            if lv is None:
                continue
            w0 = float(base[f"p_willing_{service}"])
            i0 = float(base[f"intensity_{service}"])
            w1 = w_after(w0, lv["reach"], lv["willing_efficacy"])
            i1 = min(1.0, i0 * lv["intensity_mult"])
            if abs(w1 - w0) > 1e-12:
                notes.append(f"  {service}: p_willing {w0:.4f} -> {w1:.4f} = {w0:.4f} + {lv['reach']} x {lv['willing_efficacy']} x (1 - {w0:.4f})")
                changed = True
                if static or w0 == 0.0:
                    _setting(scenario)[f"p_willing_{service}"] = round(w1, 6)
                    if not static:
                        notes.append(f"    (p_willing was 0: set directly, not scaled)")
                else:
                    timeline[f"pc_{service}_willing"] = {
                        "parameter": f"care_seeking|setting|{SETTING}|p_willing_{service}",
                        "start_time": start_step, "stop_time": TIMELINE_STOP, "scalar": round(w1 / w0, 6)}
            if abs(i1 - i0) > 1e-12:
                notes.append(f"  {service}: intensity {i0:.4f} -> {i1:.4f} (x{lv['intensity_mult']}"
                             + (", capped at 1" if i0 * lv['intensity_mult'] > 1.0 else "") + ")")
                changed = True
                if static or i0 == 0.0:
                    _setting(scenario)[f"intensity_{service}"] = round(i1, 6)
                else:
                    timeline[f"pc_{service}_intensity"] = {
                        "parameter": f"care_seeking|setting|{SETTING}|intensity_{service}",
                        "start_time": start_step, "stop_time": TIMELINE_STOP, "scalar": round(i1 / i0, 6)}
        for service, a1 in levers.get("acceptance", {}).items():
            top, field = ACCEPTANCE_PATH[service]
            a0 = float(baseline[top][field])
            if abs(a1 - a0) < 1e-12 or a0 <= 0.0:
                continue
            m0 = float(base.get(f"acceptance_mult_{service}", 1.0))
            m1 = a1 / a0
            notes.append(f"  {service}: acceptance {a0:.3f} -> {a1:.3f} for offers made in {SETTING} only"
                         f" (acceptance_mult_{service} {m0:.3f} -> {m1:.3f}; {top}.{field} itself unchanged)")
            changed = True
            if static:
                _setting(scenario)[f"acceptance_mult_{service}"] = round(m1, 6)
            else:
                timeline[f"pc_{service}_acceptance"] = {
                    "parameter": f"care_seeking|setting|{SETTING}|acceptance_mult_{service}",
                    "start_time": start_step, "stop_time": TIMELINE_STOP, "scalar": round(m1 / m0, 6)}
        # the test-and-offer bundle at the visit where the test comes back
        # negative (2026-10-09): setting fields read per step, dated by their
        # own start step
        for k, v in levers.get("bundle", {}).items():
            _setting(scenario)[k] = v
            notes.append(f"  {SETTING}.{k} = {v} (was {base.get(k, 'unset')})")
            changed = True
        if levers.get("bundle"):
            _setting(scenario)["prep_bundle_start_step"] = -999 if static else int(start_step)
        if not changed:
            notes.append("  WARNING: no lever moves at this variant -- the arm is the baseline")
        if timeline:
            scenario.setdefault("features", {})["timeline_scaling"] = True
            scenario.setdefault("timeline_scaling", {}).setdefault("timeline", {}).update(timeline)
            _setting(scenario)["pool_refresh_time"] = int(start_step)
            notes.append(f"  timeline entries: {', '.join(sorted(timeline))}; pool_refresh_time {start_step}")
        notes.append("  other settings (ED, sud_treatment, mental_health) unchanged")
        notes.extend(f"  source: {s}" for s in spec["sources"])
        out[f"{key}_{variant}{'_static' if static else ''}.yml"] = (scenario, notes)
    return out


def write_scenario(path, scenario, notes, baseline_path):
    header = ["# GENERATED by scripts/make_phase7_pc_arms.py -- do not edit by hand.",
              f"# Baseline: {baseline_path}. Regenerate after every refit; --check guards drift.",
              "# Worked arithmetic:"] + [f"#   {line}" for line in notes] + [""]
    with open(path, "w") as fh:
        fh.write("\n".join(header))
        yaml.safe_dump(scenario, fh, default_flow_style=False, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="the calibrated baseline scenario (full YAML)")
    ap.add_argument("--out-dir", default=str(REPO / "scenarios" / "pc"))
    ap.add_argument("--start-step", type=int, default=DEFAULT_START_STEP)
    ap.add_argument("--static", action="store_true", help="comparative statics instead of a dated start")
    ap.add_argument("--variant", choices=("central", "low", "high"), default="central")
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on drift")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    baseline = load_yaml(baseline_path)
    out_dir = Path(args.out_dir)
    arms = build_arms(baseline, start_step=args.start_step, static=args.static, variant=args.variant)
    stale = []
    for name, (scenario, notes) in arms.items():
        path = out_dir / name
        print(name)
        for line in notes:
            print(f"    {line}")
        if args.check:
            if not path.exists() or load_yaml(path) != scenario:
                stale.append(name)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        write_scenario(path, scenario, notes, baseline_path)
        print(f"    wrote {path}")
    if args.check:
        if stale:
            print(f"\nSTALE: {', '.join(stale)} -- regenerate with scripts/make_phase7_pc_arms.py")
            return 1
        print("\nOK: every primary-care arm matches the baseline plus its spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
