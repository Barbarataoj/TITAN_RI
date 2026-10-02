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
STRATEGIES = {
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
        "central": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.67, "intensity_mult": 1.5},
            "prep": {"reach": 1.0, "willing_efficacy": 0.0, "intensity_mult": 2.6},
            "acceptance": {"hiv_test": 0.65},
        },
        "low": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.35, "intensity_mult": 1.2},
            "prep": {"reach": 1.0, "willing_efficacy": 0.0, "intensity_mult": 1.3},
            "acceptance": {"hiv_test": 0.60},
        },
        "high": {
            "hiv_test": {"reach": 1.0, "willing_efficacy": 0.9, "intensity_mult": 2.0},
            "prep": {"reach": 1.0, "willing_efficacy": 0.2, "intensity_mult": 4.5},
            "acceptance": {"hiv_test": 0.85},
        },
        "sources": [
            "testing: EHR reminder x4 on receipt sustained 2 yr (24363958); BPA + incentive x5.1 (39214701); passive alert x1.9 (27378268); 70-80% offered under facilitation (23605307)",
            "testing acceptance: opt-out 0.60-0.66 vs opt-in 0.38 (26786744, 31365316); nurse standing orders 0.85 (18421508)",
            "PrEP: CDS I x1.3 overall, x2.6 among willing providers (38412047); x4.5 from a near-zero base (40376113); passive BPA null (35225082)",
        ],
    },
    "pc3_navigation_same_day": {
        "label": "Patient navigation + same-day rapid testing / PrEP start (A lever)",
        "central": {
            "hiv_test": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "prep": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.83, "prep": 0.75},
        },
        "low": {
            "hiv_test": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "prep": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.66, "prep": 0.55},
        },
        "high": {
            "hiv_test": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "prep": {"reach": 0.0, "willing_efficacy": 0.0, "intensity_mult": 1.0},
            "acceptance": {"hiv_test": 0.85, "prep": 0.77},
        },
        "sources": [
            "testing: navigation RR 1.5-1.65 on screening uptake (32700218), cap 0.85 (18421508 nurse-initiated); rapid POC +0.05 (18421508)",
            "PrEP: same-day / on-site offer to PWID accepted 0.55-0.77 (42201314, 33148998, 31944854); referral-out 0.10-0.15 (39886694)",
        ],
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
