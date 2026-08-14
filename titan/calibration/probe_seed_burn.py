"""
Diagnostic: can a data-anchored HIV initial condition replace the 720-step burn-in,
and does it resolve the diagnosis-cascade saturation?

BACKGROUND

Upstream TITAN's design contract treats the initial state of the epidemic as a
data input: `hiv.init` is documented as "Probability that an agent in this class
initially has HIV", `hiv.dx.init` as the probability that such an agent "is
initially diagnosed and 'knows' their HIV status", and the shipped `burn_steps`
default is 6. TITAN_RI moved both of those slots into the search, then on
2026-08-07 zeroed the NonInj seed entirely and lengthened the burn to 720 so the
reservoir would rebuild endogenously (see optuna_calibration.objective and
docs/calibration_log/phase_6_burnin_equilibrium_2026-08-07.md).

That decision is sound on its own terms but has three measured costs:

  * 720 burn + 132 scored steps means 84.5% of all compute is burn-in;
  * the burn requirement is level-dependent -- job 4874631's cold trials
    equilibrate by month 96 while its hot trials are still climbing at t=0, so
    `burn_steps` is an unfitted parameter with real leverage on the HIV objective;
  * running a free annual testing probability over 852 steps with no offsetting
    outflow drives the diagnosed fraction to 0.992, against RIDOH's measured
    93.2%. The undiagnosed reservoir that scenario 5 acts on is ~1/8 of its real
    size.

Note the coupling recorded in optuna_calibration: the zero seed "requires
burn_steps >= 720 ... at the pinned 60 this zeroes a reservoir that has no time
to rebuild". Seed and burn length must therefore move together, which is what
this probe varies.

DATA ANCHORS (RIDOH 2023 HIV/STI/Viral Hepatitis/TB Surveillance Report)

  * HIV care continuum, 2022: 93.2% of people living with HIV in Rhode Island
    know their status (2,796 of an estimated ~3,000); 76.1% engaged in care;
    71% virally suppressed. This closes citations GAP #29.
  * ~3,000 people living with HIV against ~885,000 Rhode Island adults gives a
    general adult prevalence of ~0.0034, which is the correct value for the
    background route's non-OUD partners. The shipped 0.01 is ~3x that. Closes
    citations GAP #27.
  * Mode of exposure, 2019-2023 new diagnoses (n=289): sexual contact 84%,
    GBMSM/IDU 9%, IDU alone 6%, perinatal 1%. Within sexual contact, 78% GBMSM.

WHAT THIS PROBE DOES *NOT* CLAIM

Converting RIDOH's exposure shares for NEW DIAGNOSES into a PREVALENT-case
breakdown is exactly the step optuna_calibration already flagged as unsupported:
"RIDOH publishes no PLWH breakdown by transmission category", and converting
"would need the MSM population share (0.05 here) and would give a far larger
number". That prediction is correct -- applying the 2019-2023 diagnosis mix to
the prevalent pool implies an MSM:heterosexual seed ratio near 34, against the
assumed NONINJ_HIV_MSM_TO_HET_RATIO = 3.4. The prevalent pool accumulated over
four decades during which the mix shifted greatly (37% of new AIDS diagnoses were
injection-attributed in 2001 against 9% of HIV diagnoses in 2023), so the recent
diagnosis mix overstates the current MSM share of prevalent cases.

This probe therefore does not assert a seed value. It runs two candidate seeds --
one derived from the RIDOH numbers above, one set at the level the model's own
burn-in currently converges to -- and reports which, if either, leaves the scored
window stationary at a short burn. Stationarity is the test; the seed is the
variable.

ARMS

  1. burn 720, seed 0,    dx 0.900, bg 0.0100   current model (control)
  2. burn 720, seed 0,    dx 0.932, bg 0.0034   data fixes only, burn unchanged
  3. burn 120, seed 0,    dx 0.932, bg 0.0034   short burn, no seed (expected to fail)
  4. burn 120, seed A,    dx 0.932, bg 0.0034   short burn + RIDOH-derived seed
  5. burn 120, seed B,    dx 0.932, bg 0.0034   short burn + equilibrium-level seed
  6. burn 240, seed A,    dx 0.932, bg 0.0034   does a longer short burn help?

Seed A (RIDOH-derived):     het 0.0034, MSM/MSMW 0.115
Seed B (model equilibrium): het 0.0200, MSM/MSMW 0.0680  (ratio 3.4 retained)

All arms hold trial 290's fitted parameters fixed -- it is job 4874631's best HIV
score (ridoh_hiv_err 0.397) and an exact hit on the injection-share target, so it
is the vector least likely to confound a seeding result with a bad fit.

Reported per arm: PWID and NonInj prevalence at the open and close of the scored
window (the stationarity test), the diagnosed fraction, and HIV incidence split
by acquisition route.

Usage:
    python -m titan.calibration.probe_seed_burn --reps 10 --pop-seeds 2
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    MSM_SEX_TYPES,
    run_titan,
    write_yaml,
)

BASE_TRIAL_YAML = Path(
    "/oscar/scratch/jtao9/titan_calibration/optuna_multi_runs/trial_290.yml"
)
OUTPUT_DIR = Path("/oscar/scratch/jtao9/titan_calibration/seed_burn_probe")

POP_SEEDS = [12345, 20260720]

# (label, burn_steps, het_seed, msm_seed, dx_init, background_partner_prev,
#  inj_seed, testing_rate)
#
# inj_seed None keeps the scenario's existing 0.075; testing_rate None keeps the
# trial's fitted hiv_testing.base_prob_annual.
ARMS = [
    ("A1_burn720_seed0_current", 720, 0.0, 0.0, 0.900, 0.0100, None, None),
    ("A2_burn720_seed0_datafix", 720, 0.0, 0.0, 0.932, 0.0034, None, None),
    ("A3_burn120_seed0", 120, 0.0, 0.0, 0.932, 0.0034, None, None),
    ("A4_burn120_seedRIDOH", 120, 0.0034, 0.115, 0.932, 0.0034, None, None),
    ("A5_burn120_seedEquil", 120, 0.0200, 0.0680, 0.932, 0.0034, None, None),
    ("A6_burn240_seedRIDOH", 240, 0.0034, 0.115, 0.932, 0.0034, None, None),
]

# ROUND 2. Round 1 established three things: a NonInj seed is required once the
# burn is short (A3's PWID prevalence fell 21.8% across the scored window without
# one), the Inj seed of 0.075 is too high (every arm relaxed downward from it
# toward 0.042-0.047), and the diagnosed fraction is insensitive to burn length
# (0.989-0.998 across all six arms against RIDOH's measured 0.932).
#
# Round 2 therefore holds the burn at 120, lowers the Inj seed toward where the
# model actually settles, and sweeps the annual testing probability -- which is
# the parameter the round-1 result implicates in the saturation. The current
# search box is [0.134, 0.546] and trial 290 sits at 0.211; every value in that
# box saturates, so the sweep goes an order of magnitude below it.
ARMS_R2 = [
    ("B1_test0.211_inj0.075", 120, 0.0034, 0.115, 0.932, 0.0034, 0.075, 0.211),
    ("B2_test0.211_inj0.045", 120, 0.0034, 0.115, 0.932, 0.0034, 0.045, 0.211),
    ("B3_test0.100_inj0.045", 120, 0.0034, 0.115, 0.932, 0.0034, 0.045, 0.100),
    ("B4_test0.050_inj0.045", 120, 0.0034, 0.115, 0.932, 0.0034, 0.045, 0.050),
    ("B5_test0.020_inj0.045", 120, 0.0034, 0.115, 0.932, 0.0034, 0.045, 0.020),
    ("B6_test0.050_seedEquil", 120, 0.0200, 0.0680, 0.932, 0.0034, 0.045, 0.050),
]


def apply_arm(scenario, burn_steps, het, msm, dx_init, bg_prev,
              inj_seed=None, testing_rate=None):
    """Apply one arm's settings in place. Returns the number of strata touched."""
    scenario["model"]["time"]["burn_steps"] = int(burn_steps)

    if testing_rate is not None:
        scenario.setdefault("hiv_testing", {})["base_prob_annual"] = float(testing_rate)

    bg = scenario.setdefault("hiv", {}).setdefault("background_sex", {})
    bg["partner_hiv_prevalence_nonoud"] = float(bg_prev)

    n = 0
    for race in scenario.get("demographics", {}).values():
        for sx_name, sx in (race or {}).get("sex_type", {}).items():
            drug = (sx or {}).get("drug_type", {})

            noninj = drug.get("NonInj")
            if noninj is not None:
                init = msm if sx_name in MSM_SEX_TYPES else het
                noninj["hiv"] = {
                    "init": float(init),
                    "dx": {"init": float(dx_init)},
                }
                n += 1

            # dx.init moves on the Inj strata too. Leaving them at 0.9 while the
            # NonInj strata carry the RIDOH value would make the diagnosed
            # fraction a function of the injection share, which is itself fitted.
            inj = drug.get("Inj")
            if inj is not None:
                ih = inj.setdefault("hiv", {})
                ih.setdefault("dx", {})["init"] = float(dx_init)
                if inj_seed is not None:
                    ih["init"] = float(inj_seed)
                n += 1
    return n


def summarize(report_path, steps_per_year=12):
    """Open/close prevalence by compartment, diagnosed fraction, route split."""
    df = pd.read_csv(report_path, sep="\t")
    scored = df[df["t"] >= 1]
    if scored.empty:
        return None

    first_year = scored["t"].min()
    last_year_start = scored["t"].max() - steps_per_year + 1

    def prev(sub, drug_type):
        s = sub[sub["drug_type"] == drug_type]
        agents = s["agents"].sum()
        return float(s["hiv"].sum() / agents) if agents else float("nan")

    open_win = scored[scored["t"] < first_year + steps_per_year]
    close_win = scored[scored["t"] >= last_year_start]

    hiv_total = scored["hiv"].sum()
    out = {
        "pwid_open": prev(open_win, "Inj"),
        "pwid_close": prev(close_win, "Inj"),
        "noninj_open": prev(open_win, "NonInj"),
        "noninj_close": prev(close_win, "NonInj"),
        "dx_among_hiv": float(scored["hiv_dx"].sum() / hiv_total) if hiv_total else float("nan"),
        "hiv_new": int(scored["hiv_new"].sum()),
        "inj": int(scored["hiv_new_inj"].sum()),
        "sex": int(scored["hiv_new_sex"].sum()),
        "bg": int(scored["hiv_new_background"].sum()),
    }
    for comp in ("pwid", "noninj"):
        o, c = out[f"{comp}_open"], out[f"{comp}_close"]
        out[f"{comp}_drift"] = float(c / o - 1.0) if o else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--pop-seeds", type=int, default=2)
    ap.add_argument("--arms", type=str, default="", help="comma-separated arm labels")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()

    import yaml

    with open(BASE_TRIAL_YAML) as fh:
        base = yaml.safe_load(fh)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = ARMS if args.round == 1 else ARMS_R2
    arms = [a for a in pool if not args.arms or a[0] in args.arms.split(",")]
    rows = []

    for label, burn, het, msm, dx_init, bg_prev, inj_seed, testing in arms:
        for pop_seed in POP_SEEDS[: args.pop_seeds]:
            scenario = copy.deepcopy(base)
            n = apply_arm(scenario, burn, het, msm, dx_init, bg_prev,
                          inj_seed=inj_seed, testing_rate=testing)
            scenario["model"].setdefault("seed", {})["ppl"] = int(pop_seed)

            tag = f"{label}_pop{pop_seed}"
            yml = OUTPUT_DIR / f"{tag}.yml"
            outdir = OUTPUT_DIR / tag
            outdir.mkdir(exist_ok=True)
            write_yaml(yml, scenario)

            print(f"[{tag}] burn={burn} het={het} msm={msm} dx={dx_init} "
                  f"bg={bg_prev} strata={n}", flush=True)
            run_titan(yml, outdir, n_reps=args.reps)

            summary = summarize(outdir / "basicReport.txt")
            if summary is None:
                print(f"[{tag}] no scored rows", flush=True)
                continue
            summary.update(arm=label, pop_seed=pop_seed, burn=burn, inj_seed=inj_seed, testing=testing,
                           het=het, msm=msm, dx_init=dx_init, bg=bg_prev)
            rows.append(summary)

    if not rows:
        print("no results")
        return

    out = pd.DataFrame(rows)
    csv = OUTPUT_DIR / f"seed_burn_summary_r{args.round}.csv"
    out.to_csv(csv, index=False)

    agg = out.groupby("arm").mean(numeric_only=True)
    print("\n" + "=" * 78)
    print("STATIONARITY OF THE SCORED WINDOW (drift = close/open - 1)")
    print("=" * 78)
    print(f"{'arm':26s} {'PWID open':>9s} {'close':>7s} {'drift':>8s} "
          f"{'NonInj open':>11s} {'close':>7s} {'drift':>8s}")
    for arm in agg.index:
        r = agg.loc[arm]
        print(f"{arm:26s} {r.pwid_open:9.4f} {r.pwid_close:7.4f} {r.pwid_drift:+7.1%} "
              f"{r.noninj_open:11.4f} {r.noninj_close:7.4f} {r.noninj_drift:+7.1%}")

    print("\n" + "=" * 78)
    print("DIAGNOSED FRACTION (RIDOH 2022 measured value: 0.932)")
    print("=" * 78)
    for arm in agg.index:
        r = agg.loc[arm]
        print(f"{arm:26s} dx|HIV {r.dx_among_hiv:6.4f}   "
              f"undiagnosed reservoir {1 - r.dx_among_hiv:6.4f}")

    print("\n" + "=" * 78)
    print("HIV INCIDENCE BY ROUTE, SCORED WINDOW")
    print("=" * 78)
    for arm in agg.index:
        r = agg.loc[arm]
        tot = r.hiv_new or 1
        print(f"{arm:26s} total {r.hiv_new:7.1f}  inj {r.inj / tot:5.1%}  "
              f"sex {r.sex / tot:5.1%}  bg {r.bg / tot:5.1%}")

    print(f"\nWritten: {csv}")


if __name__ == "__main__":
    main()
