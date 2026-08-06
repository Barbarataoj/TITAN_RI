"""
Diagnostic: is the RIDOH HIV target reachable at all, and at what cost?

Run before committing to a refit. The 2026-08-04 front sits 1.8-4.8x above RIDOH
on all-route HIV diagnoses among PWID and 2.3-6.3x on the injection-attributed
subset, and `probe_burnin_length` showed prevalence still climbing after 30 years
of burn-in -- no endemic equilibrium at the current transmission level. Both point
the same way: transmission is several times too hot. Neither shows whether a
colder setting exists that RIDOH would accept.

`calibration.acquisition` is the lever to sweep because it scales ALL THREE
routes at once -- injection, network sex, and the background force of infection.
Its search bound is [0.1, 3.0] and the current fitted value is ~2.04, so a 4-8x
reduction is well inside the space the optimizer could already have reached. It
did not go there because the objective it was fitting (the APCD per-PWID-year
incidence series) was itself several times too high; that target is gone.

This was written expecting acquisition to move the LEVEL without touching the
route MIX, since it scales every route by the same factor. At 40 reps x 3 seeds
(job 4738184) the injection share of PWID infections runs:

    acquisition   2.04     1.0     0.5    0.35    0.25
    inj share    38.8%   23.9%   26.5%   29.2%   31.2%
    pooled n      1058     510     249     130      96

So the mix is NOT invariant -- 2.04 sits distinctly higher than everything below
it -- but there is no clean trend within the cold range, where it wanders around
24-31% against RIDOH's 29-40%. A 10-rep version of this probe showed an apparent
monotonic collapse to 20.0%; that was sampling noise and it did not survive more
reps. Do not read a trend off the cold end of this table.

Three questions, and a refit is only worth launching if all three answer well:

  1. Does objective slot 1 (ridoh_hiv_err) approach its 1.1 gate, or bottom out
     above it? If it bottoms out, no setting of this parameter satisfies RIDOH
     and the problem is structural.
  2. Does PWID HIV prevalence stop climbing? Re-runs the burn-in question at
     each transmission level: a model RIDOH would accept should approach an
     endemic plateau rather than growing without bound.
  3. Does the injection route stay alive? A fit that reaches RIDOH by inerting
     injection is useless for an intervention study about PWID, and would show
     up here as the injection-attributed share of diagnoses collapsing.

REPS. The quantities here are RARE EVENTS and need far more reps than a typical
probe. Rhode Island sees ~7 HIV diagnoses per year among people who inject, and
the model population is 1/49.8 of the state, so one rep produces on the order of
ONE diagnosis. At 10 reps x 3 seeds the pooled PWID infection count at a
RIDOH-consistent acquisition is ~40, which puts a 6.8pp standard error on a
route share -- +/-13pp at 95%. Two runs of this probe at 10 reps disagreed by
exactly that much (injection share of diagnoses 29.8% vs 16.3% at acq 0.35),
which is sampling noise and not a defect. 40 reps is the default for that
reason; it costs about 8 minutes.

Usage:
    python -m titan.calibration.probe_transmission_feasibility \
        --scenario calibrated_scenario_best.yml \
        --acquisitions 2.04,1.0,0.5,0.35,0.25 --seeds 3 --reps 40
"""

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    REPO_ROOT,
    STEPS_PER_YEAR,
    build_targets,
    load_yaml,
    multi_error,
    parse_basic_report,
    run_titan,
    write_yaml,
)

POP_SEEDS = [12345, 20260720, 8675309, 314159, 271828]

SCRATCH = Path(os.environ.get("SCRATCH", f"/oscar/scratch/{os.environ['USER']}"))
OUT_DIR = SCRATCH / "titan_calibration" / "transmission_probe"


def run_one(args):
    """Run one (acquisition, pop_seed) configuration and score it."""
    base, targets, acq, seed, reps = args
    tag = f"acq{acq:g}_seed{seed}"
    outdir = OUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = copy.deepcopy(base)
    scenario.setdefault("calibration", {})["acquisition"] = float(acq)
    scenario.setdefault("model", {}).setdefault("seed", {})["ppl"] = int(seed)
    # 0 means "draw a fresh random seed for every rep", so two runs of this probe
    # are INDEPENDENT DRAWS, not a reproducible measurement -- jobs 4673148 and
    # 4673743 scored acquisition 2.04 at 18.58 and 11.01 on objective slot 1 for
    # that reason alone. It stays 0 anyway, because nothing better is reachable
    # from this file:
    #
    #   * -1 does NOT work, despite what titan/params/model.yml says about
    #     "stepwise through num_reps". Model.__init__ validates the seed through
    #     utils.get_check_rand_int, which raises "Random seed must be positive
    #     integer" on anything negative. Tried in job 4736883; all 40 reps died
    #     in Model creation. The model.yml description is stale for this path.
    #   * A positive integer is worse than useless: every rep would get the SAME
    #     seed and the Monte Carlo would collapse to one replicate.
    #
    # Reproducibility would mean driving reps from here -- one run_titan call per
    # rep at n_reps=1 with an explicit per-rep seed, each in its own output
    # directory, concatenating the reports afterwards. That also buys common
    # random numbers across acquisitions, which would sharpen the sweep's
    # differences. It is a real refactor and has not been done.
    #
    # Precision, which is the thing that actually matters here, comes from --reps
    # instead. See the REPS note in the module docstring.
    scenario["model"]["seed"]["run"] = 0

    # Beside the output directory, never inside it: run_titan rmtree's its outdir
    # before the params file is read.
    yml = OUT_DIR / f"{tag}.yml"
    write_yaml(yml, scenario)
    stats = run_titan(yml, outdir, n_reps=reps)

    errs = multi_error(stats, targets)
    scale = targets["pop_scale"]

    def per_year(key):
        s = stats.get(key)
        return float(np.mean(s)) * scale if s is not None and len(s) else np.nan

    idu = per_year("idu_dx_counts_per_year")
    pwid = per_year("pwid_dx_counts_per_year")

    # Route split of incident infections, from the counters added 2026-08-04.
    # `background` is the one no modelled intervention can reach, so its share is
    # the number that decides whether scenario contrasts mean anything.
    #
    # Kept as raw COUNTS, not shares. report() pools numerators and denominators
    # across population seeds before dividing; averaging each seed's own share
    # would be a mean of ratios, which is biased and needlessly noisy when the
    # denominators are ~10 events. Per-rep rates are recovered with n_reps.
    #
    # Split BOTH ways. The model-wide denominator is the one to quote for how the
    # epidemic runs overall, but it is the wrong one for any question about
    # people who inject: measured 2026-08-05, sex carries 84% of infections
    # model-wide and 69% among PWID, because MSM and MSMW are 5.5% of the
    # population and take 65% of all infections. Only the PWID figure is
    # comparable to RIDOH's IDU vs MSM/IDU split.
    df = pd.read_csv(outdir / "basicReport.txt", sep="\t")
    n_reps = int(df["run_id"].nunique()) if "run_id" in df.columns else 1
    post = df[df["t"] >= 1]
    post_pwid = post[post["drug_type"] == "Inj"]

    routes, routes_pwid = {}, {}
    for r in ("inj", "sex", "background"):
        col = f"hiv_new_{r}"
        routes[r] = float(post[col].sum()) if col in post.columns else 0.0
        routes_pwid[r] = float(post_pwid[col].sum()) if col in post.columns else 0.0
    total_incident = sum(routes.values())
    total_incident_pwid = sum(routes_pwid.values())

    # Sliced from `df`, NOT from `post`: `post` drops every t < 1, so t == 0 --
    # the end of burn-in and the start of the scored window -- is not in it, and
    # reading prevalence there off `post` silently returns 0/1 = 0.0 for every
    # configuration. The burn-in report also carries negative t (the run starts
    # at -burn_steps), so this slice must see the whole frame.
    inj_all = df[df["drug_type"] == "Inj"]

    def prev_at(t):
        s = inj_all[inj_all["t"] == t]
        return float(s["hiv"].sum() / max(s["agents"].sum(), 1)) if len(s) else np.nan

    prev_burn_start = prev_at(inj_all["t"].min())
    prev_burn_end = prev_at(0)
    prev_end = prev_at(inj_all["t"].max())

    row = {
        "acquisition": acq,
        "seed": seed,
        "ridoh_hiv_err": errs[1],
        "oud_rmse": errs[0],
        "od_err": errs[2],
        "moud_err": errs[3],
        "inj_share_err": errs[4],
        "idu_dx_per_year_ri": idu,
        "pwid_dx_per_year_ri": pwid,
        "inj_share_of_dx": idu / pwid if pwid else np.nan,
        "prev_pwid_burn_start": prev_burn_start,
        "prev_pwid_burn_end": prev_burn_end,
        "prev_pwid_end": prev_end,
        # Two separate movements, and they can point opposite ways: burn-in drift
        # says whether the transmission level is compatible with the seeded
        # prevalence at all, while drift over the scored window says whether the
        # model is stable where the objective actually looks.
        "prev_burn_drift": prev_burn_end - prev_burn_start,
        "prev_growth": prev_end - prev_burn_end,
        "n_reps": n_reps,
        # Raw counts, pooled over reps. report() sums these across seeds.
        "n_incident": total_incident,
        "n_incident_pwid": total_incident_pwid,
        "route_inj_n": routes["inj"],
        "route_sex_n": routes["sex"],
        "route_background_n": routes["background"],
        "route_inj_pwid_n": routes_pwid["inj"],
        "route_sex_pwid_n": routes_pwid["sex"],
        "route_background_pwid_n": routes_pwid["background"],
        # Total incident HIV across the whole model population, scaled by
        # pop_scale.
        #
        # DO NOT COMPARE THIS TO RI's STATEWIDE HIV DIAGNOSES. It was added on
        # 2026-08-06 to do exactly that and the comparison is INVALID, because
        # pop_scale is not a whole-population scale factor:
        #
        #     pop_scale = ri_active_oud / (oud_active_frac * population.size)
        #               = 13704 / 275 = 49.8
        #
        # It maps the model's OUD COMPARTMENT onto RI's 13,704 active-OUD adults.
        # The model is an OUD-focused synthetic population -- 55.6% active OUD,
        # 24.7% PWID -- not a scaled-down Rhode Island. Its non-OUD agents are the
        # partner and contact pool the network needs, and 45% x 500 x 49.8 claims
        # they stand for 11,205 people against RI's ~880,000 adults, a 78x
        # under-representation. So total model HIV x pop_scale is BOUND to land
        # far below RI's ~62 dx/yr no matter how well calibrated the model is,
        # and reading that gap as a defect produced a "PWID are 3x
        # over-represented" conclusion that was an artifact of the scale factor.
        #
        # Kept only as a relative diagnostic -- how total incidence moves ACROSS
        # the sweep, where the same invalid constant cancels. There is no valid
        # whole-population comparator for this model.
        "hiv_new_per_year_scaled": (
            total_incident / n_reps / (post["t"].max() / STEPS_PER_YEAR) * scale
        ),
    }
    print(f"  done {tag}: slot1={errs[1]:.2f} idu={idu:.1f} pwid={pwid:.1f}", flush=True)
    return row


def report(df, acqs, targets):
    idu_t = [w["cases_per_year"] for w in targets["ridoh_windows"]]
    pwid_t = [w["cases_per_year"] for w in targets["ridoh_pwid_windows"]]

    print("\n" + "=" * 92)
    print("Q1  DOES THE HIV OBJECTIVE REACH ITS GATE?   (slot 1, gate 1.1, "
          "null model 4.78)")
    print("=" * 92)
    print(f"  {'acquisition':>12}{'slot 1':>10}{'IDU dx/yr':>12}{'PWID dx/yr':>12}"
          f"{'x RIDOH IDU':>13}{'x RIDOH PWID':>14}")
    for a in acqs:
        s = df[df["acquisition"] == a]
        idu, pwid = s["idu_dx_per_year_ri"].mean(), s["pwid_dx_per_year_ri"].mean()
        print(f"  {a:>12g}{s['ridoh_hiv_err'].mean():>10.2f}{idu:>12.1f}{pwid:>12.1f}"
              f"{idu / np.mean(idu_t):>12.1f}x{pwid / np.mean(pwid_t):>13.1f}x")
    print(f"  {'RIDOH':>12}{'--':>10}{np.mean(idu_t):>12.1f}{np.mean(pwid_t):>12.1f}")

    print("\n" + "=" * 92)
    print("Q2  DOES PREVALENCE STOP CLIMBING?   (PWID HIV prevalence, "
          "6.2% reference)")
    print("=" * 92)
    print(f"  {'acquisition':>12}{'burn start':>12}{'burn end':>10}{'burn drift':>12}"
          f"{'sim end':>10}{'sim drift':>11}")
    for a in acqs:
        s = df[df["acquisition"] == a]
        print(f"  {a:>12g}{s['prev_pwid_burn_start'].mean():>12.3f}"
              f"{s['prev_pwid_burn_end'].mean():>10.3f}"
              f"{s['prev_burn_drift'].mean():>12.3f}"
              f"{s['prev_pwid_end'].mean():>10.3f}{s['prev_growth'].mean():>11.3f}")

    print("\n" + "=" * 92)
    print("Q3  DOES THE INJECTION ROUTE STAY ALIVE?   (RIDOH puts the injection "
          "share of PWID infections at 29-40%)")
    print("=" * 92)
    print("  Route split AMONG PWID -- the model-wide split is a different "
          "question, reported below.")
    print("  Shares are POOLED over population seeds (sum numerators, sum "
          "denominators), not")
    print("  averaged per-seed: a mean of ratios is biased when each denominator "
          "is ~10 events.")
    print(f"\n  {'acquisition':>12}{'PWID n':>9}{'inj':>9}{'sex':>9}"
          f"{'background':>12}   {'vs RIDOH 29-40%':>16}")
    for a in acqs:
        s = df[df["acquisition"] == a]
        n = s["n_incident_pwid"].sum()
        if not n:
            print(f"  {a:>12g}{0:>9.0f}{'--':>9}{'--':>9}{'--':>12}")
            continue
        inj = s["route_inj_pwid_n"].sum() / n
        verdict = "in band" if 0.29 <= inj <= 0.40 else (
            "BELOW" if inj < 0.29 else "above")
        print(f"  {a:>12g}{n:>9.0f}{inj:>9.1%}"
              f"{s['route_sex_pwid_n'].sum() / n:>9.1%}"
              f"{s['route_background_pwid_n'].sum() / n:>12.1%}   {verdict:>16}")

    print("\n  Model-wide, ALL agents (dominated by MSM/MSMW, ~5.5% of the "
          "population):")
    print(f"  {'acquisition':>12}{'all n':>9}{'inj':>9}{'sex':>9}{'background':>12}")
    for a in acqs:
        s = df[df["acquisition"] == a]
        n = s["n_incident"].sum()
        if not n:
            continue
        print(f"  {a:>12g}{n:>9.0f}{s['route_inj_n'].sum() / n:>9.1%}"
              f"{s['route_sex_n'].sum() / n:>9.1%}"
              f"{s['route_background_n'].sum() / n:>12.1%}")

    print("\n" + "=" * 92)
    print("TOTAL HIV INCIDENCE   (RELATIVE ONLY -- not comparable to RI "
          "statewide counts)")
    print("=" * 92)
    print(f"  {'acquisition':>12}{'total new HIV/yr, scaled':>28}"
          f"{'vs acq 2.04':>14}")
    ref = df[df["acquisition"] == acqs[0]]["hiv_new_per_year_scaled"].mean()
    for a in acqs:
        s = df[df["acquisition"] == a]
        v = s["hiv_new_per_year_scaled"].mean()
        print(f"  {a:>12g}{v:>28.1f}{v / ref if ref else np.nan:>13.2f}x")
    print("  pop_scale maps the model's OUD COMPARTMENT onto RI's active-OUD "
          "population, not")
    print("  the model population onto RI. The model is 55.6% active OUD and "
          "24.7% PWID by")
    print("  construction, so this column has no statewide comparator -- read "
          "only the ratios.")

    print("\n" + "=" * 92)
    print("OTHER OBJECTIVES  (transmission should not be bought with the OUD fit)")
    print("=" * 92)
    print(f"  {'acquisition':>12}{'OUD':>9}{'overdose':>11}{'MOUD':>9}"
          f"{'inj share':>11}")
    for a in acqs:
        s = df[df["acquisition"] == a]
        print(f"  {a:>12g}{s['oud_rmse'].mean():>9.4f}{s['od_err'].mean():>11.4f}"
              f"{s['moud_err'].mean():>9.4f}{s['inj_share_err'].mean():>11.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=str(REPO_ROOT / "calibrated_scenario_best.yml"))
    ap.add_argument("--acquisitions", default="2.04,1.0,0.5,0.35,0.25")
    ap.add_argument("--seeds", type=int, default=3)
    # 40, not 10: at a RIDOH-consistent acquisition the model produces on the
    # order of one diagnosis per rep, and 10 reps x 3 seeds leaves a +/-13pp
    # confidence interval on a route share. See the REPS note in the module
    # docstring.
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--from-csv", default="")
    args = ap.parse_args()

    targets = build_targets(load_yaml(CALIB_YAML)["calibration"])
    acqs = [float(a) for a in args.acquisitions.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "transmission_probe.csv"

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        acqs = sorted(df["acquisition"].unique(), reverse=True)
    else:
        base = load_yaml(args.scenario)
        seeds = POP_SEEDS[: args.seeds]
        jobs = [(base, targets, a, s, args.reps) for a in acqs for s in seeds]
        print(f"{len(jobs)} configs: acquisitions={acqs} seeds={seeds} "
              f"reps={args.reps}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_one, jobs))
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)

    report(df, acqs, targets)
    print(f"\nWrote {csv}")


if __name__ == "__main__":
    main()
