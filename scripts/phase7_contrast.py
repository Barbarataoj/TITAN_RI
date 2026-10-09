#!/usr/bin/env python3
"""
Phase 7 contrasts: each arm against the baseline, within population seed.

Reads the validate_candidates output tree of one Phase 7 run set
(``<run_dir>/<trial>_seed<seed>/basicReport.txt``, one file per arm x
population seed, every replicate appended with its own ``run_id``) and reports,
for every arm, the within-seed difference from the baseline on the three Phase
7 outcomes over the scenario horizon (model steps t >= 1, after burn-in, as parse_basic_report does):

  * new HIV infections among people who inject drugs (``hiv_new`` summed over
    the Inj stratum; all routes),
  * overdose deaths (``overdose_deaths``),
  * MOUD coverage among active OUD (``moud_active`` / ``oud_active``, averaged
    over the horizon),

plus HIV diagnoses among PWID by route as supporting detail. Counts are per
replicate over the whole horizon, averaged over replicates within a seed, then
differenced against the baseline on the same seed. The table gives the mean,
standard deviation, minimum and maximum of that difference across seeds, so
the spread is parameter-free Monte Carlo plus network-draw variance, not
between-population variance (section 10.3 item 4 of the Phase 6 pins note).

Counts are reported on the model's own population (N = 500 by default) and
scaled to the state with ``calibration.population.pop_scale`` (49.8: the OUD
compartment to Rhode Island's diagnosed, insured people with OUD). The scale
applies to OUD-compartment quantities only; every outcome here is one.

Usage:
    python scripts/phase7_contrast.py RUN_DIR --baseline scenario_1_baseline
        [--out CSV] [--t-min 133] [--t-max N]

`model.time.num_steps` counts POST-burn steps (the calibration runs 132 =
2012-2022 after a 120-step burn-in), so a 240-step Phase 7 run covers
2012-2031 and `--t-min 133` restricts the contrast to 2023 onward.

RUN_DIR is the TITAN_VAL_OUT the arms were scored into; arm names are the
``trial_<name>.yml`` basenames without the prefix. Needs titan_env (pandas).
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CALIB_YAML = REPO / "titan" / "calibration" / "calibration_targets.yml"


def pop_scale():
    # The same resolved targets the validator scores with (pop_scale lives there).
    sys.path.insert(0, str(REPO))
    from titan.calibration.validate_candidates import load_targets  # noqa: E402
    return float(load_targets()["pop_scale"])


def per_rep_outcomes(report_path, t_min=1, t_max=None):
    """One row per replicate: totals over model steps t_min <= t <= t_max."""
    df = pd.read_csv(report_path, sep="\t")
    df = df[df["t"] >= t_min]   # t >= 1 is the post-burn convention of parse_basic_report
    if t_max is not None:
        df = df[df["t"] <= t_max]
    inj = df[df["drug_type"] == "Inj"]
    by_run_inj = inj.groupby("run_id")
    by_run_all = df.groupby("run_id")
    out = pd.DataFrame({
        "hiv_new_pwid": by_run_inj["hiv_new"].sum(),
        "hiv_new_pwid_inj_route": by_run_inj["hiv_new_inj"].sum() if "hiv_new_inj" in inj else np.nan,
        "hiv_dx_new_pwid": by_run_inj["hiv_dx_new"].sum() if "hiv_dx_new" in inj else np.nan,
        "overdose_deaths": by_run_all["overdose_deaths"].sum(),
        "overdose_events": by_run_all["overdose_events"].sum(),
    })
    # Expected infections (2026-10-09): the hazard the engine rolled, summed
    # over exposures; same expectation as the counts, without the coin-flip
    # noise. _averted_prep is the part of the hazard PrEP removed (direct
    # protection only; onward transmission it prevents shows up in the hazard).
    for col, name in (("hiv_hazard", "hiv_hazard_pwid"), ("hiv_hazard_inj", "hiv_hazard_pwid_inj_route"),
                      ("hiv_hazard_averted_prep", "hiv_averted_prep_pwid"),
                      ("hiv_hazard_averted_prep_inj", "hiv_averted_prep_pwid_inj_route")):
        if col in inj:
            out[name] = by_run_inj[col].sum()
    if "hiv_hazard" in df:
        out["hiv_hazard_all"] = by_run_all["hiv_hazard"].sum()
        out["hiv_averted_prep_all"] = by_run_all["hiv_hazard_averted_prep"].sum()
    # coverage: mean over steps of (moud_active / oud_active) per run
    cov = (df.groupby(["run_id", "t"])[["moud_active", "oud_active"]].sum())
    cov = (cov["moud_active"] / cov["oud_active"].replace(0, np.nan)).groupby("run_id").mean()
    out["moud_coverage"] = cov
    if "od_survivors_on_moud" in df and "moud_start_postod_1step" in df:
        surv = by_run_all[["od_survivors_on_moud", "od_survivors_off_moud", "moud_start_postod_1step"]].sum()
        denom = (surv["od_survivors_on_moud"] + surv["od_survivors_off_moud"]).replace(0, np.nan)
        out["survivor_any_moud_30d"] = (surv["od_survivors_on_moud"] + surv["moud_start_postod_1step"]) / denom
    # Activity the cash-flow layer prices (2026-10-02): counts summed over the
    # window per run; `prep` and `care_reached` are per-step stocks, so their
    # sums are person-months.
    for col, name in (("hiv_tests", "hiv_tests"), ("care_encounters_primary_care", "pc_visits"),
                      # test-and-offer bundle and workload counts (2026-10-09)
                      ("prep_bundle_offers", "prep_bundle_offers"), ("prep_new_bundled", "prep_starts_bundled"),
                      ("prep_new_same_day", "prep_starts_same_day"), ("prep_new_deferred", "prep_starts_deferred"),
                      ("prep_indicated", "prep_indicated_person_months"),
                      ("care_offer_primary_care_hiv_test", "pc_test_offers"),
                      ("care_offer_primary_care_prep", "pc_prep_offers"),
                      ("prep_new", "prep_starts"), ("prep", "prep_person_months"),
                      ("prep_monitoring_visits", "prep_monitoring_visits"),
                      ("care_reach_visits", "reach_visits"), ("care_reached", "reached_person_months")):
        if col in df:
            out[name] = by_run_all[col].sum()
    # per-product PrEP person-months and injections (prep.use_products)
    for col in df.columns:
        if col.startswith("prep_injection") or (col.startswith("prep_") and col[5:] in ("oral", "cab_la", "lenacapavir")):
            out[col] = by_run_all[col].sum()
    out["n_steps"] = df.groupby("run_id")["t"].nunique()
    return out


def _one(args):
    path, t_min, t_max = args
    d = path.name.rsplit("_seed", 1)
    rep = per_rep_outcomes(path / "basicReport.txt", t_min, t_max)
    rep["arm"] = d[0].replace("trial", "", 1)
    rep["seed"] = d[1]
    return rep.reset_index()


def member_scales(spec, default):
    """`--member-scale 40=68.5,45=60.9`: a statewide scale per member label.

    The ensemble's members share one population scale, but a sensitivity set
    that varies the OUD split (scripts/make_oud_split_sensitivity.py) does not:
    the statewide pool is 13,704 / split, so each member needs its own. Members
    not listed get the calibration's scale.
    """
    out = {}
    for item in (spec or "").split(","):
        if item.strip():
            label, value = item.split("=")
            out[label.strip()] = float(value)
    return lambda member: out.get(str(member), default)


def ensemble_contrasts(reps, outcomes, args):
    """Within-(member, seed) differences against each member's own baseline, pooled over members.

    Pooled mean = mean over (member, seed) of the difference; its standard error treats
    members as the sampling unit (sd of member means / sqrt(members)), which carries the
    parameter-uncertainty spread the ensemble exists to measure. The per-member table is
    written beside the pooled one; since 2026-10-07 it carries each member's baseline
    mean, the Monte Carlo standard error of its difference over population seeds, the
    percentage change and the statewide difference on the member's own scale
    (--member-scale), so a set whose "members" are deliberate variants (a sensitivity
    axis) can be read member by member rather than only pooled.
    """
    parts = reps["arm"].str.split("__", n=1, expand=True)
    reps = reps.assign(member=parts[0], arm=parts[1])
    scale = pop_scale()
    scale_of = member_scales(getattr(args, "member_scale", None), scale)
    years = reps.groupby(["arm"])["n_steps"].mean().div(args.steps_per_year)
    seed_mean = reps.groupby(["member", "arm", "seed"])[outcomes].mean()
    base = seed_mean.xs(args.baseline, level="arm")
    per_member, pooled = [], []
    for arm in sorted(set(reps.arm)):
        a = seed_mean.xs(arm, level="arm")
        common = a.index.intersection(base.index)
        diff = (a.loc[common] - base.loc[common]).reset_index()
        by_member = diff.groupby("member")[outcomes].mean()
        by_member_se = diff.groupby("member")[outcomes].agg(
            lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan)
        base_member = base.loc[common].reset_index().groupby("member")[outcomes].mean()
        for oc in outcomes:
            is_count = oc not in ("moud_coverage", "survivor_any_moud_30d")
            for member, row in by_member.iterrows():
                bm = float(base_member.loc[member, oc])
                se = float(by_member_se.loc[member, oc])
                per_member.append({
                    "arm": arm, "member": member, "outcome": oc,
                    "baseline_mean": bm, "diff_mean": row[oc], "diff_se": se,
                    "pct": 100.0 * row[oc] / bm if bm else np.nan,
                    "pct_se": 100.0 * se / bm if bm else np.nan,
                    "diff_statewide": row[oc] * scale_of(member) if is_count else np.nan,
                })
            mm = by_member[oc]
            # Two standard errors for the pooled difference (2026-10-09): the
            # spread of the member means (parameter uncertainty + noise, but
            # an 11-sample SD that can come out small by chance: it did on
            # 2026-10-08, 1.7x too small) and the Monte Carlo error implied by
            # each member's own seeds. The reported diff_se is the larger.
            se_members = mm.std(ddof=1) / np.sqrt(len(mm)) if len(mm) > 1 else np.nan
            se_mc = float(np.sqrt(np.nanmean(by_member_se[oc] ** 2) / len(mm))) if len(mm) else np.nan
            pooled.append({
                "arm": arm, "outcome": oc, "n_members": len(mm), "n_member_seed": len(diff),
                "baseline_mean": base.loc[common][oc].mean(), "arm_mean": a.loc[common][oc].mean(),
                "diff_mean": diff[oc].mean(),
                "diff_se": np.nanmax([se_members, se_mc]),
                "diff_se_members": se_members, "diff_se_mc": se_mc,
                "member_sd": mm.std(ddof=1) if len(mm) > 1 else np.nan,
                "member_min": mm.min(), "member_max": mm.max(),
                "share_members_negative": float((mm < 0).mean()),
                # each (member, seed) difference on its member's own scale
                "diff_mean_statewide": (float((diff[oc] * diff["member"].map(scale_of)).mean())
                                        if is_count else np.nan),
                "horizon_years": float(years.loc[arm]),
            })
    table = pd.DataFrame(pooled)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(f"window: model steps {args.t_min}..{args.t_max if args.t_max is not None else 'end'}; "
          f"members: {sorted(set(reps.member))}; baseline: {args.baseline}; pop_scale {scale:.2f}")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        pd.DataFrame(per_member).to_csv(args.out.with_name(args.out.stem + "_per_member.csv"), index=False)
        print(f"\nwritten {args.out} and the per-member table beside it")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", required=True, help="arm name of the baseline, e.g. scenario_1_baseline")
    ap.add_argument("--t-min", type=int, default=1, help="first model step counted (post-burn steps start at 1; "
                    "the calibration window is steps 1-132 = 2012-2022, so --t-min 133 starts the horizon in 2023)")
    ap.add_argument("--t-max", type=int, default=None, help="last model step counted (inclusive)")
    ap.add_argument("--steps-per-year", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split-members", action="store_true",
                    help="arm names are '<member>__<arm>' (make_phase7_ensemble.py): contrast within (member, seed) "
                         "against that member's baseline, then pool over members")
    ap.add_argument("--member-scale", default=None,
                    help="per-member statewide scale for the per-member table, e.g. '40=68.5,45=60.9' "
                         "(OUD-split sensitivity: 13,704 / split / 500); unlisted members use the calibration scale")
    args = ap.parse_args()

    dirs = sorted(p for p in args.run_dir.iterdir() if p.is_dir() and (p / "basicReport.txt").exists())
    if not dirs:
        print(f"no basicReport.txt under {args.run_dir}")
        return 1
    with Pool(args.workers) as pool:
        reps = pd.concat(pool.map(_one, [(d, args.t_min, args.t_max) for d in dirs]))

    outcomes = [c for c in ("hiv_new_pwid", "hiv_new_pwid_inj_route", "hiv_dx_new_pwid",
                            "hiv_hazard_pwid", "hiv_hazard_pwid_inj_route", "hiv_averted_prep_pwid",
                            "hiv_averted_prep_pwid_inj_route", "hiv_hazard_all", "hiv_averted_prep_all",
                            "overdose_deaths", "overdose_events", "moud_coverage",
                            "hiv_tests", "pc_visits", "pc_test_offers", "pc_prep_offers", "prep_starts",
                            "prep_bundle_offers", "prep_starts_bundled", "prep_starts_same_day", "prep_starts_deferred",
                            "prep_indicated_person_months",
                            "prep_person_months", "prep_monitoring_visits", "reach_visits", "reached_person_months",
                            "prep_injections", "prep_injections_cab_la", "prep_injections_lenacapavir", "prep_injection_visits",
                            "prep_cab_la", "prep_lenacapavir", "prep_oral",
                            "survivor_any_moud_30d") if c in reps]
    if args.split_members:
        return ensemble_contrasts(reps, outcomes, args)
    seed_mean = reps.groupby(["arm", "seed"])[outcomes].mean()
    if args.baseline not in seed_mean.index.get_level_values(0):
        print(f"baseline arm '{args.baseline}' not found; arms present: {sorted(set(seed_mean.index.get_level_values(0)))}")
        return 1
    base = seed_mean.xs(args.baseline, level="arm")
    scale = pop_scale()
    years = reps.groupby(["arm", "seed"])["n_steps"].mean().div(args.steps_per_year)

    rows = []
    for arm in sorted(set(seed_mean.index.get_level_values(0))):
        a = seed_mean.xs(arm, level="arm")
        common = a.index.intersection(base.index)
        diff = a.loc[common] - base.loc[common]
        for oc in outcomes:
            is_count = oc not in ("moud_coverage", "survivor_any_moud_30d")
            rows.append({
                "arm": arm, "outcome": oc, "n_seeds": len(common),
                "baseline_mean": a.loc[common][oc].mean() if arm == args.baseline else base.loc[common][oc].mean(),
                "arm_mean": a.loc[common][oc].mean(),
                "diff_mean": diff[oc].mean(), "diff_sd": diff[oc].std(ddof=1) if len(common) > 1 else np.nan,
                "diff_min": diff[oc].min(), "diff_max": diff[oc].max(),
                "diff_mean_statewide": diff[oc].mean() * scale if is_count else np.nan,
                "horizon_years": float(years.loc[arm].mean()),
            })
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(f"window: model steps {args.t_min}..{args.t_max if args.t_max is not None else 'end'}")
    print(f"arms: {sorted(set(table.arm))}; baseline: {args.baseline}; seeds: {sorted(set(reps.seed))}; "
          f"replicates per (arm, seed): {int(reps.groupby(['arm', 'seed']).size().median())}; pop_scale {scale:.2f}")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        reps.to_csv(args.out.with_name(args.out.stem + "_per_rep.csv"), index=False)
        print(f"\nwritten {args.out} and the per-replicate table beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
