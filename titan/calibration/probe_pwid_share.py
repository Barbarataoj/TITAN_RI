"""
Diagnostic: can the model put the right SHARE of its HIV epidemic in PWID?

Run before committing to a refit. `probe_transmission_feasibility` (job 4738184)
established that `calibration.acquisition` can move the LEVEL of HIV onto RIDOH's
PWID diagnosis counts, but that doing so leaves the statewide epidemic 4-5x too
small:

    acquisition        2.04     1.0     0.5    0.35    0.25
    new HIV/yr ri      97.0    53.6    23.0    11.9     8.9      (RI sees ~62 dx/yr)
    PWID dx/yr         40.4    20.1    10.4     5.5     4.0      (RIDOH 7.0)

At acquisition 1.0 the statewide total is about right and PWID diagnoses are 2.9x
too high; at 0.4 PWID lands on RIDOH and the state epidemic collapses. Read as a
share, the model puts ~37% of its epidemic in PWID where RIDOH implies 7.0/62 =
11.3%. A global scalar has one degree of freedom and cannot change a ratio, so no
setting of it fixes this.

THE LEVER UNDER TEST. `calibration.noninj_hiv_init_het` seeds HIV prevalence on
every NonInj stratum (MSM/MSMW get it times NONINJ_HIV_MSM_TO_HET_RATIO = 3.4),
leaving Inj untouched at hiv.init = 0.075. It is the only fitted parameter that
moves the PWID/non-PWID balance rather than the overall level, so it is the only
candidate for closing the gap without new code.

Its search bound is [0.0, 0.04] and the current fitted value is 0.0283 -- already
71% of the way up. That is the reason for this probe rather than a refit: if the
share only reaches 11% somewhere past the bound, or not at all, then the refit
needs a parameter that does not exist yet and launching it would waste the run.

Note what this lever is and is not. It sets INITIAL prevalence, not ongoing
transmission, so it works by giving the non-PWID epidemic a larger reservoir to
grow from rather than by making non-PWID transmission more intense. Whether that
is enough is exactly the question.

Three questions:

  1. Does the PWID share of diagnoses reach RIDOH's 11.3%, and at what value of
     noninj_hiv_init_het? If it needs a value past 0.04, the bound is the
     binding constraint and can be moved. If it plateaus above 11.3% before
     then, no seeding value works and the model needs a PWID-specific
     transmission parameter instead.
  2. Can the share and the LEVEL be satisfied together? The share is only useful
     at a transmission level where total diagnoses are near RI's ~62/yr and PWID
     diagnoses are near RIDOH's 7.0/yr. A cell that fixes the share by shrinking
     PWID below target has not helped.
  3. Does objective slot 1 (ridoh_hiv_err) improve, and does PWID HIV prevalence
     stay near its 6.2% reference? Raising the NonInj reservoir feeds back into
     PWID through sexual mixing, so neither is guaranteed.

Everything is compared on DIAGNOSES, not infections. RIDOH's ~62/yr and 7.0/yr
are both diagnosis counts, and the previous probe's headline comparison mixed
model infections against RIDOH diagnoses, which understates the PWID share.

Usage:
    python -m titan.calibration.probe_pwid_share \
        --scenario calibrated_scenario_best.yml \
        --hets 0.0,0.01,0.02,0.028,0.04 --acquisitions 1.0,0.5,0.35 \
        --seeds 3 --reps 40
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
    NONINJ_HIV_MSM_TO_HET_RATIO,
    REPO_ROOT,
    STEPS_PER_YEAR,
    build_targets,
    load_yaml,
    multi_error,
    run_titan,
    set_noninj_hiv_init,
    write_yaml,
)

POP_SEEDS = [12345, 20260720, 8675309, 314159, 271828]

# RIDOH statewide HIV diagnoses per year, all exposure categories. The PWID
# target (IDU + MSM/IDU) is 7.0/yr against this, i.e. an 11.3% share.
RI_TOTAL_DX_PER_YEAR = 62.0
RI_PWID_DX_PER_YEAR = 7.0
RI_PWID_SHARE = RI_PWID_DX_PER_YEAR / RI_TOTAL_DX_PER_YEAR

SCRATCH = Path(os.environ.get("SCRATCH", f"/oscar/scratch/{os.environ['USER']}"))
OUT_DIR = SCRATCH / "titan_calibration" / "pwid_share_probe"


def run_one(args):
    """Run one (het, acquisition, pop_seed) cell and score it."""
    base, targets, het, acq, seed, reps = args
    tag = f"het{het:g}_acq{acq:g}_seed{seed}"
    outdir = OUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = copy.deepcopy(base)
    scenario.setdefault("calibration", {})["acquisition"] = float(acq)
    scenario.setdefault("model", {}).setdefault("seed", {})["ppl"] = int(seed)
    # 0 = fresh random seed per rep. -1 is NOT available -- utils.get_check_rand_int
    # rejects negatives despite what titan/params/model.yml claims -- and a positive
    # integer would give every rep the same seed. Precision comes from --reps.
    scenario["model"]["seed"]["run"] = 0

    # Same helper the calibration objective uses, so the seeding is laid down at
    # the same 24 paths with the same MSM/het ratio rather than re-derived here.
    set_noninj_hiv_init(
        scenario, het=het, msm=het * NONINJ_HIV_MSM_TO_HET_RATIO
    )

    yml = OUT_DIR / f"{tag}.yml"
    write_yaml(yml, scenario)
    stats = run_titan(yml, outdir, n_reps=reps)

    errs = multi_error(stats, targets)
    scale = targets["pop_scale"]

    df = pd.read_csv(outdir / "basicReport.txt", sep="\t")
    n_reps = int(df["run_id"].nunique()) if "run_id" in df.columns else 1
    post = df[df["t"] >= 1]
    post_pwid = post[post["drug_type"] == "Inj"]
    years = post["t"].max() / STEPS_PER_YEAR

    def per_year(frame, col):
        """Statewide-equivalent count per year of one report column."""
        return float(frame[col].sum()) / n_reps / years * scale

    total_dx = per_year(post, "hiv_dx_new")
    pwid_dx = per_year(post_pwid, "hiv_dx_new")
    idu_dx = per_year(post_pwid, "hiv_dx_new_inj")
    total_new = per_year(post, "hiv_new")
    pwid_new = per_year(post_pwid, "hiv_new")

    # Raw counts too, so report() can pool numerators and denominators across
    # population seeds instead of averaging each seed's own ratio.
    n_dx_total = float(post["hiv_dx_new"].sum())
    n_dx_pwid = float(post_pwid["hiv_dx_new"].sum())
    n_new_total = float(post["hiv_new"].sum())
    n_new_pwid = float(post_pwid["hiv_new"].sum())
    n_inj_pwid = float(post_pwid["hiv_new_inj"].sum())

    inj_all = df[df["drug_type"] == "Inj"]

    def prev_at(t):
        s = inj_all[inj_all["t"] == t]
        return float(s["hiv"].sum() / max(s["agents"].sum(), 1)) if len(s) else np.nan

    row = {
        "het": het,
        "acquisition": acq,
        "seed": seed,
        "ridoh_hiv_err": errs[1],
        "oud_rmse": errs[0],
        "od_err": errs[2],
        "total_dx_per_year_ri": total_dx,
        "pwid_dx_per_year_ri": pwid_dx,
        "idu_dx_per_year_ri": idu_dx,
        "total_new_per_year_ri": total_new,
        "pwid_new_per_year_ri": pwid_new,
        "n_dx_total": n_dx_total,
        "n_dx_pwid": n_dx_pwid,
        "n_new_total": n_new_total,
        "n_new_pwid": n_new_pwid,
        "n_inj_pwid": n_inj_pwid,
        "prev_pwid_burn_end": prev_at(0),
        "prev_pwid_end": prev_at(inj_all["t"].max()),
    }
    print(f"  done {tag}: slot1={errs[1]:.2f} total_dx={total_dx:.1f} "
          f"pwid_dx={pwid_dx:.1f} share={n_dx_pwid / max(n_dx_total, 1):.1%}",
          flush=True)
    return row


def report(df, hets, acqs):
    def cell(h, a):
        return df[(df["het"] == h) & (df["acquisition"] == a)]

    def pooled(s, num, den):
        d = s[den].sum()
        return s[num].sum() / d if d else np.nan

    print("\n" + "=" * 100)
    print(f"Q1  PWID SHARE OF DIAGNOSES   (RIDOH: {RI_PWID_SHARE:.1%} = "
          f"{RI_PWID_DX_PER_YEAR:g} of {RI_TOTAL_DX_PER_YEAR:g} per year)")
    print("=" * 100)
    print("  Pooled over population seeds. Rows are noninj_hiv_init_het "
          "(current fit 0.0283, bound 0.04).")
    print(f"\n  {'het':>8}" + "".join(f"{'acq ' + f'{a:g}':>14}" for a in acqs))
    for h in hets:
        cells = "".join(
            f"{pooled(cell(h, a), 'n_dx_pwid', 'n_dx_total'):>13.1%} " for a in acqs
        )
        print(f"  {h:>8g}{cells}")
    print(f"\n  {'RIDOH':>8}" + "".join(f"{RI_PWID_SHARE:>13.1%} " for _ in acqs))

    print("\n" + "=" * 100)
    print("Q2  CAN THE SHARE AND THE LEVEL HOLD AT ONCE?")
    print("=" * 100)
    print(f"  Each cell: total dx/yr | PWID dx/yr   "
          f"(RI {RI_TOTAL_DX_PER_YEAR:g} | RIDOH {RI_PWID_DX_PER_YEAR:g})")
    print(f"\n  {'het':>8}" + "".join(f"{'acq ' + f'{a:g}':>18}" for a in acqs))
    for h in hets:
        cells = ""
        for a in acqs:
            s = cell(h, a)
            cells += (f"{s['total_dx_per_year_ri'].mean():>9.1f} |"
                      f"{s['pwid_dx_per_year_ri'].mean():>7.1f} ")
        print(f"  {h:>8g}{cells}")

    print("\n" + "=" * 100)
    print("Q3  OBJECTIVE SLOT 1 AND PWID PREVALENCE   (gate 1.1; prevalence "
          "reference 6.2%)")
    print("=" * 100)
    print(f"  Each cell: ridoh_hiv_err | PWID prevalence at end of run")
    print(f"\n  {'het':>8}" + "".join(f"{'acq ' + f'{a:g}':>18}" for a in acqs))
    for h in hets:
        cells = ""
        for a in acqs:
            s = cell(h, a)
            cells += (f"{s['ridoh_hiv_err'].mean():>9.2f} |"
                      f"{s['prev_pwid_end'].mean():>7.3f} ")
        print(f"  {h:>8g}{cells}")

    print("\n" + "=" * 100)
    print("BEST CELLS   (ranked by how far the worst of the three headline "
          "quantities is off)")
    print("=" * 100)
    scored = []
    for h in hets:
        for a in acqs:
            s = cell(h, a)
            share = pooled(s, "n_dx_pwid", "n_dx_total")
            tot = s["total_dx_per_year_ri"].mean()
            pw = s["pwid_dx_per_year_ri"].mean()
            if not np.isfinite(share) or not tot or not pw:
                continue
            # Worst proportional miss across share, statewide total and PWID
            # count. Max rather than mean: a cell is only useful if ALL THREE
            # are close, and averaging lets one of them hide.
            worst = max(
                abs(share - RI_PWID_SHARE) / RI_PWID_SHARE,
                abs(tot - RI_TOTAL_DX_PER_YEAR) / RI_TOTAL_DX_PER_YEAR,
                abs(pw - RI_PWID_DX_PER_YEAR) / RI_PWID_DX_PER_YEAR,
            )
            scored.append((worst, h, a, share, tot, pw,
                           s["ridoh_hiv_err"].mean()))
    scored.sort()
    print(f"  {'het':>8}{'acq':>7}{'share':>9}{'total dx':>11}{'PWID dx':>10}"
          f"{'slot 1':>9}{'worst miss':>13}")
    for worst, h, a, share, tot, pw, slot1 in scored[:8]:
        print(f"  {h:>8g}{a:>7g}{share:>9.1%}{tot:>11.1f}{pw:>10.1f}"
              f"{slot1:>9.2f}{worst:>12.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario",
                    default=str(REPO_ROOT / "calibrated_scenario_best.yml"))
    # 0.028 is the current fitted value and 0.04 the current upper bound; 0.0 is
    # the no-NonInj-reservoir control that motivated adding the parameter.
    ap.add_argument("--hets", default="0.0,0.01,0.02,0.028,0.04")
    ap.add_argument("--acquisitions", default="1.0,0.5,0.35")
    ap.add_argument("--seeds", type=int, default=3)
    # 40 for the same reason as probe_transmission_feasibility: these are rare
    # events and a 10-rep run leaves +/-13pp on a share.
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--from-csv", default="")
    args = ap.parse_args()

    targets = build_targets(load_yaml(CALIB_YAML)["calibration"])
    hets = [float(x) for x in args.hets.split(",")]
    acqs = [float(x) for x in args.acquisitions.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "pwid_share_probe.csv"

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        hets = sorted(df["het"].unique())
        acqs = sorted(df["acquisition"].unique(), reverse=True)
    else:
        base = load_yaml(args.scenario)
        seeds = POP_SEEDS[: args.seeds]
        jobs = [(base, targets, h, a, s, args.reps)
                for h in hets for a in acqs for s in seeds]
        print(f"{len(jobs)} configs: hets={hets} acquisitions={acqs} "
              f"seeds={seeds} reps={args.reps}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_one, jobs))
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)

    report(df, hets, acqs)
    print(f"\nWrote {csv}")


if __name__ == "__main__":
    main()
