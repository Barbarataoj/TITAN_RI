"""
Diagnostic: how long a burn-in does this model need to reach steady state?

The scored window is fitted on the assumption that the burn-in has absorbed the
initial conditions. Measured 2026-08-04 on the current 60-step (5-year) burn-in,
it has not: over the burn-in alone, PWID HIV prevalence rises 9.7% -> 16.6% and
the PWID count falls 144 -> 113 per rep, and both are still moving at t = 0. The
scored window then sits on whatever level the burn-in happened to reach, which
makes the initial conditions an unfitted free parameter.

This matters for a specific decision. `calibration.noninj_hiv_init_het` and
`calibration.noninj_hiv_init_msm` are fitted parameters that exist only because
the NonInj HIV reservoir grew from zero across the scored window. If a long
enough burn-in lets that reservoir build endogenously, both parameters can be
deleted rather than re-anchored -- and the burn-in transient in the injection
share, MOUD coverage and active-OUD share goes away with them.

So this sweeps burn_steps and asks, for each quantity, whether it has stopped
moving by the end of the burn-in. The reported slope is the per-year change over
the final 12 burn-in steps; "flat" means that slope is small relative to the
quantity's own level, not that it is exactly zero (at N=500 it never is).

Cost scales with burn length: the 132-step scored window is unchanged, so a
360-step burn-in is ~2.6x the wall-clock of a 60-step one per rep.

Usage:
    python -m titan.calibration.probe_burnin_length \
        --scenario calibrated_scenario_best.yml \
        --burns 60,120,240,360 --seeds 3 --reps 10
"""

import argparse
import copy
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from titan.calibration.optuna_calibration import (
    REPO_ROOT,
    STEPS_PER_YEAR,
    load_yaml,
    run_titan,
    write_yaml,
)

# Same list validate_candidates uses, so a probe result and a validation result
# for the same configuration are comparable seed-for-seed.
POP_SEEDS = [12345, 20260720, 8675309, 314159, 271828]

SCRATCH = Path(os.environ.get("SCRATCH", f"/oscar/scratch/{os.environ['USER']}"))
OUT_DIR = SCRATCH / "titan_calibration" / "burnin_probe"


def measure(path):
    """
    Read one basicReport.txt into the level-and-slope summary this probe reports.

    Slopes are per year, taken over the final 12 steps of the burn-in (is it
    still moving when scoring starts?) and over the final 12 steps of the run
    (is it still moving at the end?).
    """
    df = pd.read_csv(path, sep="\t")
    n_reps = int(df["run_id"].nunique()) if "run_id" in df.columns else 1
    grp = df.groupby("t").sum(numeric_only=True)
    inj = df[df["drug_type"] == "Inj"].groupby("t").sum(numeric_only=True)
    non = df[df["drug_type"] == "NonInj"].groupby("t").sum(numeric_only=True)
    first_t, last_t = int(grp.index.min()), int(grp.index.max())

    def series(name):
        if name == "prev_pwid":
            return (inj["hiv"] / inj["agents"].replace(0, np.nan)).reindex(grp.index)
        if name == "prev_noninj":
            return (non["hiv"] / non["agents"].replace(0, np.nan)).reindex(grp.index)
        if name == "prev_all":
            return grp["hiv"] / grp["agents"].replace(0, np.nan)
        if name == "inj_share":
            inj_active = df[df["drug_type"] == "Inj"].groupby("t")["oud_active"].sum()
            return (inj_active / grp["oud_active"].replace(0, np.nan)).reindex(grp.index)
        if name == "moud":
            return grp["moud_active"] / grp["oud_active"].replace(0, np.nan)
        if name == "oud_active":
            return grp["oud_active"] / grp["agents"].replace(0, np.nan)
        raise KeyError(name)

    names = ["prev_pwid", "prev_noninj", "prev_all", "inj_share", "moud", "oud_active"]
    row = {"n_reps": n_reps, "first_t": first_t, "last_t": last_t}

    for name in names:
        s = series(name)
        # Level at the two anchors: end of burn-in, and end of run.
        row[f"{name}@burn_end"] = float(s.loc[0]) if 0 in s.index else np.nan
        row[f"{name}@end"] = float(s.loc[last_t])
        # Per-year slope over the last 12 steps of each phase.
        for tag, t_hi in (("burn", 0), ("run", last_t)):
            t_lo = t_hi - STEPS_PER_YEAR
            if t_lo < first_t:
                row[f"{name}_slope_{tag}"] = np.nan
                continue
            row[f"{name}_slope_{tag}"] = float(s.loc[t_hi] - s.loc[t_lo])

    # Scored-window HIV output, for continuity with the RIDOH objective.
    post = df[df["t"] >= 1]
    n_years = (last_t) // STEPS_PER_YEAR
    pwid_post = post[post["drug_type"] == "Inj"]
    row["pwid_dx_per_model_yr"] = float(pwid_post["hiv_dx_new"].sum()) / n_reps / n_years
    if "hiv_dx_new_inj" in pwid_post.columns:
        row["idu_dx_per_model_yr"] = (
            float(pwid_post["hiv_dx_new_inj"].sum()) / n_reps / n_years
        )
    return row


def run_one(args):
    """Run one (burn_steps, pop_seed) configuration and measure it."""
    base, burn, seed, reps = args
    tag = f"burn{burn}_seed{seed}"
    outdir = OUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = copy.deepcopy(base)
    scenario.setdefault("model", {}).setdefault("time", {})["burn_steps"] = int(burn)
    scenario["model"].setdefault("seed", {})["ppl"] = int(seed)
    # Per-rep run seeds stay random, matching validate_candidates: the sweep is
    # over population seeds, not over the Monte Carlo draw.
    scenario["model"]["seed"]["run"] = 0

    # Beside the output directory, never inside it: run_titan.py rmtree's its
    # outdir before reading the params file, so a params.yml written in there
    # is deleted before it can be parsed.
    yml = OUT_DIR / f"{tag}.yml"
    write_yaml(yml, scenario)
    run_titan(yml, outdir, n_reps=reps)

    row = measure(outdir / "basicReport.txt")
    row["burn"] = burn
    row["seed"] = seed
    print(f"  done {tag}", flush=True)
    return row


def plot(df, burns, out):
    """
    Two panels: the quantities that move with burn length, and those that do not.

    The question the figure answers is whether any curve turns over. A quantity
    approaching a steady state flattens as the burn-in grows; one that keeps
    climbing has no steady state to burn in to, and no burn-in length will fix
    it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Burn-in length probe: does the model reach a steady state?",
        fontsize=13, weight="bold",
    )

    for ax, keys, title, ylab in (
        (axes[0],
         [("prev_pwid", "HIV prevalence, PWID"),
          ("prev_noninj", "HIV prevalence, NonInj"),
          ("prev_all", "HIV prevalence, all agents")],
         "HIV: no plateau at any burn-in length", "prevalence at end of burn-in"),
        (axes[1],
         [("inj_share", "injection share of active OUD"),
          ("moud", "MOUD coverage"),
          ("oud_active", "active OUD share")],
         "OUD side: flat, and insensitive to burn-in", "share at end of burn-in"),
    ):
        for key, label in keys:
            y = [df[df["burn"] == b][f"{key}@burn_end"].mean() for b in burns]
            lo = [df[df["burn"] == b][f"{key}@burn_end"].min() for b in burns]
            hi = [df[df["burn"] == b][f"{key}@burn_end"].max() for b in burns]
            ax.plot(burns, y, "-o", lw=2, ms=6, label=label)
            ax.fill_between(burns, lo, hi, alpha=0.15, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("burn_steps  (12 = 1 year)")
        ax.set_ylabel(ylab)
        ax.set_xticks(burns)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    # Reference level for scale: what the PWID prevalence target used to be
    # before it was dropped as an objective on 2026-08-04. Every burn-in length
    # tested overshoots it, and the overshoot grows with the burn-in.
    axes[0].axhline(0.062, color="k", ls="--", lw=1.5)
    axes[0].text(burns[0], 0.068, "APCD PWID prevalence 6.2% (no longer scored)",
                 fontsize=8, va="bottom")

    top = max(
        df[df["burn"] == burns[-1]]["prev_pwid@burn_end"].mean() for _ in [0]
    )
    axes[0].annotate(
        "still climbing after 30 years of burn-in:\n"
        "no endemic equilibrium at this transmission level,\n"
        "so no burn-in length can absorb the initial conditions",
        xy=(burns[-1], top), xytext=(burns[0] + 15, top * 0.80),
        fontsize=8.5, style="italic", color="#7f1d1d",
        arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.2),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=str(REPO_ROOT / "calibrated_scenario_best.yml"))
    ap.add_argument("--burns", default="60,120,240,360")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--figure", default="", help="path to write the summary figure")
    ap.add_argument(
        "--from-csv",
        default="",
        help="skip the runs and re-report/re-plot an existing burnin_probe.csv",
    )
    args = ap.parse_args()

    burns = [int(b) for b in args.burns.split(",")]
    seeds = POP_SEEDS[: args.seeds]
    base = load_yaml(args.scenario)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "burnin_probe.csv"

    if args.from_csv:
        # Re-report or re-plot without paying for the runs again. A 360-step
        # burn-in is ~2.6x the wall-clock of a 60-step one, so iterating on the
        # figure should not re-run the sweep.
        df = pd.read_csv(args.from_csv)
        burns = sorted(df["burn"].unique())
    else:
        jobs = [(base, b, s, args.reps) for b in burns for s in seeds]
        print(f"{len(jobs)} configs: burns={burns} seeds={seeds} reps={args.reps}",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_one, jobs))
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)

    if args.figure:
        plot(df, burns, Path(args.figure))

    # ---- report: level and slope at the end of the burn-in, by burn length ----
    print("\n" + "=" * 78)
    print("LEVEL AND PER-YEAR SLOPE AT END OF BURN-IN (mean over seeds)")
    print("=" * 78)
    for name, label in [
        ("prev_pwid", "HIV prevalence, PWID"),
        ("prev_noninj", "HIV prevalence, NonInj"),
        ("prev_all", "HIV prevalence, all"),
        ("inj_share", "injection share of active OUD"),
        ("moud", "MOUD coverage"),
        ("oud_active", "active OUD share"),
    ]:
        print(f"\n{label}")
        print(f"  {'burn':>6}{'level@burn_end':>16}{'slope/yr':>12}"
              f"{'|slope|/level':>15}{'level@end':>12}{'slope/yr@end':>14}")
        for b in burns:
            sub = df[df["burn"] == b]
            lvl = sub[f"{name}@burn_end"].mean()
            slp = sub[f"{name}_slope_burn"].mean()
            rel = abs(slp) / lvl if lvl else np.nan
            print(f"  {b:>6}{lvl:>16.4f}{slp:>12.4f}{rel:>14.1%}"
                  f"{sub[f'{name}@end'].mean():>12.4f}"
                  f"{sub[f'{name}_slope_run'].mean():>14.4f}")

    print("\n" + "=" * 78)
    print("SCORED-WINDOW HIV OUTPUT (mean over seeds, model counts per model-year)")
    print("=" * 78)
    print(f"  {'burn':>6}{'PWID dx/yr':>14}{'IDU dx/yr':>12}")
    for b in burns:
        sub = df[df["burn"] == b]
        idu = sub["idu_dx_per_model_yr"].mean() if "idu_dx_per_model_yr" in sub else np.nan
        print(f"  {b:>6}{sub['pwid_dx_per_model_yr'].mean():>14.3f}{idu:>12.3f}")

    print(f"\nWrote {csv}")
    print(
        "\nRead: a quantity is equilibrated when |slope|/level at the end of the "
        "burn-in\nis small AND does not shrink further with a longer burn-in. If "
        "PWID and NonInj\nprevalence both flatten by some burn length, the fitted "
        "NonInj seeds can be\ndeleted and the reservoir left to build endogenously."
    )


if __name__ == "__main__":
    main()
