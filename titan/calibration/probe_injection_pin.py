"""
Step 4 verification: what pinning injection transmission at 0.0063 does.

Step 4 of the 2026-08-12 plan fixes `partnership.injection.transmission.base` at
the published per-act value for sharing injection equipment, 0.0063 (Patel et
al. / CDC). The search box it replaces was [0.000581, 0.00466] -- the literature
value sat ABOVE the entire range, so every trial ever run used a lower one, the
best-scoring by about five times.

That is a large move on the route that carries the interventions, and it is made
against a model that has historically run 1.8-4.8x HOT on all-route PWID
diagnoses. Both directions are plausible a priori:

  * the injection route rises toward its measured intensity, the route mix
    shifts away from the sexual side, and RIDOH's IDU series becomes reachable
    without crushing the parameter the interventions act on; or
  * total HIV goes further above RIDOH and the refit is unfittable at 0.0063,
    in which case the honest conclusion is that some other term is absorbing the
    difference and the pin has to wait for it.

Rather than spend a 26-hour refit finding out, this sweeps the parameter across
the old box and past it, holding everything else at the post-step-3 anchors, and
reports the objective the refit will actually be scored on.

ARMS. `base` at the old fitted value, the old ceiling, the literature value, and
one above it to show the gradient. `calibration.sex.act` is swept as a second
axis on the literature arm only, because the pin's whole argument is that the
sexual lever should absorb the level while the injection route is fixed by
evidence -- so the question is not just "is 0.0063 too hot" but "is there a
sex.act that makes 0.0063 fit".

Run on a compute node:

    sbatch --cpus-per-task=4 --time=04:00:00 scripts/run_on_node.slurm \\
      python -m titan.calibration.probe_injection_pin --reps 20
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from titan.calibration.optuna_calibration import (
    CALIB_YAML,
    INJECTION_TRANSMISSION_BASE,
    build_targets,
    hiv_null_value,
    load_yaml,
    parse_basic_report,
    ridoh_hiv_rmse,
    run_titan,
    set_hiv_initial_conditions,
    set_oud_initial_conditions,
    set_per_act_transmission,
    set_population_window,
    set_yaml_value,
    write_yaml,
)

BASE_SCENARIO = Path("/oscar/data/jtao9/TITAN_RI/calibrated_scenario_best.yml")
OUTPUT_DIR = Path("/oscar/scratch/jtao9/titan_calibration/injection_pin_probe_burn120")

POP_SEEDS = [12345, 20260720]

# The fitted value in the canonical pick of job 4874631, and the old box edges.
FITTED_BASE = 0.003943
OLD_BOX_CEILING = 0.00465604

# (label, injection base, sex.act override or None to keep the scenario's)
ARMS = [
    ("A1_base_fitted", FITTED_BASE, None),
    ("A2_base_oldceiling", OLD_BOX_CEILING, None),
    ("A3_base_literature", INJECTION_TRANSMISSION_BASE, None),
    ("A4_base_above", 0.0090, None),
]

# Second axis on the literature arm: can the sexual lever absorb the level?
# calibration.sex.act's box is [0.25, 3.0]; the canonical pick sits near 0.87.
SEX_ACT_ARMS = [
    ("B1_lit_sexact0.25", INJECTION_TRANSMISSION_BASE, 0.25),
    ("B2_lit_sexact0.50", INJECTION_TRANSMISSION_BASE, 0.50),
    ("B3_lit_sexact1.50", INJECTION_TRANSMISSION_BASE, 1.50),
]


def apply_arm(scenario, calib, base, sex_act):
    """Post-step-3 anchors, then this arm's two knobs. Returns sex.act in force."""
    set_hiv_initial_conditions(scenario)
    set_oud_initial_conditions(scenario, calib)
    set_per_act_transmission(scenario)
    # Explicit, not inherited from the base scenario: the first run of this probe
    # (2026-08-13, before set_population_window existed) reported itself as burn
    # 120 while every arm actually ran the base scenario's 720.
    set_population_window(scenario, calib)

    set_yaml_value(scenario, "partnership.injection.transmission.base", base)

    calib_block = scenario.setdefault("calibration", {})
    if sex_act is not None:
        calib_block.setdefault("sex", {})["act"] = float(sex_act)
    return float(calib_block.get("sex", {}).get("act", float("nan")))


def summarize(outdir, targets, steps_per_year=12):
    """The HIV objective the refit will see, plus the route split behind it."""
    report = outdir / "basicReport.txt"
    if not report.exists():
        return None
    stats = parse_basic_report(report)

    err = ridoh_hiv_rmse(
        stats.get("idu_dx_counts_per_year"),
        stats.get("pwid_dx_counts_per_year"),
        targets["first_year"],
        targets,
    )

    df = pd.read_csv(report, sep="\t")
    scored = df[df["t"] >= 1]
    inj = int(scored["hiv_new_inj"].sum())
    sex = int(scored["hiv_new_sex"].sum())
    bg = int(scored["hiv_new_background"].sum())
    total = inj + sex + bg

    def prev(sub, drug_type):
        s = sub[sub["drug_type"] == drug_type]
        agents = s["agents"].sum()
        return float(s["hiv"].sum() / agents) if agents else float("nan")

    first, last = scored["t"].min(), scored["t"].max()
    open_win = scored[scored["t"] < first + steps_per_year]
    close_win = scored[scored["t"] >= last - steps_per_year + 1]

    pwid_open, pwid_close = prev(open_win, "Inj"), prev(close_win, "Inj")
    return {
        "ridoh_hiv_err": float(err),
        "hiv_new": total,
        "inj_share_of_incidence": float(inj / total) if total else float("nan"),
        "sex_share_of_incidence": float(sex / total) if total else float("nan"),
        "bg_share_of_incidence": float(bg / total) if total else float("nan"),
        "pwid_prev_open": pwid_open,
        "pwid_prev_close": pwid_close,
        "pwid_drift": float(pwid_close / pwid_open - 1.0) if pwid_open else float("nan"),
        "inj_share_active_oud": float(stats.get("inj_share_active_oud", float("nan"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--pop-seeds", type=int, default=2)
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--with-sex-act", action="store_true", default=True)
    args = ap.parse_args()

    calib = load_yaml(CALIB_YAML)
    targets = build_targets(calib["calibration"])
    null_value = hiv_null_value(calib)

    with open(BASE_SCENARIO) as fh:
        base_scenario = yaml.safe_load(fh)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = ARMS + (SEX_ACT_ARMS if args.with_sex_act else [])
    arms = [a for a in pool if not args.arms or a[0] in args.arms.split(",")]

    print(f"HIV null value (zero-transmission floor): {null_value:.4f}")
    print(f"Pass threshold for ridoh_hiv_err: 1.1\n", flush=True)

    rows = []
    for label, base, sex_act in arms:
        for pop_seed in POP_SEEDS[: args.pop_seeds]:
            scenario = copy.deepcopy(base_scenario)
            in_force = apply_arm(scenario, calib, base, sex_act)
            scenario["model"].setdefault("seed", {})["ppl"] = int(pop_seed)

            tag = f"{label}_pop{pop_seed}"
            yml = OUTPUT_DIR / f"{tag}.yml"
            outdir = OUTPUT_DIR / tag
            outdir.mkdir(exist_ok=True)
            write_yaml(yml, scenario)

            print(f"[{tag}] base={base:.6f} sex.act={in_force:.3f}", flush=True)
            run_titan(yml, outdir, n_reps=args.reps)

            summary = summarize(outdir, targets)
            if summary is None:
                print(f"[{tag}] no basicReport", flush=True)
                continue
            summary.update(arm=label, pop_seed=pop_seed, base=base, sex_act=in_force)
            rows.append(summary)
            print(f"[{tag}] ridoh_hiv_err={summary['ridoh_hiv_err']:.3f} "
                  f"inj_share_incid={summary['inj_share_of_incidence']:.3f} "
                  f"pwid_drift={summary['pwid_drift']:+.1%}", flush=True)

    if not rows:
        print("no results")
        return

    out = pd.DataFrame(rows)
    csv = OUTPUT_DIR / "injection_pin_summary.csv"
    out.to_csv(csv, index=False)

    cols = ["arm", "base", "sex_act", "ridoh_hiv_err", "hiv_new",
            "inj_share_of_incidence", "sex_share_of_incidence",
            "pwid_prev_close", "pwid_drift"]
    agg = out.groupby("arm", sort=False)[
        [c for c in cols if c not in ("arm",)]
    ].mean()
    print("\nmeans across population seeds:")
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {csv}")

    lit = out[out["base"] == INJECTION_TRANSMISSION_BASE]["ridoh_hiv_err"]
    if len(lit):
        print(f"\nliterature-pin arms: ridoh_hiv_err {lit.min():.3f}-{lit.max():.3f} "
              f"against pass threshold 1.1 and null {null_value:.3f}")


if __name__ == "__main__":
    main()
