#!/usr/bin/env python
"""
Ensemble membership from fresh draws (rule 4: one draw selects, another measures).

Reads pooled-score CSVs (``validation_pooled*.csv`` from validate_candidates or
``scripts/rescore_pooled_reports.py``), one or more per trial, and decides
membership on the MEAN of a trial's draws: a trial is a member when the mean
of each objective over its draws clears that objective's gate. The draw that
selected a trial must not be among its inputs; the caller chooses the files.

Reports, per trial: the gate ratio per objective (mean over draws), the number
of draws, pass/fail, and for the HIV objective the margin in standard errors
(single-draw SD estimated from the draw pairs when two or more draws exist,
else the value given by --hiv-draw-sd). Writes the table and a pass list.

Usage:
    python scripts/membership_from_draws.py <csv> [<csv> ...] \
        --table <out.csv> --pass-list <out.txt> [--hiv-draw-sd 0.236]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.calibration.optuna_calibration import OBJECTIVE_GATES  # noqa: E402
from titan.calibration.validate_candidates import OBJECTIVES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--pass-list", type=Path, required=True)
    ap.add_argument("--hiv-draw-sd", type=float, default=None,
                    help="single-draw SD of the HIV objective as a fraction of its gate, used when a trial "
                         "has one draw; default: estimated from the trials with two or more draws")
    args = ap.parse_args()

    frames = []
    for path in args.csvs:
        df = pd.read_csv(path)
        df["trial"] = df["trial"].astype(str)
        df["source"] = path.name
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    errs = [f"err_{i}" for i in range(len(OBJECTIVES))]
    names = [n for n, _ in OBJECTIVES]

    g = allp.groupby("trial")
    mean = g[errs].mean()
    n_draws = g.size().rename("n_draws")
    table = pd.DataFrame({"n_draws": n_draws})
    for i, (name, gate) in enumerate(OBJECTIVES):
        table[f"{name}_ratio"] = mean[f"err_{i}"] / gate
        table[f"{name}_pass"] = mean[f"err_{i}"] <= gate
    table["member"] = table[[f"{n}_pass" for n in names]].all(axis=1)

    # HIV margin in SEs: single-draw SD from the within-trial spread of draws.
    hiv = f"err_{1}"
    spread = g[hiv].apply(lambda s: s.std(ddof=1) if len(s) > 1 else np.nan) / OBJECTIVE_GATES[1]
    sd1 = float(np.sqrt(np.nanmean(spread ** 2))) if spread.notna().any() else args.hiv_draw_sd
    if args.hiv_draw_sd is not None and not spread.notna().any():
        sd1 = args.hiv_draw_sd
    table["hiv_single_draw_sd"] = sd1
    table["hiv_margin_se"] = (1.0 - table[f"{names[1]}_ratio"]) / (sd1 / np.sqrt(table["n_draws"]))
    table["hiv_resolved"] = table["hiv_margin_se"] >= 2.0
    extra = [c for c in ("hiv_tests_per_pwid_yr", "idu_dx_2015_2019_ri", "pwid_dx_2015_2019_ri",
                         "idu_dx_per_year_ri", "pwid_dx_per_year_ri") if c in allp]
    table = table.join(g[extra].mean()) if extra else table
    table = table.sort_values([f"{names[1]}_ratio"])

    pd.set_option("display.width", 250)
    print(table.round(3).to_string())
    members = sorted(table.index[table.member], key=lambda s: (len(s), s))
    print(f"\nmembers ({len(members)}): {', '.join(members)}")
    print(f"HIV single-draw SD used: {sd1:.3f} of the gate; resolved (>= 2 SE): "
          f"{', '.join(sorted(table.index[table.member & table.hiv_resolved], key=lambda s: (len(s), s)))}")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table)
    args.pass_list.write_text("\n".join(members) + "\n")
    print(f"written {args.table} and {args.pass_list}")


if __name__ == "__main__":
    main()
