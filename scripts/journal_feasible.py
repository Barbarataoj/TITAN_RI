#!/usr/bin/env python
"""
List an Optuna JournalStorage study's trials with their objective values and
parameters, and the feasible ones (all gates), without importing optuna.

    python scripts/journal_feasible.py <study.journal> [--csv out.csv] [--gates 0.05,1.0,0.01,0.05]

Prints a summary and the feasible trial numbers (comma-separated on the last
line, for the re-score wrapper). Trial numbers are the order of creation in the
journal (one study per journal file). 2026-10-02.
"""
import argparse
import json
import statistics as st
import sys

# Must match OBJECTIVE_GATES in titan/calibration/optuna_calibration.py (kept as a
# literal so this script runs without importing optuna). MOUD gate 0.10 -> 0.05 on
# 2026-10-05 with the weighted MOUD trajectory.
GATES = (0.05, 1.0, 0.01, 0.05)   # HIV gate 1.1 -> 1.0 on 2026-10-08, see OBJECTIVE_GATES
NAMES = ("oud_rmse", "ridoh_hiv_err", "od_err", "moud_traj_rmse")


def read(journal):
    number = {}
    params, values, state = {}, {}, {}
    order = 0
    for line in open(journal):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        op = d.get("op_code")
        tid = d.get("trial_id")
        if op == 4:                               # create trial: ids are sequential by creation
            tid = order; order += 1
            number[tid] = tid
            params.setdefault(tid, {})
        elif op == 5:                             # set param
            params.setdefault(tid, {})[d.get("param_name")] = d.get("param_value_internal")
        elif op == 6:                             # set state / values
            if d.get("values") is not None:
                values[tid] = d["values"]
            state[tid] = d.get("state")
    return number, params, values, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("journal")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--gates", default=",".join(map(str, GATES)))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    gates = tuple(float(x) for x in a.gates.split(","))
    number, params, values, state = read(a.journal)
    rows = []
    for tid, v in values.items():
        if state.get(tid) != 1 or len(v) != 4:
            continue
        rows.append((number[tid], v, params.get(tid, {})))
    rows.sort()
    feas = [n for n, v, _ in rows if all(x <= g for x, g in zip(v, gates))]
    if not a.quiet:
        print(f"trials complete: {len(rows)}; created: {len(number)}")
        print("per-gate pass:", {nm: sum(v[i] <= gates[i] for _, v, _ in rows) for i, nm in enumerate(NAMES)})
        print(f"feasible 4/4: {len(feas)}; 3/4: {sum(sum(x <= g for x, g in zip(v, gates)) == 3 for _, v, _ in rows)}")
        blocks = [sum(1 for n, v, _ in rows if b <= n < b + 100 and all(x <= g for x, g in zip(v, gates)))
                  for b in range(0, (max(n for n, _, _ in rows) // 100 + 1) * 100, 100)] if rows else []
        print("feasible per 100-trial block:", blocks)
        for n, v, p in rows:
            if n in feas:
                print(f"  trial {n}: " + " ".join(f"{nm}={x:.4f}" for nm, x in zip(NAMES, v))
                      + f" | acc {p.get('moud.acceptance', float('nan')):.3f} noncl {p.get('hiv_testing.nonclinical_annual', float('nan')):.3f}"
                      + f" msm {p.get('calibration.background.msm_inj_mult', float('nan')):.2f} act {p.get('calibration.injection.act', float('nan')):.3f}")
        odp = [p.get("moud.acceptance") for _, v, p in rows if v[2] <= gates[2] and p.get("moud.acceptance") is not None]
        if odp:
            print("moud.acceptance among od passers (n=%d): min %.2f med %.2f max %.2f" % (len(odp), min(odp), st.median(odp), max(odp)))
        nc = [p.get("hiv_testing.nonclinical_annual") for _, v, p in rows if v[1] <= gates[1] and p.get("hiv_testing.nonclinical_annual") is not None]
        if len(nc) > 3:
            print("nonclinical_annual among hiv passers (n=%d): quartiles %s" % (len(nc), [round(q, 3) for q in st.quantiles(nc, n=4)]))
    if a.csv:
        import csv
        keys = sorted({k for _, _, p in rows for k in p})
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["trial", *NAMES, *keys, "feasible"])
            for n, v, p in rows:
                w.writerow([n, *v, *[p.get(k) for k in keys], n in feas])
    print(",".join(map(str, feas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
