#!/usr/bin/env python
"""
Gate check for the insurance / SDH sensitivity set (work record §19.44): each sensitivity's status quo is re-run
through the validator at 500 agents on the ensemble members, one draw of 8 populations x 40 common-random-number
replicates on the population seeds of the base re-validation's draw A, so the four gates (OUD distribution, RIDOH
PWID diagnoses, overdose, MOUD trajectory) are reported for every variant on the same members and seeds as the base.

The variant is applied by analysis/chain_sensitivity_2026-10-12.py::modify_status_quo (one owner for each
sensitivity's definition; the 2,000-agent strategy rounds use the same function), to the built 500-agent scenarios
of the base re-validation, so nothing else differs.

Usage:
  python scripts/stage_sens_gates.py <base_reval_dir> <ensemble_file> <out_root> <variant,variant,...> [shards_per_variant=3]

Writes <out_root>/<variant>/trials/trial_<m>.yml, shard_<i>.txt, manifest.txt and <out_root>/submit.sh. Run
submit.sh from the frozen worktree (W=<worktree> bash <out_root>/submit.sh); it reads the population seeds and
CRN base from <base_reval_dir>/draw_seeds.txt when present, else the §19.43 draw-A values.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chain_sens", REPO / "analysis" / "chain_sensitivity_2026-10-12.py")
chain_sens = importlib.util.module_from_spec(spec); spec.loader.exec_module(chain_sens)

DRAW_A_POP_SEEDS = "1313,2424,3535,4646,5757,6868,7979,8080"   # §19.43 bundled re-validation, draw A
DRAW_A_CRN_BASE = 830000


def changed_paths(a, b, prefix=""):
    """Dotted paths whose values differ between two nested dicts (leaves compared with ==)."""
    out = []
    keys = set(a) | set(b)
    for k in sorted(keys, key=str):
        pa, pb = a.get(k), b.get(k)
        if isinstance(pa, dict) and isinstance(pb, dict):
            out += changed_paths(pa, pb, f"{prefix}{k}.")
        elif pa != pb:
            out.append(f"{prefix}{k}: {pa} -> {pb}")
    return out


def main():
    base_dir = Path(sys.argv[1]); ens_file = Path(sys.argv[2]); root = Path(sys.argv[3])
    variants = sys.argv[4].split(","); shards = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    members = [m for m in ens_file.read_text().replace(",", " ").split() if m]
    seeds_file = base_dir / "draw_seeds.txt"
    pop_seeds, crn = DRAW_A_POP_SEEDS, DRAW_A_CRN_BASE
    if seeds_file.exists():
        line = seeds_file.read_text().split("\n")[0].split()
        pop_seeds, crn = line[0], int(line[1])
    root.mkdir(parents=True, exist_ok=True)
    submit = ["#!/bin/bash", "# Gate check of the sensitivity set: one validator job per shard; run from the frozen worktree.",
              "set -e", f"ROOT={root}", 'W=${W:?set W to the frozen worktree of the committed revision}', "cd $W", ": > $ROOT/jobs.txt"]
    for v in variants:
        out = root / v; (out / "trials").mkdir(parents=True, exist_ok=True)
        manifest = []
        for m in members:
            base = yaml.safe_load((base_dir / "trials" / f"trial_{m}.yml").read_text())
            assert base["model"]["num_pop"] == 500 and base["features"]["sdh"] and base["features"]["insurance"], m
            sc = copy.deepcopy(base)
            note = chain_sens.modify_status_quo(sc, v)
            diff = changed_paths(base, sc)
            assert diff, f"{v} changed nothing on member {m}"
            (out / "trials" / f"trial_{m}.yml").write_text(yaml.safe_dump(sc, sort_keys=False))
            manifest.append(f"{m}: {note}; changed {len(diff)} fields: {diff}")
        (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
        chunks = [members[i::shards] for i in range(shards)]
        for i, ch in enumerate(chunks):
            (out / f"shard_{i}.txt").write_text(",".join(ch))
            submit += [f"export TITAN_VAL_TRIALS=$(cat {out}/shard_{i}.txt) TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=40 TITAN_VAL_CRN_BASE={crn} \\",
                       f"       TITAN_VAL_POP_SEEDS={pop_seeds} TITAN_VAL_TRIAL_DIR={out}/trials TITAN_VAL_OUT={out}/drawA_out_{i}",
                       f"J=$(sbatch --parsable --export=ALL --job-name=gate_{v}_{i} --cpus-per-task=16 --mem=32G --time=05:00:00 run_validate_candidates.slurm)",
                       f'echo $J >> $ROOT/jobs.txt; echo "{v} shard {i} -> job $J"']
        print(f"{v}: {len(members)} members, {shards} shards of {[len(c) for c in chunks]}; {manifest[0][:200]}")
    (root / "submit.sh").write_text("\n".join(submit) + "\n")
    (root / "members.txt").write_text("\n".join(members) + "\n")
    print(f"seeds {pop_seeds} crn {crn}; submit: W=<worktree> bash {root}/submit.sh")


if __name__ == "__main__":
    main()
