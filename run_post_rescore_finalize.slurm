#!/bin/bash
#SBATCH --job-name=finalize_refit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/finalize_%j.out
# After the pooled 8x40 re-score shards of a refit have finished:
#  1. merge the Phase 7 development branch into the main tree (fast-forward only;
#     the re-score ran on the frozen main tree, so this is the first safe moment),
#  2. build the pass list from the shards' validation_pooled.csv (pass4 on the pooled gate),
#  3. regenerate the Phase 7 baseline from the best pooled trial (overlay_care_layer carries
#     the new care-layer scalars), and
#  4. stage the primary-care arm ensemble (pc1-pc4, baseline) for every pass-list member
#     WITHOUT submitting it (make_phase7_ensemble.py without --submit): a human looks at the
#     pass list and the staged arithmetic, then submits.
# Usage: sbatch --dependency=afterok:<shard ids> --export=ALL,TAG=care_refit2,DEV_BRANCH=calibration/phase7-dev run_post_rescore_finalize.slurm
set -euo pipefail
module purge; module load python/3.9
source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI
TAG=${TAG:-care_refit2}; DEV_BRANCH=${DEV_BRANCH:-calibration/phase7-dev}
S=/oscar/scratch/${USER}/titan_calibration; VAL=${S}/val_${TAG}
ENS=/oscar/scratch/${USER}/titan_phase7/ensemble_${TAG}

echo "== 1. merge ${DEV_BRANCH} (ff-only)"
git status --short | grep -v '^??' && { echo "main tree has uncommitted changes; refusing to merge"; exit 1; } || true
git merge --ff-only "${DEV_BRANCH}" && git log --oneline -1

echo "== 2. pass list from the pooled re-score"
python - "${VAL}" <<'PY'
import csv, glob, sys, pathlib
val = pathlib.Path(sys.argv[1]); rows = []
for f in sorted(glob.glob(str(val / "out_*" / "validation_pooled.csv"))):
    rows += list(csv.DictReader(open(f)))
if not rows:
    sys.exit("no validation_pooled.csv found under " + str(val))
keys = ["trial", "err_0", "err_1", "err_2", "err_3", "raw_0", "raw_1", "raw_2", "raw_3", "pass4", "hiv_new_pooled"]
with open(val / "pooled_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=[k for k in keys if k in rows[0]], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
passed = [r for r in rows if r.get("pass4") == "True"]
print(f"re-scored {len(rows)} trials; pass4 on the pooled gate: {len(passed)}: {[r['trial'] for r in passed]}")
# best = lowest sum of gate-normalized errors among passers (else among all)
# one owner for the gates (rule 6): a hard-coded (0.05, 1.1, 0.01, 0.10) carried the
# pre-2026-10-05 MOUD gate here until 2026-10-08
sys.path.insert(0, '.'); from titan.calibration.optuna_calibration import OBJECTIVE_GATES as gates
pool = passed or rows
best = min(pool, key=lambda r: sum(float(r[f"err_{i}"]) / g for i, g in enumerate(gates)))
(val / "pass_list.txt").write_text(",".join(r["trial"] for r in passed) + "\n")
(val / "best_pooled.txt").write_text(best["trial"] + "\n")
print("best pooled trial:", best["trial"], {k: round(float(best[f"err_{i}"]), 4) for i, k in enumerate(("oud", "hiv", "od", "moud"))})
PY
BEST=$(cat ${VAL}/best_pooled.txt); PASS=$(cat ${VAL}/pass_list.txt)

echo "== 3. Phase 7 baseline from trial ${BEST} (re-anchored + care-layer overlay)"
cp ${VAL}/trials/trial_${BEST}.yml calibrated_scenario_best.yml
python scripts/make_phase7_baseline.py --input calibrated_scenario_best.yml --steps 240
python scripts/make_phase7_baseline.py --input calibrated_scenario_best.yml --steps 240 --check | tail -1

echo "== 4. stage the primary-care arm ensemble (not submitted)"
mkdir -p ${ENS}
if [ -n "${PASS}" ]; then MEMBERS="${PASS}"; else MEMBERS="${BEST}"; echo "no pass4 member: staging the best pooled trial only"; fi
python scripts/make_phase7_ensemble.py --out-dir ${ENS} --members "${MEMBERS}" --trials-dir ${VAL}/trials \
    --arms-module make_phase7_pc_arms --steps 240 --seeds 8 --reps 40 --batches 8 --hours 10
echo "staged in ${ENS}/trials; arithmetic in ${ENS}/arithmetic.txt; submit with the same command plus --submit after review"
