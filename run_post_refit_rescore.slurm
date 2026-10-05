#!/bin/bash
#SBATCH --job-name=post_refit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/post_refit_%j.out
# After a refit lands: list its in-search 4/4 trials, copy their trial YAMLs out of
# the shared optuna_multi_runs/ (overwritten per study), and submit the pooled
# 8x40 re-score (validate_candidates, pooled gate statistic) in parallel shards.
# Usage: sbatch --dependency=afterok:<refit job> --export=ALL,REFIT_JOB=<id>,TAG=<tag> run_post_refit_rescore.slurm
set -euo pipefail
module purge; module load python/3.9
source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI
REFIT_JOB=${REFIT_JOB:?}; TAG=${TAG:-care_refit2}; SHARDS=${SHARDS:-3}
S=/oscar/scratch/${USER}/titan_calibration
JOURNAL=${S}/optuna_multi_runs/study_${TAG}.journal
VAL=${S}/val_${TAG}; mkdir -p ${VAL}/trials
# feasible trials from the journal (Optuna's per-trial log lines go to stderr, not the .out)
python scripts/journal_feasible.py "${JOURNAL}" --csv ${VAL}/trials_${TAG}.csv | tee ${VAL}/journal_summary.txt | tail -1 > ${VAL}/feasible.txt
python - "${VAL}/feasible.txt" "${VAL}/trials" <<'PY'
import shutil, sys, pathlib
feas = [t for t in open(sys.argv[1]).read().strip().split(",") if t]
src = pathlib.Path("/oscar/scratch/jtao9/titan_calibration/optuna_multi_runs"); dest = pathlib.Path(sys.argv[2])
for t in feas:
    shutil.copy2(src / f"trial_{t}.yml", dest / f"trial_{t}.yml")
print("copied", len(feas), "trial YAMLs to", dest)
PY
LIST=$(cat ${VAL}/feasible.txt)
[ -z "$LIST" ] && { echo "no feasible trials; nothing to re-score"; exit 0; }
IFS=',' read -r -a ARR <<< "$LIST"
n=${#ARR[@]}; per=$(( (n + SHARDS - 1) / SHARDS ))
export TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=40 TITAN_VAL_TRIAL_DIR=${VAL}/trials
for ((k=0; k<SHARDS; k++)); do
  chunk=("${ARR[@]:k*per:per}"); [ ${#chunk[@]} -eq 0 ] && continue
  export TITAN_VAL_TRIALS=$(IFS=,; echo "${chunk[*]}") TITAN_VAL_OUT=${VAL}/out_${k}
  hours=$(( ${#chunk[@]} * 20 / 60 + 2 ))
  jid=$(sbatch --parsable --export=ALL --job-name=rescore_${TAG}_${k} --cpus-per-task=16 --time=${hours}:00:00 run_validate_candidates.slurm)
  echo "shard ${k}: job ${jid}: ${TITAN_VAL_TRIALS} -> ${TITAN_VAL_OUT}"; echo "${jid}" >> ${VAL}/jobs.txt
done
echo "re-score shards submitted; read validation_pooled.csv in each out_<k>/ (pooled gate statistic, pass4)"
