#!/bin/bash
#SBATCH --job-name=rebuild_sum
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/titan_phase7/ensemble/rebuild_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_phase7/ensemble/rebuild_%j.err
#
# Rebuild the ensemble-wide validation_summary from saved basicReport.txt files.
# See scripts/rebuild_validation_summary.py for why. Incremental: safe to re-run
# after more shards finish. Override the shard output dir with TITAN_VAL_OUT.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
module purge
module load python/3.9
source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI

# The refactor that exposed build_row must not have broken the shard script.
python -m pytest -q tests/search_design_test.py tests/traj_rmse_alignment_test.py 2>&1 | tail -3

OUT=${TITAN_VAL_OUT:-/oscar/scratch/${USER}/titan_phase7/ensemble/out}
python scripts/rebuild_validation_summary.py --out-dir "$OUT" --reps "${TITAN_VAL_REPS:-40}" --workers "$SLURM_CPUS_PER_TASK" ${TITAN_REBUILD_ARGS:-}

echo "Job ${SLURM_JOB_ID} finished at $(date)"
