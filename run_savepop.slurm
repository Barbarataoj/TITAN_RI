#!/bin/bash
#SBATCH --job-name=titan_savepop
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/titan_savepop/savepop_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_savepop/savepop_%j.err
#
# Phase 5/7 prerequisite: save the canonical calibrated population ONCE so every
# downstream "what if" scenario (Phase 7) re-loads the same baseline via --poppath.
# Uses the finalized trial-137 scenario (calibrated_scenario_best.yml), the pick
# resolved in the 8-seed validation (job 4273764).
#
# Population creation only (no model run beyond burn-in), so this is fast (~1 min).
# Output persisted to ~/data/TITAN_RI/results_RI_pop (scratch has a 30-day TTL).

set -euo pipefail

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

OUTDIR="/oscar/data/jtao9/TITAN_RI/results_RI_pop"
mkdir -p "${OUTDIR}"
mkdir -p "/oscar/scratch/${USER}/titan_savepop"

run_titan \
    -p calibrated_scenario_best.yml \
    -n 1 \
    --savepop \
    -o "${OUTDIR}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Saved population in: ${OUTDIR}"
ls -la "${OUTDIR}"
