#!/bin/bash
#SBATCH --job-name=titan_scenario
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/titan_scenario/scenario_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_scenario/scenario_%j.err

# ---------------------------------------------------------------------------
# Run a single TITAN_RI scenario against calibrated_scenario_multi_5obj.yml
#
# Oscar conventions (per CCV docs):
#   * output goes to /oscar/scratch/$USER (home is quota-limited; scratch
#     has a 30-day TTL — archive results to ~/data afterwards if needed)
#   * module purge + module load python before activating venv
#   * use SLURM_CPUS_PER_TASK for nMC parallelism (run_titan.py reads it)
#
# Usage:
#   sbatch run_scenario.slurm
#   sbatch --export=NMC=20 run_scenario.slurm   # override replicate count
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"

module purge
module load python/3.9

source ~/envs/titan_env/bin/activate

cd ~/projects/TITAN_RI

NMC="${NMC:-5}"
OUTDIR="/oscar/scratch/${USER}/titan_scenario/run_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

python -m titan.run_titan \
    -p calibrated_scenario_multi_5obj.yml \
    -o "${OUTDIR}" \
    -n "${NMC}"

echo "Job finished at $(date)"
echo "Results in: ${OUTDIR}"
