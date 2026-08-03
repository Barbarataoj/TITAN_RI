#!/bin/bash
#SBATCH --job-name=titan_scen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/titan_scenario/scen_%A_%a.out
#SBATCH --error=/oscar/scratch/%u/titan_scenario/scen_%A_%a.err
#
# Phase 7 — run one "what if" scenario per array task, each re-loading the SAME
# calibrated trial-137 population saved in Phase 5 (job 4277443). Reusing one
# baseline population means every scenario differs only by its parameter deltas,
# not by a fresh random draw.
#
# Submit (sizes the array to the number of active manifest lines):
#   cd /oscar/data/jtao9/TITAN_RI
#   N=$(grep -cvE '^#|^$' scenarios/manifest.txt)
#   sbatch --array=1-${N} run_scenarios.slurm
#   # override reps:  sbatch --array=1-${N} --export=NMC=20 run_scenarios.slurm
#
# Results archived to ~/data/TITAN_RI/results_scenarios/<scenario>/ (scratch has a
# 30-day TTL). Each scenario's outputs land in its own subdir.

set -euo pipefail

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9
source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI

MANIFEST="scenarios/manifest.txt"
NMC="${NMC:-10}"

# Nth active (non-comment, non-blank) manifest line for this array index.
SCEN=$(grep -vE '^#|^$' "${MANIFEST}" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [[ -z "${SCEN}" ]]; then
    echo "ERROR: no manifest entry for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
SCEN_YML="scenarios/${SCEN}"
if [[ ! -f "${SCEN_YML}" ]]; then
    echo "ERROR: scenario file ${SCEN_YML} does not exist" >&2
    exit 1
fi

# Resolve the saved-population tarball (Phase 5 output; random-prefixed filename).
POP=$(ls results_RI_pop/pop/*_pop.tar.gz 2>/dev/null | head -1)
if [[ -z "${POP}" ]]; then
    echo "ERROR: no saved population under results_RI_pop/pop/ — run run_savepop.slurm first" >&2
    exit 1
fi

NAME="${SCEN%.yml}"
OUTDIR="/oscar/data/jtao9/TITAN_RI/results_scenarios/${NAME}"
mkdir -p "${OUTDIR}"
mkdir -p "/oscar/scratch/${USER}/titan_scenario"

echo "Scenario : ${SCEN_YML}"
echo "Poppath  : ${POP}"
echo "Reps     : ${NMC}"
echo "Outdir   : ${OUTDIR}"

run_titan \
    -p "${SCEN_YML}" \
    -n "${NMC}" \
    --poppath "${POP}" \
    -o "${OUTDIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} (${NAME}) finished at $(date)"
ls -la "${OUTDIR}"
