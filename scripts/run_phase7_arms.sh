#!/usr/bin/env bash
# Stage the Phase 7 scenarios for validate_candidates and submit them.
#
# Every scenario under scenarios/ (the baseline and the generated arms) is run
# over the same fixed population seeds and replicate count that the calibration
# pass list was validated with -- validate_candidates' POP_SEEDS x TITAN_VAL_REPS
# -- so a contrast is paired within population seed by construction (section
# 10.3 item 4 of phase_6_pins_and_noise_correction_2026-09-29.md). The old
# run_scenarios.slurm / run_savepop.slurm path is not used: --savepop writes the
# population as CREATED (run_titan.single_run saves before the model runs) and a
# loaded population still starts at t = -burn_steps, so it pairs nothing that the
# population seed does not already pair, and it covers one population only.
#
# Usage:
#   scripts/run_phase7_arms.sh RUN_DIR [SEEDS] [REPS] [BATCHES]
#     RUN_DIR  where to stage trials/ and write out/ (default: /oscar/scratch/$USER/titan_phase7/<date>)
#     SEEDS    population seeds (default 8), REPS replicates per seed (default 40)
#     BATCHES  number of SLURM jobs to split the arms over (default 2, 16 cores each)
#
# Afterwards:
#   python scripts/phase7_contrast.py RUN_DIR/out --baseline scenario_1_baseline --out RUN_DIR/contrasts.csv
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-/oscar/scratch/${USER}/titan_phase7/$(date +%Y-%m-%d)}"
SEEDS="${2:-8}"
REPS="${3:-40}"
BATCHES="${4:-2}"

mkdir -p "${RUN_DIR}/trials"
names=()
for f in "${REPO}"/scenarios/scenario_*.yml; do
    base="$(basename "${f}" .yml)"
    cp "${f}" "${RUN_DIR}/trials/trial_${base}.yml"
    names+=("${base}")
done
if [ "${#names[@]}" -eq 0 ]; then
    echo "no scenarios/scenario_*.yml to run -- generate them first (make_phase7_baseline.py, make_phase7_scenarios.py)" >&2
    exit 1
fi
echo "staged ${#names[@]} scenarios in ${RUN_DIR}/trials: ${names[*]}"

# Split the arm list over BATCHES jobs. List-valued variables go through the
# environment with --export=ALL: sbatch splits --export=...,VAR=a,b on the commas.
export TITAN_VAL_SEEDS="${SEEDS}" TITAN_VAL_REPS="${REPS}" \
       TITAN_VAL_TRIAL_DIR="${RUN_DIR}/trials" TITAN_VAL_OUT="${RUN_DIR}/out"
per=$(( (${#names[@]} + BATCHES - 1) / BATCHES ))
for ((b = 0; b < BATCHES; b++)); do
    chunk=("${names[@]:b*per:per}")
    [ "${#chunk[@]}" -eq 0 ] && continue
    list="$(IFS=,; echo "${chunk[*]}")"
    hours=$(( (${#chunk[@]} * SEEDS * REPS) / 1200 + 2 ))   # ~6 min per 320 runs on 16 cores, plus margin
    job=$(cd "${REPO}" && TITAN_VAL_TRIALS="${list}" sbatch --parsable --job-name="phase7_${b}" \
          --cpus-per-task=16 --time="${hours}:00:00" --export=ALL run_validate_candidates.slurm)
    echo "batch ${b}: job ${job}: ${list}"
done
echo "outputs: ${RUN_DIR}/out/<arm>_seed<seed>/basicReport.txt; summary ${RUN_DIR}/out/validation_summary.csv (last batch wins -- use phase7_contrast.py)"
