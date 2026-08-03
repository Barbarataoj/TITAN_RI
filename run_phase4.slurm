#!/bin/bash
#SBATCH --job-name=titan_p4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/p4_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/p4_%j.err
#
# Phase 4 production calibration run (200 trials x 10 MC reps).
# Prereq: mkdir -p /oscar/scratch/$USER/titan_calibration   (SBATCH --output dir must exist)
#
# NOTE: the Optuna study is in-memory (no storage= in optuna_calibration.py), so a
# walltime kill loses ALL progress. Estimated runtime ~1-1.5 h at 16 workers
# (12 trials x 10 reps took ~8 min on 6 workers in the 2026-07-13 smoke); 8 h ceiling
# is a deliberate safety margin. Do not lower it below ~3 h.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

# Phase 4 knobs. Reps are driven ONLY by TITAN_CALIB_REPS -> run_titan -n
# (the model.num_reps YAML key is dead config). SLURM_CPUS_PER_TASK sets Optuna workers.
export TITAN_CALIB_TRIALS=200
export TITAN_CALIB_REPS=10

echo "Config: TRIALS=${TITAN_CALIB_TRIALS} REPS=${TITAN_CALIB_REPS} WORKERS=${SLURM_CPUS_PER_TASK}"
echo "Scenario: calibrated_scenario_multi_5obj.yml  (HIV seeding + start_time + dx.prob + 2014 target fixes applied 2026-07-13)"

python -m titan.calibration.optuna_calibration

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Outputs: /oscar/scratch/${USER}/titan_calibration/optuna_multi_runs/ (pareto_front.csv, calibrated_scenario_best.yml)"
