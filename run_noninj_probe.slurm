#!/bin/bash
#SBATCH --job-name=titan_seed_probe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/probe_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/probe_%j.err
#
# Sweeps the NonInj hiv.init seed and reports PWID HIV incidence trajectory shape.
#
# Tests whether the flat model trajectory measured 2026-08-03 (job 4576015: model
# T1->T2 +1% against the target's -36%, no candidate beating a flat-at-mean
# predictor) is an initial-condition artifact. hiv.init is set on drug_type Inj
# only, so 75% of every sex_type stratum -- including all MSM -- starts
# HIV-negative and grows from zero across the scored window.
#
# A 4-rep single-seed pilot already moved T1->T2 from -13% (seed 0.0) to -53%
# (seed 0.03), bracketing the target. This resolves the grid between them at
# 10 reps x 3 population seeds x 2 trials.
#
# ~20 s per config => 6 inits x 3 seeds x 2 trials is roughly 12 min. 2 h is margin.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

python -m titan.calibration.probe_noninj_seed \
    --trials "${PROBE_TRIALS:-2945,2401}" \
    --inits "${PROBE_INITS:-0.0,0.005,0.010,0.015,0.020,0.030}" \
    --seeds "${PROBE_SEEDS:-3}" \
    --reps "${PROBE_REPS:-10}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
