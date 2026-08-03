#!/bin/bash
#SBATCH --job-name=titan_discrim
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/discrim_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/discrim_%j.err
#
# Go/no-go on the refit: measures, per objective, whether the spread across
# parameter vectors exceeds the run-to-run noise in a single 10-rep evaluation.
# Job 4576015 measured 1.15 and 1.50 for the two HIV objectives (need >2), which
# is why the last front's HIV ranking was meaningless. This checks whether the
# new calibration.noninj_hiv_init_{het,msm} dimensions widened that spread,
# BEFORE run_calibration.slurm spends 6 h on the assumption that they did.
#
# RE-RUN 2026-08-03 with calibration.sex.act swept. The first pass (job 4580476)
# measured HIV trajectory at 0.93 and HIV prevalence at 1.93 -- but it was taken
# while calibration.sex.act was still a dead search dimension, so the network
# sexual route sat at the engine default in every config. That lever is the one
# that fits the sexual route and is recorded as violently sensitive, so the first
# pass understates the between-trial spread on exactly the two objectives whose
# discrimination decides the refit budget.
#
# 24 configs x 3 batches x 10 reps at ~10 s per batch is roughly 15 min.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

python -m titan.calibration.probe_discrimination \
    --trials "${DISC_TRIALS:-2566,2945,2812}" \
    --het "${DISC_HET:-0.000,0.025}" \
    --msm "${DISC_MSM:-0.00,0.12}" \
    --sexact "${DISC_SEXACT:-0.5,1.5,2.5}" \
    --batches "${DISC_BATCHES:-3}" \
    --reps "${DISC_REPS:-10}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
