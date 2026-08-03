#!/bin/bash
#SBATCH --job-name=titan_val
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/val_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/val_%j.err
#
# Multi-seed re-evaluation of the post-refit Pareto front (job 4436043, 640
# members). No re-optimization: replays saved trial_<N>.yml under independent
# population seeds.
#
# Purpose here is de-noising at SELECTION rather than adjudicating one candidate.
# NSGA-II sorted that front on 10-rep values whose HIV terms are mostly Monte
# Carlo noise: at trial 3404 hiv_traj_rmse is 0.00411 against a ~0.00326 noise
# floor (79% of the objective's variance), and PWID prevalence carries a 55% CV
# between runs. The top two by composite are 0.2448 vs 0.2465 -- inside the
# noise. 5 seeds x 10 reps drops the standard error ~2.2x, which is what makes a
# canonical pick meaningful.
#
# Candidate list is the transmission-screened top 20 by composite, recovered from
# the calibration stdout by titan.calibration.rank_pareto_from_log (the study was
# created without storage, so the front exists nowhere else).
#
# Batches run serially; run_titan parallelizes its reps over SLURM_CPUS_PER_TASK.
# ~90 s per batch => 20 trials x 5 seeds x 10 reps is roughly 2.5 h. 8 h is margin.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

export TITAN_VAL_TRIALS=${TITAN_VAL_TRIALS:-3404,349,1988,2945,3087,1628,1126,2439,2811,2181,2812,988,3178,2256,1330,934,1429,2401,3374,3709}
export TITAN_VAL_SEEDS=${TITAN_VAL_SEEDS:-5}
export TITAN_VAL_REPS=${TITAN_VAL_REPS:-10}

echo "Config: TRIALS=${TITAN_VAL_TRIALS} SEEDS=${TITAN_VAL_SEEDS} REPS=${TITAN_VAL_REPS} CORES=${SLURM_CPUS_PER_TASK}"

python -m titan.calibration.validate_candidates

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Outputs: /oscar/scratch/${USER}/titan_calibration/validation/validation_summary.csv"
