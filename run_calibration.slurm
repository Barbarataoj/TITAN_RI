#!/bin/bash
#SBATCH --job-name=titan_calib
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/calib_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/calib_%j.err

# Fail the SLURM job when python fails. Without this, job 4586379 died after 14
# of 2000 trials on a storage error and still recorded State=COMPLETED, because
# the trailing echo ran anyway and became the job's exit code. A six-hour silent
# failure looked identical to success.
set -euo pipefail

echo "Job started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

# Trial budget. The default of 120 was set when the search space was smaller; at
# 19 dimensions and 5 objectives it is ~2 generations of a 50-member population,
# which is not a search. Measured cost 2026-07-30: ~80 s/trial at n_runs=10 and
# size=500, so 4000 trials across 16 workers is ~6 h against a 24 h wall.
# TITAN_CALIB_POP=100 (see optuna_calibration.py) makes that 40 generations.
#
# DIMENSION COUNT, 2026-08-06: 19 free parameters against 5 objectives. 18 come
# from calibration_targets.yml parameters_to_optimize and one is
# calibration.noninj_hiv_init_het, suggested directly in objective(). Was 20
# until calibration.acquisition left the search on 2026-08-06 as an exactly
# redundant dimension -- it multiplied both routes on top of their own bases, so
# `injection.transmission.base x acquisition` had one identifiable product and
# NSGA-II was searching a ridge. See the block in objective().
#
# The starting region is known from probe_transmission_feasibility (job 4738184)
# and probe_pwid_share (job 4740073), converted to the acquisition=1.0
# parameterisation: injection.transmission.base ~0.0038 of [0.001, 0.012],
# calibration.sex.act ~0.87 of [0.25, 3.0], noninj_hiv_init_het 0.02-0.028 of
# [0.0, 0.04]. All three sit well inside their bounds, so no bound needs moving.
#
# REPS RAISED 10 -> 20 AND TRIALS CUT 4000 -> 2000, 2026-08-03.
#
# probe_discrimination measures, per objective, the spread across parameter
# vectors against the run-to-run noise in ONE evaluation -- i.e. whether NSGA-II
# is ranking signal or coin flips. Ratios below ~2 rank near-randomly.
#
# Job 4581459 (2026-08-03) is the authoritative measurement: 3 trials x 2 het x
# 2 msm x 3 sex.act configs, 3 independent 10-rep batches each, at the pinned
# calibration population.
#
#   objective            @10 reps   @20 reps
#   OUD distribution         1.67       2.36
#   HIV trajectory           1.64       2.32
#   HIV prevalence           3.06       4.33
#   Overdose                 1.71       2.42
#   MOUD trajectory         15.77      22.30
#   Injection share          1.57       2.22
#
# All six clear 2.0 at 20 reps, so every objective ranks on signal. Noise falls
# as 1/sqrt(reps); halving the trials keeps the ~6 h wall and still gives 20
# generations at TITAN_CALIB_POP=100.
#
# SUPERSEDES job 4580476, which measured HIV trajectory at 0.93 and prevalence at
# 1.93 and prompted a note here that the trajectory was unrankable at any
# affordable rep count (46 reps for 2.0) and had to be abandoned to the selection
# stage. That probe was taken while calibration.sex.act was still a dead search
# dimension -- the sampler suggested it, but a `scenario["calibration"] = {...}`
# assignment discarded it, so the network sexual route ran at the engine default
# in every config. Restoring the lever roughly doubled trajectory spread and
# raised prevalence 1.6x. Both objectives are now fitted here, not deferred.
#
# Caveat that survives: probe_noninj_seed (job 4578413) shows the model still
# cannot reproduce the target's -36% T1->T2 decline at any NonInj seed value
# (best -15% at init 0.020). Discrimination >= 2 means the front is sorted on
# signal, not that a good fit exists. Expect HIV trajectory to converge to the
# least-bad near-flat shape and confirm at the selection stage.
export TITAN_CALIB_TRIALS=2000
export TITAN_CALIB_REPS=20

python titan/calibration/optuna_calibration.py

echo "Job finished at $(date)"
