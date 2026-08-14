#!/bin/bash
#SBATCH --job-name=titan_calib
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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
# DIMENSION COUNT, 2026-08-13: 16 free parameters against 5 objectives, all 16
# from calibration_targets.yml parameters_to_optimize. Was 18 until step 2 of the
# 2026-08-12 plan took inject_start_rate_per_year and inject_stop_rate_per_year
# out of the search and fixed them from the cohort literature -- no scenario
# changes either rate, so under upstream TITAN's data-slot rule they are inputs.
# Was 20 until
# calibration.acquisition left the search on 2026-08-06 as an exactly redundant
# dimension -- it multiplied both routes on top of their own bases, so
# `injection.transmission.base x acquisition` had one identifiable product and
# NSGA-II was searching a ridge -- and 19 until calibration.noninj_hiv_init_het
# left it on 2026-08-07, when job 4779604 showed the NonInj reservoir builds
# itself during a 720-step burn-in. See the blocks in objective().
#
# STARTING REGION, revised 2026-08-07. The old figures here came from
# probe_transmission_feasibility (job 4738184) and probe_pwid_share (job
# 4740073), both measured at burn_steps=60, and both are wrong at burn 720 --
# run to equilibrium the acquisition~0.5 region they identified is 3-5x hot on
# RIDOH. The equilibrated level is acquisition~0.35 equivalent:
#
#   injection.transmission.base   ~0.00138 of [0.0003, 0.012]
#   calibration.sex.act           ~0.75    of [0.25, 3.0]
#
# The injection lower bound was moved 0.001 -> 0.0003 for this: the feasible
# value sat 38% above the old bound and would have come back as a boundary
# solution. sex.act is comfortably interior and its box is unchanged.
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
#
# WALLTIME RAISED 24 h -> 48 h, 2026-08-07. burn_steps went 60 -> 720: 852 steps
# per rep against 192, a factor of 4.4, but burn steps skip report writing so the
# measured runtime penalty is 3.3x (trial_1.yml, 4 reps: 9.07 s at burn 60,
# 29.80 s at burn 720). Job 4743939 ran 2000 trials x 20 reps in 7 h 14 m at burn
# 60, so the same search is ~24 h -- right on the old wall. The study uses
# JournalStorage (commit 8357b2aa), so a timeout is recoverable by relaunching
# against the same journal rather than fatal, but 48 h avoids needing to.
#
# A pre-burned saved population does NOT help here: the equilibrium the burn-in
# reaches depends on the transmission parameters being searched, so it would
# have to be rebuilt per trial anyway.
#
# TRIALS CUT 2000 -> 1000 AND REPS RAISED 20 -> 40, 2026-08-10. Same 40,000 total
# model runs, so the walltime estimate above is unchanged; this is a reallocation,
# not an increase.
#
# WHY. The discrimination measurements above were taken at the OLD transmission
# level. The refit targets `acquisition ~0.35` equivalent (job 4779604), where HIV
# events are much rarer: the probe puts IDU-attributed diagnoses at 0.050 per
# model-year, i.e. ~0.55 events per rep over the 11-year scored window, or ~11
# pooled events at 20 reps. Poisson noise alone is then ~30% on the quantity
# `ridoh_hiv_err` is fitted against, and the HIV objective would rank substantially
# on sampling noise. [[titan-ri-transmission-feasibility-2026-08-05]] already
# recorded the rule this follows from: anything measured on incident HIV counts at
# a RIDOH-consistent level needs 40+ reps.
#
# The other four objectives are population-wide shares and were already clearing
# the discrimination threshold at 20 reps, so they lose nothing here. HIV is the
# objective this entire refit exists to fix, so it is the one to spend reps on.
export TITAN_CALIB_TRIALS=1000
export TITAN_CALIB_REPS=40

# NSGA-II population halved with the trial budget so the generation count is
# preserved: 1000/50 = 20 generations, the same depth 2000/100 gave. Leaving the
# default 100 would have given 10 generations, which is not enough for the front
# to converge.
export TITAN_CALIB_POP=50

# ---------------------------------------------------------------------------
# RUN OF 2026-08-11: same budget, rebased search box
# ---------------------------------------------------------------------------
# The trial/rep/population settings above are unchanged from job 4830724 on
# purpose -- what changed is WHERE they are spent. calibration_targets.yml's
# parameters_to_optimize bounds were narrowed to the HIV/overdose intersection
# box (see the long comment at the head of that list). Job 4830724 cleared the
# HIV gate for the first time but returned zero candidates clearing HIV and
# overdose together, while corr(od_err, hiv_err) = -0.019 and the two passing
# regions overlap in all 18 dimensions -- an unsampled corner, not a conflict.
#
# The new box is 3766x smaller than what 4830724 searched, so the same 1000
# trials buy far more resolution inside the region that matters. Only 7 of that
# job's 1000 trials fall inside it, and none of its best candidates do, so this
# is genuinely new ground rather than a re-run.
#
# Expect the same ~26 h. Two changes are also merged since 4830724: the
# canonical-pick null screen (without which the automated pick drifts to a
# near-null-HIV trial, as it did on 4743939 and 4830724), and the injection
# recency mechanism, merged INERT at its 999-year default.
#
# READ THE RESULT AGAINST validate_candidates.OBJECTIVES, not eyeballed
# thresholds: OUD 0.05, RIDOH HIV 1.1, overdose 0.01, MOUD 0.10, injection
# share 0.0025. If this run again returns zero 5-of-5 candidates, that is the
# structural result and the response is a targets conversation, not a third
# rebound of the box.
python titan/calibration/optuna_calibration.py

echo "Job finished at $(date)"
