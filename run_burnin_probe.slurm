#!/bin/bash
#SBATCH --job-name=titan_burnin_probe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/burnin_probe_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/burnin_probe_%j.err
#
# Sweeps burn_steps and asks whether each quantity has stopped moving by the time
# scoring starts. Run AFTER the 2026-08-07 intervention re-anchoring and BEFORE
# committing to a refit.
#
# WHY THIS RUN EXISTS. The 2026-08-04 burn-in probe concluded that HIV prevalence
# never equilibrates. That conclusion was confounded: syringe services were
# configured with start_time 1, and burn-in runs at negative model time, so the
# burn-in and the scored window were running different intervention regimes. HIV
# prevalence was not failing to converge, it was converging to one level and then
# being scored against another. Measured on job 4743939's canonical trial,
# injection-route HIV incidence per 100 PWID-years was 0.111 during burn-in and
# 0.017 across the scored window.
#
# The re-anchoring moved syringe services to start_time -999 (ENCORE has operated
# continuously since 1994, so it covers the burn-in as well as the window) and set
# coverage and risk from NHBS-PWID 2018. On a 10-rep smoke run the same two
# numbers came back 0.126 and 0.120, so the discontinuity is gone and the question
# "does the burn-in converge?" can finally be asked cleanly.
#
# PrEP is deliberately still absent from the burn-in. It was not approved until
# July 2012, so its regime difference across t=0 is real rather than an artefact.
#
# WHAT WOULD CHANGE THE PLAN. If PWID HIV prevalence flattens by 120 or 240 steps,
# the burn-in length is simply too short and the fix is to lengthen it -- which
# would also let the NonInj reservoir build endogenously and retire
# calibration.noninj_hiv_init_het, the parameter that drifted to its search floor
# on job 4743939 because no objective constrains it. If prevalence is still moving
# at 360 steps even with a consistent intervention regime, there is no endemic
# equilibrium at these transmission parameters and the seed has to stay, in which
# case it needs to be pinned rather than fitted.
#
# The scenario is trial 1267's parameter vector with the re-anchored intervention
# layer applied, built by hand on 2026-08-07. Its transmission parameters are
# STALE -- they were fitted against the old syringe-services block and overshoot
# RIDOH by 3-4x once the injection route is live -- so read the SHAPE of the
# burn-in curves here, not their levels.
#
# 4 burn lengths x 3 population seeds = 12 configs at 20 reps each, 8 at a time.
# A 10-rep 60-step-burn run takes ~27 s, and a 360-step burn is ~2.6x that per
# rep, so the sweep is roughly 15 minutes. 4 h is margin.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

python -m titan.calibration.probe_burnin_length \
    --scenario "${PROBE_SCENARIO:-/oscar/scratch/${USER}/titan_calibration/burnin_probe/scenario_trial1267_fixed.yml}" \
    --burns "${PROBE_BURNS:-60,120,240,360}" \
    --seeds "${PROBE_SEEDS:-3}" \
    --reps "${PROBE_REPS:-20}" \
    --workers "${PROBE_WORKERS:-8}" \
    --figure "${PROBE_FIGURE:-/oscar/scratch/${USER}/titan_calibration/burnin_probe/burnin_probe_2026-08-07.png}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Output: /oscar/scratch/${USER}/titan_calibration/burnin_probe/burnin_probe.csv"
