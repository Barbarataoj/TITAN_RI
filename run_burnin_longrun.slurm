#!/bin/bash
#SBATCH --job-name=titan_burnin_long
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/burnin_long_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/burnin_long_%j.err
#
# Does this model have a long-run HIV steady state at all, and if so at what
# transmission level?
#
# WHY THIS RUN EXISTS. Job 4770370 swept burn_steps to 360 with a consistent
# intervention regime and found HIV prevalence still climbing, with the scored
# window's HIV output scaling 2.9x across the sweep. The response to that cannot
# be to pin burn_steps at 60 and call it an initial condition: if the model has no
# long-run steady state then the fitted transmission parameters describe a
# transient, every scenario contrast inherits the drift, and the projection horizon
# is not defensible. What the burn-in does after step 60 has to be known, not
# assumed away.
#
# TWO DIMENSIONS, because a one-dimensional answer would be confounded. Trial
# 1267's transmission parameters are stale -- they were fitted against a syringe
# services block that suppressed the injection route 13x, and with that block
# re-anchored they overshoot RIDOH by 3-4x. An epidemic held well above its
# threshold grows until it runs out of susceptibles, so "no steady state at burn
# 1440" measured only at the hot setting would say nothing about whether the model
# is structurally sound at a RIDOH-consistent level.
#
#   burn_steps            60, 240, 720, 1440   (5, 20, 60 and 120 model years)
#   calibration.acquisition   1.0, 0.5, 0.35
#
# acquisition is used here purely as a probe knob. It left the search on
# 2026-08-06 as a redundant dimension and is pinned at 1.0 in every scenario, but
# it still multiplies the per-act probability of both routes in
# titan/exposures/hiv.py, so it moves the transmission level without touching the
# route mix -- the same use probe_transmission_feasibility makes of it. 0.5 is the
# region that probe identified as RIDOH-consistent on 2026-08-05; 0.35 brackets it
# from below.
#
# WHAT EACH OUTCOME MEANS.
#   * Prevalence flattens at some burn length at acquisition 0.5 or 0.35 but not
#     at 1.0 -> the model is structurally fine and the missing steady state is a
#     level artefact of the stale fit. Refit, then set burn_steps from where the
#     curve actually flattens at the fitted level.
#   * Prevalence flattens nowhere, at any level -> there is no endemic equilibrium
#     to calibrate to, and the burn-in cannot be made to absorb the initial
#     conditions. That is a structural finding about the model, and the fix is in
#     the demography or the exit/enter machinery, not in the calibration.
#   * Prevalence collapses to zero at 0.35 but grows at 1.0 -> the threshold sits
#     between them, and the RIDOH target may be reachable only near the critical
#     point, where the fit is inherently ill-conditioned. Worth knowing before
#     spending another 2000-trial calibration on it.
#
# Also read the injection share of active OUD, which job 4770370 showed is a
# second unconverged transient (0.3802 -> 0.5117 from burn 60 to 360, against a
# 0.45 target). It does not depend on acquisition, so all three arms measure it;
# if it flattens by 720 or 1440 that settles what the OUD-transition parameters
# should be fitted against.
#
# 3 acquisitions x 4 burns x 2 seeds = 24 configs at 20 reps, 8 at a time. Step
# counts per config are 192/372/852/1572 against the 192-1332 of job 4770370's
# 5m21s sweep, so expect roughly 30-45 minutes. 8 h is margin for the 1440 arm.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

PROBE_DIR="/oscar/scratch/${USER}/titan_calibration/burnin_probe"

for ACQ in 1.0 0.5 0.35; do
    echo ""
    echo "################ acquisition = ${ACQ} ################"
    python -m titan.calibration.probe_burnin_length \
        --scenario "${PROBE_DIR}/scen_acq${ACQ}.yml" \
        --burns "${PROBE_BURNS:-60,240,720,1440}" \
        --seeds "${PROBE_SEEDS:-2}" \
        --reps "${PROBE_REPS:-20}" \
        --workers "${PROBE_WORKERS:-8}"
    # probe_burnin_length always writes burnin_probe.csv to the same path, so keep
    # each arm before the next one overwrites it.
    mv "${PROBE_DIR}/burnin_probe.csv" "${PROBE_DIR}/burnin_long_acq${ACQ}.csv"
    echo "saved ${PROBE_DIR}/burnin_long_acq${ACQ}.csv"
done

echo ""
echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Outputs: ${PROBE_DIR}/burnin_long_acq{1.0,0.5,0.35}.csv"
