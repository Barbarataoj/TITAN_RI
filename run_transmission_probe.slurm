#!/bin/bash
#SBATCH --job-name=titan_trans_probe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/trans_probe_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/trans_probe_%j.err
#
# Sweeps calibration.acquisition and asks whether the RIDOH HIV target is
# reachable at all. Run BEFORE committing to a refit.
#
# The 2026-08-04 front sits 1.8-4.8x above RIDOH on all-route HIV diagnoses among
# PWID and 2.3-6.3x on the injection-attributed subset, and run_burnin_probe
# showed prevalence still climbing after 30 years of burn-in. Both point at
# transmission being several times too hot; neither shows whether a colder
# setting exists that RIDOH would accept.
#
# acquisition is the right lever because it scales all three routes at once
# (injection, network sex, background force of infection), so it moves the level
# without touching the route mix. Its search bound is [0.1, 3.0] and the fitted
# value is 2.0417, so the swept range is well inside the space the optimizer
# could already have reached.
#
# The three questions the report answers, all of which must answer well before a
# refit is worth launching, are documented in the module docstring.
#
# 5 acquisitions x 3 population seeds = 15 configs at 40 reps each, 8 at a time.
# ~2 min per config at 10 reps => roughly 15 min at 40. 4 h is margin.
#
# 40 reps rather than 10 because the quantities are rare events: at a
# RIDOH-consistent acquisition one rep yields about one diagnosis, and a 10-rep
# run leaves +/-13pp on a route share. Two 10-rep runs (jobs 4673148, 4673743)
# disagreed by exactly that. See the REPS note in the module docstring.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

python -m titan.calibration.probe_transmission_feasibility \
    --scenario "${PROBE_SCENARIO:-calibrated_scenario_best.yml}" \
    --acquisitions "${PROBE_ACQS:-2.04,1.0,0.5,0.35,0.25}" \
    --seeds "${PROBE_SEEDS:-3}" \
    --reps "${PROBE_REPS:-40}" \
    --workers "${PROBE_WORKERS:-8}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Output: /oscar/scratch/${USER}/titan_calibration/transmission_probe/transmission_probe.csv"
