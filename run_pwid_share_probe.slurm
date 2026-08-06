#!/bin/bash
#SBATCH --job-name=titan_pwid_share
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/pwid_share_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/pwid_share_%j.err
#
# 2-D sweep of calibration.noninj_hiv_init_het x calibration.acquisition, asking
# whether the model can put RIDOH's SHARE of the HIV epidemic in PWID (11.3%)
# rather than only its level.
#
# Motivated by job 4738184, which showed a global transmission scalar can hit
# RIDOH's PWID diagnosis counts OR RI's statewide total, never both: the model
# puts ~37% of its epidemic in PWID against RIDOH's 11.3%, and a one-degree-of-
# freedom scalar cannot change a ratio. noninj_hiv_init_het is the only fitted
# parameter that moves the PWID/non-PWID balance, and it is already at 0.0283 of
# a 0.04 bound, so whether it has the authority to close the gap decides whether
# the refit needs a parameter that does not exist yet.
#
# 5 hets x 3 acquisitions x 3 population seeds = 45 configs at 40 reps each, 8 at
# a time. The 15-config version of this took 6.5 min, so roughly 20 min. 6 h is
# margin.

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

python -m titan.calibration.probe_pwid_share \
    --scenario "${PROBE_SCENARIO:-calibrated_scenario_best.yml}" \
    --hets "${PROBE_HETS:-0.0,0.01,0.02,0.028,0.04}" \
    --acquisitions "${PROBE_ACQS:-1.0,0.5,0.35}" \
    --seeds "${PROBE_SEEDS:-3}" \
    --reps "${PROBE_REPS:-40}" \
    --workers "${PROBE_WORKERS:-8}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Output: /oscar/scratch/${USER}/titan_calibration/pwid_share_probe/pwid_share_probe.csv"
