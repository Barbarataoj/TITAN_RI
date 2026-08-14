#!/bin/bash
#SBATCH --job-name=titan_adhoc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/titan_calibration/adhoc_%j.out
#SBATCH --error=/oscar/scratch/%u/titan_calibration/adhoc_%j.err

# Generic compute-node wrapper for one-off analysis and probe scripts.
#
#   sbatch scripts/run_on_node.slurm python analysis/some_probe.py
#   sbatch --cpus-per-task=1 --time=00:30:00 scripts/run_on_node.slurm python -m pytest tests/
#
# Anything after the script name is run verbatim, from the repository root, with
# the shared environment already active. Resource flags passed to sbatch override
# the defaults above.
#
# WHY THIS EXISTS: CCV has issued usage warnings for running compute on the login
# and VS Code nodes. The offenders were not simulations -- they were "small"
# numpy analysis scripts that turned out to run for tens of minutes. There is no
# reliable way to tell in advance, so anything that loops belongs here. Reading a
# YAML, grepping, or a smoke test that builds one scenario dict is fine
# interactively; a parameter grid is not.
#
# For an interactive session on a compute node, use CCV's wrapper instead:
#
#   interact -n 4 -t 02:00:00 -m 16g
#
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/run_on_node.slurm <command> [args...]" >&2
    exit 2
fi

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Command: $*"

module purge
module load python/3.9

source /oscar/data/jtao9/envs/titan_env/bin/activate

cd /oscar/data/jtao9/TITAN_RI

"$@"

echo "Job finished at $(date)"
