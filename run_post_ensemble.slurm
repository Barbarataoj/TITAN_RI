#!/bin/bash
#SBATCH --job-name=post_ensemble
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/titan_phase7/post_ensemble_%j.out
# After a Phase 7 ensemble: pooled contrasts (2023-2031 intervention window and the
# full 2012-2031 run) split by member, the cash-flow ledger, and copies of the
# small result files into the repository archive and the persistent archive.
# Usage: sbatch --dependency=afterany:<batch ids> --export=ALL,ENS=<ensemble dir>,TAG=<tag> run_post_ensemble.slurm
set -uo pipefail
module purge; module load python/3.9
source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI
ENS=${ENS:?}; TAG=${TAG:?}
ARCH=/oscar/data/jtao9/TITAN_RI/calibration_archive/phase7_${TAG}
mkdir -p ${ARCH}
echo "== completed scenario batches:"; ls -d ${ENS}/out/*_seed* 2>/dev/null | wc -l
python scripts/phase7_contrast.py ${ENS}/out --baseline scenario_1_baseline --t-min 133 --split-members \
    --workers 8 --out ${ENS}/contrasts_2023_2031.csv > ${ENS}/contrast_2023_2031_stdout.txt 2>&1; echo "contrast 2023-2031 exit $?"
python scripts/phase7_contrast.py ${ENS}/out --baseline scenario_1_baseline --split-members \
    --workers 8 --out ${ENS}/contrasts_2012_2031.csv > ${ENS}/contrast_2012_2031_stdout.txt 2>&1; echo "contrast 2012-2031 exit $?"
python scripts/phase7_cashflow.py --contrasts ${ENS}/contrasts_2023_2031.csv --costs docs/phase7_unit_costs.yml \
    --out ${ENS}/cashflow_2023_2031.csv > ${ENS}/cashflow_stdout.txt 2>&1; echo "cashflow exit $?"
python scripts/rebuild_validation_summary.py --out-dir ${ENS}/out --csv ${ENS}/validation_summary_rebuilt.csv --full \
    --workers 8 > ${ENS}/rebuild_stdout.txt 2>&1; echo "rebuild exit $?"
cp -p ${ENS}/contrasts_*.csv ${ENS}/cashflow_2023_2031.csv ${ENS}/*_stdout.txt ${ENS}/arithmetic.txt ${ENS}/members.txt ${ENS}/jobs.txt ${ARCH}/ 2>/dev/null
[ -f ${ENS}/validation_summary_rebuilt.csv ] && cp -p ${ENS}/validation_summary_rebuilt.csv ${ARCH}/
echo "archived to ${ARCH}:"; ls ${ARCH}
head -60 ${ENS}/contrast_2023_2031_stdout.txt
