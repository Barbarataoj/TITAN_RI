#!/bin/bash
# Sensitivity ensembles for the 2027-2036 analysis at 2,000 agents (2026-10-20).
# Usage: run_chain_sensitivity_2027.sh <variant> [arms] [shards] [post_dependency_job]
#   variant   one of low, high, combo, art_ri, dxinj0, ehr_reminder_only, moud_decline (staged as n2000+<variant>)
#   arms      comma list of arm keys (default pc2,pc4,pc3)
#   shards    number of validator shards (default 8; ~6 scenario files each at 2,000 agents x 300 steps ~ 6 h)
#   post_dependency_job   optional Slurm job id the shards wait for (afterok), e.g. the primary round's post
# Windows: --t-max 180 (placebo, 2012-2026) and --t-min 181 (strategy window, 2027-2036); statewide scale
# 12.4575 = 49.83 x 500 / 2,000 passed per member so that diff_mean_statewide is on the 2,000-agent footing.
set -euo pipefail
NAME=$1; ARMS=${2:-pc2,pc4,pc3}; SHARDS=${3:-8}; AFTER=${4:-}
ROOT=/oscar/scratch/jtao9/titan_phase7/round2027_2026-10-19; S=$ROOT/n2000_$NAME; mkdir -p $S/logs
cd /oscar/data/jtao9/TITAN_RI
python analysis/chain_sensitivity_2026-10-12.py $ROOT "n2000+$NAME" $ARMS
MS=$(awk 'NF{printf "%s%s=12.4575", (n++?",":""), $1}' $S/members.txt)
IFS=',' read -r -a arr <<< "$(cat $S/trials.txt)"
per=$(( (${#arr[@]} + SHARDS - 1) / SHARDS )); : > $S/jobs.txt
DEPOPT=""; [ -n "$AFTER" ] && DEPOPT="--dependency=afterok:$AFTER"
for ((b=0; b<SHARDS; b++)); do
  chunk=("${arr[@]:b*per:per}"); [ "${#chunk[@]}" -eq 0 ] && continue
  list=$(IFS=,; echo "${chunk[*]}")
  export TITAN_VAL_TRIALS="$list" TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=40 TITAN_VAL_CRN_BASE=700000 TITAN_VAL_TRIAL_DIR=$S/trials TITAN_VAL_OUT=$S/out
  jid=$(sbatch --parsable $DEPOPT --job-name=s2k_${NAME}_$b --partition=batch --cpus-per-task=16 --mem=64G --time=14:00:00 \
        --output=$S/logs/shard_${b}_%j.out --error=$S/logs/shard_${b}_%j.err --export=ALL \
        --wrap="module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate; cd /oscar/data/jtao9/TITAN_RI; python -m titan.calibration.validate_candidates > $S/logs/validate_$b.txt 2>&1; echo validate exit \$?" 2>/dev/null)
  echo $jid >> $S/jobs.txt
done
DEP=$(paste -sd: $S/jobs.txt)
cat > $S/post.sbatch <<EOP
#!/bin/bash
#SBATCH -J post_s2k_$NAME
#SBATCH -t 03:00:00
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -o $S/logs/post_%j.out
#SBATCH -e $S/logs/post_%j.err
module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-max 180 --split-members --workers 8 --member-scale $MS --out $S/contrasts_pre_2027.csv > $S/contrast_pre_stdout.txt 2>&1; echo "pre contrast exit \$?"
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-min 181 --split-members --workers 8 --member-scale $MS --out $S/contrasts_2027_2036.csv > $S/contrast_post_stdout.txt 2>&1; echo "post contrast exit \$?"
EOP
P=$(sbatch --parsable --dependency=afterok:$DEP $S/post.sbatch 2>/dev/null); echo $P > $S/post_job.txt
echo "n2000_$NAME: ${#arr[@]} trials, shards $(paste -sd, $S/jobs.txt); post $P"
