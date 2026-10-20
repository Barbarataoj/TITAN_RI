#!/bin/bash
# Usage: run_sensitivity.sh <name> [arms] [shards]   (from the repo, titan_env active)
# DEPRECATED 2026-10-20: this launcher belongs to the 2023-2031 development-set rounds (strategies from step 133,
# 240 steps, outputs contrasts_2023_2031.csv, readout over 9 years). The 2027-2036 analysis uses
# scripts/run_chain_sensitivity_2027.sh. Since 2026-10-19 make_phase7_pc_arms.py starts strategies at step 181 and
# the builder stages 300 steps, so a run from this script would mix a 2027 start with 2023 output names; it refuses.
echo "run_chain_sensitivity.sh is deprecated for the 2027 analysis; use scripts/run_chain_sensitivity_2027.sh" >&2; exit 2
set -euo pipefail
NAME=$1; ARMS=${2:-pc2,pc4,pc3}; SHARDS=${3:-8}
ROOT=/oscar/scratch/jtao9/titan_phase7/sens_2026-10-12; S=$ROOT/$NAME; mkdir -p $S/logs $S/check
cd /oscar/data/jtao9/TITAN_RI
python analysis/chain_sensitivity_2026-10-12.py $ROOT $NAME $ARMS
IFS=',' read -r -a arr <<< "$(cat $S/trials.txt)"
per=$(( (${#arr[@]} + SHARDS - 1) / SHARDS )); : > $S/jobs.txt
for ((b=0; b<SHARDS; b++)); do
  chunk=("${arr[@]:b*per:per}"); [ "${#chunk[@]}" -eq 0 ] && continue
  list=$(IFS=,; echo "${chunk[*]}")
  export TITAN_VAL_TRIALS="$list" TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=40 TITAN_VAL_CRN_BASE=700000 TITAN_VAL_TRIAL_DIR=$S/trials TITAN_VAL_OUT=$S/out
  jid=$(sbatch --parsable --job-name=sens_${NAME}_$b --partition=batch --cpus-per-task=16 --mem=32G --time=04:00:00 \
        --output=$S/logs/shard_${b}_%j.out --error=$S/logs/shard_${b}_%j.err --export=ALL \
        --wrap="module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate; cd /oscar/data/jtao9/TITAN_RI; python -m titan.calibration.validate_candidates > $S/logs/validate_$b.txt 2>&1; echo validate exit \$?")
  echo $jid >> $S/jobs.txt
done
DEP=$(paste -sd: $S/jobs.txt)
cp /oscar/scratch/jtao9/titan_phase7/ensemble_chain4_2026-10-13/check/readout_base.py $S/check/
cp /oscar/scratch/jtao9/titan_phase7/ensemble_chain4_2026-10-13/check/readout.py $S/check/
cat > $S/post.sbatch <<EOP
#!/bin/bash
#SBATCH -J post_$NAME
#SBATCH -t 01:00:00
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -o $S/logs/post_%j.out
#SBATCH -e $S/logs/post_%j.err
module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate
cd /oscar/data/jtao9/TITAN_RI
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-max 132 --split-members --workers 8 --out $S/contrasts_pre_2023.csv > $S/contrast_pre_stdout.txt 2>&1; echo "pre contrast exit \$?"
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-min 133 --split-members --workers 8 --out $S/contrasts_2023_2031.csv > $S/contrast_post_stdout.txt 2>&1; echo "post contrast exit \$?"
python $S/check/readout.py $S /oscar/scratch/jtao9/titan_phase7/chain_fix_members_v3_2026-10-13; echo "readout exit \$?"
EOP
P=$(sbatch --parsable --dependency=afterok:$DEP $S/post.sbatch); echo $P > $S/post_job.txt
echo "$NAME: shards $(paste -sd, $S/jobs.txt); post $P"
