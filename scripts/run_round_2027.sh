#!/bin/bash
# Strategy round for the 2027-2036 analysis at N agents on the cohort-footing members (2026-10-24, work record
# §19.40): every arm for every member, 8 populations x 40 CRN replicates, staged by analysis/chain_sensitivity_2026-10-12.py
# and run by validate_candidates from a FROZEN worktree; the contrasts (placebo window and strategy window) post after
# all shards. Differences from run_chain_sensitivity_2027.sh: members and their directory come from the environment,
# the code runs from the worktree, and the per-member statewide scale is the whole-cohort base (13,704 / N); the
# contrast divides it by each member's insured share itself (phase7_contrast.insured_scale).
# Usage: W=<worktree> TITAN_MEMBERS_DIR=<dir of trial_<m>.yml> TITAN_MEMBERS_FILE=<ids> \
#        [REPS=<replicates per population, 40>] [CPUS=<cores per shard, 16>] [MEM=<memory per shard, 64G>] [HOURS_PER_FILE=<1.2>] \
#        scripts/run_round_2027.sh <name> [arms] [shards] [N] [root]
set -euo pipefail
NAME=$1; ARMS=${2:-pc1,pc2,pc3,pc4,pc6,pc2b,pc4b}; SHARDS=${3:-16}; N=${4:-2000}
ROOT=${5:-/oscar/scratch/jtao9/titan_phase7/round2027_2026-10-24}
: "${W:?set W to the frozen worktree}"; : "${TITAN_MEMBERS_DIR:?}"; : "${TITAN_MEMBERS_FILE:?}"
S=$ROOT/n${N}_$NAME; mkdir -p $S/logs
cd $W
python analysis/chain_sensitivity_2026-10-12.py $ROOT "n${N}+$NAME" $ARMS
SCALE=$(python -c "import sys; sys.path.insert(0,'scripts'); from phase7_contrast import pop_scale; print(round(pop_scale()*500/$N, 6))")
MS=$(awk -v sc="$SCALE" 'NF{printf "%s%s=%s", (n++?",":""), $1, sc}' $S/members.txt)
echo "statewide base scale per member (whole cohort, before the insured share): $SCALE"
IFS=',' read -r -a arr <<< "$(cat $S/trials.txt)"
per=$(( (${#arr[@]} + SHARDS - 1) / SHARDS )); : > $S/jobs.txt
for ((b=0; b<SHARDS; b++)); do
  chunk=("${arr[@]:b*per:per}"); [ "${#chunk[@]}" -eq 0 ] && continue
  list=$(IFS=,; echo "${chunk[*]}")
  # ~1.2 h per trial file of 320 runs at 2,000 agents x 300 steps on 16 cores (measured 9-14 h for 12 files), plus 4 h
  # margin; HOURS_PER_FILE overrides it (the validator runs populations serially and only replicates in parallel, so a
  # file of 8 populations x 2 replicates at 14,704 agents takes 8 x 27 min on 2 cores whatever the core count).
  hours=$(awk -v n=${#chunk[@]} -v h=${HOURS_PER_FILE:-1.2} 'BEGIN{x=n*h+4; printf "%d", (x==int(x))?x:int(x)+1}')
  [ $hours -gt 48 ] && hours=48
  export TITAN_VAL_TRIALS="$list" TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=${REPS:-40} TITAN_VAL_CRN_BASE=700000 TITAN_VAL_TRIAL_DIR=$S/trials TITAN_VAL_OUT=$S/out
  jid=$(sbatch --parsable --job-name=r2k_${NAME}_$b --partition=batch --cpus-per-task=${CPUS:-16} --mem=${MEM:-64G} --time=${hours}:00:00 \
        --output=$S/logs/shard_${b}_%j.out --error=$S/logs/shard_${b}_%j.err --export=ALL \
        --wrap="module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate; cd $W; python -m titan.calibration.validate_candidates > $S/logs/validate_$b.txt 2>&1; echo validate exit \$?" 2>>$S/logs/sbatch.err) || { echo "sbatch failed for shard $b"; scancel $(cat $S/jobs.txt) 2>/dev/null; exit 1; }
  echo $jid >> $S/jobs.txt
done
DEP=$(paste -sd: $S/jobs.txt)
T0=$(python -c "import sys; sys.path.insert(0,'scripts'); from make_phase7_pc_arms import DEFAULT_START_STEP as d; print(d)")
[ "$T0" -eq 181 ] || { echo "DEFAULT_START_STEP is $T0, not 181"; exit 1; }
cat > $S/post.sbatch <<EOP
#!/bin/bash
#SBATCH -J post_r2k_$NAME
#SBATCH -t 08:00:00
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -o $S/logs/post_%j.out
#SBATCH -e $S/logs/post_%j.err
module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate
cd $W
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-max 180 --split-members --workers 8 --member-scale $MS --out $S/contrasts_pre_2027.csv > $S/contrast_pre_stdout.txt 2>&1; echo "pre contrast exit \$?"
python scripts/phase7_contrast.py $S/out --baseline scenario_1_baseline --t-min 181 --split-members --workers 8 --member-scale $MS --out $S/contrasts_2027_2036.csv > $S/contrast_post_stdout.txt 2>&1; echo "post contrast exit \$?"
python analysis/yearly_contrasts_by_arm.py $S/out $S/members.txt 300 $S/yearly_by_arm.csv > $S/yearly_stdout.txt 2>&1; echo "yearly exit \$?"
EOP
P=$(sbatch --parsable --dependency=afterok:$DEP $S/post.sbatch 2>>$S/logs/sbatch.err) || { echo "post submission failed; cancelling shards"; scancel $(cat $S/jobs.txt); exit 1; }; echo $P > $S/post_job.txt
echo "n${N}_$NAME: ${#arr[@]} trial files, $(wc -l < $S/jobs.txt) shards ($(paste -sd, $S/jobs.txt)); post $P"
