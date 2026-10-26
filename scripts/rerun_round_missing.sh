#!/bin/bash
# Re-run the trial files of a 2027 strategy round that did not finish (2026-10-25, work record §19.55).
#
# A shard launched by scripts/run_round_2027.sh runs its trial files in sequence, eight populations each, and the validator
# has no resume: a shard that hits its time limit leaves every later trial file unrun and the current one partial. The
# marker of a finished trial file is its pooled report, $S/out/trial<T>_pooled_basicReport.txt, written only after all
# eight populations complete. This script lists the trial files without that marker and, unless --check is given,
# submits them as new shards (at most FILES_PER_SHARD files each) with the launcher's environment, then replaces the
# round's post job, whose afterok dependency on a timed-out shard can never be satisfied.
#
# Usage: W=<frozen worktree> [HOURS_PER_FILE=2] [FILES_PER_SHARD=2] [CPUS=16] [MEM=64G] [REPS=40] \
#        scripts/rerun_round_missing.sh <round dir S> [--check]
# Refuses to submit while any shard listed in $S/jobs.txt or $S/jobs_rerun.txt is still queued or running.
set -euo pipefail
S=${1:?round dir}; MODE=${2:-submit}
: "${W:?set W to the frozen worktree the round ran from}"
NAME=$(basename "$S" | sed 's/^n[0-9]*_//')
IFS=',' read -r -a all <<< "$(cat "$S/trials.txt")"
missing=()
for t in "${all[@]}"; do [ -f "$S/out/trial${t}_pooled_basicReport.txt" ] || missing+=("$t"); done
echo "$(date '+%F %T') $S: ${#all[@]} trial files, ${#missing[@]} without a pooled report"
[ "${#missing[@]}" -eq 0 ] && { echo "round complete; nothing to do"; exit 0; }
printf '  %s\n' "${missing[@]}"
[ "$MODE" = "--check" ] && exit 0
live=$(cat "$S/jobs.txt" "$S/jobs_rerun.txt" 2>/dev/null | paste -sd,)
if [ -n "$live" ] && [ -n "$(squeue -h -j "$live" 2>/dev/null)" ]; then
  echo "shards still queued or running ($(squeue -h -j "$live" -o %i | paste -sd,)); not submitting"; exit 2
fi
cd "$W"
echo "code: $W ($(git -C "$W" rev-parse --short HEAD)); titan import path: $(python -c 'import titan, os; print(os.path.dirname(titan.__file__))')"
per=${FILES_PER_SHARD:-2}; : > "$S/jobs_rerun.txt"; b=0
for ((i=0; i<${#missing[@]}; i+=per)); do
  chunk=("${missing[@]:i:per}"); list=$(IFS=,; echo "${chunk[*]}")
  hours=$(awk -v n=${#chunk[@]} -v h=${HOURS_PER_FILE:-2} 'BEGIN{x=n*h+4; printf "%d", (x==int(x))?x:int(x)+1}')
  export TITAN_VAL_TRIALS="$list" TITAN_VAL_SEEDS=8 TITAN_VAL_REPS=${REPS:-40} TITAN_VAL_CRN_BASE=700000 TITAN_VAL_TRIAL_DIR=$S/trials TITAN_VAL_OUT=$S/out
  jid=$(sbatch --parsable --job-name=rr_${NAME}_$b --partition=batch --cpus-per-task=${CPUS:-16} --mem=${MEM:-64G} --time=${hours}:00:00 \
        --output=$S/logs/rerun_${b}_%j.out --error=$S/logs/rerun_${b}_%j.err --export=ALL \
        --wrap="module purge; module load python/3.9; source /oscar/data/jtao9/envs/titan_env/bin/activate; cd $W; python -c 'import titan, os; print(\"titan import path:\", os.path.dirname(titan.__file__))'; python -m titan.calibration.validate_candidates > $S/logs/validate_rerun_$b.txt 2>&1; echo validate exit \$?")
  echo "$jid" >> "$S/jobs_rerun.txt"; echo "rerun shard $b -> job $jid: $list (${hours} h)"; b=$((b+1))
done
old=$(cat "$S/post_job.txt" 2>/dev/null | tail -1)
if [ -n "$old" ] && [ -n "$(squeue -h -j "$old" 2>/dev/null)" ]; then scancel "$old" && echo "cancelled post job $old (dependency on a failed shard)"; fi
P=$(sbatch --parsable --dependency=afterok:$(paste -sd: "$S/jobs_rerun.txt") "$S/post.sbatch"); echo "$P" >> "$S/post_job.txt"
echo "post job $P (afterok on the rerun shards)"
