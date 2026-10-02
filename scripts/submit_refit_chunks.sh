#!/bin/bash
# Submit a refit as a chain of resumable chunks so no job exceeds the 24 h wall.
#
# Usage:
#   scripts/submit_refit_chunks.sh <n_chunks> <trials_per_chunk> <tag> [after_job_id] [cpus] [hours]
# Example (1000 trials as 3 x 334 after the pilot 6937150 succeeds):
#   scripts/submit_refit_chunks.sh 3 334 care_refit 6937150 16 14
#
# Every chunk opens the same JournalStorage (TITAN_CALIB_JOURNAL) and study name,
# so Optuna resumes: completed trials are read back, warm starts are skipped if
# already present, and NSGA-II continues its generations. Each chunk runs
# <trials_per_chunk> NEW trials, uses its own sampler seed (base + chunk index,
# so chunk 2 does not replay chunk 1's random stream) and writes its canonical
# pick to scratch, never to the repo's calibrated_scenario_best.yml. Chunks are
# chained with --dependency=afterany so a chunk that times out still hands the
# journal on to the next one.
#
# Never run two studies at once against optuna_multi_runs/: trial directories
# are numbered per study from 0 and run_titan clears them on start, so
# concurrent studies destroy each other's runs (2026-10-01).
set -euo pipefail
n_chunks=${1:?n_chunks}; per=${2:?trials_per_chunk}; tag=${3:?tag}
after=${4:-}; cpus=${5:-16}; hours=${6:-14}
scratch=/oscar/scratch/${USER}/titan_calibration/optuna_multi_runs
journal=${scratch}/study_${tag}.journal
study=titan_ri_${tag}
base_seed=20260730
cd /oscar/data/jtao9/TITAN_RI
prev=${after}
ids=()
for k in $(seq 1 "$n_chunks"); do
  dep=""
  if [ -n "$prev" ]; then
    if [ "$k" -eq 1 ]; then dep="--dependency=afterok:${prev}"; else dep="--dependency=afterany:${prev}"; fi
  fi
  jid=$(sbatch --parsable $dep \
      --job-name="${tag}_c${k}" --cpus-per-task="$cpus" --time="${hours}:00:00" \
      --export=ALL,TITAN_CALIB_TRIALS="$per",TITAN_CALIB_JOURNAL="$journal",TITAN_CALIB_STUDY="$study",TITAN_CALIB_SEED=$((base_seed + k)),TITAN_CALIB_BEST_OUT="${scratch}/best_${tag}_chunk${k}.yml" \
      run_calibration.slurm)
  echo "chunk $k: job $jid (${per} trials, ${cpus} cpus, ${hours} h${dep:+, $dep})"
  ids+=("$jid"); prev=$jid
done
printf '%s\n' "${ids[@]}" > "${scratch}/chunks_${tag}.txt"
echo "journal: $journal"; echo "study:   $study"; echo "job ids: ${scratch}/chunks_${tag}.txt"
