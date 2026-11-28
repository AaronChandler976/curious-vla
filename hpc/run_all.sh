#!/bin/bash
# One command for the rest of the experiment, run on the HPC3 login node:
#   ssh hpc3 'cd ~/curious-vla && bash hpc/run_all.sh'
# 1. The dry run (every code path, tiny, on CPU). Nothing is submitted unless it passes.
# 2. A check of the demos-only runs that were submitted before build_dataset.py existed: they
#    train on the demo dataset directly (--dataset.episodes). verify_training_data.py --direct
#    loads each run's episodes exactly as lerobot-train does and compares the frames with the
#    raw rows.
#      - all OK:      those runs are kept as they are.
#      - a mismatch:  their jobs are cancelled, their outputs moved to runs/invalid/, and all
#                     demos-only runs (n=10 and 25 with seeds 0-2, n=50 with seed 0) are
#                     resubmitted on the verified data path.
#      - an error:    stop and change nothing (the check itself failed; paste the output).
# 3. The exploration collections and the mixed fine-tunes (submit_exploration.sh).
# 4. Cleanup of finished runs (keeps the final and 10k-step weights).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh

echo "=== 1. dry run"
SKIP_EVALS=1 bash hpc/dry_run.sh || { echo "STOPPED: the dry run failed; nothing was submitted."; exit 1; }

echo
echo "=== 2. demos-only runs trained on the demo dataset directly"
activate_env >/dev/null || exit 1
DATA_ROOT=$(dataset_root) || exit 1
mkdir -p runs/logs
status=ok
# The dataset loader caches the selected rows; keep that cache out of the nearly full home cache.
export HF_DATASETS_CACHE="$RUNS_DIR/dryrun/hf_datasets_cache"
shopt -s nullglob
sels=("$RUNS_DIR"/selections/demos_*.json)
if (( ${#sels[@]} == 0 )); then echo "no earlier demos-only selections found"; fi
for sel in "${sels[@]}"; do
  run=$(basename "$sel" .json)
  log="$RUNS_DIR/dryrun/direct_$run.log"
  policy=()
  [[ -f "$RUNS_DIR/base_policy/$run/config.json" ]] && policy=(--policy-dir "$RUNS_DIR/base_policy/$run")
  python hpc/verify_training_data.py --root "$DATA_ROOT" --repo "$DEMO_DATASET" --selection "$sel" \
    --direct "${policy[@]}" >"$log" 2>&1
  if grep -q "^TRAINING DATA OK" "$log"; then
    echo "  $run: OK ($(grep '^checked' "$log"))"
  elif grep -q "^TRAINING DATA MISMATCH" "$log"; then
    echo "  $run: MISMATCH"; grep -A4 "^TRAINING DATA MISMATCH" "$log" | sed 's/^/      /'
    status=mismatch
  else
    echo "  $run: the check failed to run. Last lines of $log:"; tail -n 25 "$log" | sed 's/^/      /'
    echo "STOPPED: nothing was cancelled or submitted."; exit 1
  fi
done

rm -rf "$HF_DATASETS_CACHE"; unset HF_DATASETS_CACHE

if [[ "$status" == mismatch ]]; then
  echo
  echo "The earlier demos-only runs did not train on the right frames: cancelling and resubmitting them."
  # No mixed fine-tunes exist yet at this point, so every queued fine-tune is an old demos-only one.
  jobs=$(squeue -u "${USER:-$(whoami)}" -h -o "%i %j" | awk '$2 == "cvla-ft-demos" || $2 ~ /^demos_/ {print $1}')
  jobs=$(echo $jobs)
  if [[ -n "$jobs" ]]; then echo "  scancel $jobs"; scancel $jobs; sleep 10; fi
  stamp=$(date +%Y%m%d-%H%M)
  mkdir -p "$RUNS_DIR/invalid/$stamp"
  for d in "$RUNS_DIR"/demos_*; do mv "$d" "$RUNS_DIR/invalid/$stamp/"; done
  for f in "$RUNS_DIR"/selections/demos_*.json; do mv "$f" "$RUNS_DIR/invalid/$stamp/"; done
  echo "  old outputs moved to $RUNS_DIR/invalid/$stamp (delete them once you no longer need them)"
  SKIP_CHECK=1 SKIP_REF=1 DEMO_BUDGETS="10 25" SEEDS="0 1 2" bash hpc/submit_baseline.sh | grep "fine-tune"
  SKIP_CHECK=1 SKIP_REF=1 DEMO_BUDGETS="50" SEEDS="0" bash hpc/submit_baseline.sh | grep "fine-tune"
else
  echo "  earlier demos-only runs are valid; leaving them as they are"
fi

echo
echo "=== 3. exploration collections and mixed fine-tunes"
bash hpc/submit_exploration.sh || { echo "STOPPED: submitting the exploration jobs failed."; exit 1; }

echo
echo "=== 4. cleanup of finished runs"
for r in "$RUNS_DIR"/*/eval/eval_info.json; do bash hpc/cleanup_run.sh "${r%/eval/eval_info.json}"; done
echo
squeue -u "${USER:-$(whoami)}" -o "%.8i %.40j %.8T %.10M %R"
echo "ALL SUBMITTED"
