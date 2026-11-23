#!/bin/bash
# Run every job's code path once, tiny and on CPU, on the HPC3 login node, with no internet.
# It catches setup errors in about 10 minutes instead of one queued GPU job at a time.
#   ssh hpc3 'cd ~/curious-vla && bash hpc/dry_run.sh'
# Steps: cache check, episode selection, a 5-step rollout of the reference policy, 2 training
# steps from smolvla_base with our camera rename map, and a 5-step rollout of that checkpoint.
# Logs go to runs/dryrun/. On a failure it prints the last 80 lines of that step's log.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
activate_env >/dev/null || exit 1
# The models run on CPU (--policy.device=cpu below). CUDA_VISIBLE_DEVICES is left alone:
# robosuite's EGL renderer parses it, and an empty value crashes it.
LOG="$RUNS_DIR/dryrun"
rm -rf "$LOG" && mkdir -p "$LOG"

step() {  # step NAME COMMAND...
  local name=$1; shift
  printf '%-28s' "$name"
  local t0=$SECONDS
  if "$@" >"$LOG/$name.log" 2>&1; then
    echo "OK ($((SECONDS - t0)) s)"
  else
    echo "FAILED ($((SECONDS - t0)) s). Last 80 lines of $LOG/$name.log:"
    echo "--------------------------------------------------------------------------------"
    tail -n 80 "$LOG/$name.log"
    echo "--------------------------------------------------------------------------------"
    echo "DRY RUN FAILED"
    exit 1
  fi
}

TASK0="${LIBERO_TASK_IDS%%,*}"
step 1-cache bash hpc/verify_cache.sh

DATA_ROOT=$(dataset_root) || exit 1
BASE_DIR=$(hf_local_model "$BASE_POLICY") || exit 1
REF_DIR=$(hf_local_model "$REFERENCE_POLICY") || exit 1

select_eps() {
  python hpc/select_episodes.py --dataset "$DEMO_DATASET" --root "$DATA_ROOT" --suite "$LIBERO_SUITE" \
    --task-ids "$LIBERO_TASK_IDS" --n-demos 1 --seed 0 | tail -n 1 > "$LOG/episodes.txt"
  grep -Eq '^\[[0-9,]+\]$' "$LOG/episodes.txt" || { echo "not a JSON list: $(cat "$LOG/episodes.txt")"; return 1; }
  cat "$LOG/episodes.txt"
}
step 2-select-episodes select_eps
EPISODES=$(cat "$LOG/episodes.txt")

step 3-eval-reference lerobot-eval \
  --policy.path="$REF_DIR" --policy.device=cpu \
  --env.type=libero --env.task="$LIBERO_SUITE" --env.task_ids="[$TASK0]" --env.episode_length=5 \
  --eval.n_episodes=1 --eval.batch_size=1 --output_dir="$LOG/eval_reference"

step 4-train-2-steps lerobot-train \
  --policy.path="$BASE_DIR" --policy.push_to_hub=false --policy.device=cpu \
  --rename_map="$FINETUNE_RENAME_MAP" \
  --dataset.repo_id="$DEMO_DATASET" --dataset.root="$DATA_ROOT" --dataset.episodes="$EPISODES" \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
  --output_dir="$LOG/train" --job_name=dryrun

step 5-eval-finetuned lerobot-eval \
  --policy.path="$LOG/train/checkpoints/last/pretrained_model" --policy.device=cpu \
  --rename_map="$FINETUNE_RENAME_MAP" \
  --env.type=libero --env.task="$LIBERO_SUITE" --env.task_ids="[$TASK0]" --env.episode_length=5 \
  --eval.n_episodes=1 --eval.batch_size=1 --output_dir="$LOG/eval_finetuned"

rm -rf "$LOG/train/checkpoints"         # ~2 GB, and /home is nearly full
echo "DRY RUN PASSED"
