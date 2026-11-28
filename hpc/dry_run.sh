#!/bin/bash
# Run every job's code path once, tiny and on CPU, on the HPC3 login node, with no internet.
# It catches setup errors in about 10 minutes instead of one queued GPU job at a time.
#   ssh hpc3 'cd ~/curious-vla && bash hpc/dry_run.sh'
# Steps: cache check, episode selection, a 5-step rollout of the reference policy, then the
# demos-only path (build the training dataset, verify its frames against the raw rows, 2 training
# steps, the feature check against the official checkpoint, a 5-step rollout of the result), then
# the exploration path (tiny random and ICM collections, the format check, the mixed dataset, its
# verification and 2 training steps on it).
# SKIP_EVALS=1 (or ONLY_EXPLORATION=1) skips the two rollouts (steps 3 and 5) once they have passed.
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
REF_DIR=$(hf_local_model "$REFERENCE_POLICY") || exit 1
BASE_DIR="$LOG/base_policy"
step 1b-base-policy python hpc/make_base_policy.py "$(hf_local_model "$BASE_POLICY")" "$REF_DIR" "$DATA_ROOT" "$BASE_DIR"

select_eps() {
  python hpc/select_episodes.py --dataset "$DEMO_DATASET" --root "$DATA_ROOT" --suite "$LIBERO_SUITE" \
    --task-ids "$LIBERO_TASK_IDS" --n-demos 1 --seed 0 --out "$LOG/selection.json" | tail -n 1 > "$LOG/episodes.txt"
  grep -Eq '^\[[0-9,]+\]$' "$LOG/episodes.txt" || { echo "not a JSON list: $(cat "$LOG/episodes.txt")"; return 1; }
  cat "$LOG/episodes.txt"
}
step 2-select-episodes select_eps
EPISODES=$(cat "$LOG/episodes.txt")

SKIP_EVALS="${SKIP_EVALS:-${ONLY_EXPLORATION:-0}}"
if [[ "$SKIP_EVALS" != "1" ]]; then
step 3-eval-reference lerobot-eval \
  --policy.path="$REF_DIR" --policy.device=cpu \
  --env.type=libero --env.task="$LIBERO_SUITE" --env.task_ids="[$TASK0]" --env.episode_length=5 \
  --eval.n_episodes=1 --eval.batch_size=1 --output_dir="$LOG/eval_reference"
fi

show() { sed "s/^/    /" "$LOG/$1.log" | grep -v -i -E "warn|torchcodec|it/s\]|examples/s" | tail -n "${2:-20}"; }

# Demos only: the same build + verify + train path as 02_finetune_demos.sbatch.
step 4-build-demos python hpc/build_dataset.py --demo-root "$DATA_ROOT" --selection "$LOG/selection.json" \
  --out "$LOG/data_demos"
step 4a-verify-demos python hpc/verify_training_data.py --root "$LOG/data_demos" \
  --selection "$LOG/selection.json" --policy-dir "$BASE_DIR"
show 4a-verify-demos 3
step 4b-train-demos-2-steps lerobot-train \
  --policy.path="$BASE_DIR" --policy.push_to_hub=false --policy.device=cpu \
  --dataset.repo_id=local/data_demos --dataset.root="$LOG/data_demos" \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
  --output_dir="$LOG/train" --job_name=dryrun

check_features() {
  python - "$LOG/train/checkpoints/last/pretrained_model/config.json" "$REF_DIR/config.json" <<'PYEOF'
import json, sys
def feats(p):
    c = json.load(open(p))
    return {k: (v["type"], list(v["shape"])) for k, v in {**c["input_features"], **c["output_features"]}.items()}
got, want = feats(sys.argv[1]), feats(sys.argv[2])
for k in sorted(set(got) | set(want)):
    print(f"{'OK ' if got.get(k) == want.get(k) else 'BAD'} {k}: trained {got.get(k)}  official {want.get(k)}")
sys.exit(0 if got == want else "trained checkpoint features differ from the official LIBERO checkpoint")
PYEOF
}
step 4c-check-features check_features
show 4c-check-features

if [[ "$SKIP_EVALS" != "1" ]]; then
step 5-eval-finetuned lerobot-eval \
  --policy.path="$LOG/train/checkpoints/last/pretrained_model" --policy.device=cpu \
  --env.type=libero --env.task="$LIBERO_SUITE" --env.task_ids="[$TASK0]" --env.episode_length=5 \
  --eval.n_episodes=1 --eval.batch_size=1 --output_dir="$LOG/eval_finetuned"
fi
rm -rf "$LOG/train/checkpoints"         # ~2 GB, and /home is nearly full

# Exploration conditions: tiny collections (2 episodes of 30 steps each), the format check
# against the demos, building the mixed dataset, and 2 training steps on it.
step 5b-pick-render python hpc/pick_egl_device.py --demo-root "$DATA_ROOT" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS"
show 5b-pick-render
for m in random icm; do
  step 6-collect-$m python hpc/collect_exploration.py --method $m --suite "$LIBERO_SUITE" \
    --task-ids "$LIBERO_TASK_IDS" --steps 60 --episode-len 30 --rollout 4 \
    --demo-root "$DATA_ROOT" --out "$LOG/exploration_$m"
done
step 7-format-check python hpc/check_exploration_format.py --demo-repo "$DEMO_DATASET" --demo-root "$DATA_ROOT" \
  --explore-root "$LOG/exploration_icm" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS"
show 7-format-check
step 8-build-mix python hpc/build_dataset.py --demo-root "$DATA_ROOT" --selection "$LOG/selection.json" \
  --explore-root "$LOG/exploration_icm" --budget 60 --out "$LOG/data_mix"
show 8-build-mix 1
step 8a-verify-mix python hpc/verify_training_data.py --root "$LOG/data_mix" \
  --selection "$LOG/selection.json" --policy-dir "$BASE_DIR"
show 8a-verify-mix 3
step 9-train-mix-2-steps lerobot-train \
  --policy.path="$BASE_DIR" --policy.push_to_hub=false --policy.device=cpu \
  --dataset.repo_id=local/data_mix --dataset.root="$LOG/data_mix" \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
  --output_dir="$LOG/train_mix" --job_name=dryrun_mix
# ICM with a warm-up (icmw), and the two-stage condition's stage 1: exploration frames alone,
# 2 training steps, then 2 demo-only steps that start from the stage-1 weights.
step 10-collect-icmw python hpc/collect_exploration.py --method icmw --warmup-steps 60 --suite "$LIBERO_SUITE" \
  --task-ids "$LIBERO_TASK_IDS" --steps 60 --episode-len 30 --rollout 4 \
  --demo-root "$DATA_ROOT" --out "$LOG/exploration_icmw"
step 10a-build-explore-only python hpc/build_dataset.py --demo-root "$DATA_ROOT" --selection "$LOG/selection.json" \
  --explore-root "$LOG/exploration_icmw" --budget 60 --explore-only --out "$LOG/data_stage1"
step 10b-verify-explore-only python hpc/verify_training_data.py --root "$LOG/data_stage1" \
  --selection "$LOG/selection.json" --policy-dir "$BASE_DIR"
show 10b-verify-explore-only 3
step 10c-train-stage1 lerobot-train \
  --policy.path="$BASE_DIR" --policy.push_to_hub=false --policy.device=cpu \
  --dataset.repo_id=local/data_stage1 --dataset.root="$LOG/data_stage1" \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
  --output_dir="$LOG/train_stage1" --job_name=dryrun_stage1
step 10d-train-stage2 lerobot-train \
  --policy.path="$LOG/train_stage1/checkpoints/000002/pretrained_model" --policy.push_to_hub=false --policy.device=cpu \
  --dataset.repo_id=local/data_demos --dataset.root="$LOG/data_demos" \
  --batch_size=1 --num_workers=0 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
  --output_dir="$LOG/train_stage2" --job_name=dryrun_stage2
rm -rf "$LOG/train_mix/checkpoints" "$LOG/train_stage1" "$LOG/train_stage2" "$LOG/data_mix" "$LOG/data_demos" "$LOG/data_stage1"
echo "DRY RUN PASSED"
