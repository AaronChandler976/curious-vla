#!/bin/bash
# Fine-tune SmolVLA on N demos per task (optionally mixed with exploration data), then evaluate.
# One job = one point on the "success vs. number of demos / exploration data" curves. Examples:
#   sbatch hpc/02_finetune_demos.sbatch                              # demos only, 10 demos, seed 0
#   N_DEMOS=25 SEED=1 sbatch hpc/02_finetune_demos.sbatch
#   EXPLORE_METHOD=icm EXPLORE_STEPS=5000 N_DEMOS=10 sbatch hpc/02_finetune_demos.sbatch
# With EXPLORE_METHOD set, the job first builds the demos + exploration mix (build_mix.py).
# Everything else (base policy, steps, batch size, normalization stats, eval) is identical.
# The variables are passed into the job because Slurm exports your environment by default.
#SBATCH --job-name=cvla-ft-demos
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env

N_DEMOS="${N_DEMOS:-10}"
SEED="${SEED:-0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-50}"
EXPLORE_METHOD="${EXPLORE_METHOD:-}"
EXPLORE_STEPS="${EXPLORE_STEPS:-0}"
SUFFIX="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}_n${N_DEMOS}_s${SEED}"
if [[ -n "$EXPLORE_METHOD" ]]; then RUN="mix-${EXPLORE_METHOD}-${EXPLORE_STEPS}_${SUFFIX}"; else RUN="demos_${SUFFIX}"; fi
OUT="$RUNS_DIR/$RUN"

DATA_ROOT=$(dataset_root)
# smolvla_base with LIBERO's features in its config (see make_base_policy.py).
BASE_DIR="$RUNS_DIR/base_policy/$RUN"
python hpc/make_base_policy.py "$(hf_local_model "$BASE_POLICY")" \
  "$(hf_local_model "$REFERENCE_POLICY")" "$DATA_ROOT" "$BASE_DIR"
echo "Dataset: $DATA_ROOT"
echo "Base policy: $BASE_DIR"

EPISODES=$(python hpc/select_episodes.py --dataset "$DEMO_DATASET" --root "$DATA_ROOT" --suite "$LIBERO_SUITE" \
  --task-ids "$LIBERO_TASK_IDS" --n-demos "$N_DEMOS" --seed "$SEED" --out "$RUNS_DIR/selections/$RUN.json" | tail -n 1)
echo "Training on episodes: $EPISODES"
[[ "$EPISODES" =~ ^\[[0-9,]+\]$ ]] || { echo "ERROR: episode selection did not return a JSON list" >&2; exit 1; }

if [[ -n "$EXPLORE_METHOD" ]]; then
  EXPLORE_ROOT="$RUNS_DIR/exploration/${EXPLORE_METHOD}_${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"
  [[ -f "$EXPLORE_ROOT/exploration_log.json" ]] || { echo "ERROR: no exploration data at $EXPLORE_ROOT; run 03_collect.sbatch first" >&2; exit 1; }
  MIX_ROOT="$RUNS_DIR/mixdata/$RUN"
  if [[ ! -f "$MIX_ROOT/mix_info.json" ]]; then
    python hpc/build_mix.py --demo-repo "$DEMO_DATASET" --demo-root "$DATA_ROOT" --episodes "$EPISODES" \
      --explore-root "$EXPLORE_ROOT" --budget "$EXPLORE_STEPS" --out "$MIX_ROOT"
  fi
  cat "$MIX_ROOT/mix_info.json"; echo
  DATA_ARGS=(--dataset.repo_id="local/$RUN" --dataset.root="$MIX_ROOT")
else
  DATA_ARGS=(--dataset.repo_id="$DEMO_DATASET" --dataset.root="$DATA_ROOT" --dataset.episodes="$EPISODES")
fi

# lerobot-train refuses to reuse an existing output dir unless it is resuming, so the
# episode selection above is saved under runs/selections/ instead of inside $OUT.
# Training-time sim eval is off (env_eval_freq=0); we evaluate the final checkpoint below.
if [[ -d "$OUT/checkpoints/last" ]]; then
  # A previous job hit the time limit: resume from its last checkpoint with its saved config.
  echo "Resuming $OUT"
  lerobot-train --resume=true --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
else
lerobot-train \
  --policy.path="$BASE_DIR" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  "${DATA_ARGS[@]}" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --env_eval_freq=0 \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  ${EXTRA_TRAIN_ARGS:-}
fi

lerobot-eval \
  --policy.path="$OUT/checkpoints/last/pretrained_model" \
  --env.type=libero \
  --env.task="$LIBERO_SUITE" \
  --env.task_ids="[$LIBERO_TASK_IDS]" \
  --eval.n_episodes="$N_EVAL_EPISODES" \
  --eval.batch_size=5 \
  --policy.device=cuda \
  --seed="$SEED" \
  --output_dir="$OUT/eval"

# /home is nearly full: keep only the final weights (and the 10k-step weights, for an
# overfitting check), and drop optimizer states and the mixed dataset.
bash hpc/cleanup_run.sh "$OUT"
if [[ -n "$EXPLORE_METHOD" ]]; then rm -rf "$MIX_ROOT"; fi
echo "Done: $OUT"
