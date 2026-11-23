#!/bin/bash
# Condition 1 (demos only): fine-tune SmolVLA on N demos per task, then evaluate it.
# One job = one point on the "success vs. number of demos" curve. Examples:
#   sbatch hpc/02_finetune_demos.sbatch                              # 10 demos, seed 0
#   N_DEMOS=25 SEED=1 sbatch hpc/02_finetune_demos.sbatch
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
RUN="demos_${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}_n${N_DEMOS}_s${SEED}"
OUT="$RUNS_DIR/$RUN"

DATA_ROOT=$(dataset_root)
BASE_DIR=$(hf_local_model "$BASE_POLICY")
echo "Dataset: $DATA_ROOT"
echo "Base policy: $BASE_DIR"

EPISODES=$(python hpc/select_episodes.py --dataset "$DEMO_DATASET" --root "$DATA_ROOT" --suite "$LIBERO_SUITE" \
  --task-ids "$LIBERO_TASK_IDS" --n-demos "$N_DEMOS" --seed "$SEED" --out "$RUNS_DIR/selections/$RUN.json")
echo "Training on episodes: $EPISODES"

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
  --rename_map="$FINETUNE_RENAME_MAP" \
  --dataset.repo_id="$DEMO_DATASET" \
  --dataset.root="$DATA_ROOT" \
  --dataset.episodes="$EPISODES" \
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
  --rename_map="$FINETUNE_RENAME_MAP" \
  --env.type=libero \
  --env.task="$LIBERO_SUITE" \
  --env.task_ids="[$LIBERO_TASK_IDS]" \
  --eval.n_episodes="$N_EVAL_EPISODES" \
  --eval.batch_size=5 \
  --policy.device=cuda \
  --seed="$SEED" \
  --output_dir="$OUT/eval"

echo "Done: $OUT"
