#!/bin/bash
# Evaluate the official SmolVLA LIBERO checkpoint on our task(s). This checks the eval pipeline
# and the action/state conventions before we trust any of our own fine-tunes: if this scores
# near 0%, the setup is wrong, not the model.
#   sbatch hpc/01_eval_reference.sbatch
#SBATCH --job-name=cvla-eval-ref
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env

N_EPISODES="${N_EPISODES:-50}"   # rollouts per task
OUT="$RUNS_DIR/eval_reference/${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}_${SLURM_JOB_ID:-local}"

lerobot-eval \
  --policy.path="$REFERENCE_POLICY" \
  --env.type=libero \
  --env.task="$LIBERO_SUITE" \
  --env.task_ids="[$LIBERO_TASK_IDS]" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size=5 \
  --policy.device=cuda \
  --output_dir="$OUT" \
  ${EXTRA_ARGS:-}

echo "Results in $OUT"
