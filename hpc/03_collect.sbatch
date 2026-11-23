#!/bin/bash
# Collect exploration data (random or ICM) in the task scenes, then check its format against
# the demos. One run at the largest budget; smaller budgets use its first N steps.
#   EXPLORE_METHOD=random sbatch hpc/03_collect.sbatch
#   EXPLORE_METHOD=icm    sbatch hpc/03_collect.sbatch
# The GPU is only used for rendering (the explorers are small CPU networks), so any GPU type
# works. submit_exploration.sh asks for a P100, which usually has no queue.
#SBATCH --job-name=cvla-collect
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env

EXPLORE_METHOD="${EXPLORE_METHOD:?set EXPLORE_METHOD=random or icm}"
COLLECT_STEPS="${COLLECT_STEPS:-20000}"
SEED="${SEED:-0}"
DATA_ROOT=$(dataset_root)
OUT="$RUNS_DIR/exploration/${EXPLORE_METHOD}_${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"

python hpc/collect_exploration.py --method "$EXPLORE_METHOD" --suite "$LIBERO_SUITE" \
  --task-ids "$LIBERO_TASK_IDS" --steps "$COLLECT_STEPS" --seed "$SEED" \
  --demo-root "$DATA_ROOT" --out "$OUT" ${EXTRA_COLLECT_ARGS:-}
python hpc/check_exploration_format.py --demo-repo "$DEMO_DATASET" --demo-root "$DATA_ROOT" \
  --explore-root "$OUT" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS"
echo "Done: $OUT"
