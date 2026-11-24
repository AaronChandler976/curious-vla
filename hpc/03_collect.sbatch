#!/bin/bash
# Collect exploration data (random or ICM) in the task scenes, then check its format against
# the demos. One run at the largest budget; smaller budgets use its first N steps.
#   EXPLORE_METHOD=random sbatch hpc/03_collect.sbatch
#   EXPLORE_METHOD=icm    sbatch hpc/03_collect.sbatch
# The GPU is only used for rendering (the explorers are small CPU networks), so any GPU type
# works; submit_exploration.sh asks for any free GPU.
#SBATCH --job-name=cvla-collect
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Find a rendering setup that works in this job (see pick_egl_device.py): a short real
# collection on the default EGL device, then each EGL device, then OSMesa.
RENDER_ENV=$(python hpc/pick_egl_device.py --demo-root "$(dataset_root)" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS")
echo "Rendering with: $RENDER_ENV"
export $RENDER_ENV

EXPLORE_METHOD="${EXPLORE_METHOD:?set EXPLORE_METHOD=random or icm}"
COLLECT_STEPS="${COLLECT_STEPS:-20000}"
# EXPLORE_SEED (or SEED) picks the collection's seed: its RNG and its scenes' initial states.
# Seed 0 is the original collection; seeds 1+ write to a directory with an _e<seed> suffix.
SEED="${EXPLORE_SEED:-${SEED:-0}}"
DATA_ROOT=$(dataset_root)
ESUFFIX=""; (( SEED > 0 )) && ESUFFIX="_e${SEED}"
OUT="$RUNS_DIR/exploration/${EXPLORE_METHOD}_${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}${ESUFFIX}"

python hpc/collect_exploration.py --method "$EXPLORE_METHOD" --suite "$LIBERO_SUITE" \
  --task-ids "$LIBERO_TASK_IDS" --steps "$COLLECT_STEPS" --seed "$SEED" \
  --demo-root "$DATA_ROOT" --out "$OUT" ${EXTRA_COLLECT_ARGS:-}
python hpc/check_exploration_format.py --demo-repo "$DEMO_DATASET" --demo-root "$DATA_ROOT" \
  --explore-root "$OUT" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS"
echo "Done: $OUT"
