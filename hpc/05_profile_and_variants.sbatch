#!/bin/bash
# Inference profiling and inference-time ablations on finished checkpoints (no training).
#   sbatch --partition=gpuqs,nsfq --gres=gpu:1 --exclude=... hpc/05_profile_and_variants.sbatch
# 1. profile_inference.py on the reference checkpoint and a demos-only fine-tune -> runs/profile/
# 2. lerobot-eval with n_action_steps (how many of each predicted 50-step chunk are executed
#    before replanning) = 10 and 25, on the reference and on demos-only n40 seeds 0-2 ->
#    runs/eval_variants/<name>/eval_info.json. The main grid uses 50 (the checkpoint's default).
#SBATCH --job-name=cvla-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env
SELF=hpc/05_profile_and_variants.sbatch
gpu_preflight "$SELF"
if ! timeout 600 python hpc/render_check.py --suite "$LIBERO_SUITE" --task-id "${LIBERO_TASK_IDS%%,*}"; then
  gpu_resubmit "$SELF"
fi
REF_DIR=$(hf_local_model "$REFERENCE_POLICY")
SUF="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"
mkdir -p "$RUNS_DIR/profile" "$RUNS_DIR/eval_variants"
nvidia-smi --query-gpu=name,memory.total --format=csv > "$RUNS_DIR/profile/gpu.txt"
python hpc/profile_inference.py --policy "$REF_DIR" --out "$RUNS_DIR/profile/reference.json"
python hpc/profile_inference.py --policy "$RUNS_DIR/demos_${SUF}_n40_s0/checkpoints/last/pretrained_model" \
  --out "$RUNS_DIR/profile/demos_n40_s0.json"
POLICIES=("reference=$REF_DIR")
for s in ${VARIANT_SEEDS:-0 1 2}; do POLICIES+=("demos_n40_s$s=$RUNS_DIR/demos_${SUF}_n40_s$s/checkpoints/last/pretrained_model"); done
for k in ${N_ACTION_STEPS_LIST:-10 25}; do
  for pv in "${POLICIES[@]}"; do
    name=${pv%%=*}; path=${pv#*=}; out="$RUNS_DIR/eval_variants/${name}_nas$k"
    [[ -f "$out/eval_info.json" ]] && { echo "$out: done"; continue; }
    rm -rf "$out"
    seed=0; [[ "$name" =~ _s([0-9]+)$ ]] && seed=${BASH_REMATCH[1]}
    lerobot-eval --policy.path="$path" --policy.n_action_steps="$k" --env.type=libero --env.task="$LIBERO_SUITE" \
      --env.task_ids="[$LIBERO_TASK_IDS]" --eval.n_episodes=50 --eval.batch_size=5 --policy.device=cuda \
      --seed="$seed" --output_dir="$out" ||
    lerobot-eval --policy.path="$path" --policy.n_action_steps="$k" --env.type=libero --env.task="$LIBERO_SUITE" \
      --env.task_ids="[$LIBERO_TASK_IDS]" --eval.n_episodes=50 --eval.batch_size=5 --policy.device=cuda \
      --seed="$seed" --output_dir="$out" --eval.use_async_envs=false
  done
done
echo "Done"
