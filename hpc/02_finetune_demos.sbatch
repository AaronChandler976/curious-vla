#!/bin/bash
# Fine-tune SmolVLA on N demos per task (optionally mixed with exploration data), then evaluate.
# One job = one point on the "success vs. number of demos / exploration data" curves. Examples:
#   sbatch hpc/02_finetune_demos.sbatch                              # demos only, 10 demos, seed 0
#   N_DEMOS=25 SEED=1 sbatch hpc/02_finetune_demos.sbatch
#   EXPLORE_METHOD=icm EXPLORE_STEPS=5000 N_DEMOS=10 sbatch hpc/02_finetune_demos.sbatch
# The job first builds its training dataset (build_dataset.py) and verifies it; with
# EXPLORE_METHOD set, that dataset is the demos + exploration mix.
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
SELF=hpc/02_finetune_demos.sbatch   # Slurm runs a copy of this file, so name it for resubmits
gpu_preflight "$SELF"
# Rendering check (see render_check.py): a node that cannot render would fail the eval at the end.
if ! timeout 600 python hpc/render_check.py --suite "$LIBERO_SUITE" --task-id "${LIBERO_TASK_IDS%%,*}"; then
  echo "render check failed on ${SLURMD_NODENAME:-$(hostname -s)}" >&2
  gpu_resubmit "$SELF"
fi

N_DEMOS="${N_DEMOS:-10}"
SEED="${SEED:-0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-50}"
EXPLORE_METHOD="${EXPLORE_METHOD:-}"
EXPLORE_STEPS="${EXPLORE_STEPS:-0}"
# Which exploration collection to mix in: seed 0 is the original one (no suffix in any name);
# seeds 1+ are independent re-collections, used to tell the effect of the method from that of
# one particular draw of exploration data. Run names get _e<seed>.
EXPLORE_SEED="${EXPLORE_SEED:-0}"
SUFFIX="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}_n${N_DEMOS}_s${SEED}"
ESUFFIX=""; (( EXPLORE_SEED > 0 )) && ESUFFIX="_e${EXPLORE_SEED}"
# EXPLORE_PCT: the percentage of training frames that are exploration frames (default 50). Other
# values are an ablation of the mixing ratio; their run names get _f<pct>.
EXPLORE_PCT="${EXPLORE_PCT:-50}"
FSUFFIX=""; [[ "$EXPLORE_PCT" != 50 ]] && FSUFFIX="_f${EXPLORE_PCT}"
if [[ -n "$EXPLORE_METHOD" ]]; then RUN="mix-${EXPLORE_METHOD}-${EXPLORE_STEPS}_${SUFFIX}${ESUFFIX}${FSUFFIX}"; else RUN="demos_${SUFFIX}"; fi
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

# Every condition trains on a fresh local dataset written by build_dataset.py (demos only, or
# demos + the first EXPLORE_STEPS of an exploration run), verified frame by frame against the
# raw source rows before training starts.
SELECTION="$RUNS_DIR/selections/$RUN.json"
BUILD_ARGS=()
if [[ -n "$EXPLORE_METHOD" ]]; then
  EXPLORE_ROOT="$RUNS_DIR/exploration/${EXPLORE_METHOD}_${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}${ESUFFIX}"
  [[ -f "$EXPLORE_ROOT/exploration_log.json" ]] || { echo "ERROR: no exploration data at $EXPLORE_ROOT; run 03_collect.sbatch first" >&2; exit 1; }
  BUILD_ARGS=(--explore-root "$EXPLORE_ROOT" --budget "$EXPLORE_STEPS" --explore-frac "$(awk -v p="$EXPLORE_PCT" 'BEGIN{print p/100}')")
fi
TRAIN_DATA="$RUNS_DIR/traindata/$RUN"
if [[ ! -f "$TRAIN_DATA/verified" ]]; then
  python hpc/build_dataset.py --demo-root "$DATA_ROOT" --selection "$SELECTION" "${BUILD_ARGS[@]}" --out "$TRAIN_DATA"
  python hpc/verify_training_data.py --root "$TRAIN_DATA" --selection "$SELECTION" --policy-dir "$BASE_DIR"
  touch "$TRAIN_DATA/verified"
fi
cat "$TRAIN_DATA/mix_info.json"; echo
DATA_ARGS=(--dataset.repo_id="local/$RUN" --dataset.root="$TRAIN_DATA")

# lerobot-train refuses to reuse an existing output dir unless it is resuming, so the
# episode selection above is saved under runs/selections/ instead of inside $OUT.
# Training-time sim eval is off (env_eval_freq=0); we evaluate the final checkpoint below.
if [[ -d "$OUT" && ! -d "$OUT/checkpoints/last" ]]; then
  # A job that died before its first checkpoint: lerobot-train would refuse this dir.
  mv "$OUT" "$OUT.failed-$(date +%Y%m%d-%H%M%S)"; echo "Moved a partial $OUT aside"
fi
if [[ -d "$OUT/checkpoints/$(printf %06d "$STEPS")/pretrained_model" ]]; then
  # Training finished in an earlier job that then died (e.g. in the eval): only evaluate.
  echo "Training already finished in $OUT; evaluating only"
elif [[ -d "$OUT/checkpoints/last" ]]; then
  # A previous job hit the time limit: resume from its last checkpoint with its saved config.
  echo "Resuming $OUT"
  run_gpu "$SELF" lerobot-train --resume=true --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"
else
run_gpu "$SELF" lerobot-train \
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

EVAL_ARGS=(
  --policy.path="$OUT/checkpoints/last/pretrained_model"
  --env.type=libero
  --env.task="$LIBERO_SUITE"
  --env.task_ids="[$LIBERO_TASK_IDS]"
  --eval.n_episodes="$N_EVAL_EPISODES"
  --eval.batch_size=5
  --policy.device=cuda
  --seed="$SEED"
  --output_dir="$OUT/eval"
)
# If a simulator worker process dies, try once more with the simulators in the main process (same
# episodes and seeds). If that fails too, the job is resubmitted to another node, where it only
# evaluates (at most GPU_MAX_RETRIES times).
if ! lerobot-eval "${EVAL_ARGS[@]}"; then
  echo "lerobot-eval failed; retrying with --eval.use_async_envs=false"
  rm -rf "$OUT/eval"
  lerobot-eval "${EVAL_ARGS[@]}" --eval.use_async_envs=false || { gpu_report; gpu_resubmit "$SELF"; }
fi

# /home is nearly full: keep only the final weights (and the 10k-step weights, for an
# overfitting check), and drop optimizer states and the training dataset.
bash hpc/cleanup_run.sh "$OUT"
rm -rf "$TRAIN_DATA"
echo "Done: $OUT"
