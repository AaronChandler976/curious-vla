#!/bin/bash
# Re-evaluate finished runs with a different n_action_steps (how many actions of each predicted
# 50-step chunk are executed before the policy replans). The main grid was evaluated with the
# checkpoint default of 50 (open loop over the whole chunk); 10 is the closed-loop setting LeRobot
# documents for LIBERO and scores much higher (reference checkpoint 64% -> 82%).
#   RUN_LIST=runs/reeval/list.txt NAS=10 sbatch hpc/06_reeval.sbatch
# Writes runs/<run>/eval_nas<NAS>/eval_info.json with the same episodes and seed as the original eval.
#SBATCH --job-name=cvla-reeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -uo pipefail
source hpc/config.sh
activate_env
SELF=hpc/06_reeval.sbatch
gpu_preflight "$SELF"
if ! timeout 600 python hpc/render_check.py --suite "$LIBERO_SUITE" --task-id "${LIBERO_TASK_IDS%%,*}"; then
  gpu_resubmit "$SELF"
fi
NAS="${NAS:-10}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-50}"
fails=0
while read -r run; do
  [[ -z "$run" ]] && continue
  dir="$RUNS_DIR/$run"; out="$dir/eval_nas$NAS"
  [[ -f "$out/eval_info.json" ]] && { echo "$run: done"; continue; }
  ck="$dir/checkpoints/last/pretrained_model"
  [[ -d "$ck" ]] || { echo "$run: no checkpoint"; continue; }
  seed=$(sed -E 's/.*_s([0-9]+).*/\1/' <<<"$run")
  rm -rf "$out"
  args=(--policy.path="$ck" --policy.n_action_steps="$NAS" --env.type=libero --env.task="$LIBERO_SUITE"
        --env.task_ids="[$LIBERO_TASK_IDS]" --eval.n_episodes="$N_EVAL_EPISODES" --eval.batch_size=5
        --policy.device=cuda --seed="$seed" --output_dir="$out")
  echo "== $run (seed $seed, n_action_steps $NAS)"
  if ! lerobot-eval "${args[@]}"; then
    rm -rf "$out"
    lerobot-eval "${args[@]}" --eval.use_async_envs=false || { echo "$run: eval failed"; rm -rf "$out"; fails=$((fails + 1)); }
  fi
  rm -rf "$out/videos"   # the original eval kept videos; these would only fill /home
done < "${RUN_LIST:?set RUN_LIST}"
(( fails == 0 )) || exit 1
