#!/bin/bash
# Submit the demos + exploration fine-tunes, and any exploration collection they still need.
#   ssh hpc3, then:  cd ~/curious-vla && bash hpc/submit_exploration.sh
# Every combination of METHODS x BUDGETS x DEMO_BUDGETS x SEEDS x EXPLORE_SEEDS is one run.
# Defaults: methods random and icm; budgets 5000 and 20000 env steps; 10, 25 and 40 demos per
# task; seeds 0-2; the original exploration collection (EXPLORE_SEEDS=0). Override with e.g.
#   METHODS=icm BUDGETS=20000 DEMO_BUDGETS=10 SEEDS="0 1 2" EXPLORE_SEEDS="1 2" bash hpc/submit_exploration.sh
# A collection (one per method and exploration seed, at the largest budget; smaller budgets use
# its first N steps) is submitted only if its data is missing and it is not already queued.
# The fine-tunes that need it wait for it (afterok). SKIP_COLLECT=1 never submits one.
# Runs that already have an eval, or are already in the queue, are skipped, so re-running only
# submits what is missing.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
mkdir -p runs/logs

METHODS="${METHODS:-random icm}"
BUDGETS="${BUDGETS:-5000 20000}"
DEMO_BUDGETS="${DEMO_BUDGETS:-10 25 40}"
SEEDS="${SEEDS:-0 1 2}"
EXPLORE_SEEDS="${EXPLORE_SEEDS:-0}"
COLLECT_STEPS="${COLLECT_STEPS:-20000}"
GPU=(--partition="$SLURM_PARTITION" --gres="$SLURM_GRES" ${SLURM_EXCLUDE:+--exclude="$SLURM_EXCLUDE"})
CGPU=(--partition="$SLURM_PARTITION" --gres="$SLURM_GRES" ${COLLECT_EXCLUDE:+--exclude="$COLLECT_EXCLUDE"})
suffix="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"
queued_id() { squeue -u "$USER" -h -n "$1" -o %i | head -n 1; }

for m in $METHODS; do
  for es in $EXPLORE_SEEDS; do
    esuf=""; (( es > 0 )) && esuf="_e$es"
    DEP=()
    cname="collect-$m$esuf"
    if [[ ! -f "$RUNS_DIR/exploration/${m}_${suffix}${esuf}/exploration_log.json" ]]; then
      c=$(queued_id "$cname")
      if [[ -z "$c" && "${SKIP_COLLECT:-0}" != "1" ]]; then
        c=$(EXPLORE_METHOD=$m EXPLORE_SEED=$es COLLECT_STEPS=$COLLECT_STEPS sbatch --parsable --job-name="$cname" \
            "${CGPU[@]}" hpc/03_collect.sbatch)
        echo "$cname ($COLLECT_STEPS steps): $c"
      elif [[ -n "$c" ]]; then
        echo "$cname: already queued as $c"
      fi
      [[ -n "$c" ]] && DEP=(--dependency=afterok:$c)
    fi
    for b in $BUDGETS; do for n in $DEMO_BUDGETS; do for s in $SEEDS; do
      run="mix-$m-${b}_${suffix}_n${n}_s${s}${esuf}"
      if [[ -f "$RUNS_DIR/$run/eval/eval_info.json" ]]; then echo "  $run: already done"; continue; fi
      if [[ -n "$(queued_id "$run")" ]]; then echo "  $run: already queued or running"; continue; fi
      # A job that died before its first checkpoint can leave an output dir that lerobot-train
      # refuses to reuse. Move it aside (nothing is deleted). With checkpoints/last the job resumes.
      if [[ -d "$RUNS_DIR/$run" && ! -d "$RUNS_DIR/$run/checkpoints/last" ]]; then
        mv "$RUNS_DIR/$run" "$RUNS_DIR/$run.failed-$(date +%Y%m%d-%H%M%S)"
        echo "  $run: moved the partial output dir aside"
      fi
      j=$(EXPLORE_METHOD=$m EXPLORE_STEPS=$b EXPLORE_SEED=$es N_DEMOS=$n SEED=$s sbatch --parsable --job-name="$run" \
          "${GPU[@]}" ${DEP[@]+"${DEP[@]}"} hpc/02_finetune_demos.sbatch)
      echo "  $run: $j"
    done; done; done
  done
done
echo
echo "Watch with:  bash hpc/status.sh"
