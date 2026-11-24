#!/bin/bash
# Submit the exploration conditions: one collection job per method, then the mixed fine-tunes,
# which start only after their method's collection (and its format check) has succeeded.
#   ssh hpc3, then:  cd ~/curious-vla && bash hpc/submit_exploration.sh
# Defaults: methods random and icm; budgets 5000 and 20000 env steps; 10 and 25 demos with
# seed 0; plus seeds 1 and 2 at 10 demos and 20000 steps. Override with e.g.
#   METHODS=icm BUDGETS=20000 DEMO_BUDGETS=10 SEEDS="0 1 2" EXTRA_SEEDS="" bash hpc/submit_exploration.sh
# SKIP_COLLECT=1 reuses exploration data that is already collected.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
mkdir -p runs/logs

METHODS="${METHODS:-random icm}"
BUDGETS="${BUDGETS:-5000 20000}"
DEMO_BUDGETS="${DEMO_BUDGETS:-10 25}"
SEEDS="${SEEDS:-0}"
EXTRA_SEEDS="${EXTRA_SEEDS-1 2}"          # at the first demo budget and the largest exploration budget
COLLECT_GRES="${COLLECT_GRES:-$SLURM_GRES}"   # the same GPU type the fine-tunes (and their evals) use
max_budget=$(echo $BUDGETS | tr ' ' '\n' | sort -n | tail -1)
first_n=$(echo $DEMO_BUDGETS | awk '{print $1}')
GPU=(--partition="$SLURM_PARTITION" --gres="$SLURM_GRES")
suffix="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"

for m in $METHODS; do
  DEP=()
  if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    c=$(EXPLORE_METHOD=$m COLLECT_STEPS=$max_budget sbatch --parsable --job-name="collect-$m" --partition="$SLURM_PARTITION" \
        --gres="$COLLECT_GRES" hpc/03_collect.sbatch)
    echo "collect $m ($max_budget steps): $c"
    DEP=(--dependency=afterok:$c)
  fi
  submit() {  # submit BUDGET N SEED
    local run="mix-$m-$1_${suffix}_n$2_s$3"
    if [[ -f "$RUNS_DIR/$run/eval/eval_info.json" ]]; then echo "  $run: already done"; return; fi
    local j
    j=$(EXPLORE_METHOD=$m EXPLORE_STEPS=$1 N_DEMOS=$2 SEED=$3 sbatch --parsable --job-name="$run" "${GPU[@]}" "${DEP[@]}" \
        hpc/02_finetune_demos.sbatch)
    echo "  $run: $j"
  }
  for b in $BUDGETS; do for n in $DEMO_BUDGETS; do for s in $SEEDS; do submit "$b" "$n" "$s"; done; done; done
  for s in $EXTRA_SEEDS; do submit "$max_budget" "$first_n" "$s"; done
done
echo
echo "Watch with:  bash hpc/status.sh"
