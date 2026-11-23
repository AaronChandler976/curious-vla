#!/bin/bash
# Submit the whole first batch in one go, so it runs overnight without you:
#   cd ~/curious-vla && bash hpc/submit_baseline.sh
# 1) the GPU check, 2) the reference eval and 3) the demos-only fine-tunes for each demo
# budget. Steps 2 and 3 start only if the check passes (Slurm afterok dependency).
# Override the sweep with e.g.:  DEMO_BUDGETS="5 10 25" SEEDS="0 1" bash hpc/submit_baseline.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
mkdir -p runs/logs

DEMO_BUDGETS="${DEMO_BUDGETS:-10 25}"
SEEDS="${SEEDS:-0}"

check=$(sbatch --parsable hpc/00_check.sbatch)
echo "check job:        $check"
ref=$(sbatch --parsable --dependency=afterok:$check hpc/01_eval_reference.sbatch)
echo "reference eval:   $ref"
for n in $DEMO_BUDGETS; do
  for s in $SEEDS; do
    j=$(N_DEMOS=$n SEED=$s sbatch --parsable --dependency=afterok:$check hpc/02_finetune_demos.sbatch)
    echo "fine-tune n=$n s=$s: $j"
  done
done
echo
echo "Watch with:  squeue -u \$USER     Logs in: runs/logs/"
