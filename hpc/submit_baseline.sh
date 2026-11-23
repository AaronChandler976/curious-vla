#!/bin/bash
# Submit the whole first batch in one go, so it runs overnight without you:
#   ssh hpc3, then:  cd ~/curious-vla && bash hpc/submit_baseline.sh
# 1) the GPU check, 2) the reference eval and 3) the demos-only fine-tunes for each demo
# budget. Steps 2 and 3 start only if the check passes (Slurm afterok dependency).
# Override the sweep with e.g.:  DEMO_BUDGETS="5 10 25" SEEDS="0 1" bash hpc/submit_baseline.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
mkdir -p runs/logs

DEMO_BUDGETS="${DEMO_BUDGETS:-10 25}"
SEEDS="${SEEDS:-0}"
GPU=(--partition="$SLURM_PARTITION" --gres="$SLURM_GRES")
echo "Submitting to partition $SLURM_PARTITION with --gres=$SLURM_GRES on $(hostname)"

# SKIP_CHECK=1 skips the GPU check once it has passed on this cluster.
if [[ "${SKIP_CHECK:-0}" == "1" ]]; then
  DEP=()
  echo "check job:        skipped"
else
  check=$(sbatch --parsable "${GPU[@]}" hpc/00_check.sbatch)
  echo "check job:        $check"
  DEP=(--dependency=afterok:$check)
fi
ref=$(sbatch --parsable "${GPU[@]}" "${DEP[@]}" hpc/01_eval_reference.sbatch)
echo "reference eval:   $ref"
for n in $DEMO_BUDGETS; do
  for s in $SEEDS; do
    j=$(N_DEMOS=$n SEED=$s sbatch --parsable "${GPU[@]}" "${DEP[@]}" hpc/02_finetune_demos.sbatch)
    echo "fine-tune n=$n s=$s: $j"
  done
done
echo
echo "Watch with:  squeue -u \$USER     Logs in: runs/logs/"
