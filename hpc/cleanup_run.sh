#!/bin/bash
# Shrink a finished run: keep checkpoints/last/pretrained_model and the 10k-step
# pretrained_model, and delete optimizer states and other checkpoints. Safe to re-run.
#   bash hpc/cleanup_run.sh runs/<run>        or, for every run that has an eval:
#   for r in runs/*/eval/eval_info.json; do bash hpc/cleanup_run.sh "${r%/eval/eval_info.json}"; done
set -euo pipefail
run=${1:?run dir}
ck="$run/checkpoints"
[[ -d "$ck" ]] || exit 0
last=$(readlink -f "$ck/last" 2>/dev/null || true)
for d in "$ck"/[0-9]*; do
  [[ -d "$d" ]] || continue
  if [[ "$(readlink -f "$d")" == "$last" || "$(basename "$d")" == "010000" ]]; then
    rm -rf "$d/training_state"
  else
    rm -rf "$d"
  fi
done
du -sh "$run" 2>/dev/null || true
