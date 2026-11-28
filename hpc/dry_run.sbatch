#!/bin/bash
# The CPU dry run (dry_run.sh), inside a short GPU job: on days when the login node cannot render
# (no EGL device, no OSMesa), the collection steps need a node with a GPU for rendering.
#   sbatch --partition=gpuqs,nsfq --gres=gpu:1 --exclude=g[1-15],g25,g27,g29,cs002 hpc/dry_run.sbatch
#SBATCH --job-name=cvla-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=runs/logs/%x-%j.out
set -uo pipefail
source hpc/config.sh
activate_env
RENDER_ENV=$(python hpc/pick_egl_device.py --demo-root "$(dataset_root)" --suite "$LIBERO_SUITE" --task-ids "$LIBERO_TASK_IDS")
echo "Rendering with: $RENDER_ENV"
export $RENDER_ENV
SKIP_EVALS="${SKIP_EVALS:-1}" bash hpc/dry_run.sh
