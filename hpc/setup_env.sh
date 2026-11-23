#!/bin/bash
# One-time setup. Run this on the LOGIN node (coe-hpc1), not in a job:
#   bash ~/curious-vla/hpc/setup_env.sh
# It installs Miniconda in your home directory, creates the conda env, installs LeRobot with
# the SmolVLA and LIBERO extras, and downloads the models, the LIBERO assets and the demo dataset.
# Running it again is safe: steps that are already done are skipped.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/config.sh"

echo "== 1/5 Miniconda in $CONDA_ROOT"
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  installer="$HOME/miniconda-installer.sh"
  curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_ROOT"
  rm -f "$installer"
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

echo "== 2/5 conda env $CONDA_ENV (Python $PYTHON_VERSION)"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"
# ffmpeg is needed to decode the dataset videos; cmake is needed to build some LIBERO deps.
conda install -y -c conda-forge "ffmpeg>=6,<8" cmake

echo "== 3/5 lerobot[smolvla,libero]==$LEROBOT_VERSION"
# The torch wheels from PyPI bundle their own CUDA runtime, so the cluster's old CUDA modules
# are not needed. They need a recent NVIDIA driver on the GPU node; 00_check.sbatch verifies it.
pip install --upgrade pip
pip install "lerobot[smolvla,libero]==$LEROBOT_VERSION"

echo "== 4/5 LIBERO config and assets"
mkdir -p "$LIBERO_CONFIG_PATH"
# LIBERO asks an interactive question on first import. Answering "n" keeps the default paths.
echo n | python - <<'EOF'
from libero.libero import get_libero_path
print("LIBERO assets at:", get_libero_path("assets"))
EOF

echo "== 5/5 Download models and the demo dataset into the cache"
python - <<EOF
from huggingface_hub import snapshot_download
for repo in ["$BASE_POLICY", "$REFERENCE_POLICY"]:
    print("model:", snapshot_download(repo_id=repo))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$DEMO_DATASET")
print(f"dataset: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames at {ds.root}")
EOF

mkdir -p "$RUNS_DIR/logs"
echo
echo "Setup done. Disk used in home:"
du -sh "$CONDA_ROOT" "$HF_HOME" 2>/dev/null || true
echo "Next: cd $PROJECT_ROOT && sbatch hpc/00_check.sbatch"
