#!/bin/bash
# HPC3 (g17) has no internet, so everything is downloaded on the desktop and copied over.
# Run this ON THE UBUNTU DESKTOP, from a clone of this branch, with the SJSU VPN on:
#   git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
#   cd ~/curious-vla && bash hpc/stage_from_desktop.sh
# It uses the network and CPU only, not the GPU, so it can run while the GPU is busy.
# Safe to re-run: finished steps are skipped and rsync only sends what changed.
#
# What it does:
#   1. Builds the conda env (Python 3.12 + lerobot[smolvla,libero]) in ~/cvla-stage and packs
#      it into one relocatable tarball with conda-pack.
#   2. Downloads SmolVLA base, the reference checkpoint, the LIBERO demo dataset and the
#      LIBERO assets into ~/cvla-stage (not into your normal caches).
#   3. Copies the repo, the env and the caches to HPC3 and unpacks the env there.
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
source "$REPO/hpc/config.sh"
export HPC_OFFLINE=0 HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0   # the desktop has internet

HOST="${HPC_HOST:-hpc3}"                 # ssh alias from ~/.ssh/config
STAGE="${STAGE_DIR:-$HOME/cvla-stage}"
MC="$STAGE/miniconda"
ENV_PREFIX="$STAGE/env-build"
TARBALL="$STAGE/cvla-env.tar.gz"
# Separate caches for the cluster copy, so your own ~/.cache is not touched.
export HF_HOME="$STAGE/hf"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
LIBERO_ASSETS="$STAGE/libero-assets"
mkdir -p "$STAGE"

echo "== 1/4 Build and pack the env"
if [[ ! -x "$MC/bin/conda" ]]; then
  curl -fsSL -o "$STAGE/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$STAGE/miniconda.sh" -b -p "$MC" && rm -f "$STAGE/miniconda.sh"
fi
if [[ ! -f "$TARBALL" ]]; then
  "$MC/bin/conda" install -y -n base -c conda-forge conda-pack
  [[ -d "$ENV_PREFIX" ]] || "$MC/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge \
      "python=$PYTHON_VERSION" "ffmpeg>=6,<8" cmake
  "$ENV_PREFIX/bin/pip" install --upgrade pip
  "$ENV_PREFIX/bin/pip" install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
  "$ENV_PREFIX/bin/pip" install "lerobot[smolvla,libero]==$LEROBOT_VERSION" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  "$MC/bin/conda-pack" -p "$ENV_PREFIX" -o "$TARBALL" --ignore-missing-files
fi
ls -lh "$TARBALL"

echo "== 2/4 Download models, demo dataset and LIBERO assets"
mkdir -p "$LIBERO_ASSETS"
HOME_FOR_LIBERO="$STAGE/libero-home"      # LIBERO writes its assets under \$HOME/.cache/libero
mkdir -p "$HOME_FOR_LIBERO"
HOME="$HOME_FOR_LIBERO" LIBERO_CONFIG_PATH="$HOME_FOR_LIBERO/.libero" \
  bash -c "echo n | '$ENV_PREFIX/bin/python' -c 'from libero.libero import get_libero_path; print(get_libero_path(\"assets\"))'"
"$ENV_PREFIX/bin/python" - <<EOF
from huggingface_hub import snapshot_download
for repo in ["$BASE_POLICY", "$REFERENCE_POLICY"]:
    print("model:", snapshot_download(repo_id=repo))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$DEMO_DATASET")
print(f"dataset: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
EOF
du -sh "$HF_HOME" "$HOME_FOR_LIBERO/.cache/libero" 2>/dev/null || true

echo "== 3/4 Copy to $HOST (repo, env, caches)"
ssh "$HOST" 'mkdir -p ~/curious-vla ~/.cache/huggingface ~/.cache/libero ~/cvla-env'
rsync -a --partial --info=progress2 --exclude runs/ --exclude .git/ "$REPO/" "$HOST:curious-vla/"
rsync -a --partial --info=progress2 "$TARBALL" "$HOST:cvla-env.tar.gz"
rsync -a --partial --info=progress2 "$HF_HOME/" "$HOST:.cache/huggingface/"
rsync -a --partial --info=progress2 "$HOME_FOR_LIBERO/.cache/libero/" "$HOST:.cache/libero/"

echo "== 4/4 Unpack the env on $HOST"
ssh "$HOST" 'bash -s' <<'EOF'
set -euo pipefail
if [[ ! -x ~/cvla-env/bin/conda-unpack ]]; then
  tar -xzf ~/cvla-env.tar.gz -C ~/cvla-env
  ~/cvla-env/bin/conda-unpack
fi
source ~/cvla-env/bin/activate
mkdir -p ~/.libero
# Create LIBERO's config with paths for this machine (the answer "n" keeps the defaults).
[[ -f ~/.libero/config.yaml ]] || echo n | python -c "import libero.libero"
HF_HUB_OFFLINE=1 python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__)"
df -h ~ | tail -1
EOF

echo
echo "Staging done. Next: ssh $HOST, then  cd ~/curious-vla && bash hpc/submit_baseline.sh"
