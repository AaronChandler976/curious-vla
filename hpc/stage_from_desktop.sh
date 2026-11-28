#!/bin/bash
# HPC3 (g17) has no internet, so everything is downloaded on the desktop and copied over.
# Run this ON THE UBUNTU DESKTOP, from a clone of this branch, with the SJSU VPN on:
#   git clone -b hpc-scripts https://github.com/AaronChandler976/curious-vla.git ~/curious-vla
#   (or, if ~/curious-vla already exists:  cd ~/curious-vla && git fetch origin && git switch hpc-scripts && git pull)
#   cd ~/curious-vla && bash hpc/stage_from_desktop.sh
# Staging goes to ~/cvla-stage on the desktop and to your home on HPC3. Don't move it to /scratch
# on HPC3: files there are deleted 21 days into each semester.
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

# Every conda call uses only conda-forge (--override-channels). That avoids the Anaconda
# channels' Terms of Service prompt, which stops non-interactive installs.
echo "== 1/5 Build and pack the env"
if [[ ! -x "$MC/bin/conda" ]]; then
  curl -fsSL -o "$STAGE/miniconda.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$STAGE/miniconda.sh" -b -p "$MC" && rm -f "$STAGE/miniconda.sh"
fi
if [[ ! -f "$TARBALL" ]]; then
  "$MC/bin/conda" install -y -n base --override-channels -c conda-forge conda-pack
  [[ -d "$ENV_PREFIX" ]] || "$MC/bin/conda" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge \
      "python=$PYTHON_VERSION" "ffmpeg>=6,<8" cmake
  "$ENV_PREFIX/bin/pip" install --upgrade pip
  "$ENV_PREFIX/bin/pip" install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
  "$ENV_PREFIX/bin/pip" install "lerobot[smolvla,libero]==$LEROBOT_VERSION" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  "$MC/bin/conda-pack" -p "$ENV_PREFIX" -o "$TARBALL" --ignore-missing-files
fi
ls -lh "$TARBALL"

echo "== 2/5 Download models, demo dataset and LIBERO assets"
# LIBERO's scene and object files (lerobot/libero-assets) are not in the pip package. They
# go into the package's own assets/ folder on HPC3, which is where LIBERO looks first.
"$ENV_PREFIX/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/libero-assets', repo_type='dataset', local_dir='$LIBERO_ASSETS')"
test -f "$LIBERO_ASSETS/scenes/libero_floor_base_style.xml" || { echo "LIBERO assets look incomplete in $LIBERO_ASSETS" >&2; exit 1; }
"$ENV_PREFIX/bin/python" - <<EOF
from huggingface_hub import snapshot_download
for repo in ["$BASE_POLICY", "$REFERENCE_POLICY"]:
    print("model:", snapshot_download(repo_id=repo))
# The backbone without its ONNX exports, which SmolVLA never uses.
for repo in "$VLM_BACKBONES".split():
    print("backbone:", snapshot_download(repo_id=repo, allow_patterns="$VLM_BACKBONE_FILES".split(",")))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("$DEMO_DATASET")
print(f"dataset: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
EOF
du -sh "$HF_HOME" "$LIBERO_ASSETS" 2>/dev/null || true

echo "== 3/5 Copy to $HOST (repo, env, caches)"
ssh "$HOST" 'mkdir -p ~/curious-vla ~/.cache/huggingface ~/cvla-env'
rsync -a --partial --info=progress2 --exclude runs/ --exclude .git/ "$REPO/" "$HOST:curious-vla/"
rsync -a --partial --info=progress2 "$TARBALL" "$HOST:cvla-env.tar.gz"
rsync -a --partial --info=progress2 "$HF_HOME/" "$HOST:.cache/huggingface/"

echo "== 4/5 Unpack the env on $HOST"
ssh "$HOST" 'bash -s' <<'EOF'
set -eo pipefail   # no -u: the activate script reads unset variables
if [[ ! -f ~/cvla-env/bin/activate ]]; then
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

echo "== 5/5 Copy the LIBERO assets into the env on $HOST"
rsync -a --partial --info=progress2 --exclude .cache/ "$LIBERO_ASSETS/" "$HOST:cvla-env/lib/python$PYTHON_VERSION/site-packages/libero/libero/assets/"
ssh "$HOST" "test -f ~/cvla-env/lib/python$PYTHON_VERSION/site-packages/libero/libero/assets/scenes/libero_floor_base_style.xml && echo 'LIBERO assets in place'"

echo
echo "Staging done. Next: ssh $HOST, then  cd ~/curious-vla && bash hpc/submit_baseline.sh"
