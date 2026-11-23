# Shared settings for the SJSU CoE HPC scripts. Every other script in hpc/ sources this file.
# Edit the values here, not in the individual scripts.

# Where everything lives on the cluster. Home is shared by the login and compute nodes.
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/curious-vla}"
export RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/runs}"
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-curious-vla}"
# HPC3 has no outbound internet, so the env is built on the desktop with conda-pack and
# copied here by stage_from_desktop.sh. If this directory exists, it is used instead of conda.
export PACKED_ENV_DIR="${PACKED_ENV_DIR:-$HOME/cvla-env}"

# Which GPUs to ask Slurm for. Submit from HPC3 (`ssh hpc3`, login node g17). From the probe of
# Nov 22 2026: gpuqs (max 2 days), gpuqm (7 days) and gpuql (14 days) share the same nodes:
# A100s on cs001-cs004, H100s on g16/g18/g19, and P100s (too old for current PyTorch) on g*.
# Always name the GPU type in the gres. If the A100 queue is long, try SLURM_GRES=gpu:h100:1,
# or the NSF H100 queue: SLURM_PARTITION=nsfq SLURM_GRES=gpu:h100:1 (g20-g32, max 24 h per job,
# at most 6 nodes per user). Our jobs ask for 24 h or less, so they fit.
# submit_baseline.sh passes these on the sbatch command line, which overrides the #SBATCH lines.
export SLURM_PARTITION="${SLURM_PARTITION:-gpuqs}"
export SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"

# Pinned versions. lerobot 0.7.0 needs Python >= 3.12 and torch >= 2.7.
export PYTHON_VERSION="3.12"
export LEROBOT_VERSION="0.7.0"
# HPC3's NVIDIA driver is 560.35 (CUDA 12.6), so the cluster env uses the CUDA 12.6 torch
# build, which supports A100 and H100. (The desktop's RTX 5070 Ti would need cu128 instead.)
export TORCH_VERSION="2.9.1"
export TORCHVISION_VERSION="0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"

# Caches. Keep them in home so that the compute nodes see what the login node downloaded.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"

# Models and data on the Hugging Face Hub.
export BASE_POLICY="lerobot/smolvla_base"              # starting point for our fine-tunes
export REFERENCE_POLICY="HuggingFaceVLA/smolvla_libero" # official LIBERO fine-tune, used as a sanity check
export DEMO_DATASET="HuggingFaceVLA/libero"            # LIBERO demos in LeRobot format

# The MVP experiment: 1-2 tasks from one suite. Task ids are indices inside the suite (0-9).
export LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
export LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-0,1}"

# Headless rendering for MuJoCo. If EGL fails on the compute node, 00_check.sbatch says so;
# then try MUJOCO_GL=osmesa (slower, CPU rendering).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

# HPC3 (g17) has no internet access (probe of Nov 22 2026: huggingface.co and pypi.org time
# out), so the Hugging Face libraries must only read the cache that stage_from_desktop.sh
# copied over. Set HPC_OFFLINE=0 on a machine with internet.
if [[ "${HPC_OFFLINE:-1}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# SmolVLA builds its backbone from this repo even when it loads a fine-tuned checkpoint,
# so it has to be in the cache too.
# smolvla_base uses the Video-Instruct backbone; HuggingFaceVLA/smolvla_libero uses the plain
# Instruct one (found by the dry run, Nov 23 2026). Space-separated.
export VLM_BACKBONES="HuggingFaceTB/SmolVLM2-500M-Video-Instruct HuggingFaceTB/SmolVLM2-500M-Instruct"

# Offline, the jobs load everything from local paths, so nothing asks the Hub which
# revision is current. Run these after activate_env.
# SmolVLA reads the backbone's config, tokenizer and processor files, and its weights if the
# checkpoint sets load_vlm_weights. The 25 ONNX exports in that repo are never used, so
# staging and the offline check skip them.
export VLM_BACKBONE_FILES='*.json,*.txt,*.jinja,*.model,model.safetensors'

hf_local_model() {  # hf_local_model REPO [PATTERNS]: prints the cached snapshot dir, or fails
  python - "$1" "${2:-}" <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
repo = sys.argv[1]
kw = {"allow_patterns": sys.argv[2].split(",")} if sys.argv[2] else {}
try:
    print(snapshot_download(repo, local_files_only=True, **kw))
    sys.exit(0)
except Exception as e:
    err = e
# The cache needs refs/main to find the snapshot by name. A copied cache can lack it, so
# if there is exactly one snapshot, point refs/main at it (transformers needs this too).
d = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
snaps = sorted(os.listdir(os.path.join(d, "snapshots"))) if os.path.isdir(os.path.join(d, "snapshots")) else []
if len(snaps) == 1:
    os.makedirs(os.path.join(d, "refs"), exist_ok=True)
    with open(os.path.join(d, "refs", "main"), "w") as f:
        f.write(snaps[0])
    print(snapshot_download(repo, local_files_only=True, **kw))
else:
    sys.exit(f"{repo}: not usable offline ({err}); cache dir {d} has snapshots {snaps}")
PYEOF
}
dataset_root() {  # prints the cached snapshot dir of the demo dataset, or fails
  local base="$HF_LEROBOT_HOME/hub/datasets--${DEMO_DATASET//\//--}/snapshots"
  local best="" n=0 c d
  for d in "$base"/*/; do
    [[ -f "$d/meta/info.json" ]] || continue
    c=$(find -L "$d" -type f | wc -l)
    if (( c > n )); then n=$c; best=${d%/}; fi
  done
  [[ -n "$best" ]] || { echo "ERROR: no cached copy of $DEMO_DATASET under $base" >&2; return 1; }
  echo "$best"
}

activate_env() {
  # Jobs use the env that stage_from_desktop.sh unpacked into $PACKED_ENV_DIR. A conda
  # install on the cluster is only a fallback, for machines set up with setup_env.sh.
  # Activation scripts touch unset variables, so `set -u` is paused around them.
  # shellcheck disable=SC1091
  set +u
  if [[ -f "$PACKED_ENV_DIR/bin/activate" ]]; then
    source "$PACKED_ENV_DIR/bin/activate"
  elif [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    set -u
    echo "ERROR: no Python env found on $(hostname)." >&2
    echo "  Expected the packed env at $PACKED_ENV_DIR/bin/activate." >&2
    echo "  If ~/cvla-env.tar.gz exists, unpack it on the login node with:" >&2
    echo "    mkdir -p ~/cvla-env && tar -xzf ~/cvla-env.tar.gz -C ~/cvla-env && ~/cvla-env/bin/conda-unpack" >&2
    echo "  Otherwise re-run hpc/stage_from_desktop.sh on the desktop." >&2
    return 1
  fi
  set -u
  patch_env
  echo "Python env: $(command -v python)"
}

patch_env() {
  # robosuite logs to the hard-coded file /tmp/robosuite.log. On a shared node that file can
  # belong to another user, and then importing LIBERO fails with PermissionError. Point it
  # at a file in our home instead. Safe to run every time: it only edits once.
  local f
  f=$(python -c "import importlib.util as u, os; s=u.find_spec('robosuite'); print(os.path.join(os.path.dirname(s.origin), 'utils', 'log_utils.py') if s else '')" 2>/dev/null)
  if [[ -n "$f" && -f "$f" ]] && grep -q '"/tmp/robosuite.log"' "$f"; then
    sed -i 's|"/tmp/robosuite.log"|__import__("os").path.expanduser("~/.robosuite.log")|' "$f"
    echo "Patched robosuite log path in $f"
  fi
}

# No camera rename map: make_base_policy.py gives the starting policy the same feature names
# and shapes as HuggingFaceVLA/smolvla_libero (cameras image/image2, 8-dim state, 7-dim
# action), which are also the dataset's and the LIBERO env's names.
