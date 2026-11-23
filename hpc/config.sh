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
fi

activate_env() {
  # shellcheck disable=SC1091
  if [[ -f "$PACKED_ENV_DIR/bin/activate" ]]; then
    source "$PACKED_ENV_DIR/bin/activate"
  else
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
}

# lerobot/smolvla_base was pretrained with cameras named camera1/camera2/camera3, while the
# LIBERO dataset and env use image (agent view) and image2 (wrist). This maps one to the other
# for fine-tuning and for evaluating our fine-tunes. Not yet verified on a real run: if
# training stops with a missing-feature error, this map is the first thing to check.
export FINETUNE_RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
