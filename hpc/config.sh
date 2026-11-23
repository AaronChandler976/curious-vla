# Shared settings for the SJSU CoE HPC scripts. Every other script in hpc/ sources this file.
# Edit the values here, not in the individual scripts.

# Where everything lives on the cluster. Home is shared by the login and compute nodes.
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/curious-vla}"
export RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/runs}"
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-curious-vla}"

# Which GPUs to ask Slurm for. Run these jobs from HPC3 (`ssh hpc3`, node g17), which has
# A100s; HPC1 (`ssh hpc`) only has P100s, which are too old for current PyTorch.
# The partition name on HPC3 is not confirmed yet: run hpc/probe_cluster.sh and set it here.
# submit_baseline.sh passes these on the sbatch command line, which overrides the #SBATCH lines.
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
export SLURM_GRES="${SLURM_GRES:-gpu:1}"

# Pinned versions. lerobot 0.7.0 needs Python >= 3.12 and torch >= 2.7.
export PYTHON_VERSION="3.12"
export LEROBOT_VERSION="0.7.0"

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

# Set HPC_OFFLINE=1 if the compute nodes have no internet access. Run setup_env.sh on the
# login node first, so that the models and the dataset are already in the cache.
if [[ "${HPC_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

activate_env() {
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

# lerobot/smolvla_base was pretrained with cameras named camera1/camera2/camera3, while the
# LIBERO dataset and env use image (agent view) and image2 (wrist). This maps one to the other
# for fine-tuning and for evaluating our fine-tunes. Not yet verified on a real run: if
# training stops with a missing-feature error, this map is the first thing to check.
export FINETUNE_RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
