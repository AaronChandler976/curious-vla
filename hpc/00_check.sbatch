#!/bin/bash
# Quick GPU node check (about 10 minutes). Run it before any long job:
#   cd ~/curious-vla && sbatch hpc/00_check.sbatch
# Checks: which GPU you got, that PyTorch can use it, and that LIBERO renders headless.
#SBATCH --job-name=cvla-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
source hpc/config.sh
activate_env

echo "Node: $(hostname)   Job: ${SLURM_JOB_ID:-none}"
nvidia-smi

python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch cannot see a GPU (driver too old for this torch build?)"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.1f} GiB")
print("torch supports archs:", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok:", float((x @ x).mean()))
if cap < (7, 0):
    print("WARNING: this GPU is too old for this project (likely a P100). Ask for an A100/H100 node.")
EOF

python - <<EOF
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
task = benchmark.get_benchmark_dict()["$LIBERO_SUITE"]().get_task(int("$LIBERO_TASK_IDS".split(",")[0]))
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
obs = env.reset()
print("LIBERO task:", task.language)
print("render ok with MUJOCO_GL=$MUJOCO_GL, agentview image", obs["agentview_image"].shape)
env.close()
EOF
echo "CHECK PASSED"
