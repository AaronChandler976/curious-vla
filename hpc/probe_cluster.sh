#!/bin/bash
# Read-only look at a cluster, to fill in the unconfirmed settings in config.sh.
# It changes nothing and submits no jobs. Run it from your desktop without cloning anything:
#   curl -fsSL https://raw.githubusercontent.com/AaronChandler976/curious-vla/hpc-scripts/hpc/probe_cluster.sh \
#     | ssh hpc3 'bash -s' > hpc3_probe.txt
# and send me hpc3_probe.txt (or paste it).
section() { echo; echo "===== $* ====="; }

section "host"
hostname; uname -r; cat /etc/os-release 2>/dev/null | head -2
section "partitions (name, limit, nodes, state, gres)"
sinfo -o "%P %l %D %t %G %N" 2>&1
section "GPU types per node"
sinfo -N -o "%N %P %G %m" 2>&1 | sort -u | head -40
section "partition details"
scontrol show partition 2>&1 | grep -E "PartitionName|MaxTime|DefaultTime|AllowAccounts|AllowQos|State"
section "my associations / QOS"
sacctmgr -n show assoc user="$USER" format=account,partition,qos 2>&1 | head
section "queue right now"
squeue -o "%P %T %D %b" 2>&1 | sort | uniq -c | sort -rn | head -20
section "modules (cuda, python, conda, singularity/apptainer)"
(module avail 2>&1 || true) | grep -iE "cuda|python|conda|anaconda|singularity|apptainer" | head -20
command -v apptainer singularity 2>/dev/null
section "home disk and quota"
df -h "$HOME" 2>&1 | tail -1
quota -s 2>/dev/null || echo "(no quota command)"
du -sh "$HOME" 2>/dev/null
section "internet from this node"
curl -sS -o /dev/null -w "huggingface.co HTTP %{http_code}\n" --max-time 10 https://huggingface.co || echo "huggingface.co unreachable"
curl -sS -o /dev/null -w "pypi.org HTTP %{http_code}\n" --max-time 10 https://pypi.org/simple/ || echo "pypi.org unreachable"
section "NVIDIA driver (only if this node has a GPU)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
