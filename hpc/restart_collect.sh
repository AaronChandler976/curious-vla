#!/bin/bash
# Restart one exploration collection and re-point its waiting mixed fine-tunes at the new job.
#   ssh hpc3 'cd ~/curious-vla && bash hpc/restart_collect.sh icm'
# Prints why the previous collection stopped, submits a new one, updates the dependencies of the
# pending mix-<method>-* jobs (found by name, no job ids to type), then follows the new job for
# up to 30 minutes after it starts and prints the GPU it picked and its progress.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh
m="${1:?usage: restart_collect.sh random|icm}"
me="${USER:-$(whoami)}"

prev=$(ls -t runs/logs/collect-$m-*.out 2>/dev/null | head -1)
if [[ -n "$prev" ]]; then
  echo "=== previous log: $prev"
  grep -v -i -E "warn|torchcodec|it/s\]" "$prev" | tail -n 25
fi
echo "=== recent collect-$m jobs"
sacct -u "$me" -S "$(date -d '1 day ago' +%F)" -X -n -o JobID,JobName%20,State,Elapsed,ExitCode | grep "collect-$m"
for j in $(squeue -u "$me" -h -n "collect-$m" -o %i); do echo "cancelling running $j"; scancel "$j"; done
sleep 5

new=$(EXPLORE_METHOD=$m COLLECT_STEPS=20000 sbatch --parsable --job-name="collect-$m" \
      --partition="$SLURM_PARTITION" --gres="$SLURM_GRES" hpc/03_collect.sbatch) || { echo "sbatch failed"; exit 1; }
echo "=== new collection: $new"
for j in $(squeue -u "$me" -h -o "%i %j" | awk -v p="^mix-$m-" '$2 ~ p {print $1}'); do
  scontrol update JobId="$j" Dependency="afterok:$new" && echo "  $j now waits for $new"
done

log="runs/logs/collect-$m-$new.out"
echo "=== waiting for $new to start"
for i in $(seq 1 40); do   # up to 20 minutes in the queue
  [[ -f "$log" || -z "$(squeue -h -j "$new" 2>/dev/null)" ]] && break
  sleep 30
done
[[ -f "$log" ]] || { echo "still queued after 20 min; check later with: grep -E 'EGL|episode' $log"; exit 0; }
for i in $(seq 1 60); do   # up to 30 minutes (the render probe alone can take ~15)
  grep -q -E "^\[$m\] episode|no rendering setup works|Timeout \(" "$log" 2>/dev/null && break
  [[ -z "$(squeue -h -j "$new" 2>/dev/null)" ]] && break
  sleep 30
done
echo "=== $log"
grep -E "CUDA_VISIBLE|EGL device|default:|OSMesa|Rendering with|^\[$m\] episode|Timeout|Error|Traceback|no NVIDIA|cannot list" "$log" | tail -n 15
echo "=== state: $(sacct -j "$new" -X -n -o State,Elapsed | head -1)"
