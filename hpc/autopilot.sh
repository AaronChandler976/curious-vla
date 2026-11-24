#!/bin/bash
# Keep the experiment plan running with no manual steps. Safe to run any number of times (cron
# runs it every 20 minutes on the login node; see the end of this comment).
#   cd ~/curious-vla && bash hpc/autopilot.sh
# For every run in hpc/plan.txt that has no eval yet and is not queued, it submits the run again,
# unless that run has already failed MAX_FAILS times (then it is listed under "needs a look"
# and left alone). Collections that the runs need are submitted first, the same way.
# Jobs whose dependency can never be satisfied (their collection failed) are cancelled so that
# the next pass submits them again behind a new collection.
# It then refreshes runs/results.csv and runs/analysis/, and appends one line to
# runs/logs/autopilot.log.
# Cron entry (crontab -e):
#   */20 * * * * cd $HOME/curious-vla && bash hpc/autopilot.sh >> runs/logs/autopilot.out 2>&1
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh >/dev/null
export PATH="/usr/bin:/bin:$PATH"   # cron has a minimal PATH
mkdir -p runs/logs
exec 9> runs/logs/.autopilot.lock
flock -n 9 || { echo "$(date '+%F %T') another autopilot pass is running"; exit 0; }

MAX_FAILS="${MAX_FAILS:-3}"
PLAN="${PLAN:-hpc/plan.txt}"
suffix="${LIBERO_SUITE}_tasks${LIBERO_TASK_IDS//,/-}"
GPU=(--partition="$SLURM_PARTITION" --gres="$SLURM_GRES" ${SLURM_EXCLUDE:+--exclude="$SLURM_EXCLUDE"})
SINCE=2026-11-24T14:00   # failures before the GPU-busy fix are not counted

# Cancel jobs stuck behind a failed dependency.
squeue -u "$USER" -h -t PD -o "%i %r" | awk '$2=="DependencyNeverSatisfied"{print $1}' | while read -r j; do
  scancel "$j" && echo "$(date '+%F %T') cancelled $j (dependency never satisfied)"
done

queued_id() { squeue -u "$USER" -h -n "$1" -o %i | head -n 1; }
# Failures of a job name: jobs that ended FAILED, TIMEOUT, OUT_OF_MEMORY or NODE_FAIL. A job that
# found a busy GPU also ends FAILED, but it resubmitted itself, so those are not counted.
n_fails() {
  local name=$1 n=0 id st
  while read -r id st; do
    [[ -z "$id" ]] && continue
    if [[ "$st" == FAILED ]] && grep -q "resubmitted as job" "runs/logs/$name-$id.out" 2>/dev/null; then continue; fi
    n=$((n + 1))
  done < <(sacct -u "$USER" -S "$SINCE" -X -n -P -o JobID,JobName%120,State --name="$name" 2>/dev/null |
           awk -F'|' '$3 ~ /^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)/{print $1, $3}')
  echo "$n"
}

submitted=0; done_n=0; queued=0; stuck=()
declare -A COLLECT_DEP
CID=""
ensure_collect() {  # ensure_collect METHOD ESEED: sets CID to the collection job to wait for ("" if the data exists)
  local m=$1 es=$2 esuf="" c
  (( es > 0 )) && esuf="_e$es"
  CID=""
  [[ -f "$RUNS_DIR/exploration/${m}_${suffix}${esuf}/exploration_log.json" ]] && return 0
  local key="$m$esuf"
  [[ -n "${COLLECT_DEP[$key]:-}" ]] && { CID=${COLLECT_DEP[$key]}; return 0; }
  c=$(queued_id "collect-$key")
  if [[ -z "$c" ]]; then
    if (( $(n_fails "collect-$key") >= MAX_FAILS )); then stuck+=("collect-$key"); return 1; fi
    c=$(EXPLORE_METHOD=$m EXPLORE_SEED=$es COLLECT_STEPS=20000 sbatch --parsable --job-name="collect-$key" \
        "${GPU[@]}" hpc/03_collect.sbatch) || return 1
    echo "$(date '+%F %T') submitted collect-$key as $c"
  fi
  COLLECT_DEP[$key]=$c
  CID=$c
}

while read -r kind a b c d e <&3; do
  [[ -z "$kind" || "$kind" == \#* ]] && continue
  dep=()
  if [[ "$kind" == demos ]]; then
    n=$a; s=$b; run="demos_${suffix}_n${n}_s${s}"; vars=(N_DEMOS=$n SEED=$s)
  elif [[ "$kind" == mix ]]; then
    m=$a; bud=$b; n=$c; s=$d; es=${e:-0}; esuf=""; (( es > 0 )) && esuf="_e$es"
    run="mix-$m-${bud}_${suffix}_n${n}_s${s}${esuf}"
    vars=(EXPLORE_METHOD=$m EXPLORE_STEPS=$bud EXPLORE_SEED=$es N_DEMOS=$n SEED=$s)
  else
    echo "unknown plan line: $kind $a $b $c $d $e" >&2; continue
  fi
  if [[ -f "$RUNS_DIR/$run/eval/eval_info.json" ]]; then done_n=$((done_n + 1)); continue; fi
  if [[ -n "$(queued_id "$run")" ]]; then queued=$((queued + 1)); continue; fi
  if (( $(n_fails "$run") >= MAX_FAILS )); then stuck+=("$run"); continue; fi
  if [[ "$kind" == mix ]]; then
    ensure_collect "$m" "$es" || { stuck+=("$run (no exploration data)"); continue; }
    [[ -n "$CID" ]] && dep=(--dependency=afterok:$CID)
  fi
  if [[ -d "$RUNS_DIR/$run" && ! -d "$RUNS_DIR/$run/checkpoints/last" ]]; then
    mv "$RUNS_DIR/$run" "$RUNS_DIR/$run.failed-$(date +%Y%m%d-%H%M%S)"
  fi
  j=$(env "${vars[@]}" sbatch --parsable --job-name="$run" "${GPU[@]}" ${dep[@]+"${dep[@]}"} hpc/02_finetune_demos.sbatch) &&
    { echo "$(date '+%F %T') submitted $run as $j"; submitted=$((submitted + 1)); }
done 3< "$PLAN"

python3 hpc/summarize_results.py "$RUNS_DIR" > /dev/null 2>&1
if [[ -x "$PACKED_ENV_DIR/bin/python" ]]; then
  "$PACKED_ENV_DIR/bin/python" hpc/analyze_results.py "$RUNS_DIR" > /dev/null 2>&1 || echo "analysis failed"
fi
total=$(grep -cvE '^\s*(#|$)' "$PLAN")
running=$(squeue -u "$USER" -h -t R | wc -l)
line="$(date '+%F %T') plan $total: done $done_n, queued $queued (running $running), submitted $submitted, needs a look ${#stuck[@]}"
echo "$line" | tee -a runs/logs/autopilot.log
for x in "${stuck[@]+"${stuck[@]}"}"; do echo "  needs a look: $x" | tee -a runs/logs/autopilot.log; done
[[ $done_n -eq $total ]] && echo "$(date '+%F %T') PLAN COMPLETE" | tee -a runs/logs/autopilot.log
exit 0
