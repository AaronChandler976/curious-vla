#!/bin/bash
# One-screen summary of the queue, recent job logs and finished results. Read-only.
#   ssh hpc3 'cd ~/curious-vla && bash hpc/status.sh'
cd "$(dirname "$(readlink -f "$0")")/.."
source hpc/config.sh >/dev/null
echo "===== queue"
squeue -u "$USER" -o "%.8i %.9P %.22j %.2t %.10M %R"
echo
echo "===== jobs from the last 2 days (sacct)"
sacct -u "$USER" -S "$(date -d '2 days ago' +%F)" -X -o JobID,JobName%22,State,Elapsed,ExitCode 2>/dev/null | tail -n 30
echo
echo "===== last error line of each failed or still-running log (newest 12)"
for f in $(ls -t runs/logs/*.out 2>/dev/null | head -12); do
  last=$(grep -E "Error|error:|Traceback|CANCELLED|DUE TO TIME" "$f" | tail -1)
  printf '%-40s %s\n' "$(basename "$f")" "${last:-ok so far: $(tail -1 "$f" | cut -c1-90)}"
done
echo
echo "===== results"
python3 hpc/summarize_results.py "$RUNS_DIR" 2>/dev/null || echo "(no finished evals yet)"
