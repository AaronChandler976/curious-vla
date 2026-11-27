"""Training cost per run from the fine-tune logs: GPU type, seconds per step, peak memory, samples/s.

  python3 hpc/train_profile.py runs/logs  [> runs/analysis/train_profile.csv]

Uses the last LeRobot training log line of each finished 20k-step run (step_s, mem_gb, smp/s are
LeRobot's own meters) and the GPU name that gpu_report printed at the start of the job.
"""
import re, sys, statistics as st
from pathlib import Path
from collections import defaultdict

rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/logs").glob("*.out")):
    if not re.match(r"(demos|mix-|seq-)", f.name):
        continue
    t = f.read_text(errors="ignore")
    steps = re.findall(r"step:(\d+)K .*?step_s:([\d.]+) smp/s:([\d.]+) mem_gb:([\d.]+)", t)
    gpu = re.search(r"^\d+, (NVIDIA [^,]+),", t, re.M)
    if not steps or not gpu:
        continue
    k, s, sps, mem = steps[-1]
    rows.append((f.name.rsplit("-", 1)[0], gpu.group(1).strip(), int(k), float(s), float(sps), float(mem)))
print("run,gpu,last_step_k,step_s,samples_per_s,mem_gb")
for r in rows:
    print(",".join(map(str, r)))
by = defaultdict(list)
for r in rows:
    if r[2] >= 19:
        by[r[1]].append(r)
for g, rs in by.items():
    print(f"# {g}: {len(rs)} runs, step {st.median(r[3] for r in rs):.3f} s, {st.median(r[4] for r in rs):.0f} samples/s, "
          f"peak mem {max(r[5] for r in rs):.2f} GB, 20k steps = {st.median(r[3] for r in rs) * 20000 / 3600:.2f} h", file=sys.stderr)
