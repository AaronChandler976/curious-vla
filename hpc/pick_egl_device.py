"""Find a rendering setup that works in this job, by running the real collector for a few steps.

The ICM collection hung inside robosuite's EGL camera render on its first reset (one core busy,
no frame for hours), while other jobs rendered fine. robosuite renders on the EGL device whose
index is CUDA_VISIBLE_DEVICES, and inside a Slurm job that need not be the GPU the job owns. So
this tries, in order: the default; each EGL device (MUJOCO_EGL_DEVICE_ID=i, with
CUDA_VISIBLE_DEVICES=i so robosuite's check passes; the collector never uses CUDA); then OSMesa
(CPU rendering, slow but safe). Each try runs collect_exploration.py with random actions for 10
steps in a subprocess with a timeout, so a hang costs at most the timeout.

Prints shell exports for the first setup that works, e.g.
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3
  eval "export $(python hpc/pick_egl_device.py --demo-root R --suite S --task-ids 0,1)"
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def egl_device_count(env) -> int:
    code = "from mujoco.egl import egl_ext as E; print(len(E.eglQueryDevicesEXT()))"
    try:
        out = subprocess.run([sys.executable, "-c", code], env={**env, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
                             capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else 0
    except Exception:
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-root", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task-ids", required=True)
    p.add_argument("--timeout", type=int, default=150)
    p.add_argument("--fake-env", action="store_true", help="test only")
    args = p.parse_args()

    base = dict(os.environ)
    n = egl_device_count(base)
    print(f"CUDA_VISIBLE_DEVICES={base.get('CUDA_VISIBLE_DEVICES')}, {n} EGL devices", file=sys.stderr)
    tries = [("default", {"MUJOCO_GL": "egl"})]
    tries += [(f"EGL device {i}", {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(i), "CUDA_VISIBLE_DEVICES": str(i)})
              for i in range(n) if str(i) != base.get("CUDA_VISIBLE_DEVICES")]  # that one is the default
    tries += [("OSMesa (CPU)", {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})]

    tmp = Path(tempfile.mkdtemp(prefix="render_probe_"))
    try:
        for name, extra in tries:
            env = {**base, **extra}
            if extra.get("MUJOCO_GL") == "egl":
                env.pop("PYOPENGL_PLATFORM", None)
            cmd = [sys.executable, str(HERE / "collect_exploration.py"), "--method", "random", "--suite", args.suite,
                   "--task-ids", args.task_ids.split(",")[0], "--steps", "10", "--episode-len", "10",
                   "--demo-root", args.demo_root, "--out", str(tmp / "probe")] + (["--fake-env"] if args.fake_env else [])
            t0 = time.time()
            try:
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  {name}: no result within {args.timeout} s (hangs)", file=sys.stderr)
                continue
            if out.returncode != 0:
                last = [l for l in out.stderr.strip().splitlines() if l.strip()][-1:] or ["no output"]
                print(f"  {name}: failed ({last[0][:150]})", file=sys.stderr)
                continue
            print(f"  {name}: works ({time.time() - t0:.0f} s for setup + 10 steps)", file=sys.stderr)
            print(" ".join(f"{k}={v}" for k, v in extra.items()))
            return
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit("no rendering setup works in this job")


if __name__ == "__main__":
    main()
