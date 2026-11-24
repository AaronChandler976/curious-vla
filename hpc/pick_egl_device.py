"""Pick the EGL device to render on, and print its index for MUJOCO_EGL_DEVICE_ID.

robosuite renders on the EGL device whose index equals CUDA_VISIBLE_DEVICES. Inside a Slurm job
that is not necessarily the GPU the job was given: on cs002 the ICM collection rendered on a
device where every frame hung (one CPU core busy, no frame for hours). This tries each EGL device
in a subprocess with a timeout, renders a few frames, and picks the fastest NVIDIA one.

  export MUJOCO_EGL_DEVICE_ID=$(python hpc/pick_egl_device.py)
"""

import os
import subprocess
import sys

PROBE = r"""
import time, mujoco
from OpenGL import GL
xml = '<mujoco><worldbody><light pos="0 0 2"/><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 256, 256)
r._gl_context.make_current()
name = GL.glGetString(GL.GL_RENDERER).decode()
t = time.time()
for _ in range(20):
    mujoco.mj_forward(m, d); r.update_scene(d); r.render()
print(f"{(time.time() - t) / 20:.5f}|{name}")
"""


def main() -> None:
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    count = subprocess.run([sys.executable, "-c",
                            "from mujoco.egl import egl_ext as E; print(len(E.eglQueryDevicesEXT()))"],
                           env=env, capture_output=True, text=True, timeout=60)
    if count.returncode != 0:
        sys.exit(f"cannot list EGL devices:\n{count.stderr[-2000:]}")
    n = int(count.stdout.strip().splitlines()[-1])
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, {n} EGL devices", file=sys.stderr)
    results = []
    for i in range(n):
        try:
            out = subprocess.run([sys.executable, "-c", PROBE], env={**env, "MUJOCO_EGL_DEVICE_ID": str(i)},
                                 capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            print(f"  EGL device {i}: no frame within 90 s", file=sys.stderr)
            continue
        if out.returncode != 0:
            print(f"  EGL device {i}: failed ({out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'no output'})",
                  file=sys.stderr)
            continue
        sec, name = out.stdout.strip().splitlines()[-1].split("|", 1)
        print(f"  EGL device {i}: {name}, {float(sec) * 1000:.1f} ms per frame", file=sys.stderr)
        results.append((float(sec), i, name))
    nvidia = [r for r in results if "nvidia" in r[2].lower()]
    if not nvidia:
        sys.exit("no NVIDIA EGL device renders here (only: " + ", ".join(r[2] for r in results) + ")")
    sec, i, name = min(nvidia)
    print(f"using EGL device {i} ({name})", file=sys.stderr)
    print(i)


if __name__ == "__main__":
    main()
