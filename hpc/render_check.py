"""Check that LIBERO renders on this node: build one task scene the way the eval does (LeRobot's
LIBERO wrapper, EGL), reset it and take a few steps. Exits non-zero if that fails.

Some nsfq nodes (g25, g29 on Nov 25 2026) train fine but abort in the eval's first render, after
the whole fine-tune. The fine-tune job runs this first (in a subprocess with a timeout), so such a
node costs a minute instead of a run.

  python hpc/render_check.py --suite libero_object --task-id 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_exploration import LiberoScene  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    a = p.parse_args()
    scene = LiberoScene(a.suite, a.task_id, 256, 256, seed=0)
    frame, _ = scene.reset()
    for _ in range(5):
        frame, _, _ = scene.step(np.zeros(7))
    img = frame["observation.images.image"]
    scene.close()
    if np.asarray(img).std() == 0:
        sys.exit("render check: blank image")
    print(f"render check ok: image {np.asarray(img).shape}")


if __name__ == "__main__":
    main()
