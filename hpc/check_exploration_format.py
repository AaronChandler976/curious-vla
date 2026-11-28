"""Check that exploration data uses the same conventions as the demos before training on it.

This guards against the silent-failure mode from the proposal (data that looks fine but uses
another state/action/image convention). For each task scene it compares the first frame of the
exploration episodes with the first frame of that task's demos:
  - features: same keys, dtypes and shapes as the demo dataset;
  - state: the mean first-frame state should be within 3 demo standard deviations per dimension
    (both start from LIBERO's initial-state distribution for that task);
  - images: the mean first image should correlate better with the demos' mean first image than
    its 180-degree rotation does (catches a missing or doubled flip);
  - actions: within [-1, 1] and gripper in {-1, +1}, like the demos.

  python hpc/check_exploration_format.py --demo-root R --explore-root E --suite libero_object --task-ids 0,1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import read_episodes, to_array  # noqa: E402


def first_frames(root, episodes, keys, features, limit=10):
    """First frame of each episode, read from the raw data rows (not through meta/episodes)."""
    rows = read_episodes(root, list(episodes[:limit]), ["episode_index", "frame_index", "task_index", *keys])
    return {k: np.stack([to_array(rows[e][k][0], features[k]) for e in episodes[:limit]]) for k in keys}


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-repo", default="HuggingFaceVLA/libero")
    p.add_argument("--demo-root", type=Path, required=True)
    p.add_argument("--explore-root", type=Path, required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task-ids", required=True)
    p.add_argument("--demo-tasks", default=None, help="test only: '|'-separated demo task strings")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    demo = LeRobotDataset(args.demo_repo, root=args.demo_root)
    explore = LeRobotDataset(f"local/{args.explore_root.name}", root=args.explore_root)
    log = json.loads((args.explore_root / "exploration_log.json").read_text())
    task_ids = [int(t) for t in args.task_ids.split(",")]
    if args.demo_tasks:
        languages = dict(zip(task_ids, args.demo_tasks.split("|")))
    else:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[args.suite]()
        languages = {t: suite.get_task(t).language for t in task_ids}

    problems = []
    ft_d, ft_e = demo.meta.features, explore.meta.features
    for k, v in ft_d.items():
        if k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        if k not in ft_e:
            problems.append(f"feature {k} missing from exploration data")
        elif v["dtype"] != ft_e[k]["dtype"] or list(v["shape"]) != list(ft_e[k]["shape"]):
            problems.append(f"feature {k}: demo {v['dtype']} {v['shape']} vs exploration {ft_e[k]['dtype']} {ft_e[k]['shape']}")
    print(f"features: {'OK' if not problems else 'MISMATCH'}")

    img_keys = [k for k, v in ft_d.items() if v["dtype"] in ("video", "image")]
    norm = lambda s: " ".join(s.lower().split())
    for t in task_ids:
        lang = norm(languages[t])
        demo_eps = [int(r["episode_index"]) for r in demo.meta.episodes if any(norm(x) == lang for x in r["tasks"])]
        exp_eps = [e["episode"] for e in log["episodes_log"] if e["task_id"] == t]
        if not demo_eps or not exp_eps:
            problems.append(f"task {t}: {len(demo_eps)} demo and {len(exp_eps)} exploration episodes")
            continue
        keys = ["observation.state", *img_keys]
        d = first_frames(args.demo_root, demo_eps, keys, ft_d)
        e = first_frames(args.explore_root, exp_eps, keys, ft_d)
        mu, sd = d["observation.state"].mean(0), d["observation.state"].std(0) + 1e-3
        z = np.abs(e["observation.state"].mean(0) - mu) / sd
        ok_state = bool((z < 3).all())
        print(f"task {t} state: max |z| = {z.max():.2f} over dims {np.round(z, 2).tolist()}  {'OK' if ok_state else 'MISMATCH'}")
        if not ok_state:
            problems.append(f"task {t}: first-frame state differs from demos (max z {z.max():.1f})")
        for k in img_keys:
            md, me = d[k].mean(0), e[k].mean(0)
            same, flipped = corr(me, md), corr(me[::-1, ::-1, :], md)  # images are H, W, C here
            tie = abs(same - flipped) < 1e-3  # symmetric images: orientation cannot be told apart
            ok_img = same > flipped or tie
            verdict = "UNDECIDED (symmetric images)" if tie else ("OK" if ok_img else "MISMATCH")
            print(f"task {t} {k}: corr {same:.3f} as stored vs {flipped:.3f} rotated 180  {verdict}")
            if not ok_img:
                problems.append(f"task {t} {k}: exploration images look rotated relative to demos")

    acts = np.asarray(explore.hf_dataset.with_format("numpy")["action"])
    grip = np.unique(np.round(acts[:, -1], 3))
    ok_act = bool(np.abs(acts).max() <= 1.0 + 1e-6 and set(grip.tolist()) <= {-1.0, 1.0})
    print(f"actions: range [{acts.min():.2f}, {acts.max():.2f}], gripper values {grip.tolist()}  {'OK' if ok_act else 'MISMATCH'}")
    if not ok_act:
        problems.append("exploration actions outside [-1, 1] or gripper not in {-1, +1}")

    if problems:
        print("FORMAT CHECK FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("FORMAT CHECK PASSED")


if __name__ == "__main__":
    main()
