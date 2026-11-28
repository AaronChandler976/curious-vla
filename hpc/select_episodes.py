"""Pick a reduced demo budget: N demo episodes for each chosen LIBERO task.

Prints a JSON list of episode indices that `lerobot-train --dataset.episodes=...` accepts, and
writes the selection with its task names to --out so that every run records which demos it used.

Example:
  python hpc/select_episodes.py --suite libero_object --task-ids 0,1 --n-demos 10 --seed 0
"""

import argparse
import contextlib
import json
import random
import re
import sys
from pathlib import Path


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceVLA/libero")
    p.add_argument("--root", default=None, help="local copy of the dataset (needed offline)")
    p.add_argument("--suite", required=True)
    p.add_argument("--task-ids", required=True, help="comma-separated ids inside the suite, e.g. 0,1")
    p.add_argument("--n-demos", type=int, required=True, help="demos per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    # LIBERO and LeRobot print info lines to stdout. The calling script reads stdout as the
    # episode list, so everything except the final JSON line goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from libero.libero import benchmark
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        suite = benchmark.get_benchmark_dict()[args.suite]()
        task_ids = [int(t) for t in args.task_ids.split(",")]
        wanted = {norm(suite.get_task(t).language): t for t in task_ids}

        meta = LeRobotDatasetMetadata(args.dataset, root=args.root)
    by_task: dict[int, list[int]] = {t: [] for t in task_ids}
    for row in meta.episodes:
        for task_text in row["tasks"]:
            t = wanted.get(norm(task_text))
            if t is not None:
                by_task[t].append(int(row["episode_index"]))

    # The same instruction can occur in more than one suite (e.g. libero_90 and libero_10),
    # so report counts and fail loudly instead of silently using the wrong demos.
    rng = random.Random(args.seed)
    chosen: list[int] = []
    summary = {}
    for t, eps in by_task.items():
        lang = suite.get_task(t).language
        print(f"task {t} '{lang}': {len(eps)} demos in dataset", file=sys.stderr)
        if len(eps) < args.n_demos:
            sys.exit(f"Only {len(eps)} demos found for task {t}, asked for {args.n_demos}.")
        pick = sorted(rng.sample(sorted(eps), args.n_demos))
        chosen += pick
        summary[t] = {"language": lang, "episodes": pick}

    chosen.sort()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"suite": args.suite, "n_demos": args.n_demos,
                                        "seed": args.seed, "tasks": summary}, indent=2))
    print(json.dumps(chosen).replace(" ", ""))


if __name__ == "__main__":
    main()
