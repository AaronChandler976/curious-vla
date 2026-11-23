"""Build one training dataset: the selected demos plus an exploration-data budget, mixed ~50/50.

- Demos: the selected demo episodes, cut out of the full demo dataset.
- Exploration: the episodes from the start of an exploration run, up to the env-step budget
  (so the 5k-step condition is the first 5k steps of the same run as the 20k condition).
- Ratio: the demo episodes are repeated k = max(1, round(E / D)) times, where D and E are the
  demo and exploration frame counts, so demos are about half of all frames at every budget.
- Normalization: meta/stats.json is replaced by the full demo dataset's stats, so every
  condition (including demos only) normalizes states and actions identically.

  python hpc/build_mix.py --demo-root R --episodes '[1,2]' --explore-root E --budget 5000 --out OUT
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-repo", default="HuggingFaceVLA/libero")
    p.add_argument("--demo-root", type=Path, required=True)
    p.add_argument("--episodes", required=True, help="JSON list of demo episode indices")
    p.add_argument("--explore-root", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True, help="exploration env steps to use")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from episode_meta import repair

    work = args.out.parent / (args.out.name + ".parts")
    for d in (args.out, work):
        if d.exists():
            shutil.rmtree(d)
    work.mkdir(parents=True)

    keep = set(json.loads(args.episodes))
    demo = LeRobotDataset(args.demo_repo, root=args.demo_root)
    repair(demo)
    drop = [e for e in range(demo.meta.total_episodes) if e not in keep]
    demo_sub = delete_episodes(demo, drop, output_dir=work / "demos", repo_id="local/demos")

    log = json.loads((args.explore_root / "exploration_log.json").read_text())
    use = [e["episode"] for e in log["episodes_log"] if e["env_steps_after"] <= args.budget]
    if not use:
        raise SystemExit(f"no exploration episodes fit in a budget of {args.budget} steps")
    explore = LeRobotDataset(f"local/{args.explore_root.name}", root=args.explore_root)
    repair(explore)
    drop = [e for e in range(explore.meta.total_episodes) if e not in set(use)]
    explore_sub = delete_episodes(explore, drop, output_dir=work / "explore", repo_id="local/explore") if drop else explore

    D, E = demo_sub.meta.total_frames, explore_sub.meta.total_frames
    k = max(1, round(E / D))
    merged = merge_datasets([demo_sub] * k + [explore_sub], output_repo_id=f"local/{args.out.name}",
                            output_dir=args.out, concatenate_videos=False)

    demo_stats = json.loads((args.demo_root / "meta" / "stats.json").read_text())
    merged_stats_path = args.out / "meta" / "stats.json"
    merged_stats = json.loads(merged_stats_path.read_text())
    for key in merged_stats:
        if key in demo_stats:
            merged_stats[key] = demo_stats[key]
    merged_stats_path.write_text(json.dumps(merged_stats, indent=2))

    info = {"demo_episodes": sorted(keep), "demo_frames": D, "demo_repeats": k,
            "explore_method": log["method"], "explore_budget_steps": args.budget,
            "explore_episodes": len(use), "explore_frames": E,
            "demo_fraction_of_frames": round(k * D / (k * D + E), 3),
            "total_frames": merged.meta.total_frames, "total_episodes": merged.meta.total_episodes}
    (args.out / "mix_info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(work)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
