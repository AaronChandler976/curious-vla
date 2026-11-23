"""Write one training dataset as a fresh local LeRobot dataset: the selected demos, optionally
mixed ~50/50 with the first BUDGET env steps of an exploration run. Every condition, including
demos only, is trained from a dataset built by this script, so all conditions share one data path.

Why not LeRobot's delete_episodes/merge_datasets: they trust the per-episode metadata of the
source dataset, and the cached HuggingFaceVLA/libero snapshot has data-file pointers in
meta/episodes that do not match its data files (delete_episodes found no rows: KeyError 0).
merge_datasets also rejects the demo and exploration data for a cosmetic difference in their
feature dicts. This script only reads the raw data parquet rows (by their episode_index and
frame_index columns) and writes everything new through LeRobotDataset.create/add_frame, so the
output's metadata is consistent by construction.

- Demos: the episodes in --episodes, each written k = max(1, round(E / D)) times (D, E: demo and
  exploration frame counts), so demos are about half of all frames at every budget; k = 1 with
  no exploration. Each demo's instruction must match the LIBERO language in --selection.
- Exploration: the episodes from the start of the run up to --budget env steps (the 5k
  condition is the first 5k steps of the same run as the 20k condition).
- Normalization: meta/stats.json is replaced by the full demo dataset's stats, which is what
  training on the demo dataset directly would use, so every condition normalizes identically.
- sources.json maps every output episode to the source episode it was copied from; the
  verification (verify_training_data.py) uses it.

  python hpc/build_dataset.py --demo-root R --selection SEL.json [--explore-root E --budget 5000] --out OUT
"""

import argparse
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_tasks(root: Path) -> dict[int, str]:
    """task_index -> instruction, from meta/tasks.parquet (v3) or meta/tasks.jsonl (v2)."""
    import pandas as pd

    pq_path, jl_path = root / "meta" / "tasks.parquet", root / "meta" / "tasks.jsonl"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        if "task" in df.columns:
            return {int(i): str(t) for i, t in zip(df["task_index"], df["task"])}
        return {int(i): str(t) for t, i in zip(df.index, df["task_index"])}
    rows = [json.loads(line) for line in jl_path.read_text().splitlines() if line.strip()]
    return {int(r["task_index"]): r["task"] for r in rows}


def read_episodes(root: Path, episodes: list[int], columns: list[str]) -> dict[int, dict[str, list]]:
    """Raw rows of the given episodes, straight from the data parquet files, ordered by frame_index."""
    import pyarrow.parquet as pq

    want = set(episodes)
    files = [str(p) for p in sorted((root / "data").glob("*/*.parquet"))
             if want & set(pq.read_table(p, columns=["episode_index"]).column(0).to_pylist())]
    if not files:
        raise SystemExit(f"none of the episodes {episodes[:5]}... are in {root}/data")
    table = pads.dataset(files, format="parquet").to_table(
        columns=columns, filter=pads.field("episode_index").isin(episodes))
    df = table.to_pandas()
    out = {}
    for e in episodes:
        g = df[df["episode_index"] == e].sort_values("frame_index")
        if len(g) == 0:
            raise SystemExit(f"episode {e}: no rows in {root}/data")
        fi = g["frame_index"].to_numpy()
        if not np.array_equal(fi, np.arange(len(g))):
            raise SystemExit(f"episode {e}: frame_index is not 0..{len(g) - 1} in {root}/data")
        out[e] = {c: g[c].tolist() for c in columns}
    return out


def to_array(value, feat: dict) -> np.ndarray:
    if feat["dtype"] in ("image", "video"):
        from PIL import Image

        if isinstance(value, dict):
            value = value.get("bytes") or Path(value["path"]).read_bytes()
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value, dtype=feat["dtype"]).reshape(feat["shape"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True, help="runs/selections/<run>.json from select_episodes.py")
    p.add_argument("--explore-root", type=Path, default=None)
    p.add_argument("--budget", type=int, default=0, help="exploration env steps to use")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--writer-threads", type=int, default=8)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((args.demo_root / "meta" / "info.json").read_text())
    # Only dtype/shape/names: the demo info.json carries extra per-feature keys (e.g. "fps")
    # that the exploration data does not; the training code reads only these three.
    features = {k: {kk: v[kk] for kk in ("dtype", "shape", "names") if kk in v}
                for k, v in info["features"].items() if k not in DEFAULT_KEYS}
    data_keys = list(features)
    columns = ["episode_index", "frame_index", "task_index", *data_keys]

    sel = json.loads(args.selection.read_text())
    expected = {int(e): t["language"] for t in sel["tasks"].values() for e in t["episodes"]}
    demo_eps = sorted(expected)
    demo_tasks = read_tasks(args.demo_root)
    demo_rows = read_episodes(args.demo_root, demo_eps, columns)
    for e in demo_eps:
        got = {demo_tasks[int(i)] for i in demo_rows[e]["task_index"]}
        if {norm(t) for t in got} != {norm(expected[e])}:
            raise SystemExit(f"demo episode {e}: instruction {got} in the data, expected '{expected[e]}'")

    explore_eps, explore_rows, explore_tasks, log = [], {}, {}, None
    if args.explore_root is not None:
        log = json.loads((args.explore_root / "exploration_log.json").read_text())
        explore_eps = [e["episode"] for e in log["episodes_log"] if e["env_steps_after"] <= args.budget]
        if not explore_eps:
            raise SystemExit(f"no exploration episodes fit in a budget of {args.budget} steps")
        einfo = json.loads((args.explore_root / "meta" / "info.json").read_text())
        for k, v in features.items():
            ev = einfo["features"].get(k)
            if ev is None or ev["dtype"] != v["dtype"] or list(ev["shape"]) != list(v["shape"]):
                raise SystemExit(f"exploration feature {k} = {ev} does not match the demos' {v}")
        explore_tasks = read_tasks(args.explore_root)
        explore_rows = read_episodes(args.explore_root, explore_eps, columns)

    D = sum(len(demo_rows[e]["frame_index"]) for e in demo_eps)
    E = sum(len(explore_rows[e]["frame_index"]) for e in explore_eps)
    k = max(1, round(E / D)) if E else 1

    if args.out.exists():
        shutil.rmtree(args.out)
    out = LeRobotDataset.create(f"local/{args.out.name}", fps=info["fps"], features=features, root=args.out,
                                robot_type=info.get("robot_type"), use_videos=False,
                                image_writer_threads=args.writer_threads)
    sources = []

    def write(rows: dict, tasks: dict, src_root: Path, src_ep: int, kind: str) -> None:
        n = len(rows["frame_index"])
        for i in range(n):
            frame = {key: to_array(rows[key][i], features[key]) for key in data_keys}
            frame["task"] = tasks[int(rows["task_index"][i])]
            out.add_frame(frame)
        out.save_episode()
        sources.append({"kind": kind, "root": str(src_root), "episode": int(src_ep), "length": n,
                        "task": tasks[int(rows["task_index"][0])]})

    for rep in range(k):
        for e in demo_eps:
            write(demo_rows[e], demo_tasks, args.demo_root, e, "demo")
        print(f"demos written {rep + 1}/{k}", file=sys.stderr, flush=True)
    for e in explore_eps:
        write(explore_rows[e], explore_tasks, args.explore_root, e, "explore")
    out.finalize()

    demo_stats = json.loads((args.demo_root / "meta" / "stats.json").read_text())
    stats_path = args.out / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for key in stats:
        if key in demo_stats:
            stats[key] = demo_stats[key]
    stats_path.write_text(json.dumps(stats, indent=2))
    (args.out / "sources.json").write_text(json.dumps(sources))

    info_out = {"demo_episodes": demo_eps, "demo_frames": D, "demo_repeats": k,
                "explore_method": log["method"] if log else None, "explore_budget_steps": args.budget if log else 0,
                "explore_episodes": len(explore_eps), "explore_frames": E,
                "demo_fraction_of_frames": round(k * D / (k * D + E), 3),
                "total_frames": k * D + E, "total_episodes": len(sources)}
    (args.out / "mix_info.json").write_text(json.dumps(info_out, indent=2))
    print(json.dumps(info_out))


if __name__ == "__main__":
    main()
