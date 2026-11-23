"""Check that the frames the training dataloader returns are the right raw frames.

Loads the dataset exactly as lerobot-train does (LeRobotDataset with the policy's delta
timestamps, i.e. a 50-step action chunk for SmolVLA, return_uint8=True) and compares sampled
items with the raw parquet rows of their source episode, read by episode_index/frame_index only
(never through meta/episodes):
  - every episode has the same number of frames as its source;
  - observation.state, both camera images and the whole action chunk (plus its padding mask)
    equal the source rows;
  - the instruction equals the source's, and for demos the LIBERO language of the task the
    episode was selected for.

Two modes:
  # a built dataset (build_dataset.py); sources.json says where each episode came from
  python hpc/verify_training_data.py --root BUILT --selection SEL.json [--policy-dir P]
  # the demo dataset itself, with --dataset.episodes as the old demos-only jobs trained on it
  python hpc/verify_training_data.py --root DEMO --repo HuggingFaceVLA/libero --selection SEL.json --direct
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DEFAULT_KEYS, norm, read_episodes, read_tasks, to_array  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", default=None, help="repo id (default local/<root name>)")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--direct", action="store_true", help="root is the demo dataset; train on --selection's episodes")
    p.add_argument("--policy-dir", type=Path, default=None, help="policy config for delta timestamps (default SmolVLA)")
    p.add_argument("--random-items", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if args.policy_dir is not None:
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (registers the smolvla type)
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(args.policy_dir)
    else:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg = SmolVLAConfig()

    sel = json.loads(args.selection.read_text())
    language = {int(e): t["language"] for t in sel["tasks"].values() for e in t["episodes"]}
    repo = args.repo or f"local/{args.root.name}"
    meta = LeRobotDatasetMetadata(repo, root=args.root)
    delta = resolve_delta_timestamps(cfg, meta)
    episodes = sorted(language) if args.direct else None
    ds = LeRobotDataset(repo, root=args.root, episodes=episodes, delta_timestamps=delta,
                        return_uint8=True, tolerance_s=1e-4)

    # Where each loaded episode really comes from.
    if args.direct:
        sources = {e: {"kind": "demo", "root": str(args.root), "episode": e} for e in episodes}
    else:
        sources = {i: s for i, s in enumerate(json.loads((args.root / "sources.json").read_text()))}

    keys = [k for k in meta.features if k not in DEFAULT_KEYS]
    img_keys = [k for k in keys if meta.features[k]["dtype"] in ("image", "video")]
    columns = ["episode_index", "frame_index", "task_index", *keys]
    raw, raw_tasks = {}, {}
    by_root: dict[str, set] = {}
    for s in sources.values():
        by_root.setdefault(s["root"], set()).add(s["episode"])
    for r, eps in by_root.items():
        rows = read_episodes(Path(r), sorted(eps), columns)
        raw_tasks[r] = read_tasks(Path(r))
        for e, v in rows.items():
            raw[(r, e)] = v

    problems: list[str] = []
    cols = ds.hf_dataset.with_format(None).select_columns(["episode_index", "frame_index"])
    ep_col = np.asarray(cols["episode_index"])
    positions: dict[int, list[int]] = {}
    for i, e in enumerate(ep_col.tolist()):
        positions.setdefault(int(e), []).append(i)
    if set(positions) != set(sources):
        problems.append(f"loaded episodes {sorted(positions)[:10]} != expected {sorted(sources)[:10]}")
    for e, pos in positions.items():
        s = sources.get(e)
        if s and len(pos) != len(raw[(s["root"], s["episode"])]["frame_index"]):
            problems.append(f"episode {e}: {len(pos)} frames loaded, source has {len(raw[(s['root'], s['episode'])]['frame_index'])}")

    picks = set()
    for pos in positions.values():
        picks.update([pos[0], pos[len(pos) // 2], pos[-1]])
    rng = random.Random(args.seed)
    picks.update(rng.sample(range(len(ds)), min(args.random_items, len(ds))))

    chunk = None
    for i in sorted(picks):
        try:
            item = ds[i]
        except Exception as exc:  # e.g. a KeyError from frame ranges that point outside the episode
            problems.append(f"item {i}: the dataloader raised {type(exc).__name__}: {exc}")
            continue
        e, fi = int(item["episode_index"]), int(item["frame_index"])
        s = sources[e]
        src = raw[(s["root"], s["episode"])]
        n = len(src["frame_index"])
        where = f"item {i} (episode {e} frame {fi}, source {s['kind']} {s['episode']})"
        state = to_array(src["observation.state"][fi], meta.features["observation.state"])
        got_state = item["observation.state"].numpy()
        if got_state.ndim == 2:
            got_state = got_state[-1]
        if got_state.shape != state.shape or not np.allclose(got_state, state, atol=1e-6):
            problems.append(f"{where}: state differs from the source row")
        act = item["action"].numpy()
        if act.ndim == 1:
            act = act[None]
        chunk = len(act)
        pad = item.get("action_is_pad")
        pad = pad.numpy() if pad is not None else np.zeros(chunk, bool)
        for j in range(chunk):
            if fi + j < n:
                want = to_array(src["action"][fi + j], meta.features["action"])
                if pad[j] or not np.allclose(act[j], want, atol=1e-6):
                    problems.append(f"{where}: action chunk step {j} differs from source frame {fi + j}")
                    break
            elif not pad[j]:
                problems.append(f"{where}: action chunk step {j} is past the episode end but not padded")
                break
        for k in img_keys:
            got = item[k].numpy()
            if got.ndim == 4:  # (1, C, H, W) when the policy asks for observation delta [0]
                got = got[-1]
            if got.dtype != np.uint8:  # image-dtype features come back as float in [0, 1]
                got = np.round(got * 255).astype(np.uint8)
            want = to_array(src[k][fi], meta.features[k]).transpose(2, 0, 1)
            if got.shape != want.shape or not np.array_equal(got, want):
                problems.append(f"{where}: {k} differs from the source image")
        task = item["task"]
        src_task = raw_tasks[s["root"]][int(src["task_index"][fi])]
        if norm(task) != norm(src_task):
            problems.append(f"{where}: instruction '{task}' but source has '{src_task}'")
        if s["kind"] == "demo" and norm(task) != norm(language.get(s["episode"], "")):
            problems.append(f"{where}: instruction '{task}', selected for '{language.get(s['episode'])}'")
        if s["kind"] == "explore" and norm(task) != norm("explore the scene"):
            problems.append(f"{where}: exploration frame has instruction '{task}'")

    kinds = {}
    for s in sources.values():
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    print(f"checked {len(picks)} items over {len(positions)} episodes {kinds}, action chunk {chunk}")
    if problems:
        print("TRAINING DATA MISMATCH:\n  " + "\n  ".join(problems[:15]))
        if len(problems) > 15:
            print(f"  ... and {len(problems) - 15} more")
        sys.exit(1)
    print("TRAINING DATA OK")


if __name__ == "__main__":
    main()
