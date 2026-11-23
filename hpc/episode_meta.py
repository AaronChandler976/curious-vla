"""Check (and repair in memory) a LeRobot dataset's per-episode metadata against its data files.

LeRobot looks an episode up by *row position* in meta/episodes (meta.episodes[e]) and trusts the
row's data/chunk_index, data/file_index and dataset_from/to_index. If the rows are not in
episode_index order, or a row points at the wrong parquet file, delete_episodes finds no rows for
the episode (KeyError in _copy_and_reindex_episodes_metadata) and frame lookups go to the wrong
episode. repair() sorts the rows by episode_index and, if needed, rewrites the data pointers from
what the data files actually contain. Nothing on disk is changed.

  python hpc/episode_meta.py --root DATASET_ROOT      # report only
"""

import argparse
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

DATA_COLS = ["data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]


def scan_data_files(root: Path) -> dict[int, dict]:
    """episode_index -> where its rows really are, read from the data parquet files."""
    truth: dict[int, dict] = {}
    offset = 0  # frame positions count rows across the sorted data files, as LeRobot loads them
    for path in sorted((root / "data").glob("*/*.parquet")):
        chunk = int(re.search(r"(\d+)", path.parent.name).group(1))
        file = int(re.search(r"(\d+)", path.stem).group(1))
        t = pq.read_table(path, columns=["episode_index"]).to_pandas()
        t["pos"] = range(offset, offset + len(t))
        offset += len(t)
        for e, g in t.groupby("episode_index"):
            e = int(e)
            if e in truth:
                raise SystemExit(f"episode {e} has rows in more than one data file ({path})")
            if g["pos"].max() - g["pos"].min() + 1 != len(g):
                raise SystemExit(f"episode {e} rows are not contiguous in {path}")
            truth[e] = {"data/chunk_index": chunk, "data/file_index": file,
                        "dataset_from_index": int(g["pos"].min()), "dataset_to_index": int(g["pos"].max()) + 1}
    return truth


def repair(ds, log=print) -> list[str]:
    """Fix ds.meta.episodes in memory. Returns a list of what was wrong (empty if it was fine)."""
    from lerobot.datasets.io_utils import load_episodes

    meta = ds.meta
    if meta.episodes is None:
        meta.episodes = load_episodes(meta.root)
    ep = meta.episodes
    issues = []

    order = [int(i) for i in ep["episode_index"]]
    if order != list(range(len(order))):
        bad = sum(1 for r, e in enumerate(order) if r != e)
        issues.append(f"{bad} of {len(order)} meta/episodes rows are not at the position of their episode_index")
        if sorted(order) != list(range(len(order))):
            raise SystemExit("meta/episodes episode_index values are not 0..N-1; cannot repair")
        ep = ep.sort("episode_index")

    truth = scan_data_files(Path(meta.root))
    rows = ep.select_columns(["episode_index", *DATA_COLS]).to_pandas()
    wrong = []
    for row in rows.to_dict("records"):
        e = int(row["episode_index"])
        if e not in truth or any(int(row[c]) != truth[e][c] for c in DATA_COLS):
            wrong.append(e)
    if wrong:
        issues.append(f"{len(wrong)} episodes point at the wrong data file or frame range (first: {wrong[:5]})")
        missing = [e for e in wrong if e not in truth]
        if missing:
            raise SystemExit(f"episodes {missing[:5]} have no rows in any data file; cannot repair")
        fixed = {c: [truth[int(e)][c] for e in ep["episode_index"]] for c in DATA_COLS}
        ep = ep.remove_columns(DATA_COLS)
        for c in DATA_COLS:
            ep = ep.add_column(c, fixed[c])

    meta.episodes = ep
    for msg in issues:
        log(f"episode metadata repaired in memory: {msg}")
    return issues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HuggingFaceVLA/libero")
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    class _DS:  # repair() only needs .meta
        meta = LeRobotDatasetMetadata(args.repo, root=args.root)

    issues = repair(_DS)
    print("EPISODE METADATA OK" if not issues else "EPISODE METADATA NEEDED REPAIR")
    sys.exit(0)


if __name__ == "__main__":
    main()
