"""Describe each exploration dataset without training anything: how much of the workspace it covers,
how often it moves objects, how close it gets to where the demos go, and whether its actions can be
predicted from its observations (the only signal a behavior-cloning loss can extract from it).

  python hpc/exploration_stats.py runs [--demo-root DATA_ROOT] [--out runs/analysis/exploration_stats.csv]

Reads only the state and action columns (no images), so it runs on the login node in a minute.
Per dataset (and for the demos of the two tasks, as a reference), at the 5k and 20k budgets:
  eef_voxels_2cm   distinct 2 cm voxels visited by the end effector (observation.state[:3])
  eef_spread_cm    mean per-axis standard deviation of the end-effector position
  near_demo_pct    % of frames whose end effector is within 3 cm of some demo frame of the same tasks
  obj_moved_pct    % of episodes that moved an object by more than 1 cm (exploration_log.json)
  grip_switch_pct  % of steps where the gripper command flips sign
  act_abs_mean     mean |action| over the 6 arm dimensions
  r2_state         held-out (by episode) R^2 of predicting the action from the 8-D state (ridge on random
                   Fourier features); near 0 means the actions carry no observation-dependent signal
  policy_std_end   ICM explorer: the policy's action std at the last PPO update (1.0 = untrained)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


def load(root: Path, episodes=None) -> pd.DataFrame:
    files = sorted(str(p) for p in (root / "data").glob("*/*.parquet"))
    ds = pads.dataset(files, format="parquet")
    flt = pads.field("episode_index").isin(episodes) if episodes is not None else None
    return ds.to_table(columns=["episode_index", "frame_index", "observation.state", "action"], filter=flt).to_pandas()


def voxels(x: np.ndarray, size=0.02) -> int:
    return len({tuple(v) for v in np.floor(x / size).astype(int)})


def r2_state(S: np.ndarray, A: np.ndarray, ep: np.ndarray, seed=0, feats=512, lam=1.0) -> float:
    rng = np.random.default_rng(seed)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Z = (S - mu) / sd
    W = rng.normal(size=(Z.shape[1], feats)); b = rng.uniform(0, 2 * np.pi, feats)
    F = np.concatenate([np.cos(Z @ W + b), Z, np.ones((len(Z), 1))], 1)
    # Hold out whole episodes: neighboring frames share a held action, so a frame split would leak.
    eps = rng.permutation(np.unique(ep)); test_eps = set(eps[: max(1, len(eps) // 5)].tolist())
    te = np.flatnonzero(np.isin(ep, list(test_eps))); tr = np.flatnonzero(~np.isin(ep, list(test_eps)))
    coef = np.linalg.solve(F[tr].T @ F[tr] + lam * len(tr) * np.eye(F.shape[1]), F[tr].T @ A[tr])
    pred = F[te] @ coef
    ss_res = ((A[te] - pred) ** 2).sum(0); ss_tot = ((A[te] - A[te].mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - ss_res / ss_tot))


def describe(df: pd.DataFrame, demo_eef: np.ndarray | None) -> dict:
    S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    A = np.stack(df["action"].to_numpy()).astype(np.float64)
    eef = S[:, :3]
    out = {"frames": len(df), "eef_voxels_2cm": voxels(eef), "eef_spread_cm": float(eef.std(0).mean() * 100),
           "act_abs_mean": float(np.abs(A[:, :6]).mean())}
    g = np.sign(A[:, -1]); same_ep = df["episode_index"].to_numpy()[1:] == df["episode_index"].to_numpy()[:-1]
    out["grip_switch_pct"] = float(100 * ((g[1:] != g[:-1]) & same_ep).mean())
    if demo_eef is not None:
        sub = eef[:: max(1, len(eef) // 4000)]
        d = np.sqrt(((sub[:, None, :] - demo_eef[None, ::4, :]) ** 2).sum(-1)).min(1)
        out["near_demo_pct"] = float(100 * (d < 0.03).mean())
    out["r2_state"] = r2_state(S, A, df["episode_index"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--demo-root", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=None, help="a runs/selections/*.json with the demo episodes to use")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = Path(a.runs)
    rows, demo_eef = [], None
    if a.demo_root is not None:
        sel = a.selection or sorted((runs / "selections").glob("demos_*_n40_s0.json"))[0]
        eps = sorted(int(e) for t in json.loads(Path(sel).read_text())["tasks"].values() for e in t["episodes"])
        dd = load(a.demo_root, eps)
        demo_eef = np.stack(dd["observation.state"].to_numpy())[:, :3].astype(np.float64)
        rows.append({"dataset": f"demos ({len(eps)} episodes)", "method": "demos", "collection_seed": "", "budget": len(dd),
                     **describe(dd, demo_eef), "obj_moved_pct": 100.0, "policy_std_end": None})
    for root in sorted((runs / "exploration").glob("*_tasks*")):
        log_path = root / "exploration_log.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        df = load(root)
        es = root.name.rsplit("_e", 1)[1] if "_e" in root.name.split("tasks")[-1] else "0"
        for budget in (5000, 20000):
            eps = [e["episode"] for e in log["episodes_log"] if e["env_steps_after"] <= budget]
            sub = df[df.episode_index.isin(eps)]
            moved = [e["max_object_qpos_change"] > 0.01 for e in log["episodes_log"] if e["episode"] in set(eps)]
            ups = log.get("icm_updates") or []
            rows.append({"dataset": root.name, "method": log["method"], "collection_seed": es, "budget": budget,
                         **describe(sub, demo_eef), "obj_moved_pct": float(100 * np.mean(moved)) if moved else None,
                         "policy_std_end": ups[-1]["policy_std"] if ups else None,
                         "ppo_updates": len(ups), "warmup_steps": log.get("warmup_steps", 0)})
    t = pd.DataFrame(rows)
    out = a.out or runs / "analysis" / "exploration_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
