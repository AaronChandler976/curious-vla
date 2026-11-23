"""Collect every finished eval under runs/ into one table (and results.csv).

  python hpc/summarize_results.py runs

Run names encode the condition: demos_<suite>_tasks<ids>_n<demos>_s<seed> for demos only, and
mix-<method>-<steps>_<suite>_tasks<ids>_n<demos>_s<seed> for demos + exploration.
"""

import csv
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^(?P<cond>demos|mix-(?P<method>\w+)-(?P<steps>\d+))_(?P<suite>\w+?)_tasks(?P<tasks>[\d-]+)_n(?P<n>\d+)_s(?P<seed>\d+)$")


def main() -> None:
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    rows = []
    for info_path in sorted(runs.glob("*/eval/eval_info.json")):
        m = NAME.match(info_path.parent.parent.name)
        if not m:
            continue
        overall = json.loads(info_path.read_text())["overall"]
        rows.append({
            "condition": "demos only" if m["cond"] == "demos" else m["method"],
            "explore_steps": int(m["steps"] or 0),
            "n_demos": int(m["n"]),
            "seed": int(m["seed"]),
            "success_pct": round(overall["pc_success"], 1),
            "n_episodes": overall.get("n_episodes"),
            "ci95": overall.get("pc_success_ci95"),
            "run": info_path.parent.parent.name,
        })
    for info_path in sorted(runs.glob("eval_reference/*/eval_info.json")):
        overall = json.loads(info_path.read_text())["overall"]
        rows.append({"condition": "official smolvla_libero", "explore_steps": 0, "n_demos": -1, "seed": 0,
                     "success_pct": round(overall["pc_success"], 1), "n_episodes": overall.get("n_episodes"),
                     "ci95": overall.get("pc_success_ci95"), "run": info_path.parent.name})
    if not rows:
        sys.exit("no finished evals yet")
    rows.sort(key=lambda r: (r["condition"], r["explore_steps"], r["n_demos"], r["seed"]))
    print(f"{'condition':26} {'expl.steps':>10} {'demos/task':>10} {'seed':>4} {'success %':>9} {'episodes':>8}")
    for r in rows:
        print(f"{r['condition']:26} {r['explore_steps']:>10} {r['n_demos']:>10} {r['seed']:>4} "
              f"{r['success_pct']:>9} {str(r['n_episodes']):>8}")
    with open(runs / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {runs / 'results.csv'}")


if __name__ == "__main__":
    main()
