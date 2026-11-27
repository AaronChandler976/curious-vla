"""Aggregate every finished eval into per-condition statistics and vector (PDF) figures.

  python hpc/analyze_results.py runs  [--out runs/analysis]

A "cell" is one condition (demos only, or demos + random / ICM exploration at a budget) at one
demo budget. Its runs differ in the training seed (which also picks the demo subset). For each
cell this reports the mean success over seeds, the standard deviation across seeds, a 95%
t-interval over seeds, and a 95% hierarchical-bootstrap interval (resample seeds, then each
seed's eval episodes), so both seed-to-seed and evaluation noise are in the interval.
Each exploration cell is compared with demos only at the same demo budget: the difference of
means, its bootstrap 95% interval, and a Welch t-test over seeds.

The main grid uses the original exploration collection (exploration seed 0). Runs on
re-collected exploration data (run names ending in _e<seed>) are reported separately, as a check
of how much one draw of exploration data matters.

Writes OUT/summary.csv, OUT/comparisons.csv, OUT/summary.md and OUT/fig_*.pdf.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NAME = re.compile(r"^(?P<cond>demos|(?P<mode>mix|seq)-(?P<method>\w+?)-(?P<steps>\d+))_(?P<suite>\w+?)_tasks(?P<tasks>[\d-]+)"
                  r"_n(?P<n>\d+)_s(?P<seed>\d+)(?:_e(?P<eseed>\d+))?(?:_f(?P<pct>\d+))?(?:_t(?P<tsteps>\d+))?$")
# Categorical slots of the reference palette.
COLORS = {"demos only": "#2a78d6", "random": "#eb6834", "icm": "#1baf7a", "icmw": "#8e5cd9",
          "demos only 10k": "#7fb2ea", "seq-random": "#f2a27f", "seq-icm": "#7fd3b5", "seq-icmw": "#bb9ee8"}
LABELS = {"demos only": "Demos only", "random": "Demos + random", "icm": "Demos + ICM",
          "icmw": "Demos + ICM (warm-up)", "demos only 10k": "Demos only, 10k steps",
          "seq-random": "Random, then demos", "seq-icm": "ICM, then demos", "seq-icmw": "ICM (warm-up), then demos"}
MARKERS = {"demos only": "o", "random": "s", "icm": "^", "icmw": "D", "demos only 10k": "o",
           "seq-random": "s", "seq-icm": "^", "seq-icmw": "D"}
MAIN = ["demos only", "random", "icm", "icmw"]   # the co-training grid at 20k training steps


def condition_of(m) -> str:
    if m["cond"] == "demos":
        return "demos only" if int(m["tsteps"] or 20000) == 20000 else f"demos only {int(m['tsteps']) // 1000}k"
    return m["method"] if m["mode"] == "mix" else f"seq-{m['method']}"
N_BOOT = 10000


def load(runs: Path) -> tuple[pd.DataFrame, float | None]:
    rows = []
    for p in sorted(runs.glob("*/eval/eval_info.json")):
        m = NAME.match(p.parent.parent.name)
        if not m:
            continue
        d = json.loads(p.read_text())
        per_task = {t["task_id"]: (t["n_success"], t["n_episodes"]) for t in d.get("per_task", [])}
        o = d["overall"]
        rows.append({
            "run": p.parent.parent.name,
            "condition": condition_of(m),
            "mode": m["mode"] or "demos",
            "train_steps": int(m["tsteps"] or 20000),
            "explore_steps": int(m["steps"] or 0),
            "n_demos": int(m["n"]),
            "seed": int(m["seed"]),
            "explore_seed": int(m["eseed"] or 0),
            "explore_pct": int(m["pct"] or (50 if m["steps"] else 0)),
            "n_success": int(o["n_success"]),
            "n_episodes": int(o["n_episodes"]),
            "success": o["n_success"] / o["n_episodes"],
            **{f"task{t}_success": s / n for t, (s, n) in per_task.items()},
        })
    ref = None
    for p in sorted(runs.glob("eval_reference/*/eval_info.json")) + sorted(runs.glob("eval_reference/eval_info.json")):
        ref = json.loads(p.read_text())["overall"]["pc_success"] / 100
    return pd.DataFrame(rows), ref


def boot_means(g: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Hierarchical bootstrap of a cell's mean success: resample seeds, then episodes."""
    k, n = g["n_success"].to_numpy(), g["n_episodes"].to_numpy()
    idx = rng.integers(0, len(k), size=(N_BOOT, len(k)))
    p = k[idx] / n[idx]
    return (rng.binomial(n[idx], p) / n[idx]).mean(axis=1)


def summarize(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    out, boots = [], {}
    for key, g in df.groupby(["condition", "explore_steps", "n_demos"]):
        s = g["success"].to_numpy()
        b = boot_means(g, rng)
        boots[key] = b
        sd = s.std(ddof=1) if len(s) > 1 else np.nan
        half = stats.t.ppf(0.975, len(s) - 1) * sd / np.sqrt(len(s)) if len(s) > 1 else np.nan
        out.append({"condition": key[0], "explore_steps": key[1], "n_demos": key[2], "n_seeds": len(s),
                    "seeds": " ".join(str(x) for x in sorted(g["seed"])),
                    "mean": s.mean(), "std": sd, "t_ci_lo": s.mean() - half, "t_ci_hi": s.mean() + half,
                    "boot_ci_lo": np.percentile(b, 2.5), "boot_ci_hi": np.percentile(b, 97.5),
                    "per_seed": " ".join(f"{x * 100:.0f}" for x in s)})
    return pd.DataFrame(out), boots


def compare(df: pd.DataFrame, summ: pd.DataFrame, boots: dict) -> pd.DataFrame:
    rows = []
    for _, r in summ[summ.condition != "demos only"].iterrows():
        base = ("demos only", 0, r.n_demos)
        if base not in boots:
            continue
        a = df[(df.condition == r.condition) & (df.explore_steps == r.explore_steps) & (df.n_demos == r.n_demos)]["success"]
        b = df[(df.condition == "demos only") & (df.n_demos == r.n_demos)]["success"]
        diff = boots[(r.condition, r.explore_steps, r.n_demos)] - boots[base]
        p = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
        rows.append({"condition": r.condition, "explore_steps": r.explore_steps, "n_demos": r.n_demos,
                     "n_seeds": len(a), "n_seeds_demos": len(b), "diff_vs_demos": a.mean() - b.mean(),
                     "boot_ci_lo": np.percentile(diff, 2.5), "boot_ci_hi": np.percentile(diff, 97.5),
                     "p_boot_le0": (diff <= 0).mean(), "welch_p": p})
    # Pooled over demo budgets: the average of the per-budget differences (each budget weighted
    # equally), with a bootstrap interval from the independent per-cell bootstraps.
    for (c, b), g in summ[summ.condition != "demos only"].groupby(["condition", "explore_steps"]):
        ns = [n for n in g.n_demos if ("demos only", 0, n) in boots]
        if len(ns) < 2:
            continue
        diff = np.mean([boots[(c, b, n)] - boots[("demos only", 0, n)] for n in ns], axis=0)
        point = np.mean([df[(df.condition == c) & (df.explore_steps == b) & (df.n_demos == n)]["success"].mean()
                         - df[(df.condition == "demos only") & (df.n_demos == n)]["success"].mean() for n in ns])
        rows.append({"condition": c, "explore_steps": b, "n_demos": "all", "n_seeds": "", "n_seeds_demos": "",
                     "diff_vs_demos": point, "boot_ci_lo": np.percentile(diff, 2.5),
                     "boot_ci_hi": np.percentile(diff, 97.5), "p_boot_le0": (diff <= 0).mean(), "welch_p": np.nan})
    # The head-to-head question: ICM vs. random at the same exploration budget and demo budget.
    for b in sorted(summ[summ.condition == "icm"].explore_steps.unique()):
        ns = [n for n in summ.n_demos.unique() if ("icm", b, n) in boots and ("random", b, n) in boots]
        per = {}
        for n in sorted(ns):
            a = df[(df.condition == "icm") & (df.explore_steps == b) & (df.n_demos == n)]["success"]
            r = df[(df.condition == "random") & (df.explore_steps == b) & (df.n_demos == n)]["success"]
            d = boots[("icm", b, n)] - boots[("random", b, n)]
            per[n] = d
            rows.append({"condition": "icm vs random", "explore_steps": b, "n_demos": n, "n_seeds": len(a),
                         "n_seeds_demos": len(r), "diff_vs_demos": a.mean() - r.mean(),
                         "boot_ci_lo": np.percentile(d, 2.5), "boot_ci_hi": np.percentile(d, 97.5),
                         "p_boot_le0": (d <= 0).mean(), "welch_p": stats.ttest_ind(a, r, equal_var=False).pvalue})
        if len(per) > 1:
            d = np.mean(list(per.values()), axis=0)
            pt = np.mean([rw["diff_vs_demos"] for rw in rows if rw["condition"] == "icm vs random"
                          and rw["explore_steps"] == b and rw["n_demos"] != "all"])
            rows.append({"condition": "icm vs random", "explore_steps": b, "n_demos": "all", "n_seeds": "",
                         "n_seeds_demos": "", "diff_vs_demos": pt, "boot_ci_lo": np.percentile(d, 2.5),
                         "boot_ci_hi": np.percentile(d, 97.5), "p_boot_le0": (d <= 0).mean(), "welch_p": np.nan})
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8, "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2, "savefig.bbox": "tight",
    })


def draw_cell(ax, x, summ_rows, cond, raw, offset, dot_colors=None):
    if summ_rows.empty:
        return
    c = COLORS[cond]
    xs = np.array(x, float) * (1 + offset) if ax.get_xscale() == "log" else np.array(x, float) + offset
    m = summ_rows["mean"].to_numpy() * 100
    lo, hi = summ_rows["boot_ci_lo"].to_numpy() * 100, summ_rows["boot_ci_hi"].to_numpy() * 100
    ax.fill_between(xs, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.plot(xs, m, color=c, marker=MARKERS[cond], markersize=6, markeredgecolor="white", markeredgewidth=1,
            label=LABELS[cond], zorder=3)
    for xi, (_, g) in zip(xs, raw):
        dc = COLORS[g["condition"].iloc[0]] if "condition" in g else c
        ax.scatter(np.full(len(g), xi), g["success"] * 100, s=9, color=dc, alpha=0.45, linewidths=0, zorder=2)


def fig_vs_demos(summ, df, ref, out: Path):
    budgets = sorted(b for b in summ.explore_steps.unique() if b > 0)
    if not budgets:
        return
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.3 * len(budgets), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, budgets):
        conds = [c for c in MAIN if c in set(summ.condition)]
        for i, cond in enumerate(conds):
            steps = 0 if cond == "demos only" else b
            s = summ[(summ.condition == cond) & (summ.explore_steps == steps)].sort_values("n_demos")
            raw = df[(df.condition == cond) & (df.explore_steps == steps)].groupby("n_demos")
            draw_cell(ax, s.n_demos, s, cond, raw, (i - (len(conds) - 1) / 2) * 0.9)
        if ref is not None:
            ax.axhline(ref * 100, color="#52514e", linestyle=(0, (4, 3)), linewidth=1)
            ax.text(10, ref * 100 + 1.5, f"official smolvla_libero ({ref * 100:.0f}%)",
                    fontsize=7, color="#52514e")
        ax.set_title(f"{b // 1000}k exploration steps mixed in")
        ax.set_xlabel("Demos per task")
        ax.set_xticks(sorted(summ.n_demos.unique()))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.savefig(out / "fig_success_vs_demos.pdf")
    plt.close(fig)


def fig_vs_exploration(summ, df, out: Path):
    ns = sorted(summ.n_demos.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.6 * len(ns), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    base_all = summ[summ.condition == "demos only"]
    for ax, n in zip(axes, ns):
        base = base_all[base_all.n_demos == n]
        econds = [c for c in MAIN[1:] if c in set(summ.condition)]
        for i, cond in enumerate(econds):
            s = summ[(summ.condition == cond) & (summ.n_demos == n)]
            s = pd.concat([base, s]).sort_values("explore_steps")
            raw = pd.concat([df[(df.condition == "demos only") & (df.n_demos == n)],
                             df[(df.condition == cond) & (df.n_demos == n)]]).groupby("explore_steps")
            xs = s.explore_steps.to_numpy() / 1000
            draw_cell(ax, xs, s, cond, [(k / 1000, g) for k, g in raw], (i - (len(econds) - 1) / 2) * 0.5)
        if not base.empty:
            ax.errorbar([0], base["mean"] * 100, fmt="o", color=COLORS["demos only"], markersize=6,
                        markeredgecolor="white", label=LABELS["demos only"], zorder=4)
        ax.set_title(f"{n} demos per task")
        ax.set_xticks([x / 1000 for x in sorted(set(summ.explore_steps) | {0})])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.supxlabel("Exploration env steps mixed in (thousands)", fontsize=9, y=-0.04)
    fig.savefig(out / "fig_success_vs_exploration.pdf")
    plt.close(fig)


def fig_replicates(rep: pd.DataFrame, out: Path):
    """Per exploration collection: does one draw of exploration data drive the result?"""
    if rep.empty or rep.explore_seed.nunique() < 2:
        return
    cells = sorted({(c, b, n) for c, b, n in zip(rep.condition, rep.explore_steps, rep.n_demos)})
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(cells), 2.8))
    for i, (c, b, n) in enumerate(cells):
        g = rep[(rep.condition == c) & (rep.explore_steps == b) & (rep.n_demos == n)]
        seeds = sorted(g.explore_seed.unique())
        for j, es in enumerate(seeds):
            h = g[g.explore_seed == es]["success"] * 100
            x = i + (j - (len(seeds) - 1) / 2) * 0.22
            ax.scatter(np.full(len(h), x), h, s=10, color=COLORS[c], alpha=0.45, linewidths=0)
            ax.plot([x - 0.08, x + 0.08], [h.mean()] * 2, color=COLORS[c], linewidth=2)
            ax.text(x, -7, f"e{es}", ha="center", fontsize=6.5, color="#52514e")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{LABELS[c].replace('Demos + ', '')}\n{b // 1000}k, n{n}" for c, b, n in cells])
    ax.set_ylim(-10, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Exploration-data replicates (e = collection seed; bar = mean over training seeds)")
    fig.savefig(out / "fig_exploration_replicates.pdf")
    plt.close(fig)


# Paired comparisons. Runs with the same demo budget and training seed share the demo subset (the
# seed picks it), the training seed and the eval seed (so the same initial states), so their
# difference removes the demo-subset variance that dominates the unpaired comparison.
PAIRS = [  # (A, A's exploration steps, B, B's exploration steps, question)
    ("random", 5000, "demos only", 0, "co-training with random data vs demos only"),
    ("random", 20000, "demos only", 0, "co-training with random data vs demos only"),
    ("icm", 5000, "demos only", 0, "co-training with ICM data vs demos only"),
    ("icm", 20000, "demos only", 0, "co-training with ICM data vs demos only"),
    ("icmw", 5000, "demos only", 0, "co-training with warmed-up ICM data vs demos only"),
    ("icmw", 20000, "demos only", 0, "co-training with warmed-up ICM data vs demos only"),
    ("icm", 5000, "random", 5000, "ICM vs random data, same budget"),
    ("icm", 20000, "random", 20000, "ICM vs random data, same budget"),
    ("icmw", 5000, "random", 5000, "warmed-up ICM vs random data, same budget"),
    ("icmw", 20000, "random", 20000, "warmed-up ICM vs random data, same budget"),
    ("icmw", 20000, "icm", 20000, "warmed-up ICM vs barely trained ICM"),
    ("demos only 10k", 0, "demos only", 0, "dilution: half the demo gradient steps"),
    ("random", 20000, "demos only 10k", 0, "50% mix vs demos at matched demo exposure"),
    ("icm", 20000, "demos only 10k", 0, "50% mix vs demos at matched demo exposure"),
    ("random", 5000, "demos only 10k", 0, "50% mix vs demos at matched demo exposure"),
    ("icm", 5000, "demos only 10k", 0, "50% mix vs demos at matched demo exposure"),
    ("seq-random", 20000, "demos only 10k", 0, "exploration pretraining vs none, matched demo steps"),
    ("seq-icm", 20000, "demos only 10k", 0, "exploration pretraining vs none, matched demo steps"),
    ("seq-icmw", 20000, "demos only 10k", 0, "exploration pretraining vs none, matched demo steps"),
    ("seq-random", 20000, "demos only", 0, "two-stage vs demos only, matched total steps"),
    ("seq-icm", 20000, "demos only", 0, "two-stage vs demos only, matched total steps"),
    ("seq-icmw", 20000, "demos only", 0, "two-stage vs demos only, matched total steps"),
    ("seq-random", 20000, "random", 20000, "two-stage vs co-training, same data"),
    ("seq-icm", 20000, "icm", 20000, "two-stage vs co-training, same data"),
    ("seq-icm", 20000, "seq-random", 20000, "ICM vs random data, two-stage"),
    ("seq-icmw", 20000, "seq-random", 20000, "warmed-up ICM vs random data, two-stage"),
]


def paired_comparisons(df: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for a, ab, b, bb, q in PAIRS:
        A = df[(df.condition == a) & (df.explore_steps == ab)].set_index(["n_demos", "seed"])
        B = df[(df.condition == b) & (df.explore_steps == bb)].set_index(["n_demos", "seed"])
        both = A.index.intersection(B.index)
        if len(both) < 3:
            continue
        d_all = (A.loc[both, "success"] - B.loc[both, "success"])
        for n in sorted(set(both.get_level_values(0))) + ["all"]:
            d = d_all if n == "all" else d_all[d_all.index.get_level_values(0) == n]
            if len(d) < 3:
                continue
            v = d.to_numpy()
            # Bootstrap over pairs (stratified by demo budget for the pooled row).
            if n == "all":
                groups = [d_all[d_all.index.get_level_values(0) == k].to_numpy() for k in set(both.get_level_values(0))]
                bs = np.mean([g[rng.integers(0, len(g), size=(N_BOOT, len(g)))].mean(1) for g in groups], axis=0)
                point = np.mean([g.mean() for g in groups])
            else:
                bs = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
                point = v.mean()
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            try:
                w = stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0
            except ValueError:
                w = np.nan
            rows.append({"a": a, "a_steps": ab, "b": b, "b_steps": bb, "question": q, "n_demos": n, "pairs": len(v),
                         "mean_diff": point, "t_ci_lo": v.mean() - half, "t_ci_hi": v.mean() + half,
                         "boot_ci_lo": np.percentile(bs, 2.5), "boot_ci_hi": np.percentile(bs, 97.5),
                         "p_paired_t": stats.ttest_1samp(v, 0).pvalue, "p_wilcoxon": w,
                         "a_wins": int((v > 0).sum()), "ties": int((v == 0).sum())})
    return pd.DataFrame(rows)


def fig_falsification(df: pd.DataFrame, out: Path):
    """Every condition at 20k total training steps (plus demos only at 10k) side by side, per demo budget."""
    order = [("demos only", 0), ("demos only 10k", 0), ("random", 20000), ("icm", 20000), ("icmw", 20000),
             ("seq-random", 20000), ("seq-icm", 20000), ("seq-icmw", 20000)]
    order = [(c, b) for c, b in order if not df[(df.condition == c) & (df.explore_steps == b)].empty]
    if len(order) < 3:
        return
    ns = sorted(df.n_demos.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.9 * len(ns), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, ns):
        for i, (c, b) in enumerate(order):
            g = df[(df.condition == c) & (df.explore_steps == b) & (df.n_demos == n)]["success"] * 100
            if g.empty:
                continue
            ax.scatter(np.full(len(g), i) + np.linspace(-0.15, 0.15, len(g)), g, s=8, color=COLORS[c], alpha=0.5, linewidths=0)
            m = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0
            ax.errorbar([i], [m], yerr=[1.96 * se], fmt=MARKERS[c], color=COLORS[c], markersize=6,
                        markeredgecolor="white", capsize=2, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LABELS[c] + ("" if b == 0 or c.startswith("seq") else f" {b // 1000}k") for c, b in order],
                           rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{n} demos per task")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)")
    fig.savefig(out / "fig_falsification.pdf")
    plt.close(fig)


def pct(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = Path(a.runs)
    out = Path(a.out) if a.out else runs / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df, ref = load(runs)
    if df.empty:
        raise SystemExit("no finished evals")
    df.to_csv(out / "runs.csv", index=False)
    main_df = df[(df.explore_seed == 0) & df.explore_pct.isin([0, 50]) & df.condition.isin(MAIN)]
    paired_df = df[(df.explore_seed == 0) & df.explore_pct.isin([0, 50])]
    pair = paired_comparisons(paired_df, rng)
    pair.to_csv(out / "paired.csv", index=False)
    summ, boots = summarize(main_df, rng)
    comp = compare(main_df, summ, boots)
    summ.to_csv(out / "summary.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)  # n_demos "all" rows: pooled over budgets
    # Replicates: every exploration cell that has runs on more than one exploration collection.
    df50 = df[df.explore_pct.isin([0, 50])]
    rep_keys = df50[df50.explore_seed > 0][["condition", "explore_steps", "n_demos"]].drop_duplicates()
    rep = df50.merge(rep_keys, on=["condition", "explore_steps", "n_demos"])
    # Mixing-ratio ablation: cells run at other exploration percentages, next to the same cell at 50%.
    abl_keys = df[~df.explore_pct.isin([0, 50])][["condition", "explore_steps", "n_demos"]].drop_duplicates()
    abl = df[df.explore_seed == 0].merge(abl_keys, on=["condition", "explore_steps", "n_demos"])
    abl_summ = (abl.groupby(["condition", "explore_steps", "n_demos", "explore_pct"])["success"]
                .agg(["count", "mean", "std"]).reset_index()) if not abl.empty else pd.DataFrame()
    rep_summ = (rep.groupby(["condition", "explore_steps", "n_demos", "explore_seed"])["success"]
                .agg(["count", "mean", "std"]).reset_index()) if not rep.empty else pd.DataFrame()

    style()
    fig_vs_demos(summ, main_df, ref, out)
    fig_vs_exploration(summ, main_df, out)
    fig_replicates(rep, out)
    fig_falsification(paired_df, out)

    lines = ["# Results summary", "", f"{len(df)} finished runs. Success over 100 eval episodes per run "
             "(50 per task). Mean and std are over training seeds; CIs are 95%.", ""]
    if ref is not None:
        lines += [f"Official `smolvla_libero` checkpoint on the same tasks: {ref * 100:.0f}%.", ""]
    lines += ["| condition | expl. steps | demos/task | seeds | mean % | std | 95% CI (t, seeds) | 95% CI (hier. bootstrap) | per seed |",
              "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for _, r in summ.sort_values(["condition", "explore_steps", "n_demos"]).iterrows():
        lines.append(f"| {r.condition} | {r.explore_steps} | {r.n_demos} | {r.n_seeds} | {pct(r['mean'])} | {pct(r['std'])} "
                     f"| {pct(r.t_ci_lo)} to {pct(r.t_ci_hi)} | {pct(r.boot_ci_lo)} to {pct(r.boot_ci_hi)} | {r.per_seed} |")
    lines += ["", "## Each exploration condition vs. demos only at the same demo budget",
              "", "Rows \"icm vs random\" compare ICM with random instead (positive: ICM better). Rows with "
              "demos/task \"all\" average the per-budget differences.", "",
              "| condition | expl. steps | demos/task | seeds | diff (pts) | 95% CI (bootstrap) | P(diff <= 0) | Welch p |",
              "|---|---:|---:|---|---:|---|---:|---:|"]
    comp["_n"] = comp.n_demos.map(lambda v: 10**6 if v == "all" else v)
    for _, r in comp.sort_values(["condition", "explore_steps", "_n"]).iterrows():
        seeds = "pooled" if r.n_demos == "all" else f"{r.n_seeds} vs {r.n_seeds_demos}"
        lines.append(f"| {r.condition} | {r.explore_steps} | {r.n_demos} | {seeds} | "
                     f"{r.diff_vs_demos * 100:+.1f} | {r.boot_ci_lo * 100:+.1f} to {r.boot_ci_hi * 100:+.1f} | "
                     f"{r.p_boot_le0:.3f} | {'' if pd.isna(r.welch_p) else f'{r.welch_p:.3f}'} |")
    if not rep_summ.empty:
        lines += ["", "## Exploration-data replicates", "",
                  "| condition | expl. steps | demos/task | collection seed | runs | mean % | std |", "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in rep_summ.iterrows():
            lines.append(f"| {r.condition} | {r.explore_steps} | {r.n_demos} | {r.explore_seed} | {r['count']} | "
                         f"{pct(r['mean'])} | {pct(r['std'])} |")
    if not abl_summ.empty:
        lines += ["", "## Mixing-ratio ablation (exploration % of training frames)", "",
                  "| condition | expl. steps | demos/task | expl. % | runs | mean % | std | demos only at this budget |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in abl_summ.iterrows():
            b = summ[(summ.condition == "demos only") & (summ.n_demos == r.n_demos)]["mean"]
            lines.append(f"| {r.condition} | {r.explore_steps} | {r.n_demos} | {r.explore_pct} | {r['count']} | "
                         f"{pct(r['mean'])} | {pct(r['std'])} | {pct(b.iloc[0]) if len(b) else ''} |")
    if not pair.empty:
        lines += ["", "## Paired comparisons (same demo subset, training seed and eval seed)", "",
                  "Each pair is one training seed at one demo budget; diff = A minus B in points. "
                  "Rows with demos/task \"all\" average the per-budget mean differences.", "",
                  "| A | B | question | demos/task | pairs | diff (pts) | 95% CI (t) | 95% CI (bootstrap) | paired t p | Wilcoxon p | A wins |",
                  "|---|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
        lab = lambda c, b: c + (f" {b // 1000}k" if b else "")
        for _, r in pair.iterrows():
            lines.append(f"| {lab(r.a, r.a_steps)} | {lab(r.b, r.b_steps)} | {r.question} | {r.n_demos} | {r.pairs} | "
                         f"{r.mean_diff * 100:+.1f} | {r.t_ci_lo * 100:+.1f} to {r.t_ci_hi * 100:+.1f} | "
                         f"{r.boot_ci_lo * 100:+.1f} to {r.boot_ci_hi * 100:+.1f} | {r.p_paired_t:.3f} | "
                         f"{'' if pd.isna(r.p_wilcoxon) else f'{r.p_wilcoxon:.3f}'} | {r.a_wins}/{r.pairs} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
