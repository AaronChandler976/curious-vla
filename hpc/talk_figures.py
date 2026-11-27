"""Figures for the talk and report (vector PDF), from the analysis outputs.

  python hpc/talk_figures.py --an runs/analysis_nas10 --an-open runs/analysis --out slides/figures

--an is the primary analysis (closed-loop eval, n_action_steps 10); --an-open the original
open-loop eval (n_action_steps 50), used only for the evaluation-protocol figure.
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"demos only": "#2a78d6", "random": "#eb6834", "icm": "#1baf7a", "icmw": "#8e5cd9"}
L = {"demos only": "Demos only", "random": "+ random", "icm": "+ ICM", "icmw": "+ ICM (trained)"}
INK, MUTED, GRID, BG = "#1d232b", "#5a6270", "#e3e1da", "#fbfaf7"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.grid.axis": "y", "grid.color": GRID, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "lines.linewidth": 1.6, "pdf.fonttype": 42})


def ref_of(an):
    try:
        for line in open(f"{an}/summary.md"):
            if line.startswith("Official"):
                return float(line.split(":")[1].strip().rstrip(".%"))
    except OSError:
        pass
    return None


def main_fig(df, ref, out, conds=("demos only", "random", "icm")):
    main = df[(df.explore_seed == 0) & df.explore_pct.isin([0, 50]) & (df.train_steps == 20000) & (df["mode"] != "seq")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
    for ax, b in zip(axes, [5000, 20000]):
        cs = [c for c in conds if c == "demos only" or not main[(main.condition == c) & (main.explore_steps == b)].empty]
        for i, c in enumerate(cs):
            st = 0 if c == "demos only" else b
            ns, m, e = [], [], []
            for n in (10, 25, 40):
                g = main[(main.condition == c) & (main.explore_steps == st) & (main.n_demos == n)].success * 100
                if len(g) > 1:
                    ns.append(n); m.append(g.mean()); e.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
            if not ns:
                continue
            x = np.array(ns) + (i - (len(cs) - 1) / 2) * 0.8
            ax.fill_between(x, np.array(m) - e, np.array(m) + e, color=C[c], alpha=0.12, lw=0)
            ax.plot(x, m, color=C[c], marker="o", ms=5, mec=BG, mew=0.8, label=L[c])
        if ref is not None:
            ax.axhline(ref, color=MUTED, ls=(0, (4, 3)), lw=0.8)
            ax.text(10, ref + 2, f"official smolvla_libero {ref:.0f}%", color=MUTED, fontsize=7)
        ax.set_title(f"{b // 1000}k exploration steps mixed in (50% of frames)", color=INK, fontsize=8.5, loc="left")
        ax.set_xticks([10, 25, 40]); ax.set_xlabel("Demos per task"); ax.set_ylim(30, 100); ax.set_xlim(7, 43)
    axes[0].set_ylabel("Task success (%)"); axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/c1_main.pdf"); plt.close(fig)


def replicate_fig(df, out):
    rep = df[(df.explore_pct == 50) & (df.n_demos == 10) & (df["mode"] == "mix")]
    fig, ax = plt.subplots(figsize=(6.6, 2.65)); xt, xl = [], []
    k = 0
    for b in [5000, 20000]:
        for c in ["random", "icm", "icmw"]:
            seeds = sorted(rep[(rep.condition == c) & (rep.explore_steps == b)].explore_seed.unique())
            for es in seeds:
                g = rep[(rep.condition == c) & (rep.explore_steps == b) & (rep.explore_seed == es)].success * 100
                ax.scatter(np.full(len(g), k) + np.linspace(-.12, .12, len(g)), g, s=9, color=C[c], alpha=.5, lw=0)
                ax.plot([k - .25, k + .25], [g.mean()] * 2, color=C[c], lw=2.5, solid_capstyle="round")
                ax.text(k, g.mean() + 3.5, f"{g.mean():.0f}", ha="center", color=INK, fontsize=7)
                xt.append(k); xl.append(f"{L[c].replace('+ ', '').replace(' (trained)', '-w')}\nc{es}\n{b // 1000}k"); k += 1
            if seeds:
                k += .5
        k += .8
    d = df[(df.condition == "demos only") & (df.n_demos == 10)].success.mean() * 100
    ax.axhline(d, color=C["demos only"], lw=1, ls=(0, (4, 3)))
    ax.text(k - 1.2, d + 2, f"demos only {d:.0f}%", color=INK, fontsize=7, ha="right")
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5); ax.set_ylim(0, 100)
    ax.set_ylabel("Task success (%), 10 demos/task")
    fig.tight_layout(); fig.savefig(f"{out}/c2_replicates.pdf"); plt.close(fig)


def ratio_fig(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharey=True)
    for ax, n in zip(axes, [10, 25]):
        d = df[(df.condition == "demos only") & (df.n_demos == n)].success.mean() * 100
        for c in ["random", "icm"]:
            xs, ms = [0], [d]
            for p in [10, 25, 50]:
                g = df[(df.condition == c) & (df.explore_steps == 5000) & (df.n_demos == n) & (df.explore_pct == p)
                       & (df.explore_seed == 0)].success * 100
                if len(g):
                    xs.append(p); ms.append(g.mean())
            ax.plot(xs, ms, color=C[c], marker="o", ms=5, mec=BG, mew=0.8, label=L[c])
        ax.set_title(f"{n} demos per task, 5k exploration steps", loc="left", fontsize=8.5, color=INK)
        ax.set_xticks([0, 10, 25, 50]); ax.set_xlabel("Exploration share of training frames (%)")
        ax.set_ylim(40, 100); ax.set_xlim(-3, 55)
    axes[0].set_ylabel("Task success (%)"); axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/c3_ratio.pdf"); plt.close(fig)


def eval_fig(df_closed, df_open, ref_closed, ref_open, out):
    """Open-loop (n_action_steps 50) vs closed-loop (10) evaluation of the same checkpoints."""
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    labels, a, b, ea, eb = [], [], [], [], []
    for n in (10, 25, 40):
        for d, lst, el in ((df_open, a, ea), (df_closed, b, eb)):
            g = d[(d.condition == "demos only") & (d.n_demos == n) & (d.train_steps == 20000)].success * 100
            lst.append(g.mean()); el.append(g.std(ddof=1))
        labels.append(f"Demos only, {n}/task")
    if ref_open is not None and ref_closed is not None:
        labels.append("Official smolvla_libero"); a.append(ref_open); b.append(ref_closed); ea.append(0); eb.append(0)
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2 - 0.01, a, w, color="#b8c4d4", label="Execute all 50 actions of a chunk (default; std over seeds)", yerr=ea,
           error_kw={"lw": 0.8, "ecolor": MUTED})
    ax.bar(x + w / 2 + 0.01, b, w, color=C["demos only"], label="Replan every 10 actions", yerr=eb,
           error_kw={"lw": 0.8, "ecolor": MUTED})
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, 4, f"{a[i]:.0f}%", ha="center", fontsize=7.5, color=INK)
        ax.text(x[i] + w / 2, 4, f"{b[i]:.0f}%", ha="center", fontsize=7.5, color="#fbfaf7")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0, 110); ax.set_ylabel("Task success (%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/c0_eval.pdf"); plt.close(fig)


def paired_fig(paired, out):
    """Pooled paired differences (all demo budgets) for the main questions, with 95% bootstrap CIs."""
    rows = [("random", 5000, "demos only", "+ random 5k vs demos only"),
            ("random", 20000, "demos only", "+ random 20k vs demos only"),
            ("icm", 5000, "demos only", "+ ICM 5k vs demos only"),
            ("icm", 20000, "demos only", "+ ICM 20k vs demos only"),
            ("icmw", 5000, "demos only", "+ ICM (trained) 5k vs demos only"),
            ("icmw", 20000, "demos only", "+ ICM (trained) 20k vs demos only"),
            ("icm", 5000, "random", "ICM vs random, 5k"),
            ("icm", 20000, "random", "ICM vs random, 20k"),
            ("icmw", 20000, "random", "ICM (trained) vs random, 20k"),
            ("demos only 10k", 0, "demos only", "Demos 10k steps vs 20k"),
            ("random", 20000, "demos only 10k", "+ random 20k vs demos 10k steps"),
            ("icm", 20000, "demos only 10k", "+ ICM 20k vs demos 10k steps"),
            ("seq-random", 20000, "demos only 10k", "Random, then demos vs demos 10k"),
            ("seq-icm", 20000, "demos only 10k", "ICM, then demos vs demos 10k"),
            ("seq-icmw", 20000, "demos only 10k", "ICM (trained), then demos vs demos 10k")]
    p = paired[paired.n_demos.astype(str) == "all"]
    sel = []
    for a, ab, b, lab in rows:
        r = p[(p.a == a) & (p.a_steps == ab) & (p.b == b)]
        if len(r):
            sel.append((lab, r.iloc[0]))
    if not sel:
        return
    fig, ax = plt.subplots(figsize=(6.6, 0.25 * len(sel) + 0.7))
    for i, (lab, r) in enumerate(sel):
        y = len(sel) - 1 - i
        lo, hi = r.t_ci_lo, r.t_ci_hi   # t interval over all pairs: honest when there are few pairs
        col = C["icm"] if lo > 0 else ("#c2410c" if hi < 0 else MUTED)
        ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot([r.mean_diff * 100], [y], "o", color=col, ms=4.5, mec=BG, mew=0.8)
        ax.text(31, y, f"{r.mean_diff * 100:+.1f}  ({int(r.pairs)} pairs)", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axvspan(-5, 5, color=GRID, alpha=0.5, lw=0)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([s[0] for s in sel][::-1], fontsize=7.5)
    ax.set_xlim(-40, 45); ax.set_xlabel("Paired difference in success (points), pooled over 10/25/40 demos; shaded: ±5; bars: 95% t-interval")
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.subplots_adjust(left=0.36); fig.savefig(f"{out}/c4_paired.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--an", default="runs/analysis_nas10")
    ap.add_argument("--an-open", default="runs/analysis")
    ap.add_argument("--out", default="slides/figures")
    a = ap.parse_args()
    df = pd.read_csv(f"{a.an}/runs.csv")
    dfo = pd.read_csv(f"{a.an_open}/runs.csv")
    ref, refo = ref_of(a.an), ref_of(a.an_open)
    main_fig(df, ref, a.out, conds=("demos only", "random", "icm", "icmw"))
    replicate_fig(df, a.out)
    ratio_fig(df, a.out)
    eval_fig(df, dfo, ref, refo, a.out)
    paired_fig(pd.read_csv(f"{a.an}/paired.csv"), a.out)
    print("ok")


if __name__ == "__main__":
    main()
