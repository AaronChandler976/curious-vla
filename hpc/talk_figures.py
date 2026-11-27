"""Figures for the talk (vector PDF), from the analysis outputs.

  python hpc/talk_figures.py runs/analysis slides/figures
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
an, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{an}/runs.csv")
C = {"demos only": "#2a78d6", "random": "#eb6834", "icm": "#1baf7a", "icmw": "#8e5cd9"}
L = {"demos only": "Demos only", "random": "+ random", "icm": "+ ICM", "icmw": "+ ICM (trained)"}
INK, MUTED, GRID, BG = "#1d232b", "#5a6270", "#e3e1da", "#fbfaf7"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "lines.linewidth": 1.6, "pdf.fonttype": 42})
main = df[(df.explore_seed == 0) & df.explore_pct.isin([0, 50])]
def cell(c, b, n):
    g = main[(main.condition == c) & (main.explore_steps == b) & (main.n_demos == n)].success * 100
    return g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(len(g)), g
# C1
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
for ax, b in zip(axes, [5000, 20000]):
    for i, c in enumerate(["demos only", "random", "icm"]):
        ns = [10, 25, 40]; st = 0 if c == "demos only" else b
        m = [cell(c, st, n)[0] for n in ns]; e = [cell(c, st, n)[1] for n in ns]
        x = np.array(ns) + (i - 1) * 0.8
        ax.fill_between(x, np.array(m) - e, np.array(m) + e, color=C[c], alpha=0.12, lw=0)
        ax.plot(x, m, color=C[c], marker="o", ms=5, mec=BG, mew=0.8, label=L[c])
        dy = {"icm": 10, "demos only": -10}.get(c, 0) if b == 5000 else 0
        ax.annotate(L[c], (x[-1], m[-1]), xytext=(5, dy*0.6), textcoords="offset points", va="center", color=INK, fontsize=7.5)
    ax.axhline(64, color=MUTED, ls=(0, (4, 3)), lw=0.8); ax.text(10, 66, "official smolvla_libero 64%", color=MUTED, fontsize=7)
    ax.set_title(f"{b // 1000}k exploration steps mixed in (50% of frames)", color=INK, fontsize=8.5, loc="left")
    ax.set_xticks([10, 25, 40]); ax.set_xlabel("Demos per task"); ax.set_ylim(0, 85); ax.set_xlim(7, 49)
axes[0].set_ylabel("Task success (%)"); axes[0].legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/c1_main.pdf"); plt.close(fig)
# C2 replicates at n10
rep = df[df.explore_pct.isin([50]) & (df.n_demos == 10)]
fig, ax = plt.subplots(figsize=(6.6, 2.65)); xt, xl = [], []
k = 0
for b in [5000, 20000]:
    for c in ["random", "icm"]:
        for es in sorted(rep[(rep.condition == c) & (rep.explore_steps == b)].explore_seed.unique()):
            g = rep[(rep.condition == c) & (rep.explore_steps == b) & (rep.explore_seed == es)].success * 100
            ax.scatter(np.full(len(g), k) + np.linspace(-.12, .12, len(g)), g, s=9, color=C[c], alpha=.5, lw=0)
            ax.plot([k - .25, k + .25], [g.mean()] * 2, color=C[c], lw=2.5, solid_capstyle="round")
            ax.text(k, g.mean() + 3.5, f"{g.mean():.0f}", ha="center", color=INK, fontsize=7.5)
            xt.append(k); xl.append(f"{L[c].replace('+ ', '')}\ncoll. {es}\n{b // 1000}k"); k += 1
        k += .5
    k += .8
d = main[(main.condition == "demos only") & (main.n_demos == 10)].success.mean() * 100
ax.axhline(d, color=C["demos only"], lw=1, ls=(0, (4, 3))); ax.text(k - 1.2, d + 2, f"demos only {d:.0f}%", color=INK, fontsize=7, ha="right")
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5); ax.set_ylim(0, 85); ax.set_ylabel("Task success (%), 10 demos/task")
fig.tight_layout(); fig.savefig(f"{out}/c2_replicates.pdf"); plt.close(fig)
# C3 mixing ratio
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharey=True)
for ax, n in zip(axes, [10, 25]):
    d = df[(df.condition == "demos only") & (df.n_demos == n)].success.mean() * 100
    for c in ["random", "icm"]:
        xs, ms = [0], [d]
        for p in [10, 25, 50]:
            g = df[(df.condition == c) & (df.explore_steps == 5000) & (df.n_demos == n) & (df.explore_pct == p) & (df.explore_seed == 0)].success * 100
            if len(g): xs.append(p); ms.append(g.mean())
        ax.plot(xs, ms, color=C[c], marker="o", ms=5, mec=BG, mew=0.8, label=L[c])
        ax.annotate(L[c], (xs[-1], ms[-1]), xytext=(8, {"random": 7, "icm": -7}[c] if n == 10 else 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_title(f"{n} demos per task, 5k exploration steps", loc="left", fontsize=8.5, color=INK)
    ax.set_xticks([0, 10, 25, 50]); ax.set_xlabel("Exploration share of training frames (%)"); ax.set_ylim(0, 70); ax.set_xlim(-3, 60)
axes[0].set_ylabel("Task success (%)"); axes[0].legend(frameon=False, loc="lower left", fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/c3_ratio.pdf"); plt.close(fig)
print("ok")
