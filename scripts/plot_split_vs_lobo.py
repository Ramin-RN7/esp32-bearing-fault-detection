"""Scatter of each ablation pipeline's balanced accuracy on the thesis split against leave-one-bearing-out.

Reads results/ablation_lobo.csv, writes results/split_vs_lobo.png. Bubble size = number of pipelines at the same point.
Run: python scripts/plot_split_vs_lobo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "ablation_lobo.csv")
x, y = "original_split_recording_ba", "lobo_recording_ba"
pts = df.groupby([x, y]).size().reset_index(name="n")
r = df[x].corr(df[y])
rho = df[x].corr(df[y], method="spearman")

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
ax.scatter(pts[x], pts[y], s=30 + 28 * pts["n"], color="#2f75b5", alpha=0.45, edgecolor="#17365d", linewidth=0.6)
for pid, label, dx, dy in (("A5-MLP", "A5 MLP", -0.012, 0.012), ("A16-MLP", "A16 MLP", -0.012, 0.014), ("A8-GMM", "A8 GMM", 0.012, -0.022)):
    row = df[df["id"] == pid].iloc[0]
    ax.annotate(label, (row[x], row[y]), xytext=(row[x] + dx, row[y] + dy), fontsize=8, ha="right" if dx < 0 else "left")
ax.plot([0.4, 1.0], [0.4, 1.0], color="#999999", linewidth=0.8, linestyle="--")
ax.set_xlim(0.4, 0.97)
ax.set_ylim(0.45, 1.0)
ax.set_xlabel("Balanced accuracy on the thesis split (9 test recordings)")
ax.set_ylabel("Balanced accuracy on unseen bearings\n(leave-one-bearing-out, 60 recordings)")
ax.set_title(f"{len(df)} ablation pipelines: Pearson r = {r:.2f}, Spearman ρ = {rho:.2f}", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
out = ROOT / "results" / "split_vs_lobo.png"
fig.savefig(out)
print(out, f"r={r:.3f} rho={rho:.3f}")
