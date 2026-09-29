#!/usr/bin/env python3
"""Build Figure 5 from the reported Settlement and provenance tables.

Panel (a) uses the separate 125M/760M validation suites reported in Appendix E.
Panel (b) uses the 125M, 31%-real-slot, three-seed corruption sweep reported in
Appendix F. These are distinct protocols and are not pooled with each other or
with the canonical long-horizon H estimates.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INK = "#252525"
GRAY = "#77736D"
LIGHT = "#D7D4CE"
GRID = "#E7E4DF"
BLUE = "#3D6997"
RED = "#AC3E2C"
GREEN = "#4C7C57"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.6,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
    }
)

fig = plt.figure(figsize=(7.1, 2.58))
grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.55), wspace=0.38)

# (a) The former main-text table, expressed as the comparison readers need.
ax = fig.add_subplot(grid[0, 0])
models = ["125M", "760M"]
y = np.array([1.0, 0.0])
closed = np.array([3.183, 1.263])
validated = np.array([0.067, -0.021])
accepted = ["22/936 generated-text writes kept", "18/312 generated-text writes kept"]

ax.axvline(0, color=GRAY, lw=1.0, ls=(0, (3, 2)), zorder=1)
for yi, c, v, kept in zip(y, closed, validated, accepted):
    ax.annotate(
        "",
        xy=(v, yi),
        xytext=(c, yi),
        arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=2.2, shrinkA=5, shrinkB=5),
        zorder=2,
    )
    ax.plot(c, yi, "o", color=RED, ms=6.0, zorder=4)
    ax.plot(v, yi, "o", color=BLUE, ms=6.0, zorder=4)
    ax.text(
        c,
        yi + 0.16,
        f"{c:.2f}",
        color=RED,
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        v,
        yi + 0.16,
        f"{v:.2f}",
        color=BLUE,
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(0.10, yi - 0.22, kept, color=GRAY, fontsize=7.5, ha="left", va="top")

ax.text(0, 1.53, "Writes Off", color=GRAY, fontsize=7.8, ha="center", va="bottom")
ax.plot([], [], "o", color=RED, label="Closed Loop")
ax.plot([], [], "o", color=BLUE, label="Settlement")
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.57, 1.18),
    ncol=2,
    frameon=False,
    handletextpad=0.35,
    columnspacing=0.9,
    fontsize=8.1,
)
ax.set_xlim(-0.25, 3.48)
ax.set_ylim(-0.50, 1.58)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks(y, models)
ax.set_xlabel(r"Endpoint gap $E$ (nats; lower is better)")
ax.grid(axis="x", color=GRID, lw=0.55)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=5)

# (b) Provenance corruption: validation uses the candidate's measured effect.
ax = fig.add_subplot(grid[0, 1])
x = np.array([0.000, 0.053, 0.124, 0.212, 0.389, 0.619, 0.876, 1.000]) * 100
mask = np.array([3.8072, 3.8116, 3.8129, 3.8377, 3.8584, 3.9242, 4.0204, 4.2544])
settle = np.array([3.8196, 3.8230, 3.8248, 3.8286, 3.8215, 3.8270, 3.8243, 3.8244])

ax.axvline(38.9, color="#AAB9C7", lw=0.8, ls=(0, (2, 2)), zorder=1)
ax.plot(x, mask, "o-", color=RED, lw=2.0, ms=4.2, label="Source Masking", zorder=4)
ax.plot(x, settle, "o-", color=BLUE, lw=2.0, ms=4.2, label="Settlement", zorder=4)
ax.axhline(3.8532, color=GRAY, lw=1.0, ls=(0, (5, 3)), label="All Writes", zorder=2)
ax.axhline(
    3.8108, color=GREEN, lw=1.0, ls=(0, (2, 2)), label="Oracle Source Masking", zorder=2
)
ax.set(
    xlim=(-2, 104),
    ylim=(3.77, 4.30),
    xticks=[0, 20, 39, 60, 80, 100],
    xlabel="Corrupted Source Labels (%)",
    ylabel="Final Real-Text NLL (lower is better)",
)
ax.grid(axis="y", color=GRID, linewidth=0.55)
ax.set_axisbelow(True)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.50, 1.18),
    ncol=2,
    frameon=False,
    handlelength=1.9,
    handletextpad=0.45,
    columnspacing=1.0,
    fontsize=7.8,
)

fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.29)
fig.text(
    0.225,
    0.035,
    "(a) Settlement closes the endpoint gap",
    ha="center",
    va="bottom",
    fontsize=9.2,
    fontweight="bold",
)
fig.text(
    0.745,
    0.035,
    "(b) Settlement remains stable as labels fail",
    ha="center",
    va="bottom",
    fontsize=9.2,
    fontweight="bold",
)

out = ROOT / "figures/fig_settlement_story.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(out)
