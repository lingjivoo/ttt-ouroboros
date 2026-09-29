#!/usr/bin/env python3
"""Draw the four commitment policies compared in Table 10."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
PREVIEW = ROOT / "tmp" / "pdfs" / "commitment_policies"
PREVIEW.mkdir(parents=True, exist_ok=True)

INK = "#252525"
GRAY = "#F2F1EE"
MID_GRAY = "#817D75"
BLUE = "#DCE9F3"
BLUE_DARK = "#3D6997"
RED = "#F2DCD6"
RED_DARK = "#AA3E2D"
GREEN = "#E0EADB"
GREEN_DARK = "#4C7C57"
GOLD = "#F5E8CA"
GOLD_DARK = "#A97215"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, x, y, w, h, text, face=GRAY, edge="none", fontsize=10.3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.25):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def panel_title(ax, label, title, color=INK):
    ax.text(
        0.5,
        -0.08,
        f"({label}) {title}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.2,
        color=color,
    )


def ordinary(ax):
    box(ax, 0.03, 0.49, 0.19, 0.23, r"$W$", face=GRAY)
    box(ax, 0.38, 0.49, 0.22, 0.23, r"$W+\delta_1$", face=BLUE)
    box(ax, 0.76, 0.49, 0.22, 0.23, r"$W+\delta_1+\delta_2$", face=BLUE)
    arrow(ax, 0.22, 0.605, 0.38, 0.605)
    arrow(ax, 0.60, 0.605, 0.76, 0.605)
    ax.text(0.30, 0.68, "write", ha="center", color=MID_GRAY, fontsize=9)
    ax.text(0.68, 0.68, "write", ha="center", color=MID_GRAY, fontsize=9)
    ax.text(
        0.5,
        0.27,
        "Every update changes the live state",
        ha="center",
        color=BLUE_DARK,
        fontsize=9.7,
    )
    panel_title(ax, "a", "Ordinary Writing", BLUE_DARK)


def joint(ax):
    box(ax, 0.02, 0.59, 0.18, 0.20, r"$W$", face=GRAY)
    box(ax, 0.31, 0.67, 0.22, 0.20, r"$W+\delta_1$", face=BLUE)
    box(ax, 0.31, 0.35, 0.22, 0.20, r"$W+\delta_2$", face=BLUE)
    arrow(ax, 0.20, 0.69, 0.31, 0.77)
    arrow(ax, 0.20, 0.69, 0.31, 0.45)
    ax.text(0.42, 0.91, "validate on $q$", ha="center", fontsize=8.8, color=MID_GRAY)
    ax.text(0.42, 0.27, "validate on $q$", ha="center", fontsize=8.8, color=MID_GRAY)
    box(
        ax,
        0.68,
        0.49,
        0.30,
        0.24,
        r"Commit $W+\delta_1+\delta_2$",
        face=RED,
        fontsize=8.9,
    )
    arrow(ax, 0.53, 0.77, 0.68, 0.64, color=RED_DARK)
    arrow(ax, 0.53, 0.45, 0.68, 0.58, color=RED_DARK)
    ax.text(
        0.83,
        0.27,
        "combined state was not validated",
        ha="center",
        color=RED_DARK,
        fontsize=8.8,
    )
    panel_title(ax, "b", "Individual Validation, Joint Commit", RED_DARK)


def sequential(ax):
    box(ax, 0.01, 0.49, 0.18, 0.23, r"$S_0=W$", face=GRAY)
    box(ax, 0.34, 0.49, 0.22, 0.23, r"$S_1$", face=BLUE)
    box(ax, 0.72, 0.49, 0.22, 0.23, r"$S_2$", face=GREEN)
    arrow(ax, 0.19, 0.605, 0.34, 0.605)
    arrow(ax, 0.56, 0.605, 0.72, 0.605)
    ax.text(
        0.265, 0.75, r"validate $\delta_1$", ha="center", fontsize=9.1, color=MID_GRAY
    )
    ax.text(
        0.64, 0.75, r"validate $\delta_2$", ha="center", fontsize=9.1, color=MID_GRAY
    )
    ax.text(
        0.83,
        0.28,
        r"commit the validated $S_2$",
        ha="center",
        color=GREEN_DARK,
        fontsize=9.7,
    )
    panel_title(ax, "c", "Sequential Validation", GREEN_DARK)


def settlement(ax):
    box(ax, 0.01, 0.54, 0.20, 0.21, r"candidate $\delta_t$", face=BLUE, fontsize=9.3)
    box(ax, 0.30, 0.54, 0.17, 0.21, "pending", face=BLUE)
    box(
        ax,
        0.57,
        0.50,
        0.23,
        0.29,
        "evaluate\n" + r"$S$ vs. $S+\delta_t$",
        face=GRAY,
        fontsize=8.9,
    )
    box(ax, 0.88, 0.48, 0.11, 0.33, "keep\n/\ndiscard", face=GREEN, fontsize=8.8)
    arrow(ax, 0.21, 0.645, 0.30, 0.645)
    arrow(ax, 0.47, 0.645, 0.57, 0.645)
    arrow(ax, 0.80, 0.645, 0.88, 0.645)
    box(ax, 0.58, 0.14, 0.21, 0.20, r"future text $q$", face=GOLD)
    arrow(ax, 0.685, 0.34, 0.685, 0.50, color=GOLD_DARK)
    ax.text(0.69, 0.39, "$q$ arrives", ha="center", fontsize=8.7, color=GOLD_DARK)
    panel_title(ax, "d", "Settlement", BLUE_DARK)


def main():
    fig = plt.figure(figsize=(7.25, 4.25))
    axes = [
        fig.add_axes([0.04, 0.58, 0.43, 0.34]),
        fig.add_axes([0.53, 0.58, 0.43, 0.34]),
        fig.add_axes([0.04, 0.12, 0.43, 0.34]),
        fig.add_axes([0.53, 0.12, 0.43, 0.34]),
    ]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    ordinary(axes[0])
    joint(axes[1])
    sequential(axes[2])
    settlement(axes[3])
    fig.text(
        0.5,
        0.015,
        "Validation text is read-only; outcome evaluation uses separate real text.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=MID_GRAY,
    )
    pdf = OUT / "fig_commitment_policies.pdf"
    png = PREVIEW / "fig_commitment_policies.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(pdf)


if __name__ == "__main__":
    main()
