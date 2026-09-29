#!/usr/bin/env python3
"""Auditable compact figures for the unified September 25 manuscript.

Figure 1a reads the unified per-book JSON and averages the canonical pilot
books 2--7 over five common seeds at 16 measured probes. Its Closed Loop
endpoint therefore uses the same estimator as the canonical H=3.0097 table
row. Figure 1b reports a separate eight-book exposure-density suite normalized
by its own 0%-exposure baseline; its absolute effects are not pooled with
Figure 1a. Lines join measured probes only; there is no fitted interpolation
or mixed-protocol uncertainty ribbon.

The stage aggregates are exact transcriptions of section 1c of
evidence/RESULTS_20260919.md. There are 3 recordings x 8 source rows, each
evaluated on 3 receivers: 72 observations per stage, not 72 independent source
trajectories. Aggregate data cannot support fabricated individual trajectories,
new confidence intervals, or a claim of irreversibility.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
PREVIEW = ROOT / "tmp/pdfs/story_revision"
PREVIEW.mkdir(parents=True, exist_ok=True)

RED = "#AC3E2C"
BLUE = "#3D6997"
GREEN = "#4C7C57"
GOLD = "#B1812B"
PALE_GOLD = "#D9C5A0"
INK = "#252525"
GRAY = "#86817A"
GRID = "#E7E4DF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11.0,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Unreviewed input changes in {path}: {digest}")


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(PREVIEW / f"{stem}.png", dpi=230, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(OUT / f"{stem}.pdf")


def style(ax, grid_axis="y"):
    ax.tick_params(length=3, width=0.7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def read_means():
    # Unified canonical suite: pilot books 2--7, five common seeds, 16 probes.
    # Its endpoint is the same estimator as the canonical H=3.0097 table row.
    path = ROOT / "data/unified_perbook_data.json"
    suites = json.loads(path.read_text())["suites"]

    def curve(suite, arm):
        rows = []
        for book in map(str, range(2, 8)):
            rows.extend(suite[arm][book].values())
        assert len(rows) == 30 and all(len(row) == 16 for row in rows)
        return np.mean(np.asarray(rows, dtype=float), axis=0)

    canonical = suites["125M ext32k p=0.95"]
    fixed = suites["125M ext32k fixedW0"]
    means = {
        "closed": curve(canonical, "treat"),
        "fixed_w0": curve(fixed, "treat"),
        "masked": curve(canonical, "ctrl"),
    }
    assert np.allclose(means["masked"], curve(fixed, "ctrl"))
    return np.arange(8, 129, 8), means


def failure():
    x, means = read_means()
    fig = plt.figure(figsize=(7.2, 3.12))
    ax = fig.add_axes([0.075, 0.29, 0.40, 0.66])
    exposure_ax = fig.add_axes([0.655, 0.29, 0.33, 0.66])
    ax.axvspan(0, 16, color="#F1ECE2", zorder=0)
    labels = {
        "closed": "Closed Loop",
        "replay": "Recorded Replay",
        "fixed_w0": "Fixed Generation",
        "masked": "Writes Off",
    }
    colors = {"closed": RED, "replay": GOLD, "fixed_w0": BLUE, "masked": GREEN}
    for condition in ("closed", "fixed_w0", "masked"):
        ax.plot(
            x,
            means[condition],
            color=colors[condition],
            lw=2.0 if condition == "closed" else 1.65,
            linestyle=(0, (5, 3)) if condition == "masked" else "-",
            label=labels[condition],
            zorder=3,
        )
    ax.set(
        xlim=(0, 130),
        ylim=(3.0, 9.3),
        xticks=[0, 32, 64, 96, 128],
        yticks=[3, 4, 5, 6, 7, 8, 9],
        xlabel="Stream Position (K Tokens)",
        ylabel="Real-Text NLL (Nats)",
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=10.2,
        handlelength=2.1,
        borderaxespad=0.1,
        labelspacing=0.16,
    )
    ax.text(
        2.5,
        6.9,
        "Earlier Evaluation\nHorizon (8K + 8K)",
        color="#8A703F",
        fontsize=9.3,
        ha="left",
        va="top",
    )
    style(ax)

    # September 22 audited exposure-density analysis: 125M mixed stream,
    # eight books, five seeds, and 105 live slots. The line is the evenly
    # spaced schedule; the second 31% point holds density fixed while making
    # external text bursty. This separates exposure density from maximum
    # uninterrupted self-generation length.
    fraction = np.array([0, 5, 10, 20, 31])
    harm = np.array([1.1858, 1.0275, 0.4389, 0.0806, 0.0694])
    lo = np.array([0.5326, 0.5679, 0.2517, 0.0657, 0.0534])
    hi = np.array([1.9430, 1.6195, 0.6465, 0.0981, 0.0843])
    # This sweep uses a dedicated long-book set [0,1,27,28,29,43,44,46],
    # whereas the canonical 3.0097-nat endpoint uses pilot books 2--7.  The
    # absolute levels therefore cannot be pooled.  Normalize within the
    # exposure suite so this panel shows the controlled cadence effect without
    # implying that 1.1858 and the canonical endpoint estimate the same book
    # population.  Intervals below are the published H intervals expressed in
    # units of the sweep's 0%-exposure point estimate.
    h0 = harm[0]
    harm, lo, hi = harm / h0, lo / h0, hi / h0
    burst, burst_lo, burst_hi = 0.7155 / h0, 0.3730 / h0, 1.1578 / h0
    exposure_ax.errorbar(
        fraction,
        harm,
        yerr=[harm - lo, hi - harm],
        fmt="o-",
        color=BLUE,
        lw=1.9,
        ms=5.1,
        capsize=2.3,
        zorder=3,
    )
    exposure_ax.errorbar(
        [31],
        [burst],
        yerr=[[burst - burst_lo], [burst_hi - burst]],
        fmt="s",
        color=RED,
        ms=5.1,
        capsize=2.3,
        zorder=4,
        label="31% Bursty",
    )
    exposure_ax.annotate(
        "same 33 real chunks,\ngrouped together",
        xy=(31, burst),
        xytext=(14.4, 1.34),
        fontsize=8.2,
        color=RED,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.75, color=RED),
    )
    exposure_ax.text(
        0.02,
        0.97,
        r"Long-book suite; $H_{0\%}=1.186$ nats",
        transform=exposure_ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    exposure_ax.set(
        xlim=(-1.5, 33.5),
        ylim=(-0.04, 1.72),
        xticks=[0, 5, 10, 20, 31],
        yticks=[0, 0.5, 1, 1.5],
        xlabel="Real-Text Write Slots (%)",
        ylabel=r"Relative Harm, $H/H_{0\%}$",
    )
    style(exposure_ax)
    for a in (ax, exposure_ax):
        a.tick_params(labelsize=11)
        a.xaxis.label.set_size(11)
        a.yaxis.label.set_size(11)
    fig.text(
        0.275,
        0.105,
        "(a) Prediction Degrades Beyond\nthe Earlier Horizon",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    fig.text(
        0.82,
        0.105,
        "(b) Frequent External Text\nInterrupts the Loop",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    save(fig, "fig_failure_story")
    print(
        "E1 endpoint differences:",
        {
            c: float((y[-1] - y[0]) - (means["masked"][-1] - means["masked"][0]))
            for c, y in means.items()
        },
    )


def controls():
    # Replay values are the frozen audited E1 aggregates reported in the
    # manuscript: read-only 1.606 and read+write 3.874 nats over Writes Off.
    h = {"replay_off": 1.606, "replay": 3.874}
    fig = plt.figure(figsize=(8.2, 2.48))
    drift_ax = fig.add_axes([0.09, 0.36, 0.115, 0.56])
    damage_ax = fig.add_axes([0.255, 0.36, 0.14, 0.56])
    b = fig.add_axes([0.485, 0.36, 0.18, 0.56])
    c = fig.add_axes([0.775, 0.36, 0.205, 0.56])

    # Acceptance P2 audited drift suite: 125M, eight books, five seeds,
    # 128 chunks, historical boundaries. Error bars are paired book-bootstrap
    # intervals for final clean-text NLL relative to Writes Off. The horizontal
    # coordinate is final ||W_t-W_0||/||W_0|| from the same trajectories.
    p2 = [
        ("Closed Loop", 0.00857, 2.8308, 1.8171, 3.8598, RED),
        ("Fixed Generation", 0.01387, 0.0707, 0.0510, 0.0924, BLUE),
        ("Real-Text Learning", 0.01759, -0.3544, -0.6048, -0.1236, GOLD),
        ("Writes Off", 0.00500, 0.0, 0.0, 0.0, GREEN),
    ]
    ypos = np.arange(3, -1, -1)
    for y_i, (label, drift, gap, lo, hi, color) in zip(ypos, p2):
        drift_ax.hlines(y_i, 0, drift, color=color, lw=2.3, alpha=0.9)
        drift_ax.plot(drift, y_i, "o", ms=5.2, color=color, zorder=3)
        xerr = None if label == "Writes Off" else [[gap - lo], [hi - gap]]
        damage_ax.errorbar(
            gap,
            y_i,
            xerr=xerr,
            fmt="o",
            ms=5.2,
            lw=1.4,
            capsize=2.4,
            color=color,
            zorder=3,
        )
    drift_ax.set(
        yticks=ypos,
        yticklabels=[row[0] for row in p2],
        xlim=(0, 0.0205),
        xticks=[0, 0.01, 0.02],
        ylim=(-0.55, 3.55),
        xlabel="Relative Drift",
    )
    drift_ax.tick_params(axis="y", length=0, labelsize=9.5)
    drift_ax.set_title("Parameter Change", fontsize=10.5, pad=5)
    style(drift_ax, "x")
    damage_ax.axvline(0, color=GRAY, lw=0.75, ls="--")
    damage_ax.set(
        yticks=ypos,
        yticklabels=[],
        xlim=(-1.0, 4.15),
        xticks=[-1, 0, 2, 4],
        ylim=(-0.55, 3.55),
        xlabel=r"Endpoint Gap, $E$ (Nats)",
    )
    damage_ax.tick_params(axis="y", length=0)
    damage_ax.set_title("Prediction Damage", fontsize=10.5, pad=5)
    style(damage_ax, "x")

    keys = ["replay_off", "replay"]
    labels = ["Read Only", "Read + Write"]
    colors = [PALE_GOLD, GOLD]
    values = [h[k] for k in keys]
    y = np.array([1, 0])
    b.barh(y, values, color=colors, height=0.48, zorder=3)
    for yi, value, color in zip(y, values, colors):
        b.plot(value, yi, "o", ms=3.5, color=color, zorder=4)
        b.text(value + 0.10, yi, f"{value:.3f}", va="center", fontsize=10.5)
    b.set(
        yticks=y,
        yticklabels=labels,
        xlim=(0, 5.05),
        ylim=(-0.50, 1.87),
        xticks=[0, 2, 4],
        xlabel=r"Excess NLL Change, $H$ (Nats)",
    )
    style(b, "x")
    b.tick_params(axis="y", length=0)
    delta = h["replay"] - h["replay_off"]
    b.plot(
        [h["replay_off"], h["replay_off"], h["replay"], h["replay"]],
        [1.36, 1.48, 1.48, 1.36],
        color=GOLD,
        lw=0.9,
    )
    b.text(
        (h["replay_off"] + h["replay"]) / 2,
        1.59,
        f"+{delta:.3f} From Writing",
        ha="center",
        va="bottom",
        color=GOLD,
        fontsize=10.5,
    )
    # Paired one-update comparison (125M), measured at four stream positions.
    # The text and state both evolve with position, so this panel shows the
    # state-dependent amplification rather than assigning it to either factor.
    position = np.array([1, 33, 65, 97])
    paths = [
        ("Closed Loop", [0.0062, 0.0087, 0.0250, 0.1126], RED),
        ("Writes Off", [0.0063, 0.0044, 0.0038, 0.0037], GREEN),
        ("Fixed Generation", [0.0042, 0.0008, 0.0011, 0.0012], BLUE),
    ]
    for label, values, color in paths:
        c.plot(
            position, values, "o-", color=color, lw=1.55, ms=3.8, label=label, zorder=3
        )
    c.annotate(
        "Closed Loop",
        xy=(97, 0.1126),
        xytext=(56, 0.098),
        fontsize=8.4,
        color=RED,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.7),
    )
    c.set(
        xlim=(-2, 101),
        ylim=(-0.006, 0.13),
        xticks=position,
        yticks=[0, 0.04, 0.08, 0.12],
        xlabel="Stream Position",
        ylabel="One-Write Cost (Nats)",
    )
    style(c)
    c.tick_params(labelsize=9.2)
    c.xaxis.label.set_size(9.6)
    c.yaxis.label.set_size(9.6)
    c.legend(
        frameon=False,
        loc="upper left",
        fontsize=7.2,
        handlelength=1.3,
        labelspacing=0.15,
    )

    fig.text(
        0.245,
        0.035,
        "(a) Drift Does Not Rank Damage",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.575,
        0.035,
        "(b) Same Text, Different Writes",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.875,
        0.035,
        "(c) Feedback Amplifies One-Write Cost",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    save(fig, "fig_controls_story")


def stages():
    # Section 1c, RESULTS_20260919.md. Exact reported aggregates only.
    stage = np.array([1, 17, 33, 49, 65, 81, 97])
    mean = np.array([0.0439, 0.0898, 0.1399, 0.3135, 0.4855, 0.4858, 0.7512])
    median = np.array([0.0422, 0.0040, 0.0039, 0.0042, 0.0047, 0.0105, 0.0064])
    count = np.array([0, 3, 3, 6, 9, 9, 12])
    fig = plt.figure(figsize=(6.5, 2.62))
    a = fig.add_axes([0.095, 0.31, 0.35, 0.67])
    b = fig.add_axes([0.63, 0.31, 0.345, 0.67])
    a.plot(stage, mean, "o-", color=RED, ms=4.1, lw=1.7, label="Mean")
    a.plot(
        stage,
        median,
        "s--",
        color=BLUE,
        ms=3.8,
        lw=1.4,
        markerfacecolor="white",
        label="Median",
    )
    a.set(
        xlim=(-3, 105),
        ylim=(-0.04, 0.90),
        xticks=stage,
        yticks=[0, 0.2, 0.4, 0.6, 0.8],
        xlabel="Source Chunk Position",
        ylabel="Passage-Level Write Cost (Nats)",
    )
    a.legend(frameon=False, loc="upper left", labelspacing=0.2)
    a.text(97, mean[-1] + 0.04, "0.751", ha="center", fontsize=10.5, color=RED)
    a.text(97, median[-1] + 0.04, "0.006", ha="center", fontsize=10.5, color=BLUE)
    b.bar(stage, count, width=8, color=GOLD, zorder=3)
    for pos, n in zip(stage, count):
        b.text(pos, n + 0.35, str(n), ha="center", va="bottom", fontsize=10.5)
    b.set(
        xlim=(-7, 105),
        ylim=(0, 14),
        xticks=stage,
        yticks=[0, 3, 6, 9, 12],
        xlabel="Source Chunk Position",
        ylabel="Count Above 0.5 Nats",
    )
    for ax in (a, b):
        style(ax)
        ax.tick_params(axis="x", labelsize=10.5)
    fig.text(
        0.275,
        0.045,
        "(a) Mean and Median Differ",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.795,
        0.045,
        "(b) High-Cost Cases (Out of 72)",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    save(fig, "fig_stage_story")


def breadth():
    """Adam control plus exploratory nine-model LoRA coverage; not pooled."""
    fig = plt.figure(figsize=(7.2, 2.85))
    a = fig.add_axes([0.14, 0.27, 0.275, 0.66])
    b = fig.add_axes([0.66, 0.27, 0.315, 0.66])
    for y, m, lo, hi in [(1, 0.026, -0.071, 0.122), (0, 1.231, 0.580, 2.350)]:
        a.errorbar(
            m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=RED, capsize=3, lw=1.5, ms=5
        )
    a.axvline(0, color=GRAY, lw=0.8, ls="--")
    a.set(
        yticks=[1, 0],
        yticklabels=[r"Adam $10^{-5}$", r"Adam $10^{-4}$"],
        ylim=(-0.7, 1.7),
        xlim=(-0.2, 2.7),
        xticks=[0, 1, 2],
        xlabel=r"Added NLL, $H$ (Nats)",
    )
    a.text(
        0.5,
        0.98,
        "8 Books, 3 Seeds",
        transform=a.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=GRAY,
    )
    models = [
        ("Gemma-4-12B", 0.183, 0.109),
        ("Gemma-4-12B-it", 5.030, -0.861),
        ("Gemma-4-31B-it", 2.243, 0.289),
        ("OLMo-2-1124-7B", 0.328, 0.209),
        ("Phi-4", 0.190, 0.148),
        ("Pythia-2.8B", 1.793, 0.107),
        ("Qwen3-8B", 0.232, 0.128),
        ("Qwen2.5-3B", 0.389, 0.193),
        ("SmolLM2-1.7B", 1.023, 0.176),
    ]
    for y, (name, u, m) in enumerate(reversed(models)):
        b.plot(u - m, y, "o", color=RED, ms=4.5)
        b.annotate(
            f"{u - m:.3f}",
            (u - m, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
        )
    b.set(
        yticks=range(9),
        yticklabels=[m[0] for m in reversed(models)],
        xscale="log",
        xlim=(0.025, 20),
        ylim=(-0.5, 8.7),
        xlabel="Uniform-minus-Masked\nNLL Change (Nats; Log Scale)",
    )
    b.set_xticks([0.03, 0.1, 0.3, 1, 3, 10], [".03", ".1", ".3", "1", "3", "10"])
    b.xaxis.set_minor_locator(plt.NullLocator())
    for ax in (a, b):
        style(ax, "x")
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.label.set_size(11)
    fig.text(0.255, 0.015, "(a) In-Place Adam: Qwen3-4B", ha="center", fontsize=11.5)
    fig.text(0.765, 0.015, "(b) Online LoRA: Nine Models", ha="center", fontsize=11.5)
    save(fig, "fig_breadth_story")


def repairs():
    """Expanded policy controls and matched read-only slots, exact report means."""
    fig = plt.figure(figsize=(7.2, 2.85))
    a = fig.add_axes([0.205, 0.28, 0.265, 0.68])
    b = fig.add_axes([0.735, 0.28, 0.25, 0.68])
    rows = [
        ("Closed Loop", 2.8853, 2.0072, 3.8310, RED),
        ("Uniform Weights", 2.4861, 1.9510, 3.0939, GOLD),
        ("Random Weights", 2.0558, 1.6339, 2.4946, GOLD),
        ("Repetition Weights", 1.5637, 1.1858, 1.9825, GOLD),
        ("Read-Only Real Text", 1.3979, 1.0988, 1.7140, BLUE),
        ("Writes Off", 0.0, 0.0, 0.0, GRAY),
    ]
    slots = [
        ("Slot Skipped", 0.0, 0.0, 0.0, GRAY),
        ("Own Generation", 0.3172, -0.4775, 1.1963, GRAY),
        ("Random Tokens", 0.9927, -0.2699, 2.0992, GRAY),
        ("Shuffled Real Text", -0.8362, -1.7643, 0.0324, GRAY),
        (r"Fixed-$W_0$ Text", -0.6130, -1.5127, 0.2200, GRAY),
        ("Real Text", -1.5693, -2.3011, -0.8433, BLUE),
    ]
    for ax, data, lim, ticks, xlabel in [
        (a, rows, (-0.15, 4.2), [0, 1, 2, 3, 4], "Final NLL Above Mask (Nats)"),
        (b, slots, (-2.65, 2.5), [-2, 0, 2], "NLL vs. Skipped Slot (Nats)"),
    ]:
        for y, (name, m, lo, hi, color) in enumerate(data):
            ax.errorbar(
                m,
                y,
                xerr=[[m - lo], [hi - m]],
                fmt="o",
                ms=4.8,
                lw=1.35,
                capsize=2.3,
                color=color,
                mfc="white" if m == 0 else color,
            )
        ax.set(
            yticks=range(len(data)),
            yticklabels=[r[0] for r in data],
            ylim=(len(data) - 0.5, -0.6),
            xlim=lim,
            xticks=ticks,
            xlabel=xlabel,
        )
        ax.axvline(0, color=GRAY, lw=0.7, ls="--")
        style(ax, "x")
        ax.tick_params(axis="y", length=0, labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.label.set_size(11)
    fig.text(0.335, 0.025, "(a) Repairs Still Leave Damage", ha="center", fontsize=11.5)
    fig.text(
        0.775, 0.025, "(b) What Fills the Read-Only Slot?", ha="center", fontsize=11.5
    )
    save(fig, "fig_repairs_story")


def read_only_schedule():
    """Vector protocol illustration; no experimental values are implied."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(2.85, 1.12))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for x, width, color, label, operation in [
        (0.02, 0.44, "#F1DCD7", "Generated ×7", "Read + Update"),
        (0.59, 0.39, "#DCE8F2", "Real ×1", "Read Only"),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.47),
                width,
                0.36,
                boxstyle="round,pad=0.015,rounding_size=0.045",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(x + width / 2, 0.65, label, ha="center", va="center", fontsize=11)
        ax.text(x + width / 2, 0.28, operation, ha="center", va="center", fontsize=10.5)
    ax.annotate(
        "",
        xy=(0.58, 0.65),
        xytext=(0.48, 0.65),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0),
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(fig, "fig_read_only_schedule")


if __name__ == "__main__":
    if "--controls-only" in sys.argv:
        controls()
    else:
        failure()
        controls()
        stages()
        breadth()
        repairs()
        read_only_schedule()
