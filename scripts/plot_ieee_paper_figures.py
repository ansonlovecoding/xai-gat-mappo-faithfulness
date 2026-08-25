#!/usr/bin/env python3
"""Generate data-driven figures used by the IEEE-style paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "story_freeze_v1" / "audit"
OUTPUT_DIR = ROOT / "docs" / "figures"

COLORS = {
    "uniform": "#B33A3A",
    "matched": "#236B8E",
    "median": "#236B8E",
    "p90": "#D28B26",
    "cap": "#66717E",
    "grid": "#D7DCE1",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_capability_spectrum() -> None:
    data = load_json(AUDIT_DIR / "capability_spectrum_clean.json")
    rows = data["rows"]

    pickups = np.array([row["mean_pickups"] for row in rows])
    labels = [
        "B2 e0",
        "B2 e50",
        "B2 best",
        "B3 best",
        "H5 s42",
        "H5 s43",
        "H5 s44",
    ]

    uniform = np.array([row["def_m_uniform"]["mean"] for row in rows])
    uniform_ci = np.array([row["def_m_uniform"]["ci95"] for row in rows])
    matched = np.array([row["def_m_type_matched"]["mean"] for row in rows])
    matched_ci = np.array([row["def_m_type_matched"]["ci95"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].errorbar(
        pickups,
        uniform,
        yerr=np.vstack((uniform - uniform_ci[:, 0], uniform_ci[:, 1] - uniform)),
        fmt="o",
        color=COLORS["uniform"],
        capsize=2,
        label="Uniform baseline",
    )
    axes[0].errorbar(
        pickups,
        matched,
        yerr=np.vstack((matched - matched_ci[:, 0], matched_ci[:, 1] - matched)),
        fmt="s",
        color=COLORS["matched"],
        capsize=2,
        label="Type-matched baseline",
    )
    label_offsets = {
        "B2 e0": (3, 3),
        "B2 e50": (3, -12),
        "B2 best": (3, 5),
        "B3 best": (3, 5),
        "H5 s42": (3, -12),
        "H5 s43": (3, 5),
        "H5 s44": (-32, 5),
    }
    for x, y, label in zip(pickups, uniform, labels):
        axes[0].annotate(
            label,
            (x, y),
            xytext=label_offsets[label],
            textcoords="offset points",
            fontsize=6,
        )
    axes[0].set_title("(a) Full metric range")
    axes[0].set_xlabel("Mean pickups per episode")
    axes[0].set_ylabel("Clean margin-DEF")
    axes[0].set_xlim(0.4, 12.4)
    axes[0].set_ylim(-2.25, 2.05)
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].axhline(0, color="#333333", linewidth=0.8)
    trained = np.arange(1, len(rows))
    axes[1].errorbar(
        trained,
        matched[1:],
        yerr=np.vstack(
            (
                matched[1:] - matched_ci[1:, 0],
                matched_ci[1:, 1] - matched[1:],
            )
        ),
        fmt="s",
        color=COLORS["matched"],
        capsize=2,
    )
    axes[1].set_xticks(trained, labels[1:], rotation=32, ha="right")
    axes[1].set_title("(b) Corrected scores for trained checkpoints")
    axes[1].set_ylabel("Type-matched margin-DEF")
    axes[1].set_ylim(-0.025, 0.025)
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.5)

    save_figure(fig, "fig7_capability_spectrum")


def plot_aoi_manipulation_check() -> None:
    data = load_json(AUDIT_DIR / "artifact_controlled_ladder.json")
    by_level = data["exposure"]["by_level"]
    levels = ["5", "15", "30", "60"]

    p50 = [by_level[level]["max_aoi_s"]["p50"] for level in levels]
    p90 = [by_level[level]["max_aoi_s"]["p90"] for level in levels]
    n_exposed = [by_level[level]["n_exposed"] for level in levels]
    at_cap = [78.3, 84.9, 81.1, 79.4]

    x = np.arange(len(levels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)

    axes[0].bar(x - width / 2, p50, width, color=COLORS["median"], label="Median")
    axes[0].bar(x + width / 2, p90, width, color=COLORS["p90"], label="90th percentile")
    axes[0].axhline(60, color="#333333", linestyle="--", linewidth=0.8, label="AoI encoding cap")
    axes[0].set_xticks(x, levels)
    axes[0].set_xlabel("Nominal outage setting (s)")
    axes[0].set_ylabel("Realised encoded AoI (s)")
    axes[0].set_ylim(0, 68)
    axes[0].set_title("(a) Realised AoI does not separate")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False)

    bars = axes[1].bar(x, at_cap, width=0.58, color=COLORS["cap"])
    axes[1].set_xticks(x, levels)
    axes[1].set_xlabel("Nominal outage setting (s)")
    axes[1].set_ylabel("Exposed decisions at 60 s cap (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) Most exposed decisions are censored")
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    for bar, value, count in zip(bars, at_cap, n_exposed):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    save_figure(fig, "fig8_aoi_manipulation_check")


def main() -> None:
    set_ieee_style()
    plot_capability_spectrum()
    plot_aoi_manipulation_check()


if __name__ == "__main__":
    main()
