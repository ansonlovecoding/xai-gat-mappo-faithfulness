"""Create a minimal, plain-language main-finding figure for Chapter 1."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

INK = "#20262E"
MUTED = "#68727E"
BLUE = "#176B87"
GREEN = "#3A7D44"
AMBER = "#B86B00"
RED = "#A93636"


def car(ax, x: float, y: float, color: str, alpha: float = 1.0) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.62, 0.22,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=color, edgecolor=INK, linewidth=1.3, alpha=alpha,
    ))
    ax.add_patch(Polygon(
        [[x + 0.13, y + 0.22], [x + 0.24, y + 0.38],
         [x + 0.47, y + 0.38], [x + 0.55, y + 0.22]],
        closed=True, facecolor=color, edgecolor=INK, linewidth=1.3, alpha=alpha,
    ))
    for cx in (x + 0.14, x + 0.49):
        ax.add_patch(Circle((cx, y - 0.01), 0.064, facecolor=INK,
                            edgecolor="white", linewidth=0.8, alpha=alpha))


def tunnel(ax, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x, y), 0.84, 0.54, facecolor="#737A82",
                           edgecolor=INK, linewidth=1.2))
    ax.add_patch(Arc((x + 0.42, y + 0.16), 0.58, 0.60, theta1=0, theta2=180,
                     color="#E0E3E6", linewidth=11))
    ax.add_patch(Rectangle((x + 0.13, y), 0.58, 0.17,
                           facecolor="#E0E3E6", edgecolor="none"))


def graph(ax, cx: float, cy: float) -> None:
    nodes = [(cx, cy + 0.52), (cx - 0.54, cy), (cx + 0.54, cy), (cx, cy - 0.52)]
    for index, (x1, y1) in enumerate(nodes):
        for x2, y2 in nodes[index + 1:]:
            ax.plot([x1, x2], [y1, y2], color="#AAB1B8", linewidth=2.0, zorder=1)
    ax.add_patch(Circle(nodes[1], 0.24, facecolor="#FFF4DE",
                        edgecolor=AMBER, linewidth=7, alpha=0.9, zorder=2))
    for (x, y), color in zip(nodes, [BLUE, AMBER, GREEN, BLUE]):
        ax.add_patch(Circle((x, y), 0.12, facecolor=color,
                            edgecolor="white", linewidth=1.8, zorder=3))


def flow_arrow(ax, x1: float, x2: float) -> None:
    ax.annotate("", xy=(x2, 2.75), xytext=(x1, 2.75),
                arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": "#8B949D"})


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5.75, 4.55, "Stale data, explanation withheld",
            ha="center", fontsize=21, fontweight="bold", color=INK)

    # Stage labels provide orientation without explanatory prose.
    for x, label, color in ((1.95, "DATA", BLUE),
                            (5.75, "ATTENTION", AMBER),
                            (9.55, "RELEASE", RED)):
        ax.text(x, 4.03, label, ha="center", fontsize=10.5,
                fontweight="bold", color=color)

    ax.plot([0.55, 3.35], [2.58, 2.58], color="#555B61", linewidth=3.2)
    ax.plot([0.55, 3.35], [2.58, 2.58], color="white", linewidth=1.0,
            linestyle=(0, (8, 8)))
    car(ax, 0.72, 2.66, AMBER, alpha=0.32)
    tunnel(ax, 1.54, 2.58)
    car(ax, 2.58, 2.66, GREEN)
    ax.annotate("", xy=(2.50, 3.44), xytext=(1.32, 3.44),
                arrowprops={"arrowstyle": "-|>", "lw": 2.0,
                            "linestyle": "--", "color": MUTED})
    ax.text(1.03, 2.15, "STALE\nOBSERVATION", ha="center", va="top",
            fontsize=10.2, color=AMBER, fontweight="bold", linespacing=1.15)
    ax.text(2.89, 2.15, "ACTUAL\nPOSITION", ha="center", va="top",
            fontsize=10.2, color=GREEN, fontweight="bold", linespacing=1.15)
    ax.text(1.95, 1.35, r"$\ne$", ha="center", fontsize=26,
            color=INK, fontweight="bold")

    flow_arrow(ax, 3.48, 4.08)

    graph(ax, 5.75, 2.89)
    ax.text(5.75, 1.83, "STILL AVAILABLE", ha="center", fontsize=12.5,
            color=AMBER, fontweight="bold")
    ax.text(5.75, 1.31, "RELIABLE EVIDENCE?", ha="center", fontsize=10.5,
            color=MUTED, fontweight="bold")
    ax.text(5.75, 0.93, "UNCERTAIN", ha="center", fontsize=13.0,
            color=RED, fontweight="bold")

    flow_arrow(ax, 7.42, 8.02)

    # Stop sign communicates the release decision before the reader reaches the label.
    stop = Circle((9.55, 2.93), 0.72, facecolor=RED, edgecolor="#852929", linewidth=2.0)
    ax.add_patch(stop)
    ax.add_patch(Rectangle((9.04, 2.78), 1.02, 0.30,
                           facecolor="white", edgecolor="white"))
    ax.text(9.55, 1.83, "WITHHOLD", ha="center", fontsize=16,
            color=RED, fontweight="bold")
    ax.text(9.55, 1.31, "EXPLANATION", ha="center", fontsize=11.2,
            color=INK, fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "v9_main_findings_summary"
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
