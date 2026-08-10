"""Methods & discussion figures for the dissertation (English, print-ready).

Generates six schematic/results figures that the results-story figures
(plot_story.py) do not cover:

  fig1_architecture.png   — system pipeline: SUMO → freeze layer → graph
                            obs → GAT → two explanation channels →
                            faithfulness pipeline
  fig2_observation_graph  — self-centric heterogeneous graph + the
                            node↔action mapping (needed to understand the
                            occlusion artifact)
  fig3_freeze_timeline    — freeze semantics: true vs observed position,
                            AoI growth, outage extension, recovery
  fig4_def_protocol       — DEF occlusion protocol, the occlusion=
                            action-deletion artifact, and the
                            type-matched control
  fig5_causal_chain       — the proposal's causal chain with per-link
                            audited verdicts
  fig6_capability_spectrum— uniform-baseline swing vs type-matched
                            collapse-to-zero across checkpoints

Usage:
  python scripts/plot_paper_figures.py [--out-dir runs/figs/paper]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Palette (matches plot_story.py / the validated reference palette).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e8e7e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.grid": False,
})


def _box(ax, xy, w, h, label, fc="white", ec=INK_2, fontsize=9, lw=1.2,
         text_color=INK):
    ax.add_patch(FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color)


def _arrow(ax, p0, p1, color=INK_2, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=12, linewidth=lw,
        color=color, linestyle=ls, shrinkA=2, shrinkB=2))


# ------------------------------------------------------------------ figure 1

def fig1_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, (0.2, 3.4), 1.7, 1.1,
         "SUMO simulator\n(ground-truth fleet,\ntunnels from OSM)", fc="#eef3fb")
    _box(ax, (2.5, 3.4), 2.0, 1.1,
         "Degradation layer\nfreeze @ observation\nboundary (max-AoI ladder)",
         fc="#fdf3e3")
    _box(ax, (5.1, 3.4), 1.9, 1.1,
         "Per-agent graph obs\nself + 5 taxis + 5\nreservations", fc="#eef3fb")
    _box(ax, (7.6, 3.4), 1.9, 1.1, "GAT encoder × 2\n(4 heads, shared\nparameters)",
         fc="#eef3fb")

    _arrow(ax, (1.9, 3.95), (2.5, 3.95))
    _arrow(ax, (4.5, 3.95), (5.1, 3.95))
    _arrow(ax, (7.0, 3.95), (7.6, 3.95))

    # clean twin side channel
    _arrow(ax, (3.5, 3.4), (3.5, 2.6), color=AQUA, ls=(0, (3, 2)))
    _box(ax, (2.6, 1.9), 1.8, 0.7, "clean twin\n(exact counterfactual)",
         fc="white", ec=AQUA, fontsize=8, text_color=AQUA)

    # heads
    _box(ax, (6.1, 1.9), 1.6, 0.8, "Actor head\nDiscrete(K+1)", fc="#eef3fb")
    _box(ax, (8.0, 1.9), 1.8, 0.8, "Attention weights\n(coupled explanation)",
         fc="#e7f6f0")
    _box(ax, (8.0, 0.7), 1.8, 0.8,
         "Distilled head\n(decoupled explanation)", fc="#e7f6f0")
    _arrow(ax, (8.3, 3.4), (6.9, 2.7))
    _arrow(ax, (8.9, 3.4), (8.9, 2.7))
    _arrow(ax, (8.9, 1.9), (8.9, 1.5), color=INK_2, ls=(0, (3, 2)))
    ax.text(8.78, 1.70, "reads detached\nembeddings", fontsize=7, color=INK_2,
            ha="right", va="center")

    # faithfulness pipeline
    _box(ax, (0.4, 0.5), 5.2, 1.1,
         "Faithfulness pipeline — DEF (occlusion, type-matched baseline), "
         "WAMSN, drift\n~37 counterfactual forwards per scored decision",
         fc="#fdeeee", ec=RED, fontsize=8.5)
    _arrow(ax, (6.9, 2.3), (5.6, 1.4), color=RED)
    _arrow(ax, (8.0, 2.3), (5.6, 1.15), color=RED)
    _arrow(ax, (8.0, 1.1), (5.6, 0.95), color=RED)
    _arrow(ax, (3.5, 1.9), (3.2, 1.6), color=AQUA, ls=(0, (3, 2)))

    ax.set_title("System architecture: control path (top) and explanation-audit path (bottom)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_architecture.png", dpi=220)
    plt.close(fig)
    print("fig1_architecture saved")


# ------------------------------------------------------------------ figure 2

def fig2_observation_graph(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.set_xlim(-3.1, 2.7)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    ax.set_aspect("equal")

    center = (0.0, 0.0)
    taxi_angles = np.linspace(120, 240, 5)
    res_angles = np.linspace(-60, 60, 5)

    for i, ang in enumerate(taxi_angles):
        p = (1.05 * np.cos(np.radians(ang)), 1.05 * np.sin(np.radians(ang)))
        _arrow(ax, center, p, color=GRID, lw=1.0, style="-")
        ax.add_patch(Circle(p, 0.13, facecolor=AQUA, edgecolor=SURFACE, lw=1.5, zorder=3))
        ax.text(*p, f"T{i+1}", ha="center", va="center", fontsize=8,
                color="white", zorder=4)
    for i, ang in enumerate(res_angles):
        p = (1.05 * np.cos(np.radians(ang)), 1.05 * np.sin(np.radians(ang)))
        _arrow(ax, center, p, color=GRID, lw=1.0, style="-")
        ax.add_patch(Circle(p, 0.13, facecolor=YELLOW, edgecolor=SURFACE, lw=1.5, zorder=3))
        ax.text(*p, f"R{i+1}", ha="center", va="center", fontsize=8,
                color="white", zorder=4)
        ax.annotate(f"action {i+1}", xy=p, xytext=(20, 0),
                    textcoords="offset points", fontsize=8, color=INK_2,
                    va="center")

    ax.add_patch(Circle(center, 0.17, facecolor=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3))
    ax.text(0, 0, "self", ha="center", va="center", fontsize=9, color="white", zorder=4)
    ax.annotate("action 0 (no-op)", xy=center, xytext=(0, -26),
                textcoords="offset points", fontsize=8, color=INK_2, ha="center")

    ax.text(-3.0, 1.38,
            "self features: [x, y, t, v, AoI]\n"
            "taxi features: [Δx, Δy, empty, dist, AoI]\n"
            "reservation: [Δpickup, Δdropoff, wait]",
            fontsize=8, color=INK_2, va="top")
    ax.text(-3.0, -0.55,
            "Masked-softmax attention over\nall valid nodes.\n\n"
            "Every reservation node IS an\naction candidate: occluding\n"
            "node R$_k$ deletes action k —\nthe root of the occlusion\n"
            "artifact (Fig. 4).",
            fontsize=8, color=INK, va="top")

    ax.set_title("Per-agent heterogeneous observation graph (K$_n$ = K$_r$ = 5)\n"
                 "and its one-to-one node↔action mapping", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_observation_graph.png", dpi=220)
    plt.close(fig)
    print("fig2_observation_graph saved")


# ------------------------------------------------------------------ figure 3

def fig3_freeze_timeline(out: Path) -> None:
    t = np.linspace(0, 80, 400)
    true_x = 10 + 12 * t / 8
    tunnel_in, tunnel_out, level = 20, 35, 30
    recover = tunnel_in + level  # AoI reaches the level → signal returns

    obs_x = true_x.copy()
    frozen_val = float(true_x[np.searchsorted(t, tunnel_in)])
    frozen = (t >= tunnel_in) & (t < recover)
    obs_x[frozen] = frozen_val

    aoi = np.zeros_like(t)
    aoi[frozen] = t[frozen] - tunnel_in

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 4.6), sharex=True,
                                   height_ratios=[1.4, 1.0])
    for ax in (ax1, ax2):
        ax.axvspan(tunnel_in, tunnel_out, color="#f0efec", zorder=0)
        ax.axvline(recover, color=INK_2, lw=1.0, ls=(0, (3, 2)))
    ax1.plot(t, true_x, color=INK_2, lw=1.4, ls=(0, (4, 3)), label="true position")
    ax1.plot(t, obs_x, color=BLUE, lw=2.0, label="observed position (all observers)")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_ylabel("position (norm.)")
    ax1.text((tunnel_in + tunnel_out) / 2, ax1.get_ylim()[1] * 0.92, "tunnel",
             ha="center", fontsize=8, color=INK_2)
    ax1.annotate("frozen at last valid reading", xy=(35, frozen_val),
                 xytext=(38, frozen_val - 24), fontsize=8, color=BLUE,
                 arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.0})
    ax1.annotate("signal re-acquired:\nsnap to truth, AoI→0",
                 xy=(recover, true_x[np.searchsorted(t, recover)]),
                 xytext=(56, 38), fontsize=8, color=INK,
                 arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 1.0})

    ax2.plot(t, aoi, color=RED, lw=2.0)
    ax2.axhline(level, color=RED, lw=1.0, ls=(0, (2, 2)))
    ax2.text(2, level + 1.5, f"severity level = max AoI = {level}s",
             fontsize=8, color=RED)
    ax2.annotate("outage persists after exit\nuntil AoI reaches the level",
                 xy=((tunnel_out + recover) / 2, 22), xytext=(46, 8),
                 fontsize=8, color=INK_2,
                 arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 1.0})
    ax2.set_ylabel("AoI (s)")
    ax2.set_xlabel("simulation time (s)")

    fig.suptitle("Freeze-based degradation: the severity level bounds the maximum AoI",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_freeze_timeline.png", dpi=220)
    plt.close(fig)
    print("fig3_freeze_timeline saved")


# ------------------------------------------------------------------ figure 4

def fig4_def_protocol(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))
    acts = ["no-op", "R1", "R2", "R3", "R4"]
    base = np.array([1.2, 3.4, 2.1, 0.8, 1.6])

    # (a) protocol
    ax = axes[0]
    ax.bar(acts, base, color=[INK_2, BLUE, INK_2, INK_2, INK_2], width=0.62)
    ax.set_ylim(0, 5.2)
    ax.set_title("(a) DEF protocol", fontsize=10, pad=10)
    ax.set_ylabel("action logits")
    ax.annotate("chosen a*\n(margin m)", xy=(1, base[1]), xytext=(16, 14),
                textcoords="offset points", ha="left", fontsize=8, color=BLUE,
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.9})
    ax.text(0.5, -0.30,
            "occlude explanation's top-k nodes →\nre-measure margin; compare "
            "with a\nsize-matched random occlusion",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK_2)

    # (b) artifact
    ax = axes[1]
    colors = [INK_2, "#d9d8d4", INK_2, INK_2, INK_2]
    ax.bar(acts, base, color=colors, width=0.62)
    ax.bar([1], [base[1]], color="none", edgecolor=RED, hatch="///", width=0.62)
    ax.set_ylim(0, 5.2)
    ax.set_title("(b) the artifact", fontsize=10, pad=10)
    ax.text(2.9, 4.55, "occluding node R1\nDELETES action R1\n→ margin clamps to −cap",
            ha="center", va="top", fontsize=8, color=RED)
    ax.annotate("", xy=(1.35, base[1] * 0.9), xytext=(1.9, 4.15),
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.9})
    ax.text(0.5, -0.30,
            "uniform random draws hit reservation\nnodes far more often than "
            "attention's\ntop-k (clamp rates 14.6% vs 1.8%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK_2)

    # (c) controls
    ax = axes[2]
    vals = [-0.554, -0.009]
    bars = ax.bar(["uniform", "type-matched"], vals,
                  color=[BLUE, AQUA], width=0.55)
    ax.set_xlabel("random-baseline scheme", fontsize=9)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_ylim(-0.62, 0.10)
    ax.annotate(f"{vals[0]:+.3f}", xy=(0, vals[0] / 2), ha="center",
                fontsize=8.5, color="white")
    ax.annotate(f"{vals[1]:+.3f}", xy=(1, vals[1]), xytext=(0, -12),
                textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_title("(c) the control", fontsize=10, pad=10)
    ax.set_ylabel("clean margin-DEF")
    ax.text(0.5, -0.38,
            "matching the random baseline's node-type\ncomposition removes "
            "98% of the deficit:\nattention is uninformative, not anti-informative",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK_2)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The occlusion = action-deletion artifact and its control",
                 fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.30, wspace=0.32)
    fig.savefig(out / "fig4_def_protocol.png", dpi=220)
    plt.close(fig)
    print("fig4_def_protocol saved")


# ------------------------------------------------------------------ figure 5

def fig5_causal_chain(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("Telemetry\ndegradation", "by construction", AQUA, "✓"),
        ("AoI ↑", "measured\n(freeze layer)", AQUA, "✓"),
        ("Attention\ndrift", "JS ≈ 0.000–0.001\nweaker than expected", INK_2, "≈0"),
        ("WAMSN ↑", "H3: ρ=+0.09\np=1e-4 (robust)", AQUA, "✓"),
        ("DEF ↓", "H1 null — channel\nuninformative from start", RED, "✗"),
        ("Faithfulness\ndecoupling", "H2 cell-level\nHolm p=0.039", YELLOW, "△"),
    ]
    w, h, y = 1.45, 0.95, 1.55
    x = 0.15
    for i, (label, note, color, mark) in enumerate(steps):
        _box(ax, (x, y), w, h, label, fc="white", ec=color, lw=1.6, fontsize=9)
        ax.text(x + w / 2, y - 0.18, note, ha="center", va="top",
                fontsize=7.5, color=INK_2)
        ax.text(x + w - 0.16, y + h - 0.16, mark, fontsize=11, color=color,
                ha="center", va="center", weight="bold")
        if i < len(steps) - 1:
            _arrow(ax, (x + w, y + h / 2), (x + w + 0.27, y + h / 2))
        x += w + 0.27

    ax.text(0.15, 0.42,
            "Audited verdicts:  ✓ supported (cluster-robust)    ≈0 present but weak    "
            "△ supported in the rate framing (H2)",
            fontsize=8, color=INK_2)
    ax.text(0.15, 0.12,
            "✗ not supported — floor effect: clean-data DEF is already ≈ 0 "
            "under the type-matched baseline (Fig. 4c), so there is nothing "
            "left to decline",
            fontsize=8, color=INK_2)
    ax.set_title("The proposed causal chain (proposal Fig. 1) with per-link audited verdicts",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_causal_chain.png", dpi=220)
    plt.close(fig)
    print("fig5_causal_chain saved")


# ------------------------------------------------------------------ figure 6

def fig6_capability_spectrum(out: Path, spectrum_json: Path) -> None:
    data = json.loads(spectrum_json.read_text())
    rows = [r for r in data["rows"] if "note" not in r]
    labels, u_vals, t_vals, pickups = [], [], [], []
    name_map = {
        "B2_gat": "B2", "B3_gat_noaoi": "B3", "H5b_freeze": "H5′ s42",
        "H5c_seed43": "H5′ s43", "H5d_seed44": "H5′ s44",
    }
    for r in rows:
        p = Path(r["checkpoint"])
        run = name_map.get(p.parent.parent.name, p.parent.parent.name)
        tag = f"{run} ep{r['epoch']}"
        labels.append(f"{tag}\n({r['mean_pickups']:.0f} pickups)")
        u_vals.append(r["def_m_uniform"]["mean"])
        t_vals.append(r["def_m_type_matched"]["mean"])
        pickups.append(r["mean_pickups"])

    order = np.argsort(pickups)
    labels = [labels[i] for i in order]
    u_vals = np.array(u_vals)[order]
    t_vals = np.array(t_vals)[order]
    ys = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for y, u, tv in zip(ys, u_vals, t_vals):
        ax.plot([u, tv], [y, y], color=GRID, lw=1.6, zorder=1)
    ax.scatter(u_vals, ys, s=52, color=BLUE, zorder=3, label="uniform baseline")
    ax.scatter(t_vals, ys, s=52, color=AQUA, zorder=4, label="type-matched baseline")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("clean margin-DEF")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Across the capability spectrum the uniform-baseline metric swings\n"
                 "−1.35…+1.78 while the type-matched metric sits at ≈ 0 everywhere",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig6_capability_spectrum.png", dpi=220)
    plt.close(fig)
    print("fig6_capability_spectrum saved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "runs" / "figs" / "paper")
    parser.add_argument("--spectrum-json", type=Path,
                        default=PROJECT_ROOT / "runs" / "capability_spectrum_clean.json")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig1_architecture(args.out_dir)
    fig2_observation_graph(args.out_dir)
    fig3_freeze_timeline(args.out_dir)
    fig4_def_protocol(args.out_dir)
    fig5_causal_chain(args.out_dir)
    if args.spectrum_json.exists():
        fig6_capability_spectrum(args.out_dir, args.spectrum_json)
    else:
        print("fig6: spectrum JSON missing — skipped")
    print(f"figures: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
