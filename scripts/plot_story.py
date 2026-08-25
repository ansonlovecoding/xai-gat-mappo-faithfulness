"""The dissertation's three headline figures (the "three-act story").

Act 1 — Is the built-in explanation faithful at all?
    Distribution of per-decision margin-DEF on clean telemetry, against
    the zero line (= a size-matched random explanation).

Act 2 — Does it degrade before performance as telemetry ages?
    Three stacked panels sharing the outage-duration axis (no dual-axis
    charts): margin-DEF, pickups, WAMSN. Degradation-naive policy (B2).

Act 3 — Can it be fixed?
    (a) The same three panels with the degradation-aware policy (H5')
        overlaid on B2 — the reversal figure.
    (b) Paired bars: coupled attention vs decoupled explainer margin-DEF.

Colors follow the entity across every figure: degradation-naive = blue,
degradation-aware = aqua (fixed categorical slots 1–2; the slot order is
the CVD-safety mechanism). Values/labels wear text colors, never series
colors.

Usage:
  python scripts/plot_story.py --b2-sweep runs/sweeps/B2_aoi_ladder \\
      [--h5-sweep runs/sweeps/H5b_aoi_ladder] \\
      [--compare <explainer_compare.json> --compare-tunnel <...json>] \\
      [--out-dir runs/figs/story]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical slots (validated reference palette, light mode).
BLUE = "#2a78d6"    # slot 1 — degradation-naive (B2) / coupled channel
AQUA = "#1baf7a"    # slot 2 — degradation-aware (H5') / decoupled channel
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3e0"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
})


# ------------------------------------------------------------- data loading


def load_sweep(sweep_dir: Path) -> dict[float, dict]:
    """Aggregate a ladder sweep: level → {def_m list, wamsn list, pickups list}."""
    levels: dict[float, dict] = {}
    cell_dir = sweep_dir / "cells"
    candidates = cell_dir.glob("*.json") if cell_dir.is_dir() else sweep_dir.glob("*.json")
    for p in sorted(candidates):
        cell = json.loads(p.read_text())
        if not (isinstance(cell, dict) and "cell" in cell and "faith_records" in cell):
            continue
        meta = cell.get("cell", {})
        if meta.get("axis") not in ("clean", "max_aoi", "outage_duration"):
            continue  # appendix axes stay out of the story figures
        lvl = float(meta.get("level", 0.0))
        agg = levels.setdefault(lvl, {"def_m": [], "wamsn": [], "pickups": [],
                                      "aoi_emp": []})
        for r in cell.get("faith_records", []):
            if r.get("valid_reservations", 0) > 0 and "def_m" in r:
                agg["def_m"].append(r["def_m"])
            agg["wamsn"].append(r["wamsn"])
        for e in cell.get("per_episode", []):
            agg["pickups"].append(e["total_pickups"])
        agg["aoi_emp"].append(cell.get("empirical_degradation_rate", 0.0))
    return dict(sorted(levels.items()))


def boot_ci(x: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo95, hi95) via bootstrap over decisions."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([
        arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(n_boot)
    ])
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ figures


def act1(b2: dict[float, dict], out: Path,
         audit_json: Path | None = None) -> None:
    """Clean-telemetry margin-DEF under two random-baseline schemes.

    The uniform baseline (standard occlusion protocol) says "worse than
    random"; the type-matched control shows 98 % of that magnitude is the
    occlusion=action-deletion artifact. The honest Act-1 claim is
    "uninformative": the corrected figure shows both, decomposed.
    """
    scores = np.asarray(b2.get(0.0, {}).get("def_m", []), dtype=np.float64)
    if scores.size == 0:
        print("act1: no clean-cell def_m records — skipped")
        return
    mean, lo, hi = boot_ci(list(scores))

    tm = None
    if audit_json is not None and audit_json.exists():
        tm = json.loads(audit_json.read_text())

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), width_ratios=[1.5, 1.0])

    # Left: distribution under the standard protocol (kept for context).
    ax.hist(scores, bins=41, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.axvline(0.0, color=INK, linewidth=1.4)
    ax.axvline(mean, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"mean {mean:+.2f}", xy=(mean, ax.get_ylim()[1] * 0.85),
                xytext=(-6, 0), textcoords="offset points", ha="right",
                fontsize=9, color=INK_2)
    ax.set_xlabel("margin-DEF per decision\n(uniform random baseline — standard protocol)")
    ax.set_ylabel("decisions")

    # Right: the artifact decomposition (uniform vs type-matched baseline).
    if tm is not None:
        u = tm["uniform_baseline"]
        t = tm["type_matched_baseline"]
        xs = [0, 1]
        vals = [u["def_m_mean"], t["def_m_mean"]]
        errs = [
            [vals[0] - u["ci95"][0], vals[1] - t["ci95"][0]],
            [u["ci95"][1] - vals[0], t["ci95"][1] - vals[1]],
        ]
        bars = axb.bar(xs, vals, width=0.62, color=[BLUE, AQUA],
                       edgecolor=SURFACE, linewidth=0.5)
        axb.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK,
                     elinewidth=1.2, capsize=4)
        axb.axhline(0.0, color=INK, linewidth=1.4)
        axb.set_xticks(xs)
        axb.set_xticklabels(["uniform\nbaseline", "type-matched\nbaseline"],
                            fontsize=9)
        for b, v in zip(bars, vals):
            axb.annotate(f"{v:+.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                         xytext=(0, -12 if v < 0 else 4),
                         textcoords="offset points", ha="center", fontsize=9,
                         color=INK)
        axb.set_ylabel("mean margin-DEF (clean)")
        axb.annotate("98% of the deficit is the\nocclusion = action-deletion\nartifact",
                     xy=(0.97, 0.40), xycoords="axes fraction", ha="right",
                     va="center", fontsize=8.5, color=INK_2)
    else:
        axb.axis("off")

    fig.suptitle("Act 1 — attention carries no measurable decision-relevant "
                 "information\n(the 'worse than random' reading is a protocol artifact)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / "story_act1_clean_def.png", dpi=200)
    plt.close(fig)
    print(f"act1: saved (n={scores.size}, uniform mean={mean:+.3f}; "
          f"type-matched {'included' if tm else 'MISSING'})")


def _ladder_panels(ax_def, ax_perf, ax_wamsn, sweep: dict[float, dict],
                   color: str, label: str) -> None:
    levels = sorted(sweep.keys())
    d = [boot_ci(sweep[l]["def_m"], seed=1) for l in levels]
    w = [boot_ci(sweep[l]["wamsn"], seed=2) for l in levels]
    p_mean = [float(np.mean(sweep[l]["pickups"])) for l in levels]
    p_sd = [float(np.std(sweep[l]["pickups"])) for l in levels]

    ax_def.plot(levels, [x[0] for x in d], color=color, linewidth=2,
                marker="o", markersize=6, label=label)
    ax_def.fill_between(levels, [x[1] for x in d], [x[2] for x in d],
                        color=color, alpha=0.15, linewidth=0)
    ax_perf.errorbar(levels, p_mean, yerr=p_sd, color=color, linewidth=2,
                     marker="o", markersize=6, capsize=3, label=label)
    ax_wamsn.plot(levels, [x[0] for x in w], color=color, linewidth=2,
                  marker="o", markersize=6, label=label)
    ax_wamsn.fill_between(levels, [x[1] for x in w], [x[2] for x in w],
                          color=color, alpha=0.15, linewidth=0)


def ladder_figure(sweeps: list[tuple[dict, str, str]], title: str,
                  fname: str, out: Path) -> None:
    """Three stacked panels sharing the severity axis (one y-scale each)."""
    fig, (ax_def, ax_perf, ax_wamsn) = plt.subplots(
        3, 1, figsize=(6.6, 7.2), sharex=True)
    for sweep, color, label in sweeps:
        _ladder_panels(ax_def, ax_perf, ax_wamsn, sweep, color, label)
    ax_def.axhline(0.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax_def.set_ylabel("margin-DEF\n(vs random = 0)")
    ax_perf.set_ylabel("pickups / episode")
    ax_wamsn.set_ylabel("WAMSN")
    ax_wamsn.set_xlabel("severity: maximum AoI (s) — 0 = clean")
    ax_def.set_title(title)
    if len(sweeps) > 1:
        ax_def.legend(frameon=False, loc="best")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200)
    plt.close(fig)
    print(f"saved {fname}")


def act3b(compare: Path | None, compare_tunnel: Path | None, out: Path) -> None:
    """Coupled attention vs decoupled explainer, paired margin-DEF."""
    conds = []
    for path, name in ((compare, "clean"), (compare_tunnel, "tunnel")):
        if path and path.exists():
            d = json.loads(path.read_text())
            m = d.get("def_m", {})
            if m:
                conds.append((name, m["coupled_mean"], m["decoupled_mean"],
                              m.get("p_decoupled_better")))
    if not conds:
        print("act3b: no explainer-compare inputs — skipped")
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = np.arange(len(conds), dtype=float)
    width = 0.34
    ax.bar(xs - width / 2 - 0.01, [c[1] for c in conds], width,
           color=BLUE, label="coupled (attention)")
    ax.bar(xs + width / 2 + 0.01, [c[2] for c in conds], width,
           color=AQUA, label="decoupled (distilled head)")
    ax.axhline(0.0, color=INK, linewidth=1.2)
    for i, (name, c, dcp, p) in enumerate(conds):
        ax.annotate(f"{c:+.2f}", (xs[i] - width / 2 - 0.01, c),
                    ha="center", va="top", fontsize=9, color=INK_2,
                    xytext=(0, -3), textcoords="offset points")
        ax.annotate(f"{dcp:+.2f}", (xs[i] + width / 2 + 0.01, dcp),
                    ha="center", va="top", fontsize=9, color=INK_2,
                    xytext=(0, -3), textcoords="offset points")
        if p is not None:
            ax.annotate(f"p = {p:.4f}", (xs[i], 0.02), ha="center",
                        fontsize=9, color=INK_2)
    ax.set_xticks(xs, [c[0] for c in conds])
    ax.set_ylabel("margin-DEF (vs random = 0)")
    ax.set_title("Act 3b — a decoupled explanation channel is more faithful")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "story_act3b_coupled_vs_decoupled.png", dpi=200)
    plt.close(fig)
    print("saved story_act3b_coupled_vs_decoupled.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b2-sweep", type=Path, required=True,
                        help="ladder sweep dir of the degradation-naive policy")
    parser.add_argument("--h5-sweep", type=Path, default=None,
                        help="ladder sweep dir of the degradation-aware policy")
    parser.add_argument("--compare", type=Path, default=None,
                        help="explainer_compare.json (clean condition)")
    parser.add_argument("--compare-tunnel", type=Path, default=None)
    parser.add_argument("--type-matched-audit", type=Path,
                        default=Path("results/story_freeze_v1/audit/type_matched_control.json"),
                        help="type-matched control JSON for the Act-1 artifact "
                             "decomposition panel")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <b2-sweep>/../figs_story")
    args = parser.parse_args()

    out = args.out_dir or (args.b2_sweep.parent / "figs_story")
    out.mkdir(parents=True, exist_ok=True)

    b2 = load_sweep(args.b2_sweep)
    if not b2:
        parser.error(f"no ladder cells found in {args.b2_sweep}")
    act1(b2, out, audit_json=args.type_matched_audit)
    ladder_figure([(b2, BLUE, "degradation-naive (B2)")],
                  "Act 2 — attention shifts to stale data;\n"
                  "faithfulness gives no warning and performance no signal",
                  "story_act2_decoupling.png", out)
    if args.h5_sweep and args.h5_sweep.exists():
        h5 = load_sweep(args.h5_sweep)
        if h5:
            ladder_figure(
                [(b2, BLUE, "degradation-naive (B2)"),
                 (h5, AQUA, "degradation-aware (H5)")],
                "Act 3a — degradation-aware training does NOT repair\n"
                "the built-in explanation channel",
                "story_act3a_training_mitigation.png", out)
    act3b(args.compare, args.compare_tunnel, out)
    print(f"figures: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
