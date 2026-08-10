"""Plot the structured-vs-random degradation ablation.

Reads a `<ckpt>.ablation.json` produced by
`scripts/eval_degradation_ablation.py` and renders a three-panel bar chart:

  Panel 1: mean pickups per condition (Δpolicy performance)
  Panel 2: DEF mean ± std per condition (Δfaithfulness)
  Panel 3: WAMSN mean ± std per condition (Δattention-on-stale-nodes)

Condition colors are fixed so multiple runs plotted side-by-side stay
consistent: off=grey (baseline), tunnel_triggered=orange (structured),
random_dropout=blue (matched-rate random).

The dissertation's headline number is `tunnel − random_dropout` at
matched rate — that's the "does structure matter, controlling for average
degradation" test. Both bars are on every panel so the delta is visual.

Usage:
  python scripts/plot_ablation.py <ckpt>.ablation.json
  python scripts/plot_ablation.py <path>.ablation.json --out figs/ablation.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Fixed order and colors — see docstring.
CONDITION_ORDER = ["off", "tunnel_triggered", "random_dropout"]
CONDITION_COLORS = {
    "off": "#7f7f7f",
    "tunnel_triggered": "#ff7f0e",
    "random_dropout": "#1f77b4",
}
CONDITION_LABELS = {
    "off": "off\n(clean)",
    "tunnel_triggered": "tunnel_triggered\n(structured)",
    "random_dropout": "random_dropout\n(matched rate)",
}


def _bar_with_err(ax, values, errs, title: str, ylabel: str, fmt: str = ".2f",
                  err_label: str = "± 1σ") -> None:
    """Draw a single-panel bar chart across the fixed condition order."""
    xs = np.arange(len(CONDITION_ORDER))
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    bars = ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6)
    if any(e > 0 for e in errs):
        ax.errorbar(xs, values, yerr=errs, fmt="none", ecolor="black",
                    capsize=4, lw=1.0, label=err_label)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels above each bar for easy dissertation-caption lookups.
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f" {v:{fmt}}", ha="center", va="bottom", fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ablation_json", type=Path,
                        help=".ablation.json produced by eval_degradation_ablation.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path; default: <input>.png alongside the JSON")
    parser.add_argument("--title", default=None,
                        help="figure super-title; default is derived from checkpoint name")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    if not args.ablation_json.exists():
        parser.error(f"ablation JSON not found: {args.ablation_json}")

    data = json.loads(args.ablation_json.read_text())
    conditions = data["conditions"]

    for c in CONDITION_ORDER:
        if c not in conditions:
            parser.error(f"missing condition '{c}' in ablation JSON")

    # ---- pickup means / stds ----
    pickup_means = [conditions[c]["mean_pickups"] for c in CONDITION_ORDER]
    pickup_stds = [conditions[c].get("std_pickups", 0.0) for c in CONDITION_ORDER]

    # ---- DEF means / stds (may be missing if no scorable decisions) ----
    def _faith_field(cond: str, key: str, default: float = np.nan) -> float:
        f = conditions[cond].get("faithfulness", {})
        v = f.get(key, default)
        return float(v) if v is not None else default

    def_means = [_faith_field(c, "def_mean") for c in CONDITION_ORDER]
    def_stds = [_faith_field(c, "def_std", 0.0) for c in CONDITION_ORDER]
    wamsn_means = [_faith_field(c, "wamsn_mean") for c in CONDITION_ORDER]
    wamsn_stds = [_faith_field(c, "wamsn_std", 0.0) for c in CONDITION_ORDER]

    fig, (ax_pick, ax_def, ax_wamsn) = plt.subplots(1, 3, figsize=(13, 4.4))

    _bar_with_err(ax_pick, pickup_means, pickup_stds,
                  title="Pickups per episode", ylabel="pickups",
                  err_label="± σ across episodes")
    _bar_with_err(ax_def, def_means, def_stds,
                  title="DEF (dispatch explanation faithfulness)",
                  ylabel="DEF ∈ [−1, 1]", fmt="+.3f",
                  err_label="± σ across decisions")
    # Zero-line reference on DEF panel — DEF = 0 means "no better than random".
    ax_def.axhline(0.0, color="black", lw=0.6, ls=":")
    _bar_with_err(ax_wamsn, wamsn_means, wamsn_stds,
                  title="WAMSN (attention mass on stale nodes)",
                  ylabel="WAMSN ∈ [0, 1]", fmt=".3f",
                  err_label="± σ across decisions")

    # Add a caption with the matched rate + observed rates so the plot is
    # self-contained for the dissertation.
    ep = data.get("episodes_per_condition", "?")
    matched_rate = data.get("matched_dropout_rate")
    tunnel_rate = conditions["tunnel_triggered"].get("empirical_degradation_rate")
    random_rate = conditions["random_dropout"].get("empirical_degradation_rate")
    caption_bits = [f"{ep} episode(s) per condition"]
    if matched_rate is not None:
        caption_bits.append(f"matched rate = {matched_rate:.4f}")
    if tunnel_rate is not None and random_rate is not None:
        caption_bits.append(
            f"observed: tunnel {tunnel_rate:.4f}, random {random_rate:.4f}"
        )
    caption = "  |  ".join(caption_bits)

    supt = args.title or f"Structured vs random telemetry degradation — {Path(data['checkpoint']).name}"
    fig.suptitle(supt, fontsize=12, y=1.02)
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="grey")
    fig.tight_layout()

    out_path = args.out or args.ablation_json.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
