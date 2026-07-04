"""Plot faithfulness convergence from a training log.

Reads a `train_log.jsonl` produced by `scripts/train.py --faith-every-epochs N`
and renders a two-panel figure:

  Top:    DEF mean ± 1σ (per-decision std) + WAMSN mean on a twin axis.
  Bottom: pickups per episode (thin line) + rolling-mean pickups (bold).

Epochs without a `faithfulness` field are silently skipped in the top panel,
so it works even when the sampler was set to fire only every Nth epoch. Both
panels share the x-axis (epoch).

Usage:
  python scripts/plot_train_faithfulness.py runs/mappo/central_park_<ts>/train_log.jsonl
  python scripts/plot_train_faithfulness.py <log.jsonl> --out figs/convergence.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_faith_series(rows: list[dict]) -> dict[str, np.ndarray]:
    faith_rows = [r for r in rows if r.get("faithfulness")]
    if not faith_rows:
        return {}
    return {
        "epoch": np.array([r["epoch"] for r in faith_rows]),
        "def_mean": np.array([r["faithfulness"].get("def_mean", np.nan) for r in faith_rows]),
        "def_std": np.array([r["faithfulness"].get("def_std", 0.0) for r in faith_rows]),
        "wamsn_mean": np.array([r["faithfulness"].get("wamsn_mean", 0.0) for r in faith_rows]),
        "n_scored": np.array([r["faithfulness"].get("n_scored", 0) for r in faith_rows]),
    }


def _extract_policy_series(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "epoch": np.array([r["epoch"] for r in rows]),
        "pickups": np.array([r["pickups"] for r in rows], dtype=float),
        "rolling": np.array(
            [r.get("rolling_mean_pickups") for r in rows], dtype=float
        ),  # numpy will coerce None → nan
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="train_log.jsonl produced by train.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path; default: <log_dir>/convergence.png")
    parser.add_argument("--title", default=None,
                        help="figure super-title; default: run directory name")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"log not found: {args.log}")

    rows = _load_rows(args.log)
    if not rows:
        parser.error(f"log is empty: {args.log}")

    faith = _extract_faith_series(rows)
    policy = _extract_policy_series(rows)

    fig, (ax_faith, ax_pick) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0.15}
    )

    # -------- Top panel: DEF ± std, WAMSN twin --------
    if faith:
        ax_faith.plot(faith["epoch"], faith["def_mean"], color="#1f77b4",
                      lw=1.8, marker="o", ms=3.5, label="DEF (mean)")
        ax_faith.fill_between(
            faith["epoch"],
            faith["def_mean"] - faith["def_std"],
            faith["def_mean"] + faith["def_std"],
            color="#1f77b4", alpha=0.15, label="DEF ± 1σ per-decision",
        )
        ax_faith.axhline(0.0, color="grey", lw=0.7, ls=":")
        ax_faith.set_ylabel("DEF", color="#1f77b4")
        ax_faith.tick_params(axis="y", labelcolor="#1f77b4")
        ax_faith.legend(loc="upper left", fontsize=8, framealpha=0.9)

        ax_wamsn = ax_faith.twinx()
        ax_wamsn.plot(faith["epoch"], faith["wamsn_mean"], color="#d62728",
                      lw=1.4, marker="s", ms=3, label="WAMSN (mean)")
        ax_wamsn.set_ylabel("WAMSN", color="#d62728")
        ax_wamsn.tick_params(axis="y", labelcolor="#d62728")
        ax_wamsn.legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax_faith.text(0.5, 0.5, "no faithfulness samples in this log\n"
                                "(train with --faith-every-epochs N)",
                      ha="center", va="center", transform=ax_faith.transAxes,
                      color="grey", fontsize=11)
        ax_faith.set_ylabel("DEF / WAMSN")

    ax_faith.grid(True, alpha=0.3)
    ax_faith.set_title("Faithfulness convergence", fontsize=10, loc="left")

    # -------- Bottom panel: pickups + rolling --------
    ax_pick.plot(policy["epoch"], policy["pickups"], color="grey", lw=0.9,
                 alpha=0.6, label="pickups (per episode)")
    # rolling is float array with NaN where the window wasn't full yet.
    if np.isfinite(policy["rolling"]).any():
        ax_pick.plot(policy["epoch"], policy["rolling"], color="black",
                     lw=1.8, label="rolling mean")
    ax_pick.set_ylabel("pickups")
    ax_pick.set_xlabel("epoch")
    ax_pick.grid(True, alpha=0.3)
    ax_pick.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_pick.set_title("Policy performance", fontsize=10, loc="left")

    supt = args.title or args.log.parent.name
    fig.suptitle(supt, fontsize=12, y=0.995)
    fig.tight_layout()

    out_path = args.out or (args.log.parent / "convergence.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"  epochs plotted: {len(rows)}  "
          f"(faithfulness samples: {len(faith.get('epoch', []))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
