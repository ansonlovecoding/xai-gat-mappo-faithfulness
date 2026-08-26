"""Create thesis figures from the validated dissertation_v4 summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "dissertation_v4"
OUT = ROOT / "docs" / "figures"
COLORS = {"B2_gat": "#176B87", "H5_gat_degraded": "#C75000"}
LABELS = {
    "B1_mlp": "B1 MLP",
    "B2_gat": "B2 GAT",
    "B3_gat_noaoi": "B3 GAT, no AoI",
    "H5_gat_degraded": "H5 degraded training",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def performance_figure() -> None:
    rows = _csv(RUNS / "performance_context.csv")
    models = list(LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for index, model in enumerate(models):
        values = [float(row["mean_pickups"]) for row in rows if row["model"] == model]
        offsets = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(index + offsets, values, s=42, color="#176B87", zorder=3)
        ax.hlines(np.mean(values), index - 0.22, index + 0.22,
                  color="#202020", linewidth=2)
    ax.set_xticks(range(len(models)), [LABELS[model] for model in models])
    ax.set_ylabel("Mean pickups per test episode")
    ax.set_title("Clean-test performance is context, not the study outcome")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "v4_clean_performance_by_training_seed")


def wamsn_figure() -> None:
    rows = _csv(RUNS / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    for ax, model in zip(axes, COLORS):
        selected = [row for row in rows
                    if row["model"] == model and row["axis"] == "outage_duration"]
        for seed in sorted({int(row["training_seed"]) for row in selected}):
            seed_rows = sorted(
                (row for row in selected if int(row["training_seed"]) == seed),
                key=lambda row: float(row["level_s"]),
            )
            ax.plot(
                [float(row["level_s"]) for row in seed_rows],
                [float(row["mean_wamsn_when_exposed_secondary"]) for row in seed_rows],
                marker="o", linewidth=1.7, label=f"training seed {seed}",
            )
        ax.set_title(LABELS[model])
        ax.set_xlabel("Observation-layer outage duration (s)")
        ax.grid(color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean WAMSN when a stale node is visible")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Stale-attention exposure rises with longer outages")
    fig.tight_layout()
    _save(fig, "v4_wamsn_by_outage_duration")


def shift_figure() -> None:
    payload = json.loads((RUNS / "training_seed_synthesis.json").read_text())
    rows = payload["per_seed"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for index, model in enumerate(COLORS):
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: row["training_seed"],
        )
        offsets = np.linspace(-0.10, 0.10, len(selected))
        for offset, row in zip(offsets, selected):
            ax.scatter(index + offset, row["stale_attention_shift"], s=55,
                       color=COLORS[model], zorder=3)
            ax.annotate(str(row["training_seed"]),
                        (index + offset, row["stale_attention_shift"]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(range(len(COLORS)), [LABELS[model] for model in COLORS])
    ax.set_ylabel("Degraded minus clean stale-attention mass")
    ax.set_title("Paired attention reallocation depends on training seed")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "v4_paired_stale_attention_shift")


def consistency_figure() -> None:
    rows = _csv(RUNS / "training_seed_synthesis.csv")
    hypotheses = ["H1", "H2", "H3", "H4"]
    x = np.arange(len(hypotheses))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    width = 0.34
    for index, row in enumerate(rows):
        counts = [int(row[f"{name}_supported_seeds"]) for name in hypotheses]
        ax.bar(x + (index - 0.5) * width, counts, width,
               label=LABELS[row["model"]], color=COLORS[row["model"]])
    ax.set_xticks(x, hypotheses)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("Training seeds supporting hypothesis (of 3)")
    ax.set_title("Only H3 is consistent across trained policies")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "v4_hypothesis_consistency")


def main() -> None:
    performance_figure()
    wamsn_figure()
    shift_figure()
    consistency_figure()
    print(f"figures: {OUT}")


if __name__ == "__main__":
    main()
