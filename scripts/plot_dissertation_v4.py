"""Create thesis figures from the validated dissertation_v4 summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


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
SEED_STYLES = {
    42: {"marker": "o", "linestyle": "-"},
    43: {"marker": "s", "linestyle": "--"},
    44: {"marker": "^", "linestyle": ":"},
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


def checkpoint_selection_figure() -> None:
    models = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        for seed in SEED_STYLES:
            path = RUNS / "training" / model / f"seed_{seed}" / "checkpoint_selection.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates = [item for item in payload["candidates"]
                          if item["checkpoint"].startswith("ckpt_epoch_")]
            candidates.sort(key=lambda item: item["epoch"])
            style = SEED_STYLES[seed]
            ax.plot([item["epoch"] for item in candidates],
                    [item["mean_pickups"] for item in candidates],
                    marker=style["marker"], linestyle=style["linestyle"],
                    markersize=3.5, linewidth=1.3, label=f"seed {seed}")
            selected = payload["selected"]
            ax.scatter(selected["epoch"], selected["mean_pickups"],
                       marker="*", s=125, facecolor="#C75000",
                       edgecolor="#202020", linewidth=0.7, zorder=5)
        ax.set_title(LABELS[model])
        ax.grid(color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean validation pickups")
    axes[1, 0].set_ylabel("Mean validation pickups")
    axes[1, 0].set_xlabel("Checkpoint epoch")
    axes[1, 1].set_xlabel("Checkpoint epoch")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle("Validation-only checkpoint selection across training seeds\n"
                 "Stars mark the frozen checkpoints used for held-out evaluation")
    fig.tight_layout()
    _save(fig, "v4_checkpoint_selection_by_model_and_seed")


def decoupling_figure() -> None:
    rows = _csv(RUNS / "summary.csv")
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.8), sharex="col")
    def_values = [float(row["mean_type_matched_def"]) for row in rows
                  if row["model"] in COLORS]
    def_limit = max(abs(value) for value in def_values) * 1.15
    for column, model in enumerate(COLORS):
        selected = [row for row in rows
                    if row["model"] == model and row["axis"] == "outage_duration"]
        for seed in sorted({int(row["training_seed"]) for row in selected}):
            seed_rows = sorted(
                (row for row in selected if int(row["training_seed"]) == seed),
                key=lambda row: float(row["level_s"]),
            )
            clean = next(row for row in rows if row["model"] == model
                         and int(row["training_seed"]) == seed
                         and row["axis"] == "clean")
            axes[0, column].plot(
                [float(row["level_s"]) for row in seed_rows],
                [float(row["mean_wamsn_when_exposed_secondary"]) for row in seed_rows],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                linewidth=1.7, label=f"training seed {seed}",
            )
            axes[1, column].plot(
                [0.0, *[float(row["level_s"]) for row in seed_rows]],
                [float(clean["mean_type_matched_def"]),
                 *[float(row["mean_type_matched_def"]) for row in seed_rows]],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                linewidth=1.7,
            )
            axes[2, column].plot(
                [0.0, *[float(row["level_s"]) for row in seed_rows]],
                [float(clean["mean_pickups"]),
                 *[float(row["mean_pickups"]) for row in seed_rows]],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                linewidth=1.7,
            )
        axes[0, column].set_title(LABELS[model])
        axes[2, column].set_xlabel("Observation-layer outage duration (s)")
        axes[1, column].set_ylim(-def_limit, def_limit)
        axes[1, column].axhline(0, color="#202020", linewidth=0.9)
        for ax in axes[:, column]:
            ax.grid(color="#D8D8D8", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Conditional WAMSN")
    axes[1, 0].set_ylabel("Type-matched DEF")
    axes[2, 0].set_ylabel("Mean pickups per episode")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Stale exposure rises while measured faithfulness and pickups remain nearly flat")
    fig.tight_layout()
    _save(fig, "v4_decoupling_by_outage_duration")


def evidence_summary_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("Outage\nwindow", "fixed observation-\nlayer windows", "#176B87"),
        ("Stale\nexposure", "H3 support:\n6/6 policies", "#3A7D44"),
        ("Attention\nshift", "positive in 4/6;\nnegative in 2/6", "#B06C00"),
        ("Faithfulness\neffect", "H1 and H2\nsupport: 0/6", "#A33A3A"),
        ("Training\nmitigation", "no consistent\nH5 mitigation", "#555555"),
    ]
    width, gap, y, height = 0.165, 0.035, 0.25, 0.50
    for index, (title, body, color) in enumerate(items):
        x = 0.015 + index * (width + gap)
        patch = FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.012,rounding_size=0.015",
                               facecolor="#F5F7F8", edgecolor=color,
                               linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.36, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, linespacing=1.05)
        ax.text(x + width / 2, y + 0.15, body, ha="center", va="center",
                fontsize=8.0, color="#202020")
        if index < len(items) - 1:
            ax.annotate("", xy=(x + width + gap * 0.82, y + height / 2),
                        xytext=(x + width + gap * 0.18, y + height / 2),
                        arrowprops={"arrowstyle": "-|>", "color": "#555555",
                                    "linewidth": 1.2})
    ax.text(0.5, 0.91, "Evidence path and final cross-seed verdicts",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.09,
            "Exposure is reproducible; attention reallocation and its relationship with faithfulness are policy-dependent.",
            ha="center", va="center", fontsize=8.8, fontweight="bold")
    fig.tight_layout()
    _save(fig, "v4_evidence_path_summary")


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
            analysis = json.loads((
                RUNS / "sweeps" / model / f"seed_{row['training_seed']}" / "analysis.json"
            ).read_text())
            low, high = analysis["robust"]["primary_stale_attention_shift"]["ci95"]
            value = row["stale_attention_shift"]
            ax.errorbar(index + offset, value,
                        yerr=[[value - low], [high - value]], fmt="o", markersize=7,
                        capsize=4, color=COLORS[model], zorder=3)
            ax.annotate(str(row["training_seed"]),
                        (index + offset, value),
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
    payload = json.loads((RUNS / "training_seed_synthesis.json").read_text())
    rows = payload["per_seed"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for index, model in enumerate(COLORS):
        selected = sorted((row for row in rows if row["model"] == model),
                          key=lambda row: row["training_seed"])
        offsets = np.linspace(-0.10, 0.10, len(selected))
        for offset, row in zip(offsets, selected):
            supported = bool(row["H4_supported"])
            ax.scatter(index + offset, row["H4_within_episode_rho"], s=70,
                       facecolor=COLORS[model] if supported else "white",
                       edgecolor=COLORS[model], linewidth=1.8, zorder=3)
            ax.annotate(str(row["training_seed"]),
                        (index + offset, row["H4_within_episode_rho"]),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(range(len(COLORS)), [LABELS[model] for model in COLORS])
    ax.set_ylabel("Within-episode Spearman rho (WAMSN vs DEF)")
    ax.set_title("The WAMSN-faithfulness relationship depends on training regime and seed")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "v4_h4_correlation_by_training_seed")


def main() -> None:
    performance_figure()
    checkpoint_selection_figure()
    decoupling_figure()
    shift_figure()
    consistency_figure()
    evidence_summary_figure()
    print(f"figures: {OUT}")


if __name__ == "__main__":
    main()
