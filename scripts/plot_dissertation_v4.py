"""Create thesis figures from a validated dissertation experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "dissertation_v4"
TRAINING_RUNS = RUNS
OUT = ROOT / "docs" / "figures"
PREFIX = "v4"
COLORS = {"B2_gat": "#176B87", "H5_gat_degraded": "#C75000"}
LABELS = {
    "B1_mlp": "MLP",
    "B2_gat": "GAT",
    "H5_gat_degraded": "GAT-Outage",
}
SEED_STYLES = {
    42: {"marker": "o", "linestyle": "-", "color": "#176B87"},
    43: {"marker": "s", "linestyle": "--", "color": "#C75000"},
    44: {"marker": "^", "linestyle": ":", "color": "#3A7D44"},
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    filename = f"{PREFIX}_{name}"
    fig.savefig(OUT / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def performance_figure() -> None:
    rows = _csv(RUNS / "performance_context.csv")
    models = list(LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for index, model in enumerate(models):
        selected = sorted((row for row in rows if row["model"] == model),
                          key=lambda row: int(row["training_seed"]))
        values = [float(row["mean_pickups"]) for row in selected]
        offsets = np.linspace(-0.10, 0.10, len(values))
        for offset, row, value in zip(offsets, selected, values):
            seed = int(row["training_seed"])
            style = SEED_STYLES[seed]
            ax.scatter(index + offset, value, s=52, marker=style["marker"],
                       color=style["color"], edgecolor="white", linewidth=0.7,
                       zorder=3)
            ax.annotate(str(seed), (index + offset, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.hlines(np.mean(values), index - 0.22, index + 0.22,
                  color="#202020", linewidth=2)
    ax.set_xticks(range(len(models)), [LABELS[model] for model in models])
    ax.set_ylabel("Mean pickups per test episode")
    ax.set_title("Clean-test pickups by trained policy")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "clean_performance_by_training_seed")


def checkpoint_selection_figure() -> None:
    models = list(LABELS)
    annotation_offsets = {
        "B1_mlp": {42: (4, -12), 43: (4, -12), 44: (-43, 7)},
        "B2_gat": {42: (4, -13), 43: (4, -12), 44: (4, 6)},
        "H5_gat_degraded": {42: (4, -12), 43: (4, -12), 44: (4, 7)},
    }
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.5), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        for seed in SEED_STYLES:
            path = (TRAINING_RUNS / "training" / model / f"seed_{seed}"
                    / "checkpoint_selection.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates = [item for item in payload["candidates"]
                          if item["checkpoint"].startswith("ckpt_epoch_")]
            candidates.sort(key=lambda item: item["epoch"])
            style = SEED_STYLES[seed]
            ax.plot([item["epoch"] for item in candidates],
                    [item["mean_pickups"] for item in candidates],
                    marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], alpha=0.62,
                    markersize=3.5, linewidth=1.2, label=f"seed {seed}")
            selected = payload["selected"]
            ax.scatter(selected["epoch"], selected["mean_pickups"],
                       marker="*", s=135, facecolor=style["color"],
                       edgecolor="#202020", linewidth=0.7, zorder=5)
            ax.annotate(f"{seed}: e{selected['epoch']}",
                        (selected["epoch"], selected["mean_pickups"]),
                        xytext=annotation_offsets[model][seed],
                        textcoords="offset points", fontsize=7.2,
                        color="#202020")
        ax.set_title(LABELS[model])
        ax.grid(color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean validation pickups")
    for ax in axes:
        ax.set_xlabel("Checkpoint epoch")
    axes[-1].legend(frameon=False, fontsize=8, ncol=1)
    fig.suptitle("Validation checkpoint selection")
    fig.tight_layout()
    _save(fig, "checkpoint_selection_by_model_and_seed")


def training_diagnostics_figure() -> None:
    """GAT optimisation diagnostics required by the strategy-capability audit."""
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.3), sharex=True)
    metrics = (
        ("reward", "Reward", True),
        ("pickups", "Training pickups", True),
        ("validation", "Validation pickups", False),
        ("entropy", "Policy entropy", True),
        ("approx_kl", "Approx. KL", True),
        ("clipfrac", "Clip fraction", True),
    )

    for seed, style in SEED_STYLES.items():
        run_dir = TRAINING_RUNS / "training" / "B2_gat" / f"seed_{seed}"
        records = [json.loads(line) for line in
                   (run_dir / "train_log.jsonl").read_text().splitlines() if line]
        epochs = np.array([row["epoch"] for row in records])
        for ax, (field, _, smooth) in zip(axes.flat, metrics):
            if field == "validation":
                selection = json.loads(
                    (run_dir / "checkpoint_selection.json").read_text()
                )
                candidates = sorted(
                    (item for item in selection["candidates"]
                     if item["checkpoint"].startswith("ckpt_epoch_")),
                    key=lambda item: item["epoch"],
                )
                ax.plot(
                    [item["epoch"] for item in candidates],
                    [item["mean_pickups"] for item in candidates],
                    marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], linewidth=1.2, markersize=3,
                    alpha=0.8, label=f"seed {seed}",
                )
                selected = selection["selected"]
                ax.scatter(selected["epoch"], selected["mean_pickups"],
                           marker="*", s=95, color=style["color"],
                           edgecolor="#202020", linewidth=0.5, zorder=4)
                continue
            values = np.array([row[field] for row in records], dtype=float)
            if smooth and values.size >= 10:
                kernel = np.ones(10) / 10
                values = np.convolve(values, kernel, mode="valid")
                x = epochs[9:]
            else:
                x = epochs
            ax.plot(x, values, linestyle=style["linestyle"],
                    color=style["color"], linewidth=1.35,
                    alpha=0.9, label=f"seed {seed}")

    for ax, (_, title, _) in zip(axes.flat, metrics):
        ax.set_title(title)
        ax.grid(color="#D8D8D8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Training epoch")
    axes[0, 2].legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "GAT training and checkpoint-selection diagnostics\n"
        "Training metrics use a 10-epoch moving mean; stars mark selected checkpoints",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "gat_training_diagnostics")


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
                [float(row["mean_stale_attention_shift_when_exposed"])
                 for row in seed_rows],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                color=SEED_STYLES[seed]["color"],
                linewidth=1.7, label=f"seed {seed}",
            )
            axes[1, column].plot(
                [0.0, *[float(row["level_s"]) for row in seed_rows]],
                [float(clean["mean_type_matched_def"]),
                 *[float(row["mean_type_matched_def"]) for row in seed_rows]],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                color=SEED_STYLES[seed]["color"],
                linewidth=1.7,
            )
            axes[2, column].plot(
                [0.0, *[float(row["level_s"]) for row in seed_rows]],
                [float(clean["mean_pickups"]),
                 *[float(row["mean_pickups"]) for row in seed_rows]],
                marker=SEED_STYLES[seed]["marker"],
                linestyle=SEED_STYLES[seed]["linestyle"],
                color=SEED_STYLES[seed]["color"],
                linewidth=1.7,
            )
        axes[0, column].set_title(LABELS[model])
        axes[2, column].set_xlabel("Observation-layer outage duration (s)")
        axes[1, column].set_ylim(-def_limit, def_limit)
        axes[1, column].axhline(0, color="#202020", linewidth=0.9)
        for ax in axes[:, column]:
            ax.grid(color="#D8D8D8", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Paired stale-attention shift")
    axes[1, 0].set_ylabel("Type-matched DEF")
    axes[2, 0].set_ylabel("Mean pickups per episode")
    axes[0, 1].legend(frameon=False, fontsize=8)
    shift_limits = [axes[0, column].get_ylim() for column in range(2)]
    axes[0, 0].set_ylim(min(v[0] for v in shift_limits),
                        max(v[1] for v in shift_limits))
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())
    pickup_limits = [axes[2, column].get_ylim() for column in range(2)]
    axes[2, 0].set_ylim(min(0, min(v[0] for v in pickup_limits)),
                        max(v[1] for v in pickup_limits))
    axes[2, 1].set_ylim(axes[2, 0].get_ylim())
    fig.suptitle("Outage-duration sweep by policy and training seed")
    fig.tight_layout()
    _save(fig, "decoupling_by_outage_duration")


def evidence_summary_figure() -> None:
    synthesis = json.loads((RUNS / "training_seed_synthesis.json").read_text())
    aggregate = synthesis["rows"]
    total = sum(int(row["training_seeds"]) for row in aggregate)
    positive_shift = sum(
        int(row["positive_stale_attention_shift_seeds"]) for row in aggregate
    )
    h1_supported = sum(int(row["H1_supported_seeds"]) for row in aggregate)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [(0.02, 0.10, "Question"), (0.13, 0.25, "Measure"),
               (0.39, 0.38, "Cross-seed result"), (0.78, 0.20, "Verdict")]
    for x, width, title in columns:
        ax.add_patch(FancyBboxPatch((x, 0.76), width, 0.11,
                                   boxstyle="round,pad=0.005,rounding_size=0.008",
                                   facecolor="#E7E9EB", edgecolor="#666666",
                                   linewidth=0.8))
        ax.text(x + width / 2, 0.815, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold")
    rows = [
        ("RQ1", "Clean DEF", "Near type-matched random baseline", "Not validated", "#A33A3A"),
        ("RQ2", "Paired attention shift",
         f"Positive in {positive_shift}/{total} trained policies",
         "Seed-dependent", "#B06C00"),
        ("RQ3", "Outage duration vs DEF",
         f"H1 supported in {h1_supported}/{total} trained policies",
         "No consistent effect", "#555555"),
        ("RQ4", "Degradation training", "No consistent mitigation", "Not supported", "#A33A3A"),
    ]
    for index, (rq, measure, result, verdict, color) in enumerate(rows):
        y = 0.61 - index * 0.135
        values = (rq, measure, result, verdict)
        for (x, width, _), value in zip(columns, values):
            ax.add_patch(FancyBboxPatch((x, y), width, 0.105,
                                       boxstyle="round,pad=0.004,rounding_size=0.006",
                                       facecolor="#F7F8F9", edgecolor=color,
                                       linewidth=0.9))
            ax.text(x + width / 2, y + 0.0525, value, ha="center", va="center",
                    fontsize=8.0, color="#202020",
                    fontweight="bold" if x in (0.02, 0.78) else "normal")
    ax.text(0.5, 0.94, "Cross-seed evidence matrix",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.055,
            "Conclusion: raw attention provides no reproducible faithfulness guarantee across trained policies.",
            ha="center", va="center", fontsize=9.0, fontweight="bold")
    fig.tight_layout()
    _save(fig, "evidence_path_summary")


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
            style = SEED_STYLES[int(row["training_seed"])]
            ax.errorbar(index + offset, value,
                        yerr=[[value - low], [high - value]],
                        fmt=style["marker"], markersize=7, capsize=4,
                        color=style["color"], zorder=3)
            ax.annotate(str(row["training_seed"]),
                        (index + offset, value),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(range(len(COLORS)), [LABELS[model] for model in COLORS])
    y_top = ax.get_ylim()[1]
    for index, model in enumerate(COLORS):
        aggregate_row = next(row for row in payload["rows"] if row["model"] == model)
        ax.text(index, y_top * 0.92,
                f"{aggregate_row['positive_stale_attention_shift_seeds']}/"
                f"{aggregate_row['training_seeds']} positive",
                ha="center", va="bottom", fontsize=8.2, color="#444444")
    ax.set_ylabel("Attention mass on exposed peers:\ndegraded minus clean twin")
    ax.set_title("Paired stale-attention shift")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "paired_stale_attention_shift")


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
            style = SEED_STYLES[int(row["training_seed"])]
            ax.scatter(index + offset, row["H4_within_episode_rho"], s=70,
                       marker=style["marker"],
                       facecolor=style["color"] if supported else "white",
                       edgecolor=style["color"], linewidth=1.8, zorder=3)
            ax.annotate(str(row["training_seed"]),
                        (index + offset, row["H4_within_episode_rho"]),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(range(len(COLORS)), [LABELS[model] for model in COLORS])
    ax.set_ylabel("Within-episode Spearman rho (WAMSN vs DEF)")
    ax.set_title("Within-episode WAMSN-DEF correlation by trained policy")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="#555555", markersize=7,
               label="H4 supported after Holm correction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#555555", markersize=7,
               label="H4 not supported"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=7.8, loc="lower left")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "h4_correlation_by_training_seed")


def main() -> None:
    global RUNS, TRAINING_RUNS, OUT, PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS)
    parser.add_argument(
        "--training-root", type=Path,
        help="experiment root containing training logs (defaults to --root)",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    RUNS = args.root
    TRAINING_RUNS = args.training_root or RUNS
    OUT = args.out
    PREFIX = args.prefix
    performance_figure()
    checkpoint_selection_figure()
    training_diagnostics_figure()
    decoupling_figure()
    shift_figure()
    consistency_figure()
    evidence_summary_figure()
    print(f"figures: {OUT}")


if __name__ == "__main__":
    main()
