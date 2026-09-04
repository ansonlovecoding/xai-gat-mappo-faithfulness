"""Create thesis figures from a validated dissertation experiment."""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "dissertation_v4"
TRAINING_RUNS = RUNS
ANALYSIS = ROOT / "results" / "dissertation_v9_exposure_audit"
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
    """Training diagnostics for both graph-attention policies."""
    fig, axes = plt.subplots(4, 3, figsize=(8.0, 7.8), sharex=True)
    metrics = (
        ("reward", "Reward", True),
        ("pickups", "Training pickups", True),
        ("validation", "Validation pickups", False),
        ("entropy", "Policy entropy", True),
        ("approx_kl", "Approx. KL", True),
        ("clipfrac", "Clip fraction", True),
    )
    models = (("B2_gat", "GAT"), ("H5_gat_degraded", "GAT-Outage"))

    for model_index, (model, model_label) in enumerate(models):
        model_axes = list(axes[model_index * 2:(model_index + 1) * 2].flat)
        for seed, style in SEED_STYLES.items():
            run_dir = TRAINING_RUNS / "training" / model / f"seed_{seed}"
            records = [
                json.loads(line)
                for line in (run_dir / "train_log.jsonl").read_text().splitlines()
                if line
            ]
            epochs = np.array([row["epoch"] for row in records])
            for ax, (field, _, smooth) in zip(model_axes, metrics):
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
                        color=style["color"], linewidth=1.1, markersize=2.8,
                        alpha=0.8, label=f"seed {seed}",
                    )
                    selected = selection["selected"]
                    ax.scatter(
                        selected["epoch"], selected["mean_pickups"],
                        marker="*", s=75, color=style["color"],
                        edgecolor="#202020", linewidth=0.5, zorder=4,
                    )
                    continue
                values = np.array([row[field] for row in records], dtype=float)
                if smooth and values.size >= 10:
                    values = np.convolve(values, np.ones(10) / 10, mode="valid")
                    x = epochs[9:]
                else:
                    x = epochs
                ax.plot(
                    x, values, linestyle=style["linestyle"],
                    color=style["color"], linewidth=1.2,
                    alpha=0.9, label=f"seed {seed}",
                )

        for ax, (_, title, _) in zip(model_axes, metrics):
            ax.set_title(title, fontsize=9.5)
            ax.grid(color="#D8D8D8", linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[model_index * 2 + 1]:
            ax.set_xlabel("Training epoch", fontsize=8.5)
        axes[model_index * 2, 0].text(
            -0.24, -0.08, model_label, transform=axes[model_index * 2, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=10,
            fontweight="bold", color="#202020",
        )

    axes[0, 2].legend(frameon=False, fontsize=7.0, loc="best")
    fig.suptitle(
        "GAT and GAT-Outage training diagnostics\n"
        "Training metrics use a 10-epoch moving mean; stars mark selected checkpoints",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.055, 0, 1, 0.94), h_pad=1.0, w_pad=0.9)
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
                [float(row["mean_wamsn_when_exposed_secondary"])
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
    axes[0, 0].set_ylabel("WAMSN (stale-exposed decisions)")
    axes[1, 0].set_ylabel("Probability DEF")
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
    negative_paired_def = sum(
        int(row["negative_paired_probability_def_delta_seeds"])
        for row in aggregate
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
        ("RQ1", "Clean decision relevance", "Near type-matched random baseline", "Not validated", "#A33A3A"),
        ("RQ2", "Paired stale-node attention",
         f"Positive in {positive_shift}/{total} trained policies",
         "Mixed by seed", "#B06C00"),
        ("RQ3", "Duration and paired DEF",
         f"Paired decline in {negative_paired_def}/{total} policies",
         "Not consistent", "#B06C00"),
        ("RQ4", "Training comparison", "No consistent improvement", "Not supported", "#A33A3A"),
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
            "Conclusion: raw attention does not provide consistent decision-relevance or degradation responses.",
            ha="center", va="center", fontsize=9.0, fontweight="bold")
    fig.tight_layout()
    _save(fig, "evidence_path_summary")


def shift_figure() -> None:
    payload = json.loads((RUNS / "training_seed_synthesis.json").read_text())
    rows = payload["per_seed"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
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
            style = SEED_STYLES[int(row["training_seed"])]
            metrics = (
                (analysis["robust"]["primary_stale_attention_shift"],
                 "mean_shift", 1.0),
                (analysis["robust"]["exposure_conditioned_paired_followup"]
                 ["probability_def"], "mean_degraded_minus_clean", 1e4),
            )
            for ax, (result, key, scale) in zip(axes, metrics):
                value = float(result[key]) * scale
                low, high = [float(bound) * scale for bound in result["ci95"]]
                ax.errorbar(index + offset, value,
                            yerr=[[value - low], [high - value]],
                            fmt=style["marker"], markersize=7, capsize=4,
                            color=style["color"], zorder=3)
                ax.annotate(str(row["training_seed"]),
                            (index + offset, value),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8)
    for ax in axes:
        ax.axhline(0, color="#202020", linewidth=1)
        ax.set_xticks(range(len(COLORS)), [LABELS[model] for model in COLORS])
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Attention mass shift\n(degraded minus clean twin)")
    axes[0].set_title("Stale-node attention")
    axes[1].set_ylabel("Probability DEF shift (x10^-4)\n(degraded minus clean twin)")
    axes[1].set_title("Explanation faithfulness")
    fig.suptitle("Exposure-conditioned paired audit by trained policy")
    fig.tight_layout()
    _save(fig, "paired_attention_and_faithfulness_shift")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including tied values."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _h4_episode_data(model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Recreate H4 episode blocks and descriptive WAMSN groups."""
    archive_path = (
        ANALYSIS / "release" / "audit_records" /
        f"{model}_seed_{seed}_cells.tar.gz"
    )
    blocks: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if "/outage_duration_" not in member.name:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = json.load(handle)
            for record in payload["faith_records"]:
                if int(record["valid_reservations"]) <= 0:
                    continue
                wamsn = float(record["wamsn"])
                probability_def = float(record["def"])
                if not np.isfinite(wamsn) or not np.isfinite(probability_def):
                    continue
                key = (member.name, int(record["episode"]))
                x_values, y_values = blocks.setdefault(key, ([], []))
                x_values.append(wamsn)
                y_values.append(probability_def)

    correlations: list[float] = []
    grouped_def_ranks: list[list[float]] = []
    for x_values, y_values in blocks.values():
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        if len(x) < 8 or np.allclose(x, x[0]):
            continue
        x_rank = _average_ranks(x)
        y_rank = _average_ranks(y)
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()
        denominator = np.sqrt(
            np.sum(x_centered ** 2) * np.sum(y_centered ** 2)
        )
        correlations.append(
            0.0 if denominator == 0
            else float(np.sum(x_centered * y_centered) / denominator)
        )

        order = np.argsort(x_rank, kind="mergesort")
        groups = np.array_split(order, 4)
        scale = max(len(y_rank) - 1, 1)
        grouped_def_ranks.append([
            float(np.mean(y_rank[group]) / scale * 100.0) for group in groups
        ])
    return np.asarray(correlations), np.asarray(grouped_def_ranks)


def consistency_figure() -> None:
    payload = json.loads((RUNS / "training_seed_synthesis.json").read_text())
    rows = payload["per_seed"]
    fig, (trend_ax, result_ax) = plt.subplots(
        1, 2, figsize=(8.0, 4.15), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )
    episode_data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    group_x = np.arange(4)
    for model in COLORS:
        model_groups = []
        for seed in SEED_STYLES:
            data = _h4_episode_data(model, seed)
            episode_data[model, seed] = data
            model_groups.append(data[1])
        grouped = np.concatenate(model_groups, axis=0)
        center = np.mean(grouped, axis=0)
        lower, upper = np.percentile(grouped, [25, 75], axis=0)
        trend_ax.plot(
            group_x, center, marker="o", linewidth=2.0,
            color=COLORS[model], label=LABELS[model],
        )
        trend_ax.fill_between(
            group_x, lower, upper, color=COLORS[model], alpha=0.13,
            linewidth=0,
        )
    trend_ax.set_xticks(group_x, ["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
    trend_ax.set_xlabel("WAMSN group within each episode")
    trend_ax.set_ylabel("Mean within-episode DEF rank (%)")
    trend_ax.set_title("(a) DEF rank across WAMSN groups")
    trend_ax.legend(frameon=False, fontsize=8, loc="best")
    trend_ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    trend_ax.spines[["top", "right"]].set_visible(False)

    ordered_rows = [
        row for model in COLORS for row in sorted(
            (candidate for candidate in rows if candidate["model"] == model),
            key=lambda candidate: candidate["training_seed"],
        )
    ]
    y_positions = np.arange(len(ordered_rows) - 1, -1, -1)
    for y_position, row in zip(y_positions, ordered_rows):
        model = row["model"]
        seed = int(row["training_seed"])
        correlations = episode_data[model, seed][0]
        rng = np.random.default_rng(
            41100 + seed + 1000 * list(COLORS).index(model)
        )
        sampled = rng.choice(
            correlations, size=(2000, len(correlations)), replace=True
        ).mean(axis=1)
        low, high = np.percentile(sampled, [2.5, 97.5])
        value = float(row["H4_within_episode_rho"])
        style = SEED_STYLES[seed]
        supported = bool(row["H4_supported"])
        result_ax.errorbar(
            value, y_position,
            xerr=[[value - low], [high - value]],
            fmt=style["marker"], markersize=7, capsize=3,
            color=style["color"], markerfacecolor=(
                style["color"] if supported else "white"
            ), markeredgewidth=1.4, zorder=3,
        )
        result_ax.text(
            0.045, y_position,
            "SUPPORTED" if supported else "NOT SUPPORTED",
            va="center", ha="left", fontsize=7.5, fontweight="bold",
            color="#2F6B3C" if supported else "#9B2C2C",
        )
    result_ax.axvline(0, color="#202020", linewidth=1)
    result_ax.axhline(2.5, color="#A8A8A8", linewidth=0.8)
    result_ax.set_yticks(
        y_positions,
        [f"{LABELS[row['model']]} {row['training_seed']}" for row in ordered_rows],
    )
    result_ax.set_xlim(-0.55, 0.23)
    result_ax.set_xlabel("Mean within-episode Spearman rho")
    result_ax.set_title("(b) H4 result: 6/6 supported")
    result_ax.grid(axis="x", color="#D8D8D8", linewidth=0.7)
    result_ax.spines[["top", "right", "left"]].set_visible(False)
    result_ax.tick_params(axis="y", length=0)

    fig.suptitle("WAMSN-DEF relationship and H4 decision", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5)
    _save(fig, "h4_correlation_by_training_seed")


def action_stratified_figure() -> None:
    """Show action volume, confidence, and paired DEF by chosen action."""
    rows = _csv(RUNS / "action_stratified.csv")
    checkpoints = sorted({(row["model"], int(row["training_seed"])) for row in rows})
    labels = [str(seed) for _, seed in checkpoints]
    x = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    by_key = {(row["model"], int(row["training_seed"]), row["stratum"]): row
              for row in rows}

    fig = plt.figure(figsize=(8.0, 6.2))
    axes = fig.subplot_mosaic(
        [["counts", "probability"], ["def_full", "def_zoom"]],
        height_ratios=[0.95, 1.05],
    )
    dispatch_fraction = np.array([
        float(by_key[model, seed, "dispatch"]["fraction_of_eligible_records"])
        for model, seed in checkpoints
    ])
    no_op_counts = np.array([
        int(by_key[model, seed, "no_op"]["n_records"])
        for model, seed in checkpoints
    ])
    dispatch_counts = np.array([
        int(by_key[model, seed, "dispatch"]["n_records"])
        for model, seed in checkpoints
    ])
    axes["counts"].bar(x, no_op_counts, color="#AEB8C2", label="No-op")
    axes["counts"].bar(
        x, dispatch_counts, bottom=no_op_counts, color="#C75000",
        hatch="///", edgecolor="white", linewidth=0.6, label="Dispatch",
    )
    count_ceiling = float(np.max(no_op_counts + dispatch_counts))
    for x_pos, total, fraction in zip(
        x, no_op_counts + dispatch_counts, dispatch_fraction
    ):
        axes["counts"].text(
            x_pos, total + count_ceiling * 0.025,
            f"{fraction * 100:.1f}% dispatch", ha="center", va="bottom",
            fontsize=7.2,
        )
    axes["counts"].set_ylim(0, count_ceiling * 1.16)
    axes["counts"].set_ylabel("Eligible decisions (count)")
    axes["counts"].set_title("(a) Chosen-action counts", loc="left")

    no_op_probability = np.array([
        float(by_key[model, seed, "no_op"]["mean_selected_action_probability"])
        for model, seed in checkpoints
    ])
    dispatch_probability = np.array([
        float(by_key[model, seed, "dispatch"]["mean_selected_action_probability"])
        for model, seed in checkpoints
    ])
    axes["probability"].scatter(
        x - 0.12, no_op_probability, s=34, marker="s", color="#176B87",
        label="No-op", zorder=3,
    )
    axes["probability"].scatter(
        x + 0.12, dispatch_probability, s=38, marker="^", color="#C75000",
        label="Dispatch", zorder=3,
    )
    axes["probability"].set_yscale("log")
    axes["probability"].set_ylim(0.005, 1.2)
    axes["probability"].set_ylabel("Mean selected-action probability")
    axes["probability"].set_title(
        "(b) Probability of the chosen action (log scale)", loc="left"
    )
    axes["probability"].legend(frameon=False, fontsize=8, loc="lower right")

    strata = (
        ("all", "All", "o", "#555555", -0.16),
        ("no_op", "No-op", "s", "#176B87", 0.0),
        ("dispatch", "Dispatch", "^", "#C75000", 0.16),
    )
    series = []
    for stratum, label, marker, color, offset in strata:
        values, ci_lows, ci_highs = [], [], []
        for model, seed in checkpoints:
            row = by_key[model, seed, stratum]
            value = float(row["paired_probability_def_delta"]) * 1e4
            values.append(value)
            ci_lows.append(float(row["paired_probability_def_ci_low"]) * 1e4)
            ci_highs.append(float(row["paired_probability_def_ci_high"]) * 1e4)
        values = np.asarray(values)
        ci_lows = np.asarray(ci_lows)
        ci_highs = np.asarray(ci_highs)
        series.append((label, marker, color, offset, values, ci_lows, ci_highs))
        axes["def_full"].errorbar(
            x + offset, values,
            yerr=[values - ci_lows, ci_highs - values], fmt=marker, color=color,
            markerfacecolor="white" if stratum == "all" else color,
            markersize=5.5, capsize=3, linewidth=1.1, label=label,
        )
    axes["def_full"].axhline(0, color="#202020", linewidth=0.9)
    axes["def_full"].set_ylabel(r"Paired probability DEF change ($\times 10^{-4}$)")
    axes["def_full"].set_title("(c) DEF change: full scale", loc="left")
    axes["def_full"].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3)

    zoom_low, zoom_high = -1.2, 1.2
    for label, marker, color, offset, values, ci_lows, ci_highs in series:
        visible = (values >= zoom_low) & (values <= zoom_high)
        if np.any(visible):
            shown = values[visible]
            shown_lows = np.maximum(ci_lows[visible], zoom_low)
            shown_highs = np.minimum(ci_highs[visible], zoom_high)
            axes["def_zoom"].errorbar(
                x[visible] + offset, shown,
                yerr=[shown - shown_lows, shown_highs - shown],
                fmt=marker, color=color, markerfacecolor=color,
                markersize=5.5, capsize=3, linewidth=1.1,
            )
        for x_pos, value in zip(x[~visible] + offset, values[~visible]):
            boundary = zoom_low + 0.04 if value < zoom_low else zoom_high - 0.04
            direction = "v" if value < zoom_low else "^"
            axes["def_zoom"].scatter(x_pos, boundary, marker=direction,
                                     s=42, color=color, zorder=4)
            axes["def_zoom"].annotate(
                f"{value:+.1f}", (x_pos, boundary),
                xytext=(8, 7 if value < zoom_low else -12),
                textcoords="offset points", ha="center", fontsize=6.8,
            )
    axes["def_zoom"].axhline(0, color="#202020", linewidth=0.9)
    axes["def_zoom"].set_ylim(zoom_low, zoom_high)
    axes["def_zoom"].set_ylabel(r"Paired probability DEF change ($\times 10^{-4}$)")
    axes["def_zoom"].set_title("(d) DEF change: enlarged near zero", loc="left")

    for ax in axes.values():
        ax.set_xticks(x, labels)
        ax.text(1, -0.16, "GAT", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(5, -0.16, "GAT-Outage", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Faithfulness analysis by action type", y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    _save(fig, "action_stratified_faithfulness")


def random_loss_robustness_figure() -> None:
    """Compare tunnel and random triggers at the same 30 s freeze duration."""
    payload = json.loads((ANALYSIS / "random_loss_robustness.json").read_text())
    rows = payload["rows"]
    checkpoints = sorted({(row["model"], int(row["training_seed"])) for row in rows})
    by_key = {
        (row["model"], int(row["training_seed"]), row["condition"]): row
        for row in rows
    }
    x = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    labels = [str(seed) for _, seed in checkpoints]
    conditions = (
        ("tunnel", "Tunnel trigger", "o", "#C75000", -0.12),
        ("random", "Random trigger", "s", "#176B87", 0.12),
    )
    fig = plt.figure(figsize=(8.0, 5.25))
    axes = fig.subplot_mosaic(
        [["exposure", "attention"], ["def_full", "def_zoom"]]
    )

    width = 0.24
    for condition, label, _, color, offset in conditions:
        exposure = [
            float(by_key[model, seed, condition]["empirical_degradation_rate"]) * 100
            for model, seed in checkpoints
        ]
        axes["exposure"].bar(
            x + offset, exposure, width=width, color=color,
            hatch="///" if condition == "random" else None,
            edgecolor="white", linewidth=0.5, label=label,
        )
    axes["exposure"].set_ylabel("Degraded observations (%)")
    axes["exposure"].set_title("(a) Observed exposure", loc="left")

    for condition, label, marker, color, offset in conditions:
        values, ci_lows, ci_highs = [], [], []
        for model, seed in checkpoints:
            row = by_key[model, seed, condition]
            value = float(row["stale_attention_shift"]) * 1e3
            values.append(value)
            ci_lows.append(float(row["stale_attention_ci_low"]) * 1e3)
            ci_highs.append(float(row["stale_attention_ci_high"]) * 1e3)
        values = np.asarray(values)
        axes["attention"].errorbar(
            x + offset, values,
            yerr=[values - np.asarray(ci_lows), np.asarray(ci_highs) - values],
            fmt=marker, color=color, markerfacecolor=color, markersize=4.8,
            linewidth=1.0, capsize=2.5, label=label,
        )
    axes["attention"].axhline(0, color="#202020", linewidth=0.8)
    axes["attention"].set_ylabel(r"Stale-attention shift ($\times 10^{-3}$)")
    axes["attention"].set_title("(b) Attention shift", loc="left")

    def_series = []
    for condition, label, marker, color, offset in conditions:
        values, ci_lows, ci_highs = [], [], []
        for model, seed in checkpoints:
            row = by_key[model, seed, condition]
            values.append(float(row["paired_probability_def_shift"]) * 1e4)
            ci_lows.append(float(row["paired_probability_def_ci_low"]) * 1e4)
            ci_highs.append(float(row["paired_probability_def_ci_high"]) * 1e4)
        values = np.asarray(values)
        ci_lows = np.asarray(ci_lows)
        ci_highs = np.asarray(ci_highs)
        def_series.append((label, marker, color, offset, values, ci_lows, ci_highs))
        axes["def_full"].errorbar(
            x + offset, values, yerr=[values - ci_lows, ci_highs - values],
            fmt=marker, color=color, markerfacecolor=color, markersize=4.8,
            linewidth=1.0, capsize=2.5, label=label,
        )
    axes["def_full"].axhline(0, color="#202020", linewidth=0.8)
    axes["def_full"].set_ylabel(r"Probability DEF shift ($\times 10^{-4}$)")
    axes["def_full"].set_title("(c) Faithfulness shift: full scale", loc="left")

    zoom_low, zoom_high = -1.0, 1.0
    for label, marker, color, offset, values, ci_lows, ci_highs in def_series:
        visible = (values >= zoom_low) & (values <= zoom_high)
        shown = values[visible]
        shown_lows = np.maximum(ci_lows[visible], zoom_low)
        shown_highs = np.minimum(ci_highs[visible], zoom_high)
        axes["def_zoom"].errorbar(
            x[visible] + offset, shown,
            yerr=[shown - shown_lows, shown_highs - shown],
            fmt=marker, color=color, markerfacecolor=color, markersize=4.8,
            linewidth=1.0, capsize=2.5,
        )
        for x_pos, value in zip(x[~visible] + offset, values[~visible]):
            boundary = zoom_low + 0.05 if value < zoom_low else zoom_high - 0.05
            direction = "v" if value < zoom_low else "^"
            axes["def_zoom"].scatter(x_pos, boundary, marker=direction,
                                     s=42, color=color, zorder=4)
            axes["def_zoom"].annotate(
                f"{value:+.1f}", (x_pos, boundary),
                xytext=(0, 7 if value < zoom_low else -12),
                textcoords="offset points", ha="center", fontsize=6.8,
            )
    axes["def_zoom"].axhline(0, color="#202020", linewidth=0.8)
    axes["def_zoom"].set_ylim(zoom_low, zoom_high)
    axes["def_zoom"].set_ylabel(r"Probability DEF shift ($\times 10^{-4}$)")
    axes["def_zoom"].set_title("(d) Faithfulness shift: enlarged near zero", loc="left")

    for ax in axes.values():
        ax.set_xticks(x, labels)
        ax.text(1, -0.17, "GAT", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(5, -0.17, "GAT-Outage", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes["exposure"].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Sensitivity to tunnel-triggered versus random telemetry loss")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save(fig, "random_loss_robustness")


def main() -> None:
    global RUNS, TRAINING_RUNS, ANALYSIS, OUT, PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS)
    parser.add_argument(
        "--training-root", type=Path,
        help="experiment root containing training logs (defaults to --root)",
    )
    parser.add_argument(
        "--analysis-root", type=Path, default=ANALYSIS,
        help="directory containing cross-sweep analysis JSON files",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()
    RUNS = args.root
    TRAINING_RUNS = args.training_root or RUNS
    ANALYSIS = args.analysis_root
    OUT = args.out
    PREFIX = args.prefix
    performance_figure()
    checkpoint_selection_figure()
    training_diagnostics_figure()
    decoupling_figure()
    shift_figure()
    consistency_figure()
    action_stratified_figure()
    random_loss_robustness_figure()
    evidence_summary_figure()
    print(f"figures: {OUT}")


if __name__ == "__main__":
    main()
