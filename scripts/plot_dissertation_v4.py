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
        ("RQ1", "Clean DEF", "Near type-matched random baseline", "Not validated", "#A33A3A"),
        ("RQ2", "Paired attention shift",
         f"Positive in {positive_shift}/{total} trained policies",
         "Mixed by seed", "#B06C00"),
        ("RQ3", "Paired DEF shift",
         f"Negative in {negative_paired_def}/{total} trained policies",
         "Mixed by seed", "#B06C00"),
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
            "Conclusion: attention and paired faithfulness responses are not consistent across trained policies.",
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


def action_stratified_figure() -> None:
    """Show action volume, confidence, and paired DEF by chosen action."""
    rows = _csv(RUNS / "action_stratified.csv")
    checkpoints = sorted({(row["model"], int(row["training_seed"])) for row in rows})
    labels = [str(seed) for _, seed in checkpoints]
    x = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    by_key = {(row["model"], int(row["training_seed"]), row["stratum"]): row
              for row in rows}

    fig = plt.figure(figsize=(8.0, 5.7))
    axes = fig.subplot_mosaic(
        [["counts", "probability"], ["def", "def"]],
        height_ratios=[0.9, 1.15],
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

    width = 0.32
    no_op_probability = np.array([
        float(by_key[model, seed, "no_op"]["mean_selected_action_probability"])
        for model, seed in checkpoints
    ])
    dispatch_probability = np.array([
        float(by_key[model, seed, "dispatch"]["mean_selected_action_probability"])
        for model, seed in checkpoints
    ])
    axes["probability"].bar(
        x - width / 2, no_op_probability, width=width, color="#176B87",
        label="No-op",
    )
    axes["probability"].bar(
        x + width / 2, dispatch_probability, width=width, color="#C75000",
        hatch="///", edgecolor="white", linewidth=0.6, label="Dispatch",
    )
    axes["probability"].set_ylim(0, 1.04)
    axes["probability"].set_ylabel("Mean selected-action probability")
    axes["probability"].set_title("(b) Probability of the chosen action", loc="left")
    axes["probability"].legend(frameon=False, fontsize=8, loc="center right")

    strata = (
        ("all", "All", "o", "#555555", -0.16),
        ("no_op", "No-op", "s", "#176B87", 0.0),
        ("dispatch", "Dispatch", "^", "#C75000", 0.16),
    )
    for stratum, label, marker, color, offset in strata:
        values = []
        lows = []
        highs = []
        for model, seed in checkpoints:
            row = by_key[model, seed, stratum]
            value = float(row["paired_probability_def_delta"]) * 1e4
            values.append(value)
            lows.append(value - float(row["paired_probability_def_ci_low"]) * 1e4)
            highs.append(float(row["paired_probability_def_ci_high"]) * 1e4 - value)
        axes["def"].errorbar(
            x + offset, values, yerr=[lows, highs], fmt=marker, color=color,
            markerfacecolor="white" if stratum == "all" else color,
            markersize=5.5, capsize=3, linewidth=1.1, label=label,
        )
    axes["def"].axhline(0, color="#202020", linewidth=0.9)
    axes["def"].set_ylabel(r"Paired probability DEF change ($\times 10^{-4}$)")
    axes["def"].set_title("(c) Degraded twin minus clean twin", loc="left")
    axes["def"].legend(frameon=False, fontsize=8, loc="lower left", ncol=3)

    for ax in axes.values():
        ax.set_xticks(x, labels)
        ax.text(1, -0.14, "GAT", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(5, -0.14, "GAT-Outage", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Action-stratified faithfulness diagnostic", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
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
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.65))

    width = 0.24
    for condition, label, _, color, offset in conditions:
        exposure = [
            float(by_key[model, seed, condition]["empirical_degradation_rate"]) * 100
            for model, seed in checkpoints
        ]
        axes[0].bar(
            x + offset, exposure, width=width, color=color,
            hatch="///" if condition == "random" else None,
            edgecolor="white", linewidth=0.5, label=label,
        )
    axes[0].set_ylabel("Degraded observations (%)")
    axes[0].set_title("(a) Observed exposure", loc="left")

    metric_specs = (
        ("stale_attention_shift", "stale_attention_ci_low",
         "stale_attention_ci_high", 1e3,
         r"Stale-attention shift ($\times 10^{-3}$)", "(b) Attention shift"),
        ("paired_probability_def_shift", "paired_probability_def_ci_low",
         "paired_probability_def_ci_high", 1e4,
         r"Probability DEF shift ($\times 10^{-4}$)", "(c) Faithfulness shift"),
    )
    for ax, (value_key, low_key, high_key, scale, ylabel, title) in zip(
        axes[1:], metric_specs
    ):
        for condition, label, marker, color, offset in conditions:
            values, lows, highs = [], [], []
            for model, seed in checkpoints:
                row = by_key[model, seed, condition]
                value = float(row[value_key]) * scale
                values.append(value)
                lows.append(value - float(row[low_key]) * scale)
                highs.append(float(row[high_key]) * scale - value)
            ax.errorbar(
                x + offset, values, yerr=[lows, highs], fmt=marker,
                color=color, markerfacecolor=color, markersize=4.8,
                linewidth=1.0, capsize=2.5, label=label,
            )
        ax.axhline(0, color="#202020", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.text(1, -0.17, "GAT", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(5, -0.17, "GAT-Outage", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Sensitivity to tunnel-triggered versus random telemetry loss")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
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
