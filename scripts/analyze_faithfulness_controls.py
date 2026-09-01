"""Aggregate the B2/D30 faithfulness-control runs at episode-block level."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "runs/dissertation_revision_v1/faithfulness_controls/B2_full",
    ROOT / "runs/dissertation_revision_v1/faithfulness_controls/D30_full",
    ROOT / "runs/dissertation_revision_v1/faithfulness_controls/B2_action_row",
    ROOT / "runs/dissertation_revision_v1/faithfulness_controls/D30_action_row",
]
DEFAULT_OUT = ROOT / "results/dissertation_revision_v1/faithfulness_controls"
FIG_DIR = ROOT / "docs/figures"
FIG_PREFIX = "v4"
MODEL_LABEL = {"B2_gat": "B2", "H5_gat_degraded": "D30"}
MODEL_DISPLAY = {"B2": "GAT", "D30": "GAT-Outage"}
RANKER_LABEL = {
    "attention_control_def_m": "Raw attention",
    "grad_x_input_def_m": "Gradient x Input",
    "loo_oracle_def_m": "LOO control",
}


def _load_cells(paths: list[Path]) -> list[dict]:
    cells = []
    for root in paths:
        cells.extend(json.loads(path.read_text())
                     for path in sorted((root / "cells").glob("*.json")))
    if not cells:
        raise SystemExit("no completed control cells found")
    return cells


def _episode_means(cells: list[dict], field: str, *, stale_only: bool = False,
                   min_valid_nodes: int | None = None) -> list[dict]:
    rows = []
    for cell in cells:
        by_episode: dict[int, list[float]] = {}
        for record in cell.get("faith_records", []):
            if record.get("valid_reservations", 0) <= 0:
                continue
            if stale_only and record.get("n_stale_veh", 0) <= 0:
                continue
            if (min_valid_nodes is not None
                    and record.get("valid_non_self_nodes", 0) < min_valid_nodes):
                continue
            value = record.get(field)
            if value is None or not np.isfinite(value):
                continue
            by_episode.setdefault(int(record["episode"]), []).append(float(value))
        for episode, values in by_episode.items():
            rows.append({
                "model_id": cell["model_id"],
                "condition": cell["condition_name"],
                "training_seed": int(cell["training_seed"]),
                "eval_seed": int(cell["eval_seed"]),
                "episode": episode,
                "value": float(np.mean(values)),
                "n_decisions": len(values),
            })
    return rows


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                  n_boot: int = 5000) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _metric_summary(cells: list[dict], rng: np.random.Generator) -> list[dict]:
    output = []
    for field, ranker in RANKER_LABEL.items():
        rows = _episode_means(cells, field)
        for model in sorted({row["model_id"] for row in rows}):
            for condition in ("clean", "outage_60s"):
                selected = [row for row in rows
                            if row["model_id"] == model and row["condition"] == condition]
                values = np.array([row["value"] for row in selected])
                output.append({
                    "model_id": model,
                    "model": MODEL_LABEL[model],
                    "condition": condition,
                    "ranker": ranker,
                    "field": field,
                    "mean": float(values.mean()) if values.size else None,
                    "ci95": _bootstrap_ci(values, rng) if values.size else None,
                    "n_episode_blocks": len(selected),
                    "n_decisions": int(sum(row["n_decisions"] for row in selected)),
                })
    return output


def _seed_summary(cells: list[dict], fields: list[str]) -> list[dict]:
    output = []
    for field in fields:
        rows = _episode_means(cells, field)
        keys = sorted({(r["model_id"], r["condition"], r["training_seed"])
                       for r in rows})
        for model, condition, seed in keys:
            selected = [r for r in rows if (r["model_id"], r["condition"],
                                             r["training_seed"]) == (model, condition, seed)]
            output.append({
                "model_id": model,
                "model": MODEL_LABEL[model],
                "condition": condition,
                "training_seed": seed,
                "field": field,
                "mean": float(np.mean([r["value"] for r in selected])),
                "n_episode_blocks": len(selected),
                "n_decisions": int(sum(r["n_decisions"] for r in selected)),
            })
    return output


def _large_graph_summary(cells: list[dict], rng: np.random.Generator) -> list[dict]:
    """P0-3 sensitivity restricted to decisions with at least six nodes."""
    output = []
    for field, ranker in RANKER_LABEL.items():
        rows = _episode_means(cells, field, min_valid_nodes=6)
        keys = sorted({(r["model_id"], r["condition"]) for r in rows})
        for model, condition in keys:
            selected = [r for r in rows
                        if r["model_id"] == model and r["condition"] == condition]
            values = np.array([r["value"] for r in selected])
            output.append({
                "model_id": model,
                "model": MODEL_LABEL[model],
                "condition": condition,
                "ranker": ranker,
                "field": field,
                "minimum_valid_non_self_nodes": 6,
                "mean": float(values.mean()),
                "ci95": _bootstrap_ci(values, rng),
                "n_episode_blocks": len(selected),
                "n_decisions": int(sum(r["n_decisions"] for r in selected)),
            })
    return output


def _overlap_summary(cells: list[dict]) -> list[dict]:
    output = []
    for model in sorted({cell["model_id"] for cell in cells}):
        records = [r for cell in cells if cell["model_id"] == model
                   for r in cell.get("faith_records", [])
                   if r.get("valid_reservations", 0) > 0]
        for k in ("1", "2", "3"):
            expected = [r["expected_random_topk_overlap_by_k"][k] for r in records
                        if k in r.get("expected_random_topk_overlap_by_k", {})]
            agreement = [r["attention_loo_topk_agreement_by_k"][k] for r in records
                         if k in r.get("attention_loo_topk_agreement_by_k", {})]
            output.append({
                "model_id": model,
                "model": MODEL_LABEL[model],
                "k": int(k),
                "expected_random_overlap": float(np.mean(expected)),
                "attention_loo_agreement": float(np.mean(agreement)),
                "n_decisions": len(expected),
            })
    return output


def _aggregation_summary(cells: list[dict]) -> list[dict]:
    output = []
    for model in sorted({cell["model_id"] for cell in cells}):
        for seed in sorted({int(cell["training_seed"]) for cell in cells
                            if cell["model_id"] == model}):
            records = [r for cell in cells
                       if cell["model_id"] == model
                       and int(cell["training_seed"]) == seed
                       and cell["condition_name"] == "outage_60s"
                       for r in cell.get("faith_records", [])
                       if r.get("n_stale_veh", 0) > 0
                       and "aggregation_stale_attention_shift" in r]
            names = sorted({name for r in records
                            for name in r["aggregation_stale_attention_shift"]})
            default_values = [r["stale_attention_shift"] for r in records
                              if r.get("stale_attention_shift") is not None]
            if default_values:
                output.append({
                    "model_id": model,
                    "model": MODEL_LABEL[model],
                    "training_seed": seed,
                    "aggregation": "default_mean",
                    "mean_stale_attention_shift": float(np.mean(default_values)),
                    "n_stale_exposed_decisions": len(default_values),
                })
            for name in names:
                values = [r["aggregation_stale_attention_shift"][name] for r in records]
                output.append({
                    "model_id": model,
                    "model": MODEL_LABEL[model],
                    "training_seed": seed,
                    "aggregation": name,
                    "mean_stale_attention_shift": float(np.mean(values)),
                    "n_stale_exposed_decisions": len(values),
                })
    return output


def _action_row_summary(cells: list[dict], rng: np.random.Generator) -> list[dict]:
    output = []
    fields = {
        "self_row_request_def_m": "Fixed self row",
        "action_row_request_def_m": "Selected request row",
        "action_row_request_gap_m": "Request row minus self row",
    }
    for field, label in fields.items():
        rows = _episode_means(cells, field)
        for model in sorted({r["model_id"] for r in rows}):
            for condition in ("clean", "outage_60s"):
                selected = [r for r in rows
                            if r["model_id"] == model and r["condition"] == condition]
                values = np.array([r["value"] for r in selected])
                if not values.size:
                    continue
                output.append({
                    "model_id": model,
                    "model": MODEL_LABEL[model],
                    "condition": condition,
                    "field": field,
                    "label": label,
                    "mean": float(values.mean()),
                    "ci95": _bootstrap_ci(values, rng),
                    "n_episode_blocks": len(selected),
                    "n_request_decisions": int(sum(
                        r["n_decisions"] for r in selected
                    )),
                })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_controls(seed_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5), sharey="col")
    fields = list(RANKER_LABEL)
    labels = [RANKER_LABEL[field] for field in fields]
    colors = {42: "#176B87", 43: "#C75000", 44: "#3A7D44"}
    for row_index, condition in enumerate(("clean", "outage_60s")):
        for col_index, model in enumerate(("B2", "D30")):
            ax = axes[row_index, col_index]
            for seed in (42, 43, 44):
                values = []
                for field in fields:
                    match = next((r for r in seed_rows
                                  if r["model"] == model
                                  and r["condition"] == condition
                                  and r["training_seed"] == seed
                                  and r["field"] == field), None)
                    values.append(match["mean"] if match else np.nan)
                ax.plot(range(len(fields)), values, "o-", linewidth=1.0,
                        color=colors[seed], label=f"seed {seed}")
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xticks(range(len(fields)), labels, rotation=18, ha="right")
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(MODEL_DISPLAY[model])
            if col_index == 0:
                condition_label = "Clean" if condition == "clean" else "60 s outage"
                ax.set_ylabel(f"{condition_label}\nDEF (logit margin)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Faithfulness controls vary across trained checkpoints")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_faithfulness_positive_controls.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_faithfulness_positive_controls.pdf",
                bbox_inches="tight")
    plt.close(fig)


def _plot_overlap(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, model in zip(axes, ("B2", "D30")):
        selected = sorted((r for r in rows if r["model"] == model),
                          key=lambda r: r["k"])
        k = [r["k"] for r in selected]
        ax.plot(k, [r["expected_random_overlap"] for r in selected], "o-",
                color="#A33A3A", label="Expected random overlap")
        ax.plot(k, [r["attention_loo_agreement"] for r in selected], "s--",
                color="#176B87", label="Attention-LOO agreement")
        ax.set_xticks(k)
        ax.set_xlabel("k")
        ax.set_title(MODEL_DISPLAY[model])
        ax.grid(color="#D8D8D8", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Top-k intersection / k")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Top-k overlap limits DEF resolution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_def_overlap_diagnostic.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_def_overlap_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_aggregation(rows: list[dict]) -> None:
    models = ("B2", "D30")
    seeds = (42, 43, 44)
    preferred = [
        "default_mean",
        *(f"first_{head}" for head in
          ("mean", "max", "head_0", "head_1", "head_2", "head_3")),
        *(f"last_{head}" for head in
          ("mean", "max", "head_0", "head_1", "head_2", "head_3")),
        "rollout_mean",
    ]
    available = {row["aggregation"] for row in rows}
    names = [name for name in preferred if name in available]
    columns = [(model, seed) for model in models for seed in seeds]
    matrix = np.full((len(names), len(columns)), np.nan)
    for row in rows:
        key = (row["model"], row["training_seed"])
        if row["aggregation"] in names and key in columns:
            i = names.index(row["aggregation"])
            j = columns.index(key)
            matrix[i, j] = row["mean_stale_attention_shift"]

    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    image = ax.imshow(matrix * 1000, cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-limit * 1000, vcenter=0,
                                       vmax=limit * 1000), aspect="auto")
    column_labels = []
    for model, seed in columns:
        display = MODEL_DISPLAY[model].replace("GAT-Outage", "GAT-\nOutage")
        column_labels.append(f"{display}\nseed {seed}")
    ax.set_xticks(range(len(columns)), column_labels)
    labels = [name.replace("default_mean", "Declared / mean")
              .replace("first", "Layer 1").replace("last", "Layer 2")
              .replace("rollout", "Rollout").replace("head_", "head ")
              .replace("_", " / ") for name in names]
    ax.set_yticks(range(len(names)), labels)
    for i in range(len(names)):
        for j in range(len(columns)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j] * 1000:+.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(matrix[i, j]) > limit * 0.55
                        else "#202020")
    ax.axvline(2.5, color="#202020", linewidth=1.2)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Stale-attention shift x 10^3")
    ax.set_title("Attention response depends on layer, head and checkpoint")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_attention_aggregation_sensitivity.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_attention_aggregation_sensitivity.pdf",
                bbox_inches="tight")
    plt.close(fig)


def _plot_action_rows(seed_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2), sharey="row")
    colors = {"Fixed self row": "#176B87", "Selected request row": "#C75000"}
    field_by_label = {
        "Fixed self row": "self_row_request_def_m",
        "Selected request row": "action_row_request_def_m",
    }
    for row_index, model in enumerate(("B2", "D30")):
        for col_index, condition in enumerate(("clean", "outage_60s")):
            ax = axes[row_index, col_index]
            for seed in (42, 43, 44):
                pair = []
                for label, field in field_by_label.items():
                    match = next(r for r in seed_rows
                                 if r["model"] == model
                                 and r["condition"] == condition
                                 and r["training_seed"] == seed
                                 and r["field"] == field)
                    pair.append(match["mean"])
                positions = np.array([seed - 0.10, seed + 0.10])
                ax.plot(positions, pair, color="#777777", linewidth=0.8, zorder=1)
                for position, value, label in zip(positions, pair, field_by_label):
                    ax.scatter(position, value, color=colors[label], s=28,
                               label=label if seed == 42 else None, zorder=2)
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xticks((42, 43, 44), ("42", "43", "44"))
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title("Clean" if condition == "clean" else "60 s outage")
            if col_index == 0:
                ax.set_ylabel(f"{MODEL_DISPLAY[model]}\nRequest-action DEF")
            if row_index == 1:
                ax.set_xlabel("Training seed")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Query-row choice changes DEF differently by checkpoint")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_action_query_row_sensitivity.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIG_PREFIX}_action_query_row_sensitivity.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global FIG_DIR, FIG_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--fig-prefix", default=FIG_PREFIX)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    FIG_DIR = args.fig_dir
    FIG_PREFIX = args.fig_prefix
    cells = _load_cells(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics = _metric_summary(cells, rng)
    seeds = _seed_summary(cells, [
        *RANKER_LABEL,
        "taxi_attention_loo_spearman",
        "self_row_request_def_m",
        "action_row_request_def_m",
        "action_row_request_gap_m",
    ])
    overlap = _overlap_summary(cells)
    aggregation = _aggregation_summary(cells)
    large_graph = _large_graph_summary(cells, rng)
    action_rows = _action_row_summary(cells, rng)
    payload = {
        "inference_unit": (
            "episode block for within-checkpoint estimates; training seed "
            "for cross-checkpoint synthesis"
        ),
        "n_cells": len(cells),
        "metrics": metrics,
        "per_training_seed": seeds,
        "overlap": overlap,
        "aggregation_sensitivity": aggregation,
        "large_graph_subset": large_graph,
        "action_row_sensitivity": action_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2))
    _write_csv(args.out / "faithfulness_controls.csv", metrics)
    _write_csv(args.out / "per_training_seed.csv", seeds)
    _write_csv(args.out / "overlap.csv", overlap)
    _write_csv(args.out / "aggregation_sensitivity.csv", aggregation)
    _write_csv(args.out / "large_graph_subset.csv", large_graph)
    _write_csv(args.out / "action_row_sensitivity.csv", action_rows)
    plot_cells = {(row["model"], row["condition"], row["ranker"])
                  for row in metrics if row["mean"] is not None}
    expected_plot_cells = {
        (model, condition, ranker)
        for model in ("B2", "D30")
        for condition in ("clean", "outage_60s")
        for ranker in RANKER_LABEL.values()
    }
    if expected_plot_cells <= plot_cells:
        _plot_controls(seeds)
        _plot_overlap(overlap)
        _plot_aggregation(aggregation)
    action_plot_cells = {
        (row["model"], row["condition"], row["label"])
        for row in action_rows
        if row["label"] in ("Fixed self row", "Selected request row")
    }
    expected_action_cells = {
        (model, condition, label)
        for model in ("B2", "D30")
        for condition in ("clean", "outage_60s")
        for label in ("Fixed self row", "Selected request row")
    }
    if expected_action_cells <= action_plot_cells:
        _plot_action_rows(seeds)
    print(f"analysed {len(cells)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
