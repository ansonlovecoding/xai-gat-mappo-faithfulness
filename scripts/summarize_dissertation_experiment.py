"""Build thesis-ready tables from preflight-passed dissertation sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.provenance import atomic_write_json  # noqa: E402


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _load_cells(sweep_dir: Path) -> list[dict]:
    return [json.loads(path.read_text())
            for path in sorted((sweep_dir / "cells").glob("*.json"))]


def summarize(root: Path) -> list[dict]:
    rows = []
    for sweep_dir in sorted((root / "sweeps").glob("*/seed_*")):
        preflight_path = sweep_dir / "preflight.json"
        if not preflight_path.exists() or not json.loads(preflight_path.read_text()).get("ok"):
            raise RuntimeError(f"sweep has not passed preflight: {sweep_dir}")
        model_id = sweep_dir.parent.name
        training_seed = int(sweep_dir.name.removeprefix("seed_"))
        cells = _load_cells(sweep_dir)
        groups = sorted({(cell["cell"]["axis"], float(cell["cell"]["level"]))
                         for cell in cells})
        for axis, level in groups:
            selected = [cell for cell in cells
                        if cell["cell"]["axis"] == axis
                        and float(cell["cell"]["level"]) == level]
            records = [record for cell in selected
                       for record in cell.get("faith_records", [])]
            scored = [record for record in records
                      if record.get("valid_reservations", 0) > 0]
            exposed = [record for record in records
                       if record.get("n_stale_veh", 0) > 0]
            paired_exposed = [record for record in exposed
                              if "paired_def_delta" in record]
            rows.append({
                "model": model_id,
                "training_seed": training_seed,
                "axis": axis,
                "level_s": level,
                "evaluation_seeds": len(selected),
                "mean_pickups": _mean([float(cell["mean_pickups"]) for cell in selected]),
                "mean_degradation_rate": _mean([
                    float(cell["empirical_degradation_rate"]) for cell in selected
                ]),
                "faithfulness_records": len(records),
                "stale_exposed_records": len(exposed),
                "paired_stale_exposed_records": len(paired_exposed),
                "mean_type_matched_def": _mean([
                    float(record["def"]) for record in scored
                ]),
                "mean_action_protected_def_m": _mean([
                    float(record.get("def_m_excl", float("nan"))) for record in scored
                ]),
                "mean_attention_drift": _mean([
                    float(record.get("drift", float("nan"))) for record in records
                ]),
                "mean_stale_attention_share_when_exposed": _mean([
                    float(record.get("stale_attention_share", float("nan")))
                    for record in exposed
                ]),
                "mean_stale_attention_mass_when_exposed": _mean([
                    float(record.get("stale_attention_mass", float("nan")))
                    for record in exposed
                ]),
                "mean_stale_attention_shift_when_exposed": _mean([
                    float(record.get("stale_attention_shift", float("nan")))
                    for record in exposed
                ]),
                "stale_in_top3_rate_when_exposed": _mean([
                    float(bool(record.get("stale_in_top3"))) for record in exposed
                ]),
                "mean_wamsn_when_exposed_secondary": _mean([
                    float(record.get("wamsn", float("nan"))) for record in exposed
                ]),
                "mean_paired_type_matched_def_delta": _mean([
                    float(record.get("paired_def_delta", float("nan")))
                    for record in paired_exposed
                ]),
                "mean_paired_type_matched_def_m_delta": _mean([
                    float(record.get("paired_def_m_delta", float("nan")))
                    for record in paired_exposed
                ]),
            })
    if not rows:
        raise RuntimeError(f"no completed sweeps found under {root / 'sweeps'}")
    return rows


def summarize_performance(root: Path) -> list[dict]:
    rows = []
    for training_dir in sorted((root / "evaluations").glob("*/seed_*")):
        evaluations = [json.loads(path.read_text())
                       for path in sorted(training_dir.glob("eval_seed_*.json"))]
        if not evaluations:
            continue
        rows.append({
            "model": training_dir.parent.name,
            "training_seed": int(training_dir.name.removeprefix("seed_")),
            "evaluation_seeds": len(evaluations),
            "episodes": sum(int(item["episodes"]) for item in evaluations),
            "mean_pickups": _mean([float(item["mean_pickups"]) for item in evaluations]),
            "mean_reward": _mean([float(item["mean_reward"]) for item in evaluations]),
        })
    if not rows:
        raise RuntimeError(f"no clean evaluations found under {root / 'evaluations'}")
    return rows


def _support_label(supported: int, total: int) -> str:
    if supported == total:
        return "consistent"
    if supported == 0:
        return "not_supported"
    return "mixed"


def summarize_training_seeds(root: Path) -> tuple[list[dict], list[dict]]:
    """Summarise inference across independently trained policy seeds.

    Each per-seed analysis already handles repeated decisions and episode
    clustering. This layer treats the trained policy as the replication unit
    and reports consistency, not a pooled p-value, because three training
    seeds are too few for a strong population-level significance claim.
    """
    aggregate_rows: list[dict] = []
    seed_rows: list[dict] = []
    for model_dir in sorted((root / "sweeps").glob("*")):
        analyses = []
        for sweep_dir in sorted(model_dir.glob("seed_*")):
            analysis_path = sweep_dir / "analysis.json"
            if not analysis_path.exists():
                continue
            analysis = json.loads(analysis_path.read_text())
            robust = analysis["robust"]
            holm = robust["confirmatory_family_holm_p"]
            paired = robust["exposure_conditioned_paired_followup"]
            row = {
                "model": model_dir.name,
                "training_seed": int(sweep_dir.name.removeprefix("seed_")),
                "stale_attention_shift": robust["primary_stale_attention_shift"]["mean_shift"],
                "paired_probability_def_delta": (
                    paired["probability_def"]["mean_degraded_minus_clean"]
                ),
                "paired_probability_def_p_decrease": (
                    paired["probability_def"]["p_one_sided_decrease"]
                ),
                "paired_margin_def_delta": (
                    paired["margin_def"]["mean_degraded_minus_clean"]
                ),
                "H1_rho": robust["H1_robust"]["rho"],
                "H2_delta": analysis["H2_outage_duration"]["mean_delta_faith_minus_perf"],
                "H3_rho": robust["H3_robust"]["rho"],
                "H4_within_episode_rho": robust["H4_within_episode"]["mean_rho_within_episode"],
                **{f"{name}_holm_p": holm[name] for name in ("H1", "H2", "H3", "H4")},
                **{f"{name}_supported": holm[name] <= 0.05
                   for name in ("H1", "H2", "H3", "H4")},
            }
            analyses.append(row)
            seed_rows.append(row)
        if not analyses:
            continue

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in analyses]

        aggregate = {
            "model": model_dir.name,
            "training_seeds": len(analyses),
        }
        for key in ("stale_attention_shift", "paired_probability_def_delta",
                    "paired_margin_def_delta", "H1_rho", "H2_delta", "H3_rho",
                    "H4_within_episode_rho"):
            observed = values(key)
            aggregate[f"{key}_mean"] = _mean(observed)
            aggregate[f"{key}_min"] = min(observed)
            aggregate[f"{key}_max"] = max(observed)
        aggregate["positive_stale_attention_shift_seeds"] = sum(
            row["stale_attention_shift"] > 0 for row in analyses
        )
        aggregate["negative_paired_probability_def_delta_seeds"] = sum(
            row["paired_probability_def_delta"] < 0 for row in analyses
        )
        for hypothesis in ("H1", "H2", "H3", "H4"):
            supported = sum(bool(row[f"{hypothesis}_supported"]) for row in analyses)
            aggregate[f"{hypothesis}_supported_seeds"] = supported
            aggregate[f"{hypothesis}_consistency"] = _support_label(supported, len(analyses))
        aggregate_rows.append(aggregate)

    if not aggregate_rows:
        raise RuntimeError(f"no completed analyses found under {root / 'sweeps'}")
    return aggregate_rows, seed_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        nargs="?", default=PROJECT_ROOT / "runs/dissertation_v4")
    parser.add_argument(
        "--performance-root", type=Path, default=None,
        help="experiment root containing frozen clean evaluations; defaults "
             "to root",
    )
    args = parser.parse_args()
    rows = summarize(args.root)
    performance_rows = summarize_performance(args.performance_root or args.root)
    training_seed_rows, per_seed_rows = summarize_training_seeds(args.root)
    atomic_write_json(args.root / "summary.json", {"rows": rows})
    csv_path = args.root / "summary.csv"
    _write_csv(csv_path, rows)
    atomic_write_json(args.root / "performance_context.json",
                      {"rows": performance_rows})
    performance_csv = args.root / "performance_context.csv"
    _write_csv(performance_csv, performance_rows)
    atomic_write_json(args.root / "training_seed_synthesis.json", {
        "interpretation": (
            "The independently trained policy is the replication unit. "
            "With three seeds, consistency is reported descriptively and no "
            "cross-seed population p-value is claimed."
        ),
        "rows": training_seed_rows,
        "per_seed": per_seed_rows,
    })
    seed_csv = args.root / "training_seed_synthesis.csv"
    _write_csv(seed_csv, training_seed_rows)
    print(f"summary rows: {len(rows)}")
    print(f"json: {args.root / 'summary.json'}")
    print(f"csv:  {csv_path}")
    print(f"performance: {performance_csv}")
    print(f"training seeds: {seed_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
