"""Build thesis-ready descriptive tables from preflight-passed v2 sweeps."""
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        nargs="?", default=PROJECT_ROOT / "runs/dissertation_v3")
    args = parser.parse_args()
    rows = summarize(args.root)
    performance_rows = summarize_performance(args.root)
    atomic_write_json(args.root / "summary.json", {"rows": rows})
    csv_path = args.root / "summary.csv"
    _write_csv(csv_path, rows)
    atomic_write_json(args.root / "performance_context.json",
                      {"rows": performance_rows})
    performance_csv = args.root / "performance_context.csv"
    _write_csv(performance_csv, performance_rows)
    print(f"summary rows: {len(rows)}")
    print(f"json: {args.root / 'summary.json'}")
    print(f"csv:  {csv_path}")
    print(f"performance: {performance_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
