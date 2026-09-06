"""Compare full-precision and four-decimal faithfulness analyses."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.provenance import atomic_write_json  # noqa: E402


def _nested(payload: dict, *keys: str):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _direction(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def compare_analyses(full: dict, rounded: dict) -> dict:
    """Return the compact fields needed to audit quantization sensitivity."""
    row: dict = {}
    for label, payload in (("full", full), ("rounded4", rounded)):
        h1 = _nested(payload, "robust", "H1_robust") or {}
        h4 = _nested(payload, "robust", "H4_within_episode") or {}
        paired = _nested(
            payload,
            "robust",
            "exposure_conditioned_paired_followup",
            "probability_def",
        ) or {}
        per_level = _nested(payload, "H1_outage_duration", "per_level_mean") or {}
        holm = _nested(payload, "robust", "confirmatory_family_holm_p") or {}
        row.update({
            f"{label}_clean_def_mean": per_level.get("0"),
            f"{label}_outage60_def_mean": per_level.get("60"),
            f"{label}_h1_rho": h1.get("rho"),
            f"{label}_h1_ci_low": (h1.get("rho_cluster_ci95") or [None, None])[0],
            f"{label}_h1_ci_high": (h1.get("rho_cluster_ci95") or [None, None])[1],
            f"{label}_h1_holm_p": holm.get("H1"),
            f"{label}_paired_def_mean": paired.get("mean_degraded_minus_clean"),
            f"{label}_paired_def_ci_low": (paired.get("ci95") or [None, None])[0],
            f"{label}_paired_def_ci_high": (paired.get("ci95") or [None, None])[1],
            f"{label}_h4_mean_rho": h4.get("mean_rho_within_episode"),
            f"{label}_h4_holm_p": holm.get("H4"),
        })

    for metric in ("h1_rho", "paired_def_mean", "h4_mean_rho"):
        row[f"{metric}_direction_changed"] = (
            _direction(row.get(f"full_{metric}"))
            != _direction(row.get(f"rounded4_{metric}"))
        )
    for hypothesis in ("H1", "H2", "H3", "H4"):
        full_p = _nested(full, "robust", "confirmatory_family_holm_p", hypothesis)
        rounded_p = _nested(
            rounded, "robust", "confirmatory_family_holm_p", hypothesis
        )
        row[f"{hypothesis.lower()}_verdict_changed"] = (
            full_p is not None
            and rounded_p is not None
            and (full_p < 0.05) != (rounded_p < 0.05)
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    rows = []
    for full_path in sorted((args.root / "sweeps").glob("*/seed_*/analysis.json")):
        rounded_path = full_path.with_name("analysis_rounded4.json")
        if not rounded_path.exists():
            raise SystemExit(f"missing rounded analysis: {rounded_path}")
        row = {
            "model": full_path.parent.parent.name,
            "training_seed": int(full_path.parent.name.removeprefix("seed_")),
            **compare_analyses(
                json.loads(full_path.read_text()),
                json.loads(rounded_path.read_text()),
            ),
        }
        rows.append(row)
    if not rows:
        raise SystemExit(f"no checkpoint analyses found under {args.root / 'sweeps'}")

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "precision_sensitivity.json", {"rows": rows})
    with (output_dir / "precision_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"precision sensitivity: {len(rows)} checkpoints")
    print(f"json: {output_dir / 'precision_sensitivity.json'}")
    print(f"csv:  {output_dir / 'precision_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
