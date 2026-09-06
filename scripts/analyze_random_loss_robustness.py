"""Summarize the frozen-policy tunnel-versus-random telemetry-loss check."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_hypotheses import (  # noqa: E402
    decisions_frame,
    load_sweep,
    paired_exposed_def_test,
    stale_attention_shift_test,
)


AXES = {
    "outage_duration": "tunnel",
    "dropout_outage_duration": "random",
}


def _condition_rates(cells: list[dict], axis: str) -> tuple[float, float]:
    selected = [cell for cell in cells if cell["cell"]["axis"] == axis]
    rates = np.array(
        [float(cell["empirical_degradation_rate"]) for cell in selected],
        dtype=np.float64,
    )
    exposed = sum(
        int(cell.get("stale_exposure_audit", {}).get(
            "stale_exposed_decisions_seen", 0
        ))
        for cell in selected
    )
    seen = sum(
        int(cell.get("stale_exposure_audit", {}).get("decisions_seen", 0))
        for cell in selected
    )
    return float(rates.mean()), (float(exposed / seen) if seen else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-permutations", type=int, default=20_000)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    args = parser.parse_args()

    sweep_dirs = sorted(args.root.glob("*/*/manifest.json"))
    if not sweep_dirs:
        parser.error(f"no robustness sweeps found below {args.root}")

    rows: list[dict] = []
    for manifest_path in sweep_dirs:
        sweep_dir = manifest_path.parent
        manifest, cells = load_sweep(sweep_dir)
        frame = decisions_frame(cells)
        model = sweep_dir.parent.name
        training_seed = int(sweep_dir.name.removeprefix("seed_"))
        for axis, condition in AXES.items():
            rng = np.random.default_rng(10_000 + training_seed)
            attention = stale_attention_shift_test(
                frame, axis, args.n_permutations, args.n_bootstrap, rng
            )
            paired_def = paired_exposed_def_test(
                frame, axis, "paired_def_delta", args.n_permutations,
                args.n_bootstrap, rng,
            )
            degradation_rate, decision_exposure_rate = _condition_rates(cells, axis)
            rows.append({
                "model": model,
                "training_seed": training_seed,
                "condition": condition,
                "outage_duration_s": float(manifest["cells"][1]["level"]),
                "configured_random_trigger_rate": (
                    float(manifest["cells"][2]["dropout_rate"])
                    if condition == "random" else None
                ),
                "empirical_degradation_rate": degradation_rate,
                "stale_exposed_decision_rate": decision_exposure_rate,
                "stale_attention_shift": attention.get("mean_shift"),
                "stale_attention_ci_low": attention.get("ci95", [None, None])[0],
                "stale_attention_ci_high": attention.get("ci95", [None, None])[1],
                "paired_probability_def_shift": paired_def.get(
                    "mean_degraded_minus_clean"
                ),
                "paired_probability_def_ci_low": paired_def.get(
                    "ci95", [None, None]
                )[0],
                "paired_probability_def_ci_high": paired_def.get(
                    "ci95", [None, None]
                )[1],
                "n_stale_exposed_decisions": attention.get("n_decisions", 0),
                "n_episode_blocks": attention.get("n_episode_blocks", 0),
            })

    comparisons = []
    for model, seed in sorted({(row["model"], row["training_seed"]) for row in rows}):
        pair = {
            row["condition"]: row for row in rows
            if row["model"] == model and row["training_seed"] == seed
        }
        comparisons.append({
            "model": model,
            "training_seed": seed,
            "exposure_rate_ratio_random_over_tunnel": (
                pair["random"]["empirical_degradation_rate"]
                / pair["tunnel"]["empirical_degradation_rate"]
            ),
            "attention_shift_same_direction": (
                np.sign(pair["random"]["stale_attention_shift"])
                == np.sign(pair["tunnel"]["stale_attention_shift"])
            ).item(),
            "probability_def_shift_same_direction": (
                np.sign(pair["random"]["paired_probability_def_shift"])
                == np.sign(pair["tunnel"]["paired_probability_def_shift"])
            ).item(),
            "random_minus_tunnel_attention_shift": (
                pair["random"]["stale_attention_shift"]
                - pair["tunnel"]["stale_attention_shift"]
            ),
            "random_minus_tunnel_probability_def_shift": (
                pair["random"]["paired_probability_def_shift"]
                - pair["tunnel"]["paired_probability_def_shift"]
            ),
        })

    out_base = args.out or args.root / "random_loss_robustness"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": (
            "Frozen checkpoints; 30 s observation-layer freeze; tunnel-entry "
            "versus fixed-rate Bernoulli trigger; held-out demand. Realized "
            "exposure is reported and is not assumed to be matched."
        ),
        "rows": rows,
        "comparisons": comparisons,
        "summary": {
            "n_checkpoints": len(comparisons),
            "attention_direction_agreement": sum(
                row["attention_shift_same_direction"] for row in comparisons
            ),
            "probability_def_direction_agreement": sum(
                row["probability_def_shift_same_direction"] for row in comparisons
            ),
            "median_exposure_rate_ratio_random_over_tunnel": float(np.median([
                row["exposure_rate_ratio_random_over_tunnel"]
                for row in comparisons
            ])),
        },
    }
    out_base.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with out_base.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["summary"], indent=2))
    print(f"json: {out_base.with_suffix('.json')}")
    print(f"csv:  {out_base.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
