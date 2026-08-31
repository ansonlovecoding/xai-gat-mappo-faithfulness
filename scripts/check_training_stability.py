"""Fail fast when a trained policy shows late-run capability collapse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess_training_stability(
    rows: list[dict],
    selected_mean_pickups: float,
    final_mean_pickups: float,
    *,
    final_window: int,
    minimum_final_mean_pickups: float,
    maximum_consecutive_zero_pickups: int,
    minimum_selected_mean_pickups: float,
    minimum_final_validation_retention: float,
) -> dict:
    if not rows:
        raise ValueError("training log is empty")
    pickups = [int(row["pickups"]) for row in rows]
    tail = pickups[-final_window:]
    longest_zero_run = 0
    current_zero_run = 0
    for value in pickups:
        current_zero_run = current_zero_run + 1 if value == 0 else 0
        longest_zero_run = max(longest_zero_run, current_zero_run)
    final_mean = sum(tail) / len(tail)
    validation_retention = (
        final_mean_pickups / selected_mean_pickups
        if selected_mean_pickups > 0 else 0.0
    )
    checks = {
        "final_mean_pickups": final_mean >= minimum_final_mean_pickups,
        "maximum_consecutive_zero_pickups": (
            longest_zero_run <= maximum_consecutive_zero_pickups
        ),
        "selected_validation_pickups": (
            selected_mean_pickups >= minimum_selected_mean_pickups
        ),
        "final_validation_retention": (
            validation_retention >= minimum_final_validation_retention
        ),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "observed": {
            "epochs": len(rows),
            "final_window": len(tail),
            "final_mean_pickups": final_mean,
            "longest_consecutive_zero_pickups": longest_zero_run,
            "selected_validation_mean_pickups": selected_mean_pickups,
            "final_validation_mean_pickups": final_mean_pickups,
            "final_validation_retention": validation_retention,
        },
        "thresholds": {
            "minimum_final_mean_pickups": minimum_final_mean_pickups,
            "maximum_consecutive_zero_pickups": maximum_consecutive_zero_pickups,
            "minimum_selected_mean_pickups": minimum_selected_mean_pickups,
            "minimum_final_validation_retention": minimum_final_validation_retention,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--final-window", type=int, default=10)
    parser.add_argument("--minimum-final-mean-pickups", type=float, default=5.0)
    parser.add_argument("--maximum-consecutive-zero-pickups", type=int, default=4)
    parser.add_argument("--minimum-selected-mean-pickups", type=float, default=5.0)
    parser.add_argument("--minimum-final-validation-retention", type=float, default=0.8)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.run_dir / "train_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selection = json.loads((args.run_dir / "checkpoint_selection.json").read_text())
    final_epoch = max(int(row["epoch"]) for row in rows)
    final_candidates = [
        candidate for candidate in selection["candidates"]
        if int(candidate["epoch"]) == final_epoch
    ]
    if not final_candidates:
        raise SystemExit(
            f"checkpoint selection does not contain final epoch {final_epoch}"
        )
    result = assess_training_stability(
        rows,
        float(selection["selected"]["mean_pickups"]),
        float(final_candidates[0]["mean_pickups"]),
        final_window=args.final_window,
        minimum_final_mean_pickups=args.minimum_final_mean_pickups,
        maximum_consecutive_zero_pickups=args.maximum_consecutive_zero_pickups,
        minimum_selected_mean_pickups=args.minimum_selected_mean_pickups,
        minimum_final_validation_retention=args.minimum_final_validation_retention,
    )
    output = args.run_dir / "training_stability.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"stability: {'OK' if result['ok'] else 'FAILED'} ({output})")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
