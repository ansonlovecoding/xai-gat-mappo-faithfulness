"""Pure helpers for validation-based checkpoint selection."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def select_best_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select by completed journeys, then reward, then the earlier epoch."""
    values = list(candidates)
    if not values:
        raise ValueError("at least one checkpoint candidate is required")
    return max(
        values,
        key=lambda item: (
            float(item["mean_pickups"]),
            float(item["mean_reward"]),
            -int(item["epoch"]),
        ),
    )


def select_trained_candidate(
    candidates: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the earliest diagnostic snapshot and best later checkpoint."""
    values = list(candidates)
    if not values:
        raise ValueError("at least one checkpoint candidate is required")
    initial = min(values, key=lambda item: int(item["epoch"]))
    trained = [
        item for item in values if int(item["epoch"]) > int(initial["epoch"])
    ]
    if not trained:
        raise ValueError("at least one trained checkpoint candidate is required")
    return initial, select_best_candidate(trained)
