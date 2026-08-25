"""Pure helpers for validation-based checkpoint selection."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def select_best_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select by validation pickups, then reward, then the earlier epoch."""
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
