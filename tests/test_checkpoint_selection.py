import pytest

from dispatch_marl.checkpoint_selection import (
    select_best_candidate,
    select_trained_candidate,
)


def test_checkpoint_selection_uses_declared_tie_breakers() -> None:
    candidates = [
        {"epoch": 30, "mean_pickups": 4.0, "mean_reward": 10.0},
        {"epoch": 20, "mean_pickups": 5.0, "mean_reward": 8.0},
        {"epoch": 10, "mean_pickups": 5.0, "mean_reward": 8.0},
    ]
    assert select_best_candidate(candidates)["epoch"] == 10


def test_checkpoint_selection_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_best_candidate([])


def test_checkpoint_selection_keeps_epoch_zero_as_diagnostic_only() -> None:
    candidates = [
        {"epoch": 0, "mean_pickups": 20.0, "mean_reward": 100.0},
        {"epoch": 10, "mean_pickups": 12.0, "mean_reward": 80.0},
        {"epoch": 20, "mean_pickups": 13.0, "mean_reward": 90.0},
    ]

    initial, selected = select_trained_candidate(candidates)

    assert initial["epoch"] == 0
    assert selected["epoch"] == 20
