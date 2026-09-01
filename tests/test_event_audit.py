import torch

from scripts.eval_policy import _has_stale_vehicle


def _observation(self_aoi: float, taxi_aoi: list[float], mask: list[int]) -> dict:
    return {
        "self": torch.tensor([[0.0, 0.0, 0.0, 0.0, self_aoi]]),
        "neighbor_taxis": torch.tensor([
            [[0.0, 0.0, 0.0, 0.0, value] for value in taxi_aoi]
        ]),
        "neighbor_taxis_mask": torch.tensor([mask]),
    }


def test_event_audit_detects_stale_self_or_visible_neighbor() -> None:
    assert _has_stale_vehicle(_observation(0.2, [0.0, 0.0], [1, 1]))
    assert _has_stale_vehicle(_observation(0.0, [0.0, 0.3], [1, 1]))


def test_event_audit_ignores_stale_padding_slot() -> None:
    assert not _has_stale_vehicle(_observation(0.0, [0.0, 0.3], [1, 0]))
