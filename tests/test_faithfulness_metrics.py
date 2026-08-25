import numpy as np

from dispatch_marl.faithfulness import (
    compute_stale_attention_mass,
    compute_stale_attention_share,
    compute_wamsn,
)


def test_binary_stale_attention_share_ignores_aoi_magnitude() -> None:
    attention = np.array([0.2, 0.3, 0.1, 0.4])
    vehicles = np.array([True, True, True, False])
    stale = np.array([False, True, False, False])

    assert compute_stale_attention_share(attention, stale, vehicles) == 0.5
    assert compute_stale_attention_mass(attention, stale) == 0.3
    assert compute_wamsn(attention, np.array([0.0, 5.0, 0.0, 0.0]), vehicles) \
        < compute_wamsn(attention, np.array([0.0, 60.0, 0.0, 0.0]), vehicles)


def test_stale_attention_share_is_zero_without_vehicle_attention() -> None:
    assert compute_stale_attention_share(
        np.array([0.0, 0.0, 1.0]),
        np.array([True, False, False]),
        np.array([True, True, False]),
    ) == 0.0
