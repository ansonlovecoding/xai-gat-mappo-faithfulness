import numpy as np

from dispatch_marl.degradation import DegradationConfig, DegradationLayer


def test_tunnel_freeze_uses_last_trusted_snapshot_until_outage_expires() -> None:
    layer = DegradationLayer(
        DegradationConfig(
            mode="tunnel_triggered", corruption="freeze", outage_duration_s=30.0
        ),
        frozenset({"tunnel"}),
        np.random.default_rng(42),
    )

    assert layer.observe("taxi", 1.0, 2.0, 3.0, "road", 0.0) == (
        1.0, 2.0, 3.0, False, 0.0
    )
    assert layer.observe("taxi", 5.0, 6.0, 7.0, "tunnel", 10.0) == (
        1.0, 2.0, 3.0, True, 10.0
    )
    assert layer.observe("taxi", 8.0, 9.0, 10.0, "road", 20.0) == (
        1.0, 2.0, 3.0, True, 20.0
    )
    assert layer.observe("taxi", 11.0, 12.0, 13.0, "road", 30.0) == (
        11.0, 12.0, 13.0, False, 0.0
    )


def test_observation_is_cached_within_a_simulation_step() -> None:
    layer = DegradationLayer(
        DegradationConfig(mode="random_dropout", dropout_rate=0.5),
        frozenset(),
        np.random.default_rng(7),
    )
    first = layer.observe("taxi", 1.0, 2.0, 3.0, "road", 0.0)
    repeated = layer.observe("taxi", 99.0, 99.0, 99.0, "other", 0.0)
    assert repeated == first
