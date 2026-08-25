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
        1.0, 2.0, 3.0, True, 30.0
    )
    assert layer.observe("taxi", 14.0, 15.0, 16.0, "road", 40.0) == (
        14.0, 15.0, 16.0, False, 0.0
    )


def test_tunnel_membership_does_not_continuously_retrigger() -> None:
    layer = DegradationLayer(
        DegradationConfig(
            mode="tunnel_triggered", corruption="freeze", outage_duration_s=10.0
        ),
        frozenset({"tunnel"}),
        np.random.default_rng(42),
    )

    layer.observe("taxi", 1.0, 2.0, 3.0, "road", 0.0)
    assert layer.observe("taxi", 4.0, 5.0, 6.0, "tunnel", 10.0)[3:] == (True, 10.0)
    assert layer.observe("taxi", 7.0, 8.0, 9.0, "tunnel", 20.0) == (
        7.0, 8.0, 9.0, False, 0.0
    )
    assert layer.observe("taxi", 10.0, 11.0, 12.0, "road", 30.0)[3] is False
    assert layer.observe("taxi", 13.0, 14.0, 15.0, "tunnel", 40.0)[3:] == (
        True, 10.0
    )


def test_outage_duration_changes_observed_staleness() -> None:
    def observed_aoi(duration: float) -> list[float]:
        layer = DegradationLayer(
            DegradationConfig(mode="tunnel_triggered", outage_duration_s=duration),
            frozenset({"tunnel"}),
            np.random.default_rng(42),
        )
        layer.observe("taxi", 0.0, 0.0, 0.0, "road", 0.0)
        return [
            layer.observe("taxi", t, 0.0, 0.0, "tunnel", t)[4]
            for t in (10.0, 20.0, 30.0, 40.0)
        ]

    assert observed_aoi(10.0) == [10.0, 0.0, 0.0, 0.0]
    assert observed_aoi(30.0) == [10.0, 20.0, 30.0, 0.0]


def test_observation_is_cached_within_a_simulation_step() -> None:
    layer = DegradationLayer(
        DegradationConfig(mode="random_dropout", dropout_rate=0.5),
        frozenset(),
        np.random.default_rng(7),
    )
    first = layer.observe("taxi", 1.0, 2.0, 3.0, "road", 0.0)
    repeated = layer.observe("taxi", 99.0, 99.0, 99.0, "other", 0.0)
    assert repeated == first
