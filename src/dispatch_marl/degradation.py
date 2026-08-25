"""Telemetry degradation layer applied at the env→agent observation boundary.

Semantics (proposal §7.2): when a vehicle's signal drops, it *stops
transmitting* — its telemetry **freezes at the last valid reading** and every
observer (the vehicle itself and all peers) acts on that last-known state
while its Age of Information grows. This is the "freeze" corruption mode and
the dissertation's primary mechanism. A legacy "noise" mode (Gaussian jitter
on the true state) is kept for backward comparison only.

Trigger modes:

- `tunnel_triggered` (primary): tunnel entry emits one trigger event.
  Deterministic w.r.t. network geometry.
- `random_dropout`: independent Bernoulli trigger per taxi per step —
  the matched-rate ablation with no spatial structure.
- `off`: no-op.

Severity is the observation-layer outage window, `outage_duration_s`. Once
tunnel entry triggers an outage, telemetry remains frozen for that declared
window even if the taxi leaves the tunnel. Remaining inside the tunnel does
not continuously retrigger the outage. The experiment uses {10, 20, 30, 60}
seconds to match the environment's 10-second observation interval. Observed
AoI is reported as a description of the resulting stale data, not as a causal
dose.

Applied to *observations only* — the policy sees stale readings, but the
underlying SUMO simulator remains ground truth. That's what lets us measure
"how much did degradation cost the policy" without changing the environment,
and what makes the per-decision clean twin (drift measurement) exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class DegradationConfig:
    mode: Literal["off", "tunnel_triggered", "random_dropout"] = "off"
    # How degraded telemetry is corrupted:
    #   "freeze" (default; proposal §7.2): serve the last valid
    #            (position, velocity) snapshot to ALL observers.
    #   "noise":  legacy Gaussian jitter on the true state (pre-freeze
    #             results in results/b1b2b3_sumo120_seed42_v1 used this).
    corruption: Literal["freeze", "noise"] = "freeze"
    # Severity knob: duration of the observation-layer outage after a trigger.
    # 0 = only the trigger observation is degraded.
    outage_duration_s: float = 0.0
    # Std dev of Gaussian noise added to (x, y) in "noise" mode.
    position_noise_m: float = 20.0
    # For random_dropout mode: per-step probability that the trigger fires.
    dropout_rate: float = 0.0


class ObservedState(tuple):
    """(x, y, v, degraded, aoi) — the telemetry every observer sees."""
    __slots__ = ()


class DegradationLayer:
    """Per-step degradation with per-taxi freeze snapshots and AoI.

    `observe()` is the single entry point: the env calls it once per alive
    taxi per step, and the returned reading is what *everyone* sees for
    that taxi — the taxi itself (self node) and its peers (neighbour
    nodes). Results are cached per (taxi, sim_time) so repeated queries
    within a step are consistent (critical for random_dropout, and for
    faithfulness-time re-queries).
    """

    def __init__(
        self,
        config: DegradationConfig,
        tunnel_edges: frozenset[str],
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.tunnel_edges = tunnel_edges
        self.rng = rng
        # Per-taxi timestamp of the last trusted reading.
        self._last_valid_time: dict[str, float] = {}
        # Per-taxi (x, y, v) snapshot at the last trusted reading.
        self._snapshot: dict[str, tuple[float, float, float]] = {}
        # Per-taxi sim-time until which the observation-layer outage lasts.
        self._outage_until: dict[str, float] = {}
        # Previous tunnel-membership state, used to detect entry events.
        self._tunnel_active: dict[str, bool] = {}
        # Per (taxi, sim_time) cache of the full observed reading.
        self._obs_cache: dict[tuple[str, float], tuple] = {}

    def reset(self) -> None:
        """Clear all bookkeeping. Call at env.reset()."""
        self._last_valid_time.clear()
        self._snapshot.clear()
        self._outage_until.clear()
        self._tunnel_active.clear()
        self._obs_cache.clear()

    # ------------------------------------------------------------- observe

    def observe(
        self,
        taxi_id: str,
        x_true: float,
        y_true: float,
        v_true: float,
        current_edge: str,
        sim_time: float,
    ) -> tuple[float, float, float, bool, float]:
        """Return (x, y, v, degraded, aoi) — the reading ALL observers get.

        Not degraded → the true state, AoI 0, snapshot refreshed.
        Degraded    → the frozen snapshot ("freeze") or noised true state
                      ("noise" legacy), with AoI = time since last trusted
                      reading.
        First-ever reading while degraded → treated as trusted (grace), to
        avoid an artificial episode-length AoI spike at spawn.
        """
        key = (taxi_id, sim_time)
        cached = self._obs_cache.get(key)
        if cached is not None:
            return cached

        trigger_active = self._compute_trigger(current_edge)
        last_valid = self._last_valid_time.get(taxi_id)

        if self.config.mode == "tunnel_triggered":
            was_active = self._tunnel_active.get(taxi_id, False)
            trigger = trigger_active and not was_active
            # A first reading receives grace. Keeping the previous state false
            # lets a taxi spawned inside a tunnel trigger on its next reading.
            self._tunnel_active[taxi_id] = trigger_active if last_valid is not None else False
        else:
            trigger = trigger_active

        if last_valid is None:
            # Grace: never seen trusted; adopt the current reading.
            degraded = False
        else:
            if trigger:
                # A trigger starts or extends a fixed observation-layer outage.
                self._outage_until[taxi_id] = max(
                    self._outage_until.get(taxi_id, 0.0),
                    sim_time + self.config.outage_duration_s,
                )
            degraded = trigger or sim_time < self._outage_until.get(taxi_id, 0.0)

        if not degraded:
            self._last_valid_time[taxi_id] = sim_time
            self._snapshot[taxi_id] = (x_true, y_true, v_true)
            result = (x_true, y_true, v_true, False, 0.0)
        else:
            aoi = max(0.0, sim_time - last_valid)
            if self.config.corruption == "freeze":
                x, y, v = self._snapshot[taxi_id]
            else:  # legacy noise
                x, y = self._apply_position_noise(x_true, y_true)
                v = self._apply_velocity_noise(v_true)
            result = (float(x), float(y), float(v), True, aoi)

        self._obs_cache[key] = result
        return result

    # ------------------------------------------------------------ internals

    def _compute_trigger(self, current_edge: str) -> bool:
        mode = self.config.mode
        if mode == "off":
            return False
        if mode == "tunnel_triggered":
            return current_edge in self.tunnel_edges
        if mode == "random_dropout":
            return bool(self.rng.random() < self.config.dropout_rate)
        raise ValueError(f"unknown degradation mode: {mode}")

    def _apply_position_noise(self, x: float, y: float) -> tuple[float, float]:
        if self.config.position_noise_m <= 0:
            return x, y
        noise = self.rng.normal(0.0, self.config.position_noise_m, size=2)
        return float(x + noise[0]), float(y + noise[1])

    def _apply_velocity_noise(self, v: float) -> float:
        if self.config.position_noise_m <= 0:
            return v
        # σ_v tied to σ_pos over one 10 s env step.
        return float(v + self.rng.normal(0.0, self.config.position_noise_m / 10.0))

    # ------------------------------------------------- back-compat helpers

    def is_degraded(self, taxi_id: str, current_edge: str, sim_time: float) -> bool:
        """Degradation verdict only (uses the same per-step cache).

        NOTE: requires the true state for snapshot upkeep, so this helper
        is only safe AFTER observe() was called for this (taxi, step) —
        which the env's obs build guarantees.
        """
        cached = self._obs_cache.get((taxi_id, sim_time))
        if cached is not None:
            return bool(cached[3])
        # Fallback: trigger-only verdict (no snapshot upkeep).
        return self._compute_trigger(current_edge)
