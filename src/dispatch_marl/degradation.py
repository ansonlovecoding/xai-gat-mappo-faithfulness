"""Telemetry degradation layer applied at the env→agent observation boundary.

Two modes:

- `tunnel_triggered` (dissertation's primary): each taxi's self-observation
  is corrupted (position noise + a `position_valid=0` flag) whenever the
  taxi's current edge is in the tunnel set. Deterministic w.r.t. network
  geometry — a taxi on a given tunnel edge is *always* degraded.

- `random_dropout`: independent Bernoulli mask per agent per step. Used as
  the matched-rate ablation against `tunnel_triggered` (same overall
  corruption rate, but uncorrelated with location).

- `off`: no-op.

Applied to *observations only* — the policy sees degraded readings, but the
underlying SUMO simulator remains ground truth. That's what lets us measure
"how much did degradation cost the policy" without changing the environment
itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class DegradationConfig:
    mode: Literal["off", "tunnel_triggered", "random_dropout"] = "off"
    # Std dev of Gaussian noise added to (x, y) when degradation is active.
    position_noise_m: float = 20.0
    # For random_dropout mode: per-step probability that an agent is degraded.
    dropout_rate: float = 0.0


class DegradationLayer:
    """Per-step degradation. Consumes fresh randomness each apply().

    Also owns per-agent **Age of Information (AoI)** bookkeeping. Whenever
    a taxi produces a *trusted* reading (i.e. it isn't currently degraded),
    the timestamp is refreshed; when the taxi is degraded, `update_and_get_aoi`
    returns the seconds elapsed since the last trusted reading. Both the
    binary `position_valid` flag and the continuous AoI are surfaced to the
    policy as observation features — that lets the policy condition on
    *how stale* its reading is, not merely on whether it's currently offline.
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
        # Per-taxi timestamp of the last trusted (non-degraded) observation.
        # Reset on env.reset() via `reset()` below.
        self._last_valid_time: dict[str, float] = {}
        # Per (taxi_id, sim_time) cache of the degradation decision. Ensures
        # multiple queries within the same sim step get a consistent answer —
        # critical for `random_dropout` mode where every fresh call would
        # otherwise draw an independent Bernoulli.
        self._step_cache: dict[tuple[str, float], bool] = {}

    def reset(self) -> None:
        """Clear AoI bookkeeping. Call at env.reset()."""
        self._last_valid_time.clear()
        self._step_cache.clear()

    def is_degraded(
        self,
        taxi_id: str,
        current_edge: str,
        sim_time: float,
    ) -> bool:
        """Return whether this taxi is currently degraded.

        Cached per (taxi_id, sim_time) so that faithfulness-time queries and
        obs-build queries agree, and so that random_dropout mode doesn't
        emit different verdicts for the same taxi at the same step.
        """
        key = (taxi_id, sim_time)
        cached = self._step_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_degraded(current_edge)
        self._step_cache[key] = result
        return result

    def _compute_degraded(self, current_edge: str) -> bool:
        mode = self.config.mode
        if mode == "off":
            return False
        if mode == "tunnel_triggered":
            return current_edge in self.tunnel_edges
        if mode == "random_dropout":
            return bool(self.rng.random() < self.config.dropout_rate)
        raise ValueError(f"unknown degradation mode: {mode}")

    def apply_position(self, x: float, y: float, degraded: bool) -> tuple[float, float]:
        """Return possibly-noised (x, y). No-op if not degraded."""
        if not degraded or self.config.position_noise_m <= 0:
            return x, y
        noise = self.rng.normal(0.0, self.config.position_noise_m, size=2)
        return float(x + noise[0]), float(y + noise[1])

    def apply_velocity(self, v: float, degraded: bool) -> float:
        """Return possibly-noised velocity (m/s). No-op if not degraded.

        Velocity noise scale is tied to position noise: a σ_pos over one
        step_length_s window is roughly a σ_v = σ_pos / step_length_s
        velocity uncertainty. We approximate step_length_s = 10 s (the env
        default) and back off if configured otherwise via σ_pos alone.
        """
        if not degraded or self.config.position_noise_m <= 0:
            return v
        sigma_v = self.config.position_noise_m / 10.0
        return float(v + self.rng.normal(0.0, sigma_v))

    def update_and_get_aoi(
        self, taxi_id: str, current_sim_time: float, degraded: bool
    ) -> float:
        """Return AoI in seconds for `taxi_id` at `current_sim_time`.

        On a non-degraded step: reset the taxi's last-valid timestamp and
        return AoI = 0.

        On a degraded step: return `current_sim_time − last_valid_time`.
        If we've never seen this taxi trusted before (first ever observation
        happens to be degraded), record the current time and return 0 to
        avoid an unbounded "AoI = full episode length" spike on step 0.
        """
        if not degraded:
            self._last_valid_time[taxi_id] = current_sim_time
            return 0.0
        last = self._last_valid_time.get(taxi_id)
        if last is None:
            self._last_valid_time[taxi_id] = current_sim_time
            return 0.0
        return max(0.0, current_sim_time - last)

    def refresh_and_get_aoi(
        self,
        taxi_id: str,
        current_edge: str,
        sim_time: float,
    ) -> float:
        """Refresh a taxi's AoI based on its current tunnel/degradation state.

        Unlike `update_and_get_aoi` (which the caller already told whether
        the taxi is degraded), this method decides degradation itself via
        `is_degraded` — so it can be called for *neighbour* taxis that
        aren't the acting agent. Combined with `is_degraded`'s per-step
        cache, this keeps neighbour AoI up to date every sim step, not
        only when the neighbour happens to be the one dispatching.
        """
        degraded = self.is_degraded(taxi_id, current_edge, sim_time)
        return self.update_and_get_aoi(taxi_id, sim_time, degraded)
