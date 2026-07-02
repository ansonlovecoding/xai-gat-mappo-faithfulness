"""PettingZoo ParallelEnv wrapping SUMO taxi-dispatch via TraCI.

Per-taxi action: Discrete(K+1)
  - 0 = no-op (idle taxi keeps randomCircling)
  - 1..K = accept the k-th nearest pending reservation

Per-taxi observation (Dict):
  - self:               [x_norm, y_norm, active_time_norm]
  - neighbor_taxis:     (K, 4)   [dx, dy, is_empty, dist_norm]
  - neighbor_taxis_mask:(K,)     1 = valid
  - reservations:       (K, 5)   [dx_pu, dy_pu, dx_do, dy_do, wait_norm]
  - reservations_mask:  (K,)     1 = valid
  - position_valid:     (1,)     1 = trustworthy, 0 = degraded

Reward is a team scalar broadcast to every agent:
  R = pickups_this_step − λ · mean_current_wait_time

Episode ends when SUMO reaches the sumocfg <time><end/> value.

Only ONE DispatchEnv per Python process — TraCI is single-connection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ._sumo import traci  # libsumo if available, else traci
import sumolib  # noqa: E402

from .degradation import DegradationConfig, DegradationLayer
from .scenario import Scenario, load_scenario


# Observation feature widths — kept in module-level constants so the GAT
# encoder later can import the same numbers.
#
# Self feature layout:
#   [x_norm, y_norm, episode_time_norm, velocity_norm, aoi_norm]
# where velocity_norm is speed / 30 m/s (~108 km/h) clipped to [0, 1], and
# aoi_norm is Age of Information / AOI_MAX_S clipped to [0, 1].
SELF_FEAT_DIM = 5
TAXI_FEAT_DIM = 4
RES_FEAT_DIM = 5

# Normalisation constants used by the observation builder. Set explicitly here
# so the policy and any future faithfulness code can import them.
VELOCITY_MAX_MS = 30.0
AOI_MAX_S = 300.0


@dataclass
class DispatchEnvConfig:
    area: str = "central_park"
    n_taxis: int = 20
    k_neighbors: int = 5
    k_reservations: int = 5
    step_length_s: int = 10  # SUMO seconds advanced per RL step
    # Reward = pickup_reward * pickups_this_step
    #        + dispatch_reward * successful_dispatches_this_step
    #        - wait_penalty_lambda * mean_pending_wait_time_s
    # Defaults intentionally give large weight to completed pickups so the
    # policy can't degenerate into "do nothing to avoid the wait penalty".
    pickup_reward: float = 10.0
    dispatch_reward: float = 0.5
    wait_penalty_lambda: float = 0.001
    seed: int = 42
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    # For debugging / smoke tests. Never set True in a training run.
    use_gui: bool = False


class DispatchEnv(ParallelEnv):
    metadata = {"name": "dispatch_marl_v0", "is_parallelizable": False}

    def __init__(self, config: DispatchEnvConfig | None = None):
        super().__init__()
        self.config = config or DispatchEnvConfig()
        self.scenario: Scenario = load_scenario(self.config.area)
        self.possible_agents: list[str] = [
            f"taxi_{i}" for i in range(self.config.n_taxis)
        ]
        self.agents: list[str] = []
        # Per-agent cache of the reservation-ID ordering we showed at obs time,
        # so an action index in {1..K} can be translated back to a concrete
        # reservation ID at step time.
        self._pending_res_map: dict[str, list[str]] = {}
        # Bookkeeping.
        self._sumo_started = False
        self._sumo_label: str | None = None  # unique TraCI connection label
        self._label_counter = 0
        self._sim_time = 0.0
        self._end_time = 0.0
        self._net_bbox = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self._net_diag = 1.0
        self._rng = np.random.default_rng(self.config.seed)
        self._degradation = DegradationLayer(
            self.config.degradation, self.scenario.tunnel_edges, self._rng
        )
        # Pre-build gymnasium spaces so PettingZoo can query them.
        self._obs_space = self._make_obs_space()
        self._act_space = spaces.Discrete(self.config.k_reservations + 1)

    # ------------------------------------------------------------------ spaces

    def _make_obs_space(self) -> spaces.Dict:
        K_n = self.config.k_neighbors
        K_r = self.config.k_reservations
        return spaces.Dict({
            "self": spaces.Box(-np.inf, np.inf, shape=(SELF_FEAT_DIM,), dtype=np.float32),
            "neighbor_taxis": spaces.Box(-np.inf, np.inf, shape=(K_n, TAXI_FEAT_DIM), dtype=np.float32),
            "neighbor_taxis_mask": spaces.Box(0, 1, shape=(K_n,), dtype=np.int8),
            "reservations": spaces.Box(-np.inf, np.inf, shape=(K_r, RES_FEAT_DIM), dtype=np.float32),
            "reservations_mask": spaces.Box(0, 1, shape=(K_r,), dtype=np.int8),
            "position_valid": spaces.Box(0, 1, shape=(1,), dtype=np.int8),
        })

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.config = DispatchEnvConfig(**{**self.config.__dict__, "seed": seed})
            self._rng = np.random.default_rng(seed)
            self._degradation = DegradationLayer(
                self.config.degradation, self.scenario.tunnel_edges, self._rng
            )
        # Always clear per-agent AoI state; a fresh episode starts every taxi
        # with no history of trusted readings.
        self._degradation.reset()

        # Prefer traci.load() to restart the same SUMO subprocess between
        # episodes — avoids process-startup overhead AND avoids the "label
        # already active" registry collision that plagues close+start.
        load_args = [
            "-c", str(self.scenario.sumocfg),
            "--no-step-log",
            "--no-warnings",
        ]
        reloaded = False
        if self._sumo_started and self._sumo_label is not None:
            try:
                traci.switch(self._sumo_label)
                traci.load(load_args)
                reloaded = True
            except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
                try:
                    traci.close()
                except (traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
                    pass
                self._sumo_started = False

        if not reloaded:
            # Fresh SUMO subprocess with a UNIQUE label so we can never
            # collide with a stale "default" entry left by a prior fatal.
            self._label_counter += 1
            self._sumo_label = f"env_{id(self):x}_{self._label_counter}"
            binary = sumolib.checkBinary("sumo-gui" if self.config.use_gui else "sumo")
            traci.start([binary, *load_args], label=self._sumo_label)
            traci.switch(self._sumo_label)
            self._sumo_started = True

        # Cache global bounds for observation normalisation.
        # getNetBoundary returns ((xmin, ymin), (xmax, ymax)).
        (xmin, ymin), (xmax, ymax) = traci.simulation.getNetBoundary()
        self._net_bbox = (xmin, ymin, xmax, ymax)
        self._net_diag = max(1.0, float(np.hypot(xmax - xmin, ymax - ymin)))
        self._end_time = float(traci.simulation.getEndTime())

        # Step until at least one taxi is alive (they spawn at t=0 but SUMO
        # inserts after the first step).
        for _ in range(2):
            traci.simulationStep()
        self._sim_time = traci.simulation.getTime()

        self.agents = self._active_idle_taxi_ids()
        obs = self._build_all_obs(self.agents)
        infos = {a: {} for a in self.agents}
        return obs, infos

    # ------------------------------------------------------------------ step

    def step(self, actions: dict[str, int]):
        try:
            return self._step_impl(actions)
        except traci.exceptions.FatalTraCIError:
            # SUMO died. Force clean episode termination so the RL loop
            # doesn't need to unwind an exception.
            self._sim_time = self._end_time
            self._sumo_started = False
            return {}, {a: 0.0 for a in actions}, {a: True for a in actions}, \
                {a: False for a in actions}, {a: {"sumo_died": True} for a in actions}

    def _step_impl(self, actions: dict[str, int]):
        # 1. Apply dispatch actions before advancing the sim. Cross-check each
        #    (agent, reservation) against SUMO's *current* state before calling
        #    dispatchTaxi — the reservation must still exist, and the taxi must
        #    still be idle. Random-init policies often violate both, and while
        #    a mismatch used to be handled by SUMO with a soft error, in
        #    recent versions it can close the TraCI connection outright.
        try:
            live_idle = set(traci.vehicle.getTaxiFleet(0))
            live_res_ids = {r.id for r in traci.person.getTaxiReservations(0)}
        except traci.exceptions.TraCIException:
            live_idle = set()
            live_res_ids = set()

        pickups_dispatched: list[str] = []  # reservation IDs we assigned
        for agent, action in actions.items():
            if action == 0 or agent not in live_idle:
                continue
            candidates = self._pending_res_map.get(agent, [])
            k_idx = int(action) - 1
            if k_idx < 0 or k_idx >= len(candidates):
                continue
            res_id = candidates[k_idx]
            if res_id in pickups_dispatched or res_id not in live_res_ids:
                continue
            try:
                traci.vehicle.dispatchTaxi(agent, [res_id])
                pickups_dispatched.append(res_id)
            except traci.exceptions.TraCIException:
                continue

        # 2. Advance the simulation by step_length_s SUMO seconds. Accumulate
        #    delivered riders (persons that arrived at destination) as we go.
        pickups_delta = 0
        target = self._sim_time + self.config.step_length_s
        while self._sim_time < target and self._sim_time < self._end_time:
            traci.simulationStep()
            pickups_delta += len(traci.simulation.getArrivedPersonIDList())
            self._sim_time = traci.simulation.getTime()

        # 3. Build reward.
        n_pickups_delta = pickups_delta
        n_dispatches_ok = len(pickups_dispatched)
        mean_wait = self._mean_pending_wait_time()
        team_r = (
            self.config.pickup_reward * float(n_pickups_delta)
            + self.config.dispatch_reward * float(n_dispatches_ok)
            - self.config.wait_penalty_lambda * mean_wait
        )

        # 4. Decide termination and next agent set.
        terminated_all = self._sim_time >= self._end_time
        self.agents = [] if terminated_all else self._active_idle_taxi_ids()

        obs = self._build_all_obs(self.agents)
        rewards = {a: team_r for a in actions}
        # For dead / newly-inactive agents that had actions this step, still
        # report the (final) reward with a terminated=True flag.
        terminations = {a: terminated_all for a in actions}
        truncations = {a: False for a in actions}
        infos = {
            a: {
                "pickups_delta": n_pickups_delta,
                "dispatches_ok": n_dispatches_ok,
                "mean_wait_time": mean_wait,
                "sim_time": self._sim_time,
            }
            for a in actions
        }
        return obs, rewards, terminations, truncations, infos

    @property
    def done(self) -> bool:
        """True when the episode can no longer be stepped meaningfully.

        Two paths get us here: (a) SUMO reached its configured end_time,
        or (b) SUMO died mid-episode (`_sumo_started` was flipped to False
        by the fatal-error handler). Either way, the training loop should
        stop stepping and move on to the next epoch.
        """
        if not self._sumo_started:
            return True
        return self._sim_time >= self._end_time

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def close(self) -> None:
        if self._sumo_started:
            try:
                if self._sumo_label is not None:
                    traci.switch(self._sumo_label)
                traci.close()
            except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
                pass
            self._sumo_started = False
            self._sumo_label = None

    # ------------------------------------------------------------------ helpers

    def _active_idle_taxi_ids(self) -> list[str]:
        """Return only taxis that are (a) alive in the sim and (b) idle."""
        try:
            alive = set(traci.vehicle.getIDList())
            idle = set(traci.vehicle.getTaxiFleet(0))
        except traci.exceptions.TraCIException:
            return []
        return sorted(
            a for a in self.possible_agents if a in alive and a in idle
        )

    def _mean_pending_wait_time(self) -> float:
        reservations = traci.person.getTaxiReservations(0)
        if not reservations:
            return 0.0
        now = self._sim_time
        waits = [max(0.0, now - r.reservationTime) for r in reservations]
        return float(np.mean(waits))

    # -------------------------------------------------------------- observations

    def _build_all_obs(self, agents: list[str]) -> dict[str, dict[str, np.ndarray]]:
        if not agents:
            self._pending_res_map = {}
            return {}
        # Cache expensive per-step queries.
        try:
            all_taxi_positions = {
                a: traci.vehicle.getPosition(a) for a in self.possible_agents
                if a in traci.vehicle.getIDList()
            }
            empty_ids = set(traci.vehicle.getTaxiFleet(0))
            reservations = list(traci.person.getTaxiReservations(0))
        except traci.exceptions.TraCIException:
            return {a: self._empty_obs() for a in agents}

        obs = {}
        for agent in agents:
            obs[agent] = self._build_agent_obs(
                agent, all_taxi_positions, empty_ids, reservations
            )
        return obs

    def _empty_obs(self) -> dict[str, np.ndarray]:
        K_n = self.config.k_neighbors
        K_r = self.config.k_reservations
        return {
            "self": np.zeros(SELF_FEAT_DIM, dtype=np.float32),
            "neighbor_taxis": np.zeros((K_n, TAXI_FEAT_DIM), dtype=np.float32),
            "neighbor_taxis_mask": np.zeros(K_n, dtype=np.int8),
            "reservations": np.zeros((K_r, RES_FEAT_DIM), dtype=np.float32),
            "reservations_mask": np.zeros(K_r, dtype=np.int8),
            "position_valid": np.zeros(1, dtype=np.int8),
        }

    def _build_agent_obs(
        self,
        agent: str,
        all_taxi_positions: dict[str, tuple[float, float]],
        empty_ids: set[str],
        reservations: list[Any],
    ) -> dict[str, np.ndarray]:
        if agent not in all_taxi_positions:
            return self._empty_obs()

        # ------------- self -------------
        x_true, y_true = all_taxi_positions[agent]
        try:
            v_true = float(traci.vehicle.getSpeed(agent))
        except traci.exceptions.TraCIException:
            v_true = 0.0
        current_edge = traci.vehicle.getRoadID(agent)
        degraded = self._degradation.is_degraded(current_edge)
        x, y = self._degradation.apply_position(x_true, y_true, degraded)
        v = self._degradation.apply_velocity(v_true, degraded)
        aoi = self._degradation.update_and_get_aoi(agent, self._sim_time, degraded)
        self_feat = np.array([
            self._norm_x(x),
            self._norm_y(y),
            min(1.0, self._sim_time / max(1.0, self._end_time)),
            min(1.0, max(0.0, v) / VELOCITY_MAX_MS),
            min(1.0, aoi / AOI_MAX_S),
        ], dtype=np.float32)

        # ------------- neighbouring taxis -------------
        K_n = self.config.k_neighbors
        others = [(other, ox, oy) for other, (ox, oy) in all_taxi_positions.items() if other != agent]
        others.sort(key=lambda t: (t[1] - x_true) ** 2 + (t[2] - y_true) ** 2)
        taxi_feat = np.zeros((K_n, TAXI_FEAT_DIM), dtype=np.float32)
        taxi_mask = np.zeros(K_n, dtype=np.int8)
        for i, (other, ox, oy) in enumerate(others[:K_n]):
            dx, dy = ox - x_true, oy - y_true
            dist = float(np.hypot(dx, dy))
            taxi_feat[i] = [dx / self._net_diag, dy / self._net_diag,
                            1.0 if other in empty_ids else 0.0,
                            dist / self._net_diag]
            taxi_mask[i] = 1

        # ------------- pending reservations -------------
        K_r = self.config.k_reservations
        # Compute pickup coords for each reservation, sort by distance to us.
        res_with_coords = []
        for r in reservations:
            try:
                px, py = traci.simulation.convert2D(r.fromEdge, 0.0, laneIndex=0)
                dx_pu, dy_pu = px - x_true, py - y_true
                dist = float(np.hypot(dx_pu, dy_pu))
                res_with_coords.append((r, px, py, dist))
            except traci.exceptions.TraCIException:
                continue
        res_with_coords.sort(key=lambda t: t[3])

        res_feat = np.zeros((K_r, RES_FEAT_DIM), dtype=np.float32)
        res_mask = np.zeros(K_r, dtype=np.int8)
        res_ids: list[str] = []
        for i, (r, px, py, _) in enumerate(res_with_coords[:K_r]):
            try:
                qx, qy = traci.simulation.convert2D(r.toEdge, 0.0, laneIndex=0)
            except traci.exceptions.TraCIException:
                qx, qy = px, py
            wait = max(0.0, self._sim_time - r.reservationTime)
            res_feat[i] = [
                (px - x_true) / self._net_diag,
                (py - y_true) / self._net_diag,
                (qx - x_true) / self._net_diag,
                (qy - y_true) / self._net_diag,
                min(1.0, wait / 600.0),
            ]
            res_mask[i] = 1
            res_ids.append(r.id)
        self._pending_res_map[agent] = res_ids

        return {
            "self": self_feat,
            "neighbor_taxis": taxi_feat,
            "neighbor_taxis_mask": taxi_mask,
            "reservations": res_feat,
            "reservations_mask": res_mask,
            "position_valid": np.array([0 if degraded else 1], dtype=np.int8),
        }

    def _norm_x(self, x: float) -> float:
        return (x - self._net_bbox[0]) / max(1.0, self._net_bbox[2] - self._net_bbox[0])

    def _norm_y(self, y: float) -> float:
        return (y - self._net_bbox[1]) / max(1.0, self._net_bbox[3] - self._net_bbox[1])
