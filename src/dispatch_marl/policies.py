"""Baseline dispatch policies.

Every policy implements the same tiny interface:

    class Policy(Protocol):
        def act(obs: dict[agent, obs_dict]) -> dict[agent, int]: ...
        def reset() -> None: ...

`obs` is the dict returned by `DispatchEnv.step`. Actions are `int` in
`{0, 1, ..., K}` where 0 = no-op and 1..K = accept the k-th nearest visible
reservation (its ordering matches the `reservations` array in the obs).

These baselines are what the GAT-MAPPO policy has to beat in the results
table. In particular:

  - RandomPolicy: lower bound. Uniform over all actions including no-op.
  - NearestReservationPolicy: naive greedy heuristic — always take the
    nearest visible reservation. Conflict resolution is left to the env
    (first-taxi-wins per step).
"""
from __future__ import annotations

import random
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, obs: dict[str, dict[str, np.ndarray]]) -> dict[str, int]: ...
    def reset(self) -> None: ...


class NoOpPolicy:
    """Every agent stays idle. Useful for stress-testing the sim without any dispatch."""

    def act(self, obs):
        return {a: 0 for a in obs}

    def reset(self):
        pass


class RandomPolicy:
    """Uniform-random over Discrete(K+1). Deterministic given seed."""

    def __init__(self, k_reservations: int, seed: int = 42):
        self._n_actions = k_reservations + 1
        self._rng = random.Random(seed)

    def act(self, obs):
        return {a: self._rng.randrange(self._n_actions) for a in obs}

    def reset(self):
        pass


class NearestReservationPolicy:
    """Greedy: each idle taxi accepts its nearest visible pending reservation.

    Because the env sorts reservations by pickup distance when building each
    taxi's observation, "action index 1" already means "closest reservation".
    Two taxis picking the same reservation is resolved by the env
    (first-in-actions-dict wins), so this policy will lose a few dispatches
    to conflicts — that's the naive-greedy behaviour, on purpose.
    """

    def act(self, obs):
        actions = {}
        for agent, o in obs.items():
            mask = o["reservations_mask"]
            valid = np.nonzero(mask)[0]
            if len(valid) > 0:
                # action = 1 corresponds to the nearest reservation.
                actions[agent] = int(valid[0]) + 1
            else:
                actions[agent] = 0
        return actions

    def reset(self):
        pass
