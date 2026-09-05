from __future__ import annotations

from dispatch_marl import DispatchEnv, DispatchEnvConfig
from dispatch_marl.env import traci


def test_global_completed_journey_survives_empty_action_step(monkeypatch) -> None:
    env = object.__new__(DispatchEnv)
    env.config = DispatchEnvConfig(step_length_s=10)
    env._sim_time = 0.0
    env._end_time = 10.0
    env._pending_res_map = {}
    env.agents = []

    clock = {"time": 0.0}

    def simulation_step() -> None:
        clock["time"] = 10.0

    monkeypatch.setattr(traci.vehicle, "getTaxiFleet", lambda _: [])
    monkeypatch.setattr(traci.person, "getTaxiReservations", lambda _: [])
    monkeypatch.setattr(traci, "simulationStep", simulation_step)
    monkeypatch.setattr(
        traci.simulation, "getArrivedPersonIDList", lambda: ["rider_0"]
    )
    monkeypatch.setattr(traci.simulation, "getTime", lambda: clock["time"])
    monkeypatch.setattr(env, "_mean_pending_wait_time", lambda: 0.0)
    monkeypatch.setattr(env, "_active_idle_taxi_ids", lambda: [])
    monkeypatch.setattr(env, "_build_all_obs", lambda agents: {})

    obs, rewards, terminations, truncations, infos = env._step_impl({})

    assert obs == {}
    assert rewards == {}
    assert terminations == {}
    assert truncations == {}
    assert infos == {}
    assert env.last_step_metrics.completed_passenger_journeys == 1
    assert env.last_step_metrics.team_reward == 10.0
    assert env.last_step_metrics.terminated
