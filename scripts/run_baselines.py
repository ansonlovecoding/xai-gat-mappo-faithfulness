"""Run baseline dispatch policies across all three Yubei scenarios and
print a comparison table.

Included baselines:

  * `random`  — RandomPolicy via DispatchEnv (lower bound).
  * `nearest` — NearestReservationPolicy via DispatchEnv (greedy heuristic).
  * `sumo_greedy` — SUMO's built-in greedy dispatcher via `sumo` CLI,
    overriding the sumocfg's `traci` setting. Doesn't route through
    DispatchEnv at all; it's SUMO's own algorithm reading the sumocfg.

The GAT-MAPPO policy will later be compared against these three.

Usage:
  python scripts/run_baselines.py
  python scripts/run_baselines.py --areas central_park --policies random,nearest
  python scripts/run_baselines.py --save-json runs/baselines.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DispatchEnv,
    DispatchEnvConfig,
    NearestReservationPolicy,
    RandomPolicy,
)
from dispatch_marl._sumo import SUMO_HOME  # noqa: E402

ALL_AREAS = ("central_park", "yuelai", "xiantao")
POLICY_FACTORIES = {
    "random": lambda cfg: RandomPolicy(cfg.k_reservations, seed=cfg.seed),
    "nearest": lambda cfg: NearestReservationPolicy(),
}


def run_env_episode(area: str, policy_name: str, seed: int) -> dict:
    cfg = DispatchEnvConfig(area=area, seed=seed)
    env = DispatchEnv(cfg)
    policy = POLICY_FACTORIES[policy_name](cfg)

    t0 = time.time()
    obs, _ = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    total_pickups = 0
    step = 0
    last_mean_wait = 0.0
    while not env.done:
        actions = policy.act(obs) if obs else {}
        obs, rewards, _, _, infos = env.step(actions)
        if rewards:
            total_reward += next(iter(rewards.values()))
        if infos:
            info = next(iter(infos.values()))
            total_pickups += info.get("pickups_delta", 0)
            last_mean_wait = info.get("mean_wait_time", last_mean_wait)
        step += 1

    env.close()
    return {
        "area": area,
        "policy": policy_name,
        "total_pickups": total_pickups,
        "total_reward": round(total_reward, 3),
        "final_mean_pending_wait_s": round(last_mean_wait, 1),
        "rl_steps": step,
        "wall_time_s": round(time.time() - t0, 1),
    }


def run_sumo_greedy(area: str) -> dict:
    """Run SUMO with its built-in greedy dispatcher. Read stats from output."""
    scenario_dir = PROJECT_ROOT / "scenarios" / "yubei" / area
    cfg = scenario_dir / f"{area}.sumocfg"
    stats_file = Path("/tmp") / f"baseline_sumo_greedy_{area}_stats.xml"
    sumo_binary = str(Path(SUMO_HOME, "bin", "sumo"))

    t0 = time.time()
    subprocess.run(
        [
            sumo_binary,
            "-c", str(cfg),
            "--device.taxi.dispatch-algorithm", "greedy",
            "--statistic-output", str(stats_file),
            # extended stats (rideStatistics, vehicleTripStatistics) are
            # only emitted when this flag is set.
            "--duration-log.statistics",
            "--no-step-log", "--no-warnings",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tree = ET.parse(stats_file)
    root = tree.getroot()
    rides = root.find("rideStatistics")
    n_rides = int(rides.get("number", "0")) if rides is not None else 0
    mean_wait = float(rides.get("waitingTime", "0.0")) if rides is not None else 0.0

    return {
        "area": area,
        "policy": "sumo_greedy",
        "total_pickups": n_rides,
        "total_reward": None,  # not computed by SUMO
        "final_mean_pending_wait_s": round(mean_wait, 1),
        "rl_steps": None,
        "wall_time_s": round(time.time() - t0, 1),
    }


def print_table(rows: list[dict]) -> None:
    header = ("area", "policy", "pickups", "reward", "mean_wait_s", "steps", "wall_s")
    fmt = f"{{:<15}} {{:<12}} {{:>7}} {{:>10}} {{:>11}} {{:>6}} {{:>7}}"
    print()
    print(fmt.format(*header))
    print("-" * 78)
    for r in rows:
        reward = "-" if r["total_reward"] is None else f"{r['total_reward']:.2f}"
        steps = "-" if r["rl_steps"] is None else str(r["rl_steps"])
        print(fmt.format(
            r["area"], r["policy"], r["total_pickups"], reward,
            r["final_mean_pending_wait_s"], steps, r["wall_time_s"],
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--areas", default=",".join(ALL_AREAS),
                        help="comma-separated list; subset of central_park,yuelai,xiantao")
    parser.add_argument("--policies", default="random,nearest,sumo_greedy",
                        help="comma-separated: any of random,nearest,sumo_greedy")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--save-json", type=Path, default=None,
                        help="write results to this JSON path")
    args = parser.parse_args()

    areas = [a.strip() for a in args.areas.split(",") if a.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in ("random", "nearest", "sumo_greedy"):
            parser.error(f"unknown policy: {p}")

    rows: list[dict] = []
    for area in areas:
        for policy in policies:
            print(f"[{area}] {policy} ...", flush=True)
            try:
                if policy == "sumo_greedy":
                    row = run_sumo_greedy(area)
                else:
                    row = run_env_episode(area, policy, args.seed)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                continue
            rows.append(row)
            print(f"  pickups={row['total_pickups']}  wall={row['wall_time_s']}s", flush=True)

    print_table(rows)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(rows, indent=2))
        print(f"\nsaved: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
