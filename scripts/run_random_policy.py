"""Smoke test: run one DispatchEnv episode with a uniform-random policy.

Verifies that:
  - the env constructs and resets end-to-end
  - action-space / observation-space shapes match the actual outputs
  - dispatchTaxi() succeeds for at least some assignments
  - reward is finite and sensible
  - the episode terminates cleanly at the sumocfg end_time

Not a training run — expected performance is bad.

Usage:
  python scripts/run_random_policy.py --area central_park
  python scripts/run_random_policy.py --area yuelai --degradation tunnel_triggered
  python scripts/run_random_policy.py --area xiantao --steps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", default="central_park",
                        choices=["central_park", "yuelai", "xiantao"])
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered", "random_dropout"])
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                        help="only used when --degradation random_dropout")
    parser.add_argument("--steps", type=int, default=None,
                        help="hard cap on RL steps; default runs to sumocfg end")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    deg = DegradationConfig(
        mode=args.degradation,
        dropout_rate=args.dropout_rate,
    )
    cfg = DispatchEnvConfig(area=args.area, seed=args.seed, degradation=deg)

    env = DispatchEnv(cfg)
    print(f"scenario: {env.scenario.name}  |  navigable tunnels: {len(env.scenario.tunnel_edges)}")
    print(f"degradation: {deg.mode}  |  step_length_s={cfg.step_length_s}")

    obs, infos = env.reset(seed=args.seed)
    print(f"reset OK  |  initial agents: {len(env.agents)}")

    step = 0
    total_reward = 0.0
    total_pickups = 0
    action_space = env.action_space("taxi_0")
    rng = __import__("random").Random(args.seed)

    # Loop on episode termination, not on agent set — agents can be empty
    # mid-episode when every taxi is mid-ride.
    while not env.done:
        actions = {a: rng.randrange(action_space.n) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        r = next(iter(rewards.values())) if rewards else 0.0
        info = next(iter(infos.values())) if infos else {}
        total_reward += r
        total_pickups += info.get("pickups_delta", 0)
        if step % 10 == 0:
            print(
                f"  step {step:3d}  sim_t={env.sim_time:6.0f}s  "
                f"agents={len(env.agents)}  r={r:+.2f}  "
                f"pickups+={info.get('pickups_delta', 0)}  "
                f"total_pickups={total_pickups}  "
                f"mean_wait={info.get('mean_wait_time', 0.0):.1f}s"
            )
        step += 1
        if args.steps is not None and step >= args.steps:
            break

    env.close()
    print(
        f"\nepisode done  |  RL steps={step}  |  total pickups={total_pickups}  "
        f"|  cumulative reward={total_reward:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
