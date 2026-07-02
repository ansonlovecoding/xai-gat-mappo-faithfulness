"""Evaluate a trained GAT-MAPPO checkpoint against the baseline table.

Runs one or more evaluation episodes with the checkpointed policy and
reports pickups + reward + wait-time — the same metrics
`scripts/run_baselines.py` produces — so the numbers slot straight into a
comparison table.

Deterministic by default (argmax over action logits). Pass --stochastic to
sample instead, which matches training-time behaviour.

Usage:
  python scripts/eval_policy.py runs/mappo/central_park_<ts>/ckpt_epoch_0299.pt
  python scripts/eval_policy.py <ckpt> --episodes 5
  python scripts/eval_policy.py <ckpt> --episodes 5 --stochastic
  python scripts/eval_policy.py <ckpt> --degradation tunnel_triggered
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
)
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    PolicyConfig,
    obs_dict_to_tensors,
)


def _choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_policy(ckpt_path: Path, device: str) -> tuple[DispatchGATPolicy, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pol_cfg = PolicyConfig(**{**ckpt["policy_config"], "device": device})
    policy = DispatchGATPolicy(pol_cfg)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, ckpt


def _run_episode(
    env: DispatchEnv,
    policy: DispatchGATPolicy,
    device: str,
    stochastic: bool,
) -> dict:
    obs_dict, _ = env.reset()
    total_reward = 0.0
    total_pickups = 0
    step = 0
    last_wait = 0.0

    while not env.done:
        if obs_dict:
            batched, agents = obs_dict_to_tensors(obs_dict, device=device)
            with torch.no_grad():
                out = policy.forward(batched)
                logits = out["logits"]
                if stochastic:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    action = logits.argmax(dim=-1)
            actions = {a: int(action[i].item()) for i, a in enumerate(agents)}
        else:
            actions = {}

        obs_dict, rewards, _, _, infos = env.step(actions)
        if rewards:
            total_reward += next(iter(rewards.values()))
        if infos:
            info = next(iter(infos.values()))
            total_pickups += int(info.get("pickups_delta", 0))
            last_wait = float(info.get("mean_wait_time", last_wait))
        step += 1

    return {
        "total_pickups": total_pickups,
        "total_reward": round(total_reward, 3),
        "final_mean_pending_wait_s": round(last_wait, 1),
        "rl_steps": step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help=".pt file saved by train.py")
    parser.add_argument("--area", default=None,
                        help="override area; defaults to whatever the checkpoint was trained on")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes to average over")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of argmax")
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered", "random_dropout"])
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    n_params = sum(p.numel() for p in policy.parameters())

    area = args.area or ckpt["env_config"]["area"]
    env_cfg = DispatchEnvConfig(
        area=area,
        seed=args.seed,
        degradation=DegradationConfig(mode=args.degradation, dropout_rate=args.dropout_rate),
    )
    env = DispatchEnv(env_cfg)

    print(f"checkpoint: {args.checkpoint.name}  (epoch {ckpt.get('epoch', '?')}, {n_params:,} params)")
    print(f"env:        {area}  |  degradation: {args.degradation}  |  device: {device}")
    print(f"policy:     {'stochastic' if args.stochastic else 'argmax (deterministic)'}")
    print()
    print(f"{'ep':>3}  {'pickups':>7}  {'reward':>10}  {'mean_wait_s':>11}  {'rl_steps':>8}  {'wall_s':>7}")
    print("-" * 60)

    results = []
    for ep in range(args.episodes):
        t0 = time.time()
        r = _run_episode(env, policy, device, args.stochastic)
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"{ep:>3}  {r['total_pickups']:>7}  {r['total_reward']:>+10.2f}  "
              f"{r['final_mean_pending_wait_s']:>11}  {r['rl_steps']:>8}  {r['wall_s']:>7.1f}")

    env.close()

    # Aggregate.
    pickups = np.array([r["total_pickups"] for r in results])
    rewards = np.array([r["total_reward"] for r in results])
    waits = np.array([r["final_mean_pending_wait_s"] for r in results])
    print()
    print(f"mean over {len(results)} episodes:")
    print(f"  pickups:   {pickups.mean():.2f} ± {pickups.std():.2f}")
    print(f"  reward:    {rewards.mean():+.2f} ± {rewards.std():.2f}")
    print(f"  mean_wait: {waits.mean():.1f}s")

    # Emit machine-readable summary for downstream comparison.
    summary_path = args.checkpoint.with_suffix(".eval.json")
    summary_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "area": area,
        "degradation": args.degradation,
        "stochastic": args.stochastic,
        "seed": args.seed,
        "episodes": len(results),
        "per_episode": results,
        "mean_pickups": float(pickups.mean()),
        "std_pickups": float(pickups.std()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
    }, indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
