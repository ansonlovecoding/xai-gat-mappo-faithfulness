"""Train the GAT-MAPPO policy on a Yubei dispatch scenario.

Simplest possible working training loop — one env, one rollout per epoch, PPO
update, then repeat. Not vectorised across envs (TraCI is single-connection
per process); we can add subprocess parallelism later.

Logs to stdout as a table AND appends a JSONL row per epoch under
``runs/mappo/<area>_<timestamp>/train_log.jsonl``. Model checkpoints saved at
``ckpt_epoch_<N>.pt`` in the same dir.

Usage:
  python scripts/train.py --area central_park --epochs 50
  python scripts/train.py --area yuelai --epochs 20 --lr 1e-4
  python scripts/train.py --area central_park --degradation tunnel_triggered
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
# Put PROJECT_ROOT on the path so `src.dispatch_marl` resolves. (This matches
# the imports below; sys.path must include the *parent* of `src/`.)
sys.path.insert(0, str(PROJECT_ROOT))

from src.dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    PPOConfig,
    collect_rollout,
    compute_gae,
    ppo_update,
)
from src.dispatch_marl.models import DispatchGATPolicy, PolicyConfig  # noqa: E402


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="central_park",
                        choices=["central_park", "yuelai", "xiantao"])
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered", "random_dropout"])
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    # Reward shaping.
    parser.add_argument("--pickup-reward", type=float, default=10.0)
    parser.add_argument("--dispatch-reward", type=float, default=0.5)
    parser.add_argument("--wait-lambda", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-gat-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    # Best-checkpoint tracking: after each epoch, compute the mean of the last
    # `--best-window` training-episode pickup counts. Whenever that rolling
    # mean improves on the previous best, snapshot the current policy to
    # `ckpt_best.pt` and update `best_metadata.json`. Set --best-window 0 to
    # disable. Kept as a rolling training statistic (rather than an extra eval
    # rollout per epoch) to keep training cheap.
    parser.add_argument("--best-window", type=int, default=10,
                        help="epochs to average for rolling-mean best tracking; 0 disables")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "runs" / "mappo")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or mps; default: auto")
    args = parser.parse_args()

    device = args.device or choose_device()

    run_dir = args.log_dir / f"{args.area}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.jsonl"
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"run dir: {run_dir}")
    print(f"device:  {device}")

    # ---- env
    env_cfg = DispatchEnvConfig(
        area=args.area,
        seed=args.seed,
        pickup_reward=args.pickup_reward,
        dispatch_reward=args.dispatch_reward,
        wait_penalty_lambda=args.wait_lambda,
        degradation=DegradationConfig(mode=args.degradation, dropout_rate=args.dropout_rate),
    )
    env = DispatchEnv(env_cfg)

    # ---- policy
    pol_cfg = PolicyConfig(
        k_neighbors=env_cfg.k_neighbors,
        k_reservations=env_cfg.k_reservations,
        hidden_dim=args.hidden_dim,
        n_gat_layers=args.n_gat_layers,
        n_heads=args.n_heads,
        device=device,
    )
    policy = DispatchGATPolicy(pol_cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"policy:  {sum(p.numel() for p in policy.parameters()):,} params")

    ppo_cfg = PPOConfig(
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
    )

    # ---- table header
    header = ("epoch", "pickups", "reward", "n_steps", "π_loss", "V_loss", "H", "KL", "clipfrac", "wall_s")
    fmt = "{:>5}  {:>7}  {:>+8.2f}  {:>7}  {:>8.4f}  {:>8.4f}  {:>7.3f}  {:>7.4f}  {:>8.3f}  {:>6.1f}"
    print()
    print("{:>5}  {:>7}  {:>8}  {:>7}  {:>8}  {:>8}  {:>7}  {:>7}  {:>8}  {:>6}".format(*header))
    print("-" * 95)

    # ---- best-checkpoint tracking state
    recent_pickups: list[int] = []
    best_rolling_mean = float("-inf")
    best_epoch = -1
    best_ckpt_path = run_dir / "ckpt_best.pt"
    best_meta_path = run_dir / "best_metadata.json"

    def _save_best(epoch: int, rolling_mean: float) -> None:
        torch.save({
            "epoch": epoch,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_config": pol_cfg.__dict__,
            "env_config": {**env_cfg.__dict__,
                           "degradation": env_cfg.degradation.__dict__},
            "ppo_config": ppo_cfg.__dict__,
            "best": {
                "rolling_mean_pickups": rolling_mean,
                "window": args.best_window,
            },
        }, best_ckpt_path)
        best_meta_path.write_text(json.dumps({
            "best_epoch": epoch,
            "best_rolling_mean_pickups": rolling_mean,
            "window": args.best_window,
            "note": "Rolling mean of training-episode pickups over the last "
                    f"{args.best_window} epochs. Updated whenever it improves.",
        }, indent=2))

    # ---- train loop
    for epoch in range(args.epochs):
        t0 = time.time()

        # 1. rollout
        buffer, ep_stats = collect_rollout(env, policy, device=device)

        # 2. GAE
        compute_gae(buffer, gamma=args.gamma, gae_lambda=args.gae_lambda)

        # 3. PPO update
        losses = ppo_update(policy, optimizer, buffer, ppo_cfg, device=device)
        loss_d = losses.as_dict()

        elapsed = time.time() - t0

        # ---- best-checkpoint tracking (rolling mean of training pickups)
        is_new_best = False
        rolling_mean = None
        if args.best_window > 0:
            recent_pickups.append(ep_stats.total_pickups)
            if len(recent_pickups) > args.best_window:
                recent_pickups.pop(0)
            if len(recent_pickups) >= args.best_window:
                rolling_mean = sum(recent_pickups) / len(recent_pickups)
                if rolling_mean > best_rolling_mean:
                    best_rolling_mean = rolling_mean
                    best_epoch = epoch
                    is_new_best = True
                    _save_best(epoch, rolling_mean)

        row = {
            "epoch": epoch,
            "pickups": ep_stats.total_pickups,
            "reward": round(ep_stats.total_reward, 3),
            "n_agent_steps": ep_stats.n_agent_steps,
            "wall_s": round(elapsed, 1),
            "rolling_mean_pickups": round(rolling_mean, 3) if rolling_mean is not None else None,
            "is_new_best": is_new_best,
            **{k: round(v, 5) for k, v in loss_d.items()},
        }
        with log_file.open("a") as f:
            f.write(json.dumps(row) + "\n")

        marker = "  ★ new best" if is_new_best else ""
        print(fmt.format(
            epoch,
            ep_stats.total_pickups,
            ep_stats.total_reward,
            ep_stats.n_agent_steps,
            loss_d["policy_loss"],
            loss_d["value_loss"],
            loss_d["entropy"],
            loss_d["approx_kl"],
            loss_d["clipfrac"],
            elapsed,
        ) + marker)

        # 4. checkpoint
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            ckpt = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "policy_config": pol_cfg.__dict__,
                "env_config": {**env_cfg.__dict__,
                               "degradation": env_cfg.degradation.__dict__},
                "ppo_config": ppo_cfg.__dict__,
            }, ckpt)

    env.close()
    print(f"\ndone. logs: {log_file}")
    if best_epoch >= 0:
        print(f"best:   epoch {best_epoch}, rolling mean pickups = {best_rolling_mean:.2f}")
        print(f"        checkpoint: {best_ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
