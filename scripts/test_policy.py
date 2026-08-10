"""Smoke test the GAT-MAPPO policy end-to-end.

Steps:
  1. Build the env, reset to get real obs
  2. Instantiate a fresh (random-init) policy
  3. Forward the obs, sample actions, verify shapes / no NaN
  4. Feed the sampled actions back into the env to close the loop
  5. Run for a few RL steps and confirm attention weights are valid softmax

Fresh init means the policy is bad — we are NOT measuring pickup counts here,
we're verifying the tensor plumbing.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import DispatchEnv, DispatchEnvConfig  # noqa: E402
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    PolicyConfig,
    obs_dict_to_tensors,
)


def main() -> int:
    device = ("mps" if torch.backends.mps.is_available()
              and platform.machine() == "arm64" else "cpu")
    print(f"device: {device}")

    env_cfg = DispatchEnvConfig(area="central_park")
    env = DispatchEnv(env_cfg)
    obs, _ = env.reset()
    print(f"env: {env_cfg.area}  |  initial idle agents: {len(env.agents)}")

    pol_cfg = PolicyConfig(
        k_neighbors=env_cfg.k_neighbors,
        k_reservations=env_cfg.k_reservations,
        device=device,
    )
    policy = DispatchGATPolicy(pol_cfg)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params:,} parameters")

    for rl_step in range(5):
        if not obs:
            # No idle agents this step — advance sim without actions.
            obs, *_ = env.step({})
            continue
        batched, agents = obs_dict_to_tensors(obs, device=device)
        with torch.no_grad():
            out = policy.get_action_and_value(batched)
        actions_np = out["action"].cpu().numpy().astype(int)
        actions = {a: int(actions_np[i]) for i, a in enumerate(agents)}

        # Shape checks.
        B = len(agents)
        assert out["logits"].shape == (B, env_cfg.k_reservations + 1), out["logits"].shape
        assert out["value"].shape == (B,), out["value"].shape
        assert out["attention"].shape == (
            pol_cfg.n_gat_layers, B, pol_cfg.n_heads,
            1 + env_cfg.k_neighbors + env_cfg.k_reservations,
            1 + env_cfg.k_neighbors + env_cfg.k_reservations,
        ), out["attention"].shape

        # NaN checks.
        for k in ("logits", "value", "attention"):
            assert not torch.isnan(out[k]).any(), f"NaN in {k}"

        # Attention rows should sum to ~1 or ~0 (all-masked-key rows).
        attn = out["attention"]  # (L, B, H, N, N)
        row_sums = attn.sum(dim=-1)  # (L, B, H, N)
        # Every row is either ~1 (softmax) or ~0 (all keys masked out).
        near_1 = torch.isclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        near_0 = torch.isclose(row_sums, torch.zeros_like(row_sums), atol=1e-6)
        assert (near_1 | near_0).all(), f"bad attention row sums: {row_sums.min():.4f}..{row_sums.max():.4f}"

        obs, rewards, _, _, infos = env.step(actions)
        r = next(iter(rewards.values())) if rewards else 0.0
        info = next(iter(infos.values())) if infos else {}
        print(
            f"  rl_step {rl_step}  sim_t={env.sim_time:6.0f}s  "
            f"agents={len(agents):2d}  actions[0..3]={actions_np[:3].tolist()}  "
            f"r={r:+.2f}  pickups+={info.get('pickups_delta', 0)}"
        )

    env.close()
    print("OK — GAT-MAPPO policy runs end-to-end, shapes and attention are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
