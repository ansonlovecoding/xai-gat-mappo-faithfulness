"""MAPPO baseline with a plain MLP encoder and no graph structure.

It uses the same observation, action space, reward, and trainer as the GAT
policy, but encodes the flattened observation with an MLP.

Interface-compatible with `DispatchGATPolicy` everywhere the training and
evaluation loops touch it (`forward` → logits/value, `get_action_and_value`
→ action/log_prob/entropy/value). Deliberately NOT compatible with the
faithfulness pipeline: an MLP has no attention channel, so there is no
attention explanation to score. `forward()` therefore returns no `attention`
key.

Design notes:

- **Input layout.** The obs dict is flattened in a fixed order:
  self ‖ neighbor_taxis ‖ neighbor_taxis_mask ‖ reservations ‖
  reservations_mask ‖ position_valid. Masks are included as inputs — the
  MLP has no masking mechanism of its own, so the mask bits are the only
  way it can distinguish "padded slot" from "valid node at the origin".
- **Invalid reservation actions are still masked to -inf** in the logits,
  exactly as in the GAT policy. Action-space semantics are identical.
- **Centralised critic** mean-pools the trunk embeddings within each recorded
  environment transition. PPO minibatches preserve those groups, so unrelated
  simulation times are never treated as one joint state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class MLPPolicyConfig:
    # Must match DispatchEnvConfig.k_neighbors / k_reservations.
    k_neighbors: int = 5
    k_reservations: int = 5
    # Per-type input feature widths must match env.py's constants.
    self_feat_dim: int = 5
    taxi_feat_dim: int = 5
    res_feat_dim: int = 5
    # 176 puts the default B1 at ~106k params vs the default GAT's ~114k,
    # so the B1-vs-B2 comparison is encoder-vs-encoder at matched capacity.
    hidden_dim: int = 176
    n_layers: int = 3
    # Same CTDE trade-off as the GAT policy; default matches B2 so the
    # B1-vs-B2 comparison isolates the encoder, not the critic regime.
    centralised_critic: bool = True
    critic_encoder_gradient_scale: float = 1.0
    device: str = "cpu"

    @property
    def input_dim(self) -> int:
        K_n, K_r = self.k_neighbors, self.k_reservations
        return (
            self.self_feat_dim
            + K_n * self.taxi_feat_dim + K_n      # neighbours + their mask
            + K_r * self.res_feat_dim + K_r       # reservations + their mask
            + 1                                    # position_valid
        )


class DispatchMLPPolicy(nn.Module):
    """Flat-MLP actor-critic over the flattened dispatch observation."""

    def __init__(self, config: MLPPolicyConfig | None = None):
        super().__init__()
        self.config = config or MLPPolicyConfig()
        c = self.config
        if not 0.0 <= c.critic_encoder_gradient_scale <= 1.0:
            raise ValueError("critic_encoder_gradient_scale must be in [0, 1]")
        d = c.hidden_dim

        layers: list[nn.Module] = [nn.Linear(c.input_dim, d), nn.GELU()]
        for _ in range(c.n_layers - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        self.trunk = nn.Sequential(*layers)

        self.actor_head = nn.Linear(d, c.k_reservations + 1)
        self.critic_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self.to(c.device)

    # ------------------------------------------------------------------ forward

    def _flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs["self"].shape[0]
        parts = [
            obs["self"],
            obs["neighbor_taxis"].reshape(B, -1),
            obs["neighbor_taxis_mask"].float(),
            obs["reservations"].reshape(B, -1),
            obs["reservations_mask"].float(),
            obs["position_valid"].float(),
        ]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        critic_group: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns {"logits": (B, K_r+1), "value": (B,)}.

        No "attention" key, by design — B1 has no explanation channel.
        """
        x = self._flatten_obs(obs)
        h = self.trunk(x)  # (B, D)

        logits = self.actor_head(h)  # (B, K_r + 1)
        res_mask = obs["reservations_mask"].bool()
        logits = torch.cat(
            [
                logits[:, :1],
                logits[:, 1:].masked_fill(~res_mask, float("-inf")),
            ],
            dim=1,
        )

        scale = self.config.critic_encoder_gradient_scale
        critic_h = h.detach() + scale * (h - h.detach())
        if self.config.centralised_critic:
            B = h.shape[0]
            if critic_group is None:
                critic_group = torch.zeros(B, dtype=torch.long, device=h.device)
            _, inverse = torch.unique(
                critic_group.to(device=h.device, dtype=torch.long),
                sorted=True,
                return_inverse=True,
            )
            n_groups = int(inverse.max().item()) + 1
            group_sum = torch.zeros(
                n_groups, h.shape[-1], dtype=h.dtype, device=h.device
            ).index_add_(0, inverse, critic_h)
            group_count = torch.zeros(
                n_groups, 1, dtype=h.dtype, device=h.device
            ).index_add_(0, inverse, torch.ones(B, 1, dtype=h.dtype, device=h.device))
            group_values = self.critic_head(group_sum / group_count).squeeze(-1)
            value = group_values[inverse]
        else:
            value = self.critic_head(critic_h).squeeze(-1)

        return {"logits": logits, "value": value}

    # ------------------------------------------------------------------ MAPPO API

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        critic_group: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Same contract as DispatchGATPolicy.get_action_and_value, minus the
        attention/node_emb aux outputs (an MLP has neither)."""
        out = self.forward(obs, critic_group=critic_group)
        dist = torch.distributions.Categorical(logits=out["logits"])
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "logits": out["logits"],
            "value": out["value"],
        }
