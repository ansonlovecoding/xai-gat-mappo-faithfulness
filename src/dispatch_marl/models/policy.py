"""GAT-MAPPO policy for the fleet-dispatch env.

Architecture (per agent, since fleet uses parameter sharing):

    obs dict ─► per-type node embeddings ─► stack of GAT layers ─► actor + critic
                                                     │
                                                     └── attention weights
                                                         are the *coupled*
                                                         explanation channel

Actor head produces logits over Discrete(K_res + 1):
    action 0        = no-op   (uses updated self-node embedding)
    action 1..K_res = accept k-th reservation  (uses updated reservation
                                                embeddings, masked to -inf
                                                where reservations_mask=0)

Critic head takes a masked mean of all node embeddings → linear → scalar V.

Coupled explanation: `forward()` returns `attention` — the per-layer, per-head
softmax weights from every GATLayer. Same tensor is used by the policy to
decide AND by the faithfulness pipeline to explain. This is precisely the
"coupled" regime the dissertation will contrast against a decoupled head.
The decoupled head is deliberately NOT added yet — we'll design it after
faithfulness metrics are picked.

The MAPPO trainer will call:
    policy.get_action_and_value(obs, action=None) → action, log_prob, entropy, value, aux
where `aux` bundles the attention weights and per-node embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat import GATLayer

# Node type slot layout used everywhere below. Node 0 is the acting taxi
# ("self"); the next K_n slots are neighbouring taxis; the last K_r slots
# are candidate reservations.
NODE_TYPE_SELF = 0
NODE_TYPE_TAXI = 1
NODE_TYPE_RES = 2


@dataclass
class PolicyConfig:
    # Must match DispatchEnvConfig.k_neighbors / k_reservations.
    k_neighbors: int = 5
    k_reservations: int = 5
    # Per-type input feature widths must match env.py's constants.
    # self: [x_norm, y_norm, episode_time_norm, velocity_norm, aoi_norm]
    self_feat_dim: int = 5
    taxi_feat_dim: int = 4
    res_feat_dim: int = 5
    hidden_dim: int = 64
    n_gat_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    device: str = "cpu"


class DispatchGATPolicy(nn.Module):
    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        c = self.config
        d = c.hidden_dim

        # Per-type embeddings project heterogeneous input feature widths
        # into a shared node-embedding space.
        self.self_embed = nn.Linear(c.self_feat_dim, d)
        self.taxi_embed = nn.Linear(c.taxi_feat_dim, d)
        self.res_embed = nn.Linear(c.res_feat_dim, d)

        # Learnable node-type embedding added to per-type projections.
        # Gives the GAT a stable "which slot am I" signal.
        self.type_embed = nn.Embedding(3, d)

        # An extra "degradation" scalar signal on the self node — the
        # position_valid flag from the obs. Lets the policy condition on
        # whether its own reading is trusted.
        self.degradation_embed = nn.Linear(1, d)

        self.gat_layers = nn.ModuleList(
            [GATLayer(d, heads=c.n_heads, dropout=c.dropout) for _ in range(c.n_gat_layers)]
        )

        # Actor head. Shares one MLP over reservation embeddings for actions
        # 1..K_r; a separate linear for the no-op logit off the self embedding.
        self.res_logit_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.noop_logit_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        # Critic: masked-mean pool of node embeddings → V.
        self.critic_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self.to(c.device)

    # ------------------------------------------------------------------ forward

    def _build_node_bank(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed inputs into (B, N, D) node bank plus a (B, N) mask.

        Layout: node 0 = self, 1..K_n = neighbour taxis, K_n+1..end = reservations.
        """
        c = self.config
        B = obs["self"].shape[0]
        d = c.hidden_dim

        self_x = self.self_embed(obs["self"])           # (B, D)
        taxi_x = self.taxi_embed(obs["neighbor_taxis"])  # (B, K_n, D)
        res_x = self.res_embed(obs["reservations"])      # (B, K_r, D)

        # Add per-type embedding.
        self_x = self_x + self.type_embed(
            torch.full((B,), NODE_TYPE_SELF, dtype=torch.long, device=self_x.device)
        )
        taxi_type = self.type_embed(
            torch.full((B, c.k_neighbors), NODE_TYPE_TAXI, dtype=torch.long, device=self_x.device)
        )
        res_type = self.type_embed(
            torch.full((B, c.k_reservations), NODE_TYPE_RES, dtype=torch.long, device=self_x.device)
        )
        taxi_x = taxi_x + taxi_type
        res_x = res_x + res_type

        # Degradation signal onto self node.
        deg = self.degradation_embed(obs["position_valid"].float())  # (B, D)
        self_x = self_x + deg

        # (B, N, D). N = 1 + K_n + K_r.
        nodes = torch.cat([self_x.unsqueeze(1), taxi_x, res_x], dim=1)

        # Node mask. Self is always valid.
        self_mask = torch.ones(B, 1, dtype=torch.int8, device=nodes.device)
        node_mask = torch.cat(
            [self_mask, obs["neighbor_taxis_mask"], obs["reservations_mask"]], dim=1
        ).to(torch.int8)

        return nodes, node_mask

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one forward pass. Returns:
            {
                "logits":   (B, K_r + 1),
                "value":    (B,),
                "node_emb": (B, N, D),
                "attention": (n_layers, B, H, N, N),
                "node_mask": (B, N),
            }
        """
        c = self.config
        nodes, node_mask = self._build_node_bank(obs)  # (B, N, D), (B, N)

        attn_stack = []
        for layer in self.gat_layers:
            nodes, attn = layer(nodes, node_mask)
            attn_stack.append(attn)
        attention = torch.stack(attn_stack, dim=0)  # (L, B, H, N, N)

        # Actor: no-op from self, reservation logits from reservation embeddings.
        self_emb = nodes[:, 0]                         # (B, D)
        res_emb = nodes[:, 1 + c.k_neighbors :]        # (B, K_r, D)

        noop_logit = self.noop_logit_head(self_emb)                   # (B, 1)
        res_logits = self.res_logit_head(res_emb).squeeze(-1)         # (B, K_r)

        # Mask invalid reservations to -inf so softmax ignores them.
        res_mask = obs["reservations_mask"].bool()                    # (B, K_r)
        res_logits = res_logits.masked_fill(~res_mask, float("-inf"))

        logits = torch.cat([noop_logit, res_logits], dim=1)  # (B, K_r + 1)

        # Critic: masked-mean of node embeddings.
        w = node_mask.float().unsqueeze(-1)             # (B, N, 1)
        pooled = (nodes * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        value = self.critic_head(pooled).squeeze(-1)    # (B,)

        return {
            "logits": logits,
            "value": value,
            "node_emb": nodes,
            "attention": attention,
            "node_mask": node_mask,
        }

    # ------------------------------------------------------------------ MAPPO API

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """PPO/MAPPO-style hook. If `action` is provided (during PPO update)
        we compute log_prob and entropy at that action; otherwise we sample.
        """
        out = self.forward(obs)
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
            "attention": out["attention"],
            "node_emb": out["node_emb"],
            "node_mask": out["node_mask"],
        }


# ---------------------------------------------------------------------- helpers


def obs_dict_to_tensors(
    obs: dict[str, dict[str, np.ndarray]],
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Stack a dict-of-dicts (agent → obs) into a batched tensor dict.

    Returns (batched_obs, agent_order). The caller uses agent_order to
    map action tensor rows back to agent IDs.
    """
    agents = sorted(obs.keys())
    if not agents:
        return {}, []

    keys = list(obs[agents[0]].keys())
    batched: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([obs[a][k] for a in agents], axis=0)
        batched[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    # Masks should be int for masking ops but float for arithmetic is fine.
    return batched, agents
