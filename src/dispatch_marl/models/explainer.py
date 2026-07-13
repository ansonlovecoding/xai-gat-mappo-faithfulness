"""Decoupled explanation head — the dissertation's architectural comparison.

The coupled explanation channel is the GAT's own attention: the same
weights drive the action AND serve as the explanation. This module is the
*decoupled* alternative: a small scorer that reads the policy's node
embeddings and outputs a per-node importance distribution, but feeds
NOTHING back into the actor. Two guarantees enforce the decoupling:

1. **No influence on the policy.** The head consumes `node_emb.detach()`,
   so training it can never change policy weights, and the policy's
   forward pass never sees the head.
2. **Same evidence.** It reads the post-GAT node bank — exactly the
   representation the actor head reads — so the comparison isolates
   *where the explanation comes from*, not what information it has.

Training signal (occlusion distillation, see scripts/distill_explainer.py):
for each decision, the "ground-truth" importance of node i is the drop in
the decision's logit margin when node i is occluded — the same
counterfactual machinery DEF is built on, so a head that learns its
targets is faithful in DEF's own currency. Targets are softmax-normalised
over the valid non-self nodes and the head is trained with KL divergence.

At evaluation time `FaithfulnessEvaluator.evaluate_decision(...,
importance_row=head_row)` scores the decoupled explanation with the
identical DEF/WAMSN machinery used for attention, enabling a paired
per-decision comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExplainerConfig:
    # Must match the policy's hidden_dim (the node-embedding width).
    hidden_dim: int = 64
    # Scorer MLP width.
    mlp_dim: int = 64
    # Softmax temperature applied to the occlusion targets at training
    # time. Lower = sharper target distributions.
    target_temperature: float = 1.0
    device: str = "cpu"


class DecoupledExplainerHead(nn.Module):
    """(B, N, D) node embeddings → (B, N) importance distribution."""

    def __init__(self, config: ExplainerConfig | None = None):
        super().__init__()
        self.config = config or ExplainerConfig()
        c = self.config
        self.scorer = nn.Sequential(
            nn.Linear(c.hidden_dim, c.mlp_dim),
            nn.GELU(),
            nn.Linear(c.mlp_dim, 1),
        )
        self.to(c.device)

    def forward(
        self,
        node_emb: torch.Tensor,   # (B, N, D) — pass DETACHED embeddings
        node_mask: torch.Tensor,  # (B, N) — 1 valid, 0 padded
        exclude_self: bool = True,
    ) -> torch.Tensor:
        """Return (B, N) importance distribution over valid nodes.

        The self node (index 0) is excluded by default — it is never an
        ablation candidate in the DEF machinery, so the explanation is a
        distribution over context nodes, mirroring how the attention row
        is consumed there.
        """
        logits = self.scorer(node_emb).squeeze(-1)  # (B, N)
        mask = node_mask.bool()
        if exclude_self:
            mask = mask.clone()
            mask[:, 0] = False
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(logits, dim=-1)

    # -------------------------------------------------------------- io

    def save(self, path: Path, extra: dict | None = None) -> None:
        torch.save({
            "model": self.state_dict(),
            "config": self.config.__dict__,
            **(extra or {}),
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> tuple["DecoupledExplainerHead", dict]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ExplainerConfig(**{**ckpt["config"], "device": device})
        head = cls(cfg)
        head.load_state_dict(ckpt["model"])
        head.eval()
        return head, ckpt


def explainer_importance_row(
    head: DecoupledExplainerHead,
    policy: torch.nn.Module,
    obs: dict[str, torch.Tensor],
) -> np.ndarray:
    """Convenience: one decision (B=1) → (N,) numpy importance row.

    Runs the policy forward to get node embeddings (no grad), then the
    head. Suitable for passing straight into
    FaithfulnessEvaluator.evaluate_decision(importance_row=...).
    """
    with torch.no_grad():
        out = policy.forward(obs)
        row = head(out["node_emb"].detach(), out["node_mask"])[0]
    return row.cpu().numpy().astype(np.float64)
