"""Multi-head graph-attention layer with explicit attention outputs.

The implementation uses scaled dot-product attention and a padding mask. It
returns the attention tensor required by the faithfulness audit without adding
a `torch-geometric` dependency.

Input assumption: the observation graph is *fully connected* among valid
nodes. Padded positions are excluded via mask.

Shape conventions everywhere in this module:
    B = batch size
    N = max number of nodes per graph (padded to a common size)
    H = number of attention heads
    D = hidden dim (per head: D // H)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Scaled dot-product multi-head attention, applied node-to-node.

    Returns updated node features AND the per-head attention weights so the
    caller can save them for the coupled-explanation faithfulness pipeline.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # A tiny feed-forward after attention, standard transformer trick.
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, N, dim)
        node_mask: torch.Tensor,  # (B, N) — 1 = valid, 0 = padded
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (updated_x, attention).

        - updated_x:  (B, N, dim)
        - attention:  (B, H, N, N)  — softmax rows over columns; padded
                      keys are set to 0, padded queries return uniform
                      (irrelevant, since caller ignores them via mask).
        """
        B, N, _ = x.shape
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, head_dim)
        # Rearrange to (B, H, N, head_dim) so einsum is intuitive.
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # (B, H, N_query, N_key)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(self.head_dim)
        # Mask out padded KEYS on every query row.
        key_mask = node_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        scores = scores.masked_fill(key_mask == 0, float("-inf"))
        # If every key is masked out (all invalid), avoid NaN from -inf softmax.
        any_valid = node_mask.sum(dim=-1, keepdim=True) > 0
        attn = torch.where(
            any_valid.unsqueeze(1).unsqueeze(1),
            F.softmax(scores, dim=-1),
            torch.zeros_like(scores),
        )
        attn = self.dropout(attn)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)  # (B, H, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        out = self.out_proj(out)

        x = residual + out
        # Feed-forward block.
        x = x + self.ff(self.norm2(x))
        return x, attn
