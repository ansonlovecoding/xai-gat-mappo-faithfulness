"""Faithfulness evaluation for GAT-MAPPO dispatch decisions.

Implements the three metrics from §7.4 of the dissertation proposal, on
top of the coupled attention channel already exposed by `DispatchGATPolicy`:

- **DEF** (Dispatch Explanation Faithfulness)
  DEF = ½ (g_comp + g_suff), where g_comp and g_suff are normalised gains
  over a size-matched random explanation. DEF ∈ [−1, 1]; DEF > 0 means
  attention identifies information more faithfully than random.

- **WAMSN** (Weighted Attention Mass on Stale Nodes)
  WAMSN = Σ_i α_i · (AoI_i / AoI_max) / Σ_i α_i, summed over vehicle nodes
  (self + neighbour taxis). WAMSN ∈ [0, 1]. Higher = attention concentrates
  on stale telemetry.

- **Attention drift**
  JS(α_clean, α_degraded), base-2 log, ∈ [0, 1]. Higher = the explanation
  shifted more as data aged.

## Design notes

- **Node indexing.** In our per-agent heterogeneous graph the ordering is:
  node 0 = self, nodes 1..K_n = neighbour taxis, nodes K_n+1..N-1 =
  candidate reservations. `self` is never ablated — removing it would
  destroy the decision context.

- **Attention aggregation.** Per-node importance is the mean over layers
  and heads of `attention[l, 0, h, 0, j]` — i.e. the attention weight from
  self (node 0) to node j. Rationale: this is the row that most directly
  drives self's final embedding, which in turn drives the actor's action
  logits.

- **Random baseline.** For each `k`, sample `n_random_baselines` random
  size-`k` subsets from the *valid non-self* nodes and average their
  Comp/Suff. Configurable via `FaithfulnessConfig`.

- **Batching.** The default implementation does one policy forward per
  counterfactual (~36 forwards per decision at default settings). On the
  hardware sizes we're using this is comfortable (< 100 ms per decision on
  MPS). Optimising to batched counterfactuals is straightforward if a
  large sweep needs the speedup.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .env import AOI_MAX_S


# ------------------------------------------------------------ config / result


@dataclass
class FaithfulnessConfig:
    """Hyper-parameters for the faithfulness metrics."""

    # k-values used for Comp/Suff. Averaged over.
    top_k_values: tuple[int, ...] = (1, 2, 3)
    # Random-baseline subsets sampled per k. More = tighter g_comp/g_suff.
    n_random_baselines: int = 5
    # AoI normalisation constant. Must match env's AOI_MAX_S.
    aoi_max_s: float = AOI_MAX_S
    # How to reduce (L, H) → per-node attention. "mean" or "last" (last layer).
    aggregate_layers: str = "mean"
    # Deterministic random-baseline sampling.
    seed: int = 42


@dataclass
class DecisionFaithfulness:
    """All faithfulness numbers for a single dispatch decision."""

    action: int
    pi_full: float                    # π(a* | full graph)
    def_score: float                  # DEF ∈ [−1, 1]
    g_comp: float                     # normalised comprehensiveness gain
    g_suff: float                     # normalised sufficiency gain
    comp: float                       # raw comprehensiveness (mean over k)
    suff: float                       # raw sufficiency (mean over k)
    comp_rand: float                  # raw comprehensiveness of random baseline
    suff_rand: float
    wamsn: float                      # WAMSN ∈ [0, 1]
    attention_row: np.ndarray         # (N,) — for later drift computation
    node_mask: np.ndarray             # (N,) bool — which nodes were valid
    per_k: dict[int, dict[str, float]] = field(default_factory=dict)


# ------------------------------------------------------------ pure functions


def aggregate_node_attention(
    attention: torch.Tensor,
    from_node: int = 0,
    layer_agg: str = "mean",
) -> np.ndarray:
    """Reduce a raw GAT attention tensor to a single per-node importance row.

    Args:
        attention: shape (L, B, H, N, N) as returned by `policy.forward()`.
                   B must equal 1 (single decision).
        from_node: which row to read. Default 0 = self.
        layer_agg: "mean" over layers (default) or "last" layer only.

    Returns:
        (N,) numpy array. Sums to ~1 (softmax rows averaged).
    """
    if attention.dim() != 5:
        raise ValueError(f"expected attention with shape (L,B,H,N,N), got {attention.shape}")
    L, B, H, N, _ = attention.shape
    if B != 1:
        raise ValueError(f"aggregate_node_attention expects B=1 (single decision), got {B}")

    if layer_agg == "mean":
        row = attention[:, 0, :, from_node, :].mean(dim=(0, 1))
    elif layer_agg == "last":
        row = attention[-1, 0, :, from_node, :].mean(dim=0)
    else:
        raise ValueError(f"unknown layer_agg: {layer_agg!r}")

    arr = row.detach().cpu().numpy().astype(np.float64)
    # Guard against numeric drift so downstream WAMSN/drift don't complain.
    total = arr.sum()
    if total > 0:
        arr = arr / total
    return arr


def compute_attention_drift(
    attn_clean: np.ndarray,
    attn_degraded: np.ndarray,
) -> float:
    """Jensen–Shannon divergence between two per-node attention rows.

    Base-2 log, so the result lies in [0, 1]:
    0 = identical distributions, 1 = disjoint supports.
    """
    p = np.asarray(attn_clean, dtype=np.float64).clip(min=0)
    q = np.asarray(attn_degraded, dtype=np.float64).clip(min=0)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum

    # Add a tiny epsilon only where a probability is zero, to keep log
    # finite without distorting the distribution.
    eps = 1e-12
    p_safe = np.where(p > 0, p, eps)
    q_safe = np.where(q > 0, q, eps)
    m = 0.5 * (p_safe + q_safe)

    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    js = 0.5 * _kl(p_safe, m) + 0.5 * _kl(q_safe, m)
    return float(np.clip(js, 0.0, 1.0))


def compute_wamsn(
    attention_row: np.ndarray,
    aoi_per_node: np.ndarray,
    node_is_vehicle: np.ndarray,
    aoi_max: float = AOI_MAX_S,
) -> float:
    """Weighted Attention Mass on Stale Nodes over vehicle-type nodes.

    Args:
        attention_row: (N,) per-node attention (from `aggregate_node_attention`).
        aoi_per_node:  (N,) AoI in seconds; entries for non-vehicle nodes
                       are ignored via the vehicle-mask.
        node_is_vehicle: (N,) bool. True for self and (valid) neighbour taxis.
        aoi_max: normalisation constant. AoI is clipped to [0, aoi_max] then
                 divided.

    Returns:
        WAMSN ∈ [0, 1]. 0 when attention on vehicles is entirely fresh;
        approaching 1 as attention concentrates on maximally stale nodes.
        Returns 0.0 if no attention lands on any vehicle node.
    """
    attn = np.asarray(attention_row, dtype=np.float64)
    aoi = np.asarray(aoi_per_node, dtype=np.float64)
    veh = np.asarray(node_is_vehicle, dtype=bool)

    if not veh.any():
        return 0.0

    attn_v = attn[veh]
    aoi_v = aoi[veh]
    denom = float(attn_v.sum())
    if denom <= 0:
        return 0.0

    staleness = np.clip(aoi_v / max(aoi_max, 1e-9), 0.0, 1.0)
    numer = float((attn_v * staleness).sum())
    return float(numer / denom)


# ------------------------------------------------------------ evaluator


class FaithfulnessEvaluator:
    """Compute DEF/WAMSN for one decision at a time via counterfactual forwards.

    Wraps a `DispatchGATPolicy` (already in .eval() mode). Owns its own RNG
    for the random-baseline sampling so results are deterministic given a
    seed.
    """

    def __init__(
        self,
        policy: Any,  # DispatchGATPolicy — untyped to avoid a heavy import cycle
        config: FaithfulnessConfig | None = None,
    ):
        self.policy = policy
        self.config = config or FaithfulnessConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # -------------------------------------------------- public

    def attention_row(self, obs: dict[str, torch.Tensor]) -> np.ndarray:
        """Aggregated per-node attention row for a single-decision obs (B=1).

        One plain forward — used by the attention-drift pipeline to score a
        counterfactual (e.g. clean-twin) obs without the full DEF machinery.
        """
        self._check_batch_one(obs)
        self.policy.eval()
        with torch.no_grad():
            out = self.policy.forward(obs)
        return aggregate_node_attention(
            out["attention"], from_node=0, layer_agg=self.config.aggregate_layers
        )

    def evaluate_decision(
        self,
        obs: dict[str, torch.Tensor],
        action: int | None = None,
    ) -> DecisionFaithfulness:
        """Compute the full DEF/WAMSN bundle for a single-decision obs (B=1)."""
        self._check_batch_one(obs)
        self.policy.eval()

        # --- baseline forward: get action, π(a*), attention, mask
        with torch.no_grad():
            out = self.policy.forward(obs)
            logits = out["logits"]                       # (1, K+1)
            probs = torch.softmax(logits, dim=-1)        # (1, K+1)
            attention = out["attention"]                 # (L, 1, H, N, N)
            node_mask = out["node_mask"][0].cpu().numpy().astype(bool)

        if action is None:
            action = int(probs.argmax(dim=-1).item())
        pi_full = float(probs[0, action].item())

        attention_row = aggregate_node_attention(
            attention, from_node=0, layer_agg=self.config.aggregate_layers
        )

        K_n = self.policy.config.k_neighbors
        K_r = self.policy.config.k_reservations
        N = 1 + K_n + K_r

        # non-self valid nodes = ablation candidates
        non_self_valid = np.array(
            [i for i in range(1, N) if node_mask[i]], dtype=np.int64
        )

        # --- DEF, per k
        per_k: dict[int, dict[str, float]] = {}
        for k in self.config.top_k_values:
            if k > len(non_self_valid):
                continue

            # Top-k by attention among the valid non-self nodes
            attn_at_valid = attention_row[non_self_valid]
            order = np.argsort(-attn_at_valid)  # descending
            top_k_idx = non_self_valid[order[:k]]

            comp_k = self._comp(obs, top_k_idx, action, pi_full)
            suff_k = self._suff(obs, top_k_idx, action, pi_full)

            comp_rand_k = 0.0
            suff_rand_k = 0.0
            for _ in range(self.config.n_random_baselines):
                rand_idx = self._rng.choice(non_self_valid, size=k, replace=False)
                comp_rand_k += self._comp(obs, rand_idx, action, pi_full)
                suff_rand_k += self._suff(obs, rand_idx, action, pi_full)
            n_rand = max(1, self.config.n_random_baselines)
            comp_rand_k /= n_rand
            suff_rand_k /= n_rand

            per_k[k] = {
                "comp": comp_k,
                "suff": suff_k,
                "comp_rand": comp_rand_k,
                "suff_rand": suff_rand_k,
                "g_comp": comp_k - comp_rand_k,
                "g_suff": suff_rand_k - suff_k,
            }

        if per_k:
            comp = float(np.mean([p["comp"] for p in per_k.values()]))
            suff = float(np.mean([p["suff"] for p in per_k.values()]))
            comp_rand = float(np.mean([p["comp_rand"] for p in per_k.values()]))
            suff_rand = float(np.mean([p["suff_rand"] for p in per_k.values()]))
            g_comp = float(np.mean([p["g_comp"] for p in per_k.values()]))
            g_suff = float(np.mean([p["g_suff"] for p in per_k.values()]))
            def_score = 0.5 * (g_comp + g_suff)
        else:
            comp = suff = comp_rand = suff_rand = 0.0
            g_comp = g_suff = def_score = 0.0

        # --- WAMSN
        aoi_per_node = self._extract_aoi_per_node(obs, K_n, K_r)
        node_is_vehicle = np.zeros(N, dtype=bool)
        node_is_vehicle[0] = True                        # self
        node_is_vehicle[1:1 + K_n] = True                # taxi slots
        node_is_vehicle &= node_mask                     # keep only valid
        wamsn = compute_wamsn(
            attention_row, aoi_per_node, node_is_vehicle, self.config.aoi_max_s
        )

        return DecisionFaithfulness(
            action=action,
            pi_full=pi_full,
            def_score=float(def_score),
            g_comp=float(g_comp),
            g_suff=float(g_suff),
            comp=float(comp),
            suff=float(suff),
            comp_rand=float(comp_rand),
            suff_rand=float(suff_rand),
            wamsn=float(wamsn),
            attention_row=attention_row,
            node_mask=node_mask,
            per_k=per_k,
        )

    # -------------------------------------------------- internals

    def _check_batch_one(self, obs: dict[str, torch.Tensor]) -> None:
        first = next(iter(obs.values()))
        if first.shape[0] != 1:
            raise ValueError(
                f"FaithfulnessEvaluator requires per-decision obs (B=1); got B={first.shape[0]}. "
                "Slice the batched obs to one agent before calling."
            )

    def _forward_prob(self, obs: dict[str, torch.Tensor], action: int) -> float:
        with torch.no_grad():
            out = self.policy.forward(obs)
            probs = torch.softmax(out["logits"], dim=-1)
        return float(probs[0, action].item())

    def _comp(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
        action: int,
        pi_full: float,
    ) -> float:
        """π(a*|G) − π(a*|G\\R_k)."""
        obs_ablated = self._mask_out(obs, node_idx)
        pi_ablated = self._forward_prob(obs_ablated, action)
        return pi_full - pi_ablated

    def _suff(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
        action: int,
        pi_full: float,
    ) -> float:
        """π(a*|G) − π(a*|R_k)."""
        obs_kept = self._keep_only(obs, node_idx)
        pi_kept = self._forward_prob(obs_kept, action)
        return pi_full - pi_kept

    def _mask_out(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Return a shallow copy of obs with the listed node slots masked to 0."""
        K_n = self.policy.config.k_neighbors
        new_obs = {k: (v.clone() if k.endswith("mask") else v) for k, v in obs.items()}
        for idx in node_idx:
            key, slot = _node_idx_to_slot(int(idx), K_n)
            if key is None:
                continue  # never mask self
            new_obs[key][0, slot] = 0
        return new_obs

    def _keep_only(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Return a shallow copy of obs with ONLY self and the listed nodes kept."""
        new_obs = {k: (v.clone() if k.endswith("mask") else v) for k, v in obs.items()}
        # Wipe all non-self masks first.
        new_obs["neighbor_taxis_mask"] = torch.zeros_like(obs["neighbor_taxis_mask"])
        new_obs["reservations_mask"] = torch.zeros_like(obs["reservations_mask"])
        K_n = self.policy.config.k_neighbors
        for idx in node_idx:
            key, slot = _node_idx_to_slot(int(idx), K_n)
            if key is None:
                continue
            new_obs[key][0, slot] = 1
        return new_obs

    def _extract_aoi_per_node(
        self,
        obs: dict[str, torch.Tensor],
        K_n: int,
        K_r: int,
    ) -> np.ndarray:
        """(N,) AoI in seconds — reversed from the normalised values in obs."""
        N = 1 + K_n + K_r
        aoi = np.zeros(N, dtype=np.float64)
        # Self AoI is at self_feat index 4, normalised by AOI_MAX_S.
        aoi[0] = float(obs["self"][0, 4].item()) * self.config.aoi_max_s
        # Neighbour AoI is at neighbor_taxis[..., 4], same normalisation.
        for i in range(K_n):
            aoi[1 + i] = float(obs["neighbor_taxis"][0, i, 4].item()) * self.config.aoi_max_s
        # Reservations have no AoI — leave at 0. They are excluded from
        # WAMSN via `node_is_vehicle`, so this value never contributes.
        return aoi


# ------------------------------------------------------------ helpers


def _node_idx_to_slot(idx: int, k_neighbors: int) -> tuple[str | None, int]:
    """Map a graph-node index to the (obs_key, slot) it lives in.

    Node 0 = self → (None, -1)   (untouchable)
    Node 1..K_n = neighbour taxis → ("neighbor_taxis_mask", idx-1)
    Node K_n+1..N-1 = reservations → ("reservations_mask", idx-1-K_n)
    """
    if idx == 0:
        return None, -1
    if idx <= k_neighbors:
        return "neighbor_taxis_mask", idx - 1
    return "reservations_mask", idx - 1 - k_neighbors
