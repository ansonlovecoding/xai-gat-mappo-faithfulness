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

- **Node indexing.** The per-agent heterogeneous graph uses this ordering:
  node 0 = self, nodes 1..K_n = neighbour taxis, nodes K_n+1..N-1 =
  candidate reservations. `self` is never ablated — removing it would
  destroy the decision context.

- **Attention aggregation.** The declared default is the mean over layers
  and heads of `attention[l, 0, h, 0, j]` — the attention weight from self
  (node 0) to node j. Layer/head alternatives and residual attention rollout
  are exposed for sensitivity analysis without changing the default.

- **Random baseline.** For each `k`, sample `n_random_baselines` random
  size-`k` subsets from the *valid non-self* nodes and average their
  Comp/Suff. Configurable via `FaithfulnessConfig`.

- **Batching.** The default implementation does one policy forward per
  counterfactual (~36 forwards per decision at default settings). On the
  tested hardware this remains practical (< 100 ms per decision on
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
    # Clamp for the logit-margin metric. Margins hit ±inf when an ablation
    # masks the chosen action out of the candidate set (its logit → -inf)
    # or leaves it unopposed; the cap keeps DEF_margin finite and bounds a
    # single decision's influence on the mean.
    margin_cap: float = 10.0
    # How to reduce layers to per-node attention.
    aggregate_layers: str = "mean"
    # How to reduce attention heads. "mean", "max", or "head_<index>".
    aggregate_heads: str = "mean"
    # Deterministic random-baseline sampling.
    seed: int = 42
    # Construct-validity audit (P2): ALSO compute the exclusion-variant DEF,
    # where the chosen action's reservation node is protected — never
    # occluded in Comp, always kept in Suff, and excluded from both top-k
    # and random candidate sets. Isolates how much of DEF comes from the
    # "occlusion = action deletion" mechanical asymmetry (masking the
    # chosen reservation removes the action itself and slams the margin to
    # −cap). Roughly doubles the counterfactual forwards per decision.
    exclusion_variant: bool = False
    # Random-baseline sampling scheme. "uniform" (legacy) draws size-k
    # subsets uniformly from all valid non-self
    # nodes. "type_matched" draws subsets with the SAME taxi/reservation
    # composition as the attention top-k set — the P2 control for the
    # no-op artifact, where uniform draws hit reservation nodes (and
    # thereby delete competing actions, clamping the margin) far more
    # often than the attention top-k does.
    random_baseline: str = "type_matched"


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
    vehicle_mask: np.ndarray          # valid self and neighbour-taxi nodes
    stale_vehicle_mask: np.ndarray    # binary stale status; no AoI weighting
    # Logit-margin variants of the DEF terms. Probability-based DEF loses
    # signal when the policy saturates (entropy → 0 makes π(a*) ≈ 1
    # insensitive to occlusion); the margin logit[a*] − max_other keeps
    # moving. Computed from the SAME counterfactual forwards, so free.
    # Units are logits (clamped to ±margin_cap), not probabilities —
    # comparable within a checkpoint, not across differently-scaled ones.
    m_full: float = 0.0               # margin on the full graph
    def_margin: float = 0.0           # ½ (g_comp_m + g_suff_m)
    g_comp_m: float = 0.0
    g_suff_m: float = 0.0
    # P2 construct-validity audit fields.
    clamp_topk_frac: float = 0.0      # frac. of top-k forwards hitting ±cap
    clamp_rand_frac: float = 0.0      # frac. of random-baseline forwards, same
    def_excl: float = float("nan")    # DEF with chosen-reservation node protected
    def_m_excl: float = float("nan")
    g_comp_excl: float = float("nan")
    g_suff_excl: float = float("nan")
    g_comp_m_excl: float = float("nan")
    g_suff_m_excl: float = float("nan")
    excl_evaluated: bool = False
    n_stale_vehicle: int = 0          # stale vehicle nodes visible this decision
    max_aoi_s: float = 0.0            # max AoI (s) over visible vehicle nodes
    stale_in_top3: bool = False       # a stale node made the explanation's top-3
    stale_attention_mass: float = 0.0   # total graph attention on stale vehicles
    stale_attention_share: float = 0.0  # share of vehicle attention on stale nodes
    per_k: dict[int, dict[str, float]] = field(default_factory=dict)


@dataclass
class LeaveOneOutImportance:
    """Per-node decision effects from masking one graph node at a time.

    ``margin_effect`` and ``probability_effect`` are positive when removing a
    node weakens the selected action. ``importance_row`` is the non-negative
    margin effect used to rank nodes as a perturbation-based positive control.
    Invalid nodes, self, and an optionally protected chosen-request node are 0.
    """

    action: int
    candidates: np.ndarray
    importance_row: np.ndarray
    margin_effect: np.ndarray
    probability_effect: np.ndarray


# ------------------------------------------------------------ pure functions


def aggregate_node_attention(
    attention: torch.Tensor,
    from_node: int = 0,
    layer_agg: str = "mean",
    head_agg: str = "mean",
) -> np.ndarray:
    """Reduce a raw GAT attention tensor to a single per-node importance row.

    Args:
        attention: shape (L, B, H, N, N) as returned by `policy.forward()`.
                   B must equal 1 (single decision).
        from_node: which row to read. Default 0 = self.
        layer_agg: "mean", "first", "last", or attention "rollout".
        head_agg: "mean", "max", or one head such as "head_0".

    Returns:
        (N,) numpy array. Sums to ~1 (softmax rows averaged).
    """
    if attention.dim() != 5:
        raise ValueError(f"expected attention with shape (L,B,H,N,N), got {attention.shape}")
    L, B, H, N, _ = attention.shape
    if B != 1:
        raise ValueError(f"aggregate_node_attention expects B=1 (single decision), got {B}")

    matrices = attention[:, 0]  # (L, H, N, N)
    if head_agg == "mean":
        matrices = matrices.mean(dim=1)
    elif head_agg == "max":
        matrices = matrices.max(dim=1).values
    elif head_agg.startswith("head_"):
        try:
            head_idx = int(head_agg.removeprefix("head_"))
        except ValueError as exc:
            raise ValueError(f"unknown head_agg: {head_agg!r}") from exc
        if not 0 <= head_idx < H:
            raise ValueError(f"head index {head_idx} outside [0, {H})")
        matrices = matrices[:, head_idx]
    else:
        raise ValueError(f"unknown head_agg: {head_agg!r}")

    if layer_agg == "mean":
        row = matrices[:, from_node, :].mean(dim=0)
    elif layer_agg == "first":
        row = matrices[0, from_node, :]
    elif layer_agg == "last":
        row = matrices[-1, from_node, :]
    elif layer_agg == "rollout":
        identity = torch.eye(N, dtype=matrices.dtype, device=matrices.device)
        joint = identity
        for layer_matrix in matrices:
            augmented = layer_matrix + identity
            augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            joint = augmented @ joint
        row = joint[from_node]
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


def compute_stale_attention_share(
    attention_row: np.ndarray,
    stale_vehicle_mask: np.ndarray,
    vehicle_mask: np.ndarray,
) -> float:
    """Share of vehicle-node attention assigned to stale vehicle nodes.

    This binary exposure metric deliberately ignores AoI magnitude. It is the
    direct measure for the claim that degradation shifts attention toward
    stale nodes; WAMSN remains a secondary, severity-weighted description.
    """
    attn = np.asarray(attention_row, dtype=np.float64)
    stale = np.asarray(stale_vehicle_mask, dtype=bool)
    vehicle = np.asarray(vehicle_mask, dtype=bool)
    denominator = float(attn[vehicle].sum())
    if denominator <= 0.0:
        return 0.0
    return float(attn[stale & vehicle].sum() / denominator)


def compute_stale_attention_mass(
    attention_row: np.ndarray,
    stale_vehicle_mask: np.ndarray,
) -> float:
    """Absolute attention mass assigned to binary-stale vehicle nodes."""
    attn = np.asarray(attention_row, dtype=np.float64)
    stale = np.asarray(stale_vehicle_mask, dtype=bool)
    return float(attn[stale].sum())


def expected_type_matched_topk_overlap(
    top_k_idx: np.ndarray,
    candidates: np.ndarray,
    k_neighbors: int,
) -> float:
    """Expected top-k overlap with a type-matched random subset.

    The random subset contains the same numbers of taxi and request nodes as
    ``top_k_idx``. The result is the expected intersection size divided by k.
    It is exact, so the overlap diagnostic is not affected by Monte Carlo
    sampling noise.
    """
    top = np.asarray(top_k_idx, dtype=np.int64)
    pool = np.asarray(candidates, dtype=np.int64)
    if top.size == 0:
        return 0.0
    if np.unique(top).size != top.size or not np.isin(top, pool).all():
        raise ValueError("top_k_idx must contain unique members of candidates")

    top_taxi = int((top <= k_neighbors).sum())
    top_request = int(top.size - top_taxi)
    pool_taxi = int((pool <= k_neighbors).sum())
    pool_request = int(pool.size - pool_taxi)
    expected_intersection = 0.0
    if top_taxi:
        if pool_taxi < top_taxi:
            raise ValueError("not enough taxi candidates for type matching")
        expected_intersection += top_taxi * top_taxi / pool_taxi
    if top_request:
        if pool_request < top_request:
            raise ValueError("not enough request candidates for type matching")
        expected_intersection += top_request * top_request / pool_request
    return float(expected_intersection / top.size)


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties.

    Returns NaN when fewer than two values are supplied or either ranking is
    constant. Keeping this helper local avoids making SciPy a runtime
    dependency for the experiment scripts.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("x and y must have the same number of values")
    if a.size < 2:
        return float("nan")

    def _average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    rank_a = _average_ranks(a)
    rank_b = _average_ranks(b)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def decision_margin_ex(
    logits: torch.Tensor, action: int, cap: float
) -> tuple[float, bool]:
    """(margin, clamped) — margin of `action` over its best alternative.

    Handles the two ablation edge cases explicitly:
    - the chosen action was masked out of the candidate set (its logit is
      -inf after occlusion) → −cap: the ablation destroyed the decision;
    - every alternative is masked (unopposed decision) → +cap.
    `clamped` is True in both edge cases and whenever |raw margin| ≥ cap —
    the P2 construct-validity audit counts these to quantify how much of
    DEF is carried by the action-deletion mechanism rather than by
    information relevance.
    """
    arr = logits.detach().cpu().numpy().astype(np.float64).ravel()
    a = arr[action]
    others = np.delete(arr, action)
    others = others[np.isfinite(others)]
    if not np.isfinite(a):
        return -cap, True
    if others.size == 0:
        return cap, True
    raw = a - others.max()
    return float(np.clip(raw, -cap, cap)), bool(abs(raw) >= cap)


def decision_margin(logits: torch.Tensor, action: int, cap: float) -> float:
    """Margin only (back-compat wrapper around decision_margin_ex)."""
    return decision_margin_ex(logits, action, cap)[0]


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

    def attention_row(
        self,
        obs: dict[str, torch.Tensor],
        from_node: int = 0,
    ) -> np.ndarray:
        """Aggregated per-node attention row for a single-decision obs (B=1).

        One plain forward — used by the attention-drift pipeline to score a
        counterfactual (e.g. clean-twin) obs without the full DEF machinery.
        """
        self._check_batch_one(obs)
        self.policy.eval()
        with torch.no_grad():
            out = self.policy.forward(obs)
        return aggregate_node_attention(
            out["attention"], from_node=from_node,
            layer_agg=self.config.aggregate_layers,
            head_agg=self.config.aggregate_heads,
        )

    def attention_rows(
        self,
        obs: dict[str, torch.Tensor],
        aggregations: list[tuple[str, str]],
    ) -> dict[str, np.ndarray]:
        """Compute several attention aggregations from one policy forward."""
        self._check_batch_one(obs)
        self.policy.eval()
        with torch.no_grad():
            attention = self.policy.forward(obs)["attention"]
        return {
            f"{layer_agg}_{head_agg}": aggregate_node_attention(
                attention,
                from_node=0,
                layer_agg=layer_agg,
                head_agg=head_agg,
            )
            for layer_agg, head_agg in aggregations
        }

    def leave_one_out_importance(
        self,
        obs: dict[str, torch.Tensor],
        action: int | None = None,
        protect_chosen_request: bool = True,
    ) -> LeaveOneOutImportance:
        """Measure each node's effect on the selected action by masking it.

        This is a positive control for the DEF implementation: a ranking built
        directly from the same single-node perturbation should outperform a
        random ranking when the evaluator can detect influential information.
        It is not an independent explanation method and must not be presented
        as proof that attention itself is faithful.

        When the selected action is a request, that request node is protected
        by default. Otherwise masking it also removes the action from the
        action space, creating a mechanical effect rather than an information
        relevance effect.
        """
        self._check_batch_one(obs)
        self.policy.eval()
        with torch.no_grad():
            out = self.policy.forward(obs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            node_mask = out["node_mask"][0].cpu().numpy().astype(bool)

        if action is None:
            action = int(probs.argmax(dim=-1).item())
        pi_full = float(probs[0, action].item())
        m_full = decision_margin(logits[0], action, self.config.margin_cap)

        k_neighbors = self.policy.config.k_neighbors
        n_nodes = 1 + k_neighbors + self.policy.config.k_reservations
        candidates = np.flatnonzero(node_mask).astype(np.int64)
        candidates = candidates[candidates != 0]
        if protect_chosen_request and action >= 1:
            chosen_node = k_neighbors + action
            candidates = candidates[candidates != chosen_node]

        margin_effect = np.zeros(n_nodes, dtype=np.float64)
        probability_effect = np.zeros(n_nodes, dtype=np.float64)
        for node_idx in candidates:
            pi, margin, _ = self._forward_stats(
                self._mask_out(obs, np.array([node_idx], dtype=np.int64)), action
            )
            probability_effect[node_idx] = pi_full - pi
            margin_effect[node_idx] = m_full - margin

        return LeaveOneOutImportance(
            action=action,
            candidates=candidates,
            importance_row=np.maximum(margin_effect, 0.0),
            margin_effect=margin_effect,
            probability_effect=probability_effect,
        )

    def gradient_x_input_importance(
        self,
        obs: dict[str, torch.Tensor],
        action: int,
    ) -> np.ndarray:
        """Node ranking from the L2 norm of Gradient x Input.

        Gradients are taken from the selected action logit to the raw feature
        tensors. Masks remain fixed. This provides a conventional post-hoc
        comparator that does not use the returned attention coefficients.
        """
        self._check_batch_one(obs)
        feature_keys = ("self", "neighbor_taxis", "reservations")
        grad_obs: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            cloned = value.detach().clone()
            if key in feature_keys:
                cloned.requires_grad_(True)
            grad_obs[key] = cloned
        grad_inputs = [grad_obs[key] for key in feature_keys]

        self.policy.eval()
        logits = self.policy.forward(grad_obs)["logits"]
        gradients = torch.autograd.grad(
            logits[0, action], grad_inputs, retain_graph=False, create_graph=False
        )
        grad_by_key = dict(zip(feature_keys, gradients, strict=True))

        k_neighbors = self.policy.config.k_neighbors
        k_reservations = self.policy.config.k_reservations
        importance = np.zeros(1 + k_neighbors + k_reservations, dtype=np.float64)
        self_score = torch.linalg.vector_norm(
            grad_by_key["self"] * grad_obs["self"], dim=-1
        )
        taxi_score = torch.linalg.vector_norm(
            grad_by_key["neighbor_taxis"] * grad_obs["neighbor_taxis"], dim=-1
        )
        request_score = torch.linalg.vector_norm(
            grad_by_key["reservations"] * grad_obs["reservations"], dim=-1
        )
        importance[0] = float(self_score[0].detach().cpu().item())
        importance[1:1 + k_neighbors] = taxi_score[0].detach().cpu().numpy()
        importance[1 + k_neighbors:] = request_score[0].detach().cpu().numpy()

        taxi_mask = obs["neighbor_taxis_mask"][0].detach().cpu().numpy().astype(bool)
        request_mask = obs["reservations_mask"][0].detach().cpu().numpy().astype(bool)
        importance[1:1 + k_neighbors][~taxi_mask] = 0.0
        importance[1 + k_neighbors:][~request_mask] = 0.0
        return importance

    def evaluate_decision(
        self,
        obs: dict[str, torch.Tensor],
        action: int | None = None,
        importance_row: np.ndarray | None = None,
    ) -> DecisionFaithfulness:
        """Compute the full DEF/WAMSN bundle for a single-decision obs (B=1).

        By default the explanation being scored is the coupled channel (the
        GAT attention row). Pass `importance_row` — any (N,) non-negative
        per-node importance — to score a different explanation over the
        same decision, for example a separate importance method. Top-k selection
        AND WAMSN then use that row; `attention_row` in the result stays
        the raw attention (so drift remains well-defined either way).
        """
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
        m_full = decision_margin(logits[0], action, self.config.margin_cap)

        attention_row = aggregate_node_attention(
            attention, from_node=0,
            layer_agg=self.config.aggregate_layers,
            head_agg=self.config.aggregate_heads,
        )
        scored_row = attention_row if importance_row is None else \
            np.asarray(importance_row, dtype=np.float64)

        K_n = self.policy.config.k_neighbors
        K_r = self.policy.config.k_reservations
        N = 1 + K_n + K_r

        # non-self valid nodes = ablation candidates
        non_self_valid = np.array(
            [i for i in range(1, N) if node_mask[i]], dtype=np.int64
        )

        # --- DEF over the full candidate set (the standard protocol)
        main = self._def_bundle(obs, action, pi_full, m_full,
                                non_self_valid, scored_row)
        per_k = main["per_k"]
        comp, suff = main["comp"], main["suff"]
        comp_rand, suff_rand = main["comp_rand"], main["suff_rand"]
        g_comp, g_suff, def_score = main["g_comp"], main["g_suff"], main["def_score"]
        g_comp_m, g_suff_m, def_margin = (main["g_comp_m"], main["g_suff_m"],
                                          main["def_margin"])
        cl = main["clamp"]
        clamp_topk_frac = (cl["topk_hits"] / cl["topk_total"]
                           if cl["topk_total"] else 0.0)
        clamp_rand_frac = (cl["rand_hits"] / cl["rand_total"]
                           if cl["rand_total"] else 0.0)

        # --- exclusion variant (construct-validity audit): the chosen
        # action's reservation node is protected from occlusion entirely.
        def_excl = def_m_excl = float("nan")
        g_comp_excl = g_suff_excl = float("nan")
        g_comp_m_excl = g_suff_m_excl = float("nan")
        excl_evaluated = False
        if self.config.exclusion_variant and action >= 1:
            chosen_node = K_n + action  # node index of the chosen reservation
            reduced = non_self_valid[non_self_valid != chosen_node]
            if chosen_node in non_self_valid and len(reduced) >= 1:
                excl = self._def_bundle(obs, action, pi_full, m_full,
                                        reduced, scored_row,
                                        protected=chosen_node)
                if excl["per_k"]:
                    def_excl = excl["def_score"]
                    def_m_excl = excl["def_margin"]
                    g_comp_excl = excl["g_comp"]
                    g_suff_excl = excl["g_suff"]
                    g_comp_m_excl = excl["g_comp_m"]
                    g_suff_m_excl = excl["g_suff_m"]
                    excl_evaluated = True

        # --- WAMSN
        aoi_per_node = self._extract_aoi_per_node(obs, K_n, K_r)
        node_is_vehicle = np.zeros(N, dtype=bool)
        node_is_vehicle[0] = True                        # self
        node_is_vehicle[1:1 + K_n] = True                # taxi slots
        node_is_vehicle &= node_mask                     # keep only valid
        wamsn = compute_wamsn(
            scored_row, aoi_per_node, node_is_vehicle, self.config.aoi_max_s
        )

        # --- staleness context (P2: conditional-WAMSN and AoI-distribution
        # reporting need to know whether stale nodes were even present).
        stale_mask = (aoi_per_node > 0) & node_is_vehicle
        n_stale_vehicle = int(stale_mask.sum())
        max_aoi_s = float(aoi_per_node[node_is_vehicle].max()) if node_is_vehicle.any() else 0.0
        top3 = non_self_valid[np.argsort(-scored_row[non_self_valid])[:3]] \
            if len(non_self_valid) else np.array([], dtype=np.int64)
        stale_in_top3 = bool(stale_mask[top3].any()) if len(top3) else False
        stale_attention_share = compute_stale_attention_share(
            scored_row, stale_mask, node_is_vehicle
        )
        stale_attention_mass = compute_stale_attention_mass(scored_row, stale_mask)

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
            vehicle_mask=node_is_vehicle,
            stale_vehicle_mask=stale_mask,
            m_full=float(m_full),
            def_margin=float(def_margin),
            g_comp_m=float(g_comp_m),
            g_suff_m=float(g_suff_m),
            clamp_topk_frac=float(clamp_topk_frac),
            clamp_rand_frac=float(clamp_rand_frac),
            def_excl=float(def_excl),
            def_m_excl=float(def_m_excl),
            g_comp_excl=float(g_comp_excl),
            g_suff_excl=float(g_suff_excl),
            g_comp_m_excl=float(g_comp_m_excl),
            g_suff_m_excl=float(g_suff_m_excl),
            excl_evaluated=excl_evaluated,
            n_stale_vehicle=n_stale_vehicle,
            max_aoi_s=max_aoi_s,
            stale_in_top3=stale_in_top3,
            stale_attention_mass=stale_attention_mass,
            stale_attention_share=stale_attention_share,
            per_k=per_k,
        )

    def evaluate_paired_decision(
        self,
        degraded_obs: dict[str, torch.Tensor],
        clean_obs: dict[str, torch.Tensor],
        *,
        action: int,
    ) -> tuple[DecisionFaithfulness, DecisionFaithfulness]:
        """Score degraded and clean-twin observations with identical controls.

        The evaluator's random-baseline stream advances exactly once. Restoring
        its state around the clean-twin call makes both sides use the same
        type-matched subsets without changing subsequent decision samples.
        """
        state_before = copy.deepcopy(self._rng.bit_generator.state)
        degraded = self.evaluate_decision(degraded_obs, action=action)
        state_after = copy.deepcopy(self._rng.bit_generator.state)
        self._rng.bit_generator.state = state_before
        try:
            clean = self.evaluate_decision(clean_obs, action=action)
        finally:
            self._rng.bit_generator.state = state_after
        return degraded, clean

    # -------------------------------------------------- internals

    def _check_batch_one(self, obs: dict[str, torch.Tensor]) -> None:
        first = next(iter(obs.values()))
        if first.shape[0] != 1:
            raise ValueError(
                f"FaithfulnessEvaluator requires per-decision obs (B=1); got B={first.shape[0]}. "
                "Slice the batched obs to one agent before calling."
            )

    def _forward_stats(
        self, obs: dict[str, torch.Tensor], action: int
    ) -> tuple[float, float, bool]:
        """One forward pass → (π(a*), decision margin, margin-clamped?).
        Both metrics share the same counterfactual forward."""
        with torch.no_grad():
            out = self.policy.forward(obs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
        pi = float(probs[0, action].item())
        margin, clamped = decision_margin_ex(
            logits[0], action, self.config.margin_cap)
        return pi, margin, clamped

    def _comp(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
        action: int,
        pi_full: float,
        m_full: float,
    ) -> tuple[float, float, bool]:
        """(π and margin) drops when the explanation's nodes are removed."""
        obs_ablated = self._mask_out(obs, node_idx)
        pi, m, clamped = self._forward_stats(obs_ablated, action)
        return pi_full - pi, m_full - m, clamped

    def _suff(
        self,
        obs: dict[str, torch.Tensor],
        node_idx: np.ndarray,
        action: int,
        pi_full: float,
        m_full: float,
        protected: int | None = None,
    ) -> tuple[float, float, bool]:
        """(π and margin) drops when ONLY the explanation's nodes are kept.
        A `protected` node (the chosen action's reservation, in the
        exclusion variant) is always kept visible alongside them."""
        keep = node_idx if protected is None else \
            np.concatenate([node_idx, np.array([protected])])
        obs_kept = self._keep_only(obs, keep)
        pi, m, clamped = self._forward_stats(obs_kept, action)
        return pi_full - pi, m_full - m, clamped

    def _def_bundle(
        self,
        obs: dict[str, torch.Tensor],
        action: int,
        pi_full: float,
        m_full: float,
        candidates: np.ndarray,
        scored_row: np.ndarray,
        protected: int | None = None,
    ) -> dict:
        """Full Comp/Suff/random-baseline computation over one candidate
        set. Returns per_k, aggregate gains for both metrics, and clamp
        tallies split by top-k vs random occlusion sets."""
        per_k: dict[int, dict[str, float]] = {}
        clamp = {"topk_hits": 0, "topk_total": 0, "rand_hits": 0, "rand_total": 0}
        for k in self.config.top_k_values:
            if k > len(candidates):
                continue
            order = np.argsort(-scored_row[candidates])
            top_k_idx = candidates[order[:k]]

            comp_k, comp_m_k, c1 = self._comp(obs, top_k_idx, action, pi_full, m_full)
            suff_k, suff_m_k, c2 = self._suff(obs, top_k_idx, action, pi_full,
                                              m_full, protected=protected)
            clamp["topk_hits"] += int(c1) + int(c2)
            clamp["topk_total"] += 2

            # Random-baseline pools for type-matched sampling (see config).
            K_n = self.policy.config.k_neighbors
            taxi_pool = candidates[candidates <= K_n]
            res_pool = candidates[candidates > K_n]
            n_res_topk = int((top_k_idx > K_n).sum())
            n_taxi_topk = k - n_res_topk
            type_matched = (
                self.config.random_baseline == "type_matched"
                and len(taxi_pool) >= n_taxi_topk and len(res_pool) >= n_res_topk
            )

            comp_rand_k = suff_rand_k = 0.0
            comp_m_rand_k = suff_m_rand_k = 0.0
            for _ in range(self.config.n_random_baselines):
                if type_matched:
                    parts = []
                    if n_taxi_topk:
                        parts.append(self._rng.choice(taxi_pool, size=n_taxi_topk,
                                                      replace=False))
                    if n_res_topk:
                        parts.append(self._rng.choice(res_pool, size=n_res_topk,
                                                      replace=False))
                    rand_idx = np.concatenate(parts)
                else:
                    rand_idx = self._rng.choice(candidates, size=k, replace=False)
                c, cm, r1 = self._comp(obs, rand_idx, action, pi_full, m_full)
                s, sm, r2 = self._suff(obs, rand_idx, action, pi_full, m_full,
                                       protected=protected)
                clamp["rand_hits"] += int(r1) + int(r2)
                clamp["rand_total"] += 2
                comp_rand_k += c
                suff_rand_k += s
                comp_m_rand_k += cm
                suff_m_rand_k += sm
            n_rand = max(1, self.config.n_random_baselines)
            comp_rand_k /= n_rand
            suff_rand_k /= n_rand
            comp_m_rand_k /= n_rand
            suff_m_rand_k /= n_rand

            per_k[k] = {
                "comp": comp_k,
                "suff": suff_k,
                "comp_rand": comp_rand_k,
                "suff_rand": suff_rand_k,
                "g_comp": comp_k - comp_rand_k,
                "g_suff": suff_rand_k - suff_k,
                "comp_m": comp_m_k,
                "suff_m": suff_m_k,
                "g_comp_m": comp_m_k - comp_m_rand_k,
                "g_suff_m": suff_m_rand_k - suff_m_k,
            }

        agg = {"per_k": per_k, "clamp": clamp}
        if per_k:
            for key in ("comp", "suff", "comp_rand", "suff_rand",
                        "g_comp", "g_suff", "g_comp_m", "g_suff_m"):
                agg[key] = float(np.mean([p[key] for p in per_k.values()]))
            agg["def_score"] = 0.5 * (agg["g_comp"] + agg["g_suff"])
            agg["def_margin"] = 0.5 * (agg["g_comp_m"] + agg["g_suff_m"])
        else:
            for key in ("comp", "suff", "comp_rand", "suff_rand", "g_comp",
                        "g_suff", "g_comp_m", "g_suff_m", "def_score",
                        "def_margin"):
                agg[key] = 0.0
        return agg

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
