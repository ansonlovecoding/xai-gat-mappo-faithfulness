"""Smoke test the faithfulness evaluation pipeline end-to-end.

Steps:
  1. Unit-test the three pure helpers on synthetic inputs where the
     expected value is known analytically.
  2. Build env + fresh-init policy (same setup as scripts/test_policy.py).
  3. Take one real decision, slice batched obs to B=1, run
     FaithfulnessEvaluator.evaluate_decision().
  4. Check shapes, ranges, and no NaN on the returned DecisionFaithfulness.

Fresh init = policy is bad. We are NOT measuring "does DEF go up over
training" here; we're only verifying the evaluator plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    aggregate_node_attention,
    compute_attention_drift,
    compute_wamsn,
)
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    PolicyConfig,
    obs_dict_to_tensors,
)


# ------------------------------------------------------------ unit tests


def test_pure_helpers() -> None:
    # --- compute_attention_drift ---
    row = np.array([0.5, 0.3, 0.2])
    assert compute_attention_drift(row, row) < 1e-9, "JS(p,p) must be 0"

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    js = compute_attention_drift(a, b)
    assert 0.99 < js <= 1.0, f"JS on disjoint one-hots should be ~1, got {js}"

    # --- compute_wamsn ---
    # All mass on a maximally-stale vehicle → WAMSN → 1.
    attn = np.array([0.0, 1.0, 0.0])
    aoi = np.array([0.0, 60.0, 0.0])
    veh = np.array([True, True, False])
    w = compute_wamsn(attn, aoi, veh, aoi_max=60.0)
    assert abs(w - 1.0) < 1e-9, f"expected 1.0, got {w}"

    # All mass on a fresh vehicle → WAMSN = 0.
    aoi_fresh = np.array([0.0, 0.0, 0.0])
    w = compute_wamsn(attn, aoi_fresh, veh, aoi_max=60.0)
    assert abs(w) < 1e-9, f"expected 0.0, got {w}"

    # All mass on a non-vehicle node → WAMSN = 0.
    attn_res = np.array([0.0, 0.0, 1.0])
    w = compute_wamsn(attn_res, aoi, veh, aoi_max=60.0)
    assert abs(w) < 1e-9, f"expected 0.0 when attn lands off vehicles, got {w}"

    # --- aggregate_node_attention ---
    # Fake (L=2, B=1, H=3, N=4, N=4) attention, all uniform → per-node row
    # should be uniform 1/N.
    L, B, H, N = 2, 1, 3, 4
    uniform = torch.full((L, B, H, N, N), 1.0 / N)
    row = aggregate_node_attention(uniform, from_node=0, layer_agg="mean")
    assert row.shape == (N,), row.shape
    assert np.allclose(row, np.full(N, 1.0 / N)), row

    print("  pure helpers   OK")


# ------------------------------------------------------------ end-to-end


def _slice_batch(obs: dict[str, torch.Tensor], i: int) -> dict[str, torch.Tensor]:
    """Return B=1 view of a single agent's row from a batched obs dict."""
    return {k: v[i : i + 1] for k, v in obs.items()}


def test_evaluator_end_to_end() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    env_cfg = DispatchEnvConfig(area="central_park")
    env = DispatchEnv(env_cfg)
    obs, _ = env.reset()

    pol_cfg = PolicyConfig(
        k_neighbors=env_cfg.k_neighbors,
        k_reservations=env_cfg.k_reservations,
        device=device,
    )
    policy = DispatchGATPolicy(pol_cfg)
    policy.eval()

    # Advance the sim until at least one acting agent has ≥1 valid
    # reservation — otherwise res_logits are all -inf, π puts 1.0 on the
    # no-op, and every ablation trivially returns 0. That still exercises
    # the plumbing, but hides bugs in the top-k / comp / suff code paths.
    batched: dict[str, torch.Tensor] = {}
    agents: list[str] = []
    picked_idx = -1
    for _ in range(200):
        if not obs:
            obs, *_ = env.step({})
            continue
        batched, agents = obs_dict_to_tensors(obs, device=device)
        # Find an agent with at least one valid reservation slot.
        res_valid = batched["reservations_mask"].sum(dim=1)  # (B,)
        candidates = (res_valid > 0).nonzero(as_tuple=True)[0]
        if candidates.numel() > 0:
            picked_idx = int(candidates[0].item())
            break
        # Cheap advance: no-op every acting agent.
        obs, *_ = env.step({a: 0 for a in agents})
    assert picked_idx >= 0, "no agent with any valid reservation in 200 steps"
    print(f"  agents in batch: {len(agents)}  picked agent idx: {picked_idx}  "
          f"(valid reservations: {int(batched['reservations_mask'][picked_idx].sum().item())})")

    single = _slice_batch(batched, picked_idx)
    evaluator = FaithfulnessEvaluator(policy, FaithfulnessConfig(seed=0))
    result = evaluator.evaluate_decision(single)

    # --- range checks ---
    assert 0.0 <= result.pi_full <= 1.0, f"pi_full out of range: {result.pi_full}"
    assert -1.0 - 1e-6 <= result.def_score <= 1.0 + 1e-6, f"DEF: {result.def_score}"
    assert 0.0 - 1e-6 <= result.wamsn <= 1.0 + 1e-6, f"WAMSN: {result.wamsn}"

    # --- shape / mask checks ---
    N = 1 + env_cfg.k_neighbors + env_cfg.k_reservations
    assert result.attention_row.shape == (N,), result.attention_row.shape
    assert result.node_mask.shape == (N,), result.node_mask.shape
    assert result.node_mask[0], "self node must always be valid"

    # --- attention row is a distribution (or all-masked → sum 0) ---
    s = result.attention_row.sum()
    assert abs(s - 1.0) < 1e-4 or abs(s) < 1e-4, f"attention_row sum: {s}"

    # --- NaN checks on every scalar field ---
    for name in ("pi_full", "def_score", "g_comp", "g_suff", "comp", "suff",
                 "comp_rand", "suff_rand", "wamsn"):
        v = getattr(result, name)
        assert np.isfinite(v), f"{name} is not finite: {v}"

    # --- per_k populated for at least one k (there should be ≥1 valid non-self node) ---
    assert len(result.per_k) >= 1, f"per_k empty; result: {result}"
    for k, d in result.per_k.items():
        for key in ("comp", "suff", "comp_rand", "suff_rand", "g_comp", "g_suff"):
            assert np.isfinite(d[key]), f"per_k[{k}][{key}] not finite"

    # --- drift on the same row should be 0 ---
    assert compute_attention_drift(result.attention_row, result.attention_row) < 1e-9

    print(
        f"  action={result.action}  pi_full={result.pi_full:.3f}  "
        f"DEF={result.def_score:+.3f}  (g_comp={result.g_comp:+.3f}, "
        f"g_suff={result.g_suff:+.3f})  WAMSN={result.wamsn:.3f}  "
        f"per_k={sorted(result.per_k.keys())}"
    )

    env.close()
    print("  evaluator      OK")


def main() -> int:
    print("test_faithfulness")
    print("  ---- pure helpers")
    test_pure_helpers()
    print("  ---- evaluator end-to-end")
    test_evaluator_end_to_end()
    print("OK — faithfulness pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
