"""Evaluate a trained GAT-MAPPO or MLP checkpoint.

Runs one or more evaluation episodes with the checkpointed policy and
reports pickups, reward, and wait time in a structured JSON result.

Deterministic by default (argmax over action logits). Pass --stochastic to
sample instead, which matches training-time behaviour.

Faithfulness: pass --faithfulness to compute DEF / WAMSN for every decision
(§7.4 metrics). Per-decision records go to `<ckpt>.faithfulness.jsonl` and
episode/run aggregates are folded into the `.eval.json` summary. Because
each decision requires ~36 counterfactual forwards, use --faithfulness-every
to sub-sample when running long episodes.

Usage:
  python scripts/eval_policy.py runs/mappo/central_park_<ts>/ckpt_epoch_0299.pt
  python scripts/eval_policy.py <ckpt> --episodes 5
  python scripts/eval_policy.py <ckpt> --episodes 5 --stochastic
  python scripts/eval_policy.py <ckpt> --degradation tunnel_triggered
  python scripts/eval_policy.py <ckpt> --faithfulness --faithfulness-every 5
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
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    compute_attention_drift,
    compute_stale_attention_mass,
    compute_stale_attention_share,
    derive_seed,
    expected_type_matched_topk_overlap,
    seed_everything,
    spearman_rank_correlation,
)
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    obs_dict_to_tensors,
)
from dispatch_marl.provenance import (  # noqa: E402
    atomic_write_json,
    runtime_provenance,
    sha256_file,
)
from dispatch_marl.runtime import (  # noqa: E402
    choose_device as _choose_device,
    load_policy as _load_policy,
)


def _has_stale_vehicle(obs: dict[str, torch.Tensor]) -> bool:
    """Return whether one batch-one observation contains a stale vehicle."""
    if float(obs["self"][0, 4].item()) > 0.0:
        return True
    taxi_aoi = obs["neighbor_taxis"][0, :, 4]
    taxi_valid = obs["neighbor_taxis_mask"][0].bool()
    return bool(((taxi_aoi > 0) & taxi_valid).any().item())


def _run_episode(
    env: DispatchEnv,
    policy: DispatchGATPolicy,
    device: str,
    stochastic: bool,
    faithfulness_evaluator: FaithfulnessEvaluator | None = None,
    faithfulness_every: int = 1,
    episode_index: int = 0,
    reset_options: dict | None = None,
    compute_drift: bool = False,
    action_seed: int | None = None,
    faithfulness_positive_control: bool = False,
    attention_aggregation_sensitivity: bool = False,
    attention_action_row_sensitivity: bool = False,
    faithfulness_request_actions_only: bool = False,
    faithfulness_exposed_every: int | None = None,
) -> tuple[dict, list[dict]]:
    """Run one eval episode; return (episode summary, per-decision faithfulness records).

    The faithfulness records list is empty unless `faithfulness_evaluator` is
    provided. Records are lightweight dicts (no attention arrays) suitable
    for JSONL streaming.

    With `compute_drift` (requires the env to be built with
    emit_clean_obs=True), each sampled decision also gets a "drift" field:
    JS divergence between the attention row on the degraded obs and on its
    clean twin. Trivially ~0 when degradation is off — a useful sanity check.
    """
    if action_seed is not None:
        seed_everything(action_seed)
    obs_dict, _ = env.reset(options=reset_options)
    total_reward = 0.0
    total_completed_journeys = 0
    step = 0
    last_wait = 0.0
    faith_records: list[dict] = []
    # A global decision counter controls sub-sampling, so the
    # cadence is consistent regardless of how many agents act on a given step.
    decision_counter = 0
    exposed_decision_counter = 0
    n_stale_exposed_seen = 0
    n_stale_exposed_scored = 0
    n_cadence_scored = 0
    # Empirical degradation rate — fraction of agent-obs pairs where
    # position_valid == 0. Used by the degradation ablation script to
    # calibrate the matched-rate random_dropout baseline.
    n_degraded_obs = 0
    n_total_obs = 0

    while not env.done:
        if obs_dict:
            batched, agents = obs_dict_to_tensors(obs_dict, device=device)
            # Track empirical degradation rate. position_valid is (B, 1) int8:
            # 1 = trustworthy obs, 0 = degraded.
            pv = batched["position_valid"]
            n_total_obs += int(pv.numel())
            n_degraded_obs += int((pv == 0).sum().item())
            with torch.no_grad():
                out = policy.forward(batched)
                logits = out["logits"]
                if stochastic:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                else:
                    action = logits.argmax(dim=-1)
            actions_np = action.cpu().numpy().astype(int)
            actions = {a: int(actions_np[i]) for i, a in enumerate(agents)}

            if faithfulness_evaluator is not None:
                # Snapshot the clean twins BEFORE env.step() rebuilds them.
                clean_obs_by_agent = dict(env.last_clean_obs) if compute_drift else {}
                for i, a in enumerate(agents):
                    single = {k: v[i : i + 1] for k, v in batched.items()}
                    stale_exposed = _has_stale_vehicle(single)
                    cadence_score = decision_counter % faithfulness_every == 0
                    exposure_score = False
                    if stale_exposed:
                        n_stale_exposed_seen += 1
                        if faithfulness_exposed_every is not None:
                            exposure_score = (
                                exposed_decision_counter
                                % faithfulness_exposed_every == 0
                            )
                        exposed_decision_counter += 1
                    should_score = cadence_score or exposure_score
                    if faithfulness_request_actions_only:
                        should_score = should_score and int(actions_np[i]) >= 1
                    if should_score:
                        n_valid_res = int(single["reservations_mask"].sum().item())
                        n_valid_taxi = int(
                            single["neighbor_taxis_mask"].sum().item()
                        )
                        clean_single = None
                        clean_twin_result = None
                        if compute_drift and a in clean_obs_by_agent:
                            clean_single = {
                                k: torch.as_tensor(
                                    v, dtype=torch.float32, device=device
                                ).unsqueeze(0)
                                for k, v in clean_obs_by_agent[a].items()
                            }
                        if stale_exposed and clean_single is not None:
                            result, clean_twin_result = (
                                faithfulness_evaluator.evaluate_paired_decision(
                                    single,
                                    clean_single,
                                    action=int(actions_np[i]),
                                )
                            )
                        else:
                            result = faithfulness_evaluator.evaluate_decision(
                                single, action=int(actions_np[i])
                            )
                        n_cadence_scored += int(cadence_score)
                        n_stale_exposed_scored += int(stale_exposed)
                        control_fields: dict = {}
                        if faithfulness_positive_control:
                            loo = faithfulness_evaluator.leave_one_out_importance(
                                single, action=result.action,
                                protect_chosen_request=True,
                            )
                            oracle = faithfulness_evaluator.evaluate_decision(
                                single,
                                action=result.action,
                                importance_row=loo.importance_row,
                            )
                            grad_x_input_row = (
                                faithfulness_evaluator.gradient_x_input_importance(
                                    single, action=result.action
                                )
                            )
                            grad_x_input = faithfulness_evaluator.evaluate_decision(
                                single,
                                action=result.action,
                                importance_row=grad_x_input_row,
                            )
                            k_neighbors = policy.config.k_neighbors
                            taxi_candidates = loo.candidates[
                                loo.candidates <= k_neighbors
                            ]
                            taxi_rho = spearman_rank_correlation(
                                result.attention_row[taxi_candidates],
                                loo.margin_effect[taxi_candidates],
                            )
                            overlap_values = []
                            agreement_values = []
                            overlap_by_k = {}
                            agreement_by_k = {}
                            for k in faithfulness_evaluator.config.top_k_values:
                                if k > len(loo.candidates):
                                    continue
                                attn_order = np.argsort(
                                    -result.attention_row[loo.candidates]
                                )
                                loo_order = np.argsort(
                                    -loo.importance_row[loo.candidates]
                                )
                                attn_top = loo.candidates[attn_order[:k]]
                                loo_top = loo.candidates[loo_order[:k]]
                                expected_overlap = expected_type_matched_topk_overlap(
                                    attn_top, loo.candidates, k_neighbors
                                )
                                agreement = len(set(attn_top) & set(loo_top)) / k
                                overlap_values.append(expected_overlap)
                                agreement_values.append(agreement)
                                overlap_by_k[str(k)] = float(expected_overlap)
                                agreement_by_k[str(k)] = float(agreement)
                            attention_def = (
                                result.def_m_excl if result.excl_evaluated
                                else result.def_margin
                            )
                            attention_g_comp = (
                                result.g_comp_m_excl if result.excl_evaluated
                                else result.g_comp_m
                            )
                            attention_g_suff = (
                                result.g_suff_m_excl if result.excl_evaluated
                                else result.g_suff_m
                            )
                            oracle_def = (
                                oracle.def_m_excl if oracle.excl_evaluated
                                else oracle.def_margin
                            )
                            oracle_g_comp = (
                                oracle.g_comp_m_excl if oracle.excl_evaluated
                                else oracle.g_comp_m
                            )
                            oracle_g_suff = (
                                oracle.g_suff_m_excl if oracle.excl_evaluated
                                else oracle.g_suff_m
                            )
                            grad_x_input_def = (
                                grad_x_input.def_m_excl
                                if grad_x_input.excl_evaluated
                                else grad_x_input.def_margin
                            )
                            grad_x_input_g_comp = (
                                grad_x_input.g_comp_m_excl
                                if grad_x_input.excl_evaluated
                                else grad_x_input.g_comp_m
                            )
                            grad_x_input_g_suff = (
                                grad_x_input.g_suff_m_excl
                                if grad_x_input.excl_evaluated
                                else grad_x_input.g_suff_m
                            )
                            control_fields = {
                                "attention_control_def_m": float(attention_def),
                                "attention_control_g_comp_m": float(attention_g_comp),
                                "attention_control_g_suff_m": float(attention_g_suff),
                                "loo_oracle_def_m": float(oracle_def),
                                "loo_oracle_g_comp_m": float(oracle_g_comp),
                                "loo_oracle_g_suff_m": float(oracle_g_suff),
                                "grad_x_input_def_m": float(grad_x_input_def),
                                "grad_x_input_g_comp_m": float(grad_x_input_g_comp),
                                "grad_x_input_g_suff_m": float(grad_x_input_g_suff),
                                "loo_control_gap_m": float(
                                    oracle_def - attention_def
                                ),
                                "loo_positive_nodes": int(
                                    (loo.importance_row[loo.candidates] > 0).sum()
                                ),
                                "taxi_attention_loo_spearman": (
                                    float(taxi_rho)
                                    if np.isfinite(taxi_rho) else None
                                ),
                                "attention_loo_topk_agreement": float(
                                    np.mean(agreement_values)
                                ) if agreement_values else None,
                                "expected_random_topk_overlap": float(
                                    np.mean(overlap_values)
                                ) if overlap_values else None,
                                "attention_loo_topk_agreement_by_k": agreement_by_k,
                                "expected_random_topk_overlap_by_k": overlap_by_k,
                            }
                        if attention_action_row_sensitivity and result.action >= 1:
                            action_query_node = (
                                policy.config.k_neighbors + result.action
                            )
                            action_row = faithfulness_evaluator.attention_row(
                                single, from_node=action_query_node
                            )
                            action_row_result = (
                                faithfulness_evaluator.evaluate_decision(
                                    single,
                                    action=result.action,
                                    importance_row=action_row,
                                )
                            )
                            self_def = (
                                result.def_m_excl if result.excl_evaluated
                                else result.def_margin
                            )
                            action_def = (
                                action_row_result.def_m_excl
                                if action_row_result.excl_evaluated
                                else action_row_result.def_margin
                            )
                            control_fields.update({
                                "action_query_node": action_query_node,
                                "self_row_request_def_m": float(self_def),
                                "action_row_request_def_m": float(action_def),
                                "action_row_request_gap_m": float(
                                    action_def - self_def
                                ),
                            })
                        primary_available = (
                            result.action == 0 or result.excl_evaluated
                        )
                        primary_variant = (
                            "standard_type_matched_no_op"
                            if result.action == 0
                            else "chosen_action_protected_dispatch"
                        )
                        primary_def = (
                            result.def_score if result.action == 0
                            else result.def_excl if result.excl_evaluated
                            else None
                        )
                        primary_def_m = (
                            result.def_margin if result.action == 0
                            else result.def_m_excl if result.excl_evaluated
                            else None
                        )
                        primary_g_comp = (
                            result.g_comp if result.action == 0
                            else result.g_comp_excl if result.excl_evaluated
                            else None
                        )
                        primary_g_suff = (
                            result.g_suff if result.action == 0
                            else result.g_suff_excl if result.excl_evaluated
                            else None
                        )
                        drift = None
                        top3_churn = None
                        stale_attention_clean = None
                        stale_attention_mass_clean = None
                        aggregation_shifts = None
                        if clean_single is not None:
                            clean_row = faithfulness_evaluator.attention_row(clean_single)
                            drift = compute_attention_drift(
                                clean_row, result.attention_row
                            )
                            # How many of the explanation's top-3 nodes changed
                            # identity vs the clean twin (self excluded).
                            deg_top3 = set(np.argsort(-result.attention_row[1:])[:3])
                            cln_top3 = set(np.argsort(-clean_row[1:])[:3])
                            top3_churn = 3 - len(deg_top3 & cln_top3)
                            stale_attention_clean = compute_stale_attention_share(
                                clean_row,
                                result.stale_vehicle_mask,
                                result.vehicle_mask,
                            )
                            stale_attention_mass_clean = compute_stale_attention_mass(
                                clean_row, result.stale_vehicle_mask
                            )
                            if attention_aggregation_sensitivity:
                                n_heads = policy.config.n_heads
                                aggregations = [
                                    (layer, head)
                                    for layer in ("first", "last")
                                    for head in (
                                        "mean", "max",
                                        *(f"head_{h}" for h in range(n_heads)),
                                    )
                                ]
                                aggregations.append(("rollout", "mean"))
                                degraded_rows = faithfulness_evaluator.attention_rows(
                                    single, aggregations
                                )
                                clean_rows = faithfulness_evaluator.attention_rows(
                                    clean_single, aggregations
                                )
                                aggregation_shifts = {
                                    name: float(
                                        compute_stale_attention_mass(
                                            degraded_rows[name],
                                            result.stale_vehicle_mask,
                                        )
                                        - compute_stale_attention_mass(
                                            clean_rows[name],
                                            result.stale_vehicle_mask,
                                        )
                                    )
                                    for name in degraded_rows
                                }
                        faith_records.append({
                            **({"drift": float(drift)} if drift is not None else {}),
                            "episode": episode_index,
                            "rl_step": step,
                            "agent": a,
                            "score_reason": (
                                "cadence+stale_exposure"
                                if cadence_score and exposure_score
                                else "stale_exposure" if exposure_score
                                else "cadence"
                            ),
                            "stale_exposed_precheck": stale_exposed,
                            "action": result.action,
                            "primary_metric_variant": primary_variant,
                            "primary_metric_available": primary_available,
                            "primary_def": primary_def,
                            "primary_def_m": primary_def_m,
                            "primary_g_comp": primary_g_comp,
                            "primary_g_suff": primary_g_suff,
                            "record_schema_version": 2,
                            "pi_full": float(result.pi_full),
                            "def": float(result.def_score),
                            "def_m": float(result.def_margin),
                            "m_full": float(result.m_full),
                            "g_comp": float(result.g_comp),
                            "g_suff": float(result.g_suff),
                            "comp": float(result.comp),
                            "suff": float(result.suff),
                            "wamsn": float(result.wamsn),
                            "valid_reservations": n_valid_res,
                            "valid_taxis": n_valid_taxi,
                            "valid_non_self_nodes": n_valid_taxi + n_valid_res,
                            "n_k_evaluated": len(result.per_k),
                            "clamp_topk": float(result.clamp_topk_frac),
                            "clamp_rand": float(result.clamp_rand_frac),
                            "n_stale_veh": result.n_stale_vehicle,
                            "max_aoi_s": float(result.max_aoi_s),
                            "stale_in_top3": result.stale_in_top3,
                            "stale_attention_share": float(
                                result.stale_attention_share
                            ),
                            "stale_attention_mass": float(
                                result.stale_attention_mass
                            ),
                            **({
                                "stale_attention_share_clean_twin": float(
                                    stale_attention_clean
                                ),
                                "stale_attention_shift": float(
                                    result.stale_attention_mass
                                    - stale_attention_mass_clean
                                ),
                            } if stale_attention_clean is not None else {}),
                            **({"def_excl": float(result.def_excl),
                                "def_m_excl": float(result.def_m_excl)}
                               if result.excl_evaluated else {}),
                            **({
                                "def_clean_twin": float(
                                    clean_twin_result.def_score
                                ),
                                "def_m_clean_twin": float(
                                    clean_twin_result.def_margin
                                ),
                                "paired_def_delta": float(
                                    result.def_score
                                    - clean_twin_result.def_score
                                ),
                                "paired_def_m_delta": float(
                                    result.def_margin
                                    - clean_twin_result.def_margin
                                ),
                                "primary_def_clean_twin": (
                                    clean_twin_result.def_score
                                    if result.action == 0
                                    else clean_twin_result.def_excl
                                    if clean_twin_result.excl_evaluated
                                    else None
                                ),
                                "primary_def_m_clean_twin": (
                                    clean_twin_result.def_margin
                                    if result.action == 0
                                    else clean_twin_result.def_m_excl
                                    if clean_twin_result.excl_evaluated
                                    else None
                                ),
                                "paired_primary_def_delta": (
                                    float(
                                        primary_def
                                        - (
                                            clean_twin_result.def_score
                                            if result.action == 0
                                            else clean_twin_result.def_excl
                                        )
                                    )
                                    if primary_available and (
                                        result.action == 0
                                        or clean_twin_result.excl_evaluated
                                    ) else None
                                ),
                                "paired_primary_def_m_delta": (
                                    float(
                                        primary_def_m
                                        - (
                                            clean_twin_result.def_margin
                                            if result.action == 0
                                            else clean_twin_result.def_m_excl
                                        )
                                    )
                                    if primary_available and (
                                        result.action == 0
                                        or clean_twin_result.excl_evaluated
                                    ) else None
                                ),
                                **({
                                    "def_excl_clean_twin": float(
                                        clean_twin_result.def_excl
                                    ),
                                    "def_m_excl_clean_twin": float(
                                        clean_twin_result.def_m_excl
                                    ),
                                    "paired_def_excl_delta": float(
                                        result.def_excl
                                        - clean_twin_result.def_excl
                                    ),
                                    "paired_def_m_excl_delta": float(
                                        result.def_m_excl
                                        - clean_twin_result.def_m_excl
                                    ),
                                } if result.excl_evaluated
                                and clean_twin_result.excl_evaluated else {}),
                            } if clean_twin_result is not None else {}),
                            **({"top3_churn": top3_churn}
                               if top3_churn is not None else {}),
                            **control_fields,
                            **({"aggregation_stale_attention_shift": aggregation_shifts}
                               if aggregation_shifts is not None else {}),
                        })
                    decision_counter += 1
        else:
            actions = {}

        obs_dict, _, _, _, _ = env.step(actions)
        metrics = env.last_step_metrics
        total_reward += metrics.team_reward
        total_completed_journeys += metrics.completed_passenger_journeys
        last_wait = metrics.mean_pending_wait_s
        step += 1

    summary = {
        "completed_passenger_journeys": total_completed_journeys,
        # Protocol-v1 compatibility alias.
        "total_pickups": total_completed_journeys,
        "total_reward": float(total_reward),
        "final_mean_pending_wait_s": float(last_wait),
        "rl_steps": step,
        "empirical_degradation_rate": (
            float(n_degraded_obs / n_total_obs) if n_total_obs > 0 else 0.0
        ),
        "decisions_seen": decision_counter,
        "stale_exposed_decisions_seen": n_stale_exposed_seen,
        "stale_exposed_decisions_scored": n_stale_exposed_scored,
        "cadence_decisions_scored": n_cadence_scored,
    }
    return summary, faith_records


def _summarise_faithfulness(records: list[dict]) -> dict:
    """Aggregate per-decision records into scalar stats worth writing to JSON.

    Filters out degenerate decisions (0 valid reservations → pi_full trivially
    1.0 on the no-op, no signal) from the DEF stats; those decisions still
    count in `n_decisions_total`.
    """
    if not records:
        return {"n_decisions_total": 0, "n_decisions_scored": 0}

    non_trivial = [r for r in records if r["valid_reservations"] > 0]
    scorable = [
        r for r in non_trivial
        if r.get("primary_metric_available", True)
        and r.get("primary_def", r.get("def")) is not None
    ]
    n_total = len(records)
    n_scored = len(scorable)

    if n_scored == 0:
        return {
            "n_decisions_total": n_total,
            "n_decisions_scored": 0,
            "note": "all decisions had 0 valid reservations (no-op-only regime)",
        }

    def_scores = np.array([
        r.get("primary_def", r["def"]) for r in scorable
    ])
    g_comps = np.array([
        r.get("primary_g_comp", r["g_comp"]) for r in scorable
    ])
    g_suffs = np.array([
        r.get("primary_g_suff", r["g_suff"]) for r in scorable
    ])
    wamsns = np.array([r["wamsn"] for r in records])  # WAMSN uses all records
    out = {
        "n_decisions_total": n_total,
        "n_decisions_scored": n_scored,
        "def_mean": float(def_scores.mean()),
        "def_std": float(def_scores.std()),
        "def_p05": float(np.percentile(def_scores, 5)),
        "def_p50": float(np.percentile(def_scores, 50)),
        "def_p95": float(np.percentile(def_scores, 95)),
        "g_comp_mean": float(g_comps.mean()),
        "g_suff_mean": float(g_suffs.mean()),
        "wamsn_mean": float(wamsns.mean()),
        "wamsn_std": float(wamsns.std()),
    }
    # Logit-margin DEF (absent in records from pre-margin checkpoints/runs).
    def_ms = np.array([
        r.get("primary_def_m", r.get("def_m")) for r in scorable
        if r.get("primary_def_m", r.get("def_m")) is not None
    ])
    if def_ms.size > 0:
        out["def_m_mean"] = float(def_ms.mean())
        out["def_m_std"] = float(def_ms.std())
        out["def_m_p50"] = float(np.percentile(def_ms, 50))
    # P2 audit aggregates (absent in pre-instrumentation records).
    excl = np.array([r["def_m_excl"] for r in non_trivial if "def_m_excl" in r])
    if excl.size > 0:
        out["def_m_excl_mean"] = float(excl.mean())

    controls = [r for r in non_trivial if "loo_oracle_def_m" in r]
    if controls:
        attention_control = np.array([
            r["attention_control_def_m"] for r in controls
        ])
        oracle = np.array([r["loo_oracle_def_m"] for r in controls])
        gap = np.array([r["loo_control_gap_m"] for r in controls])
        agreement = np.array([
            r["attention_loo_topk_agreement"] for r in controls
            if r["attention_loo_topk_agreement"] is not None
        ])
        expected_overlap = np.array([
            r["expected_random_topk_overlap"] for r in controls
            if r["expected_random_topk_overlap"] is not None
        ])
        taxi_rho = np.array([
            r["taxi_attention_loo_spearman"] for r in controls
            if r["taxi_attention_loo_spearman"] is not None
        ])
        out.update({
            "n_positive_control_decisions": len(controls),
            "attention_control_def_m_mean": float(attention_control.mean()),
            "attention_control_g_comp_m_mean": float(np.mean([
                r["attention_control_g_comp_m"] for r in controls
            ])),
            "attention_control_g_suff_m_mean": float(np.mean([
                r["attention_control_g_suff_m"] for r in controls
            ])),
            "loo_oracle_def_m_mean": float(oracle.mean()),
            "loo_oracle_g_comp_m_mean": float(np.mean([
                r["loo_oracle_g_comp_m"] for r in controls
            ])),
            "loo_oracle_g_suff_m_mean": float(np.mean([
                r["loo_oracle_g_suff_m"] for r in controls
            ])),
            "grad_x_input_def_m_mean": float(np.mean([
                r["grad_x_input_def_m"] for r in controls
            ])),
            "grad_x_input_g_comp_m_mean": float(np.mean([
                r["grad_x_input_g_comp_m"] for r in controls
            ])),
            "grad_x_input_g_suff_m_mean": float(np.mean([
                r["grad_x_input_g_suff_m"] for r in controls
            ])),
            "loo_control_gap_m_mean": float(gap.mean()),
            "attention_loo_topk_agreement_mean": (
                float(agreement.mean()) if agreement.size else None
            ),
            "expected_random_topk_overlap_mean": (
                float(expected_overlap.mean()) if expected_overlap.size else None
            ),
            "taxi_attention_loo_spearman_mean": (
                float(taxi_rho.mean()) if taxi_rho.size else None
            ),
            "n_taxi_spearman_defined": int(taxi_rho.size),
        })
        for field in ("valid_taxis", "valid_reservations", "valid_non_self_nodes"):
            values = np.array([r[field] for r in controls], dtype=np.float64)
            out[f"{field}_distribution"] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        available_k = sorted({
            k for r in controls
            for k in r.get("expected_random_topk_overlap_by_k", {})
        }, key=int)
        out["topk_overlap_by_k"] = {
            k: {
                "expected_random_overlap": float(np.mean([
                    r["expected_random_topk_overlap_by_k"][k]
                    for r in controls
                    if k in r.get("expected_random_topk_overlap_by_k", {})
                ])),
                "attention_loo_agreement": float(np.mean([
                    r["attention_loo_topk_agreement_by_k"][k]
                    for r in controls
                    if k in r.get("attention_loo_topk_agreement_by_k", {})
                ])),
            }
            for k in available_k
        }

    sensitivity_records = [
        r for r in records
        if r.get("n_stale_veh", 0) > 0
        and "aggregation_stale_attention_shift" in r
    ]
    if sensitivity_records:
        names = sensitivity_records[0]["aggregation_stale_attention_shift"]
        out["aggregation_sensitivity"] = {
            name: {
                "mean_stale_attention_shift": float(np.mean([
                    r["aggregation_stale_attention_shift"][name]
                    for r in sensitivity_records
                ])),
                "n_stale_exposed_decisions": len(sensitivity_records),
            }
            for name in names
        }

    action_row_records = [
        r for r in non_trivial if "action_row_request_def_m" in r
    ]
    if action_row_records:
        out.update({
            "n_action_row_request_decisions": len(action_row_records),
            "self_row_request_def_m_mean": float(np.mean([
                r["self_row_request_def_m"] for r in action_row_records
            ])),
            "action_row_request_def_m_mean": float(np.mean([
                r["action_row_request_def_m"] for r in action_row_records
            ])),
            "action_row_request_gap_m_mean": float(np.mean([
                r["action_row_request_gap_m"] for r in action_row_records
            ])),
        })
        out["n_excl_evaluated"] = int(excl.size)
    clamps_t = np.array([r["clamp_topk"] for r in non_trivial if "clamp_topk" in r])
    clamps_r = np.array([r["clamp_rand"] for r in non_trivial if "clamp_rand" in r])
    if clamps_t.size > 0:
        out["clamp_topk_mean"] = float(clamps_t.mean())
        out["clamp_rand_mean"] = float(clamps_r.mean())
    stale = np.array([r.get("n_stale_veh", 0) for r in records])
    if "n_stale_veh" in (records[0] if records else {}):
        stale_present = stale > 0
        out["stale_exposure_rate"] = float(stale_present.mean())
        cond = np.array([r["wamsn"] for r, sp in zip(records, stale_present) if sp])
        if cond.size > 0:
            out["wamsn_conditional_mean"] = float(cond.mean())
            stale_share = np.array([
                r["stale_attention_share"] for r, sp in zip(records, stale_present)
                if sp and "stale_attention_share" in r
            ])
            stale_shift = np.array([
                r["stale_attention_shift"] for r, sp in zip(records, stale_present)
                if sp and "stale_attention_shift" in r
            ])
            if stale_share.size:
                out["stale_attention_share_conditional_mean"] = float(
                    stale_share.mean()
                )
            stale_mass = np.array([
                r["stale_attention_mass"] for r, sp in zip(records, stale_present)
                if sp and "stale_attention_mass" in r
            ])
            if stale_mass.size:
                out["stale_attention_mass_conditional_mean"] = float(
                    stale_mass.mean()
                )
            if stale_shift.size:
                out["stale_attention_shift_conditional_mean"] = float(
                    stale_shift.mean()
                )
        out["stale_in_top3_rate"] = float(np.mean(
            [bool(r.get("stale_in_top3")) for r in records]))
    paired = [
        r for r in scorable
        if r.get("stale_exposed_precheck")
        and r.get("paired_primary_def_delta", r.get("paired_def_delta"))
        is not None
    ]
    if paired:
        out.update({
            "n_stale_exposed_paired_decisions": len(paired),
            "paired_def_delta_mean": float(np.mean([
                r.get("paired_primary_def_delta", r["paired_def_delta"])
                for r in paired
            ])),
            "paired_def_m_delta_mean": float(np.mean([
                r.get("paired_primary_def_m_delta", r["paired_def_m_delta"])
                for r in paired
            ])),
        })
        paired_excl = [r for r in paired if "paired_def_m_excl_delta" in r]
        if paired_excl:
            out.update({
                "n_stale_exposed_paired_exclusion_decisions": len(paired_excl),
                "paired_def_excl_delta_mean": float(np.mean([
                    r["paired_def_excl_delta"] for r in paired_excl
                ])),
                "paired_def_m_excl_delta_mean": float(np.mean([
                    r["paired_def_m_excl_delta"] for r in paired_excl
                ])),
            })
    # Attention drift — like WAMSN, meaningful for every decision.
    drifts = np.array([r["drift"] for r in records if "drift" in r])
    if drifts.size > 0:
        out["drift_mean"] = float(drifts.mean())
        out["drift_std"] = float(drifts.std())
        out["drift_p95"] = float(np.percentile(drifts, 95))
        out["n_drift_scored"] = int(drifts.size)
    return out


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
    parser.add_argument("--outage-duration", type=float, default=0.0,
                        help="observation freeze duration in seconds after signal loss")
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="summary JSON path; default is beside the checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output (disabled by default)")
    parser.add_argument("--faithfulness", action="store_true",
                        help="compute DEF/WAMSN per decision; ~36 extra forwards per decision")
    parser.add_argument("--faithfulness-every", type=int, default=1,
                        help="only score 1-in-N decisions when --faithfulness is set")
    parser.add_argument("--faithfulness-top-k", type=int, nargs="+", default=[1, 2, 3],
                        help="top-k values used for Comp/Suff (averaged)")
    parser.add_argument("--faithfulness-random-baselines", type=int, default=5,
                        help="random subsets sampled per k for the DEF baseline")
    parser.add_argument("--random-baseline", default="type_matched",
                        choices=["uniform", "type_matched"],
                        help="DEF control; type_matched is the reportable default")
    parser.add_argument("--exclusion-variant", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="compute chosen-action-protected construct audit")
    parser.add_argument("--demand-split", default=None,
                        choices=["train", "val", "test"],
                        help="rotate rider-demand variants from this chronological "
                             "split per episode (episode i → variant i mod N). "
                             "Use 'test' for held-out protocol evaluation.")
    parser.add_argument("--demand-variant", default=None,
                        help="single demand-variant filename to use for every episode")
    parser.add_argument("--drift", action="store_true",
                        help="also compute attention drift per sampled decision: "
                             "JS(α_clean, α_degraded) against the clean twin of "
                             "the same obs (requires --faithfulness; ~0 unless "
                             "--degradation is on)")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.drift and not args.faithfulness:
        parser.error("--drift requires --faithfulness (drift is recorded "
                     "into the per-decision faithfulness records)")

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    requested_output = args.output or args.checkpoint.with_suffix(".eval.json")
    if requested_output.exists() and not args.overwrite:
        parser.error(f"output already exists: {requested_output}; use --overwrite explicitly")

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    n_params = sum(p.numel() for p in policy.parameters())

    area = args.area or ckpt["env_config"]["area"]
    env_cfg = DispatchEnvConfig(
        area=area,
        seed=args.seed,
        degradation=DegradationConfig(
            mode=args.degradation,
            dropout_rate=args.dropout_rate,
            outage_duration_s=args.outage_duration,
        ),
        emit_clean_obs=args.drift,
    )
    env = DispatchEnv(env_cfg)

    faith_evaluator: FaithfulnessEvaluator | None = None
    if args.faithfulness and ckpt.get("policy_type", "gat") == "mlp":
        parser.error("--faithfulness requires a GAT checkpoint: the MLP "
                     "baseline (B1) has no attention channel to score")
    if args.faithfulness:
        faith_cfg = FaithfulnessConfig(
            top_k_values=tuple(args.faithfulness_top_k),
            n_random_baselines=args.faithfulness_random_baselines,
            seed=args.seed,
            random_baseline=args.random_baseline,
            exclusion_variant=args.exclusion_variant,
        )
        faith_evaluator = FaithfulnessEvaluator(policy, faith_cfg)

    print(f"checkpoint: {args.checkpoint.name}  (epoch {ckpt.get('epoch', '?')}, {n_params:,} params)")
    print(f"env:        {area}  |  degradation: {args.degradation}  |  device: {device}")
    print(f"policy:     {'stochastic' if args.stochastic else 'argmax (deterministic)'}")
    if args.faithfulness:
        print(f"faith:      k={args.faithfulness_top_k}  "
              f"random_baselines={args.faithfulness_random_baselines}  "
              f"every={args.faithfulness_every}")
    print()
    header_extra = "     DEF_mean  WAMSN_mean" if args.faithfulness else ""
    print(f"{'ep':>3}  {'pickups':>7}  {'reward':>10}  {'mean_wait_s':>11}  "
          f"{'rl_steps':>8}  {'wall_s':>7}{header_extra}")
    print("-" * (60 + len(header_extra)))

    # Demand-variant selection (E-section protocol).
    demand_files: list[str] = []
    if args.demand_split and args.demand_variant:
        parser.error("--demand-split and --demand-variant are mutually exclusive")
    if args.demand_split:
        from dispatch_marl.scenario import demand_split_files
        demand_files = demand_split_files(area, args.demand_split)
        print(f"demand:     {len(demand_files)} '{args.demand_split}' variants, "
              f"rotated per episode")
    elif args.demand_variant:
        demand_files = [args.demand_variant]

    results: list[dict] = []
    all_faith_records: list[dict] = []
    for ep in range(args.episodes):
        t0 = time.time()
        reset_options = None
        if demand_files:
            reset_options = {"taxi_route_file": demand_files[ep % len(demand_files)]}
        r, faith_records = _run_episode(
            env, policy, device, args.stochastic,
            faithfulness_evaluator=faith_evaluator,
            faithfulness_every=args.faithfulness_every,
            episode_index=ep,
            compute_drift=args.drift,
            reset_options=reset_options,
            action_seed=derive_seed(args.seed, "evaluation", ep),
        )
        if reset_options:
            r["demand_variant"] = reset_options["taxi_route_file"]
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        all_faith_records.extend(faith_records)

        extra = ""
        if args.faithfulness and faith_records:
            ep_summary = _summarise_faithfulness(faith_records)
            r["faithfulness"] = ep_summary
            if ep_summary["n_decisions_scored"] > 0:
                extra = f"     {ep_summary['def_mean']:+.3f}     {ep_summary['wamsn_mean']:.3f}"
            else:
                extra = "         n/a         n/a"

        print(f"{ep:>3}  {r['total_pickups']:>7}  {r['total_reward']:>+10.2f}  "
              f"{r['final_mean_pending_wait_s']:>11}  {r['rl_steps']:>8}  "
              f"{r['wall_s']:>7.1f}{extra}")

    env.close()

    # Aggregate policy metrics.
    completed_journeys = np.array([
        r["completed_passenger_journeys"] for r in results
    ])
    rewards = np.array([r["total_reward"] for r in results])
    waits = np.array([r["final_mean_pending_wait_s"] for r in results])
    print()
    print(f"mean over {len(results)} episodes:")
    print(
        "  completed passenger journeys: "
        f"{completed_journeys.mean():.2f} ± {completed_journeys.std():.2f}"
    )
    print(f"  reward:    {rewards.mean():+.2f} ± {rewards.std():.2f}")
    print(f"  mean_wait: {waits.mean():.1f}s")

    # Aggregate faithfulness across all episodes' records — same filtering
    # rules as the per-episode summary. Report distribution stats since DEF
    # is per-decision and heavy-tailed.
    faith_run_summary: dict = {}
    if args.faithfulness:
        faith_run_summary = _summarise_faithfulness(all_faith_records)
        if faith_run_summary.get("n_decisions_scored", 0) > 0:
            print(
                f"  DEF:       {faith_run_summary['def_mean']:+.3f} ± "
                f"{faith_run_summary['def_std']:.3f}  "
                f"(p05={faith_run_summary['def_p05']:+.3f}, "
                f"p50={faith_run_summary['def_p50']:+.3f}, "
                f"p95={faith_run_summary['def_p95']:+.3f}, "
                f"n={faith_run_summary['n_decisions_scored']})"
            )
            if "def_m_mean" in faith_run_summary:
                print(
                    f"  DEF_m:     {faith_run_summary['def_m_mean']:+.3f} ± "
                    f"{faith_run_summary['def_m_std']:.3f}  (logit margin)"
                )
            print(
                f"  WAMSN:     {faith_run_summary['wamsn_mean']:.3f} ± "
                f"{faith_run_summary['wamsn_std']:.3f}"
            )
            if "drift_mean" in faith_run_summary:
                print(
                    f"  drift:     {faith_run_summary['drift_mean']:.4f} ± "
                    f"{faith_run_summary['drift_std']:.4f}  "
                    f"(p95={faith_run_summary['drift_p95']:.4f}, "
                    f"n={faith_run_summary['n_drift_scored']})"
                )
        else:
            print(f"  faithfulness: no scorable decisions ({faith_run_summary.get('note', '')})")

    # Emit per-decision faithfulness records as JSONL alongside the summary.
    if args.faithfulness and all_faith_records:
        jsonl_path = ((args.output.with_suffix(".faithfulness.jsonl"))
                      if args.output else args.checkpoint.with_suffix(".faithfulness.jsonl"))
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for rec in all_faith_records:
                f.write(json.dumps(rec) + "\n")
        print(f"faithfulness records: {jsonl_path} ({len(all_faith_records)} decisions)")

    # Emit machine-readable summary for downstream comparison.
    summary_path = args.output or args.checkpoint.with_suffix(".eval.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "schema_version": 2,
        "protocol_version": "2.0",
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "area": area,
        "degradation": args.degradation,
        "outage_duration_s": args.outage_duration,
        "stochastic": args.stochastic,
        "seed": args.seed,
        "episodes": len(results),
        "demand_split": args.demand_split,
        "demand_files": demand_files,
        "per_episode": results,
        "mean_completed_passenger_journeys": float(completed_journeys.mean()),
        "std_completed_passenger_journeys": float(completed_journeys.std()),
        # Protocol-v1 compatibility aliases.
        "mean_pickups": float(completed_journeys.mean()),
        "std_pickups": float(completed_journeys.std()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "provenance": runtime_provenance(PROJECT_ROOT),
    }
    if args.faithfulness:
        summary["faithfulness_summary"] = faith_run_summary
        summary["faithfulness_config"] = {
            "top_k_values": list(args.faithfulness_top_k),
            "n_random_baselines": args.faithfulness_random_baselines,
            "faithfulness_every": args.faithfulness_every,
            "random_baseline": args.random_baseline,
            "exclusion_variant": args.exclusion_variant,
        }
    atomic_write_json(summary_path, summary)
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
