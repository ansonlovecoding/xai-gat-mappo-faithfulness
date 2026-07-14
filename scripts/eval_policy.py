"""Evaluate a trained GAT-MAPPO checkpoint against the baseline table.

Runs one or more evaluation episodes with the checkpointed policy and
reports pickups + reward + wait-time — the same metrics
`scripts/run_baselines.py` produces — so the numbers slot straight into a
comparison table.

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
import platform
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
)
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    DispatchMLPPolicy,
    MLPPolicyConfig,
    PolicyConfig,
    obs_dict_to_tensors,
)


def _choose_device() -> str:
    # MPS on Intel macs (AMD GPUs) is broken in torch 2.2 (multi-dim
    # reduction kernels assert) and removed in 2.3+ — Apple Silicon only.
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_policy(ckpt_path: Path, device: str):
    """Load a checkpoint, instantiating the class recorded in policy_type.

    Checkpoints saved before the B1 baseline landed have no "policy_type"
    key — they are all GAT.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_type = ckpt.get("policy_type", "gat")
    if policy_type == "mlp":
        pol_cfg = MLPPolicyConfig(**{**ckpt["policy_config"], "device": device})
        policy = DispatchMLPPolicy(pol_cfg)
    else:
        pol_cfg = PolicyConfig(**{**ckpt["policy_config"], "device": device})
        policy = DispatchGATPolicy(pol_cfg)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, ckpt


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
    obs_dict, _ = env.reset(options=reset_options)
    total_reward = 0.0
    total_pickups = 0
    step = 0
    last_wait = 0.0
    faith_records: list[dict] = []
    # A global counter over decisions is what we sub-sample against, so the
    # cadence is consistent regardless of how many agents act on a given step.
    decision_counter = 0
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
                    if decision_counter % faithfulness_every == 0:
                        single = {k: v[i : i + 1] for k, v in batched.items()}
                        n_valid_res = int(single["reservations_mask"].sum().item())
                        result = faithfulness_evaluator.evaluate_decision(
                            single, action=int(actions_np[i])
                        )
                        drift = None
                        if compute_drift and a in clean_obs_by_agent:
                            clean_single = {
                                k: torch.as_tensor(
                                    v, dtype=torch.float32, device=device
                                ).unsqueeze(0)
                                for k, v in clean_obs_by_agent[a].items()
                            }
                            clean_row = faithfulness_evaluator.attention_row(clean_single)
                            drift = compute_attention_drift(
                                clean_row, result.attention_row
                            )
                        faith_records.append({
                            **({"drift": round(drift, 4)} if drift is not None else {}),
                            "episode": episode_index,
                            "rl_step": step,
                            "agent": a,
                            "action": result.action,
                            "pi_full": round(result.pi_full, 4),
                            "def": round(result.def_score, 4),
                            "def_m": round(result.def_margin, 4),
                            "m_full": round(result.m_full, 4),
                            "g_comp": round(result.g_comp, 4),
                            "g_suff": round(result.g_suff, 4),
                            "comp": round(result.comp, 4),
                            "suff": round(result.suff, 4),
                            "wamsn": round(result.wamsn, 4),
                            "valid_reservations": n_valid_res,
                            "n_k_evaluated": len(result.per_k),
                        })
                    decision_counter += 1
        else:
            actions = {}

        obs_dict, rewards, _, _, infos = env.step(actions)
        if rewards:
            total_reward += next(iter(rewards.values()))
        if infos:
            info = next(iter(infos.values()))
            total_pickups += int(info.get("pickups_delta", 0))
            last_wait = float(info.get("mean_wait_time", last_wait))
        step += 1

    summary = {
        "total_pickups": total_pickups,
        "total_reward": round(total_reward, 3),
        "final_mean_pending_wait_s": round(last_wait, 1),
        "rl_steps": step,
        "empirical_degradation_rate": (
            round(n_degraded_obs / n_total_obs, 4) if n_total_obs > 0 else 0.0
        ),
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
    n_total = len(records)
    n_scored = len(non_trivial)

    if n_scored == 0:
        return {
            "n_decisions_total": n_total,
            "n_decisions_scored": 0,
            "note": "all decisions had 0 valid reservations (no-op-only regime)",
        }

    def_scores = np.array([r["def"] for r in non_trivial])
    g_comps = np.array([r["g_comp"] for r in non_trivial])
    g_suffs = np.array([r["g_suff"] for r in non_trivial])
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
    def_ms = np.array([r["def_m"] for r in non_trivial if "def_m" in r])
    if def_ms.size > 0:
        out["def_m_mean"] = float(def_ms.mean())
        out["def_m_std"] = float(def_ms.std())
        out["def_m_p50"] = float(np.percentile(def_ms, 50))
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
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--faithfulness", action="store_true",
                        help="compute DEF/WAMSN per decision; ~36 extra forwards per decision")
    parser.add_argument("--faithfulness-every", type=int, default=1,
                        help="only score 1-in-N decisions when --faithfulness is set")
    parser.add_argument("--faithfulness-top-k", type=int, nargs="+", default=[1, 2, 3],
                        help="top-k values used for Comp/Suff (averaged)")
    parser.add_argument("--faithfulness-random-baselines", type=int, default=5,
                        help="random subsets sampled per k for the DEF baseline")
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

    if args.drift and not args.faithfulness:
        parser.error("--drift requires --faithfulness (drift is recorded "
                     "into the per-decision faithfulness records)")

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    n_params = sum(p.numel() for p in policy.parameters())

    area = args.area or ckpt["env_config"]["area"]
    # B3 checkpoints must be evaluated with the same AoI-unaware obs they
    # were trained on; older checkpoints predate the flag, hence .get().
    aoi_unaware = bool(ckpt["env_config"].get("aoi_unaware", False))
    env_cfg = DispatchEnvConfig(
        area=area,
        seed=args.seed,
        degradation=DegradationConfig(mode=args.degradation, dropout_rate=args.dropout_rate),
        aoi_unaware=aoi_unaware,
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
        )
        faith_evaluator = FaithfulnessEvaluator(policy, faith_cfg)

    print(f"checkpoint: {args.checkpoint.name}  (epoch {ckpt.get('epoch', '?')}, {n_params:,} params)")
    print(f"env:        {area}  |  degradation: {args.degradation}  |  device: {device}"
          + ("  |  AoI-unaware (B3)" if aoi_unaware else ""))
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
    pickups = np.array([r["total_pickups"] for r in results])
    rewards = np.array([r["total_reward"] for r in results])
    waits = np.array([r["final_mean_pending_wait_s"] for r in results])
    print()
    print(f"mean over {len(results)} episodes:")
    print(f"  pickups:   {pickups.mean():.2f} ± {pickups.std():.2f}")
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
        jsonl_path = args.checkpoint.with_suffix(".faithfulness.jsonl")
        with jsonl_path.open("w") as f:
            for rec in all_faith_records:
                f.write(json.dumps(rec) + "\n")
        print(f"faithfulness records: {jsonl_path} ({len(all_faith_records)} decisions)")

    # Emit machine-readable summary for downstream comparison.
    summary_path = args.checkpoint.with_suffix(".eval.json")
    summary: dict = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "area": area,
        "degradation": args.degradation,
        "stochastic": args.stochastic,
        "seed": args.seed,
        "episodes": len(results),
        "per_episode": results,
        "mean_pickups": float(pickups.mean()),
        "std_pickups": float(pickups.std()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
    }
    if args.faithfulness:
        summary["faithfulness_summary"] = faith_run_summary
        summary["faithfulness_config"] = {
            "top_k_values": list(args.faithfulness_top_k),
            "n_random_baselines": args.faithfulness_random_baselines,
            "faithfulness_every": args.faithfulness_every,
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
