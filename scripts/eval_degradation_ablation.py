"""Structured vs random telemetry degradation ablation.

Runs the same checkpoint under three conditions on identical seeds and prints
one comparison table. This is the dissertation's flagship ablation (README
§5, project_dissertation memory): does a policy behave (and does its
explanation stay faithful) *differently* under physically-grounded
degradation than under matched-rate random noise?

Conditions:

  1. `off`               — clean baseline. Establishes upper bound.
  2. `tunnel_triggered`  — structured: degradation fires when a taxi's edge
                           is in `tunnels.json`. Measures empirical rate.
  3. `random_dropout`    — matched-rate Bernoulli. `dropout_rate` is set to
                           the empirical rate observed in condition 2, so
                           the *average* degradation is identical — only the
                           *structure* differs.

Reported per condition: pickups, reward, mean_wait, empirical degradation
rate, DEF (mean ± std, plus p05/p50/p95), WAMSN (mean ± std). All from the
same `--faithfulness`-enabled machinery in `eval_policy.py` — this script is
just an orchestrator that runs it three times.

Usage:
  python scripts/eval_degradation_ablation.py <ckpt>
  python scripts/eval_degradation_ablation.py <ckpt> --episodes 3
  python scripts/eval_degradation_ablation.py <ckpt> --faithfulness-every 10
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the eval_policy machinery so this script stays in sync with any
# future changes to the eval loop. eval_policy is script-shaped but this
# import is exactly how a research codebase should compose CLI tools.
from eval_policy import (  # noqa: E402
    _choose_device,
    _load_policy,
    _run_episode,
    _summarise_faithfulness,
)

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
)


CONDITION_ORDER = ["off", "tunnel_triggered", "random_dropout"]


def _run_condition(
    policy,
    device: str,
    area: str,
    seed: int,
    episodes: int,
    degradation_mode: str,
    dropout_rate: float,
    faith_evaluator: FaithfulnessEvaluator,
    faith_every: int,
    aoi_unaware: bool = False,
    position_noise_m: float = 20.0,
    keep_records: bool = False,
    stochastic: bool = False,
    outage_duration_s: float = 0.0,
    corruption: str = "freeze",
    demand_files: list[str] | None = None,
) -> dict:
    """Run `episodes` eval episodes under one degradation condition.

    Returns a dict with policy metrics, empirical degradation rate, and
    aggregate faithfulness stats over all decisions across all episodes.
    With `keep_records`, the raw per-decision faithfulness dicts are
    included under "faith_records" (used by the severity sweep, whose
    analysis bootstraps over decisions rather than episodes).
    """
    env_cfg = DispatchEnvConfig(
        area=area,
        seed=seed,
        degradation=DegradationConfig(
            mode=degradation_mode,
            dropout_rate=dropout_rate,
            position_noise_m=position_noise_m,
            outage_duration_s=outage_duration_s,
            corruption=corruption,
        ),
        aoi_unaware=aoi_unaware,
        emit_clean_obs=True,  # attention drift vs the clean twin, per decision
    )
    env = DispatchEnv(env_cfg)

    per_episode: list[dict] = []
    all_faith: list[dict] = []
    t0 = time.time()
    for ep in range(episodes):
        reset_options = None
        if demand_files:
            reset_options = {"taxi_route_file": demand_files[ep % len(demand_files)]}
        summary, faith_records = _run_episode(
            env, policy, device,
            stochastic=stochastic,
            faithfulness_evaluator=faith_evaluator,
            faithfulness_every=faith_every,
            episode_index=ep,
            compute_drift=True,
            reset_options=reset_options,
        )
        per_episode.append(summary)
        all_faith.extend(faith_records)
    env.close()

    pickups = np.array([e["total_pickups"] for e in per_episode])
    rewards = np.array([e["total_reward"] for e in per_episode])
    waits = np.array([e["final_mean_pending_wait_s"] for e in per_episode])
    deg_rates = np.array([e["empirical_degradation_rate"] for e in per_episode])
    faith_summary = _summarise_faithfulness(all_faith)

    out = {
        "condition": degradation_mode,
        "configured_dropout_rate": dropout_rate if degradation_mode == "random_dropout" else None,
        "corruption": corruption,
        "outage_duration_s": outage_duration_s,
        "position_noise_m": position_noise_m,
        "demand_files": demand_files,
        "empirical_degradation_rate": float(deg_rates.mean()),
        "mean_pickups": float(pickups.mean()),
        "std_pickups": float(pickups.std()),
        "mean_reward": float(rewards.mean()),
        "mean_wait_s": float(waits.mean()),
        "faithfulness": faith_summary,
        "per_episode": per_episode,
        "n_faith_records": len(all_faith),
        "wall_s": round(time.time() - t0, 1),
    }
    if keep_records:
        out["faith_records"] = all_faith
    return out


def _fmt_faith(f: dict, key: str, fmt: str = "+.3f", nan: str = "n/a") -> str:
    v = f.get(key)
    if v is None:
        return nan
    return format(v, fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help=".pt file saved by train.py")
    parser.add_argument("--area", default=None,
                        help="override area; defaults to whatever the checkpoint was trained on")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--faithfulness-every", type=int, default=5,
                        help="1-in-N sub-sample rate for DEF/WAMSN "
                             "(default 5 — full-cadence would be very slow)")
    parser.add_argument("--faithfulness-top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--faithfulness-random-baselines", type=int, default=5)
    parser.add_argument("--matched-rate", type=float, default=None,
                        help="override the matched dropout rate; default is the empirical "
                             "rate measured in the tunnel_triggered condition")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the comparison JSON "
                             "(default <ckpt>.ablation.json)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    if ckpt.get("policy_type", "gat") == "mlp":
        parser.error("the degradation ablation scores DEF/WAMSN, which needs "
                     "a GAT checkpoint — the MLP baseline (B1) has no "
                     "attention channel")
    area = args.area or ckpt["env_config"]["area"]
    aoi_unaware = bool(ckpt["env_config"].get("aoi_unaware", False))

    # One evaluator shared across conditions — its RNG is reseeded per
    # __init__, so we're consistent. Sub-samples the same set of decisions
    # per RL step regardless of condition.
    faith_cfg = FaithfulnessConfig(
        top_k_values=tuple(args.faithfulness_top_k),
        n_random_baselines=args.faithfulness_random_baselines,
        seed=args.seed,
    )
    faith_evaluator = FaithfulnessEvaluator(policy, faith_cfg)

    print(f"checkpoint: {args.checkpoint.name}  (epoch {ckpt.get('epoch', '?')})")
    print(f"env:        {area}  |  device: {device}  |  seed: {args.seed}")
    print(f"episodes:   {args.episodes} per condition")
    print(f"faith:      k={args.faithfulness_top_k}  "
          f"random_baselines={args.faithfulness_random_baselines}  "
          f"every={args.faithfulness_every}")
    print()

    # --- Run tunnel_triggered first, to measure the empirical rate ---
    # (Ordering off → tunnel → random would be more natural to read, but
    # random_dropout needs the tunnel rate as input. Off is trivially fast
    # so we run it last for the print-order.)
    print("[1/3] tunnel_triggered → measuring empirical rate…")
    tunnel = _run_condition(
        policy, device, area, args.seed, args.episodes,
        degradation_mode="tunnel_triggered", dropout_rate=0.0,
        faith_evaluator=faith_evaluator, faith_every=args.faithfulness_every,
        aoi_unaware=aoi_unaware,
    )
    tunnel_rate = tunnel["empirical_degradation_rate"]
    matched_rate = args.matched_rate if args.matched_rate is not None else tunnel_rate
    print(f"        empirical tunnel rate = {tunnel_rate:.4f}  "
          f"→ using {matched_rate:.4f} for random_dropout")

    print("[2/3] random_dropout at matched rate…")
    random_dropout = _run_condition(
        policy, device, area, args.seed, args.episodes,
        degradation_mode="random_dropout", dropout_rate=matched_rate,
        faith_evaluator=faith_evaluator, faith_every=args.faithfulness_every,
        aoi_unaware=aoi_unaware,
    )

    print("[3/3] off (clean baseline)…")
    off = _run_condition(
        policy, device, area, args.seed, args.episodes,
        degradation_mode="off", dropout_rate=0.0,
        faith_evaluator=faith_evaluator, faith_every=args.faithfulness_every,
        aoi_unaware=aoi_unaware,
    )

    # --- Comparison table ---
    conditions = {"off": off, "tunnel_triggered": tunnel, "random_dropout": random_dropout}
    print()
    print(f"{'condition':<20} {'deg_rate':>9} {'pickups':>8} {'reward':>9} "
          f"{'wait_s':>7} {'DEF_mean':>9} {'DEF_std':>8} {'WAMSN':>8} {'drift':>7}")
    print("-" * 93)
    for cond_name in CONDITION_ORDER:
        r = conditions[cond_name]
        f = r["faithfulness"]
        print(
            f"{cond_name:<20} "
            f"{r['empirical_degradation_rate']:>9.4f} "
            f"{r['mean_pickups']:>8.2f} "
            f"{r['mean_reward']:>+9.2f} "
            f"{r['mean_wait_s']:>7.1f} "
            f"{_fmt_faith(f, 'def_mean'):>9} "
            f"{_fmt_faith(f, 'def_std', '.3f'):>8} "
            f"{_fmt_faith(f, 'wamsn_mean', '.3f'):>8} "
            f"{_fmt_faith(f, 'drift_mean', '.4f'):>7}"
        )

    # --- Ablation-specific commentary ---
    # The dissertation's claim rests on the *difference* between tunnel and
    # random_dropout at matched rate. Surface those deltas immediately so
    # the reader/user doesn't need to eyeball columns.
    print()
    print("structured vs random (tunnel_triggered − random_dropout, at matched rate):")
    if tunnel["faithfulness"].get("def_mean") is not None and \
       random_dropout["faithfulness"].get("def_mean") is not None:
        d_def = tunnel["faithfulness"]["def_mean"] - random_dropout["faithfulness"]["def_mean"]
        d_wamsn = tunnel["faithfulness"]["wamsn_mean"] - random_dropout["faithfulness"]["wamsn_mean"]
        d_pickups = tunnel["mean_pickups"] - random_dropout["mean_pickups"]
        print(f"  ΔDEF     = {d_def:+.3f}   (tunnel more faithful if positive)")
        print(f"  ΔWAMSN   = {d_wamsn:+.3f}   (tunnel puts more attention on stale nodes if positive)")
        print(f"  Δpickups = {d_pickups:+.2f}")
        if tunnel["faithfulness"].get("drift_mean") is not None and \
           random_dropout["faithfulness"].get("drift_mean") is not None:
            d_drift = tunnel["faithfulness"]["drift_mean"] - random_dropout["faithfulness"]["drift_mean"]
            print(f"  Δdrift   = {d_drift:+.4f}   (tunnel shifts attention more if positive)")

    # --- Save comparison JSON ---
    out_path = args.output or args.checkpoint.with_suffix(".ablation.json")
    payload = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "area": area,
        "seed": args.seed,
        "episodes_per_condition": args.episodes,
        "faithfulness_config": {
            "top_k_values": list(args.faithfulness_top_k),
            "n_random_baselines": args.faithfulness_random_baselines,
            "faithfulness_every": args.faithfulness_every,
        },
        "matched_dropout_rate": matched_rate,
        "conditions": conditions,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"comparison: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
