"""Coupled vs decoupled explanation: the dissertation's headline comparison.

For each sampled decision in fresh evaluation episodes, computes the full
faithfulness bundle TWICE over identical counterfactuals:

  * **coupled**   — top-k nodes chosen by the GAT attention row
  * **decoupled** — top-k nodes chosen by the distilled explainer head

and reports paired per-decision differences (margin-DEF is the primary
metric; probability-DEF, and WAMSN under degradation, come along for
free). Significance via a paired sign-flip permutation test.

Run after `scripts/distill_explainer.py`. Evaluation seeds default to
42/43 — disjoint from the distillation rollout seed (7), so the head is
scored on decisions it never trained on.

Usage:
  python scripts/eval_explainer.py <policy_ckpt.pt> [<explainer_head.pt>]
  python scripts/eval_explainer.py <ckpt> --degradation tunnel_triggered
  python scripts/eval_explainer.py <ckpt> --episodes 3 --every 5
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_policy import _choose_device, _load_policy  # noqa: E402

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
)
from dispatch_marl.models import (  # noqa: E402
    DecoupledExplainerHead,
    explainer_importance_row,
    obs_dict_to_tensors,
)


def _signflip_p(deltas: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """One-sided: H0 mean(delta) <= 0."""
    obs = deltas.mean()
    count = sum(
        1 for _ in range(n_perm)
        if (deltas * rng.choice([-1.0, 1.0], size=len(deltas))).mean() >= obs
    )
    return float((count + 1) / (n_perm + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="frozen GAT policy .pt")
    parser.add_argument("explainer", type=Path, nargs="?", default=None,
                        help="explainer_head.pt (default: next to the ckpt)")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered", "random_dropout"])
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--every", type=int, default=5,
                        help="score 1-in-N decisions")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--random-baselines", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--random-baseline", default="uniform",
                        choices=["uniform", "type_matched"],
                        help="type_matched = the P2 artifact control: random "
                             "subsets share each channel's top-k node-type "
                             "composition, so occlusion=action-deletion hits "
                             "both sides equally")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="default <ckpt>.explainer_compare.json")
    args = parser.parse_args()

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    if ckpt.get("policy_type", "gat") == "mlp":
        parser.error("needs a GAT checkpoint")
    explainer_path = args.explainer or (args.checkpoint.parent / "explainer_head.pt")
    if not explainer_path.exists():
        parser.error(f"explainer head not found: {explainer_path} "
                     "(run scripts/distill_explainer.py first)")
    head, head_ckpt = DecoupledExplainerHead.load(explainer_path, device)
    area = ckpt["env_config"]["area"]
    aoi_unaware = bool(ckpt["env_config"].get("aoi_unaware", False))

    print(f"policy:    {args.checkpoint.name} (epoch {ckpt.get('epoch')})")
    print(f"explainer: {explainer_path.name} "
          f"(val Spearman {head_ckpt.get('report', {}).get('val_spearman_mean', float('nan')):+.3f})")
    print(f"env:       {area}  degradation={args.degradation}  "
          f"seeds={args.seeds}  episodes={args.episodes}/seed")
    print()

    records: list[dict] = []
    for seed in args.seeds:
        env = DispatchEnv(DispatchEnvConfig(
            area=area, seed=seed, aoi_unaware=aoi_unaware,
            degradation=DegradationConfig(mode=args.degradation,
                                          dropout_rate=args.dropout_rate),
        ))
        # Two evaluators with the SAME seed → identical random-baseline
        # subsets per decision, so the coupled/decoupled comparison is
        # exactly paired (only the top-k selection differs).
        faith_cfg = FaithfulnessConfig(
            top_k_values=tuple(args.top_k),
            n_random_baselines=args.random_baselines, seed=seed,
            random_baseline=args.random_baseline,
        )
        ev_coupled = FaithfulnessEvaluator(policy, faith_cfg)
        ev_decoupled = FaithfulnessEvaluator(policy, faith_cfg)

        for ep in range(args.episodes):
            obs_dict, _ = env.reset()
            counter = 0
            while not env.done:
                if obs_dict:
                    batched, agents = obs_dict_to_tensors(obs_dict, device=device)
                    with torch.no_grad():
                        logits = policy.forward(batched)["logits"]
                        acts = torch.distributions.Categorical(logits=logits).sample()
                    for i in range(len(agents)):
                        if counter % args.every == 0:
                            single = {k: v[i : i + 1] for k, v in batched.items()}
                            if int(single["reservations_mask"].sum()) > 0:
                                action = int(acts[i])
                                r_c = ev_coupled.evaluate_decision(single, action=action)
                                head_row = explainer_importance_row(head, policy, single)
                                r_d = ev_decoupled.evaluate_decision(
                                    single, action=action, importance_row=head_row)
                                records.append({
                                    "seed": seed, "episode": ep,
                                    "def_m_coupled": r_c.def_margin,
                                    "def_m_decoupled": r_d.def_margin,
                                    "def_coupled": r_c.def_score,
                                    "def_decoupled": r_d.def_score,
                                    "wamsn_coupled": r_c.wamsn,
                                    "wamsn_decoupled": r_d.wamsn,
                                })
                        counter += 1
                    actions = {a: int(acts[i]) for i, a in enumerate(agents)}
                else:
                    actions = {}
                obs_dict, *_ = env.step(actions)
        env.close()
        print(f"seed {seed}: cumulative paired decisions = {len(records)}")

    if len(records) < 10:
        print("not enough paired decisions — increase --episodes or lower --every")
        return 1

    rng = np.random.default_rng(0)
    out: dict = {"n_decisions": len(records),
                 "explainer": str(explainer_path),
                 "checkpoint": str(args.checkpoint),
                 "degradation": args.degradation}
    print()
    for metric, higher_better in (("def_m", True), ("def", True)):
        c = np.array([r[f"{metric}_coupled"] for r in records])
        d = np.array([r[f"{metric}_decoupled"] for r in records])
        delta = d - c
        p = _signflip_p(delta if higher_better else -delta,
                        args.n_permutations, rng)
        out[metric] = {
            "coupled_mean": float(c.mean()), "coupled_std": float(c.std()),
            "decoupled_mean": float(d.mean()), "decoupled_std": float(d.std()),
            "delta_mean": float(delta.mean()),
            "p_decoupled_better": p,
        }
        print(f"{metric:>6}: coupled {c.mean():+.4f}±{c.std():.4f}   "
              f"decoupled {d.mean():+.4f}±{d.std():.4f}   "
              f"Δ={delta.mean():+.4f}  p(decoupled>coupled)={p:.4f}")
    wc = np.array([r["wamsn_coupled"] for r in records])
    wd = np.array([r["wamsn_decoupled"] for r in records])
    out["wamsn"] = {"coupled_mean": float(wc.mean()),
                    "decoupled_mean": float(wd.mean())}
    print(f" wamsn: coupled {wc.mean():.4f}   decoupled {wd.mean():.4f}")

    out_path = args.out or args.checkpoint.with_suffix(".explainer_compare.json")
    out_path.write_text(json.dumps({**out, "records": records}, indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
