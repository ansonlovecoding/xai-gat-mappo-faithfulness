"""Faithfulness across the capability spectrum (P3) + H5 across seeds (P4).

For each checkpoint: N clean episodes on held-out test demand, sampled
decisions scored with BOTH random-baseline schemes (uniform = standard
protocol, type_matched = artifact control), plus policy entropy and
pickups. Establishes whether the "attention is uninformative" finding is
an artifact of any particular capability level, and whether the H5
verdict holds across training seeds. Optionally repeats under a
degradation condition (--degradation --outage).

Usage:
  python scripts/capability_spectrum.py <ckpt1> <ckpt2> ... [--episodes 2]
  python scripts/capability_spectrum.py <ckpts...> --degradation tunnel_triggered --outage 60
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

from eval_policy import _choose_device, _load_policy  # noqa: E402

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
)
from dispatch_marl.models import obs_dict_to_tensors  # noqa: E402
from dispatch_marl.scenario import demand_split_files  # noqa: E402


def evaluate_ckpt(ckpt_path: Path, device: str, episodes: int, seeds: list[int],
                  every: int, degradation: str, outage: float) -> dict:
    policy, ckpt = _load_policy(ckpt_path, device)
    if ckpt.get("policy_type", "gat") == "mlp":
        return {"checkpoint": str(ckpt_path), "note": "mlp — no attention channel"}
    area = ckpt["env_config"]["area"]
    aoi_unaware = bool(ckpt["env_config"].get("aoi_unaware", False))
    demand = demand_split_files(area, "test")

    ev_u = FaithfulnessEvaluator(policy, FaithfulnessConfig(seed=0))
    ev_t = FaithfulnessEvaluator(policy, FaithfulnessConfig(
        seed=0, random_baseline="type_matched"))

    vals_u, vals_t, entropies, pickups = [], [], [], []
    for seed in seeds:
        env = DispatchEnv(DispatchEnvConfig(
            area=area, seed=seed, aoi_unaware=aoi_unaware,
            degradation=DegradationConfig(mode=degradation,
                                          outage_duration_s=outage),
        ))
        for ep in range(episodes):
            obs, _ = env.reset(options={"taxi_route_file": demand[ep % len(demand)]})
            counter, ep_pickups = 0, 0
            while not env.done:
                if obs:
                    batched, agents = obs_dict_to_tensors(obs, device=device)
                    with torch.no_grad():
                        logits = policy.forward(batched)["logits"]
                    dist = torch.distributions.Categorical(logits=logits)
                    acts = dist.sample()
                    entropies.append(float(dist.entropy().mean()))
                    for i in range(len(agents)):
                        if counter % every == 0:
                            single = {k: v[i:i + 1] for k, v in batched.items()}
                            if int(single["reservations_mask"].sum()) > 0:
                                a = int(acts[i])
                                vals_u.append(ev_u.evaluate_decision(
                                    single, action=a).def_margin)
                                vals_t.append(ev_t.evaluate_decision(
                                    single, action=a).def_margin)
                        counter += 1
                    actions = {ag: int(acts[j]) for j, ag in enumerate(agents)}
                else:
                    actions = {}
                obs, _, _, _, infos = env.step(actions)
                if infos:
                    ep_pickups += int(next(iter(infos.values())).get("pickups_delta", 0))
            pickups.append(ep_pickups)
        env.close()

    u, t = np.array(vals_u), np.array(vals_t)
    rng = np.random.default_rng(0)

    def ci(x):
        if len(x) == 0:
            return [float("nan")] * 2
        means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    return {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "n_decisions": int(len(u)),
        "mean_pickups": float(np.mean(pickups)) if pickups else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "def_m_uniform": {"mean": float(u.mean()) if len(u) else float("nan"),
                          "ci95": ci(u)},
        "def_m_type_matched": {"mean": float(t.mean()) if len(t) else float("nan"),
                               "ci95": ci(t)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=2, help="per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered"])
    parser.add_argument("--outage", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "runs" / "capability_spectrum.json")
    args = parser.parse_args()

    device = args.device or _choose_device()
    rows = []
    for ckpt in args.checkpoints:
        t0 = time.time()
        r = evaluate_ckpt(ckpt, device, args.episodes, args.seeds,
                          args.every, args.degradation, args.outage)
        rows.append(r)
        if "note" in r:
            print(f"{ckpt.name}: {r['note']}")
            continue
        print(f"{ckpt.parent.parent.name}/{ckpt.name} (ep {r['epoch']}): "
              f"pickups={r['mean_pickups']:.1f}  H={r['mean_entropy']:.3f}  "
              f"def_m uniform={r['def_m_uniform']['mean']:+.3f}  "
              f"type-matched={r['def_m_type_matched']['mean']:+.3f} "
              f"{r['def_m_type_matched']['ci95']}  "
              f"(n={r['n_decisions']}, {time.time()-t0:.0f}s)")

    payload = {"degradation": args.degradation, "outage_s": args.outage,
               "episodes_per_seed": args.episodes, "seeds": args.seeds,
               "rows": rows}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
