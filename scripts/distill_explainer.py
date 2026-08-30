"""Train the decoupled explanation head by occlusion distillation.

Pipeline:

  1. Roll out N episodes with the (frozen) policy, collecting the exact
     per-decision observations.
  2. For each sampled decision, occlude each valid non-self node in turn
     and record the drop in the decision's logit margin — that vector is
     the decision's "ground-truth" node-importance signature (the same
     counterfactual machinery DEF is built on).
  3. Train `DecoupledExplainerHead` on (detached) policy node embeddings
     to reproduce the softmax-normalised occlusion targets (KL loss).
  4. Report a held-out validation KL and per-decision Spearman rank
     correlation between predicted and true importance — the number that
     says whether the head actually learned the ranking DEF cares about.

The policy is never updated: embeddings are detached and only the head's
parameters are in the optimiser. Saves `explainer_head.pt` next to the
policy checkpoint (plus a .json training report).

Usage:
  python scripts/distill_explainer.py <policy_ckpt.pt>
  python scripts/distill_explainer.py <ckpt> --episodes 3 --max-decisions 2000
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
import torch.nn.functional as F
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_policy import _choose_device, _load_policy  # noqa: E402

from dispatch_marl import DispatchEnv, DispatchEnvConfig  # noqa: E402
from dispatch_marl.faithfulness import decision_margin  # noqa: E402
from dispatch_marl.models import (  # noqa: E402
    DecoupledExplainerHead,
    ExplainerConfig,
    obs_dict_to_tensors,
)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman over one decision's nodes (tiny n, ties broken by order)."""
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(-x)).astype(np.float64)
    ry = np.argsort(np.argsort(-y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def collect_dataset(
    policy, area: str, seed: int, episodes: int, max_decisions: int,
    margin_cap: float, device: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Gather (obs, occlusion-target) pairs from fresh rollouts.

    Only decisions with ≥2 valid non-self nodes are kept — a single-node
    "distribution" carries no ranking signal.
    """
    from dispatch_marl.faithfulness import FaithfulnessEvaluator, FaithfulnessConfig

    env = DispatchEnv(DispatchEnvConfig(area=area, seed=seed))
    evaluator = FaithfulnessEvaluator(policy, FaithfulnessConfig(seed=seed))
    K_n = policy.config.k_neighbors
    K_r = policy.config.k_reservations
    N = 1 + K_n + K_r

    samples: list[dict] = []
    for ep in range(episodes):
        obs_dict, _ = env.reset()
        while not env.done and len(samples) < max_decisions * 2:
            if obs_dict:
                batched, agents = obs_dict_to_tensors(obs_dict, device=device)
                with torch.no_grad():
                    out = policy.forward(batched)
                    dist = torch.distributions.Categorical(logits=out["logits"])
                    acts = dist.sample()
                for i in range(len(agents)):
                    single = {k: v[i : i + 1] for k, v in batched.items()}
                    action = int(acts[i])
                    with torch.no_grad():
                        s_out = policy.forward(single)
                    node_mask = s_out["node_mask"][0].cpu().numpy().astype(bool)
                    valid = np.array([j for j in range(1, N) if node_mask[j]])
                    if len(valid) < 2:
                        continue
                    m_full = decision_margin(s_out["logits"][0], action, margin_cap)
                    # Per-node occlusion: importance_j = m_full − m(occlude j)
                    deltas = np.zeros(len(valid))
                    for jj, j in enumerate(valid):
                        occluded = evaluator._mask_out(single, np.array([j]))
                        with torch.no_grad():
                            o_out = policy.forward(occluded)
                        m_j = decision_margin(o_out["logits"][0], action, margin_cap)
                        deltas[jj] = m_full - m_j
                    samples.append({
                        "node_emb": s_out["node_emb"][0].detach().cpu(),
                        "node_mask": s_out["node_mask"][0].detach().cpu(),
                        "valid_idx": valid,
                        "deltas": deltas,
                    })
                actions = {a: int(acts[i]) for i, a in enumerate(agents)}
            else:
                actions = {}
            obs_dict, *_ = env.step(actions)
        print(f"  episode {ep}: {len(samples)} decisions so far")
    env.close()

    if len(samples) > max_decisions:
        keep = rng.choice(len(samples), size=max_decisions, replace=False)
        samples = [samples[int(i)] for i in keep]
    return samples


def train_head(
    samples: list[dict], head: DecoupledExplainerHead,
    epochs: int, lr: float, batch_size: int, val_frac: float,
    device: str, rng: np.random.Generator,
) -> dict:
    idx = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * val_frac))
    val_set = [samples[int(i)] for i in idx[:n_val]]
    train_set = [samples[int(i)] for i in idx[n_val:]]
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    T = head.config.target_temperature

    def _target_dist(s) -> torch.Tensor:
        """Softmax over valid non-self nodes of (Δmargin / T), embedded (N,)."""
        t = torch.full((s["node_mask"].shape[0],), float("-inf"))
        t[s["valid_idx"]] = torch.as_tensor(s["deltas"] / T, dtype=torch.float32)
        return torch.softmax(t, dim=-1)

    def _batch_loss(batch, train: bool) -> float:
        emb = torch.stack([s["node_emb"] for s in batch]).to(device)
        mask = torch.stack([s["node_mask"] for s in batch]).to(device)
        target = torch.stack([_target_dist(s) for s in batch]).to(device)
        pred = head(emb, mask)  # (B, N), softmax over valid non-self
        eps = 1e-12
        kl = (target * ((target + eps).log() - (pred + eps).log())).sum(dim=-1).mean()
        if train:
            optimizer.zero_grad()
            kl.backward()
            optimizer.step()
        return float(kl)

    history = []
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(train_set))
        tr_losses = []
        for start in range(0, len(train_set), batch_size):
            batch = [train_set[int(i)] for i in order[start:start + batch_size]]
            tr_losses.append(_batch_loss(batch, train=True))
        head.eval()
        with torch.no_grad():
            val_loss = _batch_loss(val_set, train=False)
        history.append({"epoch": epoch, "train_kl": float(np.mean(tr_losses)),
                        "val_kl": val_loss})
        print(f"  epoch {epoch:>3}: train KL {np.mean(tr_losses):.4f}  val KL {val_loss:.4f}")

    # Validation rank correlation: does the head rank nodes like occlusion does?
    head.eval()
    rhos = []
    with torch.no_grad():
        for s in val_set:
            pred = head(s["node_emb"].unsqueeze(0).to(device),
                        s["node_mask"].unsqueeze(0).to(device))[0].cpu().numpy()
            rhos.append(_spearman(pred[s["valid_idx"]], s["deltas"]))
    return {
        "n_train": len(train_set),
        "n_val": len(val_set),
        "history": history,
        "val_spearman_mean": float(np.mean(rhos)),
        "val_spearman_std": float(np.std(rhos)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="frozen GAT policy .pt")
    parser.add_argument("--area", default=None)
    parser.add_argument("--episodes", type=int, default=3,
                        help="rollout episodes used to harvest decisions")
    parser.add_argument("--max-decisions", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7,
                        help="rollout + split seed (deliberately ≠ the eval "
                             "seeds 42-44 so evaluation decisions are fresh)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--margin-cap", type=float, default=10.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: explainer_head.pt next to the ckpt")
    args = parser.parse_args()

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    if ckpt.get("policy_type", "gat") == "mlp":
        parser.error("the explainer reads GAT node embeddings — B1 has none")
    area = args.area or ckpt["env_config"]["area"]
    rng = np.random.default_rng(args.seed)

    print(f"policy: {args.checkpoint.name} (epoch {ckpt.get('epoch')})  area: {area}")
    print(f"[1/2] collecting occlusion-labelled decisions "
          f"({args.episodes} episodes, cap {args.max_decisions})…")
    t0 = time.time()
    samples = collect_dataset(
        policy, area, args.seed, args.episodes, args.max_decisions,
        args.margin_cap, device, rng,
    )
    print(f"      {len(samples)} decisions in {time.time()-t0:.0f}s")

    print(f"[2/2] training explainer head ({args.epochs} epochs)…")
    head = DecoupledExplainerHead(ExplainerConfig(
        hidden_dim=policy.config.hidden_dim,
        mlp_dim=args.mlp_dim,
        target_temperature=args.target_temperature,
        device=device,
    ))
    report = train_head(
        samples, head, args.epochs, args.lr, args.batch_size,
        args.val_frac, device, rng,
    )
    print(f"\nval Spearman(pred, occlusion) = "
          f"{report['val_spearman_mean']:+.3f} ± {report['val_spearman_std']:.3f}  "
          f"(n={report['n_val']})")

    out_path = args.out or args.checkpoint.parent / "explainer_head.pt"
    head.save(out_path, extra={
        "policy_checkpoint": str(args.checkpoint),
        "distill_args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
        "report": report,
    })
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"head:   {out_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
