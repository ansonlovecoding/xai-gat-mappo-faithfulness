"""Train a MAPPO dispatch policy on a Yubei SUMO scenario.

The loop collects one rollout per epoch and then performs a PPO update. It uses
one environment because TraCI provides one connection per process.

Logs to stdout as a table AND appends a JSONL row per epoch under
``runs/mappo/<area>_<timestamp>/train_log.jsonl``. Model checkpoints saved at
``ckpt_epoch_<N>.pt`` in the same dir.

Optional faithfulness sampling: pass --faith-every-epochs N to evaluate
DEF/WAMSN on a random sample of decisions from each Nth epoch's rollout.
Aggregate stats land in the JSONL row under `faithfulness`, ready for a
"faithfulness vs training progress" figure. Off by default because a
faithfulness pass adds ~1000 counterfactual forwards per triggering epoch.

Usage:
  python scripts/train.py --area central_park --epochs 50
  python scripts/train.py --area yuelai --epochs 20 --lr 1e-4
  python scripts/train.py --area central_park --degradation tunnel_triggered
  python scripts/train.py --area central_park --epochs 100 --faith-every-epochs 5
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
# Allows direct execution before the package has been installed editable.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DegradationConfig,
    DispatchEnv,
    DispatchEnvConfig,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    PPOConfig,
    collect_rollout,
    compute_gae,
    ppo_update,
    seed_everything,
)
from dispatch_marl.models import (  # noqa: E402
    DispatchGATPolicy,
    DispatchMLPPolicy,
    MLPPolicyConfig,
    PolicyConfig,
)
from dispatch_marl.provenance import (  # noqa: E402
    atomic_write_json,
    create_or_validate_manifest,
    file_inventory,
    runtime_provenance,
)
from dispatch_marl.scenario import demand_split_files  # noqa: E402
from dispatch_marl.runtime import choose_device  # noqa: E402
from dispatch_marl.training import AgentStep  # noqa: E402


def _sample_faithfulness(
    buffer: list[AgentStep],
    evaluator: FaithfulnessEvaluator,
    sample_size: int,
    device: str,
    rng: np.random.Generator,
) -> dict:
    """Evaluate DEF/WAMSN on a random sub-sample of decisions from the rollout.

    The buffer is already collected under the current policy weights, and
    each AgentStep carries the exact obs used to make its decision. That's
    what the faithfulness metrics need — no extra environment steps.

    Called between `collect_rollout` and `compute_gae` so the metrics reflect
    the same policy that produced the rollout.

    Returns aggregate stats suitable for a JSONL row; filters out decisions
    with zero valid reservations from DEF stats (their DEF is trivially 0
    and would bias the mean down).
    """
    n = min(sample_size, len(buffer))
    if n == 0:
        return {"n_scored": 0, "n_total": 0}

    # Random subset without replacement — keeps eval cheap and lets us
    # support later bootstrap confidence intervals.
    idx = rng.choice(len(buffer), size=n, replace=False)

    def_scores: list[float] = []
    def_margins: list[float] = []
    g_comps: list[float] = []
    g_suffs: list[float] = []
    wamsns: list[float] = []
    n_scored = 0
    for i in idx:
        step = buffer[int(i)]
        # obs dict is per-agent numpy; add batch dim and move to device.
        obs = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in step.obs.items()
        }
        result = evaluator.evaluate_decision(obs, action=step.action)
        # WAMSN is meaningful for every decision (needs no valid-reservation
        # signal), DEF is only meaningful when there's at least one non-self
        # ablatable node. Filter DEF to match the ablation logic in
        # eval_policy._summarise_faithfulness.
        n_valid_res = int(step.obs["reservations_mask"].sum())
        wamsns.append(result.wamsn)
        if n_valid_res > 0:
            def_scores.append(result.def_score)
            def_margins.append(result.def_margin)
            g_comps.append(result.g_comp)
            g_suffs.append(result.g_suff)
            n_scored += 1

    out: dict = {"n_scored": n_scored, "n_total": n}
    if n_scored > 0:
        out["def_mean"] = float(np.mean(def_scores))
        out["def_std"] = float(np.std(def_scores))
        out["def_m_mean"] = float(np.mean(def_margins))
        out["def_m_std"] = float(np.std(def_margins))
        out["g_comp_mean"] = float(np.mean(g_comps))
        out["g_suff_mean"] = float(np.mean(g_suffs))
    if wamsns:
        out["wamsn_mean"] = float(np.mean(wamsns))
        out["wamsn_std"] = float(np.std(wamsns))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="central_park",
                        choices=["central_park", "yuelai", "xiantao"])
    parser.add_argument("--degradation", default="off",
                        choices=["off", "tunnel_triggered", "random_dropout"])
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--outage-duration", type=float, default=0.0,
                        help="observation-layer outage duration after a trigger "
                             "(seconds)")
    # Reward shaping.
    parser.add_argument(
        "--completed-journey-reward", "--pickup-reward",
        dest="pickup_reward", type=float, default=10.0,
        help="team reward for one passenger reaching their destination",
    )
    parser.add_argument("--dispatch-reward", type=float, default=0.5)
    parser.add_argument("--dispatch-credit-reward", type=float, default=0.5,
                        help="extra training-only credit for the taxi whose dispatch succeeds")
    parser.add_argument("--wait-lambda", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic-torch", action="store_true",
                        help="request deterministic PyTorch kernels where available")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="linearly decay the learning rate to zero across training")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--entropy-floor", type=float, default=-1.0,
                        help="adapt entropy regularisation below this decision "
                             "entropy; negative disables")
    parser.add_argument("--max-ent-coef", type=float, default=-1.0,
                        help="upper bound for adaptive entropy coefficient; "
                             "negative leaves it unbounded")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-clip-ratio", type=float, default=0.2,
                        help="PPO value-function clipping range; negative disables")
    parser.add_argument("--target-kl", type=float, default=0.015,
                        help="stop a PPO update early above this approximate KL; negative disables")
    parser.add_argument("--policy", default="gat", choices=["gat", "mlp"],
                        help="gat = GAT-MAPPO (B2, the proposed model); "
                             "mlp = MAPPO+MLP baseline (B1, no graph — "
                             "no attention channel, so faithfulness "
                             "sampling is unavailable)")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-gat-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=176,
                        help="hidden width for --policy mlp (176 ≈ param-"
                             "matches the default GAT config: ~106k vs ~114k)")
    parser.add_argument("--mlp-layers", type=int, default=3,
                        help="trunk depth for --policy mlp")
    parser.add_argument("--centralised-critic",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="MAPPO CTDE: critic is fed a joint pooled embedding "
                             "over all agents in the batch (proposal §7.7). "
                             "DEFAULT: enabled. Use --no-centralised-critic to "
                             "fall back to a per-agent V.")
    parser.add_argument("--critic-encoder-gradient-scale", type=float, default=1.0,
                        help="fraction of critic gradient allowed into the shared "
                             "policy encoder (0=actor-only encoder, 1=fully shared)")
    parser.add_argument("--save-every", type=int, default=10)
    # Best-checkpoint tracking: after each epoch, compute the mean of the last
    # `--best-window` training-episode pickup counts. Whenever that rolling
    # mean improves on the previous best, snapshot the current policy to
    # `ckpt_best.pt` and update `best_metadata.json`. Set --best-window 0 to
    # disable. Kept as a rolling training statistic (rather than an extra eval
    # rollout per epoch) to keep training cheap.
    parser.add_argument("--best-window", type=int, default=10,
                        help="epochs to average for rolling-mean best tracking; 0 disables")
    # Faithfulness sampling. Off by default because a full pass adds
    # ~sample_size × 25 counterfactual forwards on top of the epoch.
    parser.add_argument("--faith-every-epochs", type=int, default=0,
                        help="evaluate DEF/WAMSN on rollout every Nth epoch; 0 disables")
    parser.add_argument("--faith-sample-size", type=int, default=128,
                        help="agent-steps sampled from the rollout buffer per faith eval")
    parser.add_argument("--faith-top-k", type=int, nargs="+", default=[1, 2, 3],
                        help="top-k values used for DEF's comp/suff terms")
    parser.add_argument("--faith-random-baselines", type=int, default=3,
                        help="random subsets per k; kept lower than eval-time (5) for speed")
    parser.add_argument("--faith-random-baseline", default="type_matched",
                        choices=["uniform", "type_matched"],
                        help="type_matched is the reportable construct-validity control")
    parser.add_argument("--demand-split", default=None,
                        choices=["train", "val", "test"],
                        help="rotate rider-demand variants from this chronological "
                             "split (demand_manifest.json) round-robin across "
                             "epochs. Default: the single committed demand file, "
                             "as before. Use 'train' for protocol-compliant runs.")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "runs" / "mappo")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="exact immutable output directory; default uses log-dir/area_timestamp")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or mps; default: auto")
    args = parser.parse_args()
    training_started = time.monotonic()

    if args.faith_every_epochs > 0 and args.policy == "mlp":
        parser.error("--faith-every-epochs requires --policy gat: the MLP "
                     "baseline has no attention channel to score")

    seed_settings = seed_everything(
        args.seed, deterministic_torch=args.deterministic_torch
    )
    update_rng = np.random.default_rng(args.seed)
    device = args.device or choose_device()

    run_dir = args.run_dir or args.log_dir / f"{args.area}_{int(time.time())}"
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"run directory is not empty: {run_dir}; training runs are immutable")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train_log.jsonl"
    atomic_write_json(run_dir / "args.json", vars(args))

    print(f"run dir: {run_dir}")
    print(f"device:  {device}")

    # ---- env
    env_cfg = DispatchEnvConfig(
        area=args.area,
        seed=args.seed,
        pickup_reward=args.pickup_reward,
        dispatch_reward=args.dispatch_reward,
        dispatch_credit_reward=args.dispatch_credit_reward,
        wait_penalty_lambda=args.wait_lambda,
        degradation=DegradationConfig(mode=args.degradation,
                                      dropout_rate=args.dropout_rate,
                                      outage_duration_s=args.outage_duration),
    )
    env = DispatchEnv(env_cfg)

    # Demand-variant rotation (E-section protocol).
    demand_files: list[str] = []
    if args.demand_split:
        demand_files = demand_split_files(args.area, args.demand_split)
        print(f"demand:  {len(demand_files)} '{args.demand_split}' variants, "
              f"rotated round-robin per epoch")

    scenario_dir = PROJECT_ROOT / "scenarios" / "yubei" / args.area
    provenance_inputs = [
        scenario_dir / f"{args.area}.sumocfg",
        scenario_dir / f"{args.area}.net.xml",
        scenario_dir / f"{args.area}.rou.xml",
        scenario_dir / "tunnels.json",
        scenario_dir / "demand_manifest.json",
        *(scenario_dir / name for name in demand_files),
    ]
    input_inventory = file_inventory(provenance_inputs, PROJECT_ROOT)
    runtime = runtime_provenance(PROJECT_ROOT)
    specification = {
        "schema_version": 2,
        "protocol_version": "2.0",
        "reward_timeline": "semi_markov_decision_intervals",
        "kind": "training",
        "arguments": {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        "seed_settings": seed_settings,
        "code_revision": runtime["git"]["revision"],
        "tracked_diff_sha256": runtime["git"]["tracked_diff_sha256"],
        "input_inventory": input_inventory,
    }
    create_or_validate_manifest(
        run_dir / "manifest.json",
        specification,
        {
            **runtime,
            "inputs": input_inventory,
        },
    )

    # ---- policy
    if args.policy == "mlp":
        pol_cfg = MLPPolicyConfig(
            k_neighbors=env_cfg.k_neighbors,
            k_reservations=env_cfg.k_reservations,
            hidden_dim=args.mlp_hidden_dim,
            n_layers=args.mlp_layers,
            centralised_critic=args.centralised_critic,
            critic_encoder_gradient_scale=args.critic_encoder_gradient_scale,
            device=device,
        )
        policy = DispatchMLPPolicy(pol_cfg)
    else:
        pol_cfg = PolicyConfig(
            k_neighbors=env_cfg.k_neighbors,
            k_reservations=env_cfg.k_reservations,
            hidden_dim=args.hidden_dim,
            n_gat_layers=args.n_gat_layers,
            n_heads=args.n_heads,
            centralised_critic=args.centralised_critic,
            critic_encoder_gradient_scale=args.critic_encoder_gradient_scale,
            device=device,
        )
        policy = DispatchGATPolicy(pol_cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"policy:  {args.policy} — {sum(p.numel() for p in policy.parameters()):,} params")

    ppo_cfg = PPOConfig(
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        entropy_floor=args.entropy_floor if args.entropy_floor >= 0 else None,
        max_ent_coef=args.max_ent_coef if args.max_ent_coef >= 0 else None,
        max_grad_norm=args.max_grad_norm,
        value_clip_ratio=(args.value_clip_ratio
                          if args.value_clip_ratio >= 0 else None),
        target_kl=args.target_kl if args.target_kl >= 0 else None,
    )

    # ---- faithfulness sampler (built once so the RNG stays consistent)
    faith_evaluator: FaithfulnessEvaluator | None = None
    faith_rng: np.random.Generator | None = None
    if args.faith_every_epochs > 0:
        faith_cfg = FaithfulnessConfig(
            top_k_values=tuple(args.faith_top_k),
            n_random_baselines=args.faith_random_baselines,
            seed=args.seed,
            random_baseline=args.faith_random_baseline,
        )
        faith_evaluator = FaithfulnessEvaluator(policy, faith_cfg)
        faith_rng = np.random.default_rng(args.seed)
        print(f"faith:   every {args.faith_every_epochs} epochs, "
              f"sample_size={args.faith_sample_size}, "
              f"top_k={args.faith_top_k}, random_baselines={args.faith_random_baselines}")

    # ---- table header
    header = ("epoch", "pickups", "reward", "n_steps", "π_loss", "V_loss", "H_dec", "noop", "KL", "clipfrac", "wall_s")
    fmt = "{:>5}  {:>7}  {:>+8.2f}  {:>7}  {:>8.4f}  {:>8.4f}  {:>7.3f}  {:>6.3f}  {:>7.4f}  {:>8.3f}  {:>6.1f}"
    print()
    print("{:>5}  {:>7}  {:>8}  {:>7}  {:>8}  {:>8}  {:>7}  {:>6}  {:>7}  {:>8}  {:>6}".format(*header))
    print("-" * 104)

    # ---- best-checkpoint tracking state
    recent_pickups: list[int] = []
    best_rolling_mean = float("-inf")
    best_epoch = -1
    best_ckpt_path = run_dir / "ckpt_best.pt"
    best_meta_path = run_dir / "best_metadata.json"

    def _save_best(epoch: int, rolling_mean: float) -> None:
        torch.save({
            "schema_version": 2,
            "protocol_version": "2.0",
            "epoch": epoch,
            "policy_type": args.policy,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_config": pol_cfg.__dict__,
            "env_config": {**env_cfg.__dict__,
                           "degradation": env_cfg.degradation.__dict__},
            "ppo_config": ppo_cfg.__dict__,
            "best": {
                "rolling_mean_completed_passenger_journeys": rolling_mean,
                "rolling_mean_pickups": rolling_mean,
                "window": args.best_window,
            },
        }, best_ckpt_path)
        best_meta_path.write_text(json.dumps({
            "best_epoch": epoch,
            "best_rolling_mean_completed_passenger_journeys": rolling_mean,
            "best_rolling_mean_pickups": rolling_mean,
            "window": args.best_window,
            "note": "Rolling mean of completed passenger journeys over the last "
                    f"{args.best_window} epochs. Updated whenever it improves.",
        }, indent=2))

    # ---- train loop
    for epoch in range(args.epochs):
        t0 = time.time()

        if args.anneal_lr:
            learning_rate = args.lr * (1.0 - epoch / max(1, args.epochs))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        else:
            learning_rate = args.lr

        # 1. rollout (on this epoch's demand variant, if rotation is on)
        reset_options = None
        if demand_files:
            reset_options = {"taxi_route_file": demand_files[epoch % len(demand_files)]}
        buffer, ep_stats = collect_rollout(env, policy, device=device,
                                           reset_options=reset_options)
        decision_steps = [
            step for step in buffer
            if np.asarray(step.obs["reservations_mask"]).sum() > 0
        ]
        decision_noop_rate = (
            sum(step.action == 0 for step in decision_steps) / len(decision_steps)
            if decision_steps else 0.0
        )

        # 1b. Optional faithfulness sample on the just-rolled-out buffer.
        # Done before compute_gae/ppo_update so the metrics reflect the same
        # policy weights that produced the rollout.
        faith_stats: dict | None = None
        if (faith_evaluator is not None
                and faith_rng is not None
                and epoch % args.faith_every_epochs == 0):
            faith_stats = _sample_faithfulness(
                buffer, faith_evaluator, args.faith_sample_size,
                device=device, rng=faith_rng,
            )

        # 2. GAE
        compute_gae(buffer, gamma=args.gamma, gae_lambda=args.gae_lambda)

        # 3. PPO update
        losses = ppo_update(
            policy, optimizer, buffer, ppo_cfg, device=device, rng=update_rng
        )
        loss_d = losses.as_dict()

        elapsed = time.time() - t0

        # ---- best-checkpoint tracking (rolling mean of training pickups)
        is_new_best = False
        rolling_mean = None
        if args.best_window > 0:
            recent_pickups.append(ep_stats.total_pickups)
            if len(recent_pickups) > args.best_window:
                recent_pickups.pop(0)
            if len(recent_pickups) >= args.best_window:
                rolling_mean = sum(recent_pickups) / len(recent_pickups)
                if rolling_mean > best_rolling_mean:
                    best_rolling_mean = rolling_mean
                    best_epoch = epoch
                    is_new_best = True
                    _save_best(epoch, rolling_mean)

        row = {
            "epoch": epoch,
            "completed_passenger_journeys": (
                ep_stats.total_completed_passenger_journeys
            ),
            "pickups": ep_stats.total_pickups,
            "reward": round(ep_stats.total_reward, 3),
            "n_agent_steps": ep_stats.n_agent_steps,
            "wall_s": round(elapsed, 1),
            "rolling_mean_pickups": round(rolling_mean, 3) if rolling_mean is not None else None,
            "is_new_best": is_new_best,
            "decision_steps": len(decision_steps),
            "decision_noop_rate": round(decision_noop_rate, 5),
            "learning_rate": learning_rate,
            **{k: round(v, 5) for k, v in loss_d.items()},
        }
        if faith_stats is not None:
            # Round for readability; consumers of the JSONL still get 4 decimals.
            row["faithfulness"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in faith_stats.items()
            }
        with log_file.open("a") as f:
            f.write(json.dumps(row) + "\n")

        marker = "  ★ new best" if is_new_best else ""
        if faith_stats is not None and faith_stats.get("n_scored", 0) > 0:
            marker = f"  DEF={faith_stats['def_mean']:+.3f} WAMSN={faith_stats.get('wamsn_mean', 0.0):.3f}{marker}"
        print(fmt.format(
            epoch,
            ep_stats.total_pickups,
            ep_stats.total_reward,
            ep_stats.n_agent_steps,
            loss_d["policy_loss"],
            loss_d["value_loss"],
            loss_d["decision_entropy"],
            decision_noop_rate,
            loss_d["approx_kl"],
            loss_d["clipfrac"],
            elapsed,
        ) + marker)

        # 4. checkpoint
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs - 1):
            ckpt = run_dir / f"ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "schema_version": 2,
                "protocol_version": "2.0",
                "epoch": epoch,
                "policy_type": args.policy,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "policy_config": pol_cfg.__dict__,
                "env_config": {**env_cfg.__dict__,
                               "degradation": env_cfg.degradation.__dict__},
                "ppo_config": ppo_cfg.__dict__,
            }, ckpt)

    env.close()
    final_ckpt_path = run_dir / "ckpt_final.pt"
    torch.save({
        "schema_version": 2,
        "protocol_version": "2.0",
        "epoch": args.epochs - 1,
        "policy_type": args.policy,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "policy_config": pol_cfg.__dict__,
        "env_config": {**env_cfg.__dict__,
                       "degradation": env_cfg.degradation.__dict__},
        "ppo_config": ppo_cfg.__dict__,
        "seed_settings": seed_settings,
    }, final_ckpt_path)
    print(f"\ndone. logs: {log_file}")
    atomic_write_json(run_dir / "timing.json", {
        "status": "completed",
        "total_training_wall_seconds": time.monotonic() - training_started,
        "scope": "initialization, all epochs and checkpoint saves; excludes validation selection",
    })
    print(f"final:  {final_ckpt_path}")
    if best_epoch >= 0:
        print(f"best:   epoch {best_epoch}, rolling mean pickups = {best_rolling_mean:.2f}")
        print(f"        checkpoint: {best_ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
