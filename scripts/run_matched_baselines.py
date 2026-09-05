"""Evaluate legal-random and nearest-request baselines on a dissertation matrix.

The runner mirrors the held-out protocol in ``dissertation_v8.toml``: the same
area, test-demand split, evaluation seeds, episodes per seed, and demand-file
rotation. Results are written to a new immutable JSON file with per-episode
records, aggregate confidence intervals, and runtime provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl import (  # noqa: E402
    DispatchEnv,
    DispatchEnvConfig,
    LegalRandomPolicy,
    NearestReservationPolicy,
    derive_seed,
)
from dispatch_marl.provenance import (  # noqa: E402
    atomic_write_json,
    file_inventory,
    runtime_provenance,
)
from dispatch_marl.scenario import demand_split_files  # noqa: E402


POLICIES = {
    "legal_random": lambda seed: LegalRandomPolicy(seed=seed),
    "greedy_nearest": lambda seed: NearestReservationPolicy(),
}


def run_episode(env: DispatchEnv, policy, *, episode: int,
                demand_variant: str) -> dict:
    started = time.time()
    obs, _ = env.reset(options={"taxi_route_file": demand_variant})
    policy.reset()
    total_reward = 0.0
    total_completed_journeys = 0
    mean_wait = 0.0
    steps = 0

    while not env.done:
        actions = policy.act(obs) if obs else {}
        obs, _, _, _, _ = env.step(actions)
        metrics = env.last_step_metrics
        total_reward += metrics.team_reward
        total_completed_journeys += metrics.completed_passenger_journeys
        mean_wait = metrics.mean_pending_wait_s
        steps += 1

    return {
        "episode": episode,
        "demand_variant": demand_variant,
        "completed_passenger_journeys": total_completed_journeys,
        "total_pickups": total_completed_journeys,
        "total_reward": total_reward,
        "final_mean_pending_wait_s": mean_wait,
        "rl_steps": steps,
        "wall_time_s": time.time() - started,
    }


def cluster_bootstrap_mean_ci(values: list[float], clusters: list[str], *,
                              seed: int, n_boot: int = 20_000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    cluster_array = np.asarray(clusters)
    if array.size == 0 or cluster_array.size != array.size:
        raise ValueError("cannot summarise an empty sample")
    unique_clusters = np.unique(cluster_array)
    cluster_means = np.asarray([
        array[cluster_array == cluster].mean() for cluster in unique_clusters
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, cluster_means.size, size=(n_boot, cluster_means.size)
    )
    means = cluster_means[indices].mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "std_between_clusters": (
            float(cluster_means.std(ddof=1)) if cluster_means.size > 1 else 0.0
        ),
        "ci95": [float(np.quantile(means, 0.025)),
                 float(np.quantile(means, 0.975))],
        "n_episode_records": int(array.size),
        "n_independent_clusters": int(cluster_means.size),
    }


def summarise(records: list[dict], *, policy_name: str, seed: int) -> dict:
    if policy_name == "greedy_nearest":
        clusters = [row["demand_variant"] for row in records]
        cluster_unit = "demand variant (deterministic policy repeats across seeds)"
    else:
        clusters = [
            f"{row['evaluation_seed']}:{row['episode']}" for row in records
        ]
        cluster_unit = "evaluation seed x episode"
    return {
        "cluster_unit": cluster_unit,
        "completed_passenger_journeys": cluster_bootstrap_mean_ci(
            [row["completed_passenger_journeys"] for row in records],
            clusters,
            seed=seed,
        ),
        "reward": cluster_bootstrap_mean_ci(
            [row["total_reward"] for row in records], clusters, seed=seed + 1),
        "final_mean_pending_wait_s": cluster_bootstrap_mean_ci(
            [row["final_mean_pending_wait_s"] for row in records],
            clusters, seed=seed + 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/dissertation_v8.toml",
    )
    parser.add_argument(
        "--policies", nargs="+", choices=sorted(POLICIES),
        default=list(POLICIES),
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT /
        "runs/dissertation_v9_exposure_audit/matched_baselines.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite explicitly")

    config = tomllib.loads(args.config.read_text())
    evaluation = config["evaluation"]
    area = config["area"]
    demand_split = evaluation["demand_split"]
    variants = demand_split_files(area, demand_split)
    seeds = [int(seed) for seed in evaluation["seeds"]]
    episodes_per_seed = int(evaluation["episodes_per_cell_seed"])

    records: list[dict] = []
    for policy_name in args.policies:
        for evaluation_seed in seeds:
            env = DispatchEnv(DispatchEnvConfig(area=area, seed=evaluation_seed))
            try:
                for episode in range(episodes_per_seed):
                    action_seed = derive_seed(
                        evaluation_seed, "matched_baseline", policy_name, episode
                    )
                    policy = POLICIES[policy_name](action_seed)
                    variant = variants[episode % len(variants)]
                    print(
                        f"[{policy_name}] seed={evaluation_seed} "
                        f"episode={episode} demand={variant}",
                        flush=True,
                    )
                    row = run_episode(
                        env, policy, episode=episode, demand_variant=variant
                    )
                    row.update({
                        "policy": policy_name,
                        "evaluation_seed": evaluation_seed,
                        "action_seed": action_seed,
                    })
                    records.append(row)
                    print(
                        f"  pickups={row['total_pickups']} "
                        f"reward={row['total_reward']:+.2f} "
                        f"wall={row['wall_time_s']:.1f}s",
                        flush=True,
                    )
            finally:
                env.close()

    summaries = {}
    for index, policy_name in enumerate(args.policies):
        policy_records = [row for row in records if row["policy"] == policy_name]
        summaries[policy_name] = summarise(
            policy_records,
            policy_name=policy_name,
            seed=derive_seed(2026, "matched_baseline_bootstrap", index),
        )

    scenario_dir = PROJECT_ROOT / "scenarios" / "yubei" / area
    manifest_path = scenario_dir / "demand_manifest.json"
    inputs = [
        args.config,
        manifest_path,
        scenario_dir / f"{area}.sumocfg",
        scenario_dir / f"{area}.net.xml",
        scenario_dir / f"{area}.rou.xml",
        scenario_dir / "tunnels.json",
        *(scenario_dir / variant for variant in variants),
    ]
    payload = {
        "schema_version": 1,
        "experiment": "dissertation_matched_baselines",
        "protocol": {
            "source_config": str(args.config.relative_to(PROJECT_ROOT)),
            "area": area,
            "demand_split": demand_split,
            "evaluation_seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "policies": args.policies,
            "random_policy": "uniform over no-op and valid request actions",
        },
        "summary": summaries,
        "records": records,
        "provenance": {
            **runtime_provenance(PROJECT_ROOT),
            "inputs": file_inventory(
                inputs, root=PROJECT_ROOT
            ),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(summaries, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
