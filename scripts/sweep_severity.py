"""Severity sweep: the dataset H1–H4 are tested on.

Runs one checkpoint across a grid of degradation severities × seeds and
writes one JSON per cell plus a versioned manifest. Two severity axes,
matching the two degradation modes:

  * **random_dropout axis** — severity = dropout rate, at fixed position
    noise. The clean condition (rate 0) is the shared reference point.
  * **tunnel_triggered axis** — the degradation *rate* is fixed by network
    geography (a taxi is degraded iff it's in a tunnel), so severity =
    position-noise magnitude instead.

Per cell we record policy metrics (pickups / reward / wait), the empirical
degradation rate, aggregate DEF/WAMSN/drift, and the raw per-decision
faithfulness records (the hypothesis analysis bootstraps over decisions —
see `src/dispatch_marl/README.md` §"Consequences for how results are
reported").

Downstream: `scripts/analyze_hypotheses.py <sweep-dir>`.

Usage:
  python scripts/sweep_severity.py <ckpt>
  python scripts/sweep_severity.py <ckpt> --episodes 3 --seeds 42 43 44
  python scripts/sweep_severity.py <ckpt> --dropout-rates 0.1 0.3 --noise-levels 20 80
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_policy import _choose_device, _load_policy  # noqa: E402
from eval_degradation_ablation import _run_condition  # noqa: E402

from dispatch_marl import FaithfulnessConfig, FaithfulnessEvaluator  # noqa: E402


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        ).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help=".pt file saved by train.py")
    parser.add_argument("--area", default=None,
                        help="override area; default: the checkpoint's training area")
    parser.add_argument("--episodes", type=int, default=3,
                        help="episodes per (cell × seed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dropout-rates", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.4],
                        help="random_dropout severity axis (0 = clean is always run)")
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[10.0, 20.0, 40.0, 80.0],
                        help="tunnel_triggered severity axis: position noise σ in metres")
    parser.add_argument("--dropout-noise-m", type=float, default=20.0,
                        help="fixed position noise σ used on the random_dropout axis")
    parser.add_argument("--deterministic", action="store_true",
                        help="argmax actions instead of sampling. Default is "
                             "STOCHASTIC: it matches the training-time and "
                             "comparison-table protocol, and argmax on an "
                             "entropy-collapsed policy degenerates to all-"
                             "no-op (0 pickups), which makes H2's performance "
                             "rate vacuous")
    parser.add_argument("--faithfulness-every", type=int, default=5)
    parser.add_argument("--faithfulness-top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--faithfulness-random-baselines", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="sweep output dir (default runs/sweeps/<ckpt-stem>_<ts>)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = args.device or _choose_device()
    policy, ckpt = _load_policy(args.checkpoint, device)
    if ckpt.get("policy_type", "gat") == "mlp":
        parser.error("the severity sweep scores DEF/WAMSN/drift, which needs "
                     "a GAT checkpoint — B1 has no attention channel")
    area = args.area or ckpt["env_config"]["area"]
    aoi_unaware = bool(ckpt["env_config"].get("aoi_unaware", False))

    out_dir = args.out or (PROJECT_ROOT / "runs" / "sweeps"
                           / f"{args.checkpoint.stem}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cell list. Severity axis + level are recorded per cell so the analysis
    # script never has to reverse-engineer them from the config.
    cells: list[dict] = [
        {"axis": "clean", "level": 0.0, "mode": "off",
         "dropout_rate": 0.0, "noise_m": 0.0},
    ]
    for r in args.dropout_rates:
        cells.append({"axis": "dropout_rate", "level": float(r),
                      "mode": "random_dropout",
                      "dropout_rate": float(r), "noise_m": args.dropout_noise_m})
    for n in args.noise_levels:
        cells.append({"axis": "tunnel_noise", "level": float(n),
                      "mode": "tunnel_triggered",
                      "dropout_rate": 0.0, "noise_m": float(n)})

    manifest = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "policy_type": ckpt.get("policy_type", "gat"),
        "area": area,
        "aoi_unaware": aoi_unaware,
        "episodes_per_cell_seed": args.episodes,
        "stochastic": not args.deterministic,
        "seeds": args.seeds,
        "cells": cells,
        "faithfulness_config": {
            "top_k_values": list(args.faithfulness_top_k),
            "n_random_baselines": args.faithfulness_random_baselines,
            "faithfulness_every": args.faithfulness_every,
        },
        "git_rev": _git_rev(),
        "created_unix": int(time.time()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_total = len(cells) * len(args.seeds)
    print(f"sweep: {len(cells)} cells × {len(args.seeds)} seeds = {n_total} runs "
          f"× {args.episodes} episodes")
    print(f"out:   {out_dir}")
    print()

    i = 0
    for cell in cells:
        for seed in args.seeds:
            i += 1
            cell_name = f"{cell['axis']}_{cell['level']:g}_seed{seed}"
            cell_path = out_dir / f"{cell_name}.json"
            if cell_path.exists():
                print(f"[{i:>3}/{n_total}] {cell_name}: exists, skipping")
                continue
            t0 = time.time()
            # Fresh evaluator per cell so the random-baseline RNG is
            # deterministic per (cell, seed), independent of run order.
            evaluator = FaithfulnessEvaluator(policy, FaithfulnessConfig(
                top_k_values=tuple(args.faithfulness_top_k),
                n_random_baselines=args.faithfulness_random_baselines,
                seed=seed,
            ))
            result = _run_condition(
                policy, device, area, seed, args.episodes,
                degradation_mode=cell["mode"],
                dropout_rate=cell["dropout_rate"],
                faith_evaluator=evaluator,
                faith_every=args.faithfulness_every,
                aoi_unaware=aoi_unaware,
                position_noise_m=cell["noise_m"],
                keep_records=True,
                stochastic=not args.deterministic,
            )
            result["cell"] = {**cell, "seed": seed}
            cell_path.write_text(json.dumps(result, indent=2))
            f = result["faithfulness"]
            print(f"[{i:>3}/{n_total}] {cell_name}: "
                  f"pickups={result['mean_pickups']:.1f}  "
                  f"deg_rate={result['empirical_degradation_rate']:.3f}  "
                  f"DEF={f.get('def_mean', float('nan')):+.3f}  "
                  f"WAMSN={f.get('wamsn_mean', float('nan')):.3f}  "
                  f"drift={f.get('drift_mean', float('nan')):.4f}  "
                  f"({time.time() - t0:.0f}s)")

    print()
    print(f"done. analyse with: python scripts/analyze_hypotheses.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
