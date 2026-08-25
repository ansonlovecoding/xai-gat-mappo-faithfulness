"""Severity sweep: the dataset the dissertation's story is tested on.

Primary (and default only) axis — **max-AoI ladder** (proposal §7.2):
tunnel-triggered signal loss with freeze corruption, severity = the
"maximum AoI" level {5, 15, 30, 60} s implemented as the outage duration
(the signal stays lost until AoI reaches the level; long tunnel transits
put a geography floor under it, so the empirical AoI is recorded too).
The clean condition (level 0) is the shared reference point.

Appendix axis (opt-in, `--with-dropout-axis`): matched-structure
random_dropout at the same outage ladder — same staleness depth, no
spatial correlation. Kept out of the main text per the simplified
three-act story.

Episodes run on the held-out **test** demand split by default
(`demand_manifest.json`; disable with `--demand-split none` for legacy
comparison). Per cell we record policy metrics, empirical degradation
rate and AoI, aggregate DEF/WAMSN/drift, and raw per-decision records.

Downstream: `scripts/analyze_hypotheses.py <sweep-dir>`.

Usage:
  python scripts/sweep_severity.py <ckpt>
  python scripts/sweep_severity.py <ckpt> --episodes 3 --seeds 42 43 44
  python scripts/sweep_severity.py <ckpt> --aoi-levels 15 60 --with-dropout-axis
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_degradation_ablation import _run_condition  # noqa: E402

from dispatch_marl import FaithfulnessConfig, FaithfulnessEvaluator  # noqa: E402
from dispatch_marl.provenance import (  # noqa: E402
    atomic_write_json,
    config_hash,
    create_or_validate_manifest,
    file_inventory,
    runtime_provenance,
    sha256_file,
)
from dispatch_marl.scenario import demand_split_files  # noqa: E402
from dispatch_marl.runtime import choose_device as _choose_device  # noqa: E402
from dispatch_marl.runtime import load_policy as _load_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help=".pt file saved by train.py")
    parser.add_argument("--area", default=None,
                        help="override area; default: the checkpoint's training area")
    parser.add_argument("--episodes", type=int, default=3,
                        help="episodes per (cell × seed); rotates through the "
                             "demand-split variants, so 3 covers all 3 test files")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--aoi-levels", type=float, nargs="+",
                        default=[5.0, 15.0, 30.0, 60.0],
                        help="max-AoI severity ladder in seconds (proposal §7.2); "
                             "0 = clean is always run")
    parser.add_argument("--with-dropout-axis", action="store_true",
                        help="ALSO run the matched random_dropout axis at the same "
                             "AoI ladder (appendix robustness check)")
    parser.add_argument("--dropout-rate", type=float, default=0.05,
                        help="trigger rate for the appendix dropout axis")
    parser.add_argument("--demand-split", default="test",
                        choices=["test", "val", "train", "none"],
                        help="which demand variants episodes run on (default: "
                             "held-out test). 'none' = the single committed "
                             "demand file (legacy)")
    parser.add_argument("--corruption", default="freeze", choices=["freeze", "noise"],
                        help="freeze = proposal semantics (default); noise = legacy")
    parser.add_argument("--deterministic", action="store_true",
                        help="argmax actions instead of sampling. Default is "
                             "STOCHASTIC: argmax on an entropy-collapsed policy "
                             "degenerates to all-no-op, making the performance "
                             "axis vacuous")
    parser.add_argument("--faithfulness-every", type=int, default=5)
    parser.add_argument("--faithfulness-top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--faithfulness-random-baselines", type=int, default=5)
    parser.add_argument("--random-baseline", default="type_matched",
                        choices=["uniform", "type_matched"],
                        help="random-baseline sampling scheme. 'uniform' is the "
                             "standard ERASER-style protocol and is artifact-prone "
                             "in this architecture: uniform draws hit reservation "
                             "nodes (deleting candidate actions, clamping the "
                             "margin) far more often than the attention top-k. "
                             "'type_matched' draws subsets with the SAME "
                             "taxi/reservation composition as the top-k set and "
                             "is the reportable metric (see "
                             "results/story_freeze_v1/audit/type_matched_control.json)")
    parser.add_argument("--exclusion-variant", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="ALSO compute the construct-validity DEF variant "
                             "with the chosen reservation's node protected from "
                             "occlusion (~2x forwards per decision)")
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

    demand_files: list[str] = []
    if args.demand_split != "none":
        demand_files = demand_split_files(area, args.demand_split)

    # Cell list. Severity axis + level are recorded per cell so the analysis
    # script never has to reverse-engineer them from the config.
    cells: list[dict] = [
        {"axis": "clean", "level": 0.0, "mode": "off",
         "dropout_rate": 0.0, "outage_s": 0.0},
    ]
    for s in args.aoi_levels:
        cells.append({"axis": "max_aoi", "level": float(s),
                      "mode": "tunnel_triggered",
                      "dropout_rate": 0.0, "outage_s": float(s)})
    if args.with_dropout_axis:
        for s in args.aoi_levels:
            cells.append({"axis": "dropout_max_aoi", "level": float(s),
                          "mode": "random_dropout",
                          "dropout_rate": args.dropout_rate, "outage_s": float(s)})

    checkpoint_sha256 = sha256_file(args.checkpoint)
    try:
        checkpoint_ref = str(args.checkpoint.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        checkpoint_ref = str(args.checkpoint.resolve())
    scenario_dir = PROJECT_ROOT / "scenarios" / "yubei" / area
    inputs = [
        args.checkpoint,
        scenario_dir / f"{area}.sumocfg",
        scenario_dir / f"{area}.net.xml",
        scenario_dir / f"{area}.rou.xml",
        scenario_dir / "tunnels.json",
        scenario_dir / "demand_manifest.json",
        *(scenario_dir / name for name in demand_files),
    ]
    input_inventory = file_inventory(inputs, PROJECT_ROOT)
    runtime = runtime_provenance(PROJECT_ROOT)
    specification = {
        "kind": "severity_sweep",
        "checkpoint": checkpoint_ref,
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": ckpt.get("epoch"),
        "policy_type": ckpt.get("policy_type", "gat"),
        "area": area,
        "aoi_unaware": aoi_unaware,
        "corruption": args.corruption,
        "episodes_per_cell_seed": args.episodes,
        "stochastic": not args.deterministic,
        "demand_split": args.demand_split,
        "demand_files": demand_files,
        "seeds": args.seeds,
        "cells": cells,
        "faithfulness_config": {
            "top_k_values": list(args.faithfulness_top_k),
            "n_random_baselines": args.faithfulness_random_baselines,
            "faithfulness_every": args.faithfulness_every,
            "random_baseline": args.random_baseline,
            "exclusion_variant": args.exclusion_variant,
        },
        "code_revision": runtime["git"]["revision"],
        "tracked_diff_sha256": runtime["git"]["tracked_diff_sha256"],
        "input_inventory": input_inventory,
    }
    experiment_id = config_hash(specification)
    out_dir = args.out or (PROJECT_ROOT / "runs" / "sweeps" / experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = out_dir / "cells"
    cell_dir.mkdir(exist_ok=True)

    manifest = create_or_validate_manifest(
        out_dir / "manifest.json",
        specification,
        {**runtime, "inputs": input_inventory},
    )

    n_total = len(cells) * len(args.seeds)
    print(f"sweep: {len(cells)} cells × {len(args.seeds)} seeds = {n_total} runs "
          f"× {args.episodes} episodes  |  corruption={args.corruption}  "
          f"demand={args.demand_split}")
    print(f"out:   {out_dir}")
    print()

    i = 0
    for cell in cells:
        for seed in args.seeds:
            i += 1
            cell_name = f"{cell['axis']}_{cell['level']:g}_seed{seed}"
            cell_path = cell_dir / f"{cell_name}.json"
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
                exclusion_variant=args.exclusion_variant,
                random_baseline=args.random_baseline,
            ))
            result = _run_condition(
                policy, device, area, seed, args.episodes,
                degradation_mode=cell["mode"],
                dropout_rate=cell["dropout_rate"],
                faith_evaluator=evaluator,
                faith_every=args.faithfulness_every,
                aoi_unaware=aoi_unaware,
                keep_records=True,
                stochastic=not args.deterministic,
                outage_duration_s=cell["outage_s"],
                corruption=args.corruption,
                demand_files=demand_files or None,
            )
            result["cell"] = {**cell, "seed": seed}
            atomic_write_json(cell_path, result)
            f = result["faithfulness"]
            print(f"[{i:>3}/{n_total}] {cell_name}: "
                  f"pickups={result['mean_pickups']:.1f}  "
                  f"deg_rate={result['empirical_degradation_rate']:.3f}  "
                  f"DEF_m={f.get('def_m_mean', float('nan')):+.3f}  "
                  f"WAMSN={f.get('wamsn_mean', float('nan')):.3f}  "
                  f"drift={f.get('drift_mean', float('nan')):.4f}  "
                  f"({time.time() - t0:.0f}s)")

    print()
    print(f"done. analyse with: python scripts/analyze_hypotheses.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
