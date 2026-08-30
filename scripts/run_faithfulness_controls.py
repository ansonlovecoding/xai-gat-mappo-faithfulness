"""Run perturbation positive controls for the dissertation faithfulness audit.

For selected B2/D30 checkpoints, this runner compares coupled GAT attention
with a leave-one-out (LOO) ranking under clean and 60 s tunnel-triggered
telemetry conditions. The LOO ranking protects the chosen request node, so
the control measures information removal rather than action deletion.

The experiment records:
  * exclusion-variant attention DEF_margin;
  * LOO perturbation-control DEF_margin and its gap over attention;
  * attention/LOO top-k agreement;
  * exact expected overlap with a type-matched random top-k;
  * taxi-only Spearman correlation between attention and LOO effect.

Use ``--smoke`` first. Cell files make a full run resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_degradation_ablation import _run_condition  # noqa: E402

from dispatch_marl import FaithfulnessConfig, FaithfulnessEvaluator  # noqa: E402
from dispatch_marl.provenance import (  # noqa: E402
    atomic_write_json,
    runtime_provenance,
    sha256_file,
)
from dispatch_marl.runtime import choose_device, load_policy  # noqa: E402
from dispatch_marl.scenario import demand_split_files  # noqa: E402


MODEL_IDS = ("B2_gat", "H5_gat_degraded")


def _condition_spec(name: str) -> dict:
    if name == "clean":
        return {"mode": "off", "outage_s": 0.0}
    if name == "outage_60s":
        return {"mode": "tunnel_triggered", "outage_s": 60.0}
    raise ValueError(f"unknown condition: {name}")


def _summarise_cells(cells: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for cell in cells:
        key = f"{cell['model_id']}:{cell['condition_name']}"
        grouped.setdefault(key, []).append(cell)

    summary: dict[str, dict] = {}
    for key, group in grouped.items():
        faith = [cell["faithfulness"] for cell in group]

        def _mean(field: str) -> float | None:
            values = [f[field] for f in faith if f.get(field) is not None]
            return float(np.mean(values)) if values else None

        summary[key] = {
            "n_cells": len(group),
            "n_positive_control_decisions": int(sum(
                f.get("n_positive_control_decisions", 0) for f in faith
            )),
            "attention_control_def_m_mean_of_cells": _mean(
                "attention_control_def_m_mean"
            ),
            "attention_control_g_comp_m_mean_of_cells": _mean(
                "attention_control_g_comp_m_mean"
            ),
            "attention_control_g_suff_m_mean_of_cells": _mean(
                "attention_control_g_suff_m_mean"
            ),
            "loo_oracle_def_m_mean_of_cells": _mean("loo_oracle_def_m_mean"),
            "loo_oracle_g_comp_m_mean_of_cells": _mean(
                "loo_oracle_g_comp_m_mean"
            ),
            "loo_oracle_g_suff_m_mean_of_cells": _mean(
                "loo_oracle_g_suff_m_mean"
            ),
            "grad_x_input_def_m_mean_of_cells": _mean(
                "grad_x_input_def_m_mean"
            ),
            "grad_x_input_g_comp_m_mean_of_cells": _mean(
                "grad_x_input_g_comp_m_mean"
            ),
            "grad_x_input_g_suff_m_mean_of_cells": _mean(
                "grad_x_input_g_suff_m_mean"
            ),
            "loo_control_gap_m_mean_of_cells": _mean("loo_control_gap_m_mean"),
            "attention_loo_topk_agreement_mean_of_cells": _mean(
                "attention_loo_topk_agreement_mean"
            ),
            "expected_random_topk_overlap_mean_of_cells": _mean(
                "expected_random_topk_overlap_mean"
            ),
            "taxi_attention_loo_spearman_mean_of_cells": _mean(
                "taxi_attention_loo_spearman_mean"
            ),
            "n_action_row_request_decisions": int(sum(
                f.get("n_action_row_request_decisions", 0) for f in faith
            )),
            "self_row_request_def_m_mean_of_cells": _mean(
                "self_row_request_def_m_mean"
            ),
            "action_row_request_def_m_mean_of_cells": _mean(
                "action_row_request_def_m_mean"
            ),
            "action_row_request_gap_m_mean_of_cells": _mean(
                "action_row_request_gap_m_mean"
            ),
        }
        sensitivity_names = sorted({
            name for f in faith for name in f.get("aggregation_sensitivity", {})
        })
        if sensitivity_names:
            summary[key]["aggregation_sensitivity"] = {}
            for name in sensitivity_names:
                entries = [
                    f["aggregation_sensitivity"][name]
                    for f in faith if name in f.get("aggregation_sensitivity", {})
                ]
                n = sum(e["n_stale_exposed_decisions"] for e in entries)
                weighted = sum(
                    e["mean_stale_attention_shift"]
                    * e["n_stale_exposed_decisions"]
                    for e in entries
                )
                summary[key]["aggregation_sensitivity"][name] = {
                    "mean_stale_attention_shift": weighted / n if n else None,
                    "n_stale_exposed_decisions": n,
                }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS,
                        default=list(MODEL_IDS))
    parser.add_argument("--training-seeds", type=int, nargs="+",
                        default=[42, 43, 44])
    parser.add_argument("--eval-seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48, 49])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--conditions", nargs="+",
                        choices=["clean", "outage_60s"],
                        default=["clean", "outage_60s"])
    parser.add_argument("--faithfulness-every", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="B2 seed 42, eval seed 42, one episode")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--action-row-only", action="store_true",
        help="score request-action decisions only, comparing the fixed self "
             "row with the selected request's attention row",
    )
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "runs" / "dissertation_revision_v1"
                        / "faithfulness_controls")
    args = parser.parse_args()

    if args.smoke:
        args.models = ["B2_gat"]
        args.training_seeds = [42]
        args.eval_seeds = [42]
        args.episodes = 1
        args.out = args.out / "smoke"
    if args.action_row_only:
        args.faithfulness_every = 1

    device = args.device or choose_device()
    args.out.mkdir(parents=True, exist_ok=True)
    cell_dir = args.out / "cells"
    cell_dir.mkdir(exist_ok=True)
    demand_files = demand_split_files("central_park", "test")
    runtime = runtime_provenance(PROJECT_ROOT)
    n_cells = (len(args.models) * len(args.training_seeds)
               * len(args.eval_seeds) * len(args.conditions))
    completed = 0

    for model_id in args.models:
        for training_seed in args.training_seeds:
            checkpoint = (PROJECT_ROOT / "runs" / "dissertation_v4" / "training"
                          / model_id / f"seed_{training_seed}" / "ckpt_selected.pt")
            if not checkpoint.exists():
                parser.error(f"checkpoint not found: {checkpoint}")
            policy, ckpt = load_policy(checkpoint, device)
            for condition_name in args.conditions:
                condition = _condition_spec(condition_name)
                for eval_seed in args.eval_seeds:
                    completed += 1
                    cell_name = (
                        f"{model_id}_train{training_seed}_{condition_name}"
                        f"_eval{eval_seed}.json"
                    )
                    cell_path = cell_dir / cell_name
                    if cell_path.exists() and not args.overwrite:
                        print(f"[{completed}/{n_cells}] {cell_name}: exists")
                        continue
                    print(f"[{completed}/{n_cells}] {cell_name}")
                    evaluator = FaithfulnessEvaluator(policy, FaithfulnessConfig(
                        top_k_values=(1, 2, 3),
                        n_random_baselines=5,
                        seed=eval_seed,
                        exclusion_variant=True,
                        random_baseline="type_matched",
                    ))
                    started = time.time()
                    result = _run_condition(
                        policy=policy,
                        device=device,
                        area="central_park",
                        seed=eval_seed,
                        episodes=args.episodes,
                        degradation_mode=condition["mode"],
                        dropout_rate=0.0,
                        faith_evaluator=evaluator,
                        faith_every=args.faithfulness_every,
                        keep_records=True,
                        stochastic=True,
                        outage_duration_s=condition["outage_s"],
                        corruption="freeze",
                        demand_files=demand_files,
                        faithfulness_positive_control=not args.action_row_only,
                        attention_aggregation_sensitivity=not args.action_row_only,
                        attention_action_row_sensitivity=args.action_row_only,
                        faithfulness_request_actions_only=args.action_row_only,
                    )
                    result.update({
                        "model_id": model_id,
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "condition_name": condition_name,
                        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "faithfulness_every": args.faithfulness_every,
                        "wall_s": round(time.time() - started, 1),
                    })
                    atomic_write_json(cell_path, result)
                    faith = result["faithfulness"]
                    if args.action_row_only:
                        print(
                            "  self={:+.3f} action_row={:+.3f} gap={:+.3f} n={}".format(
                                faith.get("self_row_request_def_m_mean", float("nan")),
                                faith.get("action_row_request_def_m_mean", float("nan")),
                                faith.get("action_row_request_gap_m_mean", float("nan")),
                                faith.get("n_action_row_request_decisions", 0),
                            )
                        )
                    else:
                        print(
                            "  attention={:+.3f} loo={:+.3f} gap={:+.3f} n={}".format(
                                faith.get("attention_control_def_m_mean", float("nan")),
                                faith.get("loo_oracle_def_m_mean", float("nan")),
                                faith.get("loo_control_gap_m_mean", float("nan")),
                                faith.get("n_positive_control_decisions", 0),
                            )
                        )

    cells = [json.loads(path.read_text()) for path in sorted(cell_dir.glob("*.json"))]
    payload = {
        "protocol": {
            "models": args.models,
            "training_seeds": args.training_seeds,
            "eval_seeds": args.eval_seeds,
            "episodes": args.episodes,
            "conditions": args.conditions,
            "demand_split": "test",
            "demand_files": demand_files,
            "stochastic": True,
            "faithfulness_every": args.faithfulness_every,
            "chosen_request_protected": True,
            "random_baseline": "type_matched",
            "action_row_only": args.action_row_only,
        },
        "runtime": runtime,
        "summary": _summarise_cells(cells),
        "cell_files": [path.name for path in sorted(cell_dir.glob("*.json"))],
    }
    atomic_write_json(args.out / "summary.json", payload)
    print(f"summary: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
