"""Run the versioned dissertation training and faithfulness protocol."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.provenance import git_state, sha256_file  # noqa: E402


def _run(command: list[str], *, dry_run: bool) -> None:
    print("$", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _selected_models(config: dict, requested: list[str]) -> list[dict]:
    models = config["models"]
    known = {model["id"] for model in models}
    unknown = set(requested) - known
    if unknown:
        raise SystemExit(f"unknown model ids: {sorted(unknown)}; expected {sorted(known)}")
    return [model for model in models if not requested or model["id"] in requested]


def _training_is_compatible(manifest_path: Path, model: dict, seed: int,
                            train_cfg: dict, area: str, revision: str) -> bool:
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    arguments = manifest.get("arguments", {})
    return (
        manifest.get("code_revision") == revision
        and arguments.get("seed") == seed
        and arguments.get("policy") == model["policy"]
        and arguments.get("area") == area
        and arguments.get("epochs") == train_cfg["epochs"]
        and arguments.get("demand_split") == train_cfg["demand_split"]
        and arguments.get("save_every") == train_cfg["save_every"]
        and arguments.get("best_window") == train_cfg["best_window"]
        and bool(arguments.get("deterministic_torch"))
            == bool(train_cfg.get("deterministic_torch"))
        and arguments.get("degradation") == model["degradation"]
        and bool(arguments.get("aoi_unaware")) == bool(model.get("aoi_unaware"))
        and float(arguments.get("outage_duration", 0.0))
            == float(model.get("outage_duration", 0.0))
    )


def _evaluation_is_compatible(path: Path, checkpoint: Path, evaluation_seed: int,
                              eval_cfg: dict, revision: str) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("checkpoint_sha256") == sha256_file(checkpoint)
        and payload.get("seed") == evaluation_seed
        and payload.get("episodes") == eval_cfg["episodes_per_cell_seed"]
        and payload.get("demand_split") == eval_cfg["demand_split"]
        and payload.get("stochastic") == eval_cfg.get("stochastic", True)
        and payload.get("provenance", {}).get("git", {}).get("revision") == revision
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/experiments/dissertation_v4.toml")
    parser.add_argument("--stage", choices=["train", "select", "evaluate", "sweep", "preflight",
                                            "analyze", "summarize", "all"],
                        default="all")
    parser.add_argument("--model", action="append", default=[],
                        help="model id to run; repeatable; default is every model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip complete immutable stages; reject partial training runs")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    if config.get("schema_version") != 1:
        parser.error("unsupported experiment-config schema")

    output_root = PROJECT_ROOT / config["output_root"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    models = _selected_models(config, args.model)
    stages = (["train", "select", "evaluate", "sweep", "preflight", "analyze", "summarize"]
              if args.stage == "all" else [args.stage])

    current_revision = git_state(PROJECT_ROOT)["revision"]
    if any(stage in stages for stage in ("train", "evaluate", "sweep")):
        _run([sys.executable, "scripts/check_environment.py"], dry_run=args.dry_run)

    for stage in stages:
        if stage == "summarize":
            _run([sys.executable, "scripts/summarize_dissertation_experiment.py",
                  str(output_root)], dry_run=args.dry_run)
            continue
        for model in models:
            for seed in train_cfg["seeds"]:
                run_dir = output_root / "training" / model["id"] / f"seed_{seed}"
                final_checkpoint = run_dir / "ckpt_final.pt"
                selection_cfg = config.get("selection")
                checkpoint = (run_dir / "ckpt_selected.pt"
                              if selection_cfg else final_checkpoint)

                if stage == "train":
                    if final_checkpoint.exists() and args.resume:
                        if _training_is_compatible(
                            run_dir / "manifest.json", model, seed, train_cfg,
                            config["area"], current_revision
                        ):
                            print(f"SKIP complete training: {final_checkpoint}")
                            continue
                        raise SystemExit(f"existing training run is incompatible: {run_dir}")
                    if run_dir.exists() and any(run_dir.iterdir()) and args.resume:
                        raise SystemExit(f"partial training run requires a new directory: {run_dir}")
                    command = [
                        sys.executable, "scripts/train.py",
                        "--area", config["area"],
                        "--policy", model["policy"],
                        "--epochs", str(train_cfg["epochs"]),
                        "--seed", str(seed),
                        "--demand-split", train_cfg["demand_split"],
                        "--save-every", str(train_cfg["save_every"]),
                        "--best-window", str(train_cfg["best_window"]),
                        "--degradation", model["degradation"],
                        "--run-dir", str(run_dir),
                    ]
                    if train_cfg.get("deterministic_torch"):
                        command.append("--deterministic-torch")
                    if model.get("aoi_unaware"):
                        command.append("--aoi-unaware")
                    if "outage_duration" in model:
                        command += ["--outage-duration", str(model["outage_duration"])]
                    _run(command, dry_run=args.dry_run)
                    continue

                if stage == "select":
                    if not selection_cfg:
                        continue
                    if not final_checkpoint.exists() and not args.dry_run:
                        raise SystemExit(f"missing completed training run: {final_checkpoint}")
                    command = [
                        sys.executable, "scripts/select_checkpoint.py", str(run_dir),
                        "--episodes", str(selection_cfg["episodes"]),
                        "--seed", str(selection_cfg["seed"]),
                        "--demand-split", selection_cfg["demand_split"],
                        "--minimum-mean-pickups",
                        str(selection_cfg["minimum_mean_pickups"]),
                        "--minimum-improvement-over-initial",
                        str(selection_cfg["minimum_improvement_over_initial"]),
                    ]
                    if args.resume:
                        command.append("--resume")
                    _run(command, dry_run=args.dry_run)
                    continue

                if not model.get("faithfulness_sweep"):
                    if stage != "evaluate":
                        continue
                if not checkpoint.exists() and not args.dry_run:
                    raise SystemExit(f"missing checkpoint; run training first: {checkpoint}")

                if stage == "evaluate":
                    for evaluation_seed in eval_cfg["seeds"]:
                        output = (output_root / "evaluations" / model["id"]
                                  / f"seed_{seed}" / f"eval_seed_{evaluation_seed}.json")
                        if output.exists() and args.resume:
                            if _evaluation_is_compatible(
                                output, checkpoint, evaluation_seed,
                                eval_cfg, current_revision
                            ):
                                print(f"SKIP complete evaluation: {output}")
                                continue
                            raise SystemExit(f"existing evaluation is incompatible: {output}")
                        if output.exists():
                            raise SystemExit(
                                f"immutable evaluation output already exists: {output}; "
                                "use --resume or a new experiment root"
                            )
                        command = [
                            sys.executable, "scripts/eval_policy.py", str(checkpoint),
                            "--episodes", str(eval_cfg["episodes_per_cell_seed"]),
                            "--seed", str(evaluation_seed),
                            "--demand-split", eval_cfg["demand_split"],
                            "--output", str(output),
                        ]
                        if eval_cfg.get("stochastic", True):
                            command.append("--stochastic")
                        _run(command, dry_run=args.dry_run)
                    continue

                sweep_dir = output_root / "sweeps" / model["id"] / f"seed_{seed}"
                if stage == "sweep":
                    command = [
                        sys.executable, "scripts/sweep_severity.py", str(checkpoint),
                        "--episodes", str(eval_cfg["episodes_per_cell_seed"]),
                        "--seeds", *(str(value) for value in eval_cfg["seeds"]),
                        "--outage-durations",
                        *(str(value) for value in eval_cfg.get(
                            "outage_durations_s", eval_cfg.get("aoi_levels", [])
                        )),
                        "--demand-split", eval_cfg["demand_split"],
                        "--corruption", eval_cfg["corruption"],
                        "--faithfulness-every", str(eval_cfg["faithfulness_every"]),
                        "--faithfulness-random-baselines", str(eval_cfg["random_baselines"]),
                        "--random-baseline", eval_cfg["random_baseline"],
                        "--out", str(sweep_dir),
                    ]
                    if not eval_cfg.get("exclusion_variant", True):
                        command.append("--no-exclusion-variant")
                    if not eval_cfg.get("stochastic", True):
                        command.append("--deterministic")
                    _run(command, dry_run=args.dry_run)
                elif stage == "preflight":
                    _run([sys.executable, "scripts/preflight_experiment.py", str(sweep_dir)],
                         dry_run=args.dry_run)
                elif stage == "analyze":
                    _run([sys.executable, "scripts/analyze_hypotheses.py", str(sweep_dir)],
                         dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
