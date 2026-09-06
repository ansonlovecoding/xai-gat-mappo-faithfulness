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
        and all(arguments.get(key) == value for key, value in train_cfg.items()
                if key in {
                    "lr", "gamma", "gae_lambda", "clip_ratio", "ppo_epochs",
                    "minibatch_size", "vf_coef", "ent_coef", "max_grad_norm",
                    "entropy_floor", "max_ent_coef",
                    "value_clip_ratio", "target_kl",
                    "dispatch_credit_reward", "critic_encoder_gradient_scale",
                })
        and bool(arguments.get("deterministic_torch"))
            == bool(train_cfg.get("deterministic_torch"))
        and bool(arguments.get("anneal_lr")) == bool(train_cfg.get("anneal_lr"))
        and arguments.get("degradation") == model["degradation"]
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
    parser.add_argument(
        "--config", type=Path, required=True,
        help="fixed experiment protocol",
    )
    parser.add_argument("--stage", choices=["train", "select", "evaluate", "diagnose",
                                            "sweep", "robustness", "preflight",
                                            "analyze", "summarize", "controls",
                                            "analyze-controls", "analyze-robustness",
                                            "audit", "framework", "all"],
                        default="all")
    parser.add_argument("--model", action="append", default=[],
                        help="model id to run; repeatable; default is every model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip complete immutable stages; reject partial training runs")
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    if config.get("schema_version") not in (1, 2):
        parser.error("unsupported experiment-config schema")

    output_root = PROJECT_ROOT / config["output_root"]
    checkpoint_root = (
        PROJECT_ROOT / config["checkpoint_root"]
        if config.get("checkpoint_root") else output_root / "training"
    )
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    stability_cfg = config.get("stability")
    models = _selected_models(config, args.model)
    framework_stages = [
        "diagnose", "sweep", "robustness", "preflight", "analyze", "summarize",
        "controls", "analyze-robustness", "analyze-controls", "audit",
    ]
    stages = (
        ["train", "select", "evaluate", *framework_stages]
        if args.stage == "all"
        else framework_stages if args.stage == "framework"
        else [args.stage]
    )

    current_revision = git_state(PROJECT_ROOT)["revision"]
    if any(stage in stages for stage in (
        "train", "evaluate", "diagnose", "sweep", "robustness"
    )):
        _run([sys.executable, "scripts/check_environment.py"], dry_run=args.dry_run)

    for stage in stages:
        audit_cfg = config.get("explanation_audit", {})
        evidence_root = PROJECT_ROOT / audit_cfg.get(
            "evidence_root", "results/dissertation_v10_corrected"
        )
        if stage == "summarize":
            command = [
                sys.executable, "scripts/summarize_dissertation_experiment.py",
                str(output_root),
            ]
            if config.get("performance_root"):
                command += [
                    "--performance-root",
                    str(PROJECT_ROOT / config["performance_root"]),
                ]
            _run(command, dry_run=args.dry_run)
            continue
        if stage == "controls":
            controls_cfg = config.get("faithfulness_controls", {})
            controls_root = output_root / "faithfulness_controls"
            for model in models:
                if not model.get("faithfulness_sweep"):
                    continue
                slug = model.get("audit_slug", model["id"])
                common = [
                    sys.executable, "scripts/run_faithfulness_controls.py",
                    "--models", model["id"],
                    "--training-seeds", *(str(seed) for seed in train_cfg["seeds"]),
                    "--eval-seeds", *(str(seed) for seed in controls_cfg.get(
                        "evaluation_seeds", eval_cfg["seeds"]
                    )),
                    "--episodes", str(controls_cfg.get("episodes", 3)),
                    "--conditions", *controls_cfg.get(
                        "conditions", ["clean", "outage_60s"]
                    ),
                    "--checkpoint-root", str(checkpoint_root),
                ]
                _run([
                    *common,
                    "--faithfulness-every", str(controls_cfg.get("faithfulness_every", 8)),
                    "--out", str(controls_root / f"{slug}_full"),
                ], dry_run=args.dry_run)
                _run([
                    *common,
                    "--action-row-only",
                    "--out", str(controls_root / f"{slug}_action_row"),
                ], dry_run=args.dry_run)
            continue
        if stage == "analyze-robustness":
            _run([
                sys.executable, "scripts/analyze_random_loss_robustness.py",
                str(output_root / "robustness" / "random_loss_30s"),
                "--out", str(evidence_root / "random_loss_robustness"),
            ], dry_run=args.dry_run)
            continue
        if stage == "analyze-controls":
            controls_root = output_root / "faithfulness_controls"
            inputs = []
            for model in models:
                if not model.get("faithfulness_sweep"):
                    continue
                slug = model.get("audit_slug", model["id"])
                inputs += [
                    str(controls_root / f"{slug}_full"),
                    str(controls_root / f"{slug}_action_row"),
                ]
            controls_cfg = config.get("faithfulness_controls", {})
            _run([
                sys.executable, "scripts/analyze_faithfulness_controls.py",
                *inputs,
                "--out", str(evidence_root / "faithfulness_controls"),
                "--fig-dir", str(PROJECT_ROOT / controls_cfg.get(
                    "figure_directory", "docs/figures"
                )),
                "--fig-prefix", controls_cfg.get("figure_prefix", "v10"),
            ], dry_run=args.dry_run)
            continue
        if stage == "audit":
            _run([
                sys.executable, "scripts/audit_explanations.py",
                "--config", str(args.config),
            ], dry_run=args.dry_run)
            continue
        for model in models:
            model_train_cfg = {
                **train_cfg,
                **model.get("training", {}),
            }
            for seed in train_cfg["seeds"]:
                training_run_dir = (
                    output_root / "training" / model["id"] / f"seed_{seed}"
                )
                run_dir = (
                    training_run_dir if stage in ("train", "select")
                    else checkpoint_root / model["id"] / f"seed_{seed}"
                )
                final_checkpoint = run_dir / "ckpt_final.pt"
                selection_cfg = config.get("selection")
                checkpoint = (run_dir / "ckpt_selected.pt"
                              if selection_cfg else final_checkpoint)

                if stage == "train":
                    if final_checkpoint.exists() and args.resume:
                        if _training_is_compatible(
                            run_dir / "manifest.json", model, seed, model_train_cfg,
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
                        "--epochs", str(model_train_cfg["epochs"]),
                        "--seed", str(seed),
                        "--demand-split", model_train_cfg["demand_split"],
                        "--save-every", str(model_train_cfg["save_every"]),
                        "--best-window", str(model_train_cfg["best_window"]),
                        "--degradation", model["degradation"],
                        "--run-dir", str(run_dir),
                    ]
                    if model_train_cfg.get("deterministic_torch"):
                        command.append("--deterministic-torch")
                    if model_train_cfg.get("anneal_lr"):
                        command.append("--anneal-lr")
                    elif "anneal_lr" in model_train_cfg:
                        command.append("--no-anneal-lr")
                    option_names = {
                        "lr": "--lr",
                        "gamma": "--gamma",
                        "gae_lambda": "--gae-lambda",
                        "clip_ratio": "--clip-ratio",
                        "ppo_epochs": "--ppo-epochs",
                        "minibatch_size": "--minibatch-size",
                        "vf_coef": "--vf-coef",
                        "ent_coef": "--ent-coef",
                        "entropy_floor": "--entropy-floor",
                        "max_ent_coef": "--max-ent-coef",
                        "max_grad_norm": "--max-grad-norm",
                        "value_clip_ratio": "--value-clip-ratio",
                        "target_kl": "--target-kl",
                        "dispatch_credit_reward": "--dispatch-credit-reward",
                        "critic_encoder_gradient_scale": "--critic-encoder-gradient-scale",
                    }
                    for key, option in option_names.items():
                        if key in model_train_cfg:
                            command += [option, str(model_train_cfg[key])]
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
                    ]
                    if model["degradation"] != "off":
                        command += ["--degradation", model["degradation"]]
                        command += [
                            "--outage-duration",
                            str(model.get("outage_duration", 0.0)),
                        ]
                    if "minimum_improvement_over_initial" in selection_cfg:
                        command += [
                            "--minimum-improvement-over-initial",
                            str(selection_cfg["minimum_improvement_over_initial"]),
                        ]
                    if args.resume:
                        command.append("--resume")
                    _run(command, dry_run=args.dry_run)
                    if stability_cfg:
                        stability_command = [
                            sys.executable, "scripts/check_training_stability.py",
                            str(run_dir),
                        ]
                        option_names = {
                            "final_window": "--final-window",
                            "minimum_final_mean_pickups": "--minimum-final-mean-pickups",
                            "maximum_consecutive_zero_pickups": "--maximum-consecutive-zero-pickups",
                            "minimum_selected_mean_pickups": "--minimum-selected-mean-pickups",
                            "minimum_final_validation_retention":
                                "--minimum-final-validation-retention",
                        }
                        for key, option in option_names.items():
                            if key in stability_cfg:
                                stability_command += [option, str(stability_cfg[key])]
                        _run(stability_command, dry_run=args.dry_run)
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
                        if output.exists() and args.resume and not args.dry_run:
                            if _evaluation_is_compatible(
                                output, checkpoint, evaluation_seed,
                                eval_cfg, current_revision
                            ):
                                print(f"SKIP complete evaluation: {output}")
                                continue
                            raise SystemExit(f"existing evaluation is incompatible: {output}")
                        if output.exists() and not args.dry_run:
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

                if stage == "diagnose":
                    diagnostic_cfg = config.get("deterministic_diagnostic")
                    if not diagnostic_cfg or not model.get("faithfulness_sweep"):
                        continue
                    output = (output_root / "deterministic_diagnostics" / model["id"]
                              / f"seed_{seed}.json")
                    if output.exists() and args.resume and not args.dry_run:
                        expected = {
                            **eval_cfg,
                            "episodes_per_cell_seed": diagnostic_cfg["episodes"],
                            "demand_split": diagnostic_cfg["demand_split"],
                            "stochastic": False,
                        }
                        if _evaluation_is_compatible(
                            output, checkpoint, diagnostic_cfg["seed"],
                            expected, current_revision
                        ):
                            print(f"SKIP complete deterministic diagnostic: {output}")
                            continue
                        raise SystemExit(
                            f"existing deterministic diagnostic is incompatible: {output}"
                        )
                    if output.exists() and not args.dry_run:
                        raise SystemExit(
                            f"immutable deterministic diagnostic already exists: {output}; "
                            "use --resume or a new experiment root"
                        )
                    command = [
                        sys.executable, "scripts/eval_policy.py", str(checkpoint),
                        "--episodes", str(diagnostic_cfg["episodes"]),
                        "--seed", str(diagnostic_cfg["seed"]),
                        "--demand-split", diagnostic_cfg["demand_split"],
                        "--output", str(output),
                    ]
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
                    option_names = {
                        "faithfulness_exposed_every":
                            "--faithfulness-exposed-every",
                        "minimum_stale_exposed_records":
                            "--minimum-stale-exposed-records",
                        "minimum_stale_exposed_episodes":
                            "--minimum-stale-exposed-episodes",
                    }
                    for key, option in option_names.items():
                        if key in eval_cfg:
                            command += [option, str(eval_cfg[key])]
                    if not eval_cfg.get("exclusion_variant", True):
                        command.append("--no-exclusion-variant")
                    if not eval_cfg.get("stochastic", True):
                        command.append("--deterministic")
                    _run(command, dry_run=args.dry_run)
                elif stage == "robustness":
                    robustness_cfg = config.get("random_loss_robustness")
                    if not robustness_cfg:
                        continue
                    robustness_dir = (
                        output_root / "robustness" / "random_loss_30s"
                        / model["id"] / f"seed_{seed}"
                    )
                    command = [
                        sys.executable, "scripts/sweep_severity.py", str(checkpoint),
                        "--episodes", str(robustness_cfg["episodes_per_cell_seed"]),
                        "--seeds", *(str(value) for value in robustness_cfg["seeds"]),
                        "--outage-durations", str(robustness_cfg["outage_duration_s"]),
                        "--with-dropout-axis",
                        "--dropout-rate", str(robustness_cfg["dropout_rate"]),
                        "--demand-split", robustness_cfg["demand_split"],
                        "--corruption", eval_cfg["corruption"],
                        "--faithfulness-every", str(eval_cfg["faithfulness_every"]),
                        "--faithfulness-random-baselines",
                        str(eval_cfg["random_baselines"]),
                        "--random-baseline", eval_cfg["random_baseline"],
                        "--out", str(robustness_dir),
                    ]
                    if "faithfulness_exposed_every" in eval_cfg:
                        command += [
                            "--faithfulness-exposed-every",
                            str(eval_cfg["faithfulness_exposed_every"]),
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
