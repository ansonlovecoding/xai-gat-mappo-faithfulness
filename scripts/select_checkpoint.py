"""Select a trained checkpoint using validation demand only."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.checkpoint_selection import select_best_candidate  # noqa: E402
from dispatch_marl.provenance import atomic_write_json, sha256_file  # noqa: E402


def _checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    return int(payload["epoch"])


def _candidate_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("ckpt_epoch_*.pt"))
    best = run_dir / "ckpt_best.pt"
    if best.exists():
        paths.append(best)

    # ckpt_best can coincide with a periodic snapshot. Avoid evaluating the
    # same model weights twice while retaining the periodic filename.
    unique: dict[str, Path] = {}
    for path in paths:
        digest = sha256_file(path)
        if digest not in unique or path.name.startswith("ckpt_epoch_"):
            unique[digest] = path
    return sorted(unique.values(), key=lambda path: (_checkpoint_epoch(path), path.name))


def _compatible_result(path: Path, checkpoint: Path, args: argparse.Namespace) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("checkpoint_sha256") == sha256_file(checkpoint)
        and result.get("episodes") == args.episodes
        and result.get("seed") == args.seed
        and result.get("demand_split") == args.demand_split
        and result.get("stochastic") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--demand-split", choices=["val"], default="val")
    parser.add_argument("--minimum-mean-pickups", type=float, default=1.0)
    parser.add_argument("--minimum-improvement-over-initial", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    candidates = _candidate_paths(run_dir)
    if not candidates:
        parser.error(f"no checkpoint candidates found in {run_dir}")

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    scored: list[dict] = []
    for checkpoint in candidates:
        result_path = validation_dir / f"{checkpoint.stem}.json"
        if not (args.resume and _compatible_result(result_path, checkpoint, args)):
            if result_path.exists():
                parser.error(
                    f"incompatible validation result exists: {result_path}; "
                    "use a new experiment root"
                )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval_policy.py"),
                str(checkpoint),
                "--episodes", str(args.episodes),
                "--seed", str(args.seed),
                "--demand-split", args.demand_split,
                "--output", str(result_path),
                "--stochastic",
            ]
            print("$", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        result = json.loads(result_path.read_text())
        scored.append({
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "epoch": int(result["epoch"]),
            "mean_pickups": float(result["mean_pickups"]),
            "mean_reward": float(result["mean_reward"]),
            "validation_result": str(result_path.relative_to(run_dir)),
        })

    selected = select_best_candidate(scored)
    initial = min(scored, key=lambda item: int(item["epoch"]))
    improvement = selected["mean_pickups"] - initial["mean_pickups"]
    if selected["mean_pickups"] < args.minimum_mean_pickups:
        raise SystemExit(
            "checkpoint selection failed: best validation mean was "
            f"{selected['mean_pickups']:.2f} pickups, below the declared minimum "
            f"of {args.minimum_mean_pickups:.2f}"
        )
    if improvement < args.minimum_improvement_over_initial:
        raise SystemExit(
            "checkpoint selection failed: best validation mean improved by "
            f"{improvement:.2f} pickups over epoch {initial['epoch']}, below the "
            f"declared minimum of {args.minimum_improvement_over_initial:.2f}"
        )

    source = run_dir / selected["checkpoint"]
    destination = run_dir / "ckpt_selected.pt"
    shutil.copy2(source, destination)
    report = {
        "selection_split": args.demand_split,
        "selection_seed": args.seed,
        "episodes": args.episodes,
        "stochastic": True,
        "metric": "mean_pickups",
        "tie_breakers": ["mean_reward", "earlier_epoch"],
        "minimum_mean_pickups": args.minimum_mean_pickups,
        "minimum_improvement_over_initial": args.minimum_improvement_over_initial,
        "initial": initial,
        "selected_improvement_over_initial": improvement,
        "selected": {
            **selected,
            "copied_to": destination.name,
            "copied_checkpoint_sha256": sha256_file(destination),
        },
        "candidates": scored,
    }
    atomic_write_json(run_dir / "checkpoint_selection.json", report)
    print(
        f"selected epoch {selected['epoch']} with validation mean "
        f"{selected['mean_pickups']:.2f} pickups: {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
