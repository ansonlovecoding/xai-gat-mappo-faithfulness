"""Audit dissertation-v4 provenance and the checkpoint table cited in Chapter 3."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.provenance import atomic_write_json, sha256_file  # noqa: E402


DISPLAY_IDS = {
    "B1_mlp": "MLP",
    "B2_gat": "GAT",
    "B3_gat_noaoi": "GAT-NoAoI",
    "H5_gat_degraded": "GAT-Outage",
}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise RuntimeError(f"missing file: {path.relative_to(PROJECT_ROOT)}") from None


def chapter_checkpoint_rows(path: Path) -> dict[str, list[int]]:
    rows = {}
    pattern = re.compile(
        r"^\|\s*(MLP|GAT|GAT-NoAoI|GAT-Outage)\s*\|"
        r"\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|$"
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group(1)] = [int(match.group(i)) for i in range(2, 5)]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/dissertation_v4.toml",
    )
    parser.add_argument(
        "--chapter", type=Path,
        default=PROJECT_ROOT / "docs/dissertation/03_methodology.md",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "runs/dissertation_v4/provenance_audit.json",
    )
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text())
    root = PROJECT_ROOT / config["output_root"]
    training = config["training"]
    evaluation = config["evaluation"]
    models = config["models"]
    chapter_rows = chapter_checkpoint_rows(args.chapter)

    errors: list[str] = []
    checks: list[dict] = []
    revisions: set[str] = set()
    selected_epochs: dict[str, dict[str, int]] = {}

    def check(ok: bool, name: str, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    for model in models:
        model_id = model["id"]
        selected_epochs[model_id] = {}
        for training_seed in training["seeds"]:
            run_dir = root / "training" / model_id / f"seed_{training_seed}"
            manifest = load_json(run_dir / "manifest.json")
            selection = load_json(run_dir / "checkpoint_selection.json")
            checkpoint = run_dir / "ckpt_selected.pt"
            selected = selection["selected"]
            checkpoint_hash = sha256_file(checkpoint)
            revision = manifest.get("code_revision", "unknown")
            revisions.add(revision)

            selected_epochs[model_id][str(training_seed)] = int(selected["epoch"])
            check(
                checkpoint_hash == selected["copied_checkpoint_sha256"],
                f"selected checkpoint hash {model_id} seed {training_seed}",
                f"actual={checkpoint_hash} metadata={selected['copied_checkpoint_sha256']}",
            )
            arguments = manifest.get("arguments", {})
            check(
                arguments.get("seed") == training_seed
                and arguments.get("policy") == model["policy"]
                and arguments.get("demand_split") == training["demand_split"]
                and arguments.get("epochs") == training["epochs"],
                f"training manifest {model_id} seed {training_seed}",
                "training arguments differ from dissertation_v4.toml",
            )

            for evaluation_seed in evaluation["seeds"]:
                result_path = (
                    root / "evaluations" / model_id / f"seed_{training_seed}"
                    / f"eval_seed_{evaluation_seed}.json"
                )
                result = load_json(result_path)
                valid = (
                    result.get("checkpoint_sha256") == checkpoint_hash
                    and result.get("seed") == evaluation_seed
                    and result.get("episodes") == evaluation["episodes_per_cell_seed"]
                    and result.get("demand_split") == evaluation["demand_split"]
                    and result.get("stochastic") == evaluation["stochastic"]
                    and len(result.get("per_episode", []))
                        == evaluation["episodes_per_cell_seed"]
                    and result.get("provenance", {}).get("git", {}).get("revision")
                        == revision
                )
                check(
                    valid,
                    f"evaluation {model_id} train {training_seed} eval {evaluation_seed}",
                    "evaluation does not match selected checkpoint or protocol",
                )

            if model.get("faithfulness_sweep"):
                sweep_dir = root / "sweeps" / model_id / f"seed_{training_seed}"
                sweep_manifest = load_json(sweep_dir / "manifest.json")
                preflight = load_json(sweep_dir / "preflight.json")
                sweep_valid = (
                    sweep_manifest.get("checkpoint_sha256") == checkpoint_hash
                    and sweep_manifest.get("demand_split") == evaluation["demand_split"]
                    and sweep_manifest.get("seeds") == evaluation["seeds"]
                    and sweep_manifest.get("episodes_per_cell_seed")
                        == evaluation["episodes_per_cell_seed"]
                    and sweep_manifest.get("stochastic") == evaluation["stochastic"]
                    and sweep_manifest.get("provenance", {}).get("git", {}).get("revision")
                        == revision
                    and preflight.get("ok") is True
                    and preflight.get("summary", {}).get("planned_cells")
                        == preflight.get("summary", {}).get("completed_cells")
                )
                check(
                    sweep_valid,
                    f"faithfulness sweep {model_id} seed {training_seed}",
                    "sweep provenance or preflight check failed",
                )

    check(
        len(revisions) == 1,
        "single training code revision",
        f"revisions={sorted(revisions)}",
    )

    for model_id, epochs_by_seed in selected_epochs.items():
        chapter_id = DISPLAY_IDS[model_id]
        expected = [epochs_by_seed[str(seed)] for seed in training["seeds"]]
        actual = chapter_rows.get(chapter_id)
        check(
            actual == expected,
            f"Chapter 3 checkpoint row {chapter_id}",
            f"chapter={actual} selected={expected}",
        )

    payload = {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "summary": {
            "checks": len(checks),
            "passed": sum(item["ok"] for item in checks),
            "failed": len(errors),
            "training_runs": len(models) * len(training["seeds"]),
            "clean_evaluations": (
                len(models) * len(training["seeds"]) * len(evaluation["seeds"])
            ),
            "faithfulness_sweeps": (
                sum(bool(model.get("faithfulness_sweep")) for model in models)
                * len(training["seeds"])
            ),
            "training_code_revisions": sorted(revisions),
        },
        "selected_epochs": selected_epochs,
        "checks": checks,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"audit: OK ({args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
