"""Package frozen checkpoints and raw audit cells for thesis reproduction."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_cells(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed) as archive:
                for path in sorted(source.glob("*.json")):
                    info = archive.gettarinfo(str(path), arcname=f"cells/{path.name}")
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def copy_evidence(source: Path, destination: Path, *, root: Path,
                  kind: str, entries: list[dict], **metadata: object) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required release evidence is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    entries.append({
        "kind": kind,
        **metadata,
        "path": str(destination.relative_to(root)),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/experiments/dissertation_v12_full_rerun.toml",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    run_root = args.run_root or ROOT / config["output_root"]
    audit_cfg = config.get("explanation_audit", {})
    evidence_root = ROOT / audit_cfg.get(
        "evidence_root", "results/dissertation_v10_corrected"
    )
    out = args.out or evidence_root / "release"
    checkpoint_root = (
        ROOT / config["checkpoint_root"]
        if config.get("checkpoint_root")
        else run_root / "training"
    )
    models = config["models"]
    entries: list[dict] = []

    for model in models:
        for seed in config["training"]["seeds"]:
            label = f"{model['id']}_seed_{seed}"
            source_dir = checkpoint_root / model["id"] / f"seed_{seed}"
            checkpoint_out = out / "checkpoints" / label
            checkpoint_out.mkdir(parents=True, exist_ok=True)
            for name in ("ckpt_selected.pt", "checkpoint_selection.json",
                         "manifest.json", "train_log.jsonl", "training_stability.json", "args.json", "best_metadata.json",
                         "timing.json"):
                source = source_dir / name
                if name == "timing.json" and not source.exists():
                    continue  # Historical runs did not record elapsed training time.
                destination = checkpoint_out / name
                copy_evidence(
                    source, destination, root=out,
                    kind="checkpoint" if name.endswith(".pt") else "training_record",
                    entries=entries, model=model["id"], training_seed=seed,
                )

            for source in sorted(source_dir.glob("validation*/*.json")):
                copy_evidence(
                    source, checkpoint_out / source.relative_to(source_dir),
                    root=out, kind="validation_evaluation", entries=entries,
                    model=model["id"], training_seed=seed,
                )
            evaluation_dir = run_root / "evaluations" / model["id"] / f"seed_{seed}"
            for source in sorted(evaluation_dir.glob("*.json")):
                copy_evidence(
                    source, out / "performance_evidence" / label / source.name,
                    root=out, kind="performance_evaluation", entries=entries,
                    model=model["id"], training_seed=seed,
                )
            diagnostic = run_root / "deterministic_diagnostics" / model["id"] / f"seed_{seed}.json"
            if diagnostic.is_file():
                copy_evidence(
                    diagnostic, out / "deterministic_evidence" / f"{label}.json",
                    root=out, kind="deterministic_evaluation", entries=entries,
                    model=model["id"], training_seed=seed,
                )
            if not model.get("faithfulness_sweep"):
                continue
            sweep_dir = run_root / "sweeps" / model["id"] / f"seed_{seed}"
            cells = sweep_dir / "cells"
            archive = out / "audit_records" / f"{label}_cells.tar.gz"
            archive_cells(cells, archive)
            entries.append({
                "kind": "raw_audit_cells",
                "model": model["id"],
                "training_seed": seed,
                "source_files": len(list(cells.glob("*.json"))),
                "path": str(archive.relative_to(out)),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            })

            for name in ("manifest.json", "preflight.json", "analysis.json", "analysis_rounded4.json"):
                copy_evidence(
                    sweep_dir / name,
                    out / "sweep_evidence" / label / name,
                    root=out, kind=f"sweep_{Path(name).stem}", entries=entries,
                    model=model["id"], training_seed=seed,
                )

    for category, pattern in (("control_records", "faithfulness_controls/*/cells"),
                              ("robustness_records", "robustness/*/*/*/cells")):
        for cells in sorted(run_root.glob(pattern)):
            relative = cells.parent.relative_to(run_root)
            label = "__".join(relative.parts)
            archive = out / category / f"{label}_cells.tar.gz"
            archive_cells(cells, archive)
            entries.append({"kind": category, "source_files": len(list(cells.glob("*.json"))),
                            "path": str(archive.relative_to(out)), "bytes": archive.stat().st_size,
                            "sha256": sha256(archive)})
            for source in sorted(cells.parent.glob("*.json")):
                copy_evidence(source, out / category / label / source.name,
                              root=out, kind=category + "_metadata", entries=entries)

    copy_evidence(
        args.config.resolve(), out / "experiment_config.toml",
        root=out, kind="experiment_config", entries=entries,
    )
    for source_name in (
        "README.md",
        "docs/REPRODUCE_EXPERIMENTS.md",
        "docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md",
        "requirements.txt",
        "requirements-v12-lock.txt",
        "docs/FULL_RERUN_V12.md",
        "requirements-macos-intel.txt",
        "pyproject.toml",
    ):
        source = ROOT / source_name
        if source.is_file():
            copy_evidence(
                source,
                out / "reproduction" / source_name,
                root=out,
                kind="reproduction_document",
                entries=entries,
            )
    for name in (
        "summary.json", "summary.csv", "performance_context.json",
        "performance_context.csv", "training_seed_synthesis.json",
        "training_seed_synthesis.csv", "action_stratified.json",
        "action_stratified.csv", "deterministic_diagnostics.json",
        "deterministic_diagnostics.csv", "precision_sensitivity.json",
        "precision_sensitivity.csv",
    ):
        source = run_root / name
        if source.is_file():
            copy_evidence(
                source, out / "summaries" / name,
                root=out, kind="summary", entries=entries,
            )

    for source in sorted((run_root / "execution_records").rglob("*")):
        if source.is_file():
            copy_evidence(source, out / "execution_records" / source.relative_to(run_root / "execution_records"),
                          root=out, kind="execution_record", entries=entries)

    if evidence_root.is_dir():
        for source in sorted(evidence_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(evidence_root)
            if relative.parts and relative.parts[0] == "release":
                continue
            copy_evidence(
                source, out / "evidence" / relative,
                root=out, kind="analysis_evidence", entries=entries,
            )

    manifest = {
        "experiment": config["name"],
        "config": str(args.config.resolve().relative_to(ROOT)),
        "contents": entries,
    }
    manifest_path = out / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"release files: {len(entries)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
