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
        default=ROOT / "configs/experiments/dissertation_v10_corrected.toml",
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
    models = [model for model in config["models"] if model.get("faithfulness_sweep")]
    entries: list[dict] = []

    for model in models:
        for seed in config["training"]["seeds"]:
            label = f"{model['id']}_seed_{seed}"
            source_dir = checkpoint_root / model["id"] / f"seed_{seed}"
            checkpoint_out = out / "checkpoints" / label
            checkpoint_out.mkdir(parents=True, exist_ok=True)
            for name in ("ckpt_selected.pt", "checkpoint_selection.json"):
                source = source_dir / name
                destination = checkpoint_out / name
                copy_evidence(
                    source, destination, root=out,
                    kind="checkpoint" if name.endswith(".pt") else "selection_record",
                    entries=entries, model=model["id"], training_seed=seed,
                )

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

            for name in ("manifest.json", "preflight.json", "analysis.json"):
                copy_evidence(
                    sweep_dir / name,
                    out / "sweep_evidence" / label / name,
                    root=out, kind=f"sweep_{Path(name).stem}", entries=entries,
                    model=model["id"], training_seed=seed,
                )

    copy_evidence(
        args.config.resolve(), out / "experiment_config.toml",
        root=out, kind="experiment_config", entries=entries,
    )
    for source_name in (
        "README.md",
        "docs/REPRODUCE_EXPERIMENTS.md",
        "docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md",
        "requirements.txt",
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
