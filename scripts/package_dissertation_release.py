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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/experiments/dissertation_v9_exposure_audit.toml",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results/dissertation_v9_exposure_audit/release",
    )
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    run_root = args.run_root or ROOT / config["output_root"]
    checkpoint_root = ROOT / config["checkpoint_root"]
    models = [model for model in config["models"] if model.get("faithfulness_sweep")]
    entries: list[dict] = []

    for model in models:
        for seed in config["training"]["seeds"]:
            label = f"{model['id']}_seed_{seed}"
            source_dir = checkpoint_root / model["id"] / f"seed_{seed}"
            checkpoint_out = args.out / "checkpoints" / label
            checkpoint_out.mkdir(parents=True, exist_ok=True)
            for name in ("ckpt_selected.pt", "checkpoint_selection.json"):
                source = source_dir / name
                destination = checkpoint_out / name
                shutil.copy2(source, destination)
                entries.append({
                    "kind": "checkpoint" if name.endswith(".pt") else "selection_record",
                    "model": model["id"],
                    "training_seed": seed,
                    "path": str(destination.relative_to(args.out)),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                })

            cells = run_root / "sweeps" / model["id"] / f"seed_{seed}" / "cells"
            archive = args.out / "audit_records" / f"{label}_cells.tar.gz"
            archive_cells(cells, archive)
            entries.append({
                "kind": "raw_audit_cells",
                "model": model["id"],
                "training_seed": seed,
                "source_files": len(list(cells.glob("*.json"))),
                "path": str(archive.relative_to(args.out)),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            })

    manifest = {
        "experiment": config["name"],
        "config": str(args.config.relative_to(ROOT)),
        "contents": entries,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"release files: {len(entries)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
