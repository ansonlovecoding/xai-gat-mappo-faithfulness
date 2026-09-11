"""Immutable experiment identity and machine-readable provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 2


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_state(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(root, "diff", "--binary", "HEAD")
    return {
        "revision": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def runtime_provenance(root: Path) -> dict[str, Any]:
    adjacent_sumo = Path(sys.executable).with_name("sumo")
    sumo_binary = shutil.which("sumo")
    if sumo_binary is None and adjacent_sumo.exists():
        sumo_binary = str(adjacent_sumo)
    sumo_result = subprocess.run(
        [sumo_binary, "--version"], text=True, capture_output=True, check=False
    ) if sumo_binary else subprocess.CompletedProcess([], 127, "", "not found")
    match = re.search(r"Version\s+(\d+\.\d+\.\d+)", sumo_result.stdout)
    try:
        from ._sumo import USING_LIBSUMO, traci as sumo_backend
        backend = sumo_backend.__name__
        using_libsumo = USING_LIBSUMO
    except (ImportError, RuntimeError):
        backend = "unavailable"
        using_libsumo = False
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": hardware_inventory(),
        "command": [str(arg) for arg in sys.argv],
        "cwd": os.getcwd(),
        "dependencies": dependency_versions(
            ["numpy", "torch", "gymnasium", "pettingzoo", "traci", "sumolib",
             "eclipse-sumo"]
        ),
        "sumo": {
            "binary_version": match.group(1) if match else "unknown",
            "binary": sumo_binary or "unavailable",
            "backend": backend,
            "using_libsumo": using_libsumo,
        },
        "git": git_state(root),
    }


def hardware_inventory() -> dict[str, Any]:
    """Record the current machine only; never backfill historical runs."""
    cpu = platform.processor() or None
    memory = None
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            result = subprocess.run(["/usr/sbin/sysctl", "-n", key],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                if key == "hw.memsize":
                    memory = int(result.stdout.strip())
                else:
                    cpu = result.stdout.strip() or cpu
    else:
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            pass
    return {"cpu_model": cpu, "logical_cpus": os.cpu_count(),
            "installed_memory_bytes": memory}


def file_inventory(paths: Iterable[Path], root: Path | None = None) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted({Path(p).resolve() for p in paths}):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"provenance input does not exist: {path}")
        try:
            display = str(path.relative_to(root.resolve())) if root else str(path)
        except ValueError:
            display = str(path)
        inventory.append({"path": display, "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    return inventory


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def create_or_validate_manifest(path: Path, specification: dict[str, Any],
                                provenance: dict[str, Any]) -> dict[str, Any]:
    experiment_id = config_hash(specification)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("experiment_id") != experiment_id:
            raise RuntimeError(
                f"output directory belongs to experiment "
                f"{existing.get('experiment_id', 'unknown')}; requested {experiment_id}"
            )
        existing_provenance = existing.get("provenance", {})
        if existing_provenance.get("inputs") != provenance.get("inputs"):
            raise RuntimeError("input-file inventory changed since the run was created")
        if (existing_provenance.get("git", {}).get("revision")
                != provenance.get("git", {}).get("revision")):
            raise RuntimeError("Git revision changed since the run was created")
        return existing

    # Keep specification fields at the top level so legacy analysis scripts
    # can read new manifests while schema-aware tools use experiment_id.
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        **specification,
        "provenance": provenance,
    }
    atomic_write_json(path, manifest)
    return manifest
