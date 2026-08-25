"""Fail fast when the local runtime cannot support a reproducible rerun."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.provenance import runtime_provenance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    provenance = runtime_provenance(PROJECT_ROOT)
    errors = []

    numpy_version = provenance["dependencies"]["numpy"]
    if numpy_version.split(".", 1)[0] != "1":
        errors.append(f"NumPy 1.x is required by the pinned Torch wheel; got {numpy_version}")

    binary = provenance["sumo"]["binary_version"]
    for package in ("traci", "sumolib"):
        version = provenance["dependencies"][package]
        if version != binary:
            errors.append(f"SUMO binary {binary} does not match {package} {version}")
    if provenance["git"]["dirty"] and not args.allow_dirty:
        errors.append("Git worktree is dirty; commit the experiment implementation first")

    print(f"python: {sys.version.split()[0]}")
    print(f"torch:  {provenance['dependencies']['torch']}")
    print(f"numpy:  {numpy_version}")
    print(f"SUMO:   {binary} via {provenance['sumo']['backend']}")
    print(f"traci:  {provenance['dependencies']['traci']}")
    print(f"git:    {provenance['git']['revision']} "
          f"({'dirty' if provenance['git']['dirty'] else 'clean'})")
    for error in errors:
        print(f"ERROR:  {error}")
    print(f"environment: {'PASS' if not errors else 'FAIL'}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
