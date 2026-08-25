"""Validate a completed severity sweep before statistical analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.experiment_validation import validate_sweep  # noqa: E402
from dispatch_marl.provenance import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="development only: do not reject dirty-worktree provenance")
    args = parser.parse_args()

    report = validate_sweep(args.sweep_dir, require_clean_git=not args.allow_dirty)
    atomic_write_json(args.sweep_dir / "preflight.json", report.as_dict())
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"preflight: {'PASS' if report.ok else 'FAIL'}")
    print(f"report:    {args.sweep_dir / 'preflight.json'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
