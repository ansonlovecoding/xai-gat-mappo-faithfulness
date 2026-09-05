"""Validation gates for a completed severity-sweep artifact."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .provenance import MANIFEST_SCHEMA_VERSION


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def _cell_paths(sweep_dir: Path) -> list[Path]:
    cell_dir = sweep_dir / "cells"
    return sorted(cell_dir.glob("*.json")) if cell_dir.is_dir() else []


def validate_sweep(sweep_dir: Path, *, require_clean_git: bool = True) -> ValidationReport:
    report = ValidationReport()
    manifest_path = sweep_dir / "manifest.json"
    if not manifest_path.exists():
        report.errors.append("manifest.json is missing")
        return report

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        report.errors.append(
            f"manifest schema must be {MANIFEST_SCHEMA_VERSION}; "
            f"got {manifest.get('schema_version')}"
        )
    if manifest.get("faithfulness_config", {}).get("random_baseline") != "type_matched":
        report.errors.append("primary DEF baseline must be type_matched")
    if not manifest.get("faithfulness_config", {}).get("exclusion_variant"):
        report.errors.append("chosen-action exclusion audit must be enabled")
    if not manifest.get("checkpoint_sha256"):
        report.errors.append("checkpoint SHA-256 is missing")

    dirty = manifest.get("provenance", {}).get("git", {}).get("dirty")
    if require_clean_git and dirty is not False:
        report.errors.append("experiment was not run from a clean Git worktree")

    provenance = manifest.get("provenance", {})
    binary_version = provenance.get("sumo", {}).get("binary_version", "unknown")
    dependencies = provenance.get("dependencies", {})
    for package in ("traci", "sumolib"):
        binding_version = dependencies.get(package, "unknown")
        if (binary_version != "unknown" and binding_version != "unknown"
                and binding_version != binary_version):
            report.errors.append(
                f"SUMO binary {binary_version} does not match {package} {binding_version}"
            )

    expected = {
        f"{cell['axis']}_{cell['level']:g}_seed{seed}.json"
        for cell in manifest.get("cells", [])
        for seed in manifest.get("seeds", [])
    }
    paths = _cell_paths(sweep_dir)
    actual = {path.name for path in paths}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        report.errors.append(f"missing {len(missing)} planned cells: {missing[:5]}")
    if unexpected:
        report.errors.append(f"found unexpected cell files: {unexpected[:5]}")

    cells = []
    for path in paths:
        try:
            cell = json.loads(path.read_text())
            if "cell" not in cell or "faith_records" not in cell:
                raise KeyError("cell or faith_records")
            cells.append(cell)
        except (json.JSONDecodeError, KeyError) as exc:
            report.errors.append(f"invalid cell file {path.name}: {exc}")

    clean = [cell for cell in cells if cell["cell"]["axis"] == "clean"]
    degraded = [cell for cell in cells if cell["cell"]["axis"] != "clean"]
    if clean and max(float(cell["empirical_degradation_rate"]) for cell in clean) > 0.0:
        report.errors.append("clean cells contain degraded observations")
    if degraded and not any(float(cell["empirical_degradation_rate"]) > 0 for cell in degraded):
        report.errors.append("degradation manipulation produced zero exposure")

    records = [record for cell in degraded for record in cell.get("faith_records", [])]
    exposed = [record for record in records if record.get("n_stale_veh", 0) > 0]
    if degraded and not records:
        report.errors.append("degraded cells contain no faithfulness records")
    if degraded and not exposed:
        report.errors.append("sampled decisions never observed a stale vehicle node")
    if exposed and not all("stale_attention_shift" in record for record in exposed):
        report.errors.append("binary stale-attention shift is missing from exposed records")

    faith_cfg = manifest.get("faithfulness_config", {})
    exposed_every = int(faith_cfg.get("faithfulness_exposed_every", 0) or 0)
    min_exposed_records = int(
        faith_cfg.get("minimum_stale_exposed_records_per_degraded_cell", 0) or 0
    )
    min_exposed_episodes = int(
        faith_cfg.get("minimum_stale_exposed_episodes_per_degraded_cell", 0) or 0
    )
    if exposed_every:
        for cell in degraded:
            name = (
                f"{cell['cell']['axis']}={cell['cell']['level']:g}, "
                f"seed={cell['cell']['seed']}"
            )
            cell_exposed = [
                record for record in cell.get("faith_records", [])
                if record.get("n_stale_veh", 0) > 0
            ]
            exposed_episode_count = len({
                int(record["episode"]) for record in cell_exposed
            })
            if len(cell_exposed) < min_exposed_records:
                report.errors.append(
                    f"{name} has {len(cell_exposed)} stale-exposed records; "
                    f"minimum is {min_exposed_records}"
                )
            if exposed_episode_count < min_exposed_episodes:
                report.errors.append(
                    f"{name} has {exposed_episode_count} stale-exposed episodes; "
                    f"minimum is {min_exposed_episodes}"
                )
            audit = cell.get("stale_exposure_audit", {})
            if exposed_every == 1 and (
                int(audit.get("stale_exposed_decisions_seen", -1))
                != int(audit.get("stale_exposed_decisions_scored", -2))
            ):
                report.errors.append(
                    f"{name} did not score every stale-exposed decision"
                )
            if cell_exposed and not all(
                record.get(
                    "paired_primary_def_delta", record.get("paired_def_delta")
                ) is not None for record in cell_exposed
            ):
                report.errors.append(
                    f"{name} is missing paired clean-twin DEF values"
                )

    clean_drifts = [
        abs(float(record["drift"])) for cell in clean
        for record in cell.get("faith_records", []) if "drift" in record
    ]
    if clean_drifts and max(clean_drifts) > 1e-4:
        report.errors.append("clean-twin drift is non-zero in the clean condition")

    levels = sorted({float(cell["cell"]["level"]) for cell in degraded})
    empirical_aoi = {
        str(level): [
            float(record.get("max_aoi_s", 0.0))
            for cell in degraded
            if float(cell["cell"]["level"]) == level
            for record in cell.get("faith_records", [])
            if record.get("n_stale_veh", 0) > 0
        ]
        for level in levels
    }
    empirical_max = {
        str(level): max(empirical_aoi[str(level)], default=0.0)
        for level in levels
    }
    empirical_median = {
        str(level): median(empirical_aoi[str(level)])
        if empirical_aoi[str(level)] else 0.0
        for level in levels
    }
    median_values = [empirical_median[str(level)] for level in levels]
    if len(levels) > 1 and len({round(value, 6) for value in median_values}) == 1:
        report.errors.append(
            "configured outage levels produced no empirical AoI differentiation"
        )
    if any(right < left for left, right in zip(median_values, median_values[1:])):
        report.warnings.append(
            "empirical median AoI is not monotonic; report the observed distribution "
            "and do not treat configured AoI as a causal dose"
        )

    shifts = [float(record["stale_attention_shift"]) for record in exposed
              if "stale_attention_shift" in record]
    report.summary = {
        "planned_cells": len(expected),
        "completed_cells": len(actual),
        "degraded_records": len(records),
        "stale_exposed_records": len(exposed),
        "minimum_stale_exposed_records_per_degraded_cell": min_exposed_records,
        "minimum_stale_exposed_episodes_per_degraded_cell": min_exposed_episodes,
        "mean_stale_attention_shift_when_exposed": (
            sum(shifts) / len(shifts) if shifts else None
        ),
        "empirical_max_aoi_s_by_level": empirical_max,
        "empirical_median_aoi_s_by_level": empirical_median,
    }
    return report
