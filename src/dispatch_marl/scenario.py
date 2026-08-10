"""Load a built Yubei scenario (network + taxis + tunnel manifest)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Import for side-effect of resolving SUMO_HOME.
from . import _sumo  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YUBEI_ROOT = PROJECT_ROOT / "scenarios" / "yubei"


@dataclass(frozen=True)
class Scenario:
    name: str
    sumocfg: Path
    tunnel_edges: frozenset[str]
    orphan_tunnel_edges: tuple[dict, ...] = field(default_factory=tuple)
    background_seed: int = 42
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def load_demand_manifest(area: str, root: Path | None = None) -> dict | None:
    """Return the demand-variant manifest for an area, or None if the
    variants haven't been generated (scripts/add_taxis.py --variants N)."""
    root = root or YUBEI_ROOT
    path = root / area / "demand_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def demand_split_files(area: str, split: str, root: Path | None = None) -> list[str]:
    """Variant filenames for one chronological split ('train'/'val'/'test')."""
    manifest = load_demand_manifest(area, root)
    if manifest is None:
        raise FileNotFoundError(
            f"no demand_manifest.json for '{area}' — run "
            f"`python scripts/add_taxis.py --area {area} --variants N` first"
        )
    try:
        return list(manifest["split"][split])
    except KeyError:
        raise KeyError(f"unknown split {split!r}; expected train/val/test") from None


def load_scenario(area: str, root: Path | None = None) -> Scenario:
    root = root or YUBEI_ROOT
    scenario_dir = root / area
    if not scenario_dir.exists():
        raise FileNotFoundError(
            f"Scenario '{area}' not found at {scenario_dir}. "
            f"Run `python scripts/build_yubei.py --area {area}` first."
        )
    sumocfg = scenario_dir / f"{area}.sumocfg"
    manifest = json.loads((scenario_dir / "tunnels.json").read_text())
    return Scenario(
        name=area,
        sumocfg=sumocfg,
        tunnel_edges=frozenset(manifest["tunnel_edges"]),
        orphan_tunnel_edges=tuple(manifest.get("orphan_tunnel_edges", [])),
        background_seed=int(manifest.get("seed", 42)),
        bbox=tuple(manifest.get("bbox", [0.0] * 4)),
    )
