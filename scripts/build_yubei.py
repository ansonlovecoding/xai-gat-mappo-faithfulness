"""Build SUMO scenarios for Yubei District tunnel areas.

Each area is a small bounding box around a part of Chongqing Yubei District
that contains tunnels. Tunnels become the natural telemetry-degradation zones
for the GAT-MAPPO dispatch experiments — agents inside a tunnel edge lose
GPS / V2X signal.

Pipeline per area:
  1. Download OSM data for the bbox via osmGet.py (Overpass API)
  2. Convert to a SUMO net via netconvert, keeping the tunnel attribute
  3. Extract tunnel edge IDs and write `tunnels.json`
  4. Generate background random trips (randomTrips.py)
  5. Write a .sumocfg pointing at all of the above

Run from project root:
  python scripts/build_yubei.py --area central_park
  python scripts/build_yubei.py --all
  python scripts/build_yubei.py --all --force        # re-fetch OSM + rebuild

Verify each scenario visually after build:
  python scripts/smoke_test.py is NOT wired for these — use sumo-gui directly:
    sumo-gui -c scenarios/yubei/central_park/central_park.sumocfg
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_SUMO_HOME_CANDIDATES = [
    "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo",
    "/opt/homebrew/share/sumo",
    "/usr/local/share/sumo",
    "/usr/share/sumo",
]


def _resolve_sumo_home() -> str:
    env_value = os.environ.get("SUMO_HOME")
    if env_value and Path(env_value, "tools").is_dir():
        return env_value
    for candidate in _SUMO_HOME_CANDIDATES:
        if Path(candidate, "tools").is_dir():
            os.environ["SUMO_HOME"] = candidate
            bin_dir = str(Path(candidate, "bin"))
            if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate
    raise RuntimeError(
        f"SUMO not found. Set SUMO_HOME or install SUMO. Checked: {_SUMO_HOME_CANDIDATES}"
    )


SUMO_HOME = Path(_resolve_sumo_home())


@dataclass(frozen=True)
class Area:
    name: str
    description: str
    # west, south, east, north — geographic (lon, lat) corners
    bbox: tuple[float, float, float, float]
    n_trips: int = 300


# Bounding boxes are FIRST-PASS APPROXIMATIONS — verify each one on
# https://www.openstreetmap.org/ before relying on numbers in the
# dissertation. Each box should:
#   - cover the named area roughly 3-5 km on a side
#   - include at least one named tunnel (check OSM for `tunnel=yes` ways)
# If `tunnels.json` reports `n_tunnel_edges = 0`, the bbox missed.
AREAS = {
    "central_park": Area(
        name="central_park",
        description="Chongqing Central Park (中央公园), Yubei. Includes Central Park Tunnel.",
        bbox=(106.523, 29.708, 106.555, 29.736),
    ),
    "yuelai": Area(
        name="yuelai",
        description=(
            "Southern Yuelai / Liangjiang corridor, Yubei. "
            "Captures Huangmaoping (黄茅坪), Xinchun (新春) and surrounding tunnels. "
            "Note: the bbox is shifted ~3 km south of the true Yuelai New Town "
            "centre because the tunnels at the New Town itself are not yet mapped in OSM."
        ),
        bbox=(106.475, 29.685, 106.525, 29.715),
    ),
    "xiantao": Area(
        name="xiantao",
        description="Xiantao Data Valley (仙桃数据谷), Yubei. Xiantao Tunnel area.",
        bbox=(106.492, 29.653, 106.524, 29.681),
    ),
}


def _run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _download_osm(area: Area, out_dir: Path, force: bool) -> Path:
    osm_file = out_dir / f"{area.name}.osm.xml"
    if osm_file.exists() and not force:
        print(f"  OSM exists ({osm_file.name}) — skip (use --force to re-fetch)")
        return osm_file

    west, south, east, north = area.bbox
    _run(
        [
            sys.executable,
            str(SUMO_HOME / "tools" / "osmGet.py"),
            "-b", f"{west},{south},{east},{north}",
            "-p", area.name,
            "-d", str(out_dir),
        ]
    )
    # osmGet.py writes <prefix>_bbox.osm.xml (or .osm.xml.gz on newer versions).
    for produced in (
        out_dir / f"{area.name}_bbox.osm.xml",
        out_dir / f"{area.name}_bbox.osm.xml.gz",
    ):
        if produced.exists():
            target = osm_file if produced.suffix == ".xml" else osm_file.with_suffix(".xml.gz")
            produced.rename(target)
            return target
    raise RuntimeError(f"osmGet.py produced no recognised output in {out_dir}")


def _run_netconvert(osm_file: Path, net_file: Path) -> None:
    netconvert = shutil.which("netconvert") or str(SUMO_HOME / "bin" / "netconvert")
    _run(
        [
            netconvert,
            "--osm-files", str(osm_file),
            "-o", str(net_file),
            # only keep car-drivable edges
            "--keep-edges.by-vclass", "passenger",
            "--remove-edges.isolated",
            # preserve tunnel/bridge OSM tags as <param> elements on edges
            "--osm.extra-attributes", "tunnel,bridge,layer",
            # quality-of-life
            "--ramps.guess",
            "--junctions.join",
            "--no-internal-links",
            "--geometry.remove",
            "--tls.guess-signals",
            "--tls.discard-simple",
        ]
    )


def _extract_tunnel_edges(net_file: Path) -> list[str]:
    """Return SUMO edge IDs whose source OSM way had tunnel=yes."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    tunnel_truthy = {"yes", "true", "1"}
    tunnel_edges = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        for param in edge.findall("param"):
            if param.get("key") == "tunnel" and param.get("value") in tunnel_truthy:
                tunnel_edges.append(edge.get("id"))
                break
    return tunnel_edges


def _random_trips(net_file: Path, rou_file: Path, n_trips: int, seed: int) -> None:
    _run(
        [
            sys.executable,
            str(SUMO_HOME / "tools" / "randomTrips.py"),
            "-n", str(net_file),
            "-r", str(rou_file),
            "-e", str(n_trips),
            "--seed", str(seed),
            "--validate",
        ]
    )


def _write_cfg(cfg_file: Path, net_file: Path, rou_file: Path, end_time: int) -> None:
    cfg_file.write_text(
        f"""<configuration>
  <input>
    <net-file value=\"{net_file.name}\"/>
    <route-files value=\"{rou_file.name}\"/>
  </input>
  <time>
    <begin value=\"0\"/>
    <end value=\"{end_time}\"/>
  </time>
</configuration>
"""
    )


def build(area: Area, out_root: Path, *, force: bool = False, seed: int = 42) -> dict:
    out_dir = out_root / area.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{area.name}] {area.description}")

    osm_file = _download_osm(area, out_dir, force=force)

    net_file = out_dir / f"{area.name}.net.xml"
    if force or not net_file.exists():
        _run_netconvert(osm_file, net_file)
    else:
        print(f"  net exists ({net_file.name}) — skip")

    tunnel_edges = _extract_tunnel_edges(net_file)

    rou_file = out_dir / f"{area.name}.rou.xml"
    _random_trips(net_file, rou_file, area.n_trips, seed)

    cfg_file = out_dir / f"{area.name}.sumocfg"
    _write_cfg(cfg_file, net_file, rou_file, end_time=area.n_trips * 3)

    manifest = {
        "area": area.name,
        "description": area.description,
        "bbox": list(area.bbox),
        "seed": seed,
        "n_trips_background": area.n_trips,
        "n_tunnel_edges": len(tunnel_edges),
        "tunnel_edges": tunnel_edges,
    }
    manifest_file = out_dir / "tunnels.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    flag = "" if tunnel_edges else "  ⚠  NO TUNNEL EDGES — bbox may be wrong"
    print(f"  → {len(tunnel_edges)} tunnel edges, cfg: {cfg_file}{flag}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", choices=list(AREAS.keys()), help="single area to build")
    parser.add_argument("--all", action="store_true", help="build every area in the registry")
    parser.add_argument("--force", action="store_true", help="re-fetch OSM and rebuild even if outputs exist")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "scenarios" / "yubei",
                        help="output root directory (default: scenarios/yubei)")
    args = parser.parse_args()

    if not (args.area or args.all):
        parser.error("specify --area NAME or --all")

    targets = list(AREAS.values()) if args.all else [AREAS[args.area]]
    print(f"Building {len(targets)} scenario(s) into {args.out}")
    print(f"SUMO_HOME = {SUMO_HOME}")

    summary = []
    for area in targets:
        manifest = build(area, args.out, force=args.force, seed=args.seed)
        summary.append((area.name, manifest["n_tunnel_edges"]))

    print("\nSummary:")
    for name, n in summary:
        marker = "OK " if n > 0 else "⚠  "
        print(f"  {marker}{name}: {n} tunnel edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
