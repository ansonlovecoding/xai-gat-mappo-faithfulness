"""Open a built Yubei scenario in sumo-gui with tunnel edges highlighted.

Reads `scenarios/yubei/<area>/tunnels.json` for the tunnel-edge list, looks
up each edge's geometry in `<area>.net.xml`, and writes a SUMO additional
file with magenta polylines drawn over the tunnels (layer 100 so they sit
above the road). Then launches sumo-gui pointing at the cfg + additional.

The launched window is fully interactive — pan, zoom, play, pause. Useful
for manually verifying that the auto-extracted tunnel edges really do run
through tunnels on the OSM-derived network.

Run from project root:
  python scripts/preview_tunnels.py --area central_park
  python scripts/preview_tunnels.py --all          # open each in turn

Close the window to exit (or use --all to advance to the next area).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
    raise RuntimeError(f"SUMO not found. Checked: {_SUMO_HOME_CANDIDATES}")


SUMO_HOME = Path(_resolve_sumo_home())
SCENARIO_ROOT = PROJECT_ROOT / "scenarios" / "yubei"

# Bright magenta — high contrast against SUMO's default road colors.
HIGHLIGHT_COLOR = "255,0,255"
HIGHLIGHT_LINE_WIDTH = "4"
HIGHLIGHT_LAYER = "100"


def _discover_areas() -> list[str]:
    if not SCENARIO_ROOT.exists():
        return []
    return sorted(p.name for p in SCENARIO_ROOT.iterdir() if (p / "tunnels.json").exists())


def _generate_highlight_file(area: str) -> tuple[Path, int]:
    """Write an .add.xml with polylines tracing each tunnel edge.

    Returns (path, n_polys). n_polys can be < len(tunnel_edges) if some edges
    in the manifest can't be matched in the .net.xml (shouldn't happen unless
    the manifest is stale relative to the net file).
    """
    scenario_dir = SCENARIO_ROOT / area
    net_file = scenario_dir / f"{area}.net.xml"
    manifest = json.loads((scenario_dir / "tunnels.json").read_text())
    tunnel_edges = set(manifest["tunnel_edges"])

    tree = ET.parse(net_file)
    root = tree.getroot()

    add_root = ET.Element("additional")
    matched = 0
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id not in tunnel_edges:
            continue
        lane = edge.find("lane")
        if lane is None:
            continue
        shape = lane.get("shape")
        if not shape:
            continue
        poly = ET.SubElement(add_root, "poly")
        poly.set("id", f"tunnel_{edge_id}")
        poly.set("type", "tunnel_highlight")
        poly.set("color", HIGHLIGHT_COLOR)
        poly.set("layer", HIGHLIGHT_LAYER)
        poly.set("lineWidth", HIGHLIGHT_LINE_WIDTH)
        poly.set("shape", shape)
        matched += 1

    out_file = scenario_dir / "tunnel_highlights.add.xml"
    ET.ElementTree(add_root).write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file, matched


def preview(area: str, delay_ms: int = 200) -> None:
    scenario_dir = SCENARIO_ROOT / area
    cfg = scenario_dir / f"{area}.sumocfg"
    if not cfg.exists():
        raise FileNotFoundError(
            f"No built scenario at {cfg}. Run `python scripts/build_yubei.py --area {area}` first."
        )

    manifest = json.loads((scenario_dir / "tunnels.json").read_text())
    print(f"\n[{area}] {manifest['description']}")
    print(f"  {len(manifest['tunnel_edges'])} tunnel edges in manifest")

    add_file, matched = _generate_highlight_file(area)
    print(f"  wrote {add_file.name} ({matched} polylines)")

    sumo_gui = shutil.which("sumo-gui") or str(SUMO_HOME / "bin" / "sumo-gui")
    print(f"  launching sumo-gui (delay {delay_ms} ms/step) — close the window to continue")
    subprocess.run(
        [
            sumo_gui,
            "-c", str(cfg),
            "--additional-files", str(add_file),
            # Slow playback so the user can actually watch traffic; without
            # this sumo-gui races through 1200s of sim in a couple of seconds.
            "--delay", str(delay_ms),
            # Suppress "ends idling in a cul-de-sac" warnings from the
            # randomCircling taxi idle algorithm. These are informational
            # only — affected taxis stay alive and remain dispatchable.
            "--no-warnings",
        ],
        check=False,
    )


def main() -> int:
    areas = _discover_areas()
    if not areas:
        print("No built scenarios found under scenarios/yubei/. Run scripts/build_yubei.py first.")
        return 1

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--area", choices=areas, help="single area to preview")
    g.add_argument("--all", action="store_true", help="preview each area in turn (close window to advance)")
    parser.add_argument("--delay", type=int, default=200,
                        help="ms delay per simulation step in sumo-gui (default 200; use 0 for max speed)")
    args = parser.parse_args()

    targets = areas if args.all else [args.area]
    for area in targets:
        preview(area, delay_ms=args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
