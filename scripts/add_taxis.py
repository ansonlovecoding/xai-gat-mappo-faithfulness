"""Add a taxi fleet and ride-request demand to a built Yubei scenario.

For each area, writes `<area>_taxis.rou.xml` containing:
  - A `vType id="taxi" vClass="taxi"` (the vClass auto-enables the SUMO taxi
    device on every vehicle of this type)
  - N taxi vehicles spawned at t=0 on randomly chosen drivable edges, each
    on a single-edge route, idling there until dispatched
  - M `<person>` ride requests with random origin/destination and gradual
    arrival times across the first half of the simulation window

Then rewrites `<area>.sumocfg` to:
  - include both the background trips AND the new taxi route file
  - set `device.taxi.dispatch-algorithm = traci`, so SUMO does *no* automatic
    dispatch — the GAT-MAPPO policy is responsible for calling
    `traci.vehicle.dispatchTaxi(taxi_id, [reservation_id])` itself

This decoupling matters for the dissertation: the env exposes the raw
reservation pool, the policy chooses the matching, and we can compare against
SUMO's built-in greedy / greedyShared baselines later by overriding the
dispatch algorithm at runtime.

Run from project root:
    python scripts/add_taxis.py --area central_park
    python scripts/add_taxis.py --all
    python scripts/add_taxis.py --all --taxis 30 --rides 80 --end-time 1800

Quick visual check (overrides traci dispatch with built-in greedy so you can
see taxis actually serve rides without writing a controller):
    sumo-gui -c scenarios/yubei/central_park/central_park.sumocfg \\
             --device.taxi.dispatch-algorithm greedy
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


_resolve_sumo_home()
SCENARIO_ROOT = PROJECT_ROOT / "scenarios" / "yubei"


def _list_drivable_edges(net_file: Path) -> list[tuple[str, float]]:
    """Return [(edge_id, length_of_lane_0)] for real drivable edges."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges: list[tuple[str, float]] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        lane = edge.find("lane")
        if lane is None:
            continue
        try:
            length = float(lane.get("length", "0"))
        except ValueError:
            continue
        edges.append((edge.get("id"), length))
    return edges




def _build_taxi_route_xml(
    edges: list[tuple[str, float]],
    n_taxis: int,
    n_rides: int,
    seed: int,
    end_time: int,
) -> ET.ElementTree:
    rng = random.Random(seed)
    routes = ET.Element("routes")

    vtype = ET.SubElement(
        routes,
        "vType",
        id="taxi",
        vClass="taxi",
        personCapacity="4",
        color="1,1,0",  # bright yellow so taxis are easy to spot in sumo-gui
        # Oversized vs. real cars (~1.8m × 4.5m) so taxis are visible even
        # when zoomed out far enough to see most of the network.
        width="3.0",
        length="7.0",
    )
    # vClass="taxi" controls *road permissions* only — it does NOT
    # automatically attach SUMO's taxi device. The device must be enabled
    # explicitly, otherwise getTaxiFleet() returns empty and dispatch fails
    # silently.
    ET.SubElement(vtype, "param", key="has.taxi.device", value="true")
    # randomCircling: when the original trip is finished, the taxi extends
    # itself with random circular edges instead of leaving the simulation.
    ET.SubElement(vtype, "param", key="device.taxi.idle-algorithm", value="randomCircling")
    # end-time = -1 means the taxi never voluntarily ends its shift.
    ET.SubElement(vtype, "param", key="device.taxi.end-time", value="-1")

    all_edge_ids = [eid for eid, _ in edges]

    # Give each taxi a multi-edge initial trip with random origin/destination.
    # During this trip the vehicle is alive and EMPTY (no reservation), so the
    # dispatcher can match it. When the trip ends, randomCircling extends it.
    taxi_elements = []
    for i in range(n_taxis):
        origin, dest = rng.sample(all_edge_ids, 2)
        trip = ET.Element(
            "trip",
            id=f"taxi_{i}",
            type="taxi",
            depart="0",
            attrib={"from": origin, "to": dest},
        )
        taxi_elements.append(trip)

    # Generate rider demand with random origin/destination and gradual arrival.
    # color="0,1,1" = cyan so riders stand out against pedestrians (if any).
    rider_elements = []
    demand_window_end = max(1, end_time // 2)
    for i in range(n_rides):
        depart = rng.randint(0, demand_window_end)
        origin, dest = rng.sample(all_edge_ids, 2)
        person = ET.Element(
            "person",
            id=f"rider_{i}",
            depart=str(depart),
            color="0,1,1",
        )
        ET.SubElement(person, "ride", attrib={"from": origin, "to": dest, "lines": "taxi"})
        rider_elements.append(person)

    # SUMO requires depart-sorted entries in route files. Sort vehicles +
    # persons by depart, keep vType at the top.
    sortable = taxi_elements + rider_elements
    sortable.sort(key=lambda e: float(e.get("depart", "0")))
    for el in sortable:
        routes.append(el)

    ET.indent(routes, space="  ")
    return ET.ElementTree(routes)


def _write_sumocfg(
    cfg_file: Path,
    net_file: Path,
    route_files: list[Path],
    end_time: int,
) -> None:
    route_names = ",".join(f.name for f in route_files)
    cfg_file.write_text(
        f"""<configuration>
  <input>
    <net-file value="{net_file.name}"/>
    <route-files value="{route_names}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{end_time}"/>
  </time>
  <processing>
    <device.taxi.dispatch-algorithm value="traci"/>
  </processing>
  <report>
    <ignore-route-errors value="true"/>
  </report>
</configuration>
"""
    )


def add_taxis(area: str, n_taxis: int, n_rides: int, seed: int, end_time: int) -> dict:
    scenario_dir = SCENARIO_ROOT / area
    net_file = scenario_dir / f"{area}.net.xml"
    background_rou = scenario_dir / f"{area}.rou.xml"
    cfg_file = scenario_dir / f"{area}.sumocfg"
    taxis_rou = scenario_dir / f"{area}_taxis.rou.xml"

    if not net_file.exists():
        raise FileNotFoundError(
            f"net file not found: {net_file}. Run scripts/build_yubei.py first."
        )

    edges = _list_drivable_edges(net_file)
    if len(edges) < 4:
        raise RuntimeError(f"too few drivable edges ({len(edges)}) to add taxis")

    tree = _build_taxi_route_xml(edges, n_taxis, n_rides, seed, end_time)
    tree.write(taxis_rou, encoding="utf-8", xml_declaration=True)

    _write_sumocfg(cfg_file, net_file, [background_rou, taxis_rou], end_time)

    return {
        "area": area,
        "n_drivable_edges": len(edges),
        "n_taxis": n_taxis,
        "n_rides": n_rides,
        "seed": seed,
        "end_time": end_time,
        "taxi_route_file": taxis_rou.name,
        "cfg_file": cfg_file.name,
    }


def _discover_areas() -> list[str]:
    if not SCENARIO_ROOT.exists():
        return []
    return sorted(p.name for p in SCENARIO_ROOT.iterdir() if (p / "tunnels.json").exists())


def main() -> int:
    areas_available = _discover_areas()
    if not areas_available:
        print("No built scenarios found. Run scripts/build_yubei.py first.")
        return 1

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--area", choices=areas_available, help="single area")
    g.add_argument("--all", action="store_true", help="apply to every built area")
    parser.add_argument("--taxis", type=int, default=20, help="fleet size (default 20)")
    parser.add_argument("--rides", type=int, default=50, help="number of ride requests (default 50)")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--end-time", type=int, default=1200,
                        help="simulation end time in seconds (default 1200)")
    args = parser.parse_args()

    targets = areas_available if args.all else [args.area]
    print(f"Adding taxis to {len(targets)} scenario(s): "
          f"{args.taxis} taxis, {args.rides} rides, seed={args.seed}, end={args.end_time}s")

    summary = []
    for area in targets:
        print(f"\n[{area}]")
        result = add_taxis(area, args.taxis, args.rides, args.seed, args.end_time)
        print(f"  drivable edges: {result['n_drivable_edges']}")
        print(f"  wrote {result['taxi_route_file']}")
        print(f"  updated {result['cfg_file']} (dispatch-algorithm=traci)")
        summary.append(result)

    print("\nSummary:")
    for r in summary:
        print(f"  {r['area']}: {r['n_taxis']} taxis, {r['n_rides']} rides on {r['n_drivable_edges']} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
