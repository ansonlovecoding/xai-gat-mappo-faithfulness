# Explainable GAT-MAPPO for Fleet Dispatch

DMU final dissertation: an Explainable Graph-Attention Multi-Agent
Reinforcement Learning framework (GAT-MAPPO) for fleet dispatch under
telemetry degradation. The research focus is **explanation faithfulness** and
whether the explanation channel can be **decoupled** from the policy without
trading off control performance.

> Status: early scaffolding. The SUMO simulator pipeline is wired up and three
> Chongqing Yubei District scenarios with tunnel-edge manifests are built. The
> multi-agent env, GAT-MAPPO policy, telemetry degradation layer, and
> faithfulness metrics are not yet implemented.

## Why Yubei? Tunnels as physical degradation zones

The experimental scenarios are three tunnel-rich slices of Chongqing's Yubei
District: **Central Park (中央公园)**, **Yuelai (悦来)**, and **Xiantao Data
Valley (仙桃数据谷)**. Inside a tunnel, real fleet vehicles lose GPS and V2X
signal — this is a *physically grounded* source of telemetry degradation,
which is a stronger framing for the dissertation than synthetic Gaussian
noise. Each scenario's `tunnels.json` lists the SUMO edges that pass through
real tunnels (extracted from the OSM `tunnel=yes` tag) so the env's
degradation layer can trigger on tunnel entry/exit per agent.

Tunnel edges are split into two sets in the manifest:

- **`tunnel_edges`** — *navigable* tunnels: edges with both incoming and
  outgoing SUMO connections. These are the canonical degradation set the
  env code should use.
- **`orphan_tunnel_edges`** — tunnel edges with one side of connectivity
  missing (`orphan-source` = vehicles can't drive in; `orphan-sink` =
  vehicles can drive in but not out). These are recorded for transparency
  but excluded from `tunnel_edges`, because the env can't realistically
  apply degradation to a tunnel no vehicle ever reaches.

Orphans are a known artifact of running `netconvert --keep-edges.by-vclass
passenger` over OSM data: separated-tube tunnels sometimes lose the
connector ramp at one end during filtering. Cite this in the methods
section if asked why effective tunnel counts differ from raw OSM counts.

## Requirements

- macOS (tested on Darwin 23, Apple Silicon)
- Python 3.11
- [SUMO](https://eclipse.dev/sumo/) 1.27.0 (DLR Homebrew tap)
- [XQuartz](https://www.xquartz.org/) (only needed for `sumo-gui` visualisation)

## Setup

### 1. Clone and create the Python environment

```bash
cd "<this directory>"
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Install SUMO

```bash
brew tap dlr-ts/sumo
brew trust dlr-ts/sumo
brew install --cask dlr-ts/sumo/sumo-gui
```

### 3. Install XQuartz (only required for `--gui` mode)

```bash
brew install --cask xquartz
```

After install, **log out of macOS and back in** so the X11 launchd socket is
registered. Open XQuartz once from `Applications → Utilities → XQuartz`.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env if your SUMO is at a non-default path.
```

`.env` is gitignored. `.env.example` is the committed template — keep them in
sync when adding new variables.

## Smoke test

The smoke test generates a 3×3 grid scenario via `netgenerate`, populates it
with random trips via `randomTrips.py`, and runs 200 simulation steps through
TraCI. It exists to catch broken `SUMO_HOME` wiring, version mismatches, and
missing dependencies before any real env code is written.

```bash
python scripts/smoke_test.py              # headless (fast)
python scripts/smoke_test.py --gui        # opens sumo-gui, auto-plays
python scripts/smoke_test.py --gui --delay 300   # slower playback (ms/step)
```

A healthy run ends with:

```
Simulation stats: {'total_departed': 100, 'max_concurrent': 67}
OK — SUMO end-to-end pipeline is healthy.
```

## Yubei scenarios

The three scenarios under `scenarios/yubei/` are built from live
OpenStreetMap data via the Overpass API. Each one consists of: the raw OSM
extract, a SUMO network with tunnel attributes preserved, baseline random
trips for traffic, a `.sumocfg`, and a `tunnels.json` manifest listing the
SUMO edges that pass through tunnels.

### Build (or rebuild) all scenarios

```bash
python scripts/build_yubei.py --all              # build the three areas
python scripts/build_yubei.py --area central_park
python scripts/build_yubei.py --all --force      # re-fetch OSM and rebuild
```

Per-area pipeline: `osmGet.py` (Overpass API) → `netconvert` (with
`--osm.extra-attributes tunnel,bridge,layer`) → `randomTrips.py` →
`tunnels.json` (tunnel edges extracted from `<param key="tunnel">` and split
into navigable vs. orphan by connectivity check).

Current tunnel-edge counts (navigable / orphaned):

| Area | Navigable | Orphans | Notes |
|---|---:|---:|---|
| `central_park` | **3** | 3 | Bbox as originally specified |
| `yuelai` | **17** | 2 | Bbox shifted ~3 km south — the New Town tunnels are not yet mapped in OSM, so the bbox now sits over the Liangjiang corridor (Huangmaoping, Xinchun, etc.). Cite this honestly when reporting results. |
| `xiantao` | **12** | 4 | Bbox as originally specified |

The bounding boxes are first-pass approximations defined in
`scripts/build_yubei.py`. If you change them, rerun the build and the
manifest updates automatically. Each `.osm.xml` is committed to pin the
network you trained on — OSM evolves, but your experiments do not.

> **Heads-up**: rerunning `build_yubei.py` rewrites each `.sumocfg` with the
> bare network + background-trips config, which overwrites any taxi fleet
> previously added by `scripts/add_taxis.py`. After every rebuild, re-run
> `python scripts/add_taxis.py --all` to restore the taxi config.

### Add a taxi fleet and ride demand

```bash
python scripts/add_taxis.py --all                                  # 20 taxis, 50 rides each
python scripts/add_taxis.py --all --taxis 30 --rides 80 --end-time 1800
python scripts/add_taxis.py --area central_park                    # one area only
```

This writes `<area>_taxis.rou.xml` (taxi fleet vehicles + person ride
requests) alongside the existing `<area>.rou.xml` background traffic, and
rewrites `<area>.sumocfg` to load both files and set
`device.taxi.dispatch-algorithm = traci`. With that setting SUMO performs
**no** automatic dispatch — the MARL policy will call
`traci.vehicle.dispatchTaxi(taxi_id, [reservation_id])` itself. This is the
correct production config for the experiments.

Taxi vehicles are configured with `has.taxi.device=true`,
`device.taxi.idle-algorithm=randomCircling`, and `device.taxi.end-time=-1`,
so they stay alive and remain dispatchable for the whole episode. (Note:
`vClass="taxi"` alone is *not* enough to attach the taxi device — that's a
road-permissions class. The explicit `has.taxi.device` param is required,
otherwise `traci.vehicle.getTaxiFleet()` will return empty silently.)

Quick visual sanity check (overrides traci dispatch with the built-in greedy
matcher so you can watch taxis actually serve riders without writing a
controller):

```bash
sumo-gui -c scenarios/yubei/central_park/central_park.sumocfg \
         --device.taxi.dispatch-algorithm greedy
```

### Visually verify tunnel highlights

```bash
python scripts/preview_tunnels.py --area yuelai     # one area
python scripts/preview_tunnels.py --all             # walk through all three
```

This generates a `tunnel_highlights.add.xml` per area, then launches
`sumo-gui` with magenta polylines overlaid on every *navigable* tunnel edge
so you can confirm they trace real tunnel sections of the road. Orphan
tunnels are not drawn (the env can't reach them either). Pan, zoom, hit
play to see vehicles flow. Close the window to exit (or, in `--all` mode,
advance to the next area).

The launched sumo-gui uses `--delay 200 --no-warnings` by default — slow
enough to watch traffic, quiet enough to ignore the "ends idling in a
cul-de-sac" messages that `randomCircling` produces when an idle taxi
wanders into a dead end (those taxis remain alive and dispatchable; the
warning is informational).

Tips for finding traffic that crosses tunnels:

- With dispatch=traci (default), taxis just randomCircle — tunnel
  crossings are rare. Override with greedy dispatch to see real
  pickup/dropoff routes that more often cross tunnels:
  ```bash
  sumo-gui -c scenarios/yubei/central_park/central_park.sumocfg \
           --additional-files scenarios/yubei/central_park/tunnel_highlights.add.xml \
           --device.taxi.dispatch-algorithm greedy --delay 200 --no-warnings
  ```
- Press **F9** → *Vehicles* tab → set *Exaggerate size by* to `5` or `10`
  so taxis are visible without zooming way in.
- Right-click any taxi → *Show Current Route* to draw its planned route.
- **Ctrl+L** → *Edge* → paste a tunnel edge ID from `tunnels.json` to
  jump-zoom to that tunnel.

Requires XQuartz (see Setup §3).

## Repository layout

```
.
├── .env.example                # template for project env vars (SUMO_HOME, DISPLAY, …)
├── requirements.txt            # pinned env-stack deps
├── scenarios/
│   ├── smoke/                  # generated 3×3 grid (gitignored)
│   └── yubei/
│       ├── central_park/       # .osm.xml + .net.xml + .rou.xml + _taxis.rou.xml + .sumocfg + tunnels.json
│       ├── yuelai/
│       └── xiantao/
└── scripts/
    ├── smoke_test.py           # SUMO + TraCI integration test
    ├── build_yubei.py          # OSM → SUMO network → trips → tunnel manifest
    ├── add_taxis.py            # taxi fleet + ride demand → updated sumocfg
    └── preview_tunnels.py      # sumo-gui with tunnel edges highlighted in magenta
```

The MARL package (`src/`), training entry points, and experiment configs will
be added in subsequent commits.

## Troubleshooting

- **`SUMO_HOME not set`** — copy `.env.example` to `.env`. Every script also
  auto-resolves common install paths as a fallback.
- **`FXApp::openDisplay: unable to open display :0.0`** — XQuartz isn't
  running. Run `open -a XQuartz`, or log out and back in once after install
  so macOS auto-starts it on demand.
- **`FatalTraCIError: Could not connect in 1 tries`** — usually a knock-on
  from the display error above; fix XQuartz first.
- **`pj_obj_create: Cannot find proj.db`** — SUMO's PROJ library can't find
  its coordinate-projection database. Harmless: SUMO falls back to a built-in
  projection. Networks build correctly.
- **`Removed invalid stop ... non existing edge`** (during `build_yubei.py`)
  — public-transit stops from OSM that reference edges filtered out by our
  `--keep-edges.by-vclass passenger` rule. Expected and harmless.
- **`tunnels.json` reports `n_tunnel_edges: 0`** — the bbox is in the wrong
  place, or the tunnels aren't mapped in OSM for that area. Either adjust the
  bbox in `scripts/build_yubei.py` or pick a different sub-area.
- **Tunnel counts dropped after a rebuild** — `build_yubei.py` now filters
  to *navigable* tunnels only (incoming + outgoing connections). Orphans are
  preserved in `orphan_tunnel_edges` for transparency; if you need to relax
  the filter, edit `_extract_tunnel_edges` in `scripts/build_yubei.py`.
- **`Vehicle 'taxi_N' ends idling in a cul-de-sac`** — `randomCircling`
  walked the taxi into a dead-end road and can't choose a next edge. Taxi
  stays parked there but remains dispatchable. Cosmetic; suppressed by
  `--no-warnings` in `preview_tunnels.py`.
- **`traci.vehicle.getTaxiFleet(0)` returns 0 even though taxis are running**
  — the taxi device isn't attached. `vClass="taxi"` alone is not enough; the
  vType needs `<param key="has.taxi.device" value="true"/>`. `scripts/add_taxis.py`
  sets this; rerun it if you've manually edited the route file.
- **`Fontconfig warning: invalid constant`** — cosmetic, comes from Homebrew's
  fontconfig interacting with XQuartz. Safe to ignore.
