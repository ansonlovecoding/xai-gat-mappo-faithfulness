# Explainable GAT-MAPPO for Fleet Dispatch

DMU final dissertation: an Explainable Graph-Attention Multi-Agent
Reinforcement Learning framework (GAT-MAPPO) for fleet dispatch under
telemetry degradation. The research focus is **explanation faithfulness** and
whether the explanation channel can be **decoupled** from the policy without
trading off control performance.

> Status: SUMO scenarios, taxi fleet + demand, PettingZoo env, telemetry-
> degradation layer, GAT-MAPPO policy, MAPPO training loop with
> best-checkpoint tracking, and non-learning baselines are all implemented
> and verified end-to-end. Faithfulness metrics and the decoupled
> explanation head are next.
>
> **See [`src/dispatch_marl/README.md`](src/dispatch_marl/README.md) for
> the MARL package's design rationale — layer by layer, including the
> reward-shaping lesson and the coupled-vs-decoupled explanation framing
> the dissertation is built around.**

## Progress checklist

Quick-glance view of what's done and what's left. Update as pieces land.

| Layer | Status |
|---|---|
| SUMO scenarios (3 Yubei areas + tunnels + taxi fleet + ride demand) | ✅ done |
| PettingZoo environment + TraCI bridge | ✅ done |
| Telemetry-degradation layer (`off` / `tunnel_triggered` / `random_dropout`) | ✅ done |
| GAT-MAPPO policy (with **coupled** attention explanation channel) | ✅ done |
| MAPPO training loop + best-checkpoint tracking | ✅ done |
| Non-learning baselines (Random / Nearest / SUMO greedy) | ✅ done |
| First trained checkpoint (7.7 pickups on central_park; beats random & nearest) | ✅ done |
| Colab GPU training workflow | ✅ done |
| **Faithfulness metrics** (comprehensiveness / sufficiency / attention-vs-gradient) | ❌ **next** |
| **Decoupled explanation head** (the dissertation's headline contribution) | ❌ blocked by faithfulness metric choice |
| Stronger training run (close the gap to SUMO greedy's 30 pickups) | ⚠️ partial |
| Degradation experiments (tunnel_triggered vs matched-rate random ablation) | ❌ not started |
| Thesis figures / result tables / write-up | ❌ not started |

**Why "faithfulness metrics" is next, not stronger training:** the dissertation's
core claim is about *whether attention faithfully explains the policy's
decisions*, not about beating a benchmark on pickup count. A faithfulness
metric can be computed on any policy (even weakly trained), and it defines
the objective the decoupled explanation head has to satisfy. Without it,
the decoupled head has nothing concrete to optimise against.

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

## MARL: env, policy, training, evaluation

The `src/dispatch_marl/` package implements a PettingZoo `ParallelEnv`
wrapping SUMO/TraCI, a hand-rolled GAT-MAPPO policy (attention weights
exposed as the coupled-explanation channel), a MAPPO training loop with
GAE + clipped PPO, a telemetry-degradation layer with `tunnel_triggered`
and `random_dropout` modes, and non-learning baselines (Random, Nearest,
plus SUMO's built-in greedy dispatcher for reference).

**Design rationale, layer by layer, is documented in
[`src/dispatch_marl/README.md`](src/dispatch_marl/README.md)** — that file
explains *why* the observation is a self-centric graph, why the reward
looks the way it does (with the v0→v1 shaping lesson), why the GAT is
hand-rolled instead of `torch-geometric`, and how the whole architecture
serves the coupled-vs-decoupled explanation experiment.

### Run baselines

```bash
python scripts/run_baselines.py                                      # all areas × all policies
python scripts/run_baselines.py --areas central_park --policies random,nearest
python scripts/run_baselines.py --save-json runs/baselines.json
```

Reports pickups, reward, and mean pending wait per (area × policy). The
`sumo_greedy` row is measured by shelling out to `sumo` with the built-in
matcher and reading `--statistic-output` — it's the strong non-learning
upper reference.

### Train a GAT-MAPPO policy

```bash
python scripts/train.py --area central_park --epochs 300 \
  --pickup-reward 10.0 --dispatch-reward 0.5 --wait-lambda 0.001 \
  --best-window 10 --save-every 50
```

Writes to `runs/mappo/<area>_<timestamp>/`:

- `train_log.jsonl` — one metrics row per epoch (pickups, reward, losses,
  rolling mean, best marker)
- `ckpt_epoch_XXXX.pt` — periodic snapshots (`--save-every` controls the
  cadence)
- `ckpt_best.pt` — best-so-far by rolling mean of training pickups
- `best_metadata.json` — which epoch and rolling-mean value won

Best-checkpoint tracking is important because PPO's classic entropy-
collapse failure mode makes the *final* checkpoint often the *worst*.
The rolling-mean-based selection kept the current best result usable
even after the late-run degradation described in each preserved run's
`SUMMARY.md`.

### Evaluate a checkpoint

```bash
python scripts/eval_policy.py <ckpt.pt> --episodes 5 --stochastic
python scripts/eval_policy.py <ckpt.pt> --episodes 5 --degradation tunnel_triggered
```

Cross-device checkpoints work transparently: a Colab CUDA-trained `.pt`
loads on local MPS without any changes, via `torch.load(map_location=…)`
plus a runtime override of `PolicyConfig.device`.

### Train on Colab (GPU)

For iteration speed the training loop runs on a Colab T4/L4 without
modification. Setup — install SUMO 1.27 from the DLR PPA, clone the repo,
run training, download the checkpoint — is walked through cell-by-cell in
[`docs/COLAB.md`](docs/COLAB.md).

### Preserved training runs

Curated "keeper" runs live under `results/` — see
[`results/README.md`](results/README.md) for the index. Each keeper has
its own `SUMMARY.md` with the exact hyperparameters, the evaluation table
against baselines, and findings worth citing in the dissertation.

## Repository layout

```
.
├── .env.example                # template for project env vars (SUMO_HOME, DISPLAY, …)
├── requirements.txt            # pinned env-stack deps
├── docs/
│   └── COLAB.md                # step-by-step: train on a Colab GPU, bring weights back
├── scenarios/
│   ├── smoke/                  # generated 3×3 grid (gitignored)
│   └── yubei/
│       ├── central_park/       # .osm.xml + .net.xml + .rou.xml + _taxis.rou.xml + .sumocfg + tunnels.json
│       ├── yuelai/
│       └── xiantao/
├── src/
│   └── dispatch_marl/          # the MARL package — see README.md inside for design rationale
│       ├── README.md           # WHY each layer looks the way it does
│       ├── _sumo.py            # resolves SUMO_HOME once, on import
│       ├── scenario.py         # loads a built Yubei scenario + tunnel manifest
│       ├── degradation.py      # telemetry-degradation layer (off / tunnel_triggered / random_dropout)
│       ├── env.py              # PettingZoo ParallelEnv wrapping SUMO via TraCI
│       ├── policies.py         # non-learning baselines (Random, Nearest, NoOp)
│       ├── training.py         # rollout + GAE + PPO update
│       └── models/
│           ├── gat.py          # hand-rolled multi-head graph attention layer
│           └── policy.py       # DispatchGATPolicy (encoder + actor + critic)
├── results/
│   └── mappo_central_park_reshaped_v1/   # first keeper training run — see SUMMARY.md
├── runs/                       # scratch — timestamped training / baseline outputs (gitignored)
└── scripts/
    ├── smoke_test.py           # SUMO + TraCI integration test
    ├── build_yubei.py          # OSM → SUMO network → trips → tunnel manifest
    ├── add_taxis.py            # taxi fleet + ride demand → updated sumocfg
    ├── preview_tunnels.py      # sumo-gui with tunnel edges highlighted in magenta
    ├── run_random_policy.py    # sanity-check the env with a uniform-random policy
    ├── test_policy.py          # shape / NaN / attention smoke test for the GAT-MAPPO policy
    ├── run_baselines.py        # random + nearest + sumo_greedy comparison table
    ├── train.py                # MAPPO training loop with best-checkpoint tracking
    └── eval_policy.py          # evaluate a trained .pt on any area / degradation mode
```

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
