# Explainable GAT-MAPPO for Fleet Dispatch

DMU final dissertation: an Explainable Graph-Attention Multi-Agent
Reinforcement Learning framework (GAT-MAPPO) for fleet dispatch under
telemetry degradation. The research focus is **explanation faithfulness** and
whether the explanation channel can be **decoupled** from the policy without
trading off control performance.

> Status: early scaffolding. Currently only the SUMO simulator pipeline is
> wired up. The multi-agent env, GAT-MAPPO policy, telemetry degradation
> layer, and faithfulness metrics are not yet implemented.

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

## Repository layout

```
.
├── .env.example          # template for project env vars (SUMO_HOME, DISPLAY, …)
├── requirements.txt      # pinned env-stack deps
├── scenarios/
│   └── smoke/            # generated grid scenario (gitignored)
└── scripts/
    └── smoke_test.py     # SUMO + TraCI integration test
```

The MARL package (`src/`), training entry points, and experiment configs will
be added in subsequent commits.

## Troubleshooting

- **`SUMO_HOME not set`** — copy `.env.example` to `.env`. The smoke test also
  auto-resolves common install paths as a fallback.
- **`FXApp::openDisplay: unable to open display :0.0`** — XQuartz isn't
  running. Run `open -a XQuartz`, or log out and back in once after install
  so macOS auto-starts it on demand.
- **`FatalTraCIError: Could not connect in 1 tries`** — usually a knock-on
  from the display error above; fix XQuartz first.
- **`Fontconfig warning: invalid constant`** — cosmetic, comes from Homebrew's
  fontconfig interacting with XQuartz. Safe to ignore.
