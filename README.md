# Explainable GAT-MAPPO for Fleet Dispatch

DMU final dissertation: an Explainable Graph-Attention Multi-Agent
Reinforcement Learning framework (GAT-MAPPO) for fleet dispatch under
telemetry degradation. The research focus is **explanation faithfulness** and
whether the explanation channel can be **decoupled** from the policy without
trading off control performance.

> Status: SUMO scenarios, taxi fleet + demand, PettingZoo env, telemetry-
> degradation layer, GAT-MAPPO policy, MAPPO training loop with
> best-checkpoint tracking, and non-learning baselines are all implemented
> and verified end-to-end. The faithfulness stack (DEF + WAMSN +
> attention-drift function, eval-time scoring, train-time sampling, and the
> structured-vs-random degradation ablation) is implemented in
> `src/dispatch_marl/faithfulness.py` + `scripts/eval_degradation_ablation.py`.
> Next: the MLP baseline, paired clean/degraded drift harness, the
> severity sweep, and the decoupled explanation head.
>
> **Definitive rerun protocol:** new experiments use
> `configs/experiments/dissertation_v4.toml` through
> `scripts/run_dissertation_experiments.py`. The protocol adds complete
> seeding, immutable manifests, input hashes, raw-cell preservation,
> type-matched DEF, binary stale-attention shift, and preflight gates. See
> [`docs/EXPERIMENT_CODEBASE.md`](docs/EXPERIMENT_CODEBASE.md). The checklist
> below is a historical development record, not the current run command.

> **Explanation release audit:** the freshness-aware explanation audit converts
> the experiment evidence into an explicit `ELIGIBLE`, `WITHHOLD`, or
> `INCOMPLETE` decision for every model and checkpoint. Run the complete frozen-
> checkpoint workflow with `scripts/run_dissertation_experiments.py --config
> configs/experiments/dissertation_v9_exposure_audit.toml --stage framework
> --resume`. The inputs, checks, outputs, and operating scope are documented in
> [`docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md`](docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md).
>
> **See [`src/dispatch_marl/README.md`](src/dispatch_marl/README.md) for
> the MARL package's design rationale — layer by layer, including the
> reward-shaping lesson and the coupled-vs-decoupled explanation framing
> the dissertation is built around.**

## Progress checklist

Organised by the proposal's objectives and milestones (M1 = 19 Jul, M2 = 9
Aug, M3 = 1 Sep). Legend: ✅ done, ⚠️ partial / in progress, ❌ not started.

### A. Simulator & environment (WP1 · Objective 2 · Milestone M1)

| Item | Status | Notes |
|---|---|---|
| SUMO scenarios (3 Yubei areas + tunnels) | ✅ | `scenarios/yubei/central_park/`, `yuelai/`, `xiantao/` |
| Taxi fleet + ride demand via `add_taxis.py` | ✅ | 20 taxis, 50 riders per scenario; `has.taxi.device=true` |
| PettingZoo `ParallelEnv` bridging SUMO via **libsumo** | ✅ | libsumo preferred, TraCI fallback (`_sumo.py`) |
| Decentralised action space `Discrete(K+1)` (no-op / accept k-th reservation) | ✅ | Section 7.1 aligned |
| Per-agent heterogeneous graph observation (self + K neighbours + K reservations) | ✅ | Section 7.1 aligned |
| Vehicle features: position, velocity, availability, AoI | ✅ | `SELF_FEAT_DIM=5`; AoI tracked per agent |
| Telemetry-degradation layer (`off` / `tunnel_triggered` / `random_dropout`) | ✅ | Section 7.2 aligned |
| AoI = current_time − last_valid_time bookkeeping | ✅ | in `DegradationLayer.update_and_get_aoi` |

### B. GAT-MAPPO dispatcher (WP2, WP3 · Objective 1 · Milestone M1)

| Item | Status | Notes |
|---|---|---|
| Hand-rolled multi-head GAT layer with softmax attention exposed | ✅ | `models/gat.py` |
| MAPPO training loop (rollout + GAE + clipped PPO) | ✅ | `training.py` |
| Best-checkpoint tracking (rolling-mean pickups) | ✅ | `train.py --best-window` |
| Colab GPU workflow with libsumo + Drive persistence | ✅ | `docs/COLAB.md` |
| First trained checkpoint on central_park (7.67 ± 2.05 pickups) | ✅ | `results/mappo_central_park_reshaped_v1/ckpt_epoch_0050.pt` |
| **Centralised critic** V(s) for CTDE (proposal §7.7) | ✅ | default ON since the A/B in `results/ab_centralised_critic_v1_seed42/`: +18 % mean pickups vs. per-agent V, and mitigates late-run entropy collapse. Toggle with `--no-centralised-critic`. |
| Stronger training run closing gap to SUMO greedy (30 pickups) | ⚠️ | best rolling-mean now 8.80 pickups (CTDE, seed 42); baseline v1 was 7.67. Still ~4× short of SUMO greedy — needs longer runs and possibly entropy-coefficient tuning next. |

### C. Baselines (Section 7.5)

| Baseline | Status | Notes |
|---|---|---|
| **B0** — Greedy dispatch (SUMO built-in bipartite matcher) | ✅ | via `scripts/run_baselines.py --policies sumo_greedy` |
| **B1** — MAPPO + MLP (no graph) | ⚠️ | `DispatchMLPPolicy` implemented (`--policy mlp`, ~param-matched to the GAT, verified end-to-end); full training run pending |
| **B2** — GAT-MAPPO (proposed) | ✅ | current `DispatchGATPolicy` |
| Extras: Random, NearestReservation (naive baselines beyond proposal) | ✅ | useful as lower bounds; kept in `policies.py` |

### D. Faithfulness evaluation infrastructure (Section 7.4 · Objectives 3 & 5 · Milestone M2 — **the core novel contribution**)

| Item | Status | Notes |
|---|---|---|
| Attention weights exposed by `policy.forward()` (Method A) | ✅ | `(L, B, H, N, N)` tensor already returned |
| Node-occlusion Method B: mask node i, measure ΔP(a*) | ✅ | `FaithfulnessEvaluator._comp/_suff` in `src/dispatch_marl/faithfulness.py` — counterfactual forwards with node masks |
| **DEF metric** (normalised comprehensiveness + sufficiency gains) | ✅ | `FaithfulnessEvaluator.evaluate_decision`; per-k Comp/Suff vs size-matched random baseline, DEF = ½(g_comp + g_suff) |
| **Margin-DEF** (logit-margin variant, saturation-robust) | ✅ | same counterfactual forwards (zero extra cost); probability-DEF loses signal on entropy-collapsed policies (π(a*)≈1 → Δπ≈0), the margin logit[a*]−max_other keeps ~500× the signal on B2's best ckpt; `def_m` in records, `H1m/H2m` in the analysis |
| **Attention drift** (JS divergence between clean and degraded α) | ✅ | `emit_clean_obs` env flag builds the clean twin of every obs; `eval_policy.py --drift` scores JS(α_clean, α_degraded) per decision; on by default in the degradation ablation |
| **WAMSN metric** (Σ α_i · AoI_i / AoI_max / Σ α_i) | ✅ | `compute_wamsn` over vehicle nodes; AoI reversed from obs features |
| Random-explanation baseline for the DEF gain terms | ✅ | size-matched random subsets, `n_random_baselines` per k, seeded RNG |
| Eval-time faithfulness scoring | ✅ | `eval_policy.py --faithfulness` → per-decision JSONL + summary |
| Train-time faithfulness sampling | ✅ | `train.py --faith-every-epochs N` → per-epoch DEF/WAMSN in `train_log.jsonl`; plot via `plot_train_faithfulness.py` |
| Structured-vs-random degradation ablation | ✅ | `eval_degradation_ablation.py`: off / tunnel_triggered / matched-rate random_dropout; plot via `plot_ablation.py` |
| Performance-degradation rate & faithfulness-degradation rate | ✅ | `scripts/sweep_severity.py`: dropout-rate axis × tunnel-noise axis × seeds, per-decision records + manifest |
| Faithfulness-decoupling operationalisation ((i) DEF vs perf gap, (ii) WAMSN–DEF correlation) | ✅ | `scripts/analyze_hypotheses.py`: H1–H4 with permutation Spearman, paired sign-flip (H2), bootstrap CIs — numpy-only |
| Decoupled explanation head (architectural comparison against coupled attention) | ⚠️ | `DecoupledExplainerHead` (reads detached post-GAT node embeddings, never feeds the actor) trained by occlusion distillation (`scripts/distill_explainer.py`: per-node Δmargin targets, KL loss, held-out Spearman report); paired coupled-vs-decoupled DEF via `scripts/eval_explainer.py`; first full run in progress |

### E. Dataset & experimental protocol (Section 7.6 · Milestone M2)

| Item | Status | Notes |
|---|---|---|
| OSM → SUMO network build reproducible from committed `.osm.xml` | ✅ | `scripts/build_yubei.py` |
| Ride-demand generation via `randomTrips.py`, seed-pinned | ✅ | seed=42 baked in |
| Fixed-length episodes (1200 s ≈ 1 h equivalent) | ✅ | `sumocfg` end=1200 |
| Chronological train/val/test split 70/15/15 | ✅ | `add_taxis.py --variants N` writes seeded rider-demand variants + `demand_manifest.json` (fleet identical, demand varies); `train.py --demand-split train` rotates per epoch, `eval_policy.py --demand-split test` evaluates held-out demand |
| Multiple degradation severity levels (paired clean/degraded variants) | ✅ | `scripts/sweep_severity.py`: clean + dropout-rate axis + tunnel-noise axis × seeds |
| Versioned dataset manifest (`.npz`/Parquet + seed/version hash) | ✅ | `tunnels.json` (network) + `demand_manifest.json` (demand variants + split) + per-sweep `manifest.json` (cells × seeds + ckpt) |

### F. Main experiment (Section 7.5 · WP4 · Objective 4 · Milestone M2)

| Item | Status | Notes |
|---|---|---|
| Sweep {B0, B1, B2} × severity × seed | ⚠️ | `sweep_severity.py` runs one GAT ckpt × severity × seed; multi-checkpoint orchestration + B0/B1 rows still manual |
| Paired significance tests over matched clean/degraded episodes | ⚠️ | H2's paired sign-flip pairs each degraded cell with the same-seed clean cell; per-episode demand variants still pending (section E) |
| **H1** — degradation ↑ → DEF ↓ | ⚠️ | test implemented in `analyze_hypotheses.py` (one-sided permutation Spearman, both axes); needs a trained ckpt + sweep to run |
| **H2** — DEF declines faster than performance (decoupling) | ⚠️ | implemented: faith-rate vs perf-rate per (level × seed), sign-flip test |
| **H3** — degradation ↑ → WAMSN ↑ | ⚠️ | implemented alongside H1 |
| **H4** — WAMSN and DEF negatively correlated | ⚠️ | implemented: pooled per-decision Spearman |
| **H5** — AoI-aware training changes the DEF trajectory (RQ4 mitigation) | ❌ | needs a training run with degradation ON (`train.py --degradation tunnel_triggered --faith-every-epochs N` already supports it) |

### G. Write-up & release (WP5, WP6 · Objective 6 · Milestone M3)

| Item | Status | Notes |
|---|---|---|
| MSc dissertation chapters (Intro / LR / Method / Experiments / Discussion / Conclusion) | ❌ | starts after M2 |
| IEEE paper submission (ITSC / ICTAI / ICC / GLOBECOM) | ❌ | condensed version of the dissertation |
| Open-source repo release (benchmark + degradation ops + DEF/WAMSN code) | ⚠️ | repo exists; needs public README polish, license header, install pipeline |
| Reproducible dataset release (episodes + degradation manifests) | ❌ | depends on E being done |

### Where the critical path runs

By milestone budget:

- **M1 (19 Jul)** — mostly done. The remaining baseline task was the B1 MAPPO+MLP condition sharing the same MAPPO trainer.
- **M2 (9 Aug)** — section D's metric core is done (DEF, WAMSN, occlusion,
  random baseline, eval- and train-time scoring, structured-vs-random
  ablation). Remaining blockers: the paired clean/degraded drift harness,
  the severity sweep, the E-section protocol (episode manifest + split),
  the H1–H5 analysis script, and the decoupled explanation head.
- **M3 (1 Sep)** — write-up, unblocked once M2 lands.

**Next best things to work on:** (1) close M1 with the B1 MAPPO+MLP baseline;
(2) the paired clean/degraded attention-drift
harness (the degradation layer only mutates observations, so the env can
emit clean obs alongside degraded ones in `infos` — no paired-episode
machinery needed); (3) the severity-sweep script that generates the dataset
H1–H4 are tested on.

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

- macOS (originally Darwin 23 / Apple Silicon; now also running on
  Darwin 24 / Intel — see the Intel note below)
- Python 3.11
- [SUMO](https://eclipse.dev/sumo/) 1.27.0 (DLR Homebrew tap) on Apple
  Silicon / Linux; **1.20.0 via the `eclipse-sumo` PyPI wheel on Intel
  macs** (no newer Intel build exists — the DLR pkg is arm64-only and
  PyPI wheels for x86_64 macOS stop at 1.20)
- [XQuartz](https://www.xquartz.org/) (only needed for `sumo-gui` visualisation)

### Intel-mac note (July 2026 machine migration)

The dev machine changed from Apple Silicon to an Intel i9. On Intel:

- `pip install eclipse-sumo==1.20.0 libsumo==1.20.0 traci==1.20.0
  sumolib==1.20.0` into the venv. `src/dispatch_marl/_sumo.py`
  auto-resolves the wheel as `SUMO_HOME`; no framework install needed.
- The committed networks declare `<net version="1.20">`, so SUMO 1.20
  loads them natively — no rebuild required.
- torch wheels for Intel macOS stop at **2.2.2**, which requires
  `numpy<2` — install `numpy==1.26.4` over the pinned 2.4.6.
- The wheel's binaries link Homebrew dylibs, including **xerces-c 3.2**
  specifically (3.3 is ABI-incompatible): `brew tap-new local/pins &&
  brew extract --version=3.2.5 xerces-c local/pins && brew install
  local/pins/xerces-c@3.2.5`, plus `brew install fox proj gettext
  fontconfig freetype jpeg-turbo libpng libtiff mesa mesa-glu libx11
  libxext libxft libxcursor libxrender libxrandr libxfixes libxi`.
- Keep `requirements.txt` at the 1.27 pins for Colab/Apple-Silicon
  reproducibility; the deviations above are machine-local only.

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
python scripts/add_taxis.py --area central_park --variants 20      # + 20 demand variants
```

`--variants N` additionally writes N rider-demand variant files
(`<area>_taxis_vNNN.rou.xml`) plus `demand_manifest.json` with a
chronological 70/15/15 train/val/test split (proposal §7.6). The taxi
fleet is identical across variants — only the rider demand changes — so
episodes differ in exactly one factor. The base `.sumocfg` is untouched;
variants are opt-in: `train.py --demand-split train` rotates them per
epoch, `eval_policy.py --demand-split test` evaluates on held-out demand.

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
│   ├── mappo_central_park_reshaped_v1/    # first "keeper" trained model — see SUMMARY.md
│   └── ab_centralised_critic_v1_seed42/   # A/B: centralised critic +18 % pickups — see SUMMARY.md
├── runs/                       # scratch dir — timestamped training / baseline outputs (gitignored)
└── scripts/
    ├── build_yubei.py          # scenario setup: OSM → SUMO network → tunnel manifest, per Yubei area
    ├── add_taxis.py            # scenario setup: add taxi fleet + rider demand to a built scenario
    ├── smoke_test.py           # smoke test: SUMO + TraCI end-to-end on a 3×3 grid (install check)
    ├── test_policy.py          # smoke test: GAT-MAPPO forward-pass shape / NaN / attention softmax
    ├── run_baselines.py        # main workflow: Random / Nearest / SUMO-greedy comparison table
    ├── train.py                # main workflow: MAPPO training loop (CTDE default) + best-ckpt tracking
    ├── eval_policy.py          # main workflow: evaluate a trained .pt on any area / degradation mode
    ├── preview_tunnels.py      # visualisation: sumo-gui with tunnel edges highlighted in magenta
    └── update_proposal.py      # docs maintenance: audited text substitutions on the proposal .docx
```

### What each script is for, in one line

- **`build_yubei.py`** — Fetch OpenStreetMap for a Yubei-district bounding
  box, convert to a SUMO `.net.xml` with tunnel-tag preservation, generate
  reproducible background traffic via `randomTrips.py`, and emit a
  `tunnels.json` that splits tunnel edges into *navigable* vs *orphan*.
  Run this once per area (or after changing a bbox).
- **`add_taxis.py`** — Add a taxi fleet (`has.taxi.device=true`,
  `randomCircling` idle behaviour, cyan riders) to a built scenario and
  rewrite its `.sumocfg` to `dispatch-algorithm=traci`. Run this after
  every `build_yubei.py` (which rewrites `.sumocfg` without the fleet).
- **`smoke_test.py`** — Boots SUMO on a synthetic 3×3 grid and steps
  through TraCI. Use it to verify a fresh Python venv, `SUMO_HOME`, and
  XQuartz install (Colab, new machine, dependency bump). Not tied to the
  Yubei scenarios.
- **`test_policy.py`** — Instantiates a fresh `DispatchGATPolicy`, feeds
  it a real observation from the env, and asserts shapes / no NaN / valid
  attention softmax. ~15 s, no training. Useful when refactoring the
  policy/env interface.
- **`run_baselines.py`** — Runs the three non-learning baselines (Random,
  NearestReservation, SUMO's built-in greedy dispatcher) across the three
  Yubei areas and prints the comparison table. Outputs JSON to
  `runs/baselines.json` for later plotting.
- **`train.py`** — The main MAPPO training loop. Rollout → GAE → clipped
  PPO update → best-checkpoint tracking. Default is CTDE (centralised
  critic); use `--no-centralised-critic` for the ablation. Writes to
  `runs/mappo/<area>_<ts>/` (JSONL log, periodic snapshots, `ckpt_best.pt`).
- **`eval_policy.py`** — Loads a `.pt` checkpoint and runs N evaluation
  episodes on any area with any degradation mode. Handles cross-device
  loading (Colab CUDA → local MPS). Emits a `.eval.json` summary next to
  the checkpoint.
- **`preview_tunnels.py`** — Launches `sumo-gui` with magenta polylines
  overlaid on the navigable tunnel edges of a chosen scenario. Used to
  visually verify a new bbox actually intersects tunnels and to generate
  screenshots for the dissertation.
- **`update_proposal.py`** — Applies a versioned list of text
  substitutions to `docs/Research_Proposal_Faithfulness_Decoupling.docx`,
  preserving formatting via per-run edits. Serves as an audit trail of
  every code-vs-proposal alignment change made so far.

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
