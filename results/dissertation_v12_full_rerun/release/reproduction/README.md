# When Explanations Outlive Their Data

This repository contains the code, evidence, figures, and dissertation for
*When Explanations Outlive Their Data: Faithfulness Decoupling in
Graph-Attention MARL Fleet Dispatch under Telemetry Degradation*.

The study asks whether graph-attention weights can be released as explanations
when vehicle telemetry becomes stale. Tunnel entry triggers an
observation-layer freeze while SUMO continues to simulate the true vehicle
state. A read-only clean twin from the same SUMO step supports paired analysis.

## Main result

The freshness-aware explanation audit returns `WITHHOLD` for GAT, GAT-Outage,
and all ten selected checkpoints. Longer outages consistently increase stale
exposure, but pooled raw attention does not beat its action-aware matched-random
control and its stale-node response changes across checkpoints and extraction
rules. The result is about explanation release, not task-performance
superiority.

## Models and evidence

| Model | Training observations | Role |
|---|---|---|
| MLP | clean | non-attention capability context |
| GAT | clean | main attention model under audit |
| GAT-Outage | 30-second tunnel-triggered outages | degradation-aware comparison |

All models use training seeds 42–46 and 50 epochs. Selected checkpoints are tested
on held-out demand. GAT and GAT-Outage are audited under clean telemetry and
10, 20, 30, and 60-second outages.

| Path | Purpose |
|---|---|
| `src/dispatch_marl/` | environment, policies, degradation, and audit measures |
| `scripts/` | experiment, analysis, packaging, figure, and thesis commands |
| `configs/experiments/dissertation_v12_full_rerun.toml` | fixed final protocol |
| `runs/dissertation_v12_sumo120/` | checkpoints and raw experiment evidence |
| `results/dissertation_v12_full_rerun/` | compact analyses and audit decision |
| `docs/dissertation/` | dissertation source chapters |
| `docs/figures/` | generated thesis figures |

The current thesis uses the completed v12 rerun. See `docs/FULL_RERUN_V12.md` for its pinned CPU/SUMO runtime; the installation below is for general development.

## Install and verify

Use Python 3.11 and a working SUMO/libsumo installation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_policy.py
.venv/bin/python scripts/test_faithfulness.py
```

On Intel macOS, use `requirements-macos-intel.txt`; it pins NumPy below 2 for
the compatible PyTorch wheel. Short TraCI connection retries are normal if
`scripts/test_faithfulness.py` later ends with `evaluator OK`.

## Reproduce the experiment

Inspect the fixed commands first:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v12_full_rerun.toml \
  --dry-run
```

Run the full pipeline:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v12_full_rerun.toml \
  --stage all --resume
```

Run only the explanation framework when checkpoints and clean evaluations are
already present:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v12_full_rerun.toml \
  --stage framework --resume
```

Run only the final release decision when all evidence exists:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v12_full_rerun.toml \
  --stage audit
```

The complete procedure, expected files, internal model identifiers, and
troubleshooting guidance are in
[`docs/REPRODUCE_EXPERIMENTS.md`](docs/REPRODUCE_EXPERIMENTS.md). The audit
contract is documented in
[`docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md`](docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md).

## Run the viva demo

For a presenter-led, six-step viva demonstration, run:

```bash
.venv/bin/python scripts/viva_demo.py --guided --gui
```

The demo defaults to **CPU**, including on Apple Silicon. This is the verified
presentation path; automatic MPS selection has produced an invalid 512 GiB
buffer allocation in the critic during inference. Accelerator devices remain
explicit opt-ins through `--device`. This demo default does not change training
or experiment device selection.

Press **Enter in the terminal** to advance each step and annotated SUMO view;
enter `q` at a prompt or press `Ctrl+C` to stop. The English prompts cover:

1. System components and data flow (1 minute).
2. Actual checkpoint loading, configuration and execution rule (1 minute).
3. Actual input/output shapes, action probabilities and SUMO dispatch (3 minutes).
4. Frozen/true positions, paired attention, WAMSN and DEF (2 minutes).
5. The stored full-experiment explanation-release audit (1.5 minutes).
6. Saved evidence and implementation recap (1 minute).

This leaves about 30 seconds of a ten-minute slot for transitions. Guided mode
waits for the presenter, so this is a rehearsal budget, not a hard deadline.
After the four annotated SUMO views, journey playback runs for at most 90
seconds by default (`--journey-seconds N` changes this). It then returns to
the metrics and audit steps, even if the journey has not finished. The GUI
closes when the final step completes. Historical fallback samples are scored
but not animated against a later, unrelated simulation state.

For a terminal-only demonstration, omit `--gui`. For a non-interactive rehearsal
that performs real inference and generates the same JSON and figure, use:

```bash
.venv/bin/python scripts/viva_demo.py --guided --auto-advance --device cpu
```

The JSON includes actual model input/output shapes and the captured taxi's
action probabilities. Probabilities displayed for different candidate taxis
belong to separate local policies; they are not a softmax over taxis. The
stored audit is explicitly identified as offline evidence, and is not an audit
of an arbitrary custom checkpoint. The script falls back to the packaged
seed-42 GAT checkpoint in `results/dissertation_v12_full_rerun/release/checkpoints/`
when the original training-run checkpoint is absent. It uses TraCI consistently
for both headless capture and GUI presentation.

Create a deterministic illustration of one stale-exposed GAT decision:

```bash
.venv/bin/python scripts/viva_demo.py
```

The script uses the selected seed-42 GAT checkpoint, the first held-out demand
file, and a 60-second tunnel-triggered freeze. It writes
`runs/viva_demo/viva_demo.png` and `runs/viva_demo/viva_demo.json`. The image
shows the frozen and true positions, clean and degraded attention, WAMSN, DEF,
and the formal audit verdict. Use `--gui` to show SUMO during the capture. The
GUI opens at near-full-screen size and begins with a passenger request beside
the highlighted telemetry-loss tunnel. It then shows up to five nearby taxis
whose local action spaces contain that request, together with each policy
probability. A separate phase identifies the stale taxi and its last valid
position before the sampled assignment is shown. The GUI then follows the
assigned taxi. The status distinguishes normal movement from traffic waiting. It changes to
`PASSENGER PICKED UP` at pickup and `PASSENGER DROPPED OFF` at the destination,
then freezes the completed journey instead of continuing the taxi's idle route.
The default presentation lasts up to 10 minutes and leaves the GUI open
afterward; close the GUI window or press `Ctrl+C` when the discussion is
finished. Change the total
duration with `--gui-duration-seconds N`, the annotated phase timing with
`--gui-phase-seconds N`, and the playback speed with `--gui-step-delay-ms N`.
The four annotated views remain static for five seconds so their events are easy
to read without advancing past a taxi-state change. One-second ease-in/out camera
moves connect the views instead of abruptly jumping between map boundaries. The
subsequent tracking uses a 0.25-second SUMO step and a 100 ms frame delay, giving
four visual updates per simulated second at about 2.5x simulation speed. Override
these with `--gui-transition-seconds`, `--gui-simulation-step-s`, and
`--gui-step-delay-ms` when needed.
Use `--close-gui-on-finish` when an automatically closing GUI is preferred. The
script also saves screenshots for tunnel entry, stale telemetry, assignment,
pickup, and drop-off in `runs/viva_demo/` as presentation backups. Override the
window dimensions with `--gui-window-size WIDTH HEIGHT`. On macOS,
SUMO GUI also requires XQuartz
(`brew install --cask xquartz`). macOS 15 on Intel requires XQuartz 2.8.7 or
newer for a working GLX context. The script checks the GUI binary and X11
server before starting the experiment.

## Build the dissertation

```bash
.venv/bin/python scripts/build_dmu_thesis_docx.py \
  --base docs/When_Explanations_Outlive_Their_Data_DMU_Thesis.docx \
  --output docs/When_Explanations_Outlive_Their_Data_DMU_Thesis_Final.docx
```

The final repository package can be rebuilt with:

```bash
.venv/bin/python scripts/package_dissertation_release.py \
  --config configs/experiments/dissertation_v12_full_rerun.toml
```

SUMO outputs are controlled synthetic evidence, not a representative sample of
all real fleets. Conclusions are limited to the tested checkpoints, action
rule, road network, demand variants, and telemetry conditions.
