# When Explanations Outlive Their Data

This repository contains the code, experiment evidence, and dissertation for
*When Explanations Outlive Their Data: Faithfulness Decoupling in
Graph-Attention MARL Fleet Dispatch under Telemetry Degradation*.

The study asks whether graph-attention weights remain trustworthy as
explanations when vehicle telemetry becomes stale. It combines a SUMO
fleet-dispatch experiment with a freshness-aware explanation audit. The audit
does not improve or retrain a policy. It decides whether the available evidence
is strong enough for attention to be presented as an explanation.

## Current conclusion

The completed audit returns `WITHHOLD` for both graph-attention model families.
The experiment evidence is complete and reproducible, but attention does not
consistently pass the decision-relevance, extraction-stability, and
checkpoint-consistency checks. Attention should therefore remain an internal
diagnostic for these checkpoints rather than a user-facing explanation.

This is a trustworthiness result, not a claim that telemetry degradation caused
a specific change in dispatch performance or that Age of Information (AoI) is a
causal dose of faithfulness.

## Experiment design

The final model set contains:

| Model | Training observations | Role |
|---|---|---|
| MLP | Clean | Performance context without graph attention |
| GAT | Clean | Primary graph-attention model under audit |
| GAT-Outage | Tunnel-triggered 30-second outages | Degradation-aware training condition |

Each model uses training seeds 42, 43, and 44. Checkpoints are selected on
validation demand before evaluation on held-out test demand. GAT and GAT-Outage
are audited under clean telemetry and 10, 20, 30, and 60-second outages.

Tunnel entry is the physical trigger, but degradation is applied at the
observation boundary. SUMO continues to update the true vehicle state while the
policy receives the last valid position and speed. A read-only clean twin of the
same SUMO state supports paired comparison and never selects an action.

## Repository map

| Path | Purpose |
|---|---|
| `src/dispatch_marl/` | Environment, telemetry degradation, policies, training, and faithfulness measures |
| `scripts/` | Experiment runner, analysis, audit, figure, and thesis build commands |
| `configs/experiments/` | Fixed training and evaluation protocols |
| `scenarios/yubei/` | Versioned SUMO networks, demand variants, and tunnel manifests |
| `runs/dissertation_v8/` | Stable training checkpoints and clean evaluation artifacts |
| `runs/dissertation_v9_exposure_audit/` | Raw final audit runs |
| `results/dissertation_v9_exposure_audit/` | Validated summaries and release-audit decision |
| `docs/dissertation/` | Dissertation chapter sources |
| `docs/figures/` | Thesis figures generated from experiment outputs |

## Installation

Use Python 3.11 and a working SUMO/libsumo installation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For Intel macOS, use the compatible dependency profile instead:

```bash
python -m pip install -r requirements-macos-intel.txt
python -m pip install -e . --no-deps
```

That profile pins NumPy to the 1.26 line because the compatible Intel PyTorch
wheel was compiled against NumPy 1.x. Platform-specific SUMO notes are in
[`docs/REPRODUCE_EXPERIMENTS.md`](docs/REPRODUCE_EXPERIMENTS.md).

## Verify the environment

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_policy.py
.venv/bin/python scripts/test_faithfulness.py
```

The faithfulness test starts SUMO for its end-to-end section. TraCI connection
retries during startup are expected when the test later ends with
`evaluator OK`.

## Reproduce the experiment

Inspect the fixed commands and settings without starting a run:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --dry-run
```

Train, select, and evaluate the stable model set:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage train
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage select
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage evaluate
```

Run the complete read-only explanation framework from the selected frozen
checkpoints:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage framework --resume
```

The final faithfulness matrix contains 1,440 episode evaluations and can take
many hours on CPU. The runner records immutable manifests and rejects partial
training runs instead of treating them as complete.

The full protocol, individual stages, model identifiers, expected files, and
troubleshooting guidance are in
[`docs/REPRODUCE_EXPERIMENTS.md`](docs/REPRODUCE_EXPERIMENTS.md).

## Run only the release audit

When all experiment evidence already exists:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage audit
```

The audit writes:

- `results/dissertation_v9_exposure_audit/explanation_audit/audit_report.json`
- `results/dissertation_v9_exposure_audit/explanation_audit/audit_checks.csv`
- `results/dissertation_v9_exposure_audit/explanation_audit/AUDIT_REPORT.md`

The input contract, nine checks, decision states, and operating scope are
documented in
[`docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md`](docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md).

## Dissertation

The chapter sources are under `docs/dissertation/`. The generated submission
files are:

- `docs/When_Explanations_Outlive_Their_Data_DMU_Thesis_v9.docx`
- `docs/When_Explanations_Outlive_Their_Data_DMU_Thesis_v9.pdf`

The code is intended to accompany the dissertation submission. Build the Word
document from the supplied DMU template with:

```bash
.venv/bin/python scripts/build_dmu_thesis_docx.py \
  --base docs/When_Explanations_Outlive_Their_Data_DMU_Thesis.docx \
  --output docs/When_Explanations_Outlive_Their_Data_DMU_Thesis_v9.docx
```

## Citation and scope

The SUMO scenarios are reproducible synthetic environments derived from
versioned OpenStreetMap road networks and seeded demand generation. They are
used for controlled internal comparison, not as a representative sample of all
real fleet operations. Consult the dissertation's limitations before extending
the findings beyond the tested checkpoints, action rule, scenarios, and
telemetry conditions.
