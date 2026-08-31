# Reproducing the dissertation experiments

This guide reproduces the final experiment for *When Explanations Outlive Their
Data: Faithfulness Decoupling in Graph-Attention MARL Fleet Dispatch under
Telemetry Degradation*.

> **Version note.** This document records the archived v4 experiment currently
> reported in the thesis. A corrected training protocol has been prepared for
> the planned full rerun, but its partial probe results must not be mixed with
> v4. See `docs/TRAINING_STABILITY_RERUN.md` and
> `configs/experiments/dissertation_v5.toml`.

The definitive protocol is:

```text
configs/experiments/dissertation_v4.toml
```

The completed local outputs are under:

```text
runs/dissertation_v4/
```

Compact citable summaries, analyses, manifests, and selection records are
committed under `results/dissertation_v4/`.

The older `results/story_freeze_v1/` directory is retained only for the
supporting construct-validity audit and historical comparison. It is not the
source of the final H1-H5 verdicts.

## 1. What the experiment does

The runner performs five linked tasks:

1. trains B1, B2, and D30 with seeds 42, 43, and 44;
2. selects checkpoints on validation demand, not test demand;
3. evaluates every selected policy on held-out clean test demand;
4. runs faithfulness sweeps for B2 and D30 under clean telemetry and 10, 20,
   30, and 60-second observation-layer outages;
5. validates, analyses, and summarises the outputs.

Tunnel entry triggers each outage. SUMO continues to simulate the true vehicle
state, while the policy observation freezes the last valid position and speed.
AoI describes the resulting stale observation; it is not treated as a causal
dose of faithfulness.

## 2. Environment

Use Python 3.11 and a working SUMO/libsumo installation.

```bash
cd "<repo-root>"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On the Intel macOS environment used for the original local runs, use the
aligned dependency profile:

```bash
python -m pip install -r requirements-macos-intel.txt
python -m pip install -e . --no-deps
```

NumPy is pinned to the 1.26 line because the installed PyTorch wheel was built
against NumPy 1.x. NumPy 2.x can cause:

```text
RuntimeError: Numpy is not available
```

Check the environment and tests:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_policy.py
.venv/bin/python scripts/test_faithfulness.py
```

`scripts/test_faithfulness.py` starts SUMO for its end-to-end section. TraCI
may print several connection retries while SUMO starts. The retries are not a
failure if the script ends with `evaluator OK`.

## 3. Inspect the fixed protocol

Print every command without running it:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --dry-run
```

Important fixed settings are:

| Item | Value |
|---|---|
| Training epochs | 150 |
| Training seeds | 42, 43, 44 |
| Checkpoint selection | stochastic validation pickups |
| Validation seed and episodes | 2026, three episodes |
| Evaluation seeds | 42-49 |
| Test episodes per seed/condition | three |
| Outage durations | 10, 20, 30, 60 seconds |
| Corruption | freeze last valid position and speed |
| Faithfulness sampling | every eight decisions |
| Random control | five type-matched subsets |

## 4. Run the complete experiment

From a new output directory, run:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --stage all
```

This is a long experiment. Training and the 240-episode faithfulness matrix for
each audited model can take many hours on CPU.

If a completed stage already exists, use `--resume`:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --stage all --resume
```

Resume skips complete immutable stages. It rejects partial training runs rather
than silently treating them as complete.

## 5. Run one stage or model

The available stages are:

```text
train -> select -> evaluate -> sweep -> preflight -> analyze -> summarize
```

Examples:

```bash
# Train and select all models
.venv/bin/python scripts/run_dissertation_experiments.py --stage train
.venv/bin/python scripts/run_dissertation_experiments.py --stage select

# Evaluate one model
.venv/bin/python scripts/run_dissertation_experiments.py \
  --stage evaluate --model B2_gat --resume

# Run and validate one faithfulness matrix
.venv/bin/python scripts/run_dissertation_experiments.py \
  --stage sweep --model B2_gat --resume
.venv/bin/python scripts/run_dissertation_experiments.py \
  --stage preflight --model B2_gat

# Analyse D30 and rebuild all summary tables. H5_gat_degraded is the
# backward-compatible code identifier for the D30 dissertation condition.
.venv/bin/python scripts/run_dissertation_experiments.py \
  --stage analyze --model H5_gat_degraded
.venv/bin/python scripts/run_dissertation_experiments.py --stage summarize
```

Valid model identifiers are:

```text
B1_mlp
B2_gat
H5_gat_degraded
```

Only B2 and D30 have faithfulness sweeps. B1 supplies performance context.

## 6. Run the revision audits

Verify that the v4 checkpoints, evaluation files, sweep manifests, source
revision, and Chapter 3 checkpoint table agree:

```bash
.venv/bin/python scripts/audit_dissertation_v4.py
```

The command should report `100` passed checks and no failures for the archived
v4 run.

Run the legal-random and greedy-nearest lower bounds on the same held-out
protocol:

```bash
.venv/bin/python scripts/run_matched_baselines.py
```

Run the additional faithfulness diagnostics. Start with the two-cell smoke
test, then run the complete B2 and D30 matrices. Cell files make these commands
resumable.

```bash
.venv/bin/python scripts/run_faithfulness_controls.py --smoke

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models B2_gat \
  --out runs/dissertation_revision_v1/faithfulness_controls/B2_full

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models H5_gat_degraded \
  --out runs/dissertation_revision_v1/faithfulness_controls/D30_full

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models B2_gat --action-row-only \
  --out runs/dissertation_revision_v1/faithfulness_controls/B2_action_row

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models H5_gat_degraded --action-row-only \
  --out runs/dissertation_revision_v1/faithfulness_controls/D30_action_row

.venv/bin/python scripts/analyze_faithfulness_controls.py
```

The control run records raw attention, Gradient x Input, the LOO perturbation
control, valid-node distributions, exact random top-k overlap, taxi-only rank
correlation, and attention aggregation sensitivity. The analysis uses episode
blocks rather than treating decisions from the same episode as independent.

## 7. Expected directory structure

```text
runs/dissertation_v4/
  training/
    <model>/seed_<training-seed>/
      ckpt_selected.pt
      checkpoint_selection.json
      validation/
  evaluations/
    <model>/seed_<training-seed>/
      eval_seed_<evaluation-seed>.json
  sweeps/
    B2_gat/seed_<training-seed>/
    H5_gat_degraded/seed_<training-seed>/
      manifest.json
      cells/
      preflight.json
      analysis.json
  summary.json
  summary.csv
  performance_context.json
  performance_context.csv
  training_seed_synthesis.json
  training_seed_synthesis.csv
```

Do not delete `cells/` if another reader needs to audit or recompute the
per-decision statistics.

## 8. Verify preflight before reading results

Every sweep used in the thesis must contain `preflight.json` with:

```json
{"ok": true}
```

Run the checks again with:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --stage preflight
```

Preflight checks the expected condition/seed matrix, source provenance,
checkpoint identity, type-matched random controls, chosen-action exclusion,
and empirical differentiation of the outage conditions.

## 9. Rebuild tables and figures

After all analyses exist:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --stage summarize
.venv/bin/python scripts/plot_dissertation_v4.py
```

The primary thesis tables are:

- `summary.csv`: condition-level measurements for each training seed;
- `performance_context.csv`: clean-test pickups and reward;
- `training_seed_synthesis.csv`: effect ranges and hypothesis consistency
  across independently trained policies.

The plotting script writes PNG and PDF versions to `docs/figures/`.

## 10. Expected qualitative result

A correct rerun should be interpreted from the generated files, not forced to
match one seed exactly across hardware. In the completed local v4 run:

- H3 is supported for all three B2 and all three D30 training seeds;
- H1 and H2 are unsupported for all six policies;
- H4 is unsupported for B2 and supported for two of three D30 seeds;
- the paired stale-attention shift is positive for seeds 42 and 43 and negative
  for seed 44 under both training regimes;
- D30 seed 44 is a weak policy replicate and is retained in the synthesis.

The conclusion is therefore not that AoI causes lower faithfulness. Longer
outages consistently increase stale-data exposure, while attention
reallocation and its relationship with DEF depend on the trained policy.

## 11. Supporting legacy audit

The older archive remains useful for understanding why the final protocol uses
type-matched random occlusions:

```text
results/story_freeze_v1/audit/type_matched_control.json
results/story_freeze_v1/audit/capability_spectrum_clean.json
results/story_freeze_v1/audit/compare_typematched_clean.json
results/story_freeze_v1/audit/compare_typematched_tunnel.json
```

These files document the construct-validity problem: deleting a request node
can also delete an action, while deleting a taxi node only removes information.
They should not be substituted for the v4 H1-H5 analysis.

## 12. Troubleshooting

### PyTorch says NumPy is unavailable

```bash
.venv/bin/python -m pip install --force-reinstall "numpy<2"
```

Then confirm:

```bash
.venv/bin/python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

### TraCI initially refuses the connection

Wait for the command to finish. Short retry messages during SUMO startup are
expected. Investigate only if the process exits non-zero or never reaches an
`OK`/completed message.

### Preflight fails

Read the `errors` array in `preflight.json`. Do not analyse or quote that sweep
until the missing cells or protocol mismatch is corrected and preflight passes.

### Results differ slightly across machines

SUMO, PyTorch, operating system, and floating-point differences can change
individual trajectories. Compare the full training-seed pattern and retain all
declared seeds rather than selecting the run that best matches the thesis.
