# Reproducing the experiments

This document explains how to reproduce the dissertation experiments for
*When Explanations Outlive Their Data*. It is written for a reader who has
cloned the repository and wants to verify the result pipeline, not modify the
code.

There are two levels of reproduction:

1. **Verify the frozen artefacts**: inspect committed summaries, figures,
   checkpoints and audit JSON files.
2. **Rerun the experiments**: train or reuse checkpoints, rerun the severity
   sweeps, and regenerate the hypothesis analyses.

The frozen dissertation result set is `results/story_freeze_v1/`.

> **New definitive rerun:** the stricter protocol is defined in
> `configs/experiments/dissertation_v4.toml` and explained in
> `docs/EXPERIMENT_CODEBASE.md`. Sections 2-9 below retain the archived v1
> evidence and legacy reproduction details.

## 1. Environment

Use Python 3.11 and SUMO/libsumo. The project was evaluated with:

- local Intel macOS: `eclipse-sumo==1.20.0`, `libsumo==1.20.0`,
  `traci==1.20.0`, `sumolib==1.20.0`
- Colab/Linux: SUMO 1.27 line, as described in `docs/COLAB.md`

Create and activate a virtual environment:

```bash
cd "<repo-root>"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e . --no-deps
```

On the Intel macOS machine used for the freeze study, use the aligned SUMO
1.20 profile instead of `requirements.txt`:

```bash
pip install -r requirements-macos-intel.txt
pip install -e . --no-deps
```

The environment intentionally pins NumPy to the 1.26 line. The PyTorch wheel
used by this project was compiled against NumPy 1.x; installing NumPy 2.x can
make Torch import with a warning and later fail with `RuntimeError: Numpy is
not available` when faithfulness code converts tensors to NumPy arrays.

If your shell does not expose `python`, use `.venv/bin/python` explicitly in
all commands below.

Check the environment, then run the automated and end-to-end tests:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_faithfulness.py
```

A healthy run ends with:

```text
OK -- faithfulness pipeline works end-to-end.
```

SUMO may print transient TraCI connection retries during startup; this is not
an error if the script finishes successfully.

## 2. Frozen artefacts

For a new v4 run, inspect the complete command plan first:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py --dry-run
```

The orchestrator trains three declared seeds per model, selects a usable
checkpoint on validation demand only, runs held-out test sweeps, validates
every cell, and only then starts statistical analysis. The validation rule is
fixed before the final run: maximise mean pickups, then mean reward, then
prefer the earlier epoch. Selection uses the same stochastic policy mode as
final evaluation with a fixed seed. Test demand is never used for checkpoint
selection. A checkpoint must also exceed the declared minimum pickup count and
improve over the epoch-0 policy before it can enter the test sweep.

The primary degradation axis is observation-layer outage duration at 10, 20,
30 and 60 seconds. Tunnel entry triggers the outage, but the freeze is applied
outside SUMO when observations are built. AoI describes the stale observations
produced by each condition and is not interpreted as a causal dose.

The main result set is:

```text
results/story_freeze_v1/
```

Important files:

```text
results/story_freeze_v1/SUMMARY.md
results/story_freeze_v1/B2_sweep/analysis.json
results/story_freeze_v1/B2_sweep/manifest.json
results/story_freeze_v1/H5b_sweep/analysis.json
results/story_freeze_v1/H5b_sweep/manifest.json
results/story_freeze_v1/audit/type_matched_control.json
results/story_freeze_v1/audit/capability_spectrum_clean.json
results/story_freeze_v1/audit/compare_typematched_clean.json
results/story_freeze_v1/audit/compare_typematched_tunnel.json
results/story_freeze_v1/figs/
```

Read `results/story_freeze_v1/SUMMARY.md` first. It records the corrected
three-act story:

- Act 1: coupled attention is uninformative under the type-matched DEF
  baseline.
- Act 2: attention shifts onto stale telemetry under freeze degradation.
- Act 3: degradation-aware training does not restore faithfulness; the
  decoupled head gives only a small clean-data improvement.

Important reproducibility note: the archived `B2_sweep/` and `H5b_sweep/`
folders contain `analysis.json` and `manifest.json`, but not the raw per-cell
JSON files expected by `scripts/analyze_hypotheses.py`. Therefore, the frozen
analysis can be inspected directly, but full re-analysis from raw cell records
requires rerunning the sweep as described below.

## 3. Reproduce the main B2 severity sweep

The B2 freeze-era manifest records:

```text
checkpoint: runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt
epoch: 91
area: central_park
seeds: 42 43 44 45 46 47 48 49
episodes per cell seed: 3
demand split: test
AoI ladder: 5 15 30 60
corruption: freeze
faithfulness every: 8
```

If the checkpoint exists locally, rerun the sweep:

```bash
.venv/bin/python scripts/sweep_severity.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  --episodes 3 \
  --seeds 42 43 44 45 46 47 48 49 \
  --aoi-levels 5 15 30 60 \
  --demand-split test \
  --corruption freeze \
  --faithfulness-every 8 \
  --faithfulness-random-baselines 5 \
  --random-baseline type_matched \
  --out runs/sweeps/reproduce_B2_freeze
```

Then analyze:

```bash
.venv/bin/python scripts/analyze_hypotheses.py \
  runs/sweeps/reproduce_B2_freeze \
  --n-permutations 10000
```

The analysis script writes `analysis.json` in the sweep directory. Compare it
against:

```text
results/story_freeze_v1/B2_sweep/analysis.json
```

Expect small differences if SUMO, PyTorch, platform, or random-number
implementation differs. The qualitative pattern should match: H1 flat/floor,
H3 positive WAMSN, H4 negative WAMSN-DEF association, and H2 weaker than the
mechanism evidence.

## 4. Reproduce the H5 degradation-aware training sweep

The freeze-era H5 manifest records:

```text
checkpoint: runs/mappo/H5b_freeze/central_park_1784138109/ckpt_best.pt
epoch: 17
area: central_park
seeds: 42 43 44 45 46 47 48 49
episodes per cell seed: 3
demand split: test
AoI ladder: 5 15 30 60
corruption: freeze
faithfulness every: 8
```

If the checkpoint exists locally:

```bash
.venv/bin/python scripts/sweep_severity.py \
  runs/mappo/H5b_freeze/central_park_1784138109/ckpt_best.pt \
  --episodes 3 \
  --seeds 42 43 44 45 46 47 48 49 \
  --aoi-levels 5 15 30 60 \
  --demand-split test \
  --corruption freeze \
  --faithfulness-every 8 \
  --faithfulness-random-baselines 5 \
  --random-baseline type_matched \
  --out runs/sweeps/reproduce_H5b_freeze
```

Then analyze:

```bash
.venv/bin/python scripts/analyze_hypotheses.py \
  runs/sweeps/reproduce_H5b_freeze \
  --n-permutations 10000
```

Compare with:

```text
results/story_freeze_v1/H5b_sweep/analysis.json
```

## 5. Train checkpoints from scratch

Training is stochastic and may not reproduce bit-identical checkpoints across
machines. Use this section when the original `runs/mappo/.../ckpt_best.pt`
files are unavailable.

Train the main B2 GAT-MAPPO policy on clean telemetry:

```bash
.venv/bin/python scripts/train.py \
  --area central_park \
  --policy gat \
  --epochs 300 \
  --seed 42 \
  --demand-split train \
  --pickup-reward 10.0 \
  --dispatch-reward 0.5 \
  --wait-lambda 0.001 \
  --centralised-critic \
  --save-every 50 \
  --best-window 10
```

Train the H5 degradation-aware policy:

```bash
.venv/bin/python scripts/train.py \
  --area central_park \
  --policy gat \
  --epochs 300 \
  --seed 42 \
  --demand-split train \
  --degradation tunnel_triggered \
  --outage-duration 30 \
  --pickup-reward 10.0 \
  --dispatch-reward 0.5 \
  --wait-lambda 0.001 \
  --centralised-critic \
  --save-every 50 \
  --best-window 10
```

Each run writes to:

```text
runs/mappo/<area>_<timestamp>/
```

Use that run's `ckpt_best.pt` in the sweep commands above.

For faster GPU training in Colab, follow `docs/COLAB.md`.

## 6. Reproduce the decoupled explanation head comparison

First distil an explanation head from a frozen GAT checkpoint:

```bash
.venv/bin/python scripts/distill_explainer.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  --episodes 3 \
  --max-decisions 1500 \
  --seed 7 \
  --epochs 30
```

This writes `explainer_head.pt` next to the checkpoint unless `--out` is
provided.

Evaluate coupled attention against the decoupled head on clean telemetry with
the corrected type-matched baseline:

```bash
.venv/bin/python scripts/eval_explainer.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  runs/mappo/B2_gat/central_park_1783964871/explainer_head.pt \
  --episodes 2 \
  --seeds 42 43 \
  --degradation off \
  --every 5 \
  --random-baseline type_matched \
  --out runs/mappo/B2_gat/central_park_1783964871/compare_typematched_clean.json
```

Evaluate under tunnel-triggered freeze degradation:

```bash
.venv/bin/python scripts/eval_explainer.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  runs/mappo/B2_gat/central_park_1783964871/explainer_head.pt \
  --episodes 2 \
  --seeds 42 43 \
  --degradation tunnel_triggered \
  --every 5 \
  --random-baseline type_matched \
  --out runs/mappo/B2_gat/central_park_1783964871/compare_typematched_tunnel.json
```

Compare with:

```text
results/story_freeze_v1/audit/compare_typematched_clean.json
results/story_freeze_v1/audit/compare_typematched_tunnel.json
```

## 7. Reproduce a single checkpoint evaluation

To evaluate policy performance only:

```bash
.venv/bin/python scripts/eval_policy.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  --episodes 3 \
  --stochastic \
  --demand-split test
```

To evaluate performance plus DEF/WAMSN and attention drift under severe
freeze degradation:

```bash
.venv/bin/python scripts/eval_policy.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  --episodes 3 \
  --stochastic \
  --demand-split test \
  --degradation tunnel_triggered \
  --faithfulness \
  --faithfulness-every 8 \
  --faithfulness-random-baselines 5 \
  --drift
```

## 8. Expected output files

A severity sweep directory should contain:

```text
manifest.json
cells/
  clean_0_seed42.json
  max_aoi_5_seed42.json
  max_aoi_15_seed42.json
  max_aoi_30_seed42.json
  max_aoi_60_seed42.json
  ...
preflight.json
analysis.json
```

The raw cell files are the inputs needed by `scripts/analyze_hypotheses.py`.
Keep them if a reader needs to audit the exact per-decision records.

## 9. Known caveats

- The learned policies are far below SUMO greedy dispatch performance. This
  is a scope condition, not a hidden claim: the dissertation studies
  explanation faithfulness in the learned policy regime actually achieved.
- The primary corruption mechanism is freeze semantics at the observation
  boundary. Legacy noise-based results in `results/b1b2b3_sumo120_seed42_v1/`
  are not the dissertation's final evidence.
- Stochastic evaluation is intentional. Deterministic argmax evaluation can
  collapse to all-no-op on entropy-collapsed checkpoints.
- Platform changes can move exact pickups and p-values. Compare qualitative
  conclusions and confidence intervals, not bit-identical JSON.
- For the final dissertation claims, prefer type-matched DEF baselines over
  uniform baselines because the audit shows uniform baselines are confounded
  in candidate-action architectures.
