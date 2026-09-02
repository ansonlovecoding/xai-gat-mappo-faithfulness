# Reproducing the dissertation experiments

This guide reproduces the final experiment for *When Explanations Outlive Their
Data: Faithfulness Decoupling in Graph-Attention MARL Fleet Dispatch under
Telemetry Degradation*.

> **Version note.** The stable models and clean evaluation are stored under
> `runs/dissertation_v8/`. The exposure-conditioned follow-up audit uses those
> frozen checkpoints and writes to a separate v9 directory. It does not retrain
> or select models after seeing the faithfulness results.

## Exposure-conditioned follow-up audit

The follow-up protocol addresses sparse tunnel exposure without changing the
policy, SUMO state, tunnel trigger, outage durations, or held-out demand. Its
fixed configuration is:

```text
configs/experiments/dissertation_v9_exposure_audit.toml
```

Compared with the original sweep, it:

1. runs six episodes per condition and evaluation seed;
2. keeps a broad background audit at one in every 16 decisions;
3. audits every decision containing at least one stale vehicle node;
4. evaluates the degraded observation and its exact clean twin with the same
   selected action and the same type-matched random subsets;
5. requires at least 20 stale-exposed records from at least three episodes in
   every degraded cell before analysis is accepted;
6. reports no-op and dispatch decisions separately, excluding forced no-op
   records with no valid request;
7. runs a three-episode deterministic argmax diagnostic for each selected GAT
   checkpoint.

Run the follow-up stages with:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage sweep
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage preflight
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage analyze
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage diagnose
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage summarize
```

The primary follow-up statistic is degraded type-matched DEF minus clean-twin
DEF among stale-exposed decisions. Negative values mean lower measured
faithfulness under the degraded observation. This paired test does not treat
the configured outage duration or AoI value as a causal dose.

The frozen-model training protocol is:

```text
configs/experiments/dissertation_v8.toml
```

The final audit protocol is:

```text
configs/experiments/dissertation_v9_exposure_audit.toml
```

Compact citable summaries, analyses, manifests, and selection records are
committed under `results/dissertation_v9_exposure_audit/`.

The older `results/story_freeze_v1/` directory is retained only for the
supporting construct-validity audit and historical comparison. It is not the
source of the final H1-H5 verdicts.

## 1. What the experiment does

The runner performs five linked tasks:

1. trains MLP, GAT, and GAT-Outage with seeds 42, 43, and 44;
2. selects checkpoints on validation demand, not test demand;
3. evaluates every selected policy on held-out clean test demand;
4. runs faithfulness sweeps for GAT and GAT-Outage under clean telemetry and 10, 20,
   30, and 60-second observation-layer outages;
5. validates and analyzes the outputs, including no-op/dispatch strata;
6. runs a descriptive deterministic diagnostic and summarizes all outputs.

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
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml --dry-run
```

Important fixed settings are:

| Item | Value |
|---|---|
| Training epochs | 40 (GAT), 50 (MLP and GAT-Outage) |
| Training seeds | 42, 43, 44 |
| Checkpoint selection | stochastic validation pickups |
| Validation seed and episodes | 2026, eight episodes |
| Evaluation seeds | 42-49 |
| Clean capability episodes per seed | three |
| Final audit episodes per seed/condition | six |
| Outage durations | 10, 20, 30, 60 seconds |
| Corruption | freeze last valid position and speed |
| Faithfulness sampling | every 16 decisions plus every stale-exposed decision |
| Random control | five type-matched subsets |
| Random-loss sensitivity | 30-second freeze; trigger probability 0.0023 |

## 4. Run the complete experiment

Train, select, and evaluate the stable model set first:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage train
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage select
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage evaluate
```

Then run the five v9 audit commands listed at the start of this guide. The final
faithfulness matrix contains 1,440 episodes and can take many hours on CPU.

Run the frozen-policy random-loss sensitivity analysis separately:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage robustness
.venv/bin/python scripts/analyze_random_loss_robustness.py \
  runs/dissertation_v9_exposure_audit/robustness/random_loss_30s \
  --out results/dissertation_v9_exposure_audit/random_loss_robustness
```

This adds 432 rollout evaluations: clean, 30-second tunnel-triggered, and
30-second randomly triggered conditions for each of six frozen checkpoints.
Each condition contains eight evaluation seeds and three held-out episodes.

If a completed stage already exists, use `--resume`:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage sweep --resume
```

Resume skips complete immutable stages. It rejects partial training runs rather
than silently treating them as complete.

## 5. Run one stage or model

The available stages are:

```text
train -> select -> evaluate
framework: diagnose -> sweep -> robustness -> preflight -> analyze -> summarize
           -> controls -> analyze-robustness -> analyze-controls -> audit
```

With selected checkpoints already available, run the complete explanation
framework in the declared order with:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage framework --resume
```

Examples:

```bash
# Train and select all models
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage train
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml --stage select

# Evaluate one model
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v8.toml \
  --stage evaluate --model B2_gat --resume

# Run and validate one faithfulness matrix
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage sweep --model B2_gat --resume
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage preflight --model B2_gat

# Analyse GAT-Outage and rebuild all summary tables.
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage analyze --model H5_gat_degraded
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage summarize
```

Valid model identifiers are:

```text
B1_mlp
B2_gat
H5_gat_degraded
```

Only GAT (`B2_gat`) and GAT-Outage (`H5_gat_degraded`) have faithfulness
sweeps. MLP (`B1_mlp`) supplies performance context.

## 6. Run the revision audits

The archived v4 audit remains available for historical verification:

```bash
.venv/bin/python scripts/audit_dissertation_v4.py
```

It is not the source of the final v9 hypothesis verdicts.

Run the legal-random and greedy-nearest lower bounds on the same held-out
protocol:

```bash
.venv/bin/python scripts/run_matched_baselines.py
```

Run the additional faithfulness diagnostics. Start with the two-cell smoke
test, then run the complete B2 and D30 matrices. Cell files make these commands
resumable.

```bash
.venv/bin/python scripts/run_faithfulness_controls.py --smoke \
  --checkpoint-root runs/dissertation_v8/training \
  --out runs/dissertation_v9_exposure_audit/faithfulness_controls

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models B2_gat \
  --checkpoint-root runs/dissertation_v8/training \
  --out runs/dissertation_v9_exposure_audit/faithfulness_controls/B2_full

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models H5_gat_degraded \
  --checkpoint-root runs/dissertation_v8/training \
  --out runs/dissertation_v9_exposure_audit/faithfulness_controls/H5_full

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models B2_gat --action-row-only \
  --checkpoint-root runs/dissertation_v8/training \
  --out runs/dissertation_v9_exposure_audit/faithfulness_controls/B2_action_row

.venv/bin/python scripts/run_faithfulness_controls.py \
  --models H5_gat_degraded --action-row-only \
  --checkpoint-root runs/dissertation_v8/training \
  --out runs/dissertation_v9_exposure_audit/faithfulness_controls/H5_action_row

.venv/bin/python scripts/analyze_faithfulness_controls.py \
  runs/dissertation_v9_exposure_audit/faithfulness_controls/B2_full \
  runs/dissertation_v9_exposure_audit/faithfulness_controls/H5_full \
  runs/dissertation_v9_exposure_audit/faithfulness_controls/B2_action_row \
  runs/dissertation_v9_exposure_audit/faithfulness_controls/H5_action_row \
  --out results/dissertation_v9_exposure_audit/faithfulness_controls \
  --fig-dir docs/figures --fig-prefix v9
```

The control run records raw attention, Gradient x Input, the LOO perturbation
control, valid-node distributions, exact random top-k overlap, taxi-only rank
correlation, and attention aggregation sensitivity. The analysis uses episode
blocks rather than treating decisions from the same episode as independent.

After the control and random-loss analyses are complete, apply the explanation
release rules:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage audit
```

This final stage is different from preflight. Preflight confirms that the
experiment evidence is valid to analyze. The explanation audit decides whether
the attention map may be presented as an explanation. See
[`FRESHNESS_AWARE_EXPLANATION_AUDIT.md`](FRESHNESS_AWARE_EXPLANATION_AUDIT.md)
for the inputs, decision rules, outputs, and use cases.

## 7. Expected directory structure

```text
runs/dissertation_v8/
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

results/dissertation_v9_exposure_audit/explanation_audit/
  audit_report.json
  audit_checks.csv
  AUDIT_REPORT.md

runs/dissertation_v9_exposure_audit/
  sweeps/
    B2_gat/seed_<training-seed>/
    H5_gat_degraded/seed_<training-seed>/
      manifest.json
      cells/
      preflight.json
      analysis.json
  faithfulness_controls/
  deterministic_diagnostics/
    <model>/seed_<training-seed>.json
  summary.json
  summary.csv
  action_stratified.json
  action_stratified.csv
  deterministic_diagnostics.json
  deterministic_diagnostics.csv
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
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage preflight
```

Preflight checks the expected condition/seed matrix, source provenance,
checkpoint identity, type-matched random controls, chosen-action exclusion,
and empirical differentiation of the outage conditions.

A passing preflight report does not mean that attention is eligible for release
as an explanation. Run the final `audit` stage after all supporting analyses.

## 9. Rebuild tables and figures

After all analyses exist:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage summarize
.venv/bin/python scripts/plot_dissertation_v4.py \
  --root runs/dissertation_v9_exposure_audit \
  --training-root runs/dissertation_v8 \
  --analysis-root results/dissertation_v9_exposure_audit \
  --out docs/figures --prefix v9
```

The primary thesis tables are:

- `summary.csv`: condition-level measurements for each training seed;
- `performance_context.csv`: clean-test pickups and reward;
- `action_stratified.csv`: eligible no-op and dispatch counts and their
  separate faithfulness diagnostics;
- `deterministic_diagnostics.csv`: three-episode argmax behavior for each
  selected GAT checkpoint;
- `training_seed_synthesis.csv`: effect ranges and hypothesis consistency
  across independently trained policies.
- `random_loss_robustness.csv`: tunnel-triggered and random-triggered paired
  shifts for the 30-second sensitivity analysis.
- `explanation_audit/AUDIT_REPORT.md`: the final explanation-release decision,
  failed checks, checkpoint evidence, and permitted use.

The plotting script writes PNG and PDF versions to `docs/figures/`.

## 10. Expected qualitative result

A correct rerun should be interpreted from the generated files, not forced to
match one seed exactly across hardware. In the completed local v9 run:

- H3 and H4 are supported for all three GAT and all three GAT-Outage seeds;
- H1 is supported for two of three GAT and all three GAT-Outage seeds;
- the paired stale-attention shift is positive for seeds 42 and 43 and negative
  for seed 44 under both training regimes;
- paired probability-DEF decreases for seeds 42 and 43 but increases for seed
  44 under both training regimes;
- eligible records are dominated by no-op decisions, and the combined H1/H4
  patterns do not reproduce in the dispatch stratum;
- the six GAT checkpoints select no-op throughout the 18 deterministic
  diagnostic episodes.
- tunnel and random triggers agree on attention-shift direction in five of six
  checkpoints and on probability-DEF direction in four of six checkpoints.

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
They should not be substituted for the v9 H1-H5 analysis.

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
