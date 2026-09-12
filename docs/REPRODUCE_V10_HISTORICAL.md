# Historical v10 procedure (superseded)

These results are retained for provenance and are not used in the updated thesis.

# Reproducing the dissertation experiments

The post-review no-op analysis and prospective five-seed matched-budget protocol
are documented in `docs/MATCHED_BUDGET_FOLLOWUP.md`. They are separate from the
historical v10 reproduction below; the v11 training run has not passed the
environment gate and contributes no results to the current thesis.

This guide reproduces the final corrected experiment for *When Explanations
Outlive Their Data: Faithfulness Decoupling in Graph-Attention MARL Fleet
Dispatch under Telemetry Degradation*.

The single source of experiment settings is:

```text
configs/experiments/dissertation_v10_corrected.toml
```

Do not combine these outputs with earlier experiment directories. The corrected
protocol retrains, reselects, and reevaluates every model after fixing reward
credit, completed-journey counting, dispatch-action protection, and stored
numeric precision.

## 1. What is reproduced

The pipeline:

1. trains MLP, GAT, and GAT-Outage with seeds 42, 43, and 44;
2. selects checkpoints on validation demand, excluding epoch 0;
3. evaluates selected policies on held-out clean demand;
4. audits GAT and GAT-Outage under clean, 10, 20, 30, and 60-second
   observation-layer conditions;
5. runs preflight, hypothesis, action-type, precision, random-trigger, and
   faithfulness-control analyses;
6. applies the explanation-release rules and writes the final audit report.

Tunnel entry starts an outage, but the degradation occurs at the observation
boundary. SUMO continues to update the true vehicle state while the policy sees
the last valid position and speed. The clean twin reads the same current SUMO
state and never selects an action.

## 2. Environment

The archived sweep manifests record Python 3.11.15, macOS 15.7.9 x86_64,
SUMO 1.20.0 and the TraCI backend (not libsumo). Their dependency versions
are retained in `docs/dissertation/requirements-evaluated.txt`. The committed
network was built using SUMO tools 1.27.0; the current general requirements
also target newer bindings. Distinguish network generation from the runtime
used to obtain the reported results. For a historical reproduction, preserve
the committed network and use the manifest-recorded simulation stack. The
commands below otherwise describe a current development setup.

Hardware model, memory and complete wall-clock timings were not retained for
the reported runs. Record them during reruns rather than estimating them from
the current workstation.

```bash
cd "<repo-root>"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For Intel macOS:

```bash
python -m pip install -r requirements-macos-intel.txt
python -m pip install -e . --no-deps
```

This profile pins NumPy below 2 because the compatible PyTorch wheel was built
against NumPy 1.x.

Verify the environment before starting a long run:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/test_policy.py
.venv/bin/python scripts/test_faithfulness.py
```

TraCI may print short connection retries while SUMO starts. They are not a
failure when the command later ends with `evaluator OK`.

## 3. Fixed protocol

| Item | Value |
|---|---|
| Training seeds | 42, 43, 44 |
| Training epochs | 40 GAT; 50 MLP and GAT-Outage |
| Checkpoint selection | mean completed journeys on validation demand |
| Validation | seed 2026, eight episodes |
| Held-out evaluation seeds | 42-49 |
| Held-out episodes per seed | six |
| Faithfulness conditions | clean, 10 s, 20 s, 30 s, 60 s |
| Corruption | freeze last valid position and speed |
| Primary action rule | sample from policy distribution |
| Random baseline | five type-matched subsets with chosen-action protection |
| Main sweep size | 1,440 episode evaluations |

Internal command-line model identifiers are `B1_mlp`, `B2_gat`, and
`H5_gat_degraded`. The dissertation displays only MLP, GAT, and GAT-Outage.

GAT and GAT-Outage use different fixed maximum training budgets. Their
comparison describes the two fitted configurations; it does not isolate outage
training as the only difference.

## 4. Inspect and run

Print all generated commands without executing them:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --dry-run
```

Run the complete experiment:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage all --resume
```

The complete CPU run can take many hours. `--resume` skips complete immutable
cells and rejects incomplete training directories rather than treating them as
finished.

Individual stages are:

```text
train -> select -> evaluate -> diagnose -> sweep -> robustness -> preflight
      -> analyze -> summarize -> controls -> analyze-robustness
      -> analyze-controls -> audit
```

Examples:

```bash
# Rerun only one selected model's clean evaluation
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage evaluate --model B2_gat --resume

# Rerun the complete read-only audit from existing checkpoints
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage framework --resume

# Reapply release rules to existing evidence
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage audit
```

## 5. Evidence checks

Every sweep used in the dissertation must contain a `preflight.json` file with:

```json
{"ok": true}
```

Preflight confirms evidence completeness and protocol identity. It is not an
explanation-release decision. The final audit separately returns `ELIGIBLE`,
`WITHHOLD`, or `INCOMPLETE`.

Important outputs are:

```text
runs/dissertation_v10_corrected/
  training/<model>/seed_<training-seed>/
    ckpt_selected.pt
    checkpoint_selection.json
    training_stability.json
  evaluations/<model>/seed_<training-seed>/
  sweeps/<gat-model>/seed_<training-seed>/
    manifest.json
    cells/
    preflight.json
    analysis.json
  faithfulness_controls/
  performance_context.csv
  precision_sensitivity.csv
  action_stratified.csv
  training_seed_synthesis.csv

results/dissertation_v10_corrected/
  matched_baselines.json
  random_loss_robustness.json
  faithfulness_controls/
  explanation_audit/
    audit_report.json
    audit_checks.csv
    AUDIT_REPORT.md
```

Do not delete `cells/` from an archival reproduction package. They contain the
per-decision evidence needed to recompute the statistics.

## 6. Supporting analyses

The runner's `framework` stage executes the declared supporting analyses. They
can also be run directly when debugging:

```bash
.venv/bin/python scripts/run_matched_baselines.py \
  --config configs/experiments/dissertation_v10_corrected.toml

.venv/bin/python scripts/analyze_random_loss_robustness.py \
  runs/dissertation_v10_corrected/robustness/random_loss_30s \
  --out results/dissertation_v10_corrected/random_loss_robustness

.venv/bin/python scripts/analyze_faithfulness_controls.py \
  runs/dissertation_v10_corrected/faithfulness_controls/B2_full \
  runs/dissertation_v10_corrected/faithfulness_controls/H5_full \
  runs/dissertation_v10_corrected/faithfulness_controls/B2_action_row \
  runs/dissertation_v10_corrected/faithfulness_controls/H5_action_row \
  --out results/dissertation_v10_corrected/faithfulness_controls \
  --fig-dir docs/figures --fig-prefix v10
```

The random trigger uses a fixed probability of 0.0023 per taxi-step. Its
realized exposure is lower than tunnel exposure, so it is a trigger-location
sensitivity check rather than an exposure-matched causal comparison.

## 7. Expected qualitative result

A correct rerun should be interpreted from its generated files rather than
forced to reproduce every floating-point value exactly. The completed local
run has the following stable pattern:

- H3 is supported for all six GAT checkpoints;
- H1, H2, and H4 are supported for none of the six checkpoints;
- stale-node attention shifts toward stale nodes for four checkpoints and away
  for two;
- paired probability DEF increases slightly for all six checkpoints;
- no-op DEF decreases while dispatch DEF increases in all six checkpoints;
- raw-attention margin-DEF is below zero, while LOO is above zero;
- alternative heads or layers can reverse the stale-attention direction;
- the final release decision is `WITHHOLD` for both GAT model families.

These results do not show that AoI improves or reduces faithfulness. They show
that freshness, decision relevance, and explanation stability must be checked
separately.

## 8. Rebuild figures and release package

```bash
.venv/bin/python scripts/plot_dissertation.py \
  --root runs/dissertation_v10_corrected \
  --training-root runs/dissertation_v10_corrected \
  --analysis-root results/dissertation_v10_corrected \
  --out docs/figures --prefix v10

.venv/bin/python scripts/package_dissertation_release.py \
  --config configs/experiments/dissertation_v10_corrected.toml
```

The package command copies the final configuration, six audited checkpoints,
raw sweep cells, analyses, audit reports, documentation, and a SHA-256
manifest. Verify the generated manifest before submission.

## 9. Troubleshooting

### PyTorch reports that NumPy is unavailable

```bash
.venv/bin/python -m pip install --force-reinstall "numpy<2"
```

### TraCI retries connection

Allow the command to finish. Investigate only if it exits non-zero or never
reaches an `OK` or completion message.

### Preflight fails

Read the `errors` array in `preflight.json`. Do not quote or analyze that sweep
until the missing cell or protocol mismatch is corrected.

### Results differ slightly across systems

Retain all declared training seeds. Compare directions and audit decisions
rather than selecting the run that most closely matches one machine.


## 7. Dissertation EDA and document regeneration

Run `python scripts/plot_dataset_eda.py` to recompute the descriptive demand
figures and `docs/figures/dataset_eda_summary.json` from committed XML inputs.
This command does not train policies or replace experimental results.

Build the Word document with `python scripts/build_dmu_thesis_docx.py`.
After rendering it to PDF, run `python scripts/update_thesis_page_map.py
<rendered.pdf>` and rebuild/render again until the page-map command reports
`changed: false`. This refreshes the contents and figure/table lists from
actual pagination. Inspect the final rendered pages before distributing both
formats. Document-generation packages are separate from the evaluated
simulation environment.
