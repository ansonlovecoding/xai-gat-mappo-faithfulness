# Freshness-Aware Explanation Audit Framework

## 1. Purpose

The framework decides whether graph-attention weights have enough evidence to
be presented as an explanation of a fleet-dispatch decision. It addresses two
separate questions:

1. Is the information behind the attention map current?
2. Does the attention ranking identify information that matters to the chosen
   action?

Good dispatch performance does not answer either question. The framework is an
offline release check for frozen policy checkpoints. It does not retrain the
policy and does not make attention more faithful.

## 2. When to use it

Run the audit:

- before showing a new model's attention maps to an operator;
- after retraining or replacing a checkpoint;
- after changing the telemetry, observation, or graph-building pipeline;
- after moving the model to a different operating scenario; and
- during periodic model review.

The implementation evaluates held-out experiments. It is not an online monitor.
If a model becomes eligible, the deployed interface must still display telemetry
freshness separately from attention strength.

## 3. Inputs

The command reads evidence that is already produced by the experiment pipeline.

| Input | Purpose | Default location |
|---|---|---|
| Sweep preflight reports | Confirm complete cells, provenance, exposure, and valid controls | `runs/dissertation_v9_exposure_audit/sweeps/*/seed_*/preflight.json` |
| Condition summary | Report stale exposure and stale-attention measurements | `runs/dissertation_v9_exposure_audit/summary.json` |
| Action strata | Separate no-op from dispatch decisions | `runs/dissertation_v9_exposure_audit/action_stratified.json` |
| Seed synthesis | Treat independently trained policies as replication units | `runs/dissertation_v9_exposure_audit/training_seed_synthesis.json` |
| Deterministic diagnostic | Check the action rule intended for deployment | `runs/dissertation_v9_exposure_audit/deterministic_diagnostics.json` |
| Faithfulness controls | Supply type-matched DEF, LOO, query-row, and extraction-sensitivity evidence | `results/dissertation_v9_exposure_audit/faithfulness_controls/summary.json` |
| Trigger robustness | Compare tunnel and random telemetry-loss triggers | `results/dissertation_v9_exposure_audit/random_loss_robustness.json` |

The audited model list and training seeds come from
`configs/experiments/dissertation_v9_exposure_audit.toml`.

## 4. Checks

The framework applies the checks below to every model family and exposes the
available evidence for every frozen checkpoint.

| Check | Question answered | Passing rule |
|---|---|---|
| Evidence integrity | Is the experiment valid to analyze? | Every checkpoint sweep passes preflight |
| Freshness visibility | Is stale-data exposure reported independently of attention? | Stale-exposed conditions contain AoI-derived and stale-attention measurements |
| Action-aware controls | Does the comparison avoid request-deletion bias? | Type matching, chosen-action protection, raw attention, and LOO are present |
| Decision relevance | Does attention beat a matched random ranking for dispatch actions? | The 95% CI lower bound of self-row dispatch margin-DEF exceeds the configured floor |
| Action composition | Are no-op and dispatch decisions reported separately? | Both strata exist and the minimum dispatch count is met |
| Extraction stability | Does the conclusion survive layer/head/rollout choices? | No audited alternative reverses the declared default direction |
| Checkpoint consistency | Does the conclusion reproduce after independent training? | Dispatch relevance and stale-attention response have the same non-zero direction across seeds |
| Deployment-action capability | Does the action rule intended for deployment operate? | Require held-out argmax capability only when deployment uses argmax; otherwise retain argmax as a diagnostic |
| Trigger robustness | Is the result limited to a fixed tunnel trigger? | A direction is supported only when its 95% CI excludes zero; supported tunnel and random directions must agree |

The declared attention extraction rule remains the self-node query row, averaged
over layers and heads. Alternatives are sensitivity tests, not opportunities to
select a favorable result after seeing the data.

## 5. Decision states

| State | Meaning | Allowed use |
|---|---|---|
| `ELIGIBLE` | Every required check passed within the recorded scope | Attention may be shown as an audited candidate explanation with freshness and scope information |
| `WITHHOLD` | At least one required check failed | Keep attention as an internal model diagnostic |
| `INCOMPLETE` | Required evidence is missing or unreadable | Collect the missing evidence; no release decision is possible |

`ELIGIBLE` does not mean that attention is a complete causal explanation. It
means only that the predeclared release checks passed for the audited models,
data, actions, and telemetry conditions.

An individual check can be `INDETERMINATE` when its evidence is complete but a
direction is not supported, for example when a trigger-comparison confidence
interval crosses zero. An indeterminate required check produces `WITHHOLD`, not
`INCOMPLETE`.

## 6. Run the audit

To execute the complete framework from the selected frozen checkpoints, run:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage framework --resume
```

This performs the deterministic diagnostic, clean/degraded sweep, random-trigger
evaluation, preflight, statistical analysis, summaries, faithfulness controls,
and final explanation audit in the required order. It does not retrain or
reselect the policies.

If all evidence has already been generated, run only the final decision stage:

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml \
  --stage audit
```

The direct final-decision command is equivalent:

```bash
.venv/bin/python scripts/audit_explanations.py \
  --config configs/experiments/dissertation_v9_exposure_audit.toml
```

For a deployment pipeline that must stop when attention is not eligible, add:

```bash
--fail-on-withhold
```

This optional flag returns a non-zero process status for both `WITHHOLD` and
`INCOMPLETE`. Without it, the command returns success when the report was
generated, regardless of the audit decision.

## 7. Outputs

The default output directory is
`results/dissertation_v9_exposure_audit/explanation_audit/`.

| File | Audience | Contents |
|---|---|---|
| `audit_report.json` | Software and later analysis | Rules, inputs, model decisions, checkpoint gates, evidence, and source paths |
| `audit_checks.csv` | Tables and comparison | One row per model and check |
| `AUDIT_REPORT.md` | Readers and reviewers | Human-readable decision, evidence table, checkpoint table, and operating instructions |

The JSON output records the exact rules used. The default rules are declared in
the `[explanation_audit]` section of the experiment configuration, including the
decision-relevance floor and optional deployment-action and trigger gates.

## 8. Interpreting the current experiment

The completed dissertation experiment produces `WITHHOLD` for both GAT and
GAT-Outage. Evidence integrity, freshness reporting, action-aware controls, and
action composition pass. The release decision fails because dispatch-action
decision relevance is not supported, the attention conclusion changes under
alternative extraction rules, and the stale-attention response is inconsistent
across checkpoints. Trigger robustness is indeterminate for both model families
because at least one confidence interval crosses zero. The held-out argmax
diagnostic produces no pickups, but it is not a release gate for the sampled
action rule audited in this experiment.

This result is the intended use of the framework: a valid and reproducible
experiment can still conclude that an attention map should not be presented as a
trustworthy explanation.

## 9. Scope and limitations

The current implementation audits a family of frozen checkpoints using the
dissertation's SUMO evidence schema. Applying the framework to another dataset
requires producing the same semantic inputs: freshness labels, action-aware
counterfactual evidence, action composition, extraction sensitivity, independent
training runs, and an evaluation of the intended deployment action rule.

The thresholds are operational release rules, not new hypothesis tests. They
must be declared before evaluating a new model and must not be adjusted to obtain
an `ELIGIBLE` result.
