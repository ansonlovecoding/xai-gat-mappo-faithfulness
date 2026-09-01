# Exposure-conditioned dissertation audit

This directory contains compact, citable outputs from the final audit. The
trained checkpoints come from `runs/dissertation_v8/training/`; no policy was
retrained or reselected after the faithfulness results were observed.

## Protocol

- Models: GAT and GAT-Outage, each with training seeds 42, 43, and 44.
- Conditions: clean telemetry and 10, 20, 30, and 60-second observation-layer
  outages.
- Evaluation: eight seeds and six episodes per condition for every frozen
  checkpoint.
- Sampling: every 16th decision plus every stale-exposed decision.
- Primary follow-up: degraded DEF minus exact clean-twin DEF, using the same
  selected action and matched-random draw.

All six sweeps pass preflight, including stale-exposure coverage gates.

## Main result

Longer outages increase stale-data exposure in all six policies. Within-policy
tests associate greater exposure with lower DEF in most checkpoints. However,
the direct paired response is not uniform: attention shift and probability-DEF
shift have the expected directions for seeds 42 and 43, while seed 44 reverses
both directions under GAT and GAT-Outage. The effects are small.

The LOO perturbation control exceeds raw-attention DEF in all six checkpoints.
Taxi-only attention-LOO agreement is positive but varies in strength.
Layer/head aggregation can reverse the stale-attention result inside every
checkpoint, and query-row improvement is concentrated in seed 43.

The supported conclusion is that raw attention does not provide a reproducible
freshness or faithfulness guarantee across independently trained policies. The
results do not establish that AoI itself causes lower faithfulness.

## Files

- `summary.csv`: condition-level audit measurements.
- `training_seed_synthesis.csv`: cross-checkpoint consistency summary.
- `analyses/`: complete per-checkpoint statistical analyses.
- `preflight/`: validation reports for all six sweeps.
- `faithfulness_controls/`: LOO, Gradient x Input, overlap, aggregation, and
  query-row control summaries.
