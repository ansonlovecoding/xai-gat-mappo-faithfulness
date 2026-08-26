# Dissertation v4 result summary

This directory contains the compact, citable outputs from the completed
`configs/experiments/dissertation_v4.toml` run. Raw cells and checkpoints stay
under ignored `runs/dissertation_v4/` because they are approximately 363 MB.

## Primary result

- H3 is supported for all three B2 and all three H5 training seeds.
- H1 and H2 are unsupported for all six trained GAT policies.
- H4 is unsupported for all B2 seeds and supported for two of three H5 seeds.
- The paired stale-attention shift is positive for seeds 42 and 43 and negative
  for seed 44 under both B2 and H5.
- H5 does not provide consistent mitigation; its seed-44 policy is also much
  weaker on clean test demand.

Longer observation outages therefore increase stale-data exposure, but the
direction of attention reallocation and its relationship with DEF depend on
the independently trained policy. The experiment does not show that AoI causes
lower faithfulness.

## Files

- `summary.csv/json`: condition-level descriptive results.
- `performance_context.csv/json`: clean-test policy performance.
- `training_seed_synthesis.csv/json`: cross-training-seed effect ranges and
  consistency counts. This is the first file to read for the thesis conclusion.
- `sweeps/<model>/<seed>/analysis.json`: within-policy hypothesis tests.
- `sweeps/<model>/<seed>/preflight.json`: protocol validation result.
- `sweeps/<model>/<seed>/manifest.json`: checkpoint and run provenance.
- `training/<model>/<seed>/checkpoint_selection.json`: validation-only model
  selection record.

The independently trained policy is the replication unit. With three training
seeds per model, cross-seed findings are reported descriptively; no
population-level cross-seed p-value is claimed.
