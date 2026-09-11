# Matched-budget follow-up and supplementary no-op analysis

## Completed supplementary analysis

`scripts/analyze_noop_absolute_def.py` reads the six immutable archived sweep
files without extracting or changing them. It reports absolute no-op probability
and margin DEF for clean and 60-second conditions, excludes forced no-op actions,
and resamples equal-weight episode means within each checkpoint. The analysis
is exploratory and its pointwise intervals are not adjusted for multiplicity.
Results and input hashes are in
`results/dissertation_v10_corrected/supplementary_review/`. Section 4.12 of the
thesis reports the findings. The original release decisions remain unchanged.

```bash
.venv/bin/python scripts/analyze_noop_absolute_def.py
```

## Prospective training protocol

`configs/experiments/dissertation_v11_matched_budget.toml` specifies GAT and
GAT-Outage, each with seeds 42, 43, 44, 45 and 46, 50 training epochs, the same
learning rate and decay horizon, PPO settings, validation selection and test
conditions. Only the training observation degradation differs between the two
model definitions. The MLP is not part of this matched treatment comparison.
The ten new runs must be analyzed as a separate experiment; do not combine them
with v10 checkpoints or extend only the favorable old runs.

The common budget is simulation epochs, not guaranteed equal collected decision
counts or PPO updates. Report those realized counts from the training logs.
Keep every seed's outcome, including stability-gate failures. Do not silently
replace a failed seed or tune only one model on test results. H5 remains a
description of association strength; consistency requires per-seed directions,
effects and uncertainty, not simply smaller absolute correlations.

```bash
.venv/bin/python scripts/run_matched_followup.py --dry-run
.venv/bin/python scripts/run_matched_followup.py --stage all
```

The wrapper records current CPU model, logical CPU count, installed memory,
configuration hash, completion status and total invocation wall time under the
new run directory. Completed training writes a separate `timing.json` including
initialization and checkpoint saves. Historical hardware fields stay unrecorded.
Validation, evaluation and other stages can be invoked separately for stage
timing. Interrupted training directories remain immutable and must not be
mistaken for complete runs.

## Execution status

Command generation was checked, and training launch was attempted on 12 September
2026. The project's environment gate stopped execution before training. The
current environment reports Python 3.11.14, NumPy 2.4.6, Torch 2.12.1, TraCI
1.27.1 and an unrecognized SUMO binary version; it does not pass the existing
pinned-runtime checks. The working tree also contains thesis and implementation
changes that have not been frozen in a clean experiment revision.

No v11 trained checkpoint or result exists, and no v11 result is claimed in the
thesis. Establish and validate the experiment runtime, freeze the implementation
in a clean revision, then run the commands above. This execution gate has not
been bypassed. The v10 results and original archived records remain intact.
