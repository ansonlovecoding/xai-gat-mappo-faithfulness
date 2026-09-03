# 6. Conclusion

This dissertation examined a practical assurance problem: a graph-attention
map can remain visible and appear complete after some vehicle telemetry has
become stale. Attention strength alone does not tell an operator whether its
source is current or whether the highlighted node affected the action.

The results lead to a clear conclusion: **this experiment does not validate raw
graph-attention weights as reliable explanations under clean or degraded
telemetry**. This is not a claim that every attention map is wrong. It means
that attention cannot be assumed trustworthy by default.

The four research questions are answered as follows:

- **RQ1: Under clean telemetry, do raw graph-attention weights identify
  decision-relevant nodes more reliably than type-matched random controls?**
  Not consistently. Raw-attention DEF ranges from -0.0009 to +0.0031 across
  the six checkpoints.
  LOO produces a larger positive result for all six checkpoints, while taxi-only
  attention-LOO rank correlation ranges from +0.088 to +0.699.
- **RQ2: Does tunnel-triggered telemetry degradation change the attention
  assigned to stale vehicle nodes relative to the paired clean observation?**
  Yes, but not in a consistent direction. Attention shifts toward stale nodes
  for checkpoints trained with seeds 42 and 43 and away from them for seed 44
  under both training regimes. The same direction appears under random
  triggering in five of six checkpoints, while individual heads can still
  reverse it within one checkpoint.
- **RQ3: As outage duration increases, are changes in stale-node attention and
  decision-level faithfulness consistent across independently trained
  checkpoints?** No. Paired stale-node attention changes direction with the
  trained checkpoint. H1 is supported in five of six selected checkpoints and H4 in all six, but the
  combined direct paired DEF shift is negative for four checkpoints and positive
  for two. The effect is small and reverses for the checkpoints trained with
  seed 44. Moreover, 93.0% of scorable decisions with at least one available
  request are no-op, and the H1/H4 patterns are not reproduced in the dispatch
  stratum.
- **RQ4: Does degradation-aware training produce more consistent attention and
  faithfulness responses under telemetry degradation than clean training?** No.
  GAT-Outage shows the same seed-dependent paired direction as GAT and provides
  no consistent improvement.

This interpretation depends on the construct-validity audit. Deleting a
request can also delete an action, while deleting a peer taxi only hides
context. Type-matched and action-protected controls remove much of this
artifact. The positive LOO result shows that DEF has some perturbation
sensitivity, but the high expected top-k overlap limits its resolution. The
conclusion draws on several checks rather than one near-zero number.

The 30-second random-trigger sensitivity analysis reduces concern that the
checkpoint-dependent pattern is produced only by the selected tunnel. It does
not remove the need for multiple tunnel placements or real telemetry traces,
because realized exposure and the stale-node population still differ between
trigger mechanisms.

The performance results also have an important limit. Sampled policies
outperform the basic lower bounds, but all six graph-attention checkpoints choose no-op in 18 of 18
deterministic diagnostic episodes. The study audits explanations for sampled
stochastic actions; it does not demonstrate a deployable argmax dispatcher.

In this thesis, **faithfulness decoupling** refers to the gap between a
visible explanation and valid supporting evidence. An attention map can
outlive the freshness of its inputs without giving a stable indication of
decision relevance. The gap does not require DEF to decline monotonically with
outage duration.

The freshness-aware explanation audit framework addresses this gap. It reports
freshness and action composition separately, uses
action-aware and type-matched tests, checks the attention extraction rule, audits every candidate
checkpoint, and requires consistency across independent training runs. Stable
dispatch output is not an explanation test.

Its demonstrative application gives `WITHHOLD` for GAT, GAT-Outage, and all six
frozen checkpoints. This means that the audit completed but the attention maps
did not meet the release conditions. An `ELIGIBLE` model is not required to
support the present finding: correctly withholding unsupported explanations is
the purpose of the framework. However, the study validates only the rejection
path on trained models. A future model with an explicit explanation objective is
needed to test the eligibility path on held-out evidence.

An `ELIGIBLE` result would permit attention to be shown only as an audited
candidate explanation within the stated scope, with freshness displayed
separately. It would not prove a complete causal explanation.

The framework is designed to reduce operational risk, not to remove
faithfulness decoupling from the model. Attention remains an internal diagnostic
unless the full audit provides consistent evidence.

All six objectives were completed: the paired benchmark and audit pipeline were
implemented, the four research questions were evaluated, and the collected
evidence produced a `WITHHOLD` decision for every tested checkpoint. Completion
refers to execution of the planned study, not support for every hypothesis or
achievement of an `ELIGIBLE` result.
