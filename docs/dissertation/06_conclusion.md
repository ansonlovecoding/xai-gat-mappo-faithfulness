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

- **RQ1: Is attention faithful under clean telemetry?** The six checkpoints do
  not show a consistent result. Raw-attention DEF ranges from -0.0009 to
  +0.0031 across checkpoints.
  LOO produces a larger positive result for all six checkpoints, while taxi-only
  attention-LOO rank correlation ranges from +0.088 to +0.699.
- **RQ2: Does attention move toward stale vehicle nodes?** Not consistently.
  Under the predefined aggregation method, attention shifts toward stale nodes
  for checkpoints trained with seeds 42 and 43 and away from them for seed 44
  under both training regimes. The same attention-shift direction appears under
  random triggering in five of six checkpoints, while individual heads
  can still reverse the direction within one checkpoint.
- **RQ3: Does the explanation remain reliable as outage duration increases?**
  The policies show no consistent relationship. H1 is supported in five of six
  selected checkpoints and H4 in all six, but the direct paired DEF shift is negative
  for four checkpoints and positive for two. The effect is small and reverses
  for the checkpoints trained with seed 44. Moreover, 93.0% of eligible
  decisions are no-op, and the H1/H4 patterns are not reproduced in the
  dispatch stratum.
- **RQ4: Does degradation-aware training improve explanation reliability?**
  Outage training provides no consistent improvement. GAT-Outage shows the same
  seed-dependent paired direction as GAT.

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
outperform the basic lower bounds, but all six GAT checkpoints choose no-op in 18 of 18
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

The framework is designed to reduce operational risk, not to remove
faithfulness decoupling from the model. Attention remains an internal diagnostic
unless the full audit provides consistent evidence.
