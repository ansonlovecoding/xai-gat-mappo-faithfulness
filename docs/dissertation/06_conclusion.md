# 6. Conclusion

This dissertation examined a practical assurance problem: a graph-attention
map can remain visible and appear complete after some vehicle telemetry has
become stale. Attention strength alone does not tell an operator whether its
source is current or whether the highlighted node affected the action.

The central answer is clear: **this experiment does not validate raw
graph-attention weights as dependable explanations under clean or degraded
telemetry**. This is not a claim that every attention map is wrong. It means
that attention cannot be assumed trustworthy by default.

The research questions lead to that answer:

- **RQ1: Is attention faithful under clean telemetry?** No stable result is
  found. Raw-attention DEF ranges from -0.0009 to +0.0031 across checkpoints.
  LOO gives a larger positive result for all six checkpoints, while taxi-only
  attention-LOO rank correlation ranges from +0.088 to +0.699.
- **RQ2: Does attention move toward stale vehicle nodes?** Not consistently.
  The declared aggregation moves toward stale nodes for checkpoints trained
  with seeds 42 and 43 and away from them for checkpoints trained with seed 44
  under both training regimes. Individual heads can reverse the direction
  within one checkpoint.
- **RQ3: Does the explanation remain dependable as outage duration increases?**
  No cross-policy guarantee is found. H1 is supported in five of six selected
  checkpoints and H4 in all six, but the direct paired DEF shift is negative
  for four checkpoints and positive for two. The effect is small and reverses
  for the checkpoints trained with seed 44. Moreover, 93.0% of eligible
  decisions are no-op, and the H1/H4 patterns are not reproduced in the
  dispatch stratum.
- **RQ4: Does degradation-aware training improve explanation reliability?** No
  consistent mitigation is observed. GAT-Outage retains the same seed-dependent
  paired direction as GAT.

The construct-validity audit is essential to this interpretation. Deleting a
request can also delete an action, while deleting a peer taxi only hides
context. Type-matched and action-protected controls remove much of this
artifact. The positive LOO result shows that DEF has some perturbation
sensitivity, but the high expected top-k overlap limits its resolution. The
conclusion is therefore based on several checks rather than one near-zero
number.

The capability evidence also has a clear boundary. Sampled policies outperform
the basic lower bounds, but all six GAT checkpoints choose no-op in 18 of 18
deterministic diagnostic episodes. The study audits explanations for sampled
stochastic actions; it does not demonstrate a deployable argmax dispatcher.

In this thesis, **faithfulness decoupling** is the assurance gap between a
visible explanation and valid supporting evidence. An attention map can
outlive the freshness of its inputs without giving a stable indication of
decision relevance. The gap does not require DEF to decline monotonically with
outage duration.

The thesis addresses this gap with a freshness-aware explanation assurance
approach. It separates freshness and action composition from attention, uses
action-aware and type-matched tests, checks the attention extraction rule, audits every candidate
checkpoint, and requires consistency across independent training runs. Stable
dispatch output is not an explanation test.

The approach controls operational risk; it does not claim to remove
faithfulness decoupling from the model. Attention remains an internal diagnostic
unless the full assurance process provides consistent evidence.
