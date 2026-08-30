# 6. Conclusion

This dissertation examined a practical assurance problem: a graph-attention
map can remain visible and appear complete after some vehicle telemetry has
become stale. Attention strength alone does not tell an operator whether its
source is current or whether the highlighted node affected the action.

The central answer is clear: **this experiment does not validate raw
graph-attention weights as dependable explanations under clean or degraded
telemetry**. This is not a claim that every attention map is wrong. It means
that attention cannot be trusted by default.

The research questions lead to that answer:

- **RQ1: Is attention faithful under clean telemetry?** No stable result is
  found. Raw-attention DEF ranges from -0.0084 to +0.0187 across checkpoints.
  LOO gives a larger positive result for all six checkpoints, while taxi-only
  attention-LOO rank correlation ranges from -0.710 to +0.319.
- **RQ2: Does attention move toward stale vehicle nodes?** Not consistently.
  The declared aggregation moves toward stale nodes for seeds 42 and 43 and
  away from them for seed 44 under both training regimes. Individual heads can
  reverse the direction within one checkpoint.
- **RQ3: Does the explanation remain dependable as outage duration increases?**
  No dependable relationship is found. Stale exposure increases, but DEF does
  not consistently decline. The 60-second ranking-control results remain close
  to their clean values.
- **RQ4: Does degradation-aware training improve explanation reliability?** No
  consistent mitigation is observed. GAT-Outage retains checkpoint, aggregation, and
  query-row dependence and includes one weak policy replicate.

The construct-validity audit is essential to this interpretation. Deleting a
request can also delete an action, while deleting a peer taxi only hides
context. Type-matched and action-protected controls remove much of this
artefact. The positive LOO result shows that DEF has some perturbation
sensitivity, but the high expected top-k overlap limits its resolution. The
conclusion is therefore based on several checks rather than one near-zero
number.

In this thesis, **faithfulness decoupling** is the assurance gap between a
visible explanation and valid supporting evidence. An attention map can
outlive the freshness of its inputs without giving a stable indication of
decision relevance. The gap does not require DEF to decline monotonically with
outage duration.

The practical resolution is to separate freshness from explanation. Stable
dispatch output is not evidence that an explanation remains trustworthy.
Attention weights should be treated as model internals until each released
checkpoint passes freshness-aware, action-aware, and aggregation-sensitive
faithfulness tests, and the result reproduces across independent training
runs.
