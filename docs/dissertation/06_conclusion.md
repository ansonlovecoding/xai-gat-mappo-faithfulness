# 6. Conclusion

This dissertation addressed one practical problem: a graph-attention map can
remain visible and look complete after some of the vehicle data behind it have
become stale. An operator may mistake that map for a trustworthy explanation,
although attention strength does not show whether the data are fresh or the
highlighted nodes support the decision.

The central answer is clear: **this experiment does not validate raw
graph-attention weights as dependable explanations under telemetry
degradation**. This does not show that every attention map is wrong; it shows
that attention cannot be trusted by default.

The four research questions provide the evidence for this answer:

- **RQ1 - Is graph attention faithful under clean telemetry?** It is not validated as faithful. Corrected DEF remains close to the type-matched random baseline, so the weights show no measured advantage over a fair random ranking.
- **RQ2 - Does attention move toward stale vehicle nodes?** Not consistently. Longer outages increase stale-data exposure in all six GAT policies, but the paired attention shift is positive for seeds 42 and 43 and negative for seed 44 in both training regimes. Equivalent training runs can therefore produce opposite explanation responses.
- **RQ3 - Does faithfulness change more than dispatch behaviour?** No. H1 and H2 are unsupported in every trained policy. Type-matched DEF and pickups both remain nearly flat across outage durations, so stable performance cannot be interpreted as evidence of a faithful explanation.
- **RQ4 - Does degradation-aware training make the explanation more reliable?** No consistent mitigation is observed. It retains the same seed-dependent attention-shift pattern, gives mixed WAMSN-DEF relationships, and includes one substantially weaker policy replicate.

The mixed attention shifts are not an absence of a conclusion. They show that
raw attention has no reproducible, model-independent response to stale
telemetry. The thesis therefore does not claim that stale telemetry always
moves attention toward stale nodes or always lowers faithfulness. Its consistent
result is the lack of a dependable explanation guarantee across independently
trained policies.

In this thesis, **faithfulness decoupling** is this assurance gap: the
explanation remains visible after the freshness of its data has expired, but
the displayed weights do not reliably establish decision relevance. This does
not require DEF to decline monotonically with outage duration.

The construct-validity audit supports this interpretation. Deleting a request
can also delete an action, whereas deleting a taxi only hides information.
Type-matched and action-protected controls are therefore required before DEF
can be interpreted.

The practical resolution is simple. Stable dispatch output is not evidence
that an explanation remains trustworthy. Graph-attention weights should be
treated as model internals, not operator-facing reasons, until each released
checkpoint passes separate freshness-aware and action-aware faithfulness tests
and the result reproduces across independent training runs.
