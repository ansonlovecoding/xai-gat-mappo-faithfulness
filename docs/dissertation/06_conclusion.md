# 6. Conclusion

This dissertation asked whether raw graph-attention weights can be trusted as
explanations of fleet-dispatch decisions when vehicle telemetry becomes stale.
The tested weights should not be released as explanations. They remain
available under clean and degraded observations, but they do not pass the
required checks for decision relevance, extraction stability, and consistency
across independently trained checkpoints.

The corrected experiment does not show that telemetry degradation reduces
DEF. Longer outages consistently increase stale-data exposure, confirming that
the manipulation worked, but H1, H2, and H4 are not supported. Paired DEF
increases slightly for all six checkpoints. This opposite-direction result
must not be interpreted as improved explanation quality because raw attention
already performs below its type-matched random control under clean telemetry.
Telemetry freshness and explanation faithfulness are separate assurance
questions.

The research questions are answered as follows.

1. **RQ1:** Raw attention does not reliably identify decision-relevant nodes
   under clean telemetry. Its action-aware margin-DEF is negative for both
   models, while LOO gives a positive control response.
2. **RQ2:** Degradation changes the attention assigned to stale nodes, but the
   direction is not consistent. Four checkpoints shift toward stale nodes and
   two shift away.
3. **RQ3:** Outage duration consistently increases WAMSN, but it does not
   produce the proposed decline in DEF. The measured response also differs
   between no-op and dispatch actions and can reverse under alternative
   attention-extraction choices.
4. **RQ4:** The tested GAT-Outage configuration does not make raw attention
   reliably decision-relevant or remove checkpoint variation. Because GAT and
   GAT-Outage use different maximum training budgets, this comparison does not
   isolate outage training as the cause.

The construct-validity audit is essential to these answers. A passenger-request
node represents both information and an available action, whereas a peer-taxi
node represents information only. Type matching and chosen-action protection
prevent request deletion from creating an artificial faithfulness result. The
positive LOO response shows that the evaluator can reward a ranking based on
output sensitivity. The top-k overlap analysis also shows that the small graph
limits measurement resolution.

The main practical contribution is the freshness-aware explanation audit
framework. It accepts frozen checkpoints and held-out evidence, checks
freshness, decision relevance, action composition, extraction stability,
checkpoint consistency, action-rule capability, and trigger sensitivity, and
returns `ELIGIBLE`, `WITHHOLD`, or `INCOMPLETE`. In this study, both model
families and all six checkpoints receive `WITHHOLD`. The evidence is complete,
but it does not justify presenting attention as the reason for a dispatch or
no-op decision.

This outcome completes the audit rather than failing it. A trustworthy
explanation system must be able to reject a visually credible explanation when
its evidence is weak or unstable. Future work should evaluate more training
seeds and road networks, calibrate random loss to match tunnel exposure, and
train a model with an explicit explanation objective. Any future `ELIGIBLE`
result should remain limited to its tested scope and should display telemetry
freshness separately.

In summary, attention can outlive the data on which it was calculated, but the
continued presence of an attention map is not proof that it explains the
decision. The proposed framework turns that distinction into a reproducible
release decision.
