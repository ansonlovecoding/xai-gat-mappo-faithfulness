# 6. Conclusion

This dissertation asked whether raw graph-attention weights can be trusted as
explanations of fleet-dispatch decisions when vehicle telemetry becomes stale.
The tested weights should not be released as explanations. They remain
available under clean and degraded observations, but they do not pass the
required checks for decision relevance, extraction stability, and consistency
across independently trained checkpoints. This conclusion concerns the
declared averaged self-row channel, not all attention-based explanation methods.

The corrected experiment does not show that telemetry degradation reduces
DEF. Longer outages consistently increase stale-data exposure, confirming that
the manipulation worked, but H1, H2, and H4 are not supported. Paired DEF
increases slightly for all six checkpoints. This increase should not be interpreted as improved explanation quality because raw attention
has dispatch margin-DEF below its type-matched random control under clean telemetry.
Telemetry freshness and explanation faithfulness therefore need separate
checks.

The research questions are answered as follows.

1. **RQ1:** The tested averaged self-row attention does not establish reliable
   dispatch decision relevance under clean telemetry. Dispatch margin-DEF is
   below the matched-random control for both models, while LOO gives a positive
   control response. Request-row sensitivity does not establish a consistent
   advantage, and supplementary no-op results vary by checkpoint and condition.
   These findings do not exclude other attention-based explanation methods.
2. **RQ2:** Degradation changes the attention assigned to stale nodes, but the
   direction is not consistent. Three checkpoints show a clear shift toward
   stale nodes, two show a clear shift away, and one is inconclusive.
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
no-op decision under the declared whole-channel release rule. This conservative
rule does not assert that every individual no-op explanation is unfaithful.

The study is limited to one simulated district, three training seeds
per model, three held-out demand variants, different clean/outage training
budgets and a small action-linked graph. The experiment demonstrates rejection
of the tested raw-attention channel; it does not yet demonstrate a trained
model that meets all the release requirements. It also does not measure human
understanding or generalization to real fleet telemetry.

Future work should first compare clean and outage training under the same
training budgets with more independent seeds, then extend evaluation to
other road networks and calibrated telemetry-loss conditions. Explanation-aware
training and alternative graph attribution methods offer further candidates
for evaluation. These experiments should retain fixed release criteria, report
both passing and failing checkpoints, and record hardware and timing details.

Any future release decision should remain tied to its evaluated conditions,
with telemetry freshness shown separately.
