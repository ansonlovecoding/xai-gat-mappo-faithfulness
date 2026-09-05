# 6. Conclusion

This dissertation examined whether graph-attention weights can be trusted as
explanations of fleet-dispatch decisions after their supporting vehicle
telemetry becomes stale. The answer is that the attention map remains
available, but the experiment does not validate its raw weights as reliable
explanations under either clean or degraded telemetry. Attention strength alone
does not show whether its source is current or whether the highlighted node
influenced the selected action.

This is the faithfulness decoupling described in the title. During an
observation-layer outage, the policy continues to produce an action and an
attention map even though some vehicle features are frozen. The study does not
claim that degradation always lowers faithfulness. Instead, it shows that the
continued availability of an explanation is not matched by consistent evidence
that the explanation is current and decision-relevant.

The four research questions are answered as follows:

- **RQ1: Under clean telemetry, do raw graph-attention weights identify
  decision-relevant nodes more reliably than type-matched random controls?**
  Not consistently. Raw-attention DEF remains close to the matched-random
  boundary and varies across the six checkpoints. LOO produces a larger
  positive response for every checkpoint, showing that the evaluator can reward
  a more informative perturbation ranking. However, LOO is a positive control,
  not a ground-truth explanation.
- **RQ2: Does tunnel-triggered telemetry degradation change the attention
  assigned to stale vehicle nodes relative to the paired clean observation?**
  Small changes are observed, but their direction is not consistent across
  independently trained checkpoints. Attention shifts toward stale nodes for
  checkpoints trained with seeds 42 and 43 and away from them for seed 44 in
  both training conditions. Alternative attention heads can also reverse the
  direction within one checkpoint.
- **RQ3: As outage duration increases, are changes in stale-node attention and
  decision-level faithfulness consistent across independently trained
  checkpoints?** No. Longer outages consistently increase stale exposure, and
  greater exposure is associated with lower DEF within the tested policies.
  However, the direct paired DEF change is small and changes direction across
  checkpoints. In addition, 93.0% of scorable decisions with an available
  request are no-op, and the main associations are not reproduced in the
  smaller dispatch group.
- **RQ4: In the tested configurations, does degradation-aware training produce
  more consistent attention and faithfulness responses under telemetry
  degradation than clean training?** No. GAT-Outage shows the same
  seed-dependent direction as GAT and gives no consistent improvement. Because
  the two configurations use different training budgets, this comparison does
  not isolate degraded training observations as the cause.

These answers depend on a valid faithfulness comparison. A passenger-request
node represents both information and an available action, so deleting it can
also delete the action being explained. Deleting a peer-taxi node only hides
context. Type-matched and action-protected controls reduce this measurement
artifact. The positive LOO response confirms that DEF has some perturbation
sensitivity, while the overlap analysis shows that the small graph limits its
resolution. The conclusion therefore rests on several checks rather than one
near-zero DEF value.

The random-trigger analysis also reduces concern that the checkpoint-dependent
pattern is produced only by the selected tunnel. It does not establish a
location-independent effect because the realized exposure rates and stale-node
populations differ between trigger mechanisms. Multiple tunnel placements,
road networks, and real telemetry traces are still needed.

The sampled policies outperform the basic lower bounds, but all six
graph-attention checkpoints choose no-op in the deterministic diagnostic. The
study therefore establishes task capability only for actions sampled from the
policy distribution; it does not demonstrate a deployable argmax dispatcher.
This limitation does not change the explanation result, but it reinforces the
need to audit the action rule intended for deployment.

The main practical contribution is the freshness-aware explanation audit
framework. It reports freshness and action composition separately, uses
action-aware faithfulness tests, checks the attention extraction rule, audits
each candidate checkpoint, and requires agreement across independent training
runs. Its application returns `WITHHOLD` for GAT, GAT-Outage, and all six frozen
checkpoints. The experiment is therefore complete even though no model reaches
`ELIGIBLE`: correctly withholding an unsupported explanation is the purpose of
the framework.

The study validates the framework's rejection path on trained models, not its
eligibility path. Future work should test a model trained with an explicit
explanation objective on held-out evidence. Even then, an `ELIGIBLE` result
would permit attention to be presented only as an audited candidate explanation
within the stated scope, with freshness displayed separately. It would not
prove a complete causal explanation.

In summary, an attention explanation can remain visible after part of its
supporting observation has become stale. Because the displayed weights do not
provide consistent evidence of freshness or decision relevance, they should
remain an internal diagnostic unless a separate audit supports their release.
The contribution of this dissertation is an evidence-based framework for making
that decision.
