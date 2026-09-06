# 5. Discussion

## 5.1 Answer to the central problem

The central problem is whether a graph-attention map can still be trusted when
some of its vehicle data are stale. The answer is clear: the map remains
available, but its displayed weights do not provide consistent evidence that
the highlighted information is current or that it influenced the selected
action. The explanation can therefore outlive the data that support it.

This finding is about explanation reliability, not dispatch performance. A
policy can still select an action while the evidence shown as its explanation
is stale or weakly related to that action. Stable task output therefore cannot
be used to validate the attention map.

The practical result is a release decision. Raw attention should remain an
internal diagnostic unless separate tests establish both freshness and
decision relevance. In this study, those tests do not provide sufficiently
consistent evidence, so every tested checkpoint receives `WITHHOLD`.

## 5.2 How to read the evidence

The experiment provides four related but different forms of evidence.

**Clean baseline.** Under clean telemetry, raw-attention margin-DEF is below
its type-matched random control for both models, while the LOO positive control
is clearly above zero. The evaluator can therefore reward a ranking built from
output changes, but raw attention does not provide the required evidence of
decision relevance.

**Stale-data exposure.** WAMSN rises as the outage becomes longer. This shows
that more displayed attention is attached to old vehicle information. It does
not show that data age causes faithfulness to decline because AoI is part of the
WAMSN definition.

**Duration and within-policy association.** The proposed negative relationships
are not supported. DEF does not decline as outage duration or WAMSN increases.
These null and opposite-direction results must be reported rather than turned
into a claim that data age improves faithfulness.

**Direct paired change.** The clean/degraded pairs compare the same decision
before and after the observation-layer change. Three checkpoints show a clear
shift toward stale nodes, two show a clear shift away, and one is inconclusive.
DEF increases slightly in all six, which is opposite to the expected decline.
This does not validate attention: the clean decision-relevance control has
already failed.

![Evidence interpretation summary](../figures/v10_evidence_path_summary.png)

**Figure 5.1.** How the four forms of evidence should be interpreted. Each form
answers a different question and leads to a separate audit requirement.

These measures answer different questions. WAMSN checks whether stale evidence
is present, the duration tests describe an ordered condition, and paired shifts
measure the direct response of the same decision to a changed observation.
Keeping them separate prevents a freshness measure from being presented as a
faithfulness effect.

## 5.3 Why the result is not consistent

**Checkpoint dependence.** Independently trained checkpoints do not respond in
the same direction. The pattern appears under both tunnel and random triggers,
which reduces concern that it is produced only by the chosen tunnel. However,
the random trigger does not create identical stale-node populations and cannot
rule out scenario effects. The result therefore supports checkpoint dependence,
not a general effect of telemetry loss.

**Action composition.** Dispatch accounts for 80.2% of scorable decisions in
the corrected experiment. Dispatch DEF increases slightly under degradation,
whereas no-op DEF decreases in every checkpoint. The combined positive shift
is therefore driven by the larger dispatch group and does not describe both
action types.

**Attention extraction.** Different heads, layers, aggregations, or query rows
can change the direction or strength of the result. The network does not supply
one self-evident attention map. Any map shown to an operator therefore needs a
predeclared extraction rule that has been tested for stability.

**Action rule.** The main audit samples actions from the learned policy,
matching the declared scope. The smaller deterministic diagnostic also
completes journeys, but it is not part of the confirmatory audit. A deployment
audit must use the same action rule that the deployed system will use.

## 5.4 Why the construct-validity controls matter

A passenger-request node represents both information and an available action.
Removing it can therefore remove the action being explained. A peer-taxi node
is different: removing it hides vehicle information but leaves the action space
unchanged. A uniform random baseline would mix these two effects and could make
an explanation appear stronger simply because an action disappeared.

The type-matched, action-protected baseline makes the comparison fairer. It
compares request nodes with request nodes and taxi nodes with taxi nodes, while
protecting the selected request action. A positive DEF can then be interpreted
as an advantage over comparable random information rather than an advantage
created by deleting the chosen action.

LOO serves as a positive control. It ranks each node by the output change caused
by removing that node, and it produces a clearer DEF response than raw attention
for every checkpoint. This shows that the evaluator can respond to a more
informative perturbation ranking. LOO is not a ground-truth explanation, and it
does not prove that the evaluator captures every form of decision relevance.

The overlap check adds one final limit. In a small graph, explanation and random
top-k sets increasingly share the same nodes as `k` grows. This reduces the
resolution of DEF. The controls therefore support a careful conclusion: raw
attention lacks reproducible evidence of faithfulness within the resolution of
the present evaluator. They do not show that every attention map is wrong.

## 5.5 Why outage training did not solve the problem

The tested configuration trained with 30-second outages does not solve the
explanation problem. GAT-Outage does not make the direct clean/degraded response
more consistent across independently trained policies. Training with stale
observations may improve robustness for other purposes, but the present results
do not show that it makes raw attention a more reliable explanation.

This is not a general rejection of degradation-aware training. The two tested
configurations have different optimization horizons, so the comparison does not
isolate the effect of degraded training observations. More importantly, the
reward contains no term for explanation stability or faithfulness. A future
test of model improvement would require a common training budget, independent
training runs, and an explicit explanation objective.

## 5.6 Proposed audit framework

### 5.6.1 Purpose and scope

To address this risk, the thesis develops and applies a **freshness-aware
explanation audit framework**. The framework leaves the trained policy unchanged
and does not claim to make attention faithful. Instead, it sets rules for
deciding when an attention map has enough evidence to be shown as an audited
candidate explanation within a stated scope.
Table 5.1 links each observed risk to the required evidence and decision rule.

| Observed risk | Required check | Evidence | Decision rule |
|---|---|---|---|
| Attention may be attached to stale data | Report freshness separately | AoI and WAMSN | Never infer freshness from attention strength |
| Attention may not identify decision-relevant nodes | Use action-aware, type-matched tests | DEF, margin-DEF, and LOO | Do not claim a reason without evidence of decision relevance |
| Combined results may be dominated by no-op | Report no-op and dispatch groups | Action counts and action-specific DEF | Do not generalize a combined result across action types |
| The result may depend on attention extraction | Test the query row, layer, head, and aggregation | Sensitivity audits | Predeclare the extraction rule and report material sensitivity |
| One checkpoint may not represent another | Audit each checkpoint before release | Per-checkpoint results | Repeat after retraining or replacement |
| A pattern may not reproduce | Compare independent training runs | Results across seeds | Require a consistent conclusion across runs |
| The audited action may differ from deployment | Test the intended deployment action rule | Sampled-policy or argmax capability evidence | Require argmax capability only when deployment uses argmax |
| A tunnel-specific pattern may not transfer | Compare tunnel and random triggers | Paired shifts and 95% confidence intervals | Withhold when supported directions conflict or remain indeterminate |

The framework is intended for offline release review before an attention map is
shown to a dispatcher or other operator. It should be repeated after retraining,
checkpoint replacement, changes to the telemetry or graph pipeline, transfer to
a new operating scenario, and during periodic model review. It is not a live
freshness monitor. A deployed system must still display freshness separately and
suppress or warn about explanations based on stale inputs. Figure 5.2 shows the
corresponding offline audit workflow.

![Freshness-aware explanation audit workflow](../freshness_aware_explanation_audit.png)

**Figure 5.2.** Saved evidence from frozen checkpoints passes through nine
release checks, grouped here into six categories. The output controls whether
attention may be presented as an explanation; it does not alter the model or
selected action.

### 5.6.2 Inputs, outputs and use

The framework accepts evidence rather than raw attention alone. Table 5.2
summarizes its inputs and outputs.

| Component | Contents | Role in the audit |
|---|---|---|
| Candidate scope | frozen checkpoints, model names, training seeds, and held-out conditions | defines exactly what the decision covers |
| Freshness evidence | paired clean/degraded records, AoI, stale-node masks, WAMSN, and stale-attention shift | separates data age from attention strength |
| Faithfulness evidence | type-matched and action-protected DEF, LOO, and dispatch-action results | tests whether highlighted nodes matter to the action |
| Stability evidence | action groups, extraction alternatives, checkpoint synthesis, deployment-action diagnostic, and trigger comparison | tests whether the conclusion survives relevant choices |
| Audit output | per-check status, per-checkpoint decision, model decision, failed reasons, and permitted use | supports a traceable explanation-release decision |

The operating procedure is:

1. freeze the candidate checkpoint and evaluation protocol;
2. generate paired freshness, action-aware faithfulness, action-composition, and
   sensitivity evidence on held-out demand;
3. run preflight to establish that the evidence is complete and valid to
   analyze;
4. apply every release check to each checkpoint and then compare independent
   training runs;
5. present attention externally only if the result is `ELIGIBLE`, together with
   its freshness indicator and audited scope.

A preflight `PASS` is not an explanation-release pass. It only confirms that the
evidence can be analyzed. The release output is `ELIGIBLE`, `WITHHOLD`, or
`INCOMPLETE`. Individual checks may also be `INDETERMINATE` when the evidence is
complete but does not support a direction. A failed or indeterminate required
check produces `WITHHOLD`; missing evidence produces `INCOMPLETE`. In both
cases, attention must not be presented as the reason for the action.

### 5.6.3 Application to this study

Applying the framework to the completed evidence gives `WITHHOLD` for GAT and
GAT-Outage and for all six individual checkpoints. This is not a failed
experiment. The audit has completed and has found that the evidence is
insufficient for explanation release. In particular, the dispatch-action
decision-relevance intervals do not exceed the matched-random boundary,
attention extraction can reverse the stale-attention direction, and the
stale-attention response changes direction across training seeds. The random
trigger check passes for GAT-Outage but is indeterminate for GAT because one
checkpoint interval crosses zero. The argmax diagnostic completes journeys,
but it remains descriptive because the audit scope uses sampled actions.

Sampled dispatch performance provides evidence of task capability, but it
cannot show that the displayed reason is current or decision-relevant.

The framework checks whether an explanation has enough evidence to be shown; it
does not make the model itself more faithful. Improving the model would require
an explicit explanation objective and independent evaluation, as discussed in
Section 5.8.

The current study demonstrates the framework's rejection path but does not show
that a real trained model can reach `ELIGIBLE`. The LOO positive control shows
that the evaluator can reward a more informative ranking, while software tests
confirm that complete passing evidence produces an `ELIGIBLE` decision. A future
faithfulness-aware model is still needed to validate the eligibility path
empirically. The current thresholds must not be weakened after seeing these
results simply to obtain a passing model.

## 5.7 Limitations

The study uses one simulated district, 20 taxis, and 50 requests. Tunnel
exposure remains scenario-dependent, although the event-aware audit scores
every observed stale-exposed decision. It evaluates three independent training
seeds per model. Three seeds reveal variation but cannot estimate the wider
distribution of training outcomes. The selected policies pass the training
stability gates, but their held-out completed-journey results overlap the
legal-random CI and the simple greedy range. These baselines are capability
context, not evidence of a task-performance advantage. The experiment also
does not isolate the contribution of graph edges to policy capability.

Clean GAT and GAT-Outage also use different maximum training budgets and
learning-rate decay horizons, selected from validation-only stability
diagnostics. Their matched-seed comparison describes the two fitted
configurations but cannot isolate the effect of degraded training observations.

The local graph is small. Type matching makes random top-k sets overlap heavily
with the explanation, which compresses DEF. LOO is based on the same
single-node perturbation as comprehensiveness and is not an independent ground
truth explanation. Other perturbation choices or post-hoc explainers may give
different results.

The degradation layer freezes position and speed for fixed windows. Real
telemetry may include delayed packets, partial failures, map-matching errors,
and asynchronous recovery. The study audits attention as an explanation; it
does not claim that attention is useless inside the policy computation.

The paired clean/degraded comparison supports internal validity for the tested
observation-layer change because both observations come from the same simulated
traffic state. Simulation and scenario-design bias remain. Road-network choice,
generated demand, vehicle behavior, and
tunnel placement may affect which taxis become stale and how those taxis are
positioned in the local graph. In particular, a fixed tunnel can create an
exposure-selection effect if its traffic conditions differ from the rest of the
network. The conclusions are therefore limited to the tested SUMO setting and
should not be read as population estimates for real fleets or cities.

## 5.8 Future work

Future experiments should add training seeds and use multiple road networks,
demand levels, tunnel placements, and public mobility traces. The present
random-trigger control should be extended to several calibrated exposure rates
and persistent loss processes so that trigger location, exposure frequency,
and outage duration can be separated. Experiments should also include
disconnected-edge and message-passing ablations to establish how much the policy uses its
graph channel. Delayed, intermittent, and biased telemetry should be compared
with fixed freezes.

The explanation audit should also compare LOO with graph-specific post-hoc
methods and evaluate objectives that directly reward explanation consistency.
Any proposed improvement should be tested on held-out checkpoints, not only on
more episodes from one trained model. A human-factors study could then test
whether explicit freshness warnings lead operators to interpret the map more
appropriately.
