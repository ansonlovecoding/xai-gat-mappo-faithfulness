# 1. Introduction

## 1.1 Context

Fleet dispatch is relational. A taxi's useful action depends on nearby
vehicles, open passenger requests, road conditions, and competition for the
same demand. Graph neural networks represent these relationships directly,
and graph attention networks (GATs) add learned weights over neighboring
nodes [6], [7]. Because the weights are easy to display, they are often read as
an explanation of which vehicles or requests influenced a decision.

That reading is convenient, but it is not automatically valid. Prior work has
shown that an attention weight can be weakly related to the evidence that
actually changes a model's output [12], [14], [13], [15]. The issue becomes more serious in
an operational system because the input itself may no longer describe the
current world.

## 1.2 Problem statement

Vehicle telemetry can be interrupted by tunnels and other signal-loss areas.
During an interruption, a dispatch platform may keep the last received
position and speed. The reading remains available, but its Age of Information
(AoI) grows [21]. An attention map can therefore look complete even though some
nodes represent old information. The interface reveals where
the model placed attention, but it does not reveal whether that information is
fresh or whether removing it would change the decision.

This creates an assurance problem. An operator may read a complete attention
map as a trustworthy reason even when its evidence is stale, and stable dispatch
performance cannot show whether the explanation is valid. The dissertation is
therefore not trying to improve the number of pickups. It is trying to determine
whether raw graph-attention weights provide dependable evidence about a
decision under clean and degraded telemetry.

This dissertation uses **faithfulness decoupling** to describe the risk that an
explanation remains available and visually plausible after its supporting data
have become stale, without a dependable link between the displayed weights and
the evidence that changes the decision. One possible pattern is declining
faithfulness while dispatch performance remains stable. The broader problem,
however, is whether attention provides reliable assurance under the tested
conditions.

## 1.3 Research questions

The central research question is:

**Can raw graph-attention weights be trusted as explanations of fleet-dispatch
decisions when vehicle telemetry becomes stale?**

Four operational questions provide the evidence needed to answer it:

1. **RQ1:** Is graph attention a faithful explanation under clean telemetry?
2. **RQ2:** When tunnel-triggered outages occur, does attention move toward
   stale vehicle nodes?
3. **RQ3:** As outage duration increases, does the relationship between
   attention, stale-data exposure, and decision relevance remain dependable?
4. **RQ4:** Does training the policy under telemetry degradation make the
   explanation response more reliable?

The construct-validity audit supports RQ1-RQ3. It is not a separate primary
problem. Its purpose is to check that the faithfulness metric measures hidden
information rather than the accidental deletion of an available action.

## 1.4 Objectives

The measurable objectives are to:

- build a reproducible SUMO fleet-dispatch benchmark with training, validation,
  and held-out test demand;
- train clean and degradation-aware policies across three independent seeds;
- trigger signal loss from tunnel entry while applying the resulting freeze at
  the observation boundary;
- compare clean telemetry with fixed observation-layer outages of 10, 20, 30,
  and 60 seconds;
- measure explanation faithfulness with action-aware, type-matched
  counterfactual controls;
- check whether the faithfulness evaluator is sensitive to an LOO perturbation
  ranking and quantify the resolution lost through top-k overlap;
- measure how much attention is assigned to stale vehicle information;
- test whether conclusions change across attention layers, heads, rollout, and
  the query row used as the explanation;
- treat the trained policy, rather than each individual decision, as the unit
  for judging whether a finding is consistent.

## 1.5 Main findings

The experiment gives one consistent overall answer: **raw attention
weights are not validated as dependable explanations by this study**. This is a
negative assurance result, supported by three connected findings.

First, clean type-matched DEF varies from -0.0009 to +0.0031 across the six
selected checkpoints. An LOO perturbation control is larger and positive in
every checkpoint. Taxi-only rank agreement with LOO is positive but ranges
from 0.088 to 0.699. Attention therefore shows some agreement with a
decision-relevant perturbation ranking, but the strength of that agreement is
not reproducible across checkpoints. The supporting audit also shows
why a uniform random baseline is misleading when request nodes are actions.

Second, longer outages consistently increase WAMSN. This result appears in all
three GAT checkpoints and all three GAT-Outage checkpoints.
It shows that more of the displayed attention is attached to stale information
when stale exposure lasts longer. It does **not** show that AoI causes lower
faithfulness.

Third, the paired change in stale-node attention is not consistent across
training seeds. It is positive for checkpoints trained with seeds 42 and 43 but
negative for checkpoints trained with seed 44 in both GAT and GAT-Outage. H1 is
supported in five of six selected checkpoints, and the within-episode WAMSN-DEF
association is supported in all six. However,
the direct clean-twin DEF shift is negative for four checkpoints and positive
for two. This distinction matters: a within-policy association can be reproducible
without giving every trained policy the same response to degradation.

A 30-second random-trigger sensitivity check retains the attention-shift
direction in five of six checkpoints and the probability-DEF direction in four
of six. This reduces concern that the checkpoint pattern is only an artifact of
the fixed tunnel location, while preserving the need for a cautious conclusion.

The action audit narrows this result further. After decisions with no available
request are removed, 93.0% of records still select no-op and only 7.0% dispatch
a request. The H1 and H4 patterns found in the combined data are not reproduced
within the smaller dispatch stratum. In four of six checkpoints, the sign of
the paired DEF change for dispatch decisions also differs from the sign in the
combined analysis. The combined result therefore describes the sampled policy
distribution; it cannot be presented as a dispatch-action-specific effect.

The result is also sensitive to how attention is turned into one explanation.
Individual heads can reverse the stale-attention direction within a checkpoint,
and changing from the self row to the selected request row mainly improves the
checkpoints trained with seed 43. These checks define more precisely why the
study does not establish a reliable default explanation.

Finally, degradation-aware training does not remove training-seed variation.
All selected policies pass the declared stochastic capability and training
stability gates. However, a deterministic diagnostic gives zero pickups for
all six GAT checkpoints over three held-out episodes each. The learned action
distributions can produce useful sampled behavior, but their argmax actions are
collapsed to no-op. This limitation narrows the setting in which the
explanation response can be interpreted.
Taken together, the results do not prove that attention is always unfaithful.
They show that attention cannot be assumed trustworthy by default; freshness and
faithfulness must be checked separately for every released policy.

## 1.6 Contributions

The dissertation contributes:

- a paired clean/degraded benchmark in which SUMO ground truth is separated
  from the observation received by the policy;
- an explicit distinction between a tunnel trigger and an observation-layer
  outage;
- an action-aware faithfulness protocol for graphs where request nodes also
  define available actions, supported by LOO and overlap diagnostics;
- a sensitivity audit showing how layer, head, rollout, and query-row choices
  affect the reported attention explanation;
- evidence across independently trained policies that stale-data exposure is
  consistent, while attention reallocation and faithfulness effects are not;
- an action-stratified diagnostic showing that combined decision statistics do
  not reliably represent the smaller set of dispatch actions;
- a freshness-aware explanation assurance approach that turns the findings
  into release checks for telemetry freshness, action-aware faithfulness,
  attention aggregation, and independent training runs;
- a reproducible experiment runner, validation-selected checkpoints,
  preflight checks, machine-readable results, and training-seed synthesis.

## 1.7 Dissertation structure

Chapter 2 reviews MARL dispatch, graph attention, explanation faithfulness,
and AoI. Chapter 3 describes the simulator, models, degradation layer,
metrics, and statistical design. Chapter 4 reports the results. Chapter 5
discusses what can and cannot be concluded. Chapter 6 closes with the practical
implications for trustworthy fleet-dispatch explanations.
