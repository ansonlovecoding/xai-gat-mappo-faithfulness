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
performance cannot show whether the explanation is valid. This dissertation
focuses on explanation reliability rather than increasing the number
of pickups. It tests whether raw graph-attention weights provide reliable
evidence about a decision under clean and degraded telemetry.

This dissertation uses **faithfulness decoupling** to describe the risk that an
explanation remains available and visually plausible after its supporting data
have become stale, without a reliable link between the displayed weights and
the evidence that changes the decision. One possible pattern is declining
faithfulness while dispatch performance remains stable. The core question is
whether attention provides reliable evidence under the tested conditions.

## 1.3 Research questions

The central research question is:

**Can raw graph-attention weights be trusted as explanations of fleet-dispatch
decisions when vehicle telemetry becomes stale?**

Four operational questions provide the evidence needed to answer it:

1. **RQ1:** Is graph attention a faithful explanation under clean telemetry?
2. **RQ2:** When tunnel-triggered outages occur, does attention move toward
   stale vehicle nodes?
3. **RQ3:** As outage duration increases, does the relationship between
   attention, stale-data exposure, and decision relevance remain reliable?
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
- test whether the paired result remains similar when telemetry loss is
  triggered randomly rather than by one fixed tunnel location;
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

Overall, **this study does not validate raw attention weights as reliable
explanations**.

Under clean telemetry, type-matched DEF remains close to its random control and
varies across the six checkpoints. A leave-one-out perturbation ranking gives a
larger positive DEF in every checkpoint, showing that the evaluator can respond
to a decision-relevant ranking even though raw attention does not provide a
stable result.

Longer outages increase AoI-weighted stale exposure for every checkpoint. The
paired response is less consistent: checkpoints trained with seeds 42 and 43
assign more attention to stale nodes on average, while those trained with seed
44 assign less. Paired DEF also changes direction across checkpoints. A
random-trigger check shows a similar overall pattern, making a tunnel-only
explanation less plausible without establishing a universal degradation
effect.

The action audit narrows the conclusion. Among decisions with at least one
available request, 93.0% select no-op. The main duration and exposure
associations do not reproduce within the smaller dispatch stratum. Attention
results also change with the layer, head, aggregation rule, and query row used
to construct the explanation.

Degradation-aware training does not remove this variation. The sampled policy
distributions outperform the declared lower bounds, but all six GAT
checkpoints choose no-op in the deterministic diagnostic. The study therefore
audits explanations for sampled stochastic decisions; it does not establish a
deployable argmax dispatcher. Overall, freshness and faithfulness must be
checked separately for every policy checkpoint before release.

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
- a random-trigger sensitivity check that tests whether the paired pattern is
  limited to one fixed tunnel location;
- an action-stratified diagnostic showing that combined decision statistics do
  not reliably represent the smaller set of dispatch actions;
- a freshness-aware explanation audit framework that turns the findings into
  release checks for telemetry freshness, action-aware faithfulness,
  attention aggregation, and independent training runs;
- a reproducible experiment runner, validation-selected checkpoints,
  preflight checks, machine-readable results, and training-seed synthesis.

## 1.7 Dissertation structure

Chapter 2 reviews MARL dispatch, graph attention, explanation faithfulness,
and AoI. Chapter 3 describes the simulator, models, degradation layer,
metrics, and statistical design. Chapter 4 reports the results. Chapter 5
discusses what can and cannot be concluded. Chapter 6 closes with the practical
implications for trustworthy fleet-dispatch explanations.
