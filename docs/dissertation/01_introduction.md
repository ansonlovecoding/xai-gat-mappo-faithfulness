# 1. Introduction

## 1.1 Context

Fleet dispatch is relational. A taxi's useful action depends on nearby
vehicles, open passenger requests, road conditions, and competition for the
same demand. Graph neural networks represent these relationships directly,
and graph attention networks (GATs) add learned weights over neighbouring
nodes [6], [7]. Because the weights are easy to display, they are often read as
an explanation of which vehicles or requests influenced a decision.

That reading is convenient, but it is not automatically valid. Prior work has
shown that an attention weight can be weakly related to the evidence that
actually changes a model's output [12]-[15]. The issue becomes more serious in
an operational system because the input itself may no longer describe the
current world.

## 1.2 Problem statement

Vehicle telemetry can be interrupted by tunnels and other signal-loss areas.
During an interruption, a dispatch platform may keep the last received
position and speed. The reading remains available, but its Age of Information
(AoI) grows [21]. An attention map can therefore continue to look complete even
though some of the nodes represent old information. The interface reveals where
the model placed attention, but it does not reveal whether that information is
fresh or whether removing it would change the decision.

This creates an assurance problem. An operator may read a complete attention
map as a trustworthy reason even when its evidence is stale, and stable dispatch
performance cannot show whether the explanation is valid. The dissertation is
therefore not trying to improve the number of pickups. It is trying to determine
whether raw graph-attention weights provide dependable evidence about a
decision under clean and degraded telemetry.

The study uses **faithfulness decoupling** for the risk that an explanation
remains available and visually plausible after the freshness of its supporting
data has expired, without a dependable link between the displayed weights and
the evidence that changes the decision. The original hypothesis expected
faithfulness to decline while performance remained stable. The final experiment
tests this possible pattern, but the broader problem is whether attention offers
a reliable assurance at all.

## 1.3 Research questions

The central research question is:

**Can raw graph-attention weights be trusted as explanations of fleet-dispatch
decisions when vehicle telemetry becomes stale?**

Four operational questions provide the evidence needed to answer it:

1. **RQ1:** Is graph attention a faithful explanation under clean telemetry?
2. **RQ2:** When tunnel-triggered outages occur, does attention move toward
   stale vehicle nodes?
3. **RQ3:** Does explanation faithfulness change more than dispatch behaviour
   as outage duration increases?
4. **RQ4:** Does training the policy under telemetry degradation make the
   explanation response more reliable?

The construct-validity audit supports RQ1-RQ3. It is not a separate primary
problem. Its purpose is to check that the faithfulness metric measures hidden
information rather than the accidental deletion of an available action.

## 1.4 Objectives

The measurable objectives are to:

- build a reproducible SUMO fleet-dispatch benchmark with train, validation,
  and held-out test demand;
- train clean and degradation-aware policies across three independent seeds;
- trigger signal loss from tunnel entry while applying the resulting freeze at
  the observation boundary;
- compare clean telemetry with fixed observation-layer outages of 10, 20, 30,
  and 60 seconds;
- measure explanation faithfulness with action-aware, type-matched
  counterfactual controls;
- measure how much attention is assigned to stale vehicle information;
- treat the trained policy, rather than each individual decision, as the unit
  for judging whether a finding is consistent.

## 1.5 Main findings

The completed v4 experiment gives one consistent overall answer: **raw attention
weights are not validated as dependable explanations by this study**. This is a
negative assurance result, supported by three connected findings.

First, type-matched DEF remains close to zero for the audited GAT policies.
This does not prove that attention is harmful; it means the experiment finds
little evidence that the displayed weights identify more decision-relevant
information than a fair random explanation. The supporting audit shows why a
uniform random baseline gives misleading results when request nodes are also
actions.

Second, longer outages consistently increase WAMSN. This result appears in all
three clean-trained B2 policies and all three degradation-trained H5 policies.
It shows that more of the displayed attention is attached to stale information
when stale exposure lasts longer. It does **not** show that AoI causes lower
faithfulness.

Third, the paired change in stale-node attention is not consistent across
training seeds. It is positive for seeds 42 and 43 but negative for seed 44 in
both B2 and H5. H1 and H2 are unsupported in every trained policy. The
association between WAMSN and DEF is supported for two H5 seeds, but not the
third and not any B2 seed. This disagreement is not an absence of a conclusion:
it shows that raw attention has no stable, model-independent response to stale
telemetry.

Finally, degradation-aware training does not remove training-seed variation.
It also produces one weak policy replicate, which limits any mitigation claim.
Taken together, the results do not prove that attention is always unfaithful.
They show that attention cannot be trusted by default; freshness and
faithfulness must be checked separately for every released policy.

## 1.6 Contributions

The dissertation contributes:

- a paired clean/degraded benchmark in which SUMO ground truth is separated
  from the observation received by the policy;
- an explicit distinction between a tunnel trigger and an observation-layer
  outage;
- a type-matched faithfulness protocol for graphs where request nodes also
  define available actions;
- evidence across independently trained policies that stale-data exposure is
  consistent, while attention reallocation and faithfulness effects are not;
- a reproducible experiment runner, validation-selected checkpoints,
  preflight checks, machine-readable results, and training-seed synthesis.

## 1.7 Dissertation structure

Chapter 2 reviews MARL dispatch, graph attention, explanation faithfulness,
and AoI. Chapter 3 describes the simulator, models, degradation layer,
metrics, and statistical design. Chapter 4 reports the v4 results. Chapter 5
discusses what can and cannot be concluded. Chapter 6 closes with the practical
implications for trustworthy fleet-dispatch explanations.
