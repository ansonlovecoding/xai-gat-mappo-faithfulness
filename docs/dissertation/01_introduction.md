# 1. Introduction

## 1.1 Context

Fleet dispatch is relational. A taxi's useful action depends on nearby
vehicles, open passenger requests, road conditions, and competition for the
same demand. Graph neural networks (GNNs) represent these relationships directly,
and graph attention networks (GATs) add learned weights over neighboring
nodes [6], [7]. Because the weights are easy to display, they are often read as
an explanation of which vehicles or requests influenced a decision.

Interpreting these weights requires care. Studies in NLP and other attention-based
models have found that attention weights can be weakly related to the evidence
that changes a model's output [12], [14], [15]. Wiegreffe and Pinter [13]
caution that such findings do not rule out all attention-based explanations.
The issue becomes more serious in
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

This makes the explanation difficult to trust. An operator may read a complete attention
map as a trustworthy reason even when its evidence is stale, and stable dispatch
performance cannot show whether the explanation is valid. This dissertation tests the reliability of raw graph-attention explanations
under clean and degraded telemetry, with dispatch performance providing context
for interpreting the explanation results.

This dissertation uses *faithfulness decoupling* to describe a situation in
which an explanation still appears credible after its supporting data have
become stale, even though the displayed attention weights may not reliably
identify the evidence that influenced the decision. One possible pattern is declining
faithfulness while dispatch performance remains stable. The core question is
whether attention provides reliable evidence under the tested conditions.

The term identifies the risk being investigated, rather than a result assumed
in advance. Showing that degradation reduces faithfulness would require a valid clean
faithfulness baseline and a lower score in the paired degraded observation.
The rerun finds declines in some checkpoints but no consistent response across seeds. Its contribution includes identifying why the declared attention channel fails the release audit and distinguishing that model-level decision from checkpoint-specific changes under degradation.

## 1.3 Research questions

The central research question is:

**Can raw graph-attention weights be trusted as explanations of fleet-dispatch
decisions when vehicle telemetry becomes stale?**

Four research questions guide the evaluation:

1. **RQ1:** Under clean telemetry, do raw graph-attention weights identify
   decision-relevant nodes more reliably than type-matched random controls?
2. **RQ2:** Does tunnel-triggered telemetry degradation change the attention
   assigned to stale vehicle nodes relative to the paired clean observation?
3. **RQ3:** As outage duration increases, are changes in stale-node attention
   and decision-level faithfulness consistent across independently trained
   checkpoints?
4. **RQ4:** In the tested configurations, does degradation-aware training
   produce more consistent attention and faithfulness responses under telemetry
   degradation than clean training?

All four questions depend on a valid faithfulness measure. The evaluation
therefore distinguishes the removal of information from the removal of an
available action, which can otherwise distort the measured effect.

## 1.4 Objectives

The measurable objectives are to:

1. develop a reproducible Simulation of Urban Mobility (SUMO)
   fleet-dispatch benchmark that creates paired clean and degraded observations
   from the same simulated state and uses held-out test demand;
2. evaluate whether raw graph-attention weights identify
   decision-relevant nodes under clean telemetry using action-protected,
   type-matched controls and a leave-one-out (LOO) positive control;
3. quantify the paired change in attention assigned to stale vehicle
   nodes under tunnel-triggered outages and test its sensitivity to randomly
   triggered telemetry loss;
4. assess whether stale-node attention and decision-level faithfulness
   responses are consistent across outage durations, independently trained
   checkpoints, action groups, and attention-extraction choices;
5. compare the matched clean-trained and degradation-aware configurations to
   assess whether the tested degradation-aware configuration gives more
   consistent attention and faithfulness responses;
6. implement and apply a freshness-aware explanation audit framework
   that converts the collected evidence into an `ELIGIBLE`, `WITHHOLD`, or
   `INCOMPLETE` explanation-release decision.

Objectives 1-2 provide the benchmark and clean controls for RQ1; objective 3
addresses the paired telemetry comparison in RQ2; objective 4 tests repeatability and sensitivity for RQ3; and objective 5 addresses RQ4.
Objective 6 uses these findings to reach the audit decision. The answers
and release outcomes are presented in Sections 4.10-4.11.

## 1.5 Main findings

The tested self-row attention channel lacks consistent evidence of decision
relevance and stability. Figure 1.1 summarizes the practical consequence: an
attention map can remain visible as telemetry becomes stale, even when the
evidence is insufficient to present it as an explanation.

![Summary of stale data and the explanation-release decision](../figures/v12_main_findings_summary.png)

**Figure 1.1.** Main finding. A tunnel-triggered outage leaves the policy with a
stale observation. Attention remains available, but the audit withholds it
because decision relevance and extraction consistency are not established.
Freshness visibility passes as a reporting check; it does not certify fresh data.

Under clean telemetry, the tested averaged self-row attention has dispatch
margin-DEF below its type-matched random control for both model families. A leave-one-out control produces a
positive response, showing that the evaluator can detect a more
decision-relevant ranking even though raw attention does not provide one.

Longer outages consistently increase the measured exposure to stale vehicle information. However, the paired response differs across trained policies: seven checkpoints shift attention toward stale nodes and three shift away. Probability-DEF intervals are positive for seven checkpoints, negative for both seed-45 checkpoints, and inconclusive for GAT-Outage seed 46. The study finds degradation-related loss of faithfulness in some checkpoints, but no consistent effect across all seeds.

The interpretation also depends on how attention is extracted and whether the selected action is dispatch or no-op. Nine checkpoints have opposite paired DEF directions for these action groups. The selected-request-row control is positive for GAT-Outage but negative for GAT, although the default self-row control is negative for both model families. Outage training does not remove checkpoint variation.

The practical conclusion is that data freshness and explanation faithfulness
must be checked separately for each trained policy. In the present experiment,
the policy may still produce an action, but its raw attention weights should be
withheld as an explanation.

## 1.6 Contributions

The dissertation makes four main contributions:

1. a reproducible paired clean/degraded benchmark that creates two observations
   from the same SUMO state, separates physical ground truth from policy input,
   and distinguishes the tunnel trigger from the observation-layer outage;
2. an action-aware, type-matched evaluation protocol that reduces the deletion
   bias caused by request nodes representing both information and available
   actions, checked through LOO, overlap, and action-type diagnostics;
3. evidence across independently trained checkpoints showing that stale-data
   exposure increases consistently with outage duration, while direct paired
   attention and DEF shifts vary across checkpoints and attention-extraction
   choices, with the pattern also tested under random telemetry loss;
4. an implemented freshness-aware explanation audit framework that combines
   freshness, faithfulness, action composition, extraction sensitivity, and
   checkpoint-consistency evidence to return an `ELIGIBLE`, `WITHHOLD`, or
   `INCOMPLETE` explanation-release decision.

## 1.7 Dissertation structure

**Chapter 1: Introduction.** Defines the research problem, research questions,
objectives, main findings, and contributions.

**Chapter 2: Literature Review.** Reviews MARL fleet dispatch, graph attention,
explanation faithfulness, graph explainability, and telemetry freshness.

**Chapter 3: Methodology.** Describes dataset selection and EDA, the SUMO environment, policy models,
observation-layer degradation, faithfulness measures, evaluation design, implementation details, and audit rules.

**Chapter 4: Results.** Reports policy capability, construct-validity checks,
faithfulness results, stale-data exposure, sensitivity analyses, and audit
decisions.

**Chapter 5: Discussion.** Interprets the findings, explains their limitations,
and presents the freshness-aware explanation audit framework.

**Chapter 6: Conclusion.** Answers the research questions and summarizes the
practical implications for trustworthy fleet-dispatch explanations.
