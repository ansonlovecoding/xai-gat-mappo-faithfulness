# 2. Literature review

This chapter develops the research gap in five steps. It first explains why
fleet dispatch is a relational reinforcement-learning problem. It then asks
what an explanation should provide in sequential decision making, why raw
attention is disputed as an explanation, how graph predictions are commonly
explained, and why faithfulness tests can themselves be misleading. The final
sections connect these issues to stale telemetry and position the experiment.

## 2.1 Reinforcement learning for fleet dispatch

Fleet dispatch links vehicles, passenger requests, and a road network over
time. A decision for one taxi changes the demand available to others, so a
collection of independent single-agent policies can create competition or
duplicated assignments. Lin et al. [1] model city-scale fleet management as a
cooperative multi-agent reinforcement-learning (MARL) problem. Qin, Zhu and Ye
[2] show that reinforcement learning is used
for matching, repositioning, pricing, and route decisions, often under changing
supply and demand [2].

MARL offers several ways to represent coordination. MADDPG learns decentralised
actors with centralised critics [3]. QMIX factorises a team value into agent
values while preserving a monotonic relation to the joint action [4]. MAPPO
uses the more familiar PPO objective with centralised training and
decentralised execution; its empirical stability makes it a practical baseline
for cooperative tasks [5]. This dissertation adopts MAPPO because the research
focus is the explanation attached to each taxi's local decision, not a new MARL
algorithm.

Graphs provide a natural representation of the local dispatch problem. Nodes
can represent the acting taxi, nearby taxis, and open requests; edges permit
their information to be combined [6]. A graph attention network (GAT) assigns
learned weights while aggregating neighbouring nodes [7]. Actor-Attention-Critic
similarly uses attention for communication in MARL [8]. These weights are easy
to visualise, which makes them tempting to present as reasons for an action.
Ease of display, however, is not evidence that the weights identify the
information that caused the decision.

Recent dispatch methods continue to use relational structure. BMG-Q represents
ride-pooling dispatch through a local bipartite matching graph [33]; CoopRide
studies cooperation across city grids [37]; and DualG-MARL combines
vehicle-state and task graphs [38]. Their primary outcomes are dispatch or
scheduling quality. They do not test whether graph weights remain valid
explanations when vehicle telemetry becomes stale. These studies are therefore
useful task benchmarks, but not direct faithfulness benchmarks.

## 2.2 Explainability in reinforcement learning

Explainable reinforcement learning (XRL) covers more than a visual saliency
map. An explanation may describe which state features mattered, summarise a
policy, contrast the selected action with an alternative, or show the sequence
of events that led to an outcome. Reviews by Heuillet, Couthouis and
Diaz-Rodriguez [10], Puiutta and Veith [32], and Milani et al. [30] all stress
that the method must match the user's question. A model developer debugging a
policy and an operator checking one live action do not need the same output.
The systematic taxonomy by Bekkemoen [31] reaches the same conclusion across a
larger body of recent XRL studies.

This dissertation concerns a local, post-decision operator question: *which
visible vehicles or requests supported this dispatch action?* Earlier XRL work
demonstrates several possible answers. Greydanus et al. [34] perturb image
regions to visualise what changes an Atari policy. Madumal et al. [11] use a
causal model to generate contrastive explanations. Mott et al. [35] introduce
an attention bottleneck intended to expose what an RL agent uses. These
approaches differ in mechanism, but they share an important lesson: a useful
picture and a faithful account of model dependence are separate properties.

That distinction is especially important in an operational dashboard. A
plausible explanation may be easy for a person to read but weakly connected to
the policy computation. A faithful explanation should change when information
that matters to the action is removed or retained. The experiment therefore
does not ask whether users like the attention map. It asks whether its ranking
passes a controlled decision-level faithfulness test.

## 2.3 Is attention an explanation?

The modern debate began with Jain and Wallace [12], who found that attention
weights could be weakly correlated with other importance measures and that very
different attention distributions could produce similar predictions.
Wiegreffe and Pinter [13] argued that this does not justify rejecting all
attention explanations and proposed more careful tests. Serrano and Smith [14]
likewise showed that removing high-attention items can affect outputs, but that
the relationship varies by model and task. The useful conclusion is not a
universal yes or no. It is that attention needs an explicit claim and an
appropriate test.

Attention became widely visible through Transformer models [9], but visibility
also encouraged a broader interpretability claim than the mechanism alone can
support. Lipton [20] warns that the word *interpretability* often combines
several different goals that should be evaluated separately.

Jacovi and Goldberg [17] separate *plausibility*, which concerns whether an
explanation looks reasonable to a person, from *faithfulness*, which concerns
whether it reflects the model's actual reasoning process. This
dissertation follows that distinction. The attention weights are not rejected
because they are internal values, and they are not accepted because they are
coupled to the actor. They are treated as a candidate explanation whose
decision relevance must be measured.

Layer and head aggregation add another difficulty. A two-layer, multi-head
network does not produce one unique attention map. Averaging heads, selecting a
layer, taking a maximum, or composing attention across layers can produce
different rankings. Abnar and Zuidema [26] propose attention rollout and flow to
account for information mixing across layers, and report stronger correlations
with ablation and gradient importance than raw attention in Transformers.
Although their model is not a GAT-MARL dispatcher, the methodological point
transfers: an attention result is incomplete unless the aggregation rule is
defined and its sensitivity is checked.

## 2.4 Explaining graph neural networks

GNN explanations may identify important features, nodes, edges, or subgraphs.
GNNExplainer learns a compact subgraph and feature mask that preserves a
prediction [23]. PGExplainer parameterises explanation generation so that it
can be shared across examples [24]. SubgraphX searches for important subgraphs
using Shapley-value approximations [36]. These methods show that graph
explanation is a distinct problem: removing one node can alter both its own
features and the messages available to other nodes.

The survey by Yuan et al. [25] organises GNN explainers by target, mechanism,
and scope. GraphFramEx goes further by comparing explanation methods under
different user needs and combining necessity and sufficiency views [29]. Its
results show that no single method dominates every evaluation dimension. This
supports the use of more than one diagnostic in the present study: DEF tests
counterfactual relevance, taxi-only rank correlation compares attention with
leave-one-out effects, and stale-attention measures describe freshness
exposure.

Most GNN explainability benchmarks study node or graph classification. Fleet
dispatch differs because some graph nodes also define actions. A passenger
request is both information and a selectable action, while a nearby taxi is
context only. Deleting these node types is therefore not the same
intervention. This action-linked graph structure motivates the construct-
validity audit in Section 3.7.

## 2.5 Faithfulness evaluation and its pitfalls

Comprehensiveness and sufficiency are common perturbation measures. In the
ERASER benchmark, comprehensiveness asks how much a prediction weakens when the
explanation is removed, while sufficiency asks how well the selected evidence
alone preserves it [16]. Liu et al. [15] similarly test attention by measuring
faithfulness violations. These measures are useful because they connect an
explanation to model behaviour instead of visual appeal.

Model-agnostic methods such as LIME also use local perturbations to estimate
which inputs support a prediction [18]. Their usefulness depends on whether the
perturbed samples remain meaningful for the model and task.

Perturbation is not automatically valid. Removing input features may create
samples unlike the data seen during training. ROAR addresses this issue by
removing features and retraining the model, showing why a simple deletion test
can confound attribution quality with distribution shift [28]. Adebayo et al.
[27] provide a different sanity check: an explanation should respond when model
parameters or training labels are randomised. Alvarez-Melis and Jaakkola [19]
show that explanation methods can also be locally unstable. Together,
these studies make the evaluator part of the object being audited.

The present action-deletion problem is related but more specific. If a selected
request node is masked, its action logit becomes unavailable. A large margin
change can then occur even if the request features carried no useful evidence.
A random baseline that removes more request nodes than the attention top-k is
not a fair comparison. Type matching, selected-action protection, and clamp
logging are introduced to control this mechanism. A leave-one-out perturbation
ranking acts as a positive control for comprehensiveness, while Gradient x
Input provides a comparator that does not use attention coefficients.

The small graph creates a second limit. When `k = 3` and only a few valid nodes
exist, a random size-three set must overlap substantially with the explanation
set. This compresses the possible difference between them. Reporting the exact
expected overlap and a taxi-only rank correlation makes that limitation
visible instead of treating every near-zero DEF as decisive evidence.

## 2.6 Age of Information and telemetry degradation

Age of Information (AoI) measures the time since the newest received update
was generated [21]. It is widely used to study communication freshness and the
effect of delayed state on estimation or control. The explanation question is
different. A dispatch action and its attention map may remain available even
when a vehicle reading has stopped updating. The operator can see a strong
weight without knowing that its source is old.

This dissertation uses AoI only as a node-level freshness descriptor and as
the weight inside WAMSN. The manipulated condition is the declared duration of
an observation-layer outage triggered by tunnel entry. The design does not
manipulate AoI independently, so it does not claim that a larger AoI value
causes lower faithfulness. Instead, it asks whether stale exposure and
attention behaviour remain separately trustworthy under controlled outages.

## 2.7 Research gap and positioning

The reviewed work establishes four points: relational RL is appropriate for
fleet dispatch; attention is easy to expose but disputed as an explanation;
GNN faithfulness requires graph-aware perturbations; and stale state matters to
operational control. What is not joined is the assurance question at their
intersection.

Within the graph- and RL-explainability studies reviewed in Sections 2.2-2.5,
no evaluation protocol was identified that combines all of the following:

- a graph-attention MARL action explained at decision level;
- paired clean and stale versions of the same operational observation;
- explicit measurement of attention placed on stale telemetry;
- a control for graph nodes that also remove available actions; and
- replication across independently trained checkpoints.

The contribution is therefore an audit protocol and an empirical test, not a
new dispatch algorithm. SUMO supplies a controlled simulated state [22]. The GAT-
MAPPO policy supplies a realistic attention channel. The research contribution
is to test whether that channel gives a reproducible freshness-aware and
decision-relevant explanation, and to state clearly when the evidence cannot
support that claim.
