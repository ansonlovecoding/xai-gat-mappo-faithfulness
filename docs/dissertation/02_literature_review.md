# 2. Literature review

This chapter reviews research on graph-based fleet dispatch, attention-based
explanations, and explanation faithfulness under changing observations. It
draws on foundational studies to establish the models and evaluation concepts,
alongside recent work on graph attribution, delayed observations, and
multi-agent communication. Particular attention is given to how these studies
assess decision relevance and whether their assumptions remain valid when
vehicle telemetry becomes stale.

## 2.1 Reinforcement learning for fleet dispatch

Fleet dispatch links vehicles, passenger requests, and a road network over
time. A decision for one taxi changes the demand available to others, so a
collection of independent single-agent policies can create competition or
duplicated assignments. Lin et al. [1] model city-scale fleet management as a
cooperative multi-agent reinforcement-learning (MARL) problem. Qin, Zhu and Ye
[2] show that reinforcement learning (RL) is used for matching, repositioning,
pricing, and route decisions, often under changing supply and demand.

MARL offers several ways to represent coordination. Multi-Agent Deep
Deterministic Policy Gradient (MADDPG) learns decentralized actors with
centralized critics [3]. QMIX factorizes a team value into agent values while
preserving a monotonic relation to the joint action [4].
Multi-Agent Proximal Policy Optimization (MAPPO) uses the more familiar
Proximal Policy Optimization (PPO) objective with centralized training and
decentralized execution; its empirical stability makes it a practical baseline
for cooperative tasks [5]. MAPPO provides a stable training basis for examining the explanation attached
to each taxi's local decision.

Graphs provide a natural representation of the local dispatch problem. Nodes
can represent the acting taxi, nearby taxis, and open requests; edges permit
their information to be combined [6]. A graph attention network (GAT) assigns
learned weights while aggregating neighboring nodes [7]. Actor-Attention-Critic
uses attention in its centralized critic to select information from other
agents [8]. GAT-MARL is a suitable test case because it represents the relational,
multi-agent structure of fleet dispatch and exposes attention weights for audit. These weights are
easy to visualize, which makes them tempting to present as reasons for an
action. Ease of display, however, is not evidence that the weights identify the
information that caused the decision.

Recent dispatch methods continue to use relational structure. Hu, Feng and Li
[33] propose Localized Bipartite Match Graph Attention Q-learning (BMG-Q), which
uses a local bipartite graph for ride-pooling dispatch. CoopRide studies
cooperation across city grids [37], while DualG-MARL combines vehicle-state and
task graphs [38]. These studies evaluate dispatch or
scheduling quality rather than whether their learned graph or attention weights
can be presented as faithful explanations when vehicle telemetry becomes stale.
They provide useful task benchmarks, but not direct faithfulness benchmarks.

## 2.2 Explainability in reinforcement learning

Explainable reinforcement learning (XRL) covers more than a visual saliency
map. An explanation may describe which state features mattered, summarize a
policy, contrast the selected action with an alternative, or show the sequence
of events that led to an outcome. Reviews by Heuillet, Couthouis and
Díaz-Rodríguez [10] and Milani et al. [30] stress that the method must match the
user's question. A model developer debugging a policy and an operator checking
one live action do not need the same output. The systematic taxonomy by
Bekkemoen [31] reaches the same conclusion across a larger body of recent XRL
studies.

This dissertation concerns a local, post-decision operator question: *which
visible vehicles or requests supported the selected action, whether dispatch or
no-op?* Earlier XRL work
offers several ways to answer this question. Greydanus et al. [34] perturb image
regions to visualize what changes an Atari policy. Madumal et al. [11] use a
causal model to generate contrastive explanations. Mott et al. [35] introduce
an attention bottleneck intended to expose what an RL agent uses. These
approaches differ in mechanism, but they share an important lesson: a useful
picture does not necessarily show what the model relied on.

For an operational dashboard, readability and faithfulness must be considered
separately. An explanation may be easy to read but weakly connected to the
policy computation. Perturbation tests assess whether removing the identified evidence weakens
the prediction and whether retaining sufficient evidence preserves it [16]. This experiment tests that link
through a controlled decision-level faithfulness test; it does not measure user
preference for the attention map.

Recent XRL work also clarifies which question an explanation answers.
Causal state distillation [62] extends reward decomposition with objectives
for sparse, sufficient and distinct causal factors. This is a model-design
route to explanatory structure, whereas the present study audits a frozen
policy with no explanatory training objective. COViz [63] displays the
outcomes of chosen and alternative actions and evaluates human understanding.
It addresses the consequences of a choice rather than the importance of each
input node. These approaches are complementary: perturbation faithfulness
cannot establish that a dispatcher understands an explanation, and a useful
counterfactual visualization cannot by itself validate raw attention.

## 2.3 Is attention an explanation?

A widely cited challenge to attention-based explanations was presented by Jain
and Wallace [12]. They found that attention weights could be weakly correlated
with other importance measures and that very different attention distributions
could produce similar predictions.
Wiegreffe and Pinter [13] argued that this does not justify rejecting all
attention explanations and proposed more careful tests. Serrano and Smith [14]
likewise showed that removing high-attention items can affect outputs, but that
the relationship varies by model and task. The literature therefore supports
neither a universal acceptance nor a universal rejection of attention. Each
explanation claim needs an explicit definition and an appropriate test.

Attention became widely visible through Transformer models [9], but visibility
also encouraged a broader interpretability claim than the mechanism alone can
support. Lipton [20] warns that the word *interpretability* often combines
several different goals that should be evaluated separately.

Jacovi and Goldberg [17] separate *plausibility*, which concerns whether an
explanation looks reasonable to a person, from *faithfulness*, which concerns
whether it reflects the model behavior that produced the output. This dissertation
follows that distinction. It treats the attention weights as a candidate
explanation and measures their decision relevance rather than accepting them
simply because they are part of the actor.

Layer and head aggregation add another difficulty. A two-layer, multi-head
network does not produce one unique attention map. Averaging heads, selecting a
layer, taking a maximum, or composing attention across layers can produce
different rankings. Abnar and Zuidema [26] propose attention rollout and flow to
account for information mixing across layers, and report stronger correlations
with ablation and gradient importance than raw attention in Transformers.
Although their model is not a GAT-MARL dispatcher, the methodological point
transfers: an attention result is incomplete unless the aggregation rule is
defined and its sensitivity is checked. Shin et al. [46] similarly show that
raw self-attention coefficients can give weak graph attributions when
message-passing paths are ignored.

## 2.4 Explaining graph neural networks

GNN explanations may identify important features, nodes, edges, or subgraphs.
GNNExplainer learns a compact subgraph and feature mask that preserves a
prediction [23]. PGExplainer parameterizes explanation generation so that it
can be shared across examples [24]. SubgraphX searches for important subgraphs
using Shapley-value approximations [36]. These methods show that graph
explanation is a distinct problem: removing one node can alter both its own
features and the messages available to other nodes.

The survey by Yuan et al. [25] organizes GNN explainers by target, mechanism,
and scope. GraphFramEx goes further by comparing explanation methods under
different user needs and combining necessity and sufficiency views [29]. Its
results show that no single method dominates every evaluation dimension. This
supports the use of several checks in the present study: DEF tests
counterfactual relevance, taxi-only rank correlation compares attention with
leave-one-out effects, and stale-attention measures describe freshness
exposure. Azzolin et al. [45] further show that GNN faithfulness metrics are not
interchangeable, reinforcing the need to define what each measure tests.

Most GNN explainability benchmarks study node or graph classification. Fleet
dispatch differs because some graph nodes also define actions. A passenger
request is both information and a selectable action, while a nearby taxi is
context only. Deleting them creates different interventions. This action-linked
graph structure motivates the
construct-validity audit in Section 3.7.

Several 2024-2025 methods offer alternatives to raw attention. GOAt [49]
analytically allocates graph outputs to input features and evaluates fidelity,
discriminability and stability. Its explicit attribution path is attractive,
but transferring it to a residual attention actor requires accounting for the
actual computation and selected output. Bui et al. [50] use the Myerson-Taylor
interaction index to respect graph connectivity and interactions. This exposes
a limitation of a single-node ranking: two jointly relevant nodes need not
have large individual LOO effects. Applying either approach to dispatch would require accounting for request
nodes that also define available actions.

The output type matters as well. RegExplainer [51] adapts graph explanations
to regression through information-bottleneck, mix-up and self-supervised
objectives. It challenges the assumption that a classification-oriented
explanation score transfers unchanged to a continuous value estimate.
Here, the target is the actor's selected action, not the critic's team value;
that distinction determines which output and perturbation to evaluate.

Global explanations answer a different question from the present local audit.
GraphTrail [52] converts predictions into logical rules over mined graph
concepts, while GNNBoundary [67] generates graphs near class boundaries.
Both reveal model-level behavior that a local ranking can miss. However,
global agreement or a plausible boundary graph cannot establish that one
particular dispatch decision used fresh vehicle data. Yu and Gao's MAGE [68]
generates motif-based molecular explanations, emphasizing domain-valid
substructures. Its molecular validity requirements illustrate why a dispatch
explainer likewise needs task-specific constraints. This MAGE is distinct
from the Myerson-Taylor explainer of Bui et al. [50].

GraphNarrator [64] generates textual explanations using saliency-derived
pseudo-labels and iterative refinement. This makes explanations easier to present,
but their quality still depends on the underlying attribution: a
fluent explanation cannot serve as an independent check of the saliency labels
that helped train it. For this dissertation, adding text to an attention map
would therefore require the same freshness and decision-relevance audit.

## 2.5 Evaluating faithfulness: methods and pitfalls

Comprehensiveness and sufficiency are common perturbation measures. In the
ERASER benchmark, comprehensiveness asks how much a prediction weakens when the
explanation is removed, while sufficiency asks how well the selected evidence
alone preserves it [16]. Liu et al. [15] similarly test attention by measuring
faithfulness violations. These measures are useful because they connect an
explanation to model behavior instead of visual appeal.

Model-agnostic methods such as LIME also use local perturbations to estimate
which inputs support a prediction [18]. Their usefulness depends on whether the
perturbed samples remain meaningful for the model and task.

Perturbation is not automatically valid. Removing input features may create
samples unlike the data seen during training. The Remove and Retrain (ROAR)
benchmark shows why deletion can confound attribution quality with distribution
shift [28]. Zheng et al. [43] identify this problem in GNN fidelity measures and
propose more robust alternatives. Adebayo et al. [27] test whether saliency methods depend on learned parameters
and training data by randomizing model parameters and retraining on randomized
labels. Li et al. [44]
show that small graph changes can preserve a prediction while substantially
changing its explanation. Together, these studies show that the evaluator
itself must be audited.

The present action-deletion problem is related but more specific. If a selected
request node is masked, its action logit becomes unavailable. A large margin
change can then occur even if the request features carried no useful evidence.
A random baseline that removes more request nodes than the attention top-k is
not a fair comparison. Type matching, selected-action protection, and clamp
logging are introduced to control this mechanism. A leave-one-out perturbation
ranking acts as a positive control for comprehensiveness, while Gradient x
Input [48] provides a comparator that does not use attention coefficients.

The small graph creates a second limit. When `k = 3` and only a few valid nodes
exist, a random size-three set must overlap substantially with the explanation
set. This compresses the possible difference between them. Reporting the exact
expected overlap and a taxi-only rank correlation makes that limitation
visible instead of treating every near-zero DEF as decisive evidence.

Recent evaluation research provides stronger alternatives to a single fidelity
number. He et al. [65] establish connections between edge-gradient and
perturbation methods, including equivalence under restricted linear settings.
This suggests that agreement between two explainers may reflect shared
mechanics rather than independent validation; their conditions should not be
assumed for the nonlinear actor used here. PowerGraph [66] supplies
cascading-failure explanation ground truth in a domain-specific benchmark.
It demonstrates the value of external explanatory labels, which the present
SUMO experiment lacks: its LOO ranking remains a positive control, not ground
truth. Saha and Bandyopadhyay [55] introduce sufficiency-risk certificates and
coverage, gain and overlap measures for model-level graph explanations.
Their uncertainty-aware evaluation is relevant to audit design, but those
certificates do not automatically apply to a local action-level DEF score.

## 2.6 Age of Information and telemetry degradation

Age of Information (AoI) measures the time since the newest received update
was generated [21]. It is widely used to study communication freshness,
including the relationship between update age, estimation, and control [39].
The explanation question is
different. A dispatch action and its attention map may remain available even
when a vehicle reading has stopped updating. The operator can see a strong
weight without knowing that its source is old.

Delayed-observation reinforcement learning studies how an agent can continue to
act when observations or actions arrive late. Bouteiller et al. [40] model
random action and observation delays and propose a delay-correcting actor-critic
method. Liotet et al. [41] learn a delayed policy from demonstrations produced
without delay. Both studies focus on maintaining decision performance under
delay. They do not test whether an explanation remains faithful when the policy
receives stale telemetry.

Recent delay-aware RL explicitly models uncertainty about the current state.
Yao, Florescu and Lee [58] use stochastic planning for delayed feedback,
whereas Wu et al. [56] directly forecast belief to reduce errors accumulated
by recursive state prediction. Both provide candidate approaches to recovering
useful state information; neither result establishes that an attention ranking
is faithful to the resulting action. Zhou et al. [57] instead make observation
timing part of the action. This separates a deliberately scheduled observation
from an externally triggered outage. The present freeze intervention models
the latter and does not optimize the communication schedule.

Communication robustness also differs from explanation robustness. MAGI [59]
uses a graph information bottleneck to learn compact, action-relevant messages
under perturbations. Soudijani and Dimitrova [60] jointly synthesize action and
communication policies subject to communication restrictions. These methods
address information exchange and task objectives; robust coordination is not
itself an explanation-release criterion. In vehicle perception, MRCNet [61]
addresses motion and sensing noise during communication and fusion. Its setting
motivates testing richer telemetry failures, although perception noise and a
last-value freeze are distinct interventions.

Dynamic graph attribution is especially close to the present problem. Liu and
Xie [53] attribute changes in predictions on evolving graphs to message flows
and layer edges, with KL-divergence used to select explanatory edges. This
supports explicitly comparing two observations rather than treating a static
map as sufficient. Their evolving edge weights differ from the feature-level
freeze with fixed paired node identities here. Liu et al. [54] use graph
curvature and resistance to improve explanation robustness without retraining
the target model. That is a potential improvement to an explainer, whereas
this study measures whether the declared raw-attention channel is stable.
Robustness to structural perturbations does not establish robustness to stale
features, so the two require separate tests.

This dissertation uses AoI only to describe the freshness of each node and as
the weight inside weighted attention mass on stale nodes (WAMSN). The
experiment manipulates the duration of an
observation-layer outage triggered by tunnel entry, not AoI itself. It therefore
makes no causal claim that a larger AoI value lowers faithfulness. The analysis
instead asks whether stale exposure and attention-based explanation behavior
provide consistent evidence under controlled outages.

## 2.7 Deciding when an explanation can be shown

Explanation tests can inform the decision to show an explanation to a user. The NIST AI Risk
Management Framework treats testing, validation, documentation, and monitoring
as parts of managing the risks of AI systems [42]. It provides general risk
management guidance rather than a test for attention explanations.

This dissertation applies that principle to a specific
question: whether an attention map has enough supporting evidence to be
presented for a fleet-dispatch decision. The resulting framework checks the
data status, the validity of the faithfulness test, and consistency across
independently trained policies. It then returns one of three decisions:
*ELIGIBLE*, *WITHHOLD*, or *INCOMPLETE*. These labels and their decision rules
are proposed in this dissertation; they are not prescribed by the NIST
framework. Recent work also shows why such caution is needed: Azzolin et al.
[47] show that a self-explaining GNN can produce an explanation unrelated to its
prediction process and that some metrics may miss this failure. Although their
setting is graph classification, the result supports auditing an explanation
before presenting it as evidence.

## 2.8 Research gap

The literature supports four points: relational RL is appropriate for fleet
dispatch; attention is easy to expose but disputed as an explanation; GNN
faithfulness requires graph-aware perturbations; and stale state matters to
operational control. Recent work examines GNN explanation fidelity, instability,
and auditing [43]-[47], but does not combine MARL fleet dispatch, paired clean
and stale observations, and action-linked request nodes. The present study brings these elements together in a controlled evaluation.

The closest task studies are BMG-Q [33], CoopRide [37] and DualG-MARL [38].
Their dispatch objectives and datasets differ from this controlled benchmark,
so their performance figures are not directly comparable. The closest
methodological studies instead concern attribution paths, fidelity measurement,
explanation instability and changes in graph inputs. Table 2.1 compares eight closely related studies, highlighting their
contributions and the adaptations needed for the dispatch setting.

Across the graph- and RL-explainability and delayed-observation studies reviewed
in Sections 2.2-2.7, no evaluation protocol was identified that combines all of
the following:

- a graph-attention MARL action explained at decision level;
- paired clean and stale versions of the same operational observation;
- explicit measurement of attention placed on stale telemetry;
- a control for graph nodes that also remove available actions;
- replication across independently trained checkpoints.

The proposed audit protocol uses the controlled state available in SUMO [22]
and the attention weights exposed by GAT-MAPPO to test explanation faithfulness
and sensitivity to stale telemetry. It links these measurements to a decision
about whether the evidence is sufficient to present the attention map to an
operator.

| Study | Method and setting | Evaluation and finding | Strength | Limit and implication for this study |
|---|---|---|---|---|
| Zheng et al. [43] | Robust fidelity for subgraph explanations | Synthetic and real graphs; conventional deletion can distort fidelity | Explicit treatment of distribution shift | Does not resolve request-linked action removal; motivates separate action protection |
| Li et al. [44] | Adversarial changes to graph structure | Explanations can change while predictions remain correct | Separates prediction and explanation robustness | Adversarial edges differ from stale vehicle features; requires a paired telemetry test |
| Azzolin et al. [45] | Analysis of regular and self-explainable GNNs | Faithfulness metrics need not agree | Exposes dependence on metric and architecture | Prevents interpreting DEF as a universal or complete causal score |
| Shin et al. [46] | GAtt computation-tree edge attribution | Synthetic and real data; improves on naive attention attribution | Accounts for message-passing paths | Edge attribution needs adaptation to residual actor outputs and action-linked nodes |
| Liu and Xie [53] | Layer-edge attribution on evolving graphs | Eight datasets; evaluates fidelity of prediction changes | Directly studies change between graphs | Evolving edge weights differ from the fixed-identity telemetry twins used here |
| Liu et al. [54] | Curvature and resistance regularization | Nine datasets across three graph tasks; improves robustness | Leaves the target model unchanged | Structural robustness does not establish stale-feature or action-level faithfulness |
| Azzolin et al. [47] | Audit of degenerate self-explanations | Accurate models can yield explanations unrelated to prediction | Shows why accuracy is insufficient for assurance | Self-explainable classifiers differ from raw-attention MARL; motivates an explicit rejection path |
| Saha and Bandyopadhyay [55] | Certified model-level evaluation | Sufficiency risk, coverage, gain and overlap with uncertainty | Evaluates quality beyond target-class score | Model-level certificates are not local DEF guarantees; motivates scoped statistical claims |
