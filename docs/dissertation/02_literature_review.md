# 2. Literature review

This work sits at the intersection of three literatures that have not
previously met: reinforcement learning for fleet dispatch, the
faithfulness of attention-based explanations, and the Age of
Information. Each is reviewed for what it contributes and for the
assumption this dissertation removes.

## 2.1 Reinforcement learning for fleet dispatch

Lin et al. framed city-scale fleet management as a contextual
multi-agent RL problem and demonstrated strong gains on real ride-hailing
data [1]; surveys confirm RL as the state of the art for ridesharing
operations [2]. Because dispatch couples agents through shared roads
and demand, cooperative MARL methods dominate: MADDPG [3], value
factorisation in QMIX [4], and — the backbone adopted here — MAPPO,
whose combination of parameter sharing and centralised training /
decentralised execution (CTDE) is simple and empirically strong [5].

Two formulations recur. *Centralised* dispatch computes a global
vehicle–order matching each step; *decentralised* dispatch lets each
idle vehicle choose independently whether to accept a nearby order
[1]. This work adopts the decentralised formulation for a reason
specific to its explanation focus (elaborated in §3.2): a per-agent
graph makes every attention distribution correspond one-to-one to a
single vehicle's decision, so faithfulness can be evaluated per
decision rather than on an aggregate dispatcher output.

Graph structure enters through graph neural networks [6]; graph
*attention* networks [7] additionally learn a per-node weighting that
attention-based MARL such as MAAC exploits for inter-agent reasoning
[8]. It is exactly this weighting that platforms surface as an
explanation — the object under audit here.

Recent dispatch studies reinforce the relevance of this setting. BMG-Q uses a
local bipartite match graph for ride-pooling dispatch (Hu, Feng and Li, 2025),
CoopRide studies cooperative city-scale MARL (Wang et al., 2025), and
DualG-MARL combines vehicle-state and task graphs for ride-sharing scheduling
(Sha et al., 2026). These studies focus on dispatch quality and coordination;
they do not test whether an attention channel remains faithful when its
vehicle telemetry is stale.

## 2.2 Is attention an explanation?

The debate opened by "Attention is not Explanation" [12] established
that attention maps can be uncorrelated with feature importance and can
be substituted without behavioural change. Wiegreffe and Pinter's reply
[13] and Serrano and Smith's analysis [14] refined rather than reversed
the verdict: whether attention "explains" depends on the definition and
the test. Jacovi and Goldberg's position piece [17] supplied the
discipline this dissertation adopts wholesale — faithfulness must be
*defined and measured*, never assumed, and the measurement must itself
be scrutinised.

The dominant measurement family is perturbation-based:
comprehensiveness and sufficiency in the ERASER benchmark [16], the
faithfulness-violation test for attention models [15], and
reference-attribution methods such as LIME [18]. Robustness studies
warn that interpretability methods are themselves fragile [19], and
Lipton's survey cautions against conflating plausibility with validity
[20].

Two gaps matter here. First, the entire debate operates on clean,
static inputs; no prior work asks what happens to faithfulness when the
*data behind the explanation* ages. Second — a gap this work discovered
rather than inherited — perturbation protocols implicitly assume that
occluding an explanation unit leaves the model's output space intact.
In RL architectures where explanation nodes double as action
candidates, that assumption fails: occlusion deletes actions. The
resulting baseline confound (Chapter 4) is closest in spirit to the
critique that removal-based evaluation puts inputs off-distribution
(the ROAR line of argument), but is more severe: the perturbation
changes the *decision problem*, not merely the input distribution. To
the author's knowledge no prior faithfulness study controls for it.

## 2.3 Age of Information

The AoI literature [21] formalises data freshness as time since last update and
studies its cost for *control* — how stale state degrades estimation and
decision quality. AoI has rarely been coupled to *explanation* quality. This
dissertation uses AoI as a descriptor of each frozen vehicle reading and as the
per-node staleness weight inside WAMSN (§3.6). The manipulated variable is the
fixed observation-layer outage duration, not AoI itself; the study therefore
does not claim that AoI causes a change in faithfulness.

## 2.4 Simulation platform

SUMO [22] provides the microscopic traffic substrate: a real
OpenStreetMap district with tunnel attributes preserved, a taxi
dispatch device driven through TraCI, and reproducible demand
generation. SUMO is an experimental vehicle here, not a contribution;
the same applies to the GAT-MAPPO dispatcher itself.

## 2.5 Positioning

Relative to [12] and its successors, this work contributes (i) a temporal axis
based on fixed observation outages and measured data freshness; (ii) the
closed-loop MARL setting, where explanations attach to individual dispatch
decisions taken under degraded observations; (iii) a paired clean/degraded
evaluation design with exact per-decision counterfactuals; and (iv) the
candidate-action occlusion artifact and its controls. The closest robustness
studies perturb model inputs [19], but do not examine whether explanation
behaviour remains stable as operational telemetry becomes stale.
