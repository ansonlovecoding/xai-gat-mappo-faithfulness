# When Explanations Outlive Their Data: Faithfulness Decoupling in Graph-Attention MARL Fleet Dispatch under Telemetry Degradation

Hongwei Lin  
De Montfort University, Dubai  
P2982757@my365.dmu.ac.uk

## Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) is a natural
architecture for fleet dispatch because dispatch decisions depend on nearby
vehicles, pending requests, and shared road structure. Its attention weights
are also tempting to expose as built-in explanations. This paper studies
whether such explanations remain faithful when operational telemetry becomes
stale. We build a SUMO-based dispatch benchmark on a tunnel-rich road network
in Chongqing, model tunnel signal loss as freeze-semantics Age-of-Information
(AoI) degradation at the observation boundary, and evaluate attention
explanations per decision using occlusion-based Dispatch Explanation
Faithfulness (DEF) and Weighted Attention Mass on Stale Nodes (WAMSN).

The central result is negative but instructive. Under the original uniform
random occlusion baseline, clean-telemetry margin-DEF appears strongly worse
than random (−0.541). A construct-validity audit shows that 98% of this
deficit is an artefact: in candidate-action dispatch architectures, occluding
a reservation node deletes the corresponding action, so uniform random
baselines perturb the decision problem differently from attention-selected
nodes. With a composition-matched random baseline, attention is near-zero
faithful (−0.009), not meaningfully anti-faithful. Under telemetry degradation,
attention mass shifts toward stale vehicle nodes (WAMSN 0→0.013, robust
p < 0.001), and within episodes the decisions attending more to stale nodes
are less faithful, while both task performance and DEF remain flat. Degradation-aware
training does not improve explanation faithfulness, and an occlusion-distilled
decoupled explanation head gives only a modest clean-data improvement
(+0.023 margin-DEF, p < 0.001) that is not detectable under degradation.
The findings suggest that attention explanations in this setting do not merely
degrade under stale telemetry; they are unfaithful from the start, and
staleness silently changes their content.

**Index Terms** — explainable reinforcement learning, graph attention,
multi-agent reinforcement learning, fleet dispatch, Age of Information,
faithfulness, SUMO.

## I. Introduction

Fleet-dispatch systems increasingly rely on learned policies that coordinate
many vehicles over spatial road networks and shared demand. Graph neural
networks are a natural fit for this setting, and graph attention networks
(GATs) are especially attractive because they produce per-decision attention
weights over neighbouring vehicles and candidate requests. In practice, these
weights can be presented to operators as explanations: they appear to indicate
which nearby agents or orders drove a dispatch decision.

The difficulty is that operational telemetry is not always fresh. A vehicle
entering a tunnel, underground road segment, or urban signal shadow may stop
transmitting GPS and connectivity updates. Downstream systems continue to act
on the last known state while its Age of Information (AoI) grows. If an
attention map still looks plausible while the data behind it has aged, the
explanation may outlive its data: it can give operators a false sense of
accountability precisely when connectivity is degraded.

This paper asks: **do attention-based explanations in GAT-MARL fleet dispatch
remain faithful under telemetry degradation?** We formalise this risk as
*faithfulness decoupling*: the possibility that explanation faithfulness
changes, or becomes unreliable, before task performance reveals a problem.
We evaluate this in a controlled simulation where degradation is injected only
at the observation boundary, so the simulator remains ground truth and every
degraded observation has a clean twin.

The work makes four contributions:

1. A reproducible SUMO/GAT-MAPPO dispatch benchmark with physically grounded
   tunnel-triggered AoI degradation.
2. A per-decision evaluation protocol combining DEF, a perturbation-based
   faithfulness score, with WAMSN, a graded measure of attention placed on
   stale telemetry.
3. A construct-validity audit showing that uniform occlusion baselines are
   confounded in candidate-action architectures because occluding a candidate
   request can delete an action.
4. Empirical evidence that the coupled attention channel is practically
   uninformative, shifts silently toward stale telemetry under degradation,
   and is not repaired by degradation-aware training or a simple decoupled
   explanation head.

![Fig. 1. System architecture: SUMO ground truth, freeze-semantics degradation at the observation boundary, per-agent graph observations, coupled attention explanations, and decoupled distilled explanations.](../results/story_freeze_v1/figs/paper/fig1_architecture.png)

**Fig. 1. System architecture.** Telemetry degradation is injected only at
the observation boundary, so the simulator remains ground truth and every
degraded observation has an exact clean twin for paired evaluation.

## II. Related Work

### A. Multi-Agent Reinforcement Learning for Fleet Dispatch

Fleet management and ride-hailing dispatch have been formulated as
large-scale reinforcement learning problems in which agents coordinate through
spatial demand and road constraints [1], [2]. Cooperative MARL methods such as
MADDPG [3], QMIX [4], and MAPPO [5] provide standard training frameworks.
Graph neural networks encode relational structure in road and vehicle systems
[6], while graph attention networks [7] and attention-based MARL architectures
[8] provide learned weights over neighbouring entities. This paper does not
claim a new dispatch algorithm; the GAT-MAPPO policy is an experimental
vehicle for studying explanation faithfulness.

### B. Attention and Faithfulness

Whether attention is a faithful explanation is contested. Jain and Wallace
showed that attention can be uncorrelated with feature importance and can be
altered without changing predictions [12]. Subsequent work argued that the
answer depends on definition and test design [13], [14], [17]. Perturbation
metrics such as comprehensiveness and sufficiency, popularised in ERASER [16],
evaluate whether removing or retaining explanation-selected inputs changes
the model output. These methods are useful but assume the perturbation keeps
the output space intact. In candidate-action RL, that assumption can fail:
an explanation unit may also be an action candidate.

### C. Age of Information and Telemetry Staleness

AoI measures the freshness of information as the time since the last update
[21]. It is widely used to study control and communication systems, but rarely
connected to explanation quality. Here AoI is both the degradation severity
axis and the per-node staleness signal used to quantify whether an explanation
allocates attention to stale telemetry.

## III. Methodology

### A. Simulation Environment

We use SUMO [22] to simulate a real OpenStreetMap extract of the Central Park
area of Chongqing's Yubei district. The area contains real tunnel edges, which
serve as physically grounded signal-loss zones. The scenario contains 20 taxis
and 50 ride requests per 1,200 s episode. Demand is generated from fixed seeds;
twenty demand variants are split into training, validation, and held-out test
sets. All headline evaluations use held-out test demand.

**Table I. Experimental Setup**

| Component | Setting |
|---|---|
| Simulator | SUMO microscopic traffic simulation |
| Map | Central Park, Yubei district, Chongqing, from OpenStreetMap |
| Fleet and demand | 20 taxis, 50 ride requests, 1,200 s episode |
| Demand protocol | 20 seeded variants, chronological train/val/test split |
| Observation | Self + 5 peer taxis + 5 candidate reservations |
| Action space | No-op or accept one of 5 nearest reservations |
| Policy | Shared GAT-MAPPO, 2 GAT layers, 4 heads, centralised critic |
| Degradation | Tunnel-triggered freeze semantics at observation boundary |
| Severity ladder | Clean + max-AoI {5, 15, 30, 60} s |
| Main sweep | 8 environment seeds x 3 held-out test episodes per cell |
| Faithfulness sampling | 1 in 8 decisions, approximately 18k scored decisions |

Each idle taxi observes a self-centric heterogeneous graph consisting of:

- a self node with position, episode time, velocity, and AoI;
- five nearest peer-taxi nodes with relative position, availability,
  distance, and AoI;
- five nearest reservation nodes with relative pickup/drop-off vectors and
  waiting time.

The action space is `Discrete(K+1)`: no-op or accept one of the five candidate
reservations. This node-action mapping is central to the faithfulness audit:
reservation nodes are explanation units and action candidates at the same
time.

![Fig. 2. Per-agent heterogeneous graph. Reservation nodes correspond directly to dispatch actions, which is useful for interpretability but creates an occlusion artefact.](../results/story_freeze_v1/figs/paper/fig2_observation_graph.png)

**Fig. 2. Per-agent observation graph.** Each idle taxi receives a
self-centred graph over itself, nearby taxis, and candidate reservations.
Reservation nodes map one-to-one to actions, which later makes naive
occlusion ill-posed.

### B. GAT-MAPPO Dispatcher

The policy uses parameter sharing across taxis, per-type input projections,
two graph-attention layers with four heads, an actor over the dispatch action
space, and a centralised critic for training. Training uses PPO clipping,
GAE, and a team reward:

```text
R = 10 * pickups + 0.5 * successful_dispatches - 0.001 * mean_pending_wait.
```

All evaluations are stochastic rather than argmax. The trained policies
exhibit entropy collapse, and argmax evaluation can degenerate into all-no-op
behaviour. Checkpoints are selected using rolling mean training pickups.

### C. Freeze-Semantics Telemetry Degradation

Degradation is applied at the observation boundary only. The underlying SUMO
state remains ground truth. When a vehicle is on a tunnel edge, its telemetry
freezes at the last valid position and speed; all observers see the frozen
reading while its AoI grows. Severity is controlled by maximum outage duration
`s ∈ {5, 15, 30, 60}` seconds, interpreted as receiver re-acquisition delay.
The clean condition `s = 0` is the shared reference.

Because degradation never mutates the simulator, each degraded observation has
an exact clean twin generated at the same simulation step. This enables paired
attention-drift and faithfulness analyses.

![Fig. 3. Freeze-semantics degradation. While a taxi is under signal loss, observers continue to see its last valid state while AoI grows.](../results/story_freeze_v1/figs/paper/fig3_freeze_timeline.png)

**Fig. 3. Freeze-semantics degradation.** A vehicle entering a tunnel stops
transmitting. The last valid state is served to all observers until signal
re-acquisition, while AoI records how stale that reading has become.

### D. Faithfulness Metrics

For a decision with chosen action `a*` and explanation-selected top-k nodes
`R_k`, comprehensiveness and sufficiency are:

```text
Comp = f(a* | G) - f(a* | G \ R_k)
Suff = f(a* | G) - f(a* | R_k)
```

They are normalised against size-matched random explanations:

```text
g_comp = Comp - Comp_rand
g_suff = Suff_rand - Suff
DEF = 0.5 * (g_comp + g_suff)
```

The proposal-defined output readout is action probability `π(a*)`. Because
trained policies saturate, we additionally report **margin-DEF**, where
`f(a*) = logit(a*) - max logit(other)`, clamped to ±10. Margin-DEF uses the
same counterfactual forward passes but preserves more signal under entropy
collapse.

WAMSN measures the fraction of vehicle-node attention assigned to stale
telemetry:

```text
WAMSN = sum_i alpha_i * (AoI_i / AoI_max) / sum_i alpha_i,
```

over self and peer-taxi nodes. Higher WAMSN means the explanation places more
attention on stale vehicle states.

### E. Construct-Validity Audit

A naive occlusion protocol is confounded in this architecture. If a reservation
node is occluded, the corresponding action may be deleted. If the chosen
reservation is deleted, the decision margin clamps mechanically, regardless of
the node's informational relevance.

We therefore audit DEF using three controls:

1. **Clamp audit**: record how often counterfactual margins hit the clamp for
   attention top-k and random-baseline occlusions.
2. **Chosen-action protection**: recompute DEF while preventing the chosen
   reservation node from being removed.
3. **Type-matched random baseline**: draw random explanations with the same
   taxi/reservation composition as the attention-selected top-k nodes.

Headline faithfulness claims use the type-matched baseline; uniform-baseline
results are retained to show the artefact.

![Fig. 4. DEF protocol and occlusion/action-deletion artefact. Uniform random occlusion baselines hit reservation nodes more often than attention top-k, so they delete actions at a different rate.](../results/story_freeze_v1/figs/paper/fig4_def_protocol.png)

**Fig. 4. Occlusion/action-deletion artefact.** In candidate-action
architectures, removing an explanation unit can remove an action from the
decision problem. Type-matched random baselines control this by matching the
taxi/reservation composition of the explanation's top-k nodes.

### F. Experimental Design and Statistics

The main sweep evaluates the B2 GAT-MAPPO checkpoint across clean telemetry
and four max-AoI levels, using 8 environment seeds and 3 held-out test-demand
episodes per cell. One decision in eight is scored for faithfulness, yielding
approximately 18,000 scored decisions per sweep.

We test:

- H1: increasing severity reduces DEF;
- H2: faithfulness degrades faster than task performance;
- H3: increasing severity increases WAMSN;
- H4: WAMSN is negatively associated with DEF;
- H5: degradation-aware training changes the faithfulness trajectory.

Because decisions are clustered within episodes and seeds, the confirmatory
analysis uses episode-block permutation tests for H1/H3, within-episode
association for H4, and cell-level paired sign-flip tests for H2. Holm
correction is applied across H1-H4.

We also evaluate two mitigations: degradation-aware training (H5′) and a
decoupled explanation head distilled from single-node occlusion targets over
detached policy embeddings.

**Table II. Hypotheses and Operational Tests**

| Hypothesis | Operational test | Primary statistic |
|---|---|---|
| H1: severity reduces faithfulness | Severity vs DEF/margin-DEF trend | Episode-block permutation Spearman |
| H2: faithfulness declines faster than performance | Faithfulness-rate minus performance-rate per severity cell | Paired sign-flip over 32 cells |
| H3: severity increases stale-node attention | Severity vs WAMSN trend | Episode-block permutation Spearman |
| H4: stale attention associates with lower faithfulness | WAMSN vs DEF within episodes | Within-episode Spearman + sign-flip |
| H5: telemetry-aware training changes faithfulness | B2 vs H5′ severity profiles | Corrected type-matched DEF comparison |

## IV. Results

### A. Dispatch Performance Context

The learned policies are not competitive with SUMO's greedy dispatch baseline.
Under stochastic evaluation on held-out test demand, the greedy matcher
completes approximately 32 pickups per episode, while learned policies
complete 6-8 pickups (B2: 6.7 ± 1.9). The following results should therefore
be interpreted as faithfulness behaviour in a low-capability learned-dispatch
regime, not as evidence for a deployable dispatcher.

**Table III. Dispatch Performance Context**

| Policy / condition | Role | Pickups per episode | Interpretation |
|---|---|---:|---|
| SUMO greedy matcher | Non-learned reference | ~32 | Strong dispatch baseline |
| B2 GAT-MAPPO | Main audited policy | 6.7 ± 1.9 | Low-capability learned policy |
| Learned policies overall | B1/B2/B3/H5′ range | 6-8 | Faithfulness claims scoped to this regime |

### B. Act 1: Coupled Attention Is Practically Uninformative

Under the original uniform-baseline protocol, clean-telemetry margin-DEF for
the B2 policy is −0.541 (95% CI [−0.556, −0.527], n = 3,565), apparently far
worse than random. The audit shows this interpretation is wrong. Random
baseline counterfactuals hit the decision-margin clamp much more often than
attention top-k occlusions (14.6% vs 1.8%), because random subsets select
reservation nodes more often and therefore delete actions.

With a type-matched baseline, clean margin-DEF moves from −0.554 to −0.009
(95% CI [−0.016, −0.001], n = 1,199 paired decisions). Thus 98% of the
apparent deficit is attributable to the occlusion/action-deletion artefact.
The corrected reading is that the attention channel is near-zero faithful and
practically uninformative, marginally below a fair random explanation.

**Table IV. Construct-Validity Audit of Clean Margin-DEF**

| Quantity | Uniform baseline | Corrected / audit value | Reading |
|---|---:|---:|---|
| Clean margin-DEF, B2 | −0.541 [−0.556, −0.527] | - | Apparent worse-than-random result |
| Paired clean margin-DEF | −0.554 | −0.009 [−0.016, −0.001] | 98% of deficit is artefact |
| Clamp rate | 14.6% random | 1.8% attention top-k | Random baseline deletes actions more often |
| Chosen-action protection | +0.586 | −0.024 | Artefact can manufacture either sign |

![Fig. 5. Clean-telemetry DEF under uniform and type-matched baselines. The apparent worse-than-random result collapses to near-zero under the corrected baseline.](../results/story_freeze_v1/figs/story_act1_clean_def.png)

**Fig. 5. Act 1: clean-data faithfulness.** The built-in attention channel
appears worse than random under the uniform baseline, but the corrected
type-matched baseline shows it is practically uninformative.

The artefact is not a one-off. Across seven checkpoints spanning pickups
1→11.8 and entropy 1.56→0.03, the uniform baseline swings from −1.35 to
+1.78, while the type-matched baseline keeps every checkpoint near zero
(−0.01 to +0.02). Uniform occlusion can therefore manufacture both
"worse-than-random" and "better-than-random" explanations in this architecture.

![Fig. 6. Capability spectrum. Uniform-baseline margin-DEF swings from strongly negative to strongly positive across checkpoints; type-matched margin-DEF remains near zero.](../results/story_freeze_v1/figs/paper/fig6_capability_spectrum.png)

**Fig. 6. Capability spectrum.** Across seven checkpoints, the uniform
baseline produces sign-flipping artefacts, while the type-matched baseline
keeps every checkpoint near zero.

### C. Act 2: Attention Shifts Silently Toward Stale Telemetry

Across the max-AoI ladder, H1 is not supported: margin-DEF remains flat at
its clean-data level. This is a floor effect; the channel starts near zero, so
there is little faithful signal to lose.

H3 is supported. WAMSN rises from 0 in the clean condition to approximately
0.013 as soon as staleness exists (ρ = +0.092, episode-block permutation
p < 0.001, cluster CI [+0.052, +0.128]). Although pooled WAMSN is small
because tunnel exposure is geographically limited, the conditional effect is
operationally meaningful: among decisions with stale vehicle nodes visible,
WAMSN averages 0.140 and a stale node enters the top-3 explanation in 8.5% of
exposed decisions.

H4 is also supported. Within episodes, the WAMSN-DEF correlation is negative
in 89% of episodes, with mean within-episode ρ = −0.140 and p < 0.001. In
plain terms, decisions whose attention rests more on stale nodes are less
faithful, even within the same episode.

H2 is supported under the rate operationalisation but should be read
cautiously. Cell-level faithfulness-degradation rates exceed
performance-degradation rates (n = 32, Holm-corrected p = 0.039), while
completed pickups remain statistically flat across severities (6.5-7.4 per
episode). The strongest evidence for the paper's mechanism is therefore H3/H4,
not H2 alone.

**Table V. B2 Severity Sweep Outcomes**

| Max AoI (s) | Margin-DEF | Pickups | WAMSN |
|---:|---:|---:|---:|
| 0 | −0.541 | 6.67 | 0.000 |
| 5 | −0.546 | 7.42 | 0.015 |
| 15 | −0.544 | 6.79 | 0.012 |
| 30 | −0.541 | 6.54 | 0.013 |
| 60 | −0.542 | 7.42 | 0.013 |

**Table VI. Hypothesis Results**

| Hypothesis | Verdict | Key result |
|---|---|---|
| H1 | Not supported | Margin-DEF flat; cluster CI for ρ [−0.016, +0.025] |
| H2 | Supported cautiously | Holm-corrected p = 0.039 over 32 cells |
| H3 | Supported | WAMSN severity trend ρ = +0.092, p < 0.001 |
| H4 | Supported | 89% of episodes negative; mean ρ = −0.140, p < 0.001 |
| H5 | No mitigation effect | H5′ remains near-zero under corrected baseline |

![Fig. 7. Severity sweep. Margin-DEF and pickups remain flat, while WAMSN increases as soon as staleness is introduced.](../results/story_freeze_v1/figs/story_act2_decoupling.png)

**Fig. 7. Act 2: silent stale-attention drift.** Faithfulness and task
performance remain flat, while attention mass shifts onto stale telemetry.

### D. Act 3: Mitigation Attempts Do Not Restore Faithfulness

The degradation-aware H5′ policy is trained with tunnel-triggered freeze
degradation at outage 30 s. Under the corrected type-matched baseline, H5′
policies remain near zero margin-DEF, matching B2. Across three H5′ seeds,
degradation-aware training shows no detectable faithfulness benefit or harm.

The decoupled explanation head performs better, but only modestly. On clean
telemetry, coupled attention scores −0.013 while the decoupled head scores
+0.010, a paired gain of +0.023 margin-DEF (n = 983, p < 0.001). Under
tunnel degradation at max-AoI 60 s, the gain falls to +0.005 and is not
statistically detectable (n = 878, p = 0.21). Directly optimising an
explanation channel helps more than trusting attention, but it does not solve
faithfulness under degradation.

**Table VII. Mitigation Results under Type-Matched DEF**

| Mitigation | Condition | Coupled / B2 | Mitigated | Delta | p |
|---|---|---:|---:|---:|---:|
| Degradation-aware training H5′ | Clean and severity ladder | ≈0 | ≈0 | ≈0 | no effect |
| Decoupled head | Clean | −0.013 | +0.010 | +0.023 | <0.001 |
| Decoupled head | Tunnel, max-AoI 60 s | −0.006 | −0.000 | +0.005 | 0.21 |

![Fig. 8. Training-side mitigation. H5′ degradation-aware training does not restore faithfulness under the corrected interpretation.](../results/story_freeze_v1/figs/story_act3a_training_mitigation.png)

**Fig. 8. Act 3a: training-side mitigation.** Training with tunnel-triggered
freeze degradation does not make the coupled attention channel faithful.

![Fig. 9. Coupled vs decoupled explanation. The decoupled head improves clean-data margin-DEF modestly, but not under tunnel degradation.](../results/story_freeze_v1/figs/story_act3b_coupled_vs_decoupled.png)

**Fig. 9. Act 3b: coupled vs decoupled explanation.** Occlusion distillation
creates the only above-random explanation channel in the study, but the effect
is small and disappears under degradation.

## V. Discussion

The original hypothesis expected a faithful explanation channel to lose
faithfulness as telemetry aged. The evidence points to a sharper conclusion:
the coupled attention channel is not meaningfully faithful even before
degradation. Staleness does not so much degrade a reliable explanation as
silently alter the content of an unreliable one.

This matters operationally. A dashboard built from attention weights could
continue to display plausible-looking explanations while more of the attention
mass moves onto stale telemetry. Neither task performance nor DEF changes
enough to alert the operator. WAMSN is therefore useful not because it proves
attention is faithful, but because it can monitor a concrete failure mode:
how much of the explanation is being spent on stale state.

The construct-validity audit is the most transferable methodological result.
In candidate-action architectures, perturbation-based faithfulness evaluation
can change the decision problem itself. This applies beyond fleet dispatch to
pointer networks, assignment policies, action-graph policies, and other RL
systems where input entities index possible actions. Composition-matched
random baselines and chosen-action protection are cheap controls and should
be standard in such settings.

![Fig. 10. Audited causal chain. The AoI to stale-attention link is supported, while DEF decline is blocked by a floor effect.](../results/story_freeze_v1/figs/paper/fig5_causal_chain.png)

**Fig. 10. Audited causal chain.** The proposed chain holds through AoI and
WAMSN, but the expected DEF decline fails because the clean explanation is
already at the floor.

## VI. Limitations

This study has several limitations.

**Table VIII. Main Limitations and Effect on Claims**

| Limitation | Effect on interpretation |
|---|---|
| Learned policy far below greedy baseline | Claims apply to the achieved low-capability learned-dispatch regime |
| One city district and sparse tunnel exposure | WAMSN pooled magnitudes are small; conditional exposure statistics matter |
| One main B2 training seed | Capability-spectrum audit helps, but stronger policies remain future work |
| One decoupled-head run | Mitigation result should be read as preliminary |
| Occlusion-only faithfulness | Other attribution methods may expose different behaviour |
| Frozen summaries archived without raw sweep cells | Full raw-cell re-analysis requires rerunning sweeps from manifests |

First, policy competence is low. The learned dispatcher achieves 6-8 pickups
against 32 for SUMO greedy. The capability-spectrum audit shows that attention
uninformativeness is stable across the checkpoints available, but it does not
prove that a strong GAT dispatcher would behave identically.

Second, scale is limited: one city district, 20 taxis, 50 requests, and
1,200 s episodes. Tunnel exposure is also sparse, around 2% of decisions.

Third, the main B2 policy is one training seed, although H5′ is evaluated
across three training seeds and the main sweeps use eight environment seeds.
The decoupled head is one distillation run.

Fourth, faithfulness is evaluated through occlusion. Although the audit
controls the main candidate-action artefact, other attribution families such
as gradients or causal interventions may expose different behaviour.

Finally, the frozen result summaries are archived, but full raw-cell
re-analysis requires rerunning the sweep from the recorded manifests and
checkpoints.

## VII. Conclusion

This paper evaluated whether attention explanations in GAT-MARL fleet
dispatch remain faithful under AoI-based telemetry degradation. In the studied
system, the answer is largely no. The coupled attention channel is practically
uninformative on clean telemetry once the occlusion/action-deletion artefact is
controlled. Under tunnel-triggered staleness, attention shifts toward stale
vehicle nodes, and decisions that attend more to stale nodes are less
faithful, while task performance and aggregate faithfulness stay flat.
Degradation-aware training does not repair the channel, and a decoupled
occlusion-distilled head improves clean-data faithfulness only modestly.

The central lesson is methodological as much as empirical: explanation
faithfulness must be audited in the architecture where explanations are used.
In candidate-action RL, naive occlusion baselines can certify uninformative
channels as either harmful or helpful. Once controlled, attention here is not
an explanation that degrades under stale data; it is an explanation-shaped
signal whose content can drift under stale data without becoming trustworthy.

## References

[1] K. Lin, R. Zhao, Z. Xu, and J. Zhou, "Efficient large-scale fleet
management via multi-agent deep reinforcement learning," in *Proc. ACM SIGKDD
Int. Conf. Knowledge Discovery and Data Mining*, 2018, pp. 1774-1783.

[2] Z. Qin, H. Zhu, and J. Ye, "Reinforcement learning for ridesharing: An
extended survey," *Transportation Research Part C: Emerging Technologies*,
vol. 144, 2022.

[3] R. Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
Environments," in *Advances in Neural Information Processing Systems*, 2017.

[4] T. Rashid et al., "Monotonic value function factorisation for deep
multi-agent reinforcement learning," *Journal of Machine Learning Research*,
vol. 21, no. 178, pp. 1-51, 2020.

[5] C. Yu et al., "The surprising effectiveness of PPO in cooperative
multi-agent games," in *Advances in Neural Information Processing Systems*,
2022.

[6] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini,
"The graph neural network model," *IEEE Transactions on Neural Networks*,
vol. 20, no. 1, pp. 61-80, 2009.

[7] P. Velickovic et al., "Graph Attention Networks," in *Proc. International
Conference on Learning Representations*, 2018.

[8] S. Iqbal and F. Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement
Learning," in *Proc. International Conference on Machine Learning*, 2019.

[9] A. Vaswani et al., "Attention is all you need," in *Advances in Neural
Information Processing Systems*, 2017.

[10] A. Heuillet, F. Couthouis, and N. Diaz-Rodriguez, "Explainability in deep
reinforcement learning," *Knowledge-Based Systems*, vol. 214, 2021.

[11] P. Madumal, T. Miller, L. Sonenberg, and F. Vetere, "Explainable
reinforcement learning through a causal lens," in *Proc. AAAI Conference on
Artificial Intelligence*, 2020.

[12] S. Jain and B. C. Wallace, "Attention is not Explanation," in *Proc.
NAACL-HLT*, 2019, pp. 3543-3556.

[13] S. Wiegreffe and Y. Pinter, "Attention is not not Explanation," in *Proc.
EMNLP-IJCNLP*, 2019, pp. 11-20.

[14] S. Serrano and N. A. Smith, "Is attention interpretable?" in *Proc. ACL*,
2019, pp. 2931-2951.

[15] Y. Liu et al., "Rethinking attention-model explainability through
faithfulness violation test," in *Proc. ICML*, 2022.

[16] J. DeYoung et al., "ERASER: A benchmark to evaluate rationalized NLP
models," in *Proc. ACL*, 2020, pp. 4443-4458.

[17] A. Jacovi and Y. Goldberg, "Towards faithfully interpretable NLP systems:
How should we define and evaluate faithfulness?" in *Proc. ACL*, 2020,
pp. 4198-4205.

[18] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?':
Explaining the predictions of any classifier," in *Proc. ACM SIGKDD*, 2016,
pp. 1135-1144.

[19] D. Alvarez-Melis and T. S. Jaakkola, "On the robustness of interpretability
methods," arXiv:1806.08049, 2018.

[20] Z. C. Lipton, "The mythos of model interpretability," *Queue*, vol. 16,
no. 3, pp. 31-57, 2018.

[21] S. Kaul, R. Yates, and M. Gruteser, "Real-time status: How often should
one update?" in *Proc. IEEE INFOCOM*, 2012, pp. 2731-2735.

[22] P. A. Lopez et al., "Microscopic traffic simulation using SUMO," in
*Proc. IEEE International Conference on Intelligent Transportation Systems*,
2018, pp. 2575-2582.
