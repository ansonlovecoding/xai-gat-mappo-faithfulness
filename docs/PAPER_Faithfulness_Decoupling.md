# Occlusion Baselines Break When Explanation Units Are Action Candidates: A Construct-Validity Audit in Graph-Attention Fleet Dispatch

Hongwei Lin  
De Montfort University, Dubai  
P2982757@my365.dmu.ac.uk

## Abstract

Perturbation-based faithfulness tests ask whether removing the inputs an
explanation selects changes the model's output, and compare that against
removing random inputs instead. This paper shows that the comparison is
invalid for an entire class of models: those in which the input entities an
explanation ranks are also the candidate actions the policy chooses between.
Occluding such an entity does not merely withhold evidence, it deletes an
option from the decision problem, and a random baseline that hits those
entities at a different rate than the explanation does is measuring
architecture rather than explanation quality.

The audit is carried out on a graph-attention multi-agent dispatcher
(GAT-MAPPO) in a SUMO taxi-dispatch benchmark, where reservation nodes are
simultaneously graph nodes and dispatch actions. Under the standard
uniform-random occlusion protocol, the attention channel scores -0.541
margin-DEF on clean telemetry, an apparently decisive "worse than random"
verdict. Uniform draws clamp the decision margin four times as often as the
attention top-k does. With a composition-matched random baseline the score is
-0.009 [-0.016, -0.001], and an independent clamp-free subset gives +0.000
[-0.003, +0.003]: the channel is uninformative, not adversarial, and 98 % of
the original deficit was protocol. The uniform metric is not merely biased but
unstable, swinging from -1.352 to +1.775 across seven checkpoints spanning a
twelvefold range in task performance, two architectures and three training
seeds — while every trained checkpoint sits within 0.015 of zero under the
fair baseline. Worse, the two checkpoints the uncorrected protocol rates most
favourably are the two best-performing ones, so a practitioner would have
concluded that stronger dispatchers have more faithful attention. A
perturbation test that manufactures both "far worse than random" and "far
better than random" from equally uninformative channels cannot be used to rank
explanations.

The dispatch setting also supplies a substantive result. Tunnel entry
triggers simulated signal loss, so the policy sees a taxi's last valid
position while its Age of Information grows, and attention mass migrates onto
stale vehicle nodes (0 to 0.014 pooled, 13-15 % of vehicle-node attention
conditional on exposure) while pickups and aggregate faithfulness stay flat.
Within episodes, decisions with more stale-node attention are less faithful
(margin-DEF rho = -0.045; probability-DEF rho = -0.149). Neither
degradation-aware training nor an occlusion-distilled decoupled head restores
faithfulness under degradation, and both apparent mitigation effects were
substantially larger before the baseline was corrected. A manipulation check
reports that the intended severity ladder did not vary realised staleness,
because AoI is censored at the observation encoding's normalisation constant
and tunnel transit time dominates the nominal outage; the degradation evidence
is therefore a binary clean-versus-degraded contrast rather than a
dose-response curve, and the preregistered dose-response hypothesis is
reported as untestable rather than unsupported.

**Index Terms** — explainable reinforcement learning, faithfulness
evaluation, construct validity, graph attention, multi-agent reinforcement
learning, fleet dispatch, Age of Information, SUMO.

## I. Introduction

Faithfulness metrics exist because an explanation that looks plausible is not
necessarily an explanation. The dominant family of tests is perturbation
based: take the inputs the explanation ranks highest, remove them, and see
whether the model's output moves. Comprehensiveness and sufficiency in the
ERASER protocol [16] are the canonical instances. Because the raw effect size
depends on how many inputs were removed and how sensitive the model is, the
measurement is always made relative to a control — the same operation applied
to randomly chosen inputs. The explanation is credited with whatever margin
it achieves over random.

That control carries an assumption which is rarely stated: occlusion is
assumed to withhold *evidence*, so that the explanation-selected set and the
random set differ only in which evidence they withhold. This paper documents a
class of models where the assumption fails outright, and where the failure is
severe enough to invert the verdict.

The class is models in which the entities an explanation ranks are also the
candidate actions the policy selects among. Pointer networks, matching and
assignment policies, retrieval-augmented rankers, and action-graph policies
all have this shape. In such a model, occluding a candidate does not merely
hide information about an option — it removes the option. If the chosen action
is deleted, the model's score for that action is not "reduced by the loss of
evidence"; it is undefined, and whatever floor the implementation substitutes
is an artifact of the harness. A random baseline that touches candidate
entities at a different rate than the explanation does will therefore be
compared against a different decision problem, and the resulting "margin over
random" measures architecture, not explanation quality.

The setting used to demonstrate this is fleet dispatch. Fleet dispatch is a
relational decision problem: a taxi should not choose a request by looking
only at itself, but also at nearby taxis and competing requests. Graph
attention networks [7] fit this structure and additionally expose per-entity
weights that are trivial to render as an explanation. In the dispatcher
studied here, each of the five candidate reservations is one node of the
observation graph and one entry of the action space at the same time, so the
pathology is present by construction rather than by contrivance.

The audit finds that the standard protocol reports the attention channel at
-0.541 margin-DEF on clean telemetry — a confident "much worse than random"
result — and that 98 % of that number is composition. Under a random baseline
matched to the node types attention actually selects, the score is -0.009.
The corrected reading is not that attention is adversarial but that it is
uninformative. More damaging for the metric than the bias is its instability:
across seven checkpoints spanning a twelvefold range of task performance, the
uniform score ranges from -1.35 to +1.78 while the matched score stays at
approximately zero throughout.

Because the benchmark was built to study a second question, the paper also
reports it. Fleet telemetry goes stale: a vehicle loses signal in a tunnel,
keeps moving, and the dispatcher retains only its last known position while
its Age of Information (AoI) [21] grows. The concern is that an attention map
can stay tidy and confident while the graph beneath it ages — the explanation
quietly changes what it is about, with no warning in task performance. That
is what happens here. Attention mass migrates onto stale vehicle nodes while
pickups and aggregate faithfulness remain flat, and decisions with more
stale-node attention are measurably less faithful within episodes.

Two things are reported as negative results rather than smoothed over. The
severity ladder the study was designed around did not manipulate realised
staleness, because tunnel transit time dominates the nominal outage duration
and the observation encoding censors AoI at its normalisation constant; the
degradation evidence is therefore a binary contrast, and the preregistered
dose-response hypothesis is untestable on this data rather than unsupported.
And the learned dispatcher is far weaker than a greedy matcher, which is why
the capability-spectrum control matters: the metric pathology is shown to be
invariant across the full range of policy quality, including an untrained
checkpoint.

The contributions are:

1. A construct-validity failure mode for perturbation-based faithfulness
   tests: when explanation units double as action candidates, occlusion
   changes the decision problem and uniform random baselines become invalid.
   Quantified at 98 % of the measured effect, with a composition-matched
   correction and an independent clamp-free control.
2. Evidence that the uncorrected metric is unstable rather than merely
   biased, manufacturing verdicts of both signs from channels that are all
   uninformative under the corrected protocol.
3. A SUMO fleet-dispatch benchmark with tunnel-triggered AoI degradation, in
   which the ground-truth traffic world and the degraded observation given to
   the policy are cleanly separated.
4. Evidence that stale telemetry changes explanation *content* — attention
   migrates onto stale nodes, and stale-node attention predicts lower
   faithfulness — while task performance and aggregate faithfulness give no
   warning.
5. A manipulation check and capability-spectrum control that scope these
   claims honestly, including a preregistered hypothesis reported as
   untestable.

## II. Related Work

### A. Reinforcement Learning for Fleet Dispatch

Reinforcement learning has been widely studied for ride-hailing and fleet
management [1], [2]. Multi-agent methods are natural because many vehicles act
in the same city at the same time, and the standard toolkit spans centralised
critics with decentralised actors [3], value factorisation [4], and on-policy
actor-critic methods [5]. This work uses MAPPO [5] as a practical backbone
with a shared policy across agents. The aim is not to introduce a new dispatch
algorithm: the learned dispatcher is a controlled setting in which the
explanation channel can be audited per decision, and Section IV-A is explicit
that its task performance is well below a greedy matcher.

Graph neural networks [6] are the usual way to encode the relational
structure, and graph attention [7] adds per-entity weights. Attention has also
been used as an architectural device in multi-agent critics [8], following its
role in sequence models [9]. The relevant property here is not performance but
that the weights are *exposed*, and therefore likely to be displayed to an
operator.

### B. Attention as Explanation

Whether attention explains a decision has been debated for several years. Jain
and Wallace argued that attention can be weakly related to feature importance
[12]; Wiegreffe and Pinter replied that this depends on what one asks of an
explanation [13]; Serrano and Smith found attention only partially identifies
the representations that matter [14]. Liu et al. formalised the pattern as a
faithfulness violation test and found violations widespread across
attention-based explainers [15]. Jacovi and Goldberg's framing is adopted
here: faithfulness is a property to be tested, not asserted, and it is
distinct from plausibility [17]. Attention is therefore treated throughout as
a *candidate* explanation.

This debate has been conducted mostly on text classifiers. For graph models
the analogous literature is a family of dedicated explainers — GNNExplainer
[23], PGExplainer [24], and subsequent work surveyed by Yuan et al. [25] —
which exist partly because raw attention was found inadequate. Those methods
are evaluated with the same perturbation logic audited here, and inherit the
same vulnerability whenever the graph entities they rank are also action
candidates. The audit is therefore not specific to attention as an explanation
channel; it applies to the evaluation protocol.

### C. Perturbation-Based Faithfulness and Its Baselines

Comprehensiveness and sufficiency [16] are the standard perturbation tests:
remove the explanation-selected inputs, or keep only those, and measure the
change in model output. Because absolute effect sizes are not comparable
across models, the result is normalised against the same operation on randomly
selected inputs. Related concerns about perturbation-based attribution include
sensitivity to the removal operator [18] and instability under small input
changes [19]; Lipton's broader critique of interpretability as an
underspecified goal [20] applies directly to the practice of reporting a
single faithfulness scalar without validating the metric.

What has received less attention is the *composition* of the random control.
Existing critiques largely concern how occlusion is implemented — masking
versus resampling, in-distribution versus out-of-distribution. The failure
documented in Section IV is different: even with a fixed occlusion operator,
the random control and the explanation can withhold structurally different
things, because in candidate-action architectures some entities carry the
action and others do not. Explainability surveys in reinforcement learning
[10] and causal-lens approaches [11] do not treat this case, and LIME-style
local surrogates [18] would face it too, since perturbing a candidate
perturbs the action set.

### D. Age of Information

Age of Information measures how old a received state update is [21]. It is a
natural description of stale telemetry and is used here in two ways: as the
severity variable for simulated signal loss, and as the node-level freshness
weight inside WAMSN. Section IV-D reports a consequence of the second use that
is easy to overlook — because AoI enters the observation normalised and
clipped, the metric cannot distinguish degrees of staleness beyond the
normalisation constant.

## III. Experimental Design

The experiment has three moving parts: the simulator, the graph-attention
policy, and the degradation layer.

![Telemetry degradation data flow](telemetry_degradation_data_flow.png)

**Fig. 1. Telemetry degradation data flow.** SUMO remains the ground-truth
world. Tunnel edges only trigger signal loss; the degradation itself is
implemented when observations are built for the policy.

### A. SUMO Dispatch Environment

The benchmark uses SUMO [22] on a real OpenStreetMap extract of Central Park
in Chongqing's Yubei district. The area contains real tunnel edges, which are
used as physically grounded signal-loss zones. Each episode has 20 taxis, 50
ride requests, and 1,200 simulated seconds. Demand is generated from fixed
seeds and split into train, validation, and held-out test variants.

Each idle taxi is an agent. At each decision step, the taxi can either do
nothing or accept one of its five nearest pending reservations:

```text
action 0     = no-op
actions 1-5 = accept one candidate reservation
```

### B. Why Several Model Conditions Are Used

The main model is B2, a GAT-MAPPO dispatcher. The other conditions are not
extra models for their own sake. Each one answers a specific control question:
what happens without learning, without graph attention, without AoI features,
or with degradation-aware training?

![Model conditions and training process](model_design_training_process.png)

**Fig. 2. Model conditions and training process.** B2 is the main audited
attention model. B0, B1, B3, and H5' are included to separate task
performance, graph attention, AoI input, and degradation-aware training.

**Table I. Model Conditions**

| Condition | Trained? | Training telemetry | Purpose |
|---|---:|---|---|
| B0 SUMO greedy | No | Not trained | Non-learning dispatch reference |
| B1 MLP-MAPPO | Yes | Clean | RL baseline without graph attention |
| B2 GAT-MAPPO | Yes | Clean | Main attention model under audit |
| B3 GAT-MAPPO without AoI | Yes | Clean | Control for explicit freshness input |
| H5' degradation-aware GAT | Yes | Tunnel-freeze degradation | Test whether degraded training helps |

B1, B2, and B3 are trained on clean telemetry. H5' is trained with
tunnel-triggered freeze degradation. After training, checkpoints are frozen
and evaluated on held-out test demand under clean telemetry and the AoI
severity ladder.

### C. Graph-Attention Policy

For each acting taxi, the environment builds one self-centred graph. The graph
contains the taxi itself, up to five nearby taxis, and up to five candidate
reservations. One graph therefore corresponds to one taxi decision.

![Observation graph for one taxi decision](../results/story_freeze_v1/figs/paper/fig2_observation_graph.png)

**Fig. 3. Observation graph for one dispatch decision.** The policy sees the
acting taxi, nearby taxi context, and candidate reservations in a single small
graph.

![Simplified graph attention design](simplified_graph_attention_design.png)

**Fig. 4. Simplified graph-attention design.** The policy first embeds the
self, taxi, and reservation nodes into a shared space. Valid nodes then attend
to each other. The final embeddings produce dispatch actions, while the
attention map is kept for faithfulness testing.

The node types are:

**Table II. Observation Graph**

| Node type | Main features | Role |
|---|---|---|
| Self taxi | position, time, velocity, AoI | The acting vehicle |
| Peer taxi | relative position, availability, distance, AoI | Local fleet context |
| Reservation | pickup/drop-off vectors, waiting time | Candidate action |

The graph is fully connected over valid nodes. This is manageable because the
graph has at most 11 nodes. Padding masks remove missing neighbours or
reservations from attention and from the action logits.

The GAT uses two layers and four attention heads. The actor reads the final
self embedding to score the no-op action, and the final reservation embeddings
to score reservation actions. The attention tensor is returned by the forward
pass and used as the **coupled explanation channel**.

This design is compact and easy to audit, and it instantiates the structural
property this paper is about:

```text
reservation node <-> reservation action
```

Each reservation node *is* an action. Occluding it therefore removes a
candidate from the action set rather than withholding evidence about it. Peer
taxi nodes carry no action, so occluding one is evidence removal in the
intended sense. A single occlusion protocol thus performs two categorically
different operations depending on which node type it happens to hit, and the
proportion of each type in a subset determines what is being measured. This is
the mechanism audited in Section IV-B.

### D. Telemetry Degradation

Tunnel degradation is implemented as freeze semantics. If a taxi is on a
tunnel edge, the degradation layer keeps returning its last valid position and
speed. AoI grows as:

```text
AoI = current time - last valid update time
```

The SUMO state is not changed. The taxi keeps moving in the simulator, and
dispatch actions still apply to the real SUMO state. Only the policy's
observation is stale.

The intended severity ladder is:

```text
clean, 5 s, 15 s, 30 s, 60 s
```

Each degraded level holds the outage until the taxi's AoI reaches that level.
Two facts about this design turn out to matter, and both are established
empirically in Section IV-D rather than assumed here. First, a taxi still
inside a tunnel when its outage expires immediately re-triggers, so tunnel
transit time places a floor under realised AoI that is independent of the
nominal level. Second, AoI reaches the policy only through the normalised
observation feature `clip(AoI / AOI_MAX_S, 0, 1)` with `AOI_MAX_S = 60 s`, and
WAMSN reuses that same clipped quantity as its staleness weight. Staleness
beyond 60 s is therefore not representable, either to the policy or to the
metric. The manipulation check in Section IV-D shows these two facts combine to
flatten the ladder into a single degraded condition.

### E. Training

All learned models use the same MAPPO training structure. Each epoch runs one
complete SUMO episode, stores every acting taxi's observation, sampled action,
log probability, value estimate, and team reward, then updates the shared
policy with PPO.

The team reward is:

```text
R = 10 * pickups + 0.5 * successful dispatches - 0.001 * mean pending wait
```

Generalised Advantage Estimation uses gamma = 0.99 and lambda = 0.95. PPO
uses a clip ratio of 0.2, four PPO epochs, minibatches of 256, and entropy
regularisation. The best checkpoint is selected by rolling mean training
pickups, because the policies often suffer entropy collapse and the final
epoch is not always the best one.

### F. Faithfulness Metrics

For a decision with selected action `a*`, DEF asks whether the top-k nodes
chosen by the explanation actually matter to the decision.

![DEF occlusion protocol](../results/story_freeze_v1/figs/paper/fig4_def_protocol.png)

**Fig. 5. DEF occlusion protocol.** The same decision is evaluated before and
after removing or keeping the explanation-selected nodes, then compared with a
matched random baseline.

Comprehensiveness removes the explanation nodes:

```text
Comp = f(a* | G) - f(a* | G without R_k)
```

Sufficiency keeps only the explanation nodes:

```text
Suff = f(a* | G) - f(a* | R_k)
```

Both are compared with random subsets of the same size:

```text
DEF = 0.5 * ((Comp - Comp_rand) + (Suff_rand - Suff))
```

Two DEF variants are computed from the same counterfactual forwards.
Probability-DEF uses `f = pi(a*)`. Margin-DEF uses the logit gap between the
chosen action and its closest competitor:

```text
margin = logit(a*) - max logit(other action)
```

**Margin-DEF is the primary metric** throughout, because the trained policies
are softmax-saturated — `pi(a*)` sits near 1 and barely responds to occlusion,
compressing probability-DEF into a range where it carries little signal.
Probability-DEF is reported alongside it wherever the two diverge, and Section
IV-D shows that for one hypothesis they diverge by a factor of three. Margins
are clamped to +/-10 logits, because occluding the chosen action's own node
sends its logit to negative infinity; that clamp is not a numerical detail but
the exact point at which the artifact of Section IV-B enters, and the clamp
rate is reported as a diagnostic.

**Random-baseline composition.** Two sampling schemes are used for the control
subsets. *Uniform* draws size-k subsets uniformly from all valid non-self
nodes; this is the standard protocol and what the original analysis used.
*Type-matched* draws subsets with the same taxi/reservation composition as the
attention top-k set, so the control withholds the same mixture of evidence and
actions that the explanation does. The comparison between the two is the audit.

WAMSN measures how much vehicle-node attention is placed on stale telemetry:

```text
WAMSN = sum_i alpha_i * clip(AoI_i / AoI_max, 0, 1) / sum_i alpha_i
```

Reservations carry no AoI, so WAMSN runs over the self and peer-taxi nodes
only. The clip is the censoring discussed in Section III-D: at
`AoI_max = 60 s` every node past one minute of staleness receives weight
exactly 1.0, so WAMSN is a *stale-attention share* rather than an
AoI-weighted mean whenever outages are long.

### G. Statistical Testing

The main sweep evaluates B2 over clean telemetry and four max-AoI levels, with
eight environment seeds and three held-out test episodes per cell. One in
eight decisions is scored for faithfulness, giving 17,764 scored decisions
(17,444 with at least one valid reservation).

Decisions are clustered inside episodes, which share demand, geography and
policy state, so no test treats decisions as independent:

- H1 and H3: episode-block permutation trend tests, where level labels are
  permuted between whole episodes and never within them.
- H2: paired sign-flip test over severity cells.
- H4: Spearman association computed *inside* each episode and aggregated
  across episodes, with a sign test on the episode-level coefficients. Ties
  are corrected by average ranks; this is not cosmetic, since WAMSN is exactly
  zero for roughly 90 % of decisions and uncorrected ranks reverse the sign of
  the coefficient.
- Confidence intervals: cluster bootstrap resampling episodes, 10,000 draws.
- Holm correction across the H1-H4 family.

Permutation tests use 10,000 draws, so the smallest attainable p-value is
1/10,001. Results at that floor are reported as `p <= 1e-4` rather than as
`p < 0.001`, which would misstate the available resolution.

Two sweeps are referenced and are kept distinct throughout. `B2_aoi_ladder` is
the source of the preregistered H1-H4 statistics. `B2_aoi_ladder_v2` is the
re-run with clamp, exposure and AoI instrumentation added, and is the source of
every diagnostic quantity. Where both are available the numbers agree to
within 0.01 margin-DEF.

## IV. Results

### A. Performance Context

The learned dispatchers are weak compared with SUMO's greedy matcher. The
greedy baseline completes about 32 pickups per episode; the learned policies
complete 6-8, with B2 at 6.7 +/- 1.9. The training runs suffer entropy
collapse, and two of three H5' seeds reach their best rolling performance at
epochs 17-22.

This is stated first because it bounds every claim that follows. Nothing here
is evidence for a deployable dispatcher. It matters less for the
methodological result than it might appear, however, and Section IV-C
establishes why: the metric pathology is demonstrated across seven checkpoints
spanning 1.0 to 11.8 pickups per episode, including an untrained one, so it is
not a property of weak policies.

**Table III. Performance Context**

| Policy | Role | Pickups per episode |
|---|---|---:|
| SUMO greedy | Non-learning reference | ~32 |
| B2 GAT-MAPPO | Main audited policy | 6.7 +/- 1.9 |
| Learned-policy range | B1/B2/B3/H5' | 6-8 |
| Capability spectrum (Table V) | Seven audited checkpoints | 1.0-11.8 |

### B. The Occlusion Baseline Measures Architecture, Not Explanation

Under the standard uniform random baseline, B2's clean-telemetry margin-DEF is
-0.541 [-0.556, -0.527]. Read at face value this is a decisive negative
result: the attention channel appears to be substantially *worse* than
choosing nodes at random, which would imply it actively points away from what
matters.

The diagnostic that undermines this reading is the clamp rate — the fraction of
counterfactual forwards in which the decision margin hits its cap because the
occlusion deleted the chosen action. Uniform draws clamp at 14.8 %; attention
top-k sets clamp at 2.2 % (per-level rates in the instrumented sweep are 16.4 %
and 4.0 %). Uniform sampling reaches reservation nodes far more often than
attention does, and every such draw scores the baseline on a decision problem
from which an option has been removed. The two arms of the comparison are not
evaluating the same model.

Correcting the composition removes almost all of the effect. With a random
baseline matched to the taxi/reservation mixture of the attention top-k set,
clean margin-DEF is **-0.009 [-0.016, -0.001]** over 1,199 paired decisions,
against -0.554 for the uniform baseline on the same decisions: **98.3 % of the
apparent deficit is protocol artifact.**

Two independent controls agree with the corrected value:

- **Clamp-free subset.** Restricting to decisions where neither the attention
  top-k nor any random draw clamped removes the action-deletion mechanism by
  construction, with no modelling assumption. Margin-DEF is **+0.000
  [-0.003, +0.003]** (n = 90). The subset is small and selected — clamp-free
  decisions have fewer competing reservations — so it corroborates rather than
  replaces the type-matched estimate.
- **Chosen-action exclusion variant.** Protecting the chosen reservation's node
  from occlusion on both sides turns an apparent +0.586 into -0.024
  [-0.025, -0.023] on the decisions where it is evaluable (n = 136 of 17,444;
  the variant only applies when the chosen action is a reservation). The small
  n is why this is a supporting rather than a headline control.

The corrected claim is therefore not that attention is anti-faithful but that
it is **uninformative**: statistically indistinguishable from, and marginally
below, a composition-matched random explanation. An occlusion-based test that
returned "much worse than random" for an uninformative channel was measuring
the coupling between nodes and actions.

![Clean-telemetry faithfulness audit](../results/story_freeze_v1/figs/story_act1_clean_def.png)

**Fig. 6. Clean-data faithfulness audit.** The apparent worse-than-random
result collapses to near zero when the random baseline is matched to the node
types selected by attention.

**Table IV. Construct-Validity Audit**

| Quantity | Value | n | Interpretation |
|---|---:|---:|---|
| Uniform clean margin-DEF | -0.541 [-0.556, -0.527] | 3,565 | Apparent worse-than-random result |
| Paired uniform baseline | -0.554 [-0.581, -0.529] | 1,199 | Same decisions, before correction |
| **Type-matched baseline** | **-0.009 [-0.016, -0.001]** | 1,199 | Corrected: uninformative, not adversarial |
| Artifact share | 98.3 % | — | Fraction of the deficit that is protocol |
| Clamp-free subset | +0.000 [-0.003, +0.003] | 90 | Independent control, no modelling assumption |
| Chosen-action exclusion variant | -0.024 [-0.025, -0.023] | 136 | Apparent +0.586 on the same decisions |
| Clamp rate, random baseline | 14.8 % (16.4 % per level) | 17,444 | Random occlusion deletes actions often |
| Clamp rate, attention top-k | 2.2 % (4.0 % per level) | 17,444 | Attention rarely selects the chosen action's node |

### C. The Uncorrected Metric Is Unstable, Not Merely Biased

A biased metric that erred consistently could still rank explanations. This one
cannot. Seven checkpoints were scored under both baselines on clean test
demand, spanning 1.0 to 11.8 pickups per episode, policy entropy 1.56 down to
0.03, two architectures, and three independent H5' training seeds.

**Table V. Capability Spectrum Under Both Baselines** (clean test demand)

| Checkpoint | Epoch | Pickups | Entropy | n | Uniform margin-DEF | Type-matched margin-DEF |
|---|---:|---:|---:|---:|---:|---:|
| B2 untrained | 0 | 1.00 | 1.560 | 56 | -1.352 | +0.158 [-0.120, +0.509] |
| B2 mid-training | 50 | 6.25 | 0.040 | 678 | -0.468 | +0.001 [-0.007, +0.013] |
| B2 best | 91 | 7.50 | 0.062 | 641 | -0.549 | -0.003 [-0.013, +0.012] |
| B3 best (no AoI input) | 25 | 9.25 | 0.087 | 537 | **+1.687** | -0.005 [-0.005, -0.004] |
| H5' seed 42 best | 17 | 7.25 | 0.061 | 578 | -0.802 | +0.015 [+0.013, +0.016] |
| H5' seed 43 best | 91 | 3.00 | 0.035 | 654 | -0.488 | -0.007 [-0.009, -0.006] |
| H5' seed 44 best | 22 | 11.75 | 0.103 | 552 | **+1.775** | +0.007 [+0.006, +0.007] |

Under the uniform baseline the score ranges from -1.352 to +1.775 — from "far
worse than random" to "far better than random". Under the type-matched
baseline the six trained checkpoints span -0.007 to +0.015, all within 0.015
of zero despite a fourfold spread in task performance. The untrained
checkpoint reads +0.158, but on only 56 scored decisions with a confidence
interval spanning zero, so it is uninformative rather than anomalous.

The uncorrected metric therefore manufactures verdicts of both signs from
channels that are indistinguishable from a fair random baseline, and the sign
it produces tracks the reservation composition of the decisions rather than
any property of the explanation. Note in particular that B3 and H5' seed 44 —
the two checkpoints scoring strongly *positive* under the uniform protocol —
are also the two with the highest pickup counts. A practitioner using the
standard protocol would have concluded that better dispatchers have more
faithful attention, and that conclusion is an artifact.

This is the strongest single result in the paper, for three reasons. It shows
the pathology is not an artifact of one weak checkpoint, which answers the
external-validity objection raised by Section IV-A. It spans two
architectures and three training seeds, so it is not a seed effect. And it
establishes that no threshold or sign convention can rescue the uncorrected
metric: a measurement that returns +1.78 and -1.35 for the same underlying
quality is not a measurement.

### D. Under Degradation, Attention Content Moves Toward Stale Nodes

Before reporting the degradation hypotheses, the manipulation must be checked,
because the check changes what can be concluded.

**The severity ladder did not vary realised staleness.** AoI reaches both the
policy and WAMSN only through `clip(AoI / 60 s, 0, 1)`, so staleness past 60 s
is not representable. Independently, a taxi still inside a tunnel when its
outage expires re-triggers immediately, so transit time places a floor under
realised AoI. The two effects combine:

**Table VI. Manipulation Check: Realised AoI by Nominal Level**

| Nominal level | Realised p50 | Realised p90 | At censoring cap | n exposed |
|---:|---:|---:|---:|---:|
| 5 s | 60.0 s | 60.0 s | 78.3 % | 345 |
| 15 s | 60.0 s | 60.0 s | 84.9 % | 416 |
| 30 s | 60.0 s | 60.0 s | 81.1 % | 349 |
| 60 s | 60.0 s | 60.0 s | 79.4 % | 388 |

Median and 90th percentile are identical at every level, and 81 % of exposed
decisions sit at the cap. The `{5, 15, 30, 60}` s ladder collapses to a single
degraded condition.

**Consequence for H1.** The preregistered hypothesis — that faithfulness
declines as severity rises — is **untestable on this data**, not unsupported.
The trend test returns rho = -0.001 (p = 0.87), but a null result against an
unmanipulated factor is uninformative: there was no severity gradient for the
metric to respond to. The earlier reading, that flat DEF reflected a floor
effect from the near-zero clean baseline, is superseded; near zero is the fair
random level, not a floor, and margin-DEF had ample room to move downward. What
the data supports is the binary clean-versus-degraded contrast, which is well
powered.

**H3, restated.** WAMSN rises from exactly 0 on clean telemetry to 0.014
pooled the moment staleness exists. On the preregistered sweep the
episode-block trend is rho = +0.092, p <= 1e-4, cluster CI [+0.052, +0.128]
(instrumented sweep: +0.081, p <= 1e-4). Within the degraded conditions,
however, there is no trend at all: rho = +0.002, p = 0.88, n = 14,154. The
entire association is the clean-versus-degraded step. The supported claim is
therefore **WAMSN rises when staleness appears and is flat thereafter** —
consistent with Table VI, and narrower than "severity increases WAMSN".

**Exposure.** The pooled WAMSN is small because most decisions never see a
stale node, and two distinct exposure rates were previously conflated:

| Quantity | Value | Meaning |
|---|---:|---|
| Acting-agent degradation rate | ~2 % | the deciding taxi's own telemetry is stale |
| Stale-node visibility | 9.8-11.7 % per level | the decision's graph contains a stale vehicle node |
| Stale node in attention top-3, given exposure | 7.7-9.3 % | a stale node reached the displayed explanation |
| WAMSN given exposure | 13.4-14.8 % | share of vehicle-node attention on stale telemetry |

Conditional on exposure is the informative framing: when a stale vehicle node
is visible at all, it takes roughly one seventh of the attention the policy
distributes over vehicles.

**H4 is supported, and is metric-dependent.** Within episodes, decisions with
more stale-node attention are less faithful. On the primary metric,
margin-DEF, mean within-episode rho = **-0.045**, negative in 64.6 % of the 96
usable episodes (`B2_aoi_ladder`: -0.043, 74.0 %). On probability-DEF the same
estimator gives rho = **-0.149**, negative in 92.7 % of episodes
(`B2_aoi_ladder`: -0.140, 88.5 %). Both replicate across sweeps, and the
preregistered one-sided sign-flip test reaches its resolution floor
(p <= 1e-4) for both; an exact two-sided sign test on the episode-level
coefficients gives p = 0.006 for margin-DEF and p < 1e-5 for probability-DEF.

The direction is robust, but the effect is roughly three times larger on the
secondary metric. The margin-DEF value is quoted as primary for consistency
with Section III-F. The divergence is itself interpretable: probability-DEF
responds mainly to decisions where occlusion collapses a saturated softmax,
which is exactly where stale peer nodes are most likely to be attended, so it
registers a stronger association than the margin does.

**H2 is the weakest result.** Faithfulness-degradation rates exceeded
performance-degradation rates across 32 cells: Holm-adjusted p = 0.039 on
probability-DEF (raw p = 0.0195; raw p = 0.028 on margin-DEF). Given that the
severity factor was not manipulated, this rate framing should be read as a
clean-versus-degraded difference and not as a dose-response. H3 and H4 carry
the degradation argument.

![Stale-attention drift under degradation](../results/story_freeze_v1/figs/story_act2_decoupling.png)

**Fig. 7. Silent stale-attention drift.** Margin-DEF and pickups stay flat
across the nominal severity ladder, while WAMSN rises the moment staleness is
introduced. Per Table VI the ladder is a single degraded condition, so the flat
profiles should be read as clean versus degraded rather than as a
dose-response.

**Table VII. B2 Severity Sweep**

Margin-DEF is reported under the **uniform** baseline, which Section IV-B shows
is artifact-dominated. The type-matched control has been run on the clean and
tunnel-60 s conditions only; extending it across the ladder requires re-running
the sweep with fresh counterfactual forwards, and the values below cannot be
corrected post hoc. They are included for the WAMSN and pickup columns and for
continuity with the preregistered analysis, not as faithfulness levels.

Margin-DEF, pickups and WAMSN are cell means over 8 seeds from
`B2_aoi_ladder`; stale-node visibility and conditional WAMSN come from the
instrumented `B2_aoi_ladder_v2`.

| Max AoI (s) | Margin-DEF (uniform, artifact-prone) | Pickups | WAMSN | Stale-node visibility | WAMSN given exposure |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.542 | 6.67 | 0.000 | 0.0 % | — |
| 5 | -0.546 | 7.42 | 0.015 | 9.8 % | 0.139 |
| 15 | -0.544 | 6.79 | 0.012 | 11.7 % | 0.148 |
| 30 | -0.541 | 6.54 | 0.013 | 10.0 % | 0.140 |
| 60 | -0.542 | 7.42 | 0.013 | 10.8 % | 0.134 |

**Table VIII. Hypothesis Outcomes**

| Hypothesis | Result | Reading |
|---|---|---|
| H1: severity lowers DEF | **Untestable** | Severity not manipulated (Table VI); rho = -0.001, p = 0.87 against an unmanipulated factor |
| H2: faithfulness drops faster than performance | Supported cautiously | Holm p = 0.039 (prob-DEF); clean-vs-degraded, not dose-response |
| H3: staleness raises WAMSN | Supported, restated | Step at onset (rho = +0.092, p <= 1e-4); no trend within degraded levels (rho = +0.002, p = 0.88) |
| H4: WAMSN relates negatively to DEF | Supported | margin-DEF rho = -0.045 (64.6 % of episodes); prob-DEF rho = -0.149 (92.7 %) |
| H5: degradation-aware training helps | No effect | H5' ~ B2 ~ 0 under the fair baseline, three seeds |

### E. Mitigation Does Not Restore Faithfulness

Two mitigations were tested, one on the training side and one on the
architecture side. Both illustrate how much the corrected baseline changes
conclusions.

**Degradation-aware training (H5').** B2's exact configuration retrained with
tunnel-freeze degradation active reaches 7.80 +/- 1.33 pickups on clean test
demand, so degraded training costs nothing in performance. Under the uniform
baseline it appeared *worse* than B2 at every level (-0.77 versus -0.54), which
was originally reported as evidence that degraded training harms
interpretability. Under the type-matched baseline that finding disappears: H5'
sits at approximately zero across three training seeds (+0.02, -0.01, +0.01 in
Table V), indistinguishable from B2. The earlier conclusion was
artifact-on-artifact, and it is a second independent demonstration of the
instability documented in Section IV-C. H5 is rejected in its weakest and most
defensible form: **degradation-aware training has no detectable effect on
explanation faithfulness in either direction.**

**Decoupled explanation head.** An occlusion-distilled explainer trained
directly to predict occlusion effects (validation Spearman +0.60) was compared
against coupled attention, paired per decision. Under the uniform baseline it
appeared to gain +0.107 on clean telemetry and +0.113 under tunnel degradation.
Under the type-matched baseline roughly 80 % of that advantage was artifact.
What survives is real but modest: on clean telemetry the decoupled head scores
+0.010 against coupled attention's -0.013, a paired gain of +0.023 (n = 983,
p <= 1e-4), and it is the **only channel tested that scores above the fair
random baseline**. Under tunnel degradation at max-AoI 60 s the gain falls to
+0.005 and is not detectable (n = 878, p = 0.21).

![Coupled versus decoupled explanation channel](../results/story_freeze_v1/figs/story_act3b_coupled_vs_decoupled.png)

**Fig. 8. Coupled vs decoupled explanation.** Training an explanation head
directly helps on clean telemetry, but the improvement is small and does not
survive degradation.

**Table IX. Mitigation Results (type-matched baseline)**

| Condition | Coupled attention | Mitigated channel | Delta | n | p |
|---|---:|---:|---:|---:|---:|
| H5' degradation-aware training, clean | ~0 | ~0 | ~0 | 3 seeds | no effect |
| Decoupled head, clean | -0.013 | **+0.010** | **+0.023** | 983 | <= 1e-4 |
| Decoupled head, tunnel 60 s | -0.006 | -0.000 | +0.005 | 878 | 0.21 |

For reference, the same comparisons under the uniform baseline gave +0.107
(clean) and +0.113 (tunnel), both at p <= 1e-4 — a mitigation effect five times
larger and apparently robust to degradation, entirely from the artifact.

## V. Discussion

### A. When Is a Perturbation Test Valid?

The condition this paper identifies can be stated compactly. A perturbation
test compares `f` evaluated on a graph with the explanation's nodes removed
against `f` evaluated on the same graph with random nodes removed. That
comparison is only interpretable if both operations leave the *decision
problem* intact and alter only the evidence available within it. When an
entity's presence determines whether an action exists, removing it changes the
problem, and the two arms are no longer comparable.

Three practical consequences follow.

First, **the random baseline must be matched on whatever structural property
distinguishes evidence removal from problem modification** — here, node type.
Matching on subset size alone, which is the standard practice, is insufficient.

Second, **implementations should report the rate at which perturbation produces
an undefined score.** The clamp rate was the diagnostic that made the artifact
visible: 14.8 % versus 2.2 % is a large asymmetry that no summary faithfulness
statistic would have revealed. Any harness that substitutes a floor value for
an undefined model output should instrument how often it does so.

Third, **a faithfulness score should not be trusted without a stability check
across models of differing quality.** The capability spectrum in Section IV-C
took seven checkpoints to expose a metric that produced -1.35 and +1.78 for
equally uninformative channels. A single number on a single checkpoint would
have looked like a finding either way.

The affected class is not exotic. Pointer networks select over input elements;
retrieval-augmented rankers score the passages they are conditioned on;
matching, assignment, and scheduling policies choose among the entities in
their observation; graph-based action spaces are common in operations research
applications of reinforcement learning. Dedicated graph explainers [23], [24],
[25] are evaluated with the same protocol and inherit the same exposure.

### B. What the Degradation Study Shows

The degradation result is narrower than originally framed, and more specific.
It is not that staleness makes explanations measurably worse — the aggregate
faithfulness score does not move. It is that the explanation's *content*
relocates onto stale data while every available warning signal stays flat.
Pickups do not change, aggregate margin-DEF does not change, and the attention
map remains a well-formed distribution over plausible entities. Meanwhile,
conditional on a stale vehicle node being visible, roughly one seventh of
vehicle-node attention lands on it, and decisions where that share is higher
are measurably less faithful.

This is the operationally dangerous configuration, and it is worth being
precise about why. An operator monitoring task performance sees nothing. An
operator monitoring a faithfulness metric sees nothing. Only an instrument that
asks *what the explanation is about* — rather than how good it is — registers
the change. WAMSN is such an instrument, and the argument for pairing a
content metric with a quality metric does not depend on this paper's
particular architecture.

The construct also needs restating more carefully than in the original framing.
"Faithfulness decoupling" suggests faithfulness diverging from task
performance; what was measured is the explanation's content diverging from the
validity of the data underneath it, with neither performance nor faithfulness
registering the divergence. The latter is the claim the data supports.

### C. Reporting Negative Structure Honestly

Two findings in this paper are failures of the study design rather than
findings about the world, and both were surfaced by post hoc checks rather than
by the preregistered analysis.

The severity ladder did not manipulate severity, because a normalisation
constant chosen for the observation encoding silently became the ceiling of the
measurement scale, and because geography dominated the nominal treatment. The
lesson generalises: when a metric reuses a clipped model input as its own
weighting, the metric inherits the clipping, and any manipulation that pushes
past the clip becomes invisible. A manipulation check on the *realised* value
of the treatment variable — not the nominal one — would have caught this before
the sweep was run.

The H4 effect size was quoted from probability-DEF in a paper whose stated
primary metric was margin-DEF, overstating it threefold. The two variants are
computed from the same forwards at no extra cost, which makes reporting both
the obvious default; and where they disagree, the disagreement is itself
information about how the occlusion operator interacts with policy saturation.

## VI. Limitations

**The severity ladder is not a severity ladder.** Table VI establishes that
realised AoI is identical at every nominal level. Preregistered H1 is untestable
here, and H2's rate framing should be read as a binary contrast. Recovering a
genuine dose-response requires raising `AOI_MAX_S` above the geographic AoI
floor and reporting realised rather than nominal severity, which needs a new
sweep.

**The severity-ladder faithfulness column is uncorrected.** The type-matched
control has been run on the clean and tunnel-60 s conditions; the intermediate
cells have not. Type-matched DEF requires fresh counterfactual forwards with
composition-matched random sets and cannot be recovered from recorded
per-decision scalars. Two offline substitutes were attempted: the clamp-free
subset agrees with the type-matched value but covers only 0.5 % of decisions,
and a clamp-differential covariate adjustment closes only 0.04 of the 0.53 gap,
confirming the artifact is not linear in the clamp differential. Table VII's
margin-DEF column therefore remains the artifact-prone metric and is labelled
as such.

**The learned policy is far below the greedy baseline.** Claims are scoped to
the learned-policy regime achieved. The capability spectrum in Section IV-C
mitigates this specifically for the metric result, which holds across 1.0 to
11.8 pickups and an untrained checkpoint, but not for the degradation result,
which rests on B2.

**Training instability.** Entropy collapses from 1.56 to 0.03, and best
checkpoints land at epochs 17-22 for two of three H5' seeds. Checkpoint
selection by rolling mean pickups is a workaround, not a fix; the reward scale
(`10 x pickups` against `0.001 x mean wait`) leaves the waiting-time term
numerically inert between sparse pickup events, which is the likely cause.

**One training seed for the main policy.** The severity sweeps use eight
environment seeds and H5' uses three training seeds, but B2 itself is one run.

**Sparse and geographically constrained exposure.** Stale-node visibility is
9.8-11.7 % of decisions, and in-tunnel transits dominate. Pooled WAMSN values
are correspondingly small, so conditional-on-exposure figures are the
informative ones.

**One district, one occlusion operator.** The benchmark is 20 taxis, 50
requests and 1,200 s episodes in one district of one city. Faithfulness is
measured through masking-based occlusion only; gradient and surrogate
attributions may show different failure modes, though any method evaluated by
perturbation inherits the audit's concern.

**Table X. Limitations and Impact**

| Limitation | Impact |
|---|---|
| Severity not manipulated (AoI censoring + geography) | H1 untestable; degradation claims are binary clean-vs-degraded |
| Ladder faithfulness column uncorrected | Table VII margin-DEF is artifact-prone; needs a type-matched re-run |
| Learned policy below greedy baseline | Degradation claims scoped to low-capability dispatch; metric claims are capability-invariant |
| Entropy collapse in training | Best-checkpoint selection is a workaround; reward scaling likely at fault |
| One main B2 training seed | Stronger training replication is future work |
| Sparse tunnel exposure | Conditional WAMSN is the informative statistic, not pooled |
| One city district | Multi-city generality remains open |
| Masking-based occlusion only | Other attribution operators should be tested |

## VII. Conclusion

The primary result of this paper is about measurement. When the input entities
an explanation ranks are also the actions a policy chooses between, occlusion
does not withhold evidence — it deletes options, and a random baseline that
does so at a different rate is not a control. In the dispatcher studied here
that mechanism accounted for 98 % of an apparently decisive "worse than random"
faithfulness verdict, and it rendered the uncorrected metric unstable rather
than merely biased: across seven checkpoints spanning a twelvefold range of
task performance it produced scores from -1.35 to +1.78 for channels that are
all uninformative under a composition-matched baseline. It also manufactured
two mitigation findings that did not survive correction — that
degradation-aware training harms interpretability, and that a decoupled
explanation head gives a large and degradation-robust improvement. The remedy
is inexpensive: match the random baseline on the structural property that
distinguishes evidence from action, and report the rate at which perturbation
leaves the model's output undefined.

The corrected substantive picture is that the coupled attention channel carries
no measurable decision-relevant information on clean telemetry, and that
staleness changes what the explanation is about rather than how good it scores.
Attention migrates onto stale vehicle nodes — about one seventh of vehicle-node
attention wherever a stale node is visible — while pickups and aggregate
faithfulness stay flat, and decisions with more stale-node attention are less
faithful within episodes. Degradation-aware training has no effect in either
direction, and an occlusion-distilled decoupled head is the only channel to
score above the fair random baseline, by +0.023 on clean telemetry and not
detectably under degradation. Faithful explanation for fleet dispatch under
stale telemetry remains open.

The practical conclusion has two parts. Attention maps should not be treated as
explanations merely because the architecture exposes them. And faithfulness
metrics should not be treated as verdicts merely because they produce a number:
in this study the metric needed auditing more urgently than the explanation
did.

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

[23] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec,
"GNNExplainer: Generating explanations for graph neural networks," in
*Advances in Neural Information Processing Systems*, 2019, pp. 9240-9251.

[24] D. Luo et al., "Parameterized explainer for graph neural network," in
*Advances in Neural Information Processing Systems*, 2020, pp. 19620-19631.

[25] H. Yuan, H. Yu, S. Gui, and S. Ji, "Explainability in graph neural
networks: A taxonomic survey," *IEEE Transactions on Pattern Analysis and
Machine Intelligence*, vol. 45, no. 5, pp. 5782-5799, 2023.
