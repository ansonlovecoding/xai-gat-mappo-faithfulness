# 3. Methodology

## 3.1 Study design

The study uses a controlled simulation because the research question requires
the same decision to be observed in two ways: once with current telemetry and
once with degraded telemetry. A real operational log cannot provide this exact
clean twin. SUMO provides the simulated physical state, while a separate
observation layer controls what the policy receives.

The final protocol is defined in
`configs/experiments/dissertation_v4.toml`. The full run contains three stages:
training, validation-based checkpoint selection, and held-out evaluation.
Faithfulness sweeps are run only after the selected checkpoints are frozen.

## 3.2 SUMO environment and data

The road network is an OpenStreetMap extract of the Central Park area in
Yubei, Chongqing. Roads marked as tunnels supply physically meaningful
signal-loss triggers. SUMO simulates 20 taxis and 50 passenger requests during
each 1,200-second episode. Taxi dispatch is controlled through TraCI, while
SUMO continues to update the true position and speed of every vehicle.

Demand variants are separated into training, validation, and test splits. The
policy is fitted on training demand. Candidate checkpoints are compared on
validation demand. The final reported runs use held-out test demand. This
separation prevents test results from influencing model selection.

SUMO is a simulator rather than a neutral source of observed data [22]. Its
outputs depend on the chosen road network, generated demand, routes, vehicle
behaviour parameters, tunnel locations, and random seeds. The experiment is
therefore not described as free from bias. Instead, it uses these assumptions
consistently in a paired design. At each sampled decision, the clean and
degraded observations share the same underlying SUMO state and trained-policy
checkpoint; only the observation-layer telemetry differs. This controls the
comparison well enough to isolate the effect of the implemented degradation
within this scenario, but it does not make the simulated fleet representative
of every real city.

## 3.3 Observation graph and action space

Each idle taxi receives a self-centred graph containing:

| Node type | Main features | Role |
|---|---|---|
| Self taxi | position, time, speed, AoI, position-valid flag | acting vehicle |
| Peer taxi | relative position, availability, distance, AoI | nearby fleet context |
| Passenger request | pickup/drop-off vectors, waiting time | candidate request and action |

The graph contains up to five nearby taxis and five nearby requests. Features
are normalised and request/taxi masks prevent padded nodes from receiving
attention.

The discrete action space contains no-op plus one action for each visible
passenger request. This one-to-one relationship is important: deleting a
passenger-request node also removes the associated action, whereas deleting a
peer-taxi node only hides information about that taxi. Section 3.7 explains
how the faithfulness audit controls this asymmetry.

![Local observation graph with peer taxis, requests and request-to-action mapping](../observation_graph_action_mapping.png)

**Figure 3.1.** Valid self, peer-taxi and request nodes form a fully connected
local graph. The red ring marks a stale peer observation, and the illustrative
edge widths show attention into the acting taxi. Only request nodes map directly
to dispatch actions, which motivates the construct-validity controls.

## 3.4 Policy models and training

The main policy uses per-node-type feature projections followed by two graph
attention layers with four heads. The actor scores no-op and the visible
requests. The critic estimates value for MAPPO training with centralised
training and decentralised execution. The GAT implementation returns its
attention tensors directly so they can be audited without changing the trained
network.

The implementation uses scaled dot-product graph attention on the fully
connected set of valid local nodes. For node `i`, node `j`, layer `l`, and head
`h`:
This differs from the additive attention score in the original GAT paper, so
"GAT" in the experiment labels denotes the graph-attention policy family, not
an exact reproduction of that layer.

```text
q_i = W_Q,h LayerNorm(h_i^l),  k_j = W_K,h LayerNorm(h_j^l)
v_j = W_V,h LayerNorm(h_j^l)
s_ij^(l,h) = (q_i dot k_j) / sqrt(d_head)
alpha_ij^(l,h) = softmax_j(s_ij^(l,h))
z_i = concat_h sum_j alpha_ij^(l,h) v_j
u_i = h_i^l + W_O z_i
h_i^(l+1) = u_i + FFN(LayerNorm(u_i))
```

The mask removes padded nodes before the softmax. A residual connection keeps
the previous node embedding. The actor reads the updated self embedding for
no-op and each updated request embedding for its matching dispatch action.
The default explanation is the self row (`i = 0`) averaged over both layers
and all four heads, then renormalised:

```text
alpha_j = normalise((1 / (L H)) sum_l sum_h alpha_0j^(l,h))
```

This aggregation is declared before evaluation. A sensitivity analysis also
reports each layer, head-wise values, head maximums, and attention rollout; it
does not select an aggregation after seeing which result is favourable.
The explanation average is an analysis choice: inside the policy, head outputs
are concatenated and projected rather than averaged.

The fixed self row represents the acting taxi's dashboard view and is the
explanation channel originally declared for audit. It directly contributes to
the no-op embedding. A request-action logit, however, is read from that
request's updated embedding, so its corresponding request row is more directly
aligned with the selected logit. The revision audit therefore reports a
separate request-action sensitivity comparison between the fixed self row and
the selected request row. This comparison is not used to redefine the primary
metric after seeing the result.

![Simplified graph-attention design](../simplified_graph_attention_design.png)

**Figure 3.2.** The policy and audit share the same two-layer attention encoder.
The audit reduces the self-node attention row to node importance without
modifying the actor, critic or selected action.

Three policy conditions are included:

| Model | Policy | Training observations | Purpose |
|---|---|---|---|
| MLP | MLP-MAPPO | clean | performance context without graph attention |
| GAT | GAT-MAPPO | clean | primary attention model under audit |
| GAT-Outage | GAT-MAPPO | tunnel-triggered 30-second outages | degradation-aware training condition |

Each condition is trained for 150 epochs with seeds 42, 43, and 44.

Candidate checkpoints are saved every ten epochs. Selection uses three
stochastic validation episodes with seed 2026 and mean pickups as the primary
criterion; mean reward and earlier epoch break ties. The test split is not
read during selection. The selected epochs are:

| Model | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| MLP | 50 | 30 | 40 |
| GAT | 10 | 70 | 10 |
| GAT-Outage | 10 | 90 | 10 |

The shared optimisation settings are shown below.

| Setting | Value |
|---|---:|
| Training epochs | 150 |
| Learning rate | 0.0003 |
| Discount factor (`gamma`) | 0.99 |
| GAE factor (`lambda`) | 0.95 |
| PPO clip ratio | 0.20 |
| PPO passes per update | 4 |
| Minibatch size | 256 |
| Value-loss coefficient | 0.50 |
| Entropy coefficient | 0.01 |
| Gradient-norm limit | 0.50 |
| GAT hidden width / layers / heads | 64 / 2 / 4 |

At simulation step `t`, the shared team reward is:

```text
r_t = 10 N_pickup,t + 0.5 N_dispatch,t - 0.001 mean_wait_t
```

The reward trains dispatch behaviour. It contains no term that rewards a
faithful, stable, or human-readable attention map.

![Model conditions and training process](../model_design_training_process.png)

**Figure 3.3.** Common training and validation-based checkpoint selection for
all three model conditions. Frozen checkpoints are evaluated on held-out demand;
the faithfulness audit compares GAT and GAT-Outage.

## 3.5 Telemetry degradation

The trigger and the degradation location are separate parts of the design.

**Step 1 - Trigger:** when a taxi enters a tunnel edge, the event starts one
outage.

**Step 2 - Observation-layer outage:** for the declared duration, the policy
sees the taxi's last valid position and speed. SUMO continues to simulate the
true state.

**Step 3 - Recovery:** after the fixed window expires, a current reading is
accepted. A taxi must leave and enter a tunnel again before another
tunnel-entry event can trigger.

The fixed outage durations are 10, 20, 30, and 60 seconds. They are easy to
monitor and compare, and they create increasing empirical degradation rates.
They are not treated as direct causal doses of explanation faithfulness.

AoI is calculated as the current simulation time minus the time of the last
valid update and is included in each GAT observation. WAMSN uses normalised AoI
as a staleness weight. For this reason, an
increase in WAMSN with outage duration partly verifies that the manipulation
created longer stale exposure; it is not by itself proof that AoI changed DEF.

![Telemetry degradation data flow](../telemetry_degradation_data_flow.png)

**Figure 3.4.** Tunnel entry starts a timed observation-layer outage. SUMO state
continues to advance and receives the policy action; the clean twin is read only
for counterfactual comparison.

## 3.6 Explanation measures

### 3.6.1 Decision-level explanation faithfulness

DEF asks whether the nodes ranked highly by an explanation affect the chosen
action more than a size- and type-matched random set. For top-k nodes
`R_k`, comprehensiveness measures the output change when those nodes are
removed; sufficiency measures the output retained when only those nodes remain.
The two gains are compared with five matched random subsets for each
`k in {1,2,3}` and then averaged.

Let `m(G,a)` be the selected action's logit minus its strongest available
alternative. For explanation subset `R_k` and a matched random subset `B_k`:

```text
Comp(R_k) = m(G,a) - m(G without R_k,a)
Suff(R_k) = m(G,a) - m(G keeping only R_k,a)
g_comp(k) = Comp(R_k) - mean_b Comp(B_k,b)
g_suff(k) = mean_b Suff(B_k,b) - Suff(R_k)
DEF = mean_k 0.5 [g_comp(k) + g_suff(k)]
```

Comprehensiveness is better when removing the explanation weakens the action
more than removing random nodes. Sufficiency is better when keeping the
explanation loses less evidence than keeping random nodes. The logit margin is
clipped to `[-10, 10]` only when an action becomes unavailable or unopposed;
every clamp is logged.

Positive DEF means the explanation is more informative than its random
control; zero means no measured advantage. Both probability DEF and an
action-protected logit-margin variant are recorded. The final interpretation
uses the type-matched baseline described below.

### 3.6.2 Stale-node attention

WAMSN measures attention attached to stale vehicle information:

```text
WAMSN = sum(attention_i * normalised_AoI_i) / sum(attention_i)
```

Only self and peer-taxi nodes contribute because request nodes do not carry
vehicle telemetry. The analysis reports WAMSN when at least one stale vehicle
node is visible, together with stale-node attention share, stale-node mass,
and whether a stale node enters the top three.

The paired stale-attention shift compares the degraded observation with its
exact clean twin at the same simulation step. Positive values mean the
degraded observation assigns more attention mass to nodes marked stale;
negative values mean it assigns less.

## 3.7 Construct-validity audit

The audit supports the primary research problem by checking whether DEF is a
valid comparison in this action-candidate graph.

A uniform random occlusion can select passenger requests more often than the
attention top-k. Removing such a request changes both the observation and the
available action set. A nearby-taxi occlusion changes only information. The
result can therefore reflect different node-type composition rather than
explanation quality.

The corrected protocol uses three controls:

- **type matching:** random subsets contain the same mixture of taxi and
  request nodes as the explanation subset;
- **chosen-action protection:** the request associated with the selected
  action is not deleted in the action-protected variant;
- **clamp logging:** counterfactual action-margin clamps are counted so that
  action deletion can be detected.

![Construct-validity controls for action-linked request nodes](../construct_validity_action_deletion.png)

**Figure 3.5.** Request-node occlusion can remove its action, while peer-taxi
occlusion removes context only. Corrected DEF matches subset size and node types,
protects the chosen request, and logs margin clamps.

The earlier uniform-baseline results are retained only as an audit trail. They
are not used for the final hypothesis verdicts.

Three additional diagnostics test whether a near-zero DEF can be interpreted.
First, a leave-one-out (LOO) control masks each valid node separately and ranks
nodes by the resulting selected-action margin loss. The selected request is
protected. This control is expected to strengthen comprehensiveness because it
is built from the same single-node perturbation, but it is not an independent
proof that the combined DEF is optimal. Second, Gradient x Input ranks each
node by the L2 norm of the selected-action logit gradient multiplied by its raw
features. It is a post-hoc comparator that does not use attention weights.
Third, the audit reports valid taxi/request counts and the exact expected
top-k overlap with a type-matched random subset. Taxi-only Spearman correlation
between attention and LOO effects supplies a ranking measure that does not
delete request actions and does not depend on random subset overlap.

## 3.8 Evaluation matrix

MLP, GAT, and GAT-Outage are evaluated under clean telemetry with eight evaluation
seeds (42-49) and three episodes per seed. This gives 24 held-out episodes per
training seed for performance context.

GAT and GAT-Outage also receive the full faithfulness sweep:

```text
3 training seeds x 5 conditions x 8 evaluation seeds x 3 episodes
```

The five conditions are clean plus four outage durations. Faithfulness is
sampled every eight decisions, with raw per-decision records retained. Each
sweep must pass preflight checks for expected cells, provenance, type-matched
controls, chosen-action exclusion, and empirical degradation differentiation.
All six evaluation sweeps pass.

## 3.9 Hypotheses and statistics

The confirmatory hypotheses are:

- **H1:** DEF decreases as outage duration increases.
- **H2:** faithfulness declines faster than dispatch performance.
- **H3:** WAMSN increases as outage duration increases.
- **H4:** decisions with greater WAMSN have lower DEF within an episode.
- **H5:** degradation-aware training changes or mitigates these relationships.

H1 and H3 use episode-block permutation trend tests and cluster bootstrap
confidence intervals. H2 uses paired cell-level sign flips relative to each
evaluation seed's clean condition. H4 calculates Spearman correlation within
each episode before a sign-flip test. Holm correction is applied to H1-H4
within each trained policy.

H2 is retained as a pre-declared but exploratory comparison. Its original rate
divides by clean DEF, which is close to zero and can make a small absolute
change look arbitrarily large. The analysis therefore reports the support
verdict and absolute trajectories, but does not use the ratio as the main
evidence for faithfulness decoupling.

Thousands of decisions improve measurement precision but do not replace
independent model replication. The final synthesis therefore treats each
training seed as one trained-policy replicate. With only three seeds per model,
the dissertation reports the range and number of supporting seeds and does not
claim a cross-seed population p-value.

The added ranking controls use the same held-out demand and stochastic action
protocol. They evaluate GAT and GAT-Outage under clean telemetry and a 60-second
outage:

```text
2 models x 3 training seeds x 2 conditions x 8 evaluation seeds x 3 episodes
```

LOO, Gradient x Input, overlap, and aggregation diagnostics are sampled every
eight decisions. Query-row sensitivity is evaluated at every selected-request
action because it asks whether the explanation row matches the node that
produces that action logit. Decision records are averaged within episodes
before bootstrap confidence intervals are calculated. Attention-aggregation
sensitivity is reported per training seed and only for decisions where at
least one stale vehicle node is visible.

## 3.10 Reproducibility

The experiment runner records the configuration, commands, seeds, checkpoint
hashes, source revision, and preflight reports under the local
`runs/dissertation_v4/` directory. Compact thesis-ready outputs are committed
under `results/dissertation_v4/`: `summary.csv`,
`performance_context.csv`, and `training_seed_synthesis.csv`. The last file
makes training-seed consistency explicit. Reproduction commands and expected
outputs are documented in `docs/REPRODUCE_EXPERIMENTS.md`.

## 3.11 Ethics and data governance

The experiment uses a simulated taxi fleet, generated passenger demand, and an
OpenStreetMap-derived road network. It contains no human participants,
personal passenger records, live vehicle identifiers, or commercial dispatch
data. The study evaluates model behaviour rather than people. Its main ethical
risk is miscommunication: presenting an attention map as a trustworthy reason
could give an operator false confidence. The reporting therefore separates
freshness exposure, task behaviour, and explanation faithfulness, and states
negative or mixed findings without turning them into stronger causal claims.
