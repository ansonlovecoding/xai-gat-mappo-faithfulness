# 3. Methodology

## 3.1 Study design

The study uses a controlled simulation because the research question requires
the same decision to be observed in two ways: once with current telemetry and
once with degraded telemetry. A real operational log cannot provide this exact
clean twin. SUMO provides the physical ground truth, while a separate
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

## 3.3 Observation graph and action space

Each idle taxi receives a self-centred graph containing:

| Node type | Main features | Role |
|---|---|---|
| Self taxi | position, time, speed, AoI | acting vehicle |
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

## 3.4 Policy models and training

The main policy uses per-node-type feature projections followed by two graph
attention layers with four heads. The actor scores no-op and the visible
requests. The critic estimates value for MAPPO training with centralised
training and decentralised execution. The GAT implementation returns its
attention tensors directly so they can be audited without changing the trained
network.

Four policy conditions are included:

| ID | Policy | Training observations | Purpose |
|---|---|---|---|
| B1 | MLP-MAPPO | clean | performance context without graph attention |
| B2 | GAT-MAPPO | clean | primary attention model under audit |
| B3 | GAT-MAPPO without AoI input | clean | structural AoI-input control |
| H5 | GAT-MAPPO | tunnel-triggered degradation | degradation-aware training condition |

Each condition is trained for 150 epochs with seeds 42, 43, and 44. B2 and B3
produce identical clean-test behaviour because AoI is always zero during clean
training and clean evaluation; B3 is useful only when observations can become
stale. It should not be described as evidence that AoI has no effect.

Candidate checkpoints are saved every ten epochs. Selection uses three
stochastic validation episodes with seed 2026 and mean pickups as the primary
criterion; mean reward and earlier epoch break ties. The test split is not
read during selection. The selected epochs are:

| Model | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| B1 | 50 | 30 | 40 |
| B2 | 10 | 70 | 10 |
| B3 | 10 | 70 | 10 |
| H5 | 10 | 90 | 10 |

## 3.5 Telemetry degradation

The trigger and the degradation location are separate parts of the design.

1. **Trigger:** when a taxi enters a tunnel edge, the event starts one outage.
2. **Observation-layer outage:** for the declared duration, the policy sees the
   taxi's last valid position and speed. SUMO continues to simulate the true
   state.
3. **Recovery:** after the fixed window expires, a current reading is accepted.
   A taxi must leave and enter a tunnel again before another tunnel-entry event
   can trigger.

The fixed outage durations are 10, 20, 30, and 60 seconds. They are easy to
monitor and compare, and they create increasing empirical degradation rates.
They are not treated as direct causal doses of explanation faithfulness.

AoI is calculated as the current simulation time minus the time of the last
valid update. It is included in the GAT observation unless the B3 control is
used. WAMSN uses normalised AoI as a staleness weight. For this reason, an
increase in WAMSN with outage duration partly verifies that the manipulation
created longer stale exposure; it is not by itself proof that AoI changed DEF.

![Telemetry degradation data flow](../telemetry_degradation_data_flow.png)

**Figure 3.1.** Tunnel entry supplies the trigger, but frozen telemetry is
created at the observation boundary. SUMO ground truth remains unchanged.

## 3.6 Explanation measures

### 3.6.1 Decision-level explanation faithfulness

DEF asks whether the nodes ranked highly by an explanation affect the chosen
action more than a size- and type-matched random set. For top-k nodes
`R_k`, comprehensiveness measures the output change when those nodes are
removed; sufficiency measures the output retained when only those nodes remain.
The two gains are compared with five matched random subsets for each
`k in {1,2,3}` and then averaged.

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

The earlier uniform-baseline results are retained only as an audit trail. They
are not used for the final v4 hypothesis verdicts.

## 3.8 Evaluation matrix

B1, B2, B3, and H5 are evaluated under clean telemetry with eight evaluation
seeds (42-49) and three episodes per seed. This gives 24 held-out episodes per
training seed for performance context.

B2 and H5 also receive the full faithfulness sweep:

```text
3 training seeds x 5 conditions x 8 evaluation seeds x 3 episodes
```

The five conditions are clean plus four outage durations. Faithfulness is
sampled every eight decisions, with raw per-decision records retained. Each
sweep must pass preflight checks for expected cells, provenance, type-matched
controls, chosen-action exclusion, and empirical degradation differentiation.
All six v4 sweeps pass.

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

Thousands of decisions improve measurement precision but do not replace
independent model replication. The final synthesis therefore treats each
training seed as one trained-policy replicate. With only three seeds per model,
the dissertation reports the range and number of supporting seeds and does not
claim a cross-seed population p-value.

## 3.10 Reproducibility

The experiment runner records the configuration, commands, seeds, checkpoint
hashes, source revision, and preflight reports under the local
`runs/dissertation_v4/` directory. Compact thesis-ready outputs are committed
under `results/dissertation_v4/`: `summary.csv`,
`performance_context.csv`, and `training_seed_synthesis.csv`. The last file
makes training-seed consistency explicit. Reproduction commands and expected
outputs are documented in `docs/REPRODUCE_EXPERIMENTS.md`.
