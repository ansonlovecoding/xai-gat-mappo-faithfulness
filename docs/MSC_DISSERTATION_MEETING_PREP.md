# MSc Dissertation Meeting Preparation

Project: *When Explanations Outlive Their Data: Faithfulness Decoupling in Graph-Attention MARL Fleet Dispatch under Telemetry Degradation*

Student: Hongwei Lin  
Programme context: MSc dissertation, De Montfort University Dubai

## 1. Problem Area / Statement

The dissertation studies explainability in multi-agent reinforcement learning
(MARL) for fleet dispatch.

The exact problem is:

> In graph-attention MARL dispatch systems, attention weights are often treated
> as explanations. However, in real fleet operations, vehicle telemetry can
> become stale because of tunnels, signal loss, or urban shadow zones. The
> project asks whether attention-based explanations remain faithful when the
> data used by the model is no longer fresh.

The problem is not simply whether the dispatcher performs well. The focus is
whether the explanation shown to a user still reflects the real decision-making
evidence when telemetry degrades.

The key risk is called **faithfulness decoupling**: task performance may look
stable while the explanation becomes unreliable or shifts toward stale data.

## 2. Literature Review

The dissertation reviews around **25 core research articles and technical
sources**, grouped into four areas:

| Area | Purpose in dissertation |
|---|---|
| Reinforcement learning and MARL for fleet dispatch | Establishes why MARL is suitable for ride-hailing and taxi dispatch |
| Graph neural networks and graph attention | Justifies representing taxis and requests as a relational graph |
| Attention and explanation faithfulness | Shows why attention cannot automatically be trusted as an explanation |
| Age of Information and telemetry staleness | Provides the concept used to model stale vehicle observations |

Important foundational literature includes MARL dispatch work, MAPPO,
Graph Attention Networks, attention-faithfulness studies, perturbation-based
faithfulness tests, Age of Information, and SUMO simulation.

## 3. Benchmark Literature

The closest recent studies for comparison are:

| Study | Relevance | Difference from this dissertation |
|---|---|---|
| **DualG-MARL**: a 2026 Scientific Reports study on graph-attentive MARL for ride-sharing dispatch. It models vehicle state graphs and task graphs to improve dispatch under dynamic urban demand. Source: [Nature Scientific Reports, 2026](https://www.nature.com/articles/s41598-026-35004-8) | Very close in problem domain: graph-attentive MARL for ride-sharing dispatch | Focuses on dispatch performance and structural modeling, not explanation faithfulness under stale telemetry |
| **BMG-Q**: a 2025/2026 arXiv work on Localized Bipartite Match Graph Attention Q-Learning for ride-pooling order dispatch. Source: [arXiv:2501.13448](https://arxiv.org/abs/2501.13448) | Very close architecturally: graph attention, MARL, vehicle-order matching | Uses graph attention for scalable dispatch quality, but does not audit whether attention is a faithful explanation |
| **CoopRide**: KDD 2025 work on city-scale ride-hailing dispatch with MARL. Source: [ACM KDD 2025](https://dl.acm.org/doi/10.1145/3690624.3709205) | Useful wider benchmark for city-scale MARL dispatch | Focuses on cooperative dispatch performance, not explanation reliability or telemetry degradation |

These studies are useful benchmarks because they show the current direction of
the field: graph/MARL dispatch models are increasingly common, but the
explanation and telemetry-degradation questions remain underexplored.

## 4. Identified Research Gaps

The dissertation identifies four main gaps:

1. **Attention is used as explanation, but rarely audited in dispatch.**  
   Graph attention weights are easy to visualise, but this does not prove that
   they caused the dispatch decision.

2. **Telemetry degradation is usually treated as a performance issue, not an
   explanation issue.**  
   Prior work studies whether dispatch quality changes, but not whether the
   explanation remains faithful when vehicle data becomes stale.

3. **Existing perturbation-based faithfulness tests can be invalid in
   candidate-action settings.**  
   In this project, reservation nodes are also possible actions. Occluding a
   reservation node can delete an action, so a naive random occlusion baseline
   can measure an artifact rather than explanation quality.

4. **There is limited evidence on mitigation.**  
   It is unclear whether degradation-aware training or a separate explanation
   head can repair faithfulness.

## 5. Research Questions and Objectives

### Research Questions

| ID | Research question |
|---|---|
| RQ1a | Are attention-based explanations faithful on clean telemetry before degradation is introduced? |
| RQ1 | How does explanation faithfulness change when telemetry becomes stale? |
| RQ2 | Does explanation faithfulness degrade faster than dispatch performance? |
| RQ3 | Does attention increasingly concentrate on stale telemetry? |
| RQ4 | Can telemetry-aware training or a decoupled explanation head mitigate the problem? |

### Measurable Objectives

| Objective | Measurement |
|---|---|
| Build a reproducible SUMO dispatch benchmark | Seeded SUMO network, demand splits, saved configs and checkpoints |
| Train and evaluate MARL dispatch models | Pickups, dispatch success, pending wait, entropy/checkpoint behaviour |
| Simulate telemetry degradation | Tunnel-triggered freeze degradation and Age of Information |
| Measure explanation faithfulness | DEF and margin-DEF using counterfactual occlusion |
| Measure stale-data attention | WAMSN and attention drift |
| Audit metric validity | Uniform vs type-matched random baselines, clamp audit, chosen-action protection |
| Test mitigation | Degradation-aware GAT training and decoupled occlusion-distilled explanation head |

## 6. Dataset Selection

The project does not use a static public tabular dataset. It uses a controlled
simulation dataset generated from SUMO.

Dataset / environment:

| Component | Description |
|---|---|
| Simulator | SUMO microscopic traffic simulator |
| Map | OpenStreetMap extract of Central Park area, Yubei district, Chongqing |
| Reason for area | Contains real tunnel edges, giving physically grounded signal-loss zones |
| Fleet | 20 taxis |
| Requests | 50 ride requests per episode |
| Episode length | 1,200 simulated seconds |
| Demand splits | Seeded train, validation and held-out test demand variants |
| Main completed result set | `results/dissertation_v4/` (compact citable outputs) |
| Supporting validity audit | `results/story_freeze_v1/audit/` |

### Suitability

The SUMO setup is suitable because the research question requires controlled
paired comparisons. A real-world dataset would not easily provide a clean twin
and a degraded twin of the exact same dispatch decision. SUMO allows the
simulator to keep ground truth while only the policy observation is degraded.
This makes the experiment cleaner and easier to audit.

## 7. Exploratory Data Analysis

The data shows four important patterns:

| Finding | Interpretation |
|---|---|
| Clean B1/B2/B3 policies average about 13 pickups, while H5 averages 9.56 and is seed-sensitive | Performance describes the policies under audit; it is not the primary research outcome |
| Tunnel/stale exposure is sparse | Only a minority of decisions include stale vehicle nodes, so conditional measures are more informative than pooled values |
| Conditional WAMSN rises with fixed outage duration in all six GAT runs | The 10/20/30/60-second observation outages increase stale-data exposure as intended |
| Paired stale-attention shift changes sign across training seeds | Seeds 42/43 shift toward stale nodes, while seed 44 shifts away, under both B2 and H5 |
| DEF does not decrease with outage duration | H1 and H2 are unsupported in all six runs; AoI should not be claimed to cause lower faithfulness |

The most defensible EDA summary is:

> The benchmark successfully creates increasing stale-data exposure. WAMSN
> rises consistently, but attention reallocation and its relationship with DEF
> vary across independently trained policies. The result is about explanation
> trust and model dependence, not a universal AoI-to-faithfulness effect.

## 8. Proposed / Developed Model

The main model is **B2: GAT-MAPPO**.

For each acting taxi, the environment builds a self-centred graph:

| Node type | Meaning |
|---|---|
| Self taxi | The acting vehicle |
| Peer taxi nodes | Nearby fleet context |
| Reservation nodes | Candidate passenger requests and candidate actions |

The policy uses:

- per-type feature embeddings,
- two graph-attention layers,
- four attention heads,
- masked attention over valid graph nodes,
- an actor head for no-op or reservation selection,
- a critic head for MAPPO training.

The attention weights are saved and treated as the **coupled explanation
channel**. The dissertation then tests whether those weights are actually
faithful explanations.

Additional model conditions:

| Condition | Purpose |
|---|---|
| B0 SUMO greedy | Non-learning reference |
| B1 MLP-MAPPO | RL baseline without graph attention |
| B2 GAT-MAPPO | Main audited attention model |
| B3 GAT-MAPPO without AoI | Control for explicit AoI input |
| H5' degradation-aware GAT | Tests whether training with degradation helps |
| Decoupled explanation head | Legacy exploratory mitigation; not part of the final v4 confirmatory matrix |

## 9. Gap-Model Alignment

| Research gap | How the model/design addresses it |
|---|---|
| Attention is often assumed to explain decisions | The GAT produces attention weights, and DEF tests whether they are faithful |
| Telemetry staleness is not linked to explanation quality | Tunnel-triggered AoI degradation creates stale observations for the policy |
| Need to separate true simulator state from degraded observation | SUMO remains ground truth; degradation is applied only at the observation boundary |
| Perturbation tests may be invalid when nodes are actions | The project adds type-matched baselines, clamp audits, and chosen-action protection |
| Mitigation is unclear | H5' and the decoupled head test training-side and architecture-side mitigation |

## 10. Objective-Model Alignment

| Objective | Enabled by |
|---|---|
| Measure clean attention faithfulness | B2 attention + DEF/margin-DEF |
| Test stale-telemetry effects | Tunnel freeze degradation + AoI |
| Detect stale-node reliance | WAMSN over vehicle nodes |
| Compare performance vs explanation behaviour | Pickups compared against DEF/WAMSN |
| Check whether graph attention matters | B1 vs B2 |
| Check whether AoI input matters | B2 vs B3 |
| Check whether degraded training helps | B2 vs H5' |
| Check whether explanations should be decoupled | Coupled attention vs decoupled head |

## 11. Implementation Status

| Area | Status |
|---|---|
| SUMO dispatch environment | Implemented |
| Tunnel-triggered telemetry degradation | Implemented |
| GAT-MAPPO policy | Implemented |
| MLP baseline | Implemented |
| AoI/no-AoI model variants | Implemented |
| Degradation-aware training condition | Implemented |
| Faithfulness evaluator | Implemented and smoke-tested |
| WAMSN metric | Implemented |
| Construct-validity audit | Implemented |
| Decoupled explanation head | Implemented; retained as legacy exploratory work |
| Completed v4 result set | Available in `results/dissertation_v4/` |
| Training-seed synthesis | Completed for B2 and H5 |
| Reproduction guide | Available in `docs/REPRODUCE_EXPERIMENTS.md` |
| Dissertation chapter drafts | Available in `docs/dissertation/` |

Current development status: the implementation, three-seed training matrix,
held-out evaluations, faithfulness sweeps, preflight checks, analyses, and
summary tables are complete. The remaining work is final document formatting,
citation checking, and visual inspection of the generated thesis/PDF.

## 12. Performance Parameters / Metrics

| Metric | Why it is used |
|---|---|
| Pickups per episode | Main task-performance measure for dispatch |
| Successful dispatches | Measures accepted dispatch actions |
| Mean pending wait | Captures passenger waiting burden |
| DEF | Tests whether explanation-selected nodes are more decision-relevant than random nodes |
| Margin-DEF | More stable than probability DEF when the policy softmax is saturated |
| WAMSN | Measures how much attention is placed on stale vehicle telemetry |
| Attention drift | Measures how much attention changes between clean and degraded observations |
| Clamp rate | Audits whether occlusion deletes candidate actions and corrupts DEF |
| Type-matched DEF | Corrects the random baseline so it matches attention's node-type composition |

These metrics are appropriate because the dissertation is not only evaluating
dispatch performance. It is evaluating whether the explanation is faithful,
whether stale data enters the explanation, and whether the measurement itself
is valid.

## 13. Benchmark Comparison

Compared with recent benchmark literature, this dissertation has a different
emphasis.

| Benchmark literature | Main aim | Comparison with this dissertation |
|---|---|---|
| DualG-MARL, 2026 | Improve ride-sharing dispatch using state/task graph modeling | This dissertation also uses graph-based MARL, but studies explanation faithfulness rather than dispatch optimisation |
| BMG-Q, 2025/2026 | Improve ride-pooling order dispatch using graph attention Q-learning and bipartite matching | This dissertation shares the graph-attention dispatch setting, but audits whether attention explanations are faithful |
| CoopRide, KDD 2025 | City-scale cooperative MARL dispatch | This dissertation is smaller scale but more focused on explainability, telemetry degradation, and metric validity |

Performance comparison should be stated carefully. The learned policy in this
project is weaker than SUMO greedy, so it should not be positioned as a
state-of-the-art dispatch algorithm. The contribution is methodological:

> Recent graph/MARL dispatch studies show that graph attention can improve
> dispatch decisions. This dissertation asks a different question: whether the
> attention channel should be trusted as an explanation, especially when the
> vehicle telemetry is stale.

## 14. Dissertation Writing Status

| Chapter / Section | Status | Notes |
|---|---|---|
| Abstract | Updated | Aligned with completed v4 results |
| Chapter 1: Introduction | Updated | Central problem and bounded claims aligned |
| Chapter 2: Literature Review | Drafted | Should add or mention 2025/2026 benchmark studies for comparison |
| Chapter 3: Methodology | Updated | Uses validation selection and fixed outage-duration protocol |
| Chapter 4: Results | Updated | Uses `runs/dissertation_v4/` and cross-seed synthesis |
| Chapter 5: Discussion | Updated | Separates WAMSN exposure from faithfulness claims |
| Chapter 6: Conclusion | Updated | Uses bounded cross-seed conclusion |
| Appendix / Deviations Register | Drafted | Important for explaining changes from proposal |
| Reproduction guide | Drafted | Includes environment and command instructions |
| Paper-style summary | Drafted | `docs/PAPER_Faithfulness_Decoupling.md` |

## 15. Short Meeting Script

If asked to summarise the dissertation in one minute:

> My dissertation studies whether graph-attention explanations in MARL fleet
> dispatch remain faithful when vehicle telemetry becomes stale. I built a SUMO
> taxi-dispatch benchmark using a tunnel-rich Chongqing road network, where
> tunnel entry triggers freeze-style telemetry degradation. The main model is a
> GAT-MAPPO dispatcher. I evaluate attention explanations using DEF and WAMSN.
> The completed experiment shows that longer observation outages consistently
> increase WAMSN, so more of the displayed attention is linked to stale data.
> However, the paired attention shift is positive for two training seeds and
> negative for one under both clean and degradation-aware training. H1 and H2
> are unsupported in all six policies, so I do not claim that AoI causes lower
> faithfulness. The contribution is a reproducible, action-aware audit showing
> that attention explanations require freshness checks and validation for each
> trained checkpoint.

## 16. Points to Be Honest About

- The learned dispatch policies are weaker than SUMO greedy, so the claim is
  not state-of-the-art dispatch performance.
- WAMSN increases with outage duration, but this is an exposure result and not
  proof that AoI causes lower DEF.
- The paired attention shift changes sign across training seeds, so a universal
  "attention shifts toward stale nodes" claim would be too strong.
- The uniform DEF baseline produced misleading results because reservation
  nodes are also actions. This became a methodological finding, not a mistake
  to hide.
- The dissertation's strength is the controlled experiment, audit trail, and
  explanation-faithfulness analysis.
