# 5. Discussion

## 5.1 Answer to the central problem

The central problem is whether a complete graph-attention map can be trusted
when some of its vehicle data are stale. The answer from this experiment is:
**not without separate validation**. Raw attention is not shown to provide a
stable assurance of freshness or decision relevance.

This answer does not depend on one expected hypothesis being confirmed. Under
clean telemetry, raw-attention DEF has mixed signs and ranges from -0.0084 to
+0.0187 across checkpoints. Under a 60-second outage, each checkpoint remains
close to its own clean value. Longer outages increase stale-data exposure, but
do not produce a consistent loss of DEF. Attention reallocation is positive
for two seeds and negative for one under both training regimes.

The added controls make the interpretation stronger and narrower. LOO produces
a positive DEF for every checkpoint, so the evaluator has some sensitivity to
a perturbation-based ranking. However, expected top-k overlap reaches about
61% at `k=3`, which limits resolution. Taxi-only attention-LOO correlation is
also inconsistent. The result is not “the evaluator is perfect and attention
fails.” It is “within a measured but limited audit, attention gives no
reproducible explanation guarantee.”

![Cross-seed evidence matrix](../figures/v4_evidence_path_summary.png)

**Figure 5.1.** The evidence separates policy capability, evaluator validity,
stale exposure, attention response, and decision relevance. No single arrow is
treated as a causal chain.

## 5.2 What the stale-exposure result means

WAMSN rises with outage duration for every GAT and GAT-Outage checkpoint. It is useful
as an operational exposure indicator: it shows when an attention map contains
more weight attached to old vehicle readings.

WAMSN is not evidence that AoI causes lower faithfulness. The metric itself
weights attention by normalised AoI, so part of the increase follows directly
from the manipulation. The paired stale-attention shift asks whether the model
reallocates attention to stale nodes; that result changes sign across seeds.
DEF asks whether ranked nodes are decision-relevant; it does not decline with
duration. These three measures answer different questions and should remain
separate.

## 5.3 Dependence on checkpoint and analysis choice

The primary aggregation gives positive stale-attention shifts for seeds 42 and
43 and negative shifts for seed 44. The values are small: approximately -0.0013
to +0.0027 of total attention mass. An operator may not perceive changes below
three tenths of one percentage point. One reasonable interpretation is that all
three are practically close to zero, even though the paired estimates have
narrow intervals.

The aggregation audit reveals a second problem. Individual heads can reverse
the sign within the same checkpoint. For GAT seed 43, alternative results span
about -0.0060 to +0.0067. An analyst could therefore obtain a different verbal
conclusion by selecting another head or layer after seeing the data. The
declared mean remains the primary result, but the sensitivity analysis shows
that it is not a unique explanation produced by the network.

Query-row choice is similarly checkpoint-dependent. The selected request row
improves request-action DEF for GAT seed 43, changes little for four checkpoints,
and sharply reduces DEF for GAT-Outage seed 43. This rules out a simple explanation
that near-zero or negative DEF occurred only because the audit used the self
row. More broadly, an operator-facing attention map needs a declared and tested
rule for choosing its query, layer, and heads.

## 5.4 Role of the construct-validity audit

The construct-validity audit supports the primary experiment. Request nodes
are both information and actions; peer taxis are context only. Uniform random
deletion therefore creates a large action-removal artefact. Type matching and
chosen-request protection make the comparison fairer.

The LOO control checks a different question: can DEF respond when nodes are
ranked by their own single-node margin loss? Its consistently positive result
shows limited sensitivity, mainly for comprehensiveness. It is not an oracle
for the combined comprehensiveness-sufficiency score. Gradient x Input is
negative in every checkpoint, showing that a familiar post-hoc method is not
automatically a stronger explanation in this task.

Together, these checks change the conclusion from a simple “attention is near
random” statement to a more defensible one. Raw attention varies in sign and
size, remains below the LOO control, correlates inconsistently with taxi-node
effects, and is sensitive to analysis choices. None of these observations
provides a stable default assurance.

## 5.5 Degradation-aware training

Training with 30-second outages does not solve the explanation problem. GAT-Outage
retains the same seed-dependent primary attention shift as GAT. Its clean raw
DEF ranges from -0.0008 to +0.0187, and its taxi-only correlation ranges from
-0.710 to +0.319. Query-row sensitivity is especially large for seed 43.

This does not show that all robustness training is ineffective. The reward
contains no term for explanation stability or faithfulness, so there is no
reason to expect task-reward optimisation alone to align attention with a
human explanation. A future intervention would need an explicit explanation
objective and evaluation on independently trained models.

## 5.6 Practical implications

An operator interface should not label raw graph-attention weights as reasons
without additional evidence. A deployment should:

- display telemetry freshness separately from attention strength;
- monitor stale-node exposure without treating it as a faithfulness score;
- declare the query row and layer/head aggregation used in the interface;
- run action-aware faithfulness tests for every released checkpoint;
- repeat the audit after retraining, because the qualitative result can change;
- avoid using stable dispatch output as evidence that explanations remain
  trustworthy.

## 5.7 Limitations

The study uses one simulated district, 20 taxis, 50 requests, and sparse tunnel
exposure. It evaluates three independent training seeds per model. Three seeds
are sufficient to expose heterogeneity, but not to estimate a population
distribution of training outcomes. GAT-Outage seed 44 is weak, and GAT training shows
early entropy collapse. The selected GAT policies outperform random and greedy
lower bounds, but the study does not isolate the contribution of graph edges to
that capability.

The local graph is small. Type matching makes random top-k sets overlap heavily
with the explanation, which compresses DEF. LOO is based on the same
single-node perturbation as comprehensiveness and is not an independent ground
truth explanation. Other perturbation choices or post-hoc explainers may give
different results.

The degradation layer freezes position and speed for fixed windows. Real
telemetry may include delayed packets, partial failures, map-matching errors,
and asynchronous recovery. The study audits attention as an explanation; it
does not claim that attention is useless inside the policy computation.

The paired clean/degraded comparison gives the study internal validity for its
implemented observation-layer intervention: both observations are derived from
the same simulated traffic state. It does not remove simulation or scenario-
design bias. Road-network choice, generated demand, vehicle behaviour, and
tunnel placement may affect which taxis become stale and how those taxis are
positioned in the local graph. In particular, a fixed tunnel can create an
exposure-selection effect if its traffic conditions differ from the rest of the
network. The conclusions are therefore limited to the tested SUMO setting and
should not be read as population estimates for real fleets or cities.

## 5.8 Future work

Future experiments should add training seeds and use multiple road networks,
demand levels, tunnel placements, and public mobility traces. A randomised
telemetry-loss control would help distinguish a general stale-node effect from
one produced by tunnel location. Experiments should also include disconnected-
edge and message-passing ablations to establish how much the policy uses its
graph channel. Delayed, intermittent, and biased telemetry should be compared
with fixed freezes.

The explanation audit should also compare LOO with graph-specific post-hoc
methods and evaluate objectives that directly reward explanation consistency.
Any proposed improvement should be tested on held-out checkpoints, not only on
more episodes from one trained model. A human-factors study could then test
whether explicit freshness warnings lead operators to interpret the map more
appropriately.
