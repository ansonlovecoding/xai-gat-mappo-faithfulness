# 5. Discussion

## 5.1 Answer to the central problem

The central problem is whether a complete graph-attention map can be trusted
when some of its vehicle data are stale. The answer from this experiment is:
**not without separate validation**. Raw attention is not shown to provide a
stable assurance of freshness or decision relevance.

This answer does not depend on one p-value. Longer outages increase stale-data
exposure, and within-policy tests usually associate that exposure with lower
DEF. The direct clean-twin comparison is less uniform: attention reallocation
and paired DEF shift have the expected directions for checkpoints trained with
seeds 42 and 43, but both directions reverse for checkpoints trained with seed
44 under both training regimes. The measured effects are also small.

The action-stratified diagnostic prevents a broader interpretation. Chosen
no-op actions account for 93.0% of eligible decisions, and the combined H1 and
H4 patterns are not reproduced among dispatch actions. Four checkpoints also
change paired-DEF direction between the combined and dispatch strata. The
primary statistics therefore characterize the sampled action mixture, not a
universal explanation response for both dispatch and no-op decisions.

The supporting controls make the interpretation stronger and narrower. The
LOO-ranked control produces a positive DEF for every checkpoint, so the
evaluator has some sensitivity to a perturbation-based ranking. However,
expected top-k overlap reaches about
60% at `k=3`, which limits resolution. Taxi-only attention-LOO correlation is
positive but varies substantially in strength. The result is not "the evaluator
is perfect and attention fails." It is "within a measured but limited audit,
attention gives no reproducible explanation guarantee."

![Cross-seed evidence matrix](../figures/v9_evidence_path_summary.png)

**Figure 5.1.** The evidence separates capability, evaluator validity, stale
exposure, attention response, and decision relevance. These are not treated as
one causal chain.

## 5.2 What the stale-exposure result means

WAMSN rises with outage duration for every GAT and GAT-Outage checkpoint. It is
useful as an operational exposure indicator: it shows when an attention map
contains more weight attached to old vehicle readings.

WAMSN is not evidence that AoI causes lower faithfulness. The metric itself
weights attention by normalized AoI, so part of the increase follows directly
from the manipulation. H1 and H4 show that DEF tends to be lower at greater
exposure within most frozen policies. The paired audit asks a stricter
question: does replacing the current observation with its degraded twin
change attention and DEF at the same decision? That result changes sign across
checkpoints trained with different seeds. Exposure, within-policy association,
and paired intervention therefore answer different questions and should remain
separate.

## 5.3 Dependence on checkpoint and analysis choice

The primary aggregation gives positive stale-attention shifts for checkpoints
trained with seeds 42 and 43 and negative shifts for checkpoints trained with
seed 44. The values are small: approximately
-0.0035 to +0.0041 of total attention mass. Paired probability-DEF shifts are
also small and reproduce the same direction in both training regimes. This
shared pattern across the two model conditions suggests sensitivity to the
trained checkpoint, not a stable response created by outage-aware training.
The between-checkpoint range is larger than the observed difference between
the two training-condition means. With only three training seeds, this is a
descriptive comparison of selected checkpoints, not a variance decomposition
or evidence that the seed value itself causes the response.

The paired probability-DEF effects are small enough that their practical
importance should not be overstated. Reading all six estimates as operationally
close to zero is a reasonable challenge. The evidence against a dependable
explanation is not a large universal decrease; it is that even these small
changes reverse direction across independently trained checkpoints, while the
raw clean-telemetry DEF is already near its matched control. The study therefore
supports a lack of reproducible assurance, not a claim of large degradation
damage.

The aggregation audit reveals a second problem. Individual heads can reverse
the sign within the same checkpoint. The six checkpoint ranges extend from
-0.0109 to +0.0089 across the tested reductions. An analyst could therefore
obtain a different verbal conclusion by selecting another head or layer after
seeing the data. The
declared mean remains the primary result, but the sensitivity analysis shows
that it is not a unique explanation produced by the network.

Query-row choice is similarly checkpoint-dependent. The selected request row
improves request-action DEF by about 0.0064 for GAT seed 43 and 0.0073 for
GAT-Outage seed 43, but changes little for the other four checkpoints. This does
not support the claim that one query-row change generally corrects near-zero or
negative DEF. An operator-facing attention map needs a declared
and tested rule for choosing its query, layer, and heads.

The deterministic diagnostic reveals a related policy limitation. All six GAT
checkpoints choose no-op in all 18 argmax episode evaluations, although their
sampled policies outperform the declared lower bounds. This does not invalidate
the faithfulness calculations for the actions actually sampled. It does mean
that the experiment audits stochastic policy decisions and does not establish
a deployable deterministic dispatcher. The assurance process should therefore
report action composition and deterministic behavior before an attention map
is exposed to an operator.

## 5.4 Role of the construct-validity audit

The construct-validity audit supports the primary experiment. Request nodes
are both information and actions; peer taxis are context only. Uniform random
deletion therefore creates a large action-removal artifact. Type matching and
chosen-request protection make the comparison fairer.

The LOO control checks a different question: can DEF respond when nodes are
ranked by their own single-node margin loss? Its consistently positive result
shows limited sensitivity, mainly for comprehensiveness. It is not an oracle
for the combined comprehensiveness-sufficiency score. Gradient x Input changes
sign across checkpoints and is not automatically a stronger explanation in
this task.

Together, these checks change the conclusion from a simple “attention is near
random” statement to a more defensible one. Raw attention remains below the
LOO control in all six checkpoints and has positive taxi-only agreement with
LOO, but both DEF and rank agreement vary by trained policy. The aggregation
and query-row audits add further sensitivity. This is partial evidence of
decision relevance, not a stable default assurance.

The top-k diagnostic further limits this interpretation. Attention-LOO overlap
is above the expected matched-random overlap at `k=1`, but it offers no clear
advantage over that baseline at `k=3`. This is consistent with reduced
resolution as larger subsets overlap, not proof that the top attention node is
the model's true reason or that every three-node display is random.

## 5.5 Degradation-aware training

Training with 30-second outages does not solve the explanation problem.
GAT-Outage retains the same seed-dependent attention and paired DEF directions
as GAT. It strengthens the H1 and H4 associations, but does not make the direct
degraded-minus-clean response consistent across independently trained
policies.

This does not show that all robustness training is ineffective. The reward
contains no term for explanation stability or faithfulness, so task-reward
optimization alone is not designed to align attention with a human explanation.
A future intervention would need an explicit explanation objective and
evaluation on independently trained models.

## 5.6 Proposed assurance approach

The thesis responds to the identified risk with a **freshness-aware explanation
assurance approach**. It does not change the trained policy or claim to make
attention faithful. Instead, it provides a release process that reduces the
risk of presenting an unsupported attention map as a trustworthy explanation.
Table 5.1 links each observed risk to the required evidence and release rule.

| Observed risk | Assurance measure | Evidence used | Release rule |
|---|---|---|---|
| Attention may be attached to stale data | Report freshness separately | AoI and WAMSN | Never infer freshness from attention strength |
| Attention may not identify decision-relevant nodes | Use action-aware, type-matched tests | DEF, margin-DEF, and LOO | Do not claim a reason without evidence of decision relevance |
| Combined results may be dominated by no-op | Report no-op and dispatch strata | Action counts and stratified DEF | Do not generalize a combined result across action types |
| The result may depend on attention extraction | Test the query row, layer, head, and aggregation | Sensitivity audits | Predeclare the extraction rule and report material sensitivity |
| One checkpoint may not represent another | Audit each release checkpoint | Per-checkpoint results | Repeat after retraining or replacement |
| A pattern may not reproduce | Compare independent training runs | Cross-seed synthesis | Require a consistent conclusion across runs |

The process is sequential: report freshness and action composition separately; run the action-aware
audit; predeclare and test the attention extraction rule; audit every candidate
checkpoint; and require a consistent conclusion across independent training
runs. A failure at any stage keeps the attention map as an internal diagnostic.

Sampled dispatch performance provides capability context, but it cannot show
that the displayed reason is current or decision-relevant.

This is a risk-control and validation solution, not a model-level cure for
faithfulness decoupling. A model-level intervention would require an explicit
explanation objective and independent evaluation, as discussed in Section 5.8.

## 5.7 Limitations

The study uses one simulated district, 20 taxis, and 50 requests. Tunnel
exposure remains scenario-dependent, although the event-aware audit scores
every observed stale-exposed decision. It evaluates three independent training
seeds per model. Three seeds are sufficient to reveal heterogeneity, but not to
estimate a population distribution of training outcomes. The selected policies
pass the training stability gates and their stochastic evaluations outperform
random and greedy lower bounds. However, every selected GAT checkpoint
collapses to no-op under the small deterministic diagnostic. The study
therefore does not establish deterministic deployment capability and does not
isolate the contribution of graph edges to sampled capability.

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
the same simulated traffic state. It does not remove simulation or
scenario-design bias. Road-network choice, generated demand, vehicle behavior, and
tunnel placement may affect which taxis become stale and how those taxis are
positioned in the local graph. In particular, a fixed tunnel can create an
exposure-selection effect if its traffic conditions differ from the rest of the
network. The conclusions are therefore limited to the tested SUMO setting and
should not be read as population estimates for real fleets or cities.

## 5.8 Future work

Future experiments should add training seeds and use multiple road networks,
demand levels, tunnel placements, and public mobility traces. A randomized
telemetry-loss control would help distinguish a general stale-node effect from
one produced by tunnel location. Experiments should also include
disconnected-edge and message-passing ablations to establish how much the policy uses its
graph channel. Delayed, intermittent, and biased telemetry should be compared
with fixed freezes.

The explanation audit should also compare LOO with graph-specific post-hoc
methods and evaluate objectives that directly reward explanation consistency.
Any proposed improvement should be tested on held-out checkpoints, not only on
more episodes from one trained model. A human-factors study could then test
whether explicit freshness warnings lead operators to interpret the map more
appropriately.
