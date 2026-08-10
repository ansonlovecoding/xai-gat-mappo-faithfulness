# 3. Methodology

The study is a controlled simulation experiment: a known cause
(telemetry staleness, parameterised by maximum AoI) is injected at the
observation boundary of a trained dispatcher, and its effect on
explanation faithfulness is measured per decision with paired
clean/degraded counterfactuals. This chapter describes the benchmark
(§3.1–3.2), the degradation framework (§3.3), the faithfulness metrics
and their audit instruments (§3.4–3.5), the experimental design (§3.6),
and the statistical methodology (§3.7). Departures from the proposal
are cross-referenced to Appendix A throughout.

## 3.1 Simulation environment

**Scenario.** Vehicle movement is simulated in SUMO [22] on a real
OpenStreetMap extract of the Central Park area of Chongqing's Yubei
district, chosen because its road network contains real tunnels —
physically grounded dead zones for GPS/V2X telemetry. Tunnel edges are
extracted from the OSM `tunnel=yes` attribute and filtered to the
navigable subset (both incoming and outgoing connections); the network,
demand files, and tunnel manifest are committed and seed-pinned.

**Fleet and demand.** 20 taxis with SUMO's taxi device
(`dispatch-algorithm=traci`, so the learned policy performs all
matching) serve 50 ride requests per 1 200 s episode. Rider demand is
generated from fixed seeds; twenty demand variants with identical fleet
initialisation and independently seeded rider realisations are split
chronologically 70/15/15 into train/validation/test sets. All headline
evaluations run on the three held-out test variants (Fig. 1).

**Observation graph** (Fig. 2). Each idle taxi observes a self-centric
heterogeneous graph: itself (position, episode time, speed, AoI), its
K_n = 5 nearest peers (relative position, availability, distance, AoI),
and the K_r = 5 nearest pending reservations (relative pickup/drop-off
vectors, waiting time). Coordinates are relative and normalised, so
learned patterns are geometric rather than absolute.

**Action space.** Discrete(K_r + 1): no-op, or accept the k-th nearest
reservation. Every reservation node therefore corresponds one-to-one to
an action — a property central to both the explanation analysis and the
artifact of §3.5.

**Reward.** A team reward broadcast to all acting agents:
R = 10·pickups + 0.5·successful dispatches − 0.001·mean pending wait.
The shaping magnitudes were established after an initial formulation
collapsed to a no-action policy (Appendix A).

## 3.2 Dispatcher

The policy is a hand-rolled GAT-MAPPO: per-type linear projections into
a shared embedding, two GAT layers with four heads (masked softmax over
valid nodes), an actor head over Discrete(K_r + 1), and a critic head.
Training uses MAPPO [5] with parameter sharing, GAE (γ = 0.99,
λ = 0.95), clipped PPO (ε = 0.2), and a centralised critic (CTDE),
which an A/B comparison showed to improve pickups by ~18 % and delay
entropy collapse. The attention implementation is deliberately
hand-rolled rather than a library layer so that attention weights are
first-class outputs of the forward pass, consumable by the faithfulness
pipeline without instrumentation tricks.

The decentralised per-agent formulation is chosen for statistical
reasons: each acting vehicle contributes one attention distribution per
decision (~1 500–2 400 per episode), giving the faithfulness analysis
per-decision resolution instead of ~120 aggregate dispatcher outputs.

**Best-checkpoint selection.** All trained policies exhibit entropy
collapse (Chapter 4); checkpoints are therefore selected by a rolling
mean of training pickups, and — because argmax evaluation of a
collapsed policy degenerates to all-no-op — *all* evaluation is
stochastic (sampled actions), matching training-time behaviour.

## 3.3 Telemetry-degradation framework

Degradation is applied **at the observation boundary only** (Fig. 1):
the simulator always holds ground truth, so any behavioural or
explanatory change is attributable exactly to what the policy observed,
and every degraded observation has an exact clean twin generated in the
same step — the basis of all paired analyses.

**Freeze semantics** (Fig. 3). When a vehicle's signal drops (its
current edge is in the tunnel set), it stops transmitting: *every*
observer — the vehicle itself and all peers — sees its last valid
position and speed, frozen, while the reading's AoI grows. Neighbour
ordering, relative vectors, and the observer's own frame all use the
frozen states, exactly as a dispatch platform working from a last-known
table would.

**Severity = maximum AoI.** Each severity level s ∈ {5, 15, 30, 60} s
keeps the signal lost until the vehicle's AoI reaches s (physically,
receiver re-acquisition delay), so the level bounds the maximum AoI
directly; geography can exceed low levels during long transits, and the
empirical AoI distribution is reported per level. The clean condition
(s = 0) is the shared reference. A matched random-dropout axis (same
outage ladder, spatially uncorrelated triggers) is retained as an
appendix robustness check.

## 3.4 Faithfulness metrics

**DEF (Dispatch Explanation Faithfulness).** For a decision with chosen
action a\*, let R_k be the explanation's top-k nodes. Comprehensiveness
is the confidence drop when R_k is occluded; sufficiency the drop when
only R_k is kept:

    Comp = f(a*|G) − f(a*|G \ R_k)        Suff = f(a*|G) − f(a*|R_k)

Each is normalised against size-matched random occlusions,
g_comp = Comp − Comp_rand, g_suff = Suff_rand − Suff, and
DEF = ½(g_comp + g_suff), averaged over k ∈ {1, 2, 3} with five random
subsets per k. DEF > 0 iff the explanation identifies information more
faithfully than a random one.

**Margin readout.** The proposal specifies f = π(a\*), the action
probability. Trained policies here are severely entropy-collapsed
(π(a\*) ≈ 0.94–1.0), and the softmax saturates: occlusions move
probabilities by ~10⁻³ regardless of relevance. The dissertation
therefore reports, alongside the probability form, a **margin-DEF**
using f = m(a\*) = logit(a\*) − max other logit (clamped to ±10),
computed from the *same* counterfactual forward passes (≈37 per scored
decision). The two readouts share the counterfactual semantics of [16]
and differ only in where the output is read; their per-decision rank
agreement is ρ ≈ 0.58–0.61. Margin-DEF is the primary lens for the
saturation reason; every probability-DEF figure is reported with it
(Appendix A).

**WAMSN (Weighted Attention Mass on Stale Nodes).** With attention
weight α_i and staleness AoI_i/AoI_max on vehicle node i:

    WAMSN = Σ α_i·(AoI_i/AoI_max) / Σ α_i   ∈ [0, 1]

graded, not thresholded. Because pooled WAMSN conflates exposure ("how
often is anything stale") with allocation ("how much attention goes to
stale nodes when present"), results also report exposure-conditional
WAMSN and two operator-facing translations: the rate at which stale
nodes enter the explanation's top-3, and top-3 membership churn against
the clean twin.

**Attention drift.** Jensen–Shannon divergence between the attention
rows of a decision's degraded observation and its clean twin (base-2,
∈ [0, 1]); reported descriptively.

## 3.5 Construct-validity instrumentation

Because every reservation node doubles as an action candidate,
occluding one deletes an action: if a\* itself is deleted the margin
clamps to −cap, an event unrelated to information content (Fig. 4).
Three instruments quantify and control this:

1. **Clamp audit.** Every counterfactual forward records whether its
   margin hit ±cap, tallied separately for explanation-top-k and
   random-baseline occlusion sets.
2. **Exclusion variant.** DEF recomputed with the chosen action's node
   *protected* — never occluded in Comp, always retained in Suff, and
   excluded from both candidate pools.
3. **Composition-matched (type-matched) baseline.** Random occlusion
   sets drawn with the same taxi/reservation composition as the
   explanation's top-k, so action-deletion events strike both sides of
   the comparison equally.

These instruments were added *after* the artifact was discovered during
an internal mock review; the discovery order is preserved in Chapter 4
and the protocol change is registered in Appendix A. Headline claims
cite the type-matched numbers; uniform-baseline numbers are retained to
document the artifact.

## 3.6 Experimental design

**Conditions.** B0: SUMO's built-in greedy matcher (non-learned upper
reference). B1: MAPPO + MLP, no graph (performance context; no
attention channel). B2: GAT-MAPPO (the audited system). B3: B2 without
the AoI input feature. H5′: B2's exact configuration trained *with*
tunnel-freeze degradation (outage 30 s), three seeds (42/43/44).
B0–B3 are trained on clean telemetry per protocol; note that under
clean training the AoI feature is identically zero, so the B2/B3
contrast is informative only as a performance control (Appendix A).

**Decoupled explanation head.** A two-layer scorer over the policy's
*detached* post-GAT node embeddings — architecturally incapable of
influencing the action — trained by occlusion distillation: targets are
per-node margin drops from single-node occlusions, softmax-normalised,
KL loss. Distillation rollouts use a seed disjoint from all evaluation
seeds; held-out rank agreement with occlusion importance is
Spearman +0.60. At evaluation the head is scored by the identical DEF
machinery via an importance-row override, in exactly-paired comparison
with attention (same decisions, same random subsets).

**Sweep.** Each faithfulness sweep covers {clean + 4 severity levels} ×
8 environment seeds × 3 test-demand episodes, scoring one decision in
eight (~18 000 scored decisions per sweep), with per-decision records,
manifests (seeds, git revision), and empirical degradation exposure
(~2 % of decisions at this geography) all archived.

## 3.7 Statistical methodology

Decisions cluster within episodes and seeds, and severity is assigned
per (level × seed) cell, so per-decision tests that assume independence
are anticonservative. The confirmatory analysis therefore uses:

- **Episode-block restricted permutation** for monotone-trend tests
  (H1, H3): severity labels are permuted across whole episode blocks
  within each seed, respecting the crossed design (120 blocks per
  sweep); reported with episode-cluster bootstrap CIs on Spearman ρ.
- **Within-episode stratification** for the WAMSN–DEF association
  (H4): ρ computed inside each episode and aggregated by sign-flip
  test, removing the between-level ecological confound.
- **Paired sign-flip tests at cell level** for the decoupling rate
  comparison (H2): per (level × seed) cell, faithfulness- and
  performance-degradation rates relative to the same seed's clean cell
  (n = 32 cells).
- **Holm correction** across the confirmatory family {H1, H2, H3, H4}
  per sweep. Margin-DEF variants and the appendix dropout axis are
  declared exploratory and reported unadjusted, labelled as such.
- **Bootstrap CIs** (2 000 resamples) for all descriptive means.

All statistics are permutation/bootstrap-based (no distributional
assumptions), implemented in numpy only, with fixed statistical seeds.

## 3.8 Reproducibility

Every result regenerates from the repository: pinned dependency
versions, committed networks and demand variants, seed-pinned training
scripts, per-sweep JSON manifests recording configuration and git
revision, committed best checkpoints, and a deviations register
(Appendix A). The development environment (SUMO 1.20 on x86-64 macOS)
and the Colab environment (SUMO 1.27) are documented per artefact; all
committed faithfulness numbers originate from the former.
