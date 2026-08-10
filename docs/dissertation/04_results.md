# 4. Results

Results are reported in the order they were established, because the
order matters: an audit in the middle of the study overturned the
reading of its first finding, and the corrected protocol then governed
everything after. All numbers derive from the archived result set
(`results/story_freeze_v1/`, including `audit/`); statistics are the
cluster-robust ones of §3.7 unless stated.

## 4.1 Performance context

Under stochastic evaluation on held-out test demand, SUMO's built-in
greedy matcher completes 32 pickups per episode; the learned policies
complete 6–8 (B2: 6.7 ± 1.9). The learned policies additionally exhibit
entropy collapse during training (entropy < 0.05 within 25–130 epochs),
which motivates rolling-mean checkpoint selection and stochastic
evaluation (§3.2). The study's claims are accordingly scoped to this
low-performance regime; §4.5 shows the faithfulness findings are
nevertheless invariant across the capability range available, and
Chapter 5 discusses the external-validity implications of the
greedy gap.

## 4.2 Act 1 — the built-in explanation is uninformative (RQ1a)

**Initial reading.** Under the proposal's protocol (uniform random
baselines), the clean-telemetry margin-DEF of the B2 policy is
**−0.541** (95 % CI [−0.556, −0.527]; n = 3 565 scored decisions) —
apparently *worse than random* (Fig. 5, left).

**The audit.** A construct-validity audit interrogated this reading
with the three instruments of §3.5:

- *Clamp audit*: 14.6 % of random-baseline counterfactual forwards hit
  the margin clamp, versus 1.8 % for attention-top-k occlusions — the
  action-deletion mechanism strikes the two sides of the comparison at
  an 8:1 ratio.
- *Exclusion variant* (dispatch decisions, n = 136): the standard
  protocol scores +0.586 — apparently *better* than random — collapsing
  to −0.024 when the chosen action's node is protected. The artifact
  manufactures both signs.
- *Type-matched control* (n = 1 199 paired decisions): clean margin-DEF
  moves from −0.554 (uniform) to **−0.009** (95 % CI [−0.016, −0.001])
  when random baselines share attention's node-type composition
  (Fig. 5, right). **98 % of the deficit is artifact.**

**Corrected finding.** The attention channel carries no measurable
decision-relevant information on clean telemetry: statistically
indistinguishable from (marginally below) a fair random explanation.
The proposal's reinterpretation clause for near-zero clean DEF applies
in full, and the artifact characterisation is promoted to a
contribution in its own right (§5.3).

## 4.3 Act 2 — silent drift under staleness (H1–H4)

B2 evaluated across the max-AoI ladder (Fig. 6; ≈18 000 scored
decisions; empirical exposure ≈2 % of decisions, dominated by in-tunnel
transit with the outage extension supplying the level dependence):

- **H3 — supported.** WAMSN rises from 0 to ≈0.013 as soon as staleness
  exists (ρ = +0.092; episode-block permutation p = 10⁻⁴; cluster CI
  [+0.052, +0.128]). Conditional on exposure (≥1 stale vehicle node
  visible, 8.4 % of decisions), WAMSN averages 0.140, and a stale node
  enters the explanation's top-3 in 8.5 % of exposed decisions.
- **H4 — supported, strengthened.** Within episodes — the stratification
  that removes the between-level confound — the WAMSN–DEF correlation
  is negative in 89 % of episodes (mean within-episode ρ = −0.140,
  p = 10⁻⁴): the more a decision's attention rests on stale nodes, the
  less faithful it is, inside the same episode.
- **H1 — not supported (floor effect).** Margin-DEF is flat at its
  clean-data level at every severity (episode-block p = 0.65; cluster
  CI for ρ [−0.016, +0.025]). Given Act 1, the predicted decline has no
  room to occur: a channel at the fair-random floor cannot lose
  faithfulness it does not have.
- **H2 — supported in the rate framing.** Cell-level
  faithfulness-degradation rates exceed performance-degradation rates
  (n = 32; Holm-corrected p = 0.039) — performance itself is
  statistically flat (6.5–7.4 pickups across severities).
- Attention drift is present but small (JS ≈ 0.000–0.001), consistent
  with a saturated encoder; top-3 explanation membership changes versus
  the clean twin in 1.4 % of decisions.

Under Holm correction over the confirmatory family, the verdict set is
H1 ✗, H2 ✓ (0.039), H3 ✓ (4×10⁻⁴), H4 ✓ (4×10⁻⁴).

**Reading.** This is the operational hazard in its purest form: the
explanation's *content* measurably shifts onto stale telemetry, while
neither the faithfulness score (already at floor) nor the KPIs
(pickups flat) provide any warning. The explanation outlives its data
not by becoming worse, but by having no fidelity to lose while its
appearance changes silently.

## 4.4 Act 3 — mitigation attempts (H5 and the decoupled head)

**Act 3a — training-side mitigation: no effect** (Fig. 7). H5′
policies (B2's configuration trained under tunnel-freeze degradation,
outage 30 s, seeds 42/43/44) show severity profiles indistinguishable
from B2's: margin-DEF flat, WAMSN and pickups equivalent. Under the
type-matched baseline all three seeds sit at margin-DEF ≈ 0
(−0.01 … +0.02), exactly where B2 sits. An earlier apparent result —
that degradation-aware training made faithfulness *worse* (−0.77 vs
−0.54) — did not survive the audit: both figures were uniform-baseline
readings, artifact-on-artifact. H5, phrased in the proposal to allow
either outcome, resolves as **no mitigation effect in either
direction**. Caveats: two of three best checkpoints select at epochs
17–22 (early training; one at epoch 91), reflecting the collapse
dynamics of §4.1.

**Act 3b — architecture-side mitigation: modest, clean-data only**
(Fig. 8). The occlusion-distilled decoupled head, compared to the
coupled attention channel on identical decisions with identical
type-matched baselines:

| condition | coupled | decoupled | Δ | paired p |
|---|---:|---:|---:|---:|
| clean (n = 983) | −0.013 | **+0.010** | **+0.023** | **10⁻⁴** |
| tunnel, max-AoI 60 s (n = 878) | −0.006 | −0.000 | +0.005 | 0.21 |

The decoupled head is the only channel in the study to score above the
fair random baseline, and its clean-data advantage is highly
significant but modest; under degradation the advantage is not
detectable. An earlier "+0.11 in both conditions" reading was ~80 %
artifact.

## 4.5 The artifact across the capability spectrum

Seven checkpoints spanning pickups 1→11.8 and entropy 1.56→0.03 —
untrained, mid-training, and best B2; the AoI-unaware B3; all three
H5′ seeds — scored under both baselines on clean test demand (Fig. 9):

- Under the **uniform** baseline the metric swings from −1.35 to +1.78,
  manufacturing "far worse than random" and "far better than random"
  verdicts from equally uninformative channels, depending only on where
  each policy's attention sits relative to reservation nodes.
- Under the **type-matched** baseline every checkpoint lies within
  −0.01 … +0.02 of zero.

The uninformativeness finding is therefore invariant to capability
level, architecture variant, and training regime — and the uniform
baseline's instability is the single strongest demonstration that
uncontrolled occlusion protocols are uninterpretable in
candidate-action architectures.

## 4.6 Summary of hypothesis outcomes

| Hypothesis | Verdict | Key statistic |
|---|---|---|
| H1 severity ↑ → DEF ↓ | not supported (floor) | ρ ≈ 0, CI [−0.016, +0.025] |
| H2 faithfulness declines faster than performance | supported (rate framing) | Holm p = 0.039, n = 32 cells |
| H3 severity ↑ → WAMSN ↑ | supported | ρ = +0.092, p = 10⁻⁴ (robust) |
| H4 WAMSN negatively associated with DEF | supported (within-episode) | 89 % episodes negative, p = 10⁻⁴ |
| H5 degradation-aware training changes the relationship | rejected as mitigation | Δ ≈ 0 across 3 seeds (type-matched) |
