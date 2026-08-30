# The three-act story under the corrected freeze mechanism (July 2026)

Definitive result set for the simplified dissertation narrative, produced
after the proposal-alignment review: freeze corruption (last-known state
visible to all observers), max-AoI severity ladder {5, 15, 30, 60} s
(proposal §7.2), AOI_MAX_S = 60, all evaluation stochastic on **held-out
test demand**, 8 seeds × 3 episodes per cell, ~18k scored decisions per
sweep. Figures in `figs/`.

## Act 1 — the built-in explanation carries no measurable decision-relevant information (RQ1a)

**Revised after the P2 construct-validity audit (`audit/`).** Under the
standard occlusion protocol, clean-telemetry margin-DEF over 3 565
decisions is **−0.541** (95 % CI [−0.556, −0.527]) — apparently "worse
than random". The audit shows **98 % of that deficit is a protocol
artifact**: in a candidate-action architecture, occluding a reservation
node *deletes the corresponding action*, and uniform random baselines
hit reservation nodes (clamping the margin) far more often than this
policy's attention top-k does (clamp rates 14.6 % vs 1.8 %). With a
**type-matched random baseline** (same taxi/reservation composition as
the attention top-k, 1 199 paired decisions) clean margin-DEF is
**−0.009** [CI −0.016, −0.001]; with the chosen reservation's node
protected from occlusion, dispatch decisions' apparent +0.59 collapses
to −0.02.

The corrected Act-1 claim: **attention is uninformative — statistically
indistinguishable from (marginally below) a composition-matched random
explanation**. The proposal's reinterpretation clause for near-zero
clean DEF applies as written. The artifact itself is a methodological
finding with scope beyond this dissertation: ERASER-style occlusion
baselines produce spurious "worse than random" verdicts whenever
explanation units double as action candidates.

## Act 2 — attention silently shifts to stale data (H1–H4, B2 sweep)

| level (max AoI) | 0 | 5 | 15 | 30 | 60 |
|---|---|---|---|---|---|
| margin-DEF | −0.541 | −0.546 | −0.544 | −0.541 | −0.542 |
| pickups | 6.67 | 7.42 | 6.79 | 6.54 | 7.42 |
| WAMSN | 0.000 | 0.015 | 0.012 | 0.013 | 0.013 |

- **H3 supported** (ρ=+0.092, p=1e-4) and **H4 supported** (ρ=−0.142,
  p=1e-4): attention mass moves onto stale nodes as soon as staleness
  exists, and decisions with more stale-node attention are less faithful.
- **H1 not supported** on the pooled per-decision test: faithfulness is
  already at its floor on clean data (Act 1) and tunnel exposure is only
  ~2 % of decisions — there is nothing left to lose. **H2 supported**
  at the cell level (p=0.020/0.028 prob/margin): the *rate* framing
  still detects faithfulness moving relative to (flat) performance.
- Reading for the dissertation: this is "explanations outlive their
  data" in its purest form — the explanation channel neither warns
  (DEF flat at the floor) nor does performance (flat), while the
  explanation's content visibly drifts onto stale telemetry (WAMSN).

## Act 3a — training-side mitigation FAILS (H5)

H5′ = B2's exact config trained WITH tunnel_triggered freeze degradation
(outage 30 s, 300 epochs, seed 42). Clean test-demand eval 7.80 ± 1.33
(B2: 6.67 — no performance cost). But margin-DEF is **worse at every
severity level: −0.77 vs B2's −0.54**, WAMSN slightly higher, same flat
profiles. The proposal's H5 wording ("results may support or reject a
mitigation effect") lands on **reject**: experiencing degradation during
training does not make the built-in attention channel faithful.

⚠️ The earlier "H5 reverses the decoupling" result
(`results/legacy_sumo120_seed42_v1/`) was obtained under the legacy
noise mechanism (self-features jittered, relative geometry left true)
and did NOT reproduce under the proposal-faithful freeze mechanism.
Cite the freeze-era numbers; treat the noise-era reversal as a
mechanism-sensitivity finding, not a result.

## Capability spectrum + H5 across seeds (P3/P4 audit, `audit/capability_spectrum_clean.json`)

Six checkpoints spanning pickups 1→11.8, entropy 1.56→0.03, including
B2 training stages and three H5′ training seeds, each scored under
both baselines on clean test demand:

| ckpt | pickups | uniform def_m | type-matched def_m |
|---|---:|---:|---:|
| B2 epoch 0 (untrained) | 1.0 | −1.35 | +0.16 [−0.12, +0.51] |
| B2 epoch 50 | 6.2 | −0.47 | +0.00 |
| B2 best (91) | 7.5 | −0.55 | −0.00 |
| H5′ s42 best (17) | 7.2 | −0.80 | +0.02 |
| H5′ s43 best (91) | 3.0 | −0.49 | −0.01 |
| H5′ s44 best (22) | 11.8 | **+1.78** | +0.01 |

Two conclusions:
1. **The uniform-baseline metric is uninterpretable**: it swings from
   −1.35 to +1.78 across checkpoints — the artifact can manufacture
   "far worse than random" AND "far better than random" from equally
   uninformative channels. This instability is the strongest single
   piece of evidence for the methodological contribution.
2. **Under the type-matched baseline every checkpoint sits at ≈ 0**
   (−0.01 … +0.02): "attention is uninformative" holds across the whole
   capability range, both architectures, and all three H5 seeds —
   answering the external-validity question (Q4 of the mock review)
   with data.

**Act 3a is therefore revised once more**: the earlier "degradation-aware
training makes faithfulness WORSE (−0.77 vs −0.54)" was artifact-on-
artifact. Under the fair baseline, H5′ ≈ B2 ≈ 0 across three seeds:
degradation-aware training has **no detectable effect on explanation
faithfulness in either direction** — H5's mitigation hypothesis is
rejected in its weakest, most defensible form. (Training-collapse
caveat from E2 still applies: two of three best checkpoints land at
epochs 17–22.)

## Act 3b — architecture-side mitigation: small but real on clean data (decoupled head)

Occlusion-distilled decoupled explainer (val Spearman +0.60) vs coupled
attention, paired per decision on fresh episodes, freeze mechanism.
Under the standard (uniform-baseline) protocol:

| condition | coupled | decoupled | Δ | p |
|---|---:|---:|---:|---:|
| clean | −0.495 | −0.389 | +0.107 | 0.0001 |
| tunnel (max-AoI 60) | −0.506 | −0.393 | +0.113 | 0.0001 |

**Revised after the P2 artifact audit** (type-matched revalidation,
`audit/compare_typematched_*.json`) — under the composition-matched
baseline the picture narrows:

| condition | coupled | decoupled | Δ | p |
|---|---:|---:|---:|---:|
| clean | −0.013 | **+0.010** | **+0.023** | **0.0001** |
| tunnel (max-AoI 60) | −0.006 | −0.000 | +0.005 | 0.21 |

The earlier +0.11 advantage was ~80 % artifact. What survives: on clean
telemetry the decoupled head is still *significantly* more faithful
(paired p=1e-4) and is the **only channel scoring above the fair random
baseline** — but the effect is modest (+0.023), and under tunnel
degradation the advantage is not statistically detectable (p=0.21).
Corrected Act-3b claim: distillation buys a real but modest clean-data
faithfulness improvement; **no tested mitigation restores meaningful
faithfulness under degradation** — which sharpens, rather than softens,
the closing statement that faithful dispatch explanation remains open.

## Provenance

- B2 ckpt: `../legacy_sumo120_seed42_v1/B2_gat/ckpt_best.pt` (epoch 91)
- H5′ ckpt: `H5b_train/ckpt_best.pt` (epoch 17, rolling 10.2)
- Sweeps: `runs/sweeps/B2_aoi_ladder`, `runs/sweeps/H5b_aoi_ladder`
  (cell JSONs with per-decision records; manifests + analyses committed
  here). Explainer head: `../legacy_sumo120_seed42_v1/explainer/`.
- Empirical note: acting-agent degradation exposure ≈ 2 % at all levels
  (geography-limited; in-tunnel transits dominate, post-exit outage adds
  the level-dependence); max observed AoI 310 s (taxi idling on a
  tunnel edge).
