# Deviations from the research proposal

This note records every place where the implemented study deviates from
the submitted proposal (*When Explanations Outlive Their Data*, 17 June
2026), with the rationale. It is source material for the dissertation's
methodology chapter — MSc examiners expect deliberate, documented
deviations rather than silent ones.

## 1. Narrative simplification (three-act structure)

The proposal defines four research questions, five hypotheses, two
explanation methods, and four metric families. All formal definitions are
retained in the methodology chapter and all five hypotheses are tested,
but the results are *narrated* as three questions:

1. **Is the built-in explanation faithful at all?** (RQ1a — clean-data
   DEF baseline)
2. **Does it degrade before performance when telemetry ages?**
   (RQ1 + RQ2 = H1 + H2, with H3 + H4 as the mechanism)
3. **Can it be fixed?** (RQ4 = H5 degradation-aware training, plus the
   decoupled explainer head)

## 2. Severity axis: fixed observation-layer outage duration

The proposal (§7.2) describes severity through maximum AoI. The archived v1
implementation did not manipulate that quantity cleanly because tunnel transit
and a clipped AoI feature collapsed the nominal levels. The definitive v4
experiment therefore uses a directly controllable observation-layer outage
window of {10, 20, 30, 60} seconds with freeze corruption.

Tunnel entry emits one trigger event. SUMO keeps the true state, while the
observation builder freezes the last valid position and speed for the declared
window. AoI describes the resulting observation and is not interpreted as a
causal dose of faithfulness. Random dropout is not part of the final
confirmatory matrix.

## 3. Margin-DEF as a methodological extension

DEF is implemented exactly as defined (§7.4: Comp/Suff vs size-matched
random explanations, normalised gains, DEF = ½(g_comp + g_suff)).
However, probability-based DEF loses signal on entropy-collapsed
policies: when π(a*) ≈ 1, occluding nodes barely moves the probability
(observed |DEF| ≈ 0.0015). The dissertation therefore adds **margin-DEF**
— the same protocol measured on the decision's logit margin
(logit[a*] − max others, clamped) — computed from the *same*
counterfactual forward passes. On the studied checkpoints margin-DEF
carries ~500× the signal. Probability-DEF (the proposal's definition)
is always reported alongside; margin-DEF is presented as the primary
lens with this justification.

## 4. Metrics trimmed from the main text

- **Attention drift (JS divergence)** is computed per decision via an
  exact clean-twin observation (possible because degradation mutates
  observations only), but is demoted to the appendix: the proposal
  itself notes drift and WAMSN are two views of one phenomenon ("how
  much it changed" vs "where it went"), and WAMSN is the one the
  hypotheses use.
- **Binary AMSN** (thresholded robustness check) is not implemented;
  graded WAMSN is used throughout.
- **Performance metrics**: pickups (task completions) and mean pending
  wait are reported; average response time, fleet utilisation and a
  separate task-completion *rate* are not, as pickups/50 requests and
  wait already characterise the low-performance regime the learned
  policies occupy.

## 5. B0 premise inversion

The proposal expects B0 (greedy) to be a floor the learned policy beats.
The definitive v4 clean evaluation gives about 11-15 pickups for B1-B2 and
4.5-12.9 for H5 across training seeds; the archived greedy reference remains
higher. The dissertation adopts the proposal's own risk-section
stance: the contribution is measuring explanation faithfulness and its
decoupling, not building a state-of-the-art dispatcher; B0 is reported
as context, and the faithfulness analysis is conditioned on the learned
policies as they are. The published-baseline sanity check ([1] Lin et
al.) is likewise not meaningful at this performance level and is
omitted.

## 6. Offline dataset → online paired evaluation

The proposal (§7.6) describes logging clean episodes and applying the
staleness operator to the *logs* offline. The implementation applies
degradation live at the observation boundary instead, for a strictly
stronger design: the policy's *closed-loop* behaviour under degradation
is measured (offline replay could not change the trajectory), while the
same mechanism still yields an exact per-decision clean twin for paired
drift measurement. Episode artefacts are versioned as JSON manifests
(network `tunnels.json`, demand `demand_manifest.json`, per-sweep
`manifest.json` with seeds and git revision) rather than .npz/Parquet.

## 8. Environment substitutions

- SUMO **1.20.0** (PyPI wheel) on the Intel-mac dev machine rather than
  1.27; the committed networks are format-1.20 and load natively.
  Colab remains on the 1.27 line per `docs/COLAB.md`.
- Episodes are 1200 s, not the proposal's "≈ 1 simulated hour"; demand
  is uniform-random OD via `randomTrips.py` seeds without the optional
  rush-hour/hotspot calibration.
- Evaluation is stochastic (sampled actions): the trained policies
  entropy-collapse, and argmax evaluation degenerates to all-no-op,
  which would make every performance comparison vacuous.

## 9. Audited refinements to the DEF protocol (post-hoc, disclosed as findings)

The proposal's DEF (§7.4) draws random explanations **uniformly** from
the valid non-self nodes. The construct-validity audit
(`results/story_freeze_v1/audit/`) found that in a candidate-action
architecture this baseline is confounded: occluding a reservation node
*deletes the corresponding action*, and uniform draws hit reservation
nodes far more often than the policy's attention top-k does (margin-
clamp rates 14.6 % vs 1.8 %), producing spurious "worse/better than
random" verdicts (−1.35 … +1.78 across checkpoints from equally
uninformative channels). Two controls were added **after** this
discovery: a **type-matched random baseline** (same node-type
composition as the attended set) and an **exclusion variant** (the
chosen action's node protected from occlusion).

These refinements are deliberately NOT retro-fitted into the proposal:
the dissertation reports the proposal-specified metric first, then the
audit and the corrected numbers, because the artifact's discovery and
characterisation is itself one of the work's contributions. All
headline faithfulness claims cite the type-matched numbers; the
uniform-baseline numbers are retained to document the artifact.

## 10. Additions beyond the proposal

- **Decoupled explanation head** (occlusion-distilled scorer over detached
  encoder embeddings) — an architectural mitigation on the explanation side.
  It remains a legacy exploratory experiment and is not part of the final v4
  confirmatory matrix.
- **Chronological demand splits** implemented as seeded rider-demand
  variants (identical fleet, varied demand), giving held-out test
  demand for all headline results.
