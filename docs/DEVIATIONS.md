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

## 2. Severity axis: one, not two

The proposal (§7.2) commits to a single degradation axis — max-AoI via
tunnel signal loss — and the implementation briefly explored a second
axis (random dropout at varying rates, plus noise-magnitude variation).
The final experiments restore the proposal's single axis: the max-AoI
ladder {5, 15, 30, 60} s with freeze corruption. The matched-structure
random-dropout comparison is retained only as an appendix robustness
check (`sweep_severity.py --with-dropout-axis`).

One refinement over the proposal: the proposal controls maximum AoI via
"tunnel transit duration", which is fixed by geography once the map is
chosen. The implementation therefore adds `outage_duration_s`: the
signal stays lost until AoI reaches the level, even after the vehicle
exits the tunnel — physically read as receiver re-acquisition delay.
Long transits can exceed low levels, so the empirical AoI distribution
is reported per level.

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
Measured: SUMO's greedy matcher completes 32 pickups vs 6–8 for every
learned policy. The dissertation adopts the proposal's own risk-section
stance: the contribution is measuring explanation faithfulness and its
decoupling, not building a state-of-the-art dispatcher; B0 is reported
as context, and the faithfulness analysis is conditioned on the learned
policies as they are. The published-baseline sanity check ([1] Lin et
al.) is likewise not meaningful at this performance level and is
omitted.

## 6. The B2-vs-B3 (AoI-feature) ablation is vacuous under clean training

Under the proposal's own protocol (§7.7: B0–B3 trained on clean
telemetry), the AoI observation feature is identically zero during
training, so B3 (AoI-unaware) differs from B2 only by an unused input.
The measured clean-eval difference (8.20 vs 5.80 pickups) is seed noise,
not an ablation effect. The meaningful AoI-awareness comparison is the
RQ4 one — degradation-aware training (H5) vs degradation-naive (B2) —
and the dissertation frames it that way.

## 7. Offline dataset → online paired evaluation

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

## 9. Additions beyond the proposal

- **Decoupled explanation head** (occlusion-distilled scorer over
  detached encoder embeddings) — an architectural mitigation on the
  explanation side, complementing the proposal's training-side
  mitigation (H5).
- **Chronological demand splits** implemented as seeded rider-demand
  variants (identical fleet, varied demand), giving held-out test
  demand for all headline results.
