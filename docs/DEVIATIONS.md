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
and a clipped AoI feature collapsed the nominal levels. The final experiment
therefore uses a directly controllable observation-layer outage
window of {10, 20, 30, 60} seconds with freeze corruption.

Tunnel entry emits one trigger event. SUMO keeps the true state, while the
observation builder freezes the last valid position and speed for the declared
window. AoI describes the resulting observation and is not interpreted as a
causal dose of faithfulness. Random dropout is not part of the final
confirmatory matrix.

## 3. Margin-DEF as a sensitivity measure

DEF is implemented exactly as defined (§7.4: Comp/Suff vs size-matched
random explanations, normalized gains, DEF = ½(g_comp + g_suff)).
Probability-based DEF can have low numerical resolution when the selected
action probability is close to zero or one. The dissertation therefore adds
**margin-DEF**
— the same protocol measured on the decision's logit margin
(logit[a*] − max others, clamped) — computed from the *same*
counterfactual forward passes. Margin-DEF has a larger numerical scale on the
studied checkpoints. Probability-DEF remains the primary measure because it is
the proposal's declared definition. Margin-DEF is reported as an
action-protected sensitivity measure and is not substituted for the primary
endpoint after observing the results.

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
  wait are reported; average response time, fleet utilization and a
  separate task-completion *rate* are not, as pickups/50 requests and
  wait already characterize the low-performance regime the learned
  policies occupy.

## 5. Baseline implementation and interpretation

The proposal expects greedy dispatch to be a floor the learned policy beats.
The implemented greedy comparator selects each taxi's nearest request without
fleet-wide assignment or conflict resolution. On the final held-out demand it
averages 2.0 pickups, while the stochastic learned-policy means are 9.42-14.58
across GAT checkpoints. It is therefore retained as a deliberately naive lower
bound, not a competitive dispatch algorithm. The dissertation does not claim a
state-of-the-art performance comparison; performance is supporting context for
the explanation audit.

## 6. Offline dataset → online paired evaluation

The proposal (§7.6) describes logging clean episodes and applying the
staleness operator to the *logs* offline. The implementation applies
degradation live at the observation boundary instead, for a strictly
stronger design: the policy's *closed-loop* behavior under degradation
is measured (offline replay could not change the trajectory), while the
same mechanism still yields an exact per-decision clean twin for paired
drift measurement. Episode artifacts are versioned as JSON manifests
(network `tunnels.json`, demand `demand_manifest.json`, per-sweep
`manifest.json` with seeds and git revision) rather than .npz/Parquet.

## 7. Environment substitutions

- SUMO **1.20.0** (PyPI wheel) on the Intel-mac dev machine rather than
  1.27; the committed networks are format-1.20 and load natively.
  Colab remains on the 1.27 line per `docs/COLAB.md`.
- Episodes are 1200 s, not the proposal's "≈ 1 simulated hour"; demand
  is uniform-random OD via `randomTrips.py` seeds without the optional
  rush-hour/hotspot calibration.
- Primary evaluation is stochastic (sampled actions), matching policy training
  and retaining both no-op and dispatch decisions. A disclosed deterministic
  diagnostic shows that all six selected GAT checkpoints choose no-op over
  three held-out episodes each. Capability claims are therefore restricted to
  the sampled policy distributions rather than a deployable argmax policy.

## 8. Audited refinements to the DEF protocol (post-hoc, disclosed as findings)

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
characterization is itself one of the work's contributions. All
headline faithfulness claims cite the type-matched numbers; the
uniform-baseline numbers are retained to document the artifact.

## 9. Additions beyond the proposal

- **Decoupled explanation head** (occlusion-distilled scorer over detached
  encoder embeddings) — an architectural mitigation on the explanation side.
  It remains a legacy exploratory experiment and is not part of the final
  confirmatory matrix.
- **Chronological demand splits** implemented as seeded rider-demand
  variants (identical fleet, varied demand), giving held-out test
  demand for all headline results.

## 10. Action-stratified post-hoc diagnostic

The primary analysis pools sampled no-op and dispatch actions after excluding
records with no valid request. A later audit found that chosen no-op accounts
for 93.0% of eligible records. The final analysis therefore reports eligible
all-action, no-op, and dispatch strata separately. Episode blocks remain the
statistical unit. This diagnostic is disclosed as exploratory because it was
added after the primary analysis. It does not replace H1-H5; it limits their
interpretation by showing that the combined H1/H4 patterns are not reproduced
among the less common dispatch actions.
