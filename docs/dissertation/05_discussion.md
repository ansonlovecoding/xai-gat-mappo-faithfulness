# 5. Discussion

## 5.1 What the findings mean

**For the title question.** Explanations do outlive their data — but
not in the way the proposal anticipated. The expected pattern was a
faithful channel losing fidelity as telemetry aged. What the audited
data show is starker: the channel never had measurable fidelity
(Act 1), its content nevertheless shifts visibly onto stale telemetry
as soon as staleness exists (Act 2, H3/H4), and no signal available to
an operator — faithfulness score, task KPIs, or the explanation's own
appearance of confidence — distinguishes the drifted state from the
fresh one. An operator dashboard built on this channel would present
equally plausible-looking attention maps before and during degradation,
with the stale-node share of the story quietly growing. That is
accountability theatre in exactly the conditions transparency
obligations are meant to cover.

**For the mechanism chain** (Fig. 10). The proposal's causal chain
survives at its front — degradation → AoI ↑ → attention mass on stale
nodes ↑ (robust) — and breaks at DEF ↓, for a reason that is itself the
finding: a floor effect. The chain's healthy links make the hazard
concrete; its broken link relocates the problem from "faithfulness
degrades" to "faithfulness was never there".

**For mitigation.** The negative results are informative. Experiencing
degradation during training gives the policy no incentive to make its
attention informative — attention serves computation, not
communication, and gradient pressure on task reward does not change
that (H5). Distillation from occlusion targets — direct optimisation
for faithfulness — helps, but modestly and only on fresh data (Act 3b).
The honest engineering conclusion: replace, don't trust, the built-in
channel; and do not expect cheap fixes to survive degradation.

## 5.2 The silent-drift hazard, operationally

WAMSN's raw magnitudes are small (pooled ≈0.013) because geographic
exposure is small (~2 % of decisions; ~8 % have any stale node
visible). The operator-facing translations matter more: conditional on
exposure, stale nodes carry 14 % of attention mass and enter the top-3
explanation in 8.5 % of decisions. On a metropolitan fleet taking
thousands of dispatch decisions per hour, "one in twelve explanations
near a tunnel silently features stale telemetry among its headline
factors" is a concrete, monitorable risk statement — and WAMSN-style
conditional monitoring is implementable on any platform that logs
attention and AoI, independent of this study's specific policy.

## 5.3 The occlusion artifact as a general warning

The methodological finding generalises beyond this benchmark. Any
explainable-RL system in which explanation units index action
candidates — pointer networks over candidate sets, attention over
requests in assignment problems, action-graph policies — shares the
structural property that occlusion perturbs the *decision problem*,
not merely the input. The capability-spectrum result (Fig. 9) shows the
consequence is not a small bias but sign-flipping uninterpretability:
the same protocol certified one checkpoint "far worse than random"
(−1.35) and another "far better" (+1.78) when both are ≈0 under a fair
baseline. Two controls suffice in practice — composition-matched random
baselines and protection of the chosen action's unit — and both are
cheap. Perturbation-based faithfulness evaluation in RL should treat
them as mandatory, the way [16]'s protocol treats size-matching.

This also reframes part of the "attention is not explanation" debate
for RL: some prior negative (or positive) verdicts obtained with
uncontrolled occlusion in candidate-action settings may partly reflect
this artifact rather than the property under test.

## 5.4 Limitations

- **Policy competence.** The learned dispatchers reach 6–8 pickups
  against a greedy matcher's 32, and entropy-collapse during training.
  The capability-spectrum analysis shows the uninformativeness finding
  is stable across the range *available* (1→11.8 pickups), but no
  checkpoint approaches competent dispatch; whether a genuinely strong
  GAT dispatcher develops informative attention is open.
- **Scale and scope.** One district, one city, 20 taxis, 50 riders,
  1 200 s episodes; tunnel exposure ~2 % of decisions. The severity
  ladder tops at 60 s while geography occasionally produces AoI up to
  ~310 s (an idling vehicle on a tunnel edge); observed AoI features
  saturate at the 60 s normalisation cap.
- **Statistical residuals.** H2 clears Holm at p = 0.039 — real but not
  overwhelming; a same-configuration smaller sweep earlier in the study
  was non-significant, and the result stabilised only at 8 seeds.
  Per-decision drift and churn are descriptive only.
- **Single-run aspects.** B2 is one training seed (H5′ has three); the
  distilled head is one distillation run. Checkpoint selection interacts
  with entropy collapse (two of three H5′ bests select before epoch 25).
- **Metric scope.** Faithfulness is operationalised solely through
  occlusion-based counterfactuals; margin-DEF's clamp parameter and the
  60 s AoI normalisation are audited but particular choices.

## 5.5 Future work

(1) A dispatcher trained to competence (curricula, entropy schedules,
or offline RL) to test whether informative attention emerges with
capability. (2) Faithfulness-aware training that optimises the
explanation channel directly during RL, rather than post-hoc
distillation. (3) Extending the artifact analysis to other
candidate-action XRL systems and to gradient-based attribution, which
may inherit an analogous confound through the action mask. (4) An
operator-facing staleness monitor built on conditional WAMSN, evaluated
in a human-factors study. (5) Multi-city transfer and richer degradation
(urban canyons, backhaul delay) under the same AoI framework.
