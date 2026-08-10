# 1. Introduction

## 1.1 Context: explanations as an operational interface

Modern ride-hailing and logistics platforms increasingly delegate
dispatch decisions to multi-agent reinforcement learning (MARL).
Because dispatch is inherently relational — vehicles interact through
shared road space, competing demand, and nearby peers — graph neural
encoders, and graph attention networks (GAT) in particular, have become
a natural architectural choice [6], [7]. GAT-based dispatchers carry an
attractive by-product: per-decision attention distributions over
neighbouring vehicles and candidate orders, which platforms surface on
operator dashboards as *explanations* of why a vehicle was assigned to
an order. The practice mirrors the broader habit, inherited from the
Transformer literature [9], of reading attention as a window into model
reasoning.

Whether that window shows anything real is contested. Jain and Wallace
demonstrated that attention is often uncorrelated with gradient-based
importance and can be perturbed without changing predictions [12];
subsequent work argued the verdict depends on how faithfulness is
defined and tested [13], [14], [17]. This debate, however, has been
conducted almost entirely on clean, static NLP data. Operational fleet
telemetry is neither clean nor static.

## 1.2 The problem: explanations that outlive their data

Fleet telemetry — GPS position, velocity, occupancy — degrades in
structured, physically grounded ways. A vehicle entering a tunnel stops
transmitting; every consumer of its state, human or model, continues to
act on the last received reading while its Age of Information (AoI)
[21] grows. If an attention-based explanation continues to look
plausible and confident while the data behind it ages, the explanation
*outlives its data*: it offers accountability theatre precisely in the
conditions — degraded, congested, abnormal — where operators most need
genuine insight. This dissertation names the risk **faithfulness
decoupling**: the hypothesised phenomenon in which explanation
faithfulness degrades faster than task performance as telemetry ages,
so neither the dashboard nor the KPIs warn the operator that the
explanation has gone bad.

## 1.3 Research questions

The study is organised around four questions, formalised in the
research proposal and answered in full in Chapters 4–5:

- **RQ1a.** Are attention-based explanations reasonably faithful on
  clean telemetry to begin with?
- **RQ1.** How does explanation faithfulness change as telemetry
  degradation increases?
- **RQ2.** Does faithfulness degrade faster than dispatch performance
  (faithfulness decoupling)?
- **RQ3.** What mechanisms drive the change — in particular, does
  attention increasingly concentrate on stale telemetry?
- **RQ4.** Can telemetry-awareness mitigate the effect?

Five preregistered hypotheses (H1–H5, Chapter 3) operationalise these
questions. The proposal deliberately included a reinterpretation
clause: *if clean-data faithfulness is itself near zero, decoupling is
reinterpreted accordingly*. That clause turned out to govern the entire
story.

## 1.4 What the study found — and how the findings changed shape

Three results, in the order they were established:

1. **The built-in explanation channel is uninformative from the start
   (RQ1a).** Under the standard occlusion protocol the attention channel
   of a trained GAT-MAPPO dispatcher scores *worse than random*
   (margin-DEF −0.54). A construct-validity audit — prompted by a
   mock-review question — showed 98 % of that deficit to be a protocol
   artifact: in an architecture where explanation nodes double as action
   candidates, occluding a reservation node deletes the corresponding
   action, and uniform random baselines trip this mechanism far more
   often than attention's top-k does. Under a composition-matched
   baseline the channel scores ≈0 — across every capability level,
   architecture variant, and training seed tested. The honest verdict is
   *uninformative, not anti-informative*; the artifact characterisation
   itself became a methodological contribution.

2. **Under staleness, the explanation drifts silently (RQ1–RQ3).**
   Attention mass shifts onto stale nodes as soon as staleness exists
   (WAMSN 0→0.013; cluster-robust p = 10⁻⁴), and within episodes the
   decisions that attend most to stale nodes are the least faithful
   (negative correlation in 89 % of episodes). Meanwhile neither the
   faithfulness score (already at its floor) nor task performance
   (statistically flat) moves — no observable signal warns that the
   explanation's content has changed.

3. **No tested mitigation restores faithfulness (RQ4).**
   Degradation-aware training leaves the channel exactly as
   uninformative as clean training, across three seeds. An
   occlusion-distilled *decoupled* explanation head — the only channel
   to score above the fair random baseline — improves faithfulness
   modestly and significantly on clean data (+0.023, p = 10⁻⁴) but
   undetectably under degradation.

## 1.5 Contributions

- **C1.** Among the first empirical studies of explanation faithfulness
  under telemetry degradation in GAT-MARL fleet dispatch, with all five
  preregistered hypotheses resolved under cluster-robust statistics.
- **C2.** The DEF protocol for dispatch explanations — and, beyond the
  proposal, the demonstration that ERASER-style occlusion baselines are
  systematically confounded in candidate-action architectures, together
  with two practical controls (composition-matched baselines; protected
  chosen-action nodes).
- **C3.** WAMSN, a graded measure of explanation reliance on stale
  telemetry, with an operational reading (conditional exposure and
  top-k membership statistics).
- **C4.** Mechanistic evidence for the silent-drift phenomenon: the
  AoI → attention-shift link is robust, while the predicted downstream
  faithfulness decline is blocked by a floor effect that is itself the
  central finding.
- **C5.** Evidence that neither training-side nor architecture-side
  mitigation restores faithful explanation under degradation — sharpening
  the open problem rather than closing it.
- A fully reproducible benchmark: seed-pinned SUMO scenarios, versioned
  manifests, committed checkpoints, and an audited deviations register.

## 1.6 Thesis structure

Chapter 2 reviews the three literatures this work joins (RL dispatch,
attention faithfulness, AoI). Chapter 3 details the benchmark,
degradation framework, metrics, audit instruments, and statistical
methodology. Chapter 4 reports results as a three-act narrative
mirroring the discovery order. Chapter 5 discusses implications,
limitations, and the generality of the occlusion artifact. Chapter 6
concludes. Appendix A is the deviations register mapping every departure
from the proposal to its rationale.
