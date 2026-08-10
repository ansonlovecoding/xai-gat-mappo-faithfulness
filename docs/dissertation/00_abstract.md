# Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) is
increasingly used for fleet dispatch, and its attention weights are
routinely surfaced on operator dashboards as built-in explanations of
dispatch decisions. In real operations the telemetry feeding such
systems degrades — vehicles entering tunnels lose GPS and V2X signal —
so the data behind an explanation can be stale precisely when operators
most need to trust it. This dissertation asks whether attention-based
explanations remain faithful as telemetry ages, formalising the risk as
*faithfulness decoupling*: explanations that outlive their data.

A reproducible benchmark is built: a SUMO simulation of a tunnel-rich
district of Chongqing with a 20-taxi fleet, a GAT-MAPPO dispatcher, a
freeze-semantics degradation layer parameterised by maximum Age of
Information (AoI), and a per-decision faithfulness protocol (DEF,
occlusion-based; WAMSN, attention mass on stale nodes) with exact
paired clean/degraded observations.

Three findings emerge. First, the built-in attention channel carries no
measurable decision-relevant information even on clean telemetry: an
apparent "worse than random" faithfulness score (margin-DEF −0.55) is
shown, via a construct-validity audit, to be 98 % a protocol artifact —
in candidate-action architectures, occluding a reservation node deletes
the corresponding action, and uniform random baselines are
systematically inflated. Under a composition-matched baseline the
channel scores ≈0 across every capability level, architecture variant,
and training seed tested. Second, under staleness the explanation's
content silently shifts onto stale nodes (WAMSN 0→0.013, p = 10⁻⁴,
cluster-robust) while neither faithfulness nor task performance
responds — the operational hazard in its purest form. Third, neither
mitigation tested restores faithfulness: degradation-aware training has
no effect in either direction, while an occlusion-distilled decoupled
explanation head yields a modest, significant improvement on clean data
only (+0.023, p = 10⁻⁴).

Beyond the empirical verdicts, the work contributes the artifact
characterisation itself — a warning applicable to any explainable-RL
system whose explanation units double as action candidates — together
with an open, seed-pinned benchmark and audit trail.
