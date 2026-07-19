# 6. Conclusion

This dissertation set out to measure whether the attention weights that
fleet-dispatch platforms present as explanations remain faithful when
the telemetry behind them ages. It built the first benchmark coupling a
GAT-MARL dispatcher to physically grounded, AoI-parameterised telemetry
degradation, with per-decision paired clean/degraded evaluation, and
resolved all five preregistered hypotheses under cluster-robust
statistics.

The answers reshape the question. The built-in attention channel of the
studied dispatcher is not a degraded explanation — it is not an
explanation at all: under a construct-valid protocol it is
statistically indistinguishable from a composition-matched random
attribution on clean data, a verdict invariant across every capability
level, architecture variant, and training regime tested. What
degradation adds is not a loss of fidelity but a silent change of
content: attention mass shifts measurably onto stale telemetry
(robustly supported), the least faithful decisions are precisely those
attending most to stale nodes, and neither task performance nor any
property of the explanation itself signals that anything has changed.
"Explanations outlive their data" proved true in a form more troubling
than hypothesised: the explanation's authority never rested on fidelity
in the first place, and staleness erodes even the correspondence its
appearance suggests.

Neither tested remedy restores faithfulness. Training under degradation
leaves the channel unchanged; distilling a decoupled head from
occlusion targets produces the study's only above-random channel, with
a modest advantage that vanishes under degradation. Faithful dispatch
explanation remains an open problem — now with a measured baseline, a
mechanism, and two negative results marking the paths that do not work.

The study's most transferable contribution emerged unplanned: the
demonstration that occlusion-based faithfulness protocols are
uninterpretable in candidate-action architectures — capable of
certifying the same uninformative channel as far worse or far better
than random — together with two inexpensive controls that repair them.
The audit trail by which this artifact was found, verified, and
propagated through every prior conclusion is preserved in the released
benchmark, and is offered as a template for how explainability claims
in reinforcement learning ought to be stress-tested before they are
believed.
