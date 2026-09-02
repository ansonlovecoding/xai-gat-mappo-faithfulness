# Abstract

Graph-attention multi-agent reinforcement learning can expose attention
weights as candidate explanations of fleet-dispatch decisions. However, an
attention map may remain visible after the vehicle telemetry behind it becomes
stale. This dissertation asks whether raw graph-attention weights can be
trusted as explanations under clean and degraded telemetry.

The experiment simulates 20 taxis in SUMO. Tunnel entry triggers an
observation-layer outage that freezes an affected taxi's last valid position
and speed for 10, 20, 30, or 60 seconds while SUMO retains the current traffic
state. This creates paired clean and degraded observations for the same
decision. GAT policies trained on clean data or with 30-second outages are
evaluated across three independent training seeds.
Decision-level explanation faithfulness (DEF) compares attention-ranked nodes
with type-matched, action-protected random controls. WAMSN separately measures
the attention assigned to stale vehicle information. Additional checks cover
leave-one-out rankings, attention extraction choices, action type, and random
telemetry-loss triggers.

The results do not validate raw attention as a reliable explanation. Clean
DEF remains close to its matched control and varies across checkpoints. Longer
outages increase stale-data exposure, but paired attention and DEF shifts
reverse direction for one training seed. The main associations are dominated
by no-op decisions and do not reproduce for dispatch actions.
Checkpoint-dependent differences remain under random triggering and
degradation-aware training. These results do not reliably show whether
attention reflects fresh information or decision relevance.

To address this problem, the dissertation proposes a freshness-aware
explanation audit framework to decide when attention may be presented as an
explanation. The framework reports telemetry freshness and action
composition separately, applies action-aware and type-matched faithfulness
tests, checks sensitivity to the attention extraction rule, and requires
consistent evidence across checkpoints and training seeds. If the evidence is
inconsistent, attention remains an internal diagnostic rather than a
trustworthy explanation. The framework governs when explanations may be
released; it does not repair faithfulness inside the model.
