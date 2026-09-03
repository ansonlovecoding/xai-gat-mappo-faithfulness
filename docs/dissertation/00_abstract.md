# Abstract

Graph-attention multi-agent reinforcement learning (MARL) can expose attention
weights as explanations of fleet-dispatch decisions. However, an
attention map may remain visible after its supporting vehicle telemetry becomes
stale. This dissertation asks whether raw graph-attention weights can be
trusted as explanations under clean and degraded telemetry.

The experiment simulates 20 taxis in Simulation of Urban Mobility (SUMO).
Tunnel entry triggers an
observation-layer outage that freezes an affected taxi's last valid position
and speed for 10, 20, 30, or 60 seconds while SUMO retains the current traffic
state. This produces paired clean and degraded observations for the same
decision. Graph-attention network (GAT) policies trained on clean data or with
30-second outages are
evaluated across three independent training seeds. Decision-level explanation
faithfulness (DEF) compares attention-ranked nodes with type-matched,
action-protected random controls. Weighted attention mass on stale nodes
(WAMSN) separately measures attention assigned to stale vehicle information.
Further checks cover leave-one-out rankings,
attention extraction, action type, and random telemetry-loss triggers.

The results do not validate raw attention as a reliable explanation. Clean
corrected DEF remains near zero and varies across checkpoints. Longer outages
increase stale-data exposure, but paired attention and DEF shifts reverse
direction for one training seed. The main associations are dominated by no-op
decisions and do not reproduce for dispatch actions. The seed-dependent pattern
also appears under random triggering and in both training regimes. Therefore,
the tested attention weights do not provide consistent evidence of freshness
or decision relevance.

The dissertation proposes a freshness-aware explanation audit framework that
reports freshness and action composition separately, applies action-aware
faithfulness tests, checks the attention extraction rule, and requires
consistent evidence across checkpoints. Attention remains an internal
diagnostic when these checks fail. The framework governs explanation release;
it does not repair the model. Its demonstrative application returns `WITHHOLD`
for both model families and all six checkpoints.
