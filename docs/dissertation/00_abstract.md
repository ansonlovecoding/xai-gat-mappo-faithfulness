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

The results do not validate raw attention as a reliable explanation. Under
clean telemetry, action-aware margin-DEF is below its matched-random control,
while a leave-one-out positive control is above zero. Longer outages
consistently increase stale-data exposure, but they do not produce the proposed
decline in faithfulness. Attention shows a clear shift toward stale nodes in
three checkpoints, a clear shift away in two, and no clear shift in one, while
paired probability DEF increases slightly in all six.
No-op and dispatch decisions move in opposite DEF directions, and alternative
attention heads or layers can reverse the stale-attention result. Therefore,
the tested weights do not provide stable evidence of decision relevance.

The dissertation develops and applies a freshness-aware explanation audit framework that
reports freshness and action composition separately, applies action-aware
faithfulness tests, checks the attention extraction rule, and requires
consistent evidence across checkpoints. Attention remains an internal
diagnostic when these checks fail. The framework governs explanation release;
it does not repair the model. Its demonstrative application returns `WITHHOLD`
for both model families and all six checkpoints.
