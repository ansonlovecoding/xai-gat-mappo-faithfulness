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
evaluated across five independent training seeds. Decision-level explanation
faithfulness (DEF) compares attention-ranked nodes with type-matched,
action-protected random controls. Weighted attention mass on stale nodes
(WAMSN) separately measures attention-weighted vehicle-information age.
Further checks cover leave-one-out rankings,
attention extraction, action type, and random telemetry-loss triggers.

The results do not validate the tested layer- and head-averaged self-row
attention channel as a reliable explanation. Under clean telemetry,
pooled dispatch-action margin-DEF is below its matched-random control,
while a leave-one-out positive control is above zero. Longer outages
consistently increase stale-data exposure, but faithfulness does not respond consistently across checkpoints. Attention shifts toward stale nodes in seven checkpoints and away in three. Paired probability-DEF intervals are positive in seven, negative in two, and inconclusive in one. Both seed-45 checkpoints support the predicted duration- and staleness-related decline; most checkpoints do not. No-op and dispatch directions differ in nine checkpoints, and alternative attention heads or layers can reverse the stale-attention result. Therefore,
the tested weights do not provide stable evidence of decision relevance.

The dissertation develops and applies a freshness-aware explanation audit framework that
reports freshness and action composition separately, applies action-aware
faithfulness tests, checks the attention extraction rule, and requires
consistent evidence across checkpoints. Attention remains an internal
diagnostic when these checks fail. Applied to the collected evidence, the framework returns `WITHHOLD` for both
model families and all ten checkpoints, restricting the tested attention
channel to diagnostic use.
