# Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) can expose
attention weights as explanations of fleet-dispatch decisions. The map remains
available when vehicle telemetry becomes stale, but does not show whether its
evidence is current or decision-relevant. This dissertation asks:
**can raw graph-attention weights be trusted as explanations when fleet
telemetry is degraded?**

The experiment simulates 20 taxis in SUMO. Tunnel entry triggers signal loss,
and the observation layer freezes the affected vehicle's last-known position
and speed for 10, 20, 30, or 60 seconds. SUMO retains the true state, allowing
paired clean and degraded observations. Clean-trained and degradation-trained
GAT policies use three independent training seeds. Decision-level explanation
faithfulness (DEF) compares attended nodes with type-matched random nodes, while
WAMSN measures stale-data exposure. A construct-validity audit controls for
request nodes that also represent actions.

The results do not validate raw attention as a dependable explanation. Under
clean telemetry, DEF varies in sign across checkpoints and is lower than a
leave-one-out perturbation control in all six policies. Longer outages increase
stale-data exposure, but the paired change in attention toward stale nodes is
positive for two training seeds and negative for one under both training
regimes. DEF does not consistently decline with outage duration, and the result
also depends on the layer, head, and query row used for explanation.

The conclusion is not that stale telemetry always reduces faithfulness. It is
that raw attention provides no stable guarantee of either data freshness or
decision relevance across independently trained policies. An attention map can
therefore outlive the data supporting it while still appearing complete.
Attention weights should be treated as model internals unless each released
checkpoint passes separate freshness and faithfulness checks.
