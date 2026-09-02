# Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) can expose
attention weights for use as candidate explanations of fleet-dispatch
decisions. An attention map can remain visible when vehicle telemetry becomes
stale, but it does not show whether its evidence is current or
decision-relevant. This dissertation asks:
**can raw graph-attention weights be trusted as explanations when fleet
telemetry is degraded?**

The experiment simulates 20 taxis in SUMO. Tunnel entry triggers a simulated
outage, and the observation layer freezes the affected vehicle's last-known
position and speed for 10, 20, 30, or 60 seconds. SUMO retains the true state,
allowing paired clean and degraded observations. GAT policies trained on clean
observations and on 30-second outage observations are evaluated across three
independent training seeds. Decision-level explanation faithfulness (DEF)
compares highly attended node subsets with type-matched random subsets, while
WAMSN measures stale-data exposure. A construct-validity audit controls for
request nodes that also define available actions.

Under stochastic action sampling, all selected policies exceed legal-random
and naive greedy lower bounds on held-out demand. A deterministic diagnostic,
however, produces no pickups for any GAT checkpoint. The reported capability
therefore belongs to the sampled policy distribution, not to a reliable
argmax dispatcher.

The results do not validate raw attention as a dependable explanation. Of the
56,550 decisions with at least one available request, 93.0% select no-op.
Action-stratified analysis shows that the main duration and exposure
associations are not reproduced for dispatch actions. In four of six
checkpoints, the paired DEF direction also differs between the all-decision and
dispatch strata. Under
clean telemetry, DEF varies in sign across checkpoints and is lower than a
leave-one-out perturbation control in all six selected checkpoints. Longer
outages increase stale-data exposure, but the paired change in attention toward
stale nodes is positive for two training seeds and negative for one under both
training regimes. Within-policy tests show an association between greater
exposure and lower DEF, but the direct paired DEF change is negative for four
checkpoints and positive for two. The result also depends on the selected
checkpoint and the rule used to aggregate attention into a single explanation.
A random-trigger control retains the attention-shift direction in five of six
checkpoints and the DEF-shift direction in four of six, so the variation cannot
be attributed only to the fixed tunnel location.

The conclusion is not that stale telemetry always reduces faithfulness. It is
that raw attention provides no stable guarantee of either data freshness or
decision relevance across independently trained policies or action types. An attention map can
therefore outlive the data supporting it while still appearing complete.
In response, the dissertation proposes a freshness-aware explanation assurance
approach that separates freshness monitoring from attention, applies
action-aware faithfulness tests, checks aggregation sensitivity, and repeats
the audit for each released checkpoint. Attention weights should remain model
internals unless these checks give consistent results across independent
training runs.
