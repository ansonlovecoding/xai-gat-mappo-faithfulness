# Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) can expose
attention weights as explanations of fleet-dispatch decisions. The practical
problem is that an attention map remains available when vehicle telemetry is
stale, but it does not warn the operator whether its highlighted evidence is
current or decision-relevant. This dissertation asks one central assurance
question: **can raw graph-attention weights be trusted as explanations when
fleet telemetry is degraded?**

The experiment uses a SUMO simulation of a 20-taxi fleet in Chongqing. Tunnel
entry triggers signal loss, while the policy's observation layer freezes the
last-known vehicle position and speed for 10, 20, 30, or 60 seconds. SUMO keeps
the true state, allowing clean and degraded versions of the same decision to be
compared. Clean-trained and degradation-trained GAT policies are evaluated
across three training seeds. Decision-level explanation faithfulness (DEF)
tests whether highly attended nodes affect the chosen action more than
type-matched random nodes. Weighted attention mass on stale nodes (WAMSN)
measures stale-data exposure.

The experiment does **not** validate raw attention weights as dependable
explanations. First, under clean telemetry, DEF remains close to the
type-matched random baseline, so attention shows no measured faithfulness
advantage over a fair random ranking. Second, longer outages increase
stale-data exposure in all six GAT policies, but the actual shift of attention
toward stale nodes is positive for two training seeds and negative for one in
both training regimes. Third, DEF does not decline with outage duration, and
degradation-aware training does not remove this variation.

A supporting construct-validity audit shows why type matching is necessary:
deleting a passenger-request node can also delete an action, whereas deleting a
nearby-taxi node only hides information. The consistent conclusion is not that
stale telemetry always reduces faithfulness. It is that attention provides no
stable guarantee of freshness or decision relevance across independently
trained policies. An attention map can therefore outlive the data supporting it
while still looking complete. Attention weights should be treated as model
internals unless each released checkpoint passes separate freshness and
faithfulness checks.
