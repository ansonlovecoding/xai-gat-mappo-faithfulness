# Abstract

Graph-attention multi-agent reinforcement learning (GAT-MARL) can expose
attention weights as explanations of fleet-dispatch decisions. Those
explanations may remain visually plausible even when the vehicle telemetry
behind them is stale. This dissertation studies that trust problem under the
title *When Explanations Outlive Their Data: Faithfulness Decoupling in
Graph-Attention MARL Fleet Dispatch under Telemetry Degradation*.

The experiment uses a SUMO simulation of a 20-taxi fleet in a tunnel-rich area
of Chongqing. A tunnel entry triggers signal loss, while degradation itself is
applied at the policy's observation boundary. The simulator continues to hold
the true vehicle state, but the policy receives a frozen last-known position
and speed for a fixed 10, 20, 30, or 60 seconds. This design provides matched
clean and degraded observations without changing vehicle movement in SUMO.

Four policy conditions are trained with seeds 42, 43, and 44. The main audit
compares a clean-trained GAT policy (B2) with a GAT policy trained under
degradation (H5). Explanation quality is measured by decision-level
explanation faithfulness (DEF), using type-matched random occlusions, and by
weighted attention mass on stale nodes (WAMSN). Each trained policy is tested
with eight evaluation seeds and three episodes per condition. Checkpoints are
selected on validation demand before the held-out test runs.

The rerun gives a narrower result than the original hypothesis. H3 is the only
confirmatory result supported for every trained GAT policy: WAMSN increases
with outage duration (within-seed Spearman rho 0.078-0.121; Holm-adjusted
p = 0.0004 in all six runs). However, the exact paired shift of attention onto
stale nodes is positive for two training seeds and negative for one in both B2
and H5. H1 and H2 are unsupported in all six runs, so the experiment does not
show that longer outages reduce DEF or that faithfulness declines faster than
dispatch performance. The WAMSN-DEF association is also training-seed
dependent. Degradation-aware training does not remove this variability and one
H5 seed learns a substantially weaker dispatcher.

A supporting construct-validity audit explains why the conclusion must be
cautious: deleting a passenger-request node also deletes its action, whereas
deleting a nearby-taxi node only hides information. Type-matched controls are
therefore necessary to avoid attributing action-space changes to explanation
faithfulness. Overall, outage duration reliably changes stale-data exposure,
but its effect on explanation allocation and faithfulness is not stable across
independently trained policies. Attention maps should not be treated as
trustworthy explanations without model-specific faithfulness and freshness
checks.
