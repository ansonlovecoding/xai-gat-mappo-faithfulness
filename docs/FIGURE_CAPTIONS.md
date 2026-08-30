# Figure captions (dissertation-ready, English)

> **Legacy figure set.** These captions describe the archived
> `results/story_freeze_v1/` figures and are retained for audit history. The
> final v4 dissertation uses Figs. 4.1-4.4 and the captions embedded in
> `docs/dissertation/04_results.md`. Do not use the H2/H4 verdicts below as the
> final experiment result.

Numbering follows the suggested placement: Figs. 1–4 methods, Figs. 5–8
results (the three acts), Figs. 9–10 discussion. Every caption is
self-contained (readable without the body text), states the data
provenance, and includes the statistics a reader needs to interpret the
panel. Adjust figure numbers to match the final chapter layout.

---

**Figure 1** (`fig1_architecture.png`)
> **System architecture.** The control path (top row) runs from the SUMO
> microscopic simulator (ground-truth fleet state; tunnel edges imported
> from OpenStreetMap) through the telemetry-degradation layer — which
> freezes a vehicle's last valid reading at the observation boundary and
> never mutates simulator state — into the per-agent heterogeneous graph
> observation and the two-layer, four-head GAT encoder shared by all
> agents. The encoder feeds the actor head (Discrete(K+1): no-op or one
> of K candidate reservations) and exposes two candidate explanation
> channels: the attention weights (coupled — the same tensor that drives
> the decision) and a distilled scorer head reading detached embeddings
> (decoupled — no influence on the policy). Because degradation touches
> observations only, every degraded observation has an exact clean twin
> (dashed), enabling paired per-decision evaluation. The faithfulness
> pipeline (bottom) audits both channels via counterfactual occlusion
> (≈37 forward passes per scored decision), WAMSN, and attention drift.

**Figure 2** (`fig2_observation_graph.png`)
> **Per-agent observation graph and its node↔action mapping.** Each
> acting vehicle observes a self-centric heterogeneous graph: itself
> (blue; features x, y, episode time, speed, Age of Information), its
> K_n = 5 nearest peer taxis (green; relative position, availability,
> distance, AoI), and the K_r = 5 nearest pending reservations (orange;
> relative pickup/drop-off vectors, waiting time). Attention is a masked
> softmax over all valid nodes. Crucially, every reservation node
> corresponds one-to-one to an action: accepting reservation R_k is
> action k, and the no-op action is anchored on the self node. This
> node↔action identity is what makes naive occlusion protocols
> ill-posed in this architecture (Fig. 4).

**Figure 3** (`fig3_freeze_timeline.png`)
> **Freeze-based degradation and the max-AoI severity ladder**
> (illustrative trace, severity level 30 s). While a vehicle traverses a
> tunnel (shaded), it stops transmitting: every observer — the vehicle
> itself and all peers — continues to see its last valid position and
> speed (solid line) while the true state (dashed) diverges, and the
> reading's Age of Information grows (bottom panel). The signal remains
> lost after tunnel exit until AoI reaches the severity level
> (physically, receiver re-acquisition delay), so each ladder level
> {5, 15, 30, 60} s bounds the maximum AoI directly; on re-acquisition
> the observed state snaps back to truth and AoI resets to zero.
> Long tunnel transits can exceed low levels, so the empirical AoI
> distribution is reported alongside each level.

**Figure 4** (`fig4_def_protocol.png`)
> **The occlusion = action-deletion artifact and its control.**
> (a) The DEF protocol scores an explanation by occluding its top-k
> nodes and measuring the drop in the decision margin
> m = logit(a*) − max logit(other), compared against size-matched random
> occlusions. (b) In a candidate-action architecture this protocol is
> confounded: occluding reservation node R_k removes action k from the
> candidate set entirely — if a* itself is deleted the margin clamps to
> −cap regardless of information content. Uniform random baselines hit
> reservation nodes far more often than the studied policy's attention
> top-k does (clamp rates 14.6 % vs 1.8 % of counterfactual forwards),
> systematically inflating the baseline. (c) Matching the random
> baseline's node-type composition to the explanation's top-k removes
> 98 % of the apparent deficit (clean margin-DEF −0.554 → −0.009,
> n = 1 199 paired decisions): the correct reading is that attention is
> *uninformative*, not anti-informative.

---

**Figure 5** (`story_act1_clean_def.png`)
> **Act 1 — the built-in explanation carries no measurable
> decision-relevant information on clean telemetry.** Left: distribution
> of per-decision margin-DEF under the standard uniform-baseline
> protocol (B2 policy, clean held-out test demand, n = 3 565 scored
> decisions; mean −0.541, 95 % bootstrap CI [−0.556, −0.527]). Right:
> the same quantity under the two baseline schemes — the apparent
> "worse than random" verdict under the uniform baseline (−0.554)
> collapses to −0.009 (95 % CI [−0.016, −0.001]) once the random
> baseline is composition-matched, identifying 98 % of the deficit as
> the occlusion = action-deletion artifact of Fig. 4. Attention is
> statistically indistinguishable from (marginally below) a fair random
> explanation before any degradation is applied.

**Figure 6** (`story_act2_decoupling.png`)
> **Act 2 — attention shifts toward stale telemetry while neither
> faithfulness nor performance responds.** B2 policy evaluated across
> the max-AoI severity ladder (clean + {5, 15, 30, 60} s; 8 environment
> seeds × 3 held-out test-demand episodes per cell; ≈18 000 scored
> decisions). Top: margin-DEF is flat at its clean-data level at every
> severity — the channel is uninformative from the start (Fig. 5), so
> H1's predicted decline has no room to occur (episode-block permutation
> p = 0.65, cluster CI for ρ [−0.016, +0.025]). Middle: completed
> pickups are statistically flat (6.5–7.4 per episode). Bottom: WAMSN
> jumps from 0 to ≈0.013 as soon as staleness exists and stays elevated
> (H3: ρ = +0.092, episode-block permutation p = 10⁻⁴, cluster CI
> [+0.052, +0.128]). Within episodes, decisions placing more attention
> on stale nodes are less faithful (H4: mean within-episode ρ = −0.140,
> negative in 89 % of episodes, p = 10⁻⁴); the cell-level rate test
> supports H2 after Holm correction (p = 0.039). The explanation
> channel's content drifts onto stale data without any warning visible
> in either faithfulness or performance.

**Figure 7** (`story_act3a_training_mitigation.png`)
> **Act 3a — training-side mitigation has no detectable effect on
> explanation faithfulness.** The B2 policy (trained on clean telemetry,
> blue) and the H5′ policy (identical configuration trained *with*
> tunnel-triggered freeze degradation at outage 30 s, green) evaluated
> on the same severity ladder and protocol as Fig. 6. Both margin-DEF
> profiles are flat across severity; pickups and WAMSN are
> indistinguishable between the two policies. The vertical offset
> between the curves is expressed in the uniform-baseline metric and is
> not interpretable: under the composition-matched baseline both
> policies — and all three H5′ training seeds (42/43/44) — sit at
> margin-DEF ≈ 0 (−0.01 … +0.02; Fig. 9), i.e. degradation-aware
> training neither improves nor harms the channel's faithfulness.
> The proposal's H5, phrased to allow either outcome, is resolved as
> "no mitigation effect".

**Figure 8** (`story_act3b_coupled_vs_decoupled.png`)
> **Act 3b — a decoupled distilled explanation channel is modestly but
> significantly more faithful on clean telemetry.** Paired comparison of
> the coupled channel (attention) and the decoupled channel (a scorer
> distilled from single-node occlusion targets; held-out Spearman
> against occlusion importance +0.60) on identical decisions with
> identical composition-matched random baselines. Clean condition
> (n = 983 paired decisions): coupled −0.013 vs decoupled +0.010,
> Δ = +0.023, paired sign-flip p = 10⁻⁴ — the decoupled head is the only
> channel scoring above the fair random baseline. Under tunnel
> degradation at max-AoI 60 s (n = 878): Δ = +0.005, p = 0.21 — the
> advantage is not statistically detectable. Distillation buys a real
> but modest improvement on fresh data; no tested mitigation restores
> meaningful faithfulness under degradation.

---

**Figure 9** (`fig6_capability_spectrum.png`)
> **The uniform-baseline metric is uninterpretable across the capability
> spectrum; the composition-matched metric is stable at zero.** Clean
> margin-DEF for seven checkpoints spanning completed pickups 1→11.8 and
> policy entropy 1.56→0.03: an untrained network, mid-training and best
> B2 checkpoints and all three
> degradation-aware H5′ seeds. Under the uniform baseline (blue) the
> metric swings from −1.35 to +1.78 — the action-deletion artifact
> manufactures both "far worse than random" and "far better than random"
> verdicts from equally uninformative channels, depending only on where
> each policy's attention happens to sit relative to reservation nodes.
> Under the type-matched baseline (green) every checkpoint lies at ≈0
> (−0.01 … +0.02): the "attention is uninformative" finding is invariant
> to capability level, architecture, and training regime.

**Figure 10** (`fig5_causal_chain.png`)
> **The proposed causal chain with per-link audited verdicts.** The
> proposal hypothesised: telemetry degradation → AoI ↑ → attention
> drift → WAMSN ↑ → DEF ↓ → faithfulness decoupling. After
> cluster-robust statistics and the construct-validity audit, the chain
> resolves as: degradation and AoI growth hold by construction and
> measurement; attention drift is present but far weaker than expected
> (per-decision Jensen–Shannon divergence ≈ 0.000–0.001); the
> attention-mass shift onto stale nodes is robustly supported (H3,
> episode-block p = 10⁻⁴); the predicted DEF decline is not supported —
> a floor effect, since clean-data faithfulness is already ≈0 under the
> composition-matched baseline; and decoupling survives in its
> rate-based formulation (H2, Holm-corrected p = 0.039). The chain's
> failure point is itself the dissertation's central finding: the
> explanation channel is not faithful enough for staleness to have
> anything to degrade.
