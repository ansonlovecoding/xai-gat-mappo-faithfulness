# B1/B2/B3 matched trio + first full severity sweep (July 2026)

First matched comparison of the proposal's learned baselines at a single
code state, plus the first run of the complete H1–H4 pipeline. Trained on
the **Intel-mac environment** (SUMO 1.20.0 via the eclipse-sumo PyPI wheel,
torch 2.2.2 on CPU, socket TraCI) — numbers are internally consistent but
not directly comparable to the older Apple-Silicon/SUMO-1.27 keepers.

## Setup

All three: central_park, 300 epochs, seed 42, CTDE centralised critic,
reward shaping defaults (pickup 10.0 / dispatch 0.5 / wait λ 0.001),
degradation **off** during training. B2/B3 additionally sampled train-time
DEF/WAMSN every 10 epochs (`--faith-every-epochs 10 --faith-sample-size 64`).

| Run | Policy | Params | Command deltas |
|---|---|---:|---|
| `B1_mlp/` | MAPPO + MLP (no graph) | 106,487 | `--policy mlp` |
| `B2_gat/` | GAT-MAPPO (proposed) | 113,731 | — |
| `B3_gat_noaoi/` | GAT-MAPPO, AoI-unaware | 113,731 | `--aoi-unaware` |

## Results (clean condition)

Rolling-mean-of-10 training pickups (best window) vs held-out eval
(5 episodes, **stochastic** — see finding 1):

| Run | best epoch | rolling best | eval pickups |
|---|---:|---:|---:|
| B2 GAT | 91 | 14.2 | 5.80 ± 1.72 |
| B1 MLP | 39 | 12.9 | 6.60 ± 2.24 |
| B3 no-AoI | 25 | 11.5 | 8.20 ± 1.60 |

Non-learning references under this SUMO version (`baselines_sumo120.json`):
sumo_greedy **32**, nearest 1, random 0.

The three learned policies are statistically indistinguishable at n=5
clean episodes; rolling-best numbers are max-selected and upward-biased.
B2's edge must be argued from its explanation channel, not clean pickups.

## Severity sweep + hypothesis tests (`sweep_B2_stoch/`)

B2's `ckpt_best.pt`, 27 cells: clean + dropout_rate ∈ {.05,.1,.2,.4}
+ tunnel position-noise ∈ {10,20,40,80} m, seeds {42,43,44}, 2 episodes
per cell, stochastic actions, per-decision records (~8k decisions).

- **H1** (severity ↑ → DEF ↓): supported on the dropout axis
  (ρ=−0.21, p=1e-4); not on the tunnel axis (tunnel exposure is only
  ~2.4 % of agent-steps, so noise magnitude barely moves pooled DEF).
- **H2** (DEF declines faster than performance): **not supported yet**
  (p=0.099, n=12 cells) — but only because *performance does not decline
  at all* under these severities, while DEF measurably does. Qualitatively
  this is the decoupling story; the rate-ratio test needs a policy with
  performance worth losing, or more cells.
- **H3** (severity ↑ → WAMSN ↑): strongly supported on both axes
  (dropout ρ=+0.71; tunnel ρ=+0.11; both p=1e-4). Cleanest result.
- **H4** (WAMSN–DEF negative): supported (ρ=−0.30, p=1e-4, pooled).

## Findings worth citing

1. **Argmax evaluation degenerates on entropy-collapsed policies.** All
   three runs entropy-collapse (B2 →0 pickups after ~epoch 130, B3 after
   ~55); pickups come from residual stochasticity, so deterministic eval
   yields exactly 0 pickups (reward −45.59). All evaluation in this
   project is therefore stochastic; a deterministic sweep of the same
   checkpoint (`runs/sweeps/B2_main`, not kept) showed 0 pickups in every
   cell, making H2 vacuous — kept sweep is the stochastic one.
2. **DEF saturates on overconfident policies.** With entropy ≈0.03,
   π(a*)≈1 regardless of node occlusion → comp/suff/DEF ≈ 0. Absolute DEF
   deltas in the sweep are ~1e-3 despite significant trends. Candidate
   fixes: logit-margin-based DEF, temperature scaling, higher ent-coef.
   B2's best checkpoint (epoch 91, entropy 0.233) is the least saturated —
   its performance peak coincides with a transient entropy recovery.
3. **Statistical caveat**: per-decision Spearman p-values ignore
   within-episode clustering; add a cluster bootstrap before publishing.

## Reproduce

```bash
python scripts/train.py --area central_park --epochs 300 --save-every 50 \
  --faith-every-epochs 10 --faith-sample-size 64          # B2 (add --aoi-unaware for B3)
python scripts/train.py --area central_park --epochs 300 --save-every 50 --policy mlp   # B1
python scripts/eval_policy.py <ckpt_best.pt> --episodes 5 --stochastic
python scripts/sweep_severity.py <B2 ckpt_best.pt> --episodes 2 --seeds 42 43 44 \
  --faithfulness-every 8 --out runs/sweeps/B2_main_stoch
python scripts/analyze_hypotheses.py runs/sweeps/B2_main_stoch
```
