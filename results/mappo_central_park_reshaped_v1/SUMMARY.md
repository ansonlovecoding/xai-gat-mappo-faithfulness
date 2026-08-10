# MAPPO run — central_park, reshaped reward (v1)

First working GAT-MAPPO training run that beats the non-learning baselines.
Preserved here (rather than left in the timestamped `runs/mappo/…` dir) so
future evaluation and dissertation write-up can reference a stable path.

## Environment & policy config

- **Scenario**: `scenarios/yubei/central_park` — 20 taxis, 50 riders,
  1200 s episode, 3 navigable tunnel edges (degradation OFF for this run)
- **Reward** (reshaped from the v0 defaults):
  - `+10.0` per completed pickup
  - `+0.5` per successful dispatch (partial credit for attempting)
  - `−0.001` per second of mean pending wait time
- **Policy**: `DispatchGATPolicy`, hidden_dim=64, 2 GAT layers, 4 heads
  → 113 539 parameters
- **PPO**: lr=3e-4, gamma=0.99, gae-lambda=0.95, clip=0.2, ppo-epochs=4,
  entropy-coef=0.01 (this turned out to be too low — see caveats)
- **Device**: MPS (Apple Silicon)
- **Training**: 300 epochs × ~2 s each ≈ 10 min wall-clock

Full args in `args.json`. Per-epoch metrics in `train_log.jsonl`.

## Evaluation results (3 stochastic episodes each)

| Method | Pickups (mean ± std) | Reward (mean) | Notes |
|---|---:|---:|---|
| Random baseline | 3.0 | −43.8 | uniform over Discrete(K+1) |
| NearestReservation baseline | 1.0 | −36.7 | greedy nearest, conflict-blind |
| **MAPPO `ckpt_epoch_0050`** | **7.67 ± 2.05** | +43.6 | **best checkpoint** |
| MAPPO `ckpt_epoch_0100` | 4.33 ± 0.47 | +1.4 | already degrading |
| MAPPO `ckpt_epoch_0250` | 2.33 ± 0.47 | −19.9 | worse than random |
| MAPPO `ckpt_epoch_0299` (final) | 0.00 | −45.6 | collapsed |
| SUMO greedy (upper reference) | 30 | — | built-in bipartite matcher |

**Use `ckpt_epoch_0050.pt` as "the trained model"** for any downstream work.

Baseline numbers are from `runs/baselines.json` and correspond to the same
central_park scenario, same seed=42.

## What the training curve shows

Per-epoch means in 30-epoch buckets:

```
 bucket   pickups     reward   V_loss     H
   0-29     4.43     +36.13    44.73  0.412
  30-59     5.23     +21.65    57.52  0.074   ← best-of-run epoch #41 = 17 pickups
  60-89     4.77     +10.53    33.36  0.069
  90-119    4.60      +5.09    21.19  0.048
 120-149    5.07     +13.39    31.30  0.055
 150-179    3.33      -8.32    14.94  0.043
 180-209    3.93      -2.33    17.98  0.033
 210-239    1.77     -25.53    14.83  0.020
 240-269    3.07     -11.90    14.03  0.025
 270-299    0.27     -42.44    17.70  0.005   ← catastrophic collapse
```

## Findings and caveats

1. **Reward reshaping was the key change.** A pre-reshape 300-epoch run
   (identical hyperparameters, `pickup_reward=1.0`, `wait_lambda=0.01`)
   converged to the *do-nothing* policy: 0 pickups, 0 dispatches. Rescaling
   pickups 10× and shrinking the wait penalty 10× flipped this.
2. **Entropy collapsed from 0.41 → 0.005.** The default `--ent-coef 0.01`
   was not strong enough to keep exploration alive. This is the root cause
   of the late-run degradation (bucket 270-299) — the policy had committed
   to a bad deterministic strategy and could not escape.
3. **No "best-checkpoint tracking" in `train.py`.** Best model is `ckpt_50`;
   the final saved checkpoint (`ckpt_299`) is unusable. A follow-up run
   should track best-so-far by rolling-mean pickups on a held-out seed.
4. **Still 4× below SUMO's built-in greedy.** That gap is what the rest of
   the dissertation has to close, both on performance and (more importantly)
   on faithfulness of explanations.

## Suggested follow-up training

```bash
python scripts/train.py --area central_park --epochs 500 \
  --pickup-reward 10.0 --dispatch-reward 0.5 --wait-lambda 0.001 \
  --ent-coef 0.03 \       # was 0.01 — keep exploration alive
  --lr 1e-4 \             # was 3e-4 — slower, more stable updates
  --clip-ratio 0.15       # was 0.2 — smaller policy steps
```

## How to reproduce the eval numbers above

From the project root, with the venv active:

```bash
python scripts/eval_policy.py results/mappo_central_park_reshaped_v1/ckpt_epoch_0050.pt --episodes 3 --stochastic
```

## Files in this directory

- `args.json` — CLI args at training time (full hyperparameter set)
- `train_log.jsonl` — one JSON object per epoch (metrics)
- `ckpt_epoch_XXXX.pt` — model + optimizer + config, saved every 50 epochs
- `ckpt_epoch_XXXX.eval.json` — eval summary per checkpoint (if evaluated)
- `SUMMARY.md` — this file
