# A/B — centralised vs decentralised critic (seed 42, central_park)

Head-to-head test that motivated flipping `PolicyConfig.centralised_critic`
to True by default (from False). Same seed, same reward-shaping, same
hyperparameters — the only difference between the two runs is the
`--centralised-critic` flag.

## Setup

- Scenario: `central_park`
- Episodes: 150 per arm
- Seed: 42
- Reward: `+10 pickup · +0.5 dispatch · −0.001 wait`
- PPO: lr 3e-4, gamma 0.99, gae-lambda 0.95, clip 0.2, ppo-epochs 4, ent 0.01
- Policy: hidden_dim 64, 2 GAT layers, 4 heads
- SUMO backend: plain TraCI (`DISPATCH_MARL_FORCE_TRACI=1`)
  because `libsumo` segfaults on the third `traci.load()` on this build

## Command lines used

```bash
# A — decentralised (per-agent V)
DISPATCH_MARL_FORCE_TRACI=1 python scripts/train.py \
  --area central_park --epochs 150 --seed 42 \
  --pickup-reward 10.0 --dispatch-reward 0.5 --wait-lambda 0.001 \
  --best-window 10 --save-every 50 \
  --log-dir runs/ab_test/decentralised

# B — centralised critic (joint pool across the batch)
DISPATCH_MARL_FORCE_TRACI=1 python scripts/train.py \
  --area central_park --epochs 150 --seed 42 \
  --pickup-reward 10.0 --dispatch-reward 0.5 --wait-lambda 0.001 \
  --best-window 10 --save-every 50 --centralised-critic \
  --log-dir runs/ab_test/centralised
```

## Headline results

| Metric | A (decentralised) | B (centralised) | Δ |
|---|---:|---:|---:|
| Mean pickups per episode | 4.07 | **4.81** | **+18 %** |
| Total pickups over 150 eps | 611 | **721** | +110 |
| Best rolling-mean-of-10 pickups | 8.00 (epoch 19) | **8.80 (epoch 117)** | +10 % |
| Best single-episode pickups | 17 (epoch 12) | 17 (epoch 21) | tie |

## Where B pulls ahead — late-training buckets

The pattern is instructive: A and B track each other for the first ~90
epochs, then A collapses (classic PPO late-run entropy failure) while B
recovers.

| Bucket | A pickups | B pickups | A reward | B reward |
|---|---:|---:|---:|---:|
| 0-14 | 3.60 | 4.80 | +36 | +37 |
| 45-59 | 4.27 | 3.80 | +6 | −5 |
| 90-104 | 4.73 | 5.13 | +12 | +11 |
| **105-119** | **1.87** | **6.20** | −23 | **+24** |
| 120-134 | 3.67 | 2.73 | +6 | −15 |
| **135-149** | **2.13** | **5.60** | −18 | **+19** |

## Why (from the loss curves)

- **B's `value_loss` stays high (~50) throughout.** Its critic is chasing
  a moving target because the joint pooled state keeps changing —
  the critic keeps providing useful gradient signal.
- **A's `value_loss` falls to ~15 by late training.** Its critic converges
  to a bland local V, then stops providing informative advantages.
- **B's `entropy` doesn't fully collapse** — stays around 0.03–0.09.
  A's collapses to 0.02 and never recovers.

## Caveats

- **1-seed comparison.** Trend is clear but not publication-strength on
  its own. A 2- or 3-seed replication would upgrade this to a properly
  reportable result.
- **`libsumo` segfault workaround required.** libsumo 1.27.1 crashes
  reliably on the third `traci.load()` — see `src/dispatch_marl/_sumo.py`
  docstring. Both arms use plain TraCI to sidestep this.
- **CTDE approximation.** Because PPO minibatches shuffle experiences
  across RL steps, the "joint pool" during update mixes states from
  different steps. This is a standard MAPPO-with-parameter-sharing
  shortcut; not strict CTDE.

## Consequence

`PolicyConfig.centralised_critic` default flipped from `False` to `True`.
The `--centralised-critic` CLI flag on `train.py` now uses
`argparse.BooleanOptionalAction`, so:

- `python scripts/train.py …` → CTDE ON (new default)
- `python scripts/train.py … --no-centralised-critic` → per-agent V (old
  default)

Older checkpoints saved with `centralised_critic=False` continue to work —
`eval_policy.py` reads the flag from the checkpoint's saved `PolicyConfig`.
