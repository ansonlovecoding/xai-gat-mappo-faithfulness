# Model design and training

This document explains why the experiment uses MLP, GAT, and GAT-Outage, and
how their checkpoints are trained and selected. The fixed settings are in
`configs/experiments/dissertation_v10_corrected.toml`.

![Model conditions and training process](model_design_training_process.png)

## Model roles

| Model | Encoder | Training telemetry | Purpose |
|---|---|---|---|
| MLP | multilayer perceptron | clean | task-capability context without an attention explanation |
| GAT | graph attention | clean | main policy whose attention weights are audited |
| GAT-Outage | graph attention | 30-second tunnel-triggered outages | tests whether the fitted outage-aware configuration is more consistent |

The dissertation uses these names rather than experimental codes. Internal
identifiers appear only in commands and result paths.

MLP is not used to answer the attention-faithfulness questions because it does
not expose graph-attention weights. GAT and GAT-Outage have the same policy
architecture. Their main difference is the training observation: GAT receives
current telemetry, whereas GAT-Outage sometimes receives the last valid
position and speed after tunnel entry.

GAT-Outage is not assumed to be better. It tests whether experience with one
30-second outage condition produces a more consistent explanation response at
10, 20, 30, and 60 seconds. The reward has no explanation-faithfulness term, so
outage training is not expected to guarantee faithful attention.

## Policy input and output

Each idle taxi receives a local graph with:

- one self-taxi node;
- up to five peer-taxi nodes;
- up to five passenger-request nodes.

The actor returns a probability distribution over no-op and one dispatch action
for each visible request. The critic returns a shared team-value estimate during
training. GAT policies also return attention tensors through a read-only audit
channel. The audit does not change the action or update the model.

## Training process

All learned models use multi-agent proximal policy optimization with
centralized training and decentralized execution. Each taxi acts from its own
local observation. During training, the centralized critic pools valid node
embeddings from the agents in the same transition.

The shared reward at simulation step `t` is:

```text
r_t = 10 * completed_passenger_journeys_t
    + 0.5 * accepted_dispatches_t
    - 0.001 * mean_pending_wait_t
```

A taxi that makes an accepted dispatch receives an additional difference
credit of 1.0. When a taxi remains busy between decisions, all intervening team
rewards are accumulated with the correct semi-Markov discount. Generalized
advantage estimation also uses the actual decision interval. This prevents
rewards earned during a passenger journey from being lost.

## Training budgets

| Setting | GAT | GAT-Outage | MLP |
|---|---:|---:|---:|
| Maximum epochs | 40 | 50 | 50 |
| Learning rate | 0.0001 | 0.0001 | 0.0003 |
| Learning-rate decay | linear | linear | none |
| Training seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 |

The maximum budgets were fixed from validation-only stability work before the
held-out test. Because GAT and GAT-Outage do not have identical optimization
horizons, their comparison describes the two fitted configurations and cannot
attribute any difference only to outage training.

## Checkpoint selection

Training, validation, and test demand files are separate. Epoch 0 is kept only
as an initialization diagnostic and cannot be selected. Periodic trained
checkpoints are evaluated over eight validation episodes with seed 2026. The
checkpoint with the highest mean completed passenger journeys is selected;
mean reward and earlier epoch break ties.

| Model | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| MLP | epoch 45 | epoch 30 | epoch 38 |
| GAT | epoch 30 | epoch 20 | epoch 39 |
| GAT-Outage | epoch 10 | epoch 40 | epoch 9 |

MLP and GAT are selected under clean validation telemetry. GAT-Outage is
selected under its 30-second training condition. The test split is never read
during checkpoint selection.

## Stability checks

A run is accepted only when it satisfies all declared checks:

| Check | Requirement |
|---|---:|
| Final-window mean completed journeys | at least 5.0 |
| Longest zero-completion run | no more than 4 epochs |
| Selected validation completed journeys | at least 5.0 |
| Final validation retention | at least 0.80 |

All nine final training runs pass. These checks establish that the checkpoints
can be evaluated; they do not validate attention as an explanation.

## Held-out evaluation

Each selected model checkpoint receives 48 clean test episodes: eight action
sampling seeds and six episodes per seed. GAT and GAT-Outage additionally
receive the five-condition faithfulness sweep. Actions are sampled from the
policy distribution in the primary audit. A three-episode argmax diagnostic is
reported separately.

The main findings are not a performance claim. Completed journeys establish
that the policies act in the environment. Explanation release depends on the
freshness-aware checks documented in
`docs/FRESHNESS_AWARE_EXPLANATION_AUDIT.md`.

## Commands

```bash
.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage train

.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage select

.venv/bin/python scripts/run_dissertation_experiments.py \
  --config configs/experiments/dissertation_v10_corrected.toml \
  --stage evaluate
```

See `docs/REPRODUCE_EXPERIMENTS.md` for the full audit and packaging workflow.
