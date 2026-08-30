# Model Design and Training Rationale

This document explains why the experiment uses multiple dispatcher conditions,
what each condition is designed to test, and how the learned models are
trained. It is intended as supporting material for the dissertation
methodology chapter.

![Model conditions and training process](model_design_training_process.png)

**Figure. Model conditions and shared MAPPO training process.** B0 is the
non-learning dispatch reference; B1/B2 are trained on clean telemetry; H5
is trained with tunnel-freeze degradation. After training, checkpoints are
selected on validation demand and evaluated on held-out test demand under
clean telemetry and fixed observation-layer outage durations.

## 1. Why multiple models are needed

The dissertation is not trying to prove that one dispatcher is the best
possible fleet-management algorithm. Its main question is about explanation
faithfulness:

> Do attention-based explanations in a GAT-MARL fleet dispatcher remain
> faithful when telemetry becomes stale?

To answer that question, the experiment needs one main attention-based model
and several comparison conditions. Each condition removes or changes one
factor so the result can be interpreted more clearly.

## 2. Model conditions

| Condition | Learned? | Training telemetry | Main purpose |
|---|---:|---|---|
| B0 SUMO greedy | No | Not trained | Non-learning dispatch reference |
| B1 MLP-MAPPO | Yes | Clean | RL baseline without graph attention |
| B2 GAT-MAPPO | Yes | Clean | Main audited attention-based dispatcher |
| H5′ degradation-aware GAT-MAPPO | Yes | Degraded | Training-side mitigation test |

Strictly, B0 is a baseline condition rather than a trained model. The learned
models are B1, B2, and H5′.

## 3. Purpose of each condition

### B0: SUMO greedy dispatch baseline

**Purpose:** provide a non-learning operational reference.

B0 uses SUMO's built-in greedy taxi matcher. It does not use reinforcement
learning and does not provide attention explanations. It is included to answer
a basic reviewer question:

> How competent are the learned policies compared with a simple dispatch
> heuristic?

In the final results, B0 completes about 32 pickups per episode, while the
learned policies complete around 6-8. This does not invalidate the
faithfulness study, but it scopes the claims: the explanation results apply to
the low-capability learned policies actually achieved.

### B1: MLP-MAPPO baseline

**Purpose:** test whether the graph/GAT architecture improves dispatch
performance over a plain neural policy.

B1 uses the same MAPPO training loop as B2 but replaces the graph-attention
encoder with an MLP policy. It has no attention channel, so it is not used for
DEF/WAMSN faithfulness scoring. Its role is mainly performance comparison:

> Is the graph architecture useful for the control task, or could a simpler
> MLP do the same?

### B2: GAT-MAPPO main model

**Purpose:** serve as the main system under audit.

B2 is the central model in the dissertation. It uses a graph-attention encoder
over the acting taxi, neighbouring taxis, and candidate reservations. Its
attention weights are the coupled explanation channel tested by the
faithfulness pipeline.

B2 answers the core research questions:

- Are attention explanations faithful on clean telemetry?
- What happens to their faithfulness under stale telemetry?
- Does attention shift toward stale vehicle nodes as AoI increases?
- Does faithfulness degrade separately from dispatch performance?

B2 is trained on clean telemetry first. Then it is evaluated under both clean
and degraded telemetry. This design simulates the realistic case where a
standard dispatcher trained under normal conditions is deployed into a world
where telemetry sometimes becomes stale.

### H5′: degradation-aware GAT-MAPPO

**Purpose:** test whether training with degradation mitigates the problem.

H5′ uses the same basic GAT-MAPPO configuration as B2, but training occurs
with tunnel-triggered freeze degradation enabled. In the final experiment, the
training degradation uses an outage level such as 30 s.

This condition answers:

> If the model experiences stale telemetry during training, does its attention
> channel become more faithful or more robust?

The final result is negative: degradation-aware training does not make the
coupled attention channel faithful. Under the corrected type-matched DEF
baseline, H5′ remains near zero, like B2.

## 4. Training process for learned models

All learned models use the same high-level MAPPO training pipeline.

```text
Initialise SUMO environment
Initialise shared policy
For each epoch:
    reset one SUMO episode
    collect one full rollout with the current policy
    store each acting taxi's observation, action, log-probability, value, reward
    compute GAE advantages per taxi trajectory
    update the shared policy using PPO mini-batches
    log pickups, reward, entropy, losses, KL, clip fraction
    save periodic checkpoints
    update ckpt_best.pt if rolling pickup mean improves
```

The entry point is:

```bash
python scripts/train.py
```

The main training script creates the environment, constructs the selected
policy, collects rollouts, computes GAE, performs PPO updates, and writes
checkpoints under:

```text
runs/mappo/<area>_<timestamp>/
```

Each run saves:

```text
args.json
train_log.jsonl
ckpt_epoch_XXXX.pt
ckpt_best.pt
best_metadata.json
```

## 5. Shared MAPPO details

The fleet uses parameter sharing: all taxis use the same policy network. At
each RL step, all currently idle taxis act as agents. The policy samples an
action for each acting taxi:

```text
0      = no-op
1..K   = accept the k-th candidate reservation
```

The action is applied to SUMO through `dispatchTaxi`, then SUMO advances by
10 simulated seconds.

The team reward is:

```text
R = 10 * pickups
    + 0.5 * successful_dispatches
    - 0.001 * mean_pending_wait_time
```

This team reward is broadcast to all acting taxis at that step.

After one full episode, the trainer computes Generalised Advantage Estimation
(GAE):

```text
delta_t = r_t + gamma * V_{t+1} - V_t
GAE_t   = delta_t + gamma * lambda * GAE_{t+1}
return_t = GAE_t + V_t
```

Default values:

```text
gamma = 0.99
lambda = 0.95
```

Then PPO updates the policy using the clipped surrogate objective:

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
```

The full loss is:

```text
loss = policy_loss
       + vf_coef * value_loss
       - ent_coef * entropy
```

Default PPO settings:

| Parameter | Default |
|---|---:|
| learning rate | 3e-4 |
| clip ratio | 0.2 |
| PPO epochs | 4 |
| minibatch size | 256 |
| value-loss coefficient | 0.5 |
| entropy coefficient | 0.01 |
| max gradient norm | 0.5 |

## 6. Example training commands

### B1: MLP-MAPPO baseline

```bash
python scripts/train.py \
  --area central_park \
  --policy mlp \
  --epochs 300 \
  --seed 42 \
  --demand-split train
```

### B2: clean-trained GAT-MAPPO

```bash
python scripts/train.py \
  --area central_park \
  --policy gat \
  --epochs 300 \
  --seed 42 \
  --demand-split train \
  --degradation off
```

### H5′: degradation-aware GAT-MAPPO

```bash
python scripts/train.py \
  --area central_park \
  --policy gat \
  --epochs 300 \
  --seed 42 \
  --demand-split train \
  --degradation tunnel_triggered \
  --outage-duration 30
```

## 7. Evaluation after training

After training, each learned checkpoint is evaluated separately. The main
evaluation pattern is:

```text
Train on clean or degraded telemetry
Freeze checkpoint
Evaluate on held-out test demand
Evaluate under clean telemetry
Evaluate under outage durations {10, 20, 30, 60} seconds
Compute pickups, reward, wait time
For GAT models, compute DEF, margin-DEF, WAMSN, and attention drift
```

The main sweep command is:

```bash
python scripts/run_dissertation_experiments.py \
  --stage sweep --model B2_gat --resume
```

This produces one clean condition plus four tunnel-degradation levels. B1 is
not scored with DEF/WAMSN because it has no graph-attention explanation
channel.

## 8. Summary of experimental logic

The model set is designed as a chain of controls:

```text
B0: Is the learned policy meaningful compared with a simple heuristic?
B1: Does a non-graph RL policy behave similarly?
B2: Are GAT attention explanations faithful?
H5′: Can degradation-aware training fix the explanation problem?
```

The final dissertation conclusion relies most heavily on B2, with B0/B1/H5′
serving as controls and mitigation checks. The negative result is important:
the coupled GAT attention channel remains practically unfaithful even after
the experiment controls for architecture, capability range, and
degradation-aware training.
