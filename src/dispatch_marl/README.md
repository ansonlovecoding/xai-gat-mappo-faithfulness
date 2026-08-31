# `dispatch_marl` — design notes

Multi-agent RL environment, policy, and training loop for fleet dispatch
under telemetry degradation. Written for the DMU dissertation on
**explanation faithfulness and faithfulness decoupling in GAT-MAPPO**. This
document explains *why* each layer looks the way it does — the *what* lives
in each module's docstring.

## Package layout

```
dispatch_marl/
├── _sumo.py         # resolves SUMO_HOME once, on import; shared by every script
├── scenario.py      # loads a built Yubei scenario (sumocfg + tunnels.json)
├── degradation.py   # telemetry-degradation layer (dissertation's core novelty)
├── env.py           # PettingZoo ParallelEnv wrapping SUMO via TraCI
├── policies.py      # non-learning baselines (Random, Nearest, NoOp)
├── models/
│   ├── gat.py       # hand-rolled multi-head graph attention layer
│   └── policy.py    # GAT-MAPPO policy (encoder + actor + critic)
└── training.py      # rollout collection + GAE + PPO update
```

**Guiding principle:** every layer has one job and can be replaced without
touching the others. In particular, `degradation.py` only mutates observation
tensors — it never touches SUMO state. That's what makes the
"tunnel-triggered vs. matched-rate random noise" ablation (one of the
dissertation's key comparisons) a single config flip rather than a fork of
the environment code.

## 1. Environment layer (`env.py`)

### Why PettingZoo `ParallelEnv`

- A fleet is a **synchronous multi-agent** system: every taxi acts within
  the same SUMO tick. `ParallelEnv`'s semantics (`step(actions_dict)` →
  `obs_dict, reward_dict, ...`) match this directly.
- MAPPO uses parameter sharing across homogeneous agents plus a centralised
  critic — this idiomatically maps onto `ParallelEnv` and *not* onto the
  turn-taking AEC API.

### Dynamic `agents` set

`env.agents` is recomputed every step from `_active_idle_taxi_ids()` — only
taxis that are (a) alive in the simulator and (b) currently idle appear in
the set. Rationale:

- Taxis mid-ride (state = PICKUP / OCCUPIED) don't need a decision — SUMO
  is already routing them.
- Taxis that fell into a cul-de-sac or otherwise exited the simulation
  just drop out.
- `ParallelEnv` explicitly supports a variable `agents` set between steps.

**Consequence for the rollout loop:** if `env.agents == []` at some step,
the RL loop still has to call `env.step({})` to advance the simulator —
another taxi may become idle a few seconds later. The training script's
outer loop terminates on `env.done`, not on empty `agents`.

### SUMO connection management

Three defensive design choices, each earned by a live failure during
development:

- **Unique connection labels per reset** (`env_<id>_<counter>`), combined
  with `traci.switch()`. Avoids the `"Connection 'default' is already
  active"` collision that surfaces when a prior fatal error left the label
  half-registered.
- **`traci.load()` is tried before `traci.start()`** between episodes.
  Reusing the same SUMO subprocess is ~2× faster than spawning a fresh one
  and avoids a whole class of orphan-process issues.
- **`_step_impl` is wrapped in `try/except FatalTraCIError`** at the
  `step()` level. If SUMO dies mid-episode, we force `env.done = True` and
  return empty dicts, letting the training loop cleanly move to the next
  epoch instead of unwinding an exception.

## 2. Observation graph structure

Each taxi sees a **self-centric graph**:

```
nodes = 1 (self) + K_n (nearest other taxis) + K_r (nearest pending reservations)
        default 1 + 5 + 5 = 11 nodes
```

### Why self-centric rather than a global graph

1. **Fixed-shape batching.** A global graph's node count varies with fleet
   size and open-reservation count; a self-centric slice is always
   `1 + K_n + K_r`, which drops straight into a shared-parameter policy.
2. **Inductive bias matches the decision.** Dispatch is fundamentally "who
   should *I* accept" — computing attention around the acting taxi is the
   right shape for that question.
3. **Attention is the explanation.** Rows of the GAT's softmax over
   neighbours/reservations map one-to-one onto candidate actions. This is
   the *coupled explanation channel* the dissertation studies — the model
   pays no extra cost to produce it.

### Feature widths (constants in `env.py`)

| Node type | Dim | Fields |
|---|---:|---|
| self | 5 | `[x_norm, y_norm, episode_time_norm, velocity_norm, aoi_norm]` |
| neighbour taxi | 5 | `[dx, dy, is_empty, dist_norm, aoi_norm]` |
| reservation | 5 | `[dx_pickup, dy_pickup, dx_dropoff, dy_dropoff, wait_norm]` |

`aoi_norm` on both vehicle-type nodes is Age-of-Information divided by
`AOI_MAX_S` (clipped to [0, 1]) — required so per-node WAMSN in the
faithfulness pipeline (§7.4) can multiply attention by a *per-node*
staleness. Neighbour AoI is refreshed every sim step by the
`DegradationLayer`, whether or not the neighbour happens to be the acting
agent (otherwise a taxi that's mid-ride would accumulate AoI forever).

Neighbour/reservation coordinates are stored **relative to self**, and all
distances are divided by the network diagonal. Two consequences:

- The policy learns geometric patterns rather than absolute positions →
  same weights transfer across the three Yubei sub-areas.
- Padded slots contribute zero after masking, so `K_n=K_r=5` is a soft
  cap — actual valid counts may be lower.

### Masking

Each node type has a `_mask` array (1 = valid, 0 = padded). The GAT layer
sets masked-key logits to `-inf` before softmax, so padded positions
contribute nothing to attention. The critic uses masked mean-pool over the
same mask.

## 3. Action space

Per-taxi: `Discrete(K_r + 1)`.

- `action = 0` → **no-op** (taxi keeps randomCircling until next decision)
- `action = k` for `k ∈ {1, ..., K_r}` → **accept the k-th nearest
  reservation** in the observation's sorted list

### Why not continuous coordinates or zone repositioning

- **Continuous (x, y) target:** huge action space, no natural mapping onto
  attention weights → useless for the faithfulness experiments.
- **Zone repositioning only:** delegates dispatch to a heuristic layer; the
  policy just moves idle taxis. This shifts the interesting decision away
  from what the GAT's attention is over, breaking the coupled-explanation
  premise.
- **"Which reservation to accept"** is the canonical framing in fleet
  dispatch MARL literature and the most defensible interpretation of the
  GAT's attention rows.

### Sorted-reservation cache

Because the observation lists reservations sorted by distance to *this*
specific taxi, action indices are meaningless without that context. `env.py`
caches the ordering in `self._pending_res_map[agent] = [res_id_1, ...]` at
observation-build time and looks the concrete reservation ID back up at
step-apply time. Prevents any drift between "the k-th slot the policy saw"
and "the k-th slot the environment will dispatch".

## 4. Reward design (and the lesson learned)

Team reward, broadcast identically to every acting agent:

```
R = pickup_reward       * pickups_this_step
  + dispatch_reward     * successful_dispatches_this_step
  - wait_penalty_lambda * mean_pending_wait_time_s
```

Defaults: `pickup_reward=10.0`, `dispatch_reward=0.5`, `wait_penalty_lambda=0.001`.

### The v0 → v1 reward-shaping lesson

The initial formula was `+1 × pickups − 0.01 × wait_time`, which — after
300 epochs — converged to **"do nothing"**: 0 pickups, 0 dispatches. The
policy discovered that dispatching is risky (many attempts fail at SUMO's
router because of network-connectivity gaps), whereas the wait-time penalty
is paid regardless. The variance-minimising strategy is inaction.

The fix scaled pickups 10× and shrank the wait penalty 10× — pickups are
now roughly 10 000× larger than a per-step wait unit. Adding a smaller
positive term for *successful* dispatches (`+0.5` per confirmed
`dispatchTaxi()`) gives the policy immediate credit for *trying*, not just
for a fully-completed delivery. After the reshape, the same architecture
climbed to 7.7 pickups per episode.

**Methodological point worth citing in the write-up:** with sparse reward
and dense penalty, MARL policies frequently degenerate to a "no-action"
attractor. Reward design must make the magnitudes of "success" and "no
action" clearly distinct — one order of magnitude is not enough here.

## 5. Telemetry-degradation layer — the dissertation's core novelty

`DegradationLayer` (in `degradation.py`) modifies **observations only**,
never simulator state. Three modes:

| `mode` | Trigger | Purpose |
|---|---|---|
| `off` | never | clean baseline |
| `tunnel_triggered` | taxi's current edge ∈ `tunnel_edges` | primary experimental condition, physically grounded |
| `random_dropout` | independent Bernoulli(`dropout_rate`) per step | matched-rate ablation with no spatial structure |

### Design decisions worth writing up

- **Applied at the observation boundary.** Ground-truth SUMO positions are
  never corrupted. That way any performance drop is attributable *exactly*
  to what the policy sees — not to knock-on effects in the simulator. This
  is what makes "faithfulness under degradation" a meaningful measurement
  rather than a confounded one.
- **`position_valid` flag is exposed on the self-node embedding.** Any
  degradation-aware policy variant can condition on whether its own reading
  is trustworthy; a degradation-blind variant simply ignores the extra
  dimension. Same environment supports both — the sensitive-vs-normal
  ablation is a training-time choice, not an environment fork.
- **`tunnel_edges` uses the navigable subset only.** Orphan tunnels (missing
  incoming or outgoing connections after netconvert filtering) are excluded
  because no agent will ever reach them via realistic traffic flow. Orphans
  are still preserved in `orphan_tunnel_edges` inside `tunnels.json` for
  transparency in the methods section.

## 6. GAT policy (`models/policy.py` and `models/gat.py`)

```
obs dict
  → per-type Linear projections into shared hidden_dim (3 node types)
  → + learned type embedding + degradation embedding (added onto self node)
  → stacked into (B, N, D) node bank plus (B, N) mask
  → GATLayer × 2  (4 heads each, LayerNorm, residual, feed-forward)
  → split back out: self embedding + reservation embeddings
  → actor head:  noop_head(self) ‖ res_head(reservations) → K_r+1 logits
  → critic head: masked-mean pool over node embeddings → scalar V
```

### Why hand-rolled attention, not `torch-geometric`

- **Install friction.** PyG requires exact matching of torch, CUDA, and
  scatter/sparse extension versions. Colab installs have broken silently in
  the past.
- **Explanation transparency.** PyG's `GATConv` hides attention weights in
  internal buffers that would need patching to expose. Our
  `GATLayer.forward()` returns the softmax tensor directly, next to the
  updated node features.
- **Small graphs.** With `N = 11` and full connectivity, scaled
  dot-product attention is ~40 lines of PyTorch. No performance win from
  the specialised graph libraries at this scale.

### The MAPPO-compatible API

`policy.get_action_and_value(obs, action=None)` covers both directions:

- `action=None` → sample from the categorical over logits (rollout use).
- `action=<tensor>` → compute `log_prob` and `entropy` at that specific
  action (PPO update use).

Same forward pass, plus the attention weights come back in the aux dict for
free — that's the tensor the faithfulness pipeline will consume.

### Coupled explanation, by construction

`policy.forward()` returns `attention` with shape
`(n_layers, B, n_heads, N, N)`. The same tensor drives the action logits
(via the transformed node embeddings) AND will drive the faithfulness
metric (via ablation studies, ERASER-style comprehensiveness/sufficiency,
etc.). This is precisely the "coupled" regime the dissertation compares
against a to-be-added decoupled head.

## 7. Training loop (`training.py`)

Three functions, one file:

1. **`collect_rollout(env, policy)`** — runs one full episode. For every
   acting agent at every RL step, appends an `AgentStep(agent, obs, action,
   log_prob, value, reward, transition_id, ...)` to a flat buffer. Team task
   reward is shared; a successful dispatcher receives additional difference
   credit used only for training.
2. **`compute_gae(buffer, gamma, gae_lambda)`** — groups by agent ID, runs
   backward-pass Generalised Advantage Estimation per trajectory. Episode
   end treated as terminal (V_{T+1} = 0).
3. **`ppo_update(policy, optimizer, buffer, config)`** — clipped surrogate
   loss + clipped value loss + decision-state entropy bonus. Minibatches keep
   complete environment transitions together, target KL limits large updates,
   and the centralised critic pools only agents from the same transition.

### Parameter sharing across the fleet

All 20 taxis use the same policy weights. Homogeneous fleet + shared
parameters means every agent-step (~2000 per episode) contributes to a
single gradient update — critical for sample efficiency, since one SUMO
episode is inherently limited by real-time constraints on the simulator.

### Centralised critic grouping

The MAPPO critic pools all active agents from one environment transition and
broadcasts that group value back to those agents. Transition ids preserve this
grouping during PPO updates, so shuffled experiences from different simulation
times cannot be mistaken for one joint state. The actor remains decentralised.

### Simplification: no explicit conflict handling in the policy

Two taxis can pick the same reservation in the same step. The environment
resolves this "first in `actions.dict()` wins", so a naive policy can waste
turns via conflicts. A proper cure is a learned assignment mechanism (e.g.
Sinkhorn-style bipartite matching); left to future work.

## 8. Baselines (`policies.py`)

- **`RandomPolicy`** — uniform samples from `Discrete(K+1)`. Lower bound
  reference.
- **`NearestReservationPolicy`** — every idle taxi accepts its nearest
  visible reservation. The catch: 20 taxis all rush the same closest
  reservation → mass conflicts → wasted dispatch actions. Empirically
  *worse than random* on central_park (1 vs 3 pickups). That result is a
  useful methodological finding — it shows why learned coordination is
  needed rather than a hand-crafted heuristic.
- **SUMO's built-in greedy dispatcher** — run via `scripts/run_baselines.py`,
  which bypasses our env entirely and lets SUMO's C++ bipartite matcher do
  the assignment. Strong upper bound (30 pickups on central_park). The gap
  between our trained policy and this baseline is what the rest of the
  dissertation aims to close on both performance and — more importantly —
  faithful explanation.

## 9. Best-checkpoint tracking

Added because PPO's classic entropy-collapse failure mode makes the *final*
checkpoint of a run often the *worst* checkpoint. The v1 run peaked at
epoch 50 and had collapsed to 0 pickups by epoch 299.

**Mechanism** (in `scripts/train.py`):

- After every epoch, append this episode's pickup count to a rolling
  window of size `--best-window` (default 10).
- Once the window is full, if its mean improves on the best mean seen so
  far, snapshot the current policy to `ckpt_best.pt` and update
  `best_metadata.json`.
- Print `★ new best` on the corresponding console-table row.

The rolling checkpoint is a training diagnostic, not the reportable model
selection rule. Dissertation runs save periodic checkpoints, evaluate them on
fixed validation demand with `scripts/select_checkpoint.py`, and copy the best
candidate to `ckpt_selected.pt`. Held-out test demand is opened only after that
selection is fixed. The v5 protocol also rejects runs whose final validation
mean retains less than 80% of the selected checkpoint's mean.

For v5, only 10% of the critic gradient enters the shared actor encoder. An
adaptive entropy coefficient also activates below a decision-entropy floor.
Both controls affect optimisation only; policy outputs and evaluation-time
action selection are unchanged.

## Design justification: decentralised vs. classical fleet dispatch

An informed reader may notice that the observation graph is **heterogeneous
and self-centric** (self + neighbour taxis + candidate reservations) rather
than the vehicle-only graph one usually sees in fleet-dispatch papers, and
that each vehicle makes its own accept/no-op decision rather than a
central dispatcher computing a global matching. This is deliberate, and
it matters for the dissertation's core claim about explanation
faithfulness. This section states the choice explicitly.

### Two families of fleet-dispatch formulations

Classical fleet-dispatch literature splits into two lineages:

- **Centralised dispatch** — a single dispatcher computes a global matching
  between all idle vehicles and all pending orders (bipartite matching, ILP,
  or a learned central network that reads the whole fleet's state).
  Vehicle-only graph, one attention distribution per dispatch step.
- **Decentralised dispatch** — each vehicle independently decides which
  order (if any) to accept, given a local view of nearby peers and
  candidate orders. Lin et al. 2018, Li et al. 2019, and Al-Abbasi et al.
  2019 all follow this framing. Per-agent graph, one attention
  distribution per acting vehicle per step.

This package implements the **decentralised** variant.

### Why decentralised is the right choice *for this dissertation*

1. **Attention rows map one-to-one onto individual decisions.**
   A per-agent heterogeneous graph produces one attention distribution per
   acting vehicle per RL step — so DEF and WAMSN are measured over
   ~1500–2400 independent decisions per episode instead of ~120 aggregate
   dispatcher outputs. That is an order-of-magnitude gain in statistical
   power for the faithfulness analysis, on the same simulation budget.

2. **AoI is defined per node; the WAMSN summand is per node.**
   The WAMSN definition sums `α_i · (AoI_i / AoI_max)` over graph nodes.
   The shape is a direct match for a per-agent graph in which `α_i` is the
   acting vehicle's attention weight over node `i`, and `AoI_i` is that
   node's Age of Information. In a centralised, vehicle-only graph it
   would be unclear whose AoI enters the numerator when multiple vehicles
   are simultaneously stale.

3. **The dissertation's causal chain is per-agent.**
   The proposal's Figure 1 causal graph — telemetry degradation → AoI ↑
   → attention drift → WAMSN ↑ → DEF ↓ — is inherently local: one
   vehicle enters a tunnel, its own AoI rises, its own attention drifts,
   its own decision becomes less faithful. In a centralised model,
   multiple agents' AoI changes propagate simultaneously through one
   dispatcher's attention, muddling attribution.

### The known weakness, turned into a research question

Decentralised dispatch has a well-known failure mode: with no explicit
coordination channel, multiple vehicles can converge on the same order.
Our `NearestReservationPolicy` baseline demonstrates this explicitly —
it *loses* to random on central_park (1 pickup vs 3 pickups) because
20 taxis all rush the same nearest reservation.

Rather than a bug, this is precisely the phenomenon the dissertation is
positioned to study: **does the GAT's attention over neighbouring vehicles
(the peer-node type in the heterogeneous graph) faithfully encode the
implicit coordination the learned policy uses?** The coupled-explanation
channel already surfaces which peers are being attended to; the
decoupled channel (still to be added) will provide a comparison point.
If either mechanism is faithful, attention on peer-vehicles should
correlate with actual coordination behaviour — which is directly a DEF
question.

### Consequences for how results are reported

Because attention is per-decision, **DEF and WAMSN distributions should
be reported over agent-step samples, not per-episode averages**.
Bootstrapping confidence intervals over per-decision samples gives
substantially tighter bounds than bootstrapping over episodes. Worth
noting now so it doesn't drift when the results section is drafted.

## Architectural overall — why this shape serves the dissertation

The package is designed around **two experimental axes the dissertation
must be able to run cleanly**:

1. **Coupled vs. decoupled explanation.** `policy.forward()` already
   returns the coupled attention weights alongside the action. A decoupled
   head can be added as a sibling to the actor without changing the env,
   the training loop, or the observation format. The comparison is
   architectural, not evaluated-on-different-models.
2. **Structured vs. random telemetry degradation.** `DegradationLayer`
   sits between the simulator (ground truth) and the policy (potentially
   degraded input). Flip a config field and you swap physically-grounded
   tunnel-triggered noise for a matched-rate Bernoulli baseline — same env
   class, same reward, same policy. This is the ablation that makes
   "explanation faithfulness under real-world-like degradation" a
   measurable claim rather than a plausible-sounding one.

Everything else — the reward tuning, the taxi conflict resolution, the
per-type node embeddings — is scaffolding for these two axes.
