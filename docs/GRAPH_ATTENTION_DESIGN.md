# Graph Attention Design Rationale

This document explains why the dissertation uses a graph-attention layer, why
the observation graph is designed as it is, and what role the attention weights
play in the experiment.

![Simplified graph attention design](simplified_graph_attention_design.png)

**Figure. Simplified graph-attention policy design.** Each acting taxi receives
a self-centred graph, node features are projected into shared embeddings,
valid nodes exchange information through multi-head attention, and the final
embeddings produce both dispatch actions and an attention map for
faithfulness testing.

## 1. Why a graph model is needed

Fleet dispatch is naturally relational. A taxi's best action depends not only
on its own state, but also on:

- nearby taxis that may compete for the same reservation;
- pending reservations around the taxi;
- the relative distance between taxis and requests;
- telemetry freshness, because stale vehicle positions can distort the
  dispatcher's view of the fleet.

A plain MLP can receive all these numbers as one flattened vector, but it does
not explicitly model the structure:

```text
acting taxi ↔ neighbouring taxis ↔ candidate reservations
```

The graph model makes that structure explicit. Each decision is represented as
a small graph centred on the acting taxi. The policy can then learn which
nearby entities matter for that taxi's dispatch decision.

## 2. Why graph attention specifically

Graph attention is used for two reasons.

First, attention is a flexible way to aggregate information from nearby
entities. Instead of hand-coding which taxi or reservation should matter most,
the model learns attention weights over valid nodes.

Second, those attention weights are the object of the dissertation's
faithfulness study. Many graph-attention systems treat attention as a built-in
explanation. This project tests whether that assumption is valid:

> Do the attention weights actually identify decision-relevant information,
> especially when telemetry becomes stale?

So attention is not included only for performance. It is included because the
project needs a realistic explanation channel to audit.

## 3. The per-agent observation graph

For each idle taxi, the environment builds one self-centred graph.

```text
node 0              = acting taxi itself
nodes 1..K_n        = K_n nearest neighbouring taxis
nodes K_n+1..N      = K_r nearest pending reservations
```

In the final experiments:

```text
K_n = 5 neighbouring taxis
K_r = 5 candidate reservations
N   = 1 + 5 + 5 = 11 graph slots
```

If fewer neighbours or reservations exist, the unused slots are padded and
masked out.

## 4. Why the graph is self-centred

The graph is built from the viewpoint of one acting taxi. This design has
three benefits.

### 4.1 One graph equals one decision

Each graph corresponds to one taxi's dispatch decision. This makes the
faithfulness analysis precise:

```text
one decision → one attention distribution → one DEF/WAMSN score
```

If the system used only one large global graph for the whole fleet, it would
be harder to say which attention weights explain which taxi's action.

### 4.2 Local candidate set keeps training tractable

The taxi only chooses among the nearest pending reservations. This keeps the
action space small:

```text
action 0      = no-op
actions 1..5  = accept one of the five candidate reservations
```

This is important because the project runs inside a small SUMO benchmark and
Colab-scale compute budget.

### 4.3 Degradation is local and measurable

Telemetry staleness belongs to individual vehicle nodes. A self-centred graph
makes it clear where AoI enters the model:

```text
self taxi AoI
neighbour taxi AoI
reservation nodes have no AoI
```

This matches the WAMSN metric, which measures how much attention falls on
stale vehicle nodes.

## 5. Node types and features

The graph has three node types.

| Node type | Features | Purpose |
|---|---|---|
| Self taxi | position, episode time, velocity, AoI | Represents the acting vehicle |
| Neighbour taxi | relative position, availability, distance, AoI | Represents nearby fleet context |
| Reservation | pickup/drop-off relative vectors, waiting time | Represents candidate dispatch actions |

These features are deliberately relative where possible. For example,
neighbour taxi positions and reservation pickup/drop-off locations are encoded
relative to the acting taxi. This helps the policy learn geometric dispatch
patterns instead of memorising absolute map coordinates.

When telemetry is degraded, the graph is built from the **observed** state,
not SUMO ground truth. Therefore stale positions can affect:

- the acting taxi's self features;
- neighbour taxi features;
- neighbour ordering;
- reservation vectors relative to the stale self position.

This is exactly the intended experimental setup: the policy acts on the world
as the telemetry system presents it.

## 6. Why the graph is fully connected

The implementation uses a fully connected graph over valid nodes. Every valid
node can attend to every other valid node.

This design is chosen because the per-agent graph is already small:

```text
maximum 11 nodes per acting taxi
```

With such a small graph, a fully connected attention layer is simple,
efficient, and avoids hand-designing edge rules. The model receives relative
distance features, so it can learn whether near or far nodes deserve attention.

Padded nodes are excluded with masks, so they do not contribute to the
attention softmax or the actor's reservation logits.

## 7. Heterogeneous embedding design

The three node types have different meanings, so the model first projects them
with separate linear layers:

```text
self features        → self embedding
neighbour features   → taxi embedding
reservation features → reservation embedding
```

Then each node receives a learned type embedding:

```text
self type
taxi type
reservation type
```

This tells the GAT whether a node is the acting taxi, another taxi, or a
candidate request. After this projection, all nodes live in the same hidden
dimension and can interact through attention.

The model also embeds the `position_valid` flag and adds it to the self node.
This lets the policy distinguish trusted self telemetry from degraded self
telemetry.

## 8. Multi-head graph attention layer

Each GAT layer uses scaled dot-product multi-head attention:

```text
score(i, j) = Q_i · K_j / sqrt(d_head)
attention(i, j) = softmax_j(score(i, j))
updated node i = weighted sum of V_j over valid nodes
```

The implementation uses:

```text
hidden_dim = 64
n_heads = 4
n_gat_layers = 2
```

The layer also uses residual connections, LayerNorm, and a small feed-forward
block. This is similar to a compact Transformer block applied to graph nodes.

The attention tensor has shape:

```text
(layers, batch, heads, query_node, key_node)
```

This is important because the same tensor is later used by the faithfulness
pipeline.

## 9. How the actor uses graph embeddings

After the GAT layers, the actor produces dispatch logits.

The no-op logit comes from the final self-node embedding:

```text
self embedding → no-op action score
```

Each reservation action logit comes from the corresponding reservation-node
embedding:

```text
reservation node 1 → action 1
reservation node 2 → action 2
...
reservation node 5 → action 5
```

Invalid reservation slots are masked to `-inf`, so the policy cannot select
actions for padded reservation nodes.

This direct reservation-node/action mapping is useful because it makes the
policy interpretable at the decision level. It also creates the
candidate-action occlusion issue discovered in the dissertation: occluding a
reservation node can remove the corresponding action from the decision
problem.

## 10. Why attention is treated as a coupled explanation

The GAT attention weights are **coupled** to the policy because they are used
inside the model's forward pass to compute the action logits. The same
attention weights are then read by the faithfulness evaluator as the candidate
explanation.

This mirrors how attention is often used in practice:

```text
the model attends to these nodes → therefore these nodes explain the decision
```

The dissertation tests whether that reasoning is justified. The final
answer is cautious: the coupled attention channel is close to the corrected
random faithfulness baseline. Longer outages increase stale-data exposure, but
the paired direction of attention reallocation changes across training seeds.

## 11. Design summary

The graph attention layer exists for two connected reasons:

1. **Control reason:** dispatch is relational, so the policy needs a way to
   combine information from the acting taxi, nearby taxis, and candidate
   reservations.
2. **Explainability reason:** attention gives a realistic built-in
   explanation channel whose faithfulness can be audited.

The design choices support the dissertation's experimental goals:

| Design choice | Purpose |
|---|---|
| Self-centred graph | One graph and attention map per taxi decision |
| Nearest taxis and reservations | Small tractable local decision problem |
| Fully connected valid nodes | Lets the model learn relations without hand-coded edges |
| Separate type embeddings | Preserves self/taxi/reservation semantics |
| AoI on vehicle nodes | Enables stale-telemetry reasoning and WAMSN |
| Reservation-node/action mapping | Makes dispatch actions interpretable but exposes occlusion artefact |
| Returned attention tensor | Enables coupled explanation faithfulness analysis |

For the thesis, the short explanation is:

> The GAT layer is used because fleet dispatch is a relational decision
> problem: each taxi must choose among requests while accounting for nearby
> vehicles and stale telemetry. The model builds a small self-centred graph per
> acting taxi, embeds self, peer-taxi, and reservation nodes into a shared
> space, and applies fully connected multi-head graph attention over valid
> nodes. The final self embedding scores the no-op action, while final
> reservation embeddings score assignment actions. The attention weights are
> returned as a coupled explanation channel, allowing the dissertation to test
> whether the model's own attention is faithful under clean and degraded
> telemetry.
