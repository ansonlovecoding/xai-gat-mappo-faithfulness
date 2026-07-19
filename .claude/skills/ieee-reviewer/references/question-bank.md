# Question bank — archetypes by dimension

Templates, not scripts: instantiate each with a concrete quote, number,
or file from THIS project's materials. One examination should draw from
at least five different categories. Items marked ★ are known pressure
points of this specific project — prefer them.

## Claims–evidence alignment

- "Your abstract/summary says X; the strongest evidence I found is Y in
  <file>. Walk me through why Y licenses X." ★ e.g. the title claim
  "explanations outlive their data" vs H1-pooled being null (floor
  effect) — is the claim carried by WAMSN alone?
- "Which single figure carries the paper? If I deleted it, what claim
  dies?"
- ★ "The earlier noise-mechanism experiments showed an H5 'reversal'
  that vanished under freeze semantics. How do you report a result you
  now believe was a mechanism artifact — and what stops a reviewer from
  asking what ELSE is mechanism-sensitive?"

## Statistical validity

- ★ "Your per-decision Spearman tests pool ~18k decisions that are
  clustered within episodes and seeds. Defend the independence
  assumption, or tell me what the cluster-robust answer would be."
- "H2 is significant at n=32 cells but was p=0.099 at n=12. Why should
  I believe the current p-value isn't the same fragility with more
  draws?"
- "You evaluate stochastically because argmax collapses. What is the
  variance story — how many episodes would change any headline
  conclusion?"

## Metric construct validity

- ★ "Margin-DEF departs from the ERASER probability semantics your
  proposal committed to. Why is a logit margin a *faithfulness*
  quantity rather than a sensitivity quantity — and what would Jacovi &
  Goldberg say?"
- ★ "DEF is negative — worse than random. Does that measure the
  explanation's unfaithfulness, or does it measure that occluding the
  chosen reservation's node destroys the action set (a mechanical
  artifact of masking = action removal)?"
- "WAMSN maxes at ~0.015 in your data. What does a 1.5 % attention mass
  shift mean operationally for a dispatcher looking at a dashboard?"

## Baseline & comparison fairness

- ★ "SUMO's greedy matcher gets 32 pickups; your learned policies get
  6–8. Why should anyone study the explanations of a policy this far
  from competent — does unfaithfulness even matter for a policy nobody
  would deploy?"
- "B1-MLP matches B2-GAT on performance. What, then, does the graph
  buy — and if the answer is 'an explanation channel that turns out
  unfaithful', what is the contribution of using a GAT at all?"
- ★ "H5′ (degradation-aware) is *less* faithful than B2 at every level
  (−0.77 vs −0.54). You frame this as 'mitigation rejected' — could it
  instead mean your training procedure, not the idea, failed? One seed,
  one outage level."

## Reproducibility & engineering

- "Your dev machine runs SUMO 1.20, Colab runs 1.27, and the proposal
  pinned 1.27. Which version produced each committed number, and would
  the story survive a version bump?"
- "Walk me from `git clone` to reproducing the Act-2 figure. Every
  command, every pinned version. Where does it break on a clean
  machine?"
- "Entropy collapse means your 'best' checkpoints are rescued by a
  rolling-mean heuristic. How sensitive are all downstream faithfulness
  numbers to that selection rule?"

## Positioning & novelty

- "Jain & Wallace showed attention unfaithfulness in 2019. State, in
  two sentences, what your Act 1 adds beyond 'their result also holds
  in a MARL dispatch domain'."
- "The AoI literature quantifies staleness costs for *control*. Name
  the closest prior work coupling staleness to *explanations* — and if
  there is none, how did you search?"

## Limitations honesty & scope

- "One map, one city, 20 taxis, 50 riders, ~2 % tunnel exposure. Which
  of your conclusions do you actually expect to transfer, and which are
  artifacts of this scale?"
- "Your severity ladder tops at 60 s but your own data shows a taxi
  idling on a tunnel edge reached AoI 310 s. Why is the ladder the
  right sweep range?"

## Code–paper consistency

- "Pick any equation in §7.4 and show me the line of code implementing
  it. I will pick a second one myself."
- "The freeze rework changed AOI_MAX_S from 300 to 60 — which committed
  results straddle that change, and how does the reader know which
  scale each number is on?"
