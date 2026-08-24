# P5 audit: what the re-analysis changed

Produced by `scripts/analyze_artifact_controlled_ladder.py` over
`runs/sweeps/B2_aoi_ladder_v2` (B2 `ckpt_best.pt`, epoch 91, 8 seeds x 3
held-out test episodes x 5 cells, 17 764 scored decisions, freeze
corruption, stochastic actions, `faithfulness_every=8`).

Output: `results/story_freeze_v1/audit/artifact_controlled_ladder.json`.

Every number below was recomputed from the committed per-decision records.
Where a stored `analysis.json` value exists it was reproduced exactly first,
so the disagreements below are disagreements with the *paper text*, not with
the analysis pipeline.

---

## Finding 1 — the severity ladder was never manipulated (most important)

`max_aoi_s` is reconstructed in `faithfulness.py::_extract_aoi_per_node` from
the observation feature `aoi_norm = clip(AoI / AOI_MAX_S, 0, 1)` with
`AOI_MAX_S = 60`. It is therefore **right-censored at 60 s**, and so is
WAMSN's staleness weight `clip(AoI / AOI_MAX_S, 0, 1)`: once a vehicle node
passes 60 s its weight is pinned at 1.0 and any further staleness is
invisible to the metric.

Realised AoI among exposed decisions:

| Nominal level | p50 | p90 | max | frac at cap | n |
|---:|---:|---:|---:|---:|---:|
| 5 s | 60.0 | 60.0 | 60.0 | 78.3 % | 345 |
| 15 s | 60.0 | 60.0 | 60.0 | 84.9 % | 416 |
| 30 s | 60.0 | 60.0 | 60.0 | 81.1 % | 349 |
| 60 s | 60.0 | 60.0 | 60.0 | 79.4 % | 388 |

Identical median *and* p90 at every level, 81 % censored overall. Tunnel
transit time, not the nominal outage duration, sets the staleness a decision
actually sees. The `{5, 15, 30, 60}` s ladder collapses to **one** degraded
condition.

**Consequence.** Flat margin-DEF and flat WAMSN across levels are the
expected behaviour of an unmanipulated factor. They are not evidence about
severity, and the paper's "floor effect" explanation for H1 is not the
operative cause. H1 as preregistered (severity lowers DEF) is **untestable
on this data**, not unsupported. The defensible contrast is binary clean vs
degraded, which is well powered and does show the WAMSN effect.

This supersedes the `SUMMARY.md` note "max observed AoI 310 s" — that figure
comes from the environment's internal AoI, which never reaches the metric.

---

## Finding 2 — H4's effect size is 3x smaller on the paper's primary metric

The paper states margin-DEF is the primary metric ("the main analysis uses
margin-DEF"), then reports H4 as `mean within-episode rho = -0.140`,
`negative in 89 % of episodes`. Those are the **probability**-DEF numbers.

Reproduced exactly, both sweeps, `stratified_h4` estimator (min 8 scored
decisions per episode, constant-WAMSN episodes dropped, tie-corrected
Spearman):

| Sweep | DEF variant | episodes | mean rho | frac negative |
|---|---|---:|---:|---:|
| B2_aoi_ladder | probability | 96 | **-0.140** | 88.5 % |
| B2_aoi_ladder | **margin** | 96 | **-0.043** | 74.0 % |
| B2_aoi_ladder_v2 | probability | 96 | -0.149 | 92.7 % |
| B2_aoi_ladder_v2 | **margin** | 96 | **-0.045** | 64.6 % |

Same sign, same direction, replicates across sweeps, still significant — but
the paper must quote **-0.043 / -0.045 with 65-74 % of episodes negative**,
or report both variants and label which is which. As written the paper
imports the effect size from the metric it declared secondary.

(A first pass at this recomputation gave `+0.07` because it used
argsort-of-argsort ranks with no tie correction. WAMSN is exactly 0 in ~90 %
of decisions, so the tied block dominates and the sign flips. The estimator
in `analyze_hypotheses.py::_rank` tie-corrects correctly; the script written
for this audit now does too.)

---

## Finding 3 — H3 is a binary step, not a severity trend

| Test | n | rho | p (episode-block permutation) |
|---|---:|---:|---:|
| WAMSN vs level, all cells | 17 764 | +0.081 | <= 1e-4 |
| WAMSN vs level, **degraded cells only** | 14 154 | +0.002 | 0.88 |

The entire H3 association is the clean-vs-degraded contrast. Within degraded
conditions there is no trend — which is exactly what Finding 1 predicts.
Table VI's row "H3: severity increases WAMSN -> Supported" overstates it.
The supported claim is: **WAMSN rises from 0 the moment staleness exists and
is flat thereafter.** The paper's prose already says this; the hypothesis
table and the H3 label do not.

---

## Finding 4 — exposure rates, corrected

Two different quantities were being conflated:

| Quantity | Value | Meaning |
|---|---:|---|
| Acting-agent degradation rate | ~2 % | the deciding taxi's own telemetry is stale |
| **Stale-node visibility** | **9.8-11.7 % per degraded level** (8.4 % pooled) | the decision's graph contains >= 1 stale vehicle node — the condition WAMSN is defined on |
| Stale node in attention top-3 \| exposed | 7.7-9.3 % | a stale node reached the explanation |
| WAMSN \| exposed | 13.4-14.8 % | confirms the paper's "about 14 %" |

The paper's "only a small fraction of decisions expose stale vehicle nodes"
should be the 9.8-11.7 % figure, not the 2 %. Both belong in the text since
they answer different questions.

---

## Finding 5 — the offline artifact control is not sufficient

Two attempts to recover an artifact-free ladder without re-running the sweep:

| Control | Result | Verdict |
|---|---:|---|
| Clamp-free subset (neither top-k nor random occlusion clamped) | def_m = **+0.0002** [-0.0027, +0.0033], n = 90 | Exact, and agrees with the type-matched clean value of -0.009. But 0.5 % of decisions and selected (clamp-free decisions have fewer competing reservations) — too small to carry a ladder. |
| Clamp-differential covariate adjustment, all 17 764 decisions | def_m_adj = -0.490 to -0.499 per level | **Fails.** Moves the score only 0.04 of the 0.53 gap. The artifact is not linear in `clamp_rand - clamp_topk`, so this cannot substitute for the protocol fix. |

Clamp rates per level: top-k 3.9-4.0 %, random 16.3-16.4 % (the pooled
`construct_audit.json` figures are 2.2 % and 14.8 %; the paper's Table IV
quotes 1.8 % and 14.6 %, which match neither — see Finding 6).

**The severity ladder must be re-run under the type-matched baseline.** The
flag now exists:

```bash
python scripts/sweep_severity.py \
  runs/mappo/B2_gat/central_park_1783964871/ckpt_best.pt \
  --seeds 42 43 44 45 46 47 48 49 --episodes 3 \
  --faithfulness-every 8 --random-baseline type_matched
```

Type-matched DEF needs fresh counterfactual policy forwards with
composition-matched random sets; it cannot be recovered from recorded
per-decision scalars. Until that run exists, the ladder should be reported
as the uniform-baseline metric with an explicit artifact warning, and the
clean-condition type-matched value (-0.009) plus the clamp-free subset
(+0.0002) carry the Act-1 claim.

---

## Finding 6 — numbers in the paper that do not match any results file

| Paper | Value | Files say |
|---|---:|---|
| Table IV clamp rate, attention top-k | 1.8 % | 2.2 % pooled (`construct_audit.json`), 3.9-4.0 % per level (v2 sweep) |
| Table IV clamp rate, random | 14.6 % | 14.8 % pooled, 16.3-16.4 % per level |
| H4 rho | -0.140 | -0.140 for probability-DEF; -0.043 for margin-DEF (Finding 2) |
| H2 p | 0.039 | 0.039 is the **Holm-adjusted** probability-DEF value; raw is 0.0195 (prob) / 0.028 (margin). Label it. |
| "p < 0.001" throughout | — | permutation resolution is 1/(10 000 + 1); report as `p <= 1e-4` |
| Table V level 0 margin-DEF | -0.541 | -0.541 in `B2_aoi_ladder`, -0.533 in `B2_aoi_ladder_v2`. Pick one canonical sweep and say which. |
| Exclusion-variant "+0.59 collapses to -0.02" | — | correct, but n = 136 of 17 444 decisions; the n must be stated |

`B2_aoi_ladder` (v1) is the source of the paper's H1-H4 statistics;
`B2_aoi_ladder_v2` is the P2-instrumented sweep and the source of every
clamp/exposure/AoI figure. The paper currently mixes them without saying so.

---

## What survives all of this, unchanged

- The occlusion artifact is real, large, and replicated: 98 % of the clean
  "worse than random" deficit is composition, and the uniform metric swings
  from -1.35 to +1.78 across checkpoints that are all ~0 under the fair
  baseline.
- Attention is uninformative on clean telemetry under the fair baseline
  (-0.009), corroborated independently by the clamp-free subset (+0.0002).
- Attention content moves onto stale telemetry when staleness exists
  (WAMSN 0 -> ~0.014 pooled, ~14 % conditional on exposure) while pickups
  and aggregate DEF stay flat.
- Within episodes, more stale-node attention goes with lower faithfulness —
  at -0.043 on margin-DEF, or -0.140 on probability-DEF.
- No tested mitigation restores meaningful faithfulness under degradation.

The reframing these findings support: the ladder is not a severity study, it
is a clean-vs-degraded study plus a methodological result about occlusion
baselines. Reported that way, every claim is backed by data that exists.
