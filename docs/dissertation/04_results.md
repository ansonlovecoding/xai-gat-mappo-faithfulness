# 4. Results

All primary numbers in this chapter come from the completed v4 experiment,
whose citable outputs are in `results/dissertation_v4/`. Every B2 and H5 sweep
passed the protocol preflight. Results are shown first for each training seed and then summarised
across independently trained policies. A per-decision p-value is not used to
claim that an effect generalises across training seeds.

## 4.1 Checkpoint selection and performance context

Validation selected different epochs for different training runs. B2 and B3
selected epochs 10, 70, and 10; H5 selected 10, 90, and 10. This variation
confirms that a fixed final epoch would not represent the same learning stage
across seeds.

![Validation checkpoint selection by model and training seed](../figures/v4_checkpoint_selection_by_model_and_seed.png)

**Figure 4.1.** Validation pickups vary strongly across checkpoints and
training seeds. Stars mark the checkpoints selected without reading the held-out
test split. The curves are selection evidence, not estimates of final test
performance.

Table 4.1 reports clean-test pickups. Performance is included to describe the
policies being explained, not as the research outcome.

| Model | Mean across training seeds | Range across training seeds |
|---|---:|---:|
| B1 MLP | 13.25 | 11.50-14.67 |
| B2 GAT | 13.29 | 11.46-14.67 |
| B3 GAT without AoI | 13.29 | 11.46-14.67 |
| H5 degradation-trained GAT | 9.56 | 4.54-12.88 |

B2 and B3 are identical under clean evaluation because their AoI inputs are
zero. H5 is less stable: seed 44 averages 4.54 pickups, compared with 12.88 and
11.25 for seeds 42 and 43. This weak replicate is retained because excluding
it would hide training instability.

![Clean-test performance by training seed](../figures/v4_clean_performance_by_training_seed.png)

**Figure 4.2.** Each point is one independently trained policy evaluated over
24 held-out episodes; horizontal lines show the mean across training seeds.

## 4.2 Supporting construct-validity result

The earlier audit showed that a uniform random occlusion is not a fair control
for this graph. Removing a passenger-request node can remove the chosen action,
while removing a taxi node only withholds information. The sign and size of DEF
can therefore be driven by node-type composition.

| Random control | Mean margin-DEF | 95% CI | Interpretation |
|---|---:|---:|---|
| Uniform random | -0.5540 | [-0.5808, -0.5288] | dominated by node-type and action-deletion imbalance |
| Type-matched random | -0.0093 | [-0.0159, -0.0013] | approximately no advantage over matched random |

The audit contains 1,199 paired clean decisions from the earlier B2 protocol.
It diagnoses the measurement problem; it is not part of the v4 hypothesis
test.

The v4 rerun addresses this problem from the start. All headline DEF values use
type-matched random subsets, and the action-protected margin variant is stored
alongside probability DEF. Under these controls, clean DEF is close to zero for
all B2 and H5 training seeds. The result should be read as "no measured
advantage over the matched random explanation", not "attention is worse than
random".

## 4.3 Manipulation check: stale exposure increases

The observation-layer outages produce increasing empirical degradation. At 60
seconds, the mean degraded-decision rate is about 0.9-1.2% for B2 and
0.7-1.3% for H5, compared with zero in the clean condition. Exposure is sparse
because it depends on taxis entering the mapped tunnels.

When at least one stale vehicle node is visible, mean WAMSN rises from about
0.023-0.031 at 10 seconds to 0.071-0.090 at 60 seconds. H3 is supported in all
six trained GAT policies. The within-policy trend correlations range from
rho = 0.082 to 0.105 for B2 and 0.078 to 0.121 for H5; every Holm-adjusted
p-value is 0.0004.

![WAMSN, type-matched DEF and pickups by outage duration](../figures/v4_decoupling_by_outage_duration.png)

**Figure 4.3.** Conditional WAMSN rises with observation-layer outage duration
for every training seed, while type-matched DEF and pickups remain nearly
flat. The three rows make the decoupling visible, but they do not imply that
AoI caused a change in faithfulness: WAMSN itself includes normalised AoI.

## 4.4 Paired attention reallocation is seed-dependent

The paired stale-attention shift compares the degraded observation with its
clean twin. The result is not consistent across trained policies.

| Model | Training seed | Mean shift | 95% CI |
|---|---:|---:|---:|
| B2 GAT | 42 | +0.002706 | [0.002395, 0.003036] |
| B2 GAT | 43 | +0.000618 | [0.000424, 0.000780] |
| B2 GAT | 44 | -0.001278 | [-0.001436, -0.001112] |
| H5 degraded training | 42 | +0.002374 | [0.002151, 0.002609] |
| H5 degraded training | 43 | +0.000783 | [0.000710, 0.000862] |
| H5 degraded training | 44 | -0.001231 | [-0.001410, -0.001072] |

Both training regimes therefore have two positive seeds out of three.

Within each seed, the confidence intervals are narrow because the same
decisions are paired. However, precision within one trained policy does not
resolve disagreement between independently trained policies. The primary
claim cannot therefore be "degradation always shifts attention toward stale
nodes". A defensible statement is that the direction and size of attention
reallocation depend on the learned policy.

![Paired stale-attention shift](../figures/v4_paired_stale_attention_shift.png)

**Figure 4.4.** Positive values mean more attention mass on stale nodes in the
degraded twin; seed 44 reverses the direction for both training regimes.

## 4.5 Faithfulness and decoupling hypotheses

H1 is not supported for any B2 or H5 seed. The robust DEF-duration
correlations remain close to zero, with all Holm-adjusted p-values above 0.67.
The experiment therefore provides no evidence that a longer outage directly
reduces DEF.

H2 is also unsupported in all six trained policies. Faithfulness does not
decline faster than pickups under the v4 protocol. Rate calculations are
especially unstable when clean DEF is close to zero, so the negative result is
more reliable than a narrative built from large percentage changes around a
near-zero denominator.

H4 is model- and seed-dependent. None of the three B2 policies support the
predicted negative within-episode WAMSN-DEF relationship; their mean
within-episode correlations are positive (0.036-0.069). H5 seeds 42 and 43 do
support a negative relationship (rho = -0.072 and -0.061, adjusted
p = 0.0004), but H5 seed 44 does not (rho = -0.001, adjusted p = 0.185).
This mixed result is exploratory evidence of a possible training-regime
interaction, not a general finding.

![Within-episode WAMSN-DEF correlation by training seed](../figures/v4_h4_correlation_by_training_seed.png)

**Figure 4.5.** Within-episode Spearman correlations between WAMSN and DEF.
Filled points passed the within-policy Holm-corrected H4 test; open points did
not. The direction differs between B2 and H5 and is not consistent across all
training seeds.

## 4.6 Degradation-aware training

H5 does not provide a consistent mitigation. Its stale-attention shifts follow
the same two-positive, one-negative pattern as B2. H1 and H2 remain unsupported.
H4 becomes negative for two seeds, but this does not reproduce in seed 44. The
large clean-performance drop for H5 seed 44 also shows that degradation-aware
training can be unstable under the present settings.

The correct conclusion is not that H5 improves or worsens faithfulness. The
experiment shows that training under one 30-second degradation condition does
not make explanation behaviour consistent across training seeds.

## 4.7 Hypothesis summary

| Hypothesis | B2 seeds supporting | H5 seeds supporting | Final verdict |
|---|---:|---:|---|
| H1: outage duration increases, DEF decreases | 0/3 | 0/3 | not supported |
| H2: faithfulness declines faster than performance | 0/3 | 0/3 | not supported |
| H3: outage duration increases, WAMSN increases | 3/3 | 3/3 | consistently supported |
| H4: higher WAMSN is associated with lower DEF | 0/3 | 2/3 | mixed, training-dependent |
| H5: degradation-aware training mitigates the effect | n/a | 0/3 consistent mitigation | not supported |

The completed experiment therefore supports a concentrated result: longer
observation outages reliably increase stale-data attention exposure, but the
paired attention shift and its relationship with faithfulness do not
generalise across independently trained policies.
