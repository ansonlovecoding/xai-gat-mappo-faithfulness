# Freshness-Aware Explanation Audit Report

**Overall decision:** WITHHOLD

## Purpose

Decide whether graph-attention weights have enough freshness and decision-relevance evidence to be presented as explanations.

This is an offline release audit for frozen policy checkpoints. It does not retrain the model, and an eligible result is not proof of a complete causal explanation.

## How to read the decision

- **ELIGIBLE:** All required checks passed within the stated scope.
- **WITHHOLD:** Evidence failed at least one required check; keep attention internal.
- **INCOMPLETE:** Required evidence is missing or unreadable; no release decision is possible.

Individual checks use PASS, FAIL, INDETERMINATE, NOT APPLICABLE, or INCOMPLETE. An indeterminate required check leads to WITHHOLD; it does not mean that evidence is missing.

## GAT

**Decision:** WITHHOLD

**Permitted use:** Internal model diagnostic only.

| Check | Status | Evidence |
|---|---|---|
| Evidence integrity | PASS | All 5 checkpoint sweeps passed experiment preflight. |
| Freshness is visible | PASS | Freshness is reported separately in 20 stale-exposed condition rows; 15,028 records were audited. |
| Action-aware controls | PASS | Type matching, chosen-action protection, raw attention, and LOO controls are present. |
| Decision relevance | FAIL | Default self-row attention did not beat its random control for dispatch decisions: clean CI95 lower=-0.00138828, outage_60s CI95 lower=-0.00115661. |
| Action composition | PASS | Reported 5,143 no-op and 14,953 dispatch decisions separately. |
| Attention extraction stability | FAIL | At least one layer/head/rollout choice reverses the default direction for seeds: 42, 43, 44, 45, 46. |
| Checkpoint consistency | FAIL | The conclusion is not consistent across checkpoints for: dispatch relevance (clean, outage_60s); stale-attention response. |
| Deployment-action capability | NOT APPLICABLE | The audited action rule uses stochastic sampling; argmax results are retained as a descriptive diagnostic. |
| Trigger robustness | INDETERMINATE | No supported direction reverses between triggers, but at least one interval crosses zero for seeds: 42, 45. |

### Checkpoint decisions

| Training seed | Decision | Failed gates | Missing gates |
|---:|---|---|---|
| 42 | WITHHOLD | dispatch decision relevance, extraction direction stable, trigger robustness | None |
| 43 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |
| 44 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |
| 45 | WITHHOLD | extraction direction stable, trigger robustness | None |
| 46 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |

### Failed or missing evidence
- **Decision relevance:** The 95% CI lower bound of dispatch-action margin-DEF must exceed 0 for clean and 60-second outage observations.
- **Attention extraction stability:** The declared self-row mean over layers and heads must not have its direction reversed by an audited alternative.
- **Checkpoint consistency:** Every configured training seed must be present and give the same non-zero direction for dispatch decision relevance and stale-attention response.
- **Trigger robustness:** A direction is supported only when its 95% confidence interval excludes zero; supported tunnel and random directions must agree.

## GAT-Outage

**Decision:** WITHHOLD

**Permitted use:** Internal model diagnostic only.

| Check | Status | Evidence |
|---|---|---|
| Evidence integrity | PASS | All 5 checkpoint sweeps passed experiment preflight. |
| Freshness is visible | PASS | Freshness is reported separately in 20 stale-exposed condition rows; 15,029 records were audited. |
| Action-aware controls | PASS | Type matching, chosen-action protection, raw attention, and LOO controls are present. |
| Decision relevance | FAIL | Default self-row attention did not beat its random control for dispatch decisions: clean CI95 lower=-0.00343498, outage_60s CI95 lower=-0.002997. |
| Action composition | PASS | Reported 4,342 no-op and 15,714 dispatch decisions separately. |
| Attention extraction stability | FAIL | At least one layer/head/rollout choice reverses the default direction for seeds: 42, 43, 44, 45, 46. |
| Checkpoint consistency | FAIL | The conclusion is not consistent across checkpoints for: stale-attention response. |
| Deployment-action capability | NOT APPLICABLE | The audited action rule uses stochastic sampling; argmax results are retained as a descriptive diagnostic. |
| Trigger robustness | INDETERMINATE | No supported direction reverses between triggers, but at least one interval crosses zero for seeds: 43, 46. |

### Checkpoint decisions

| Training seed | Decision | Failed gates | Missing gates |
|---:|---|---|---|
| 42 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |
| 43 | WITHHOLD | dispatch decision relevance, extraction direction stable, trigger robustness | None |
| 44 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |
| 45 | WITHHOLD | dispatch decision relevance, extraction direction stable | None |
| 46 | WITHHOLD | dispatch decision relevance, extraction direction stable, trigger robustness | None |

### Failed or missing evidence
- **Decision relevance:** The 95% CI lower bound of dispatch-action margin-DEF must exceed 0 for clean and 60-second outage observations.
- **Attention extraction stability:** The declared self-row mean over layers and heads must not have its direction reversed by an audited alternative.
- **Checkpoint consistency:** Every configured training seed must be present and give the same non-zero direction for dispatch decision relevance and stale-attention response.
- **Trigger robustness:** A direction is supported only when its 95% confidence interval excludes zero; supported tunnel and random directions must agree.

## Operational use

1. Freeze the candidate checkpoint and evaluation protocol.
2. Generate clean/degraded pairs, action-aware faithfulness records, action strata, and sensitivity controls.
3. Run experiment preflight. A preflight pass only confirms evidence integrity.
4. Run this audit and inspect every model-level check.
5. Present attention externally only when the decision is ELIGIBLE, together with telemetry freshness and scope.
6. Repeat the audit after retraining, checkpoint replacement, telemetry changes, or deployment-scenario changes.
