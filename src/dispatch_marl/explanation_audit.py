"""Decision logic for the freshness-aware explanation audit framework.

The experiment pipeline produces evidence.  This module turns that evidence
into an explicit decision about whether an attention map may be presented as
an explanation.  It deliberately does not change or retrain the policy.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
INDETERMINATE = "INDETERMINATE"
NOT_APPLICABLE = "NOT APPLICABLE"
ELIGIBLE = "ELIGIBLE"
WITHHOLD = "WITHHOLD"


@dataclass(frozen=True)
class AuditRules:
    """Predeclared rules used to convert evidence into an audit decision."""

    decision_relevance_ci_floor: float = 0.0
    minimum_dispatch_decisions: int = 1
    direction_epsilon: float = 1e-9
    require_deterministic_capability: bool = True
    require_trigger_robustness: bool = True


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    name: str
    status: str
    purpose: str
    evidence: str
    decision_rule: str
    sources: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _direction(value: Any, epsilon: float) -> int:
    if not _finite(value) or abs(float(value)) <= epsilon:
        return 0
    return 1 if float(value) > 0 else -1


def _interval_direction(low: Any, high: Any) -> int | None:
    """Return a supported direction, zero for uncertainty, or None if missing."""
    if not _finite(low) or not _finite(high):
        return None
    if float(low) > 0:
        return 1
    if float(high) < 0:
        return -1
    return 0


def _check(
    check_id: str,
    name: str,
    status: str,
    purpose: str,
    evidence: str,
    decision_rule: str,
    *sources: Path,
) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        name=name,
        status=status,
        purpose=purpose,
        evidence=evidence,
        decision_rule=decision_rule,
        sources=tuple(str(path) for path in sources),
    )


def _model_rows(payload: dict[str, Any] | None, model_id: str, key: str = "rows") -> list[dict]:
    if not payload:
        return []
    return [row for row in payload.get(key, []) if row.get("model") == model_id]


def _integrity_check(run_root: Path, model_id: str, seeds: Iterable[int]) -> AuditCheck:
    paths = [run_root / "sweeps" / model_id / f"seed_{seed}" / "preflight.json"
             for seed in seeds]
    payloads = [_read_json(path) for path in paths]
    if any(payload is None for payload in payloads):
        status = INCOMPLETE
        evidence = "One or more checkpoint preflight reports are missing or unreadable."
    elif all(payload.get("ok") is True for payload in payloads):
        status = PASS
        evidence = f"All {len(paths)} checkpoint sweeps passed experiment preflight."
    else:
        status = FAIL
        failed = [path.name for path, payload in zip(paths, payloads)
                  if payload and payload.get("ok") is not True]
        evidence = f"Experiment preflight failed for: {', '.join(failed)}."
    return _check(
        "evidence_integrity", "Evidence integrity", status,
        "Confirm that the evidence is complete, reproducible, and valid for analysis.",
        evidence,
        "Every frozen checkpoint sweep must pass preflight. This is not an explanation-release decision.",
        *paths,
    )


def _freshness_check(run_root: Path, summary: dict[str, Any] | None,
                     model_id: str) -> AuditCheck:
    path = run_root / "summary.json"
    rows = _model_rows(summary, model_id)
    required = (
        "stale_exposed_records",
        "mean_stale_attention_share_when_exposed",
        "mean_stale_attention_shift_when_exposed",
    )
    exposed_rows = [row for row in rows if int(row.get("stale_exposed_records", 0)) > 0]
    if not rows:
        status = INCOMPLETE
        evidence = "No condition-level summary rows were found."
    elif not exposed_rows:
        status = FAIL
        evidence = "No decision exposed to stale vehicle telemetry was observed."
    elif not all(all(_finite(row.get(field)) for field in required) for row in exposed_rows):
        status = FAIL
        evidence = "At least one stale-exposed condition is missing a separate freshness measurement."
    else:
        status = PASS
        evidence = (
            f"Freshness is reported separately in {len(exposed_rows)} stale-exposed condition rows; "
            f"{sum(int(row['stale_exposed_records']) for row in exposed_rows):,} records were audited."
        )
    return _check(
        "freshness_visibility", "Freshness is visible", status,
        "Prevent attention strength from being mistaken for data freshness.",
        evidence,
        "AoI-derived exposure and stale-attention measurements must be present and non-empty.",
        path,
    )


def _action_control_check(run_root: Path, controls_path: Path, controls: dict[str, Any] | None,
                          model_id: str, seeds: Iterable[int]) -> AuditCheck:
    manifests = [run_root / "sweeps" / model_id / f"seed_{seed}" / "manifest.json"
                 for seed in seeds]
    configs = [(_read_json(path) or {}).get("faithfulness_config", {}) for path in manifests]
    controls_rows = [row for row in (controls or {}).get("metrics", [])
                     if row.get("model_id") == model_id]
    rankers = {row.get("ranker") for row in controls_rows}
    if any(not config for config in configs) or not controls_rows:
        status = INCOMPLETE
        evidence = "Action-aware manifests or faithfulness-control results are missing."
    elif not all(config.get("random_baseline") == "type_matched" for config in configs):
        status = FAIL
        evidence = "At least one sweep did not use a type-matched random baseline."
    elif not all(config.get("exclusion_variant") is True for config in configs):
        status = FAIL
        evidence = "At least one sweep did not protect the chosen request action."
    elif not {"Raw attention", "LOO control"} <= rankers:
        status = FAIL
        evidence = "Raw-attention and LOO control results are not both available."
    else:
        status = PASS
        evidence = "Type matching, chosen-action protection, raw attention, and LOO controls are present."
    return _check(
        "action_aware_controls", "Action-aware controls", status,
        "Avoid treating request deletion and taxi-information deletion as equivalent interventions.",
        evidence,
        "All sweeps must use type matching and chosen-action protection, with a LOO positive control.",
        *manifests, controls_path,
    )


def _decision_relevance_check(controls_path: Path, controls: dict[str, Any] | None,
                              model_id: str, rules: AuditRules) -> AuditCheck:
    rows = [row for row in (controls or {}).get("action_row_sensitivity", [])
            if row.get("model_id") == model_id
            and row.get("field") == "self_row_request_def_m"]
    expected_conditions = {"clean", "outage_60s"}
    present = {row.get("condition") for row in rows}
    if not expected_conditions <= present:
        status = INCOMPLETE
        evidence = "Dispatch-action DEF is missing for clean or 60-second outage observations."
    else:
        selected = [row for row in rows if row.get("condition") in expected_conditions]
        failing = [row for row in selected
                   if not _finite((row.get("ci95") or [None])[0])
                   or float(row["ci95"][0]) <= rules.decision_relevance_ci_floor]
        if failing:
            status = FAIL
            details = ", ".join(
                f"{row['condition']} CI95 lower={float(row['ci95'][0]):+.6g}"
                for row in failing
            )
            evidence = f"Default self-row attention did not beat its random control for dispatch decisions: {details}."
        else:
            status = PASS
            evidence = "Dispatch-action margin-DEF is above the configured floor in clean and outage conditions."
    return _check(
        "decision_relevance", "Decision relevance", status,
        "Check whether the displayed node ranking is more informative than its type-matched random control.",
        evidence,
        f"The 95% CI lower bound of dispatch-action margin-DEF must exceed {rules.decision_relevance_ci_floor:g} "
        "for clean and 60-second outage observations.",
        controls_path,
    )


def _action_composition_check(run_root: Path, action_payload: dict[str, Any] | None,
                              model_id: str, rules: AuditRules) -> AuditCheck:
    path = run_root / "action_stratified.json"
    rows = _model_rows(action_payload, model_id)
    strata = {row.get("stratum"): row for row in rows}
    if not {"no_op", "dispatch"} <= set(strata):
        status = INCOMPLETE
        evidence = "No-op and dispatch strata are not both available."
    elif int(strata["dispatch"].get("total_records", 0)) < rules.minimum_dispatch_decisions:
        status = FAIL
        evidence = f"Only {strata['dispatch'].get('total_records', 0)} dispatch decisions were audited."
    else:
        status = PASS
        evidence = (
            f"Reported {int(strata['no_op']['total_records']):,} no-op and "
            f"{int(strata['dispatch']['total_records']):,} dispatch decisions separately."
        )
    return _check(
        "action_composition", "Action composition", status,
        "Stop a no-op majority from hiding the result for actual dispatch actions.",
        evidence,
        f"Both strata must be reported and dispatch must contain at least {rules.minimum_dispatch_decisions} decisions.",
        path,
    )


def _extraction_check(controls_path: Path, controls: dict[str, Any] | None,
                      model_id: str, rules: AuditRules) -> AuditCheck:
    rows = [row for row in (controls or {}).get("aggregation_sensitivity", [])
            if row.get("model_id") == model_id]
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["training_seed"]), []).append(row)
    if not by_seed or any(not any(row.get("aggregation") == "default_mean" for row in group)
                          for group in by_seed.values()):
        status = INCOMPLETE
        evidence = "The declared mean-aggregation result or its alternatives are missing."
    else:
        reversals: list[str] = []
        for seed, group in sorted(by_seed.items()):
            default = next(row for row in group if row["aggregation"] == "default_mean")
            default_direction = _direction(default.get("mean_stale_attention_shift"), rules.direction_epsilon)
            alternatives = {
                _direction(row.get("mean_stale_attention_shift"), rules.direction_epsilon)
                for row in group if row["aggregation"] != "default_mean"
            }
            if any(direction != 0 and direction != default_direction for direction in alternatives):
                reversals.append(str(seed))
        if reversals:
            status = FAIL
            evidence = f"At least one layer/head/rollout choice reverses the default direction for seeds: {', '.join(reversals)}."
        else:
            status = PASS
            evidence = "Alternative extraction rules do not reverse the declared default direction."
    return _check(
        "extraction_stability", "Attention extraction stability", status,
        "Check that the conclusion is not created by an arbitrary layer, head, or rollout choice.",
        evidence,
        "The declared self-row mean over layers and heads must not have its direction reversed by an audited alternative.",
        controls_path,
    )


def _checkpoint_check(run_root: Path, controls_path: Path,
                      synthesis: dict[str, Any] | None, controls: dict[str, Any] | None,
                      model_id: str, seeds: Iterable[int], rules: AuditRules) -> AuditCheck:
    path = run_root / "training_seed_synthesis.json"
    expected_seeds = set(int(seed) for seed in seeds)
    rows = [row for row in (controls or {}).get("per_training_seed", [])
            if row.get("model_id") == model_id
            and row.get("field") == "self_row_request_def_m"]
    found_seeds = {int(row["training_seed"]) for row in rows}
    synthesis_rows = _model_rows(synthesis, model_id)
    synthesis_seed_rows = [row for row in (synthesis or {}).get("per_seed", [])
                           if row.get("model") == model_id]
    synthesis_seeds = {int(row["training_seed"]) for row in synthesis_seed_rows}
    if (found_seeds != expected_seeds or synthesis_seeds != expected_seeds
            or not synthesis_rows):
        status = INCOMPLETE
        evidence = "Per-checkpoint decision-relevance results or training-seed synthesis are incomplete."
    else:
        by_condition: dict[str, set[int]] = {}
        for row in rows:
            by_condition.setdefault(str(row["condition"]), set()).add(
                _direction(row.get("mean"), rules.direction_epsilon)
            )
        inconsistent = [condition for condition, directions in by_condition.items()
                        if len(directions) != 1 or 0 in directions]
        stale_directions = {
            _direction(row.get("stale_attention_shift"), rules.direction_epsilon)
            for row in synthesis_seed_rows
        }
        stale_inconsistent = len(stale_directions) != 1 or 0 in stale_directions
        if inconsistent or stale_inconsistent:
            status = FAIL
            issues = []
            if inconsistent:
                issues.append(f"dispatch relevance ({', '.join(inconsistent)})")
            if stale_inconsistent:
                issues.append("stale-attention response")
            evidence = f"The conclusion is not consistent across checkpoints for: {'; '.join(issues)}."
        else:
            status = PASS
            evidence = (
                f"Dispatch relevance and stale-attention response have consistent directions "
                f"across all {len(expected_seeds)} training seeds."
            )
    return _check(
        "checkpoint_consistency", "Checkpoint consistency", status,
        "Treat independently trained policies, rather than repeated decisions, as the replication unit.",
        evidence,
        "Every configured training seed must be present and give the same non-zero direction for dispatch "
        "decision relevance and stale-attention response.",
        path, controls_path,
    )


def _deterministic_check(run_root: Path, payload: dict[str, Any] | None,
                         model_id: str, rules: AuditRules) -> AuditCheck:
    path = run_root / "deterministic_diagnostics.json"
    if not rules.require_deterministic_capability:
        return _check(
            "deterministic_capability", "Deployment-action capability", NOT_APPLICABLE,
            "Confirm that the explanation belongs to the action rule intended for deployment.",
            "The audited action rule uses stochastic sampling; argmax results are retained as a descriptive diagnostic.",
            "Require argmax capability only when argmax is the intended deployment action rule.", path,
        )
    rows = _model_rows(payload, model_id)
    if not rows:
        status = INCOMPLETE
        evidence = "No held-out deterministic diagnostic was found."
    elif any(row.get("all_episodes_zero_pickups") is True for row in rows):
        status = FAIL
        failed = [str(row.get("training_seed")) for row in rows
                  if row.get("all_episodes_zero_pickups") is True]
        evidence = f"Argmax evaluation produced zero pickups in every episode for seeds: {', '.join(failed)}."
    else:
        status = PASS
        evidence = "Every checkpoint completed at least one pickup under held-out argmax evaluation."
    return _check(
        "deterministic_capability", "Deployment-action capability", status,
        "Avoid presenting an explanation for a sampled action rule when deployment uses an unusable argmax policy.",
        evidence,
        "Every checkpoint must complete at least one pickup in the held-out deterministic diagnostic.",
        path,
    )


def _trigger_seed_status(rows: list[dict], seed: int) -> str | None:
    pair = {row.get("condition"): row for row in rows
            if int(row.get("training_seed", -1)) == seed}
    if not {"tunnel", "random"} <= set(pair):
        return None
    directions = []
    for stem in ("stale_attention", "paired_probability_def"):
        directions.append(tuple(
            _interval_direction(
                pair[condition].get(f"{stem}_ci_low"),
                pair[condition].get(f"{stem}_ci_high"),
            )
            for condition in ("tunnel", "random")
        ))
    if any(None in values for values in directions):
        return None
    if any(left != 0 and right != 0 and left != right
           for left, right in directions):
        return FAIL
    if any(left == 0 or right == 0 for left, right in directions):
        return INDETERMINATE
    return PASS


def _trigger_check(evidence_root: Path, payload: dict[str, Any] | None,
                   model_id: str, seeds: Iterable[int], rules: AuditRules) -> AuditCheck:
    path = evidence_root / "random_loss_robustness.json"
    if not rules.require_trigger_robustness:
        return _check(
            "trigger_robustness", "Trigger robustness", PASS,
            "Check whether the conclusion is specific to one tunnel placement.",
            "This gate is disabled by the audit configuration.",
            "No random-trigger comparison is required.", path,
        )
    rows = [row for row in (payload or {}).get("rows", [])
            if row.get("model") == model_id]
    expected = set(int(seed) for seed in seeds)
    found = {int(row["training_seed"]) for row in rows}
    expected_cells = {(seed, condition) for seed in expected
                      for condition in ("tunnel", "random")}
    found_cells = {(int(row["training_seed"]), row.get("condition")) for row in rows}
    if found != expected or not expected_cells <= found_cells:
        status = INCOMPLETE
        evidence = "The tunnel/random trigger comparison is incomplete."
    else:
        seed_statuses = {seed: _trigger_seed_status(rows, seed) for seed in sorted(expected)}
        if any(value is None for value in seed_statuses.values()):
            status = INCOMPLETE
            evidence = "A tunnel/random confidence interval is missing or invalid."
        else:
            contradictions = [str(seed) for seed, value in seed_statuses.items()
                              if value == FAIL]
            uncertain = [str(seed) for seed, value in seed_statuses.items()
                         if value == INDETERMINATE]
            if contradictions:
                status = FAIL
                evidence = (
                    "Tunnel and random triggers support opposite directions for seeds: "
                    f"{', '.join(contradictions)}."
                )
            elif uncertain:
                status = INDETERMINATE
                evidence = (
                    "No supported direction reverses between triggers, but at least one interval "
                    f"crosses zero for seeds: {', '.join(uncertain)}."
                )
            else:
                status = PASS
                evidence = "Tunnel and random triggers support the same direction for attention and DEF in every checkpoint."
    return _check(
        "trigger_robustness", "Trigger robustness", status,
        "Test whether the result extends beyond the fixed tunnel-trigger mechanism.",
        evidence,
        "A direction is supported only when its 95% confidence interval excludes zero; supported tunnel and random directions must agree.",
        path,
    )


def _checkpoint_evidence(
    *,
    run_root: Path,
    controls: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    deterministic: dict[str, Any] | None,
    action_payload: dict[str, Any] | None,
    model_id: str,
    seeds: Iterable[int],
    rules: AuditRules,
) -> list[dict[str, Any]]:
    """Expose the evidence used for each candidate checkpoint."""
    output = []
    per_seed_controls = (controls or {}).get("per_training_seed", [])
    aggregation = (controls or {}).get("aggregation_sensitivity", [])
    trigger_rows = [row for row in (trigger or {}).get("rows", [])
                    if row.get("model") == model_id]
    deterministic_rows = (deterministic or {}).get("rows", [])
    action_rows = (action_payload or {}).get("per_seed", [])
    for seed in seeds:
        seed = int(seed)
        preflight = _read_json(
            run_root / "sweeps" / model_id / f"seed_{seed}" / "preflight.json"
        )
        relevance = [row for row in per_seed_controls
                     if row.get("model_id") == model_id
                     and int(row.get("training_seed", -1)) == seed
                     and row.get("field") == "self_row_request_def_m"
                     and row.get("condition") in {"clean", "outage_60s"}]
        relevance_ok = None if len(relevance) != 2 else all(
            _finite(row.get("mean"))
            and float(row["mean"]) > rules.decision_relevance_ci_floor
            for row in relevance
        )

        agg_rows = [row for row in aggregation
                    if row.get("model_id") == model_id
                    and int(row.get("training_seed", -1)) == seed]
        default = next((row for row in agg_rows
                        if row.get("aggregation") == "default_mean"), None)
        if default is None:
            extraction_ok = None
        else:
            default_direction = _direction(
                default.get("mean_stale_attention_shift"), rules.direction_epsilon
            )
            alternative_directions = {
                _direction(row.get("mean_stale_attention_shift"), rules.direction_epsilon)
                for row in agg_rows if row.get("aggregation") != "default_mean"
            }
            extraction_ok = default_direction != 0 and not any(
                direction != 0 and direction != default_direction
                for direction in alternative_directions
            )

        diagnostic = next((row for row in deterministic_rows
                           if row.get("model") == model_id
                           and int(row.get("training_seed", -1)) == seed), None)
        deterministic_ok = (
            True if not rules.require_deterministic_capability
            else None if diagnostic is None
            else diagnostic.get("all_episodes_zero_pickups") is False
        )
        trigger_status = _trigger_seed_status(trigger_rows, seed)
        trigger_ok = (
            True if not rules.require_trigger_robustness
            else None if trigger_status is None
            else trigger_status == PASS
        )
        dispatch = next((row for row in action_rows
                         if row.get("model") == model_id
                         and int(row.get("training_seed", -1)) == seed
                         and row.get("stratum") == "dispatch"), None)
        dispatch_ok = (
            None if dispatch is None
            else int(dispatch.get("n_records", 0)) >= rules.minimum_dispatch_decisions
        )
        gates = {
            "preflight": None if preflight is None else preflight.get("ok") is True,
            "dispatch_decisions_present": dispatch_ok,
            "dispatch_decision_relevance": relevance_ok,
            "extraction_direction_stable": extraction_ok,
            "deterministic_capability": deterministic_ok,
            "trigger_robustness": trigger_ok,
        }
        if any(value is None for value in gates.values()):
            decision = INCOMPLETE
        elif all(gates.values()):
            decision = ELIGIBLE
        else:
            decision = WITHHOLD
        output.append({
            "training_seed": seed,
            "decision": decision,
            "gates": gates,
            "failed_gates": [name for name, value in gates.items() if value is False],
            "missing_gates": [name for name, value in gates.items() if value is None],
        })
    return output


def audit_model(
    *,
    run_root: Path,
    evidence_root: Path,
    controls_path: Path,
    model_id: str,
    model_name: str,
    seeds: Iterable[int],
    rules: AuditRules,
) -> dict[str, Any]:
    """Audit one model family and return a machine-readable decision."""
    seeds = tuple(int(seed) for seed in seeds)
    summary = _read_json(run_root / "summary.json")
    synthesis = _read_json(run_root / "training_seed_synthesis.json")
    action_payload = _read_json(run_root / "action_stratified.json")
    deterministic = _read_json(run_root / "deterministic_diagnostics.json")
    controls = _read_json(controls_path)
    trigger = _read_json(evidence_root / "random_loss_robustness.json")

    checks = [
        _integrity_check(run_root, model_id, seeds),
        _freshness_check(run_root, summary, model_id),
        _action_control_check(run_root, controls_path, controls, model_id, seeds),
        _decision_relevance_check(controls_path, controls, model_id, rules),
        _action_composition_check(run_root, action_payload, model_id, rules),
        _extraction_check(controls_path, controls, model_id, rules),
        _checkpoint_check(run_root, controls_path, synthesis, controls, model_id, seeds, rules),
        _deterministic_check(run_root, deterministic, model_id, rules),
        _trigger_check(evidence_root, trigger, model_id, seeds, rules),
    ]
    if any(check.status == INCOMPLETE for check in checks):
        decision = INCOMPLETE
        permitted_use = "Do not present attention as an explanation; collect the missing evidence."
    elif any(check.status in {FAIL, INDETERMINATE} for check in checks):
        decision = WITHHOLD
        permitted_use = "Internal model diagnostic only."
    else:
        decision = ELIGIBLE
        permitted_use = (
            "Attention may be presented as an audited candidate explanation with an explicit freshness indicator and scope. "
            "Eligibility is evidence of audit completion, not proof of a complete causal explanation."
        )
    return {
        "model_id": model_id,
        "model": model_name,
        "decision": decision,
        "permitted_use": permitted_use,
        "scope": {
            "training_seeds": list(seeds),
            "policy_state": "frozen checkpoints",
            "evaluation_mode": "offline held-out audit",
        },
        "checks": [asdict(check) for check in checks],
        "checkpoints": _checkpoint_evidence(
            run_root=run_root,
            controls=controls,
            trigger=trigger,
            deterministic=deterministic,
            action_payload=action_payload,
            model_id=model_id,
            seeds=seeds,
            rules=rules,
        ),
        "failed_checks": [check.check_id for check in checks if check.status == FAIL],
        "indeterminate_checks": [check.check_id for check in checks if check.status == INDETERMINATE],
        "incomplete_checks": [check.check_id for check in checks if check.status == INCOMPLETE],
    }


def audit_framework(
    *,
    run_root: Path,
    evidence_root: Path,
    controls_path: Path,
    models: Iterable[tuple[str, str]],
    seeds: Iterable[int],
    rules: AuditRules,
) -> dict[str, Any]:
    """Run the framework over all requested model families."""
    reports = [audit_model(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        model_id=model_id,
        model_name=model_name,
        seeds=seeds,
        rules=rules,
    ) for model_id, model_name in models]
    decisions = {report["decision"] for report in reports}
    if INCOMPLETE in decisions:
        overall = INCOMPLETE
    elif WITHHOLD in decisions:
        overall = WITHHOLD
    else:
        overall = ELIGIBLE
    return {
        "schema_version": 2,
        "framework": "freshness-aware explanation audit",
        "purpose": (
            "Decide whether graph-attention weights have enough freshness and "
            "decision-relevance evidence to be presented as explanations."
        ),
        "decision_meaning": {
            ELIGIBLE: "All required checks passed within the stated scope.",
            WITHHOLD: "Evidence failed at least one required check; keep attention internal.",
            INCOMPLETE: "Required evidence is missing or unreadable; no release decision is possible.",
        },
        "check_status_meaning": {
            PASS: "The required evidence supports the check.",
            FAIL: "The required evidence contradicts the release rule.",
            INDETERMINATE: "The evidence is complete but does not support a direction.",
            NOT_APPLICABLE: "The check is outside the declared deployment scope.",
            INCOMPLETE: "Evidence required for the check is missing or unreadable.",
        },
        "rules": asdict(rules),
        "inputs": {
            "run_root": str(run_root),
            "evidence_root": str(evidence_root),
            "controls_summary": str(controls_path),
        },
        "overall_decision": overall,
        "models": reports,
    }
