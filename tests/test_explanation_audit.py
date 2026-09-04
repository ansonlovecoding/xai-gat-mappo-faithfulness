import json
from pathlib import Path

from dispatch_marl.explanation_audit import (
    AuditRules,
    ELIGIBLE,
    INDETERMINATE,
    INCOMPLETE,
    NOT_APPLICABLE,
    WITHHOLD,
    audit_model,
)


MODEL = "gat"
SEEDS = (42, 43)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bundle(tmp_path: Path, *, relevance_lower: float = 0.1,
            reverse_extraction: bool = False) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "runs"
    evidence_root = tmp_path / "evidence"
    controls_path = evidence_root / "faithfulness_controls" / "summary.json"
    for seed in SEEDS:
        sweep = run_root / "sweeps" / MODEL / f"seed_{seed}"
        _write(sweep / "preflight.json", {"ok": True})
        _write(sweep / "manifest.json", {"faithfulness_config": {
            "random_baseline": "type_matched", "exclusion_variant": True,
        }})
    _write(run_root / "summary.json", {"rows": [
        {
            "model": MODEL,
            "stale_exposed_records": 0,
            "mean_stale_attention_share_when_exposed": None,
            "mean_stale_attention_shift_when_exposed": None,
        },
        {
            "model": MODEL,
            "stale_exposed_records": 10,
            "mean_stale_attention_share_when_exposed": 0.2,
            "mean_stale_attention_shift_when_exposed": 0.01,
        },
    ]})
    _write(run_root / "training_seed_synthesis.json", {
        "rows": [{"model": MODEL}],
        "per_seed": [
            {"model": MODEL, "training_seed": seed, "stale_attention_shift": 0.01}
            for seed in SEEDS
        ],
    })
    _write(run_root / "action_stratified.json", {
        "rows": [
            {"model": MODEL, "stratum": "no_op", "total_records": 20},
            {"model": MODEL, "stratum": "dispatch", "total_records": 40},
        ],
        "per_seed": [
            {"model": MODEL, "training_seed": seed, "stratum": "dispatch", "n_records": 20}
            for seed in SEEDS
        ],
    })
    _write(run_root / "deterministic_diagnostics.json", {"rows": [
        {"model": MODEL, "training_seed": seed, "all_episodes_zero_pickups": False}
        for seed in SEEDS
    ]})
    aggregation = []
    for seed in SEEDS:
        aggregation += [
            {"model_id": MODEL, "training_seed": seed, "aggregation": "default_mean",
             "mean_stale_attention_shift": 0.02},
            {"model_id": MODEL, "training_seed": seed, "aggregation": "first_head_0",
             "mean_stale_attention_shift": -0.01 if reverse_extraction else 0.01},
        ]
    _write(controls_path, {
        "metrics": [
            {"model_id": MODEL, "ranker": "Raw attention"},
            {"model_id": MODEL, "ranker": "LOO control"},
        ],
        "action_row_sensitivity": [
            {"model_id": MODEL, "condition": condition,
             "field": "self_row_request_def_m", "ci95": [relevance_lower, 0.2]}
            for condition in ("clean", "outage_60s")
        ],
        "aggregation_sensitivity": aggregation,
        "per_training_seed": [
            {"model_id": MODEL, "training_seed": seed, "condition": condition,
             "field": field, "mean": 0.1}
            for seed in SEEDS for condition in ("clean", "outage_60s")
            for field in ("attention_control_def_m", "self_row_request_def_m")
        ],
    })
    trigger_rows = []
    for seed in SEEDS:
        for condition in ("tunnel", "random"):
            trigger_rows.append({
                "model": MODEL,
                "training_seed": seed,
                "condition": condition,
                "stale_attention_ci_low": 0.01,
                "stale_attention_ci_high": 0.03,
                "paired_probability_def_ci_low": -0.03,
                "paired_probability_def_ci_high": -0.01,
            })
    _write(evidence_root / "random_loss_robustness.json", {"rows": trigger_rows})
    return run_root, evidence_root, controls_path


def _audit(tmp_path: Path, **bundle_options) -> dict:
    run_root, evidence_root, controls_path = _bundle(tmp_path, **bundle_options)
    return audit_model(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        model_id=MODEL,
        model_name="GAT",
        seeds=SEEDS,
        rules=AuditRules(),
    )


def test_complete_consistent_evidence_is_eligible(tmp_path: Path) -> None:
    report = _audit(tmp_path)
    assert report["decision"] == ELIGIBLE
    assert all(check["status"] == "PASS" for check in report["checks"])
    assert all(check["decision"] == ELIGIBLE for check in report["checkpoints"])


def test_failed_relevance_and_extraction_withhold_explanation(tmp_path: Path) -> None:
    report = _audit(tmp_path, relevance_lower=-0.1, reverse_extraction=True)
    assert report["decision"] == WITHHOLD
    assert {"decision_relevance", "extraction_stability"} <= set(report["failed_checks"])
    assert report["permitted_use"] == "Internal model diagnostic only."


def test_missing_required_evidence_is_incomplete(tmp_path: Path) -> None:
    run_root, evidence_root, controls_path = _bundle(tmp_path)
    controls_path.unlink()
    report = audit_model(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        model_id=MODEL,
        model_name="GAT",
        seeds=SEEDS,
        rules=AuditRules(),
    )
    assert report["decision"] == INCOMPLETE
    assert "action_aware_controls" in report["incomplete_checks"]


def test_uncertain_trigger_direction_withholds_without_calling_it_a_failure(
    tmp_path: Path,
) -> None:
    run_root, evidence_root, controls_path = _bundle(tmp_path)
    path = evidence_root / "random_loss_robustness.json"
    payload = json.loads(path.read_text())
    payload["rows"][0]["paired_probability_def_ci_high"] = 0.01
    _write(path, payload)
    report = audit_model(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        model_id=MODEL,
        model_name="GAT",
        seeds=SEEDS,
        rules=AuditRules(),
    )
    trigger = next(check for check in report["checks"]
                   if check["check_id"] == "trigger_robustness")
    assert trigger["status"] == INDETERMINATE
    assert report["decision"] == WITHHOLD


def test_argmax_gate_can_be_not_applicable_for_sampled_action_rule(tmp_path: Path) -> None:
    run_root, evidence_root, controls_path = _bundle(tmp_path)
    report = audit_model(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        model_id=MODEL,
        model_name="GAT",
        seeds=SEEDS,
        rules=AuditRules(require_deterministic_capability=False),
    )
    capability = next(check for check in report["checks"]
                      if check["check_id"] == "deterministic_capability")
    assert capability["status"] == NOT_APPLICABLE
    assert report["decision"] == ELIGIBLE
