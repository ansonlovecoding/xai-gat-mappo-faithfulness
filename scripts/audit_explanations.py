"""Apply the freshness-aware explanation audit to completed experiment results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dispatch_marl.explanation_audit import AuditRules, audit_framework  # noqa: E402
from dispatch_marl.provenance import atomic_write_json  # noqa: E402


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_csv(path: Path, report: dict) -> None:
    rows = []
    for model in report["models"]:
        for check in model["checks"]:
            rows.append({
                "model": model["model"],
                "model_id": model["model_id"],
                "audit_decision": model["decision"],
                "check_id": check["check_id"],
                "check": check["name"],
                "status": check["status"],
                "evidence": check["evidence"],
                "decision_rule": check["decision_rule"],
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Freshness-Aware Explanation Audit Report",
        "",
        f"**Overall decision:** {report['overall_decision']}",
        "",
        "## Purpose",
        "",
        report["purpose"],
        "",
        "This is an offline release audit for frozen policy checkpoints. It does not retrain the model, "
        "and an eligible result is not proof of a complete causal explanation.",
        "",
        "## How to read the decision",
        "",
    ]
    for status, meaning in report["decision_meaning"].items():
        lines.append(f"- **{status}:** {meaning}")
    for model in report["models"]:
        lines += [
            "",
            f"## {model['model']}",
            "",
            f"**Decision:** {model['decision']}",
            "",
            f"**Permitted use:** {model['permitted_use']}",
            "",
            "| Check | Status | Evidence |",
            "|---|---|---|",
        ]
        for check in model["checks"]:
            evidence = check["evidence"].replace("|", "\\|")
            lines.append(f"| {check['name']} | {check['status']} | {evidence} |")
        lines += [
            "",
            "### Checkpoint decisions",
            "",
            "| Training seed | Decision | Failed gates | Missing gates |",
            "|---:|---|---|---|",
        ]
        for checkpoint in model["checkpoints"]:
            failed = ", ".join(
                name.replace("_", " ") for name in checkpoint["failed_gates"]
            ) or "None"
            missing = ", ".join(
                name.replace("_", " ") for name in checkpoint["missing_gates"]
            ) or "None"
            lines.append(
                f"| {checkpoint['training_seed']} | {checkpoint['decision']} | "
                f"{failed} | {missing} |"
            )
        lines += ["", "### Failed or missing evidence"]
        issues = [check for check in model["checks"] if check["status"] != "PASS"]
        if issues:
            for check in issues:
                lines.append(f"- **{check['name']}:** {check['decision_rule']}")
        else:
            lines.append("- None.")
    lines += [
        "",
        "## Operational use",
        "",
        "1. Freeze the candidate checkpoint and evaluation protocol.",
        "2. Generate clean/degraded pairs, action-aware faithfulness records, action strata, and sensitivity controls.",
        "3. Run experiment preflight. A preflight pass only confirms evidence integrity.",
        "4. Run this audit and inspect every model-level check.",
        "5. Present attention externally only when the decision is ELIGIBLE, together with telemetry freshness and scope.",
        "6. Repeat the audit after retraining, checkpoint replacement, telemetry changes, or deployment-scenario changes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/dissertation_v9_exposure_audit.toml",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--fail-on-withhold", action="store_true",
        help="return a non-zero status when the audit is not ELIGIBLE",
    )
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    audit_cfg = config.get("explanation_audit", {})
    run_root = args.run_root or _resolve(config["output_root"])
    evidence_root = args.evidence_root or _resolve(
        audit_cfg.get("evidence_root", "results/dissertation_v9_exposure_audit")
    )
    out = args.out or _resolve(
        audit_cfg.get("output_root", "results/dissertation_v9_exposure_audit/explanation_audit")
    )
    controls_path = evidence_root / audit_cfg.get(
        "controls_summary", "faithfulness_controls/summary.json"
    )
    rules = AuditRules(
        decision_relevance_ci_floor=float(audit_cfg.get("decision_relevance_ci_floor", 0.0)),
        minimum_dispatch_decisions=int(audit_cfg.get("minimum_dispatch_decisions", 1)),
        direction_epsilon=float(audit_cfg.get("direction_epsilon", 1e-9)),
        require_deterministic_capability=bool(
            audit_cfg.get("require_deterministic_capability", True)
        ),
        require_trigger_robustness=bool(audit_cfg.get("require_trigger_robustness", True)),
    )
    models = [
        (model["id"], model.get("label", model["id"]))
        for model in config["models"] if model.get("faithfulness_sweep")
    ]
    report = audit_framework(
        run_root=run_root,
        evidence_root=evidence_root,
        controls_path=controls_path,
        models=models,
        seeds=config["training"]["seeds"],
        rules=rules,
    )
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "audit_report.json", report)
    _write_csv(out / "audit_checks.csv", report)
    _write_markdown(out / "AUDIT_REPORT.md", report)

    print(f"overall decision: {report['overall_decision']}")
    for model in report["models"]:
        print(f"  {model['model']}: {model['decision']}")
        for check in model["checks"]:
            print(f"    {check['status']:<10} {check['name']}")
    print(f"report: {out / 'AUDIT_REPORT.md'}")
    return 2 if args.fail_on_withhold and report["overall_decision"] != "ELIGIBLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
