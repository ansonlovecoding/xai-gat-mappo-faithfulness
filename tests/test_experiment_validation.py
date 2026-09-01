import json

from dispatch_marl.experiment_validation import validate_sweep
from dispatch_marl.provenance import MANIFEST_SCHEMA_VERSION


def test_preflight_accepts_complete_type_matched_sweep(tmp_path) -> None:
    cells = [
        {"axis": "clean", "level": 0.0},
        {"axis": "outage_duration", "level": 10.0},
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": "abc",
        "faithfulness_config": {
            "random_baseline": "type_matched",
            "exclusion_variant": True,
        },
        "provenance": {"git": {"dirty": False}},
        "cells": cells,
        "seeds": [42],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    clean = {
        "cell": {**cells[0], "seed": 42},
        "empirical_degradation_rate": 0.0,
        "faith_records": [{"drift": 0.0, "n_stale_veh": 0}],
    }
    degraded = {
        "cell": {**cells[1], "seed": 42},
        "empirical_degradation_rate": 0.1,
        "faith_records": [{
            "drift": 0.1,
            "n_stale_veh": 1,
            "max_aoi_s": 10.0,
            "stale_attention_shift": 0.2,
        }],
    }
    (cell_dir / "clean_0_seed42.json").write_text(json.dumps(clean))
    (cell_dir / "outage_duration_10_seed42.json").write_text(json.dumps(degraded))

    report = validate_sweep(tmp_path)
    assert report.ok, report.errors
    assert report.summary["mean_stale_attention_shift_when_exposed"] == 0.2


def test_preflight_rejects_levels_without_empirical_differentiation(tmp_path) -> None:
    cells = [
        {"axis": "outage_duration", "level": 10.0},
        {"axis": "outage_duration", "level": 20.0},
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": "abc",
        "faithfulness_config": {
            "random_baseline": "type_matched",
            "exclusion_variant": True,
        },
        "provenance": {"git": {"dirty": False}},
        "cells": cells,
        "seeds": [42],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    for level in (10, 20):
        payload = {
            "cell": {"axis": "outage_duration", "level": level, "seed": 42},
            "empirical_degradation_rate": 0.1,
            "faith_records": [{
                "n_stale_veh": 1,
                "max_aoi_s": 10.0,
                "stale_attention_shift": 0.1,
            }],
        }
        (cell_dir / f"outage_duration_{level}_seed42.json").write_text(
            json.dumps(payload)
        )

    report = validate_sweep(tmp_path)
    assert not report.ok
    assert any("no empirical AoI differentiation" in error for error in report.errors)


def test_preflight_enforces_exposure_conditioned_audit_gates(tmp_path) -> None:
    cell = {"axis": "outage_duration", "level": 10.0}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": "abc",
        "faithfulness_config": {
            "random_baseline": "type_matched",
            "exclusion_variant": True,
            "faithfulness_exposed_every": 1,
            "minimum_stale_exposed_records_per_degraded_cell": 3,
            "minimum_stale_exposed_episodes_per_degraded_cell": 2,
        },
        "provenance": {"git": {"dirty": False}},
        "cells": [cell],
        "seeds": [42],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cell_dir = tmp_path / "cells"
    cell_dir.mkdir()
    payload = {
        "cell": {**cell, "seed": 42},
        "empirical_degradation_rate": 0.1,
        "stale_exposure_audit": {
            "stale_exposed_decisions_seen": 2,
            "stale_exposed_decisions_scored": 2,
        },
        "faith_records": [
            {
                "episode": episode,
                "n_stale_veh": 1,
                "max_aoi_s": 10.0,
                "stale_attention_shift": 0.1,
                "paired_def_delta": -0.01,
            }
            for episode in (0, 1)
        ],
    }
    (cell_dir / "outage_duration_10_seed42.json").write_text(
        json.dumps(payload)
    )

    report = validate_sweep(tmp_path)
    assert not report.ok
    assert any("has 2 stale-exposed records; minimum is 3" in error
               for error in report.errors)
