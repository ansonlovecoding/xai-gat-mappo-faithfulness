import json

import pytest

from dispatch_marl.provenance import create_or_validate_manifest


def test_manifest_accepts_only_the_same_experiment_specification(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    first = create_or_validate_manifest(path, {"seed": 42}, {"machine": "test"})
    repeated = create_or_validate_manifest(path, {"seed": 42}, {"machine": "other"})
    assert repeated["experiment_id"] == first["experiment_id"]
    assert json.loads(path.read_text())["seed"] == 42

    with pytest.raises(RuntimeError, match="belongs to experiment"):
        create_or_validate_manifest(path, {"seed": 43}, {"machine": "test"})


def test_manifest_rejects_changed_input_inventory(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    create_or_validate_manifest(
        path, {"seed": 42}, {"inputs": [{"path": "scenario", "sha256": "a"}]}
    )
    with pytest.raises(RuntimeError, match="input-file inventory changed"):
        create_or_validate_manifest(
            path, {"seed": 42}, {"inputs": [{"path": "scenario", "sha256": "b"}]}
        )
