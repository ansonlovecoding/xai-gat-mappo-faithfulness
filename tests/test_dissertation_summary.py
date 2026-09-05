import json

import pytest

from scripts.analyze_hypotheses import action_stratum_masks, decisions_frame
from scripts.summarize_dissertation_experiment import (
    summarize_deterministic_diagnostics,
    summarize_training_seeds,
)


def _analysis(shift: float, paired_def_delta: float, supported: set[str]) -> dict:
    holm = {name: (0.01 if name in supported else 0.5)
            for name in ("H1", "H2", "H3", "H4")}
    return {
        "H2_outage_duration": {"mean_delta_faith_minus_perf": 0.2},
        "robust": {
            "primary_stale_attention_shift": {"mean_shift": shift},
            "exposure_conditioned_paired_followup": {
                "probability_def": {
                    "mean_degraded_minus_clean": paired_def_delta,
                    "p_one_sided_decrease": 0.01,
                },
                "margin_def": {
                    "mean_degraded_minus_clean": paired_def_delta * 10,
                },
            },
            "H1_robust": {"rho": -0.1},
            "H3_robust": {"rho": 0.3},
            "H4_within_episode": {"mean_rho_within_episode": -0.2},
            "confirmatory_family_holm_p": holm,
        },
    }


def test_training_seed_synthesis_reports_consistency_without_pooling(tmp_path) -> None:
    model_dir = tmp_path / "sweeps" / "B2_gat"
    fixtures = [
        (42, 0.3, -0.03, {"H3"}),
        (43, 0.1, -0.01, {"H3", "H4"}),
        (44, -0.2, 0.02, {"H3"}),
    ]
    for seed, shift, paired_def_delta, supported in fixtures:
        sweep_dir = model_dir / f"seed_{seed}"
        sweep_dir.mkdir(parents=True)
        (sweep_dir / "analysis.json").write_text(
            json.dumps(_analysis(shift, paired_def_delta, supported))
        )

    aggregate, per_seed = summarize_training_seeds(tmp_path)

    assert len(per_seed) == 3
    assert aggregate[0]["positive_stale_attention_shift_seeds"] == 2
    assert aggregate[0]["negative_paired_probability_def_delta_seeds"] == 2
    assert aggregate[0]["paired_probability_def_delta_min"] == -0.03
    assert aggregate[0]["stale_attention_shift_min"] == -0.2
    assert aggregate[0]["H3_consistency"] == "consistent"
    assert aggregate[0]["H4_consistency"] == "mixed"
    assert aggregate[0]["H1_consistency"] == "not_supported"


def test_action_strata_exclude_forced_no_op_and_split_eligible_actions() -> None:
    cells = [{
        "cell": {"axis": "clean", "level": 0.0, "seed": 42},
        "faith_records": [
            {
                "action": 0, "pi_full": 1.0, "def": 0.0, "def_m": 0.0,
                "wamsn": 0.0, "valid_reservations": 0,
            },
            {
                "action": 0, "pi_full": 0.9, "def": 0.1, "def_m": 0.2,
                "wamsn": 0.0, "valid_reservations": 2,
            },
            {
                "action": 2, "pi_full": 0.1, "def": 0.3, "def_m": 0.4,
                "wamsn": 0.0, "valid_reservations": 2,
            },
        ],
    }]

    frame = decisions_frame(cells)
    masks = action_stratum_masks(frame)

    assert masks["all"].tolist() == [False, True, True]
    assert masks["no_op"].tolist() == [False, True, False]
    assert masks["dispatch"].tolist() == [False, False, True]


def test_revised_frame_uses_hybrid_primary_def_and_keeps_standard_sensitivity() -> None:
    cells = [{
        "cell": {"axis": "clean", "level": 0.0, "seed": 42},
        "faith_records": [
            {
                "record_schema_version": 2,
                "action": 0, "pi_full": 0.8,
                "def": 0.1, "def_m": 0.2,
                "primary_def": 0.11, "primary_def_m": 0.21,
                "wamsn": 0.0, "valid_reservations": 2,
            },
            {
                "record_schema_version": 2,
                "action": 1, "pi_full": 0.2,
                "def": 0.3, "def_m": 0.4,
                "def_excl": 0.5, "def_m_excl": 0.6,
                "primary_def": 0.5, "primary_def_m": 0.6,
                "wamsn": 0.0, "valid_reservations": 2,
            },
        ],
    }]

    frame = decisions_frame(cells)

    assert frame["def"].tolist() == pytest.approx([0.11, 0.5])
    assert frame["def_standard"].tolist() == pytest.approx([0.1, 0.3])
    assert frame["def_m"].tolist() == pytest.approx([0.21, 0.6])


def test_revised_frame_rejects_mixed_precision_schemas() -> None:
    base = {
        "action": 0, "pi_full": 0.8, "def": 0.1, "def_m": 0.2,
        "wamsn": 0.0, "valid_reservations": 2,
    }
    cells = [{
        "cell": {"axis": "clean", "level": 0.0, "seed": 42},
        "faith_records": [
            dict(base),
            {**base, "record_schema_version": 2, "primary_def": 0.1},
        ],
    }]

    with pytest.raises(ValueError, match="mixed faithfulness record schemas"):
        decisions_frame(cells)


def test_deterministic_diagnostic_counts_zero_pickup_episodes(tmp_path) -> None:
    output = tmp_path / "deterministic_diagnostics" / "B2_gat" / "seed_42.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({
        "seed": 42,
        "episodes": 3,
        "mean_pickups": 0.0,
        "mean_reward": -2.0,
        "checkpoint_sha256": "abc",
        "per_episode": [
            {"total_pickups": 0, "final_mean_pending_wait_s": 10.0},
            {"total_pickups": 0, "final_mean_pending_wait_s": 20.0},
            {"total_pickups": 0, "final_mean_pending_wait_s": 30.0},
        ],
    }))

    rows = summarize_deterministic_diagnostics(tmp_path)

    assert rows[0]["training_seed"] == 42
    assert rows[0]["mean_final_pending_wait_s"] == 20.0
    assert rows[0]["zero_pickup_episodes"] == 3
    assert rows[0]["all_episodes_zero_pickups"] is True
