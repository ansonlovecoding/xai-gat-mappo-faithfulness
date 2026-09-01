import json

from scripts.summarize_dissertation_experiment import summarize_training_seeds


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
