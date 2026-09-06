import numpy as np

from scripts.analyze_hypotheses import spearman, spearman_permutation_p
from scripts.summarize_precision_sensitivity import compare_analyses


def _analysis(h1_rho: float, h1_p: float, paired: float, h4_rho: float) -> dict:
    return {
        "H1_outage_duration": {"per_level_mean": {"0": 0.001, "60": 0.0}},
        "robust": {
            "H1_robust": {
                "rho": h1_rho,
                "rho_cluster_ci95": [-0.2, 0.1],
            },
            "H4_within_episode": {"mean_rho_within_episode": h4_rho},
            "exposure_conditioned_paired_followup": {
                "probability_def": {
                    "mean_degraded_minus_clean": paired,
                    "ci95": [-0.001, 0.001],
                },
            },
            "confirmatory_family_holm_p": {
                "H1": h1_p,
                "H2": 0.5,
                "H3": 0.01,
                "H4": 0.2,
            },
        },
    }


def test_compare_analyses_reports_direction_and_verdict_changes() -> None:
    row = compare_analyses(
        _analysis(-0.1, 0.04, -0.00001, -0.2),
        _analysis(0.1, 0.06, 0.0, -0.1),
    )

    assert row["h1_rho_direction_changed"] is True
    assert row["paired_def_mean_direction_changed"] is True
    assert row["h4_mean_rho_direction_changed"] is False
    assert row["h1_verdict_changed"] is True
    assert row["h3_verdict_changed"] is False


def test_rank_permutation_matches_direct_spearman_permutation() -> None:
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    y = np.array([0.2, 0.1, 0.4, 0.4, 0.3, 0.6])
    permutations = 200
    seed = 17

    rho, optimized_p = spearman_permutation_p(
        x, y, "less", permutations, np.random.default_rng(seed)
    )
    direct_rng = np.random.default_rng(seed)
    shuffled = y.copy()
    count = 0
    for _ in range(permutations):
        direct_rng.shuffle(shuffled)
        if spearman(x, shuffled) <= rho:
            count += 1
    direct_p = (count + 1) / (permutations + 1)

    assert rho == spearman(x, y)
    assert optimized_p == direct_p
