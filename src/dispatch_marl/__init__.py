"""Multi-agent RL environment for fleet dispatch under telemetry degradation."""
from .degradation import DegradationConfig
from .env import DispatchEnv, DispatchEnvConfig
from .faithfulness import (
    DecisionFaithfulness,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
    aggregate_node_attention,
    compute_attention_drift,
    compute_stale_attention_mass,
    compute_stale_attention_share,
    compute_wamsn,
)
from .policies import (
    NearestReservationPolicy,
    NoOpPolicy,
    Policy,
    RandomPolicy,
)
from .reproducibility import derive_seed, seed_everything
from .scenario import Scenario, load_scenario
from .training import (
    EpisodeStats,
    PPOConfig,
    collect_rollout,
    compute_gae,
    ppo_update,
)

__all__ = [
    "DispatchEnv",
    "DispatchEnvConfig",
    "DegradationConfig",
    "Scenario",
    "load_scenario",
    "Policy",
    "NoOpPolicy",
    "RandomPolicy",
    "NearestReservationPolicy",
    "PPOConfig",
    "EpisodeStats",
    "collect_rollout",
    "compute_gae",
    "ppo_update",
    "FaithfulnessConfig",
    "DecisionFaithfulness",
    "FaithfulnessEvaluator",
    "aggregate_node_attention",
    "compute_attention_drift",
    "compute_stale_attention_mass",
    "compute_stale_attention_share",
    "compute_wamsn",
    "derive_seed",
    "seed_everything",
]
