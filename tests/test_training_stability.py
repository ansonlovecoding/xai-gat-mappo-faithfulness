from __future__ import annotations

import numpy as np
import pytest
import torch

from dispatch_marl.models import (
    DispatchGATPolicy,
    DispatchMLPPolicy,
    MLPPolicyConfig,
    PolicyConfig,
)
from dispatch_marl import StepMetrics
from dispatch_marl.training import AgentStep, PPOConfig, compute_gae
from dispatch_marl.training import _effective_entropy_coefficient
from dispatch_marl.training import _grouped_minibatches
from dispatch_marl.training import collect_rollout
from scripts.check_training_stability import assess_training_stability


def _obs(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "self": torch.randn(batch_size, 5),
        "neighbor_taxis": torch.randn(batch_size, 5, 5),
        "neighbor_taxis_mask": torch.ones(batch_size, 5),
        "reservations": torch.randn(batch_size, 5, 5),
        "reservations_mask": torch.ones(batch_size, 5),
        "position_valid": torch.ones(batch_size, 1),
    }


@pytest.mark.parametrize("policy_kind", ["gat", "mlp"])
def test_centralised_critic_keeps_transition_groups_independent(policy_kind: str) -> None:
    torch.manual_seed(4)
    if policy_kind == "gat":
        policy = DispatchGATPolicy(PolicyConfig(hidden_dim=16, n_heads=2))
    else:
        policy = DispatchMLPPolicy(MLPPolicyConfig(hidden_dim=16, n_layers=1))
    policy.eval()

    obs = _obs()
    groups = torch.tensor([10, 10, 20, 20])
    first = policy.forward(obs, critic_group=groups)["value"].detach()

    changed = {key: value.clone() for key, value in obs.items()}
    for key in ("self", "neighbor_taxis", "reservations"):
        changed[key][2:] += 100.0
    second = policy.forward(changed, critic_group=groups)["value"].detach()

    assert torch.allclose(first[:2], second[:2], atol=1e-6)
    assert torch.allclose(first[0], first[1], atol=1e-6)
    assert torch.allclose(second[2], second[3], atol=1e-6)
    assert not torch.allclose(first[2:], second[2:])


def test_grouped_minibatches_never_split_a_transition() -> None:
    transition_ids = np.array([0, 0, 1, 1, 1, 2, 2, 3])
    batches = _grouped_minibatches(
        transition_ids, minibatch_size=4, rng=np.random.default_rng(7)
    )

    observed: list[int] = []
    for batch in batches:
        observed.extend(batch.tolist())
        for transition_id in np.unique(transition_ids[batch]):
            expected = set(np.flatnonzero(transition_ids == transition_id))
            assert expected <= set(batch.tolist())
    assert sorted(observed) == list(range(len(transition_ids)))


def test_adaptive_entropy_coefficient_is_bounded_and_inactive_above_floor() -> None:
    config = PPOConfig(ent_coef=0.05, entropy_floor=0.2, max_ent_coef=0.2)

    assert _effective_entropy_coefficient(config, 0.4) == pytest.approx(0.05)
    assert _effective_entropy_coefficient(config, 0.1) == pytest.approx(0.1)
    assert _effective_entropy_coefficient(config, 0.01) == pytest.approx(0.2)


def test_rollout_keeps_team_metric_but_assigns_individual_dispatch_credit() -> None:
    one = {key: value[0].numpy() for key, value in _obs(batch_size=1).items()}

    class FakeEnv:
        done = False

        def reset(self, options=None):
            return {"taxi_0": one, "taxi_1": one}, {}

        def step(self, actions):
            self.done = True
            infos = {
                "taxi_0": {"pickups_delta": 1, "team_reward": 10.5,
                           "training_reward": 10.5},
                "taxi_1": {"pickups_delta": 1, "team_reward": 10.5,
                           "training_reward": 10.0},
            }
            return {}, {"taxi_0": 10.5, "taxi_1": 10.5}, {}, {}, infos

    class FakePolicy:
        def eval(self):
            return self

        def get_action_and_value(self, obs):
            return {
                "action": torch.tensor([1, 0]),
                "log_prob": torch.zeros(2),
                "value": torch.zeros(2),
            }

    buffer, stats = collect_rollout(FakeEnv(), FakePolicy())

    assert stats.total_reward == 10.5
    assert stats.total_pickups == 1
    assert [step.reward for step in buffer] == [10.5, 10.0]
    assert {step.transition_id for step in buffer} == {0}


def test_busy_agent_receives_team_reward_until_next_decision() -> None:
    one = {key: value[0].numpy() for key, value in _obs(batch_size=1).items()}

    class FakeEnv:
        done = False
        last_step_metrics = StepMetrics()

        def reset(self, options=None):
            self.index = 0
            self.done = False
            return {"taxi_0": one, "taxi_1": one}, {}

        def step(self, actions):
            sequence = [
                (0.0, {"taxi_1": one}, False),
                (10.0, {"taxi_0": one, "taxi_1": one}, False),
                (0.0, {}, True),
            ]
            reward, obs, terminal = sequence[self.index]
            self.index += 1
            self.done = terminal
            self.last_step_metrics = StepMetrics(
                completed_passenger_journeys=int(reward > 0),
                team_reward=reward,
                terminated=terminal,
            )
            infos = {
                agent: {"training_reward": reward, "team_reward": reward}
                for agent in actions
            }
            return obs, {agent: reward for agent in actions}, {}, {}, infos

    class FakePolicy:
        def eval(self):
            return self

        def get_action_and_value(self, obs):
            size = len(obs["self"])
            return {
                "action": torch.zeros(size, dtype=torch.long),
                "log_prob": torch.zeros(size),
                "value": torch.zeros(size),
            }

    buffer, stats = collect_rollout(FakeEnv(), FakePolicy())
    compute_gae(buffer, gamma=1.0, gae_lambda=1.0)

    taxi_0 = [step for step in buffer if step.agent == "taxi_0"]
    taxi_1 = [step for step in buffer if step.agent == "taxi_1"]
    assert taxi_0[0].reward_trace == [0.0, 10.0]
    assert taxi_0[0].duration_steps == 2
    assert taxi_0[0].ret == pytest.approx(10.0)
    assert taxi_1[0].ret == pytest.approx(10.0)
    assert stats.total_completed_passenger_journeys == 1
    assert stats.total_reward == pytest.approx(10.0)


def test_interval_aware_gae_uses_elapsed_steps_and_terminal_boundary() -> None:
    empty_obs: dict[str, np.ndarray] = {}
    first = AgentStep(
        agent="taxi_0", obs=empty_obs, action=1, log_prob=0.0, value=3.0,
        reward=3.0, reward_trace=[1.0, 2.0], duration_steps=2,
        transition_id=0,
    )
    final = AgentStep(
        agent="taxi_0", obs=empty_obs, action=0, log_prob=0.0, value=5.0,
        reward=4.0, reward_trace=[4.0], duration_steps=1, terminal=True,
        transition_id=2,
    )

    compute_gae([first, final], gamma=0.5, gae_lambda=0.5)

    assert final.ret == pytest.approx(4.0)
    assert first.reward == pytest.approx(2.0)
    assert first.advantage == pytest.approx(0.1875)
    assert first.ret == pytest.approx(3.1875)


def test_stability_gate_rejects_late_zero_pickup_collapse() -> None:
    rows = [{"pickups": value} for value in [8, 9, 7, 0, 0, 0, 0, 0]]
    result = assess_training_stability(
        rows,
        selected_mean_pickups=12.0,
        final_mean_pickups=2.0,
        final_window=5,
        minimum_final_mean_pickups=5.0,
        maximum_consecutive_zero_pickups=4,
        minimum_selected_mean_pickups=5.0,
        minimum_final_validation_retention=0.8,
    )
    assert not result["ok"]
    assert not result["checks"]["final_mean_pickups"]
    assert not result["checks"]["maximum_consecutive_zero_pickups"]
    assert not result["checks"]["final_validation_retention"]


@pytest.mark.parametrize("policy_kind", ["gat", "mlp"])
def test_critic_gradient_scale_blocks_value_gradient_from_shared_encoder(
    policy_kind: str,
) -> None:
    torch.manual_seed(11)
    if policy_kind == "gat":
        policy = DispatchGATPolicy(PolicyConfig(
            hidden_dim=16, n_heads=2, critic_encoder_gradient_scale=0.0
        ))
        shared = policy.self_embed.weight
    else:
        policy = DispatchMLPPolicy(MLPPolicyConfig(
            hidden_dim=16, n_layers=1, critic_encoder_gradient_scale=0.0
        ))
        shared = policy.trunk[0].weight

    policy.forward(_obs())["value"].sum().backward()

    assert shared.grad is None or torch.count_nonzero(shared.grad) == 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.critic_head.parameters()
    )
