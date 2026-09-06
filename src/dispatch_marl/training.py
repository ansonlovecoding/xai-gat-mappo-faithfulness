"""MAPPO rollout collection, advantage estimation, and optimization utilities.

Components:

  * ``AgentStep`` — one decision and the rewards observed until that taxi's
    next decision opportunity.
  * ``collect_rollout`` — run one episode, produce a flat list of
    ``AgentStep`` records across all taxis.
  * ``compute_gae`` — group by agent, run backward-pass GAE per trajectory.
  * ``ppo_update`` — one round of PPO gradient steps over the collected
    buffer, with mini-batching, clipped surrogate loss, value loss, and
    entropy bonus.

Design decisions:

  * All taxis share one policy. Every collected agent step therefore contributes
    to the same gradient, which improves sample efficiency.
  * Team task reward is shared, while the taxi whose dispatch succeeds receives
    an additional difference-credit bonus during training. This distinguishes
    its action from another taxi's no-op without changing reported team reward.
    A taxi records no actor decisions while it is serving a passenger, but its
    pending decision continues to receive the shared reward.
  * **Interval-aware GAE**: each taxi's trajectory is a sequence of decision
    events. Returns use the number of environment steps between events;
    episode end is terminal (V_{T+1} = 0).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import DispatchEnv
from .models.policy import DispatchGATPolicy, obs_dict_to_tensors


# ============================================================ Rollout buffer =


@dataclass
class AgentStep:
    """One agent decision and its semi-Markov reward interval."""
    agent: str
    obs: dict[str, np.ndarray]  # copy of the obs dict for this agent (numpy)
    action: int
    log_prob: float
    value: float
    reward: float
    reward_trace: list[float] = field(default_factory=list)
    duration_steps: int = 0
    terminal: bool = False
    # All agents acting in one environment step share a transition id.  The
    # centralised critic uses it to reconstruct joint states during PPO.
    transition_id: int = 0
    # Filled in by compute_gae.
    advantage: float = 0.0
    ret: float = 0.0  # target for the critic


@dataclass
class EpisodeStats:
    total_completed_passenger_journeys: int = 0
    total_reward: float = 0.0
    rl_steps: int = 0
    n_agent_steps: int = 0

    @property
    def total_pickups(self) -> int:
        """Protocol-v1 compatibility alias for completed passenger journeys."""
        return self.total_completed_passenger_journeys


def collect_rollout(
    env: DispatchEnv,
    policy: DispatchGATPolicy,
    device: str = "cpu",
    reset_options: dict | None = None,
) -> tuple[list[AgentStep], EpisodeStats]:
    """Run one full episode with the current policy, return per-agent-step records.

    `reset_options` is forwarded to env.reset() — used by the demand-variant
    protocol to rotate rider files per episode.
    """
    buffer: list[AgentStep] = []
    pending: dict[str, AgentStep] = {}
    stats = EpisodeStats()

    obs_dict, _ = env.reset(options=reset_options)
    policy.eval()

    while not env.done:
        if obs_dict:
            batched, agents = obs_dict_to_tensors(obs_dict, device=device)
            with torch.no_grad():
                out = policy.get_action_and_value(batched)
            actions_np = out["action"].cpu().numpy().astype(int)
            log_probs = out["log_prob"].cpu().numpy()
            values = out["value"].cpu().numpy()
            actions = {a: int(actions_np[i]) for i, a in enumerate(agents)}

            # A new decision opportunity closes the previous action's reward
            # interval for that taxi. The current value becomes V(s_{t+Δ})
            # through the next AgentStep in its per-agent trajectory.
            for agent in agents:
                previous = pending.pop(agent, None)
                if previous is not None:
                    buffer.append(previous)
        else:
            actions = {}
            agents = []
            actions_np = np.zeros(0, dtype=int)
            log_probs = np.zeros(0, dtype=np.float32)
            values = np.zeros(0, dtype=np.float32)

        # env.step advances the sim by step_length_s and returns team reward.
        next_obs, rewards, _, _, infos = env.step(actions)
        transition_id = stats.rl_steps
        metrics = getattr(env, "last_step_metrics", None)
        info = next(iter(infos.values())) if infos else {}
        fallback_reward = float(next(iter(rewards.values()))) if rewards else 0.0
        team_reward = float(
            metrics.team_reward if metrics is not None
            else info.get("team_reward", fallback_reward)
        )
        stats.rl_steps += 1
        stats.total_reward += team_reward
        stats.total_completed_passenger_journeys += int(
            metrics.completed_passenger_journeys if metrics is not None
            else info.get(
                "completed_passenger_journeys_delta",
                info.get("pickups_delta", 0),
            )
        )

        # Open one pending interval for every decision made on this step.
        for i, agent in enumerate(agents):
            per_agent_obs = {k: v[i].cpu().numpy() for k, v in batched.items()}
            pending[agent] = AgentStep(
                agent=agent,
                obs=per_agent_obs,
                action=int(actions_np[i]),
                log_prob=float(log_probs[i]),
                value=float(values[i]),
                reward=0.0,
                transition_id=transition_id,
            )

        # Every unresolved decision receives the shared team outcome, including
        # decisions made by taxis that are currently busy and absent from the
        # action dictionary. Actor-specific dispatch credit applies only on the
        # step where that action was submitted.
        for agent, step in pending.items():
            step_reward = team_reward
            if step.transition_id == transition_id:
                step_reward = float(
                    infos.get(agent, {}).get("training_reward", team_reward)
                )
            step.reward_trace.append(step_reward)
            step.reward += step_reward
            step.duration_steps += 1

        terminated = bool(metrics.terminated) if metrics is not None else env.done
        if terminated:
            for step in pending.values():
                step.terminal = True
                buffer.append(step)
            pending.clear()
        stats.n_agent_steps += len(agents)
        obs_dict = next_obs

    # Defensive finalization for custom environments that set ``done`` without
    # exposing a terminal StepMetrics record.
    for step in pending.values():
        step.terminal = True
        buffer.append(step)

    return buffer, stats


# ================================================================== GAE ======


def compute_gae(
    buffer: list[AgentStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """Fill in `advantage` and `ret` on each AgentStep, per agent trajectory.

    Rewards within an interval are discounted per environment step. Bootstrap
    and trace discounts use the interval length, so decisions separated by a
    long service period are not treated as adjacent environment steps.
    """
    by_agent: dict[str, list[AgentStep]] = defaultdict(list)
    for step in buffer:
        by_agent[step.agent].append(step)

    for steps in by_agent.values():
        gae = 0.0
        for t in reversed(range(len(steps))):
            step = steps[t]
            rewards = step.reward_trace or [step.reward]
            duration = max(1, step.duration_steps or len(rewards))
            interval_reward = float(sum(
                (gamma ** offset) * reward
                for offset, reward in enumerate(rewards)
            ))
            step.reward = interval_reward
            has_next = t + 1 < len(steps) and not step.terminal
            v_next = steps[t + 1].value if has_next else 0.0
            delta = interval_reward + (gamma ** duration) * v_next - step.value
            continuation = ((gamma * gae_lambda) ** duration) * gae if has_next else 0.0
            gae = delta + continuation
            step.advantage = gae
            step.ret = gae + step.value


# ============================================================ PPO update =====


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    entropy_floor: float | None = None
    max_ent_coef: float | None = None
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    value_clip_ratio: float | None = 0.2
    target_kl: float | None = 0.015


@dataclass
class UpdateLog:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    decision_entropy: float = 0.0
    effective_ent_coef: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    n_updates: int = 0
    kl_early_stops: int = 0

    def as_dict(self) -> dict[str, float]:
        n = max(1, self.n_updates)
        return {
            "policy_loss": self.policy_loss / n,
            "value_loss": self.value_loss / n,
            "entropy": self.entropy / n,
            "decision_entropy": self.decision_entropy / n,
            "effective_ent_coef": self.effective_ent_coef / n,
            "approx_kl": self.approx_kl / n,
            "clipfrac": self.clipfrac / n,
            "kl_early_stops": float(self.kl_early_stops),
        }


def _stack_obs(buffer: list[AgentStep], device: str) -> dict[str, torch.Tensor]:
    """Turn list[AgentStep].obs (dict-of-arrays) into a batched tensor dict."""
    keys = list(buffer[0].obs.keys())
    stacked = {}
    for k in keys:
        arr = np.stack([s.obs[k] for s in buffer], axis=0)
        stacked[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return stacked


def _grouped_minibatches(
    transition_ids: np.ndarray,
    minibatch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Build minibatches without splitting one environment transition."""
    groups = np.unique(transition_ids)
    rng.shuffle(groups)
    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []
    current_size = 0
    for group in groups:
        members = np.flatnonzero(transition_ids == group)
        if current and current_size + len(members) > minibatch_size:
            batches.append(np.concatenate(current))
            current = []
            current_size = 0
        current.append(members)
        current_size += len(members)
    if current:
        batches.append(np.concatenate(current))
    return batches


def _effective_entropy_coefficient(
    config: PPOConfig,
    decision_entropy: float,
) -> float:
    """Return the bounded entropy coefficient for the current minibatch."""
    coefficient = config.ent_coef
    if config.entropy_floor is not None:
        coefficient *= max(1.0, config.entropy_floor / max(decision_entropy, 1e-6))
    if config.max_ent_coef is not None:
        coefficient = min(coefficient, config.max_ent_coef)
    return coefficient


def ppo_update(
    policy: DispatchGATPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: list[AgentStep],
    config: PPOConfig,
    device: str = "cpu",
    rng: np.random.Generator | None = None,
) -> UpdateLog:
    """One round of PPO updates over the buffer. Returns mean-loss stats."""
    if not buffer:
        return UpdateLog()

    policy.train()

    # Materialise buffer into torch tensors once.
    obs_stacked = _stack_obs(buffer, device)
    actions = torch.as_tensor([s.action for s in buffer], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor([s.log_prob for s in buffer], dtype=torch.float32, device=device)
    old_values = torch.as_tensor([s.value for s in buffer], dtype=torch.float32, device=device)
    advantages = torch.as_tensor([s.advantage for s in buffer], dtype=torch.float32, device=device)
    returns = torch.as_tensor([s.ret for s in buffer], dtype=torch.float32, device=device)
    transition_ids = np.asarray([s.transition_id for s in buffer], dtype=np.int64)
    if config.normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rng = rng or np.random.default_rng()
    log = UpdateLog()

    for _ in range(config.ppo_epochs):
        stop_for_kl = False
        for mb_idx in _grouped_minibatches(
            transition_ids, config.minibatch_size, rng
        ):
            mb = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

            mb_obs = {k: v[mb] for k, v in obs_stacked.items()}
            critic_group = torch.as_tensor(
                transition_ids[mb_idx], dtype=torch.long, device=device
            )
            out = policy.get_action_and_value(
                mb_obs, action=actions[mb], critic_group=critic_group
            )

            new_log_probs = out["log_prob"]
            values = out["value"]
            entropy_all = out["entropy"].mean()
            decision_mask = mb_obs["reservations_mask"].sum(dim=1) > 0
            decision_entropy = (
                out["entropy"][decision_mask].mean()
                if bool(decision_mask.any()) else entropy_all
            )
            effective_ent_coef = _effective_entropy_coefficient(
                config, float(decision_entropy.detach())
            )

            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = log_ratio.exp()
            mb_adv = advantages[mb]

            # Clipped policy loss.
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # PPO-style value clipping limits critic jumps that can overwhelm
            # the shared encoder and erase a useful actor policy.
            if config.value_clip_ratio is None:
                value_loss = 0.5 * F.mse_loss(values, returns[mb])
            else:
                value_delta = values - old_values[mb]
                clipped_values = old_values[mb] + value_delta.clamp(
                    -config.value_clip_ratio, config.value_clip_ratio
                )
                value_loss = 0.5 * torch.maximum(
                    (values - returns[mb]).pow(2),
                    (clipped_values - returns[mb]).pow(2),
                ).mean()

            loss = (
                policy_loss
                + config.vf_coef * value_loss
                - effective_ent_coef * decision_entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log.policy_loss += float(policy_loss)
                log.value_loss += float(value_loss)
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                log.entropy += float(entropy_all)
                log.decision_entropy += float(decision_entropy)
                log.effective_ent_coef += effective_ent_coef
                log.approx_kl += float(approx_kl)
                log.clipfrac += float(((ratio - 1.0).abs() > config.clip_ratio).float().mean())
                log.n_updates += 1
                if config.target_kl is not None and float(approx_kl) > config.target_kl:
                    log.kl_early_stops += 1
                    stop_for_kl = True
            if stop_for_kl:
                break
        if stop_for_kl:
            break

    return log
