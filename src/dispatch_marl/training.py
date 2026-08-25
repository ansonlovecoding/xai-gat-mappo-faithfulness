"""MAPPO training utilities for the dispatch env.

Kept intentionally as one module rather than a package until it needs to
grow — most of the logic here is orchestration glue that reads better in
one file.

Components:

  * ``AgentStep`` — one atomic (obs, action, log_prob, value, reward)
    record for one taxi at one RL step.
  * ``collect_rollout`` — run one episode, produce a flat list of
    ``AgentStep`` records across all taxis.
  * ``compute_gae`` — group by agent, run backward-pass GAE per trajectory.
  * ``ppo_update`` — one round of PPO gradient steps over the collected
    buffer, with mini-batching, clipped surrogate loss, value loss, and
    entropy bonus.

Design notes worth citing later in the thesis:

  * We use a **shared policy** across taxis (parameter sharing MAPPO). Homogeneous
    fleet, minimal parameter budget, and every collected agent-step contributes
    to the same gradient — critical for sample efficiency on the ~2 k
    agent-steps a 1200-second episode yields.
  * **Team reward** is broadcast to every acting agent at the same step. Value
    function absorbs credit-assignment across the future. When a taxi is
    mid-ride (not in ``env.agents``), it accrues no records — a simplification
    of the underlying SMDP that we may revisit if training stalls.
  * **GAE per agent trajectory**: each taxi's trajectory is a contiguous
    sequence of records (in the order it was seen); we treat episode end as
    terminal (V_{T+1} = 0).
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
    """One agent's decision at one RL step, plus what happened afterwards."""
    agent: str
    obs: dict[str, np.ndarray]  # copy of the obs dict for this agent (numpy)
    action: int
    log_prob: float
    value: float
    reward: float
    # Filled in by compute_gae.
    advantage: float = 0.0
    ret: float = 0.0  # target for the critic


@dataclass
class EpisodeStats:
    total_pickups: int = 0
    total_reward: float = 0.0
    rl_steps: int = 0
    n_agent_steps: int = 0


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
        else:
            actions = {}
            agents = []
            actions_np = np.zeros(0, dtype=int)
            log_probs = np.zeros(0, dtype=np.float32)
            values = np.zeros(0, dtype=np.float32)

        # env.step advances the sim by step_length_s and returns team reward.
        next_obs, rewards, _, _, infos = env.step(actions)
        team_reward = float(next(iter(rewards.values()))) if rewards else 0.0
        info = next(iter(infos.values())) if infos else {}
        stats.rl_steps += 1
        stats.total_reward += team_reward
        stats.total_pickups += int(info.get("pickups_delta", 0))

        # One AgentStep per agent that acted this step.
        for i, agent in enumerate(agents):
            per_agent_obs = {k: v[i].cpu().numpy() for k, v in batched.items()}
            buffer.append(AgentStep(
                agent=agent,
                obs=per_agent_obs,
                action=int(actions_np[i]),
                log_prob=float(log_probs[i]),
                value=float(values[i]),
                reward=team_reward,
            ))
        stats.n_agent_steps += len(agents)
        obs_dict = next_obs

    return buffer, stats


# ================================================================== GAE ======


def compute_gae(
    buffer: list[AgentStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """Fill in `advantage` and `ret` on each AgentStep, per agent trajectory.

    Modifies the buffer in place. Assumes agent-steps for the same agent are
    stored in temporal order (which `collect_rollout` guarantees).
    """
    by_agent: dict[str, list[AgentStep]] = defaultdict(list)
    for step in buffer:
        by_agent[step.agent].append(step)

    for steps in by_agent.values():
        gae = 0.0
        for t in reversed(range(len(steps))):
            v_next = steps[t + 1].value if t + 1 < len(steps) else 0.0
            delta = steps[t].reward + gamma * v_next - steps[t].value
            gae = delta + gamma * gae_lambda * gae
            steps[t].advantage = gae
            steps[t].ret = gae + steps[t].value


# ============================================================ PPO update =====


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


@dataclass
class UpdateLog:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    n_updates: int = 0

    def as_dict(self) -> dict[str, float]:
        n = max(1, self.n_updates)
        return {
            "policy_loss": self.policy_loss / n,
            "value_loss": self.value_loss / n,
            "entropy": self.entropy / n,
            "approx_kl": self.approx_kl / n,
            "clipfrac": self.clipfrac / n,
        }


def _stack_obs(buffer: list[AgentStep], device: str) -> dict[str, torch.Tensor]:
    """Turn list[AgentStep].obs (dict-of-arrays) into a batched tensor dict."""
    keys = list(buffer[0].obs.keys())
    stacked = {}
    for k in keys:
        arr = np.stack([s.obs[k] for s in buffer], axis=0)
        stacked[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return stacked


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
    advantages = torch.as_tensor([s.advantage for s in buffer], dtype=torch.float32, device=device)
    returns = torch.as_tensor([s.ret for s in buffer], dtype=torch.float32, device=device)
    if config.normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rng = rng or np.random.default_rng()
    n = len(buffer)
    idxs = np.arange(n)
    log = UpdateLog()

    for _ in range(config.ppo_epochs):
        rng.shuffle(idxs)
        for start in range(0, n, config.minibatch_size):
            mb_idx = idxs[start:start + config.minibatch_size]
            mb = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

            mb_obs = {k: v[mb] for k, v in obs_stacked.items()}
            out = policy.get_action_and_value(mb_obs, action=actions[mb])

            new_log_probs = out["log_prob"]
            values = out["value"]
            entropy = out["entropy"].mean()

            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = log_ratio.exp()
            mb_adv = advantages[mb]

            # Clipped policy loss.
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (unclipped MSE is fine at this scale).
            value_loss = F.mse_loss(values, returns[mb])

            loss = (
                policy_loss
                + config.vf_coef * value_loss
                - config.ent_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log.policy_loss += float(policy_loss)
                log.value_loss += float(value_loss)
                log.entropy += float(entropy)
                log.approx_kl += float((-log_ratio).mean())
                log.clipfrac += float(((ratio - 1.0).abs() > config.clip_ratio).float().mean())
                log.n_updates += 1

    return log
