from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch.nn import functional as F

from .controller import RecedingHorizonStepPolicy
from .critic import QualityCritic
from .state import TrajectoryEntry


@dataclass(frozen=True)
class UpdateMetrics:
    reward_mean: float
    reward_std: float
    policy_loss: float
    critic_loss: float
    entropy: float
    advantage_mean: float
    advantage_std: float
    action_l2: float
    policy_kl: float
    critic_prediction_mean: float

    @property
    def reward(self) -> float:
        """Backward-compatible name for single-trajectory callers."""

        return self.reward_mean


class ActorCriticUpdater:
    """One terminal-reward update over a completed fixed-budget trajectory."""

    def __init__(
        self,
        policy: RecedingHorizonStepPolicy,
        critic: QualityCritic,
        optimizer: torch.optim.Optimizer,
        *,
        exploration_std: float,
        entropy_weight: float = 1.0e-3,
        action_l2_weight: float = 0.0,
        kl_weight: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer
        self.exploration_std = float(exploration_std)
        self.entropy_weight = float(entropy_weight)
        self.action_l2_weight = float(action_l2_weight)
        self.kl_weight = float(kl_weight)
        self.max_grad_norm = float(max_grad_norm)

    def update(
        self, trajectory: Sequence[TrajectoryEntry], terminal_reward: float
    ) -> UpdateMetrics:
        return self.update_batch(
            [trajectory], [terminal_reward], normalize_advantages=False
        )

    def update_batch(
        self,
        trajectories: Sequence[Sequence[TrajectoryEntry]],
        terminal_rewards: Sequence[float],
        *,
        normalize_advantages: bool = True,
    ) -> UpdateMetrics:
        """Update once from several complete fixed-budget trajectories."""

        if len(trajectories) != len(terminal_rewards) or not trajectories:
            raise ValueError("Trajectories and rewards must be equally sized and non-empty")
        usable_trajectories = [
            [entry for entry in trajectory if entry.policy_features is not None]
            for trajectory in trajectories
        ]
        if any(not trajectory for trajectory in usable_trajectories):
            raise ValueError("Every trajectory must contain policy observations")
        device = next(self.policy.parameters()).device
        observations = torch.tensor(
            [
                entry.policy_features
                for trajectory in usable_trajectories
                for entry in trajectory
            ],
            device=device,
            dtype=torch.float32,
        )
        actions = torch.tensor(
            [entry.action for trajectory in usable_trajectories for entry in trajectory],
            device=device,
            dtype=torch.float32,
        )
        returns = torch.cat(
            [
                torch.full(
                    (len(trajectory),), float(reward), device=device, dtype=torch.float32
                )
                for trajectory, reward in zip(usable_trajectories, terminal_rewards)
            ]
        )

        means = self.policy.forward_features(observations)
        distribution = torch.distributions.Normal(means, self.exploration_std)
        values = self.critic(observations)
        raw_advantage = (returns - values).detach()
        advantage = raw_advantage
        if normalize_advantages and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1.0e-8
            )
        policy_loss = -(distribution.log_prob(actions) * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        action_l2 = means.square().mean()
        policy_kl = (means.square() / (2.0 * self.exploration_std**2)).mean()
        loss = (
            policy_loss
            + critic_loss
            - self.entropy_weight * entropy
            + self.action_l2_weight * action_l2
            + self.kl_weight * policy_kl
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()
        rewards = [float(value) for value in terminal_rewards]
        reward_mean = sum(rewards) / len(rewards)
        reward_variance = sum((value - reward_mean) ** 2 for value in rewards) / len(rewards)
        return UpdateMetrics(
            reward_mean=reward_mean,
            reward_std=math.sqrt(reward_variance),
            policy_loss=float(policy_loss.detach().cpu()),
            critic_loss=float(critic_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            advantage_mean=float(raw_advantage.mean().cpu()),
            advantage_std=float(raw_advantage.std(unbiased=False).cpu()),
            action_l2=float(action_l2.detach().cpu()),
            policy_kl=float(policy_kl.detach().cpu()),
            critic_prediction_mean=float(values.detach().mean().cpu()),
        )
