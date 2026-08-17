from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from .controller import RecedingHorizonStepPolicy
from .critic import QualityCritic
from .state import TrajectoryEntry


@dataclass(frozen=True)
class UpdateMetrics:
    reward: float
    policy_loss: float
    critic_loss: float
    entropy: float


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
    ) -> None:
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer
        self.exploration_std = float(exploration_std)
        self.entropy_weight = float(entropy_weight)

    def update(
        self, trajectory: Sequence[TrajectoryEntry], terminal_reward: float
    ) -> UpdateMetrics:
        usable = [entry for entry in trajectory if entry.policy_features is not None]
        if not usable:
            raise ValueError("Trajectory contains no policy observations")
        device = next(self.policy.parameters()).device
        observations = torch.tensor(
            [entry.policy_features for entry in usable], device=device, dtype=torch.float32
        )
        actions = torch.tensor(
            [entry.action for entry in usable], device=device, dtype=torch.float32
        )
        returns = torch.full_like(actions, float(terminal_reward))

        means = self.policy.forward_features(observations)
        distribution = torch.distributions.Normal(means, self.exploration_std)
        values = self.critic(observations)
        advantage = (returns - values).detach()
        policy_loss = -(distribution.log_prob(actions) * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + critic_loss - self.entropy_weight * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.critic.parameters()), 1.0
        )
        self.optimizer.step()
        return UpdateMetrics(
            reward=float(terminal_reward),
            policy_loss=float(policy_loss.detach().cpu()),
            critic_loss=float(critic_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
        )
