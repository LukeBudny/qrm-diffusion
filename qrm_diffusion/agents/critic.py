from __future__ import annotations

import torch
from torch import nn


class QualityCritic(nn.Module):
    """Predict terminal relative image reward from a sampler observation."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != 6:
            raise ValueError("Critic observations must contain six features")
        return self.network(observations.float()).squeeze(-1)
