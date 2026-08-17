from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TrajectoryEntry:
    """Diagnostics for one model evaluation and solver transition."""

    step_index: int
    sigma: float
    next_sigma: float
    step_size: float | None
    action: float
    quality: tuple[float, ...] | None = None
    modulation_norm: float | None = None
    denoised_norm: float | None = None
    sample_norm: float | None = None
    policy_features: tuple[float, ...] | None = None


@dataclass
class AdaptiveSamplerState:
    """Mutable receding-horizon sampler state shared by solver/controller."""

    current_sigma: torch.Tensor
    previous_sigma: torch.Tensor | None
    previous_denoised: torch.Tensor | None
    remaining_steps: int
    previous_step_size: torch.Tensor | None = None
    step_index: int = 0
    trajectory_trace: list[TrajectoryEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.current_sigma.numel() != 1:
            raise ValueError("The timestep controller requires one shared sigma per image batch")
        if self.remaining_steps <= 0:
            raise ValueError("remaining_steps must be positive before a solver transition")
        if not torch.isfinite(self.current_sigma).all() or self.current_sigma.item() <= 0:
            raise ValueError("current_sigma must be finite and strictly positive")
