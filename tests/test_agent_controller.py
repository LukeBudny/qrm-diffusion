from __future__ import annotations

import pytest
import torch

from qrm_diffusion.agents import (
    AdaptiveSamplerState,
    FixedBudgetSigmaController,
    RecedingHorizonStepPolicy,
    StepConstraints,
)


def test_learnable_step_policy_is_initialized_to_exact_zero() -> None:
    policy = RecedingHorizonStepPolicy(hidden_dim=16)
    state = AdaptiveSamplerState(
        current_sigma=torch.tensor(0.5),
        previous_sigma=torch.tensor(0.75),
        previous_denoised=None,
        remaining_steps=4,
        previous_step_size=torch.tensor(0.2),
        step_index=2,
    )
    guided = torch.randn(1, 16, 4, 4)
    action = policy(state, guided)
    assert action.item() == 0.0


def test_learnable_policy_rejects_uncombined_image_batch() -> None:
    policy = RecedingHorizonStepPolicy(hidden_dim=8)
    state = AdaptiveSamplerState(
        current_sigma=torch.tensor(0.5),
        previous_sigma=None,
        previous_denoised=None,
        remaining_steps=2,
    )
    with pytest.raises(ValueError, match="one generated image"):
        policy(state, torch.zeros(2, 16, 4, 4))


def test_reference_schedule_must_obey_declared_constraints() -> None:
    schedule = torch.tensor([1.0, 0.9, 0.1, 0.0])
    constraints = StepConstraints(
        min_step_size=0.2,
        max_step_size=1.0,
        min_step_ratio=0.5,
        max_step_ratio=2.0,
    )
    with pytest.raises(ValueError, match="absolute step-size"):
        FixedBudgetSigmaController(schedule, constraints=constraints)


def test_trainable_zero_action_keeps_a_schedule_gradient() -> None:
    class TrainableZeroPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action = torch.nn.Parameter(torch.zeros(()))

        def forward(self, state, guided_denoised):
            del state, guided_denoised
            return self.action

    policy = TrainableZeroPolicy()
    schedule = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float64)
    controller = FixedBudgetSigmaController(schedule, policy=policy)
    state = AdaptiveSamplerState(
        current_sigma=schedule[0],
        previous_sigma=None,
        previous_denoised=None,
        remaining_steps=controller.nfe_budget,
    )
    next_sigma, _, _ = controller.choose_next_sigma(state, torch.zeros(1))
    next_sigma.backward()
    assert policy.action.grad is not None
    assert policy.action.grad.abs().item() > 0
