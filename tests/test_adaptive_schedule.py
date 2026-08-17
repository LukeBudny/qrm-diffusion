from __future__ import annotations

import pytest
import torch

from qrm_diffusion.agents.controller import (
    FixedBudgetSigmaController,
    StepConstraints,
)
from qrm_diffusion.samplers.common import initialize_state


def _schedule() -> torch.Tensor:
    return torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0], dtype=torch.float64)


def test_zero_action_reproduces_reference_schedule_and_exact_budget() -> None:
    controller = FixedBudgetSigmaController(_schedule())
    state = initialize_state(torch.zeros(1), controller)
    denoised = torch.zeros(1)
    selected = [float(state.current_sigma)]

    while state.remaining_steps:
        next_sigma, action, step_size = controller.choose_next_sigma(state, denoised)
        selected.append(float(next_sigma))
        state.previous_sigma = state.current_sigma
        state.current_sigma = next_sigma
        if torch.isfinite(step_size):
            state.previous_step_size = step_size
        state.remaining_steps -= 1
        state.step_index += 1

    assert selected == pytest.approx(_schedule().tolist(), abs=1e-12)
    assert state.step_index == controller.nfe_budget
    assert state.remaining_steps == 0


class _ExtremePolicy:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, state, guided_denoised):
        del state, guided_denoised
        return self.value


@pytest.mark.parametrize("action", [-1e30, 1e30])
def test_extreme_actions_are_finite_monotonic_and_bounded(action: float) -> None:
    constraints = StepConstraints(
        min_step_size=0.05,
        max_step_size=1.0,
        min_step_ratio=0.5,
        max_step_ratio=2.0,
    )
    controller = FixedBudgetSigmaController(
        _schedule(), constraints=constraints, policy=_ExtremePolicy(action)
    )
    state = initialize_state(torch.zeros(1), controller)
    denoised = torch.zeros(1)
    positive_sigmas = [float(state.current_sigma)]
    finite_steps: list[float] = []

    while state.remaining_steps:
        next_sigma, _, step_size = controller.choose_next_sigma(state, denoised)
        if next_sigma.item() > 0:
            positive_sigmas.append(float(next_sigma))
        if torch.isfinite(step_size):
            finite_steps.append(float(step_size))
        state.previous_sigma = state.current_sigma
        state.current_sigma = next_sigma
        if torch.isfinite(step_size):
            state.previous_step_size = step_size
        state.remaining_steps -= 1
        state.step_index += 1

    assert all(a > b for a, b in zip(positive_sigmas, positive_sigmas[1:]))
    assert state.current_sigma.item() == 0.0
    assert all(constraints.min_step_size <= h <= constraints.max_step_size for h in finite_steps)
    assert all(
        constraints.min_step_ratio <= current / previous <= constraints.max_step_ratio
        for previous, current in zip(finite_steps, finite_steps[1:])
    )


def test_rejects_non_monotonic_or_nonterminal_schedule() -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        FixedBudgetSigmaController(torch.tensor([1.0, 0.5, 0.5, 0.0]))
    with pytest.raises(ValueError, match="terminate"):
        FixedBudgetSigmaController(torch.tensor([1.0, 0.5, 0.1]))
