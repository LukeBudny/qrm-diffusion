from __future__ import annotations

from typing import Any

import torch

from qrm_diffusion.agents.controller import FixedBudgetSigmaController

from .common import (
    AdaptiveSamplerResult,
    Denoiser,
    advance_state,
    capture_model_diagnostics,
    extract_denoised,
    initialize_state,
    record_transition,
)


@torch.no_grad()
def sample_adaptive_dpmpp_2m(
    model: Denoiser,
    x: torch.Tensor,
    controller: FixedBudgetSigmaController,
    *,
    extra_args: dict[str, Any] | None = None,
) -> AdaptiveSamplerResult:
    """DPM-Solver++(2M) using the actual controller-selected step history."""

    args = {} if extra_args is None else dict(extra_args)
    state = initialize_state(x, controller)
    s_in = x.new_ones([x.shape[0]])

    while state.remaining_steps:
        sigma = state.current_sigma
        output = model(x, sigma * s_in, **args)
        capture_model_diagnostics(state, output)
        guided_denoised = extract_denoised(output)
        next_sigma, action, step_size = controller.choose_next_sigma(state, guided_denoised)

        if next_sigma.item() == 0.0:
            x = guided_denoised
        else:
            t = -torch.log(sigma)
            t_next = -torch.log(next_sigma)
            h = t_next - t
            if state.previous_denoised is None:
                denoised_d = guided_denoised
            else:
                if state.previous_step_size is None:
                    raise RuntimeError("DPM++ history is missing its previous step size")
                ratio = state.previous_step_size / h
                denoised_d = (
                    (1 + 1 / (2 * ratio)) * guided_denoised
                    - (1 / (2 * ratio)) * state.previous_denoised
                )
            # Preserve the legacy native operation sequence exactly. Although
            # next_sigma / sigma is algebraically equivalent, reconstructing
            # both values from solver time rounds differently in FP32.
            sigma_from_t = torch.exp(-t)
            next_sigma_from_t = torch.exp(-t_next)
            x = (
                (next_sigma_from_t / sigma_from_t) * x
                - torch.expm1(-h) * denoised_d
            )

        record_transition(
            state,
            next_sigma=next_sigma,
            action=action,
            step_size=step_size,
            denoised=guided_denoised,
            sample=x,
        )
        advance_state(
            state,
            denoised=guided_denoised,
            next_sigma=next_sigma,
            step_size=step_size,
        )

    return AdaptiveSamplerResult(sample=x, state=state)
