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
def sample_adaptive_euler(
    model: Denoiser,
    x: torch.Tensor,
    controller: FixedBudgetSigmaController,
    *,
    extra_args: dict[str, Any] | None = None,
) -> AdaptiveSamplerResult:
    """Euler sampler that replans one shared next sigma after every CFG output."""

    args = {} if extra_args is None else dict(extra_args)
    state = initialize_state(x, controller)
    s_in = x.new_ones([x.shape[0]])

    while state.remaining_steps:
        sigma = state.current_sigma
        output = model(x, sigma * s_in, **args)
        capture_model_diagnostics(state, output)
        guided_denoised = extract_denoised(output)
        next_sigma, action, step_size = controller.choose_next_sigma(state, guided_denoised)

        sigma_broadcast = sigma[(...,) + (None,) * (x.ndim - sigma.ndim)]
        derivative = (x - guided_denoised) / sigma_broadcast
        x = x + derivative * (next_sigma - sigma)

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
