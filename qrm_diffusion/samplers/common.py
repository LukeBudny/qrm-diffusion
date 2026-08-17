from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from qrm_diffusion.agents.controller import FixedBudgetSigmaController
from qrm_diffusion.agents.state import AdaptiveSamplerState, TrajectoryEntry


Denoiser = Callable[..., torch.Tensor | tuple[Any, ...] | list[Any]]


@dataclass(frozen=True)
class AdaptiveSamplerResult:
    sample: torch.Tensor
    state: AdaptiveSamplerState


def extract_denoised(output: Any) -> torch.Tensor:
    """Normalize legacy SD3/QRM and plain denoiser return conventions."""

    value = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError("Denoiser output must contain a tensor")
    return value


def capture_model_diagnostics(state: AdaptiveSamplerState, output: Any) -> None:
    """Capture optional diagnostics from the legacy native SD3/QRM return tuple."""

    if not isinstance(output, (tuple, list)) or len(output) < 4:
        return
    modulation_delta = output[3]
    if torch.is_tensor(modulation_delta):
        state.metadata["modulation_norm"] = float(
            torch.linalg.vector_norm(modulation_delta.detach().float()).cpu()
        )


def initialize_state(x: torch.Tensor, controller: FixedBudgetSigmaController) -> AdaptiveSamplerState:
    # Solver time/sigma arithmetic must retain the schedule dtype. Native SD3
    # latents are FP16, while the established sampler computes its DPM++
    # history from FP32 sigmas; downcasting here breaks zero-action parity.
    sigma = controller.reference_sigmas[0].to(device=x.device)
    return AdaptiveSamplerState(
        current_sigma=sigma,
        previous_sigma=None,
        previous_denoised=None,
        remaining_steps=controller.nfe_budget,
    )


def record_transition(
    state: AdaptiveSamplerState,
    *,
    next_sigma: torch.Tensor,
    action: torch.Tensor,
    step_size: torch.Tensor,
    denoised: torch.Tensor,
    sample: torch.Tensor,
) -> None:
    quality = state.metadata.pop("quality", None)
    modulation_norm = state.metadata.pop("modulation_norm", None)
    observation = state.metadata.pop("policy_features", None)
    state.trajectory_trace.append(
        TrajectoryEntry(
            step_index=state.step_index,
            sigma=float(state.current_sigma.detach().cpu()),
            next_sigma=float(next_sigma.detach().cpu()),
            step_size=None if not torch.isfinite(step_size) else float(step_size.detach().cpu()),
            action=float(action.detach().cpu()),
            quality=None if quality is None else tuple(float(v) for v in quality),
            modulation_norm=None if modulation_norm is None else float(modulation_norm),
            denoised_norm=float(torch.linalg.vector_norm(denoised.detach().float()).cpu()),
            sample_norm=float(torch.linalg.vector_norm(sample.detach().float()).cpu()),
            policy_features=observation,
        )
    )


def advance_state(
    state: AdaptiveSamplerState,
    *,
    denoised: torch.Tensor,
    next_sigma: torch.Tensor,
    step_size: torch.Tensor,
) -> None:
    state.previous_sigma = state.current_sigma
    state.previous_denoised = denoised
    state.current_sigma = next_sigma
    if torch.isfinite(step_size):
        state.previous_step_size = step_size
    state.remaining_steps -= 1
    state.step_index += 1
