from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .state import AdaptiveSamplerState


def policy_features(
    state: AdaptiveSamplerState, guided_denoised: torch.Tensor
) -> torch.Tensor:
    """Build the shared six-value observation used by actor and critic."""

    latent = guided_denoised.detach().float()
    mean = latent.mean()
    std = latent.std(unbiased=False)
    rms = latent.square().mean().sqrt()
    sigma = state.current_sigma.to(device=latent.device, dtype=torch.float32)
    solver_time = -torch.log(torch.clamp(sigma, min=1e-12))
    total_steps = state.step_index + state.remaining_steps
    remaining_fraction = latent.new_tensor(state.remaining_steps / total_steps)
    previous_h = (
        latent.new_zeros(())
        if state.previous_step_size is None
        else state.previous_step_size.to(device=latent.device, dtype=torch.float32)
    )
    return torch.stack((mean, std, rms, solver_time, remaining_fraction, previous_h))


class StepPolicy(Protocol):
    def __call__(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> torch.Tensor | float: ...


class ZeroStepPolicy:
    """Zero-initialized policy used to establish fixed-schedule parity."""

    def __call__(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> float:
        del state, guided_denoised
        return 0.0


class RecedingHorizonStepPolicy(nn.Module):
    """Small zero-initialized head for the next solver-time action.

    The initial policy is exactly the fixed reference schedule. It consumes the
    CFG-guided predicted-clean latent, never the concatenated conditional and
    unconditional branch batch.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.backbone = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> torch.Tensor:
        if guided_denoised.shape[0] != 1:
            raise ValueError(
                "The learned timestep policy requires one generated image; "
                "CFG branches must be combined before calling it"
            )
        return self.forward_features(policy_features(state, guided_denoised))

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 6:
            raise ValueError("Policy observations must contain six features")
        unbatched = features.ndim == 1
        if unbatched:
            features = features.unsqueeze(0)
        result = self.output(self.backbone(features.float())).squeeze(-1)
        return result.squeeze(0) if unbatched else result


class ExplorationStepPolicy:
    """Sample rollout actions while retaining a deterministic inference actor."""

    def __init__(self, policy: RecedingHorizonStepPolicy, std: float) -> None:
        if std <= 0:
            raise ValueError("Exploration standard deviation must be positive")
        self.policy = policy
        self.std = float(std)

    def __call__(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> torch.Tensor:
        mean = self.policy(state, guided_denoised)
        return (mean + torch.randn_like(mean) * self.std).detach()


@dataclass(frozen=True)
class StepConstraints:
    # Defaults are intentionally broad enough to admit the existing SD3.5
    # schedule unchanged while still making every adaptive action finite.
    min_step_size: float = 1e-8
    max_step_size: float = 1e6
    min_step_ratio: float = 1e-3
    max_step_ratio: float = 1e3

    def validate(self) -> None:
        if self.min_step_size <= 0 or self.max_step_size < self.min_step_size:
            raise ValueError("Invalid absolute step-size bounds")
        if self.min_step_ratio <= 0 or self.max_step_ratio < self.min_step_ratio:
            raise ValueError("Invalid consecutive step-size ratio bounds")


class FixedBudgetSigmaController:
    """Choose one next sigma while preserving a fixed number of evaluations.

    The supplied reference schedule is the nominal receding-horizon plan. A
    zero action follows it exactly. Non-zero actions perturb the next solver
    time step by ``exp(beta * tanh(action))`` and are projected onto absolute
    and consecutive-step bounds. The final transition always lands at zero.
    """

    def __init__(
        self,
        reference_sigmas: torch.Tensor,
        *,
        beta: float = 0.35,
        constraints: StepConstraints | None = None,
        policy: StepPolicy | None = None,
        epsilon: float = 1e-12,
    ) -> None:
        sigmas = torch.as_tensor(reference_sigmas).detach().flatten()
        if sigmas.numel() < 2:
            raise ValueError("reference_sigmas must contain at least one transition")
        if not torch.isfinite(sigmas[:-1]).all() or (sigmas[:-1] <= 0).any():
            raise ValueError("All non-terminal reference sigmas must be finite and positive")
        if sigmas[-1].item() != 0.0:
            raise ValueError("The reference schedule must terminate at sigma=0")
        if not torch.all(sigmas[:-1] > sigmas[1:]):
            raise ValueError("reference_sigmas must be strictly decreasing")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.reference_sigmas = sigmas
        self.beta = float(beta)
        self.constraints = constraints or StepConstraints()
        self.constraints.validate()
        positive_times = -torch.log(sigmas[:-1].detach().double().cpu())
        if sigmas.numel() > 2:
            positive_next_times = -torch.log(sigmas[1:-1].detach().double().cpu())
            reference_steps = positive_next_times - positive_times[:-1]
            if (
                (reference_steps < self.constraints.min_step_size).any()
                or (reference_steps > self.constraints.max_step_size).any()
            ):
                raise ValueError("Reference schedule violates absolute step-size bounds")
            if reference_steps.numel() > 1:
                ratios = reference_steps[1:] / reference_steps[:-1]
                if (
                    (ratios < self.constraints.min_step_ratio).any()
                    or (ratios > self.constraints.max_step_ratio).any()
                ):
                    raise ValueError(
                        "Reference schedule violates consecutive step-size ratio bounds"
                    )
        self.policy = policy or ZeroStepPolicy()
        self.epsilon = float(epsilon)

    @property
    def nfe_budget(self) -> int:
        return self.reference_sigmas.numel() - 1

    @staticmethod
    def _solver_time(sigma: torch.Tensor, epsilon: float) -> torch.Tensor:
        return -torch.log(torch.clamp(sigma, min=epsilon))

    def _scalar_action(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> torch.Tensor:
        features = policy_features(state, guided_denoised)
        state.metadata["policy_features"] = tuple(
            float(value) for value in features.detach().cpu()
        )
        action = torch.as_tensor(
            self.policy(state, guided_denoised),
            device=state.current_sigma.device,
            dtype=torch.float64,
        ).flatten()
        if action.numel() != 1 or not torch.isfinite(action).all():
            raise ValueError("The timestep policy must return one finite scalar per image")
        return action[0]

    def choose_next_sigma(
        self, state: AdaptiveSamplerState, guided_denoised: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state.validate()
        expected_remaining = self.nfe_budget - state.step_index
        if state.remaining_steps != expected_remaining:
            raise ValueError(
                f"remaining_steps={state.remaining_steps} does not match fixed budget "
                f"state ({expected_remaining})"
            )

        action = self._scalar_action(state, guided_denoised)
        if state.remaining_steps == 1:
            zero = state.current_sigma.new_zeros(())
            return zero, action, state.current_sigma.new_full((), float("inf"))

        reference_current = self.reference_sigmas[state.step_index].to(
            device=state.current_sigma.device, dtype=state.current_sigma.dtype
        )
        reference_next_exact = self.reference_sigmas[state.step_index + 1].to(
            device=state.current_sigma.device, dtype=state.current_sigma.dtype
        )
        # Exact zero is the compatibility contract: when the trajectory is on
        # the nominal plan, do not round-trip sigma through log/exp.
        if (
            not action.requires_grad
            and action.item() == 0.0
            and torch.equal(state.current_sigma, reference_current)
        ):
            exact_h = self._solver_time(
                reference_next_exact, self.epsilon
            ) - self._solver_time(state.current_sigma, self.epsilon)
            return reference_next_exact, action, exact_h

        current = state.current_sigma.to(dtype=torch.float64)
        reference_next = self.reference_sigmas[state.step_index + 1].to(
            device=current.device, dtype=current.dtype
        )
        current_tau = self._solver_time(current, self.epsilon)
        reference_tau = self._solver_time(reference_next, self.epsilon)
        nominal_h = reference_tau - current_tau

        # If previous adaptive decisions moved past the nominal point, retain a
        # strictly positive receding-horizon step instead of reversing sigma.
        nominal_h = torch.clamp(nominal_h, min=self.constraints.min_step_size)
        h = nominal_h * torch.exp(self.beta * torch.tanh(action))
        lower = current.new_tensor(self.constraints.min_step_size)
        upper = current.new_tensor(self.constraints.max_step_size)
        if state.previous_step_size is not None and torch.isfinite(state.previous_step_size):
            previous_h = state.previous_step_size.to(device=current.device, dtype=current.dtype)
            lower = torch.maximum(lower, previous_h * self.constraints.min_step_ratio)
            upper = torch.minimum(upper, previous_h * self.constraints.max_step_ratio)
        if lower > upper:
            raise ValueError("Step constraints have no feasible intersection")
        h = torch.clamp(h, min=lower, max=upper)
        next_sigma = torch.exp(-(current_tau + h)).to(state.current_sigma.dtype)

        if not torch.isfinite(next_sigma) or not (0 < next_sigma < state.current_sigma):
            raise RuntimeError("Controller failed to produce a finite, decreasing sigma")
        actual_h = self._solver_time(
            next_sigma, self.epsilon
        ) - self._solver_time(state.current_sigma, self.epsilon)
        return next_sigma, action, actual_h
