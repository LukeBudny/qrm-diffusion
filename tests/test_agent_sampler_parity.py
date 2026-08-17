from __future__ import annotations

import pytest
import torch

from qrm_diffusion.agents.controller import FixedBudgetSigmaController
from qrm_diffusion.samplers import sample_adaptive_dpmpp_2m, sample_adaptive_euler
from sd3_impls import sample_adaptive_dpmpp_2m as sample_native_adaptive_dpmpp_2m
from sd3_impls import sample_adaptive_euler as sample_native_adaptive_euler
from sd3_impls import sample_euler as sample_native_fixed_euler


def _denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.view(-1, *([1] * (x.ndim - 1)))
    return 0.2 * x + 0.1 * sigma


def _legacy_euler(x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = _denoiser(x, sigma * s_in)
        derivative = (x - denoised) / sigma
        x = x + derivative * (sigmas[index + 1] - sigma)
    return x


def _legacy_dpmpp_2m(x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    for index in range(len(sigmas) - 1):
        denoised = _denoiser(x, sigmas[index] * s_in)
        if sigmas[index + 1].item() == 0.0:
            x = denoised
        else:
            t = -torch.log(sigmas[index])
            t_next = -torch.log(sigmas[index + 1])
            h = t_next - t
            if old_denoised is None:
                denoised_d = denoised
            else:
                h_last = t - (-torch.log(sigmas[index - 1]))
                ratio = h_last / h
                denoised_d = (
                    (1 + 1 / (2 * ratio)) * denoised
                    - (1 / (2 * ratio)) * old_denoised
                )
            sigma_from_t = torch.exp(-t)
            next_sigma_from_t = torch.exp(-t_next)
            x = (
                (next_sigma_from_t / sigma_from_t) * x
                - torch.expm1(-h) * denoised_d
            )
        old_denoised = denoised
    return x


def _schedule() -> torch.Tensor:
    return torch.tensor([1.0, 0.6, 0.3, 0.12, 0.0], dtype=torch.float64)


def test_zero_action_euler_matches_fixed_schedule() -> None:
    sigmas = _schedule()
    initial = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(1, 3, 2, 2)
    expected = _legacy_euler(initial.clone(), sigmas)
    result = sample_adaptive_euler(
        _denoiser, initial.clone(), FixedBudgetSigmaController(sigmas)
    )

    torch.testing.assert_close(result.sample, expected, rtol=1e-12, atol=1e-12)
    assert len(result.state.trajectory_trace) == len(sigmas) - 1
    assert result.state.trajectory_trace[-1].next_sigma == 0.0


def test_zero_action_dpmpp_2m_matches_fixed_schedule_with_unequal_steps() -> None:
    sigmas = _schedule()
    initial = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(1, 3, 2, 2)
    expected = _legacy_dpmpp_2m(initial.clone(), sigmas)
    result = sample_adaptive_dpmpp_2m(
        _denoiser, initial.clone(), FixedBudgetSigmaController(sigmas)
    )

    torch.testing.assert_close(result.sample, expected, rtol=1e-12, atol=1e-12)
    assert result.state.step_index == len(sigmas) - 1
    assert result.state.remaining_steps == 0


def test_fp16_latents_keep_fp32_schedule_arithmetic() -> None:
    sigmas = _schedule().float()
    initial = torch.linspace(-1, 1, 12, dtype=torch.float16).reshape(1, 3, 2, 2)
    expected = _legacy_dpmpp_2m(initial.clone(), sigmas)
    result = sample_adaptive_dpmpp_2m(
        _denoiser, initial.clone(), FixedBudgetSigmaController(sigmas)
    )

    assert result.state.previous_sigma is not None
    assert result.state.previous_sigma.dtype == torch.float32
    torch.testing.assert_close(result.sample, expected, rtol=0, atol=0)


class _BatchCheckingController(FixedBudgetSigmaController):
    def choose_next_sigma(self, state, guided_denoised):
        assert guided_denoised.shape[0] == 2
        return super().choose_next_sigma(state, guided_denoised)


def test_controller_observes_guided_image_batch_not_cfg_branch_batch() -> None:
    sigmas = _schedule()
    initial = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
    controller = _BatchCheckingController(sigmas)
    result = sample_adaptive_euler(_denoiser, initial, controller)
    assert result.state.step_index == controller.nfe_budget


def test_legacy_modulation_norm_is_recorded() -> None:
    sigmas = _schedule()
    initial = torch.zeros(1, 1, 2, 2, dtype=torch.float64)

    def legacy_model(x, sigma):
        denoised = _denoiser(x, sigma)
        modulation = torch.tensor([[3.0, 4.0]], dtype=x.dtype)
        return denoised, None, None, modulation, [], []

    result = sample_adaptive_euler(
        legacy_model, initial, FixedBudgetSigmaController(sigmas)
    )
    assert all(entry.modulation_norm == 5.0 for entry in result.state.trajectory_trace)


def test_native_wrapper_uses_sigma_region_and_exports_trace() -> None:
    sigmas = _schedule()
    initial = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    qrm_flags: list[bool] = []
    observed_t_raw: list[float] = []

    def native_model(x, sigma, *, t_raw, use_qrm, **kwargs):
        del kwargs
        qrm_flags.append(bool(use_qrm))
        observed_t_raw.append(float(t_raw[0]))
        denoised = _denoiser(x, sigma)
        modulation = torch.ones(1, 2, dtype=x.dtype) if use_qrm else None
        return denoised, None, None, modulation, [], []

    trace = []
    sample = sample_native_adaptive_dpmpp_2m(
        native_model,
        initial,
        sigmas,
        use_qrm=True,
        extra_args={
            "qrm_start_step": 1,
            "qrm_end_step": 2,
            "trajectory_trace": trace,
        },
    )

    assert sample.shape == initial.shape
    assert qrm_flags == [False, True, True, False]
    assert observed_t_raw == pytest.approx((-torch.log(sigmas[:-1])).tolist())
    assert len(trace) == len(sigmas) - 1
    assert [entry.modulation_norm is not None for entry in trace] == qrm_flags


def test_native_sigma_region_keeps_fp16_rounded_boundaries_inclusive() -> None:
    sigmas = _schedule().float()
    initial = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
    qrm_flags: list[bool] = []

    def native_model(x, sigma, *, t_raw, use_qrm, **kwargs):
        del t_raw, kwargs
        qrm_flags.append(bool(use_qrm))
        return _denoiser(x, sigma), None, None, None, [], []

    sample_native_adaptive_euler(
        native_model,
        initial,
        sigmas,
        use_qrm=True,
        extra_args={"qrm_start_step": 1, "qrm_end_step": 2},
    )
    assert qrm_flags == [False, True, True, False]


def test_native_zero_action_euler_matches_fixed_euler() -> None:
    sigmas = _schedule().float()
    initial = torch.linspace(-1, 1, 8, dtype=torch.float16).reshape(1, 1, 2, 4)

    def native_model(x, sigma, *, t_raw, use_qrm, **kwargs):
        del t_raw, use_qrm, kwargs
        return _denoiser(x, sigma), None, None, None, [], []

    fixed = sample_native_fixed_euler(native_model, initial.clone(), sigmas)
    adaptive = sample_native_adaptive_euler(
        native_model, initial.clone(), sigmas
    )
    torch.testing.assert_close(adaptive, fixed, rtol=0, atol=0)
