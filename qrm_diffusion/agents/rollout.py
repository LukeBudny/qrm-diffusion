from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch


def controller_options(agent_config) -> dict[str, float]:
    schedule = agent_config.schedule
    return {
        "beta": schedule.beta,
        "min_step_size": schedule.min_step_size,
        "max_step_size": schedule.max_step_size,
        "min_step_ratio": schedule.min_step_ratio,
        "max_step_ratio": schedule.max_step_ratio,
    }


def paired_schedule_rollout(
    backend,
    config,
    agent_config,
    *,
    prompt: str,
    seed: int,
    fixed_path: Path,
    policy_path: Path,
    policy: Any,
) -> list[Any]:
    """Render fixed and adaptive schedules from identical conditioning and noise."""

    _, traces = grouped_schedule_rollout(
        backend,
        config,
        agent_config,
        prompt=prompt,
        seed=seed,
        fixed_path=fixed_path,
        candidate_paths=[policy_path],
        policies=[policy],
    )
    return traces[0]


def grouped_schedule_rollout(
    backend,
    config,
    agent_config,
    *,
    prompt: str,
    seed: int,
    fixed_path: Path,
    candidate_paths: Sequence[Path],
    policies: Sequence[Any],
) -> tuple[list[Any], list[list[Any]]]:
    """Render one fixed reference and several policy schedules from shared inputs."""

    if len(candidate_paths) != len(policies) or not candidate_paths:
        raise ValueError("Candidate paths and policies must be equally sized and non-empty")

    inferencer = backend.inferencer
    if inferencer is None:
        raise RuntimeError("Paired rollout requires a loaded native SD3.5 backend")
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    for path in candidate_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    initial = inferencer.get_empty_latent(
        1, config.generation.width, config.generation.height, seed, "cpu"
    )
    conditioning = inferencer.get_cond(prompt)
    negative = inferencer.get_cond(config.generation.negative_prompt or "")
    common = dict(
        seed=seed,
        conditioning=conditioning,
        neg_cond=negative,
        steps=config.generation.steps,
        cfg_scale=config.generation.guidance_scale,
        controlnet_cond=None,
        denoise=1.0,
        skip_layer_config={},
        prompt=prompt,
        use_qrm=config.qrm.enabled,
        qrm_start_step=backend.qrm_start_step,
        qrm_end_step=backend.qrm_end_step,
    )
    reference_sampler = (
        "euler" if agent_config.sampler == "adaptive_euler" else "dpmpp_2m"
    )
    reference_trace: list[Any] = []
    traces: list[list[Any]] = []
    with torch.inference_mode():
        fixed_latent, _ = inferencer.do_sampling(
            initial.clone(),
            sampler=reference_sampler,
            trajectory_trace=reference_trace,
            **common,
        )
        inferencer.vae_decode(fixed_latent).save(fixed_path)
        for path, policy in zip(candidate_paths, policies):
            trace: list[Any] = []
            candidate_latent, _ = inferencer.do_sampling(
                initial.clone(),
                sampler=agent_config.sampler,
                trajectory_trace=trace,
                sigma_policy=policy,
                sigma_controller_options=controller_options(agent_config),
                **common,
            )
            inferencer.vae_decode(candidate_latent).save(path)
            if len(trace) != config.generation.steps:
                raise RuntimeError(
                    "Grouped rollout violated the identical fixed NFE budget"
                )
            traces.append(trace)
    if reference_trace and len(reference_trace) != config.generation.steps:
        raise RuntimeError("Reference rollout violated the fixed NFE budget")
    return reference_trace, traces
