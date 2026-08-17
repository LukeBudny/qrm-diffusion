from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from qrm_diffusion.backends import create_backend
from qrm_diffusion.config import load_config
from qrm_diffusion.memory import apply_cuda_memory_policy, cuda_memory_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed and zero-action adaptive native SD3.5 sampling"
    )
    parser.add_argument("--config", default="configs/models/sd35-medium.toml")
    parser.add_argument("--output-dir", default="outputs/sampler-parity")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-pixel-delta", type=int, default=0)
    parser.add_argument("--max-latent-delta", type=float, default=0.0)
    parser.add_argument("--qrm-checkpoint")
    parser.add_argument("--qrm-type", default="QRMModulatorLatentV6")
    parser.add_argument("--qrm-start-step", type=int)
    parser.add_argument("--qrm-end-step", type=int)
    parser.add_argument(
        "--candidate-sampler",
        default="adaptive_dpmpp_2m",
        choices=("adaptive_dpmpp_2m", "adaptive_euler", "dpmpp_2m", "euler"),
        help="Use dpmpp_2m as a determinism control run",
    )
    parser.add_argument(
        "--reference-sampler",
        default="dpmpp_2m",
        choices=("dpmpp_2m", "euler"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.qrm_checkpoint:
        config = replace(
            config,
            qrm=replace(
                config.qrm,
                enabled=True,
                checkpoint=args.qrm_checkpoint,
                qrm_type=args.qrm_type,
                start_step=args.qrm_start_step,
                end_step=args.qrm_end_step,
            ),
        )
    generation = replace(
        config.generation,
        prompt=args.prompt or config.generation.prompt,
        seed=config.generation.seed if args.seed is None else args.seed,
        steps=config.generation.steps if args.steps is None else args.steps,
        width=config.generation.width if args.width is None else args.width,
        height=config.generation.height if args.height is None else args.height,
    )
    config = replace(config, generation=generation)

    # This must remain before backend creation/model loading.
    memory = apply_cuda_memory_policy(config.memory)
    print(
        f"cuda={memory.device_name} limit={memory.limit_gib:.2f} GiB "
        f"allocator_fraction={memory.allocator_fraction:.4f}"
    )
    backend = create_backend(config.model.backend, config)
    backend.load()
    print(
        f"qrm_enabled={config.qrm.enabled} qrm_start_step={backend.qrm_start_step} "
        f"qrm_end_step={backend.qrm_end_step}"
    )

    output_dir = config.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = output_dir / f"fixed-{args.reference_sampler}.png"
    adaptive_path = output_dir / f"candidate-{args.candidate_sampler}.png"
    fixed_trace: list[object] = []
    trace: list[object] = []
    inferencer = getattr(backend, "inferencer", None)
    if inferencer is None:
        raise RuntimeError("Parity validation currently requires the sd35_native backend")

    # Encode once and reuse the exact same initial latent/conditioning. This
    # isolates the sampler rather than comparing two end-to-end generations.
    initial = inferencer.get_empty_latent(
        1, generation.width, generation.height, generation.seed, "cpu"
    )
    conditioning = inferencer.get_cond(generation.prompt)
    negative = inferencer.get_cond(generation.negative_prompt or "")

    common = dict(
        seed=generation.seed,
        conditioning=conditioning,
        neg_cond=negative,
        steps=generation.steps,
        cfg_scale=generation.guidance_scale,
        controlnet_cond=None,
        denoise=1.0,
        skip_layer_config={},
        prompt=generation.prompt,
        use_qrm=config.qrm.enabled,
        qrm_start_step=backend.qrm_start_step,
        qrm_end_step=backend.qrm_end_step,
    )
    fixed_latent, _ = inferencer.do_sampling(
        initial.clone(),
        sampler=args.reference_sampler,
        trajectory_trace=fixed_trace,
        **common,
    )
    candidate_latent, _ = inferencer.do_sampling(
        initial.clone(),
        sampler=args.candidate_sampler,
        trajectory_trace=trace,
        **common,
    )
    inferencer.vae_decode(fixed_latent).save(fixed_path)
    inferencer.vae_decode(candidate_latent).save(adaptive_path)

    fixed = np.asarray(Image.open(fixed_path), dtype=np.int16)
    adaptive = np.asarray(Image.open(adaptive_path), dtype=np.int16)
    difference = np.abs(fixed - adaptive)
    max_delta = int(difference.max())
    mean_delta = float(difference.mean())
    changed = int(np.count_nonzero(difference))
    latent_difference = (fixed_latent - candidate_latent).detach().float().abs()
    max_latent_delta = float(latent_difference.max().cpu())
    mean_latent_delta = float(latent_difference.mean().cpu())
    expected_nfe = generation.steps

    print(
        f"parity max_pixel_delta={max_delta} mean_pixel_delta={mean_delta:.8f} "
        f"changed_channels={changed} trace_steps={len(trace)} expected_nfe={expected_nfe}"
    )
    print(
        f"latent max_delta={max_latent_delta:.10g} "
        f"mean_delta={mean_latent_delta:.10g}"
    )
    if trace:
        active_qrm_steps = [
            entry.step_index for entry in trace if entry.modulation_norm is not None
        ]
        print(f"trace qrm_active_steps={active_qrm_steps}")
        first_trace_difference = next(
            (
                (fixed.step_index, fixed.denoised_norm, adaptive.denoised_norm,
                 fixed.sample_norm, adaptive.sample_norm)
                for fixed, adaptive in zip(fixed_trace, trace)
                if fixed.denoised_norm != adaptive.denoised_norm
                or fixed.sample_norm != adaptive.sample_norm
            ),
            None,
        )
        print(f"trace first_norm_difference={first_trace_difference}")
    print(f"cuda_memory {cuda_memory_summary(config.memory.device)}")
    if args.candidate_sampler.startswith("adaptive") and len(trace) != expected_nfe:
        print("FAIL: trajectory length does not match the requested NFE budget")
        return 2
    if max_delta > args.max_pixel_delta:
        print(f"FAIL: max pixel delta exceeds {args.max_pixel_delta}")
        return 1
    if max_latent_delta > args.max_latent_delta:
        print(f"FAIL: max latent delta exceeds {args.max_latent_delta}")
        return 3
    print(
        f"PASS: zero-action {args.candidate_sampler} reproduces "
        f"fixed {args.reference_sampler} output"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
