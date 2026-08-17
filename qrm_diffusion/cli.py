from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .backends import available_backends, create_backend
from .backends.base import GenerationRequest
from .config import PROJECT_MAX_VRAM_GIB, load_config
from .memory import apply_cuda_memory_policy, cuda_memory_summary


DEFAULT_CONFIG = "configs/models/sd35-medium.toml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Backend-neutral QRM diffusion inference")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", action="append", help="Prompt; may be supplied more than once")
    parser.add_argument("--prompts-file", help="UTF-8 file containing one prompt per line")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-vram-gib", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without loading CUDA")
    parser.add_argument("--list-backends", action="store_true")
    return parser.parse_args(argv)


def _prompts(args, config) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        path = Path(args.prompts_file).expanduser()
        if not path.is_absolute():
            path = config.root / path
        prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    prompts = [prompt for prompt in prompts if prompt]
    return prompts or [config.generation.prompt]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_backends:
        print("\n".join(available_backends()))
        return 0

    config = load_config(args.config)
    if args.max_vram_gib is not None:
        if args.max_vram_gib > PROJECT_MAX_VRAM_GIB:
            raise ValueError(
                f"--max-vram-gib cannot exceed the project limit of "
                f"{PROJECT_MAX_VRAM_GIB:.0f} GiB"
            )
        config = replace(config, memory=replace(config.memory, max_vram_gib=args.max_vram_gib))

    prompts = _prompts(args, config)
    output_dir = config.resolve_path(args.output_dir or config.generation.output_dir)
    print(f"config={config.source}")
    print(f"backend={config.model.backend} model={config.model.name}")
    print(f"vram_budget={config.memory.max_vram_gib:.2f} GiB device={config.memory.device}")
    print(f"prompts={len(prompts)} output={output_dir}")
    if args.dry_run:
        print("configuration valid; CUDA/model loading skipped")
        return 0

    memory = apply_cuda_memory_policy(config.memory)
    print(
        f"cuda={memory.device_name} total={memory.total_gib:.2f} GiB "
        f"limit={memory.limit_gib:.2f} GiB allocator_fraction={memory.allocator_fraction:.4f}"
    )
    backend = create_backend(config.model.backend, config)
    backend.load()

    for index, prompt in enumerate(prompts):
        output_path = output_dir / config.model.name / f"{index:04d}.png"
        request = GenerationRequest(
            prompt=prompt,
            negative_prompt=config.generation.negative_prompt,
            output_path=output_path,
            width=config.generation.width,
            height=config.generation.height,
            steps=config.generation.steps,
            guidance_scale=config.generation.guidance_scale,
            seed=config.generation.seed + index,
            extra=config.generation.extra,
        )
        print(f"[{index + 1}/{len(prompts)}] {output_path}")
        backend.generate(request)
        print(f"cuda_memory {cuda_memory_summary(config.memory.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
