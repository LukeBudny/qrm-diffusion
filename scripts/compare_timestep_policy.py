from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from qrm_diffusion.agents import (
    create_policy,
    load_agent_config,
    load_controller_checkpoint,
)
from qrm_diffusion.agents.reward import create_reward
from qrm_diffusion.agents.evaluation import evaluate_joint_gate
from qrm_diffusion.backends import create_backend
from qrm_diffusion.backends.base import GenerationRequest
from qrm_diffusion.config import load_config
from qrm_diffusion.memory import apply_cuda_memory_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-NFE fixed versus learned schedule comparison")
    parser.add_argument("--config", default="configs/models/sd35-medium.toml")
    parser.add_argument("--agent-config", default="configs/agents/sd35-qrm-timestep.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", default="outputs/timestep-policy-comparison")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    return parser.parse_args()


def _request(config, prompt, output_path, seed, sampler, trace=None):
    extra = dict(config.generation.extra, sampler=sampler)
    if trace is not None:
        extra["trajectory_trace"] = trace
    return GenerationRequest(
        prompt=prompt,
        negative_prompt=config.generation.negative_prompt,
        output_path=output_path,
        width=config.generation.width,
        height=config.generation.height,
        steps=config.generation.steps,
        guidance_scale=config.generation.guidance_scale,
        seed=seed,
        extra=extra,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = replace(
        config,
        generation=replace(
            config.generation,
            steps=config.generation.steps if args.steps is None else args.steps,
            width=config.generation.width if args.width is None else args.width,
            height=config.generation.height if args.height is None else args.height,
        ),
    )
    agent = load_agent_config(args.agent_config)
    apply_cuda_memory_policy(config.memory)
    backend = create_backend(config.model.backend, replace(
        config, controller=replace(config.controller, enabled=False)
    ))
    backend.load()
    policy = create_policy(agent, device=f"cuda:{config.memory.device}")
    load_controller_checkpoint(args.checkpoint, policy=policy, map_location=f"cuda:{config.memory.device}")
    policy.eval()
    reward = create_reward(agent.reward)
    prompts = [
        line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    output_dir = config.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = []
    for index, prompt in enumerate(prompts):
        seed = config.generation.seed + index
        fixed_path = output_dir / f"{index:05d}-fixed.png"
        policy_path = output_dir / f"{index:05d}-policy.png"
        backend.set_controller(agent, None)
        reference_sampler = (
            "euler" if agent.sampler == "adaptive_euler" else "dpmpp_2m"
        )
        backend.generate(
            _request(config, prompt, fixed_path, seed, reference_sampler)
        )
        trace = []
        backend.set_controller(agent, policy)
        backend.generate(_request(config, prompt, policy_path, seed, agent.sampler, trace))
        if len(trace) != config.generation.steps:
            raise RuntimeError("Candidate and reference do not have identical NFE budgets")
        delta = reward.relative(policy_path, fixed_path, prompt)
        deltas.append(delta)
        print(f"prompt={index} nfe={len(trace)} reward_delta={delta:+.6f}")
    result = evaluate_joint_gate(deltas, agent.evaluation)
    report = {
        "passed": result.passed,
        "prompt_count": result.prompt_count,
        "mean_reward_delta": result.mean_reward_delta,
        "positive_fraction": result.positive_fraction,
        "nfe": config.generation.steps,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"mean_reward_delta={result.mean_reward_delta:+.6f} "
        f"positive_fraction={result.positive_fraction:.3f} "
        f"prompts={result.prompt_count} nfe={config.generation.steps} "
        f"joint_gate={'PASS' if result.passed else 'FAIL'}"
    )
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
