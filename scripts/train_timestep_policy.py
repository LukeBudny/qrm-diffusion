from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from qrm_diffusion.agents import (
    ExplorationStepPolicy,
    QualityCritic,
    create_policy,
    load_agent_config,
    load_controller_checkpoint,
    save_controller_checkpoint,
)
from qrm_diffusion.agents.reward import create_reward
from qrm_diffusion.agents.training import ActorCriticUpdater
from qrm_diffusion.backends import create_backend
from qrm_diffusion.backends.base import GenerationRequest
from qrm_diffusion.config import load_config
from qrm_diffusion.memory import apply_cuda_memory_policy, cuda_memory_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed-budget SD3.5 timestep policy from relative image reward"
    )
    parser.add_argument("--config", default="configs/models/sd35-medium.toml")
    parser.add_argument("--agent-config", default="configs/agents/sd35-qrm-timestep.toml")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output-dir", default="outputs/timestep-policy-training")
    parser.add_argument("--checkpoint", default="outputs/controller-checkpoints/latest.pt")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    return parser.parse_args()


def _request(config, prompt: str, path: Path, seed: int, sampler: str, trace=None):
    extra = dict(config.generation.extra)
    extra["sampler"] = sampler
    if trace is not None:
        extra["trajectory_trace"] = trace
    return GenerationRequest(
        prompt=prompt,
        negative_prompt=config.generation.negative_prompt,
        output_path=path,
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
    if config.model.backend != "sd35_native":
        raise ValueError("Timestep-policy training requires backend='sd35_native'")
    if not agent.training.quality_critic or not agent.training.timestep_policy:
        raise ValueError("Enable quality_critic and timestep_policy in the agent TOML")
    if agent.training.joint_controller:
        raise ValueError("Joint control remains gated; disable joint_controller for this stage")

    memory = apply_cuda_memory_policy(config.memory)
    print(
        f"cuda={memory.device_name} limit={memory.limit_gib:.2f} GiB "
        f"allocator_fraction={memory.allocator_fraction:.4f}"
    )
    backend = create_backend(config.model.backend, replace(
        config, controller=replace(config.controller, enabled=False)
    ))
    backend.load()
    for parameter in backend.inferencer.sd3.model.parameters():
        parameter.requires_grad_(False)
    for parameter in backend.inferencer.vae.model.parameters():
        parameter.requires_grad_(False)
    if backend.inferencer.sd3.model.qrm is not None:
        for parameter in backend.inferencer.sd3.model.qrm.parameters():
            parameter.requires_grad_(False)

    device = torch.device(f"cuda:{config.memory.device}")
    policy = create_policy(agent, device=device)
    critic = QualityCritic(agent.critic.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": agent.training.policy_learning_rate},
            {"params": critic.parameters(), "lr": agent.training.critic_learning_rate},
        ]
    )
    if args.resume:
        load_controller_checkpoint(
            args.resume, policy=policy, critic=critic, optimizer=optimizer, map_location=device
        )
    updater = ActorCriticUpdater(
        policy,
        critic,
        optimizer,
        exploration_std=agent.policy.exploration_std,
        entropy_weight=agent.training.entropy_weight,
    )
    reward = create_reward(agent.reward)

    prompt_path = Path(args.prompts_file).expanduser().resolve()
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts were supplied")
    output_dir = config.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = agent.training.epochs if args.epochs is None else args.epochs

    for epoch in range(epochs):
        for index, prompt in enumerate(prompts):
            seed = config.generation.seed + index
            reference_path = output_dir / f"e{epoch:03d}-{index:05d}-fixed.png"
            candidate_path = output_dir / f"e{epoch:03d}-{index:05d}-policy.png"
            backend.set_controller(agent, None)
            reference_sampler = (
                "euler" if agent.sampler == "adaptive_euler" else "dpmpp_2m"
            )
            backend.generate(
                _request(config, prompt, reference_path, seed, reference_sampler)
            )
            trace = []
            backend.set_controller(
                agent, ExplorationStepPolicy(policy, agent.policy.exploration_std)
            )
            backend.generate(
                _request(config, prompt, candidate_path, seed, agent.sampler, trace)
            )
            if len(trace) != config.generation.steps:
                raise RuntimeError("Adaptive rollout violated the fixed NFE budget")
            terminal_reward = reward.relative(candidate_path, reference_path, prompt)
            metrics = updater.update(trace, terminal_reward)
            print(
                f"epoch={epoch} prompt={index} reward={metrics.reward:+.6f} "
                f"policy_loss={metrics.policy_loss:+.6f} critic_loss={metrics.critic_loss:.6f}"
            )
        save_controller_checkpoint(
            config.resolve_path(args.checkpoint),
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            metadata={"epoch": epoch + 1, "agent_config": str(agent.source)},
        )
        print(f"cuda_memory {cuda_memory_summary(config.memory.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
