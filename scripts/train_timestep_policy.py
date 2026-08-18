from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean

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
from qrm_diffusion.agents.rollout import grouped_schedule_rollout
from qrm_diffusion.agents.diagnostics import trajectory_diagnostics
from qrm_diffusion.agents.training import ActorCriticUpdater
from qrm_diffusion.backends import create_backend
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
    parser.add_argument("--validation-prompts-file")
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--candidates-per-prompt", type=int)
    parser.add_argument("--batch-prompts", type=int)
    parser.add_argument("--validation-interval-prompts", type=int)
    parser.add_argument("--validation-max-prompts", type=int)
    return parser.parse_args()


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
    agent = replace(
        agent,
        training=replace(
            agent.training,
            candidates_per_prompt=(
                agent.training.candidates_per_prompt
                if args.candidates_per_prompt is None
                else args.candidates_per_prompt
            ),
            batch_prompts=(
                agent.training.batch_prompts
                if args.batch_prompts is None
                else args.batch_prompts
            ),
            validation_interval_prompts=(
                agent.training.validation_interval_prompts
                if args.validation_interval_prompts is None
                else args.validation_interval_prompts
            ),
            validation_max_prompts=(
                agent.training.validation_max_prompts
                if args.validation_max_prompts is None
                else args.validation_max_prompts
            ),
        ),
    )
    if (
        agent.training.candidates_per_prompt < 2
        or agent.training.batch_prompts <= 0
        or agent.training.validation_interval_prompts <= 0
        or agent.training.validation_max_prompts <= 0
    ):
        raise ValueError("Training and validation CLI overrides must be positive")
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
    resume_metadata = {}
    if args.resume:
        resume_metadata = load_controller_checkpoint(
            args.resume, policy=policy, critic=critic, optimizer=optimizer, map_location=device
        )
    updater = ActorCriticUpdater(
        policy,
        critic,
        optimizer,
        exploration_std=agent.policy.exploration_std,
        entropy_weight=agent.training.entropy_weight,
        action_l2_weight=agent.training.action_l2_weight,
        kl_weight=agent.training.kl_weight,
        max_grad_norm=agent.training.max_grad_norm,
    )
    reward = create_reward(agent.reward)

    prompt_path = Path(args.prompts_file).expanduser().resolve()
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts were supplied")
    validation_path = (
        Path(args.validation_prompts_file).expanduser().resolve()
        if args.validation_prompts_file
        else prompt_path.with_name("validation.txt")
    )
    if not validation_path.is_file():
        raise FileNotFoundError(
            "A separate validation prompt file is required for checkpoint selection: "
            f"{validation_path}"
        )
    validation_prompts = [
        line.strip()
        for line in validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: agent.training.validation_max_prompts]
    if not validation_prompts:
        raise ValueError("The validation prompt split is empty")
    output_dir = config.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.resolve_path(args.checkpoint)
    best_checkpoint_path = config.resolve_path(
        args.best_checkpoint
        or str(Path(args.checkpoint).with_name(Path(args.checkpoint).stem + "-best.pt"))
    )
    metrics_path = output_dir / "training.jsonl"
    validation_metrics_path = output_dir / "validation.jsonl"
    epochs = agent.training.epochs if args.epochs is None else args.epochs
    start_epoch = int(resume_metadata.get("epoch", 0))
    start_prompt = int(resume_metadata.get("next_prompt", 0))
    if start_prompt >= len(prompts):
        start_epoch += start_prompt // len(prompts)
        start_prompt %= len(prompts)
    best_validation_mean = float(resume_metadata.get("best_validation_mean", "-inf"))
    update_index = int(resume_metadata.get("update_index", 0))

    def checkpoint_metadata(epoch: int, next_prompt: int) -> dict:
        return {
            "epoch": epoch,
            "next_prompt": next_prompt,
            "update_index": update_index,
            "best_validation_mean": best_validation_mean,
            "agent_config": str(agent.source),
            "prompts_file": str(prompt_path),
            "validation_prompts_file": str(validation_path),
        }

    def validate_policy(epoch: int, trained_prompts: int) -> tuple[float, float]:
        validation_dir = output_dir / "validation" / f"u{update_index:05d}"
        deltas = []
        for validation_index, validation_prompt in enumerate(validation_prompts):
            seed = config.generation.seed + 500_000 + validation_index
            fixed_path = validation_dir / f"{validation_index:05d}-fixed.png"
            candidate_path = validation_dir / f"{validation_index:05d}-policy.png"
            reference_trace, traces = grouped_schedule_rollout(
                backend,
                config,
                agent,
                prompt=validation_prompt,
                seed=seed,
                fixed_path=fixed_path,
                candidate_paths=[candidate_path],
                policies=[policy],
            )
            reward_components = reward.relative_many_components(
                [candidate_path], fixed_path, validation_prompt
            )[0]
            delta = reward_components["reward"]
            deltas.append(delta)
            with validation_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "trained_prompts": trained_prompts,
                            "update_index": update_index,
                            "prompt_index": validation_index,
                            "seed": seed,
                            "nfe": len(traces[0]),
                            "reward_delta": delta,
                            "reward_components": reward_components,
                            "trajectory": trajectory_diagnostics(
                                traces[0], reference_trace, critic=critic
                            ),
                        }
                    )
                    + "\n"
                )
        return mean(deltas), sum(value > 0 for value in deltas) / len(deltas)

    for epoch in range(start_epoch, epochs):
        pending_trajectories = []
        pending_rewards = []
        pending_records = []
        pending_prompt_count = 0
        for index, prompt in enumerate(prompts):
            if epoch == start_epoch and index < start_prompt:
                continue
            seed = config.generation.seed + index
            reference_path = output_dir / f"e{epoch:03d}-{index:05d}-fixed.png"
            candidate_paths = [
                output_dir
                / f"e{epoch:03d}-{index:05d}-policy-c{candidate:02d}.png"
                for candidate in range(agent.training.candidates_per_prompt)
            ]
            exploration_policies = [
                ExplorationStepPolicy(
                    policy,
                    agent.policy.exploration_std,
                    seed=(
                        config.generation.seed
                        + epoch * 1_000_000
                        + index * agent.training.candidates_per_prompt
                        + candidate
                    ),
                )
                for candidate in range(agent.training.candidates_per_prompt)
            ]
            reference_trace, traces = grouped_schedule_rollout(
                backend,
                config,
                agent,
                prompt=prompt,
                seed=seed,
                fixed_path=reference_path,
                candidate_paths=candidate_paths,
                policies=exploration_policies,
            )
            reward_components = reward.relative_many_components(
                candidate_paths, reference_path, prompt
            )
            terminal_rewards = [item["reward"] for item in reward_components]
            for candidate, (trace, terminal_reward) in enumerate(
                zip(traces, terminal_rewards)
            ):
                pending_trajectories.append(trace)
                pending_rewards.append(terminal_reward)
                pending_records.append(
                    {
                        "epoch": epoch,
                        "prompt_index": index,
                        "candidate_index": candidate,
                        "seed": seed,
                        "nfe": len(trace),
                        "reward": terminal_reward,
                        "reward_components": reward_components[candidate],
                        "trajectory": trajectory_diagnostics(
                            trace, reference_trace, critic=critic
                        ),
                    }
                )
            pending_prompt_count += 1
            next_prompt = index + 1
            should_update = (
                pending_prompt_count >= agent.training.batch_prompts
                or next_prompt == len(prompts)
            )
            if not should_update:
                continue
            metrics = updater.update_batch(
                pending_trajectories,
                pending_rewards,
                normalize_advantages=agent.training.normalize_advantages,
            )
            update_index += 1
            batch_metrics = asdict(metrics)
            with metrics_path.open("a", encoding="utf-8") as handle:
                for record in pending_records:
                    record["update_index"] = update_index
                    record["batch_metrics"] = batch_metrics
                    handle.write(json.dumps(record) + "\n")
            print(
                f"epoch={epoch} prompts_through={index} candidates={len(pending_rewards)} "
                f"reward={metrics.reward_mean:+.6f}+/-{metrics.reward_std:.6f} "
                f"policy_loss={metrics.policy_loss:+.6f} critic_loss={metrics.critic_loss:.6f} "
                f"kl={metrics.policy_kl:.6f}"
            )
            pending_trajectories.clear()
            pending_rewards.clear()
            pending_records.clear()
            pending_prompt_count = 0
            # Persist the completed optimizer update before the more expensive
            # validation render, so interruption never loses a training batch.
            save_controller_checkpoint(
                checkpoint_path,
                policy=policy,
                critic=critic,
                optimizer=optimizer,
                metadata=checkpoint_metadata(epoch, next_prompt),
            )
            if (
                next_prompt % agent.training.validation_interval_prompts == 0
                or next_prompt == len(prompts)
            ):
                validation_mean, validation_positive = validate_policy(
                    epoch, next_prompt
                )
                print(
                    f"validation prompts={len(validation_prompts)} "
                    f"mean={validation_mean:+.6f} positive={validation_positive:.3f}"
                )
                if validation_mean > best_validation_mean:
                    best_validation_mean = validation_mean
                    save_controller_checkpoint(
                        best_checkpoint_path,
                        policy=policy,
                        critic=critic,
                        optimizer=optimizer,
                        metadata=checkpoint_metadata(epoch, next_prompt),
                    )
                    print(f"new_best_checkpoint={best_checkpoint_path}")
                save_controller_checkpoint(
                    checkpoint_path,
                    policy=policy,
                    critic=critic,
                    optimizer=optimizer,
                    metadata=checkpoint_metadata(epoch, next_prompt),
                )
        save_controller_checkpoint(
            checkpoint_path,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            metadata=checkpoint_metadata(epoch + 1, 0),
        )
        print(f"cuda_memory {cuda_memory_summary(config.memory.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
