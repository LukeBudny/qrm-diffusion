from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from qrm_diffusion.agents import (
    QualityCritic,
    create_policy,
    load_agent_config,
    load_controller_checkpoint,
)
from qrm_diffusion.agents.reward import create_reward
from qrm_diffusion.agents.rollout import grouped_schedule_rollout
from qrm_diffusion.agents.diagnostics import (
    summarize_policy_records,
    trajectory_diagnostics,
)
from qrm_diffusion.agents.evaluation import evaluate_joint_gate
from qrm_diffusion.backends import create_backend
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
    parser.add_argument(
        "--seed-offsets",
        default="0,10000,20000",
        help="Comma-separated repeat offsets; every repeat must pass the gate",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run the requested repeats and write diagnostics without applying the joint gate",
    )
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
    apply_cuda_memory_policy(config.memory)
    backend = create_backend(config.model.backend, replace(
        config, controller=replace(config.controller, enabled=False)
    ))
    backend.load()
    policy = create_policy(agent, device=f"cuda:{config.memory.device}")
    critic = QualityCritic(agent.critic.hidden_dim).to(f"cuda:{config.memory.device}")
    load_controller_checkpoint(
        args.checkpoint,
        policy=policy,
        critic=critic,
        map_location=f"cuda:{config.memory.device}",
    )
    policy.eval()
    critic.eval()
    reward = create_reward(agent.reward)
    prompt_path = Path(args.prompts_file).expanduser().resolve()
    prompts = [
        line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    metadata_path = prompt_path.with_suffix(".jsonl")
    prompt_metadata = {}
    if metadata_path.is_file():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                prompt_metadata[int(item["prompt_index"])] = item
    output_dir = config.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "evaluation.jsonl"
    completed = {}
    if progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[(int(item["repeat"]), int(item["prompt_index"]))] = item
    seed_offsets = [int(value.strip()) for value in args.seed_offsets.split(",") if value.strip()]
    if not args.diagnostic_only and len(seed_offsets) < agent.evaluation.required_repeats:
        raise ValueError(
            f"The consistency gate requires at least {agent.evaluation.required_repeats} repeats"
        )
    repeats = []
    for repeat_index, seed_offset in enumerate(seed_offsets):
        deltas = []
        repeat_dir = output_dir / f"repeat-{repeat_index:02d}"
        for index, prompt in enumerate(prompts):
            previous = completed.get((repeat_index, index))
            if previous is not None:
                deltas.append(float(previous["reward_delta"]))
                continue
            seed = config.generation.seed + seed_offset + index
            fixed_path = repeat_dir / f"{index:05d}-fixed.png"
            policy_path = repeat_dir / f"{index:05d}-policy.png"
            reference_trace, traces = grouped_schedule_rollout(
                backend,
                config,
                agent,
                prompt=prompt,
                seed=seed,
                fixed_path=fixed_path,
                candidate_paths=[policy_path],
                policies=[policy],
            )
            trace = traces[0]
            delta = reward.relative(policy_path, fixed_path, prompt)
            deltas.append(delta)
            metadata = prompt_metadata.get(index, {})
            record = {
                "repeat": repeat_index,
                "seed_offset": seed_offset,
                "prompt_index": index,
                "seed": seed,
                "nfe": len(trace),
                "reward_delta": delta,
                "prompt": prompt,
                "category": metadata.get("category"),
                "challenge": metadata.get("challenge"),
                "trajectory": trajectory_diagnostics(
                    trace, reference_trace, critic=critic
                ),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"repeat={repeat_index} prompt={index} nfe={len(trace)} "
                f"reward_delta={delta:+.6f}"
            )
        result = evaluate_joint_gate(deltas, agent.evaluation)
        repeats.append({
            "repeat": repeat_index,
            "seed_offset": seed_offset,
            "passed": result.passed,
            "prompt_count": result.prompt_count,
            "mean_reward_delta": result.mean_reward_delta,
            "positive_fraction": result.positive_fraction,
        })
        print(
            f"repeat={repeat_index} mean_reward_delta={result.mean_reward_delta:+.6f} "
            f"positive_fraction={result.positive_fraction:.3f} "
            f"gate={'PASS' if result.passed else 'FAIL'}"
        )
        if not args.diagnostic_only and not result.passed:
            print("Stopping early: consistency requires every repeat to pass")
            break
    passed = len(repeats) >= agent.evaluation.required_repeats and all(
        item["passed"] for item in repeats
    )
    all_records = [
        item
        for item in completed.values()
        if int(item.get("repeat", 0)) < len(seed_offsets)
    ]
    progress_records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {
        (int(item["repeat"]), int(item["prompt_index"])): item
        for item in [*all_records, *progress_records]
    }
    diagnostic_records = [
        item for item in by_key.values() if item.get("trajectory") is not None
    ]
    if diagnostic_records:
        diagnostics = summarize_policy_records(
            diagnostic_records,
            bootstrap_samples=agent.evaluation.bootstrap_samples,
            bootstrap_confidence=agent.evaluation.bootstrap_confidence,
        )
        (output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "passed": passed,
        "required_repeats": agent.evaluation.required_repeats,
        "early_stopped": len(repeats) < len(seed_offsets),
        "repeats": repeats,
        "nfe": config.generation.steps,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "diagnostic_only": args.diagnostic_only,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"repeats={len(repeats)} prompts_per_repeat={len(prompts)} "
        f"nfe={config.generation.steps} joint_gate={'PASS' if passed else 'FAIL'}"
    )
    return 0 if args.diagnostic_only or passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
