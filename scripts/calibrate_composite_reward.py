from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pvariance

from qrm_diffusion.agents import load_agent_config
from qrm_diffusion.agents.reward import create_reward
from qrm_diffusion.config import load_config
from qrm_diffusion.memory import apply_cuda_memory_policy, cuda_memory_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure composite reward component scales on existing paired images"
    )
    parser.add_argument("--config", default="configs/models/sd35-medium-qrm-policy.toml")
    parser.add_argument("--agent-config", default="configs/agents/sd35-qrm-timestep.toml")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    variance = pvariance(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean(values),
        "variance": variance,
        "standard_deviation": variance**0.5,
        "mean_absolute": mean(abs(value) for value in values),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    agent = load_agent_config(args.agent_config)
    apply_cuda_memory_policy(config.memory)
    reward = create_reward(agent.reward)
    prompts = [
        line.strip()
        for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    image_dir = Path(args.image_dir).expanduser().resolve()
    records = []
    for index, prompt in enumerate(prompts):
        fixed = image_dir / f"{index:05d}-fixed.png"
        policy = image_dir / f"{index:05d}-policy.png"
        if not fixed.is_file() or not policy.is_file():
            raise FileNotFoundError(f"Missing paired images for prompt {index}")
        components = reward.relative_many_components([policy], fixed, prompt)[0]
        records.append({"prompt_index": index, **components})
        print(
            f"prompt={index} alignment={components['alignment_delta']:+.6f} "
            f"preference={components['preference_delta']:+.6f}"
        )
    component_names = [
        key for key in records[0] if key not in {"prompt_index", "reward"}
    ]
    report = {
        "components": {
            key: _summary([float(record[key]) for record in records])
            for key in component_names
        },
        "configured_scales": {
            "alignment": agent.reward.alignment_scale,
            "preference": agent.reward.preference_scale,
        },
        "records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["components"], indent=2))
    print(f"cuda_memory {cuda_memory_summary(config.memory.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
