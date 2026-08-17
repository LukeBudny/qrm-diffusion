from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    prompt_count: int
    mean_reward_delta: float
    positive_fraction: float


def evaluate_joint_gate(deltas: Sequence[float], settings) -> EvaluationResult:
    if not deltas:
        raise ValueError("At least one reward delta is required")
    mean = sum(float(value) for value in deltas) / len(deltas)
    positive_fraction = sum(float(value) > 0 for value in deltas) / len(deltas)
    passed = (
        len(deltas) >= settings.min_prompts
        and mean >= settings.min_mean_reward_delta
        and positive_fraction >= settings.min_positive_fraction
    )
    return EvaluationResult(passed, len(deltas), mean, positive_fraction)
