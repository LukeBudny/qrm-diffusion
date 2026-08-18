from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, pvariance
from typing import Any, Callable, Sequence

import torch

from .state import TrajectoryEntry


def trajectory_diagnostics(
    trajectory: Sequence[TrajectoryEntry],
    reference: Sequence[TrajectoryEntry],
    critic=None,
) -> dict[str, Any]:
    reference_by_step = {entry.step_index: entry for entry in reference}
    actions = [float(entry.action) for entry in trajectory]
    log_deviations: list[float | None] = []
    reference_next_sigmas: list[float | None] = []
    for entry in trajectory:
        fixed = reference_by_step.get(entry.step_index)
        fixed_next = None if fixed is None else float(fixed.next_sigma)
        reference_next_sigmas.append(fixed_next)
        if fixed_next is None or fixed_next <= 0 or entry.next_sigma <= 0:
            log_deviations.append(None)
        else:
            log_deviations.append(abs(math.log(float(entry.next_sigma) / fixed_next)))

    critic_values = None
    usable = [entry for entry in trajectory if entry.policy_features is not None]
    if critic is not None and len(usable) == len(trajectory):
        device = next(critic.parameters()).device
        observations = torch.tensor(
            [entry.policy_features for entry in usable],
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            critic_values = [float(value) for value in critic(observations).cpu()]

    finite_deviations = [value for value in log_deviations if value is not None]
    return {
        "actions": actions,
        "sigmas": [float(entry.sigma) for entry in trajectory],
        "next_sigmas": [float(entry.next_sigma) for entry in trajectory],
        "reference_next_sigmas": reference_next_sigmas,
        "absolute_log_sigma_deviations": log_deviations,
        "critic_values": critic_values,
        "mean_absolute_action": mean(abs(value) for value in actions),
        "max_absolute_action": max(abs(value) for value in actions),
        "mean_absolute_log_sigma_deviation": (
            mean(finite_deviations) if finite_deviations else 0.0
        ),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    samples: int,
    confidence: float,
    seed: int = 3505,
) -> tuple[float, float]:
    if not values or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap inputs")
    rng = random.Random(seed)
    estimates = sorted(
        statistic([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _reward_summary(
    values: Sequence[float], *, samples: int, confidence: float
) -> dict[str, Any]:
    values = [float(value) for value in values]
    result = {
        "count": len(values),
        "mean": mean(values),
        "variance": pvariance(values) if len(values) > 1 else 0.0,
        "positive_fraction": sum(value > 0 for value in values) / len(values),
    }
    result["mean_bootstrap_interval"] = list(
        bootstrap_interval(
            values, mean, samples=samples, confidence=confidence
        )
    )
    result["positive_fraction_bootstrap_interval"] = list(
        bootstrap_interval(
            values,
            lambda sample: sum(value > 0 for value in sample) / len(sample),
            samples=samples,
            confidence=confidence,
            seed=3506,
        )
    )
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    return None if denominator == 0 else numerator / denominator


def summarize_policy_records(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one policy record is required")
    rewards = [float(record["reward_delta"]) for record in records]
    grouped: dict[str, dict[str, list[float]]] = {
        "category": defaultdict(list),
        "challenge": defaultdict(list),
    }
    for record, reward in zip(records, rewards):
        for field in grouped:
            value = record.get(field)
            if value:
                grouped[field][str(value)].append(reward)

    trajectories = [record.get("trajectory") for record in records]
    trajectories = [value for value in trajectories if isinstance(value, dict)]
    action_by_step: list[dict[str, float | int]] = []
    schedule_by_step: list[dict[str, float | int]] = []
    max_steps = max((len(item.get("actions", [])) for item in trajectories), default=0)
    for step in range(max_steps):
        actions = [
            abs(float(item["actions"][step]))
            for item in trajectories
            if step < len(item.get("actions", []))
        ]
        deviations = [
            item["absolute_log_sigma_deviations"][step]
            for item in trajectories
            if step < len(item.get("absolute_log_sigma_deviations", []))
            and item["absolute_log_sigma_deviations"][step] is not None
        ]
        if actions:
            action_by_step.append(
                {
                    "step": step,
                    "mean_absolute_action": mean(actions),
                    "max_absolute_action": max(actions),
                }
            )
        if deviations:
            schedule_by_step.append(
                {
                    "step": step,
                    "mean_absolute_log_sigma_deviation": mean(deviations),
                    "max_absolute_log_sigma_deviation": max(deviations),
                }
            )

    critic_predictions: list[float] = []
    critic_targets: list[float] = []
    for record in records:
        trajectory = record.get("trajectory") or {}
        values = trajectory.get("critic_values")
        if values:
            critic_predictions.extend(float(value) for value in values)
            critic_targets.extend([float(record["reward_delta"])] * len(values))
    critic_summary: dict[str, Any] = {"observation_count": len(critic_predictions)}
    if critic_predictions:
        errors = [
            prediction - target
            for prediction, target in zip(critic_predictions, critic_targets)
        ]
        target_mean = mean(critic_targets)
        mse = mean(error * error for error in errors)
        baseline_mse = mean((target - target_mean) ** 2 for target in critic_targets)
        critic_summary.update(
            {
                "pearson_correlation": _pearson(critic_predictions, critic_targets),
                "mae": mean(abs(error) for error in errors),
                "rmse": math.sqrt(mse),
                "constant_mean_rmse": math.sqrt(baseline_mse),
                "r_squared": None if baseline_mse == 0 else 1.0 - mse / baseline_mse,
            }
        )

    return {
        "reward": _reward_summary(
            rewards,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
        ),
        "by_category": {
            key: _reward_summary(
                values, samples=bootstrap_samples, confidence=bootstrap_confidence
            )
            for key, values in sorted(grouped["category"].items())
        },
        "by_challenge": {
            key: _reward_summary(
                values, samples=bootstrap_samples, confidence=bootstrap_confidence
            )
            for key, values in sorted(grouped["challenge"].items())
        },
        "action_by_step": action_by_step,
        "schedule_by_step": schedule_by_step,
        "critic": critic_summary,
    }
