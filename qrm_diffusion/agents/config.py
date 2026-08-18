from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ScheduleSettings:
    beta: float = 0.35
    min_step_size: float = 1.0e-8
    max_step_size: float = 1.0e6
    min_step_ratio: float = 1.0e-3
    max_step_ratio: float = 1.0e3
    terminal_sigma: float = 0.0
    fixed_nfe: bool = True


@dataclass(frozen=True)
class InitializationSettings:
    schedule_action: float = 0.0
    modulation_action: float = 0.0


@dataclass(frozen=True)
class PolicySettings:
    hidden_dim: int = 128
    exploration_std: float = 0.15


@dataclass(frozen=True)
class CriticSettings:
    hidden_dim: int = 128


@dataclass(frozen=True)
class RewardSettings:
    scorer: str = "clip"
    device: str = "cpu"
    model_id: str = "openai/clip-vit-base-patch32"
    local_files_only: bool = False


@dataclass(frozen=True)
class DiagnosticsSettings:
    record_sigma: bool = True
    record_action: bool = True
    record_step_size: bool = True
    record_quality: bool = True
    record_modulation_norm: bool = True


@dataclass(frozen=True)
class TrainingSettings:
    quality_critic: bool = False
    timestep_policy: bool = False
    joint_controller: bool = False
    policy_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    entropy_weight: float = 1.0e-3
    candidates_per_prompt: int = 4
    batch_prompts: int = 4
    normalize_advantages: bool = True
    action_l2_weight: float = 1.0e-3
    kl_weight: float = 1.0e-3
    max_grad_norm: float = 1.0
    validation_interval_prompts: int = 32
    validation_max_prompts: int = 32
    epochs: int = 1


@dataclass(frozen=True)
class EvaluationSettings:
    min_prompts: int = 32
    min_mean_reward_delta: float = 0.0
    min_positive_fraction: float = 0.6
    required_repeats: int = 3
    bootstrap_samples: int = 2000
    bootstrap_confidence: float = 0.95


@dataclass(frozen=True)
class AgentConfig:
    source: Path
    schema_version: int
    name: str
    architecture: str
    sampler: str
    schedule: ScheduleSettings
    initialization: InitializationSettings
    policy: PolicySettings
    critic: CriticSettings
    reward: RewardSettings
    diagnostics: DiagnosticsSettings
    training: TrainingSettings
    evaluation: EvaluationSettings


def _table(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a TOML table")
    return value


def load_agent_config(path: str | Path) -> AgentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Agent configuration file not found: {source}")
    with source.open("rb") as handle:
        data = tomllib.load(handle)

    schedule_data = _table(data, "schedule")
    initialization_data = _table(data, "initialization")
    policy_data = _table(data, "policy")
    critic_data = _table(data, "critic")
    reward_data = _table(data, "reward")
    diagnostics_data = _table(data, "diagnostics")
    training_data = _table(data, "training")
    evaluation_data = _table(data, "evaluation")
    schedule = ScheduleSettings(
        beta=float(schedule_data.get("beta", 0.35)),
        min_step_size=float(schedule_data.get("min_step_size", 1.0e-8)),
        max_step_size=float(schedule_data.get("max_step_size", 1.0e6)),
        min_step_ratio=float(schedule_data.get("min_step_ratio", 1.0e-3)),
        max_step_ratio=float(schedule_data.get("max_step_ratio", 1.0e3)),
        terminal_sigma=float(schedule_data.get("terminal_sigma", 0.0)),
        fixed_nfe=bool(schedule_data.get("fixed_nfe", True)),
    )
    initialization = InitializationSettings(
        schedule_action=float(initialization_data.get("schedule_action", 0.0)),
        modulation_action=float(initialization_data.get("modulation_action", 0.0)),
    )
    policy = PolicySettings(
        hidden_dim=int(policy_data.get("hidden_dim", 128)),
        exploration_std=float(policy_data.get("exploration_std", 0.15)),
    )
    critic = CriticSettings(hidden_dim=int(critic_data.get("hidden_dim", 128)))
    reward = RewardSettings(
        scorer=str(reward_data.get("scorer", "clip")),
        device=str(reward_data.get("device", "cpu")),
        model_id=str(reward_data.get("model_id", "openai/clip-vit-base-patch32")),
        local_files_only=bool(reward_data.get("local_files_only", False)),
    )
    diagnostics = DiagnosticsSettings(
        record_sigma=bool(diagnostics_data.get("record_sigma", True)),
        record_action=bool(diagnostics_data.get("record_action", True)),
        record_step_size=bool(diagnostics_data.get("record_step_size", True)),
        record_quality=bool(diagnostics_data.get("record_quality", True)),
        record_modulation_norm=bool(
            diagnostics_data.get("record_modulation_norm", True)
        ),
    )
    training = TrainingSettings(
        quality_critic=bool(training_data.get("quality_critic", False)),
        timestep_policy=bool(training_data.get("timestep_policy", False)),
        joint_controller=bool(training_data.get("joint_controller", False)),
        policy_learning_rate=float(training_data.get("policy_learning_rate", 1.0e-4)),
        critic_learning_rate=float(training_data.get("critic_learning_rate", 3.0e-4)),
        entropy_weight=float(training_data.get("entropy_weight", 1.0e-3)),
        candidates_per_prompt=int(training_data.get("candidates_per_prompt", 4)),
        batch_prompts=int(training_data.get("batch_prompts", 4)),
        normalize_advantages=bool(training_data.get("normalize_advantages", True)),
        action_l2_weight=float(training_data.get("action_l2_weight", 1.0e-3)),
        kl_weight=float(training_data.get("kl_weight", 1.0e-3)),
        max_grad_norm=float(training_data.get("max_grad_norm", 1.0)),
        validation_interval_prompts=int(
            training_data.get("validation_interval_prompts", 32)
        ),
        validation_max_prompts=int(training_data.get("validation_max_prompts", 32)),
        epochs=int(training_data.get("epochs", 1)),
    )
    evaluation = EvaluationSettings(
        min_prompts=int(evaluation_data.get("min_prompts", 32)),
        min_mean_reward_delta=float(
            evaluation_data.get("min_mean_reward_delta", 0.0)
        ),
        min_positive_fraction=float(
            evaluation_data.get("min_positive_fraction", 0.6)
        ),
        required_repeats=int(evaluation_data.get("required_repeats", 3)),
        bootstrap_samples=int(evaluation_data.get("bootstrap_samples", 2000)),
        bootstrap_confidence=float(
            evaluation_data.get("bootstrap_confidence", 0.95)
        ),
    )

    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("Only agent schema_version=1 is supported")
    if str(data.get("architecture", "")) != "sd35_native":
        raise ValueError("The adaptive controller currently supports architecture='sd35_native'")
    if str(data.get("sampler", "")) not in ("adaptive_euler", "adaptive_dpmpp_2m"):
        raise ValueError("Agent sampler must be adaptive_euler or adaptive_dpmpp_2m")
    if not schedule.fixed_nfe or schedule.terminal_sigma != 0.0:
        raise ValueError("The controller requires fixed_nfe=true and terminal_sigma=0")
    if policy.hidden_dim <= 0 or policy.exploration_std <= 0:
        raise ValueError("Policy hidden_dim and exploration_std must be positive")
    if critic.hidden_dim <= 0:
        raise ValueError("Critic hidden_dim must be positive")
    if training.joint_controller and not (
        training.quality_critic and training.timestep_policy
    ):
        raise ValueError("joint_controller requires critic and timestep policy training")
    if (
        training.candidates_per_prompt < 2
        or training.batch_prompts <= 0
        or training.action_l2_weight < 0
        or training.kl_weight < 0
        or training.max_grad_norm <= 0
        or training.validation_interval_prompts <= 0
        or training.validation_max_prompts <= 0
    ):
        raise ValueError("Invalid batched training settings")
    if (
        evaluation.min_prompts <= 0
        or not 0 <= evaluation.min_positive_fraction <= 1
        or evaluation.required_repeats <= 0
        or evaluation.bootstrap_samples <= 0
        or not 0 < evaluation.bootstrap_confidence < 1
    ):
        raise ValueError("Invalid evaluation gate settings")

    return AgentConfig(
        source=source,
        schema_version=1,
        name=str(data.get("name", source.stem)),
        architecture="sd35_native",
        sampler=str(data["sampler"]),
        schedule=schedule,
        initialization=initialization,
        policy=policy,
        critic=critic,
        reward=reward,
        diagnostics=diagnostics,
        training=training,
        evaluation=evaluation,
    )
