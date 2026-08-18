from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from qrm_diffusion.agents import (
    AdaptiveSamplerState,
    ExplorationStepPolicy,
    QualityCritic,
    RecedingHorizonStepPolicy,
    load_agent_config,
    load_controller_checkpoint,
    save_controller_checkpoint,
)
from qrm_diffusion.agents.state import TrajectoryEntry
from qrm_diffusion.agents.training import ActorCriticUpdater
from qrm_diffusion.agents.evaluation import evaluate_joint_gate
from qrm_diffusion.agents.reward import CLIPReward
from qrm_diffusion.agents.diagnostics import summarize_policy_records


ROOT = Path(__file__).resolve().parents[1]


def test_agent_toml_enables_schedule_training_but_not_joint_control() -> None:
    config = load_agent_config(ROOT / "configs/agents/sd35-qrm-timestep.toml")
    assert config.architecture == "sd35_native"
    assert config.sampler == "adaptive_dpmpp_2m"
    assert config.schedule.fixed_nfe
    assert config.training.quality_critic
    assert config.training.timestep_policy
    assert not config.training.joint_controller
    assert config.evaluation.required_repeats == 3
    assert config.training.candidates_per_prompt == 4
    assert config.training.batch_prompts == 4
    assert config.training.normalize_advantages
    assert config.training.action_l2_weight > 0
    assert config.training.kl_weight > 0


def test_controller_checkpoint_round_trip(tmp_path: Path) -> None:
    source = RecedingHorizonStepPolicy(hidden_dim=8)
    critic = QualityCritic(hidden_dim=8)
    with torch.no_grad():
        source.output.bias.fill_(0.25)
    path = save_controller_checkpoint(
        tmp_path / "controller.pt",
        policy=source,
        critic=critic,
        metadata={"epoch": 3},
    )
    restored = RecedingHorizonStepPolicy(hidden_dim=8)
    restored_critic = QualityCritic(hidden_dim=8)
    metadata = load_controller_checkpoint(
        path, policy=restored, critic=restored_critic
    )
    assert restored.output.bias.item() == 0.25
    assert metadata == {"epoch": 3}


def test_actor_critic_update_changes_policy_from_zero_initialization() -> None:
    torch.manual_seed(0)
    policy = RecedingHorizonStepPolicy(hidden_dim=8)
    critic = QualityCritic(hidden_dim=8)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=1.0e-2
    )
    updater = ActorCriticUpdater(
        policy, critic, optimizer, exploration_std=0.2, entropy_weight=0.0
    )
    trajectory = [
        TrajectoryEntry(
            step_index=index,
            sigma=1.0 / (index + 1),
            next_sigma=0.5 / (index + 1),
            step_size=0.1,
            action=0.2,
            policy_features=(0.1, 0.2, 0.3, float(index), 0.5, 0.1),
        )
        for index in range(3)
    ]
    before = policy.output.bias.detach().clone()
    metrics = updater.update(trajectory, terminal_reward=0.5)
    assert metrics.reward == 0.5
    assert not torch.equal(policy.output.bias.detach(), before)


def test_joint_gate_requires_enough_prompts_and_consistent_improvement() -> None:
    config = load_agent_config(ROOT / "configs/agents/sd35-qrm-timestep.toml")
    smoke = evaluate_joint_gate([0.01], config.evaluation)
    assert not smoke.passed
    passing = evaluate_joint_gate([0.01] * config.evaluation.min_prompts, config.evaluation)
    assert passing.passed


def test_parti_training_and_heldout_splits_are_disjoint() -> None:
    split_dir = ROOT / "configs/prompts/parti"
    train = set((split_dir / "train.txt").read_text(encoding="utf-8").splitlines())
    heldout = set((split_dir / "heldout.txt").read_text(encoding="utf-8").splitlines())
    validation = set(
        (split_dir / "validation.txt").read_text(encoding="utf-8").splitlines()
    )
    assert len(train) == 96
    assert len(heldout) == 48
    assert len(validation) == 48
    assert train.isdisjoint(heldout)
    assert train.isdisjoint(validation)
    assert heldout.isdisjoint(validation)


def test_clip_reward_truncates_long_prompts(tmp_path: Path) -> None:
    captured = {}

    class Batch(dict):
        def to(self, _device):
            return self

    class Processor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return Batch()

    class Model:
        config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )

        def __call__(self, **_kwargs):
            embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
            return SimpleNamespace(image_embeds=embeddings, text_embeds=embeddings)

    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    reward = CLIPReward.__new__(CLIPReward)
    reward.device = torch.device("cpu")
    reward.processor = Processor()
    reward.model = Model()
    reward.score([image_path, image_path], ["word " * 100, "word " * 100])
    assert captured["truncation"] is True
    assert captured["max_length"] == 77


def test_batched_actor_critic_normalizes_and_regularizes() -> None:
    torch.manual_seed(1)
    policy = RecedingHorizonStepPolicy(hidden_dim=8)
    critic = QualityCritic(hidden_dim=8)
    with torch.no_grad():
        policy.output.bias.fill_(0.1)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=1.0e-2
    )
    updater = ActorCriticUpdater(
        policy,
        critic,
        optimizer,
        exploration_std=0.2,
        entropy_weight=0.0,
        action_l2_weight=0.01,
        kl_weight=0.01,
    )
    trajectories = []
    for trajectory_index in range(4):
        trajectories.append(
            [
                TrajectoryEntry(
                    step_index=step,
                    sigma=1.0 / (step + 1),
                    next_sigma=0.5 / (step + 1),
                    step_size=0.1,
                    action=(-0.2 + trajectory_index * 0.1 + step * 0.01),
                    policy_features=(
                        0.1 * trajectory_index,
                        0.2,
                        0.3,
                        float(step),
                        0.5,
                        0.1,
                    ),
                )
                for step in range(3)
            ]
        )
    before = policy.output.bias.detach().clone()
    metrics = updater.update_batch(
        trajectories, [-0.2, -0.05, 0.1, 0.3], normalize_advantages=True
    )
    assert not torch.equal(policy.output.bias.detach(), before)
    assert metrics.reward_std > 0
    assert metrics.advantage_std > 0
    assert metrics.action_l2 > 0
    assert metrics.policy_kl > 0


def test_candidate_exploration_uses_independent_seeded_rng_streams() -> None:
    policy = RecedingHorizonStepPolicy(hidden_dim=8)
    state = AdaptiveSamplerState(
        current_sigma=torch.tensor(1.0),
        previous_sigma=None,
        previous_denoised=None,
        remaining_steps=3,
    )
    guided = torch.ones(1, 2, 2, 2)
    first = ExplorationStepPolicy(policy, 0.2, seed=100)
    first_replay = ExplorationStepPolicy(policy, 0.2, seed=100)
    second = ExplorationStepPolicy(policy, 0.2, seed=101)
    first_action = first(state, guided)
    assert torch.equal(first_action, first_replay(state, guided))
    assert not torch.equal(first_action, second(state, guided))


def test_policy_diagnostics_include_bootstrap_strata_schedule_and_critic() -> None:
    records = []
    for index, reward in enumerate([-0.02, 0.01, 0.03, 0.04]):
        records.append(
            {
                "reward_delta": reward,
                "category": "Animals" if index < 2 else "Artifacts",
                "challenge": "Basic" if index % 2 == 0 else "Complex",
                "trajectory": {
                    "actions": [0.1 + index * 0.01, -0.05],
                    "absolute_log_sigma_deviations": [0.01 * index, None],
                    "critic_values": [reward * 0.8, reward * 0.9],
                },
            }
        )
    summary = summarize_policy_records(records, bootstrap_samples=100)
    assert summary["reward"]["count"] == 4
    assert len(summary["reward"]["mean_bootstrap_interval"]) == 2
    assert set(summary["by_category"]) == {"Animals", "Artifacts"}
    assert len(summary["action_by_step"]) == 2
    assert len(summary["schedule_by_step"]) == 1
    assert summary["critic"]["pearson_correlation"] > 0.9
