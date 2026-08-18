from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from qrm_diffusion.agents import (
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
    assert len(train) == 96
    assert len(heldout) == 48
    assert train.isdisjoint(heldout)


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
