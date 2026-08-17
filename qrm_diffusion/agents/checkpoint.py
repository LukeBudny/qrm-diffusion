from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .controller import RecedingHorizonStepPolicy
from .critic import QualityCritic


CHECKPOINT_FORMAT = "qrm_diffusion.controller.v1"


def save_controller_checkpoint(
    path: str | Path,
    *,
    policy: RecedingHorizonStepPolicy,
    critic: QualityCritic | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "policy": policy.state_dict(),
        "critic": None if critic is None else critic.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
    return target


def load_controller_checkpoint(
    path: str | Path,
    *,
    policy: RecedingHorizonStepPolicy,
    critic: QualityCritic | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Controller checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported controller checkpoint format: {payload.get('format')!r}")
    policy.load_state_dict(payload["policy"])
    if critic is not None and payload.get("critic") is not None:
        critic.load_state_dict(payload["critic"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return dict(payload.get("metadata") or {})
