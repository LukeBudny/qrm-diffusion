from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..config import ProjectConfig


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str | None
    output_path: Path
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    extra: dict[str, object]


class DiffusionBackend(ABC):
    supports_qrm = False

    def __init__(self, config: ProjectConfig):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load model components after the CUDA memory policy has been applied."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> Path:
        """Generate one image and return its path."""
