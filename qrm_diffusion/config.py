from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


PROJECT_MAX_VRAM_GIB = 28.0


@dataclass(frozen=True)
class MemoryConfig:
    max_vram_gib: float = PROJECT_MAX_VRAM_GIB
    device: int = 0
    enforce: bool = True
    expandable_segments: bool = True


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str = "A cinematic photograph of a mountain lake at sunrise."
    negative_prompt: str | None = None
    output_dir: str = "outputs"
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QRMConfig:
    enabled: bool = False
    checkpoint: str | None = None
    qrm_type: str = "QRMModulatorLatentV6"
    start_step: int | None = None
    end_step: int | None = None


@dataclass(frozen=True)
class ControllerConfig:
    enabled: bool = False
    config: str | None = None
    checkpoint: str | None = None


@dataclass(frozen=True)
class ProjectConfig:
    source: Path
    root: Path
    memory: MemoryConfig
    model: ModelConfig
    generation: GenerationConfig
    qrm: QRMConfig
    controller: ControllerConfig

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a TOML table")
    return dict(value)


def load_config(path: str | Path) -> ProjectConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")

    with source.open("rb") as handle:
        data = tomllib.load(handle)

    project_data = _table(data, "project")
    root_value = Path(str(project_data.get("root", "."))).expanduser()
    root = root_value if root_value.is_absolute() else source.parent / root_value
    root = root.resolve()

    memory_data = _table(data, "memory")
    memory = MemoryConfig(
        max_vram_gib=float(
            memory_data.get("max_vram_gib", PROJECT_MAX_VRAM_GIB)
        ),
        device=int(memory_data.get("device", 0)),
        enforce=bool(memory_data.get("enforce", True)),
        expandable_segments=bool(memory_data.get("expandable_segments", True)),
    )

    model_data = _table(data, "model")
    backend = str(model_data.pop("backend", "")).strip()
    name = str(model_data.pop("name", backend)).strip()
    if not backend:
        raise ValueError("[model].backend is required")

    generation_data = _table(data, "generation")
    generation_extra = generation_data.pop("extra", {})
    if not isinstance(generation_extra, dict):
        raise ValueError("[generation.extra] must be a TOML table")
    generation = GenerationConfig(
        prompt=str(generation_data.get("prompt", GenerationConfig.prompt)),
        negative_prompt=generation_data.get("negative_prompt"),
        output_dir=str(generation_data.get("output_dir", "outputs")),
        width=int(generation_data.get("width", 1024)),
        height=int(generation_data.get("height", 1024)),
        steps=int(generation_data.get("steps", 30)),
        guidance_scale=float(generation_data.get("guidance_scale", 5.0)),
        seed=int(generation_data.get("seed", 42)),
        extra=dict(generation_extra),
    )

    qrm_data = _table(data, "qrm")
    qrm = QRMConfig(
        enabled=bool(qrm_data.get("enabled", False)),
        checkpoint=qrm_data.get("checkpoint"),
        qrm_type=str(qrm_data.get("qrm_type", "QRMModulatorLatentV6")),
        start_step=qrm_data.get("start_step"),
        end_step=qrm_data.get("end_step"),
    )

    controller_data = _table(data, "controller")
    controller = ControllerConfig(
        enabled=bool(controller_data.get("enabled", False)),
        config=controller_data.get("config"),
        checkpoint=controller_data.get("checkpoint"),
    )

    if memory.max_vram_gib <= 0:
        raise ValueError("[memory].max_vram_gib must be greater than zero")
    if memory.max_vram_gib > PROJECT_MAX_VRAM_GIB:
        raise ValueError(
            f"[memory].max_vram_gib cannot exceed the project limit of "
            f"{PROJECT_MAX_VRAM_GIB:.0f} GiB"
        )
    if generation.width <= 0 or generation.height <= 0:
        raise ValueError("Generation width and height must be greater than zero")
    if generation.width % 8 or generation.height % 8:
        raise ValueError("Generation width and height must be divisible by 8")
    if generation.steps <= 0:
        raise ValueError("[generation].steps must be greater than zero")
    if qrm.enabled and not qrm.checkpoint:
        raise ValueError("[qrm].checkpoint is required when QRM is enabled")
    if controller.enabled and not controller.config:
        raise ValueError("[controller].config is required when the controller is enabled")

    return ProjectConfig(
        source=source,
        root=root,
        memory=memory,
        model=ModelConfig(backend=backend, name=name, options=model_data),
        generation=generation,
        qrm=qrm,
        controller=controller,
    )
