from __future__ import annotations

from typing import TypeVar

from .base import DiffusionBackend


BackendType = TypeVar("BackendType", bound=type[DiffusionBackend])
_BACKENDS: dict[str, type[DiffusionBackend]] = {}


def register_backend(name: str):
    key = name.strip().lower()
    if not key:
        raise ValueError("Backend name cannot be empty")

    def decorator(cls: BackendType) -> BackendType:
        if key in _BACKENDS:
            raise ValueError(f"Backend already registered: {key}")
        _BACKENDS[key] = cls
        return cls

    return decorator


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_backend(name: str, config) -> DiffusionBackend:
    key = name.strip().lower()
    try:
        backend_class = _BACKENDS[key]
    except KeyError as exc:
        choices = ", ".join(available_backends()) or "none"
        raise ValueError(f"Unknown backend '{name}'. Available backends: {choices}") from exc
    return backend_class(config)
