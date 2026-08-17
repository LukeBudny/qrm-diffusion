from __future__ import annotations

from dataclasses import dataclass
import os

from .config import MemoryConfig, PROJECT_MAX_VRAM_GIB


GIB = 1024**3


@dataclass(frozen=True)
class CudaMemoryState:
    device: int
    device_name: str
    total_gib: float
    limit_gib: float
    allocator_fraction: float


def allocator_fraction(max_vram_gib: float, total_bytes: int) -> float:
    if max_vram_gib <= 0 or total_bytes <= 0:
        raise ValueError("VRAM budget and total device memory must be positive")
    return min((max_vram_gib * GIB) / total_bytes, 1.0)


def apply_cuda_memory_policy(config: MemoryConfig) -> CudaMemoryState:
    if config.max_vram_gib > PROJECT_MAX_VRAM_GIB:
        raise ValueError(
            f"Requested CUDA budget {config.max_vram_gib:.2f} GiB exceeds the "
            f"project maximum of {PROJECT_MAX_VRAM_GIB:.2f} GiB"
        )
    if config.expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the configured diffusion runtime")
    if config.device < 0 or config.device >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {config.device} is unavailable; found {torch.cuda.device_count()} device(s)"
        )

    torch.cuda.set_device(config.device)
    properties = torch.cuda.get_device_properties(config.device)
    fraction = allocator_fraction(config.max_vram_gib, properties.total_memory)
    if config.enforce:
        torch.cuda.set_per_process_memory_fraction(fraction, config.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    return CudaMemoryState(
        device=config.device,
        device_name=properties.name,
        total_gib=properties.total_memory / GIB,
        limit_gib=min(config.max_vram_gib, properties.total_memory / GIB),
        allocator_fraction=fraction,
    )


def cuda_memory_summary(device: int = 0) -> str:
    import torch

    allocated = torch.cuda.memory_allocated(device) / GIB
    reserved = torch.cuda.memory_reserved(device) / GIB
    peak = torch.cuda.max_memory_allocated(device) / GIB
    return f"allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB peak={peak:.2f} GiB"
