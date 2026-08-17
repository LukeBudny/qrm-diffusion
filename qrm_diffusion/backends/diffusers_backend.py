from __future__ import annotations

import inspect

from .base import DiffusionBackend, GenerationRequest
from .registry import register_backend


@register_backend("diffusers")
class DiffusersBackend(DiffusionBackend):
    """Baseline text-to-image support for Hugging Face Diffusers pipelines."""

    def __init__(self, config):
        super().__init__(config)
        self.pipeline = None
        self.device = f"cuda:{config.memory.device}"

    def load(self) -> None:
        if self.config.qrm.enabled:
            raise ValueError("QRM hooks are currently available only with backend='sd35_native'")

        import torch
        from diffusers import DiffusionPipeline

        options = self.config.model.options
        source = options.get("model_id") or options.get("checkpoint")
        if not source:
            raise ValueError("diffusers requires [model].model_id or [model].checkpoint")
        if options.get("checkpoint") and not options.get("model_id"):
            source = str(self.config.resolve_path(str(source)))

        dtype_name = str(options.get("dtype", "float16"))
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype_name}'")

        load_kwargs = {
            "torch_dtype": dtype_map[dtype_name],
            "low_cpu_mem_usage": bool(options.get("low_cpu_mem_usage", True)),
        }
        for key in (
            "local_files_only",
            "revision",
            "variant",
            "use_safetensors",
            "trust_remote_code",
        ):
            if key in options:
                load_kwargs[key] = options[key]

        pipeline = DiffusionPipeline.from_pretrained(str(source), **load_kwargs)
        if bool(options.get("attention_slicing", True)) and hasattr(
            pipeline, "enable_attention_slicing"
        ):
            pipeline.enable_attention_slicing()
        if bool(options.get("vae_slicing", True)) and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if bool(options.get("cpu_offload", True)):
            pipeline.enable_model_cpu_offload(gpu_id=self.config.memory.device)
        else:
            pipeline.to(self.device)
        self.pipeline = pipeline

    def _accepts(self, name: str) -> bool:
        signature = inspect.signature(self.pipeline.__call__)
        return name in signature.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )

    def generate(self, request: GenerationRequest):
        if self.pipeline is None:
            raise RuntimeError("Backend has not been loaded")

        import torch

        kwargs: dict[str, object] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width,
            "height": request.height,
            "generator": torch.Generator(device=self.device).manual_seed(request.seed),
        }
        if request.negative_prompt and self._accepts("negative_prompt"):
            kwargs["negative_prompt"] = request.negative_prompt
        kwargs.update(request.extra)
        kwargs = {key: value for key, value in kwargs.items() if self._accepts(key)}

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.pipeline(**kwargs)
        result.images[0].save(request.output_path)
        return request.output_path
