from __future__ import annotations

import os
from pathlib import Path

from .base import DiffusionBackend, GenerationRequest
from .registry import register_backend


@register_backend("sd35_native")
class SD35NativeBackend(DiffusionBackend):
    """Existing custom SD3.5/QRM stack behind the shared backend interface."""

    supports_qrm = True

    def __init__(self, config):
        super().__init__(config)
        self.inferencer = None
        self.qrm_start_step = 25
        self.qrm_end_step = 47
        self.agent_config = None
        self.sigma_policy = None

    def load(self) -> None:
        options = self.config.model.options
        if bool(options.get("local_files_only", False)):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch

        from qrm.qrm_models import QRMRegistry
        from sd3_infer import SD3Inferencer

        checkpoint_value = options.get("checkpoint")
        if not checkpoint_value:
            raise ValueError("sd35_native requires [model].checkpoint")
        checkpoint = self.config.resolve_path(str(checkpoint_value))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SD3.5 checkpoint not found: {checkpoint}")

        model_folder = self.config.resolve_path(str(options.get("model_folder", "models")))
        inferencer = SD3Inferencer()
        inferencer.load(
            model=str(checkpoint),
            vae=None,
            shift=float(options.get("shift", 3.0)),
            controlnet_ckpt=None,
            model_folder=str(model_folder),
            text_encoder_device=str(options.get("text_encoder_device", "cpu")),
            load_tokenizers=bool(options.get("load_tokenizers", True)),
            eval_model=str(options.get("eval_model", "none")),
            inference=True,
            qrm_type=self.config.qrm.qrm_type,
        )

        if self.config.qrm.enabled:
            qrm_path = self.config.resolve_path(str(self.config.qrm.checkpoint))
            if not qrm_path.is_file():
                raise FileNotFoundError(f"QRM checkpoint not found: {qrm_path}")
            state = torch.load(qrm_path, map_location="cpu")
            qrm_type = str(state.get("qrm_type", self.config.qrm.qrm_type))
            if qrm_type not in QRMRegistry:
                raise ValueError(f"Unknown QRM type in checkpoint: {qrm_type}")
            qrm_class = QRMRegistry[qrm_type]
            if qrm_type.startswith("QRMModulatorLatentV"):
                qrm = qrm_class(inferencer.sd3.model._qrm_block_spans)
            else:
                qrm = qrm_class()
            qrm.load_state_dict(state["model"])
            inferencer.sd3.model.qrm = qrm.to(f"cuda:{self.config.memory.device}")
            inferencer.sd3.model.qrm_inference = True
            inferencer.sd3.model.qrm_type = qrm_type
            inferencer.qrm_type = qrm_type
            self.qrm_start_step = int(
                self.config.qrm.start_step
                if self.config.qrm.start_step is not None
                else state.get("qrm_start_step", 25)
            )
            self.qrm_end_step = int(
                self.config.qrm.end_step
                if self.config.qrm.end_step is not None
                else state.get("qrm_end_step", 47)
            )
        else:
            inferencer.sd3.model.qrm = None
            inferencer.sd3.model.qrm_inference = False

        if self.config.controller.enabled:
            from qrm_diffusion.agents import (
                create_policy,
                load_agent_config,
                load_controller_checkpoint,
            )

            agent_path = self.config.resolve_path(str(self.config.controller.config))
            agent_config = load_agent_config(agent_path)
            policy = create_policy(
                agent_config, device=f"cuda:{self.config.memory.device}"
            )
            checkpoint_value = self.config.controller.checkpoint
            if checkpoint_value:
                load_controller_checkpoint(
                    self.config.resolve_path(checkpoint_value),
                    policy=policy,
                    map_location=f"cuda:{self.config.memory.device}",
                )
            policy.eval()
            self.set_controller(agent_config, policy)

        self.inferencer = inferencer

    def set_controller(self, agent_config, policy) -> None:
        if agent_config.architecture != "sd35_native":
            raise ValueError("Native SD3.5 requires an sd35_native agent configuration")
        self.agent_config = agent_config
        self.sigma_policy = policy

    def generate(self, request: GenerationRequest) -> Path:
        if self.inferencer is None:
            raise RuntimeError("Backend has not been loaded")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        sampler = str(request.extra.get("sampler", "dpmpp_2m"))
        policy = self.sigma_policy if sampler.startswith("adaptive_") else None
        controller_options = None
        if policy is not None:
            schedule = self.agent_config.schedule
            controller_options = {
                "beta": schedule.beta,
                "min_step_size": schedule.min_step_size,
                "max_step_size": schedule.max_step_size,
                "min_step_ratio": schedule.min_step_ratio,
                "max_step_ratio": schedule.max_step_ratio,
            }
        self.inferencer.gen_image(
            prompts=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.guidance_scale,
            sampler=sampler,
            seed=request.seed,
            seed_type="fixed",
            out_dir=str(request.output_path.parent),
            save_names=request.output_path.name,
            use_qrm=self.config.qrm.enabled,
            qrm_start_step=self.qrm_start_step,
            qrm_end_step=self.qrm_end_step,
            trajectory_trace=request.extra.get("trajectory_trace"),
            sigma_policy=policy,
            sigma_controller_options=controller_options,
        )
        return request.output_path
