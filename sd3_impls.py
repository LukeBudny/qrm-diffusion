### Impls of the SD3 core diffusion model and VAE

import math
import re

import einops
from safetensors import safe_open
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from dit_embedder import ControlNetEmbedder
from mmditx import MMDiTX
from typing import Tuple
from transformers import CLIPProcessor, CLIPModel
from qrm.qrm_models import QRMRegistry
import time
import os


#################################################################################################
### MMDiT Model Wrapping
#################################################################################################


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image

# def print_param_dtypes(model, name="MODEL"):
#     print(f"[DEBUG] {name} PARAMETER DTYPES")
#     for n, p in model.named_parameters():
#         print(f"{n}: {p.dtype}")

class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(
        self,
        shift=1.0,
        device=None,
        dtype=torch.float32,
        file=None,
        prefix="",
        control_model_ckpt=None,
        verbose=False,
        qrm_model_checkpoint = None,
        qrm_type="QRMModulatorLatent",
    ):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape
        self._qrm_block_spans = None

        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )
        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )

        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }
        self.diffusion_model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)
        self.control_model = None
        if control_model_ckpt is not None:
            n_controlnet_layers = len(
                list(
                    filter(
                        re.compile(".*.attn.proj.weight").match,
                        control_model_ckpt.keys(),
                    )
                )
            )

            hidden_size = 64 * depth
            num_heads = depth
            head_dim = hidden_size // num_heads
            pooled_projection_size = control_model_ckpt.get_tensor('time_text_embed.text_embedder.linear_1.weight').shape[1]
            if verbose:
                print(
                    f"Initializing ControlNetEmbedder with {n_controlnet_layers} layers, y_in of {pooled_projection_size}"
                )
            self.control_model = ControlNetEmbedder(
                img_size=None,
                patch_size=patch_size,
                in_chans=16,
                num_layers=n_controlnet_layers,
                attention_head_dim=head_dim,
                num_attention_heads=num_heads,
                pooled_projection_size=pooled_projection_size,
                device=device,
                dtype=dtype,
            )         
        self.qrm_type =qrm_type 
        if qrm_type not in QRMRegistry:
            raise ValueError(f"Unknown QRM type: {qrm_type}")
        if qrm_type in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
            dm   = self.diffusion_model
            dev  = next(dm.parameters()).device
            dty  = next(dm.parameters()).dtype

            # 1) Collect adaLN heads in a fixed order
            adaln_modules = []
            for jb in dm.joint_blocks:
                adaln_modules.append(jb.context_block.adaLN_modulation)
                adaln_modules.append(jb.x_block.adaLN_modulation)
            adaln_modules.append(dm.final_layer.adaLN_modulation)
            self._adaln_modules = adaln_modules

            # 2) Compute spans once with a tiny dummy input (NO hooks yet)
            hidden_size = dm.joint_blocks[0].x_block.adaLN_modulation[1].in_features
            dummy_c     = torch.zeros(1, hidden_size, device=dev, dtype=dty)

            spans, offset = [], 0
            with torch.no_grad():
                for mod in self._adaln_modules:
                    m = mod(dummy_c).shape[1]
                    spans.append((offset, offset + m))
                    offset += m

            # 3) Keep only the last K joint blocks + final
            # print("#######################################################")
            # print("Total adaLN blocks:", 2 * len(dm.joint_blocks) + 1)
            num_blocks = 8 if qrm_type == "QRMModulatorLatentV3" else 49
            K = min(num_blocks, len(dm.joint_blocks))        # tune K
            keep_count = 2*K + 1                     # [context, x] per block + final
            self._adaln_modules   = self._adaln_modules[-keep_count:]
            self._qrm_block_spans = spans[-keep_count:]

            # 4) Register hooks once; guard with a flag during collection
            self._adaln_hooks = []
            self._collecting_scale_shift = False
            for mod, (s, e) in zip(self._adaln_modules, self._qrm_block_spans):
                def _hook(mod, inputs, output, start=s, end=e, model_ref=self):
                    # If we’re currently collecting baseline scale/shift, skip adding delta
                    if getattr(model_ref, "_collecting_scale_shift", False):
                        return output
                    qd = getattr(model_ref, "qrm_delta", None)
                    if qd is None:
                        return output
                    return output + qd[:, start:end]
                self._adaln_hooks.append(mod.register_forward_hook(_hook))

            # 5) Instantiate QRM with trimmed spans (once)
            self.qrm = QRMRegistry[qrm_type](self._qrm_block_spans).to(device=dev, dtype=dty)
        else:
            self.qrm = QRMRegistry[qrm_type]()
        self.qrm.to(device=device,dtype=dtype)
        self.qrm_model_checkpoint = qrm_model_checkpoint


    def apply_model(self, x, sigma,t_raw=None, c_crossattn=None, y=None, skip_layers=[], controlnet_cond=None, q_t_training=False,use_qrm = False,**kwargs):
        qrm_time,sd35_time = [],[]
        dtype = next(self.diffusion_model.parameters()).dtype
        dev   = next(self.diffusion_model.parameters()).device
        start_time_qrm = time.time()
        B = y.shape[0]
        if not use_qrm and hasattr(self, "qrm_delta"):
            self.qrm_delta = None
        if not use_qrm:
            q_t = None
            timestep = self.model_sampling.timestep(sigma).float()
        elif q_t_training:
            timestep = self.model_sampling.timestep(sigma).to(next(self.qrm.parameters()).device)
            if t_raw.ndim == 1:
                t_raw = t_raw.expand(B)
            if kwargs.get("qrm_type",self.qrm_type) == "QRMModulatorLatent":
                q_t = self.qrm(t_raw.float(), x.float(), y.float()).cuda()
            elif kwargs.get("qrm_type", self.qrm_type) in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
                t32 = t_raw.to(device=dev, dtype=torch.float32)
                y32 = y.to(device=dev, dtype=torch.float32)
                c_base32 = (self.diffusion_model.t_embedder(t32, dtype=torch.float32) +
                            self.diffusion_model.y_embedder(y32))  # module outputs choose their own dtype; we force fp32 via inputs

                with torch.no_grad():
                    self._collecting_scale_shift = True
                    D_all = self._qrm_block_spans[-1][1]
                    scale_shift32 = torch.zeros(B, D_all, device=dev, dtype=torch.float32)
                    for mod, (s, e) in zip(self._adaln_modules, self._qrm_block_spans):
                        out = mod(c_base32)               # force fp32 inputs -> fp32 numerics inside LN/MLPs
                        scale_shift32[:, s:e] = out.to(torch.float32)
                    self._collecting_scale_shift = False

                finite_mask = torch.isfinite(scale_shift32)
                if not finite_mask.all():
                    bad_count = (~finite_mask).sum().item()
                    nan_count = torch.isnan(scale_shift32).sum().item()
                    inf_count = torch.isinf(scale_shift32).sum().item()
                    print(
                        f"[diag] scale_shift anomalies: "
                        f"bad={bad_count}, nan={nan_count}, inf={inf_count}, "
                        f"min={scale_shift32.min().item()}, max={scale_shift32.max().item()}, "
                        f"mean={scale_shift32.mean().item()}"
                    )

                qrm_in_x = x.to(dev, torch.float32).detach()
                if kwargs.get("qrm_type", self.qrm_type) == "QRMModulatorLatentV2":
                    qrm_delta32 = self.qrm(qrm_in_x, scale_shift32)
                elif kwargs.get("qrm_type", self.qrm_type) in ["QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
                    qrm_delta32 = self.qrm(qrm_in_x, scale_shift32,y.float(),timestep.float())
                self.qrm_delta = qrm_delta32.to(dtype=dtype, device=dev)
                q_t = None

        elif getattr(self, "qrm_inference", False):
            timestep = self.model_sampling.timestep(sigma).to(next(self.qrm.parameters()).device)
            y = y.to(next(self.qrm.parameters()).device)
            if kwargs.get("qrm_type",self.qrm_type) == "QRMModulatorLatent":
                t_raw = t_raw.expand(B)
                q_t = self.qrm(t_raw.float(), x.float(), y.float()).cuda()
            elif kwargs.get("qrm_type", self.qrm_type) in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
                t32 = t_raw.to(device=dev, dtype=torch.float32)
                y32 = y.to(device=dev, dtype=torch.float32)
                c_base32 = (self.diffusion_model.t_embedder(t32, dtype=torch.float32) +
                            self.diffusion_model.y_embedder(y32))  # module outputs choose their own dtype; we force fp32 via inputs

                with torch.no_grad():
                    self._collecting_scale_shift = True
                    D_all = self._qrm_block_spans[-1][1]
                    scale_shift32 = torch.zeros(B, D_all, device=dev, dtype=torch.float32)
                    for mod, (s, e) in zip(self._adaln_modules, self._qrm_block_spans):
                        out = mod(c_base32)               # force fp32 inputs -> fp32 numerics inside LN/MLPs
                        scale_shift32[:, s:e] = out.to(torch.float32)
                    self._collecting_scale_shift = False

                finite_mask = torch.isfinite(scale_shift32)
                if not finite_mask.all():
                    bad_count = (~finite_mask).sum().item()
                    nan_count = torch.isnan(scale_shift32).sum().item()
                    inf_count = torch.isinf(scale_shift32).sum().item()
                    print(
                        f"[diag] scale_shift anomalies: "
                        f"bad={bad_count}, nan={nan_count}, inf={inf_count}, "
                        f"min={scale_shift32.min().item()}, max={scale_shift32.max().item()}, "
                        f"mean={scale_shift32.mean().item()}"
                    )

                qrm_in_x = x.to(dev, torch.float32).detach()  
                if kwargs.get("qrm_type", self.qrm_type) == "QRMModulatorLatentV2":
                    qrm_delta32 = self.qrm(qrm_in_x, scale_shift32)
                elif kwargs.get("qrm_type", self.qrm_type)in ["QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
                    qrm_delta32 = self.qrm(qrm_in_x, scale_shift32,y.float(),timestep.float())

                self.qrm_delta = qrm_delta32.to(dtype=dtype, device=dev)
                q_t = None
        else:
            q_t = None
            timestep = self.model_sampling.timestep(sigma).float()
        
        controlnet_hidden_states = None

        if controlnet_cond is not None:
            y_cond = y.to(dtype)
            controlnet_cond = controlnet_cond.to(dtype=x.dtype, device=x.device)
            controlnet_cond = controlnet_cond.repeat(x.shape[0], 1, 1, 1)

            if not self.control_model.using_8b_controlnet:
                y_cond = self.diffusion_model.y_embedder(y)
            
            x_controlnet = x
            if self.control_model.using_8b_controlnet:
                hw = x.shape[-2:]
                x_controlnet = self.diffusion_model.x_embedder(x) + self.diffusion_model.cropped_pos_embed(hw)
            controlnet_hidden_states = self.control_model(
                x_controlnet, controlnet_cond, y_cond, 1, sigma.to(torch.float32)
            )
        x = x.to(next(self.diffusion_model.parameters()).dtype)

        if kwargs.get("qrm_type",self.qrm_type) in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
            q_t = None

        qrm_time.append(time.time() - start_time_qrm)
        start_time_sd35 = time.time()
        model_output,c,c2,c3 = self.diffusion_model(
            x.to(dtype),
            timestep.to(x.device),
            y=y,
            context=c_crossattn.to(dtype) if c_crossattn is not None else None,
            controlnet_hidden_states=controlnet_hidden_states,
            skip_layers=skip_layers,
            q_t=q_t,
            **kwargs
        )
        model_output = model_output.float()

        if kwargs.get("qrm_type",self.qrm_type) in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"] and use_qrm:
                c = scale_shift32
                c2 = scale_shift32 + self.qrm_delta
                c3 = self.qrm_delta
        sd35_time.append(time.time() - start_time_sd35)
        return self.model_sampling.calculate_denoised(sigma, model_output, x),c,c2,c3,qrm_time,sd35_time
        

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, *args):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        use_qrm = False,
        **kwargs,
    ):
        
        
        batched,c,c2,c3,qrm_time,sd35_time = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            use_qrm = use_qrm,
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled,c,c2,c3,qrm_time,sd35_time


class SkipLayerCFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, steps,vae=None,clip_model=None,clip_processor=None,prompt_text=None, skip_layer_config=None):
        super().__init__()
        self.model = model
        self.steps = steps
        self.slg = skip_layer_config["scale"]
        self.skip_start = skip_layer_config["start"]
        self.skip_end = skip_layer_config["end"]
        self.skip_layers = skip_layer_config["layers"]
        self.step = 0

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        use_qrm = False,
        **kwargs,
    ):

        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            use_qrm = use_qrm,
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        # Then run with skip layer
        if (
            self.slg > 0
            and self.step > (self.skip_start * self.steps)
            and self.step < (self.skip_end * self.steps)
        ):
            skip_layer_out = self.model.apply_model(
                x,
                timestep,
                c_crossattn=cond["c_crossattn"],
                y=cond["y"],
                skip_layers=self.skip_layers,
            )
            # Then scale acc to skip layer guidance
            scaled = scaled + (pos_out - skip_layer_out) * self.slg
        self.step += 1
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor(
            [
                [-0.0645, 0.0177, 0.1052],
                [0.0028, 0.0312, 0.0650],
                [0.1848, 0.0762, 0.0360],
                [0.0944, 0.0360, 0.0889],
                [0.0897, 0.0506, -0.0364],
                [-0.0020, 0.1203, 0.0284],
                [0.0855, 0.0118, 0.0283],
                [-0.0539, 0.0658, 0.1047],
                [-0.0057, 0.0116, 0.0700],
                [-0.0412, 0.0281, -0.0039],
                [0.1106, 0.1171, 0.1220],
                [-0.0248, 0.0682, -0.0481],
                [0.0815, 0.0846, 0.1207],
                [-0.0120, -0.0055, -0.0867],
                [-0.0749, -0.0634, -0.0456],
                [-0.1418, -0.1457, -0.1259],
            ],
            device="cpu",
        )
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (
            ((latent_image + 1) / 2)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
            .byte()
        ).cpu()

        return Image.fromarray(latents_ubyte.numpy())


#################################################################################################
### Samplers
#################################################################################################


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x



# @torch.no_grad()
# @torch.autocast("cuda", dtype=torch.float16)
# def sample_dpmpp_2m(model, x, sigmas, extra_args=None):
#     """DPM-Solver++(2M)."""
#     extra_args = {} if extra_args is None else extra_args
#     s_in = x.new_ones([x.shape[0]])
#     sigma_fn = lambda t: t.neg().exp()
#     t_fn = lambda sigma: sigma.log().neg()
#     old_denoised = None
#     for i in tqdm(range(len(sigmas) - 1)):
#         denoised = model(x, sigmas[i] * s_in, **extra_args)
#         t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
#         h = t_next - t
#         if old_denoised is None or sigmas[i + 1] == 0:
#             x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
#         else:
#             h_last = t - t_fn(sigmas[i - 1])
#             r = h_last / h
#             denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
#             x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
#         old_denoised = denoised
#     return x

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_dpmpp_2m(model, x, sigmas, text_inputs=None, use_qrm=False, extra_args=None):
    extra_args = {} if extra_args is None else extra_args
    s_in  = x.new_ones([x.shape[0]])  # [B]
    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()
    old_denoised = None
    B = x.shape[0]
    qrm_start_step = extra_args.get("qrm_start_step", 25)
    qrm_end_step = extra_args.get("qrm_end_step", 47)
    sampler_time = []

    for i in tqdm(range(len(sigmas) - 1), leave=False):
        # per-step raw time (float32 for stability)
        t_raw = t_fn(sigmas[i]).expand(B).to(dtype=torch.float32, device=x.device)
        
        use_qrm_step = bool(use_qrm) and (i >= qrm_start_step) and (i <= qrm_end_step)

        # strip helper keys; keep only what the UNet expects
        clean_args = {k: v for k, v in extra_args.items()
                      if k not in ("prompt", "uncond_prompt")}

        # move cond to device
        for key in ("cond", "uncond"):
            if key in clean_args and isinstance(clean_args[key], dict):
                for sub in ("c_crossattn", "y"):
                    if sub in clean_args[key] and clean_args[key][sub] is not None:
                        clean_args[key][sub] = clean_args[key][sub].to(x.device)

        # --- denoise (QRM on/off per step) ---
        out, c, c2, c3,qrm_time,sd35_time = model(x, sigmas[i] * s_in, t_raw=t_raw, use_qrm=use_qrm_step, **clean_args)
        start_time_sampler = time.time()
        denoised = out[0] if isinstance(out, (tuple, list)) else out

        # --- DPM++(2M) update ---
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t

        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1/(2*r)) * denoised - (1/(2*r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d

        old_denoised = denoised

        # optional per-step save
        step_num = i + 1
        is_last  = (i == len(sigmas) - 2)
        if bool(extra_args.get("save_per_5_step", False)) and ((step_num % 5 == 0) or is_last):
            save_and_decode_x(
                x,
                extra_args.get("per_step_dir", None),
                extra_args.get("vae_decode", None),
                extra_args.get("process_out", None),
                step_num
            )
        sampler_time.append(time.time() - start_time_sampler)
    
    # print("time spent on qrm: ", sum(qrm_time),"time spent on sd35: ", sum(sd35_time),"time spent on sampler: ", sum(sampler_time))
    return x

def save_and_decode_x(latents,per_step_dir,vae_decode,process_out,step_idx: int,):
    """
    Decode current latents and save a PNG named {step_idx:03d}.png in per_step_dir.
    Assumes batch=1; extend as needed for larger batches.
    """
    os.makedirs(per_step_dir, exist_ok=True)
    latents = process_out(latents)
    # If your decode function expects fp16/cuda, just pass latents through.
    # If it needs fp32/cpu, you could do: latents.float().to('cpu') instead.
    img = vae_decode(latents)  # should return a PIL.Image or list of PIL.Image
    if isinstance(img, (list, tuple)):
        img = img[0]
    out_path = os.path.join(per_step_dir, f"{step_idx:03d}.png")
    img.save(out_path)

def sample_qrm_one_step(model, sigmas_schedule, text_inputs, 
                        extra_args=None, num_selected_steps=1, 
                        step_indices=None, fixed_noise=None, 
                        loss_type="margin_seeking",
                        sampler: str = "dpmpp2m"):  # ← NEW: choose "dpmpp2m" or "euler"
    assert num_selected_steps == 1
    device = "cuda"
    latent_tmpl = torch.empty((1, 16, 64, 64), device=device)
    extra_args = {} if extra_args is None else dict(extra_args)
    B = latent_tmpl.shape[0]
    sigmas = sigmas_schedule.to(device)

    if step_indices is None or (torch.is_tensor(step_indices) and step_indices.numel() == 0) or (isinstance(step_indices, (list, tuple)) and len(step_indices) == 0):
        raise ValueError("step_indices must be provided (single index for RF one-step)")
    if torch.is_tensor(step_indices):
        step_indices = step_indices.tolist()

    idx = int(step_indices[0])

    # --- init from pure noise at sigma_max ----------------------------------
    noise = fixed_noise if fixed_noise is not None else torch.randn_like(latent_tmpl)
    x = noise * sigmas[0].view(1, 1, 1, 1)  # start at highest noise
    x_qrm = x.clone()  

    s_in = torch.ones(B, device=device)

    @torch.no_grad()
    def fwd_base(x_in, sigma_scalar):
        with torch.autocast("cuda"):
            den, _, _, _ = model(
                x_in, sigma_scalar * s_in, use_qrm=False,
                **{k: v for k, v in extra_args.items() if k not in ("prompt", "uncond_prompt")}
            )
        return den  # fp16

    def fwd_qrm(x_in, sigma_scalar, need_stats: bool):
        t_raw = (-sigma_scalar.log()).expand(B)
        with torch.autocast("cuda"):
            den, c, c2, c3,qrm_time,sd35_time = model(
                x_in, sigma_scalar * s_in, t_raw=t_raw, use_qrm=True,
                **{k: v for k, v in extra_args.items() if k not in ("prompt", "uncond_prompt")}
            )
        if need_stats:
            return den, c, c2, c3, t_raw
        else:
            return den, None, None, None, t_raw

    # === DPM++(2M) helpers ===================================================
    sigma_fn = lambda t: t.neg().exp()
    t_fn     = lambda sigma: sigma.log().neg()

    # ---------- BASELINE roll to idx ----------
    if loss_type == "margin_seeking":
        old_denoised = None
        for i in range(0, idx + 1):
            den_i = fwd_base(x, sigmas[i].view(B))  # fp16, no grad

            if sampler.lower() == "euler":
                # Euler: x <- x + d * dt
                d  = to_d(x, sigmas[i], den_i)  # uses your existing helper
                dt = sigmas[i + 1] - sigmas[i]
                x  = x + d * dt
            else:
                # DPM++(2M) (unchanged)
                t      = t_fn(sigmas[i])
                t_next = t_fn(sigmas[i + 1])
                h = t_next - t
                if old_denoised is None or sigmas[i + 1] == 0:
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * den_i
                else:
                    h_last = t - t_fn(sigmas[i - 1])
                    r = h_last / h
                    den_d = (1 + 1/(2*r)) * den_i - (1/(2*r)) * old_denoised
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * den_d
                    del den_d

            old_denoised = den_i

        x_k_baseline = x.detach()

    # ---------- QRM roll to idx (no grad until idx) ----------
    old_denoised_qrm = None
    den_for_qrm_qrm  = None
    for i in range(0, idx + 1):
        with torch.no_grad():
            den_i_qrm, _, _, _, _ = fwd_qrm(x_qrm, sigmas[i].view(B), need_stats=False)

            if sampler.lower() == "euler":
                d  = to_d(x_qrm, sigmas[i], den_i_qrm)
                dt = sigmas[i + 1] - sigmas[i]
                x_qrm = x_qrm + d * dt
            else:
                t      = t_fn(sigmas[i])
                t_next = t_fn(sigmas[i + 1])
                h = t_next - t
                if old_denoised_qrm is None or sigmas[i + 1] == 0:
                    x_qrm = (sigma_fn(t_next) / sigma_fn(t)) * x_qrm - (-h).expm1() * den_i_qrm
                else:
                    h_last = t - t_fn(sigmas[i - 1])
                    r = h_last / h
                    den_d_qrm = (1 + 1/(2*r)) * den_i_qrm - (1/(2*r)) * old_denoised_qrm
                    x_qrm = (sigma_fn(t_next) / sigma_fn(t)) * x_qrm - (-h).expm1() * den_d_qrm
                    del den_d_qrm

            if i == idx - 1:
                x_pre_k_qrm     = x_qrm.detach()
                den_for_qrm_qrm = den_i_qrm

            old_denoised_qrm = den_i_qrm

    # --- states at k ----------------------------------------------------------
    sigma_t = sigmas[idx].view(B)

    # # --- single step (grad ON at idx for QRM) --------------------------------
    den_k, c, c2, c3, t_raw = fwd_qrm(x_pre_k_qrm, sigma_t, need_stats=True)

    if sampler.lower() == "euler":
        d  = to_d(x_pre_k_qrm, sigmas[idx], den_k)
        dt = sigmas[idx + 1] - sigmas[idx]
        x_k = x_pre_k_qrm + d * dt
    else:
        t      = t_fn(sigmas[idx])
        t_next = t_fn(sigmas[idx + 1])
        h = t_next - t
        if den_for_qrm_qrm is None or sigmas[idx + 1] == 0:
            x_k = (sigma_fn(t_next) / sigma_fn(t)) * x_pre_k_qrm - (-h).expm1() * den_k
        else:
            h_last = t - t_fn(sigmas[idx - 1]) if idx > 0 else h
            r = h_last / h
            den_d = (1 + 1/(2*r)) * den_k - (1/(2*r)) * den_for_qrm_qrm
            x_k = (sigma_fn(t_next) / sigma_fn(t)) * x_pre_k_qrm - (-h).expm1() * den_d
            del den_d

    sigma_next = sigmas[idx + 1].view(B)

    with torch.autocast("cuda"):
        den_next_qrm, _, _, _,qrm_time,sd35_time = model(
            x_k, sigma_next * s_in,
            t_raw=(-sigma_next.log()).expand(B),
            use_qrm=True,
            **{k: v for k, v in extra_args.items() if k not in ("prompt", "uncond_prompt")}
        )

    if loss_type == "margin_seeking":
        with torch.no_grad():
            den_next_base = fwd_base(x_k_baseline, sigma_next)

    if not isinstance(c2, int):
        c  = c.detach()
        c2 = c2.detach()
        c3 = c3.detach()

    if loss_type == "margin_seeking":
        return den_next_qrm, den_next_base, c, c2, c3, idx
    elif loss_type == "reward_maximization":
        return den_next_qrm, _, c, c2, c3, idx
    else:
        raise Exception("undentified loss")
    
#################################################################################################
### VAE
#################################################################################################


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


class ResnetBlock(torch.nn.Module):
    def __init__(
        self, *, in_channels, out_channels=None, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(),
            (q, k, v),
        )
        hidden = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dtype=dtype,
                        device=device,
                    )
                )
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        resolution=256,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dtype=dtype,
                        device=device,
                    )
                )
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

