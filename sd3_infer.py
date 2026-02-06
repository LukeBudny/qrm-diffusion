# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)
import time
import datetime
import math
import os
import re
import fire
import numpy as np
import sd3_impls
import torch
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from PIL import Image
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
from transformers import CLIPTokenizer, CLIPModel
import time
import torch




#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e
            
CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cuda"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cuda",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cuda", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device=device
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu",qrm_model_checkpoint=None, qrm_type = 'QRMModulatorLatent'
    ):

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            if control_model_file is not None:
                control_model_ckpt = safe_open(
                    control_model_file, framework="pt", device=device
                )
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
                qrm_model_checkpoint=qrm_model_checkpoint,
                qrm_type = qrm_type,
            ).eval()
            load_into(f, self.model, "model.", "cuda", torch.float16)
        if control_model_file is not None:
            control_model_ckpt = safe_open(
                control_model_file, framework="pt", device=device
            )
            self.model.control_model = self.model.control_model.to(device)
            load_into(
                control_model_ckpt,
                self.model.control_model,
                "",
                device,
                dtype=torch.float16,
                remap=CONTROLNET_MAP,
            )

            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet
        control_model_ckpt = None


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


#################################################################################################
### Main inference logic
#################################################################################################


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 512
HEIGHT = 512
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
MODEL = "models/sd3_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
# MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"
import torch.nn.functional as F

CLIP_MEAN = torch.tensor([0.4815, 0.4578, 0.4082], dtype=torch.float32).view(1, 3, 1, 1).cuda()
CLIP_STD  = torch.tensor([0.2686, 0.2613, 0.2758], dtype=torch.float32).view(1, 3, 1, 1).cuda()

CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "models/sd3.5_medium.safetensors": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "models/sd3.5_large_turbo.safetensors": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}



class SD3Inferencer:

    def __init__(self):
        self.verbose = False
        self.inference = False

    def print(self, txt):
        if self.verbose:
            print(txt)
       
    def load(
        self,
        model=MODEL,
        vae=VAEFile,
        shift=SHIFT,
        controlnet_ckpt=None,
        model_folder: str = MODEL_FOLDER,
        text_encoder_device: str = "cpu",
        verbose=False,
        load_tokenizers: bool = True,
        load_non_tokenizers: bool = True,
        qrm_model_checkpoint = None,
        eval_model: str = "clip",
        inference: bool = False,
        qrm_type = "QRMModulatorLatent",
        config = CONFIGS
    ):
        
        self.verbose = verbose
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading tokenizers...")
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, "cpu", torch.float32)
            # self.t5xxl = T5XXL_new(text_encoder_device)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, "cuda")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        if load_non_tokenizers:
            print("Loading VAE model...")
            self.vae = VAE(vae or model)
            self.vae.model.decoder = self.vae.model.decoder.to("cuda").eval()
            self.eval_model = eval_model.lower()
            print(f"Using vision feature model: {self.eval_model}")
            if self.eval_model in ["clip", "raw_clip", "clipscore","all"]:
                print("Loading CLIP model...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()


            # if self.eval_model in ["ir","irscore","all"]:
            #     print("Loading ImageReward model...")
            #     import ImageReward as reward
            #     self.reward_model = reward.load("ImageReward-v1.0").to("cuda").eval()
            if self.eval_model in ["hps","hpsscore","all"]:
                print("Preparing HPS v2.1 scorer...")
                # self.hps_version = "v2.1"

            # if self.eval_model not in ["clip", "clipscore","hpsscore","irscore", "raw_clip","ir","hps","all"]:
            #     raise ValueError(f"Unsupported vision_feature_model: {self.eval_model}")
            
            print(f"Loading SD3 model {os.path.basename(model)}...")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.qrm_model_checkpoint = qrm_model_checkpoint
            self.model_name = model
            self.configs = config
            self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, "cuda",qrm_model_checkpoint=self.qrm_model_checkpoint,qrm_type=qrm_type)
            # self.sd3.model.to(device="cuda", dtype=torch.float16).eval()
            self.qrm_type = self.sd3.model.qrm_type
            self.denoiser = CFGDenoiser(self.sd3.model)
            # print("before force half")
            # time.sleep(10)
            # # self.sd3.model.half().eval()
            # # self.vae.model.decoder = self.vae.model.decoder.to("cuda", dtype=torch.float16).eval()
            # print("after force half")
            # time.sleep(10)


    def get_denoiser(self):
        """Returns the initialized CFGDenoiser"""
        return self.denoiser
    
    def get_vae(self):
        return self.vae
        
    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        if self.inference:
            t5_out, _ = self.t5xxl.encode_token_weights(tokens["t5xxl"], return_pooled=True)
        else:
            t5_out, _ = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        device = t5_out.device
        lg_out = torch.cat([l_out, g_out], dim=-1).to(device)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled.to(device), g_pooled.to(device)), dim=-1
        )
    
    def _batch_encode_token_weights(self,encoder,name, token_weight_batch):
        """
        token_weight_batch : list[ token_weight_pairs ]  (one prompt per entry)
        returns            : hidden_states, pooled  (both fp16 / CUDA)
        """
        batch_tokens = [[tok for tok, _ in tw_pairs[0]]
                        for tw_pairs in token_weight_batch]      # list[list[int]]

        if name == "t5":
            PAD, MAX_LEN = 0, 77
            padded = [seq[:MAX_LEN] + [PAD] * (MAX_LEN - len(seq)) for seq in batch_tokens]

            hidden, pooled = encoder(padded)   # fp32
            hidden = hidden.to("cuda", non_blocking=True)
            if pooled is not None:
                pooled = pooled.to("cuda", non_blocking=True)
        else:
            
            hidden, pooled = encoder(batch_tokens)   # fp32
            hidden = hidden.to("cuda", non_blocking=True)
            if pooled is not None:
                pooled = pooled.to("cuda", non_blocking=True)

        # slice back to per-prompt tensors
        out_list    = list(hidden.unbind(0))                 # len == B
        pooled_list = ([None] * len(out_list) if pooled is None
                    else list(pooled.unbind(0)))
        return out_list, pooled_list
    
    def get_cond_batch(self, prompts: list[str]):
        """
        Vectorised text-conditioning for a whole prompt batch.
        Returns:
            {
            "c_crossattn": Float16 CUDA tensor  [B , L_total , 4096],
            "y"          : Float16 CUDA tensor  [B , 2048]
            }
        """
        # --- tokenise all prompts once ---------------------------------
        toks = [self.tokenizer.tokenize_with_weights(p) for p in prompts]

        # Pull out the token-weight groups per encoder
        l_batch  = [t["l"]     for t in toks]
        g_batch  = [t["g"]     for t in toks]
        t5_batch = [t["t5xxl"] for t in toks]

        # --- batched forward passes ------------------------------------
        l_outs, l_pooled = self._batch_encode_token_weights(self.clip_l.model, "n/a", l_batch)
        g_outs, g_pooled = self._batch_encode_token_weights(self.clip_g.model,"n/a" , g_batch)
        t_outs, _        = self._batch_encode_token_weights(self.t5xxl.model,"t5",   t5_batch)

        # --- merge encoder streams -------------------------------------
        lg_pad = []
        cross  = []
        for l, g, t in zip(l_outs, g_outs, t_outs):
            lg     = torch.cat([l, g], dim=-1)                  # [L , 768+1280]
            lg_pad.append(torch.nn.functional.pad(lg, (0, 4096 - lg.shape[-1])))
            cross.append(torch.cat([lg_pad[-1], t], dim=-2))    # [L+L_t5 , 4096]

        c_crossattn = torch.stack(cross)                # [B , L_tot , 4096]
        pooled      = torch.cat([
                torch.stack([p.squeeze(0) for p in l_pooled]),
                torch.stack([p.squeeze(0) for p in g_pooled])
        ], dim=-1)

        return {"c_crossattn": c_crossattn, "y": pooled}

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}
    
    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        controlnet_cond=None,
        denoise=1.0,
        skip_layer_config={},
        prompt=None,
        use_qrm = False,
        save_per_5_step=False,
        per_step_dir=None,
        qrm_start_step = 25,
        qrm_end_step = 47) -> torch.Tensor:
        
        if not use_qrm:
            print("I am not using QRM!")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        # NEW: accept dicts that are already { "c_crossattn": ..., "y": ... }
        def _ensure_fixed(x):
            if isinstance(x, dict):
                d = {}
                for k, v in x.items():
                    if torch.is_tensor(v):
                        d[k] = v.to(device="cuda")
                    else:
                        d[k] = v
                return d
            else:
                return self.fix_cond(x)  # old path: tuple -> dict

        conditioning = _ensure_fixed(conditioning)
        neg_cond     = _ensure_fixed(neg_cond)
            
        extra_args = {
                "cond": conditioning,
                "uncond": neg_cond,
                "cond_scale": cfg_scale,
                "controlnet_cond": controlnet_cond,
                "prompt":prompt,
                "save_per_5_step":save_per_5_step,
                "per_step_dir":per_step_dir,
                "vae_decode":self.vae_decode,
                "process_out":SD3LatentFormat().process_out,
                "qrm_start_step":qrm_start_step,
                "qrm_end_step":qrm_end_step,
                "qrm_type":self.qrm_type,
                }
        
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )

        sample_fn = getattr(sd3_impls, f"sample_{sampler}")

        denoiser = (
            SkipLayerCFGDenoiser(self.sd3.model, steps, skip_layer_config)
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser(self.sd3.model)
        )
        # start = time.time()
        if isinstance(prompt, str):
            prompt = [prompt]
        # text_inputs = self.clip_tokenizer(prompt + [""], return_tensors="pt", padding=True, truncation=True)
        # text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
        text_inputs = ""

        if save_per_5_step:
            os.makedirs(per_step_dir, exist_ok=True)


        latent = sample_fn(denoiser, noise_scaled, sigmas, text_inputs,use_qrm=use_qrm, extra_args=extra_args)
        latent = SD3LatentFormat().process_out(latent)
        return latent, conditioning

    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch)
        # self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent.cpu()

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def gen_image(
        self,
        prompts=[PROMPT],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        controlnet_cond_image=CONTROLNET_COND_IMAGE,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
        save_names=None,
        cached_conds=None,          # NEW: list[dict], len == len(prompts)
        cached_uncond=None,          # NEW: dict with keys {'c_crossattn','y'}
        use_qrm = False,
        save_per_5_step=False, 
        per_step_dir=None,
        qrm_start_step = 25,
        qrm_end_step = 47
    ):
        controlnet_cond = None
        if init_image:
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        if controlnet_cond_image:
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
        neg_cond = cached_uncond if cached_uncond is not None else self.get_cond("")
        seed_num = None
        # pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        # for i, prompt in pbar:
        if seed_type == "roll":
            seed_num = seed if seed_num is None else seed_num + 1
        elif seed_type == "rand":
            seed_num = torch.randint(0, 100000, (1,)).item()
        else:  # fixed
            seed_num = seed
        if cached_conds is not None:
            conditioning = cached_conds
        else:
            conditioning = self.get_cond(prompts)
        outputs = self.do_sampling(
            latent,
            seed_num,
            conditioning,
            neg_cond,
            steps,
            cfg_scale,
            sampler,
            controlnet_cond,
            denoise if init_image else 1.0,
            skip_layer_config,
            prompt = prompts,
            use_qrm = use_qrm,
            save_per_5_step=save_per_5_step,         
            per_step_dir=per_step_dir,
            qrm_start_step = qrm_start_step,
            qrm_end_step = qrm_end_step
        )

        sampled_latent, conditioning = outputs

        image = self.vae_decode(sampled_latent)
        name = save_names
        save_path = os.path.join(out_dir, name)
        self.print(f"Saving to to {save_path}")
        image.save(save_path)
        self.print("Done")





@torch.no_grad()
def main(
    prompt=PROMPT,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    controlnet_cond_image=None,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
    model_folder=MODEL_FOLDER,
    text_encoder_device="cpu",
    qrm_checkpoint = None,
    **kwargs,
):
    assert not kwargs, f"Unknown arguments: {kwargs}"

    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _cfg = cfg or config.get("cfg", 5)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")


    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    if controlnet_ckpt is not None:
        controlnet_config = CONFIGS.get(
            os.path.splitext(os.path.basename(controlnet_ckpt))[0], {}
        )
        _shift = shift or controlnet_config.get("shift", shift)
        _steps = steps or controlnet_config.get("steps", steps)
        _cfg = cfg or controlnet_config.get("cfg", cfg)
        _sampler = sampler or controlnet_config.get("sampler", sampler)

    inferencer = SD3Inferencer()
    print("first ################################",qrm_checkpoint)
    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        model_folder,
        text_encoder_device,
        verbose,
        qrm_model_checkpoint=qrm_checkpoint
    )

    if isinstance(prompt, str):
        if os.path.splitext(prompt)[-1] == ".txt":
            with open(prompt, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
        else:
            prompts = [prompt]

    sanitized_prompt = re.sub(r"[^\w\-\.]", "_", prompt)
    out_dir = os.path.join(
        out_dir,
        (
            os.path.splitext(os.path.basename(model))[0]
            + (
                "_" + os.path.splitext(os.path.basename(controlnet_ckpt))[0]
                if controlnet_ckpt is not None
                else ""
            )
        ),
        os.path.splitext(os.path.basename(sanitized_prompt))[0][:50]
        + (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
    )

    os.makedirs(out_dir, exist_ok=False)

    inferencer.gen_image(
        prompts,
        width,
        height,
        _steps,
        _cfg,
        _sampler,
        seed,
        seed_type,
        out_dir,
        controlnet_cond_image,
        init_image,
        denoise,
        skip_layer_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
