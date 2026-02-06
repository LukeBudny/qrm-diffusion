import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import json
import time
import numpy as np
from qrm.qrm_models import QRMRegistry
from transformers import get_cosine_schedule_with_warmup
from sd3_impls import sample_qrm_one_step,sample_dpmpp_2m, CFGDenoiser,SD3LatentFormat
from torch.utils.data import Dataset
import torch,csv, math
import os
from pathlib import Path
import bisect
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
import re, hashlib
from torch.distributions import Beta

os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silence warning



def identity_collate(x):
    # DataLoader calls collate on a list of items; we want the single item back
    return x[0]

class FlatFileDatasetLazy(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("batch_*.pt"))
        self.lengths = []
        for f in self.files:
            b = torch.load(f, map_location="cpu")
            self.lengths.append(len(b))
        self.cum = []
        s = 0
        for L in self.lengths:
            s += L
            self.cum.append(s)
        self._cache_file_idx = None
        self._cache_batch = None

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    def _load_file(self, f_idx: int):
        if self._cache_file_idx != f_idx:
            b = torch.load(self.files[f_idx], map_location="cpu")
            self._cache_file_idx = f_idx
            self._cache_batch = b
        return self._cache_batch

    def __getitem__(self, idx: int):
        # find file containing idx
        f_idx = bisect.bisect_right(self.cum, idx)
        prev_cum = self.cum[f_idx - 1] if f_idx > 0 else 0
        inner = idx - prev_cum

        batch = self._load_file(f_idx)

        # stable key order (dict preserves insertion in Python 3.7+, but we make it explicit):
        keys = list(batch.keys())
        prompt = keys[inner]
        cond = batch[prompt]
        return prompt, cond
    
def pad_to_match(a, b):
    diff = b.size(1) - a.size(1)
    if diff > 0:
        pad = torch.zeros(a.size(0), diff, a.size(2), dtype=a.dtype, device=a.device)
        return torch.cat([a, pad], dim=1)
    elif diff < 0:
        return a[:, :b.size(1), :]
    return a

class QRMTrainer_batches:

    def __init__(self, inferencer, device="cuda", lr=1e-5, qrm_checkpoint_path=None,time_bool=True,qrm_type="mlp",
                num_epochs = 10,subset_size = 256,batch_size = 1,accum_steps = 4,use_scheduler=False,collate_fn=None):
        self.device = device
        self.inferencer = inferencer
        from sd3_impls import BaseModel,SD3LatentFormat
        self.latent_fmt = SD3LatentFormat()
        self.sd_model: BaseModel = inferencer.sd3.model.to(device)
        self.collate_fn = collate_fn
        self.time_bool = time_bool
        self.qrm_type = qrm_type
        if qrm_type in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
            self.sd_model.qrm = QRMRegistry[self.qrm_type](self.sd_model._qrm_block_spans).to(device)
        else:
            self.sd_model.qrm = QRMRegistry[self.qrm_type]().to(device)
        self.tokenizer = inferencer.tokenizer
        self.lr = lr
        self.vae = self.inferencer.vae.model

        self.val_prompts = ["an elder politician giving a campaign speech","the word 'START' written in chalk on a sidewalk","a chess queen to the right of a chess knight","a view of the Big Dipper in the night sky",
    "Four deer surrounding a moose.","five chairs","matching socks with cute cats on them","The Oriental Pearl in oil painting","a plate with white rice topped by cooked vegetables","a scientist",
    "a yellow wall with the word KA-BOOM on it","a grumpy porcupine handing a check for $10,000 to a smiling peacock","Three-quarters front view of a yellow 2017 Corvette coming around a curve in a mountain road and looking over a green valley on a cloudy day.",
    "a t-shirt with Carpe Diem written on it","five frosted glass bottles","a can of Spam on an elegant plate","Portrait of a tiger wearing a train conductor's hat and holding a skateboard that has a yin-yang symbol on it. charcoal sketch",
    "a helicopter hovering over Times Square","A bowl of soup that looks like a monster knitted out of wool","a glass of orange juice with an orange peel stuck on the rim","a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
    "an abstract painting of a house on a mountain","a white robot passing a soccer ball to a red robot","a man chasing a cat","an airplane flying into a cloud that looks like monster","the Mona Lisa in the style of Minecraft","a man with puppet that looks like a king",
    "A photo of a Ming Dynasty vase on a leather topped table.","a portrait of a postal worker who has forgotten their mailbag","a chair"]

        for p in self.sd_model.parameters():
            p.requires_grad_(False)
        for p in self.sd_model.qrm.parameters():
            p.requires_grad_(True)

        # encoder on GPU float32
        self.vae.encoder.to(self.device).eval()
        self.vae.decoder.to(self.device).eval()

        self.vae.decoder.to(memory_format=torch.channels_last)
        if self.inferencer.eval_model == "clip":
            self.inferencer.clip_model.vision_model.to(memory_format=torch.channels_last)
            try:
                self.inferencer.clip_model.vision_model.gradient_checkpointing_enable()
            except Exception:
                pass
            self.inferencer.clip_model.eval()
            for mp in (self.inferencer.clip_model.parameters()):
                mp.requires_grad_(False)
 
        self.vae.decoder.eval()
        for mp in (self.vae.decoder.parameters()):
                mp.requires_grad_(False)


        #  Load QRM weights if provided
        if qrm_checkpoint_path is not None:
            print(f"📦 Loading QRM weights from: {qrm_checkpoint_path}")
            checkpoint = torch.load(qrm_checkpoint_path, map_location=self.device)
            self.sd_model.qrm.load_state_dict(checkpoint["model"])

        self.qrm_checkpoint_path = qrm_checkpoint_path

        param_groups = [p for p in self.sd_model.qrm.parameters() if p.requires_grad]
                

        proj = self.sd_model.diffusion_model.x_embedder.proj

        print("[DEBUG] Before fix: proj weight dtype =", proj.weight.dtype,
            "bias dtype =", proj.bias.dtype)

        proj.weight = torch.nn.Parameter(proj.weight.detach())
        if proj.bias is not None:
            proj.bias = torch.nn.Parameter(proj.bias.detach())

        print("[DEBUG] After fix: proj weight dtype =", proj.weight.dtype,
            "bias dtype =", proj.bias.dtype)

        
        self.sd_model.qrm.train()

        self.optimizer = optim.Adam(param_groups, lr=lr, weight_decay=0.01)

        self.num_epochs = num_epochs
        # self.full_dataset = full_dataset
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        
        self.scheduler = None
        if use_scheduler:
            # total steps = number of steps in 1 epoch * number of epochs
            self.total_steps = num_epochs * (subset_size//accum_steps)

            warmup_ratio = 0.05
            warmup_steps = int(warmup_ratio * num_epochs * (subset_size // accum_steps))


            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps
            )
        self.use_scheduler = use_scheduler
        self.uncond = torch.load("qrm/precomputed_prompt_folder/cached_ir_prompts/uncond.pt", map_location=self.device)
        self.scaler = torch.amp.GradScaler('cuda')

        # --- adaptive margin state ---
        self.m_ema   = 0.0    # EMA of positive deltas
        self.m_alpha = 0.03   # EMA smoothing
        self.m_base  = 0.04   # baseline target improvement
        self.m_gain  = 0.60    # how much ema lifts the target
        self.m_min   = 0.03
        self.m_max   = 0.15

    def reward_score(self, images_rgb, prompts, scorer: str = "clip", hps_version: str = "v2.1",allow_grad: bool = True):
        """
        scorer: "clip" (default) or "hps"
        Returns a 1D tensor [B] on images_rgb.device.
        """
        device = "cuda"
        B = images_rgb.shape[0]

        # normalize prompts to list of length B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        elif isinstance(prompts, list) and len(prompts) == 1 and B > 1:
            prompts = prompts * B

        # ---- your existing differentiable CLIP path (unchanged) ----
        def img_fwd(pv):
            return self.inferencer.clip_model.get_image_features(pixel_values=pv)

        image_size = 224
        if images_rgb.shape[-2:] != (image_size, image_size):
            pixel_values = F.interpolate(images_rgb, size=(image_size, image_size),
                                        mode="bicubic", align_corners=False, antialias=True)
        else:
            pixel_values = images_rgb
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        pixel_values = (pixel_values - mean) / std

        if scorer.lower() == "clip":
            with torch.no_grad():
                toks = self.inferencer.clip_tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                txt_e = self.inferencer.clip_model.get_text_features(**toks)
                txt_n = txt_e / txt_e.norm(dim=-1, keepdim=True)

            img_e = img_fwd(pixel_values)
            img_n = img_e / img_e.norm(dim=-1, keepdim=True)
            return (img_n * txt_n).sum(dim=-1)
        elif scorer.lower() in ("hps", "hpsv2", "hps2.1", "hpsv2.1"):
            import hpsv2.img_score as hps_mod
            import huggingface_hub
            from hpsv2.utils import hps_version_map

            # ensure cached model exists
            hps_mod.initialize_model()
            model = hps_mod.model_dict["model"]

            # devices/dtypes for each tower
            vis_dev   = next(model.visual.parameters()).device
            vis_dtype = next(model.visual.parameters()).dtype
            # pick a text-encoder param to get its device (CPU in your setup)
            txt_dev = model.token_embedding.weight.device  # or next(model.transformer.parameters()).device

            # ensure the requested version is loaded once
            if getattr(hps_mod, "_loaded_hps_version", None) != hps_version:
                cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
                state = torch.load(cp, map_location=vis_dev)
                model.load_state_dict(state["state_dict"])
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                if getattr(hps_mod, "_cached_tokenizer", None) is None:
                    from hpsv2.src.open_clip import get_tokenizer as _get_tok
                    hps_mod._cached_tokenizer = _get_tok("ViT-H-14")
                hps_mod._loaded_hps_version = hps_version

            # text: run on the text device (CPU), no grad
            with torch.no_grad():
                toks = hps_mod._cached_tokenizer(prompts).to(txt_dev, non_blocking=True)
                txt_feat = model.encode_text(toks)            # on txt_dev
                txt_feat = F.normalize(txt_feat, dim=-1)

            # image: run on visual device (CUDA fp16), keep grad if you need it
            img_in = pixel_values.to(vis_dev, dtype=vis_dtype, non_blocking=True)
            img_in = img_in.contiguous(memory_format=torch.channels_last)  # <- removed stray 's'
            
            if not allow_grad:
                # Baseline branch: no grad
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=img_in.is_cuda):
                    img_feat = model.encode_image(img_in)
                img_feat = F.normalize(img_feat, dim=-1)
            else:
                # QRM branch: grad ON
                img_in = img_in.clone()

                def _hps_img(xx):
                    with torch.cuda.amp.autocast(enabled=xx.is_cuda):
                        return model.encode_image(xx)

                img_feat = _hps_img(img_in)
                img_feat = F.normalize(img_feat, dim=-1)

            # do cosine on the image device
            txt_feat = txt_feat.to(img_feat.device, dtype=img_feat.dtype, non_blocking=True)
            scores = (img_feat * txt_feat).sum(dim=-1)
            return scores.to(device=device, dtype=images_rgb.dtype)
        else:
            raise ValueError(f"Unknown scorer '{scorer}'. Use 'clip' or 'hps'.")

    @torch.no_grad()
    def run_qrm_validation(
        self,
        model_tag,
        epoch,
        cfg_scale,
        qrm_type,
        qrm_start_step:int,
        qrm_end_step:int,
        out_dir = "validation_training_images/",
        device="cuda",
        hps_version="v2.1",
        baseline_run = False
    ):
        # print("starting validation run")
        model_out_dir = os.path.join(out_dir, model_tag+"_e"+str(epoch))
        os.makedirs(model_out_dir, exist_ok=True)
        baseline_out_dir = os.path.join(out_dir, "Baseline")
        os.makedirs(baseline_out_dir, exist_ok=True)
        self.inferencer.sd3.model.qrm_inference = True

        def pil_to_bchw_uint801(pil_img, device, dtype=torch.float32):
            # PIL -> [1,3,H,W] float in [0,1]
            arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # BCHW
            return t.to(device=device, dtype=dtype, non_blocking=True)
        
        def tensor_to_pil(x: torch.Tensor) -> Image.Image:
            if x.dim() == 4:
                x = x[0]  # drop batch
            x = x.clamp(0,1).cpu()
            to_pil = T.ToPILImage()
            return to_pil(x)

        # ---------- load HPSv2 once ----------
        import hpsv2.img_score as hps_mod
        hps_mod.initialize_model()

        cache = torch.load("qrm/precomputed_prompt_folder/cached_validation_prompts/batch_validation.pt", map_location="cpu")
        uncond = torch.load("qrm/precomputed_prompt_folder/cached_validation_prompts/uncond.pt", map_location="cpu")
        seed = 32

        model = CFGDenoiser(self.sd_model)

        rows = [("model_tag","epoch","idx","prompt","score_base","score_qrm","delta")]
        for i, prompt in enumerate(self.val_prompts[:30]):

            base_name = f"{i:03d}_base.png"
            base_path = os.path.join(baseline_out_dir, base_name)
            qrm_name = f"{i:03d}_qrm.png"
            qrm_path  = os.path.join(model_out_dir, qrm_name)

            rec = cache[prompt]                         # "c_crossattn": [Lc, D], "y": [Dy]
            cond_ref = rec["c_crossattn"].unsqueeze(0)  # [1, Lc, D]  <-- good reference

            # --- build per-prompt uncond_i without mutating the base dict ---
            uc = pad_to_match(uncond["c_crossattn"], cond_ref)  # [1, Lc, D], still on CPU
            uncond_i = {
                "c_crossattn": uc.to(device).contiguous(),
                "y":           uncond["y"].to(device).contiguous()
            }

            # --- cond_i with batch dim + device ---
            c_attn = rec["c_crossattn"].unsqueeze(0).to(device).contiguous()  # [1, Lc, D]
            y_vec  = rec["y"].unsqueeze(0).to(device).contiguous()            # [1, Dy]
            cond_i = {"c_crossattn": c_attn, "y": y_vec}

            extra_args = {
                    "cond":   cond_i,
                    "uncond": uncond_i,
                    "cond_scale":     cfg_scale,
                    "prompt":           prompt,
                    "uncond_prompt":    [""],
                    "q_t_training":     False,
                    "controlnet_cond": None,
                    "qrm_type": qrm_type,
                    "qrm_start_step":qrm_start_step,
                    "qrm_end_step":qrm_end_step,
                    "save_per_5_step":False
                }
            
            need_base = not os.path.exists(base_path)
            need_qrm  = not os.path.exists(qrm_path)

            if need_base or need_qrm:

                latent = self.inferencer.get_empty_latent(1, 512, 512, seed, "cpu").cuda()
                latent = latent.cuda()
                noise = self.inferencer.get_noise(seed, latent).cuda()
                sigmas = self.inferencer.get_sigmas(self.inferencer.sd3.model.model_sampling, 50).cuda()
                sigmas = sigmas[int(50 * (1 - 1.0)) :]
                noise_scaled = self.inferencer.sd3.model.model_sampling.noise_scaling(sigmas[0], noise, latent, self.inferencer.max_denoise(sigmas))
            
            if need_base:
                # -------- baseline image --------
                img_base = sample_dpmpp_2m(
                    model=model,
                    x=noise_scaled,                       # or your latent init
                    sigmas=sigmas,
                    extra_args=extra_args,
                    use_qrm = False
                )
                img_base = SD3LatentFormat().process_out(img_base)
                img_base = self.inferencer.vae.model.decode(img_base)
                img_base = torch.clamp((img_base + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
                img_base = tensor_to_pil(img_base)
                img_base.save(os.path.join(baseline_out_dir, base_name))
            else:
                img_base = Image.open(base_path).convert("RGB")

            if need_qrm:
                # -------- QRM image (only at start step) --------
                img_qrm = sample_dpmpp_2m(
                    model=model,
                    x=noise_scaled,
                    sigmas=sigmas,
                    extra_args=extra_args,
                    use_qrm = True
                )
                img_qrm = SD3LatentFormat().process_out(img_qrm)
                img_qrm = self.inferencer.vae.model.decode(img_qrm)
                img_qrm = torch.clamp((img_qrm + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
                img_qrm = tensor_to_pil(img_qrm)
                img_qrm.save(os.path.join(model_out_dir, qrm_name))
            else:
                img_qrm = Image.open(qrm_path).convert("RGB")

            # ----- HPS scores -----
            sq = float(self.reward_score(pil_to_bchw_uint801(img_qrm,device), prompt,scorer = "hps", hps_version="v2.1",allow_grad=False))
            sb = float(self.reward_score(pil_to_bchw_uint801(img_base,device), prompt,scorer = "hps", hps_version="v2.1",allow_grad=False))
            delta = sq - sb
            rows.append((model_tag,epoch,i, prompt, sb, sq, delta))
            # keep VRAM stable per-iter
            torch.cuda.empty_cache()

        # ----- write CSV and print summary -----
        csv_path = os.path.join(out_dir, "val_hps_qrm_vs_base.csv")
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(rows[0])       # write header once
            w.writerows(rows[1:])         # append only new data rows


        deltas = [r[5] for r in rows[1:]]
        mean_delta = sum(deltas)/max(1,len(deltas))
        print(f"[VAL] mean HPS delta over {len(deltas)} prompts: {mean_delta:+.4f}")
        self.inferencer.sd3.model.qrm_inference = False

    
    def train(self, cfg_scale=4.5,start_epoch=0,test = False,model_tag='unnamed?',warmup_q_t_steps=0,
              ignore_sigma=False,start=25,end=47,loss_type = "reward_maximization",lr_range=False):
        print("batch_size ", self.batch_size)
        epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list,c5_stats_list,c7_stats_list = [],[],[],[],[],[],[],[]
        overall_start = time.time()
        self.reward_model = None
        self.hps_version = "v2.1"
        opt_loss_accum = 0.0

        training_start_timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(lr_range)
        if lr_range:
            # Finder knobs (all local here; no extra function args)
            lr_min = 1e-6            # widen if your curve is too flat early
            lr_max = 1e-3            # 1e-3 gives a clearer elbow for your QRM
            lr_steps = 350           # optimizer steps (not microsteps)
            ema_alpha = 0.05         # heavier smoothing for BS=1
            abort_factor = 300.0      # early abort if loss >> recent median
            lr_csv = f"lr_results/{training_start_timestamp}_{model_tag}_lr_range.csv"

            os.makedirs(os.path.dirname(lr_csv), exist_ok=True)

            # Exponential schedule across OPTIMIZER steps
            lr_curr = float(lr_min)
            lr_mult = (float(lr_max) / float(lr_min)) ** (1.0 / max(1, lr_steps - 1))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_curr

            # Tracking
            lr_hist = []        # rows: {"step", "lr", "loss", "ema"}
            ema_loss = None
            recent = []         # robust median window
            opt_steps_done = 0

            # Disable any external scheduler during the finder
            self.scheduler = None

        if self.qrm_checkpoint_path is not None and not lr_range:
            checkpoint_dir = os.path.join("models", os.path.basename(os.path.dirname(self.qrm_checkpoint_path)))
            model_tag = os.path.basename(os.path.dirname(self.qrm_checkpoint_path))
            log_file = os.path.join("logs", f"train_log_{model_tag}_{training_start_timestamp}.json")
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        else:
            model_tag = f"{training_start_timestamp}_{model_tag}"

            checkpoint_dir = os.path.join("models", model_tag)
            log_file = os.path.join("logs", f"train_log_{model_tag}.json")
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")

            os.makedirs("logs", exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)

                # Save metadata once
        metadata = {
            "model_folder_name": model_tag,
            "training_start_timestamp": training_start_timestamp,
            "qrm_type": self.qrm_type,
            "lr": self.lr,
            "accum_steps": self.accum_steps,
            "cfg_scale": cfg_scale,
            "time_bool": self.time_bool,
            "use_scheduler": self.use_scheduler,
            "num_epochs": self.num_epochs,
            "subset_size": self.subset_size,
            "batch_size": self.batch_size
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save initial checkpoint BEFORE training starts
        if start_epoch == 0 and not(lr_range):
            init_ckpt_path = os.path.join(checkpoint_dir, f"qrmmlp_joint_epoch_0.pth")
            torch.save({
                "model": self.sd_model.qrm.state_dict(),
                "qrm_type": self.qrm_type,
                "model_folder_name": model_tag
            }, init_ckpt_path)
            print(f"💾 Initial (epoch 0) checkpoint saved to {init_ckpt_path}")

        all_ds = FlatFileDatasetLazy("qrm/precomputed_prompt_folder/cached_ir_prompts")
        N = len(all_ds)             # e.g., ~404
        subset_size = self.subset_size

        for epoch in range(start_epoch+1, start_epoch+self.num_epochs+1):
            samples_done,epoch_loss = 0,0.0
            start_time = time.time()
            start_s = (epoch - 1) * subset_size
            idxs = [ (start_s + i) % N for i in range(subset_size) ]   # wraps around the dataset
            ds = torch.utils.data.Subset(all_ds, idxs)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
            self.optimizer.zero_grad()  # Important: zero grad before starting epoch

            # Wrap dataloader in tqdm, one tick per file
            for step, (prompt, cond) in enumerate(tqdm(dl, total=subset_size, desc=f"Epoch {epoch}")):
                step_start = time.time()
                print("the test parameter value is :",test)
                if test and (step + 1) == self.accum_steps + 1:
                    break
                c_crossattn = cond["c_crossattn"].to(self.device)
                y_vec       = cond["y"].to(self.device)
                uncond = {
                    "c_crossattn": self.uncond["c_crossattn"].expand(self.batch_size, -1, -1),
                    "y":           self.uncond["y"].expand(self.batch_size, -1)
                }
                uncond["c_crossattn"] = pad_to_match(uncond["c_crossattn"], c_crossattn)
                steps = self.inferencer.configs.get(self.inferencer.model_name).get("steps")
                sigmas_full = self.inferencer.get_sigmas(self.sd_model.model_sampling, steps).to(self.device)
                denoiser = CFGDenoiser(self.sd_model)
                text_inputs = ""

                extra_args = {
                    "cond":   {"c_crossattn": c_crossattn, "y": y_vec},
                    "uncond": uncond,
                    "cond_scale":     cfg_scale,
                    "prompt":           prompt,                # list[str]
                    "uncond_prompt":    [""] * self.batch_size,
                    "q_t_training":     True,
                    "qrm_type": self.qrm_type
                }
                warmup_mode = (epoch == 1) and (step < warmup_q_t_steps)
                extra_args["warmup_mode"] = warmup_mode
                
                if start == 2 and end == 2:
                    step_indices = [start]
                else:
                    a, b = 1.5, 7.0
                    sample = Beta(a, b).sample()
                    idx = int(start + (end - start) * sample.item())
                    idx = max(start, min(idx, end))
                    step_indices = [idx]

                if self.inferencer.model_name == "models/sd3.5_large_turbo.safetensors":
                    sampler = "euler"
                else:
                    sampler = "dpmpp2m"          
                
                with torch.amp.autocast('cuda'): #,dtype=torch.float32
                    x_k,x_k_baseline,c,c2,c3,idx = sample_qrm_one_step( #sample_qrm_one_step
                        denoiser, sigmas_full, text_inputs, extra_args, 
                        num_selected_steps=1,step_indices=step_indices, loss_type=loss_type, sampler = sampler
                    )

                # time.sleep(15)

                # =========================
                # Loss 1: margin seeking reward loss
                # =========================
                if loss_type == "margin_seeking":
                    # ---------- BASELINE (no grad, fp16 decode OK) ----------
                    with torch.no_grad():
                        with torch.autocast("cuda"):
                            x_k_baseline = SD3LatentFormat().process_out(x_k_baseline)
                            I_base_step = self.inferencer.vae.model.decode(x_k_baseline)
                            I_base_step = torch.clamp((I_base_step + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
                            s_base = self.reward_score(I_base_step, prompt,self.inferencer.eval_model, allow_grad=False)

                    torch.cuda.empty_cache()

                    with torch.autocast("cuda"):
                        x_k = SD3LatentFormat().process_out(x_k)
                        I_qrm_step = self.inferencer.vae.model.decode(x_k)
                        I_qrm_step = torch.clamp((I_qrm_step + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
                    I_qrm_step = I_qrm_step.contiguous()

                    s_qrm = self.reward_score(I_qrm_step, prompt,self.inferencer.eval_model, allow_grad=True)
                    base_cpu,qrm_cpu = 0,0
                    if not lr_range and (int(step) + 1) % 9 == 0:
                        model_out_dir = os.path.join("training_onestep_images/", model_tag)
                        os.makedirs(model_out_dir, exist_ok=True)
                        base_cpu = I_base_step[0].detach().cpu()
                        qrm_cpu  = I_qrm_step[0].detach().cpu()
                        p_tag   = self.prompt_slug(prompt, max_len=60)
                        base_tag = f"e{epoch}_s{step}_idx{idx}_sb{float(s_base):.3f}_{p_tag}"
                        qrm_tag = f"e{epoch}_s{step}_idx{idx}_sq{float(s_qrm):.3f}_{p_tag}"
                        save_image(base_cpu, os.path.join(model_out_dir, f"{base_tag}_base.png"))
                        save_image(qrm_cpu,  os.path.join(model_out_dir, f"{qrm_tag}_qrm.png"))


                    del x_k_baseline,I_base_step,I_qrm_step,x_k,base_cpu,qrm_cpu

                    margin = .05
                    # --- margin-seeking logistic (smooth ranking) ---
                    L_reward = F.relu(-s_qrm+(s_base+margin)).mean()

                # =========================
                # Loss 2: reward loss
                # =========================
                elif loss_type == "reward_maximization":
                    # -------- decode + score (QRM path only; no baseline) --------
                    with torch.autocast("cuda"):
                        x_k = SD3LatentFormat().process_out(x_k)
                        I_qrm_step = self.inferencer.vae.model.decode(x_k)
                        I_qrm_step = torch.clamp((I_qrm_step + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
                    I_qrm_step = I_qrm_step.contiguous()

                    s_qrm = self.reward_score(I_qrm_step, prompt,self.inferencer.eval_model, allow_grad=True)
                    margin = 0.40
                    L_reward = F.softplus(margin - s_qrm, beta=5.0).mean()

                    if not lr_range and (int(step) + 1) % 9 == 0:
                            model_out_dir = os.path.join("training_onestep_images/", model_tag)
                            os.makedirs(model_out_dir, exist_ok=True)
                            qrm_cpu  = I_qrm_step[0].detach().cpu()
                            p_tag   = self.prompt_slug(prompt, max_len=60)
                            qrm_tag = f"e{epoch}_s{step}_idx{idx}_sq{float(s_qrm):.3f}_{p_tag}"
                            save_image(qrm_cpu,  os.path.join(model_out_dir, f"{qrm_tag}_qrm.png"))

                else:
                    print("loss type not defined")
                    break

                # time.sleep(15)

                # ------------- final loss (1 term) -------------
                lambda_reward = 1.0
                total_loss = lambda_reward * L_reward

                opt_loss_accum += float(total_loss.detach().item())

                # scale by accum and backprop later as you already do
                loss = total_loss / self.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == self.subset_size:
                    self.scaler.unscale_(self.optimizer)
                    if lr_range:
                        gn = float(torch.nn.utils.clip_grad_norm_(self.sd_model.qrm.parameters(), max_norm=1e9).item())
                        # You already print c3 stats; use its norm for qrm-delta-norm
                        qd = float(c3.norm().item()) if isinstance(c3, torch.Tensor) else float('nan')

                    torch.nn.utils.clip_grad_norm_(self.sd_model.qrm.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # ---- LR-finder bookkeeping ----
                    if lr_range:
                        # Use the TOTAL loss for the just-finished optimizer step
                        # (i.e., before dividing by accum_steps). You have `total_loss` above.
                        step_loss = opt_loss_accum / float(self.accum_steps)
                        opt_loss_accum = 0.0

                        # Smooth it
                        ema_loss = step_loss if ema_loss is None else (1.0 - ema_alpha) * ema_loss + ema_alpha * step_loss
                        lr_hist.append({
                            "step": opt_steps_done + 1,
                            "lr": lr_curr,
                            "loss": step_loss,
                            "ema": ema_loss,
                            "grad_norm": gn,
                            "qrm_delta_norm": qd,
                            "med":(sorted(recent)[len(recent)//2] if recent else step_loss) * abort_factor
                        })

                        # Robust early-abort on blow-up
                        recent.append(step_loss)
                        if len(recent) > 50: recent.pop(0)
                        med = sorted(recent)[len(recent)//2] if recent else step_loss
                        if (len(recent) >= 20) and (not math.isfinite(step_loss) ): #or (step_loss*-1) > (abort_factor * med*-1)
                            print(f"[LR-Finder] abort at step {opt_steps_done+1}: loss={step_loss:.4f} (median={med:.4f})")
                            with open(lr_csv, "w", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=["step","lr","loss","ema","grad_norm","qrm_delta_norm","med"])
                                w.writeheader(); w.writerows(lr_hist)
                            return

                        # Advance LR and step counter
                        opt_steps_done += 1
                        if opt_steps_done >= lr_steps:
                            with open(lr_csv, "w", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=["step","lr","loss","ema","grad_norm","qrm_delta_norm","med"])
                                w.writeheader(); w.writerows(lr_hist)
                            print(f"[LR-Finder] wrote {len(lr_hist)} rows to {lr_csv}")
                            return

                        lr_curr = float(lr_curr * lr_mult)
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr_curr

                    # Normal schedule only in non-LR mode
                    if (not lr_range) and self.scheduler:
                        self.scheduler.step()

                epoch_loss += loss.item() * self.accum_steps

          
                # torch.cuda.empty_cache()

                # print("scale_shift min:", c.min().item(),"max: ", c.max().item(),"mean: ", c.mean().item(),"Nan?: ", torch.isnan(c).any().item(),"norm: ",c.norm().item())
                # print("scale_shift+self.qrm_delta min:", c2.min().item(),"max: ", c2.max().item(),"mean: ", c2.mean().item(),"Nan?: ", torch.isnan(c2).any().item(),"norm: ",c2.norm().item())
                # print("self.qrm_delta min:", c3.min().item(),"max: ", c3.max().item(),"mean: ", c3.mean().item(),"Nan?: ", torch.isnan(c3).any().item(),"norm: ",c3.norm().item())
                print("prompt was ",prompt)
                print(f"[E{epoch} S{step}] loss={(loss.item() * self.accum_steps):.4f} | rew_sco={s_qrm.item()} | idx={idx} | max scale_shift={c.max().item()} | max qrm_delta={c3.max().item()} | step_time={time.time() - step_start:.2f}s |")

                c1_stats_list.append(f"q_t+t+y min: {c.min().item()}, max: {c.max().item()}, mean:{ c.mean().item()}, Nan?: {torch.isnan(c).any().item()},norm:{c.norm().item()}")
                c2_stats_list.append(f"t+y min: {c2.min().item()}, max: {c2.max().item()}, mean:{ c2.mean().item()}, Nan?: {torch.isnan(c2).any().item()},norm:{c2.norm().item()}")
                c3_stats_list.append(f"q_t min: {c3.min().item()}, max: {c3.max().item()}, mean:{ c3.mean().item()}, Nan?: {torch.isnan(c3).any().item()},norm:{c3.norm().item()}")
                c5_stats_list.append(float((L_reward*lambda_reward).detach().item()))
                if torch.is_tensor(idx):
                    idx = int(idx.detach().item())
                c7_stats_list.append(idx)

                if loss_type == "margin_seeking":
                    del s_qrm, s_base
                else:
                    del s_qrm
            if not lr_range:
                self.run_qrm_validation(model_tag,epoch,cfg_scale,self.qrm_type,start,end)

            avg_loss = epoch_loss / max(1, samples_done)

            epoch_list.append(epoch)
            loss_list.append(avg_loss)
            time_list.append(time.time() - start_time)
            if not lr_range and int(epoch) % 5 == 0:
                # Save checkpoint with hyperparameters in folder name
                print(f"🔧 Saving QRM to folder: {checkpoint_dir}")
                checkpoint_path = os.path.join(checkpoint_dir, f"qrmmlp_joint_epoch_{epoch}.pth")
                torch.save({
                    "model": self.sd_model.qrm.state_dict(),
                    "qrm_start_step":start,
                    "qrm_end_step":end,
                    "qrm_type": self.qrm_type,
                    "model_folder_name": model_tag,
                    "vision_feature_model": self.inferencer.eval_model
                }, checkpoint_path)
                
                print(f"💾 Saved checkpoint to {checkpoint_path}")
                print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {time_list[-1]:.2f}s")

            # Update log after each epoch
            self.save_training_log(log_file, epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list,c5_stats_list,c7_stats_list)

        total_time = time.time() - overall_start
        print(f"🚀 Training complete in {total_time / 60:.2f} minutes")

    def prompt_slug(self, prompt, max_len=64):
        # Handle tuple/list prompts like ('text',) from your trace
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if len(prompt) else ""
        p_str = str(prompt)

        # Keep an 8-char hash for uniqueness
        h = hashlib.sha1(p_str.encode("utf-8")).hexdigest()[:8]

        # ASCII-only, collapse spaces, strip illegal filename chars
        s = p_str.encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r'[^A-Za-z0-9._ -]+', "", s)   # drop weird symbols
        s = s.replace(" ", "_")

        # Truncate and append hash
        s = s[:max_len]
        return f"{s}__{h}"


    def save_training_log(self, log_file, epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list,c5_stats_list,c7_stats_list):
        history = {
            "epoch": epoch_list,
            "loss": loss_list,
            "time": time_list,
            "q_t+t+y:": c1_stats_list,
            "t+y": c2_stats_list,
            "q_t": c3_stats_list,
            "reward_loss": c5_stats_list,
            # "safe_loss": c6_stats_list,
            "idx":c7_stats_list
        }

        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"📄 Training log updated at {log_file}")