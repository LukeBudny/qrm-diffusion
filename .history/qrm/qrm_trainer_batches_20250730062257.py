# qrm/qrm_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from qrm.qrm_lora import LoraInjectedLinear
import torch.optim as optim
from itertools import cycle
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import os
import json
import time
import re
import numpy as np
from qrm.qrm_models import QRMRegistry
from transformers import get_cosine_schedule_with_warmup
from sd3_impls import sample_dpmpp_2m_qrm,sample_qrm_one_step, CFGDenoiser
from torch.utils.data import DataLoader, Subset, IterableDataset, RandomSampler
import torch, random, math
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silence warning

# #old simplified loss
# true_flow = (latent - x_t_i) / sigma_i
# pred_flow = guided - x_t_i
# main_loss = F.mse_loss(pred_flow, true_flow) * cfg_weight
# total_loss += main_loss

def infer_vision_feature_dim(inferencer):
    """
    Returns the output dimension of get_vision_feature() by running it
    once on a dummy latent and prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_latent = torch.randn(1, 16, 128, 128).to(device)  # Assumes SD3 latent shape
    dummy_prompt = ["a photo of a cat"]
    text_inputs = inferencer.clip_tokenizer(dummy_prompt + [""], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        vision_feature = inferencer.get_vision_feature(dummy_latent, dummy_prompt,text_inputs)
    return vision_feature.shape[-1]

def get_epoch_loader(dataset,
                     subset_size=None,
                     batch_size=1,
                     num_workers=4,
                     pin_memory=True,
                     collate_fn=None,
                     seed=42):
    """
    Works for both map-style datasets and IterableDatasets.
    • IterableDataset → just wrap in DataLoader (dataset itself decides epoch size).
    • Map-style Dataset → optional random subset of `subset_size`.
    """
    # CachedCOCOIterable 
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          pin_memory=pin_memory,
                          collate_fn=collate_fn)

    # COCOPromptDataset 
    if subset_size is not None and subset_size < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)

    sampler = RandomSampler(dataset, replacement=False)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)


def sample_mode_timesteps(s: float, batch_size: int, device="cuda"):
    """
    Sample skewed timesteps t ∈ (0,1) using mode distribution π_mode(t; s).
    s=1.0 corresponds to rf/lognorm(0.00, 1.00) in the SD3 paper.
    """
    u = torch.rand(batch_size, device=device)
    return (1 + s) * u / (u + s + 1e-8)

def pad_to_match(a, b):
    diff = b.size(1) - a.size(1)
    if diff > 0:
        pad = torch.zeros(a.size(0), diff, a.size(2), dtype=a.dtype, device=a.device)
        return torch.cat([a, pad], dim=1)
    elif diff < 0:
        return a[:, :b.size(1), :]
    return a

def sample_sigma_lognormal(batch_size, device, s=1.0):
    """
    Sample σ ~ exp(-t), where t is mode-biased like lognormal(0,1).
    s: controls sharpness (default = 1.0 is reasonable)
    """
    u = torch.rand(batch_size, device=device)
    t = (1 + s) * u / (u + s + 1e-8)  # Sample mode-biased time t ∈ (0,1)
    sigma = torch.exp(-t)            # Convert time t to sigma
    return sigma.view(-1, 1, 1, 1)   # Shape [B,1,1,1] for broadcasting


class QRMTrainer_batches:
    def __init__(self, inferencer, device="cuda", lr=1e-5, qrm_checkpoint_path=None,time_bool=True,qrm_type="mlp",
                num_epochs = 10,full_dataset = None,subset_size = 256,batch_size = 1,accum_steps = 4,lora_rank=None,use_scheduler=False,collate_fn=None, lora_only=False):
        self.device = device
        self.inferencer = inferencer
        from sd3_impls import BaseModel,SD3LatentFormat
        self.latent_fmt = SD3LatentFormat()
        self.sd_model: BaseModel = inferencer.sd3.model.to(device)
        self.collate_fn = collate_fn
        self.time_bool = time_bool
        self.qrm_type = qrm_type
        vision_dim = infer_vision_feature_dim(self.inferencer)
        self.sd_model.qrm = QRMRegistry[self.qrm_type](vision_dim=vision_dim).to(device)
        self.vae = inferencer.get_vae().model.to(device)
        self.tokenizer = inferencer.tokenizer
        self.lr = lr
        self.lora_only = lora_only
        self.clip_model = self.inferencer.clip_model
        self.clip_processor = self.inferencer.clip_processor
        self.vae = self.inferencer.vae 
        self.vae.model.decoder = self.inferencer.vae.model.decoder

        # Freeze everything...
        for p in self.sd_model.parameters():
            p.requires_grad_(False)
        # ...then un-freeze QRMMLP
        # Decide whether to train QRM or not
        if not self.lora_only:
            for p in self.sd_model.qrm.parameters():
                p.requires_grad_(True)
        else:
            for p in self.sd_model.qrm.parameters():
                p.requires_grad_(False)

        #  Load QRM weights if provided
        if qrm_checkpoint_path is not None:
            print(f"📦 Loading QRM weights from: {qrm_checkpoint_path}")
            checkpoint = torch.load(qrm_checkpoint_path, map_location=self.device)
            self.sd_model.qrm.load_state_dict(checkpoint["model"])

        param_groups = [p for p in self.sd_model.qrm.parameters() if p.requires_grad]

        self.injected_lora = []
        print(f"[DEBUG] LoRA rank: {lora_rank}")
        if lora_rank is not None:
            self.injected_lora = inject_lora_into_mmditx(
                self.sd_model.diffusion_model,
                r=lora_rank, dropout_p=0.05, scale=1.0
            )
                    
            for _, lora_mod in self.injected_lora:
                # (1) Set .linear weight/bias to float16 for AMP
                if hasattr(lora_mod, 'linear'):
                    lora_mod.linear.weight = torch.nn.Parameter(lora_mod.linear.weight.detach().half())
                    if lora_mod.linear.bias is not None:
                        lora_mod.linear.bias = torch.nn.Parameter(lora_mod.linear.bias.detach().half())
                
                # (2) Initialize and cast LoRA up/down to float32
                torch.nn.init.kaiming_uniform_(lora_mod.lora_down.weight, a=math.sqrt(5))
                lora_mod.lora_up.weight.data.zero_()
                lora_mod.lora_up.weight = torch.nn.Parameter(lora_mod.lora_up.weight.detach().float())
                lora_mod.lora_down.weight = torch.nn.Parameter(lora_mod.lora_down.weight.detach().float())

                for p in lora_mod.lora_up.parameters():
                    p.requires_grad_(True)
                for p in lora_mod.lora_down.parameters():
                    p.requires_grad_(True)

                # Don't cast lora_mod with `.to()` here — it undoes dtype corrections
                param_groups += list(lora_mod.lora_down.parameters()) + list(lora_mod.lora_up.parameters())


                
        # Convert non-LoRA layers like proj to float16 to match AMP input
        for name, module in self.sd_model.diffusion_model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Conv2d):
                if module.weight.dtype != torch.float16:
                    module.weight.data = module.weight.data.half()
                if module.bias is not None and module.bias.dtype != torch.float16:
                    module.bias.data = module.bias.data.half()

        proj = self.sd_model.diffusion_model.x_embedder.proj

        print("[DEBUG] Before fix: proj weight dtype =", proj.weight.dtype,
            "bias dtype =", proj.bias.dtype)

        proj.weight = torch.nn.Parameter(proj.weight.detach().half())
        if proj.bias is not None:
            proj.bias = torch.nn.Parameter(proj.bias.detach().half())

        print("[DEBUG] After fix: proj weight dtype =", proj.weight.dtype,
            "bias dtype =", proj.bias.dtype)

        
        self.sd_model.qrm.train()
        for _, lora_mod in self.injected_lora:
            lora_mod.train()

        self.lora_rank = lora_rank

        self.optimizer = optim.Adam(param_groups, lr=lr)
        
        self.num_epochs = num_epochs
        self.full_dataset = full_dataset
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        
        self.scheduler = None
        if use_scheduler:
            # total steps = number of steps in 1 epoch * number of epochs
            total_steps = num_epochs * (subset_size//accum_steps)

            warmup_steps = int(0.05 * total_steps)

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        self.use_scheduler = use_scheduler
        self.uncond = torch.load("cached_prompts/uncond.pt", map_location=self.device)
        self.scaler = GradScaler()
 
    def train(self, cfg_scale=4.5, cfg_weight=0.0,contrastive_weight=0.0,clip_weight=0.0,start_epoch=0,test = False,model_tag='unnamed?',warmup_q_t_steps=100):
        print("batch_size ", self.batch_size)
        epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list = [],[],[],[],[],[]
        clip_loss = 0
        overall_start = time.time()

        # Unique identifiers including hyperparameters
        training_start_timestamp = time.strftime("%Y%m%d_%H%M%S")

        model_tag = f"{training_start_timestamp}_{model_tag}"

        if self.use_scheduler:
            model_tag += "_sched"

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
            "lora_only": self.lora_only,
            "lora_rank": self.lora_rank,
            "lr": self.lr,
            "accum_steps": self.accum_steps,
            "cfg_weight": cfg_weight,
            "contrastive_weight": contrastive_weight,
            "clip_weight": clip_weight,
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
        init_ckpt_path = os.path.join(checkpoint_dir, f"qrmmlp_joint_epoch_0.pth")
        torch.save({
            "model": self.sd_model.qrm.state_dict(),
            "time_bool": self.sd_model.qrm.time_bool,
            "qrm_type": self.qrm_type,
            "model_folder_name": model_tag
        }, init_ckpt_path)
        print(f"💾 Initial (epoch 0) checkpoint saved to {init_ckpt_path}")

        # Save LoRA weights if applicable
        if self.injected_lora:
            init_lora_path = os.path.join(checkpoint_dir, f"mmditx_lora_epoch_0.pth")
            save_lora_weights(self.injected_lora, init_lora_path)
            print(f"💾 Initial LoRA weights saved to {init_lora_path}")


        for epoch in range(start_epoch+1, start_epoch+self.num_epochs+1):
            dataloader = get_epoch_loader(
                self.full_dataset,
                subset_size=self.subset_size,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                seed=42 + epoch)
            q_t_means, q_t_stds = [], []
            start_time = time.time()
            epoch_loss = 0.0
            self.optimizer.zero_grad()  # Important: zero grad before starting epoch

            for step, (images, prompts, cond) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch}")):

                if test and (step + 1) == self.accum_steps + 1:
                    break

                step_start = time.time()
                images = images.to(self.device) * 2 - 1

                with torch.no_grad():
                    enc = self.vae.encode(images)
                    latent = self.latent_fmt.process_in(enc).half()
                noise = torch.randn_like(latent)
                c_crossattn = cond["c_crossattn"].to(self.device) # fp16 CUDA
                y_vec       = cond["y"].to(self.device) 
                uncond = {
                    "c_crossattn": self.uncond["c_crossattn"].expand(self.batch_size, -1, -1),
                    "y":           self.uncond["y"].expand(self.batch_size, -1)
                }
                uncond["c_crossattn"] = pad_to_match(uncond["c_crossattn"], c_crossattn)

                # sigma_i = sample_sigma_lognormal(images.size(0), device=self.device)  # [B,1,1,1]
                sigmas_full = self.inferencer.get_sigmas(self.sd_model.model_sampling, 40).to(self.device)
                sigma_0 = sigmas_full[0].view(1, 1, 1, 1)
                sigma_i = sigma_0.expand(images.size(0), 1, 1, 1)
                max_denoise_flag = self.inferencer.max_denoise(sigmas_full)
                # max_denoise_flag = self.inferencer.max_denoise(sigma_i.view(-1))
                x_t = self.sd_model.model_sampling.noise_scaling(
                    sigma_i, noise, latent, max_denoise_flag)
                denoiser = CFGDenoiser(self.sd_model)
                text_inputs = self.inferencer.clip_tokenizer(prompts + [""], return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}

                def dynamic_vision_fn(x_t, prompt,text_inputs):
                    return self.inferencer.get_vision_feature(x_t, prompt,text_inputs)

                extra_args = {
                    "cond":   {"c_crossattn": c_crossattn, "y": y_vec},
                    "uncond": uncond,
                    "cond_scale":     cfg_scale,
                    "dynamic_vision_fn": dynamic_vision_fn,
                    "prompt":           prompts,                # list[str]
                    "uncond_prompt":    [""] * self.batch_size,
                    "q_t_training":     True,
                }
                warmup_mode = (epoch == 1) and (step < warmup_q_t_steps)
                extra_args["warmup_mode"] = warmup_mode
                

                if self.lora_only:
                    print("lora_only run")
                    with torch.cuda.amp.autocast(enabled=False):  
                        guided_list, q_t_list, x_t_list, sigma_list,t_raw,c,c2,c3 = sample_dpmpp_2m_qrm(
                            denoiser, x_t, sigmas_full, text_inputs, extra_args, num_selected_steps=1, lora_only=True
                        )
                else:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        guided_list, q_t_list, x_t_list, sigma_list,t_raw,c,c2,c3 = sample_dpmpp_2m_qrm(
                            denoiser, x_t, sigmas_full, text_inputs, extra_args, num_selected_steps=1, lora_only=False
                        )
                print("q_t+t+y min:", c.min().item(),"max: ", c.max().item(),"mean: ", c.mean().item(),"Nan?: ", torch.isnan(c).any().item(),"norm: ",c.norm().item())
                print("t+y min:", c2.min().item(),"max: ", c2.max().item(),"mean: ", c2.mean().item(),"Nan?: ", torch.isnan(c2).any().item(),"norm: ",c2.norm().item())
                print("q_t min:", c3.min().item(),"max: ", c3.max().item(),"mean: ", c3.mean().item(),"Nan?: ", torch.isnan(c3).any().item(),"norm: ",c3.norm().item())
                total_loss = 0
                for guided, q_t, x_t_i, sigma_i in zip(guided_list, q_t_list, x_t_list, sigma_list):

                    #new loss from rectified flow paper
                    # Compute predicted and target noise from current x_t_i, guided, and latent
                    safe_sigma_i = sigma_i.clamp(min=1e-4)
                    predicted_epsilon = (x_t_i - guided) / safe_sigma_i
                    target_epsilon = (x_t_i - latent) / safe_sigma_i

                    log_t = sigma_i.clamp(min=1e-4).log()
                    weight_t = torch.exp(-(log_t ** 2) / 2)
                    weight_t = weight_t.clamp(min=1e-3, max=10.0)
                    diff = predicted_epsilon - target_epsilon         # [B, C, H, W]
                    mse_per_sample = diff.pow(2).mean(dim=[1, 2, 3])  # [B]
                    main_loss = 0.5 * (weight_t * mse_per_sample).mean()
                    total_loss += main_loss
                    
                    if clip_weight != 0:
                        with torch.no_grad():
                            x_t_i = ((x_t_i + 1) / 2).clamp(0, 1)
                            x_t_i = F.interpolate(x_t_i, size=(224, 224), mode="bicubic", align_corners=False)
                            rgb_imgs = self.vae.decode(self.latent_fmt.process_out(x_t_i.float()))
                            rgb_imgs = ((rgb_imgs + 1) / 2).clamp(0, 1)  # [0,1]
                            rgb_imgs = F.interpolate(rgb_imgs, size=(224, 224), mode="bicubic", align_corners=False)
                            pil_images = [transforms.ToPILImage()(img.cpu()) for img in rgb_imgs]
                            inputs = self.clip_processor(images=pil_images, text=prompts, return_tensors="pt", padding=True)
                            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            outputs = self.clip_model(**inputs)
                            sim = F.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=-1)
                            clip_loss = (1 - sim.mean()) * clip_weight
                        total_loss += clip_loss


                loss = total_loss / len(guided_list)
                loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(dataloader):
                    # Unscaled gradients can't be clipped correctly
                    self.scaler.unscale_(self.optimizer)

                    # Now clip safely
                    torch.nn.utils.clip_grad_norm_(self.sd_model.qrm.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

                epoch_loss += loss.item() * self.accum_steps
                # q_t_means.append(q_t.mean().item())
                # q_t_stds.append(q_t.std().item())

                print(f"[E{epoch} S{step}] loss={(loss.item() * self.accum_steps):.4f} | step_time={time.time() - step_start:.2f}s | clip loss={clip_loss}")

                c1_stats_list.append(f"q_t+t+y min: {c.min().item()}, max: {c.max().item()}, mean:{ c.mean().item()}, Nan?: {torch.isnan(c).any().item()},norm:{c.norm().item()}")
                c2_stats_list.append(f"t+y min: {c2.min().item()}, max: {c2.max().item()}, mean:{ c2.mean().item()}, Nan?: {torch.isnan(c2).any().item()},norm:{c2.norm().item()}")
                c3_stats_list.append(f"q_t min: {c3.min().item()}, max: {c3.max().item()}, mean:{ c3.mean().item()}, Nan?: {torch.isnan(c3).any().item()},norm:{c3.norm().item()}")

                if step == 30:
                    break
            avg_loss = epoch_loss / (len(dataloader) * self.batch_size) 

            epoch_list.append(epoch)
            loss_list.append(avg_loss)
            time_list.append(time.time() - start_time)

            # Save checkpoint with hyperparameters in folder name
            print(f"🔧 Saving QRM to folder: {checkpoint_dir}")
            checkpoint_path = os.path.join(checkpoint_dir, f"qrmmlp_joint_epoch_{epoch}.pth")
            torch.save({
                "model": self.sd_model.qrm.state_dict(),
                "time_bool": self.sd_model.qrm.time_bool,
                "qrm_type": self.qrm_type,
                "model_folder_name": model_tag
            }, checkpoint_path)

            # save LoRA
            print(f"🔧 Saving LORA to folder: {checkpoint_dir}")
            if self.injected_lora:                                   # only when LoRA active
                lora_path = os.path.join(
                    checkpoint_dir, f"mmditx_lora_epoch_{epoch}.pth")
                save_lora_weights(self.injected_lora, lora_path)     # <-- helper already present
                print(f"💾  LoRA weights saved to {lora_path}")
            
            print(f"💾 Saved checkpoint to {checkpoint_path}")
            print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {time_list[-1]:.2f}s")

            # Update log after each epoch
            self.save_training_log(log_file, epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list)

        total_time = time.time() - overall_start
        print(f"🚀 Training complete in {total_time / 60:.2f} minutes")

    def save_training_log(self, log_file, epoch_list, loss_list, time_list, c1_stats_list,c2_stats_list,c3_stats_list):
        history = {
            "epoch": epoch_list,
            "loss": loss_list,
            "time": time_list,
            "q_t+t+y:": c1_stats_list,
            "t+y": c2_stats_list,
            "q_t": c3_stats_list
        }

        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"📄 Training log updated at {log_file}")
        
def inject_lora_into_mmditx(mmditx_model, r=4, dropout_p=0.05, scale=.5, verbose=False):
    injected_layers = []
    # Dynamically identify all joint_blocks.N.* layers
    block_indices = set()
    # seen_blocks = set()
    for name, _ in mmditx_model.named_modules():
        match = re.match(r"joint_blocks\.(\d+)", name)
        if match:
            block_indices.add(int(match.group(1)))
    
    for module_name, module in mmditx_model.named_modules():
        if isinstance(module, nn.Linear):
            for block_idx in sorted(block_indices):
                block_name = f"joint_blocks.{block_idx}"

                # block_root = module_name.split('.')[0:3]  # e.g., ['joint_blocks', '22', 'context_block']
                # block_name = ".".join(block_root)
                # if block_name not in seen_blocks:
                #     print(f"[UNIQUE BLOCK] {block_name}")
                #     seen_blocks.add(block_name)

                if any(name in module_name for name in [
                        f"{block_name}.context_block.adaLN_modulation",
                        f"{block_name}.final_layer.adaLN_modulation",
                ]):
                    parent_name, attr_name = module_name.rsplit('.', 1)
                    parent_module = dict(mmditx_model.named_modules())[parent_name]

                    lora_linear = LoraInjectedLinear(
                        module.in_features,
                        module.out_features,
                        r=r,
                        dropout_p=dropout_p,
                        scale=scale
                    ).to(module.weight.device).to(module.weight.dtype)

                    # Copy original weights and bias
                    lora_linear.linear.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        # If the original layer has bias, explicitly add it to your LoRA layer
                        lora_linear.linear.bias = nn.Parameter(module.bias.data.clone())

                    setattr(parent_module, attr_name, lora_linear.to(module.weight.device))
                    injected_layers.append((module_name, lora_linear))

                    if verbose:
                        print(f"LoRA explicitly injected into MLP layer: {module_name}")
    print(f"[DEBUG] Total matched LoRA layers: {len(injected_layers)}")

    return injected_layers
    
def save_lora_weights(injected_layers, path="mmditx_lora.pth"):
    """
    Save LoRA weights for all injected MM-DiT-X layers.
    - injected_layers: output from inject_lora_into_mmditx (list of tuples)
    - path: file path to save weights
    """
    lora_state = {}
    for name, lora_module in injected_layers:
        lora_state[name + ".lora_up"] = lora_module.lora_up.state_dict()
        lora_state[name + ".lora_down"] = lora_module.lora_down.state_dict()
    torch.save(lora_state, path)
    print(f"[DEBUG] Saving LoRA weights for {len(injected_layers)} layers to {path}")


def load_lora_weights(mmditx_model, path="mmditx_lora.pth"):
    """
    Load LoRA weights from a saved checkpoint into your MM-DiT-X model.
    - mmditx_model: your model into which LoRA was previously injected
    - path: file path to load weights from
    """
    lora_state = torch.load(path)
    for module_name, module in mmditx_model.named_modules():
        if isinstance(module, LoraInjectedLinear):
            module.lora_up.load_state_dict(lora_state[module_name + ".lora_up"])
            module.lora_down.load_state_dict(lora_state[module_name + ".lora_down"])