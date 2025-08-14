# qrm/qrm_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from qrm.qrm_lora import LoraInjectedLinear
import torch.optim as optim
from itertools import cycle
from torch.utils.data import DataLoader


class QRMTrainer:
    def __init__(self, inferencer, device="cuda", lr=1e-5, lora_rank=4,qrm_checkpoint_path=None):
        self.device = device
        self.inferencer = inferencer
        from sd3_impls import BaseModel,SD3LatentFormat
        self.latent_fmt = SD3LatentFormat()
        self.sd_model: BaseModel = inferencer.sd3.model.to(device)
        self.vae = inferencer.get_vae().model.to(device)
        self.tokenizer = inferencer.tokenizer
        # your integrated QRMMLP lives at self.sd_model.qrm

        # 1️⃣ Freeze everything...
        for p in self.sd_model.parameters():
            p.requires_grad_(False)
        # ...then un-freeze QRMMLP
        for p in self.sd_model.qrm.parameters():
            p.requires_grad_(True)

        # 🔄 Load QRM weights if provided
        if qrm_checkpoint_path is not None:
            print(f"📦 Loading QRM weights from: {qrm_checkpoint_path}")
            self.sd_model.qrm.load_state_dict(torch.load(qrm_checkpoint_path, map_location=self.device))

        # # 2️⃣ Inject LoRA into MMDiT and un-freeze LoRA weights
        # self.injected_lora = inject_lora_into_mmditx(
        #     self.sd_model.diffusion_model,
        #     r=lora_rank, dropout_p=0.05, scale=1.0
        # )
        # lora_params = []
        # for _, lora_mod in self.injected_lora:
        #     lora_params += list(lora_mod.lora_down.parameters()) + list(lora_mod.lora_up.parameters())
        # for p in lora_params:
        #     p.requires_grad_(True)

        # 3️⃣ One optimizer for QRMMLP + all LoRA
        self.optimizer = optim.Adam(
            list(self.sd_model.qrm.parameters()),
            # + lora_params,
            lr=lr)
 
    def train(self, dataloader: DataLoader, epochs=50, cfg_scale=4.5):
        
        total_steps = len(dataloader)
        loader = cycle(dataloader)

        for epoch in range(epochs):
            for step in range(total_steps):
                prompt, image = next(loader)
                prompt = prompt[0]
                image = image.to(self.device)
                image = image * 2 - 1
                with torch.no_grad():
                    enc = self.vae.encode(image)
                latent = self.latent_fmt.process_in(enc).to(self.device).half()

                noise = torch.randn_like(latent)
                sigmas = self.inferencer.get_sigmas(self.sd_model.model_sampling, 50).cuda()
                sigmas = sigmas[int(50 * (1 - 0.8)) :]
                sigma = sigmas[0:1].expand(latent.size(0)).contiguous()  # shape [B]
                x_t = self.sd_model.model_sampling.noise_scaling(sigma, noise, latent, self.inferencer.max_denoise(sigmas))
                t = self.sd_model.model_sampling.timestep(sigma).to(self.device)


                # — get conditional and unconditional conditioning —
                cond = self.inferencer.get_cond(prompt)      
                uncond = self.inferencer.get_cond("")  
                cond = self.inferencer.fix_cond(cond)
                uncond = self.inferencer.fix_cond(uncond)

                # — get vision feature —
                vision_feature = self.inferencer.get_vision_feature(x_t, prompt).half()
                uncond_vision_feature = self.inferencer.get_vision_feature(x_t, "").half()

                
                print("q_t run #######################")
                # — batched apply_model (replicates CFG logic) —
                batched_out = self.sd_model.apply_model(
                    torch.cat([x_t, x_t]),
                    torch.cat([sigma, sigma]),
                    t_raw=torch.cat([t, t]),
                    c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
                    y=torch.cat([cond["y"], uncond["y"]]),
                    vision_feature=torch.cat([vision_feature,uncond_vision_feature]),
                    q_t_training = True
                ).float()

                print("→ batched_out stats:", batched_out.min().item(), batched_out.max().item(), torch.isnan(batched_out).any().item())

                cond_out, uncond_out = batched_out.chunk(2)

                # — classifier-free guidance —
                guided = uncond_out + (cond_out - uncond_out) * cfg_scale

                # Original denoising loss
                loss_guided = F.mse_loss(guided.float(), latent.float())

                # Regularization loss: encourage q_t ≈ t_emb + y_emb
                with torch.no_grad():
                    t_embed = self.sd_model.qrm.t_embedder(t.float(), dtype=torch.float32)  # [B, 1536]
                    y_embed = self.sd_model.qrm.y_embedder(cond["y"].float())               # [B, 1536]
                    c_target = t_embed + y_embed

                q_t = self.sd_model.qrm.q_t_last  # buffer from forward pass (add below)
                loss_reg = F.mse_loss(q_t.float(), c_target)

                # Combine losses
                lambda_reg = 0.1  # Tune as needed
                loss = loss_guided + lambda_reg * loss_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if step % 50 == 0:
                print(f"[E{epoch} S{step}/{total_steps}] loss={loss.item():.4f}")
                    
        # ——— final save ———
        torch.save(self.sd_model.qrm.state_dict(),       "models/qrmmlp_joint.pth")
        # save_lora_weights(self.injected_lora,             "models/mmditx_lora.pth")
        
def inject_lora_into_mmditx(mmditx_model, r=4, dropout_p=0.05, scale=1.0, verbose=False):
    injected_layers = []
    target_blocks = [0,1,2,3,20, 21, 22, 23]
    for module_name, module in mmditx_model.named_modules():
        if isinstance(module, nn.Linear):
            for block_idx in target_blocks:
                block_name = f"joint_blocks.{block_idx}"
                # print(f"Found linear layer: {module_name}")
                if any(name in module_name for name in [
                    f"{block_name}.context_block.mlp.fc1",
                    f"{block_name}.context_block.mlp.fc2",
                    f"{block_name}.x_block.mlp.fc1",
                    f"{block_name}.x_block.mlp.fc2"
                ]):
                    # or "attn.qkv" in module_name or "attn.proj" in module_name:
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