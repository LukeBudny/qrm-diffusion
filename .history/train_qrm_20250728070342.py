from qrm.qrm_trainer_batches import QRMTrainer_batches
from qrm.qrm_dataloader import CachedCOCOIterable
from sd3_infer import SD3Inferencer
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import torch
from qrm import _trainer
QRMTrainer_batches = _trainer()

import re

def get_start_epoch_from_path(path):
    match = re.search(r"epoch_(\d+)", path)
    return int(match.group(1)) + 1 if match else 0

# === Dataset location ===
coco_json = "annotations/merged_one_caption_per_image.json"
coco_images = "train2014"

EXPERIMENTS = [
    # # ── 1) Transformer QRM, *no* LoRA  ──────────────────────
                    dict(
        model_tag    = "transformer_10ep_rf_loss_lr5e-6_additive_qt_10_warmup",
        qrm_type     = "transformer",
        lr           = 5e-6,       
        num_epochs   = 3,
        eval_model   = 'clip',
        lora_only    = False,
        warmup_q_t_steps = 10
    ),
            dict(
        model_tag    = "transformer_10ep_rf_loss_lr5e-6_additive_qt_0_warmup",
        qrm_type     = "transformer",
        lr           = 5e-6,       
        num_epochs   = 3,
        eval_model   = 'clip',
        lora_only    = False,
        warmup_q_t_steps = 0
    ),
                dict(
        model_tag    = "transformer_10ep_rf_loss_lr5e-6_additive_qt_50_warmup",
        qrm_type     = "transformer",
        lr           = 5e-6,       
        num_epochs   = 3,
        eval_model   = 'clip',
        lora_only    = False,
        warmup_q_t_steps = 50
    ),
                dict(
        model_tag    = "transformer_10ep_rf_loss_lr5e-6_additive_qt_100_warmup",
        qrm_type     = "transformer",
        lr           = 5e-6,       
        num_epochs   = 3,
        eval_model   = 'clip',
        lora_only    = False,
        warmup_q_t_steps = 100
    ),
]


# === Image preprocessing ===
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

dataset = CachedCOCOIterable(
    annotation_path=coco_json,
    image_root=coco_images,
    cache_dir="cached_prompts",
    files_per_epoch=3,
    transform=image_transform,
    seed=123
)

def collate(batch):
    # tensors on CPU first; move to CUDA afterwards
    imgs   = torch.stack([b["image"] for b in batch])
    ccross = torch.stack([b["cond"]["c_crossattn"] for b in batch])
    yvec   = torch.stack([b["cond"]["y"] for b in batch])
    prompts = [b["prompt"] for b in batch]        # ← keep caption strings
    return imgs, prompts, {"c_crossattn": ccross, "y": yvec}

COMMON = dict(
    lr            = 1e-4,
    cfg_scale     = 4.5,
    cfg_weight   = 1.0,
    contrastive_weight   = 0.0,
    clip_weight   = 0.0,
    accum_steps   = 3,
    time_bool     = True,
    use_scheduler = True,
    num_epochs    = 10,
    subset_size = 399,
    full_dataset = dataset,
    batch_size = 1,
    device = "cuda",
    test = False,
    eval_model = 'clip',
    warmup_q_t_steps=100
)


for exp_id, exp in enumerate(EXPERIMENTS, start=1):

    inferencer = SD3Inferencer()
    inferencer.load(
        model              = "models/sd3.5_medium.safetensors",
        vae                = None,
        shift              = 3.0,
        controlnet_ckpt    = None,
        model_folder       = "models",
        text_encoder_device= "cpu",
        load_tokenizers    = False,
        eval_model= exp.get("eval_model", COMMON["eval_model"])

    )

    resume_ckpt = exp.get("resume_from", None)
    start_epoch = get_start_epoch_from_path(resume_ckpt) if resume_ckpt else 0


    print(f"\n🚀  Starting run #{exp_id}:  {exp['model_tag']}")
    trainer = QRMTrainer_batches(
        inferencer       = inferencer,
        device           = "cuda",
        lr               = exp.get("lr",COMMON["lr"]),
        time_bool        = COMMON["time_bool"],
        qrm_type         = exp["qrm_type"],
        num_epochs       = exp.get("num_epochs", COMMON["num_epochs"]),
        full_dataset     = COMMON["full_dataset"],
        collate_fn       = collate,
        subset_size      = COMMON["subset_size"],
        batch_size       = COMMON["batch_size"],
        accum_steps      = COMMON["accum_steps"],
        lora_rank        = exp.get("lora_rank",None),
        use_scheduler    = COMMON["use_scheduler"],
        qrm_checkpoint_path = exp.get("resume_from", None),
        lora_only = exp.get("lora_only", False),
    )

    trainer.train(
        cfg_scale    = COMMON["cfg_scale"],
        cfg_weight   = exp.get("cfg_weight", COMMON["cfg_weight"]),
        contrastive_weight   = exp.get("contrastive_weight", COMMON["contrastive_weight"]),
        clip_weight   = exp.get("clip_weight", COMMON["clip_weight"]),
        start_epoch  = start_epoch,
        test = exp.get("test", COMMON["test"]),
        model_tag = exp["model_tag"],
        warmup_q_t_steps = exp.get("warmup_q_t_steps", COMMON["warmup_q_t_steps"]),
    )
