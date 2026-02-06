from qrm.qrm_trainer_batches import QRMTrainer_batches
from qrm.qrm_dataloader import CachedCOCOIterable
from sd3_infer import SD3Inferencer
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import torch
from qrm import _trainer
QRMTrainer_batches = _trainer()
import gc
import os
import subprocess
import time
import pandas as pd
import re
from collections import defaultdict, Counter
import json
import sys


# "20251110_033131_QRMModulatorLatentV6_clip_score_rm_s1_50_steps_v5_beta_distribution_e25": "models/20251110_033131_QRMModulatorLatentV6_clip_score_rm_s1_50_steps_v5_beta_distribution/qrmmlp_joint_epoch_25.pth",
# "20250928_063404_QRMModulatorLatentV3_hps_margin_seek_start_15_lr1e6_all_qrm_e25": "models/20250928_063404_QRMModulatorLatentV3_hps_margin_seek_start_15_lr1e6_all_qrm/qrmmlp_joint_epoch_25.pth",
# "20250929_190929_QRMModulatorLatentV4_hps_rm_s15_all_qrm_allblock_2layers_e25": "models/20250929_190929_QRMModulatorLatentV4_hps_rm_s15_all_qrm_allblock_2layers/qrmmlp_joint_epoch_25.pth",
# "20250930_082151_QRMModulatorLatentV4_hps_ms_s15_all_qrm_allblock_2layers_e25": "models/20250930_082151_QRMModulatorLatentV4_hps_ms_s15_all_qrm_allblock_2layers/qrmmlp_joint_epoch_25.pth",
# "20251019_112957_QRMModulatorLatentV6_hps_rm_s1_e25": "models/20251019_112957_QRMModulatorLatentV6_hps_rm_s1/qrmmlp_joint_epoch_25.pth",
# "20251019_220458_QRMModulatorLatentV6_hps_ms_s1_e25": "models/20251019_220458_QRMModulatorLatentV6_hps_ms_s1/qrmmlp_joint_epoch_25.pth",
# "20251021_094258_QRMModulatorLatentV6_hps_rm_s1_5_steps_e25": "models/20251021_094258_QRMModulatorLatentV6_hps_rm_s1_5_steps/qrmmlp_joint_epoch_25.pth",
# "20251021_210121_QRMModulatorLatentV6_hps_ms_s1_5_steps_e25": "models/20251021_210121_QRMModulatorLatentV6_hps_ms_s1_5_steps/qrmmlp_joint_epoch_25.pth",
# "20251023_101324_QRMModulatorLatentV6_hps_rm_s1_2_steps_e25": "models/20251023_101324_QRMModulatorLatentV6_hps_rm_s1_2_steps/qrmmlp_joint_epoch_25.pth",
# "20251023_222112_QRMModulatorLatentV6_hps_ms_s1_2_steps_e25": "models/20251023_222112_QRMModulatorLatentV6_hps_ms_s1_2_steps/qrmmlp_joint_epoch_25.pth",
# "20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps_e25": "models/20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps/qrmmlp_joint_epoch_25.pth",
# "20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps_e24": "models/20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps/qrmmlp_joint_epoch_24.pth",
# "20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps_e20": "models/20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps/qrmmlp_joint_epoch_20.pth",
# "20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps_e10": "models/20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps/qrmmlp_joint_epoch_10.pth",
# "20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps_e01": "models/20251027_152551_QRMModulatorLatentV6_hps_rm_s1_49_steps/qrmmlp_joint_epoch_1.pth",
# "20251101_140324_QRMModulatorLatentV6_hps_rm_s26_50_steps_e4": "models/20251101_140324_QRMModulatorLatentV6_hps_rm_s26_50_steps/qrmmlp_joint_epoch_4.pth",
# "20251101_014254_QRMModulatorLatentV6_hps_rm_s1_25_steps_e24": "models/20251101_014254_QRMModulatorLatentV6_hps_rm_s1_25_steps/qrmmlp_joint_epoch_24.pth",
# "20251102_024813_QRMModulatorLatentV6_hps_rm_s13_38_steps_e24": "models/20251102_024813_QRMModulatorLatentV6_hps_rm_s13_38_steps/qrmmlp_joint_epoch_24.pth",
# "20251103_222152_QRMModulatorLatentV6_hps_rm_s1_15_steps_e24": "models/20251103_222152_QRMModulatorLatentV6_hps_rm_s1_15_steps/qrmmlp_joint_epoch_24.pth",
# "20251104_153423_QRMModulatorLatentV6_hps_rm_s1_25_steps_v2_e24": "models/20251104_153423_QRMModulatorLatentV6_hps_rm_s1_25_steps_v2/qrmmlp_joint_epoch_24.pth",
# "20251106_204616_QRMModulatorLatentV6_hps_rm_s1_25_steps_v4_e25": "models/20251106_204616_QRMModulatorLatentV6_hps_rm_s1_25_steps_v4/qrmmlp_joint_epoch_25.pth",
# "20251107_122318_QRMModulatorLatentV6_hps_rm_s1_25_steps_v5_e25": "models/20251107_122318_QRMModulatorLatentV6_hps_rm_s1_25_steps_v5/qrmmlp_joint_epoch_25.pth",
# "20251108_005939_QRMModulatorLatentV6_hps_rm_s1_50_steps_v5_beta_distribution_e25": "models/20251108_005939_QRMModulatorLatentV6_hps_rm_s1_50_steps_v5_beta_distribution/qrmmlp_joint_epoch_25.pth"


def get_start_epoch_from_path(path):
    match = re.search(r"epoch_(\d+)", path)
    return int(match.group(1)) if match else 0

# === Dataset location === 
coco_json = "qrm/annotations/merged_one_caption_per_image.json"

EXPERIMENTS = [ 
    #use paths like this "/mnt/c/Users/lukes/Desktop/QRM Diffusion Project/sd3.5/models/20250830_165406_x_k_vs_x_k_baseline_ignore_sigma_transformer_lr1e-6_add_qt_one_step_train_sched/qrmmlp_joint_epoch_9.pth",
    # REMEMBER TO CHECK THE LOSSES FOR THE BELOW
    #     dict(
    #     model_tag    = "QRMModulatorLatentV6_hps_rm_start_15_lr1e6_testing_for_git",
    #     qrm_type     = "QRMModulatorLatentV6",
    #     lr           = 1e-6,       
    #     num_epochs   = 5,
    #     eval_model   = 'hps',
    #     loss_type    = 'reward_maximization',
    #     start = 1,
    #     end = 25,
    #     test = True
    # ),
    ]

# === Image preprocessing ===
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def collate(batch):
    # tensors on CPU first; move to CUDA afterwards
    imgs   = torch.stack([b["image"] for b in batch])
    ccross = torch.stack([b["cond"]["c_crossattn"] for b in batch])
    yvec   = torch.stack([b["cond"]["y"] for b in batch])
    prompts = [b["prompt"] for b in batch]        # ← keep caption strings
    return imgs, prompts, {"c_crossattn": ccross, "y": yvec}

COMMON = dict(
    lr            = 1e-4,
    cfg_scale     = 5.0,
    accum_steps   = 3,
    time_bool     = True,
    use_scheduler = True,
    ignore_sigma = False,
    num_epochs    = 10,
    subset_size = 399,
    # full_dataset = dataset,
    batch_size = 1,
    device = "cuda",
    test = False,
    eval_model = 'clip',
    warmup_q_t_steps=0
)

def run_one_experiment(exp):

    print(f"\n🚀 Starting run: {exp['model_tag']}")

    torch.cuda.set_per_process_memory_fraction(1.0, device=0) 
    torch.backends.cuda.matmul.allow_tf32 = True

    inferencer = SD3Inferencer()
    inferencer.load(
        model              = exp.get("model","models/sd3.5_medium.safetensors"),
        vae                = None,
        shift              = 3.0,
        controlnet_ckpt    = None,
        model_folder       = "models",
        text_encoder_device= "cpu",
        load_tokenizers    = False,
        eval_model= exp.get("eval_model", COMMON["eval_model"]),
        qrm_type = exp["qrm_type"]
    )

    resume_ckpt = exp.get("resume_from", None)
    start_epoch = get_start_epoch_from_path(resume_ckpt) if resume_ckpt else 0


    trainer = QRMTrainer_batches(
        inferencer       = inferencer,
        device           = "cuda",
        lr               = exp.get("lr",COMMON["lr"]),
        time_bool        = COMMON["time_bool"],
        qrm_type         = exp["qrm_type"],
        num_epochs       = exp.get("num_epochs", COMMON["num_epochs"]),
        # full_dataset     = COMMON["full_dataset"],
        collate_fn       = collate,
        subset_size      = COMMON["subset_size"],
        batch_size       = COMMON["batch_size"],
        accum_steps      = COMMON["accum_steps"],
        use_scheduler    = COMMON["use_scheduler"],
        qrm_checkpoint_path = exp.get("resume_from", None),
    )

    trainer.train(
        cfg_scale    = exp.get("cfg_scale",COMMON["cfg_scale"]),
        start_epoch  = start_epoch,
        test = exp.get("test", COMMON["test"]),
        model_tag = exp["model_tag"],
        warmup_q_t_steps = exp.get("warmup_q_t_steps", COMMON["warmup_q_t_steps"]),
        ignore_sigma    = exp.get("ignore_sigma", COMMON["ignore_sigma"]),
        start = exp.get("start",25),
        end = exp.get("end",47),
        loss_type = exp.get("loss_type","reward_maximization"),
        lr_range = exp.get("lr_range", False)
    )

    # --- CLEANUP ---
    try:
        trainer.sd_model.to("cpu")
        trainer.vae.to("cpu")
        inferencer.sd3.model.to("cpu")
    except Exception:
        pass

    del trainer
    del inferencer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

torch.cuda.empty_cache()

if not EXPERIMENTS:
    print("no training")
else:
    for exp_id, exp in enumerate(EXPERIMENTS, start=1):
        try:
            run_one_experiment(exp)
        except:
            print("experiment " ,exp_id, " failed")

# === LIGHTWEIGHT CONFIG (set to False to disable) ===
LIGHTWEIGHT = True
LIGHTWEIGHT_VARIANTS = 5 if LIGHTWEIGHT else 1
# You can list IDs inline or put them in a text file
LIGHTWEIGHT_ONLY_IDS = [
    "00017/0000.png",
    "00049/0000.png","00121/0000.png","00172/0000.png",
    "00178/0000.png","00180/0000.png","00250/0000.png","00254/0000.png",
    "00255/0000.png","00273/0000.png","00283/0000.png","00304/0000.png",
    "00316/0000.png","00319/0000.png","00354/0000.png","00357/0000.png",
    "00391/0000.png","00397/0000.png","00446/0000.png","00468/0000.png",
    "00475/0000.png","00479/0000.png",
]

# === CONFIGURATION ===
geneval_prompt_file = "geneval/prompts/evaluation_metadata.jsonl"
valcoco2014_prompt_file = "qrm/annotations/coco_val_prompts.jsonl"
model_file = "models/sd3.5_medium.safetensors"
models_parti_json = "qrm/model_jsons/models_parti.json"
models_json = "qrm/model_jsons/models_geneval.json"

# Redirect all output dirs when lightweight
geneval_output_dir = "experiments/comparative_geneval_images" if LIGHTWEIGHT else "geneval/comparative_images"
parti_output_dir = "experiments/comparative_parti_images"
valcoco2014_output_dir = "coco_val_images/comparative_initial_images" if LIGHTWEIGHT else "coco_val_images/comparative_images"
results_dir = "comparative_results" if LIGHTWEIGHT else "comparative_results"

script_path = "generate_eval_images.py"
venv_sd35 = os.path.join("sd35-env", "Scripts", "python.exe")

os.makedirs(geneval_output_dir, exist_ok=True)
os.makedirs(valcoco2014_output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# If using inline IDs, write them to a small helper file we can pass along
only_ids_file = None
if LIGHTWEIGHT and LIGHTWEIGHT_ONLY_IDS:
    only_ids_file = os.path.join(results_dir, "only_image_ids.txt")  # results_dir already exists
    with open(only_ids_file, "w") as f:
        for x in LIGHTWEIGHT_ONLY_IDS:
            f.write(x.strip() + "\n")

PY = sys.executable

gen_cmd = [
    PY,"-m",# venv_sd35,
    "qrm.generate_images",
    "--prompt_file", geneval_prompt_file,
    "--sd3_path", model_file,
    "--models_path", models_parti_json,
    "--out_dir", geneval_output_dir,
    "--num_prompts", "-1",
    "--num_variants", str(LIGHTWEIGHT_VARIANTS),            # baseline default
    "--seed", "42",
]
if LIGHTWEIGHT:
    gen_cmd += ["--lightweight_variants", str(LIGHTWEIGHT_VARIANTS)]
    if only_ids_file:
        gen_cmd += ["--only_image_ids_file", only_ids_file]
subprocess.run(gen_cmd, check=True)

parti_cmd = [
    PY, "-m",# venv_sd35, 
    "qrm.evaluate_image_metrics",
    parti_output_dir,
    "--outdir", parti_output_dir,
    "--num_variants", str(1)  # e.g., 5
]
subprocess.run(parti_cmd, check=True)

torch.cuda.empty_cache()