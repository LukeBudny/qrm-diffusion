from sd3_infer import SD3Inferencer
import torch, os, gc, re, json
from tqdm import tqdm
from pathlib import Path

# ───────────────────────── helpers ──────────────────────────────────
def find_last_finished_batch(out_dir: Path, pattern=r"batch_(\d{5})\.pt") -> int:
    rx = re.compile(pattern)
    nums = [int(rx.match(f.name).group(1))
            for f in out_dir.glob("batch_*.pt") if rx.match(f.name)]
    return max(nums) if nums else -1

def _is_clean(path: Path) -> bool:
    blob = torch.load(path, map_location="cpu")
    for d in blob.values():
        if torch.isnan(d["c_crossattn"]).any() or torch.isinf(d["c_crossattn"]).any():
            return False
        if torch.isnan(d["y"]).any() or torch.isinf(d["y"]).any():
            return False
    return True

# ───────────────────────── settings ─────────────────────────────────
DEVICE     = "cuda"
BATCH_SIZE = 553   # Adjust if needed
OUT_DIR    = Path("cached_coco_val_prompts")
OUT_DIR.mkdir(exist_ok=True)

last_done  = find_last_finished_batch(OUT_DIR)
start_idx  = (last_done + 1) * BATCH_SIZE

# ───────────────────────── load inferencer ──────────────────────────
inferencer = SD3Inferencer()
inferencer.load(
    model="models/sd3.5_medium.safetensors",
    vae=None,
    shift=3.0,
    controlnet_ckpt=None,
    model_folder="models",
    text_encoder_device="cuda",
    load_non_tokenizers=False
)
print("finished loading")

# ───────────────────────── load Geneval prompts ──────────────────────
geneval_file = "annotations/coco_val_prompts.jsonl"

with open(geneval_file, "r") as f:
    prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

prompts = sorted(set(prompts))
print(f"Total prompts: {len(prompts)} | Batch size: {BATCH_SIZE}")

# ───────────────────────── batched pre-compute ──────────────────────

with torch.inference_mode():
    for i in tqdm(range(start_idx, len(prompts), BATCH_SIZE), desc="Precomputing"):
        batch_id      = last_done + 1 + (i - start_idx) // BATCH_SIZE
        batch_prompts = prompts[i: i + BATCH_SIZE]

        cond = inferencer.fix_cond(inferencer.get_cond_batch(batch_prompts))

        batch_dict = {
            p: {
                "c_crossattn": cond["c_crossattn"][j].cpu(),
                "y":           cond["y"][j].cpu()
            }
            for j, p in enumerate(batch_prompts)
        }

        tmp_path  = OUT_DIR / f"batch_{batch_id:05d}.tmp"
        final_path = OUT_DIR / f"batch_{batch_id:05d}.pt"

        torch.save(batch_dict, tmp_path)
        os.replace(tmp_path, final_path)

        if not _is_clean(final_path):
            raise RuntimeError(f"NaNs or Infs detected in {final_path}")

        del cond, batch_dict
        gc.collect()
        torch.cuda.empty_cache()

# ───────────────────────── unconditional prompt ─────────────────────
uncond = inferencer.fix_cond(inferencer.get_cond_batch([""]))
torch.save({
    "c_crossattn": uncond["c_crossattn"].cpu(),
    "y":           uncond["y"].cpu()
}, OUT_DIR / "uncond.pt")

print("✓ all batches written to", OUT_DIR)


# from sd3_infer import SD3Inferencer
# import torch, os, gc, re
# from tqdm import tqdm
# from torchvision import transforms
# from qrm.qrm_dataloader import COCOPromptDataset
# from pathlib import Path

# # ───────────────────────── helpers ──────────────────────────────────
# def find_last_finished_batch(out_dir: Path, pattern=r"batch_(\d{5})\.pt") -> int:
#     rx = re.compile(pattern)
#     nums = [int(rx.match(f.name).group(1))
#             for f in out_dir.glob("batch_*.pt") if rx.match(f.name)]
#     return max(nums) if nums else -1

# def _to_fp16(d):               # cast whole prompt-dict to fp16
#     return {k: v.half() for k, v in d.items()}

# def _is_clean(path: Path) -> bool:
#     blob = torch.load(path, map_location="cpu")
#     for d in blob.values():                 # one dict per prompt
#         if torch.isnan(d["c_crossattn"]).any() or torch.isinf(d["c_crossattn"]).any():
#             return False
#         if torch.isnan(d["y"]).any() or torch.isinf(d["y"]).any():
#             return False
#     return True

# # ───────────────────────── settings ─────────────────────────────────
# DEVICE     = "cuda"
# BATCH_SIZE = 50
# OUT_DIR    = Path("cached_prompts")
# OUT_DIR.mkdir(exist_ok=True)

# last_done  = find_last_finished_batch(OUT_DIR)
# start_idx  = (last_done + 1) * BATCH_SIZE

# # ───────────────────────── load inferencer ──────────────────────────
# inferencer = SD3Inferencer()
# inferencer.load(
#     model="models/sd3.5_medium.safetensors",
#     vae=None,
#     shift=3.0,
#     controlnet_ckpt=None,
#     model_folder="models",
#     text_encoder_device="cuda",
#     load_non_tokenizers=False
# )
# print("finished loading")

# # ───────────────────────── dataset prompts ──────────────────────────
# coco_json   = "annotations/merged_one_caption_per_image.json"
# coco_images = "val2014"
# dataset     = COCOPromptDataset(
#     annotation_path=coco_json,
#     image_root=coco_images,
#     # transform=transforms.Compose([
#     #     transforms.Resize((512, 512)),
#     #     transforms.ToTensor()
#     # ]),
#     subset_size=None,
#     seed=123
# )

# prompts = sorted(set(
#     tqdm((entry["caption"] for entry in dataset.captions),
#          total=len(dataset.captions),
#          desc="Extracting prompts")
# ))
# print(f"Total prompts: {len(prompts)} | Batch size: {BATCH_SIZE}")

# # ───────────────────────── batched pre-compute ──────────────────────
# with torch.inference_mode():
#     for i in tqdm(range(start_idx, len(prompts), BATCH_SIZE), desc="Precomputing"):
#         batch_id      = last_done + 1 + (i - start_idx) // BATCH_SIZE
#         batch_prompts = prompts[i: i + BATCH_SIZE]

#         cond = inferencer.fix_cond(inferencer.get_cond_batch(batch_prompts))

#         batch_dict = {
#             p: {
#                 "c_crossattn": cond["c_crossattn"][j].cpu(),  # fp32
#                 "y":           cond["y"][j].cpu()
#             }
#             for j, p in enumerate(batch_prompts)
#         }

#         tmp_path  = OUT_DIR / f"batch_{batch_id:05d}.tmp"
#         final_path = OUT_DIR / f"batch_{batch_id:05d}.pt"

#         torch.save(batch_dict, tmp_path)
#         os.replace(tmp_path, OUT_DIR / f"batch_{batch_id:05d}.pt")
#         # ── integrity check ────────────────────────────────────────
#         if not _is_clean(final_path):
#             raise RuntimeError(f"NaNs or Infs detected in {final_path}; "
#                             f"check text-encoder output before saving.")

#         del cond, batch_dict
#         gc.collect()
#         torch.cuda.empty_cache()

# # ───────────────────────── unconditional prompt ─────────────────────
# uncond = inferencer.fix_cond(inferencer.get_cond_batch([""]))
# torch.save({
#     "c_crossattn": uncond["c_crossattn"].cpu(),
#     "y":           uncond["y"].cpu()
# }, OUT_DIR / "uncond.pt")

# print("✓ all batches written to", OUT_DIR)