# from sd3_infer import SD3Inferencer
# import torch, os, gc, re, json, csv
# from tqdm import tqdm
# from pathlib import Path

# # ───────────────────────── helpers ──────────────────────────────────
# def find_last_finished_batch(out_dir: Path, pattern=r"batch_(\d{5})\.pt") -> int:
#     rx = re.compile(pattern)
#     nums = [int(rx.match(f.name).group(1))
#             for f in out_dir.glob("batch_*.pt") if rx.match(f.name)]
#     return max(nums) if nums else -1

# def _is_clean(path: Path) -> bool:
#     blob = torch.load(path, map_location="cpu")
#     for d in blob.values():
#         if torch.isnan(d["c_crossattn"]).any() or torch.isinf(d["c_crossattn"]).any():
#             return False
#         if torch.isnan(d["y"]).any() or torch.isinf(d["y"]).any():
#             return False
#     return True

# def load_parti_prompts_tsv(tsv_path: str) -> list[str]:
#     """
#     Read PartiPrompts TSV and return a list of prompt strings (in file order).
#     Columns: Prompt, Category, Challenge, Note
#     """
#     prompts = []
#     with open(tsv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for row in reader:
#             p = (row.get("Prompt") or "").strip()
#             if p:
#                 prompts.append(p)
#     # de-duplicate while preserving order
#     seen = set()
#     unique_prompts = []
#     for p in prompts:
#         if p not in seen:
#             unique_prompts.append(p)
#             seen.add(p)
#     return unique_prompts

# # ───────────────────────── settings ─────────────────────────────────
# DEVICE     = "cuda"
# BATCH_SIZE = 50   # Adjust if needed

# # where you want Parti conds/unconds cached
# OUT_DIR    = Path("cached_parti_prompts")
# OUT_DIR.mkdir(exist_ok=True)

# PARTI_TSV  = "annotations/PartiPrompts.tsv"   # update path if needed

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
#     load_non_tokenizers=False   # keep consistent with your Geneval script
# )
# print("finished loading")

# # ───────────────────────── load Parti prompts ───────────────────────
# prompts = load_parti_prompts_tsv(PARTI_TSV)

# print(f"Total Parti prompts: {len(prompts)} | Batch size: {BATCH_SIZE}")

# # ───────────────────────── batched pre-compute ──────────────────────
# with torch.inference_mode():
#     for i in tqdm(range(start_idx, len(prompts), BATCH_SIZE), desc="Precomputing"):
#         batch_id      = last_done + 1 + (i - start_idx) // BATCH_SIZE
#         batch_prompts = prompts[i: i + BATCH_SIZE]

#         # identical to your Geneval path
#         cond = inferencer.fix_cond(inferencer.get_cond_batch(batch_prompts))

#         batch_dict = {
#             p: {
#                 "c_crossattn": cond["c_crossattn"][j].cpu(),
#                 "y":           cond["y"][j].cpu()
#             }
#             for j, p in enumerate(batch_prompts)
#         }

#         tmp_path   = OUT_DIR / f"batch_{batch_id:05d}.tmp"
#         final_path = OUT_DIR / f"batch_{batch_id:05d}.pt"

#         torch.save(batch_dict, tmp_path)
#         os.replace(tmp_path, final_path)

#         if not _is_clean(final_path):
#             raise RuntimeError(f"NaNs or Infs detected in {final_path}")

#         del cond, batch_dict
#         gc.collect()
#         torch.cuda.empty_cache()

# # ───────────────────────── unconditional prompt ─────────────────────
# uncond = inferencer.fix_cond(inferencer.get_cond_batch([""]))
# torch.save({
#     "c_crossattn": uncond["c_crossattn"].cpu(),
#     "y":           uncond["y"].cpu()
# }, OUT_DIR / "uncond.pt")

# print("✓ all Parti batches written to", OUT_DIR)


from sd3_infer import SD3Inferencer
import torch, os, gc, re, json
from tqdm import tqdm
from pathlib import Path


# ───────────────────────── settings ─────────────────────────────────
DEVICE     = "cuda"
BATCH_SIZE = 30   # Adjust if needed
OUT_DIR    = Path("cached_validation_prompts")
OUT_DIR.mkdir(exist_ok=True)


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

val_prompts = ["an elder politician giving a campaign speech","the word 'START' written in chalk on a sidewalk","a chess queen to the right of a chess knight","a view of the Big Dipper in the night sky",
    "Four deer surrounding a moose.","five chairs","matching socks with cute cats on them","The Oriental Pearl in oil painting","a plate with white rice topped by cooked vegetables","a scientist",
    "a yellow wall with the word KA-BOOM on it","a grumpy porcupine handing a check for $10,000 to a smiling peacock","Three-quarters front view of a yellow 2017 Corvette coming around a curve in a mountain road and looking over a green valley on a cloudy day.",
    "a t-shirt with Carpe Diem written on it","five frosted glass bottles","a can of Spam on an elegant plate","Portrait of a tiger wearing a train conductor's hat and holding a skateboard that has a yin-yang symbol on it. charcoal sketch",
    "a helicopter hovering over Times Square","A bowl of soup that looks like a monster knitted out of wool","a glass of orange juice with an orange peel stuck on the rim","a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
    "an abstract painting of a house on a mountain","a white robot passing a soccer ball to a red robot","a man chasing a cat","an airplane flying into a cloud that looks like monster","the Mona Lisa in the style of Minecraft","a man with puppet that looks like a king",
    "A photo of a Ming Dynasty vase on a leather topped table.","a portrait of a postal worker who has forgotten their mailbag","a chair"]


print(f"Total prompts: {len(val_prompts)} | Batch size: {BATCH_SIZE}")

# ───────────────────────── batched pre-compute ──────────────────────

with torch.inference_mode():

        # run sub-batches on GPU
        conds_all = {"c_crossattn": [], "y": []}
        cond_sub = inferencer.fix_cond(inferencer.get_cond_batch(val_prompts))

        conds_all["c_crossattn"].append(cond_sub["c_crossattn"].cpu())
        conds_all["y"].append(cond_sub["y"].cpu())

        conds_all["c_crossattn"] = torch.cat(conds_all["c_crossattn"], dim=0)
        conds_all["y"]           = torch.cat(conds_all["y"], dim=0)

        batch_dict = {
            p: {
                "c_crossattn": conds_all["c_crossattn"][j],
                "y":           conds_all["y"][j]
            }
            for j, p in enumerate(val_prompts)
        }

        final_path = OUT_DIR / f"batch_validation.pt"
        torch.save(batch_dict, final_path)

        del batch_dict, conds_all
        gc.collect()

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