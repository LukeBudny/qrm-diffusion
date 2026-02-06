# evaluate_images.py

import os, json, argparse
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import ImageReward as reward
import hpsv2
import pandas as pd
from pathlib import Path
import time
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
reward_model = reward.load("ImageReward-v1.0").to(DEVICE)

def reward_score(img_path, prompts, hps_version: str = "v2.1",allow_grad: bool = True):
        """
        scorer: "clip" (default) or "hps"
        Returns a 1D tensor [B] on images_rgb.device.
        """
        images_rgb = Image.open(img_path).convert("RGB")

                    # PIL -> [1,3,H,W] float in [0,1]
        images_rgb = np.array(images_rgb.convert("RGB"), dtype=np.float32) / 255.0
        images_rgb = torch.from_numpy(images_rgb).permute(2,0,1).unsqueeze(0).to(device="cuda")  # BCHW


        device = "cuda"
        B = images_rgb.shape[0]

        # normalize prompts to list of length B
        if isinstance(prompts, str):
            prompts = [prompts] * B
        elif isinstance(prompts, list) and len(prompts) == 1 and B > 1:
            prompts = prompts * B

        image_size = 224
        if images_rgb.shape[-2:] != (image_size, image_size):
            pixel_values = F.interpolate(images_rgb, size=(image_size, image_size),
                                        mode="bicubic", align_corners=False, antialias=True)
        else:
            pixel_values = images_rgb
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        pixel_values = (pixel_values - mean) / std

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
        return scores.to(device=device, dtype=images_rgb.dtype).item()


def evaluate_image(img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True,truncation=True).to(DEVICE)
    # clip_start = time.time() 
    outputs = clip_model(**inputs)
    clip_score = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
    # print("clip_score time: ",time.time()-clip_start)
    with torch.no_grad():
        # ir_start = time.time() 
        ir_score = reward_model.score(prompt, img_path)
        # print("ir time: ",time.time()-ir_start)
        # hps_start = time.time() 
        hpsv2_score = reward_score(img_path, prompt, hps_version="v2.1")
        # print("hps time: ",time.time()-hps_start)

    # print(f"[diag] inputs device: {inputs['pixel_values'].device}")

    return clip_score, ir_score, hpsv2_score


def expected_keys_for_model(model_dir: str, num_variants: int) -> set[tuple]:
    out = set()
    model_name = os.path.basename(model_dir.rstrip("/\\"))
    for folder_id in sorted(os.listdir(model_dir)):
        samples_dir = os.path.join(model_dir, folder_id, "samples")
        if not os.path.isdir(samples_dir):
            continue
        for v in range(1, num_variants + 1):
            img_name = f"{v:04d}.png"
            qrm_img = os.path.join(samples_dir, img_name)
            if os.path.exists(qrm_img):
                image_id = f"{folder_id}/{img_name}"
                out.add((model_name, image_id, v))
    return out


def _load_existing_keys(all_scores_path: str) -> set[tuple]:
    if not os.path.exists(all_scores_path):
        return set()
    df = pd.read_csv(all_scores_path)
    if "variant" in df.columns:
        # normalize to int for set keys
        df["variant"] = pd.to_numeric(df["variant"], errors="coerce").fillna(1).astype(int)
        return set(zip(df["model"], df["image_id"], df["variant"]))
    else:
        # very old format w/o variant; assume 1
        return set(zip(df["model"], df["image_id"], [1] * len(df)))


def evaluate_folder(
    folder: str,
    outdir: str = "eval_scores",
    num_variants: int = 1,
    models_path: str | None = None,
):
    """
    Evaluate a generated images root (folder) and append results to CSVs in outdir.

    folder: images root that contains:
        <folder>/<model>/<00000>/samples/0001.png
        <folder>/baseline/<00000>/samples/baseline_0001.png
    outdir: where to write evaluation CSVs
    num_variants: how many variants per prompt to look for (e.g., 1 for Parti)
    models_path: optional path to a models.json-like file; if given, restrict eval
                 to the intersection of its model keys and the subdirs present.
    """
    os.makedirs(outdir, exist_ok=True)

    summary_path = os.path.join(outdir, "evaluation_summary.csv")
    all_scores_path = os.path.join(outdir, "evaluation_all_scores.csv")
    first_write = not (os.path.exists(summary_path) and os.path.exists(all_scores_path))

    existing_keys = _load_existing_keys(all_scores_path)
    baseline_root = os.path.join(folder, "baseline")

    # Decide which model subfolders to evaluate
    subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
    subdirs = [d for d in subdirs if d not in {"multi_step_image_sets", "baseline"}]

    if models_path is not None and os.path.exists(models_path):
        try:
            with open(models_path, "r", encoding="utf-8") as f:
                model_keys = set(json.load(f).keys())
            # Only evaluate models that are both in the folder and in the models file
            subdirs = [d for d in subdirs if d in model_keys]
        except Exception as e:
            print(f"[warn] Failed to read models from {models_path}: {e}")

    all_rows = []
    mean_results = []

    for model_name in subdirs:
        model_path = os.path.join(folder, model_name)
        print(model_path)

        # Early whole-model skip if everything already evaluated
        exp_keys = expected_keys_for_model(model_path, num_variants)
        if exp_keys and exp_keys.issubset(existing_keys):
            print(f"[SKIP MODEL] {model_name}: all {len(exp_keys)} pairs already evaluated.")
            continue

        print(f"Evaluating images for model: {model_name}")
        clip_scores, reward_scores, hpsv2_scores = [], [], []
        baseline_clip_scores, baseline_reward_scores, baseline_hpsv2_scores = [], [], []
        diff_clip_scores, diff_reward_scores, diff_hpsv2_scores = [], [], []
        model_rows = []

        for folder_id in sorted(os.listdir(model_path)):
            folder_path = os.path.join(model_path, folder_id)
            baseline_folder_path = os.path.join(baseline_root, folder_id)
            if not os.path.isdir(folder_path):
                continue

            # metadata.jsonl OR metadata.json
            meta_jsonl = os.path.join(folder_path, "metadata.jsonl")
            meta_json = os.path.join(folder_path, "metadata.json")
            if os.path.exists(meta_jsonl):
                print(meta_jsonl)
                with open(meta_jsonl, "r", encoding="utf-8") as f:
                    metadata = json.loads(f.readline())
            elif os.path.exists(meta_json):
                with open(meta_json, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                continue
            prompt = metadata.get("prompt", "")

            samples_dir = os.path.join(folder_path, "samples")
            baseline_samples_dir = os.path.join(baseline_folder_path, "samples")

            # 1-indexed variants: 0001.png, 0002.png, ...
            for v in range(1, num_variants + 1):
                idx_str = f"{v:04d}"
                img_name = f"{idx_str}.png"
                image_path = os.path.join(samples_dir, img_name)
                image_id = f"{folder_id}/{img_name}"
                key = (model_name, image_id, v)

                if not os.path.exists(image_path):
                    continue
                if key in existing_keys:
                    continue

                clip_score, reward_score, hpsv2_score = evaluate_image(image_path, prompt)

                baseline_path = os.path.join(baseline_samples_dir, f"baseline_{idx_str}.png")
                baseline_clip_score = baseline_reward_score = baseline_hpsv2_score = None
                diff_clip_score = diff_reward_score = diff_hpsv2_score = None

                if os.path.exists(baseline_path):
                    b_clip, b_reward, b_hps = evaluate_image(baseline_path, prompt)
                    baseline_clip_score = b_clip
                    baseline_reward_score = b_reward
                    baseline_hpsv2_score = b_hps

                    if clip_score is not None and baseline_clip_score is not None:
                        diff_clip_score = clip_score - baseline_clip_score
                    if reward_score is not None and baseline_reward_score is not None:
                        diff_reward_score = reward_score - baseline_reward_score
                    if hpsv2_score is not None and baseline_hpsv2_score is not None:
                        diff_hpsv2_score = hpsv2_score - baseline_hpsv2_score

                clip_scores.append(clip_score)
                reward_scores.append(reward_score)
                hpsv2_scores.append(hpsv2_score)

                if baseline_clip_score is not None:
                    baseline_clip_scores.append(baseline_clip_score)
                    baseline_reward_scores.append(baseline_reward_score)
                    baseline_hpsv2_scores.append(baseline_hpsv2_score)

                if diff_clip_score is not None:
                    diff_clip_scores.append(diff_clip_score)
                if diff_reward_score is not None:
                    diff_reward_scores.append(diff_reward_score)
                if diff_hpsv2_score is not None:
                    diff_hpsv2_scores.append(diff_hpsv2_score)

                row = {
                    "model": model_name,
                    "image_id": image_id,
                    "variant": v,
                    "prompt": prompt,
                    "qrm_clip_score": clip_score,
                    "qrm_image_reward": reward_score,
                    "qrm_hpsv2_reward": hpsv2_score,
                    "baseline_clip_score": baseline_clip_score,
                    "baseline_image_reward": baseline_reward_score,
                    "baseline_hpsv2_reward": baseline_hpsv2_score,
                    "diff_clip_score": diff_clip_score,
                    "diff_image_reward": diff_reward_score,
                    "diff_hpsv2_reward": diff_hpsv2_score,
                }
                model_rows.append(row)

        def safe_mean(xs):
            xs = [x for x in xs if x is not None]
            return float(sum(xs) / len(xs)) if xs else float("nan")

        if model_rows:
            # accumulate and append for this model
            all_rows.extend(model_rows)
            mean_results.append({
                "model": model_name,
                "mean_qrm_clip_score": safe_mean(clip_scores),
                "mean_qrm_image_reward": safe_mean(reward_scores),
                "mean_qrm_hpsv2_reward": safe_mean(hpsv2_scores),
                "mean_baseline_clip_score": safe_mean(baseline_clip_scores),
                "mean_baseline_image_reward": safe_mean(baseline_reward_scores),
                "mean_baseline_hpsv2_reward": safe_mean(baseline_hpsv2_scores),
                "mean_diff_clip_score": safe_mean(diff_clip_scores),
                "mean_diff_image_reward": safe_mean(diff_reward_scores),
                "mean_diff_hpsv2_reward": safe_mean(diff_hpsv2_scores),
                "num_pairs": max(len(diff_clip_scores), len(diff_reward_scores), len(diff_hpsv2_scores)),
            })

        # write after each model (append-only)
        if mean_results or all_rows:
            pd.DataFrame(mean_results).to_csv(summary_path, mode="a", header=first_write, index=False)
            pd.DataFrame(all_rows).to_csv(all_scores_path, mode="a", header=first_write, index=False)
            first_write = False
            mean_results.clear()
            all_rows.clear()

    print(f"Summary saved to: {summary_path}")
    print(f"Full scores saved to: {all_scores_path}")

    # de-dup rows by (model, image_id, variant)
    if os.path.exists(all_scores_path):
        df_all = pd.read_csv(all_scores_path)
        subset_cols = ["model", "image_id", "variant"] if "variant" in df_all.columns else ["model", "image_id"]
        df_all = df_all.drop_duplicates(subset=subset_cols)
        df_all.to_csv(all_scores_path, index=False)

    # After deduplication block
    if not os.path.exists(summary_path) and os.path.exists(all_scores_path):
        print("[info] Rebuilding evaluation_summary.csv from evaluation_all_scores.csv...")
        df_all = pd.read_csv(all_scores_path)
        summary = (
            df_all.groupby("model", as_index=False)
            .agg({
                "diff_clip_score": "mean",
                "diff_image_reward": "mean",
                "diff_hpsv2_reward": "mean",
                "qrm_clip_score": "mean",
                "qrm_image_reward": "mean",
                "qrm_hpsv2_reward": "mean",
                "baseline_clip_score": "mean",
                "baseline_image_reward": "mean",
                "baseline_hpsv2_reward": "mean",
            })
            .assign(num_pairs=df_all.groupby("model")["image_id"].nunique().values)
        )
        summary.to_csv(summary_path, index=False)
        print(f"[rebuild] Summary regenerated at {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to generated images root (e.g., geneval/comparative_initial_images)")
    parser.add_argument("--outdir", type=str, default="eval_scores", help="Directory to store evaluation CSVs")
    parser.add_argument("--num_variants", type=int, default=1, help="How many variants per folder to evaluate")
    parser.add_argument("--models_path", type=str, default=None, help="Optional models.json to restrict which models to evaluate")
    args = parser.parse_args()

    evaluate_folder(
        folder=args.folder,
        outdir=args.outdir,
        num_variants=args.num_variants,
        models_path=args.models_path,
    )


if __name__ == "__main__":
    main()
