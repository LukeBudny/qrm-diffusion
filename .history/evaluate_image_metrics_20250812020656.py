# evaluate_images.py

import os, json, argparse
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval
import ImageReward as reward
import hpsv2
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(DEVICE)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
reward_model = reward.load("ImageReward-v1.0").to(DEVICE)

def evaluate_image(img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    print('###########################',prompt)
    print(img_path)
    print(fdgsdg)
    inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    clip_score = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()

    blip_inputs = blip_processor(images=image, text=prompt[0], return_tensors="pt").to(DEVICE)
    itm_logits = blip_model(**blip_inputs)[0]
    blip_score = torch.softmax(itm_logits, dim=-1)[0, 1].item()

    with torch.no_grad():
        reward_score = reward_model.score(prompt[0], img_path)

    hpsv2_score = float(hpsv2.score(img_path, prompt[0], hps_version="v2.1")[0])

    return clip_score, blip_score, reward_score,hpsv2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Path to generated eval_images folder")
    parser.add_argument("--outdir", type=str, default="eval_scores", help="Directory to store evaluation CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # === Load existing image evaluations for skipping ===
    existing_keys = set()
    all_scores_path = os.path.join(args.outdir, "evaluation_all_scores.csv")
    if os.path.exists(all_scores_path):
        existing_df = pd.read_csv(all_scores_path)
        existing_keys = set(zip(existing_df["model"], existing_df["image_id"]))

    all_rows = []
    mean_results = []

    for model_name in sorted(os.listdir(args.folder)):
        model_path = os.path.join(args.folder, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"🔍 Evaluating images for model: {model_name}")
        clip_scores, blip_scores, reward_scores,hpsv2_scores = [],[],[],[]
        model_rows = []

        for folder_id in sorted(os.listdir(model_path)):
            folder_path = os.path.join(model_path, folder_id)
            if not os.path.isdir(folder_path):
                continue

            metadata_path = os.path.join(folder_path, "metadata.jsonl")
            image_path = os.path.join(folder_path, "samples", "0000.png")
            image_id = f"{folder_id}/0000.png"
            key = (model_name, image_id)

            if not os.path.exists(metadata_path) or not os.path.exists(image_path):
                continue

            if key in existing_keys:
                print(f"⏩ Skipping {key}, already evaluated.")
                continue

            with open(metadata_path, "r") as f:
                metadata = json.loads(f.readline())
            prompt = metadata.get("prompt", "")

            clip_score, blip_score, reward_score, hpsv2_score = evaluate_image(image_path, prompt)

            clip_scores.append(clip_score)
            blip_scores.append(blip_score)
            reward_scores.append(reward_score)
            hpsv2_scores.append(hpsv2_score)

            row = {
                "model": model_name,
                "image_id": image_id,
                "prompt": prompt,
                "clip_score": clip_score,
                "blip_relevance": blip_score,
                "image_reward": reward_score,
                "hpsv2_reward": hpsv2_score,
            }
            model_rows.append(row)
        # print(hpsv2_scores)
        if model_rows:
            all_rows.extend(model_rows)
            mean_results.append({
                "model": model_name,
                "mean_clip_score": sum(clip_scores) / len(clip_scores),
                "mean_blip_relevance": sum(blip_scores) / len(blip_scores),
                "mean_image_reward": sum(reward_scores) / len(reward_scores),
                "mean_hpsv2_reward": sum(hpsv2_scores) / len(hpsv2_scores),
            })

    # === Save or append results ===
    summary_path = os.path.join(args.outdir, "evaluation_summary.csv")
    all_scores_path = os.path.join(args.outdir, "evaluation_all_scores.csv")

    pd.DataFrame(mean_results).to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)
    pd.DataFrame(all_rows).to_csv(all_scores_path, mode="a", header=not os.path.exists(all_scores_path), index=False)

    print(f"✅ Summary saved to: {summary_path}")
    print(f"📄 Full scores saved to: {all_scores_path}")

    # === Optional: Remove duplicate entries (model + image_id) ===
    print("🧹 De-duplicating full score table (if needed)...")
    df_all = pd.read_csv(all_scores_path)
    df_all = df_all.drop_duplicates(subset=["model", "image_id"])
    df_all.to_csv(all_scores_path, index=False)

if __name__ == "__main__":
    main()
