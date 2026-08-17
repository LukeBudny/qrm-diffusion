from torchvision import transforms
import torch
import gc
import os
import subprocess
import time
import pandas as pd
import re
from collections import defaultdict, Counter
import json

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
valcoco2014_prompt_file = "annotations/coco_val_prompts.jsonl"
model_file = "models/sd3.5_medium.safetensors"
models_json = "models_parti.json"

# Redirect all output dirs when lightweight
geneval_output_dir = "geneval/comparative_geneval_images" if LIGHTWEIGHT else "geneval/comparative_images"
parti_output_dir = "geneval/comparative_parti_images"
valcoco2014_output_dir = "coco_val_images/comparative_initial_images" if LIGHTWEIGHT else "coco_val_images/comparative_images"
results_dir = "comparative_results" if LIGHTWEIGHT else "comparative_results"

script_path = "generate_eval_images.py"
venv_sd35 = os.path.join("sd35-env", "Scripts", "python.exe")
venv_geneval = venv_sd35 #os.path.join("geneval-env", "Scripts", "python.exe")


# === STEP 2A: Evaluate centralized baseline ONCE ===
baseline_dir = os.path.join(geneval_output_dir, "baseline")
baseline_result_file = os.path.join(results_dir, "baseline_eval.jsonl")

if os.path.isdir(baseline_dir):
    if os.path.exists(baseline_result_file):
        print(f" Skipping baseline, results already exist at {baseline_result_file}")
    else:
        print("🧪 Evaluating centralized baseline...")
        subprocess.run(
            f'"{venv_geneval}" geneval/evaluation/evaluate_images.py "{baseline_dir}" '
            f'--outfile "{baseline_result_file}" --model-path "geneval/pretrained_models"',
            shell=True,
            check=True
        )

# Build a fast lookup: (prompt, tag, image_id, variant) -> baseline_correct (0/1)
base_dict = {}
if os.path.exists(baseline_result_file):
    with open(baseline_result_file, "r") as f:
        for line in f:
            r = json.loads(line)
            fn = str(r["filename"]).replace("\\", "/")
            if "/samples/" not in fn:
                continue
            leaf = fn.rsplit("/", 1)[-1]                 # e.g., baseline_0001.png
            if not leaf.startswith("baseline_"):
                continue
            image_id = fn.split("/samples/")[0].rsplit("/", 1)[-1]
            variant = int(leaf.split("_")[1].split(".")[0])  # 0001 -> 1
            key = (r.get("prompt", ""), r.get("tag", ""), image_id, variant)
            base_dict[key] = 1 if str(r.get("correct")).lower() in ("1", "true") else 0
else:
    print(" Baseline not found; baseline deltas will be empty.")

# For diagnostics only (no logic changes)
base_by_full   = set(base_dict.keys())                 # (prompt, tag, image_id, variant)
base_by_idvar  = {(k[2], k[3]) for k in base_by_full}  # (image_id, variant)

print(f"[diag] baseline rows: {len(base_by_full)}")
print(f"[diag] unique (image_id,variant): {len(base_by_idvar)}")
print("[diag] baseline variant counts:", sorted(Counter(v for *_, v in base_by_full).items()))


# === STEP 2B: Evaluate each QRM model folder (QRM ONLY) ===
print(" Running Geneval evaluation for each model...")

with open(models_json, "r") as f:
    model_folders = list(json.load(f).keys())

all_results = []   # raw QRM rows (for downstream CSV merge)
pairs_rows  = []   # paired rows with baseline

for model_name in model_folders:
    if model_name == "baseline":
        continue  # baseline handled above

    model_path = os.path.join(geneval_output_dir, model_name)
    model_result_file = os.path.join(results_dir, f"{model_name}_eval.jsonl")

    if os.path.exists(model_result_file):
        print(f" Skipping {model_name}, results already exist at {model_result_file}")
    else:
        print(f" Evaluating model: {model_name}")
        subprocess.run(
            f'"{venv_geneval}" geneval/evaluation/evaluate_images.py "{model_path}" '
            f'--outfile "{model_result_file}" --model-path "geneval/pretrained_models"',
            shell=True,
            check=True
        )

    # Load QRM results and form pairs with centralized baseline
    with open(model_result_file, "r") as f:
        for line in f:
            r = json.loads(line)
            r["model"] = model_name
            all_results.append(r)

            fn = str(r["filename"]).replace("\\", "/")
            if "/samples/" not in fn:
                continue
            leaf = fn.rsplit("/", 1)[-1]          # e.g., 0001.png
            if leaf.startswith("baseline_"):
                continue  # we don't store per-model baselines anymore

            image_id = fn.split("/samples/")[0].rsplit("/", 1)[-1]
            variant  = int(leaf.split(".")[0])    # "0001" -> 1

            key = (r.get("prompt", ""), r.get("tag", ""), image_id, variant)
            if key not in base_dict:
                has_idvar = (image_id, variant) in base_by_idvar
                # Log only a handful so your console doesn't explode
                if "diag_misses" not in locals():
                    diag_misses = 0
                if diag_misses < 8:
                    print("\n[diag] MISSING PAIR")
                    print("  model_fn: ", fn)
                    print("  key     : ", key)
                    print("  reason  : ", "prompt/tag mismatch" if has_idvar else "baseline missing this (image_id,variant)")
                    if has_idvar:
                        # show what baseline has for this (image_id,variant)
                        candidates = [k for k in base_by_full if (k[2], k[3]) == (image_id, variant)]
                        print("  baseline candidate key(s): ", candidates[:2])
                        if candidates:
                            bp, bt, _, _ = candidates[0]
                            print("  cmp prompt equal? ", bp == r.get("prompt", ""))
                            print("  cmp tag equal?    ", bt == r.get("tag", ""))
                    diag_misses += 1
                continue

            qrm_correct      = 1 if str(r.get("correct")).lower() in ("1", "true") else 0
            baseline_correct = base_dict[key]
            pairs_rows.append({
                "model": model_name,
                "prompt": r.get("prompt", ""),
                "tag":    r.get("tag", ""),
                "image_id": image_id,
                "variant": variant,
                "qrm_correct": qrm_correct,
                "baseline_correct": baseline_correct,
                "diff_geneval": float(qrm_correct - baseline_correct),
            })

# === STEP 3: Save all combined results as CSV ===
print("Saving combined CSV to results/qrm_eval_results.csv")

# QRM rows (raw) for downstream merges
df = pd.DataFrame(all_results).assign(source="geneval")
if "metadata" in df.columns:
    df["metadata"] = df["metadata"].apply(json.loads)

pairs = pd.DataFrame(pairs_rows)

pairs = pd.DataFrame(pairs_rows) if pairs_rows else pd.DataFrame(columns=["variant"])
print(f"[diag] paired rows: {len(pairs_rows)}")
if not pairs.empty and "variant" in pairs.columns:
    print("[diag] paired variant counts:", sorted(pairs["variant"].value_counts().sort_index().items()))

# --- Summaries from pairs ---
task_categories = ["single object", "two object", "counting", "colors", "position", "color attribution"]  # use your exact tag strings

g = pairs.groupby(["model", "tag"])
task_summary_delta = g["diff_geneval"].mean().unstack(fill_value=0).add_suffix("_delta")

overall_delta        = (pairs.groupby(["model","prompt"])["diff_geneval"].mean()
                             .groupby("model").mean().rename("overall"))
overall_baseline_acc = pairs.groupby("model")["baseline_correct"].mean().rename("overall_baseline_acc")
overall_qrm_acc      = pairs.groupby("model")["qrm_correct"].mean().rename("overall_qrm_acc")

# --- Merge with CLIP/BLIP/ImageReward (standardized columns) ---
clip_scores_file = os.path.join(geneval_output_dir, "evaluation_all_scores.csv")
if os.path.exists(clip_scores_file):
    df_clip = pd.read_csv(clip_scores_file)
    agg_cols = [
        "qrm_clip_score", "qrm_image_reward", "qrm_hpsv2_reward",
        "baseline_clip_score", "baseline_image_reward", "baseline_hpsv2_reward",
        "diff_clip_score", "diff_image_reward", "diff_hpsv2_reward",
    ]
    present = [c for c in agg_cols if c in df_clip.columns]
    if present:
        df_clip_agg = df_clip.groupby(["model", "prompt"], as_index=False)[present].mean()
        df = pd.merge(df, df_clip_agg, on=["model", "prompt"], how="left")

# Ensure downstream keys exist even if image_id was aggregated away elsewhere
if "image_id" not in df.columns:
    df["image_id"] = "n/a"

csv_path = os.path.join(results_dir, "qrm_eval_results.csv")
df_combined = df

if os.path.exists(csv_path):
    existing = pd.read_csv(csv_path)
    combined = pd.concat([existing, df_combined], ignore_index=True)
else:
    combined = df_combined

subset_cols = [c for c in ["model", "filename", "source"] if c in combined.columns]
if subset_cols:
    combined = combined.drop_duplicates(subset=subset_cols, keep="last")

combined.to_csv(csv_path, index=False)
print("Merge completed. Columns:", combined.columns.tolist())

# Build per-model summary table
geneval = combined[combined["source"] == "geneval"].copy()

pieces = [
    overall_delta,
    overall_baseline_acc,
    overall_qrm_acc,
    task_summary_delta,  # keep only _delta per-tag
]

for name, out in [
    ("diff_clip_score",   "mean_clip_geneval"),
    ("diff_image_reward", "mean_reward_geneval"),
    ("diff_hpsv2_reward", "mean_hpsv2_geneval"),
]:
    if name in geneval.columns:
        pieces.append(geneval.groupby("model")[name].mean().rename(out))

# keep absolute geneval means (not per-tag)
for col in [
    "baseline_clip_score","baseline_image_reward","baseline_hpsv2_reward",
    "qrm_clip_score","qrm_image_reward","qrm_hpsv2_reward",
]:
    if col in geneval.columns:
        pieces.append(geneval.groupby("model")[col].mean().rename(f"{col}_geneval"))

full_summary = pd.concat(pieces, axis=1).reset_index().round(4)

# ---- Keep only the desired columns (drops coco + tag-specific baseline/qrm) ----
desired_order = (
    ["model", "overall", "mean_clip_geneval", "mean_reward_geneval", "mean_hpsv2_geneval"] +
    [f"{t}_delta" for t in task_categories] +
    ["overall_baseline_acc", "overall_qrm_acc",
     "baseline_clip_score_geneval", "baseline_image_reward_geneval", "baseline_hpsv2_reward_geneval",
     "qrm_clip_score_geneval",      "qrm_image_reward_geneval",      "qrm_hpsv2_reward_geneval"]
)
cols_present = [c for c in desired_order if c in full_summary.columns]
full_summary = full_summary.reindex(columns=cols_present)

# Write, aligning old files to the same schema (prevents coco cols from resurfacing)
task_csv_path = os.path.join(results_dir, "qrm_eval_model_task_breakdown.csv")
if os.path.exists(task_csv_path):
    old_summary = pd.read_csv(task_csv_path)
    old_summary = old_summary.reindex(columns=cols_present)
    full_summary = pd.concat([old_summary, full_summary], ignore_index=True)
    full_summary = full_summary.drop_duplicates(subset=["model"], keep="last")
full_summary = full_summary.reindex(columns=cols_present)
full_summary.to_csv(task_csv_path, index=False)
print("Task-level breakdown (with scores) saved to:", task_csv_path)
