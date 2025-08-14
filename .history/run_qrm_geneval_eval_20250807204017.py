import os
import subprocess
import json
import pandas as pd


# "20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched_e01": "models/20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched/qrmmlp_joint_epoch_1.pth",
# "20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched_e02": "models/20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched/qrmmlp_joint_epoch_2.pth",
# "20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched_e05": "models/20250730_131417_transformer_10ep_rf_loss_lr5e-6_add_qt_one_step_train_sched/qrmmlp_joint_epoch_5.pth",
# "20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched_e01": "models/20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched/qrmmlp_joint_epoch_1.pth",
# "20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched_e02": "models/20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched/qrmmlp_joint_epoch_2.pth",
# "20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched_e05": "models/20250730_135242_transformer_10ep_rf_loss_lr1e-4_add_qt_one_step_train_sched/qrmmlp_joint_epoch_5.pth",
# "20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched_e00": "models/20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched/qrmmlp_joint_epoch_0.pth",
# "20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched_e01": "models/20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched/qrmmlp_joint_epoch_1.pth",
# "20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched_e02": "models/20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched/qrmmlp_joint_epoch_2.pth",
# "20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched_e03": "models/20250730_071812_transformer_10ep_rf_loss_lr1e-5_add_qt_sched/qrmmlp_joint_epoch_3.pth",



# === CONFIGURATION ===
geneval_prompt_file = "geneval/prompts/evaluation_metadata.jsonl"
valcoco2014_prompt_file = "annotations/coco_val_prompts.jsonl"
model_file = "models/sd3.5_medium.safetensors"
models_json = "models.json"
geneval_output_dir = "geneval/images"
valcoco2014_output_dir = "coco_val_images/images"
results_dir = "results"
summary_json = "results/qrm_eval_summary.json"
script_path = "generate_eval_images.py"
venv_sd35 = os.path.join("sd35-env", "Scripts", "python.exe")
venv_geneval = os.path.join("geneval-env", "Scripts", "python.exe")

# Step 1A: Generate Images for Geneval (uses SD3 + QRM)
subprocess.run([
    venv_sd35, "generate_images.py",
    "--prompt_file", geneval_prompt_file,
    "--sd3_path", model_file,
    "--models_path", models_json,
    "--out_dir", geneval_output_dir,
    "--num_prompts", "-1",
    "--num_variants", "1",
    "--seed", "42"
], check=True)

# Step 1B: Evaluate Images for Geneval (CLIP, BLIP, ImageReward)
subprocess.run([
    venv_sd35, "evaluate_image_metrics.py",
    geneval_output_dir,
    "--outdir", geneval_output_dir
], check=True)

# # Step 1C: Generate Images for Validation COCO 2014 (uses SD3 + QRM)
# subprocess.run([
#     venv_sd35, "generate_images.py",
#     "--prompt_file", valcoco2014_prompt_file,
#     "--sd3_path", model_file,
#     "--models_path", models_json,
#     "--out_dir", valcoco2014_output_dir,
#     "--num_prompts", "-1",
#     "--num_variants", "1",
#     "--seed", "42"
# ], check=True)

# # Step 1D: Evaluate generated Images for Validation COCO 2014 (uses SD3 + QRM)
# subprocess.run([
#     venv_sd35, "evaluate_image_metrics.py",
#     valcoco2014_output_dir,
#     "--outdir", valcoco2014_output_dir
# ], check=True)

# to be added later
# # Step 1E: Evaluate real Images for Validation COCO 2014 (uses SD3 + QRM)
# subprocess.run([
#     venv_sd35, "evaluate_image_metrics.py",
#     valcoco2014_output_dir,
#     "--outdir", valcoco2014_output_dir
# ], check=True)

# === STEP 2: Evaluate each model folder ===
print("📊 Running Geneval evaluation for each model...")

with open(models_json, "r") as f:
    model_folders = list(json.load(f).keys())
all_results = []

for model_name in model_folders:
    model_path = os.path.join(geneval_output_dir, model_name)
    model_result_file = os.path.join(results_dir, f"{model_name}_eval.jsonl")

    if os.path.exists(model_result_file):
        print(f"⏩ Skipping {model_name}, results already exist at {model_result_file}")
    else:    
        print(f"🧪 Evaluating model: {model_name}")
        subprocess.run(
        f'"{venv_geneval}" geneval/evaluation/evaluate_images.py "{model_path}" '
        f'--outfile "{model_result_file}" --model-path "geneval/pretrained_models"',
        shell=True,
        check=True
    )

    # Load results into DataFrame
    with open(model_result_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
        for row in data:
            row["model"] = model_name
        all_results.extend(data)

# === STEP 3: Summarize evaluation results ===
print("📈 Summarizing evaluation results...")

# subprocess.run([
#     venv_geneval, "geneval/evaluation/summary_scores.py", raw_eval_json], check=True)

# === STEP 3: Save all combined results as CSV ===
print("📁 Saving combined CSV to results/qrm_eval_results.csv")

df = pd.DataFrame(all_results)
if "metadata" in df.columns:
    df["metadata"] = df["metadata"].apply(json.loads)

csv_path = os.path.join(results_dir, "qrm_eval_results.csv")
# Load existing results if they exist
print("✅ Evaluation complete.")

print("📊 Appending per-model task breakdown to qrm_eval_model_task_breakdown.csv")

# Merge with CLIP/BLIP/ImageReward scores BEFORE breakdown
clip_scores_file = os.path.join(geneval_output_dir, "evaluation_all_scores.csv")
if os.path.exists(clip_scores_file):
    df_clip = pd.read_csv(clip_scores_file)

    # Strip image_id from df_clip only
    if "image_id" in df_clip.columns:
        df_clip["image_id"] = df_clip["image_id"].str.strip()

    # Merge — df will now include image_id after this
    df = pd.merge(df, df_clip, on=["model", "prompt"], how="left")

    # Strip image_id post-merge if needed
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].str.strip()
else:
    print(f"⚠️ Missing CLIP/BLIP/ImageReward file at {clip_scores_file}")

# Tag Geneval data
df["source"] = "geneval"

# === Load and tag COCO val scores ===
coco_scores_file = os.path.join(valcoco2014_output_dir, "evaluation_all_scores.csv")
if os.path.exists(coco_scores_file):
    df_coco = pd.read_csv(coco_scores_file)

    if "image_id" in df_coco.columns:
        df_coco["image_id"] = df_coco["image_id"].astype(str).str.strip()

    df_coco["source"] = "coco_val"

    # Add missing dummy columns for compatibility
    if "tag" not in df_coco.columns:
        df_coco["tag"] = "n/a"
    if "correct" not in df_coco.columns:
        df_coco["correct"] = pd.NA
else:
    print(f"⚠️ Missing COCO score file at {coco_scores_file}")
    df_coco = pd.DataFrame()

# === Combine Geneval and COCO into one evaluation DataFrame ===
df_combined = pd.concat([df, df_coco], ignore_index=True)
if os.path.exists(csv_path):
    existing = pd.read_csv(csv_path)
    # Combine old and new, then deduplicate
    combined = pd.concat([existing, df_combined], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model", "prompt", "source", "image_id"], keep="last")
else:
    combined = df_combined

# Save full (deduplicated) results
combined.to_csv(csv_path, index=False)

# === Now compute task-level breakdown using only Geneval data
df_geneval_only = df_combined[df_combined["source"] == "geneval"].copy()
tag_mapping = {
    "single_object": "single object",
    "two_object": "two object",
    "counting": "counting",
    "colors": "colors",
    "position": "position",
    "color_attr": "color attribution"
}
task_categories = ["single object", "two object", "counting", "colors", "position", "color attribution"]

df_geneval_only["tag"] = df_geneval_only["tag"].str.strip().str.lower()
df_geneval_only["tag_normalized"] = df_geneval_only["tag"].map(tag_mapping)

task_summary = (
    df_geneval_only.groupby(["model", "tag_normalized"])["correct"]
    .mean()
    .unstack(fill_value=0)
    .reindex(columns=task_categories, fill_value=0)
)

overall_accuracy = df_geneval_only.groupby("model")["correct"].mean().rename("overall")

print("✅ Merge completed. Columns in df:", df_combined.columns.tolist())
print("🧪 Sample df rows after merge:")
print(df_combined.head(2).to_dict(orient="records"))

required_cols = ["clip_score", "blip_relevance", "image_reward"]
missing_cols = [col for col in required_cols if col not in df_combined.columns]

if missing_cols:
    raise KeyError(f"❌ Missing expected columns after merge: {missing_cols}")

# Separate metric summaries for Geneval and COCO
mean_clip_geneval = df_combined[df_combined["source"] == "geneval"].groupby("model")["clip_score"].mean().rename("mean_clip_geneval")
mean_blip_geneval = df_combined[df_combined["source"] == "geneval"].groupby("model")["blip_relevance"].mean().rename("mean_blip_geneval")
mean_reward_geneval = df_combined[df_combined["source"] == "geneval"].groupby("model")["image_reward"].mean().rename("mean_reward_geneval")
mean_hpsv2_geneval = df_combined[df_combined["source"] == "geneval"].groupby("model")["hpsv2_reward"].mean().rename("mean_hpsv2_geneval")



mean_clip_coco = df_combined[df_combined["source"] == "coco_val"].groupby("model")["clip_score"].mean().rename("mean_clip_coco")
mean_blip_coco = df_combined[df_combined["source"] == "coco_val"].groupby("model")["blip_relevance"].mean().rename("mean_blip_coco")
mean_reward_coco = df_combined[df_combined["source"] == "coco_val"].groupby("model")["image_reward"].mean().rename("mean_reward_coco")
mean_hpsv2_coco = df_combined[df_combined["source"] == "coco_val"].groupby("model")["hpsv2_reward"].mean().rename("mean_hpsv2_coco")



# === Final Summary Table
full_summary = pd.concat(
    [
        overall_accuracy,
        mean_clip_geneval, mean_blip_geneval, mean_reward_geneval,mean_hpsv2_geneval,
        mean_clip_coco, mean_blip_coco, mean_reward_coco,mean_hpsv2_coco,
        task_summary
    ],
    axis=1
).reset_index().round(4)

task_csv_path = os.path.join(results_dir, "qrm_eval_model_task_breakdown.csv")
if os.path.exists(task_csv_path):
    old_summary = pd.read_csv(task_csv_path)
    full_summary = pd.concat([old_summary, full_summary], ignore_index=True)
    full_summary = full_summary.drop_duplicates(subset=["model"], keep="last")

full_summary.to_csv(task_csv_path, index=False)

print("📄 Task-level breakdown (with scores) saved to:", task_csv_path)
