import os
import subprocess
import json
import pandas as pd

# === CONFIGURATION ===
# PartiPrompts dataset (general CLIP/BLIP/ImageReward evaluations)
parti_prompt_file = "path/to/parti_prompts.jsonl"
# Geneval prompts (existing Geneval evaluations)
geneval_prompt_file = "geneval/prompts/evaluation_metadata.jsonl"

model_file = "models/sd3.5_medium.safetensors"
models_json = "models.json"

# Output folders
general_output_dir = "general_eval/images"
geneval_output_dir = "geneval/images"

# Results folders
general_results_dir = "results/general"
geneval_results_dir = "results/geneval"

script_path = "generate_eval_images.py"
venv_sd35 = os.path.join("sd35-env", "Scripts", "python.exe")
venv_geneval = os.path.join("geneval-env", "Scripts", "python.exe")

# === STEP 1a: General evaluations on PartiPrompts ===
os.makedirs(general_output_dir, exist_ok=True)
os.makedirs(general_results_dir, exist_ok=True)
subprocess.run([
    venv_sd35, script_path,
    "--dataset", "parti",
    "--prompt_file", parti_prompt_file,
    "--sd3_path", model_file,
    "--models_path", models_json,
    "--out_dir", general_output_dir,
    "--num_prompts", "-1",
    "--num_variants", "1",
    "--seed", "42"
], check=True)

# === STEP 1b: Generate images for Geneval (no CLIP/BLIP/ImageReward) ===
os.makedirs(geneval_output_dir, exist_ok=True)
subprocess.run([
    venv_sd35, script_path,
    "--dataset", "geneval",
    "--prompt_file", geneval_prompt_file,
    "--sd3_path", model_file,
    "--models_path", models_json,
    "--out_dir", geneval_output_dir,
    "--num_prompts", "-1",
    "--num_variants", "1",
    "--seed", "42",
    "--eval_only"
], check=True)

# === STEP 2: Run Geneval evaluation metrics ===
print("📊 Running Geneval evaluation for each model...")
with open(models_json, "r") as f:
    model_folders = list(json.load(f).keys())
all_results = []

for model_name in model_folders:
    model_path = os.path.join(geneval_output_dir, model_name)
    result_file = os.path.join(geneval_results_dir, f"{model_name}_eval.jsonl")
    print(f"🧪 Evaluating model: {model_name}")
    subprocess.run(
        [venv_geneval,
         "geneval/evaluation/evaluate_images.py",
         model_path,
         "--outfile", result_file,
         "--model-path", "geneval/pretrained_models"],
        check=True
    )

    # Load results
    with open(result_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["model"] = model_name
            all_results.append(row)

# === STEP 3: Summarize Geneval results ===
os.makedirs(geneval_results_dir, exist_ok=True)
df = pd.DataFrame(all_results)
if "metadata" in df.columns:
    df["metadata"] = df["metadata"].apply(json.loads)

geneval_csv = os.path.join(geneval_results_dir, "qrm_eval_results.csv")
df.to_csv(geneval_csv, mode="a", header=not os.path.exists(geneval_csv), index=False)
print(f"Geneval evaluation complete. Results saved to {geneval_csv}")

# Task breakdown
task_categories = ["single object", "two object", "counting", "colors", "position", "color attribution"]
tag_mapping = {
    "single_object": "single object",
    "two_object": "two object",
    "counting": "counting",
    "colors": "colors",
    "position": "position",
    "color_attr": "color attribution"
}

# Normalize tags and compute per-task accuracy
if "tag" in df.columns:
    df["tag"] = df["tag"].str.strip().str.lower()
    df["tag_normalized"] = df["tag"].map(tag_mapping)
    task_summary = (
        df.groupby(["model", "tag_normalized"])["correct"]
          .mean()
          .unstack(fill_value=0)
          .reindex(columns=task_categories, fill_value=0)
    )
    overall = df.groupby("model")["correct"].mean().rename("overall")
    full = pd.concat([overall, task_summary], axis=1).reset_index().round(4)
    breakdown_csv = os.path.join(geneval_results_dir, "qrm_eval_model_task_breakdown.csv")
    full.to_csv(breakdown_csv, mode="a", header=not os.path.exists(breakdown_csv), index=False)
    print(f"Task-level breakdown saved to: {breakdown_csv}")

# Note: general evaluation summaries are already written under general_results_dir by generate_eval_images.py
