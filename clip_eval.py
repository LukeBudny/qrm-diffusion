import torch
import clip
from PIL import Image
import os
import argparse
import glob

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Default SD3.5 output directory
OUTPUT_DIR = "outputs/sd3.5_medium/"  # Change to "sd3.5_large" if needed

def get_latest_generated_image(output_dir):
    """Finds the most recent generated image inside the latest timestamped folder."""
    # Get all timestamped output folders
    folders = sorted(glob.glob(os.path.join(output_dir, "*")), key=os.path.getmtime, reverse=True)
    
    if not folders:
        print("❌ No generated output folders found in:", output_dir)
        return None, None

    latest_folder = folders[0]  # Most recent timestamped folder

    # Find image files inside the latest folder
    image_files = sorted(glob.glob(os.path.join(latest_folder, "*.png")) +
                         glob.glob(os.path.join(latest_folder, "*.jpg")) +
                         glob.glob(os.path.join(latest_folder, "*.jpeg")),
                         key=os.path.getmtime, reverse=True)
    
    if not image_files:
        print(f"❌ No images found in latest output folder: {latest_folder}")
        return None, None

    latest_image = image_files[0]  # Most recent image
    return latest_image, latest_folder

def extract_prompt_from_folder(folder_path):
    """Extracts the original prompt from the folder name."""
    basename = os.path.basename(folder_path)
    parts = basename.split("_")
    prompt = " ".join(parts[:-1])  # Remove timestamp
    return prompt.replace("-", " ")  # Convert hyphens to spaces

def evaluate_clip(image_path, prompt):
    """Computes CLIP similarity between an image and a text prompt."""
    image = Image.open(image_path).convert("RGB")  # Ensure it's an image
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    text_tokens = clip.tokenize([prompt]).to(device)

    # Compute embeddings
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # Normalize embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    clip_score = (image_features @ text_features.T).item()
    return clip_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=["medium", "large"], default="medium",
                        help="Choose between sd3.5_medium or sd3.5_large outputs.")
    
    args = parser.parse_args()
    
    # Set correct output directory based on model size
    output_dir = f"outputs/sd3.5_{args.model_size}/"
    
    latest_image, latest_folder = get_latest_generated_image(output_dir)
    if latest_image is None:
        exit(1)  # Exit if no images found

    prompt = extract_prompt_from_folder(latest_folder)
    print(f"🔍 Found latest image: {latest_image}")
    print(f"📜 Extracted Prompt: \"{prompt}\"")

    # Compute CLIP score
    score = evaluate_clip(latest_image, prompt)
    print(f"🎯 CLIP Similarity Score: {score:.4f}")
