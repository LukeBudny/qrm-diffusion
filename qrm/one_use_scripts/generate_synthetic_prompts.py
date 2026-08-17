import os
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ==== CONFIGURATION ====
IMAGE_DIR = "train2014"
OUTPUT_PATH = "synthetic_captions_blip2.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # BLIP-2 works well with fp16
GEN_KWARGS = {"max_new_tokens": 30}

# ==== LOAD MODEL ====
print("🔄 Loading BLIP-2 model and processor...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print("✅ Model loaded on device:", next(model.parameters()).device)

# ==== CAPTIONING ====
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
print(f"📸 Found {len(image_paths)} images.")

captions = {"annotations": [], "images": []}

for idx, image_path in enumerate(tqdm(image_paths, desc="🧠 Generating captions")):
    image_id = int(image_path.stem.split("_")[-1])  # COCO-style image_id
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)

    with torch.no_grad():
        start_time = time.time()
        output = model.generate(**inputs, **GEN_KWARGS)
        elapsed = time.time() - start_time
        print(f"🕒 {elapsed:.2f} sec/image")

        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    captions["annotations"].append({
        "image_id": image_id,
        "id": idx + 1,
        "caption": caption
    })
    captions["images"].append({
        "id": image_id,
        "file_name": image_path.name
    })

    if (idx + 1) % 1000 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"💾 Saved intermediate results at {idx + 1} images.")

# Final save
with open(OUTPUT_PATH, "w") as f:
    json.dump(captions, f, indent=2)

print(f"✅ Done! Captions saved to: {OUTPUT_PATH}")
