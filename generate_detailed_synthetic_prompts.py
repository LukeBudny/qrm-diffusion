import os
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ==== CONFIGURATION ====
IMAGE_DIR = "train2014"
OUTPUT_PATH = "synthetic_captions_instructblip_v2.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
QUERY = "Describe this image in rich detail, including objects, colours, people, setting, and actions."

# ==== LOAD MODEL ====
print("🔄 Loading InstructBLIP model and processor...")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=False)
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xl",
    torch_dtype=DTYPE,
    device_map="auto"
).eval()

print("✅ Model loaded on device:", next(model.parameters()).device)

# ==== CAPTIONING ====


BATCH_SIZE = 6  # Try 8–16; depends on GPU memory
image_paths = sorted(Path(IMAGE_DIR).glob("*.jpg"))
print(f"📸 Found {len(image_paths)} images.")
captions = {"annotations": [], "images": []}

for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="🧠 Generating captions"):
    batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]
    batch_images, batch_ids = [], []

    for image_path in batch_paths:
        image_id = int(image_path.stem.split("_")[-1])
        image = Image.open(image_path).convert("RGB")
        batch_images.append(image)
        batch_ids.append(image_id)

    inputs = processor(images=batch_images, text=[QUERY]*len(batch_images), return_tensors="pt", padding=True).to(DEVICE, DTYPE)

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=3,            # <-- for speed
            max_new_tokens=256,      # <-- good balance
            repetition_penalty=1.2,
            length_penalty=1.0,
            temperature=1.0,
        )
        elapsed = (time.time() - start_time) / len(batch_images)
        print(f"🕒 {elapsed:.2f} sec/image (batch of {len(batch_images)})")

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)

    for image_id, path, caption in zip(batch_ids, batch_paths, decoded):
        captions["annotations"].append({
            "image_id": image_id,
            "id": image_id,
            "caption": caption.strip()
        })
        captions["images"].append({
            "id": image_id,
            "file_name": path.name
        })

    if (batch_start + BATCH_SIZE) % 100 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"💾 Saved intermediate results at {batch_start + BATCH_SIZE} images.")
