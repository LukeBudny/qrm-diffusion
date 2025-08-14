import json
import random

random.seed(123)

# Load COCO data
with open("annotations/captions_train2014.json", "r") as f:
    coco_data = json.load(f)
coco_annotations = coco_data["annotations"]
coco_images = {img["id"]: img for img in coco_data["images"]}

# Load synthetic data
with open("annotations/synthetic_captions_instructblip.json", "r") as f:
    synthetic_data = json.load(f)
synthetic_annotations = synthetic_data["annotations"]
synthetic_images = {img["id"]: img for img in synthetic_data["images"]}

# Group captions by image_id
from collections import defaultdict

coco_by_image = defaultdict(list)
for ann in coco_annotations:
    coco_by_image[ann["image_id"]].append(ann["caption"])

synthetic_by_image = defaultdict(list)
for ann in synthetic_annotations:
    synthetic_by_image[ann["image_id"]].append(ann["caption"])

# Union of all image_ids from both datasets
all_image_ids = list(set(coco_by_image.keys()) | set(synthetic_by_image.keys()))
random.shuffle(all_image_ids)

# Build new merged annotations with one caption per image
merged_annotations = []
merged_images = []
for image_id in all_image_ids:
    use_coco = random.random() < 0.5
    if use_coco and image_id in coco_by_image:
        caption = random.choice(coco_by_image[image_id])
    elif image_id in synthetic_by_image:
        caption = random.choice(synthetic_by_image[image_id])
    elif image_id in coco_by_image:  # fallback
        caption = random.choice(coco_by_image[image_id])
    else:
        continue  # skip if no caption at all

    merged_annotations.append({
        "image_id": image_id,
        "id": image_id,  # unique ID (can match image_id)
        "caption": caption
    })

    # Add the image record (assume synthetic and coco share image filenames)
    if image_id in coco_images:
        merged_images.append(coco_images[image_id])
    elif image_id in synthetic_images:
        merged_images.append(synthetic_images[image_id])
random.shuffle(merged_annotations)
# Final merged format
merged_json = {
    "annotations": merged_annotations,
    "images": merged_images
}

# Save to file
with open("annotations/merged_one_caption_per_image.json", "w") as f:
    json.dump(merged_json, f, indent=2)

print(f"✅ Saved {len(merged_annotations)} annotations across {len(merged_images)} images")
