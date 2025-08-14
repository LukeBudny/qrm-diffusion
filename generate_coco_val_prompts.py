import json
import random
from pycocotools.coco import COCO

coco = COCO("annotations/captions_val2014.json")
out_path = "annotations/coco_val_prompts.jsonl"

num_samples = 50
random.seed(42)

all_img_ids = coco.getImgIds()
selected_ids = random.sample(all_img_ids, num_samples)

with open(out_path, "w") as f:
    for img_id in selected_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for a in anns[:1]:  # use just 1 caption per image
            f.write(json.dumps({"prompt": a["caption"].strip(), "id": img_id}) + "\n")
