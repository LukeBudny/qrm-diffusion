from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random

class COCOPromptDataset(Dataset):
    def __init__(self, annotation_path, image_root, transform=None,subset_size=None, seed=42):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Map image_id to file_name
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        image_to_caption = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_to_caption:
                image_to_caption[img_id] = ann["caption"]

        self.captions = [{"image_id": k, "caption": v} for k, v in image_to_caption.items()]

        # === Subset selection ===
        if subset_size:
            random.seed(seed)
            random.shuffle(self.captions)
            self.captions = self.captions[:subset_size]

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_entry = self.captions[idx]
        image_id = caption_entry["image_id"]
        caption = caption_entry["caption"]
        image_path = os.path.join(self.image_root, self.image_id_to_filename[image_id])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return caption, image

# ------------------------------------------------------------------
# iterable_cache_dataset.py
# ------------------------------------------------------------------
import torch, random, glob
from torch.utils.data import IterableDataset
from pathlib import Path
from PIL import Image
import json, os

class CachedCOCOIterable(IterableDataset):
    """
    Streams (image, cond_dict) pairs, picking N random cached files per epoch.
    """
    def __init__(
        self,
        annotation_path: str,
        image_root: str,
        cache_dir: str,
        files_per_epoch: int = 3,
        seed: int = 42,
        transform=None,
    ):
        super().__init__()
        self.transform       = transform
        self.files_per_epoch = files_per_epoch
        self.rng             = random.Random(seed)
        self.image_root = image_root

        # ---- caption → image mapping --------------------------------
        data   = json.load(open(annotation_path))
        self.id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
        self.caption_map = {ann["caption"]: ann["image_id"] for ann in data["annotations"]}

        # ---- cached prompt files ------------------------------------
        self.pt_files = sorted(Path(cache_dir).glob("batch_*.pt"))
        self.uncond   = torch.load(Path(cache_dir) / "uncond.pt", map_location="cpu")

    def __iter__(self):
        # choose fresh random subset for *this* epoch
        chosen = self.rng.sample(self.pt_files, k=self.files_per_epoch)

        for fpath in chosen:
            batch = torch.load(fpath, map_location="cpu")  # dict[prompt]→cond

            for prompt, cond in batch.items():
                # lookup & load image
                img_id  = self.caption_map[prompt]
                img_p   = os.path.join(self.image_root, self.id_to_fname[img_id])
                image   = Image.open(img_p).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                yield {
                    "image"  : image,
                    "prompt" : prompt,
                    "cond"   : cond           # c_crossattn + y (CPU fp16)
                }
    def __len__(self):
        # 3 files × 133 prompts each  (adjust if you change either value)
        return self.files_per_epoch * 133