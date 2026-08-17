import torch
import os

ckpt_paths = [

]

for path in ckpt_paths:
    checkpoint = torch.load(path)
    if "vision_feature_modeld" not in checkpoint:
        if "raw_clip" in path:
            checkpoint["vision_feature_model"] = "raw_clip"
        elif "blip" in path:
            checkpoint["vision_feature_model"] = "blip"
        elif "dinov2" in path:
            checkpoint["vision_feature_model"] = "dinov2"
        elif "concat" in path:
            checkpoint["vision_feature_model"] = "all"
        else:
            checkpoint["vision_feature_model"] = "clip"
        torch.save(checkpoint, path)
        print(f"✅ Patched: {path}")
