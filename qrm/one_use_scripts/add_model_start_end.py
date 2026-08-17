import os
import torch

def add_qrm_metadata(base_dir, checkpoint_folders, start, end):
    """
    Add qrm_start_step and qrm_end_epoch to all checkpoints in given folders.

    Args:
        base_dir (str): Path to 'sd3.5/models'
        checkpoint_folders (list[str]): List of folder names under base_dir
        start (int): Value for qrm_start_step
        end (int): Value for qrm_end_epoch
    """
    for folder in checkpoint_folders:
        folder_path = os.path.join(folder)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Skipping missing folder {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            if not fname.endswith(".pth"):
                continue
            ckpt_path = os.path.join(folder_path, fname)
            print(f"Updating {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint["qrm_start_step"] = start
            checkpoint["qrm_end_step"] = end
            torch.save(checkpoint, ckpt_path)

    print("✅ Done updating checkpoints.")

if __name__ == "__main__":
    # Example usage:
    base_dir = "sd3.5/models"
folders_15 = [
    "20250911_233511_x_k_vs_x_k_baseline_transformer_lr1e-6_t15_sched",
    "20250911_150444_x_k_vs_x_k_baseline_transformer_lr1e-6_t15_sched",
    "20250911_063000_x_k_vs_x_k_baseline_transformer4_lr1e-6_t15_sched",
]
folders_30 = [
"20250911_185442_x_k_vs_x_k_baseline_transformer_lr1e-6_t30_sched",
"20250911_101946_x_k_vs_x_k_baseline_transformer4_lr1e-6_t30_sched",
]
add_qrm_metadata(base_dir, folders_15, start=15, end=47)
add_qrm_metadata(base_dir, folders_30, start=30, end=47)
