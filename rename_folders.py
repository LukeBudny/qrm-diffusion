import os

def normalize_image_folder_indices(base_path):
    """
    Renames all subfolders in each model's image directory to be zero-based,
    starting from 00000, 00001, ..., ordered by their original numeric names.
    """
    for model_name in os.listdir(base_path):
        model_dir = os.path.join(base_path, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"🔧 Normalizing folder names in: {model_name}")
        folders = [f for f in os.listdir(model_dir) if f.isdigit()]
        if not folders:
            continue

        sorted_folders = sorted(folders, key=int)
        min_index = int(sorted_folders[0])

        for old_name in sorted_folders:
            old_path = os.path.join(model_dir, old_name)
            new_index = int(old_name) - min_index
            new_name = f"{new_index:05d}"
            new_path = os.path.join(model_dir, new_name)
            print(f"🔁 Renaming {old_name} → {new_name}")
            os.rename(old_path, new_path)

    print("✅ Folder renaming complete.")

# Example usage
normalize_image_folder_indices("geneval/images")