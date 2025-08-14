from pathlib import Path
import json
import shutil

images_root = Path("/mnt/c/Users/lukes/Desktop/QRM Diffusion Project/sd3.5/geneval/images")
captions_path = Path("/mnt/c/Users/lukes/Desktop/QRM Diffusion Project/sd3.5/geneval/prompts/evaluation_metadata.jsonl")

with open(captions_path, "r") as f:
    prompt_order = [json.loads(line)["prompt"] for line in f if line.strip()]
prompt_to_index = {p: i for i, p in enumerate(prompt_order)}

for model_dir in images_root.iterdir():
    if not model_dir.is_dir():
        continue

    print(f"Processing model: {model_dir.name}")
    temp_dir = model_dir / "_temp_reorder"
    moved_any = False

    for folder in sorted(model_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_temp"):
            continue

        metadata_path = folder / "metadata.jsonl"
        if not metadata_path.exists():
            continue

        with open(metadata_path, "r") as f:
            metadata = json.loads(f.readline())

        print(metadata)
        prompt = metadata.get("prompt")

        if prompt not in prompt_to_index:
            print(f"  WARNING: Prompt not in prompt_order: {prompt}")
            continue

        target_idx = prompt_to_index[prompt]
        current_idx = int(folder.name)

        print(prompt_to_index)
        print(prompt)
        print(prompt_to_index[prompt])

        if current_idx != target_idx:  # Only move if wrong
            print(current_idx,prompt,target_idx,temp_dir)
            # moved_any = True
            # temp_dir.mkdir(exist_ok=True)
            # new_folder_name = f"{target_idx:05d}"
            # new_folder_path = temp_dir / new_folder_name

            # if new_folder_path.exists():
            #     print(f"  WARNING: Target folder already exists: {new_folder_name}")
            # shutil.move(str(folder), str(new_folder_path))
        
        print(sdfs)

    # Only move back if we moved something
    if moved_any:
        for folder in temp_dir.iterdir():
            shutil.move(str(folder), str(model_dir / folder.name))
        temp_dir.rmdir()

print("Reordering complete (only incorrect ones moved).")
