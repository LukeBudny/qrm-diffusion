import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import torch
from tqdm import tqdm
from sd3_infer import SD3Inferencer
import argparse
from qrm.qrm_models import QRMRegistry
from qrm.qrm_lora import LoraInjectedLinear
import random
import numpy as np
from itertools import islice

# === CONFIGURATION ===
CAPTIONS_PATH = "annotations/captions_val2014.json"
OUT_DIR = "eval_images"
MODEL_PATH = "models/sd3.5_medium.safetensors"
MODELS_PATH = "models.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROMPTS = -1
NUM_VARIANTS = 1

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_all_seeds(42)
SEED = 42

def infer_vision_feature_dim(inferencer):
    """
    Returns the output dimension of get_vision_feature() by running it
    once on a dummy latent and prompt.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_latent = torch.randn(1, 16, 128, 128).to(device)  # Assumes SD3 latent shape
    dummy_prompt = ["a photo of a cat"]
    text_inputs = inferencer.clip_tokenizer(dummy_prompt + [""], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        vision_feature = inferencer.get_vision_feature(dummy_latent, dummy_prompt,text_inputs)
    return vision_feature.shape[-1]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default=CAPTIONS_PATH)
    parser.add_argument("--sd3_path", type=str, default=MODEL_PATH)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--num_prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--num_variants", type=int, default=NUM_VARIANTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models_path", type=str,default=MODELS_PATH)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()

def load_lora_weights(mmditx_model, path="mmditx_lora.pth"):
    """
    Load LoRA weights from a saved checkpoint into your MM-DiT-X model.
    - mmditx_model: your model into which LoRA was previously injected
    - path: file path to load weights from
    """
    lora_state = torch.load(path)
    for module_name, module in mmditx_model.named_modules():
        if isinstance(module, LoraInjectedLinear):
            module.lora_up.load_state_dict(lora_state[module_name + ".lora_up"])
            module.lora_down.load_state_dict(lora_state[module_name + ".lora_down"])

def load_metadata_prompts(path, n):
    with open(path, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return lines if n == -1 else lines[:n]

def chunked(iterable, size):
    """Yield successive chunks of given size from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def main():

    args = parse_args()

    CAPTIONS_PATH = args.prompt_file
    MODEL_PATH = args.sd3_path
    OUT_DIR = args.out_dir
    DEVICE = args.device
    NUM_PROMPTS = args.num_prompts
    NUM_VARIANTS = args.num_variants
    SEED = args.seed
    with open(args.models_path, "r") as f:
        MODELS = json.load(f)

    BATCH_SIZE = 7

    metadata_list = load_metadata_prompts(CAPTIONS_PATH, NUM_PROMPTS)

    for model_name, qrm_ckpt in MODELS.items():
        print(f"\n🔧 Generating for model: {model_name}")
        inferencer = SD3Inferencer()

        sample_id = 0
        count = 0

        for metadata_batch in chunked(metadata_list, BATCH_SIZE):
            prompts_batch = [m["prompt"] for m in metadata_batch]

            if count == 0:
                if qrm_ckpt is not None:
                    checkpoint = torch.load(qrm_ckpt, map_location=DEVICE)
                else:
                    print("########### No qrm loaded ############")
                try:
                    eval_model = checkpoint.get("vision_feature_model","clip")
                    print("eval_model: ", eval_model)
                except:
                    eval_model = 'clip'
                    print("eval_model not found, using clip")

                print("is this not running????????????????????????????????????????????????????????")
                inferencer.load(
                    model="models/sd3.5_medium.safetensors",
                    vae=None,
                    shift=3.0,
                    controlnet_ckpt=None,
                    model_folder="models",
                    text_encoder_device="cuda",
                    load_tokenizers=True,
                    eval_model=eval_model,
                    inference = True
                )
                vision_dim = infer_vision_feature_dim(inferencer)

                inferencer.sd3.model.qrm = None

                for mod in inferencer.sd3.model.diffusion_model.modules():
                    if isinstance(mod, LoraInjectedLinear):
                        mod.reset_parameters()

                if qrm_ckpt is not None and "model" in checkpoint:
                    print("Basic check, this is actually running right??????????????????????????")
                    qrm_type = checkpoint.get("qrm_type", "mlp")
                    time_bool = checkpoint.get("time_bool", True)
                    inferencer.sd3.model.qrm = QRMRegistry[qrm_type](time_bool=time_bool,vision_dim = vision_dim)
                    inferencer.sd3.model.qrm.load_state_dict(checkpoint["model"])
                    inferencer.sd3.model.qrm_inference = True
                    lora_path = qrm_ckpt.replace("qrmmlp_joint", "mmditx_lora")  # or keep a table
                    if os.path.exists(lora_path):
                        print("lora running for some reason???")
                        load_lora_weights(inferencer.sd3.model.diffusion_model, lora_path)                    
                count +=1

                        # === Create folders and metadata for each image in batch ===
            batch_save_paths = []
            for b_idx, metadata in enumerate(metadata_batch):
                for variant in range(NUM_VARIANTS):
                    folder_id = f"{sample_id:05d}"
                    folder_path = os.path.join(OUT_DIR, model_name, folder_id)
                    samples_path = os.path.join(folder_path, "samples")
                    os.makedirs(samples_path, exist_ok=True)
                    save_name = f"{variant:04d}.png"
                    save_path = os.path.join(samples_path, save_name)
                    metadata_path = os.path.join(folder_path, "metadata.jsonl")

                    if not os.path.exists(save_path):
                        with open(metadata_path, "w") as f:
                            f.write(json.dumps(metadata) + "\n")

                    batch_save_paths.append((samples_path, save_name))
                sample_id += 1


            if qrm_ckpt is not None:
                #insert batch logic here
                print("starting batch logic")
            else:
                for prompt, (samples_path, save_name) in zip(prompts_batch, batch_save_paths):
                    if not os.path.exists(os.path.join(samples_path, save_name)):
                        inferencer.gen_image(
                            prompts=[prompt],
                            out_dir=samples_path,
                            seed=SEED,
                            steps=40,
                            cfg_scale=4.5,
                            save_names=[save_name]
                        )

            sample_id += 1
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

if __name__ == "__main__":
    main()
