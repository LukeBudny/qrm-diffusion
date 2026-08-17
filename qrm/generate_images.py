import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import torch
from sd3_infer import SD3Inferencer
import argparse
from qrm.qrm_models import QRMRegistry
import random
import numpy as np
from itertools import islice
from pathlib import Path
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from qrm_diffusion.config import MemoryConfig
from qrm_diffusion.memory import apply_cuda_memory_policy

# === CONFIGURATION ===
CAPTIONS_PATH = "qrm/annotations/captions_val2014.json"
OUT_DIR = "geneval/comparative_initial_images"
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
    parser.add_argument("--only_image_ids_file", type=str, default=None)
    parser.add_argument("--only_image_ids", type=str, default=None)
    parser.add_argument("--lightweight_variants", type=int, default=None)
    # --- PartiPrompts (optional) ---
    parser.add_argument(
        "--parti_prompts_tsv",
        type=str,
        default="qrm/annotations/PartiPrompts.tsv",
        help="Path to PartiPrompts TSV (e.g., sd3.5/annotations/PartiPrompts.tsv). If not provided, the Parti pass is skipped.",
    )
    parser.add_argument(
        "--models_path_parti",
        type=str,
        default="qrm/model_jsons/models_parti.json",
        help="Path to models.json to use for PartiPrompts run (default: models_parti.json).",
    )
    parser.add_argument(
        "--models_path_geneval",
        type=str,
        default="qrm/model_jsons/models_geneval.json",
        help="Path to models.json to use for Geneval run (default: models_geneval.json).",
    )
    parser.add_argument(
        "--out_dir_parti",
        type=str,
        default=os.path.join("experiments", "comparative_parti_images"),
        help="Output root for PartiPrompts images; internal structure mirrors the original loop.",
    )
    parser.add_argument(
        "--out_dir_geneval",
        type=str,
        default=os.path.join("experiments", "comparative_geneval_images"),
        help="Output root for geneval images; internal structure mirrors the original loop.",
    )
    parser.add_argument(
        "--parti_num_prompts",
        type=int,
        default=None,
        help="Optional cap for number of PartiPrompts (useful for smoke tests).",
    )
    return parser.parse_args()

def load_metadata_prompts(path, n):
    with open(path, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return lines if n == -1 else lines[:n]

def load_parti_prompts_tsv(path: str, limit: int | None = None):
    """
    Reads a TSV with header:
      Prompt\tCategory\tChallenge\tNote
    Returns a list of dicts: {"prompt", "category", "challenge", "note"}
    """
    import csv
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, r in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append({
                "prompt":   (r.get("Prompt") or "").strip(),
                "category": (r.get("Category") or "").strip(),
                "challenge":(r.get("Challenge") or "").strip(),
                "note":     (r.get("Note") or "").strip(),
            })
    return rows

def chunked(iterable, size):
    """Yield successive chunks of given size from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

class CachedCondsIterable(IterableDataset):
    def __init__(self, cache_dir: str):
        super().__init__()
        self.pt_files = sorted(Path(cache_dir).glob("batch_*.pt"))
        self.uncond   = torch.load(Path(cache_dir) / "uncond.pt", map_location="cpu")

    def __iter__(self):
        for f in self.pt_files:
            batch = torch.load(f, map_location="cpu")      # {prompt: {"c_crossattn":..., "y":...}}
            for prompt, cond in batch.items():
                yield prompt, cond, self.uncond

def run_generation_from_cached(
    *,
    args,
    models_path: str,
    out_dir: str,
    cache_dir: Path,
    prompt_order: list[str],
    metadata_map: dict[str, dict],
    source_tag: str | None = None,
    allowed_indices: set[int] | None = None,
):
    DEVICE   = args.device
    SEED     = args.seed
    variants = args.lightweight_variants if args.lightweight_variants is not None else args.num_variants

    # build ordered iterable from cached conds
    loader = DataLoader(CachedCondsIterable(cache_dir), batch_size=1, shuffle=False)
    order_map = {p: i for i, p in enumerate(prompt_order)}
    sorted_batch = sorted(loader, key=lambda x: order_map.get(x[0][0], float("inf")))

    with open(models_path, "r") as f:
        MODELS = json.load(f)

    # initial lightweight load (skip tokenizers since we’re using cached conds)
    inferencer = SD3Inferencer()
    inferencer.load(
        model=args.sd3_path,
        vae=None,
        shift=1.0,
        controlnet_ckpt=None,
        model_folder="models",
        text_encoder_device="cuda",
        load_tokenizers=False,           # cached conds path
        eval_model="clip",
        inference=True,
    )
    inferencer.sd3.model.qrm = None

    TARGET_IMAGE_ID = ["43","263","349","490","540","541","693","705","764","767","797","840","870","890","900","1100","1188","1101","1205","1401","1442","1553","1555","1511","1579","1617"]
    TARGET_VARIANT  = "1"

    for model_name, qrm_ckpt in MODELS.items():
        print(f"\n Generating for model: {model_name}")

        sample_id = 0
        count = 0
        qrm_start_step = None
        qrm_end_step   = None

        for i, (prompt, cond_cpu, uncond_cpu) in enumerate(sorted_batch):
            prompt_text = prompt[0] if isinstance(prompt, tuple) else prompt

            if allowed_indices is not None and sample_id not in allowed_indices:
                sample_id += 1
                continue

            folder_id    = f"{sample_id:05d}"
            folder_path  = os.path.join(out_dir, model_name, folder_id)
            base_folder  = os.path.join(out_dir, "baseline", folder_id)
            base_samples = os.path.join(base_folder, "samples")
            samples_path = os.path.join(folder_path, "samples")
            os.makedirs(samples_path, exist_ok=True)
            os.makedirs(base_samples, exist_ok=True)

            # metadata.jsonl
            meta_path = os.path.join(folder_path, "metadata.jsonl")
            if not os.path.exists(meta_path):
                with open(meta_path, "w", encoding="utf-8") as f:
                    meta = dict(metadata_map.get(prompt_text, {"prompt": prompt_text}))
                    if source_tag:
                        meta["source"] = source_tag
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # check if all variants already exist
            all_exist = True
            for v in range(variants):
                save_name = f"{v+1:04d}.png"
                save_path = os.path.join(samples_path, save_name)
                multi_step_path = os.path.join(out_dir, "multi_step_image_sets", model_name)
                if str(sample_id) in TARGET_IMAGE_ID:
                    multi_step_path = os.path.join(multi_step_path,folder_id)
                if not os.path.exists(save_path) or not os.path.exists(multi_step_path):
                    print(f"[diag] Missing variant {v+1} for sample {sample_id}") #: {save_path}
                    all_exist = False
                    break
            if all_exist:
                sample_id += 1
                continue

            # prepare cached conds
            uncond_cpu["c_crossattn"] = uncond_cpu["c_crossattn"].squeeze(1)
            uncond_cpu["y"]           = uncond_cpu["y"].squeeze(1)
            if count == 0:
                # lazily load QRM (no vision features/vision_dim needed)
                if qrm_ckpt is not None:
                    checkpoint = torch.load(qrm_ckpt, map_location=DEVICE)
                else:
                    print("########### No qrm loaded ############")
                    checkpoint = {}
                print(checkpoint.get("vision_feature_model", "clip"))
                print(qrm_ckpt)

                if qrm_ckpt is not None:
                    eval_model = checkpoint.get("vision_feature_model", "clip")
                    print("eval_model: ", eval_model)
                    inferencer = SD3Inferencer()
                    inferencer.load(
                        model=args.sd3_path,
                        vae=None,
                        shift=3.0,
                        controlnet_ckpt=None,
                        model_folder="models",
                        text_encoder_device="cuda",
                        load_tokenizers=False,   # still false; we’re not computing vision features
                        eval_model=eval_model,
                        inference=True,
                        qrm_type = checkpoint.get("qrm_type", True)
                    )
                    inferencer.sd3.model.qrm = None
                else:
                    print("eval_model not found, using clip")
                if qrm_ckpt is not None:
                    qrm_type = checkpoint.get("qrm_type", "mlp")
                    qrm_start_step = checkpoint.get("qrm_start_step", 25)
                    qrm_end_step   = checkpoint.get("qrm_end_step", 47)
                    if qrm_type in ["QRMModulatorLatentV2","QRMModulatorLatentV3","QRMModulatorLatentV4","QRMModulatorLatentV5","QRMModulatorLatentV6"]:
                        inferencer.sd3.model.qrm = QRMRegistry[qrm_type](inferencer.sd3.model._qrm_block_spans)
                    else:
                        inferencer.sd3.model.qrm = QRMRegistry[qrm_type]()
                    inferencer.sd3.model.qrm.load_state_dict(checkpoint["model"])
                    inferencer.sd3.model.qrm_inference = True
                    inferencer.sd3.model.qrm_type = qrm_type
                    inferencer.qrm_type = qrm_type
                count += 1

            # move cached conds to CUDA/fp16
            def to_cuda_fp16(d):
                c = d["c_crossattn"].to(DEVICE, dtype=torch.float16)
                y = d["y"].to(DEVICE, dtype=torch.float16)
                return {"c_crossattn": c, "y": y}

            pos = to_cuda_fp16(cond_cpu)
            neg = to_cuda_fp16(uncond_cpu)

            # paired generation
            for v in range(variants):
                display_idx   = v + 1
                qrm_name      = f"{display_idx:04d}.png"
                baseline_name = f"baseline_{display_idx:04d}.png"

                qrm_path  = os.path.join(samples_path, qrm_name)
                base_path = os.path.join(base_samples, baseline_name)

                need_qrm  = (qrm_ckpt is not None and "model" in checkpoint)
                this_seed = SEED + v

                # QRM image
                if need_qrm and not os.path.exists(qrm_path):
                    print("this is the QRM running")
                    inferencer.gen_image(
                        prompts=[prompt_text],
                        out_dir=samples_path,
                        seed_type="fixed",
                        seed=this_seed,
                        steps=50,
                        cfg_scale=5,
                        save_names=qrm_name,
                        cached_conds=pos,
                        cached_uncond=neg,
                        use_qrm=True,
                        qrm_start_step=qrm_start_step,
                        qrm_end_step=qrm_end_step,
                    )

                # Baseline image (only when QRM present, matching prior behavior)
                if need_qrm and not os.path.exists(base_path):
                    print("this is the the baseline running")
                    inferencer.gen_image(
                        prompts=[prompt_text],
                        out_dir=base_samples,
                        seed_type="fixed",
                        seed=this_seed,
                        steps=50,
                        cfg_scale=5,
                        save_names=baseline_name,
                        cached_conds=pos,
                        cached_uncond=neg,
                        use_qrm=False,
                        qrm_start_step=qrm_start_step,
                        qrm_end_step=qrm_end_step,
                    )

                # optional per-5-step dumps
                image_id_dir       = os.path.basename(os.path.dirname(samples_path))
                qrm_steps_dir      = os.path.join(out_dir, "multi_step_image_sets", model_name)
                baseline_steps_dir = os.path.join(out_dir, "multi_step_image_sets", "baseline")
                if str(sample_id) in TARGET_IMAGE_ID:
                    qrm_steps_dir = os.path.join(qrm_steps_dir,folder_id)
                    baseline_steps_dir = os.path.join(baseline_steps_dir,folder_id)
                # print(str(int(image_id_dir)), TARGET_IMAGE_ID,str(int(image_id_dir)) in TARGET_IMAGE_ID, display_idx, TARGET_VARIANT,int(display_idx) == TARGET_VARIANT, qrm_steps_dir,not os.path.exists(qrm_steps_dir),need_qrm)
                print(qrm_steps_dir, samples_path)
                if str(int(image_id_dir)) in TARGET_IMAGE_ID and str(int(display_idx)) == TARGET_VARIANT and not os.path.exists(qrm_steps_dir) and need_qrm:
                    print("this is the the per-5-step running")
                    inferencer.gen_image(
                        prompts=[prompt_text],
                        out_dir=samples_path,
                        seed_type="fixed",
                        seed=this_seed,
                        steps=50,
                        cfg_scale=5,
                        save_names=qrm_name,
                        cached_conds=pos,
                        cached_uncond=neg,
                        use_qrm=True,
                        save_per_5_step=True,
                        per_step_dir=qrm_steps_dir,
                        qrm_start_step=qrm_start_step,
                        qrm_end_step=qrm_end_step,
                    )

                base_meta_path = os.path.join(base_folder, "metadata.jsonl")
                if not os.path.exists(base_meta_path):
                    with open(base_meta_path, "w", encoding="utf-8") as f:
                        meta = dict(metadata_map.get(prompt_text, {"prompt": prompt_text}))
                        if source_tag:
                            meta["source"] = source_tag
                        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                if str(int(image_id_dir)) in TARGET_IMAGE_ID and str(int(display_idx)) == TARGET_VARIANT and not os.path.exists(baseline_steps_dir) and need_qrm:
                    print("this is the the per-5-step baseline running")
                    inferencer.gen_image(
                        prompts=[prompt_text],
                        out_dir=samples_path,
                        seed_type="fixed",
                        seed=this_seed,
                        steps=50,
                        cfg_scale=5,
                        save_names=baseline_name,
                        cached_conds=pos,
                        cached_uncond=neg,
                        use_qrm=False,
                        save_per_5_step=True,
                        per_step_dir=baseline_steps_dir,
                        qrm_start_step=qrm_start_step,
                        qrm_end_step=qrm_end_step,
                    )

            sample_id += 1

    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())


def main():

    args = parse_args()
    if torch.cuda.is_available():
        memory = apply_cuda_memory_policy(MemoryConfig())
        print(
            f"CUDA allocator budget: {memory.limit_gib:.2f} GiB "
            f"({memory.allocator_fraction:.4f} of {memory.total_gib:.2f} GiB)"
        )

    if args.only_image_ids_file or args.only_image_ids:
        ids = []
        if args.only_image_ids_file and os.path.exists(args.only_image_ids_file):
            with open(args.only_image_ids_file, "r") as f:
                ids += [ln.strip() for ln in f if ln.strip()]
        if args.only_image_ids:
            ids += [s.strip() for s in args.only_image_ids.split(",") if s.strip()]
        try:
            allowed_indices = set(int(x.split("/")[0]) for x in ids)
        except Exception as e:
            raise ValueError(f"Failed parsing only_image_ids: {e}")

    THIS_DIR = Path(__file__).resolve().parent

    # === Parti pass (cached) ===
    if getattr(args, "parti_prompts_tsv", None):
        parti_rows = load_parti_prompts_tsv(args.parti_prompts_tsv, getattr(args, "parti_num_prompts", None))
        if parti_rows:
            parti_prompt_order = [r["prompt"] for r in parti_rows if r["prompt"]]
            parti_metadata_map = {r["prompt"]: r for r in parti_rows}
            parti_cache = THIS_DIR / "precomputed_prompt_folder/cached_parti_prompts"
            args.lightweight_variants = 1

            run_generation_from_cached(
                args=args,
                models_path=args.models_path_parti,
                out_dir=args.out_dir_parti,
                cache_dir=parti_cache,
                prompt_order=parti_prompt_order,
                metadata_map=parti_metadata_map,
                source_tag="PartiPrompts",
                allowed_indices=None,
            )

if __name__ == "__main__":
    main()
