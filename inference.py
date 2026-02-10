"""
Simple SD3.5 inference script (baseline OR QRMModulatorLatentV6 only).

Defaults are chosen so you can run with zero arguments.

Usage:
  # Baseline (defaults)
  python simple_sd35_qrm_v6_infer.py

  # QRM with default checkpoint
  python simple_sd35_qrm_v6_infer.py --mode qrm

  # QRM with custom checkpoint
  python simple_sd35_qrm_v6_infer.py --mode qrm --qrm_ckpt path/to/ckpt.pth

Prompts:
- By default it reads prompts from prompts.txt (one per line).
- If prompts.txt does not exist, it falls back to a small built-in prompt list.
"""

import argparse
import time
from pathlib import Path

import torch

from sd3_infer import SD3Inferencer
from qrm.qrm_models import QRMModulatorLatentV6


DEFAULT_SD35 = "models/sd3.5_medium.safetensors"
DEFAULT_OUT_DIR = "outputs"
DEFAULT_PROMPTS_FILE = "prompts.txt"
DEFAULT_QRM_CKPT = r"models/qrm_checkpoint/qrm_epoch25.pth"


FALLBACK_PROMPTS = [
    "A DSLR photo of a golden retriever wearing sunglasses at the beach, shallow depth of field.",
    "A cinematic still of a futuristic city at night, neon signs, rain, reflections, ultra detailed.",
    "A watercolor illustration of a cozy cabin in a snowy forest, warm window light.",
]


def parse_args():
    p = argparse.ArgumentParser()

    # model paths
    p.add_argument("--sd3_path", type=str, default=DEFAULT_SD35)
    p.add_argument("--model_folder", type=str, default="models")

    # output / prompts
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--prompts_file",
        type=str,
        default=DEFAULT_PROMPTS_FILE,
        help="One prompt per line (default: prompts.txt). If missing, uses built-in fallback prompts.",
    )

    # mode
    p.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "qrm"],
        default="baseline",
        help="baseline or qrm (default: baseline)",
    )

    # sampling / image params
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_type", type=str, choices=["fixed", "roll", "rand"], default="fixed")

    # QRM (used only in qrm mode)
    p.add_argument(
        "--qrm_ckpt",
        type=str,
        default=DEFAULT_QRM_CKPT,
        help=r"QRM checkpoint path (default: models/qrm_checkpoint/qrm_epoch25.pth). Used only in qrm mode.",
    )
    p.add_argument("--qrm_start_step", type=int, default=None, help="Override checkpoint qrm_start_step")
    p.add_argument("--qrm_end_step", type=int, default=None, help="Override checkpoint qrm_end_step")

    return p.parse_args()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This script assumes CUDA because your SD3Inferencer/sd3_infer path uses .cuda()."
        )
    torch.backends.cuda.matmul.allow_tf32 = True


def load_prompts_or_fallback(prompts_file: str) -> list[str]:
    p = Path(prompts_file)
    if not p.exists():
        print(f"[PROMPTS] {p} not found; using {len(FALLBACK_PROMPTS)} built-in fallback prompts.")
        return list(FALLBACK_PROMPTS)

    prompts: list[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)

    if not prompts:
        print(f"[PROMPTS] {p} is empty; using {len(FALLBACK_PROMPTS)} built-in fallback prompts.")
        return list(FALLBACK_PROMPTS)

    print(f"[PROMPTS] Loaded {len(prompts)} prompts from {p}.")
    return prompts


def attach_qrm_v6(
    inferencer: SD3Inferencer,
    ckpt_path: str,
    override_start: int | None,
    override_end: int | None,
) -> tuple[int, int]:
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"QRM checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_p), map_location="cuda")

    qrm_start_step = ckpt.get("qrm_start_step", 25)
    qrm_end_step = ckpt.get("qrm_end_step", 47)
    if override_start is not None:
        qrm_start_step = int(override_start)
    if override_end is not None:
        qrm_end_step = int(override_end)

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint at {ckpt_path} missing key 'model' (QRM state_dict).")

    inferencer.sd3.model.qrm = QRMModulatorLatentV6(inferencer.sd3.model._qrm_block_spans).cuda()
    inferencer.sd3.model.qrm.load_state_dict(ckpt["model"])
    inferencer.sd3.model.qrm_inference = True

    # Explicit, in case any of your codepaths references these
    inferencer.sd3.model.qrm_type = ckpt.get("qrm_type", True)
    inferencer.qrm_type = ckpt.get("qrm_type", True)

    return qrm_start_step, qrm_end_step


def main():
    args = parse_args()
    require_cuda()

    prompts = load_prompts_or_fallback(args.prompts_file)

    # Output directory (mode-separated)
    out_dir = Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # If QRM: use eval_model from ckpt if present (your load() may wire vision feature bits based on this).
    eval_model = "clip"
    if args.mode == "qrm":
        tmp = torch.load(args.qrm_ckpt, map_location="cpu")
        eval_model = tmp.get("vision_feature_model", "clip")

    inferencer = SD3Inferencer()
    inferencer.load(
        model=args.sd3_path,
        vae=None,
        shift=3.0,  # sd3.5_medium config
        controlnet_ckpt=None,
        model_folder=args.model_folder,
        text_encoder_device="cuda",
        load_tokenizers=True,  # self-contained (no cached conds)
        eval_model=eval_model,
        inference=True,
        qrm_type="QRMModulatorLatentV6",  # keep hooks available even if baseline
    )

    use_qrm = args.mode == "qrm"
    qrm_start_step, qrm_end_step = 25, 47

    if use_qrm:
        qrm_start_step, qrm_end_step = attach_qrm_v6(
            inferencer,
            args.qrm_ckpt,
            args.qrm_start_step,
            args.qrm_end_step,
        )
        print(
            f"[QRM V6] ckpt={args.qrm_ckpt} start={qrm_start_step} end={qrm_end_step} eval_model={eval_model}"
        )
    else:
        inferencer.sd3.model.qrm = None
        inferencer.sd3.model.qrm_inference = False
        print("[BASELINE] QRM disabled")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            save_name = f"{i:04d}.png"
            print(f"\n[{args.mode}] {i+1}/{len(prompts)} -> {save_name}")

            seed = args.seed
            if args.seed_type == "roll":
                seed = args.seed + i

            inferencer.gen_image(
                prompts=prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg,
                sampler="dpmpp_2m",
                seed=seed,
                seed_type=args.seed_type,
                out_dir=str(out_dir),
                save_names=save_name,
                use_qrm=use_qrm,
                qrm_start_step=qrm_start_step,
                qrm_end_step=qrm_end_step,
            )

    elapsed = time.time() - start
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"\n[CUDA] Peak allocated: {peak:.2f} GB | Peak reserved: {reserved:.2f} GB")

    print(f"[DONE] Wrote {len(prompts)} images to {out_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
