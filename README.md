# QRM: Quality-Aware Modulation for Diffusion Transformers

<p align="center">
  <img src="grid_3x2.png" width="900">
</p>

**QRM** is a lightweight, reward-guided module that improves semantic alignment and visual quality of diffusion transformers by injecting quality-aware modulation signals during the denoising process, while keeping the backbone model fully frozen.

---

## Abstract

Modern text-to-image diffusions models, such as diffusion transformers (DiT), rely on timestep or prompt embeddings to modulate the strength of the denoising process in each timestep. While this modulation communicates the current noise level, it does not provide any quality-aware information, which can lead to generated images that are unaligned, visually inconsistent, and lacking in fidelity. In this paper, we propose the Quality Representation Module (QRM), a lightweight transformer module that learns a quality-aware representation based on existing model inputs, and produces a set of vectors $M_{qrm}$. These vectors adjust the adaptive LayerNorm modulation within the DiT transformer blocks, thereby injecting a quality-sensitive signal into the denoising parameters. The QRM introduces no significant changes to the sampling schedule or diffusion backbone. Experiments include ablations on QRM training losses and architectures, as well as empirical results demonstrating consistent image quality improvements over baseline DiT's for its primary evaluation metric.

---


## Runtime architecture

The tracked runtime is separated from local datasets, checkpoints, generated images, and experiment archives. Model-specific code lives behind a small backend interface:

- `sd35_native` uses the existing custom SD3.5 implementation and supports the current QRM modulation hooks.
- `diffusers` loads other Hugging Face Diffusers text-to-image pipelines for baseline inference.

QRM is **not automatically architecture-independent**. A new backbone can run through the Diffusers backend, but it needs a dedicated adapter and tested modulation injection points before it can use QRM.

```text
configs/models/           Model, generation, QRM, and memory settings
configs/agents/           Fixed-budget controller settings and training gates
qrm_diffusion/backends/   Model backend adapters
qrm_diffusion/agents/     Receding-horizon controller state and timestep policy
qrm_diffusion/samplers/   Adaptive Euler and unequal-step DPM++ 2M solvers
qrm/                      Existing QRM models and training code
scripts/                  Runtime parity validators
tests/                    CPU-focused configuration, controller, and parity tests
```

## GPU memory policy

All new model configurations use a **28 GiB PyTorch CUDA allocator limit** on device 0. The fraction is calculated from the memory PyTorch reports at runtime. The local RTX 5090 currently reports about 31.84 GiB, producing a fraction of approximately `0.8794` and leaving about 3.84 GiB of headroom.

The policy is applied before model loading. It constrains PyTorch allocator usage and causes an out-of-memory error instead of allowing PyTorch to grow beyond the configured budget. CUDA context memory and allocations made outside PyTorch are not included in PyTorch's allocator accounting, so peak allocated/reserved memory is also printed after each image.

Do not raise the limit above 28 GiB without explicitly changing the project requirement.

## Installation

Use Python 3.11 or newer. In an activated virtual environment:

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

For the native SD3.5 backend, place these files in `models/`:

- `clip_g.safetensors`
- `clip_l.safetensors`
- `sd3.5_medium.safetensors`
- `t5xxl.safetensors`

Model weights and local environments are intentionally ignored by Git.

## Inference

Validate a configuration without importing PyTorch or loading CUDA:

```bash
python -m qrm_diffusion --config configs/models/sd35-medium.toml --dry-run
```

Run native SD3.5 baseline inference:

```bash
python -m qrm_diffusion \
  --config configs/models/sd35-medium.toml \
  --prompt "A studio photograph of a red fox."
```

To enable QRM, copy `configs/models/sd35-medium.toml`, set `[qrm].enabled = true`, and point `[qrm].checkpoint` to a compatible checkpoint.

### Fixed-budget adaptive sampling

The native SD3.5 backend exposes two closed-loop sampler names:

- `adaptive_euler`
- `adaptive_dpmpp_2m`

Set `[generation.extra].sampler` to either name. The initial timestep policy is
exactly zero, so it follows the supplied fixed schedule with the same NFE
budget. After each CFG-guided predicted-clean latent, the controller chooses
one next sigma, enforces monotonic and bounded solver-time steps, and records a
trajectory containing sigma, action, step size, latent/denoised norms, optional
quality values, and QRM modulation norms. QRM activation is expressed as a
sigma region in the adaptive path, not a mutable loop index.

`configs/agents/sd35-qrm-timestep.toml` records the initial controller bounds
and the policy, critic, reward, training, and evaluation-gate settings. Schedule
parity has been revalidated, so critic and timestep-policy training are enabled;
joint schedule/QRM control remains disabled until the equal-NFE evaluation gate
passes.

To use a trained deterministic policy for native inference, enable
`[controller]` in the SD3.5 model TOML and set its checkpoint. A missing
checkpoint constructs the exact zero-initialized policy. Controller checkpoints
contain versioned policy/critic states and optional optimizer metadata.

Train only the timestep actor and quality critic while SD3.5 and QRM remain
frozen:

```bash
python scripts/train_timestep_policy.py \
  --prompts-file configs/prompts/controller-smoke.txt \
  --checkpoint outputs/controller-checkpoints/latest.pt
```

Compare a learned schedule against its matching fixed solver at exactly the
same NFE budget:

```bash
python scripts/compare_timestep_policy.py \
  --checkpoint outputs/controller-checkpoints/latest.pt \
  --prompts-file configs/prompts/controller-smoke.txt
```

The comparison writes `evaluation.json` and exits successfully only when the
configured minimum prompt count, mean CLIP-reward improvement, and positive
prompt fraction all pass. Passing this gate is necessary before implementing or
enabling joint schedule/QRM actions.

Run a Diffusers model:

```bash
python -m qrm_diffusion \
  --config configs/models/sdxl-base.toml \
  --prompt "A studio photograph of a red fox."
```

The example Diffusers configuration enables model CPU offload, attention slicing, and VAE slicing. Models are downloaded through the normal Hugging Face cache unless `local_files_only = true` is added to `[model]`.

List installed backend adapters:

```bash
python -m qrm_diffusion --list-backends
```

The old `inference.py` entry point remains available for reproducing the original SD3.5 workflow, but new model integrations should use `qrm_diffusion` and a TOML configuration.

## Adding another diffusion model

For a standard Diffusers pipeline, copy `configs/models/sdxl-base.toml` and change `name`, `model_id`, dtype, and generation settings. For a custom architecture, add a backend under `qrm_diffusion/backends/`, register it with `@register_backend(...)`, and add a corresponding TOML file.

Before claiming QRM support for a new architecture, implement and test how QRM outputs map into that model's transformer blocks. Baseline generation support alone is not QRM support.

## Validation

Run the CPU-focused suite:

```bash
python -m pytest tests -q
```

Validate native zero-action parity on the local SD3.5 model:

```bash
python scripts/validate_adaptive_sampler_parity.py --width 512 --height 512 --steps 8
```
