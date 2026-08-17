# Project operating rules

- Treat 28 GiB as the hard per-process CUDA allocator budget on the 32 GB GPU.
- Apply the memory policy before loading any CUDA model or checkpoint.
- New diffusion models must be added through `qrm_diffusion/backends/` and a TOML file in `configs/models/`.
- QRM support is currently specific to the native SD3.5 backend; do not claim QRM compatibility for another architecture without implementing and testing its modulation hooks.
- Keep model weights, datasets, environments, caches, generated images, logs, and experiment outputs out of Git.
- Do not move or delete existing research artifacts unless the user explicitly requests it.
- Prefer configuration over new hard-coded model paths or CUDA device strings.
