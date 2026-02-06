# QRM: Quality-Aware Modulation for Diffusion Transformers

<p align="center">
  <img src="grid_3x2.png" width="900">
</p>

**QRM** is a lightweight, reward-guided module that improves semantic alignment and visual quality of diffusion transformers by injecting quality-aware modulation signals during the denoising process, while keeping the backbone model fully frozen.

---

## Abstract

Modern text-to-image diffusions models, such as diffusion transformers (DiT), rely on timestep or prompt embeddings to modulate the strength of the denoising process in each timestep. While this modulation communicates the current noise level, it does not provide any quality-aware information, which can lead to generated images that are unaligned, visually inconsistent, and lacking in fidelity. In this paper, we propose the Quality Representation Module (QRM), a lightweight transformer module that learns a quality-aware representation based on existing model inputs, and produces a set of vectors $M_{qrm}$. These vectors adjust the adaptive LayerNorm modulation within the DiT transformer blocks, thereby injecting a quality-sensitive signal into the denoising parameters. The QRM introduces no significant changes to the sampling schedule or diffusion backbone. Experiments include ablations on QRM training losses and architectures, as well as empirical results demonstrating consistent image quality improvements over baseline DiT's for its primary evaluation metric.

---


## Installation

```bash
git clone https://github.com/LukeBudny/qrm-diffusion.git
cd qrm-diffusion
pip install -r requirements.txt

