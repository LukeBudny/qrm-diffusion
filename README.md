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

Please download the following SD3.5 model files from the SD#.5 github and move to the models folder:
clip_g.safetensors
clip_l.safetensors
sd3.5_medium.safetensors
t5xxl.safetensors

```bash
git clone https://github.com/LukeBudny/qrm-diffusion.git
cd qrm-diffusion

sudo apt update
sudo apt install -y python3.13 python3.13-venv git-lfs

python3.13 -m venv qrm-env
source qrm-env/bin/activate
pip install -U pip setuptools wheel

# Install PyTorch CUDA 12.8 wheels (cu128)
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Install packages
pip install -r requirements.txt

#combine qrm checkpoint files
cat qrm_epoch25.pth.part* > qrm_epoch25.pth

## Inference

```bash
python inference.py --mode qrm
#or for baseline sd3.5
python inference.py