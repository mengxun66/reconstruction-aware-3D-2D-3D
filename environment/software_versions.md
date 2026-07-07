# Software Versions

This file records the software environments used for LoRA training,
image generation, geometric reconstruction, and quantitative analysis.

## 1. LoRA training

- Base model: Stable Diffusion v1.5
- Training framework: sd-scripts/train_network.py with networks.lora
- xformers: Enabled
- GPU: NVIDIA RTX 3080Ti
- Mixed precision: fp16

## 2. D3 and D3-ext generation

- Interface: AUTOMATIC1111 Stable Diffusion WebUI v1.7.0
- Base checkpoint: v1-5-pruned-emaonly
- Model hash: 6ce0161689
- Sampler: Euler a
- Schedule type: Automatic
- Steps: 20
- CFG scale: 7
- CLIP skip: 2
- ENSD: 31337
- Resolution: 512 × 512
- LoRA multiplier: 1.0
- Low-rise checkpoint: Epoch 16
- High-rise checkpoint: Epoch 20

## 3. RGB encoding and inverse reconstruction

- Rhinoceros: Rhino 7
- Grasshopper: bundled with Rhino 7

## 4. Notes

PyCharm was used only as a code editor and is not required to reproduce
the analyses.
