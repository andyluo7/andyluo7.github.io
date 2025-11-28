---
layout: post
title: "Running FLUX.2, HunyuanVideo-1.5, and Z-Image-Turbo on AMD MI300X"
date: 2025-11-27 12:00:00 -0500
categories: ai
---

I spent some time bringing a few popular open models to an AMD MI300X box and wanted to jot down a repeatable path. The focus here is on making the hardware happy (ROCm), keeping dependencies containerized, and getting first frames/images out quickly.

## 1) Base setup

- OS: recent Ubuntu (22.04 or similar) with kernel that ships ROCm 6.x drivers.
- GPU runtime: ROCm 6.1+ with `rocminfo` and `rocm-smi` working.
- Docker: `24.x`+ with rootless or root; add the ROCm device nodes and permissions to your daemon defaults if needed.

Quick sanity:

```bash
rocminfo | head
rocm-smi --showproductname --showvbios
```

## 2) Container image

Use an image that already bundles ROCm and PyTorch for AMD:

```bash
docker pull rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.3.1
```

Run with GPU access (example for a single MI300X):

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $PWD:/workspace -w /workspace \
  rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.3.1 \
  /bin/bash
```

Inside the container:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

You should see ROCm-backed PyTorch and the MI300X device name.

## 3) Common deps

```bash
pip install --upgrade pip
pip install safetensors accelerate transformers==4.45.0 diffusers==0.31.0 einops sentencepiece
```

Adjust versions if a specific repo pins them.

## 4) FLUX.2 (image)

Minimal script (assumes HF auth token in `HF_TOKEN` if the model is gated):

```bash
python - <<'PY'
import torch
from diffusers import FluxPipeline

device = "cuda"  # ROCm is presented as CUDA in PyTorch
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe = pipe.to(device)

prompt = "A cinematic photo of a coastline at sunset, shot on 50mm"
image = pipe(prompt, num_inference_steps=30, guidance_scale=3.5).images[0]
image.save("flux2.png")
print("Saved flux2.png")
PY
```

Tips: use `torch_dtype=torch.bfloat16` for memory headroom; drop steps to 20 if you see OOM.

## 5) HunyuanVideo-1.5 (video)

Tencent’s repo expects recent diffusers + a video VAE. A slim testing script:

```bash
pip install imageio-ffmpeg

python - <<'PY'
import torch
from diffusers import HunyuanVideoPipeline

pipe = HunyuanVideoPipeline.from_pretrained(
    "TencentARC/HunyuanVideo-1.5",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

prompt = "A slow pan across a futuristic city skyline at dawn"
result = pipe(prompt, num_frames=48, num_inference_steps=25, guidance_scale=5.5)
video = result.frames[0]  # list of PIL images

import imageio
imageio.mimwrite("hunyuan.mp4", video, fps=12, quality=8)
print("Saved hunyuan.mp4")
PY
```

Notes: MI300X has plenty of HBM, but if you’re tight, reduce `num_frames` or steps. ROCm presents as CUDA so the `.to("cuda")` call is correct.

## 6) Z-Image-Turbo (image)

This model emphasizes speed. Example:

```bash
python - <<'PY'
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "Z-Research/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

prompt = "A watercolor painting of mountains with mist"
img = pipe(prompt, num_inference_steps=12, guidance_scale=4.0).images[0]
img.save("z-image-turbo.png")
print("Saved z-image-turbo.png")
PY
```

Lower steps keep latency small; bump to ~20 if you want cleaner outputs.

## 7) Performance and troubleshooting

- **Memory**: Prefer BF16; avoid FP32. For video, limit frames or resolution first.
- **Speed**: Use `torch.backends.cudnn.benchmark = True` equivalent is not needed; PyTorch with ROCm tunes kernels automatically.
- **IO**: Cache models under `/workspace/.cache/huggingface` to avoid re-downloads between runs.
- **Stability**: If you see `no kernel image` errors, ensure ROCm version matches the PyTorch ROCm build tag.

## 8) Publishing

Save outputs (`flux2.png`, `hunyuan.mp4`, `z-image-turbo.png`) somewhere visible in a post or upload short clips to a hosting service, then embed links or thumbnails in a follow-up entry. This post is a setup log; the next one can showcase results.
