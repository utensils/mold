# Getting Started

mold is a single-binary CLI for generating AI images on your own GPU. No cloud
APIs, no Python environment, no heavyweight dependencies.

## How It Works

```
mold run "a cat"
  │
  ├─ Server running? → send request over HTTP
  │
  └─ No server? → load model locally on GPU
       ├─ Encode prompt (T5/CLIP text encoders)
       ├─ Denoise latent (transformer/UNet)
       ├─ Decode pixels (VAE)
       └─ Save PNG
```

mold tries to connect to a running `mold serve` instance first. If no server is
available, it falls back to local GPU inference — auto-downloading the model if
needed.

![Tea house — generated with FLUX Dev Q4](/gallery/flux-dev-teahouse.png)

## Quick Start

```bash
# Install (auto-detects your GPU)
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh

# Generate your first image
mold run "a sunset over mountains"

# That's it — the model downloads on first run (~12GB for flux-schnell:q8)
```

## What You Get

- **10 model families** — FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein,
  Qwen-Image, Qwen-Image-Edit, Wuerstchen v2, LTX Video
- **txt2img, img2img, multimodal edit, inpainting, ControlNet** — all in one binary
- **Image upscaling** — Real-ESRGAN super-resolution (2x/4x) via CLI, server API, or TUI
- **Pipe-friendly** — `mold run "a cat" | viu -` just works
- **Client-server** — run the GPU part on one machine, generate from anywhere
- **Prompt expansion** — short prompts become detailed via local LLM
- **LoRA adapters** — apply fine-tuned styles to FLUX models
- **PNG metadata** — generation parameters embedded for reproducibility

## Requirements

- **NVIDIA GPU** with CUDA or **Apple Silicon** with Metal
- Models auto-download on first use (~2–30 GB depending on model)

## Next Steps

- [Installation](/guide/installation) — all the ways to install mold
- [Configuration](/guide/configuration) — environment variables, config file
- [Generating Images](/guide/generating) — full usage guide
- [Feature Support](/guide/feature-matrix) — which model families support which
  features
- [Remote Workflows](/guide/remote-workflows) — laptop-to-GPU-server setups
- [Performance](/guide/performance) — speed and VRAM tuning
- [Custom Models & LoRA](/guide/custom-models) — manual config and adapter
  workflows
- [Models](/models/) — which model to pick for your use case
