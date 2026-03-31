---
layout: home

hero:
  name: mold
  text: AI Image Generation on Your GPU
  tagline:
    'FLUX, SD3.5, SDXL, SD 1.5, Z-Image, Flux.2 Klein & more — no cloud, no
    Python, no fuss.'
  image:
    src: /logo-transparent.png
    alt: mold logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View Models
      link: /models/
    - theme: alt
      text: GitHub
      link: https://github.com/utensils/mold

features:
  - icon: ⚡
    title: One Command
    details:
      'mold run "a cat" — that''s it. Auto-downloads the model, generates an
      image, saves to disk.'
  - icon: 🎯
    title: 8 Model Families
    details:
      FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image, and
      Wuerstchen v2. Quantized variants fit any GPU.
  - icon: 🔧
    title: Pure Rust
    details:
      Single binary built on candle. No Python runtime, no libtorch, no ONNX.
      CUDA on Linux, Metal on macOS.
  - icon: 🌐
    title: Client-Server
    details:
      Run mold serve on a GPU host, generate from anywhere. REST API, SSE
      streaming, Discord bot included.
  - icon: 🖼️
    title: img2img & ControlNet
    details:
      Transform existing images, inpaint regions with masks, guide generation
      with ControlNet conditioning.
  - icon: 📦
    title: Deploy Anywhere
    details:
      Docker images for RunPod and any NVIDIA host. Nix flake, systemd service,
      NixOS module included.
---

## Quick Example

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh

# Generate
mold run "a cat riding a motorcycle through neon-lit streets"

# Pick a model
mold run flux-dev:q4 "a sunset over mountains"

# Pipe to an image viewer
mold run "neon cityscape" | viu -
```

## Gallery

<div class="gallery-grid">

![Snow leopard — FLUX Schnell](/gallery/flux-schnell-leopard.png)

![Tea house — FLUX Dev](/gallery/flux-dev-teahouse.png)

![Astronaut — Z-Image Turbo](/gallery/zimage-astronaut.png)

![Clocktower — SD 3.5](/gallery/sd35-clocktower.png)

![Street market — SDXL Turbo](/gallery/sdxl-turbo-market.png)

![Fantasy castle — DreamShaper v8](/gallery/sd15-castle.png)

</div>
