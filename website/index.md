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
  - icon:
      src: /icons/terminal.svg
    title: One Command
    details:
      'mold run "a cat" — that''s it. Auto-downloads the model, generates an
      image, saves to disk.'
  - icon:
      src: /icons/grid.svg
    title: 8 Model Families
    details:
      FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image, and
      Wuerstchen v2. Quantized variants fit any GPU.
  - icon:
      src: /icons/rust.svg
    title: Pure Rust — Linux & macOS
    details:
      Single binary built on candle. CUDA on Linux (NVIDIA), Metal on macOS
      (Apple Silicon). No Python, no libtorch, no ONNX.
  - icon:
      src: /icons/server.svg
    title: Client-Server
    details:
      Run mold serve on a GPU host, generate from anywhere. REST API with SSE
      streaming for real-time progress.
  - icon:
      src: /icons/discord.svg
    title: Discord Bot
    details:
      Built-in Discord bot with /generate, /expand, /models, and /status slash
      commands. Run standalone or embedded in the server.
  - icon:
      src: /icons/layers.svg
    title: img2img & ControlNet
    details:
      Transform existing images, inpaint regions with masks, guide generation
      with ControlNet conditioning. LoRA adapters for FLUX.
  - icon:
      src: /icons/runpod.svg
    title: Deploy Anywhere
    details:
      Docker images for RunPod and any NVIDIA host. Nix flake, systemd service,
      NixOS module included.
  - icon:
      src: /icons/cloud.svg
    title: Prompt Expansion
    details:
      Local LLM expands short prompts into detailed descriptions. Auto-downloads
      Qwen3-1.7B, dropped before diffusion runs.
  - icon:
      src: /icons/openclaw.svg
    title: OpenClaw Skill
    details:
      Use mold from OpenClaw as a workspace skill while your GPU server runs
      elsewhere. Point `MOLD_HOST` at the server and generate from agent flows.
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

All images generated locally with mold — click any to see the model and prompt.

<div class="gallery-grid">
<figure>

![Snow leopard — FLUX Schnell](/gallery/flux-schnell-leopard.png)

**flux-schnell:q8** — 4 steps, seed 42 _"A majestic snow leopard perched on a
Himalayan cliff at golden hour, cinematic lighting, photorealistic"_

</figure>
<figure>

![Tea house — FLUX Dev](/gallery/flux-dev-teahouse.png)

**flux-dev:q4** — 25 steps, seed 1337 _"A cozy Japanese tea house interior with
warm lantern light, steam rising from ceramic cups, watercolor style"_

</figure>
<figure>

![Astronaut — Z-Image Turbo](/gallery/zimage-astronaut.png)

**z-image-turbo:q8** — 9 steps, seed 777 _"An astronaut floating through a
bioluminescent underwater cave, reflections on the helmet visor, science fiction
art"_

</figure>
<figure>

![Clocktower — SD 3.5](/gallery/sd35-clocktower.png)

**sd3.5-large:q8** — 28 steps, seed 2024 _"A steampunk clocktower in a Victorian
city at sunset, gears and cogs visible through glass walls, dramatic clouds"_

</figure>
<figure>

![Street market — SDXL Turbo](/gallery/sdxl-turbo-market.png)

**sdxl-turbo:fp16** — 4 steps, seed 88 _"A vibrant street food market in Bangkok
at night, neon signs, steam from woks, bustling crowd"_

</figure>
<figure>

![Fantasy castle — DreamShaper v8](/gallery/sd15-castle.png)

**dreamshaper-v8:fp16** (SD 1.5) — 25 steps, seed 555 _"A fantasy castle perched
on floating islands above clouds, magical waterfalls, ethereal glow"_

</figure>
</div>
