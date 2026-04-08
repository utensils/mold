---
layout: home

hero:
  name: mold
  text: AI Image Generation on Your GPU
  tagline:
    'CUDA on Linux. Metal on macOS. FLUX, SD3.5, SDXL, SD 1.5, Z-Image, Flux.2
    Klein, LTX Video & more — no cloud, no Python, no fuss.'
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
    title: 9 Model Families
    details: FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image,
      Wuerstchen v2, and LTX Video. Images and video. Quantized variants fit any GPU.
  - icon:
      src: /icons/rust.svg
    title: Linux NVIDIA + macOS Metal
    details: Single binary built on candle. NVIDIA GPUs on Linux via CUDA, Apple
      Silicon on macOS via Metal. No Python, no libtorch, no ONNX.
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

![Winter cabin — Qwen-Image 2512](/gallery/qwen-image-cabin.png)

**qwen-image-2512:q4** — 50 steps, seed 888 _"A snowy mountain cabin at twilight,
warm orange light pouring from the windows, aurora borealis in the sky above, tall
pine trees covered in snow, peaceful winter scene"_

</figure>
<figure>

![Overgrown greenhouse — Qwen-Image 2512](/gallery/qwen-image-greenhouse.png)

**qwen-image-2512:q4** — 50 steps, seed 2024 _"An abandoned greenhouse
overgrown with exotic flowers and vines, cracked glass roof letting in shafts
of golden light, butterflies and hummingbirds, lush and magical"_

</figure>
<figure>

![Bottle ship — Flux.2 Klein-9B Q4](/gallery/flux2-klein-9b-bottle-ship.png)

**flux2-klein-9b:q4** — 4 steps, seed 999 _"A glass bottle ship inside a stormy
ocean wave, dramatic lightning, hyperrealistic macro photography"_

</figure>
<figure>

![Owl — Flux.2 Klein BF16](/gallery/flux2-klein-owl.png)

**flux2-klein:bf16** — 4 steps _"a majestic owl perched on a mossy branch in a
moonlit forest"_

</figure>
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
<figure>

![Lighthouse — Wuerstchen v2](/gallery/wuerstchen-lighthouse.png)

**wuerstchen-v2:fp16** — 30 steps, seed 42 _"A lighthouse on a rocky coast during
a dramatic sunset, oil painting style, vibrant orange and purple sky"_

</figure>
<figure>

![Hot air balloon — Qwen-Image 2512](/gallery/qwen-image-balloon.png)

**qwen-image-2512:q4** — 50 steps, seed 314 _"A colorful hot air balloon floating
over a misty valley at sunrise, the balloon has the word MOLD written in bold white
letters on the side, mountains in the background, dreamy atmosphere"_

</figure>
<figure>

![Northern lights — LTX Video](/gallery/ltx-aurora.webp)

**ltx-video-0.9.6-distilled:bf16** — 8 steps, 33 frames, seed 1234 _"Northern
lights dancing over a frozen lake in Iceland, green and purple aurora ribbons
reflected in the ice, stars visible, time-lapse photography"_

</figure>
<figure>

![Jellyfish — LTX Video 0.9.8](/gallery/ltx-jellyfish.webp)

**ltx-video-0.9.6-distilled:bf16** — 8 steps, 33 frames, seed 707 _"Underwater
footage of a jellyfish pulsing through deep blue water, bioluminescent glow,
particles floating, ethereal slow motion"_

</figure>
</div>
