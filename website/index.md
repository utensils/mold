---
layout: home

hero:
  name: mold
  text: Local AI Image & Video Generation on Your GPU
  tagline: 'Create locally with NVIDIA CUDA or Apple Silicon Metal. Use the CLI,
    desktop app, web studio, or terminal interface, and connect from iPhone or
    Android when you are away from the GPU.'
  image:
    src: /logo-transparent.png
    alt: mold logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: Download for Mac (Apple Silicon)
      link: https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg
    - theme: alt
      text: Windows Nightly
      link: https://github.com/utensils/mold/releases/download/latest/Mold-windows-x64-self-signed.exe
    - theme: alt
      text: View Models
      link: /models/
    - theme: alt
      text: GitHub
      link: https://github.com/utensils/mold

features:
  - icon:
      src: /icons/terminal.svg
    title: CLI-Native
    details:
      'mold run "a cat" -- that''s it. Predictable stdin, stdout, files, exit
      behavior, and machine-readable output make the same workflows natural for
      terminals, shell pipelines, CI, and agents.'
  - icon:
      src: /icons/grid.svg
    title: Broad Model Support
    details: FLUX.1, SDXL, SD 1.5, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image,
      Qwen-Image-Edit, Wuerstchen v2, LTX Video (0.9.x, 2, 2.3, and 2.5),
      Wan 2.1/2.2, and MiniMax H3. Create images, video, and generated audio
      with model variants suited to a wide range of GPUs.
  - icon:
      src: /icons/rust.svg
    title: Native GPU engine
    details: Single binary built on candle. NVIDIA GPUs on Linux and locally
      built x64 Windows packages use CUDA; Apple Silicon uses Metal. No Python
      or libtorch.
    link: https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg
    linkText: Download the macOS desktop app
  - icon:
      src: /icons/windows.svg
    title: Windows Desktop + CLI
    details: Self-signed x64 Desktop and CPU/remote-client CLI builds ship with
      a pinned public certificate and explicit trust instructions.
    link: https://github.com/utensils/mold/releases/download/latest/Mold-windows-x64-self-signed.exe
    linkText: Download the Windows nightly
  - icon:
      src: /icons/server.svg
    title: Client-Server
    details:
      Run mold serve on a GPU host, generate from anywhere. REST API with SSE
      streaming for real-time progress.
  - icon:
      src: /icons/grid.svg
    title: iPhone Remote Studio
    details:
      Generate, browse a merged gallery, manage models, and inspect remote hosts
      from an iPhone-first Tauri app over LAN or Tailscale.
    link: /guide/iphone
    linkText: Explore the iPhone app
  - icon:
      src: /icons/discord.svg
    title: Discord Bot
    details:
      Built-in Discord bot with /generate, durable /sequence, /expand, /models,
      and /status slash commands. Run standalone or embedded in the server.
  - icon:
      src: /icons/layers.svg
    title: img2img, Edit & ControlNet
    details:
      Transform existing images, run multimodal Qwen edit workflows, inpaint
      regions with masks, and guide generation with ControlNet conditioning.
      LoRA adapters for FLUX, Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image,
      Qwen-Image-Edit, Wan, and Z-Image.
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

## Born in the terminal, flexible everywhere

Mold began as a single-binary command-line tool and the CLI remains its stable
foundation. Every core workflow can be run by a person, composed in a shell
pipeline, automated in CI, or called by an agent. `mold serve`, REST/SSE, MCP,
and the desktop, web, and TUI surfaces all extend that same engine and contract;
the iPhone and Android apps use the same contract as remote-only clients.

## Mold Studio for desktop

Create locally on Apple Silicon or NVIDIA, or connect every Mold machine you
use. The native macOS, Linux, and Windows desktop app keeps generation, a
unified multi-host Library, model discovery, live downloads, queues, telemetry,
and RunPod in one focused workspace.

[![Mold Studio desktop app generating an owl](/screenshots/mold-studio-desktop.png)](/guide/desktop)

<div class="platform-downloads">
  <a class="platform-download" href="https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg">
    <img src="/icons/apple.svg" alt="" />
    <span><strong>macOS Desktop</strong><small>Signed and notarized · Apple Silicon</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/Mold-windows-x64-self-signed.exe">
    <img src="/icons/windows.svg" alt="" />
    <span><strong>Windows Desktop (Nightly)</strong><small>Self-signed NSIS installer · x64</small></span>
  </a>
  <a class="platform-download" href="https://github.com/utensils/mold/releases/download/latest/mold-x86_64-pc-windows-msvc-cpu.zip">
    <img src="/icons/terminal.svg" alt="" />
    <span><strong>Windows CLI (Nightly)</strong><small>Self-signed CPU / remote client · x64</small></span>
  </a>
</div>

**[Explore the desktop app](/guide/desktop)** · **[Windows CLI instructions](/guide/installation#windows-cli)**

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

For the latest rolling CLI build from `main`, install with
`curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | MOLD_CHANNEL=nightly sh`.

## Gallery

All images generated locally with mold -- click any to see the model and prompt.

<div class="gallery-grid">
<figure>

![Winter cabin -- Qwen-Image 2512](/gallery/qwen-image-cabin.png)

**qwen-image-2512:q4** -- 50 steps, seed 888 _"A snowy mountain cabin at twilight,
warm orange light pouring from the windows, aurora borealis in the sky above, tall
pine trees covered in snow, peaceful winter scene"_

</figure>
<figure>

![Overgrown greenhouse -- Qwen-Image 2512](/gallery/qwen-image-greenhouse.png)

**qwen-image-2512:q4** -- 50 steps, seed 2024 _"An abandoned greenhouse
overgrown with exotic flowers and vines, cracked glass roof letting in shafts
of golden light, butterflies and hummingbirds, lush and magical"_

</figure>
<figure>

![Bottle ship -- Flux.2 Klein-9B Q4](/gallery/flux2-klein-9b-bottle-ship.png)

**flux2-klein-9b:q4** -- 4 steps, seed 999 _"A glass bottle ship inside a stormy
ocean wave, dramatic lightning, hyperrealistic macro photography"_

</figure>
<figure>

![Owl -- Flux.2 Klein BF16](/gallery/flux2-klein-owl.png)

**flux2-klein:bf16** -- 4 steps _"a majestic owl perched on a mossy branch in a
moonlit forest"_

</figure>
<figure>

![Snow leopard -- FLUX Schnell](/gallery/flux-schnell-leopard.png)

**flux-schnell:q8** -- 4 steps, seed 42 _"A majestic snow leopard perched on a
Himalayan cliff at golden hour, cinematic lighting, photorealistic"_

</figure>
<figure>

![Tea house -- FLUX Dev](/gallery/flux-dev-teahouse.png)

**flux-dev:q4** -- 25 steps, seed 1337 _"A cozy Japanese tea house interior with
warm lantern light, steam rising from ceramic cups, watercolor style"_

</figure>
<figure>

![Astronaut -- Z-Image Turbo](/gallery/zimage-astronaut.png)

**z-image-turbo:q8** -- 9 steps, seed 777 _"An astronaut floating through a
bioluminescent underwater cave, reflections on the helmet visor, science fiction
art"_

</figure>
<figure>

![Clocktower -- SD 3.5](/gallery/sd35-clocktower.png)

**sd3.5-large:q8** -- 28 steps, seed 2024 _"A steampunk clocktower in a Victorian
city at sunset, gears and cogs visible through glass walls, dramatic clouds"_

</figure>
<figure>

![Street market -- SDXL Turbo](/gallery/sdxl-turbo-market.png)

**sdxl-turbo:fp16** -- 4 steps, seed 88 _"A vibrant street food market in Bangkok
at night, neon signs, steam from woks, bustling crowd"_

</figure>
<figure>

![Fantasy castle -- DreamShaper v8](/gallery/sd15-castle.png)

**dreamshaper-v8:fp16** (SD 1.5) -- 25 steps, seed 555 _"A fantasy castle perched
on floating islands above clouds, magical waterfalls, ethereal glow"_

</figure>
<figure>

![Lighthouse -- Wuerstchen v2](/gallery/wuerstchen-lighthouse.png)

**wuerstchen-v2:fp16** -- 30 steps, seed 42 _"A lighthouse on a rocky coast during
a dramatic sunset, oil painting style, vibrant orange and purple sky"_

</figure>
<figure>

![Hot air balloon -- Qwen-Image 2512](/gallery/qwen-image-balloon.png)

**qwen-image-2512:q4** -- 50 steps, seed 314 _"A colorful hot air balloon floating
over a misty valley at sunrise, the balloon has the word MOLD written in bold white
letters on the side, mountains in the background, dreamy atmosphere"_

</figure>
<figure>

<video autoplay muted loop playsinline aria-label="Northern lights -- LTX Video" src="/gallery/ltx-aurora.webm"></video>

**ltx-video-0.9.6-distilled:bf16** -- 8 steps, 33 frames, seed 1234 _"Northern
lights dancing over a frozen lake in Iceland, green and purple aurora ribbons
reflected in the ice, stars visible, time-lapse photography"_

</figure>
<figure>

<video autoplay muted loop playsinline aria-label="Jellyfish -- LTX Video" src="/gallery/ltx-jellyfish.webm"></video>

**ltx-video-0.9.6-distilled:bf16** -- 8 steps, 33 frames, seed 707 _"Underwater
footage of a jellyfish pulsing through deep blue water, bioluminescent glow,
particles floating, ethereal slow motion"_

</figure>
</div>
