# mold

[![CI](https://github.com/utensils/mold/actions/workflows/ci.yml/badge.svg)](https://github.com/utensils/mold/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/utensils/mold/graph/badge.svg)](https://codecov.io/gh/utensils/mold)
[![FlakeHub](https://img.shields.io/endpoint?url=https://flakehub.com/f/utensils/mold/badge)](https://flakehub.com/flake/utensils/mold)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Nix Flake](https://img.shields.io/badge/nix-flake-blue.svg)](https://nixos.wiki/wiki/Flakes)
[![CLI native](https://img.shields.io/badge/CLI-native-7c3aed.svg)](https://utensils.io/mold/guide/cli-reference)
[![Agent ready](https://img.shields.io/badge/agents-ready-0891b2.svg)](https://utensils.io/mold/guide/openclaw)
[![REST + SSE](https://img.shields.io/badge/API-REST_%2B_SSE-16a34a.svg)](https://utensils.io/mold/api/)

Local AI image and video generation on your own GPU. Mold supports NVIDIA CUDA
and Apple Silicon Metal, with a CLI, native desktop app, web studio, TUI, mobile
companions, Discord bot, and REST/SSE API built on the same engine.

**[Documentation](https://utensils.io/mold/)** ·
**[Models](https://utensils.io/mold/models/)** ·
**[Desktop guide](https://utensils.io/mold/guide/desktop)** ·
**[API](https://utensils.io/mold/api/)**

![Mold Studio desktop app generating an owl](website/public/screenshots/mold-studio-desktop.png)

## Install

Stable release:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

Nightly CLI from the latest published `main` build:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | MOLD_CHANNEL=nightly sh
```

The installer selects a compatible build and verifies its checksum. See the
[installation guide](https://utensils.io/mold/guide/installation) for Nix,
Arch, Windows, Android, and source builds.
GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.

## Quick start

```bash
# Generate with the default model
mold run "a cat riding a motorcycle through neon-lit streets"

# Choose a model and reproducible seed
mold run flux-dev:q4 "a sunset over mountains" --seed 42

# Edit an image
mold run qwen-image-edit-2511:q4 "make the chair red" --image chair.png

# Generate video
mold run ltx-video-0.9.6-distilled:bf16 "a fox in the snow" --frames 25

# Turn a photo into a 3D mesh
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb

# Upscale a Library video as a durable framewise job
mold video-upscale create clip.mp4 --wait

# Launch the web studio and API
mold serve
```

Models download automatically on first use. Generated media is saved locally
with prompt, model, seed, and generation metadata.
Framewise video upscale also needs the host codec bridge: Nix packages and
CUDA containers include it, while raw binary installs must provide `ffmpeg`
and `ffprobe` on `PATH` before the server advertises that feature.

## What it supports

- **Models:** FLUX.1, Flux.2, Stable Diffusion, Z-Image, Qwen-Image,
  Wuerstchen, LTX Video, Wan, MiniMax H3, and Hunyuan3D. See the
  [model catalog](https://utensils.io/mold/models/) for variants and hardware
  requirements.
- **Images:** text-to-image, image editing, inpainting, ControlNet, LoRA,
  identity photos, prompt expansion, and upscaling.
- **Video and audio:** text/image-to-video, sequences, clip continuation,
  lip dub, text-to-audio, and MP4 output with generated audio.
- **3D:** image-to-mesh with Hunyuan3D, published to the Library as binary
  glTF with a rendered poster tile.
- **Multiple machines:** connect local, LAN, Tailscale, and RunPod hosts, then
  route work and browse one combined Library.
- **Organization:** title, favorite, tag, collect, restore, and manage prints
  across the desktop and web apps.

Model weights keep their own licenses. See each model page for terms and
current platform support.

## Mold Studio

The desktop app brings Create, Library, Models, Machines, and Settings into one
workspace for local and remote generation. It also pairs with the iPhone and
Android companions.

**[Download Mold for macOS (Apple Silicon)](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)**
· [Explore the desktop app](https://utensils.io/mold/guide/desktop)

**[Download Mold for Windows (x86_64)](https://github.com/utensils/mold/releases/latest/download/Mold-windows-x64-self-signed.exe)**
— a self-signed NSIS installer. The published build is CPU / remote-hosts
only; see the [desktop guide](https://utensils.io/mold/guide/desktop) for the
CUDA recipe. Verify and explicitly trust the release's
`mold-windows-self-signing.cert.cer` before installing; the certificate is not
publicly trusted and does not suppress SmartScreen on its own.

Linux desktop builds are source/CI distributions for now — `nix build
.#mold-desktop` or the devshell's `desktop-build` CUDA AppImage. See the
[desktop guide](https://utensils.io/mold/guide/desktop).

Android uses the same remote-only Mold Studio mobile interface. Download the
signed universal nightly APK directly; there is no zip to unpack:

**[Download nightly Android APK](https://github.com/utensils/mold/releases/download/latest/Mold-android.apk)**
· [Android installation guide](https://utensils.io/mold/guide/android)

## More ways to create

Preview generations directly in supported terminals:

```bash
mold run "a cat" --preview
```

<p align="center">
  <img src="docs/terminal-preview-example.png" alt="Generating the Mold logo with an inline terminal preview" width="720" />
  <br/>
  <em>Inline image generation in Ghostty with <code>--preview</code></em>
</p>

Or open the keyboard-first terminal interface with `mold tui`:

<p align="center">
  <img src="website/public/gallery/tui-generate.png" alt="Mold TUI Create workspace with image preview" width="720" />
  <br/>
  <em>The TUI Create workspace with a native terminal image preview</em>
</p>

Run the engine where the GPU lives and connect from another machine:

```bash
mold serve                                      # GPU machine
MOLD_HOST=http://gpu-server:7680 mold run "a cat"  # laptop
```

See the [remote workflow](https://utensils.io/mold/guide/remote-workflows) and
[RunPod](https://utensils.io/mold/deployment/runpod-cli) guides. Use
`mold queue` to manage remote work and `mold library` to browse and organize
the host's prints. To install Mold's Agent Skill for supported coding agents,
run:

```bash
mold skill install --detected
```

The installed bundle uses each agent's native metadata and discovery contract,
with a concise router, safety guidance, tested examples, a shared prompting
guide, one base guide per manifest family, and task leaves only for the distinct
H3, Wan, and LTX-2 grammars that need them. Hunyuan3D's base guide is the one
that tells an agent NOT to write a prompt.

## Project

Mold is a Rust workspace built on
[candle](https://github.com/huggingface/candle). The documentation covers the
[CLI](https://utensils.io/mold/guide/cli-reference),
[configuration](https://utensils.io/mold/guide/configuration),
[deployment](https://utensils.io/mold/deployment/), and
[HTTP API](https://utensils.io/mold/api/).

Core contributors:
[James Brink](https://jamesbrink.online/) and
[Jeffrey Dilley](mailto:jeff.dilley@gmail.com).

Licensed under the [MIT License](LICENSE). Third-party components and model
licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[model documentation](https://utensils.io/mold/models/). InsightFace identity
weights require separate acceptance and are limited to non-commercial research
use.
