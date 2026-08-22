# mold

[![CI](https://github.com/utensils/mold/actions/workflows/ci.yml/badge.svg)](https://github.com/utensils/mold/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/utensils/mold/graph/badge.svg)](https://codecov.io/gh/utensils/mold)
[![FlakeHub](https://img.shields.io/endpoint?url=https://flakehub.com/f/utensils/mold/badge)](https://flakehub.com/flake/utensils/mold)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Nix Flake](https://img.shields.io/badge/nix-flake-blue.svg)](https://nixos.wiki/wiki/Flakes)
[![CLI native](https://img.shields.io/badge/CLI-native-7c3aed.svg)](https://utensils.io/mold/guide/cli-reference)
[![Agent ready](https://img.shields.io/badge/agents-ready-0891b2.svg)](https://utensils.io/mold/guide/openclaw)
[![REST + SSE](https://img.shields.io/badge/API-REST_%2B_SSE-16a34a.svg)](https://utensils.io/mold/api/)

Local AI image and video generation on your own GPU — NVIDIA CUDA and Apple
Silicon Metal, no Python, no cloud account, no usage fees. CLI-native and
pipe-friendly, with a native desktop app, web studio, TUI, iPhone companion,
Discord bot, and REST/SSE API built on the same engine.

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

The installer picks the right prebuilt binary for your GPU and verifies its
checksum. Use `mold update` to stay on stable or `mold update --nightly` to
install the newest nightly. Nix (`nix run github:utensils/mold`), Arch
(`paru -S mold-ai-bin`), and source builds are covered in the
[installation guide](https://utensils.io/mold/guide/installation);
binaries and checksums are on the
[releases page](https://github.com/utensils/mold/releases/latest).
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

# Launch the web studio and API
mold serve
```

Models download automatically on first use. Generated media is saved locally
with prompt, model, seed, and generation metadata.

## What it supports

- **Models**: FLUX.1, Flux.2 Klein/Dev, SD 1.5, SDXL, SD 3.5, Z-Image,
  Qwen-Image, Qwen-Image-Edit, Wuerstchen v2, LTX Video, LTX-2 / LTX-2.3,
  Wan 2.1/2.2, and MiniMax H3 — see the
  [model catalog](https://utensils.io/mold/models/) for sizes, VRAM needs, and
  settings
- **Images**: text-to-image, img2img, multimodal editing, inpainting,
  ControlNet, LoRA, prompt expansion, and Real-ESRGAN upscaling
- **Face identity (PuLID-FLUX)**: keep one person's face across arbitrary
  prompts with `--id-image`, on `flux-dev:q4` and `flux-dev:q8` — pure Rust
  SCRFD, ArcFace, EVA02-CLIP, and IDFormer feeding twenty cross-attention
  modules inside the FLUX transformer
- **Video and audio**: text/image-to-video, multi-prompt sequences, clip
  continuation (`--extend`), lip dub (`--pipeline lip-dub`), text-to-audio
  (`--pipeline t2a`), native MP4 with generated audio, and LTX-2 output up to
  4K via [tiled composition](https://utensils.io/mold/models/ltx2#resolution)
- **Fits your hardware**: quantized variants, encoder fallback, smart VRAM
  placement, block offloading, and spatial tiling (`--spatial-tile`)
- **Multi-machine**: connect LAN/Tailscale hosts and RunPod, route jobs by
  capability, and browse every machine's gallery in one place
- **Library organization**: title (`--title`), favorite, tag, and collect
  prints — or file them at creation with `--tag` / `--collection` so they
  arrive organized — with a per-host trash and configurable retention
  (`gallery.trash_retention_days`, `mold trash`) instead of permanent
  delete — merged across machines in the web and desktop Library
  (Prints | Collections | Trash)

MiniMax H3 weights use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE),
not Mold's MIT license. H3 may be used through Mold in every territory and
workflow — local, remote, shared, hosted, output distribution, and
redistribution — with no separate acceptance step; review the linked terms for
your use. The reviewed FL2VA Turbo distillations are ordinary model tags
(`minimax-h3-fl2va:comfy-pruned-int8-turbo-8step` and
`…-turbo-4step-768p`) that pull the same compact stack plus one pinned LoRA
adapter and render at their tier's fixed step count. Current capability limits
(FL2VA on SM89 CUDA only) are documented in the
[H3 model guide](https://utensils.io/mold/models/minimax-h3).

## Mold Studio

One native desktop app for macOS and Linux with five workspaces — Create,
Library, Models, Machines, and Settings — spanning local and remote generation,
a merged multi-machine gallery, model discovery from Hugging Face and Civitai,
GPU telemetry, and QR pairing for the iPhone companion.

**[Download Mold for macOS (Apple Silicon)](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)**
· [Explore the desktop app](https://utensils.io/mold/guide/desktop)

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

Run the engine where the GPU lives and point any client at it:

```bash
mold serve                                      # GPU machine
MOLD_HOST=http://gpu-server:7680 mold run "a cat"  # laptop
```

See the [remote workflow](https://utensils.io/mold/guide/remote-workflows) and
[RunPod](https://utensils.io/mold/deployment/runpod-cli) guides.

Install Mold's embedded Agent Skill for your coding agent:

```bash
mold skill install --detected
```

`mold skill list` shows all supported agents and paths; explicit targets such
as `mold skill install claude codex` and project installs with `--project` are
also supported.

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

Licensed under the [MIT License](LICENSE).

**Third-party code.** The LTX-Video transformer, 3D causal VAE, and flow-match
scheduler (`crates/mold-candle/src/ltx_video/`), and the LTX-2 video
transformer and VAE derived from them, were ported from
[candle-video](https://github.com/FerrisMind/candle-video) by FerrisMind
(Copyright 2025 FerrisMind), licensed under the
[Apache License 2.0](https://github.com/FerrisMind/candle-video/blob/main/LICENSE)
— itself a Rust port of Hugging Face
[diffusers](https://github.com/huggingface/diffusers). Those files remain
Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for this and
every other third-party notice.

**Face-identity weights.** Face identity additionally downloads two InsightFace
**pretrained models** (`scrfd_10g_bnkps`, `glintr100`), which are licensed for
**non-commercial research purposes only** — the InsightFace _code_ is MIT, the
_weights_ are not. Mold ships neither and refuses to download them until you
record acceptance with `mold pull pulid-flux --accept-license insightface-antelopev2`;
`mold licenses` lists what has been accepted. The PuLID adapter is Apache-2.0 and
the EVA02-CLIP tower is MIT.
