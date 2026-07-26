# mold

[![CI](https://github.com/utensils/mold/actions/workflows/ci.yml/badge.svg)](https://github.com/utensils/mold/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/utensils/mold/graph/badge.svg)](https://codecov.io/gh/utensils/mold)
[![FlakeHub](https://img.shields.io/endpoint?url=https://flakehub.com/f/utensils/mold/badge)](https://flakehub.com/flake/utensils/mold)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Nix Flake](https://img.shields.io/badge/nix-flake-blue.svg)](https://nixos.wiki/wiki/Flakes)
[![CLI native](https://img.shields.io/badge/CLI-native-7c3aed.svg)](https://utensils.io/mold/guide/cli-reference)
[![Agent ready](https://img.shields.io/badge/agents-ready-0891b2.svg)](https://utensils.io/mold/guide/openclaw)
[![REST + SSE](https://img.shields.io/badge/API-REST_%2B_SSE-16a34a.svg)](https://utensils.io/mold/api/)

Local AI image and video generation on your own GPU. Mold runs on NVIDIA CUDA
and Apple Silicon Metal with no Python, no cloud account, and no usage fees.

Mold began at the command line and remains **CLI-native**: every core workflow
is available as a composable command with predictable stdin, stdout, files, and
exit behavior. Use it interactively, script it in a pipeline, give it to an
agent, run it in CI, or build on the same engine through REST and SSE. The
desktop, web, TUI, and iPhone experiences are additional interfaces—not
replacements for the CLI foundation.

**[Documentation](https://utensils.io/mold/)** ·
**[Models](https://utensils.io/mold/models/)** ·
**[Desktop guide](https://utensils.io/mold/guide/desktop)** ·
**[API](https://utensils.io/mold/api/)**

![Mold Studio desktop app generating an owl](website/public/screenshots/mold-studio-desktop.png)

## Mold Studio

Mold Studio brings local and remote generation into one native desktop app for
macOS and Linux:

- **Create** images, edits, upscales, and multi-stage videos with
  capability-aware controls, prompt expansion, batches, LoRAs, ControlNet, and
  reproducible seeds.
- **Library** merges prints from every connected machine into a fast,
  searchable gallery with saved prompts, settings, provenance, and native media
  actions, plus a remembered thumbnail-size control on web and desktop.
- **Models** discovers and installs checkpoints from Mold, Hugging Face, and
  Civitai, with live download progress and per-machine routing.
- **Machines** connects this device, LAN/Tailscale hosts, and RunPod while
  showing GPU telemetry, queues, downloads, and installed models.
- **Settings** includes Mold and Safelight themes, Stable/Nightly updates, local
  performance controls, and secure provider credentials.

**[Download Mold for macOS (Apple Silicon)](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)**
· [Explore the desktop app](https://utensils.io/mold/guide/desktop)

Mold also ships a responsive web studio with every `mold serve`, a
keyboard-first terminal UI, a REST/SSE API, and a remote iPhone companion.
On CUDA servers, resource telemetry is joined to CUDA's startup-visible
devices by UUID. `CUDA_VISIBLE_DEVICES` remains a hard boundary even when it
reorders cards or uses GPU/MIG UUID selectors: hidden GPUs are not published,
and a MIG worker never inherits its parent GPU's full-memory sample.
GPU startup defaults to `all`. `--gpus` / `MOLD_GPUS` also accepts `none`
(maintenance mode), visible ordinals, Mold stable IDs (`cuda:<uuid>` or
`metal:default`), and NVIDIA `GPU-...` / `MIG-...` UUIDs. Prefer stable IDs in
persistent configuration because ordinals are process-local. Maintenance mode
keeps inventory, telemetry, downloads, and settings available but rejects
generation and model-load requests.
Scheduler admission uses each device's sampled free VRAM rather than total
capacity and freezes a concrete per-component execution plan before dispatch.
Explicit CPU/device placement is a hard constraint; missing devices and
cross-GPU component pins fail instead of falling back. `mold run --local
--batch N` uses the same deterministic assignment core across every GPU
selected by `--gpus`/`MOLD_GPUS`; single-item local runs retain best-free-GPU
selection.

`MOLD_DISPATCH_MODE=legacy` remains the default server dispatch owner until
the Phase A–E cutover gates pass. Set `MOLD_DISPATCH_MODE=v2` explicitly to
qualify the new scheduler. During the
rollout window, restart with `legacy` for rollback or `observe` to keep legacy
dispatch authoritative while recording the read-only V2 decision at each
legacy dispatch point. The mode is restart-only; Mold never switches CUDA
owners or contexts live. Legacy/observe retain their depth-two transport, but
transport backlog is separate from the same binary execution claim used by V2
and durable chains. Queue pause applies to generation, utility, and admin GPU
work in every mode. Phase E/F cutover explicitly changes the compiled default
back to V2 after the A–E acceptance matrix passes; Phase B does not imply that
cutover.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

The installer downloads the latest release to `~/.local/bin/mold`, verifies
its SHA-256 checksum, inspects every Linux GPU visible through
`CUDA_VISIBLE_DEVICES`, and selects one compatible NVIDIA build independent of
device order. Unsupported mixed architecture groups fail with an actionable
override/source-build error instead of silently targeting GPU 0. macOS installs
the Metal build.

<details>
<summary>Other install options</summary>

```bash
# Nix — NVIDIA Ada / RTX 40-series
nix run github:utensils/mold -- run "a cat"

# Nix — NVIDIA Ampere / RTX 3090 or A40
nix run github:utensils/mold#mold-sm86 -- run "a cat"

# Nix — NVIDIA datacenter Blackwell / B200 or GB200
nix run github:utensils/mold#mold-sm100 -- run "a cat"

# Nix — NVIDIA consumer Blackwell / RTX 50-series
nix run github:utensils/mold#mold-sm120 -- run "a cat"

# Arch Linux
paru -S mold-ai-bin

# Build from source on Linux
./scripts/ensure-web-dist.sh
cargo build --release -p mold-ai --features cuda

# Build from source on Apple Silicon
./scripts/ensure-web-dist.sh
cargo build --release -p mold-ai --features metal
```

B200/GB200 sm_100 builds and synthetic multi-device coverage are simulated,
not hardware-qualified. Real 8×B200 qualification remains deferred.
The deferred RTX 3090 acceptance runner and evidence schema live in
[`docs/qualification/`](docs/qualification/); no passing report is claimed.

Prebuilt binaries and checksums are available on the
[releases page](https://github.com/utensils/mold/releases/latest). See the
[installation guide](https://utensils.io/mold/guide/installation) for platform
details.

</details>

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

- FLUX.1, Flux.2 Klein, SD 1.5, SDXL, SD 3.5, Z-Image, Qwen-Image,
  Qwen-Image-Edit, Wuerstchen v2, LTX Video, and LTX-2 / LTX-2.3
- Text-to-image, image-to-image, multimodal editing, inpainting, ControlNet,
  LoRA, prompt expansion, and Real-ESRGAN upscaling
- Text/image-to-video, multi-prompt video chains, native MP4 output, and
  checkpoint-dependent generated audio
- Guided Sequence authoring with per-clip prompts, duration, source images,
  Continue motion / Cut / Crossfade transitions, resumable jobs, and explicit
  remote-machine routing
- Quantized model variants, encoder fallback, smart VRAM placement, and FLUX
  block offloading
- Local CLI, native desktop, browser, TUI, iPhone, Discord, and authenticated
  REST/SSE clients

Browse the [model catalog](https://utensils.io/mold/models/) for sizes, VRAM
requirements, and recommended settings.

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

Or open the keyboard-first Mold Studio terminal interface:

```bash
mold tui
```

<p align="center">
  <img src="website/public/gallery/tui-generate.png" alt="Mold TUI Create workspace with image preview" width="720" />
  <br/>
  <em>The TUI Create workspace with a native terminal image preview</em>
</p>

## Remote and cloud GPUs

Run the engine where the GPU lives and use any Mold client over HTTP:

```bash
# GPU machine
mold serve

# Laptop or another client
MOLD_HOST=http://gpu-server:7680 mold run "a cat"
```

For managed cloud jobs, `mold runpod` can provision a GPU, reuse network
volumes, stream progress, and save the result locally. See the
[remote workflow](https://utensils.io/mold/guide/remote-workflows) and
[RunPod](https://utensils.io/mold/deployment/runpod-cli) guides.

## Project

Mold is a Rust workspace built on
[candle](https://github.com/huggingface/candle). Its CLI-native architecture
keeps the engine useful from a terminal, shell pipeline, agent, CI job, native
app, or custom client. The documentation covers the
[CLI](https://utensils.io/mold/guide/cli-reference),
[configuration](https://utensils.io/mold/guide/configuration),
[deployment](https://utensils.io/mold/deployment/), and
[HTTP API](https://utensils.io/mold/api/).

Core contributors:
[James Brink](https://jamesbrink.online/) and
[Jeffrey Dilley](mailto:jeff.dilley@gmail.com).

Licensed under the [MIT License](LICENSE).
