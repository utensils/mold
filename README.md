# mold 🧪

[![CI](https://github.com/utensils/mold/actions/workflows/ci.yml/badge.svg)](https://github.com/utensils/mold/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://www.rust-lang.org)
[![Nix Flake](https://img.shields.io/badge/nix-flake-blue.svg)](https://nixos.wiki/wiki/Flakes)

Local AI image generation CLI — run FLUX, SDXL, and Z-Image diffusion models directly on your GPU, with a built-in inference server for remote rendering.

## What it does

- **Just works**: `mold run "a cat"` — auto-pulls model, runs locally on your GPU
- **Three model families**: FLUX.1 (schnell, dev, krea), SDXL (base, turbo, DreamShaper, Juggernaut, RealVis, Playground), and Z-Image (turbo)
- **Model management**: `mold pull`, `mold list`, `mold rm`, `mold ps` — download, list, remove, and manage models
- **Remote capable**: Point at a GPU server with `MOLD_HOST` or run `mold serve` for a REST API
- **Pipe-friendly**: `mold run "a cat" | viu -` — composable with Unix tools
- **Quantized models**: GGUF Q4/Q6/Q8 for FLUX and Z-Image, FP16 safetensors for SDXL
- **Smart memory management**: Sequential loading, drop-and-reload, quantized encoder auto-fallback

## Requirements

- NVIDIA GPU with CUDA **or** Apple Silicon with Metal
- Model files auto-downloaded on first use via `mold pull` or `mold run`

## Quick start

### Run directly from GitHub (Nix)

No clone needed — run mold straight from the flake:

```bash
# Run on macOS (Metal) or Linux (CUDA)
nix run github:utensils/mold -- run "a cat riding a motorcycle"

# Enter a dev shell with all tools
nix develop github:utensils/mold
```

### Build from source

```bash
# With Nix (recommended)
nix build                # Builds with GPU support (CUDA on Linux, Metal on macOS)
nix run -- run "a cat"   # Build and run in one step

# With Cargo
cargo build --release -p mold-cli --features cuda    # Linux
cargo build --release -p mold-cli --features metal   # macOS
```

### Generate images

```bash
# Generate an image (auto-pulls model on first use)
mold run "a cat riding a motorcycle through neon-lit streets"

# Use a specific model
mold run flux-dev:q4 "a turtle in the desert"
mold run sdxl-turbo "a sunset over mountains"
mold run dreamshaper-xl "fantasy castle on a cliff"

# Pipe to an image viewer
mold run "neon cityscape" | viu -
mold run "a portrait" | img2sixel

# Or start a server for remote rendering
mold serve
MOLD_HOST=http://gpu-host:7680 mold run "a sunset"
```

## Available models

### FLUX (GGUF quantized)

| Name | Steps | Size | Description |
|------|-------|------|-------------|
| `flux-schnell:q8` | 4 | 12GB | Fast 4-step, general purpose |
| `flux-schnell:q4` | 4 | 7.5GB | Fast 4-step, smaller footprint |
| `flux-dev:q8` | 25 | 12GB | Full quality, 20+ steps |
| `flux-dev:q4` | 25 | 7GB | Smaller/faster, good quality |
| `flux-krea:q8` | 25 | 12.7GB | Aesthetic photography fine-tune |

### SDXL (FP16 safetensors)

| Name | Steps | Size | Description |
|------|-------|------|-------------|
| `sdxl-base:fp16` | 25 | 5.1GB | Official Stability AI base |
| `sdxl-turbo:fp16` | 4 | 5.1GB | Ultra-fast 1-4 step generation |
| `dreamshaper-xl:fp16` | 8 | 5.1GB | Fantasy, concept art, stylized |
| `juggernaut-xl:fp16` | 30 | 5.1GB | Photorealism, cinematic |
| `realvis-xl:fp16` | 25 | 5.1GB | Photorealism, versatile |
| `playground-v2.5:fp16` | 25 | 5.1GB | Aesthetic quality, artistic |

### Z-Image (GGUF quantized)

| Name | Steps | Size | Description |
|------|-------|------|-------------|
| `z-image-turbo:q8` | 9 | 6.6GB | Fast 9-step, Qwen3 text encoder |
| `z-image-turbo:q4` | 9 | 3.8GB | Smaller footprint, good quality |
| `z-image-turbo:bf16` | 9 | 12.2GB | Full precision BF16 |

Bare model names default to `:q8` (FLUX/Z-Image) or `:fp16` (SDXL).

## API

```
GET    /health              → 200 OK
GET    /api/status          → ServerStatus JSON
GET    /api/models          → ModelInfo[] JSON
POST   /api/generate        → image/png bytes
POST   /api/models/load     → 200 OK (hot-swap model)
DELETE /api/models/unload   → 200 OK (free GPU memory)
```

### Generate request

```json
{
  "prompt": "a glowing robot",
  "model": "flux-schnell",
  "width": 768,
  "height": 768,
  "steps": 4,
  "guidance": 3.5,
  "seed": null,
  "batch_size": 1,
  "output_format": "png"
}
```

Constraints: width/height must be multiples of 16, max 1024, min 1. Steps 1–100.

## Config

`~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy):

```toml
default_model = "flux-schnell:q8"
server_port = 7680
default_width = 1024
default_height = 1024

[models."flux-schnell:q8"]
transformer = "/path/to/flux1-schnell-Q8_0.gguf"
vae = "/path/to/ae.safetensors"
t5_encoder = "/path/to/t5xxl_fp16.safetensors"
clip_encoder = "/path/to/clip_l.safetensors"
t5_tokenizer = "/path/to/t5.tokenizer.json"
clip_tokenizer = "/path/to/clip.tokenizer.json"
```

Config is auto-written by `mold pull` — manual editing is rarely needed.

## Shell completions

```bash
# bash
source <(mold completions bash)

# zsh
source <(mold completions zsh)

# fish
mold completions fish | source
```

## Architecture

```
mold-cli       Single binary: CLI + serve (mold run / serve / pull / list / completions)
mold-server    Axum REST server (library, used by mold-cli via `mold serve`)
mold-inference FLUX + SDXL engines (candle ML framework, smart GPU/CPU encoder placement)
mold-core      Shared types, config, model manifests, HuggingFace download client
```

Built with [candle](https://github.com/huggingface/candle) — pure Rust ML framework with CUDA and Metal backends. No Python, no libtorch.
