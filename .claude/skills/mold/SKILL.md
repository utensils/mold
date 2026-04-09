---
name: mold
description: Generate AI images and video locally using the mold CLI. Use when asked to generate images from text prompts, create video clips, transform existing images (img2img), or manage local AI models.
argument-hint: [prompt or command]
allowed-tools: Bash, Read, Glob, Grep
---

# mold — Local AI Image Generation CLI

Generate images and video from text prompts using FLUX, SD1.5, SDXL, SD3.5, Z-Image, Flux.2 Klein, Qwen-Image, LTX Video, and Wuerstchen diffusion models running on local GPU hardware.

## Quick Reference

```bash
mold run "a cat on a skateboard"                    # Generate with default model
mold run flux-dev:q4 "a sunset over mountains"      # Specific model
mold run "a portrait" -o portrait.png               # Custom output path
mold run "a dog" --seed 42 --steps 20               # Reproducible generation
mold run "watercolor" --image photo.png --strength 0.7  # img2img
mold run qwen-image-edit-2511:q4 "make the chair red leather" --image chair.png --image swatch.png --qwen2-variant q4
mold run qwen-image:q2 "a poster" --qwen2-variant q6    # Qwen-Image quantized text encoder
mold run flux-dev:bf16 "portrait" --lora style.safetensors --lora-scale 0.8  # LoRA adapter
```

## How to Use This Skill

Parse `$ARGUMENTS` to determine the action:

- If arguments look like a **prompt** (natural language), run `mold run "<prompt>"` with sensible defaults
- If arguments start with a **subcommand** (`pull`, `list`, `default`, `config`, `serve`, `server`, `info`, `ps`, `rm`, `unload`, `update`, `stats`, `clean`, `tui`, `completions`, `version`), run that subcommand
- If arguments include **flags** (`--model`, `--image`, `--steps`, etc.), pass them through

## Generating Images

### Basic Usage

```bash
# Default model (flux2-klein:q8 — fast 4-step, Apache 2.0, fully ungated)
mold run "a red apple on a wooden table"

# Explicit model selection
mold run flux-dev:q4 "a photorealistic landscape at golden hour"

# With output path
mold run "cyberpunk cityscape" -o cityscape.png

# Reproducible output (same seed = same image)
mold run "a cat" --seed 42

# Custom dimensions (must be multiples of 16)
mold run "a banner" --width 1024 --height 512

# Batch generation (multiple images)
mold run "abstract art" --batch 4 --seed 100 -o art.png

# JPEG output
mold run "a sunset" --format jpeg -o sunset.jpg

# Disable PNG metadata embedding
mold run "a cat" --no-metadata

# Display image inline in terminal after generation (requires `preview` feature)
mold run "a cat" --preview

# Negative prompt (CFG-based models: SD1.5, SDXL, SD3, Wuerstchen)
mold run sd15:fp16 "a portrait" -n "blurry, watermark, ugly, bad anatomy"
mold run sdxl:fp16 "a landscape" --negative-prompt "low quality, jpeg artifacts"
mold run sd15:fp16 "a cat" --no-negative  # suppress config default
```

### Prompt Expansion

Expand short prompts into detailed image generation prompts using a local LLM (Qwen3-1.7B). The expansion model auto-downloads on first use (~1.8GB).

```bash
# Preview expanded prompt without generating
mold expand "a cat"

# Expand with multiple variations
mold expand "cyberpunk city" --variations 5

# Expand as JSON
mold expand "a cat" --variations 3 --json

# Generate with expansion (short prompt -> detailed prompt -> image)
mold run "a cat" --expand

# Batch + expand: each image gets a unique expanded prompt
mold run "a sunset" --expand --batch 4

# Use a specific expansion backend (OpenAI-compatible API)
mold run "a cat" --expand --expand-backend http://localhost:11434/v1

# Disable expansion (overrides config/env default)
mold run "a cat" --no-expand
```

The expansion model is dropped from memory before diffusion begins, so it doesn't compete for VRAM.

### LoRA Adapters

Apply LoRA (Low-Rank Adaptation) fine-tuned adapters on top of FLUX BF16 base models:

```bash
# Basic LoRA usage
mold run flux-dev:bf16 "a portrait" --lora /path/to/adapter.safetensors

# Adjust LoRA strength (0.0 = no effect, 1.0 = full, up to 2.0)
mold run flux-dev:bf16 "anime style" --lora style.safetensors --lora-scale 0.7

# LoRA with other options (img2img, seed, etc.)
mold run flux-dev:bf16 "oil painting" --lora art.safetensors --image photo.png --strength 0.6
```

**Requirements:**
- FLUX models only (BF16 or GGUF quantized)
- LoRA file must be `.safetensors` format (diffusers-format keys)
- BF16 models on 24GB cards auto-use block-level offloading (3-5x slower but fits in VRAM)
- GGUF Q4/Q6 work at 1024x1024; Q8 works at 512x512 (Q8 + LoRA at 1024x1024 is tight on 24GB, see #95)

**Per-model config defaults** (config.toml):
```toml
[models."flux-dev:bf16"]
# ... other fields ...
lora = "/path/to/default-adapter.safetensors"
lora_scale = 0.8
```

### Video Generation

Generate video clips with LTX Video models. Output defaults to APNG (lossless, with metadata).

```bash
# Basic video generation (25 frames, APNG output)
mold run ltx-video-0.9.6-distilled:bf16 "a cat walking across a windowsill" --frames 25

# Custom frame count (must be 8n+1: 9, 17, 25, 33, 49, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "ocean waves at sunset" --frames 49

# MP4 output (QuickTime compatible)
mold run ltx-video-0.9.6-distilled:bf16 "a campfire at night" --frames 17 --format mp4

# GIF for pipe-friendly output
mold run ltx-video-0.9.6-distilled:bf16 "a sunset" --format gif | mpv -

# WebP animated output
mold run ltx-video-0.9.6-distilled:bf16 "a waterfall" --frames 9 --format webp -o waterfall.webp
```

**Constraints:** Frame count must be 8n+1 (9, 17, 25, 33, 49, ...). Dimensions must be multiples of 32. Current LTX defaults are 1216x704, 25 frames, 30 fps. Distilled models use fewer steps.

**Current status:** `ltx-video-0.9.6-distilled:bf16` is still the safest default, but the `0.9.8` models now run the full multiscale refinement path. mold pulls the required spatial upscaler asset explicitly, keeps the shared T5 assets under `shared/flux/...`, and intentionally continues using the compatible `LTX-Video-0.9.5` VAE source until the newer VAE layout is ported.

**Output formats:** `apng` (default, lossless, metadata), `gif` (256 colors), `mp4` (H.264, requires `mp4` feature), `webp` (requires `webp` feature).

### Model Selection Guide

Pick the right model for the task:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `flux-schnell:q8` | Fast (4 steps) | Good | Quick iterations, drafts |
| `flux-dev:q4` | Slow (25 steps) | Excellent | Final quality, detailed |
| `flux2-klein:q8` | Fast (4 steps) | Good | Low VRAM, lightweight FLUX |
| `flux2-klein-9b:q8` | Fast (4 steps) | Excellent | Higher quality 9B, non-commercial |
| `sdxl-turbo:fp16` | Fast (4 steps) | Good | Quick SDXL generation |
| `sd15:fp16` | Medium (25 steps) | Good | ControlNet, 512x512 |
| `z-image-turbo:q8` | Fast (9 steps) | Excellent | High quality, Qwen3 encoder |
| `qwen-image:q4` | Slow (50 steps) | Good | Stable base Qwen GGUF on 24 GB cards |
| `qwen-image-2512:q4` | Slow (50 steps) | Good | Stable 2512 GGUF on 24 GB cards |
| `qwen-image:q8` | Slow (50 steps) | Better | Best base GGUF quality, validated at 768x768 on 24 GB |
| `ltx-video-0.9.6-distilled:bf16` | Fast (8 steps) | Good | Text-to-video, 30fps |
| `ltx-video-0.9.8-2b-distilled:bf16` | Fast (7+3 steps) | Better | Newer checkpoint family with full multiscale refinement |

Default model if none specified: `flux2-klein:q8`

### Model Defaults

| Model | Steps | Guidance | Resolution |
|-------|-------|----------|------------|
| `flux-schnell` | 4 | 0.0 | 1024x1024 |
| `flux-dev` | 25 | 3.5 | 1024x1024 |
| `sdxl-base` | 25 | 7.5 | 1024x1024 |
| `sdxl-turbo` | 4 | 0.0 | 512x512 |
| `sd15` | 25 | 7.5 | 512x512 |
| `sd3.5-large` | 28 | 4.0 | 1024x1024 |
| `z-image-turbo` | 9 | 0.0 | 1024x1024 |
| `flux2-klein` | 4 | 0.0 | 1024x1024 |
| `flux2-klein-9b` | 4 | 1.0 | 1024x1024 |
| `qwen-image` | 50 | 4.0 | 1328x1328 |
| `qwen-image-2512` | 50 | 4.0 | 1328x1328 |
| `ltx-video-0.9.6-distilled` | 8 | 1.0 | 1216x704 (25 frames, 30fps) |
| `ltx-video-0.9.8-2b-distilled` | 7+3 | 1.0 | 1216x704 (25 frames, 30fps, multiscale refine) |

### Available Models

**FLUX.1**: `flux-schnell:q8`, `flux-schnell:q6`, `flux-schnell:q4`, `flux-schnell:bf16`, `flux-dev:q8`, `flux-dev:q6`, `flux-dev:q4`, `flux-dev:bf16`, `flux-krea:q8`, `flux-krea:q6`, `flux-krea:q4`, `flux-krea:fp8`

**FLUX.1 Fine-tunes**: `jibmix-flux:q4`, `jibmix-flux:q5`, `jibmix-flux:fp8`, `ultrareal-v4:q8`, `ultrareal-v4:q5`, `ultrareal-v4:q4`, `ultrareal-v3:q8`, `ultrareal-v3:q6`, `ultrareal-v3:q4`, `ultrareal-v2:bf16`, `iniverse-mix:fp8`

**SDXL**: `sdxl-base:fp16`, `sdxl-turbo:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, `playground-v2.5:fp16`, `dreamshaper-xl:fp16`, `pony-v6:fp16`, `cyberrealistic-pony:fp16`

**SD 1.5**: `sd15:fp16`, `dreamshaper-v8:fp16`, `realistic-vision-v5:fp16`

**SD 3.5**: `sd3.5-large:q8`, `sd3.5-large:q4`, `sd3.5-large-turbo:q8`, `sd3.5-medium:q8`

**Z-Image**: `z-image-turbo:bf16`, `z-image-turbo:q8`, `z-image-turbo:q6`, `z-image-turbo:q4`

**Flux.2 Klein**: `flux2-klein:bf16`, `flux2-klein:q8`, `flux2-klein:q6`, `flux2-klein:q4`

**Flux.2 Klein-9B**: `flux2-klein-9b:bf16`, `flux2-klein-9b:q8`, `flux2-klein-9b:q6`, `flux2-klein-9b:q4`

**Wuerstchen**: `wuerstchen-v2:fp16`

**Qwen-Image**: `qwen-image:q8`, `qwen-image:q6`, `qwen-image:q5`, `qwen-image:q4`, `qwen-image:q3`, `qwen-image:q2`, `qwen-image:fp8`, `qwen-image:bf16`

**Qwen-Image-2512**: `qwen-image-2512:q8`, `qwen-image-2512:q6`, `qwen-image-2512:q5`, `qwen-image-2512:q4`, `qwen-image-2512:q3`, `qwen-image-2512:q2`, `qwen-image-lightning:fp8`, `qwen-image-lightning:fp8-8step`, `qwen-image-2512:bf16`

**LTX Video**: `ltx-video-0.9.6:bf16`, `ltx-video-0.9.6-distilled:bf16`, `ltx-video-0.9.8-2b-distilled:bf16`, `ltx-video-0.9.8-13b-dev:bf16`, `ltx-video-0.9.8-13b-distilled:bf16`
**Qwen-Image text encoder controls**:
- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`
- On Apple Metal/MPS, `auto` prefers quantized Qwen2.5-VL GGUF text encoders (`q6`, then `q4`) to reduce memory pressure
- On CUDA, `auto` prefers BF16 when there is enough post-transformer headroom and falls back to quantized GGUF variants for resident/edit paths when BF16 would be too heavy
- `qwen-image-edit-2511:*` uses repeatable `--image` inputs and a distinct `qwen-image-edit` family. Local inference is implemented with the Qwen2.5-VL vision tower, packed edit latents, and true-CFG norm rescaling. Quantized `--qwen2-variant` values are supported for the edit family through a GGUF language path plus staged vision sidecar.
**ControlNet (SD1.5)**: `controlnet-canny-sd15:fp16`, `controlnet-depth-sd15:fp16`, `controlnet-openpose-sd15:fp16`

**Utility (LLM)**: `qwen3-expand:q8`, `qwen3-expand-small:q8`

### Name Resolution

Bare names auto-resolve: `flux2-klein` -> `flux2-klein:q8`, `flux-dev` -> `flux-dev:q8`, `sdxl-base` -> `sdxl-base:fp16`, `sd15` -> `sd15:fp16`

FP8 safetensors models are automatically quantized to Q8 GGUF on first use (one-time conversion, cached at `$MOLD_HOME/cache/`).

## img2img (Image-to-Image)

Transform an existing image with a text prompt:

```bash
# Basic img2img
mold run "oil painting style" --image photo.png --strength 0.7

# Low strength = subtle changes (close to original)
mold run "enhance details" --image photo.png --strength 0.3

# High strength = major transformation
mold run "anime style" --image photo.png --strength 0.9

# From stdin
cat photo.png | mold run "watercolor" --image - --strength 0.6
```

**Strength guide**: `0.0` = no change, `0.3` = subtle, `0.5` = balanced, `0.75` = strong (default), `1.0` = full txt2img

### Inpainting

Repaint specific regions using a mask:

```bash
mold run "a golden retriever" --image park.png --mask mask.png
# mask: white = repaint, black = preserve
```

### ControlNet (SD1.5 only)

```bash
mold run "a person" --control edges.png --control-model controlnet-canny-sd15:fp16
mold run "interior" --control depth.png --control-model controlnet-depth-sd15:fp16 --control-scale 0.8
```

## Piping

mold is pipe-friendly. When stdout is not a TTY, image bytes go to stdout and status to stderr:

```bash
mold run "a cat" | viu -                           # Preview in terminal
mold run "a cat" | convert - output.webp           # Convert format
echo "a dog in space" | mold run flux2-klein        # Prompt from stdin
cat photo.png | mold run "style" --image - | viu - # Full pipeline
```

Force stdout in interactive mode: `mold run "a cat" --output -`

## Upscaling

Upscale images to 2x or 4x resolution using Real-ESRGAN super-resolution models.

```bash
# Upscale with default model (real-esrgan-x4plus:fp16, auto-downloads ~32MB)
mold upscale photo.png

# Choose a specific model
mold upscale photo.png -m real-esrgan-x4plus-anime:fp16

# Custom output path
mold upscale photo.png -o photo_4x.png

# Display upscaled image inline
mold upscale photo.png --preview

# Pipe: generate then upscale
mold run "a cat" | mold upscale -

# Force local (skip server)
mold upscale photo.png --local

# Smaller tile size for limited VRAM
mold upscale large_photo.png --tile-size 256
```

### Available Upscaler Models

| Model | Scale | Size | Best For |
|-------|-------|------|----------|
| `real-esrgan-x4plus:fp16` | 4x | 32 MB | General photos (default) |
| `real-esrgan-x4plus:fp32` | 4x | 64 MB | General photos (full precision) |
| `real-esrgan-x2plus:fp16` | 2x | 32 MB | Subtle 2x enhancement |
| `real-esrgan-x4plus-anime:fp16` | 4x | 8.5 MB | Anime/illustration |
| `real-esrgan-anime-v3:fp32` | 4x | 2.4 MB | Fast anime/video |

## Model Management

```bash
mold list                    # List downloaded + available models
mold pull flux-dev:q4        # Download a model
mold pull flux-dev:q4 --skip-verify  # Download, skip SHA-256 check
mold default                 # Show current default model and how it was resolved
mold default flux-dev:q4     # Set default model (validates name, warns if not downloaded)
mold info                    # Installation overview (paths, models, server status)
mold info flux-dev:q4        # Show model details and file sizes
mold rm flux-dev:q4          # Remove a downloaded model
mold rm flux-dev:q4 --force  # Remove without confirmation
```

## Configuration Management

View and edit `config.toml` settings from the CLI using dot-notation keys:

```bash
mold config list                          # Show all settings grouped by section
mold config list --json                   # Machine-readable output
mold config get server_port               # Get a single value
mold config get server_port --raw         # Raw value for scripting
mold config set server_port 8080          # Set and persist a value
mold config set expand.enabled true       # Nested key (dot-notation)
mold config set output_dir none           # Clear an optional field
mold config set models.flux-dev:q4.default_steps 30  # Per-model setting
mold config path                          # Show config file location
mold config edit                          # Open in $EDITOR
```

Keys use dot-notation matching the TOML structure. Boolean values accept `true`/`false`, `on`/`off`, or `1`/`0`. Use `none` to clear optional fields. Values are validated (port range, enum options, numeric bounds) before saving. Environment variable overrides are shown when active.

## Self-Update

```bash
mold update                       # Update to latest GitHub release
mold update --check               # Check for updates without installing
mold update --version v0.6.0      # Install a specific version
mold update --force               # Reinstall even if already up-to-date
```

Downloads the correct platform-specific binary from GitHub releases, verifies SHA-256 checksum, and replaces the running binary in-place. Detects Nix/Homebrew installations and suggests using the package manager instead. Respects `GITHUB_TOKEN` for API rate limits and `MOLD_CUDA_ARCH` for GPU architecture override on Linux.

## Server Mode

```bash
mold serve                           # Start foreground server on 0.0.0.0:7680
mold serve --port 8080               # Custom port

# Daemon management (background server)
mold server start                    # Start background server daemon
mold server start --port 8080        # Custom port
mold server start --bind 127.0.0.1   # Custom bind address
mold server start --models-dir /path # Custom models directory
mold server start --log-file         # Enable file logging
mold server status                   # Show managed server status (PID, port, uptime, models)
mold server stop                     # Graceful shutdown (HTTP → SIGTERM → SIGKILL)

mold ps                              # Check server status
mold unload                          # Free GPU memory

# Connect from another machine
MOLD_HOST=http://gpu-host:7680 mold run "a cat"

# Custom image output directory (default: ~/.mold/output/)
MOLD_OUTPUT_DIR=/srv/mold/output mold serve
```

### Prometheus Metrics

When built with the `metrics` feature flag (included in Docker images and Nix builds), the server exposes a `GET /metrics` endpoint in Prometheus text exposition format. This endpoint is excluded from auth and rate limiting for monitoring scrapers.

Metrics include: HTTP request rates/latency, generation duration, queue depth, model load tracking, GPU memory usage, and server uptime.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOLD_HOME` | `~/.mold` | Base directory for config, cache, and default models |
| `MOLD_DEFAULT_MODEL` | `flux2-klein` | Default model (smart fallback to only downloaded model) |
| `MOLD_HOST` | `http://localhost:7680` | Remote server URL |
| `MOLD_MODELS_DIR` | `$MOLD_HOME/models` | Model storage path |
| `MOLD_OUTPUT_DIR` | `~/.mold/output` | Image output directory (set empty to disable) |
| `MOLD_THUMBNAIL_WARMUP` | unset | Set `1` to prebuild gallery thumbnails at server startup |
| `MOLD_PORT` | `7680` | Server port |
| `MOLD_LOG` | `warn` | Log level (trace/debug/info/warn/error) |
| `MOLD_EAGER` | unset | Set `1` to keep all components loaded |
| `MOLD_OFFLOAD` | unset | Set `1` to force CPU↔GPU block streaming (reduces VRAM, slower) |
| `MOLD_EMBED_METADATA` | `1` | Set `0` to disable PNG metadata |
| `MOLD_PREVIEW` | unset | Set `1` to display generated images inline in the terminal |
| `MOLD_T5_VARIANT` | `auto` | T5 encoder: auto/fp16/q8/q6/q5/q4/q3 |
| `MOLD_QWEN3_VARIANT` | `auto` | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3 |
| `MOLD_SCHEDULER` | unset | SD1.5/SDXL: ddim/euler-ancestral/uni-pc |
| `MOLD_API_KEY` | unset | API key for server auth (single, comma-separated, or `@/path/to/keys.txt`) |
| `MOLD_RATE_LIMIT` | unset | Per-IP rate limit for generation endpoints (e.g., `10/min`) |
| `MOLD_RATE_LIMIT_BURST` | unset | Burst allowance override (defaults to 2x rate) |
| `MOLD_CORS_ORIGIN` | unset | Restrict server CORS to specific origin |
| `MOLD_UPSCALE_MODEL` | unset | Default upscaler model for `mold upscale` |
| `MOLD_UPSCALE_TILE_SIZE` | unset | Tile size for memory-efficient upscaling (0 to disable) |
| `MOLD_EXPAND` | unset | Set `1` to enable prompt expansion by default |
| `MOLD_EXPAND_BACKEND` | `local` | Expansion backend: `local` or OpenAI-compatible API URL |
| `MOLD_EXPAND_MODEL` | `qwen3-expand:q8` | LLM model for local expansion |
| `MOLD_EXPAND_TEMPERATURE` | `0.7` | Sampling temperature for expansion |
| `MOLD_EXPAND_THINKING` | unset | Set `1` to enable thinking mode in expansion LLM |
| `MOLD_EXPAND_SYSTEM_PROMPT` | unset | Custom single-expansion system prompt template |
| `MOLD_EXPAND_BATCH_PROMPT` | unset | Custom batch-variation system prompt template |
| `HF_TOKEN` | unset | HuggingFace token for gated models |

## Inference Modes

1. **Remote** (default): connects to `mold serve` via HTTP
2. **Local fallback**: if server unreachable, auto-falls back to local GPU
3. **Local forced** (`--local`): skip server, run on local GPU directly

Models auto-pull if not downloaded: `mold run flux2-klein "a cat"` will download the model first if needed.

## Practical Tips

- Use `flux2-klein:q8` for fast iterations (4 steps, ~10s on RTX 4090)
- Use `flux-dev:q4` for final quality images (25 steps)
- Use `--seed` for reproducibility — same seed + same prompt = same image
- Quantized models (q4/q6/q8) use less VRAM than fp16/bf16
- FP8 safetensors models auto-convert to Q8 GGUF on first use (fits 24GB cards)
- `--eager` trades VRAM for speed (keeps encoders loaded between generations)
- Dimensions must be multiples of 16; total pixels capped at ~1.1 megapixels
- For img2img, source images auto-resize to fit the model's native resolution (preserving aspect ratio). A 1024x1024 source with SD1.5 (512x512 native) generates at 512x512; a 1920x1080 source generates at 512x288. Use `--width`/`--height` to override
- Set `MOLD_HOME` to relocate all mold data (config, cache, models)
- LoRA adapters require FLUX BF16 models; use `--lora-scale 0.5-0.8` for subtle effects
- On 24GB cards, LoRA + BF16 auto-offloads (slower but avoids OOM)

## Discord Bot

Mold includes an optional Discord bot that bridges Discord slash commands to a running `mold serve` instance. The bot depends only on `mold-core` (HTTP client) — no GPU needed on the bot host.

### Running

```bash
# Run server + bot in one process
MOLD_DISCORD_TOKEN="your-bot-token" mold serve --discord

# Or run the bot separately (connects to a remote server)
export MOLD_DISCORD_TOKEN="your-bot-token"
export MOLD_HOST="http://gpu-host:7680"  # optional, defaults to localhost
mold discord
```

### Slash Commands

- `/generate <prompt> [model] [width] [height] [steps] [guidance] [seed]` — generate an image
- `/expand <prompt> [model_family] [variations]` — expand a short prompt into detailed image generation prompts
- `/models` — list available models with status
- `/status` — show server health, GPU info, uptime
- `/quota` — check remaining daily generation quota
- `/admin reset-quota @user` — reset a user's daily quota (requires Manage Server)
- `/admin block @user` — temporarily block a user from generating (requires Manage Server)
- `/admin unblock @user` — unblock a previously blocked user (requires Manage Server)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLD_DISCORD_TOKEN` | — | Bot token (falls back to `DISCORD_TOKEN`) |
| `MOLD_HOST` | `http://localhost:7680` | mold server URL |
| `MOLD_DISCORD_COOLDOWN` | `10` | Per-user cooldown (seconds) |
| `MOLD_DISCORD_ALLOWED_ROLES` | — | Comma-separated role names/IDs for access control (unset = all) |
| `MOLD_DISCORD_DAILY_QUOTA` | — | Max generations per user per UTC day (unset = unlimited) |

### NixOS

```nix
services.mold.discord = {
  enable = true;
  package = inputs.mold.packages.${system}.mold-discord;
  tokenFile = config.age.secrets.discord-token.path;
};
```
