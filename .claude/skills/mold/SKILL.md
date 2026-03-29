---
name: mold
description: Generate AI images locally using the mold CLI. Use when asked to generate, create, or produce images from text prompts, transform existing images (img2img), or manage local AI models.
argument-hint: [prompt or command]
allowed-tools: Bash, Read, Glob, Grep
---

# mold — Local AI Image Generation CLI

Generate images from text prompts using FLUX, SD1.5, SDXL, SD3.5, Z-Image, and other diffusion models running on local GPU hardware.

## Quick Reference

```bash
mold run "a cat on a skateboard"                    # Generate with default model
mold run flux-dev:q4 "a sunset over mountains"      # Specific model
mold run "a portrait" -o portrait.png               # Custom output path
mold run "a dog" --seed 42 --steps 20               # Reproducible generation
mold run "watercolor" --image photo.png --strength 0.7  # img2img
mold run flux-dev:bf16 "portrait" --lora style.safetensors --lora-scale 0.8  # LoRA adapter
```

## How to Use This Skill

Parse `$ARGUMENTS` to determine the action:

- If arguments look like a **prompt** (natural language), run `mold run "<prompt>"` with sensible defaults
- If arguments start with a **subcommand** (`pull`, `list`, `default`, `serve`, `info`, `ps`, `rm`, `unload`), run that subcommand
- If arguments include **flags** (`--model`, `--image`, `--steps`, etc.), pass them through

## Generating Images

### Basic Usage

```bash
# Default model (flux-schnell:q8 — fast 4-step)
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

# Negative prompt (CFG-based models: SD1.5, SDXL, SD3)
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
- Base model must be FLUX BF16 (e.g. `flux-dev:bf16`, `flux-schnell:bf16`)
- LoRA file must be `.safetensors` format (diffusers-format keys)
- GGUF quantized models do not support LoRA yet
- On 24GB cards, LoRA auto-uses block-level offloading (3-5x slower but fits in VRAM)

**Per-model config defaults** (config.toml):
```toml
[models."flux-dev:bf16"]
# ... other fields ...
lora = "/path/to/default-adapter.safetensors"
lora_scale = 0.8
```

### Model Selection Guide

Pick the right model for the task:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `flux-schnell:q8` | Fast (4 steps) | Good | Quick iterations, drafts |
| `flux-dev:q4` | Slow (25 steps) | Excellent | Final quality, detailed |
| `sdxl-turbo:fp16` | Fast (4 steps) | Good | Quick SDXL generation |
| `sd15:fp16` | Medium (25 steps) | Good | ControlNet, 512x512 |
| `z-image-turbo:q8` | Fast (9 steps) | Excellent | High quality, Qwen3 encoder |

Default model if none specified: `flux-schnell:q8`

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

### Available Models

**FLUX.1**: `flux-schnell:q8`, `flux-schnell:q6`, `flux-schnell:q4`, `flux-schnell:bf16`, `flux-dev:q8`, `flux-dev:q6`, `flux-dev:q4`, `flux-dev:bf16`, `flux-krea:q8`, `flux-krea:q6`, `flux-krea:q4`, `flux-krea:fp8`

**FLUX.1 Fine-tunes**: `jibmix-flux:q4`, `jibmix-flux:q5`, `jibmix-flux:fp8`, `ultrareal-v4:q8`, `ultrareal-v4:q5`, `ultrareal-v4:q4`, `ultrareal-v3:q8`, `ultrareal-v3:q6`, `ultrareal-v3:q4`, `ultrareal-v2:bf16`, `iniverse-mix:fp8`

**SDXL**: `sdxl-base:fp16`, `sdxl-turbo:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, `playground-v2.5:fp16`, `dreamshaper-xl:fp16`, `pony-v6:fp16`, `cyberrealistic-pony:fp16`

**SD 1.5**: `sd15:fp16`, `dreamshaper-v8:fp16`, `realistic-vision-v5:fp16`

**SD 3.5**: `sd3.5-large:q8`, `sd3.5-large:q4`, `sd3.5-large-turbo:q8`, `sd3.5-medium:q8`

**Z-Image**: `z-image-turbo:bf16`, `z-image-turbo:q8`, `z-image-turbo:q6`, `z-image-turbo:q4`

**Flux.2 Klein**: `flux2-klein:bf16`, `flux2-klein:q8`, `flux2-klein:q6`, `flux2-klein:q4`

**Alpha**: `qwen-image:bf16`, `qwen-image:q8`, `qwen-image:q6`, `qwen-image:q4`, `wuerstchen-v2:fp16`

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
echo "a dog in space" | mold run flux-schnell      # Prompt from stdin
cat photo.png | mold run "style" --image - | viu - # Full pipeline
```

Force stdout in interactive mode: `mold run "a cat" --output -`

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

## Server Mode

```bash
mold serve                           # Start on 0.0.0.0:7680
mold serve --port 8080               # Custom port
mold ps                              # Check server status
mold unload                          # Free GPU memory

# Connect from another machine
MOLD_HOST=http://gpu-host:7680 mold run "a cat"

# Save all generated images to disk (server-side persistence)
MOLD_OUTPUT_DIR=/srv/mold/gallery mold serve
```

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOLD_HOME` | `~/.mold` | Base directory for config, cache, and default models |
| `MOLD_DEFAULT_MODEL` | `flux-schnell` | Default model (smart fallback to only downloaded model) |
| `MOLD_HOST` | `http://localhost:7680` | Remote server URL |
| `MOLD_MODELS_DIR` | `$MOLD_HOME/models` | Model storage path |
| `MOLD_OUTPUT_DIR` | unset | Save server-generated images to this directory |
| `MOLD_PORT` | `7680` | Server port |
| `MOLD_LOG` | `warn` | Log level (trace/debug/info/warn/error) |
| `MOLD_EAGER` | unset | Set `1` to keep all components loaded |
| `MOLD_OFFLOAD` | unset | Set `1` to force CPU↔GPU block streaming (reduces VRAM, slower) |
| `MOLD_EMBED_METADATA` | `1` | Set `0` to disable PNG metadata |
| `MOLD_PREVIEW` | unset | Set `1` to display generated images inline in the terminal |
| `MOLD_T5_VARIANT` | `auto` | T5 encoder: auto/fp16/q8/q6/q5/q4/q3 |
| `MOLD_QWEN3_VARIANT` | `auto` | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3 |
| `MOLD_SCHEDULER` | unset | SD1.5/SDXL: ddim/euler-ancestral/uni-pc |
| `MOLD_CORS_ORIGIN` | unset | Restrict server CORS to specific origin |
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

Models auto-pull if not downloaded: `mold run flux-schnell "a cat"` will download the model first if needed.

## Practical Tips

- Use `flux-schnell:q8` for fast iterations (4 steps, ~10s on RTX 4090)
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

Mold includes an optional Discord bot (`mold-discord`) that bridges Discord slash commands to a running `mold serve` instance. The bot depends only on `mold-core` (HTTP client) — no GPU needed on the bot host.

### Running

```bash
export MOLD_DISCORD_TOKEN="your-bot-token"
export MOLD_HOST="http://gpu-host:7680"  # optional, defaults to localhost
mold-discord
```

### Slash Commands

- `/generate <prompt> [model] [width] [height] [steps] [guidance] [seed]` — generate an image
- `/expand <prompt> [model_family] [variations]` — expand a short prompt into detailed image generation prompts
- `/models` — list available models with status
- `/status` — show server health, GPU info, uptime

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLD_DISCORD_TOKEN` | — | Bot token (falls back to `DISCORD_TOKEN`) |
| `MOLD_HOST` | `http://localhost:7680` | mold server URL |
| `MOLD_DISCORD_COOLDOWN` | `10` | Per-user cooldown (seconds) |

### NixOS

```nix
services.mold.discord = {
  enable = true;
  package = inputs.mold.packages.${system}.mold-discord;
  tokenFile = config.age.secrets.discord-token.path;
};
```
