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
```

## How to Use This Skill

Parse `$ARGUMENTS` to determine the action:

- If arguments look like a **prompt** (natural language), run `mold run "<prompt>"` with sensible defaults
- If arguments start with a **subcommand** (`pull`, `list`, `serve`, `info`, `ps`, `rm`, `unload`), run that subcommand
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
```

### Model Selection Guide

Pick the right model for the task:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `flux-schnell:q8` | Fast (4 steps) | Good | Quick iterations, drafts |
| `flux-dev:q4` | Slow (25 steps) | Excellent | Final quality, detailed |
| `sdxl-turbo:fp16` | Fast (4 steps) | Good | Quick SDXL generation |
| `sd15:fp16` | Medium (25 steps) | Good | ControlNet, 512x512 |
| `z-image-turbo:q8` | Medium (30 steps) | Excellent | High quality, Qwen3 encoder |

Default model if none specified: `flux-schnell:q8`

### Model Defaults

| Model | Steps | Guidance | Resolution |
|-------|-------|----------|------------|
| `flux-schnell` | 4 | 0.0 | 768x768 |
| `flux-dev` | 25 | 3.5 | 768x768 |
| `sdxl-base` / `sdxl-turbo` | 25 / 4 | 7.5 / 0.0 | 1024x1024 |
| `sd15` | 25 | 7.5 | 512x512 |
| `sd3.5-large` | 30 | 5.0 | 1024x1024 |
| `z-image-turbo` | 30 | 7.5 | 768x768 |

### Available Models

**FLUX.1**: `flux-schnell:q8`, `flux-schnell:q6`, `flux-schnell:q4`, `flux-dev:q8`, `flux-dev:q6`, `flux-dev:q4`, `flux-krea:q8`, `flux-krea:q6`, `flux-krea:q4`

**SDXL**: `sdxl-base:fp16`, `sdxl-turbo:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, `playground-v2.5:fp16`

**SD 1.5**: `sd15:fp16`, `dreamshaper-v8:fp16`, `realistic-vision-v5:fp16`

**SD 3.5**: `sd3.5-large:q8`, `sd3.5-large:q4`, `sd3.5-large-turbo:q8`, `sd3.5-medium:q8`

**Z-Image**: `z-image-turbo:bf16`, `z-image-turbo:q8`, `z-image-turbo:q6`, `z-image-turbo:q4`

**Other**: `flux2-klein:bf16`, `qwen-image:bf16/:q8/:q6/:q4`, `wuerstchen-v2:fp16`

**ControlNet (SD1.5)**: `controlnet-canny-sd15:fp16`, `controlnet-depth-sd15:fp16`, `controlnet-openpose-sd15:fp16`

### Name Resolution

Bare names auto-resolve: `flux-dev` -> `flux-dev:q8`, `sdxl-base` -> `sdxl-base:fp16`, `sd15` -> `sd15:fp16`

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
```

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOLD_HOST` | `http://localhost:7680` | Remote server URL |
| `MOLD_MODELS_DIR` | `~/.mold/models` | Model storage path |
| `MOLD_PORT` | `7680` | Server port |
| `MOLD_LOG` | `warn` | Log level (trace/debug/info/warn/error) |
| `MOLD_EAGER` | unset | Set `1` to keep all components loaded |
| `MOLD_T5_VARIANT` | `auto` | T5 encoder: auto/fp16/q8/q6/q5/q4/q3 |
| `MOLD_SCHEDULER` | unset | SD1.5/SDXL: ddim/euler-ancestral/uni-pc |
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
- `--eager` trades VRAM for speed (keeps encoders loaded between generations)
- Dimensions must be multiples of 16; total pixels capped at ~1.1 megapixels
- For img2img, large source images auto-resize to fit the megapixel limit
