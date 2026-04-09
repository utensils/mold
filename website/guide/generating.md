# Generating Images

## Basic Usage

```bash
# Default model (flux2-klein:q8 — fast 4-step, Apache 2.0, fully ungated)
mold run "a red apple on a wooden table"

# Explicit model selection
mold run flux-dev:q4 "a photorealistic landscape at golden hour"

# Custom output path
mold run "cyberpunk cityscape" -o cityscape.png

# Reproducible output (same seed = same image)
mold run "a cat" --seed 42

# Custom dimensions (multiples of 16)
# See each model's recommended dimensions to avoid warnings
mold run "a banner" --width 1024 --height 512

# Batch generation (multiple images with incrementing seeds)
mold run "abstract art" --batch 4 --seed 100

# JPEG output
mold run "a sunset" --format jpeg -o sunset.jpg
```

Here's what that looks like — FLUX Schnell Q8, seed 42:

![Snow leopard — FLUX Schnell](/gallery/flux-schnell-leopard.png)

Need a quick answer on family capabilities or expected speed? See
[Feature Support](/guide/feature-matrix) and [Performance](/guide/performance).

## Recommended Dimensions

Each model family has a set of recommended dimensions that produce the best
results. Using non-recommended dimensions will trigger a warning (generation
still proceeds). All dimensions must be multiples of 16.

```bash
# Square (works with all families)
mold run "a cat" --width 1024 --height 1024

# Landscape (FLUX, Flux.2 Klein)
mold run flux2-klein "a panorama" --width 1024 --height 576

# Portrait (SDXL, SD 3.5, Qwen-Image)
mold run sdxl-turbo "a portrait" --width 832 --height 1216
```

See each [model family page](/models/) for the full list of recommended
dimensions and aspect ratios.

## Image Editing

`qwen-image-edit-2511:*` is a distinct edit family, not standard img2img. It
uses one or more ordered `--image` inputs, supports negative prompts, and
derives default output dimensions from the first input image when you omit
`--width` and `--height`.

```bash
# Single-image edit
mold run qwen-image-edit-2511:q4 \
  --image ./chair.png \
  "turn this fabric chair into dark red leather"

# Multi-image edit
mold run qwen-image-edit-2511:q4 \
  --image ./chair.png \
  --image ./swatch.png \
  "make Picture 1 match the leather color and finish from Picture 2"
```

Use regular img2img families when you need `--strength`-based denoising.
Use `qwen-image-edit` when you want instruction-following edits against one or
more reference images.

## Video Generation

mold supports text-to-video generation with the LTX Video model family. Video
output defaults to APNG, with GIF, WebP, and MP4 also supported.

```bash
# Generate a 25-frame video clip with the fast distilled path
mold run ltx-video-0.9.6-distilled:bf16 "A cat walking across a sunlit windowsill"

# Custom frame count (must be 8n+1: 9, 17, 25, 33, 49, 97, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "Ocean waves at sunset" --frames 33

# Custom FPS (current LTX defaults use 30 FPS)
mold run ltx-video-0.9.6:bf16 "A timelapse of clouds" --frames 49 --fps 30

# Pipe to a video player
mold run ltx-video-0.9.6-distilled:bf16 "A robot dancing" | mpv -

# Direct MP4 output
mold run ltx-video-0.9.6-distilled:bf16 "A waterfall" --format mp4 -o waterfall.mp4
```

`ltx-video-0.9.6-distilled:bf16` is the recommended default today. The
`0.9.8` family is also supported end to end: mold pulls the required spatial
upscaler asset, runs the full multiscale refinement path, and keeps the current
compatible VAE on the published `LTX-Video-0.9.5` source until the newer VAE
layout is ported.

::: tip Frame count constraint
LTX Video requires frame counts of the form **8n+1** (9, 17, 25, 33, 49, 97,
etc.) due to the VAE's 8x temporal compression. mold will reject invalid counts
with a helpful error message.
:::

::: warning VRAM usage
LTX Video uses sequential load-use-drop to manage VRAM: T5 encoder loads first,
then drops before the transformer loads, then the transformer drops before VAE
decode. Peak VRAM depends heavily on the selected LTX checkpoint.
:::

Video dimensions must be multiples of 32 (not 16 like images). Current LTX
defaults use 1216×704 at 30 FPS.

## Joint Audio-Video Generation

LTX-2 / LTX-2.3 is exposed as a separate `ltx2` family. Unlike `ltx-video`,
its default container is MP4 and it can keep a synchronized audio track when
the request stays in MP4.

```bash
# Text-to-audio+video
mold run ltx-2-19b-distilled:fp8 \
  "a toy train rolling through a snowy diorama, gentle mechanical hum" \
  --frames 97 \
  --format mp4

# Audio-to-video
mold run ltx-2-19b-distilled:fp8 \
  "abstract paper sculpture reacting to a cello performance" \
  --audio-file ./cello.wav

# Keyframe interpolation
mold run ltx-2-19b-distilled:fp8 \
  "a drone shot over a canyon river" \
  --pipeline keyframe \
  --frames 97 \
  --keyframe 0:./start.png \
  --keyframe 96:./end.png
```

LTX-2 also adds:

- `--audio` / `--no-audio`
- `--audio-file`
- `--video`
- repeatable `--keyframe <frame:path>`
- `--pipeline one-stage|two-stage|two-stage-hq|distilled|ic-lora|keyframe|a2vid|retake`
- `--retake <start:end>`
- repeatable `--lora`
- `--camera-control <preset-or-path>`

::: warning Runtime requirements
The current LTX-2 implementation uses the upstream Lightricks Python pipelines
through a bridge. Local runs require `python3`, `uv`, `ffmpeg`, and the
upstream checkout at `tmp/LTX-2-upstream`.
:::

## Negative Prompts

Guide what the model should avoid. Works with CFG-based models (SD1.5, SDXL,
SD3, Wuerstchen, Qwen-Image, Qwen-Image-Edit); ignored by FLUX, Z-Image, and
Flux.2 Klein.

```bash
mold run sd15:fp16 "a portrait" -n "blurry, watermark, ugly, bad anatomy"
mold run sdxl:fp16 "a landscape" --negative-prompt "low quality, jpeg artifacts"

# Suppress config default
mold run sd15:fp16 "a cat" --no-negative
```

Precedence: CLI `--negative-prompt` > per-model config > global config > empty.

## Scheduler Selection

Choose the noise scheduler for SD1.5/SDXL models:

```bash
mold run sd15:fp16 "a cat" --scheduler uni-pc         # Fast convergence
mold run sd15:fp16 "a cat" --scheduler euler-ancestral # Stochastic
```

## LoRA Adapters (FLUX)

Apply fine-tuned style adapters to FLUX models:

```bash
# Basic LoRA
mold run flux-dev:bf16 "a portrait" --lora style.safetensors

# Adjust strength (0.0 = no effect, 1.0 = full, up to 2.0)
mold run flux-dev:bf16 "anime style" --lora style.safetensors --lora-scale 0.7

# Works with quantized models too
mold run flux-dev:q4 "a portrait" --lora style.safetensors --lora-scale 0.8
```

::: tip LoRA requirements
FLUX models only (BF16 or GGUF quantized). Requires `.safetensors` format with diffusers-format keys. BF16 models on 24GB cards auto-use block-level offloading.
:::

## Inline Preview

Display generated images in the terminal:

```bash
mold run "a cat" --preview
```

Requires the `preview` feature at build time. Auto-detects Kitty graphics,
iTerm2, Sixel, or Unicode half-block fallback.

Set `MOLD_PREVIEW=1` to enable permanently.

## PNG Metadata

Generated PNGs embed prompt, model, seed, size, steps, and a `mold:parameters`
JSON chunk by default. Disable with:

```bash
mold run "a cat" --no-metadata
# or globally
MOLD_EMBED_METADATA=0 mold run "a cat"
```

## Piping

mold is pipe-friendly in both directions. When stdout is not a terminal, raw
image bytes go to stdout and status goes to stderr.

```bash
# Pipe output to an image viewer
mold run "neon cityscape" | viu -

# Pipe prompt from stdin
echo "a cat riding a motorcycle" | mold run flux2-klein

# Full pipeline
echo "cyberpunk samurai" | mold run flux-dev:q4 | viu -

# Force stdout in interactive mode
mold run "a cat" --output -
```

## Inference Modes

1. **Remote** (default) — connects to `mold serve` via HTTP
2. **Local fallback** — if server unreachable, auto-falls back to local GPU
3. **Local forced** (`--local`) — skip server, run on local GPU directly

Models auto-pull if not downloaded.
