# Generating Images

## Basic Usage

```bash
# Default model (flux-schnell:q8 — fast 4-step)
mold run "a red apple on a wooden table"

# Explicit model selection
mold run flux-dev:q4 "a photorealistic landscape at golden hour"

# Custom output path
mold run "cyberpunk cityscape" -o cityscape.png

# Reproducible output (same seed = same image)
mold run "a cat" --seed 42

# Custom dimensions (multiples of 16)
mold run "a banner" --width 1024 --height 512

# Batch generation (multiple images with incrementing seeds)
mold run "abstract art" --batch 4 --seed 100

# JPEG output
mold run "a sunset" --format jpeg -o sunset.jpg
```

## Negative Prompts

Guide what the model should avoid. Works with CFG-based models (SD1.5, SDXL,
SD3, Wuerstchen); ignored by FLUX and other flow-matching models.

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

- FLUX models only (BF16 or GGUF quantized)
- `.safetensors` format (diffusers-format keys)
- BF16 models on 24GB cards auto-use block-level offloading :::

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
echo "a cat riding a motorcycle" | mold run flux-schnell

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
