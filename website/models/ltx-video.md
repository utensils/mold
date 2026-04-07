# LTX Video

A text-to-video generation model from [Lightricks](https://lightricks.com),
based on a DiT (Diffusion Transformer) architecture with T5-XXL text encoding
and a 3D causal video VAE. Generates short video clips from text prompts.

- **Developer**: [Lightricks](https://huggingface.co/Lightricks)
- **License**: LTXV Open Weights License (custom, revenue-gated at $10M)
- **HuggingFace**:
  [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)

> **Note**: Video output defaults to APNG format (lossless, with embedded
> metadata). Also supports GIF, WebP, and MP4 via `--format`.
> Frame count must be 8n+1 (9, 17, 25, 33, 49, ...) due to the VAE's 8x
> temporal compression.

## Variants

| Model | Steps | Approx total pull | Notes |
| ----- | ----- | ----------------- | ----- |
| `ltx-video-0.9.6:bf16` | 40 | ~17.4 GB | Higher-quality 2B path, 30 FPS defaults |
| `ltx-video-0.9.6-distilled:bf16` | 8 | ~17.4 GB | Fast default single-pass path |
| `ltx-video-0.9.8-2b-distilled:bf16` | 7 | ~17.8 GB | 0.9.8 checkpoint plus spatial upscaler asset |
| `ltx-video-0.9.8-13b-dev:bf16` | 30 | ~38.5 GB | Highest-quality 13B checkpoint |
| `ltx-video-0.9.8-13b-distilled:bf16` | 7 | ~38.5 GB | Faster 13B checkpoint |

The 0.9.8 variants require the published spatial upscaler asset. mold pulls and
tracks that file explicitly.

These sizes are approximate full-download totals, including the shared T5
encoder, tokenizer, VAE, and the `0.9.8` spatial upscaler where applicable.

Today, mold runs the `0.9.8` first pass correctly and resolves the upscaler
asset, but it does not yet execute the second multiscale refinement pass. That
means `0.9.8` is usable and materially better wired than before, but still not
at full upstream quality parity.

## Defaults

- **Resolution**: 1216x704
- **Frames**: 25
- **FPS**: 30
- **Default model**: `ltx-video-0.9.6-distilled:bf16`
- **Steps**: 8 on distilled models, 40 on `0.9.6`, 7 on current `0.9.8` first-pass presets
- **Output format**: APNG (animated PNG with metadata)

## Output Formats

| Format | Flag                      | Quality    | Metadata          | Notes                      |
| ------ | ------------------------- | ---------- | ----------------- | -------------------------- |
| APNG   | `--format apng` (default) | Lossless   | Yes (tEXt chunks) | Opens as `.png` everywhere |
| GIF    | `--format gif`            | 256 colors | No                | Pipe-friendly              |
| WebP   | `--format webp`           | Lossy      | No                | Requires `webp` feature    |
| MP4    | `--format mp4`            | H.264      | No                | Requires `mp4` feature     |

## Recommended Dimensions

| Width | Height | Aspect Ratio   |
| ----- | ------ | -------------- |
| 1216  | 704    | current mold default |
| 1024  | 576    | 16:9           |
| 768   | 512    | 3:2            |
| 512   | 768    | 2:3 (portrait) |
| 512   | 512    | 1:1 (square)   |

Dimensions must be multiples of 32. Frame count must be 8n+1.

## Architecture

LTX Video uses a 3-stage sequential pipeline:

1. **T5-XXL text encoder** (shared with FLUX) — encodes the prompt into
   4096-dim embeddings
2. **LTXVideoTransformer3DModel** — 28-layer DiT with 3D rotary position
   embeddings, self-attention + cross-attention, flow matching denoising
3. **3D Causal Video VAE** — decodes latents to video frames with 32x spatial
   and 8x temporal compression (128 latent channels)

Each component is loaded, used, then dropped to free VRAM for the next stage.
The T5-XXL encoder is shared with FLUX via mold's shared component cache.

## VRAM Usage

The sequential pipeline keeps peak VRAM manageable on 24GB cards for the 2B
checkpoints:

- T5-XXL FP16: ~10 GB (dropped after encoding)
- Transformer BF16: model-dependent; 2B fits comfortably, 13B requires much more VRAM
- VAE: ~2.5 GB (dropped after decoding)

## Example

```bash
# Fast default path
mold run ltx-video-0.9.6-distilled:bf16 "A cat walking across a sunlit windowsill" --frames 25

# Higher-quality 2B path
mold run ltx-video-0.9.6:bf16 "waves crashing on a rocky coastline at sunset" --frames 17 --steps 40

# GIF output for piping
mold run ltx-video-0.9.6-distilled:bf16 "a campfire at night" --format gif | mpv -

# 0.9.8 checkpoint family
mold run ltx-video-0.9.8-2b-distilled:bf16 "a humanoid robot walking" --frames 49
```

If you want the safest current quality path in mold, start with
`ltx-video-0.9.6-distilled:bf16`. If you want to evaluate the newer checkpoint
family and are comfortable with the current first-pass-only limitation, try
`ltx-video-0.9.8-2b-distilled:bf16`.
