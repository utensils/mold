# LTX Video 0.9.x

The legacy LTX Video checkpoints from [Lightricks](https://lightricks.com) use a
DiT (Diffusion Transformer) architecture with T5-XXL text encoding and a 3D
causal video VAE. They generate short video clips from text prompts. For the
complete family, including LTX-2, LTX-2.3, and LTX-2.5, see the
[LTX Video overview](./ltx2.md).

<video autoplay muted loop playsinline aria-label="Northern lights, LTX Video 0.9.6 distilled" src="/gallery/ltx-aurora.webm"></video>
_"Northern lights dancing over a frozen lake in Iceland, green and purple aurora ribbons reflected in the ice"_; **ltx-video-0.9.6-distilled:bf16**, 8 steps, 33 frames, seed 1234

<video autoplay muted loop playsinline aria-label="Jellyfish, LTX Video 0.9.6 distilled" src="/gallery/ltx-jellyfish.webm"></video>
_"Underwater footage of a jellyfish pulsing through deep blue water, bioluminescent glow, particles floating"_; **ltx-video-0.9.6-distilled:bf16**, 8 steps, 33 frames, seed 707

- **Developer**: [Lightricks](https://huggingface.co/Lightricks)
- **License**: LTXV Open Weights License (custom, revenue-gated at $10M)
- **HuggingFace**:
  [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- **Implementation**: mold's LTX-Video transformer, 3D causal VAE, and
  flow-match scheduler were ported from
  [FerrisMind/candle-video](https://github.com/FerrisMind/candle-video)
  (Copyright 2025 FerrisMind,
  [Apache License 2.0](https://github.com/FerrisMind/candle-video/blob/main/LICENSE)),
  itself a Rust port of Hugging Face
  [diffusers](https://github.com/huggingface/diffusers). Those files stay
  Apache-2.0 inside mold's MIT codebase; see the repository's
  `THIRD_PARTY_NOTICES.md`.

> **Note**: Video output defaults to MP4. Also supports GIF, WebP, and APNG
> via `--format`. (Builds compiled without the `mp4` feature fall back to
> APNG; release builds ship MP4.)
> Frame count must be 8n+1 (9, 17, 25, 33, 49, ...) due to the VAE's 8x
> temporal compression.

## Variants

| Model                                | Steps | Approx total pull | Notes                                                                |
| ------------------------------------ | ----- | ----------------- | -------------------------------------------------------------------- |
| `ltx-video-0.9.6:bf16`               | 40    | ~17.4 GB          | Higher-quality 2B path, 30 FPS defaults                              |
| `ltx-video-0.9.6-distilled:bf16`     | 8     | ~17.4 GB          | Fast default single-pass path                                        |
| `ltx-video-0.9.8-2b-distilled:bf16`  | 7+3   | ~17.8 GB          | 0.9.8 checkpoint plus spatial upscaler asset                         |
| `ltx-video-0.9.8-13b-dev:bf16`       | 30    | ~38.5 GB          | Highest-quality 13B multiscale dev path; 40 GB-class GPU, no offload |
| `ltx-video-0.9.8-13b-distilled:bf16` | 7+3   | ~38.5 GB          | Faster 13B checkpoint; 40 GB-class GPU, no offload                   |

The 0.9.8 variants require the published spatial upscaler asset. mold pulls and
tracks that file explicitly.

::: warning 13B BF16 tiers need a 40 GB-class GPU
The two 13B BF16 checkpoints keep the whole ~28.6 GB transformer resident on
the GPU (plus ~2 GB runtime headroom). This legacy family has no block-offload
path and is not getting one; it is superseded by [LTX-2](./ltx2.md), whose FP8
tiers run on 24 GB cards with adaptive offload. On a 24 GB card mold refuses the
13B tiers up front with an error naming these alternatives; use an LTX-2 model
or `ltx-video-0.9.8-2b-distilled:bf16` instead.
:::

These sizes are approximate full-download totals, including the shared T5
encoder, tokenizer, VAE, and the `0.9.8` spatial upscaler where applicable.

The `0.9.8` family now runs the full two-pass multiscale refinement path. mold
keeps the shared T5 assets in `shared/flux/...`, stores the `0.9.8` spatial
upscaler under `shared/LTX-Video/...`, and intentionally continues using the
compatible `LTX-Video-0.9.5` VAE source until the newer VAE layout is ported.

## Defaults

- **Resolution**: 1216x704
- **Frames**: 25
- **FPS**: 30
- **Default model**: `ltx-video-0.9.6-distilled:bf16`
- **Steps**: 8 on `0.9.6-distilled`, 40 on `0.9.6`, 7+3 on `0.9.8` distilled multiscale presets
- **Output format**: MP4 (APNG fallback on builds without the `mp4` feature)

## Output Formats

| Format | Flag                     | Quality    | Metadata          | Notes                                                       |
| ------ | ------------------------ | ---------- | ----------------- | ----------------------------------------------------------- |
| MP4    | `--format mp4` (default) | H.264      | No                | Requires `mp4` feature                                      |
| APNG   | `--format apng`          | Lossless   | Yes (tEXt chunks) | Opens as `.png` everywhere; default on builds without `mp4` |
| GIF    | `--format gif`           | 256 colors | No                | Pipe-friendly                                               |
| WebP   | `--format webp`          | Lossy      | No                | Requires `webp` feature                                     |

## Recommended Dimensions

| Width | Height | Aspect Ratio         |
| ----- | ------ | -------------------- |
| 1216  | 704    | current mold default |
| 1024  | 576    | 16:9                 |
| 768   | 512    | 3:2                  |
| 512   | 768    | 2:3 (portrait)       |
| 512   | 512    | 1:1 (square)         |

Dimensions must be multiples of 32. Frame count must be 8n+1.

## Architecture

LTX Video uses a 3-stage sequential pipeline:

1. **T5-XXL text encoder** (shared with FLUX): encodes the prompt into
   4096-dim embeddings
2. **LTXVideoTransformer3DModel**: 28-layer DiT with 3D rotary position
   embeddings, self-attention + cross-attention, flow matching denoising
3. **3D Causal Video VAE**: decodes latents to video frames with 32x spatial
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
`ltx-video-0.9.6-distilled:bf16`. If you want the newer upstream 0.9.8
checkpoint family with full multiscale refinement, try
`ltx-video-0.9.8-2b-distilled:bf16`.
