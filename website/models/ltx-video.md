# LTX Video

A text-to-video generation model from [Lightricks](https://lightricks.com),
based on a DiT (Diffusion Transformer) architecture with T5-XXL text encoding
and a 3D causal video VAE. Generates short video clips from text prompts.

- **Developer**: [Lightricks](https://huggingface.co/Lightricks)
- **License**: LTXV Open Weights License (custom, revenue-gated at $10M)
- **HuggingFace**:
  [Lightricks/LTX-Video-0.9.5](https://huggingface.co/Lightricks/LTX-Video-0.9.5)

> **Note**: Video output defaults to APNG format (lossless, with embedded
> metadata). Also supports GIF, WebP, and MP4 via `--format`.
> Frame count must be 8n+1 (9, 17, 25, 33, 49, ...) due to the VAE's 8x
> temporal compression.

## Variants

| Model                   | Steps | Size    | Notes                                    |
| ----------------------- | ----- | ------- | ---------------------------------------- |
| `ltx-video-0.9.5:bf16`  | 40    | ~6.3 GB | v0.9.5 transformer + 1024-ch VAE (sharp) |
| `ltx-video-0.9:bf16`    | 40    | ~9.4 GB | v0.9 transformer + 512-ch VAE (legacy)   |

GGUF quantized transformer variants (Q3-Q8) exist on HuggingFace via
[city96/LTX-Video-0.9.5-gguf](https://huggingface.co/city96/LTX-Video-0.9.5-gguf)
but are not yet supported.

## Defaults

- **Resolution**: 768x512
- **Frames**: 25 (approximately 1 second at 24 fps)
- **FPS**: 24
- **Guidance**: 3.0
- **Steps**: 40
- **Output format**: APNG (animated PNG with metadata)

## Output Formats

| Format | Flag | Quality | Metadata | Notes |
| ------ | ---- | ------- | -------- | ----- |
| APNG | `--format apng` (default) | Lossless | Yes (tEXt chunks) | Opens as `.png` everywhere |
| GIF | `--format gif` | 256 colors | No | Pipe-friendly |
| WebP | `--format webp` | Lossy | No | Requires `webp` feature |
| MP4 | `--format mp4` | H.264 | No | Requires `mp4` feature |

## Recommended Dimensions

| Width | Height | Aspect Ratio   |
| ----- | ------ | -------------- |
| 768   | 512    | 3:2 (default)  |
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

The sequential pipeline keeps peak VRAM manageable on 24GB cards:

- T5-XXL FP16: ~10 GB (dropped after encoding)
- Transformer BF16: ~3.8 GB (dropped after denoising)
- VAE v0.9.5: ~2.5 GB (dropped after decoding)

## Example

```bash
# Default: 25 frames, APNG output
mold run ltx-video-0.9.5:bf16 "A cat walking across a sunlit windowsill" --frames 25

# Higher quality with more steps
mold run ltx-video-0.9.5:bf16 "waves crashing on a rocky coastline at sunset" --frames 17 --steps 50

# GIF output for piping
mold run ltx-video-0.9.5:bf16 "a campfire at night" --format gif | mpv -

# Longer video (49 frames ≈ 2 seconds)
mold run ltx-video-0.9.5:bf16 "a humanoid robot walking" --frames 49 --steps 40
```
