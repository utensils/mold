# LTX Video

A text-to-video generation model from [Lightricks](https://lightricks.com),
based on a DiT (Diffusion Transformer) architecture with T5-XXL text encoding
and a 3D causal video VAE. Generates short video clips from text prompts.

- **Developer**: [Lightricks](https://huggingface.co/Lightricks)
- **License**: LTXV Open Weights License (custom, revenue-gated at $10M)
- **HuggingFace**:
  [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)

> **Note**: This is mold's first video model. Output is encoded as animated GIF.
> Frame count must be 8n+1 (9, 17, 25, 33, ...) due to the VAE's 8x temporal
> compression.

## Variants

| Model                       | Steps | Size    | Notes                     |
| --------------------------- | ----- | ------- | ------------------------- |
| `ltx-video-0.9.5:bf16`     | 40    | ~7.6 GB | BF16 safetensors pipeline |

GGUF quantized transformer variants (Q3-Q8) exist on HuggingFace via
[city96/LTX-Video-0.9.5-gguf](https://huggingface.co/city96/LTX-Video-0.9.5-gguf)
but are not yet supported.

## Defaults

- **Resolution**: 768x512
- **Frames**: 25 (3.1 seconds at 24 fps)
- **FPS**: 24
- **Guidance**: 3.0
- **Steps**: 40

## Recommended Dimensions

| Width | Height | Aspect Ratio   |
| ----- | ------ | -------------- |
| 768   | 512    | 3:2 (default)  |
| 512   | 768    | 2:3 (portrait) |
| 512   | 512    | 1:1 (square)   |

Dimensions must be multiples of 32. Frame count must be 8n+1.

## Architecture

LTX Video uses a unique 3-stage pipeline:

1. **T5-XXL text encoder** (same as FLUX) — encodes the prompt into 4096-dim
   embeddings
2. **LTXVideoTransformer3DModel** — 28-layer DiT with 3D rotary position
   embeddings, self-attention + cross-attention, flow matching denoising
3. **3D Causal Video VAE** — decodes latents to video frames with 32x spatial
   and 8x temporal compression (128 latent channels)

The T5-XXL encoder is shared with FLUX models via mold's shared component
cache, so downloading LTX Video reuses the encoder if you already have FLUX.

## Notes

Generation produces an animated GIF file. For higher quality output, pipe
through ffmpeg to convert to MP4. VRAM usage is significant — the sequential
load-use-drop pipeline keeps peak VRAM manageable on 24GB cards.

The 3D causal VAE simulates 3D convolutions via temporal slicing of 2D
convolutions, which is computationally expensive. VAE decode time scales
with frame count.

Features not yet supported: img2vid, GGUF quantized transformer, vid2vid,
ControlNet conditioning, batch generation.

## Example

**LTX Video 0.9.5 BF16** — 40 steps, 25 frames, seed 42:

```bash
mold run ltx-video-0.9.5:bf16 "A cat walking across a sunlit windowsill, warm afternoon light" --frames 25 --seed 42
```
