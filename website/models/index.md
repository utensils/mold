# Models

mold supports 9 model families spanning different architectures, quality levels,
and VRAM requirements — including both image and video generation.

## Choosing a Model

| Need              | Recommended                     | Why                            |
| ----------------- | ------------------------------- | ------------------------------ |
| Fast iterations   | `flux2-klein:q8`                | 4 steps, ungated, Apache 2.0   |
| Best quality      | `flux-dev:q4`                   | 25 steps, excellent detail     |
| Low VRAM (<8 GB)  | `flux2-klein:q4`                | 2.6 GB, 4 steps                |
| Classic ecosystem | `sd15:fp16` or `dreamshaper-v8` | Huge model library, ControlNet |
| Fast + great      | `z-image-turbo:q8`              | 9 steps, excellent quality     |
| SDXL              | `sdxl-turbo:fp16`               | 4 steps, 1024x1024             |
| **Video**         | `ltx-video-0.9.5:bf16`         | Text-to-video, 24fps, GIF out  |

## VRAM Guide

| Model              | Variant | Approx. VRAM | Speed              | Quality                      |
| ------------------ | ------- | ------------ | ------------------ | ---------------------------- |
| `flux-schnell:q8`  | Q8      | ~12 GB       | Fast, 4 steps      | Good                         |
| `flux-schnell:q6`  | Q6      | ~14 GB       | Fast, 4 steps      | Better than Q8               |
| `flux-dev:q4`      | Q4      | ~8 GB        | Slow, 25 steps     | Excellent                    |
| `flux-dev:q6`      | Q6      | ~10 GB       | Slow, 25 steps     | Best FLUX quality/size trade |
| `flux-dev:bf16`    | BF16    | ~24 GB       | Slow, 25 steps     | Best FLUX quality            |
| `flux2-klein:q4`   | Q4      | ~4 GB        | Fast, 4 steps      | Good for very small GPUs     |
| `z-image-turbo:q8` | Q8      | ~10 GB       | Fast, 9 steps      | Excellent                    |
| `sdxl-turbo:fp16`  | FP16    | ~10 GB       | Very fast, 4 steps | Good                         |
| `sd15:fp16`        | FP16    | ~6 GB        | Medium, 25 steps   | Good, broad ecosystem        |
| `qwen-image:q4`    | Q4      | ~14 GB       | Slow, 50 steps     | Strong                       |
| `ltx-video:bf16`   | BF16    | ~12 GB       | Slow, 40 steps     | Video (25 frames @ 24fps)    |

If you are close to your card limit, start with a smaller quantization or use
`--offload`. Full BF16 FLUX can run on 24 GB cards, but offloading may kick in
automatically and slow generation down.

<div class="gallery-grid">

![Flux.2 Klein — 4 steps](/gallery/flux2-klein-owl.png)

![FLUX Schnell — 4 steps](/gallery/flux-schnell-leopard.png)

![FLUX Dev Q4 — 25 steps](/gallery/flux-dev-teahouse.png)

![Z-Image Turbo — 9 steps](/gallery/zimage-astronaut.png)

![SD 3.5 Large — 28 steps](/gallery/sd35-clocktower.png)

![SDXL Turbo — 4 steps](/gallery/sdxl-turbo-market.png)

![DreamShaper v8 — 25 steps](/gallery/sd15-castle.png)

</div>

## Model Management

```bash
mold pull flux2-klein:q8     # Download a model
mold list                    # See what you have
mold info                    # Installation overview
mold info flux-dev:q4        # Model details + disk usage
mold rm dreamshaper-v8       # Remove a model
mold default flux-dev:q4     # Set default model
```

## Name Resolution

Bare names auto-resolve by trying `:q8` → `:fp16` → `:bf16` → `:fp8`:

```bash
mold run flux2-klein "a cat"   # resolves to flux2-klein:q8
mold run sdxl-base "a cat"     # resolves to sdxl-base:fp16
```

## HuggingFace Auth

Some model repos require authentication:

```bash
export HF_TOKEN=hf_...
mold pull flux-dev:q4
```

## All Families

| Family                           | Native Resolution | Architecture                   |
| -------------------------------- | ----------------- | ------------------------------ |
| [FLUX.2](/models/flux2)          | 1024x1024         | Qwen3 encoder, 4B transformer  |
| [FLUX.1](/models/flux)           | 1024x1024         | Flow-matching transformer      |
| [SDXL](/models/sdxl)             | 1024x1024         | Dual-CLIP, UNet                |
| [SD 1.5](/models/sd15)           | 512x512           | CLIP-L, UNet                   |
| [SD 3.5](/models/sd35)           | 1024x1024         | Triple encoder, MMDiT          |
| [Z-Image](/models/z-image)       | 1024x1024         | Qwen3 encoder, 3D RoPE         |
| [Wuerstchen](/models/wuerstchen) | 1024x1024         | 3-stage cascade, 42x compress  |
| [Qwen-Image](/models/qwen-image) | 1024x1024         | Qwen2.5-VL, flow-matching, CFG |
| [LTX Video](/models/ltx-video)   | 768x512           | T5-XXL, DiT, 3D causal VAE     |

Each family page lists recommended dimensions for non-square aspect ratios.
Using non-recommended dimensions will trigger a warning.
