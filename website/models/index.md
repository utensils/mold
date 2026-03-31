# Models

mold supports 8 model families spanning different architectures, quality levels,
and VRAM requirements.

## Choosing a Model

| Need              | Recommended                     | Why                            |
| ----------------- | ------------------------------- | ------------------------------ |
| Fast iterations   | `flux-schnell:q8`               | 4 steps, ~10s on RTX 4090      |
| Best quality      | `flux-dev:q4`                   | 25 steps, excellent detail     |
| Low VRAM (<8 GB)  | `flux2-klein:q4`                | 2.6 GB, 4 steps                |
| Classic ecosystem | `sd15:fp16` or `dreamshaper-v8` | Huge model library, ControlNet |
| Fast + great      | `z-image-turbo:q8`              | 9 steps, excellent quality     |
| SDXL              | `sdxl-turbo:fp16`               | 4 steps, 1024x1024             |

## Model Management

```bash
mold pull flux-schnell:q8   # Download a model
mold list                    # See what you have
mold info                    # Installation overview
mold info flux-dev:q4        # Model details + disk usage
mold rm dreamshaper-v8       # Remove a model
mold default flux-dev:q4     # Set default model
```

## Name Resolution

Bare names auto-resolve by trying `:q8` → `:fp16` → `:bf16` → `:fp8`:

```bash
mold run flux-schnell "a cat"  # resolves to flux-schnell:q8
mold run sdxl-base "a cat"     # resolves to sdxl-base:fp16
```

## HuggingFace Auth

Some model repos require authentication:

```bash
export HF_TOKEN=hf_...
mold pull flux-dev:q4
```

## All Families

- [FLUX.1](/models/flux) — best quality, flow-matching transformer
- [SDXL](/models/sdxl) — fast and flexible, dual-CLIP
- [SD 1.5](/models/sd15) — lightweight, huge ecosystem
- [SD 3.5](/models/sd35) — triple encoder, MMDiT
- [Z-Image](/models/z-image) — Qwen3 text encoder, 3D RoPE
- [Flux.2 Klein](/models/flux2) — lightweight 4B FLUX variant
- [Wuerstchen](/models/wuerstchen) — 3-stage cascade, 42x compression
- [Qwen-Image](/models/qwen-image) — alpha, flow-matching with CFG
