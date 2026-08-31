# Models

Mold supports image and video model families across a range of quality levels,
hardware requirements, and creative workflows.

::: tip Community models from Civitai
The models documented here are not the limit. Open **Models → Discover** in
Mold Studio and choose **Civitai** (or **All**) to browse and install compatible
community checkpoints and LoRAs. See the
[Model Discovery Catalog](/docs/catalog) for details.
:::

## Choosing a Model

| Need                | Recommended                           | Why                                                   |
| ------------------- | ------------------------------------- | ----------------------------------------------------- |
| Fast iterations     | `flux2-klein:q8`                      | 4 steps, ungated, Apache 2.0                          |
| Best quality        | `flux-dev:q4`                         | 25 steps, excellent detail                            |
| Smallest checkpoint | `flux2-klein:q4`                      | 2.6 GB transformer, 4 steps                           |
| Classic ecosystem   | `sd15:fp16` or `dreamshaper-v8`       | Huge model library, ControlNet                        |
| Fast + great        | `z-image-turbo:q8`                    | 9 steps, excellent quality                            |
| SDXL                | `sdxl-turbo:fp16`                     | 4 steps, 512x512 default (1024x1024 presets)          |
| **LTX Video**       | `ltx-video-0.9.6-distilled:bf16`      | Broad text-to-video default; LTX-2.x adds joint audio |
| **Wan video**       | `wan22-ti2v-5b:q8`                    | Text/image-to-video with broad Wan workflow support   |
| **Reference AV**    | `minimax-h3-ref2va:comfy-pruned-int8` | Ordered image/video/audio references; SM89 CUDA build |

## Choosing a Video Family

| Family                           | Start with                           | Best for                                              | Important boundary                                       |
| -------------------------------- | ------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------- |
| [LTX Video](/models/ltx2)        | `ltx-video-0.9.6-distilled:bf16`     | Text-to-video plus joint audio-video on LTX-2.x       | Legacy models have no generated audio; newer packs gated |
| [Wan Video](/models/wan)         | `wan22-ti2v-5b:q8`                   | Text/image-to-video, first/last frames, and sequences | CUDA-qualified; Metal/CPU correctness-only               |
| [MiniMax H3](/models/minimax-h3) | `minimax-h3-fl2va:comfy-pruned-int8` | First-frame or ordered-reference audio-video          | 42.482 GB base pull; runtime is build-scoped             |

MiniMax H3's pull size is disk/download size, not peak VRAM. Its compact
runtime is available on H3-enabled SM89 CUDA builds; Metal is shipped as an
unqualified correctness-only route, and CPU is unsupported. Check the model
row's `runtime_available` reason before downloading on another target.

## Image VRAM Guide

These estimates include the transformer, text encoder(s), VAE, and ~2 GB
activation headroom. The **default** column is sequential mode (drop-and-reload),
which loads components one at a time. **Eager** mode keeps everything on GPU
simultaneously for faster inference but needs more VRAM.

| Model                | Variant | Default VRAM | Eager VRAM | Speed              | Quality                      |
| -------------------- | ------- | ------------ | ---------- | ------------------ | ---------------------------- |
| `flux-schnell:q8`    | Q8      | ~15 GB       | ~25 GB     | Fast, 4 steps      | Good                         |
| `flux-dev:q4`        | Q4      | ~10 GB       | ~15 GB     | Slow, 25 steps     | Excellent                    |
| `flux-dev:q6`        | Q6      | ~12 GB       | ~20 GB     | Slow, 25 steps     | Best FLUX quality/size trade |
| `flux-dev:bf16`      | BF16    | ~26 GB       | ~36 GB     | Slow, 25 steps     | Best FLUX quality            |
| `flux2-klein:q4`     | Q4      | ~5 GB        | ~11 GB     | Fast, 4 steps      | Good for very small GPUs     |
| `flux2-klein:q8`     | Q8      | ~6 GB        | ~13 GB     | Fast, 4 steps      | Good                         |
| `z-image-turbo:q8`   | Q8      | ~9 GB        | ~13 GB     | Fast, 9 steps      | Excellent                    |
| `sdxl-turbo:fp16`    | FP16    | ~8 GB        | ~11 GB     | Very fast, 4 steps | Good                         |
| `sd15:fp16`          | FP16    | ~6 GB        | ~6 GB      | Medium, 25 steps   | Good, broad ecosystem        |
| `sd3.5-large:q8`     | Q8      | ~12 GB       | ~22 GB     | Medium, 28 steps   | Excellent                    |
| `qwen-image:q4`      | Q4      | ~14 GB       | ~22 GB     | Slow, 50 steps     | Good, validated at 1024      |
| `qwen-image-2512:q4` | Q4      | ~14 GB       | ~22 GB     | Slow, 50 steps     | Good, validated at 1328      |
| `qwen-image:q8`      | Q8      | ~22 GB       | ~24+ GB    | Slow, 50 steps     | Best GGUF, validated at 768  |

::: tip Sequential vs Eager
In **sequential mode** (the default), mold loads each component (encoder →
transformer → VAE) one at a time, freeing GPU memory between phases. This
reduces peak VRAM by 30-50% but adds 10-20% to generation time.

Use `--eager` to keep all components loaded simultaneously for faster inference
on high-VRAM cards. FLUX.1, FLUX.2, Z-Image, Qwen-Image, SD 3.5, LTX-2, and Wan
also support `--offload` for block-level CPU↔GPU streaming (~24 GB down to
~2-4 GB peak, 3-5x slower).
:::

<div class="gallery-grid">

![Flux.2 Klein, 4 steps](/gallery/flux2-klein-owl.png)

![FLUX Schnell, 4 steps](/gallery/flux-schnell-leopard.png)

![FLUX Dev Q4, 25 steps](/gallery/flux-dev-teahouse.png)

![Z-Image Turbo, 9 steps](/gallery/zimage-astronaut.png)

![SD 3.5 Large, 28 steps](/gallery/sd35-clocktower.png)

![SDXL Turbo, 4 steps](/gallery/sdxl-turbo-market.png)

![DreamShaper v8, 25 steps](/gallery/sd15-castle.png)

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

Some model repos (marked `[gated]`) require a
[HuggingFace access token](https://huggingface.co/settings/tokens). You may
need to accept the model's license on its HuggingFace page before downloading.

**Option 1: Environment variable** (simplest):

```bash
export HF_TOKEN=hf_...
mold pull flux-dev:q4
```

**Option 2: HuggingFace CLI** (persists the token):

```bash
# Install the HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Log in (saves token to ~/.cache/huggingface/)
hf auth login
```

Once logged in, `mold pull` picks up the stored token automatically; no
`HF_TOKEN` export needed.

See the [HuggingFace CLI docs](https://huggingface.co/docs/huggingface_hub/guides/cli)
for more options.

## All Families

| Family                                | Native Resolution             | Architecture                                   |
| ------------------------------------- | ----------------------------- | ---------------------------------------------- |
| [FLUX.2](/models/flux2)               | 1024x1024                     | Klein Qwen3 or Dev Mistral3 transformer family |
| [FLUX.1](/models/flux)                | 1024x1024                     | Flow-matching transformer                      |
| [SDXL](/models/sdxl)                  | 1024x1024                     | Dual-CLIP, UNet                                |
| [SD 1.5](/models/sd15)                | 512x512                       | CLIP-L, UNet                                   |
| [SD 3.5](/models/sd35)                | 1024x1024                     | Triple encoder, MMDiT                          |
| [Z-Image](/models/z-image)            | 1024x1024                     | Qwen3 encoder, 3D RoPE                         |
| [Wuerstchen](/models/wuerstchen)      | 1024x1024                     | 3-stage cascade, 42x compress                  |
| [Qwen-Image](/models/qwen-image)      | 1328x1328                     | Qwen2.5-VL, flow-matching, CFG                 |
| [Qwen-Image-Edit](/models/qwen-image) | Derived from first edit image | Qwen2.5-VL multimodal edit, flow-matching, CFG |
| [LTX Video](/models/ltx2)             | 768x512 / 1216x704            | T5/Gemma, video transformers, causal VAEs      |
| [MiniMax H3](/models/minimax-h3)      | 1344x768                      | Qwen3-VL, joint audio-video DiT, dual VAEs     |
| [Wan Video](/models/wan)              | 832x480 / 1280x704            | UMT5-XXL, flow DiT, causal 3D VAE, A14B MoE    |
| [Upscalers](/models/upscalers)        | 2x / 4x source size           | Real-ESRGAN super-resolution                   |

Each family page lists its actual shape contract. Bucketed families may warn
when a request misses their recommended dimensions. MiniMax H3 instead accepts
any 32-aligned canvas inside its documented continuous area and aspect bounds.

Maintainers should use the
[model resolution and aspect-ratio matrix](https://github.com/utensils/mold/blob/main/docs/model-resolution-matrix.md)
for exact reduced ratios, checkpoint and pipeline exceptions, Mold admission
bounds, profile hashes, and pinned upstream provenance.

::: warning Backend qualification
All image families and `LTX Video` run on CUDA, Apple Metal, and CPU. LTX-2 /
LTX-2.3 is performance-qualified on CUDA and Apple Metal (measured on the 19B
and 22B distilled FP8 tiers; Metal is slower than a comparable CUDA card).
LTX-2.5's compact distilled INT8 ConvRot split pack is qualified on Apple
Metal; Q3_K_M, Q4_K_M, and Q6_K GGUF are also Metal-qualified, while its BF16
route remains operator-deferred. CUDA has a separate completed qualification
campaign. The LTX-2 family CPU path stays
correctness-oriented and can be extremely slow. Wan is
performance-qualified on CUDA; its CPU and Apple
Metal paths are correctness-oriented (fp8-scaled Wan checkpoints stay
CUDA-only; Metal has no fp8 widening kernel). MiniMax H3 compact checkpoints
are downloadable everywhere, and both reviewed routes (FL2VA's boundary frame
and Ref2VA's ordered image/video/audio references) execute on supported SM89
CUDA builds; the CPU path is unsupported, and the Apple Metal route is admitted
and shipped but correctness-only and not yet hardware-qualified.
:::
