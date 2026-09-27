# Qwen Image 2.1

Qwen Image 2.1 pairs a Qwen3-VL-8B conditioner with a 32-block
causal-condition flow transformer and a four-channel (RGBA) VAE. One model
covers text-to-image, instruction editing, multi-reference composition and
transparent-background renders. It is a separate family (`qwen-image21`) from
[Qwen-Image / 2512](/models/qwen-image): the conditioner, transformer and VAE
all differ, and nothing is shared between the two installs.

- **Developer**: [Alibaba / Qwen Team](https://huggingface.co/Qwen)
- **Licence**: Qwen Research License — **non-commercial** research and
  evaluation only (see [Licence](#licence))
- **Upstream**: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  pinned at revision `b3179ad`
- **Quantized transformers**:
  [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
  (INT8 ConvRot),
  [unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8),
  [leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF)
- **Turbo adapter**:
  [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
  (v0.2.1, 6-step, rank 256)

```bash
mold run qwen-image-2.1 'A tiny artisan bakery with a hand-painted sign reading "MOLD & FLOUR"'
mold run qwen-image-2.1-turbo "A lighthouse on a basalt cliff at dusk, oil painting" --seed 7
```

The bare name `qwen-image-2.1` resolves to `:bf16` (and `qwen-image-2.1-turbo`
to `qwen-image-2.1-turbo:bf16`); the usual `:q8`-first tag search would
otherwise pick a quantized tier.

## Tiers

Every tier shares one runtime: the Qwen3-VL-8B conditioner (BF16 shards, which
also hold the vision tower), the RGBA VAE and the processor tokenizer. These
are stored once under `shared/qwen-image21/` and reused by every tier, so a
second tier downloads only its transformer.

| Model                            | Transformer                   | Transformer size | Notes                                                         |
| -------------------------------- | ----------------------------- | ---------------- | ------------------------------------------------------------- |
| `qwen-image-2.1:bf16`            | Official BF16 shards          | 14.2 GB          | Reference quality; the bare-name default                      |
| `qwen-image-2.1:int8-conv`       | Comfy-Org INT8 ConvRot (W8A8) | 7.3 GB           | About the same speed as BF16 on CUDA at half the weights      |
| `qwen-image-2.1:fp8`             | unsloth row-scaled F8E4M3     | 7.1 GB           | Widened per forward                                           |
| `qwen-image-2.1:q8`              | GGUF Q8_0                     | 7.7 GB           |                                                               |
| `qwen-image-2.1:q6`              | GGUF Q6_K                     | 6.0 GB           |                                                               |
| `qwen-image-2.1:q5`              | GGUF Q5_0                     | 5.1 GB           |                                                               |
| `qwen-image-2.1:q4`              | GGUF Q4_K                     | 4.2 GB           |                                                               |
| `qwen-image-2.1:q3`              | GGUF Q3_K                     | 3.3 GB           | Coherent; fine lettering is softer                            |
| `qwen-image-2.1:q2`              | GGUF Q2_K                     | 2.6 GB           | Last resort for small cards: visibly degraded, text illegible |
| `qwen-image-2.1-turbo:bf16`      | BF16 + Viggle 6-step LoRA     | 14.2 + 1.4 GB    | See [Turbo](#turbo)                                           |
| `qwen-image-2.1-turbo:int8-conv` | INT8 ConvRot + Viggle LoRA    | 7.3 + 1.4 GB     |                                                               |
| `qwen-image-2.1-turbo:q8`        | GGUF Q8_0 + Viggle LoRA       | 7.7 + 1.4 GB     |                                                               |

The shared runtime adds about 18.9 GB of downloads (text encoder 17.5 GB, VAE
1.35 GB, tokenizer). The GGUF files come from stable-diffusion.cpp's author, so
the same bytes also run under that reference implementation. Every GGUF tier
checks each step's prediction for NaN or infinity and names the tier and
`MOLD_QWEN_IMAGE21_QMATMUL` if one appears.

### Text encoder

The conditioner's language model can run from the BF16 shards or from the
official [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
files, chosen by `MOLD_QWEN3_VARIANT`:

| Value            | Language model                          |
| ---------------- | --------------------------------------- |
| `auto` (default) | BF16 if it fits, otherwise Q8_0         |
| `bf16`           | BF16 shards (about 16.4 GB)             |
| `q8`             | Q8_0 GGUF (8.7 GB)                      |
| `q4`             | Q4_K_M GGUF (5.0 GB), **explicit only** |

Q4_K_M fails the hidden-state parity gate against BF16, so `auto` never picks
it; set `MOLD_QWEN3_VARIANT=q4` to accept that trade on a small card. The
vision tower is always read from the BF16 shards, whatever the language-model
variant.

### Which tier for which card

The transformer and text encoder dominate memory. BF16 needs about 14.8 GB for
the transformer alone; INT8 ConvRot needs 7.8 GB.

| VRAM    | Recommendation                                    | Measured peak at 1024²                 |
| ------- | ------------------------------------------------- | -------------------------------------- |
| ≥ 44 GB | `bf16`; everything stays resident at 1024²        | 37.3 GB (`int8-conv` 31.9, `q4` 25.5)  |
| 32 GB   | `bf16` with the Q8 text encoder (`auto` picks it) | 25.3 GB                                |
| 24 GB   | `int8-conv` (or `q8`) with the Q8 text encoder    | 18.7 GB (`q8` 19.1, `q4` 19.6)         |
| 16 GB   | `q4` with `MOLD_QWEN3_VARIANT=q4`, 1024² only     | 12.8 GB                                |
| ≤ 12 GB | `q3` or `q2`, sequential loading                  | 7.8 GB (`q2` 6.7); encoder runs on CPU |

The peaks are whole-process device memory sampled with `nvidia-smi` on an
NVIDIA L40S during a server render, with the smaller cards simulated by
`MOLD_RESERVE_VRAM_MB`. Denoise time is 14–20 s at 1024² on every tier; the
12 GB plan adds about 25 s of CPU text encoding.

At the 2K presets the encoder is parked in host RAM for the denoise, and the
transformer is released before the VAE decode when the card cannot hold both.
Mold makes that decision itself from the free memory; you do not need a flag.
It budgets each phase on its own — the prompt and reference encode, the
denoise with its prefix cache, the decode — because they never overlap, so on
a 46–48 GB card the encoder stays resident for a reference render whose
denoise leaves room for it (one to three references at 1024² without a
negative prompt). It still parks when both CFG branches retain a cache beside
several references, because a retained cache is worth far more than the park.

## Canvas and 2K presets

Canvases are multiples of 32. The default is 1024x1024. The profile lists the
model card's native 2K sizes and Mold's own ~1 MP presets at the same aspect
ratios:

| Aspect | ~1 MP (Mold preset) | Native 2K (model card) |
| ------ | ------------------- | ---------------------- |
| 1:1    | 1024x1024           | 2048x2048              |
| 4:3    | 1184x896            | 2400x1792              |
| 3:4    | 896x1184            | 1792x2400              |
| 3:2    | 1248x832            | 2528x1696              |
| 2:3    | 832x1248            | 1696x2528              |
| 16:9   | 1376x768            | 2752x1536              |
| 9:16   | 768x1376            | 1536x2752              |

The family's limits are 4,300,800 pixels (2400x1792) and a longest side of 2752. The standard recipe is 40 steps at guidance 1.0; setting guidance above
1 turns on classifier-free guidance and the negative prompt.

## Reference images

Up to **10** ordered reference images ride the request as `edit_images`
(`--image`, repeated, or `--reference`, never both). None of them is a special
source: one path serves editing and multi-reference composition alike, so the
prompt addresses them by position ("the jacket from image 1 on the person in
image 2").

```bash
mold run qwen-image-2.1 "Change the background to a sunset beach" --image portrait.jpg
mold run qwen-image-2.1 "Put the jacket from image 1 on the person in image 2" \
  --image jacket.png --image person.jpg
```

- **Formats**: PNG, JPEG or WebP. References are never flattened: an RGBA
  reference is read with its alpha.
- **Sizing**: each reference is resized to about 1024² pixels before encoding.
- **Canvas**: the profile advertises `canvas: last-reference`. With neither
  `--width` nor `--height`, the output takes the **last** reference's aspect
  ratio at the default area, on the 32 px grid. Any explicit size wins. This
  is a client rule; the server renders exactly the size in the request.
- **No source image**: references replace img2img, so `--strength`, `--mask`
  and ControlNet are not offered, and a `source_image` is refused with
  "uses edit_images instead of source_image".

## Transparent backgrounds

The **Transparent background** toggle (`--transparent`,
`transparent_background: true`) renders the subject as a cut-out with an alpha
channel.

```bash
mold run qwen-image-2.1 "A red paper lantern with a gold tassel" --transparent --format webp -o lantern.webp
mold run qwen-image-2.1 "Extract the lantern from image 1" --image street.webp --transparent -o lantern.png
```

- **Describe only the subject**, with no scenery, backdrop or floor.
- The engine wraps your prompt in the model card's RGBA recipe:
  `This is an RGBA image with transparency. <your prompt>. The image has alpha channel and the background is transparent.`
  The stored prompt, Reuse, Expand and the Library all keep your own words.
- Alpha needs **PNG** (the default) or **WebP**. JPEG with the toggle on is
  refused before anything loads.
- **When the output keeps alpha**: exactly when the toggle is on, or when at
  least one reference image has a transparent pixel. An ordinary text-to-image
  render is delivered as RGB, byte-identical to earlier releases. If a
  transparent reference meets a JPEG output, the picture is composited over
  white and the request carries a warning.
- The print records `has_alpha: true` when the stored file carries alpha, and
  the Library, lightbox and result canvas draw it over a checkerboard.

## LoRA

Every tier accepts LoRA adapters trained for **Qwen Image 2.1**. Adapters are
applied at forward time and never merged into the weights, so a LoRA works
the same way on BF16, INT8, FP8 and GGUF tiers.

::: warning Qwen-Image / 2512 LoRAs do not apply
Qwen Image 2.1 is a different architecture from Qwen-Image and 2512. An
adapter trained for those models is refused by name rather than loaded onto
the wrong layers, and the catalog offers only `qwen-image21` adapters for this
family.
:::

```bash
mold run qwen-image-2.1:int8-conv "A watercolour fox" --lora ./my-qwen21-style.safetensors --lora-scale 0.8
```

## Turbo

`qwen-image-2.1-turbo:<tag>` (tags `bf16`, `int8-conv`, `q8`) is the base
tier's files plus Viggle's 6-step distilled LoRA (Distribution Matching
Distillation). It stores its transformer under the base tier, so a host that
already has `qwen-image-2.1:q8` pulls only the 1.4 GB adapter for
`qwen-image-2.1-turbo:q8`.

The recipe is fixed by the model, not chosen: 6 steps on the raw sigmas
`[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, guidance 1.0 (no negative prompt),
no terminal shift, adapter scale 1.0. The profile pins those values and
hides the steps, guidance and negative-prompt controls; admission refuses a
request whose steps or guidance disagree. References, `--transparent` and
your own `--lora` all work on top of it.

Viggle reports the turbo is about 5x faster end to end and hard to tell apart
from the 40-step base on most prompts. The base model is still ahead on
small, dense text and on complicated edits (several references, face swaps,
identity-preserving changes).

## Licence

Every file of every Qwen Image 2.1 tier — base, quantized and turbo — is
covered by the **Qwen Research License Agreement** (`qwen-research`): use
is limited to non-commercial research and evaluation, and commercial use
needs a separate licence from Qwen. The Viggle turbo adapter is published
under the same agreement.

Mold refuses to download any tier until you have recorded an acceptance, so a
server-side auto-pull can never acquire the weights on your behalf. The apps
show the licence dialog before the first download; on the CLI:

```bash
mold licenses                              # read the terms
mold licenses accept qwen-research         # agree without downloading
mold pull qwen-image-2.1 --accept-license qwen-research
```

One acceptance covers every tier, turbo included.

## Performance

### CUDA

On CUDA, Qwen Image 2.1 takes the FastStill policy, like FLUX.1 and FLUX.2:
FlashAttention through a segment-aware dispatch, a fused projection and RoPE
kernel, cached F32 RoPE tables, compact modulation, fused adaLN, and cuDNN for
the VAE. Measured on an NVIDIA L40S, BF16, 40 steps, warm denoise:

| Case                        | v0.32  | Now    | Speedup |
| --------------------------- | ------ | ------ | ------- |
| 1024², guidance 1           | 38.9 s | 15.2 s | 2.6x    |
| 1344x768, guidance 4 + neg. | 76.7 s | 31.0 s | 2.5x    |
| 2048², guidance 1           | 525 s  | 82.6 s | 6.4x    |
| 2752x1536, guidance 1       | 537 s  | 85.9 s | 6.2x    |

`int8-conv` renders at about the BF16 speed (15.3 s at 1024²) with half the
transformer memory. Batching both classifier-free-guidance branches into one
forward was measured and made every size slower on this card, so the engine
keeps them sequential.

Reference renders on the same card, 1024² output, 40 steps, each reference a
1536x1024 image:

| References, guidance  | Prefix cache                | Denoise |
| --------------------- | --------------------------- | ------- |
| 3, guidance 4 + neg.  | kept (12.0 GiB)             | 44.2 s  |
| 10, guidance 1        | recomputed (needs 19.9 GiB) | 343.7 s |
| 10, guidance 4 + neg. | recomputed (needs 39.8 GiB) | 696.5 s |

The three-reference guided render took 167.4 s before the cache followed the
card's memory. Ten references do not leave room for their cache beside the
transformer on a 46 GB card, so they recompute their prefix every step and
say so in a request warning; forcing `MOLD_QWEN_IMAGE21_KV_CACHE=on` there is
refused by the planner (it would need about 61 GB).

The fast path changes pixels relative to v0.32 for the same seed. To
reproduce a v0.32 render byte for byte, start the generating server with
`MOLD_ATTN=math MOLD_CONV=im2col`.

Full measurements, including the 2K sweep and the VAE decode study, are in
[`docs/qualification/qwen-image-2.1-cuda-performance.md`](https://github.com/utensils/mold/blob/main/docs/qualification/qwen-image-2.1-cuda-performance.md).

### Apple Metal

Metal runs a BF16 denoiser with fused unmasked image attention and compact
cached-step operations; the text encoder and VAE stay F32.
`MOLD_QWEN_IMAGE21_DTYPE=f32` selects a full-precision denoiser, and adding
`MOLD_ATTN=math` restores the original Metal computation path. The 1024²
default is qualified on Metal
([`docs/qualification/qwen-image-2.1-metal-performance.md`](https://github.com/utensils/mold/blob/main/docs/qualification/qwen-image-2.1-metal-performance.md)).
The 2K presets, reference conditioning, transparency and turbo have been
verified on CUDA; Metal verification of those paths is tracked separately.

### Prefix cache

Each classifier-free-guidance branch can keep its text-and-reference prefix
K/V across steps instead of recomputing it, as upstream always does.
`MOLD_QWEN_IMAGE21_KV_CACHE` (`auto`, `on`, `off`) controls it.

- On the CUDA fast path, `auto` keeps every branch's cache whenever it fits
  in the memory the render has left beside the weights and the denoise
  workspace; the plan parks the text encoder to make room when it has to. If
  it does not fit, the render recomputes the prefix every step and says so in
  a request warning. **This means that on the fast path, whether a render
  retains its cache — and therefore the exact pixels of a very long prefix,
  such as several reference images — can depend on the card's free memory**,
  much as upstream simply runs out of memory where the cache does not fit.
  Set `on` or `off` to pin it.
- Under `MOLD_ATTN=math` (the v0.32 path), on Metal and on CPU, `auto` keeps
  the request-only rule: a text-to-image prompt of up to 512 tokens is cached
  exactly as v0.32 did, and a reference request is cached only when all its
  branches fit in a fixed 6 GiB budget, so the same request renders the same
  bytes on any card.

Cached and uncached renders are not bit-identical in BF16, which is why the
setting is part of the execution identity.
