# Flux.2

Mold supports the distilled Klein checkpoints, the undistilled Klein base
checkpoints, and the full FLUX.2 Dev checkpoint.

## Reference-image editing

Every FLUX.2 checkpoint speaks the same reference protocol — Black Forest Labs
ships text-to-image, single-reference editing, and multi-reference editing in
one model for Klein 4B and 9B, both Base tiers, and Dev. Mold accepts up to
four ordered PNG/JPEG references per render, each placed on its own time plane
in the order given, so "image 1" in the prompt is the first reference.

```bash
# Klein: --reference carries the ordered group
mold run flux2-klein:bf16 "put sunglasses on the person, keep the pose and background" \
  --reference person.jpg

mold run flux2-klein-9b:q8 "the woman from image 1 wearing the eyeglasses from image 2" \
  --reference person.jpg --reference glasses.jpg

# Dev reads its ordered references from repeated --image
mold run flux2-dev:q6 "the jacket from image 1 on the model from image 2" \
  --image jacket.png --image model.png
```

The difference between the tiers is what references do to the source image, and
`/api/models[].generation_profile.capabilities.reference_images` is the single
place that says so — no client derives it from the model name:

| Tier                       | Source image                                     | References                  | Together?                                   |
| -------------------------- | ------------------------------------------------ | --------------------------- | ------------------------------------------- |
| Klein (distilled and Base) | `--image`, with `--strength`, `--mask`, and LoRA | `--reference`, up to 4      | No — one pass renders from one or the other |
| Dev                        | none                                             | repeated `--image`, up to 4 | n/a                                         |

Klein's relation is _exclusive_: passing `--reference` and `--image` together is
refused rather than silently dropping one, and a Klein render with no references
attached is an ordinary img2img (or text-to-image) pass with every control
intact. Name each reference's role in the prompt — BFL's guidance is to
"clearly describe the role of each: subject from image 1, style from image 2,
background from image 3" — because references with no stated role blend.

## Flux.2 Dev

The 32B-class checkpoint uses a streamed Mistral3 prompt encoder and
automatically block-offloads transformer blocks when the selected CUDA GPU
cannot keep the transformer resident. Expect substantial host-RAM and model
storage requirements even when GPU residency is bounded.

- **Defaults**: 50 steps, guidance 4.0, 1024x1024
- **License**: FLUX Non-Commercial License
- **Conditioning**: text-to-image or up to four ordered PNG/JPEG references

### Variants

| Model            | Size  | Gated | Notes                               |
| ---------------- | ----- | ----- | ----------------------------------- |
| `flux2-dev:q4`   | 20 GB | no    | Smallest dev tier; fits a 24 GB GPU |
| `flux2-dev:q6`   | 27 GB | no    | Fits a 32 GB GPU with room to spare |
| `flux2-dev:q8`   | 35 GB | no    | Near-BF16 quality                   |
| `flux2-dev:fp8`  | 35 GB | no    | Mixed FP8 — BF16 attention, FP8 MLP |
| `flux2-dev:bf16` | 65 GB | yes   | Full precision, 7 shards            |

Sizes are the transformer alone. Every tier also pulls the Mistral3 encoder,
VAE, and tokenizer (~36 GB), shared across tiers.

The bare name `flux2-dev` means `flux2-dev:bf16`; name a tag for the others.
Only the safetensors tiers (`bf16`, `fp8`) block-offload when a CUDA GPU
cannot hold the transformer — a GGUF tier stays fully resident, so its size
above is the VRAM it needs.

Only `flux2-dev:bf16` is gated: it comes from Black Forest Labs'
[FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repo, which
holds the transformer and the encoder together. The quantized tiers pull
their transformers from
[unsloth/FLUX.2-dev-GGUF](https://huggingface.co/unsloth/FLUX.2-dev-GGUF) and
[Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev), and their
runtime assets from ungated mirrors of the same bytes — so they install with
no license acceptance and no HuggingFace token.

```bash
mold pull flux2-dev:q4
mold run flux2-dev:q4 "a cinematic portrait in rain"
mold run flux2-dev:q4 "preserve the subject, change the lighting" \
  --image reference.png

# The BF16 tier is gated:
hf auth login
mold pull flux2-dev:bf16
```

### The prompt encoder does not need 36 GB of VRAM

The Mistral3 encoder is 36 GB on disk and mold used to plan for all of it,
which made even an idle 46 GB card look over-subscribed: the encoder was moved
to the CPU, where it runs at F32, and a cache-miss prompt took **78.8 seconds**
with the GPU completely idle.

It never needed that much. The encoder streams — it memory-maps the shards and
builds one decoder layer at a time, holding the running layer and the next one
— so it runs on the GPU in bf16 at a peak of about **3.6 GB**, and those 36 GB
of shards stay reclaimable page cache rather than memory anything has to
reserve. mold now plans for the streamed peak on both sides, so:

- The encoder stays on the GPU on a 24 GB card as well as a 46 GB one, even
  beside a resident Q8 transformer. It runs before the transformer denoises,
  so the two phases do not overlap.
- **Host RAM**: you need room for the working set, not for the file — roughly
  7 GB if you deliberately pin the encoder to the CPU with
  `--device-text-encoders cpu`, and effectively nothing beyond page cache
  otherwise. A 64 GB desktop used to be refused outright.
- **Disk cache**: the shards are read through the page cache, so the second
  render of a session is much faster than the first on a machine with enough
  free RAM to keep them.

Pinning the encoder to the CPU is still honoured; it is just no longer chosen
for you on a card that had the room all along.

### A second render of the same prompt reuses what the first built

Two things now survive a render rather than being rebuilt from disk.

The **transformer** stays GPU-resident when the card has room for it beside
the VAE decode. The decision is a measurement taken per render — the resident
checkpoint, the denoise workspace, the decode workspace and a 1 GB allocator
margin against the card's usable free VRAM — so it moves with the canvas as
well as the card. Flux.2 renders are capped at 1.8 megapixels (1328x1328 at
the square) and the decode wants about 2.7 GB in bf16 at 1024x1024 and about
4.6 GB at that ceiling, so across everything mold will render: a 46 GB card
keeps a 33 GB Q8 [dev] transformer resident, a 24 GB card never does, and a
24 GB card keeps a Q8 Klein tier, 4B or 9B. It is released before the encoder streams
whenever the two would not fit together, and reused only when the LoRA stack,
the working precision, the GPU and the resolved architecture all match.
`MOLD_FLUX_KEEP_TRANSFORMER=0` forces the old drop-every-render behaviour.

The **encoder prefix** stays in host RAM when the machine can afford it, which
turns a cache-miss prompt into a host-to-device copy per layer instead of a
page fault, a dtype conversion and a copy. The park is measured, not a flag:
the prefix, the transformer that loads beside it, and a `max(15 % of RAM,
8 GiB)` floor must all fit in available memory, so a 64 GB desktop keeps
streaming and a 1.5 TB host parks and page-locks. Only the layers the encoder
actually runs are parked — the vision tower, the projector and layers 30-39
that the single-file republication also ships are never touched.
`MOLD_KEEP_TE_RAM=0` opts out; `MOLD_KEEP_TE_RAM=1` parks wherever the encoder
alone clears the floor.

Klein's Qwen3 encoder takes the same decision, quantized tiers included.

Classic strength-based img2img, masks, ControlNet, LoRA, and batches with
references are rejected because the checkpoint-native reference protocol does
not implement those controls. Text-only batches remain supported. This is Dev
alone: Klein keeps img2img, masks, and LoRA, because its references are an
alternative to the source image rather than a replacement for it.

## Flux.2 Klein

A lightweight 4B parameter FLUX variant. Fast 4-step generation with low VRAM
requirements.

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Apache 2.0
- **HuggingFace**:
  [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
  (BF16 transformer, Qwen3 encoder shards, VAE, tokenizer);
  [unsloth/FLUX.2-klein-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF)
  (quantized tiers)

## Variants

| Model              | Steps | Size   | Notes             |
| ------------------ | ----- | ------ | ----------------- |
| `flux2-klein:q8`   | 4     | 4.3 GB | Good quality      |
| `flux2-klein:q6`   | 4     | 3.4 GB | Better quality    |
| `flux2-klein:q4`   | 4     | 2.6 GB | Smallest FLUX     |
| `flux2-klein:fp8`  | 4     | 4.1 GB | BFL's own FP8     |
| `flux2-klein:bf16` | 4     | 7.8 GB | Full precision 4B |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 1.0
- **Steps**: 4

---

# Flux.2 Klein Base

The undistilled Klein checkpoints. Same architecture as the distilled tiers —
same encoders, same VAE, same shapes — but trained without step or guidance
distillation, so they trade speed for flexibility: ~50 steps, real
classifier-free guidance, and higher output diversity. Black Forest Labs
publishes them as the base for fine-tuning, LoRA training, and custom
pipelines.

These are the only Flux.2 checkpoints that use a **negative prompt**. Guidance
above 1.0 adds an unconditional branch; `--guidance 1` skips it entirely. Where
the card has room, both branches ride in ONE batch-2 forward per step, so each
weight is read once for the pair and a guided render costs far less than two
separate ones — on a bandwidth-bound quantized tier, closer to 1.2x a distilled
render than 2x. Both prompts are padded to the same fixed 512 tokens, so the
only thing that can send a render back to two sequential forwards is the
doubled activations not fitting beside the weights on this card. The progress
line says which ran.

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Apache 2.0 (4B), Non-Commercial (9B)
- **HuggingFace**:
  [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B),
  [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
  (gated),
  [unsloth/FLUX.2-klein-base-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-base-4B-GGUF),
  [unsloth/FLUX.2-klein-base-9B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-base-9B-GGUF)

## Variants

| Model                      | Steps | Size   | Notes                    |
| -------------------------- | ----- | ------ | ------------------------ |
| `flux2-klein-base:q4`      | 50    | 2.6 GB | Smallest base tier       |
| `flux2-klein-base:q6`      | 50    | 3.4 GB | Better quality           |
| `flux2-klein-base:q8`      | 50    | 4.3 GB | Near-BF16 quality        |
| `flux2-klein-base:bf16`    | 50    | 7.8 GB | Full precision 4B        |
| `flux2-klein-base-9b:q4`   | 50    | 5.9 GB | Smallest 9B base         |
| `flux2-klein-base-9b:q6`   | 50    | 7.9 GB | Better quality           |
| `flux2-klein-base-9b:q8`   | 50    | 10 GB  | Near-BF16 quality        |
| `flux2-klein-base-9b:bf16` | 50    | 18 GB  | Full precision, 2 shards |

Every base tier shares the distilled tiers' encoder and VAE bytes, so a host
that already has `flux2-klein` installed downloads only the transformer.

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 4.0 (a true CFG scale)
- **Steps**: 50

## Example

```bash
mold pull flux2-klein-base:q8
mold run flux2-klein-base:q8 \
  "a weathered brass diving helmet on a workbench, single window light" \
  --guidance 4 --steps 50 \
  --negative-prompt "blurry, low contrast, plastic"
```

---

# Flux.2 Klein-9B

A larger 9B parameter FLUX variant. Distilled for fast 4-step generation with
higher quality than the 4B Klein. Uses a Qwen3-8B text encoder (hidden_size=4096)
vs Klein-4B's Qwen3-4B (hidden_size=2560).

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Non-Commercial
- **HuggingFace**:
  [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  (gated; requires HuggingFace license acceptance)

## Variants

| Model                 | Steps | Size   | Notes                           |
| --------------------- | ----- | ------ | ------------------------------- |
| `flux2-klein-9b:q8`   | 4     | 10 GB  | Good quality                    |
| `flux2-klein-9b:q6`   | 4     | 7.9 GB | Better quality                  |
| `flux2-klein-9b:q4`   | 4     | 5.9 GB | Smallest 9B                     |
| `flux2-klein-9b:fp8`  | 4     | 9.4 GB | BFL's own FP8, gated            |
| `flux2-klein-9b:bf16` | 4     | 18 GB  | Full precision, gated, 2 shards |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 1.0
- **Steps**: 4

> **Note**: GGUF quantized variants (Q4/Q6/Q8) use ~6-10GB VRAM. The BF16
> variant requires ~18GB VRAM, is gated on HuggingFace, and requires license
> acceptance before download. Use `--offload` with BF16 when VRAM is tight;
> GGUF and LoRA offload are rejected.

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1024  | 1024   | 1:1 (native) |
| 1024  | 768    | 4:3          |
| 768   | 1024   | 3:4          |
| 1024  | 576    | 16:9         |
| 576   | 1024   | 9:16         |
| 768   | 768    | 1:1          |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.

## Example

**Flux.2 Klein Q8**: 4 steps, seed 100:

```bash
mold run flux2-klein:q8 \
  "A minimalist zen garden with raked sand patterns, \
  a single cherry blossom tree, morning mist" \
  --seed 100
```

![Zen garden, Flux.2 Klein](/gallery/flux2-klein-zen.png)

**Flux.2 Klein BF16**: 4 steps:

```bash
mold run flux2-klein:bf16 \
  "a majestic owl perched on a mossy branch in a moonlit forest"
```

![Owl, Flux.2 Klein BF16](/gallery/flux2-klein-owl.png)

**Flux.2 Klein-9B Q4**: 4 steps, seed 999:

```bash
mold run flux2-klein-9b:q4 \
  "A glass bottle ship inside a stormy ocean wave, \
  dramatic lightning, hyperrealistic macro photography" \
  --seed 999
```

![Bottle ship, Flux.2 Klein-9B Q4](/gallery/flux2-klein-9b-bottle-ship.png)

## Architecture

Flux.2 Klein uses a Qwen3 text encoder (BF16 or GGUF, layers 9/18/27), a shared
modulation transformer (BF16 or GGUF), and a BN-VAE decoder. Klein-4B uses
Qwen3-4B (hidden_size=2560), Klein-9B uses Qwen3-8B (hidden_size=4096). GGUF
variants keep weights quantized in VRAM with on-the-fly dequantization per
matmul, minimizing memory usage.

Every Klein prompt is truncated and padded to a fixed 512 tokens before it
reaches the transformer, and every text token carries its own running position
— both are Black Forest Labs' own conditioning contract, which mold did not
follow before 0.29, so a Klein or dev render made with an earlier version will
not reproduce from the same seed and settings.

## Speed

On CUDA, Flux.2 renders through FlashAttention-2 and cuDNN by default wherever
the artifact compiled them — `mold` (sm89), `mold-sm86`, `mold-sm100` and the
matching desktop packages compile both. `mold-sm120` uses math attention:
FlashAttention's tile selection reads consumer Blackwell as a datacenter part
with far more shared memory than it has, and mold owns no RTX 50-series card to
measure the result on, so that artifact waits for qualification. A self-built
`--features cuda` binary without `flash-attn`, and every Metal build, take the
math path too: still correct, but carrying 0.29's byte change without its
speedup — the byte change is a property of every CUDA build, not of the kernel.
`MOLD_ATTN=math`
and `MOLD_CONV=im2col` render the byte-stable way instead; a print archived
before mold 0.29 does not re-render byte-for-byte after it under any setting,
and renders made from 0.29 on are reproducible among themselves. Every other
still family keeps the math/im2col defaults it has always had.

**GGUF tiers run in BF16.** Activations follow the working dtype instead of
being cast to F32 at the transformer boundary, which halves the bandwidth every
matmul moves; the weights stay quantized in VRAM exactly as before, and
position ids stay F32 because the rotary embedding is built from them. The
transformer also no longer wraps all eighteen of its linear sites in a
full-tensor NaN compare — about half a second per step spent masking a fault
that had never been observed, and which would have been the wrong thing to hide
anyway. `MOLD_FLUX_DEBUG_NONFINITE=1` replaces it with one check per denoise
step that names the step and fails; `MOLD_FLUX2_QMATMUL=0` restores the
per-forward dequantization arm if a render comes out wrong.

**FP8 tiers widen their weights once.** An FP8 layer used to rebuild a
working-dtype copy of its whole slab on every call, so the tier chosen to save
VRAM paid full BF16 bandwidth for its weights. Where the card has room the
widening now happens once at load and the packed slab is dropped — the same
arithmetic in the same order, so the picture does not change. Free VRAM decides,
measured before the first weight lands; `MOLD_FLUX2_FP8_CACHE=1` or `=0` forces
it either way.

**A guided Klein Base step is one forward, not two.** Both branches denoise the
same latent, so they ride one batch-2 forward and every weight is read once for
the pair — which is what Black Forest Labs' own sampler does. mold falls back
to two sequential forwards when the doubled activations would not fit beside
the weights on this card; the progress line names which shape ran — `one
batched forward per step` or `two forwards per step` — and deliberately names
no cause, because on a real render the budget is the only one. (A batch-2
forward also needs both branches to be the same length, but every Klein prompt
is padded to a fixed 512 rows, so that gate now survives only as a structural
guard and no render reaches it.) `--guidance 1` still skips the branch entirely.

**Smaller operations got out of the way.** The Q/K norms and the affine-less
LayerNorms now hit candle's fused kernels rather than a ten-operation strided
fallback, the rotary embedding takes candle's fused interleaved kernel where
the layout allows, the double blocks issue one fused Q/K/V projection per
stream instead of three, and the VAE's mid-block attention no longer
materialises a full 16384x16384 score matrix during decode — the spike that
used to push a loaded card into the much slower tiled-decode recovery.

**Loading is 8-10x faster on a cold checkpoint.** A whole GGUF file is read in
contiguous batches across eight threads into a reused page-locked staging
buffer and uploaded from there, with two buffers alternating so one is filling
while the other is still in flight. Reading it through a memory mapping instead
is one page fault per 4 KiB on ZFS, which is where `$MOLD_HOME` lives on every
machine mold is qualified on. Measured with the page cache dropped,
`flux2-dev-Q8_0` went from 41.7 s to 4.3 s. The weights are byte-identical, so
renders are too; macOS and CPU keep the mapping, where staging would be a pure
extra copy.
