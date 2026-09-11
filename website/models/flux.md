# FLUX.1

The highest quality model family. T5-XXL + CLIP-L text encoding with a
flow-matching transformer.

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Apache 2.0 (Schnell),
  [FLUX.1 Dev Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
  (Dev)
- **HuggingFace**:
  [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell),
  [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## Variants

| Model               | Steps | Size    | Notes                        |
| ------------------- | ----- | ------- | ---------------------------- |
| `flux-schnell:q8`   | 4     | 12 GB   | Fast, general purpose        |
| `flux-schnell:q6`   | 4     | 9.8 GB  | Best quality/size trade-off  |
| `flux-schnell:bf16` | 4     | 23.8 GB | Full precision (>24 GB VRAM) |
| `flux-schnell:q4`   | 4     | 7.5 GB  | Lighter                      |
| `flux-dev:q8`       | 25    | 12 GB   | Full quality                 |
| `flux-dev:q6`       | 25    | 9.9 GB  | Best quality/size trade-off  |
| `flux-dev:bf16`     | 25    | 23.8 GB | Full precision (>24 GB VRAM) |
| `flux-dev:q4`       | 25    | 7 GB    | Full quality, less VRAM      |

## Fine-Tunes

| Model               | Steps | Size    | Style                   |
| ------------------- | ----- | ------- | ----------------------- |
| `flux-krea:q8`      | 25    | 12.7 GB | Aesthetic photography   |
| `flux-krea:q6`      | 25    | 9.8 GB  | Aesthetic photography   |
| `flux-krea:q4`      | 25    | 7.5 GB  | Aesthetic photography   |
| `flux-krea:fp8`     | 25    | 11.9 GB | Aesthetic photography   |
| `jibmix-flux:fp8`   | 25    | 11.9 GB | Photorealistic          |
| `jibmix-flux:q5`    | 25    | 8.4 GB  | Photorealistic          |
| `jibmix-flux:q4`    | 25    | 6.9 GB  | Photorealistic          |
| `jibmix-flux:q3`    | 25    | 5.4 GB  | Photorealistic, lighter |
| `ultrareal-v4:q8`   | 25    | 12.6 GB | Photorealistic (latest) |
| `ultrareal-v4:q5`   | 25    | 8.0 GB  | Photorealistic          |
| `ultrareal-v4:q4`   | 25    | 6.7 GB  | Photorealistic, lighter |
| `ultrareal-v3:q8`   | 25    | 12.7 GB | Photorealistic          |
| `ultrareal-v3:q6`   | 25    | 9.8 GB  | Photorealistic          |
| `ultrareal-v3:q4`   | 25    | 6.8 GB  | Photorealistic, lighter |
| `ultrareal-v2:bf16` | 25    | 23.8 GB | Full precision          |
| `iniverse-mix:fp8`  | 25    | 11.9 GB | Realistic SFW/NSFW mix  |

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

## Examples

**FLUX Schnell Q8**: 4 steps, seed 42:

```bash
mold run flux-schnell:q8 \
  "A majestic snow leopard perched on a Himalayan cliff \
  at golden hour, cinematic lighting, photorealistic" \
  --seed 42
```

![Snow leopard, FLUX Schnell](/gallery/flux-schnell-leopard.png)

**FLUX Dev Q4**: 25 steps, seed 1337:

```bash
mold run flux-dev:q4 \
  "A cozy Japanese tea house interior with warm lantern light, \
  steam rising from ceramic cups, watercolor style" \
  --seed 1337
```

![Tea house, FLUX Dev](/gallery/flux-dev-teahouse.png)

## LoRA Support

FLUX models support LoRA adapters in both BF16 and GGUF quantized modes:

```bash
mold run flux-dev:bf16 "a portrait" --lora style.safetensors --lora-scale 0.8
mold run flux-dev:q4 "a portrait" --lora style.safetensors --lora-scale 0.8
```

## Speed

On CUDA, FLUX renders through FlashAttention-2 and cuDNN by default wherever
the artifact compiled them: `mold` (sm89), `mold-sm86`, `mold-sm100` and the
matching desktop packages compile both.

**`mold-sm120` is the exception and uses math attention.** FlashAttention
picks its tile from a runtime check that reads consumer Blackwell as a
large-shared-memory datacenter part, which it is not, so an RTX 50-series card
would get a tile tuned for an A100 — correct, but of unmeasured speed, and mold
has no 50-series card to measure it on. It is qualified there when someone
measures it. The byte change below still applies to that artifact: it is a
property of every CUDA build, not of the kernel.

A build you make yourself with `--features cuda` and no `flash-attn` is in the
same position — correct, byte-changed, and on the math path — so add
`flash-attn` to the feature list on sm86, sm89 or sm100. Metal builds have no
flash kernel at all. The GGUF
tiers also run their activations in BF16 rather than F32, which halves the
bandwidth every matmul moves and lets the tensor cores engage; candle's
quantized kernels take BF16 and return it, so the weights stay quantized in
VRAM exactly as before. Set `MOLD_ATTN=math` and `MOLD_CONV=im2col` to render
the byte-stable way instead.

A FLUX print archived before mold 0.29 will not re-render byte-for-byte after
it, under any setting: flash attention, cuDNN and BF16 activations all change
the order the same sums are accumulated in. Renders made from 0.29 on are
reproducible among themselves — same seed, same settings, same backend, same
bytes.

The full-precision `:bf16` tier is the one exception on the attention side: it
runs through upstream Candle's own attention, which has no backend switch, so
it stays on the math path.

Under that, a step spends much less time in its small operations: the Q/K norms
and the affine-less LayerNorms now hit candle's fused kernels instead of a
ten-operation strided fallback, and the rotary embedding takes candle's fused
interleaved kernel wherever the layout allows. FLUX.1 also builds that rotary
embedding in float32, as upstream does — it previously used whatever dtype the
render was in, so a half-precision render computed every sine and cosine of
every token position with eight bits of mantissa. That one is a correctness
fix, and it changes the picture.

### Loading a GGUF checkpoint

A whole quantized checkpoint is read in contiguous batches across eight threads
into a reused page-locked staging buffer, and uploaded to the GPU from there
with two buffers alternating so one is filling while the other is still in
flight. It used to be read through a memory mapping, which on ZFS — where
`$MOLD_HOME` lives on every machine mold is qualified on — is one page fault
per 4 KiB with no readahead. Measured with the file's page cache dropped,
`flux1-dev-Q8_0` went from 15.5 s to 1.6 s. The weights are byte-identical, so
renders are too. macOS and CPU keep the mapping, where staging would be a pure
extra copy.

## Memory

### The transformer stays on the card when it fits

mold used to drop the transformer before every VAE decode and rebuild it on
the next render, whatever the card had room for. On a 46 GB GPU that is 8.4
seconds of disk read per print for a Q8 tier, and with a LoRA stack the
rebuild peaks at around 95 GB of host RAM.

The decision is now a measurement, taken per render: the resident checkpoint,
this render's denoise workspace, the VAE decode's workspace and a 1 GB
allocator margin, against the card's usable free VRAM. The three workspace
terms are added rather than maxed, because the answer has to hold for the
denoise and the decode both.

The canvas is part of that sum — the decode workspace is about 2.7 GB in bf16
at 1024x1024 and grows with area — but FLUX renders are capped at 1.8
megapixels (1328x1328 at the square), so across the whole range mold will
actually render, the answer comes down to the checkpoint and the card:

- **24 GB** keeps a Q8 tier (~12.6 GB) resident, and never has room for the
  BF16 one (~23.8 GB).
- **46 GB** keeps the BF16 tier resident as well.

`MOLD_FLUX_KEEP_TRANSFORMER=0` forces the old drop if you need the VRAM for
something else. `=1` is accepted and means the same thing as the default: an
explicit keep has always had to yield to a card that cannot afford it, and the
budget is now what expresses that for everyone.

What the print's execution fingerprint records is the **request**, not the
outcome: `0` is its own execution class and unset and `1` share the other, so a
forced drop is never filed with a budgeted render. It cannot record the outcome
— that is a per-render measurement against whatever VRAM happened to be free,
so two prints made by the same command on the same card can legitimately differ
and are still the same execution. The measurement is reported in the server log
instead, one line per render, naming which way it went and why: `Transformer
kept resident: the residency budget fits`, `Transformer dropped before VAE
decode: the residency budget does not fit this card at this resolution`, or
`Transformer dropped before VAE decode (MOLD_FLUX_KEEP_TRANSFORMER=0)`.

### Text encoders

T5 and CLIP are dropped after they encode, so the denoise has the VRAM.
`MOLD_KEEP_TE_RAM=1` parks them in host RAM between requests instead, which
turns a cache-miss prompt into a host-to-device copy rather than a re-read from
disk. That is unchanged for FLUX.1, and it still costs one copy rather than the
two it used to: parking a 9.79 GB T5 briefly needed 19.6 GB, for a feature
whose purpose is fitting that encoder in host RAM.

The variable is now tri-state, but the third state is about the other
families. Unset is `auto`, and `auto` measures the host — it parks only when
the encoder, the transformer loading beside it, and a `max(15% of RAM, 8 GiB)`
safety floor all fit in available memory — for the encoders that decision was
built for: [Flux.2](/models/flux2)'s Mistral3 and Qwen3, and Z-Image's Qwen3.
FLUX.1's T5 and CLIP keep the old rule, so unset still means "do not park"
here. `0` never parks anywhere, and Metal never parks at all — there the parked
copy would sit in the pool the encoder already runs from.

### VRAM notes

- Full BF16 (23 GB) auto-offloads on 24 GB cards; blocks stream CPU↔GPU. The
  decision asks what the card has free, not only what the file weighs, so a
  23.8 GB checkpoint no longer streams on a 46 GB GPU that could hold it.
- GGUF quantized (Q4/Q8) fits without offloading
- Use `--eager` to keep encoders loaded between generations (faster, more VRAM)
- T5-XXL encoder auto-selects quantized variant when VRAM is tight
