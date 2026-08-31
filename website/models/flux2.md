# Flux.2

Mold supports the distilled Klein checkpoints, the undistilled Klein base
checkpoints, and the full FLUX.2 Dev checkpoint.

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

Classic strength-based img2img, masks, ControlNet, LoRA, and batches with
references are rejected because the checkpoint-native reference protocol does
not implement those controls. Text-only batches remain supported.

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
above 1.0 runs a second, unconditional forward per step, so a base render costs
roughly twice a distilled render of the same step count; `--guidance 1` skips
the branch entirely.

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
  --negative "blurry, low contrast, plastic"
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
