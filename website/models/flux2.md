# Flux.2 Klein

A lightweight 4B parameter FLUX variant. Fast 4-step generation with low VRAM
requirements.

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Apache 2.0
- **HuggingFace**:
  [black-forest-labs/FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-Klein)

## Variants

| Model              | Steps | Size   | Notes             |
| ------------------ | ----- | ------ | ----------------- |
| `flux2-klein:q8`   | 4     | 4.3 GB | Good quality      |
| `flux2-klein:q6`   | 4     | 3.4 GB | Better quality    |
| `flux2-klein:q4`   | 4     | 2.6 GB | Smallest FLUX     |
| `flux2-klein:bf16` | 4     | 7.8 GB | Full precision 4B |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 0.0
- **Steps**: 4

---

# Flux.2 Klein-9B

A larger 9B parameter FLUX variant. Distilled for fast 4-step generation with
higher quality than the 4B Klein. Uses a Qwen3-8B text encoder (hidden_size=4096)
vs Klein-4B's Qwen3-4B (hidden_size=2560).

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Non-Commercial
- **HuggingFace**:
  [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  (gated — requires HuggingFace license acceptance)

## Variants

| Model                 | Steps | Size   | Notes                           |
| --------------------- | ----- | ------ | ------------------------------- |
| `flux2-klein-9b:q8`   | 4     | 10 GB  | Good quality                    |
| `flux2-klein-9b:q6`   | 4     | 7.9 GB | Better quality                  |
| `flux2-klein-9b:q4`   | 4     | 5.9 GB | Smallest 9B                     |
| `flux2-klein-9b:bf16` | 4     | 18 GB  | Full precision, gated, 2 shards |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 1.0
- **Steps**: 4

> **Note**: GGUF quantized variants (Q4/Q6/Q8) use ~6-10GB VRAM. The BF16
> variant requires ~18GB VRAM, is gated on HuggingFace, and requires license
> acceptance before download.

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

**Flux.2 Klein Q8** — 4 steps, seed 100:

```bash
mold run flux2-klein:q8 "A minimalist zen garden with raked sand patterns, a single cherry blossom tree, morning mist" --seed 100
```

![Zen garden — Flux.2 Klein](/gallery/flux2-klein-zen.png)

**Flux.2 Klein BF16** — 4 steps:

```bash
mold run flux2-klein:bf16 "a majestic owl perched on a mossy branch in a moonlit forest"
```

![Owl — Flux.2 Klein BF16](/gallery/flux2-klein-owl.png)

## Architecture

Flux.2 Klein uses a Qwen3 text encoder (BF16 or GGUF, layers 9/18/27), a shared
modulation transformer (BF16 or GGUF), and a BN-VAE decoder. Klein-4B uses
Qwen3-4B (hidden_size=2560), Klein-9B uses Qwen3-8B (hidden_size=4096). GGUF
variants keep weights quantized in VRAM with on-the-fly dequantization per
matmul, minimizing memory usage.
