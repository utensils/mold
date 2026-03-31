# FLUX.1

The highest quality model family. T5-XXL + CLIP-L text encoding with a
flow-matching transformer.

- **Developer**: [Black Forest Labs](https://blackforestlabs.ai/)
- **License**: Apache 2.0 (Schnell), [FLUX.1 Dev Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) (Dev)
- **HuggingFace**: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## Variants

| Model               | Steps | Size    | Notes                        |
| ------------------- | ----- | ------- | ---------------------------- |
| `flux-schnell:q8`   | 4     | 12 GB   | Fast, general purpose        |
| `flux-schnell:bf16` | 4     | 23.8 GB | Full precision (>24 GB VRAM) |
| `flux-schnell:q4`   | 4     | 7.5 GB  | Lighter                      |
| `flux-dev:q8`       | 25    | 12 GB   | Full quality                 |
| `flux-dev:bf16`     | 25    | 23.8 GB | Full precision (>24 GB VRAM) |
| `flux-dev:q4`       | 25    | 7 GB    | Full quality, less VRAM      |

## Fine-Tunes

| Model               | Steps | Size    | Style                   |
| ------------------- | ----- | ------- | ----------------------- |
| `flux-krea:q8`      | 25    | 12.7 GB | Aesthetic photography   |
| `flux-krea:fp8`     | 25    | 11.9 GB | Aesthetic photography   |
| `jibmix-flux:q4`    | 25    | 6.9 GB  | Photorealistic          |
| `ultrareal-v4:q8`   | 25    | 12.6 GB | Photorealistic (latest) |
| `ultrareal-v4:q4`   | 25    | 6.7 GB  | Photorealistic, lighter |
| `ultrareal-v3:q8`   | 25    | 12.7 GB | Photorealistic          |
| `ultrareal-v2:bf16` | 25    | 23.8 GB | Full precision          |
| `iniverse-mix:fp8`  | 25    | 11.9 GB | Realistic SFW/NSFW mix  |

## Examples

**FLUX Schnell Q8** — 4 steps, seed 42:

```bash
mold run flux-schnell:q8 "A majestic snow leopard perched on a Himalayan cliff at golden hour, cinematic lighting, photorealistic" --seed 42
```

![Snow leopard — FLUX Schnell](/gallery/flux-schnell-leopard.png)

**FLUX Dev Q4** — 25 steps, seed 1337:

```bash
mold run flux-dev:q4 "A cozy Japanese tea house interior with warm lantern light, steam rising from ceramic cups, watercolor style" --seed 1337
```

![Tea house — FLUX Dev](/gallery/flux-dev-teahouse.png)

## LoRA Support

FLUX models support LoRA adapters in both BF16 and GGUF quantized modes:

```bash
mold run flux-dev:bf16 "a portrait" --lora style.safetensors --lora-scale 0.8
mold run flux-dev:q4 "a portrait" --lora style.safetensors --lora-scale 0.8
```

## VRAM Notes

- Full BF16 (23 GB) auto-offloads on 24 GB cards — blocks stream CPU↔GPU
- GGUF quantized (Q4/Q8) fits without offloading
- Use `--eager` to keep encoders loaded between generations (faster, more VRAM)
- T5-XXL encoder auto-selects quantized variant when VRAM is tight
