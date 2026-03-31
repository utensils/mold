# Z-Image

Qwen3 text encoder with a flow-matching transformer using 3D RoPE positional
encoding. Excellent quality at just 9 steps.

- **Developer**: [Z-Potentials](https://huggingface.co/Z-Potentials)
- **License**: Apache 2.0
- **HuggingFace**:
  [Z-Potentials/Z-Image-v1-Turbo](https://huggingface.co/Z-Potentials/Z-Image-v1-Turbo)

## Variants

| Model                | Steps | Size    | Notes          |
| -------------------- | ----- | ------- | -------------- |
| `z-image-turbo:q8`   | 9     | 6.6 GB  | Fast, great    |
| `z-image-turbo:q4`   | 9     | 3.8 GB  | Lighter        |
| `z-image-turbo:bf16` | 9     | 12.2 GB | Full precision |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 0.0
- **Steps**: 9

## Example

**Z-Image Turbo** — 9 steps, seed 777:

```bash
mold run z-image-turbo:q8 "An astronaut floating through a bioluminescent underwater cave, reflections on the helmet visor, science fiction art" --seed 777
```

![Astronaut — Z-Image Turbo](/gallery/zimage-astronaut.png)

## Notes

Z-Image uses a Qwen3 text encoder (BF16 or GGUF with auto-fallback). The
quantized transformer is implemented directly in mold (not upstream candle) due
to GGUF tensor naming differences.
