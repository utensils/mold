# Z-Image

Qwen3 text encoder with a flow-matching transformer using 3D RoPE positional
encoding. Excellent quality at just 9 steps.

- **Developer**: [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) (Alibaba)
- **License**: Apache 2.0
- **HuggingFace**:
  [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
  (BF16 transformer shards, Qwen3 text encoder, VAE, tokenizer);
  [leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF)
  (quantized tiers)

## Variants

| Model                | Steps | Size    | Notes             |
| -------------------- | ----- | ------- | ----------------- |
| `z-image-turbo:q8`   | 9     | 6.6 GB  | Fast, great       |
| `z-image-turbo:q6`   | 9     | 5.3 GB  | Best quality/size |
| `z-image-turbo:q4`   | 9     | 3.8 GB  | Lighter           |
| `z-image-turbo:bf16` | 9     | 24.6 GB | Full precision    |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 0.0
- **Steps**: 9

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1024  | 1024   | 1:1 (native) |
| 1152  | 896    | 9:7          |
| 896   | 1152   | 7:9          |
| 1152  | 864    | 4:3          |
| 864   | 1152   | 3:4          |
| 1248  | 832    | 3:2          |
| 832   | 1248   | 2:3          |
| 1280  | 720    | 16:9         |
| 720   | 1280   | 9:16         |
| 1344  | 576    | 7:3          |
| 576   | 1344   | 3:7          |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.

## Example

**Z-Image Turbo**: 9 steps, seed 777:

```bash
mold run z-image-turbo:q8 \
  "An astronaut floating through a bioluminescent underwater cave, \
  reflections on the helmet visor, science fiction art" \
  --seed 777
```

![Astronaut, Z-Image Turbo](/gallery/zimage-astronaut.png)

## Notes

Z-Image uses a Qwen3 text encoder (BF16 or GGUF with auto-fallback). The
quantized transformer is implemented directly in mold (not upstream candle) due
to GGUF tensor naming differences.

On CUDA, the quantized transformer's linears dequantize each weight per
forward rather than feeding candle's quantized fast-matmul kernels, which
return non-finite values for Z-Image's layers and produced solid-black
renders. Metal keeps the fast quantized path, which is validated against
stable-diffusion.cpp. `MOLD_ZIMAGE_QMATMUL=1` re-enables the CUDA fast path
for kernel debugging only.

GGUF transformers always stay quantized at rest. Mold does not expose the old
dense-map diagnostic route: expanding a Q4 checkpoint from roughly 3.4 GB to
about 12 GB on CUDA or 24 GB on Metal was never memory-planned, its speed
advantage was unmeasured, and the Metal route produced corrupted renders under
that pressure ([#1109](https://github.com/utensils/mold/issues/1109)).

Catalog `cv:*` Z-Image checkpoints use the hidden `z-image-te` companion for
the same Qwen3 text encoder shards, tokenizer, and VAE. When the Civitai
version publishes its own text-encoder file, that per-version file is downloaded
and used instead of the shared encoder shards. An existing `z-image-turbo`
install satisfies the shared files.
