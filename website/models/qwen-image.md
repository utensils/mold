# Qwen-Image (Alpha)

::: warning Alpha Status Qwen-Image is partially working and still in active
alpha development. Results may vary. :::

Qwen2.5-VL text encoder with a 3D causal VAE (2D temporal-slice) and
flow-matching with classifier-free guidance.

- **Developer**: [Alibaba / Qwen Team](https://huggingface.co/Qwen)
- **License**: Apache 2.0
- **HuggingFace**:
  [Qwen/Qwen2.5-Image-2512](https://huggingface.co/Qwen/Qwen2.5-Image-2512)

## Variants

| Model             | Steps | Size    | Notes                           |
| ----------------- | ----- | ------- | ------------------------------- |
| `qwen-image:bf16` | 30    | 44+ GB  | Full precision, maximum quality |
| `qwen-image:q8`   | 30    | 21.8 GB | Best quality                    |
| `qwen-image:q6`   | 30    | 16.8 GB | Best quality/size trade-off     |
| `qwen-image:q4`   | 30    | 12.3 GB | Smallest practical footprint    |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 3.0
- **Steps**: 30

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1024  | 1024   | 1:1 (native) |
| 1152  | 896    | 9:7          |
| 896   | 1152   | 7:9          |
| 1216  | 832    | 19:13        |
| 832   | 1216   | 13:19        |
| 1344  | 768    | 7:4          |
| 768   | 1344   | 4:7          |
| 1536  | 640    | 12:5         |
| 640   | 1536   | 5:12         |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.
