# Qwen-Image (Alpha)

::: warning Alpha Status Qwen-Image is partially working and still in active
alpha development. Results may vary. :::

Qwen2.5-VL text encoder with a 3D causal VAE (2D temporal-slice) and
flow-matching with classifier-free guidance.

## Variants

| Model           | Steps | Size    | Notes                        |
| --------------- | ----- | ------- | ---------------------------- |
| `qwen-image:q8` | 30    | 21.8 GB | Best quality                 |
| `qwen-image:q6` | 30    | 16.8 GB | Best quality/size trade-off  |
| `qwen-image:q4` | 30    | 12.3 GB | Smallest practical footprint |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 7.0
- **Steps**: 30
