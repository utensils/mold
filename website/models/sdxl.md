# SDXL

Fast and flexible. Dual-CLIP text encoding (CLIP-L + CLIP-G) with UNet denoising
and classifier-free guidance.

## Variants

| Model                      | Steps | Size   | Notes                   |
| -------------------------- | ----- | ------ | ----------------------- |
| `sdxl-turbo:fp16`          | 4     | 5.1 GB | Ultra-fast, 1–4 steps   |
| `dreamshaper-xl:fp16`      | 8     | 5.1 GB | Fantasy, concept art    |
| `juggernaut-xl:fp16`       | 30    | 5.1 GB | Photorealism, cinematic |
| `realvis-xl:fp16`          | 25    | 5.1 GB | Photorealism, versatile |
| `playground-v2.5:fp16`     | 25    | 5.1 GB | Artistic, aesthetic     |
| `sdxl-base:fp16`           | 25    | 5.1 GB | Official base model     |
| `pony-v6:fp16`             | 25    | 5.1 GB | Anime, art, stylized    |
| `cyberrealistic-pony:fp16` | 25    | 5.1 GB | Photorealistic Pony     |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 7.5 (0.0 for turbo)
- **Scheduler**: DDIM (also supports euler-ancestral, uni-pc)

## Example

**SDXL Turbo** — 4 steps, seed 88:

```bash
mold run sdxl-turbo:fp16 "A vibrant street food market in Bangkok at night, neon signs, steam from woks, bustling crowd" --seed 88
```

![Street market — SDXL Turbo](/gallery/sdxl-turbo-market.png)

## Negative Prompts

SDXL uses classifier-free guidance — negative prompts have a strong effect:

```bash
mold run sdxl-base:fp16 "a landscape" -n "low quality, blurry, watermark"
```
