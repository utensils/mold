# SDXL

Fast and flexible. Dual-CLIP text encoding (CLIP-L + CLIP-G) with UNet denoising
and classifier-free guidance.

- **Developer**: [Stability AI](https://stability.ai/)
- **License**:
  [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- **HuggingFace**:
  [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

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
