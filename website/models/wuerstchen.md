# Wuerstchen v2

A research model featuring a unique 3-stage cascade architecture with 42x
latent compression. CLIP-G text encoder feeds into Prior → Decoder → VQ-GAN
stages. Developed in 2023, Wuerstchen is no longer actively maintained — its
authors went on to create Stable Cascade (also discontinued).

- **Developer**: [Wuerstchen Team](https://huggingface.co/warp-ai)
- **License**: MIT
- **HuggingFace**:
  [warp-ai/wuerstchen](https://huggingface.co/warp-ai/wuerstchen)

> **Recommendation**: For most use cases,
> [Flux.2 Klein](/models/flux2) produces significantly better images at similar
> or lower VRAM usage in fewer steps. Wuerstchen is best suited for users
> interested in the cascade architecture or who prefer its natural painterly
> aesthetic.

## Variants

| Model                | Steps | Size   | Notes                 |
| -------------------- | ----- | ------ | --------------------- |
| `wuerstchen-v2:fp16` | 30    | 5.6 GB | Full cascade pipeline |

No quantized (GGUF) variants are available for this model.

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 4.0
- **Steps**: 30

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1024  | 1024   | 1:1 (native) |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.

## Notes

Wuerstchen produces softer, painterly images compared to FLUX or SDXL. Output
quality is lower than other model families — expect less fine detail and
occasional anatomical inconsistencies. There is no community ecosystem of
LoRA adapters, fine-tunes, or ControlNet support for this model.

Wuerstchen includes a default negative prompt. Negative prompts are supported
and effective with this model. The 42x latent compression means the diffusion
process operates in a very compact space (24x24 for 1024x1024 output).

Features not yet supported: img2img, inpainting, ControlNet.

## Example

**Wuerstchen v2 FP16** — 30 steps, seed 42:

```bash
mold run wuerstchen-v2:fp16 "A lighthouse on a rocky coast during a dramatic sunset, oil painting style, vibrant orange and purple sky" --seed 42
```

![Lighthouse — Wuerstchen v2](/gallery/wuerstchen-lighthouse.png)
