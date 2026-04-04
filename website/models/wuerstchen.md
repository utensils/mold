# Wuerstchen v2

A unique 3-stage cascade architecture with 42x latent compression. CLIP-G text
encoder feeds into Prior → Decoder → VQ-GAN stages.

- **Developer**: [Wuerstchen Team](https://huggingface.co/warp-ai)
- **License**: MIT
- **HuggingFace**:
  [warp-ai/wuerstchen](https://huggingface.co/warp-ai/wuerstchen)

## Variants

| Model                | Steps | Size   | Notes                 |
| -------------------- | ----- | ------ | --------------------- |
| `wuerstchen-v2:fp16` | 60    | 5.6 GB | Full cascade pipeline |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 4.0
- **Steps**: 60

## Recommended Dimensions

| Width | Height | Aspect Ratio |
| ----- | ------ | ------------ |
| 1024  | 1024   | 1:1 (native) |

Using non-recommended dimensions will trigger a warning. All values must be
multiples of 16.

## Notes

Wuerstchen includes a default negative prompt. The 42x latent compression means
the diffusion process operates in a very compact space, which allows for
efficient generation despite the multi-stage pipeline.

Negative prompts are supported and effective with this model.

## Example

**Wuerstchen v2 FP16** — 60 steps, seed 42:

```bash
mold run wuerstchen-v2:fp16 "A lighthouse on a rocky coast during a dramatic sunset, oil painting style, vibrant orange and purple sky" --seed 42
```

![Lighthouse — Wuerstchen v2](/gallery/wuerstchen-lighthouse.png)
