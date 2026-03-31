# Wuerstchen v2

A unique 3-stage cascade architecture with 42x latent compression. CLIP-G text
encoder feeds into Prior → Decoder → VQ-GAN stages.

## Variants

| Model                | Steps | Size   | Notes                 |
| -------------------- | ----- | ------ | --------------------- |
| `wuerstchen-v2:fp16` | 60    | 5.6 GB | Full cascade pipeline |

## Defaults

- **Resolution**: 1024x1024
- **Guidance**: 4.0
- **Steps**: 60

## Notes

Wuerstchen includes a default negative prompt. The 42x latent compression means
the diffusion process operates in a very compact space, which allows for
efficient generation despite the multi-stage pipeline.

Negative prompts are supported and effective with this model.
