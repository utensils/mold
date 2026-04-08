# Feature Support

This page answers the practical question: which model families support which
features today?

## Quick Picks

| Need                          | Best Starting Point |
| ----------------------------- | ------------------- |
| LoRA adapters                 | FLUX.1              |
| ControlNet                    | SD 1.5              |
| img2img at 1024 output        | FLUX.1 or SDXL      |
| broadest feature surface      | SD 1.5 or SDXL      |
| best prompt-following quality | FLUX.1 or SD 3.5    |

## Source Image Workflows

| Family        | img2img | Inpainting |
| ------------- | ------- | ---------- |
| FLUX.1        | Yes     | Yes        |
| SDXL          | Yes     | Yes        |
| SD 1.5        | Yes     | Yes        |
| SD 3.5        | Yes     | Yes        |
| Z-Image       | Yes     | Yes        |
| Flux.2 Klein  | Yes     | Yes        |
| Wuerstchen v2 | Yes     | Yes        |
| Qwen-Image    | Yes     | Yes        |
| LTX Video     | Not yet | Not yet    |

## Control and Adapters

| Family        | ControlNet | LoRA |
| ------------- | ---------- | ---- |
| FLUX.1        | No         | Yes  |
| SDXL          | No         | No   |
| SD 1.5        | Yes        | No   |
| SD 3.5        | No         | No   |
| Z-Image       | No         | No   |
| Flux.2 Klein  | No         | No   |
| Wuerstchen v2 | No         | No   |
| Qwen-Image    | No         | No   |
| LTX Video     | No         | No   |

## Prompt Conditioning

| Family        | Negative Prompts | Scheduler Override |
| ------------- | ---------------- | ------------------ |
| FLUX.1        | No               | No                 |
| SDXL          | Yes              | Yes                |
| SD 1.5        | Yes              | Yes                |
| SD 3.5        | Yes              | No                 |
| Z-Image       | No               | No                 |
| Flux.2 Klein  | No               | No                 |
| Wuerstchen v2 | Yes              | No                 |
| Qwen-Image    | Yes              | No                 |
| LTX Video     | No               | No                 |

## Video Generation

| Family     | txt2vid | img2vid |
| ---------- | ------- | ------- |
| LTX Video  | Yes     | Not yet |
| All others | No      | No      |

Video output defaults to APNG (lossless, metadata-rich). Also supports GIF
(256-color, pipe-friendly), WebP (feature-gated), and MP4/H.264 (feature-gated).
Use `--format apng|gif|webp|mp4`. Frame count must be 8n+1 (9, 17, 25, 33, ...).
Dimensions must be multiples of 32.

The recommended LTX default today is `ltx-video-0.9.6-distilled:bf16`. The
`0.9.8` family is available, pulls its spatial upscaler asset, and now runs
the full multiscale refinement path.

## Notes

- ControlNet is currently available only for SD 1.5.
- LoRA adapters are currently available only for FLUX models.
- `--scheduler` applies only to SD 1.5 and SDXL.
- Negative prompts are meaningful for CFG-based families and ignored by FLUX,
  Z-Image, and Flux.2 Klein.

For model size and VRAM fit, see [Models Overview](/models/). For usage
examples, see [Generating Images](/guide/generating).
