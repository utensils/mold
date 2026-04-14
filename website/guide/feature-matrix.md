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

| Family          | img2img | Inpainting | Edit-family refs |
| --------------- | ------- | ---------- | ---------------- |
| FLUX.1          | Yes     | Yes        | No               |
| SDXL            | Yes     | Yes        | No               |
| SD 1.5          | Yes     | Yes        | No               |
| SD 3.5          | Yes     | Yes        | No               |
| Z-Image         | Yes     | Yes        | No               |
| Flux.2 Klein    | Yes     | Yes        | No               |
| Wuerstchen v2   | Yes     | Yes        | No               |
| Qwen-Image      | Yes     | Yes        | No               |
| Qwen-Image-Edit | No      | No         | Yes              |
| LTX Video       | Not yet | Not yet    | Not yet          |
| LTX-2           | Yes     | No         | Keyframes        |

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
| LTX-2         | No         | Yes  |

## Prompt Conditioning

| Family          | Negative Prompts | Scheduler Override |
| --------------- | ---------------- | ------------------ |
| FLUX.1          | No               | No                 |
| SDXL            | Yes              | Yes                |
| SD 1.5          | Yes              | Yes                |
| SD 3.5          | Yes              | No                 |
| Z-Image         | No               | No                 |
| Flux.2 Klein    | No               | No                 |
| Wuerstchen v2   | Yes              | No                 |
| Qwen-Image      | Yes              | No                 |
| Qwen-Image-Edit | Yes              | No                 |
| LTX Video       | No               | No                 |

## Video Generation

| Family     | txt2vid | img2vid | audio2vid | keyframe | retake | IC-LoRA | audio track |
| ---------- | ------- | ------- | --------- | -------- | ------ | ------- | ----------- |
| LTX Video  | Yes     | Not yet | No        | No       | No     | No      | No          |
| LTX-2      | Yes     | Yes     | Yes       | Yes      | Yes    | Yes     | Yes         |
| All others | No      | No      | No        | No       | No     | No      | No          |

LTX Video defaults to APNG (lossless, metadata-rich). LTX-2 defaults to MP4 so
it can preserve synchronized audio when requested. Both families also support
GIF, and feature-gated WebP/MP4 outputs where applicable. Use
`--format apng|gif|webp|mp4`. Frame count must be 8n+1 (9, 17, 25, 33, ...).
Dimensions must be multiples of 32.

The recommended LTX default today is `ltx-video-0.9.6-distilled:bf16`. The
`0.9.8` family is available, pulls its spatial upscaler asset, and now runs
the full multiscale refinement path.

## Notes

- ControlNet is currently available only for SD 1.5.
- General LoRA adapters are currently available only for FLUX models; LTX-2 has
  its own stacked video-adapter path plus camera-control presets.
- LTX-2 adds stacked LoRAs plus camera-control presets for the published 19B adapters.
- `--scheduler` applies only to SD 1.5 and SDXL.
- Negative prompts are meaningful for CFG-based families and ignored by FLUX,
  Z-Image, and Flux.2 Klein.
- `qwen-image-edit` is a distinct edit family, not a standard img2img mode.
- The CLI and API support multiple ordered input images for `qwen-image-edit`;
  the TUI keeps the edit flow to a single source image in v1.
- `qwen-image-edit` can use quantized `--qwen2-variant` language weights while
  still loading the Qwen2.5-VL vision tower for image conditioning.
- LTX-2 now wires `x2` spatial upscaling across the family, `x1.5` spatial
  upscaling for `ltx-2.3-*`, and `x2` temporal upscaling in the native runtime.
- LTX-2's native CUDA path is validated across text+audio-video, image-to-video,
  audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale, and
  temporal upscale workflows.

For model size and VRAM fit, see [Models Overview](/models/). For usage
examples, see [Generating Images](/guide/generating).
