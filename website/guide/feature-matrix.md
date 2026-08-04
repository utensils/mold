# Feature Support

This page answers the practical question: which model families support which
features today?

## Quick Picks

| Need                          | Best Starting Point         |
| ----------------------------- | --------------------------- |
| LoRA adapters                 | FLUX.1, SDXL, or Qwen-Image |
| ControlNet                    | SD 1.5                      |
| img2img at 1024 output        | FLUX.1 or SDXL              |
| broadest feature surface      | SD 1.5 or SDXL              |
| best prompt-following quality | FLUX.1 or SD 3.5            |

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

| Family          | Internal family | ControlNet | LoRA |
| --------------- | --------------- | ---------- | ---- |
| FLUX.1          | flux            | No         | Yes  |
| Flux.2 Klein    | flux2           | No         | Yes  |
| LTX-2           | ltx2            | No         | Yes  |
| SD 1.5          | sd15            | Yes        | Yes  |
| SD 3.5          | sd3             | No         | Yes  |
| SDXL            | sdxl            | No         | Yes  |
| Qwen-Image      | qwen-image      | No         | Yes  |
| Qwen-Image-Edit | qwen-image-edit | No         | Yes  |
| Z-Image         | z-image         | No         | Yes  |
| Wuerstchen v2   | wuerstchen      | No         | No   |
| LTX Video       | ltx-video       | No         | No   |

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

| Family     | txt2vid | img2vid | audio2vid | keyframe | retake | lip dub | IC-LoRA | audio track |
| ---------- | ------- | ------- | --------- | -------- | ------ | ------- | ------- | ----------- |
| LTX Video  | Yes     | Not yet | No        | No       | No     | No      | No      | No          |
| LTX-2      | Yes     | Yes     | Yes       | Yes      | Yes    | Yes     | Yes     | Yes         |
| All others | No      | No      | No        | No       | No     | No      | No      | No          |

LTX Video defaults to APNG (lossless, metadata-rich). LTX-2 defaults to MP4 so
it can preserve synchronized audio when requested. Both families also support
GIF, and feature-gated WebP/MP4 outputs where applicable. Use
`--format apng|gif|webp|mp4`. Frame count must be 8n+1 (9, 17, 25, 33, ...).
Dimensions must be multiples of 32 — 64 for LTX-2 lip dub, which always renders
in two stages and takes its frame count and rate from the reference clip.

The recommended LTX default today is `ltx-video-0.9.6-distilled:bf16`. The
`0.9.8` family is available, pulls its spatial upscaler asset, and now runs
the full multiscale refinement path.

## Backend Support

| Family          | CUDA | Metal       | CPU              |
| --------------- | ---- | ----------- | ---------------- |
| FLUX.1 / FLUX.2 | Yes  | Yes         | Yes (slow)       |
| SDXL / SD 1.5   | Yes  | Yes         | Yes              |
| SD 3.5          | Yes  | Yes         | Yes              |
| Z-Image         | Yes  | Yes         | Yes              |
| Wuerstchen v2   | Yes  | Yes         | Yes              |
| Qwen-Image      | Yes  | Yes         | Yes              |
| Qwen-Image-Edit | Yes  | Yes         | Yes              |
| LTX Video       | Yes  | Yes         | Yes              |
| **LTX-2**       | Yes  | **Not yet** | Correctness-only |

::: warning LTX-2 is CUDA-only for real generation
LTX-2 / LTX-2.3 does **not** support Apple Metal in this release. The native
runtime runs on CUDA; the CPU path exists for correctness-oriented coverage and
can be extremely slow. On macOS you can still use every other family through
the Metal backend — LTX-2 is the only family that is currently CUDA-gated.
:::

## Native app surfaces

Both native apps use the family capabilities above and the same generation
request contract. Their platform roles differ intentionally:

| Area      | Desktop                                                                                                                                                                                                                   | iPhone                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Engine    | Built-in local engine plus remote hosts                                                                                                                                                                                   | Remote hosts only                                                                                                                                                          |
| Create    | Capability-driven image/video controls, review-first prepared expansion batches, estimates                                                                                                                                | Same exact-N, frozen-host review lifecycle and independent siblings, full-screen Advanced sheet, optimized for touch                                                       |
| Library   | Unified local/remote grid with persisted top-bar thumbnail sizing, host filters, deduplication, batch provenance, History drawer, and desktop file actions                                                                | Merged saved-host library with batch/source provenance, full-screen image/video, swipe navigation, Use as prompt, and Use as source                                        |
| Models    | Installed/Discover, kind and explicit 18+ NSFW badges, rich detail, install onto any machine still missing the model (Repair once all have it), download progress                                                         | Installed/live union, matching kind/18+ badges and detail, same install-or-repair host picker, pull progress, cancel, load/unload/remove                                   |
| Machines  | This device plus remembered/discovered remote hosts, host detail, automatic routing choices                                                                                                                               | Bonjour, IP/DNS/HTTPS, or Tailscale MagicDNS; explicit generation host and detailed telemetry                                                                              |
| Queue     | Shared multi-host console: live progress, per-job cancel, Pause/Resume, and drag-reorder (`queue.can_reorder`)                                                                                                            | Per-host queue with progress and per-job cancel                                                                                                                            |
| Settings  | Single column: Appearance, Updates, About, a Hosts link into Machines, and collapsed performance/generation/accounts/advanced sections                                                                                    | Mold Studio families (Mold/Safelight), System/Dark/Light, remote-host shortcut, version, and TestFlight update channel                                                     |
| Sequences | **Output** setting inside Create, clip rail with named seams, exact-host **Validate plan**, in-place editing of a finished sequence, Library ▸ History ▸ Sequences, TOML authoring, RunPod provisioning (inside Machines) | Same **Output** setting and clip list with exact-host **Validate plan** and durable recovery; no in-place editing, Sequences history tab, TOML editor, or RunPod workspace |
| Updates   | Signed Stable/Nightly in-app updater on macOS                                                                                                                                                                             | Internal TestFlight builds after eligible `main` CI                                                                                                                        |

See the [Desktop App](/guide/desktop) and [iPhone App](/guide/iphone) guides for
complete workflows.

## Notes

- ControlNet is currently available only for SD 1.5.
- LoRA-capable families are `flux`, `flux2`, `ltx2`, `sd15`, `sd3`, `sdxl`,
  `qwen-image`, `qwen-image-edit`, and `z-image`. Wuerstchen and LTX Video are
  not wired for LoRA yet.
- LTX-2 adds stacked LoRAs plus camera-control presets for the published 19B
  adapters.
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
  audio-to-video, keyframe, retake, lip dub, public IC-LoRA, spatial upscale,
  and temporal upscale workflows.
- LTX-2's multimodal guider exposes optional per-request overrides for STG
  scale/blocks, CFG-rescale, cross-modality scale, and the guidance skip
  stride on the CLI and in web, desktop, and iPhone Advanced video controls. They apply to the
  `two-stage`, `two-stage-hq`, `keyframe`, and `a2-vid` pipelines; unset fields
  keep each pipeline's own constants. The TUI still keeps those guidance and
  pipeline controls at their defaults, but its Video accordion exposes the
  shared `enable_audio` contract as a checkpoint-aware default/on/off choice
  and family-gated `spatial_upscale` / `temporal_upscale` native modes. The
  remaining pipeline, guidance, conditioning-file, and chain-job controls
  remain a separate tracked gap.
- LTX-2 is CUDA-only for real generation: CPU is correctness-only, and Metal is
  not supported in this release.

For model size and VRAM fit, see [Models Overview](/models/). For usage
examples, see [Generating Images](/guide/generating).
