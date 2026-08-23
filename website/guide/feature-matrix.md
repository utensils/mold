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
| Wan Video       | Not yet | Not yet    | Not yet          |

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
| Wan Video       | wan             | No         | Yes  |

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
| Wan Video       | Yes              | No                 |

## Video Generation

| Family     | txt2vid | img2vid | audio2vid | keyframe   | retake | lip dub | IC-LoRA | audio track |
| ---------- | ------- | ------- | --------- | ---------- | ------ | ------- | ------- | ----------- |
| LTX Video  | Yes     | Not yet | No        | No         | No     | No      | No      | No          |
| LTX-2      | Yes     | Yes     | Yes       | Yes        | Yes    | Yes     | Yes     | Yes         |
| Wan Video  | Yes     | Yes     | No        | First/last | No     | No      | No      | No          |
| All others | No      | No      | No        | No         | No     | No      | No      | No          |

LTX Video, LTX-2, and Wan all default to MP4 — LTX-2 so it can preserve
synchronized audio when requested; Wan renders video only and has no audio
path. A build compiled without the `mp4` feature falls back to APNG. GIF and
APNG remain available for all three families, plus feature-gated WebP. Use
`--format apng|gif|webp|mp4`. Wan keyframing is first/last-frame interpolation
only (`--image` + `--last-image`); other keyframe layouts are refused at
admission. Frame grids are per family: LTX Video and LTX-2
take 8n+1 frame counts (9, 17, 25, 33, ...) with dimensions in multiples of
32 — 64 for LTX-2 lip dub, which always renders in two stages and takes its
frame count and rate from the reference clip. Wan takes 4n+1 frame counts
(49, 53, 81, ...) with dimensions in multiples of 16, except `wan22-ti2v-5b`,
whose 2.2 VAE requires multiples of 32.

The `--output` extension outranks those family defaults: `mold run … -o clip.gif`
writes a real GIF even where the family would have picked MP4. An extension this
binary cannot encode — `.mp4` without the `mp4` feature, `.webp` without `webp` —
is refused before any weight is read rather than filled with another container's
bytes, and an explicit `--format` that disagrees with the filename is reported
instead of silently overriding it. `--output -` writes to stdout and claims no
extension, so it keeps whatever container the family resolved.

The recommended LTX default today is `ltx-video-0.9.6-distilled:bf16`. The
`0.9.8` family is available, pulls its spatial upscaler asset, and now runs
the full multiscale refinement path.

## Backend Support

| Family          | CUDA | Metal            | CPU              |
| --------------- | ---- | ---------------- | ---------------- |
| FLUX.1 / FLUX.2 | Yes  | Yes              | Yes (slow)       |
| SDXL / SD 1.5   | Yes  | Yes              | Yes              |
| SD 3.5          | Yes  | Yes              | Yes              |
| Z-Image         | Yes  | Yes              | Yes              |
| Wuerstchen v2   | Yes  | Yes              | Yes              |
| Qwen-Image      | Yes  | Yes              | Yes              |
| Qwen-Image-Edit | Yes  | Yes              | Yes              |
| LTX Video       | Yes  | Yes              | Yes              |
| **LTX-2**       | Yes  | Yes              | Correctness-only |
| Wan Video       | Yes  | Correctness-only | Correctness-only |

::: tip LTX-2 Metal qualification
LTX-2 / LTX-2.3's Apple Metal path is performance-qualified: BF16 transformer
compute, fused attention, streamed FP8 lookup-table widening, and temporal VAE
chunks, measured end-to-end on the 19B and 22B distilled FP8 tiers on Apple
Silicon. Metal remains slower than a comparable CUDA card — streaming trades
speed for fitting the model in unified memory.
:::

Wan Video's Metal path is correctness-qualified (family-scoped BF16, chunked
attention; fp8-scaled Wan checkpoints are refused on Metal), pending
performance UAT.

## Native app surfaces

Both native apps use the family capabilities above and the same generation
request contract. Their platform roles differ intentionally:

| Area      | Desktop                                                                                                                                                                                                                   | iPhone                                                                                                                                                                                                                                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Platforms | macOS, Linux, and Windows (x64 and ARM64)                                                                                                                                                                                 | iPhone and iPad                                                                                                                                                                                                                                                                                                  |
| Engine    | Built-in local engine (Metal on macOS, CUDA on Linux and x64 Windows, otherwise CPU) plus remote hosts                                                                                                                    | Remote hosts only                                                                                                                                                                                                                                                                                                |
| Create    | Capability-driven image/video controls, review-first prepared expansion batches, estimates                                                                                                                                | Same exact-N, frozen-host review lifecycle and independent siblings, full-screen Advanced sheet, optimized for touch                                                                                                                                                                                             |
| Library   | Unified local/remote grid with persisted top-bar thumbnail sizing, host filters, deduplication, batch provenance, History drawer, and desktop file actions                                                                | Merged saved-host library with persisted pinch-to-resize thumbnail sizing, Prints/Collections/Trash scopes, favorites/tag chips, titles via the viewer Info sheet, per-host trash retention on host detail, batch/source provenance, full-screen image/video, swipe navigation, Use as prompt, and Use as source |
| Models    | Installed/Discover, kind and explicit 18+ NSFW badges, rich detail, install onto any machine still missing the model (Repair once all have it), download progress                                                         | Installed/live union, matching kind/18+ badges and detail, same install-or-repair host picker, pull progress, cancel, load/unload/remove                                                                                                                                                                         |
| Machines  | This device plus remembered/discovered remote hosts, host detail, automatic routing choices                                                                                                                               | Bonjour, IP/DNS/HTTPS, or Tailscale MagicDNS; explicit generation host and detailed telemetry                                                                                                                                                                                                                    |
| Queue     | Shared multi-host console: live progress, per-job cancel, Pause/Resume, and drag-reorder (`queue.can_reorder`)                                                                                                            | Per-host queue with progress and per-job cancel                                                                                                                                                                                                                                                                  |
| Settings  | Single column: Appearance, Updates, About, a Hosts link into Machines, and collapsed performance/generation/accounts/advanced sections                                                                                    | Mold Studio families (Mold/Safelight), System/Dark/Light, remote-host shortcut, version, and TestFlight update channel                                                                                                                                                                                           |
| Sequences | **Output** setting inside Create, clip rail with named seams, exact-host **Validate plan**, in-place editing of a finished sequence, Library ▸ History ▸ Sequences, TOML authoring, RunPod provisioning (inside Machines) | Same **Output** setting and clip list with exact-host **Validate plan** and durable recovery; no in-place editing, Sequences history tab, TOML editor, or RunPod workspace                                                                                                                                       |
| Updates   | Signed Stable/Nightly in-app updater on macOS; Linux and Windows are replaced manually                                                                                                                                    | Internal TestFlight builds after eligible `main` CI                                                                                                                                                                                                                                                              |

See the [Desktop App](/guide/desktop) and [iPhone App](/guide/iphone) guides for
complete workflows.

## Notes

- ControlNet is currently available only for SD 1.5.
- LoRA-capable families are `flux`, `flux2`, `ltx2`, `sd15`, `sd3`, `sdxl`,
  `qwen-image`, `qwen-image-edit`, `wan`, and `z-image`. Wuerstchen and LTX
  Video are not wired for LoRA yet.
- Wan adapters cover low-rank pairs and full-weight `.diff`/`.diff_b` deltas:
  on bf16 safetensors they merge as the weights are read, on GGUF they apply
  as a parallel branch at full precision. fp8-scaled Wan checkpoints refuse
  adapter stacks rather than re-round their weights.
- LTX-2 adds stacked LoRAs plus camera-control presets for the published 19B
  adapters.
- `--scheduler` applies only to SD 1.5 and SDXL.
- Negative prompts are meaningful for CFG-based families and ignored by FLUX,
  Z-Image, and Flux.2 Klein. Wan checkpoints were tuned against a specific
  negative prompt, which mold applies automatically when a request leaves it
  unset; `/api/models` advertises it per model (`default_negative_prompt`),
  every surface prefills it, and clearing the field (or `--no-negative`)
  sends an explicit empty negative instead.
- `qwen-image-edit` is a distinct edit family, not a standard img2img mode.
- The CLI and API support multiple ordered input images for `qwen-image-edit`;
  the TUI keeps the edit flow to a single source image in v1.
- `qwen-image-edit` can use quantized `--qwen2-variant` language weights while
  still loading the Qwen2.5-VL vision tower for image conditioning.
- LTX-2 now wires `x2` spatial upscaling across the family, `x1.5` spatial
  upscaling for `ltx-2.3-*`, and `x2` temporal upscaling in the native runtime.
- LTX-2's native CUDA path is validated across text+audio-video, text-to-audio,
  image-to-video, audio-to-video, keyframe, retake, lip dub, public IC-LoRA,
  spatial upscale, and temporal upscale workflows.
- LTX-2 renders audio on its own with `--pipeline t2a` (`pipeline: "t2a"`):
  no video, duration from `frames`/`fps`, and a 16-bit PCM stereo `wav`
  artifact that lands in the gallery with a rendered waveform tile.
- LTX-2's multimodal guider exposes optional per-request overrides for STG
  scale/blocks, CFG-rescale, cross-modality scale, and the guidance skip
  stride on the CLI and in web, desktop, iPhone, and TUI Advanced video controls. They apply to the
  `two-stage`, `two-stage-hq`, `keyframe`, `a2-vid`, and `t2a` pipelines; unset
  fields keep each pipeline's own constants. The TUI uses bounded keyboard
  cycles for the numeric guidance values and validates comma-separated STG
  blocks before closing the editor; untouched values remain absent from the
  request. Its Video accordion also exposes the shared `enable_audio` contract
  as a checkpoint-aware default/on/off choice, family-gated
  `spatial_upscale` / `temporal_upscale` native modes, and the source-free
  `one-stage`, `two-stage`, `two-stage-hq`, and `distilled` recipes while Auto
  leaves `pipeline` absent. Conditioning-file modes, the audio-only `t2a`
  pipeline, and chain-job administration remain a separate tracked gap.
- Completed LTX-2 videos report the runtime-resolved pipeline separately from
  the requested Auto/explicit choice. Server, CLI, and TUI saves preserve that
  response in gallery metadata, and web, desktop, iPhone, and TUI Library
  details show it when present; older and non-LTX prints simply omit the row.
- LTX-2 is performance-qualified on CUDA and Apple Metal (19B/22B distilled
  FP8 tiers, checkpoint-backed); CPU stays correctness-only.
- Library organization — titles, ♥ favorites, tags, collections, and a trash
  with per-host retention (`gallery.trash_retention_days`) — is stored per
  host in that host's `mold.db` and merged across hosts by every client. The
  web Library exposes it as the **Prints | Collections | Trash** scope
  control, the filter-chip row, the print viewer's editable aside, and the
  selection bar's Add to collection / Tag / Favorite / Trash actions; Settings
  ▸ Library and Machines ▸ host edit retention. Hosts that do not advertise
  `gallery.organize` / `gallery.trash` show none of it and keep permanent
  deletes. See [Generating ▸ Browser UI](/guide/generating#browser-ui).
- Creation-time filing ("File under") lets a print arrive already organized:
  `GenerateRequest` and the chain body carry additive `title`, `tags`, and
  `collection`, seeded onto the gallery row once, as it is created. The CLI
  spells it `mold run --title/--tag/--collection` (with `--no-auto-tag` and
  the `generate.auto_tag_title` preference); web, desktop, and iPhone Create
  render a capability-gated **File under** group, and the TUI keeps it as the
  last Create ▸ Advanced section. A sequence files the stitched print only,
  batch and prepared siblings inherit their parent's filing, and a filing the
  host cannot apply is dropped and reported on `x-mold-request-warning`
  rather than failing the render. See
  [Generating ▸ File under](/guide/generating#file-under).

For model size and VRAM fit, see [Models Overview](/models/). For usage
examples, see [Generating Images](/guide/generating).
