# LTX-2 / LTX-2.3

LTX-2 is Lightricks' joint audio-video family. In mold it is exposed as a
separate `ltx2` family from the older `ltx-video` checkpoints, with defaults
aimed at synchronized MP4 output and the upstream two-stage / distilled
pipelines.

::: tip Current status
LTX-2 now runs through mold's in-tree Rust runtime. CUDA is the supported
backend for real local generation, CPU is a correctness-oriented fallback, and
Metal is explicitly unsupported for this family. The native CUDA workflow
matrix is validated across 19B/22B text+audio-video, image-to-video,
audio-to-video, keyframe, retake, lip dub, public IC-LoRA, spatial upscale
(`x1.5` / `x2` where published), and temporal upscale (`x2`).
:::

## Supported Models

| Model                        | Path      | Notes                                         |
| ---------------------------- | --------- | --------------------------------------------- |
| `ltx-2-19b-dev:fp8`          | Two-stage | Highest-quality published 19B FP8 checkpoint  |
| `ltx-2-19b-distilled:fp8`    | Distilled | Fastest 19B path, recommended default         |
| `ltx-2.3-22b-dev:fp8`        | Two-stage | High-quality 22B FP8 checkpoint               |
| `ltx-2.3-22b-distilled:fp8`  | Distilled | Fastest 22B path                              |
| `ltx-2.3-22b-dev:bf16`       | Two-stage | Full-quality, trainable 22B reference weights |
| `ltx-2.3-22b-distilled:bf16` | Distilled | Full-precision eight-step 22B checkpoint      |

Bare `ltx-2.3-22b-dev` and `ltx-2.3-22b-distilled` names continue to select
FP8. Choose `:bf16` explicitly for the upstream reference precision used for
training and quality evaluation. Each BF16 checkpoint is about 46.1 GB
(43.0 GiB); a 48 GB+ CUDA card is the practical target for resident weights.
On smaller CUDA cards, mold's native LTX-2 runtime adaptively streams
transformer blocks from host memory, trading speed and substantial system RAM
for lower VRAM use — see [Memory on 24 GB cards](#memory-on-24-gb-cards). The
shared gated Gemma encoder and optional upscaler/LoRA assets add to download and
disk requirements.

## Implemented Request Surface

- Text-to-audio+video with synchronized MP4 output
- Text-to-audio (audio only, no video) via `--pipeline t2a`
- First-frame image-to-video via `--image`, with an
  [optional prompt](#the-prompt-is-optional-for-image-to-video)
- Audio-to-video via `--audio-file`
- Keyframe interpolation via repeatable `--keyframe`
- Retake / partial regeneration via `--video` + `--retake`
- Lip dub / re-voicing via `--pipeline lip-dub` + `--ic-lora-control lipdub`
- IC-LoRA and stacked LoRAs via repeatable `--lora`
- Official IC-LoRA reference controls via `--ic-lora-control`
- Camera-control preset names for the published LTX-2 19B camera LoRAs
- Spatial upscale `x2` across the family and `x1.5` for `ltx-2.3-*`
- Temporal upscale `x2`

## Native Parity Matrix

The in-tree test matrix in `crates/mold-inference/src/ltx2/runtime.rs` keeps
the supported native planning surface explicit without requiring full weights.
It covers the real-runtime route for these published workflow combinations:

| Workflow                      | 19B                                   | 22B / LTX-2.3       | Coverage                                       |
| ----------------------------- | ------------------------------------- | ------------------- | ---------------------------------------------- |
| Text-to-audio+video           | Yes                                   | Yes                 | Planning test + manual CUDA smoke              |
| Text-to-audio (audio only)    | Yes                                   | Yes                 | Block-parity + planning tests + manual CUDA    |
| First-frame image-to-video    | Yes                                   | Yes                 | Planning test                                  |
| Audio-to-video                | Yes                                   | Yes                 | Planning test                                  |
| Keyframe interpolation        | Yes                                   | Yes                 | Planning test                                  |
| Retake / partial regeneration | Yes                                   | Yes                 | Planning test                                  |
| Official IC-LoRA controls     | Union, Pose, Detailer                 | Union, Motion Track | Registry, planning, and request-contract tests |
| Two-stage dev checkpoint      | Yes                                   | Yes                 | Planning test                                  |
| Two-stage HQ                  | Not published as the default 19B path | Yes                 | Planning test                                  |
| Spatial upscale `x2`          | Yes                                   | Yes                 | Planning test                                  |
| Spatial upscale `x1.5`        | Not published                         | Yes                 | Planning test                                  |
| Temporal upscale `x2`         | Yes                                   | Yes                 | Planning test                                  |

The fixed-seed CUDA reference case is tracked in the matrix with the 22B
distilled docs-gallery seed (`424303`). Full numeric comparisons still require
installed gated weights, CUDA, and checked-in reference artifacts; the unit
matrix therefore validates routing and configuration, while manual parity runs
should compare generated contact sheets or clips from that fixed seed.

## Current Constraints

- Default output is `mp4` for this family. `gif`, `apng`, and `webp` are also
  supported, but they are treated as silent exports.
- `x2` spatial upscaling is wired across the family. `x1.5` is wired for
  `ltx-2.3-*` by resolving the published upstream asset on demand.
- `x2` temporal upscaling is wired through the native LTX-2 runtime.
- Camera-control preset aliases are currently published for LTX-2 19B only. For
  LTX-2.3, pass an explicit `.safetensors` path.
- Built-in reference controls require an effective distilled checkpoint. Mold
  rejects dev or architecture-unknown catalog checkpoints before starting a
  download. Raw custom IC-LoRAs remain available through
  `--pipeline ic-lora --lora /path/custom.safetensors`.
- Lip dub takes its length and frame rate from the reference clip — `--frames`
  and `--fps` are overridden, and mold says so when it does. The reference must
  carry an audio track, because its speech is what the dub imitates; a silent
  reference is rejected up front rather than after the checkpoint has loaded.
  Both axes must be multiples of 64 because the pipeline always renders in two
  stages. The frame count is rounded **down** onto the `8k+1` grid, so a
  100-frame reference renders 97 frames.
- Guide video formats are adapter-specific: Union consumes an already
  preprocessed Canny, depth, or pose video; Motion Track consumes colored
  trajectory overlays; Pose consumes a rendered pose video; Detailer consumes
  the ordinary source clip. Preprocessing, attention masks, and multiple
  reference videos are not performed by Mold.
- The Gemma text encoder source is gated on Hugging Face, so you must have
  access approved before `mold pull` will complete.
- When you send source media through `mold serve`, the built-in request body
  limit is `64 MiB`, which covers common inline retake and audio-to-video
  requests.
- Trusted server deployments can use `audio_file_path` and `source_video_path`
  instead of inline base64 for larger local media. Configure `media_roots` or
  `MOLD_MEDIA_ROOTS`; mold canonicalizes the target and rejects missing files,
  directories, traversal, or symlink escapes outside the allow roots.
- On CUDA, explicit LTX-2 unload drops the retained native runtime, safely
  synchronizes pending work, and samples the actual free VRAM without
  invalidating the process-owned primary context. To manually verify OOM
  recovery, run a GPU-resident LTX-2 request, force unload by switching models
  or using the server unload/admin path, then confirm the next LTX-2 request
  logs a fresh runtime load rather than reusing stale allocations.
- On 24 GB Ada GPUs such as the RTX 4090, mold keeps the native runtime on the
  compatible `fp8-cast` path rather than Hopper-only `fp8-scaled-mm`.

### HDR output

The HDR IC-LoRA re-grades a reference video into ARRI LogC3, which is a _log_
signal — an ordinary 8-bit export throws away the range the adapter exists to
produce. Ask for an EXR sequence alongside the video:

```bash
mold run "a neon alley at dusk" \
  --model ltx-2.3-22b-distilled:fp8 \
  --pipeline ic-lora --ic-lora-control hdr \
  --video reference.mp4 \
  --hdr-exr-dir ./shot_exr
```

One `frame_00000.exr` per frame, scene-referred linear, sRGB/Rec.709 primaries
with a D65 white point. Add `--hdr-exr-full-float` for 32-bit samples instead
of the 16-bit default.

The HDR adapter ships pre-computed text embeddings beside its weights, and
mold uses them the way upstream does: they _replace_ prompt encoding rather
than supplementing it. The adapter was trained against that one fixed scene
context, so a prompt of your own would be out of distribution. Mold loads the
companion and skips the Gemma encode entirely, which also makes an HDR render
noticeably faster — about 120 s instead of 180 s for a 25-frame 704x448 clip on
a 4090. Your prompt text is ignored for this control; that is upstream's design,
not a limitation of the port.

EXR export covers **one render**, so it cannot be combined with auto-chaining.
A frame count above the model's per-clip cap is split into stitched clips, and
each clip would overwrite the previous one's frame numbering — so mold refuses
the combination rather than leaving you an empty directory beside a video that
reported success. Render within the per-clip cap, or drop `--hdr-exr-dir`.

The EXR sequence is a **sidecar**, not the gallery artifact: a sequence is many
files and gigabytes, so the ordinary tonemapped video is still written and is
what appears in the Library. Note that LogC3's toe decodes pure black slightly
below zero (about -0.0173); that is faithful to the transfer function and is
written verbatim, so a compositor sees the same values upstream produces.

### Resolution

LTX-2 renders on a 32-pixel grid, under two independent limits: a total-pixel
budget and a **per-axis span**. The span is the one that decides how far the
resolution ladder goes. The checkpoints normalize RoPE pixel positions by
**2048px**, so a longer edge is outside the trained range even when the frame's
total area is small — 3200x512 is 1.64 MP and still out of distribution.

Both limits are **per model**, because they depend on how the checkpoint
renders:

| Checkpoint                                             | Long edge | Total pixels       |
| ------------------------------------------------------ | --------- | ------------------ |
| Renders in one pass (single-file `cv:` / `hf:` builds) | 2048px    | 1920x1088 (2.1 MP) |
| Ships the spatial upsampler (every manifest LTX-2)     | 4096px    | 4096x2176 (8.9 MP) |

A checkpoint that ships the spatial upsampler does not denoise the requested
shape. It renders **stage 1 at half the target**, upsamples that latent x2 with
the learned upsampler, then refines the result with a stage-2 pass over
[latent tiles](#spatial-tiling-spatial-tile) each brought back inside the
trained span. That composition is what lets the output exceed a span the
transformer never saw — and it is also where the ladder stops. 4096px is
exactly the widest target whose halved stage 1 still lands at 2048px; mold
applies one spatial rung, so there is no second halving to rescue anything
wider.

Nothing is composed by hand: pick the output size and mold runs the
composition. `/api/models` carries the per-model `max_pixels`,
`max_axis_pixels`, `dimension_alignment`, and `recommended_dimensions` so
clients do not hardcode any of it, and a checkpoint that cannot compose is
never offered a rung it would reject.

Bigger is not automatically better. A single-pass render at 1080p on the 19B
checkpoint can show edge artifacts, since that is well above the resolution it
was trained at; 1216x704 remains the quality sweet spot.

The 4096px figure belongs to the **x2 rung**, which is what the pipeline
applies by default. `--spatial-upscale x1.5` only divides by 1.5, so its
stage 1 is larger for the same output: a 3840px frame would render stage 1 at
2560px, past the span. x1.5 therefore reaches **3072px** on the long edge, and
mold refuses the combination rather than rendering it — stage 2 tiles the
refinement, never stage 1. Choosing a single-pass pipeline explicitly
(`--pipeline one-stage`, retake, lip dub) drops the ceiling back to 2048px for
the same reason.

#### The output ladder

Every rung is a multiple of 64, so the halved stage-1 shape still lands on the
VAE's 32-pixel latent grid — upstream's own `divisor = 64 if is_two_stage`
rule.

| Rung          | Output    | Stage 1   | Stage-2 tiles |
| ------------- | --------- | --------- | ------------- |
| 720p HD       | 1280x704  | 640x352   | 1             |
| 1080p Full HD | 1920x1088 | 960x544   | 1             |
| 1440p QHD     | 2560x1408 | 1280x704  | 2 (2x1)       |
| 4K UHD        | 3840x2176 | 1920x1088 | 4 (2x2)       |
| 4K DCI        | 4096x2176 | 2048x1088 | 4 (2x2)       |

4K UHD is 3840x**2176**, not 3840x2160: 2160 is not a multiple of 64, and mold
rejects an unrenderable shape rather than quietly resizing it. Upstream hits
the same constraint and resolves it the other way, aligning 2160 down to 2112
or up to 2176 inside its own pipeline.

Portrait is the same rung transposed — 2176x3840 composes and costs exactly
what 3840x2176 does.

#### What this costs, and what has actually been tested

::: warning Above 1080p is unqualified on a 24 GB card.

The rungs above 1080p are **not verified on consumer hardware**, and this
section is deliberately specific about which claim is which.

Each stage-2 tile is a full denoise pass, so a 4K render does four of them
where 1080p does one, on top of a stage 1 that is itself a 1080p render. The
only published VRAM figures for these output shapes are upstream's, and they
are for a **different pipeline** — HDR IC-LoRA, 161 frames, the 22B checkpoint
— so they are a reference point, not mold's requirement
([`hdr_ic_lora.py:780-786`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py)):

| Output    | 80 GB (H100) | 48 GB (A6000) |
| --------- | ------------ | ------------- |
| 1280x720  | 161+ frames  | 161+ frames   |
| 1920x1080 | 161+ frames  | 161+ frames   |
| 2048x1080 | 161+ frames  | 161+ frames   |
| 2560x1440 | 161+ frames  | 137 frames    |
| 3840x2160 | 121 frames   | 49 frames     |
| 4096x2160 | 105 frames   | 49 frames     |

Upstream's own smallest column for 4K is 48 GB. Note also that the module its
`--help` points at for per-configuration estimates,
`ltx_pipelines.utils.vram_budget`, is not published in the repository — the
table is the only figure available.

mold admits these shapes because the composition that reaches them is correct
and testable, not because it has rendered them on a 24 GB card. If a rung does
not fit, the render fails with an out-of-memory error; mold will not silently
drop to a smaller shape.

:::

### Spatial tiling (`--spatial-tile`)

Past 2048px on an axis, stage-2 refinement runs over overlapping **latent
tiles** instead of the whole frame. Each tile is denoised at a shape inside the
trained span, with its RoPE positions renormalized so the tile looks like a
sequence starting at zero and its own noise drawn from `seed + tile_index`, and
the results are recombined with a separable trapezoidal window. The VAE decode
is tiled the same way when a single frame is too large to decode in one pass.
This is upstream's technique, from
[`hdr_ic_lora.py`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py).

`--spatial-tile` (or `MOLD_LTX2_SPATIAL_TILE`, which `mold serve` also reads)
controls it:

| Value              | Effect                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `auto` _(default)_ | Tile only past the 2048px trained span, and only decode-tile when the decode would not otherwise fit.   |
| `off`              | Never tile. A render past the trained span is **refused** rather than quietly degraded.                 |
| `<px>`             | Force tiles of at most `<px>` on each spatial axis, with a 256px overlap. Multiples of 32, at least 64. |
| `<px>:<overlap>`   | As above with an explicit overlap, also a multiple of 32 and smaller than the tile.                     |

`auto` engages exactly at the 2048px span and not one pixel earlier, so every
resolution up to and including 1080p renders as it always has. `off` past the
span is an error, not a warning: the failure mode there is a finished video
with degraded large-scale structure, which nobody would notice was wrong.
Forcing a tile size is mainly a way to compare a tiled render against an
untiled one at a resolution that needs neither.

Two things a tiled refinement gives up. Tiles are refined independently, so a
structure crossing a seam is resolved by two passes that cannot see each other
— wider overlaps hide more of this. And a tiled stage 2 refines **video only**,
carrying stage 1's audio track through unrefined. This is upstream's own
behaviour — each tile runs "a tile-sized `ModalitySpec` for video only (audio
is omitted entirely for HDR)" — and the reason holds independently: a spatial
tile carries no statement about an audio track, so refining one once per tile
would denoise the same track once per tile with no defensible way to recombine
the results.

## The prompt is optional for image-to-video

LTX-2 and the older `ltx-video` family accept an **empty prompt**, but only when
the request already carries something to animate — a source image, keyframes, a
source video, or an `--extend` continuation:

```bash
# Animate a still with no prompt at all
mold run ltx-2-19b-distilled:fp8 --image portrait.png --frames 97 --format mp4
```

This is not a mold extension. The Gemma tokenizer pads to a fixed 1,024-token
context and the embeddings connector replaces every padded position with learned
register embeddings, so the transformer always sees a full context; `""` is a
trained input upstream ships itself.

Two things worth being blunt about:

- **It saves no memory.** The prompt context is a fixed `[1, 1024, 4096]`
  tensor whose size does not depend on how many tokens you typed. Leaving the
  prompt blank will not make a shape fit that otherwise does not.
- **Expect near-static output.** With nothing describing the motion, the model
  tends toward a blink or micro-motion. If you want the subject to _do_
  something, say so.

Everything else keeps the prompt required: text-to-video with no conditioning,
and every image family (FLUX, Flux.2, SD1.5/SDXL/SD3.5, Qwen-Image, Z-Image,
Wuerstchen) even when you pass `--image`. A blank prompt also disables prompt
expansion for that run — mold will not let the expander invent the prompt that
then gets recorded in your metadata — and is not written to prompt history.

Web, desktop, and iPhone Create all enable **Generate** once a source image is
attached to a compatible model and say the same thing in the prompt
placeholder; sequence clips may be left undescribed under the same rule. On
Discord, `/generate`'s `prompt` option is optional when you attach a source
image.

## Memory on 24 GB cards

The 19B and 22B checkpoints are far larger than a consumer card, so the native
runtime plans residency rather than assuming it. Two mechanisms do the work:

- **Admission** reads the checkpoint's own safetensors header and reconstructs
  the plan the engine will build — per-block sizes, the non-block transformer
  weights that streaming never offloads, the bundled video VAE, a token-based
  activation budget for the exact render shape, runtime headroom, and a
  fragmentation margin — before anything is loaded. Each chain stage is priced
  at its own shape, so stage 1 of a two-stage distilled render is charged for
  its half-resolution render, not the final one.
- **Adaptive residency** then keeps as many transformer blocks GPU-resident as
  that budget allows and streams the rest from host memory. When the request
  came through `mold serve`, the planner is bound by the scheduler's admitted
  peak (`min(grant, usable free VRAM)`) instead of expanding to fill whatever
  the card happens to have free.

A shape that cannot run is therefore rejected **before** the two-minute load.
The rejection names the per-device shortfall and, for LTX-2, resolution/frame
combinations that do fit on that card — for example:

```
no device has enough effective VRAM capacity for a safe execution plan:
cuda:0 needs ~26.4 GB of ~23.0 GB usable (needs ~26.4 GB on a 23.0 GB card;
1024x1024 at 65 frames, or 896x896 at 97 frames fits)
```

A predicted peak that no device in the pool could ever hold fails immediately
instead of waiting forever for pressure that will never clear. A CUDA OOM
message carries the same "this shape … fits" advice.

If CUDA still runs out of memory, the denoise stage retries at a reduced
budget, the OOM cooldown is keyed on `(model, shape, GPU)` so a single-GPU host
stops re-admitting the identical failing shape, and one conservative retry at a
smaller grant is offered per shape. A _fatal_ CUDA fault (illegal address,
uncorrectable ECC, launch failure) is never retried — the worker is quarantined
and the process stops, as everywhere else in mold.

Practical guidance for a 24 GB card: `ltx-2-19b-distilled:fp8` is the intended
path, and 1024x1024 x 97 frames is close to the ceiling. Lower the resolution
before the frame count if you need headroom — attention cost grows with the
square of the token count, and tokens scale with area × latent frames.

## Examples

```bash
# Fast default: text to synchronized MP4
mold run ltx-2-19b-distilled:fp8 \
  "cinematic close-up of rain on a neon taxi window" \
  --frames 97 \
  --format mp4

# Audio-to-video
mold run ltx-2-19b-distilled:fp8 \
  "paper cutout forest reacting to a violin solo" \
  --audio-file ./solo.wav \
  --format mp4

# Keyframe interpolation
mold run ltx-2-19b-distilled:fp8 \
  "a drone shot over volcanic cliffs" \
  --pipeline keyframe \
  --frames 97 \
  --keyframe 0:./start.png \
  --keyframe 96:./end.png

# Camera-control preset
mold run ltx-2-19b-distilled:fp8 \
  "a lantern-lit cave entrance" \
  --camera-control dolly-in \
  --format mp4

# Official Union control. The guide must already be a frame-aligned
# Canny, depth, or pose video; Mold does not preprocess it.
mold run ltx-2-19b-distilled:fp8 \
  "a dancer follows the guide" \
  --ic-lora-control union \
  --video ./canny-guide.mp4 \
  --format mp4

# LTX-2.3 Motion Track consumes a video with colored trajectory overlays.
mold run ltx-2.3-22b-distilled:fp8 \
  "the drone follows the marked trajectory" \
  --ic-lora-control motion-track \
  --video ./trajectory-overlay.mp4 \
  --format mp4

# Retake a source clip over a time range
mold run ltx-2-19b-distilled:fp8 \
  "replace the actor with a chrome mannequin" \
  --pipeline retake \
  --video ./source.mp4 \
  --retake 1.5:3.5 \
  --format mp4

# Lip dub: re-voice a clip of someone speaking. The reference video supplies
# the frame count, the frame rate, and the voice; the prompt supplies the new
# line. Width and height must be multiples of 64.
mold run ltx-2.3-22b-distilled:fp8 \
  "she says: the harbour freezes over every winter" \
  --ic-lora-control lipdub \
  --video ./speaker.mp4 \
  --width 704 --height 448 \
  --format mp4

# Spatial upscale on a published LTX-2.3 asset
mold run ltx-2.3-22b-distilled:fp8 \
  "red sports car in rain, cinematic reflections" \
  --spatial-upscale x1.5 \
  --format mp4
```

## Text-to-audio

`--pipeline t2a` renders sound with no video at all — speech, ambience, or
music straight from a prompt. It is upstream's `T2AOneStagePipeline`.

```bash
mold run ltx-2.3-22b-dev:fp8 \
  "heavy rain on a tin roof, distant thunder" \
  --pipeline t2a \
  --frames 121 --fps 24 \
  --output rain.wav
```

Things worth knowing:

- **Duration comes from `--frames` / `--fps`,** the same pair a video render
  uses: 121 frames at 24 fps is 5.04 seconds. There is no separate duration
  flag, because the model's temporal budget is expressed in the video clock
  either way.
- **The output is a 16-bit PCM stereo WAV** at the vocoder's own rate — 24 kHz,
  or 48 kHz on checkpoints that ship the bandwidth-extension stage. `--format`
  defaults to `wav` for this pipeline and rejects every other value; `wav`
  likewise requires `--pipeline t2a`.
- **It needs a checkpoint with audio.** A video-only LTX-2 export has no audio
  VAE or vocoder, and the request is refused before any weights load.
- **Steps default to the non-distilled schedule** — 40 on LTX-2 19B, 30 on
  LTX-2.3 22B. The family's usual 8-step default is tuned for the distilled
  _video_ ladder and produces hiss here, so a smaller `--steps` is raised to
  the preset default and the run says so. A larger value is used as given.
- **No conditioning inputs.** `--image`, `--video`, `--audio-file`,
  `--keyframe`, `--retake`, the upscalers, and `--upscale` are all rejected
  rather than silently ignored.
- **No auto-chaining.** A long video request is split into clips because one
  GPU can only hold so many video latents; audio has neither clips nor that
  ceiling, so a large `--frames` stays a single request.
- **`--modality-scale` must stay `1.0`.** Cross-modal guidance steers the
  audio↔video attention, and there is no video branch here.
- **`--batch N` renders N takes,** each written as it lands under its own
  index (`take-0.wav`, `take-1.wav`, …) so a later take failing cannot
  discard the ones already rendered.

Audio prints land in the gallery like any other output, with a rendered
waveform as the tile. Web, desktop, and the CLI all play or save them; the
Library's kind filter gains an **Audio** chip.

Audio-only generation loads just the `audio_*` half of the checkpoint — about
a quarter of the per-block parameters, with both cross-modal attentions gone —
so it fits comfortably on a 24 GB card without block streaming.

## Advanced guidance controls

LTX-2's multimodal guider takes more than the base guidance scale. Each
pipeline ships tuned constants for spatiotemporal guidance (STG), CFG-rescale,
audio/video cross-modality guidance, and a guidance skip stride; the flags
below override one constant each for a single request. Anything you leave
unset keeps the pipeline's own value, so an unflagged render is bit-for-bit
what it was before these flags existed.

| Flag                   | Default                                                  | What it does                                                                                                     |
| ---------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--stg-scale`          | `1.0` (two-stage, keyframe, a2-vid) · `0` (two-stage HQ) | Strength of the perturbed-attention pass. Higher adds motion structure and detail; too high destabilizes motion. |
| `--stg-blocks`         | `29` on LTX-2 19B · `28` on LTX-2.3 22B                  | Which transformer blocks the perturbed pass skips. Earlier blocks perturb harder. Comma-separated, up to 8.      |
| `--rescale-scale`      | `0.7` (two-stage, keyframe, a2-vid) · `0.45`/`1.0` (HQ)  | CFG-rescale between 0 and 1. Raise it when strong guidance washes out contrast.                                  |
| `--modality-scale`     | `3.0`                                                    | Audio ↔ video cross-modality guidance. `1.0` turns the isolated-modality pass off.                               |
| `--guidance-skip-step` | `0` (every step)                                         | With `n`, guidance is applied every `n + 1` steps and the conditional prediction is taken otherwise.             |

```bash
# Softer STG on an earlier block, with a stronger rescale
mold run ltx-2-19b-distilled:fp8 \
  "handheld shot through a night market" \
  --pipeline two-stage \
  --stg-scale 0.6 --stg-blocks 20,29 --rescale-scale 0.9 \
  --format mp4
```

Three limits are worth knowing before you reach for these:

- **Only pipelines that run the multimodal guider read them.** That is
  `two-stage`, `two-stage-hq`, `keyframe`, `a2-vid`, and `t2a`. The
  `distilled`, `one-stage`, `ic-lora`, and `retake` pipelines pin guidance to
  their own path and the overrides are inert there. `t2a` additionally rejects
  a `--modality-scale` other than `1.0`, since it has no video branch.
- **They never switch a guider on.** `a2-vid` runs audio positive-only by
  design; an override tunes the video guider and leaves the audio guider off
  rather than buying an extra transformer pass you did not ask for.
- **Sequences ignore them.** Chain stages render through their own pipeline
  constants, so `mold run` warns and continues when a guidance flag meets a
  chained request.

Enabling STG or cross-modality guidance where the pipeline had it off adds a
forward pass per denoise step: expect a slower render and more VRAM.

## Chained video output

The LTX-2 distilled pipeline maxes out at 97 pixel frames per clip (13 latent
frames after the VAE's 8× temporal compression — `8 × 12 + 1 = 97` satisfies the
`8k+1` frame-grid constraint). For anything longer, mold renders a _chain_: the
request is split into N sub-clips, each generated back-to-back, and stitched
into a single MP4 at the end. mold keeps the last few frames of clip _N_'s
final latents in memory and threads them directly into clip _N+1_'s
conditioning, skipping a VAE encode/decode round-trip so the continuation
stays visually coherent.

`mold run` routes automatically: when `--frames` is `≤ 97` you stay on the
single-clip path; above 97 the request is rewritten into a chain and dispatched
to the new `/api/generate/chain/stream` endpoint. Chaining is supported for
LTX-2 19B and 22B distilled today. Other model families reject `--frames` past
their own single-request ceiling with an actionable error rather than silently
over-producing.

::: tip 97 is a routing default, not the model's ceiling
LTX-2's real single-request limit is a **20-second runtime budget** — 484 frames
at 24 fps (see [Frame ceiling](#frame-ceiling) below). 97 is simply the clip
size that fits comfortably on one consumer GPU, so auto-chaining uses it. Raise
`--clip-frames` to render one long coherent clip instead of a stitched
sequence: `--frames 241 --clip-frames 241` gives a single 10-second shot with no
seams, at the cost of far more VRAM and time.
:::

```console
$ mold run ltx-2-19b-distilled:fp8 "a cat walking through autumn leaves" \
    --image cat.png --frames 400

→ Chain mode: 400 frames → 5 stages × 97 frames (tail 4)
Chain [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 385/385 frames (stages 5)
  Stage 1  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 8/8 steps
  Stage 2  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 8/8 steps
  Stage 3  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 8/8 steps
  Stage 4  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 8/8 steps
  Stage 5  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 8/8 steps
✓ Saved: mold-ltx-2-19b-distilled-<ts>.mp4 (400 frames, 1216x704, 24 fps)
✓ Done — ltx-2-19b-distilled:fp8 in 226.8s (400 frames, seed: 42)
```

### Motion-tail carryover

`--motion-tail N` (default 4) controls how many trailing pixel frames of each
clip are reused as latent-space conditioning for the next. Instead of decoding
the prior clip's last frame back to RGB and re-encoding it through the VAE as
a new `source_image`, mold narrows the final denoise tensor along its time
axis and patchifies those latent tokens directly into the next stage's
`StageVideoConditioning` — so the handoff never leaves latent space. At stitch
time, every stage after the first drops its leading `N` output frames because
those are the overlap region shared with the prior clip.

- `--motion-tail 0` — hard concatenation, no overlap. Visible seams are common
  at clip boundaries; useful when you _want_ discrete shots.
- `--motion-tail 4` — the default. One latent frame of carryover at `fps=24`
  gives the transformer enough temporal context to continue motion, object
  identity, and lighting across the seam without wasting new frames.
- Higher values buy more seam-smoothing at the cost of fewer fresh pixel
  frames per clip. Must stay strictly below `--clip-frames`.

### Flags

| Flag              | Default       | Description                                                                         |
| ----------------- | ------------- | ----------------------------------------------------------------------------------- |
| `--frames N`      | model default | Total stitched length. Above `--clip-frames`, auto-chains.                          |
| `--clip-frames N` | `97`          | Per-clip length. Must be `8k+1`; clamped to the model's real budget with a warning. |
| `--motion-tail N` | `4`           | Pixel-frame overlap between clips. `0` disables carryover.                          |

### Continuing an existing video

`--extend` continues a clip you already have instead of starting a new one:

```console
$ mold run ltx-2-19b-distilled:fp8 "the car rounds the headland into fog" \
    --extend coast.mp4 --frames 97 --extend-overlap 17

✓ Saved: mold-ltx-2-19b-distilled-<ts>.mp4 (321 frames, 704x480, 24 fps)
```

The delivered file is the original followed by the new footage. `--frames` is
the length of the _rendered continuation_, and its leading `--extend-overlap`
frames re-render the source tail as motion context — those are dropped from the
result, so the run appends `frames - overlap` new frames.

| Flag                 | Default | Description                                                                              |
| -------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `--extend PATH`      | —       | Video to continue. LTX-2 only.                                                           |
| `--extend-overlap N` | `17`    | Pixel frames of the source tail used as motion context. Must be `8k+1` and `< --frames`. |

Constraints, all enforced before any GPU work:

- The continuation must render at the source clip's **resolution and frame
  rate**. Mold rejects a mismatch rather than rescaling mid-video.
- `--extend` cannot be combined with `--image`, `--video`, or `--keyframe`.
  Each of those claims authority over the same opening frames.
- The overlap must sit on the `8k+1` grid so the carried frames re-encode
  cleanly through the video VAE, and must be strictly below `--frames` so the
  continuation adds at least one new frame.

Under the hood this is the same motion-tail handoff a sequence uses between
clips — the carryover simply comes from a file instead of the previous stage.
To chain several continuations, extend the result again.

### Frame ceiling

LTX-2's single-request ceiling is a **duration**, not a frame count. The
checkpoints ship `pos_embed_max_pos = 20`, and the temporal RoPE axis is
normalized in seconds — the pixel-frame coordinate is divided by fps before
`max_pos` normalization. So the budget is 20 seconds of runtime:

```
max_frames = 20 * fps + 4      (capped at 604 frames)
```

| fps | Ceiling    | Runtime |
| --- | ---------- | ------- |
| 6   | 124 frames | ~20 s   |
| 12  | 244 frames | ~20 s   |
| 24  | 484 frames | ~20 s   |
| 30  | 604 frames | ~20 s   |

`GET /api/models` advertises `max_frames` at the model's own `default_fps`,
plus `max_runtime_seconds` so clients can recompute it when the user changes
fps. `--temporal-upscale x2` does **not** extend the budget: it halves the
stage-1 frame count _and_ the stage-1 fps, so stage 1 renders the same runtime
at half the frame rate.

Whether a long single clip actually _fits_ is a separate question from whether
the model allows it — attention cost grows with the square of the token count,
so a 481-frame render at 1216x704 needs far more VRAM than most cards have.
The validator permits it; auto-chaining stays at 97-frame clips by default.

When the final clip over-produces (stage math rarely lands exactly on
`total_frames`), mold trims from the tail so the user-anchored starting image
at the head stays intact.

### v1 constraints

- **LTX-2 19B and 22B distilled only.** Other LTX-2 / LTX-Video variants and
  every image-family model reject `--frames` above their single-clip budget.
- **Single GPU per chain.** Every stage runs on the GPU the engine was loaded
  onto — multi-GPU stage fan-out is a v2 movie-maker feature.
- **Fail-closed.** If any stage errors, the whole chain returns `502` and
  nothing is written to the gallery. There is no partial-resume in v1.
- **Multiple CLI authoring modes.** A large `--frames` request still replicates
  the main prompt across stages, but `mold run --prompt ... --prompt ...` builds
  one stage per prompt and `mold run --script shot.toml` sends the canonical
  `mold.chain.v1` script with per-stage prompts, source images, frame counts,
  and transitions.

The rest of the LTX-2 surface — `--image`, `--audio-file`, `--lora`,
`--camera-control`, `--spatial-upscale`, `--temporal-upscale`, and so on —
applies to chain renders the same way it applies to single-clip renders. The
exception is the advanced guidance overrides above: chain stages keep their
pipeline's guider constants, and `mold run` says so rather than pretending the
flags landed. An
`--image` supplied on the CLI lands on `stages[0]` and is carried forward by
the motion-tail latents from there.

## Example Clips

Here are a few longer LTX-2 examples rendered with mold. The docs page embeds
lightweight `webm` previews so the examples load quickly in the browser.

<div class="gallery-grid">
<figure>

<video controls muted loop playsinline preload="metadata" src="/gallery/ltx2/ltx2-docs-candidate-lighthouse-640x384-97f-12fps-seed424301.webm"></video>

**ltx-2-19b-distilled:fp8** — 97 frames, 640x384, 12 fps

_Storm-lashed lighthouse at dusk, gliding coastal pass, thunder, rain, wind,
and surf._

</figure>
<figure>

<video controls muted loop playsinline preload="metadata" src="/gallery/ltx2/ltx2-docs-candidate-subway-drummer-640x384-97f-12fps-seed424302.webm"></video>

**ltx-2-19b-distilled:fp8** — 97 frames, 640x384, 12 fps

_Subway-tunnel drummer performance, orbiting concert camera, percussion, reverb,
and distant train rumble._

</figure>
<figure>

<video controls muted loop playsinline preload="metadata" src="/gallery/ltx2/ltx2-docs-candidate-seaplane-640x384-97f-12fps-seed424303.webm"></video>

**ltx-2.3-22b-distilled:fp8** — 97 frames, 640x384, 12 fps

_Red seaplane over an Arctic fjord at sunrise, wingtip bank, spray off the
floats, propeller engine, wind, and water hiss._

</figure>
</div>

## Notes

- `--audio` and `--no-audio` control whether the returned MP4 keeps the audio
  track. If you explicitly choose `gif`, `apng`, or `webp`, mold exports a
  silent animation.
- `--lora` is repeatable for this family. The single legacy `lora` request
  field is still populated for backward compatibility, but the LTX-2 runtime
  uses the stacked `loras` list.
