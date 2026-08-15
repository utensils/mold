# Generating Images

The web Create composer shows an advisory peak-VRAM estimate for the selected
machine before you submit. Video models expose their supported camera-motion
controls under Advanced, and a durable sequence remains attached after a page
reload so its server-side progress does not disappear from the workspace.
Setting **Output** to **Sequence** filters the model picker to compatible
installed video checkpoints and selects one when possible, remembering your
one-shot model for the way back. If none are installed, **Browse video models**
opens Models → Discover with the Video and Models filters already applied.
Two-stage LTX-2 dev checkpoints render single clips; multi-stage sequences
require a distilled or one-stage checkpoint.

On phone-sized web views, Create keeps every interactive target at least 44px
high and editable fields at a zoom-safe 16px.

While a print renders, the web canvas develops it live: for families that
stream latent previews (FLUX.1, Flux.2, Z-Image) the forming image appears
under the film-grain wash, its blur tightening step by step while the grain
thins away, on a bed matching the print's aspect ratio. The progress ring and
stage line cover the bed only until the first preview arrives. Once a run
completes, the Seed section offers **lock last (seed)** to pin that print's
seed for the next generate.

## Basic Usage

```bash
# Default model (flux2-klein:q8 — fast 4-step, Apache 2.0, fully ungated)
mold run "a red apple on a wooden table"

# Explicit model selection
mold run flux-dev:q4 "a photorealistic landscape at golden hour"

# Custom output path
mold run "cyberpunk cityscape" -o cityscape.png

# Reproducible output (same seed = same image)
mold run "a cat" --seed 42

# Custom dimensions (multiples of 16)
# See each model's recommended dimensions to avoid warnings
mold run "a banner" --width 1024 --height 512

# Batch generation (multiple images with incrementing seeds)
mold run "abstract art" --batch 4 --seed 100

# JPEG output
mold run "a sunset" --format jpeg -o sunset.jpg
```

Here's what that looks like — FLUX Schnell Q8, seed 42:

![Snow leopard — FLUX Schnell](/gallery/flux-schnell-leopard.png)

Need a quick answer on family capabilities or expected speed? See
[Feature Support](/guide/feature-matrix) and [Performance](/guide/performance).

## Recommended Dimensions

Each model family has a set of recommended dimensions that produce the best
results. Using non-recommended dimensions will trigger a warning (generation
still proceeds). All dimensions must be multiples of 16.

```bash
# Square (works with all families)
mold run "a cat" --width 1024 --height 1024

# Landscape (FLUX, Flux.2 Klein)
mold run flux2-klein "a panorama" --width 1024 --height 576

# Portrait (SDXL, SD 3.5, Qwen-Image)
mold run sdxl-turbo "a portrait" --width 832 --height 1216
```

See each [model family page](/models/) for the full list of recommended
dimensions and aspect ratios.

## Image Editing

`qwen-image-edit-2511:*` is a distinct edit family, not standard img2img. It
uses one or more ordered `--image` inputs, supports negative prompts, and
derives default output dimensions from the first input image when you omit
`--width` and `--height`.

```bash
# Single-image edit
mold run qwen-image-edit-2511:q4 \
  --image ./chair.png \
  "turn this fabric chair into dark red leather"

# Multi-image edit
mold run qwen-image-edit-2511:q4 \
  --image ./chair.png \
  --image ./swatch.png \
  "make Picture 1 match the leather color and finish from Picture 2"
```

Use regular img2img families when you need `--strength`-based denoising.
Use `qwen-image-edit` when you want instruction-following edits against one or
more reference images.

In Mold Studio's web Create workspace, open **Advanced → Edit images** to attach
the ordered target and reference images. Ordinary source-image families instead
show a strength slider plus all five fit policies — labelled **Denoise
strength** for SD-lineage img2img (higher = more change) and **Source
strength** for LTX-2 image-to-video (higher keeps more of the source; 1.0
pins the opening frame); SD1.5 also suggests installed
ControlNet checkpoints while allowing a custom checkpoint name.

On a phone, Create follows one vertical workflow: prompt and style, model and
core controls, Generate, the developing/result canvas, then recent prints.
Advanced options remain in the mobile sheet so the primary flow stays compact.

For a multi-prompt video, set **Output** — the control beside Model in the
Create settings column — from **One shot** to **Sequence**. Multi-clip video is
a setting, not a separate page: the composer becomes a clip rail. Mold starts
with two clips, requires a description for each, and joins them with seam pills
that name each transition in words — **Smooth**, **Cut**, or **Fade 8f**, with
LTX-Video's zero-overlap joins reading **Join**. Clicking a seam pill
opens the seam editor with its three teaching rows and the fade-length stepper.
New clips take their frame count from the selected model's own advertised
default rather than a fixed constant, and the summary shows the stitched
duration before you generate. Switching Output back to One shot keeps clip 1 as
the prompt and parks the rest — nothing is erased. **Sequence tools** contains
TOML import/export and other script-oriented controls. On a multi-machine web
setup, the durable job and its live progress stay on the machine selected by
**Run on**.

Sequences queue in the same activity strip as ordinary prints, with watch,
cancel, and resume. Once one settles it leaves the strip: the video lands in the
Create canvas with **Edit sequence** and **Show in library**, its print is in the
Library, and its job record is in **Library ▸ History ▸ Sequences** — which is
also where the host-scoped **Clear inactive** and **Clean up disk** actions live.

A finished sequence can be edited in place. Its clips reload onto the rail and
each pill shows whether that clip is cached (✓) or will re-render (↻) as you
edit; **Update sequence** re-renders only from the earliest changed clip.
Changing a transition type or a fade length re-renders nothing at all — those
are applied when the video is stitched. From a sequence print in the Library,
**Edit sequence** is the primary action and re-enters the original job on the
machine that produced it so rendered clips stay cached. **Duplicate as new**
starts a fresh sequence from the recorded clips, telling you how many it
restored and naming anything a print does not record.

On every surface, a **↺ Reset** in the Create settings header restores the
generation settings to the selected model's defaults — shape, resolution,
detail, prompt strength, seed, and the Advanced groups — while keeping your
prompt, model choice, and batch. On the web it is undoable from the toast it
raises.

## Video Generation

mold supports text-to-video generation with the LTX Video, LTX-2 (next
section), and Wan 2.1/2.2 model families. LTX Video output defaults to APNG,
with GIF, WebP, and MP4 also supported; LTX-2 and Wan default to MP4.

```bash
# Generate a 25-frame video clip with the fast distilled path
mold run ltx-video-0.9.6-distilled:bf16 "A cat walking across a sunlit windowsill"

# Custom frame count (must be 8n+1: 9, 17, 25, 33, 49, 97, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "Ocean waves at sunset" --frames 33

# Custom FPS (current LTX defaults use 30 FPS)
mold run ltx-video-0.9.6:bf16 "A timelapse of clouds" --frames 49 --fps 30

# Pipe to a video player
mold run ltx-video-0.9.6-distilled:bf16 "A robot dancing" | mpv -

# Direct MP4 output
mold run ltx-video-0.9.6-distilled:bf16 "A waterfall" --format mp4 -o waterfall.mp4
```

`ltx-video-0.9.6-distilled:bf16` is the recommended default today. The
`0.9.8` family is also supported end to end: mold pulls the required spatial
upscaler asset, runs the full multiscale refinement path, and keeps the current
compatible VAE on the published `LTX-Video-0.9.5` source until the newer VAE
layout is ported.

::: tip Frame count constraint
LTX Video requires frame counts of the form **8n+1** (9, 17, 25, 33, 49, 97,
etc.) due to the VAE's 8x temporal compression. mold will reject invalid counts
with a helpful error message.
:::

::: warning VRAM usage
LTX Video uses sequential load-use-drop to manage VRAM: T5 encoder loads first,
then drops before the transformer loads, then the transformer drops before VAE
decode. Peak VRAM depends heavily on the selected LTX checkpoint.
:::

LTX video dimensions must be multiples of 32 (not 16 like images). Current LTX
defaults use 1216×704 at 30 FPS.

### Wan Video

Wan 2.1/2.2 is a separate `wan` family: MP4 by default, frames on a 4n+1 grid
(49, 53, 81, 121, ...), dimensions in multiples of 16 — except
`wan22-ti2v-5b`, whose 2.2 VAE requires multiples of 32.

```bash
# 480p16 text-to-video (defaults: 81 frames @ 16 fps)
mold run wan21-t2v-1.3b "a red fox trotting through fresh snow, golden hour"

# 720p24 — Wan 2.2 5B, text- or image-to-video
mold run wan22-ti2v-5b "waves breaking on a black sand beach" \
  --width 1280 --height 704 --frames 121 --fps 24

# Wan 2.2 A14B, 4-step Lightning tier (defaults: 53 frames @ 16 fps)
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"

# A14B image-to-video from a still
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
```

Wan checkpoints were tuned against a specific negative prompt; mold applies it
automatically when `--negative` is not given. A14B is a two-expert mixture
with one 14B expert resident at a time, and its 53/33-frame defaults are the
measured 24 GB envelope — larger cards pass `--frames 81` explicitly. See
[Wan Video](/models/wan) for variants, defaults, and limits.

## Joint Audio-Video Generation

LTX-2 / LTX-2.3 is exposed as a separate `ltx2` family. Unlike `ltx-video`,
its default container is MP4 and it can keep a synchronized audio track when
the request stays in MP4.

```bash
# Text-to-audio+video
mold run ltx-2-19b-distilled:fp8 \
  "a toy train rolling through a snowy diorama, gentle mechanical hum" \
  --frames 97 \
  --format mp4

# Audio-to-video
mold run ltx-2-19b-distilled:fp8 \
  "abstract paper sculpture reacting to a cello performance" \
  --audio-file ./cello.wav

# Keyframe interpolation
mold run ltx-2-19b-distilled:fp8 \
  "a drone shot over a canyon river" \
  --pipeline keyframe \
  --frames 97 \
  --keyframe 0:./start.png \
  --keyframe 96:./end.png

# Image-to-video with no prompt at all
mold run ltx-2-19b-distilled:fp8 --image ./still.png --frames 97 --format mp4
```

The prompt is optional for `ltx2` and `ltx-video` — and only for them — when the
request already carries visual conditioning (`--image`, `--keyframe`, `--video`,
or `--extend`). It saves no VRAM and usually yields near-static motion; see
[the LTX-2 page](/models/ltx2#the-prompt-is-optional-for-image-to-video).

LTX-2 also adds:

- `--audio` / `--no-audio`
- `--audio-file`
- `--video`
- repeatable `--keyframe <frame:path>`
- `--pipeline one-stage|two-stage|two-stage-hq|distilled|ic-lora|keyframe|a2-vid|retake|lip-dub`
- `--pipeline one-stage|two-stage|two-stage-hq|distilled|ic-lora|keyframe|a2-vid|retake|t2a`
- `--retake <start:end>`
- repeatable `--lora`
- `--camera-control <preset-or-path>`
- `--spatial-upscale <x1.5|x2>`
- `--temporal-upscale x2`

Catalog checkpoints may contain the LTX-2 transformer without `vae.*` weights.
`mold pull cv:<id>` detects that layout and fetches the matching LTX-2 or
LTX-2.3 video VAE automatically. Diffusion-only LTX-2.3 exports also fetch the
separate Gemma hidden-state projection. The resolved assets are pinned in each
chain stage, so multi-prompt chains do not fall back to the transformer file.
ConvRot W4A4 exports use automatic full block streaming because their packed
on-disk byte size understates the BF16 weights reconstructed by the runtime.
If the Gemma prompt encoder exhausts VRAM, Mold retries only Gemma on CPU while
keeping the transformer and video VAE on CUDA.
Multi-prompt chains support both one-stage and distilled LTX-2 checkpoints;
multi-pass and specialized conditioning pipelines remain explicit non-chain
modes. Mold checks this before creating a durable job and keeps server, stage,
cancel, resume, and retake errors visible on the job card.

Some community checkpoints contain only the video transformer (plus Mold's
separate video VAE) and do not include the audio VAE or vocoder. Mold detects
this from the installed safetensors and disables **Generate audio** in web,
desktop, and iPhone while leaving text/image-to-video available. CLI users can
pass `--no-audio`; an explicit unsupported audio request is rejected before
prompt encoding or denoising.

The native CUDA matrix is validated across 19B/22B text+audio-video,
image-to-video, audio-to-video, keyframe, retake, public IC-LoRA, spatial
upscale, and temporal upscale workflows.

::: warning Backend policy
LTX-2 now runs natively in Rust inside `mold-inference`. CUDA is the supported
backend for real local generation, CPU is correctness-only, and Metal is
unsupported for this family.
:::

## Negative Prompts

Guide what the model should avoid. Works with CFG-based models (SD1.5, SDXL,
SD3, Wuerstchen, Qwen-Image, Qwen-Image-Edit); ignored by FLUX, Z-Image, and
Flux.2 Klein.

```bash
mold run sd15:fp16 "a portrait" -n "blurry, watermark, ugly, bad anatomy"
mold run sdxl:fp16 "a landscape" --negative-prompt "low quality, jpeg artifacts"

# Disable every default negative — config defaults and Wan's tuned model
# default alike — by sending an explicit empty negative
mold run wan22-t2v-a14b:q5 "a cat" --no-negative
```

Precedence: CLI `--negative-prompt` > per-model config > global config > the
model family's tuned default (Wan) > empty. Wan's tuned default is advertised
per model via `/api/models` (`default_negative_prompt`) and prefilled into
the Negative control on web, desktop, iPhone, and the TUI; clearing that
field is the same explicit opt-out as `--no-negative`.

## Scheduler Selection

Choose the noise scheduler for SD1.5/SDXL models:

```bash
mold run sd15:fp16 "a cat" --scheduler uni-pc         # Fast convergence
mold run sd15:fp16 "a cat" --scheduler euler-ancestral # Stochastic
```

## LoRA Adapters

Apply fine-tuned style adapters across the supported families — **FLUX, Flux.2,
LTX-2, SD1.5, SD3, SDXL, Qwen-Image (+ Qwen-Image-Edit), Wan, Z-Image**:

```bash
# Basic LoRA (FLUX example)
mold run flux-dev:bf16 "a portrait" --lora style.safetensors

# Adjust strength (0.0 = no effect, 1.0 = full, up to 2.0)
mold run flux-dev:bf16 "anime style" --lora style.safetensors --lora-scale 0.7

# Works with quantized models too
mold run flux-dev:q4 "a portrait" --lora style.safetensors --lora-scale 0.8

# Same flag syntax across families
mold run sdxl:fp16    "a sunset" --lora sdxl-style.safetensors
mold run z-image:bf16 "anime"    --lora cv:2904324
```

::: tip LoRA requirements
Requires `.safetensors` format. Z-Image / FLUX accept diffusers (PEFT canonical),
Kohya/sd-scripts, OneTrainer, and PEFT default-adapter naming. BF16 FLUX on
24 GB cards can adaptive-offload, keeping fitting blocks on GPU and streaming
only overflow blocks; LTX-2 can use the conservative full-streaming offload path.
Wuerstchen and legacy LTX-Video are not yet wired — attaching a LoRA there
returns a 400 with the supported-family list.
:::

## Inline Preview

Display generated images in the terminal:

```bash
mold run "a cat" --preview
```

Requires the `preview` feature at build time. Auto-detects Kitty graphics,
iTerm2, Sixel, or Unicode half-block fallback.

Set `MOLD_PREVIEW=1` to enable permanently.

## PNG Metadata

Generated PNGs embed prompt, model, seed, size, steps, and a `mold:parameters`
JSON chunk by default. Disable with:

```bash
mold run "a cat" --no-metadata
# or globally
MOLD_EMBED_METADATA=0 mold run "a cat"
```

## Piping

mold is pipe-friendly in both directions. When stdout is not a terminal, raw
image bytes go to stdout and status goes to stderr.

```bash
# Pipe output to an image viewer
mold run "neon cityscape" | viu -

# Pipe prompt from stdin
echo "a cat riding a motorcycle" | mold run flux2-klein

# Full pipeline
echo "cyberpunk samurai" | mold run flux-dev:q4 | viu -

# Force stdout in interactive mode
mold run "a cat" --output -
```

## Inference Modes

1. **Remote** (default) — connects to `mold serve` via HTTP
2. **Local fallback** — if server unreachable, auto-falls back to local GPU
3. **Local forced** (`--local`) — skip server, run on local GPU directly

Models auto-pull if not downloaded.

## Browser UI

`mold serve` ships with a Vue 3 SPA baked into the binary. Visit
`http://<host>:7680/` to open the Create composer. The canonical routes are
`/create`, `/library`, `/models`, `/machines`, and `/settings`; retired paths
such as `/generate` and `/catalog` render Page Not Found:

- The model selector shows human-readable catalog names while preserving
  `cv:` / `hf:` identifiers internally for requests.
- Enter submits, Shift+Enter inserts a newline, empty Enter is a no-op.
- Per-GPU running-job cards stream SSE progress (stage, denoise step N/M, VAE
  decode) and tag the finished image with the GPU ordinal that produced it.
- Fire multiple prompts in quick succession; the server queues them and the UI
  surfaces HTTP 503 / `Retry-After` cleanly when `--queue-size` is reached.
- img2img works via upload or the From Gallery picker; video-family models are
  grouped with a 🎬 badge and frames are clamped to 8n+1 automatically.
- Library's print viewer keeps media bound to its owning host, restores the
  saved model family on **Reuse settings** (a print a sequence produced reloads
  its clips onto the Create clip rail instead, with **Edit sequence** offered
  when its durable job still exists on the machine that made it), and shows the
  recorded steps,
  guidance, scheduler, LoRAs, prompts, file details, and copyable prompt/seed.
  **Upscale...** returns the print to Create with the installed default
  upscaler selected.
- Prompt expansion modal offers live preview + variation picker (requires
  `qwen3-expand` installed on the server).
- Prompt, model, size, steps, guidance, and batch persist in `localStorage`.
- Modal and sheet workflows contain keyboard focus, lock background scrolling,
  close on Escape, and restore focus to the control that opened them. The
  Templates popover also dismisses on Escape or an outside click.
