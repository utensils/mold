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
audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale (`x1.5` /
`x2` where published), and temporal upscale (`x2`).
:::

## Supported Models

| Model                       | Path      | Notes                                        |
| --------------------------- | --------- | -------------------------------------------- |
| `ltx-2-19b-dev:fp8`         | Two-stage | Highest-quality published 19B FP8 checkpoint |
| `ltx-2-19b-distilled:fp8`   | Distilled | Fastest 19B path, recommended default        |
| `ltx-2.3-22b-dev:fp8`       | Two-stage | Highest-quality 22B FP8 checkpoint           |
| `ltx-2.3-22b-distilled:fp8` | Distilled | Fastest 22B path                             |

## Implemented Request Surface

- Text-to-audio+video with synchronized MP4 output
- First-frame image-to-video via `--image`
- Audio-to-video via `--audio-file`
- Keyframe interpolation via repeatable `--keyframe`
- Retake / partial regeneration via `--video` + `--retake`
- IC-LoRA and stacked LoRAs via repeatable `--lora`
- Camera-control preset names for the published LTX-2 19B camera LoRAs
- Spatial upscale `x2` across the family and `x1.5` for `ltx-2.3-*`
- Temporal upscale `x2`

## Current Constraints

- Default output is `mp4` for this family. `gif`, `apng`, and `webp` are also
  supported, but they are treated as silent exports.
- `x2` spatial upscaling is wired across the family. `x1.5` is wired for
  `ltx-2.3-*` by resolving the published upstream asset on demand.
- `x2` temporal upscaling is wired through the native LTX-2 runtime.
- Camera-control preset aliases are currently published for LTX-2 19B only. For
  LTX-2.3, pass an explicit `.safetensors` path.
- The Gemma text encoder source is gated on Hugging Face, so you must have
  access approved before `mold pull` will complete.
- When you send source media through `mold serve`, the built-in request body
  limit is `64 MiB`, which covers common inline retake and audio-to-video
  requests.
- On 24 GB Ada GPUs such as the RTX 4090, mold keeps the native runtime on the
  compatible `fp8-cast` path rather than Hopper-only `fp8-scaled-mm`.

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

# Retake a source clip over a time range
mold run ltx-2-19b-distilled:fp8 \
  "replace the actor with a chrome mannequin" \
  --pipeline retake \
  --video ./source.mp4 \
  --retake 1.5:3.5 \
  --format mp4

# Spatial upscale on a published LTX-2.3 asset
mold run ltx-2.3-22b-distilled:fp8 \
  "red sports car in rain, cinematic reflections" \
  --spatial-upscale x1.5 \
  --format mp4
```

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
LTX-2 19B and 22B distilled today. Other model families reject
`--frames > 97` with an actionable error rather than silently over-producing.

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

| Flag              | Default          | Description                                                                          |
| ----------------- | ---------------- | ------------------------------------------------------------------------------------ |
| `--frames N`      | model default    | Total stitched length. Above the per-clip cap (97 for LTX-2 distilled), auto-chains. |
| `--clip-frames N` | model cap (`97`) | Per-clip length. Must be `8k+1`; values above the cap are clamped with a warning.    |
| `--motion-tail N` | `4`              | Pixel-frame overlap between clips. `0` disables carryover.                           |

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
- **Single prompt per chain from the CLI.** The server already accepts
  per-stage prompts (see [`POST /api/generate/chain`](/api/#api-generate-chain)),
  but `mold run` replicates one prompt across every stage for now.

The rest of the LTX-2 surface — `--image`, `--audio-file`, `--lora`,
`--camera-control`, `--spatial-upscale`, `--temporal-upscale`, and so on —
applies to chain renders the same way it applies to single-clip renders. An
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
