---
title: Video Generation
---

# Video Generation

mold supports generating video clips using LTX Video and LTX-2 models. Every
LTX-2 checkpoint can chain multiple clips together for longer videos with
scene-by-scene direction. A dev checkpoint renders its clips through the
two-stage pipeline, so expect roughly twice the wall time per clip as a
distilled one — stage 1 runs classifier-free guidance as two sequential
forward passes.

## Single-clip generation

```bash
mold run ltx-2-19b-distilled:fp8 "a cat walks through autumn leaves" --frames 97
```

## Resolution and spatial tiling

LTX-2 normalizes RoPE pixel positions by a 2048px span, so a longer edge is
outside the trained range even when the frame's area is small. A checkpoint
that ships the spatial upsampler reaches past it by **composing**: stage 1
renders at half the requested size, the learned upsampler doubles it, and
stage 2 refines the result over tiles each brought back inside the span. That
puts the generation ceiling at 4096px on the long edge — exactly where a
single halving stops landing stage 1 inside the span — and gives an output
ladder of 1280x704, 1920x1088, 2560x1408 and 3840x2112. A checkpoint that
renders in one pass stays capped at 2048px and 1920x1088.

The ladder stops at 3840x2112 because of the **encoder**, not the model: the
bundled OpenH264 encoder refuses anything past 3840x2160, so 3840x2176 (what
rounding 2160 up onto the /64 grid would give) generates correctly and then
fails at save time.

There is nothing to assemble by hand: choose the output size and mold runs the
composition. `/api/models` advertises the per-model `max_pixels`,
`max_axis_pixels` and `recommended_dimensions`, so a picker never offers a rung
the selected checkpoint would reject.

**4K needs `--spatial-tile 768` on a 24 GB card.** Measured on an RTX 4090 with
`ltx-2-19b-distilled:fp8` at 25 frames: 1080p peaks at 18.4 GB, 1440p at
18.1 GB, and 4K completes at 18.2 GB with the smaller tile. With the default
1280px tile, 4K reaches the VAE decode and fails there with an out-of-memory
error naming the phase — it does not silently render something smaller. Those
are single-configuration measurements at 25 frames, not a support matrix; see
[LTX-2 → Resolution](/models/ltx2#resolution) for the full numbers and
caveats.

`--spatial-tile` (or `MOLD_LTX2_SPATIAL_TILE`, which `mold serve` reads) takes
`auto` — the default, which tiles only where it buys something — `off`, or an
explicit `<px>` / `<px>:<overlap>` in multiples of 32. `auto` engages exactly
at the span and no earlier, so every resolution up to 1080p renders as it
always has; `off` past the span is refused rather than quietly degraded. See
[LTX-2 → Spatial tiling](/models/ltx2#spatial-tiling-spatial-tile).

## Multi-prompt scripts (v2)

Direct any-length video scene-by-scene with a TOML script. Each prompt becomes a stage; each boundary has a `transition` (`smooth`, `cut`, or `fade`).

### Canonical form

```bash
mold run --script shot.toml
mold run --script shot.toml --dry-run   # print stage summary, don't submit
mold chain validate shot.toml            # parse without submitting
```

### Sugar form (uniform smooth chains)

```bash
mold run ltx-2-19b-distilled:fp8 \
  --prompt "a cat walks into the autumn forest" \
  --prompt "the forest opens to a clearing" \
  --prompt "a spaceship lands" \
  --frames-per-clip 97
```

Per-stage transitions or per-stage frames require `--script`.

### Transitions

| Mode                 | Behavior                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `smooth` _(default)_ | Motion-tail carryover — prompt change produces a visual morph between scenes.                     |
| `cut`                | Fresh latent, no carryover. If the stage has `source_image`, it's used as an image-to-video seed. |
| `fade`               | Cut + post-stitch alpha blend of `fade_frames` (default 8) on each side of the boundary.          |

### Example `shot.toml`

```toml
schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
seed = 42
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 25
output_format = "mp4"

[[stage]]
prompt = "a cat walks into the autumn forest"
frames = 97

[[stage]]
prompt = "the forest opens to a clearing"
frames = 49

[[stage]]
prompt = "a spaceship lands"
frames = 97
transition = "cut"

[[stage]]
prompt = "the cat looks up in wonder"
frames = 97
transition = "fade"
fade_frames = 12
```

### Sequences in the apps

On web, desktop, and iPhone, multi-clip video is a setting rather than a
separate page. In Create, set **Output** (beside Model) to **Sequence** and the
composer becomes a clip rail: clip pills carrying a prompt and frame count,
joined by seam pills. Seams are named in words — **Smooth**, **Cut**, and
**Fade 8f** — and a click opens the seam editor with its fade-length stepper.
LTX-Video has no motion tail, so its seams read **Join**. New clips
default to the selected model's advertised frame count. Frame choices and
timeline metadata include their duration at the live model FPS, such as
`97f · 4.0s`.

Sequences run as durable jobs in the same activity strip as ordinary prints.
Desktop and web show each scene as a 16:9 filmstrip tile with its poster,
render progress, and cache state. As soon as a stage finishes, its play control
opens the raw scene in the main Create canvas while later stages continue
rendering; **Return to live render** switches the canvas back to progress.
Desktop and web can edit a finished sequence in place: its clips reload onto the
rail, each pill shows cached (✓) versus re-render (↻), and **Update sequence**
re-renders only from the earliest changed clip — transition and fade-length
edits re-stitch with no re-render at all. Every settled sequence job is listed
in **Library ▸ History ▸ Sequences**, and a sequence print in the Library uses
**Edit sequence** as its primary desktop/web action (re-enter the original job
with its cached clips). **Duplicate as new** starts a fresh sequence from the
recorded clips. Desktop and web keep TOML import/export for `mold.chain.v1`
scripts under the composer's file tools. Web, desktop, and iPhone also offer
**Validate plan**: it
asks the currently selected authenticated host to normalize the live draft,
then shows each clip's input/output frames, transition, conditioning inputs,
warnings, and VRAM estimate when available. Validation creates no job and
starts no download or inference work.

### TUI chain composer

Press `c` from Create's navigation mode in `mold tui` to author a
`mold.chain.v1` script with per-stage prompts, frame counts, source images, and
`smooth` / `cut` / `fade` transitions. See the
[TUI guide](/guide/tui#chain-composer).

### Capabilities endpoint

```
GET /api/capabilities/chain-limits?model=<name>
```

Returns per-model caps used by every sequence UI:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "frames_per_clip_cap": 97,
  "frames_per_clip_recommended": 97,
  "max_stages": 16,
  "max_total_frames": 1552,
  "fade_frames_max": 32,
  "transition_modes": ["smooth", "cut", "fade"],
  "quantization_family": "ltx2",
  "supports_audio": true,
  "supports_sequence": true
}
```

`frames_per_clip_recommended` follows the model's own default frame count — 97
for LTX-2, 25 for LTX-Video — so clients do not have to hardcode one.
`supports_sequence` is model-specific and is also advertised per model on
`GET /api/models`, so a picker never has to infer it from the checkpoint name.
A family with no chain path reports `false` with a
`sequence_unsupported_reason`. `GET /api/models` carries the matching per-model
`default_frames`, `default_fps`, `max_frames`, and `frame_step` fields.
