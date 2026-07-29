---
title: Video Generation
---

# Video Generation

mold supports generating video clips using LTX Video and LTX-2 models. LTX-2
distilled and one-stage checkpoints support chaining multiple clips together
for longer videos with scene-by-scene direction. Two-stage dev checkpoints are
single-clip pipelines and are rejected before a sequence job is created.

## Single-clip generation

```bash
mold run ltx-2-19b-distilled:fp8 "a cat walks through autumn leaves" --frames 97
```

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
LTX-Video has no motion tail, so its seams read **Join clips**. New clips
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
recorded clips. Desktop and web keep TOML import/export for `mold.chain.v1` scripts
under the composer's file tools.

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
`supports_sequence` is model-specific: a two-stage LTX-2 dev checkpoint reports
`false` with a `sequence_unsupported_reason`. `GET /api/models` carries the
matching per-model `default_frames`, `default_fps`, `max_frames`, and
`frame_step` fields.
