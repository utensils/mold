---
title: Video Generation
---

# Video Generation

Mold supports generating video clips using LTX Video (including LTX-2.5), Wan
2.1/2.2, and MiniMax H3. Every LTX-2 and Wan
checkpoint can chain multiple clips together for longer videos with
scene-by-scene direction; H3 is single-clip only. A dev checkpoint
renders its clips through the two-stage pipeline, so expect roughly twice the
wall time per clip as a distilled one; stage 1 runs classifier-free guidance
as two sequential forward passes.

An opening image conditions the first clip only. On LTX-2 and LTX-2.3 the
same image also rides every smooth continuation as a soft identity anchor;
on LTX-2.5, which was trained with keyframe conditioning and would read that
anchor as a keyframe to cut back to, continuations carry the motion tail alone.

### Wan sequences

What crosses a wan clip boundary depends on the checkpoint, not the family:
wan has no latent motion tail, so its smooth handoff is last-frame image
conditioning. `wan22-ti2v-5b` and `wan22-i2v-a14b` continue across the seam;
a text-to-video checkpoint concatenates independent clips (Join / Cut /
Crossfade). Clip lengths sit on wan's `4k+1` grid, and `--frames` past the
per-clip envelope auto-chains instead of failing.

```bash
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" \
  --frames 100 --clip-frames 49
```

## Single-clip generation

```bash
mold pull ltx-2.5-22b-distilled:int8-conv
mold run ltx-2.5-22b-distilled:int8-conv \
  "a cat walks through autumn leaves" --frames 97
```

The compact LTX-2.5 INT8 ConvRot split pack is the recommended path on Apple
Metal, where it is qualified. CUDA has also completed its separate NVIDIA
qualification campaign. The pack is gated on Hugging Face and includes the
matching Gemma 4 encoder, audio/video VAEs, duration head, and latent
upscalers. On CUDA its blocks stay resident in packed W8A8 form and execute a
native INT8 GEMM.

Seven pinned distilled GGUF tiers run natively beside it and are the smaller
option on a 24 GB card — `ltx-2.5-22b-distilled` in tags `q3-k-s`, `q3`,
`q4-k-s`, `q4`, `q5`, `q6`, and `q8`, of which Q4_K_M (`q4`) sits fully
resident:

```bash
mold run ltx-2.5-22b-distilled:q4 "a cat walks through autumn leaves" --frames 97
```

On Apple Metal, these GGUF tiers retain only the packed transformer blocks
that fit beyond the live unified-memory safety floor. Remaining blocks stream
one tensor at a time from the checkpoint with bounded synchronization, so a
smaller-memory Mac trades disk traffic and speed for a bounded working set.
If even the minimum streaming working set would cross the live macOS safety
floor, Mold refuses the request before transformer allocation with advice to
free memory or reduce the request shape.
Q3_K_M, Q4_K_M, and Q6_K are hardware-qualified on a 48 GiB Apple M4 Max at
512x512, 9 frames, and 8 steps with visual prompt-fidelity inspection. Q3_K_M
also completed a 97-frame MP4. The K_S, Q5_K_M, and Q8_0 tiers remain runnable
but are not claimed by that Metal qualification campaign.

See [LTX Video](/models/ltx2) for per-tier download sizes, qualified workflows,
and the BF16 packs, whose execution remains operator-deferred on Metal.

## Wan Video

Wan renders single clips and multi-clip sequences, and defaults to MP4:

```bash
# 480p16 text-to-video (defaults: 81 frames @ 16 fps)
mold run wan21-t2v-1.3b "a red fox trotting through fresh snow, golden hour"

# Wan 2.2 A14B, 4-step Lightning tier (defaults: 81 frames @ 16 fps)
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"

# A14B image-to-video from a still
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
```

Wan's frame grid is 4n+1 (49, 53, 81, 121, ...) from its VAE's 4x temporal
compression, and dimensions must be multiples of 16; except `wan22-ti2v-5b`,
whose 2.2 VAE requires multiples of 32. A14B is a two-expert mixture: mold
keeps one 14B expert resident at a time, so VRAM is the larger expert, not
the sum. The `:q5`/`:q4` tiers default to the checkpoint's trained 81
frames (automatic partial block offload fits them on a 24 GB card) while
`:q8` defaults to 73 frames and `:fp8` to 45, their measured 24 GB
envelopes. Wan checkpoints were tuned against a
specific negative prompt that mold applies automatically when a request
carries no negative at all; every surface shows it, editing replaces it, and
clearing it (`--no-negative` on the CLI) sends a real empty negative. See
[Wan Video](/models/wan) for variants, defaults, and limits.

## MiniMax H3

MiniMax H3 always returns MP4 with synchronized generated audio. The compact
FL2VA route requires a first frame; the compact Ref2VA route takes 1–12 ordered
image, video, or audio references. Both use 24 fps and a `17n+5` frame grid
from 107 through 345 frames.

```bash
mold run minimax-h3-fl2va:comfy-pruned-int8 \
  "the camera circles the lantern as wind moves the trees" \
  --first-frame lantern.png --duration 5
```

Generation is available on H3-enabled SM89 CUDA builds. The shipped Apple
Metal route is correctness-only and not yet hardware-qualified; CPU is
unsupported. H3 does not participate in the sequence workflow below. See
[MiniMax H3](/models/minimax-h3) for Ref2VA uploads, Turbo tags, download-only
layouts, and exact request limits.

## Resolution and spatial tiling

LTX-2 normalizes RoPE pixel positions by a 2048px span, so a longer edge is
outside the trained range even when the frame's area is small. A checkpoint
that ships the spatial upsampler reaches past it by **composing**: stage 1
renders at half the requested size, the learned upsampler doubles it, and
stage 2 refines the result over tiles each brought back inside the span. That
puts the generation ceiling at 4096px on the long edge (exactly where a
single halving stops landing stage 1 inside the span) and gives an output
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
error naming the phase; it does not silently render something smaller. Those
are single-configuration measurements at 25 frames, not a support matrix; see
[LTX-2 → Resolution](/models/ltx2#resolution) for the full numbers and
caveats.

`--spatial-tile` (or `MOLD_LTX2_SPATIAL_TILE`, which `mold serve` reads) takes
`auto` (the default, which tiles only where it buys something) `off`, or an
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
| `smooth` _(default)_ | Motion-tail carryover; prompt change produces a visual morph between scenes.                      |
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
joined by seam pills. Seams are named in words (**Smooth**, **Cut**, and
**Fade 8f**) and a click opens the seam editor with its fade-length stepper.
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
re-renders only from the earliest changed clip; transition and fade-length
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
GET /api/capabilities/chain-limits?model=<name>[&fps=<n>]
```

Returns per-model caps used by every sequence UI:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "frames_per_clip_cap": 97,
  "fps": 24,
  "frames_per_clip_runtime_seconds": 20,
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

`frames_per_clip_cap` is the model's own clip size (what one generation
renders when a long one-shot request is chained automatically (97 for LTX-2;
for Wan the checkpoint's own manifest default over a 53-frame A14B /
121-frame floor, e.g. 121 for TI2V-5B)) so a sequence clip can never be longer
than the clips the Duration slider would have produced.
`fps` echoes the frame rate the cap was computed at; it defaults to the
model's own default fps when the query omits it. It matters because LTX-2's
family ceiling is a runtime duration (`frames_per_clip_runtime_seconds`, 20 s),
so pass the fps you will actually render at and the cap moves with it.
`frames_per_clip_recommended` follows the model's own default frame count (97
for LTX-2, 25 for LTX-Video) so clients do not have to hardcode one.
`supports_sequence` is model-specific and is also advertised per model on
`GET /api/models`, so a picker never has to infer it from the checkpoint name.
A family with no chain path reports `false` with a
`sequence_unsupported_reason`. `GET /api/models` carries the matching per-model
`default_frames`, `default_fps`, `max_frames`, and `frame_step` fields.

## Framewise upscale

From Library, choose **Framewise upscale**, or run:

```bash
mold video-upscale create clip.mp4 --wait
```

This is spatial super-resolution with native Real-ESRGAN applied independently
to each frame. It is durable, checkpointed, pausable, resumable, cancellable,
and publishes a new gallery MP4 without replacing the source. It preserves and
verifies constant FPS, frame count, duration, and a codec-compatible primary
audio track. Temporal flicker may remain.

The published MP4 is H.264 at level 5.2 or below, so it decodes on phones,
in browsers, and in the Library thumbnailer. Real-ESRGAN always runs at its
native factor; when the enlarged frame would exceed that level (more than
36 864 macroblocks, or an edge over 4096 px) the encoder resamples it to the
largest frame of the same aspect ratio that fits. A 960×960 clip upscaled ×4
therefore publishes at 3072×3072 rather than 3840×3840, and the print's
metadata records the published size beside the source size.

The MVP rejects rather than discards VFR timing, HDR/high-bit-depth video,
subtitles, chapters, multiple audio tracks, and incompatible audio codecs.
