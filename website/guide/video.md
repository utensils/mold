---
title: Video Generation
---

# Video Generation

Mold supports generating video clips using LTX Video (including LTX-2.5), Wan
2.1/2.2, and MiniMax H3. The CLI and API can stitch authored clips into a
durable sequence. LTX-2, image-conditioned Wan and H3 FL2VA can carry context
across smooth seams; text-only Wan, legacy LTX-Video and H3 Ref2VA compose
independent clips. Automatic long one-shots are limited to models with a
continuation path. A dev checkpoint
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
per-clip envelope auto-chains instead of failing — the last clip is sized so
the stitched video delivers exactly the frames you asked for.

A **text-to-video** tier is the exception: it hands nothing across a seam, so
an automatic split would render the same clip again rather than extend it.
`mold run`, the apps, and `POST /api/chain-jobs` all refuse a one-shot past
that tier's clip size with the same sentence. A scripted sequence still
composes those clips deliberately.

Legacy LTX-Video also has no image-to-video handoff. Mold therefore keeps its
one-shot requests as one denoise up to the 257-frame engine ceiling instead of
automatically splitting at 97 and rerunning the same prompt and seed. Use
LTX-2.3/LTX-2.5 for image-conditioned continuation; an explicitly scripted
sequence may still join independent legacy clips on purpose.

```bash
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" \
  --frames 97 --clip-frames 49
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
FL2VA route renders text-only or from a first frame, a last frame, or both;
the compact Ref2VA route takes 1–12 ordered
image, video, or audio references. Both use 24 fps and a `17n+5` frame grid
from 107 through 345 frames.

```bash
mold run minimax-h3-fl2va:comfy-pruned-int8 \
  "the camera circles the lantern as wind moves the trees" \
  --first-frame lantern.png --duration 5
```

Generation is available on H3-enabled SM89 CUDA builds and the shipped Apple
Metal build. Metal streams the compact stack and applies exact per-request
unified-memory admission; CPU is unsupported. H3 participates in the
sequence workflow below through an H3-enabled server. FL2VA uses the previous
clip’s final RGB frame for smooth continuation; Ref2VA renders independent
clips with each stage’s ordered references. Local H3 chains remain refused.
See [MiniMax H3](/models/minimax-h3) for Ref2VA
uploads, Turbo tags, download-only layouts, and exact request limits.

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

**VAE decoding uses spatial tiles automatically.** Convolutional LTX-2 VAE
Auto uses upstream aspect-coupled 768px long-side tiles with 64px overlap,
independent of the temporal workspace estimate. Temporal context, retained
output strips and final output still use memory, so long clips can still OOM.
The historical RTX 4090 measurements at 25 frames used an explicit square
768px tile with 256px overlap; they do not qualify the current Auto geometry.
See [LTX-2 → Resolution](/models/ltx2#resolution) for the measured configurations.

`--spatial-tile` (or `MOLD_LTX2_SPATIAL_TILE`, which `mold serve` reads) takes
`auto` (default), `off`, or explicit `<px>` / `<px>:<overlap>` in multiples of 32. Bare explicit sizes retain a 256px overlap. Stage-2 Auto still engages
only past the trained 2048px span; `off` past that span is refused. See
[LTX-2 → Spatial tiling](/models/ltx2#spatial-tiling-spatial-tile).

## Multi-prompt scripts

Direct a video scene by scene with a TOML script. Each prompt becomes a stage; each boundary has a `transition` (`smooth`, `cut`, or `fade`).

::: tip Scripts are the only way to direct scenes
Scene-by-scene authoring lives in the CLI and the HTTP API. The apps (web,
desktop, iPhone) make one clip at a time from one prompt: set the length you
want and the host chains and stitches the clips for you.
:::

### Canonical form

```bash
mold run --script shot.toml
mold run --script shot.toml --dry-run   # print stage summary, don't submit
mold chain validate shot.toml            # parse without submitting
```

Validation and `--dry-run` plan locally; they do not exercise remote model
admission. A remote run prints its durable **Sequence job** ID. Retain that ID
and inspect `mold jobs show ID --json --host HOST` after a disconnected stream
before submitting again. `--output` writes on the CLI machine; the host also
keeps the stitched print in its gallery. Download its exact output filename
with `mold library download NAME --host HOST`.

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

| Mode                 | Behavior                                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `smooth` _(default)_ | Model-specific carryover: LTX-2 latent tail, Wan I2V/TI2V or H3 FL2VA final frame; text-only models and H3 Ref2VA join independent clips. |
| `cut`                | Fresh latent, no carryover. If the stage has `source_image`, it's used as an image-to-video seed.                                         |
| `fade`               | Cut + post-stitch alpha blend of `fade_frames` (default 8) on each side of the boundary.                                                  |

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

### H3 stage inputs

FL2VA stages accept `first_frame_path` and `last_frame_path`, relative to the
script file. These become typed boundary keyframes; do not mix them with
`source_image` or `keyframes`. A smooth continuation supplies the incoming
first frame itself, so an authored first frame requires a cut or fade. Ref2VA
stages accept ordered typed `generation_references`, using the same reference
wire schema as ordinary generation, and reject source images or keyframes.
For CLI files use `[[stage.generation_references]]` with `kind` and `path`
as in the example below.
The API stage fields are `keyframes` and `generation_references`. Generated
audio is retained through stitching.

### H3 Ref2VA with a local image

Save this as `ref-chain.toml` beside an existing `hero.png`. Inspect the exact
model on the target host first; this example uses its reviewed compact Turbo
recipe. Paths resolve relative to the script and are probed before submission.
The CLI inlines at most 32 MiB of authored media per part (source images, keyframes and references combined).

```toml
schema = "mold.chain.v1"

[chain]
model = "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step"
width = 1344
height = 768
fps = 24
seed = 42
steps = 5
guidance = 0.0
strength = 1.0
motion_tail_frames = 0
output_format = "mp4"

[[stage]]
prompt = "integrated_multimodal_description: [Shot 1] <Subject 1> in a garden, slow camera push. overall_soundscape: Wind in leaves. non_diegetic_music: None."
frames = 124
[[stage.generation_references]]
kind = "image"
path = "hero.png"

[[stage]]
prompt = "integrated_multimodal_description: [Shot 1] <Subject 1> in the garden, a wide view. overall_soundscape: Wind in leaves. non_diegetic_music: None."
frames = 124
transition = "cut"
[[stage.generation_references]]
kind = "image"
path = "hero.png"
```

Use `kind = "video"` or `"audio"` for those media. Follow the H3 Ref2VA
prompting guide for ordered subject references and exact prompt structure.
Each stage is an independent clip; repeated references guide identity without
claiming motion continuation.

```bash
mold list --json --host HOST
mold chain validate ref-chain.toml
mold run --script ref-chain.toml --dry-run --host HOST
mold run --script ref-chain.toml --host HOST --output ref-chain.mp4
```

### Longer clips in the apps

Web, desktop, and iPhone have no scene composer. Create makes one clip from one
prompt, and the only length control is the clip's own duration. Ask for more
frames than the checkpoint renders in a single pass and the host splits the work
into clips on its own, carries the motion tail across each seam, and stitches
the result — you get one video and one Library entry, and the progress readout
names the clip it is on. Text-only Wan refuses automatic splitting, and legacy
LTX-Video remains one render up to its own ceiling. H3 scripted sequences are
CLI/API-only; GUI authoring is unchanged.

Clips made by an older build's sequence composer are still in your library with
their per-scene provenance intact. **Use these settings again** on one of them
restores a plain one-shot clip built from the first scene's prompt.

### Continue beyond 16 stages

A part accepts at most 16 stages. Continue with separately viewable linked MP4
parts; there is no overall part-count or duration limit, and no unbounded single
MP4. Wait for the previous job to complete, then author new clips in `next.toml`
with the same model identity, canvas, fps, carry and audio settings. The first
stage must be `smooth`. For models that carry context, omit authored first
source images and frame-0 keyframes that conflict with the incoming tail.
Independent-clip models may use their ordinary authored inputs.

```bash
mold jobs show PART1_JOB_ID --json --host plato
mold run --script next.toml --continue-from PART1_JOB_ID --dry-run --host plato
mold run --script next.toml --continue-from PART1_JOB_ID --host plato --output /storage/mold/uat/part2.mp4
```

The output path belongs to the CLI machine; run on Plato for that Plato path.
The server also publishes every part in its normal `MOLD_HOME` Library.
The flag requires `--script`, refuses `--local`, and binds the latest completed
take. A stale predecessor revision is refused. Older servers without linked-part
capability are refused before submission; update the server first. To pin the latest completed take explicitly
in TOML, omit the flag and add `[chain.continue_from]` with `job_id` and
`finalize_take = 1` (use the actual latest take number). Place this table after
the chain settings and before `[[stage]]`. Retaking or amending the predecessor
invalidates that authority until a new latest take is completed. JSON/API
requests use the same `continue_from` object.
When reusing an exported script, remove its old `[chain.continue_from]` table
before selecting a new predecessor with the flag. Conflicting flag and TOML
authority is refused. A continuation job link remains
host-bound in exported scripts; its predecessor must exist on the same host.

The new job privately snapshots the last tail before creation, surviving prior
job garbage collection. Earlier clips are never rerendered or reencoded; the new
MP4 contains new clips only. LTX-2 carries latent motion, H3 FL2VA and
image-conditioned Wan carry the last RGB frame. H3 Ref2VA, text-only Wan and
legacy LTX-Video continue as independent clips. Duration estimates subtract the
initial carried boundary when the model actually carries context.

### HTTP API

Submit a canonical JSON `ChainRequest` to `POST /api/chain-jobs` using the
host’s normal authentication. Unlike TOML, JSON uses `stages` and has no
`schema` or `[chain]` wrapper:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "width": 512,
  "height": 512,
  "fps": 24,
  "seed": 42,
  "steps": 8,
  "guidance": 3.0,
  "strength": 1.0,
  "motion_tail_frames": 17,
  "output_format": "mp4",
  "stages": [
    { "prompt": "a cat wakes on a windowsill", "frames": 49 },
    { "prompt": "the cat lands on a rug", "frames": 49, "transition": "cut" }
  ]
}
```

Keep the returned `job_id`. Read `GET /api/chain-jobs/{id}` or follow
`GET /api/chain-jobs/{id}/events`; once `state` is `completed`,
`finalizes[-1].gallery_filename` names the stitched print. `mold jobs show ID --script`
uses authenticated `GET /api/chain-jobs/{id}?include_source_media=true` to
export a portable script with its retained input bytes. Ordinary detail and
event snapshots omit payloads; exports return `Cache-Control: private, no-store`.
TOML seeds and stage seed offsets accept integers or decimal strings; exports
quote values above the signed 64-bit TOML limit without changing their value.
H3 also publishes individual stage clips tagged `chain:<id>` and
`chain-stage:<index>`; its final stitched print remains the chain result.
Use `POST /api/generate/chain/validate` for read-only server validation.
The deleted `/api/generate/chain` and `/api/generate/chain/stream` render
routes are unavailable. Resume, retake and amend use the durable job routes.

H3 API stages use `keyframes: [{"frame": 0, "image": "BASE64"}]` for
FL2VA, or `generation_references` for Ref2VA. An image reference has
`kind: "image"`, `media: {"authority": "inline", "data": "BASE64"}`,
`mime_type`, `width` and `height`; video/audio references require their exact
probed timing fields as defined in `/api/docs`. Client filesystem paths are
CLI sugar and are never API media authority.

### Capabilities endpoint

```
GET /api/capabilities/chain-limits?model=<name>[&fps=<n>]
```

Returns the per-model caps the CLI and the chain API validate against:

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
than the clips an automatically chained one-shot would have produced.
`fps` echoes the frame rate the cap was computed at; it defaults to the
model's own default fps when the query omits it. It matters because LTX-2's
family ceiling is a runtime duration (`frames_per_clip_runtime_seconds`, 20 s),
so pass the fps you will actually render at and the cap moves with it.
`frames_per_clip_recommended` follows the model's own default frame count (97
for LTX-2, 25 for LTX-Video) so clients do not have to hardcode one.
`supports_sequence` is model-specific and is also advertised per model on
`GET /api/models`. A family with no chain path reports `false` with a
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
36 864 macroblocks, more than 2 073 600 macroblocks per second at the source
frame rate, or an edge over 4096 px) the encoder resamples it to the largest
frame of the same aspect ratio that fits. A 960×960 clip upscaled ×4 at 24 fps
therefore publishes at 3072×3072 rather than 3840×3840, and the print's
metadata records the published size beside the source size. A job paused
before this fit existed re-renders from frame 0 on resume rather than
stitching checkpoints of two sizes.

The MVP rejects rather than discards VFR timing, HDR/high-bit-depth video,
subtitles, chapters, multiple audio tracks, and incompatible audio codecs.
