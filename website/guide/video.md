---
title: Video Generation
---

# Video Generation

Mold supports generating video clips using LTX Video (including LTX-2.5), Wan
2.1/2.2, and MiniMax H3. The CLI and API can stitch authored clips into a
durable sequence. LTX-2, image-conditioned Wan and H3 FL2VA can carry context
across smooth seams; text-only Wan, legacy LTX-Video and H3 Ref2VA compose
independent clips. Duration jobs use the selected recipe's continuation or
blended-clip path to produce one longer movie. A dev checkpoint
renders its clips through the two-stage pipeline, so expect roughly twice the
wall time per clip as a distilled one; stage 1 runs classifier-free guidance
as two sequential forward passes.

An opening image conditions the first clip only. On LTX-2 and LTX-2.3 the
same image also rides every smooth continuation as a soft identity anchor;
on LTX-2.5, which was trained with keyframe conditioning and would read that
anchor as a keyframe to cut back to, continuations carry the motion tail alone.

## Longer videos

Enter one prompt and choose a length up to one generation: MiniMax H3 renders
up to 14.4 seconds in a single clip. For a longer movie, turn on **Longer video**
(native apps) or **Define scenes** (web and desktop) and give each scene its own
prompt and length. Keep each scene at or under one generation, describe only
what happens in that scene, and start it where the previous scene ended.

A single prompt longer than one generation still works, but every clip replays
that prompt, so its action repeats. The length summary warns when that happens
and when a scene is shorter than the shortest generation.

Library **Use settings** restores the original requested duration and optional
scenes, exact seed, recipe controls and authenticated retained inputs. It prepares
a fresh editable generation. For an extended movie, the duration describes the
added footage and the original movie is not attached; choose **Extend video**
to continue the full movie again. Scripted chains retain their per-clip settings
where recorded. Missing historical settings or inputs are disclosed before a new
generation, rather than silently shortening or flattening the movie.

```bash
mold --host workstation run MODEL "a paper boat drifts along a sunlit stream" --duration 12
mold --host workstation run MODEL --scenes scenes.toml --dry-run
```

A scenes file has one `[[scene]]` per part, each with `seconds` and its own
`prompt`; the movie length is their sum, so `--duration` is not needed:

```toml
[[scene]]
seconds = 12
prompt = """A paper boat drifts down a sunlit stream past mossy stones."""

[[scene]]
seconds = 12
prompt = """The boat tips over a small waterfall and spins in the foam below."""
```

The dry run prints the scene timeline, the clip count and any repeat warning.
For MiniMax H3, Mold opens every scene that continues from the previous clip's
last frame with the model's own frame-alignment line; do not write it yourself.

Choose `MODEL` from that host's `mold list --json` results. Duration is in seconds
and accepts decimal values. `--dry-run` checks the complete request without
queuing it; `--detach` returns after acceptance. Longer duration jobs require an
updated server, including H3-enabled builds for MiniMax H3. Use MP4 output and
omit frame-count, overlap and script overrides when choosing duration.

The server renders one bounded clip at a time and joins the result into one
Library MP4. Models with continuation support carry motion context; independent
clips use distinct deterministic seeds and a blended join. Overlap is chosen
from the selected recipe. There is no fixed overall duration or clip-count cap;
the host must have enough memory for each clip and enough disk for the movie.
Longer duration does not require keeping the entire movie in GPU memory.

### Extend a Library movie

Choose **Extend video** from a Library movie, enter the next prompt, and choose
how many seconds to **add**. Mold keeps the original movie and publishes a new
movie containing the original plus extension. It starts from the original
model and settings and validates the current host recipe before submission.

```bash
mold --host workstation run "the boat passes beneath a stone bridge" \
  --extend-library PRINT.mp4 --duration 10
```

The server publication goes to its active `MOLD_HOME` Library automatically.
`--output movie.mp4` downloads an additional copy onto the CLI machine.

### Recover a duration job

Keep the printed video UUID. After a lost reply or disconnect, inspect it before
submitting another movie. A completed `show --json` response contains the exact
`output_filename` to download.

```bash
mold --host workstation jobs show VIDEO_UUID --json
mold --host workstation jobs resume VIDEO_UUID
mold --host workstation jobs cancel VIDEO_UUID --video
mold --host workstation library download PRINT.mp4 --output ./
```

**Stop** pauses an accepted Duration job, retaining completed clips and its frozen
recipe for an explicit **Resume** on the original machine. Historical Cancelled
Duration jobs require **Start again** or **Send** when the server offers them.
A `410 VIDEO_OPERATION_CANCELLED` for an unanswered admission permanently
cancels that UUID; it cannot be resumed or reused.

**Start again** creates a new attempt with the original inputs and the
destination's current recipe. **Send to another machine** uses the selected
machine's current recipe:

```bash
mold --host SOURCE jobs restart VIDEO_UUID
mold --host SOURCE jobs restart VIDEO_UUID --to DESTINATION
```

Keep the new UUID too. Repeating the original submission with
`--operation-id UUID` on the same CLI installation reconciles its private local
receipt; a repeated `run` command still needs `--duration`. Use `jobs show` when
inspecting an ID from another client. Stop also works for a pending submission
that has not yet appeared in Queue. Local adapter files require the original
host. Failed jobs and their recovery inputs remain available until discarded.

The advanced script workflow below retains exact per-stage frame and transition
controls and can publish separate linked parts.

### Duration API

Capture the server's `instance_id` from `/api/status` and send it as
`x-mold-destination-instance` on every duration request. Check the additive
`video_jobs` capability block before offering this workflow. An older server
does not implicitly support duration jobs.

1. `POST /api/video-jobs/validate?planning_only=true` returns a cheap scalar
   length summary. It neither uploads media nor reserves a generation slot.
2. Prepare the selected recipe's normal media inputs, then
   `POST /api/video-jobs/validate` for full validation and its revision proof.
3. Generate and retain an operation UUID **before** submitting. Send the same
   request to `POST /api/video-jobs`, with `x-mold-operation-id` and the full
   validation revision. A lost reply is reconciled with that UUID.
4. Observe `GET /api/video-jobs/{id}` or its `/events` SSE stream. SSE uses
   `snapshot` events containing the full public job. A completed job exposes
   `output_filename` for its Library movie.

The duration payload wraps the normal generation request. Duration/count fields
use decimal strings so clients can retain their complete unsigned range. Choose
all generation controls from the selected host recipe. For example, this request
uses the installed H3 PDMD 2-step recipe, whose terminal-inclusive grid requires
three steps:

```json
{
  "generation": {
    "model": "minimax-h3-fl2va:comfy-pruned-int8-pdmd-2step",
    "prompt": "a paper boat drifts along a sunlit stream",
    "width": 768,
    "height": 768,
    "steps": 3,
    "guidance": 0,
    "strength": 1,
    "fps": 24,
    "enable_audio": true,
    "output_format": "mp4",
    "seed": 42
  },
  "duration_ms": "30000"
}
```

H3 requires guidance `0` and strength `1`; generic generation defaults do not
apply. Other H3 variants can require a different step count, so use the selected
recipe rather than copying this example to another model. Optional `scenes`
contain `prompt` and decimal-string `duration_ms`, summing exactly to the total.
The API preserves literal prompt text. Forward exported `request_json` and
`upload_spec_json` strings unchanged when a client cannot preserve full u64
numbers in parsed JSON.

`GET /api/video-jobs` provides bounded, installation-bound cursor pages.
`POST /api/video-jobs/{id}/cancel` durably accepts Stop even for a submission
whose acceptance is uncertain. `/resume` preserves the frozen recipe;
`DELETE /api/video-jobs/{id}` discards a settled queue entry while retaining any
published Library movie. `/restart-options`, `/restart-export` and
`/restart-source` support a fresh attempt with original inputs. Prefix upload
uses `/api/video-source-uploads`, bounded chunks and destination-side SHA
verification. Use the Rust SDK's transfer helpers or the CLI for this handshake.
The running server's `/api/docs` documents exact headers and response shapes.

### Wan sequences

What crosses a wan clip boundary depends on the checkpoint, not the family:
wan has no latent motion tail, so its smooth handoff is last-frame image
conditioning. `wan22-ti2v-5b` and `wan22-i2v-a14b` continue across the seam;
a text-to-video checkpoint concatenates independent clips (Join / Cut /
Crossfade). Clip lengths sit on wan's `4k+1` grid, and `--frames` past the
per-clip envelope auto-chains instead of failing — the last clip is sized so
the stitched video delivers exactly the frames you asked for.

A **text-to-video** tier hands no image context across a seam. Duration jobs
join independently seeded clips with blending. A direct frame-count one-shot
through the older generation/chain route still refuses requests past that
tier's clip size; a scripted sequence can also compose clips deliberately.

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

::: tip Scenes in the apps
On updated machines, use **Duration** with optional **Define scenes** in browser/Tauri apps, or **Length** with **Longer video** in native macOS/iOS. Each scene has its own prompt and length; the host validates the plan before generation. Older hosts and restored authored sequences keep the compact clip editor. Library **Continue video** makes a separate linked MP4 from the latest completed authored take on its original machine; **Extend video** instead publishes the original plus added footage as one new movie.
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

On updated machines, use **Duration** with optional **Define scenes** in browser/Tauri apps, or **Length** with **Longer video** in native macOS/iOS. Each scene has its own prompt and length; the host validates the plan before generation. Older hosts and restored authored sequences keep the compact clip editor. Library **Continue video** makes a separate linked MP4 from the latest completed authored take on its original machine; **Extend video** instead publishes the original plus added footage as one new movie.

Smooth is the default join. The selected model determines whether it carries motion or makes independent clips. Each part has bounded clips and memory; continue completed parts without a lifetime part limit. A continued part is a separate video. Unsupported source combinations are explained before submission.

Clips made by an older build's sequence composer retain their per-scene
provenance. **Use settings** restores the recorded ordered stages, effective
seeds, joins and inputs into an editable draft. Missing historical settings or
unavailable required media are disclosed before generation.

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

### Start a stopped video again

In the web, desktop and mobile Queue, choose **Start again…** on a failed, interrupted or paused multi-clip video. Keep the original machine or choose another connected machine, then confirm **Start again** or **Send video**. This creates a new attempt with its saved prompts, seed, settings and retained inputs; the original stays unchanged. A machine-local adapter requires the original machine. Unsupported or unavailable inputs explain why the action cannot proceed. After a dropped reply, **Check queue** recovers the known attempt instead of creating another one. **Stop new attempt** targets that destination only.

Web and Tauri Queue refresh saved video jobs automatically when a machine reconnects or its connection changes. **Check saved video jobs** also refreshes them on demand. Recovery controls stay unavailable until the original machine and installation are verified.

On updated machines, **Delete recovery data…** is available for a failed or interrupted multi-clip job. Its separate confirmation deletes that job and retained inputs; saved Library videos remain available. The original installation and job revision must still match. **Stop new attempt** requires the updated installation-fenced control; an interrupted cancellation reply keeps its intent for reconciliation with the same job ID.
