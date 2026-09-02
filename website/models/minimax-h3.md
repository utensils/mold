# MiniMax H3

MiniMax H3 is an audio-video generation family from
[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3). Mold can discover,
download, verify, repair, inventory, and remove two task partitions across
nine compact Comfy tags, plus two official BF16 qualification references. The
files are downloaded directly from their pinned Hugging Face repositories;
Mold does not bundle or mirror the weights.

## Generated Examples

Every clip contains synchronized audio generated with the video. Press play to
hear the model output.

<div class="gallery-grid">
<figure>

<video controls loop playsinline preload="metadata" poster="/gallery/minimax-h3-mold-speech-poster.webp" aria-label="A presenter turns toward the camera and speaks about Mold" src="/gallery/minimax-h3-mold-speech.webm"></video>

**minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p**: 992x992, 124 frames,
24 fps, seed 83009. The prompt uses H3's timed `<d>[English] ...</d>` dialogue
grammar: _"With Mold, your ideas render right here."_

</figure>
<figure>

<video controls loop playsinline preload="metadata" poster="/gallery/minimax-h3-balloon-poster.webp" aria-label="A Mold hot-air balloon moving through a misty sunrise valley with generated audio" src="/gallery/minimax-h3-balloon.webm"></video>

**minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p**: 992x992, 124 frames,
24 fps, seed 83004. First-frame conditioning preserves the balloon layout and
the word “MOLD” while adding camera motion and a generated soundscape.

</figure>
<figure>

<video controls loop playsinline preload="metadata" poster="/gallery/minimax-h3-foundry-15s-poster.webp" aria-label="Molten metal pouring into a foundry mold with generated narration and industrial audio" src="/gallery/minimax-h3-foundry-15s.webm"></video>

**minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p**: 992x992, 345 frames,
24 fps, seed 83031. A 14.38-second continuous foundry pour with generated
industrial sound, score, and prompted offscreen narration: _"With Mold, raw
ideas become living images."_

</figure>
<figure>

<video controls loop playsinline preload="metadata" poster="/gallery/minimax-h3-greenhouse-poster.webp" aria-label="Flowers opening inside a magical greenhouse with generated botanical audio" src="/gallery/minimax-h3-greenhouse.webm"></video>

**minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p**: 992x992, 175 frames,
24 fps, seed 83033. The conditioned greenhouse blooms around moving butterflies
and a hummingbird with synchronized water, wings, foliage, chimes, and score.

</figure>
</div>

::: warning CUDA is the supported runtime
Both compact variants can be downloaded on any Mold host. Mold's SM89 CUDA
release can run the compact FL2VA **and** Ref2VA models for the supported
request profiles below; the Apple Silicon Metal route below is admitted and
shipped but not yet hardware-qualified.
The CPU backend remains unavailable. Broader request
shapes also remain unavailable until those paths are implemented and
tested; Mold reports that limitation normally rather than treating it as a
licensing or authorization failure.
:::

::: info Apple Metal is a correctness-only path in progress
The Apple Silicon execution path exists as of #1164; family-scoped BF16, a
folded audio-VAE reduction, chunked dense attention sized so the score matrix
fits a Metal buffer, the portable INT8 ConvRot arm, and fp8-scaled weights
refused by name because candle has no Metal fp8 widening kernel. It is
advertised as **correctness-only**, the same tier Wan and LTX-2 landed on
before their performance qualification. Admission now accepts a Metal device,
the public runtime profile is `supported-compact-fl2va-cuda-sm89-or-metal`, and
the released macOS builds carry the `h3` feature. The route exists in a
shipped binary. What is still missing is hardware qualification:
no H3 checkpoint has ever completed a render on Metal. A Metal attempt is
refused below a unified-memory floor that the compact stack's ~42.5 GB working
set puts out of reach of a 48 GB machine, so lifting this tier needs a
64 GB-class Apple Silicon host. Expect Metal to be slow when it is qualified (the reference MLX
port measures minutes per step at 5 s) so this is a portability path, not a
speed one.
:::

## Compact variants

| Model                                                      | Task                                                   | Total pull | Runtime status                       |
| ---------------------------------------------------------- | ------------------------------------------------------ | ---------: | ------------------------------------ |
| `minimax-h3-fl2va:comfy-pruned-int8`                       | First/last-frame conditioning with audio               |  42.482 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step`           | FL2VA + reviewed Turbo 8-step LoRA (9 steps)           |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`      | FL2VA + reviewed Turbo 4-step 768p LoRA (5 steps)      |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1` | FL2VA + reviewed Turbo 4-step 768p v1.1 LoRA (5 steps) |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p`      | FL2VA + reviewed Turbo 8-step 768p LoRA (9 steps)      |  44.438 GB | CUDA generation; first-frame profile |
| `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step`          | Ref2VA + reviewed Turbo 4-step LoRA (5 steps)          |  44.438 GB | CUDA generation; reference profile   |
| `minimax-h3-ref2va:comfy-pruned-int8`                      | Reference media to video with audio                    |  42.482 GB | CUDA generation; ordered references  |
| `minimax-h3-fl2va:comfy-pruned-nvfp4`                      | First/last-frame conditioning with audio               |  34.040 GB | Downloadable; execution unavailable  |
| `minimax-h3-ref2va:comfy-pruned-nvfp4`                     | Reference media to video with audio                    |  34.040 GB | Downloadable; execution unavailable  |

The official `minimax-h3-fl2va:official-bf16` and
`minimax-h3-ref2va:official-bf16` identities are also visible downloads. They
are large qualification references with no public execution arm, so their
model rows report `runtime_available: false` before the pull.

The two NVFP4 rows pin a pruned NVFP4 transformer in place of the INT8 one.
Mold has no engine arm that reads that weight layout yet, so they download,
verify, appear in **Models → Installed**, and remove like any other model,
while generation is refused up front (before any weights are loaded) and
`GET /api/models` reports `"runtime_available": false` on the row. They share
the whole rest of the compact stack with the INT8 variants, so if you already
have one installed the pull is only the 12.529 GB transformer.

## Which models this build can run

The H3 catalog rows ship on every release target; the H3 _engine_ does not.
The macOS and Linux sm89 artifacts are built with it; on an RTX 3090/A40
(sm86), a B200/B300 (sm100), an RTX 50-series card (sm120), or Windows, the H3
models download and verify normally and generate nothing. Both compact task
partitions (FL2VA and Ref2VA) run wherever the engine is built.

Rather than let you discover that after a 21–42 GB pull, every H3 row carries
its answer. `GET /api/models` reports:

- `runtime_available`: `false` when this server cannot execute the model,
  whatever the cause. Absent on servers that predate the field, which clients
  read as runnable.
- `runtime_unavailable_reason`: one sentence naming the obstacle, present
  exactly when `runtime_available` is `false`. There are three, and they have
  three different remedies:
  - **no engine arm for this weight layout**: the `official-bf16`
    qualification references and the pruned NVFP4 tags. No build runs these.
  - **no runtime for this task partition**: the task, not the machine. A
    different artifact will not help. No released identity reports this today:
    both compact partitions execute, and the axis survives for a future task.
  - **this build was compiled without the H3 engine**: use the macOS or
    Linux sm89 release, or build with the `h3` feature.

Mold Studio renders that on the model card _before_ the pull: web, desktop,
and the iPhone app show a **Download only** badge on the Discover row and the
full sentence in the detail pane, with the Pull action still enabled; the
model genuinely is downloadable. `mold pull` prints the same sentence in place
of the `mold run` hint, and `mold run --local` refuses before opening a single
checkpoint.

Submitting one anyway returns HTTP `501` with code
`MINIMAX_H3_RUNTIME_UNAVAILABLE` and that same sentence; deliberately not the
`451` licensing refusal, because none of these is a licensing problem.

Every H3 render carries synchronized generated audio, and no request can turn
it off. `GET /api/models` says so directly: each H3 entry reports
`"supports_audio": true` (including variants that are not downloaded yet) so
a client reads the capability rather than inferring it from the family name.

Pull a variant from the CLI, or install it from **Models → Discover** in Mold
Studio:

```bash
mold pull minimax-h3-fl2va:comfy-pruned-int8
mold pull minimax-h3-fl2va:comfy-pruned-nvfp4
mold pull minimax-h3-fl2va:comfy-pruned-int8-turbo-8step
mold pull minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1
mold pull minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p
mold pull minimax-h3-ref2va:comfy-pruned-int8
```

The files are revision-pinned and SHA-256 verified before Mold marks the model
complete. Raw repository IDs, custom manifests, configured aliases, and live
catalog recipes cannot substitute for any registered graph.

## Reviewed Turbo tiers

A Turbo tier is a reviewed LoRA adapter overlaid on the **same** compact INT8
checkpoint of its own task; nothing about the base artifact contract relaxes,
and the only request axis a tier moves is its fixed step count. Each Turbo
model tag pulls the complete base stack of its task plus one pinned adapter
(1,956,193,000 bytes for FL2VA 8-step, Ref2VA 4-step, and FL2VA 8-step 768p,
1,956,192,992 bytes for FL2VA 4-step 768p and its v1.1 successor, stored once
under `shared/minimax-h3/loras/` and shared by every tag that names one):

- `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step`: 9 terminal-inclusive
  sampler grid points (8 model evaluations)
- `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p`: 5 terminal-inclusive
  sampler grid points (4 model evaluations)
- `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1`: 5
  terminal-inclusive sampler grid points (4 model evaluations)
- `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p`: 9 terminal-inclusive
  sampler grid points (8 model evaluations)
- `minimax-h3-ref2va:comfy-pruned-int8-turbo-4step`: 5 terminal-inclusive
  sampler grid points (4 model evaluations)

An adapter is reviewed for exactly one task partition, so a `ref2v` adapter
can never mint an FL2VA qualification and vice versa.

A Turbo tag stores its base stack in its own task's base checkpoint directory
(`minimax-h3-fl2va-comfy-pruned-int8/` or `minimax-h3-ref2va-comfy-pruned-int8/`,
plus the shared family bucket), so a machine that already has that base tag
installed downloads only the ~1.96 GB adapter, and pulling a Turbo tag first means a
later base pull downloads nothing. Because the shared bytes genuinely
constitute a complete base install, a Turbo-only pull also makes the base tag
read as installed. Removal ref-counts every complete install: removing a
Turbo tag deletes only its adapter while the base stack has another owner,
and removing the base keeps every file a fully installed Turbo tag still
uses (a Turbo tag missing its adapter owns nothing). Freeing everything is
two removals (the Turbo tag, then the base) and each removal reports the
kept files with the tags that still use them.

Selecting a Turbo model resolves the base INT8 transformer, the tier's pinned
adapter, and the tier's reviewed step count with no extra configuration. The
Create screen locks Steps at that count and prints the reason underneath it
("Fixed by the 8-step Turbo tier: 9 terminal-inclusive sampler grid points
(8 denoise intervals)."), so the `9` is not a surprise; Guidance is locked the
same way, because H3 has no classifier-free branch and pins the scale at 0.
Both sentences come from the server's generation profile, so the web, desktop,
and iPhone apps show the same words. The
`MOLD_H3_TURBO_ADAPTER` / `MOLD_H3_TURBO_TIER` environment pair is a
capture-scope UAT override honored only by `h3-private-uat` builds; ordinary
builds refuse a set pair rather than letting two selection authorities
disagree.

## Download size and sources

Each compact variant has the same component graph except for its task-specific
transformer:

| Component                                              |              Bytes |  Decimal size | Upstream source                                                                                                                                               |
| ------------------------------------------------------ | -----------------: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task transformer (INT8 ConvRot)                        |     20,970,379,616 |     20.970 GB | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/eb8a16107c595128b3a578f82d2ce2f75920c355)                                           |
| Task transformer (pruned NVFP4)                        |     12,528,636,865 |     12.529 GB | [`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/tree/908eccad7e68751190d04c171956f163bfeed741) |
| Qwen3-VL NVFP4-AWQ text encoder                        |     15,687,142,551 |     15.687 GB | `Comfy-Org/MiniMax-H3`                                                                                                                                        |
| FP16 video VAE                                         |      5,207,808,496 |      5.208 GB | `Comfy-Org/MiniMax-H3`                                                                                                                                        |
| FP32 audio VAE                                         |        605,254,808 |      0.605 GB | `Comfy-Org/MiniMax-H3`                                                                                                                                        |
| Tokenizer, processor, scheduler, and component configs |         11,504,847 |      0.012 GB | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/bfc8ed0353f5a9733be73e6b2c98ec0948195b86)                                           |
| **One complete INT8 variant**                          | **42,482,090,318** | **42.482 GB** | Both pinned repositories                                                                                                                                      |
| **One complete NVFP4 variant**                         | **34,040,347,567** | **34.040 GB** | All three pinned repositories                                                                                                                                 |

A Turbo tag adds one adapter to this graph:
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/dc559027db79c174125df4d827db55cd11178860)
`loras/` at pinned revision `dc559027db79c174125df4d827db55cd11178860`
(1,956,193,000 bytes for `-turbo-8step`, 1,956,192,992 bytes for
`-turbo-4step-768p`), bringing one complete Turbo variant to 44,438,283,318 or
44,438,283,310 bytes (44.438 GB).

The `-turbo-4step-768p-v1.1` and `-turbo-8step-768p` tags pull their adapter
from a second third-party source instead:
[`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/05ef678438e84933c406131b59abbf86919b3aac)
at the repository ROOT (no `loras/` directory) at pinned revision
`05ef678438e84933c406131b59abbf86919b3aac` — `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors`
(1,956,192,992 bytes) and `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`
(1,956,193,000 bytes), bringing those Turbo variants to 44,438,283,310 and
44,438,283,318 bytes (44.438 GB) respectively. lightx2v declares `apache-2.0`
for the adapters themselves; the MiniMax H3 Community License still governs
the base checkpoint each tag executes on. Every adapter, whichever source it
comes from, lands at the same `shared/minimax-h3/loras/` path keyed by its own
basename.

The encoder, VAEs, and common support files are shared between every compact
variant, INT8 and NVFP4 alike. After one complete variant is installed, adding
another downloads only its transformer (20.970 GB for an INT8 tag, 12.529 GB
for an NVFP4 one) plus a 546-byte task config when the task differs. Removing
one variant keeps the shared graph and names the variants still using it. Both variants together occupy 63.452
GB (63,452,470,480 bytes) of model payloads, excluding filesystem and Hugging
Face cache overhead.

Sizes above are decimal gigabytes (`1 GB = 1,000,000,000 bytes`) and describe
downloads and disk use, not peak VRAM. They come from Mold's registered,
full-file manifest identities rather than estimates from repository listings.

## Supported FL2VA request

The current compact implementation supports this request profile:

- an SM89 CUDA GPU with sufficient VRAM and the H3 attention/runtime operators
  enabled (an Apple Silicon Metal GPU is admitted but unqualified; see above)
- any canvas the compact rule admits (both axes a multiple of 32, each at
  least 256 px, at most 1,032,192 pixels in total (the area of `1344x768`),
  aspect between 1:4 and 4:1) batch size 1
- 107 to 345 frames on the `17n+5` grid at 24 fps (124 is the default)
- 2 to 50 terminal-inclusive sampler grid points for the base model (21 is the
  default); a reviewed Turbo tag instead requires exactly its tier's own count
  (9 for `-turbo-8step` and `-turbo-8step-768p`, 5 for `-turbo-4step-768p` and
  `-turbo-4step-768p-v1.1`), because that count is the distilled adapter's own
  schedule length
- one required first-frame image; the current compact runtime refuses a
  closing endpoint
- MP4 output with synchronized generated audio
- a prompt of roughly 1,000 tokens or fewer: the reviewed conditioner sequence
  budgets 2,048 rows, of which the first-frame image's vision pads and label
  take 1,014, and a longer prompt is refused immediately with its exact budget
  named rather than after artifact verification

```bash
mold run minimax-h3-fl2va:comfy-pruned-int8 \
  "the camera drifts toward the illuminated pavilion" \
  --first-frame pavilion.png --duration 5
```

### Canvas and duration

The checkpoint accepts any 32-aligned canvas between 1:4 and 4:1, and clip
lengths from 107 to 345 frames. Mold admits that whole space and lets the
memory estimate decide what actually fits, rather than pinning the shapes a
hardware campaign happened to run.

The canvas rule is:

- both axes a multiple of 32 (one packed row is a 32x32 pixel cell)
- each axis at least 256 px
- at most **1,032,192 pixels** in total; the area of `1344x768`, which is what
  every memory measurement below was captured at, so a larger canvas would have
  to be priced by extrapolation
- aspect between 1:4 and 4:1

`1344x768` and `768x768` remain the recommended defaults because they have real
hardware evidence. Measured on an RTX 4090 24 GB at 124 frames / 24 fps, wall
clock from request to MP4 bytes on a cold process:

| Canvas     | Aspect | Base tier (21 steps) | `-turbo-8step` (9 steps) |
| ---------- | ------ | -------------------- | ------------------------ |
| `1344x768` | 7:4    | 1216 s, 10.8 GB VRAM | 759.5 s, 13.5-14.6 GB    |
| `768x768`  | 1:1    | 937 s, 7.4 GB VRAM   | 664 s, 9.2 GB VRAM       |

The two 768p-native Turbo tiers added after that campaign,
`-turbo-4step-768p-v1.1` and `-turbo-8step-768p`, were measured separately on
plato (NVIDIA L40S 46 GB, not the hal9000 RTX 4090 above) during the lightx2v
Turbo tier UAT campaign (2026-09-02), same 124 frames / 24 fps, wall clock
from request to MP4 bytes on a cold-to-warm process, VRAM peak from a 1 Hz
`nvidia-smi` sampler, both tiers at their default shift 6:

| Canvas     | Aspect | `-turbo-4step-768p-v1.1` (5 steps) | `-turbo-8step-768p` (9 steps) |
| ---------- | ------ | ---------------------------------- | ----------------------------- |
| `1344x768` | 7:4    | 269.9 s, 12.6 GiB VRAM             | 453.3 s, 12.7 GiB VRAM        |
| `768x768`  | 1:1    | 206.4 s, 9.3 GiB VRAM              | 344.9 s, 9.2 GiB VRAM         |

Full per-render evidence — host/driver/binary provenance, adapter pull facts,
`scheduler_estimates` rows, and visual verification for these two tiers plus
the shift-6-vs-shift-12 A/B on the v1.1 tier — is recorded in the
[qualification record](https://github.com/utensils/mold/blob/main/docs/qualification/minimax-h3.md#the-lightx2v-turbo-tiers-campaign-2026-09-02).

Every other shape is priced by scaling those measurements: the denoise
workspaces with the packed sequence, the audio decode with the clip length, and
the video decode with the canvas area. A long clip therefore costs real VRAM:
345 frames at `1344x768` asks for a device floor of about 24.3 GB against 9.7 GB
at the default, and a host that cannot supply it is refused with those numbers
rather than by a rule.

Note that a Turbo tag costs slightly _more_ VRAM than the base tier on the same
canvas: it is the same compact stack plus a resident adapter, and the step
count it moves buys time, not memory.

When a first-frame image is attached without explicit `--width`/`--height`, the
CLI and Discord builders render the source's own aspect at the largest size the
area ceiling allows; a 16:9 source gets `1312x736`, a square one `992x992`;
and the engine fits the frame internally. The free-form aspect-derived
short-edge canvas applies only to the download-only official BF16 references.

Create on web, desktop, and iPhone offers the recommended canvases as shape
chips and a live frame slider, and a request outside the rule is still refused
at submission instead of after the model loads.

Mold rejects rather than silently resizing, rerouting, changing steps, dropping
the source image, or falling back to another backend. A downloaded checkpoint
can remain stored on an unsupported host; Create and request routing become
available only when that host advertises the matching runtime capability.
The public runtime uses a source-controlled conservative memory profile;
it does not require private authorization or qualification-record files.

## Supported Ref2VA request

`minimax-h3-ref2va:comfy-pruned-int8` conditions on an **ordered set of
references** instead of a boundary frame. Everything about the generated side
is FL2VA's (the same canvas rule, the same `17n+5` frame grid at 24 fps, the
same 2-50 sampler grid points, MP4 with synchronized audio) and the
conditioning side is the set:

- 1 to 12 references in total, at most 9 images, 3 videos, and 3 audio files
- each reference between 2 and 15 seconds where it has a duration; video and
  soundtrack references are truncated to the generated clip's own length
- images larger than a 2048 short edge are scaled down onto their own
  2048-short-edge canvas and smaller ones keep their native geometry (never
  upscaled, matching ComfyUI); videos are likewise scaled down onto the
  reference canvas only when they exceed it; every soundtrack is resampled to
  32 kHz stereo
- **order is authority.** The set is presented to the conditioner as
  `<Picture n>`, `<Video n>`, and `<Audio n>` in the order you supply, and the
  frozen plan carries a reference fingerprint over that order. The same
  files in a different order are a different render, not the same one.

An image reference can be **cropped** before it is sent: the reference row's
**Crop** action on web, desktop, and iPhone opens a drag rectangle with Free,
1:1, 4:3, 3:2, and 16:9 presets, a 64 px minimum per axis, and a live
vision-pad cost hint (a 1:1 crop of a 1080p photograph is 1,156 pads instead
of 2,040). The crop is applied at the photograph's original resolution before the
reference is digested and uploaded, so the server only ever sees the cropped
image; it is recorded in the print's metadata as `references[].crop` and Reuse
settings restores it when you reattach the same original. This is a choice of
_which part of the photograph is the reference_, never a fit to the output
canvas; Mold still scales an oversized cropped image down onto its own
2048-short-edge canvas (never up), and the generated print's size is unchanged.

There is no reviewed list of reference sets. The runtime qualification is
minted per request from the set's own preprocessing shapes: the conditioner
sequence, the conditioning latents, and every memory bound scale with what you
actually attached, and a set the device cannot hold is refused with the numbers
rather than by a rule. A prompt still gets roughly a thousand tokens on top of
the references' own vision pads.

From the CLI, each reference is a `KIND=PATH` pair and the kinds are `image`,
`video`, and `audio`:

```bash
mold run minimax-h3-ref2va:comfy-pruned-int8 "a slow dolly through the scene" \
  --reference image=hero.png \
  --reference video=clip.mp4 \
  --reference audio=score.wav \
  --width 1344 --height 768 --frames 124 --fps 24 \
  --steps 21 --guidance 0 --strength 1.0 --format mp4
```

References are uploaded through the authenticated streaming reference-upload
endpoints, so `MOLD_API_KEY` must be configured; Mold never puts reference
bytes in the request body, the queue journal, or saved metadata; only their
redacted metadata and digests. An upload session binds one request, so
`--batch N` above 1 is refused with uploaded references — submit siblings one
at a time.

## License and support boundary

H3 model weights and upstream model assets use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/bfc8ed0353f5a9733be73e6b2c98ec0948195b86/LICENSE),
at pinned revision `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`, not
Mold's MIT license. Review those terms for your intended use. Mold's source code
and H3 integration remain under Mold's repository license. Mold downloads the
weights directly from the pinned upstream repositories and verifies every file;
it does not bundle or mirror the payloads in Mold releases. The project's
[license and integration record](https://github.com/utensils/mold/blob/main/docs/architecture/minimax-h3-authorization.md)
documents the completed governance decision.

The completed project review authorizes H3 use in every territory and across
Mold's CLI, server/API, Discord, desktop, web, iPhone, TUI, gallery,
remote-client, shared-server, and hosted paths. It also covers generated-output
distribution and model distribution or redistribution. Technical availability
remains limited to routes Mold has implemented and qualified; authorization
does not make an unsupported task, device, or request shape runnable.

The license link and notice in this guide and Mold's README are the project's
required user-facing license, attribution, disclosure, downstream-term, and
acceptable-use delivery. Mold does not require a separate clickthrough,
geolocation check, H3-specific generated-content label, downstream contract, or
surface-specific acceptable-use control. Existing Mold authentication,
validation, capability, safety, and operational controls continue to apply.

The official BF16 checkpoints remain download-only qualification references.
Their much larger artifact graphs are visible to preserve exact acquisition
and inventory authority, but no released build executes them.
