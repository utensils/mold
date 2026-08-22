---
name: mold
description: Generate and manage AI images and video with the mold CLI. Use when asked to create images or clips, transform source media, operate local or remote Mold servers, manage models and queues, inspect galleries, configure GPU inference, or automate any Mold CLI, REST, or MCP workflow.
allowed-tools: Bash, Read, Glob, Grep
---

# mold — Local AI Image Generation CLI

Generate images and video from text prompts using FLUX, SD1.5, SDXL, SD3.5, Z-Image, Flux.2 Klein and Dev, Qwen-Image, LTX Video, LTX-2 / LTX-2.3, Wan 2.1/2.2, and Wuerstchen diffusion models running on local GPU hardware.

MiniMax H3's reviewed compact manifest IDs — FL2VA, Ref2VA, and the FL2VA Turbo tags `minimax-h3-fl2va:comfy-pruned-int8-turbo-8step` (9 steps) and `minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p` (5 steps), each the same compact stack plus one pinned LoRA adapter — are ordinary model identities on every build; list, download, repair, and auto-pull work like any registered model, and remaining execution limits (Ref2VA execution, CPU) are reported as real capability limits, not license gates. Every H3 render carries synchronized audio and no request can disable it, and `/api/models[].supports_audio` (plus the profile recipe's `supports_audio`) advertises that for every reviewed identity, downloaded or not — read the field rather than inferring the family's AV behaviour from its name. The reviewed FL2VA request envelope budgets a prompt of roughly 1,000 tokens beside the required first-frame image; a longer prompt is refused immediately, with the exact budget named, rather than after artifact verification. The `MOLD_H3_TURBO_ADAPTER`/`MOLD_H3_TURBO_TIER` pair is a capture-scope UAT override honored only under `h3-private-uat`; select a Turbo model tag instead. Apple Metal is a correctness-only path in progress (#1164): the candle execution route exists — chunked dense attention, family-scoped BF16, the portable INT8 arm, fp8 refused by name — and `mold-core::minimax_h3` advertises `metal: CorrectnessOnly`, but the public runtime profile is still SM89 CUDA and Metal execution waits on hardware qualification. Build it with `h3` + `metal`; `h3-cuda` is the shipping CUDA edge. Everything outside those exact IDs stays fail-closed through `mold-core::model_policy`: raw `hf:` identities and metadata-resolved `cv:` identities must both pass it, and no environment variable, client flag, or local weight can activate an unreviewed H3 identity. The completed #831 governance decision permits H3 use in every territory and workflow (local, remote, hosted, redistribution) with no H3-specific clickthrough, geolocation, labeling, or surface control; the README and H3 user guide carry the full user-facing license/attribution/disclosure text. Re-review `docs/architecture/minimax-h3-authorization.md` before any release that touches H3.

The native apps' public privacy policy is sourced at `website/privacy.md`,
published at `https://utensils.io/mold/privacy`, and linked from the desktop and
iPhone Settings → About surfaces. Keep the page and app links aligned when data
practices change.

## Quick Reference

```bash
mold run "a cat on a skateboard"                    # Generate with default model
mold run flux-dev:q4 "a sunset over mountains"      # Specific model
mold run "a portrait" -o portrait.png               # Custom output path
mold run "a dog" --seed 42 --steps 20               # Reproducible generation
mold run "a village" --title "Smurf village"        # Titled print (metadata + gallery row + ~slug in filename)
mold run "a village" --title "Smurfs" --tag blue --collection "Village studies"  # …filed at creation
mold run "watercolor" --image photo.png --strength 0.7  # img2img
# NOTE --strength is family-specific: SD img2img higher = more change;
# LTX-2 I2V higher = MORE source preservation (1.0 pins the opening frame)
mold run qwen-image-edit-2511:q4 "make the chair red leather" --image chair.png --image swatch.png --qwen2-variant q4
mold run qwen-image:q2 "a poster" --qwen2-variant q6    # Qwen-Image quantized text encoder
mold run flux-dev:bf16 "portrait" --lora style.safetensors --lora-scale 0.8  # LoRA adapter
mold mcp --host http://localhost:7680                # Stdio MCP bridge for LM Studio
mold lambda deploy --instance-type gpu_1x_a10 --region us-west-1  # Private Lambda Cloud web UI
mold gpu list                                       # Stable GPU/MIG inventory
mold gpu disable cuda:<stable-id>                   # Drain, then disable
mold gpu enable cuda:<stable-id>                    # Re-enable runtime scheduling
mold trash list                                     # Trashed prints on $MOLD_HOST with purge countdowns
mold trash restore <filename>...                    # Back to the live gallery
mold skill install --detected                       # Install this skill for detected agents
```

`mold mcp` exposes synchronous image generation, async generation with status
polling, gallery search/fetch, model and LoRA listing, and server status tools.

## How to Use This Skill

Parse `$ARGUMENTS` to determine the action:

- If arguments look like a **prompt** (natural language), run `mold run "<prompt>"` with sensible defaults
- If arguments start with a **subcommand** (`pull`, `list`, `default`, `config`, `serve`, `server`, `mcp`, `info`, `ps`, `rm`, `unload`, `update`, `stats`, `clean`, `tui`, `completions`, `version`, `runpod`, `lambda`, `jobs`, `gpu`, `trash`, `skill`), run that subcommand
- If arguments include **flags** (`--model`, `--image`, `--steps`, etc.), pass them through

## Generating Images

### Basic Usage

```bash
# Default model (flux2-klein:q8 — fast 4-step, Apache 2.0, fully ungated)
mold run "a red apple on a wooden table"

# Explicit model selection
mold run flux-dev:q4 "a photorealistic landscape at golden hour"

# With output path
mold run "cyberpunk cityscape" -o cityscape.png

# Reproducible output (same seed = same image)
mold run "a cat" --seed 42

# Custom dimensions (must be multiples of 16)
mold run "a banner" --width 1024 --height 512

# Batch generation (multiple images)
mold run "abstract art" --batch 4 --seed 100 -o art.png

# JPEG output
mold run "a sunset" --format jpeg -o sunset.jpg

# Disable PNG metadata embedding
mold run "a cat" --no-metadata

# Display image inline in terminal after generation (requires `preview` feature)
mold run "a cat" --preview

# Negative prompt (CFG-based models: SD1.5, SDXL, SD3, Wuerstchen, Wan)
mold run sd15:fp16 "a portrait" -n "blurry, watermark, ugly, bad anatomy"
mold run sdxl:fp16 "a landscape" --negative-prompt "low quality, jpeg artifacts"
# --no-negative sends an explicit empty negative, disabling config defaults
# AND wan's tuned model default (advertised per model via
# /api/models[].default_negative_prompt and prefilled on every surface)
mold run wan22-t2v-a14b:q5 "a cat" --no-negative
```

### Prompt Expansion

Expand short prompts into detailed image generation prompts using a local LLM (Qwen3-1.7B). The expansion model auto-downloads on first use (~1.8GB).

```bash
# Preview expanded prompt without generating
mold expand "a cat"

# Expand with multiple variations
mold expand "cyberpunk city" --variations 5

# Expand as JSON
mold expand "a cat" --variations 3 --json

# Preview conditioned video expansion without attaching media
mold expand "she turns" --model ltx-2-19b-distilled:fp8 --task image-to-video

# Generate with expansion (short prompt -> detailed prompt -> image)
mold run "a cat" --expand

# Batch + expand: each image gets a unique expanded prompt
mold run "a sunset" --expand --batch 4

# Use a specific expansion backend (OpenAI-compatible API)
mold run "a cat" --expand --expand-backend http://localhost:11434/v1

# Disable expansion (overrides config/env default)
mold run "a cat" --no-expand
```

The expansion model is dropped from memory before diffusion begins, so it doesn't compete for VRAM.

### LoRA Adapters

Apply LoRA (Low-Rank Adaptation) fine-tuned adapters on top of FLUX BF16 base models:

```bash
# Basic LoRA usage
mold run flux-dev:bf16 "a portrait" --lora /path/to/adapter.safetensors

# Adjust LoRA strength (0.0 = no effect, 1.0 = full, up to 2.0)
mold run flux-dev:bf16 "anime style" --lora style.safetensors --lora-scale 0.7

# LoRA with other options (img2img, seed, etc.)
mold run flux-dev:bf16 "oil painting" --lora art.safetensors --image photo.png --strength 0.6

# Stack multiple LoRAs (deltas merge additively: W' = W + Σ scale_i · B_i @ A_i)
mold run flux-dev:bf16 "epic shot" \
  --lora cinematic.safetensors --lora-scale 0.8 \
  --lora dramatic-lighting.safetensors --lora-scale 0.4

# Qwen-Image Lightning: the few-step distill as an adapter on a checkpoint you
# already have, instead of a second ~20 GB pre-merged transformer. Always
# CFG-free — set --steps to the adapter's step count and --guidance 1.0.
mold run qwen-image-2512:q8 "a snowy mountain cabin at twilight" \
  --lora ~/loras/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors \
  --steps 8 --guidance 1.0
```

**Qwen-Image Lightning adapters** (user-supplied files, no manifest entry — the
pre-merged route is `qwen-image-lightning:fp8` / `:fp8-8step` and
`qwen-image-edit-lightning:fp8`):
`lightx2v/Qwen-Image-Lightning` ships
`Qwen-Image-Lightning-{4,8}steps-V2.0-bf16.safetensors` for the `qwen-image:*`
line, and `lightx2v/Qwen-Image-2512-Lightning` ships
`Qwen-Image-2512-Lightning-{4,8}steps-V1.0-bf16.safetensors` for
`qwen-image-2512:*`. Match the adapter's line to the checkpoint's. ModelTC
authors the distill; `lightx2v` is the Hugging Face org that hosts the weights,
so the HF path is always `lightx2v/...`. These step/guidance recipes are
upstream's — no on-disk Qwen-Image adapter is in mold's test suite yet.

**Repeated stacks reuse the merge:** FLUX and Qwen-Image fingerprint the active
adapter set, its order, and its scales; a still-resident transformer built with
exactly that stack is reused instead of rebuilt — which for GGUF is a
dequantize → merge → re-quantize of every affected tensor. Any change to the
stack invalidates. Only resident paths can elide: Qwen-Image's sequential
load-use-drop strategy and any render whose VAE decode drops the transformer
pay the merge again on the next request (FLUX's `LoraDeltaCache` /
`MOLD_LORA_BYPASS` have no Qwen equivalent yet), and Qwen block offload refuses
LoRAs outright.

**Models browse:** the desktop's single **Models** workspace (Installed +
Discover segments) stacks installed
models above the live catalog with **All / Images / Video** media chips, a
model-kind chip row (Models / LoRAs / CLIP / Text encoders / VAEs / Tokenizers
/ ControlNet), a Downloads / Rating / Recent sort, and
Grid / Table layouts; active downloads pin to the top with a source glyph and
target host. Cards, table rows, and details always name the model kind, and
mature entries use the explicit `18+ NSFW` badge; details show descriptions,
tags, license, source, format, and popularity only when the catalog provides
them. The iPhone Models view shares the same metadata treatment and kind/sort
options. The Installed segment merges every ready host with host badges and
host-routed actions; host detail mirrors that host's active pulls. Rows label
the actual kind's weights separately from the footprint including shared
runtime files.
Curated manifest variants replace ambiguous multi-checkpoint Hugging Face
repositories so Pull always targets one runnable model. Live Hugging Face LoRA
collections also select one preferred adapter rather than pulling every
mutually exclusive step/precision variant; the desktop treats an accepted
response with no queued job as an error.
Every remembered host reconnects immediately on launch, independently of the
local engine; failed hosts remain listed and polling retries them.
The catalog uses cache-stable 512 px Civitai derivatives plus
lazy async decoding and per-card layout/paint containment instead of
source-resolution preview images.
Entries without working preview art use a local model-family mark instead of a
blank card or another image request.
The Discover segment proxies live HF + Civitai searches (filter by **LoRAs** to narrow).
Install with `mold pull cv:<id>` / `mold pull hf:<author>/<repo>` or Pull in
the Models workspace. With several ready desktop hosts, Pull asks for the destination. Remote
desktop catalog requests can carry request-scoped HF/Civitai fallback tokens;
server env tokens retain precedence and forwarded values are not persisted.
Once installed, the LoRA appears in the **Create → LoRA**
picker for any compatible model family. The CLI no longer ships a `mold catalog`
subcommand — every read is live, no scan to run.

**MCP / REST discovery:** `GET /api/loras?model=<name>` and the MCP
`list_loras` tool return installed compatible LoRAs with ids and server-side
paths. MCP generation accepts ids, paths, or objects like `{ "id": "cv:827325" }`
and `{ "path": "...", "scale": 0.8 }`; omitted scale defaults to `1.0`.

**Web UI multi-LoRA + trigger words:** the LoRA picker stacks up to 4 adapters
per generation (each with its own scale slider) and supports drag reordering
with the request preserving visual stack order. Civitai LoRAs ship trigger
phrases (`trainedWords`); they render as click-to-insert chips beside the
selected LoRA — clicking a chip appends the phrase to the active prompt.

**Model install targets (web, desktop, iPhone):** a Models row is merged
across machines, so `installed` means "installed somewhere", never "installed
here". Every surface derives its action from `planModelInstall`
(`studio/lib/modelInstallTargets.ts`): each reachable machine missing the
model stays an install target, owners degrade to repair, and the action
collapses to Repair only once every reachable machine has it. Never hide the
install because another machine has the model. The destination picker names
the per-machine outcome, and on desktop and web a machine whose `/api/models`
has not been read is never offered as an install target. Route every download
on the id shape: `/api/downloads` takes manifest names, `/api/catalog/:id/download`
takes `cv:` / `hf:` ids, and each rejects the other with HTTP 400.

**Create web UI:** the Create tab only lists downloaded standalone
generation models; the Models tab is the install/repair surface for missing
models and companions. Web and desktop selectors label installed `cv:` / `hf:`
entries with their catalog descriptions while keeping those ids as the actual
form, routing, and generation values. The controls rail mirrors placement controls, exposes
family-specific schedulers, uses installed upscaler dropdowns, shows
server-derived peak-memory estimates, reports component readiness via
`GET /api/models/:model/components`, and can save/load named web-local
generation templates. The running strip consumes `GET /api/queue` and
`PATCH /api/queue/:id` to render queued/running work in GPU lanes and change
the real queued-job dispatch order.

**Requirements:**

- FLUX (BF16 + GGUF), Flux.2, LTX-2, SD1.5, SD3, SDXL, Qwen-Image (+ Qwen-Image-Edit),
  Wan, and Z-Image families support LoRAs. Wuerstchen and LTX-Video are not yet wired —
  the server returns a 400 with the list of supported families for any other family.
  (The message text is centralised in `mold-core::validation::require_lora_capable_family`
  and will list whichever families are wired at the time it's surfaced.)
- Wan adapters cover low-rank pairs and full-weight `.diff`/`.diff_b` deltas: bf16
  safetensors merge as the weights are read, GGUF applies them as a parallel branch
  at full precision (never requantized). fp8-scaled Wan checkpoints refuse adapter
  stacks rather than re-round their weights.
- LoRA file must be `.safetensors` format. Z-Image / FLUX accept
  diffusers (PEFT canonical), Kohya/sd-scripts, OneTrainer, and PEFT
  default-adapter naming. Z-Image fused-QKV LoRAs (cv:2904324) splat
  across the split `attention.to_q/to_k/to_v` candle tensors automatically.
- FLUX, Flux.2, Z-Image, Qwen-Image, and LTX-2 CUDA offload use mold-owned block streaming. FLUX / Flux.2 / Z-Image / Qwen-Image keep fitting blocks resident and stream overflow blocks; LTX-2 defaults to adaptive residency and `MOLD_OFFLOAD=1` forces full streaming. SD3 offload still streams every MMDiT block.
- Scheduler V2 resolves normalized request/environment/persisted component placement into an exact artifact/device/load-strategy plan using sampled free VRAM and aggregate host-RAM admission. Explicit CPU/device values are hard, cross-GPU component pins reject, and owners validate the device plus artifact fingerprints before CUDA. Forced-local batches use the same scheduler core across every selected GPU; one-item local generation retains best-free-GPU selection.
- Server discovery is consumed once into the versioned `DeviceRegistry`. Startup owners, dynamic owner construction, UUID-joined telemetry targets, scheduler candidates, `/api/devices`, and legacy status must derive from those canonical records; never add an ordinal-only worker, telemetry, or status inventory beside it, and never rediscover CUDA/NVML from a request path.
- Learned estimates keep setup and execution separate. Engine progress reports typed cold-load, warm-reload, prompt-encode, denoise, VAE, and upscale phases; metadata schema v15 stores an independent runtime EWMA. Planning adds the candidate's cold or warm setup disposition to that runtime, never both a setup-inclusive total and another setup charge.
- Server-owned batch planning reads one static `mold-inference` family registry before engine load, constructs every advertised alias in tests, and validates the instantiated runtime against it. Every production family currently declares only native size `[1]`; initial-wave cardinality is lexically first, homogeneous singleton planning is exact at arbitrary size, and host-coupled heterogeneous singleton planning is exact through 8 devices at arbitrary child count. Singleton results retain one arithmetic lane per selected device with typed random access and bounded 4096-record windows. Ordinary native inputs expose the deterministic 64-strategy bounded heuristic; huge native parents visibly use the mandatory compact singleton fallback as `BoundedHeuristic`. Raw server `batch_size > 1` is routed as ordered singleton children only when Scheduler V2 is authoritative and gallery output is enabled; `/api/capabilities.queue.server_batch` advertises exactly that conjunction and remains false otherwise. No engine is represented as returning multiple native outputs.
- GGUF Q4/Q6 work at 1024x1024; Q8 works at 512x512 (Q8 + LoRA at 1024x1024 is tight on 24GB, see #95)

**Per-model config defaults** (config.toml):

```toml
[models."flux-dev:bf16"]
# ... other fields ...
lora = "/path/to/default-adapter.safetensors"
lora_scale = 0.8
```

### Video Generation

Generate video clips with the LTX Video and Wan families. LTX Video output defaults to APNG (lossless, with metadata); Wan defaults to MP4.

```bash
# Basic video generation (25 frames, APNG output)
mold run ltx-video-0.9.6-distilled:bf16 "a cat walking across a windowsill" --frames 25

# Custom frame count (must be 8n+1: 9, 17, 25, 33, 49, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "ocean waves at sunset" --frames 49

# MP4 output (QuickTime compatible)
mold run ltx-video-0.9.6-distilled:bf16 "a campfire at night" --frames 17 --format mp4

# GIF for pipe-friendly output
mold run ltx-video-0.9.6-distilled:bf16 "a sunset" --format gif | mpv -

# Wan 2.1 text-to-video (frames must be 4n+1: 77, 81, 121, ...; MP4 default)
mold run wan21-t2v-1.3b "a red fox trotting through snow" --frames 81 --fps 16

# Wan 2.1 14B — the dense 2.1 quality tier (bare name resolves :q8)
mold run wan21-t2v-14b "a red fox trotting through snow"

# Wan sequences: past the per-clip envelope this auto-chains and stitches one MP4.
# The seam continues only on an image-conditioned checkpoint (TI2V-5B, A14B I2V);
# a T2V checkpoint concatenates independent clips. Clips are 4k+1.
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" --frames 100 --clip-frames 49

# Wan 2.2 5B at 720p24
mold run wan22-ti2v-5b "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24

# Wan 2.2 A14B, 4-step Lightning tier (two experts, one resident at a time)
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png

# Low-VRAM tiers: Q4_K_M A14B keeps the same Lightning recipe; Q8_0 5B reaches small cards
mold run wan22-t2v-a14b:q4 "a paper boat drifting down a rain gutter"
mold run wan22-ti2v-5b:q8 "waves on a black sand beach" --width 1280 --height 704

# Wan single-frame text-to-image: --frames 1 renders a PNG still (png default, jpeg allowed)
mold run wan22-t2v-a14b:q5 "a lighthouse at dusk, volumetric fog" --frames 1 -o still.png

# Wan recipe controls: flow shift, sample solver, per-expert distill strength
mold run wan22-t2v-a14b:q8 "storm waves" --sample-shift 12          # upstream 720p quality shift
mold run wan22-t2v-a14b:q5 "storm waves" --sample-solver euler      # the Lightning-tuned solver
mold run wan22-t2v-a14b:q5 "storm waves" --distill-strength high=1.8,low=1.0 --steps 6  # motion-restore recipe

# Wan first/last-frame interpolation (A14B I2V or TI2V-5B; endpoints only, no mid-clip keyframes)
mold run wan22-i2v-a14b:q5 "the sapling grows into an oak" --image sapling.png --last-image oak.png

# Wan fp8-scaled A14B quality tier (20-step recipe; same speed as :q8, ~2.6 GB lower peak VRAM)
mold run wan22-t2v-a14b:fp8 "storm waves crash over the lighthouse"

# WebP animated output
mold run ltx-video-0.9.6-distilled:bf16 "a waterfall" --frames 9 --format webp -o waterfall.webp
```

**Constraints:** LTX frame counts must be 8n+1 (9, 17, 25, 33, 49, ...) with dimensions in multiples of 32; Wan frame counts must be 4n+1 (1, 49, 53, 81, ...) with dimensions in multiples of 16 (32 for `wan22-ti2v-5b`). Wan `--frames 1` is a still: png/jpeg admitted (png default); every other frame count is video-only (mp4/gif/apng/webp). The A14B `:q8` quality tier defaults to upstream's per-expert guidance (T2V 4.0 high-noise / 3.0 low-noise, I2V 3.5/3.5); an explicit `--guidance` pins one scale. Current LTX defaults are 1216x704, 25 frames, 30 fps. A14B defaults to 81 frames (`:q5`/`:q4`, reached through block offload), 73 (`:q8`), and 45 (`:fp8`, which cannot park) — each the measured 24 GB envelope, and each at the edge of what admission accepts on an idle card, so a card sharing VRAM should pass a smaller `--frames`. A14B at 720p tops out at 33 frames (~39.5 GB estimated at 81); the family's 720p path is `wan22-ti2v-5b`, measured at 81 frames / 1280x704 in 470.7 s at 19,402 MiB. Distilled models use fewer steps. Wan DiT safetensors load in two key layouts — the upstream/Comfy names (bare or under the `model.diffusion_model.` repack prefix) and diffusers' `WanTransformer3DModel` spelling (`blocks.0.attn1.to_q`, `condition_embedder.*`, `proj_out`) — detected from the file's own header rather than its filename and renamed at load; Wan 2.1 I2V is refused in **either** spelling (original `cross_attn.k_img`, diffusers `attn2.add_k_proj` / `condition_embedder.image_embedder.*`) because its CLIP-vision branch is unimplemented, a diffusers-layout file carrying anything else the rename table does not cover (VACE `vace_blocks.*`, or the named key) is refused rather than partially loaded, and fp8-scaled checkpoints read in the original layout only.

**Current status:** `ltx-video-0.9.6-distilled:bf16` is still the safest default, but the `0.9.8` models now run the full multiscale refinement path. mold pulls the required spatial upscaler asset explicitly, keeps the shared T5 assets under `shared/flux/...`, and intentionally continues using the compatible `LTX-Video-0.9.5` VAE source until the newer VAE layout is ported. Legacy LTX-Video 13B BF16 still has no streaming transformer; CUDA runs preflight full-resident VRAM and fail before allocation when it cannot fit.

**Output formats:** `apng` (default, lossless, metadata), `gif` (256 colors), `mp4` (H.264, requires `mp4` feature), `webp` (requires `webp` feature), `wav` (16-bit PCM, LTX-2 `--pipeline t2a` only).

**`--output` extensions are authoritative for video.** A `.gif` / `.png` / `.apng` / `.webp` / `.mp4` name outranks the family's container default, and an extension this build cannot encode (`.mp4` without the `mp4` feature, `.webp` without `webp`) is refused before any weight is read instead of being written with another container's bytes. A raster or audio extension on a video render is refused the same way, an explicit `--format` that disagrees with the filename is reported rather than silently applied, and `--output -` (stdout) claims no extension so it keeps the resolved container.

### Joint Audio-Video Generation (LTX-2 / LTX-2.3)

Generate synchronized MP4 clips with the LTX-2 family. This family defaults to
MP4 output and exposes audio/video-specific controls.

```bash
# Fast default joint audio-video generation
mold run ltx-2-19b-distilled:fp8 "rain on a neon taxi window" --frames 97 --format mp4

# Audio-to-video
mold run ltx-2-19b-distilled:fp8 "paper sculpture reacting to music" --audio-file cello.wav

# Keyframe interpolation
mold run ltx-2-19b-distilled:fp8 "a canyon flyover" \
  --pipeline keyframe --frames 97 \
  --keyframe 0:start.png --keyframe 96:end.png

# Camera-control preset
mold run ltx-2-19b-distilled:fp8 "lantern-lit cave entrance" --camera-control dolly-in

# Lip dub — re-voice a clip of someone speaking. Frames and fps come from the
# reference video (rounded down to 8k+1); width/height must be multiples of 64.
mold run ltx-2.3-22b-distilled:fp8 "she says: the harbour freezes every winter" \
  --ic-lora-control lipdub --video speaker.mp4 --width 704 --height 448

# HDR regrade with an EXR sidecar — scene-referred linear frames beside the
# tonemapped video. Requires the (gated) HDR adapter and an SDR reference.
mold run ltx-2.3-22b-distilled:fp8 "" \
  --ic-lora-control hdr --video reference.mp4 \
  --frames 121 --local --hdr-exr-dir ./shot_exr

# Advanced guidance overrides (two-stage / two-stage-hq / keyframe / a2-vid / t2a)
mold run ltx-2-19b-distilled:fp8 "handheld shot through a night market" \
  --pipeline two-stage --stg-scale 0.6 --stg-blocks 20,29 --rescale-scale 0.9

# Text-to-audio: no video at all, WAV output
mold run ltx-2.3-22b-dev:fp8 "heavy rain on a tin roof, distant thunder" \
  --pipeline t2a --frames 121 --fps 24 --output rain.wav
```

**Text-to-audio (`--pipeline t2a`)** renders audio only. Duration comes from
`--frames` / `--fps` (121 @ 24 = 5.04 s), the output is a 16-bit PCM stereo WAV
at the vocoder's rate (24 kHz, or 48 kHz with the bandwidth-extension stage),
and `--format` defaults to `wav` — which in turn requires `t2a`. It needs a
checkpoint that ships the audio VAE and vocoder, rejects every conditioning
input and upscaler rather than ignoring them, and rejects `--modality-scale`
other than `1.0` (there is no video branch to guide against). Auto-chaining
never applies: a large `--frames` is a longer take, not more clips. `--batch N`
renders N takes and writes each one as it lands under its own index. Steps default to
the non-distilled schedule — 40 on 19B, 30 on 22B — and a smaller `--steps` is
raised to that default with a message, because the family's 8-step video
default renders hiss here. Only the `audio_*` half of the checkpoint is loaded,
so it fits a 24 GB card without block streaming. Audio prints appear in the
gallery with a rendered waveform tile and an **Audio** kind filter.

**Models:** `ltx-2-19b-dev:fp8`, `ltx-2-19b-distilled:fp8`, `ltx-2.3-22b-dev:fp8`, `ltx-2.3-22b-distilled:fp8`

**Lip dub** (`--pipeline lip-dub`, or just `--ic-lora-control lipdub`, which
selects it) re-voices an existing clip on the gated LTX-2.3 22B DubIt adapter.
The reference video is the authority for length and frame rate — `--frames` /
`--fps` are replaced and mold says so — and it must carry an audio track,
because the reference speech is what the dub imitates. Both axes must be
multiples of 64: the pipeline always renders in two stages, and unlike generic
IC-LoRA it keeps the adapter and the reference on both of them, freezes the
audio through stage 2, and exports the audio stage 1 generated.

**HDR EXR export** (`--hdr-exr-dir <dir>`, `--hdr-exr-full-float`) writes one
scene-referred linear `frame_%05d.exr` per frame from the HDR adapter's LogC3
signal (`--ic-lora-control hdr` + `--video <SDR reference>`; the adapter's
scene embeddings replace the prompt). Frame counts past the per-clip cap
auto-chain with **global** frame numbering across the stitched timeline — a
121-frame render is exactly `frame_00000..frame_00120`, each stage regrading
its own window of the reference, with the last stage sized to the exact
remainder. Chained EXR export requires `--local` (a remote server cannot write
the sidecar to your machine), a reference covering the full duration at the
render fps, and Smooth/Cut transitions only.

**Important flags:** `--audio`, `--no-audio`, `--audio-file`, `--video`, repeatable `--keyframe`, repeatable `--lora`, `--pipeline`, `--retake`, `--camera-control`, `--spatial-upscale`, `--temporal-upscale`, `--clip-frames`, `--motion-tail`, `--stg-scale`, `--stg-blocks`, `--rescale-scale`, `--modality-scale`, `--guidance-skip-step`, `--hdr-exr-dir`, `--hdr-exr-full-float`

The five guidance flags (wire: an additive optional `guidance_overrides`
object) each replace one per-(pipeline, stage) guider constant. Omitting a flag
keeps its constant, so an unflagged request reproduces earlier outputs exactly.
They are read only by pipelines that run the multimodal guider — `two-stage`,
`two-stage-hq`, `keyframe`, `a2-vid`, `t2a` — never enable a guider a pipeline
deliberately disables (`a2-vid` audio), and are ignored by chained/sequence
renders, which say so instead of pretending the flag landed. Non-LTX-2
families and out-of-range values are rejected with HTTP 422.
Web, desktop, iPhone, and TUI expose the same optional fields in their LTX-2
Advanced video controls. Native graphical shells restore them from templates
and Library metadata, validate before queueing, and refuse automatic long-video
routing when the chain wire would discard them. TUI values live for the current
Create session, validate STG block lists before the editor closes, and keep the
wire object absent until a field is touched.

The TUI's Advanced → Video accordion exposes the shared synchronized-audio,
latent-upscale, and guidance-override contracts for LTX-2. Audio cycles `default` / `on` / `off`,
where `default` omits `enable_audio`, `on` selects MP4, and a current checkpoint
advertising `supports_audio:false` hides the row. Spatial cycles `native` /
`1.5×` / `2×`; Temporal cycles `native` / `2×`; native omits the matching
request field. STG scale/blocks, CFG rescale, modality scale, and guidance skip
also default to absent; scales and skip use bounded keyboard cycles, while Enter
on STG blocks opens a validated comma-separated editor. Pipeline and
conditioning-file controls remain the separate broader TUI gap.

`VideoData.pipeline` is the runtime authority for completed LTX-2 videos,
including an implicit Auto selection. Server, CLI, and TUI save paths preserve
it in `OutputMetadata.pipeline`; web, desktop, iPhone, and both TUI Library
detail presentations show it only when present.

The TUI consumes the same family-agnostic `SseProgressEvent::Preview` frames as
web, desktop, and iPhone. It bounds and decodes each base64 PNG into a transient
Create preview, replaces the fixed-protocol render cache on every frame, and
keeps one readable denoise-status row. Kitty, Sixel, and iTerm2 rendering must
stay centered through `ui::gallery::center_rect`; malformed frames retain the
last valid image, while a new run, completion, or error clears transient state.
The plain CLI intentionally remains a text progress surface.

Community LTX-2 checkpoints can be video-only even when their transformer and
video VAE are complete. Mold inspects the installed safetensors for both the
audio VAE and vocoder; web, desktop, and iPhone disable generated audio when
either is absent, while text/image-to-video continues normally. `--audio` and
direct API requests then fail before prompt encoding or denoising; use
`--no-audio` for those checkpoints.

**Chained (arbitrary-length) video output:** for LTX-2 19B and 22B distilled models, `--frames` above the 97-frame default clip size automatically renders multiple clips with a motion-tail of latents carried across each clip boundary, then stitches them into a single MP4. The CLI picks this path transparently — `mold run ltx-2-19b-distilled:fp8 "a cat walking" --frames 400` produces one 400-frame MP4 from 5 chained stages. 97 is a _routing_ default chosen to fit one consumer GPU, not the model ceiling: `--clip-frames N` (must be `8k+1`) raises it all the way to the model's real single-request budget, so `--frames 241 --clip-frames 241` renders one coherent 10s clip instead of a stitched sequence. `--motion-tail N` sets the overlap (default 4 pixel frames, 0 disables carryover). Legacy `mold run` returns only the final output, while durable job workflows use `mold jobs` / `/api/chain-jobs` for resume and retake. Non-chainable families reject `--frames` past their own single-request ceiling with an actionable error.

**Continuing an existing video (`--extend`):** `mold run <model> "prompt" --extend clip.mp4` continues a clip in one request and returns the original plus the new footage. `--frames` is the length of the _rendered continuation_, of which `--extend-overlap N` re-renders the source tail as motion context and is dropped from the delivered result — so the run appends `frames - overlap` new frames. Both the grid and the default are per family: LTX-2 re-encodes a latent motion tail, so its overlap is `8k+1` and defaults to 17, while wan's continuation is seeded with the source's final frame, so its overlap is `4k+1` and defaults to — and is the only value its engine accepts — **1**. It must stay `< --frames`. Extend is LTX-2 and Wan only (wan per checkpoint: a text-to-video checkpoint has no channel to accept the seed frame and is refused at admission), cannot be combined with `--image`, `--video`, or `--keyframe`, and requires the request's width/height/fps to match the source clip. Over HTTP it is `GenerateRequest.extend_video` (inline base64) or `extend_video_path` (server-local, resolved against the media allow roots), plus `extend_overlap_frames`. `/api/models` advertises additive `supports_extend` and `extend_default_overlap_frames` per model; absence means the host predates continuation, and clients must hide the control rather than send a request that will be rejected. A continuation that names no overlap has the family default written into the request at admission (and on the forced-local CLI path), so the saved metadata records the overlap that actually rendered — an installed `cv:` / `hf:` checkpoint has no manifest to resolve a family through later. `mold run <text-to-video model> --extend clip.mp4` reports the extend gate before the source-image contract, so the error names `--extend` rather than an image the request never supplied. Web, desktop, and iPhone derive the whole control from `studio/lib/extend.ts` — the advertised per-model field decides, never a family set, and both the overlap grid (`sequenceFrameStep`) and the submitted value (`resolveExtendOverlapFrames`) follow the family. A client continuation always sends `extend_overlap_frames` explicitly, so what the picker shows is what the request carries; leaving it absent inherits the host's family-blind default, which wan refuses. A continuation carries its own source frames — they come from the tail of the clip being continued — so it satisfies an image-to-video checkpoint's source requirement with no image attached, on the server (`request_carries_source_frames`) and on every client alike; a text-to-video checkpoint is refused for the continuation itself.

**The prompt is optional for conditioned LTX-2 / LTX-Video runs.** `mold run ltx-2-19b-distilled:fp8 --image still.png --frames 97` is a complete command. An empty (or whitespace-only) prompt is accepted **only** when the resolved family is `ltx2` or `ltx-video` **and** the request carries visual conditioning — `source_image`, non-empty `keyframes[]`, `source_video`/`source_video_path`, or `extend_video`/`extend_video_path`. Pure text-to-video and every image family stay required, including img2img. The rule lives in `mold_core::prompt_required_for` / `prompt_required_with_conditioning` (server, CLI, TUI, Discord) and its browser mirror `studio/lib/promptRequirement.ts` (web, desktop, iPhone) — never re-derive it. Why it works: LTX-2's Gemma tokenizer pads to a fixed 1,024-token context and the connector replaces padded positions with learned register embeddings, so `""` is a trained context. Two things to never get wrong in user-facing text: it saves **zero** VRAM (the prompt context is a fixed `[1, 1024, 4096]` tensor), and it renders near-static micro-motion. An empty prompt suppresses prompt expansion (the server clears `expand` in `maybe_expand_prompt`, and CLI/TUI do the same) so the expander can never invent the recorded prompt, and it is not written to prompt history. Discord's `/generate` `prompt` is now `Option<String>` — **this requires a slash-command re-registration**. MCP is deliberately unchanged: its generate tools take no conditioning input, so both the schema `required` array and the runtime check stay strict.

**LTX-2's frame ceiling is a duration, not a frame count.** The checkpoints ship `pos_embed_max_pos = 20` and the temporal RoPE axis is normalized in _seconds_ (the pixel-frame coordinate is divided by fps before `max_pos` normalization), so the real budget is 20 s of runtime: `frames <= 20 * fps + 4`, capped by a 604-frame resource guard. That is 484 frames at 24 fps and 244 at 12 fps, but only 124 at 6 fps — the ceiling moves in both directions, so never hard-code it. `--temporal-upscale x2` does not extend it: `derive_stage1_render_shape` halves the stage-1 frame count _and_ the stage-1 fps, so stage 1 renders the same runtime at half the frame rate.

**LTX-2 memory is planned, not sampled.** Admission (`mold-server/src/ltx2_admission.rs`) reads the checkpoint's safetensors header and reconstructs the engine's residency plan — per-block sizes, the non-block transformer weights streaming never offloads (~2.1 GB on the 19B FP8 preset), the bundled video VAE (~2.4 GB), a token-based activation budget for that exact shape, runtime headroom, and a fragmentation margin — under the same 90% device cap. Each chain stage is priced at _its own_ shape (stage 1 of a two-stage distilled render is half-resolution). The engine then honours the scheduler's frozen `predicted_vram_peak_bytes`: workers bind it with `device::init_thread_vram_grant_bytes` and `plan_adaptive_residency` budgets `min(grant, usable free VRAM)` instead of expanding to fill the card, counting `fixed_resident_bytes` it previously priced at zero. Activation cost comes from `device::ltx2_activation_budget_bytes` (tokens = latent frames × (h/32) × (w/32)), never a pixel-area heuristic. Infeasible shapes are rejected **before** the multi-minute load with a message naming resolution/frame combinations that fit; the same advice rides on a CUDA OOM. OOM cooldowns are keyed `(model, shape, GPU)` so a single-GPU host stops re-admitting the identical failing shape, one conservative reduced-grant retry is offered per shape, and the denoise stage retries a recoverable OOM at a shrinking budget — a fatal CUDA fault still quarantines the worker and stops the server.

**Current constraints:** `x2` spatial upscaling is wired across the family, `x1.5` spatial upscaling is wired for `ltx-2.3-*`, and `x2` temporal upscaling is wired in the native runtime. Camera-control preset aliases currently auto-resolve the published LTX-2 19B LoRAs only. The family runs through the native Rust stack in `mold-inference`, with CUDA and Apple Metal as supported backends for real local generation (Metal is slower — streamed FP8 widening trades speed for fitting the model in unified memory) and CPU as a correctness-only fallback. On 24 GB Ada GPUs such as the RTX 4090, the validated path stays on the compatible `fp8-cast` mode rather than Hopper-only `fp8-scaled-mm`. The native CUDA matrix is validated across 19B/22B text+audio-video, image-to-video, audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale (`x1.5` / `x2` where published), and temporal upscale (`x2`). Explicit LTX-2 unload drops the retained runtime, safely synchronizes pending work, and re-samples actual free VRAM without resetting the process-owned primary context; CPU fallback unload remains a plain state clear. When requests go through `mold serve`, the built-in body limit is `64 MiB`, which is enough for common inline source-video and source-audio workflows.

## Multi-prompt Chain (v2)

Direct any-length video scene-by-scene with a TOML script or sugar flags.

```bash
# Canonical TOML script (schema: mold.chain.v1)
mold run --script shot.toml
mold run --script shot.toml --dry-run    # Print stage summary, don't submit

# Validate only
mold chain validate shot.toml

# Sugar: repeated --prompt (uniform smooth chains only)
mold run ltx-2-19b-distilled:fp8 \
  --prompt "a cat walks into the autumn forest" \
  --prompt "the forest opens to a clearing" \
  --frames-per-clip 97
```

### Transitions

- `smooth` _(default)_: motion-tail carryover, visual morph between scenes
- `cut`: fresh latent, no carryover; optional `source_image` for i2v seed
- `fade`: cut + post-stitch alpha blend of `fade_frames` (default 8)

### API

- Chain endpoint: `POST /api/generate/chain[/stream]`
- Validation endpoint: `POST /api/generate/chain/validate` accepts the same body and returns a normalized, no-queue plan (per-stage input/output frames, transitions, source/negative presence, warnings, and optional `vram_estimate`). It never creates a job, starts a download, or touches inference. Web, desktop, and iPhone expose it as **Validate plan** on the exact selected host and discard stale responses after live draft changes.
- Capabilities: `GET /api/capabilities/chain-limits?model=<name>&fps=<n>` — also carries `frames_per_clip_recommended` (the model's own default), the echoed `fps`, `frames_per_clip_runtime_seconds` for duration-budgeted families, `supports_audio`, and model-specific `supports_sequence` + `sequence_unsupported_reason`
- Per-model frame semantics ride on each `GET /api/models` row (flattened, video models only): `default_frames`, `default_fps`, `max_frames` (ceiling at `default_fps`), `max_runtime_seconds` + `max_frames_absolute` when the ceiling is a duration, `frame_step` (valid counts are `k·step+1`). Absent on image models — never substitute a constant
- Max stages: 16. `frames_per_clip_cap` is the model's own clip size — the clip one generation renders when auto-chaining (97 for LTX-2, a flat 97 for LTX-Video, and for Wan the checkpoint's own manifest default frame count over a 53-frame A14B / 121-frame floor — 81 for A14B Q5, 121 for TI2V-5B) — and every Studio sequence picker locks to it. Chain admission rejects a stage above the family's single-request ceiling at the chain's fps (481 on-grid for LTX-2 at 24 fps), which is the most an explicit `--clip-frames` can reach.

### mold jobs CLI

Durable chain jobs can be inspected and controlled through `mold jobs` against
a running server:

```bash
mold jobs list [--json]
mold jobs show <id> [--json]
mold jobs resume <id>
mold jobs retake <id> --stage <N> [--mode cascade|splice] [--seed-offset <U64>] [--prompt <TEXT>]
mold jobs cancel <id>
mold jobs delete <id> [--yes]
mold jobs gc
```

The commands use `MOLD_HOST` and `MOLD_API_KEY` like other remote
CLI surfaces. `mold jobs gc` mirrors `POST /api/chain-jobs/gc`, pruning
successful ephemeral shim jobs and completed non-ephemeral artifacts older than
`chain.jobs_artifact_ttl_days`.

### mold trash CLI

Deleting a print moves it to the serving host's trash
(`<output_dir>/.trash/`); the sweeper purges it after
`gallery.trash_retention_days` (default 30, `0` = keep forever). `mold trash`
inspects and acts on that trash over HTTP against `MOLD_HOST` (with
`MOLD_API_KEY`); there is deliberately no local fallback.

```bash
mold trash list [--json]          # FILENAME · TITLE · TRASHED (3h ago) · PURGES (in 27d | kept | due) · SIZE
mold trash restore <FILENAME>...  # POST /api/gallery/trash/restore — 409 GALLERY_RESTORE_CONFLICT if a live print took the name
mold trash empty [--yes]          # DELETE /api/gallery/trash — confirms [y/N] unless --yes
mold trash sweep                  # POST /api/gallery/trash/sweep — prints purged= and remaining=
```

Titles: `mold run ... --title "Smurf village"` (≤120 chars, validated at
parse time) rides `GenerateRequest.title` → `OutputMetadata.title` → the
gallery row and folds into the default filename as
`mold-{model}-{ts}[-{idx}]~{slug}.{ext}`; an explicit `-o` path is used
verbatim and the file is never renamed later. `--title` applies to
single-clip runs only (chain scripts / multi-`--prompt` sequences refuse it),
though the HTTP chain body does carry `title` for the stitched print.

**File under (creation-time organization).** A print can arrive already
organized: `--tag <TAG>` (repeatable, ≤20 tags of 1–64 chars, matched
case-insensitively) and `--collection <NAME>` ride `GenerateRequest.tags` /
`.collection` → `OutputMetadata` → the gallery row. Collections resolve by
slug and are created when absent, which is how one name means one collection
across a fleet. Seeding is **once, at row insert**: organization is
user-owned afterwards, so a reconcile or a re-publication never resurrects a
tag the user removed.

`generate.auto_tag_title` (DB key, default `true`) makes a titled CLI/TUI run
also tag the print with its title slug, disclosed before it is applied — on
stderr for the CLI (`filing under tag "smurf-village"`), on the Tags row and
section summary for the TUI. `--no-auto-tag` turns it off for one CLI
invocation; the TUI's toggle is Settings ▸ Library ▸ **Tag by title**. It is
deliberately a **client** default — the server never auto-tags, because it
cannot tell a typed title from a scripted one.

In the TUI, filing lives in Create ▸ Advanced ▸ **File under** (Title, Tags,
Collection). Each is absent until touched and validated before its editor can
close, so an untouched form's request is unchanged and nothing admission would
refuse reaches the wire.

Nothing about filing can fail a render. On `MOLD_DB_DISABLE=1`, or when a
`{id}` collection was deleted between listing and Generate, the filing is
dropped and reported on the `x-mold-request-warning` header — never silently,
never as a refusal. `MoldClient` reads that header on all four response paths
into additive `GenerateResponse.request_warnings` / `ChainResponse
.request_warnings`; `mold run` prints each one through `status!` (stderr when
piped), and the TUI shows them on the Create view — the `!` advisory row, never
the error slot, since the host accepted and rendered the request — and records
each one in the Timeline. Starting the next generation clears the row. Never
split the header on `; ` — the advisory prose contains that sequence, so a
split renders one advisory as two half-sentences.

## Reference implementations

When debugging or changing a model family, compare against the documented upstream — and prefer one you can **run** over one you can only read.

| Family  | Primary reference                                                                                                             |
| ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Z-Image | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — `src/model/diffusion/z_image.hpp`, `docs/z_image.md` |
| LTX-2   | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) — `packages/ltx-core`, `packages/ltx-pipelines`                       |
| Wan     | diffusers/Lightning flow-UniPC schedule (deliberately _not_ upstream `fm_solvers_unipc.py`)                                   |

sd.cpp is Z-Image's primary reference because it is a runnable oracle: it publishes the `leejet/Z-Image-Turbo-GGUF` checkpoints mold downloads, supports Metal, and renders those exact files correctly — so it answers "is this mold's bug?" in one command instead of an argument. Clone references into gitignored `tmp/`.

```bash
# Build sd.cpp with Metal (Accelerate off avoids Apple's vDSP.h/-Welaborated-enum-base
# failures on recent SDKs; the deployment target avoids MTL4CommandQueue availability errors)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 -DCMAKE_C_FLAGS="-Wno-error" -DCMAKE_OBJC_FLAGS="-Wno-error"
cmake --build build --config Release -j 8

# Render the same checkpoint mold uses
./build/bin/sd-cli --diffusion-model z_image_turbo-Q4_K.gguf --vae <flux-ae.safetensors> \
  --llm <qwen3-4b.gguf> -p "a single red apple on a white table" --cfg-scale 1.0 -H 768 -W 768 --steps 8
```

## Model Selection Guide

Pick the right model for the task:

| Model                               | Speed             | Quality   | Best For                                                |
| ----------------------------------- | ----------------- | --------- | ------------------------------------------------------- |
| `flux-schnell:q8`                   | Fast (4 steps)    | Good      | Quick iterations, drafts                                |
| `flux-dev:q4`                       | Slow (25 steps)   | Excellent | Final quality, detailed                                 |
| `flux2-klein:q8`                    | Fast (4 steps)    | Good      | Low VRAM, lightweight FLUX                              |
| `flux2-klein-9b:q8`                 | Fast (4 steps)    | Excellent | Higher quality 9B, non-commercial                       |
| `flux2-dev:bf16`                    | Slow (50 steps)   | Excellent | Full FLUX.2 Dev; gated, non-commercial, high host RAM   |
| `sdxl-turbo:fp16`                   | Fast (4 steps)    | Good      | Quick SDXL generation                                   |
| `sd15:fp16`                         | Medium (25 steps) | Good      | ControlNet, 512x512                                     |
| `z-image-turbo:q8`                  | Fast (9 steps)    | Excellent | High quality, Qwen3 encoder                             |
| `qwen-image:q4`                     | Slow (50 steps)   | Good      | Stable base Qwen GGUF on 24 GB cards                    |
| `qwen-image-2512:q4`                | Slow (50 steps)   | Good      | Stable 2512 GGUF on 24 GB cards                         |
| `qwen-image:q8`                     | Slow (50 steps)   | Better    | Best base GGUF quality, validated at 768x768 on 24 GB   |
| `qwen-image-flash:q4`               | Fast (4 steps)    | Good      | Fastest Qwen path; DMD2 distill, weak on hard detail    |
| `qwen-image-distill:q4`             | Medium (15 steps) | Better    | Faster Qwen with more of the base model's fidelity      |
| `qwen-image-edit-lightning:fp8`     | Fastest (4 steps) | Good      | Official lightx2v fused Lightning edit distill          |
| `ltx-video-0.9.6-distilled:bf16`    | Fast (8 steps)    | Good      | Text-to-video, 30fps                                    |
| `ltx-video-0.9.8-2b-distilled:bf16` | Fast (7+3 steps)  | Better    | Newer checkpoint family with full multiscale refinement |
| `ltx-2-19b-distilled:fp8`           | Slow (8 steps)    | Better    | Joint audio-video, recommended LTX-2 default            |
| `ltx-2.3-22b-distilled:fp8`         | Slow (8 steps)    | Best      | Larger joint audio-video path                           |

Default model if none specified: `flux2-klein:q8`

### Model Defaults

| Model                          | Steps | Guidance | Resolution                                     |
| ------------------------------ | ----- | -------- | ---------------------------------------------- |
| `flux-schnell`                 | 4     | 0.0      | 1024x1024                                      |
| `flux-dev`                     | 25    | 3.5      | 1024x1024                                      |
| `sdxl-base`                    | 25    | 7.5      | 1024x1024                                      |
| `sdxl-turbo`                   | 4     | 0.0      | 512x512                                        |
| `sd15`                         | 25    | 7.5      | 512x512                                        |
| `sd3.5-large`                  | 28    | 4.0      | 1024x1024                                      |
| `z-image-turbo`                | 9     | 0.0      | 1024x1024                                      |
| `flux2-klein`                  | 4     | 0.0      | 1024x1024                                      |
| `flux2-klein-9b`               | 4     | 1.0      | 1024x1024                                      |
| `qwen-image`                   | 50    | 4.0      | 1328x1328                                      |
| `qwen-image-2512`              | 50    | 4.0      | 1328x1328                                      |
| `qwen-image-flash`             | 4     | 1.0      | 1328x1328                                      |
| `qwen-image-distill`           | 15    | 1.0      | 1328x1328                                      |
| `qwen-image-edit-lightning`    | 4     | 1.0      | 1024x1024                                      |
| `ltx-video-0.9.6-distilled`    | 8     | 1.0      | 1216x704 (25 frames, 30fps)                    |
| `ltx-video-0.9.8-2b-distilled` | 7+3   | 1.0      | 1216x704 (25 frames, 30fps, multiscale refine) |
| `ltx-2-19b-distilled`          | 8     | 3.0      | 1216x704 (97 frames, 24fps, mp4 default)       |
| `ltx-2.3-22b-distilled`        | 8     | 3.0      | 1216x704 (97 frames, 24fps, mp4 default)       |

### Available Models

**FLUX.1**: `flux-schnell:q8`, `flux-schnell:q6`, `flux-schnell:q4`, `flux-schnell:bf16`, `flux-dev:q8`, `flux-dev:q6`, `flux-dev:q4`, `flux-dev:bf16`, `flux-krea:q8`, `flux-krea:q6`, `flux-krea:q4`, `flux-krea:fp8`

**FLUX.1 Fine-tunes**: `jibmix-flux:q4`, `jibmix-flux:q5`, `jibmix-flux:fp8`, `ultrareal-v4:q8`, `ultrareal-v4:q5`, `ultrareal-v4:q4`, `ultrareal-v3:q8`, `ultrareal-v3:q6`, `ultrareal-v3:q4`, `ultrareal-v2:bf16`, `iniverse-mix:fp8`

**SDXL**: `sdxl-base:fp16`, `sdxl-turbo:fp16`, `juggernaut-xl:fp16`, `realvis-xl:fp16`, `playground-v2.5:fp16`, `dreamshaper-xl:fp16`, `pony-v6:fp16`, `cyberrealistic-pony:fp16`

**SD 1.5**: `sd15:fp16`, `dreamshaper-v8:fp16`, `realistic-vision-v5:fp16`

**SD 3.5**: `sd3.5-large:q8`, `sd3.5-large:q4`, `sd3.5-large-turbo:q8`, `sd3.5-medium:q8`

**Z-Image**: `z-image-turbo:bf16`, `z-image-turbo:q8`, `z-image-turbo:q6`, `z-image-turbo:q4`

**Flux.2 Klein**: `flux2-klein:bf16`, `flux2-klein:q8`, `flux2-klein:q6`, `flux2-klein:q4`

**Flux.2 Klein-9B**: `flux2-klein-9b:bf16`, `flux2-klein-9b:q8`, `flux2-klein-9b:q6`, `flux2-klein-9b:q4`

**Flux.2 Dev**: `flux2-dev:bf16` (gated, non-commercial; 50 steps, guidance 4.0; automatically block-offloads on constrained CUDA GPUs)

**Wuerstchen**: `wuerstchen-v2:fp16`

**Qwen-Image**: `qwen-image:q8`, `qwen-image:q6`, `qwen-image:q5`, `qwen-image:q4`, `qwen-image:q3`, `qwen-image:q2`, `qwen-image:fp8`, `qwen-image:bf16`

**Qwen-Image-2512**: `qwen-image-2512:q8`, `qwen-image-2512:q6`, `qwen-image-2512:q5`, `qwen-image-2512:q4`, `qwen-image-2512:q3`, `qwen-image-2512:q2`, `qwen-image-lightning:fp8`, `qwen-image-lightning:fp8-8step`, `qwen-image-2512:bf16`

**Qwen-Image few-step distills** (all CFG-free at guidance 1.0): `qwen-image-flash:q8`, `qwen-image-flash:q4` (NVIDIA DMD2, 4 steps), `qwen-image-distill:q8`, `qwen-image-distill:q4` (DiffSynth Distill-Full, 15 steps), `qwen-image-edit-lightning:fp8` (official lightx2v 4-step fused Lightning edit distill)

Flash runs its own packaged scheduler — `use_dynamic_shifting=false`, `shift=3.0`, `shift_terminal=null` — not the base model's resolution-dependent schedule. Every other Qwen checkpoint (including the Distill-Full and Lightning merges, which are transformer-only exports) keeps the base contract.

**LTX Video**: `ltx-video-0.9.6:bf16`, `ltx-video-0.9.6-distilled:bf16`, `ltx-video-0.9.8-2b-distilled:bf16`, `ltx-video-0.9.8-13b-dev:bf16`, `ltx-video-0.9.8-13b-distilled:bf16` (the 13B BF16 tiers need a 40 GB-class GPU — no offload path; LTX-2 supersedes them on 24 GB cards)

**LTX-2 / LTX-2.3**: `ltx-2-19b-dev:fp8`, `ltx-2-19b-distilled:fp8`, `ltx-2.3-22b-dev:fp8`, `ltx-2.3-22b-distilled:fp8`
**Qwen-Image text encoder controls**:

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`
- On Apple Metal/MPS, `auto` prefers quantized Qwen2.5-VL GGUF text encoders (`q6`, then `q4`) to reduce memory pressure
- On CUDA, `auto` prefers BF16 when there is enough text-encoder headroom and falls back to quantized GGUF variants for local sequential, resident, and edit paths when BF16 would be too heavy
- Hot CUDA Qwen-Image may keep Qwen2.5 on GPU after a prompt-cache miss only when measured free VRAM still covers denoise and VAE decode reserves; cache hits and pressure cases drop/park before denoise
- `qwen-image-edit-2511:*` uses repeatable `--image` inputs and a distinct `qwen-image-edit` family. Local inference is implemented with the Qwen2.5-VL vision tower, packed edit latents, and true-CFG norm rescaling. Quantized `--qwen2-variant` values are supported for the edit family through a GGUF language path plus staged vision sidecar. CUDA quantized edit transformers always use split CFG; do not re-enable batched packed-edit CFG based only on free VRAM.
- Context-killing CUDA errors (illegal address, uncorrectable ECC, launch failure/assert, and related faults) permanently quarantine that GPU worker and stop the server with an error so service supervision can restart the process; the embedded desktop relaunches the whole app. Apply quarantine to normal jobs, durable chains, admin loads, post-upscalers, and already-buffered worker jobs. Reject and settle every accepted-but-unstarted item before owner teardown, and retain a poisoned worker's cache untouched for process exit. Do not route fatal faults through the ordinary timed degraded cooldown or reset Candle/cudarc's primary context in-process. CUDA builds enable NVML automatically; `/api/status` and `/api/resources` join NVML/nvidia-smi VRAM to CUDA's frozen runtime-visible inventory by UUID, never physical ordinal. `CUDA_VISIBLE_DEVICES` is a hard exposure boundary. A MIG worker accepts only its exact MIG UUID; leave parent/profile metadata null if the adapter cannot prove it rather than overlaying the physical GPU.
- `MOLD_DISPATCH_MODE` is restart-only and defaults to `v2`: V2 owns binary worker leases, `legacy` restores the depth-two one-release rollback transport, and `observe` keeps legacy authoritative while computing V2 decisions read-only. Rollback transport depth is accounting only; its owner must acquire the same fair binary execution claim as V2 and durable chains after dequeue and before any GPU action. A waiting chain may be bypassed by at most three younger owner starts globally. Queue pause gates generations and utility/admin GPU work in every mode. `/api/capabilities.dispatch.v2_authoritative` and `observes_v2_decisions` report the runtime actually started, so CPU fallback and maintenance mode never claim V2 authority merely because `v2` was configured.
- Internal F1 durable batch recovery is parent-scoped and fail closed. Before reading or healing a parent, transaction, or committed archive, hold gallery bookkeeping long enough to try-claim the stable hashed parent authority and every discovered attempt generation in `bookkeeping → parent → attempt` order; release bookkeeping before any wait (claims are nonblocking), retain parent longer than the transaction, and never unlink the stable parent pathname. Discovery before the claims is names-only. Parent journals permit only `v1* → v2*`; receipt extraction must use the same validated replay as state. Oversized v1 active/out-of-order state drains in v1 without new grants until representable, then transitions once. The joint bridge must preview the exact successful reducer completion before staging; stale/closed/fenced/validation-error results never gain a receipt. An uncertain parent-persistence failure after valid staging must retain the receipt for joint recovery rather than tombstoning evidence the parent journal may have accepted. Any uncertain transaction-delta append poisons the live object; only recovery may continue.
- GPU startup selection accepts `all`, `none`, legacy visible ordinals, Mold stable IDs (`cuda:<uuid>`, `metal:default`), and NVIDIA `GPU-...` / `MIG-...` UUIDs. Prefer `/api/devices` IDs for persisted configuration. `none` is fail-closed maintenance mode: do not construct an ordinal-0 engine, spawn an inference queue, or accept generation/admin model-load work. A GPU-feature build with no safely selected worker is also unavailable; only a true CPU-only build may use the CPU correctness fallback.
- Runtime device lifecycle uses `mold gpu list|disable|enable` or authenticated `PATCH /api/devices/{stable-id}`. Disable removes future eligibility immediately, lets an active lease finish, drops device-backed caches on the CUDA owner thread, and joins it; re-enable allocates a monotonic owner epoch and returns `202 Starting` before probing the fresh owner context. Every worker event carries that owner epoch, and exact `(device, epoch)` reaping prevents delayed predecessor events from removing a replacement. A failed probe leaves the device desired but unavailable with an actionable reason so enable can be retried. Never reset the primary context. Desired enablement is machine-wide and persists for absent devices. Live mutation is scheduler-V2-only. Legacy/observe/maintenance expose only restart recovery: a persistently-disabled, startup-selected GPU may be enabled for the next boot, stays `restart_required` on device polls, and cannot be disabled live. Startup exclusions still require a broader selector plus restart, and all-disabled is a valid maintenance state with administrative/read APIs alive.
  **ControlNet (SD1.5)**: `controlnet-canny-sd15:fp16`, `controlnet-depth-sd15:fp16`, `controlnet-openpose-sd15:fp16`

**Utility (LLM)**: `qwen3-expand:q8`, `qwen3-expand-small:q8`

### Name Resolution

Bare names auto-resolve: `flux2-klein` -> `flux2-klein:q8`, `flux-dev` -> `flux-dev:q8`, `sdxl-base` -> `sdxl-base:fp16`, `sd15` -> `sd15:fp16`

FP8 safetensors models are automatically quantized to Q8 GGUF on first use (one-time conversion, cached at `$MOLD_HOME/cache/`).

## img2img (Image-to-Image)

Transform an existing image with a text prompt:

```bash
# Basic img2img
mold run "oil painting style" --image photo.png --strength 0.7

# Low strength = subtle changes (close to original)
mold run "enhance details" --image photo.png --strength 0.3

# High strength = major transformation
mold run "anime style" --image photo.png --strength 0.9

# From stdin
cat photo.png | mold run "watercolor" --image - --strength 0.6
```

**Strength guide**: `0.0` = no change, `0.3` = subtle, `0.5` = balanced, `0.75` = strong (default), `1.0` = full txt2img

### Inpainting

Repaint specific regions using a mask:

```bash
mold run "a golden retriever" --image park.png --mask mask.png
# mask: white = repaint, black = preserve
```

### Face-identity conditioning (PuLID-FLUX)

Keep one person's face across arbitrary prompts. Qualified for `flux-dev:q4`
and `flux-dev:q8`, on CUDA and Metal, in every official release build.

```bash
# One-time setup: the bundle is licence-gated and will not download without this
mold pull pulid-flux --accept-license insightface-antelopev2

mold run flux-dev:q4 "an astronaut in a diner" --id-image face.jpg
mold run flux-dev:q4 "a Renaissance portrait" --id-image face.jpg --id-weight 0.6
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg --id-start-step 4

# Several references of the same person, averaged (up to 4)
mold run flux-dev:q4 "a chef in a kitchen" \
  --id-image front.jpg --id-image side.jpg --id-image smiling.jpg

# A real negative branch (upstream advises --guidance 1.0 with it)
mold run flux-dev:q4 "a hiker on a ridge" --id-image face.jpg \
  --true-cfg 2.0 --guidance 1.0 --negative-prompt "blurry, cartoon"
```

| Flag                  | Default | Range                                    | Notes                                                                                                                 |
| --------------------- | ------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--id-image <path>`   | —       | PNG/JPEG, ≤16 MiB, ≤8192 px/axis, ≤32 MP | Reference photograph. **Repeatable, up to 4** — several references of one person are averaged into one identity; whole-set byte/pixel caps sit below the per-image limits times the count. |
| `--id-weight <f>`     | `1.0`   | `0.0`–`3.0`                              | `0` is completely inert: nothing pulled, loaded, or extracted, and the render is byte-identical to no identity at all |
| `--id-start-step <n>` | `0`     | `< --steps`                              | First denoise step identity applies from                                                                              |
| `--true-cfg <f>`      | `1.0`   | `1.0`–`10.0`                              | True classifier-free guidance scale. `1.0` is off. Requires `--id-image` and a non-zero `--id-weight`; drop `--guidance` to `1.0` alongside it. ~2x denoise time. |
| `--cfg-start-step <n>` | `1`    | `< --steps`, requires `--true-cfg`       | First denoise step the true-CFG negative branch runs at                                                              |

Refused, by name rather than silently: any other model, a LoRA alongside an
identity, img2img alongside an identity, a build without the `pulid` feature,
both `--id-image` repeated beyond 4, and `--true-cfg` without an active
identity. `--negative-prompt` does nothing on FLUX unless `--true-cfg` is
above `1.0` — FLUX.1-dev is guidance-distilled and has no negative branch
without it. Works over `$MOLD_HOST` and with `--local`; the bundle and the
licence acceptance must be on the machine that RENDERS, which for a remote run
is the server. Repeated `--id-image` and `--true-cfg` additionally need a
server that advertises `capabilities.identity`: against an older one `mold run`
refuses by name rather than submitting, because that server would drop the
fields and render without them, while a single `--id-image` works against any
identity-capable server. Saved metadata records the photograph's file name(s),
SHA-256(s), and the applied weight and start step — plural fields only for the
plural form, never the photograph. `mold rm pulid-flux` removes the bundle and
the derived vision tower and parser. Full guide: `website/guide/identity.md`.

Wire contract — additive `GenerateRequest` fields (read `supports_identity`, never the feature, to decide whether to offer the control):

| Field            | Purpose                                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id_image`       | Face-identity reference as base64 PNG/JPEG bytes. Bounds-checked from its header alone (≤ 16 MiB encoded, ≤ 8192 px per axis, ≤ 32 MP) before anything decodes it.                         |
| `id_image_name`  | Client-supplied provenance label for `id_image`; recorded into `OutputMetadata.id_image_name` so Reuse settings can restore the reference. The engine never reads it. Requires `id_image`. |
| `id_weight`      | Identity strength in `0.0..=3.0`. Absent means `1.0`. Requires `id_image`.                                                                                                                 |
| `id_start_step`  | First denoise step at which identity is applied, so composition settles before the face is pinned. Must be `< steps`. Absent means `0`. Requires `id_image`.                               |
| `id_images`      | Plural shape of `id_image`: several references of one person, up to `ID_IMAGES_MAX` (4), averaged post-IDFormer into one identity. Mutually exclusive with `id_image` — sending both is an error, never a precedence rule. Whole-set encoded-byte and decoded-pixel caps sit below the per-image limits times the count. A photograph with no detectable face refuses the whole request and names its one-based position. |
| `id_image_names` | Plural form of `id_image_name`, one per entry in `id_images`, in the same order.                                                                                                          |
| `true_cfg`       | True classifier-free guidance scale, `1.0..=10.0`. Absent or `1.0` is off. Qualified only alongside an active identity (an image and non-zero `id_weight`); refused on a plain FLUX render rather than accepted and ignored. Reuses `negative_prompt`.                                    |
| `cfg_start_step` | First denoise step the true-CFG negative branch runs at. Must be `< steps`. Absent means `1`. Requires `true_cfg`.                                                                        |

Milestone-1 gate — every rule below is a 422 at admission, never a silent drop:

- Only the identity-qualified checkpoints `flux-dev:q4` and `flux-dev:q8` are
  accepted (bare `flux-dev` resolves to `:q8`, and the legacy dash form
  `flux-dev-q4` resolves too). `flux-dev:bf16` and every other model are refused.
- Identity may not be combined with a LoRA or with an img2img `source_image` —
  neither combination is qualified yet.
- Any of `id_weight` / `id_start_step` / `id_image_name` without `id_image`
  (or `id_image_names` without `id_images`) is an error, not an ignored field.
  Sending both `id_image` and `id_images` is likewise an error.
- `true_cfg` / `cfg_start_step` without an active identity (a photograph and a
  non-zero `id_weight`) is an error — true CFG never runs on an ordinary FLUX
  render.
- A server that cannot execute identity conditioning refuses any request
  carrying an identity field. It never accepts-and-ignores, because that would
  render a print with no face in it and say nothing. Two distinct messages, so
  a client sends the operator to the right fix: a build **without** the `pulid`
  feature answers `mold_core::identity::IDENTITY_BUILD_UNSUPPORTED` ("this
  server was built without PuLID face-identity support" — needs a differently
  compiled binary), and a build that links `pulid` while the runtime adapter is
  still pending answers `IDENTITY_RUNTIME_PENDING` ("identity conditioning is
  not available in this build yet" — needs a newer one). A request with no
  identity fields is untouched on every build.

`mold_core::identity` is the single authority for all of this.
`identity_runtime_available()` — the `pulid` feature AND `IDENTITY_RUNTIME_READY`
— is the one predicate; never re-spell it as a bare feature check.
`/api/models[]` advertises additive `supports_identity` per model, true only
when that predicate holds AND the checkpoint is qualified, so the capability is
advertised only once the runtime adapter is present; it is absent on servers
that predate identity conditioning, which clients read as "no". The same fact
rides `generation_profile.capabilities.supports_identity`; never derive a second
predicate. `id_images` and `true_cfg` are gated separately: `GET
/api/capabilities.identity` (`{ multi_photo, max_photos, true_cfg }`) is the
authority for those two additive shapes specifically, and absence — the whole
block is all-false on an older server — reads as no, so a client that needs
either shape probes first and refuses by name rather than sending fields an
older server would silently drop. The singular `id_image` predates that block
and needs no probe. Saved metadata records `id_image_name(s)`,
`id_image_sha256(s)`, `id_weight`, `id_start_step`, and — for a true-CFG print —
the scale and start step actually used, only when the print carried the
relevant field — hashes and names, never the face payload.

Assets: identity conditioning needs the hidden auxiliary bundle `pulid-flux` —
five files, about 2.2 GB (PuLID-FLUX v0.9.1 adapter, the EVA02-CLIP-L-14-336
vision tower, the InsightFace antelopev2 face detector + recognizer, and
facexlib's BiSeNet face parser (`parsing_bisenet.pth`, MIT, no acceptance
needed) that masks the aligned crop before the vision tower sees it), every
file SHA-256 pinned, stored under `shared/pulid/`. An install made before mold
0.24 has four files; re-running the pull below fetches only the missing
parser. The two antelopev2 files are licensed for non-commercial research
only, so every pull path — `mold pull`, the server's auto-pull, client-triggered
downloads — refuses them until acceptance is recorded once per `MOLD_HOME`:

```bash
mold pull pulid-flux --accept-license insightface-antelopev2
```

The flag prints the restriction and the pinned terms URL, then records the
acceptance on **whichever machine runs the pull** — see
[Third-party model licenses](#third-party-model-licenses). A refusal names the
license, its URL, and that exact command. `mold rm pulid-flux` removes the
bundle; `mold pull pulid-flux` repairs a partial one.

You do not have to pull it by hand. A request that actually conditions on a
face plans the bundle through the same dependency preparation the encoder
ladders use: `POST /api/generate/placement-preview` reports whatever is missing
under `pending_downloads` (kinds `identity_adapter`, `identity_vision_encoder`,
`face_detector`, `face_recognizer`, `face_parser`) without fetching anything,
and admission materializes it into `shared/pulid/` — after the license gate, so
an unaccepted antelopev2 fails the job with that same `--accept-license`
message instead of downloading. Every file is verified against its manifest
SHA-256 pin before it can be used — Hugging Face `main` is a mutable branch —
so a changed or tampered artifact is named, deleted, and refused rather than
frozen into a plan. The bytes are always hashed (through a retained no-follow
descriptor, once per process per unchanged file); the `.sha256-verified`
sidecar is still written for installed-state reporting but is never read as
proof, because a group-writable model root lets whoever writes the weights
write the sidecar too. An `id_weight` of `0` applies no identity at all and is
completely inert: it plans no assets, downloads nothing, and adds no memory
demand; the same holds for `true_cfg` absent or `1.0`.

Studio surfaces: web and desktop Create render an **Identity** photo well
beside the source-image wells with **Identity strength** and **Identity start
step** in Advanced, gated on the advertised `supports_identity` — an
unqualified checkpoint or an older host shows nothing at all and a staged
photo parks (retained, off the wire, Generate unaffected) until a qualified
model is selected again. `id_weight` and
`id_start_step` stay off the wire until the user touches them, the photo is
never fitted or cropped to the canvas, and an unqualified combination is
reported inline with Generate blocked. The Library shows the recorded name,
digest, strength, and start step, and Reuse settings restores them plus the
photo itself when the device still holds it. iPhone, TUI, and Discord expose
the same conditioning, as described below. Multiple photographs (`id_images`)
and true CFG are core, server, and CLI only so far (#1226) — every Studio
surface, web and desktop included, still offers a single photograph and no
true-CFG control.

**Surfaces.** The TUI's Create form carries an **Identity photo** section in
the Advanced accordion — a local path row plus Strength and Start step — shown
only while the selected checkpoint's `/api/models[]` entry advertises
`supports_identity`. The path is opened no-follow and bounds-checked at entry,
so a rejected file never leaves the picker; a model switch to an unqualified
checkpoint keeps the photo, shows `mold_core::identity`'s own refusal on the
row, and blocks Generate (`↺ Reset to model defaults` clears it). Library
Details and the full print view render the saved provenance.

The Discord bot exposes identity as its own `/identity` command — `identity`
(PNG/JPEG attachment), `identity_strength`, `identity_start_step`, plus
prompt/model/size/steps/guidance/seed — rather than as options on `/generate`,
which already sits at Discord's hard 25-option ceiling. The bot refuses an
oversized or wrong-container attachment before downloading it, checks both
knobs and the model gate against the server's advertised `supports_identity`,
and names the reference in the result embed. Both surfaces derive every label,
range, limit, and refusal from `mold_core::identity`; neither restates one, and
neither reads a local build feature to decide whether the renderer can execute
identity conditioning.

The iPhone app ships the same four fields: an Identity well in the primary Create stack (gated on positive `supportsIdentity` knowledge, parked on a capability-losing model switch), strength and start step in the Advanced sheet, provenance in the Library Info sheet, and Use-as-prompt reattachment from the shared stash.

### ControlNet (SD1.5 only)

```bash
mold run "a person" --control edges.png --control-model controlnet-canny-sd15:fp16
mold run "interior" --control depth.png --control-model controlnet-depth-sd15:fp16 --control-scale 0.8
```

## Piping

mold is pipe-friendly. When stdout is not a TTY, image bytes go to stdout and status to stderr:

```bash
mold run "a cat" | viu -                           # Preview in terminal
mold run "a cat" | convert - output.webp           # Convert format
echo "a dog in space" | mold run flux2-klein        # Prompt from stdin
cat photo.png | mold run "style" --image - | viu - # Full pipeline
```

Force stdout in interactive mode: `mold run "a cat" --output -`

## Upscaling

Upscale images to 2x or 4x resolution using Real-ESRGAN super-resolution models.
Standalone and post-generation upscales auto-download the selected model on the
server that runs the job, including remote multi-GPU hosts.
Post-generation upscale retains distinct `-original` and `-upscaled` gallery
artifacts; reuse restores the pre-upscale generation canvas from metadata.
An upscale-only failure falls back to the successful original as one artifact.

```bash
# Upscale with default model (real-esrgan-x4plus:fp16, auto-downloads ~32MB)
mold upscale photo.png

# Choose a specific model
mold upscale photo.png -m real-esrgan-x4plus-anime:fp16

# Custom output path
mold upscale photo.png -o photo_4x.png

# Display upscaled image inline
mold upscale photo.png --preview

# Pipe: generate then upscale
mold run "a cat" | mold upscale -

# Force local (skip server)
mold upscale photo.png --local

# Smaller tile size for limited VRAM
mold upscale large_photo.png --tile-size 256
```

### Available Upscaler Models

| Model                           | Scale | Size   | Best For                        |
| ------------------------------- | ----- | ------ | ------------------------------- |
| `real-esrgan-x4plus:fp16`       | 4x    | 32 MB  | General photos (default)        |
| `real-esrgan-x4plus:fp32`       | 4x    | 64 MB  | General photos (full precision) |
| `real-esrgan-x2plus:fp16`       | 2x    | 32 MB  | Subtle 2x enhancement           |
| `real-esrgan-x4plus-anime:fp16` | 4x    | 8.5 MB | Anime/illustration              |
| `real-esrgan-anime-v3:fp32`     | 4x    | 2.4 MB | Fast anime/video                |

## Model Management

```bash
mold list                    # List downloaded + available models
mold pull flux-dev:q4        # Download a model
mold pull flux-dev:q4 --skip-verify  # Download, skip SHA-256 check
mold default                 # Show current default model and how it was resolved
mold default flux-dev:q4     # Set default model (validates name, warns if not downloaded)
mold info                    # Installation overview (paths, models, server status)
mold info flux-dev:q4        # Show model details and file sizes
mold rm flux-dev:q4          # Remove a downloaded model
mold rm flux-dev:q4 --force  # Remove without confirmation
mold licenses                # Third-party model licenses and acceptance state
mold licenses --local        # ...this machine's own, without asking the server
```

### Third-party model licenses

Some auxiliary weights carry terms Mold's own license does not cover (today:
the InsightFace antelopev2 face models PuLID needs, non-commercial research
only). Mold refuses to download them until acceptance is on record.

```bash
mold licenses                # what needs accepting, and on which machine
mold licenses --local        # this machine's own acceptances, never the server's
mold pull pulid-flux --accept-license insightface-antelopev2
```

**Acceptance is per `MOLD_HOME`, on the machine that does the downloading.**
`mold pull` sends the id to `MOLD_HOST` when a server answers, so the SERVER
records it in its own root; only a forced-local or fallback pull records it on
this machine. `mold licenses` reports which root it read for the same reason —
recording locally and pulling remotely was a real bug, not a hypothetical.

The record is an owner-only (`0600`) `$MOLD_HOME/license-acceptances.json`
bound to the `(url, sha256)` pair of the exact license text shown. The URL is a
commit-pinned, immutable link, so a Mold release that re-pins a license to a
newer upstream revision invalidates existing acceptances and asks again.
Accepting is offline — Mold never fetches the license text, so it works
air-gapped. There is no environment-variable bypass.

Consent is bound to the terms that were displayed. When a server will record
the acceptance, `mold pull` reads `GET /api/licenses` from THAT server, shows
its terms, and sends back exactly those — a bare id would let a server on a
different release resolve its own revision and record agreement to text the
user never read.

Over HTTP: `GET /api/licenses` lists ids, terms links, `sha256`, `accepted`,
and `required_by`. `POST /api/downloads` and `POST /api/models/pull` take an
additive `accept_licenses: [{ id, url, sha256 }]`. A gated download without one
is `403` / `LICENSE_NOT_ACCEPTED`; terms the server does not pin are `409` /
`LICENSE_TERMS_MISMATCH` carrying the server's own `url`/`sha256`/`canonical`
so a client can re-display and retry. Both refusals write nothing. Servers
advertise `capabilities.licenses: true`.

## Model discovery catalog

**Browse:** web UI `/models` route, **Discover** segment — cards and detail
drawer. Retired browser routes such as `/catalog` render Page Not Found. Every read is
a live HF + Civitai proxy through `GET /api/catalog/search` with a 5-min
in-process cache keyed by `sort=downloads|recent|rating` (no SQLite catalog
table, no scanner, no scrape); unknown sort values return 422.
Sequence-mode browse links enter this view with Video + Models filters. Web,
desktop, and iPhone Create pickers contain only sequence-capable installed
models and select one when available. A new Sequence starts with two required clip
descriptions, uses Smooth / Cut / Fade seam labels for context-capable
LTX-2 and Join clips for LTX-Video's zero-tail fallback, and keeps frame
choices strictly above the active motion tail. Seed, source, audio, and TOML tools stay under
progressive disclosure. Durable sequence creation, events, previews, and
actions follow the selected machine with its API key in headers rather than
falling back to another engine. iPhone persists only the host identity and
durable job ID; the snapshotted route's API key stays in Keychain, and a saved
instance identity must match exactly before recovery reattaches.

**Pull catalog ids:** `mold pull hf:author/repo` and `mold pull cv:618692`
hit the upstream APIs directly for the recipe. HF separated-bundling entries
and supported SD1.5, SDXL, FLUX, Z-Image, LTX-Video, LTX-2, LTX-2.3, and Wan
single-file Civitai checkpoints download with companions and are runnable
via `mold run cv:<id>`. Wan Civitai installs pull `wan-umt5` plus the
VAE matching the sub-family (`wan21-vae` for everything except TI2V-5B's
`wan22-vae`). Wan 2.2 A14B fine-tunes are a two-expert pair that Civitai
publishes as separate High/Low model versions: the catalog pairs the sibling
versions into one two-file install (high-noise primary + `low-noise-transformer`
recipe role), shows one row per pair, and resolves either expert's `cv:` id to
the same install; a version whose counterpart cannot be identified with
confidence is refused with the reason — never installed as a single expert.
Half-downloaded pairs report not-installed and a re-install resumes the
missing half. Z-Image fine-tunes pull `z-image-te`
(Tongyi-MAI Qwen3 shards + tokenizer + fallback VAE; satisfied by an existing
`z-image-turbo` install) and use recipe-provided text-encoder files
when the Civitai version publishes them. Flux.2 fine-tunes pull `flux2-vae` (168 MB
Klein VAE, ungated) and either `flux2-te` (Qwen3-4B, ungated, for
`sub_family=klein-4b`) or `flux2-te-9b` (Qwen3-8B, **HF_TOKEN required**,
for `klein-9b` / `flux2-d`). LTX-Video entries pull
`ltx-video-vae` as a companion (Civitai fine-tunes are transformer-only);
LTX-2 and LTX-2.3 entries pull their version-matched video VAE when the
checkpoint is transformer-only; LTX-2.3 entries also pull the standalone Gemma
hidden-state projection used by diffusion-only/quantized exports. Combined
checkpoints keep using their bundled assets. ConvRot W4A4 exports full-stream
automatically because the compatibility backend reconstructs BF16 block weights.
Native multi-prompt chains accept every LTX-2 checkpoint; a dev checkpoint renders its clips through the two-stage pipeline, which costs roughly twice the wall time per clip because stage 1 runs CFG as two sequential forwards.
Installed catalog checkpoints with opaque `cv:` / `hf:` IDs and no bundled
spatial upscaler use the one-stage path and remain sequence-capable.
Two-stage LTX-2 dev checkpoints are rejected before a durable sequence job is
created. Aggregate Hugging Face repositories are marked unsupported for Pull;
cards and details surface download failures as toasts.
Single-file format detection is key-based (reads safetensors header only).

**Auth:** `HF_TOKEN` for gated HF repos; `CIVITAI_TOKEN` for early-access
/ NSFW Civitai. Web Settings persists these to `mold.db` `settings`
(`huggingface.token`, `civitai.token`).

**Internals:** `mold-catalog::live` is the proxy + cache — HF and Civitai
are fetched concurrently and cached per source/page, and Civitai paginates
via its real cursor chain (deep pages past row 100 work); `live::fetch_civitai_version`
and `live::fetch_hf_repo` resolve single ids to a `CatalogEntry` with a
fully-rendered `DownloadRecipe`. Entries carry an additive `page_url`
(HF repo page / Civitai `models/{modelId}?modelVersionId={vid}` page;
`None` when uncomposable — e.g. a version-detail body without `modelId`).
Per-install **`mold-catalog.json`
sidecars** sit next to each downloaded primary file and back the LoRA
picker's "what's installed" list — sidecars travel with the model
file, so a copy to another mold install retains trigger words.

## Configuration Management

View and edit settings from the CLI using dot-notation keys. Settings are split between `config.toml` (paths, ports, credentials) and `mold.db` (user preferences — `expand.*`, `scheduler.*`, generation defaults, per-model generation overrides). `mold config` routes by key prefix transparently.

```bash
mold config list                          # Show all settings grouped by section
mold config list --json                   # Machine-readable output
mold config get server_port               # Get a single value
mold config get server_port --raw         # Raw value for scripting
mold config set server_port 8080          # Bootstrap key → written to config.toml
mold config set expand.enabled true       # User-preference → written to mold.db
mold config set default_width 1024        # Generation default → written to mold.db
mold config set scheduler.replan_debounce_ms 2000  # Scheduler timing → mold.db
mold config set gallery.trash_retention_days 7     # Library trash retention → mold.db (0 = keep forever, max 3650)
mold config set output_dir none           # Clear an optional field
mold config set models.flux-dev:q4.default_steps 30   # Per-model generation default → model_prefs (DB)
mold config where expand.enabled          # Print "db" or "file" so operators know the surface
mold config reset expand.enabled          # Drop the DB row; next read falls back to config.toml/env/default
mold config reset --all --yes             # Drop every DB row under the active profile
mold config --profile portrait set default_steps 30   # Scope a command to an explicit profile (v6)
mold config path                          # Show config file location
mold config edit                          # Open config.toml in $EDITOR
```

Keys use dot-notation matching the TOML / DB layout. Boolean values accept `true`/`false`, `on`/`off`, or `1`/`0`. Use `none` to clear optional fields. Values are validated (port range, enum options, numeric bounds) before saving. Environment variable overrides are shown when active. `mold config list` output tags each row with its surface (`[db]` / `[file]` / `[env]`), and `mold config set` tags the surface it wrote to (e.g. `Set expand.enabled = true [db]`).

Scheduler timing preferences are profile-scoped and loaded when the server's V2 coordinator starts: `scheduler.replan_debounce_ms` defaults to 2000, `scheduler.replan_max_delay_ms` to 5000, and `scheduler.warm_wait_max_ms` to 2000. Each accepts 0–30000, and max delay must be at least the debounce. Restart the server after changing them.

`gallery.trash_retention_days` (section Gallery, `gallery.` ⇒ DB surface, env `MOLD_GALLERY_TRASH_RETENTION_DAYS`) is the per-host Library trash retention: days a trashed print waits in `<output_dir>/.trash/` before the hourly/startup sweep purges it; default 30, `0` keeps trashed prints forever, max 3650. It is read fresh on every sweep (no restart) and advertised as `capabilities.gallery.trash.retention_days`; desktop/web/iPhone edit a remote host's value through that host's `/api/config`.

`generate.auto_tag_title` (profile-scoped DB key, default `true`) is the client-side **File under** preference: a titled `mold run` or TUI print also picks up its own title slug as a tag, disclosed before it is applied. `mold run --no-auto-tag` overrides it for one invocation; see **File under** above.

On first launch after upgrading from a pre-#265 release, mold imports the `[expand]`/generation-defaults slices of `config.toml` into the DB (gated by `config.migrated_from_toml`), renames the original `config.toml` to `config.toml.migrated` as a one-release downgrade safety net, and rewrites `config.toml` as a stripped **bootstrap-only** file (paths, ports, credentials, per-model file paths — nothing the DB now owns). Multi-profile scoping landed in schema v6: set `MOLD_PROFILE=dev` or pass `--profile dev` to any `mold config` subcommand. Device enablement preferences are machine-wide in `device_preferences`, not profile-scoped; a missing row means enabled by default and discovery never writes one. SQLite corruption detected at open or during a gallery listing quarantines `mold.db` plus its WAL/SHM sidecars as `mold.db.corrupt-<timestamp>*`, rebuilds the schema, and reconciles gallery rows from disk; preferences and prompt history reset unless manually salvaged from that retained copy.

## Self-Update

```bash
mold update                       # Update to latest GitHub release
mold update --nightly             # Install latest rolling build from main
mold update --check               # Check for updates without installing
mold update --version v0.6.0      # Install a specific version
mold update --force               # Reinstall even if already up-to-date
```

Downloads the correct platform-specific binary from GitHub releases, verifies SHA-256 checksum, and replaces the running binary in-place. Linux inspects every device allowed by `CUDA_VISIBLE_DEVICES`, including MIG UUIDs mapped to their physical parent, independently of order: homogeneous 8.6 selects sm86, homogeneous 8.9 selects sm89, mixed 8.6/8.9 selects the release-gated sm86 embedded-PTX floor artifact, homogeneous 10.x selects sm100, and homogeneous 12.x selects sm120. Compute capability 8.0, 9.x, and unproven mixed groups fail closed. `MOLD_CUDA_ARCH` must equal the target selected for every visible device. Missing artifacts are never replaced with a higher compute target; only an old unsuffixed sm89 filename can substitute for sm89. Detects Nix/Homebrew installations and suggests using the package manager instead. Respects `GITHUB_TOKEN` for API rate limits. B200/sm100 is simulated, not hardware-qualified.

## RunPod Cloud GPUs

Manage RunPod pods end-to-end from `mold`. All subcommands use the REST API at
`https://rest.runpod.io/v1/` plus GraphQL for account info and GPU/datacenter
discovery (those aren't exposed via REST).

### One-time setup

```bash
# Get an API key at https://www.runpod.io/console/user/settings (Read/Write scope)
mold config set runpod.api_key <key>             # persist to config.toml
# or
export RUNPOD_API_KEY=<key>                      # env var (overrides config)

mold runpod doctor                               # verify auth + balance
```

### Killer feature — `mold runpod run`

Creates a pod if needed, waits for the mold server inside to boot, generates
via SSE (so it survives RunPod's 100s Cloudflare proxy timeout), saves the
output, and leaves the pod warm for reuse.

```bash
mold runpod run "a cat on a skateboard"          # smart defaults
mold runpod run "a sunset" --model flux-dev:q4   # preload a model
mold runpod run "a cat" --gpu 5090               # force GPU family
mold runpod run "a cat" --dc US-IL-1             # pin datacenter
mold runpod run "a cat" --network-volume nv-abc123 # persistent /workspace
mold runpod run "a cat" --keep                   # don't park — leave running
mold runpod run "a cat" --steps 28 --seed 42     # forward standard flags
mold runpod run "a cat" --output-dir ./renders   # custom save path
```

Outputs save to `./mold-outputs/runpod-<pod-id>-<ts>.png` (directory
auto-created, `.gitignore`'d by default).

### Full subcommand reference

```bash
# Discovery
mold runpod gpus                                 # table view, aggregate stock
mold runpod gpus --json                          # machine-readable
mold runpod gpus --all                           # include uncommon GPU families
mold runpod datacenters --gpu "RTX 5090"         # per-DC availability

# Lifecycle
mold runpod create --gpu 5090                    # smart defaults fill the rest
mold runpod create --dry-run                     # print plan, don't create
mold runpod create --cloud community             # secure is default
mold runpod create --hf-token                    # wire HF_TOKEN secret into pod env
mold runpod create --network-volume nv-abc123    # attach pre-created volume
mold runpod network-volume list                  # list persistent volumes
mold runpod network-volume get nv-abc123         # inspect one volume
mold runpod network-volume create --name models --size 100 --dc US-KS-2
mold runpod network-volume update nv-abc123 --size 200 # grow only
mold runpod network-volume update nv-abc123 --name shared-models
mold runpod network-volume delete nv-abc123      # permanently deletes data
mold runpod list
mold runpod list --json
mold runpod get <pod-id>
mold runpod stop <pod-id>                        # pause billing, keep storage
mold runpod start <pod-id>                       # resume
mold runpod delete <pod-id>                      # immediate, non-interactive teardown

# Connecting
mold runpod connect <pod-id>                     # print export MOLD_HOST=…
eval "$(mold runpod connect <pod-id>)"           # exec the export in your shell
mold runpod connect <pod-id> --check             # also probe the pod first

# Observability
mold runpod logs <pod-id>                        # validate pod + print console logs handoff
mold runpod usage                                # balance + active pods
mold runpod usage --since 7d                     # with historical spend window
mold runpod usage --json                         # machine-readable
```

Production network volumes accept 10–3999 GB. Current Pod list/get responses
identify attachments with `networkVolumeId` and assigned GPUs with
`machine.gpuTypeId`; callers must handle those shapes even when expanded
`networkVolume` / `gpu.displayName` fields are absent. Volume-backed Pods use
Secure Cloud in the volume's datacenter, request a 0 GB ordinary workspace
disk, cannot be stopped, and must be deleted before the volume can be removed.

### Config keys under `[runpod]`

| Key                         | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| `api_key`                   | API key (env `RUNPOD_API_KEY` wins). Redacted in `config list`. |
| `default_gpu`               | Pin a GPU family (e.g. `RTX 5090`). Overrides smart-pick.       |
| `default_datacenter`        | Pin a datacenter (e.g. `EUR-IS-2`). Overrides smart-pick.       |
| `default_network_volume_id` | Attach a network volume to every new pod.                       |
| `auto_teardown`             | If true, delete pods after `run` instead of parking.            |
| `auto_teardown_idle_mins`   | Idle reap window (default 20). `0` disables.                    |
| `cost_alert_usd`            | Abort a session that exceeds this many USD. `0` disables.       |
| `endpoint`                  | Override REST base URL (mostly for testing).                    |

All settable via `mold config set runpod.<key> <value>`. Clear with `none`.

### Smart defaults

When `--gpu`/`--dc` aren't pinned:

1. Aggregate stock across datacenters per GPU family.
2. Pick cheapest family with **High** or **Medium** stock from: 4090 > 5090 > L40S > A100.
3. The displayed image target comes from the shared GPU table: A30/A100 and generic Ampere → `:<version>-sm80`; A2/A10/A16/A40, RTX A4000–A6000, and RTX 3050–3090 → `:<version>-sm86`; Ada → `:<version>`; H100/H200 → `:<version>-sm90`; B200/B300 → `:<version>-sm100`; Grace Hopper and Grace Blackwell are unsupported; named RTX PRO/GeForce 50-series → `:<version>-sm120`. Ambiguous generic Blackwell falls back to sm89 instead of guessing. Stable official clients fetch the exact release/source manifest and submit `@sha256`; missing or inconsistent manifests fail closed. Main/source/Nix clients use mutable `latest*`. B200 support is simulated, not hardware-qualified.
4. No datacenter pin — let RunPod's scheduler pick any machine.
5. If scheduling stalls (runtime still null + machine unassigned after 90s), delete the stuck pod and try the next stock-ranked DC.

### Common failure modes

- **"pod didn't schedule within 90s"** — RunPod capacity signal (stockStatus) is optimistic. The scheduler couldn't actually place a machine. Retry fallback handles this; if all candidates fail, capacity genuinely isn't there.
- **"value must be one of …" on `/pods`** — you pinned a datacenter that isn't in RunPod's REST enum whitelist. GraphQL exposes more DCs than REST accepts. Omit `--dc` or pick from the REST-accepted list in the error message.
- **Cloudflare 404 during boot** — the mold server inside the pod hasn't started yet. `wait_for_mold` polls `/api/status` for valid JSON with a `version` field to distinguish proxy-404 from real readiness.

### State persistence

`$MOLD_HOME/` (default `~/.mold/`) holds:

- `runpod-state.json` — warm-pod pointer used by `run` for reuse.
- `runpod-history.jsonl` — append-only log used by `usage --since`.

Safe to delete; they're caches, not sources of truth.

## Server Mode

```bash
mold serve                           # Start foreground server on 0.0.0.0:7680
mold serve --port 8080               # Custom port

# Daemon management (background server)
mold server start                    # Start background server daemon
mold server start --port 8080        # Custom port
mold server start --bind 127.0.0.1   # Custom bind address
mold server start --models-dir /path # Custom models directory
mold server start --log-file         # Enable file logging
mold server status                   # Show managed server status (PID, port, uptime, models)
mold server stop                     # Graceful shutdown (HTTP → SIGTERM → SIGKILL)

mold ps                              # Check server status
mold unload                          # Free GPU memory

# LAN auto-discovery (mDNS/DNS-SD, `mdns` feature — on in release builds)
mold server discover                 # Browse the network for advertised mold servers (table)
mold server discover --json          # Machine-readable list
mold server discover --probe         # Add a /health latency column
mold serve --no-mdns                 # Disable advertising + server-assisted browse (also MOLD_MDNS=0)

# Connect from another machine
MOLD_HOST=http://gpu-host:7680 mold run "a cat"

# Custom image output directory (default: ~/.mold/output/)
MOLD_OUTPUT_DIR=/srv/mold/output mold serve
```

### Durable queue and restarts

Queued generations are recorded in `mold.db` and replayed automatically after a
restart, under their original job ids — `systemctl restart mold` during a busy
queue no longer loses work. `GET /api/capabilities` reports
`queue.durable_queue`, and each `GET /api/queue` row reports whether that
particular job is durable (a job with no gallery target, one carrying
reference-upload media, or one whose request exceeds the payload ceiling runs
normally but is not replayed).

On `SIGTERM` the running generation is aborted at its next inference checkpoint
and requeued, new requests get `503 SERVER_RESTARTING` with `Retry-After`, and
the server exits within `MOLD_SHUTDOWN_ABORT_SECS` (45 s) — a hard deadline that
ends the process if the drain overruns, since a cold model load cannot be
interrupted. A job that keeps
failing to finish is **held** — listed as `state: "held"` with a reason, never
started automatically. Clear one with `DELETE /api/queue/{id}`.

Never delete a queued job just because its stream died against a
`durable_queue` host: on that host the job is still going to run.

### HTTP API Endpoints

Core endpoints exposed by `mold serve` (full list + schemas at `/api/docs`):

- `POST /api/generate` — image/video generation, raw bytes response
- `POST /api/generate/stream` — SSE progress + base64 complete event
- `POST /api/generate/placement-preview` — read-only Scheduler V2 projection for `{ request, copies }` (`copies` is `1..=64`). Ordinary generations return a version-1 authoritative candidate only when the request is admissible; the call never leases a device, loads a model, starts a download, or mutates queue state. `planned` may include additive `pending_downloads` for known encoder dependencies selected by that candidate plan, using a preview-only registry identity and low-confidence estimate until admission lands and re-fingerprints the file. Cold installed `cv:`/`hf:` IDs resolve only from contained local sidecars, and their synthesized runtime config remains attached to prepared work through planning and pre-CUDA validation across config refreshes. `infeasible` may include additive `missing_components` with `repair_model`. Clients queue nothing on infeasible and only offer repair/resume when they own the exact host's complete grouped pull. Prepared Batch N clients send one sibling-shaped request (`batch_size: 1`) with `copies: N`.
- `POST /api/generate/chain` — chained arbitrary-length video (LTX-2 distilled); body is `mold_core::chain::ChainRequest` (canonical `stages[]` or auto-expand `prompt`+`total_frames`+`clip_frames`); executes through the durable chain-job runner while keeping the legacy response shape
- `POST /api/generate/chain/stream` — same as above, SSE progress with per-stage `denoise_step` events and additive `job_id` fields on progress frames
- `POST /api/generate/chain/validate` — read-only normalization and family validation for the same chain body; returns stage contribution math, conditioning presence, warnings, and optional VRAM without queue/download/inference side effects
- Durable chain jobs:
  - `POST /api/chain-jobs` · `GET /api/chain-jobs` · `GET /api/chain-jobs/:id`
  - `POST /api/chain-jobs/placement-preview` — accepts preferred `{ request, copies }` and legacy raw-chain bodies, but currently returns a valid version-1 non-authoritative `unsupported` response. Do not use it to claim an exact per-device chain-stage plan yet.
  - `GET /api/chain-jobs/:id/events` — SSE snapshot + live job events
  - `POST /api/chain-jobs/:id/resume` · `POST /api/chain-jobs/:id/retake` · `POST /api/chain-jobs/:id/cancel`
  - `POST /api/chain-jobs/:id/amend` — edit a settled/queued sequence in place: body is `AmendRequest` (the FULL edited `stages[]` plus optional `motion_tail_frames`/`fps`/`seed`/`steps`/`guidance`/`enable_audio` overlays; model, size, output format, placement, strength, and batch provenance are NOT amendable). Returns 202 `AmendResponse` (flattened `ChainJobSummary` + `preserved_stages`) and requeues from the earliest dirty clip; `cut`↔`fade` and fade-length edits re-finalize with zero re-renders
  - `DELETE /api/chain-jobs/:id` · `POST /api/chain-jobs/gc` · `GET /api/chain-jobs/:id/stages/:idx/preview`
- `POST /api/expand` — LLM prompt expansion; optional `style` is absorbed as a natural-language directive and additive `task` selects T2V/I2V/V2V/retake/keyframe/audio-driven/audio policy without carrying source media. `GET /api/capabilities.expand` reports `{ configured, model_present, backend, remix }` plus the additive `model` naming the manifest expansion model (local backends only), so clients offer the pull without hard-coding `qwen3-expand`. Desktop/web expansion follows the generation route unless that host's `model_present` is `false`, then re-ranks the eligible machines that have it via `studio/lib/expansionRouting.ts` (all ready machines under Auto/Most capable, only the pinned one when pinned); the print itself never follows
- `GET /api/models` · `GET /api/loras` · `POST /api/models/load` · `POST /api/models/pull` · `DELETE /api/models/unload`
- `GET /api/discovery/peers` — cached `_mold._tcp` peers visible from the server's LAN; call only when `/api/capabilities.discovery.can_browse` is true, then connect to the returned URL directly
- `DELETE /api/models/:model` — remove a downloaded model (HTTP `mold rm`): deletes only exclusively-owned files, keeps shared components, returns `{ removed, kept, freed_bytes }`; 409 while loaded
- `GET /api/gallery[?view=library|trash]` · `POST /api/gallery/media-token` · `GET /api/gallery/image/:name` · `GET /api/gallery/thumbnail/:name` · `DELETE /api/gallery/image/:name[?permanent=true]` (trash by default; `permanent` hard-deletes) · `PATCH /api/gallery/image/:name` (`{title?, favorite?, tags?, add_tags?, remove_tags?}` → updated `GalleryImage`)
- Library organization (per host; `capabilities.gallery.organize`): `POST /api/gallery/organize` (bulk `{filenames, favorite?, add_tags?, remove_tags?, add_to_collections?, remove_from_collections?}`) · `GET/POST /api/gallery/collections` · `GET/PATCH/DELETE /api/gallery/collections/:id` · `PUT /api/gallery/collections/:id/items` (`{add, remove}`) · `GET /api/gallery/tags` · `PATCH /api/gallery/tags/:name` (`{name}`) · `DELETE /api/gallery/tags/:name`
- Trash (`capabilities.gallery.trash = { enabled, retention_days }`): `POST /api/gallery/trash` / `POST /api/gallery/trash/restore` (`{filenames}`; restore is `409 GALLERY_RESTORE_CONFLICT` when a live print holds the name) · `DELETE /api/gallery/trash` → `{purged}` · `POST /api/gallery/trash/sweep` → `{purged, remaining}`; rows from `?view=trash` carry `trashed_at` / `purge_at`. SSE adds `gallery_updated`, `gallery_trashed`, `gallery_restored`, `gallery_collections_changed`; purge reuses `gallery_removed`
- `GET/POST /api/downloads` · `DELETE /api/downloads/:id` · `GET /api/downloads/stream` — bounded-parallel model pulls (two active per host); listings expose `active_jobs` plus legacy first-job `active`; cancel works for queued and active jobs. Desktop keeps one host-keyed stream per selected download target — active pulls pin to the top of the Models view with a source glyph and target host — so progress, completion refresh, and cancellation stay routed to the correct server.
- `POST /api/upscale` · `POST /api/upscale/stream`
- `GET /api/queue` — authoritative server-side listing plus additive scheduler `plan` (per-device lanes, timing estimates, blocked reasons, plan/replan versions). The plan is advisory until the worker revalidates its exact execution fingerprint and frozen artifacts.
- `PATCH /api/queue/:id` — re-lane and/or reorder a queued job (`target_gpu?`, queued-only 0-based `position?`); omitted fields stay unchanged
- `DELETE /api/queue/:id` — cancel a still-queued generation job (204; 404 unknown; 409 once running)
- Durable chain summaries expose additive `cancelling: true` after a running cancellation is accepted. Keep the UI in Cancelling until the runner settles `cancelled`; do not infer completion from a finalized file alone.
- `GET /api/events` — one server-wide SSE stream of `job_queued`/`job_started`/`job_ended`, `gallery_added`/`gallery_removed`, `queue_paused`/`queue_resumed`, plus the additive durable-sequence lifecycle `chain_job_queued` (`id`, `model`, `stage_count`), `chain_job_started` (`id`, `model`), and `chain_job_ended` (`id`, `state`). Deltas only — subscribe first, then bootstrap from `/api/queue` + `/api/gallery`. Ephemeral legacy-shim chain jobs stay silent
- `GET /api/history?query=&limit=` · `DELETE /api/history[?keep=N]` — prompt history (newest first, substring filter, limit ≤ 500; 503 when the metadata DB is disabled)
- `GET /api/config` · `GET/PUT/DELETE /api/config/:key` — the `mold config` verbs over HTTP: rows are `{ key, value, source: db|file|env, env_var? }`; PUT routes by surface like `config set` (403 on env-overridden keys), DELETE resets DB-backed keys like `config reset`
- `GET /api/config/profiles` · `PUT /api/config/profile` — list/switch the active settings profile (503 when the metadata DB is disabled)
- `GET /api/status` · `GET /health` · `GET /api/capabilities`

Placement probes remove prompt, negative/original prompt, source/mask/control
images, edit images, keyframe images, audio, and source-video contents while
retaining planning structure. LoRA paths are intentionally retained because
they name server-local artifacts whose presence and size affect exact
feasibility; candidate probing is therefore limited to the configured hosts
the user has connected. A response is usable only when its full version-1
planned or explicit non-authoritative-unsupported shape validates. Clients may
retain compatible routing for that strict `unsupported` shape or a legacy
`404`/`405`; every other HTTP or malformed response is a definitive failure.
Local prompt-expansion and post-generation-upscale utility previews likewise
remain non-authoritative `unsupported` until the runtime's dynamic CPU
fallback and GPU-lease behavior have exact plans.

### Prometheus Metrics

When built with the `metrics` feature flag (included in Docker images and Nix builds), the server exposes a `GET /metrics` endpoint in Prometheus text exposition format. This endpoint is excluded from auth and rate limiting for monitoring scrapers.

Metrics include: HTTP request rates/latency, generation duration, queue depth, model load tracking, GPU memory usage, and server uptime.

## Key Environment Variables

| Variable                              | Default                           | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MOLD_HOME`                           | `~/.mold`                         | Base directory for config, cache, and default models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_DEFAULT_MODEL`                  | `flux2-klein:q8`                  | Default model (smart fallback to only downloaded model)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_HOST`                           | `http://localhost:7680`           | Remote server URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_MODELS_DIR`                     | `$MOLD_HOME/models`               | Model storage path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_OUTPUT_DIR`                     | `~/.mold/output`                  | Image output directory (set empty to disable)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_THUMBNAIL_WARMUP`               | unset                             | Set `1` to prebuild gallery thumbnails at server startup                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `MOLD_GALLERY_TRASH_RETENTION_DAYS`   | `30`                              | Env override of `gallery.trash_retention_days`: days a trashed print stays in `<output_dir>/.trash/` before the sweep purges it; `0` keeps trashed prints forever, max 3650                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_PORT`                           | `7680`                            | Server port                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_LOG`                            | `warn`                            | Log level (trace/debug/info/warn/error)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_DISTRIBUTION_IMAGE_VERSION`     | `latest`                          | Release-build-only input: official stable builds embed the exact release and resolve its target `@sha256` manifest; rolling/source/Nix builds use mutable `latest*`. Do not treat it as a runtime routing override.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_EAGER`                          | unset                             | Set `1` to keep all components loaded                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_OFFLOAD`                        | unset                             | Set `1` to force block offload for FLUX, Flux.2, Z-Image, Qwen-Image, LTX-2, Wan, and SD3 BF16/FP8 paths where implemented. FLUX/Flux.2/Z-Image/Qwen keep fitting blocks GPU-resident; LTX-2, Wan (GGUF only — parks every block), and SD3 full-stream.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_RESERVE_VRAM_MB`                | 400 (Linux), 600 (Win), 0 (macOS) | OS / cuBLAS workspace reserve subtracted from `free_vram_bytes` before any budget decision. `0` disables                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `MOLD_QUEUE_JOURNAL_DISABLE`          | unset                             | `1` turns the durable generation queue off. Jobs still run; nothing survives a restart and `queue.durable_queue` reports false.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_QUEUE_JOURNAL_MAX_BYTES`        | 32 MiB                            | Ceiling on one recorded request. A larger one (an inline video) runs normally and reports `durable: false` rather than being half-persisted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `MOLD_QUEUE_MAX_DISPATCH_ATTEMPTS`    | `2`                               | Starts before a job is held instead of retried. Charged only when a worker actually claims it, so waiting behind a long render through many restarts costs nothing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_QUEUE_MAX_REPLAY_SEEN`          | `10`                              | Restarts that may replay a job which never starts before it is held. Sized for a crash loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `MOLD_QUEUE_ADOPT_OWNER`              | unset                             | Adopt a specific orphaned queue by owner id (printed in the startup warning). Only needed when several retained queues share one `MOLD_HOME` and none was last used by this server.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_SHUTDOWN_ABORT_SECS`            | `45`                              | Seconds the server waits for its GPU workers after `SIGTERM` before exiting anyway. Under systemd set `services.mold.shutdown.abortSeconds` instead, so `TimeoutStopSec` stays derived from it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_KEEP_TE_RAM`                    | off                               | `1` parks text encoders on CPU host RAM between requests; any other value (and unset) keeps drop-and-reload. Opt-in on purpose — a park is a multi-GB host allocation and mold cannot see a container's cgroup limit. Since #1044 it also covers Qwen-Image's quantized GGUF Qwen2 encoder, whose `QTensor` bytes move losslessly instead of being re-read from disk (35.1 s per cold prompt); other families' GGUF encoders (T5, Qwen3) still drop and reload. Disabled on Metal. Covers Wan's UMT5-XXL, whose parked F16 copy saves an 11.4 GB disk read per consecutive render.                                                                                                                                                                                                                                                    |
| `MOLD_LORA_BYPASS`                    | `auto`                            | FLUX LoRA application path: `auto` (bypass when LoRAs present, covers offload AND GGUF/quantized via `quantized_transformer.rs`), `on` (always bypass), `off` (legacy merge / `gguf_lora_var_builder`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_STEP_PREVIEW`                   | `1`                               | Live denoise previews on `/api/generate/stream` (`preview` SSE events, FLUX.1/Flux.2/Z-Image/Wan — Wan projects the clip's middle latent frame): latent-resolution PNG per step from the x0 estimate via linear latent→RGB. Rendered on the desktop, web, and iPhone Create canvases (developing under the shared grain). `0` disables.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_VAE_TILED`                      | `auto`                            | Tiled VAE decode for FLUX/FLUX2/SDXL/SD3: `auto` (retry on OOM), `force` (always tile), `off` (disable). Saves VRAM when transformer + LoRAs are still resident.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_LONG_PROMPTS`                   | unset                             | Set `1` to enable ComfyUI-style chunked CLIP encoding (75-token windows; pooled outputs averaged into FLUX's 768-dim `vector_in`). Default off — pre-Tier-2 truncation at 77 preserved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_ATTN`                           | `math`                            | Attention backend: `math` (hand-rolled SDP) or `flash` (candle-flash-attn v2; needs a `--features cuda,flash-attn` build and a CUDA fp16/bf16 tensor). Opt-in even in that build — the default is `math` everywhere so seed reproducibility never depends on the artifact. Ineligible tensors fall back to math silently; a build without the feature warns once.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MOLD_ATTN_CHUNK`                     | auto                              | Override math-attention query chunk size. Positive integers below sequence length enable chunking; `0` / `off` disables. CUDA auto-chunks long queries at `512`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_FLUX_DELTA_CACHE`               | `1`                               | Set `0` to disable FLUX LoRA delta caching, reducing standing host RAM during GGUF + LoRA rebuilds at the cost of recompute on the next rebuild.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_FLUX_KEEP_TRANSFORMER`          | `0`                               | Set `1` to keep the FLUX transformer loaded through VAE decode when enough VRAM headroom remains; mold force-drops per request when decode headroom is too low.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_OFFLOAD_PREFETCH`               | `on`                              | FLUX offload async H2D prefetch stream (`off` reverts to synchronous)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_PINNED_VRAM_MAX_GB`             | RAM × 0.5 (Linux)                 | Cap on cumulative pinned host memory used by the FLUX offload path                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_EMBED_METADATA`                 | `1`                               | Set `0` to disable PNG metadata                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_MEDIA_ROOTS`                    | unset                             | Platform path-list of allow roots for trusted server-local LTX-2 `audio_file_path` / `source_video_path` API requests. Canonical target files must stay under one configured root.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_PREVIEW`                        | unset                             | Set `1` to display generated images inline in the terminal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_T5_VARIANT`                     | `auto`                            | T5 encoder: auto/fp16/q8/q6/q5/q4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_QWEN3_VARIANT`                  | `auto`                            | Qwen3 encoder: auto/bf16/q8/q6/iq4/q3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_SCHEDULER`                      | unset                             | SD1.5/SDXL: ddim/euler-ancestral/uni-pc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_CFG_PLUS`                       | unset                             | Set `1` to enable CFG++ (manifold-projection guidance). Drops usable CFG to ~1.5–2.5, removes guidance artifacts. Per-request `--cfg-plus` overrides. Supported on SD3, SDXL, and SD1.5 (DDIM scheduler only — Euler-A / UniPC fall back to standard CFG with a warn). Ignored by guidance-distilled families (FLUX, Z-Image, Flux.2) and whenever guidance does not activate CFG.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_VAE_DTYPE`                      | `auto`                            | Override VAE precision: `auto` (per-pipeline default), `bf16`, `fp16`, `fp32`. Use `fp32` to fix banding artifacts on FLUX/SD3 finetuned VAEs (~2× decode VRAM; tiled VAE absorbs OOM). Wired into FLUX, FLUX2, SD3, SDXL, SD1.5; no-op for Z-Image CPU VAE / Wuerstchen / Qwen-Image (already F32).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_NVFP4_BACKEND`                  | `auto`                            | NVFP4 backend for Flux.2 and LTX-2: `auto` / `portable` use CPU BF16 streaming dequant; `native` is reserved for validated sm_120/Blackwell tensor-core execution and fails clearly on non-Blackwell hosts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_LTX2_GEMMA_DEVICE`              | `auto`                            | LTX-2 Gemma 3 12B prompt encoder placement: `auto` uses the GPU leased to the stage when it has more than 6 GB free, otherwise CPU; it never allocates on an unleased sibling GPU. (The threshold was 24 GB, which described a long-removed eager loader — `load_from_assets` builds through `new_streaming` and drops each of the 48 decoder layers before the next, so real peak residency is ~3.3 GB and a 4090 could never qualify.) `cpu` forces system RAM (~30–60 s encode vs ~1–3 s on GPU); `gpu` pins the assigned GPU and surfaces OOM instead of auto-offloading. An auto-placement OOM retries only Gemma on CPU; the transformer and video VAE remain on CUDA. The deprecated `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER=1` is a one-shot-warn alias for `cpu`. Server-side preflight uses the same resolver as runtime. |
| `MOLD_LTX2_GEMMA_VARIANT`             | `auto`                            | LTX-2 Gemma 3 12B weight format: `auto` (BF16 if both formats present, GGUF if only GGUF), `q4` (force Q4 GGUF — `google/gemma-3-12b-it-qat-q4_0-gguf`, ~7 GB; fits comfortably on a 24 GB card alongside the streaming transformer), `bf16` (force BF16 split — `google/gemma-3-12b-it-qat-q4_0-unquantized`, ~23 GB; historical default). Auto-detection scans the gemma_root for `*.gguf` and `model*.safetensors`. Place the Q4 GGUF manually in your gemma_root for V1 — manifest auto-fetch is deferred.                                                                                                                                                                                                                                                                                                                        |
| `MOLD_LTX2_KEEP_SESSION`              | on                                | LTX-2 retains its runtime session across generations (#1099): a same-prompt repeat serves the cached encoding instead of reloading the ~24 GB Gemma encoder. The session holds only the cached prompt encoding + device handle (the encoder is consumed on first prepare, and neither transformer nor VAE lives in it), so a changed prompt still pays a full load. `0`/`false`/`off` restores the old drop-after-every-generation behavior. Released by model unload/eviction.                                                                                                                                                                                                                                                                                                                                                       |
| `MOLD_LTX2_SPATIAL_TILE`              | `auto`                            | LTX-2 spatial tiling for stage-2 refinement and VAE decode (`--spatial-tile` sets it per run). `auto` engages only past the 2048-px axis span the checkpoints were trained on, so no shape that rendered before the composed ceiling changes; `off` turns a past-the-span render into an error rather than a quietly degraded video; `<px>` / `<px>:<overlap>` (multiples of 32) forces a tile size — used to compare tiled against untiled at a resolution that does not need tiling. Tiles are denoised with per-tile positions renormalized to zero and per-tile noise seeded `seed + tile_index`, then recombined with a separable trapezoidal window; a tiled stage 2 refines video only and carries stage 1's audio through unrefined.                                                                                          |
| `MOLD_LTX2_VAE_FORCE_FULL_DECODE`     | unset                             | Set `1` to disable adaptive temporal chunked LTX-2 VAE decode and force one full decode pass. Useful for debugging/comparison; long or high-resolution clips may OOM.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_LTX2_VAE_FORCE_FRAMEWISE`       | unset                             | Set `1` to force temporal-chunk LTX-2 VAE decode even when a full decode would fit. Reduces peak VRAM at a small decode-time cost.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_LTX2_VAE_DECODE_CHUNK_FRAMES`   | `4` latent frames                 | Positive integer number of latent frames per LTX-2 VAE decode chunk when chunked decode is active.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_LTX2_VAE_DECODE_CONTEXT_FRAMES` | auto                              | Positive integer latent-frame overlap/context around each LTX-2 decode chunk. Default derives from the decoder causal-conv receptive field.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_WAN_SHIFT`                      | per tier (`8.0`; A14B `5.0`)      | Wan flow shift (upstream `--sample_shift`). Process-wide fallback; the request-level `sample_shift` / `--sample-shift` always wins. Finite and positive.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `MOLD_WAN_SOLVER`                     | `unipc`                           | Wan sample solver fallback: `unipc`, `euler` (the Lightning-tuned solver), or `dpm++` (upstream's alternative grid). The request-level `scheduler` / `--sample-solver` always wins.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_WAN_OFFLOAD_BLOCKS`             | auto                              | Trailing Wan DiT blocks to park in host RAM (#776). Auto-engages only when the render will not otherwise fit; `0` disables; an explicit count wins over `MOLD_OFFLOAD`, which parks every block. GGUF only. 81f/832x480 A14B q5: 316.3 s at 17,322 MiB, previously OOM; `:q8` 73f at 16,650 MiB. Trades wall clock for VRAM.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `MOLD_WAN_PREFETCH`                   | `1`                               | `0` disables the background page-cache warm for the A14B partner expert (#802). Host I/O only; never touches VRAM. Swap load 8.7-22.2 s cold vs 5.6 s warm on a 4090.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `MOLD_WAN_STEP_CACHE`                 | `off`                             | Wan first-block residual reuse (#801) on the non-distilled tiers: `off`, `auto` (threshold 0.10, measured 1.85x on the A14B `:q8` tier), or an explicit positive relative-L1 threshold. Refused with a message on distilled or sub-12-step schedules. `off` is bit-identical to denoising every block.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_WAN_STEP_PROFILE`               | off                               | Diagnostic (#775): `1` prints one per-phase, device-synced timing line per Wan denoise step (self/cross attention, ffn, quantized matmuls, casts). Syncs inflate wall time — never leave on in production.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_WAN_FORCE_DMMV`                 | off                               | Diagnostic (#775): `1` forces candle's quantized matmuls onto the dequantize-per-forward fallback for A/B against the MMQ fast path. Changes numerics, runtime, and transient memory; fingerprint-registered so learned estimates never mix with normal runs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_QWEN_QMATMUL`                   | `0`                               | Quantized (GGUF) Qwen-Image linears on CUDA (#1045). Experimental, off by default: `1`/`true`/`on`/`yes` feeds candle's MMQ/MMVQ kernels, which currently NaN on Qwen's shapes (4090-measured; render aborts loudly) — the reproduction recipe, the contrast case, the already-eliminated zero-`amax` hypothesis, and the next steps are in `docs/architecture/qwen-mmq-nan.md` (#1048). Default keeps the per-forward dequantize-to-BF16 arm. MMQ-ineligible (`IQ*`/float-stored, or a row width the MMQ block size does not divide) weights, CPU-staged weights, and a process already on the fallback via `MOLD_WAN_FORCE_DMMV=1` keep the dequant arm regardless; Metal and CPU ignore it. Fingerprint-registered — the two settings never share a learned estimate.                                                              |
| `MOLD_ZIMAGE_QMATMUL`                 | `0`                               | Quantized (GGUF) Z-Image linears on CUDA. Experimental, off by default: `1`/`true`/`on`/`yes` feeds candle's MMQ/MMVQ kernels, which return non-finite values for Z-Image's linears (4090-measured: feed-forward `inf` from finite inputs past t≈0.07; renders come out solid black) — same defect family as `MOLD_QWEN_QMATMUL` (#1048). Default dequantizes each weight per forward on CUDA; Metal and CPU keep `QMatMul` (the #942 sd.cpp-validated arm). Fingerprint-registered.                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_QWEN_FP8_CACHE`                 | off                               | `1` caches the widened BF16 weights of an FP8 Qwen-Image checkpoint instead of re-casting per forward (#1045). Roughly doubles resident transformer VRAM, so it is opt-in. In-memory transformer only — `--offload` blocks go through `candle_nn::Linear`, which has no FP8 arm, so they always widen once at load whatever this is set to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_MAX_CACHED_MODELS`              | `3`                               | Maximum cached engines, including the GPU-resident model and parked entries. Range: `1`-`16`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_CACHE_IDLE_TTL_SECS`            | `1800`                            | Idle TTL for parked cache entries before background eviction. Range: `60`-`86400`; the GPU-resident model is never evicted for age.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `MOLD_QUEUE_LOOKAHEAD_BUFFER`         | `8`                               | Number of queued jobs considered for same-loaded-model locality reordering. Range: `1`-`64`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `MOLD_QUEUE_MAX_DEFERRALS`            | `3`                               | Maximum times the head job can be deferred for locality before force-dispatch. Range: `0`-`32`; `0` disables deferral.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `MOLD_MALLOC_TRIM`                    | `1`                               | Linux/glibc only: set `0` to skip `malloc_trim(0)` after generation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `MOLD_API_KEY`                        | unset                             | API key for server auth (single, comma-separated, or `@/path/to/keys.txt`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `MOLD_RATE_LIMIT`                     | unset                             | Per-IP rate limit for generation endpoints (e.g., `10/min`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `MOLD_RATE_LIMIT_BURST`               | unset                             | Burst allowance override (defaults to 2x rate)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MOLD_CORS_ORIGIN`                    | unset                             | Restrict server CORS to specific origin                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_UPSCALE_MODEL`                  | unset                             | Default upscaler model for `mold upscale`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `MOLD_UPSCALE_TILE_SIZE`              | unset                             | Tile size for memory-efficient upscaling (0 to disable)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_EXPAND`                         | unset                             | Set `1` to enable prompt expansion by default                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_EXPAND_BACKEND`                 | `local`                           | Expansion backend: `local` or OpenAI-compatible API URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `MOLD_EXPAND_MODEL`                   | `qwen3-expand:q8`                 | LLM model for local expansion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `MOLD_EXPAND_TEMPERATURE`             | `0.7`                             | Sampling temperature for expansion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `MOLD_EXPAND_THINKING`                | unset                             | Set `1` to enable thinking mode in expansion LLM                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `MOLD_EXPAND_SYSTEM_PROMPT`           | unset                             | Custom single-expansion system prompt template                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MOLD_EXPAND_BATCH_PROMPT`            | unset                             | Custom batch-variation system prompt template                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `HF_TOKEN`                            | unset                             | HuggingFace token for gated models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

For the web server, `HF_TOKEN` and `CIVITAI_TOKEN` are defaults. A credential
saved in web Settings overrides the matching environment value until the saved
override is cleared.

## Inference Modes

1. **Remote** (default): connects to `mold serve` via HTTP
2. **Local fallback**: if server unreachable, auto-falls back to local GPU
3. **Local forced** (`--local`): skip server, run on local GPU directly

Models auto-pull if not downloaded: `mold run flux2-klein "a cat"` will download the model first if needed.

## Practical Tips

- Use `flux2-klein:q8` for fast iterations (4 steps, ~10s on RTX 4090)
- Use `flux-dev:q4` for final quality images (25 steps)
- Use `--seed` for reproducibility — same seed + same prompt = same image
- Quantized models (q4/q6/q8) use less VRAM than fp16/bf16
- FP8 safetensors models auto-convert to Q8 GGUF on first use (fits 24GB cards)
- `--eager` trades VRAM for speed (keeps encoders loaded between generations)
- Dimensions must be multiples of 16 (32 for LTX video families and `wan22-ti2v-5b`, whose 2.2 VAE needs the wider grid — `/api/models` advertises the per-model `dimension_alignment`); total pixels capped at 1.8 MP. LTX-2's ceiling is **per model**: a checkpoint that ships the spatial upsampler composes (stage 1 at half size, learned x2 upsample, tiled stage-2 refinement) and reaches 4096px on the long edge / 8.9 MP total; one that renders in a single pass (single-file `cv:` / `hf:` builds) stays at 2048px / 2.09 MP. `/api/models` advertises both as `max_pixels` and the additive `max_axis_pixels` — read them, never hardcode
- LTX-2 output ladder: 1280x704, 1920x1088, 2560x1408, 3840x2112 (4K UHD), each with its portrait transpose. Every rung is /64 so the halved stage-1 shape lands on the VAE's /32 grid. Generation is admitted to 4096px — that is where the halved stage 1 itself leaves the trained span — but the ladder stops at 3840 because the bundled OpenH264 encoder refuses past 3840x2160, so 3840x2176 renders and then fails at save time. The 4096px ceiling belongs to the default x2 rung: `--spatial-upscale x1.5` divides by 1.5 and so reaches only 3072px, while explicitly choosing a single-pass pipeline (`one-stage`, `retake`, `lip-dub`) drops it back to 2048px. **4K needs `--spatial-tile 768` on a 24 GB card** (measured RTX 4090, 19B fp8, 25 frames: 1080p 18.4 GB / 1440p 18.1 GB / 4K 18.2 GB; 4K with the default 1280px tile OOMs in the VAE decode rather than rendering smaller)
- For img2img, source images auto-resize to fit the model's native resolution (preserving aspect ratio). A 1024x1024 source with SD1.5 (512x512 native) generates at 512x512; a 1920x1080 source generates at 512x288. Use `--width`/`--height` to override
- Set `MOLD_HOME` to relocate all mold data (config, cache, models)
- LoRA adapters require FLUX BF16 models; use `--lora-scale 0.5-0.8` for subtle effects
- LTX-2 `--spatial-tile` (= `MOLD_LTX2_SPATIAL_TILE`) splits stage-2 refinement and the VAE decode into overlapping spatial tiles. `auto` (default) engages only past the 2048-px axis span the checkpoints were trained on, so no currently renderable shape changes; `off` refuses a render past the span rather than degrading it quietly; `<px>` / `<px>:<overlap>` forces a tile size, which is how a tiled render is compared against an untiled one. A tiled stage 2 refines video only and carries stage 1's audio through unrefined, matching upstream `hdr_ic_lora.py:504-507`.
- On 24GB cards, use `--offload` with BF16 FLUX / Flux.2 / Z-Image / Qwen-Image / SD3 when quantization is not acceptable, and with LTX-2 when you want the conservative full-streaming path. FLUX / Flux.2 / Z-Image / Qwen-Image keep fitting blocks resident; LTX-2 and SD3 full-stream.

## Discord Bot

Mold includes an optional Discord bot that bridges Discord slash commands to a running `mold serve` instance. The bot depends only on `mold-core` (HTTP client) — no GPU needed on the bot host.

### Running

```bash
# Run server + bot in one process
MOLD_DISCORD_TOKEN="your-bot-token" mold serve --discord

# Or run the bot separately (connects to a remote server)
export MOLD_DISCORD_TOKEN="your-bot-token"
export MOLD_HOST="http://gpu-host:7680"  # optional, defaults to localhost
mold discord
```

### Slash Commands

- `/generate [prompt] [model] [width] [height] [steps] [guidance] [seed]` — generate an image. `prompt` is optional only when a `source_image` is attached (LTX-2 image-to-video); the server's family-aware validator still rejects an empty prompt everywhere else. Changing it from required to optional needs a slash-command re-registration before Discord shows the new signature.
- `/expand <prompt> [model_family] [variations]` — expand a short prompt into detailed image generation prompts
- `/models` — list available models with status
- `/status` — show server health, GPU info, uptime
- `/quota` — check remaining daily generation quota
- `/admin reset-quota @user` — reset a user's daily quota (requires Manage Server)
- `/admin block @user` — temporarily block a user from generating (requires Manage Server)
- `/admin unblock @user` — unblock a previously blocked user (requires Manage Server)

`/status` reports every runtime-visible GPU/MIG device. Large fleets are split
into deterministic follow-up embeds so each field, embed, and message remains
within Discord limits; no device is silently dropped at 64-device scale.

### Environment Variables

| Variable                     | Default                 | Description                                                     |
| ---------------------------- | ----------------------- | --------------------------------------------------------------- |
| `MOLD_DISCORD_TOKEN`         | —                       | Bot token (falls back to `DISCORD_TOKEN`)                       |
| `MOLD_HOST`                  | `http://localhost:7680` | mold server URL                                                 |
| `MOLD_DISCORD_COOLDOWN`      | `10`                    | Per-user cooldown (seconds)                                     |
| `MOLD_DISCORD_ALLOWED_ROLES` | —                       | Comma-separated role names/IDs for access control (unset = all) |
| `MOLD_DISCORD_DAILY_QUOTA`   | —                       | Max generations per user per UTC day (unset = unlimited)        |

### NixOS

```nix
services.mold.discord = {
  enable = true;
  package = inputs.mold.packages.${system}.mold-discord;
  tokenFile = config.age.secrets.discord-token.path;
};
```

## Desktop App

The local server is the sole gallery authority whenever it exists, including unhealthy startup/shutdown-timeout states. Native list/save/delete/import use authenticated loopback HTTP, media elements exchange the durable key for exact-path tickets, and direct filesystem access is legal only while the lifecycle lock proves `LocalServer::Off`.

The native macOS desktop app (Tauri 2 + Vue 3) lives in `desktop/`. It auto-detects a running server on `localhost:7680` or embeds an authenticated Metal server bound to the LAN and advertised over mDNS. That local server is permanently the app's own engine (**This device**); remote servers are host-list entries managed in the **Machines** workspace (This-device card with a copyable persistent API key, Add host, Connected, Remembered, and network discovery), deduplicated by each server's stable instance UUID with display names that follow the server hostname — old remote-primary installs migrate into the list automatically. Clicking a host in Machines opens a detail view with live GPU/CPU/RAM telemetry, models-disk usage, queue state, and that host's installed models. Create uses the union of models installed on every connected host and shows the first-model pull screen only after all hosts report none, so remote-only models route without a local download. Its settings inspector resizes from the left edge across 280–480 px, persists committed widths, defaults to a no-wrap 340 px, and resets on divider double-click; the simplified essentials remain its default view, while Advanced toggles capability-gated, always-open icon sections inline below them without covering the canvas; web uses the same icon-led sections with no nested disclosure. Dropping a PNG/JPEG anywhere in Create attaches it as the family-appropriate source, and embedded Mold metadata restores its settings first. Composer Up/Down recall merges prompt history from all ready hosts and includes a just-submitted remote prompt immediately. Sequences (the Create inspector's Output = Sequence setting) use the same all-host union for video models and keep limits, creation, job actions, events, and previews on the selected model's host. The five workspaces are Create (live "Develop" progress), Library (the unified multi-host gallery with a persisted Lightroom-style thumbnail-size slider and its Runs + Prompts History drawer), Models (the single install/repair workspace with pinned parallel cancellable pulls), Machines (host list, per-host detail, the shared reorderable queue, and full RunPod pod and network-volume lifecycle management), and Settings; plus full-resolution image clipboard copy, persistent 80–130% whole-app scaling (⌘+/⌘−/⌘0), provenance-tagged settings, a StatusPopover at the collapsible sidebar's foot, and a ⌘K command palette. That palette (on web too) searches the whole fleet for models: **Use `<name>`** for a model the next job's machine has, **Use `<name>` · on `<machine>`** for one only another connected machine has (picking it repins the generation target there, while automatic Auto / Most capable routing is left alone), and **Install `<name>` · not installed** from a debounced live checkpoint catalog search for one nobody has — queued on the first machine that can take it and named in the toast, without opening the Models machine picker. Sequence clips live on a rail inside the Create composer; seam pills between clips open the transition editor, and running sequence jobs share the Create activity strip with prints. That strip is present tense: it shows in-flight work plus at most two dismissible failed/interrupted rows that expire after five minutes, plus one digest chip counting settled sequences that opens `/library?panel=history&tab=sequences`. A finished sequence resolves to the Create canvas (**Edit sequence** / **Show in library**), the print in the Library, and its job record in the History drawer's **Sequences** tab — which is also where the host-scoped **Clear inactive** and **Clean up disk** maintenance lives. A sequence print carries per-clip provenance, so **Reuse settings** on it loads those clips onto the Create clip rail as a NEW sequence (desktop, web, and iPhone) instead of the newline-joined single prompt, and desktop/web add **Edit sequence** to re-enter the original durable job on its origin host with cached clips — checked once on click, falling back to reuse with a toast on 404 and refusing to downgrade when the host is unreachable. Web has the same History drawer at `?panel=history`. RunPod volume selection persists; volume-backed launches force Secure Cloud in the volume's datacenter and omit the redundant workspace disk.

Desktop and iPhone Create use Batch as the prompt-expansion count. Batch 1 keeps quick Expand/undo and freezes one concrete route through the next Generate/Develop. A stale quick rewrite must offer explicit re-expand-from-original on the current route, generate-the-visible-rewrite-anyway on the current route, and restore-original actions instead of an enabled dead-end submit. Batch N greater than 1 prepares exactly N editable, non-empty prompts on that host before queuing anything; the same frozen route is used for every sibling with the source prompt retained as provenance. Prepared siblings carry additive `batch_id`, one-based `batch_index`, and `batch_count` through long-video chains, completion, and Library metadata. Edits are valid work. Source/model/family/host/count changes preserve the set as stale and block generation until refresh or discard. Never silently resize, reroute, fall back, or erase a prepared set, including through missing-model pull/resume. Expansion-model recovery stays inside Create for both batch modes, follows the returned job ID (or a newly observed exact-model row from older timing), and renders Connecting, Starting, Queued, Pulling details, Ready, failure, and cancellation without redirecting to Models. Desktop reads `useDownloadsStore`; iPhone shares `useMobileDownloadsStore` with the Models view and freezes the selected remote host ID, URL, Keychain key, and server instance in one immutable record without importing desktop-primary stores. Its exact-route lease belongs only to one pull attempt: it joins a compatible Models POST already in Starting, releases after every terminal/error/stale/superseded/aborted path, and is reacquired by Retry. Editing/removing reviewed work supersedes a pending replacement. Unique view consumer IDs prevent remount teardown races, and partial prepared failures name their one-based variation plus reviewed prompt alongside any unconfirmed-cancellation caveat while successful prints remain available. Generation labels resolve opaque catalog IDs through current model metadata for display only; wire requests and persisted identity keep the raw ID.

Signed builds also expose **Settings → Updates** with persisted **Stable** and **Nightly** channels. Startup performs a best-effort check only; available updates appear in a persistent app banner and as a native notification while backgrounded, the menu and Settings offer the same manual check, and nothing downloads or installs until the user chooses **Update and restart**. Stable follows tagged releases through the public `mold-desktop-stable.json` manifest. Nightly follows signed, notarized builds from desktop-relevant `main` commits through the rolling `mold-desktop-nightly.json` manifest. Before touching the installed app, Tauri verifies the Minisign signature and Mold fully extracts the archive to temporary storage, binds the bundle ID/version to the manifest, runs strict Apple signature and Gatekeeper checks, validates the running bundle and install location, and proves the bundle can be replaced. Only then does Mold atomically exchange the staged and installed bundles with macOS `RENAME_SWAP` and restart. There is no post-launch watchdog or automatic rollback: the update either passes preflight and installs or fails while the running version remains installed. Selecting Stable from a newer Nightly does not downgrade immediately; it waits for a newer stable version.

Maintainer note: updater publishing additionally requires the GitHub Actions secrets `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. Keep the private key out of source and logs, retain a controlled offline backup, and treat rotation as a staged release: existing clients must first receive the replacement public key in an artifact signed by the old key.

The iPhone companion is a separate, remote-only Tauri shell in
`apps/mobile/src-tauri`, with its shared Vue entry in `desktop/src/mobile`. Its
primary tabs are Create, Library, Models, and Machines, with a header-pushed
Settings screen. It accepts IP/DNS/HTTPS and Tailscale MagicDNS names and uses
Apple DNS-SD to discover `_mold._tcp`. Host metadata stays in WebView storage;
API keys stay in the iOS Keychain. Desktop and web Settings create a 256-bit
random, one-use, two-minute pairing ticket; the QR carries the reachable host
URL and server instance ID but never the API key. Native camera scanning
redeems the ticket against that exact instance and writes the returned key
directly to Keychain, while Bonjour and manual entry remain available.

Create shares desktop's capability/request logic, prompt tools/templates,
independently cancellable batch queue, source/edit/mask/ControlNet/LoRA inputs,
resolution/seed controls, estimates, a full-screen Advanced sheet, prompt style
presets that compose at submit, and image/video parameters. Library merges
all saved hosts, streams native video through short-lived exact-path media
tickets, swipes between full-screen prints, exposes native Copy image / Save
photo plus Use as prompt/source, and opens generated stills in the same viewer.
Pinching the Library grid resizes thumbnails across 2–5 columns (default 3) —
the iPhone answer to the desktop/web thumbnail-size slider, persisted per
device rather than through the shared pixel-target key.
Its persistent New markers mirror desktop Library visits; both shells badge
upscaled images from saved output provenance.
Host detail shows telemetry, models-disk, queue, downloads, and installed
models; still-queued rows have a confirmed 44pt cancellation action routed to
that exact Keychain-authenticated host, while running work remains
non-cancellable. Models merges installed/live results, lets Pull target a different host
without changing Create, and immediately shows Connecting → Starting → Queued
→ Pulling N% while preventing duplicate/racing requests.

Settings persists the Mold Studio families (Mold/Safelight) × System/Dark/Light,
defaults fresh installs to Safelight + System with Photos auto-save enabled
while retaining valid saved choices, and synchronizes UIKit's appearance/status
bar. Its default-on Photos option fetches completed
stills from the authenticated host gallery and saves them through the native
bridge; post-generation upscales save original and upscaled images, while
videos remain remote. Preserve safe areas, 44pt controls, 16px editable text,
disabled document zoom and overscroll bounce, plus the Library's scoped swipe.
The TestFlight workflow runs after eligible successful iOS `main` CI (no cron),
uploads a build eligible for internal groups and external Beta App Review,
waits for App Store Connect `VALID`, and verifies the baseline `Mold Internal`
tester membership. Never set Apple's `testFlightInternalTestingOnly` export
option to true. See `apps/mobile/README.md` for the complete developer and
release contract.

Devshell commands (run inside `nix develop`):

```bash
desktop-dev        # Tauri app with hot reload (Vite on :1430)
desktop-build      # build the Mold.app bundle
desktop-check      # CI gate: rustfmt, clippy, vue-tsc, prettier
desktop-test       # cargo test + vitest
desktop-ui         # frontend-only Vite server (pair with a running `serve`)
frontend-bun-lock  # regenerate the repo-root bun.lock and bun.nix
ios-dev            # run the iPhone app with Tauri hot reload
ios-run            # production run on an iPhone or simulator
ios-check          # cross-check the simulator Rust target
ios-build          # archive/export for App Store Connect
```

The mobile source entry is `index.mobile.html`, but Tauri's packaged resolver
boots `index.html`; `vite.mobile.config.ts` performs that rename. Run
`scripts/tests/ios-release-assets.sh` before shipping, and use
`scripts/generate-ios-icons.sh` to regenerate the opaque Apple catalog from the
desktop icon master.

## Agent Skill Management

Mold embeds this skill and can install it for the same agents and paths as nxv:

```bash
mold skill list                              # Paths, detection, install status
mold skill install codex                     # One user-wide agent
mold skill install claude codex              # Several explicit agents
mold skill install --detected                # Detected user configurations
mold skill install --all                     # Every supported agent
mold skill install codex --project           # Current project
mold skill install --detected --dir ~/repo   # Another project's agent paths
mold skill uninstall codex                   # Remove one user-wide copy
mold skill uninstall --project               # Sweep project-level Mold copies
mold skill show                              # Print the embedded SKILL.md
```

| Agent      | User-wide                 | Project-level     |
| ---------- | ------------------------- | ----------------- |
| `claude`   | `~/.claude/skills/`       | `.claude/skills/` |
| `codex`    | `~/.codex/skills/`        | `.agents/skills/` |
| `pi`       | `~/.pi/agent/skills/`     | `.pi/skills/`     |
| `openclaw` | `~/.openclaw/skills/`     | `.agents/skills/` |
| `copilot`  | `~/.copilot/skills/`      | `.github/skills/` |
| `cursor`   | `~/.cursor/skills/`       | `.agents/skills/` |
| `gemini`   | `~/.gemini/skills/`       | `.agents/skills/` |
| `amp`      | `~/.config/amp/skills/`   | `.agents/skills/` |
| `goose`    | `~/.config/goose/skills/` | `.agents/skills/` |
| `agents`   | `~/.agents/skills/`       | `.agents/skills/` |

Installation requires agent names, `--detected`, or `--all`; a bare install
writes nothing. Shared project paths are written once. Install atomically
replaces only `mold/SKILL.md`. Uninstall removes only that file and removes the
`mold/` directory only when it is empty.

## Updating This Skill

The Mold binary embeds the canonical skill and installs it atomically. Refresh
one user-wide agent install after upgrading Mold:

```bash
mold update
mold skill install codex
```

Use `mold skill install --detected` to refresh detected agents, `--all` for every
supported agent, or add `--project` / `--dir <PATH>` for project-local installs.
`mold skill show` prints the exact embedded `SKILL.md`.
