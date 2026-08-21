# Server API

When running `mold serve`, the same engine behind Mold's CLI becomes a REST API
with SSE progress. This keeps shell scripts, agents, native apps, browser
clients, and custom integrations on one generation contract.

## Endpoints

| Method   | Path                                          | Description                                                                                                       |
| -------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `POST`   | `/api/generate`                               | Generate images from prompt                                                                                       |
| `POST`   | `/api/generate/stream`                        | Generate with SSE progress streaming                                                                              |
| `POST`   | `/api/generate/estimate`                      | Estimate request-sensitive peak memory for a generation request                                                   |
| `POST`   | `/api/generate/chain`                         | Chained video generation (LTX-2, LTX-Video, Wan)                                                                  |
| `GET`    | `/api/capabilities/ltx2-control-adapters`     | Compatible official IC-LoRA controls for an installed LTX-2 model                                                 |
| `POST`   | `/api/generate/chain/stream`                  | Chained video with SSE progress                                                                                   |
| `POST`   | `/api/generate/chain/validate`                | Normalize and validate a chain without queueing work                                                              |
| `POST`   | `/api/chain-jobs`                             | Create a durable async chain job                                                                                  |
| `GET`    | `/api/chain-jobs`                             | List durable chain jobs                                                                                           |
| `GET`    | `/api/chain-jobs/:id`                         | Get durable chain-job detail                                                                                      |
| `GET`    | `/api/chain-jobs/:id/events`                  | Durable chain-job SSE events                                                                                      |
| `POST`   | `/api/chain-jobs/:id/resume`                  | Resume a failed, interrupted, or cancelled chain job                                                              |
| `POST`   | `/api/chain-jobs/:id/retake`                  | Retake one chain-job stage                                                                                        |
| `POST`   | `/api/chain-jobs/:id/amend`                   | Replace a chain job's stage list in place, reusing cached clips                                                   |
| `POST`   | `/api/chain-jobs/:id/cancel`                  | Cancel a queued or running chain job                                                                              |
| `DELETE` | `/api/chain-jobs/:id`                         | Delete a non-running chain job                                                                                    |
| `POST`   | `/api/chain-jobs/gc`                          | Run chain-job artifact GC                                                                                         |
| `GET`    | `/api/chain-jobs/:id/stages/:idx/preview`     | Fetch a stage preview JPEG                                                                                        |
| `GET`    | `/api/chain-jobs/:id/stages/:idx/media`       | Stream a completed stage MP4; HEAD and byte ranges are supported                                                  |
| `POST`   | `/api/chain-jobs/:id/stages/:idx/media-token` | Mint a short-lived ticket scoped to that exact stage-media path                                                   |
| `POST`   | `/api/expand`                                 | Expand a prompt using LLM, optionally absorbing a visual `style` directive                                        |
| `POST`   | `/api/remix`                                  | Generate exact-count, subject-preserving prompt alternatives with structured provenance                           |
| `GET`    | `/api/models`                                 | List available models                                                                                             |
| `GET`    | `/api/models/:model/components`               | List required model component readiness and paths                                                                 |
| `GET`    | `/api/loras`                                  | List installed LoRAs, optionally filtered by `?model=` compatibility                                              |
| `POST`   | `/api/models/load`                            | Load/swap the active model                                                                                        |
| `POST`   | `/api/models/pull`                            | Pull/download a model                                                                                             |
| `DELETE` | `/api/models/unload`                          | Unload model to free GPU memory                                                                                   |
| `DELETE` | `/api/models/:model`                          | Remove a downloaded model (keeps components shared with other models)                                             |
| `GET`    | `/api/gallery`                                | List saved images                                                                                                 |
| `POST`   | `/api/gallery/media-token`                    | Mint a short-lived, read-only ticket for one full-size gallery path                                               |
| `POST`   | `/api/pairing/sessions`                       | Mint an authenticated, one-use, two-minute iPhone pairing ticket                                                  |
| `POST`   | `/api/pairing/claim`                          | Redeem a pairing ticket once; the durable key is never present in the QR                                          |
| `GET`    | `/api/gallery/image/:name`                    | Fetch a saved image                                                                                               |
| `DELETE` | `/api/gallery/image/:name`                    | Delete a saved image                                                                                              |
| `GET`    | `/api/gallery/thumbnail/:name`                | Fetch a cached thumbnail                                                                                          |
| `GET`    | `/api/gallery/preview/:name`                  | Fetch a cached GIF preview for video gallery rows                                                                 |
| `GET`    | `/api/downloads`                              | List up to two active downloads (`active_jobs`), plus queued, failed, and completed jobs                          |
| `POST`   | `/api/downloads`                              | Queue a manifest model download                                                                                   |
| `DELETE` | `/api/downloads/:id`                          | Cancel a queued or active download                                                                                |
| `GET`    | `/api/downloads/stream`                       | Download queue updates as SSE                                                                                     |
| `GET`    | `/api/catalog/families`                       | Live catalog family/kind metadata                                                                                 |
| `GET`    | `/api/catalog/search`                         | Search the live HF/Civitai catalog; sort by downloads, recent additions, or rating                                |
| `GET`    | `/api/catalog/installed`                      | List installed catalog entries and LoRAs                                                                          |
| `GET`    | `/api/catalog/credentials`                    | Return masked status for server-owned HF/Civitai credentials                                                      |
| `PUT`    | `/api/catalog/credentials/:provider`          | Save an `hf` or `civitai` credential on the serving host                                                          |
| `DELETE` | `/api/catalog/credentials/:provider`          | Remove a saved host credential and fall back to its environment-provided default                                  |
| `GET`    | `/api/catalog/:id`                            | Resolve one `hf:` or `cv:` catalog entry                                                                          |
| `POST`   | `/api/catalog/:id/download`                   | Queue a catalog entry plus missing companions                                                                     |
| `POST`   | `/api/upscale`                                | Upscale image with Real-ESRGAN                                                                                    |
| `POST`   | `/api/upscale/stream`                         | Upscale with SSE tile progress                                                                                    |
| `GET`    | `/api/resources`                              | Latest RAM/GPU resource snapshot                                                                                  |
| `GET`    | `/api/resources/stream`                       | Resource snapshots as SSE                                                                                         |
| `GET`    | `/api/devices`                                | Stable runtime-visible device inventory with nullable cached telemetry                                            |
| `PATCH`  | `/api/devices/:id`                            | Persist and apply a scheduler-V2 GPU enable/disable lifecycle request                                             |
| `GET`    | `/api/events`                                 | Server-wide lifecycle events (job + gallery) as SSE                                                               |
| `GET`    | `/api/queue`                                  | Server-authoritative job listing (queued + running, UUIDv4 ids); used by the SPA to reconcile dropped SSE streams |
| `PATCH`  | `/api/queue/:id`                              | Update the preferred GPU lane and/or dispatch position for a queued job                                           |
| `DELETE` | `/api/queue/:id`                              | Cancel a still-queued generation job                                                                              |
| `GET`    | `/api/history`                                | Prompt history, newest first (`?query=` substring filter, `?limit=` up to 500)                                    |
| `DELETE` | `/api/history`                                | Clear prompt history (`?keep=N` trims to the most recent N)                                                       |
| `GET`    | `/api/capabilities`                           | Feature capabilities, including optional per-host expansion and LAN-discovery state                               |
| `GET`    | `/api/discovery/peers`                        | DNS-SD peers visible on the serving host's LAN (when `discovery.can_browse`)                                      |
| `GET`    | `/api/capabilities/chain-limits`              | Chain-generation request limits                                                                                   |
| `GET`    | `/api/config`                                 | List every effective config row with its source (`db`/`file`/`env`)                                               |
| `GET`    | `/api/config/:key`                            | Read one config key (value + owning source)                                                                       |
| `PUT`    | `/api/config/:key`                            | Set a config key, routed by surface like `mold config set`                                                        |
| `DELETE` | `/api/config/:key`                            | Reset a DB-backed key like `mold config reset`                                                                    |
| `GET`    | `/api/config/profiles`                        | List settings profiles and the active one                                                                         |
| `PUT`    | `/api/config/profile`                         | Switch the active settings profile                                                                                |
| `PUT`    | `/api/config/model/:name/placement`           | Save model-specific device placement defaults                                                                     |
| `DELETE` | `/api/config/model/:name/placement`           | Clear model-specific device placement defaults                                                                    |
| `POST`   | `/api/shutdown`                               | Trigger graceful server shutdown                                                                                  |
| `GET`    | `/api/status`                                 | Server health + status                                                                                            |
| `GET`    | `/health`                                     | Simple 200 OK health check                                                                                        |
| `GET`    | `/api/openapi.json`                           | OpenAPI spec                                                                                                      |
| `GET`    | `/api/docs`                                   | Interactive API docs (Scalar)                                                                                     |
| `GET`    | `/metrics`                                    | Prometheus metrics (feature-gated)                                                                                |

### Capability discovery

`GET /api/capabilities` is additive and safe to feature-detect. Current
servers advertise the accepted catalog sort values in
`catalog.sort` (`downloads`, `recent`, `rating`), queue controls as
`queue.can_pause`, `queue.can_cancel_all`, and `queue.can_reorder`, and
server-assisted DNS-SD as `discovery.can_browse`, and the read-only device
resource as `devices.available`. `devices.lifecycle` is true only when
scheduler V2 is the authoritative runtime; legacy, observe, maintenance, and
unavailable runtimes report false. Those runtimes advertise
`devices.restart_enable` instead: clients may offer **Enable on restart** only
for a device whose persisted preference is disabled. Live controls must also
require `dispatch.v2_authoritative`. Stable pin support is advertised as
`devices.stable_pins`; versioned lanes and learned ETA are advertised as
`devices.planned_lanes` and `devices.learned_eta` only while V2 is
authoritative. Dispatch rollout is exposed as `dispatch.active_mode`,
`dispatch.v2_authoritative`, and `dispatch.observes_v2_decisions`. Clients must
only request
`GET /api/discovery/peers` when that discovery flag is true. Older servers may
omit these fields; clients must treat missing arrays as empty and missing
booleans as `false`.

`GET /api/capabilities/ltx2-control-adapters?model=<id>` returns only the
controls compatible with that host's effective installed model profile. Each
row includes `id`, `label`, guide-video `guide`, `size_bytes`, `installed`,
and the exact `download_model`, `download_repo`, `download_filename`, and
`download_sha256` identity. Dev checkpoints, unknown catalog architectures,
and unsupported profiles return `422`.

## Authentication

When `MOLD_API_KEY` is set, all API requests (except `/health`, `/api/docs`,
`/api/openapi.json`, and `/metrics`) must include an `X-Api-Key` header:

```bash
curl -H "X-Api-Key: your-secret-key" http://localhost:7680/api/status
```

Without the header (or with an invalid key), the server returns
`401 Unauthorized`:

```json
{ "error": "missing X-Api-Key header", "code": "UNAUTHORIZED" }
```

The `MOLD_API_KEY` variable supports multiple formats:

- **Single key**: `MOLD_API_KEY=my-secret`
- **Multiple keys**: `MOLD_API_KEY=key1,key2,key3`
- **File reference**: `MOLD_API_KEY=@/path/to/keys.txt` (one key per line, `#`
  comments supported)

When `MOLD_API_KEY` is unset, no authentication is required (backward
compatible).

The `mold` CLI reads `MOLD_API_KEY` from the environment and sends the header
automatically.

### Gallery media tickets

Browser and native `<video>` elements cannot attach an `X-Api-Key` header to
their own streaming and Range requests. An authenticated client can exchange
normal API-key authentication for a short-lived credential scoped to one
full-size gallery path:

```bash
curl -X POST \
  -H 'Content-Type: application/json' \
  -H 'X-Api-Key: your-secret-key' \
  -d '{"path":"/api/gallery/image/clip.mp4"}' \
  http://localhost:7680/api/gallery/media-token
```

The response is
`{"token":"...","expires_at":1234567890,"auth_required":true}`. For the
next 15 minutes, append the token as `media_token` and the Unix expiry as
`expires` to that exact gallery-image URL. It authorizes only `GET` or `HEAD`
reads, including HTTP Range requests; it cannot access another file, delete
media, or call any other endpoint. The signing secret is per server process,
the response is `no-store`, and request tracing omits the bearer query. When API
authentication is disabled, the endpoint returns `auth_required: false` and
the direct media URL needs no ticket.

### iPhone pairing tickets

An authenticated desktop or web Settings client calls
`POST /api/pairing/sessions`. Its `no-store` response includes a 256-bit random
token, two-minute Unix expiry, server instance ID, and hostname. The QR adds
the operator-confirmed reachable base URL; it does not include the API key.
Mold for iPhone posts the token to `/api/pairing/claim`, verifies that the
returned instance ID matches the scanned envelope, and stores the returned key
in the iOS Keychain. A token is removed before a successful response, so a
second or concurrent redemption fails with `PAIRING_TOKEN_INVALID`. When API
authentication is disabled, both token and returned key are `null` because the
host requires no credential.

## Rate Limiting

When `MOLD_RATE_LIMIT` is set, per-IP rate limiting is enforced with two tiers:

- **Generation tier** (configured rate): `/api/generate`,
  `/api/generate/stream`, `/api/expand`, `/api/upscale`,
  `/api/upscale/stream`, `/api/models/load`, `/api/models/pull`,
  `/api/models/unload`
- **Read tier** (10x the configured rate): `/api/models`, `/api/loras`,
  `/api/status`, `/api/gallery/*`

Health, docs, and `/metrics` endpoints are exempt from rate limiting.

Example: `MOLD_RATE_LIMIT=10/min` allows 10 generation requests per minute per
IP, and 100 read requests per minute per IP.

Supported period formats: `sec` (or `s`), `min` (or `m`), `hour` (or `h`).

Override burst size with `MOLD_RATE_LIMIT_BURST` (defaults to 2x the rate,
capped at 100).

When rate limited, the server returns `429 Too Many Requests` with a
`Retry-After` header:

```json
{ "error": "rate limit exceeded", "code": "RATE_LIMITED" }
```

## Request IDs

Every response includes an `X-Request-ID` header for correlation. If the client
sends one, it is preserved; otherwise the server generates a UUID v4.

## Quick Examples

```bash
# Generate an image
curl -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a glowing robot"}' \
  -o robot.png

# Generate with API key authentication
curl -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-secret-key" \
  -d '{"prompt": "a glowing robot"}' \
  -o robot.png

# Check status
curl http://localhost:7680/api/status

# List models
curl http://localhost:7680/api/models

# List installed LoRAs compatible with a model
curl "http://localhost:7680/api/loras?model=flux-dev:q8"

# Load a specific model
curl -X POST http://localhost:7680/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "flux-dev:q4"}'

# Upscale an image (base64 input, raw image output)
curl -X POST http://localhost:7680/api/upscale \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"real-esrgan-x4plus:fp16\",\"image\":\"$(base64 < photo.png)\"}" \
  -o photo_4x.png

# Interactive docs
open http://localhost:7680/api/docs
```

## `/api/generate`

`POST /api/generate` returns raw image bytes for `batch_size = 1`. A raw
server-owned batch (`batch_size > 1`) returns one ordered
`BatchGenerateResponse` JSON parent after its gallery transaction commits.
The server includes an `x-mold-seed-used` header with the effective seed on
singleton responses.

```bash
curl -i -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a glowing robot in a rainy alley",
    "model": "flux-schnell:q8",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance": 0.0,
    "output_format": "png"
  }' \
  -o robot.png
```

Representative headers:

```http
HTTP/1.1 200 OK
content-type: image/png
x-mold-seed-used: 42
x-mold-dimension-warning: dimensions adjusted from 1000x1000 to 1024x1024
x-mold-request-warning: dimensions adjusted from 1000x1000 to 1024x1024
```

The `x-mold-dimension-warning` header is present when the requested dimensions
were adjusted to fit model constraints (e.g. multiples of 16, pixel cap). It
carries dimension adjustments only.

The `x-mold-request-warning` header carries **every** advisory about a request
that was still accepted, `;`-separated — dimension adjustments plus anything
else, such as a lip-dub render taking its frame count and frame rate from the
reference clip instead of the values you sent. Prefer it over the dimension
header if you want to surface all of them; the streaming endpoints deliver the
same list as `info` progress events.

### Video and audio responses

A video render returns the encoded clip itself, with `x-mold-video-frames`,
`x-mold-video-fps`, `x-mold-video-width`, `x-mold-video-height`, and — when
the container carries a soundtrack — `x-mold-video-has-audio`,
`x-mold-video-duration-ms`, `x-mold-video-audio-sample-rate`, and
`x-mold-video-audio-channels`.

An audio-only render (`pipeline: "t2a"`) returns the WAV itself as
`content-type: audio/wav`, never its waveform tile:

```http
HTTP/1.1 200 OK
content-type: audio/wav
x-mold-seed-used: 42
x-mold-audio-format: wav
x-mold-audio-sample-rate: 24000
x-mold-audio-channels: 2
x-mold-audio-duration-ms: 5010
x-mold-audio-thumbnail-width: 640
x-mold-audio-thumbnail-height: 360
```

`x-mold-audio-format` states the container the server actually produced — a
request may omit `output_format` and let the server normalise an audio-only
pipeline to `wav`, so the request is not evidence of what came back.

The two `x-mold-audio-thumbnail-*` headers describe the waveform PNG the
server rendered for gallery grids — audio has no dimensions of its own, and
the tile's bytes cannot ride along in a body that is already the WAV. Probe
`x-mold-audio-sample-rate` before the video headers: an audio print has no
frames, so a video-shaped probe falls through and mislabels the response.

## Generate Request Shape

```json
{
  "prompt": "a cat on a skateboard",
  "model": "flux-schnell:q8",
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "seed": 42,
  "guidance": 0.0,
  "batch_size": 1,
  "negative_prompt": "",
  "source_image": "<base64>",
  "edit_images": ["<base64>", "<base64 reference>"],
  "strength": 0.75,
  "mask_image": "<base64>",
  "control_image": "<base64>",
  "control_model": "controlnet-canny-sd15",
  "control_scale": 1.0,
  "loras": [
    { "path": "/path/to/style.safetensors", "scale": 0.8 },
    { "path": "/path/to/detail.safetensors", "scale": 0.4 }
  ],
  "frames": 97,
  "fps": 24,
  "enable_audio": true,
  "audio_file": "<base64 wav>",
  "audio_file_path": "/srv/mold-media/voice.wav",
  "source_video": "<base64 mp4>",
  "source_video_path": "/srv/mold-media/clip.mp4",
  "keyframes": [{ "frame": 0, "image": "<base64 png>" }],
  "pipeline": "keyframe",
  "retake_range": { "start_seconds": 1.5, "end_seconds": 3.5 },
  "spatial_upscale": "x2",
  "temporal_upscale": "x2",
  "guidance_overrides": { "stg_scale": 1.5, "stg_blocks": [28, 29] },
  "placement": { "text_encoders": { "kind": "cpu" } },
  "cfg_plus": true,
  "embed_metadata": true,
  "upscale_model": "real-esrgan-x4plus:fp16",
  "expand": false,
  "output_format": "png"
}
```

`prompt` is the only field without a default, and it is required in every case
but one. All other fields have defaults or model-specific validation.

::: tip Optional prompt (LTX-2 / LTX-Video image-to-video)
An empty or whitespace-only `prompt` is accepted **only** when both of these
hold:

1. the resolved model family is `ltx2` or `ltx-video`, and
2. the request carries visual conditioning — `source_image`, a non-empty
   `keyframes[]`, `source_video` / `source_video_path`, or `extend_video` /
   `extend_video_path`.

Anything else — pure text-to-video, or any image family even with a
`source_image` — still fails with `prompt must not be empty`. For `cv:` / `hf:`
catalog IDs the family comes from the server's catalog resolution, so a
promptless request works only on a host that can resolve the model to one of
those two families.

LTX-2's Gemma encoder pads to a fixed 1,024-token context and replaces padded
positions with learned register embeddings, so `""` is a trained context rather
than a degenerate one. Two consequences worth passing on to your users: it
**does not reduce VRAM use** (the prompt context is a fixed-size tensor), and it
usually yields near-static output because nothing describes the motion. An
empty prompt also suppresses prompt expansion (`expand` is forced to `false`
rather than letting the expander invent a prompt) and is not written to prompt
history.
:::

Authoritative Scheduler V2 servers with gallery output enabled advertise
`queue.server_batch = true` and `queue.server_batch_max_outputs = 64` from
`GET /api/capabilities`. The latter is the live atomic HTTP
delivery/materialization limit, not a GPU planner limit. Requests above it
fail promptly with HTTP 422 and stable code
`BATCH_OUTPUT_LIMIT_EXCEEDED`, before model preparation, child enumeration, or
gallery filename reservation. Clients that need more outputs should submit
multiple parents or independent prepared siblings.

Important fields:

| Field                                                        | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source_image`, `mask_image`                                 | img2img/inpainting source media as base64 PNG/JPEG bytes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `edit_images`                                                | ordered Qwen-Image-Edit target/reference images; use this instead of `source_image` for `qwen-image-edit`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `control_image`, `control_model`, `control_scale`            | SD1.5 ControlNet conditioning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `lora`, `loras`                                              | singular legacy adapter or repeatable stack; `loras[]` wins when both are set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `frames`, `fps`, `output_format`                             | video/animation length and encoder selection                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `enable_audio`, `audio_file`, `audio_file_path`              | LTX-2 synchronized audio toggle and audio-to-video input. Path input is server-local and requires configured `media_roots` / `MOLD_MEDIA_ROOTS`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `source_video`, `source_video_path`, `retake_range`          | LTX-2 retake/video-conditioning source and seconds range. Path input is server-local and cannot be combined with inline base64 bytes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `extend_video`, `extend_video_path`, `extend_overlap_frames` | Continue an existing clip in one request: the last `extend_overlap_frames` frames are re-encoded as conditioning and the stitched result drops the duplicated overlap. Mutually exclusive with `source_video`, `source_image`, and `keyframes`. The overlap must sit on the family's frame grid and be strictly below `frames`; when omitted the server materializes the family default (`extend_default_overlap_frames`: 17 for LTX-2, 1 for Wan) into the request at admission so saved provenance records the real overlap. Resolution and fps must match the source clip. Path input is server-local and requires configured `media_roots` / `MOLD_MEDIA_ROOTS`. |
| `keyframes`, `pipeline`                                      | Keyframe conditioning and explicit LTX-2 pipeline selection (`one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2-vid`, `retake`, `lip-dub`, `t2a`). Wan first/last-frame interpolation (FLF-capable checkpoints) uses a two-entry `keyframes` list anchoring pixel frames 0 and F-1; any other keyframe layout is refused at admission for Wan.                                                                                                                                                                                                                                                                                        |
| `ic_lora_control`                                            | Canonical official control ID. LTX-2.0 checkpoints accept `union`, `pose`, `detailer`; LTX-2.3 accepts `union`, `motion-track`, `lipdub`, `hdr`. Every ID implies `pipeline=ic-lora` except `lipdub`, which selects the dedicated `lip-dub` pipeline. Requires source video and precedes custom `loras[]`. Use `GET /api/capabilities/ltx2-control-adapters?model=<id>` for the controls compatible with an installed model.                                                                                                                                                                                                                                         |
| `spatial_upscale`, `temporal_upscale`                        | LTX-2 latent upscaling modes such as `x1-5` and `x2`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `guidance_overrides`                                         | Additive LTX-2 multimodal-guider overrides: `stg_scale`, `stg_blocks[]`, `rescale_scale`, `modality_scale`, `skip_step`. Each omitted field keeps the pipeline default.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `scheduler`                                                  | Denoise solver. Wan accepts `uni-pc` (default), `euler` (the lightx2v 4-step Lightning recipe's solver), and `dpm-pp`; the UNet schedulers `ddim` / `euler-ancestral` are rejected for Wan and the Wan solvers are rejected for every other family.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `sample_shift`                                               | Wan flow shift, the family's primary quality/character knob. Additive; absent keeps the tier default. Precedence: request > `MOLD_WAN_SHIFT` > per-tier default. Rejected for non-Wan families and recorded in saved metadata.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `distill_strength_high`, `distill_strength_low`              | Per-expert scale for Wan A14B manifest Lightning adapters (absent = 1.0). Refused, not ignored, on tiers that ship no distill in the addressed slot.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `placement`                                                  | per-request device placement override; persisted defaults use `/api/config/model/:name/placement`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `cfg_plus`                                                   | CFG++ guidance for supported SD-family scheduler paths                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `embed_metadata`                                             | override config/env metadata embedding for this request                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `batch_id`, `batch_index`, `batch_count`                     | optional native prepared-batch identity plus one-based sibling position/total; copied unchanged into complete-event and Gallery metadata                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `source_fit`                                                 | additive, engine-ignored provenance recording the client-side source-image resize/crop policy; echoed verbatim into gallery `OutputMetadata.source_fit` so Reuse settings and running-job selection can restore the crop choice                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `upscale_model`                                              | post-generation Real-ESRGAN model applied before returning images                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

`guidance_overrides` is optional and every field inside it is optional. An
absent field keeps the engine's per-(pipeline, stage) constant, so a request
without the object reproduces pre-override outputs exactly. The server rejects
the object for non-LTX-2 families and bounds each value (`rescale_scale` is
`0..=1`; `stg_scale` and `modality_scale` are `0..=10`; `skip_step` is `0..=8`;
`stg_blocks` holds up to 8 distinct in-range indices) with HTTP 422 before any
queue work. Only pipelines that run the multimodal guider (`two-stage`,
`two-stage-hq`, `keyframe`, `a2-vid`) read the overrides, and a guider the
pipeline disables entirely stays disabled. Accepted values are recorded in
gallery metadata under the same key.

When `upscale_model` is set, the server gallery retains both artifacts as
`-original` and `-upscaled` files. The SSE `complete` event returns the
upscaled image in `image` and includes additive `original_image`,
`original_width`, and `original_height` fields so remote clients can mirror
the pair. Gallery metadata exposes the saved file size in `width` / `height`
and the reusable pre-upscale canvas in `generation_width` /
`generation_height`.

The exhaustive schema for enums and nested objects is served by the running
server at `/api/docs` and `/api/openapi.json`.

## `/api/generate/estimate`

`POST /api/generate/estimate` accepts the same JSON shape as
`/api/generate` and returns the server's current peak-memory estimate for that
request. The estimate accounts for model files, resolution, batch, frames,
placement, and runtime load strategy.

```bash
curl -X POST http://localhost:7680/api/generate/estimate \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-dev:q8","prompt":"a cat","width":1024,"height":1024}'
```

The response includes `peak_memory_bytes`, `activation_memory_bytes`,
`load_strategy`, and optional available-memory fit fields.

## `/api/models/:model/components`

`GET /api/models/:model/components` reports the component assets the server
expects for a model and whether each one is present. The Generate UI uses this
to highlight missing text encoders, VAEs, transformers, and companion files
with a path back to the model catalog.

```bash
curl "http://localhost:7680/api/models/flux-dev:q8/components"
```

## `DELETE /api/models/:model`

`DELETE /api/models/:model` removes a downloaded model — the HTTP counterpart
of `mold rm`. Several models share components (T5/CLIP/Qwen encoders, VAEs)
under the models directory, so removal ref-counts every file across all
installed models and deletes only files exclusively owned by the target;
shared components still referenced by another downloaded model are kept.
Hardlinked hf-hub cache blobs are cleaned up too, so `freed_bytes` reflects
real disk savings.

```bash
curl -X DELETE http://localhost:7680/api/models/flux-schnell:q8
```

```json
{
  "removed": ["/models/flux-schnell-q8/flux1-schnell-Q8_0.gguf"],
  "kept": [
    {
      "component": "/models/shared/flux/ae.safetensors",
      "used_by": ["flux-dev:q8"]
    }
  ],
  "freed_bytes": 12726374912
}
```

Returns `404` (`UNKNOWN_MODEL`) when the model isn't installed, and `409`
(`MODEL_LOADED`) while the model is GPU-resident — unload it first via
`DELETE /api/models/unload`. This is a destructive endpoint; pair with
`MOLD_API_KEY` when the server is exposed beyond localhost.

## `/api/config`

The HTTP counterpart of the `mold config` CLI verbs. Config values live in
two stores — `config.toml` for bootstrap/paths/credentials and the settings
DB for user preferences — with `MOLD_*` environment variables overriding
both at runtime. Every row carries a `source` tag saying which surface owns
it; rows with `source: "env"` also carry the overriding variable name.

`GET /api/config` lists every effective row (like `mold config list --json`):

```json
{
  "profile": "default",
  "entries": [
    { "key": "models_dir", "value": "~/.mold/models", "source": "file" },
    { "key": "expand.enabled", "value": false, "source": "db" },
    {
      "key": "embed_metadata",
      "value": true,
      "source": "env",
      "env_var": "MOLD_EMBED_METADATA"
    }
  ]
}
```

`GET /api/config/:key` reads one row. `PUT /api/config/:key` sets it,
routed by surface exactly like `mold config set` — DB-backed keys
(`expand.*`, generation defaults, `models.<name>.<pref>`) land in the
settings DB for the active profile, file keys rewrite `config.toml`:

```bash
curl -X PUT http://localhost:7680/api/config/default_steps \
  -H "Content-Type: application/json" -d '{"value": 12}'
```

Env-overridden keys reject writes with `403` (`ENV_OVERRIDDEN`) naming the
variable to unset; unknown keys and out-of-range values return `422`.

`DELETE /api/config/:key` resets a DB-backed key like `mold config reset`
(drops the row for the active profile) and responds with the fallback value
(`source: "default"`). File-backed keys return `422` (`FILE_BACKED_KEY`) —
edit those via `PUT` instead.

`GET /api/config/profiles` lists settings profiles and the active one;
`PUT /api/config/profile` with `{"name":"dev"}` switches the stored active
profile (a `MOLD_PROFILE` env var still wins at runtime):

```bash
curl -X PUT http://localhost:7680/api/config/profile \
  -H "Content-Type: application/json" -d '{"name":"dev"}'
```

DB-requiring operations return `503` (`CONFIG_UNAVAILABLE`) when the
metadata DB is disabled (`MOLD_DB_DISABLE=1`).

## `/api/queue`

`GET /api/queue` keeps `entries` limited to queued and running generation jobs.
Running jobs carry their actual `gpu`; queued jobs carry an
optional `target_gpu` so UI clients can render one lane per GPU plus an
automatic lane. Current authoritative V2 servers also return a nullable,
additive `plan` snapshot with versioned stable-device lanes, ordinary
generation plus scheduler-owned utility and durable-chain work items, estimated
start/finish times, confidence, blocked reasons, and the next tentative replan
deadline. Clients must treat it as advisory: the server revalidates the exact
execution fingerprint and frozen artifacts before CUDA.

Use `PATCH /api/queue/:id` to update a queued job's preferred lane and/or its
0-based position among queued jobs:

```bash
curl -X PATCH http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000 \
  -H "Content-Type: application/json" \
  -d '{"hard_pinned_device_id":"cuda:0123...","position":1}'
```

Set `target_gpu` to `null` to return the queued job to automatic placement.
`hard_pinned_device_id` accepts the opaque ID from `/api/devices`; send `null`
to return to Auto. If both ordinal and stable-ID pins are supplied, they must
name the same device.
Omitting either field leaves it unchanged. `position` is clamped to the current
queued range, so a large value sends a job to the back. Reordering changes real
dispatch priority, not only the listing returned by `GET /api/queue`.
Already-running jobs reject lane and position changes.

Use `DELETE /api/queue/:id` to cancel a queued or running singleton job:

```bash
curl -X DELETE http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000
```

Returns `204 No Content` as soon as cancellation authority is revoked and `404`
for unknown ids. Queued work is removed immediately; running work stops at the
next model safe point. Feature-detect active support through
`queue.cooperative_cancellation`. The waiting client observes a terminal
`CANCELLED` error and a streaming connection receives an `error` event before
it closes.

## `/api/history`

`GET /api/history` returns recent prompt history from the metadata DB, newest
first. `?query=` filters by case-insensitive prompt substring; `?limit=`
bounds the row count (default 50, max 500). `used_at` is Unix epoch
milliseconds.

The server records history automatically: every accepted `POST /api/generate`
or `POST /api/generate/stream` appends the typed prompt (before prompt
expansion), negative prompt, and model. Consecutive identical rows are
collapsed, so batch siblings and retries produce a single entry.

```bash
curl "http://localhost:7680/api/history?query=sunset&limit=10"
```

```json
{
  "entries": [
    {
      "prompt": "sunset over sea",
      "model": "flux-dev:q8",
      "used_at": 1700000000000
    }
  ]
}
```

`DELETE /api/history` clears the history (`204 No Content`). Pass `?keep=N`
to trim to the most recent N entries instead:

```bash
curl -X DELETE "http://localhost:7680/api/history?keep=100"
```

Both endpoints return `503` (`HISTORY_UNAVAILABLE`) when the metadata DB is
disabled (`MOLD_DB_DISABLE=1`).

## `/api/loras`

`GET /api/loras` returns installed LoRA adapters. Add `?model=<name>` to
restrict the list to the model family's compatible LoRAs. Use the returned
`path` values in `loras[].path` on `/api/generate` or `/api/generate/stream`.

```bash
curl "http://localhost:7680/api/loras?model=realistic-vision-v5:fp16"
```

## `/api/generate/stream`

The `/api/generate/stream` endpoint sends Server-Sent Events for progress:

```text
event: progress
data: {"type":"queued","position":1}

event: progress
data: {"type":"stage_start","name":"Loading model weights"}

event: progress
data: {"type":"denoise_step","step":1,"total":25,"elapsed_ms":640}

event: progress
data: {"type":"preview","image":"<base64 PNG>","step":1,"total":25}

event: complete
data: {"images":[{"data":[137,80,78,71],"format":"png","width":1024,"height":1024,"index":0}],"generation_time_ms":12345,"model":"flux-dev:q4","seed_used":42}
```

`preview` events are live latent previews for FLUX.1, Flux.2, Z-Image, and Wan (video previews project the clip's middle latent frame):
a small PNG at latent resolution (~width/8 × height/8 for most families;
Wan 2.2 TI2V's VAE compresses 16×, so ~width/16) produced by a linear
latent→RGB projection — no VAE involved, so the cost per step is negligible.
Emitted at most every ~700 ms plus always on the final step; clients upscale
and blur it. Disable with `MOLD_STEP_PREVIEW=0` on the server.

Typical terminal usage:

```bash
curl -N http://localhost:7680/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a glowing robot",
    "model": "flux-dev:q4",
    "steps": 25,
    "width": 1024,
    "height": 1024
  }'
```

For `batch_size = 1`, the final `complete` event matches the
`GenerateResponse` JSON shape used by the server internally. A server-owned
batch emits one ordered `batch_complete` event after durable commit and uses
the same advertised 64-output live limit.

::: tip RunPod Note
RunPod's proxy has a 100-second timeout. Use the SSE streaming endpoint for long generations to keep the connection alive.
:::

## `/api/events`

`GET /api/events` is a single SSE stream of **server-wide** lifecycle events:
generation jobs, gallery changes, versioned queue replans, and semantic device
lifecycle/health transitions. Raw utilization and memory telemetry remain on
`GET /api/resources/stream`. Frames use the event name `event` with an
internally tagged JSON payload:

```text
event: event
data: {"type":"job_queued","id":"6f9c…","model":"flux-dev:q4"}

event: event
data: {"type":"job_started","id":"6f9c…","model":"flux-dev:q4","gpu":0}

event: event
data: {"type":"gallery_added","filename":"mold-flux-dev-q4-1752300000000.png","image":{"filename":"…","metadata":{…},"timestamp":1752300000,"format":"png","size_bytes":1830421}}

event: event
data: {"type":"job_ended","id":"6f9c…"}

event: event
data: {"type":"gallery_removed","filename":"mold-flux-dev-q4-1752300000000.png"}

event: event
data: {"type":"chain_job_queued","id":"550e8400-…","model":"ltx-2-19b-distilled:fp8","stage_count":3}

event: event
data: {"type":"chain_job_started","id":"550e8400-…","model":"ltx-2-19b-distilled:fp8"}

event: event
data: {"type":"chain_job_ended","id":"550e8400-…","state":"completed"}
```

Event semantics:

| `type`                 | Meaning                                                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `job_queued`           | A generation was accepted into the queue (`id`, `model`).                                                                                                                                         |
| `job_started`          | A worker began the job. `gpu` is the ordinal on multi-GPU servers, omitted on single-GPU.                                                                                                         |
| `job_ended`            | The job left the queue for **any** reason — completed, errored, or cancelled. Use the per-job stream for outcomes; `gallery_added` is the durable success signal.                                 |
| `gallery_added`        | A new output landed on disk. `image` carries the full gallery row when the metadata DB recorded it (insert it directly); when the DB is disabled `image` is omitted — refetch `GET /api/gallery`. |
| `gallery_removed`      | An output was deleted via `DELETE /api/gallery/image/:name`.                                                                                                                                      |
| `queue_plan_changed`   | The scheduler published a newer versioned queue plan. Replace tentative lanes only when `plan_version` advances.                                                                                  |
| `device_state_changed` | Device administration, worker health, or activity changed. Treat the payload as a hint and refetch `GET /api/devices`; telemetry-only samples do not emit this event.                             |
| `chain_job_queued`     | A durable chain job entered the queue — created, resumed, retaken, or amended. Carries `id`, `model`, and `stage_count`.                                                                          |
| `chain_job_started`    | The chain runner claimed the job and began rendering stages (`id`, `model`).                                                                                                                      |
| `chain_job_ended`      | The job settled. `state` is `completed`, `failed`, or `cancelled`. Terminal chain jobs stay listed on `/api/chain-jobs` — this only says the runner is done with it.                              |

The three `chain_job_*` events are additive and deliberately distinct from
`job_queued` / `job_started` / `job_ended`: chain jobs do not support the
print-queue affordances (`PATCH`/`DELETE /api/queue/:id`), and older clients
ignore unknown `type` tags. The ephemeral jobs backing
`/api/generate/chain` stay silent — only durable `/api/chain-jobs` work is
announced. Clients that render sequences in a unified activity surface can
use these instead of polling `GET /api/chain-jobs`.

The stream carries **deltas only** — there is no initial snapshot. Subscribe
first, then bootstrap current state from `GET /api/queue`, `GET /api/devices`,
and `GET /api/gallery`. Refetch those authoritative snapshots after every
reconnect because lagged broadcast frames are intentionally not replayed.
Feature-detect with
`GET /api/capabilities` (`"events": {"available": true}`); servers older than
this endpoint omit the field. Keep-alive pings arrive every 15 s.

```bash
curl -N http://localhost:7680/api/events
```

## `/api/generate/chain`

Chained video generation for the LTX-2, LTX-Video, and Wan families, including
installed catalog checkpoints with opaque `cv:` / `hf:` IDs. Splits a long
video into N per-clip renders and returns a single stitched MP4. The seam is
family- and checkpoint-specific: LTX-2 threads a motion tail of latents across
each boundary (default 17 frames), Wan continues via last-frame image
conditioning on image-conditioned checkpoints (the overlap is always 1 frame;
text-to-video checkpoints concatenate independent clips), and LTX-Video joins
independently rendered clips. See the
[LTX-2 chained video output guide](/models/ltx2#chained-video-output) for the
user-facing story; this section documents the wire format.

The request body maps to `mold_core::chain::ChainRequest`; the response body
maps to `mold_core::chain::ChainResponse`. The canonical schema lives in the
interactive docs at `/api/docs` (served by the running mold server) and in the
OpenAPI JSON at `/api/openapi.json`.

This legacy endpoint now executes through the durable chain-job runner
internally. The response shape stays the same, while the backing ephemeral job
is cleaned up after a successful response is assembled.

The server accepts either a pre-authored `stages[]` body or the auto-expand
form (single `prompt` + `total_frames` + `clip_frames`). Auto-expand is the
shape `mold run` sends; the canonical `stages[]` shape is reserved for the
forthcoming movie-maker UI that will author per-stage prompts/keyframes. Both
normalise to the same internal `Vec<ChainStage>` before any engine work kicks
off.

Both forms also accept optional `original_prompt`, `batch_id`, `batch_index`,
and `batch_count` provenance. These fields survive normalization and durable
resume and are copied into the stitched output's completion and Gallery
metadata.

**Auto-expand body** (what `mold run --frames N` emits):

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "prompt": "a cat walking through autumn leaves",
  "total_frames": 400,
  "clip_frames": 97,
  "source_image": "<base64 PNG>",
  "motion_tail_frames": 4,
  "width": 1216,
  "height": 704,
  "fps": 24,
  "seed": 42,
  "steps": 8,
  "guidance": 3.0,
  "strength": 1.0,
  "output_format": "mp4"
}
```

**Canonical body** (what the v2 movie-maker UI will author):

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "stages": [
    { "prompt": "a cat walking", "frames": 97, "source_image": "<base64 PNG>" },
    { "prompt": "a cat walking", "frames": 97 },
    { "prompt": "a cat walking", "frames": 97 },
    { "prompt": "a cat walking", "frames": 97 }
  ],
  "motion_tail_frames": 4,
  "width": 1216,
  "height": 704,
  "fps": 24,
  "seed": 42,
  "steps": 8,
  "guidance": 3.0,
  "strength": 1.0,
  "output_format": "mp4"
}
```

**Response:**

```json
{
  "video": {
    "data": "<base64 mp4>",
    "format": "mp4",
    "width": 1216,
    "height": 704,
    "frames": 400,
    "fps": 24,
    "thumbnail": "<base64 png>",
    "gif_preview": "<base64 gif>",
    "has_audio": false,
    "duration_ms": 16666
  },
  "stage_count": 5,
  "gpu": 0
}
```

**Error cases:**

- `422 Unprocessable Entity` — validation failure (missing `prompt` +
  `total_frames` in the auto-expand form, a stage whose `frames` is off the
  family's advertised grid (`k · frame_step + 1` — 8 for the LTX families, 4
  for Wan), `motion_tail_frames >= clip_frames`, more than 16 stages, etc.).
- `422 Unprocessable Entity` — unsupported model family. Only LTX-2,
  LTX-Video, and Wan expose a chain renderer; other families are rejected with
  an error that names the constraint.
- `502 Bad Gateway` — the backing job failed before a legacy `ChainResponse`
  could be assembled. Use `/api/chain-jobs` for explicit durable
  resume/retake workflows.

::: tip Runner behaviour
The legacy chain endpoints are shims over the durable runner. The runner
checkpoints each stage under `MOLD_HOME/jobs/<job_id>`, yields at stage
boundaries when other work is waiting, then deletes successful ephemeral shim
artifacts after building the legacy response. The public chain-job API keeps
artifacts for resume and retake.
:::

## `/api/generate/chain/validate`

Accepts the same `ChainRequest` body as `/api/generate/chain`, but performs
only build-feature, model-family, and structural normalization. It does not
create a durable job, start a download, lease a device, or touch inference.
The response reports normalized stage transitions, each stage's contributed
output frames, source/negative-prompt presence, warnings, and the optional
`vram_estimate` field:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "width": 1216,
  "height": 704,
  "fps": 24,
  "motion_tail_frames": 17,
  "stage_count": 2,
  "estimated_total_frames": 177,
  "estimated_duration_ms": 7375,
  "stages": [
    {
      "prompt": "a cat enters the forest",
      "frames": 97,
      "output_frames": 97,
      "transition": "smooth",
      "has_source_image": true,
      "has_negative_prompt": false
    },
    {
      "prompt": "the forest opens to a clearing",
      "frames": 97,
      "output_frames": 80,
      "transition": "smooth",
      "has_source_image": false,
      "has_negative_prompt": false
    }
  ],
  "warnings": [],
  "vram_estimate": null
}
```

Media and negative-prompt contents are not echoed. HTTP `422` uses the normal
structured `VALIDATION_ERROR` response. `vram_estimate`, when present, reports
`worst_case_bytes` (the max over stages, never their sum — stages run
serially) and an advisory `fits` verdict computed against stable device
capacity. It is `null` when the server cannot price the run (model not
downloaded, or no device sample); it is advisory and never gates submission.

## `/api/generate/chain/stream`

Same request body as `/api/generate/chain`, with the response delivered as
Server-Sent Events. Progress frames stream as `event: progress` and the
terminal frame is either `event: complete` (success) or `event: error`
(failure; the connection closes after the error frame).

Progress event payloads map to `mold_core::chain::ChainProgressEvent` variants:

```text
event: progress
data: {"type":"chain_start","job_id":"550e8400-e29b-41d4-a716-446655440000","stage_count":5,"estimated_total_frames":485}

event: progress
data: {"type":"stage_start","job_id":"550e8400-e29b-41d4-a716-446655440000","stage_idx":0}

event: progress
data: {"type":"denoise_step","job_id":"550e8400-e29b-41d4-a716-446655440000","stage_idx":0,"step":1,"total":8}

event: progress
data: {"type":"stage_done","job_id":"550e8400-e29b-41d4-a716-446655440000","stage_idx":0,"frames_emitted":97}

event: progress
data: {"type":"stitching","job_id":"550e8400-e29b-41d4-a716-446655440000","total_frames":385}

event: complete
data: {"video":"<base64 mp4>","format":"mp4","width":1216,"height":704,"frames":400,"fps":24,"thumbnail":"<base64 png>","gif_preview":"<base64 gif>","has_audio":false,"duration_ms":16666,"stage_count":5,"gpu":0,"generation_time_ms":226812}
```

The `complete` event payload maps to `mold_core::chain::SseChainCompleteEvent`.
Non-denoise engine events (weight loads, cache hits, etc.) are intentionally
not forwarded in v1 — the UX goal is per-stage progress, not per-component
telemetry.

`job_id` is an additive field on progress events so clients can correlate a
legacy stream with the backing durable job. The terminal `complete` payload
keeps the legacy shape.

```bash
curl -N -X POST http://localhost:7680/api/generate/chain/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ltx-2-19b-distilled:fp8",
    "prompt": "a cat walking through autumn leaves",
    "total_frames": 400,
    "clip_frames": 97,
    "motion_tail_frames": 4,
    "width": 1216, "height": 704, "fps": 24,
    "steps": 8, "guidance": 3.0,
    "output_format": "mp4"
  }'
```

## `/api/chain-jobs`

Durable async chain jobs persist the request, per-stage state, retakes, and
final outputs under `MOLD_HOME/jobs/<job_id>` and mirror query state in
`mold.db`. They use the same `mold_core::chain::ChainRequest` body as
`/api/generate/chain`, but return immediately with `202 Accepted`:

```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

Endpoints:

- `POST /api/chain-jobs` — create a queued job.
- `GET /api/chain-jobs` — list summaries, newest first.
- `GET /api/chain-jobs/:id` — detail including stages, retakes, finalizes, and effective script.
- `GET /api/chain-jobs/:id/events` — SSE stream; first frame is always a snapshot.
- `POST /api/chain-jobs/:id/resume` — requeue `interrupted`, `failed`, or `cancelled`.
- `POST /api/chain-jobs/:id/retake` — body is `RetakeRequest` (`stage_idx`, `mode`, optional `seed_offset`, optional `prompt`).
- `POST /api/chain-jobs/:id/amend` — replace the whole stage list in place, reusing cached clips. See below.
- `POST /api/chain-jobs/:id/cancel` — queued jobs settle as `cancelled`; an accepted running cancellation returns `202`, exposes `summary.cancelling: true`, and cannot publish a completed stage/job after that barrier.
- `DELETE /api/chain-jobs/:id` — remove a non-running job and its job directory.
- `POST /api/chain-jobs/gc` — explicitly sweep eligible ephemeral jobs and discard completed durable stage caches while retaining final outputs and job metadata. Automatic maintenance retains durable caches.
- `GET /api/chain-jobs/:id/stages/:idx/preview` — returns `image/jpeg` when that stage has a preview.
- `GET` or `HEAD /api/chain-jobs/:id/stages/:idx/media` — streams a completed raw stage MP4, including byte-range `206` and unsatisfiable-range `416` behavior.
- `POST /api/chain-jobs/:id/stages/:idx/media-token` — mints a short-lived ticket restricted to the exact corresponding media path.

Common errors: `503 CHAIN_JOBS_UNAVAILABLE` when the metadata DB is disabled,
`404 CHAIN_JOB_NOT_FOUND`, and `409 CHAIN_JOB_RUNNING` for mutations that
cannot safely run while the job is active.

### `POST /api/chain-jobs/:id/amend`

Edits a settled or queued sequence in place instead of creating a new job, so
clips that did not change are never re-rendered. This is what the Studio
surfaces call behind **Update sequence**.

The body maps to `mold_core::chain_job::AmendRequest`. `stages` is the
**complete** edited stage list in canonical order (not a patch) — the same
`ChainStage` shape `/api/generate/chain` accepts. Everything else is an
optional chain-level overlay applied over the job's current effective
request:

```json
{
  "stages": [
    { "prompt": "a cat walks into the autumn forest", "frames": 97 },
    { "prompt": "the forest opens to a clearing", "frames": 49 },
    {
      "prompt": "a spaceship lands",
      "frames": 97,
      "transition": "fade",
      "fade_frames": 12
    }
  ],
  "motion_tail_frames": 8,
  "fps": 24,
  "seed": "42",
  "steps": 8,
  "guidance": 3.0,
  "enable_audio": false
}
```

`seed` is a full-range `u64` encoded as a decimal **string**, matching the
rest of the chain wire format. Omitted overlays keep the job's current value.

**Not amendable.** `AmendRequest` carries no other fields, so `model`,
`width`, `height`, `output_format`, GPU `placement`, `strength`, and the
`batch_id` / `batch_index` / `batch_count` provenance are inherited from the
original request and cannot be changed — create a fresh job for those. The
amended candidate must still pass every create-time gate: `normalise()`,
the family/audio check, and the durable-job `output_format = "mp4"` rule.

**Response** — `202 Accepted`. The body is the updated `ChainJobSummary`
flattened, plus `preserved_stages`:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "queued",
  "model": "ltx-2-19b-distilled:fp8",
  "stage_count": 3,
  "current_stage": 2,
  "created_at_unix_ms": 1752300000000,
  "updated_at_unix_ms": 1752300600000,
  "error": null,
  "ephemeral": false,
  "preserved_stages": 2
}
```

`preserved_stages` is the count of leading stages whose cached artifacts were
kept; rendering requeues from that index. The job is requeued and a
`chain_job_queued` event is published on `/api/events`.

**Invalidation semantics.** The preserved prefix is the longest run of leading
stages whose render identity is unchanged, clamped to the leading run of
already-completed stages. A chain-level change to `seed`, `steps`, `guidance`,
`fps`, or `motion_tail_frames`, or turning `enable_audio` **on**, invalidates
everything (turning it off preserves every clip — finalize simply ignores the
audio sidecars). Otherwise a stage is dirty when its `prompt`, `frames`,
`negative_prompt`, `source_image`, effective per-stage seed, or its
_smooth-carry_ status changes — where carry means "not the first stage and
`transition == "smooth"`". Because stage artifacts are stored as **raw,
untrimmed** segments with every boundary trim and crossfade applied at
finalize time, `cut` ↔ `fade` toggles and `fade_frames` edits break no prefix
at all and re-finalize with zero re-renders; `smooth` ↔ (`cut`|`fade`) changes
the rendered pixels and does break it. Appending clips renders only the new
ones, and removing trailing clips renders nothing. Jobs written by older
versions carry baked-in boundaries, so a preserved legacy stage is dropped
from the prefix when its artifacts cannot serve the new boundary plan.

Each amend is appended to the manifest with the pre-amend **effective**
request (retakes folded in) and its `preserved_stages`; any pending retakes
are folded and cleared. `GET /api/chain-jobs/:id` exposes the additive
`amends` array (`at_unix_ms`, `previous_request_json`, `preserved_stages`).

**Errors:**

- `409 CHAIN_JOB_RUNNING` — the job is rendering; cancel it first.
- `409 CHAIN_JOB_EPHEMERAL` — the job backs a legacy `/api/generate/chain` shim.
- `409 CHAIN_JOB_NOT_AMENDABLE` — the job left an amendable state mid-request. Amendable states are `queued`, `interrupted`, `failed`, `cancelled`, and `completed`.
- `422` — the amended request failed validation (bad frame counts, motion tail ≥ clip frames, too many stages, a non-`mp4` output format, an unsupported family, audio on a checkpoint without an audio path).
- `404 CHAIN_JOB_NOT_FOUND` — unknown id.

## `/api/models`

`GET /api/models` lists every known model. Each row is a flattened
`ModelInfoExtended`: identity (`name`, `family`, `hf_repo`, …), the
`ModelDefaults` block (`default_steps`, `default_guidance`, `default_width`,
`default_height`, `description`, …), and installation state, all at the top
level of the object — the defaults are not nested under a `defaults` key.
Video models additionally advertise their frame semantics there, so clients
stop hardcoding a frame count that ignores the selected checkpoint:

| Field                           | Meaning                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default_frames`                | Default frame count for one clip. LTX-2 defaults to 97, LTX-Video to its shipped 25.                                                                                                                                                                                                                                                                              |
| `default_fps`                   | Default frames per second.                                                                                                                                                                                                                                                                                                                                        |
| `max_frames`                    | Ceiling for a single request **at `default_fps`** — 484 for LTX-2 at 24 fps, 257 for LTX-Video.                                                                                                                                                                                                                                                                   |
| `supports_extend`               | Whether this model can continue an existing video in one request. Every LTX-2 checkpoint can; Wan answers per checkpoint, from the same `source_image` contract its seam reads. Absent on servers that predate continuation — read absence as "no".                                                                                                               |
| `extend_default_overlap_frames` | Overlap applied when a continuation omits `extend_overlap_frames`. Per family, because it is the family's carryover: 17 on LTX-2, 1 on Wan.                                                                                                                                                                                                                       |
| `max_runtime_seconds`           | Present when the family's real ceiling is a duration rather than a frame count (LTX-2: 20). Recompute `max_frames` at another fps as `max_runtime_seconds · fps + 4`.                                                                                                                                                                                             |
| `max_frames_absolute`           | fps-independent frame guard paired with `max_runtime_seconds` (LTX-2: 604).                                                                                                                                                                                                                                                                                       |
| `frame_step`                    | Valid frame counts are `k · frame_step + 1`; 8 for the LTX families, 4 for Wan.                                                                                                                                                                                                                                                                                   |
| `source_image`                  | Per-checkpoint image-conditioning contract: `"unsupported"` (an attached image is rejected at admission), `"optional"`, or `"required"` (the checkpoint cannot generate without one). Omitted = unknown; clients must treat absence as unknown, not as a heuristic license. Derived from manifest task structure or the installed checkpoint's own tensor shapes. |
| `dimension_alignment`           | Pixel grid both dimensions must sit on. Most Wan checkpoints use 16; `wan22-ti2v-5b` advertises 32 (its 16× VAE stride times the 2×2 DiT patch). Off-grid canvases are rejected at admission on the generate and chain routes.                                                                                                                                    |

Models with a tuned default negative prompt (wan today) additionally
advertise `default_negative_prompt`: the negative the engine applies when a
request omits `negative_prompt` entirely. An explicit `""` in a request stays
a real empty uncond — clients prefill the advertised value, keep an untouched
field absent, and send `""` to opt out.

::: tip LTX-2's ceiling is a duration, not a frame count
The LTX-2 checkpoints ship `pos_embed_max_pos = 20`, and the temporal RoPE axis
is normalized in **seconds** (the pixel-frame coordinate is divided by fps
before `max_pos` normalization). So the budget is 20 seconds of runtime — 484
frames at 24 fps, but only 124 at 6 fps. Clients that let the user change fps
must recompute `max_frames` from `max_runtime_seconds`; treating the advertised
scalar as fixed will be wrong in both directions.

`--temporal-upscale x2` does **not** extend this budget: it halves the stage-1
frame count _and_ the stage-1 fps, so stage 1 renders the same runtime at half
the frame rate.
:::

All of these are additive and omitted entirely on image models — clients must
treat their absence as "not a video model" rather than substituting a
constant. They come from the same manifest defaults and validator constants
the server enforces, and the server-side validator stays authoritative.

`GET /api/capabilities/chain-limits?model=<name>&fps=<n>` reports
`frames_per_clip_recommended` from the same per-model default, and its
`frames_per_clip_cap` is the model's own clip size — the number of frames one
generation renders when a long request is chained automatically (97 for
LTX-2; for Wan the checkpoint's own manifest default over a 53-frame A14B /
121-frame floor, e.g. 121 for TI2V-5B),
bounded above by the family's single-request ceiling at `fps` when that is
smaller. Every sequence picker locks its per-clip choices to this value. The
response echoes the `fps` it was computed at and, for families with a duration
budget, `frames_per_clip_runtime_seconds`. Chain admission itself enforces the
family's single-request ceiling, which is what an explicit CLI `--clip-frames`
may reach. Pass `fps` when the user is editing that control so the advertised
cap matches what the server will hold the request to.

## `/api/status`

Example response:

```json
{
  "version": "0.10.0",
  "git_sha": "da039e1",
  "build_date": "2026-05-24",
  "models_loaded": ["flux-schnell:q8", "ltx-2-19b-distilled:fp8"],
  "busy": true,
  "gpu_info": null,
  "gpus": [
    {
      "ordinal": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_bytes": 25757220864,
      "vram_used_bytes": 12918456320,
      "loaded_model": "flux-schnell:q8",
      "state": "idle"
    },
    {
      "ordinal": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_bytes": 25757220864,
      "vram_used_bytes": 21474836480,
      "loaded_model": "ltx-2-19b-distilled:fp8",
      "state": "generating"
    }
  ],
  "queue_depth": 1,
  "queue_capacity": 200,
  "uptime_secs": 3600,
  "hostname": "gpu-box"
}
```

Older single-GPU clients can still read `gpu_info`; multi-GPU-aware clients
should prefer `gpus[]`, `queue_depth`, and `queue_capacity`.

`GET /api/resources` and `GET /api/resources/stream` expose only the GPU
inventory that CUDA made visible when the server started. CUDA builds sample
NVML by the raw CUDA/NVIDIA UUID rather than a physical ordinal, so numeric
`CUDA_VISIBLE_DEVICES` reordering and `GPU-...` selectors preserve the correct
process-local ordinal and hidden physical cards are not published. The
`nvidia-smi` fallback applies the same UUID filter and converts its MiB values
to binary bytes. A MIG worker accepts only telemetry carrying its matching
`MIG-...` UUID; Mold does not substitute the parent GPU's full-memory sample.
When the installed NVML adapter cannot prove a MIG parent UUID or profile,
`mig_parent_uuid` and `mig_profile` remain `null`.

## `/api/devices`

`GET /api/devices` is the stable multi-device resource. Device `id` values are
opaque and must be URL-encoded rather than parsed; CUDA ordinals are
process-local display hints. Operational values that the active sampler cannot
provide are JSON `null`, not zero.

```json
{
  "devices": [
    {
      "id": "cuda:0123456789abcdef0123456789abcdef",
      "backend": "cuda",
      "ordinal": 0,
      "device_kind": "full_gpu",
      "nvml_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
      "physical_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
      "mig_uuid": null,
      "mig_parent_uuid": null,
      "mig_profile": null,
      "name": "NVIDIA GeForce RTX 3090",
      "pci_bus_id": "00000000:01:00.0",
      "compute_capability": "8.6",
      "memory": {
        "total_bytes": 25769803776,
        "used_bytes": 8589934592,
        "mold_used_bytes": null,
        "other_used_bytes": null
      },
      "telemetry": {
        "utilization_percent": 41,
        "temperature_c": null,
        "power_w": null
      },
      "desired_enabled": true,
      "restart_required": false,
      "admin_state": "enabled",
      "health": "healthy",
      "activity": "idle",
      "schedulable": true,
      "unschedulable_reason": null,
      "loaded_models": [],
      "active_work_id": null,
      "planned_work_ids": []
    }
  ],
  "plan_version": 0
}
```

`plan_version` remains `0` until the versioned scheduler plan is active.
Desired enablement is machine-wide; a newly seen device with no explicit
preference defaults to enabled.

`PATCH /api/devices/{url-encoded-stable-id}` accepts
`{"enabled":false}` or `{"enabled":true}`. Disabling removes the device from
new scheduling immediately and returns `202` while an active lease drains.
Re-enabling starts a fresh owner lifetime and returns `202` with
`admin_state:"starting"` without waiting for CUDA context creation. The first
epoch-qualified ready event changes it to enabled. If context creation fails,
the desired preference remains enabled, health is unavailable, and
`unschedulable_reason` reports `device_start_failed: ...`; retry the PATCH or
restart after correcting the driver/device fault. A delayed ready, stopped, or
completion event from the predecessor cannot mutate or reap the replacement.

Runtime mutation requires scheduler V2. In legacy, observe, or maintenance
mode, disabling still returns `409`, but enabling a persistently-disabled,
startup-selected device records the preference for the next boot. The first
such PATCH returns `202` with `restart_required:true`; repeated identical
PATCHes return `200`, and subsequent device polls retain that flag until a
restart creates the owner. Startup-excluded devices cannot use this recovery
path and return `409`.

## `/api/models/pull`

Plain blocking response:

```bash
curl -X POST http://localhost:7680/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell:q8"}'
```

Example text response:

```text
model 'flux-schnell:q8' pulled successfully
```

SSE streaming response:

```bash
curl -N http://localhost:7680/api/models/pull \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell:q8"}'
```

Representative events:

```text
event: progress
data: {"type":"download_progress","filename":"flux1-schnell-Q8_0.gguf","file_index":1,"total_files":6,"bytes_downloaded":1048576,"bytes_total":12714452256}

event: progress
data: {"type":"pull_complete","model":"flux-schnell:q8"}
```

## `/api/expand`

Expand a short prompt into one or more detailed generation prompts using the
configured expansion LLM.

```bash
curl -X POST http://localhost:7680/api/expand \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat",
    "model_family": "flux",
    "variations": 3,
    "style": "gritty film noir"
  }'
```

**Request fields:**

| Field          | Type   | Required | Description                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prompt`       | string | yes      | Short prompt to expand                                                                                                                                                                                                                                                                                                                                                                                 |
| `model_family` | string | no       | Model family for prompt style (`flux` default; `sdxl`, `sd15`, `sd3`, ...)                                                                                                                                                                                                                                                                                                                             |
| `variations`   | number | no       | Number of prompt variations to generate (default 1, max 10)                                                                                                                                                                                                                                                                                                                                            |
| `style`        | string | no       | Visual style to absorb into the expansion (e.g. a style preset label). Passed to the LLM as a natural-language directive, never a literal suffix.                                                                                                                                                                                                                                                      |
| `task`         | string | no       | Resolved generation/conditioning task, additive: `text-to-image` (default), `text-to-video`, `image-to-video`, `video-to-video`, `retake`, `keyframe-interpolation`, `audio-driven-video`, `reference-to-audio-video`, `text-to-audio`. When omitted, the server infers text-to-video for known video families and text-to-image otherwise. Carries only the semantic task — never source media bytes. |

**Response:**

```json
{
  "original": "a cat",
  "expanded": ["a sleek black cat prowling a rain-slicked alley, ..."]
}
```

## `/api/remix`

Remix is separate from Expand so an older host returns `404` instead of
silently applying expansion semantics. The server preserves the source subject
and explicit constraints, assigns each result a deterministic creative
dimension, and rejects dimensions that contradict image/video/audio
conditioning authority.

```bash
curl -X POST http://localhost:7680/api/remix \
  -H 'Content-Type: application/json' \
  -d '{
    "source_prompt": "a red lighthouse with exactly three windows",
    "root_prompt": "a red lighthouse",
    "source_kind": "current",
    "model_family": "flux",
    "variations": 3,
    "dimensions": ["camera", "lighting"]
  }'
```

Dimensions are `composition`, `camera`, `lighting`, `setting`, `mood`,
`movement`, and `style`. When `style` is supplied it is a locked constraint and
cannot also appear in `dimensions`. An omitted dimension list uses task-aware
defaults. The response returns `variants[]` with both `prompt` and the exact
`dimensions` label used for that alternative. Remix accepts the same additive
`task` field as `/api/expand`, with the same semantics.

## `/api/upscale`

Upscale an image using Real-ESRGAN super-resolution models.

```bash
curl -X POST http://localhost:7680/api/upscale \
  -H "Content-Type: application/json" \
  -d '{
    "model": "real-esrgan-x4plus:fp16",
    "image": "<base64-encoded PNG or JPEG>",
    "output_format": "png",
    "tile_size": 512
  }' \
  --output upscaled.png
```

**Request fields:**

| Field           | Type   | Required | Description                                               |
| --------------- | ------ | -------- | --------------------------------------------------------- |
| `model`         | string | yes      | Upscaler model name (e.g. `real-esrgan-x4plus:fp16`)      |
| `image`         | string | yes      | Base64-encoded input image (PNG or JPEG)                  |
| `output_format` | string | no       | `png` (default) or `jpeg`                                 |
| `metadata`      | object | no       | Generation metadata to embed in the upscaled output       |
| `tile_size`     | number | no       | Tile size for memory-efficient processing (0 = no tiling) |

**Response:** Raw image bytes (PNG or JPEG) with `Content-Type` header.

## `/api/upscale/stream`

Same request format as `/api/upscale`, but returns SSE events for tile-by-tile progress:

```bash
curl -N -X POST http://localhost:7680/api/upscale/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "real-esrgan-x4plus:fp16",
    "image": "<base64-encoded PNG or JPEG>"
  }'
```

Representative events (tile progress reuses the `denoise_step` event type):

```text
event: progress
data: {"type":"denoise_step","step":1,"total":9,"elapsed_ms":1200}

event: complete
data: {"image":"<base64>","model":"real-esrgan-x4plus:fp16","scale_factor":4,"width":2048,"height":2048}
```

The server caches the upscaler engine between requests — repeated upscales with the same model skip weight loading.

## Saved output metadata

Gallery rows (`GET /api/gallery`, the `image.metadata` object on
`gallery_added`, and the embedded `mold:parameters` chunk) map to
`mold_core::OutputMetadata`. The request's engine-ignored `source_fit`
provenance, when sent, is echoed verbatim here as `source_fit`. Two additive
fields record sequence provenance:

- `chain_job_id` — the durable chain job this output was finalized from.
  Absent for single generations, the ephemeral `/api/generate/chain` shim, and
  legacy rows.
- `chain` — structured per-clip provenance, so a sequence is never recorded
  under clip 1's prompt alone. Absent for single generations and legacy rows.

```json
{
  "chain_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "chain": {
    "stage_count": 3,
    "motion_tail_frames": 8,
    "stages": [
      {
        "prompt": "a cat walks into the autumn forest",
        "frames": 97,
        "transition": "smooth",
        "seed": "42"
      },
      {
        "prompt": "the forest opens to a clearing",
        "frames": 49,
        "transition": "cut",
        "seed": "43"
      },
      {
        "prompt": "a spaceship lands",
        "frames": 97,
        "transition": "fade",
        "fade_frames": 12,
        "seed": "44"
      }
    ]
  }
}
```

Each stage carries `prompt`, `frames`, and `transition`; `fade_frames` and the
effective per-stage `seed` (a full-range `u64` as a decimal string) appear only
when known. The top-level `prompt` of a multi-clip row holds every distinct
clip prompt joined one per line, so gallery search matches any clip. The CLI's
local chain saves write the same `chain` block without a `chain_job_id`.

Studio clients read this block to reload a finished sequence's clips onto the
Create clip rail (**Reuse settings**); `chain_job_id` is what lets **Edit
sequence** re-enter the original durable job with its cached clips.

## Image Output

Generated images are saved to `~/.mold/output/` by default. Override with a
custom path:

```bash
MOLD_OUTPUT_DIR=/srv/mold/output mold serve
```

To disable image persistence (TUI gallery will not function):

```bash
MOLD_OUTPUT_DIR="" mold serve
```
