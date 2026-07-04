# Server API

When running `mold serve`, you get a REST API for remote image generation.

## Endpoints

| Method   | Path                                      | Description                                                                                                       |
| -------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `POST`   | `/api/generate`                           | Generate images from prompt                                                                                       |
| `POST`   | `/api/generate/stream`                    | Generate with SSE progress streaming                                                                              |
| `POST`   | `/api/generate/estimate`                  | Estimate request-sensitive peak memory for a generation request                                                   |
| `POST`   | `/api/generate/chain`                     | Chained video generation (LTX-2)                                                                                  |
| `POST`   | `/api/generate/chain/stream`              | Chained video with SSE progress                                                                                   |
| `POST`   | `/api/chain-jobs`                         | Create a durable async chain job                                                                                  |
| `GET`    | `/api/chain-jobs`                         | List durable chain jobs                                                                                           |
| `GET`    | `/api/chain-jobs/:id`                     | Get durable chain-job detail                                                                                      |
| `GET`    | `/api/chain-jobs/:id/events`              | Durable chain-job SSE events                                                                                      |
| `POST`   | `/api/chain-jobs/:id/resume`              | Resume a failed, interrupted, or cancelled chain job                                                              |
| `POST`   | `/api/chain-jobs/:id/retake`              | Retake one chain-job stage                                                                                        |
| `POST`   | `/api/chain-jobs/:id/cancel`              | Cancel a queued or running chain job                                                                              |
| `DELETE` | `/api/chain-jobs/:id`                     | Delete a non-running chain job                                                                                    |
| `POST`   | `/api/chain-jobs/gc`                      | Run chain-job artifact GC                                                                                         |
| `GET`    | `/api/chain-jobs/:id/stages/:idx/preview` | Fetch a stage preview JPEG                                                                                        |
| `POST`   | `/api/expand`                             | Expand a prompt using LLM                                                                                         |
| `GET`    | `/api/models`                             | List available models                                                                                             |
| `GET`    | `/api/models/:model/components`           | List required model component readiness and paths                                                                 |
| `GET`    | `/api/loras`                              | List installed LoRAs, optionally filtered by `?model=` compatibility                                              |
| `POST`   | `/api/models/load`                        | Load/swap the active model                                                                                        |
| `POST`   | `/api/models/pull`                        | Pull/download a model                                                                                             |
| `DELETE` | `/api/models/unload`                      | Unload model to free GPU memory                                                                                   |
| `GET`    | `/api/gallery`                            | List saved images                                                                                                 |
| `GET`    | `/api/gallery/image/:name`                | Fetch a saved image                                                                                               |
| `DELETE` | `/api/gallery/image/:name`                | Delete a saved image                                                                                              |
| `GET`    | `/api/gallery/thumbnail/:name`            | Fetch a cached thumbnail                                                                                          |
| `GET`    | `/api/gallery/preview/:name`              | Fetch a cached GIF preview for video gallery rows                                                                 |
| `GET`    | `/api/downloads`                          | List active, queued, failed, and completed downloads                                                              |
| `POST`   | `/api/downloads`                          | Queue a manifest model download                                                                                   |
| `DELETE` | `/api/downloads/:id`                      | Cancel a queued or active download                                                                                |
| `GET`    | `/api/downloads/stream`                   | Download queue updates as SSE                                                                                     |
| `GET`    | `/api/catalog/families`                   | Live catalog family/kind metadata                                                                                 |
| `GET`    | `/api/catalog/search`                     | Search the live HF/Civitai catalog                                                                                |
| `GET`    | `/api/catalog/installed`                  | List installed catalog entries and LoRAs                                                                          |
| `GET`    | `/api/catalog/:id`                        | Resolve one `hf:` or `cv:` catalog entry                                                                          |
| `POST`   | `/api/catalog/:id/download`               | Queue a catalog entry plus missing companions                                                                     |
| `POST`   | `/api/upscale`                            | Upscale image with Real-ESRGAN                                                                                    |
| `POST`   | `/api/upscale/stream`                     | Upscale with SSE tile progress                                                                                    |
| `GET`    | `/api/resources`                          | Latest RAM/GPU resource snapshot                                                                                  |
| `GET`    | `/api/resources/stream`                   | Resource snapshots as SSE                                                                                         |
| `GET`    | `/api/queue`                              | Server-authoritative job listing (queued + running, UUIDv4 ids); used by the SPA to reconcile dropped SSE streams |
| `PATCH`  | `/api/queue/:id`                          | Update the preferred GPU lane for a queued job                                                                    |
| `GET`    | `/api/capabilities`                       | Feature capabilities (gallery delete, chain limits, …)                                                            |
| `GET`    | `/api/capabilities/chain-limits`          | Chain-generation request limits                                                                                   |
| `PUT`    | `/api/config/model/:name/placement`       | Save model-specific device placement defaults                                                                     |
| `DELETE` | `/api/config/model/:name/placement`       | Clear model-specific device placement defaults                                                                    |
| `POST`   | `/api/shutdown`                           | Trigger graceful server shutdown                                                                                  |
| `GET`    | `/api/status`                             | Server health + status                                                                                            |
| `GET`    | `/health`                                 | Simple 200 OK health check                                                                                        |
| `GET`    | `/api/openapi.json`                       | OpenAPI spec                                                                                                      |
| `GET`    | `/api/docs`                               | Interactive API docs (Scalar)                                                                                     |
| `GET`    | `/metrics`                                | Prometheus metrics (feature-gated)                                                                                |

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

`POST /api/generate` returns raw image bytes, not a JSON envelope. The response
`Content-Type` matches the requested format, and the server includes an
`x-mold-seed-used` header with the effective seed.

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
```

The `x-mold-dimension-warning` header is present when the requested dimensions
were adjusted to fit model constraints (e.g. multiples of 16, pixel cap).

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
  "placement": { "text_encoders": { "kind": "cpu" } },
  "cfg_plus": true,
  "embed_metadata": true,
  "upscale_model": "real-esrgan-x4plus:fp16",
  "expand": false,
  "output_format": "png"
}
```

Only `prompt` is required. All other fields have defaults or model-specific
validation.

Important fields:

| Field                                               | Purpose                                                                                                                                          |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `source_image`, `mask_image`                        | img2img/inpainting source media as base64 PNG/JPEG bytes                                                                                         |
| `edit_images`                                       | ordered Qwen-Image-Edit target/reference images; use this instead of `source_image` for `qwen-image-edit`                                        |
| `control_image`, `control_model`, `control_scale`   | SD1.5 ControlNet conditioning                                                                                                                    |
| `lora`, `loras`                                     | singular legacy adapter or repeatable stack; `loras[]` wins when both are set                                                                    |
| `frames`, `fps`, `output_format`                    | video/animation length and encoder selection                                                                                                     |
| `enable_audio`, `audio_file`, `audio_file_path`     | LTX-2 synchronized audio toggle and audio-to-video input. Path input is server-local and requires configured `media_roots` / `MOLD_MEDIA_ROOTS`. |
| `source_video`, `source_video_path`, `retake_range` | LTX-2 retake/video-conditioning source and seconds range. Path input is server-local and cannot be combined with inline base64 bytes.            |
| `keyframes`, `pipeline`                             | LTX-2 keyframe and explicit pipeline selection (`one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2vid`, `retake`) |
| `spatial_upscale`, `temporal_upscale`               | LTX-2 latent upscaling modes such as `x1-5` and `x2`                                                                                             |
| `placement`                                         | per-request device placement override; persisted defaults use `/api/config/model/:name/placement`                                                |
| `cfg_plus`                                          | CFG++ guidance for supported SD-family scheduler paths                                                                                           |
| `embed_metadata`                                    | override config/env metadata embedding for this request                                                                                          |
| `upscale_model`                                     | post-generation Real-ESRGAN model applied before returning images                                                                                |

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

## `/api/queue`

`GET /api/queue` returns queued and running generation jobs. Running jobs carry
their actual `gpu`; queued jobs carry an optional `target_gpu` so UI clients
can render one lane per GPU plus an automatic lane.

Use `PATCH /api/queue/:id` to update a queued job's preferred lane:

```bash
curl -X PATCH http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000 \
  -H "Content-Type: application/json" \
  -d '{"target_gpu":0}'
```

Set `target_gpu` to `null` to return the queued job to automatic placement.
Already-running jobs reject lane changes.

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

event: complete
data: {"images":[{"data":[137,80,78,71],"format":"png","width":1024,"height":1024,"index":0}],"generation_time_ms":12345,"model":"flux-dev:q4","seed_used":42}
```

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

The final `complete` event matches the `GenerateResponse` JSON shape used by the
server internally.

::: tip RunPod Note
RunPod's proxy has a 100-second timeout. Use the SSE streaming endpoint for long generations to keep the connection alive.
:::

## `/api/generate/chain`

Chained video generation for LTX-2 distilled models. Splits a long video into
N per-clip renders, threads a motion-tail of latents across each clip
boundary, and returns a single stitched MP4. See the
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
  `total_frames` in the auto-expand form, a stage with non-`8k+1` `frames`,
  `motion_tail_frames >= clip_frames`, more than 16 stages, etc.).
- `422 Unprocessable Entity` — unsupported model family. Only LTX-2 distilled
  engines expose a chain renderer; other families are rejected with an
  error that names the constraint.
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
- `POST /api/chain-jobs/:id/cancel` — queued jobs settle as `cancelled`; running jobs stop at the next boundary/progress check.
- `DELETE /api/chain-jobs/:id` — remove a non-running job and its job directory.
- `POST /api/chain-jobs/gc` — prune successful ephemeral jobs and completed non-ephemeral job artifacts older than `chain.jobs_artifact_ttl_days`.
- `GET /api/chain-jobs/:id/stages/:idx/preview` — returns `image/jpeg` when that stage has a preview.

Common errors: `503 CHAIN_JOBS_UNAVAILABLE` when the metadata DB is disabled,
`404 CHAIN_JOB_NOT_FOUND`, and `409 CHAIN_JOB_RUNNING` for mutations that
cannot safely run while the job is active.

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
