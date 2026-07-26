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
| `POST`   | `/api/expand`                             | Expand a prompt using LLM, optionally absorbing a visual `style` directive                                        |
| `GET`    | `/api/models`                             | List available models                                                                                             |
| `GET`    | `/api/models/:model/components`           | List required model component readiness and paths                                                                 |
| `GET`    | `/api/loras`                              | List installed LoRAs, optionally filtered by `?model=` compatibility                                              |
| `POST`   | `/api/models/load`                        | Load/swap the active model                                                                                        |
| `POST`   | `/api/models/pull`                        | Pull/download a model                                                                                             |
| `DELETE` | `/api/models/unload`                      | Unload model to free GPU memory                                                                                   |
| `DELETE` | `/api/models/:model`                      | Remove a downloaded model (keeps components shared with other models)                                             |
| `GET`    | `/api/gallery`                            | List saved images                                                                                                 |
| `POST`   | `/api/gallery/media-token`                | Mint a short-lived, read-only ticket for one full-size gallery path                                               |
| `GET`    | `/api/gallery/image/:name`                | Fetch a saved image                                                                                               |
| `DELETE` | `/api/gallery/image/:name`                | Delete a saved image                                                                                              |
| `GET`    | `/api/gallery/thumbnail/:name`            | Fetch a cached thumbnail                                                                                          |
| `GET`    | `/api/gallery/preview/:name`              | Fetch a cached GIF preview for video gallery rows                                                                 |
| `GET`    | `/api/downloads`                          | List up to two active downloads (`active_jobs`), plus queued, failed, and completed jobs                          |
| `POST`   | `/api/downloads`                          | Queue a manifest model download                                                                                   |
| `DELETE` | `/api/downloads/:id`                      | Cancel a queued or active download                                                                                |
| `GET`    | `/api/downloads/stream`                   | Download queue updates as SSE                                                                                     |
| `GET`    | `/api/catalog/families`                   | Live catalog family/kind metadata                                                                                 |
| `GET`    | `/api/catalog/search`                     | Search the live HF/Civitai catalog; sort by downloads, recent additions, or rating                                |
| `GET`    | `/api/catalog/installed`                  | List installed catalog entries and LoRAs                                                                          |
| `GET`    | `/api/catalog/credentials`                | Return masked status for server-owned HF/Civitai credentials                                                      |
| `PUT`    | `/api/catalog/credentials/:provider`      | Save an `hf` or `civitai` credential on the serving host                                                          |
| `DELETE` | `/api/catalog/credentials/:provider`      | Remove a saved host credential and fall back to its environment-provided default                                  |
| `GET`    | `/api/catalog/:id`                        | Resolve one `hf:` or `cv:` catalog entry                                                                          |
| `POST`   | `/api/catalog/:id/download`               | Queue a catalog entry plus missing companions                                                                     |
| `POST`   | `/api/upscale`                            | Upscale image with Real-ESRGAN                                                                                    |
| `POST`   | `/api/upscale/stream`                     | Upscale with SSE tile progress                                                                                    |
| `GET`    | `/api/resources`                          | Latest RAM/GPU resource snapshot                                                                                  |
| `GET`    | `/api/resources/stream`                   | Resource snapshots as SSE                                                                                         |
| `GET`    | `/api/events`                             | Server-wide lifecycle events (job + gallery) as SSE                                                               |
| `GET`    | `/api/queue`                              | Server-authoritative job listing (queued + running, UUIDv4 ids); used by the SPA to reconcile dropped SSE streams |
| `PATCH`  | `/api/queue/:id`                          | Update the preferred GPU lane and/or dispatch position for a queued job                                           |
| `DELETE` | `/api/queue/:id`                          | Cancel a still-queued generation job                                                                              |
| `GET`    | `/api/history`                            | Prompt history, newest first (`?query=` substring filter, `?limit=` up to 500)                                    |
| `DELETE` | `/api/history`                            | Clear prompt history (`?keep=N` trims to the most recent N)                                                       |
| `GET`    | `/api/capabilities`                       | Feature capabilities, including optional per-host expansion and LAN-discovery state                               |
| `GET`    | `/api/discovery/peers`                    | DNS-SD peers visible on the serving host's LAN (when `discovery.can_browse`)                                      |
| `GET`    | `/api/capabilities/chain-limits`          | Chain-generation request limits                                                                                   |
| `GET`    | `/api/config`                             | List every effective config row with its source (`db`/`file`/`env`)                                               |
| `GET`    | `/api/config/:key`                        | Read one config key (value + owning source)                                                                       |
| `PUT`    | `/api/config/:key`                        | Set a config key, routed by surface like `mold config set`                                                        |
| `DELETE` | `/api/config/:key`                        | Reset a DB-backed key like `mold config reset`                                                                    |
| `GET`    | `/api/config/profiles`                    | List settings profiles and the active one                                                                         |
| `PUT`    | `/api/config/profile`                     | Switch the active settings profile                                                                                |
| `PUT`    | `/api/config/model/:name/placement`       | Save model-specific device placement defaults                                                                     |
| `DELETE` | `/api/config/model/:name/placement`       | Clear model-specific device placement defaults                                                                    |
| `POST`   | `/api/shutdown`                           | Trigger graceful server shutdown                                                                                  |
| `GET`    | `/api/status`                             | Server health + status                                                                                            |
| `GET`    | `/health`                                 | Simple 200 OK health check                                                                                        |
| `GET`    | `/api/openapi.json`                       | OpenAPI spec                                                                                                      |
| `GET`    | `/api/docs`                               | Interactive API docs (Scalar)                                                                                     |
| `GET`    | `/metrics`                                | Prometheus metrics (feature-gated)                                                                                |

### Capability discovery

`GET /api/capabilities` is additive and safe to feature-detect. Current
servers advertise the accepted catalog sort values in
`catalog.sort` (`downloads`, `recent`, `rating`), queue controls as
`queue.can_pause`, `queue.can_cancel_all`, and `queue.can_reorder`, and
server-assisted DNS-SD as `discovery.can_browse`. Clients must only request
`GET /api/discovery/peers` when that discovery flag is true. Older servers may
omit these fields; clients must treat missing arrays as empty and missing
booleans as `false`.

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
| `batch_id`, `batch_index`, `batch_count`            | optional native prepared-batch identity plus one-based sibling position/total; copied unchanged into complete-event and Gallery metadata         |
| `upscale_model`                                     | post-generation Real-ESRGAN model applied before returning images                                                                                |

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

`GET /api/queue` returns queued and running generation jobs. Running jobs carry
their actual `gpu`; queued jobs carry an optional `target_gpu` so UI clients
can render one lane per GPU plus an automatic lane.

Use `PATCH /api/queue/:id` to update a queued job's preferred lane and/or its
0-based position among queued jobs:

```bash
curl -X PATCH http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000 \
  -H "Content-Type: application/json" \
  -d '{"target_gpu":0,"position":1}'
```

Set `target_gpu` to `null` to return the queued job to automatic placement.
Omitting either field leaves it unchanged. `position` is clamped to the current
queued range, so a large value sends a job to the back. Reordering changes real
dispatch priority, not only the listing returned by `GET /api/queue`.
Already-running jobs reject lane and position changes.

Use `DELETE /api/queue/:id` to cancel a job that is still queued:

```bash
curl -X DELETE http://localhost:7680/api/queue/00000000-0000-0000-0000-000000000000
```

Returns `204 No Content` on success, `404` for unknown ids, and `409`
(`QUEUE_JOB_RUNNING`) once a GPU worker owns the job — only queued jobs are
cancelable. The waiting client observes the cancellation immediately: a
blocking `POST /api/generate` resolves with a `499` `CANCELLED` error, and a
`POST /api/generate/stream` connection receives a terminal `error` event and
closes.

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

`preview` events are live latent previews for FLUX.1, Flux.2, and Z-Image:
a small PNG at latent resolution (~width/8 × height/8) produced by a linear
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

The final `complete` event matches the `GenerateResponse` JSON shape used by the
server internally.

::: tip RunPod Note
RunPod's proxy has a 100-second timeout. Use the SSE streaming endpoint for long generations to keep the connection alive.
:::

## `/api/events`

`GET /api/events` is a single SSE stream of **server-wide** lifecycle events —
every generation job's queued/started/ended transitions plus gallery
additions and removals — so a client can keep its gallery and queue views
live over one connection instead of holding a stream per job. Frames use the
event name `event` with an internally tagged JSON payload:

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
```

Event semantics:

| `type`            | Meaning                                                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `job_queued`      | A generation was accepted into the queue (`id`, `model`).                                                                                                                                         |
| `job_started`     | A worker began the job. `gpu` is the ordinal on multi-GPU servers, omitted on single-GPU.                                                                                                         |
| `job_ended`       | The job left the queue for **any** reason — completed, errored, or cancelled. Use the per-job stream for outcomes; `gallery_added` is the durable success signal.                                 |
| `gallery_added`   | A new output landed on disk. `image` carries the full gallery row when the metadata DB recorded it (insert it directly); when the DB is disabled `image` is omitted — refetch `GET /api/gallery`. |
| `gallery_removed` | An output was deleted via `DELETE /api/gallery/image/:name`.                                                                                                                                      |

The stream carries **deltas only** — there is no initial snapshot. Subscribe
first, then bootstrap current state from `GET /api/queue` and
`GET /api/gallery` so nothing lands in the gap. Feature-detect with
`GET /api/capabilities` (`"events": {"available": true}`); servers older than
this endpoint omit the field. Keep-alive pings arrive every 15 s.

```bash
curl -N http://localhost:7680/api/events
```

## `/api/generate/chain`

Chained video generation for one-stage and distilled LTX-2 models, including
installed catalog checkpoints with opaque `cv:` / `hf:` IDs. Splits a long
video into N per-clip renders, threads a motion-tail of latents across each clip
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

| Field          | Type   | Required | Description                                                                                                                                       |
| -------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | string | yes      | Short prompt to expand                                                                                                                            |
| `model_family` | string | no       | Model family for prompt style (`flux` default; `sdxl`, `sd15`, `sd3`, ...)                                                                        |
| `variations`   | number | no       | Number of prompt variations to generate (default 1, max 10)                                                                                       |
| `style`        | string | no       | Visual style to absorb into the expansion (e.g. a style preset label). Passed to the LLM as a natural-language directive, never a literal suffix. |

**Response:**

```json
{
  "original": "a cat",
  "expanded": ["a sleek black cat prowling a rain-slicked alley, ..."]
}
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
