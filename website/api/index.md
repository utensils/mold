# Server API

When running `mold serve`, you get a REST API for remote image generation.

## Endpoints

| Method   | Path                   | Description                          |
| -------- | ---------------------- | ------------------------------------ |
| `POST`   | `/api/generate`        | Generate images from prompt          |
| `POST`   | `/api/generate/stream` | Generate with SSE progress streaming |
| `POST`   | `/api/expand`          | Expand a prompt using LLM            |
| `GET`    | `/api/models`          | List available models                |
| `POST`   | `/api/models/load`     | Load/swap the active model           |
| `POST`   | `/api/models/pull`     | Pull/download a model                |
| `DELETE` | `/api/models/unload`   | Unload model to free GPU memory      |
| `GET`    | `/api/status`          | Server health + status               |
| `GET`    | `/health`              | Simple 200 OK health check           |
| `GET`    | `/api/openapi.json`    | OpenAPI spec                         |
| `GET`    | `/api/docs`            | Interactive API docs (Scalar)        |

## Authentication

When `MOLD_API_KEY` is set, all API requests (except `/health`, `/api/docs`, and `/api/openapi.json`) must include an `X-Api-Key` header:

```bash
curl -H "X-Api-Key: your-secret-key" http://localhost:7680/api/status
```

Without the header (or with an invalid key), the server returns `401 Unauthorized`:

```json
{"error": "missing X-Api-Key header", "code": "UNAUTHORIZED"}
```

The `MOLD_API_KEY` variable supports multiple formats:
- **Single key**: `MOLD_API_KEY=my-secret`
- **Multiple keys**: `MOLD_API_KEY=key1,key2,key3`
- **File reference**: `MOLD_API_KEY=@/path/to/keys.txt` (one key per line, `#` comments supported)

When `MOLD_API_KEY` is unset, no authentication is required (backward compatible).

The `mold` CLI reads `MOLD_API_KEY` from the environment and sends the header automatically.

## Rate Limiting

When `MOLD_RATE_LIMIT` is set, per-IP rate limiting is enforced with two tiers:

- **Generation tier** (configured rate): `/api/generate`, `/api/generate/stream`, `/api/expand`, `/api/models/load`, `/api/models/pull`, `/api/models/unload`
- **Read tier** (10x the configured rate): `/api/models`, `/api/status`, `/api/gallery/*`

Health and docs endpoints are exempt from rate limiting.

Example: `MOLD_RATE_LIMIT=10/min` allows 10 generation requests per minute per IP, and 100 read requests per minute per IP.

Supported period formats: `sec` (or `s`), `min` (or `m`), `hour` (or `h`).

Override burst size with `MOLD_RATE_LIMIT_BURST` (defaults to 2x the rate, capped at 100).

When rate limited, the server returns `429 Too Many Requests` with a `Retry-After` header:

```json
{"error": "rate limit exceeded", "code": "RATE_LIMITED"}
```

## Request IDs

Every response includes an `X-Request-ID` header for correlation. If the client sends one, it is preserved; otherwise the server generates a UUID v4.

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

# Load a specific model
curl -X POST http://localhost:7680/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "flux-dev:q4"}'

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
```

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
  "batch": 1,
  "negative_prompt": "",
  "image": "<base64>",
  "strength": 0.75,
  "mask": "<base64>",
  "lora": "/path/to/adapter.safetensors",
  "lora_scale": 1.0,
  "expand": false
}
```

Only `prompt` is required. All other fields have defaults.

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
  -d '{"prompt":"a glowing robot","model":"flux-dev:q4","steps":25,"width":1024,"height":1024}'
```

The final `complete` event matches the `GenerateResponse` JSON shape used by the
server internally.

<!-- prettier-ignore-start -->
::: tip RunPod Note
RunPod's proxy has a 100-second timeout. Use the SSE streaming endpoint for long generations to keep the connection alive.
:::
<!-- prettier-ignore-end -->

## `/api/status`

Example response:

```json
{
  "version": "0.3.1",
  "git_sha": "da039e1",
  "build_date": "2026-03-25",
  "models_loaded": ["flux-schnell:q8"],
  "busy": false,
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 4090",
    "vram_total_mb": 24564,
    "vram_used_mb": 8192
  },
  "uptime_secs": 3600
}
```

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
