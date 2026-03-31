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

## Quick Examples

```bash
# Generate an image
curl -X POST http://localhost:7680/api/generate \
  -H "Content-Type: application/json" \
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

## Generate Request

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

## SSE Streaming

The `/api/generate/stream` endpoint sends Server-Sent Events for progress:

```
event: progress
data: {"step": 1, "total": 25, "percent": 4.0}

event: progress
data: {"step": 25, "total": 25, "percent": 100.0}

event: complete
data: {"images": ["<base64 PNG>"]}
```

::: tip RunPod Note RunPod's proxy has a 100-second timeout. Use the SSE
streaming endpoint for long generations to keep the connection alive. :::

## Server Image Persistence

Save copies of all server-generated images:

```bash
MOLD_OUTPUT_DIR=/srv/mold/gallery mold serve
```
