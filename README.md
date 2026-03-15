# mold 🧪

Like ollama, but for diffusion models. Local FLUX image generation — runs on your GPU, works standalone or as a server.

## What it does

- Loads FLUX.1 models (schnell / dev) from local GGUF or safetensors files
- **Just works**: `mold run "a cat"` — no server needed (auto-pulls model, runs locally)
- **Remote capable**: Point at a GPU server with `MOLD_HOST` or run `mold serve` for a REST API
- CLI: `mold run "a glowing robot"`

## Requirements

- NVIDIA GPU with CUDA **or** Apple Silicon with Metal
- FLUX model files (see below)

## Quick start

```bash
# Build (CUDA on Linux, Metal on macOS)
cargo build --release -p mold-cli --features cuda    # Linux
cargo build --release -p mold-cli --features metal   # macOS

# Set model paths
export MOLD_TRANSFORMER_PATH=/path/to/flux1-schnell-Q8_0.gguf
export MOLD_VAE_PATH=/path/to/ae.safetensors
export MOLD_T5_PATH=/path/to/t5xxl_fp16.safetensors
export MOLD_CLIP_PATH=/path/to/clip_l.safetensors

# Generate an image (no server needed — auto-pulls model, runs locally)
./target/release/mold run "a cat riding a motorcycle through neon-lit streets"

# Use a specific model
./target/release/mold run flux-dev:q4 "a turtle in the desert"

# Or start a server for remote rendering
./target/release/mold serve
MOLD_HOST=http://gpu-host:7680 mold run "a sunset"

# Force local inference (skip server check)
./target/release/mold run --local "a glowing robot"
```

## Model files

Download from HuggingFace:

| File | Source |
|------|--------|
| `flux1-schnell-Q8_0.gguf` | [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) |
| `ae.safetensors` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| `t5xxl_fp16.safetensors` | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |
| `clip_l.safetensors` | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |

## API

```
GET  /health           → 200 OK
GET  /api/status       → ServerStatus JSON
GET  /api/models       → ModelInfo[] JSON
POST /api/generate     → image/png bytes
```

### Generate request

```json
{
  "prompt": "a glowing robot",
  "model": "flux-schnell",
  "width": 768,
  "height": 768,
  "steps": 4,
  "seed": null,
  "batch_size": 1,
  "output_format": "png"
}
```

Constraints: width/height must be multiples of 16, max 1024, min 1. Steps 1–100.

## Config

`~/.mold/config.toml` (optional):

```toml
default_model = "flux-schnell"
server_port = 7680
default_width = 768
default_height = 768

[models.flux-schnell]
transformer = "/models/flux1-schnell-Q8_0.gguf"
vae = "/models/ae.safetensors"
t5_encoder = "/models/t5xxl_fp16.safetensors"
clip_encoder = "/models/clip_l.safetensors"
```

## Shell completions

```bash
# bash
source <(mold completions bash)

# zsh
source <(mold completions zsh)

# fish
mold completions fish | source
```

## Architecture

```
mold-cli       Single binary: CLI + serve (mold run / serve / pull / completions)
mold-server    axum REST server (library, used by mold-cli via `mold serve`)
mold-inference FLUX engine (candle: T5/CLIP on CPU, transformer+VAE on GPU)
mold-core      Shared types, config, HTTP client
```

T5-XXL (9.2GB) runs on CPU to fit FLUX transformer + VAE on a single 24GB GPU.
