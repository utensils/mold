# mold — Architecture & Development Guide

> Like ollama, but for diffusion models.

mold is a CLI/TUI tool for AI image generation using FLUX models via the [candle](https://github.com/huggingface/candle) ML framework. It provides a local inference server that runs on GPU hosts and a client CLI that can generate images locally or by connecting to a remote server.

## Project Vision

- **Local-first**: Run FLUX models directly on your GPU
- **Remote-capable**: Point `MOLD_HOST` at a GPU server and generate from anywhere
- **Model management**: Pull, list, load/unload models (like `ollama pull`)
- **Interactive TUI**: Real-time image generation session with `mold run`
- **Future**: OCI registry for model distribution (like ollama's docker registry format)

## Crate Structure

```
mold/
├── Cargo.toml                    # Workspace root
├── scripts/
│   ├── deploy.sh                 # Build + deploy to hal9000
│   └── fetch-tokenizers.sh       # Download tokenizer files
├── crates/
│   ├── mold-core/                # Shared types, API protocol, HTTP client, config
│   ├── mold-inference/           # Candle-based FLUX inference engine
│   ├── mold-server/              # Axum HTTP inference server (lib + binary)
│   └── mold-cli/                 # Main binary — CLI (clap) + TUI (ratatui)
```

### mold-core

Shared library used by all other crates. Contains:

- **`types.rs`** — API request/response types (`GenerateRequest`, `GenerateResponse`, `ModelInfo`, `ServerStatus`, etc.)
- **`client.rs`** — `MoldClient` HTTP client for communicating with mold-server
- **`config.rs`** — `Config` struct, `ModelConfig`, `ModelPaths`; loads from `~/.mold/config.toml`
- **`error.rs`** — `MoldError` enum with thiserror

Key types:

```rust
GenerateRequest     // prompt, model, dimensions, steps, seed, batch_size, format
GenerateResponse    // images (Vec<ImageData>), generation_time_ms, model, seed_used
ImageData           // data (bytes), format, width, height, index
OutputFormat        // Png | Jpeg
ModelInfo           // name, family, size_gb, is_loaded, last_used, hf_repo
ServerStatus        // version, models_loaded, gpu_info, uptime_secs
GpuInfo             // name, vram_total_mb, vram_used_mb
LoadModelRequest    // model name
ModelConfig         // per-model path overrides (transformer, vae, t5_encoder, clip_encoder)
ModelPaths          // resolved PathBufs for all model components
```

### mold-inference

FLUX diffusion model inference using candle. Structure:

```
src/
├── lib.rs
├── engine.rs           # InferenceEngine trait + FluxEngine implementation
├── error.rs            # InferenceError enum
├── model_registry.rs   # Known models → HF repo mapping
└── flux/
    └── mod.rs           # Module docs (pipeline uses candle_transformers directly)
```

The `InferenceEngine` trait:
```rust
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
}
```

**`FluxEngine`** implements real FLUX.1 inference using candle-transformers:

1. **T5 encoding** — `candle_transformers::models::t5::T5EncoderModel` encodes prompt to 4096-dim embeddings (padded to 256 tokens)
2. **CLIP encoding** — `candle_transformers::models::clip::text_model::ClipTextTransformer` encodes prompt to 768-dim embeddings
3. **Noise generation** — `flux::sampling::get_noise()` creates initial latent noise
4. **State construction** — `flux::sampling::State::new()` packages text + image embeddings
5. **Timestep scheduling** — `flux::sampling::get_schedule()` (4 steps for schnell, configurable for dev with shift)
6. **Denoising** — `flux::sampling::denoise()` runs the FLUX transformer with Euler sampling
7. **Unpacking** — `flux::sampling::unpack()` converts packed latents to spatial layout
8. **VAE decoding** — `flux::autoencoder::AutoEncoder::decode()` converts latents to pixels
9. **Image encoding** — Tensor → RGB → PNG/JPEG bytes via the `image` crate

Model loading is **lazy** (on first generation request) and uses **mmap** for safetensors files.

Feature flags: `cuda` (CUDA backend), `metal` (Metal backend).

### mold-server

Axum-based HTTP server that wraps the inference engine. Runs on the GPU host. Has both a library crate and a standalone binary (`mold-server`).

Routes:
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate images from prompt |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/status` | Server health + status |
| `GET` | `/health` | Simple 200 OK health check |

State is managed via `AppState` which holds a `tokio::sync::Mutex<FluxEngine>`. Model is loaded lazily on first `/api/generate` request.

### mold-cli

Main binary crate. Provides both CLI commands and an interactive TUI.

## CLI Command Reference

```
mold generate [OPTIONS] <PROMPT>
    -m, --model <MODEL>         Model to use [default: flux-schnell]
    -o, --output <PATH>         Output file path [default: ./mold-output-{timestamp}.png]
        --width <N>             Image width [default: 1024]
        --height <N>            Image height [default: 1024]
        --steps <N>             Inference steps [default: 4 (schnell), 25 (dev)]
        --seed <N>              Random seed [random if not set]
        --batch <N>             Number of images [default: 1]
        --host <URL>            Override MOLD_HOST env var
        --format <FORMAT>       png or jpeg [default: png]

mold serve [OPTIONS]
        --port <N>              Server port [default: 7680]
        --bind <ADDR>           Bind address [default: 0.0.0.0]
        --models-dir <PATH>     Override MOLD_MODELS_DIR

mold pull <MODEL>               Download model weights from HuggingFace
mold list                       List locally available models
mold ps                         Show server status + loaded models
mold run [MODEL]                Interactive TUI session [default: flux-schnell]
mold version                    Show version information
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLD_HOST` | `http://localhost:7680` | Remote server URL for client commands |
| `MOLD_MODELS_DIR` | `~/.mold/models` | Model storage directory |
| `MOLD_PORT` | `7680` | Server port (used by `mold serve`) |
| `MOLD_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `MOLD_TRANSFORMER_PATH` | — | Path to FLUX transformer safetensors |
| `MOLD_VAE_PATH` | — | Path to VAE safetensors |
| `MOLD_T5_PATH` | — | Path to T5-XXL encoder safetensors |
| `MOLD_CLIP_PATH` | — | Path to CLIP-L encoder safetensors |
| `MOLD_T5_TOKENIZER_PATH` | — | Path to T5 tokenizer.json |
| `MOLD_CLIP_TOKENIZER_PATH` | — | Path to CLIP tokenizer.json |

## Config File

Location: `~/.mold/config.toml`

```toml
default_model = "flux-dev"
models_dir = "~/.mold/models"
server_port = 7680
output_dir = "."
default_width = 1024
default_height = 1024

[models.flux-dev]
transformer = "/path/to/flux1-dev.safetensors"
vae = "/path/to/ae.safetensors"
t5_encoder = "/path/to/t5xxl_fp16.safetensors"
clip_encoder = "/path/to/clip_l.safetensors"
```

Loaded via `Config::load_or_default()` — falls back to defaults if the file doesn't exist. Model paths can also be set via env vars (see above).

## Remote Rendering

mold supports a client-server architecture for remote GPU rendering:

1. On the GPU host: `mold-server serve --port 7680`
2. On any client: `MOLD_HOST=http://gpu-host:7680 mold generate "a sunset"`

The client sends a `GenerateRequest` via HTTP POST to `/api/generate` and receives the generated image bytes in the response. All CLI commands (`generate`, `list`, `ps`) communicate through the same HTTP API.

## Deployment to hal9000

Model files on hal9000 (already downloaded):
```
Transformer:  /home/jamesbrink/AI/models/unet/flux1-dev.safetensors
VAE:          /home/jamesbrink/AI/models/vae/ae.safetensors
T5 encoder:   /home/jamesbrink/AI/models/text_encoders/t5xxl_fp16.safetensors
CLIP-L:       /home/jamesbrink/AI/models/clip/clip_l.safetensors
```

### First-time setup

```bash
# On hal9000: download tokenizer files
ssh jamesbrink@hal9000.home.urandom.io
bash /home/jamesbrink/mold/scripts/fetch-tokenizers.sh
```

### Deploy

```bash
# From the mold project root on your local machine:
./scripts/deploy.sh
```

This will:
1. rsync source to hal9000
2. Build with CUDA on hal9000
3. Stop any running mold-server
4. Start mold-server with model paths configured via env vars

### Test after deploy

```bash
MOLD_HOST=http://hal9000.home.urandom.io:7680 mold generate "a rusty robot on a beach"
```

## Known Models

| Name | HuggingFace Repo | Size |
|------|-----------------|------|
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | ~23.8 GB |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | ~23.8 GB |

Model registry is defined in `mold-inference/src/model_registry.rs`.

## TUI (mold run)

Interactive terminal UI built with ratatui:
- Header showing current model
- Status/output area
- Prompt input at bottom
- Keybindings: Enter=generate, Esc/q=quit, type to enter prompt

## Future: OCI Registry for Models

Planned support for distributing models via OCI-compatible registries (similar to how ollama uses Docker registry format). This would allow:
- `mold pull registry.example.com/flux-schnell:latest`
- Private model registries
- Versioned model artifacts with layers for components (text encoder, VAE, transformer)

## Development

### Prerequisites

- Rust stable (1.75+)
- For GPU inference: CUDA toolkit (candle CUDA backend)

### Building

```bash
cargo build                     # Debug build (all crates)
cargo build --release           # Release build
cargo build -p mold-cli         # Just the CLI binary
cargo build -p mold-server --features cuda  # Server with CUDA
```

### Running

```bash
cargo run -p mold-cli -- generate "a cat"
cargo run -p mold-cli -- serve
cargo run -p mold-cli -- list
cargo run -p mold-cli -- run
```

### Checking

```bash
cargo check                     # Type check
cargo clippy                    # Lint
cargo fmt --check               # Format check
```

### Testing

```bash
cargo test                      # Run all tests
cargo test -p mold-core         # Test specific crate
```

## Key Design Decisions

1. **Workspace structure**: Separating core types, inference, server, and CLI into distinct crates allows independent compilation and clean dependency boundaries. The CLI doesn't need candle, and the server doesn't need clap/ratatui.

2. **candle over tch/ort**: candle is pure Rust, provides first-class FLUX support, and doesn't require libtorch system dependencies. It supports CUDA, Metal, and CPU backends.

3. **Axum for the server**: Modern, ergonomic, and built on tokio/tower. Natural fit for the async Rust ecosystem.

4. **Lazy model loading**: Model components (transformer, T5, CLIP, VAE) are loaded on first request rather than at startup, so the server starts fast and only uses VRAM when needed.

5. **`InferenceEngine` trait**: Allows swapping backends (e.g., candle CUDA vs CPU, or future ONNX backend) without changing server/CLI code.

6. **`tokio::sync::Mutex` for engine state**: Async-aware mutex prevents blocking the tokio runtime during generation. Single-model-at-a-time design is appropriate for GPU workloads.

7. **mmap for safetensors**: Model weights are memory-mapped rather than read into memory, allowing the OS to manage paging efficiently.

8. **Env var + config file model paths**: Supports both deployment-friendly env vars and local config file paths. Env vars take precedence over config.
