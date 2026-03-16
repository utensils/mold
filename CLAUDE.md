# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# mold — Architecture & Development Guide

> Local AI image generation CLI — FLUX, SDXL & Z-Image diffusion models on your GPU.

mold is a CLI tool for AI image generation using FLUX, SDXL, and Z-Image models via the [candle](https://github.com/huggingface/candle) ML framework. It provides a local inference server that runs on GPU hosts and a client CLI that can generate images locally or by connecting to a remote server.

## Build & Development Commands

### Nix (preferred)

```bash
nix build                                            # Build mold (default, includes serve with GPU)
nix run                                              # Run mold CLI
nix develop                                          # Enter devshell (auto via direnv)
nix fmt                                              # Format Nix + Rust (nixfmt + rustfmt)
nix flake check                                      # Validate formatting + flake
```

### Devshell commands (available inside `nix develop`)

| Category | Command | Description |
|----------|---------|-------------|
| build | `build` | `cargo build` (debug, all crates) |
| build | `build-release` | `cargo build --release` |
| build | `build-server` | `cargo build -p mold-cli --features {cuda\|metal}` (single binary with GPU) |
| check | `check` | `cargo check` |
| check | `clippy` | `cargo clippy` |
| check | `run-tests` | `cargo test` |
| check | `fmt` | `cargo fmt` |
| check | `fmt-check` | `cargo fmt --check` |
| run | `mold` | Run mold CLI (e.g. `mold list`, `mold ps`) |
| run | `serve` | Start the mold server |
| run | `generate` | Generate an image from a prompt |

### Cargo (direct)

```bash
cargo build                                          # Debug build (all crates)
cargo build --release                                # Release build
cargo build -p mold-cli                              # Just the CLI
cargo build -p mold-cli --features cuda              # CLI with CUDA (includes serve)
cargo check                                          # Type check
cargo clippy                                         # Lint
cargo fmt --check                                    # Format check
cargo test                                           # All tests
cargo test -p mold-core                              # Single crate
cargo run -p mold-cli -- run "a cat"                 # Generate image
cargo run -p mold-cli -- serve                       # Start server
```

### CI (GitHub Actions)

CI runs on every push and PR (`.github/workflows/ci.yml`): `cargo check`, `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo test --workspace`. All four must pass.

## Project Vision

- **Local-first**: Run diffusion models directly on your GPU
- **Remote-capable**: Point `MOLD_HOST` at a GPU server and generate from anywhere
- **Model management**: Pull, list, load/unload models (`mold pull`, `mold list`)
- **Simple CLI**: `mold run flux-dev:q4 "a cat"` — just works
- **Pipe-friendly**: `mold run "a cat" | viu -` — composable with Unix tools
- **Future**: OCI registry for model distribution

## Crate Structure

```
crates/
├── mold-core/                # Shared types, API protocol, HTTP client, config, model manifests
├── mold-inference/           # Candle-based inference engine (FLUX, SDXL, Z-Image)
├── mold-server/              # Axum HTTP inference server (lib + binary)
└── mold-cli/                 # Main binary — CLI (clap)
```

### mold-core

Shared library used by all other crates:

- **`types.rs`** — API request/response types (`GenerateRequest`, `GenerateResponse`, `ModelInfo`, `ServerStatus`, etc.)
- **`client.rs`** — `MoldClient` HTTP client; `is_connection_error()` for local fallback detection
- **`config.rs`** — `Config`, `ModelConfig`, `ModelPaths`; loads from `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy)
- **`manifest.rs`** — `ModelManifest` registry of downloadable models with HF sources; `resolve_model_name()` for `name:tag` resolution
- **`download.rs`** — `pull_model()` wrapping `hf-hub` with progress bars
- **`validation.rs`** — `validate_generate_request()` — shared validation (used by both server and CLI)
- **`error.rs`** — `MoldError` enum with thiserror

### mold-inference

Three model families, each with its own pipeline implementing the `InferenceEngine` trait:

```rust
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<()>;
    fn set_on_progress(&mut self, _callback: ProgressCallback) {} // default no-op
}
```

**Engine factory** — `create_engine()` in `factory.rs` auto-detects the model family and returns:
- `"flux"` → `FluxEngine` — T5 + CLIP-L text encoding, flow-matching transformer, VAE decode
- `"sdxl"` → `SDXLEngine` — Dual-CLIP (CLIP-L + CLIP-G), UNet with DDIM/Euler Ancestral, classifier-free guidance
- `"z-image"` → `ZImageEngine` — Qwen3 text encoder, flow-matching transformer with 3D RoPE

**Key architectural patterns:**

- **Lazy loading** — Models load on first generation request, not at startup
- **mmap for safetensors** — OS manages paging for memory-mapped model weights
- **BF16/GGUF dual support** — Each engine's transformer type wraps both formats (e.g. `FluxTransformer` enum). Auto-detected by `.gguf` extension.
- **Drop-and-reload for text encoders** — T5/CLIP (FLUX) and Qwen3 (Z-Image) are dropped from GPU after encoding to free VRAM for denoising, then reloaded next generation
- **Dynamic device placement** — Text encoders placed on GPU or CPU based on remaining VRAM after transformer loads (thresholds in `device.rs`)
- **Quantized encoder auto-fallback** — When FP16/BF16 encoder doesn't fit in VRAM, auto-selects largest quantized GGUF variant that fits. Custom `GgufT5Encoder` and `GgufQwen3Encoder` in `encoders/` handle GGUF-specific tensor naming. Override with `--t5-variant` / `--qwen3-variant` flags.

**Z-Image GGUF specifics** — The quantized Z-Image transformer (`zimage/quantized_transformer.rs`) lives in this crate (not candle) because candle has no quantized Z-Image model. Key GGUF tensor name differences from BF16: fused `attention.qkv` vs separate Q/K/V, `x_embedder` vs `all_x_embedder.2-1`, etc.

Feature flags: `cuda` (CUDA backend), `metal` (Metal backend).

### mold-server

Axum HTTP server wrapping the inference engine. Used as a library by `mold-cli` (via `mold serve`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate images from prompt |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/models/load` | Load/swap the active model |
| `GET` | `/api/status` | Server health + status |
| `GET` | `/health` | Simple 200 OK health check |

State managed via `AppState` with `tokio::sync::Mutex` around the engine.

### mold-cli

Main binary. Feature flags `cuda` and `metal` forward through `mold-server` → `mold-inference` for GPU support in both `mold serve` and `mold run --local`.

## CLI Command Reference

```
mold run [MODEL] [PROMPT...] [OPTIONS]
    First positional arg is MODEL if it matches a known model name; otherwise it's prompt.

    -m, --model <MODEL>         Explicit model override
    -o, --output <PATH>         Output file [default: ./mold-{model}-{timestamp}.png]
        --width/--height <N>    Image dimensions [default: from model config]
        --steps <N>             Inference steps [default: from model config]
        --seed <N>              Random seed
        --batch <N>             Number of images [default: 1]
        --host <URL>            Override MOLD_HOST
        --format <FORMAT>       png or jpeg [default: png]
        --local                 Skip server, run inference locally (requires GPU features)
        --eager                 Keep all model components loaded simultaneously (faster, more memory)
        --t5-variant <TAG>      T5 encoder: auto, fp16, q8, q6, q5, q4, q3
        --qwen3-variant <TAG>   Qwen3 encoder (Z-Image): auto, bf16, q8, q6, iq4, q3

mold serve [--port N] [--bind ADDR] [--models-dir PATH]
mold pull <MODEL>               Download model from HuggingFace
mold list                       List configured and available models
mold ps                         Show server status + loaded models
mold version                    Show version
mold completions <SHELL>        Generate shell completions
```

**Piping**: When stdout is not a TTY, raw image bytes go to stdout, status/progress to stderr. `mold run "a cat" | viu -` just works. `--output -` forces stdout even in interactive terminals.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLD_HOST` | `http://localhost:7680` | Remote server URL |
| `MOLD_MODELS_DIR` | `~/.mold/models` | Model storage directory |
| `MOLD_PORT` | `7680` | Server port |
| `MOLD_LOG` | `warn` (CLI) / `info` (server) | Log level (trace, debug, info, warn, error) |
| `MOLD_EAGER` | — | Set `1` to keep all model components loaded simultaneously |
| `MOLD_TRANSFORMER_PATH` | — | Override transformer path |
| `MOLD_VAE_PATH` | — | Override VAE path |
| `MOLD_T5_PATH` | — | Override T5-XXL encoder path |
| `MOLD_CLIP_PATH` | — | Override CLIP-L encoder path |
| `MOLD_T5_TOKENIZER_PATH` | — | Override T5 tokenizer path |
| `MOLD_CLIP_TOKENIZER_PATH` | — | Override CLIP tokenizer path |
| `MOLD_T5_VARIANT` | `auto` | T5 encoder variant: auto, fp16, q8, q6, q5, q4, q3 |
| `MOLD_QWEN3_VARIANT` | `auto` | Qwen3 encoder variant: auto, bf16, q8, q6, iq4, q3 |
| `MOLD_CLIP2_PATH` | — | Override CLIP-G encoder path (SDXL) |
| `MOLD_CLIP2_TOKENIZER_PATH` | — | Override CLIP-G tokenizer path (SDXL) |

Env vars take precedence over config file values. `mold pull` auto-writes config entries pointing to hf-hub cache paths.

## Config File

Location: `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy — used if `~/.mold/` exists)

```toml
default_model = "flux-schnell:q8"
models_dir = "~/.mold/models"
server_port = 7680
default_width = 1024
default_height = 1024
# t5_variant = "auto"
# qwen3_variant = "auto"

[models."flux-schnell:q8"]
transformer = "/path/to/flux1-schnell-Q8_0.gguf"
vae = "/path/to/ae.safetensors"
t5_encoder = "/path/to/t5xxl_fp16.safetensors"
clip_encoder = "/path/to/clip_l.safetensors"
t5_tokenizer = "/path/to/t5.tokenizer.json"
clip_tokenizer = "/path/to/clip.tokenizer.json"
default_steps = 4
default_guidance = 0.0
is_schnell = true
```

## Model System

**Name resolution**: `model:tag` format (e.g. `flux-dev:q4`). Bare names default to `:q8` (FLUX/Z-Image) or `:fp16` (SDXL). Legacy dash format (`flux-dev-q4`) resolves to colon format. See `manifest.rs` for all available models and HF sources.

**`mold run` inference modes:**
1. **Remote (default)**: Connects to `mold serve` via HTTP
2. **Local fallback**: If server unreachable, falls back to local GPU inference (with auto-pull if model not downloaded)
3. **Local forced (`--local`)**: Skip server attempt entirely

> **VRAM note**: Full BF16 FLUX dev (23GB) causes OOM on 24GB cards. Always use GGUF quantized models. SDXL FP16 UNets (~5GB) fit comfortably.

## Deployment

**Nix build with CUDA**: `nix build .#mold` or `nix develop --command cargo build --release -p mold-cli --features cuda`

**Systemd service**: `~/.config/systemd/user/mold-server.service` — configure with `LD_LIBRARY_PATH=/run/opengl-driver/lib` for NixOS CUDA driver access.

**Remote setup**: GPU host runs `mold serve --port 7680`, clients set `MOLD_HOST=http://gpu-host:7680`.

## Key Design Decisions

1. **Workspace crate separation** — core/inference/server/cli have clean dependency boundaries. CLI doesn't need candle, server doesn't need clap.
2. **candle over tch/ort** — Pure Rust, first-class FLUX support, no libtorch dependency. CUDA, Metal, and CPU backends.
3. **Single binary** — `mold` includes `serve` via `mold-server` library. GPU feature flags (`cuda`/`metal`) forward through `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` for engine state** — Async-aware mutex; single-model-at-a-time is appropriate for GPU workloads. Inference runs in `spawn_blocking`.
5. **Smart VRAM management** — Dynamic device placement + drop-and-reload + quantized encoder auto-fallback. See `device.rs` for thresholds.
6. **Model pull via hf-hub** — rustls TLS (no OpenSSL), shared components deduplicated by hf-hub cache, `Progress` trait adapter bridges to `indicatif::ProgressBar`.
7. **Nix flake (flake-parts + crane)** — Pure Nix Rust builds, numtide devshell, treefmt-nix. CUDA 12.8 + `CUDA_COMPUTE_CAP=89` on Linux (RTX 4090), Metal on macOS. Devshell sets `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` for CUDA compilation.
8. **Shell completions** — Static via `clap_complete` + dynamic via `CompleteEnv` with `ArgValueCandidates` for model names.
9. **Pipe-friendly output** — `IsTerminal` detection routes image bytes to stdout, status to stderr. SIGPIPE reset to default. `status!` macro handles routing.
10. **Unified `run` command** — First positional arg disambiguated at runtime: known model name vs prompt text.
