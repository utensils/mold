# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# mold — Architecture & Development Guide

> Like ollama, but for diffusion models.

mold is a CLI tool for AI image generation using FLUX models via the [candle](https://github.com/huggingface/candle) ML framework. It provides a local inference server that runs on GPU hosts and a client CLI that can generate images locally or by connecting to a remote server.

## Build & Development Commands

### Nix (preferred)

```bash
# Pure Nix builds via crane (single binary with GPU features)
nix build                                            # Build mold (default)
nix build .#mold                                     # Build mold (explicit, includes serve with GPU)

# Run directly
nix run                                              # Run mold CLI

# Dev shell (numtide devshell with menu commands)
nix develop                                          # Enter devshell (auto via direnv)

# Formatting
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
| run | `generate` | Generate an image from a prompt |
| deploy | `deploy` | Deploy to hal9000 |

### Cargo (direct)

```bash
cargo build                                          # Debug build (all crates)
cargo build --release                                # Release build
cargo build -p mold-cli                              # Just the CLI
cargo build -p mold-cli --features cuda               # CLI with CUDA (includes serve)
cargo check                                          # Type check
cargo clippy                                         # Lint
cargo fmt --check                                    # Format check
cargo test                                           # All tests
cargo test -p mold-core                              # Single crate
cargo run -p mold-cli -- generate "a cat"            # Generate image
cargo run -p mold-cli -- serve                       # Start server
cargo run -p mold-cli -- run "a cat"                 # Generate image
./scripts/deploy.sh                                  # Deploy to hal9000
```

## Project Vision

- **Local-first**: Run FLUX models directly on your GPU
- **Remote-capable**: Point `MOLD_HOST` at a GPU server and generate from anywhere
- **Model management**: Pull, list, load/unload models (like `ollama pull`)
- **Simple CLI**: `mold run flux-dev:q4 "a cat"` — just works
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
│   └── mold-cli/                 # Main binary — CLI (clap)
```

### mold-core

Shared library used by all other crates. Contains:

- **`types.rs`** — API request/response types (`GenerateRequest`, `GenerateResponse`, `ModelInfo`, `ServerStatus`, etc.)
- **`client.rs`** — `MoldClient` HTTP client for communicating with mold-server; includes `is_connection_error()` for fallback detection
- **`config.rs`** — `Config` struct, `ModelConfig`, `ModelPaths`; loads from `~/.mold/config.toml` (legacy) or `~/.config/mold/config.toml` (XDG); supports `save()` and `upsert_model()` for post-pull config writes
- **`manifest.rs`** — `ModelManifest`, `ModelFile`, `ModelComponent` types; `known_manifests()` registry of downloadable models with HF sources and generation defaults; `resolve_model_name()` for `name:tag` resolution
- **`download.rs`** — `pull_model()` download engine wrapping `hf-hub` crate with `indicatif` progress bars; handles gated model errors
- **`validation.rs`** — `validate_generate_request()` — shared request validation (used by both server and CLI local inference)
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
├── lib.rs                    # Re-exports (same public API)
├── engine.rs                 # InferenceEngine trait + rand_seed()
├── device.rs                 # VRAM queries, should_use_gpu(), fmt_gb(), thresholds
├── image.rs                  # Tensor → PNG/JPEG encoding
├── error.rs                  # InferenceError enum
├── progress.rs               # ProgressEvent enum + ProgressCallback type
├── model_registry.rs         # Delegates to mold_core::manifest for known models
├── encoders/
│   ├── mod.rs                # pub mod t5; pub mod clip;
│   ├── t5.rs                 # T5Encoder struct: config, load, encode, drop, reload
│   └── clip.rs               # ClipEncoder struct: config, load, encode, drop, reload
└── flux/
    ├── mod.rs                # Module declarations + re-exports
    ├── transformer.rs        # FluxTransformer enum (BF16/Quantized) + denoise()
    └── pipeline.rs           # FluxEngine + LoadedFlux + InferenceEngine impl
```

The `InferenceEngine` trait:
```rust
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<()>;
}
```

**`FluxEngine`** (in `flux/pipeline.rs`) implements real FLUX.1 inference using candle-transformers:

1. **T5 encoding** — `candle_transformers::models::t5::T5EncoderModel` encodes prompt to 4096-dim embeddings (padded to 256 tokens)
2. **CLIP encoding** — `candle_transformers::models::clip::text_model::ClipTextTransformer` encodes prompt to 768-dim embeddings
3. **Noise generation** — `flux::sampling::get_noise()` creates initial latent noise
4. **State construction** — `flux::sampling::State::new()` packages text + image embeddings
5. **Timestep scheduling** — `flux::sampling::get_schedule()` (4 steps for schnell, configurable for dev with shift)
6. **Denoising** — `flux::sampling::denoise()` runs the FLUX transformer with Euler sampling
7. **Unpacking** — `flux::sampling::unpack()` converts packed latents to spatial layout
8. **VAE decoding** — `flux::autoencoder::AutoEncoder::decode()` converts latents to pixels
9. **Image encoding** — Tensor → RGB → PNG/JPEG bytes via the `image` crate

**Device split** — FLUX transformer and VAE always load on **GPU**. T5 and CLIP encoders are placed on **GPU or CPU** dynamically based on remaining VRAM after the transformer is loaded (see `device.rs` thresholds). When placed on GPU, they are dropped after encoding to free VRAM for the denoising pass, then reloaded on the next generation. This allows optimal placement: e.g. Q4 models leave enough VRAM for T5 on GPU, while Q8/BF16 models fall back to CPU.

**Quantized T5 auto-fallback** — When FP16 T5 (9.2GB) doesn't fit in VRAM but a smaller quantized T5 GGUF does, the engine automatically selects the largest quantized variant that fits on GPU and downloads it if needed. This means Q8 transformer + Q8 T5 (5.06GB) = ~17GB total, well within 24GB, giving GPU-speed encoding instead of CPU fallback. Control via `MOLD_T5_VARIANT` env var, `t5_variant` config, or `--t5-variant` CLI flag.

**GGUF quantized models** — the engine auto-detects `.gguf` extension on `MOLD_TRANSFORMER_PATH` and uses `candle_transformers::quantized_var_builder` + `flux::quantized_model::Flux`. GGUF quantized models (Q4_1 = 7GB, Q8_0 = 12GB) leave plenty of VRAM for activations. When quantized, state tensors use F32; when BF16, they use BF16.

**FluxTransformer enum** (in `flux/transformer.rs`) wraps either `flux::model::Flux` (BF16) or `flux::quantized_model::Flux` (GGUF quantized). Both implement `flux::WithForward` so the same `denoise()` call works for both.

Model loading is **lazy** (on first generation request) and uses **mmap** for safetensors files.

Feature flags: `cuda` (CUDA backend), `metal` (Metal backend).

### mold-server

Axum-based HTTP server that wraps the inference engine. Runs on the GPU host. Used as a library by `mold-cli` (via `mold serve`); the standalone `mold-server` binary exists in the crate but is not built by the Nix flake.

Routes:
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate images from prompt |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/status` | Server health + status |
| `GET` | `/health` | Simple 200 OK health check |

**API validation**: width/height must be multiples of 16, max 1024, min 1. Steps 1–100.

State is managed via `AppState` which holds a `tokio::sync::Mutex<FluxEngine>`. Model is loaded lazily on first `/api/generate` request.

### mold-cli

Main binary crate. Provides CLI commands and shell completions. Feature flags `cuda` and `metal` forward through both `mold-server` and `mold-inference` for GPU-accelerated `mold serve` and local `mold run` inference.

## CLI Command Reference

```
mold run [MODEL] [PROMPT...] [OPTIONS]
    If PROMPT provided → one-shot generation
    PROMPT is required
    First positional arg is MODEL if it matches a known model name

    -m, --model <MODEL>         Explicit model override (bypasses arg heuristics)
    -o, --output <PATH>         Output file path [default: ./mold-{model}-{timestamp}.png]
        --width <N>             Image width [default: model config value]
        --height <N>            Image height [default: model config value]
        --steps <N>             Inference steps [default: model config value]
        --seed <N>              Random seed [random if not set]
        --batch <N>             Number of images [default: 1]
        --host <URL>            Override MOLD_HOST env var
        --format <FORMAT>       png or jpeg [default: png]
        --local                 Skip server, run inference locally (requires GPU features)
        --t5-variant <TAG>      T5 encoder variant: auto, fp16, q8, q6, q5, q4, q3

    Examples:
        mold run flux-dev:q4 "a turtle in the desert"    # specific model + prompt
        mold run "a sunset over mountains"                # default model + prompt
        mold run --model flux-krea:q8 "a sunset"            # explicit model flag

mold serve [OPTIONS]
        --port <N>              Server port [default: 7680]
        --bind <ADDR>           Bind address [default: 0.0.0.0]
        --models-dir <PATH>     Override MOLD_MODELS_DIR

mold pull <MODEL>               Download model from HuggingFace (e.g. flux-schnell, flux-dev:q4)
mold list                       List configured and available models
mold ps                         Show server status + loaded models
mold version                    Show version information
mold completions <SHELL>        Generate shell completions (bash, zsh, fish, elvish, powershell)
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
| `MOLD_T5_VARIANT` | — | T5 encoder variant: auto (default), fp16, q8, q6, q5, q4, q3 |

## Config File

Location: `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy — used if `~/.mold/` exists)

`mold pull` automatically writes model configs after downloading. Manual example:

```toml
default_model = "flux-schnell:q8"
models_dir = "~/.mold/models"
server_port = 7680
output_dir = "."
default_width = 1024
default_height = 1024
# t5_variant = "auto"  # auto, fp16, q8, q6, q5, q4, q3

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

Loaded via `Config::load_or_default()` — falls back to defaults if the file doesn't exist. Model paths can also be set via env vars (see above).

## Model Pull System

`mold pull <model>` downloads all required files from HuggingFace and writes the config automatically:

```bash
mold pull flux-schnell          # Downloads flux-schnell:q8 (default tag)
mold pull flux-dev:q4           # Downloads specific quantization
mold pull flux-dev-q4           # Legacy format, same as flux-dev:q4
```

**Name resolution**: Names use ollama-style `model:tag` format. Bare names default to `:q8`. Legacy dash format (`flux-dev-q4`) resolves to colon format (`flux-dev:q4`).

**Available models**:

| Name | Transformer Source | Total Size |
|------|-------------------|-----------|
| `flux-schnell:q8` | `city96/FLUX.1-schnell-gguf` / `flux1-schnell-Q8_0.gguf` | 12.0GB |
| `flux-schnell:q6` | `city96/FLUX.1-schnell-gguf` / `flux1-schnell-Q6_K.gguf` | 9.8GB |
| `flux-schnell:q4` | `city96/FLUX.1-schnell-gguf` / `flux1-schnell-Q4_1.gguf` | 7.5GB |
| `flux-dev:q8` | `city96/FLUX.1-dev-gguf` / `flux1-dev-Q8_0.gguf` | 12.0GB |
| `flux-dev:q6` | `city96/FLUX.1-dev-gguf` / `flux1-dev-Q6_K.gguf` | 9.9GB |
| `flux-dev:q4` | `city96/FLUX.1-dev-gguf` / `flux1-dev-Q4_1.gguf` | 7.0GB |
| `flux-krea:q8` | `QuantStack/FLUX.1-Krea-dev-GGUF` / `flux1-krea-dev-Q8_0.gguf` | 12.7GB |
| `flux-krea:q6` | `QuantStack/FLUX.1-Krea-dev-GGUF` / `flux1-krea-dev-Q6_K.gguf` | 9.9GB |
| `flux-krea:q4` | `QuantStack/FLUX.1-Krea-dev-GGUF` / `flux1-krea-dev-Q4_1.gguf` | 7.5GB |

Sizes are transformer only. First pull also downloads ~9.8GB of shared components (T5, CLIP, VAE, tokenizers), cached and reused across all models via hf-hub (`~/.cache/huggingface/hub/`).

Model manifests are defined in `mold-core/src/manifest.rs`.

## Local & Remote Inference

`mold run` works in three modes:

1. **Remote (default)**: Connects to a running `mold serve` instance via HTTP. Set `MOLD_HOST` to point at a remote GPU server.
2. **Local fallback**: If no server is running (connection refused), automatically falls back to local GPU inference when built with `--features cuda` or `--features metal`. If the model isn't downloaded yet, it is auto-pulled before inference. If built without GPU features, fails with a clear error message.
3. **Local forced (`--local`)**: Skip the server attempt entirely with `mold run --local "prompt"`. Goes straight to local inference (with auto-pull if needed).

For remote rendering:

1. On the GPU host: `mold serve --port 7680`
2. On any client: `MOLD_HOST=http://gpu-host:7680 mold run "a sunset"`

The client sends a `GenerateRequest` via HTTP POST to `/api/generate` and receives the generated image bytes in the response. All CLI commands (`generate`, `list`, `ps`) communicate through the same HTTP API.

## Deployment to hal9000

### Hardware
- NixOS, RTX 4090 (24GB VRAM)
- SSH: `jamesbrink@hal9000.home.urandom.io`
- If direct TCP fails: `ssh -J bender.tail1f4f9.ts.net jamesbrink@10.70.100.206`

### Model files on hal9000 (already on disk)
```
# GGUF quantized (recommended — fits in 24GB with room for activations)
flux1-schnell-Q8_0.gguf   /home/jamesbrink/AI/models/unet/flux1-schnell-Q8_0.gguf   (12GB) ← ACTIVE
flux1-dev-Q8_0.gguf       /home/jamesbrink/AI/models/unet/flux1-dev-Q8_0.gguf       (12GB)
flux1-dev-Q4_1.gguf       /home/jamesbrink/AI/models/unet/flux1-dev-Q4_1.gguf       (7GB)

# BF16 safetensors (23GB — causes CUDA OOM during denoising, avoid)
flux1-dev.safetensors      /home/jamesbrink/AI/models/unet/flux1-dev.safetensors     (23GB)

# Shared components (used regardless of transformer choice)
VAE:          /home/jamesbrink/AI/models/vae/ae.safetensors
T5 encoder:   /home/jamesbrink/AI/models/text_encoders/t5xxl_fp16.safetensors
CLIP-L:       /home/jamesbrink/AI/models/clip/clip_l.safetensors
T5 tokenizer: /home/jamesbrink/AI/models/tokenizers/t5-v1_1-xxl.tokenizer.json
CLIP tokenizer: /home/jamesbrink/AI/models/tokenizers/clip-vit-large-patch14.tokenizer.json
```

### Building with CUDA (Nix devshell)

The `flake.nix` (flake-parts + numtide devshell + crane) provides a devshell with all CUDA 12.8 dependencies and pure Nix builds:

```bash
# On hal9000 — pure Nix build (preferred):
cd /home/jamesbrink/mold
git pull
nix build .#mold

# Or via devshell:
nix develop --command cargo build --release -p mold-cli --features cuda
# Or inside the shell:
nix develop
build-server  # devshell command, builds mold-cli with --features cuda on Linux
```

### Systemd user service

The server runs as a systemd user service on hal9000:

```bash
# Check status
systemctl --user is-active mold-server
journalctl --user -u mold-server -f

# Restart
systemctl --user restart mold-server

# Service file location
~/.config/systemd/user/mold-server.service
```

The service is configured with all required env vars including `LD_LIBRARY_PATH=/run/opengl-driver/lib` for CUDA driver access.

### First-time tokenizer setup

```bash
ssh jamesbrink@hal9000.home.urandom.io
bash /home/jamesbrink/mold/scripts/fetch-tokenizers.sh
```

### Deploy from local

```bash
# From the mold project root:
./scripts/deploy.sh
```

### Test

```bash
# Direct HTTP test
curl -X POST http://hal9000.home.urandom.io:7680/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat on Mars","model":"flux-schnell","width":512,"height":512,"steps":4,"batch_size":1,"output_format":"png"}' \
  -o output.png

# Or via ProxyJump if local TCP is broken:
ssh -J bender.tail1f4f9.ts.net jamesbrink@10.70.100.206 \
  'curl -X POST http://localhost:7680/api/generate ...'
```

## Known Models

| Name | Transformer | Recommended File | Size |
|------|-------------|------------------|------|
| `flux-schnell:q8` | `city96/FLUX.1-schnell-gguf` | `flux1-schnell-Q8_0.gguf` | 12.0GB |
| `flux-schnell:q6` | `city96/FLUX.1-schnell-gguf` | `flux1-schnell-Q6_K.gguf` | 9.8GB |
| `flux-schnell:q4` | `city96/FLUX.1-schnell-gguf` | `flux1-schnell-Q4_1.gguf` | 7.5GB |
| `flux-dev:q8` | `city96/FLUX.1-dev-gguf` | `flux1-dev-Q8_0.gguf` | 12.0GB |
| `flux-dev:q6` | `city96/FLUX.1-dev-gguf` | `flux1-dev-Q6_K.gguf` | 9.9GB |
| `flux-dev:q4` | `city96/FLUX.1-dev-gguf` | `flux1-dev-Q4_1.gguf` | 7.0GB |
| `flux-krea:q8` | `QuantStack/FLUX.1-Krea-dev-GGUF` | `flux1-krea-dev-Q8_0.gguf` | 12.7GB |
| `flux-krea:q6` | `QuantStack/FLUX.1-Krea-dev-GGUF` | `flux1-krea-dev-Q6_K.gguf` | 9.9GB |
| `flux-krea:q4` | `QuantStack/FLUX.1-Krea-dev-GGUF` | `flux1-krea-dev-Q4_1.gguf` | 7.5GB |

> **Note:** The full BF16 safetensors FLUX dev (23GB) fills all 24GB VRAM and causes CUDA OOM during the denoising activation pass. Always use GGUF quantized models on the RTX 4090.

**HuggingFace download sources for shared components:**
| File | Source |
|------|--------|
| `flux1-schnell-Q8_0.gguf` | `city96/FLUX.1-schnell-gguf` |
| `ae.safetensors` | `black-forest-labs/FLUX.1-schnell` |
| `t5xxl_fp16.safetensors` | `comfyanonymous/flux_text_encoders` |
| `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` |
| `t5-v1_1-xxl.tokenizer.json` (T5) | `lmz/mt5-tokenizers` |
| `tokenizer.json` (CLIP) | `openai/clip-vit-large-patch14` |

Model manifests are defined in `mold-core/src/manifest.rs`. The inference crate's `model_registry.rs` delegates to the manifest.

## Future: OCI Registry for Models

Planned support for distributing models via OCI-compatible registries (similar to how ollama uses Docker registry format). This would allow:
- `mold pull registry.example.com/flux-schnell:latest`
- Private model registries
- Versioned model artifacts with layers for components (text encoder, VAE, transformer)

## Key Design Decisions

1. **Workspace structure**: Separating core types, inference, server, and CLI into distinct crates allows independent compilation and clean dependency boundaries. The CLI doesn't need candle, and the server doesn't need clap.

2. **candle over tch/ort**: candle is pure Rust, provides first-class FLUX support, and doesn't require libtorch system dependencies. It supports CUDA, Metal, and CPU backends.

3. **Axum for the server**: Modern, ergonomic, and built on tokio/tower. Natural fit for the async Rust ecosystem.

4. **Lazy model loading**: Model components (transformer, T5, CLIP, VAE) are loaded on first request rather than at startup, so the server starts fast and only uses VRAM when needed.

5. **`InferenceEngine` trait**: Allows swapping backends (e.g., candle CUDA vs CPU, or future ONNX backend) without changing server/CLI code.

6. **`tokio::sync::Mutex` for engine state**: Async-aware mutex prevents blocking the tokio runtime during generation. Single-model-at-a-time design is appropriate for GPU workloads.

7. **mmap for safetensors**: Model weights are memory-mapped rather than read into memory, allowing the OS to manage paging efficiently.

8. **Env var + config file model paths**: Supports both deployment-friendly env vars and local config file paths. Env vars take precedence over config. `mold pull` auto-writes config entries pointing to hf-hub cache paths.

9. **GGUF over BF16 safetensors for 24GB VRAM**: The full BF16 FLUX dev model (23GB) fills all VRAM, leaving nothing for activations. GGUF Q8_0 (12GB) or Q4_1 (7GB) leave plenty of room. Engine auto-detects `.gguf` extension.

10. **Smart T5/CLIP device placement**: Text encoders are placed on GPU or CPU dynamically based on free VRAM after the transformer loads (thresholds in `device.rs`). When on GPU, encoder weights are dropped after prompt encoding to free VRAM for denoising, then reloaded on the next generation. Embeddings are always moved to GPU regardless of encoder placement.

11. **Single binary**: `mold` is the only binary built by the Nix flake. It includes `serve` (via `mold-server` library), so GPU feature flags (`cuda`/`metal`) are forwarded through `mold-cli` → `mold-server` → `mold-inference`. No separate `mold-server` package is needed.

12. **Nix flake (flake-parts + crane)**: `flake.nix` uses flake-parts for structure, crane for pure Nix Rust builds (`nix build .#mold`), numtide devshell with categorized menu commands, and treefmt-nix for `nix fmt`. CUDA 12.8 packages and `CUDA_COMPUTE_CAP=89` are configured for Linux (RTX 4090). Metal is used on macOS. The devshell sets `CPATH` (cuda_cudart + cuda_cccl includes for kernel compilation), `LIBRARY_PATH` (link-time CUDA libs including stubs), and `LD_LIBRARY_PATH` (runtime — real driver from `/run/opengl-driver/lib` first, no stubs).

13. **Shell completions**: Two completion mechanisms are supported. Static: `mold completions <shell>` generates completion scripts via `clap_complete`. Dynamic: `CompleteEnv` enables the binary itself to respond to shell completion requests, with `ArgValueCandidates` providing model name completions for `mold run` and `mold pull` positional args. Set up dynamic completions with `source <(COMPLETE=bash mold)` (bash) or `source <(COMPLETE=zsh mold)` (zsh).

14. **Local inference fallback with auto-pull**: `mold run "a cat"` tries the remote server first, then falls back to local GPU inference if the server isn't running. If the model isn't downloaded yet, it is auto-pulled before inference — `mold run flux-dev:q4 "a turtle"` just works on first use. The fallback is behind `cfg(feature = "cuda"/"metal")` so non-GPU builds get a clear error instead. Inference runs in `tokio::task::spawn_blocking` to avoid blocking the async runtime.

15. **Shared validation**: `validate_generate_request()` lives in `mold-core` and is used by both the HTTP server (for API requests) and the CLI (for local inference), ensuring consistent validation rules.

16. **Model pull via hf-hub**: `mold pull` uses the `hf-hub` crate (rustls TLS, no OpenSSL dependency) with `download_with_progress()` for per-file progress bars. Shared FLUX components (VAE, T5, CLIP, tokenizers) are defined once in the manifest and deduplicated by hf-hub's cache. The `Progress` trait adapter bridges hf-hub's async progress callbacks to `indicatif::ProgressBar`.

17. **XDG directory support**: New installs use `~/.config/mold/config.toml` (config) and `~/.local/share/mold/` (data). Existing `~/.mold/` directories keep working — checked via `Config::legacy_dir_exists()`. No migration needed.

18. **Ollama-style model:tag naming**: Models use `name:tag` format (e.g., `flux-dev:q4`). `resolve_model_name()` handles bare names (default to `:q8`) and legacy dash format (`flux-dev-q4` → `flux-dev:q4`). Config lookup tries both forms for backward compatibility.

19. **Quantized T5 encoder with auto-fallback**: When the FP16 T5-XXL (9.2GB) doesn't fit in remaining VRAM alongside the FLUX transformer, the engine automatically selects the largest quantized T5 GGUF that fits on GPU. Available variants from `city96/t5-v1_1-xxl-encoder-gguf`: Q8 (5.06GB), Q6 (3.91GB), Q5 (3.39GB), Q4 (2.9GB), Q3 (2.1GB). Uses `candle_transformers::models::quantized_t5::T5EncoderModel` with the same tokenizer. Auto-downloaded via hf-hub sync API on first use. Override with `MOLD_T5_VARIANT` env var, `t5_variant` config field, or `--t5-variant` CLI flag. Priority: CLI flag > env var > config > auto.

20. **Unified `run` command with positional model arg**: `mold run` is the primary command. The first positional arg is disambiguated at runtime: if it matches a known model (manifests + config), it's treated as the model; otherwise it's part of the prompt. `mold run flux-dev:q4 "prompt"` = specific model; `mold run "prompt"` = default model. This mirrors `ollama run <model> <prompt>`.

## Confirmed Working Configuration (hal9000, 2026-03-15)

```
Model:      flux-dev Q4 GGUF (local inference fallback, no server)
Generation: 768×768, 20 steps, ~211s (debug build, model load included)
GPU:        RTX 4090 (CUDA 12.8, driver 580.119.02)

Model:      flux-schnell Q8_0 GGUF (via mold serve)
VRAM used:  ~12GB (transformer) + ~300MB (VAE) = ~12.3GB / 24GB
Generation: 512×512, 4 steps, ~36s first run (model load included)
GPU:        RTX 4090 (CUDA 12.8, driver 580.119.02)
```
