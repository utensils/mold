# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# mold — Architecture & Development Guide

> Local AI image generation CLI — FLUX, SD3.5, Stable Diffusion 1.5, SDXL, Z-Image, Flux.2, Qwen-Image, & Wuerstchen diffusion models on your GPU.

mold is a CLI tool for AI image generation using FLUX.1, SD3.5, Stable Diffusion 1.5, SDXL, Z-Image, Flux.2 Klein, Qwen-Image, and Wuerstchen v2 models via the [candle](https://github.com/huggingface/candle) ML framework. It provides a local inference server that runs on GPU hosts and a client CLI that can generate images locally or by connecting to a remote server. Supports txt2img, img2img, inpainting, and ControlNet conditioning.

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
| build | `build` | Fast local `mold` build (`cargo build --profile dev-fast -p mold-ai`) with the web bundle embedded |
| build | `build-workspace` | `cargo build` (debug, all crates) |
| build | `build-release` | Shipping `cargo build --release -p mold-ai --features {gpu},preview,discord,expand,tui,webp,mp4,metrics` |
| build | `build-server` | Fast local single-binary server build with GPU + preview + expand and embedded web UI |
| build | `build-discord` | Fast local `cargo build --profile dev-fast -p mold-ai --features discord` |
| check | `check` | `cargo check` |
| check | `clippy` | `cargo clippy` |
| check | `run-tests` | `cargo test` |
| check | `coverage` | Test coverage report (`--html` for browsable report) |
| check | `fmt` | `cargo fmt` |
| check | `fmt-check` | `cargo fmt --check` |
| run | `mold` | Run mold CLI (e.g. `mold list`, `mold ps`) |
| run | `serve` | Start the mold server |
| run | `generate` | Generate an image from a prompt |
| run | `discord-bot` | Start the mold Discord bot (`mold discord`) |

### Cargo (direct)

```bash
cargo build                                          # Debug build (all crates)
cargo build --profile dev-fast                       # Fast local optimized build
cargo build --release                                # Release build
cargo build -p mold-ai                               # Just the CLI
cargo build -p mold-ai --features cuda               # CLI with CUDA (includes serve)
cargo check                                          # Type check
cargo clippy                                         # Lint
cargo fmt --check                                    # Format check
cargo test                                           # All tests
cargo test -p mold-ai-core                           # Single crate
./scripts/coverage.sh                                # Test coverage summary
./scripts/coverage.sh --html                         # HTML coverage report
./scripts/fetch-tokenizers.sh                        # Pre-download tokenizer files
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai --features metal,preview,expand -- run "a cat"
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai --features metal,preview,expand -- serve
cargo run -p mold-ai-inference --features dev-bins --bin ltx2_review -- clip.mp4
```

### CI (GitHub Actions)

CI runs on every push and PR (`.github/workflows/ci.yml`). All jobs must pass:

| Job | What it checks |
|-----|----------------|
| `rust` | `cargo fmt --all -- --check && cargo check --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4` |
| `coverage` | `cargo llvm-cov` → Codecov upload |
| `docs` | `bun run fmt:check && bun run verify && bun run build` in `website/` |

> **Note:** `mold-inference` and `mold-server` have `[lib] test = false` in their `Cargo.toml` files. The test harness for these crates links against candle/CUDA which triggers heavy model weight initialization (~32GB RAM, 40+ min hang). The `mold-server` binary target also has `test = false`. Unit tests in `mold-core` and `mold-cli` run normally. If you add tests to `mold-inference` or `mold-server`, run them with `cargo test -p <crate> --lib` after temporarily removing the `test = false` flag.

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
├── mold-db/                  # SQLite metadata DB (rusqlite, bundled) — gallery, TUI settings, per-model prefs, prompt history
├── mold-inference/           # Candle-based inference engine (FLUX, SD1.5, SDXL, SD3, Z-Image, Flux.2, Qwen-Image, Wuerstchen)
├── mold-server/              # Axum HTTP inference server (library, consumed by mold-cli)
├── mold-cli/                 # Main binary — CLI (clap), single `mold` binary with feature flags
├── mold-discord/             # Discord bot library (feature-gated, consumed by mold-cli via `discord` feature)
└── mold-tui/                 # Interactive terminal UI (feature-gated, consumed by mold-cli via `tui` feature)
```

**Package names** — Directory names differ from Cargo package names. Use `-p <package>` with these:

| Directory | Package (`-p`) | Binary/Lib |
|-----------|---------------|------------|
| `mold-cli/` | `mold-ai` | binary: `mold` |
| `mold-core/` | `mold-ai-core` | lib: `mold_core` |
| `mold-db/` | `mold-ai-db` | lib: `mold_db` |
| `mold-inference/` | `mold-ai-inference` | lib: `mold_inference` |
| `mold-server/` | `mold-ai-server` | lib: `mold_server` |
| `mold-discord/` | `mold-ai-discord` | lib: `mold_discord` |
| `mold-tui/` | `mold-ai-tui` | — |

**MSRV**: 1.85

**Feature flags** (on `mold-cli`): `cuda` (CUDA GPU), `metal` (Metal GPU), `preview` (terminal image display), `discord` (Discord bot subcommand + `mold serve --discord`), `expand` (local LLM prompt expansion via `mold-inference`), `tui` (interactive terminal UI via `mold tui`), `metrics` (Prometheus `/metrics` endpoint via `mold-server`).

**Feature flags** (on `mold-inference`): `cuda`, `metal`, `expand` (same as above), `webp` (animated WebP output via libwebp FFI), `mp4` (AAC encoding for MP4 muxing on native video paths), `dev-bins` (native LTX-2 review/probe helper binaries). H.264/MP4 decode is part of the baseline LTX-2 source-video ingest stack, so `mp4` no longer gates the full video media dependency graph. GIF and APNG output still work without enabling `mp4`.

### mold-core

Shared library used by all other crates:

- **`types.rs`** — API request/response types (`GenerateRequest`, `GenerateResponse`, `ModelInfo`, `ServerStatus`, `ExpandRequest`, `ExpandResponse`, `LoraWeight`, etc.)
- **`client.rs`** — `MoldClient` HTTP client; `is_connection_error()` for local fallback detection; `expand_prompt()` for server-side expansion
- **`config.rs`** — `Config`, `ModelConfig`, `ModelPaths`; loads from `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy); per-model `lora` path and `lora_scale` defaults
- **`manifest.rs`** — `ModelManifest` registry of downloadable models with HF sources; `resolve_model_name()` for `name:tag` resolution; `UTILITY_FAMILIES` constant for non-diffusion models (e.g. `qwen3-expand`); `is_utility()` for family-based classification
- **`expand.rs`** — `PromptExpander` trait, `ApiExpander` (OpenAI-compatible HTTP), `ExpandConfig`/`ExpandSettings`/`FamilyOverride` settings with env var support (`MOLD_EXPAND_*`); user-configurable system prompt templates and per-family word limits/style notes
- **`download.rs`** — `pull_model()` wrapping `hf-hub` with progress bars; SHA-256 integrity verification (fails on mismatch, `--skip-verify` to override); `.pulling` marker for atomic pull detection; `PullOptions` for controlling verification behavior
- **`validation.rs`** — `validate_generate_request()` — shared validation (used by both server and CLI); `fit_to_model_dimensions()` — aspect-ratio-preserving resize of source images to model-native resolution for img2img; LoRA scale range [0.0, 2.0] and .safetensors extension validation
- **`error.rs`** — `MoldError` enum with thiserror

### mold-inference

Ten model families, each with its own pipeline implementing the `InferenceEngine` trait:

```rust
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<()>;
    fn unload(&mut self) {}                                       // free GPU memory
    fn set_on_progress(&mut self, _callback: ProgressCallback) {} // default no-op
    fn clear_on_progress(&mut self) {}
}
```

**Engine factory** — `create_engine()` in `factory.rs` auto-detects the model family and returns:
- `"flux"` → `FluxEngine` — T5 + CLIP-L text encoding, flow-matching transformer, VAE decode
- `"sd15"` (also `"sd1.5"`, `"stable-diffusion-1.5"`) → `SD15Engine` — CLIP-L text encoding, UNet with DDIM, classifier-free guidance (512x512 default)
- `"sdxl"` → `SDXLEngine` — Dual-CLIP (CLIP-L + CLIP-G), UNet with DDIM/Euler Ancestral, classifier-free guidance
- `"sd3"` (also `"sd3.5"`, `"stable-diffusion-3"`) → `SD3Engine` — Triple encoder (CLIP-L + CLIP-G + T5-XXL), quantized MMDiT with NaN-safe inference
- `"flux2"` (also `"flux.2"`, `"flux2-klein"`) → `Flux2Engine` — Qwen3 text encoder (BF16 or GGUF, layers 9/18/27), shared modulation transformer (BF16 or GGUF), BN-VAE
- `"qwen-image"` (also `"qwen_image"`) → `QwenImageEngine` — Qwen2.5-VL text encoder, 3D causal VAE (2D temporal-slice), flow-matching with classifier-free guidance
- `"z-image"` → `ZImageEngine` — Qwen3 text encoder, flow-matching transformer with 3D RoPE
- `"wuerstchen"` (also `"wuerstchen-v2"`) → `WuerstchenEngine` — CLIP-G text encoder, 3-stage cascade (Prior → Decoder → VQ-GAN), 42x latent compression
- `"ltx-video"` → `LtxVideoEngine` — T5-XXL text encoding, flow-matching transformer, 3D causal VAE, text-to-video (APNG/GIF/WebP/MP4 output). Current supported checkpoints are `0.9.6` and `0.9.8`; `0.9.8` pulls a spatial upscaler asset and currently runs first-pass generation only.
- `"ltx2"` (also `"ltx-2"`) → `Ltx2Engine` — Gemma 3 text encoder with LTX-2 connectors, joint audio-video DiT transformer, 3D causal video VAE, audio VAE + vocoder, MP4-first output with real AAC audio. Covers 19B/22B text+audio-video, image-to-video, audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale (`x1.5`/`x2`), and temporal upscale (`x2`). **CUDA-only for real generation** — CPU is a correctness-oriented fallback, and Metal is explicitly unsupported for this family. The native runtime is in `ltx2/` with runtime orchestration, guidance/perturbation paths, stacked LoRAs + camera-control presets, and a native MP4/GIF/APNG/WebP media pipeline.

**Additional modules:**
- `encoders/variant_resolution.rs` — Shared T5/Qwen3 encoder variant resolution (auto-fallback quantization)
- `scheduler.rs` — Configurable scheduler builder (DDIM, Euler Ancestral, UniPC) for SD1.5/SDXL
- `img_utils.rs` — Source image decoding, resizing, mask processing for img2img/inpainting
- `controlnet/` — ControlNet model (UNet encoder copy with zero convolutions) for SD1.5
- `weight_loader.rs` — `load_safetensors_with_progress()` utility: replaces opaque `VarBuilder::from_mmaped_safetensors()` with per-tensor loading that emits `WeightLoad` progress events for byte-level progress bars

**Key architectural patterns:**

- **Lazy loading** — Models load on first generation request, not at startup
- **mmap for safetensors** — OS manages paging for memory-mapped model weights
- **BF16/GGUF dual support** — Each engine's transformer type wraps both formats (e.g. `FluxTransformer` enum). Auto-detected by `.gguf` extension.
- **Drop-and-reload for text encoders** — T5/CLIP (FLUX) and Qwen3 (Z-Image) are dropped from GPU after encoding to free VRAM for denoising, then reloaded next generation
- **Dynamic device placement** — Text encoders placed on GPU or CPU based on remaining VRAM after transformer loads (thresholds in `device.rs`)
- **Quantized encoder auto-fallback** — When FP16/BF16 encoder doesn't fit in VRAM, auto-selects largest quantized GGUF variant that fits. Custom `GgufT5Encoder` and `GgufQwen3Encoder` in `encoders/` handle GGUF-specific tensor naming. Override with `--t5-variant` / `--qwen3-variant` flags.
- **Block-level offloading** — `flux/offload.rs`: streams transformer blocks between CPU and GPU one at a time, reducing VRAM from ~24GB to ~2-4GB (3-5x slower). Auto-enabled when VRAM is insufficient; force with `--offload` / `MOLD_OFFLOAD=1`.
- **LoRA adapter support** — `flux/lora.rs`: custom `SimpleBackend` (`LoraBackend`) that wraps mmap'd base weights and applies LoRA deltas inline during model construction. Parses diffusers-format LoRA safetensors (A/B weight pairs + alpha), maps keys to candle's fused tensor layout (QKV, linear1), and merges via `W' = W + scale * (B @ A)`. Compatible with block-level offloading — LoRA patches are baked into blocks on CPU, then streamed to GPU during inference.
- **LoRA fingerprint caching** — `FluxEngine` tracks an `active_lora: Option<LoraFingerprint>` (path hash + scale) to skip redundant transformer rebuilds when the same LoRA is used across requests. Cleared on `unload()`/`load()`.
- **LoRA delta caching** — `LoraDeltaCache` in `flux/lora.rs` caches pre-computed `B @ A * scale` delta tensors on CPU (~80-120MB for FLUX). Cache keys include `patch_index` to disambiguate fused QKV slices. Survives across transformer rebuilds within the same engine lifetime.
- **Shared tokenizer pool** — `shared_pool.rs`: cross-engine `SharedPool` caches T5 and CLIP tokenizers (keyed by file path) via `Arc<Tokenizer>`. All FLUX variants share the same tokenizer files, so model switches skip ~100-150ms of re-initialization. Passed through `create_engine_with_pool()` in `factory.rs`.

**Z-Image GGUF specifics** — The quantized Z-Image transformer (`zimage/quantized_transformer.rs`) lives in this crate (not candle) because candle has no quantized Z-Image model. Key GGUF tensor name differences from BF16: fused `attention.qkv` vs separate Q/K/V, `x_embedder` vs `all_x_embedder.2-1`, etc.

**Prompt expansion** — `expand.rs` (behind `expand` feature): `LocalExpander` wraps `candle_transformers::models::quantized_qwen3::ModelWeights` for local GGUF text generation. Uses `PromptExpander` trait from `mold-core`. Includes progress reporting, VRAM-aware device placement (`should_use_gpu()`), and Darwin memory safety guards (`preflight_memory_check()`). The LLM is always dropped before diffusion runs.

Feature flags: `cuda` (CUDA backend), `metal` (Metal backend), `expand` (local LLM prompt expansion).

### mold-server

> Feature flags: `cuda`, `metal`, `expand` (forwarded to mold-inference), `metrics` (Prometheus `/metrics` endpoint + instrumentation).

Axum HTTP server wrapping the inference engine. Used as a library by `mold-cli` (via `mold serve`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate images from prompt |
| `POST` | `/api/generate/stream` | Generate with SSE progress streaming |
| `POST` | `/api/expand` | Expand a prompt using LLM |
| `GET` | `/api/models` | List available models |
| `POST` | `/api/models/load` | Load/swap the active model |
| `POST` | `/api/models/pull` | Pull/download a model |
| `DELETE` | `/api/models/unload` | Unload model to free GPU memory |
| `GET` | `/api/gallery` | List saved images |
| `GET` | `/api/gallery/image/:name` | Fetch a saved image |
| `DELETE` | `/api/gallery/image/:name` | Delete a saved image |
| `GET` | `/api/gallery/thumbnail/:name` | Fetch a cached thumbnail |
| `POST` | `/api/upscale` | Upscale image with Real-ESRGAN |
| `POST` | `/api/upscale/stream` | Upscale with SSE tile progress streaming |
| `POST` | `/api/shutdown` | Trigger graceful server shutdown |
| `GET` | `/api/status` | Server health + status |
| `GET` | `/health` | Simple 200 OK health check |
| `GET` | `/api/openapi.json` | OpenAPI spec |
| `GET` | `/api/docs` | Interactive API docs (Scalar) |
| `GET` | `/metrics` | Prometheus metrics (feature-gated) |

State managed via `AppState` with `tokio::sync::Mutex<ModelCache>` (LRU cache, max 3 models). `ModelResidency` tracks each engine as `Gpu`, `Parked` (weights dropped but tokenizers/caches retained for fast reload), or `Unloaded`. At most one engine is GPU-resident at a time. `AppState` also holds a `shared_pool: Arc<Mutex<SharedPool>>` for cross-engine tokenizer caching and `upscaler_cache: Arc<std::sync::Mutex<Option<Box<dyn UpscaleEngine>>>>` for reusing loaded upscaler models across requests.

### mold-cli

Main binary. Feature flags `cuda` and `metal` forward through `mold-server` → `mold-inference` for GPU support in both `mold serve` and `mold run --local`.

### mold-discord

Discord bot library using **poise 0.6 + serenity 0.12**. Depends only on `mold-core` (no GPU features). Connects to a running `mold serve` via `MoldClient` HTTP/SSE API. Provides `/generate`, `/expand`, `/models`, `/status`, `/quota`, and `/admin` slash commands. Consumed by `mold-cli` behind the `discord` feature flag — invoked via `mold discord` (standalone bot) or `mold serve --discord` (server + bot in one process).

Key modules: `commands/` (slash command handlers — `generate`, `expand`, `models`, `status`, `quota`, `admin`), `handler.rs` (SSE streaming orchestration), `format.rs` (pure formatting functions including `format_expand_result()` and `format_quota()`), `cooldown.rs` (per-user rate limiting), `quota.rs` (per-user daily quota tracking), `access.rs` (role-based access control and block list), `checks.rs` (shared authorization checks), `state.rs` (shared bot state).

Token: `MOLD_DISCORD_TOKEN` (preferred) or `DISCORD_TOKEN` (fallback).

## CLI Quick Reference

Core commands: `mold run`, `mold serve`, `mold pull`, `mold list`, `mold ps`, `mold tui`, `mold upscale`, `mold expand`, `mold config`, `mold update`, `mold runpod`. Run `mold --help` or `mold <command> --help` for full flag details. `mold runpod run "<prompt>"` creates a RunPod pod, generates, and saves to `./mold-outputs/` in one command — see `website/deployment/runpod-cli.md`.

**Key behaviors:**
- `mold run [MODEL] [PROMPT]` — first positional arg is MODEL if it matches a known name, otherwise it's prompt
- **Pipe-friendly**: `echo "a cat" | mold run flux2-klein | viu -` — stdin for prompt, stdout for image bytes when not a TTY. `--output -` forces stdout. `--image -` reads source image from stdin.
- **Inference modes**: remote (default → `MOLD_HOST`), local fallback (auto if server unreachable), `--local` (forced local GPU)
- **VRAM management**: `--offload` streams blocks CPU↔GPU (~24GB→2-4GB, slower), `--eager` keeps everything loaded, `--t5-variant`/`--qwen3-variant`/`--qwen2-variant` control encoder quantization

## Environment Variables

All vars prefixed `MOLD_`. Key ones: `MOLD_HOST` (server URL, default `http://localhost:7680`), `MOLD_DEFAULT_MODEL` (default `flux2-klein`), `MOLD_MODELS_DIR` (default `~/.mold/models`), `MOLD_LOG` (default `warn` CLI / `info` server), `MOLD_PORT` (default `7680`), `MOLD_OUTPUT_DIR` (default `~/.mold/output`). Component path overrides: `MOLD_TRANSFORMER_PATH`, `MOLD_VAE_PATH`, `MOLD_T5_PATH`, `MOLD_CLIP_PATH`, `MOLD_CLIP2_PATH` + tokenizer variants. Encoder variant selection: `MOLD_T5_VARIANT`, `MOLD_QWEN3_VARIANT`, `MOLD_QWEN2_VARIANT`. Runtime toggles: `MOLD_OFFLOAD`, `MOLD_EAGER`, `MOLD_PREVIEW`, `MOLD_EXPAND`, `MOLD_EMBED_METADATA`. Server security: `MOLD_API_KEY`, `MOLD_RATE_LIMIT`, `MOLD_CORS_ORIGIN`. Discord: `MOLD_DISCORD_TOKEN`, `MOLD_DISCORD_COOLDOWN`, `MOLD_DISCORD_DAILY_QUOTA`. Debug: `MOLD_QWEN_DEBUG`, `MOLD_SD3_DEBUG`. Env vars override config file. Full list in source or `mold --help`.

## Config File

Location: `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy — used if `~/.mold/` exists). Structure defined in `crates/mold-core/src/config.rs`.

**After issue #265**, the surface is split between two stores with a single logical `Config` view:

| Surface | Owns | Keys |
|---|---|---|
| `config.toml` (bootstrap) | paths, ports, credentials | `default_model`, `models_dir`, `output_dir`, `server_port`, `gpus`, `queue_size`, `[logging]`, `[runpod]`, per-model `transformer`/`vae`/encoder/tokenizer paths |
| `mold.db` `settings` table | user preferences | `expand.*`, `generate.*`, `tui.*`, legacy `last-model` sidecar |
| `mold.db` `model_prefs` table | per-model generation defaults | `default_steps`, `default_guidance`, `default_width`, `default_height`, `scheduler`, `negative_prompt`, `lora`, `lora_scale` |
| `MOLD_*` env vars | runtime override | all of the above at read time (highest precedence) |

Every binary's `fn main()` (`mold`, `mold-server`, `mold-discord`) calls `mold_db::config_sync::install_config_post_load_hook()`, which wires a `Config::install_post_load_hook()` into `mold-core`. On first boot it runs an idempotent one-shot `config.toml → DB` import (gated by `config.migrated_from_toml` in `settings`), renames the original `config.toml` → `config.toml.migrated`, and rewrites the file via `Config::save_bootstrap_only` so it only carries bootstrap fields going forward. Subsequent loads overlay DB values onto every `Config::load_or_default()`. Consumers keep reading `cfg.expand.*` / `cfg.default_width` / `cfg.models[…]` unchanged — the hook is invisible. `mold-db::config_sync` exposes the typed load/save helpers (`save_expand_to_db`, `save_generate_globals_to_db`, `migrate_config_toml_to_db`, `hydrate_config_from_db`, `rewrite_stripped_config_toml`, `detect_stale_backups`, `cleanup_stale_backups`). `mold pull` auto-writes model path entries to TOML.

`mold config` subcommands route by key prefix:

- `mold config set expand.enabled true` → writes the DB (new).
- `mold config set models_dir /mnt/...` → writes `config.toml` (unchanged).
- `mold config where <key>` → prints `file` or `db` and reports any env-var override.
- `mold config list` / `--json` → each row tagged `[db]` / `[file]` / `[env]`; JSON mode returns `{ value, surface }` per key.
- `mold config reset <key>` → drops the DB row so the next read falls back to the TOML/env/default; TOML-only keys are rejected with a pointer at `mold config set`.
- `mold config reset --all [--yes]` → drops every DB-backed row under the active profile.
- `mold config --profile <name> <subcommand>` → operate on an explicit profile (v6 scoping). Overrides `MOLD_PROFILE` for the command.

**Multi-profile (v6 settings schema).** `settings` and `model_prefs` rows are keyed on `(profile, key)` / `(profile, model)`. Active profile resolves in priority order: `MOLD_PROFILE` env var → `settings.profile.active` (under the `default` profile) → `"default"`. `Settings::new(db)` / `ModelPrefs::load(db, model)` use the resolved active profile; `Settings::for_profile(db, name)` and `ModelPrefs::load_in/save_in/list_in/delete_in` take an explicit profile for cross-profile work.

## Model System

**Name resolution**: `model:tag` format (e.g. `flux-dev:q4`). Bare names resolve by trying `:q8` → `:fp16` → `:bf16` → `:fp8` in order, picking the first match. Legacy dash format (`flux-dev-q4`) resolves to colon format. See `manifest.rs` for all available models and HF sources.

**`mold run` inference modes:**
1. **Remote (default)**: Connects to `mold serve` via HTTP
2. **Local fallback**: If server unreachable, falls back to local GPU inference (with auto-pull if model not downloaded)
3. **Local forced (`--local`)**: Skip server attempt entirely

> **VRAM note**: Full BF16 FLUX dev (23GB) auto-offloads on 24GB cards (blocks stream CPU↔GPU). GGUF quantized models fit without offloading. SDXL FP16 UNets (~5GB) fit comfortably. LoRA adapters work with both BF16 (via offload path) and GGUF quantized FLUX models.

## Deployment

**Nix build with CUDA**: `nix build .#mold` (Ada/sm_89) or `nix build .#mold-sm120` (Blackwell/sm_120)

**Systemd service**: `~/.config/systemd/user/mold-server.service` — configure with `LD_LIBRARY_PATH=/run/opengl-driver/lib` for NixOS CUDA driver access.

**Remote setup**: GPU host runs `mold serve --port 7680`, clients set `MOLD_HOST=http://gpu-host:7680`.

**Docker / RunPod**: Multi-stage `Dockerfile` at project root. Stage 1 builds with `nvidia/cuda:12.8.1-devel-ubuntu22.04` + Rust + cargo (`--features cuda,expand`). Stage 2 copies the binary into `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3.4 GB image, 33 MB binary). `CUDA_COMPUTE_CAP` build arg (default `89`) targets the GPU architecture. `docker/start.sh` is the RunPod-convention entrypoint: detects `/workspace` network volume, sets `MOLD_HOME`/`MOLD_MODELS_DIR`, runs `mold serve --bind 0.0.0.0`. Note: `libcuda.so.1` (NVIDIA driver) is injected at runtime by the NVIDIA Container Toolkit — the binary cannot run without GPU access.

**Documentation site**: VitePress 1.6 + Tailwind CSS v4 + bun in `website/`. Dev server: `cd website && bun install && bun run dev -- --host 0.0.0.0`. Build: `bun run build`. Deployed to GitHub Pages via `.github/workflows/pages.yml` on push to `main` (website/** paths). Base path is `/mold/` (served at `utensils.github.io/mold/`).

**Web gallery UI** (separate from the docs site): Vue 3 + Vite 7 + Tailwind CSS v4.2 SPA in `web/`. The SPA is **embedded directly into the `mold` binary at compile time** via [`rust-embed`](https://crates.io/crates/rust-embed), so `nix build` produces a single-file server that serves the gallery with zero runtime filesystem dependency. Local devshell `build`, `build-server`, `mold`, `serve`, and `generate` commands now call `./scripts/ensure-web-dist.sh` first, so `target/dev-fast/mold` normally includes the real SPA by default instead of the placeholder stub. `crates/mold-server/build.rs` resolves the bundle from one of three sources, in order: `$MOLD_WEB_DIST` (set by the Nix flake's `mold-web` derivation — built reproducibly via [bun2nix](https://github.com/nix-community/bun2nix) from `web/bun.lock` → `web/bun.nix`), `<repo>/web/dist`, or a generated placeholder stub (`$OUT_DIR/web-stub/__mold_placeholder`). The stub is detected at runtime and swapped for the inline "mold is running" page so a bare `cargo build` still produces a working binary. For SPA hot-iteration without Rust recompiles, `MOLD_WEB_DIR` (and the legacy `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, `<binary dir>/web`, `./web/dist` candidates) still take precedence over the embedded bundle — so `bun run dev` or a local `web/dist` can be swapped in without rebuilding Rust. Dev server: `bun run dev` (proxies `/api` + `/health` to `http://localhost:7680`; override with `MOLD_API_ORIGIN`). See `crates/mold-server/src/web_ui.rs` for the resolver + embed wiring and `web/README.md` for the frontend stack.

**Metadata DB at `MOLD_HOME/mold.db`** (override with `MOLD_DB_PATH`, disable with `MOLD_DB_DISABLE=1`). The `mold-db` crate (rusqlite + bundled SQLite, WAL mode) owns four tables (current schema: `SCHEMA_VERSION = 6`):

| Table | Purpose | Typed API |
|-------|---------|-----------|
| `generations` (v1) | One row per saved gallery file — full `OutputMetadata` + `file_mtime_ms`, `file_size_bytes`, `generation_time_ms`, `backend`, `hostname`, `format`, `source`. | `MetadataDb::upsert` / `get` / `list` / `delete` |
| `settings` (v3 + v6) | Typed KV for TUI state and user-facing preferences — `tui.theme`, `tui.last_model`, `tui.last_prompt`, `tui.negative_collapsed`, `tui.migrated_from_json` sentinel, `expand.*`, `generate.*` (room for future sections). v6 adds `profile TEXT NOT NULL DEFAULT 'default'` with PK `(profile, key)`. | `mold_db::Settings` (resolves active profile) / `Settings::for_profile(db, name)` |
| `model_prefs` (v4 + v6) | One row per resolved `(profile, model)` pair with the generation params the TUI last used for that model (width, height, steps, guidance, scheduler, seed_mode, batch, format, lora_path, lora_scale, expand, offload, strength, control_scale, frames, fps, last_prompt, last_negative). Keyed on `manifest::resolve_model_name(name)` so `flux-dev` and `flux-dev:q4` collapse to one row per profile. | `mold_db::ModelPrefs::load/save/list/delete` (active profile) + `load_in/save_in/list_in/delete_in/delete_all_in` (explicit profile) |
| `prompt_history` (v5) | Replaces `~/.mold/prompt-history.jsonl`. Bounded via `trim_to(N)` (TUI caps at 500). Indexed on `created_at_ms DESC` and `model`. | `mold_db::PromptHistory` |

Migrations are forward-only and tracked via SQLite's `PRAGMA user_version`; each migration runs in its own transaction and can be either pure SQL (`MigrationKind::Sql`) or a Rust rewrite of existing rows (`MigrationKind::Rust`, used by v2's `/tmp` canonicalization). See `crates/mold-db/src/migrations.rs` — append a new entry to `MIGRATIONS[]` with the next version number, bump `SCHEMA_VERSION`, and the framework handles the rest. The TUI auto-saves per-model prefs on every model switch (outgoing model's current params → `model_prefs` row, incoming model's row → `GenerateParams`). Legacy `tui-session.json` and `prompt-history.jsonl` are imported into the DB on first launch after upgrade (gated by the `tui.migrated_from_json` sentinel) and renamed to `<name>.migrated` as a one-release downgrade safety net.

Gallery writes still happen in both surfaces:

- **Server**: `crates/mold-server/src/queue.rs` `save_image_to_dir` / `save_video_to_dir` upserts after writing the file. `crates/mold-server/src/lib.rs` opens the DB in `AppState.metadata_db` (typed `Arc<Option<MetadataDb>>`) and spawns a background `MetadataDb::reconcile(output_dir)` on startup that imports new files, refreshes mtime/size on changed files, and prunes rows whose backing files have disappeared. `/api/gallery` (`routes.rs::list_gallery`) prefers the DB and only falls back to `scan_gallery_dir` when the DB is `None` or returns no rows for the dir. `DELETE /api/gallery/image/:filename` also calls `db.delete()`.
- **CLI**: `crates/mold-cli/src/metadata_db.rs` exposes a process-wide `OnceLock<Option<MetadataDb>>` plus `record_local_save`, called from `save_and_preview_image` and the video save path in `crates/mold-cli/src/commands/generate.rs` so `mold run --local` and the local-fallback path persist rows just like the server.
- **TUI**: `crates/mold-tui/src/gallery_scan.rs::scan_images_local` queries the DB first when present, falling back to the existing on-disk walk + embedded-metadata parser. Server-mode TUI gallery already goes through `/api/gallery`.

The DB is opt-out, additive (PNG/JPEG embedded `mold:parameters` chunks still get written), and fail-safe — if open or upsert fails the surface logs a warning and keeps working without persistence.

**Gallery scan validates files at list time.** `/api/gallery` filters out corrupt/stub outputs before returning — files below a format-specific size floor (128 B for GIF, 256 B for other raster formats, 4 KB for MP4), files with invalid headers (image crate `into_dimensions()` fails, or MP4 missing the `ftyp` atom), and **solid-black raster outputs** (suspect-size file whose 16×16 sample stays under the intensity ceiling) are skipped entirely rather than shown as broken tiles in the UI. Header validation is cheap; the solid-black check only decodes files under a per-format "suspect size" so it never spends time on real multi-KB outputs. Helpers live in `routes.rs`: `min_valid_size`, `image_header_dims`, `has_ftyp_box`, `is_probably_solid_black`, `scan_gallery_dir`.

**MP4 thumbnails use openh264.** `get_gallery_thumbnail` extracts the first frame of MP4 gallery items via `mold_inference::ltx2::media::extract_thumbnail` (the same openh264 pipeline used by the video probe), resizes to 256 px max through the `image` crate, and caches the result at `~/.mold/cache/thumbnails/<name>.png`. Thumbnails for all formats now land under the same `.png`-suffixed cache path — callers should use `thumb_dir.join(format!("{name}.png"))`. Decode failures fall back to the inline SVG play-icon placeholder so the SPA never renders a truly broken poster.

**Gallery delete is opt-in.** `DELETE /api/gallery/image/:filename` returns `403 Forbidden` unless `MOLD_GALLERY_ALLOW_DELETE=1` is set on the server. `GET /api/capabilities` returns `{ gallery: { can_delete: bool } }` so the SPA can hide the delete affordance instead of inviting a 403. Pair the env toggle with the existing `MOLD_API_KEY` when the server is reachable from outside localhost.

**Web gallery view modes.** The SPA's top bar toggles between Feed (single-column Instagram/Tumblr-style stream, default, persisted in `localStorage`) and Grid (dense masonry, 2→6 columns). `GalleryCard` accepts a `variant` prop (`"feed" | "grid"`) that changes its caption layout; `GalleryFeed` picks page size based on mode (40 for feed, 150 for grid) since feed cards are taller. `<video>` elements use the full-file URL for `src` and the thumbnail URL as the static `poster` — never swap these, a `<video>` cannot decode a PNG.

**Detail drawer has two layouts.** Below `lg` the drawer renders as a fullscreen swipe-navigable media viewer: vertical swipes step through the filtered list (**down = next**, **up = previous**; horizontal drift > 80 px cancels), the metadata panel is tucked into a bottom sheet toggled by an info button, and the top bar shows close / counter / details toggle. At `lg+` the drawer falls back to the media pane + always-visible right sidebar. The metadata body is shared between both layouts via `web/src/components/Metadata.vue` so the copy/UI stays in sync. Keyboard: Esc / ← / → / ↑ ↓ / j k / i (toggle sheet).

**Touch-action strategy for the drawer.** `.drawer-root { touch-action: none }` suppresses Chrome's built-in pan/zoom so the swipe-nav `touchmove` stream isn't swallowed. Descendants marked `data-swipe-ignore` (the top bar buttons, native `<video>`, the open metadata sheet) get `touch-action: pan-y` back so they can still be tapped/scrolled natively — and the JS handler filters the same `data-swipe-ignore` targets on `touchstart`, so browser scroll and our swipe nav never fight. At `lg+` everything reverts to `auto` since desktop doesn't use swipe nav.

**`/api/gallery/image/:filename` supports HTTP Range.** Required for `<video>` scrubbing on MP4 outputs. Helpers live in `routes.rs`: `parse_byte_range`, `serve_range`. Responses: 200 (full file, streamed, `Accept-Ranges: bytes`), 206 (partial, streamed, `Content-Range`), 416 (unsatisfiable, `Content-Range: bytes */<total>`). Single-range form only.

## Development Workflow

- **Use TDD (test-driven development).** For every bug fix and new feature, write a failing test that encodes the expected behaviour *before* changing the implementation. Red → Green → Refactor. Prefer unit tests that exercise the exported contract (key→action mapping, focus transitions, serialization round-trips, layout invariants) over end-to-end flows. For layout constants, add an assertion that the inner content area can fit the rendered row count — constants without a guarding test drift into bugs. It's fine to commit the test and the fix together on small changes; the discipline is writing the test first, not commit ceremony.

## Maintenance Notes

- **Keep `CHANGELOG.md` updated** — Follow [Keep a Changelog](https://keepachangelog.com/) format. Add entries under `[Unreleased]` when implementing features, fixes, or breaking changes. Group under Added/Changed/Fixed/Removed headings.
- **Keep `.claude/skills/mold/SKILL.md` in sync** — This skill file is used by OpenClaw, ClawdBot, and other AI agents. Update it whenever models, CLI flags, env vars, or features change.
- **Keep `website/` docs in sync** — Update the VitePress docs site when models, CLI flags, env vars, API endpoints, or deployment options change.
- **Preserve centered gallery thumbnails in the TUI** — `crates/mold-tui/src/ui/gallery.rs` must keep using the fixed-protocol thumbnail path for the gallery grid. Do not switch grid thumbnails back to pure `StatefulImage` rendering for Kitty/Sixel/iTerm2 terminals; that reintroduces top-left-padded thumbnails instead of properly centered, aspect-correct ones. Keep the gallery thumbnail regression tests passing when touching this code.

## Key Design Decisions

1. **Workspace crate separation** — core/inference/server/cli have clean dependency boundaries. CLI doesn't need candle, server doesn't need clap.
2. **candle over tch/ort** — Pure Rust, first-class FLUX support, no libtorch dependency. CUDA, Metal, and CPU backends. Uses a published fork (`candle-*-mold` on crates.io) to fix Metal quantized matmul precision and seed buffer size bugs.
3. **Single binary** — `mold` includes `serve` via `mold-server` library. GPU feature flags (`cuda`/`metal`) forward through `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` for engine state** — Async-aware mutex; single-model-at-a-time is appropriate for GPU workloads. Inference runs in `spawn_blocking`.
5. **Smart VRAM management** — Dynamic device placement + drop-and-reload + quantized encoder auto-fallback. See `device.rs` for thresholds.
6. **Model pull via hf-hub** — rustls TLS (no OpenSSL), shared components deduplicated by hf-hub cache, `Progress` trait adapter bridges to `indicatif::ProgressBar`.
7. **Nix flake (flake-parts + crane)** — Pure Nix Rust builds, numtide devshell, treefmt-nix. CUDA 12.8 on Linux, Metal on macOS. Default `CUDA_COMPUTE_CAP=89` (Ada/RTX 4090); `packages.x86_64-linux.mold-sm120` for Blackwell (RTX 50-series). `mkMold` helper builds for any compute capability. Devshell sets `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` for CUDA compilation.
8. **Shell completions** — Static via `clap_complete` + dynamic via `CompleteEnv` with `ArgValueCandidates` for model names.
9. **Pipe-friendly output** — `IsTerminal` detection routes image bytes to stdout, status to stderr. SIGPIPE reset to default. `status!` macro handles routing.
10. **Unified `run` command** — First positional arg disambiguated at runtime: known model name vs prompt text.
11. **CPU-based noise generation** — Initial denoising noise is generated on CPU with a deterministic Rust RNG (`StdRng`/ChaCha20), then moved to GPU. This ensures same seed produces identical images across CUDA, Metal, and CPU backends. See `seeded_randn()` in `engine.rs`.
12. **LoRA via custom VarBuilder backend** — candle has no built-in LoRA support. For BF16 models, a custom `SimpleBackend` (`LoraBackend`) wraps mmap'd safetensors and intercepts `vb.get()` calls during model construction — each tensor loads from mmap → device, and LoRA deltas (`B @ A`) are applied inline. For GGUF models, `gguf_lora_var_builder()` selectively dequantizes LoRA-affected tensors to F32 on CPU, merges deltas, and re-quantizes back to the original GGML dtype. Non-LoRA tensors stay quantized. Works with block-level offloading by targeting CPU device. `LoraDeltaCache` caches computed deltas on CPU across rebuilds; `LoraFingerprint` tracks the active LoRA to skip redundant drops.
