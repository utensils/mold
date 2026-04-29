# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# mold — Architecture & Development Guide

> Local AI image/video generation CLI — FLUX, SD3.5, SD1.5, SDXL, Z-Image, Flux.2, Qwen-Image, Qwen-Image-Edit, Wuerstchen, LTX Video, & LTX-2 models on your GPU.

mold is a CLI for AI image and short-video generation across 11 model families (80+ manifest variants) via the [candle](https://github.com/huggingface/candle) ML framework. It bundles a local inference server and a client CLI that can generate locally or against a remote `MOLD_HOST`. Supports txt2img, img2img, multimodal edit, inpainting, and ControlNet conditioning.

## Build & Development Commands

### Nix (preferred)

```bash
nix build                  # Build mold (default, includes serve with GPU)
nix run                    # Run mold CLI
nix develop                # Enter devshell (auto via direnv)
nix fmt                    # Format Nix + Rust (nixfmt + rustfmt)
nix flake check            # Validate formatting + flake
```

### Devshell commands (inside `nix develop`)

| Category | Command | Description |
|----------|---------|-------------|
| build | `build` | Fast local `mold` build (`cargo build --profile dev-fast -p mold-ai`) with embedded web bundle |
| build | `build-workspace` | `cargo build` (debug, all crates) |
| build | `build-release` | Shipping `cargo build --release -p mold-ai --features {gpu},preview,discord,expand,tui,webp,mp4,metrics` |
| build | `build-server` | Fast local single-binary server build with GPU + preview + expand and embedded web UI |
| build | `build-discord` | Fast local `cargo build --profile dev-fast -p mold-ai --features discord` |
| check | `check` / `clippy` / `fmt` / `fmt-check` / `run-tests` | Standard cargo wrappers (`run-tests` not `test` to avoid shadowing builtins) |
| check | `coverage` | Test coverage report (`--html` for browsable report) |
| run | `mold` / `serve` / `generate` / `discord-bot` | Run the CLI / server / generate helper / Discord bot |

### Cargo (direct)

```bash
cargo build [--profile dev-fast] [--release] [-p mold-ai] [--features cuda|metal,...]
cargo check / clippy / fmt --check / test [-p mold-ai-core]
./scripts/coverage.sh [--html]            # Test coverage summary / HTML report
./scripts/fetch-tokenizers.sh             # Pre-download tokenizer files
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai --features metal,preview,expand -- run "a cat"
./scripts/ensure-web-dist.sh && cargo run --profile dev-fast -p mold-ai --features metal,preview,expand -- serve
cargo run -p mold-ai-inference --features dev-bins --bin ltx2_review -- clip.mp4
```

### CI (GitHub Actions, `.github/workflows/ci.yml`) — all jobs must pass

| Job | What it checks |
|-----|----------------|
| `rust` | `cargo fmt --all -- --check && cargo check --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4` |
| `coverage` | `cargo llvm-cov` → Codecov upload |
| `docs` | `bun run fmt:check && bun run verify && bun run build` in `website/` |

> **Note:** `mold-inference` and `mold-server` set `[lib] test = false` (the `mold-server` binary too). The candle/CUDA test harness triggers ~32GB RAM + 40+ min weight init. Unit tests in `mold-core` and `mold-cli` run normally. To test those crates, temporarily drop `test = false` and run `cargo test -p <crate> --lib`.

## Project Vision

- **Local-first**: Diffusion runs on your GPU; **remote-capable** via `MOLD_HOST`; **cloud** via `mold runpod` (creates pod → generates → saves to `./mold-outputs/` in one command).
- **Model management**: `mold pull`, `mold list`, load/unload via API.
- **Simple CLI**: `mold run flux-dev:q4 "a cat"` — first positional arg auto-disambiguated as model vs prompt.
- **Pipe-friendly**: `mold run "a cat" | viu -`.
- **Future**: OCI registry for model distribution.

## Crate Structure

```
crates/
├── mold-core/        # Shared types, API protocol, HTTP client, config, manifest      → -p mold-ai-core      (lib mold_core)
├── mold-db/          # SQLite metadata DB (rusqlite bundled): gallery, settings, prefs, history → -p mold-ai-db (lib mold_db)
├── mold-inference/   # Candle-based engines (11 families)                              → -p mold-ai-inference (lib mold_inference)
├── mold-server/      # Axum HTTP inference server (library, consumed by mold-cli)      → -p mold-ai-server    (lib mold_server)
├── mold-cli/         # Main `mold` binary (clap), single binary with feature flags    → -p mold-ai           (binary mold)
├── mold-discord/     # Discord bot library (feature `discord`)                         → -p mold-ai-discord   (lib mold_discord)
└── mold-tui/         # Interactive terminal UI (feature `tui`)                         → -p mold-ai-tui
```

Directory names differ from Cargo package names — use `-p <package>` from the table above. **MSRV**: 1.85.

**Feature flags (`mold-cli`):** `cuda`, `metal`, `preview` (terminal image display), `discord` (`mold discord` + `mold serve --discord`), `expand` (local LLM prompt expansion), `tui` (`mold tui`), `metrics` (`/metrics` endpoint).

**Feature flags (`mold-inference`):** `cuda`, `metal`, `expand`, `webp` (animated WebP via libwebp FFI), `mp4` (AAC encoding for MP4 muxing on native video paths), `dev-bins` (native LTX-2 review/probe binaries). H.264/MP4 decode is in the baseline LTX-2 source-video stack — `mp4` no longer gates the full video media graph; GIF/APNG output works without it.

### mold-core

- **`types.rs`** — API request/response types: `GenerateRequest`, `GenerateResponse`, `ModelInfo`, `ServerStatus`, `ExpandRequest`, `ExpandResponse`, `LoraWeight`, etc.
- **`client.rs`** — `MoldClient` HTTP client; `is_connection_error()` for local fallback detection; `expand_prompt()` for server-side expansion.
- **`config.rs`** — `Config`, `ModelConfig`, `ModelPaths`; loads from `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy if `~/.mold/` exists); per-model `lora` path and `lora_scale` defaults.
- **`manifest.rs`** — `ModelManifest` registry of downloadable models with HF sources; `resolve_model_name()` for `name:tag` resolution; `UTILITY_FAMILIES` constant + `is_utility()` for non-diffusion models (e.g. `qwen3-expand`).
- **`expand.rs`** — `PromptExpander` trait, `ApiExpander` (OpenAI-compatible HTTP), `ExpandConfig`/`ExpandSettings`/`FamilyOverride` settings with `MOLD_EXPAND_*` env support; user-configurable system prompts and per-family word limits/style notes.
- **`download.rs`** — `pull_model()` wraps `hf-hub` with progress bars; SHA-256 integrity verification (fails on mismatch, `--skip-verify` to override); `.pulling` marker for atomic-pull detection; `PullOptions`.
- **`validation.rs`** — `validate_generate_request()` (shared by server + CLI); `fit_to_model_dimensions()` (aspect-ratio-preserving resize to model-native resolution for img2img); LoRA scale [0.0, 2.0] + `.safetensors` extension validation.
- **`error.rs`** — `MoldError` (thiserror).

### mold-inference

Eleven engines implementing the `InferenceEngine` trait:

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

`create_engine()` in `factory.rs` resolves the family (config override → manifest → default `flux`) and dispatches:

- `"flux"` → `FluxEngine` — T5 + CLIP-L, flow-matching transformer, VAE.
- `"sd15"` (also `"sd1.5"`, `"stable-diffusion-1.5"`) → `SD15Engine` — CLIP-L, UNet w/ DDIM, CFG, 512×512 default.
- `"sdxl"` → `SDXLEngine` — Dual CLIP-L+CLIP-G, UNet w/ DDIM/Euler-A, CFG.
- `"sd3"` (also `"sd3.5"`, `"stable-diffusion-3"`/`-3.5"`) → `SD3Engine` — CLIP-L + CLIP-G + T5-XXL, quantized MMDiT w/ NaN-safe inference.
- `"flux2"` (also `"flux.2"`, `"flux2-klein"`) → `Flux2Engine` — Qwen3 encoder (BF16 or GGUF, layers 9/18/27), shared-modulation transformer, BN-VAE.
- `"qwen-image"` (also `"qwen_image"`) → `QwenImageEngine` — Qwen2.5-VL encoder, 3D causal VAE (2D temporal slice), flow-matching CFG.
- `"qwen-image-edit"` → `QwenImageEngine` (multimodal-edit config) — same engine class as `qwen-image`; repeatable `--image` source/reference inputs and negative prompts.
- `"z-image"` → `ZImageEngine` — Qwen3 encoder, flow-matching transformer w/ 3D RoPE.
- `"wuerstchen"` (also `"wuerstchen-v2"`) → `WuerstchenEngine` — CLIP-G, 3-stage cascade (Prior → Decoder → VQ-GAN), 42× latent compression.
- `"ltx-video"` → `LtxVideoEngine` — T5-XXL, flow-matching transformer, 3D causal VAE; APNG/GIF/WebP/MP4. Checkpoints `0.9.6` and `0.9.8`; `0.9.8` pulls the spatial upscaler asset and now runs the full multiscale refinement path.
- `"ltx2"` (also `"ltx-2"`) → `Ltx2Engine` — Gemma-3 encoder w/ LTX-2 connectors, joint audio-video DiT, 3D causal video VAE, audio VAE + vocoder, MP4-first w/ real AAC. Covers 19B/22B text+audio-video, image-to-video, audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale (`x1.5`/`x2`), temporal upscale (`x2`). **CUDA-only for real generation** (CPU is correctness fallback; Metal explicitly unsupported). Native runtime in `ltx2/`: orchestration, guidance/perturbation paths, stacked LoRAs + camera-control presets, native MP4/GIF/APNG/WebP media pipeline.

**Additional modules:** `encoders/variant_resolution.rs` (T5/Qwen3 auto-fallback quantization); `scheduler.rs` (DDIM, Euler-A, UniPC for SD1.5/SDXL); `img_utils.rs` (decode/resize/mask for img2img/inpainting); `controlnet/` (UNet encoder copy + zero convs for SD1.5); `weight_loader.rs::load_safetensors_with_progress()` replaces opaque `VarBuilder::from_mmaped_safetensors()` with per-tensor loading that emits `WeightLoad` events.

**Architectural patterns:**

- **Lazy loading** — Engines load on first generation, not at startup.
- **mmap safetensors** — OS handles paging.
- **BF16/GGUF dual support** — Each engine's transformer wraps both (e.g. `FluxTransformer` enum); auto-detected by `.gguf` extension.
- **Drop-and-reload encoders** — T5/CLIP (FLUX) and Qwen3 (Z-Image) drop from GPU after encoding to free VRAM, reload next request.
- **Dynamic device placement** — Encoders go to GPU or CPU based on remaining VRAM after transformer load (thresholds in `device.rs`).
- **Quantized encoder auto-fallback** — When FP16/BF16 doesn't fit, picks largest quantized GGUF that does. Custom `GgufT5Encoder` and `GgufQwen3Encoder` in `encoders/` handle GGUF tensor naming. Override with `--t5-variant` / `--qwen3-variant`.
- **Block-level offloading** (`flux/offload.rs`) — Streams transformer blocks CPU↔GPU one at a time (~24GB → 2-4GB, 3-5× slower). Auto-enabled when VRAM is short; force with `--offload` / `MOLD_OFFLOAD=1`.
- **LoRA adapter support** (`flux/lora.rs`) — Custom `SimpleBackend` (`LoraBackend`) wraps mmap'd base weights and applies LoRA deltas inline during model construction. Parses diffusers-format LoRA safetensors (A/B + alpha), maps keys to candle's fused tensors (QKV, linear1), merges via `W' = W + scale * (B @ A)`. Compatible with offloading — patches bake into blocks on CPU then stream to GPU.
- **LoRA fingerprint caching** — `FluxEngine` tracks `active_lora: Option<LoraFingerprint>` (path hash + scale) to skip redundant transformer rebuilds when the same LoRA recurs. Cleared on `unload()`/`load()`.
- **LoRA delta caching** — `LoraDeltaCache` in `flux/lora.rs` caches pre-computed `B @ A * scale` deltas on CPU (~80-120 MB for FLUX). Cache keys include `patch_index` to disambiguate fused QKV slices. Survives across rebuilds within an engine lifetime.
- **Shared tokenizer pool** (`shared_pool.rs`) — Cross-engine `SharedPool` caches T5/CLIP tokenizers via `Arc<Tokenizer>` keyed by file path. All FLUX variants share files, so model switches skip ~100-150 ms init. Threaded through `create_engine_with_pool()`.

**Z-Image GGUF specifics** — Quantized transformer in `zimage/quantized_transformer.rs` (candle has no quantized Z-Image model). GGUF tensor name diffs vs BF16: fused `attention.qkv` vs separate Q/K/V; `x_embedder` vs `all_x_embedder.2-1`; etc.

**Prompt expansion** (`expand.rs`, `expand` feature) — `LocalExpander` wraps `candle_transformers::models::quantized_qwen3::ModelWeights` for local GGUF text generation. Implements `PromptExpander` from `mold-core`. Includes progress reporting, VRAM-aware placement (`should_use_gpu()`), Darwin memory-safety guards (`preflight_memory_check()`). The LLM is always dropped before diffusion runs.

### mold-server

> Feature flags: `cuda`, `metal`, `expand` (forwarded to mold-inference), `metrics` (Prometheus + instrumentation).

Axum HTTP server wrapping the inference engine. Library consumed by `mold-cli` (`mold serve`).

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
| `GET` | `/api/gallery/image/:name` | Fetch a saved image (Range-aware) |
| `DELETE` | `/api/gallery/image/:name` | Delete a saved image (opt-in) |
| `GET` | `/api/gallery/thumbnail/:name` | Fetch a cached thumbnail |
| `POST` | `/api/upscale` | Upscale image with Real-ESRGAN |
| `POST` | `/api/upscale/stream` | Upscale with SSE tile progress streaming |
| `POST` | `/api/shutdown` | Trigger graceful server shutdown |
| `GET` | `/api/status` | Server health + status |
| `GET` | `/health` | Simple 200 OK |
| `GET` | `/api/openapi.json` | OpenAPI spec |
| `GET` | `/api/docs` | Interactive API docs (Scalar) |
| `GET` | `/metrics` | Prometheus metrics (feature-gated) |
| `GET` | `/api/capabilities/chain-limits` | Per-model chain frame caps + supported transitions |

State: `AppState` with `tokio::sync::Mutex<ModelCache>` (LRU, max 3 models). `ModelResidency` per engine: `Gpu`, `Parked` (weights dropped, tokenizers/caches retained for fast reload), or `Unloaded`. At most one engine is GPU-resident at a time. `AppState` also holds `shared_pool: Arc<Mutex<SharedPool>>` and `upscaler_cache: Arc<std::sync::Mutex<Option<Box<dyn UpscaleEngine>>>>`.

### mold-cli

Main binary. `cuda`/`metal` features forward through `mold-server` → `mold-inference` for both `mold serve` and `mold run --local`.

### mold-discord

Discord bot using **poise 0.6 + serenity 0.12**. Depends only on `mold-core` (no GPU). Connects to a running `mold serve` via `MoldClient` HTTP/SSE. Slash commands: `/generate`, `/expand`, `/models`, `/status`, `/quota`, `/admin`. Invoked via `mold discord` (standalone) or `mold serve --discord` (combined).

Modules: `commands/` (handlers — `generate`, `expand`, `models`, `status`, `quota`, `admin`), `handler.rs` (SSE streaming orchestration), `format.rs` (`format_expand_result()`, `format_quota()`), `cooldown.rs` (per-user rate limit), `quota.rs` (per-user daily quota), `access.rs` (RBAC + block list), `checks.rs` (shared auth checks), `state.rs` (shared bot state).

Token env: `MOLD_DISCORD_TOKEN` (preferred) or `DISCORD_TOKEN` (fallback).

## CLI Quick Reference

Core commands: `mold run`, `mold serve`, `mold pull`, `mold list`, `mold ps`, `mold tui`, `mold chain`, `mold upscale`, `mold expand`, `mold config`, `mold update`, `mold runpod`. See `mold --help` and `website/deployment/runpod-cli.md` for full details.

**Key behaviors:**
- `mold run [MODEL] [PROMPT]` — first positional arg is MODEL if it matches a known name, otherwise prompt.
- **Pipe-friendly**: stdin → prompt; stdout → image bytes when not a TTY. `--output -` forces stdout. `--image -` reads source from stdin.
- **Inference modes**: remote (default → `MOLD_HOST`) → local fallback (auto if server unreachable, with auto-pull) → `--local` (forced local GPU, skip server).
- **VRAM**: `--offload` streams blocks (~24GB → 2-4GB, slower); `--eager` keeps everything resident; `--t5-variant` / `--qwen3-variant` / `--qwen2-variant` control encoder quantization.

### Multi-prompt chain authoring

- `mold run --script shot.toml` — canonical TOML chain (schema `mold.chain.v1`), per-stage `prompt`/`frames`/`transition`.
- `mold run --script shot.toml --dry-run` — print normalised stages + total frames, exit.
- `mold chain validate shot.toml` — parse + normalise without submitting.
- Sugar: `mold run <model> --prompt "..." --prompt "..." --frames-per-clip 97` (uniform smooth only).
- Transitions: `smooth` (default, motion-tail morph), `cut` (fresh latent), `fade` (cut + RGB crossfade).
- Per-stage starting images: each `[[stage]]` accepts `source_image_path = "./hero.png"` (resolved relative to script, base64-encoded at load) or `source_image_b64 = "<base64>"`. Equivalent to the canonical `source_image` field; setting more than one per stage is a validation error. Resolved by `mold_core::chain_toml::read_script_resolving_paths`, used by both `mold chain validate` and `mold run --script`.
- Web composer: `/generate` has `Single` | `Script` toggle; each stage has its own 🖼️ attach/clear. localStorage drafts strip base64 bytes (filenames only) for the 5-10 MB quota.
- TUI: `s` from hub opens Script mode.

## Environment Variables

All vars prefixed `MOLD_`. Env overrides config file.

- **Server/client**: `MOLD_HOST` (default `http://localhost:7680`), `MOLD_DEFAULT_MODEL` (default `flux2-klein`), `MOLD_MODELS_DIR` (default `~/.mold/models`), `MOLD_OUTPUT_DIR` (default `~/.mold/output`), `MOLD_PORT` (default `7680`), `MOLD_LOG` (default `warn` CLI / `info` server).
- **Component paths**: `MOLD_TRANSFORMER_PATH`, `MOLD_VAE_PATH`, `MOLD_T5_PATH`, `MOLD_CLIP_PATH`, `MOLD_CLIP2_PATH` + tokenizer variants.
- **Encoder variants**: `MOLD_T5_VARIANT`, `MOLD_QWEN3_VARIANT`, `MOLD_QWEN2_VARIANT`.
- **Runtime toggles**: `MOLD_OFFLOAD`, `MOLD_EAGER`, `MOLD_PREVIEW`, `MOLD_EXPAND`, `MOLD_EMBED_METADATA`.
- **Server security**: `MOLD_API_KEY`, `MOLD_RATE_LIMIT`, `MOLD_CORS_ORIGIN`.
- **Discord**: `MOLD_DISCORD_TOKEN`, `MOLD_DISCORD_COOLDOWN`, `MOLD_DISCORD_DAILY_QUOTA`.
- **Debug**: `MOLD_QWEN_DEBUG`, `MOLD_SD3_DEBUG`.

Full list in source or `mold --help`.

## Config File

Location: `~/.config/mold/config.toml` (XDG) or `~/.mold/config.toml` (legacy if `~/.mold/` exists). Structure in `crates/mold-core/src/config.rs`.

**After issue #265**, the surface splits across two stores behind one logical `Config`:

| Surface | Owns | Keys |
|---|---|---|
| `config.toml` (bootstrap) | paths, ports, credentials | `default_model`, `models_dir`, `output_dir`, `server_port`, `gpus`, `queue_size`, `[logging]`, `[runpod]`, per-model `transformer`/`vae`/encoder/tokenizer paths |
| `mold.db` `settings` table | user preferences | `expand.*`, `generate.*`, `tui.*`, legacy `last-model` sidecar |
| `mold.db` `model_prefs` table | per-model gen defaults | `default_steps`, `default_guidance`, `default_width`, `default_height`, `scheduler`, `negative_prompt`, `lora`, `lora_scale` |
| `MOLD_*` env vars | runtime override (highest precedence) | all of the above at read time |

Every binary's `fn main()` (`mold`, `mold-server`, `mold-discord`) calls `mold_db::config_sync::install_config_post_load_hook()`, which wires `Config::install_post_load_hook()` into `mold-core`. First boot runs an idempotent `config.toml → DB` import (gated by `config.migrated_from_toml` in `settings`), renames the original to `config.toml.migrated`, and rewrites the file via `Config::save_bootstrap_only` (bootstrap fields only). Subsequent loads overlay DB values onto every `Config::load_or_default()` — consumers keep reading `cfg.expand.*` / `cfg.default_width` / `cfg.models[…]` unchanged. Helpers in `mold-db::config_sync`: `save_expand_to_db`, `save_generate_globals_to_db`, `migrate_config_toml_to_db`, `hydrate_config_from_db`, `rewrite_stripped_config_toml`, `detect_stale_backups`, `cleanup_stale_backups`. `mold pull` auto-writes model path entries to TOML.

`mold config` routes by key prefix:

- `set expand.enabled true` → DB. `set models_dir /mnt/...` → `config.toml`.
- `where <key>` → prints `file` or `db`, reports any env-var override.
- `list` / `--json` → rows tagged `[db]` / `[file]` / `[env]`; JSON returns `{ value, surface }` per key.
- `reset <key>` → drops DB row so next read falls back; TOML-only keys are rejected with a pointer to `set`.
- `reset --all [--yes]` → drops every DB-backed row under the active profile.
- `--profile <name> <subcommand>` → operate on an explicit profile (overrides `MOLD_PROFILE`).

**Multi-profile (v6 schema).** `settings` and `model_prefs` are keyed on `(profile, key)` / `(profile, model)`. Active profile resolves: `MOLD_PROFILE` env → `settings.profile.active` (under `default`) → `"default"`. `Settings::new(db)` / `ModelPrefs::load(db, model)` use the resolved active profile; `Settings::for_profile(db, name)` and `ModelPrefs::load_in/save_in/list_in/delete_in` take an explicit profile.

## Model System

**Name resolution**: `model:tag` (e.g. `flux-dev:q4`). Bare names try `:q8` → `:fp16` → `:bf16` → `:fp8` and pick the first match. Legacy dash form (`flux-dev-q4`) resolves to colon form. See `manifest.rs` for the full registry (80+ variants) and HF sources.

**`mold run` inference modes:** remote (default, HTTP) → local fallback (auto if server unreachable, with auto-pull) → `--local` (force, skip server).

> **VRAM**: Full BF16 FLUX dev (23 GB) auto-offloads on 24 GB cards. GGUF quantized models fit without offloading. SDXL FP16 UNets (~5 GB) fit comfortably. LoRA works with both BF16 (via offload path) and GGUF quantized FLUX.

## Deployment

- **Nix CUDA**: `nix build .#mold` (Ada/sm_89) or `nix build .#mold-sm120` (Blackwell/sm_120).
- **Systemd**: `~/.config/systemd/user/mold-server.service` — set `LD_LIBRARY_PATH=/run/opengl-driver/lib` for NixOS CUDA driver access.
- **Remote**: GPU host runs `mold serve --port 7680`; clients set `MOLD_HOST=http://gpu-host:7680`.
- **Docker / RunPod**: Multi-stage `Dockerfile` at root. Stage 1: `nvidia/cuda:12.8.1-devel-ubuntu22.04` + Rust + cargo (`--features cuda,expand`). Stage 2: copies binary into `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3.4 GB image, 33 MB binary). `CUDA_COMPUTE_CAP` build arg defaults to `89`. `docker/start.sh` is the RunPod entrypoint: detects `/workspace`, sets `MOLD_HOME`/`MOLD_MODELS_DIR`, runs `mold serve --bind 0.0.0.0`. `libcuda.so.1` is injected at runtime by NVIDIA Container Toolkit — the binary cannot run without GPU access.
- **Documentation site**: VitePress 1.6 + Tailwind v4 + bun in `website/`. Dev: `bun run dev -- --host 0.0.0.0`. Build: `bun run build`. Deployed to GitHub Pages via `.github/workflows/pages.yml` on push to `main` (`website/**`). Base path `/mold/` (served at `utensils.github.io/mold/`).

### Web gallery UI

Vue 3 + Vite 7 + Tailwind v4.2 SPA in `web/`. **Embedded into the `mold` binary at compile time** via [`rust-embed`](https://crates.io/crates/rust-embed) — `nix build` produces a single-file server. Local devshell `build`/`build-server`/`mold`/`serve`/`generate` call `./scripts/ensure-web-dist.sh` first so `target/dev-fast/mold` ships the real SPA, not the placeholder.

`crates/mold-server/build.rs` resolves the bundle in order: `$MOLD_WEB_DIST` (set by the Nix `mold-web` derivation, built reproducibly via [bun2nix](https://github.com/nix-community/bun2nix) from `web/bun.lock` → `web/bun.nix`) → `<repo>/web/dist` → generated placeholder stub at `$OUT_DIR/web-stub/__mold_placeholder` (detected at runtime, swapped for the inline "mold is running" page so a bare `cargo build` works).

For SPA hot-iteration without Rust rebuilds, `MOLD_WEB_DIR` (and legacy `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, `<binary dir>/web`, `./web/dist`) takes precedence over the embed. SPA dev: `bun run dev` (proxies `/api` + `/health` to `http://localhost:7680`; override with `MOLD_API_ORIGIN`). See `crates/mold-server/src/web_ui.rs` and `web/README.md`.

### Metadata DB

At `MOLD_HOME/mold.db` (override `MOLD_DB_PATH`, disable `MOLD_DB_DISABLE=1`). `mold-db` crate (rusqlite + bundled SQLite, WAL). Schema `SCHEMA_VERSION = 6`:

| Table | Purpose | Typed API |
|-------|---------|-----------|
| `generations` (v1) | One row per saved gallery file — full `OutputMetadata` + `file_mtime_ms`, `file_size_bytes`, `generation_time_ms`, `backend`, `hostname`, `format`, `source` | `MetadataDb::upsert/get/list/delete` |
| `settings` (v3 + v6) | Typed KV — `tui.theme`, `tui.last_model`, `tui.last_prompt`, `tui.negative_collapsed`, `tui.migrated_from_json` sentinel, `expand.*`, `generate.*`. v6 adds `profile TEXT NOT NULL DEFAULT 'default'`, PK `(profile, key)` | `mold_db::Settings` (active profile) / `Settings::for_profile(db, name)` |
| `model_prefs` (v4 + v6) | One row per resolved `(profile, model)` — width, height, steps, guidance, scheduler, seed_mode, batch, format, lora_path, lora_scale, expand, offload, strength, control_scale, frames, fps, last_prompt, last_negative. Keyed on `manifest::resolve_model_name(name)` so `flux-dev` and `flux-dev:q4` collapse | `ModelPrefs::load/save/list/delete` (active profile) + `load_in/save_in/list_in/delete_in/delete_all_in` (explicit profile) |
| `prompt_history` (v5) | Replaces `~/.mold/prompt-history.jsonl`. Bounded via `trim_to(N)` (TUI caps at 500). Indexed on `created_at_ms DESC`, `model` | `mold_db::PromptHistory` |

Migrations are forward-only via SQLite's `PRAGMA user_version`; each runs in its own transaction as either `MigrationKind::Sql` or `MigrationKind::Rust` (used by v2's `/tmp` canonicalization). Append a new entry to `MIGRATIONS[]` in `crates/mold-db/src/migrations.rs`, bump `SCHEMA_VERSION`. The TUI auto-saves per-model prefs on every model switch (outgoing → row, incoming row → `GenerateParams`). Legacy `tui-session.json` and `prompt-history.jsonl` are imported on first launch (gated by `tui.migrated_from_json`) and renamed to `<name>.migrated` as a one-release downgrade safety net.

Gallery writes happen in both surfaces:

- **Server** — `mold-server/src/queue.rs` `save_image_to_dir`/`save_video_to_dir` upsert after writing. `mold-server/src/lib.rs` opens the DB into `AppState.metadata_db` (`Arc<Option<MetadataDb>>`) and spawns a background `MetadataDb::reconcile(output_dir)` on startup that imports new files, refreshes mtime/size on changed files, and prunes rows whose backing files are gone. `/api/gallery` (`routes.rs::list_gallery`) prefers the DB and only falls back to `scan_gallery_dir` when DB is `None` or empty. `DELETE /api/gallery/image/:filename` also calls `db.delete()`.
- **CLI** — `mold-cli/src/metadata_db.rs` exposes `OnceLock<Option<MetadataDb>>` + `record_local_save`, called from `save_and_preview_image` and the video save path in `mold-cli/src/commands/generate.rs` so `mold run --local` and the local-fallback path persist rows.
- **TUI** — `mold-tui/src/gallery_scan.rs::scan_images_local` queries the DB first, falling back to disk walk + embedded-metadata parser. Server-mode TUI gallery already goes via `/api/gallery`.

The DB is opt-out, additive (PNG/JPEG embedded `mold:parameters` chunks still written), and fail-safe — open/upsert failures log a warning and keep working without persistence.

### Gallery quality + delivery

- **Gallery scan validates at list time.** `/api/gallery` filters out corrupt/stub outputs: files below format-specific size floor (128 B GIF, 256 B other raster, 4 KB MP4); files with invalid headers (image crate `into_dimensions()` fails, MP4 missing `ftyp`); **solid-black raster outputs** (suspect-size file whose 16×16 sample stays under intensity ceiling). Header check is cheap; solid-black only decodes files under a per-format "suspect size" so it never spends time on real multi-KB outputs. Helpers in `routes.rs`: `min_valid_size`, `image_header_dims`, `has_ftyp_box`, `is_probably_solid_black`, `scan_gallery_dir`.
- **MP4 thumbnails use openh264.** `get_gallery_thumbnail` extracts the first frame via `mold_inference::ltx2::media::extract_thumbnail` (same openh264 path as the video probe), resizes to 256 px max via `image`, caches at `~/.mold/cache/thumbnails/<name>.png`. All formats land at the `.png`-suffixed cache path — callers should use `thumb_dir.join(format!("{name}.png"))`. Decode failures fall back to the inline SVG play-icon placeholder.
- **Gallery delete is opt-in.** `DELETE /api/gallery/image/:filename` returns `403 Forbidden` unless `MOLD_GALLERY_ALLOW_DELETE=1`. `GET /api/capabilities` returns `{ gallery: { can_delete: bool } }` so the SPA can hide the affordance instead of inviting a 403. Pair with `MOLD_API_KEY` when reachable from outside localhost.
- **Web gallery view modes.** Top bar toggles Feed (single-column, default, persisted in `localStorage`) and Grid (dense masonry, 2→6 columns). `GalleryCard` accepts `variant: "feed" | "grid"`; `GalleryFeed` page size is 40 (feed) / 150 (grid) since feed cards are taller. **`<video>` elements use the full-file URL for `src` and the thumbnail URL as the static `poster` — never swap these, a `<video>` cannot decode a PNG.**
- **Detail drawer has two layouts.** Below `lg`: fullscreen swipe-navigable viewer — vertical swipes step through the filtered list (**down = next**, **up = previous**; horizontal drift > 80 px cancels), metadata is in a bottom sheet toggled by an info button, top bar shows close / counter / details toggle. At `lg+`: media pane + always-visible right sidebar. Metadata body is shared via `web/src/components/Metadata.vue`. Keys: Esc / ← / → / ↑ ↓ / j k / i (toggle sheet).
- **Touch-action strategy.** `.drawer-root { touch-action: none }` suppresses Chrome pan/zoom so the swipe `touchmove` stream isn't swallowed. Descendants marked `data-swipe-ignore` (top bar buttons, native `<video>`, open metadata sheet) get `touch-action: pan-y` back; the JS handler also filters `data-swipe-ignore` on `touchstart` so browser scroll and swipe nav don't fight. At `lg+` everything reverts to `auto`.
- **HTTP Range** — `/api/gallery/image/:filename` supports Range; required for `<video>` scrubbing on MP4. Helpers in `routes.rs`: `parse_byte_range`, `serve_range`. Responses: 200 (full file, streamed, `Accept-Ranges: bytes`), 206 (partial, streamed, `Content-Range`), 416 (unsatisfiable, `Content-Range: bytes */<total>`). Single-range form only.

## Development Workflow

- **Use TDD.** For every bug fix and new feature, write a failing test that encodes the expected behaviour *before* changing the implementation — Red → Green → Refactor. Prefer unit tests on the exported contract (key→action mapping, focus transitions, serialization round-trips, layout invariants) over end-to-end flows. For layout constants, add an assertion that the inner content area can fit the rendered row count — constants without a guarding test drift into bugs. Commit test + fix together; the discipline is writing the test first, not commit ceremony.

## Maintenance Notes

- **Keep `CHANGELOG.md` updated** — [Keep a Changelog](https://keepachangelog.com/) format. Add entries under `[Unreleased]`, grouped Added/Changed/Fixed/Removed.
- **Keep `.claude/skills/mold/SKILL.md` in sync** — Used by OpenClaw, ClawdBot, and other AI agents. Update whenever models, CLI flags, env vars, or features change.
- **Keep `website/` docs in sync** — VitePress site mirrors models, CLI flags, env vars, API endpoints, and deployment options.
- **Preserve centered TUI gallery thumbnails** — `crates/mold-tui/src/ui/gallery.rs` must keep the fixed-protocol thumbnail path for the gallery grid. Do not switch grid thumbnails back to pure `StatefulImage` rendering for Kitty/Sixel/iTerm2 — that reintroduces top-left padding instead of properly centered, aspect-correct tiles. Keep the gallery thumbnail regression tests passing.

## Key Design Decisions

1. **Workspace crate separation** — clean dependency boundaries: CLI doesn't need candle, server doesn't need clap.
2. **candle over tch/ort** — Pure Rust, first-class FLUX, no libtorch. CUDA + Metal + CPU. Uses a published fork (`candle-*-mold` on crates.io) to fix Metal quantized-matmul precision and seed-buffer-size bugs.
3. **Single binary** — `mold` includes `serve` via `mold-server` library. GPU features (`cuda`/`metal`) forward `mold-cli` → `mold-server` → `mold-inference`.
4. **`tokio::sync::Mutex` for engine state** — Async-aware; single-model-at-a-time fits GPU workloads. Inference runs in `spawn_blocking`.
5. **Smart VRAM management** — Dynamic device placement + drop-and-reload + quantized encoder auto-fallback. Thresholds in `device.rs`.
6. **Model pull via hf-hub** — rustls TLS (no OpenSSL); shared components dedup'd by hf-hub cache; `Progress` trait adapter bridges to `indicatif::ProgressBar`.
7. **Nix flake (flake-parts + crane)** — Pure Nix Rust builds, numtide devshell, treefmt-nix. CUDA 12.8 on Linux, Metal on macOS. Default `CUDA_COMPUTE_CAP=89`; `packages.x86_64-linux.mold-sm120` for Blackwell. `mkMold` helper builds for any compute cap. Devshell sets `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` for CUDA compilation.
8. **Shell completions** — Static via `clap_complete`; dynamic via `CompleteEnv` + `ArgValueCandidates` for model names.
9. **Pipe-friendly output** — `IsTerminal` detection routes image bytes to stdout, status to stderr. SIGPIPE reset to default. `status!` macro handles routing.
10. **Unified `run` command** — First positional arg disambiguated at runtime: known model name vs prompt text.
11. **CPU-based noise generation** — Initial denoising noise is generated on CPU with a deterministic Rust RNG (`StdRng`/ChaCha20), then moved to GPU. Same seed produces identical images across CUDA, Metal, and CPU. See `seeded_randn()` in `engine.rs`.
12. **LoRA via custom VarBuilder backend** — candle has no built-in LoRA. BF16 path: custom `SimpleBackend` (`LoraBackend`) wraps mmap'd safetensors, intercepts `vb.get()` during construction, applies `B @ A` deltas inline. GGUF path: `gguf_lora_var_builder()` selectively dequantizes LoRA-affected tensors to F32 on CPU, merges deltas, re-quantizes back to original GGML dtype; non-LoRA tensors stay quantized. Compatible with offloading by targeting CPU device. `LoraDeltaCache` caches deltas across rebuilds; `LoraFingerprint` tracks the active LoRA to skip redundant drops.
