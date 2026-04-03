# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-03

### Added

- **`mold server start|status|stop`**: daemon management for standalone background server — PID file tracking, health polling, graceful shutdown cascade (HTTP → SIGTERM → SIGKILL), unmanaged process detection
- **Graceful server shutdown**: `POST /api/shutdown` endpoint triggers clean drain via `axum::serve::with_graceful_shutdown`; SIGTERM handler uses the same channel
- **API key authentication**: opt-in `MOLD_API_KEY` with single key, comma-separated, or `@/path/to/keys.txt` file reference; exempt `/health` and `/api/docs` ([#150](https://github.com/utensils/mold/pull/150))
- **Per-IP rate limiting**: `MOLD_RATE_LIMIT` with separate generation and read tiers (10x), `Retry-After` headers, configurable burst via `MOLD_RATE_LIMIT_BURST` ([#150](https://github.com/utensils/mold/pull/150))
- **Request ID correlation**: `X-Request-ID` header on all responses, preserved from client or auto-generated UUID v4 ([#150](https://github.com/utensils/mold/pull/150))
- **`mold stats` command**: disk usage overview for models, output, logs, and shared components with `--json` output ([#151](https://github.com/utensils/mold/pull/151))
- **`mold clean` command**: clean orphaned files, stale `.pulling` markers, hf-cache transient files, and old output images; dry-run by default with `--force` and `--older-than` flags ([#151](https://github.com/utensils/mold/pull/151))
- **Discord role-based access**: `MOLD_DISCORD_ALLOWED_ROLES` for comma-separated role names/IDs, with friendly denial messages ([#149](https://github.com/utensils/mold/pull/149))
- **Discord per-user daily quotas**: `MOLD_DISCORD_DAILY_QUOTA` with midnight UTC reset, `/quota` command, and `/admin reset-quota` / `/admin block` / `/admin unblock` commands ([#149](https://github.com/utensils/mold/pull/149))
- **TUI model deletion**: delete models from the TUI with confirmation dialog, disk space preview, and shared-file warnings ([#148](https://github.com/utensils/mold/pull/148))
- **TUI version display**: version string shown in upper-right corner of tab bar
- **Version logging**: server and TUI log version on startup for diagnostics
- **NixOS module**: `discord.environment` option for arbitrary bot env vars, `logToFile`/`logDir`/`logRetentionDays` server options ([#152](https://github.com/utensils/mold/pull/152))

### Fixed

- **Server production panics**: replaced `.unwrap()` and `.expect()` calls with proper error propagation in model_manager, logging, image metadata, cache mutex locks, and CLI commands ([#146](https://github.com/utensils/mold/pull/146))
- **Download progress blocking**: SHA-256 pre-scan moved to `spawn_blocking` so SSE progress events stream in real time instead of being blocked for 10-30s
- **Download status accuracy**: TUI now shows "Verifying...", "Downloading...", or "Preparing..." based on current phase instead of always showing "Generating..."
- **TUI session persistence**: settings now saved on quit (not just after generation), bare model names resolved correctly on restore, and fallback path no longer applies wrong model-specific params ([#153](https://github.com/utensils/mold/pull/153))
- **TUI download progress**: progress bar now shows file counter `[2/5]` with per-file and batch byte totals during model pulls ([#153](https://github.com/utensils/mold/pull/153))
- **Gallery thumbnail cache**: fixed missing cache entry when prepending new gallery images
- **VitePress container rendering**: prettier no longer collapses `:::` callout markers; set `proseWrap: "preserve"` for markdown files ([#152](https://github.com/utensils/mold/pull/152))

### Changed

- **README trimmed**: reduced from 718 to ~190 lines; detailed content moved to the documentation website ([#152](https://github.com/utensils/mold/pull/152))
- Documentation synced across website, CLAUDE.md, SKILL.md — added `mold stats`/`mold clean` to CLI reference, gallery API endpoints, Discord bot features, NixOS module options, Docker env vars ([#152](https://github.com/utensils/mold/pull/152))
- Comprehensive config.rs test coverage (843 lines of tests for config loading, migrations, path resolution) ([#147](https://github.com/utensils/mold/pull/147))

### Performance

- **Gallery thumbnail caching**: fixed-protocol thumbnails cached across render frames, eliminating per-frame `image::open()` + protocol creation

## [0.4.1] - 2026-04-02

### Fixed

- **crates.io publish**: add missing `mold-ai-discord` to release workflow publish chain — was never published, causing `mold-ai` publish to fail on 0.4.0

## [0.4.0] - 2026-04-02

### Added

- **Interactive Terminal UI**: full-featured TUI with Generate, Gallery, Models, and Settings views — built on ratatui with Kitty/Sixel/iTerm2/halfblock image preview ([#133](https://github.com/utensils/mold/pull/133), [#134](https://github.com/utensils/mold/pull/134))
- **TUI Gallery**: thumbnail grid with cached 256x256 previews, detail view with metadata, edit/regenerate/delete actions ([#134](https://github.com/utensils/mold/pull/134))
- **Settings editor**: `mold config` subcommands (`list`, `get`, `set`, `path`, `edit`) for managing config.toml from the CLI, plus a TUI Settings view for interactive editing ([#136](https://github.com/utensils/mold/pull/136))
- **Prompt history**: persisted across sessions with fuzzy search and Up/Down recall ([#133](https://github.com/utensils/mold/pull/133))
- **Session persistence**: TUI saves and restores all generation parameters across launches ([#133](https://github.com/utensils/mold/pull/133))
- **Auto-start server**: TUI automatically starts a background `mold serve` process, killed on exit ([#133](https://github.com/utensils/mold/pull/133))
- **File logging**: `--log-file` flag and `[logging]` config section for rotated log files in `~/.mold/logs/` ([#133](https://github.com/utensils/mold/pull/133))
- **Config migrations**: automatic migration from legacy config locations and formats ([#133](https://github.com/utensils/mold/pull/133))
- **LoRA fingerprint caching**: skip redundant transformer rebuilds when the same LoRA is reused across requests ([#131](https://github.com/utensils/mold/pull/131))
- **LoRA delta caching**: `LoraDeltaCache` caches pre-computed `B @ A * scale` delta tensors on CPU across transformer rebuilds ([#131](https://github.com/utensils/mold/pull/131))
- **Shared tokenizer pool**: cross-engine `SharedPool` caches T5 and CLIP tokenizers via `Arc<Tokenizer>`, saving ~100-150ms on model switches ([#131](https://github.com/utensils/mold/pull/131))

### Fixed

- **MPS memory guards**: prevent OOM on low-RAM Apple Silicon by checking available memory before allocation ([#134](https://github.com/utensils/mold/pull/134))
- **Test race condition**: eliminate flaky `queued_stream_receives_position_event` test ([#132](https://github.com/utensils/mold/pull/132))

### Changed

- Multi-model cache enhanced with `ModelResidency` states (Gpu, Parked, Unloaded) for smarter VRAM management ([#131](https://github.com/utensils/mold/pull/131))
- Documentation synced across website, README, CLAUDE.md, and SKILL.md — TUI, `mold config`, `[logging]` section, and Discord bot command references

## [0.3.1] - 2026-04-01

### Fixed

- **OOM prevention in eager/server mode**: drop transformer/UNet before VAE decode in all 8 pipelines — prevents out-of-memory on GPUs where transformer + VAE decode intermediates exceed VRAM ([#128](https://github.com/utensils/mold/pull/128))
- **Batch cache optimization**: skip text encoder load on prompt cache hit in all 8 sequential/CLI pipelines, saving 2-10s per batch image depending on model family ([#128](https://github.com/utensils/mold/pull/128))
- **Wuerstchen cache key**: include negative prompt and CFG flags in cache key to prevent silent cache collisions ([#128](https://github.com/utensils/mold/pull/128))
- **Unified memory fixes for low-RAM Apple Silicon**: drop text encoders after encoding on Metal (unified memory), use available memory (free + inactive) for variant selection on macOS, and show "Metal out of memory" instead of misleading CUDA error ([#126](https://github.com/utensils/mold/pull/126))
- **Skip encoder reload on cache hit**: Flux2 and Z-Image eager paths no longer reload encoder from disk before checking prompt cache ([#126](https://github.com/utensils/mold/pull/126))
- **Stale alpha labels**: removed incorrect alpha labels from Klein-4B model descriptions; Klein-9B correctly retains alpha ([#128](https://github.com/utensils/mold/pull/128))

### Changed

- Candle dependencies switched from git refs to crates.io 0.9.3 — unblocks `cargo publish` for all mold crates

## [0.3.0] - 2026-04-01

### Added

- **Flux.2 Klein-9B model (alpha)**: 9B parameter distilled model with larger Qwen3 encoder (hidden_size=4096), 4-step generation ([#123](https://github.com/utensils/mold/pull/123))
- **SD 3.5 model family**: Triple encoder (CLIP-L + CLIP-G + T5-XXL), quantized MMDiT with NaN-safe inference, SLG support ([#111](https://github.com/utensils/mold/pull/111))
- **Qwen-Image model family**: Qwen2.5-VL text encoder, 3D causal VAE, flow-matching with classifier-free guidance ([#111](https://github.com/utensils/mold/pull/111))
- **LLM-powered prompt expansion**: local Qwen3-1.7B GGUF expansion with `--expand` flag, user-configurable templates, per-family tuning ([#86](https://github.com/utensils/mold/pull/86), [#88](https://github.com/utensils/mold/pull/88))
- **CUDA block-level offloading**: streams transformer blocks CPU↔GPU to reduce VRAM from ~24GB to ~2-4GB via `--offload` flag ([#84](https://github.com/utensils/mold/pull/84))
- **LoRA adapter support**: custom VarBuilder backend for FLUX BF16 and GGUF quantized models with `--lora` and `--lora-scale` flags ([#94](https://github.com/utensils/mold/pull/94), [#97](https://github.com/utensils/mold/pull/97))
- **Negative prompt support**: `--negative-prompt` and `--no-negative` flags for CFG-based model families ([#90](https://github.com/utensils/mold/pull/90))
- **Dimension validation**: warns when dimensions don't match model recommendations ([#116](https://github.com/utensils/mold/pull/116))
- **Discord bot integration**: `/generate`, `/expand`, `/models`, `/status` slash commands, standalone or embedded in server ([#72](https://github.com/utensils/mold/pull/72), [#75](https://github.com/utensils/mold/pull/75))
- **Docker/RunPod deployment**: multi-stage Dockerfile, RunPod entrypoint, network volume support ([#110](https://github.com/utensils/mold/pull/110))
- **VitePress documentation site**: full docs at utensils.github.io/mold with model guides, API reference, deployment docs ([#114](https://github.com/utensils/mold/pull/114))
- **Multi-CUDA architecture support**: Ada (sm_89) + Blackwell (sm_120) build targets ([#91](https://github.com/utensils/mold/pull/91))
- **Terminal image preview**: `--preview` flag to display images inline ([#55](https://github.com/utensils/mold/pull/55))
- **`mold default` command**: get/set default model ([#64](https://github.com/utensils/mold/pull/64))
- **`mold info` command**: installation overview and per-model details with optional SHA-256 verify ([#53](https://github.com/utensils/mold/pull/53))
- **Multi-model server cache**: server LoRA support and server-side prompt expansion ([#107](https://github.com/utensils/mold/pull/107))
- **SHA-256 download verification**: fails on integrity mismatch, `--skip-verify` to override ([#56](https://github.com/utensils/mold/pull/56))
- **One-line install script**: `curl | sh` installer with GPU auto-detection

### Changed

- **Default model changed to `flux2-klein:q8`**: fully ungated (no HuggingFace auth), Apache 2.0 licensed, fast 4-step generation ([#125](https://github.com/utensils/mold/pull/125))
- Lazy mmap weight loading replaces eager per-tensor loading for better performance ([#122](https://github.com/utensils/mold/pull/122))
- `mold run` auto-resizes source images to model-native resolution for img2img ([#51](https://github.com/utensils/mold/pull/51))
- Server queues concurrent requests instead of dropping them ([#68](https://github.com/utensils/mold/pull/68))
- Git SHA and build date embedded in version output ([#67](https://github.com/utensils/mold/pull/67))
- CLI validates file-based arguments before expansion/inference ([#101](https://github.com/utensils/mold/pull/101))

### Fixed

- Model weight loading and Qwen-Image quantized inference performance ([#124](https://github.com/utensils/mold/pull/124))
- Batch generation saves and previews images immediately ([#120](https://github.com/utensils/mold/pull/120))
- Expand segfault, crash reporting, and ps process detection ([#118](https://github.com/utensils/mold/pull/118))
- LoRA VRAM leak in GGUF re-quantization ([#100](https://github.com/utensils/mold/pull/100))
- Weight loader CPU fallback for dtype conversion ([#98](https://github.com/utensils/mold/pull/98))
- Server unloads current model before loading a different one ([#106](https://github.com/utensils/mold/pull/106))
- Server fully releases GPU memory on unload ([#62](https://github.com/utensils/mold/pull/62))
- FLUX auto-patch for city96-format GGUFs missing embedding layers ([#61](https://github.com/utensils/mold/pull/61))
- `mold rm` comprehensive cleanup for shared files, hf-cache, and stale markers ([#93](https://github.com/utensils/mold/pull/93))
- Qwen-Image denoising quality and performance ([#111](https://github.com/utensils/mold/pull/111))
- SD1.5 repo updated to canonical location ([#113](https://github.com/utensils/mold/pull/113))
- Wuerstchen v2 sampling alignment and default negative prompt ([#99](https://github.com/utensils/mold/pull/99))

### Removed

- **Flux.2 Klein-base-4B model**: all variants removed — denoising fails to converge, producing incoherent output ([#125](https://github.com/utensils/mold/pull/125))

## [0.2.0] - 2026-03-24

### Added

- **Wuerstchen v2 model family**: 3-stage cascade pipeline (Prior → Decoder → VQ-GAN) with CLIP-G text encoder and 42x latent compression ([#25](https://github.com/utensils/mold/pull/25))
- **Image-to-image mode**: `--image` and `--strength` flags for SD1.5, SDXL, and FLUX; stdin support with `--image -` for piping ([#27](https://github.com/utensils/mold/pull/27))
- **Inpainting**: `--mask` flag for selective repaint (white=repaint, black=preserve) ([#27](https://github.com/utensils/mold/pull/27))
- **ControlNet conditioning for SD1.5**: `--control`, `--control-model`, `--control-scale` flags with canny, depth, and openpose models ([#28](https://github.com/utensils/mold/pull/28))
- **Configurable noise schedulers**: DDIM, Euler Ancestral, UniPC for SD1.5/SDXL via `--scheduler` flag ([#26](https://github.com/utensils/mold/pull/26))
- **Inference caching**: LRU caches for text encoder embeddings, source image latents, masks, and control tensors — avoids re-encoding on repeated prompts/batches ([#41](https://github.com/utensils/mold/pull/41))
- **Unified model catalog**: automatic model discovery from `MOLD_MODELS_DIR` without requiring config entries; single source of truth for CLI and server ([#41](https://github.com/utensils/mold/pull/41))
- **PNG generation metadata**: prompt, model, seed, steps, guidance embedded as PNG text chunks by default; disable with `--no-metadata` ([#43](https://github.com/utensils/mold/pull/43))
- **18 community models**: FLUX fine-tunes (Krea, JibMix, Ultrareal, Iniverse), SDXL fine-tunes (Pony, Cyberrealistic), and Flux.2 Klein GGUF quantizations ([#39](https://github.com/utensils/mold/pull/39))
- **NixOS module**: declarative `services.mold` with systemd hardening, GPU access, HF token support, firewall options, and shell completions
- **Centralized color/theme system**: semantic icon/prefix helpers with ANSI 16-color palette for broad terminal compatibility ([#42](https://github.com/utensils/mold/pull/42))

### Changed

- Custom `thiserror` error enum replaces generic `MoldError` for precise error handling ([#16](https://github.com/utensils/mold/pull/16))
- Server returns structured JSON error responses with error codes ([#18](https://github.com/utensils/mold/pull/18))
- `mold list` shows separate SIZE and FETCH columns for accurate disk usage ([#33](https://github.com/utensils/mold/pull/33), [#37](https://github.com/utensils/mold/pull/37))
- `mold pull` routes through server when available, falls back to local download ([#41](https://github.com/utensils/mold/pull/41))
- Batch generation reuses loaded engine across iterations for faster multi-image runs ([#41](https://github.com/utensils/mold/pull/41))
- Generation output shows prompt in header block and model name in completion summary ([#42](https://github.com/utensils/mold/pull/42))

### Fixed

- Batch image generation with proper seed increment across images ([#17](https://github.com/utensils/mold/pull/17))
- FLUX img2img auto-resize, schedule, and VAE normalization ([#34](https://github.com/utensils/mold/pull/34))
- Wuerstchen pipeline reliability, context-aware image sizes, and Ctrl+C handling ([#31](https://github.com/utensils/mold/pull/31))
- Server reliability improvements and code quality cleanup ([#38](https://github.com/utensils/mold/pull/38))
- UTF-8 safe prompt truncation prevents panics on multi-byte characters ([#42](https://github.com/utensils/mold/pull/42))
- Multiple Nix build/service fixes: CUDA toolkit discovery, modelsDir permissions, EnvironmentFile loading, cross-eval
- Broken model descriptions renamed to alpha ([#40](https://github.com/utensils/mold/pull/40))

### Improved

- CLI help messages with examples, environment variable documentation, and grouped options ([#12](https://github.com/utensils/mold/pull/12))
- Code quality, validation, and deduplication from peer review ([#15](https://github.com/utensils/mold/pull/15))

## [0.1.0] - 2026-03-19

Initial public release on [crates.io](https://crates.io/crates/mold-ai).

### Added

- **Seven model families**: FLUX.1 (dev/schnell), SD 1.5, SDXL, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image
- **CLI commands**: `run`, `serve`, `pull`, `rm`, `list`, `info`, `unload`, `ps`, `version`, `completions`
- **Local + remote inference**: run models on local GPU or connect to a remote `mold serve` instance
- **Automatic local fallback**: if server is unreachable, falls back to local GPU inference
- **GGUF quantized model support**: BF16 and GGUF (Q3–Q8) for reduced VRAM usage
- **Smart VRAM management**: dynamic device placement, drop-and-reload text encoders, quantized encoder auto-fallback
- **Pipe-friendly output**: `mold run "a cat" | viu -` just works (image bytes to stdout, status to stderr)
- **Stdin prompt support**: `echo "a cat" | mold run flux-schnell`
- **Model management**: `mold pull` downloads from HuggingFace, `mold list` shows disk usage, `mold rm` cleans up
- **GPU backends**: CUDA (Linux) and Metal (macOS) via candle
- **Deterministic seeds**: CPU-based noise generation ensures same seed produces identical images across backends
- **SSE progress streaming**: `POST /api/generate/stream` for real-time denoising progress
- **OpenAPI docs**: interactive API documentation at `/api/docs`
- **Shell completions**: static and dynamic completions for bash, zsh, fish, elvish, powershell
- **Nix flake**: reproducible builds with Metal (macOS) and CUDA (Linux) support
- **Pre-built binaries**: GitHub releases for macOS (Metal) and Linux (CUDA)

### crates.io packages

| Crate | Description |
|-------|-------------|
| [`mold-ai`](https://crates.io/crates/mold-ai) | CLI binary (`cargo install mold-ai`) |
| [`mold-ai-core`](https://crates.io/crates/mold-ai-core) | Shared types, API protocol, HTTP client |
| [`mold-ai-inference`](https://crates.io/crates/mold-ai-inference) | Candle-based inference engine |
| [`mold-ai-server`](https://crates.io/crates/mold-ai-server) | Axum HTTP inference server |

[0.5.0]: https://github.com/utensils/mold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/utensils/mold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/utensils/mold/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/utensils/mold/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/utensils/mold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/utensils/mold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
