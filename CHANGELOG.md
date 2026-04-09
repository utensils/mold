# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

- **LTX-2 / LTX-2.3 joint audio-video generation**: added the new `ltx2` model family with `ltx-2-19b-{dev,distilled}:fp8` and `ltx-2.3-22b-{dev,distilled}:fp8` manifests, synchronized MP4-first video metadata, request fields for audio/video/keyframes/retake/upscaling, and a separate `Ltx2Engine` wired into the inference factory through the in-tree Rust runtime.
- **LTX-2 CLI surface**: added `--audio`, `--no-audio`, `--audio-file`, `--video`, repeatable `--keyframe`, repeatable `--lora`, `--pipeline`, `--retake`, `--camera-control`, `--spatial-upscale`, and `--temporal-upscale` to `mold run`.
- **LTX-2 devshell helpers**: the flake devshell now includes `gh`, `jq`, `imagemagick`, `python3`, and `uv`, plus `build-ltx2`, `test-ltx2`, `smoke-ltx2`, `contact-sheet`, and `issue-note`.
- **Qwen-Image-Edit-2511 request surface**: added `qwen-image-edit-2511:{bf16,q8,q6,q5,q4,q3,q2}` manifests, `GenerateRequest.edit_images`, family-aware validation, and distinct shared storage paths for edit-family VAE/text-encoder assets.
- **Qwen-Image-Edit-2511 inference pipeline**: added the Qwen2.5-VL multimodal edit encoder, condition-image preprocessing, packed edit-latent concatenation with `img_shapes`, `zero_cond_t` transformer support, and true-CFG norm rescaling for local image editing.

### Changed

- **Video defaults for `ltx2`**: LTX-2 requests now default to MP4 output instead of PNG/APNG and strip audio automatically when exporting GIF/APNG/WebP.
- **Manifest plumbing**: `ModelPaths`, manifests, validation, and `mold info` now understand temporal upscalers and distilled LoRAs.
- **LTX-2 native upscaling path**: temporal `x2` upscaling now reaches the native Rust runtime, and the stage-1 render plan derives lower-resolution/lower-fps shapes before native spatial and temporal upsampling restore the requested output dimensions.
- **`--image` CLI semantics**: `mold run --image` is now repeatable. Non-edit families still accept at most one source image; `qwen-image-edit` maps repeated `--image` flags into `edit_images`.
- **TUI capability modeling**: `qwen-image-edit` now appears as a source-image editing family instead of img2img, so the TUI exposes a source image and negative prompt without img2img-only controls like `strength`, `mask`, `ControlNet`, or `LoRA`.

### Fixed

- **LTX-2 manifest accounting**: model-size and download-path resolution now treat the single-file LTX-2 checkpoints correctly without requiring a standalone VAE asset.
- **LTX-2 camera-control presets**: `--camera-control dolly-in|dolly-left|dolly-out|dolly-right|jib-down|jib-up|static` now resolves the published LTX-2 19B camera LoRAs instead of failing validation.
- **LTX-2 local Ada runtime**: local 24 GB FP8 runs now stay on the native Rust path and the compatible `fp8-cast` mode instead of Hopper-only `fp8-scaled-mm`, avoiding the TensorRT-LLM dependency.
- **LTX-2 native FP8 prompt/runtime path**: the native Gemma and embeddings stack now streams decoder layers, normalizes BF16/F32 CPU inspection paths, and materializes contiguous connector attention tensors so local CUDA smoke clips can complete without bridge fallbacks or prompt-path dtype/layout failures.
- **LTX-2 CLI LoRA aliases**: `mold run --lora camera-control:<preset>` now reaches the native LTX-2 resolver instead of failing early in generic file-path validation, so `ic-lora` smoke requests can use published camera-control aliases from the CLI.
- **LTX-2.3 x1.5 spatial upscale requests**: `--spatial-upscale x1.5` now passes validation and resolves the published `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` asset on demand instead of failing before the request reaches the engine.
- **LTX-2 temporal upscale requests**: `--temporal-upscale x2` now passes validation, resolves the configured temporal upsampler asset, and executes the native temporal interpolation path instead of failing as unimplemented.
- **TUI remote server awareness**: the Info panel, model defaults, and model management now reflect the connected server instead of the local machine ([#158](https://github.com/utensils/mold/issues/158)):
  - Info panel queries `/api/status` for memory, GPU, and busy state when connected to a remote server
  - Model parameter defaults (steps, guidance, width, height) come from the server's catalog instead of local `config.toml`
  - Model pull routes through the server API when connected remotely
  - Reset Defaults action uses server catalog values when connected
  - Tab bar shows hostname (e.g. "hal9000") when connected to a remote server, "local" in local mode, "connecting..." during health check
  - `/api/status` now includes `hostname` and `memory_status` fields (backward-compatible, older servers omit them)
- **Qwen edit-family encoder selection**: `qwen-image-edit` now supports quantized `--qwen2-variant` values by splitting the multimodal stack into a GGUF Qwen2.5 language path plus a safetensor-backed Qwen2.5-VL vision tower for image conditioning.
- **Qwen edit-family RAM use**: `qwen-image-edit` now stages the Qwen2.5 encoder instead of keeping it resident after model load, drops encoder weights immediately after edit conditioning, and avoids loading the full BF16 language stack just to run multimodal edits. A local `qwen-image-edit-2511:q4 --qwen2-variant q4` smoke run completed with roughly 1.8 GB max RSS instead of the earlier tens-of-GB host spike.
- **Qwen2.5 CPU dtype handling**: CPU Qwen2.5 encoder paths now stay in `F32` where candle CPU matmul requires it, while lower-memory edit inference uses quantized GGUF language weights plus the staged vision sidecar instead of relying on unsupported CPU `BF16` matmuls.
- **Qwen2.5 vision progress reporting**: the staged vision-tower load now reports only the bytes for `visual.*` tensors instead of the full shared text-encoder shard set, removing the misleading `15.45 GiB` progress line during edit encoder reloads.
## [0.6.2] - 2026-04-08

### Added

- **`mold update` command**: self-update from GitHub releases with SHA-256 checksum verification, platform auto-detection (macOS Metal, Linux CUDA sm89/sm120), atomic binary replacement with rollback, and package manager detection (Nix, Homebrew). Supports `--check`, `--force`, and `--version` flags.
- **img2img and inpainting for all remaining model families**: SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image, and Wuerstchen v2 now support `--image` for img2img and `--mask` for inpainting. Includes VAE encoder implementations for Flux.2 (BN-VAE with patchified BatchNorm) and Qwen-Image (3D causal VAE specialized to 2D via temporal slice extraction). Wuerstchen uses VQ-GAN encoding with Prior bypass for img2img. All families support strength-based schedule trimming and per-step inpainting blending. ([#174](https://github.com/utensils/mold/issues/174))

### Fixed

- **Img2img denoising strength alignment**: all model families now map `--strength` to scheduler start steps using reference img2img semantics instead of treating the user value as a raw flow sigma. This restores expected source-image retention for low-strength edits and removes washed-out flow-model img2img results.
- **Img2img VAE source normalization**: Flux.2, Qwen-Image, and SD3 img2img now encode source images in the autoencoder's expected `[-1, 1]` pixel range, eliminating the white, foggy washout that appeared even at `--strength 0.0`.
- **Deterministic img2img VAE encodes**: Flux.2, Qwen-Image, SD1.5, SDXL, and SD3 now use posterior-mean latents when encoding source images, removing unseeded VAE sampling noise so repeated img2img runs with the same source image and seed stay reproducible.
- **`--strength 0` validation**: img2img requests now accept `strength = 0.0` as documented, preserving the source image with zero denoise instead of rejecting the request.
- **Z-Image GGUF inference**: `z-image-turbo:q4/q6/q8` now load GGUF weights into the standard dense Z-Image transformer instead of the broken custom quantized transformer path, restoring coherent txt2img output and fixing sequential img2img by encoding source images before the transformer is loaded.

## [0.6.1] - 2026-04-07

### Added

- **TUI UAT harness**: native Ghostty 1.3+ AppleScript-based acceptance testing harness (`scripts/tui-uat.sh`) with launch, capture, send keystrokes, view navigation, screenshot, and assert commands — pixel-perfect terminal screenshots via `screencapture -l<windowID>` ([#173](https://github.com/utensils/mold/issues/173))

### Changed

- **Qwen-Image CUDA execution planning**: local sequential runs now preserve the stable load-use-drop path, while hot server mode keeps more of the quantized CUDA stack resident, skips redundant prompt re-encoding on cache hits, and splits CFG passes when batching would overrun VRAM

### Fixed

- **CLI Ghostty preview regression**: `mold run --preview` and `mold upscale --preview` now bypass `viuer`'s leaking Kitty capability probe in Ghostty without falling back to blocky half-block rendering, preserving crisp inline previews while preventing visible `^[_Gi=31;OK...` control text
- **Qwen-Image-2512 CUDA black images / OOMs**: quantized CUDA generation now uses a staged fallback ladder for VAE decode (full GPU, tiled GPU, then CPU reload), keeps the quantized transformer resident across decode when that is safe, and aligns BF16/GGUF text masking behavior with upstream Qwen
- **Server startup thumbnail warmup**: gallery thumbnail generation is now opt-in via `MOLD_THUMBNAIL_WARMUP=1`, preventing avoidable CPU/RAM spikes on startup when large or bad output directories are present

## [0.6.0] - 2026-04-07

### Added

- **LTX Video — text-to-video generation**: first video model family in mold. Generate animated video clips from text prompts using LTX 0.9.6 and 0.9.8 checkpoints (2B and 13B variants) with APNG, GIF, WebP, and MP4 output formats ([#172](https://github.com/utensils/mold/issues/172))
- **Video CLI flags**: `--frames <N>` (must be 8n+1), `--fps <N>` (default 24), `--format apng|gif|webp|mp4` for video models
- **Image upscaling**: `mold upscale <image>` with 7 Real-ESRGAN variants (2x/4x, RRDBNet + SRVGGNetCompact), tiled processing for large images, server API (`POST /api/upscale`), and TUI gallery integration with background upscaling ([#170](https://github.com/utensils/mold/issues/170))
- **Qwen-Image enhancements**: FP8 support, block-level offloading for 24GB cards, negative prompt support, full Q2-Q8 GGUF tier coverage for both base and 2512 variants, and encoder variant selection via `--qwen2-variant` / `--qwen2-text-encoder-mode` CLI flags ([#178](https://github.com/utensils/mold/issues/178))
- **Prometheus metrics**: `GET /metrics` endpoint with HTTP rates, generation duration, queue depth, GPU memory, and uptime (behind `metrics` feature flag) ([#142](https://github.com/utensils/mold/issues/142))
- **Custom MP4 muxer**: minimal QuickTime-compatible H.264 writer with faststart layout, replacing muxide dependency ([#181](https://github.com/utensils/mold/issues/181))
- **CI feature matrix**: new `check-features` job validates all optional feature combinations
- **GIF preview cache**: animated GIF previews cached in `~/.mold/cache/previews/` for TUI gallery

### Changed

- **LTX 0.9.8 multiscale refinement**: 0.9.8 checkpoints now run full two-pass generation with spatial upsampler and continuation window ([#201](https://github.com/utensils/mold/issues/201))
- **LTX model catalog**: replaced legacy 0.9/0.9.5 manifests with 0.9.6, 0.9.6-distilled, 0.9.8-2b-distilled, 0.9.8-13b-dev, and 0.9.8-13b-distilled
- **LTX shared asset layout**: canonical paths now match upstream sources; legacy installs auto-migrate on pull ([#204](https://github.com/utensils/mold/issues/204))
- **Qwen-Image model sources**: base GGUF models now map to `Qwen/Qwen-Image` + `city96/Qwen-Image-gguf`; 2512 variants use `unsloth/Qwen-Image-2512-GGUF` ([#178](https://github.com/utensils/mold/issues/178))
- **Batch limit removed**: `--batch` no longer capped at 16
- **Nine model families**: `LtxVideoEngine` added to engine factory; `UpscaleEngine` trait added for upscaler models
- **CI consolidated**: merged check/clippy/test into single job with shared rust-cache

### Fixed

- **Qwen-Image CUDA black images**: Metal GGUF denoising optimization (#207) replaced per-forward BF16 dequantization with QMatMul/F32 globally, breaking CUDA inference. Fixed with device-gated dispatch — BF16 dequant on CUDA, QMatMul on Metal, F32 on CPU ([#207](https://github.com/utensils/mold/pull/207))
- **Qwen-Image Metal performance**: quantized GGUF transformer uses Candle's quantized linear path and caches RoPE tensors, improving Q4 denoising from ~0.03 it/s to ~0.15 it/s on M4 Max ([#202](https://github.com/utensils/mold/issues/202))
- **Qwen-Image GGUF prompt adherence**: restored combined padding + causal attention masking in Qwen2 text encoder ([#178](https://github.com/utensils/mold/issues/178))
- **Qwen-Image Metal text encoding**: `auto` now prefers quantized GGUF encoders on Metal/MPS
- **Stale pull markers**: manifest-backed models self-heal stale `.pulling` markers; partial installs auto-repair on use
- **Upscaler/utility model listing**: `mold list` correctly shows non-diffusion models as installed ([#184](https://github.com/utensils/mold/issues/184))
- **LTX inference quality**: versioned transformer/VAE presets with `decode_noise_scale` for timestep-conditioned VAE decode
- **MP4 QuickTime compatibility**: correct ftyp brands, colr/pasp atoms, BT.601 range alignment ([#181](https://github.com/utensils/mold/issues/181))
- **Server video handling**: queue worker and CLI batch loop no longer discard video responses
- **Test isolation**: config/cleanup tests use temp `MOLD_HOME` to avoid touching live model caches

### Removed

- **Legacy LTX manifests**: removed `ltx-video-0.9:bf16` and `ltx-video-0.9.5:bf16`

## [0.5.3] - 2026-04-04

### Added

- **Qwen3-8B encoder support**: Klein-9B models now use the correct Qwen3-8B text encoder (hidden_size=4096) with auto-selection between 4B and 8B GGUF variant registries ([#157](https://github.com/utensils/mold/issues/157), [#166](https://github.com/utensils/mold/pull/166))
- **Qwen3-8B GGUF variant registry**: Q8/Q6/IQ4/Q3 quantized encoders from `unsloth/Qwen3-8B-GGUF` for Klein-9B, with separate cache directory (`shared/qwen3-8b-gguf`) ([#166](https://github.com/utensils/mold/pull/166))
- **Gallery images**: Klein-9B bottle ship and Wuerstchen v2 lighthouse added to homepage gallery and model pages ([#166](https://github.com/utensils/mold/pull/166))

### Changed

- **Quantized inference for Flux.2 GGUF**: rewrote the Flux.2 quantized transformer to keep weights in compressed GGUF format in VRAM and dequantize on-the-fly per matmul via `QMatMul`, matching ComfyUI and InvokeAI; Klein-9B Q4 VRAM usage drops from ~18GB to ~6GB, load time from minutes to seconds ([#166](https://github.com/utensils/mold/pull/166))
- **Default model**: changed from bare `flux2-klein` to explicit `flux2-klein:q8` for new users ([#166](https://github.com/utensils/mold/pull/166))
- **Klein guidance default**: both Klein-4B and Klein-9B now default to guidance=1.0, matching BFL model cards and InvokeAI ([#166](https://github.com/utensils/mold/pull/166))

### Fixed

- **Klein-9B OOM on CUDA**: "Metal out of memory" error on Linux/CUDA when loading Klein-9B models; root cause was wrong Qwen3 encoder (4B instead of 8B) and full dequantization exhausting VRAM ([#157](https://github.com/utensils/mold/issues/157), [#166](https://github.com/utensils/mold/pull/166))
- **CUDA OOM error message**: now shows "CUDA out of memory" on Linux instead of raw candle backtrace ([#166](https://github.com/utensils/mold/pull/166))
- **GGUF NaN safety**: quantized Flux.2 transformer wraps all linear operations with NaN-safe filter for CUDA, following SD3's established pattern ([#166](https://github.com/utensils/mold/pull/166))
- **Pre-existing TUI test failures**: settings tests expected bare `flux2-klein` but config resolves to `flux2-klein:q8` ([#166](https://github.com/utensils/mold/pull/166))

### Removed

- **Klein-9B alpha status**: all Klein-9B model descriptions, docs, and skills updated to remove alpha labels — Klein-9B is now fully supported ([#166](https://github.com/utensils/mold/pull/166))

## [0.5.2] - 2026-04-03

### Fixed

- **TUI metadata race condition**: switching models while a generation was running caused the saved image to record the newly selected model instead of the model that actually generated it; now uses `response.model` from the server as source of truth ([#161](https://github.com/utensils/mold/issues/161), [#163](https://github.com/utensils/mold/pull/163))
- **TUI batch mode**: setting batch > 1 in the TUI had no effect — only one image was generated; now loops client-side with seed increment, matching CLI behavior ([#162](https://github.com/utensils/mold/issues/162), [#163](https://github.com/utensils/mold/pull/163))
- **SSE protocol**: `SseCompleteEvent` now includes the `model` field so the server confirms which model generated the image; backward-compatible with older servers via `#[serde(default)]`

## [0.5.1] - 2026-04-03

### Fixed

- **TUI info panel crash**: model descriptions containing multi-byte UTF-8 characters (e.g., em dash `—`) caused a panic when truncated to fit the panel width; now uses `str::floor_char_boundary()` for safe truncation ([#159](https://github.com/utensils/mold/issues/159), [#160](https://github.com/utensils/mold/pull/160))

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

[Unreleased]: https://github.com/utensils/mold/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/utensils/mold/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/utensils/mold/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/utensils/mold/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/utensils/mold/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/utensils/mold/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/utensils/mold/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/utensils/mold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/utensils/mold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/utensils/mold/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/utensils/mold/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/utensils/mold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/utensils/mold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
