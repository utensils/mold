# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/utensils/mold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
