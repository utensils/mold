# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
