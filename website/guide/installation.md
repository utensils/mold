# Installation

## One-Line Install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

Downloads the latest pre-built binary to `~/.local/bin/mold`. On Linux, the
installer auto-detects your NVIDIA GPU architecture (RTX 40-series or RTX
50-series). macOS builds include Metal support.

Override the GPU architecture:

```bash
MOLD_CUDA_ARCH=sm120 curl -fsSL ... | sh  # Blackwell (RTX 50-series)
MOLD_CUDA_ARCH=sm89 curl -fsSL ... | sh   # Ada (RTX 40-series)
```

## Nix

```bash
# Run directly — no install needed
nix run github:utensils/mold -- run "a cat"

# Blackwell / RTX 50-series
nix run github:utensils/mold#mold-sm120 -- run "a cat"

# Add to your system profile
nix profile install github:utensils/mold
```

## From Source

::: code-group

```bash [Linux (CUDA)]
cargo build --release -p mold-cli --features cuda
```

```bash [macOS (Metal)]
cargo build --release -p mold-cli --features metal
```

:::

Requires Rust 1.85+ and CUDA toolkit (Linux) or Xcode (macOS).

## Docker

```bash
docker pull ghcr.io/utensils/mold:latest
docker run --gpus all -p 7680:7680 ghcr.io/utensils/mold:latest
```

See [Docker & RunPod](/deployment/docker) for full deployment instructions.

## Pre-Built Binaries

Available on the [releases page](https://github.com/utensils/mold/releases):

| Platform                 | File                                              |
| ------------------------ | ------------------------------------------------- |
| macOS Apple Silicon      | `mold-aarch64-apple-darwin.tar.gz`                |
| Linux x86_64 (Ada)       | `mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz`  |
| Linux x86_64 (Blackwell) | `mold-x86_64-unknown-linux-gnu-cuda-sm120.tar.gz` |

## Shell Completions

```bash
source <(mold completions bash)    # bash
source <(mold completions zsh)     # zsh
mold completions fish | source     # fish
```
