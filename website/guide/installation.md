# Installation

## One-Line Install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

Downloads the **latest tagged release** from
[github.com/utensils/mold/releases/latest](https://github.com/utensils/mold/releases/latest)
and installs it to `~/.local/bin/mold`. On Linux, the installer auto-detects
your NVIDIA GPU architecture (RTX 40-series or RTX 50-series). macOS builds
include Metal support.

### Options

All options are passed as environment variables:

```bash
# Install to a custom path
curl -fsSL ... | MOLD_INSTALL_DIR=/usr/local/bin sh

# Pin to a specific release tag (default: latest)
curl -fsSL ... | MOLD_VERSION=v0.9.0 sh

# Force a GPU architecture (default: auto-detect on Linux)
curl -fsSL ... | MOLD_CUDA_ARCH=sm120 sh   # Blackwell (RTX 50-series)
curl -fsSL ... | MOLD_CUDA_ARCH=sm89  sh   # Ada (RTX 40-series)
```

> **Note:** the env var has to be on the `sh` side of the pipe — with
> `VAR=value curl ... | sh`, the variable only applies to `curl` and the
> installer itself still sees the default.

`MOLD_VERSION` accepts any tag that exists on the
[releases page](https://github.com/utensils/mold/releases) — for example
`v0.8.0` to reproduce an older install. Without it the script follows the
`releases/latest` redirect on GitHub and installs whatever that currently
points at.

## Updating

```bash
mold update                       # Update to latest release
mold update --check               # Check for updates without installing
mold update --version v0.7.0      # Install a specific version
```

Or re-run the install script:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
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

```bash [Linux (CUDA), fast local build]
./scripts/ensure-web-dist.sh && cargo build --profile dev-fast -p mold-ai --features cuda
```

```bash [macOS (Metal), fast local build]
./scripts/ensure-web-dist.sh && cargo build --profile dev-fast -p mold-ai --features metal
```

```bash [Linux (CUDA), shipping build]
cargo build --release -p mold-ai --features cuda
```

```bash [macOS (Metal), shipping build]
cargo build --release -p mold-ai --features metal
```

:::

Requires Rust 1.85+ and CUDA toolkit (Linux) or Xcode (macOS).

Optional features can be added to the same build, for example
`--features cuda,preview,expand,discord,tui` or
`--features metal,preview,expand,discord,tui` if you also want terminal preview,
local prompt expansion, the Discord bot, or the interactive TUI.

`dev-fast` is the repo's local-iteration profile: it keeps debuginfo, enables
incremental compilation, and uses thin LTO plus more codegen units so optimized
builds stay much faster than the shipping `--release` profile.

## Docker

```bash
docker pull ghcr.io/utensils/mold:latest
docker run --gpus all -p 7680:7680 ghcr.io/utensils/mold:latest
```

See [Docker & RunPod](/deployment/docker) for full deployment instructions.

## Pre-Built Binaries

The one-line installer always targets the latest tag from the
[releases page](https://github.com/utensils/mold/releases). Each release ships
the following assets:

| Platform                                       | File                                              |
| ---------------------------------------------- | ------------------------------------------------- |
| macOS Apple Silicon                            | `mold-aarch64-apple-darwin.tar.gz`                |
| Linux x86_64 (Ada, RTX 4090 / 40-series)       | `mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz`  |
| Linux x86_64 (Blackwell, RTX 5090 / 50-series) | `mold-x86_64-unknown-linux-gnu-cuda-sm120.tar.gz` |

To install an older tag, put `MOLD_VERSION=<tag>` on the `sh` side of the
pipe, e.g. `curl -fsSL ... | MOLD_VERSION=v0.8.0 sh`. Placing it on the
`curl` side (`VAR=value curl ... | sh`) exports the variable to `curl` only;
the installer still sees the default and installs the latest release.

## Shell Completions

```bash
source <(mold completions bash)    # bash
source <(mold completions zsh)     # zsh
mold completions fish | source     # fish
```
