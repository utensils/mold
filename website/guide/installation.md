# Installation

## Native apps

- **macOS desktop:** download the signed and notarized
  [Mold DMG](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg),
  then follow the [Desktop App guide](/guide/desktop).
- **iPhone:** the current remote-only app is distributed through the project's
  invited internal and external TestFlight groups; there is not yet a public
  App Store listing. See the [iPhone App guide](/guide/iphone) for supported
  workflows and host setup.

Mold is CLI-native. The command-line installation below installs the primary
Mold interface and the same engine/server used by both native apps, scripts,
agents, and custom API clients.

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
curl -fsSL ... | MOLD_VERSION=v0.10.0 sh

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

## Arch Linux / AUR

Three packages on the [AUR](https://aur.archlinux.org/):

```bash
paru -S mold-ai-bin     # Prebuilt binary, CUDA sm_89 (RTX 40-series). Fastest.
paru -S mold-ai         # Builds from source — for Blackwell, prefix CUDA_COMPUTE_CAP=120
paru -S mold-ai-git     # Builds from main HEAD
```

Substitute `yay`, `pikaur`, or any other AUR helper as appropriate. With
vanilla `makepkg`:

```bash
git clone https://aur.archlinux.org/mold-ai-bin.git
cd mold-ai-bin
makepkg -si
```

**Conflict with `extra/mold`**: All three packages declare `conflicts=('mold')`
because they install `/usr/bin/mold` — the same path used by the
[rui314 linker](https://archlinux.org/packages/extra/x86_64/mold/). You cannot
have both installed simultaneously. If you need the linker for your build
toolchain, install mold via Nix or the one-line installer (which targets
`~/.local/bin`) instead.

**Blackwell (RTX 50-series)**: The `mold-ai-bin` package ships the sm_89 (Ada)
tarball. For sm_120 (Blackwell) use the source PKGBUILD with an explicit
compute capability:

```bash
CUDA_COMPUTE_CAP=120 paru -S mold-ai
```

To upgrade: `paru -Syu mold-ai-bin` (or `mold-ai` / `mold-ai-git`). `mold update`
will detect a pacman-managed install and direct you here instead of attempting
to overwrite the binary.

To uninstall: `sudo pacman -R mold-ai-bin` (or whichever package you installed).

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
