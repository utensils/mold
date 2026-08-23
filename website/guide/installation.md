# Installation

## Native apps

- **macOS desktop:** download the signed and notarized
  [Mold DMG](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg),
  then follow the [Desktop App guide](/guide/desktop).
- **Windows desktop:** download the self-signed
  [NSIS installer](https://github.com/utensils/mold/releases/latest/download/Mold-windows-x64-self-signed.exe)
  and its [public certificate](https://github.com/utensils/mold/releases/latest/download/mold-windows-self-signing.cert.cer),
  then follow the [Windows desktop trust instructions](/guide/desktop#windows).
- **iPhone:** the current remote-only app is distributed through the project's
  invited internal and external TestFlight groups; there is not yet a public
  App Store listing. See the [iPhone App guide](/guide/iphone) for supported
  workflows and host setup.

Mold is CLI-native. The command-line installation below installs the primary
Mold interface and the same engine/server used by both native apps, scripts,
agents, and custom API clients.

## Windows CLI

The prebuilt x64 Windows CLI is a self-signed CPU inference and remote-client
build. Download and unpack it with PowerShell:

```powershell
$release = 'https://github.com/utensils/mold/releases/latest/download'
Invoke-WebRequest "$release/mold-x86_64-pc-windows-msvc-cpu.zip" `
  -OutFile mold-windows.zip
Expand-Archive .\mold-windows.zip `
  -DestinationPath "$env:LOCALAPPDATA\Mold\bin" `
  -Force
& "$env:LOCALAPPDATA\Mold\bin\mold.exe" version
```

The zip includes `mold-windows-self-signing.cert.cer`. Verify its SHA-1
thumbprint is `E8DA2990155CCC6E9278A8319008A763AC5DFC79` before trusting it;
the [Desktop App guide](/guide/desktop#windows) has the exact PowerShell trust
commands. Add `%LOCALAPPDATA%\Mold\bin` to your user `PATH` for a permanent
`mold` command. This published CLI does not include CUDA; use it for CPU work
or point it at a GPU host with `mold run --host http://gpu-host:7680 "prompt"`
or `MOLD_HOST`. Build from source on an x64 machine with the CUDA toolkit to get a
CUDA-enabled local binary; the [Windows contributor commands](/guide/desktop#commands)
cover the supported helper.

## One-Line Install (recommended)

Stable release:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

Downloads the **latest tagged release** from
[github.com/utensils/mold/releases/latest](https://github.com/utensils/mold/releases/latest)
and installs it to `~/.local/bin/mold`. On Linux, the installer queries every
GPU visible through `CUDA_VISIBLE_DEVICES` and selects a compatible release
binary independent of device order. It supports any homogeneous device count,
including RTX 3050/30-series, RTX 50-series, and named RTX PRO variants.
macOS builds include Metal support.

The installer downloads the exact release's `SHA256SUMS` and verifies the
selected archive before extraction. Missing sm86, sm100, or sm120
artifacts fail closed; a higher compute target is never substituted for a
lower-capability GPU. Only sm89 may use an old release's unsuffixed archive,
which was the historical name for that same target. Timeouts, authentication
failures, server failures, duplicate checksums, and checksum mismatches stop
installation.

### Nightly CLI

Install the latest successfully published CLI build from `main`:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | MOLD_CHANNEL=nightly sh
```

Nightly uses the rolling `latest` prerelease and the same platform detection
and SHA-256 verification as stable. It may be less tested than a tagged
release. Re-run that command or use `mold update --nightly` to refresh it.

### Options

All options are passed as environment variables:

```bash
# Install to a custom path
curl -fsSL ... | MOLD_INSTALL_DIR=/usr/local/bin sh

# Pin to a specific release tag (default: latest)
curl -fsSL ... | MOLD_VERSION=v0.10.0 sh

# Install the latest rolling build from main
curl -fsSL ... | MOLD_CHANNEL=nightly sh

# Force a GPU architecture (default: auto-detect on Linux)
curl -fsSL ... | MOLD_CUDA_ARCH=sm86  sh   # Ampere (RTX 3090 / A40)
curl -fsSL ... | MOLD_CUDA_ARCH=sm89  sh   # Ada (RTX 40-series)
curl -fsSL ... | MOLD_CUDA_ARCH=sm100 sh   # Datacenter Blackwell (B200 / B300)
curl -fsSL ... | MOLD_CUDA_ARCH=sm120 sh   # Consumer Blackwell (RTX 50-series)
```

> **Note:** the env var has to be on the `sh` side of the pipe — with
> `VAR=value curl ... | sh`, the variable only applies to `curl` and the
> installer itself still sees the default.

An explicit `MOLD_CUDA_ARCH` must equal the target selected for every visible
GPU. Homogeneous 8.6 and 8.9 fleets use sm86 and sm89 respectively. A mixed
8.6/8.9 fleet uses sm86: release CI requires exact sm86 PTX embedded in the
final executable, which both Ampere and the forward-compatible Ada driver can
JIT. Homogeneous 10.x and 12.x fleets use sm100 and sm120. Compute capability
8.0, 9.x, and unproven mixed families fail closed
because no release tarball is qualified for their floor. Narrow
`CUDA_VISIBLE_DEVICES` or build from source with verified targets.

PTX compatibility is forward-only toward equal-or-higher device compute
capabilities. An sm86 artifact can JIT on sm86 or sm89, but sm89 PTX cannot JIT
backward on an RTX 3090 (sm86). The RTX 3090 distribution gate therefore
requires a successful sm86 generation and treats the same-source sm89 failure
as an expected incompatibility regression, not a successful smoke.

`MOLD_VERSION` accepts any tag that exists on the
[releases page](https://github.com/utensils/mold/releases) — for example
`v0.8.0` to reproduce an older install. Without it the script follows the
`releases/latest` redirect on GitHub and installs whatever that currently
points at.

## Updating

```bash
mold update                       # Update to latest release
mold update --nightly             # Install latest rolling build from main
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
paru -S mold-ai         # Builds from source — set CUDA_COMPUTE_CAP for other GPUs
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

The existing `mold-ai-bin` package deliberately remains on sm_89. Use the
source PKGBUILD with an explicit compute capability for other families:

```bash
CUDA_COMPUTE_CAP=86 paru -S mold-ai   # RTX 3090 / A40
CUDA_COMPUTE_CAP=100 paru -S mold-ai  # B200 / B300
CUDA_COMPUTE_CAP=120 paru -S mold-ai  # RTX 50-series
```

There is no `mold-ai-bin-sm100` package before real B200 qualification.

To upgrade: `paru -Syu mold-ai-bin` (or `mold-ai` / `mold-ai-git`). `mold update`
will detect a pacman-managed install and direct you here instead of attempting
to overwrite the binary.

To uninstall: `sudo pacman -R mold-ai-bin` (or whichever package you installed).

## Nix

The flake is a source revision channel, not the self-updater's binary release
channel. Pin the flake input or Git revision for reproducibility. Nix packages
do not consult `MOLD_DISTRIBUTION_IMAGE_VERSION`; source and Nix-built cloud
clients default to the mutable `latest*` container channel unless a release
builder explicitly embeds an official distribution version.

```bash
# Run directly — no install needed
nix run github:utensils/mold -- run "a cat"

# RTX 3090 / A40
nix run github:utensils/mold#mold-sm86 -- run "a cat"

# B200 / B300
nix run github:utensils/mold#mold-sm100 -- run "a cat"

# RTX 50-series
nix run github:utensils/mold#mold-sm120 -- run "a cat"

# Add to your system profile
nix profile install github:utensils/mold
```

## From Source

Source builds likewise do not imply that a same-version GHCR image exists.
They use rolling `latest*` cloud images by default. For local CUDA compilation,
set `CUDA_COMPUTE_CAP` to the target required by the GPUs you intend to expose.

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

Requires Rust 1.93+ and CUDA toolkit (Linux) or Xcode (macOS).

Optional features can be added to the same build, for example
`--features cuda,preview,expand,discord,tui` or
`--features metal,preview,expand,discord,tui` if you also want terminal preview,
local prompt expansion, the Discord bot, or the interactive TUI.

`pulid` — [face identity](/guide/identity) — is one of those features, and it is
the only one with a build-time dependency of its own: `protoc` must be on `PATH`,
because `candle-onnx`'s build script drives `prost-build`. It is in the `nix
develop` shell already; otherwise install it before building, since the build
fails partway through without it.

```bash
brew install protobuf                       # macOS
sudo apt-get install -y protobuf-compiler   # Debian/Ubuntu
sudo pacman -S protobuf                     # Arch
sudo dnf install protobuf-compiler          # Fedora
```

Every official binary — the release tarballs, the Nix packages, the AUR
packages, and the desktop app — already ships with `pulid` on, so this only
concerns building it yourself.

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

The one-line installer targets the latest stable tag by default; with
`MOLD_CHANNEL=nightly`, it targets the rolling `latest` prerelease. Both use
assets from the [releases page](https://github.com/utensils/mold/releases):

| Platform                                         | File                                              |
| ------------------------------------------------ | ------------------------------------------------- |
| macOS Apple Silicon                              | `mold-aarch64-apple-darwin.tar.gz`                |
| Linux x86_64 (Ampere, RTX 3090 / A40)            | `mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz`  |
| Linux x86_64 (Ada, RTX 4090 / 40-series)         | `mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz`  |
| Linux x86_64 (datacenter Blackwell, B200 / B300) | `mold-x86_64-unknown-linux-gnu-cuda-sm100.tar.gz` |
| Linux x86_64 (consumer Blackwell, RTX 50-series) | `mold-x86_64-unknown-linux-gnu-cuda-sm120.tar.gz` |

B200 support is simulated, not hardware-qualified. The sm_100 artifact passes
hosted build, CUDA-image, loader, NVML, archive, and synthetic scheduler checks;
real 8×B200 and MIG qualification remain deferred.
GH200, GB200, and GB300 require future linux/arm64 artifacts and are unsupported.
Current Linux release archives and containers are amd64-only.

To install an older tag, put `MOLD_VERSION=<tag>` on the `sh` side of the
pipe, e.g. `curl -fsSL ... | MOLD_VERSION=v0.8.0 sh`. Placing it on the
`curl` side (`VAR=value curl ... | sh`) exports the variable to `curl` only;
the installer still sees the default and installs the latest release.

Pinned releases remain pinned. If an older tag lacks the selected native
sm86, sm100, or sm120 asset, the installer and `mold update --version` fail
closed. An old unsuffixed CUDA archive is considered only when the selected
target is sm89, because that was the former filename for the same target.

## Shell Completions

```bash
source <(mold completions bash)    # bash
source <(mold completions zsh)     # zsh
mold completions fish | source     # fish
```
