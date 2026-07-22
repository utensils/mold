# mold

[![CI](https://github.com/utensils/mold/actions/workflows/ci.yml/badge.svg)](https://github.com/utensils/mold/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/utensils/mold/graph/badge.svg)](https://codecov.io/gh/utensils/mold)
[![FlakeHub](https://img.shields.io/endpoint?url=https://flakehub.com/f/utensils/mold/badge)](https://flakehub.com/flake/utensils/mold)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![Nix Flake](https://img.shields.io/badge/nix-flake-blue.svg)](https://nixos.wiki/wiki/Flakes)

Generate images and short video clips on your own GPU. No cloud, no Python, no fuss.

Mold is equally owned and maintained by its core contributors,
[James Brink](https://jamesbrink.online/) and
[Jeffrey Dilley](mailto:jeff.dilley@gmail.com).

**[Documentation](https://utensils.io/mold/)** | **[Getting Started](https://utensils.io/mold/guide/)** | **[Models](https://utensils.io/mold/models/)** | **[API](https://utensils.io/mold/api/)**

```bash
mold run "a cat riding a motorcycle through neon-lit streets"
```

That's it. Mold auto-downloads the model on first run and saves the image to your current directory.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | sh
```

This downloads the **latest tagged release** from
[releases/latest](https://github.com/utensils/mold/releases/latest) and
installs it to `~/.local/bin/mold`. On Linux, the installer auto-detects your
NVIDIA GPU and picks the right binary (RTX 40-series or RTX 50-series). macOS
builds include Metal support.

Pin a specific version with `MOLD_VERSION`:

```bash
curl -fsSL https://raw.githubusercontent.com/utensils/mold/main/install.sh | MOLD_VERSION=v0.10.0 sh
```

<details>
<summary>Other install methods</summary>

### Nix

```bash
nix run github:utensils/mold -- run "a cat"                   # Ada / RTX 40-series
nix run github:utensils/mold#mold-sm120 -- run "a cat"        # Blackwell / RTX 50-series
```

### AUR — Arch Linux

```bash
paru -S mold-ai-bin     # Prebuilt binary, CUDA sm_89 (RTX 40-series). Fastest.
paru -S mold-ai         # Builds from source — set CUDA_COMPUTE_CAP=120 for RTX 50-series
paru -S mold-ai-git     # Builds from main HEAD
```

Conflicts with `extra/mold` (the rui314 linker) — they share the `/usr/bin/mold`
path. See [`packaging/aur/README.md`](packaging/aur/README.md) for details and
the Blackwell (sm_120) build flag.

### From source

```bash
./scripts/ensure-web-dist.sh && cargo build --profile dev-fast -p mold-ai --features cuda   # Linux (NVIDIA), fast local build
./scripts/ensure-web-dist.sh && cargo build --profile dev-fast -p mold-ai --features metal  # macOS (Apple Silicon), fast local build
cargo build --release -p mold-ai --features cuda                                          # Linux (NVIDIA), shipping build
cargo build --release -p mold-ai --features metal                                         # macOS (Apple Silicon), shipping build
```

Add `preview`, `expand`, `discord`, or `tui` to the features list as needed.

### Manual download

Pre-built binaries on the [releases page](https://github.com/utensils/mold/releases).

</details>

## Web app

Every `mold serve` includes the Mold Studio web interface with Create, Library,
Models, Machines, and Settings. The Machines workspace keeps the serving host
and remembered remotes together. On builds with mDNS support, **Add machine →
Local network** asks the primary server for DNS-SD peers, then the browser tests
and connects to the selected peer directly; API keys remain per-host headers and
stable instance UUIDs prevent duplicate rows. See the [Machines guide](https://utensils.io/mold/guide/machines).
On phones, Create follows the compact production order: prompt and style,
model and core controls, Generate, canvas, then recent prints.
Host detail aggregates all GPUs plus CPU/system RAM and owns capability-gated
pause, resume, cancel-all, queued-job GPU lanes, rename, and forget actions.
Library merges prints from connected hosts without falling back across host
boundaries, restores model-aware settings, and exposes the full recorded
generation metadata with prompt/seed copy and an Upscale handoff to Create.
Create preflights peak VRAM against the selected machine, exposes LTX-2 camera
motion presets, and reconnects to the last durable sequence after a reload.
Machine chips filter that merged feed, tiles expose a context action menu, and
the visible Library refreshes automatically as new prints land.
Keyboard users get trapped focus, reliable Escape dismissal, opener focus
restoration, and scroll locking across the web app's modal and sheet surfaces.
The global chrome reports when the serving engine is offline, and its ranked
command palette covers workspace sub-surfaces as well as models and themes.
Models refreshes Installed as pulls complete, exposes missing component status
with exact Repair actions, and explains that loaded models must be unloaded
before deletion.
Settings exposes the serving engine's complete configuration and profile
surface with search, per-key provenance, environment-override locks, typed
editors, and individual reset controls; prompt-expansion preferences are
available there alongside appearance and browser-local catalog tokens.

## Desktop app

Mold also has a native desktop app for macOS and Linux. It brings the Create,
Library, Models, Machines, and Settings workspaces into one interface — chains
live inside Create, history inside Library, and the job queue and RunPod inside
Machines — and can use this device alongside multiple remote Mold hosts. The
retired `/jobs` URL redirects to Machines; queue controls remain on their owning
machine, and only the five documented workspace shortcuts are active. The
Models workspace splits into Installed and Discover segments, keeps image/video
filters and pinned downloads in Discover, and lets you choose which host
receives a download. Its Installed segment merges every connected host with host
badges, while both that segment and each host detail page show host-scoped pull
progress. Size labels
separate checkpoint weights from the larger footprint including shared runtime
components.
Curated built-in variants take precedence over ambiguous multi-checkpoint
Hugging Face repositories, so a pull targets one runnable model instead of an
entire aggregate repository.
Every remembered remote host is retried immediately whenever the app launches,
independently of This Mac's startup; unreachable hosts stay visible and retry.
Pod-routed Create jobs show their live RunPod accrued cost in the activity strip.
Chain authoring inside Create also uses the all-host video-model union and keeps
creation, progress, previews, and durable job actions routed to the selected
model's host.
Desktop Settings → About links to the public
[Mold privacy policy](https://utensils.io/mold/privacy), matching the iPhone app.

Desktop prompt expansion follows the visible Batch count. Batch 1 is a quick
rewrite with undo; larger batches prepare exactly N editable variations for
review before any work is queued. Expansion and every generated sibling stay
on one resolved host (including Batch 1's next generation), and changes to the
source prompt, model, host, or count preserve the prepared work while requiring
an explicit refresh or discard. Library records the prepared batch identity and
each sibling's position. If that host lacks the expansion model, Create keeps
the recovery in place and shows its exact-host pull from connection through
queue, byte/file progress, readiness, or retry without hiding reviewed prompts.

**[Download Mold for macOS](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)** · **[Desktop guide](https://utensils.io/mold/guide/desktop)**

Web, desktop, and iPhone frontend dependencies are installed once from the
private repo-root Bun workspace. `studio/` contains browser-safe API contracts,
Pinia state, and shared domain logic; `ui/` contains Mold Studio tokens and
low-level Vue primitives; `web/` and `desktop/` are platform shells. Run `bun
install --frozen-lockfile` at the repository root, followed by `bun run
build:web`, `bun run build:desktop`, or `bun run build:mobile`.

The browser uses `/create`, `/library`, `/models`, `/machines`, and `/settings`.
The root redirects to Create; retired `/generate` and `/catalog` URLs render
Page Not Found and the web client requires the current server API contract.

The macOS DMG is signed and notarized. Linux builds are currently available
through Nix or as source/CI artifacts; tagged releases do not publish an
AppImage yet. Detailed setup, multi-host behavior, and update-channel guidance
live in the desktop guide.

## iPhone app

Mold for iPhone is a remote-only Tauri companion with first-class Create,
Library, Models, and Machines views, plus a dedicated Settings screen. Add a host by LAN discovery, IP address,
hostname, or Tailscale MagicDNS name; generation, model management, and media
remain on that remote Mold server. Host details expose live resources, queue,
downloads, and installed models, while Models can browse one host and send a
pull to another without silently changing the host selected for generation.
Pull actions immediately progress through Connecting, Starting, Queued, and
Pulling states, with live percentage and cancellation instead of leaving the
button looking idle.
The app uses the same HTTP, SSE, model defaults, resolution presets, and
generation request contract as the desktop app. New installs use the Safelight
theme family with system appearance, while saved theme choices persist. Create brings its
capability-aware controls to touch: Batch 1 prompt expansion with undo, Batch N
prepared-variation review, recent prompts, saved templates, source/edit images, masks, ControlNet, LoRA,
schedulers, CFG++, upscaling, and the video/LTX-2 pipeline controls. Estimates
come from the selected remote host, while orientation, proportional aspect-ratio
buttons, resolution tiers, and explicit Random/Fixed seed modes keep the common
choices direct. Every queued print has independent status and cancellation.
Prepared variations stay bound to the selected host's snapshotted endpoint,
Keychain credential, and server identity through expansion, any required model
pull, source preprocessing, and sibling submission. The phone requires exactly
the visible Batch count, preserves edited work with named stale reasons, and
shows batch position and source-prompt provenance in Library. A missing-model
pull leases that frozen route only for the attempt, shares a compatible Models
pull already in Starting, and returns authority to Models on terminal,
stale, or superseded outcomes without discarding the recovery record.
Saved results stream from the host instead of crossing the iPhone WebView as
encoded media; full-screen images swipe between prints, videos retain native
playback controls, and a still can become the next generation's source image.
Settings offers Mold or Safelight color families in System, Dark, or Light
appearance and keeps native iOS chrome in sync. The app prevents input-focus and
double-tap page zoom, removes rubber-band overscroll, and preserves the
gallery's horizontal swipe gesture. Settings links to the public
[Mold privacy policy](https://utensils.io/mold/privacy), which describes local
storage and user-selected remote-server traffic. Internal TestFlight builds are produced
after mobile-relevant `main` changes pass iOS CI; release checks verify Tauri's
embedded `index.html`, the opaque Mold-branded Apple icon catalog, App Store
Connect `VALID` processing, and internal tester access.

On macOS, `nix develop` exposes `ios-dev`, `ios-run`, `ios-check`, and
`ios-build`. Xcode, CocoaPods, and Apple signing are required for device and App
Store builds. See the [iPhone guide](https://utensils.io/mold/guide/iphone) for
host setup, Tailscale, every current screen, and TestFlight boundaries.

## Usage

```bash
mold run "a sunset over mountains"                    # Generate with default model
mold run flux-dev:q4 "a turtle in the desert"         # Pick a model
mold run "a portrait" --width 768 --height 1024       # Custom size
mold run "a sunset" --batch 4 --seed 42               # Batch with reproducible seeds
mold run "oil painting" --image photo.png              # img2img
mold run qwen-image-edit-2511:q4 "make the chair red leather" --image chair.png --image swatch.png
mold run ltx-video-0.9.6-distilled:bf16 "a fox in the snow" --frames 25
mold run "a cat" --expand                              # LLM prompt expansion
mold run qwen-image:q2 "a poster" --qwen2-variant q6  # Qwen-Image quantized text encoder
mold run flux-dev:bf16 "portrait" --lora style.safetensors  # LoRA adapter
```

Quantized Qwen-Image-Edit uses the safer split-CFG CUDA path at every
resolution. If CUDA reports a context-killing fault such as an illegal address,
Mold quarantines that GPU worker and exits so service supervision can recreate the context.

### Inline preview

Display generated images directly in the terminal (requires `preview` feature):

```bash
mold run "a cat" --preview
```

<p align="center">
  <img src="docs/terminal-preview-example.png" alt="Generating the mold logo with --preview in Ghostty" width="720" />
  <br/>
  <em>Generating the mold logo with <code>--preview</code> in Ghostty</em>
</p>

### Piping

```bash
mold run "neon cityscape" | viu -                     # Pipe to image viewer
echo "a cat" | mold run flux-schnell                  # Pipe prompt from stdin
```

### Terminal UI

```bash
mold tui
```

<p align="center">
  <img src="website/public/gallery/tui-generate.png" alt="mold TUI — Create view with image preview" width="720" />
  <br/>
  <em>The TUI Create view with Kitty graphics protocol image preview in Ghostty</em>
</p>

### Model management

```bash
mold list                    # See what you have
mold pull flux-dev:q4        # Download a model
mold rm dreamshaper-v8       # Remove a model
mold stats                   # Disk usage overview
mold clean                   # Clean orphaned files (dry-run)
mold clean --force           # Actually delete
```

### Remote rendering

```bash
# On your GPU server
mold serve

# From your laptop
MOLD_HOST=http://gpu-server:7680 mold run "a cat"
```

### Cloud GPU via `mold runpod`

Generate on a cloud GPU without managing pods yourself:

```bash
mold config set runpod.api_key <key>         # one-time setup
mold runpod run "a cat on a skateboard"       # creates pod → generates → saves to ./mold-outputs/
mold runpod network-volume create --name models --size 100 --dc US-KS-2
mold runpod run "a cat" --network-volume <volume-id>
```

`mold runpod run` selects an available GPU, streams progress, and keeps the pod
warm for reuse. See the [RunPod CLI guide](https://utensils.io/mold/deployment/runpod-cli)
for provisioning, storage, diagnostics, and lifecycle commands.

See the full [CLI reference](https://utensils.io/mold/guide/cli-reference), [configuration guide](https://utensils.io/mold/guide/configuration), and [model catalog](https://utensils.io/mold/models/) in the documentation.

## Models

Supports 11 model families with 80+ variants:

| Family              | Models                     | Highlights                                                       |
| ------------------- | -------------------------- | ---------------------------------------------------------------- |
| **FLUX.1**          | schnell, dev, + fine-tunes | Best quality, 4-25 steps, LoRA support                           |
| **Flux.2 Klein**    | 4B and 9B                  | Fast 4-step, low VRAM, default model                             |
| **SDXL**            | base, turbo, + fine-tunes  | Fast, flexible, negative prompts                                 |
| **SD 1.5**          | base + fine-tunes          | Lightweight, ControlNet support                                  |
| **SD 3.5**          | large, medium, turbo       | Triple encoder, high quality                                     |
| **Z-Image**         | turbo                      | Fast 9-step, Qwen3 encoder                                       |
| **Qwen-Image**      | base + 2512                | High resolution, CFG guidance, GGUF quant support                |
| **Qwen-Image-Edit** | 2511                       | Multimodal image editing, repeatable `--image`, negative prompts |
| **Wuerstchen**      | v2                         | 42x latent compression                                           |
| **LTX-2 / LTX-2.3** | 19B, 22B                   | Joint audio-video generation, MP4-first workflows                |
| **LTX Video**       | 0.9.6, 0.9.8               | Text-to-video with APNG/GIF/WebP/MP4 output                      |

Bare names auto-resolve: `mold run flux-schnell "a cat"` picks the best available variant.

See the full [model catalog](https://utensils.io/mold/models/) for sizes, VRAM requirements, and recommended settings.

## Features

- **txt2img, img2img, multimodal edit, inpainting** — full generation pipeline
- **Image upscaling** — Real-ESRGAN super-resolution (2x/4x) via `mold upscale`, server API, or TUI; first use auto-downloads the upscaler on the host running the job, and post-generation upscale retains both original and upscaled gallery artifacts
- **LoRA adapters** — FLUX, Flux.2, LTX-2, SD1.5, SD3/SD3.5, SDXL,
  Qwen-Image, Qwen-Image-Edit, and Z-Image
- **ControlNet** — canny, depth, openpose (SD1.5)
- **Prompt expansion** — local LLM (Qwen3-1.7B) enriches short prompts
- **Negative prompts** — CFG-based models (SD1.5, SDXL, SD3, Wuerstchen)
- **Pipe-friendly** — `echo "a cat" | mold run | viu -`
- **PNG metadata** — embedded prompt, seed, model info
- **Terminal preview** — Kitty, Sixel, iTerm2, halfblock
- **Smart VRAM** — quantized encoders, block offloading, drop-and-reload
- **Qwen family encoder control** — selectable Qwen2.5-VL variants for Qwen-Image and Qwen-Image-Edit, with quantized auto-fallback when BF16 would be too heavy
- **Shell completions** — bash, zsh, fish, elvish, powershell
- **REST API** — `mold serve` with SSE streaming, auth, rate limiting
- **Discord bot** — slash commands with role permissions and quotas
- **Interactive TUI** — the five Mold Studio workspaces in the terminal:
  Create, an all-machines Library (prints merged from every connected host,
  details side panel, `/` filter), Models, multi-host Machines (connect
  remote servers, telemetry, queue lanes, generation target), and Settings
- **Native desktop** — local and multi-host generation across the Create,
  Library, Models, Machines, and Settings workspaces (chains, history, the job
  queue, and RunPod included)
- **Native iPhone** — remote multi-host generation, library, models, machine
  telemetry, and appearance settings through internal or external TestFlight

## Deployment

| Method              | Guide                                                                    |
| ------------------- | ------------------------------------------------------------------------ |
| **NixOS module**    | [Deployment: NixOS](https://utensils.io/mold/deployment/nixos)           |
| **Docker / RunPod** | [Deployment: Docker](https://utensils.io/mold/deployment/docker)         |
| **mold runpod CLI** | [Deployment: RunPod CLI](https://utensils.io/mold/deployment/runpod-cli) |
| **Systemd**         | [Deployment: Overview](https://utensils.io/mold/deployment/)             |

## How it works

Single Rust binary built on [candle](https://github.com/huggingface/candle) for the in-tree model families. LTX-2 now runs through the native Rust stack in `mold-inference`, so the full model surface stays in Rust with no libtorch dependency.

```
mold run "a cat"
  │
  ├─ Server running? → send request over HTTP
  │
  └─ No server? → load model locally on GPU
       ├─ Encode prompt (T5/CLIP text encoders)
       ├─ Denoise latent (transformer/UNet)
       ├─ Decode pixels (VAE)
       └─ Save PNG
```

## Requirements

- **NVIDIA GPU** with CUDA or **Apple Silicon** with Metal
- Models auto-download on first use (~2-30GB depending on model)
