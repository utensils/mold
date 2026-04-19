# CLI Reference

## `mold run`

Generate images from text prompts.

```
mold run [MODEL] [PROMPT...] [OPTIONS]
```

The first positional argument is treated as MODEL if it matches a known model
name; otherwise it is the prompt. Prompt can also be piped via stdin.

### Options

| Flag                               | Description                                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `-m, --model <MODEL>`              | Explicit model override                                                                                         |
| `-o, --output <PATH>`              | Output file (default: `./mold-{model}-{ts}.png`)                                                                |
| `--width <N>`                      | Image width                                                                                                     |
| `--height <N>`                     | Image height                                                                                                    |
| `--steps <N>`                      | Inference steps                                                                                                 |
| `--seed <N>`                       | Random seed                                                                                                     |
| `--batch <N>`                      | Number of images (1+)                                                                                           |
| `--guidance <N>`                   | Guidance scale                                                                                                  |
| `--frames <N>`                     | Video frame count (8n+1, video models only)                                                                     |
| `--fps <N>`                        | Video frames per second (default: 24)                                                                           |
| `--audio`, `--no-audio`            | Keep or strip the synchronized audio track for LTX-2 MP4 output                                                 |
| `--audio-file <PATH>`              | Conditioning audio file for LTX-2 audio-to-video                                                                |
| `--video <PATH>`                   | Source video for LTX-2 retake / video-conditioning flows                                                        |
| `--keyframe <FRAME:PATH>`          | LTX-2 keyframe conditioning. Repeat for multiple keyframes                                                      |
| `--pipeline <MODE>`                | LTX-2 pipeline: `one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2vid`, `retake` |
| `--retake <START:END>`             | LTX-2 retake time range in seconds                                                                              |
| `--camera-control <NAME\|PATH>`    | LTX-2 camera-control preset name or explicit `.safetensors` path                                                |
| `--spatial-upscale <MODE>`         | LTX-2 spatial upscaling (`x2` across the family, `x1.5` for `ltx-2.3-*`)                                        |
| `--temporal-upscale <MODE>`        | LTX-2 temporal upscaling (`x2` in the native runtime)                                                           |
| `--format <FMT>`                   | `png`, `jpeg`, `gif`, `apng`, `webp`, `mp4`                                                                     |
| `--local`                          | Skip server, run locally                                                                                        |
| `--eager`                          | Keep all components loaded (more VRAM)                                                                          |
| `--offload`                        | CPU↔GPU block streaming (less VRAM)                                                                             |
| `--lora <PATH>`                    | LoRA adapter safetensors. Repeat for stacked LTX-2 adapters                                                     |
| `--lora-scale <FLOAT>`             | LoRA strength (0.0–2.0)                                                                                         |
| `-i, --image <PATH>`               | Source image. Repeat for `qwen-image-edit`; `-` stdin is single-image only                                      |
| `--strength <FLOAT>`               | Denoising strength (0.0–1.0)                                                                                    |
| `--mask <PATH>`                    | Inpainting mask                                                                                                 |
| `--control <PATH>`                 | ControlNet control image                                                                                        |
| `--control-model <NAME>`           | ControlNet model name                                                                                           |
| `--control-scale <FLOAT>`          | ControlNet scale (0.0–2.0)                                                                                      |
| `-n, --negative-prompt`            | Negative prompt (CFG models)                                                                                    |
| `--no-negative`                    | Suppress config default negative                                                                                |
| `--no-metadata`                    | Disable PNG metadata                                                                                            |
| `--preview`                        | Display image inline in terminal                                                                                |
| `--expand`                         | Enable prompt expansion                                                                                         |
| `--no-expand`                      | Disable prompt expansion                                                                                        |
| `--expand-backend <URL>`           | Expansion backend URL                                                                                           |
| `--expand-model <MODEL>`           | LLM model for expansion                                                                                         |
| `--t5-variant <TAG>`               | T5 encoder variant                                                                                              |
| `--qwen3-variant <TAG>`            | Qwen3 encoder variant                                                                                           |
| `--qwen2-variant <TAG>`            | Qwen2.5-VL text encoder variant for the Qwen family                                                             |
| `--qwen2-text-encoder-mode <MODE>` | Qwen2.5-VL placement/staging mode for the Qwen family                                                           |
| `--scheduler <SCHED>`              | Noise scheduler (ddim, euler-ancestral, uni-pc)                                                                 |
| `--host <URL>`                     | Override MOLD_HOST                                                                                              |

### Qwen Family Encoder Controls

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

`auto` prefers BF16 on CUDA when enough headroom remains after the transformer
load, and falls back to quantized GGUF variants for resident/edit paths when
that BF16 encoder would be too heavy. On Apple Metal/MPS, `auto` now prefers
quantized Qwen2.5-VL GGUF text encoders for Qwen-Image (`q6`, then `q4`) to
avoid the BF16 text-encoder memory spike. If you force `bf16` on Metal,
sequential mode still stages prompt conditioning through CPU after encoding to
reduce unified-memory pressure during denoising.

### Repeated `--image`

- Non-edit families still accept at most one `--image`; it maps to `source_image`.
- `qwen-image-edit-2511:*` treats repeated `--image` flags as ordered `edit_images`.
- `qwen-image-edit` does not support `--image -`.
- `qwen-image-edit` supports quantized `--qwen2-variant` values by pairing GGUF language weights with the staged Qwen2.5-VL vision tower used for image conditioning.
- The first edit image drives the default output width/height when you omit both flags.

### LTX-2 Notes

- This family defaults to `mp4` when you do not explicitly choose another video format.
- If you explicitly choose `gif`, `apng`, or `webp`, mold exports a silent animation.
- `--camera-control dolly-in|dolly-left|dolly-out|dolly-right|jib-down|jib-up|static`
  auto-resolves the published LTX-2 19B camera LoRAs.
- LTX-2 runs natively in Rust inside `mold-inference`; no Python bridge or
  upstream checkout is required.
- Backend policy: CUDA is supported, CPU is correctness-only, and Metal is
  unsupported for this family.
- On 24 GB Ada GPUs such as the RTX 4090, the verified local FP8 path uses
  native staged loading, layer streaming, and the compatible `fp8-cast` mode
  rather than Hopper-only `fp8-scaled-mm`.
- The native CUDA matrix is validated across text+audio-video, image-to-video,
  audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale, and
  temporal upscale workflows.
- `mold serve` accepts inline source media up to `64 MiB` by default, which is
  enough for common retake and audio-to-video requests without extra server
  tuning.

## `mold expand`

Preview prompt expansion without generating.

```bash
mold expand <PROMPT> [OPTIONS]
```

| Flag                     | Description                |
| ------------------------ | -------------------------- |
| `-m, --model <MODEL>`    | Target model (for style)   |
| `--variations <N>`       | Number of variations       |
| `--json`                 | Output as JSON array       |
| `--backend <URL>`        | Expansion backend override |
| `--expand-model <MODEL>` | LLM model override         |

## `mold serve`

Start the HTTP inference server.

```bash
mold serve [--port N] [--bind ADDR] [--models-dir PATH] [--gpus SPEC] [--queue-size N] [--log-format json|text] [--log-file] [--discord]
```

| Flag                  | Description                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--port <N>`          | Port to listen on, defaults to `7680` or `MOLD_PORT`                                                                             |
| `--bind <ADDR>`       | Bind address, defaults to `0.0.0.0`                                                                                              |
| `--models-dir <PATH>` | Override the models directory for this process                                                                                   |
| `--gpus <SPEC>`       | Which GPUs to use: comma-separated ordinals (`0,1`) or `all`. Defaults to every visible GPU (see [Multi-GPU](#multi-gpu))        |
| `--queue-size <N>`    | Max queued generation jobs across all workers; overflow returns HTTP 503 + `Retry-After`. Defaults to `200` or `MOLD_QUEUE_SIZE` |
| `--log-format <FMT>`  | `json` or `text`; defaults to `json` for production-friendly logs                                                                |
| `--log-file`          | Enable rotated file logging to `~/.mold/logs/`                                                                                   |
| `--discord`           | Also starts the built-in Discord bot in the same process                                                                         |

`--discord` is only available when the binary was built with the `discord`
feature. The bot still talks to the same local server API and reads
`MOLD_DISCORD_TOKEN` or `DISCORD_TOKEN` from the environment.

### Multi-GPU

Each GPU gets a dedicated worker thread with its own model cache. Jobs are
routed across workers using idle-first placement with a VRAM-fit fallback, so
two different models can load onto two different cards and run concurrently.
Same-model parallel requests serialize on whichever GPU already has that model
loaded.

- `--gpus all` (default): use every visible CUDA/Metal device.
- `--gpus 0,1`: pin the server to specific ordinals.
- `MOLD_GPUS=0` is the env equivalent and is useful inside systemd / container
  units.

When the queue is full, generation endpoints return HTTP 503 with a
`Retry-After` header; `mold run` and the web UI handle the backpressure
transparently. `GET /api/status` returns a `gpus[]` array with per-worker state
(model, residency, in-flight count, free VRAM), and every `GenerateResponse`
carries a `"gpu": <ordinal>` field so clients can see which card produced a
given output.

## `mold pull`

Download a model from HuggingFace.

```bash
mold pull <MODEL> [--skip-verify]
```

`mold pull` writes a `.pulling` marker while a download is in progress and
removes it on success. If a pull is interrupted, `mold list` will show the model
as incomplete and a later `mold pull` or `mold rm` can clean things up.

`--skip-verify` disables the post-download SHA-256 integrity check. That is
mainly useful when an upstream file has intentionally changed before the
expected checksum in `manifest.rs` has been updated yet.

## `mold list`

List configured and available models with download status and disk usage.

Installed models are shown with these columns:

| Column        | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| `NAME`        | Model name, with `★` for the default model and `●` if loaded |
| `FAMILY`      | Model family such as FLUX, SDXL, or Z-Image                  |
| `SIZE`        | Model-specific download size                                 |
| `DISK`        | Actual local disk usage for that model                       |
| `STEPS`       | Default inference step count                                 |
| `GUIDANCE`    | Default guidance value                                       |
| `WIDTH`       | Default image width                                          |
| `HEIGHT`      | Default image height                                         |
| `DESCRIPTION` | Summary, plus `[gated]` or `[incomplete]` when relevant      |

The "Available to pull" section adds a `FETCH` column, which is how much data
still needs to be downloaded after accounting for shared cached files.

## `mold info`

Show installation overview, or model details with optional SHA-256 verification.

```bash
mold info              # overview
mold info flux-dev:q4  # model details
mold info --verify     # verify all checksums
```

## `mold default`

Get or set the default model.

```bash
mold default              # show current
mold default flux-dev:q4  # set default
```

## `mold config`

View and edit configuration settings using dot-notation keys.

```bash
mold config list [--json]
mold config get <KEY> [--raw]
mold config set <KEY> <VALUE>
mold config path
mold config edit
```

### Subcommands

| Subcommand | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| `list`     | Show all settings grouped by section. `--json` for machine-readable output. |
| `get`      | Get a single value. `--raw` outputs bare value for scripting.               |
| `set`      | Set a value and persist to config.toml. Validates type and range.           |
| `path`     | Show the config file path.                                                  |
| `edit`     | Open config file in `$EDITOR` (falls back to `$VISUAL`, then `vi`).         |

### Key Names

Keys use dot-notation matching the TOML structure:

| Section   | Keys                                                                                                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| General   | `default_model`, `models_dir`, `output_dir`, `server_port`, `default_width`, `default_height`, `default_steps`, `embed_metadata`, `t5_variant`, `qwen3_variant`, `default_negative_prompt` |
| Expand    | `expand.enabled`, `expand.backend`, `expand.model`, `expand.api_model`, `expand.temperature`, `expand.top_p`, `expand.max_tokens`, `expand.thinking`                                       |
| Logging   | `logging.level`, `logging.file`, `logging.dir`, `logging.max_days`                                                                                                                         |
| Per-model | `models.<name>.<field>` where field is one of: `default_steps`, `default_guidance`, `default_width`, `default_height`, `scheduler`, `negative_prompt`, `lora`, `lora_scale`                |

### Examples

```bash
# List all settings
mold config list

# Scripting: capture a value
PORT=$(mold config get server_port --raw)

# Set values with validation
mold config set server_port 8080
mold config set expand.enabled true
mold config set logging.level debug

# Clear optional fields
mold config set output_dir none

# Per-model defaults
mold config set models.flux-dev:q4.default_steps 30
mold config set models.sd15:fp16.scheduler euler-ancestral

# JSON output for tooling
mold config list --json | jq '.server_port'
```

Boolean values accept `true`/`false`, `on`/`off`, or `1`/`0`. Use `none` to
clear optional string fields. Environment variable overrides are flagged in
`list` output and warned about when using `set`.

## `mold tui`

Launch the interactive terminal UI.

```bash
mold tui [--host URL] [--local]
```

| Flag           | Description                     |
| -------------- | ------------------------------- |
| `--host <URL>` | Server URL override             |
| `--local`      | Skip server, use local GPU only |

Requires the `tui` feature flag (included in pre-built releases and Nix
packages). See the full [TUI documentation](/guide/tui) for views, keybindings,
and configuration.

## `mold discord`

Start the Discord bot (connects to a running `mold serve` via `MOLD_HOST`).

```bash
mold discord
```

Requires the `discord` feature flag. The bot can also be started alongside the
server with `mold serve --discord`. See [Discord Bot](/api/discord) for setup.

## `mold stats`

Show disk usage overview for models, output, logs, and shared components.

```bash
mold stats [--json]
```

| Flag     | Description             |
| -------- | ----------------------- |
| `--json` | Machine-readable output |

## `mold clean`

Clean up orphaned files, stale downloads, and old output images. Dry-run by
default — shows what would be removed without deleting anything.

```bash
mold clean [--force] [--older-than DURATION]
```

| Flag                      | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `--force`                 | Actually delete files (default is dry-run)           |
| `--older-than <DURATION>` | Include output images older than this (e.g. 30d, 7d) |

Detects stale `.pulling` markers from interrupted downloads, orphaned shared
files not referenced by any model, hf-cache transient files, and output images
older than the specified age.

## `mold server`

Manage a background mold server daemon.

### `mold server start`

Start the server as a detached background process.

```bash
mold server start [--port N] [--bind ADDR] [--models-dir PATH] [--log-file]
```

| Flag                  | Description                         |
| --------------------- | ----------------------------------- |
| `--port <N>`          | Server port (default: 7680)         |
| `--bind <ADDR>`       | Bind address (default: 0.0.0.0)     |
| `--models-dir <PATH>` | Override models directory           |
| `--log-file`          | Enable file logging (default: true) |

Writes a PID file to `~/.mold/mold-server.pid` for lifecycle tracking.

### `mold server status`

Show the status of the managed server (PID, port, version, uptime, models, GPU).

### `mold server stop`

Gracefully stop the managed server. Tries HTTP shutdown first, falls back to
SIGTERM, then SIGKILL.

## `mold rm`

Remove downloaded models.

```bash
mold rm <MODELS...> [--force]
```

`--force` skips the interactive confirmation prompt. Shared files such as VAEs,
encoders, and tokenizers are preserved until no remaining model references them.

## `mold ps`

Show server status and loaded model. When the server is unreachable, scans for
running `mold` processes (e.g. `mold run --local`) and displays their PID,
subcommand, runtime, and memory usage.

## `mold unload`

Unload the current model from the server to free GPU memory.

## `mold update`

Update mold to the latest release from GitHub.

```bash
mold update [OPTIONS]
```

Downloads the correct platform-specific binary, verifies its SHA-256 checksum,
and replaces the current binary in-place.

### Options

| Flag              | Description                                   |
| ----------------- | --------------------------------------------- |
| `--check`         | Only check for updates, don't install         |
| `--force`         | Reinstall even if the current version matches |
| `--version <TAG>` | Install a specific version (e.g. `v0.7.0`)    |

Respects `GITHUB_TOKEN` for API authentication (avoids rate limits). On Linux,
respects `MOLD_CUDA_ARCH` for GPU architecture override.

If mold was installed via a package manager (Nix, Homebrew), the command will
detect the read-only install path and suggest using the package manager instead.

## `mold version`

Show version, build date, and git SHA.

## `mold completions`

Generate shell completions.

```bash
mold completions bash | zsh | fish
```
