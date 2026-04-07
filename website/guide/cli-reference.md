# CLI Reference

## `mold run`

Generate images from text prompts.

```
mold run [MODEL] [PROMPT...] [OPTIONS]
```

The first positional argument is treated as MODEL if it matches a known model
name; otherwise it is the prompt. Prompt can also be piped via stdin.

### Options

| Flag                               | Description                                      |
| ---------------------------------- | ------------------------------------------------ |
| `-m, --model <MODEL>`              | Explicit model override                          |
| `-o, --output <PATH>`              | Output file (default: `./mold-{model}-{ts}.png`) |
| `--width <N>`                      | Image width                                      |
| `--height <N>`                     | Image height                                     |
| `--steps <N>`                      | Inference steps                                  |
| `--seed <N>`                       | Random seed                                      |
| `--batch <N>`                      | Number of images (1–16)                          |
| `--guidance <N>`                   | Guidance scale                                   |
| `--frames <N>`                     | Video frame count (8n+1, video models only)      |
| `--fps <N>`                        | Video frames per second (default: 24)            |
| `--format <FMT>`                   | `png`, `jpeg`, `gif`, `apng`, `webp`, `mp4`      |
| `--local`                          | Skip server, run locally                         |
| `--eager`                          | Keep all components loaded (more VRAM)           |
| `--offload`                        | CPU↔GPU block streaming (less VRAM)              |
| `--lora <PATH>`                    | LoRA adapter safetensors                         |
| `--lora-scale <FLOAT>`             | LoRA strength (0.0–2.0)                          |
| `-i, --image <PATH>`               | Source image for img2img (`-` for stdin)         |
| `--strength <FLOAT>`               | Denoising strength (0.0–1.0)                     |
| `--mask <PATH>`                    | Inpainting mask                                  |
| `--control <PATH>`                 | ControlNet control image                         |
| `--control-model <NAME>`           | ControlNet model name                            |
| `--control-scale <FLOAT>`          | ControlNet scale (0.0–2.0)                       |
| `-n, --negative-prompt`            | Negative prompt (CFG models)                     |
| `--no-negative`                    | Suppress config default negative                 |
| `--no-metadata`                    | Disable PNG metadata                             |
| `--preview`                        | Display image inline in terminal                 |
| `--expand`                         | Enable prompt expansion                          |
| `--no-expand`                      | Disable prompt expansion                         |
| `--expand-backend <URL>`           | Expansion backend URL                            |
| `--expand-model <MODEL>`           | LLM model for expansion                          |
| `--t5-variant <TAG>`               | T5 encoder variant                               |
| `--qwen3-variant <TAG>`            | Qwen3 encoder variant                            |
| `--qwen2-variant <TAG>`            | Qwen2.5-VL text encoder variant for Qwen-Image   |
| `--qwen2-text-encoder-mode <MODE>` | Qwen2.5-VL placement/staging mode for Qwen-Image |
| `--scheduler <SCHED>`              | Noise scheduler (ddim, euler-ancestral, uni-pc)  |
| `--host <URL>`                     | Override MOLD_HOST                               |

### Qwen-Image Encoder Controls

- `--qwen2-variant auto|bf16|q8|q6|q5|q4|q3|q2`
- `--qwen2-text-encoder-mode auto|gpu|cpu-stage|cpu`

`auto` keeps CUDA behavior unchanged. On Apple Metal/MPS, `auto` now prefers
quantized Qwen2.5-VL GGUF text encoders for Qwen-Image (`q6`, then `q4`) to
avoid the BF16 text-encoder memory spike. If you force `bf16` on Metal,
sequential mode still stages prompt conditioning through CPU after encoding to
reduce unified-memory pressure during denoising.

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
mold serve [--port N] [--bind ADDR] [--models-dir PATH] [--log-format json|text] [--log-file] [--discord]
```

| Flag                  | Description                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `--port <N>`          | Port to listen on, defaults to `7680` or `MOLD_PORT`              |
| `--bind <ADDR>`       | Bind address, defaults to `0.0.0.0`                               |
| `--models-dir <PATH>` | Override the models directory for this process                    |
| `--log-format <FMT>`  | `json` or `text`; defaults to `json` for production-friendly logs |
| `--log-file`          | Enable rotated file logging to `~/.mold/logs/`                    |
| `--discord`           | Also starts the built-in Discord bot in the same process          |

`--discord` is only available when the binary was built with the `discord`
feature. The bot still talks to the same local server API and reads
`MOLD_DISCORD_TOKEN` or `DISCORD_TOKEN` from the environment.

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

## `mold version`

Show version, build date, and git SHA.

## `mold completions`

Generate shell completions.

```bash
mold completions bash | zsh | fish
```
