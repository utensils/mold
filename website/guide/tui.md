# Terminal UI <Badge type="warning" text="beta" />

mold includes an interactive terminal UI for browsing models, tuning parameters,
generating images with live progress, and previewing results — all without
leaving the terminal.

Built on [ratatui](https://ratatui.rs) with Kitty graphics protocol support for
pixel-perfect image preview in terminals like Ghostty, kitty, and WezTerm. Falls
back to halfblock rendering in other terminals.

::: warning Beta
The TUI is under active development. Core generation, model management, gallery, and image preview work well. Some features (prompt expansion, theme customization) are planned but not yet implemented.
:::

![mold TUI — Generate view with image preview](/gallery/tui-generate.png)

## Quick Start

```bash
mold tui
```

The TUI launches in the **Generate** view with your cursor in the prompt field.
Type a prompt, press **Enter**, and watch the progress panel as your image
generates. The result appears in the Preview panel and is saved to
`~/.mold/output/`.

::: tip
The `tui` feature must be compiled in. Pre-built releases and the Nix package include it by default. If building from source, add `--features tui` to your build command.
:::

## Auto-Start Server

By default, `mold tui` automatically starts a background `mold serve` process if
no server is already running. This keeps models hot between generations for
faster subsequent runs. The server is killed when you quit the TUI.

- `mold tui` — auto-starts server on `localhost:7680`
- `mold tui --local` — skip server, use local GPU only
- `mold tui --host http://gpu:7680` — connect to an existing remote server

### Server Logs

When the TUI auto-starts a background server, logs are written to
`~/.mold/logs/` with daily rotation. This is useful for debugging generation
failures since the server's stderr is suppressed while the TUI controls the
terminal.

Log files follow the naming pattern `mold-server.YYYY-MM-DD.log` and are
automatically cleaned up after 7 days (configurable via `logging.max_days` in
`~/.mold/config.toml`).

To view live logs while the TUI is running, open a second terminal:

```bash
tail -f ~/.mold/logs/mold-server.$(date +%Y-%m-%d).log
```

You can also enable file logging for manual `mold serve` with `--log-file`, or
permanently via the config file:

```toml
[logging]
level = "info"
file = true
# dir = "~/.mold/logs"
# max_days = 7
```

## Views

The TUI has four main views, shown as tabs at the top of the screen:

| View     | Purpose                                               |
| -------- | ----------------------------------------------------- |
| Generate | Write prompts, tune parameters, generate images/video |
| Gallery  | Browse generated images and videos with preview       |
| Models   | View installed and available models                   |
| Settings | View and edit all config.toml settings                |

Switch views with **Esc** then **1**/**2**/**3**/**4**, arrow keys, or click the
tabs. **Alt+1**/**Alt+2**/**Alt+3**/**Alt+4** works from anywhere.

## Generate View

The main workspace with five panels:

- **Prompt** — Multi-line text area (Shift+Enter for newlines, emacs
  keybindings)
- **Negative** — (CFG models only) Describes what to avoid
- **Parameters** — Model, dimensions, steps, guidance, seed, and more
- `qwen-image-edit` shows a source image and negative prompt without img2img
  `strength` or `mask` controls
- **Info** — Model description, system memory, process memory usage
- **Preview** — Generated image with Kitty/sixel/halfblock rendering

### Editing Parameters

Navigate to Parameters with **Tab** or click, then:

- **j**/**k** or arrow keys to move between fields
- **+**/**-** or left/right to adjust numeric values
- **Enter** or click to activate a field:
  - **Model** — opens the fuzzy model selector
  - **Frames** / **FPS** — (video models only) adjust frame count and FPS
  - **Seed** (mode row) — cycles random / fixed / increment
  - **Seed** (value row) — opens seed value input popup
  - **Format** / **Mode** / **Expand** / **Offload** — toggles the value
  - **Scheduler** — cycles through Ddim, Euler Ancestral, UniPC
  - **Reset** — restores all parameters to model defaults (keeps prompt)
  - **Unload** — unloads the model from GPU to free memory

### Seed Modes

Cycle with **Ctrl+R** or **+/-** on the Seed field:

| Mode      | Behavior                               |
| --------- | -------------------------------------- |
| random    | New random seed each generation        |
| fixed     | Same seed every time (reproducibility) |
| increment | Seed +1 after each generation          |

Press **Enter** on the Seed value row to type an exact value.

### Model Selector

Press **Enter** on the Model field or **Ctrl+M** from anywhere:

- Type to fuzzy-filter the model list
- **j**/**k**, arrow keys, or scroll wheel to navigate
- **Enter** to select — parameters update to model defaults
- **Esc** to cancel

### Prompt History

Previous prompts persist across sessions in `~/.mold/prompt-history.jsonl`:

- **Up/Down** arrows at top/bottom of prompt recall history
- **Ctrl+P**/**Ctrl+N** also navigate history
- **/** in navigation mode opens fuzzy search over all prompts

### Shell Keybindings

The prompt editor supports standard emacs/shell keybindings:

| Key    | Action              |
| ------ | ------------------- |
| Ctrl+A | Beginning of line   |
| Ctrl+E | End of line         |
| Ctrl+K | Kill to end of line |
| Ctrl+U | Kill to start       |
| Ctrl+W | Delete word back    |
| Ctrl+D | Delete forward      |
| Ctrl+F | Forward char        |
| Ctrl+B | Backward char       |

## Gallery View

Browse generated images stored in `~/.mold/output/` (or `MOLD_OUTPUT_DIR`).
Images are displayed as a thumbnail grid with cached 256x256 thumbnails for fast
loading. Only images with embedded `mold:parameters` metadata are shown (PNG and
JPEG).

### Grid Mode

| Key        | Action                           |
| ---------- | -------------------------------- |
| h/j/k/l    | Navigate the grid                |
| Arrow keys | Navigate the grid                |
| Enter      | Open detail view                 |
| e          | Load into Generate (edit)        |
| u          | Upscale with AI model            |
| d          | Delete image (with confirmation) |
| o          | Open in system viewer            |
| Esc        | Back to Generate                 |

### Detail Mode

Press **Enter** on a grid thumbnail to see the full image with all metadata.

| Key | Action                    |
| --- | ------------------------- |
| e   | Load into Generate (edit) |
| r   | Regenerate immediately    |
| u   | Upscale with AI model     |
| d   | Delete image              |
| o   | Open in system viewer     |
| j/k | Previous / next image     |
| Esc | Back to grid              |

### Thumbnails

Thumbnails are cached at `~/.mold/cache/thumbnails/` and generated automatically
on first scan and after each generation. Delete the cache directory to force
regeneration.

## Models View

See all installed and available models with family, size, defaults, and status.

| Key   | Action                        |
| ----- | ----------------------------- |
| j/k   | Navigate the model list       |
| Enter | Set as default model          |
| p     | Pull (download) a model       |
| u     | Unload the active model (GPU) |
| Esc   | Back to Generate              |

## Settings View

Edit all `config.toml` settings without leaving the TUI. Settings are organized
into four sections: **General**, **Expand**, **Logging**, and **Model
Defaults**.

Changes persist immediately to `config.toml` on each edit.

| Key        | Action                                       |
| ---------- | -------------------------------------------- |
| j/k        | Navigate settings                            |
| +/- or L/R | Adjust numeric or cycle toggle values        |
| Enter      | Edit text/path field (opens popup) or toggle |
| Esc        | Back to Generate                             |

### Field Types

- **Numeric** (port, width, steps, etc.) — adjust with +/- keys
- **Boolean** (metadata, expand enabled, etc.) — toggle with Enter or +/-
- **Toggle** (T5 variant, log level, scheduler) — cycle with +/- or Enter
- **Text/Path** (model name, directories, prompts) — Enter opens edit popup
- **Read-only** (model file paths) — displayed dimmed, not editable

Environment variable overrides are shown with an **(env)** indicator in yellow.
Per-model defaults show resolved values from the manifest (not raw config `None`
values), so you always see the effective runtime value.

### Model Defaults

The Model Defaults section shows settings for a specific model. Use
**Left/Right** on the Model selector row to cycle through configured models.
Editable fields include steps, guidance, dimensions, scheduler, negative prompt,
LoRA path, and LoRA scale. File paths (transformer, VAE) are read-only since
they are managed by `mold pull`.

## Qwen-Image-Edit

On this branch the TUI treats `qwen-image-edit` as a distinct edit family:

- single source image only in the TUI
- no img2img `strength`
- no inpainting mask
- no ControlNet or LoRA controls
- default width/height derived from the selected source image at roughly `1024x1024` area

Local inference uses the Qwen2.5-VL multimodal edit encoder. In v1 the TUI
keeps the flow single-image only, even though the CLI and API accept multiple
ordered `--image` inputs for `qwen-image-edit`.

## Navigation

Press **Esc** to enter navigation mode, where number keys and arrows switch
views.

### Global Shortcuts

| Key               | Action                            |
| ----------------- | --------------------------------- |
| Esc               | Unfocus / navigation mode         |
| 1 / 2 / 3 / 4     | Switch views (in navigation mode) |
| Left / Right      | Cycle views (in navigation mode)  |
| Alt+1 / 2 / 3 / 4 | Switch views (from anywhere)      |
| Tab               | Cycle focus to next panel         |
| Shift+Tab         | Cycle focus to previous panel     |
| Ctrl+C            | Quit                              |
| q                 | Quit (when not in a text field)   |
| ?                 | Show help overlay                 |

### Generate Shortcuts

| Key    | Context    | Action                           |
| ------ | ---------- | -------------------------------- |
| Enter  | Prompt     | Start generation                 |
| Enter  | Parameters | Activate field (selector/toggle) |
| Ctrl+G | Any        | Start generation                 |
| Ctrl+M | Any        | Open model selector              |
| Ctrl+R | Any        | Cycle seed mode                  |
| +/-    | Parameters | Adjust numeric value             |
| j/k    | Parameters | Navigate fields                  |

### Mouse Support

- Click tabs to switch views
- Click panels to focus them
- Click parameter rows to select and activate
- Click gallery thumbnails to select, double-click for detail view
- Click model rows to select
- Scroll wheel navigates lists and popups

## Session Persistence

All settings are saved to `~/.mold/tui-session.json` after each generation and
restored on next launch:

- Prompt and negative prompt text
- Model selection
- All generation parameters (dimensions, steps, guidance, seed mode, batch,
  format, scheduler, lora, expand, offload, strength)

Use **Reset** in the Actions section of Parameters to restore model defaults
without losing your prompt. **Unload** frees GPU memory by unloading the active
model.

Generated images are saved to `~/.mold/output/` by default (override with
`MOLD_OUTPUT_DIR` env var or `output_dir` in config). All images include
embedded PNG metadata that preserves the full generation parameters, making them
portable across machines.

## Info Panel

The Info panel below Parameters shows:

- **Model description** from the manifest
- **System memory** — free and available (macOS unified memory / NVIDIA VRAM)
- **Mold memory** — total physical footprint of all mold processes (includes
  mmap'd model weights)

## Server Fallback

The TUI uses a three-mode inference system:

| Mode   | Behavior                                           |
| ------ | -------------------------------------------------- |
| auto   | Try server first, fall back to local GPU (default) |
| local  | Force local GPU only                               |
| remote | Force remote server only (error if unreachable)    |

Cycle with **+/-** on the Mode field. The Host field (visible in auto/remote
mode) can be edited with **Enter** to point at a custom server. Host input is
normalized automatically: `hal9000` becomes `http://hal9000:7680`,
`hal9000:8080` becomes `http://hal9000:8080`, and full URLs like
`https://gpu.example.com` are used as-is. The TUI verifies connectivity before
switching.

## Image Preview

The TUI auto-detects your terminal's graphics protocol at startup:

| Protocol   | Terminals               | Quality        |
| ---------- | ----------------------- | -------------- |
| Kitty      | Ghostty, kitty, WezTerm | Pixel-perfect  |
| Sixel      | foot, xterm, mlterm     | Full color     |
| iTerm2     | iTerm2, Hyper           | Full color     |
| Halfblocks | Everything else         | Unicode blocks |

## Building with TUI Support

The TUI is behind the `tui` feature flag on `mold-ai`:

::: code-group

```bash [macOS (Metal)]
cargo build --release -p mold-ai --features metal,tui
```

```bash [Linux (CUDA)]
cargo build --release -p mold-ai --features cuda,tui
```

```bash [All features]
cargo build --release -p mold-ai --features metal,preview,discord,expand,tui
```

:::

The Nix flake, pre-built releases, and Docker images include the TUI by default.

## Theme

The TUI uses a Catppuccin Mocha color palette. Theme customization via
`config.toml` is planned for a future release.
