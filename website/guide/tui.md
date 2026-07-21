# Terminal UI

mold includes an interactive terminal UI for browsing models, tuning parameters,
generating images with live progress, and previewing results — all without
leaving the terminal.

Built on [ratatui](https://ratatui.rs) with Kitty graphics protocol support for
pixel-perfect image preview in terminals like Ghostty, kitty, and WezTerm. Falls
back to halfblock rendering in other terminals.

::: warning Beta
The TUI is under active development. Core generation, script-mode video
authoring, model management, gallery, settings, theming, and image preview
work well.
:::

![mold TUI — Create view with image preview](/gallery/tui-generate.png)

## Quick Start

```bash
mold tui
```

The TUI launches in the **Create** workspace with your cursor in the prompt field.
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

## Workspaces

The TUI shares the five Mold Studio workspaces with the desktop, web, and
iPhone apps, shown as tabs at the top of the screen:

| Workspace | Key | Purpose                                               |
| --------- | --- | ----------------------------------------------------- |
| Create    | 1   | Write prompts, tune parameters, generate images/video |
| Library   | 2   | Browse generated prints with preview                  |
| Models    | 3   | View installed and available models                   |
| Machines  | 4   | Connect remote Mold hosts, telemetry, queue, target   |
| Settings  | 5   | Theme picker plus file-backed and DB-backed settings  |

Switch workspaces with **Esc** then **1**–**5**, or click the tabs.
**Alt+1**–**Alt+5** works from anywhere, and **Esc** from any other
workspace returns to Create. The chain composer opens with **c** from
Create — it is a Create sub-mode, not a tab.

## Command Palette

**Ctrl+K** opens the command palette from any workspace or focus — the
TUI's version of the GUI surfaces' ⌘K launcher. Type to filter, **Up**/
**Down** to select, **Enter** to run, **Esc** to close. It covers
navigation (all five workspaces, the chain composer), actions (toggle
Advanced, randomize seed, expand prompt, prompt history, help, quit),
and switching between all eleven theme presets.

## Create View

The main workspace with four panels:

- **Prompt** — Multi-line text area (Shift+Enter for newlines, emacs
  keybindings)
- **Parameters** — six essentials rows plus the Advanced accordion
- **Preview** — idle hint, live "Developing…" progress, then the finished
  print with a `model · seed · time · host` caption
  (Kitty/sixel/halfblock rendering)
- **Timeline** — the glyph-styled session log (`•` info, `✓` done with
  stage timings, `!` warning, `✗` error, `★` model loaded), including a
  `✓ Saved <file>` entry per print; shows "— idle. no runs this session."
  until the first run. Hidden when `tui.show_timeline` is off or the
  terminal is too short.

The model's description is the dim line under the Model row; host and
memory telemetry live in the Machines workspace and the chrome host chip.

### Essentials

| Row             | Shows                    | ◀▶ / +/-             | Enter                |
| --------------- | ------------------------ | -------------------- | -------------------- |
| Model           | model name + description | —                    | fuzzy model selector |
| Size            | `1024 × 1024`            | cycle aspect presets | type an exact `WxH`  |
| Detail          | `●●●●○○○○ 28` step dots  | adjust steps         | —                    |
| Prompt strength | guidance                 | adjust               | —                    |
| Seed            | `random` / `fixed · 42`  | cycle seed mode      | type an exact seed   |
| Batch           | image count              | adjust               | —                    |

Size's `◀▶` cycles 1:1, 3:2, 2:3, 16:9, and 9:16 presets fitted to the
model's default pixel area (64-aligned). Seed modes are `random` (new seed
every run), `fixed` (reproducible), and `increment` (+1 per run) — **Ctrl+R**
still cycles them from anywhere.

### Advanced accordion

Press **A** anywhere in Create (or **Enter**/**→** on the `▸ Advanced`
row) to open the disclosure; **Enter** or **→** on a section expands it,
collapsing any other, and **←** collapses. Each collapsed section shows a
summary (`default`, `off`, `none`, `png`, or the set value) and the header
carries a count of advanced values that differ from their defaults. The
open state and expanded section persist across sessions
(`tui.advanced_open` / `tui.advanced_section`).

| Section                | Rows                                                 |
| ---------------------- | ---------------------------------------------------- |
| Scheduler & sampling   | Scheduler (CFG models), Expand prompt, Offload       |
| Negative prompt        | inline editor (CFG models; **Alt+N** jumps here)     |
| Source image           | Source, Strength, Mask, ControlNet (per model)       |
| LoRA                   | LoRA path + scale                                    |
| Upscale after generate | post-generate upscaler (Enter picks, `(off)` clears) |
| Output format          | png / jpeg / gif / apng / webp / mp4                 |
| Video                  | Frames, FPS (video models only)                      |

The `↺ Reset to model defaults` action row at the bottom restores every
parameter (keeping the model and your prompt). `qwen-image-edit` shows a
source image and negative prompt without img2img `strength` or `mask`
controls.

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

| Key    | Action            |
| ------ | ----------------- |
| Ctrl+A | Beginning of line |
| Ctrl+E | End of line       |
| Ctrl+U | Kill to start     |
| Ctrl+W | Delete word back  |
| Ctrl+D | Delete forward    |
| Ctrl+F | Forward char      |
| Ctrl+B | Backward char     |

## Library View

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
| e          | Load into Create (edit)          |
| u          | Upscale with AI model            |
| d          | Delete image (with confirmation) |
| o          | Open in system viewer            |
| Esc        | Back to Create                   |

### Detail Mode

Press **Enter** on a grid thumbnail to see the full image with all metadata.

| Key | Action                  |
| --- | ----------------------- |
| e   | Load into Create (edit) |
| r   | Regenerate immediately  |
| u   | Upscale with AI model   |
| d   | Delete image            |
| o   | Open in system viewer   |
| j/k | Previous / next image   |
| Esc | Back to grid            |

### Thumbnails

Thumbnails are cached at `~/.mold/cache/thumbnails/` and generated automatically
on first scan and after each generation. Delete the cache directory to force
regeneration.

## Chain Composer

The chain composer authors `mold.chain.v1` TOML for LTX-2 chains. Press
**c** from Create's navigation mode to open it (Esc returns to composing —
a chain in progress survives switching workspaces). It lets you build
per-stage prompts, frame counts, source images, and `smooth` / `cut` / `fade`
transitions, then submit the normalised script through the same chain endpoint
used by `mold run --script`.

## Models View

See all installed and available models with family, size, defaults, and status.

| Key   | Action                        |
| ----- | ----------------------------- |
| j/k   | Navigate the model list       |
| Enter | Set as default model          |
| p     | Pull (download) a model       |
| u     | Unload the active model (GPU) |
| Esc   | Back to Create                |

## Machines View

Manage every Mold server the TUI can generate on — the same multi-host
Machines workspace as the desktop, web, and iPhone apps. The left pane
lists machines with a status dot (green ready, yellow connecting, red
offline), the machine name, and a dim hardware line (GPU, VRAM, backend,
`host:port`); **This Mac** (or **This machine**) is always the first row.
The right pane shows the selected machine's telemetry (GPU, VRAM,
models-disk storage, loaded models, uptime, version) and its live queue
lanes — `▶` running jobs with elapsed time and GPU ordinal, `●` queued
jobs with their position. Offline hosts stay listed and recover
automatically when the server comes back.

| Key   | Action                                                   |
| ----- | -------------------------------------------------------- |
| j/k   | Select a machine (or a queue lane with detail focus)     |
| Enter | Set as the generation target (again reverts to Auto)     |
| Tab   | Switch focus between the host list and the detail lanes  |
| c     | Connect a machine (also in the ⌘K palette from anywhere) |
| d     | Forget the selected host (confirms; deletes its API key) |
| r     | Refresh telemetry and queue now                          |
| x     | Cancel the selected queued job (detail focus, confirms)  |
| Esc   | Back to Create                                           |

### Connecting a machine

Press **c** for the stepped connect flow: enter the server address (bare
hostname, `host:port`, or full URL — bare names default to port 7680),
then an optional API key (masked; **Enter** skips it), and the TUI tests
the connection. On success the host is saved with its display name taken
from the server's hostname; if the same server is already registered
(matched by its instance ID, URL, or id) you get "Already connected as
…" instead of a duplicate row. On failure the error is shown with
**Enter** to retry or **e** to edit the address.

Hosts persist in the settings DB (`mold.db`) under `tui.hosts.v1`, with
each host's API key stored in its own `tui.host_key.<id>` settings row —
keys are sent as the `x-api-key` header, never placed in URLs, and are
deleted when the host is forgotten.

### Generation target

**Enter** on a machine row makes it the sticky generation target
(persisted as `tui.generate_target`): **This Mac** forces the local
engine, a remote host routes every Generate to that server with its API
key — and never silently falls back to local. If a targeted host is
unreachable the run fails with an error naming the host so you can fix
it in Machines or press Enter on the target row again to return to
**Auto** (remote when connected, local fallback — the default).

## Settings View

Edit settings without leaving the TUI. Bootstrap values such as paths, ports,
credentials, and logging persist to `config.toml`; user preferences and
per-model generation defaults persist to the SQLite settings DB at
`$MOLD_HOME/mold.db`.

| Key        | Action                                       |
| ---------- | -------------------------------------------- |
| j/k        | Navigate settings                            |
| +/- or L/R | Adjust numeric or cycle toggle values        |
| Enter      | Edit text/path field (opens popup) or toggle |
| Esc        | Back to Create                               |

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

The TUI treats `qwen-image-edit` as a distinct edit family:

- single source image only in the TUI
- no img2img `strength`
- no inpainting mask
- no ControlNet controls
- LoRA controls are available because `qwen-image-edit` is LoRA-capable
- default width/height derived from the selected source image at roughly `1024x1024` area

Local inference uses the Qwen2.5-VL multimodal edit encoder. In v1 the TUI
keeps the flow single-image only, even though the CLI and API accept multiple
ordered `--image` inputs for `qwen-image-edit`.

## Navigation

Press **Esc** to enter navigation mode, where number keys and arrows switch
views.

### Global Shortcuts

| Key           | Action                                |
| ------------- | ------------------------------------- |
| Esc           | Unfocus / navigation mode             |
| 1 – 5         | Switch workspace (in navigation mode) |
| Left / Right  | Cycle workspaces (in navigation mode) |
| Alt+1 – Alt+5 | Switch workspace (from anywhere)      |
| Tab           | Cycle focus to next panel             |
| Shift+Tab     | Cycle focus to previous panel         |
| Ctrl+C        | Quit                                  |
| q             | Quit (when not in a text field)       |
| ?             | Show help overlay                     |

### Create Shortcuts

| Key    | Context    | Action                                   |
| ------ | ---------- | ---------------------------------------- |
| Enter  | Prompt     | Start generation                         |
| Enter  | Parameters | Activate row (selector/toggle/expand)    |
| Ctrl+G | Any        | Start generation                         |
| Ctrl+M | Any        | Open model selector                      |
| Ctrl+R | Any        | Cycle seed mode                          |
| Ctrl+E | Any        | Expand prompt via LLM                    |
| Alt+N  | Any        | Open Advanced → Negative → focus editor  |
| c      | Navigation | Open chain composer                      |
| A      | Anywhere   | Toggle the Advanced accordion            |
| +/-    | Parameters | Adjust value / expand-collapse a section |
| j/k    | Parameters | Navigate rows (flat, sections included)  |

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

Use the `↺ Reset to model defaults` row at the bottom of Parameters to restore
model defaults without losing your prompt. Unloading the active model moved to
the Models workspace (**u**).

Generated images are saved to `~/.mold/output/` by default (override with
`MOLD_OUTPUT_DIR` env var or `output_dir` in config). All images include
embedded PNG metadata that preserves the full generation parameters, making them
portable across machines.

## Routing

Where Generate runs is owned by the **Machines** workspace, not the Create
form: press **Enter** on a machine row to pin it as the sticky generation
target (persisted as `tui.generate_target`), or leave it on **Auto** — try
the connected server first, fall back to the local GPU. `mold tui --local`
pins the session to the local engine. A pinned remote host never silently
falls back to local; an unreachable target fails with an error naming the
host. See the Machines section above for connecting hosts and API keys.

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

## Motion

Switching workspaces fades the new content in and a finished print sweeps
its caption row — the TUI's only two animations, matching the design
system's restraint budget. Turn them off with the **Reduce motion**
preference in Settings (persisted as `tui.reduce_motion`) or the
`MOLD_TUI_NO_MOTION=1` environment variable (useful for scripted terminal
captures).

## Theme

The TUI ships eleven theme presets, selectable live from **Settings →
Appearance** (Left/Right cycles with immediate apply; the choice persists in
the metadata DB under `tui.theme`).

The default is **Studio Dark** — the same Mold Studio look as the desktop,
web, and iPhone apps — with its dual-accent role model: the warm _safelight_
accent marks focus, selection, and primary actions, while the cool _halide_
accent marks info and live state.

| Preset          | Slug              | Notes                                     |
| --------------- | ----------------- | ----------------------------------------- |
| Studio Dark     | `studio-dark`     | Default. Mold family, dual accent         |
| Studio Light    | `studio-light`    | Mold family, dual accent                  |
| Safelight Dark  | `safelight-dark`  | Warm darkroom family, dual accent         |
| Safelight Light | `safelight-light` | Warm darkroom family, dual accent         |
| Mocha           | `mocha`           | Catppuccin Mocha — the pre-Studio default |
| Latte           | `latte`           | Catppuccin Latte (light)                  |
| Ristretto       | `ristretto`       | Monokai Ristretto                         |
| Gruvbox         | `gruvbox`         | Gruvbox Dark (hard)                       |
| Tokyo           | `tokyo`           | Tokyo Night (storm)                       |
| Nord            | `nord`            | Nord                                      |
| Dracula         | `dracula`         | Dracula                                   |

`studio` and `safelight` are accepted as slug aliases for the dark variants.
