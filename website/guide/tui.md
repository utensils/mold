# Terminal UI

mold includes an interactive terminal UI for browsing models, tuning parameters,
generating images with live progress, and previewing results — all without
leaving the terminal.

Built on [ratatui](https://ratatui.rs) with Kitty graphics protocol support for
pixel-perfect image preview in terminals like Ghostty, kitty, and WezTerm. Falls
back to halfblock rendering in other terminals.

## Quick Start

```bash
mold tui
```

The TUI launches in the **Generate** view with your cursor in the prompt field.
Type a prompt, press **Enter**, and watch the progress panel as your image
generates. The result appears in the Preview panel.

::: tip The `tui` feature must be compiled in. Pre-built releases and the Nix
package include it by default. If building from source, add `--features tui` to
your build command. :::

## Views

The TUI has three main views, shown as tabs at the top of the screen:

| View     | Purpose                                         |
| -------- | ----------------------------------------------- |
| Generate | Write prompts, tune parameters, generate images |
| Gallery  | Browse previously generated images              |
| Models   | View installed and available models             |

Switch between views by pressing **Esc** to enter navigation mode, then **1**,
**2**, or **3**. Or use **Alt+1**/**Alt+2**/**Alt+3** from anywhere.

## Generate View

The main workspace, split into four panels:

- **Prompt** — Multi-line text area for your image prompt
- **Negative** — (CFG models only) Describes what to avoid
- **Parameters** — Model, dimensions, steps, guidance, and more
- **Preview** — Shows the generated image with Kitty/sixel/halfblock rendering

### Workflow

1. Type your prompt in the Prompt panel
2. Press **Tab** to move to Parameters and adjust settings
3. Press **Enter** on the Model field to open the model selector
4. Press **Enter** in the Prompt panel (or **Ctrl+G** from anywhere) to generate
5. Watch the Progress panel for loading stages and denoising progress
6. The result appears in Preview when done

### Editing Parameters

Navigate to the Parameters panel with **Tab**, then:

- **j**/**k** or arrow keys to move between fields
- **+**/**-** or arrow left/right to adjust numeric values
- **Enter** to activate a field:
  - **Model** — opens the fuzzy model selector
  - **Format** / **Mode** / **Expand** / **Offload** — toggles the value
  - **Scheduler** — cycles through Ddim, Euler Ancestral, UniPC
  - **Seed** — randomizes it

Width and height change in steps of 64. Steps increment by 1, guidance by 0.5.

### Model Selector

Press **Enter** on the Model field or **Ctrl+M** from anywhere to open the model
selector popup:

- Type to fuzzy-filter the model list
- **j**/**k** or arrow keys to navigate
- **Enter** to select — parameters update to the model's defaults
- **Esc** to cancel

### Auto-Pull

If you generate with a model that isn't downloaded, the TUI automatically pulls
it — just like the CLI. Download progress appears in the Progress panel with
per-file progress bars. Generation continues automatically once the pull
completes.

## Gallery View

Browse images from the current session. Select an entry to see it in the preview
panel with generation metadata.

| Key   | Action                               |
| ----- | ------------------------------------ |
| j/k   | Navigate the history list            |
| Enter | Re-generate with the same parameters |
| e     | Load parameters into Generate view   |
| d     | Delete the image file                |
| o     | Open in system image viewer          |
| Esc   | Back to Generate                     |

## Models View

See all installed and available models with their families, sizes, default
steps, guidance, and dimensions.

| Key   | Action                        |
| ----- | ----------------------------- |
| j/k   | Navigate the model list       |
| Enter | Set as default model          |
| p     | Pull (download) a model       |
| r     | Remove a downloaded model     |
| u     | Unload the active model (GPU) |
| Esc   | Back to Generate              |

## Navigation

The TUI uses a focus-based navigation model. Press **Esc** to unfocus and enter
navigation mode, where number keys switch views.

### Global Shortcuts

| Key       | Action                            |
| --------- | --------------------------------- |
| Esc       | Unfocus / navigation mode         |
| 1 / 2 / 3 | Switch views (in navigation mode) |
| Alt+1/2/3 | Switch views (from anywhere)      |
| Tab       | Cycle focus to next panel         |
| Shift+Tab | Cycle focus to previous panel     |
| Ctrl+C    | Quit                              |
| q         | Quit (when not in a text field)   |
| ?         | Show help overlay                 |

### Generate Shortcuts

| Key    | Context    | Action                           |
| ------ | ---------- | -------------------------------- |
| Enter  | Prompt     | Start generation                 |
| Enter  | Parameters | Activate field (selector/toggle) |
| Ctrl+G | Any        | Start generation                 |
| Ctrl+M | Any        | Open model selector              |
| Ctrl+R | Any        | Randomize seed                   |
| Ctrl+E | Any        | Expand prompt via LLM            |
| Ctrl+S | Any        | Save current image               |
| +/-    | Parameters | Adjust numeric value             |
| j/k    | Parameters | Navigate fields                  |

## Server Fallback

The TUI mirrors the CLI's connection behavior:

1. **Remote first** — connects to the server at `MOLD_HOST` (default
   `localhost:7680`)
2. **Auto fallback** — if the server is unreachable, transparently falls back to
   local GPU inference
3. **Local mode** — force local-only with `mold tui --local`

Override the server URL:

```bash
mold tui --host http://gpu-server:7680
```

## Image Preview

The TUI auto-detects your terminal's graphics protocol at startup:

| Protocol   | Terminals               | Quality        |
| ---------- | ----------------------- | -------------- |
| Kitty      | Ghostty, kitty, WezTerm | Pixel-perfect  |
| Sixel      | foot, xterm, mlterm     | Full color     |
| iTerm2     | iTerm2, Hyper           | Full color     |
| Halfblocks | Everything else         | Unicode blocks |

No configuration needed — the best available protocol is used automatically.

## Progress Display

The Progress panel shows real-time feedback during generation:

- **Stage completion** — checkmarks with elapsed time (Loading T5, Encoding
  prompt, VAE decode, etc.)
- **Denoising** — progress gauge with step count and iterations/sec
- **Weight loading** — progress gauge with bytes loaded/total
- **Downloads** — yellow progress gauge for model auto-pull
- **Info messages** — server fallback notices, cache hits

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

The TUI uses a Catppuccin Mocha color palette by default. Theme customization
via `config.toml` is planned for a future release.
