---
name: tui-uat
description: Run acceptance tests on the mold TUI. Use when asked to test, verify, or UAT the TUI, or after making TUI changes that need visual verification.
argument-hint: "test-scope"
---

# TUI Acceptance Testing via Ghostty

Run the mold TUI inside a native Ghostty terminal window for automated acceptance testing.
Uses Ghostty 1.3+ AppleScript API for real terminal rendering — proper fonts, true colors,
and pixel-perfect screenshots with no conversion artifacts.

## Prerequisites

- **Ghostty 1.3+** with `macos-applescript = true` (default)
- A debug build of mold with TUI support:

```bash
cargo build -p mold-cli --features tui
```

Or use the devshell helper: `build`

## Helper Script

All TUI interaction goes through `scripts/tui-uat.sh`:

```bash
scripts/tui-uat.sh launch [--local] [--host URL]  # Start TUI in a Ghostty window
scripts/tui-uat.sh capture                          # Print current screen (plain text)
scripts/tui-uat.sh screenshot [output.png]          # Native Ghostty screenshot (PNG)
scripts/tui-uat.sh view <1-4|name>                  # Navigate to a view reliably
scripts/tui-uat.sh send <key> [key...]              # Send keystrokes
scripts/tui-uat.sh wait-for <pattern> [timeout]     # Wait for text (default 5s)
scripts/tui-uat.sh assert <pattern>                  # Assert text is on screen
scripts/tui-uat.sh quit                              # Close UAT window
scripts/tui-uat.sh status                            # Check if running
```

## How to Run a UAT Session

### 1. Launch

```bash
scripts/tui-uat.sh launch --local
```

This opens a new Ghostty window running `mold tui`. The `--local` flag runs inference locally without a server. Use `--host http://server:7680` to test against a remote server.

### 2. Navigate Views

```bash
scripts/tui-uat.sh view generate   # or: view 1
scripts/tui-uat.sh view gallery    # or: view 2
scripts/tui-uat.sh view models     # or: view 3
scripts/tui-uat.sh view settings   # or: view 4
```

The `view` command handles the Generate prompt-focus quirk automatically — it detects prompt focus and uses Tab + Escape to reach Nav mode before sending the view key.

### 3. Send Keystrokes

```bash
scripts/tui-uat.sh send tab          # Tab key
scripts/tui-uat.sh send enter        # Enter key
scripts/tui-uat.sh send escape       # Escape key
scripts/tui-uat.sh send ctrl+c       # Ctrl+C
scripts/tui-uat.sh send ctrl+g       # Ctrl+G (generate)
scripts/tui-uat.sh send ctrl+m       # Ctrl+M (model selector)
scripts/tui-uat.sh send ctrl+r       # Ctrl+R (randomize seed)
scripts/tui-uat.sh send j            # Literal 'j' key (sent as text input)
scripts/tui-uat.sh send j k enter    # Multiple keys in sequence
```

**Key name reference:**
- Special keys: `enter`, `escape`, `tab`, `space`, `up`, `down`, `left`, `right`, `backspace`, `delete`, `home`, `end`, `page_up`, `page_down`, `f1`-`f12`
- Modifiers: `ctrl+<key>`, `alt+<key>`, `shift+<key>`, `cmd+<key>`
- Legacy tmux notation: `C-c`, `C-g`, etc. (still supported)
- Anything else is sent as literal text via `input text`

### 4. Read and Assert Screen Content

```bash
scripts/tui-uat.sh capture                 # Full screen dump (plain text)
scripts/tui-uat.sh assert "Parameters"     # Check text exists on screen
scripts/tui-uat.sh wait-for "Loaded" 10    # Wait up to 10s for text
```

`capture` uses Ghostty's `write_screen_file` action to get the terminal's rendered text content.

### 5. Take Screenshots

```bash
scripts/tui-uat.sh screenshot                      # Default: tui-screenshot.png
scripts/tui-uat.sh screenshot /tmp/gallery-view.png  # Custom output path
```

Screenshots are taken with `screencapture -l<windowID>` — capturing the actual Ghostty window with native font rendering, true terminal colors, and proper image display. No ANSI-to-HTML conversion, no scanline artifacts.

### 6. Tear Down

```bash
scripts/tui-uat.sh quit
```

## TUI Views and Landmarks

| View | Key | Unique landmark | Content |
|------|-----|----------------|---------|
| Generate | 1 | `┌ Parameters` or `┌ Prompt` | Prompt, Parameters, Preview, Info, Progress |
| Gallery | 2 | `┌ Gallery` | Image thumbnails in grid, detail view on Enter |
| Models | 3 | `┌ Installed` or `┌ Available` | Model list with name, family, size, status |
| Settings | 4 | `┌ Settings` | Config values: model, dirs, server, expand |

## Key Bindings Reference

**Global:** Ctrl+C = quit, Alt+1-4 = switch view

**Generate (prompt focused):** Enter = generate, Tab = next focus, Escape = nav mode, Ctrl+G = generate, Ctrl+M = model selector, Ctrl+R = randomize seed

**Generate (nav mode):** 1-4 = switch view, q = quit, Enter = focus prompt

**Gallery (grid):** hjkl/arrows = navigate, Enter = detail, e = edit, d = delete, u = upscale, o = open

**Models:** j/k = navigate, Enter = select, p = pull, r = remove, u = unload, / = filter

**Settings:** j/k = navigate, +/- = adjust values

## Known Quirks

1. **Escape from prompt focus**: The `view` command works around Generate's prompt focus by detecting "Esc Nav" in the footer and sending Tab + Escape to reach Nav mode.

2. **First key after Nav mode**: The first character key after entering Generate nav mode may be consumed by a crossterm timing issue. The `view` command retries automatically.

3. **Session persistence**: The TUI saves session state to `~/.mold/tui-session.json`. Remove this file before testing if you need a clean slate: `rm -f ~/.mold/tui-session.json`

4. **`MOLD_BIN`**: Override the binary path: `MOLD_BIN=./target/release/mold scripts/tui-uat.sh launch`

5. **Clipboard**: The `capture` command temporarily uses the clipboard (via `write_screen_file:copy,plain`). It saves and restores the previous clipboard content.

## Example: Full Smoke Test

```bash
scripts/tui-uat.sh launch --local
scripts/tui-uat.sh view generate
scripts/tui-uat.sh assert "Parameters"
scripts/tui-uat.sh assert "Model"
scripts/tui-uat.sh assert "Preview"
scripts/tui-uat.sh screenshot /tmp/generate-view.png
scripts/tui-uat.sh view gallery
scripts/tui-uat.sh assert "Gallery"
scripts/tui-uat.sh view models
scripts/tui-uat.sh assert "flux2-klein"
scripts/tui-uat.sh screenshot /tmp/models-view.png
scripts/tui-uat.sh view settings
scripts/tui-uat.sh assert "Settings"
scripts/tui-uat.sh quit
```
