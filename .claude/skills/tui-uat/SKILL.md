---
name: tui-uat
description: Run acceptance tests on the mold TUI. Use when asked to test, verify, or UAT the TUI, or after making TUI changes that need visual verification.
argument-hint: [test-scope]
allowed-tools: Bash Read Grep Glob
---

# TUI Acceptance Testing via tmux

Run the mold TUI inside a tmux session for automated acceptance testing.
The TUI renders in a virtual terminal — you can send keystrokes and read the screen.

## Prerequisites

The devshell must have `tmux` (included in flake.nix). A debug build of mold with TUI support must exist:

```bash
cargo build -p mold-cli --features tui
```

Or use the devshell helper: `build`

## Helper Script

All TUI interaction goes through `scripts/tui-uat.sh`:

```bash
scripts/tui-uat.sh launch [--local] [--host URL]  # Start TUI in tmux
scripts/tui-uat.sh capture                          # Print current screen (plain text)
scripts/tui-uat.sh view <1-4|name>                  # Navigate to a view reliably
scripts/tui-uat.sh send <key> [key...]              # Send keystrokes
scripts/tui-uat.sh wait-for <pattern> [timeout]     # Wait for text (default 5s)
scripts/tui-uat.sh assert <pattern>                  # Assert text is on screen
scripts/tui-uat.sh quit                              # Tear down session
scripts/tui-uat.sh status                            # Check if running
```

## How to Run a UAT Session

### 1. Launch

```bash
scripts/tui-uat.sh launch --local
```

This starts the TUI in a detached tmux session (140x45). The `--local` flag runs inference locally without a server. Use `--host http://server:7680` to test against a remote server.

### 2. Navigate Views

```bash
scripts/tui-uat.sh view generate   # or: view 1
scripts/tui-uat.sh view gallery    # or: view 2
scripts/tui-uat.sh view models     # or: view 3
scripts/tui-uat.sh view settings   # or: view 4
```

The `view` command handles the Generate prompt-focus quirk automatically — it uses Tab + Escape to reach Nav mode before sending the view key.

### 3. Send Keystrokes

```bash
scripts/tui-uat.sh send Tab          # Tab key
scripts/tui-uat.sh send Enter        # Enter key
scripts/tui-uat.sh send Escape       # Escape (named key, works from non-prompt states)
scripts/tui-uat.sh send C-c          # Ctrl+C
scripts/tui-uat.sh send C-g          # Ctrl+G (generate)
scripts/tui-uat.sh send C-m          # Ctrl+M (model selector)
scripts/tui-uat.sh send C-r          # Ctrl+R (randomize seed)
scripts/tui-uat.sh send j            # Literal 'j' key
scripts/tui-uat.sh send j k Enter    # Multiple keys in sequence
```

### 4. Read and Assert Screen Content

```bash
scripts/tui-uat.sh capture                 # Full screen dump (plain text)
scripts/tui-uat.sh assert "Parameters"     # Check text exists on screen
scripts/tui-uat.sh wait-for "Loaded" 10    # Wait up to 10s for text
```

`capture` returns the screen as plain text — ratatui's box-drawing characters, labels, and values are all readable. Use `grep` patterns for assertions.

### 5. Tear Down

```bash
scripts/tui-uat.sh quit
```

## TUI Views and Landmarks

| View | Key | Unique landmark | Content |
|------|-----|----------------|---------|
| Generate | 1 | `Parameters` | Prompt, Parameters, Preview, Info, Progress |
| Gallery | 2 | `┌ Gallery` | Image thumbnails in grid, detail view on Enter |
| Models | 3 | `┌ Installed` | Model list with name, family, size, status |
| Settings | 4 | `┌ Settings` | Config values: model, dirs, server, expand |

## Key Bindings Reference

**Global:** Ctrl+C = quit, Alt+1-4 = switch view

**Generate (prompt focused):** Enter = generate, Tab = next focus, Escape = nav mode, Ctrl+G = generate, Ctrl+M = model selector, Ctrl+R = randomize seed

**Generate (nav mode):** 1-4 = switch view, q = quit, Enter = focus prompt

**Gallery (grid):** hjkl/arrows = navigate, Enter = detail, e = edit, d = delete, u = upscale, o = open

**Models:** j/k = navigate, Enter = select, p = pull, r = remove, u = unload, / = filter

**Settings:** j/k = navigate, +/- = adjust values

## Known Quirks

1. **Escape from prompt focus**: tmux's escape-time handling can swallow `C-[` keystrokes. The `view` command works around this by using Tab→Escape (named key) via Parameters focus.

2. **First key after Nav mode**: The first character key after entering Generate nav mode is consumed by a crossterm/tmux timing issue. The `view` command retries automatically.

3. **Session persistence**: The TUI saves session state to `~/.mold/tui-session.json`. Remove this file before testing if you need a clean slate: `rm -f ~/.mold/tui-session.json`

4. **`MOLD_BIN`**: Override the binary path with `MOLD_BIN=./target/release/mold scripts/tui-uat.sh launch` for release builds.

## Example: Full Smoke Test

```bash
scripts/tui-uat.sh launch --local
scripts/tui-uat.sh view generate
scripts/tui-uat.sh assert "Parameters"
scripts/tui-uat.sh assert "Model"
scripts/tui-uat.sh assert "Preview"
scripts/tui-uat.sh view gallery
scripts/tui-uat.sh assert "┌ Gallery"
scripts/tui-uat.sh view models
scripts/tui-uat.sh assert "flux2-klein"
scripts/tui-uat.sh view settings
scripts/tui-uat.sh assert "┌ Settings"
scripts/tui-uat.sh quit
```
