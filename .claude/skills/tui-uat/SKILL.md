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
# Lifecycle
scripts/tui-uat.sh launch [--fresh] [--env K=V]* [--local] [--host URL]
scripts/tui-uat.sh quit
scripts/tui-uat.sh status
scripts/tui-uat.sh env              # Print MOLD_HOME / MOLD_DB_PATH

# Screen I/O
scripts/tui-uat.sh capture          # Print current screen (plain text)
scripts/tui-uat.sh screenshot [output.png]
scripts/tui-uat.sh view <1-5|name>  # 1=Generate 2=Gallery 3=Models 4=Queue 5=Settings
scripts/tui-uat.sh send <key>...
scripts/tui-uat.sh wait-for <pattern> [timeout]
scripts/tui-uat.sh assert <pattern>

# DB / persistence helpers
scripts/tui-uat.sh db [--write] [--force] <sql>    # sqlite3 against session DB
scripts/tui-uat.sh db-get <key>                     # One value from settings table
scripts/tui-uat.sh db-assert <key> <value>          # Pass/fail equality check
scripts/tui-uat.sh db-dump                           # Pretty-print settings + model_prefs

# Settings helpers
scripts/tui-uat.sh settings-focus <appearance|configuration>
scripts/tui-uat.sh theme-set <slug>  # mocha|latte|ristretto|gruvbox|tokyo|nord|dracula

# Model + per-model prefs helpers
scripts/tui-uat.sh model <name>                      # Full picker dance — Parameters → Model → type filter → Enter. Pass the exact `model:tag` to avoid matching the wrong variant.
scripts/tui-uat.sh db-model-assert <model> <col> <v> # Pass/fail on a single `model_prefs` column
```

**`--fresh`** creates a tmp MOLD_HOME and injects it into the TUI's env — zero chance of clobbering the user's real `~/.mold/` state. The isolated directory persists across `quit` so you can relaunch with `--env MOLD_HOME=$(mktemp -d)/…` or reuse the path from `status` to validate persistence.

**`db-*` commands** refuse to write to the user's real DB without `--force`. Use `--fresh` for any test that mutates state.

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

3. **Session persistence** (since #264): TUI state lives in the SQLite metadata DB at `~/.mold/mold.db` — `settings` table for global TUI prefs (theme, last_model, last_prompt, negative_collapsed), `model_prefs` table for per-model generation parameters (one row per resolved model tag), `prompt_history` table for the prev/next prompt stack. `~/.mold/tui-session.json` and `~/.mold/prompt-history.jsonl` are imported once on first launch and renamed to `.migrated`; they're no longer written. For a clean slate, isolate the DB with `MOLD_DB_PATH=$(mktemp -d)/mold.db scripts/tui-uat.sh launch …` (legacy: deleting `~/.mold/mold.db` also works but wipes the gallery DB too). `MOLD_DB_DISABLE=1` boots the TUI with in-memory-only defaults — useful for verifying the fail-safe fallback.

4. **`MOLD_BIN`**: Override the binary path: `MOLD_BIN=./target/release/mold scripts/tui-uat.sh launch`

5. **Clipboard**: The `capture` command temporarily uses the clipboard (via `write_screen_file:copy,plain`). It saves and restores the previous clipboard content.

6. **Per-model prefs auto-save** (since #264): switching model via `Ctrl+M → pick → Enter` snapshots the outgoing model's generation params into `model_prefs` keyed on the resolved tag, then overlays the incoming model's saved row on top of manifest/catalog defaults. Prompts are *not* restored on switch — only generation params (width/height/steps/guidance/scheduler/seed_mode/batch/format/lora/expand/offload/strength/control_scale). To UAT: set FLUX to 512×512 steps=4, switch to SDXL, verify SDXL's own defaults appear; switch back to FLUX, verify 512×512 steps=4 returned.

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

## Example: SQLite-Backed Persistence UAT (#264)

Full quit/relaunch round-trip against an isolated DB. `--fresh` allocates a
tmp MOLD_HOME; stash its path from `env` so the re-launch can reuse it.

```bash
# Round 1: fresh isolated env, set theme via helper, write a prompt
scripts/tui-uat.sh launch --fresh --local
eval "$(scripts/tui-uat.sh env)"              # exports MOLD_HOME + MOLD_DB_PATH
scripts/tui-uat.sh theme-set dracula          # cycles + asserts
scripts/tui-uat.sh view generate
scripts/tui-uat.sh send "a test prompt"       # single arg is sent as literal text
scripts/tui-uat.sh send escape                # exit textarea focus
scripts/tui-uat.sh send ctrl+c                # quit (writes settings + model_prefs)

# Round 2: relaunch with the *same* MOLD_HOME → DB survives
scripts/tui-uat.sh launch --env "MOLD_HOME=$MOLD_HOME" --local
scripts/tui-uat.sh db-assert tui.theme dracula
scripts/tui-uat.sh db-assert tui.last_prompt "a test prompt"
scripts/tui-uat.sh screenshot /tmp/uat-persistence.png
scripts/tui-uat.sh quit
rm -rf "$MOLD_HOME"                            # tmp dir cleanup is manual
```

## Example: Per-Model Preferences UAT (#264) — full param coverage

The cleanest strategy is DB-seed-then-verify-UI, plus a `model` switch
that exercises `update_model`'s snapshot/restore path. Every field in
`model_prefs` gets checked: width, height, steps, guidance, scheduler,
seed_mode, batch, format, lora_path, lora_scale, expand, offload,
strength, control_scale.

```bash
ISO=$(mktemp -d /tmp/mold-uat.XXXXXX)
scripts/tui-uat.sh launch --env "MOLD_HOME=$ISO" --local
sleep 2

# Seed two models with DISTINCT values across every preserved field.
scripts/tui-uat.sh db --write "
INSERT OR REPLACE INTO model_prefs
  (model, width, height, steps, guidance, scheduler, seed_mode, batch, format,
   lora_path, lora_scale, expand, offload, strength, control_scale, updated_at_ms)
VALUES
  ('flux2-klein:q8', 832, 1152, 7, 2.5, 'ddim',            'fixed',     3, 'jpeg',
   '/a.safetensors', 0.75, 1, 1, 0.8, 1.5, 1000),
  ('flux-dev:q4',    1024, 768, 30, 4.0, 'eulerancestral', 'increment', 2, 'png',
   '/b.safetensors', 0.50, 0, 0, 0.6, 1.0, 1000);
INSERT OR REPLACE INTO settings (key, value, value_type, updated_at_ms)
VALUES ('tui.last_model', 'flux2-klein:q8', 'string', 1000);
"
scripts/tui-uat.sh quit
scripts/tui-uat.sh launch --env "MOLD_HOME=$ISO" --local
sleep 3

# Starts on flux2-klein:q8, which should load its seeded values.
scripts/tui-uat.sh assert "832"
scripts/tui-uat.sh assert "1152"
scripts/tui-uat.sh assert "Steps     7"
scripts/tui-uat.sh assert "Guidance  2.5"

# Switch to flux-dev:q4 via the real popup flow.
scripts/tui-uat.sh model 'flux-dev:q4'
sleep 0.5
scripts/tui-uat.sh assert "1024"
scripts/tui-uat.sh assert "Steps     30"
scripts/tui-uat.sh assert "Guidance  4.0"

# Switch back — flux2-klein's saved row must overlay manifest defaults.
scripts/tui-uat.sh model 'flux2-klein:q8'
sleep 0.5
scripts/tui-uat.sh assert "832"
scripts/tui-uat.sh assert "Steps     7"

# Direct DB assertions for the fields that don't always render in the UI
# (scheduler, lora, strength, control_scale are conditionally visible).
for col in width height steps guidance scheduler seed_mode batch format \
           lora_scale expand offload strength control_scale; do
    scripts/tui-uat.sh db-model-assert flux2-klein:q8 "$col" \
      "$(sqlite3 "$ISO/mold.db" "SELECT $col FROM model_prefs WHERE model='flux2-klein:q8';")"
    scripts/tui-uat.sh db-model-assert flux-dev:q4 "$col" \
      "$(sqlite3 "$ISO/mold.db" "SELECT $col FROM model_prefs WHERE model='flux-dev:q4';")"
done

scripts/tui-uat.sh quit
rm -rf "$ISO"
```

### Why DB seeding + UI verification (instead of UI-driven everything)

- **Step-size fragility**: each param has its own increment granularity
  (width/height ± 64, steps ± 1, guidance ± 0.1, etc.). Hardcoding
  press counts in a script is brittle across model families.
- **Conditional fields**: scheduler, strength, control_scale, and the
  LoRA params only render under specific capabilities/modes. Relying
  on UI navigation to reach them tangles the test with feature gating
  — the DB is the single source of truth.
- **Execution speed**: a seed + relaunch round-trip is ~3 s; the
  equivalent key-sequence UI drive is ~15 s and 10× more flaky.

## Example: DB-Disabled Fallback

```bash
scripts/tui-uat.sh launch --fresh --env MOLD_DB_DISABLE=1 --local
scripts/tui-uat.sh view settings
scripts/tui-uat.sh assert "Appearance"         # still renders
scripts/tui-uat.sh quit                         # no crash on shutdown save
```

## Example: One-Shot Assertion Script

Run all three fast checks in sequence — suitable for CI.

```bash
set -e
scripts/tui-uat.sh launch --fresh --local
scripts/tui-uat.sh view settings
scripts/tui-uat.sh theme-set gruvbox
scripts/tui-uat.sh db-assert tui.theme gruvbox
scripts/tui-uat.sh view generate
scripts/tui-uat.sh assert "flux2-klein"
scripts/tui-uat.sh view gallery && scripts/tui-uat.sh assert "Gallery"
scripts/tui-uat.sh view queue && scripts/tui-uat.sh assert "Queue"
scripts/tui-uat.sh view models && scripts/tui-uat.sh assert "FAMILY"
scripts/tui-uat.sh quit
```
