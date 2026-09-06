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
cargo build -p mold-ai --features tui
```

Or use the devshell helper: `build`

## Helper Script

All TUI interaction goes through `scripts/tui-uat.sh`:

```bash
# Lifecycle
scripts/tui-uat.sh launch [--fresh] [--env K=V]* [--local] [--host URL]
scripts/tui-uat.sh quit             # Close the tracked session's Ghostty window
scripts/tui-uat.sh cleanup          # Reap any orphan mold-TUI Ghostty window (state-file-free)
scripts/tui-uat.sh status
scripts/tui-uat.sh env              # Print MOLD_HOME / MOLD_DB_PATH

# Screen I/O
scripts/tui-uat.sh capture          # Print current screen (plain text)
scripts/tui-uat.sh screenshot [output.png]
scripts/tui-uat.sh view <1-5|name>  # 1=Create 2=Library 3=Models 4=Machines 5=Settings
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
scripts/tui-uat.sh theme-set <slug>  # studio-dark|studio-light|safelight-dark|safelight-light|mocha|latte|ristretto|gruvbox|tokyo|nord|dracula

# Model + per-model prefs helpers
scripts/tui-uat.sh model <name>                      # Full picker dance — Parameters → Model → type filter → Enter. Pass the exact `model:tag` to avoid matching the wrong variant.
scripts/tui-uat.sh db-model-assert <model> <col> <v> # Pass/fail on a single `model_prefs` column
```

**`--fresh`** creates a tmp MOLD_HOME and injects it into the TUI's env — zero chance of clobbering the user's real `~/.mold/` state. The isolated directory persists across `quit` so you can relaunch with `--env MOLD_HOME=$(mktemp -d)/…` or reuse the path from `status` to validate persistence.

**Motion**: launch with `--env MOLD_TUI_NO_MOTION=1` when a scenario asserts on exact screen content immediately after a workspace switch — tachyonfx transitions recolor cells for ~160/320 ms and can race a capture.

**`db-*` commands** refuse to write to the user's real DB without `--force`. Use `--fresh` for any test that mutates state.

## How to Run a UAT Session

> **Never close Ghostty.** Ghostty is the user's terminal (cmux is built on
> it), so a UAT flow must never quit the app, close its windows, or kill its
> processes — not on success, not on failure, not from a trap. `quit` and
> `cleanup` exist for a human at the keyboard; an agent does not call them.
> When a flow ends, exit `mold tui` normally (its quit key, so the shell
> prompt returns) and leave the window open. Killing a `mold serve` you
> started yourself is fine.

### 1. Launch

```bash
scripts/tui-uat.sh launch --local
```

This opens a new Ghostty window running `mold tui`. The `--local` flag runs inference locally without a server. Use `--host http://server:7680` to test against a remote server.

### 2. Navigate Views

```bash
scripts/tui-uat.sh view create     # or: view 1 (legacy alias: generate)
scripts/tui-uat.sh view library    # or: view 2 (legacy alias: gallery)
scripts/tui-uat.sh view models     # or: view 3
scripts/tui-uat.sh view machines   # or: view 4 (legacy alias: queue)
scripts/tui-uat.sh view settings   # or: view 5
```

The `view` command handles the Create prompt-focus quirk automatically — it detects prompt focus and uses Tab + Escape to reach Nav mode before sending the view key.

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

Exit the TUI from inside it (`q` from Nav mode, or `ctrl+c`) so the shell
prompt returns in the same window, then stop. Do not run `quit` or `cleanup`:
they close Ghostty windows, and the window stays open by rule (see the note
at the top of this section).

## TUI Views and Landmarks

| View | Key | Unique landmark | Content |
|------|-----|----------------|---------|
| Create | 1 | `┌ Parameters` or `┌ Prompt` | Prompt; Parameters (6 essentials + `▸ Advanced` accordion + `↺ Reset to model defaults`); Preview; Timeline |
| Library | 2 | `┌ Library` | All-machines print grid (local + connected server + every Machines host, deduped by filename) + Details side panel on wide terminals; header hint `{n} prints [· host \| · all machines] [· k hosts offline]` |
| Models | 3 | `┌ Installed` or `┌ Available` | Model list with name, family, size, status |
| Machines | 4 | `┌ Machines` | Host rows (local first) + telemetry/queue detail pane; connect flow on `c` |
| Settings | 5 | `┌ Appearance` or `┌ Configuration` | Theme cards + Preferences + config values |

## Key Bindings Reference

**Global:** Ctrl+C = quit, Alt+1-5 = switch workspace

**Create (prompt focused):** Enter = generate, Tab = next focus, Escape = nav mode, Ctrl+G = generate, Ctrl+M = model selector, Ctrl+R = randomize seed, Ctrl+E = expand prompt, Ctrl+Shift+E = remix prompt, Ctrl+S = save image, Ctrl+T = retry held prints, Ctrl+P / Ctrl+N = previous / next prompt in history, Alt+N = open Advanced → Negative → focus inline editor (every Ctrl binding works from any Create focus)

**Create (nav mode):** 1-5 = switch workspace, A = toggle Advanced, q = quit, Enter = focus prompt

**Create (Parameters focus):** j/k = flat row traversal (essentials → `▸ Advanced` header → section rows → `↺ Reset`), +/- or ←/→ = adjust a field / expand-collapse a section, Enter = activate (Model picker, Size `WxH` popup, Seed value popup, section expand, Negative inline editor focus), A = toggle the accordion. Essentials order: Model (first row — `model <name>` relies on this), Size, Detail, Prompt strength, Seed, Batch; a video model inserts `Predict duration` (only when the checkpoint advertises `supports_duration_prediction`) and `Duration` between Prompt strength and Seed, so it has 7–8 essentials. Accordion landmarks: `▸ Advanced` collapsed / `▾ Advanced` open with section rows (`Scheduler & sampling`, `Negative prompt`, `Source image`, `Identity photo` on identity-capable checkpoints, `LoRA`, `Upscale after generate`, `Output format`, `Video` on video models, `File under`). Persisted keys for `db-get`: `tui.advanced_open` (`true`/`false`), `tui.advanced_section` (section slug or empty).

**Library (grid):** hjkl/arrows = navigate, Enter = detail, e/r = recall into Create, d = delete (multi-host prints delete on every owning host; confirm names the count), u = upscale (routes to the owning host), o = open, / = filter by prompt/model/filename (typed chars edit, Enter applies, Esc clears; Esc with a filter applied clears it before leaving the view)

**Models:** j/k = navigate, Enter = select, p = pull, r = remove, u = unload, / = filter

**Machines:** j/k = select row, Enter = set generation target (again = back to Auto), Tab = host list ↔ detail lanes, c = connect a machine (stepped URL → API key → test popup), d = disconnect/reconnect the selected remembered host, f = forget host (confirm; deletes its saved API key), r = refresh, x = cancel the selected queued or running job when the host supports it (detail focus, confirm), l = load the next durable queue page, g = next visible GPU, `[` / `]` = device selection back/forward, e = enable or drain/disable the selected GPU. Persisted keys for `db-get`: `tui.hosts.v1` (JSON registry), `tui.generate_target` (`auto`|`local`|`host:<id>`), `tui.host_key.<id>`.

**Settings:** j/k = navigate, +/- = adjust values, Tab = flip Appearance ↔ Configuration focus. On the Appearance theme-card grid, Up/Down move by card rows (Down past the bottom row enters Configuration) and Left/Right/+/- cycle presets linearly with live apply — `theme-set` relies on the linear `+` cycle and the `theme · <slug>` header hint. The Configuration list starts with a DB-backed Preferences section (`tui.default_format`, `tui.reduce_motion`, `tui.show_timeline`, `tui.confirm_destructive`); with `tui.confirm_destructive` off, destructive actions skip the Confirm popup.

## Known Quirks

1. **Escape from prompt focus**: The `view` command works around Create's prompt focus by detecting "Esc Nav" in the footer and sending Tab + Escape to reach Nav mode.

2. **First key after Nav mode**: The first character key after entering Create nav mode may be consumed by a crossterm timing issue. The `view` command retries automatically.

3. **Session persistence** (since #264): TUI state lives in the SQLite metadata DB at `~/.mold/mold.db` — `settings` table for global TUI prefs (theme, last_model, last_prompt, advanced_open/advanced_section), `model_prefs` table for per-model generation parameters (one row per resolved model tag), `prompt_history` table for the prev/next prompt stack. `~/.mold/tui-session.json` and `~/.mold/prompt-history.jsonl` are imported once on first launch and renamed to `.migrated`; they're no longer written. For a clean slate, isolate the DB with `MOLD_DB_PATH=$(mktemp -d)/mold.db scripts/tui-uat.sh launch …` (legacy: deleting `~/.mold/mold.db` also works but wipes the gallery DB too). `MOLD_DB_DISABLE=1` boots the TUI with in-memory-only defaults — useful for verifying the fail-safe fallback.

4. **`MOLD_BIN`**: Override the binary path: `MOLD_BIN=./target/release/mold scripts/tui-uat.sh launch`

5. **Clipboard**: The `capture` command temporarily uses the clipboard (via `write_screen_file:copy,plain`). It saves and restores the previous clipboard content.

6. **Per-model prefs auto-save** (since #264): switching model via `Ctrl+M → pick → Enter` snapshots the outgoing model's generation params into `model_prefs` keyed on the resolved tag, then overlays the incoming model's saved row on top of manifest/catalog defaults. Prompts are *not* restored on switch — only generation params (width/height/steps/guidance/scheduler/seed_mode/batch/format/lora/expand/offload/strength/control_scale). To UAT: set FLUX to 512×512 steps=4, switch to SDXL, verify SDXL's own defaults appear; switch back to FLUX, verify 512×512 steps=4 returned.

## Example: Full Smoke Test

```bash
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true' EXIT INT TERM
scripts/tui-uat.sh launch --local
scripts/tui-uat.sh view create
scripts/tui-uat.sh assert "Parameters"
scripts/tui-uat.sh assert "Model"
scripts/tui-uat.sh assert "Preview"
scripts/tui-uat.sh screenshot /tmp/generate-view.png
scripts/tui-uat.sh view library
scripts/tui-uat.sh assert "Library"
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
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true; [ -n "${MOLD_HOME:-}" ] && rm -rf "$MOLD_HOME"' EXIT INT TERM
# Round 1: fresh isolated env, set theme via helper, write a prompt
scripts/tui-uat.sh launch --fresh --local
eval "$(scripts/tui-uat.sh env)"              # exports MOLD_HOME + MOLD_DB_PATH
scripts/tui-uat.sh theme-set dracula          # cycles + asserts
scripts/tui-uat.sh view create
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

## Example: Preferences Toggles

The Preferences rows sit at the top of the Configuration list (row order:
Format, Reduce Motion, Show Timeline, Confirmations) and persist the moment
they flip — no quit needed. Bools store as `1`/`0`.

```bash
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true' EXIT INT TERM
scripts/tui-uat.sh launch --fresh --local
scripts/tui-uat.sh view settings
scripts/tui-uat.sh settings-focus configuration   # lands on the Format row
scripts/tui-uat.sh send j                          # down to Reduce Motion
scripts/tui-uat.sh send +                          # toggle on
scripts/tui-uat.sh db-assert tui.reduce_motion 1
scripts/tui-uat.sh send +                          # toggle back off
scripts/tui-uat.sh db-assert tui.reduce_motion 0
scripts/tui-uat.sh quit
```

## Example: Per-Model Preferences UAT (#264) — full param coverage

The cleanest strategy is DB-seed-then-verify-UI, plus a `model` switch
that exercises `update_model`'s snapshot/restore path. Every generation-parameter field in
`model_prefs` gets checked: width, height, steps, guidance, scheduler,
seed_mode, batch, format, lora_path, lora_scale, expand, offload,
strength, control_scale, frames, fps. (The table also carries the
`profile` primary-key component and the `last_prompt` / `last_negative`
columns; prompts stay out of this check by design.)

```bash
ISO=$(mktemp -d /tmp/mold-uat.XXXXXX)
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true; rm -rf "$ISO"' EXIT INT TERM
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
           lora_scale expand offload strength control_scale frames fps; do
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
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true' EXIT INT TERM
scripts/tui-uat.sh launch --fresh --env MOLD_DB_DISABLE=1 --local
scripts/tui-uat.sh view settings
scripts/tui-uat.sh assert "Appearance"         # still renders
scripts/tui-uat.sh quit                         # no crash on shutdown save
```

## Example: One-Shot Assertion Script

Run all three fast checks in sequence — suitable for CI.

```bash
set -e
trap 'scripts/tui-uat.sh cleanup >/dev/null 2>&1 || true' EXIT INT TERM
scripts/tui-uat.sh launch --fresh --local
scripts/tui-uat.sh view settings
scripts/tui-uat.sh theme-set gruvbox
scripts/tui-uat.sh db-assert tui.theme gruvbox
scripts/tui-uat.sh view create
scripts/tui-uat.sh assert "flux2-klein"
scripts/tui-uat.sh view library && scripts/tui-uat.sh assert "Library"
scripts/tui-uat.sh view machines && scripts/tui-uat.sh assert "Machines"
scripts/tui-uat.sh view models && scripts/tui-uat.sh assert "FAMILY"
scripts/tui-uat.sh quit
```
