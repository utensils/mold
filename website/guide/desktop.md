# Desktop App <Badge type="warning" text="experimental" />

mold ships an experimental native macOS desktop app — a Tauri 2 shell around a
Vue 3 + TypeScript frontend with its own **Safelight** design language, a warm,
matte "digital darkroom" that treats every generation as a print being
developed.

::: warning Experimental
The desktop app lives in `desktop/` on the `experiment/desktop` branch and is
under active development. It is macOS-first (Apple Silicon, Metal) and not yet
part of a tagged release. Build it from source with the devshell commands below.
:::

## What it is

A single native window that puts the full mold workflow behind a keyboard-driven
UI, instead of the CLI or the browser SPA. The same `mold-ai-server` HTTP + SSE
surface powers it, so anything the app does maps to a documented endpoint.

## Features

- **Generation workspace** — a capability-driven inspector that shows only the
  controls a model's family supports (negative prompt, scheduler, CFG++, LoRA
  stack, img2img source/mask/control, video frames/fps/audio). Generation is
  visualized as a print _developing_: a deterministic grain field, seeded from
  the job's real seed, resolves in lockstep with `DenoiseStep` events. Batches
  run sequentially with `base_seed + i`, and a VRAM preflight forecasts fit
  before you press Generate.
- **Gallery** — a justified, virtualized contact-sheet grid. **Space** opens
  Quick Look, ←/→ navigate, and **Reuse settings** jumps back to Generate with
  every parameter restored. ⌘0 / ⌘+ / ⌘− adjust thumbnail size.
- **Models & catalog** — installed models grouped by family with residency and
  disk usage, plus a live HuggingFace/Civitai catalog. Pulls render **SIZE vs
  FETCH** honestly (model weights vs. the full download including shared
  components) and stream through a downloads tray.
- **Chains** — a filmstrip editing bench for multi-stage video
  (`mold.chain.v1`): per-stage prompts and frame counts (validated `8n+1`),
  splice transitions (smooth / cut / fade) you click to cycle, a live
  fits/duration forecast against `/api/capabilities/chain-limits`, TOML
  import/export, and a durable jobs list with resume, cancel, and retake.
- **History** — a fast, searchable list of past prompts; ↩ refills the composer.
- **Settings** — every configuration row carries a provenance tag (⌂ db /
  ⛁ file / ⚿ env / default) over `/api/config`; environment-overridden rows are
  locked with the variable that owns them. Includes profile switching.
- **Command palette** — **⌘K** for navigation, actions, model search, and
  prompt-history search in one field.
- **Native macOS** — menu bar, keyboard shortcuts, and background notifications
  on generation, chain, and pull completion.

### Keyboard map

| Shortcut     | Action                                  |
| ------------ | --------------------------------------- |
| ⌘1–⌘5 / ⌘,   | Screens / Settings                      |
| ⌘K           | Command palette                         |
| ⌘N           | New generation (clear composer, focus)  |
| ⌘↩           | Generate                                |
| ⌘E           | Expand prompt                           |
| ⌘R           | Randomize seed                          |
| ⌘.           | Cancel the running job                  |
| ⌘\           | Toggle sidebar                          |
| Space        | Quick Look in Gallery                   |
| ←/→, ⌫       | Gallery navigate / delete               |
| ⇧⌘C          | Copy seed (lightbox)                    |
| ⌘0 / ⌘+ / ⌘− | Gallery thumbnail zoom reset / in / out |

## How it connects

The app talks to a `mold-ai-server` over localhost HTTP + SSE using the same
wire types as the CLI and web UI:

- **Built-in engine** — embeds the server in-process and runs on Metal, so no
  separate `mold serve` is required.
- **Existing server** — auto-detects a running `mold serve` on
  `localhost:7680`.
- **Remote host** — point it at a remote GPU box (e.g. a Linux CUDA machine for
  LTX-2), configured in Settings → Engine, with the API key stored in the macOS
  Keychain.

## Development

Run inside `nix develop` (the devshell wires up Metal, Bun, and the Tauri
toolchain):

```bash
desktop-dev        # Tauri app with hot reload (Vite on :1430)
desktop-build      # build the Mold.app bundle
desktop-check      # CI gate: rustfmt, clippy, vue-tsc, prettier
desktop-test       # cargo test (CPU) + vitest
desktop-ui         # frontend-only Vite server (pair with a running `serve`)
desktop-bun-lock   # regenerate desktop/bun.nix from bun.lock
```

The Rust crate under `desktop/src-tauri` is its own cargo root (excluded from
the workspace); the frontend lives in `desktop/src`. CI runs the `desktop-check`
and `desktop-test` gates via `.github/workflows/desktop.yml`.
