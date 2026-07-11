# Desktop App <Badge type="warning" text="experimental" />

mold ships an experimental native macOS desktop app — a Tauri 2 shell around a
Vue 3 + TypeScript frontend with its own **Safelight** design language, a warm,
matte "digital darkroom" that treats every generation as a print being
developed.

::: warning Experimental
The desktop app lives in `desktop/` and is under active development. It is
macOS-first (Apple Silicon, Metal) and not yet part of a tagged release. Build
it from source with the devshell commands below.
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
  every parameter restored. When the active engine is remote, **This Mac**
  remains available as a separate local-gallery source. Still images offer
  full-resolution **Copy image** from tile and lightbox right-click menus.
- **Models & catalog** — installed models grouped by family with residency and
  disk usage, plus a live HuggingFace/Civitai catalog. Pulls render **SIZE vs
  FETCH** honestly (model weights vs. the full download including shared
  components), run in parallel, and stream through a downloads tray whose
  cancel action aborts queued or active work rather than only hiding the row.
- **Chains** — a filmstrip editing bench for multi-stage video
  (`mold.chain.v1`): per-stage prompts and frame counts (validated `8n+1`),
  splice transitions (smooth / cut / fade) you click to cycle, a live
  fits/duration forecast against `/api/capabilities/chain-limits`, TOML
  import/export, and a durable jobs list with resume, cancel, and retake.
- **History** — a fast, searchable list of past prompts; ↩ refills the composer.
- **RunPod** — secure account setup, balance and live spend, GPU and
  datacenter discovery, pod launch/lifecycle/connection, and persistent network
  volume create/select/rename/grow/delete. A selected volume is remembered,
  forces Secure Cloud in its datacenter, replaces the ordinary workspace disk,
  and cannot be deleted while attached to a pod. Because RunPod cannot stop a
  network-volume Pod, the app hides Start/Stop for those rows and explains that
  deleting the compute instance preserves `/workspace` on the volume. Logs use
  a supported handoff to the RunPod console rather than a nonexistent REST
  endpoint. Production network volumes accept 10–3999 GB; the form and native
  validation enforce that live bound before launch.
- **Settings** — a full preferences bench with a section rail and
  cross-section search: Engine (connection, native folder pickers for the
  models/output directories), Performance (the `MOLD_*` engine knobs as real
  controls, applied on engine restart), Generation defaults, a Prompt
  expansion form, Accounts & tokens (Hugging Face / Civitai keys in the macOS
  Keychain in signed builds and an owner-only local file in debug builds,
  exported to the engine as `HF_TOKEN`/`CIVITAI_TOKEN`), Appearance
  (the website-aligned Mold palette by default or the original Safelight,
  each with System/Dark/Light; media never inverts), Profiles (switch or create), and
  Advanced — every remaining `/api/config` row with its provenance tag (⌂ db /
  ⛁ file / ⚿ env); environment-overridden rows are locked with the variable
  that owns them.
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
| ⌘0 / ⌘+ / ⌘− | Interface size reset / larger / smaller |

Interface scaling applies to the complete app, including fixed overlays and
right-click menus. Choose 80–130% from **Settings → Appearance & app → Interface size**, or
use the View menu and keyboard shortcuts. The selected level is restored on
the next launch.

## Generation templates

Save the current Generate form as a named, recallable preset. Open the
**Templates** panel below the LoRA stack, give the current settings a name, and
it is stored as a template you can load, rename, or delete later. Loading a
template restores every parameter — model, prompt, dimensions, steps, guidance,
scheduler, LoRA stack, and the rest — in one click.

Templates capture _parameters_, not media: source, mask, and control images
(and LTX-2 source video / keyframes) are referenced but never stored, so after
loading a template that used them the app reminds you to re-select the files.
If the template's model isn't installed you still get its settings, with a
prompt to pull the model.

Templates are stored locally, per machine (in the app's own storage), and are
never shared with the browser SPA or synced to the server — they travel with
the Mac you saved them on.

## Device placement

**Settings → Advanced → Device placement** saves a per-model default for _where_
a model's components run. Pick an installed model, then set its **Text
encoders** — the Tier-1 group knob covering T5, CLIP, and Qwen encoders — to
**Auto**, **CPU**, or a specific **GPU**. For Tier-2 families (FLUX, Flux.2,
Z-Image, Qwen-Image) an **Advanced** disclosure exposes per-component overrides
for the transformer, VAE, and each text encoder; any encoder can also be left to
follow the group knob.

**Save as default** persists the choice for that model and **Clear** removes it.
Placement is applied the next time the model loads, so save it before you
generate. GPU choices come from the connected engine's live device list.

This is the desktop surface for the same mechanism the CLI's `--device-*` flags
and `MOLD_PLACE_*` variables drive — see
[Configuration → Per-component device placement](./configuration.md#per-component-device-placement)
for the full component list and semantics.

## How it connects

The app talks to a `mold-ai-server` over localhost HTTP + SSE using the same
wire types as the CLI and web UI:

- **Built-in engine** — embeds the server in-process and runs on Metal, so no
  separate `mold serve` is required.
- **Existing server** — auto-detects a running `mold serve` on
  `localhost:7680`.
- **Remote host** — point it at a remote GPU box (e.g. a Linux CUDA machine for
  LTX-2), configured in Settings → Engine, with the API key stored in the macOS
  Keychain in signed builds (the owner-only debug secret store is used during
  local development). A bare hostname is enough: `hal9000` expands to
  `http://hal9000:7680`. The network list uses the operating system's native
  DNS-SD browser on macOS, so advertised `_mold._tcp` services share the same
  cache and interface handling as Finder and `dns-sd`.

## Development

Run inside `nix develop` (the devshell wires up Metal, Bun, and the Tauri
toolchain):

```bash
desktop-dev        # Tauri app with hot reload (Vite on :1430)
desktop-build      # build the Mold.app bundle
desktop-release    # signed + notarized + stapled app and DMG, then verify
desktop-check      # CI gate: rustfmt, clippy, vue-tsc, prettier
desktop-test       # cargo test (CPU) + vitest
desktop-ui         # frontend-only Vite server (pair with a running `serve`)
desktop-bun-lock   # regenerate desktop/bun.nix from bun.lock
```

The Rust crate under `desktop/src-tauri` is its own cargo root (excluded from
the workspace); the frontend lives in `desktop/src`. CI runs the `desktop-check`
and `desktop-test` gates via `.github/workflows/desktop.yml`.

## Signed distribution

`desktop-release` is the release-grade local path. It requires
`.secrets/signing.env` (gitignored) with `APPLE_SIGNING_IDENTITY` and App Store
Connect API credentials (`APPLE_API_ISSUER`, `APPLE_API_KEY`, and
`APPLE_API_KEY_PATH`). The command builds the Metal-enabled app and DMG, waits
for Apple notarization, staples the ticket, then verifies the hardened-runtime
signature, entitlements, Gatekeeper acceptance, and staple on both artifacts.

The Desktop GitHub Actions workflow runs the same signed distribution job on
`v*` tags and manual dispatch. Repository secrets hold the exported Developer
ID certificate and App Store Connect key; the private key is written only to
the runner's temporary directory and the temporary signing keychain is removed
even if the build fails.
