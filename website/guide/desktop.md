# Desktop App <Badge type="warning" text="experimental" />

mold ships an experimental native macOS and Linux desktop app — a Tauri 2 shell around a
Vue 3 + TypeScript frontend with its own **Safelight** design language, a warm,
matte "digital darkroom" that treats every generation as a print being
developed.

::: warning Experimental
The desktop app lives in `desktop/` and is under active development. Local
generation uses Metal on Apple Silicon and CUDA on x86_64 Linux.
:::

## Download

Every tagged release ships a signed, notarized, stapled DMG:

**[⬇ Download Mold for macOS (Apple Silicon)](https://github.com/utensils/mold/releases/latest/download/Mold-macos-arm64.dmg)**

Open the DMG and drag **Mold** to Applications — no quarantine dance needed.
Version-pinned DMGs and `SHA256SUMS` are on the
[releases page](https://github.com/utensils/mold/releases). You can also build
from source with the devshell commands below.

Linux builds are currently source/CI distributions: `nix build
.#mold-desktop` produces the native sm_89 package (`.#mold-desktop-sm120` for
Blackwell). `desktop-build` produces that native package on NixOS and a CUDA
AppImage on conventional Linux. Tagged releases do not publish the AppImage yet.

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
  every parameter restored. When the active engine is remote, **This device**
  remains available as a separate local-gallery source. Still images offer
  full-resolution **Copy image** from tile and lightbox right-click menus.
- **Models & catalog** — one searchable installed-first library with All,
  Installed, and Available filters plus compact Grid and Table layouts. The
  desktop reuses cacheable 512 px Civitai thumbnails across both layouts,
  lazily decodes them, and contains each card's layout and paint work. The live
  Hugging Face/Civitai catalog renders **SIZE vs FETCH** honestly. With
  several hosts connected, Pull asks which host should store the model.
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
  validation enforce that live bound before launch. Region selectors show both
  the geographic location and RunPod ID, while the volume form limits choices
  to datacenters that currently support persistent volumes.
- **Settings** — a full preferences bench with a section rail and
  cross-section search: Engine (connection, native folder pickers for the
  models/output directories), Performance (the `MOLD_*` engine knobs as real
  controls, applied on engine restart), Generation defaults, a Prompt
  expansion form, Accounts & tokens (Hugging Face / Civitai keys in an
  owner-only local file under the app's data directory — no Keychain prompts —
  exported to the engine as `HF_TOKEN`/`CIVITAI_TOKEN`), Appearance
  (the website-aligned Mold palette by default or the original Safelight,
  each with System/Dark/Light; media never inverts), Profiles (switch or create), and
  Advanced — every remaining `/api/config` row with its provenance tag (⌂ db /
  ⛁ file / ⚿ env); environment-overridden rows are locked with the variable
  that owns them.
- **Command palette** — **Cmd/Ctrl+K** for navigation, actions, model search, and
  prompt-history search in one field.
- **Native desktop integration** — platform menus and shortcuts, Linux native
  window decorations, macOS overlay chrome, and background notifications on
  generation, chain, and pull completion.

## Updates

Automatic signed updates are currently macOS-only. Linux Nix/AppImage builds
report updates as unsupported and are replaced manually.

Signed desktop builds keep update checks separate from installation. Mold makes
a best-effort check after the app opens, and **Mold → Check for Updates…** plus
**Settings → Updates → Check for updates** run the same check manually. A
check only reports what is available: Mold does not download, install, or
restart until you explicitly choose **Update and restart**.

Choose the release stream in **Settings → Updates**:

- **Stable** (default) follows tagged, production releases.
- **Nightly** follows signed and notarized builds from desktop-relevant commits
  on `main`, after both desktop frontend and Rust CI gates pass. Nightlies expose
  changes sooner and may contain regressions.

Both channels use public, HTTPS-hosted manifests:

- [Stable manifest](https://github.com/utensils/mold/releases/latest/download/mold-desktop-stable.json)
- [Nightly manifest](https://github.com/utensils/mold/releases/download/latest/mold-desktop-nightly.json)

Startup checks are non-destructive. When an update is available, Mold shows a
persistent banner in the app; if Mold is backgrounded it also sends a native
notification. Download and installation begin only after you choose **Update
and restart**.

Tauri's updater signature check is mandatory. Before the installed app is
changed, the complete archive passes Minisign verification against Mold's
embedded public key and is fully extracted into temporary storage. Mold rejects
unsafe paths and extra app bundles, binds the bundle identifier and version to
the manifest, runs strict Apple code-signature verification and a Gatekeeper
assessment, validates the currently running bundle, rejects DMG or translocated
launches, and proves the install directory can be replaced. Downloads stop
cleanly after 15 minutes.

Only after every preflight check succeeds does Mold use macOS's atomic bundle
exchange and restart. Mold does not run a post-launch health watchdog or
roll back after a few seconds: the update either verifies and installs, or it
fails before installation and the running version remains in place.

Switching from Nightly to Stable changes which manifest Mold checks, but never
silently downgrades the installed app. If your nightly version is newer than the
latest stable version, Mold reports Stable as current until a newer tagged
release is published.

### Keyboard map

| Shortcut            | Action                                 |
| ------------------- | -------------------------------------- |
| Cmd/Ctrl+1–6, comma | Screens / Settings                     |
| Cmd/Ctrl+K          | Command palette                        |
| Cmd/Ctrl+N          | New generation (clear composer, focus) |
| Cmd/Ctrl+Enter      | Generate                               |
| Cmd/Ctrl+E          | Expand prompt                          |
| Cmd/Ctrl+R          | Randomize seed                         |
| Cmd/Ctrl+.          | Cancel the running job                 |
| Cmd/Ctrl+\          | Toggle sidebar                         |
| Space               | Quick Look in Gallery                  |
| ←/→, ⌫              | Gallery navigate / delete              |
| Shift+Cmd/Ctrl+C    | Copy seed (lightbox)                   |
| Cmd/Ctrl+0 / + / −  | Interface size reset/larger/smaller    |

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

- **Built-in engine and LAN server** — embeds the server in-process and runs on
  Metal on macOS or CUDA on Linux, so no separate `mold serve` is required. It
  listens on port 7680, advertises itself over mDNS, and remains online as
  **This device** even when a remote host is primary. Settings → Engine exposes
  the persistent per-device API
  key that another Mold client needs to connect. If an unrelated process owns
  7680, Mold uses and advertises an ephemeral port instead.
- **Existing server** — auto-detects a running `mold serve` on
  `localhost:7680`.
- **Remote host** — point it at a remote GPU box (e.g. a Linux CUDA machine for
  LTX-2), configured in Settings → Engine, with the API key stored in an
  owner-only file under the app's data directory (never the macOS Keychain, so
  connecting never triggers Keychain prompts). A bare hostname is enough:
  `hal9000` expands to `http://hal9000:7680`. Hosts you connect to are
  remembered as **Recent hosts** for one-click reconnect, each with its own
  stored key, and the last-used host is restored on launch — if it is
  unreachable the app falls back to the built-in engine for that session
  without forgetting your preference. The network list uses the operating
  system's native DNS-SD browser on macOS, so advertised `_mold._tcp` services
  share the same cache and interface handling as Finder and `dns-sd`.
- **Generation controls** — the Size block quick-selects common, per-family
  model-native resolutions (with manual width/height for anything else) and a
  live aspect-ratio/orientation diagram; Seed is an explicit
  **Random | Fixed** toggle with one-click "lock last seed"; the model picker
  marks each model's source (Hugging Face / Civitai / local) and ends in
  **Browse all models →** straight into the catalog, installed models first;
  the VRAM badge states plainly what fits ("VRAM · fits — est. 2.3 GB of
  64.0 GB").
- **Upscaling** — pick a Real-ESRGAN model in the Print panel to upscale every
  print as it develops (the engine pulls the model on first use and retains
  both the original and `-upscaled` result), or
  right-click any gallery image → **Upscale**; the result lands in this Mac's
  gallery. **Reuse settings** always restores the generation canvas, not the
  upscaled file's physical dimensions.
- **Jobs (Cmd/Ctrl+6)** — a queue console for every connected host: the full
  server-side queue (other clients' jobs included), live thumbnails and step
  progress for this app's own jobs, per-job cancel, **Pause/Resume** of a
  host's queue (the running job finishes; nothing new starts), a two-step
  **Cancel all**, and a "Finished this session" list with one-click reuse.
  Pause and cancel-all are feature-detected via `/api/capabilities`, so older
  servers simply hide the controls.
- **History** — two lenses: **Runs** (every finished generation with its
  thumbnail, model, size, seed, and step count — click to reuse the full
  settings including the seed) and **Prompts** (the raw prompt log, searchable,
  for prompts whose outputs are gone).
- **Remote prints saved locally** — generations from remote hosts and RunPod
  are also written into this Mac's output directory (Settings → App → "Save
  remote prints locally", on by default), with embedded metadata intact, so
  your local gallery stays the complete record even when the GPU lives
  elsewhere. The gallery's right-click menu adds **Save to this Mac** for
  pulling any older remote print down on demand.
- **Several hosts at once** — beyond the primary engine, any number of extra
  hosts can be live simultaneously (**Add as extra host** in Settings →
  Engine, or the **+** next to a detected server in the sidebar's HOSTS
  section). With more than one live host, the Generate inspector grows a
  **Host** selector: pick one explicitly, leave it on **Auto** to route
  each batch to the least-busy host by live queue depth, or choose **Most
  capable** to always target the strongest GPU (CUDA over Metal, then most
  VRAM, then shallowest queue). Both automatic modes prefer hosts that
  already have the selected model installed, and the model picker lists
  every connected host's models — one that only lives on an extra host is
  tagged with the host that has it, and routing there just works. Jobs
  stream progress from — and cancel against — the host they queued on, so a
  long LTX-2 render on a CUDA box never blocks quick local prints. Extra
  connections are remembered and restored on the next launch.

  The Generate workspace waits for those remembered hosts and their model
  inventories before deciding the machine is empty. The “pull your first
  model” screen appears only when every connected host reports zero installed
  generation models; a remote-only model is selected and routed without a
  local download.

## Development

Run inside `nix develop` (the devshell wires up Metal or CUDA, Bun, Tauri, and
Linux WebKitGTK/GStreamer dependencies):

```bash
desktop-dev        # Tauri app with hot reload (Vite on :1430)
desktop-build      # build Mold.app, a Linux AppImage, or the native NixOS package
desktop-release    # signed + notarized + stapled app and DMG, then verify
desktop-check      # CI gate: rustfmt, clippy, vue-tsc, prettier
desktop-test       # cargo test (CPU) + vitest
desktop-ui         # frontend-only Vite server (pair with a running `serve`)
desktop-bun-lock   # regenerate desktop/bun.nix from bun.lock
```

On Linux, `nix build .#mold-desktop` builds the sm_89 native package and
`nix build .#mold-desktop-sm120` targets Blackwell. `CUDA_COMPUTE_CAP` controls
local dev/AppImage compilation. `desktop-release` remains the macOS
sign/notarize path and intentionally exits on Linux.

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

CI runs the same signed distribution job from
`.github/workflows/desktop-distribution.yml`. Tagged releases publish the Stable
DMG, updater archive, signature, and `mold-desktop-stable.json`; desktop-relevant
commits on `main` publish their signed Nightly counterparts to the rolling
`latest` prerelease only after desktop CI passes. Both publication paths verify
the archived app and updater signature against the exact public key embedded in
Mold, then prove that the public manifest points at an anonymously downloadable
payload before moving the channel pointer. Nightly publication prunes only old
desktop assets after that verification and retains ten generations; unrelated
CLI assets and the current manifest target are never selected.

Repository secrets hold the exported Developer ID certificate, App Store
Connect key, and the Tauri updater credentials `TAURI_SIGNING_PRIVATE_KEY` and
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. Never print or commit the updater private
key. Keep a controlled offline backup: losing it prevents already installed
copies from trusting future updates. Key rotation must be staged by first
shipping the replacement public key in an update signed with the existing key.
Runner-only key material is written to temporary paths, and the temporary
signing keychain is removed even if the build fails.
