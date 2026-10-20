# mold web gallery

A small Vue 3 + Vite + Tailwind v4.2 SPA that serves as the browser-facing
Mold Studio interface for [mold](../). It talks to the serving `mold serve`
origin and can connect to remembered remote servers from its Machines
workspace. The independently hosted HTTPS Studio is a paired client rather than
an engine: **Settings → Machines → Switch machine** returns to its connection
screen without deleting saved pairings. The selected machine's action menu and
detail page also offer **Disconnect** and **Forget**. Forget removes its secure
local pairing before clearing the selection; a failed removal remains available
for retry. Choose another saved machine or paste a fresh invitation to reconnect.
The numeric loopback engine entry remains fixed in locally served Studio.

Each hosted tab keeps its accepted machine fixed while Studio is mounted. A
selection change reloads the document before another machine can use the workspace,
so cached model details and download actions cannot follow the new selection.
Switch and Forget leave remembered alternatives at the connection screen until
you choose one. Thumbnails are cached by the concrete machine URL and instance,
including across reloads; forgetting that machine evicts its cached tiles.

## Library copies and original inputs

Print menus offer **Copy to** another connected machine. Selection mode can copy several prints. Browser Copy is manual, serial, cancellable, and limited to 64 MiB per print including retained inputs; it uses each machine's captured authenticated Secure or Regular connection. The destination must support retained-media transfer before a source-bearing print is imported. Interrupted copies keep a repair receipt, so repeating Copy repairs retained inputs and organization without intentionally duplicating the output. An unfinished copy of a hidden print remains excluded from default views and source browsing, including collision-renamed outputs, until fresh destination organization confirms privacy. This quarantine grants no verification or organization ownership. An older print whose originals were never retained is reported separately; damaged or unauthorized originals refuse the copy.

Source Media cards and their original viewer offer a host-scoped `source:<print filename>#<member-id>`, a shell-quoted CLI download command, Copy File Name, Save a Copy, and file clipboard/share actions where the browser supports them. Source IDs refer to the displayed owning machine. Image sources and image print menus offer **Use as first frame** / **Use as last frame** when the selected recipe accepts them, including LTX endpoint keyframes. Established input absence says **No source media.**; read failures remain retryable. Hidden membership on a trashed copy continues protecting live copies until its removal is confirmed.

## Inspecting inputs

Use **View** (native apps) or **View original** (web, desktop and mobile) beside an attached input to inspect it without replacing or removing it. References, source/target pictures, boundary frames, masks, control and identity pictures, and attached video/audio have separate viewing actions. Image previews fit the whole picture. Use **Actual size** on desktop and web, or pinch to zoom in the native iOS app, for close inspection. **Done** or Escape returns to the attachment or picker. Picker previews do not select or attach the picture. Queue viewing actions show the full submitted media, which may already have been fitted to the generation canvas; older machines without the original-input endpoint report it unavailable.

## Quick start

```bash
cd web
bun install
bun run dev          # http://localhost:5174 (proxies /api to :7680)
```

For remote GPU hosts:

```bash
MOLD_API_ORIGIN=http://hal9000:7680 bun run dev
```

## Production build

The SPA is **embedded into the `mold` binary at compile time** — the
Nix flake builds it reproducibly via [`bun2nix`](https://github.com/nix-community/bun2nix)
(see `web/bun.nix`) and passes the output to `crates/mold-server/build.rs`
through `MOLD_WEB_DIST`, which stamps `MOLD_EMBED_WEB_DIR` for
[`rust-embed`](https://crates.io/crates/rust-embed). Result: `nix build`
produces a single-file server that serves this gallery with no runtime
dependency on `~/.mold/web`, `share/mold/web`, or any external `dist/`.

```bash
bun run build        # outputs to web/dist
```

For **plain cargo builds** outside Nix, `build.rs` picks up a
`web/dist/` next to the crate automatically. Run the SPA build before
`cargo build` if you want the real gallery baked in; skip it and the
binary falls back to an inline "mold is running" placeholder page.

For **SPA hot-iteration without recompiling Rust**, point a running
server at a local `dist/`:

```bash
MOLD_WEB_DIR=$(pwd)/dist mold serve
```

`MOLD_WEB_DIR` (plus the legacy `$XDG_DATA_HOME/mold/web`,
`~/.mold/web`, `<binary dir>/web`, and `./web/dist` candidates) is
resolved at request time and takes precedence over the embedded bundle,
so you can swap in new builds without restarting — no recompile. API
routes (`/api/*`, `/health`, `/metrics`) are matched first by the axum
router; the SPA fallback handles everything else and reuses
`index.html` for SPA deep-link routes.

## What you get

- **Multi-host Machines workspace** with direct per-host status and detail
  views. Current mDNS-enabled servers expose peers visible on the server's LAN
  through `GET /api/discovery/peers`; the browser feature-detects this via
  `discovery.can_browse`, test-connects each selection directly, and dedupes by
  stable instance UUID. Older servers keep Local network disabled without a
  probe. Create snapshots the chosen route for generation, preprocessing,
  expansion, cancellation, and queue reconciliation; prepared expansion
  batches keep their reviewed prompts and provenance together on that machine.
  Generation captions and stale reasons use inventory-backed model titles while
  retaining stable catalog IDs in requests and persisted metadata.
- **Feed ↔ Grid toggle** in the header, persisted in `localStorage`.
  **Feed** (default) is a Tumblr-style single-column stream — full-bleed
  edge-to-edge cards on mobile, constrained reading width on desktop,
  media rendered at its natural aspect ratio via `object-contain`. Full
  resolution is loaded directly (no thumbnail stage) so HiDPI phones
  show sharp images. **Grid** is a dense masonry (2 → 6 columns) with a
  hover-reveal caption and `object-cover` tiles — good for scanning
  thousands of items.
- **Chunked rendering** — 40 items/page in feed mode, 150 in grid
  (feed cards are taller). Loads more via an IntersectionObserver
  sentinel 800 px from the bottom. Searching or switching modes
  resets the window.
- **Fallback chain** `thumbnail → full image → broken-file tile` means
  truly unreadable files show a subtle red "can't render" card instead
  of the browser's default broken-icon.
- **Videos autoplay** while on-screen (loop + playsinline) and pause
  automatically when you scroll past them. No hover required. A speaker
  toggle in the header flips the global mute preference (persisted in
  `localStorage`) — browsers require a user gesture before the first
  unmuted autoplay, and clicking the toggle satisfies that, so every
  subsequent in-view video plays with sound.
- `<video>` uses the full-file URL as `src` and the thumbnail URL as
  the static `poster` — `src` must be a video file, never a PNG.
- **Search bar** (180 ms debounce) matches prompts, model names, and
  filenames. All / Images / Video filter pills compose with it.
- **Detail drawer** adapts to screen size:
  - **Mobile** (< lg) — fullscreen media viewer. Swipe **down** for
    the next (older) item, **up** for the previous. The top bar shows
    close / `N / total` counter / details toggle. Metadata tucks away
    behind the details toggle and pops up from the bottom as a sheet;
    dismissible via backdrop tap, Esc, or re-tap.
  - **Desktop** (lg+) — media pane + always-visible right sidebar with
    the full metadata panel.
  - Keyboard: Esc closes, ← / → or ↑ / ↓ or k / j to step through the
    filtered list, `i` toggles the mobile sheet.
  - Full `OutputMetadata` panel, prompt + seed copy-to-clipboard,
    download, delete.
- **Supports every `OutputFormat`** the server can emit: PNG, JPEG,
  GIF, APNG, WebP, MP4. MP4 thumbnails are real first-frame PNGs.
- **Transparent mold logo** mirrored from the docs site at
  `web/public/logo.png` — also used as the favicon / apple-touch-icon.

## Stack

- Vue 3 composition API + TypeScript
- Vite 7
- Tailwind CSS v4.2 via `@tailwindcss/vite`

Vue Router provides the five-workspace shell; composables and module-singleton
stores provide state without a separate state-library dependency.

Queue details show a plain-language explanation and expandable, selectable
**Technical details** with the original machine diagnostic and job identity.
Held work keeps **Cancel job** available even before settings finish loading.
Cancellation rechecks the original server instance and current job state;
completed work and an unreachable or replaced machine cannot be cancelled.

Queue controls follow the job’s current state and owning machine. Waiting jobs
offer Pause and paused jobs Resume when supported by the machine. Held jobs
offer Retry or Download and Retry when the host supplies durable recovery authority. Running cancellation requires the
host’s advertised support; completed, cancelled and failed jobs remain read-only.
Busy, offline or changed-machine rows keep Details accessible. On current Mold
servers, Held cancellation remains held-only even if another client starts the
job before the request arrives.

Fresh H3 Ref2VA video attachments use the selected machine’s advertised reference-video normalization capability. Common negotiated containers can be attached without a browser decoder; original bytes and their SHA-256 remain in the local draft media store. Preparing video selects one eligible machine under the existing Auto/Most capable policy, then pins this submission’s placement and admission to that machine. Explicit machine choices stay explicit. The machine probes or converts the original, and the client checks every original/canonical digest and receipt before explicit finalization. Duration and soundtrack budgets are verified at preparation; no generation estimate is shown while these facts are pending. The per-file limit is 256 MiB or the machine’s smaller advertised limit. Older machines retain the existing canonical MP4 path. Retained sources, source/continuation video and existing scoped authorities retain their current contracts.

A stalled job remains live and may still reserve its device. Queue rows and Job Details show the reported stage, elapsed time without completed work, cancellation state and recovery guidance. An accepted cancellation can say **Cancel requested; device still held** until the machine reports a terminal outcome. It never frees the device optimistically. Copy Details includes the machine’s stall diagnostic for an operator.

### Starting a stopped video again

In Queue, a failed, interrupted or paused multi-clip video offers **Start again…**. Review the destination, then choose **Start again** on its original machine or **Send video** on another connected machine. Saved prompts, seed, settings and inputs stay together; the original remains unchanged. Machine-local adapters can restart only on their original machine. Unsupported retained inputs explain the refusal before admission. A lost reply keeps the known job for **Check queue**, and **Stop new attempt** cancels that destination operation without changing the original.

On updated machines, **Delete recovery data…** is available for a failed or interrupted multi-clip job. Its separate confirmation deletes that job and retained inputs; saved Library videos remain available. The original installation and job revision must still match. **Stop new attempt** requires the updated installation-fenced control; an interrupted cancellation reply keeps its intent for reconciliation with the same job ID.
