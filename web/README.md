# mold web gallery

A small Vue 3 + Vite + Tailwind v4.2 SPA that serves as the browser-facing
gallery for [mold](../). It talks to a running `mold serve` via the existing
`/api/gallery`, `/api/gallery/image/:name`, `/api/gallery/thumbnail/:name`,
and `DELETE /api/gallery/image/:name` endpoints.

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

No router, no state library — the feature is small enough that component
refs and `watch` are plenty.
