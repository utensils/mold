# Web UI: generation capability

**Status:** draft — 2026-04-19
**Branch:** `feat/web-generate-ui` (base: `multi-gpu`)
**Owner:** Jeff

## 1. Background

Today the SPA at `web/` is gallery-only: it lists, views, and optionally deletes saved images. All generation happens through `mold run` (CLI) or `mold tui` (terminal). The server already exposes the full generation surface over HTTP, including SSE streaming progress and per-GPU worker status from the multi-GPU branch.

This feature adds a browser-driven generation experience that talks to the same `/api/generate*` endpoints the CLI and TUI use, with no server-side changes in v1.

## 2. Goals

- A browser-native way to create generations without leaving the gallery.
- First-class multi-GPU visibility — per-job GPU ordinal and live step progress.
- Reuses the existing gallery aesthetic (`glass`, rounded, dark-first, indigo accent).
- Works on mobile (the existing gallery is responsive; the generate page follows suit).
- No server-side changes for v1 — all wiring uses endpoints that already ship in `multi-gpu`.

## 3. Non-goals (v1)

Explicitly deferred:

- LoRA adapter selection.
- Inpainting mask upload.
- ControlNet conditioning.
- LTX-2 advanced modes: keyframes, audio-to-video, retake, spatial/temporal upscale.
- Post-generate upscale chain (`upscale_model`).
- Embed-metadata toggle.
- Pull-on-demand from the UI — undownloaded models are shown as disabled with CLI guidance.
- Per-request LLM / temperature / thinking-mode knobs for prompt expansion (server-side config only today).
- Qwen-Image-Edit multi-reference `edit_images`.

All of the above are tracked in a "Future work" section at the end.

## 4. User stories

1. *As a user on my laptop*, I open `http://host:7680/generate`, type a prompt, press Enter, and watch a FLUX generation stream to completion with per-step progress and a thumbnail.
2. *As a user with a 2x-GPU host*, I fire three prompts in a row. Two land on GPU 0 and GPU 1 simultaneously; the third is queued. The UI shows all three as distinct running cards with GPU badges.
3. *As a user browsing the gallery*, I click "Generate" in the top bar, get dropped into `/generate`, and my last-used prompt + settings are still in the form from localStorage.
4. *As a user remixing an existing image*, I drag one of my gallery images onto the composer; a thumbnail chip appears; the settings modal reveals a `strength` slider; generation runs as img2img.
5. *As a user generating a video*, I open the settings modal, pick an `ltx-video` model (clearly labeled with a video badge), set frames/fps, and generate. The result lands in the gallery feed as a playable MP4.
6. *As a user writing a short prompt*, I click `✨`, toggle expansion on with 3 variations, hit "Preview", see three rewrites, pick one, then submit.

## 5. UX

### 5.1 Routing

- Introduce `vue-router@4`.
- Two routes: `/` (gallery, unchanged) and `/generate`.
- `TopBar` gets a two-tab switcher. Active route styled with `brand-500` accent; inactive is `text-slate-400`.
- The existing `/api/*`, `/health`, `/metrics` routes are matched server-side first; the SPA fallback already handles deep links via `index.html`, so no server change is needed.

### 5.2 /generate page layout (desktop, lg+)

Vertical stack inside the same `max-w-[1800px]` container used by the gallery:

1. **Composer** — full-width `glass` rounded block.
2. **Running strip** — horizontal row of in-flight job cards, hidden when empty.
3. **Gallery feed** — reuses `<GalleryFeed>` verbatim with the existing global `/api/gallery` data.

### 5.3 Composer

A rounded `glass` container. Inside:

- A large multi-line `<textarea>` with autogrow (up to ~8 lines, then scroll).
- **Keyboard contract:** `Enter` submits the current form; `Shift+Enter` inserts a newline. No visible submit button.
- Right-edge vertical pill of three icon buttons:
  - `🖼️` — opens the image-picker modal. Also acts as the drop target for dragged files or gallery items.
  - `✨` — opens the expand modal. Icon pulses when expansion is enabled.
  - `⚙` — opens the settings modal. A subtle dot on the icon indicates non-default values.
- When a source image is attached, a small (~48px) thumbnail chip appears to the left of the textarea with an `✕` button to clear it. The chip shows a play-icon overlay if the attached item is a video (MP4/GIF/APNG/WebP).
- A tiny one-line status below the textarea when relevant: queue depth (`queue 1/8`), GPU state pills (`GPU 0 ▮ · GPU 1 ▯`), polled from `/api/status` every 5s while the tab is visible.

### 5.4 Running strip

- Hidden (`display: none`) when `runningJobs.length === 0`.
- Each job card (fixed width ~280px on desktop, 100% on mobile stack):
  - Header: model name, GPU ordinal badge (`GPU 0`), elapsed time.
  - Progress: step bar (`14 / 28`), stage label (`Denoising` / `Loading T5 encoder` / `Queued, position 3`).
  - Body: reserved 1:1 tile that shows a placeholder gradient until `SseCompleteEvent` arrives, then the final thumbnail.
  - Footer: `✕` cancel button (for v1: closes the local stream; server continues — that's fine for now and we document it).
- A completed card stays visible for ~1 second with a `✓` and then slides out of the strip.

### 5.5 Settings modal (`⚙`)

Centered `glass` modal, fullscreen sheet on mobile. Sections, top to bottom:

- **Model** — dropdown populated from `GET /api/models`, filtered to generation families (not upscalers, not utility). Default list shows `downloaded: true` models; a toggle "Show all models (N)" reveals the rest (the rest are disabled with a tooltip *"Not downloaded — run `mold pull <name>` on the host."*). Video families (`ltx-video`, `ltx2`) render in a dedicated sub-section with a `🎬` badge and a tiny italic caption *"video"*.
- **Size** — width/height chips: 512, 768, 1024 + a Custom row that exposes two numeric inputs. Defaults pulled from `ModelDefaults` when the model changes.
- **Steps** — slider, default from model.
- **Guidance** — slider 0.0–20.0, default from model.
- **Seed** — numeric input + `🎲 random` toggle. Random = send `seed: null`.
- **Negative prompt** — `<textarea>`; hidden when the selected model family is in `{flux, z-image, flux2, qwen-image}` (the families that don't use CFG).
- **Batch size** — 1 | 2 | 3 | 4 chips.
- **Context-sensitive rows:**
  - `Strength` slider (0.0–1.0, default 0.75) — only when a source image is attached.
  - `Frames` stepper (must be `8n+1`, clamped) and `FPS` numeric — only when the selected model family is `ltx-video` or `ltx2`.
- **Advanced** (collapsed by default):
  - Scheduler dropdown — only when family is `sd15` or `sdxl`.
  - Output format — PNG / JPEG / WebP (image families); MP4 / GIF / APNG / WebP (video families).
- **Reset to defaults** at the bottom, affects every field above.

All state persists to `localStorage["mold.generate.form"]` on change.

### 5.6 Expand modal (`✨`)

Smaller centered `glass` modal. Fields:

- **Enabled** toggle. Off by default.
- **Variations** — chips: 1 | 3 | 5. Default 1.
- **Model family** — hidden behind a "Custom family" disclosure; defaults to the selected model's family.
- **Preview** button — calls `POST /api/expand` with `{ prompt, model_family, variations }`. Renders results below the button:
  - If `variations === 1`, show the single rewrite and a "Use this" button that replaces the composer prompt.
  - If `variations > 1`, show each rewrite as a selectable card; clicking one replaces the composer prompt and closes the modal.
- A note at the bottom: *"Backend model / temperature / thinking mode are controlled by the server config. Ask your operator to adjust them."*

When enabled and no preview is shown, the composer submits with `GenerateRequest.expand: true` — the server runs expansion inline and records the original in `metadata.original_prompt`. Server-side inline expansion produces a single rewrite; `variations > 1` therefore requires the Preview flow (the client calls `/api/expand`, the user picks one, the picked variant becomes the prompt, and the request is submitted with `expand: false`).

### 5.7 Image-picker modal (`🖼️`)

Two tabs:

- **Upload** — a large drop-zone + file input. Accepts PNG, JPEG, WebP up to a client-enforced 10 MB (server won't refuse larger, but payloads over ~15 MB stress the SSE path). Read as `Blob`, converted to base64 on the client.
- **From Gallery** — a compact grid populated by `GET /api/gallery`. Clicking a tile fetches `GET /api/gallery/image/:name`, base64-encodes the bytes, and stores them as the active `source_image`. Video items are selectable when a video family is active (LTX-2 video-to-video uses `source_video`; out of v1 — gate that tab item with `disabled` when the selected model is not a video family).

Drag-and-drop onto the composer is a shortcut for the Upload tab. Dragging a tile from `/` onto `/generate` is **out of v1** (requires cross-route drag and we don't want to start tracking global drag state yet).

### 5.8 Gallery feed below

Literally `<GalleryFeed :entries="galleryEntries" />` with the same props and keyboard/drawer behavior as `/`. The `view` (feed vs grid) and `muted` preferences read from the same `localStorage` keys as the gallery page, so toggling on either route stays in sync. Two additions:

- On every `SseCompleteEvent`, call `listGallery()` and merge by `filename` (existing code already de-dupes on `filename` in the drawer). This keeps the feed fresh without polling.
- New items pulse briefly (`animate-pulse` for ~1s) so returning users can see what just finished.

### 5.9 Responsive behavior

- `lg+` (≥ 1024px): layout described above.
- Below `lg`:
  - Composer: textarea expands, icons stack into a row along the bottom.
  - Running strip: horizontal overflow-scroll with snap; card width drops to ~240px.
  - Settings / Expand / Image-picker modals: fullscreen bottom sheets with the same touch-action pattern already used by `DetailDrawer`.

## 6. API surface (no server changes)

All endpoints already exist on `multi-gpu`.

| Endpoint | Usage |
|---|---|
| `POST /api/generate/stream` | One SSE stream per job. Body is `GenerateRequest`. We parse progress events and the completion event. |
| `POST /api/expand` | On-demand rewrite preview. |
| `GET /api/models` | Populate the model picker; shows `downloaded`, `family`, `defaults`. |
| `GET /api/status` | Polled every 5s while `/generate` is focused — drives the GPU state pills and queue readouts. |
| `GET /api/gallery` | The feed below the running strip. Also the "From Gallery" picker. |
| `GET /api/gallery/image/:name` | Used by the picker to pull base64 bytes for `source_image`. |
| `GET /api/capabilities` | Already fetched on mount; reuse. |

### 6.1 SSE client

The browser `EventSource` API only does `GET`. We need `POST` with a JSON body. Use `fetch()` + `ReadableStream` + a tiny line-based SSE parser (the same technique the CLI/TUI use against the server). Cancelation = `AbortController.abort()`.

### 6.2 Error paths

- **`503 queue_full`** — read `Retry-After`, show an inline banner on the composer, auto-retry once. If the second attempt also 503s, surface the error and stop retrying.
- **`4xx` validation** — toast in the existing error slot; highlight the offending field in the settings modal.
- **Stream disconnect mid-job** — the running card turns `rose-500` with "disconnected — result may still have saved" and a `Refresh gallery` button.
- **Selected model is undownloaded** — the submit is blocked client-side with a tooltip pointing at `mold pull`.

## 7. State model

Three top-level reactive refs inside `/generate`:

- `formState: ref<GenerateFormState>` — the fully serializable form shape. Persisted to `localStorage["mold.generate.form"]` via a `watch(formState, { deep: true })` with a 300 ms debounce. Defaults are applied field-by-field when missing (graceful migration).
- `runningJobs: ref<Job[]>` — in-memory only. Each `Job` holds `{ id: string, request: GenerateRequest, controller: AbortController, progress: ProgressState, result?: SseCompleteEvent, error?: string }`.
- `galleryEntries: ref<GalleryImage[]>` — the feed backing store. Refreshed on completion.

`GenerateFormState` shape (TypeScript):

```ts
interface GenerateFormState {
  prompt: string;
  negativePrompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null;          // null = random
  batchSize: number;
  strength: number;
  frames: number | null;
  fps: number | null;
  scheduler: Scheduler | null;
  outputFormat: OutputFormat;
  expand: { enabled: boolean; variations: number; familyOverride: string | null };
  sourceImage: { kind: "upload" | "gallery"; filename: string; base64: string } | null;
}
```

No Pinia/Vuex — `ref` + `watch` is enough. The form state is the single source of truth; the `GenerateRequest` is derived in a single `toRequest(formState)` helper.

## 8. Persistence

- `localStorage["mold.generate.form"]` — full form state minus `sourceImage.base64` (we drop the base64 on persist to avoid 5 MB of quota churn; the chip is re-attachable on reload).
- `localStorage["mold.generate.showAllModels"]` — the "show all 97" toggle.
- Version key in the persisted blob: `{ version: 1, ... }` so future shape changes can migrate or reset cleanly.

## 9. Files

### 9.1 New

```
web/src/pages/GeneratePage.vue
web/src/components/Composer.vue
web/src/components/RunningStrip.vue
web/src/components/RunningJobCard.vue
web/src/components/SettingsModal.vue
web/src/components/ExpandModal.vue
web/src/components/ImagePickerModal.vue
web/src/components/ModelPicker.vue
web/src/composables/useGenerateForm.ts
web/src/composables/useGenerateStream.ts
web/src/composables/useStatusPoll.ts
web/src/lib/sse.ts
web/src/lib/base64.ts
web/src/router.ts
```

### 9.2 Modified

```
web/src/main.ts              — register the router
web/src/App.vue              — becomes a thin <router-view> wrapper
web/src/pages/GalleryPage.vue — extracted from today's App.vue (no behavior change)
web/src/components/TopBar.vue — add tab switcher
web/src/api.ts               — add generateStream, fetchModels, fetchStatus, expandPrompt
web/src/types.ts             — add GenerateRequest, GenerateResponse, SseProgressEvent, SseCompleteEvent, ModelInfoExtended, ServerStatus, GpuWorkerStatus, GenerateFormState
web/package.json             — add vue-router@^4
```

### 9.3 Not modified

No server crate changes. No `crates/mold-core` wire-format changes. No `crates/mold-server` route changes. `build.rs` already picks up the new `dist/` without work.

## 10. Testing

- Manual UAT against the running `multi-gpu` server on BEAST via the existing SSH tunnel (`MOLD_HOST=http://localhost:7680`).
- `bun run dev` on the laptop for hot-iteration; Vite proxies `/api` to `:7680`.
- The two downloaded models there (`sd15:fp16`, `sdxl-turbo:fp16`) cover two CFG-using families — good enough for v1 smoke tests. Video testing requires pulling `ltx-video` or `ltx2` on the server.
- Cypress / Playwright tests are **out of v1**. Author will do scripted UAT (checklist in the plan) before marking ready.
- TypeScript: `bun run check` (or Vite's type check script) must pass.
- Prettier: `bun run fmt` must be clean.

## 11. Open questions

None that block v1. Captured for v2:

- Server-side additions for the expand modal (LLM choice, temperature, thinking mode).
- A `source_filename` field on `GenerateRequest` to avoid base64 round-trip for gallery-sourced images.
- Drag-and-drop from the `/` gallery onto `/generate` composer.

## 12. Future work (post-v1, in rough priority order)

1. LoRA adapter picker (stacked LoRAs when the family supports it).
2. Inpainting mask upload — needs a paint-mask editor, which is a project on its own.
3. ControlNet conditioning for SD1.5.
4. Video advanced (keyframes, audio-to-video, retake, spatial/temporal upscale for LTX-2).
5. Post-generate upscale chain (`upscale_model`).
6. Pull-on-demand from the picker (streams `/api/models/pull`).
7. Expand modal — expose server-side knobs if/when `ExpandRequest` gains fields.
8. Session-only filter in the gallery feed.
9. Qwen-Image-Edit `edit_images` multi-reference.
