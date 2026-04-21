# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Multi-GPU worker affinity now holds end-to-end for queued generation, prompt expansion, and upscaling.** The generation dispatcher no longer rejects work just because the tiny per-worker channels are full; jobs stay pending in the configured global queue until a worker can accept them. Explicit placement GPU ordinals are now validated against the active worker pool and may target only one worker GPU per request/config entry, so per-component overrides can no longer silently allocate on a sibling card while auto-placement heuristics continue reading VRAM from the bound worker. Busy workers keep advertising their active model during the cache `take()` window, so follow-up requests queue behind the warm copy instead of spuriously reloading elsewhere. Server-side local prompt expansion now honors the selected GPU set (and prefers an explicitly requested worker GPU when present), Qwen-Image offload budgeting reads the worker's real ordinal instead of GPU 0, and multi-GPU upscaling now routes through the pool instead of a process-global GPU-0 singleton. Disconnected queued jobs are skipped before expensive work begins, multi-GPU `/api/status` now reports real prompt hashes/timestamps for active generations, the TUI info panel reads per-GPU status, and `MoldClient` can target model unloads by GPU/model instead of only exposing the legacy global unload.
- **LTX-2 image-to-video no longer locks the first latent frame to a noisy ghost of the source image at `strength < 1.0`.** In `run_real_distilled_stage` (`crates/mold-inference/src/ltx2/runtime.rs`) the "clean reference" that the per-step denoise-mask blend pulls the conditioned tokens toward was sourced by cloning `video_latents` *after* `apply_stage_video_conditioning` had already soft-blended the first-latent-frame positions with the initial noise (`noise*(1-s) + source*s`). Used as the clean target, that pre-blended tensor pinned the first latent to a noisy copy of the image instead of the pure image at every step — so i2v runs with `--strength 0.75` (the CLI default) produced a first frame that was 25 % noise + 75 % image rather than the source image. A new helper `clean_latents_for_conditioning` re-applies the replacements with strength 1.0 on top of the post-apply tensor so replacement positions hold pure source image tokens while appended keyframe tokens and pure-noise regions pass through unchanged. `strength = 1.0` and pure-T2V paths are bit-for-bit identical to before. Covered by two new regression tests (`clean_latents_replace_soft_blended_positions_with_pure_source`, `clean_latents_passthrough_when_no_replacements`).
- **city96-format FLUX fine-tune GGUFs now fail with an honest, actionable error when no dev reference is downloaded, and surface the dependency at pull time instead of inside `ensure_gguf_embeddings`.** Community fine-tune GGUFs (e.g. the `silveroxides/ultrareal-fine-tune-GGUF` tree that powers `ultrareal-v4:q{8,5,4}`) ship only the diffusion blocks and expect the base FLUX input embedding layers (`img_in`, `time_in`, `vector_in`, `guidance_in`) to be patched in from a separately-downloaded flux-dev reference. Two bugs made this fail confusingly: (1) `find_flux_reference_gguf` in `crates/mold-inference/src/flux/pipeline.rs` returned the first candidate with `img_in.weight`, which let `flux-schnell:q8` pass the probe even though schnell is distilled without `guidance_in` — the subsequent patch loop bailed with `reference GGUF (.../flux-schnell-q8/flux1-schnell-Q8_0.gguf) is also missing required tensor 'guidance_in.in_layer.weight'`, making it look like schnell itself was broken. (2) The manifest didn't express the dependency at all, so the first indication a user had that `mold pull ultrareal-v4:q8` wasn't self-sufficient was an HTTP 500 on their first generation. Fixed by (a) adding a `needs_guidance: bool` parameter to `find_flux_reference_gguf` that skips schnell candidates for dev-family targets and verifies candidates contain `guidance_in.in_layer.weight` before accepting them, (b) rewriting both error messages so the source model is named and the reference path is shown as a filename rather than a full path, and (c) adding a pull-time probe in `crates/mold-core/src/download.rs` (`warn_if_flux_gguf_needs_reference`) that scans the first 4 MiB of any downloaded `.gguf` transformer for `img_in.weight`, and prints a one-line warning via the download callback when the GGUF is incomplete and no suitable dev reference is already on disk. Works for both the CLI (`pull_model`) and server (`pull_model_with_callback`) paths. New regression test `find_flux_reference_skips_schnell_when_dev_needed` covers the reference-picker behaviour.

- **Prompt expansion can no longer OOM on a multi-GPU box with a tight main GPU.** `LocalExpander` previously hardcoded `gpu_ordinal: 0` and gated placement with a static 2 GB VRAM threshold — on a dual-GPU system with a busy main card it fell back to CPU unnecessarily, and on a q8/bf16 expand model (4+ GB weights) the 2 GB threshold under-budgeted activations so the GPU placement check could pass and the load then OOM. The expander now sizes its budget dynamically (`model_size + 2 GB activations`, matching the T5/Qwen3 pattern) and cascades through devices: main GPU → remaining GPUs in ordinal order → CPU, with `preflight_memory_check()` as the final hard-fail guard when system RAM can't hold the model either. Unified-memory Metal placements also run the RAM preflight (Metal allocations draw from the same pool). Device selection logic is factored into a pure `select_expand_device(gpus, threshold, is_metal) -> ExpandPlacement` helper with unit tests for every branch.
- **Discord `/generate` model dropdown no longer fails with "Loading options failed".** The autocomplete handler previously called `cached_models()` which, on a stale cache, synchronously fetched `/api/models` from the mold server on the hot path. When the server was slow or cold (or when a `/generate` invocation happened before the first successful fetch), the round-trip regularly exceeded Discord's 3-second autocomplete budget and the client rendered "Loading options failed" instead of the model list. Fixed by: (1) spawning a background refresher at bot startup that keeps `model_cache` warm every 30 s, (2) making `cached_models()` a pure lock-read that never blocks on I/O, (3) falling back to the static manifest names (excluding utility families) when the cache is still cold so the dropdown renders immediately on first use, and (4) wrapping the autocomplete computation in a 1.5 s `tokio::time::timeout` as a last-ditch guard. `MoldClient` now derives `Clone` so the refresher task can own its own handle.

### Added

- **Render chain for arbitrary-length LTX-2 distilled video.** `mold run ltx-2-19b-distilled:fp8 "a cat walking" --frames 400` now produces a single stitched MP4 by splitting the request into multiple per-clip renders and carrying a motion-tail of latents across each clip boundary so the continuation stays coherent without a VAE encode/decode round-trip. New server endpoints `POST /api/generate/chain` and `POST /api/generate/chain/stream` (SSE) accept either a canonical `stages[]` body or an auto-expand form (`prompt` + `total_frames` + `clip_frames`) — the wire format is stages-based from day one so the v2 movie-maker UI can author per-stage prompts/keyframes without a breaking change. Request/response/event types live in `crates/mold-core/src/chain.rs` (`ChainRequest`, `ChainResponse`, `ChainProgressEvent`, `SseChainCompleteEvent`); the LTX-2 orchestrator is in `crates/mold-inference/src/ltx2/chain.rs` (`Ltx2ChainOrchestrator`, `ChainTail`); the server routes in `crates/mold-server/src/routes_chain.rs`; and the CLI side in `crates/mold-cli/src/commands/chain.rs`. `mold run` auto-routes to the chain endpoint when `--frames` exceeds the model's per-clip cap (97 for LTX-2 19B/22B distilled); non-distilled families fail fast with an actionable error instead of silently over-producing. New flags `--clip-frames N` (default = model cap) and `--motion-tail N` (default 4, 0 disables carryover) let advanced callers tune the split. The orchestrator derives per-stage seeds as `base_seed ^ ((stage_idx as u64) << 32)` so the whole chain reproduces from a single seed without identical-noise artefacts when every stage shares a prompt. Over-production at the final clip is trimmed from the tail (the head carries the user-anchored starting image and is perceptually load-bearing); mid-chain failures fail closed with HTTP 502 and no partial stitch is ever written to the gallery. Chains run on a single GPU — the chain handler bypasses the single-job queue and holds the `ModelCache` lock for the full chain duration (a multi-minute compound operation would otherwise stall the FIFO queue). Both the remote SSE path and the `--local` in-process path funnel through the same orchestrator via `Ltx2Engine::as_chain_renderer`, and `mold run` renders stacked `indicatif` progress bars (parent "Chain" frame counter + per-stage denoise-step bar). v1 is LTX-2 distilled only, single-GPU, and single-prompt; per-stage prompts, keyframes, selective regen, and multi-GPU stage fan-out are v2 movie-maker work.
- **In-browser model downloads with queued progress, ETA, cancel, and retry** ([#255](https://github.com/utensils/mold/pull/255)). `ModelPicker.vue` now shows `(X GB)` next to every model — click an undownloaded one to enqueue a pull without leaving the generate flow. A new `DownloadsDrawer` (opened from a TopBar button with an active/queued count badge) shows per-file progress, client-computed ETA (10 s sliding window), and cancel/retry controls. Undownloaded models in the picker switch to inline progress or a "Queued (#N)" chip while their job is alive, and the picker auto-refreshes on `JobDone` so the model becomes selectable without a page reload. Server-side: a new single-writer `DownloadQueue` in `AppState` drives the existing `mold_core::download::pull_model_with_callback` one model at a time (files sequential inside a set — HF's CDN is bandwidth-bound, so file-level parallelism would only trip rate limits), with one auto-retry on transient failure. Cancellation aborts the in-flight pull, cleans up partials under `MOLD_MODELS_DIR/<model>/` while preserving any `.sha256-verified` markers, and leaves the HF blob cache intact so resume is cheap. The same cleanup runs on terminal failures, not just cancel. New routes: `POST /api/downloads` (idempotent — returns the existing job id on a second enqueue), `DELETE /api/downloads/:id`, `GET /api/downloads` (active + queued + last 20 history), `GET /api/downloads/stream` (SSE multiplex of `DownloadEvent` frames — `Enqueued`, `Started`, `Progress`, `FileDone`, `JobDone`, `JobFailed`, `JobCancelled`). Existing `POST /api/models/pull` becomes a thin compat shim that enqueues via the queue and re-emits the legacy SSE event shape, so the TUI keeps working unchanged.
- **Always-visible VRAM + system RAM telemetry on `/generate`** ([#254](https://github.com/utensils/mold/pull/254)). A new `ResourceStrip.vue` docks at the bottom of the Composer sidebar on desktop (and collapses to a `🧠 used · total` chip in the TopBar on narrow viewports), showing one stacked-bar row per discovered GPU plus one for system RAM. Each row breaks usage into `mold` / `other` / `free` on CUDA hosts with per-process attribution (NVML feature-gated as `mold-ai-server` `--features nvml`, `nvidia-smi` subprocess fallback on by default) — on Metal the per-process fields are intentionally `None` and the SPA hides those breakdowns, since macOS doesn't expose per-process GPU attribution without private entitlements. Aggregated once per second on the server into a `ResourceSnapshot { hostname, gpus, system_ram }`, exposed as `GET /api/resources` (one-shot; `503` before the first aggregator tick) and `GET /api/resources/stream` (SSE broadcast with 15 s keepalive and the cached snapshot prepended as the first frame so new subscribers don't wait a full second). The aggregator handle is bound to `axum::serve`'s shutdown path so it's aborted on graceful exit. The strip's `useResources` composable is a provide/inject singleton mounted in `App.vue`, and it exposes a `gpuList: ComputedRef<GpuSnapshot[]>` that the new device-placement UI consumes directly.
- **Per-component device placement for FLUX, Flux.2, Z-Image, and Qwen-Image** ([#256](https://github.com/utensils/mold/pull/256)). A new `PlacementPanel` disclosure inside the Composer lets users override which device each part of the pipeline runs on. Tier 1 is a single "Text encoders: Auto / CPU / GPU N" dropdown that applies to every model family (SD1.5, SDXL, SD3.5, Wuerstchen, LTX-Video, LTX-2 in addition to the Tier 2 four) — picking CPU reliably moves the text encoder off-GPU so a large transformer can stay on-device without triggering block-level offload. Tier 2 adds per-component selects (transformer, VAE, and family-appropriate encoder slots) for FLUX, Flux.2, Z-Image, and Qwen-Image, where the plumbing is cheapest and the value is clearest. SD3.5 was marked stretch in the design and cut cleanly — the UI correctly hides Advanced for SD3.5 with a tooltip so no user sees an override that silently no-ops. A new `DevicePlacement` serde type (`DeviceRef = Auto | Cpu | Gpu(ordinal)` plus an optional `AdvancedPlacement` sub-struct for per-component overrides) rides as an optional field on `GenerateRequest`; `None` preserves the existing VRAM-aware auto-placement end-to-end. A shared `resolve_device()` helper in `mold_inference::device` (and a companion `effective_device_ref()` shared by the four Tier-2 engines) maps each `DeviceRef` variant to a `candle_core::Device`, returning a clean `anyhow::Error` for bad ordinals instead of panicking. Defaults are saved per-model in `[models."name:tag".placement]` (with `MOLD_PLACE_TEXT_ENCODERS`, `MOLD_PLACE_TRANSFORMER`, `MOLD_PLACE_VAE`, `MOLD_PLACE_CLIP_L`, `MOLD_PLACE_CLIP_G`, `MOLD_PLACE_T5`, `MOLD_PLACE_QWEN` env overrides) via a new `PUT /api/config/model/:name/placement` route (with `DELETE` to clear); the route now returns a real `500` when `Config::save()` fails instead of silently lying to the client. The placement UI reads its GPU list from `useResources().gpuList`, so spinning up a mold server on a dual-3090 box auto-populates "GPU 0 · RTX 3090" / "GPU 1 · RTX 3090" in every dropdown without any extra discovery wiring. `mold run` gains matching CLI flags — `--device-text-encoders`, `--device-transformer`, `--device-vae`, `--device-t5`, `--device-clip-l`, `--device-clip-g`, `--device-qwen` — which override env vars and config; flag parse errors surface with the specific flag name so `--device-vae banana` reports `--device-vae: invalid device 'banana' (expected auto|cpu|gpu[:N])` instead of a generic failure. Documented in `website/guide/configuration.md` (new "Per-component device placement" section) and `website/guide/performance.md` (the "CPU text encoders" subsection now points at the CLI flags for deliberate VRAM tuning).
- **Discord bot now supports video generation and img2img / img-to-video.** `/generate` accepts an optional `source_image` attachment (PNG/JPEG, ≤10 MiB) that is forwarded to the server as `source_image` — image-family models run img2img, LTX-2 runs image-to-video with the attachment as the first frame. New `video_format` choice (MP4 default, animated GIF) plus `frames`, `fps`, `audio` (LTX-2 only), `pipeline` (LTX-2 one-stage / two-stage / two-stage-hq / distilled / a2vid / retake), `strength`, and `negative_prompt` params wire through to the underlying `GenerateRequest`. Video families default to 25 frames @ 24 fps when unspecified. The handler picks up `GenerateResponse.video` and attaches the MP4/GIF bytes (plus a "Video Generated" embed with frame count, fps, and optional audio flag); when the primary MP4 exceeds Discord's ~24 MiB effective upload cap it automatically falls back to the always-generated GIF preview and notes the swap in the embed footer.

### Fixed

- **Discord bot no longer silently drops video-family generations.** Previously `/generate` hard-coded `output_format: Png` and `resp.images.first()` for the attachment, so LTX-Video / LTX-2 jobs either bounced off server validation (LTX-2 rejects non-video containers) or came back with the video missing from the reply.
- **Discord `/generate` now delivers LTX-Video / LTX-2 MP4s as a playable inline video instead of a static first-frame image.** Two bugs compounded to the same symptom. **Server side**: the multi-GPU SSE path (`gpu_worker::process_job`, used whenever `gpu_pool.worker_count() > 0` — i.e. every real GPU deploy) built the `complete` wire event by always base64-encoding the synthesized video-thumbnail `ImageData` as the payload and hard-coding every `video_*` field to `None`, so the client's `generate_stream` never saw `video_frames` / `video_fps` and fell through to treating the response as an image. The single-GPU `queue::process_job` branch had a correct video-aware branch, but that path is only taken on CPU-only hosts. Fixed by extracting a shared `build_sse_complete_event(&GenerateResponse, &ImageData) -> SseCompleteEvent` helper in `queue.rs` and routing both paths through it so the two can never drift on which `video_*` fields are populated. **Discord side**: the handler was calling `embed.attachment("mold-*.mp4")`, which writes `image.url = attachment://mold-*.mp4` into the embed. Discord's CDN always serves embed images through a WebP transcode pipeline (the response URL even appends `?format=webp&width=...&height=...`), so an MP4 "embedded" this way shows up as a static first-frame WebP with no play controls — regardless of how valid the MP4 bytes are on disk or in the CDN. Fixed by keeping the `embed.attachment(...)` reference only for image-shaped formats (PNG / JPEG / GIF / APNG / WebP — all of which render correctly inside an embed image), and leaving `.mp4` attachments unreferenced by the embed so Discord renders them as their own inline video-player block below the embed with the normal scrubber / play / volume controls.

## [0.9.0] - 2026-04-19

*Multi-GPU inference server, browser-driven generate UI, SQLite gallery metadata, and animated video previews.*

### Added

- **Multi-GPU inference server** ([#209](https://github.com/utensils/mold/issues/209), [#245](https://github.com/utensils/mold/pull/245)). Each GPU now gets a dedicated OS thread with its own `ModelCache`; a new `GpuPool` routes requests using idle-first + VRAM-fit placement, so two different models load onto two different cards and run concurrently. New CLI flags / env: `--gpus` / `MOLD_GPUS` (comma-separated ordinals or `all`) and `--queue-size` / `MOLD_QUEUE_SIZE` (bounded request queue, default 200, returns HTTP 503 + `Retry-After` on overflow). `GenerateResponse` now carries a `"gpu": <ordinal>` field, `/api/status` aggregates per-GPU worker state (with legacy single-GPU `model` / `gpu_info` fields still populated for back-compat), `mold ps` renders a multi-GPU table, and per-GPU `load` / `unload` / status endpoints are exposed. CUDA contexts are thread-local, so each worker owns a `std::thread::spawn` OS thread (not a tokio blocking-pool task); `reclaim_gpu_memory(ordinal)` resets a specific GPU's primary context inside that worker's `model_load_lock`. `GpuSelection` / `GpuWorkerStatus` added to `mold-core`, GPU discovery (`discover_gpus`, `filter_gpus`, `select_best_gpu`) and ordinal-aware `create_device` / `free_vram_bytes` added to `mold-inference`, and `gpu_ordinal` threaded through `EngineBase` + every per-family engine (FLUX, SD1.5, SDXL, SD3, Z-Image, Flux.2, Qwen-Image, Wuerstchen, LTX-Video). Spec: `docs/superpowers/specs/2026-04-08-multi-gpu-support-design.md`; plan: `docs/superpowers/plans/2026-04-08-multi-gpu-support.md`.
- **Browser-driven generation UI at `/generate`** ([#248](https://github.com/utensils/mold/pull/248)). The web SPA gains a new route with a minimal composer (Enter submits, Shift+Enter newline, empty Enter no-ops), per-GPU running-job cards that stream SSE progress (stage, denoise_step N/M, VAE decode), the existing gallery feed reused beneath, and an inline DetailDrawer when you click a tile so the URL stays on `/generate`. Fires multiple prompts in quick succession — concurrency is governed by server queue backpressure (503 `queue_full`); same-model parallel requests serialize on the GPU that already has that model loaded. img2img works via upload or a From Gallery picker, video-family models are grouped with a 🎬 badge and frames are clamped to 8n+1. Prompt expansion modal offers live preview + variation picker; surfaces server 422 cleanly when `qwen3-expand` isn't installed. `localStorage` persists prompt/model/size/steps/guidance/batch on reload (base64 `source_image` is stripped on persist). No server-side changes — consumes endpoints shipped with the multi-GPU work. New components: `Composer.vue`, `ExpandModal.vue`, `ImagePickerModal.vue`, `ModelPicker.vue`, `RunningJobCard.vue`, `RunningStrip.vue`, `SettingsModal.vue`; new composables: `useGenerateForm`, `useGenerateStream`, `useStatusPoll`; new libs: `sse.ts`, `base64.ts` (all with vitest coverage). Spec: `tasks/web-generate-ui-spec.md`; plan: `tasks/web-generate-ui-plan.md`.
- **`GET /api/gallery/preview/:filename` streams cached animated GIF previews for video gallery entries** ([#242](https://github.com/utensils/mold/issues/242)). Both the legacy single-GPU save path (`queue::process_job`) and the multi-GPU worker save path (`gpu_worker::process_job`) now persist each generated video's `gif_preview` sidecar to `MOLD_HOME/cache/previews/<filename>.preview.gif` alongside the primary MP4, and the new endpoint streams it back as `image/gif` (404 when the source MP4 has been deleted, no preview was generated, or the file simply isn't a video). The lookup requires the underlying gallery file to still exist, so a stale sidecar can't leak deleted content; the delete handler also removes the sidecar in lock-step with the primary file. The TUI's server-backed gallery detail pane tries this endpoint first for MP4 / WebM / MOV / MKV entries and installs the decoded frames through the existing animation pipeline; on 404 it falls back to the PNG thumbnail (which the thumbnail endpoint already generates via openh264 first-frame extraction) so preview-less videos still show a still frame rather than a blank pane. Prior behavior was: remote video entries downloaded the raw MP4 via `/api/gallery/image/:filename`, `image::open` failed to decode it, and the `Preview` pane sat on `Loading…` forever. New `MoldClient::get_gallery_preview(filename)` helper returns `Option<Vec<u8>>` so 404 is a normal signal rather than an error.
- **SQLite gallery metadata DB at `MOLD_HOME/mold.db`**: every saved generation — from both the CLI's local path and the HTTP server — now writes a row with prompt, negative prompt, model, seed, steps, guidance, dimensions, LoRA, scheduler, file mtime/size, generation duration, hostname, backend (`cuda`/`metal`/`cpu`), and a `source` column (`server` / `cli` / `backfill`). Embedded `mold:parameters` chunks in PNG/JPEG outputs continue to work — the DB is additive. New crate `mold-db` (rusqlite, bundled SQLite, WAL mode). Server boots an async reconciliation pass that imports existing files into the DB on first run, refreshes rows whose mtime/size diverge, and prunes rows for files removed out-of-band. `/api/gallery` queries the DB first (with the existing filesystem walk as a fallback) so listings stay fast on large directories and surface metadata for formats that don't embed it (mp4, gif, webp, apng). `DELETE /api/gallery/image/:filename` removes the matching row. The TUI's local gallery scan and the web gallery API both pick up the DB data automatically. Opt out with `MOLD_DB_DISABLE=1`; override location with `MOLD_DB_PATH`. NixOS module exposes `services.mold.metadataDb.{enable,path}`.
- **Animated video playback in TUI gallery detail and CLI `--preview`**: GIF/APNG/animated-WebP previews now play their full frame sequence instead of freezing on the first frame. The TUI gallery detail view, generation viewport, and CLI `--preview` all decode every frame up front, advance on each frame's recorded delay (clamped to ≥20 ms / ≤50 fps), and loop. The CLI replays short clips (<1.5 s gets 3 plays, <3 s gets 2, longer plays once) using ANSI cursor save/restore so each frame overwrites the previous in place. Decoder lives in `crates/mold-tui/src/animation.rs` and is shared by the gallery preview path and the post-generation video preview ([#179](https://github.com/utensils/mold/issues/179)).

### Fixed

- **Multi-GPU dispatcher no longer parks new jobs behind a busy GPU while a sibling sits idle.** During an in-flight generation the worker thread calls `cache.take()`, which removes the cache entry for the duration of inference — so `cache.active_model()` and `cache.get(model).residency == Gpu` both returned None/false. That made a busy GPU mid-inference look identical to a truly empty idle GPU in `GpuPool::select_worker`'s classifier, and a brand-new job for a *different* (smaller) model could get routed to the busy card's 2-slot channel instead of the genuinely free GPU. `select_worker` now also checks `in_flight > 0` (set by the dispatcher before send) and `active_generation.is_some()` (set by the worker around the take-and-restore window); either one demotes the worker out of the `idle_empty` tier. Adds four `gpu_pool::tests` cases covering the cache-empty-but-busy window, the active-generation flag, the idle-spread happy path, and the all-busy headroom fallback.
- **Remote CUDA OOMs no longer mislabeled as "Metal out of memory" on macOS clients** ([#241](https://github.com/utensils/mold/issues/241)). Previously `mold run --batch N` against a remote CUDA server would report `Metal out of memory` with local-only hints (`--width 512 --height 512`, "source image resolution is used by default") whenever the server's GPU OOM'd, because the error formatter keyed off `cfg!(target_os = "macos")` rather than where the generation actually ran. Errors originating from the remote API path are now tagged with the server host via a new `RemoteInferenceError` wrapper; the top-level handler downcasts to that tag before labeling the error and picks hints that make sense for the remote context (reduce `--batch`, use a smaller model, ask the operator to `mold unload`, fall back to `--local`). Local Metal / CUDA OOM paths keep their existing behavior. (Issue also notes a separate server-side VRAM-growth suspicion on Qwen-Image-Edit between batch iterations; that is tracked independently.)

### Changed

- **Gallery metadata DB now canonicalizes `output_dir` before keying rows** so the CLI (which canonicalized via `std::fs::canonicalize`) and the server (which used the raw `config.effective_output_dir()` value) can no longer produce two rows for the same underlying file. On macOS `/tmp/foo` and `/private/tmp/foo` collapse to one `UNIQUE(output_dir, filename)` key, so the gallery no longer shows duplicates when both surfaces share a single `MOLD_HOME/mold.db`. A v2 data migration rewrites existing v1 rows to the canonical form on first open so upgrades from unreleased-feature builds don't leave orphaned legacy-keyed rows behind.
- **Metadata DB now uses a real forward-only migration framework** instead of a monolithic `CREATE TABLE IF NOT EXISTS` block. Schema version tracked via SQLite's `PRAGMA user_version`, migrations applied inside a transaction, partial failures roll back the version bump. Adds `MetadataDb::schema_version()` and public `mold_db::SCHEMA_VERSION` so operators can assert what schema level they're on; future column adds (controlnet, source-image hash, etc.) drop into `MIGRATIONS` as an `ALTER TABLE ADD COLUMN`.
- **Metadata DB verifies WAL mode actually took** instead of silently swallowing the pragma return. Filesystems that reject WAL (tmpfs, certain network mounts) now log a warning so operators know concurrent writers will block longer than expected. Other pragma failures are logged rather than `.ok()`-dropped.

## [0.8.1] - 2026-04-17

*Single-binary web gallery: SPA is now embedded at compile time.*

### Fixed

- **`nix build github:utensils/mold` now delivers a single binary that serves the web gallery UI** ([#239](https://github.com/utensils/mold/issues/239)). Previously the flake produced a binary that always fell through to the inline "web gallery UI isn't installed" placeholder because the flake never built `web/dist/` or staged it anywhere the runtime resolver looked. Fixed by:
  - Adding a `mold-web` derivation built via [`bun2nix`](https://github.com/nix-community/bun2nix) that reproducibly compiles the Vue 3 SPA from `web/bun.lock` → `web/bun.nix` (committed).
  - Adding `crates/mold-server/build.rs` that stages the built `dist/` (from `$MOLD_WEB_DIST`, a repo-relative `web/dist`, or a generated stub) and stamps `MOLD_EMBED_WEB_DIR` so `rust-embed` bakes the assets into the final binary.
  - Rewriting `crates/mold-server/src/web_ui.rs` to serve the embedded bundle as the axum fallback, with content-type / ETag / immutable-cache headers for hashed assets and SPA deep-link fallback to `index.html`. `$MOLD_WEB_DIR` (and the other legacy filesystem candidates) still take precedence so developers can hot-reload the SPA without recompiling Rust.
  - Dropping the never-wired `share/mold/web` copy path from `mkMold`'s `postInstall` — `$out/bin/mold` is now truly self-contained.

## [0.8.0] - 2026-04-17

*Native RunPod CLI, web gallery UI, and hardening across the board.*

### Added

- **Native RunPod support via `mold runpod` subcommand tree**: manage RunPod cloud GPU pods end-to-end from the same `mold` binary. `mold runpod run "<prompt>"` creates a pod with smart defaults (cheapest GPU with enough VRAM for the requested model, matching GHCR image tag, automatic DC fallback when scheduling stalls), waits for the mold server to boot, streams generation progress over SSE, and saves the result to `./mold-outputs/`. Full subcommand surface: `doctor`, `gpus`, `datacenters`, `list`, `get`, `create`, `stop`, `start`, `delete`, `connect`, `logs`, `usage`, `run`. Adds a `[runpod]` config section (`api_key`, `default_gpu`, `default_datacenter`, `default_network_volume_id`, `auto_teardown`, `auto_teardown_idle_mins`, `cost_alert_usd`, `endpoint`), `RUNPOD_API_KEY` env precedence, dynamic shell completion for pod/gpu/datacenter/cloud-type arguments, and spend history in `~/.mold/runpod-history.jsonl`. NixOS module gains a `runpodApiKeyFile` option that mirrors the existing `hfTokenFile`/`apiKeyFile` secret-loader pattern.
- **Agent-safe `mold runpod`**: all runpod subcommands are non-interactive. `delete` no longer prompts for confirmation — passing an explicit pod id is enough signal — so scripts and AI agents can drive the full pod lifecycle without stdin.
- **SIGINT-safe pod provisioning**: Ctrl-C during pod scheduling now deletes the in-flight pod before exiting instead of orphaning billing. Stale-pod sweep on `mold runpod run` cleans up prior Ctrl-C survivors.
- **Model-size-aware GPU auto-selection**: `mold runpod run --model <model>` reads the manifest and picks the cheapest GPU with ≥1.8× model weights in VRAM. LTX-2.3 22B FP8 (29 GB) now selects a 5090/L40/H100 automatically instead of overflowing a 24 GB 4090.
- **Smart HF_TOKEN resolution**: gated manifests auto-enable `HF_TOKEN` passthrough in `mold runpod run` without `--hf-token`. Token priority is local `HF_TOKEN` env → RunPod secret template — the local-env path guarantees the pod can pull gated weights even when the RunPod account has no `HF_TOKEN` secret configured. Plan output shows which source fed the token.
- **Auto-MP4 output for video models in `mold runpod run`**: LTX-2 and LTX-Video families now default to `OutputFormat::Mp4`, fixing the `"LTX-2 outputs must use mp4, gif, apng, or webp"` validation failure; image models still default to PNG.
- **Network-volume-aware datacenter pinning**: when `--network-volume` or `runpod.default_network_volume_id` is set, the pod is automatically pinned to the volume's `data_center_id`. The retry loop preserves the volume pin so every attempt targets the only region that can mount the volume.
- **`runpodctl` in Nix devshell**: contributors get the official RunPod CLI on their `PATH` without manual install.
- **Web gallery bundled into the RunPod container**: the `ghcr.io/utensils/mold` Docker images now build `web/dist` in a dedicated `bun` stage and ship it at `/opt/mold/web`, with `MOLD_WEB_DIR` pre-set so `mold serve` picks it up as the SPA fallback. Visiting `https://<pod-id>-7680.proxy.runpod.net/` opens the Vue 3 gallery directly — no manual `MOLD_WEB_DIR` or volume copy required. `docker/start.sh` prints the gallery + API-docs URLs in its boot banner when `RUNPOD_POD_ID` is present, and both `mold runpod connect <id>` and `mold runpod run` now surface the browsable URL alongside the existing `export MOLD_HOST=…` line.

- **Web gallery UI**: the `mold serve` binary now ships with a Vue 3 + Vite + Tailwind CSS v4.2 single-page app that renders the on-disk gallery as a mobile-first **Tumblr-style feed** (default) or dense **masonry grid** (2 cols on mobile → 6 cols on xl), toggleable from the header and persisted in `localStorage`. The SPA is served as the axum fallback, so `/api/*`, `/health`, and `/metrics` keep working unchanged. Bundle lives in `web/` (`bun install && bun run build`); the server resolves it from `MOLD_WEB_DIR`, `$XDG_DATA_HOME/mold/web`, `~/.mold/web`, a `web/` dir next to the binary, or `web/dist` (for `cargo run` in the repo). Features: IntersectionObserver-driven lazy media and chunked rendering (40 feed / 150 grid per page), thumbnails in grid with `thumbnail → full → broken-tile` fallback chain, full-resolution edge-to-edge media in feed on mobile, videos autoplay muted while on-screen and pause when scrolled off, global sound toggle (header speaker button; first click satisfies the browser gesture requirement for unmuted autoplay), prompt/model/filename search with 180 ms debounce, All / Images / Video filter pills, fullscreen swipe-navigable detail drawer on mobile (swipe down → next, swipe up → prev, horizontal drift > 80 px cancels) with tucked-away metadata in a bottom sheet, desktop side-by-side drawer with always-visible metadata sidebar, keyboard nav (Esc / ← / → / ↑ / ↓ / j / k / i), prompt + seed copy-to-clipboard, download. Uses the mold transparent logo at `web/public/logo.png` (mirrored from the docs site, also favicon + apple-touch-icon). Non-sticky mobile header with a back-to-top FAB for long feeds.
- **Gallery API: all output formats + richer item metadata**: `GET /api/gallery` now returns `png`, `jpeg`, `gif`, `apng`, `webp`, and `mp4` outputs (previously png/jpeg only). Each `GalleryImage` gains optional `format`, `size_bytes`, and `metadata_synthetic` fields. Files without embedded `mold:parameters` chunks (typically videos) get a synthesized `OutputMetadata` whose `model` is parsed from the `mold-<model>-<unix>[-<idx>].<ext>` filename; real `width`/`height` are backfilled from the decoded image header when possible. `GET /api/gallery/image/:filename` serves the full matrix of content types with streaming responses and **supports HTTP `Range` requests** so `<video>` scrubbing works (single-range form, `bytes=start-end`, `bytes=start-`, `bytes=-N`; unsatisfiable ranges return `416` per RFC 9110). `GET /api/gallery/thumbnail/:filename` renders first-frame thumbnails for all raster formats via the `image` crate, extracts real first-frame PNG thumbnails for MP4 via `mold_inference::ltx2::media::extract_thumbnail` (openh264), and falls back to source bytes (raster) or an inline SVG play-icon (mp4) on decode failure. Cached at `~/.mold/cache/thumbnails/<name>.png` regardless of source format.
- **Gallery validation at scan time**: `scan_gallery_dir` now filters out corrupt / stub outputs up front rather than surfacing them as broken tiles. A format-specific size floor (128 B for GIF, 256 B for other raster formats, 4 KB for MP4) skips truncated writes; a fast header-only decode (or `ftyp`-box check for MP4) catches files with invalid signatures; and a **solid-black heuristic** decodes suspect-size raster files to a 16×16 sample and drops them when every channel is below a ~6 % intensity ceiling — catching aborted / NaN-poisoned outputs that wrote all-zero tensors. In the mold dev harness this drops the listing from 1507 → ~290 items.
- **Gallery delete is opt-in**: `DELETE /api/gallery/image/:filename` now returns `403 Forbidden` by default. Operators enable destructive writes with `MOLD_GALLERY_ALLOW_DELETE=1` (combined with the existing API-key middleware when the server is exposed beyond localhost). A new `GET /api/capabilities` endpoint returns `{ "gallery": { "can_delete": bool } }` so clients can hide the UI affordance instead of inviting a 403 — the SPA uses this to conditionally render the delete button in the detail drawer.

### Fixed

- **`mold runpod run` warm-pod reuse now respects `--gpu` and `--dc`**: an existing warm pod on a different GPU/datacenter is deleted and replaced instead of silently reused, so explicit overrides can't be ignored by a stale `runpod-state.json`.
- **Warm-pod reuse no longer deletes healthy pods when REST v1 omits `runtime`/`gpuDisplayName`**: readiness is now detected by probing the RunPod proxy directly. If `desired_status=RUNNING` and the proxy responds, the pod is reused; otherwise it is deleted.
- **GraphQL endpoint override honors `runpod.endpoint`**: `doctor`, `gpus`, `datacenters`, and `usage` now route through the configured endpoint when set (previously only REST calls did), so mock/staging setups exercise the same GraphQL path.
- **Manually-created pods are no longer subject to the idle reaper**: `mold runpod create` used to enroll the new pod as `last_pod_id`, letting the 20-minute idle reaper delete it during a subsequent invocation. Create-made pods are now tracked only in history and deleted only via explicit `mold runpod delete`.
- **LTX-2.3 camera-control preset validation**: `--camera-control` preset aliases (`dolly-in`, `dolly-left`, `dolly-out`, `dolly-right`, `jib-down`, `jib-up`, `static`) now fail locally at the CLI layer with an explicit "Lightricks has not released camera-control LoRAs for LTX-2.3 yet" message when paired with an LTX-2.3 model, instead of failing server-side after the HTTP round-trip. Explicit `.safetensors` paths still work for LTX-2.3. `--camera-control` help text now documents the 19B-only preset limitation ([#227](https://github.com/utensils/mold/issues/227)).
- **Remote pull progress bars dropped file names on completion**: completed download progress bars showed `done` as the prefix instead of the file name (e.g. `[1/20] config.json`), so only the in-flight file was identifiable. Completed bars now keep their `[i/N] <filename>` label both during and after download ([#223](https://github.com/utensils/mold/issues/223)).
- **`.dockerignore` was dropping `web/public/logo.png`**: the blanket `*.png` / `*.jpeg` / `*.jpg` patterns excluded the SPA logo + favicon from the Docker build context, shipping a broken web gallery. Added `!web/public/**` / `!web/src/**` negations so the bundler stage gets the assets it needs.
- **Docker image build resilient to transient apt flakes**: both the builder and runtime `apt-get install` steps now retry up to 5 times with exponential backoff. Main-branch build `24552477974` failed on an `archive.ubuntu.com` timeout fetching `tini`; future builds ride out similar mirror outages instead of hard-failing CI.


## [0.7.1] - 2026-04-16

### Fixed

- **Video outputs saved as single-frame PNGs instead of real mp4/gif/apng files**: both the SSE and non-SSE HTTP transport paths between server and client discarded `VideoData`, causing the CLI to save a first-frame PNG thumbnail with the requested video extension. The SSE `SseCompleteEvent` now carries optional video metadata fields (`video_frames`, `video_fps`, `video_thumbnail`, `video_gif_preview`, audio metadata) and the non-SSE path sends `x-mold-video-*` response headers so the client can reconstruct the full `VideoData` in both code paths ([#224](https://github.com/utensils/mold/issues/224)).

## [0.7.0] - 2026-04-14

*Native Rust LTX-2 / LTX-2.3 joint audio-video generation.*

### Added

- **LTX-2 / LTX-2.3 joint audio-video generation**: added the new `ltx2` model family with `ltx-2-19b-{dev,distilled}:fp8` and `ltx-2.3-22b-{dev,distilled}:fp8` manifests, synchronized MP4-first video metadata, request fields for audio/video/keyframes/retake/upscaling, and a separate `Ltx2Engine` wired into the inference factory through the in-tree Rust runtime.
- **LTX-2 CLI surface**: added `--audio`, `--no-audio`, `--audio-file`, `--video`, repeatable `--keyframe`, repeatable `--lora`, `--pipeline`, `--retake`, `--camera-control`, `--spatial-upscale`, and `--temporal-upscale` to `mold run`.
- **LTX-2 native review utility**: added a pure-Rust MP4 review path that decodes native LTX-2 smoke clips and writes GIF previews plus contact-sheet PNGs without relying on ffmpeg, shellouts, or Python tooling.
- **LTX-2 devshell helpers**: the flake devshell now includes the native LTX-2 workflow helpers `build-ltx2`, `test-ltx2`, `smoke-ltx2`, and `contact-sheet`, plus the repo tools needed to run them.

### Changed

- **Video defaults for `ltx2`**: LTX-2 requests now default to MP4 output instead of PNG/APNG and strip audio automatically when exporting GIF/APNG/WebP.
- **Manifest plumbing**: `ModelPaths`, manifests, validation, and `mold info` now understand temporal upscalers and distilled LoRAs.
- **LTX-2 native upscaling path**: temporal `x2` upscaling now reaches the native Rust runtime, and the stage-1 render plan derives lower-resolution/lower-fps shapes before native spatial and temporal upsampling restore the requested output dimensions.
- **LTX-2 operator docs**: README, website docs, and the shared mold skill now describe the completed native Rust workflow matrix instead of the earlier partial-acceptance state.
- **LTX-2 CLI plumbing**: the internal `mold run` video-generation call path now bundles LTX-2-specific knobs into a dedicated `Ltx2Options` struct instead of threading another long positional argument list through `generate::run`.
- **LTX-2 developer binaries**: `ltx2_review`, `ltx2_checkpoint_probe`, and `ltx2_vae_probe` now build only when `mold-ai-inference` is compiled with `--features dev-bins`, so normal workspace and CI builds no longer compile those helper binaries implicitly.

### Fixed

- **LTX-2 manifest accounting**: model-size and download-path resolution now treat the single-file LTX-2 checkpoints correctly without requiring a standalone VAE asset.
- **LTX-2 camera-control presets**: `--camera-control dolly-in|dolly-left|dolly-out|dolly-right|jib-down|jib-up|static` now resolves the published LTX-2 19B camera LoRAs instead of failing validation.
- **LTX-2 local Ada runtime**: local 24 GB FP8 runs now stay on the native Rust path and the compatible `fp8-cast` mode instead of Hopper-only `fp8-scaled-mm`, avoiding the TensorRT-LLM dependency.
- **LTX-2 native FP8 prompt/runtime path**: the native Gemma and embeddings stack now streams decoder layers, normalizes BF16/F32 CPU inspection paths, and materializes contiguous connector attention tensors so local CUDA smoke clips can complete without bridge fallbacks or prompt-path dtype/layout failures.
- **LTX-2.3 22B native coherence**: the native FP8 path now applies checkpoint `weight_scale` correctly, matches the embedded 22B connector/VAE layout more closely, and resizes decoded frames to the requested output size so CUDA smoke clips render coherent motion instead of random-color collapse.
- **LTX-2 CLI LoRA aliases**: `mold run --lora camera-control:<preset>` now reaches the native LTX-2 resolver instead of failing early in generic file-path validation, so `ic-lora` smoke requests can use published camera-control aliases from the CLI.
- **LTX-2.3 x1.5 spatial upscale requests**: `--spatial-upscale x1.5` now passes validation and resolves the published `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` asset on demand instead of failing before the request reaches the engine.
- **LTX-2 temporal upscale requests**: `--temporal-upscale x2` now passes validation, resolves the configured temporal upsampler asset, and executes the native temporal interpolation path instead of failing as unimplemented.
- **LTX-2 native acceptance closure**: the public native Rust CUDA workflow matrix is now validated across 19B/22B text+audio-video, image-to-video, audio-to-video, keyframe, retake, public IC-LoRA, spatial upscale (`x1.5` / `x2` where published), and temporal upscale (`x2`).
- **LTX-2 request validation**: megapixel-limit errors now report the current `1.8MP` ceiling, current LTX frame-grid validation is scoped to the LTX families that require `8n+1`, unknown-family errors for LTX-2-only request fields are clearer, and oversized inline `audio_file` / `source_video` payloads now fail fast with a `64 MiB` limit.

## [0.6.3] - 2026-04-08

*Qwen-Image-Edit-2511 support and TUI remote server awareness.*

### Added

- **Qwen-Image-Edit-2511 request surface**: added `qwen-image-edit-2511:{bf16,q8,q6,q5,q4,q3,q2}` manifests, `GenerateRequest.edit_images`, family-aware validation, and distinct shared storage paths for edit-family VAE/text-encoder assets (#219)
- **Qwen-Image-Edit-2511 inference pipeline**: added the Qwen2.5-VL multimodal edit encoder, condition-image preprocessing, packed edit-latent concatenation with `img_shapes`, `zero_cond_t` transformer support, and true-CFG norm rescaling for local image editing (#219)

### Changed

- **`--image` CLI semantics**: `mold run --image` is now repeatable. Non-edit families still accept at most one source image; `qwen-image-edit` maps repeated `--image` flags into `edit_images` (#219)
- **TUI capability modeling**: `qwen-image-edit` now appears as a source-image editing family instead of img2img, so the TUI exposes a source image and negative prompt without img2img-only controls like `strength`, `mask`, `ControlNet`, or `LoRA` (#219)

### Fixed

- **TUI remote server awareness**: the Info panel, model defaults, and model management now reflect the connected server instead of the local machine (#218, [#158](https://github.com/utensils/mold/issues/158)):
  - Info panel queries `/api/status` for memory, GPU, and busy state when connected to a remote server
  - Model parameter defaults (steps, guidance, width, height) come from the server's catalog instead of local `config.toml`
  - Model pull routes through the server API when connected remotely
  - Reset Defaults action uses server catalog values when connected
  - Tab bar shows hostname (e.g. "hal9000") when connected to a remote server, "local" in local mode, "connecting..." during health check
  - `/api/status` now includes `hostname` and `memory_status` fields (backward-compatible, older servers omit them)
- **Qwen edit-family encoder selection**: `qwen-image-edit` now supports quantized `--qwen2-variant` values by splitting the multimodal stack into a GGUF Qwen2.5 language path plus a safetensor-backed Qwen2.5-VL vision tower for image conditioning (#219)
- **Qwen edit-family RAM use**: `qwen-image-edit` now stages the Qwen2.5 encoder instead of keeping it resident after model load, drops encoder weights immediately after edit conditioning, and avoids loading the full BF16 language stack just to run multimodal edits. A local `qwen-image-edit-2511:q4 --qwen2-variant q4` smoke run completed with roughly 1.8 GB max RSS instead of the earlier tens-of-GB host spike (#219)
- **Qwen2.5 CPU dtype handling**: CPU Qwen2.5 encoder paths now stay in `F32` where candle CPU matmul requires it, while lower-memory edit inference uses quantized GGUF language weights plus the staged vision sidecar instead of relying on unsupported CPU `BF16` matmuls (#219)
- **Qwen2.5 vision progress reporting**: the staged vision-tower load now reports only the bytes for `visual.*` tensors instead of the full shared text-encoder shard set, removing the misleading `15.45 GiB` progress line during edit encoder reloads (#219)

## [0.6.2] - 2026-04-08

### Added

- **`mold update` command**: self-update from GitHub releases with SHA-256 checksum verification, platform auto-detection (macOS Metal, Linux CUDA sm89/sm120), atomic binary replacement with rollback, and package manager detection (Nix, Homebrew). Supports `--check`, `--force`, and `--version` flags.
- **img2img and inpainting for all remaining model families**: SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image, and Wuerstchen v2 now support `--image` for img2img and `--mask` for inpainting. Includes VAE encoder implementations for Flux.2 (BN-VAE with patchified BatchNorm) and Qwen-Image (3D causal VAE specialized to 2D via temporal slice extraction). Wuerstchen uses VQ-GAN encoding with Prior bypass for img2img. All families support strength-based schedule trimming and per-step inpainting blending. ([#174](https://github.com/utensils/mold/issues/174))

### Fixed

- **Img2img denoising strength alignment**: all model families now map `--strength` to scheduler start steps using reference img2img semantics instead of treating the user value as a raw flow sigma. This restores expected source-image retention for low-strength edits and removes washed-out flow-model img2img results.
- **Img2img VAE source normalization**: Flux.2, Qwen-Image, and SD3 img2img now encode source images in the autoencoder's expected `[-1, 1]` pixel range, eliminating the white, foggy washout that appeared even at `--strength 0.0`.
- **Deterministic img2img VAE encodes**: Flux.2, Qwen-Image, SD1.5, SDXL, and SD3 now use posterior-mean latents when encoding source images, removing unseeded VAE sampling noise so repeated img2img runs with the same source image and seed stay reproducible.
- **`--strength 0` validation**: img2img requests now accept `strength = 0.0` as documented, preserving the source image with zero denoise instead of rejecting the request.
- **Z-Image GGUF inference**: `z-image-turbo:q4/q6/q8` now load GGUF weights into the standard dense Z-Image transformer instead of the broken custom quantized transformer path, restoring coherent txt2img output and fixing sequential img2img by encoding source images before the transformer is loaded.

## [0.6.1] - 2026-04-07

### Added

- **TUI UAT harness**: native Ghostty 1.3+ AppleScript-based acceptance testing harness (`scripts/tui-uat.sh`) with launch, capture, send keystrokes, view navigation, screenshot, and assert commands — pixel-perfect terminal screenshots via `screencapture -l<windowID>` ([#173](https://github.com/utensils/mold/issues/173))

### Changed

- **Qwen-Image CUDA execution planning**: local sequential runs now preserve the stable load-use-drop path, while hot server mode keeps more of the quantized CUDA stack resident, skips redundant prompt re-encoding on cache hits, and splits CFG passes when batching would overrun VRAM

### Fixed

- **CLI Ghostty preview regression**: `mold run --preview` and `mold upscale --preview` now bypass `viuer`'s leaking Kitty capability probe in Ghostty without falling back to blocky half-block rendering, preserving crisp inline previews while preventing visible `^[_Gi=31;OK...` control text
- **Qwen-Image-2512 CUDA black images / OOMs**: quantized CUDA generation now uses a staged fallback ladder for VAE decode (full GPU, tiled GPU, then CPU reload), keeps the quantized transformer resident across decode when that is safe, and aligns BF16/GGUF text masking behavior with upstream Qwen
- **Server startup thumbnail warmup**: gallery thumbnail generation is now opt-in via `MOLD_THUMBNAIL_WARMUP=1`, preventing avoidable CPU/RAM spikes on startup when large or bad output directories are present

## [0.6.0] - 2026-04-07

### Added

- **LTX Video — text-to-video generation**: first video model family in mold. Generate animated video clips from text prompts using LTX 0.9.6 and 0.9.8 checkpoints (2B and 13B variants) with APNG, GIF, WebP, and MP4 output formats ([#172](https://github.com/utensils/mold/issues/172))
- **Video CLI flags**: `--frames <N>` (must be 8n+1), `--fps <N>` (default 24), `--format apng|gif|webp|mp4` for video models
- **Image upscaling**: `mold upscale <image>` with 7 Real-ESRGAN variants (2x/4x, RRDBNet + SRVGGNetCompact), tiled processing for large images, server API (`POST /api/upscale`), and TUI gallery integration with background upscaling ([#170](https://github.com/utensils/mold/issues/170))
- **Qwen-Image enhancements**: FP8 support, block-level offloading for 24GB cards, negative prompt support, full Q2-Q8 GGUF tier coverage for both base and 2512 variants, and encoder variant selection via `--qwen2-variant` / `--qwen2-text-encoder-mode` CLI flags ([#178](https://github.com/utensils/mold/issues/178))
- **Prometheus metrics**: `GET /metrics` endpoint with HTTP rates, generation duration, queue depth, GPU memory, and uptime (behind `metrics` feature flag) ([#142](https://github.com/utensils/mold/issues/142))
- **Custom MP4 muxer**: minimal QuickTime-compatible H.264 writer with faststart layout, replacing muxide dependency ([#181](https://github.com/utensils/mold/issues/181))
- **CI feature matrix**: new `check-features` job validates all optional feature combinations
- **GIF preview cache**: animated GIF previews cached in `~/.mold/cache/previews/` for TUI gallery

### Changed

- **LTX 0.9.8 multiscale refinement**: 0.9.8 checkpoints now run full two-pass generation with spatial upsampler and continuation window ([#201](https://github.com/utensils/mold/issues/201))
- **LTX model catalog**: replaced legacy 0.9/0.9.5 manifests with 0.9.6, 0.9.6-distilled, 0.9.8-2b-distilled, 0.9.8-13b-dev, and 0.9.8-13b-distilled
- **LTX shared asset layout**: canonical paths now match upstream sources; legacy installs auto-migrate on pull ([#204](https://github.com/utensils/mold/issues/204))
- **Qwen-Image model sources**: base GGUF models now map to `Qwen/Qwen-Image` + `city96/Qwen-Image-gguf`; 2512 variants use `unsloth/Qwen-Image-2512-GGUF` ([#178](https://github.com/utensils/mold/issues/178))
- **Batch limit removed**: `--batch` no longer capped at 16
- **Nine model families**: `LtxVideoEngine` added to engine factory; `UpscaleEngine` trait added for upscaler models
- **CI consolidated**: merged check/clippy/test into single job with shared rust-cache

### Fixed

- **Qwen-Image CUDA black images**: Metal GGUF denoising optimization (#207) replaced per-forward BF16 dequantization with QMatMul/F32 globally, breaking CUDA inference. Fixed with device-gated dispatch — BF16 dequant on CUDA, QMatMul on Metal, F32 on CPU ([#207](https://github.com/utensils/mold/pull/207))
- **Qwen-Image Metal performance**: quantized GGUF transformer uses Candle's quantized linear path and caches RoPE tensors, improving Q4 denoising from ~0.03 it/s to ~0.15 it/s on M4 Max ([#202](https://github.com/utensils/mold/issues/202))
- **Qwen-Image GGUF prompt adherence**: restored combined padding + causal attention masking in Qwen2 text encoder ([#178](https://github.com/utensils/mold/issues/178))
- **Qwen-Image Metal text encoding**: `auto` now prefers quantized GGUF encoders on Metal/MPS
- **Stale pull markers**: manifest-backed models self-heal stale `.pulling` markers; partial installs auto-repair on use
- **Upscaler/utility model listing**: `mold list` correctly shows non-diffusion models as installed ([#184](https://github.com/utensils/mold/issues/184))
- **LTX inference quality**: versioned transformer/VAE presets with `decode_noise_scale` for timestep-conditioned VAE decode
- **MP4 QuickTime compatibility**: correct ftyp brands, colr/pasp atoms, BT.601 range alignment ([#181](https://github.com/utensils/mold/issues/181))
- **Server video handling**: queue worker and CLI batch loop no longer discard video responses
- **Test isolation**: config/cleanup tests use temp `MOLD_HOME` to avoid touching live model caches

### Removed

- **Legacy LTX manifests**: removed `ltx-video-0.9:bf16` and `ltx-video-0.9.5:bf16`

## [0.5.3] - 2026-04-04

### Added

- **Qwen3-8B encoder support**: Klein-9B models now use the correct Qwen3-8B text encoder (hidden_size=4096) with auto-selection between 4B and 8B GGUF variant registries ([#157](https://github.com/utensils/mold/issues/157), [#166](https://github.com/utensils/mold/pull/166))
- **Qwen3-8B GGUF variant registry**: Q8/Q6/IQ4/Q3 quantized encoders from `unsloth/Qwen3-8B-GGUF` for Klein-9B, with separate cache directory (`shared/qwen3-8b-gguf`) ([#166](https://github.com/utensils/mold/pull/166))
- **Gallery images**: Klein-9B bottle ship and Wuerstchen v2 lighthouse added to homepage gallery and model pages ([#166](https://github.com/utensils/mold/pull/166))

### Changed

- **Quantized inference for Flux.2 GGUF**: rewrote the Flux.2 quantized transformer to keep weights in compressed GGUF format in VRAM and dequantize on-the-fly per matmul via `QMatMul`, matching ComfyUI and InvokeAI; Klein-9B Q4 VRAM usage drops from ~18GB to ~6GB, load time from minutes to seconds ([#166](https://github.com/utensils/mold/pull/166))
- **Default model**: changed from bare `flux2-klein` to explicit `flux2-klein:q8` for new users ([#166](https://github.com/utensils/mold/pull/166))
- **Klein guidance default**: both Klein-4B and Klein-9B now default to guidance=1.0, matching BFL model cards and InvokeAI ([#166](https://github.com/utensils/mold/pull/166))

### Fixed

- **Klein-9B OOM on CUDA**: "Metal out of memory" error on Linux/CUDA when loading Klein-9B models; root cause was wrong Qwen3 encoder (4B instead of 8B) and full dequantization exhausting VRAM ([#157](https://github.com/utensils/mold/issues/157), [#166](https://github.com/utensils/mold/pull/166))
- **CUDA OOM error message**: now shows "CUDA out of memory" on Linux instead of raw candle backtrace ([#166](https://github.com/utensils/mold/pull/166))
- **GGUF NaN safety**: quantized Flux.2 transformer wraps all linear operations with NaN-safe filter for CUDA, following SD3's established pattern ([#166](https://github.com/utensils/mold/pull/166))
- **Pre-existing TUI test failures**: settings tests expected bare `flux2-klein` but config resolves to `flux2-klein:q8` ([#166](https://github.com/utensils/mold/pull/166))

### Removed

- **Klein-9B alpha status**: all Klein-9B model descriptions, docs, and skills updated to remove alpha labels — Klein-9B is now fully supported ([#166](https://github.com/utensils/mold/pull/166))

## [0.5.2] - 2026-04-03

### Fixed

- **TUI metadata race condition**: switching models while a generation was running caused the saved image to record the newly selected model instead of the model that actually generated it; now uses `response.model` from the server as source of truth ([#161](https://github.com/utensils/mold/issues/161), [#163](https://github.com/utensils/mold/pull/163))
- **TUI batch mode**: setting batch > 1 in the TUI had no effect — only one image was generated; now loops client-side with seed increment, matching CLI behavior ([#162](https://github.com/utensils/mold/issues/162), [#163](https://github.com/utensils/mold/pull/163))
- **SSE protocol**: `SseCompleteEvent` now includes the `model` field so the server confirms which model generated the image; backward-compatible with older servers via `#[serde(default)]`

## [0.5.1] - 2026-04-03

### Fixed

- **TUI info panel crash**: model descriptions containing multi-byte UTF-8 characters (e.g., em dash `—`) caused a panic when truncated to fit the panel width; now uses `str::floor_char_boundary()` for safe truncation ([#159](https://github.com/utensils/mold/issues/159), [#160](https://github.com/utensils/mold/pull/160))

## [0.5.0] - 2026-04-03

### Added

- **`mold server start|status|stop`**: daemon management for standalone background server — PID file tracking, health polling, graceful shutdown cascade (HTTP → SIGTERM → SIGKILL), unmanaged process detection
- **Graceful server shutdown**: `POST /api/shutdown` endpoint triggers clean drain via `axum::serve::with_graceful_shutdown`; SIGTERM handler uses the same channel
- **API key authentication**: opt-in `MOLD_API_KEY` with single key, comma-separated, or `@/path/to/keys.txt` file reference; exempt `/health` and `/api/docs` ([#150](https://github.com/utensils/mold/pull/150))
- **Per-IP rate limiting**: `MOLD_RATE_LIMIT` with separate generation and read tiers (10x), `Retry-After` headers, configurable burst via `MOLD_RATE_LIMIT_BURST` ([#150](https://github.com/utensils/mold/pull/150))
- **Request ID correlation**: `X-Request-ID` header on all responses, preserved from client or auto-generated UUID v4 ([#150](https://github.com/utensils/mold/pull/150))
- **`mold stats` command**: disk usage overview for models, output, logs, and shared components with `--json` output ([#151](https://github.com/utensils/mold/pull/151))
- **`mold clean` command**: clean orphaned files, stale `.pulling` markers, hf-cache transient files, and old output images; dry-run by default with `--force` and `--older-than` flags ([#151](https://github.com/utensils/mold/pull/151))
- **Discord role-based access**: `MOLD_DISCORD_ALLOWED_ROLES` for comma-separated role names/IDs, with friendly denial messages ([#149](https://github.com/utensils/mold/pull/149))
- **Discord per-user daily quotas**: `MOLD_DISCORD_DAILY_QUOTA` with midnight UTC reset, `/quota` command, and `/admin reset-quota` / `/admin block` / `/admin unblock` commands ([#149](https://github.com/utensils/mold/pull/149))
- **TUI model deletion**: delete models from the TUI with confirmation dialog, disk space preview, and shared-file warnings ([#148](https://github.com/utensils/mold/pull/148))
- **TUI version display**: version string shown in upper-right corner of tab bar
- **Version logging**: server and TUI log version on startup for diagnostics
- **NixOS module**: `discord.environment` option for arbitrary bot env vars, `logToFile`/`logDir`/`logRetentionDays` server options ([#152](https://github.com/utensils/mold/pull/152))

### Fixed

- **Server production panics**: replaced `.unwrap()` and `.expect()` calls with proper error propagation in model_manager, logging, image metadata, cache mutex locks, and CLI commands ([#146](https://github.com/utensils/mold/pull/146))
- **Download progress blocking**: SHA-256 pre-scan moved to `spawn_blocking` so SSE progress events stream in real time instead of being blocked for 10-30s
- **Download status accuracy**: TUI now shows "Verifying...", "Downloading...", or "Preparing..." based on current phase instead of always showing "Generating..."
- **TUI session persistence**: settings now saved on quit (not just after generation), bare model names resolved correctly on restore, and fallback path no longer applies wrong model-specific params ([#153](https://github.com/utensils/mold/pull/153))
- **TUI download progress**: progress bar now shows file counter `[2/5]` with per-file and batch byte totals during model pulls ([#153](https://github.com/utensils/mold/pull/153))
- **Gallery thumbnail cache**: fixed missing cache entry when prepending new gallery images
- **VitePress container rendering**: prettier no longer collapses `:::` callout markers; set `proseWrap: "preserve"` for markdown files ([#152](https://github.com/utensils/mold/pull/152))

### Changed

- **README trimmed**: reduced from 718 to ~190 lines; detailed content moved to the documentation website ([#152](https://github.com/utensils/mold/pull/152))
- Documentation synced across website, CLAUDE.md, SKILL.md — added `mold stats`/`mold clean` to CLI reference, gallery API endpoints, Discord bot features, NixOS module options, Docker env vars ([#152](https://github.com/utensils/mold/pull/152))
- Comprehensive config.rs test coverage (843 lines of tests for config loading, migrations, path resolution) ([#147](https://github.com/utensils/mold/pull/147))

### Performance

- **Gallery thumbnail caching**: fixed-protocol thumbnails cached across render frames, eliminating per-frame `image::open()` + protocol creation

## [0.4.1] - 2026-04-02

### Fixed

- **crates.io publish**: add missing `mold-ai-discord` to release workflow publish chain — was never published, causing `mold-ai` publish to fail on 0.4.0

## [0.4.0] - 2026-04-02

### Added

- **Interactive Terminal UI**: full-featured TUI with Generate, Gallery, Models, and Settings views — built on ratatui with Kitty/Sixel/iTerm2/halfblock image preview ([#133](https://github.com/utensils/mold/pull/133), [#134](https://github.com/utensils/mold/pull/134))
- **TUI Gallery**: thumbnail grid with cached 256x256 previews, detail view with metadata, edit/regenerate/delete actions ([#134](https://github.com/utensils/mold/pull/134))
- **Settings editor**: `mold config` subcommands (`list`, `get`, `set`, `path`, `edit`) for managing config.toml from the CLI, plus a TUI Settings view for interactive editing ([#136](https://github.com/utensils/mold/pull/136))
- **Prompt history**: persisted across sessions with fuzzy search and Up/Down recall ([#133](https://github.com/utensils/mold/pull/133))
- **Session persistence**: TUI saves and restores all generation parameters across launches ([#133](https://github.com/utensils/mold/pull/133))
- **Auto-start server**: TUI automatically starts a background `mold serve` process, killed on exit ([#133](https://github.com/utensils/mold/pull/133))
- **File logging**: `--log-file` flag and `[logging]` config section for rotated log files in `~/.mold/logs/` ([#133](https://github.com/utensils/mold/pull/133))
- **Config migrations**: automatic migration from legacy config locations and formats ([#133](https://github.com/utensils/mold/pull/133))
- **LoRA fingerprint caching**: skip redundant transformer rebuilds when the same LoRA is reused across requests ([#131](https://github.com/utensils/mold/pull/131))
- **LoRA delta caching**: `LoraDeltaCache` caches pre-computed `B @ A * scale` delta tensors on CPU across transformer rebuilds ([#131](https://github.com/utensils/mold/pull/131))
- **Shared tokenizer pool**: cross-engine `SharedPool` caches T5 and CLIP tokenizers via `Arc<Tokenizer>`, saving ~100-150ms on model switches ([#131](https://github.com/utensils/mold/pull/131))

### Fixed

- **MPS memory guards**: prevent OOM on low-RAM Apple Silicon by checking available memory before allocation ([#134](https://github.com/utensils/mold/pull/134))
- **Test race condition**: eliminate flaky `queued_stream_receives_position_event` test ([#132](https://github.com/utensils/mold/pull/132))

### Changed

- Multi-model cache enhanced with `ModelResidency` states (Gpu, Parked, Unloaded) for smarter VRAM management ([#131](https://github.com/utensils/mold/pull/131))
- Documentation synced across website, README, CLAUDE.md, and SKILL.md — TUI, `mold config`, `[logging]` section, and Discord bot command references

## [0.3.1] - 2026-04-01

### Fixed

- **OOM prevention in eager/server mode**: drop transformer/UNet before VAE decode in all 8 pipelines — prevents out-of-memory on GPUs where transformer + VAE decode intermediates exceed VRAM ([#128](https://github.com/utensils/mold/pull/128))
- **Batch cache optimization**: skip text encoder load on prompt cache hit in all 8 sequential/CLI pipelines, saving 2-10s per batch image depending on model family ([#128](https://github.com/utensils/mold/pull/128))
- **Wuerstchen cache key**: include negative prompt and CFG flags in cache key to prevent silent cache collisions ([#128](https://github.com/utensils/mold/pull/128))
- **Unified memory fixes for low-RAM Apple Silicon**: drop text encoders after encoding on Metal (unified memory), use available memory (free + inactive) for variant selection on macOS, and show "Metal out of memory" instead of misleading CUDA error ([#126](https://github.com/utensils/mold/pull/126))
- **Skip encoder reload on cache hit**: Flux2 and Z-Image eager paths no longer reload encoder from disk before checking prompt cache ([#126](https://github.com/utensils/mold/pull/126))
- **Stale alpha labels**: removed incorrect alpha labels from Klein-4B model descriptions; Klein-9B correctly retains alpha ([#128](https://github.com/utensils/mold/pull/128))

### Changed

- Candle dependencies switched from git refs to crates.io 0.9.3 — unblocks `cargo publish` for all mold crates

## [0.3.0] - 2026-04-01

### Added

- **Flux.2 Klein-9B model (alpha)**: 9B parameter distilled model with larger Qwen3 encoder (hidden_size=4096), 4-step generation ([#123](https://github.com/utensils/mold/pull/123))
- **SD 3.5 model family**: Triple encoder (CLIP-L + CLIP-G + T5-XXL), quantized MMDiT with NaN-safe inference, SLG support ([#111](https://github.com/utensils/mold/pull/111))
- **Qwen-Image model family**: Qwen2.5-VL text encoder, 3D causal VAE, flow-matching with classifier-free guidance ([#111](https://github.com/utensils/mold/pull/111))
- **LLM-powered prompt expansion**: local Qwen3-1.7B GGUF expansion with `--expand` flag, user-configurable templates, per-family tuning ([#86](https://github.com/utensils/mold/pull/86), [#88](https://github.com/utensils/mold/pull/88))
- **CUDA block-level offloading**: streams transformer blocks CPU↔GPU to reduce VRAM from ~24GB to ~2-4GB via `--offload` flag ([#84](https://github.com/utensils/mold/pull/84))
- **LoRA adapter support**: custom VarBuilder backend for FLUX BF16 and GGUF quantized models with `--lora` and `--lora-scale` flags ([#94](https://github.com/utensils/mold/pull/94), [#97](https://github.com/utensils/mold/pull/97))
- **Negative prompt support**: `--negative-prompt` and `--no-negative` flags for CFG-based model families ([#90](https://github.com/utensils/mold/pull/90))
- **Dimension validation**: warns when dimensions don't match model recommendations ([#116](https://github.com/utensils/mold/pull/116))
- **Discord bot integration**: `/generate`, `/expand`, `/models`, `/status` slash commands, standalone or embedded in server ([#72](https://github.com/utensils/mold/pull/72), [#75](https://github.com/utensils/mold/pull/75))
- **Docker/RunPod deployment**: multi-stage Dockerfile, RunPod entrypoint, network volume support ([#110](https://github.com/utensils/mold/pull/110))
- **VitePress documentation site**: full docs at utensils.github.io/mold with model guides, API reference, deployment docs ([#114](https://github.com/utensils/mold/pull/114))
- **Multi-CUDA architecture support**: Ada (sm_89) + Blackwell (sm_120) build targets ([#91](https://github.com/utensils/mold/pull/91))
- **Terminal image preview**: `--preview` flag to display images inline ([#55](https://github.com/utensils/mold/pull/55))
- **`mold default` command**: get/set default model ([#64](https://github.com/utensils/mold/pull/64))
- **`mold info` command**: installation overview and per-model details with optional SHA-256 verify ([#53](https://github.com/utensils/mold/pull/53))
- **Multi-model server cache**: server LoRA support and server-side prompt expansion ([#107](https://github.com/utensils/mold/pull/107))
- **SHA-256 download verification**: fails on integrity mismatch, `--skip-verify` to override ([#56](https://github.com/utensils/mold/pull/56))
- **One-line install script**: `curl | sh` installer with GPU auto-detection

### Changed

- **Default model changed to `flux2-klein:q8`**: fully ungated (no HuggingFace auth), Apache 2.0 licensed, fast 4-step generation ([#125](https://github.com/utensils/mold/pull/125))
- Lazy mmap weight loading replaces eager per-tensor loading for better performance ([#122](https://github.com/utensils/mold/pull/122))
- `mold run` auto-resizes source images to model-native resolution for img2img ([#51](https://github.com/utensils/mold/pull/51))
- Server queues concurrent requests instead of dropping them ([#68](https://github.com/utensils/mold/pull/68))
- Git SHA and build date embedded in version output ([#67](https://github.com/utensils/mold/pull/67))
- CLI validates file-based arguments before expansion/inference ([#101](https://github.com/utensils/mold/pull/101))

### Fixed

- Model weight loading and Qwen-Image quantized inference performance ([#124](https://github.com/utensils/mold/pull/124))
- Batch generation saves and previews images immediately ([#120](https://github.com/utensils/mold/pull/120))
- Expand segfault, crash reporting, and ps process detection ([#118](https://github.com/utensils/mold/pull/118))
- LoRA VRAM leak in GGUF re-quantization ([#100](https://github.com/utensils/mold/pull/100))
- Weight loader CPU fallback for dtype conversion ([#98](https://github.com/utensils/mold/pull/98))
- Server unloads current model before loading a different one ([#106](https://github.com/utensils/mold/pull/106))
- Server fully releases GPU memory on unload ([#62](https://github.com/utensils/mold/pull/62))
- FLUX auto-patch for city96-format GGUFs missing embedding layers ([#61](https://github.com/utensils/mold/pull/61))
- `mold rm` comprehensive cleanup for shared files, hf-cache, and stale markers ([#93](https://github.com/utensils/mold/pull/93))
- Qwen-Image denoising quality and performance ([#111](https://github.com/utensils/mold/pull/111))
- SD1.5 repo updated to canonical location ([#113](https://github.com/utensils/mold/pull/113))
- Wuerstchen v2 sampling alignment and default negative prompt ([#99](https://github.com/utensils/mold/pull/99))

### Removed

- **Flux.2 Klein-base-4B model**: all variants removed — denoising fails to converge, producing incoherent output ([#125](https://github.com/utensils/mold/pull/125))

## [0.2.0] - 2026-03-24

### Added

- **Wuerstchen v2 model family**: 3-stage cascade pipeline (Prior → Decoder → VQ-GAN) with CLIP-G text encoder and 42x latent compression ([#25](https://github.com/utensils/mold/pull/25))
- **Image-to-image mode**: `--image` and `--strength` flags for SD1.5, SDXL, and FLUX; stdin support with `--image -` for piping ([#27](https://github.com/utensils/mold/pull/27))
- **Inpainting**: `--mask` flag for selective repaint (white=repaint, black=preserve) ([#27](https://github.com/utensils/mold/pull/27))
- **ControlNet conditioning for SD1.5**: `--control`, `--control-model`, `--control-scale` flags with canny, depth, and openpose models ([#28](https://github.com/utensils/mold/pull/28))
- **Configurable noise schedulers**: DDIM, Euler Ancestral, UniPC for SD1.5/SDXL via `--scheduler` flag ([#26](https://github.com/utensils/mold/pull/26))
- **Inference caching**: LRU caches for text encoder embeddings, source image latents, masks, and control tensors — avoids re-encoding on repeated prompts/batches ([#41](https://github.com/utensils/mold/pull/41))
- **Unified model catalog**: automatic model discovery from `MOLD_MODELS_DIR` without requiring config entries; single source of truth for CLI and server ([#41](https://github.com/utensils/mold/pull/41))
- **PNG generation metadata**: prompt, model, seed, steps, guidance embedded as PNG text chunks by default; disable with `--no-metadata` ([#43](https://github.com/utensils/mold/pull/43))
- **18 community models**: FLUX fine-tunes (Krea, JibMix, Ultrareal, Iniverse), SDXL fine-tunes (Pony, Cyberrealistic), and Flux.2 Klein GGUF quantizations ([#39](https://github.com/utensils/mold/pull/39))
- **NixOS module**: declarative `services.mold` with systemd hardening, GPU access, HF token support, firewall options, and shell completions
- **Centralized color/theme system**: semantic icon/prefix helpers with ANSI 16-color palette for broad terminal compatibility ([#42](https://github.com/utensils/mold/pull/42))

### Changed

- Custom `thiserror` error enum replaces generic `MoldError` for precise error handling ([#16](https://github.com/utensils/mold/pull/16))
- Server returns structured JSON error responses with error codes ([#18](https://github.com/utensils/mold/pull/18))
- `mold list` shows separate SIZE and FETCH columns for accurate disk usage ([#33](https://github.com/utensils/mold/pull/33), [#37](https://github.com/utensils/mold/pull/37))
- `mold pull` routes through server when available, falls back to local download ([#41](https://github.com/utensils/mold/pull/41))
- Batch generation reuses loaded engine across iterations for faster multi-image runs ([#41](https://github.com/utensils/mold/pull/41))
- Generation output shows prompt in header block and model name in completion summary ([#42](https://github.com/utensils/mold/pull/42))

### Fixed

- Batch image generation with proper seed increment across images ([#17](https://github.com/utensils/mold/pull/17))
- FLUX img2img auto-resize, schedule, and VAE normalization ([#34](https://github.com/utensils/mold/pull/34))
- Wuerstchen pipeline reliability, context-aware image sizes, and Ctrl+C handling ([#31](https://github.com/utensils/mold/pull/31))
- Server reliability improvements and code quality cleanup ([#38](https://github.com/utensils/mold/pull/38))
- UTF-8 safe prompt truncation prevents panics on multi-byte characters ([#42](https://github.com/utensils/mold/pull/42))
- Multiple Nix build/service fixes: CUDA toolkit discovery, modelsDir permissions, EnvironmentFile loading, cross-eval
- Broken model descriptions renamed to alpha ([#40](https://github.com/utensils/mold/pull/40))

### Improved

- CLI help messages with examples, environment variable documentation, and grouped options ([#12](https://github.com/utensils/mold/pull/12))
- Code quality, validation, and deduplication from peer review ([#15](https://github.com/utensils/mold/pull/15))

## [0.1.0] - 2026-03-19

Initial public release on [crates.io](https://crates.io/crates/mold-ai).

### Added

- **Seven model families**: FLUX.1 (dev/schnell), SD 1.5, SDXL, SD 3.5, Z-Image, Flux.2 Klein, Qwen-Image
- **CLI commands**: `run`, `serve`, `pull`, `rm`, `list`, `info`, `unload`, `ps`, `version`, `completions`
- **Local + remote inference**: run models on local GPU or connect to a remote `mold serve` instance
- **Automatic local fallback**: if server is unreachable, falls back to local GPU inference
- **GGUF quantized model support**: BF16 and GGUF (Q3–Q8) for reduced VRAM usage
- **Smart VRAM management**: dynamic device placement, drop-and-reload text encoders, quantized encoder auto-fallback
- **Pipe-friendly output**: `mold run "a cat" | viu -` just works (image bytes to stdout, status to stderr)
- **Stdin prompt support**: `echo "a cat" | mold run flux-schnell`
- **Model management**: `mold pull` downloads from HuggingFace, `mold list` shows disk usage, `mold rm` cleans up
- **GPU backends**: CUDA (Linux) and Metal (macOS) via candle
- **Deterministic seeds**: CPU-based noise generation ensures same seed produces identical images across backends
- **SSE progress streaming**: `POST /api/generate/stream` for real-time denoising progress
- **OpenAPI docs**: interactive API documentation at `/api/docs`
- **Shell completions**: static and dynamic completions for bash, zsh, fish, elvish, powershell
- **Nix flake**: reproducible builds with Metal (macOS) and CUDA (Linux) support
- **Pre-built binaries**: GitHub releases for macOS (Metal) and Linux (CUDA)

### crates.io packages

| Crate                                                             | Description                             |
| ----------------------------------------------------------------- | --------------------------------------- |
| [`mold-ai`](https://crates.io/crates/mold-ai)                     | CLI binary (`cargo install mold-ai`)    |
| [`mold-ai-core`](https://crates.io/crates/mold-ai-core)           | Shared types, API protocol, HTTP client |
| [`mold-ai-inference`](https://crates.io/crates/mold-ai-inference) | Candle-based inference engine           |
| [`mold-ai-server`](https://crates.io/crates/mold-ai-server)       | Axum HTTP inference server              |

[Unreleased]: https://github.com/utensils/mold/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/utensils/mold/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/utensils/mold/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/utensils/mold/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/utensils/mold/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/utensils/mold/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/utensils/mold/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/utensils/mold/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/utensils/mold/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/utensils/mold/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/utensils/mold/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/utensils/mold/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/utensils/mold/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/utensils/mold/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/utensils/mold/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/utensils/mold/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/utensils/mold/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/utensils/mold/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
