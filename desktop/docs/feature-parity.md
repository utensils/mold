# mold — User-Facing Feature Inventory (for desktop-app parity)

Sources of truth: CLI clap defs (`crates/mold-cli/src/main.rs`), core wire types (`crates/mold-core/src/types.rs`), per-family capability map (`web/src/lib/generateCapabilities.ts`), server routes (`crates/mold-server/src/routes.rs`), catalog (`crates/mold-catalog/`), config (`crates/mold-core/src/config.rs`, `crates/mold-db/src/settings.rs`), chain (`crates/mold-core/src/chain.rs`, `chain_toml.rs`), Discord (`crates/mold-discord/src/commands/`), TUI (`crates/mold-tui/src/`).

---

## 1. Core generation parameters (the shared `GenerateRequest`)

Every surface builds this one struct; the desktop app should model it 1:1. Defaults come from the model's `ModelConfig` unless overridden.

| Param                                | Type                           | Default                   | Notes                                                                                                                                                                               |
| ------------------------------------ | ------------------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`                             | string                         | —                         | Required. stdin-pipeable in CLI.                                                                                                                                                    |
| `negative_prompt`                    | string?                        | model/config              | Only effective on CFG families (SD1.5, SDXL, SD3, Wuerstchen). Ignored by distilled/flow-matching families. CLI `-n/--negative-prompt`; `--no-negative` forces empty unconditional. |
| `model`                              | string                         | `default_model`           | `model:tag` resolution — see §2.                                                                                                                                                    |
| `width` / `height`                   | u32                            | model native              | Snapped to multiples of 16, megapixel-clamped. img2img defaults to fitted source size; qwen-image-edit derives from first edit image (~1024² target area).                          |
| `steps`                              | u32                            | model native              |                                                                                                                                                                                     |
| `guidance`                           | f64                            | 3.5 (0 for schnell/turbo) | Guidance scale.                                                                                                                                                                     |
| `seed`                               | u64?                           | random                    | Batch uses `base_seed + i` (wrapping). CPU-seeded noise → cross-backend determinism.                                                                                                |
| `batch_size`                         | u32                            | 1                         | Batch loops single-image requests; qwen-image-edit forces batch=1. Not allowed to stdout.                                                                                           |
| `output_format`                      | enum?                          | family-aware              | `png`, `jpeg`/`jpg`, `gif`, `apng`, `webp`, `mp4`. Video families default `mp4` (ltx2/ltx-video) else `apng`; images default `png`. Explicit wrong choice → 422.                    |
| `embed_metadata`                     | bool?                          | true                      | Embeds `mold:parameters` metadata. CLI `--no-metadata`; env `MOLD_EMBED_METADATA`.                                                                                                  |
| `scheduler`                          | enum?                          | model                     | `default`, `ddim`, `euler-ancestral`, `unipc`. UNet-only (SD1.5, SDXL); ignored by flow-matching families.                                                                          |
| `cfg_plus`                           | bool?                          | false                     | CFG++ manifold-projection guidance. SD3/SD3.5 (and SDXL/SD1.5 with DDIM only); lowers usable CFG to ~1.5–2.5. Ignored by distilled families.                                        |
| `source_image`                       | bytes?                         | —                         | img2img. CLI `-i/--image` (repeatable), `-` = stdin (single-image families).                                                                                                        |
| `edit_images`                        | bytes[]?                       | —                         | Qwen-Image-Edit multi-image (first = primary target, rest = references).                                                                                                            |
| `strength`                           | f64                            | 0.75                      | Denoising strength (0=preserve, 1=full noise).                                                                                                                                      |
| `mask_image`                         | bytes?                         | —                         | Inpainting (white=repaint, black=preserve). Requires source_image. Not for qwen-edit.                                                                                               |
| `control_image`                      | bytes?                         | —                         | ControlNet conditioning.                                                                                                                                                            |
| `control_model`                      | string?                        | —                         | e.g. `controlnet-canny-sd15`. SD1.5 only.                                                                                                                                           |
| `control_scale`                      | f64                            | 1.0                       | 0–2.0.                                                                                                                                                                              |
| `expand`                             | bool?                          | config                    | LLM prompt expansion (§8). CLI `--expand`/`--no-expand`; env `MOLD_EXPAND`.                                                                                                         |
| `original_prompt`                    | string?                        | —                         | Pre-expansion prompt (client-side expansion).                                                                                                                                       |
| `lora`                               | `{path, scale}`?               | —                         | Single adapter.                                                                                                                                                                     |
| `loras`                              | `{path, scale}[]`?             | —                         | Repeatable stack (LTX-2 / multi-adapter families).                                                                                                                                  |
| `frames`                             | u32?                           | model                     | Video only. LTX requires 8n+1 (9,17,25,…). Implies video output. LTX-2 >97 auto-chains (§7).                                                                                        |
| `fps`                                | u32?                           | 24                        | Video output fps.                                                                                                                                                                   |
| `upscale_model`                      | string?                        | —                         | Post-gen upscale (e.g. `real-esrgan-x4plus:fp16`).                                                                                                                                  |
| `gif_preview`                        | bool                           | false                     | Request animated GIF preview alongside video.                                                                                                                                       |
| `enable_audio`                       | bool?                          | family                    | LTX-2/2.3 synchronized audio. CLI `--audio`/`--no-audio`. mp4 only.                                                                                                                 |
| `audio_file` / `audio_file_path`     | bytes?/str?                    | —                         | Audio-to-video conditioning (path only on trusted server).                                                                                                                          |
| `source_video` / `source_video_path` | bytes?/str?                    | —                         | Video-to-video / retake.                                                                                                                                                            |
| `keyframes`                          | `{frame, image}[]`?            | —                         | LTX-2 keyframe interpolation. CLI `--keyframe <frame:path>` repeatable.                                                                                                             |
| `pipeline`                           | enum?                          | —                         | LTX-2 mode: `one-stage`, `two-stage`, `two-stage-hq`, `distilled`, `ic-lora`, `keyframe`, `a2vid`, `retake`.                                                                        |
| `retake_range`                       | `{start_seconds,end_seconds}`? | —                         | Partial regen time window. CLI `--retake <start:end>`.                                                                                                                              |
| `spatial_upscale`                    | enum?                          | —                         | LTX-2.3 latent spatial upscale: `x1-5`, `x2`.                                                                                                                                       |
| `temporal_upscale`                   | enum?                          | —                         | LTX-2.3 latent temporal upscale: `x2`.                                                                                                                                              |
| `placement`                          | DevicePlacement?               | auto                      | Per-component device override (§6).                                                                                                                                                 |

CLI-only extras layered on top: `--clip-frames`, `--motion-tail` (chain tuning), `--frames-per-clip` (multi-prompt sugar), `--lora-scale`, `--camera-control` (LTX-2 camera LoRA presets: dolly-in/-left/-out/-right, jib-down/-up, static), text-encoder variant flags (§6), `--eager`, `--offload`.

Desktop parity: `upscale_model` is surfaced by the "Upscale" select in `ParamPanel.vue` (image families only; lists every known upscaler from `/api/models`, downloads on first use on the selected generation host, including multi-GPU remotes), and the gallery's right-click **Upscale** action drives `POST /api/upscale` directly, saving the result to the local gallery (`/api/upscale` does not persist server-side). `source_image` / `mask_image` are surfaced by `SourceImageWell.vue` — a **Choose from gallery…** picker (`ImagePickerModal.vue`, upload or authed gallery thumbnails) plus an **Edit mask…** brush/erase canvas (`MaskEditorModal.vue`, invert / clear / undo-redo cap 20 → white-on-transparent PNG). The LTX-2 advanced fields (`pipeline`, `spatial_upscale`, `temporal_upscale`, `retake_range`, `source_video`, `keyframes`, and `audio_file` for the a2vid pipeline) are surfaced by an "LTX-2 pipeline" disclosure in `ParamPanel.vue` (ltx2 family only, gated by `capabilities.supportsAdvancedVideo`; kebab-case enum values, nulls omitted, pruned on family change). The gallery image picker (`ImagePickerModal.vue`) filters its grid to PNG/JPEG, the only still formats the generate endpoints accept. Keyframe frame indices follow the web SPA (0-based, +24 suggestion) — not the 8n+1 total-frame-count constraint.

Post-generation upscale persists distinct `-original` and `-upscaled` entries on the serving host. Remote-output auto-save mirrors both to This Mac, and gallery reuse restores the pre-upscale generation dimensions rather than the upscaled raster size. If upscaling fails after generation, the job completes with the original image as a single artifact.

`GenerateResponse`: `images[]` (`{data,width,height,index}`), optional `video` (`{data,format,width,height,frames,fps,thumbnail,gif_preview}`), `generation_time_ms`, `model`, `seed_used`, `gpu` (ordinal, multi-GPU).

---

## 2. Model families & catalog

**Family taxonomy** (load-bearing slugs): `flux`, `flux2`, `sd15`, `sdxl`, `sd3`, `z-image`, `ltx-video`, `ltx2`, `qwen-image`, `qwen-image-edit`, `wuerstchen`, plus utility `controlnet`, `upscaler`, `companion`, `qwen3-expand`.

**Per-family capability matrix** (from `generateCapabilities.ts` — drive UI enable/disable):

| Family          | Neg prompt | Scheduler              | CFG++    | Video | Audio | LoRA | ControlNet | Source-image mode | Mask | batch=1 forced |
| --------------- | ---------- | ---------------------- | -------- | ----- | ----- | ---- | ---------- | ----------------- | ---- | -------------- |
| flux / flux2    | ✗          | ✗                      | ✗        | ✗     | ✗     | ✓    | ✗          | single            | ✓    | ✗              |
| sd15            | ✓          | ✓ (ddim/euler-a/unipc) | ✓ (DDIM) | ✗     | ✗     | ✓    | ✓          | single            | ✓    | ✗              |
| sdxl            | ✓          | ✓                      | ✓ (DDIM) | ✗     | ✗     | ✓    | ✗          | single            | ✓    | ✗              |
| sd3/sd3.5       | ✓          | ✗                      | ✓        | ✗     | ✗     | ✓    | ✗          | single            | ✓    | ✗              |
| z-image         | ✗          | ✗                      | ✗        | ✗     | ✗     | ✓    | ✗          | single            | ✓    | ✗              |
| qwen-image      | ✗          | ✗                      | ✗        | ✗     | ✗     | ✓    | ✗          | single            | ✓    | ✗              |
| qwen-image-edit | ✗          | ✗                      | ✗        | ✗     | ✗     | ✓    | ✗          | qwen-edit (multi) | ✗    | ✓              |
| wuerstchen      | ✓          | ✗                      | ✗        | ✗     | ✗     | ✗    | ✗          | single            | ✓    | ✗              |
| ltx-video       | ✗          | ✗                      | ✗        | ✓     | ✗     | ✓    | ✗          | single            | —    | ✗              |
| ltx2            | ✗          | ✗                      | ✗        | ✓     | ✓     | ✓    | ✗          | single            | —    | ✗              |

**Quirks:**

- **LTX-2 is CUDA-only** for real generation (CPU correctness-only, Metal unsupported). Has its own MP4+AAC media pipeline. Camera-control LoRA presets currently resolve only Lightricks LTX-2 19B (LTX-2.3 needs explicit .safetensors).
- Video families: `ltx-video`, `ltx2`. Audio: `ltx2` only (mp4 only).
- Z-Image has a bespoke quantized transformer (GGUF naming differs from BF16).
- Flow-matching families (FLUX, SD3, Z-Image, Flux.2, Qwen-Image) ignore scheduler.
- H.264 decode baseline for LTX-2 source ingest; `mp4` feature only gates AAC mux.

**Built-in catalog (name:tag), quantization variants:**

- FLUX: `flux-schnell`, `flux-dev`, `flux-krea`, `jibmix-flux`, `iniverse-mix`, `ultrareal-v2/v3/v4` — tags `bf16`/`fp8`/`q8`/`q6`/`q5`/`q4`/`q3`.
- Flux.2 Klein: `flux2-klein`, `flux2-klein-9b` — `bf16`/`q8`/`q6`/`q4` (4B and 9B variants).
- SD1.5: `sd15:fp16`, `dreamshaper-v8`, `realistic-vision-v5`.
- SDXL: `sdxl-base`, `sdxl-turbo`, `juggernaut-xl`, `dreamshaper-xl`, `realvis-xl`, `playground-v2.5`, `pony-v6`, `cyberrealistic-pony` — `fp16`.
- SD3.5: `sd3.5-large:q4/q8`, `sd3.5-large-turbo:q8`, `sd3.5-medium:q8`.
- Z-Image: `z-image-turbo:bf16/q8/q6/q4`.
- Qwen-Image: `qwen-image`, `qwen-image-2512` (`bf16`,`q2`–`q8`), `qwen-image-lightning:fp8`/`fp8-8step`.
- Qwen-Image-Edit: `qwen-image-edit-2511:bf16/q2..q8`.
- Wuerstchen: `wuerstchen-v2:fp16`.
- LTX-Video: `ltx-video-0.9.6`, `-0.9.6-distilled`, `-0.9.8-2b-distilled`, `-0.9.8-13b-dev`, `-0.9.8-13b-distilled` (`bf16`).
- LTX-2: multiple 19B/22B distilled variants (`fp8` etc.).
- ControlNet: `controlnet-canny-sd15`, `-depth-sd15`, `-openpose-sd15` (`fp16`).
- Upscalers: `real-esrgan-x4plus:fp16`, `real-esrgan-x2plus:fp16`, `real-esrgan-x4plus-anime:fp16`.

**Name resolution** (`manifest::resolve_model_name`): `model:tag`; bare names try `:q8`→`:fp16`→`:bf16`→`:fp8`; legacy dash form (`flux-dev-q4`) → colon form.

**Live catalog search** (`GET /api/catalog/search`, `useCatalog.ts`): HF + Civitai proxy, 5-min in-proc cache. Query params: `q`, `family`, `kind`, `source` (hf/civitai), `include_nsfw`, `page`, `page_size`. Catalog IDs `cv:<id>` / `hf:<author>/<name>` usable directly as model names (auto-resolve + download). Endpoints: `/api/catalog/families`, `/api/catalog/installed`, `/api/catalog/*id` (GET entry / POST dispatch = download). The desktop Catalog view stacks installed inventory above available catalog rows in one screen (All / Images / Video media chips via `?type=`, active downloads pinned top with source glyph + target host), supports Grid / Table density, and asks for the destination host when several ready hosts are connected. Civitai source previews are normalized server-side and client-side to a shared 512 px CDN derivative, then lazy-decoded inside layout/paint-contained cards so old remote hosts and both layouts use the same browser-cached asset. Missing and failed previews fall back to `ModelFamilyPlaceholder`, a zero-network family mark sized for both layouts. On remote catalog requests it may send request-scoped `X-Mold-HF-Token` / `X-Mold-Civitai-Token` fallback headers from its local secret store; server environment credentials retain precedence. Civitai LoRAs carry `trained_words` (trigger phrases). Entries carry an additive `page_url` (HF repo page / Civitai `models/{id}?modelVersionId={vid}` page; null when uncomposable, absent on older servers) — the desktop falls back to deriving HF pages from `source_id` (`lib/catalog.ts catalogPageUrl`). HF search-summary rows have no `size_bytes`; catalog cards resolve it lazily via `GET /api/catalog/{id}` (`lib/catalogSizes.ts`, memoized, ≤4 in flight). `MOLD_CATALOG_DISABLE=1` disables. LoRA list: `GET /api/loras?model=` filters by compatible family.

---

## 3. Model management

- **list** (`mold list`/`ls`, `GET /api/models`): installed models + disk usage + available-to-pull.
- **pull** (`mold pull <model>`, `POST /api/models/pull`, SSE `/api/downloads/stream`): download with live progress; `--skip-verify` (SHA-256). Auto-pull on generate if missing (server or local fallback). HF_TOKEN / CIVITAI_TOKEN for gated repos.
- **rm** (`mold rm <model...>`/`remove`, `DELETE /api/models/unload` is separate): removes model + unique files; shared files (VAE/CLIP) kept until unreferenced; `--force` skips confirm.
- **info** (`mold info [model]`, `--verify`): model details or install overview; SHA-256 integrity verify. `GET /api/models/:model/components`.
- **stats** (`mold stats`, `--json`): disk usage for models/output/logs/shared components.
- **clean** (`mold clean`, `--force`, `--older-than 30d`): dry-run by default; stale `.pulling` markers (>6h), orphaned shared files, hf-cache transients, old output images.
- **load/unload** (`mold unload`, `POST /api/models/load`, `DELETE /api/models/unload`): free GPU memory.
- **ps** (`mold ps`, `GET /api/status`): server status + loaded models.
- **downloads drawer** (web/desktop): `GET/POST /api/downloads`, `DELETE /api/downloads/:id`, live SSE stream (cancelable, progress bytes/percent). Desktop keeps one stream for every ready host, pins active rows at the top of the Catalog view, mirrors a host's rows on its detail page, and labels/routes each row by origin (source glyph + target host; primary rows resolve to the local host's label).
- **multi-host Installed shelf (desktop)**: merges every ready host's `/api/models` result, deduplicates equal names, collects host badges, and routes load/unload/info/remove to the row's owning host (preferring the local copy when several hosts have one). Host detail forces a fresh inventory fetch rather than trusting the 60-second routing cache.
- **multi-host Chains (desktop)**: merges installed video models from every ready host and routes chain limits, creation, durable job actions, SSE progress, and stage previews to the host selected for that model. Remote CUDA LTX-2 stays available when the built-in Mac engine is Metal.
- **boot reconnect (desktop)**: every `savedHosts` entry is attempted immediately on each launch, concurrently with local-engine startup. `connectedHostIds` is retained for migration/order compatibility but never suppresses a remembered host; failed probes remain visible and polling retries them.
- **SIZE vs FETCH semantics**: catalog entries distinguish declared size vs actual fetch. Installed rows separately label primary checkpoint **weights** (`size_gb`) and the larger footprint **with shared runtime** (`disk_usage_bytes`, including referenced shared encoders/VAEs); the all-host header does not sum per-model runtime footprints because that would double-count shared files.
- **safe manifest pull targets (desktop)**: runnable built-in variants precede live Hugging Face results; aggregate `separated` checkpoint repositories and repositories already represented by the manifest are suppressed so one click cannot accidentally fetch every checkpoint in a multi-model LTX repository.

---

## 4. Gallery / media management

- **Browse** (`mold` web `GalleryPage`, `GET /api/gallery`): returns `GalleryImage[]` = `{filename, metadata (OutputMetadata), timestamp, format, size_bytes, metadata_synthetic}`.
- **Image/thumbnail** URLs: `GET /api/gallery/image/:filename`, `/api/gallery/preview/:filename`, thumbnail endpoint.
- **Delete**: `DELETE /api/gallery/image/:filename` — always enabled (`capabilities.gallery.can_delete: true`); pair with `MOLD_API_KEY` if exposed.
- **Embedded metadata**: PNG (per-field tEXt/iTXt + composite `mold:parameters` JSON), JPEG (COM marker + XMP APP1 — readable by exiftool/Photoshop/Lightroom/GIMP), mp4/webp/apng/gif carry metadata too. Server reconciles DB from embedded metadata on startup; `metadata_synthetic` flags synthesized rows.
- **Video playback**: gallery cards render mp4/animated formats; VideoData carries `thumbnail` (first-frame PNG) + `gif_preview`.
- **Metadata display** (web `Metadata.vue`, TUI info panel): full generation params from embedded chunk.
- **Filtering / columns**: TUI gallery has configurable columns (`tui.gallery_columns`), view mode (`tui.view_mode`), centered aspect-correct thumbnails (Kitty/Sixel/iTerm2 protocol path — must stay centered). Web `GalleryFeed` with cards + detail drawer.
- Gallery writes happen in server (queue upsert + background reconcile), CLI (`record_local_save`), TUI (`gallery_scan`). DB additive; embedded metadata always written.
- **Live updates**: the desktop subscribes app-wide to `GET /api/events` (`events` store) and inserts/removes gallery tiles in place from `gallery_added` / `gallery_removed` frames — the Gallery view stays current while generations run anywhere (this window, another client, the queue). Older servers without the endpoint (`capabilities.events` absent) fall back to polling `GET /api/gallery` every 5 s while jobs are pending. Web SPA still refreshes on its own generation completions only.
- **Unified multi-host gallery (desktop only)**: one date-sorted grid merges a bucket per connected host (`gallery` store, keyed by host id); there is no separate "This Mac" IPC bucket — the local primary's `/api/gallery` IS this Mac's gallery (IPC-saved files included), so nothing double-lists. All deduplicates matching filenames across buckets, prefers the This Mac copy for media/actions, and lists every available location on the tile and in the lightbox; individual source filters remain complete raw host views. `HostFilterChips` (All · This Mac · host · …, live counts) filter the grid, with All reporting the deduplicated count. Media blobs are cached per (host, path) and evicted per origin (`evictHostMedia`). Actions are origin-aware: delete hits the represented host over HTTP (legacy bare-`local` keys fall back to IPC), **Save to this Mac** pulls bytes from any remote host, Upscale stays primary-engine-only (follow-up: route to origin), Reveal in Finder only for files on this machine. SSE stays primary-only; non-primary buckets poll every 15 s while the view is mounted and refresh when a routed job completes. Sidebar host rows get **View gallery** → `/gallery?host=<id>`.

---

## 5. Prompt history & prompt expansion

**Prompt history** (`prompt_history` table): `push`, `recent(limit)`, `search(query,limit)`, `count`, `trim_to(keep)`, `clear`. Entry = `{prompt, model, timestamp}`. TUI: HistoryPrev/HistoryNext/SearchHistory actions.

- **Unified multi-host history (desktop only)**: the History view merges every connected host. The **Runs** tab reads the unified gallery store (`gallery.filtered`) with the same `HostFilterChips` filter and per-row host chips as the Gallery; thumbnails authenticate against their origin host. The **Prompts** tab fans `GET /api/history` out over every ready host (`fetchHistoryAll` — `Promise.allSettled`, entries tagged `hostId`/`hostLabel`, merged newest-first); hosts that 404/503 (`HISTORY_UNAVAILABLE`, pre-history servers) are skipped, and the unavailable empty-state shows only when no host supports history. **Clear** follows the active chip via `clearScope`: the one filtered host, or every history-capable host under All (confirm step names them).

**Prompt expansion** (`mold expand`, `POST /api/expand`, web `ExpandModal.vue`): LLM expands a short prompt into detailed generation prompt(s), model-aware style.

- CLI: `mold expand "<prompt>" --model --variations N --json --backend --expand-model`.
- Config (`ExpandSettings`, DB `expand.*`): `enabled`, `backend` (`local` GGUF or OpenAI-compatible URL), `model` (default `qwen3-expand:q8`), `api_model` (e.g. `qwen2.5:3b`), `temperature`, `top_p`, `max_tokens`, `thinking` (Qwen3 reasoning), `system_prompt` (placeholders `{WORD_LIMIT}`,`{MODEL_NOTES}`), `batch_prompt` (placeholders `{N}`,`{WORD_LIMIT}`,`{MODEL_NOTES}`), per-family `families` overrides (word limit + style notes). Env: `MOLD_EXPAND*`.
- Inline on generate via `--expand`; batch expansion produces per-image prompt variations.

---

## 6. Device placement, encoders & performance knobs

**Per-component placement** (`DevicePlacement`, CLI `--device-*`, env `MOLD_PLACE_*`): accepts `auto`/`cpu`/`gpu`/`gpu:N` per component:

- `--device-text-encoders` (all), `--device-transformer`, `--device-vae`, `--device-t5`, `--device-clip-l`, `--device-clip-g`, `--device-qwen`.

**Text-encoder quantization variants** (auto-fallback to largest that fits): `--t5-variant` (auto/fp16/q8/q6/q5/q4/q3), `--qwen3-variant` (Z-Image: auto/bf16/q8/q6/iq4/q3), `--qwen2-variant` (Qwen-Image: auto/bf16/q8..q2), `--qwen2-text-encoder-mode` (auto/gpu/cpu-stage/cpu). Env `MOLD_*_VARIANT`.

**Perf**: `--offload` / `MOLD_OFFLOAD=1` (FLUX block-level CPU↔GPU streaming: ~24GB→2–4GB, 3–5× slower, auto under pressure), `--eager` (keep all components loaded), `--gpus` / `MOLD_GPUS` (ordinal selection). Tier-1 knobs: `MOLD_KEEP_TE_RAM`, `MOLD_LORA_BYPASS`, `MOLD_VAE_TILED`, `MOLD_ATTN` (+ `MOLD_ATTN_CHUNK`), plus many LTX-2/flux debug/tuning envs. Web `PlacementPanel.vue` surfaces placement UI; desktop `PlacementSection.vue` (Settings → Advanced) saves per-model placement defaults via `PUT`/`DELETE /api/config/model/:name/placement`.

---

## 7. Chain jobs / multi-prompt video (`mold.chain.v1`)

**Authoring paths:**

- `mold run --script shot.toml` (canonical TOML) or `mold chain validate shot.toml` / `--dry-run`.
- Sugar: `mold run <model> --prompt "..." --prompt "..." --frames-per-clip 97` (uniform smooth chain).
- Auto-chain: LTX-2 distilled with `--frames > clip cap (97)` auto-splits.

**TOML schema** (`ChainScript`): top `schema = "mold.chain.v1"`; `[chain]` = model, width, height, fps, seed?, steps, guidance, strength, motion_tail_frames, output_format, enable_audio?; `[[stages]]` array. Per-stage (`ChainStage`): `prompt`, `frames`, `transition` (`smooth` default/motion-tail morph, `cut` fresh latent, `fade` cut+RGB crossfade), `fade_frames?`, `source_image` (or `source_image_path` relative to script / `source_image_b64`), `negative_prompt?`, `seed_offset?`, `model?` (per-stage override), `loras[]`, `references[]` (named refs).

**Durable jobs** (`mold jobs`, `POST /api/chain/jobs`, SSE events): survive restart, store artifacts (TTL `chain.jobs_artifact_ttl_days`).

- `jobs list [--json]`, `jobs show <id>`, `jobs resume <id>`, `jobs cancel <id>`, `jobs delete <id> [--yes]`, `jobs gc`.
- `jobs retake <id> --stage N --mode cascade|splice --seed-offset --prompt`: regenerate one stage (cascade re-renders downstream, splice replaces in place).
- Web: `createChainJob`, `listChainJobs`, `getChainJob`, `resumeChainJob`, `retakeChainJob`, `cancelChainJob`, `deleteChainJob`, `gcChainJobs`, per-stage preview URL, SSE events URL. Components: `ScriptComposer.vue`, `StageCard.vue`, `ChainJobCard.vue`, `JobsPanel.vue`. TUI has full script composer (add/reorder/delete stages, cycle transition, edit prompt/frames modals, save/load).
- Chain limits: `GET /api/chain/limits` per model; VRAM preflight estimate (worst_case_bytes, fits).

---

## 8. Queue & job lifecycle

- **Single model at a time**: `tokio::Mutex` + `spawn_blocking`. `AppState.model_cache` = LRU (max 3, `MOLD_MAX_CACHED_MODELS`) with `ModelResidency { Gpu, Parked, Unloaded }` — at most one GPU-resident.
- **Job states** (`JobLifecycle`): `Queued` → `Running`; target-GPU editable only while Queued. Queue max size (`--queue-size`/`MOLD_QUEUE_SIZE`, default 200) → 503 when full. Look-ahead/deferral tuning envs.
- Endpoints: `GET /api/queue`, `PATCH /api/queue/:id` (retarget GPU), SSE progress (`generate/stream`, includes `Queued`/step progress). Web `RunningStrip`, `RunningJobCard`, `JobsPanel`, `useQueue`, `useQueueReconciler`.
- **Generation estimate**: `POST /api/generate/estimate` (VRAM/time preflight).

---

## 9. Config & settings system

**Two stores, one `Config` view (profile-scoped):**

- `config.toml` (XDG `~/.config/mold/` or legacy `~/.mold/`): bootstrap — `default_model`, `models_dir`, `server_port`, `default_width/height/steps`, `embed_metadata`, `output_dir`, `media_roots`, `default_negative_prompt`, `[expand]`, `[logging]` (level/file/dir/max_days), `[runpod]`, `[lambda]`, `gpus`, `queue_size`, per-model `[models.<name>]` component paths + defaults (transformer/vae/encoders/steps/guidance/width/height/frames/fps/scheduler/lora/placement/etc.).
- SQLite `settings` (KV, profile-scoped) + `model_prefs` (per-resolved-model params, profile-scoped). Known keys: `tui.theme/last_model/last_prompt/last_negative/negative_collapsed/view_mode/gallery_columns`, `expand.*` (enabled/temperature/top_p/max_tokens/thinking/system_prompt/batch_prompt/backend/model/api_model/families_json), `generate.default_width/height/steps/default_negative_prompt/embed_metadata/t5_variant/qwen3_variant`, `chain.jobs_artifact_ttl_days`, `profile.active`.
- `MOLD_*` env (highest precedence) — ~120 vars.

**CLI** (`mold config`): `list [--json]` (tags rows `[db]`/`[file]`/`[env]`), `get <key> [--raw]`, `set <key> <val>` (routes by prefix — `expand.*`→DB, `models_dir`→TOML; `none` clears), `path`, `edit` ($EDITOR), `where <key>` (which surface owns it), `reset <key>|--all [--yes]` (drop DB key → fall back). `--profile <name>` global override. Multi-profile keyed on `(profile,key)`; resolves `MOLD_PROFILE`→`settings.profile.active`→`default`. One-shot idempotent config.toml→DB migration on first boot. Web `PreferencesModal.vue`, `GenerationTemplatesPanel.vue` (saved param presets); desktop `TemplatesPanel.vue` (parity — `localStorage` key `mold.desktop.generation.templates.v1`, base64 media stripped on save).

**Metadata DB**: `MOLD_HOME/mold.db` (`MOLD_DB_PATH` override, `MOLD_DB_DISABLE=1`). Tables: `generations`, `settings`, `model_prefs`, `prompt_history`, `chain_jobs`. Forward-only migrations.

---

## 10. Remote vs local modes

- **Remote** (default): HTTP to `MOLD_HOST` (default `http://localhost:7680`). SSE streaming with blocking fallback.
- **Desktop local server:** the desktop always keeps this Mac online as the app's permanent primary engine — remote servers are only ever host-list entries (Settings → Hosts), deduped by server instance UUID with hostname-based display names; a boot migration re-homes old remote-primary installs. It reuses an existing local server or starts an authenticated wildcard-bound server on port 7680 (advertised ephemeral fallback on conflict), publishes `_mold._tcp` (including an `id` instance TXT record), and exposes its persistent API key in Settings → Hosts. Every connected host's models participate in the installed-model union; the first-model screen appears only after all connected hosts report zero installed generation models.
- **Local fallback**: server unreachable → local GPU (auto-pulls missing model).
- **Forced local**: `--local` skips server (requires GPU-feature build; errors gracefully otherwise).
- **Server**: `mold serve` (`--port`/`MOLD_PORT`, `--bind`, `--models-dir`, `--log-format json|…`, `--log-file`, `--gpus`, `--queue-size`, `--discord`). Managed daemon: `mold server start|status|stop` (`--port`, `--bind`, `--models-dir`, `--log-file`). Auth via `MOLD_API_KEY`; CORS `MOLD_CORS_ORIGIN`; rate limit `MOLD_RATE_LIMIT`/`_BURST`. `GET /api/resources` + `/stream` (GPU/VRAM/CPU live), `/api/capabilities`, `/api/status`, `/health`, `/api/openapi.json`, `/api/docs` (Scalar), `POST /api/shutdown`.
- **Cloud provisioning**: the desktop RunPod workspace covers API setup, account/spend status, GPU stock, geographically labeled datacenters, full network-volume lifecycle management (create, select, rename/grow, delete), pod create/start/stop/delete, RunPod console handoff for logs, automatic refresh, and connecting the app to a pod. The volume form only offers datacenters that support persistent volumes. Volume selection persists across app launches and automatically enforces Secure Cloud plus the volume's datacenter. Network-volume pods omit unsupported stop/start actions; deleting their compute instance preserves `/workspace` on the volume. The CLI additionally provides doctor/get/usage and `runpod run` one-shot generation on fresh/warm pods. Lambda provisioning remains CLI-only (doctor/availability/deploy tunneled web UI/status/logs/tunnel/ssh/filesystems/terminate/reset).
- **Interface scaling**: ⌘+/⌘−/⌘0, the native View menu, and Settings → Appearance & app scale the entire webview from 80–130%. The preference persists, and fixed overlays plus custom context menus remain viewport-clamped at every level.
- **Native image clipboard**: right-click Copy image is available from Gallery tiles, the Gallery lightbox, and completed Generate canvases. Images are copied at full resolution; video copy remains intentionally disabled.
- **MCP server**: `mold mcp [--host]` — stdio MCP for LM Studio / MCP hosts (exposes generation tools; requires running `mold serve`).
- **Upscale** standalone: `mold upscale <img|-> [-m model] [-o] [--format png|jpeg] [--tile-size] [--host] [--local] [--preview]`; `POST /api/upscale` + `/stream`.
- **Self-update**: `mold update [--check] [--force] [--version vX]` (GitHub release, SHA256-verified; bows out on distro `/usr/bin/mold` linker collision).
- **Shell completions**: `mold completions <shell>` (bash/zsh/fish/elvish/powershell) + dynamic model-name completion.

---

## 11. CLI I/O niceties (carry into desktop where relevant)

- Pipe-friendly: stdin prompt, stdout image bytes when non-TTY; `--output -` forces stdout; `--image -` reads source from stdin. `mold run flux2-klein "cat" | viu -`.
- `--preview` (env `MOLD_PREVIEW`): inline terminal image/animation preview (viuer + Ghostty/Kitty pixel protocol; animates GIF/APNG/WebP).
- First positional disambiguation: matches known model → model, else prompt.
- Batch to files with auto-suffixed names; overwrite warnings; default filenames sanitize colons.

---

## 12. Discord bot (feature subset to mirror where cloud-shared)

Slash commands (`poise`): `/generate` (prompt, model[autocomplete], source_image attachment, video_format, frames, fps, width, height, steps, guidance, seed, strength, audio, pipeline, negative_prompt), `/expand`, `/models`, `/status`, `/quota`, admin commands (MANAGE_GUILD gated: allow/block users, etc.). Access control: block list, allowed roles (`MOLD_DISCORD_ALLOWED_ROLES`), cooldown (`MOLD_DISCORD_COOLDOWN`), daily quota (`MOLD_DISCORD_DAILY_QUOTA`) with atomic slot consume + refund on failure. HTTP-only dep on mold-core (talks to `MOLD_HOST`).

---

## 13. TUI-only niceties worth carrying over

Views: Generate, Gallery, Models, Queue, Settings, Script (composer). Actions include: live generate with progress + preview, model selector/compare, randomize seed, expand prompt inline, save/delete/upscale image, regenerate & edit-and-generate (reuse gallery item's params), open file, pull/remove/unload/filter models, gallery pan/zoom/reset/grid-nav (image viewer), prompt history nav + search, toggle negative prompt, theme selection, remembered last model/prompt/negative. Script composer: add before/after, reorder up/down, delete stage, cycle transition, prompt/frames editor modals, save/load/submit chain. Centered aspect-correct thumbnails via fixed-protocol path (Kitty/Sixel/iTerm2) — a differentiator vs naive grid rendering.
