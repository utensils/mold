# Mold Docs-Update Plan

Derived from a 13-area verified audit. **343 findings; 338 confirmed, 5 not.** No files were edited.

Path abbreviations used throughout: `wg/` = `website/guide/`, `wm/` = `website/models/`, `wa/` = `website/api/`, `wd/` = `website/deployment/`, `ws/` = `website/` (root), `dd/` = `desktop/docs/`, `da/` = `docs/architecture/`, `cr/` = `.claude/rules/`, `SKILL` = the pre-#1497 monolithic `crates/mold-cli/src/skill/SKILL.md` (since split into `crates/mold-cli/src/skill/template.md` plus the prompting corpus under `crates/mold-core/src/prompting/`; `SKILL:<line>` anchors below refer to the historical file and need re-anchoring case by case).

---

## 1. Executive summary

### Counts by severity

| Severity | Count |
|---|---:|
| wrong | 90 |
| stale | 109 |
| missing | 101 |
| inconsistent | 19 |
| nit | 24 |
| **Total** | **343** |

### Counts by area

| Area | Findings | Confirmed | wrong | stale | missing | inconsistent | nit |
|---|---:|---:|---:|---:|---:|---:|---:|
| app-docs | 59 | 57 | 8 | 25 | 22 | 1 | 3 |
| arch-docs | 34 | 34 | 8 | 20 | 3 | 1 | 2 |
| api | 33 | 33 | 10 | 2 | 20 | 1 | 0 |
| skill | 30 | 29 | 7 | 9 | 11 | 2 | 1 |
| guide-generate | 26 | 25 | 9 | 6 | 9 | 1 | 1 |
| site-misc | 26 | 26 | 8 | 9 | 4 | 3 | 2 |
| claude-md | 25 | 25 | 8 | 7 | 5 | 1 | 4 |
| guide-mobile-tui | 24 | 24 | 6 | 7 | 8 | 1 | 2 |
| guide-core | 23 | 23 | 2 | 6 | 9 | 3 | 3 |
| models-image | 19 | 19 | 13 | 2 | 2 | 1 | 1 |
| guide-ops | 17 | 17 | 6 | 3 | 3 | 1 | 4 |
| readme-docs | 14 | 13 | 1 | 11 | 2 | 0 | 0 |
| models-video | 13 | 13 | 4 | 2 | 3 | 3 | 1 |

### The 5 most consequential inaccuracies

1. **Identity/PuLID qualification is documented as a 4–6 checkpoint allowlist; it is family-wide minus one entry.** `IdentityFamily::qualifies_manifest` accepts every `family: "flux"` and every `family: "sdxl"` manifest except `sdxl-turbo:fp16`. Six documents (`wg/identity.md` ×3, `wg/generating.md`, `CLAUDE.md`, `cr/inference.md` ×2, `da/pulid-uat.md` ×3) tell users their working checkpoint is unsupported. Highest blast radius of any single fact in the corpus.
2. **Copy-paste-broken commands across the website.** `mold run --model …` (no such flag — model is positional), `sdxl:fp16` ×2, `z-image:bf16`, `"MOLD_DEFAULT_MODEL":"ltx-2"`, `--negative`, `mold runpod generate`. Each fails at the first invocation; several are the page's primary example.
3. **LTX-2 `max_frames` documented as 484 at 24 fps (advertised value is 481) and the recompute formula is off-grid.** `wa/index.md:1399/1403/1420`. A client clamping a slider to the documented maximum submits a value the server rejects with 422 — the exact failure `validation.rs`'s own comment says the grid-snap exists to prevent.
4. **Deleted endpoints documented as live, and shipped recovery documented as absent.** `dd/server-api.md` still lists `POST /api/generate/chain` and `/api/generate/chain/stream` (both deleted; only `/chain/validate` survives), while `wm/ltx2.md:841` says a failed chain returns 502 with "no partial-resume in v1" — `POST /api/chain-jobs/{id}/resume` and `mold jobs resume` both ship. Readers lose recoverable work.
5. **Model download sizes and generation defaults are wrong on the pages users size disks and pick parameters from.** SD 1.5 listed at 1.7 GB (actual 3.44 GB unet), Z-Image `:bf16` at 12.2 GB (actual 24.6 GB), Wuerstchen at 5.6 GB (actual 3.97 GB model-specific / ~12.5 GB total); SDXL's stated guidance is wrong for 5 of 8 checkpoints, its stated scheduler for 4 of 8, and Playground's step count is 50 not 25.

Runners-up worth naming in review: `wd/docker.md:243` says `flux-schnell` needs no HF token (its VAE and transformer are both `gated: true`); `ws/docs/catalog.md:32` states NSFW rows are included by default in the UI when the client explicitly sends `include_nsfw=false`; `wa/index.md:907` says `/api/queue` lists only queued and running rows, hiding the held-row contract entirely.

---

## 2. Coverage gaps and recommended follow-up

### Deliberately unverified (no in-repo oracle) — recommend a dedicated measurement pass

Every area excluded performance numbers, and they are the single largest unaudited class:

- `wg/performance.md` — **all** wall-clock benchmarks unverified.
- `wm/index.md` "Image VRAM Guide" default/eager columns; the "30–50% / 10–20%" sequential-vs-eager figures; "~148 s cold on a 4090".
- `wm/{ltx2,wan,minimax-h3}.md` — all RTX 4090 / M4 Max / upstream H100 / A6000 benchmark tables.
- `da/pulid-perf.md` §4 numbers, `wg/identity.md` timing prose.

**Follow-up:** these should be pinned to committed artifacts (`docs/qualification/*.md`, `docs/qualification/*.schema.json`) with the capture script named beside each table, so a future audit has an oracle. Where no artifact exists, mark the number with its capture date and host.

### Partially audited files

| File | What was skipped | Follow-up |
|---|---|---|
| `da/multi-gpu-scheduler-v2-design.md` (2793 ln) | Read selectively (§1, §2, §12.3–12.5, §13, §16); future-tense design text audited only where present-tense | Full pass, or add a "superseded by" banner and stop auditing it |
| `da/research-multi-model-cache-and-compute-boundaries.md` | Tail 260–528 grep-scanned, not read line-by-line | Same — banner as a 2026-04 snapshot |
| `docs/model-resolution-matrix.md` (2852 ln) | Read structurally; it is generated and CI-drift-gated, verified current at `121f144f` | None needed; **make other docs link to it instead of restating defaults** |
| `crates/mold-server/src/routes.rs` (~8k ln, skill area) | Only route table + capabilities + selected handlers read | Generate the endpoint table from the utoipa spec rather than hand-maintaining three copies |
| `dd/feature-parity.md` §6/§9 (~120 `MOLD_*` vars) | Not exhaustively diffed | Script an env-var diff (`runtime_env::ENGINE_SHAPING_VARIABLES` + clap `env =` + `std::env::var` grep) vs the three env tables |
| `da/pulid-uat.md` | Produced 3 findings but is **not** in arch-docs' stated read list | Treat as unaudited; full pass |
| `ws/.vitepress/` | Only `config.ts` audited; `theme/` and `dist/` excluded | `theme/` carries no doc claims — leave; but the link check should be re-run with `dist/` excluded and anchors resolved |

### Not audited at all / behavioural-only

- Purely visual or interaction claims with no machine-checkable source: iOS long-press system menu contents, film-grain preview wash, "44pt" gesture prose, LM Studio UI instructions, Models kind-chip row contents.
- Third-party behaviour: RunPod console + REST/GraphQL, Lambda pricing/stock, external URLs, GitHub milestone links.
- Upstream `file:line` citations into gitignored `tmp/` clones (Lightricks/LTX-2, sd.cpp, PuLID, ComfyUI, Wan) — **recommend a scripted citation checker** that clones the pinned refs and asserts each cited line still contains the cited symbol.
- Android surfaces beyond the identity plugin (`wg/android.md` is 68 lines and documents little else) — recommend a content pass, not just an accuracy pass.
- `contrib/README.md`, `packaging/` docs, `docs/qualification/*` were referenced as evidence but never audited as targets.

### Data-integrity caveat on the app-docs block

Starting at `dd/feature-parity.md:34`, the verifier notes in the app-docs area are **offset by one row** (the note attached to `:291` describes `:34`, the note on `:360` describes `:291`, and so on). The findings themselves were each independently sourced, but the `confirmed` flags in that stretch should be spot-re-checked before a fix lands. `dd/feature-parity.md:34` in particular carries a bookkeeping artifact rather than a verdict.

---

## 3. Plan — 8 PR-sized batches

**Per-PR gates for every batch:** `website/` and `desktop/` each have their own prettier gate — run the surface's own formatter, not a repo-wide one. Docs-only PRs ship with the `skip-changelog` label unless they document a shipped user-visible behaviour, in which case add a `changelog.d/<slug>.md` fragment (never edit `CHANGELOG.md`).

---

### Batch 1 — Broken commands and invalid model identifiers (P0)

Everything here fails on first execution. Smallest batch, highest user impact; land first.

**Files:** `wm/ltx2.md`, `wg/generating.md`, `wm/index.md`, `wm/z-image.md` (id only), `wd/docker.md`, `wd/runpod-cli.md`

- [ ] `wm/ltx2.md:220` — `--model ltx-2.3-22b-distilled:fp8` → model as first positional: `mold run ltx-2.3-22b-distilled:fp8 "…"`. **Derive from:** `crates/mold-cli/src/main.rs:937-943` (`Run { model_or_prompt, … }`; `--model` exists only on `RunpodAction::Run`).
- [ ] `wg/generating.md:320` — `--negative` → `--negative-prompt` (opt-out is `--no-negative`; clap does no prefix matching). **Derive from:** `main.rs:1436-1445`.
- [ ] `wg/generating.md:441,479` — `sdxl:fp16` → `sdxl-base:fp16`. **Derive from:** `manifest.rs:1788` + `resolve_model_name` (`manifest.rs:3803-3806`: a string containing `:` is returned verbatim, so no alias saves it).
- [ ] `wg/generating.md:480` — `z-image:bf16` → `z-image-turbo:bf16`. **Derive from:** `manifest.rs:2098`.
- [ ] `wm/index.md:58` — `sd35-large:q8` → `sd3.5-large:q8`. **Derive from:** `manifest.rs:1479`.
- [ ] `wd/docker.md:284` — `"MOLD_DEFAULT_MODEL":"ltx-2"` → `"ltx-2-19b-distilled:fp8"`. **Derive from:** `manifest.rs:4953` (there is no `ltx-2` family alias).
- [ ] `wd/docker.md:191` — `mold runpod generate` → `mold runpod run`. **Derive from:** `main.rs:230-418` (`RunpodAction` has no `Generate`).
- [ ] `wd/runpod-cli.md:88` — `--keep # don't park pod for reuse` → `# keep the pod running (skips auto-teardown / idle reap)`; the comment inverts the flag and contradicts the page's own step 7. **Derive from:** `main.rs:396-398`, `commands/runpod.rs:1882-1900`.

---

### Batch 2 — Identity / PuLID qualification, stated once (P0, cross-tier)

**One fact, six documents.** State it once in `wg/identity.md` and make every other site link there. Deliberately crosses the user/internal boundary because splitting it would leave contradicting copies in the tree.

**Files:** `wg/identity.md`, `wg/generating.md`, `wg/configuration.md`, `CLAUDE.md`, `cr/inference.md`, `da/pulid-uat.md`

**Canonical wording** (put in `wg/identity.md`, link from all others): *"Every FLUX.1 checkpoint takes PuLID-FLUX v0.9.1; every SDXL checkpoint except `sdxl-turbo:fp16` takes PuLID v1.1. Qualification is derived from the manifest family, so a new FLUX/SDXL entry inherits it automatically."*

- [ ] `wg/identity.md:29` — "enumerated list, not a family match" → family-wide with one denylist entry.
- [ ] `wg/identity.md:32` — delete the six-row "Which models" table (or relabel *examples*); 7 of 8 SDXL manifests qualify.
- [ ] `wg/identity.md:41` — delete the three exclusion bullets; `playground-v2.5:fp16`, `pony-v6:fp16`, `cyberrealistic-pony:fp16` all qualify. Replace with the Turbo-only exclusion and its reason (PuLID v1.1's own release note on the distilled base).
- [ ] `wg/identity.md:361` — licenses row → "Any SDXL checkpoint except `sdxl-turbo:fp16`".
- [ ] `wg/generating.md:199` — replace the six-model enumeration with a link to `wg/identity.md`.
- [ ] `wg/configuration.md:384` — name both bundles (`pulid-flux`, `pulid-sdxl`) sharing one `insightface-antelopev2` acceptance.
- [ ] `CLAUDE.md` PuLID bullet — replace the enumerated SDXL list and the "turbo / Playground / two Pony derivatives are refused" claim.
- [ ] `cr/inference.md:48` and `:59` — same replacement; the code comment (`identity.rs:262`) states the opposite of the current doc text.
- [ ] `da/pulid-uat.md:191,201,501` — annotate the transcript and the `supports_identity` rows as captured pre-#1228; row 24's exclusion table is now wrong for three checkpoints.
- [ ] `wg/generating.md:221` — the identity well is on web, desktop **and iPhone**. **Derive from:** `desktop/src/mobile/MobileApp.vue:435,1990,11582`.

**Derive all of the above from:** `crates/mold-core/src/identity.rs:260-292` (`qualifies_manifest`, `IDENTITY_EXCLUDED_SDXL_MODELS`, `identity_qualified_models`) and the pin test at `identity.rs:1435-1480`. Never from another doc.

---

### Batch 3 — HTTP API reference (`wa/index.md` + `dd/server-api.md`)

These two files document the same router and disagree with it in the same places. Fix together; make `dd/server-api.md` a thin delta over `wa/index.md` rather than a second table.

**Files:** `wa/index.md`, `dd/server-api.md`

**Correctness (do first):**
- [ ] `wa/index.md:1399/1403/1420` — `max_frames` 484 → **481** at 24 fps, 124 → **121** at 6 fps; recompute rule is `min(seconds·fps+4, max_frames_absolute)` **rounded down onto the frame grid**. **Derive from:** `validation.rs:218-248,389-395`; `catalog.rs:1005`.
- [ ] `wa/index.md:1056/1080` — the SSE terminal frame is `SseCompleteEvent` (single base64 `image`), not `GenerateResponse`/`images[]`. **Derive from:** `types.rs:4511-4560`.
- [ ] `wa/index.md:1867` — upscale complete carries `format`/`original_width`/`original_height`/`upscale_time_ms`, no `width`/`height`. **Derive from:** `types.rs:4610-4619`.
- [ ] `wa/index.md:1777` — `variations` max 10 → **10,000**. **Derive from:** `expand.rs:35`.
- [ ] `wa/index.md:1325/1357` + `SKILL:1443` — `strength` **is** amendable and **is** a chain-level invalidator. **Derive from:** `chain_job.rs:738`, `chain_job_runner.rs:2558,2570`.
- [ ] `wa/index.md:1360` — add the **LoRA stack** to the per-stage dirty tuple. **Derive from:** `chain_job_runner.rs:2573`.
- [ ] `wa/index.md:907` — `/api/queue` lists queued, running **and held** rows; document `held_reason`/`error`/`retryable`/`dispatch_attempts`. **Derive from:** `job_registry.rs:49-56,111-125`.
- [ ] `wa/index.md:218/214` — read tier is "every GET plus anything unclassified, ×10 capped at 1000/period"; gallery `DELETE`, both placement-previews, queue retry/sweep, and `PATCH /api/devices/*` are **generation** tier. **Derive from:** `rate_limit.rs:109-207`.
- [ ] `dd/server-api.md:222` — live denoise previews exist (`SseProgressEvent::Preview`, `GET /api/queue/:id/preview`). **Derive from:** `types.rs:4457`, `routes.rs:764`.
- [ ] `dd/server-api.md:248` — CORS allows GET/HEAD/POST/**PATCH**/**PUT**/DELETE. **Derive from:** `lib.rs:1656-1663`.
- [ ] `dd/server-api.md:16,17,220` — delete `/api/generate/chain` and `/api/generate/chain/stream`; `routes_chain.rs` holds only `/chain/validate`; ephemerality is `ChainRequest.ephemeral`.

**Completeness (single sweep, generate rather than hand-list):**
- [ ] `wa/index.md:9` and `dd/server-api.md:91` — add the ~30 missing routes: the 11 gallery organization/trash routes, both placement-previews, the 3 reference-upload routes, `GET /api/queue/:id`, `DELETE /api/queue`, queue pause/resume, `DELETE /api/models/:model`, `/api/activity`, `/api/history`, `/api/licenses`, `/api/config*`, `/api/pairing/*`, `/api/catalog/credentials*`, `ltx2-control-adapters`/`ltx2-camera-controls`, batch events + `DELETE`, chain operation cancel, gallery export/import. **Derive from:** `routes.rs:514-808` (`create_router`) — ideally emit the table from the utoipa spec.
- [ ] `wa/index.md:1127` + `dd/server-api.md:75` — add the 8 missing `ServerEvent` types (`job_state_committed`, `generation_states_committed`, `gallery_updated`, `gallery_trashed`, `gallery_restored`, `gallery_collections_changed`, `queue_paused`, `queue_resumed`, plus `chain_job_*`). **Derive from:** `types.rs:10446-10548`.
- [ ] `wa/index.md:100` + `dd/server-api.md:82` — add `gallery`, `durable_media`, `expand`, `reference_uploads`, `model_access`, `minimax_h3`, `generation_profile_v1`, and **`identity`** (`{multi_photo, max_photos, true_cfg}`, absence reads as no). **Derive from:** `routes.rs:7109-7185`, `types.rs:9705-9758`.
- [ ] `wa/index.md:1395` — add `supports_audio`, `supports_sequence`, `max_pixels`, `max_axis_pixels`, `recommended_dimensions`, `generation_profile`, `supports_duration_prediction`, `runtime_ready`. **Derive from:** `types.rs:2884-3069`.
- [ ] Remaining rows for both files → Appendix A (auth exemption `/api/pairing/claim`; `gallery_removed` vs `gallery_trashed`; `DELETE`-is-trash; queue `limit`/`cursor`/`live_only_entries`/`page`; `paused` child + chain state; video response headers; `/api/status.instance_id`; config `409 RESTART_REQUIRED` / `404` / `restart_required`; `provenance.crop`; thumbnail `?size=&fmt=`; `Collection.hidden`; `host_memory.reclaimable_zfs_arc_bytes`; conditional GET + `media_version`; `Ltx2ControlAdapterInfo.gated` + `?detail=`).

---

### Batch 4 — Model reference pages: sizes, defaults, presets, sources

**Files:** `wm/{index,sd15,sdxl,sd35,z-image,wuerstchen,upscalers,flux,flux2,ltx-video,ltx2,wan,minimax-h3}.md`

**Rule for this batch:** stop restating per-model defaults. `docs/model-resolution-matrix.md` is generated from `generation_profile.rs` and CI-drift-gated; link to it and keep only the family-level prose.

- [ ] `wm/sd15.md:20` — all three rows 1.7 GB → **3.4 GB**. **Derive from:** `manifest.rs:1645,1678,1711`.
- [ ] `wm/z-image.md:18` — `:bf16` 12.2 GB → **24.6 GB** (3 shards). **Derive from:** `manifest.rs:2102-2135`.
- [ ] `wm/wuerstchen.md:23` — 5.6 GB → **3.7 GB** prior transformer / ~12.5 GB total cascade. **Derive from:** `manifest.rs:3723-3788` + `model_size_bytes()`.
- [ ] `wm/flux.md:44` — `ultrareal-v3:q4` 7.5 GB → **6.8 GB**. **Derive from:** `manifest.rs:1220-1234`.
- [ ] `wm/sdxl.md:20,27,28,29` — Playground steps 25 → **50**; guidance is per model (7.5 / 7.0 ×3 / 3.0 / 2.0 / 0.0); scheduler is per model (DDIM ×3, euler-ancestral ×4, `edm-dpm-pp-2m` ×1); resolution 1024² except `sdxl-turbo:fp16` = 512². **Derive from:** `manifest.rs` `sdxl_manifests()` + `docs/model-resolution-matrix.md`.
- [ ] `wm/index.md:22` — `sdxl-turbo:fp16` "1024×1024" → 512×512 default.
- [ ] `wm/index.md:69` — `--offload` is ~24 GB → 2–4 GB, **3–5×** slower, and covers FLUX, Flux.2, Z-Image, Qwen-Image, SD3, LTX-2, Wan. **Derive from:** `main.rs:1306-1309`.
- [ ] `wm/index.md:60` — `qwen-image-2512:q4` "validated at 1024" → **1328**, matching `wm/qwen-image.md:70`.
- [ ] `wm/z-image.md:6` — developer/repo `Z-Potentials` → **`Tongyi-MAI/Z-Image-Turbo`** (+ `leejet/Z-Image-Turbo-GGUF` for quantized tiers). **Derive from:** `manifest.rs:2049-2221`.
- [ ] `wm/z-image.md:28` — replace the 3-row table with the **11** real presets; `1024x768` currently triggers the dimension warning the page claims to avoid. **Derive from:** `generation_profile.rs:801-813`.
- [ ] `wm/upscalers.md:11,71` — add `real-esrgan-x4plus-anime:fp32` and `real-esrgan-x2plus:fp32`; sources are `hlky/RealESRGAN_*` and `wkrettek/real-esrgan-models`, **not** `Comfy-Org/Real-ESRGAN_repackaged`. **Derive from:** `manifest.rs:6721-6825`.
- [ ] `wm/sd35.md:57,61` — add the `[gated]` note (all SD 3.5 shared files are `gated: true`); scope or drop the BF16 `--offload` sentence (no BF16 SD 3.5 ships). **Derive from:** `manifest.rs shared_sd3_files()`, `sd3/pipeline.rs:77-95`.
- [ ] `wm/flux2.md:41` — link `black-forest-labs/FLUX.2-klein-4B` (+ `unsloth/FLUX.2-klein-4B-GGUF`).
- [ ] `wm/ltx2.md:841` — replace "502, nothing written, no partial-resume" with the durable-job reality (`POST /api/chain-jobs/{id}/resume`, `mold jobs resume`; ephemeral auto-chains refuse resume). **Derive from:** `chain_job.rs:98`, `routes.rs:579`.
- [ ] `wm/ltx2.md:704,834` — only the **audio-only** pipeline declines chaining; IC-LoRA HDR chains (the same page says so at :239). **Derive from:** `commands/chain.rs:119-127`, `commands/generate.rs:1145-1262`.
- [ ] `wm/ltx2.md:29` — drop the "On Apple Silicon" qualifier from the bare-name `:int8-conv` default. **Derive from:** `manifest.rs:3828-3832`.
- [ ] `wm/wan.md:364` — `:fp8` routes at **53** frames (`max(tier_default, floor)`), not 45. **Derive from:** `chain.rs:729-737`.
- [ ] `wm/minimax-h3.md:292` — reference duration 1–15 s → **2–15 s**. **Derive from:** `minimax_h3.rs:554,1487-1494`. *(Same fix in `SKILL:11` — Batch 7.)*
- [ ] `wm/minimax-h3.md:268` — "the hidden official BF16 reference" → "the download-only official BF16 references" (they are `hidden: false`, as the page says at :51).
- [ ] `wm/ltx-video.md:85` — add the 3 missing presets (704×480, 768×768, 576×1024). **Derive from:** `generation_profile.rs:825-834`.
- [ ] Missing flags/notes: `wm/ltx2.md:898` `--video-only`; `wm/ltx2.md:160` `--predict-duration`; `wm/minimax-h3.md:333` uploaded references force `--batch 1`.

---

### Batch 5 — CLI reference + configuration guide (+ their `dd/feature-parity.md` twins)

**Files:** `wg/cli-reference.md`, `wg/configuration.md`, `dd/feature-parity.md` (§1 request fields, §CLI extras, §config keys, §env)

**Cross-doc facts to state once:**
- *Queue size is the hydrated runtime window; the durable backlog is uncapped* → canonical in `wg/configuration.md`, linked from `wg/cli-reference.md:303` and `wd/nixos.md:179`.
- *Config lives at `$MOLD_HOME/config.toml` (default `~/.mold/`); `~/.config/mold/home` is a pointer file only* → canonical in `wg/configuration.md:5`; `CLAUDE.md:115` is the one that is wrong.

- [ ] `wg/cli-reference.md:418` + `wg/configuration.md:12` — `umt5_variant` is registered but has **no** get/set arm and **no** DB slot; it is `Surface::File`, not DB-owned. Footnote #778 or drop it. **Derive from:** `config_keys.rs:108-113,489-568,643-670,903-928`.
- [ ] `wg/cli-reference.md:88` + `wg/configuration.md:198` + `dd/feature-parity.md:34` — the `Scheduler` enum is `ddim`, `euler-ancestral`, `uni-pc`, `edm-dpm-pp-2m` (Playground only), plus wan-only `euler`/`dpm-pp`. There is no `default` variant and no `unipc` spelling. **Derive from:** `types.rs:134-165`.
- [ ] `wg/cli-reference.md:303` + `wg/configuration.md:242` — reword `--queue-size` / `MOLD_QUEUE_SIZE` per the cross-doc fact above. **Derive from:** `main.rs:1511-1513`.
- [ ] `wg/configuration.md:450,454` — three surfaces write gallery rows (server, CLI, **TUI**); `source` has five values (`server|cli|tui|backfill|unknown`). **Derive from:** `mold-db/src/record.rs:7-46`, `mold-tui/src/app.rs:8548`.
- [ ] `wg/cli-reference.md:424` — add the `Scheduler`, `Queue`, and `Generate` key sections. **Derive from:** `config_keys.rs:175-214,670-700`.
- [ ] `wg/configuration.md:15` — add `default_frames`, `default_fps` to the per-model DB slice. **Derive from:** `config.rs:49-60`, `config_sync.rs:321-358`.
- [ ] `wg/configuration.md:260` — add `MOLD_HOST_RAM_ZFS_ARC`. **Derive from:** `zfs_arc.rs:37`.
- [ ] `wg/cli-reference.md` missing flags/commands: `--video-only` (:48), `--hdr-exr-dir`/`--hdr-exr-full-float` (:65), the whole Identity heading `--id-image`/`--id-weight`/`--id-start-step`/`--true-cfg`/`--cfg-start-step` (:92), `mold pull --skip-verify`/`--accept-license` (:398), `mold licenses [--local]` (:525). **Derive from:** `main.rs` `Run`/`Pull`/`Licenses` variants.
- [ ] `wg/cli-reference.md:541` — the requires-a-server table must include `queue` and `library` (the page's own prose at :151 and :185 already says so).
- [ ] `wg/cli-reference.md:384` — MCP exposes **nine** tools; add job status and retry. **Derive from:** `commands/mcp.rs`.
- [ ] `wg/cli-reference.md:517,368` — `mold clean` is dry-run by default; `--probe` hits `/health` **and** `/api/status`.
- [ ] `wg/configuration.md:203` ↔ `SKILL:1533` — the `MOLD_LTX2_GEMMA_VARIANT auto` rule is memory-aware on the server path and presence-only on the forced-local fallback; SKILL is the stale half. **Derive from:** `variant_dependencies.rs:1193-1215`, `ltx2/text/gemma.rs:513-520`.
- [ ] `dd/feature-parity.md:21,68` — add the missing `GenerateRequest` fields (`title`/`tags`/`collection`, `extend_video*`, `references`, `ic_lora_control`, `id_images`/`true_cfg`/`cfg_start_step`, `source_fit`, `video_only`, `hdr_exr_*`, `sample_shift`, `distill_strength_*`) and the ~26 missing `mold run` flags. **Derive from:** `types.rs` `GenerateRequest`, `main.rs:937-1470`.
- [ ] `dd/feature-parity.md:306,307,321` — config.toml gains `config_version`/`[scheduler]`/`[gallery]`/`[queue]`/`[generate]`; DB keys gain `queue.held_retention_days`/`generate.auto_tag_title`/`tui.*`; `mold.db` has **21** tables at `SCHEMA_VERSION 31`, not 5.

---

### Batch 6 — TUI, iPhone, Android guides (+ the two internal TUI docs)

Key-map facts are duplicated in `wg/tui.md`, `cr/tui.md`, and `.claude/skills/tui-uat/SKILL.md` and are wrong in the same places. Fix as one unit.

**Files:** `wg/tui.md`, `wg/iphone.md`, `wg/android.md`, `wg/desktop.md` (mobile default only), `cr/tui.md`, `.claude/skills/tui-uat/SKILL.md`, `SKILL:1694`, `dd/feature-parity.md:312`

**Cross-doc fact to state once:** *fresh mobile installs default to Safelight + **Dark**, with Photos auto-save and `autoTagTitle` on* — canonical in `wg/iphone.md`; `wg/desktop.md:390`, `SKILL:1694`, and `dd/feature-parity.md:312` all restate it wrongly. **Derive from:** `desktop/src/mobile/settings.ts:17-22`.

**Keymap corrections** — derive every one from `crates/mold-tui/src/event.rs` (`map_machines_key`, `map_models_key`, `control_shortcut`) and `action.rs`:
- [ ] `wg/tui.md:427` + `tui-uat/SKILL.md:167` — `d` = disconnect/reconnect (`MachinesToggleConnection`); **`f`** = forget host. Currently documented as the opposite; a UAT script following it toggles a connection instead of forgetting.
- [ ] `wg/tui.md:573` — workspace cycling is **Alt+**Left/Right, globally; plain arrows are unbound in nav mode.
- [ ] `wg/tui.md:298` — remove the `Ctrl+E` end-of-line row; it is bypassed to Expand.
- [ ] `tui-uat/SKILL.md:157` — `Ctrl+R` = **randomize seed** (the same file says so at :106).
- [ ] `wg/tui.md:399` add `r`/`/`; `:432` and `tui-uat:167` add `l`, `g`, `[`, `]`, `e`; `tui-uat:157` add `Ctrl+E`, `Ctrl+Shift+E`, `Ctrl+S`, `Ctrl+T`, `Ctrl+P`, `Ctrl+N`.
- [ ] `wg/tui.md:400` — Enter on Models selects into Create, it does not set the persisted default.

**TUI content:**
- [ ] `wg/tui.md:285,608` — prompt history and session state live in `$MOLD_HOME/mold.db`; the legacy JSON/JSONL files are imported once and renamed `.migrated`. **Derive from:** `history.rs:7,207`, `session.rs:1-8`.
- [ ] `wg/tui.md:507,508` — drop "(consumed by upcoming releases)" from Reduce Motion and Show Timeline; both are read today. **Derive from:** `motion.rs:53-84`, `ui/generate.rs:72`.
- [ ] `wg/tui.md:107` + `tui-uat:161` + `cr/tui.md:11` — video models add **Duration** (and **Predict duration** where advertised); the Advanced accordion has an **Identity photo** section and a Wan `SampleShift` row; `tui-uat:161`'s landmark list also omits **File under**. **Derive from:** `ui/create_form.rs:30-55,153-294`.
- [ ] `wg/tui.md:153` — add `References` to the Source image section.
- [ ] `wg/tui.md:134` — Size cycles the model's authored profile presets; the 5-ratio computation is only the no-profile fallback.
- [ ] `wg/tui.md:96,51,661` — palette also has *Connect a machine* and *Retry held prints*; add `mold-tui.YYYY-MM-DD.log`; the "All features" tab omits `webp,mp4,metrics,mdns,pulid`.
- [ ] `tui-uat/SKILL.md:167` — `x` cancels queued **or running** work on hosts advertising `queue.cooperative_cancellation`.

**Mobile:**
- [ ] `wg/iphone.md:87` — one-shot generation no longer issues a placement preview; automatic routing ranks from cached telemetry and freezes. Placement preview is the placement planner. _(Scene authoring was retired; there is no sequence path on the phone.)_ **Derive from:** `studio/lib/generationSubmissionPolicy.ts:60-81`, `desktop/src/mobile/mobileGenerationRouting.ts:187-310`.
- [ ] `wg/iphone.md:179` — canonical families include **5:4 / 4:5**. **Derive from:** `studio/lib/outputShape.ts:139-151`.
- [ ] `wg/iphone.md:24` — button is **Create pairing code**, not "Pair an iPhone".
- [ ] `wg/iphone.md:414,416` — add the **Model licenses** section; rename "GPUs" to **Compute devices**.
- [ ] `wg/android.md:11` — add the tagged-release `Mold-android.apk` + `SHA256SUMS` channel beside the nightly.
- [ ] `CLAUDE.md` (seam labels) — "Continue motion / Cut / Crossfade" → **Smooth / Cut / Fade** (Join at zero motion tail). `wg/iphone.md:451` is correct. **Derive from:** `ui/lib/seam.ts:16-25`.

---

### Batch 7 — Deployment, site-misc, nav, ops/desktop guides, README, SKILL

**Files:** `wd/{docker,nixos,runpod-cli,index}.md`, `ws/{index,privacy}.md`, `ws/docs/catalog.md`, `ws/api/discord.md`, `ws/.vitepress/config.ts`, `wg/{desktop,feature-matrix,performance,troubleshooting,custom-models,remote-workflows}.md`, `README.md`, `SKILL`

**Cross-doc facts to state once:**
- *Windows is a full desktop target; the published Windows binary is CPU/remote-client only, from the `Windows Nightly` workflow* → canonical in `wg/desktop.md`; `wg/feature-matrix.md:162`, `README.md:88`, `dd/architecture.md:3` link.
- *Android ships from the same remote-only crate and Vue surface; signed universal APK on every nightly and tag* → canonical in `wg/android.md`; `ws/index.md`, `wg/desktop.md:669`, `SKILL:1711`, `docs/design/README.md` link.
- *Video families default to MP4 in release builds; APNG is the no-`mp4` fallback* → canonical in `wg/generating.md`; `SKILL:288,291,347` link. **Derive from:** `types.rs:1738-1743`, `commands/generate.rs:2206-2214`.
- *Upscaler catalog is 7 models* → canonical in `wm/upscalers.md`; `wg/upscaling.md:23` and `SKILL:1063` link.

**Broken links / nav:**
- [ ] `wg/performance.md:127` + `wg/troubleshooting.md:183` — `#performance-knobs` anchor does not exist. Either retarget to `/guide/configuration#generation` or add the sub-heading in `wg/configuration.md` above the `MOLD_KEEP_TE_RAM`…`MOLD_ATTN_CHUNK` rows (preferred — two docs link it).
- [ ] `ws/.vitepress/config.ts:141` — add `/models/ltx-video` to the Models sidebar (currently reachable only from an inline link).
- [ ] `ws/index.md:66` — add an Android feature card (the hero already promises Android).

**Wrong:**
- [ ] `wg/desktop.md:684` — `desktop-windows` never runs on pull requests (`if: github.event_name != 'pull_request'`); it runs after merge on `main` / manual / nightly. **Derive from:** `.github/workflows/desktop.yml:229-236`.
- [ ] `wg/desktop.md:262` — desktop default theme family is **Safelight** (default mode Dark). **Derive from:** `desktop/src-tauri/src/settings.rs:283,541`.
- [ ] `wg/desktop.md:59` — the rolling `latest` Windows assets come from `Windows Nightly`; `Desktop` only keeps a 14-day CI artifact.
- [ ] `wg/custom-models.md:146` — LoRA family refusal is **422 `VALIDATION_ERROR`**, not 400. **Derive from:** `routes.rs:73-75`.
- [ ] `wd/docker.md:227` — `MOLD_DEFAULT_MODEL` default is `flux2-klein:q8` (baked into the image ENV).
- [ ] `wd/docker.md:243` — **`flux-schnell` is gated**; its VAE and transformer are both `gated: true`. Remove it from the "no token" list.
- [ ] `wd/nixos.md:52` — Cachix publishes `mold`, `mold-sm86`, `mold-sm100`, `mold-desktop-sm86` on tags. **Derive from:** `.github/workflows/nix-cache.yml:24-67`.
- [ ] `wd/nixos.md:254` — the HF token **is** in the service process environment (runtime-only 0600 `EnvironmentFile`); what it never enters is the Nix store.
- [ ] `ws/docs/catalog.md:32` — the UI **excludes** NSFW by default (client sends `include_nsfw=false`); only the raw API defaults a missing parameter to true. **Derive from:** `web/src/composables/useCatalog.ts:85-90`, `catalog_api.rs:877`.
- [ ] `SKILL:479` — `--motion-tail` default **17**, not 4. **Derive from:** `main.rs:1041`, `chain.rs:632`.
- [ ] `SKILL:1459` — `DELETE /api/queue/:id` has no 409 arm; running work cancels cooperatively (204/404 only).
- [ ] `SKILL:1185` — catalog credentials live in owner-only `$MOLD_HOME/catalog-credentials.json`, **not** `mold.db` settings keys. **Derive from:** `catalog_credentials.rs:1-25`.
- [ ] `SKILL:703` — `flux2-klein` guidance **1.0**, not 0.0.

**Stale / missing (representative; full rows in Appendix A):**
- [ ] `wd/runpod-cli.md:50` — `mold runpod run` **does** show per-step and model-pull progress via the durable observer.
- [ ] `wd/runpod-cli.md:143` + `SKILL:1352` — the GPU preference ladder has 10 entries (CLI) / 6 (`GPU_PREFERENCE`); both docs list 4 and omit RTX A6000 and L40. Reconcile against `runpod.rs:888-896` and `commands/runpod.rs:1152-1195` and state which list belongs to which code path.
- [ ] `wd/{docker,nixos,runpod-cli}.md` (×4 sites) — `/` opens Mold Studio **Create**; the gallery is `/library`.
- [ ] `wd/index.md:33` — user-mode systemd uses `contrib/mold-server.user.service` + `loginctl enable-linger`.
- [ ] `wd/nixos.md:190` — add `metadataDb.enable`, `metadataDb.path`, `runpodApiKeyFile`.
- [ ] `ws/docs/catalog.md:8,52,128` — there is no Catalog view on any surface (it is **Models ▸ Discover**); `MOLD_CATALOG_HF_BASE`/`_CIVITAI_BASE` no longer exist.
- [ ] `ws/privacy.md:6,17` — broaden scope to Android + desktop (both link this URL); document the Android Keystore storage beside the iOS Keychain; the product name is **Mold**, not "Mold Remote".
- [ ] `ws/api/discord.md:30` — document `reference_1` / `reference_2` (ordered H3 references) and their exclusivity.
- [ ] `wg/feature-matrix.md:72,187` — Wan **does** take a scheduler override (`--sample-solver unipc|euler|dpm++`, same wire slot).
- [ ] `wg/desktop.md:266,448,504,581,669` + `wg/remote-workflows.md:39` — add Saved media/Library accordion sections; **Add machine** not "Add host"; **Appearance & app** not "App"; `windows.ps1` also takes `bundle`/`clean`/`features`; add the seven `android-*` devshell commands; refresh the sample version to 0.26.0.
- [ ] `README.md:88` — add the Windows desktop installer and the Linux AppImage pointer.
- [ ] `SKILL` structural gaps: routing list missing 8 subcommands (:60); no `mold library` section (:583); no `mold remix` (:104); no `mold stats`/`mold clean` usage (:1073); Available Models missing **Wan**, **Qwen-Image-Edit**, **MiniMax H3** (:715) and the LTX-2.5/2.3-bf16 tiers (:747, contradicting :410); ~~`/sequence` missing (:1618)~~ (command retired); no Lambda section (:1248); env table missing 7 vars (:1488); `MOLD_LOG` default is `info` for the server (:1499); `server start --log-file` is on by default (:1383); reference duration 2–15 s (:11); FP8→Q8 auto-conversion is **FLUX-only** (:769, :1590).

---

### Batch 8 — Internal architecture, design, and agent docs

Mostly "mark superseded / repoint symbols". Low user risk, high agent-confusion risk. Land last, but land it — several of these actively misdirect a future implementer.

**Files:** `CLAUDE.md`, `cr/{inference,server-queue,studio-web,minimax-h3,desktop,tui}.md`, `da/{candle-extension,qwen-mmq-nan,pulid,pulid-face-extraction,pulid-perf,pulid-uat,wan-comfyui-parity-ledger,research-multi-model-cache-and-compute-boundaries,multi-gpu-scheduler-v2-design,minimax-h3-authorization}.md`, `dd/{architecture,feature-parity}.md`, `docs/{ltx-2.5,feasibility-recovery-plan,generate-studio-console-followups}.md`, `docs/design/README.md`, `docs/design/notes/activity-history-and-sequence-reuse.md`

**Rule glob fixes (these silently disable rules — do first):**
- [ ] `cr/studio-web.md:8` — `crates/mold-server/src/generation_profile*` matches **nothing**; the file is `crates/mold-core/src/generation_profile.rs`. The rule never loads when its own cited authority is edited.
- [ ] `cr/minimax-h3.md:7` — `crates/mold-server/src/private_*` matches nothing (all `private_*.rs` are under `crates/mold-inference/src/minimax_h3/`); `:5`'s `**/h3/**` matches no directory.

**`CLAUDE.md` / rules corrections:**
- [ ] `CLAUDE.md:53` — `mold-inference` depends on `mold-ai-catalog`, so `mold-tui` transitively does too. Only `mold-discord` is clean. Either correct the invariant or fix the dependency.
- [ ] `CLAUDE.md:115` — config path (see Batch 5 cross-doc fact).
- [ ] `CLAUDE.md:137` — model cache LRU is `MOLD_MAX_CACHED_MODELS` (default 3, range 1–16).
- [ ] `CLAUDE.md:34` — `check:frontend` has no lint step.
- [ ] `CLAUDE.md` release bullet — the rolling `latest*` container aliases come from the **schedule**, not main pushes (`release.yml:665-671`; `wd/docker.md:38` is correct).
- [ ] `cr/inference.md:12` — feature list omits `nvml`, `h3`, `h3-cuda`, `pulid`, `mdns`; `:50` — the "whichever issue first ships `pulid` must add protobuf" clause is done and shipped; `:61` — rejoin the broken `` `mold rm pulid-flux` `` code span.
- [ ] `cr/server-queue.md:14` — qualify the H3 conditioner sentence to the `HostCpuThenDrop` arm; `cr/minimax-h3.md:15` already documents the #1423 device fit and the two rules contradict.
- [ ] `cr/server-queue.md:27` — `GalleryCapabilities` has 7 fields; add `mold library` to the CLI surfaces.
- [ ] `cr/desktop.md:12` — there is no `desktop-bun-lock`; it is repo-root `frontend-bun-lock`. Add `desktop-release`.

**PuLID docs (`da/pulid*.md`) — one superseding change, applied consistently:** #1227 phase 2 moved extraction **inside the lease, onto the render's device, as `ProgressPhase::IdentityExtract`**, retiring `ExtractionSlot`.
- [ ] `da/pulid.md:396,410,413,420,450,484,493,553` — repoint to `mold_server::identity_extraction::resolve_identity_for_lease`; delete the `batch_runtime::submit_child` reference (that substrate is deleted); host peak is **2.4 GB** not 1.4 GB (×3 sites); add the additive `EXTRACTION_DEVICE_PEAK_BYTES` (700 MB) device row; `IdentityAssetDigests` has **five** digests not four; `pulid_paths`/`missing_pulid_files` → `*_for(family)`.
- [ ] `da/pulid-face-extraction.md:201,207,280,690` — `IdentityExtractor::load` honours a device (no CPU assertion); `pulid` **already ships** in every release recipe with `pkgs.protobuf` in crane and CI; `pulid_manifest()` → `pulid_manifest_for(family)`; add `scrfd_net.rs`, `arcface_net.rs`, `onnx_weights.rs`, `extraction.rs` to the Files table.
- [ ] `da/pulid-perf.md:165,270,286,292` — annotate §1/§2 as superseded; `IDENTITY_PIPELINE_VERSION` exists; five digests not four.
- [ ] `da/pulid-uat.md` — handled in Batch 2.

**Candle pinning (`da/candle-extension.md`, `da/qwen-mmq-nan.md`):**
- [ ] `candle-extension.md:76` — the identity script **permits** `version` beside `git`+`rev`; it bans `branch`/`tag`/`path`. The doc inverts its own guard.
- [ ] `candle-extension.md:87,104` + `qwen-mmq-nan.md:97` — there is no Candle `[patch]` (root patches only `cudarc`); every candle dep is a bare `git`+`rev` on the **renamed** `*-mold` packages at `5de41be79c45b6b82f8da0f8efd1b6ed11bb91b4`. Restate the pin as a revision and drop the "resolve from crates.io" claim. **Derive from:** `Cargo.toml:63-65`, `crates/mold-candle/Cargo.toml:37-39`.

**Historical plan documents — banner rather than rewrite:**
- [ ] `da/multi-gpu-scheduler-v2-design.md:5,1566,1589` — Status line; §12.2–12.5's `GalleryPublicationGate` / `.mold-batch-transactions` / `PlannedBatchPartition` were **replaced**, not implemented. Point at `mold_db::generation_batches` and CLAUDE.md's durable-queue invariant.
- [ ] `da/research-multi-model-cache-and-compute-boundaries.md:33,75,157,220` — banner as a 2026-04 snapshot: `AppState` fields, the "LoRA locked to sequential" claim, the candle 0.9.3 version, and the "add a `Parked` tier" proposal are all superseded.
- [ ] `docs/feasibility-recovery-plan.md:3` — pull-and-resume **shipped** (#1162); narrow the not-implemented list to companion-component repair.
- [ ] `docs/generate-studio-console-followups.md:10,22,83,323` — all nine "Likely files" are gone; capability map is `studio/lib/`; Catalog → Models ▸ Discover; verification command misses the root studio gates. Mark the whole doc a closed historical backlog.
- [ ] `docs/design/notes/activity-history-and-sequence-reuse.md:166,436,627` — `RAIL_SETTLED_KEEP` 3 → `GENERATION_HISTORY_LIMIT` 50; `routes_chain.rs` → `ChainRequest.ephemeral`; the amend route is registered.
- [ ] `docs/design/README.md:3,18` — six surfaces (desktop is macOS/Linux/Windows; Android ships); the TUI already carries the five-workspace IA — only the restyle remains.
- [ ] `docs/ltx-2.5.md:79,153` — gate is `supports_duration_prediction === true && runtime_ready !== false` (absence passes); add the CUDA qualification harness scripts + schema.

**`dd/architecture.md` / `dd/feature-parity.md` residue:**
- [ ] `dd/architecture.md:3,127,143,310` — add Windows as a full target; feature set is `expand`+`mdns` always, `mp4` non-Windows, `metal`/`cuda` opt-in; LTX-2 Metal is **performance-qualified** (also `dd/feature-parity.md:103`).
- [ ] `dd/architecture.md:190,199,236,242,280,361,449,487,109` — refresh CSP, `minWidth` 1080, the IPC command list (five named commands do not exist), the four `core:window:*` permissions, drop `@testing-library/vue`, refresh the Cargo sketch to 0.26.0, repo-relative paths, both workspace excludes.
- [ ] `dd/feature-parity.md:84,88,110,182,188,253,263,264,291,292,310,321,354,360,72` — add `wan`/`minimax-h3` to the taxonomy and matrix; fix `/api/chain/jobs` → `/api/chain-jobs` and `/api/chain/limits` → `/api/capabilities/chain-limits`; `ModelResidency { Gpu, Parked }`; TUI views are the five workspaces; repoint the six deleted component paths; `JobLifecycle` has four states; `sourceFit*` moved to `studio/lib/`.
- [ ] `da/minimax-h3-authorization.md:83` — the authorized listing scope now covers three reviewed Turbo tags plus the download-only pinned tiers, not "the two compact Comfy manifests".
- [ ] `da/wan-comfyui-parity-ledger.md:126` — drop `extend_video` from the R3 blanket refusal (per-checkpoint since #783).
- [ ] `apps/mobile/README.md:510` *(unconfirmed — verify first)* — add the missing `mold.mobile.*` storage keys.

---

## Appendix A — Confirmed findings (338)

Abbreviations as declared above. `sev`: W=wrong, S=stale, M=missing, I=inconsistent, N=nit.

### guide-core (23)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wg/cli-reference.md:418 | W | `umt5_variant` settable → registered, no get/set arm | drop row or footnote #778 |
| wg/configuration.md:12 | W | `umt5_variant` DB-owned → `Surface::File`, no DB slot | remove from DB list |
| wg/cli-reference.md:88 | S | 3 schedulers → 6 (`edm-dpm-pp-2m`, `euler`, `dpm-pp`) | list all; note wan uses `--sample-solver` |
| wg/configuration.md:198 | S | `MOLD_SCHEDULER` 3 values → full enum | same |
| wg/cli-reference.md:303 | S | `--queue-size` caps queued jobs → hydrated window only | reword; backlog uncapped |
| wg/configuration.md:242 | S | `MOLD_QUEUE_SIZE` caps queue → hydrated window | same |
| wg/configuration.md:454 | S | source `server/cli/backfill` → +`tui`,`unknown` | list 5 |
| wg/configuration.md:450 | S | two surfaces write rows → three (TUI) | add TUI |
| wg/cli-reference.md:424 | M | key sections omit Scheduler/Queue/Generate | add 3 sections |
| wg/configuration.md:15 | M | per-model DB list omits `default_frames`/`default_fps` | add both |
| wg/cli-reference.md:48 | M | `--video-only` absent | add row |
| wg/cli-reference.md:65 | M | `--hdr-exr-dir`/`--hdr-exr-full-float` absent | add rows |
| wg/cli-reference.md:92 | M | whole Identity flag heading absent | add 5 flags + link |
| wg/cli-reference.md:398 | M | `mold pull` flags absent | add `--skip-verify`, `--accept-license` |
| wg/cli-reference.md:525 | M | `mold licenses` absent | add row |
| wg/cli-reference.md:541 | I | requires-server table omits `queue`/`library` (page says so at :151/:185) | add both |
| wg/cli-reference.md:384 | M | MCP "6 things" → 9 tools | add status + retry |
| wg/configuration.md:260 | M | `MOLD_HOST_RAM_ZFS_ARC` absent | add Server row |
| wg/configuration.md:203 | I | SKILL:1533 still presence-only gemma rule | fix SKILL, note local fallback |
| wg/configuration.md:5 | I | CLAUDE.md claims XDG config.toml | fix CLAUDE.md |
| wg/cli-reference.md:517 | N | `mold clean` implied destructive → dry-run default | reword |
| wg/cli-reference.md:368 | N | `--probe` `/health` → `/health` + `/api/status` | add |
| wg/configuration.md:384 | N | only `pulid-flux` named → also `pulid-sdxl` | name both |

### guide-generate (25)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wg/identity.md:29 | W | "enumerated, not family match" → family-wide minus 1 | restate |
| wg/identity.md:32 | W | 6 qualified models → all FLUX + 7/8 SDXL | drop table |
| wg/identity.md:41 | W | turbo/Playground/Pony refused → only `sdxl-turbo:fp16` | one bullet |
| wg/identity.md:361 | S | 4 SDXL in licenses row → 7 | "any SDXL except turbo" |
| wg/generating.md:199 | W | 6-model enumeration | link identity.md |
| wg/generating.md:221 | S | identity well web+desktop → +iPhone | add iPhone |
| wg/generating.md:320 | W | `--negative` → `--negative-prompt` | fix flag |
| wg/generating.md:441 | W | `sdxl:fp16` not a model | `sdxl-base:fp16` |
| wg/generating.md:479 | W | `sdxl:fp16` | `sdxl-base:fp16` |
| wg/generating.md:480 | W | `z-image:bf16` | `z-image-turbo:bf16` |
| wg/generating.md:406 | S | ConvRot always force-streams → CUDA keeps packed residency | scope to Metal/CPU |
| wg/generating.md:394 | M | `--video-only` absent from LTX-2 list | add |
| wg/generating.md:626 | S | "🎬 badge" → optgroup family headers | reword |
| wg/generating.md:667 | W | F/T/⌘⇧N/⌫/⌘⌫ shortcuts → none bound | delete or reduce to ⌘K/Esc/←/→ |
| wg/generating.md:63 | I | "all dimensions ×16" → 32 for ltx-video/ltx2/H3/`wan22-ti2v-5b` | per-family |
| wg/video.md:46 | S | LTX-2.5 CUDA deferred → shipped #1461 | drop "and CUDA" |
| wg/video.md:37 | M | only `:int8-conv` named → 12 runnable manifests incl. 7 GGUF | add tiers |
| wg/video.md:244 | M | chain-limits `?model=` only → also `?fps=` | document fps |
| wg/video.md:249 | S | sample body omits `fps`, `frames_per_clip_runtime_seconds` | add both |
| wg/expansion.md:34 | M | 8 `task` values → 9 (`reference-to-audio-video`) | add |
| wg/expansion.md:133 | M | `[expand]` omits `api_model`,`top_p`,`max_tokens`,`thinking`,`batch_prompt` | add (esp. `api_model`) |
| wg/expansion.md:112 | M | `mold expand --backend`/`--expand-model` undocumented | add example |
| wg/upscaling.md:54 | M | `--preview` absent | add |
| wg/upscaling.md:23 | M | 5 upscalers → 7 | add 2 `:fp32` rows |
| wg/upscaling.md:55 | N | default `real-esrgan-x4plus:fp16` → env, then first installed, then that | reword |

### guide-ops (17)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wg/desktop.md:684 | W | Windows job runs on every PR → excluded from PRs | reword to post-merge |
| wg/desktop.md:262 | W | Mold default theme → Safelight (Dark) | swap |
| wg/desktop.md:390 | W | iPhone default System → Dark | fix |
| wg/feature-matrix.md:187 | S | `--scheduler` SD1.5/SDXL only → Wan via `--sample-solver` | reword |
| wg/feature-matrix.md:72 | S | Wan scheduler override "No" → Yes | fix cell |
| wg/performance.md:127 | W | dead `#performance-knobs` anchor | retarget or add heading |
| wg/troubleshooting.md:183 | W | dead `#performance-knobs` anchor | same |
| wg/custom-models.md:146 | W | LoRA family refusal 400 → 422 `VALIDATION_ERROR` | fix code |
| wg/custom-models.md:148 | N | `.safetensors` only → also `camera-control:` preset | add |
| wg/desktop.md:581 | M | windows.ps1 verbs omit `bundle`,`clean`,`features` | add |
| wg/desktop.md:669 | M | devshell list omits 7 `android-*` | add + android.yml |
| wg/desktop.md:59 | S | Desktop workflow publishes Windows assets → Windows Nightly does | reattribute |
| wg/desktop.md:504 | N | "Settings → App" → "Appearance & app" | rename |
| wg/desktop.md:448 | N | "Add host" ×2 → "Add machine" | rename |
| wg/desktop.md:266 | M | accordion list omits Saved media, Library | add |
| wg/feature-matrix.md:162 | I | unconditional CUDA on x64 Windows → published build is CPU-only | qualify |
| wg/remote-workflows.md:39 | N | sample VERSION 0.14.0 → 0.26.0 | bump |

### guide-mobile-tui (24)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wg/tui.md:427 | W | `d` = forget → `d` = toggle connection, `f` = forget | swap; add `f` |
| wg/tui.md:573 | W | Left/Right cycle workspaces → Alt+Left/Right | fix |
| wg/tui.md:298 | W | Ctrl+E end-of-line → bypassed to Expand | delete row |
| wg/tui.md:285 | S | `~/.mold/prompt-history.jsonl` → mold.db, jsonl migrated once | fix |
| wg/tui.md:608 | S | `~/.mold/tui-session.json` → mold.db settings/model_prefs | fix |
| wg/tui.md:507 | S | Reduce Motion "upcoming" → consumed today | drop clause |
| wg/tui.md:508 | S | Show Timeline "upcoming" → consumed today | drop clause |
| wg/tui.md:399 | M | Models keys omit `r`, `/` | add |
| wg/tui.md:400 | N | Enter = set default → select into Create | reword |
| wg/tui.md:432 | M | Machines keys omit `l` | add |
| wg/tui.md:107 | M | "six essentials" → +Duration/+Predict duration on video | qualify |
| wg/tui.md:153 | M | Source image row omits `References` | add |
| wg/tui.md:134 | S | Size cycles 5 computed ratios → model's authored presets first | reword |
| wg/tui.md:96 | M | palette omits Connect a machine, Retry held prints | add |
| wg/tui.md:51 | M | only `mold-server.*.log` → also `mold-tui.*.log` | add |
| wg/tui.md:661 | S | "All features" omits webp,mp4,metrics,mdns,pulid | relabel or extend |
| wg/iphone.md:422 | W | fresh install System appearance → Dark | fix |
| wg/iphone.md:87 | S | asks each machine for a placement plan → telemetry-only for one-shots | reword |
| wg/iphone.md:179 | W | canonical families omit 5:4 / 4:5 | add |
| wg/iphone.md:24 | W | "Pair an iPhone" button → "Create pairing code" | rename |
| wg/iphone.md:416 | N | "GPUs" → "Compute devices" | rename |
| wg/iphone.md:414 | M | settings list omits Model licenses | add bullet |
| wg/android.md:11 | M | nightly APK only → also tagged releases + SHA256SUMS | add |
| wg/iphone.md:451 | I | CLAUDE.md says Continue motion/Crossfade → code says Smooth/Fade | fix CLAUDE.md |

### models-image (19)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wm/index.md:58 | W | `sd35-large:q8` → no such model | `sd3.5-large:q8` |
| wm/index.md:22 | W | sdxl-turbo 1024² → 512² default | fix |
| wm/index.md:69 | W | offload FLUX-only, ~4–5 GB, 2–4× → 7 families, 2–4 GB, 3–5× | rewrite |
| wm/index.md:60 | I | qwen q4 "validated at 1024" → 1328 per qwen-image.md | fix |
| wm/flux.md:44 | W | ultrareal q4 7.5 GB → 6.8 GB | fix |
| wm/sd15.md:20 | W | 1.7 GB ×3 → 3.4 GB | fix |
| wm/sdxl.md:20 | W | playground 25 steps → 50 | fix |
| wm/sdxl.md:28 | W | guidance 7.5 → wrong for 5 of 8 | per-model list |
| wm/sdxl.md:29 | W | scheduler DDIM → wrong for 5 of 8; omits `edm-dpm-pp-2m` | per-model list |
| wm/sdxl.md:27 | S | 1024² → 512² for turbo | qualify |
| wm/sd35.md:57 | M | no gating note → all SD3.5 files `gated: true` | add note |
| wm/sd35.md:61 | S | "use `--offload` with BF16" → no BF16 ships; GGUF refused | scope/drop |
| wm/z-image.md:6 | W | Z-Potentials → Tongyi-MAI + leejet GGUF | fix |
| wm/z-image.md:18 | W | bf16 12.2 GB → 24.6 GB | fix |
| wm/z-image.md:28 | W | 3 presets incl. non-existent 1024×768 → 11 real presets | replace |
| wm/wuerstchen.md:23 | W | 5.6 GB → 3.7 GB model / ~12.5 GB total | fix |
| wm/upscalers.md:11 | M | 5 models → 7 | add 2 rows |
| wm/upscalers.md:71 | W | Comfy-Org repackaged → hlky/* + wkrettek | fix sources |
| wm/flux2.md:41 | N | `FLUX.2-Klein` link dead → `FLUX.2-klein-4B` | fix link |

### models-video (13)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wm/ltx2.md:220 | W | `--model` flag → model is positional | rewrite example |
| wm/ltx2.md:841 | S | fail-closed 502, no resume → durable job + resume | rewrite |
| wm/ltx2.md:704,834 | W | keyframe/A2V/IC-LoRA/retake/lip-dub don't chain → only audio-only declines | rewrite; IC-LoRA HDR chains |
| wm/wan.md:364 | W | `:fp8` routes at 45 → 53 (family floor) | fix |
| wm/minimax-h3.md:292 | W | references 1–15 s → 2–15 s | fix |
| wm/ltx-video.md:85 | S | 5 presets → 8 | add 3 rows |
| wm/ltx2.md:898 | M | `--video-only` absent | add note |
| wm/ltx2.md:160 | M | automatic duration named, no flag → `--predict-duration` | name flag + field |
| wm/minimax-h3.md:333 | M | no batch note → uploaded refs force batch 1 | add |
| wm/ltx2.md:29 | I | "On Apple Silicon" default → platform-independent (:82 correct) | drop qualifier |
| wm/minimax-h3.md:268 | I | "hidden official BF16" → `hidden: false` (:51 correct) | "download-only" |
| wm/ltx2.md:761 | I | correct 17; SKILL:479 says 4 | fix SKILL |
| wm/ltx-video.md:1 | N | page unlisted in sidebar | add nav entry or note |

### api (33)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wa/index.md:1399 | W | max_frames 484 @24fps → 481 | fix |
| wa/index.md:1420 | W | 484 / 124 → 481 / 121 | fix |
| wa/index.md:1403 | W | recompute `sec·fps+4` → must snap down onto grid | fix formula |
| wa/index.md:1777 | W | variations max 10 → 10,000 | fix |
| wa/index.md:1056 | W | SSE complete `images[]`+byte array → single base64 `image` | replace example |
| wa/index.md:1080 | W | complete = `GenerateResponse` → `SseCompleteEvent` | fix |
| wa/index.md:1867 | W | upscale complete `width`/`height` → `original_*`+`format`+`upscale_time_ms` | replace |
| wa/index.md:1325 | W | `strength` not amendable → it is | move to overlay list |
| wa/index.md:1357 | M | chain invalidators omit `strength` | add |
| wa/index.md:1360 | M | stage dirty tuple omits LoRA stack | add |
| wa/index.md:1382 | I | "non-`mp4`" → non-video | fix |
| wa/index.md:1274 | M | resume states omit `paused` | add |
| wa/index.md:907 | W | queue lists queued+running → also held | document held fields |
| wa/index.md:907 | M | omits `live_only_entries`, `page`, `?limit=`, `?cursor=` | add |
| wa/index.md:683 | M | child states omit `paused` | add |
| wa/index.md:1133 | S | `gallery_removed` on DELETE → DELETE trashes (`gallery_trashed`) | split |
| wa/index.md:1127 | M | 9 event types → 17 | add 8 |
| wa/index.md:48 | S | "Delete a saved image" → trashes; also PATCH | fix + add PATCH row |
| wa/index.md:9 | M | endpoint table omits 11 org/trash routes | add |
| wa/index.md:9 | M | omits both placement-preview routes | add + contract |
| wa/index.md:9 | M | omits 3 reference-upload routes | add |
| wa/index.md:9 | M | omits ~14 further routes (queue/:id, pause, activity, history, licenses, config, pairing, …) | add |
| wa/index.md:218 | W | read tier list → every GET + unclassified, ×10 cap 1000; gallery DELETE is generation | rewrite |
| wa/index.md:214 | M | generation tier omits 6 routes | add |
| wa/index.md:145 | M | exempt paths omit `/api/pairing/claim` | add + reason |
| wa/index.md:1395 | M | `/api/models` fields omit 10 | add |
| wa/index.md:100 | M | capabilities omit 7 blocks incl. `identity` | add |
| wa/index.md:707 | M | references omit `provenance.crop` | document |
| wa/index.md:509 | M | video headers omit 7 incl. `x-mold-video-pipeline` | add |
| wa/index.md:1483 | M | `/api/status` omits `instance_id` etc.; version stale | add + bump |
| wa/index.md:885 | M | omits 409 `RESTART_REQUIRED`, 404 on GET, `restart_required` | add |
| wa/index.md:136 | M | control adapters omit `gated`, `?detail=` | add |
| wa/index.md:43 | M | gallery listing omits conditional GET / `media_version` | add |

### site-misc (26)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| wd/docker.md:191 | W | `mold runpod generate` → `run` | fix |
| wd/docker.md:227 | W | `MOLD_DEFAULT_MODEL` default `--` → `flux2-klein:q8` | fix |
| wd/docker.md:243 | W | flux-schnell needs no token → gated | move to gated list |
| wd/docker.md:284 | W | `"ltx-2"` not resolvable | use real tag |
| wd/docker.md:184 | S | `/` lists gallery → Create; gallery is `/library` | fix |
| wd/docker.md:38 | I | CLAUDE.md says main publishes `latest*` → schedule does | fix CLAUDE.md |
| wd/runpod-cli.md:50 | S | "no per-step progress" → durable observer shows it | rewrite |
| wd/runpod-cli.md:143 | S | 4-GPU ladder → 10 entries + VRAM floor + High-stock fallback | rewrite |
| wd/runpod-cli.md:63 | S | proxy root = gallery → Create | fix |
| wd/runpod-cli.md:88 | W | `--keep` = don't park → keeps pod running | fix comment |
| wd/nixos.md:52 | W | CI publishes sm89 only → 4 outputs | fix |
| wd/nixos.md:72 | S | `:7680/` opens gallery → Create | fix |
| wd/nixos.md:190 | M | options omit metadataDb.{enable,path}, runpodApiKeyFile | add |
| wd/nixos.md:179 | S | queueSize caps queued jobs → hydrated window | reword |
| wd/nixos.md:254 | W | "never in process env" → it is; never in Nix store | reword |
| wd/nixos.md:142 | I | SKILL:1496 says warmup opt-in → on by default | fix SKILL |
| wd/index.md:33 | I | user-mode uses `mold-server.service` → `.user.service` + linger | fix |
| ws/docs/catalog.md:32 | W | UI includes NSFW by default → excludes | fix |
| ws/docs/catalog.md:8,52 | S | "Catalog view" → Models ▸ Discover on all surfaces | rename |
| ws/docs/catalog.md:128 | S | `MOLD_CATALOG_*_BASE` → removed | delete paragraph |
| ws/privacy.md:6 | S | scope iPhone/iPad → also Android + desktop link here | broaden |
| ws/privacy.md:17 | M | iOS Keychain only → Android Keystore unmentioned | add |
| ws/privacy.md:6 | N | "Mold Remote" → productName is "Mold" | rename |
| ws/.vitepress/config.ts:141 | N | `/models/ltx-video` unlisted | add nav item |
| ws/api/discord.md:30 | M | `/generate` omits `reference_1`/`reference_2` | document |
| ws/index.md:66 | M | no Android feature card | add |

### readme-docs (13)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| docs/ltx-2.5.md:79 | W | gate needs both true → `runtime_ready !== false` | reword |
| docs/ltx-2.5.md:153 | M | CUDA qualification harness absent | add scripts + schema |
| docs/design/README.md:3 | S | five surfaces, macOS desktop → six; +Linux/Windows | rewrite |
| docs/design/README.md:18 | S | TUI "not yet implemented" → IA + Create form shipped | reword to restyle-only |
| …/activity-history…md:436 | S | `routes_chain.rs` sets `chain_job_id: None` → `ChainRequest.ephemeral` | repoint |
| …/activity-history…md:166 | S | `.slice(-3)` / `RAIL_SETTLED_KEEP` 3 → `GENERATION_HISTORY_LIMIT` 50 | fix |
| …/activity-history…md:627 | S | amend route "in-flight" → registered | drop parenthetical |
| docs/feasibility-recovery-plan.md:3 | S | repair/resume unimplemented → shipped #1162 | narrow scope |
| docs/generate-studio-…md:22 | S | 9 "Likely files" → none exist | mark historical / repoint |
| docs/generate-studio-…md:83 | S | `web/src/lib/generationCapabilities.ts` → `studio/lib/` | repoint |
| docs/generate-studio-…md:10 | S | "Catalog" install surface → Models ▸ Discover | rename |
| docs/generate-studio-…md:323 | S | web-only verify trio → root `check:frontend` | fix |
| README.md:88 | M | macOS download only → Windows installer ships | add |

### skill (29)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| SKILL:479 | W | `--motion-tail` default 4 → 17 | fix |
| SKILL:1459 | W | DELETE queue 409 once running → 204/404; cooperative cancel | rewrite |
| SKILL:1443 | W | `strength` not amendable → it is | move |
| SKILL:1185 | W | catalog tokens in mold.db → owner-only JSON on host | rewrite |
| SKILL:703 | W | flux2-klein guidance 0.0 → 1.0 | fix |
| SKILL:11 | W | references 1–15 s → 2–15 s | fix + aggregate caps |
| SKILL:1694 | W | mobile Safelight+System → Safelight+Dark (+autoTagTitle) | fix |
| SKILL:288 | S | LTX Video defaults APNG → MP4 in release builds | fix |
| SKILL:347 | S | apng "default" → mp4 default, apng fallback | swap |
| SKILL:291 | S | example comment "APNG output" → MP4 | fix |
| SKILL:769 | S | FP8→Q8 auto-convert all → FLUX only | scope |
| SKILL:1590 | S | same tip | scope |
| SKILL:1352 | S | GPU pref 4 entries → 6 (`GPU_PREFERENCE`) | fix |
| SKILL:60 | M | routing list omits 8 subcommands | extend |
| SKILL:583 | M | no `mold library` section | add |
| SKILL:104 | M | no `mold remix` / `POST /api/remix` | add |
| SKILL:1073 | M | no `mold stats` / `mold clean` usage | add |
| SKILL:715 | M | Available Models omits Wan, Qwen-Image-Edit, MiniMax H3 | add |
| SKILL:747 | I | LTX list fp8-only; contradicts :410 | merge lists |
| SKILL:719 | N | omits `jibmix-flux:q3` | add |
| SKILL:1063 | M | 5 upscalers → 7 | add 2 |
| SKILL:1451 | M | trash routes omit `delete-forever` | add |
| SKILL:1618 | M | slash commands omit `/sequence` | add |
| SKILL:1711 | M | devshell omits all `android-*` + `desktop-release` | add + Android para |
| SKILL:1655 | S | windows.ps1 omits bundle/clean/features | add |
| SKILL:1383 | S | `server start --log-file` implied opt-in → default on | reword |
| SKILL:1499 | S | `MOLD_LOG` default warn → info for server | qualify |
| SKILL:1488 | M | env table omits 7 vars it cites elsewhere | add |
| SKILL:1248 | M | no Lambda section (RunPod has one) | add |

### app-docs (57)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| dd/server-api.md:16 | S | `/api/generate/chain` row → route deleted | delete |
| dd/server-api.md:17 | S | `/api/generate/chain/stream` row → deleted | delete; add `/chain/validate` |
| dd/server-api.md:220 | S | "two chain tiers" → one; validate-only + `ephemeral` | rewrite |
| dd/server-api.md:222 | W | "no live preview stream" → `Preview` event + queue preview route | rewrite |
| dd/server-api.md:213 | S | variant list omits DependencyWait/StageProgress/Preview | add |
| dd/server-api.md:75 | M | ServerEvent omits 7 incl. `chain_job_*` | add |
| dd/server-api.md:60 | S | thumbnail has no query params → `?size=&fmt=` + header | document |
| dd/server-api.md:91 | M | "complete surface" omits ~25 routes | add or label subset |
| dd/server-api.md:50 | M | `CollectionUpdateRequest` omits `hidden` | add |
| dd/server-api.md:77 | S | `QueueListing{entries}` → `QueueListingResponse` + paging | fix |
| dd/server-api.md:76 | M | host_memory omits `reclaimable_zfs_arc_bytes` | add |
| dd/server-api.md:82 | M | capabilities omit 8 blocks incl. `identity` | add |
| dd/server-api.md:226 | M | exempt paths omit `/api/pairing/claim` | add |
| dd/server-api.md:248 | W | CORS GET/POST/DELETE → +HEAD/PATCH/PUT | fix |
| dd/server-api.md:9 | M | generate headers omit 3 `x-mold-video-*` | add |
| dd/server-api.md:261 | N | `serve.rs::run` line 132 → 143 | fix or drop |
| dd/feature-parity.md:253 | W | `POST /api/chain/jobs` → `/api/chain-jobs` | fix |
| dd/feature-parity.md:264 | W | `GET /api/chain/limits` → `/api/capabilities/chain-limits` | fix |
| dd/feature-parity.md:291 | W | `ModelResidency{Gpu,Parked,Unloaded}` → 2 variants | fix |
| dd/feature-parity.md:360 | W | TUI views Generate/Gallery/Queue/Script → 5 workspaces | fix |
| dd/feature-parity.md:103 | S | LTX-2 Metal correctness-only → performance-qualified | fix |
| dd/feature-parity.md:84 | M | taxonomy omits wan, minimax-h3, ltx2-control(-camera) | add |
| dd/feature-parity.md:88 | M | matrix omits wan, minimax-h3 rows | add |
| dd/feature-parity.md:110 | S | catalog omits Wan, H3, flux2-dev, LTX-2.5 tiers, `:fp32` upscalers | add (note: LTX-Video 0.9/0.9.5 were *removed*, don't re-add) |
| dd/feature-parity.md:62 | M | pipeline enum omits `t2a` | add |
| dd/feature-parity.md:21 | M | `GenerateRequest` table omits ~18 fields | add |
| dd/feature-parity.md:68 | M | CLI extras omit ~26 flags | add |
| dd/feature-parity.md:182 | S | `GalleryPage` → `LibraryPage.vue`; add `media_version` | fix |
| dd/feature-parity.md:188 | S | `Metadata.vue` does not exist | repoint |
| dd/feature-parity.md:263 | S | 4 named sequence components do not exist | repoint |
| dd/feature-parity.md:293 | S | RunningStrip/RunningJobCard/JobsPanel/useQueue gone | repoint |
| dd/feature-parity.md:310 | S | `PreferencesModal.vue` → `SettingsPage.vue` | fix |
| dd/feature-parity.md:321 | S | 5 DB tables → 21 at SCHEMA_VERSION 31 | fix |
| dd/feature-parity.md:306 | M | config.toml omits `config_version`,`[scheduler]`,`[gallery]`,`[queue]`,`[generate]` | add |
| dd/feature-parity.md:307 | M | DB keys omit queue/generate/tui.* | add |
| dd/feature-parity.md:312 | I | mobile settings "only theme+family" → 4 fields | add 2 |
| dd/feature-parity.md:354 | M | Discord commands omit `/identity`, `/sequence` | add |
| dd/feature-parity.md:332 | M | serve/server omit `--no-mdns`, `discover` | add |
| dd/feature-parity.md:163 | M | `DELETE /api/models/:model` undocumented | add |
| dd/feature-parity.md:162 | M | pull omits `--accept-license`; no `mold licenses` | add |
| dd/feature-parity.md:292 | S | JobLifecycle Queued→Running → +Paused,Held | add |
| dd/feature-parity.md:10 | N | routes omit `/machines/:id`, catch-all | add |
| dd/feature-parity.md:211 | M | §5 omits Remix | add |
| dd/feature-parity.md:72 | S | `sourceFit.ts` a desktop port → shared `studio/lib` | repoint |
| dd/architecture.md:3 | M | platforms omit Windows | add + §5 detail |
| dd/architecture.md:127 | S | features `metal,expand,webp,mp4` → expand+mdns always, mp4 non-Win, metal/cuda opt-in | fix |
| dd/architecture.md:143 | S | LTX-2 Metal correctness-only | fix |
| dd/architecture.md:310 | S | risk 7 (Metal correctness-only) | resolve |
| dd/architecture.md:199 | S | quoted CSP stale (no mold-local:/mold-thumb:/ipc:) | replace |
| dd/architecture.md:190 | S | minWidth 1024 → 1080; omits maximized + platform split | fix |
| dd/architecture.md:236 | W | 4 named IPC commands do not exist | replace with real list |
| dd/architecture.md:242 | M | capabilities omit 4 `core:window:*` | add |
| dd/architecture.md:361 | S | Cargo sketch 0.1.0/0.14.0, no updater/patch | refresh or label |
| dd/architecture.md:449 | S | lib.rs sketch names 5 nonexistent commands | refresh or label |
| dd/architecture.md:280 | S | `@testing-library/vue` not a dependency | drop |
| dd/architecture.md:487 | N | absolute macOS paths + external aethon path | repo-relative |
| dd/architecture.md:109 | S | exclude list omits `apps/mobile/src-tauri` | add |

### arch-docs (34)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| da/candle-extension.md:76 | W | script bans crates.io `version` → permits it beside git+rev | rewrite |
| da/candle-extension.md:87 | S | `[patch]` + declared 0.11 deps → no candle patch, no versions | rewrite |
| da/candle-extension.md:104 | I | "branch, not the renamed fork" → pinned rev of renamed packages | fix |
| da/qwen-mmq-nan.md:97 | S | fork pinned by root `[patch]` → per-crate git deps | fix |
| da/pulid.md:37 | N | `pulid_paths`/`missing_pulid_files` → `*_for(family)` | rename |
| da/pulid.md:396 | W | extraction in `prepare_inputs_for_devices` → `resolve_identity_for_lease` | rewrite |
| da/pulid.md:410 | S | `batch_runtime::submit_child` → deleted | repoint |
| da/pulid.md:413 | W | "on the CPU, no typed phase" → leased device, `IdentityExtract` phase | rewrite |
| da/pulid.md:420 | S | 1.4 GB released before dispatch → 2.4 GB inside lease | fix |
| da/pulid.md:431 | W | four asset digests → five | fix |
| da/pulid.md:450 | S | 1.4 GB probe → 2.4 GB | fix |
| da/pulid.md:493 | W | "extractor does not run on the generation device" → it does | rewrite |
| da/pulid.md:484 | M | memory table omits `EXTRACTION_DEVICE_PEAK_BYTES`, true-CFG term | add |
| da/pulid.md:553 | S | "1.4 GB peak" → 2.4 GB | fix |
| da/pulid-face-extraction.md:201 | W | load rejects non-CPU → honours device | rewrite bullet |
| da/pulid-face-extraction.md:207 | W | "no release enables pulid yet" → ships everywhere; protobuf wired | rewrite |
| da/pulid-face-extraction.md:280 | N | `pulid_manifest()` → `pulid_manifest_for(family)` | rename |
| da/pulid-face-extraction.md:690 | M | Files table omits 4 shipped files | add |
| da/pulid-perf.md:165 | S | "load asserts CPU" → no assertion | annotate superseded |
| da/pulid-perf.md:270 | S | cache in `resolve_identity_embedding`/`ExtractionSlot` → both gone | annotate |
| da/pulid-perf.md:286 | S | `IDENTITY_PIPELINE_VERSION` "does not exist" → it does | annotate shipped |
| da/pulid-perf.md:292 | W | four-asset key → five | fix |
| da/pulid-uat.md:191 | S | refusal transcript pre-#1228 wording | annotate |
| da/pulid-uat.md:201 | S | "exactly two qualified tiers" | annotate |
| da/pulid-uat.md:501 | S | exclusion table wrong for playground/pony ×2 | annotate |
| da/wan-comfyui-parity-ledger.md:126 | S | R3 blanket includes `extend_video` → per-checkpoint since #783 | fix |
| da/research-multi-model…md:33 | S | AppState fields do not exist | snapshot banner |
| da/research-multi-model…md:75 | S | "LoRA locked to sequential" → server accepts LoRAs | mark superseded |
| da/research-multi-model…md:220 | S | proposes a `Parked` tier that shipped; line ranges stale | mark superseded |
| da/research-multi-model…md:157 | S | candle-core-mold 0.9.3 → pinned fork rev | fix or drop version |
| da/multi-gpu-scheduler…md:5 | S | "ready to split into phases" → A–G shipped, parts replaced | status banner |
| da/multi-gpu-scheduler…md:1589 | S | `GalleryPublicationGate` / `.mold-batch-transactions` never existed | mark superseded |
| da/multi-gpu-scheduler…md:1566 | S | `PlannedBatchPartition` / `BatchChild` do not exist | repoint |
| da/minimax-h3-authorization.md:83 | M | scope "two compact manifests" → +3 Turbo tags +pinned-unrunnable tiers | extend |

### claude-md (25)

| file:line | sev | claim → actual | fix |
|---|---|---|---|
| CLAUDE.md:53 | W | mold-tui must not depend on catalog → it does, via mold-inference | fix doc or dep |
| CLAUDE.md:115 | W | XDG `config.toml` → `$MOLD_HOME/config.toml` | fix |
| CLAUDE.md:137 | S | LRU max 3 → default 3, range 1–16 | fix |
| CLAUDE.md:34 | N | `check:frontend` "lint + typecheck + tests" → no lint | reword |
| cr/inference.md:12 | S | feature list omits nvml,h3,h3-cuda,pulid,mdns | extend |
| cr/inference.md:48 | W | enumerated PuLID allowlist → family-wide minus turbo | rewrite |
| cr/inference.md:59 | W | "qualified tiers flux-dev:q4/:q8" | rewrite |
| cr/inference.md:50 | S | "must add protobuf when pulid ships" → already done | delete clause |
| cr/inference.md:61 | N | broken inline code span across blank line | rejoin |
| cr/server-queue.md:14 | I | H3 conditioner always CPU on CUDA → device fit (#1423) | qualify |
| cr/server-queue.md:27 | M | GalleryCapabilities 3 fields → 7 | extend |
| cr/server-queue.md:27 | M | CLI surfaces omit `mold library` | add |
| cr/studio-web.md:8 | W | glob `mold-server/src/generation_profile*` matches nothing | → mold-core |
| cr/minimax-h3.md:7 | S | glob `mold-server/src/private_*` matches nothing | → mold-inference |
| cr/minimax-h3.md:5 | N | `**/h3/**` matches no directory | drop |
| cr/desktop.md:12 | W | `desktop-bun-lock` does not exist | → `frontend-bun-lock`; add `desktop-release` |
| cr/tui.md:11 | M | accordion omits Identity photo section + Wan SampleShift | add |
| tui-uat/SKILL.md:167 | W | `d` = forget → toggle; `f` = forget | swap |
| tui-uat/SKILL.md:167 | S | `x` cancels queued only → also running | reword |
| tui-uat/SKILL.md:167 | M | Machines keys omit l,g,[,],e | add |
| tui-uat/SKILL.md:157 | W | Ctrl+R "cycle seed mode" → randomize seed (:106 correct) | fix |
| tui-uat/SKILL.md:157 | M | omits Ctrl+E/Shift+E/S/T/P/N | add |
| tui-uat/SKILL.md:161 | S | landmarks omit Identity photo, File under | add |
| tui-uat/SKILL.md:161 | S | "6 essentials" → 7–8 on video models | qualify |
| tui-uat/SKILL.md:252 | N | model_prefs column list omits profile/frames/fps/last_* | extend |

---

## Appendix B — Unconfirmed or refuted (5)

| area | file:line | claim | verifier note | action |
|---|---|---|---|---|
| guide-generate | wg/generating.md:255 | LTX-2.5 not named among video families | **no verdict** — verifier died | Re-verify. Likely valid: `ltx25_manifest.rs` ships 12 runnable `ltx2`-family manifests and `wg/video.md:7` names LTX-2.5. Cheap to confirm. |
| readme-docs | docs/feasibility-recovery-plan.md:61 | `resolveFeasibleSubmitRoute` no longer exists | **REFUTED** — it exists at `web/src/pages/CreatePage.vue:3381`, wrapping `routing.resolveFeasible` | **Do not act.** The finding's core claim is false. The toast-string half may still be stale — check separately if the section is being rewritten anyway (batch 8). |
| skill | SKILL:803 | SDXL identity cross-doc conflict | **NOT ACTIONABLE for SKILL** — SKILL.md:803 is *correct*; the finding's own fix says "leave SKILL.md as is" and blames CLAUDE.md | Already folded into **Batch 2** as the CLAUDE.md fix. No SKILL.md edit. |
| app-docs | dd/feature-parity.md:34 | `scheduler` enum values wrong | **bookkeeping artifact** — the note reads "no finding at this index; skip", not a verdict | Re-verify. The claim is independently corroborated by `types.rs:137-165` and by the confirmed `wg/cli-reference.md:88` finding, so it is almost certainly valid. Scheduled in Batch 5. |
| app-docs | apps/mobile/README.md:510 | localStorage key list incomplete | **no verdict** — verifier died | Re-verify against `grep -ohE 'mold\.(mobile\|sequence)\.[a-zA-Z.-]+'`. Scheduled last in Batch 8. |

**Also re-check before acting:** every `confirmed` flag in `dd/feature-parity.md:34` → `dd/architecture.md:109` (the verifier-note offset described in §2). The findings were independently sourced; only the verdict attribution is suspect.