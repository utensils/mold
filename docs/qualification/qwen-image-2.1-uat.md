# Qwen Image 2.1 — live UAT on plato (2026-09-27)

This UAT used scratch `mold serve` instances built from the
`feat/qwen-image-21-complete` branch. The build was a release sm_89 binary
with the full release feature set (`h3-cuda,cudnn,preview,discord,expand,webp,mp4,metrics,mdns,pulid,mesh-*`)
and the real embedded web SPA. The servers ran as the `mold` user against
the shared `/storage/mold` home, so every render also went into the gallery
that production (v0.32, `:7680`) serves:

- `:7684` ran on CUDA ordinal 0.
- `:7683` ran on CUDA ordinal 3.

GPU 1 was not used, because of the PCIe fault on that slot. GPU 2 belonged to
the benchmark agent.

Before any server started, I checked `git diff 5d2b951 HEAD -- crates/mold-db crates/mold-server/src/gallery*`
for storage-format changes. There were none. The only schema change is two
additive `Option` fields in `mold-db`. There is no `STORAGE_VERSION`, marker or
migration bump.

The binaries were rebuilt as fixes landed: `cafcb027` first, then `4f84f44c`,
`83e09295`, `0d4c5f9a`, `101c8b07`, `0cdacc23` (with the prefix-cache merge
`a9acc386`), and finally `28915be7`. Each row below names the build it ran on.
The evidence is in `/storage/mold/uat-qwen21/uat/`:

- renders
- per-run CLI transcripts, each with its wall-clock window
- `vram.csv`, `nvidia-smi` sampled every 250 ms on GPUs 0 and 3
- `ui/` screenshots
- `bench/`

The server logs are in `/storage/mold/uat-qwen21/logs/`.

**Every render was checked.** 51 distinct images were viewed, and the
reruns were byte-compared against them:

- 45 were viewed individually, including one bench image per case. Transparent outputs were viewed composited over a checkerboard, next to their alpha plane.
- 6 were viewed as a grid: the 24/16/12 GB simulation tiers.
- 38 reruns were byte-identical to an image that was viewed:
  - 15 bench repeats
  - the 16 LoRA-toggle reruns
  - 2 turbo reruns
  - 4 post-merge reference reruns
  - the 24 GB `q4` render

The screenshots were viewed separately.

## Matrix

| #   | Item                                                                                    | Result                          | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --- | --------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | t2i bf16 1024², guidance 4 + negative (fast path, server)                               | Pass                            | `u1-t2i-g4neg.png`: RGB, coherent. Denoise 29.3 s on a cold first request.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2   | References through the server: 1, 2, 3 and 10 refs, RGBA ref, extract, last-ref canvas  | Pass, after one fix (F4)        | See the reference table below. The canvas defaulted to the last reference (1344x768 beach → 1344x768). The RGBA reference kept its alpha in the output. "Extract the subject" with `--transparent` gave a clean cat cut-out (alpha 0 at the border). Raw `POST /api/generate` with `edit_images` worked (cat + beanie). The raw API still requires `width`/`height`, as documented: the canvas rule is a client rule.                                                                                                                                                          |
| 3   | Transparency: PNG and WebP, JPEG refusal, toggle off → RGB                              | Pass                            | `--transparent --format png` produced an RGBA PNG (border alpha mean 1.4, subject 255). `--format webp` produced `VP8X+ALPH+VP8` with no `ANIM` chunk (border alpha 0.2). JPEG + `--transparent` was refused by both the CLI and `POST /api/generate` with 422 `transparent_background needs a format with an alpha channel; use png or webp instead of jpeg`. A plain t2i came out RGB.                                                                                                                                                                                       |
| 4   | JPEG with an alpha-carrying reference                                                   | Pass                            | Raw API: 200 `image/jpeg`, and `x-mold-request-warning: A Qwen Image 2.1 reference carries transparency, which JPEG cannot store; the output was composited over white. …` reached the client. The CLI printed the same sentence. Both outputs are RGB and composited over white.                                                                                                                                                                                                                                                                                              |
| 5   | Turbo bf16 and int8-conv, a user LoRA on the base tier and stacked on turbo, no rebuild | Pass, after three fixes (F1–F3) | Turbo is 6 steps: 2.9 s (bf16) and 3.2 s (int8-conv). The Viggle r128 file as a user LoRA on bf16 took 0.9 s to install (0.63 GiB bypass). Stacked on turbo it installed 2 adapters (1.90 GiB). On `cafcb027`, every add, remove or rescale of a LoRA logged `recreating cached engine … execution-fingerprint-differs` and reloaded 28.6 GB. On `101c8b07`, one load served none → LoRA 0.5 → LoRA 0.8 → none → turbo → turbo + LoRA with zero rebuilds. Outputs were byte-identical to the rebuild-every-request run. Wall time per LoRA request fell from 26.8 s to 21.0 s. |
| 6   | Every tier at 1024², plus VRAM table checks                                             | Pass                            | See the tier table. The docs VRAM table was confirmed at 44+, 32, 24, 16 and ≤12 GB (simulated with `MOLD_RESERVE_VRAM_MB`), and a measured-peak column was added to `website/models/qwen-image-21.md`.                                                                                                                                                                                                                                                                                                                                                                        |
| 7   | 2K presets through the server                                                           | Pass                            | 2048²: denoise 83.3 s, total 97.1 s, peak 37.4 GB. 2752x1536: denoise 85.5 s, total 103.7 s, peak 37.0 GB. The text encoder parks for the denoise. Lettering was sharp in both.                                                                                                                                                                                                                                                                                                                                                                                                |
| 8   | WebP stills for FLUX and SDXL                                                           | Pass                            | `flux-schnell:q8` and `cv:1759168` (Juggernaut XL) with `--format webp`: RIFF/WEBP with a single `VP8 ` chunk and no `ANIM`. PIL reads them as RGB non-animated. They appear as images in the gallery grid and lightbox.                                                                                                                                                                                                                                                                                                                                                       |
| 9   | Web SPA flows, desktop and phone width                                                  | Pass, after one fix (F5)        | Details are under "Frontend" below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 10  | `scripts/bench-qwen21.sh` live                                                          | Pass, all 4 M5 gates            | See the "Server end-to-end" section of `qwen-image-2.1-cuda-performance.md`: 15.0 / 30.3 / 83.4 / 84.7 s, and int8-conv 14.3 s.                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 11  | Discord `/transparent` unit tests, MCP `generate_image`                                 | Pass                            | `cargo test -p mold-ai-discord`: 182 passed, including all 7 `commands::transparent` tests. The bot needs a token, so it was not run live. MCP over stdio (`mold mcp --host :7684`): `generate_image` with `reference_images` and `transparent_background: true` returned an RGBA PNG (the beanie cut out of `src-hat.png`, border alpha 0.1).                                                                                                                                                                                                                                 |

### References (server, bf16 1024-class canvas, guidance 1 unless noted)

| Refs                          | Build      | Denoise | Total   | Peak (whole process) | Notes                                                                                                                                                                                                                                                                                                          |
| ----------------------------- | ---------- | ------- | ------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1                             | `cafcb027` | 17.0 s  | 30.2 s  | 35.3 GB              | Sunglasses on the cat                                                                                                                                                                                                                                                                                          |
| 1                             | `0cdacc23` | 17.5 s  | 30.7 s  | 35.4 GB              | Byte-identical to the `cafcb027` render                                                                                                                                                                                                                                                                        |
| 2                             | `cafcb027` | 19.5 s  | 33.9 s  | 35.3 GB              | Cat in the jacket                                                                                                                                                                                                                                                                                              |
| 2                             | `0cdacc23` | 20.1 s  | 35.8 s  | 35.4 GB              | Byte-identical                                                                                                                                                                                                                                                                                                 |
| 3                             | `cafcb027` | 22.0 s  | 37.4 s  | —                    | Cat in the jacket on the beach. The canvas came from the last reference: 1344x768.                                                                                                                                                                                                                             |
| 3                             | `0cdacc23` | 22.7 s  | 39.5 s  | 35.6 GB              | Byte-identical                                                                                                                                                                                                                                                                                                 |
| 3, guidance 4 + negative      | `0cdacc23` | 45.3 s  | 63.9 s  | 35.6 GB              | Both branches retained their prefix cache                                                                                                                                                                                                                                                                      |
| 10 (PNG, WebP, RGBA mixed)    | `0cdacc23` | 350.0 s | 379.2 s | 37.5 GB              | The recompute warning reached the client: "recomputed its 20.0 GiB prefix cache every step because it did not fit the 16.3 GiB this card had left … `MOLD_QWEN_IMAGE21_KV_CACHE=on` forces the cache." The output is RGBA because two references carry alpha (the rule), and it composes 9 of the 10 subjects. |
| RGBA ref (transparent teapot) | `cafcb027` | 17.5 s  | 30.9 s  | —                    | Red teapot. Alpha kept (border 0).                                                                                                                                                                                                                                                                             |
| Extract + `--transparent`     | `cafcb027` | 17.5 s  | 31.6 s  | —                    | Clean cut-out                                                                                                                                                                                                                                                                                                  |

Each reference request pays about 12–13 s on top of its denoise on this 46 GB
card. The engine's residency decision parks the 17 GB Qwen3-VL encoder to host
RAM for every reference request, then restores it before the next encode. The
same happens at 2K, where it costs about 20 s. This is not a failure: the
decision budgets the encode-phase working set and the denoise workspace as one
sum. Measured `nvidia-smi` peaks with the encoder resident stayed at or below
37.5 GB, so a max-of-phases budget would keep the encoder on 46/48 GB cards. I
have reported this to the orchestrator as a follow-up for the residency
owners, and did not change it here.

### Every tier at 1024² (server, 46 GB L40S, GPU 3, build `cafcb027`)

| Tier        | Denoise | Peak (whole process, encoder resident) | Image                                                                                                                                 |
| ----------- | ------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `bf16`      | 14.9 s  | 37.3 GB                                | Sharp "MOLD & FLOUR"                                                                                                                  |
| `int8-conv` | 14.5 s  | 31.9 GB                                | Near-identical to bf16                                                                                                                |
| `fp8`       | 19.0 s  | 27.9 GB                                | Clean                                                                                                                                 |
| `q8`        | 24.5 s  | 28.6 GB                                | Clean. The slow time is an outlier: its window overlapped a load on the other server, and the 24 GB-simulation rerun measured 19.2 s. |
| `q6`        | 19.3 s  | 27.6 GB                                | Clean                                                                                                                                 |
| `q5`        | 19.3 s  | 26.7 GB                                | Clean                                                                                                                                 |
| `q4`        | 19.3 s  | 25.5 GB                                | Clean                                                                                                                                 |
| `q3`        | 19.4 s  | 24.2 GB                                | Coherent, with a softer serif                                                                                                         |
| `q2`        | 19.2 s  | 23.1 GB                                | Visibly degraded, and the lettering reads "MOLD & FLOTR". Matches the docs.                                                           |

### Smaller cards (simulated on GPU 3 with `MOLD_RESERVE_VRAM_MB`, build `28915be7`)

| Card                            | Tier        | Plan                                         | Denoise | Peak    |
| ------------------------------- | ----------- | -------------------------------------------- | ------- | ------- |
| 32 GB                           | `bf16`      | Eager, Q8 GGUF encoder (auto), encoder parks | 15.2 s  | 25.3 GB |
| 24 GB                           | `int8-conv` | Eager, Q8 GGUF encoder, parks                | 14.3 s  | 18.7 GB |
| 24 GB                           | `q8`        | Eager, Q8 encoder                            | 19.2 s  | 19.1 GB |
| 24 GB                           | `q4`        | Eager, Q8 encoder                            | 19.0 s  | 19.6 GB |
| 24 GB                           | `bf16`      | Sequential, BF16 encoder on CPU (20 s load)  | 15.0 s  | 14.9 GB |
| 16 GB (`MOLD_QWEN3_VARIANT=q4`) | `q4`        | Eager, Q4_K_M encoder                        | 19.8 s  | 12.8 GB |
| 12 GB                           | `q3`        | Sequential, encoder on CPU                   | 19.2 s  | 7.8 GB  |
| 12 GB                           | `q2`        | Sequential, encoder on CPU                   | 19.0 s  | 6.7 GB  |

Every recommendation in the docs VRAM table held.

## Frontend (embedded web SPA on `:7684`, headless Chromium via Playwright)

I checked these at 1600x1000 and at a 390x844 phone viewport. The screenshots
are in `/storage/mold/uat-qwen21/uat/ui/`.

- **Style picker.** Qwen Image 2.1 BF16 appears, along with the other eight tiers and three turbo tiers. The resolution presets are the advertised aspect groups, with 1024² and 2048² at 1:1.
- **Transparent background.** The toggle is in More settings → Output & seed. With the toggle on, JPEG is disabled and the page shows "JPEG has no transparency, so a transparent background saves as PNG or WebP". The phone rail has the same toggle.
- **References.** "Start from a photo" opens a _Reference images_ panel reading "Up to 10 ordered references". Uploading three files, including a WebP with alpha, was accepted ("src-cat.png +2 more").
  - At the time of this UAT the web strip showed the first file name and a count, not ordered thumbnails. That was later replaced: every web, desktop and phone reference strip now draws one numbered thumbnail per picture (see "Resolved after this UAT").
- **LoRA.** The _Add-on looks_ row is visible for Qwen Image 2.1 and hidden for `flux2-klein:fp8`, whose recipe has `lora.mode: hidden`.
- **Checkerboard.** Transparent prints show a checkerboard (`ms-alpha-bed`):
  - in the gallery grid: the teapots, bottle, cat cut-out and beanie
  - in the desktop lightbox
  - in the phone lightbox

  The WebP stills from FLUX and SDXL display as images.

- **Desktop Tauri app.** It cannot run on this headless Linux host. Its surfaces share `studio/` with the web SPA checked here.

## Fixes (commits on the UAT branch)

| #   | Commit                             | What                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1  | `4f84f44c`                         | **A LoRA change rebuilt the whole Qwen Image 2.1 engine.** The server's warm-reuse fingerprint hashed the adapter stack. `qwen_image21` installs adapters per request into bypass slots (`active_lora`), so its warm identity now ignores the stack.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| F2  | `83e09295`, `0d4c5f9a`, `101c8b07` | F1 alone did not fix it live. An eager engine retains no residency, so it is compared by the EXACT fingerprint, which still hashed three things that change per request: the adapter stack, the planner's per-request component plan (encoder park or drop, decode park), and the `Lora` authored-placement role. For `qwen-image21`, `engine_fingerprints` now takes both identities over the load-plan-independent, adapter-free components and constraints. Checkpoint content, dtype, quantization, encoder variant, config and authored placement still move the identity. Load strategy and block offload are still caught by the planned-mode check. This was pinned by `a_qwen_image21_warm_engine_serves_any_adapter_stack` and verified live: zero rebuilds across 6 LoRA/turbo toggles, with byte-identical pixels. This fix is in the scheduler/execution-plan area (`crates/mold-server/src/execution_plan.rs`). |
| F3  | —                                  | (Part of F2.) The turbo tier plus a user LoRA stacked on it did not rebuild.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| F4  | `28915be7`                         | **A "not a recommended resolution" advisory fired on the canvas the recipe itself chose.** `mold run` with three references, last reference 1344x768, printed "1344x768 is not a recommended resolution…". `routes::dimension_advisory` now skips exactly the `canvas: last-reference` canvas, which is `fit_to_target_area_ties_even` of the last reference, mirroring upstream `calculate_dimensions`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| F5  | `ee1cccc4`                         | **The web SPA replayed one "installed …" toast per past pull on every page load.** Twelve toasts covered the page after the tier pulls. `installNotifications` seeded its seen set before the downloads singleton had read `/api/downloads`. The first loaded listing is now the baseline. This is `web/src/lib/notifications.ts`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| F6  | `881b789e`                         | **A host-placed Qwen3-VL encoder reported "No GPU detected"** under the 24 GB and 12 GB sequential plans on an L40S. The line now says the encoder is placed on the host, unless the process has no CUDA or Metal device (`encoders/variant_resolution.rs`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

Gates run for the fixes:

- `cargo fmt --all`
- `cargo clippy -p mold-ai-server --all-targets -D warnings`
- `cargo clippy -p mold-ai-inference --all-targets -D warnings`, and the same with `--features cuda --lib`
- `cargo test -p mold-ai-server --lib` for these modules:
  - `execution_plan`: 111
  - `gpu_worker`: 131
  - `scheduler`: 183
  - `memory_preflight`: 52
  - `queue`: 472
  - `routes`: 569
- `cargo test -p mold-ai-inference --lib variant_resolution`
- `cargo test -p mold-ai-discord`: 182
- `bunx vitest run src/lib/notifications.test.ts`: 12
- prettier

## Observed, not changed

These are the observations as recorded during the UAT. Most were fixed
afterwards; see the next section.

- **Encoder parking on 46/48 GB cards.** Each reference request adds about 12 s, and each 2K request about 20 s, while the 17 GB Qwen3-VL encoder is parked and restored. The cause is the budget sum described above. I left it to the residency owners.
- **Ten references are slow at bf16 1024².** The denoise took 350 s because the 20 GiB prefix cache did not fit and is recomputed every step. The client is told so, and `MOLD_QWEN_IMAGE21_KV_CACHE=on` is named.
- **Single-request validation errors carry a `requests[1]:` prefix** on `/api/generate`, for example `requests[1]: steps must be >= 1`. v0.32 production does the same, so this predates the campaign.
- **One shutdown hung.** The first two scratch servers were sent SIGTERM about 20 s after start, while the 107 s "warmed installed artifact facts" startup task was still running. Both hung in shutdown until I killed them with `kill -9`. Every later shutdown exited cleanly within seconds.
- **Two ~32 s publication stalls hit both scratch servers at the same instant.** In one, `write_gallery_bytes_no_replace` returned at the same millisecond on both servers. The pattern points to shared-pool I/O on the ZFS home rather than mold, and it did not recur.
- **The MCP server answers `initialize` with protocol `2025-06-18`** when the client asks for `2024-11-05`. The tool calls worked.

## Resolved after this UAT

Each item below was fixed on the branch after the UAT ran. The fixes are
pinned by tests; unless stated, they were not re-measured on the scratch
servers.

| Observation                                                          | Fix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Commits                                                    |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Encoder parking on 46/48 GB cards (about 12 s per reference request) | The residency decision budgets the largest phase (encode, denoise, decode) instead of their sum, no longer charges the finished encode after it has run, and releases the vision tower and VAE encoder before it parks the text encoder. On a 46/48 GB card at 1024² the encoder now stays resident for one to three references, and for one or two with guidance and a negative prompt. Three references with guidance and a negative prompt still park it so both branches keep their prefix cache. | `d17730ba`, `5617cfc1`                                     |
| Web reference strip showed a file name and a count                   | One shared, ordered strip on web, desktop and phone: a numbered thumbnail per picture on the alpha checkerboard, per-picture remove, drag and keyboard reorder, and a "Sets canvas" mark on Qwen Image 2.1's last reference. The strip wraps so all ten references stay in view.                                                                                                                                                                                                                      | `f3eaa2d1`, `19938138`, `5ee47af0`, `b9708bee`, `a9d64b50` |
| `requests[1]:` prefix on single-request validation errors            | A single request's refusal carries no prefix; a real batch still names its row.                                                                                                                                                                                                                                                                                                                                                                                                                       | `d53142e4`                                                 |
| Shutdown hung during the startup artifact warm                       | Shutdown abandons the warm instead of waiting for it.                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `27c3b905`                                                 |
| MCP `initialize` answered `2025-06-18` to a `2024-11-05` client      | `initialize` echoes the client's protocol version when mold supports it.                                                                                                                                                                                                                                                                                                                                                                                                                              | `92235e16`                                                 |

Two observations stand. Ten references at bf16 1024² still recompute a
prefix cache that does not fit a 46 GB card, and say so in a request warning.
The publication stalls were not reproduced and point at the shared ZFS pool,
not at mold.
