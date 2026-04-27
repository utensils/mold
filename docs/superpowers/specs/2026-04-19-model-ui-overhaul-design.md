# Model UI Overhaul Design

> In-browser model downloads with progress + queue, per-component device placement for high-value model families, and always-on VRAM/RAM telemetry in the generate UI.

## Goals

- Download any manifest model from the web UI with progress, ETA, cancel, and auto-refresh on completion.
- Enforce one-model-at-a-time downloading server-side, with a visible queue and persistence across page reloads.
- Show total model size next to every model in the picker, before download.
- Show always-visible VRAM + system RAM telemetry in the generate page, per-GPU, with per-process attribution ("mold" vs "other") on CUDA.
- Let users pick which device (CPU or a specific GPU) runs the text encoders for any model.
- Let users pick per-component placement (text encoders, transformer, VAE) for a curated set of families where plumbing is cheap and the value is clearest: FLUX, Flux.2, Z-Image, Qwen-Image, and SD3.5 (stretch).
- Persist placement per-model in config via a "Save as default" action; ephemeral per-session otherwise.

## Non-Goals

- File-level parallelism inside a single model pull (bandwidth-bound; HF rate-limits parallel requests). Sequential inside a set is the final design.
- Per-component device placement for model families outside the curated set. Those families show Tier 1 (group text-encoders knob) only; the Advanced panel is disabled with a tooltip.
- Playwright / browser e2e testing. Unit-level coverage only; manual UAT on <gpu-host> closes the loop.
- Model delete / disk management from the UI. `mold rm` CLI remains the only path for this PR.
- Free-disk-space enforcement before download — soft warning only.
- VRAM-fit prediction for a chosen placement config. Stretch only if Agent C has spare time.
- Per-process GPU attribution on Metal. macOS doesn't expose this to userspace; the SPA hides those rows on Metal hosts.

## Delivery Shape

- **Umbrella branch** `feat/model-ui-overhaul`, cut from `main`. Draft PR to `main` opened on day zero so CI runs.
- **Three worktrees, three agents in parallel** off the umbrella:
  - `wt/downloads-ui` → Agent A
  - `wt/resource-telemetry` → Agent B
  - `wt/device-placement` → Agent C
- Each agent opens a non-draft PR **into the umbrella** when their phase is green.
- After all three land: main agent runs the full suite, fixes merge breakage, pushes; <gpu-host> pulls and final UAT runs. Then flip the umbrella PR to ready-for-review against `main`.
- **UAT cadence**: <gpu-host> syncs after each agent's merge into umbrella, and again after the final merge/fix pass.

---

## 1. Phase 1 — Downloads UI (Agent A)

### 1.1 Server — `mold-server/src/downloads.rs` (new module)

Single-writer queue over the existing `mold_core::download::pull_model_with_callback`.

```rust
pub struct DownloadJob {
    pub id: String,              // UUID v4, string for JSON ergonomics
    pub model: String,           // resolved name:tag
    pub status: JobStatus,
    pub files_done: usize,
    pub files_total: usize,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub current_file: Option<String>,
    pub started_at: Option<i64>,    // unix ms
    pub completed_at: Option<i64>,  // unix ms
    pub error: Option<String>,
}

pub enum JobStatus { Queued, Active, Completed, Failed, Cancelled }

pub struct DownloadQueue {
    active: tokio::sync::Mutex<Option<ActiveHandle>>,
    queued: std::sync::Mutex<VecDeque<DownloadJob>>,
    history: std::sync::Mutex<VecDeque<DownloadJob>>, // capped at 20, oldest evicted
    events: tokio::sync::broadcast::Sender<DownloadEvent>,
}

pub struct ActiveHandle {
    pub job: DownloadJob,
    pub abort: tokio_util::sync::CancellationToken,
    pub task: tokio::task::JoinHandle<()>,
}
```

**Event stream** (`tokio::sync::broadcast`, buffer 256):

```rust
pub enum DownloadEvent {
    Enqueued   { id, model, position },
    Dequeued   { id },               // removed from queue before starting
    Started    { id, files_total, bytes_total },
    Progress   { id, files_done, bytes_done, current_file },
    FileDone   { id, filename },
    JobDone    { id, model },
    JobFailed  { id, error },
    JobCancelled { id },
}
```

**Driver**: a single long-running task spawned in `AppState::new` takes the next queued job, sets `active`, spawns a pull task with the existing `pull_model_with_callback`, forwards each `DownloadProgressEvent` to a `DownloadEvent`, and cleans up on completion or cancellation. On failure: 1 auto-retry with 5 s backoff before emitting `JobFailed`.

**Cancellation semantics**:
- `active` cancel: `CancellationToken::cancel()`, `.abort()` the pull task, delete any partials under `MOLD_MODELS_DIR/<model>/` that don't match `*.sha256-verified` markers. HF cache under `~/.cache/huggingface` is left intact so retry is cheap.
- `queued` cancel: just remove from `VecDeque`, emit `Dequeued`.

### 1.2 Server — routes

All under `AppState` and wired in `mold-server/src/routes.rs`:

```
POST   /api/downloads          body: { "model": "flux-dev:q4" }
       → 200 { "id": "...", "position": 0 }   (0 = started, N = queued behind N jobs)
       → 400 if model unknown in manifest
       → 409 if that model already active/queued (idempotent: return existing id)

DELETE /api/downloads/:id      → 204 on success, 404 if id not found

GET    /api/downloads          → 200 { active, queued: [...], history: [...] }

GET    /api/downloads/stream   → text/event-stream, multiplexes all DownloadEvents
                                 with 15 s keepalive ping (same pattern as /api/generate/stream)
```

Existing `POST /api/models/pull` becomes a thin compatibility wrapper: enqueue via the queue; if the caller sent `Accept: text/event-stream`, stream only this job's events and close on `JobDone`/`JobFailed`. TUI continues to work unchanged.

### 1.3 Web — `DownloadsDrawer.vue` (new)

Opens from a new TopBar icon with a badge showing `active ? 1 + queued.length : queued.length`. Three sections:

- **Active** — one card with model name, size, stacked byte progress bar, files `N / M`, current filename, ETA (client-computed), and a cancel button.
- **Queued** — compact list, each row `model · 3.2 GB · #2` with an X button. No reorder in v1.
- **Recent** — collapsed by default, last 20 with status badge; failed rows show a Retry button that re-enqueues at head.

### 1.4 Web — `ModelPicker.vue` changes

- Always render `(3.2 GB)` next to every model (use `ModelManifest::total_size_bytes` → served in `ModelInfoExtended`).
- Undownloaded models: replace the disabled button with a compact "Download" inline button. Clicking POSTs `/api/downloads`.
- Models with active download: show an inline compact progress bar in place of the download button, unselectable until `JobDone`.
- Models in queue: show `Queued (#2)` chip with small X.
- Show-all toggle already exists — keep it.

### 1.5 Web — `useDownloads.ts` (new composable)

- Subscribes to `/api/downloads/stream` on mount (shared singleton, mounted in `App.vue` so state survives navigation).
- Maintains reactive `active: Ref<DownloadJob | null>`, `queued: Ref<DownloadJob[]>`, `history: Ref<DownloadJob[]>`.
- On `JobDone`: calls `fetchModels()` and emits a global event that `ModelPicker.vue` listens for.
- Computes ETA client-side from a 10-second sliding window of `{ts, bytes_done}` pairs. Server emits raw counters only.

### 1.6 ETA math (client)

```
history: Array<{ts: ms, bytes: u64}>   // keep only last 10 s
rate_bps = (history.last.bytes - history.first.bytes) / ((history.last.ts - history.first.ts) / 1000)
eta_sec  = (bytes_total - bytes_done) / rate_bps
```

If `rate_bps <= 0` or history is shorter than 2 samples, display `—`.

### 1.7 Downloads API types in `mold-core/src/types.rs`

Mirror the server structs (`DownloadJob`, `JobStatus`, `DownloadEvent`) on the Rust side for shared serde. Frontend gets them via the existing `types.ts` hand-maintained mirror (matches the project's current pattern).

### 1.8 Tests

- `mold-server`: `downloads_test.rs` — queue transitions (Enqueue → Started → Progress → JobDone), cancel while active, cancel while queued, retry-on-failure via a faked `pull_model_with_callback`.
- `mold-server`: `routes_test.rs` — `POST /api/downloads` happy path, unknown model → 400, duplicate enqueue → 409 + idempotent, DELETE happy path + 404.
- Web: `useDownloads.test.ts` — ETA calculation, SSE reconnection, history truncation. `DownloadsDrawer.test.ts` — renders active/queued/recent correctly from mock state.

---

## 2. Phase 2 — Resource Telemetry (Agent B)

### 2.1 Server — `mold-server/src/resources.rs` (new module)

```rust
pub struct ResourceSnapshot {
    pub hostname: String,
    pub timestamp: i64,  // unix ms
    pub gpus: Vec<GpuSnapshot>,
    pub system_ram: RamSnapshot,
}

pub struct GpuSnapshot {
    pub ordinal: usize,
    pub name: String,
    pub backend: GpuBackend,     // Cuda | Metal
    pub vram_total: u64,
    pub vram_used: u64,
    pub vram_used_by_mold: Option<u64>,   // None on Metal
    pub vram_used_by_other: Option<u64>,  // None on Metal
}

pub struct RamSnapshot {
    pub total: u64,
    pub used: u64,
    pub used_by_mold: u64,
    pub used_by_other: u64,      // total - used, OR used - mold, see 2.2
}

pub enum GpuBackend { Cuda, Metal }
```

### 2.2 Data sources

- **CUDA**: `nvml-wrapper` (new dep) — `Device::memory_info()` for totals, `Device::running_compute_processes()` filtered by `std::process::id()` for `vram_used_by_mold`. `vram_used_by_other = vram_used - vram_used_by_mold`. If NVML load fails at startup: fall back to the existing `nvidia-smi` subprocess in `device.rs:2445`, and set `vram_used_by_mold` / `vram_used_by_other` to `None` (SPA hides those rows).
- **Metal** (macOS): `sysinfo::System::total_memory` for unified-memory total; `used_memory` for used. Per-process GPU attribution unavailable → both `Option` fields `None`.
- **System RAM**: `sysinfo::System` (new dep, or reuse if already transitive). `used_by_mold = System::process(pid).memory()`. `used_by_other = used - used_by_mold`.

Compile probe: if `nvml-wrapper` conflicts with candle's `cudarc` linkage, fall back to pure subprocess path and drop the dep. Check on day zero.

### 2.3 Aggregator

A single `tokio::spawn` tick at 1 Hz builds a `ResourceSnapshot` and pushes into `tokio::sync::broadcast::Sender<ResourceSnapshot>` with buffer 4. Cheap enough to always run (~200 µs/tick).

### 2.4 Routes

```
GET  /api/resources         → 200 ResourceSnapshot (latest)
GET  /api/resources/stream  → text/event-stream, broadcast snapshots with 15 s keepalive
```

### 2.5 Web — `ResourceStrip.vue` (new)

Docked at the bottom of the Composer column in `GeneratePage.vue`. Always visible at `lg+`. One row per GPU plus one for system RAM:

```
GPU 0 · RTX 3090      14.2 / 24.0 GB  [████░░░░░░]  (mold 10.1, other 4.1)
GPU 1 · RTX 3090       0.8 / 24.0 GB  [░░░░░░░░░░]  (mold 0, other 0.8)
RAM                   38.4 / 64.0 GB  [██████░░░░]  (mold 22.1, other 16.3)
```

- Click any row to open a side-sheet with a numeric breakdown, hostname at the top, and the raw NVML-or-subprocess source tagged (for debugging remote issues).
- On `<lg`, collapse to a single chip in the TopBar: `🧠 14.2 GB · 64 GB`; tap opens a bottom sheet with the full layout.
- If `/api/resources` returns no GPUs (CPU-only host): show only the RAM row, no empty GPU panel.

### 2.6 Web — `useResources.ts` (new composable)

Singleton in `App.vue`. Subscribes to `/api/resources/stream`, exposes `snapshot: Ref<ResourceSnapshot | null>`, auto-reconnects on disconnect. Also exposes `gpuList: ComputedRef<GpuSnapshot[]>` that Agent C's `PlacementPanel.vue` consumes.

### 2.7 Tests

- `mold-server`: `resources_test.rs` — snapshot builder happy path, NVML-missing fallback, Metal path (behind `#[cfg(target_os = "macos")]`).
- `mold-server`: `routes_test.rs` — `GET /api/resources` returns valid JSON with expected shape.
- Web: `useResources.test.ts` — SSE reconnect, snapshot replacement. `ResourceStrip.test.ts` — renders correctly with CUDA data, hides per-process rows when `vram_used_by_mold` is None.

---

## 3. Phase 3 — Device Placement (Agent C)

### 3.1 Core types — `mold-core/src/types.rs`

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind", content = "ordinal")]
pub enum DeviceRef {
    Auto,              // keep current behavior
    Cpu,
    Gpu(usize),        // ordinal
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DevicePlacement {
    pub text_encoders: DeviceRef,             // Tier 1 — group knob
    pub advanced: Option<AdvancedPlacement>,  // Tier 2 — None = Tier 1 only
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdvancedPlacement {
    pub transformer: DeviceRef,
    pub vae: DeviceRef,
    pub clip_l: Option<DeviceRef>,
    pub clip_g: Option<DeviceRef>,
    pub t5: Option<DeviceRef>,
    pub qwen: Option<DeviceRef>,
}

impl Default for DeviceRef { fn default() -> Self { DeviceRef::Auto } }
```

New optional field on `GenerateRequest`:

```rust
pub placement: Option<DevicePlacement>,
```

`None` preserves current auto behavior end-to-end.

### 3.2 Engine support matrix

| Family | Tier 1 | Tier 2 | Agent C scope |
|---|---|---|---|
| FLUX (dev/schnell/fill/depth/canny) | ✅ | ✅ | In |
| Flux.2 (klein) | ✅ | ✅ | In |
| Z-Image | ✅ | ✅ | In |
| Qwen-Image | ✅ | ✅ | In |
| SD3.5 | ✅ | 🟡 | Stretch — cut cleanly if behind. Tier 1 must still land. |
| SDXL | ✅ | ❌ | Tier 1 only; Advanced panel disabled with tooltip. |
| SD1.5 | ✅ | ❌ | Same as SDXL. |
| Wuerstchen | ✅ | ❌ | Cascade architecture; Advanced disabled. |
| LTX-Video | ✅ | ❌ | Advanced disabled; Tier 1 limited to T5 device. |
| LTX-2 | ✅ | ❌ | CUDA-only engine; Metal hosts show everything disabled. |

"Tier 1 only" = we add a `text_encoders: DeviceRef` parameter to the engine's existing text-encoder selection logic, replacing the auto-decide branch when `!= Auto`. That's typically ~20 lines per engine.

### 3.3 Plumbing per Tier 2 family

For each Tier 2 engine, replace the fixed auto-device resolution in `load()` with a `resolve_device(placement.advanced?.field, auto_fallback)` helper:

```rust
// crates/mold-inference/src/device.rs (new helper)
pub fn resolve_device(
    req: Option<DeviceRef>,
    auto: impl FnOnce() -> Result<Device>,
) -> Result<Device> {
    match req {
        None | Some(DeviceRef::Auto) => auto(),
        Some(DeviceRef::Cpu) => Ok(Device::Cpu),
        Some(DeviceRef::Gpu(ord)) => Device::new_cuda(ord).or_else(|_| Device::new_metal(ord)),
    }
}
```

Touch points (~60 lines each):
- `flux/pipeline.rs`: T5, CLIP-L, transformer, VAE selection.
- `flux2/pipeline.rs`: Qwen3, transformer, VAE.
- `zimage/pipeline.rs`: Qwen3, transformer, VAE.
- `qwen_image/pipeline.rs`: Qwen2.5-VL, transformer, VAE.
- `sd3/pipeline.rs` (stretch): CLIP-L, CLIP-G, T5-XXL, MMDiT, VAE.

Block-level offload stays separately controlled (`MOLD_OFFLOAD`) — it interacts with `transformer: Gpu(n)` by still streaming blocks from CPU, just to the chosen ordinal.

### 3.4 Config — `mold-core/src/config.rs`

Per-model placement under the existing `ModelConfig`:

```toml
[models."flux-dev:q4".placement]
text_encoders = "auto"   # or "cpu", or { gpu = 1 }

[models."flux-dev:q4".placement.advanced]
transformer = { gpu = 0 }
vae = "cpu"
t5 = "cpu"
```

Env overrides follow the existing pattern: `MOLD_PLACE_TEXT_ENCODERS=cpu`, `MOLD_PLACE_TRANSFORMER=gpu:0`, etc.

New route for persistence:

```
PUT /api/config/model/:name/placement   body: DevicePlacement
    → 200 on success, 400 if invalid
```

### 3.5 Web — `PlacementPanel.vue` (new)

Collapsible section in `Composer.vue`, below the existing Settings disclosure:

- **Tier 1 single select**: options `Auto | CPU | GPU 0 (name) | GPU 1 (name) | …` populated from `useResources().gpuList`. Hidden if the list is empty (CPU-only host).
- **Advanced disclosure**:
  - For Tier 2 families: renders per-component selects (transformer, vae, and family-appropriate encoder fields). Encoder selects default to "follow Text encoders group". Each field has an inline ⓘ tooltip explaining what it controls.
  - For non-Tier-2 families: disclosure button is disabled, tooltip reads "Advanced placement is not yet available for `<family>` — Tier 1 controls all encoders as a group."
- **"Save as default for `<model>`"** button — active when the current placement differs from saved config. PUTs `/api/config/model/:name/placement` (see §3.4).
- On model select: load saved placement from the model config (already returned by `/api/models`).

### 3.6 Web — `usePlacement.ts` (new composable)

- Reactive `placement: Ref<DevicePlacement>` per model-name keyed cache.
- `supportsAdvanced(family): boolean` — hardcoded list matching the matrix in §3.2.
- Merges saved default + session override.
- Emits on change so `useGenerateForm` can attach `placement` to the outgoing `GenerateRequest`.

### 3.7 Tests

- `mold-core`: `placement_test.rs` — serde round-trip for `DevicePlacement` with all enum variants; config TOML round-trip.
- `mold-inference`: `device_test.rs` — `resolve_device` happy paths (Auto falls back, Cpu returns Cpu, Gpu returns Gpu, invalid Gpu surfaces a clear error).
- `mold-server`: `routes_test.rs` — `PUT /api/config/model/:name/placement` happy path + validation failures.
- Web: `usePlacement.test.ts` — saved-vs-session merge, family support gating.

---

## 4. Cross-cutting

### 4.1 Shared API contracts (frozen)

Everything in §1.2, §2.4, §3.4 is the **contract between agents**. Agents may not change request/response shapes without ping-back to the main agent, because:

- Agent A's drawer depends on `/api/downloads/stream`.
- Agent C's PlacementPanel depends on `/api/resources` via `useResources.gpuList`.
- Main agent's final merge assumes these shapes stable.

### 4.2 File-overlap map (minimizes merge conflict)

| File | A | B | C | Plan |
|---|---|---|---|---|
| `mold-server/src/routes.rs` | adds 4 routes + handlers | adds 2 routes + handlers | adds 1 route | All three append to the router `.route(...)` chain; merge order dictates order of the `.route` calls. Conflicts are trivial. |
| `mold-server/src/lib.rs` | mod decl | mod decl | — | Two-way, trivial. |
| `mold-server/src/state.rs` | add `downloads: Arc<DownloadQueue>` | add `resources: Arc<ResourceBroadcaster>` | — | Both append to `AppState`; Agents A + B coordinate on field order. |
| `mold-core/src/types.rs` | `DownloadJob`, `JobStatus`, `DownloadEvent` | `ResourceSnapshot`, `GpuSnapshot`, `RamSnapshot`, `GpuBackend` | `DeviceRef`, `DevicePlacement`, `AdvancedPlacement` | All additive, separate sections. |
| `mold-core/src/config.rs` | — | — | nested `[models."name:tag".placement]` | Clean single-agent touch. |
| `web/src/App.vue` | mount `useDownloads` singleton + `DownloadsDrawer` | mount `useResources` singleton | — | Two-agent; trivial append to `<script>`/`<template>`. |
| `web/src/components/TopBar.vue` | Downloads button + badge | optional narrow-viewport chip | — | Two-agent; distinct zones. |
| `web/src/pages/GeneratePage.vue` | — | render `ResourceStrip` | — | Single-agent. |
| `web/src/components/Composer.vue` | — | — | render `PlacementPanel` | Single-agent. |
| `web/src/components/ModelPicker.vue` | show size + download button | — | — | Single-agent. |
| `web/src/types.ts` | Download types | Resource types | Placement types | All additive, separate sections. |

### 4.3 Testing gates (per-agent, green before PR-into-umbrella)

Every agent must pass locally:

```
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
( cd web && bun run fmt:check && bun run test && bun run build )
```

Plus at least one new test file proving the phase's core behavior (§1.8, §2.7, §3.7).

### 4.4 Manual UAT on <gpu-host>

After each agent merges into umbrella:

1. On <gpu-host>: `cd ~/github/mold && git fetch && git checkout feat/model-ui-overhaul && git pull`.
2. `cargo build --release --features cuda` (<arch-tag> compute cap).
3. Restart `mold serve` (check for existing systemd user service first).
4. Hit `http://<gpu-host>:7680/` from this machine's browser.

Scenarios to walk:

- **Downloads**: start a small model (e.g. `sd1.5:fp16`), watch progress; while it's running, enqueue a second (`flux-schnell:q4`); cancel the second, cancel the first mid-stream, verify files cleaned up; start a gated model without `HF_TOKEN`, verify error surfaces cleanly; reload browser mid-download, verify queue reappears.
- **Telemetry**: confirm both 3090 rows appear with names, total 24 GB each; load a model on GPU 0, watch `mold` number rise on that row only; run a competing process (simple `python -c 'import torch; torch.cuda.set_device(1); x = torch.zeros(4_000_000_000, dtype=torch.uint8, device="cuda")'`), watch "other" rise on GPU 1.
- **Placement**: with FLUX selected, set text encoders to CPU, generate, verify it still works (slower); set transformer to GPU 1 via Advanced, verify allocation shifts to that device; save as default, reload, verify persistence; switch to SDXL, verify Advanced is disabled with tooltip.

### 4.5 Risks

- **`nvml-wrapper` vs `cudarc` linker conflict** — candle already links CUDA; adding NVML may double-link. Mitigation: compile probe on Agent B day 1; fall back to the `nvidia-smi` subprocess path we already use in `device.rs:2445` and drop the dep.
- **`hf-hub` cancellation leaves partials** — dropping the tokio future is safe but files on disk may be half-written. Mitigation: explicit cleanup pass in the cancel path, tested in `downloads_test.rs`.
- **SD3.5 Tier 2 scope creep** — stretch. Cut without ceremony if Agent C is behind; Tier 1 for SD3.5 must still land.
- **Three-way merge noise in `types.rs` / `routes.rs`** — mitigated by file-overlap map (§4.2). Conflicts expected to be mechanical append-merges.
- **Remote-host resource attribution subtleties** — if <gpu-host>'s `mold` runs inside a shared user session, `used_by_mold` only counts our PID, not child processes. For v1 that's acceptable (mold doesn't fork CUDA-using children), but we call it out.

---

## 5. Open Questions (pre-implementation)

- **NVML dep** — decided at compile-probe time on Agent B day 1. Default: try `nvml-wrapper`, fall back to subprocess if link fails. Document the outcome in the implementation plan.
- **SD3.5 Tier 2** — flagged as stretch, go/no-go at Agent C's midpoint.

## 6. Sign-off

This spec is the contract for Agents A/B/C. Any agent needing to deviate from an API shape in §1.2/§2.4/§3.4 must pause and ping the main agent for cross-agent alignment.
