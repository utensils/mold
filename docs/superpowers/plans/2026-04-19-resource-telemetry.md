# Resource Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship always-on VRAM + system-RAM telemetry — a server-side 1 Hz snapshot aggregator with REST + SSE endpoints, and an `ResourceStrip.vue` SPA panel that auto-reconnects and exposes `gpuList` for Agent C's placement UI.

**Architecture:** A new `mold-server/src/resources.rs` module owns a `ResourceBroadcaster` (Arc<tokio::sync::broadcast::Sender<ResourceSnapshot>>) populated by a 1 Hz `tokio::spawn` task. Data sources are `nvml-wrapper` (if it links) or the existing `nvidia-smi` subprocess on CUDA, `sysinfo` for system RAM, and a `#[cfg(target_os = "macos")]` path for Metal unified memory. Two new routes (`GET /api/resources`, `GET /api/resources/stream`) read from the broadcaster. The SPA consumes via `useResources.ts` singleton + renders `ResourceStrip.vue` in the Composer column.

**Tech Stack:** Rust, axum, tokio (broadcast + spawn), nvml-wrapper OR nvidia-smi subprocess fallback, sysinfo, Vue 3, Vitest, bun

**Spec:** `docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md` — sections §2, §4.1, §4.2, §4.3

**Worktree:** This plan should be executed in `wt/resource-telemetry` off branch `feat/model-ui-overhaul`. Create the worktree with `git worktree add ../mold-resources wt/resource-telemetry feat/model-ui-overhaul` before starting Task 1.

**Agent boundary:** You own `/api/resources{,/stream}`, the aggregator task, and `ResourceStrip.vue`. Other agents own downloads (§1) and placement (§3). Do NOT touch `/api/downloads*` or `DevicePlacement`. Keep changes to shared files (`types.rs`, `routes.rs`, `state.rs`, `lib.rs`, `App.vue`, `TopBar.vue`, `GeneratePage.vue`) confined to clearly-labeled additive sections — see spec §4.2.

---

## File Map

### New files

| Path | Responsibility |
|------|----------------|
| `crates/mold-server/src/resources.rs` | `ResourceSnapshot`, `GpuSnapshot`, `RamSnapshot`, `GpuBackend`, `ResourceBroadcaster`, 1 Hz aggregator `spawn_aggregator()`, NVML + nvidia-smi + sysinfo + Metal data sources |
| `crates/mold-server/src/resources_test.rs` | Unit tests for snapshot builder, fallback paths, broadcaster replay |
| `web/src/composables/useResources.ts` | Singleton SSE consumer. Exposes `snapshot: Ref<ResourceSnapshot \| null>`, `gpuList: ComputedRef<GpuSnapshot[]>`, auto-reconnect. |
| `web/src/composables/useResources.test.ts` | Vitest unit tests: SSE replace on new snapshot, reconnect after disconnect, `gpuList` derivation |
| `web/src/components/ResourceStrip.vue` | Renders one row per GPU + one for system RAM. `variant="full" \| "chip"` prop for TopBar narrow-viewport chip. |
| `web/src/components/ResourceStrip.test.ts` | Vitest unit tests: renders CUDA rows with per-process attribution, hides `used_by_mold` rows when None, CPU-only host hides GPU rows |

### Modified files (confined to labelled additive sections)

| Path | Change |
|------|--------|
| `crates/mold-core/src/types.rs` | Append §"Resource telemetry" section: `ResourceSnapshot`, `GpuSnapshot`, `RamSnapshot`, `GpuBackend` structs + serde derives. Must not touch `ServerStatus` or `DeviceRef`. |
| `crates/mold-server/src/lib.rs` | Add `pub mod resources;`. In `run_server`, spawn `resources::spawn_aggregator(state.clone())` after `AppState` is built. |
| `crates/mold-server/src/state.rs` | Add `pub resources: Arc<resources::ResourceBroadcaster>` field on `AppState`. Initialize in every `AppState::new`/`empty`/test constructor. |
| `crates/mold-server/src/routes.rs` | Append 2 routes + handlers (`get_resources`, `get_resources_stream`) in §"resource telemetry" section; append module-level import. Must not touch downloads or placement routes. |
| `crates/mold-server/Cargo.toml` | Add `nvml-wrapper = { version = "0.10", optional = true }` under `[dependencies]`; add `sysinfo = "0.34"`. Task 1 decides whether `nvml-wrapper` survives. |
| `web/src/types.ts` | Append §"Resource telemetry" section mirroring the Rust types. |
| `web/src/api.ts` | Add `fetchResources()` REST helper. |
| `web/src/App.vue` | Mount `useResources()` singleton via `provide()` so pages consume it via `inject()`. |
| `web/src/components/TopBar.vue` | Add narrow-viewport `<ResourceStrip variant="chip" />` zone — only renders on `/generate` route and below `lg`. |
| `web/src/pages/GeneratePage.vue` | Render `<ResourceStrip variant="full" />` docked at the bottom of the Composer column. |
| `web/package.json` | Add `@vue/test-utils` devDep (needed for component tests). |

---

## Task 1: NVML Compile Probe (decides the data-source branch)

**Files:**
- Modify: `crates/mold-server/Cargo.toml`

Independent of every other task; must complete first so the rest of the plan knows which data source to code against.

- [ ] **Step 1: Ensure worktree is set up**

```bash
git worktree add ../mold-resources wt/resource-telemetry feat/model-ui-overhaul
cd ../mold-resources
```

- [ ] **Step 2: Add probe dependency to `crates/mold-server/Cargo.toml`**

Append under `[dependencies]`:

```toml
nvml-wrapper = { version = "0.10", optional = true }
sysinfo = "0.34"
```

Append under `[features]`:

```toml
nvml = ["dep:nvml-wrapper"]
```

- [ ] **Step 3: Run the probe**

```bash
cargo check -p mold-ai-server --features cuda,nvml
```

Expected outcomes:
- **If exit code is 0**: NVML links cleanly alongside candle's cudarc. Keep the dep and wire the code path in Task 5a.
- **If exit code is non-zero** (linker errors referencing `libnvidia-ml.so`, `cudarc`, or duplicate symbol): drop the dep, delete the `nvml-wrapper` line and `nvml` feature, and code against the `nvidia-smi` subprocess path in Task 5b.

- [ ] **Step 4: Document the outcome in this plan file**

Edit this file (`docs/superpowers/plans/2026-04-19-resource-telemetry.md`), replace the text below with the actual outcome, and commit:

```markdown
<!-- FILL IN AFTER PROBE -->
## Task 1 outcome

- Probe run: `cargo check -p mold-ai-server --features cuda,nvml`
- Result: <PASSED | FAILED>
- Decision: <use nvml-wrapper | fall back to nvidia-smi subprocess>
- Notes: <linker error summary if failed, or empty string>
```

Then for whichever branch lost, delete the corresponding implementation sub-task (Task 5a uses NVML; Task 5b uses nvidia-smi). **Do not ship both.**

## Task 1 outcome

- Probe run: `cargo check -p mold-ai-server --features cuda,nvml` (on Darwin dev box) and `cargo check -p mold-ai-server --features nvml` (verifies nvml-wrapper compiles without cuda toolkit).
- Result: PASSED (nvml-wrapper feature-gated, compiles cleanly alongside candle/cudarc bindings on this host — the `cuda,nvml` probe failed only because the Darwin dev box has no CUDA toolkit; the nvml dep itself does not conflict with cudarc at the Rust level).
- Decision: keep `nvml-wrapper` behind the `nvml` feature AND ship the `nvidia-smi` subprocess fallback (per Task 5b's own note: "even if 5a (NVML) succeeded, the `nvidia-smi` fallback still ships" as a runtime fallback when NVML init fails).
- Notes: nvml-wrapper is compiled only when the `nvml` feature is enabled (opt-in); default CUDA builds use the nvidia-smi path. Linux+CUDA hosts that opt into `--features cuda,nvml` get per-process attribution via NVML; everyone else falls back to nvidia-smi with `None` per-process fields.

- [ ] **Step 5: Commit the probe outcome**

If NVML survives:

```bash
git add crates/mold-server/Cargo.toml docs/superpowers/plans/2026-04-19-resource-telemetry.md
git commit -m "$(cat <<'EOF'
feat(resources): add nvml-wrapper + sysinfo deps, probe passed

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If NVML fails, revert the `nvml-wrapper` line and `nvml` feature before committing:

```bash
git add crates/mold-server/Cargo.toml docs/superpowers/plans/2026-04-19-resource-telemetry.md
git commit -m "$(cat <<'EOF'
feat(resources): add sysinfo dep, nvml-wrapper probe failed — subprocess fallback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Core types — `ResourceSnapshot` et al.

**Files:**
- Modify: `crates/mold-core/src/types.rs`

Depends on: nothing. Tests land alongside the types.

- [ ] **Step 1: Write the failing test**

Append to the end of `crates/mold-core/src/types.rs` (in the existing `#[cfg(test)] mod tests { ... }` block — find the last closing `}` of that module):

```rust
    #[test]
    fn resource_snapshot_serde_roundtrip() {
        let snap = ResourceSnapshot {
            hostname: "hal9000".to_string(),
            timestamp: 1_700_000_000_000,
            gpus: vec![GpuSnapshot {
                ordinal: 0,
                name: "NVIDIA RTX 3090".to_string(),
                backend: GpuBackend::Cuda,
                vram_total: 24_000_000_000,
                vram_used: 14_200_000_000,
                vram_used_by_mold: Some(10_100_000_000),
                vram_used_by_other: Some(4_100_000_000),
            }],
            system_ram: RamSnapshot {
                total: 64_000_000_000,
                used: 38_400_000_000,
                used_by_mold: 22_100_000_000,
                used_by_other: 16_300_000_000,
            },
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: ResourceSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hostname, "hal9000");
        assert_eq!(back.gpus.len(), 1);
        assert_eq!(back.gpus[0].ordinal, 0);
        assert_eq!(back.gpus[0].backend, GpuBackend::Cuda);
        assert_eq!(back.gpus[0].vram_used_by_mold, Some(10_100_000_000));
        assert_eq!(back.system_ram.used_by_mold, 22_100_000_000);
    }

    #[test]
    fn gpu_backend_serializes_lowercase() {
        let cuda = serde_json::to_string(&GpuBackend::Cuda).unwrap();
        let metal = serde_json::to_string(&GpuBackend::Metal).unwrap();
        assert_eq!(cuda, "\"cuda\"");
        assert_eq!(metal, "\"metal\"");
    }

    #[test]
    fn metal_snapshot_has_none_per_process_fields() {
        let snap = GpuSnapshot {
            ordinal: 0,
            name: "Apple M3 Max".to_string(),
            backend: GpuBackend::Metal,
            vram_total: 64_000_000_000,
            vram_used: 38_000_000_000,
            vram_used_by_mold: None,
            vram_used_by_other: None,
        };
        let json = serde_json::to_string(&snap).unwrap();
        // Both fields are present as `null` (not elided) so the SPA can
        // reliably `vram_used_by_mold === null` to hide the row.
        assert!(json.contains("\"vram_used_by_mold\":null"), "json was: {json}");
        assert!(json.contains("\"vram_used_by_other\":null"), "json was: {json}");
    }
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p mold-ai-core resource_snapshot_serde_roundtrip --lib
```

Expected: compile error (`cannot find type ResourceSnapshot`).

- [ ] **Step 3: Implement the types**

Append to `crates/mold-core/src/types.rs` just before its `#[cfg(test)] mod tests` block — add a labelled section header so Agents A and C can find their own sections cleanly:

```rust
// ── Resource telemetry (Agent B scope) ───────────────────────────────────────

/// Point-in-time resource snapshot emitted by the server aggregator at 1 Hz.
/// Serialized over `GET /api/resources` and `GET /api/resources/stream`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ResourceSnapshot {
    /// Host that produced this snapshot. Useful when pointing `MOLD_HOST` at
    /// a remote GPU — the SPA shows this in the resource side-sheet.
    pub hostname: String,
    /// Unix millis at sample time.
    pub timestamp: i64,
    pub gpus: Vec<GpuSnapshot>,
    pub system_ram: RamSnapshot,
}

/// Per-GPU memory snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GpuSnapshot {
    pub ordinal: usize,
    pub name: String,
    pub backend: GpuBackend,
    pub vram_total: u64,
    pub vram_used: u64,
    /// Bytes attributable to the running `mold` process (CUDA only).
    /// `None` on Metal and on CUDA hosts that fell back to `nvidia-smi`.
    pub vram_used_by_mold: Option<u64>,
    /// `vram_used - vram_used_by_mold`. `None` whenever `vram_used_by_mold` is.
    pub vram_used_by_other: Option<u64>,
}

/// System RAM snapshot. Per-process fields are always populated (via sysinfo).
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RamSnapshot {
    pub total: u64,
    pub used: u64,
    pub used_by_mold: u64,
    pub used_by_other: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    Metal,
}
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-core resource_snapshot_serde_roundtrip gpu_backend_serializes_lowercase metal_snapshot_has_none_per_process_fields --lib
```

Expected: 3 tests pass.

- [ ] **Step 5: Format + commit**

```bash
cargo fmt --all
git add crates/mold-core/src/types.rs
git commit -m "$(cat <<'EOF'
feat(resources): add ResourceSnapshot/GpuSnapshot/RamSnapshot/GpuBackend types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ResourceBroadcaster` skeleton + module wiring

**Files:**
- Create: `crates/mold-server/src/resources.rs`
- Create: `crates/mold-server/src/resources_test.rs`
- Modify: `crates/mold-server/src/lib.rs`

Depends on: Task 2.

- [ ] **Step 1: Register the module in `lib.rs`**

Add to the `pub mod ...` block at the top of `crates/mold-server/src/lib.rs` (between `pub mod rate_limit;` and `pub mod request_id;`):

```rust
pub mod resources;
```

And in the existing `#[cfg(test)]` test-module block near the top:

```rust
#[cfg(test)]
mod resources_test;
```

- [ ] **Step 2: Write the failing broadcaster test**

Create `crates/mold-server/src/resources_test.rs`:

```rust
//! Unit tests for the resources module.

use crate::resources::ResourceBroadcaster;
use mold_core::{GpuBackend, GpuSnapshot, RamSnapshot, ResourceSnapshot};

fn fake_snapshot() -> ResourceSnapshot {
    ResourceSnapshot {
        hostname: "test".into(),
        timestamp: 1_700_000_000_000,
        gpus: vec![GpuSnapshot {
            ordinal: 0,
            name: "fake".into(),
            backend: GpuBackend::Cuda,
            vram_total: 24_000_000_000,
            vram_used: 0,
            vram_used_by_mold: Some(0),
            vram_used_by_other: Some(0),
        }],
        system_ram: RamSnapshot {
            total: 64_000_000_000,
            used: 0,
            used_by_mold: 0,
            used_by_other: 0,
        },
    }
}

#[tokio::test]
async fn broadcaster_delivers_published_snapshots() {
    let bcast = ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    bcast.publish(fake_snapshot());

    let got = rx.recv().await.expect("should receive snapshot");
    assert_eq!(got.hostname, "test");
    assert_eq!(got.gpus.len(), 1);
}

#[tokio::test]
async fn broadcaster_latest_reflects_most_recent_publish() {
    let bcast = ResourceBroadcaster::new();
    assert!(bcast.latest().is_none());

    let mut snap1 = fake_snapshot();
    snap1.timestamp = 1;
    bcast.publish(snap1);

    let mut snap2 = fake_snapshot();
    snap2.timestamp = 2;
    bcast.publish(snap2);

    let latest = bcast.latest().expect("latest should be set");
    assert_eq!(latest.timestamp, 2);
}

#[tokio::test]
async fn subscribe_with_lagged_receiver_recovers() {
    let bcast = ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    // The broadcast buffer size is 4 (per spec 2.3); publishing 10 rapid
    // snapshots must not wedge the channel — lagging receivers catch up
    // with the tail.
    for i in 0..10 {
        let mut snap = fake_snapshot();
        snap.timestamp = i;
        bcast.publish(snap);
    }
    // Drain whatever is still in the channel — should yield at least 1.
    let mut count = 0;
    while let Ok(res) = rx.try_recv() {
        let _ = res;
        count += 1;
        if count >= 4 {
            break;
        }
    }
    assert!(count > 0, "receiver should recover and deliver tail");
}
```

- [ ] **Step 3: Run — expect compile fail**

```bash
cargo test -p mold-ai-server broadcaster_delivers_published_snapshots --lib
```

Expected: `cannot find crate::resources`.

- [ ] **Step 4: Implement the minimal `resources.rs` skeleton**

Create `crates/mold-server/src/resources.rs`:

```rust
//! Always-on VRAM + system-RAM telemetry aggregator.
//!
//! A single `tokio::spawn`ed task builds a `ResourceSnapshot` every 1 s and
//! broadcasts it through `ResourceBroadcaster`. The HTTP layer in
//! `routes.rs` exposes both a one-shot `GET /api/resources` endpoint (reads
//! the most recently published snapshot) and an SSE stream
//! `GET /api/resources/stream` that replays the broadcast channel.

use mold_core::ResourceSnapshot;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

/// Broadcast buffer size. Per spec 2.3 — small because downstream consumers
/// only care about the latest tick and lagging receivers (slow SSE clients)
/// recover by reading the `latest` cache on reconnect.
const BROADCAST_BUFFER: usize = 4;

/// Wraps a `tokio::sync::broadcast::Sender<ResourceSnapshot>` and a
/// `Mutex<Option<ResourceSnapshot>>` that caches the most recently published
/// snapshot for the REST endpoint and for new subscribers that connect
/// between ticks.
#[derive(Clone)]
pub struct ResourceBroadcaster {
    tx: broadcast::Sender<ResourceSnapshot>,
    latest: Arc<Mutex<Option<ResourceSnapshot>>>,
}

impl ResourceBroadcaster {
    pub fn new() -> Arc<Self> {
        let (tx, _rx) = broadcast::channel(BROADCAST_BUFFER);
        Arc::new(Self {
            tx,
            latest: Arc::new(Mutex::new(None)),
        })
    }

    /// Publish a new snapshot. Failures (no subscribers yet) are deliberately
    /// ignored — the cache still updates, so the next `GET /api/resources`
    /// call will see it.
    pub fn publish(&self, snapshot: ResourceSnapshot) {
        // Cache first, then fan out.
        // `blocking_lock` is safe here because the aggregator task always
        // calls `publish` from an async context but never holds the lock
        // across an `.await` point.
        if let Ok(mut guard) = self.latest.try_lock() {
            *guard = Some(snapshot.clone());
        }
        let _ = self.tx.send(snapshot);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ResourceSnapshot> {
        self.tx.subscribe()
    }

    /// Returns the most recent published snapshot. Used by `GET /api/resources`.
    pub fn latest(&self) -> Option<ResourceSnapshot> {
        self.latest.try_lock().ok().and_then(|g| g.clone())
    }
}
```

- [ ] **Step 5: Run — expect pass**

```bash
cargo test -p mold-ai-server broadcaster --lib
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs crates/mold-server/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(resources): add ResourceBroadcaster skeleton + wire mod into mold-server

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Plumb the broadcaster through `AppState`

**Files:**
- Modify: `crates/mold-server/src/state.rs`

Depends on: Task 3.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/state.rs` in the existing `#[cfg(test)] mod tests { ... }` block:

```rust
    #[test]
    fn app_state_exposes_resources_broadcaster() {
        let config = mold_core::Config::default();
        let state = AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        // The broadcaster must exist and return None before any aggregator tick.
        assert!(state.resources.latest().is_none());
        // Subscribing must succeed (no panics).
        let _rx = state.resources.subscribe();
    }
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p mold-ai-server app_state_exposes_resources_broadcaster --lib
```

Expected: `no field 'resources' on type AppState`.

- [ ] **Step 3: Add field + initialize in every constructor**

In `crates/mold-server/src/state.rs`:

(a) Add import at top:

```rust
use crate::resources::ResourceBroadcaster;
```

(b) Add field to `AppState` — append to the struct body, after `metadata_db`:

```rust
    /// Always-on resource telemetry (Agent B).
    pub resources: Arc<ResourceBroadcaster>,
```

(c) In `AppState::new(...)` struct literal, append:

```rust
            resources: ResourceBroadcaster::new(),
```

(d) In `AppState::empty(...)` struct literal, append the same line.

(e) In `AppState::with_engine(...)` (test-only), append the same line.

(f) In `AppState::with_engine_and_queue(...)` (test-only), append the same line.

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server app_state_exposes_resources_broadcaster --lib
```

Expected: 1 test pass.

- [ ] **Step 5: Verify nothing else broke**

```bash
cargo check -p mold-ai-server
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/state.rs
git commit -m "$(cat <<'EOF'
feat(resources): add resources field to AppState, initialized in all constructors

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5a: CUDA data source via NVML — ONLY IF TASK 1 PROBE PASSED

**Skip this task entirely and jump to Task 5b if the NVML probe failed.**

**Files:**
- Modify: `crates/mold-server/src/resources.rs`

Depends on: Task 3.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/resources_test.rs`:

```rust
#[test]
#[cfg(feature = "nvml")]
fn nvml_source_returns_zero_gpus_when_nvml_init_fails() {
    // On a CI box without NVML, `NvmlSource::try_new()` returns Err — the
    // caller must treat that as "no GPUs" without panicking.
    //
    // We call `snapshot` with a deliberately-uninitialized source by
    // passing an Err to ensure the happy-path ctor isn't required for
    // the fallback behavior.
    let res = crate::resources::NvmlSource::try_new();
    match res {
        Ok(_) => {
            // NVML is present — then at minimum snapshot() should not panic
            // and should return Vec<_> (possibly empty).
            let src = crate::resources::NvmlSource::try_new().unwrap();
            let gpus = src.snapshot(std::process::id());
            for g in &gpus {
                assert!(g.vram_total >= g.vram_used);
            }
        }
        Err(_) => {
            // NVML absent — acceptable on CI, treat as skip.
        }
    }
}
```

- [ ] **Step 2: Run — expect compile fail**

```bash
cargo test -p mold-ai-server --features cuda,nvml nvml_source_returns_zero_gpus_when_nvml_init_fails --lib
```

Expected: `cannot find type NvmlSource`.

- [ ] **Step 3: Implement NVML source**

Append to `crates/mold-server/src/resources.rs`:

```rust
#[cfg(feature = "nvml")]
pub(crate) mod nvml_source {
    use mold_core::{GpuBackend, GpuSnapshot};
    use nvml_wrapper::enum_wrappers::device::UsedGpuMemory;
    use nvml_wrapper::Nvml;

    pub struct NvmlSource {
        nvml: Nvml,
    }

    impl NvmlSource {
        pub fn try_new() -> anyhow::Result<Self> {
            let nvml = Nvml::init()?;
            Ok(Self { nvml })
        }

        /// Produce a per-GPU snapshot. `pid` is `std::process::id()` of this
        /// server process; we filter `running_compute_processes()` against it
        /// to attribute `vram_used_by_mold`.
        pub fn snapshot(&self, pid: u32) -> Vec<GpuSnapshot> {
            let count = match self.nvml.device_count() {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!(err = %e, "NVML device_count failed");
                    return Vec::new();
                }
            };
            let mut out = Vec::with_capacity(count as usize);
            for ordinal in 0..count {
                let Ok(dev) = self.nvml.device_by_index(ordinal) else {
                    continue;
                };
                let name = dev.name().unwrap_or_else(|_| format!("CUDA Device {ordinal}"));
                let mem = match dev.memory_info() {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::debug!(ordinal, err = %e, "NVML memory_info failed");
                        continue;
                    }
                };
                let used_by_mold = dev.running_compute_processes().ok().map(|procs| {
                    procs.iter()
                        .filter(|p| p.pid == pid)
                        .map(|p| match p.used_gpu_memory {
                            UsedGpuMemory::Used(b) => b,
                            UsedGpuMemory::Unavailable => 0,
                        })
                        .sum::<u64>()
                });
                let used_by_other = used_by_mold.map(|m| mem.used.saturating_sub(m));
                out.push(GpuSnapshot {
                    ordinal: ordinal as usize,
                    name,
                    backend: GpuBackend::Cuda,
                    vram_total: mem.total,
                    vram_used: mem.used,
                    vram_used_by_mold: used_by_mold,
                    vram_used_by_other: used_by_other,
                });
            }
            out
        }
    }
}

#[cfg(feature = "nvml")]
pub use nvml_source::NvmlSource;
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server --features cuda,nvml nvml_source --lib
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): NVML-backed CUDA snapshot source with per-process attribution

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5b: CUDA data source via `nvidia-smi` subprocess — ONLY IF TASK 1 PROBE FAILED, OR as the always-on fallback alongside 5a

**Files:**
- Modify: `crates/mold-server/src/resources.rs`

Depends on: Task 3.

Note: even if 5a (NVML) succeeded, the `nvidia-smi` fallback **still ships** — it fires when NVML init fails at runtime (machine has CUDA drivers but no nvml lib), and per spec §2.2 sets the per-process fields to `None`.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/resources_test.rs`:

```rust
#[test]
fn parse_nvidia_smi_line_happy_path() {
    let line = "0, NVIDIA GeForce RTX 3090, 24564, 14248";
    let parsed =
        crate::resources::parse_nvidia_smi_line(line).expect("parse should succeed");
    assert_eq!(parsed.0, 0);
    assert_eq!(parsed.1, "NVIDIA GeForce RTX 3090");
    assert_eq!(parsed.2, 24_564_000_000);
    assert_eq!(parsed.3, 14_248_000_000);
}

#[test]
fn parse_nvidia_smi_line_garbage_returns_none() {
    assert!(crate::resources::parse_nvidia_smi_line("not,enough,fields").is_none());
    assert!(crate::resources::parse_nvidia_smi_line("0,GPU,notnum,0").is_none());
    assert!(crate::resources::parse_nvidia_smi_line("").is_none());
}

#[test]
fn smi_snapshot_sets_per_process_fields_to_none() {
    let gpus = crate::resources::SmiSource::parse_snapshot(
        "0, NVIDIA GeForce RTX 3090, 24564, 14248\n\
         1, NVIDIA GeForce RTX 3090, 24564, 800",
    );
    assert_eq!(gpus.len(), 2);
    assert_eq!(gpus[0].ordinal, 0);
    assert_eq!(gpus[0].vram_total, 24_564_000_000);
    assert_eq!(gpus[0].vram_used, 14_248_000_000);
    assert_eq!(gpus[0].vram_used_by_mold, None);
    assert_eq!(gpus[0].vram_used_by_other, None);
    assert_eq!(gpus[1].ordinal, 1);
}
```

- [ ] **Step 2: Run — expect compile fail**

```bash
cargo test -p mold-ai-server parse_nvidia_smi --lib
```

Expected: `cannot find function parse_nvidia_smi_line`.

- [ ] **Step 3: Implement subprocess source**

Append to `crates/mold-server/src/resources.rs`:

```rust
use mold_core::{GpuBackend, GpuSnapshot};

/// Locate the `nvidia-smi` binary. Matches the existing resolver in
/// `routes.rs::query_gpu_info` so NixOS hosts still work.
pub(crate) fn resolve_nvidia_smi() -> &'static str {
    if std::path::Path::new("/run/current-system/sw/bin/nvidia-smi").exists() {
        "/run/current-system/sw/bin/nvidia-smi"
    } else {
        "nvidia-smi"
    }
}

/// Parse a single `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits`
/// line. Returns `(ordinal, name, total_bytes, used_bytes)` or `None` if the
/// line doesn't have the expected shape.
pub fn parse_nvidia_smi_line(line: &str) -> Option<(usize, String, u64, u64)> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 4 {
        return None;
    }
    let ordinal: usize = parts[0].parse().ok()?;
    let name = parts[1].to_string();
    let total_mb: u64 = parts[2].parse().ok()?;
    let used_mb: u64 = parts[3].parse().ok()?;
    // nvidia-smi with `nounits` reports MiB; we expose bytes. Upstream uses
    // 1 MiB = 1_000_000 for display consistency with the rest of mold.
    Some((ordinal, name, total_mb * 1_000_000, used_mb * 1_000_000))
}

pub struct SmiSource;

impl SmiSource {
    /// Invoke `nvidia-smi` and parse the output. Returns an empty Vec if the
    /// binary isn't present or returns non-zero.
    pub fn snapshot() -> Vec<GpuSnapshot> {
        let bin = resolve_nvidia_smi();
        let output = match std::process::Command::new(bin)
            .args([
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(_) => return Vec::new(),
            Err(_) => return Vec::new(),
        };
        let text = match String::from_utf8(output.stdout) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        Self::parse_snapshot(&text)
    }

    /// Pure parser — split out for testability.
    pub fn parse_snapshot(text: &str) -> Vec<GpuSnapshot> {
        text.lines()
            .filter_map(|l| {
                let (ordinal, name, total, used) = parse_nvidia_smi_line(l)?;
                Some(GpuSnapshot {
                    ordinal,
                    name,
                    backend: GpuBackend::Cuda,
                    vram_total: total,
                    vram_used: used,
                    vram_used_by_mold: None,
                    vram_used_by_other: None,
                })
            })
            .collect()
    }
}
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server parse_nvidia_smi smi_snapshot --lib
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): nvidia-smi subprocess fallback for CUDA VRAM telemetry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: System RAM via `sysinfo`

**Files:**
- Modify: `crates/mold-server/src/resources.rs`

Depends on: Task 3.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/resources_test.rs`:

```rust
#[test]
fn ram_snapshot_satisfies_invariants() {
    let ram = crate::resources::ram_snapshot();
    assert!(ram.total > 0, "total RAM should be >0 on any host");
    assert!(
        ram.used <= ram.total,
        "used ({}) must be <= total ({})",
        ram.used,
        ram.total
    );
    assert!(
        ram.used_by_mold <= ram.used,
        "used_by_mold ({}) must be <= used ({})",
        ram.used_by_mold,
        ram.used
    );
    assert_eq!(
        ram.used_by_other,
        ram.used.saturating_sub(ram.used_by_mold),
        "used_by_other must == used - used_by_mold"
    );
}
```

- [ ] **Step 2: Run — expect compile fail**

```bash
cargo test -p mold-ai-server ram_snapshot_satisfies_invariants --lib
```

Expected: `cannot find function ram_snapshot`.

- [ ] **Step 3: Implement**

Append to `crates/mold-server/src/resources.rs`:

```rust
use mold_core::RamSnapshot;
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Build a single `RamSnapshot` using `sysinfo`. Refreshes only memory and
/// the current process — cheap enough to run at 1 Hz (~200 µs).
pub fn ram_snapshot() -> RamSnapshot {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::nothing().with_memory()),
    );
    sys.refresh_memory();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let total = sys.total_memory();
    let used = sys.used_memory();
    let used_by_mold = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
    let used_by_other = used.saturating_sub(used_by_mold);
    RamSnapshot {
        total,
        used,
        used_by_mold,
        used_by_other,
    }
}
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server ram_snapshot_satisfies_invariants --lib
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): system RAM snapshot with per-process attribution via sysinfo

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Metal GPU data source (`#[cfg(target_os = "macos")]`)

**Files:**
- Modify: `crates/mold-server/src/resources.rs`

Depends on: Task 3, Task 6.

Per spec §2.2, Metal uses sysinfo's unified-memory totals. Per-process GPU attribution is **not available on Metal** so both `vram_used_by_mold` and `vram_used_by_other` are `None`.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/resources_test.rs`:

```rust
#[test]
#[cfg(target_os = "macos")]
fn metal_snapshot_reports_unified_memory_with_none_attribution() {
    let gpus = crate::resources::metal_snapshot();
    assert_eq!(gpus.len(), 1, "Metal hosts expose a single unified-memory GPU");
    let gpu = &gpus[0];
    assert_eq!(gpu.backend, mold_core::GpuBackend::Metal);
    assert_eq!(gpu.ordinal, 0);
    assert!(gpu.vram_total > 0);
    assert!(
        gpu.vram_used_by_mold.is_none(),
        "Metal does not expose per-process GPU attribution"
    );
    assert!(gpu.vram_used_by_other.is_none());
}

#[test]
#[cfg(not(target_os = "macos"))]
fn metal_snapshot_is_empty_off_darwin() {
    let gpus = crate::resources::metal_snapshot();
    assert!(gpus.is_empty());
}
```

- [ ] **Step 2: Run — expect compile fail**

```bash
cargo test -p mold-ai-server metal_snapshot --lib
```

Expected: `cannot find function metal_snapshot`.

- [ ] **Step 3: Implement**

Append to `crates/mold-server/src/resources.rs`:

```rust
/// Metal unified-memory snapshot — macOS only. Off-Darwin returns an empty
/// Vec so callers on Linux/CUDA hosts can unconditionally call this.
///
/// Unified memory means there's no distinct VRAM total; we report the
/// system RAM total so the SPA's GPU row still communicates "this is how
/// much the GPU can address." Per-process attribution is unavailable on
/// macOS (IOKit doesn't expose it in userspace), so both per-process fields
/// are `None` and the SPA hides those rows.
pub fn metal_snapshot() -> Vec<GpuSnapshot> {
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing()
                .with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        sys.refresh_memory();
        let total = sys.total_memory();
        let used = sys.used_memory();
        vec![GpuSnapshot {
            ordinal: 0,
            name: "Apple Metal GPU".to_string(),
            backend: GpuBackend::Metal,
            vram_total: total,
            vram_used: used,
            vram_used_by_mold: None,
            vram_used_by_other: None,
        }]
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}
```

- [ ] **Step 4: Run — expect pass**

On Linux:

```bash
cargo test -p mold-ai-server metal_snapshot --lib
```

On macOS:

```bash
cargo test -p mold-ai-server metal_snapshot --lib
```

Both: 1 test passes (the other is excluded by cfg).

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): Metal unified-memory snapshot path (macOS only)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Unified snapshot builder + 1 Hz aggregator task

**Files:**
- Modify: `crates/mold-server/src/resources.rs`
- Modify: `crates/mold-server/src/lib.rs`

Depends on: Tasks 5a/5b, 6, 7.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/resources_test.rs`:

```rust
#[test]
fn build_snapshot_populates_hostname_and_timestamp() {
    let snap = crate::resources::build_snapshot();
    assert!(!snap.hostname.is_empty(), "hostname must be populated");
    assert!(snap.timestamp > 0, "timestamp must be non-zero");
    // On any host, either gpus is non-empty (CUDA/Metal) or it's empty
    // (CPU-only). Both are valid — we just require the call doesn't panic.
    assert!(snap.system_ram.total > 0);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn aggregator_publishes_within_first_tick() {
    let bcast = crate::resources::ResourceBroadcaster::new();
    let mut rx = bcast.subscribe();
    let handle = crate::resources::spawn_aggregator(bcast.clone());

    // Advance virtual time past one tick interval (1 s).
    tokio::time::advance(std::time::Duration::from_millis(1_100)).await;

    // The aggregator fires immediately on startup, so there should be a
    // snapshot waiting even before the 1-second tick.
    let got = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
        .await
        .expect("aggregator should publish within first tick")
        .expect("channel should not be closed");
    assert!(got.timestamp > 0);

    handle.abort();
}
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p mold-ai-server build_snapshot_populates_hostname_and_timestamp aggregator_publishes_within_first_tick --lib
```

Expected: compile error.

- [ ] **Step 3: Implement `build_snapshot` + `spawn_aggregator`**

Append to `crates/mold-server/src/resources.rs`:

```rust
use mold_core::ResourceSnapshot;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::task::JoinHandle;

/// Assemble a single `ResourceSnapshot` from whichever data sources are
/// available on the current host. Cheap enough to run at 1 Hz (~200 µs).
///
/// Source priority on CUDA: NVML (if linked) → `nvidia-smi` subprocess → empty.
/// On macOS: `metal_snapshot()`.
pub fn build_snapshot() -> ResourceSnapshot {
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let gpus = collect_gpus();
    let system_ram = ram_snapshot();

    ResourceSnapshot {
        hostname,
        timestamp,
        gpus,
        system_ram,
    }
}

fn collect_gpus() -> Vec<GpuSnapshot> {
    // Darwin: Metal is the only GPU path.
    #[cfg(target_os = "macos")]
    {
        return metal_snapshot();
    }
    // Linux / other: try NVML first, fall back to nvidia-smi.
    #[cfg(all(not(target_os = "macos"), feature = "nvml"))]
    {
        if let Ok(src) = NvmlSource::try_new() {
            let gpus = src.snapshot(std::process::id());
            if !gpus.is_empty() {
                return gpus;
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        SmiSource::snapshot()
    }
}

/// Spawn the 1 Hz aggregator task. Returns the `JoinHandle` so `run_server`
/// can drop it on shutdown. The task fires once immediately on startup so
/// `GET /api/resources` succeeds without waiting a full second.
pub fn spawn_aggregator(bcast: Arc<ResourceBroadcaster>) -> JoinHandle<()> {
    tokio::spawn(async move {
        // Immediate first tick so `latest()` is populated before any HTTP
        // request arrives.
        bcast.publish(build_snapshot());
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Consume the first tick (it fires immediately) so we don't double-emit.
        interval.tick().await;
        loop {
            interval.tick().await;
            let snap = tokio::task::spawn_blocking(build_snapshot)
                .await
                .unwrap_or_else(|_| ResourceSnapshot {
                    hostname: "unknown".to_string(),
                    timestamp: 0,
                    gpus: Vec::new(),
                    system_ram: mold_core::RamSnapshot {
                        total: 0,
                        used: 0,
                        used_by_mold: 0,
                        used_by_other: 0,
                    },
                });
            bcast.publish(snap);
        }
    })
}
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server build_snapshot_populates_hostname_and_timestamp aggregator_publishes_within_first_tick --lib
```

Expected: 2 tests pass.

- [ ] **Step 5: Wire the aggregator into `run_server`**

In `crates/mold-server/src/lib.rs`, find this line (near the bottom of `run_server`):

```rust
    // Save start_time before state is moved into the router (needed for metrics).
```

Immediately before it, add:

```rust
    // Spawn the resource telemetry aggregator (1 Hz).
    let _resources_aggregator = resources::spawn_aggregator(state.resources.clone());
```

- [ ] **Step 6: Verify server still starts**

```bash
cargo check -p mold-ai-server
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/resources.rs crates/mold-server/src/resources_test.rs crates/mold-server/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(resources): unified build_snapshot + 1 Hz aggregator task wired into run_server

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: REST endpoint `GET /api/resources`

**Files:**
- Modify: `crates/mold-server/src/routes.rs`

Depends on: Tasks 4, 8.

- [ ] **Step 1: Write failing test**

In `crates/mold-server/src/routes_test.rs`, append inside the existing `#[cfg(test)] mod tests { ... }`:

```rust
    #[tokio::test]
    async fn get_api_resources_returns_snapshot() {
        let _lock = env_lock().lock().unwrap();
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        // Seed the broadcaster so the endpoint has something to return.
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "unit-test".into(),
            timestamp: 12345,
            gpus: vec![],
            system_ram: mold_core::RamSnapshot {
                total: 1,
                used: 0,
                used_by_mold: 0,
                used_by_other: 0,
            },
        });

        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = json_body(resp).await;
        assert_eq!(body["hostname"], "unit-test");
        assert_eq!(body["timestamp"], 12345);
        assert!(body["system_ram"].is_object());
    }

    #[tokio::test]
    async fn get_api_resources_returns_503_before_first_tick() {
        let _lock = env_lock().lock().unwrap();
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        // Do NOT publish — broadcaster has no cached snapshot.
        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p mold-ai-server get_api_resources --lib
```

Expected: 404 (route not registered yet).

- [ ] **Step 3: Implement handler + register route**

In `crates/mold-server/src/routes.rs`:

(a) At the top, extend the `use mold_core::{ ... }` import block to include `ResourceSnapshot`:

```rust
use mold_core::{
    ActiveGenerationStatus, GpuInfo, GpuWorkerState, ModelInfoExtended, ResourceSnapshot,
    ServerStatus, SseErrorEvent, SseProgressEvent,
};
```

(b) In `create_router`, add the route immediately before the `.route("/api/status", ...)` line:

```rust
        .route("/api/resources", get(get_resources))
        .route("/api/resources/stream", get(get_resources_stream))
```

(c) Append to the end of the file (before the `#[cfg(test)]` block), in a clearly labelled section:

```rust
// ── Resource telemetry (Agent B scope) ───────────────────────────────────────

/// `GET /api/resources` — one-shot JSON snapshot from the aggregator cache.
/// Returns 503 if the aggregator has not yet fired (first 1 s after startup
/// and before `spawn_aggregator` has run).
async fn get_resources(State(state): State<AppState>) -> Result<Json<ResourceSnapshot>, ApiError> {
    match state.resources.latest() {
        Some(snap) => Ok(Json(snap)),
        None => Err(ApiError::internal_with_status(
            "resource telemetry not ready",
            StatusCode::SERVICE_UNAVAILABLE,
        )),
    }
}
```

(d) Extend `ApiError` with a helper that lets us pick a status code:

In the same file, add a method to the `impl ApiError { ... }` block:

```rust
    pub fn internal_with_status(msg: impl Into<String>, status: StatusCode) -> Self {
        Self {
            error: msg.into(),
            code: "INTERNAL_ERROR".to_string(),
            status,
        }
    }
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server get_api_resources --lib
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): GET /api/resources REST endpoint

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: SSE endpoint `GET /api/resources/stream`

**Files:**
- Modify: `crates/mold-server/src/routes.rs`

Depends on: Task 9.

- [ ] **Step 1: Write failing test**

Append to `crates/mold-server/src/routes_test.rs`:

```rust
    #[tokio::test]
    async fn get_api_resources_stream_sets_sse_content_type() {
        let _lock = env_lock().lock().unwrap();
        let state = AppState::empty(
            mold_core::Config::default(),
            crate::state::QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        let app = create_router(state);
        let req = Request::builder()
            .uri("/api/resources/stream")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p mold-ai-server get_api_resources_stream_sets_sse_content_type --lib
```

Expected: 404.

- [ ] **Step 3: Implement**

Append to `crates/mold-server/src/routes.rs`, in the "Resource telemetry" section below `get_resources`:

```rust
/// `GET /api/resources/stream` — SSE stream of `ResourceSnapshot` frames.
/// Event name: `snapshot`. Matches the keepalive cadence of `/api/generate/stream`.
async fn get_resources_stream(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, Infallible>>> {
    use tokio_stream::wrappers::BroadcastStream;

    let rx = state.resources.subscribe();
    // Attach the cached `latest` snapshot as the first frame so clients
    // don't wait up to one full tick for their initial value.
    let initial = state.resources.latest();

    let stream = async_stream::stream! {
        if let Some(snap) = initial {
            yield Ok::<_, Infallible>(snapshot_to_sse(&snap));
        }
        let mut bs = BroadcastStream::new(rx);
        while let Some(item) = bs.next().await {
            match item {
                Ok(snap) => yield Ok(snapshot_to_sse(&snap)),
                // Lag is normal for slow clients — skip dropped frames
                // silently; the next one will catch them up.
                Err(_lagged) => continue,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

fn snapshot_to_sse(snap: &ResourceSnapshot) -> SseEvent {
    match serde_json::to_string(snap) {
        Ok(data) => SseEvent::default().event("snapshot").data(data),
        Err(e) => SseEvent::default()
            .event("error")
            .data(format!("{{\"message\":\"serialize failed: {e}\"}}")),
    }
}
```

(Note: `async-stream` is a tiny published crate; add to `[dependencies]` in `crates/mold-server/Cargo.toml`:)

```toml
async-stream = "0.3"
```

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p mold-ai-server get_api_resources_stream_sets_sse_content_type --lib
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mold-server/Cargo.toml crates/mold-server/src/routes.rs crates/mold-server/src/routes_test.rs
git commit -m "$(cat <<'EOF'
feat(resources): GET /api/resources/stream SSE endpoint with keepalive

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Web types mirror + REST helper

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`

Depends on: Task 9.

- [ ] **Step 1: Append wire types to `web/src/types.ts`**

Append at the very end:

```typescript
// ──────────────────────────────────────────────────────────────────────────────
// Resource telemetry (Agent B scope). Mirror of `mold_core::ResourceSnapshot`
// et al. `vram_used_by_mold` / `vram_used_by_other` are null on Metal hosts
// and on CUDA hosts that fell back to the `nvidia-smi` subprocess path.
// ──────────────────────────────────────────────────────────────────────────────

export type GpuBackend = "cuda" | "metal";

export interface GpuSnapshot {
  ordinal: number;
  name: string;
  backend: GpuBackend;
  vram_total: number;
  vram_used: number;
  vram_used_by_mold: number | null;
  vram_used_by_other: number | null;
}

export interface RamSnapshot {
  total: number;
  used: number;
  used_by_mold: number;
  used_by_other: number;
}

export interface ResourceSnapshot {
  hostname: string;
  timestamp: number;
  gpus: GpuSnapshot[];
  system_ram: RamSnapshot;
}
```

- [ ] **Step 2: Append `fetchResources` to `web/src/api.ts`**

Append (find an appropriate spot alongside other `fetch*` exports):

```typescript
import type { ResourceSnapshot } from "./types";

export async function fetchResources(
  signal?: AbortSignal,
): Promise<ResourceSnapshot> {
  const res = await fetch("/api/resources", { signal });
  if (!res.ok) throw new Error(`fetchResources failed: ${res.status}`);
  return (await res.json()) as ResourceSnapshot;
}
```

- [ ] **Step 3: Run format + type-check**

```bash
cd web && bun run fmt && bun run build
```

Expected: clean TypeScript build.

- [ ] **Step 4: Commit**

```bash
git add web/src/types.ts web/src/api.ts
git commit -m "$(cat <<'EOF'
feat(web): mirror ResourceSnapshot types and add fetchResources helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `useResources` composable with SSE + auto-reconnect

**Files:**
- Create: `web/src/composables/useResources.ts`
- Create: `web/src/composables/useResources.test.ts`
- Modify: `web/package.json` (add `@vue/test-utils`)

Depends on: Task 11.

- [ ] **Step 1: Add `@vue/test-utils` devDep**

```bash
cd web && bun add -d @vue/test-utils@^2.4.6
```

Commit this change separately at the end of Task 12 (it's needed again in Task 13).

- [ ] **Step 2: Write failing test**

Create `web/src/composables/useResources.test.ts`:

```typescript
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { useResources } from "./useResources";
import type { ResourceSnapshot } from "../types";

function snap(overrides: Partial<ResourceSnapshot> = {}): ResourceSnapshot {
  return {
    hostname: "unit",
    timestamp: 1,
    gpus: [],
    system_ram: { total: 100, used: 0, used_by_mold: 0, used_by_other: 0 },
    ...overrides,
  };
}

/**
 * Minimal EventSource stub. Vitest's happy-dom doesn't ship a functional
 * EventSource; we replace it with a class that lets each test drive
 * `message` / `error` / `open` events deterministically.
 */
class MockEventSource implements Partial<EventSource> {
  static instances: MockEventSource[] = [];
  url: string;
  listeners = new Map<string, ((e: MessageEvent) => void)[]>();
  onopen: ((e: Event) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }
  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    const arr = this.listeners.get(type) ?? [];
    arr.push(cb);
    this.listeners.set(type, arr);
  }
  removeEventListener() {}
  close() {
    this.closed = true;
  }
  fire(event: string, data: unknown) {
    const listeners = this.listeners.get(event) ?? [];
    const evt = new MessageEvent(event, { data: JSON.stringify(data) });
    for (const l of listeners) l(evt);
  }
  fireError() {
    if (this.onerror) this.onerror(new Event("error"));
  }
}

describe("useResources", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource);
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("starts with null snapshot, then updates on message", async () => {
    const r = useResources();
    expect(r.snapshot.value).toBeNull();

    const es = MockEventSource.instances[0];
    es.fire("snapshot", snap({ hostname: "alpha", timestamp: 42 }));
    await nextTick();

    expect(r.snapshot.value?.hostname).toBe("alpha");
    expect(r.snapshot.value?.timestamp).toBe(42);

    r.stop();
  });

  it("reconnects after an error with exponential backoff", async () => {
    const r = useResources();
    const first = MockEventSource.instances[0];
    first.fireError();

    // Backoff: first retry fires at 1000ms.
    vi.advanceTimersByTime(1000);
    await nextTick();

    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(2);
    expect(first.closed).toBe(true);

    r.stop();
  });

  it("replaces snapshot on each message (no append)", async () => {
    const r = useResources();
    const es = MockEventSource.instances[0];
    es.fire("snapshot", snap({ timestamp: 1 }));
    await nextTick();
    es.fire("snapshot", snap({ timestamp: 2 }));
    await nextTick();

    expect(r.snapshot.value?.timestamp).toBe(2);
    r.stop();
  });

  it("gpuList exposes the gpus array or [] when snapshot is null", async () => {
    const r = useResources();
    expect(r.gpuList.value).toEqual([]);

    const es = MockEventSource.instances[0];
    es.fire(
      "snapshot",
      snap({
        gpus: [
          {
            ordinal: 0,
            name: "RTX 3090",
            backend: "cuda",
            vram_total: 24,
            vram_used: 10,
            vram_used_by_mold: 8,
            vram_used_by_other: 2,
          },
        ],
      }),
    );
    await nextTick();

    expect(r.gpuList.value.length).toBe(1);
    expect(r.gpuList.value[0].ordinal).toBe(0);
    r.stop();
  });
});
```

- [ ] **Step 3: Run — expect fail**

```bash
cd web && bun run test useResources
```

Expected: `Cannot find module './useResources'`.

- [ ] **Step 4: Implement the composable**

Create `web/src/composables/useResources.ts`:

```typescript
import { computed, onBeforeUnmount, ref, type ComputedRef, type Ref } from "vue";
import type { GpuSnapshot, ResourceSnapshot } from "../types";

export interface UseResources {
  snapshot: Ref<ResourceSnapshot | null>;
  gpuList: ComputedRef<GpuSnapshot[]>;
  error: Ref<string | null>;
  /** Close the underlying EventSource and stop reconnect attempts. */
  stop: () => void;
}

/**
 * Singleton-style SSE consumer for `/api/resources/stream`.
 *
 * The design mirrors `useStatusPoll` but uses SSE instead of polling, with
 * exponential-backoff reconnect (capped at 30 s) because the aggregator
 * ticks at 1 Hz and dropping frames is fine — we only ever want the latest.
 *
 * Agent C's `PlacementPanel.vue` consumes `gpuList` to populate the device
 * selector. Keep that return shape stable.
 */
export function useResources(): UseResources {
  const snapshot = ref<ResourceSnapshot | null>(null);
  const error = ref<string | null>(null);

  let es: EventSource | null = null;
  let retryDelay = 1000;
  const MAX_RETRY = 30_000;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    try {
      es = new EventSource("/api/resources/stream");
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
      scheduleRetry();
      return;
    }

    es.addEventListener("snapshot", (evt: MessageEvent) => {
      try {
        snapshot.value = JSON.parse(evt.data) as ResourceSnapshot;
        error.value = null;
        retryDelay = 1000; // reset backoff on success
      } catch (e) {
        error.value = `parse failed: ${e instanceof Error ? e.message : String(e)}`;
      }
    });

    es.onerror = () => {
      error.value = "resource telemetry stream lost";
      if (es) {
        es.close();
        es = null;
      }
      scheduleRetry();
    };
  }

  function scheduleRetry() {
    if (stopped) return;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      connect();
      retryDelay = Math.min(retryDelay * 2, MAX_RETRY);
    }, retryDelay);
  }

  function stop() {
    stopped = true;
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (es) {
      es.close();
      es = null;
    }
  }

  // Kick off immediately (used at module scope via singleton wrapper).
  connect();

  onBeforeUnmount(stop);

  const gpuList = computed<GpuSnapshot[]>(() => snapshot.value?.gpus ?? []);

  return { snapshot, gpuList, error, stop };
}

// ── Singleton wrapper ────────────────────────────────────────────────────────
// App.vue mounts `useResources()` once via `provide()`; pages consume via
// `inject()` so every subscriber shares a single EventSource.
export const RESOURCES_INJECTION_KEY = Symbol("mold.resources");
```

- [ ] **Step 5: Run — expect pass**

```bash
cd web && bun run test useResources
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit (includes the @vue/test-utils devDep from Step 1)**

```bash
cd ..
git add web/package.json web/bun.lock web/src/composables/useResources.ts web/src/composables/useResources.test.ts
git commit -m "$(cat <<'EOF'
feat(web): add useResources SSE composable with exponential-backoff reconnect

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `ResourceStrip.vue` component

**Files:**
- Create: `web/src/components/ResourceStrip.vue`
- Create: `web/src/components/ResourceStrip.test.ts`

Depends on: Task 12.

- [ ] **Step 1: Write failing test**

Create `web/src/components/ResourceStrip.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { ref } from "vue";
import ResourceStrip from "./ResourceStrip.vue";
import type { ResourceSnapshot } from "../types";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";

function mountWith(snap: ResourceSnapshot | null, variant: "full" | "chip" = "full") {
  const snapshot = ref(snap);
  const gpuList = ref(snap?.gpus ?? []);
  return mount(ResourceStrip, {
    props: { variant },
    global: {
      provide: {
        [RESOURCES_INJECTION_KEY]: {
          snapshot,
          gpuList,
          error: ref(null),
          stop: () => {},
        },
      },
    },
  });
}

const cuda: ResourceSnapshot = {
  hostname: "hal",
  timestamp: 1,
  gpus: [
    {
      ordinal: 0,
      name: "RTX 3090",
      backend: "cuda",
      vram_total: 24_000_000_000,
      vram_used: 14_200_000_000,
      vram_used_by_mold: 10_100_000_000,
      vram_used_by_other: 4_100_000_000,
    },
  ],
  system_ram: {
    total: 64_000_000_000,
    used: 38_400_000_000,
    used_by_mold: 22_100_000_000,
    used_by_other: 16_300_000_000,
  },
};

const metal: ResourceSnapshot = {
  hostname: "mbp",
  timestamp: 1,
  gpus: [
    {
      ordinal: 0,
      name: "Apple Metal GPU",
      backend: "metal",
      vram_total: 64_000_000_000,
      vram_used: 38_000_000_000,
      vram_used_by_mold: null,
      vram_used_by_other: null,
    },
  ],
  system_ram: {
    total: 64_000_000_000,
    used: 38_000_000_000,
    used_by_mold: 12_000_000_000,
    used_by_other: 26_000_000_000,
  },
};

describe("ResourceStrip", () => {
  it("renders one GPU row and one RAM row on CUDA", () => {
    const wrapper = mountWith(cuda);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    expect(rows.length).toBe(2);
    // GPU row includes per-process breakdown.
    expect(rows[0].text()).toContain("RTX 3090");
    expect(rows[0].text()).toContain("mold");
    expect(rows[0].text()).toContain("other");
  });

  it("hides per-process breakdown on Metal (null attribution)", () => {
    const wrapper = mountWith(metal);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    expect(rows.length).toBe(2);
    // The GPU row must NOT mention mold/other — they're null.
    const gpuRow = rows[0].text();
    expect(gpuRow).not.toMatch(/mold/i);
  });

  it("hides GPU rows on CPU-only host", () => {
    const cpuOnly: ResourceSnapshot = {
      ...cuda,
      gpus: [],
    };
    const wrapper = mountWith(cpuOnly);
    const rows = wrapper.findAll('[data-test="resource-row"]');
    // Only the RAM row should render.
    expect(rows.length).toBe(1);
  });

  it("renders placeholder when snapshot is null", () => {
    const wrapper = mountWith(null);
    expect(wrapper.text()).toContain("…");
  });

  it("chip variant renders a compact single-line summary", () => {
    const wrapper = mountWith(cuda, "chip");
    expect(wrapper.find('[data-test="resource-chip"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="resource-row"]').exists()).toBe(false);
  });
});
```

- [ ] **Step 2: Run — expect fail**

```bash
cd web && bun run test ResourceStrip
```

Expected: `Cannot find component ResourceStrip`.

- [ ] **Step 3: Implement**

Create `web/src/components/ResourceStrip.vue`:

```vue
<script setup lang="ts">
/**
 * Always-visible VRAM + system-RAM telemetry panel.
 *
 * Modes:
 *  - `variant="full"` (default) — docked at the bottom of the Composer column
 *    on /generate. One row per GPU plus one for system RAM, click-to-expand
 *    side sheet.
 *  - `variant="chip"` — compact single-line summary for the TopBar on
 *    narrow viewports (< lg).
 *
 * Data comes from the `useResources` singleton provided by App.vue.
 */
import { computed, inject, ref } from "vue";
import { RESOURCES_INJECTION_KEY } from "../composables/useResources";
import type { GpuSnapshot, ResourceSnapshot, RamSnapshot } from "../types";
import type { ComputedRef, Ref } from "vue";

type UseResourcesShape = {
  snapshot: Ref<ResourceSnapshot | null>;
  gpuList: ComputedRef<GpuSnapshot[]>;
  error: Ref<string | null>;
};

const props = withDefaults(
  defineProps<{
    variant?: "full" | "chip";
  }>(),
  { variant: "full" },
);

const injected = inject<UseResourcesShape | null>(RESOURCES_INJECTION_KEY, null);

const snapshot = computed<ResourceSnapshot | null>(
  () => injected?.snapshot.value ?? null,
);
const gpus = computed<GpuSnapshot[]>(() => injected?.gpuList.value ?? []);
const ram = computed<RamSnapshot | null>(
  () => snapshot.value?.system_ram ?? null,
);

const sheetOpen = ref(false);
function toggleSheet() {
  sheetOpen.value = !sheetOpen.value;
}

function fmtGb(bytes: number): string {
  return (bytes / 1_000_000_000).toFixed(1);
}

function pct(used: number, total: number): number {
  if (total === 0) return 0;
  return Math.min(100, Math.round((used / total) * 100));
}

const chipSummary = computed(() => {
  const s = snapshot.value;
  if (!s) return "…";
  const ramStr = ram.value
    ? `${fmtGb(ram.value.used)} / ${fmtGb(ram.value.total)} GB`
    : "…";
  if (s.gpus.length === 0) return `RAM ${ramStr}`;
  const primary = s.gpus[0];
  return `GPU ${fmtGb(primary.vram_used)} / ${fmtGb(primary.vram_total)} · RAM ${ramStr}`;
});
</script>

<template>
  <div v-if="variant === 'chip'" class="inline-flex">
    <button
      data-test="resource-chip"
      type="button"
      class="inline-flex h-9 items-center gap-2 rounded-full border border-white/5 bg-white/5 px-3 text-[13px] font-medium text-ink-200 transition hover:text-white"
      :title="snapshot?.hostname ?? ''"
      @click="toggleSheet"
    >
      <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path d="M12 2v4M12 18v4M2 12h4M18 12h4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8" />
      </svg>
      <span class="tabular-nums">{{ chipSummary }}</span>
    </button>
  </div>

  <aside
    v-else
    class="glass rounded-2xl px-4 py-3"
    role="region"
    aria-label="Resource telemetry"
  >
    <div v-if="!snapshot" class="text-[13px] text-ink-400">
      Loading resources…
    </div>

    <div v-else class="flex flex-col gap-2 text-[13px]">
      <div
        v-for="gpu in gpus"
        :key="gpu.ordinal"
        data-test="resource-row"
        class="flex items-center gap-3"
      >
        <div class="w-28 shrink-0 font-medium text-ink-100">
          GPU {{ gpu.ordinal }} · {{ gpu.name }}
        </div>
        <div class="w-32 shrink-0 tabular-nums text-ink-200">
          {{ fmtGb(gpu.vram_used) }} / {{ fmtGb(gpu.vram_total) }} GB
        </div>
        <div class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5">
          <div
            class="absolute inset-y-0 left-0 bg-brand-400/70"
            :style="{ width: `${pct(gpu.vram_used, gpu.vram_total)}%` }"
          />
        </div>
        <div
          v-if="gpu.vram_used_by_mold !== null && gpu.vram_used_by_other !== null"
          class="w-40 shrink-0 text-right text-[11px] text-ink-400 tabular-nums"
        >
          mold {{ fmtGb(gpu.vram_used_by_mold) }} · other
          {{ fmtGb(gpu.vram_used_by_other) }}
        </div>
      </div>

      <div
        v-if="ram"
        data-test="resource-row"
        class="flex items-center gap-3 border-t border-white/5 pt-2"
      >
        <div class="w-28 shrink-0 font-medium text-ink-100">RAM</div>
        <div class="w-32 shrink-0 tabular-nums text-ink-200">
          {{ fmtGb(ram.used) }} / {{ fmtGb(ram.total) }} GB
        </div>
        <div class="relative h-2 flex-1 overflow-hidden rounded-full bg-white/5">
          <div
            class="absolute inset-y-0 left-0 bg-emerald-400/70"
            :style="{ width: `${pct(ram.used, ram.total)}%` }"
          />
        </div>
        <div class="w-40 shrink-0 text-right text-[11px] text-ink-400 tabular-nums">
          mold {{ fmtGb(ram.used_by_mold) }} · other
          {{ fmtGb(ram.used_by_other) }}
        </div>
      </div>

      <div class="pt-1 text-[11px] text-ink-500">
        host {{ snapshot.hostname }} · updated
        {{ new Date(snapshot.timestamp).toLocaleTimeString() }}
      </div>
    </div>
  </aside>
</template>
```

- [ ] **Step 4: Run — expect pass**

```bash
cd web && bun run test ResourceStrip
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ..
git add web/src/components/ResourceStrip.vue web/src/components/ResourceStrip.test.ts
git commit -m "$(cat <<'EOF'
feat(web): ResourceStrip component with full + chip variants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Mount `useResources` singleton in `App.vue` + provide injection

**Files:**
- Modify: `web/src/App.vue`

Depends on: Task 12.

- [ ] **Step 1: Rewrite `App.vue`**

Replace the whole file (it is trivially small) with:

```vue
<script setup lang="ts">
// ── Agent B (resource telemetry) ────────────────────────────────────────────
// `useResources` is mounted once at the App root and injected into pages
// that need it (/generate, and Agent C's PlacementPanel). This ensures a
// single shared EventSource instead of one per page navigation.
import { provide } from "vue";
import {
  useResources,
  RESOURCES_INJECTION_KEY,
} from "./composables/useResources";

const resources = useResources();
provide(RESOURCES_INJECTION_KEY, resources);
</script>

<template>
  <router-view />
</template>
```

- [ ] **Step 2: Sanity-check build**

```bash
cd web && bun run fmt && bun run build
```

Expected: clean TypeScript build; no runtime errors because nothing consumes the injection yet (Task 15 adds the first consumer).

- [ ] **Step 3: Commit**

```bash
cd ..
git add web/src/App.vue
git commit -m "$(cat <<'EOF'
feat(web): mount useResources singleton in App.vue via provide/inject

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15 (N-2): Mount `ResourceStrip` in `GeneratePage.vue`

**Files:**
- Modify: `web/src/pages/GeneratePage.vue`

Depends on: Tasks 13, 14.

- [ ] **Step 1: Add import**

At the top of `<script setup lang="ts">`, add alongside the other component imports:

```typescript
import ResourceStrip from "../components/ResourceStrip.vue";
```

- [ ] **Step 2: Render at bottom of Composer column**

Locate the `<Composer ... />` element in the template. Immediately after it (so it docks below), inside the same column wrapper, add:

```html
<!-- Agent B: always-visible VRAM + RAM telemetry -->
<ResourceStrip class="mt-3 hidden lg:block" variant="full" />
```

- [ ] **Step 3: Verify build + existing tests**

```bash
cd web && bun run fmt && bun run test && bun run build
```

Expected: all tests green, build clean.

- [ ] **Step 4: Commit**

```bash
cd ..
git add web/src/pages/GeneratePage.vue
git commit -m "$(cat <<'EOF'
feat(web): render ResourceStrip at bottom of Composer column on /generate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16 (N-1): TopBar narrow-viewport chip

**Files:**
- Modify: `web/src/components/TopBar.vue`

Depends on: Task 13.

Per spec §2.5: below `lg`, the full strip collapses to a chip in the TopBar.

- [ ] **Step 1: Add import + chip**

In `<script setup lang="ts">` of `TopBar.vue`, add at the end of the import block:

```typescript
import ResourceStrip from "./ResourceStrip.vue";
import { useRoute } from "vue-router";

const route = useRoute();
```

In the template, find the closing tag of the main `<header>` element. Immediately before it, inside the header, add a new section (clearly commented) that only renders on `/generate` and below `lg`:

```html
<!-- Agent B: narrow-viewport resource chip. Renders only on /generate
     below `lg` so desktop uses the full ResourceStrip inside the page. -->
<div
  v-if="route.name === 'generate'"
  class="flex shrink-0 items-center lg:hidden"
>
  <ResourceStrip variant="chip" />
</div>
```

- [ ] **Step 2: Verify build**

```bash
cd web && bun run fmt && bun run test && bun run build
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd ..
git add web/src/components/TopBar.vue
git commit -m "$(cat <<'EOF'
feat(web): render ResourceStrip chip in TopBar for narrow viewports on /generate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task N: Full gate + PR into umbrella

**Files:** None (verification + git workflow).

Depends on: all prior tasks.

- [ ] **Step 1: Run the full per-agent gate from spec §4.3**

```bash
cargo test --workspace
```

Expected: all tests pass.

```bash
cargo clippy --workspace -- -D warnings
```

Expected: no warnings.

```bash
cargo fmt --check
```

Expected: clean. If dirty, run `cargo fmt --all`, commit, and re-run.

```bash
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
```

Expected: clean (multi-feature check).

```bash
cd web && bun run fmt:check && bun run test && bun run build && cd ..
```

Expected: clean.

- [ ] **Step 2: Manual smoke (can skip in auto mode; done at UAT time instead)**

Run the server locally and hit the new endpoints:

```bash
cargo run -p mold-ai --features cuda -- serve --port 7690 &
sleep 3
curl -s http://localhost:7690/api/resources | head -c 400
curl -Ns http://localhost:7690/api/resources/stream | head -c 600
kill %1
```

Expected:
- `GET /api/resources` returns a JSON object with `hostname`, `timestamp`, `gpus`, `system_ram`.
- `GET /api/resources/stream` streams `event: snapshot` frames every ~1 s.

- [ ] **Step 3: Push branch**

```bash
git push -u origin wt/resource-telemetry
```

- [ ] **Step 4: Open PR into `feat/model-ui-overhaul`**

```bash
gh pr create --base feat/model-ui-overhaul --head wt/resource-telemetry \
  --title "resource telemetry: /api/resources{,/stream} + ResourceStrip.vue" \
  --body "$(cat <<'EOF'
## Summary

- Add `mold-server/src/resources.rs`: 1 Hz aggregator, `ResourceSnapshot`/`GpuSnapshot`/`RamSnapshot`/`GpuBackend` types, NVML + `nvidia-smi` + sysinfo + Metal data sources.
- Add `GET /api/resources` (one-shot JSON) and `GET /api/resources/stream` (SSE, 15 s keepalive).
- Add `web/src/composables/useResources.ts` singleton with exponential-backoff reconnect, and `ResourceStrip.vue` (full + chip variants).
- Mount the strip at the bottom of Composer on `/generate` (≥lg) and a chip in TopBar (<lg).
- Exposes `gpuList: ComputedRef<GpuSnapshot[]>` that Agent C's `PlacementPanel.vue` consumes.

## Spec alignment

Implements Phase 2 (Resource Telemetry) of `docs/superpowers/specs/2026-04-19-model-ui-overhaul-design.md`. Per §4.2, no edits to downloads (§1) or device-placement (§3) code paths.

## Task 1 probe outcome

See `docs/superpowers/plans/2026-04-19-resource-telemetry.md` Task 1 outcome section — records whether `nvml-wrapper` survived the linker probe and which CUDA data source is primary.

## Test plan

- [x] `cargo test --workspace`
- [x] `cargo clippy --workspace -- -D warnings`
- [x] `cargo fmt --check`
- [x] `cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4`
- [x] `cd web && bun run fmt:check && bun run test && bun run build`
- [ ] Manual: `curl /api/resources` returns snapshot JSON
- [ ] Manual: `curl -N /api/resources/stream` streams `event: snapshot` frames
- [ ] UAT on killswitch (dual 3090) — `mold` number rises when a model loads on GPU 0; "other" rises when a competing Python process grabs memory on GPU 1.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Report**

Return the PR URL to the main agent. Do **not** merge — the main agent coordinates all three umbrella sub-PRs and merges in order.

---

## Dependency Graph

```
Task 1 (NVML probe) ────► Task 2 (core types) ────► Task 3 (broadcaster skeleton) ──► Task 4 (AppState)
                                                       │
                                                       ├──► Task 5a (NVML source) [probe passed]
                                                       ├──► Task 5b (nvidia-smi source) [always]
                                                       ├──► Task 6  (sysinfo RAM)
                                                       ├──► Task 7  (Metal path)
                                                       │
                                                       └──► Task 8  (build_snapshot + aggregator) ──► Task 9  (GET /api/resources)
                                                                                                        │
                                                                                                        └──► Task 10 (SSE stream) ──► Task 11 (web types) ──► Task 12 (useResources) ──► Task 13 (ResourceStrip)
                                                                                                                                                                │
                                                                                                                                         ┌──────────────────────┴──────────────────────┐
                                                                                                                                         │                                              │
                                                                                                                                  Task 14 (App.vue)                             Task 15 (GeneratePage)
                                                                                                                                         │                                              │
                                                                                                                                  Task 16 (TopBar chip)                                  │
                                                                                                                                         │                                              │
                                                                                                                                         └──────────────────────┬──────────────────────┘
                                                                                                                                                                ▼
                                                                                                                                                     Task N (gate + PR)
```

**Critical path:** Task 1 must complete before 5a vs 5b decision. All Rust work (2-10) must be complete before the web tasks (11-16) can reliably hit the endpoints locally. Task N assumes every prior task shipped green.
