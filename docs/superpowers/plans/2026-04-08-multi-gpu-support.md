# Multi-GPU Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable concurrent model placement across multiple GPUs with smart routing, configurable GPU selection, and bounded request queuing.

**Architecture:** Per-GPU worker pool where each GPU gets a dedicated OS thread, its own ModelCache, and independent model lifecycle. A GpuPool routes requests using idle-first + VRAM-fit placement strategy. Atomic in-flight counters prevent TOCTOU races under burst load.

**Tech Stack:** Rust, candle (CUDA/Metal), tokio (async dispatch), std::thread (GPU workers), axum (HTTP server), clap (CLI)

**Spec:** `docs/superpowers/specs/2026-04-08-multi-gpu-support-design.md`

---

## File Map

### New Files
- `crates/mold-server/src/gpu_pool.rs` — GpuPool, GpuWorker, placement strategy, GpuJob
- `crates/mold-server/src/gpu_worker.rs` — Dedicated OS thread worker loop, process_job(), take-and-restore pattern

### Modified Files
| File | What Changes |
|------|-------------|
| `crates/mold-core/src/types.rs` | GpuInfo expanded, GpuSelection, GpuStatus, GpuWorkerStatus types. ServerStatus gets `gpus` array + queue fields. GenerateResponse gets `gpu` field. |
| `crates/mold-core/src/config.rs` | Config gets `gpus` and `queue_size` fields |
| `crates/mold-inference/src/device.rs` | `create_device(ordinal)`, `discover_gpus()`, `free_vram_bytes(ordinal)`, `vram_used_estimate(ordinal)`, `reclaim_gpu_memory(ordinal)` |
| `crates/mold-inference/src/engine_base.rs` | `gpu_ordinal: usize` field |
| `crates/mold-inference/src/factory.rs` | `ordinal` param threaded through |
| `crates/mold-inference/src/expand.rs` | Ordinal-aware device creation |
| `crates/mold-server/src/state.rs` | AppState: replace model_cache/engine_snapshot/model_load_lock with gpu_pool |
| `crates/mold-server/src/model_cache.rs` | Minor: expose `remove()` + `insert()` for take-and-restore |
| `crates/mold-server/src/model_manager.rs` | Operate on GpuWorker instead of global state |
| `crates/mold-server/src/queue.rs` | Multi-GPU dispatcher with in-flight tracking |
| `crates/mold-server/src/routes.rs` | Status, load, unload endpoints updated for multi-GPU |
| `crates/mold-server/src/lib.rs` | Server startup creates GpuPool |
| `crates/mold-cli/src/main.rs` | `--gpus`, `--queue-size` flags |
| `crates/mold-cli/src/commands/generate.rs` | Best-GPU selection for `--local` |
| `crates/mold-cli/src/commands/ps.rs` | Multi-GPU status display |

---

## Task 1: Core Types & Config (mold-core)

**Files:**
- Modify: `crates/mold-core/src/types.rs`
- Modify: `crates/mold-core/src/config.rs`

No dependencies. Can start immediately.

- [ ] **Step 1: Add GpuSelection type to types.rs**

Add after the existing `GpuInfo` struct (~line 585):

```rust
/// GPU selection for multi-GPU setups
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuSelection {
    /// Use all discovered GPUs (default)
    All,
    /// Use only these specific GPU ordinals
    Specific(Vec<usize>),
}

impl Default for GpuSelection {
    fn default() -> Self {
        Self::All
    }
}

impl GpuSelection {
    /// Parse from comma-separated string like "0,1,2"
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        if s.is_empty() || s.to_lowercase() == "all" {
            return Ok(Self::All);
        }
        let ordinals: Vec<usize> = s
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("invalid GPU ordinal: {e}"))?;
        if ordinals.is_empty() {
            return Ok(Self::All);
        }
        Ok(Self::Specific(ordinals))
    }
}
```

- [ ] **Step 2: Add GpuWorkerStatus type to types.rs**

Add after GpuSelection:

```rust
/// Per-GPU worker status for multi-GPU status reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuWorkerStatus {
    pub ordinal: usize,
    pub name: String,
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,
    pub loaded_model: Option<String>,
    pub state: GpuWorkerState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GpuWorkerState {
    Idle,
    Generating,
    Loading,
    Degraded,
}
```

- [ ] **Step 3: Update ServerStatus in types.rs**

Find the existing `ServerStatus` struct (~line 556-573) and add the new fields. Keep existing fields for backwards compat:

```rust
pub struct ServerStatus {
    pub version: String,
    pub models_loaded: Vec<String>,
    pub model: Option<String>,       // backwards compat: first GPU's model
    pub busy: bool,
    pub current_generation: Option<ActiveGenerationStatus>,
    pub gpu_info: Option<GpuInfo>,   // backwards compat: first GPU
    pub uptime_secs: Option<u64>,
    // New multi-GPU fields
    pub gpus: Option<Vec<GpuWorkerStatus>>,
    pub queue_depth: Option<usize>,
    pub queue_capacity: Option<usize>,
}
```

- [ ] **Step 4: Add `gpu` field to GenerateResponse**

Find GenerateResponse (~line 287-300) and add:

```rust
pub struct GenerateResponse {
    // ... existing fields ...
    /// Which GPU ordinal handled this request (multi-GPU only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<usize>,
}
```

- [ ] **Step 5: Update Config struct**

In `crates/mold-core/src/config.rs`, find the `Config` struct (~line 312-372) and add:

```rust
pub struct Config {
    // ... existing fields ...
    /// GPU ordinals to use (None = all available)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<usize>>,
    /// Max queued requests before 503 (default: 200)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_size: Option<usize>,
}
```

- [ ] **Step 6: Add GpuSelection helper to Config**

Add a method to Config:

```rust
impl Config {
    pub fn gpu_selection(&self) -> GpuSelection {
        match &self.gpus {
            Some(ordinals) if !ordinals.is_empty() => GpuSelection::Specific(ordinals.clone()),
            _ => GpuSelection::All,
        }
    }

    pub fn queue_size(&self) -> usize {
        self.queue_size.unwrap_or(200)
    }
}
```

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p mold-ai-core`
Expected: PASS (no other crates modified yet)

- [ ] **Step 8: Commit**

```bash
git add crates/mold-core/src/types.rs crates/mold-core/src/config.rs
git commit -m "feat(core): add multi-GPU types — GpuSelection, GpuWorkerStatus, ServerStatus gpus array"
```

---

## Task 2: GPU Discovery & Device Ordinal (mold-inference)

**Files:**
- Modify: `crates/mold-inference/src/device.rs`

Depends on: Task 1 (uses GpuInfo from mold-core)

- [ ] **Step 1: Add GpuInfo struct and discover_gpus()**

Add at the top of device.rs (after imports):

```rust
use mold_core::types::GpuSelection;

/// Discovered GPU information
#[derive(Debug, Clone)]
pub struct DiscoveredGpu {
    pub ordinal: usize,
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

/// Discover all available GPUs on the system
pub fn discover_gpus() -> Vec<DiscoveredGpu> {
    let mut gpus = Vec::new();

    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            if let Ok(count) = cudarc::driver::result::device::get_count() {
                for ordinal in 0..count as usize {
                    if let Ok(device) = cudarc::driver::CudaDevice::new(ordinal) {
                        let name = device
                            .name()
                            .unwrap_or_else(|_| format!("CUDA Device {ordinal}"));
                        let (free, total) = cudarc::driver::result::mem_get_info()
                            .unwrap_or((0, 0));
                        gpus.push(DiscoveredGpu {
                            ordinal,
                            name,
                            total_vram_bytes: total as u64,
                            free_vram_bytes: free as u64,
                        });
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if candle_core::utils::metal_is_available() {
            // Metal: single device on macOS
            let total = available_system_memory_bytes().unwrap_or(0);
            let free = free_system_memory_bytes().unwrap_or(0);
            gpus.push(DiscoveredGpu {
                ordinal: 0,
                name: "Apple Metal GPU".to_string(),
                total_vram_bytes: total,
                free_vram_bytes: free,
            });
        }
    }

    gpus
}

/// Filter discovered GPUs by user selection
pub fn filter_gpus(gpus: &[DiscoveredGpu], selection: &GpuSelection) -> Vec<DiscoveredGpu> {
    match selection {
        GpuSelection::All => gpus.to_vec(),
        GpuSelection::Specific(ordinals) => gpus
            .iter()
            .filter(|g| ordinals.contains(&g.ordinal))
            .cloned()
            .collect(),
    }
}

/// Select the single best GPU (most free VRAM) for local CLI use
pub fn select_best_gpu(gpus: &[DiscoveredGpu]) -> Option<&DiscoveredGpu> {
    gpus.iter().max_by_key(|g| g.free_vram_bytes)
}
```

- [ ] **Step 2: Update create_device() to take ordinal**

Change the existing `create_device()` signature (~line 6-28):

```rust
/// Create a device on the specified GPU ordinal.
/// Use ordinal 0 for single-GPU setups.
pub fn create_device(ordinal: usize, progress: &ProgressReporter) -> anyhow::Result<candle_core::Device> {
    if std::env::var("MOLD_DEVICE")
        .map(|v| v.to_lowercase() == "cpu")
        .unwrap_or(false)
    {
        progress.report(ProgressEvent::Status {
            message: "Using CPU device (MOLD_DEVICE=cpu)".to_string(),
        });
        return Ok(candle_core::Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        progress.report(ProgressEvent::Status {
            message: format!("Using CUDA device {ordinal}"),
        });
        return Ok(candle_core::Device::new_cuda(ordinal)?);
    }

    if candle_core::utils::metal_is_available() {
        progress.report(ProgressEvent::Status {
            message: format!("Using Metal device {ordinal}"),
        });
        return Ok(candle_core::Device::new_metal(ordinal)?);
    }

    progress.report(ProgressEvent::Status {
        message: "No GPU detected, using CPU".to_string(),
    });
    Ok(candle_core::Device::Cpu)
}
```

- [ ] **Step 3: Update free_vram_bytes() to take ordinal**

Find `free_vram_bytes()` (~line 204-235) and update:

```rust
pub fn free_vram_bytes(ordinal: usize) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        // Set context to the specified device before querying
        if let Ok(device) = cudarc::driver::CudaDevice::new(ordinal) {
            let (free, _total) = cudarc::driver::result::mem_get_info().ok()?;
            return Some(free as u64);
        }
        return None;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        None
    }
}
```

- [ ] **Step 4: Update vram_used_estimate() to take ordinal**

Find `vram_used_estimate()` (~line 224-233) and update:

```rust
pub fn vram_used_estimate(ordinal: usize) -> u64 {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = cudarc::driver::CudaDevice::new(ordinal) {
            if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
                return (total - free) as u64;
            }
        }
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        0
    }
}
```

- [ ] **Step 5: Update reclaim_gpu_memory() to take ordinal**

Find `reclaim_gpu_memory()` (~line 168-192) and update:

```rust
#[cfg(feature = "cuda")]
pub fn reclaim_gpu_memory(ordinal: usize) {
    // Reset only the specified device's primary context.
    // SAFETY: Caller must hold the per-worker model_load_lock,
    // guaranteeing exclusive access to this GPU during model swaps.
    use cudarc::driver::sys;
    let cu_device = ordinal as i32;
    if let Err(e) = unsafe { sys::lib().cuDevicePrimaryCtxReset_v2(cu_device) }.result() {
        tracing::warn!("Failed to reset CUDA context for device {ordinal}: {e:?}");
    }
}

#[cfg(not(feature = "cuda"))]
pub fn reclaim_gpu_memory(_ordinal: usize) {}
```

- [ ] **Step 6: Update all callers of these functions**

Search the codebase for calls to `create_device(`, `free_vram_bytes()`, `vram_used_estimate()`, `reclaim_gpu_memory()` and update them to pass ordinal. Most callers will get ordinal from `self.base.gpu_ordinal` (added in Task 3). For now, callers that don't yet have an ordinal should pass `0` as a temporary default — Task 3 will thread the real ordinal through.

Callers to update (pass `0` temporarily):
- Every engine pipeline's `load()` method (flux/pipeline.rs, sd15/pipeline.rs, sdxl/pipeline.rs, sd3/pipeline.rs, zimage/pipeline.rs, flux2/pipeline.rs, qwen_image/pipeline.rs, wuerstchen/pipeline.rs, ltx_video/pipeline.rs)
- `expand.rs` device creation (~line 118)
- `upscaler/engine.rs` device creation
- `model_manager.rs` calls to `reclaim_gpu_memory()` and `vram_used_estimate()`

- [ ] **Step 7: Verify compilation**

Run: `cargo check --workspace`
Expected: PASS (all callers updated to pass ordinal 0 as default)

- [ ] **Step 8: Commit**

```bash
git add crates/mold-inference/src/device.rs
git add -u  # all callers updated
git commit -m "feat(inference): add GPU discovery, ordinal-aware device creation and VRAM functions"
```

---

## Task 3: Engine Ordinal Threading (mold-inference)

**Files:**
- Modify: `crates/mold-inference/src/engine_base.rs`
- Modify: `crates/mold-inference/src/factory.rs`
- Modify: `crates/mold-inference/src/expand.rs`
- Modify: All engine pipeline.rs files (update `create_device` calls)

Depends on: Task 2

- [ ] **Step 1: Add gpu_ordinal to EngineBase**

In `engine_base.rs` (~line 18-24), add the field:

```rust
pub struct EngineBase<L> {
    pub loaded: Option<L>,
    pub model_name: String,
    pub paths: ModelPaths,
    pub progress: ProgressReporter,
    pub load_strategy: LoadStrategy,
    pub gpu_ordinal: usize,  // NEW
}
```

Update `new()` (~line 28-36) to accept and store it:

```rust
pub fn new(model_name: String, paths: ModelPaths, load_strategy: LoadStrategy, gpu_ordinal: usize) -> Self {
    Self {
        loaded: None,
        model_name,
        paths,
        progress: ProgressReporter::new(),
        load_strategy,
        gpu_ordinal,
    }
}
```

- [ ] **Step 2: Update factory to accept and pass ordinal**

In `factory.rs`, update both function signatures:

```rust
pub fn create_engine(
    model_name: &str,
    config: &Config,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,  // NEW
) -> anyhow::Result<Box<dyn InferenceEngine>> {
    create_engine_with_pool(model_name, config, load_strategy, gpu_ordinal, None, false, None, None, None, None)
}

pub fn create_engine_with_pool(
    model_name: &str,
    config: &Config,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,  // NEW
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
    // ... rest of existing params
) -> anyhow::Result<Box<dyn InferenceEngine>> {
```

Thread `gpu_ordinal` through to each engine constructor. Each engine's `new()` must pass it to `EngineBase::new()`.

- [ ] **Step 3: Update each engine's load() to use self.base.gpu_ordinal**

In every pipeline's `load()` method, change:

```rust
// Before (temporary ordinal 0 from Task 2)
let device = crate::device::create_device(0, &self.base.progress)?;

// After
let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
```

Similarly update VRAM queries:

```rust
// Before
let free_vram = crate::device::free_vram_bytes(0);

// After
let free_vram = crate::device::free_vram_bytes(self.base.gpu_ordinal);
```

Files to update:
- `flux/pipeline.rs`
- `sd15/pipeline.rs`
- `sdxl/pipeline.rs`
- `sd3/pipeline.rs`
- `zimage/pipeline.rs`
- `flux2/pipeline.rs`
- `qwen_image/pipeline.rs`
- `wuerstchen/pipeline.rs`
- `ltx_video/pipeline.rs`
- `upscaler/engine.rs`

- [ ] **Step 4: Update expand.rs**

In `expand.rs` (~line 118), the `LocalExpander` needs an ordinal. Add `gpu_ordinal: usize` field and use it:

```rust
let device = crate::device::create_device(self.gpu_ordinal, &progress)?;
```

- [ ] **Step 5: Update all factory callers**

Search for `create_engine(` and `create_engine_with_pool(` across the workspace. Update callers to pass `gpu_ordinal`. Callers that don't yet have a real ordinal (server code — updated in Task 4+) should pass `0` temporarily.

Key callers:
- `crates/mold-server/src/lib.rs` (~line 78-85)
- `crates/mold-server/src/model_manager.rs` (~line 354-444)
- `crates/mold-cli/src/commands/generate.rs` (local mode)

- [ ] **Step 6: Verify compilation**

Run: `cargo check --workspace`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "feat(inference): thread gpu_ordinal through EngineBase, factory, and all engine pipelines"
```

---

## Task 4: GpuPool & GpuWorker (mold-server)

**Files:**
- Create: `crates/mold-server/src/gpu_pool.rs`
- Create: `crates/mold-server/src/gpu_worker.rs`
- Modify: `crates/mold-server/src/model_cache.rs`

Depends on: Task 1, Task 2

- [ ] **Step 1: Add remove() to ModelCache for take-and-restore**

In `model_cache.rs`, add a `remove()` method that takes an engine out of the cache without dropping it:

```rust
/// Remove an engine from the cache, returning it.
/// Used by take-and-restore pattern: remove before inference, re-insert after.
pub fn take(&mut self, model_name: &str) -> Option<CachedEngine> {
    if let Some(entry) = self.entries.remove(model_name) {
        self.lru_order.retain(|n| n != model_name);
        Some(entry)
    } else {
        None
    }
}

/// Re-insert a taken engine after inference completes.
pub fn restore(&mut self, cached: CachedEngine) {
    let name = cached.model_name.clone();
    self.lru_order.push(name.clone());
    self.entries.insert(name, cached);
}
```

- [ ] **Step 2: Create gpu_pool.rs with GpuWorker and GpuPool**

```rust
use crate::model_cache::{ModelCache, ModelResidency};
use mold_core::types::{GpuWorkerState, GpuWorkerStatus};
use mold_inference::device::DiscoveredGpu;
use mold_inference::shared_pool::SharedPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

pub struct GpuWorker {
    pub gpu: DiscoveredGpu,
    pub model_cache: Arc<Mutex<ModelCache>>,
    pub active_generation: Arc<RwLock<Option<ActiveGeneration>>>,
    pub model_load_lock: Arc<Mutex<()>>,
    pub shared_pool: Arc<Mutex<SharedPool>>,
    pub in_flight: AtomicUsize,
    pub consecutive_failures: AtomicUsize,
    pub degraded_until: RwLock<Option<Instant>>,
    pub job_tx: std::sync::mpsc::SyncSender<GpuJob>,
}

#[derive(Debug)]
pub struct ActiveGeneration {
    pub model: String,
    pub started_at: Instant,
}

pub struct GpuJob {
    pub model: String,
    pub request: mold_core::GenerateRequest,
    pub progress_tx: Option<tokio::sync::mpsc::Sender<mold_inference::ProgressEvent>>,
    pub result_tx: tokio::sync::oneshot::Sender<anyhow::Result<mold_core::GenerateResponse>>,
    pub output_dir: Option<std::path::PathBuf>,
    pub config: Arc<tokio::sync::RwLock<mold_core::Config>>,
}

pub struct GpuPool {
    pub workers: Vec<Arc<GpuWorker>>,
}

impl GpuWorker {
    pub fn is_degraded(&self) -> bool {
        if self.consecutive_failures.load(Ordering::SeqCst) < 3 {
            return false;
        }
        match *self.degraded_until.read().unwrap() {
            Some(until) => Instant::now() < until,
            None => false,
        }
    }

    pub fn status(&self) -> GpuWorkerStatus {
        let cache = self.model_cache.lock().unwrap();
        let loaded_model = cache.active_model().map(|s| s.to_string());
        let active_gen = self.active_generation.read().unwrap();

        let state = if self.is_degraded() {
            GpuWorkerState::Degraded
        } else if active_gen.is_some() {
            GpuWorkerState::Generating
        } else {
            GpuWorkerState::Idle
        };

        GpuWorkerStatus {
            ordinal: self.gpu.ordinal,
            name: self.gpu.name.clone(),
            vram_total_bytes: self.gpu.total_vram_bytes,
            vram_used_bytes: mold_inference::device::vram_used_estimate(self.gpu.ordinal),
            loaded_model,
            state,
        }
    }
}

impl GpuPool {
    /// Find the worker that already has this model loaded on GPU
    pub fn find_loaded(&self, model_name: &str) -> Option<Arc<GpuWorker>> {
        let mut candidates: Vec<_> = self.workers.iter()
            .filter(|w| {
                let cache = w.model_cache.lock().unwrap();
                cache.get(model_name)
                    .map(|e| e.residency == ModelResidency::Gpu)
                    .unwrap_or(false)
            })
            .collect();

        // Prefer least in-flight if multiple have it
        candidates.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
        candidates.into_iter().next().cloned()
    }

    /// Select the best worker for a model that needs loading
    pub fn select_worker(&self, model_name: &str, estimated_vram: u64) -> Option<Arc<GpuWorker>> {
        // 1. Already loaded on a GPU?
        if let Some(w) = self.find_loaded(model_name) {
            return Some(w);
        }

        // 2. Find idle (no GPU-resident model) workers, skip degraded
        let mut idle: Vec<_> = self.workers.iter()
            .filter(|w| {
                if w.is_degraded() { return false; }
                let cache = w.model_cache.lock().unwrap();
                cache.active_model().is_none()
            })
            .collect();

        if !idle.is_empty() {
            // VRAM-fit tiebreaker: smallest GPU that fits
            idle.sort_by_key(|w| w.gpu.total_vram_bytes);
            if let Some(w) = idle.iter().find(|w| w.gpu.total_vram_bytes >= estimated_vram) {
                return Some((*w).clone());
            }
            // No idle GPU fits — pick the largest idle GPU anyway (eviction will help)
            return idle.last().cloned();
        }

        // 3. All GPUs busy — evict LRU on the GPU with most headroom
        let mut busy: Vec<_> = self.workers.iter()
            .filter(|w| !w.is_degraded())
            .collect();
        busy.sort_by(|a, b| {
            let a_headroom = a.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            let b_headroom = b.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            b_headroom.cmp(&a_headroom) // most headroom first
        });
        busy.into_iter().next().cloned()
    }

    pub fn gpu_status(&self) -> Vec<GpuWorkerStatus> {
        self.workers.iter().map(|w| w.status()).collect()
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}
```

- [ ] **Step 3: Create gpu_worker.rs with dedicated OS thread worker loop**

```rust
use crate::gpu_pool::{ActiveGeneration, GpuJob, GpuWorker};
use crate::model_cache::ModelResidency;
use mold_inference::device;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Spawn the dedicated OS thread for a GPU worker.
/// Returns the JoinHandle (caller should keep it alive).
pub fn spawn_gpu_thread(
    worker: Arc<GpuWorker>,
    job_rx: std::sync::mpsc::Receiver<GpuJob>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(format!("gpu-worker-{}", worker.gpu.ordinal))
        .spawn(move || {
            tracing::info!(gpu = worker.gpu.ordinal, name = %worker.gpu.name, "GPU worker thread started");
            for job in job_rx.iter() {
                process_job(&worker, job);
            }
            tracing::info!(gpu = worker.gpu.ordinal, "GPU worker thread exiting");
        })
        .expect("failed to spawn GPU worker thread")
}

fn process_job(worker: &GpuWorker, job: GpuJob) {
    let model_name = job.model.clone();
    let ordinal = worker.gpu.ordinal;

    // Acquire per-GPU load lock
    let _load_lock = worker.model_load_lock.lock().unwrap();

    // Ensure model is loaded on this GPU
    if let Err(e) = ensure_model_ready_sync(worker, &model_name, &job) {
        tracing::error!(gpu = ordinal, model = %model_name, "Failed to load model: {e}");
        let _ = job.result_tx.send(Err(e));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        record_failure(worker);
        return;
    }

    // Set active generation
    {
        let mut gen = worker.active_generation.write().unwrap();
        *gen = Some(ActiveGeneration {
            model: model_name.clone(),
            started_at: Instant::now(),
        });
    }

    // Take-and-restore: remove engine from cache, release lock during inference
    let taken = {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.take(&model_name)
    };

    let Some(mut cached_engine) = taken else {
        let _ = job.result_tx.send(Err(anyhow::anyhow!("Engine not found in cache after load")));
        worker.in_flight.fetch_sub(1, Ordering::SeqCst);
        clear_active_generation(worker);
        return;
    };

    // Set progress callback if SSE streaming
    if let Some(ref progress_tx) = job.progress_tx {
        let tx = progress_tx.clone();
        cached_engine.engine.set_on_progress(Arc::new(move |event| {
            let _ = tx.blocking_send(event);
        }));
    }

    // Run inference — cache mutex is FREE during this
    let result = cached_engine.engine.generate(&job.request);

    // Clear progress callback
    cached_engine.engine.clear_on_progress();

    // Restore engine to cache
    {
        let mut cache = worker.model_cache.lock().unwrap();
        cache.restore(cached_engine);
    }

    // Clear active generation
    clear_active_generation(worker);

    // Update health tracking
    match &result {
        Ok(_) => {
            worker.consecutive_failures.store(0, Ordering::SeqCst);
        }
        Err(e) => {
            tracing::warn!(gpu = ordinal, model = %model_name, "Generation failed: {e}");
            record_failure(worker);
        }
    }

    // Decrement in-flight and send result
    worker.in_flight.fetch_sub(1, Ordering::SeqCst);

    // Attach GPU ordinal to response
    let result = result.map(|mut resp| {
        resp.gpu = Some(ordinal);
        resp
    });

    let _ = job.result_tx.send(result);
}

fn ensure_model_ready_sync(
    worker: &GpuWorker,
    model_name: &str,
    job: &GpuJob,
) -> anyhow::Result<()> {
    let mut cache = worker.model_cache.lock().unwrap();

    // Already loaded?
    if let Some(entry) = cache.get(model_name) {
        if entry.residency == ModelResidency::Gpu {
            return Ok(());
        }
    }

    // Need to load — unload active model first
    cache.unload_active();
    drop(cache);

    // Reclaim GPU memory
    device::reclaim_gpu_memory(worker.gpu.ordinal);

    // Create and load engine
    let config = job.config.blocking_read();
    let mut engine = mold_inference::create_engine_with_pool(
        model_name,
        &config,
        mold_inference::engine::LoadStrategy::default(),
        worker.gpu.ordinal,
        Some(worker.shared_pool.clone()),
        false, // offload
        None, None, None, None, // variant overrides
    )?;
    drop(config);

    engine.load()?;

    let vram = device::vram_used_estimate(worker.gpu.ordinal);

    let mut cache = worker.model_cache.lock().unwrap();
    cache.insert_loaded(model_name.to_string(), engine, vram);

    Ok(())
}

fn record_failure(worker: &GpuWorker) {
    let failures = worker.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
    if failures >= 3 {
        let mut degraded = worker.degraded_until.write().unwrap();
        *degraded = Some(Instant::now() + Duration::from_secs(60));
        tracing::warn!(
            gpu = worker.gpu.ordinal,
            "GPU marked degraded after {failures} consecutive failures (60s cooldown)"
        );
    }
}

fn clear_active_generation(worker: &GpuWorker) {
    let mut gen = worker.active_generation.write().unwrap();
    *gen = None;
}
```

- [ ] **Step 4: Register new modules**

In `crates/mold-server/src/lib.rs`, add:

```rust
pub mod gpu_pool;
pub mod gpu_worker;
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p mold-ai-server`
Expected: May have issues due to AppState not yet updated — that's Task 5. Verify at least the new modules parse correctly.

- [ ] **Step 6: Commit**

```bash
git add crates/mold-server/src/gpu_pool.rs crates/mold-server/src/gpu_worker.rs crates/mold-server/src/model_cache.rs crates/mold-server/src/lib.rs
git commit -m "feat(server): add GpuPool, GpuWorker, and GPU worker thread with take-and-restore pattern"
```

---

## Task 5: AppState Refactor & Server Startup (mold-server)

**Files:**
- Modify: `crates/mold-server/src/state.rs`
- Modify: `crates/mold-server/src/lib.rs`

Depends on: Task 4

- [ ] **Step 1: Refactor AppState**

In `state.rs`, replace the single-GPU fields with GpuPool. Keep fields that are genuinely global (config, pull_lock, etc.):

```rust
pub struct AppState {
    pub gpu_pool: Arc<GpuPool>,
    pub config: Arc<tokio::sync::RwLock<Config>>,
    pub queue_tx: tokio::sync::mpsc::Sender<QueuedRequest>,
    pub pull_lock: Arc<tokio::sync::Mutex<()>>,
    pub shared_pool: Arc<std::sync::Mutex<SharedPool>>,
    pub startup_time: Instant,
    pub queue_capacity: usize,
    // ... keep other global fields (shutdown_tx, etc.)
}
```

Remove: `model_cache`, `engine_snapshot`, `model_load_lock`, `active_generation`, `upscaler_cache` — these now live in GpuWorker.

- [ ] **Step 2: Update AppState constructors**

Replace `AppState::new()` and `AppState::empty()` with a unified constructor that builds the GpuPool:

```rust
impl AppState {
    pub fn new(
        config: Config,
        gpu_selection: &GpuSelection,
        queue_size: usize,
        shared_pool: Arc<std::sync::Mutex<SharedPool>>,
    ) -> anyhow::Result<(Self, Vec<std::thread::JoinHandle<()>>)> {
        let discovered = mold_inference::device::discover_gpus();
        let selected = mold_inference::device::filter_gpus(&discovered, gpu_selection);

        if selected.is_empty() && !discovered.is_empty() {
            anyhow::bail!("No GPUs matched selection {:?} (discovered: {:?})", gpu_selection, discovered);
        }

        let mut workers = Vec::new();
        let mut thread_handles = Vec::new();

        for gpu in &selected {
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel(queue_size);
            let worker = Arc::new(GpuWorker {
                gpu: gpu.clone(),
                model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
                active_generation: Arc::new(RwLock::new(None)),
                model_load_lock: Arc::new(Mutex::new(())),
                shared_pool: shared_pool.clone(),
                in_flight: AtomicUsize::new(0),
                consecutive_failures: AtomicUsize::new(0),
                degraded_until: RwLock::new(None),
                job_tx,
            });

            let handle = gpu_worker::spawn_gpu_thread(worker.clone(), job_rx);
            thread_handles.push(handle);
            workers.push(worker);
        }

        let gpu_pool = Arc::new(GpuPool { workers });

        // ... construct AppState with gpu_pool, return (state, thread_handles)
    }
}
```

- [ ] **Step 3: Update server startup in lib.rs**

In `run_server()` (~line 29-210), replace the existing engine creation + AppState construction with GpuPool initialization:

```rust
pub async fn run_server(config: Config, gpu_selection: GpuSelection, queue_size: usize) -> anyhow::Result<()> {
    let shared_pool = Arc::new(std::sync::Mutex::new(SharedPool::new()));

    let (state, _gpu_threads) = AppState::new(
        config,
        &gpu_selection,
        queue_size,
        shared_pool,
    )?;

    // Log discovered GPUs
    for status in state.gpu_pool.gpu_status() {
        tracing::info!(
            gpu = status.ordinal,
            name = %status.name,
            vram_mb = status.vram_total_bytes / 1_000_000,
            "GPU worker ready"
        );
    }

    // ... rest of server setup (router, middleware, bind, serve) stays the same
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p mold-ai-server`
Expected: Errors in routes.rs, queue.rs, model_manager.rs — they still reference old AppState fields. That's expected; Tasks 6-7 fix those.

- [ ] **Step 5: Commit**

```bash
git add crates/mold-server/src/state.rs crates/mold-server/src/lib.rs
git commit -m "feat(server): refactor AppState to use GpuPool, multi-GPU server startup"
```

---

## Task 6: Queue Dispatcher (mold-server)

**Files:**
- Modify: `crates/mold-server/src/queue.rs`

Depends on: Task 4, Task 5

- [ ] **Step 1: Refactor queue to multi-GPU dispatch**

Replace the single-threaded `run_queue_worker()` with a dispatcher that routes to GPU worker threads:

```rust
pub async fn run_queue_dispatcher(
    mut job_rx: tokio::sync::mpsc::Receiver<QueuedRequest>,
    state: Arc<AppState>,
) {
    while let Some(request) = job_rx.recv().await {
        let model_name = &request.model;

        // Estimate VRAM for placement
        let estimated_vram = {
            let config = state.config.read().await;
            estimate_model_vram(model_name, &config)
        };

        // Select worker via placement strategy
        let worker = match state.gpu_pool.select_worker(model_name, estimated_vram) {
            Some(w) => w,
            None => {
                tracing::error!(model = %model_name, "No GPU available for model");
                let _ = request.result_tx.send(Err(anyhow::anyhow!(
                    "No GPU available for model {model_name}"
                )));
                continue;
            }
        };

        // Increment in-flight BEFORE sending to worker (prevents TOCTOU)
        worker.in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Build GpuJob
        let job = GpuJob {
            model: model_name.to_string(),
            request: request.generate_request,
            progress_tx: request.progress_tx,
            result_tx: request.result_tx,
            output_dir: request.output_dir,
            config: state.config.clone(),
        };

        // Dispatch to worker's dedicated thread
        if worker.job_tx.try_send(job).is_err() {
            worker.in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            tracing::warn!(gpu = worker.gpu.ordinal, "GPU worker channel full");
            // Job is lost here — the result_tx was moved into the job
            // This shouldn't happen if queue_size is configured properly
        }
    }
}

fn estimate_model_vram(model_name: &str, config: &mold_core::Config) -> u64 {
    // Use manifest metadata for estimation
    mold_inference::device::estimate_peak_memory_by_name(model_name, config)
        .unwrap_or(8_000_000_000) // 8GB default fallback
}
```

- [ ] **Step 2: Update queue spawn in lib.rs**

Replace the old `run_queue_worker` spawn with:

```rust
tokio::spawn(queue::run_queue_dispatcher(job_rx, state.clone()));
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p mold-ai-server`

- [ ] **Step 4: Commit**

```bash
git add crates/mold-server/src/queue.rs crates/mold-server/src/lib.rs
git commit -m "feat(server): multi-GPU queue dispatcher with TOCTOU-safe in-flight tracking"
```

---

## Task 7: API Routes Update (mold-server)

**Files:**
- Modify: `crates/mold-server/src/routes.rs`

Depends on: Task 5, Task 6

- [ ] **Step 1: Update server_status() endpoint**

Replace the existing status handler (~line 1055-1095) to use GpuPool:

```rust
async fn server_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let gpu_statuses = state.gpu_pool.gpu_status();
    let models_loaded: Vec<String> = gpu_statuses.iter()
        .filter_map(|g| g.loaded_model.clone())
        .collect();
    let busy = gpu_statuses.iter().any(|g| g.state == GpuWorkerState::Generating);

    let status = ServerStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded: models_loaded.clone(),
        model: models_loaded.first().cloned(), // backwards compat
        busy,
        current_generation: None, // aggregated from workers if needed
        gpu_info: gpu_statuses.first().map(|g| GpuInfo {
            name: g.name.clone(),
            vram_total_mb: (g.vram_total_bytes / 1_000_000) as u32,
            vram_used_mb: (g.vram_used_bytes / 1_000_000) as u32,
        }),
        uptime_secs: Some(state.startup_time.elapsed().as_secs()),
        gpus: Some(gpu_statuses),
        queue_depth: Some(0), // TODO: track queue depth
        queue_capacity: Some(state.queue_capacity),
    };

    Json(status)
}
```

- [ ] **Step 2: Update load_model() endpoint**

Add optional `gpu` field to the load request body:

```rust
#[derive(Deserialize)]
struct LoadModelRequest {
    model: String,
    #[serde(default)]
    gpu: Option<usize>,
}
```

Route to specific GPU if requested, otherwise let placement strategy decide.

- [ ] **Step 3: Update unload_model() endpoint**

Accept optional `model` or `gpu` field:

```rust
#[derive(Deserialize)]
struct UnloadRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    gpu: Option<usize>,
}
```

If model specified, find which worker has it and unload. If gpu specified, unload that worker. If neither, unload all.

- [ ] **Step 4: Add queue-full 503 response**

In the generate endpoint, when the queue channel is full, return:

```rust
(StatusCode::SERVICE_UNAVAILABLE, Json(json!({
    "error": "Queue full — all GPUs busy",
    "queue_size": state.queue_capacity,
    "active_gpus": state.gpu_pool.worker_count(),
})))
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p mold-ai-server`

- [ ] **Step 6: Commit**

```bash
git add crates/mold-server/src/routes.rs
git commit -m "feat(server): multi-GPU status, load/unload, and queue-full 503 endpoints"
```

---

## Task 8: CLI Flags & Display (mold-cli)

**Files:**
- Modify: `crates/mold-cli/src/main.rs`
- Modify: `crates/mold-cli/src/commands/generate.rs`
- Modify: `crates/mold-cli/src/commands/ps.rs`

Depends on: Task 1, Task 2

- [ ] **Step 1: Add --gpus and --queue-size flags to serve command**

In the serve command args:

```rust
/// Comma-separated GPU ordinals to use (default: all)
#[arg(long, env = "MOLD_GPUS")]
gpus: Option<String>,

/// Max queued requests before 503 (default: 200)
#[arg(long, env = "MOLD_QUEUE_SIZE", default_value = "200")]
queue_size: usize,
```

Parse `--gpus` into `GpuSelection`:

```rust
let gpu_selection = match &args.gpus {
    Some(s) => GpuSelection::parse(s)?,
    None => config.gpu_selection(),
};
```

Pass to `run_server()`.

- [ ] **Step 2: Add --gpus flag to run command for local mode**

```rust
/// Comma-separated GPU ordinals to use for local generation (default: all)
#[arg(long, env = "MOLD_GPUS")]
gpus: Option<String>,
```

In `generate_local()`, use `select_best_gpu()`:

```rust
let gpu_selection = match &args.gpus {
    Some(s) => GpuSelection::parse(s)?,
    None => config.gpu_selection(),
};
let gpus = discover_gpus();
let available = filter_gpus(&gpus, &gpu_selection);
let best = select_best_gpu(&available).unwrap_or_else(|| {
    // Fallback to ordinal 0
    &DiscoveredGpu { ordinal: 0, name: "default".into(), total_vram_bytes: 0, free_vram_bytes: 0 }
});
// Pass best.ordinal to create_engine_with_pool()
```

- [ ] **Step 3: Update mold ps display for multi-GPU**

In the ps command, parse the new `gpus` array from `ServerStatus`:

```rust
if let Some(gpus) = &status.gpus {
    for gpu in gpus {
        let model = gpu.loaded_model.as_deref().unwrap_or("(none)");
        let state_str = match gpu.state {
            GpuWorkerState::Generating => "[generating]",
            GpuWorkerState::Idle => "[idle]",
            GpuWorkerState::Loading => "[loading]",
            GpuWorkerState::Degraded => "[degraded]",
        };
        let vram_used_gb = gpu.vram_used_bytes as f64 / 1e9;
        let vram_total_gb = gpu.vram_total_bytes as f64 / 1e9;
        println!(
            "GPU {} ({}, {:.0}GB):  {:<20} {}  VRAM: {:.1}/{:.1} GB",
            gpu.ordinal, gpu.name, vram_total_gb, model, state_str, vram_used_gb, vram_total_gb
        );
    }
    if let (Some(depth), Some(capacity)) = (status.queue_depth, status.queue_capacity) {
        println!("Queue: {}/{}", depth, capacity);
    }
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p mold-ai`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/mold-cli/
git commit -m "feat(cli): add --gpus and --queue-size flags, multi-GPU ps display"
```

---

## Task 9: Full Workspace Compilation & Tests

**Files:** All

Depends on: All previous tasks

- [ ] **Step 1: Full workspace check**

Run: `cargo check --workspace`
Fix any remaining compilation errors across crate boundaries.

- [ ] **Step 2: Run existing tests**

Run: `cargo test --workspace`
All existing tests must pass. Fix any breakage caused by the new ordinal parameters.

- [ ] **Step 3: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Fix all warnings.

- [ ] **Step 4: Run fmt**

Run: `cargo fmt --all`

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve compilation errors and test failures from multi-GPU refactor"
```

---

## Task 10: Integration Verification

**Files:** None (testing only)

Depends on: Task 9

- [ ] **Step 1: Verify single-GPU backwards compat**

On a single-GPU machine, start the server with no `--gpus` flag:
```bash
cargo run -p mold-ai -- serve
```
Verify `GET /api/status` returns a `gpus` array with one entry and the existing `model` field still populated.

- [ ] **Step 2: Verify --gpus flag parsing**

```bash
MOLD_GPUS=0 cargo run -p mold-ai -- serve
```
Should start with only GPU 0.

- [ ] **Step 3: Verify mold ps output**

```bash
cargo run -p mold-ai -- ps
```
Should show per-GPU status lines.

- [ ] **Step 4: Verify queue-full 503**

Send 201+ concurrent requests to a server with `--queue-size 200`. The 201st should get a 503.

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit -m "feat: multi-GPU support with per-GPU worker pool and smart placement"
```

---

## Dependency Graph

```
Task 1 (core types) ──┬──► Task 2 (device.rs) ──► Task 3 (engine ordinal)
                       │                                    │
                       ├──► Task 4 (GpuPool) ◄─────────────┘
                       │         │
                       │         ▼
                       │    Task 5 (AppState) ──► Task 6 (queue) ──► Task 7 (routes)
                       │
                       └──► Task 8 (CLI flags)
                                                                         │
                                                              Task 9 (compile) ◄──┘
                                                                   │
                                                              Task 10 (verify)
```

**Parallelizable:** Tasks 1→2→3 can run in parallel with nothing. Tasks 4 and 8 can start as soon as Tasks 1+2 complete. Task 5 needs 4. Tasks 6+7 need 5. Task 9 is the join point.
