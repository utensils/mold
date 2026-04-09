# Multi-GPU Support Design

> Concurrent model placement across multiple GPUs with smart routing and configurable GPU selection.

## Goals

- Automatically discover and use all available GPUs
- Allow users to select specific GPUs via CLI flag, env var, or config
- Place models on GPUs intelligently (idle-first, VRAM-fit tiebreaker)
- Process requests concurrently across GPUs (one generation per GPU at a time)
- Queue requests with configurable backpressure (default 200)
- Maintain backwards compatibility for single-GPU setups

## Non-Goals

- Tensor parallelism (sharding a single model across GPUs) — candle lacks NCCL/all-reduce primitives
- Pipeline parallelism (T5 on GPU 0, transformer on GPU 1 for one model) — future work
- Cross-GPU batch splitting — each request runs entirely on one GPU

## Architecture: Per-GPU Worker Pool (Approach A)

```
Request Queue (bounded, 200 default)
    │
    ▼
GpuPool (placement decision)
    ├──► GpuWorker[0] → ModelCache[0] → Engine on GPU 0
    ├──► GpuWorker[1] → ModelCache[1] → Engine on GPU 1
    └──► GpuWorker[2] → ModelCache[2] → Engine on GPU 2
```

Each GPU gets its own worker with its own `ModelCache`, load lock, and engine snapshot. Workers operate independently — GPU 0 generating doesn't block GPU 1 from loading a model.

---

## 1. GPU Discovery & Configuration

### GpuInfo

```rust
pub struct GpuInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}
```

### Discovery

`discover_gpus()` in `device.rs`:
- CUDA: iterates `cuDeviceGetCount()` → `cuDeviceGetName()` + `cuMemGetInfo()` per ordinal
- Metal: returns single device (macOS doesn't expose multi-GPU for Metal in practice)
- CPU: returns empty vec

### Selection

```rust
pub enum GpuSelection {
    All,
    Specific(Vec<usize>),
}
```

Parsed from:
- CLI: `--gpus 0,1,2`
- Env: `MOLD_GPUS=0,1,2`
- Config: `gpus = [0, 1, 2]`
- Omitted: `All` (auto-detect)

Precedence: CLI > env > config > default.

Validates that requested ordinals exist in discovered GPUs.

### create_device() Change

```rust
// Before
pub fn create_device(progress: &ProgressReporter) -> Result<Device>

// After
pub fn create_device(ordinal: usize, progress: &ProgressReporter) -> Result<Device>
```

All callers updated to pass the ordinal from their assigned GPU.

### VRAM Functions

All existing VRAM functions (`free_vram_bytes`, `vram_used_estimate`, `reclaim_gpu_memory`) take an `ordinal` parameter instead of hardcoding device 0.

---

## 2. GpuPool & GpuWorker

### GpuWorker

```rust
pub struct GpuWorker {
    pub gpu: GpuInfo,
    pub model_cache: Arc<Mutex<ModelCache>>,
    pub engine_snapshot: Arc<RwLock<EngineSnapshot>>,
    pub model_load_lock: Arc<Mutex<()>>,
    pub shared_pool: Arc<Mutex<SharedPool>>,  // Shared across all workers
}
```

Each worker gets its own `ModelCache` instance (existing LRU type). The `SharedPool` for tokenizer caching is shared across all workers since tokenizers are CPU-side and read-only after init.

### GpuPool

```rust
pub struct GpuPool {
    workers: Vec<GpuWorker>,
}

impl GpuPool {
    pub fn select_worker(&self, model_name: &str, estimated_vram: u64) -> Option<&GpuWorker>;
    pub fn find_loaded(&self, model_name: &str) -> Option<&GpuWorker>;
    pub fn gpu_status(&self) -> Vec<GpuWorkerStatus>;
}
```

### Placement Strategy (in order)

1. **Already loaded** — Check all workers' caches for the model with `Gpu` residency. If found, route there. If multiple GPUs have it, prefer the one with fewer in-flight requests.
2. **Idle GPU** — Find workers with no `Gpu`-resident model. Pick the one where the model fits best (smallest GPU with enough VRAM — VRAM-fit tiebreaker).
3. **Evict LRU** — All GPUs busy. Pick the GPU where the model fits with the most headroom after evicting its LRU model.
4. **Doesn't fit** — Return error (model too large for any available GPU).

### AppState Changes

```rust
// Before
pub model_cache: Arc<Mutex<ModelCache>>,
pub engine_snapshot: Arc<RwLock<EngineSnapshot>>,
pub model_load_lock: Arc<Mutex<()>>,

// After
pub gpu_pool: Arc<GpuPool>,
```

Existing `model_manager.rs` functions refactored to operate on a specific `GpuWorker` rather than global state. Internal logic stays nearly identical — just scoped to one GPU.

---

## 3. Queue & Concurrency

### Multi-GPU Dispatch

The queue becomes a dispatcher. Multiple jobs run concurrently (one per GPU):

```rust
loop {
    let job = queue.recv().await;
    let worker = gpu_pool.select_worker(&job.model, estimated_vram)?;

    tokio::spawn(async move {
        let _lock = worker.model_load_lock.lock().await;
        ensure_model_ready(worker, &job.model).await?;

        let result = tokio::task::spawn_blocking(move || {
            let mut cache = worker.model_cache.blocking_lock();
            let engine = cache.get_mut(&job.model).unwrap();
            engine.generate(&job.request)
        }).await?;

        job.response_tx.send(result);
    });
}
```

### Bounded Queue

`tokio::sync::mpsc::channel(queue_size)` where `queue_size` defaults to 200.

Configurable via:
- CLI: `--queue-size 200`
- Env: `MOLD_QUEUE_SIZE=200`
- Config: `queue_size = 200`

When full, returns HTTP 503 with:
```json
{
    "error": "Queue full — all GPUs busy",
    "queue_size": 200,
    "active_gpus": 3
}
```

Includes `Retry-After` header.

### In-Flight Tracking

Track active jobs per GPU for placement tiebreaking. When multiple GPUs have the same model loaded, prefer the one with fewer in-flight requests.

### SSE Streaming

No changes. Progress callbacks already work per-engine. Each GPU's engine fires progress events independently to its request's SSE channel.

---

## 4. CLI Changes

### `mold run --local`

Pick the best single GPU — the one with the most free VRAM from the allowed set:

```rust
let gpus = discover_gpus();
let available = filter_gpus(&gpus, &gpu_selection);
let best = select_best_gpu(&available, estimated_model_vram);
```

No caching, no placement strategy. Just "biggest available card."

`--gpus` flag still respected to constrain which GPU it picks from.

### `mold serve` New Flags

```
--gpus <ORDINALS>     Comma-separated GPU ordinals (default: all)
--queue-size <N>      Max queued requests before 503 (default: 200)
```

### `mold ps` Output

```
GPU 0 (RTX 4090, 24GB):  flux-dev:q8        [generating]  VRAM: 18.2/24.0 GB
GPU 1 (RTX 4090, 24GB):  sdxl-turbo:fp16    [idle]        VRAM: 6.1/24.0 GB
GPU 2 (RTX 3060, 12GB):  sd15:fp16          [idle]        VRAM: 4.2/12.0 GB
Queue: 2/200
```

### `mold run` (Remote)

No client-side GPU flags in remote mode. Server decides placement transparently.

### New Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MOLD_GPUS` | (all) | Comma-separated GPU ordinals |
| `MOLD_QUEUE_SIZE` | `200` | Max queued requests |

### Config File Additions

```toml
gpus = [0, 1, 2]    # optional, omit for all
queue_size = 200     # optional
```

---

## 5. API Surface Changes

### `GET /api/status`

```json
{
    "status": "running",
    "gpus": [
        {
            "ordinal": 0,
            "name": "NVIDIA RTX 4090",
            "vram_total_bytes": 25769803776,
            "vram_used_bytes": 19327352832,
            "loaded_model": "flux-dev:q8",
            "state": "generating"
        },
        {
            "ordinal": 1,
            "name": "NVIDIA RTX 4090",
            "vram_total_bytes": 25769803776,
            "vram_used_bytes": 6543982592,
            "loaded_model": "sdxl-turbo:fp16",
            "state": "idle"
        }
    ],
    "queue_depth": 2,
    "queue_capacity": 200
}
```

Backwards compat: keep existing top-level `model` field populated with first GPU's loaded model (or null).

### `POST /api/models/load`

Optional `gpu` field to pin to a specific ordinal:

```json
{
    "model": "flux-dev:q8",
    "gpu": 1
}
```

Omit `gpu` and placement strategy decides.

### `DELETE /api/models/unload`

Now accepts optional target:

```json
{ "model": "flux-dev:q8" }
```

Or by GPU: `{ "gpu": 1 }`. Omit both to unload all (backwards compat).

### `POST /api/generate`

No request changes. Server decides GPU placement.

### `GenerateResponse`

New optional `gpu` field:

```json
{
    "images": [...],
    "seed": 42,
    "gpu": 0
}
```

---

## 6. Inference Layer Changes

### EngineBase

```rust
pub struct EngineBase<L> {
    // ... existing fields ...
    pub gpu_ordinal: usize,
}
```

Each engine's `load()` calls `create_device(self.base.gpu_ordinal, ...)`.

### Factory

`create_engine()` and `create_engine_with_pool()` gain `ordinal: usize` parameter, threaded through to `EngineBase`.

### What Changes

- `device.rs`: all functions take ordinal parameter
- `factory.rs`: passes ordinal through to engines
- `engine_base.rs`: stores `gpu_ordinal`
- `expand.rs`: accepts ordinal, defaults to least-loaded GPU

### What Doesn't Change

- Individual pipeline implementations (FLUX, SD3, SDXL, etc.) — use ordinal through `create_device()`, everything else same
- LoRA loading — per-engine, device-agnostic
- Tokenizer shared pool — CPU-side, shared across all GPUs
- Scheduler logic (DDIM, Euler, etc.) — operates on tensors already on the right device
- Offloading — works per-GPU, no cross-GPU offloading
- VRAM threshold logic — already parameterized by `free_vram`

---

## Crate Impact Summary

| Crate | Changes |
|-------|---------|
| `mold-core` | `GpuInfo`, `GpuSelection`, `GpuStatus` types. `ServerStatus` updated with `gpus` array. `GenerateResponse` gets `gpu` field. Config gets `gpus` and `queue_size`. |
| `mold-inference` | `device.rs` overhaul (ordinal params, `discover_gpus()`). `EngineBase` gets `gpu_ordinal`. Factory passes ordinal. `expand.rs` ordinal-aware. |
| `mold-server` | `GpuPool`, `GpuWorker` structs. `AppState` refactored. Queue becomes multi-GPU dispatcher. Routes updated for new API fields. `model_manager.rs` scoped to per-worker. |
| `mold-cli` | `--gpus` and `--queue-size` flags. `mold ps` multi-GPU display. `mold run --local` best-GPU selection. |
| `mold-discord` | Minor: parse `gpu` field from `GenerateResponse` for display. |
| `mold-tui` | Minor: display per-GPU status if connected to multi-GPU server. |
