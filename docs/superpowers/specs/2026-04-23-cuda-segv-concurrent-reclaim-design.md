# CUDA SEGV — unify chain-route GPU lock under GpuPool

**Date:** 2026-04-23
**Branch:** `feat/multi-prompt-chain-v2-phase3` (fix applies equally to `main`)
**Incident:** `tasks/cuda-segv-concurrent-reclaim-handoff.md`

## Context

On 2026-04-23 at 06:48:56 UTC, the production `mold` server on
`<gpu-host>` (dual GPUs) received `SIGSEGV` inside
`libcuda.so.1::cuModuleGetFunction` while loading `qwen-image-2512:q8` on
device 0. The crash occurred after three `POST /api/generate/chain/stream`
requests for `ltx-2.3-22b-distilled:fp8` and three concurrent
`POST /api/generate/stream` requests for `qwen-image-2512:q8` arrived within
a ~70 s window, with two `"CUDA primary context reset for device 0"` log
events bracketing the crash.

### What actually happened

Two call paths inside the same server process raced
`cuDevicePrimaryCtxReset_v2` against GPU 0's primary context:

| Path | Lock held | Cache used | Reclaim ordinal |
|------|-----------|------------|-----------------|
| chain route (`routes_chain::run_chain`) | `state.model_load_lock` (global) via `model_manager::ensure_model_ready` | `state.model_cache` | hardcoded `0` |
| single-clip (`gpu_worker::process_job`) | `workers[n].model_load_lock` (per-GPU) | `workers[n].model_cache` | `worker.gpu.ordinal` |

These are **entirely disjoint mutexes**. They do not serialize against
each other. The CUDA primary context on device 0, however, is a single
shared resource. When the single-clip path called
`reclaim_gpu_memory(0)` → `cuDevicePrimaryCtxReset_v2` while the chain
path's `spawn_blocking` task was mid-module-load into the same context,
every module/function handle was invalidated. The next
`cuModuleGetFunction` dereferenced freed memory and died.

### The deeper issue

The chain route is the only remaining caller of the **legacy single-GPU
subsystem** (`state.model_load_lock` + `state.model_cache` + hardcoded
`reclaim_gpu_memory(0)` inside `model_manager`). Everything else already
migrated to the per-worker `GpuPool` when that subsystem was introduced
for multi-GPU support:

- `run_queue_worker` (legacy) is never spawned when
  `gpu_pool.worker_count() > 0` — `run_queue_dispatcher` replaces it
  (`lib.rs:244`).
- `load_model` HTTP handler branches on pool presence (`routes.rs:1214`).
- `unload_model` HTTP handler branches on pool presence (`routes.rs:1455`).
- `/api/upscale` takes `worker.model_load_lock` (`routes.rs:748, 944`).

Only `routes_chain.rs:345` still calls
`model_manager::ensure_model_ready`. That's the regression surface.

### Latent VRAM bug

Beyond the SEGV, the dual-cache arrangement leaks GPU memory silently.
When a chain completes, the engine is restored to `state.model_cache`
and stays there. If a subsequent single-clip request targets the same
model, it checks `worker.model_cache` (empty on the worker side), calls
`ensure_model_ready_sync` which calls `cache.unload_active()` (on
`worker.model_cache` — the already-empty one), and loads a fresh copy.
Two copies of (e.g.) LTX-2 22B now share the GPU. On 24 GB cards, this
is OOM. The fix for the SEGV also eliminates this latent bug by
consolidating onto `worker.model_cache`.

## Goals

1. Eliminate the `cuDevicePrimaryCtxReset_v2` race between chain and
   single-clip paths.
2. Eliminate the dual-cache VRAM leak.
3. Preserve the signed-off v1 render-chain constraint: one chain runs
   on one GPU (`tasks/render-chain-v1-handoff.md`, decision #5).
4. Keep the fix narrow — no refactors unrelated to the bug.

## Non-goals

- Multi-GPU stage fan-out for chains (that's v2).
- Global "one GPU-resident model server-wide" serialization (load-bearing
  for throughput).
- Removing `model_manager` or the CPU/no-worker fallback path.
- Per-call `cuCtxSynchronize` "try to make the reset safe under
  concurrency" schemes. The driver API contract explicitly forbids
  concurrent live objects during reset.
- Extending the fix to Metal or CPU backends. `reclaim_gpu_memory` is a
  no-op outside CUDA (`device.rs:498`), so the bug physically cannot
  manifest there.

## Design

### Lock invariant (post-fix)

> For every GPU ordinal *N*, every call to `reclaim_gpu_memory(N)`
> occurs inside a critical section guarded by
> `workers[N].model_load_lock`.

This invariant holds trivially for all existing multi-GPU call sites
today. The chain route is the sole violator, and this design brings
it into compliance.

### Component 1 — new helper `gpu_worker::run_chain_blocking`

Lives alongside `load_blocking` / `unload_blocking` in
`crates/mold-server/src/gpu_worker.rs`.

```rust
/// Run a blocking chain operation on a specific GPU worker. Acquires
/// the per-worker load lock for the full duration, ensures the model
/// is on GPU, takes the engine out of the worker's cache (`take`),
/// passes it to `with_engine`, and restores it afterward. The caller's
/// closure is responsible for running the orchestrator and capturing
/// its output.
///
/// Safe to call from a `tokio::task::spawn_blocking` context. The
/// calling thread is bound to `worker.gpu.ordinal` via
/// `init_thread_gpu_ordinal` for the duration, satisfying
/// `debug_assert_ordinal_matches_thread` inside `reclaim_gpu_memory`.
pub fn run_chain_blocking<T>(
    worker: &GpuWorker,
    model_name: &str,
    config: &Config,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> anyhow::Result<T>,
) -> anyhow::Result<T>;
```

Internally:

1. `init_thread_gpu_ordinal(worker.gpu.ordinal)` behind an RAII
   `ThreadGpuGuard` that calls `clear_thread_gpu_ordinal()` on Drop.
2. Acquire `worker.model_load_lock` (guard held until function return).
3. Call `ensure_model_ready_sync(worker, model_name, config)` — this
   handles load-from-disk, reload-from-parked, and the reclaim-on-swap
   path using `worker.gpu.ordinal`.
4. `let cached = worker.model_cache.take(model_name)?` — take the
   entry out so the engine can mutate during the chain.
5. `let result = with_engine(cached.engine.as_mut())`.
6. `worker.model_cache.restore(cached)` unconditionally (even on error,
   mirroring the pattern in `routes_chain::run_chain`).
7. Return `result`.

### Component 2 — `routes_chain::run_chain` branches on pool presence

```rust
async fn run_chain(
    state: &AppState,
    req: ChainRequest,
    progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    if state.gpu_pool.worker_count() > 0 {
        run_chain_pooled(state, req, progress_cb).await
    } else {
        run_chain_legacy(state, req, progress_cb).await   // existing code
    }
}
```

`run_chain_legacy` is today's `run_chain` body, renamed and otherwise
untouched. It keeps using `state.chain_lock`, `state.model_load_lock`,
`state.model_cache`, and hardcoded `reclaim_gpu_memory(0)`. This path
is reachable only when no GPU workers were discovered at startup
(CPU-only dev boxes, CI).

`run_chain_pooled`:

1. **Select worker.** First honor explicit placement via
   `gpu_pool.resolve_explicit_placement_gpu(req.placement.as_ref())`.
   If that returns `Some(ord)`, use `gpu_pool.worker_by_ordinal(ord)`.
   Otherwise call `gpu_pool.select_worker(&req.model, estimated_vram)`.
   If both paths return `None`, return
   `ChainRunError::NoWorker(format!("no GPU worker available for model '{model}'"))`.

2. **Announce busy state.** Before `spawn_blocking`:
   - `worker.in_flight.fetch_add(1, SeqCst)`
   - Set `worker.active_generation = Some(ActiveGeneration { model, ... })`

   Both use RAII guards so they're cleared even on panic or error
   (matches `QueueSlot` pattern from `gpu_worker.rs:54`). This makes
   `select_worker` bias other requests away from the chain's worker.

3. **Run the chain.** Inside `spawn_blocking`:

   ```rust
   run_chain_blocking(&worker, &req.model, &config_snapshot, |engine| {
       let renderer = engine.as_chain_renderer()
           .ok_or_else(|| anyhow!("model '{model}' does not support chain"))?;
       let mut orch = Ltx2ChainOrchestrator::new(renderer);
       orch.run(&req, progress_cb.as_deref_mut())
           .map_err(|e| e.into())
   })
   ```

   The `with_engine` closure encapsulates orchestrator setup, run, and
   error-variant translation (for `ChainOrchestratorError::StageFailed`
   / `Invalid` → `ChainRunError`). Keeps `run_chain_blocking` generic.

4. **Post-processing** (stitch, encode, save, gallery, metadata DB)
   runs outside the spawn_blocking task, same as today's legacy path.

### Component 3 — error enum additions

`ChainRunError` gains:

```rust
enum ChainRunError {
    /// No GPU worker available for this model (503 Service Unavailable).
    NoWorker(String),
    /// spawn_blocking task failed to join (500).
    Join(String),
    ...existing variants...
}
```

`NoWorker` maps to `503` via a new branch in
`impl From<ChainRunError> for ApiError`. `Join` maps to `500` through
the existing `ApiError::internal` route. Every other variant's mapping
is unchanged.

### Component 4 — VRAM estimate

`select_worker` wants `estimated_vram: u64`. Reuse
`crate::queue::estimate_model_vram(&req.model)`, which
`routes.rs::load_model` already uses (`routes.rs:1226`). No new code.

### Component 5 — what stays legacy

| Field / function | Fate |
|------------------|------|
| `state.model_cache` | unchanged — used by `run_chain_legacy` |
| `state.model_load_lock` | unchanged — used by `run_chain_legacy` and `model_manager` |
| `state.chain_lock` | unchanged — used by `run_chain_legacy` only (per user call in brainstorm) |
| `state.engine_snapshot` | unchanged |
| `model_manager::ensure_model_ready` | unchanged; add comment noting hardcoded `reclaim_gpu_memory(0)` is legacy-path-only |
| `model_manager::unload_model`, `create_and_load_engine` | unchanged |

All of these remain reachable only when `gpu_pool.worker_count() == 0`.

## Testing

### TDD red test

Location: `crates/mold-server/src/gpu_worker.rs`, under
`#[cfg(test)] mod tests`.

**Fake engine:**

```rust
struct FakeSlowEngine {
    name: String,
    loaded: bool,
    load_sleep: Duration,
}

impl InferenceEngine for FakeSlowEngine {
    fn model_name(&self) -> &str { &self.name }
    fn is_loaded(&self) -> bool { self.loaded }
    fn load(&mut self) -> anyhow::Result<()> {
        std::thread::sleep(self.load_sleep);
        self.loaded = true;
        Ok(())
    }
    fn generate(&mut self, _: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
        unimplemented!("not needed for lock test")
    }
}
```

**Critical-section overlap detector:**

```rust
static MAX_CONCURRENT: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

fn entering() {
    let now = ACTIVE.fetch_add(1, SeqCst) + 1;
    MAX_CONCURRENT.fetch_max(now, SeqCst);
}
fn leaving() { ACTIVE.fetch_sub(1, SeqCst); }
```

**Test body:** build a single-worker pool (ordinal 0) with a
pre-seeded `FakeSlowEngine` parked in `worker.model_cache`. Spawn two
threads, both calling `run_chain_blocking(&worker, model, &config, cb)`
where the closure wraps its body in `entering()` / `leaving()` and
sleeps 50 ms. This tests the new helper's own lock surface end to end.

**Assertion:** `MAX_CONCURRENT.load(SeqCst) == 1`. Before the fix,
`run_chain_blocking` doesn't exist — the test doesn't compile. After
the fix, both callers take `worker.model_load_lock` → serialize → max
is 1.

**Companion test** — verify `run_chain_blocking` serializes against
`load_blocking` specifically (the path hit on 2026-04-23): Thread A
calls `run_chain_blocking` with the instrumented closure; Thread B
calls `load_blocking(&worker, model, &config)` before which and after
which the test harness (not `load_blocking` internals) increments /
decrements `ACTIVE`. The assertion is the same. By inspection of
`gpu_worker.rs:374`, `load_blocking` already takes
`worker.model_load_lock`, so this test guards against anyone
accidentally changing that.

### Secondary tests

1. **Explicit placement routing** — a chain with `placement.gpu = 1`
   on a two-worker pool calls `run_chain_blocking` against worker 1,
   not worker 0. Unit test on the selection logic only (no GPU needed).

2. **`NoWorker` error** — empty pool + chain request returns
   `ChainRunError::NoWorker` → 503 status. Route-level test with mock
   state.

3. **`active_generation` / `in_flight` side effects cleared on
   completion and on error.** Drop-guard RAII test: spawn a chain that
   panics inside the orchestrator closure, verify `worker.in_flight ==
   0` and `worker.active_generation.is_none()` after `spawn_blocking`
   returns.

### Regression tests that must stay green

- `gpu_pool::tests::*` (8 tests — selection, placement, degraded
  skipping).
- `routes_test::*`.
- `mold-core` `chain_client` wiremock integration.
- `mold-inference` `ltx2::chain::*` orchestrator + tail helpers.

### Build matrix

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` (runs the new test)
- `cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4`
- `bun run fmt:check && bun run verify && bun run build` in `website/`

No CUDA feature required for the new test. `FakeSlowEngine` is pure Rust.

## Open risks

- **`spawn_blocking` thread pool exhaustion.** A long chain parks a
  blocking worker for the full duration. Today the chain also parks
  one via `spawn_blocking` → no change. Tokio's default pool is 512;
  with v1's single-chain-per-GPU, dual-GPU servers use at most two.
- **`worker.model_load_lock` held for 10+ minutes.** Admin
  `/models/load` on the busy worker, `/models/unload` on the busy
  worker, `/api/upscale` pinned to the busy worker — all block until
  chain completes. This matches today's semantics for the single-clip
  path, which already holds the same lock for the duration of a
  single generation. Consistent.
- **`active_generation.model` semantics.** Today this field describes
  a *single-clip* generation. Setting it for a chain means status
  endpoints (`/api/status`, TUI) will report the chain's model as the
  active generation, which is correct. No schema change; the model name
  is the only field consumed.

## Commit shape

One commit: `fix(server): chain route acquires per-GPU worker lock to
prevent CUDA primary-context reset race`. Includes:

- `run_chain_blocking` in `gpu_worker.rs`.
- `run_chain_pooled` + `run_chain_legacy` split in `routes_chain.rs`.
- `ChainRunError::{NoWorker, Join}` variants + mappings.
- Regression tests for lock serialization and placement routing.
- Inline comments on the three `reclaim_gpu_memory(0)` sites in
  `model_manager.rs` clarifying they're legacy-path-only.

Commit message explains (per the handoff): (a) what the two schedulers
were doing, (b) why `cuDevicePrimaryCtxReset_v2` is unsafe under that
race, (c) the one-lock-per-ordinal resolution via `GpuPool`, (d) the
regression test.
