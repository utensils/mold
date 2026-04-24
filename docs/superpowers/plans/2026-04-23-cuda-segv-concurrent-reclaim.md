# CUDA SEGV — Chain-Route GPU Lock Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the chain route (`routes_chain::run_chain`) to use `GpuPool`'s per-worker `model_load_lock` so it can't race `cuDevicePrimaryCtxReset_v2` against single-clip generation on the same GPU.

**Architecture:** New `gpu_worker::run_chain_blocking` helper encapsulates the lock+ensure+take/restore pattern. `run_chain` splits into `run_chain_pooled` (multi-worker, new path) and `run_chain_legacy` (no-worker CPU fallback, existing code verbatim). Legacy `state.{model_cache, model_load_lock, chain_lock}` stay for CPU fallback only.

**Tech Stack:** Rust 2021, tokio (async runtime + spawn_blocking), std::sync::Mutex (per-worker lock), existing axum/axum-extra routing, anyhow for errors. TDD via `std::thread` and atomic counters — no CUDA feature required for the regression test.

**Spec:** `docs/superpowers/specs/2026-04-23-cuda-segv-concurrent-reclaim-design.md`

**Verification commands (run between tasks as noted):**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-server --lib                    # fastest — new tests live here
cargo test --workspace                                # full sweep before final commit
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
```

---

## File Structure

**Modify:**

- `crates/mold-server/src/gpu_worker.rs` — add `run_chain_blocking` helper + `#[cfg(test)] mod tests` additions.
- `crates/mold-server/src/routes_chain.rs` — split `run_chain` into `run_chain_pooled` + `run_chain_legacy`; add `ChainRunError::{NoWorker, Join}` + `ApiError` mappings; call `run_chain_blocking` from the pooled path.
- `crates/mold-server/src/model_manager.rs` — add inline comments on the three `reclaim_gpu_memory(0)` sites clarifying they're legacy-path-only.

**No new files.** No changes to `mold-core`, `mold-inference`, `mold-db`, `mold-cli`, `mold-tui`, or `web/`.

---

## Task 1: Scaffold test harness (FakeSlowEngine) and a failing red test for `run_chain_blocking`

**Why first:** TDD — we need a weight-free `InferenceEngine` impl to drive concurrent threads through the helper. Writing the harness and test before the helper exists guarantees the test fails in the right way (doesn't compile) until the fix lands.

**Files:**
- Modify: `crates/mold-server/src/gpu_worker.rs` (add `#[cfg(test)] mod tests`; if one already exists, append inside).

- [ ] **Step 1.1: Add the test module skeleton with `FakeSlowEngine` and a helper to build a single-worker pool**

Append to `crates/mold-server/src/gpu_worker.rs` at end of file (after `clear_active_generation` at line 410):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_cache::{ModelCache, ModelResidency};
    use mold_core::{Config, GenerateRequest, GenerateResponse};
    use mold_inference::device::DiscoveredGpu;
    use mold_inference::shared_pool::SharedPool;
    use mold_inference::InferenceEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::Duration;

    /// Weight-free engine that sleeps in `load()` to widen the critical-section
    /// window during concurrency tests.
    struct FakeSlowEngine {
        name: String,
        loaded: bool,
        load_sleep: Duration,
    }

    impl FakeSlowEngine {
        fn boxed(name: &str, load_sleep: Duration) -> Box<dyn InferenceEngine> {
            Box::new(Self {
                name: name.to_string(),
                loaded: false,
                load_sleep,
            })
        }
    }

    impl InferenceEngine for FakeSlowEngine {
        fn generate(&mut self, _req: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
            unreachable!("FakeSlowEngine is not used for generation in tests")
        }
        fn model_name(&self) -> &str {
            &self.name
        }
        fn is_loaded(&self) -> bool {
            self.loaded
        }
        fn load(&mut self) -> anyhow::Result<()> {
            std::thread::sleep(self.load_sleep);
            self.loaded = true;
            Ok(())
        }
        fn unload(&mut self) {
            self.loaded = false;
        }
    }

    fn single_worker_pool_with_parked(model: &str, load_sleep: Duration) -> Arc<GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel::<GpuJob>(2);
        let mut cache = ModelCache::new(3);
        // Seed as Unloaded so `ensure_model_ready_sync` hits its reload path
        // and calls `engine.load()` — that's where the sleep widens the window.
        cache.insert(FakeSlowEngine::boxed(model, load_sleep), 0);
        Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal: 0,
                name: "fake-gpu-0".to_string(),
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            degraded_until: RwLock::new(None),
            job_tx,
        })
    }
}
```

- [ ] **Step 1.2: Verify the skeleton compiles**

Run: `cargo test -p mold-ai-server --lib gpu_worker::tests -- --nocapture`
Expected: zero tests pass (no `#[test]` functions yet) but code compiles without errors.

- [ ] **Step 1.3: Add the failing lock-serialization test**

Append inside the `tests` module from Step 1.1, just before the closing `}`:

```rust
    /// Two concurrent callers into `run_chain_blocking` on the same worker
    /// must serialize — `MAX_CONCURRENT` must never exceed 1.
    ///
    /// Fails to compile until `run_chain_blocking` is implemented in Task 2.
    #[test]
    fn run_chain_blocking_serializes_same_worker() {
        let worker = single_worker_pool_with_parked("fake-model", Duration::from_millis(30));
        let config = Config::default();

        let active = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let instrumented = |active: Arc<AtomicUsize>, max_concurrent: Arc<AtomicUsize>| {
            move |_engine: &mut dyn InferenceEngine| -> anyhow::Result<()> {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(50));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
        };

        let worker_a = worker.clone();
        let config_a = config.clone();
        let a = active.clone();
        let m = max_concurrent.clone();
        let t_a = std::thread::spawn(move || {
            run_chain_blocking(&worker_a, "fake-model", &config_a, instrumented(a, m))
                .expect("prep ok")
                .expect("closure ok");
        });

        let worker_b = worker.clone();
        let config_b = config.clone();
        let a = active.clone();
        let m = max_concurrent.clone();
        let t_b = std::thread::spawn(move || {
            run_chain_blocking(&worker_b, "fake-model", &config_b, instrumented(a, m))
                .expect("prep ok")
                .expect("closure ok");
        });

        t_a.join().unwrap();
        t_b.join().unwrap();

        assert_eq!(
            max_concurrent.load(Ordering::SeqCst),
            1,
            "two concurrent run_chain_blocking calls must serialize on worker.model_load_lock"
        );
    }
```

- [ ] **Step 1.4: Run the test and confirm it fails with the expected error**

Run: `cargo test -p mold-ai-server --lib gpu_worker::tests::run_chain_blocking_serializes_same_worker 2>&1 | tail -15`
Expected: compile error referencing `run_chain_blocking` not found in scope (or similar `cannot find function` message). This is the intended red state.

- [ ] **Step 1.5: Commit (test-only)**

```bash
git add crates/mold-server/src/gpu_worker.rs
git commit -m "$(cat <<'EOF'
test(server): failing test for concurrent run_chain_blocking serialization

Scaffolds FakeSlowEngine (weight-free InferenceEngine) and a
single-worker-pool fixture. The test drives two threads through the
not-yet-existing run_chain_blocking helper and asserts they serialize
via worker.model_load_lock — red until Task 2 lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement `run_chain_blocking` and make the test pass

**Why:** The helper is the whole point of the fix — it's the single entry point through which every chain request in multi-worker mode will pass, and the place where the per-worker lock is acquired.

**Files:**
- Modify: `crates/mold-server/src/gpu_worker.rs` (add `pub fn run_chain_blocking` + a public `ChainPrep` type alias).

- [ ] **Step 2.1: Add the `ChainPrep` type alias and `run_chain_blocking` function**

Append to `crates/mold-server/src/gpu_worker.rs` right before the `#[cfg(test)]` module from Task 1 (i.e. after `clear_active_generation` at line 410, before line 412):

```rust
/// Return type for [`run_chain_blocking`]. The outer `Result` carries
/// helper-prep errors (ensure_model_ready + cache take); the inner `Result`
/// is whatever the caller's closure returned. Closure errors pass through
/// unchanged so the caller can distinguish orchestrator-specific failures
/// (StageFailed, Invalid) from prep failures (ensure/cache).
pub type ChainPrep<T, E> = Result<Result<T, E>, anyhow::Error>;

/// Run a blocking chain operation on a specific GPU worker.
///
/// Acquires `worker.model_load_lock` for the full duration, binds the current
/// thread to `worker.gpu.ordinal` (so `reclaim_gpu_memory` debug asserts are
/// satisfied), ensures the model is loaded on GPU, takes the engine out of
/// the worker's cache, passes it to `with_engine`, and restores the engine
/// unconditionally on both success and closure failure.
///
/// Safe to call from inside `tokio::task::spawn_blocking`. The calling thread
/// can be any thread — the `ThreadGpuGuard` clears the thread-local on return.
///
/// # Errors
///
/// Returns `Err(anyhow::Error)` from the outer Result if:
/// - `ensure_model_ready_sync` fails (bad config, disk IO, load error).
/// - The engine vanishes from the cache between ensure and take (cache race).
///
/// Returns `Ok(Err(E))` if the closure itself returned an error — caller
/// preserves the closure's typed error for precise HTTP status mapping.
pub fn run_chain_blocking<T, E>(
    worker: &GpuWorker,
    model_name: &str,
    config: &mold_core::Config,
    with_engine: impl FnOnce(&mut dyn mold_inference::InferenceEngine) -> Result<T, E>,
) -> ChainPrep<T, E> {
    // Bind the thread to this worker's ordinal for the duration of the call.
    // `reclaim_gpu_memory` inside ensure_model_ready_sync debug-asserts this
    // matches its ordinal argument; without it, a stray caller on an unbound
    // thread would panic in debug builds.
    struct ThreadGpuGuard;
    impl Drop for ThreadGpuGuard {
        fn drop(&mut self) {
            mold_inference::device::clear_thread_gpu_ordinal();
        }
    }
    mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
    let _thread_gpu = ThreadGpuGuard;

    // Acquire the per-worker load lock. Held for the entire chain duration —
    // single-clip generations on this worker queue behind us on the same lock.
    let _load_lock = worker
        .model_load_lock
        .lock()
        .map_err(|e| anyhow::anyhow!("worker.model_load_lock poisoned: {e}"))?;

    // Ensure the model is GPU-resident on this worker. Handles load-from-disk,
    // parked-reload, and the reclaim-on-swap path using worker.gpu.ordinal.
    ensure_model_ready_sync(worker, model_name, config)?;

    // Take the engine out of the worker's cache so the closure can mutate it.
    let cached = {
        let mut cache = worker
            .model_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("worker.model_cache poisoned: {e}"))?;
        cache.take(model_name).ok_or_else(|| {
            anyhow::anyhow!("cache race: engine '{model_name}' vanished after ensure_model_ready")
        })?
    };

    // Run the closure. Capture panics so we can still restore the engine
    // before propagating — otherwise a panic leaks the engine out of the cache.
    let mut cached = cached;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_engine(cached.engine.as_mut())
    }));

    // Restore unconditionally. On success, the engine is ready for the next
    // request. On closure error, same. On panic, same — and we re-raise after.
    {
        let mut cache = worker
            .model_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("worker.model_cache poisoned during restore: {e}"))?;
        cache.restore(cached);
    }

    match result {
        Ok(inner) => Ok(inner),
        Err(panic_payload) => std::panic::resume_unwind(panic_payload),
    }
}
```

- [ ] **Step 2.2: Run the red test — it should now pass**

Run: `cargo test -p mold-ai-server --lib gpu_worker::tests::run_chain_blocking_serializes_same_worker -- --nocapture`
Expected: PASS.

- [ ] **Step 2.3: Run the full mold-ai-server test suite to catch regressions**

Run: `cargo test -p mold-ai-server --lib`
Expected: all existing tests pass, new test passes, zero failures.

- [ ] **Step 2.4: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: zero output from fmt, zero warnings from clippy.

- [ ] **Step 2.5: Commit**

```bash
git add crates/mold-server/src/gpu_worker.rs
git commit -m "$(cat <<'EOF'
feat(server): add run_chain_blocking helper on GpuPool

New helper encapsulates the lock+ensure+take+restore pattern used by
the chain route, acquiring worker.model_load_lock so chains serialize
against single-clip generations on the same GPU. Thread is bound to
worker.gpu.ordinal via ThreadGpuGuard so reclaim_gpu_memory debug
asserts stay satisfied. Restores engine to cache on closure error or
panic so a failed chain doesn't leak the engine out of the cache.

Task 1's red test now passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `ChainRunError::NoWorker` + `ChainRunError::Join` variants and ApiError mappings

**Why:** The new pooled path has two error sources the legacy path doesn't: (1) no GPU worker available for this model, (2) `spawn_blocking` `JoinError`. Without these variants, we'd collapse them into `Internal` and lose the 503-for-capacity-exhaustion semantic.

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs:280-320` (enum + From impl).

- [ ] **Step 3.1: Add the two new variants to the `ChainRunError` enum**

Modify `crates/mold-server/src/routes_chain.rs` — change the enum at lines 280-295 to add `NoWorker` and `Join`:

```rust
enum ChainRunError {
    /// Model family doesn't support chain rendering (422).
    UnsupportedModel(String),
    /// Engine missing from cache after `ensure_model_ready` (500).
    CacheMiss(String),
    /// Orchestrator returned an error mid-chain from an invalid request (502).
    Inference(String),
    /// Orchestrator returned a typed stage failure mid-chain (502 with body).
    StageFailed(mold_core::chain::ChainFailure),
    /// Output encoding failure (500).
    Encode(String),
    /// `StitchPlan::assemble` failed (500).
    StitchFailed(String),
    /// Task panic or join error (500).
    Internal(String),
    /// No GPU worker available to service this chain (503).
    NoWorker(String),
    /// `spawn_blocking` task failed to join (500).
    Join(String),
}
```

- [ ] **Step 3.2: Extend the `From<ChainRunError> for ApiError` impl**

Modify the `impl From<ChainRunError> for ApiError` block at lines 297-320 — add arms for the two new variants inside the `match err` at line 299:

```rust
impl From<ChainRunError> for ApiError {
    fn from(err: ChainRunError) -> Self {
        match err {
            ChainRunError::UnsupportedModel(msg) => ApiError::validation(msg),
            ChainRunError::CacheMiss(msg) => ApiError::internal(msg),
            ChainRunError::Inference(msg) => {
                ApiError::internal_with_status(msg, axum::http::StatusCode::BAD_GATEWAY)
            }
            ChainRunError::StageFailed(failure) => ApiError::internal_with_status(
                failure.stage_error,
                axum::http::StatusCode::BAD_GATEWAY,
            ),
            ChainRunError::Encode(msg) => ApiError::internal(msg),
            ChainRunError::StitchFailed(msg) => ApiError::internal(msg),
            ChainRunError::Internal(msg) => ApiError::internal(msg),
            ChainRunError::NoWorker(msg) => {
                ApiError::internal_with_status(msg, axum::http::StatusCode::SERVICE_UNAVAILABLE)
            }
            ChainRunError::Join(msg) => ApiError::internal(msg),
        }
    }
}
```

(The existing `StageFailed` doc comment above it should remain — do not drop the explanation about the SSE error channel being string-only.)

- [ ] **Step 3.3: Verify compilation**

Run: `cargo check -p mold-ai-server`
Expected: clean.

- [ ] **Step 3.4: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 3.5: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs
git commit -m "$(cat <<'EOF'
feat(server): ChainRunError variants for NoWorker (503) and Join (500)

Prep for the pooled-chain path: NoWorker covers the case where
gpu_pool.select_worker returns None, Join covers spawn_blocking
JoinError. NoWorker maps to 503 Service Unavailable so clients can
back off and retry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Split `run_chain` into `run_chain_pooled` + `run_chain_legacy`

**Why:** The multi-worker path is the fix. The legacy path is today's code, preserved verbatim for no-worker (CPU-only) servers. Splitting first — before filling in `run_chain_pooled` — makes the diff reviewable: one commit moves code, one commit adds new logic.

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs:325-500` (the `async fn run_chain` body).

- [ ] **Step 4.1: Rename `run_chain` to `run_chain_legacy` and add a `run_chain_pooled` stub**

In `crates/mold-server/src/routes_chain.rs`:

1. Rename the existing `async fn run_chain(...)` at line 325 to `async fn run_chain_legacy(...)`. Keep its body exactly as today.

2. Add a new `async fn run_chain` dispatcher ABOVE `run_chain_legacy`:

```rust
/// Dispatch a chain request to the pooled or legacy handler based on
/// whether the server discovered any GPU workers at startup.
///
/// In multi-worker mode (production CUDA / Metal), the pooled path
/// uses `gpu_worker::run_chain_blocking` to acquire the target GPU's
/// per-worker `model_load_lock` — preventing the SEGV race that arose
/// when the legacy path's `reclaim_gpu_memory(0)` collided with a
/// single-clip worker's reset on the same context.
///
/// No-worker mode (CPU-only dev boxes, CI) falls through to the legacy
/// path, which still uses `state.chain_lock` + `state.model_cache`.
async fn run_chain(
    state: &AppState,
    req: ChainRequest,
    progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    if state.gpu_pool.worker_count() > 0 {
        run_chain_pooled(state, req, progress_cb).await
    } else {
        run_chain_legacy(state, req, progress_cb).await
    }
}

/// Multi-worker chain path (stub — filled in by Task 5).
async fn run_chain_pooled(
    _state: &AppState,
    _req: ChainRequest,
    _progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    Err(ChainRunError::Internal(
        "run_chain_pooled not yet implemented (Task 5)".to_string(),
    ))
}
```

- [ ] **Step 4.2: Verify compilation and tests still pass**

Run: `cargo test -p mold-ai-server --lib`
Expected: all existing tests pass (the stub isn't called by any existing test path).

Run: `cargo check -p mold-ai-server`
Expected: clean — the stub compiles.

- [ ] **Step 4.3: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 4.4: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs
git commit -m "$(cat <<'EOF'
refactor(server): split run_chain into pooled + legacy dispatchers

No behavior change. run_chain_legacy holds today's body verbatim for
no-worker (CPU-only) servers. run_chain_pooled is a stub filled in by
the next commit. The top-level run_chain dispatches based on
gpu_pool.worker_count().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `run_chain_pooled`

**Why:** This is the actual fix. It selects a worker, announces busy state so the dispatcher biases away, spawns a blocking task that calls `run_chain_blocking`, and drives the orchestrator inside the helper's closure. Stitching, encoding, and saving happen outside the lock (after the helper returns) — same sequencing as today.

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs` (replace the `run_chain_pooled` stub from Task 4).

- [ ] **Step 5.1: Add the imports needed for `run_chain_pooled`**

Add to the top imports block in `crates/mold-server/src/routes_chain.rs` (near existing `use crate::model_manager;` at line 32):

```rust
use crate::gpu_worker;
use crate::gpu_pool::{ActiveGeneration, GpuWorker};
use mold_inference::ltx2::{ChainOrchestratorError, Ltx2ChainOrchestrator};
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
```

(If any of these are already imported in the file, omit duplicates — `cargo check` will flag it.)

- [ ] **Step 5.2: Replace the `run_chain_pooled` stub with the real implementation**

Replace the stub body from Task 4 with:

```rust
async fn run_chain_pooled(
    state: &AppState,
    req: ChainRequest,
    progress_cb: Option<Box<dyn FnMut(ChainProgressEvent) + Send>>,
) -> Result<(ChainResponse, u64), ChainRunError> {
    // ── Worker selection ────────────────────────────────────────────
    let worker = select_worker_for_chain(state, &req)?;

    // ── Announce busy state so the dispatcher biases away ──────────
    // RAII guards: drop unconditionally on the way out (success or error)
    // so select_worker's "Tier 1 idle" / "Tier 2 busy" logic sees this
    // worker correctly after the chain ends.
    let _in_flight_guard = InFlightGuard::increment(worker.clone());
    let _active_gen_guard = ActiveGenerationGuard::set(worker.clone(), &req)
        .map_err(|e| ChainRunError::Internal(e.to_string()))?;

    // ── Run the chain inside spawn_blocking ─────────────────────────
    let config_snapshot = state.config.read().await.clone();
    let worker_task = worker.clone();
    let req_task = req.clone();
    let progress_cb_task = progress_cb;

    let join_result = tokio::task::spawn_blocking(move || -> ChainPooledOutcome {
        let model_name = req_task.model.clone();
        let mut progress_cb = progress_cb_task;
        gpu_worker::run_chain_blocking(
            &worker_task,
            &model_name,
            &config_snapshot,
            move |engine| -> Result<mold_inference::ltx2::ChainOutcome, ClosureError> {
                let renderer = engine.as_chain_renderer().ok_or_else(|| {
                    ClosureError::Unsupported(format!(
                        "model '{}' does not support chained video generation",
                        req_task.model
                    ))
                })?;
                let mut orch = Ltx2ChainOrchestrator::new(renderer);
                let run_result = if let Some(cb) = progress_cb.as_deref_mut() {
                    orch.run(&req_task, Some(cb))
                } else {
                    orch.run(&req_task, None)
                };
                run_result.map_err(ClosureError::Orchestrator)
            },
        )
    })
    .await;

    // ── Unwrap the three layers (join, helper-prep, closure) ────────
    let chain_output = match join_result {
        Err(join_err) => {
            return Err(ChainRunError::Join(format!(
                "chain task failed: {join_err}"
            )));
        }
        Ok(Err(prep_err)) => {
            return Err(ChainRunError::CacheMiss(format!("{prep_err:#}")));
        }
        Ok(Ok(Err(ClosureError::Unsupported(msg)))) => {
            return Err(ChainRunError::UnsupportedModel(msg));
        }
        Ok(Ok(Err(ClosureError::Orchestrator(orch_err)))) => {
            return Err(match orch_err {
                ChainOrchestratorError::StageFailed {
                    stage_idx,
                    elapsed_stages,
                    elapsed_ms,
                    inner,
                } => ChainRunError::StageFailed(mold_core::chain::ChainFailure {
                    error: "stage render failed".into(),
                    failed_stage_idx: stage_idx,
                    elapsed_stages,
                    elapsed_ms,
                    stage_error: format!("{inner:#}"),
                }),
                ChainOrchestratorError::Invalid(inner) => {
                    ChainRunError::Inference(format!("{inner:#}"))
                }
            });
        }
        Ok(Ok(Ok(outcome))) => outcome,
    };

    // ── Stitch / encode / save / return ────────────────────────────
    // This block is identical to the tail of run_chain_legacy — the only
    // thing that changed is which code path produced `chain_output`.
    let stage_count = chain_output.stage_count;
    let generation_time_ms = chain_output.generation_time_ms;

    let mut frames = stitch_chain_output(chain_output, &req)
        .map_err(|e| ChainRunError::StitchFailed(e.to_string()))?;
    trim_to_total_frames(&mut frames, req.total_frames);

    if frames.is_empty() {
        return Err(ChainRunError::Encode(
            "chain run emitted zero frames after trim".to_string(),
        ));
    }

    let (bytes, output_format, gif_preview) =
        encode_chain_output(&frames, req.fps, req.output_format)
            .map_err(|e| ChainRunError::Encode(format!("encode chain output: {e:#}")))?;
    let thumbnail = chain_thumbnail(&frames);
    let frame_count = frames.len() as u32;

    // Save to the gallery directory (best-effort, non-blocking).
    let output_dir = {
        let config = state.config.read().await;
        if config.is_output_disabled() {
            None
        } else {
            Some(config.effective_output_dir())
        }
    };
    if let Some(dir) = output_dir {
        let metadata = chain_output_metadata(&req, frame_count);
        let bytes_clone = bytes.clone();
        let gif_clone = gif_preview.clone();
        let model = req.model.clone();
        let db = state.metadata_db.clone();
        tokio::task::spawn_blocking(move || {
            crate::queue::save_video_to_dir(
                &dir,
                &bytes_clone,
                &gif_clone,
                output_format,
                &model,
                &metadata,
                Some(generation_time_ms as i64),
                db.as_ref().as_ref(),
            );
        });
    }

    let response = ChainResponse {
        video: mold_core::VideoData {
            data: bytes,
            format: output_format,
            width: frames[0].width() as u32,
            height: frames[0].height() as u32,
            frames: frame_count,
            fps: req.fps,
            duration_ms: ((frame_count as f64 / req.fps as f64) * 1000.0) as u32,
            has_audio: false,
            audio_sample_rate: 0,
            audio_channels: 0,
            gif_preview,
            thumbnail,
        },
        generation_time_ms,
        stage_count,
    };

    Ok((response, generation_time_ms))
}

/// Internal typed error returned by the `run_chain_blocking` closure in
/// `run_chain_pooled`. Lets the caller distinguish an unsupported-model
/// bailout from an orchestrator failure without string-matching.
enum ClosureError {
    Unsupported(String),
    Orchestrator(ChainOrchestratorError),
}

/// Concrete `ChainPrep` type used by `run_chain_pooled`'s spawn_blocking
/// task. Names the Result-in-Result-in-Result explicitly so the unwrap
/// block above can match exhaustively.
type ChainPooledOutcome = gpu_worker::ChainPrep<mold_inference::ltx2::ChainOutcome, ClosureError>;

/// Pick a `GpuWorker` for this chain. Honours `req.placement` if set,
/// otherwise delegates to `gpu_pool.select_worker` with the same VRAM
/// estimate logic as single-clip dispatch.
fn select_worker_for_chain(
    state: &AppState,
    req: &ChainRequest,
) -> Result<Arc<GpuWorker>, ChainRunError> {
    if let Some(ord) = state
        .gpu_pool
        .resolve_explicit_placement_gpu(req.placement.as_ref())
        .map_err(ChainRunError::UnsupportedModel)?
    {
        return state
            .gpu_pool
            .worker_by_ordinal(ord)
            .ok_or_else(|| ChainRunError::NoWorker(format!("gpu:{ord} is not in the worker pool")));
    }

    let est = crate::queue::estimate_model_vram(&req.model);
    state
        .gpu_pool
        .select_worker(&req.model, est)
        .ok_or_else(|| {
            ChainRunError::NoWorker(format!(
                "no GPU worker available for model '{}'",
                req.model
            ))
        })
}

/// RAII guard that bumps `worker.in_flight` on creation and decrements it on Drop.
struct InFlightGuard {
    worker: Arc<GpuWorker>,
}

impl InFlightGuard {
    fn increment(worker: Arc<GpuWorker>) -> Self {
        worker.in_flight.fetch_add(1, Ordering::SeqCst);
        Self { worker }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.worker.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// RAII guard that sets `worker.active_generation` on creation and clears it on Drop.
struct ActiveGenerationGuard {
    worker: Arc<GpuWorker>,
}

impl ActiveGenerationGuard {
    fn set(worker: Arc<GpuWorker>, req: &ChainRequest) -> anyhow::Result<Self> {
        let first_prompt = req
            .stages
            .first()
            .map(|s| s.prompt.as_str())
            .unwrap_or("");
        let mut slot = worker
            .active_generation
            .write()
            .map_err(|e| anyhow::anyhow!("active_generation lock poisoned: {e}"))?;
        *slot = Some(ActiveGeneration {
            model: req.model.clone(),
            prompt_sha256: format!("{:x}", Sha256::digest(first_prompt.as_bytes())),
            started_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            started_at: Instant::now(),
        });
        Ok(Self { worker })
    }
}

impl Drop for ActiveGenerationGuard {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.worker.active_generation.write() {
            *slot = None;
        }
    }
}
```

- [ ] **Step 5.3: Build and resolve any compile errors**

Run: `cargo check -p mold-ai-server`
Expected: clean.

Common gotchas to watch for:
- `ChainOutcome` — confirm the type is `mold_inference::ltx2::ChainOutcome` by running `grep -n "pub struct ChainOutcome\|pub fn run" crates/mold-inference/src/ltx2/chain.rs`. Adjust the type path if needed.
- `ChainResponse` field names — check `crates/mold-core/src/chain.rs:pub struct ChainResponse` and make sure every field is populated.
- If `sha2` isn't already a direct dep of `mold-server`, `cargo check` will flag it — it's already used by `gpu_worker.rs` so it should be in the crate's `Cargo.toml`.

- [ ] **Step 5.4: Run the full mold-ai-server tests**

Run: `cargo test -p mold-ai-server --lib`
Expected: all existing tests pass, Task 1's serialization test still passes.

- [ ] **Step 5.5: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5.6: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs
git commit -m "$(cat <<'EOF'
fix(server): chain route acquires per-GPU worker lock via run_chain_blocking

Fixes the CUDA SEGV race diagnosed on 2026-04-23: the chain route was
the last caller of the legacy state.model_load_lock subsystem and
hardcoded reclaim_gpu_memory(0), so it could race
cuDevicePrimaryCtxReset_v2 against any single-clip gpu_worker on the
same physical GPU. The new run_chain_pooled path selects a worker via
GpuPool.select_worker (honouring explicit placement), acquires
worker.model_load_lock for the full chain duration via
gpu_worker::run_chain_blocking, and uses worker.model_cache for
take/restore — removing the dual-cache VRAM leak as a bonus.

Legacy state.chain_lock, state.model_cache, and state.model_load_lock
stay reachable only in no-worker (CPU-only) mode via run_chain_legacy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add a test that RAII guards clear `in_flight` and `active_generation` on panic

**Why:** The guards are the only thing stopping a panicked chain from leaving the worker forever-busy (`in_flight > 0` or `active_generation.is_some()` → `select_worker` would bias away from this worker forever). A guard that clears only on success is a latent correctness bug.

**Files:**
- Modify: `crates/mold-server/src/routes_chain.rs` (add `#[cfg(test)] mod tests` if missing, or append inside).

- [ ] **Step 6.1: Verify whether `routes_chain.rs` already has a `#[cfg(test)] mod tests`**

Run: `grep -n "#\[cfg(test)\]" crates/mold-server/src/routes_chain.rs`
If a test module exists, append inside it. If not, add a new `#[cfg(test)] mod tests { ... }` at end of file.

- [ ] **Step 6.2: Write the failing test**

Append inside (or add) the `#[cfg(test)] mod tests` in `crates/mold-server/src/routes_chain.rs`:

```rust
    use super::*;
    use crate::gpu_pool::GpuWorker;
    use mold_inference::device::DiscoveredGpu;
    use mold_inference::shared_pool::SharedPool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, RwLock};

    fn minimal_worker() -> Arc<GpuWorker> {
        let (job_tx, _job_rx) = std::sync::mpsc::sync_channel::<crate::gpu_pool::GpuJob>(2);
        Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal: 0,
                name: "fake".to_string(),
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(crate::model_cache::ModelCache::new(3))),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            degraded_until: RwLock::new(None),
            job_tx,
        })
    }

    fn minimal_chain_request() -> ChainRequest {
        // Use the smallest ChainRequest the struct allows — the guard
        // logic only reads `model` and `stages[0].prompt`.
        ChainRequest {
            model: "fake-model".to_string(),
            stages: vec![mold_core::chain::ChainStage {
                prompt: "hello".to_string(),
                frames: 24,
                transition: Default::default(),
                seed_offset: None,
                source_image: None,
            }],
            total_frames: None,
            fps: 24,
            output_format: mold_core::OutputFormat::Mp4,
            placement: None,
            motion_tail: None,
            base_seed: None,
            negative_prompt: None,
        }
    }

    /// Guards must clear worker state even when the protected scope unwinds.
    #[test]
    fn guards_clear_state_on_panic() {
        let worker = minimal_worker();
        let req = minimal_chain_request();

        let worker_for_catch = worker.clone();
        let result = std::panic::catch_unwind(move || {
            let _in_flight = InFlightGuard::increment(worker_for_catch.clone());
            let _active = ActiveGenerationGuard::set(worker_for_catch.clone(), &req)
                .expect("set active_generation");
            assert_eq!(worker_for_catch.in_flight.load(Ordering::SeqCst), 1);
            assert!(worker_for_catch.active_generation.read().unwrap().is_some());
            panic!("simulated orchestrator failure");
        });

        assert!(result.is_err(), "panic must propagate");
        assert_eq!(
            worker.in_flight.load(Ordering::SeqCst),
            0,
            "InFlightGuard must decrement on panic"
        );
        assert!(
            worker.active_generation.read().unwrap().is_none(),
            "ActiveGenerationGuard must clear on panic"
        );
    }
```

- [ ] **Step 6.3: Run the test**

Run: `cargo test -p mold-ai-server --lib routes_chain::tests::guards_clear_state_on_panic`
Expected: PASS (the guards from Task 5 already implement `Drop` correctly).

If it FAILS, fix the guards in Task 5's code so their `Drop` implementations don't early-return on a poisoned lock — the test body doesn't poison anything, so the only failure mode is implementation bugs in the Drop.

- [ ] **Step 6.4: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6.5: Commit**

```bash
git add crates/mold-server/src/routes_chain.rs
git commit -m "$(cat <<'EOF'
test(server): InFlight/ActiveGeneration guards clear on panic

Regression-protects the chain route's RAII pattern: a panicked chain
must not leave the worker forever-busy. Without the Drop impls
running on unwind, select_worker's in_flight/active_generation bias
would permanently route new work away from the panicked worker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Annotate legacy `reclaim_gpu_memory(0)` sites in `model_manager.rs`

**Why:** The three hardcoded `reclaim_gpu_memory(0)` calls in `model_manager.rs` are now unreachable in multi-worker mode (chain route no longer calls this function). They're reachable only in no-worker mode where device 0 is indeed the only GPU (or no GPU at all → no-op). A comment prevents future maintainers from copying the pattern into new paths that might reintroduce the race.

**Files:**
- Modify: `crates/mold-server/src/model_manager.rs:245, 384, 426`.

- [ ] **Step 7.1: Add the legacy-only comment to each `reclaim_gpu_memory(0)` call site**

Modify `crates/mold-server/src/model_manager.rs`. At each of the three sites (lines 245, 384, 426), replace:

```rust
                mold_inference::reclaim_gpu_memory(0);
```

with:

```rust
                // Legacy no-worker path only: hardcoded ordinal 0 is safe here
                // because `state.model_load_lock` (taken above) is the only
                // lock protecting GPU 0's primary context on this path — the
                // GpuPool path uses `worker.model_load_lock` and
                // `reclaim_gpu_memory(worker.gpu.ordinal)` via `gpu_worker`.
                mold_inference::reclaim_gpu_memory(0);
```

(Match the indentation of the surrounding code at each site — two of the three are at different indentation levels inside different match arms.)

- [ ] **Step 7.2: Verify compilation**

Run: `cargo check -p mold-ai-server`
Expected: clean.

- [ ] **Step 7.3: fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy -p mold-ai-server --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 7.4: Commit**

```bash
git add crates/mold-server/src/model_manager.rs
git commit -m "$(cat <<'EOF'
docs(server): mark reclaim_gpu_memory(0) sites as legacy-path-only

After the chain-route GpuPool migration, these three call sites in
model_manager are only reachable in no-worker (CPU-only) mode. The
hardcoded ordinal is safe there because no other caller touches GPU 0.
Comment prevents future copy-paste into multi-GPU paths, which was the
regression surface for the 2026-04-23 SEGV.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Workspace verification

**Why:** Everything passes in isolation, but we need the full green bar before calling it done. Also checks feature-flag-gated crates (`mold-ai` with all features) and the web bundle build.

**Files:** none modified.

- [ ] **Step 8.1: Full workspace fmt + clippy**

Run: `cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings`
Expected: zero output.

- [ ] **Step 8.2: Full workspace test**

Run: `cargo test --workspace`
Expected: all green. Note — `mold-inference` and `mold-server` have `[lib] test = false` per CLAUDE.md (though render-chain v1 handoff notes this may be stale). If tests in those crates don't run, that's expected; only `--lib` targets we added run via `cargo test -p mold-ai-server --lib`.

- [ ] **Step 8.3: Feature-gated build check (mold-cli with all features)**

Run: `cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4`
Expected: clean.

- [ ] **Step 8.4: Confirm no unintended file changes**

Run: `git status`
Expected: working tree clean after the task-7 commit; no stray edits outside the three files listed in the File Structure section.

- [ ] **Step 8.5: Inspect the commit log**

Run: `git log --oneline origin/main..HEAD`
Expected: 7 new commits (one spec doc, six implementation) plus any prior unrelated commits on the branch. No destructive ops.

- [ ] **Step 8.6: (Optional) Browse the full diff one last time**

Run: `git diff origin/main..HEAD -- crates/mold-server/src/`
Expected: scan for anything surprising. The diff should be bounded to `gpu_worker.rs` (helper + tests), `routes_chain.rs` (split + pooled + guard tests + error variants), and `model_manager.rs` (comments).

---

## Self-review

Against the spec (`docs/superpowers/specs/2026-04-23-cuda-segv-concurrent-reclaim-design.md`):

- **Goal 1 — eliminate the reclaim race:** Tasks 1+2 establish the lock. Task 5 routes the chain through it. Task 1's test proves serialization.
- **Goal 2 — eliminate the dual-cache leak:** Task 5 uses `worker.model_cache` in the pooled path. The legacy cache is never touched in multi-worker mode.
- **Goal 3 — respect single-GPU-per-chain:** Task 5's `select_worker_for_chain` picks exactly one worker per request.
- **Goal 4 — narrow fix:** Touches 3 files, adds 1 helper + 2 error variants + 1 split. No `mold-inference` changes. No new crates.

**Non-goals respected:** No multi-GPU fan-out. No global-serialization policy. `model_manager` preserved. No `cuCtxSynchronize` hope-and-pray. Metal/CPU paths untouched.

**Placeholder scan:** no "TBD", no "implement later", every code step shows complete code, every test step shows assertions.

**Type consistency check:**
- `run_chain_blocking<T, E>` — signature in Task 2.1 matches usage in Task 5.2 (via `ChainPooledOutcome` type alias).
- `ChainPrep<T, E>` — defined in Task 2.1, referenced by alias in Task 5.2.
- `ClosureError` — defined in Task 5.2, pattern-matched exhaustively in the same step's unwrap block.
- `InFlightGuard` / `ActiveGenerationGuard` — defined in Task 5.2, constructed and dropped in Tasks 5.2 and 6.2.
- `ChainRunError::{NoWorker, Join}` — added in Task 3, used in Task 5.

All signatures consistent across tasks.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-cuda-segv-concurrent-reclaim.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
