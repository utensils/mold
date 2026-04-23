# CUDA SEGV — concurrent reclaim_gpu_memory race (handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to drive the investigation and fix. Everything above it is
> reference material the prompt points at.

## Incident summary

**When:** 2026-04-23 06:48:56 UTC on `killswitch@192.168.1.67` (dual RTX 3090).

**Binary:** `target/release/mold` built from `feat/multi-prompt-chain-v2-phase3`
at commit `855b228` (`feat(chain): per-stage starting images in script mode`).
The feature commit is not implicated — the crash path exists on `main` and on
every recent commit.

**Signal:** `SIGSEGV` in thread `2161532`. Crashing frame:

```
#0  libcuda.so.1 + 0x1199370
#1  libcuda.so.1 + 0x32027a
#2  cuModuleGetFunction  (libcuda.so.1 + 0x31152a)
#3  mold + 0x1ef9387        ← Rust → CUDA module lookup
#4  mold + 0x1a3c5ce
#5  mold + 0x1a8a389
#6  mold + 0x1b02e28
#7  mold + 0x1cb4b85
#8  mold + 0x2aaa75e
#9  mold + 0x2840dab        ← spawn_blocking closure
#10 mold + 0x27b7955
#11 mold + 0x277d127
```

**Coredump:**
`/var/lib/systemd/coredump/core.mold.1000.954c331646f548bf9c12fd7673ce2a8c.2160328.1776926936000000.zst`
(367 MB, stripped release binary — addr2line will need a debug rebuild at
the same commit to symbolicate).

**Server log (tail):**
`~/github/mold/serve-20260422-234323.log` on killswitch.

## Timeline (from `serve-20260422-234323.log`)

```
06:43:24  multi-GPU mode starts — 2× RTX 3090 ready
06:43:24  starting mold server 0.0.0.0:7680 (v0.9.0, 855b228)
06:47:44  POST /api/generate/chain/stream  ltx-2.3-22b-distilled:fp8  4 stages 1024×1024
          → mold_server::model_manager: "loading model..."
06:47:48  mold_inference::device: "CUDA primary context reset for device 0"
06:48:03  POST /api/generate/chain/stream  (same model, queued)
06:48:04  POST /api/generate/chain/stream  (same model, queued)
06:48:54  POST /api/generate/stream        qwen-image-2512:q8
          mold_server::gpu_worker: dispatched job, gpu=0
06:48:54  mold_inference::device: "CUDA primary context reset for device 0"  ← second reset of the same GPU in 66s
          mold_server::gpu_worker: loading model... gpu=0 qwen-image-2512:q8
06:48:55  POST /api/generate/stream        qwen-image-2512:q8
          mold_server::gpu_worker: dispatched job, gpu=1
06:48:55  mold_inference::device: "CUDA primary context reset for device 1"
          mold_server::gpu_worker: loading model... gpu=1 qwen-image-2512:q8
06:48:55  POST /api/generate/stream        qwen-image-2512:q8 (queued)
<<< SEGV, no further log lines >>>
```

## Diagnosis

Two independent schedulers own the GPUs and do not coordinate their
`reclaim_gpu_memory()` / `cuDevicePrimaryCtxReset_v2` calls:

### Path A — chain requests

`crates/mold-server/src/routes_chain.rs:345`
```rust
model_manager::ensure_model_ready(state, &req.model, None)
```

`crates/mold-server/src/model_manager.rs`:
- Line 208: `let _guard = state.model_load_lock.lock().await;`  ← **global** singleton
- Lines 245 / 384 / 426: `mold_inference::reclaim_gpu_memory(0);`  ← **hard-coded GPU 0**

No awareness of `state.gpu_pool` or its per-worker locks.

### Path B — single-clip generate/stream requests

`crates/mold-server/src/gpu_worker.rs`:
- Line 71: `let _load_lock = worker.model_load_lock.lock().unwrap();`  ← **per-GPU** mutex (Mutex on `GpuWorker`)
- Line 309 / 338 / 390: `device::reclaim_gpu_memory(worker.gpu.ordinal);`  ← ordinal-parameterised

`crates/mold-inference/src/device.rs:486`:
```rust
let result = unsafe { sys::cuDevicePrimaryCtxReset_v2(cu_device) };
```
Comment at line 460: **"Must only be called when no CUDA objects (tensors,
devices, engines) exist on this device."** That precondition is violated by
the cross-scheduler race.

### Sequence that SEGV'd

1. Path A took `state.model_load_lock` at 06:47:44, started LTX-2 22B
   load on device 0 via `create_engine_with_pool`. This is module-loading
   heavy — cuBLAS / cuDNN / candle kernels registered into the primary
   context on device 0. It did reset device 0 once at 06:47:48 before
   the load proper.
2. Path B's `gpu-worker-0` thread, holding `worker.model_load_lock` (a
   **different** mutex), took the qwen-image-2512 job at 06:48:54,
   saw `unload_active` and called `device::reclaim_gpu_memory(0)` —
   which ran `cuDevicePrimaryCtxReset_v2` on device 0 **while Path A's
   spawn_blocking worker was still mid-module-load on the same context**.
3. The reset invalidated every module handle on device 0.
4. Path B then called `engine.load()` → `cuModuleLoad` → (some frames
   later) `cuModuleGetFunction` against a handle the driver had torn
   down → SEGV.

Path B's ordinal-1 reset 600 ms later is a coincidence — the ordinal-0
race alone is sufficient. Both GPUs *and* the chain path were in play,
so the real window is actually wider.

### Why this hasn't been seen constantly

- Chain requests are rare vs single-clip.
- LTX-2 22B load is slow (tens of seconds) so the window is wide when
  they collide.
- Users typically serialise their own requests.

Today it hit because the user submitted 3 chain requests, then 3
single-clip Qwen-Image requests, all within ~70 s.

## Where to fix

The correct lock surface is **"one lock per physical GPU ordinal that
any scheduler must hold before touching that GPU's primary context"**.
Today:

| Lock | Owner | Scope |
|------|-------|-------|
| `state.model_load_lock` | `AppState` | global singleton, used by `model_manager` |
| `worker.model_load_lock` (per `GpuWorker`) | `gpu_pool` | per GPU, used only by `gpu_worker` |

Both need to converge on a single "GPU 0 is mine" / "GPU 1 is mine"
mutex that both schedulers acquire. Shape options:

1. **`state.gpu_pool.workers[n].model_load_lock` becomes the only lock
   per GPU.** Delete `state.model_load_lock`. Rewrite `model_manager`
   to take a GPU ordinal (either from the chain route — single-GPU per
   chain is a signed-off v1 constraint per `tasks/render-chain-v1-handoff.md`
   decision #5 — or by choosing an idle worker) and acquire that
   worker's lock. This also fixes the hard-coded `reclaim_gpu_memory(0)`
   call sites in `model_manager`.
2. **Introduce `state.gpu_locks: Vec<Arc<Mutex<()>>>` keyed by ordinal.**
   `GpuWorker` consults the same vector. Smaller surgery; keeps
   model_manager's existing structure.

Prefer (1) — it eliminates the duplicate "which GPU am I on?" question
and surfaces the decision to the chain route, where ordinal selection
has to happen anyway for VRAM accounting.

Either way, **every `reclaim_gpu_memory(ordinal)` call site must be
inside a critical section protected by the per-ordinal lock.** Grep
shows these today (before any fix):

```
crates/mold-server/src/gpu_worker.rs:309   reclaim_gpu_memory(worker.gpu.ordinal);   ← OK under worker.model_load_lock
crates/mold-server/src/gpu_worker.rs:338   reclaim_gpu_memory(worker.gpu.ordinal);   ← OK under worker.model_load_lock
crates/mold-server/src/gpu_worker.rs:390   reclaim_gpu_memory(worker.gpu.ordinal);   ← OK under worker.model_load_lock
crates/mold-server/src/model_manager.rs:245 reclaim_gpu_memory(0);                    ← RACE
crates/mold-server/src/model_manager.rs:384 reclaim_gpu_memory(0);                    ← RACE
crates/mold-server/src/model_manager.rs:426 reclaim_gpu_memory(0);                    ← RACE
```

## Non-goals for this fix

- **Don't** introduce a new global "only one GPU-resident model across
  the whole server" constraint. The v1 design *is* multi-GPU and that's
  load-bearing for throughput.
- **Don't** rip out `model_manager` entirely. It also owns model-pull
  side-effects and the OpenAPI schema. Just narrow its GPU-touch
  surface to go through the per-ordinal lock.
- **Don't** switch to per-call `cuCtxSynchronize` hopes. The driver
  comment at `device.rs:460` already says "no live CUDA objects" —
  the fix is serialising the reset, not trying to make it safe under
  concurrency.
- **Don't** blame the feature branch. `git log --oneline main..HEAD`
  on this branch and on recent merges shows zero touches to `device.rs`,
  `model_manager.rs`, `gpu_worker.rs`, or `gpu_pool.rs`.

## Reproduction plan

The window is racy by nature; don't expect a deterministic repro on
the first try. In order of likelihood:

1. **Concurrent-load regression test (cheapest)** — with CUDA feature
   off, mock a fake "slow load" engine that sleeps for 2 s inside
   `load()`. Drive two threads: one goes through `model_manager`, one
   through `gpu_worker`, both targeting ordinal 0. Assert the second
   caller blocks until the first completes (i.e. a single per-ordinal
   lock is held across both paths).
2. **Bench on killswitch** — after fix, replay today's pattern: 3
   chain submits to LTX-2 22B followed by 3 single-clip Qwen-Image
   submits within ~70 s, watch `coredumpctl list --since` for fresh
   entries. Pre-fix reproduces in the single-digit-trial range.
3. **`CUDA_LAUNCH_BLOCKING=1`** — set on the server env; forces
   synchronous kernel launches and tends to shrink races into
   determinism. Useful for the bench test.

## Artefacts to preserve

- Coredump (above path on killswitch).
- `serve-20260422-234323.log` (last pre-crash server log).
- This handoff file.

`coredumpctl dump 2160328 > /tmp/mold-2160328.core` if you want a
copy off-box. Addr2line needs a matching debug binary:
`CUDA_COMPUTE_CAP=86 cargo build -p mold-ai --features cuda,preview,discord,expand,tui,webp,mp4,metrics`
(drop `--release` so frame pointers + symbols survive).

## What's currently running

After the crash I restarted with:
```
LD_LIBRARY_PATH=/opt/cuda/lib64 \
  MOLD_GALLERY_ALLOW_DELETE=1 \
  RUST_BACKTRACE=full \
  nohup ./target/release/mold serve --bind 0.0.0.0 > serve-<ts>.log 2>&1 &
```
So a Rust-side panic will emit a backtrace next time. A SEGV from
libcuda still won't — that's what the coredump is for.

---

# Prompt (paste this into a fresh session)

You are taking over an investigation on the mold repo
(`/Users/jeffreydilley/github/mold`, feature branch
`feat/multi-prompt-chain-v2-phase3`). A production mold server on
killswitch@192.168.1.67 crashed with SIGSEGV inside
`libcuda.so.1::cuModuleGetFunction` while two independent schedulers
(the chain-route `model_manager` and a per-GPU `gpu_worker`) raced
`cuDevicePrimaryCtxReset_v2` against the same GPU primary context.

The complete incident report, timeline, stack trace, suspected-race
analysis, and fix sketch live in
`tasks/cuda-segv-concurrent-reclaim-handoff.md`. Read that first.

Your job:

1. Reproduce the race with a unit/integration test that does not need
   a live CUDA device. Use a stand-in "slow load" engine so two
   concurrent callers exercise `model_manager::ensure_model_ready`
   and `gpu_worker::ensure_model_ready_sync` against the same GPU
   ordinal. The test must fail on the current tree (two threads enter
   their critical sections simultaneously) and pass after the fix.
2. Unify the per-ordinal GPU lock. Preferred approach (option 1 in
   the handoff): route `model_manager` through the `GpuPool`'s
   per-worker `model_load_lock`. The legacy `state.model_load_lock`
   singleton goes away. `model_manager::reclaim_gpu_memory(0)` hard-
   coded call sites must become `reclaim_gpu_memory(ordinal)` driven
   by the selected worker.
3. Respect the render-chain v1 signed-off constraint: "single GPU per
   chain" (`tasks/render-chain-v1-handoff.md`, decision #5). Chain
   routes pick an ordinal at dispatch time, not per-stage.
4. Do not expand the fix to multi-GPU chain fan-out — that's v2.
5. Verify `cargo fmt --check`, `cargo clippy --workspace
   --all-targets -- -D warnings`, `cargo test --workspace`, and
   `bun run test` in `web/` stay green.

Constraints:

- `mold-inference` and `mold-server` have `[lib] test = false` for
  the main harness — see `CLAUDE.md` "Build & Development Commands".
  For tests in those crates, run `cargo test -p <crate> --lib` after
  temporarily clearing the flag, then restore it before committing.
- TDD: write the failing test first, then the fix.
- Do not touch the user-facing chain / UI / TOML surfaces added on
  this branch. The crash is a pre-existing concurrency bug, not a
  regression from that work.
- No blanket `RwLock → Mutex` conversions or wholesale refactors.
  Narrow change: one lock surface unified across both schedulers.

When done, summarise the fix in a commit message that explains (a)
what the two schedulers were doing, (b) why
`cuDevicePrimaryCtxReset_v2` is unsafe under that race, (c) the
one-lock-per-ordinal resolution, and (d) the regression test.
