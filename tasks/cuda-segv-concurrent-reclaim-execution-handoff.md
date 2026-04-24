# CUDA SEGV chain-lock fix — execution handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to drive implementation of the signed-off design and plan.
> Everything above it is reference material the prompt points at.

## Status on entry

- Branch: `feat/multi-prompt-chain-v2-phase3`
- Spec (committed): `docs/superpowers/specs/2026-04-23-cuda-segv-concurrent-reclaim-design.md` at `d768ae2`
- Plan (committed): `docs/superpowers/plans/2026-04-23-cuda-segv-concurrent-reclaim.md` at `f090f99`
- Incident report: `tasks/cuda-segv-concurrent-reclaim-handoff.md`

The spec and the plan have both been reviewed and signed off. Brainstorming
is **done** — do NOT re-invoke `superpowers:brainstorming`, do NOT
re-litigate the approach. The task is now execution.

## The bug in one paragraph

On 2026-04-23 at 06:48 UTC on killswitch (dual RTX 3090), the mold server
SEGV'd inside `libcuda.so.1::cuModuleGetFunction` during model load. Root
cause: the chain route (`routes_chain::run_chain`) is the last caller of
the **legacy single-GPU subsystem** (`state.model_load_lock` +
`state.model_cache` + hardcoded `reclaim_gpu_memory(0)` inside
`model_manager`). Every other path already migrated to the per-worker
`GpuPool` when multi-GPU support landed. That means the chain route's
`cuDevicePrimaryCtxReset_v2` on GPU 0 can race a single-clip
`gpu_worker::process_job`'s reset on the same device, invalidating
modules mid-load. The chain route is the only remaining regression
surface.

## The fix in one paragraph

Migrate the chain route to the `GpuPool` in multi-worker mode. New helper
`gpu_worker::run_chain_blocking` acquires `worker.model_load_lock`, runs
`ensure_model_ready_sync`, does `cache.take()`/`cache.restore()` around a
caller-provided closure. `routes_chain::run_chain` splits into
`run_chain_pooled` (uses the helper) and `run_chain_legacy` (today's code,
preserved for CPU-only dev boxes). Legacy `state.chain_lock`,
`state.model_cache`, `state.model_load_lock` stay — only reachable in
no-worker mode. Also fixes a latent dual-cache VRAM leak for free.

## Signed-off design decisions (do NOT re-litigate)

From the brainstorm:

1. **Approach A** — migrate chain route to `GpuPool`, not B (shared
   per-ordinal lock keeping two caches) or C (full teardown of legacy
   singletons).
2. **Worker selection** — honour `req.placement` first via
   `gpu_pool.resolve_explicit_placement_gpu`, fall back to
   `gpu_pool.select_worker(&model, estimated_vram)`. Same policy as
   single-clip dispatch.
3. **Cross-chain concurrency** — two chains on different workers run in
   parallel (per v1 sign-off #5: "single GPU per chain", not "all chains
   on one GPU"). `state.chain_lock` is NOT needed in multi-worker mode.
4. **No-worker mode** — leave the legacy path untouched. `run_chain`
   branches on `gpu_pool.worker_count()` and the legacy branch is
   today's code verbatim, renamed `run_chain_legacy`. `state.chain_lock`
   stays, reachable only in no-worker mode.
5. **Error shape** — new `ChainRunError::NoWorker` → 503 Service
   Unavailable. New `ChainRunError::Join` → 500 Internal. The typed
   `ChainFailure` body for orchestrator `StageFailed` is preserved via
   a Result-in-Result pattern (`ChainPrep<T, E> = Result<Result<T, E>,
   anyhow::Error>`) so the helper's prep errors don't swallow the
   orchestrator's typed errors.

## Non-goals (spec §Non-goals)

- Multi-GPU stage fan-out for chains (v2).
- Global "one GPU-resident model server-wide" serialization.
- Removing `model_manager` or the CPU fallback path.
- Per-call `cuCtxSynchronize` hopes of making the reset safe under
  concurrency. The driver API contract forbids concurrent live objects
  during reset.
- Metal/CPU path changes. `reclaim_gpu_memory` is a no-op outside CUDA
  (`device.rs:498`), so the bug physically cannot manifest there.

## What the plan expects

Eight tasks, bite-sized, TDD red-first:

1. Scaffold `FakeSlowEngine` + failing lock-serialization test.
2. Implement `run_chain_blocking` — test goes green.
3. Add `ChainRunError::{NoWorker, Join}` + `ApiError` mappings.
4. Split `run_chain` into `run_chain_pooled` (stub) +
   `run_chain_legacy` (rename-only).
5. Fill in `run_chain_pooled` — worker selection, RAII guards,
   spawn_blocking, typed-error unwrap.
6. Regression test — guards clear `in_flight`/`active_generation` on
   panic.
7. Inline comments on `model_manager::reclaim_gpu_memory(0)` sites
   marking them legacy-path-only.
8. Workspace verification (`cargo fmt --check`, `cargo clippy
   --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
   feature-gated `cargo check` on mold-cli).

Each task in the plan has exact code, exact commands, exact expected
output, and a commit step.

## Gotchas scouted during planning

- **`mold-inference` / `mold-server` `[lib] test = false` note in
  CLAUDE.md is stale.** The render-chain v1 handoff confirmed tests
  run normally (586 tests in `mold-inference`, plenty in
  `mold-server`). Run `cargo test -p mold-ai-server --lib` directly.
  Don't waste time temporarily clearing the flag.
- **`reclaim_gpu_memory` is a no-op on non-CUDA builds.** Test machine
  can be a Mac — that's fine. The regression test uses a sleep in
  `FakeSlowEngine::load()` to widen the critical-section window; it
  does NOT require CUDA.
- **`debug_assert_ordinal_matches_thread`** panics in debug builds if
  a thread calls `reclaim_gpu_memory(N)` without first binding via
  `init_thread_gpu_ordinal(N)`. The new helper uses a
  `ThreadGpuGuard` pattern (matching `routes.rs:739-748`) to satisfy
  this. Don't skip the guard.
- **`engine.as_chain_renderer()` returns `Option<&mut dyn
  ChainStageRenderer>`.** Only LTX-2 distilled overrides the default
  `None`. An unsupported model should surface as
  `ChainRunError::UnsupportedModel` (422), not
  `NoWorker`/`Internal`.
- **`cache.take()` returning `None` after `ensure_model_ready` is a
  cache race.** Maps to `ChainRunError::CacheMiss` (500), same as
  today's legacy path. The plan's `run_chain_blocking` already handles
  this.
- **Chain runs in `tokio::spawn_blocking`**, not on the worker's
  dedicated GPU thread. The worker thread processes single-clip jobs
  via its `job_tx`; chain runs on the tokio blocking pool. The
  per-worker mutex is what serializes them, not thread identity.
  Tokio's default blocking pool is 512 threads — no exhaustion risk.
- **The chain holds `worker.model_load_lock` for 10+ minutes**
  (entire chain duration). Admin `/models/load`, `/models/unload`,
  `/api/upscale` pinned to the busy worker will block until chain
  completes. This matches today's single-clip semantics and is the
  correct behavior — a lock held through slow work is not a bug.

## Convention reminders

- **TDD red-first.** Each task starts with a failing test / assertion
  (Task 1 doesn't compile; Task 2 turns it green). The plan spells
  this out step-by-step.
- **Commit after every task.** Commit messages follow the pattern
  used on this branch (`feat(server):`, `test(server):`,
  `docs(server):`, `refactor(server):`). Plan includes drafted
  commit messages for every task.
- **Verification between tasks.** Run at minimum `cargo test -p
  mold-ai-server --lib` after any task that touches server code. The
  plan's Task 8 runs the full workspace sweep as a final gate.
- **No mid-plan push.** Accumulate commits locally. User opens the PR
  at end.
- **Don't touch `mold-inference`, `mold-core`, `mold-cli`, `mold-tui`,
  `web/`.** The fix is entirely within `mold-server`.

## Verification commands (between tasks and at end)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-server --lib                # fastest
cargo test --workspace                            # full sweep (Task 8)
cargo check -p mold-ai --features preview,discord,expand,tui,webp,mp4
```

---

# Prompt (paste this into a fresh session)

I'm taking over execution of a signed-off fix for a CUDA SEGV race in the
mold repo (`/Users/jeffreydilley/github/mold`, branch
`feat/multi-prompt-chain-v2-phase3`). Brainstorm and plan are done —
your job is to execute the plan task-by-task.

## Read first, in this order

1. `CLAUDE.md` at repo root (project conventions, build commands,
   feature flags).
2. `tasks/cuda-segv-concurrent-reclaim-execution-handoff.md` — this
   handoff document. **Read end-to-end before touching code.** It has
   signed-off design decisions, non-goals, and gotchas scouted during
   planning. Don't re-litigate any of them.
3. `docs/superpowers/specs/2026-04-23-cuda-segv-concurrent-reclaim-design.md`
   — the architectural spec.
4. `docs/superpowers/plans/2026-04-23-cuda-segv-concurrent-reclaim.md`
   — the 8-task implementation plan with exact code, commands, and
   commit messages for every step.
5. `tasks/cuda-segv-concurrent-reclaim-handoff.md` — the original
   incident report (symptoms, timeline, coredump, diagnosis). Useful
   context; not a source of requirements.

## What you're doing

Execute the 8-task plan using
**`superpowers:subagent-driven-development`** — fresh subagent per task,
two-stage review between tasks.

- Dispatch one subagent per task with the task's entire content from the
  plan as the subagent's prompt (plus the "read first" files above).
- Each subagent owns one task end-to-end: failing test → implementation
  → passing test → fmt → clippy → commit.
- Between tasks, you (the driver) read the diff, run
  `cargo test -p mold-ai-server --lib` yourself to verify, and only then
  dispatch the next task's subagent.
- If a subagent's work fails the review, loop back to it with the
  specific feedback. Don't accept work that doesn't match the plan.

## Guardrails

- **Do NOT re-plan.** The plan is signed off. If you find a surprise
  that invalidates an assumption, stop and ask the user — don't
  paper over it or rewrite the plan.
- **Do NOT push.** User opens the PR at the end.
- **Do NOT skip TDD.** Each task's failing test must land (or fail to
  compile, for Task 1) before the implementation commit. The plan
  enforces this with explicit red/green steps.
- **Do NOT touch crates other than `mold-server`.** The fix is bounded
  to three files: `crates/mold-server/src/gpu_worker.rs`,
  `crates/mold-server/src/routes_chain.rs`,
  `crates/mold-server/src/model_manager.rs`.
- **Do NOT remove `state.chain_lock`, `state.model_cache`, or
  `state.model_load_lock`.** They're preserved for the no-worker CPU
  fallback path. Removing them is out of scope (spec non-goals).
- **Do NOT introduce CUDA-specific test infrastructure.** The
  regression test uses `FakeSlowEngine` (weight-free) with
  `std::thread::sleep` to widen the critical-section window — no
  CUDA feature required.

## Start here

1. Run `git status && git log --oneline -10` to confirm the spec and
   plan commits are on the tree (expect to see `d768ae2` and `f090f99`
   near HEAD).
2. Read the 5 files listed above in order.
3. Dispatch the Task 1 subagent per `superpowers:subagent-driven-development`.
4. Review, verify, commit check, proceed to Task 2.
5. Repeat through Task 8.

When Task 8 passes, summarise the stack of commits (expect 7 new:
Task 1 test, Task 2 impl, Task 3 error variants, Task 4 split, Task 5
pooled impl, Task 6 panic test, Task 7 comments; Task 8 is
verification-only, no commit) and stop. User will open the PR.
