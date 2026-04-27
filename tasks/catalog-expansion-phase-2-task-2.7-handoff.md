# Phase-2 task 2.7 — Server companion auto-pull (kickoff handoff)

> Paste the prompt at the bottom of this file into a fresh Claude Code
> session to start task 2.7. Everything above is reference material for
> the author / for skimming.

## Where phase 2 stands on entry

Branch `feat/catalog-expansion`. The six phase-2 commits are
**local-only** — phase 2 lands as one push at 2.10:

| Commit | Origin? | Scope |
|---|---|---|
| `14c6061` | local | feat(inference): single-file factory routing + load() (phase 2.6) |
| `c072967` | yes | docs(tasks): catalog-expansion phase-2 task 2.6 kickoff handoff |
| `a579a0c` | local | feat(inference): SDXL single-file engine constructor (phase 2.5) |
| `9e00bd3` | yes | docs(tasks): catalog-expansion phase-2 task 2.5 kickoff handoff |
| `f87687b` | local | feat(inference): SD1.5 single-file engine constructor (phase 2.4) |
| `e50128f` | yes | docs(tasks): catalog-expansion phase-2 task 2.4 kickoff handoff |
| `a6991b4` | local | feat(inference): single-file checkpoint dispatcher (phase 2.3) |
| `e7d4f4a` | yes | docs(tasks): catalog-expansion phase-2 task 2.3 kickoff handoff |
| `4970a92` | yes | docs(tasks): SD1.5 + SDXL tensor-prefix audit findings (phase 2.2) |
| `cb13e06` | yes | feat(inference): sd_singlefile_inspect dev-bin for tensor-prefix audit |

`tasks/catalog-expansion-phase-2-handoff.md` is your task-list source of
truth. The full single-file ingest stack — dispatcher → SD1.5 keys →
SDXL keys → engine constructors → factory routing → `SingleFileBackend`
— is shipped end-to-end inside `mold-inference`. 2.7 wires it up at the
**server** end: companion auto-pull on `POST /api/catalog/:id/download`.

### Done

- 2.1 pre-flight, 2.2 tensor audit, 2.3 dispatcher, 2.4 SD1.5
  constructor, 2.5 SDXL constructor (incl. CLIP-G fused QKV), 2.6 factory
  routing + `is_turbo` threading + `SingleFileBackend` (load-bearing
  diffusers→A1111 projection, including the row-wise slice path for
  CLIP-G `attn.in_proj_*` → `self_attn.{q,k,v}_proj.*`).

### Carried forward from 2.6 — read this carefully

**There's a candle-fork-accessor blocker on actual model
materialisation.** 2.6's `load_components_single_file` (in both
`sd15/pipeline.rs` and `sdxl/pipeline.rs`) prepares the rename surface
and constructs the `SingleFileBackend`, then bails with a sentinel
error:

```
single-file SD1.5 load is staged behind a candle-transformers-mold fork bump:
StableDiffusionConfig.unet/.autoencoder are private in candle-mold 0.9.10
with no accessor, so UNet2DConditionModel::new / AutoEncoderKL::new can't
be reached with a custom VarBuilder. The SingleFileBackend (built and
validated above) is ready; follow-up commit will publish
candle-transformers-mold 0.9.11 with pub accessors and wire
VarBuilder::from_backend(SingleFileBackend) into the three candle
constructors. Until then: pull diffusers-layout shards or wait for the
bump.
```

This is **explicitly sanctioned by the 2.6 handoff** — "at minimum with
a stub that returns a meaningful error so the failure mode is 'your
single-file path isn't materialising weights yet' rather than
'falls through to a diffusers loader that crashes on a missing
config.json'."

**The fork bump is not 2.7's job.** 2.7 only deals with companion
ordering; the existing 2.10 UAT will surface the candle-mold accessor
need as a "follow-up commit" item if the bump hasn't landed yet. Two
viable paths:

- **Option A** (aggressive): 2.7's commit also includes the candle-mold
  fork bump (publish `candle-transformers-mold` 0.9.11 with pub
  accessors), then a 30-line wiring change in
  `load_components_single_file` to drop in
  `VarBuilder::from_backend(SingleFileBackend)`. This unblocks 2.10 UAT.
- **Option B** (cautious): 2.7 ships server companion auto-pull only;
  the candle-mold bump is a 2.7.5 commit before 2.10. Keeps blast radius
  small.

The session driving 2.7 should brainstorm with the user which to take.
Option A is faster to 2.10; Option B is safer if publishing
`candle-transformers-mold` to crates.io has friction.

### Not yet done

- 2.7 — **this handoff's task** (companion auto-pull on the server).
- 2.7.5 (provisional) — candle-mold accessor bump if 2.7 doesn't include it.
- 2.8 CLI integration — `mold pull cv:<id>` recipe path with companion auto-pull.
- 2.9 web gate flip — `cat.canDownload(entry)` from `engine_phase === 1` to `<= 2`.
- 2.10 UAT — full <gpu-host> run (Pony / Juggernaut XL / DreamShaper 8).

## What 2.7 produces

Per the canonical phase-2 spec
(`tasks/catalog-expansion-phase-2-handoff.md` § "Task list" 2.7 row):

> Companion auto-pull on `POST /api/catalog/:id/download` (TDD) — drop the
> engine_phase gate from `>= 2` to `>= 3`; for entries with `companions:
> ["clip-l", "sdxl-vae", ...]`, enqueue each missing companion **before**
> the entry itself; surface as `Vec<job_id>` in response. Companion
> presence check: look on disk, not just in DB, since phase 1 may have
> left half-pulled state.

Three concrete deliverables:

### A. Drop the engine_phase gate

`crates/mold-server/src/catalog_api.rs::post_catalog_download` currently
returns 409 for `engine_phase >= 2`. Drop to `>= 3` so SD1.5 + SDXL
(phase 2) entries become downloadable. **Don't drop further** — phases
3+ (FLUX single-file etc.) don't have engine support yet.

### B. Companion enqueue ordering

For each entry with non-empty `companions`, the server must:
1. Resolve each companion through `mold_catalog::companions::resolve()`
   (canonical companion registry — already exists).
2. Check **on-disk presence** — phase 1 might have left half-pulled
   marker files (`.pulling`) or partially-downloaded shards. Don't trust
   the DB alone.
3. Enqueue each missing companion via
   `crates/mold-server/src/downloads.rs::DownloadQueue::enqueue`
   **before** the entry itself.
4. Return `Vec<job_id>` (one per enqueued companion + one for the entry)
   so the client can poll all of them.

Companion enqueue races: `DownloadQueue::enqueue` already idempotent —
re-enqueuing a job that's mid-flight returns the existing job id. Verify
this assumption in the test suite before depending on it.

### C. Response schema bump

The `POST /api/catalog/:id/download` response today is `{ job_id:
String }`. After 2.7 it must surface every queued job:

```json
{
  "primary_job_id": "uuid-of-entry-job",
  "companion_jobs": [
    { "name": "clip-l", "job_id": "uuid-1" },
    { "name": "sdxl-vae", "job_id": "uuid-2" }
  ]
}
```

This is a **breaking** response shape — the web client's
`useCatalog.ts::download()` consumer must update at the same time
(could be 2.9 or here in 2.7). Confirm the update site before changing
the response shape.

### TDD shape

Three rounds, mirroring 2.4/2.5/2.6:

#### Round 1 — gate drop unit tests

`post_catalog_download_returns_402_for_engine_phase_3` — phase 3 still
gated. `post_catalog_download_accepts_engine_phase_2` — phase 2 (SD15 +
SDXL) now flows. Use `axum::Router::test`-style harness if the existing
catalog_api tests have one; otherwise unit-test the gate predicate
directly.

#### Round 2 — companion ordering tests

`download_with_companions_enqueues_companions_first` — assert the order
of `DownloadQueue::enqueue` calls (companions, then entry).
`download_with_already_present_companion_skips_it` — on-disk presence
check fires.
`download_with_partial_companion_pulls_it` — `.pulling` marker exists
but file is incomplete; must re-enqueue.

#### Round 3 — response schema

`download_response_surfaces_all_job_ids` — multi-job array in the JSON
body.

## Out of scope for 2.7

- **CLI plumbing** — `mold pull cv:<id>` recipe path is 2.8.
- **Web gate flip** — `cat.canDownload(entry)` predicate is 2.9.
- **End-to-end UAT** — real Civitai pulls + generations are 2.10.
- **The candle-mold accessor bump** — unless explicitly chosen as part
  of 2.7 (Option A above).

## Working conventions to preserve

- TDD — failing tests first per round.
- One scope per commit. 2.7 is one commit:
  `feat(server): catalog companion auto-pull on download (phase 2.7)`
  (plus optional second commit `chore(deps): bump
  candle-transformers-mold to 0.9.11 with SD config accessors` if Option
  A).
- **Phase 2 lands as one push when 2.10 is gate-green.** All phase-2
  commits stay local.
- `superpowers:subagent-driven-development` is appropriate here — the
  task is mechanical (gate flip + queue ordering).
- `superpowers:verification-before-completion` before declaring done.
- **Pre-grep cross-crate before subagent dispatch.**
  `grep -rn 'engine_phase\|companions' crates/mold-server/ crates/mold-catalog/`
  to enumerate the gate sites.

## Verification commands

```bash
cd /Users/jeffreydilley/github/mold

# Pre-flight
git status                                      # clean
git log --oneline origin/main..HEAD | head -10  # phase-2 commit chain so far

# After implementation
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-server --lib
cargo test --workspace                          # retry once on TUI flake before blaming your changes
```

The TUI theme test flake
(`theme_save_then_load_round_trip_preserves_preset`) is documented in
user memory — retry once before blaming your changes.

## Reference reading

- `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
- `tasks/catalog-expansion-phase-2-task-2.6-handoff.md` — 2.6 brief, in
  particular the `load_components_single_file` candle-fork-accessor
  blocker section (search for "candle-transformers-mold fork bump").
- `crates/mold-inference/src/loader/single_file_backend.rs` — the 2.6
  `SingleFileBackend` that 2.7 doesn't touch but the candle-fork follow-up
  will.
- `crates/mold-inference/src/sd15/pipeline.rs::load_components_single_file`
  and `crates/mold-inference/src/sdxl/pipeline.rs::load_components_single_file`
  — the sentinel-error stubs that pin where the candle-fork wiring lands.
- `crates/mold-server/src/catalog_api.rs::post_catalog_download` — the
  primary edit site.
- `crates/mold-server/src/downloads.rs::DownloadQueue::enqueue` — the
  companion-enqueue surface.
- `crates/mold-catalog/src/companions.rs` — the canonical companion
  registry.

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I'm starting **task 2.7** of the mold catalog-expansion phase 2 — server
companion auto-pull on `POST /api/catalog/:id/download`. Tasks 2.1–2.6
are done. **2.6 (commit `14c6061`, local-only) shipped factory routing
plus the load-bearing `SingleFileBackend`, but `load()` materialisation
is staged behind a `candle-transformers-mold` fork bump** — see the
handoff for context. 2.7 itself doesn't touch that; it's purely server
companion ordering.

## Read first, in this order

1. `~/.claude-personal/CLAUDE.md` and
   `/Users/jeffreydilley/github/mold/CLAUDE.md` — coding conventions.
2. `tasks/catalog-expansion-phase-2-handoff.md` — overall phase-2 brief.
   **§ "Task list" 2.7 row** is the canonical spec.
3. `tasks/catalog-expansion-phase-2-task-2.7-handoff.md` — this file.
   Spells out the gate drop, companion-ordering rules, and the
   response-schema breakage to coordinate with the web client.
4. `tasks/catalog-expansion-phase-2-task-2.6-handoff.md` — 2.6 context,
   in particular the candle-fork-accessor blocker that 2.7 may want to
   bundle (Option A).
5. `crates/mold-server/src/catalog_api.rs::post_catalog_download` — the
   primary edit site. Grep for `engine_phase` to find the current 409
   gate.
6. `crates/mold-server/src/downloads.rs` — `DownloadQueue::enqueue`
   surface. Confirm idempotency assumption before depending on it.
7. `crates/mold-catalog/src/companions.rs` — companion registry. The
   `resolve()` function is what 2.7 uses to look up each companion's
   manifest.
8. `web/src/composables/useCatalog.ts::download` — the consumer of the
   response shape that 2.7 changes. Coordinate the update.

## What you're doing

Implement task 2.7 per this handoff. Three deliverables:

1. **Drop the `engine_phase >= 2` gate to `>= 3`** in
   `post_catalog_download`.
2. **Wire companion auto-pull** with on-disk presence check — for each
   missing companion, enqueue it via `DownloadQueue::enqueue` **before**
   the primary entry job.
3. **Bump the response schema** to `{ primary_job_id, companion_jobs:
   [{name, job_id}] }`. Coordinate the web client update.

Optional fourth deliverable (Option A): bundle the candle-mold accessor
bump (publish `candle-transformers-mold` 0.9.11 with pub
`unet()`/`autoencoder()` accessors on `StableDiffusionConfig`) and wire
`VarBuilder::from_backend(SingleFileBackend)` into the
`load_components_single_file` stubs in `sd15/pipeline.rs` and
`sdxl/pipeline.rs`. Brainstorm Option A vs B with the user before
starting.

TDD: three rounds, failing tests first per round, mirroring 2.4–2.6.

## How to work

1. Pre-flight: confirm `git status` clean and `cargo test --workspace`
   green (~661+ inference tests pass) before touching code.
2. **Pre-grep cross-crate.**
   `grep -rn 'engine_phase\|companions' crates/mold-server/ crates/mold-catalog/`
   to enumerate gate sites and companion consumers.
3. Use `superpowers:test-driven-development`.

## Verification gate before committing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mold-ai-server --lib
cargo test --workspace                  # retry once on TUI flake (`theme_save_then_load_round_trip_preserves_preset`)
```

## Commit shape

Single commit when gate-green:

```
feat(server): catalog companion auto-pull on download (phase 2.7)

Drops the `engine_phase >= 2` gate to `>= 3` in
`post_catalog_download` so SD1.5 + SDXL entries (phase 2) become
downloadable. For entries with non-empty `companions`, the handler now
resolves each companion via `mold_catalog::companions::resolve`, checks
on-disk presence (phase 1 may have left half-pulled `.pulling` markers),
and enqueues missing companions through `DownloadQueue::enqueue` before
enqueuing the primary entry. The response shape grows from `{ job_id }`
to `{ primary_job_id, companion_jobs: [{name, job_id}] }`; the web
client's `useCatalog.ts::download()` consumer is updated in lockstep.

[If Option A: also bumps candle-transformers-mold to 0.9.11 to expose
`StableDiffusionConfig.unet/.autoencoder` accessors, and wires
`VarBuilder::from_backend(SingleFileBackend)` into the
`load_components_single_file` stubs that 2.6 left behind. Single-file
SD1.5 / SDXL `load()` now produces a real candle model end-to-end.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Do not push.** Phase 2 lands as one push at 2.10. The 2.3-2.7 commits
all stay local.

## If you hit a surprise

- If `DownloadQueue::enqueue` isn't idempotent, the companion-already-in-flight
  test will fail. Surface the gap before working around it — the queue
  layer should own that contract, not the catalog handler.
- If the on-disk presence check is harder than expected (e.g.
  multi-shard companions where partial files are valid mid-pull),
  document and ask. Don't paper over with a "good enough" check.
- If the candle-mold fork bump path (Option A) hits crates.io
  publishing friction, fall back to Option B and document the gap.

When 2.7 is gate-green and the test rounds pass, write
`tasks/catalog-expansion-phase-2-task-2.8-handoff.md` (template: this
file) and stop. Do not start 2.8 in the same session.
