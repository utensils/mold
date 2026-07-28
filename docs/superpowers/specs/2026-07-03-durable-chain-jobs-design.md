# Durable chain jobs — design

**Date:** 2026-07-03
**Status:** Draft for review
**Scope:** Chained video generation becomes a persistent, resumable, retakeable
job with per-stage checkpoints, scheduled fairly against other GPU work.

## 1. Problem

Chained video generation today is one giant blocking call:

- `POST /api/generate/chain` checks the engine out via
  `gpu_worker::run_chain_blocking`, which holds the per-worker
  `model_load_lock` for the **entire chain**. Single-clip jobs (web, CLI,
  Discord) queue behind a multi-minute chain.
- Every decoded frame of every stage stays in RAM
  (`ChainRunOutput.stage_frames: Vec<Vec<RgbImage>>`) until a monolithic
  stitch — ~250 MB/stage raw at 1216×704. This is the real ceiling behind
  `MAX_STAGES = 16` and blocks the 5+ minute target.
- Mid-chain failure discards all completed stages (signed-off v1 decision).
  `JobRegistry` is in-memory, so a server restart also loses everything.
- There is no way to regenerate a single bad stage; the only recourse is
  re-running the whole chain.

Goals, in priority order:

1. **Durability** — per-stage artifacts on disk; a failure or restart costs
   at most one stage.
2. **Resume** — interrupted/failed chains continue from the last completed
   stage, deterministically.
3. **Retake** — regenerate stage N (new seed / edited prompt) without paying
   for stages 0..N-1.
4. **Fairness** — chains yield to queued single-clip work at stage
   boundaries.
5. **Flat memory** — RAM high-water mark is one stage regardless of chain
   length.

Non-goals (explicitly deferred):

- Cloud-burst execution (RunPod/Lambda). The job-dir format is designed to be
  location-independent so this is possible later, but no transport is built.
- New model families. The rider refactor (§10) makes the orchestrator
  model-agnostic, but only the LTX-2 family implements it in v1.
- Multi-chain parallelism across GPUs. One chain job runs at a time (existing
  `chain_lock` semantics keep their spirit); parallel chains are a follow-up.

## 2. Design overview

A chain becomes a **job**: a row in `mold.db` plus a self-contained directory
under `MOLD_HOME/jobs/<job_id>/`. A new `ChainJobRunner` drives the stage
loop. After each stage it:

1. encodes the stage's frames to an H.264 segment (boundary handling already
   applied, §5),
2. writes the tail RGB frames (the resume/retake carryover) as PNGs,
3. writes a PCM audio sidecar when the stage produced audio,
4. updates `manifest.toml` and the DB row atomically-enough (§4.3),
5. checks the yield condition (§6) before starting the next stage.

Finalize is a separate, re-runnable step: stream segments through an
incremental encoder into the final MP4, mux AAC, write the gallery entry.
Because finalize consumes only on-disk segments, retake can re-run it after
swapping one segment out.

```
POST /api/chain-jobs ──▶ chain_jobs row (queued) ──▶ ChainJobRunner
                                                        │ per stage:
                                                        │  acquire worker lock
                                                        │  render stage
                                                        │  write artifacts
                                                        │  release lock if contended
                                                        ▼
                                    MOLD_HOME/jobs/<id>/ segments + manifest
                                                        │ all stages done
                                                        ▼
                                                    finalize ──▶ gallery
```

## 3. Job store

### 3.1 DB schema (migration v11)

Two tables in `mold.db`, forward-only migration per the existing
`MIGRATIONS[]` / `SCHEMA_VERSION` pattern in `crates/mold-db/src/migrations.rs`:

```sql
CREATE TABLE chain_jobs (
    id            TEXT PRIMARY KEY,          -- ULID
    state         TEXT NOT NULL,             -- queued|running|interrupted|failed|completed|cancelled
    model         TEXT NOT NULL,
    request_json  TEXT NOT NULL,             -- full normalised ChainRequest
    job_dir       TEXT NOT NULL,             -- absolute path to the job directory
    stage_count   INTEGER NOT NULL,
    current_stage INTEGER NOT NULL DEFAULT 0,
    error         TEXT,                      -- terminal error message, if failed
    created_at    INTEGER NOT NULL,          -- unix ms
    updated_at    INTEGER NOT NULL,
    finalized_at  INTEGER                    -- last successful finalize, unix ms
);

CREATE TABLE chain_job_stages (
    job_id             TEXT NOT NULL REFERENCES chain_jobs(id) ON DELETE CASCADE,
    stage_idx          INTEGER NOT NULL,
    state              TEXT NOT NULL,        -- pending|running|completed|failed
    seed               INTEGER NOT NULL,     -- effective stage seed (base ^ offset)
    frames_emitted     INTEGER,
    generation_time_ms INTEGER,
    segment_rel_path   TEXT,                 -- relative to job_dir
    error              TEXT,
    updated_at         INTEGER NOT NULL,
    PRIMARY KEY (job_id, stage_idx)
);
```

The DB is the **queryable index** (job lists, states, GC scans). The manifest
(§3.2) is the **portable source of truth**: if the two disagree, the manifest
wins and a startup reconcile repairs the DB row — mirroring the existing
gallery `reconcile(output_dir)` philosophy.

`MOLD_DB_DISABLE=1` disables chain jobs entirely (the endpoints return 503
with a clear message). Durability without a DB is not worth a second
bookkeeping path.

### 3.2 Job directory layout

```
MOLD_HOME/jobs/<job_id>/
├── manifest.toml            # self-contained job description + per-stage status
├── stages/
│   ├── 000/
│   │   ├── segment.mp4      # H.264 segment. Amended (2026-07-28, §17): RAW —
│   │   │                    #   every frame the engine emitted, no boundary
│   │   │                    #   trims/blends. Legacy stages (raw_segment=false)
│   │   │                    #   hold the old trimmed, boundary-handled form.
│   │   ├── tail/            # trailing RGB frames, %03d.png — amended (§17):
│   │   │                    #   ALWAYS written when the engine produced a tail
│   │   │                    #   (bit-exact carry for resume/amend). Legacy
│   │   │                    #   stages wrote it only before a Smooth successor.
│   │   ├── boundary-out/    # LEGACY only — trailing fade_len raw frames,
│   │   │                    #   written before a Fade successor (blend source)
│   │   ├── boundary-in/     # LEGACY only — leading fade_len raw (pre-blend)
│   │   │                    #   frames of a Fade stage (splice re-blend, §8.2)
│   │   ├── audio.pcm        # f32-interleaved PCM sidecar (only when audio
│   │   │                    #   enabled). Amended (§17): full untrimmed track
│   │   │                    #   for raw stages.
│   │   └── preview.jpg      # last-frame thumbnail for the composer stage card
│   ├── 001/ …
└── final/
    └── output-<n>.mp4       # finalize results, versioned (§8.3)
```

`manifest.toml` schema `mold.chainjob.v1`:

```toml
schema = "mold.chainjob.v1"
job_id = "01J..."
created_at_unix_ms = 1783200000000

[request]        # the full normalised ChainRequest, TOML-serialised
model = "ltx-2-19b-distilled:fp8"
# ... width/height/fps/seed/steps/guidance/motion_tail_frames/stages ...

[[stage_status]]
idx = 0
state = "completed"
seed = 42
frames_emitted = 97
segment = "stages/000/segment.mp4"   # all paths relative to the job dir
tail_frames = 4
audio = "stages/000/audio.pcm"
```

Everything needed to resume or retake lives inside the directory; no
absolute paths, no DB dependency. That is the entire cloud-burst
contract for now.

### 3.3 Write ordering

Per stage, in order: artifacts → manifest rewrite (write-temp + rename) → DB
update. A crash between manifest and DB leaves the DB stale by one stage;
startup reconcile (§7) trusts the manifest. A crash mid-artifact leaves a
partial `stages/NNN/` with manifest state still `running`/`pending`; resume
deletes and re-renders that stage.

## 4. Stage rendering and carryover

The existing carryover mechanism is kept exactly: a stage's `ChainTail` is
its trailing decoded RGB frames, and the next stage VAE-encodes them fresh
(the causal-slot-semantics rationale in `ltx2/chain.rs` is unchanged). The
only difference is that the tail now round-trips through PNGs on disk:

- **Live path:** the runner passes the in-memory `ChainTail` straight to the
  next stage (no decode from disk) _and_ persists it. Zero added latency.
- **Resume/retake path:** `ChainTail` is reconstructed by loading
  `stages/NNN/tail/*.png`. PNG is lossless, so the reconstructed tail is
  bit-identical to the live one, and the VAE encode of it is deterministic —
  a resumed stage renders exactly what it would have rendered.

Seed determinism already holds: stage seeds are stable-by-design
(`derive_stage_seed`), initial noise is CPU-seeded (`seeded_randn`). The
stage's _effective_ seed is persisted in both manifest and DB so a retake can
change it without touching the base request.

## 5. Boundary handling at write time

Today `StitchPlan::assemble` does all boundary work at the end, in RAM. This
design moves it to stage-write time so final assembly is concatenation:

- **Smooth:** continuation stages drop their leading `motion_tail_frames`
  _before_ encoding their segment. (The orchestrator already knows the
  incoming transition; the trim moves from stitch to segment-encode.)
- **Cut:** segment encoded as-is.
- **Fade:** the incoming stage blends its leading `fade_len` frames against
  the prior stage's stored `boundary-out/` frames, then encodes (its raw
  pre-blend leading frames are persisted to `boundary-in/` first). The prior
  segment must _not_ contain those consumed frames — so a stage followed by a
  Fade trims its trailing `fade_len` frames from its own segment. Because the
  following stage's transition is known from the request up front (not
  discovered later), each stage knows its trailing trim at encode time.
  Retake preserves this: transitions are part of the request, and a retake
  that edits a stage's _transition_ re-renders is out of scope for v1 (seed
  and prompt edits only, §8).
- **Audio:** per-stage PCM sidecars carry the same trims (samples are
  frame-aligned: `samples_per_frame = sample_rate / fps` at constant rate).
  Finalize concatenates PCM and muxes AAC once, via the existing
  `attach_aac_track_to_mp4_bytes` path.

Invariant: **concatenating all segments in order, with no further frame
surgery, yields the final video.** This is the property that makes finalize
re-runnable and retake-splice (§8.2) cheap. A unit test asserts frame-count
equality against the current `StitchPlan::assemble` for mixed-transition
chains (the `mixed_transitions_end_to_end` case: 97+72+97+89 = 355).

> **Amended (2026-07-28, §17):** this section describes the LEGACY artifact
> contract, still honored for stages with `raw_segment = false`. New stages
> are written RAW and the concat invariant no longer holds for them —
> finalize applies the whole boundary plan (§17).

### 5.1 Finalize

Finalize streams: decode each segment (H.264 decode is baseline for LTX-2
builds), feed frames incrementally into one encoder, mux audio. Memory is
bounded by the decoder/encoder window, not chain length. `video_enc` gains an
incremental push-frame API (`Mp4StreamEncoder::push(frame)` / `finish()`);
the existing slice-based `encode_mp4` becomes a thin wrapper over it.

If segment codec parameters are provably identical, a future optimization is
container-level concat without re-encode; v1 does the single re-encode pass
for simplicity and to keep GIF-preview/APNG fallbacks working unchanged.

## 6. Execution model: cooperative yield

A single `ChainJobRunner` task (spawned per job, at most one active job — the
existing `chain_lock` becomes "one chain _runner_ at a time") drives:

```
for stage in remaining_stages:
    run_stage_blocking(worker, model, |engine| render_stage(...))   # lock held per stage
    write_artifacts(stage)
    if state.queue.pending() > 0 or worker.in_flight() > 0:
        yield_point()    # lock already released; let the dispatcher drain small jobs
```

`run_stage_blocking` is `run_chain_blocking` with the lock scope narrowed
from whole-chain to one stage; the `ensure_model_ready_sync` call inside it
already handles "engine was swapped out while we yielded" (parked-reload /
load-from-disk), so re-acquisition needs no new machinery. When nothing else
is queued, the lock is dropped and immediately re-taken with the engine still
GPU-resident — cost is a mutex round-trip, not a model load.

Worker selection (`select_worker_for_chain`), `InFlightGuard`,
`ActiveGenerationGuard`, and the OOM-cooldown logic in `gpu_pool` are reused
per-stage unchanged. The legacy no-worker path (CPU-only CI) drives the same
runner with the legacy engine checkout.

Cancellation: `POST /:id/cancel` sets a flag the runner checks at every stage
boundary and (best-effort) between denoise steps via the existing progress
callback. A cancelled job keeps its completed-stage artifacts and can be
resumed later — cancel is pause-with-intent, not delete. `DELETE /:id`
removes the job and its directory.

## 7. Crash recovery and resume

On server startup:

1. Scan `chain_jobs` for `running` rows → flip to `interrupted`.
2. Reconcile each non-terminal job against its manifest (manifest wins).
3. Do **not** auto-resume (decision: manual resume). The web UI shows
   interrupted jobs with a Resume button.

`POST /api/chain-jobs/:id/resume`:

1. Load manifest; find the first stage not `completed`.
2. Delete any partial artifacts for that stage.
3. Reconstruct `ChainTail` from the previous stage's `tail/` PNGs (or `None`
   / Cut-Fade semantics per the transition, exactly as the orchestrator does
   today).
4. Re-enter the runner loop from that stage.

Resume is idempotent and repeatable — a job that fails at stage 12, resumes,
and fails at stage 15 can resume again from 15.

## 8. Retake (v1 scope)

Retake regenerates one stage of a completed (or partially completed) job.

```
POST /api/chain-jobs/:id/retake
{ "stage_idx": 7, "mode": "cascade" | "splice",
  "seed_offset": 12345,            # optional; new effective seed for the stage
  "prompt": "..." }                # optional; replaces the stage's prompt
```

### 8.1 Cascade (default for Smooth-following stages)

Stage N is re-rendered with the stored carryover from stage N-1; stages
N+1..end are then re-rendered too, because their conditioning input (stage
N's tail) changed. Correct by construction; costs the tail of the chain.

### 8.2 Splice

Only stage N is re-rendered; N+1..end keep their segments. Sound **only**
when stage N+1's transition is `Cut` or `Fade` (no temporal handoff across
that boundary, so the seam is seamless by construction). For fade, stage N's
new `boundary-out/` frames re-blend against stage N+1's stored raw
`boundary-in/` frames, and only N+1's segment is re-_encoded_ — no re-render
(this is why `boundary-in/` keeps the pre-blend frames, §3.2). The API
rejects `splice` when N+1's transition is `Smooth` with an error explaining
why. Amended (Run 3): the machine-readable code is
`RETAKE_SPLICE_REQUIRES_CUT_OR_FADE`.

### 8.3 Versioned finalize

Each finalize writes `final/output-<n>.mp4` and a new gallery row (the
gallery is append-only; users compare takes side by side). The manifest
records finalize history with the per-stage seeds that produced each output,
so any take is attributable and reproducible.

Retake edits are recorded as amendments in the manifest (`[[retakes]]` list:
stage, old/new seed, old/new prompt, timestamp) — the original request stays
intact for provenance.

## 9. API surface

New (all under the existing auth / `MOLD_API_KEY` regime):

| Route                                         | Effect                                                                                    |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `POST /api/chain-jobs`                        | Validate + normalise ChainRequest, create job dir + rows, start runner. 202 `{job_id}`    |
| `GET /api/chain-jobs`                         | List jobs (state, model, progress, timestamps)                                            |
| `GET /api/chain-jobs/:id`                     | Job detail incl. per-stage states + preview URLs                                          |
| `GET /api/chain-jobs/:id/events`              | SSE; re-attachable at any time (replays current state as first event, then live progress) |
| `POST /api/chain-jobs/:id/resume`             | Resume interrupted/failed/cancelled job                                                   |
| `POST /api/chain-jobs/:id/retake`             | §8                                                                                        |
| `POST /api/chain-jobs/:id/cancel`             | Stop at next boundary, keep artifacts                                                     |
| `DELETE /api/chain-jobs/:id`                  | Remove job + directory (409 while running)                                                |
| `POST /api/chain-jobs/gc`                     | Amended (Run 3): force retention/ephemeral artifact GC                                    |
| `GET /api/chain-jobs/:id/stages/:idx/preview` | Stage thumbnail (`preview.jpg`)                                                           |

Compatibility: `POST /api/generate/chain` and `/api/generate/chain/stream`
become shims — create a job, block/stream on its events, return the existing
response shapes (`ChainResponse` / SSE `complete` with `SseChainCompleteEvent`).
CLI, TUI, and older SPAs keep working unchanged; they migrate to the job API
incrementally. The shims mark the job `ephemeral` in the manifest so GC can
treat sync-path jobs like today's behavior (artifacts deleted after
success).

Amended (Run 3): chain progress events gain `job_id` as an additive field.
The extra-event design is withdrawn; existing progress event variants keep
their shapes aside from optional correlation metadata.

## 10. Rider refactor: model-agnostic chain module

Moves as part of this work (not a separate workstream):

- `ltx2/chain.rs` (orchestrator, `ChainTail`, `ChainStageRenderer`,
  `StageOutcome`) and `ltx2/stitch.rs` → `mold_inference::chain`. The RGB
  carryover currency is already model-agnostic; only the
  `tail_latent_frame_count` VAE-ratio helper stays behind in `ltx2` (it moves
  into the LTX-2 renderer impl).
- `chain_limits::family_cap` + `family_supports_audio` matches → a
  `ChainCapability` descriptor returned by the engine family:

  ```rust
  pub struct ChainCapability {
      pub frames_per_clip_cap: u32,
      pub carryover: CarryoverKind,   // TemporalHandoff | IndependentClips
      pub supports_audio: bool,
  }
  ```

  `/api/capabilities/chain-limits` derives from it; LTX-Video's
  independent-clip fallback becomes `CarryoverKind::IndependentClips` instead
  of a comment. New families implement the trait and appear in chain limits
  automatically.

- `runtime.rs` (7.5k lines) is **not** decomposed wholesale; the pieces this
  project touches (chain-stage entry points) move out with the module split,
  consistent with refactor-as-we-go.

## 11. Web ScriptComposer

- Submission switches to `POST /api/chain-jobs`; the running card subscribes
  to `/events` and reconciles against `GET /api/chain-jobs` on reconnect
  (same dead-letter pattern `useQueueReconciler` uses for `GET /api/queue`).
- Stage timeline: each stage card shows pending/running/completed state, the
  checkpoint thumbnail on completion, and per-stage denoise progress.
- Completed/interrupted jobs render Resume / Cancel / Retake controls; retake
  opens seed/prompt overrides and offers cascade vs splice (splice disabled
  with a tooltip when the next boundary is Smooth).
- A Jobs view (or a section of the existing downloads/queue drawer) lists
  non-terminal and recent jobs.

## 12. Retention / GC

- Final outputs are gallery files — never GC'd by this system.
- Stage artifacts of **completed** jobs are deleted after
  `chain.jobs_artifact_ttl_days` (DB-backed setting, default 7) — preserving
  retake ability for a week.
- **Failed / interrupted / cancelled** jobs are exempt from TTL (they hold
  the only copy of paid-for work) until resolved or deleted.
- `ephemeral` (sync-shim) jobs: artifacts deleted immediately after success.
- Amended (Run 3): ephemeral cleanup has highest precedence. Successful
  ephemeral shim jobs are swept immediately even though completed jobs
  normally retain artifacts. TTL pruning applies only to non-ephemeral
  completed jobs whose `finalized_at` is older than
  `chain.jobs_artifact_ttl_days`.
- Amended (Run 3): an orphan artifact directory is a directory under
  `MOLD_HOME/jobs` with no DB row and no active ephemeral claim or running
  lock. Claimed or active directories are not orphans.
- GC runs at startup and daily; `mold jobs gc` (CLI) forces it. A `mold jobs`
  subcommand family (`list`, `resume`, `retake`, `gc`) mirrors the API for
  headless use.

## 13. Error handling

- Stage render failure → stage `failed`, job `failed`, artifacts kept, error
  in DB + manifest + SSE. Resume retries the failed stage.
- OOM specifically: flows through the existing `record_model_cuda_oom`
  cooldown; resume after cooldown is the recovery path. (The known LRU-VRAM
  leak is tracked separately in
  `tasks/known-issue-lru-eviction-not-freeing-vram.md` and is a prerequisite
  for long chains in practice, but this design does not depend on its fix.)
- Disk-full during artifact write → job `failed` with an explicit disk-space
  error; no partial manifest update (temp+rename).
- Manifest/DB divergence → manifest wins, reconcile logs a warning.
- Finalize failure (encode/mux) → job stays `completed`-stages but
  `finalize_failed`; finalize is retryable without re-rendering.

## 14. Testing

TDD per repo rules; the existing weight-free `FakeRenderer` pattern extends
to the runner:

- **Runner:** fake renderer + tempdir job dir — checkpoint write ordering,
  yield-point behavior (inject a fake queue-depth probe), cancel-at-boundary,
  resume-from-stage-N, retake cascade/splice state transitions, splice
  rejected on Smooth.
- **Determinism:** tail PNGs round-trip bit-identical → `ChainTail`
  equality; effective-seed persistence.
- **Stitch parity:** segment-concat frame counts equal `StitchPlan::assemble`
  for smooth/cut/fade mixes (355-frame fixture).
- **DB:** migration v11 up-from-v10; reconcile repairs a stale row from
  manifest; TTL GC selects the right victims.
- **API:** routes tests for the new endpoints incl. shim equivalence
  (`/api/generate/chain` result matches job-API result for the same request);
  SSE re-attach replays state.
- **Web:** ScriptComposer submit/reconcile/retake flows in vitest, mirroring
  existing `useGenerateStream` test structure.
- **UAT (killswitch):** 8-stage chain kill-and-resume; retake stage mid-chain
  both modes; concurrent single-clip generate lands between stages.

## 15. Rollout

1. `mold_inference::chain` module move + `ChainCapability` (pure refactor,
   no behavior change, lands first).
2. `video_enc` incremental encoder.
3. mold-db migration + job dir/manifest read-write.
4. Runner with checkpointing behind the new job API (old endpoints untouched).
5. Old endpoints become shims; SPA switches to job API; stage timeline +
   resume/retake UI.
6. GC + CLI `mold jobs`.

Each step is independently shippable and CI-green; docs
(`website/`, `.claude/skills/mold/SKILL.md`, `CHANGELOG.md`) update with the
steps that change user-facing surface, per repo policy.

## 16. Open questions (non-blocking)

- Segment codec-param pinning for the future no-re-encode concat: worth
  recording encoder settings in the manifest now so old jobs qualify later.
- Whether `mold-discord` should expose chain jobs (it only depends on
  `mold-core`; the job API is plain HTTP, so nothing structural blocks it).
- Retake with edited `frames`/`transition` (changes boundary math of
  neighbors) — deliberately excluded from v1; revisit with real usage.

## 17. Amend + raw segments (amended 2026-07-28)

Users can edit an existing multi-clip sequence and re-render only the clips
that need it, reusing cached stages. Two coupled changes:

### 17.1 Raw-segment artifact contract

`write_stage_artifacts` no longer trims or blends. For every new stage:

- `segment.mp4` is RAW — every frame the engine emitted.
- The audio sidecar is the FULL untrimmed track.
- `tail/` PNGs are ALWAYS written when the engine produced a motion tail —
  even when the next transition isn't Smooth, and even for the last stage.
  They are the bit-exact carry source for resume/amend; carry is never
  derived from lossy H.264 decode.
- `boundary-in/` / `boundary-out/` are no longer written.
- `preview.jpg` is unchanged.

`StageStatus` gains `#[serde(default)] raw_segment: bool` to distinguish raw
artifacts from legacy trimmed ones; the manifest schema string stays
`mold.chainjob.v1` and old manifests parse via the default.

**Finalize is the single place boundary math happens** for raw stages, from
the EFFECTIVE script: a continuation entering with Smooth drops its leading
`motion_tail_frames` (frames + audio); a Fade boundary blends the prior raw
segment's trailing `fade_len` with the incoming stage's leading `fade_len`
via `fade_boundary` (with an audio crossfade) while the prior stage's
trailing `fade_len` is withheld from the output. Legacy stages pass through
exactly as before, and mixed legacy+raw jobs finalize correctly (a legacy
predecessor's blend inputs come from its `boundary-out/`).
`maybe_reencode_next_after_fade` survives ONLY for legacy Fade successors;
raw successors are never re-encoded in place — a retake/amend of the stage
before them simply re-finalizes.

`frames_emitted` keeps its wire meaning — frames the stage contributes to
the final video after boundary accounting — computed by the shared pure
helper `mold_core::chain::stage_contributed_frames`, which
`ChainRequest::estimated_total_frames` also sums.

**Downgrade note:** an older mold finalizing a raw-segment job would
concatenate untrimmed segments and produce a wrong video. Downgrade across
this change is unsupported. Upgrade is safe: legacy jobs resume, retake, and
finalize unchanged.

### 17.2 Amend endpoint

`POST /api/chain-jobs/:id/amend` accepts `AmendRequest`: the FULL edited
stage list (canonical order) plus optional chain-level overlays
(`motion_tail_frames`, `fps`, `seed` as a decimal string, `steps`,
`guidance`, `enable_audio`; omitted = keep current). NOT amendable — the
client must create a fresh job: model, width, height, output_format,
placement, strength, batch provenance.

Semantics (all under the per-job mutation lock, like retake):

- 409 `CHAIN_JOB_RUNNING` while running; 409 `CHAIN_JOB_EPHEMERAL` for shim
  jobs; allowed from Queued/Interrupted/Failed/Cancelled/Completed; 422 when
  the candidate request fails the create-time gates
  (`validate_and_normalize_chain_family` + `normalise()` + Mp4-only).
- The candidate request is the current `effective_request` (retakes folded)
  with the stages replaced and overlays applied.
- **Invalidation** — `preserved_stages` is:
  1. `0` when seed/steps/guidance/fps/motion_tail_frames changed or
     `enable_audio` flipped OFF→ON (ON→OFF preserves everything; finalize
     just ignores sidecars).
  2. Otherwise the longest common prefix of per-stage render identity:
     `(prompt, frames, negative_prompt, source_image bytes, effective
per-stage seed, uses_carry)` with `uses_carry = idx > 0 && transition
== Smooth`. Cut↔Fade toggles and `fade_frames` edits do NOT break the
     prefix (finalize-only under raw segments); Smooth↔(Cut|Fade) does.
  3. Clamped to the leading run of Completed stages, then shrunk past any
     LEGACY stage whose baked-in artifacts can't serve the new boundary plan
     (changed incoming boundary, missing tail before a new Smooth successor,
     trailing truncation that no longer matches the new successor).
  4. Appending clips preserves all old stages; removing trailing clips can
     make the amend "boundary-only" (zero renders, just a re-finalize).
     Preserved stage dirs are never renumbered.
- Application: CAS → Queued; `AmendRecord { at_unix_ms,
previous_request_json (pre-amend EFFECTIVE request), preserved_stages }`
  appended to the manifest's `amends`; `request_json` rewritten to the
  normalized candidate; `retakes` cleared (folded into the snapshot);
  preserved `stage_status` rows kept verbatim (raw rows' `frames_emitted`
  re-derived for the new plan), fresh Pending rows appended, trailing rows
  dropped; invalidated `stages/NNN/` dirs deleted; DB index follows
  (`set_request_json`, `delete_stages_from`, stage upserts,
  `update_stage_shape`); runner kicked; `chain_job_queued` emitted.
- A Completed job whose amend preserves every stage still requeues: the
  runner's stage loop finds no incomplete stage and falls through to
  `finalize_job`, producing a new versioned take from the cached raw
  segments under the new boundary plan.
- Response: 202 `AmendResponse` (`ChainJobSummary` + `preserved_stages`).
  `ChainJobDetail` gains the additive `amends` history.

## 18. Shipped (2026-07-28)

Section 17 is no longer a proposal — `POST /api/chain-jobs/:id/amend` and the
raw-segment / finalize-time-boundary artifact contract are both merged, along
with the additive `chain_job_queued` / `chain_job_started` / `chain_job_ended`
`ServerEvent` variants that let clients track sequences without polling
`/api/chain-jobs`. The user-facing story is the **Update sequence** action on
the unified Create clip rail (desktop and web).

See `CHANGELOG.md` `[Unreleased]` — "Sequences can now be edited in place with
cached clips reused" and "Chain jobs now announce their lifecycle on the server
event stream" — and the wire reference in `website/api/index.md`
(`POST /api/chain-jobs/:id/amend`, `/api/events`).
