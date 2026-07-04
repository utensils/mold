#![allow(dead_code)] // Phase 3 placeholder — removed in Phase 6

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mold_core::chain::ChainRequest;
use mold_core::chain_job::{ChainJobEvent, ChainJobManifest, RetakeRequest};
use mold_core::GenerateRequest;
use mold_db::chain_jobs::ChainJobRow;
use mold_db::MetadataDb;
use mold_inference::chain::{ChainTail, StageOutcome};

pub struct ChainJobRunnerHandle {
    kick_tx: tokio::sync::mpsc::UnboundedSender<()>,
    cancel: Arc<CancelRegistry>,
    events: Arc<JobEventBus>,
}

pub struct CancelRegistry(std::sync::Mutex<std::collections::HashSet<String>>);

pub struct JobEventBus {
    senders: std::sync::Mutex<
        std::collections::HashMap<String, tokio::sync::broadcast::Sender<ChainJobEvent>>,
    >,
}

pub struct RunnerDeps {
    pub db: Arc<Option<mold_db::MetadataDb>>,
    pub jobs_root: PathBuf,
    pub executor: Arc<dyn StageExecutor>,
    pub queue_probe: Arc<dyn QueueProbe>,
    pub events: Arc<JobEventBus>,
    pub cancel: Arc<CancelRegistry>,
    pub output_dir: Option<PathBuf>,
}

pub trait StageExecutor: Send + Sync {
    fn render_stage(
        &self,
        model: &str,
        stage_req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_frames: u32,
        progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
    ) -> anyhow::Result<StageRenderOutcome>;
}

pub enum StageRenderOutcome {
    Done(StageOutcome),
    Cancelled,
}

pub trait QueueProbe: Send + Sync {
    fn small_jobs_waiting(&self) -> usize;
}

struct StageArtifactPaths {
    segment_rel: String,
    audio_rel: Option<String>,
    tail_frames: u32,
    preview_written: bool,
}

impl ChainJobRunnerHandle {
    /// Nudge the runner: a job row was inserted/reset to queued.
    pub fn kick(&self) {
        // TODO(BR55 Phase 6): send a runner wake-up signal over kick_tx.
        todo!()
    }

    /// Mark cancel-requested; false when the job is unknown to the registry.
    pub fn request_cancel(&self, _job_id: &str) -> bool {
        // TODO(BR55 Phase 6): record the job id in CancelRegistry if it is known.
        todo!()
    }

    /// SSE attach: MUST be called BEFORE snapshot synthesis (buffered
    /// duplicates arriving after Snapshot are the correct side of the race;
    /// clients apply idempotently). Lagged receivers re-attach → fresh Snapshot.
    pub fn subscribe(&self, _job_id: &str) -> tokio::sync::broadcast::Receiver<ChainJobEvent> {
        // TODO(BR55 Phase 6): create or reuse the per-job broadcast sender and return a receiver.
        todo!()
    }
}

/// Spawn the single long-lived runner task. Called once from lib.rs startup,
/// strictly AFTER startup_reconcile completes (sequential await — no race).
pub fn spawn_runner(_deps: RunnerDeps) -> ChainJobRunnerHandle {
    // TODO(BR55 Phase 6): spawn the durable FIFO chain-job runner task.
    todo!()
}

/// Spec §7 steps 1–2: flip running→interrupted; repair rows from manifests
/// (manifest wins). Returns (flipped, repaired) for the startup log.
pub fn startup_reconcile(
    _db: &mold_db::MetadataDb,
    _jobs_root: &Path,
) -> anyhow::Result<(usize, usize)> {
    // TODO(BR55 Phase 6): reconcile running DB rows and manifest-derived job state on startup.
    todo!()
}

// ── runner internals (private; unit-tested via RunnerDeps fakes) ──

async fn run_loop(_deps: Arc<RunnerDeps>, _kick_rx: tokio::sync::mpsc::UnboundedReceiver<()>) {
    // TODO(BR55 Phase 6): run the queued-job polling loop until shutdown.
    todo!()
}

/// FIFO by created_at among state=queued.
fn next_queued_job(_db: &MetadataDb) -> anyhow::Result<Option<ChainJobRow>> {
    // TODO(BR55 Phase 6): query the oldest queued chain job row.
    todo!()
}

/// Execute stages from start_stage. Per stage: §3.3 ordering (artifacts →
/// manifest → DB); yield check (deps.queue_probe) + cancel check at each
/// boundary; events via bus. RE-BLEND CONTRACT (spec §8.2 splice-after-fade,
/// placed HERE not in apply_retake because it needs the re-rendered stage's
/// NEW boundary-out/ frames): after writing a re-rendered stage N whose
/// OUTGOING transition (stage N+1's incoming) is Fade and whose next stage
/// is already completed, re-encode stage N+1's segment by decoding it
/// (decode_video_frames_from_path), replacing its leading fade_len frames
/// with fade_boundary(new boundary-out/, stored raw boundary-in/), and
/// re-encoding via Mp4StreamEncoder. No re-render; audio sidecar unchanged.
/// Derivability: StageState has no "needs re-encode" variant (Run 1 enum
/// frozen) — the condition is recomputed at write time purely from the
/// effective script's transitions + next-stage completion state.
/// Ends in exactly one of: finalize_job | Failed(+error) | Cancelled.
/// JobEventBus entry for the job is cleaned up at terminal state.
fn execute_job(_deps: &RunnerDeps, _job: &ChainJobRow, _start_stage: u32) -> anyhow::Result<()> {
    // TODO(BR55 Phase 6): execute queued stages with artifact-first persistence and terminal cleanup.
    todo!()
}

/// One stage: build the stage GenerateRequest from the EFFECTIVE script
/// (effective_stage_seed + retake amendments); render via deps.executor.
/// The progress callback bridges CancelRegistry → ControlFlow::Break and
/// forwards DenoiseStep events; it is Box-moved into spawn_blocking (a
/// borrowed &dyn Fn cannot cross the 'static bound — P1 advisory).
fn execute_stage(
    _deps: &RunnerDeps,
    _job: &ChainJobRow,
    _manifest: &ChainJobManifest,
    _stage_idx: u32,
    _carry: Option<&ChainTail>,
) -> anyhow::Result<StageRenderOutcome> {
    // TODO(BR55 Phase 6): build the per-stage request and invoke the configured StageExecutor.
    todo!()
}
// StageRenderOutcome is the P1-approved enum { Done(StageOutcome), Cancelled }
// (defined alongside the StageExecutor trait). execute_stage returns it
// UNCHANGED from the executor: ControlFlow::Break in the progress callback
// maps to Ok(StageRenderOutcome::Cancelled) inside render_stage; Err is
// reserved for real failures. No parallel result type exists.

/// Write-time boundary handling (spec §5): smooth → drop leading
/// motion_tail frames pre-encode; outgoing-fade → trim trailing fade_len
/// from own segment, persist raw trailing frames to boundary-out/;
/// incoming-fade → persist raw leading frames to boundary-in/, blend
/// against prior stage's boundary-out/ before encode. Segment via
/// Mp4StreamEncoder; tail/*.png (next-transition-Smooth only); audio.pcm
/// sidecar (f32 interleaved) when audio present; preview.jpg (last frame).
/// Then manifest.write_atomic, then DB upsert_stage + set_current_stage.
/// Disk-full or any write error → job Failed with explicit error (§13).
fn write_stage_artifacts(
    _layout: &mold_core::chain_job::JobDirLayout,
    _manifest: &mut ChainJobManifest,
    _stage_idx: u32,
    _outcome: &StageOutcome,
    _effective: &ChainRequest,
) -> anyhow::Result<StageArtifactPaths> {
    // TODO(BR55 Phase 6): persist stage media, boundary data, manifest, and DB stage state in order.
    todo!()
}

/// Resume path: rebuild carryover from prev stage's tail/*.png (lossless);
/// None when the incoming transition is Cut/Fade. Rejects `..`/absolute
/// manifest paths BEFORE any join (consumer-side discipline, Run 1 inv 12 —
/// applies equally to request_json-derived paths).
fn resume_carry_from_disk(
    _job_dir: &Path,
    _manifest: &ChainJobManifest,
    _stage_idx: u32,
) -> anyhow::Result<Option<ChainTail>> {
    // TODO(BR55 Phase 6): rebuild a lossless carry tail from validated manifest-relative paths.
    todo!()
}

/// Re-runnable finalize (spec §5.1): per-segment decode (memory bound = one
/// segment) → Mp4StreamEncoder push → PCM concat → AAC mux
/// (attach_aac_track_from_f32_interleaved) → final/output-<n>.mp4 with
/// n = manifest.finalizes.len() + 1 → gallery row (save_video_to_dir + DB
/// upsert) → FinalizeRecord + set_finalized_at. Failure → job Failed +
/// error; resume with all stages completed re-enters HERE (the §13
/// "finalize_failed" mapping onto the frozen six-state enum).
fn finalize_job(
    _deps: &RunnerDeps,
    _job: &ChainJobRow,
    _manifest: &mut ChainJobManifest,
) -> anyhow::Result<String> {
    // TODO(BR55 Phase 6): decode completed segments, mux final output, upsert gallery metadata, and record finalize state.
    todo!()
}

/// Spec §8 request-time half ONLY: validate stage_idx bounds and mode
/// legality (splice ⇒ stage N+1's transition is Cut or Fade, else
/// RETAKE_SPLICE_REQUIRES_CUT_OR_FADE), record RetakeAmendment (old/new
/// seed+prompt), reset stage rows/manifest statuses (cascade: N..end →
/// pending; splice: N only), set job state to queued. The CALLER
/// (retake_chain_job, which holds AppState.chain_jobs) performs the runner
/// kick after this returns — no handle is reachable from this signature,
/// by design (keeps the fn sync-testable over &MetadataDb). The splice-after-fade
/// re-encode of N+1 happens in execute_job AFTER N re-renders (see above).
pub fn apply_retake(
    _db: &MetadataDb,
    _jobs_root: &Path,
    _job_id: &str,
    _req: &RetakeRequest,
) -> anyhow::Result<ChainJobRow> {
    // TODO(BR55 Phase 6): validate and persist retake amendments before requeueing the job.
    todo!()
}

pub struct ProductionStageExecutor {/* Arc<GpuPool>, config snapshot, model cache handles */}

impl StageExecutor for ProductionStageExecutor {
    fn render_stage(
        &self,
        _model: &str,
        _stage_req: &GenerateRequest,
        _carry: Option<&ChainTail>,
        _motion_tail_frames: u32,
        _progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
    ) -> anyhow::Result<StageRenderOutcome> {
        // TODO(BR55 Phase 6): select a GPU worker and call gpu_worker::run_stage_blocking.
        todo!()
    }
    // impl detail: select worker → gpu_worker::run_stage_blocking (per-stage lock)
}

pub struct ProductionQueueProbe {/* QueueHandle clone + Arc<GpuPool> */}

impl QueueProbe for ProductionQueueProbe {
    fn small_jobs_waiting(&self) -> usize {
        // TODO(BR55 Phase 6): return queued small jobs plus in-flight worker count.
        todo!()
    }
}

// ── Phase 3 committed test mapping (spec §13) ─────────────────────────
// TODO(BR55 Phase 6): test — disk-full (or any write error) during
// write_stage_artifacts marks the job Failed with an explicit error
// naming the failed write; no partial manifest update.
// TODO(BR55 Phase 6): test — finalize failure leaves stages completed and
// job Failed; resume with all stages completed re-enters finalize_job.
