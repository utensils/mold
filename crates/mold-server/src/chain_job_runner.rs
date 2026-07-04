use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context};
use image::codecs::jpeg::JpegEncoder;
use image::{ImageEncoder, RgbImage};
use mold_core::chain::{ChainRequest, ChainStage, TransitionMode};
use mold_core::chain_job::{
    effective_stage_seed, settled, ChainJobEvent, ChainJobManifest, ChainJobState, FinalizeRecord,
    GcOutcome, JobDirLayout, RetakeAmendment, RetakeMode, RetakeRequest, StageState, STAGES_DIR,
};
use mold_core::{GenerateRequest, OutputFormat, OutputMetadata};
use mold_db::chain_jobs::{self, ChainJobRow, ChainJobStageRow};
use mold_db::{settings, MetadataDb};
use mold_inference::audio::NativeAudioTrack;
use mold_inference::chain::stitch::fade_boundary;
use mold_inference::chain::{ChainTail, StageOutcome, StageProgressEvent};
use mold_inference::ltx_video::video_enc;
use sha2::{Digest, Sha256};

use crate::gpu_pool::{ActiveGeneration, GpuPool, GpuWorker};
use crate::gpu_worker;
use crate::model_manager;
use crate::queue::save_video_to_dir;
use crate::state::QueueHandle;

const EVENT_BUS_CAPACITY: usize = 256;
const DEFAULT_FADE_FRAMES: u32 = 8;
const AUDIO_SIDECAR_MAGIC: &[u8; 8] = b"MOLDPCM1";
pub const EPHEMERAL_GRACE_SECS: u64 = 900;

pub struct ChainJobRunnerHandle {
    kick_tx: tokio::sync::mpsc::UnboundedSender<RunnerCmd>,
    cancel: Arc<CancelRegistry>,
    events: Arc<JobEventBus>,
    job_locks: Arc<JobMutationLocks>,
    claims: Arc<EphemeralClaims>,
}

pub struct CancelRegistry {
    known: Mutex<HashSet<String>>,
    cancelled: Mutex<HashSet<String>>,
}

/// RAII claim. Held by the SHIM'S WORKER TASK (pinned P0 note 2 — never only
/// by the SSE stream future) across create→settle→read→delete. Drop = sync
/// release (std Mutex, matching Run 2 registry conventions). GC never sweeps
/// an ephemeral with a live claim.
pub struct EphemeralClaimGuard {
    job_id: String,
    claims: Arc<EphemeralClaims>,
}

#[derive(Default)]
pub struct EphemeralClaims {
    claimed: std::sync::Mutex<std::collections::HashSet<String>>,
}

pub struct JobEventBus {
    senders: Mutex<HashMap<String, tokio::sync::broadcast::Sender<ChainJobEvent>>>,
}

/// Process-local per-job exclusion for manifest and row read-modify-write
/// sections.
///
/// This is a single-writer-process guard: it serializes the runner and HTTP
/// handlers inside one `mold serve` process. Cross-process writers are still
/// excluded by the SQLite state predicates, not by this in-memory map.
pub struct JobMutationLocks {
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

pub struct RunnerDeps {
    pub db: Arc<Option<mold_db::MetadataDb>>,
    pub jobs_root: PathBuf,
    pub executor: Arc<dyn StageExecutor>,
    pub queue_probe: Arc<dyn QueueProbe>,
    pub events: Arc<JobEventBus>,
    pub cancel: Arc<CancelRegistry>,
    pub job_locks: Arc<JobMutationLocks>,
    pub claims: Arc<EphemeralClaims>,
    pub output_dir: Option<PathBuf>,
}

pub(crate) struct CreateJobParams {
    pub id: String,
    pub ephemeral: bool,
    pub request: ChainRequest,
}

pub enum RunnerCmd {
    Kick,
    Gc {
        reply: tokio::sync::oneshot::Sender<std::result::Result<GcOutcome, String>>,
    },
}

// GC scheduling + trigger pin: the DAILY TICK lives inside run_loop via
// tokio::time::interval; the HTTP route NEVER runs a pass itself — it asks
// the runner. The kick channel widens to a command channel:
//   enum RunnerCmd { Kick, Gc { reply: tokio::sync::oneshot::Sender<Result<GcOutcome, String>> } }
//   ChainJobRunnerHandle::kick() sends RunnerCmd::Kick;
//   pub async fn request_gc(&self) -> Result<GcOutcome> sends RunnerCmd::Gc
//   and awaits the reply. Single executor preserved (run_loop owns every
//   pass). SERVICING PIN: run_loop drains the command channel NON-BLOCKINGLY
//   (try_recv loop) at the top of each outer iteration — i.e. between jobs —
//   so a Gc reply (and the daily tick, polled at the same points) waits at
//   most ONE job's runtime, never the whole queue. STATED RESIDUAL: under a
//   long-running multi-stage job, POST /api/chain-jobs/gc and `mold jobs gc`
//   block up to that one job's duration; accepted for v1 (GC is not
//   latency-sensitive; the daily tick absorbs routine cleanup).
//   ttl_days is read by the RUNNER from the settings DB
//   (CHAIN_JOBS_ARTIFACT_TTL_DAYS, default 7) at each pass.
// resume handler behavior add: ephemeral job → 409 CHAIN_JOB_EPHEMERAL.

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
    preview_written: bool,
}

impl CancelRegistry {
    pub fn new() -> Self {
        Self {
            known: Mutex::new(HashSet::new()),
            cancelled: Mutex::new(HashSet::new()),
        }
    }

    fn register(&self, job_id: &str) {
        self.known
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job_id.to_string());
    }

    fn unregister(&self, job_id: &str) {
        self.known
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
        self.cancelled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
    }

    fn request(&self, job_id: &str) -> bool {
        let known = self
            .known
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(job_id);
        if known {
            self.cancelled
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(job_id.to_string());
            true
        } else {
            false
        }
    }

    fn is_cancelled(&self, job_id: &str) -> bool {
        self.cancelled
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(job_id)
    }
}

impl Default for CancelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl EphemeralClaims {
    /// RAII claim; guard held by the SHIM WORKER TASK across
    /// create→settle→read→delete (P0 note 2). Sync-only Drop.
    pub fn claim(self: &Arc<Self>, job_id: &str) -> EphemeralClaimGuard {
        self.claimed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(job_id.to_string());
        EphemeralClaimGuard {
            job_id: job_id.to_string(),
            claims: self.clone(),
        }
    }

    pub fn is_claimed(&self, job_id: &str) -> bool {
        self.claimed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains(job_id)
    }
}

impl Drop for EphemeralClaimGuard {
    fn drop(&mut self) {
        self.claims
            .claimed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.job_id);
    }
}

impl JobMutationLocks {
    pub fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
        }
    }

    fn mutex_for(&self, job_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks
            .entry(job_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    pub async fn lock(&self, job_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        self.mutex_for(job_id).lock_owned().await
    }

    pub fn blocking_lock(&self, job_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        self.mutex_for(job_id).blocking_lock_owned()
    }

    pub fn remove(&self, job_id: &str) {
        self.locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
    }
}

impl Default for JobMutationLocks {
    fn default() -> Self {
        Self::new()
    }
}

impl JobEventBus {
    pub fn new() -> Self {
        Self {
            senders: Mutex::new(HashMap::new()),
        }
    }

    pub fn subscribe_for_job(
        &self,
        db: &MetadataDb,
        job_id: &str,
    ) -> anyhow::Result<tokio::sync::broadcast::Receiver<ChainJobEvent>> {
        let mut senders = self
            .senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = chain_jobs::get_job(db, job_id)?.map(|row| row.state);
        if state.is_none_or(settled) {
            let (_tx, rx) = tokio::sync::broadcast::channel(EVENT_BUS_CAPACITY);
            return Ok(rx);
        }
        Ok(senders
            .entry(job_id.to_string())
            .or_insert_with(|| tokio::sync::broadcast::channel(EVENT_BUS_CAPACITY).0)
            .subscribe())
    }

    #[cfg(test)]
    pub(crate) fn subscribe_persistent_for_tests(
        &self,
        job_id: &str,
    ) -> tokio::sync::broadcast::Receiver<ChainJobEvent> {
        let mut senders = self
            .senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        senders
            .entry(job_id.to_string())
            .or_insert_with(|| tokio::sync::broadcast::channel(EVENT_BUS_CAPACITY).0)
            .subscribe()
    }

    pub fn publish(&self, job_id: &str, event: ChainJobEvent) {
        let sender = {
            let senders = self
                .senders
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            senders.get(job_id).cloned()
        };
        if let Some(sender) = sender {
            let _ = sender.send(event);
        }
    }

    pub fn publish_then_remove(&self, job_id: &str, event: ChainJobEvent) {
        let mut senders = self
            .senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(sender) = senders.get(job_id) {
            let _ = sender.send(event);
        }
        senders.remove(job_id);
    }

    pub fn remove(&self, job_id: &str) {
        self.senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
    }

    #[cfg(test)]
    pub(crate) fn contains_for_tests(&self, job_id: &str) -> bool {
        self.senders
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(job_id)
    }
}

impl Default for JobEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainJobRunnerHandle {
    #[cfg(test)]
    pub(crate) fn inert_for_tests() -> Self {
        let (kick_tx, _kick_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            kick_tx,
            cancel: Arc::new(CancelRegistry::new()),
            events: Arc::new(JobEventBus::new()),
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
        }
    }

    /// Nudge the runner: a job row was inserted/reset to queued.
    pub fn kick(&self) {
        let _ = self.kick_tx.send(RunnerCmd::Kick);
    }

    /// Mark cancel-requested; false when the job is unknown to the registry.
    pub fn request_cancel(&self, job_id: &str) -> bool {
        self.cancel.request(job_id)
    }

    pub fn unregister_cancel(&self, job_id: &str) {
        self.cancel.unregister(job_id);
    }

    pub async fn lock_job(&self, job_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        self.job_locks.lock(job_id).await
    }

    pub(crate) fn blocking_lock_job(&self, job_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        self.job_locks.blocking_lock(job_id)
    }

    pub fn remove_job_lock(&self, job_id: &str) {
        self.job_locks.remove(job_id);
    }

    pub(crate) fn claim_ephemeral(&self, job_id: &str) -> EphemeralClaimGuard {
        self.claims.claim(job_id)
    }

    pub async fn request_gc(&self) -> anyhow::Result<GcOutcome> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.kick_tx
            .send(RunnerCmd::Gc { reply })
            .map_err(|_| anyhow!("chain job runner stopped before GC request could be sent"))?;
        rx.await
            .map_err(|_| anyhow!("chain job runner stopped before replying to GC request"))?
            .map_err(|msg| anyhow!(msg))
    }

    pub fn cleanup_deleted(&self, job_id: &str) {
        self.cancel.unregister(job_id);
        self.events.remove(job_id);
    }

    pub fn publish_settled_state(&self, job_id: &str, state: ChainJobState, error: Option<String>) {
        self.events
            .publish_then_remove(job_id, ChainJobEvent::StateChanged { state, error });
        self.cancel.unregister(job_id);
    }

    /// SSE attach: MUST be called BEFORE snapshot synthesis (buffered
    /// duplicates arriving after Snapshot are the correct side of the race;
    /// clients apply idempotently). Lagged receivers re-attach -> fresh Snapshot.
    pub fn subscribe(
        &self,
        db: &MetadataDb,
        job_id: &str,
    ) -> anyhow::Result<tokio::sync::broadcast::Receiver<ChainJobEvent>> {
        self.events.subscribe_for_job(db, job_id)
    }

    #[cfg(test)]
    pub(crate) fn events_for_tests(&self) -> &JobEventBus {
        &self.events
    }
}

/// Spawn the single long-lived runner task. Called once from lib.rs startup,
/// strictly AFTER startup_reconcile completes (sequential await; no race).
pub fn spawn_runner(deps: RunnerDeps) -> ChainJobRunnerHandle {
    let (kick_tx, kick_rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel = deps.cancel.clone();
    let events = deps.events.clone();
    let job_locks = deps.job_locks.clone();
    let claims = deps.claims.clone();
    let deps = Arc::new(deps);
    tokio::spawn(run_loop(deps, kick_rx));
    ChainJobRunnerHandle {
        kick_tx,
        cancel,
        events,
        job_locks,
        claims,
    }
}

/// Factored creation entry (P1 pin): called by routes_chain_jobs.rs
/// create_chain_job (generated id, ephemeral=false) AND the shims
/// (pre-generated claimed id, ephemeral=true). VALIDATION SPLIT PIN: the
/// async CALLERS perform family validation (validate_and_normalize_chain_
/// family is async + &AppState), normalise(), and the non-Mp4 422 gate
/// (ApiError::validation) BEFORE calling this. This fn is sync
/// storage-only: job dir + manifest + rows; anyhow errors = internal 500.
pub(crate) fn create_job_with_params(
    db: &MetadataDb,
    jobs_root: &Path,
    params: CreateJobParams,
) -> anyhow::Result<ChainJobRow> {
    std::fs::create_dir_all(jobs_root)
        .with_context(|| format!("creating chain jobs root '{}'", jobs_root.display()))?;
    let job_dir = jobs_root.join(&params.id);
    let layout = JobDirLayout::new(job_dir.clone());
    layout.ensure_root()?;

    let now = now_ms_i64();
    let mut manifest = ChainJobManifest::new(params.id.clone(), now.max(0) as u64, &params.request)
        .map_err(|e| anyhow!("{e:#}"))?;
    manifest.ephemeral = params.ephemeral;
    manifest
        .write_atomic(&job_dir)
        .map_err(|e| anyhow!("{e:#}"))?;
    let request_json = serde_json::to_string(&params.request)?;
    let row = ChainJobRow {
        id: params.id.clone(),
        state: ChainJobState::Queued,
        model: params.request.model.clone(),
        request_json,
        job_dir,
        stage_count: params.request.stages.len() as u32,
        current_stage: 0,
        error: None,
        created_at_ms: now,
        updated_at_ms: now,
        finalized_at_ms: None,
    };
    chain_jobs::insert_job(db, &row)?;
    for stage in &manifest.stage_status {
        chain_jobs::upsert_stage(
            db,
            &ChainJobStageRow {
                job_id: row.id.clone(),
                stage_idx: stage.idx,
                state: stage.state,
                seed: stage.seed,
                frames_emitted: None,
                generation_time_ms: None,
                segment_rel_path: None,
                error: None,
                updated_at_ms: now,
            },
        )?;
    }
    Ok(row)
}

/// Spec section 7 steps 1-2: flip running->interrupted; repair rows from
/// manifests (manifest wins). Returns (flipped, repaired) for the startup log.
///
/// Manifest artifact paths are rejected during `ChainJobManifest::read_from_dir`
/// before any later startup/resume path joins them to the job directory. Keep
/// that rejection-before-join ordering; it is the traversal boundary.
pub fn startup_reconcile(db: &MetadataDb, jobs_root: &Path) -> anyhow::Result<(usize, usize)> {
    let now = now_ms_i64();
    let running = chain_jobs::jobs_in_state(db, ChainJobState::Running)?;
    let mut flipped = 0;
    for row in running {
        if chain_jobs::update_job_state(
            db,
            &row.id,
            ChainJobState::Interrupted,
            Some("server restarted while chain job was running"),
            now,
        )? {
            flipped += 1;
        }
    }

    let mut repaired = 0;
    for row in chain_jobs::list_jobs(db)? {
        if row.state.is_terminal() {
            continue;
        }
        let job_dir = if row.job_dir.is_absolute() {
            row.job_dir.clone()
        } else {
            jobs_root.join(&row.job_dir)
        };
        let manifest = match ChainJobManifest::read_from_dir(&job_dir) {
            Ok(manifest) => manifest,
            Err(err) => {
                tracing::warn!(job_id = %row.id, "chain job manifest missing/unreadable during reconcile: {err:#}");
                continue;
            }
        };

        let mut changed = false;
        for stage in &manifest.stage_status {
            let db_stage = ChainJobStageRow {
                job_id: row.id.clone(),
                stage_idx: stage.idx,
                state: stage.state,
                seed: stage.seed,
                frames_emitted: stage.frames_emitted,
                generation_time_ms: stage.generation_time_ms,
                segment_rel_path: stage.segment.clone(),
                error: stage.error.clone(),
                updated_at_ms: now,
            };
            chain_jobs::upsert_stage(db, &db_stage)?;
        }

        let (state, current_stage, mut error) = manifest_index_state(&manifest, row.state);
        if state == ChainJobState::Interrupted && error.is_none() {
            error = row.error.clone();
        }
        let finalized_at = row.finalized_at_ms.or_else(|| {
            manifest
                .finalizes
                .last()
                .and_then(|record| i64::try_from(record.at_unix_ms).ok())
        });
        if row.state != state
            || row.current_stage != current_stage
            || row.error.as_deref() != error.as_deref()
            || row.finalized_at_ms != finalized_at
        {
            changed = chain_jobs::repair_job_from_manifest(
                db,
                &row.id,
                state,
                current_stage,
                error.as_deref(),
                now,
                finalized_at,
            )?;
        }
        if changed {
            repaired += 1;
        }
    }

    Ok((flipped, repaired))
}

/// One GC pass (daily tick + POST /api/chain-jobs/gc). Orphan predicate per
/// P0 rev3: ephemeral && state != running && !is_claimed, settled gated by
/// EPHEMERAL_GRACE_SECS; non-ephemeral completed jobs older than
/// chain.jobs_artifact_ttl_days lose stages/ artifact dirs only (final/ +
/// manifest + rows retained). EVERY delete goes through the guarded-delete
/// discipline: per-job mutation lock + state-predicated row ops (inv 17 /
/// P0 note 1). Sweep of an ephemeral removes dir + row (+ bus entry).
pub(crate) fn run_gc_pass(
    deps: &RunnerDeps,
    ttl_days: i64,
    now_ms: i64,
) -> anyhow::Result<GcOutcome> {
    let db = deps
        .db
        .as_ref()
        .as_ref()
        .ok_or_else(|| anyhow!("chain job GC invoked without metadata DB"))?;
    let ttl_ms = ttl_days.max(0).saturating_mul(86_400_000);
    let grace_ms = (EPHEMERAL_GRACE_SECS as i64).saturating_mul(1_000);
    let mut outcome = GcOutcome {
        swept_ephemeral_jobs: 0,
        pruned_artifact_dirs: 0,
    };

    for row in chain_jobs::list_jobs(db)? {
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).ok();
        if manifest.as_ref().is_some_and(|manifest| manifest.ephemeral) {
            if deps.claims.is_claimed(&row.id) {
                continue;
            }
            if settled(row.state) && now_ms.saturating_sub(row.updated_at_ms) < grace_ms {
                continue;
            }
            let remove_lock = {
                let _guard = deps.job_locks.blocking_lock(&row.id);
                let current = match chain_jobs::get_job(db, &row.id)? {
                    Some(current) => current,
                    None => {
                        continue;
                    }
                };
                let within_grace = settled(current.state)
                    && now_ms.saturating_sub(current.updated_at_ms) < grace_ms;
                if current.state == ChainJobState::Running
                    || deps.claims.is_claimed(&row.id)
                    || within_grace
                {
                    false
                } else {
                    if current.job_dir.exists() {
                        std::fs::remove_dir_all(&current.job_dir).with_context(|| {
                            format!(
                                "removing ephemeral chain job '{}'",
                                current.job_dir.display()
                            )
                        })?;
                    }
                    chain_jobs::delete_job_not_running(db, &current.id)?
                }
            };
            if remove_lock {
                outcome.swept_ephemeral_jobs += 1;
                deps.cancel.unregister(&row.id);
                deps.events.remove(&row.id);
                deps.job_locks.remove(&row.id);
            }
            continue;
        }

        if row.state == ChainJobState::Completed
            && ttl_ms >= 0
            && now_ms.saturating_sub(row.updated_at_ms) >= ttl_ms
        {
            let stages_dir = row.job_dir.join(STAGES_DIR);
            if stages_dir.exists() {
                let _guard = deps.job_locks.blocking_lock(&row.id);
                let Some(current) = chain_jobs::get_job(db, &row.id)? else {
                    continue;
                };
                if current.state == ChainJobState::Completed && stages_dir.exists() {
                    std::fs::remove_dir_all(&stages_dir).with_context(|| {
                        format!("pruning chain job stages '{}'", stages_dir.display())
                    })?;
                    outcome.pruned_artifact_dirs += 1;
                }
            }
        }
    }

    Ok(outcome)
}

/// Startup sweep (unconditional on claims — none survive restart): all
/// non-running ephemerals removed. Ordering pin: lib.rs calls
/// startup_reconcile → THIS → spawn_runner, strictly sequential.
pub(crate) fn startup_gc_sweep(db: &MetadataDb, jobs_root: &Path) -> anyhow::Result<GcOutcome> {
    let mut outcome = GcOutcome {
        swept_ephemeral_jobs: 0,
        pruned_artifact_dirs: 0,
    };
    for row in chain_jobs::list_jobs(db)? {
        let job_dir = if row.job_dir.is_absolute() {
            row.job_dir.clone()
        } else {
            jobs_root.join(&row.job_dir)
        };
        let Ok(manifest) = ChainJobManifest::read_from_dir(&job_dir) else {
            continue;
        };
        if manifest.ephemeral && row.state != ChainJobState::Running {
            if job_dir.exists() {
                std::fs::remove_dir_all(&job_dir).with_context(|| {
                    format!(
                        "removing startup ephemeral chain job '{}'",
                        job_dir.display()
                    )
                })?;
            }
            if chain_jobs::delete_job_not_running(db, &row.id)? {
                outcome.swept_ephemeral_jobs += 1;
            }
        }
    }
    Ok(outcome)
}

async fn run_loop(
    deps: Arc<RunnerDeps>,
    mut kick_rx: tokio::sync::mpsc::UnboundedReceiver<RunnerCmd>,
) {
    let mut daily = tokio::time::interval(Duration::from_secs(24 * 60 * 60));
    daily.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    daily.tick().await;
    loop {
        while let Ok(cmd) = kick_rx.try_recv() {
            if !handle_runner_cmd(deps.clone(), cmd).await {
                return;
            }
        }

        let mut ran_job = false;
        if let Some(db) = deps.db.as_ref() {
            let job = match next_queued_job(db) {
                Ok(Some(job)) => Some(job),
                Ok(None) => None,
                Err(err) => {
                    tracing::warn!("chain-job queued lookup failed: {err:#}");
                    None
                }
            };
            if let Some(job) = job {
                let deps_for_job = deps.clone();
                let job_id = job.id.clone();
                let start_stage = job.current_stage;
                let join = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
                    if !claim_for_execution(&deps_for_job, &job)? {
                        return Ok(false);
                    }
                    execute_job(&deps_for_job, &job, start_stage)?;
                    Ok(true)
                })
                .await;
                match join {
                    Ok(Ok(claimed)) => {
                        if claimed {
                            ran_job = true;
                        }
                    }
                    Ok(Err(err)) => {
                        tracing::warn!(job_id = %job_id, "chain job execution failed: {err:#}");
                        break;
                    }
                    Err(err) => {
                        tracing::warn!(job_id = %job_id, "chain job task join failed: {err}");
                        break;
                    }
                }
            }
        }

        if ran_job {
            tokio::task::yield_now().await;
            continue;
        }

        tokio::select! {
            maybe_cmd = kick_rx.recv() => {
                let Some(cmd) = maybe_cmd else { break };
                if !handle_runner_cmd(deps.clone(), cmd).await {
                    break;
                }
            }
            _ = daily.tick() => {
                if let Err(err) = run_gc_for_runner(deps.clone()).await {
                    tracing::warn!("daily chain job GC failed: {err}");
                }
            }
        }
    }
}

async fn handle_runner_cmd(deps: Arc<RunnerDeps>, cmd: RunnerCmd) -> bool {
    match cmd {
        RunnerCmd::Kick => true,
        RunnerCmd::Gc { reply } => {
            let result = run_gc_for_runner(deps).await;
            let _ = reply.send(result);
            true
        }
    }
}

async fn run_gc_for_runner(deps: Arc<RunnerDeps>) -> std::result::Result<GcOutcome, String> {
    tokio::task::spawn_blocking(move || {
        let db = deps
            .db
            .as_ref()
            .as_ref()
            .ok_or_else(|| "chain job GC invoked without metadata DB".to_string())?;
        let ttl_days = settings::Settings::new(db)
            .get_int(settings::CHAIN_JOBS_ARTIFACT_TTL_DAYS)
            .map_err(|e| format!("{e:#}"))?
            .unwrap_or(settings::CHAIN_JOBS_ARTIFACT_TTL_DEFAULT);
        run_gc_pass(&deps, ttl_days, now_ms_i64()).map_err(|e| format!("{e:#}"))
    })
    .await
    .map_err(|e| format!("chain job GC task failed: {e}"))?
}

/// FIFO by created_at among state=queued.
fn next_queued_job(db: &MetadataDb) -> anyhow::Result<Option<ChainJobRow>> {
    chain_jobs::next_queued_job(db)
}

fn claim_for_execution(deps: &RunnerDeps, job: &ChainJobRow) -> anyhow::Result<bool> {
    let db = deps
        .db
        .as_ref()
        .as_ref()
        .ok_or_else(|| anyhow!("chain job runner invoked without metadata DB"))?;
    let _guard = deps.job_locks.blocking_lock(&job.id);
    let claimed = chain_jobs::claim_job(db, &job.id)?;
    if claimed {
        deps.cancel.register(&job.id);
    }
    Ok(claimed)
}

fn execute_job(deps: &RunnerDeps, job: &ChainJobRow, start_stage: u32) -> anyhow::Result<()> {
    let db = deps
        .db
        .as_ref()
        .as_ref()
        .ok_or_else(|| anyhow!("chain job runner invoked without metadata DB"))?;
    let mut terminal = false;

    let current = chain_jobs::get_job(db, &job.id)?
        .ok_or_else(|| anyhow!("chain job '{}' disappeared before execution", job.id))?;
    match current.state {
        ChainJobState::Queued => {
            if !claim_for_execution(deps, &current)? {
                return Ok(());
            }
        }
        ChainJobState::Running => {
            deps.cancel.register(&job.id);
        }
        other => bail!(
            "chain job '{}' is not executable from state {}",
            job.id,
            other.as_str()
        ),
    }
    let run_result = (|| -> anyhow::Result<()> {
        let mut manifest = {
            let _guard = deps.job_locks.blocking_lock(&job.id);
            match ChainJobManifest::read_from_dir(&job.job_dir) {
                Ok(manifest) => manifest,
                Err(err) => {
                    fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                    terminal = true;
                    return Ok(());
                }
            }
        };
        let layout = JobDirLayout::new(job.job_dir.clone());
        let mut effective = {
            let _guard = deps.job_locks.blocking_lock(&job.id);
            match effective_request(&manifest) {
                Ok(effective) => effective,
                Err(err) => {
                    fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                    terminal = true;
                    return Ok(());
                }
            }
        };
        let mut carry: Option<ChainTail> = None;

        deps.events.publish(
            &job.id,
            ChainJobEvent::StateChanged {
                state: ChainJobState::Running,
                error: None,
            },
        );

        let mut stage_idx = first_incomplete_stage(&manifest).unwrap_or(start_stage);
        if stage_idx < start_stage {
            stage_idx = start_stage;
        }

        while stage_idx < manifest.stage_status.len() as u32 {
            if deps.cancel.is_cancelled(&job.id) {
                if let Err(err) = set_cancelled(db, deps, &job.id) {
                    fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                }
                terminal = true;
                return Ok(());
            }

            if manifest.stage_status[stage_idx as usize].state == StageState::Completed {
                match carry_after_skipping_completed_stage(&layout, &manifest, stage_idx) {
                    Ok(next_carry) => {
                        carry = next_carry;
                        stage_idx += 1;
                        continue;
                    }
                    Err(err) => {
                        let fail_stage = (stage_idx + 1)
                            .min(manifest.stage_status.len().saturating_sub(1) as u32);
                        fail_stage_job(
                            db,
                            deps,
                            &mut manifest,
                            &layout,
                            &job.id,
                            fail_stage,
                            format!("{err:#}"),
                        )?;
                        terminal = true;
                        return Ok(());
                    }
                }
            }

            let stage_dir = layout.stage_dir(stage_idx);
            if stage_dir.exists() {
                if let Err(err) = std::fs::remove_dir_all(&stage_dir).with_context(|| {
                    format!(
                        "removing partial chain stage directory '{}'",
                        stage_dir.display()
                    )
                }) {
                    fail_stage_job(
                        db,
                        deps,
                        &mut manifest,
                        &layout,
                        &job.id,
                        stage_idx,
                        format!("{err:#}"),
                    )?;
                    terminal = true;
                    return Ok(());
                }
            }

            if carry.is_none() {
                match resume_carry_from_disk(layout.root(), &manifest, stage_idx) {
                    Ok(next_carry) => carry = next_carry,
                    Err(err) => {
                        fail_stage_job(
                            db,
                            deps,
                            &mut manifest,
                            &layout,
                            &job.id,
                            stage_idx,
                            format!("{err:#}"),
                        )?;
                        terminal = true;
                        return Ok(());
                    }
                }
            }

            deps.events
                .publish(&job.id, ChainJobEvent::StageStart { stage_idx });
            if let Err(err) = mark_stage_running(db, &job.id, &manifest, stage_idx) {
                fail_stage_job(
                    db,
                    deps,
                    &mut manifest,
                    &layout,
                    &job.id,
                    stage_idx,
                    format!("{err:#}"),
                )?;
                terminal = true;
                return Ok(());
            }

            let stage_carry = match effective.stages[stage_idx as usize].transition {
                TransitionMode::Smooth => carry.as_ref(),
                TransitionMode::Cut | TransitionMode::Fade => None,
            };
            let render_outcome = match execute_stage(deps, job, &manifest, stage_idx, stage_carry) {
                Ok(outcome) => outcome,
                Err(err) => {
                    let error = format!("{err:#}");
                    {
                        let _guard = deps.job_locks.blocking_lock(&job.id);
                        let _ =
                            mark_manifest_stage_failed(&mut manifest, &layout, stage_idx, &error);
                    }
                    fail_job(db, deps, &job.id, Some(stage_idx), error)?;
                    terminal = true;
                    return Ok(());
                }
            };
            match render_outcome {
                StageRenderOutcome::Cancelled => {
                    set_cancelled(db, deps, &job.id)?;
                    terminal = true;
                    return Ok(());
                }
                StageRenderOutcome::Done(outcome) => {
                    let stage_artifacts = {
                        let _guard = deps.job_locks.blocking_lock(&job.id);
                        write_stage_artifacts(
                            &layout,
                            &mut manifest,
                            stage_idx,
                            &outcome,
                            &effective,
                        )
                    };
                    let paths = match stage_artifacts {
                        Ok(paths) => paths,
                        Err(err) => {
                            let error = format!("{err:#}");
                            {
                                let _guard = deps.job_locks.blocking_lock(&job.id);
                                let _ = mark_manifest_stage_failed(
                                    &mut manifest,
                                    &layout,
                                    stage_idx,
                                    &error,
                                );
                            }
                            fail_job(db, deps, &job.id, Some(stage_idx), error)?;
                            terminal = true;
                            return Ok(());
                        }
                    };

                    let now = now_ms_i64();
                    let status = manifest.stage_status[stage_idx as usize].clone();
                    if let Err(err) = chain_jobs::upsert_stage(
                        db,
                        &ChainJobStageRow {
                            job_id: job.id.clone(),
                            stage_idx,
                            state: StageState::Completed,
                            seed: status.seed,
                            frames_emitted: status.frames_emitted,
                            generation_time_ms: status.generation_time_ms,
                            segment_rel_path: Some(paths.segment_rel.clone()),
                            error: None,
                            updated_at_ms: now,
                        },
                    ) {
                        fail_stage_job(
                            db,
                            deps,
                            &mut manifest,
                            &layout,
                            &job.id,
                            stage_idx,
                            format!("{err:#}"),
                        )?;
                        terminal = true;
                        return Ok(());
                    }
                    if let Err(err) = chain_jobs::set_current_stage(db, &job.id, stage_idx + 1, now)
                    {
                        fail_stage_job(
                            db,
                            deps,
                            &mut manifest,
                            &layout,
                            &job.id,
                            stage_idx,
                            format!("{err:#}"),
                        )?;
                        terminal = true;
                        return Ok(());
                    }

                    if let Err(err) =
                        maybe_reencode_next_after_fade(&layout, &manifest, stage_idx, &effective)
                    {
                        let fail_stage = (stage_idx + 1)
                            .min(manifest.stage_status.len().saturating_sub(1) as u32);
                        fail_stage_job(
                            db,
                            deps,
                            &mut manifest,
                            &layout,
                            &job.id,
                            fail_stage,
                            format!("{err:#}"),
                        )?;
                        terminal = true;
                        return Ok(());
                    }

                    deps.events.publish(
                        &job.id,
                        ChainJobEvent::StageDone {
                            stage_idx,
                            frames_emitted: status.frames_emitted.unwrap_or(0),
                            has_preview: paths.preview_written,
                        },
                    );

                    carry = Some(outcome.tail);

                    publish_yield_if_contended(deps, &job.id);
                }
            }
            stage_idx += 1;
            effective = match effective_request(&manifest) {
                Ok(effective) => effective,
                Err(err) => {
                    fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                    terminal = true;
                    return Ok(());
                }
            };
        }

        let output = match finalize_job(deps, job, &mut manifest) {
            Ok(output) => output,
            Err(err) => {
                fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                terminal = true;
                return Ok(());
            }
        };
        let take = manifest.finalizes.len() as u32;
        deps.events
            .publish(&job.id, ChainJobEvent::Finalized { output, take });
        let completed = match chain_jobs::try_transition(
            db,
            &job.id,
            &[ChainJobState::Running],
            ChainJobState::Completed,
            None,
            now_ms_i64(),
        ) {
            Ok(completed) => completed,
            Err(err) => {
                fail_job(db, deps, &job.id, None, format!("{err:#}"))?;
                terminal = true;
                return Ok(());
            }
        };
        if !completed {
            tracing::warn!(job_id = %job.id, "chain job completed but running->completed CAS lost");
            terminal = true;
            return Ok(());
        }
        deps.events.publish_then_remove(
            &job.id,
            ChainJobEvent::StateChanged {
                state: ChainJobState::Completed,
                error: None,
            },
        );
        deps.cancel.unregister(&job.id);
        terminal = true;
        Ok(())
    })();

    deps.cancel.unregister(&job.id);
    if terminal {
        deps.events.remove(&job.id);
    }
    run_result
}

fn execute_stage(
    deps: &RunnerDeps,
    job: &ChainJobRow,
    manifest: &ChainJobManifest,
    stage_idx: u32,
    carry: Option<&ChainTail>,
) -> anyhow::Result<StageRenderOutcome> {
    let effective = effective_request(manifest)?;
    let stage = effective
        .stages
        .get(stage_idx as usize)
        .ok_or_else(|| anyhow!("stage index {stage_idx} out of bounds"))?;
    let stage_seed = manifest.stage_status[stage_idx as usize].seed;
    let stage_req = build_stage_generate_request(stage, &effective, stage_seed, stage_idx as usize);
    let executor = deps.executor.clone();
    let model = effective.model.clone();
    let carry_owned = carry.cloned();
    let job_id = job.id.clone();
    let events = deps.events.clone();
    let cancel = deps.cancel.clone();
    let motion_tail_frames = effective.motion_tail_frames;
    let progress: Box<dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync> =
        Box::new(move |step, total| {
            events.publish(
                &job_id,
                ChainJobEvent::DenoiseStep {
                    stage_idx,
                    step,
                    total,
                },
            );
            if cancel.is_cancelled(&job_id) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle
            .block_on(tokio::task::spawn_blocking(move || {
                executor.render_stage(
                    &model,
                    &stage_req,
                    carry_owned.as_ref(),
                    motion_tail_frames,
                    progress.as_ref(),
                )
            }))
            .map_err(|e| anyhow!("chain stage task failed: {e}"))?,
        Err(_) => deps.executor.render_stage(
            &model,
            &stage_req,
            carry,
            motion_tail_frames,
            progress.as_ref(),
        ),
    }
}

fn write_stage_artifacts(
    layout: &JobDirLayout,
    manifest: &mut ChainJobManifest,
    stage_idx: u32,
    outcome: &StageOutcome,
    effective: &ChainRequest,
) -> anyhow::Result<StageArtifactPaths> {
    layout.ensure_stage_dirs(stage_idx)?;
    let stage = effective
        .stages
        .get(stage_idx as usize)
        .ok_or_else(|| anyhow!("stage index {stage_idx} out of bounds"))?;
    let next_transition = effective
        .stages
        .get(stage_idx as usize + 1)
        .map(|s| s.transition);
    let incoming = stage.transition;
    let mut frames = outcome.frames.clone();
    let mut audio = outcome.audio.clone();

    if matches!(incoming, TransitionMode::Smooth) && stage_idx > 0 {
        let drop = effective.motion_tail_frames as usize;
        if frames.len() < drop {
            bail!(
                "stage {stage_idx} emitted {} frames, cannot drop {drop} smooth carry frames",
                frames.len()
            );
        }
        frames.drain(0..drop);
        trim_audio_front(&mut audio, effective.motion_tail_frames, effective.fps)?;
    }

    if matches!(incoming, TransitionMode::Fade) && stage_idx > 0 {
        let fade_len = stage.fade_frames.unwrap_or(DEFAULT_FADE_FRAMES);
        let n = fade_len as usize;
        if frames.len() < n {
            bail!(
                "stage {stage_idx} emitted {} frames, cannot apply incoming fade_len {n}",
                frames.len()
            );
        }
        write_frames_to_dir(&layout.boundary_in_dir(stage_idx), &frames[..n])?;
        let previous_out = read_frames_from_dir(&layout.boundary_out_dir(stage_idx - 1), n)?;
        let blended = fade_boundary(&previous_out, &frames, fade_len);
        for (idx, frame) in blended.into_iter().enumerate() {
            frames[idx] = frame;
        }
        blend_audio_front_from_previous_boundary(
            layout,
            stage_idx,
            &mut audio,
            fade_len,
            effective.fps,
        )?;
    }

    if matches!(next_transition, Some(TransitionMode::Fade)) {
        let next = &effective.stages[stage_idx as usize + 1];
        let fade_len = next.fade_frames.unwrap_or(DEFAULT_FADE_FRAMES);
        let n = fade_len as usize;
        if frames.len() < n {
            bail!(
                "stage {stage_idx} emitted {} frames, cannot reserve outgoing fade_len {n}",
                frames.len()
            );
        }
        write_frames_to_dir(
            &layout.boundary_out_dir(stage_idx),
            &frames[frames.len() - n..],
        )?;
        write_audio_boundary_out(layout, stage_idx, &audio, fade_len, effective.fps)?;
        frames.truncate(frames.len() - n);
        trim_audio_back(&mut audio, fade_len, effective.fps)?;
    }

    if frames.is_empty() {
        bail!("stage {stage_idx} produced no frames after boundary handling");
    }

    let segment_bytes = video_enc::encode_mp4(&frames, effective.fps)
        .with_context(|| format!("encoding chain stage {stage_idx} segment"))?;
    write_file(&layout.segment_path(stage_idx), &segment_bytes)?;

    let tail_frames = if matches!(next_transition, Some(TransitionMode::Smooth)) {
        write_frames_to_dir(&layout.tail_dir(stage_idx), &outcome.tail.tail_rgb_frames)?;
        outcome.tail.frames
    } else {
        0
    };

    let audio_rel = if let Some(track) = audio.as_ref() {
        write_audio_sidecar(&layout.audio_path(stage_idx), track)?;
        Some(layout.audio_rel(stage_idx))
    } else {
        None
    };

    write_preview_jpeg(&layout.preview_path(stage_idx), frames.last().unwrap())?;

    let status = manifest
        .stage_status
        .get_mut(stage_idx as usize)
        .ok_or_else(|| anyhow!("manifest missing status for stage {stage_idx}"))?;
    status.state = StageState::Completed;
    status.frames_emitted = Some(frames.len() as u32);
    status.generation_time_ms = Some(outcome.generation_time_ms);
    status.segment = Some(layout.segment_rel(stage_idx));
    status.tail_frames = Some(tail_frames);
    status.audio = audio_rel.clone();
    status.error = None;

    manifest.write_atomic(layout.root())?;

    Ok(StageArtifactPaths {
        segment_rel: layout.segment_rel(stage_idx),
        preview_written: true,
    })
}

fn resume_carry_from_disk(
    job_dir: &Path,
    manifest: &ChainJobManifest,
    stage_idx: u32,
) -> anyhow::Result<Option<ChainTail>> {
    if stage_idx == 0 {
        return Ok(None);
    }
    let effective = effective_request(manifest)?;
    let Some(stage) = effective.stages.get(stage_idx as usize) else {
        bail!("stage index {stage_idx} out of bounds for resume carry");
    };
    if !matches!(stage.transition, TransitionMode::Smooth) {
        return Ok(None);
    }
    let prev_idx = stage_idx - 1;
    let prev_status = manifest
        .stage_status
        .get(prev_idx as usize)
        .ok_or_else(|| anyhow!("manifest missing previous stage {prev_idx}"))?;
    if let Some(segment) = &prev_status.segment {
        safe_join_manifest_rel(job_dir, segment)?;
    }
    let expected = prev_status.tail_frames.unwrap_or(0);
    if expected == 0 {
        return Ok(None);
    }
    let tail_dir = JobDirLayout::new(job_dir.to_path_buf()).tail_dir(prev_idx);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&tail_dir)
        .with_context(|| format!("reading chain tail directory '{}'", tail_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    if paths.len() != expected as usize {
        bail!(
            "chain tail for stage {prev_idx} has {} PNG(s), expected {expected}",
            paths.len()
        );
    }
    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        frames.push(
            image::open(&path)
                .with_context(|| format!("decoding chain tail PNG '{}'", path.display()))?
                .to_rgb8(),
        );
    }
    Ok(Some(ChainTail {
        frames: expected,
        tail_rgb_frames: frames,
    }))
}

/// Finalize by joining manifest-relative segment/audio paths only after
/// manifest validation has rejected absolute paths and `..` components. The
/// rejection-before-join contract is load-bearing for malicious resume data.
fn finalize_job(
    deps: &RunnerDeps,
    job: &ChainJobRow,
    manifest: &mut ChainJobManifest,
) -> anyhow::Result<String> {
    let db = deps
        .db
        .as_ref()
        .as_ref()
        .ok_or_else(|| anyhow!("chain job finalizer invoked without metadata DB"))?;
    let effective = effective_request(manifest)?;
    let layout = JobDirLayout::new(job.job_dir.clone());
    let total_frames: u32 = manifest
        .stage_status
        .iter()
        .map(|stage| stage.frames_emitted.unwrap_or(0))
        .sum();
    deps.events
        .publish(&job.id, ChainJobEvent::Finalizing { total_frames });

    let mut encoder: Option<video_enc::Mp4StreamEncoder> = None;
    let mut audio_samples = Vec::new();
    let mut audio_format: Option<(u32, u16)> = None;
    let mut frame_count = 0u32;

    for stage in &manifest.stage_status {
        if stage.state != StageState::Completed {
            bail!(
                "cannot finalize chain job with non-completed stage {}",
                stage.idx
            );
        }
        let segment = stage
            .segment
            .as_ref()
            .ok_or_else(|| anyhow!("completed stage {} has no segment path", stage.idx))?;
        let segment_path = safe_join_manifest_rel(layout.root(), segment)?;
        let (metadata, frames) =
            mold_inference::ltx2::media::decode_video_frames_from_path(&segment_path)
                .with_context(|| format!("decoding chain segment '{}'", segment_path.display()))?;
        if encoder.is_none() {
            encoder = Some(video_enc::Mp4StreamEncoder::new(
                metadata.width,
                metadata.height,
                effective.fps,
            )?);
        }
        let enc = encoder.as_mut().unwrap();
        for frame in frames {
            enc.push(&frame)?;
            frame_count += 1;
        }
        if let Some(audio_rel) = stage.audio.as_ref() {
            let audio_path = safe_join_manifest_rel(layout.root(), audio_rel)?;
            let track = read_audio_sidecar(&audio_path)?;
            match audio_format {
                None => audio_format = Some((track.sample_rate, track.channels)),
                Some((sample_rate, channels))
                    if sample_rate == track.sample_rate && channels == track.channels => {}
                Some((sample_rate, channels)) => {
                    bail!(
                        "stage {} audio format {} Hz/{} ch does not match previous {} Hz/{} ch",
                        stage.idx,
                        track.sample_rate,
                        track.channels,
                        sample_rate,
                        channels
                    );
                }
            }
            audio_samples.extend_from_slice(&track.interleaved_samples);
        }
    }

    let encoder = encoder.ok_or_else(|| anyhow!("cannot finalize chain job with no frames"))?;
    let video_bytes = encoder.finish()?;
    let video_bytes = if !audio_samples.is_empty() {
        #[cfg(feature = "mp4")]
        {
            let (sample_rate, channels) = audio_format.expect("samples imply format");
            mold_inference::ltx2::media::attach_aac_track_to_mp4_bytes(
                &video_bytes,
                &audio_samples,
                sample_rate,
                channels,
            )?
        }
        #[cfg(not(feature = "mp4"))]
        {
            bail!("chain job finalization with audio requires the mp4 feature for AAC muxing");
        }
    } else {
        video_bytes
    };

    let take = manifest.finalizes.len() as u32 + 1;
    let output_path = layout.final_output_path(take);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating chain final directory '{}'", parent.display()))?;
    }
    write_file(&output_path, &video_bytes)?;

    if !manifest.ephemeral {
        if let Some(output_dir) = deps.output_dir.as_ref() {
            let metadata = output_metadata_for_chain(&effective, frame_count);
            save_video_to_dir(
                output_dir,
                &video_bytes,
                &[],
                OutputFormat::Mp4,
                &effective.model,
                &metadata,
                None,
                Some(db),
            );
        }
    }

    let now = now_ms_u64();
    let output = format!("final/output-{take}.mp4");
    {
        let _guard = deps.job_locks.blocking_lock(&job.id);
        manifest.finalizes.push(FinalizeRecord {
            output: output.clone(),
            at_unix_ms: now,
            stage_seeds: manifest
                .stage_status
                .iter()
                .map(|stage| stage.seed)
                .collect(),
        });
        manifest.write_atomic(layout.root())?;
        chain_jobs::set_finalized_at(db, &job.id, i64::try_from(now).unwrap_or(i64::MAX))?;
    }
    Ok(output)
}

pub fn apply_retake(
    db: &MetadataDb,
    jobs_root: &Path,
    job_id: &str,
    req: &RetakeRequest,
) -> anyhow::Result<ChainJobRow> {
    let job = chain_jobs::get_job(db, job_id)?.ok_or_else(|| anyhow!("chain job not found"))?;
    if job.state == ChainJobState::Running {
        bail!("CHAIN_JOB_RUNNING");
    }
    let allowed_from = [
        ChainJobState::Interrupted,
        ChainJobState::Failed,
        ChainJobState::Cancelled,
        ChainJobState::Completed,
    ];
    if !allowed_from.contains(&job.state) {
        bail!(
            "chain job is not retakeable from state {}",
            job.state.as_str()
        );
    }
    let job_dir = if job.job_dir.is_absolute() {
        job.job_dir.clone()
    } else {
        jobs_root.join(&job.job_dir)
    };
    let mut manifest = ChainJobManifest::read_from_dir(&job_dir)?;
    let effective = effective_request(&manifest)?;
    let stage_idx = req.stage_idx as usize;
    if stage_idx >= manifest.stage_status.len() {
        bail!("stage_idx {} out of bounds", req.stage_idx);
    }
    if req.mode == RetakeMode::Splice && stage_idx + 1 < effective.stages.len() {
        let next_transition = effective.stages[stage_idx + 1].transition;
        if matches!(next_transition, TransitionMode::Smooth) {
            bail!("RETAKE_SPLICE_REQUIRES_CUT_OR_FADE");
        }
    }

    let now = now_ms_u64();
    let base_seed = effective.seed.unwrap_or(0);
    let old_seed = manifest.stage_status[stage_idx].seed;
    let new_seed = req
        .seed_offset
        .map(|offset| effective_stage_seed(base_seed, Some(offset)))
        .unwrap_or(old_seed);
    let old_prompt = effective.stages[stage_idx].prompt.clone();
    let new_prompt = req.prompt.clone().unwrap_or_else(|| old_prompt.clone());
    let now_i64 = i64::try_from(now).unwrap_or(i64::MAX);

    if !chain_jobs::try_transition(
        db,
        job_id,
        &allowed_from,
        ChainJobState::Queued,
        None,
        now_i64,
    )? {
        let observed = chain_jobs::get_job(db, job_id)?
            .map(|row| row.state.as_str().to_string())
            .unwrap_or_else(|| "missing".to_string());
        bail!("chain job is not retakeable from current state {observed}");
    }

    manifest.retakes.push(RetakeAmendment {
        stage_idx: req.stage_idx,
        mode: req.mode,
        old_seed,
        new_seed,
        old_prompt: (new_prompt != old_prompt).then_some(old_prompt),
        new_prompt: req.prompt.clone(),
        at_unix_ms: now,
    });
    manifest.stage_status[stage_idx].seed = new_seed;

    let reset_end = match req.mode {
        RetakeMode::Cascade => manifest.stage_status.len(),
        RetakeMode::Splice => stage_idx + 1,
    };
    for idx in stage_idx..reset_end {
        let status = &mut manifest.stage_status[idx];
        status.state = StageState::Pending;
        status.frames_emitted = None;
        status.generation_time_ms = None;
        status.segment = None;
        status.tail_frames = None;
        status.audio = None;
        status.error = None;
        let stage_dir = JobDirLayout::new(job_dir.clone()).stage_dir(idx as u32);
        if stage_dir.exists() {
            std::fs::remove_dir_all(&stage_dir).with_context(|| {
                format!(
                    "removing reset chain stage directory '{}'",
                    stage_dir.display()
                )
            })?;
        }
    }
    manifest.write_atomic(&job_dir)?;

    match req.mode {
        RetakeMode::Cascade => {
            chain_jobs::reset_stages_from(db, job_id, req.stage_idx, now_i64)?;
        }
        RetakeMode::Splice => {
            chain_jobs::reset_one_stage(db, job_id, req.stage_idx, now_i64)?;
        }
    }
    for idx in stage_idx..reset_end {
        chain_jobs::upsert_stage(
            db,
            &ChainJobStageRow {
                job_id: job_id.to_string(),
                stage_idx: idx as u32,
                state: StageState::Pending,
                seed: manifest.stage_status[idx].seed,
                frames_emitted: None,
                generation_time_ms: None,
                segment_rel_path: None,
                error: None,
                updated_at_ms: now_i64,
            },
        )?;
    }
    chain_jobs::set_current_stage(db, job_id, req.stage_idx, now_i64)?;

    chain_jobs::get_job(db, job_id)?.ok_or_else(|| anyhow!("chain job disappeared after retake"))
}

pub struct ProductionStageExecutor {
    gpu_pool: Arc<GpuPool>,
    config: mold_core::Config,
}

impl ProductionStageExecutor {
    pub fn new(gpu_pool: Arc<GpuPool>, config: mold_core::Config) -> Self {
        Self { gpu_pool, config }
    }
}

impl StageExecutor for ProductionStageExecutor {
    fn render_stage(
        &self,
        model: &str,
        stage_req: &GenerateRequest,
        carry: Option<&ChainTail>,
        motion_tail_frames: u32,
        progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
    ) -> anyhow::Result<StageRenderOutcome> {
        let worker = select_worker_for_stage(&self.gpu_pool, model)
            .ok_or_else(|| anyhow!("no GPU worker available for chain stage model '{model}'"))?;
        let _in_flight = WorkerInFlightGuard::new(worker.clone());
        let _active = WorkerActiveGenerationGuard::new(worker.clone(), model, &stage_req.prompt)?;
        let hint = model_manager::family_for_model_sync(model, &self.config).map(|family| {
            model_manager::ActivationHint {
                width: stage_req.width,
                height: stage_req.height,
                batch: 1,
                dtype_bytes: 2,
                family: mold_inference::device::activation_family_for(&family),
            }
        });
        let carry_owned = carry.cloned();
        let stage_req = stage_req.clone();
        let prep = gpu_worker::run_stage_blocking(
            &worker,
            model,
            &self.config,
            hint,
            move |engine| -> anyhow::Result<StageRenderOutcome> {
                let renderer = engine.as_chain_renderer().ok_or_else(|| {
                    anyhow!(
                        "model '{}' does not support chained video generation",
                        model
                    )
                })?;
                let mut cancelled = false;
                let mut stage_progress = |event: StageProgressEvent| match event {
                    StageProgressEvent::DenoiseStep { step, total } => {
                        if progress(step, total).is_break() {
                            cancelled = true;
                        }
                    }
                };
                let outcome = renderer.render_stage(
                    &stage_req,
                    carry_owned.as_ref(),
                    motion_tail_frames,
                    Some(&mut stage_progress),
                )?;
                if cancelled {
                    Ok(StageRenderOutcome::Cancelled)
                } else {
                    Ok(StageRenderOutcome::Done(outcome))
                }
            },
        )?;
        prep
    }
}

pub struct ProductionQueueProbe {
    queue: QueueHandle,
    gpu_pool: Arc<GpuPool>,
}

impl ProductionQueueProbe {
    pub fn new(queue: QueueHandle, gpu_pool: Arc<GpuPool>) -> Self {
        Self { queue, gpu_pool }
    }
}

impl QueueProbe for ProductionQueueProbe {
    fn small_jobs_waiting(&self) -> usize {
        self.queue.pending()
            + self
                .gpu_pool
                .workers
                .iter()
                .map(|worker| worker.in_flight.load(Ordering::SeqCst))
                .sum::<usize>()
    }
}

fn mark_stage_running(
    db: &MetadataDb,
    job_id: &str,
    manifest: &ChainJobManifest,
    stage_idx: u32,
) -> anyhow::Result<()> {
    let status = &manifest.stage_status[stage_idx as usize];
    chain_jobs::upsert_stage(
        db,
        &ChainJobStageRow {
            job_id: job_id.to_string(),
            stage_idx,
            state: StageState::Running,
            seed: status.seed,
            frames_emitted: None,
            generation_time_ms: None,
            segment_rel_path: None,
            error: None,
            updated_at_ms: now_ms_i64(),
        },
    )
}

fn mark_manifest_stage_failed(
    manifest: &mut ChainJobManifest,
    layout: &JobDirLayout,
    stage_idx: u32,
    error: &str,
) -> anyhow::Result<()> {
    let status = manifest
        .stage_status
        .get_mut(stage_idx as usize)
        .ok_or_else(|| anyhow!("manifest missing status for stage {stage_idx}"))?;
    status.state = StageState::Failed;
    status.error = Some(error.to_string());
    status.frames_emitted = None;
    status.generation_time_ms = None;
    status.segment = None;
    status.tail_frames = None;
    status.audio = None;
    manifest.write_atomic(layout.root())?;
    Ok(())
}

fn fail_job(
    db: &MetadataDb,
    deps: &RunnerDeps,
    job_id: &str,
    stage_idx: Option<u32>,
    error: String,
) -> anyhow::Result<()> {
    if let Some(stage_idx) = stage_idx {
        let stages = chain_jobs::stages_for_job(db, job_id)?;
        let seed = stages
            .iter()
            .find(|stage| stage.stage_idx == stage_idx)
            .map(|stage| stage.seed)
            .unwrap_or(0);
        chain_jobs::upsert_stage(
            db,
            &ChainJobStageRow {
                job_id: job_id.to_string(),
                stage_idx,
                state: StageState::Failed,
                seed,
                frames_emitted: None,
                generation_time_ms: None,
                segment_rel_path: None,
                error: Some(error.clone()),
                updated_at_ms: now_ms_i64(),
            },
        )?;
    }
    let changed = chain_jobs::try_transition(
        db,
        job_id,
        &[ChainJobState::Running],
        ChainJobState::Failed,
        Some(&error),
        now_ms_i64(),
    )?;
    if changed {
        deps.events.publish_then_remove(
            job_id,
            ChainJobEvent::StateChanged {
                state: ChainJobState::Failed,
                error: Some(error),
            },
        );
        deps.cancel.unregister(job_id);
    }
    Ok(())
}

fn fail_stage_job(
    db: &MetadataDb,
    deps: &RunnerDeps,
    manifest: &mut ChainJobManifest,
    layout: &JobDirLayout,
    job_id: &str,
    stage_idx: u32,
    error: String,
) -> anyhow::Result<()> {
    {
        let _guard = deps.job_locks.blocking_lock(job_id);
        let _ = mark_manifest_stage_failed(manifest, layout, stage_idx, &error);
    }
    fail_job(db, deps, job_id, Some(stage_idx), error)
}

fn set_cancelled(db: &MetadataDb, deps: &RunnerDeps, job_id: &str) -> anyhow::Result<()> {
    let changed = chain_jobs::try_transition(
        db,
        job_id,
        &[ChainJobState::Running],
        ChainJobState::Cancelled,
        None,
        now_ms_i64(),
    )?;
    if changed {
        deps.events.publish_then_remove(
            job_id,
            ChainJobEvent::StateChanged {
                state: ChainJobState::Cancelled,
                error: None,
            },
        );
        deps.cancel.unregister(job_id);
    }
    Ok(())
}

fn carry_after_skipping_completed_stage(
    layout: &JobDirLayout,
    manifest: &ChainJobManifest,
    completed_stage_idx: u32,
) -> anyhow::Result<Option<ChainTail>> {
    let next_idx = completed_stage_idx + 1;
    let Some(next_status) = manifest.stage_status.get(next_idx as usize) else {
        return Ok(None);
    };
    if next_status.state == StageState::Completed {
        return Ok(None);
    }
    resume_carry_from_disk(layout.root(), manifest, next_idx)
}

/// Publishes the inter-stage yield signal.
///
/// With per-stage GPU locks, the inter-stage gap is the yield: by the time
/// this runs, the stage render lock has already been released. The `Yielded`
/// event reports contention observed through `queue_probe`; it does not hold
/// the lock or sleep inside the chain runner.
fn publish_yield_if_contended(deps: &RunnerDeps, job_id: &str) {
    let pending_small_jobs = deps.queue_probe.small_jobs_waiting();
    if pending_small_jobs > 0 {
        deps.events
            .publish(job_id, ChainJobEvent::Yielded { pending_small_jobs });
    }
}

fn first_incomplete_stage(manifest: &ChainJobManifest) -> Option<u32> {
    manifest
        .stage_status
        .iter()
        .find(|stage| stage.state != StageState::Completed)
        .map(|stage| stage.idx)
}

fn manifest_index_state(
    manifest: &ChainJobManifest,
    existing_state: ChainJobState,
) -> (ChainJobState, u32, Option<String>) {
    if let Some(stage) = manifest
        .stage_status
        .iter()
        .find(|stage| stage.state == StageState::Failed)
    {
        return (
            ChainJobState::Failed,
            stage.idx,
            stage.error.clone().or_else(|| Some("stage failed".into())),
        );
    }
    let current_stage =
        first_incomplete_stage(manifest).unwrap_or(manifest.stage_status.len() as u32);
    if current_stage == manifest.stage_status.len() as u32 && !manifest.finalizes.is_empty() {
        (ChainJobState::Completed, current_stage, None)
    } else if existing_state == ChainJobState::Running {
        (
            ChainJobState::Interrupted,
            current_stage,
            Some("server restarted while chain job was running".into()),
        )
    } else {
        (existing_state, current_stage, None)
    }
}

pub(crate) fn effective_request(manifest: &ChainJobManifest) -> anyhow::Result<ChainRequest> {
    let mut request = manifest.request()?;
    for retake in &manifest.retakes {
        if let Some(stage) = request.stages.get_mut(retake.stage_idx as usize) {
            if let Some(prompt) = retake.new_prompt.as_ref() {
                stage.prompt = prompt.clone();
            }
            let base_seed = request.seed.unwrap_or(0);
            stage.seed_offset = Some(base_seed ^ retake.new_seed);
        }
    }
    Ok(request)
}

fn build_stage_generate_request(
    stage: &ChainStage,
    chain: &ChainRequest,
    stage_seed: u64,
    idx: usize,
) -> GenerateRequest {
    GenerateRequest {
        prompt: stage.prompt.clone(),
        negative_prompt: stage.negative_prompt.clone(),
        model: chain.model.clone(),
        width: chain.width,
        height: chain.height,
        steps: chain.steps,
        guidance: chain.guidance,
        seed: Some(stage_seed),
        batch_size: 1,
        output_format: Some(OutputFormat::Mp4),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        source_image: stage.source_image.clone(),
        edit_images: None,
        strength: if idx == 0 { chain.strength } else { 1.0 },
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: Some(stage.frames),
        fps: Some(chain.fps),
        upscale_model: None,
        gif_preview: false,
        enable_audio: Some(chain.enable_audio.unwrap_or(false)),
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: chain.placement.clone(),
    }
}

fn maybe_reencode_next_after_fade(
    layout: &JobDirLayout,
    manifest: &ChainJobManifest,
    stage_idx: u32,
    effective: &ChainRequest,
) -> anyhow::Result<()> {
    let next_idx = stage_idx + 1;
    let Some(next_stage) = effective.stages.get(next_idx as usize) else {
        return Ok(());
    };
    if !matches!(next_stage.transition, TransitionMode::Fade) {
        return Ok(());
    }
    let Some(next_status) = manifest.stage_status.get(next_idx as usize) else {
        return Ok(());
    };
    if next_status.state != StageState::Completed {
        return Ok(());
    }
    let fade_len = next_stage.fade_frames.unwrap_or(DEFAULT_FADE_FRAMES);
    let n = fade_len as usize;
    let next_segment = safe_join_manifest_rel(
        layout.root(),
        next_status
            .segment
            .as_ref()
            .ok_or_else(|| anyhow!("completed next stage {next_idx} has no segment"))?,
    )?;
    let (_metadata, mut frames) =
        mold_inference::ltx2::media::decode_video_frames_from_path(&next_segment)?;
    if frames.len() < n {
        bail!("next stage {next_idx} segment shorter than fade_len {n}");
    }
    let new_boundary_out = read_frames_from_dir(&layout.boundary_out_dir(stage_idx), n)?;
    let raw_boundary_in = read_frames_from_dir(&layout.boundary_in_dir(next_idx), n)?;
    let blended = fade_boundary(&new_boundary_out, &raw_boundary_in, fade_len);
    for (idx, frame) in blended.into_iter().enumerate() {
        frames[idx] = frame;
    }
    let bytes = video_enc::encode_mp4(&frames, effective.fps)?;
    write_file(&next_segment, &bytes)?;
    write_preview_jpeg(&layout.preview_path(next_idx), frames.last().unwrap())?;
    Ok(())
}

fn safe_join_manifest_rel(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(rel);
    if path.is_absolute()
        || rel.starts_with('/')
        || rel.starts_with('\\')
        || rel.as_bytes().get(0..3).is_some_and(|b| {
            b[0].is_ascii_alphabetic() && b[1] == b':' && matches!(b[2], b'/' | b'\\')
        })
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        || rel.split(['/', '\\']).any(|part| part == "..")
    {
        bail!("manifest artifact path '{rel}' must be relative and must not contain '..'");
    }
    Ok(root.join(path))
}

fn write_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating directory '{}'", parent.display()))?;
    }
    std::fs::write(path, bytes).with_context(|| format!("writing '{}'", path.display()))
}

fn write_frames_to_dir(dir: &Path, frames: &[RgbImage]) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating '{}'", dir.display()))?;
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading '{}'", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            std::fs::remove_file(&path)
                .with_context(|| format!("removing stale frame '{}'", path.display()))?;
        }
    }
    for (idx, frame) in frames.iter().enumerate() {
        let path = dir.join(format!("{idx:03}.png"));
        frame
            .save_with_format(&path, image::ImageFormat::Png)
            .with_context(|| format!("writing PNG frame '{}'", path.display()))?;
    }
    Ok(())
}

fn read_frames_from_dir(dir: &Path, expected: usize) -> anyhow::Result<Vec<RgbImage>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading frame directory '{}'", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| path.extension().and_then(|s| s.to_str()) == Some("png"));
    paths.sort();
    if paths.len() < expected {
        bail!(
            "frame directory '{}' has {} PNG(s), expected at least {expected}",
            dir.display(),
            paths.len()
        );
    }
    paths
        .into_iter()
        .take(expected)
        .map(|path| {
            image::open(&path)
                .with_context(|| format!("decoding PNG frame '{}'", path.display()))
                .map(|img| img.to_rgb8())
        })
        .collect()
}

fn write_preview_jpeg(path: &Path, frame: &RgbImage) -> anyhow::Result<()> {
    let mut bytes = Vec::new();
    {
        let mut cursor = Cursor::new(&mut bytes);
        let encoder = JpegEncoder::new_with_quality(&mut cursor, 85);
        encoder.write_image(
            frame.as_raw(),
            frame.width(),
            frame.height(),
            image::ExtendedColorType::Rgb8,
        )?;
    }
    write_file(path, &bytes)
}

fn trim_audio_front(
    audio: &mut Option<NativeAudioTrack>,
    frames: u32,
    fps: u32,
) -> anyhow::Result<()> {
    let Some(track) = audio.as_mut() else {
        return Ok(());
    };
    let samples = samples_for_frames(track, frames, fps);
    if track.interleaved_samples.len() < samples {
        bail!("audio sidecar too short for front trim");
    }
    track.interleaved_samples.drain(0..samples);
    Ok(())
}

fn trim_audio_back(
    audio: &mut Option<NativeAudioTrack>,
    frames: u32,
    fps: u32,
) -> anyhow::Result<()> {
    let Some(track) = audio.as_mut() else {
        return Ok(());
    };
    let samples = samples_for_frames(track, frames, fps);
    if track.interleaved_samples.len() < samples {
        bail!("audio sidecar too short for back trim");
    }
    let keep = track.interleaved_samples.len() - samples;
    track.interleaved_samples.truncate(keep);
    Ok(())
}

fn write_audio_boundary_out(
    layout: &JobDirLayout,
    stage_idx: u32,
    audio: &Option<NativeAudioTrack>,
    frames: u32,
    fps: u32,
) -> anyhow::Result<()> {
    let Some(track) = audio.as_ref() else {
        return Ok(());
    };
    let samples = samples_for_frames(track, frames, fps);
    if track.interleaved_samples.len() < samples {
        bail!("audio sidecar too short for boundary-out");
    }
    let mut boundary = track.clone();
    boundary.interleaved_samples =
        boundary.interleaved_samples[boundary.interleaved_samples.len() - samples..].to_vec();
    write_audio_sidecar(
        &layout.boundary_out_dir(stage_idx).join("audio.pcm"),
        &boundary,
    )
}

fn blend_audio_front_from_previous_boundary(
    layout: &JobDirLayout,
    stage_idx: u32,
    audio: &mut Option<NativeAudioTrack>,
    frames: u32,
    fps: u32,
) -> anyhow::Result<()> {
    let Some(track) = audio.as_mut() else {
        return Ok(());
    };
    let samples = samples_for_frames(track, frames, fps);
    if samples == 0 {
        return Ok(());
    }
    if track.interleaved_samples.len() < samples {
        bail!("audio sidecar too short for boundary-in");
    }
    let mut boundary_in = track.clone();
    boundary_in.interleaved_samples = track.interleaved_samples[..samples].to_vec();
    write_audio_sidecar(
        &layout.boundary_in_dir(stage_idx).join("audio.pcm"),
        &boundary_in,
    )?;
    let previous = read_audio_sidecar(&layout.boundary_out_dir(stage_idx - 1).join("audio.pcm"))?;
    if previous.sample_rate != track.sample_rate || previous.channels != track.channels {
        bail!("fade audio boundary format mismatch");
    }
    if previous.interleaved_samples.len() < samples {
        bail!("previous fade audio boundary too short");
    }
    let channels = track.channels as usize;
    let frame_samples = channels.max(1);
    let denom = (samples / frame_samples).max(1) as f32;
    for sample_idx in 0..samples {
        let t = (sample_idx / frame_samples) as f32 / denom;
        let prior = previous.interleaved_samples[sample_idx];
        let next = track.interleaved_samples[sample_idx];
        track.interleaved_samples[sample_idx] = prior * (1.0 - t) + next * t;
    }
    Ok(())
}

fn samples_for_frames(track: &NativeAudioTrack, frames: u32, fps: u32) -> usize {
    if fps == 0 {
        return 0;
    }
    ((track.sample_rate as u64 * frames as u64) / fps as u64) as usize * track.channels as usize
}

fn write_audio_sidecar(path: &Path, track: &NativeAudioTrack) -> anyhow::Result<()> {
    let mut bytes = Vec::with_capacity(16 + track.interleaved_samples.len() * 4);
    bytes.extend_from_slice(AUDIO_SIDECAR_MAGIC);
    bytes.extend_from_slice(&track.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&track.channels.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    for sample in &track.interleaved_samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    write_file(path, &bytes)
}

fn read_audio_sidecar(path: &Path) -> anyhow::Result<NativeAudioTrack> {
    let bytes = std::fs::read(path).with_context(|| format!("reading '{}'", path.display()))?;
    if bytes.len() < 16 || &bytes[..8] != AUDIO_SIDECAR_MAGIC {
        bail!("audio sidecar '{}' has unsupported format", path.display());
    }
    let sample_rate = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let channels = u16::from_le_bytes(bytes[12..14].try_into().unwrap());
    let sample_bytes = &bytes[16..];
    if sample_bytes.len() % 4 != 0 {
        bail!(
            "audio sidecar '{}' has truncated f32 sample",
            path.display()
        );
    }
    let interleaved_samples = sample_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok(NativeAudioTrack {
        interleaved_samples,
        sample_rate,
        channels,
    })
}

fn output_metadata_for_chain(req: &ChainRequest, frame_count: u32) -> OutputMetadata {
    let first_stage = req.stages.first();
    OutputMetadata {
        prompt: first_stage.map(|s| s.prompt.clone()).unwrap_or_default(),
        negative_prompt: first_stage.and_then(|s| s.negative_prompt.clone()),
        original_prompt: None,
        model: req.model.clone(),
        seed: req.seed.unwrap_or(0),
        steps: req.steps,
        guidance: req.guidance,
        width: req.width,
        height: req.height,
        strength: Some(req.strength),
        scheduler: None,
        output_format: Some(OutputFormat::Mp4),
        cfg_plus: None,
        lora: None,
        lora_scale: None,
        loras: None,
        control_model: None,
        control_scale: None,
        upscale_model: None,
        gif_preview: None,
        enable_audio: req.enable_audio,
        audio_file_path: None,
        source_video_path: None,
        pipeline: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        frames: Some(frame_count),
        fps: Some(req.fps),
        version: mold_core::build_info::version_string().to_string(),
    }
}

fn select_worker_for_stage(gpu_pool: &GpuPool, model: &str) -> Option<Arc<GpuWorker>> {
    let est = crate::queue::estimate_model_vram(model);
    gpu_pool.select_worker(model, est)
}

struct WorkerInFlightGuard {
    worker: Arc<GpuWorker>,
}

impl WorkerInFlightGuard {
    fn new(worker: Arc<GpuWorker>) -> Self {
        worker.in_flight.fetch_add(1, Ordering::SeqCst);
        Self { worker }
    }
}

impl Drop for WorkerInFlightGuard {
    fn drop(&mut self) {
        self.worker.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

struct WorkerActiveGenerationGuard {
    worker: Arc<GpuWorker>,
}

impl WorkerActiveGenerationGuard {
    fn new(worker: Arc<GpuWorker>, model: &str, prompt: &str) -> anyhow::Result<Self> {
        let mut active = worker
            .active_generation
            .write()
            .map_err(|e| anyhow!("active_generation lock poisoned: {e}"))?;
        *active = Some(ActiveGeneration {
            model: model.to_string(),
            prompt_sha256: format!("{:x}", Sha256::digest(prompt.as_bytes())),
            started_at_unix_ms: now_ms_u64(),
            started_at: std::time::Instant::now(),
        });
        drop(active);
        Ok(Self { worker })
    }
}

impl Drop for WorkerActiveGenerationGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.worker.active_generation.write() {
            *active = None;
        }
    }
}

fn now_ms_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn now_ms_i64() -> i64 {
    i64::try_from(now_ms_u64()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use mold_core::chain::{ChainStage, TransitionMode};
    use mold_core::types::OutputFormat;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    fn stage(prompt: &str, transition: TransitionMode) -> ChainStage {
        ChainStage {
            prompt: prompt.into(),
            frames: 9,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition,
            fade_frames: Some(2),
            model: None,
            loras: vec![],
            references: vec![],
        }
    }

    fn request(transitions: Vec<TransitionMode>) -> ChainRequest {
        ChainRequest {
            model: "ltx-2-19b-distilled:fp8".into(),
            stages: transitions
                .into_iter()
                .enumerate()
                .map(|(idx, transition)| stage(&format!("stage {idx}"), transition))
                .collect(),
            motion_tail_frames: 1,
            width: 64,
            height: 48,
            fps: 8,
            seed: Some(42),
            steps: 2,
            guidance: 1.0,
            strength: 1.0,
            output_format: OutputFormat::Mp4,
            placement: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
            enable_audio: None,
        }
    }

    fn job(
        id: &str,
        state: ChainJobState,
        created_at_ms: i64,
        dir: PathBuf,
        req: &ChainRequest,
    ) -> ChainJobRow {
        ChainJobRow {
            id: id.into(),
            state,
            model: req.model.clone(),
            request_json: serde_json::to_string(req).unwrap(),
            job_dir: dir,
            stage_count: req.stages.len() as u32,
            current_stage: 0,
            error: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
            finalized_at_ms: None,
        }
    }

    fn frame(value: u8) -> RgbImage {
        RgbImage::from_pixel(64, 48, Rgb([value, value, value]))
    }

    fn outcome(value: u8) -> StageOutcome {
        let frames = (0..9).map(|i| frame(value + i)).collect::<Vec<_>>();
        StageOutcome {
            tail: ChainTail {
                frames: 1,
                tail_rgb_frames: vec![frame(value + 8)],
            },
            frames,
            audio: None,
            generation_time_ms: 10,
        }
    }

    struct FakeExecutor {
        calls: AtomicUsize,
        cancel_on_progress: AtomicBool,
    }

    impl StageExecutor for FakeExecutor {
        fn render_stage(
            &self,
            _model: &str,
            _stage_req: &GenerateRequest,
            _carry: Option<&ChainTail>,
            _motion_tail_frames: u32,
            progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
        ) -> anyhow::Result<StageRenderOutcome> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) as u8;
            if self.cancel_on_progress.load(Ordering::SeqCst) && progress(1, 2).is_break() {
                return Ok(StageRenderOutcome::Cancelled);
            }
            let _ = progress(1, 2);
            Ok(StageRenderOutcome::Done(outcome(10 + call * 20)))
        }
    }

    struct FakeProbe(AtomicUsize);

    impl QueueProbe for FakeProbe {
        fn small_jobs_waiting(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }

    struct CarryInspectExecutor {
        seen: Mutex<Vec<Option<[u8; 3]>>>,
    }

    impl StageExecutor for CarryInspectExecutor {
        fn render_stage(
            &self,
            _model: &str,
            _stage_req: &GenerateRequest,
            carry: Option<&ChainTail>,
            _motion_tail_frames: u32,
            progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
        ) -> anyhow::Result<StageRenderOutcome> {
            let pixel = carry
                .and_then(|tail| tail.tail_rgb_frames.first())
                .map(|frame| frame.get_pixel(0, 0).0);
            self.seen.lock().unwrap().push(pixel);
            let _ = progress(1, 2);
            Ok(StageRenderOutcome::Done(outcome(140)))
        }
    }

    struct BoundaryCancelExecutor {
        calls: AtomicUsize,
        cancel: Arc<CancelRegistry>,
        job_id: String,
    }

    impl StageExecutor for BoundaryCancelExecutor {
        fn render_stage(
            &self,
            _model: &str,
            _stage_req: &GenerateRequest,
            _carry: Option<&ChainTail>,
            _motion_tail_frames: u32,
            progress: &(dyn Fn(u32, u32) -> ControlFlow<()> + Send + Sync),
        ) -> anyhow::Result<StageRenderOutcome> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let _ = progress(1, 2);
            if call == 0 {
                assert!(self.cancel.request(&self.job_id));
            }
            Ok(StageRenderOutcome::Done(outcome(30 + call as u8 * 20)))
        }
    }

    fn deps(
        db: MetadataDb,
        root: PathBuf,
        executor: Arc<FakeExecutor>,
        probe: Arc<FakeProbe>,
    ) -> RunnerDeps {
        RunnerDeps {
            db: Arc::new(Some(db)),
            jobs_root: root,
            executor,
            queue_probe: probe,
            events: Arc::new(JobEventBus::new()),
            cancel: Arc::new(CancelRegistry::new()),
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
            output_dir: None,
        }
    }

    fn persist_job(
        db: &MetadataDb,
        dir: &Path,
        id: &str,
        req: &ChainRequest,
        state: ChainJobState,
    ) -> ChainJobRow {
        let manifest = ChainJobManifest::new(id.into(), 1_000, req).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        manifest.write_atomic(dir).unwrap();
        let row = job(id, state, 1_000, dir.to_path_buf(), req);
        chain_jobs::insert_job(db, &row).unwrap();
        for stage in &manifest.stage_status {
            chain_jobs::upsert_stage(
                db,
                &ChainJobStageRow {
                    job_id: id.into(),
                    stage_idx: stage.idx,
                    state: stage.state,
                    seed: stage.seed,
                    frames_emitted: None,
                    generation_time_ms: None,
                    segment_rel_path: None,
                    error: None,
                    updated_at_ms: 1_000,
                },
            )
            .unwrap();
        }
        row
    }

    #[test]
    fn corrupt_manifest_marks_job_failed_without_rendering() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55CORRUPT", &req, ChainJobState::Queued);
        std::fs::write(job_dir.join("manifest.toml"), "not valid toml").unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        let failed = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(failed.state, ChainJobState::Failed);
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|err| err.contains("manifest TOML parse failed")),
            "parse error must be persisted, got {:?}",
            failed.error
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn resume_carry_error_after_running_marks_job_failed_and_unlocks_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55BADTAIL", &req, ChainJobState::Queued);
        let layout = JobDirLayout::new(job_dir.clone());
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        write_stage_artifacts(&layout, &mut manifest, 0, &outcome(20), &req).unwrap();
        std::fs::remove_dir_all(layout.tail_dir(0)).unwrap();
        chain_jobs::upsert_stage(
            &db,
            &ChainJobStageRow {
                job_id: row.id.clone(),
                stage_idx: 0,
                state: StageState::Completed,
                seed: manifest.stage_status[0].seed,
                frames_emitted: manifest.stage_status[0].frames_emitted,
                generation_time_ms: manifest.stage_status[0].generation_time_ms,
                segment_rel_path: manifest.stage_status[0].segment.clone(),
                error: None,
                updated_at_ms: 1_000,
            },
        )
        .unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        let failed = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(failed.state, ChainJobState::Failed);
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|err| err.contains("chain tail directory")),
            "resume error must be persisted, got {:?}",
            failed.error
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
        assert!(chain_jobs::set_job_queued(db, &row.id, 1, now_ms_i64()).unwrap());
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Queued
        );
        assert!(chain_jobs::delete_job(db, &row.id).unwrap());
    }

    #[test]
    fn skipping_completed_stage_reloads_tail_for_next_smooth_stage() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![
            TransitionMode::Smooth,
            TransitionMode::Smooth,
            TransitionMode::Smooth,
        ]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55SKIPCARRY",
            &req,
            ChainJobState::Queued,
        );
        let layout = JobDirLayout::new(job_dir.clone());
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        write_stage_artifacts(&layout, &mut manifest, 0, &outcome(20), &req).unwrap();
        write_stage_artifacts(&layout, &mut manifest, 1, &outcome(80), &req).unwrap();
        for idx in 0..2 {
            let status = &manifest.stage_status[idx];
            chain_jobs::upsert_stage(
                &db,
                &ChainJobStageRow {
                    job_id: row.id.clone(),
                    stage_idx: idx as u32,
                    state: status.state,
                    seed: status.seed,
                    frames_emitted: status.frames_emitted,
                    generation_time_ms: status.generation_time_ms,
                    segment_rel_path: status.segment.clone(),
                    error: None,
                    updated_at_ms: 1_000,
                },
            )
            .unwrap();
        }
        let executor = Arc::new(CarryInspectExecutor {
            seen: Mutex::new(Vec::new()),
        });
        let deps = RunnerDeps {
            db: Arc::new(Some(db)),
            jobs_root: dir.path().join("jobs"),
            executor: executor.clone(),
            queue_probe: Arc::new(FakeProbe(AtomicUsize::new(0))),
            events: Arc::new(JobEventBus::new()),
            cancel: Arc::new(CancelRegistry::new()),
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
            output_dir: None,
        };

        execute_job(&deps, &row, 0).unwrap();

        assert_eq!(
            executor.seen.lock().unwrap().as_slice(),
            &[Some([88, 88, 88])],
            "stage 2 must receive stage 1's persisted smooth tail"
        );
    }

    #[test]
    fn runner_next_queued_job_uses_fifo_order() {
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let newer = job(
            "01JBR55NEWER",
            ChainJobState::Queued,
            2_000,
            std::env::temp_dir().join("newer"),
            &req,
        );
        let older = job(
            "01JBR55OLDER",
            ChainJobState::Queued,
            1_000,
            std::env::temp_dir().join("older"),
            &req,
        );
        chain_jobs::insert_job(&db, &newer).unwrap();
        chain_jobs::insert_job(&db, &older).unwrap();

        let got = next_queued_job(&db).unwrap().expect("queued job");
        assert_eq!(got.id, older.id);
    }

    #[test]
    fn safe_join_manifest_rel_rejects_windows_and_backslash_traversal() {
        let root = Path::new("/tmp/mold-chain-job");
        for rel in [
            r"C:\tmp\segment.mp4",
            r"stages\000\..\segment.mp4",
            r"\stages\000\segment.mp4",
        ] {
            let err = safe_join_manifest_rel(root, rel).unwrap_err();
            assert!(
                err.to_string().contains("must be relative"),
                "expected traversal rejection for {rel:?}, got {err:#}"
            );
        }
    }

    #[test]
    fn completed_job_subscriptions_are_non_persistent() {
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let row = job(
            "01JBR55DONEATTACH",
            ChainJobState::Completed,
            1_000,
            std::env::temp_dir().join("done-attach"),
            &req,
        );
        chain_jobs::insert_job(&db, &row).unwrap();
        let events = JobEventBus::new();

        let _rx1 = events.subscribe_for_job(&db, &row.id).unwrap();
        let _rx2 = events.subscribe_for_job(&db, &row.id).unwrap();

        assert!(!events.contains_for_tests(&row.id));
    }

    #[tokio::test]
    async fn subscribe_receives_live_event_buffered_after_attach() {
        let (kick_tx, _kick_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = ChainJobRunnerHandle {
            kick_tx,
            cancel: Arc::new(CancelRegistry::new()),
            events: Arc::new(JobEventBus::new()),
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
        };

        let mut rx = handle
            .events
            .subscribe_persistent_for_tests("01JBR55EVENTS");
        handle.events.publish(
            "01JBR55EVENTS",
            ChainJobEvent::StateChanged {
                state: ChainJobState::Running,
                error: None,
            },
        );

        let event = rx.recv().await.unwrap();
        assert!(matches!(
            event,
            ChainJobEvent::StateChanged {
                state: ChainJobState::Running,
                error: None
            }
        ));
    }

    #[test]
    fn execute_job_persists_stage_artifacts_manifest_then_db_and_finalizes() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55ORDER", &req, ChainJobState::Queued);
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();

        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        let db = deps.db.as_ref().as_ref().unwrap();
        let job_after = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert!(
            job_dir.join("stages/000/segment.mp4").exists(),
            "job state {:?}, error {:?}",
            job_after.state,
            job_after.error
        );
        assert!(job_dir.join("stages/000/tail/000.png").exists());
        assert!(job_dir.join("stages/001/segment.mp4").exists());
        assert_eq!(manifest.stage_status[0].state, StageState::Completed);
        assert_eq!(manifest.stage_status[1].state, StageState::Completed);
        assert_eq!(manifest.finalizes.len(), 1);

        let stages = chain_jobs::stages_for_job(db, &row.id).unwrap();
        assert_eq!(stages[0].state, StageState::Completed);
        assert_eq!(
            stages[0].segment_rel_path.as_deref(),
            Some("stages/000/segment.mp4")
        );
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Completed
        );
    }

    #[test]
    fn ephemeral_execute_job_defers_gallery_record_to_legacy_shim() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55EPHGALLERY",
            &req,
            ChainJobState::Queued,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&job_dir).unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let output_dir = dir.path().join("gallery");
        let mut deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        deps.output_dir = Some(output_dir.clone());

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(
            db.list(Some(&output_dir)).unwrap().len(),
            0,
            "ephemeral legacy shim jobs must not write runner-side gallery rows"
        );
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Completed
        );
    }

    #[test]
    fn durable_execute_job_records_exactly_one_runner_gallery_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55DURGALLERY",
            &req,
            ChainJobState::Queued,
        );
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let output_dir = dir.path().join("gallery");
        let mut deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        deps.output_dir = Some(output_dir.clone());

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        let rows = db.list(Some(&output_dir)).unwrap();
        assert_eq!(rows.len(), 1, "durable jobs save exactly one gallery row");
        assert_eq!(rows[0].format, OutputFormat::Mp4);
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Completed
        );
    }

    #[test]
    fn cancel_via_progress_marks_job_cancelled_and_keeps_completed_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55CANCEL", &req, ChainJobState::Queued);
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(true),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        deps.cancel.register(&row.id);
        assert!(deps.cancel.request(&row.id));

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Cancelled
        );
        assert!(!job_dir.join("stages/000/segment.mp4").exists());
    }

    #[test]
    fn cancel_then_retake_runs_to_completion_without_stale_cancel_mark() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55CANCELRETAKE",
            &req,
            ChainJobState::Queued,
        );
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        deps.cancel.register(&row.id);
        assert!(deps.cancel.request(&row.id));
        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Cancelled
        );
        let updated = apply_retake(
            db,
            dir.path(),
            &row.id,
            &RetakeRequest {
                stage_idx: 0,
                mode: RetakeMode::Cascade,
                seed_offset: Some(9),
                prompt: None,
            },
        )
        .unwrap();

        execute_job(&deps, &updated, 0).unwrap();

        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Completed
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancel_mark_after_claim_is_honored_before_first_stage() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55CLAIMCANCEL",
            &req,
            ChainJobState::Queued,
        );
        assert!(chain_jobs::claim_job(&db, &row.id).unwrap());
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        deps.cancel.register(&row.id);
        assert!(deps.cancel.request(&row.id));
        let running = chain_jobs::get_job(deps.db.as_ref().as_ref().unwrap(), &row.id)
            .unwrap()
            .unwrap();

        execute_job(&deps, &running, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Cancelled
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn retake_aborts_after_resume_wins_without_manifest_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Cut]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55RETAKERACE",
            &req,
            ChainJobState::Failed,
        );
        assert!(chain_jobs::try_transition(
            &db,
            &row.id,
            &[ChainJobState::Failed],
            ChainJobState::Queued,
            None,
            2_000,
        )
        .unwrap());

        let err = apply_retake(
            &db,
            dir.path(),
            &row.id,
            &RetakeRequest {
                stage_idx: 0,
                mode: RetakeMode::Cascade,
                seed_offset: Some(5),
                prompt: Some("should not land".into()),
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("not retakeable")
                || err.to_string().contains("CHAIN_JOB_RUNNING"),
            "retake loser must abort with a current-state error, got {err:#}"
        );
        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        assert!(manifest.retakes.is_empty());
        assert_eq!(manifest.stage_status[0].seed, 42);
    }

    #[test]
    fn apply_retake_rejects_splice_before_smooth_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let _row = persist_job(
            &db,
            &job_dir,
            "01JBR55RETAKE",
            &req,
            ChainJobState::Completed,
        );

        let err = apply_retake(
            &db,
            dir.path(),
            "01JBR55RETAKE",
            &RetakeRequest {
                stage_idx: 0,
                mode: RetakeMode::Splice,
                seed_offset: Some(9),
                prompt: None,
            },
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("RETAKE_SPLICE_REQUIRES_CUT_OR_FADE"));
    }

    #[test]
    fn apply_retake_cascade_resets_target_through_end_and_records_amendment() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![
            TransitionMode::Smooth,
            TransitionMode::Cut,
            TransitionMode::Cut,
        ]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55CASCADE",
            &req,
            ChainJobState::Completed,
        );
        let layout = JobDirLayout::new(job_dir.clone());
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        for stage in &mut manifest.stage_status {
            stage.state = StageState::Completed;
            stage.segment = Some(layout.segment_rel(stage.idx));
            layout.ensure_stage_dirs(stage.idx).unwrap();
            std::fs::write(layout.stage_dir(stage.idx).join("stale.txt"), b"stale").unwrap();
            chain_jobs::upsert_stage(
                &db,
                &ChainJobStageRow {
                    job_id: row.id.clone(),
                    stage_idx: stage.idx,
                    state: StageState::Completed,
                    seed: stage.seed,
                    frames_emitted: Some(9),
                    generation_time_ms: Some(10),
                    segment_rel_path: stage.segment.clone(),
                    error: None,
                    updated_at_ms: 1_000,
                },
            )
            .unwrap();
        }
        manifest.write_atomic(&job_dir).unwrap();

        let updated = apply_retake(
            &db,
            dir.path(),
            &row.id,
            &RetakeRequest {
                stage_idx: 1,
                mode: RetakeMode::Cascade,
                seed_offset: Some(7),
                prompt: Some("new middle".into()),
            },
        )
        .unwrap();

        assert_eq!(updated.state, ChainJobState::Queued);
        assert_eq!(updated.current_stage, 1);
        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        assert_eq!(manifest.retakes.len(), 1);
        assert_eq!(manifest.retakes[0].stage_idx, 1);
        assert_eq!(
            manifest.retakes[0].new_prompt.as_deref(),
            Some("new middle")
        );
        assert_eq!(manifest.stage_status[0].state, StageState::Completed);
        assert_eq!(manifest.stage_status[1].state, StageState::Pending);
        assert_eq!(manifest.stage_status[2].state, StageState::Pending);
        assert!(!layout.stage_dir(1).exists());
        assert!(!layout.stage_dir(2).exists());
        let stages = chain_jobs::stages_for_job(&db, &row.id).unwrap();
        assert_eq!(stages[1].state, StageState::Pending);
        assert_eq!(stages[2].state, StageState::Pending);
        let effective = effective_request(&manifest).unwrap();
        assert_eq!(effective.stages[1].prompt, "new middle");
        assert_eq!(
            manifest.stage_status[1].seed,
            effective_stage_seed(req.seed.unwrap(), Some(7))
        );
    }

    #[test]
    fn finalize_versions_outputs_and_records_stage_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55VERSION", &req, ChainJobState::Queued);
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        let first_seed = manifest.stage_status[0].seed;
        assert_eq!(manifest.finalizes[0].output, "final/output-1.mp4");

        let second = finalize_job(&deps, &row, &mut manifest).unwrap();

        assert_eq!(second, "final/output-2.mp4");
        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        assert_eq!(manifest.finalizes.len(), 2);
        assert_eq!(manifest.finalizes[0].stage_seeds, vec![first_seed]);
        assert_eq!(manifest.finalizes[1].stage_seeds, vec![first_seed]);
        assert!(job_dir.join("final/output-1.mp4").exists());
        assert!(job_dir.join("final/output-2.mp4").exists());
    }

    #[test]
    fn completed_fade_successor_is_reencoded_after_previous_stage_changes_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let job_dir = dir.path().join("job");
        let layout = JobDirLayout::new(job_dir.clone());
        layout.ensure_root().unwrap();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Fade]);
        let mut manifest = ChainJobManifest::new("01JBR55FADE".into(), 1_000, &req).unwrap();

        write_stage_artifacts(&layout, &mut manifest, 0, &outcome(20), &req).unwrap();
        write_stage_artifacts(&layout, &mut manifest, 1, &outcome(80), &req).unwrap();
        let before = std::fs::read(layout.segment_path(1)).unwrap();

        write_frames_to_dir(&layout.boundary_out_dir(0), &[frame(210), frame(220)]).unwrap();
        maybe_reencode_next_after_fade(&layout, &manifest, 0, &req).unwrap();

        let after = std::fs::read(layout.segment_path(1)).unwrap();
        assert_ne!(
            before, after,
            "changing outgoing fade boundary must re-encode completed successor segment"
        );
    }

    #[test]
    fn splice_retake_skips_completed_successors_and_only_reencodes_fade_boundary() {
        for (next_transition, successor_should_change, id) in [
            (TransitionMode::Fade, true, "01JBR55SPLICEFADE"),
            (TransitionMode::Cut, false, "01JBR55SPLICECUT"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = db();
            let req = request(vec![
                TransitionMode::Smooth,
                next_transition,
                TransitionMode::Cut,
                TransitionMode::Cut,
            ]);
            let job_dir = dir.path().join("job");
            let row = persist_job(&db, &job_dir, id, &req, ChainJobState::Queued);
            let executor = Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                cancel_on_progress: AtomicBool::new(false),
            });
            let deps = deps(
                db,
                dir.path().join("jobs"),
                executor.clone(),
                Arc::new(FakeProbe(AtomicUsize::new(0))),
            );
            execute_job(&deps, &row, 0).unwrap();

            let layout = JobDirLayout::new(job_dir.clone());
            let before_1 = std::fs::read(layout.segment_path(1)).unwrap();
            let before_2 = std::fs::read(layout.segment_path(2)).unwrap();
            let before_3 = std::fs::read(layout.segment_path(3)).unwrap();
            let calls_before = executor.calls.load(Ordering::SeqCst);
            let db = deps.db.as_ref().as_ref().unwrap();
            let updated = apply_retake(
                db,
                dir.path(),
                &row.id,
                &RetakeRequest {
                    stage_idx: 0,
                    mode: RetakeMode::Splice,
                    seed_offset: Some(9),
                    prompt: None,
                },
            )
            .unwrap();

            execute_job(&deps, &updated, 0).unwrap();

            assert_eq!(
                executor.calls.load(Ordering::SeqCst) - calls_before,
                1,
                "splice retake must render exactly the target stage for {id}"
            );
            let after_1 = std::fs::read(layout.segment_path(1)).unwrap();
            let after_2 = std::fs::read(layout.segment_path(2)).unwrap();
            let after_3 = std::fs::read(layout.segment_path(3)).unwrap();
            if successor_should_change {
                assert_ne!(
                    before_1, after_1,
                    "fade successor must be re-encoded at the edited boundary"
                );
            } else {
                assert_eq!(
                    before_1, after_1,
                    "cut successor must not be touched by splice retake"
                );
            }
            assert_eq!(before_2, after_2, "stage N+2 must remain untouched");
            assert_eq!(before_3, after_3, "stage N+3 must remain untouched");
        }
    }

    #[test]
    fn disk_write_failure_marks_job_and_stage_failed_with_error() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55DISK", &req, ChainJobState::Queued);
        std::fs::write(job_dir.join("stages"), b"not a directory").unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        let job_after = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(job_after.state, ChainJobState::Failed);
        assert!(
            job_after.error.as_deref().is_some_and(
                |error| error.contains("stages/000") || error.contains("Not a directory")
            ),
            "explicit filesystem error should be persisted, got {:?}",
            job_after.error
        );
        let stage = chain_jobs::stages_for_job(db, &row.id).unwrap().remove(0);
        assert_eq!(stage.state, StageState::Failed);
        assert!(stage
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("stages/000"));
        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        assert_eq!(manifest.stage_status[0].state, StageState::Failed);
    }

    #[test]
    fn finalize_failure_then_resume_reenters_finalize_without_rerendering() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55FINALRETRY",
            &req,
            ChainJobState::Queued,
        );
        std::fs::write(job_dir.join("final"), b"not a directory").unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();
        let db = deps.db.as_ref().as_ref().unwrap();
        let failed = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(failed.state, ChainJobState::Failed);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert!(ChainJobManifest::read_from_dir(&job_dir)
            .unwrap()
            .finalizes
            .is_empty());

        std::fs::remove_file(job_dir.join("final")).unwrap();
        assert!(chain_jobs::try_transition(
            db,
            &row.id,
            &[ChainJobState::Failed],
            ChainJobState::Queued,
            None,
            now_ms_i64(),
        )
        .unwrap());
        let queued = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        execute_job(&deps, &queued, req.stages.len() as u32).unwrap();

        let completed = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(completed.state, ChainJobState::Completed);
        assert_eq!(
            executor.calls.load(Ordering::SeqCst),
            1,
            "resume from completed stages must not render again"
        );
        let manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        assert_eq!(manifest.finalizes.len(), 1);
        assert_eq!(manifest.finalizes[0].output, "final/output-1.mp4");
    }

    #[test]
    fn boundary_cancel_keeps_completed_stage_artifacts_and_stops_next_stage() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55BOUNDARY",
            &req,
            ChainJobState::Queued,
        );
        let cancel = Arc::new(CancelRegistry::new());
        let executor = Arc::new(BoundaryCancelExecutor {
            calls: AtomicUsize::new(0),
            cancel: cancel.clone(),
            job_id: row.id.clone(),
        });
        let deps = RunnerDeps {
            db: Arc::new(Some(db)),
            jobs_root: dir.path().join("jobs"),
            executor,
            queue_probe: Arc::new(FakeProbe(AtomicUsize::new(0))),
            events: Arc::new(JobEventBus::new()),
            cancel,
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
            output_dir: None,
        };

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(
            chain_jobs::get_job(db, &row.id).unwrap().unwrap().state,
            ChainJobState::Cancelled
        );
        assert!(job_dir.join("stages/000/segment.mp4").exists());
        assert!(!job_dir.join("stages/001/segment.mp4").exists());
    }

    #[test]
    fn yields_between_stages_when_small_jobs_are_waiting_and_cleans_bus_at_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(&db, &job_dir, "01JBR55YIELD", &req, ChainJobState::Queued);
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor,
            Arc::new(FakeProbe(AtomicUsize::new(2))),
        );
        let mut rx = deps.events.subscribe_persistent_for_tests(&row.id);

        execute_job(&deps, &row, 0).unwrap();

        let mut events = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(event) => events.push(event),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                | Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
                Err(err) => panic!("unexpected broadcast receive error: {err}"),
            }
        }
        assert!(matches!(
            events.as_slice(),
            [
                ChainJobEvent::StateChanged {
                    state: ChainJobState::Running,
                    ..
                },
                ChainJobEvent::StageStart { stage_idx: 0 },
                ChainJobEvent::DenoiseStep { stage_idx: 0, .. },
                ChainJobEvent::StageDone { stage_idx: 0, .. },
                ..
            ]
        ));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                ChainJobEvent::Yielded {
                    pending_small_jobs: 2
                }
            )
        }));
        assert!(!deps.events.senders.lock().unwrap().contains_key(&row.id));
    }

    #[test]
    fn resume_carry_from_disk_loads_smooth_tail_and_rejects_cut_or_fade_carry() {
        let dir = tempfile::tempdir().unwrap();
        let job_dir = dir.path().join("job");
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let mut manifest = ChainJobManifest::new("01JBR55CARRY".into(), 1_000, &req).unwrap();
        manifest.stage_status[0].state = StageState::Completed;
        manifest.stage_status[0].segment = Some("stages/000/segment.mp4".into());
        manifest.stage_status[0].tail_frames = Some(1);
        let layout = JobDirLayout::new(job_dir.clone());
        layout.ensure_stage_dirs(0).unwrap();
        frame(77).save(layout.tail_dir(0).join("000.png")).unwrap();
        manifest.write_atomic(&job_dir).unwrap();

        let carry = resume_carry_from_disk(&job_dir, &manifest, 1)
            .unwrap()
            .expect("smooth continuation should reload tail");
        assert_eq!(carry.frames, 1);
        assert_eq!(carry.tail_rgb_frames[0].get_pixel(0, 0).0, [77, 77, 77]);

        for transition in [TransitionMode::Cut, TransitionMode::Fade] {
            let req = request(vec![TransitionMode::Smooth, transition]);
            let mut manifest = ChainJobManifest::new("01JBR55CARRY".into(), 1_000, &req).unwrap();
            manifest.stage_status[0].state = StageState::Completed;
            manifest.stage_status[0].tail_frames = Some(1);
            assert!(resume_carry_from_disk(&job_dir, &manifest, 1)
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn malicious_manifest_resume_fails_and_job_remains_deletable() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Smooth]);
        let job_dir = dir.path().join("job");
        let row = persist_job(
            &db,
            &job_dir,
            "01JBR55TRAVERSAL",
            &req,
            ChainJobState::Queued,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        manifest.stage_status[0].state = StageState::Completed;
        manifest.stage_status[0].segment = Some("../outside.mp4".into());
        manifest.stage_status[0].tail_frames = Some(1);
        manifest.write_atomic(&job_dir).unwrap();
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
            cancel_on_progress: AtomicBool::new(false),
        });
        let deps = deps(
            db,
            dir.path().join("jobs"),
            executor.clone(),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );

        execute_job(&deps, &row, 0).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        let failed = chain_jobs::get_job(db, &row.id).unwrap().unwrap();
        assert_eq!(failed.state, ChainJobState::Failed);
        assert!(
            failed
                .error
                .as_deref()
                .is_some_and(|err| err.contains("must be relative")),
            "expected traversal error to be persisted, got {:?}",
            failed.error
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
        assert!(chain_jobs::delete_job_not_running(db, &row.id).unwrap());
        assert!(chain_jobs::get_job(db, &row.id).unwrap().is_none());
    }

    #[test]
    fn startup_reconcile_flips_running_to_interrupted_and_repairs_from_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let job_dir = dir.path().join("jobs/01JBR55RECON");
        let row = persist_job(&db, &job_dir, "01JBR55RECON", &req, ChainJobState::Running);
        let mut manifest = ChainJobManifest::read_from_dir(&job_dir).unwrap();
        manifest.stage_status[0].state = StageState::Completed;
        manifest.stage_status[0].frames_emitted = Some(9);
        manifest.stage_status[0].generation_time_ms = Some(123);
        manifest.stage_status[0].segment = Some("stages/000/segment.mp4".into());
        manifest.write_atomic(&job_dir).unwrap();

        let (flipped, repaired) = startup_reconcile(&db, &dir.path().join("jobs")).unwrap();

        assert_eq!(flipped, 1);
        assert!(repaired >= 1);
        let row_after = chain_jobs::get_job(&db, &row.id).unwrap().unwrap();
        assert_eq!(row_after.state, ChainJobState::Interrupted);
        assert_eq!(row_after.current_stage, 1);
        assert_eq!(
            row_after.error.as_deref(),
            Some("server restarted while chain job was running")
        );
        let stage = chain_jobs::stages_for_job(&db, &row.id).unwrap().remove(0);
        assert_eq!(stage.state, StageState::Completed);
        assert_eq!(
            stage.segment_rel_path.as_deref(),
            Some("stages/000/segment.mp4")
        );
    }

    #[test]
    fn ephemeral_claim_guard_releases_on_drop() {
        let claims = Arc::new(EphemeralClaims::default());
        {
            let _guard = claims.claim("01JBR55CLAIM");
            assert!(claims.is_claimed("01JBR55CLAIM"));
        }
        assert!(!claims.is_claimed("01JBR55CLAIM"));
    }

    #[test]
    fn create_job_with_params_persists_ephemeral_manifest_and_stage_rows() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth, TransitionMode::Cut]);

        let row = create_job_with_params(
            &db,
            dir.path(),
            CreateJobParams {
                id: "01JBR55CREATE".into(),
                ephemeral: true,
                request: req.clone(),
            },
        )
        .unwrap();

        assert_eq!(row.id, "01JBR55CREATE");
        assert_eq!(row.state, ChainJobState::Queued);
        assert_eq!(row.stage_count, 2);
        assert!(row.job_dir.ends_with("01JBR55CREATE"));
        let manifest = ChainJobManifest::read_from_dir(&row.job_dir).unwrap();
        assert!(manifest.ephemeral);
        assert_eq!(manifest.request().unwrap(), req);
        let stages = chain_jobs::stages_for_job(&db, &row.id).unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].state, StageState::Pending);
    }

    #[test]
    fn startup_gc_sweep_removes_non_running_ephemerals_only() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let jobs_root = dir.path().join("jobs");
        let ephemeral_done = persist_job(
            &db,
            &jobs_root.join("01JBR55EPHDONE"),
            "01JBR55EPHDONE",
            &req,
            ChainJobState::Completed,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&ephemeral_done.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&ephemeral_done.job_dir).unwrap();
        let ephemeral_running = persist_job(
            &db,
            &jobs_root.join("01JBR55EPHRUN"),
            "01JBR55EPHRUN",
            &req,
            ChainJobState::Running,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&ephemeral_running.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&ephemeral_running.job_dir).unwrap();
        let durable_done = persist_job(
            &db,
            &jobs_root.join("01JBR55DURABLE"),
            "01JBR55DURABLE",
            &req,
            ChainJobState::Completed,
        );

        let outcome = startup_gc_sweep(&db, &jobs_root).unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 1);
        assert!(chain_jobs::get_job(&db, &ephemeral_done.id)
            .unwrap()
            .is_none());
        assert!(!ephemeral_done.job_dir.exists());
        assert!(chain_jobs::get_job(&db, &ephemeral_running.id)
            .unwrap()
            .is_some());
        assert!(ephemeral_running.job_dir.exists());
        assert!(chain_jobs::get_job(&db, &durable_done.id)
            .unwrap()
            .is_some());
        assert!(durable_done.job_dir.exists());
    }

    #[test]
    fn gc_sweeps_unclaimed_ephemeral_after_grace_but_preserves_live_claim() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let jobs_root = dir.path().join("jobs");
        let old = persist_job(
            &db,
            &jobs_root.join("01JBR55OLDCLAIMLESS"),
            "01JBR55OLDCLAIMLESS",
            &req,
            ChainJobState::Completed,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&old.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&old.job_dir).unwrap();
        chain_jobs::update_job_state(&db, &old.id, ChainJobState::Completed, None, 1_000).unwrap();
        let claimed = persist_job(
            &db,
            &jobs_root.join("01JBR55CLAIMEDGC"),
            "01JBR55CLAIMEDGC",
            &req,
            ChainJobState::Completed,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&claimed.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&claimed.job_dir).unwrap();

        let deps = deps(
            db,
            jobs_root,
            Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                cancel_on_progress: AtomicBool::new(false),
            }),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        let _claim = deps.claims.claim(&claimed.id);
        let outcome =
            run_gc_pass(&deps, 7, 1_000 + (EPHEMERAL_GRACE_SECS as i64 + 1) * 1_000).unwrap();

        let db = deps.db.as_ref().as_ref().unwrap();
        assert_eq!(outcome.swept_ephemeral_jobs, 1);
        assert!(chain_jobs::get_job(db, &old.id).unwrap().is_none());
        assert!(chain_jobs::get_job(db, &claimed.id).unwrap().is_some());
        assert!(claimed.job_dir.exists());
    }

    #[test]
    fn gc_respects_ephemeral_grace_and_prunes_only_completed_durable_stages() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let jobs_root = dir.path().join("jobs");
        let recent_ephemeral = persist_job(
            &db,
            &jobs_root.join("01JBR55RECENTEPH"),
            "01JBR55RECENTEPH",
            &req,
            ChainJobState::Completed,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&recent_ephemeral.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&recent_ephemeral.job_dir).unwrap();
        let now = 1_000 + 8 * 86_400_000;
        chain_jobs::update_job_state(
            &db,
            &recent_ephemeral.id,
            ChainJobState::Completed,
            None,
            now - (EPHEMERAL_GRACE_SECS as i64 * 1_000) + 100,
        )
        .unwrap();

        let durable = persist_job(
            &db,
            &jobs_root.join("01JBR55TTL"),
            "01JBR55TTL",
            &req,
            ChainJobState::Completed,
        );
        let layout = JobDirLayout::new(durable.job_dir.clone());
        std::fs::create_dir_all(layout.stage_dir(0)).unwrap();
        std::fs::write(layout.stage_dir(0).join("segment.mp4"), b"stage").unwrap();
        std::fs::create_dir_all(durable.job_dir.join("final")).unwrap();
        std::fs::write(durable.job_dir.join("final/output-1.mp4"), b"final").unwrap();
        chain_jobs::update_job_state(&db, &durable.id, ChainJobState::Completed, None, 1_000)
            .unwrap();

        let deps = deps(
            db,
            jobs_root,
            Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                cancel_on_progress: AtomicBool::new(false),
            }),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        let outcome = run_gc_pass(&deps, 7, now).unwrap();
        let db = deps.db.as_ref().as_ref().unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 0);
        assert_eq!(outcome.pruned_artifact_dirs, 1);
        assert!(chain_jobs::get_job(db, &recent_ephemeral.id)
            .unwrap()
            .is_some());
        assert!(recent_ephemeral.job_dir.exists());
        assert!(!durable.job_dir.join("stages").exists());
        assert!(durable.job_dir.join("final/output-1.mp4").exists());
        assert!(durable.job_dir.join("manifest.toml").exists());
        assert!(chain_jobs::get_job(db, &durable.id).unwrap().is_some());
    }

    #[test]
    fn gc_exempts_durable_failed_interrupted_cancelled_but_ephemeral_overrides() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let jobs_root = dir.path().join("jobs");
        let mut durable_rows = Vec::new();
        for (id, state) in [
            ("01JBR55FAILED", ChainJobState::Failed),
            ("01JBR55INTR", ChainJobState::Interrupted),
            ("01JBR55CANCELLED", ChainJobState::Cancelled),
        ] {
            let row = persist_job(&db, &jobs_root.join(id), id, &req, state);
            std::fs::create_dir_all(row.job_dir.join("stages/000")).unwrap();
            durable_rows.push(row);
        }
        let eph_failed = persist_job(
            &db,
            &jobs_root.join("01JBR55EPHFAILED"),
            "01JBR55EPHFAILED",
            &req,
            ChainJobState::Failed,
        );
        let mut manifest = ChainJobManifest::read_from_dir(&eph_failed.job_dir).unwrap();
        manifest.ephemeral = true;
        manifest.write_atomic(&eph_failed.job_dir).unwrap();
        chain_jobs::update_job_state(&db, &eph_failed.id, ChainJobState::Failed, None, 1_000)
            .unwrap();

        let deps = deps(
            db,
            jobs_root,
            Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                cancel_on_progress: AtomicBool::new(false),
            }),
            Arc::new(FakeProbe(AtomicUsize::new(0))),
        );
        let outcome =
            run_gc_pass(&deps, 7, 1_000 + (EPHEMERAL_GRACE_SECS as i64 + 1) * 1_000).unwrap();
        let db = deps.db.as_ref().as_ref().unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 1);
        assert!(chain_jobs::get_job(db, &eph_failed.id).unwrap().is_none());
        for row in durable_rows {
            assert!(chain_jobs::get_job(db, &row.id).unwrap().is_some());
            assert!(row.job_dir.join("stages").exists());
        }
    }

    #[tokio::test]
    async fn runner_request_gc_replies_between_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let db = db();
        let req = request(vec![TransitionMode::Smooth]);
        let jobs_root = dir.path().join("jobs");
        let _row = persist_job(
            &db,
            &jobs_root.join("01JBR55REQUESTGC"),
            "01JBR55REQUESTGC",
            &req,
            ChainJobState::Completed,
        );
        let deps = RunnerDeps {
            db: Arc::new(Some(db)),
            jobs_root,
            executor: Arc::new(FakeExecutor {
                calls: AtomicUsize::new(0),
                cancel_on_progress: AtomicBool::new(false),
            }),
            queue_probe: Arc::new(FakeProbe(AtomicUsize::new(0))),
            events: Arc::new(JobEventBus::new()),
            cancel: Arc::new(CancelRegistry::new()),
            job_locks: Arc::new(JobMutationLocks::new()),
            claims: Arc::new(EphemeralClaims::default()),
            output_dir: None,
        };
        let handle = spawn_runner(deps);

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(1), handle.request_gc())
            .await
            .expect("GC request should be serviced without waiting for the whole queue")
            .unwrap();

        assert_eq!(outcome.swept_ephemeral_jobs, 0);
    }
}
