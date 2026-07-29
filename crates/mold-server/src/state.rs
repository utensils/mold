use mold_core::Config;
use mold_inference::InferenceEngine;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::Mutex;

use mold_inference::shared_pool::SharedPool;

use crate::downloads::DownloadQueue;
use crate::events::EventBroadcaster;
use crate::gpu_pool::GpuPool;
use crate::job_registry::{JobRegistry, SharedJobRegistry};
use crate::model_cache::ModelCache;
use crate::resources::ResourceBroadcaster;

#[derive(Debug, Clone, Default)]
pub struct EngineSnapshot {
    /// Currently GPU-loaded model (None if no model on GPU).
    pub model_name: Option<String>,
    pub is_loaded: bool,
    /// All models in the cache (loaded + unloaded), for status display.
    pub cached_models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ActiveGenerationSnapshot {
    pub model: String,
    pub prompt_sha256: String,
    pub started_at_unix_ms: u64,
    pub started_at: Instant,
}

// ── Generation queue types ──────────────────────────────────────────────────

/// Internal SSE message type used by both the queue worker and SSE streams.
pub enum SseMessage {
    Progress(mold_core::SseProgressEvent),
    Complete(Box<mold_core::SseCompleteEvent>),
    BatchComplete(Box<mold_core::SseBatchCompleteEvent>),
    UpscaleComplete(mold_core::SseUpscaleCompleteEvent),
    Error(mold_core::SseErrorEvent),
}

/// Media bytes included in an SSE completion. Mobile clients request
/// metadata-only completions and load the saved file with Range support.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SseCompletionPayload {
    #[default]
    Full,
    MetadataOnly,
}

/// A generation job submitted to the queue worker.
pub struct GenerationJob {
    /// Server-assigned UUIDv4. Echoed back to the client in the initial
    /// `SseProgressEvent::Queued` event so the SPA can later reconcile
    /// a persisted "running" card against `GET /api/queue` — anything
    /// the registry doesn't know about is a zombie left over from a
    /// dropped SSE stream and gets dead-lettered client-side. Always
    /// non-empty for jobs created through the public API; tests that
    /// construct `GenerationJob` directly may leave it empty.
    pub id: String,
    pub request: mold_core::GenerateRequest,
    pub completion_payload: SseCompletionPayload,
    /// Channel to send SSE progress/complete/error events (None for non-streaming).
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    /// Oneshot to return the final result for non-streaming callers.
    pub result_tx: tokio::sync::oneshot::Sender<Result<GenerationJobResult, String>>,
    /// Pre-resolved output directory for server-side image saving.
    pub output_dir: Option<PathBuf>,
    /// Server-owned adaptive-batch child authority. Public singleton and
    /// client-owned prepared siblings leave this absent.
    pub batch_child: Option<BatchChildExecution>,
}

#[derive(Clone, Debug)]
pub struct BatchChildExecution {
    pub lease: crate::batch_parent::BatchChildLease,
    /// Attempt-scoped cooperative cancellation authority. Cancelling the
    /// parent signals every active child through the exact token returned by
    /// its durable lease grant.
    pub cancellation: mold_inference::InferenceCancellationToken,
    /// Parent-level deterministic execution identity. The coordinator filters
    /// every later re-resolution through this fence.
    pub execution_equivalence_fingerprint: String,
    /// Concrete dependency variants frozen once for the whole parent.
    pub prepared_inputs: crate::execution_plan::PreparedExecutionInputs,
}

pub struct GenerationJobResult {
    pub response: mold_core::GenerateResponse,
    pub image: mold_core::ImageData,
}

/// Handle for submitting jobs to the generation queue.
#[derive(Clone)]
pub struct QueueHandle {
    job_tx: tokio::sync::mpsc::Sender<GenerationJob>,
    pending_count: Arc<AtomicUsize>,
    capacity_notify: Arc<tokio::sync::Notify>,
    capacity_waiter: Arc<tokio::sync::Mutex<()>>,
}

/// Reason a `QueueHandle::submit` attempt failed.
#[derive(Debug)]
pub enum SubmitError {
    /// Queue is at capacity — caller should return 503 with `Retry-After`.
    Full {
        pending: usize,
        capacity: usize,
    },
    Cancelled,
    /// Receiving end is gone (server shutting down).
    Shutdown,
}

impl QueueHandle {
    pub fn new(job_tx: tokio::sync::mpsc::Sender<GenerationJob>) -> Self {
        Self {
            job_tx,
            pending_count: Arc::new(AtomicUsize::new(0)),
            capacity_notify: Arc::new(tokio::sync::Notify::new()),
            capacity_waiter: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Submit a generation job.
    ///
    /// Atomically reserves a slot against `capacity` using fetch_add, so a
    /// burst of concurrent callers cannot all slip past a separate pending()
    /// pre-check (TOCTOU).  Returns the queue position on success.
    pub async fn submit(&self, job: GenerationJob, capacity: usize) -> Result<usize, SubmitError> {
        let prev = self.pending_count.fetch_add(1, Ordering::SeqCst);
        if prev >= capacity {
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            return Err(SubmitError::Full {
                pending: prev,
                capacity,
            });
        }
        if self.job_tx.send(job).await.is_err() {
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            self.capacity_notify.notify_waiters();
            return Err(SubmitError::Shutdown);
        }
        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_queue_submit();
            crate::metrics::record_queue_depth(self.pending_count.load(Ordering::SeqCst));
        }
        Ok(prev)
    }

    /// Wait for shared queue capacity without converting temporary occupancy
    /// into a terminal batch failure. Cancellation before transport drops the
    /// unsubmitted job and releases any reserved counter slot.
    pub async fn submit_when_available(
        &self,
        job: GenerationJob,
        capacity: usize,
        cancellation: &tokio_util::sync::CancellationToken,
    ) -> Result<usize, SubmitError> {
        let mut job = Some(job);
        let waiter = self.capacity_waiter.lock();
        tokio::pin!(waiter);
        let _waiter = tokio::select! {
            guard = &mut waiter => guard,
            _ = cancellation.cancelled() => return Err(SubmitError::Cancelled),
        };
        loop {
            if cancellation.is_cancelled() {
                return Err(SubmitError::Cancelled);
            }
            let notified = self.capacity_notify.notified();
            let previous = self.pending_count.fetch_add(1, Ordering::SeqCst);
            if previous < capacity {
                let send = self
                    .job_tx
                    .send(job.take().expect("batch queue job submitted once"));
                tokio::pin!(send);
                return tokio::select! {
                    result = &mut send => {
                        if result.is_err() {
                            self.pending_count.fetch_sub(1, Ordering::SeqCst);
                            self.capacity_notify.notify_waiters();
                            Err(SubmitError::Shutdown)
                        } else {
                            Ok(previous)
                        }
                    }
                    _ = cancellation.cancelled() => {
                        self.pending_count.fetch_sub(1, Ordering::SeqCst);
                        self.capacity_notify.notify_waiters();
                        Err(SubmitError::Cancelled)
                    }
                };
            }
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            tokio::select! {
                _ = notified => {}
                _ = cancellation.cancelled() => return Err(SubmitError::Cancelled),
            }
        }
    }

    pub fn decrement(&self) {
        let _ = self
            .pending_count
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |pending| {
                Some(pending.saturating_sub(1))
            });
        self.capacity_notify.notify_one();
    }

    pub fn pending(&self) -> usize {
        self.pending_count.load(Ordering::SeqCst)
    }
}

// ── AppState ────────────────────────────────────────────────────────────────

/// Cached server-assisted LAN discovery state. The mDNS browser updates this
/// from a dedicated thread; HTTP handlers only take short read locks.
#[derive(Default)]
pub struct DiscoveryState {
    can_browse: std::sync::atomic::AtomicBool,
    pub peers: std::sync::RwLock<Vec<mold_core::DiscoveryPeer>>,
}

impl DiscoveryState {
    pub fn can_browse(&self) -> bool {
        self.can_browse.load(Ordering::SeqCst)
    }

    pub fn set_can_browse(&self, enabled: bool) {
        self.can_browse.store(enabled, Ordering::SeqCst);
    }

    /// Mark the browser unavailable and prevent stale peers from being served.
    pub fn mark_unavailable(&self) {
        self.peers
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.can_browse.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone)]
pub struct AppState {
    /// Stable UUID identifying this server installation. `run_server`
    /// overwrites the constructor's ephemeral default with the persisted id
    /// from `crate::instance::resolve_instance_id` before the router (and the
    /// mDNS TXT records) are built.
    pub instance_id: Arc<String>,
    /// Long-lived DNS-SD browser cache used by `/api/discovery/peers`.
    pub discovery: Arc<DiscoveryState>,
    // ── Multi-GPU fields ────────────────────────────────────────────────────
    /// GPU worker pool for multi-GPU dispatch.
    pub gpu_pool: Arc<GpuPool>,
    /// `Some` means inference is intentionally unavailable (for example
    /// `--gpus none`, or a GPU build whose visible devices have no usable
    /// stable identity). Read routes and downloads remain available.
    pub generation_unavailable_reason: Arc<RwLock<Option<String>>>,
    /// Read-only authoritative device inventory and preference projection.
    /// Dispatch remains on `gpu_pool` until scheduler V2 lands.
    pub device_registry: Arc<crate::device_registry::DeviceRegistry>,
    /// Maximum queue capacity (for status reporting and 503 responses).
    pub queue_capacity: usize,

    // ── Legacy single-GPU fields (retained during migration) ────────────────
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// Uses std::sync::RwLock (not tokio) because it's only accessed from
    /// synchronous contexts (inside spawn_blocking closures and brief reads).
    /// Must never be held across an .await point.
    pub active_generation: Arc<RwLock<Option<ActiveGenerationSnapshot>>>,
    pub config: Arc<tokio::sync::RwLock<Config>>,
    /// Hard output kill switch used by isolated server test fixtures.
    ///
    /// This is intentionally separate from `Config`: environment overrides
    /// remain authoritative in production, while mock-generation fixtures
    /// must never inherit any process-level `MOLD_OUTPUT_DIR`.
    pub output_disabled_override: bool,
    pub start_time: Instant,
    /// Guards concurrent model loads and hot-swaps.
    pub model_load_lock: Arc<Mutex<()>>,
    /// Guards concurrent pulls — only one download at a time.
    pub pull_lock: Arc<Mutex<()>>,
    /// Generation request queue.
    pub queue: QueueHandle,
    /// Scheduler-owned ingress for GPU-consuming utility and administrative
    /// work. In GPU-worker mode this is the only route to an owner thread.
    pub scheduled_work: crate::scheduler::ScheduledWorkHandle,
    /// Authoritative registry of in-flight jobs (queued + running). Powers
    /// `GET /api/queue` so the SPA can reconcile persisted "running" cards
    /// against server reality and dead-letter zombies whose SSE stream
    /// silently dropped.
    pub job_registry: SharedJobRegistry,
    /// Serializes queue mutations with the scheduler's final lease claim.
    ///
    /// The registry's own lock protects each edit, but the scheduler also
    /// needs pause state and its final worker send to be one transaction.
    /// Mutation routes await this fence before returning, so a successful
    /// response is an acknowledgement that no older plan can still grant the
    /// pre-mutation state.
    pub scheduler_mutation_fence: Arc<tokio::sync::Mutex<()>>,
    /// Dispatch pause gate toggled by `POST /api/queue/pause` /
    /// `POST /api/queue/resume`. The dispatch loops park on this at the top of
    /// each iteration; a job already running on a worker finishes untouched.
    pub queue_pause: Arc<crate::queue::QueuePause>,
    /// Shared tokenizer pool for cross-engine caching.
    pub shared_pool: Arc<std::sync::Mutex<SharedPool>>,
    /// Shutdown trigger for graceful shutdown via `/api/shutdown` endpoint.
    pub shutdown_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    /// Cached upscaler engine to avoid recreating per request. Small models (2-64MB), single slot.
    pub upscaler_cache: Arc<std::sync::Mutex<Option<Box<dyn mold_inference::UpscaleEngine>>>>,
    /// SQLite-backed gallery metadata store. `None` when MOLD_DB_DISABLE=1 or
    /// when MOLD_HOME could not be resolved — callers must fall back to the
    /// filesystem walk in `routes::scan_gallery_dir`.
    pub metadata_db: Arc<Option<mold_db::MetadataDb>>,
    /// Global logical publication barrier for every gallery observer and
    /// mutator. Batch commits take the writer side until files, DB rows, and
    /// the durable manifest agree.
    pub gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate,
    /// Durable chain-job runner handle. `None` when DB-backed chain jobs are
    /// unavailable; chain-job API handlers return 503 in that state.
    pub chain_jobs: Option<Arc<crate::chain_job_runner::ChainJobRunnerHandle>>,
    // ── Downloads UI (Agent A) ──────────────────────────────────────────────
    /// Bounded-parallel download queue.
    pub downloads: Arc<DownloadQueue>,
    /// Always-on resource telemetry (Agent B).
    pub resources: Arc<ResourceBroadcaster>,
    /// Server-wide lifecycle broadcast backing `GET /api/events` — job
    /// queued/started/ended (mirrored by `job_registry`) plus gallery
    /// added/removed. One SSE connection observes the whole server.
    pub events: Arc<EventBroadcaster>,
    // ── Catalog (live HF + Civitai) ─────────────────────────────────────────
    /// In-process TTL cache backing `/api/catalog/search`.
    pub catalog_live_cache: mold_catalog::live::LiveCache,
    /// Upstream base URL for live Civitai search. Production runs with
    /// the public host; tests override via `with_civitai_base`.
    pub catalog_live_civitai_base: Arc<String>,
    /// Cache of pure synthesis intents for installed catalog (`cv:*`/`hf:*`)
    /// IDs, keyed by ID. Resolution into a `ModelConfig` is deferred to
    /// engine-load time so disk-state races (file present vs not) can't
    /// seal a wrong config; see `model_manager::resolve_intent_to_paths`.
    pub catalog_intents: Arc<
        tokio::sync::RwLock<
            std::collections::HashMap<String, mold_catalog::synthesis::CatalogModelIntent>,
        >,
    >,
    /// Cached disk snapshot for the models dir, served by `/api/status`.
    /// Refreshed off the request path — statvfs can hang outright on a
    /// wedged FUSE mount, so the handler must never wait on a disk scan.
    pub models_disk_cache: Arc<ModelsDiskCache>,
}

/// TTL for the cached models-disk snapshot. Matches the ~10s status poll
/// cadence: pollers see numbers at most one refresh stale.
pub const MODELS_DISK_TTL: std::time::Duration = std::time::Duration::from_secs(15);

/// Last-good disk stats for the models dir plus a single-flight refresh
/// claim. `/api/status` reads the snapshot synchronously and — when it has
/// gone stale — at most one request wins the claim and kicks a background
/// `spawn_blocking` refresh; everyone else keeps serving the last-good value.
/// A hung statvfs therefore costs one leaked blocking-pool thread and a stale
/// stat, never a parked runtime worker.
#[derive(Default)]
pub struct ModelsDiskCache {
    inner: std::sync::Mutex<ModelsDiskSnapshot>,
    refreshing: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct ModelsDiskSnapshot {
    usage: Option<mold_core::DiskUsage>,
    fetched_at: Option<Instant>,
}

impl ModelsDiskCache {
    /// Current snapshot plus whether the caller won the refresh claim.
    /// Returns `true` at most once per stale period — the winner must call
    /// [`ModelsDiskCache::store`] (or [`ModelsDiskCache::release`]) to let
    /// the next refresh happen.
    pub fn read(&self) -> (Option<mold_core::DiskUsage>, bool) {
        self.read_at(Instant::now())
    }

    fn read_at(&self, now: Instant) -> (Option<mold_core::DiskUsage>, bool) {
        let (usage, stale) = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let stale = inner
                .fetched_at
                .is_none_or(|at| now.duration_since(at) >= MODELS_DISK_TTL);
            (inner.usage.clone(), stale)
        };
        if !stale {
            return (usage, false);
        }
        let won_claim = !self
            .refreshing
            .swap(true, std::sync::atomic::Ordering::AcqRel);
        (usage, won_claim)
    }

    /// Publish a fresh snapshot and release the refresh claim.
    pub fn store(&self, usage: Option<mold_core::DiskUsage>) {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.usage = usage;
            inner.fetched_at = Some(Instant::now());
        }
        self.release();
    }

    /// Release the refresh claim without publishing (refresh task failed).
    pub fn release(&self) {
        self.refreshing
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Default TTL for the live-search cache. Five minutes is long enough
/// to absorb SPA re-mounts and quick paging without serving stale
/// rankings.
const CATALOG_LIVE_CACHE_TTL_SECS: u64 = 300;
/// Cap on cached query keys. The working set per user is small —
/// caps too high just retain stale rows past their TTL.
const CATALOG_LIVE_CACHE_MAX_KEYS: usize = 256;
/// Production Civitai endpoint. Tests inject a wiremock URI via
/// [`AppState::with_civitai_base`].
pub const CATALOG_LIVE_CIVITAI_BASE: &str = "https://civitai.com";

fn default_live_cache() -> mold_catalog::live::LiveCache {
    mold_catalog::live::LiveCache::new(
        std::time::Duration::from_secs(CATALOG_LIVE_CACHE_TTL_SECS),
        CATALOG_LIVE_CACHE_MAX_KEYS,
    )
}

/// Default maximum number of cached models (GPU-resident + parked engine structs).
pub const DEFAULT_MAX_CACHED_MODELS: usize = 3;
/// Lower / upper bounds applied to env-overridden cache caps. Below 1 the
/// cache can't hold the active engine; above 16 the OOM risk dwarfs the
/// hit-rate gains for a typical local server.
const MAX_CACHED_MODELS_LOWER: usize = 1;
const MAX_CACHED_MODELS_UPPER: usize = 16;
/// Env var that overrides `DEFAULT_MAX_CACHED_MODELS` at runtime.
pub const MAX_CACHED_MODELS_ENV: &str = "MOLD_MAX_CACHED_MODELS";

/// Default idle-TTL for parked cache entries — 30 minutes. Tuned for a
/// local-first workflow: long enough that a user toggling between two
/// models inside a session never pays a reload, short enough that
/// background memory pressure doesn't accumulate overnight.
pub const DEFAULT_CACHE_IDLE_TTL_SECS: u64 = 1800;
const CACHE_IDLE_TTL_LOWER_SECS: u64 = 60;
const CACHE_IDLE_TTL_UPPER_SECS: u64 = 86_400;
/// Env var that overrides `DEFAULT_CACHE_IDLE_TTL_SECS`.
pub const CACHE_IDLE_TTL_ENV: &str = "MOLD_CACHE_IDLE_TTL_SECS";

/// Resolve the cache idle-TTL from env, falling back to the default.
/// Out-of-range or unparseable values log a warning and use the default.
pub fn resolve_cache_idle_ttl_secs() -> u64 {
    match std::env::var(CACHE_IDLE_TTL_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(n) if (CACHE_IDLE_TTL_LOWER_SECS..=CACHE_IDLE_TTL_UPPER_SECS).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    env = CACHE_IDLE_TTL_ENV,
                    value = n,
                    lower = CACHE_IDLE_TTL_LOWER_SECS,
                    upper = CACHE_IDLE_TTL_UPPER_SECS,
                    "ignoring out-of-range cache idle-TTL; using default"
                );
                DEFAULT_CACHE_IDLE_TTL_SECS
            }
            Err(e) => {
                tracing::warn!(
                    env = CACHE_IDLE_TTL_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable cache idle-TTL; using default"
                );
                DEFAULT_CACHE_IDLE_TTL_SECS
            }
        },
        Err(_) => DEFAULT_CACHE_IDLE_TTL_SECS,
    }
}

/// Resolve the model-cache capacity from env, falling back to the default.
/// Out-of-range or unparseable values log a warning and use the default so
/// a typo in the env never silently shrinks the cache to an unusable size.
pub fn resolve_max_cached_models() -> usize {
    match std::env::var(MAX_CACHED_MODELS_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if (MAX_CACHED_MODELS_LOWER..=MAX_CACHED_MODELS_UPPER).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    env = MAX_CACHED_MODELS_ENV,
                    value = n,
                    lower = MAX_CACHED_MODELS_LOWER,
                    upper = MAX_CACHED_MODELS_UPPER,
                    "ignoring out-of-range cache cap; using default"
                );
                DEFAULT_MAX_CACHED_MODELS
            }
            Err(e) => {
                tracing::warn!(
                    env = MAX_CACHED_MODELS_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable cache cap; using default"
                );
                DEFAULT_MAX_CACHED_MODELS
            }
        },
        Err(_) => DEFAULT_MAX_CACHED_MODELS,
    }
}

impl AppState {
    /// Test states must never inherit the production gallery fallback.
    ///
    /// `Config::default()` intentionally resolves output to
    /// `~/.mold/output`, which means a successful mock generation can write
    /// into a developer's real Library unless output is explicitly disabled.
    /// Persistence tests opt back in with their own temporary output dir.
    #[cfg(test)]
    pub(crate) fn test_config() -> Config {
        Config {
            output_dir: Some(String::new()),
            ..Config::default()
        }
    }

    pub(crate) fn is_output_disabled(&self, config: &Config) -> bool {
        self.output_disabled_override || config.is_output_disabled()
    }

    /// Create state with a pre-loaded engine (server starts with a configured model).
    pub fn new(
        engine: Box<dyn InferenceEngine>,
        config: Config,
        queue: QueueHandle,
        gpu_pool: Arc<GpuPool>,
        queue_capacity: usize,
    ) -> Self {
        let mut cache = ModelCache::new(resolve_max_cached_models());
        cache.insert(engine, 0);
        let events = EventBroadcaster::new();
        Self {
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(DiscoveryState::default()),
            gpu_pool,
            generation_unavailable_reason: Arc::new(RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            output_disabled_override: false,
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: JobRegistry::with_events(events.clone()),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            events,
            catalog_live_cache: default_live_cache(),
            catalog_live_civitai_base: Arc::new(CATALOG_LIVE_CIVITAI_BASE.to_string()),
            catalog_intents: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            models_disk_cache: Arc::new(ModelsDiskCache::default()),
        }
    }

    /// Create state with no engine (zero-config startup, models pulled on demand).
    pub fn empty(
        config: Config,
        queue: QueueHandle,
        gpu_pool: Arc<GpuPool>,
        queue_capacity: usize,
    ) -> Self {
        let discovered = gpu_pool
            .worker_snapshot()
            .into_iter()
            .map(|worker| worker.gpu.clone())
            .collect::<Vec<_>>();
        let device_registry = if discovered.is_empty() {
            crate::device_registry::DeviceRegistry::empty()
        } else {
            Arc::new(
                crate::device_registry::DeviceRegistry::from_worker_inventory(
                    discovered,
                    Arc::new(None),
                ),
            )
        };
        Self::empty_with_device_registry(config, queue, gpu_pool, queue_capacity, device_registry)
    }

    pub(crate) fn empty_with_device_registry(
        config: Config,
        queue: QueueHandle,
        gpu_pool: Arc<GpuPool>,
        queue_capacity: usize,
        device_registry: Arc<crate::device_registry::DeviceRegistry>,
    ) -> Self {
        let events = EventBroadcaster::new();
        Self {
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(DiscoveryState::default()),
            gpu_pool,
            generation_unavailable_reason: Arc::new(RwLock::new(None)),
            device_registry,
            queue_capacity,
            model_cache: Arc::new(Mutex::new(ModelCache::new(resolve_max_cached_models()))),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            output_disabled_override: false,
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: JobRegistry::with_events(events.clone()),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            events,
            catalog_live_cache: default_live_cache(),
            catalog_live_civitai_base: Arc::new(CATALOG_LIVE_CIVITAI_BASE.to_string()),
            catalog_intents: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            models_disk_cache: Arc::new(ModelsDiskCache::default()),
        }
    }

    /// Create an empty GpuPool for testing (no GPU workers).
    #[cfg(test)]
    pub(crate) fn empty_gpu_pool() -> Arc<GpuPool> {
        Arc::new(GpuPool {
            workers: Vec::new().into(),
        })
    }

    /// Alias for `empty_gpu_pool` — exposed for tests in sibling modules
    /// (routes_test.rs, downloads_test.rs) that live in the crate but not
    /// in the same file.
    #[cfg(test)]
    pub(crate) fn empty_gpu_pool_for_test() -> Arc<GpuPool> {
        Self::empty_gpu_pool()
    }

    pub fn set_generation_unavailable(&self, reason: impl Into<String>) {
        *self
            .generation_unavailable_reason
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(reason.into());
    }

    pub fn generation_unavailable(&self) -> Option<String> {
        if self.device_registry.all_startup_allowed_devices_disabled() {
            return Some(
                "generation is unavailable because all GPU devices are disabled (maintenance mode)"
                    .to_string(),
            );
        }
        self.generation_unavailable_reason
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let mut cache = ModelCache::new(resolve_max_cached_models());
        cache.insert(Box::new(engine), 0);
        let events = EventBroadcaster::new();
        Self {
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(DiscoveryState::default()),
            gpu_pool: Self::empty_gpu_pool(),
            generation_unavailable_reason: Arc::new(RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Self::test_config())),
            output_disabled_override: true,
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: JobRegistry::with_events(events.clone()),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            events,
            catalog_live_cache: default_live_cache(),
            catalog_live_civitai_base: Arc::new(CATALOG_LIVE_CIVITAI_BASE.to_string()),
            catalog_intents: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            models_disk_cache: Arc::new(ModelsDiskCache::default()),
        }
    }

    /// Create state with a queue whose receiver is returned for testing.
    #[cfg(test)]
    pub fn with_engine_and_queue(
        engine: impl InferenceEngine + 'static,
    ) -> (Self, tokio::sync::mpsc::Receiver<GenerationJob>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let mut cache = ModelCache::new(resolve_max_cached_models());
        cache.insert(Box::new(engine), 0);
        let events = EventBroadcaster::new();
        let state = Self {
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(DiscoveryState::default()),
            gpu_pool: Self::empty_gpu_pool(),
            generation_unavailable_reason: Arc::new(RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Self::test_config())),
            output_disabled_override: true,
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: JobRegistry::with_events(events.clone()),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            events,
            catalog_live_cache: default_live_cache(),
            catalog_live_civitai_base: Arc::new(CATALOG_LIVE_CIVITAI_BASE.to_string()),
            catalog_intents: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            models_disk_cache: Arc::new(ModelsDiskCache::default()),
        };
        (state, rx)
    }

    /// Construct an empty AppState for integration tests. Catalog
    /// surfaces hit live HF/Civitai (test callers point those at a
    /// wiremock instance via `with_civitai_base`).
    pub fn for_tests() -> Self {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let events = EventBroadcaster::new();
        Self {
            instance_id: Arc::new(uuid::Uuid::new_v4().to_string()),
            discovery: Arc::new(DiscoveryState::default()),
            gpu_pool: Arc::new(GpuPool {
                workers: Vec::new().into(),
            }),
            generation_unavailable_reason: Arc::new(RwLock::new(None)),
            device_registry: crate::device_registry::DeviceRegistry::empty(),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(ModelCache::new(resolve_max_cached_models()))),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            output_disabled_override: false,
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            scheduled_work: crate::scheduler::ScheduledWorkHandle::default(),
            job_registry: JobRegistry::with_events(events.clone()),
            scheduler_mutation_fence: Arc::new(tokio::sync::Mutex::new(())),
            queue_pause: crate::queue::QueuePause::new(),
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            gallery_publication_gate: crate::batch_transaction::GalleryPublicationGate::default(),
            chain_jobs: None,
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            events,
            catalog_live_cache: default_live_cache(),
            catalog_live_civitai_base: Arc::new(CATALOG_LIVE_CIVITAI_BASE.to_string()),
            catalog_intents: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            models_disk_cache: Arc::new(ModelsDiskCache::default()),
        }
    }

    /// Override the live-search Civitai base URL on an existing state.
    /// Tests point this at a wiremock instance; production never calls
    /// it (the `new` / `empty` constructors set the public host).
    pub fn with_civitai_base(mut self, base: impl Into<String>) -> Self {
        self.catalog_live_civitai_base = Arc::new(base.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue_job(id: &str) -> GenerationJob {
        let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "queue",
            "model": "flux-dev:q8",
            "width": 64,
            "height": 64,
            "steps": 1
        }))
        .unwrap();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        GenerationJob {
            id: id.to_string(),
            request,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            batch_child: None,
        }
    }

    #[test]
    fn models_disk_cache_serves_last_good_and_single_flights_refreshes() {
        let cache = ModelsDiskCache::default();

        // Fresh cache: nothing to serve, first reader wins the refresh claim…
        let (usage, refresh) = cache.read();
        assert_eq!(usage, None);
        assert!(refresh, "first reader must win the refresh claim");
        // …and nobody else does until the refresh completes (single-flight —
        // a hung statvfs must not leak one blocking thread per status poll).
        let (usage, refresh) = cache.read();
        assert_eq!(usage, None);
        assert!(!refresh, "claim must not be handed out twice");

        // A completed refresh publishes and is served without re-claiming.
        let stats = mold_core::DiskUsage {
            total_bytes: 500,
            free_bytes: 50,
        };
        cache.store(Some(stats.clone()));
        let (usage, refresh) = cache.read();
        assert_eq!(usage, Some(stats.clone()));
        assert!(!refresh, "fresh snapshot must not trigger a refresh");

        // Past the TTL the stale value keeps being served while exactly one
        // reader wins the next refresh claim.
        let later = Instant::now() + MODELS_DISK_TTL;
        let (usage, refresh) = cache.read_at(later);
        assert_eq!(usage, Some(stats.clone()), "stale value still served");
        assert!(refresh, "stale snapshot must trigger one refresh");
        let (usage, refresh) = cache.read_at(later);
        assert_eq!(usage, Some(stats));
        assert!(!refresh);
    }

    #[test]
    fn models_disk_cache_release_reopens_the_claim_without_publishing() {
        let cache = ModelsDiskCache::default();
        let (_, refresh) = cache.read();
        assert!(refresh);
        // A refresh task that failed must hand the claim back, or the cache
        // would never refresh again for the life of the process.
        cache.release();
        let (usage, refresh) = cache.read();
        assert_eq!(usage, None, "release publishes nothing");
        assert!(refresh, "claim must be available again after release");
    }

    #[test]
    fn unavailable_discovery_clears_cached_peers() {
        let discovery = DiscoveryState::default();
        discovery.set_can_browse(true);
        discovery
            .peers
            .write()
            .unwrap()
            .push(mold_core::DiscoveryPeer {
                name: "studio-7680".to_string(),
                url: "http://192.168.1.20:7680".to_string(),
                host: "192.168.1.20".to_string(),
                port: 7680,
                version: None,
                auth_required: false,
                instance_id: Some("studio-instance".to_string()),
                is_this_machine: false,
            });

        discovery.mark_unavailable();

        assert!(!discovery.can_browse());
        assert!(discovery.peers.read().unwrap().is_empty());
    }

    #[test]
    fn engine_snapshot_default_is_unloaded() {
        let snap = EngineSnapshot::default();
        assert!(snap.model_name.is_none());
        assert!(!snap.is_loaded);
    }

    #[test]
    fn active_generation_snapshot_stores_fields() {
        let snap = ActiveGenerationSnapshot {
            model: "flux-dev:q8".to_string(),
            prompt_sha256: "abc123".to_string(),
            started_at_unix_ms: 1700000000000,
            started_at: std::time::Instant::now(),
        };
        assert_eq!(snap.model, "flux-dev:q8");
        assert_eq!(snap.prompt_sha256, "abc123");
        assert_eq!(snap.started_at_unix_ms, 1700000000000);
    }

    #[test]
    fn queue_handle_pending_starts_at_zero() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<GenerationJob>(16);
        let handle = QueueHandle::new(tx);
        assert_eq!(handle.pending(), 0);
    }

    #[test]
    fn queue_handle_decrement_saturates_at_zero() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<GenerationJob>(1);
        let handle = QueueHandle::new(tx);

        handle.decrement();
        handle.decrement();

        assert_eq!(handle.pending(), 0);
    }

    #[tokio::test]
    async fn waitable_submission_backpressures_on_unrelated_occupancy() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let handle = QueueHandle::new(tx);
        handle.submit(queue_job("ordinary"), 1).await.unwrap();

        let cancellation = tokio_util::sync::CancellationToken::new();
        let waiting = tokio::spawn({
            let handle = handle.clone();
            let cancellation = cancellation.clone();
            async move {
                handle
                    .submit_when_available(queue_job("batch"), 1, &cancellation)
                    .await
            }
        });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        assert_eq!(
            handle.pending(),
            1,
            "waiting must not reserve over capacity"
        );

        assert_eq!(rx.recv().await.unwrap().id, "ordinary");
        handle.decrement();
        assert_eq!(waiting.await.unwrap().unwrap(), 0);
        assert_eq!(rx.recv().await.unwrap().id, "batch");
        assert_eq!(handle.pending(), 1);
    }

    #[tokio::test]
    async fn waitable_submission_is_fifo_and_cancellation_releases_its_turn() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let handle = QueueHandle::new(tx);
        handle.submit(queue_job("occupied"), 1).await.unwrap();

        let first_cancel = tokio_util::sync::CancellationToken::new();
        let first = tokio::spawn({
            let handle = handle.clone();
            let cancellation = first_cancel.clone();
            async move {
                handle
                    .submit_when_available(queue_job("first"), 1, &cancellation)
                    .await
            }
        });
        tokio::task::yield_now().await;
        let second = tokio::spawn({
            let handle = handle.clone();
            async move {
                handle
                    .submit_when_available(
                        queue_job("second"),
                        1,
                        &tokio_util::sync::CancellationToken::new(),
                    )
                    .await
            }
        });
        tokio::task::yield_now().await;
        first_cancel.cancel();
        assert!(matches!(first.await.unwrap(), Err(SubmitError::Cancelled)));
        assert_eq!(handle.pending(), 1);

        assert_eq!(rx.recv().await.unwrap().id, "occupied");
        handle.decrement();
        assert_eq!(second.await.unwrap().unwrap(), 0);
        assert_eq!(rx.recv().await.unwrap().id, "second");
    }

    #[test]
    fn upscaler_cache_starts_empty() {
        let config = mold_core::Config::default();
        let state = AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        let cache = state.upscaler_cache.lock().unwrap();
        assert!(cache.is_none());
    }

    #[test]
    fn upscaler_cache_cleared_by_setting_none() {
        let config = mold_core::Config::default();
        let state = AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        {
            let mut cache = state.upscaler_cache.lock().unwrap();
            *cache = None;
        }
        let cache = state.upscaler_cache.lock().unwrap();
        assert!(cache.is_none());
    }

    #[test]
    fn app_state_exposes_resources_broadcaster() {
        let config = mold_core::Config::default();
        let state = AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            AppState::empty_gpu_pool(),
            200,
        );
        // The broadcaster must exist and return None before any aggregator tick.
        assert!(state.resources.latest().is_none());
        // Subscribing must succeed (no panics).
        let _rx = state.resources.subscribe();
    }

    /// Serializes every test that touches a process-wide env var via
    /// `std::env::set_var` — mutating env is global state, so without this
    /// guard parallel tests would race on the env table.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Set `name` to `value` (or remove it when `value` is `None`), invoke
    /// `f`, then restore the original value. Lock-serialized so concurrent
    /// tests don't race on the env table.
    fn with_env<R>(name: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var(name).ok();
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        out
    }

    #[test]
    fn resolve_max_cached_uses_default_when_env_missing() {
        let n = with_env(MAX_CACHED_MODELS_ENV, None, resolve_max_cached_models);
        assert_eq!(n, DEFAULT_MAX_CACHED_MODELS);
    }

    #[test]
    fn resolve_max_cached_honors_env_within_range() {
        let n = with_env(MAX_CACHED_MODELS_ENV, Some("8"), resolve_max_cached_models);
        assert_eq!(n, 8);
    }

    #[test]
    fn resolve_max_cached_clamps_zero_back_to_default() {
        let n = with_env(MAX_CACHED_MODELS_ENV, Some("0"), resolve_max_cached_models);
        assert_eq!(n, DEFAULT_MAX_CACHED_MODELS);
    }

    #[test]
    fn resolve_max_cached_clamps_overflow_back_to_default() {
        let n = with_env(
            MAX_CACHED_MODELS_ENV,
            Some("999"),
            resolve_max_cached_models,
        );
        assert_eq!(n, DEFAULT_MAX_CACHED_MODELS);
    }

    #[test]
    fn resolve_max_cached_falls_back_when_env_unparseable() {
        let n = with_env(
            MAX_CACHED_MODELS_ENV,
            Some("not-a-number"),
            resolve_max_cached_models,
        );
        assert_eq!(n, DEFAULT_MAX_CACHED_MODELS);
    }

    #[test]
    fn resolve_cache_idle_ttl_uses_default_when_env_missing() {
        let n = with_env(CACHE_IDLE_TTL_ENV, None, resolve_cache_idle_ttl_secs);
        assert_eq!(n, DEFAULT_CACHE_IDLE_TTL_SECS);
    }

    #[test]
    fn resolve_cache_idle_ttl_honors_env_within_range() {
        // 600s is comfortably inside [60, 86_400] — the resolver must echo it.
        let n = with_env(CACHE_IDLE_TTL_ENV, Some("600"), resolve_cache_idle_ttl_secs);
        assert_eq!(n, 600);
    }

    #[test]
    fn resolve_cache_idle_ttl_clamps_zero_back_to_default() {
        // 0 is below the 60s lower bound; falls back to the default with a warn log.
        let n = with_env(CACHE_IDLE_TTL_ENV, Some("0"), resolve_cache_idle_ttl_secs);
        assert_eq!(n, DEFAULT_CACHE_IDLE_TTL_SECS);
    }

    #[test]
    fn resolve_cache_idle_ttl_clamps_overflow_back_to_default() {
        // 100_000 is above the 86_400s upper bound; falls back to the default.
        let n = with_env(
            CACHE_IDLE_TTL_ENV,
            Some("100000"),
            resolve_cache_idle_ttl_secs,
        );
        assert_eq!(n, DEFAULT_CACHE_IDLE_TTL_SECS);
    }

    #[test]
    fn resolve_cache_idle_ttl_falls_back_when_env_unparseable() {
        let n = with_env(
            CACHE_IDLE_TTL_ENV,
            Some("not-a-number"),
            resolve_cache_idle_ttl_secs,
        );
        assert_eq!(n, DEFAULT_CACHE_IDLE_TTL_SECS);
    }
}
