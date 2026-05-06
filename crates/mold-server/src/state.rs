use mold_core::Config;
use mold_inference::InferenceEngine;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::Mutex;

use mold_inference::shared_pool::SharedPool;

use crate::catalog_api::CatalogScanQueue;
use crate::downloads::DownloadQueue;
use crate::gpu_pool::GpuPool;
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
    Complete(mold_core::SseCompleteEvent),
    UpscaleComplete(mold_core::SseUpscaleCompleteEvent),
    Error(mold_core::SseErrorEvent),
}

/// A generation job submitted to the queue worker.
pub struct GenerationJob {
    pub request: mold_core::GenerateRequest,
    /// Channel to send SSE progress/complete/error events (None for non-streaming).
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    /// Oneshot to return the final result for non-streaming callers.
    pub result_tx: tokio::sync::oneshot::Sender<Result<GenerationJobResult, String>>,
    /// Pre-resolved output directory for server-side image saving.
    pub output_dir: Option<PathBuf>,
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
}

/// Reason a `QueueHandle::submit` attempt failed.
#[derive(Debug)]
pub enum SubmitError {
    /// Queue is at capacity — caller should return 503 with `Retry-After`.
    Full { pending: usize, capacity: usize },
    /// Receiving end is gone (server shutting down).
    Shutdown,
}

impl QueueHandle {
    pub fn new(job_tx: tokio::sync::mpsc::Sender<GenerationJob>) -> Self {
        Self {
            job_tx,
            pending_count: Arc::new(AtomicUsize::new(0)),
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
            return Err(SubmitError::Shutdown);
        }
        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_queue_submit();
            crate::metrics::record_queue_depth(self.pending_count.load(Ordering::SeqCst));
        }
        Ok(prev)
    }

    pub fn decrement(&self) {
        self.pending_count.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn pending(&self) -> usize {
        self.pending_count.load(Ordering::SeqCst)
    }
}

// ── AppState ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    // ── Multi-GPU fields ────────────────────────────────────────────────────
    /// GPU worker pool for multi-GPU dispatch.
    pub gpu_pool: Arc<GpuPool>,
    /// Maximum queue capacity (for status reporting and 503 responses).
    pub queue_capacity: usize,

    // ── Legacy single-GPU fields (retained during migration) ────────────────
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// Uses std::sync::RwLock (not tokio) because it's only accessed from
    /// synchronous contexts (inside spawn_blocking closures and brief reads).
    /// Must never be held across an .await point.
    pub active_generation: Arc<RwLock<Option<ActiveGenerationSnapshot>>>,
    pub config: Arc<tokio::sync::RwLock<Config>>,
    pub start_time: Instant,
    /// Guards concurrent model loads and hot-swaps.
    pub model_load_lock: Arc<Mutex<()>>,
    /// Guards concurrent pulls — only one download at a time.
    pub pull_lock: Arc<Mutex<()>>,
    /// Serializes chained video renders. The chain handler removes the
    /// engine from `model_cache` and runs blocking work outside that
    /// lock for the full multi-minute chain; without a dedicated lock two
    /// concurrent chain requests race on `cache.take()` and the loser
    /// surfaces "engine vanished from cache after ensure_model_ready".
    /// Held for the entire chain (load + all stages + restore); other
    /// single-clip requests continue to queue normally on `queue`.
    pub chain_lock: Arc<Mutex<()>>,
    /// Generation request queue.
    pub queue: QueueHandle,
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
    // ── Downloads UI (Agent A) ──────────────────────────────────────────────
    /// Single-writer download queue.
    pub downloads: Arc<DownloadQueue>,
    /// Always-on resource telemetry (Agent B).
    pub resources: Arc<ResourceBroadcaster>,
    // ── Catalog (sub-project A) ─────────────────────────────────────────────
    /// Single-writer catalog scan queue.
    pub catalog_scan: Arc<CatalogScanQueue>,
    /// Catalog + gallery metadata DB. Shared across catalog endpoints.
    pub catalog_db: Arc<mold_db::MetadataDb>,
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

/// Open the default catalog DB, falling back to in-memory if unavailable.
fn open_catalog_db() -> Arc<mold_db::MetadataDb> {
    // Try the real on-disk DB first (honours MOLD_DB_PATH / MOLD_HOME).
    if let Ok(Some(db)) = mold_db::open_default() {
        return Arc::new(db);
    }
    // Fall back to an ephemeral in-memory DB (tests and disabled-DB mode).
    Arc::new(mold_db::MetadataDb::open_in_memory().expect("in-memory DB"))
}

impl AppState {
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
        Self {
            gpu_pool,
            queue_capacity,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            chain_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            catalog_scan: Arc::new(CatalogScanQueue::new()),
            catalog_db: open_catalog_db(),
        }
    }

    /// Create state with no engine (zero-config startup, models pulled on demand).
    pub fn empty(
        config: Config,
        queue: QueueHandle,
        gpu_pool: Arc<GpuPool>,
        queue_capacity: usize,
    ) -> Self {
        Self {
            gpu_pool,
            queue_capacity,
            model_cache: Arc::new(Mutex::new(ModelCache::new(resolve_max_cached_models()))),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            chain_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            catalog_scan: Arc::new(CatalogScanQueue::new()),
            catalog_db: open_catalog_db(),
        }
    }

    /// Create an empty GpuPool for testing (no GPU workers).
    #[cfg(test)]
    pub(crate) fn empty_gpu_pool() -> Arc<GpuPool> {
        Arc::new(GpuPool {
            workers: Vec::new(),
        })
    }

    /// Alias for `empty_gpu_pool` — exposed for tests in sibling modules
    /// (routes_test.rs, downloads_test.rs) that live in the crate but not
    /// in the same file.
    #[cfg(test)]
    pub(crate) fn empty_gpu_pool_for_test() -> Arc<GpuPool> {
        Self::empty_gpu_pool()
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let mut cache = ModelCache::new(resolve_max_cached_models());
        cache.insert(Box::new(engine), 0);
        Self {
            gpu_pool: Self::empty_gpu_pool(),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            chain_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            catalog_scan: Arc::new(CatalogScanQueue::new()),
            catalog_db: Arc::new(mold_db::MetadataDb::open_in_memory().expect("in-memory DB")),
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
        let state = Self {
            gpu_pool: Self::empty_gpu_pool(),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(cache)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            chain_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            catalog_scan: Arc::new(CatalogScanQueue::new()),
            catalog_db: Arc::new(mold_db::MetadataDb::open_in_memory().expect("in-memory DB")),
        };
        (state, rx)
    }

    /// Create state wired to a specific catalog DB — used by integration tests
    /// that need pre-seeded catalog data.
    pub fn for_tests(catalog_db: Arc<mold_db::MetadataDb>) -> Self {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        Self {
            gpu_pool: Arc::new(GpuPool {
                workers: Vec::new(),
            }),
            queue_capacity: 200,
            model_cache: Arc::new(Mutex::new(ModelCache::new(resolve_max_cached_models()))),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            chain_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
            metadata_db: Arc::new(None),
            downloads: DownloadQueue::new(),
            resources: ResourceBroadcaster::new(),
            catalog_scan: Arc::new(CatalogScanQueue::new()),
            catalog_db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
