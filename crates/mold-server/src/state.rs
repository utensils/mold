use mold_core::Config;
use mold_inference::InferenceEngine;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::Mutex;

use mold_inference::shared_pool::SharedPool;

use crate::model_cache::ModelCache;

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

impl QueueHandle {
    pub fn new(job_tx: tokio::sync::mpsc::Sender<GenerationJob>) -> Self {
        Self {
            job_tx,
            pending_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Submit a generation job. Returns the queue position (0-based).
    pub async fn submit(&self, job: GenerationJob) -> Result<usize, String> {
        let position = self.pending_count.fetch_add(1, Ordering::SeqCst);
        if let Err(_e) = self.job_tx.send(job).await {
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            return Err("generation queue shut down".to_string());
        }
        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_queue_submit();
            crate::metrics::record_queue_depth(self.pending_count.load(Ordering::SeqCst));
        }
        Ok(position)
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
    pub model_cache: Arc<Mutex<ModelCache>>,
    pub engine_snapshot: Arc<tokio::sync::RwLock<EngineSnapshot>>,
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
    /// Generation request queue.
    pub queue: QueueHandle,
    /// Shared tokenizer pool for cross-engine caching.
    pub shared_pool: Arc<std::sync::Mutex<SharedPool>>,
    /// Shutdown trigger for graceful shutdown via `/api/shutdown` endpoint.
    pub shutdown_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    /// Cached upscaler engine to avoid recreating per request. Small models (2-64MB), single slot.
    pub upscaler_cache: Arc<std::sync::Mutex<Option<Box<dyn mold_inference::UpscaleEngine>>>>,
}

/// Default maximum number of cached models (loaded + unloaded engine structs).
const DEFAULT_MAX_CACHED_MODELS: usize = 3;

impl AppState {
    /// Create state with a pre-loaded engine (server starts with a configured model).
    pub fn new(engine: Box<dyn InferenceEngine>, config: Config, queue: QueueHandle) -> Self {
        let name = engine.model_name().to_string();
        let loaded = engine.is_loaded();
        let mut cache = ModelCache::new(DEFAULT_MAX_CACHED_MODELS);
        cache.insert(engine, 0);
        let snapshot = EngineSnapshot {
            model_name: Some(name),
            is_loaded: loaded,
            cached_models: cache.cached_model_names(),
        };
        Self {
            model_cache: Arc::new(Mutex::new(cache)),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(snapshot)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Create state with no engine (zero-config startup, models pulled on demand).
    pub fn empty(config: Config, queue: QueueHandle) -> Self {
        Self {
            model_cache: Arc::new(Mutex::new(ModelCache::new(DEFAULT_MAX_CACHED_MODELS))),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(EngineSnapshot::default())),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let name = engine.model_name().to_string();
        let loaded = engine.is_loaded();
        let mut cache = ModelCache::new(DEFAULT_MAX_CACHED_MODELS);
        cache.insert(Box::new(engine), 0);
        let snapshot = EngineSnapshot {
            model_name: Some(name),
            is_loaded: loaded,
            cached_models: cache.cached_model_names(),
        };
        Self {
            model_cache: Arc::new(Mutex::new(cache)),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(snapshot)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Create state with a queue whose receiver is returned for testing.
    #[cfg(test)]
    pub fn with_engine_and_queue(
        engine: impl InferenceEngine + 'static,
    ) -> (Self, tokio::sync::mpsc::Receiver<GenerationJob>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let queue = QueueHandle::new(tx);
        let name = engine.model_name().to_string();
        let loaded = engine.is_loaded();
        let mut cache = ModelCache::new(DEFAULT_MAX_CACHED_MODELS);
        cache.insert(Box::new(engine), 0);
        let snapshot = EngineSnapshot {
            model_name: Some(name),
            is_loaded: loaded,
            cached_models: cache.cached_model_names(),
        };
        let state = Self {
            model_cache: Arc::new(Mutex::new(cache)),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(snapshot)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
            queue,
            shared_pool: Arc::new(std::sync::Mutex::new(SharedPool::new())),
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(None)),
            upscaler_cache: Arc::new(std::sync::Mutex::new(None)),
        };
        (state, rx)
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
        let state = AppState::empty(config, QueueHandle::new(tokio::sync::mpsc::channel(1).0));
        let cache = state.upscaler_cache.lock().unwrap();
        assert!(cache.is_none());
    }

    #[test]
    fn upscaler_cache_cleared_by_setting_none() {
        let config = mold_core::Config::default();
        let state = AppState::empty(config, QueueHandle::new(tokio::sync::mpsc::channel(1).0));
        {
            let mut cache = state.upscaler_cache.lock().unwrap();
            *cache = None;
        }
        let cache = state.upscaler_cache.lock().unwrap();
        assert!(cache.is_none());
    }
}
