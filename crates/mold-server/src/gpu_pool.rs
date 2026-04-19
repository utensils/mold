use crate::model_cache::{ModelCache, ModelResidency};
use mold_core::types::{GpuWorkerState, GpuWorkerStatus};
use mold_inference::device::DiscoveredGpu;
use mold_inference::shared_pool::SharedPool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Per-GPU worker state. Each GPU gets its own model cache, load lock, and health tracking.
pub struct GpuWorker {
    pub gpu: DiscoveredGpu,
    pub model_cache: Arc<Mutex<ModelCache>>,
    pub active_generation: Arc<RwLock<Option<ActiveGeneration>>>,
    pub model_load_lock: Arc<Mutex<()>>,
    pub shared_pool: Arc<Mutex<SharedPool>>,
    pub in_flight: AtomicUsize,
    pub consecutive_failures: AtomicUsize,
    pub degraded_until: RwLock<Option<Instant>>,
    pub job_tx: std::sync::mpsc::SyncSender<GpuJob>,
}

/// Tracks the currently active generation on a GPU worker.
#[derive(Debug)]
pub struct ActiveGeneration {
    pub model: String,
    pub started_at: Instant,
}

/// A job dispatched to a GPU worker thread for processing.
pub struct GpuJob {
    pub model: String,
    pub request: mold_core::GenerateRequest,
    pub progress_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::state::SseMessage>>,
    pub result_tx: tokio::sync::oneshot::Sender<Result<crate::state::GenerationJobResult, String>>,
    pub output_dir: Option<std::path::PathBuf>,
    pub config: Arc<tokio::sync::RwLock<mold_core::Config>>,
    /// Decrement the global queue counter when the worker finishes this job.
    pub queue: crate::state::QueueHandle,
}

/// Pool of GPU workers with placement strategy.
pub struct GpuPool {
    pub workers: Vec<Arc<GpuWorker>>,
}

impl GpuWorker {
    /// Check if this worker is in a degraded state (3+ consecutive failures, within cooldown).
    pub fn is_degraded(&self) -> bool {
        if self.consecutive_failures.load(Ordering::SeqCst) < 3 {
            return false;
        }
        match *self.degraded_until.read().unwrap() {
            Some(until) => Instant::now() < until,
            None => false,
        }
    }

    /// Build a status snapshot for this worker.
    pub fn status(&self) -> GpuWorkerStatus {
        let active_gen = self.active_generation.read().unwrap();
        // Prefer the active-generation model name — during inflight generation
        // the cache entry is taken out of the cache (take-and-restore pattern),
        // so `cache.active_model()` returns None. Falling back to the cache
        // afterwards handles the idle-but-loaded case.
        let loaded_model = active_gen
            .as_ref()
            .map(|g| g.model.clone())
            .or_else(|| {
                let cache = self.model_cache.lock().unwrap();
                cache.active_model().map(|s| s.to_string())
            });

        let state = if self.is_degraded() {
            GpuWorkerState::Degraded
        } else if active_gen.is_some() {
            GpuWorkerState::Generating
        } else {
            GpuWorkerState::Idle
        };

        GpuWorkerStatus {
            ordinal: self.gpu.ordinal,
            name: self.gpu.name.clone(),
            vram_total_bytes: self.gpu.total_vram_bytes,
            vram_used_bytes: mold_inference::device::vram_used_estimate(self.gpu.ordinal),
            loaded_model,
            state,
        }
    }
}

impl GpuPool {
    /// Find a non-degraded worker that already has this model loaded on GPU.
    /// If multiple workers have it, prefer the one with fewer in-flight requests.
    pub fn find_loaded(&self, model_name: &str) -> Option<Arc<GpuWorker>> {
        let mut candidates: Vec<_> = self
            .workers
            .iter()
            .filter(|w| {
                if w.is_degraded() {
                    return false;
                }
                let cache = w.model_cache.lock().unwrap();
                cache
                    .get(model_name)
                    .map(|e| e.residency == ModelResidency::Gpu)
                    .unwrap_or(false)
            })
            .collect();

        candidates.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
        candidates.into_iter().next().cloned()
    }

    /// Select the best worker for a model, using the placement strategy
    /// (checked in order):
    /// 1. Loaded and idle (model on GPU, no in-flight requests).
    /// 2. Idle GPU with no model (spreads hot models across free GPUs).
    /// 3. Loaded but busy — whichever loaded copy has the fewest in-flight.
    /// 4. Non-degraded worker with the most headroom (will evict LRU).
    pub fn select_worker(&self, model_name: &str, estimated_vram: u64) -> Option<Arc<GpuWorker>> {
        self.select_worker_excluding(model_name, estimated_vram, &[])
    }

    /// Same as [`select_worker`], but skips workers whose ordinal is in `skip`.
    /// Used by the dispatcher to retry after a `try_send` failure.
    pub fn select_worker_excluding(
        &self,
        model_name: &str,
        estimated_vram: u64,
        skip: &[usize],
    ) -> Option<Arc<GpuWorker>> {
        let eligible: Vec<&Arc<GpuWorker>> = self
            .workers
            .iter()
            .filter(|w| !w.is_degraded() && !skip.contains(&w.gpu.ordinal))
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Classify each eligible worker.
        let mut loaded_idle: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut loaded_busy: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut idle_empty: Vec<&Arc<GpuWorker>> = Vec::new();
        let mut other: Vec<&Arc<GpuWorker>> = Vec::new();

        for w in &eligible {
            let (has_model, has_any_loaded) = {
                let cache = w.model_cache.lock().unwrap();
                let has_model = cache
                    .get(model_name)
                    .map(|e| e.residency == ModelResidency::Gpu)
                    .unwrap_or(false);
                (has_model, cache.active_model().is_some())
            };
            let in_flight = w.in_flight.load(Ordering::SeqCst);

            if has_model && in_flight == 0 {
                loaded_idle.push(w);
            } else if has_model {
                loaded_busy.push(w);
            } else if !has_any_loaded {
                idle_empty.push(w);
            } else {
                other.push(w);
            }
        }

        // 1. Loaded and idle — least in-flight first (should all be 0).
        if !loaded_idle.is_empty() {
            loaded_idle.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
            return loaded_idle.first().map(|w| (*w).clone());
        }

        // 2. Idle GPU with no model — spread! Prefer smallest GPU that fits.
        if !idle_empty.is_empty() {
            idle_empty.sort_by_key(|w| w.gpu.total_vram_bytes);
            if let Some(w) = idle_empty
                .iter()
                .find(|w| w.gpu.total_vram_bytes >= estimated_vram)
            {
                return Some((*w).clone());
            }
            // No idle GPU fits — pick the largest idle GPU.
            return idle_empty.last().map(|w| (*w).clone());
        }

        // 3. Loaded but busy — least in-flight wins.
        if !loaded_busy.is_empty() {
            loaded_busy.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
            return loaded_busy.first().map(|w| (*w).clone());
        }

        // 4. All GPUs busy with other models — most headroom first (evict LRU there).
        let mut busy = other;
        busy.sort_by(|a, b| {
            let a_headroom = a.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            let b_headroom = b.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            b_headroom.cmp(&a_headroom)
        });
        busy.first().map(|w| (*w).clone())
    }

    /// Collect status from all workers.
    pub fn gpu_status(&self) -> Vec<GpuWorkerStatus> {
        self.workers.iter().map(|w| w.status()).collect()
    }

    /// Number of GPU workers in the pool.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}
