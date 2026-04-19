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
        let cache = self.model_cache.lock().unwrap();
        let loaded_model = cache.active_model().map(|s| s.to_string());
        let active_gen = self.active_generation.read().unwrap();

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
    /// Find a worker that already has this model loaded on GPU.
    /// If multiple workers have it, prefer the one with fewer in-flight requests.
    pub fn find_loaded(&self, model_name: &str) -> Option<Arc<GpuWorker>> {
        let mut candidates: Vec<_> = self
            .workers
            .iter()
            .filter(|w| {
                let cache = w.model_cache.lock().unwrap();
                cache
                    .get(model_name)
                    .map(|e| e.residency == ModelResidency::Gpu)
                    .unwrap_or(false)
            })
            .collect();

        // Prefer least in-flight if multiple have it loaded.
        candidates.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
        candidates.into_iter().next().cloned()
    }

    /// Select the best worker for a model, using the placement strategy:
    /// 1. Already loaded on a GPU
    /// 2. Idle GPU (no model loaded), smallest that fits
    /// 3. Busy GPU with most headroom (will evict LRU)
    pub fn select_worker(&self, model_name: &str, estimated_vram: u64) -> Option<Arc<GpuWorker>> {
        // 1. Already loaded on a GPU?
        if let Some(w) = self.find_loaded(model_name) {
            return Some(w);
        }

        // 2. Find idle (no GPU-resident model) workers, skip degraded.
        let mut idle: Vec<_> = self
            .workers
            .iter()
            .filter(|w| {
                if w.is_degraded() {
                    return false;
                }
                let cache = w.model_cache.lock().unwrap();
                cache.active_model().is_none()
            })
            .collect();

        if !idle.is_empty() {
            // VRAM-fit tiebreaker: smallest GPU that fits.
            idle.sort_by_key(|w| w.gpu.total_vram_bytes);
            if let Some(w) = idle
                .iter()
                .find(|w| w.gpu.total_vram_bytes >= estimated_vram)
            {
                return Some((*w).clone());
            }
            // No idle GPU fits — pick the largest idle GPU anyway (eviction will help).
            return idle.last().cloned().cloned();
        }

        // 3. All GPUs busy — evict LRU on the GPU with most headroom.
        let mut busy: Vec<_> = self.workers.iter().filter(|w| !w.is_degraded()).collect();
        busy.sort_by(|a, b| {
            let a_headroom = a.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            let b_headroom = b.gpu.total_vram_bytes.saturating_sub(estimated_vram);
            b_headroom.cmp(&a_headroom) // most headroom first
        });
        busy.into_iter().next().cloned()
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
