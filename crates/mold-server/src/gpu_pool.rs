use crate::model_cache::{ModelCache, ModelResidency};
use mold_core::types::{DevicePlacement, DeviceRef, GpuWorkerState, GpuWorkerStatus};
use mold_db::MetadataDb;
use mold_inference::device::DiscoveredGpu;
use mold_inference::shared_pool::SharedPool;
use std::collections::BTreeSet;
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
    pub prompt_sha256: String,
    pub started_at_unix_ms: u64,
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
    /// Metadata DB handle so the worker can record a row alongside the
    /// on-disk save. `Arc<Option<...>>` mirrors `AppState.metadata_db` —
    /// `None` when the DB failed to open or is disabled.
    pub metadata_db: Arc<Option<MetadataDb>>,
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
        let in_flight = self.in_flight.load(Ordering::SeqCst);
        // Prefer the active-generation model name — during inflight generation
        // the cache entry is taken out of the cache (take-and-restore pattern),
        // so `cache.active_model()` returns None. Falling back to the cache
        // afterwards handles the idle-but-loaded case.
        let loaded_model = active_gen.as_ref().map(|g| g.model.clone()).or_else(|| {
            let cache = self.model_cache.lock().unwrap();
            cache.active_model().map(|s| s.to_string())
        });

        let state = if self.is_degraded() {
            GpuWorkerState::Degraded
        } else if active_gen.is_some() || in_flight > 0 {
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
    /// Return the worker bound to `ordinal`, if present in this pool.
    pub fn worker_by_ordinal(&self, ordinal: usize) -> Option<Arc<GpuWorker>> {
        self.workers
            .iter()
            .find(|w| w.gpu.ordinal == ordinal)
            .cloned()
    }

    /// Validate a request/config placement against the active worker pool.
    ///
    /// In multi-GPU worker mode a request may explicitly pin components to at
    /// most one GPU ordinal. Cross-GPU component placement would bypass the
    /// worker-affinity model entirely, so reject it here instead of letting the
    /// engines silently allocate on a sibling GPU.
    pub fn resolve_explicit_placement_gpu(
        &self,
        placement: Option<&DevicePlacement>,
    ) -> Result<Option<usize>, String> {
        if self.workers.is_empty() {
            return Ok(None);
        }
        let Some(placement) = placement else {
            return Ok(None);
        };

        let ordinals = placement_gpu_ordinals(placement);
        if ordinals.is_empty() {
            return Ok(None);
        }
        if ordinals.len() > 1 {
            let rendered = ordinals
                .iter()
                .map(|o| format!("gpu:{o}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "multi-GPU worker mode only supports placement on one GPU ordinal per request; got {rendered}"
            ));
        }

        let ordinal = *ordinals.iter().next().expect("checked non-empty");
        if self.worker_by_ordinal(ordinal).is_none() {
            let available = self
                .workers
                .iter()
                .map(|w| w.gpu.ordinal.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "gpu:{ordinal} is not available in this server's worker pool [{available}]"
            ));
        }
        Ok(Some(ordinal))
    }

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
                let active_gen = w.active_generation.read().unwrap();
                if active_gen.as_ref().is_some_and(|g| g.model == model_name) {
                    return true;
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
    /// 2. Loaded but busy — queue behind the warm copy instead of reloading.
    /// 3. Idle GPU with no model (spreads cold loads across free GPUs).
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
            let active_gen = w.active_generation.read().unwrap();
            let active_model = active_gen.as_ref().map(|g| g.model.as_str());
            let (has_model, has_any_loaded) = {
                let cache = w.model_cache.lock().unwrap();
                let has_model = active_model == Some(model_name)
                    || cache
                        .get(model_name)
                        .map(|e| e.residency == ModelResidency::Gpu)
                        .unwrap_or(false);
                (
                    has_model,
                    active_model.is_some() || cache.active_model().is_some(),
                )
            };
            let in_flight = w.in_flight.load(Ordering::SeqCst);
            // During an in-flight generation the worker thread calls
            // `cache.take()`, which removes the entry entirely — so
            // `cache.active_model()` and `cache.get(model).residency == Gpu`
            // both return None/false for the duration of that generation.
            // That used to let a busy GPU mid-inference look identical to
            // a truly empty idle GPU, which meant a new job for a *different*
            // model could be dispatched to the busy card while a sibling GPU
            // sat idle. `in_flight > 0` (set by the dispatcher before send)
            // and `active_generation.is_some()` (set by the worker around
            // the take-and-restore window) together cover every moment
            // between "about to pick up a job" and "just finished".
            let is_busy = in_flight > 0 || active_model.is_some();

            if has_model && !is_busy {
                loaded_idle.push(w);
            } else if has_model {
                loaded_busy.push(w);
            } else if !has_any_loaded && !is_busy {
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

        // 2. Loaded but busy — least in-flight wins.
        if !loaded_busy.is_empty() {
            loaded_busy.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
            return loaded_busy.first().map(|w| (*w).clone());
        }

        // 3. Idle GPU with no model — spread! Prefer smallest GPU that fits.
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

fn placement_gpu_ordinals(placement: &DevicePlacement) -> BTreeSet<usize> {
    let mut ordinals = BTreeSet::new();
    collect_gpu_ordinal(placement.text_encoders, &mut ordinals);
    if let Some(adv) = placement.advanced.as_ref() {
        collect_gpu_ordinal(adv.transformer, &mut ordinals);
        collect_gpu_ordinal(adv.vae, &mut ordinals);
        if let Some(device) = adv.clip_l {
            collect_gpu_ordinal(device, &mut ordinals);
        }
        if let Some(device) = adv.clip_g {
            collect_gpu_ordinal(device, &mut ordinals);
        }
        if let Some(device) = adv.t5 {
            collect_gpu_ordinal(device, &mut ordinals);
        }
        if let Some(device) = adv.qwen {
            collect_gpu_ordinal(device, &mut ordinals);
        }
    }
    ordinals
}

fn collect_gpu_ordinal(device: DeviceRef, out: &mut BTreeSet<usize>) {
    if let DeviceRef::Gpu { ordinal } = device {
        out.insert(ordinal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_cache::ModelCache;
    use mold_core::types::AdvancedPlacement;
    use mold_inference::shared_pool::SharedPool;

    /// Build a test GpuWorker with a scratch job channel and everything else
    /// in neutral defaults. Returns the worker plus the receiver so the test
    /// can verify what was dispatched.
    fn test_worker(
        ordinal: usize,
        total_vram_bytes: u64,
    ) -> (Arc<GpuWorker>, std::sync::mpsc::Receiver<GpuJob>) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(2);
        let worker = Arc::new(GpuWorker {
            gpu: DiscoveredGpu {
                ordinal,
                name: format!("test-gpu-{ordinal}"),
                total_vram_bytes,
                free_vram_bytes: total_vram_bytes,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    /// When GPU 0 is actively generating a different model, the cache
    /// take-and-restore pattern has already removed its entry — so
    /// `cache.active_model()` returns None and the worker LOOKS idle
    /// to the old classifier. The dispatcher must fall back to
    /// `in_flight > 0` (or `active_generation`) to avoid routing a
    /// brand-new job to the busy GPU while a sibling sits idle.
    #[test]
    fn select_worker_prefers_truly_idle_gpu_over_busy_gpu_with_empty_cache() {
        let (busy, _busy_rx) = test_worker(0, 24_000_000_000);
        let (idle, _idle_rx) = test_worker(1, 24_000_000_000);

        // Simulate the dispatcher having incremented in_flight before send,
        // and the worker thread having called cache.take() → empty cache.
        busy.in_flight.store(1, Ordering::SeqCst);

        let pool = GpuPool {
            workers: vec![busy.clone(), idle.clone()],
        };

        let picked = pool
            .select_worker("some-small-model:q4", 6_000_000_000)
            .expect("a worker should be selected");
        assert_eq!(
            picked.gpu.ordinal, 1,
            "new job for an unloaded model must go to the truly idle GPU, \
             not to the one whose cache momentarily looks empty because \
             generation is in progress"
        );
    }

    /// active_generation is set before take() and cleared after restore(),
    /// so a worker mid-inference should be treated as busy even if the
    /// dispatcher hasn't yet bumped in_flight (belt-and-suspenders).
    #[test]
    fn select_worker_respects_active_generation_flag() {
        let (busy, _busy_rx) = test_worker(0, 24_000_000_000);
        let (idle, _idle_rx) = test_worker(1, 24_000_000_000);

        *busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "big-model".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![busy.clone(), idle.clone()],
        };

        let picked = pool.select_worker("small-model:q4", 6_000_000_000).unwrap();
        assert_eq!(picked.gpu.ordinal, 1);
    }

    /// Regression guard for the happy path — both GPUs are idle and empty.
    /// The strategy says "prefer the smallest GPU that fits" to spread
    /// hot models across free cards.
    #[test]
    fn select_worker_spreads_to_smallest_fitting_idle_gpu() {
        let (big, _big_rx) = test_worker(0, 24_000_000_000);
        let (small, _small_rx) = test_worker(1, 12_000_000_000);

        let pool = GpuPool {
            workers: vec![big.clone(), small.clone()],
        };

        // A 6GB model fits on both — should pick the smaller card.
        let picked = pool.select_worker("flux-dev:q4", 6_000_000_000).unwrap();
        assert_eq!(picked.gpu.ordinal, 1);
    }

    /// If both eligible GPUs are busy with *other* models, fall back to
    /// the "most headroom" tier instead of deadlocking.
    #[test]
    fn select_worker_falls_back_when_all_gpus_busy_with_other_models() {
        let (a, _a_rx) = test_worker(0, 24_000_000_000);
        let (b, _b_rx) = test_worker(1, 12_000_000_000);
        a.in_flight.store(1, Ordering::SeqCst);
        b.in_flight.store(1, Ordering::SeqCst);

        let pool = GpuPool {
            workers: vec![a.clone(), b.clone()],
        };

        let picked = pool.select_worker("new-model", 6_000_000_000).unwrap();
        // Both busy → "most headroom" — the larger GPU wins.
        assert_eq!(picked.gpu.ordinal, 0);
    }

    #[test]
    fn select_worker_keeps_queueing_behind_busy_warm_worker() {
        let (warm_busy, _warm_busy_rx) = test_worker(0, 24_000_000_000);
        let (cold_idle, _cold_idle_rx) = test_worker(1, 24_000_000_000);

        warm_busy.in_flight.store(1, Ordering::SeqCst);
        *warm_busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "flux-dev:q4".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![warm_busy.clone(), cold_idle.clone()],
        };

        let picked = pool
            .select_worker("flux-dev:q4", 6_000_000_000)
            .expect("warm worker should be preferred");
        assert_eq!(picked.gpu.ordinal, 0);
    }

    #[test]
    fn resolve_explicit_placement_gpu_accepts_single_worker_ordinal() {
        let (worker, _rx) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker],
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                ..AdvancedPlacement::default()
            }),
        };

        assert_eq!(
            pool.resolve_explicit_placement_gpu(Some(&placement))
                .unwrap(),
            Some(1)
        );
    }

    #[test]
    fn resolve_explicit_placement_gpu_rejects_cross_gpu_requests() {
        let (worker0, _rx0) = test_worker(0, 24_000_000_000);
        let (worker1, _rx1) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker0, worker1],
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::gpu(0),
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(1),
                ..AdvancedPlacement::default()
            }),
        };

        let err = pool
            .resolve_explicit_placement_gpu(Some(&placement))
            .unwrap_err();
        assert!(err.contains("one GPU ordinal per request"), "{err}");
    }

    #[test]
    fn resolve_explicit_placement_gpu_rejects_ordinals_outside_pool() {
        let (worker1, _rx1) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![worker1],
        };
        let placement = DevicePlacement {
            text_encoders: DeviceRef::Auto,
            advanced: Some(AdvancedPlacement {
                transformer: DeviceRef::gpu(0),
                ..AdvancedPlacement::default()
            }),
        };

        let err = pool
            .resolve_explicit_placement_gpu(Some(&placement))
            .unwrap_err();
        assert!(err.contains("gpu:0"), "{err}");
        assert!(err.contains("[1]"), "{err}");
    }
}
