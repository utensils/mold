use crate::model_cache::{ModelCache, ModelResidency};
use mold_core::types::{DevicePlacement, DeviceRef, GpuWorkerState, GpuWorkerStatus};
use mold_db::MetadataDb;
use mold_inference::device::DiscoveredGpu;
use mold_inference::shared_pool::SharedPool;
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

const MODEL_CUDA_OOM_COOLDOWN: Duration = Duration::from_secs(60);

#[derive(Debug, Default)]
struct ModelCudaOomState {
    failed_ordinals: BTreeSet<usize>,
    unschedulable_until: Option<Instant>,
}

static MODEL_CUDA_OOMS: LazyLock<RwLock<HashMap<String, ModelCudaOomState>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone)]
pub(crate) struct ModelCudaOomOutcome {
    unschedulable_until: Option<Instant>,
}

impl ModelCudaOomOutcome {
    pub(crate) fn is_unschedulable(&self) -> bool {
        self.unschedulable_until
            .is_some_and(|until| Instant::now() < until)
    }
}

pub(crate) fn record_model_cuda_oom(model_name: &str, ordinal: usize) -> ModelCudaOomOutcome {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let state = states.entry(model_name.to_string()).or_default();

    if let Some(until) = state.unschedulable_until {
        if now < until {
            return ModelCudaOomOutcome {
                unschedulable_until: Some(until),
            };
        }
        state.unschedulable_until = None;
        state.failed_ordinals.clear();
    }

    state.failed_ordinals.insert(ordinal);
    let unschedulable_until = if state.failed_ordinals.len() >= 2 {
        let until = now + MODEL_CUDA_OOM_COOLDOWN;
        state.unschedulable_until = Some(until);
        tracing::warn!(
            model = %model_name,
            failed_gpus = ?state.failed_ordinals,
            cooldown_secs = MODEL_CUDA_OOM_COOLDOWN.as_secs(),
            "model marked temporarily unschedulable after CUDA OOM on multiple GPUs"
        );
        Some(until)
    } else {
        None
    };

    ModelCudaOomOutcome {
        unschedulable_until,
    }
}

pub(crate) fn model_unschedulable_message(model_name: &str) -> Option<String> {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let state = states.get_mut(model_name)?;
    let until = state.unschedulable_until?;
    if now >= until {
        states.remove(model_name);
        return None;
    }
    let remaining = until.saturating_duration_since(now).as_secs().max(1);
    Some(format!(
        "model '{model_name}' is temporarily unschedulable after CUDA OOM on multiple GPUs; \
         retry in {remaining}s or use a quantized/smaller variant."
    ))
}

pub(crate) fn failed_ordinals_for_model(model_name: &str) -> Vec<usize> {
    let now = Instant::now();
    let mut states = MODEL_CUDA_OOMS.write().unwrap();
    let Some(state) = states.get_mut(model_name) else {
        return Vec::new();
    };
    if let Some(until) = state.unschedulable_until {
        if now >= until {
            states.remove(model_name);
        }
        return Vec::new();
    }
    state.failed_ordinals.iter().copied().collect()
}

pub(crate) fn clear_model_cuda_oom(model_name: &str) {
    MODEL_CUDA_OOMS.write().unwrap().remove(model_name);
}

#[cfg(test)]
pub(crate) fn clear_model_cuda_ooms_for_tests() {
    MODEL_CUDA_OOMS.write().unwrap().clear();
}

/// Per-GPU worker state. Each GPU gets its own model cache, load lock, and health tracking.
pub struct GpuWorker {
    pub gpu: DiscoveredGpu,
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// Driver-free snapshot of the model currently resident on this worker.
    ///
    /// Status endpoints must never lock `model_cache`: an engine drop/load can
    /// block there while owning CUDA resources. Model-loading code updates this
    /// snapshot only after a successful residency transition.
    pub resident_model: Arc<RwLock<Option<String>>>,
    pub active_generation: Arc<RwLock<Option<ActiveGeneration>>>,
    pub model_load_lock: Arc<Mutex<()>>,
    pub shared_pool: Arc<Mutex<SharedPool>>,
    pub in_flight: AtomicUsize,
    pub consecutive_failures: AtomicUsize,
    /// Fatal CUDA errors poison a process-owned context permanently. A poisoned
    /// worker is quarantined until the server process is restarted.
    pub poisoned: AtomicBool,
    /// Shared process-level signal. A fatal context requires process teardown;
    /// supervised servers restart after `run_server` returns an error.
    pub fatal_cuda_error: Arc<AtomicBool>,
    pub fatal_cuda_shutdown: Arc<tokio::sync::Notify>,
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
    /// Server-assigned UUIDv4 carried over from `GenerationJob.id`. Used by
    /// the worker to flip the registry entry from `Queued` → `Running` (and
    /// to remove it when the job finishes), and surfaced to clients via
    /// `GET /api/queue` for zombie-card reconciliation.
    pub id: String,
    pub model: String,
    pub request: mold_core::GenerateRequest,
    pub completion_payload: crate::state::SseCompletionPayload,
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
    /// Job registry handle so the worker can flip state to Running on pickup
    /// and remove the entry on completion / error. Cheap clone — registry is
    /// behind an `Arc<RwLock>` internally.
    pub registry: crate::job_registry::SharedJobRegistry,
    /// Server-wide event broadcast so the worker's save path can emit
    /// `gallery_added` alongside the DB upsert (mirrors `AppState.events`).
    pub events: Arc<crate::events::EventBroadcaster>,
    /// Version fence proving this job was granted by the authoritative
    /// scheduler after the worker published the matching Ready generation.
    /// `None` exists only for legacy unit tests and the single-GPU adapter.
    pub lease: Option<crate::scheduler::LeaseFence>,
}

/// Pool of GPU workers with placement strategy.
pub struct GpuPool {
    pub workers: Vec<Arc<GpuWorker>>,
}

impl GpuWorker {
    pub(crate) fn set_resident_model(&self, model: Option<&str>) {
        *self
            .resident_model
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = model.map(str::to_string);
    }

    /// Check if this worker is in a degraded state (3+ consecutive failures, within cooldown).
    ///
    /// When the cooldown has expired we clear the failure counter and the
    /// `degraded_until` timestamp so the next single failure doesn't
    /// immediately re-degrade the GPU. Without this lazy reset, a worker
    /// that has 3 historical failures followed by a long idle period would
    /// flip back to Degraded on the very first post-cooldown failure
    /// (because `consecutive_failures` is still >= 3 from before).
    pub fn is_degraded(&self) -> bool {
        if self.poisoned.load(Ordering::SeqCst) {
            return true;
        }
        if self.consecutive_failures.load(Ordering::SeqCst) < 3 {
            return false;
        }
        let cooldown_active = match *self.degraded_until.read().unwrap() {
            Some(until) => Instant::now() < until,
            None => false,
        };
        if !cooldown_active {
            // Lazy clear: cooldown elapsed, treat the worker as healthy
            // again. A new failure burst still has to reach 3 consecutive
            // failures before re-degrading.
            self.consecutive_failures.store(0, Ordering::SeqCst);
            *self.degraded_until.write().unwrap() = None;
        }
        cooldown_active
    }

    /// Build a status snapshot for this worker.
    pub fn status(&self) -> GpuWorkerStatus {
        let active_gen = self.active_generation.read().unwrap();
        let in_flight = self.in_flight.load(Ordering::SeqCst);
        let loaded_model = active_gen
            .as_ref()
            .map(|g| g.model.clone())
            .or_else(|| self.resident_model.read().unwrap().clone());

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
            // The registry overlays cached sampler telemetry. Never query the
            // CUDA driver while serving a status request.
            vram_used_bytes: 0,
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

        // 2. Idle GPU with no model that FITS the estimate — spread beats
        //    queueing behind a busy warm card: the cold load is paid once,
        //    then that GPU is warm and tier 1 takes over for later jobs.
        //    (Warm-busy used to outrank this, which serialized same-model
        //    jobs on one card of a multi-GPU box while siblings sat idle.)
        idle_empty.sort_by_key(|w| w.gpu.total_vram_bytes);
        if let Some(w) = idle_empty
            .iter()
            .find(|w| w.gpu.total_vram_bytes >= estimated_vram)
        {
            return Some((*w).clone());
        }

        // 3. Loaded but busy — least in-flight wins. Reached only when no
        //    idle GPU can hold the model: waiting for the warm card beats
        //    cold-loading onto a card that would have to offload.
        if !loaded_busy.is_empty() {
            loaded_busy.sort_by_key(|w| w.in_flight.load(Ordering::SeqCst));
            return loaded_busy.first().map(|w| (*w).clone());
        }

        // 4. Idle GPU that doesn't fit — largest wins; block offloading may
        //    still make the model viable there.
        if !idle_empty.is_empty() {
            return idle_empty.last().map(|w| (*w).clone());
        }

        // 5. Everything left — busy with other models, or idle with a
        //    different model loaded (evicting a warm sibling has its own
        //    cost). Most headroom first; the LRU cache evicts there.
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

    static MODEL_CUDA_OOM_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
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
                stable_id: Some(format!("cuda:{ordinal:032x}")),
                raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: mold_core::types::GpuBackend::Cuda,
                name: format!("test-gpu-{ordinal}"),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes,
                free_vram_bytes: total_vram_bytes,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
            resident_model: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    #[test]
    fn worker_status_does_not_lock_the_model_cache_or_query_the_driver() {
        let (worker, _job_rx) = test_worker(0, 24_000_000_000);
        *worker.resident_model.write().unwrap() = Some("flux-dev:q4".to_string());

        let cache = worker.model_cache.clone();
        let _ = std::thread::spawn(move || {
            let _guard = cache.lock().unwrap();
            panic!("poison cache to prove status never acquires it");
        })
        .join();

        let status = worker.status();
        assert_eq!(status.loaded_model.as_deref(), Some("flux-dev:q4"));
        assert_eq!(status.vram_used_bytes, 0);
    }

    #[test]
    fn poisoned_worker_stays_degraded_after_cooldown_expiry() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.poisoned.store(true, Ordering::SeqCst);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() = Some(Instant::now() - Duration::from_secs(1));

        assert!(worker.is_degraded());
        assert_eq!(worker.consecutive_failures.load(Ordering::SeqCst), 3);
        assert!(worker.degraded_until.read().unwrap().is_some());
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

    /// A second job for the model that's RUNNING on GPU 0 must spread to an
    /// idle empty sibling instead of queueing behind the warm-but-busy card.
    /// The cold load is paid once; afterwards that sibling is warm too and
    /// tier 1 takes over. (Previously warm-busy beat idle-empty, so a 4-GPU
    /// box serialized same-model jobs on one card while three sat idle.)
    #[test]
    fn select_worker_spreads_same_model_to_idle_gpu_over_busy_warm_worker() {
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
            .expect("idle sibling should be preferred");
        assert_eq!(
            picked.gpu.ordinal, 1,
            "same-model job must run in parallel on the idle GPU, not queue \
             behind the busy warm one"
        );
    }

    /// The spread only happens when the idle GPU can actually hold the
    /// model: if no idle card fits, queueing behind the warm busy worker
    /// beats cold-loading onto a card that would have to offload.
    #[test]
    fn select_worker_queues_behind_warm_worker_when_no_idle_gpu_fits() {
        let (warm_busy, _warm_busy_rx) = test_worker(0, 48_000_000_000);
        let (small_idle, _small_idle_rx) = test_worker(1, 12_000_000_000);

        warm_busy.in_flight.store(1, Ordering::SeqCst);
        *warm_busy.active_generation.write().unwrap() = Some(ActiveGeneration {
            model: "ltx-2.3-22b-dev:fp8".to_string(),
            prompt_sha256: String::new(),
            started_at_unix_ms: 0,
            started_at: Instant::now(),
        });

        let pool = GpuPool {
            workers: vec![warm_busy.clone(), small_idle.clone()],
        };

        let picked = pool
            .select_worker("ltx-2.3-22b-dev:fp8", 34_000_000_000)
            .expect("a worker should be selected");
        assert_eq!(
            picked.gpu.ordinal, 0,
            "a too-small idle GPU must not win over the warm busy worker"
        );
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

    /// `is_degraded()` lazily resets the failure counter when the cooldown
    /// has expired. Without this, a worker that took 3 historical failures
    /// would re-degrade on the very first post-cooldown failure (because
    /// `consecutive_failures` was still ≥ 3 from before, even though the
    /// time-based gate had already opened back up).
    #[test]
    fn is_degraded_clears_counter_when_cooldown_has_expired() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        // Simulate "cooldown expired 1 second ago".
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() - std::time::Duration::from_secs(1));

        assert!(
            !worker.is_degraded(),
            "expired cooldown must mark the worker as healthy again",
        );
        assert_eq!(
            worker.consecutive_failures.load(Ordering::SeqCst),
            0,
            "expired cooldown must lazy-reset the failure counter so a \
             single post-cooldown failure doesn't immediately re-degrade",
        );
        assert!(
            worker.degraded_until.read().unwrap().is_none(),
            "expired cooldown must clear the timestamp",
        );
    }

    #[test]
    fn is_degraded_respects_active_cooldown() {
        let (worker, _rx) = test_worker(0, 24_000_000_000);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        // Cooldown still active for another 60s.
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() + std::time::Duration::from_secs(60));

        assert!(
            worker.is_degraded(),
            "active cooldown must keep the worker degraded",
        );
        assert_eq!(
            worker.consecutive_failures.load(Ordering::SeqCst),
            3,
            "active cooldown must NOT reset the counter",
        );
    }

    #[test]
    fn model_oom_on_sibling_gpu_marks_model_unschedulable() {
        let _guard = MODEL_CUDA_OOM_TEST_LOCK.lock().unwrap();
        clear_model_cuda_ooms_for_tests();
        let model = "flux2-klein-9b:bf16";

        let first = record_model_cuda_oom(model, 0);
        assert!(
            !first.is_unschedulable(),
            "first OOM only records the failed ordinal"
        );
        assert!(
            model_unschedulable_message(model).is_none(),
            "a single-GPU OOM should not cool down the model yet"
        );

        let second = record_model_cuda_oom(model, 1);
        assert!(
            second.is_unschedulable(),
            "OOM on a sibling GPU should mark the model unschedulable"
        );
        let msg = model_unschedulable_message(model).expect("cooldown message");
        assert!(msg.contains(model), "{msg}");
        assert!(msg.contains("temporarily unschedulable"), "{msg}");

        clear_model_cuda_ooms_for_tests();
    }

    #[test]
    fn failed_model_ordinals_can_be_skipped_before_cooldown() {
        let _guard = MODEL_CUDA_OOM_TEST_LOCK.lock().unwrap();
        clear_model_cuda_ooms_for_tests();
        let (failed, _failed_rx) = test_worker(0, 24_000_000_000);
        let (untested, _untested_rx) = test_worker(1, 24_000_000_000);
        let pool = GpuPool {
            workers: vec![failed, untested.clone()],
        };
        let model = "flux2-klein-9b:bf16";

        record_model_cuda_oom(model, 0);
        let skip = failed_ordinals_for_model(model);
        let picked = pool
            .select_worker_excluding(model, 32_000_000_000, &skip)
            .expect("sibling GPU should be tried before cooldown");

        assert_eq!(picked.gpu.ordinal, untested.gpu.ordinal);
        clear_model_cuda_ooms_for_tests();
    }
}
