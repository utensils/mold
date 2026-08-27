//! Release mold's own reclaimable model cache before refusing a generation
//! for want of HOST memory (#1289).
//!
//! Host admission refuses on a headroom sample, and on a single-user box most
//! of what that sample is missing is usually held by mold itself: the LRU model
//! cache keeps recently used engines resident so the next print skips a cold
//! load. That is the right default until a request cannot be admitted because
//! of it — at which point the user is handed a number they cannot act on and
//! whose only remedy is a `DELETE /api/models/unload` call no client surfaces.
//!
//! So a host-headroom refusal asks one more question before it answers: does
//! mold hold engines whose release would close the gap? If it does, they are
//! evicted least-recently-used first, the sample is re-taken, and admission is
//! retried once. If it does not — or if what was released was not enough — the
//! refusal names what was given back, so the number a user finally reads is a
//! number that already includes every byte mold could return.
//!
//! Four rules keep this safe, and each is load-bearing:
//!
//! - **Host only.** A device shortfall is already handled by placement and
//!   `ModelCache`'s own evict-to-fit path; it must not gain a second one.
//! - **Never the requested model, never a leased engine.** A checked-out engine
//!   is absent from [`crate::model_cache::ModelCache::reclaimable`] by
//!   construction, and the requested model is filtered here — evicting it would
//!   turn a warm admission into a cold reload of the very weights being asked
//!   for.
//! - **Never a worker that cannot take the job.** Eviction runs as `Admin` owner
//!   work on the thread that owns the device context, which is exactly the
//!   thread a running render occupies. A busy worker would park this
//!   preparation behind a whole generation for memory that render is about to
//!   give back anyway, and a quarantined, shutting-down, or draining one would
//!   never be leased the hard-pinned job at all — the scheduler refuses
//!   releasing work on a poisoned device (its memory comes back with the
//!   process) and a draining device never reports Idle. Both would leave a
//!   request waiting on a oneshot nobody will send, so both are filtered out
//!   and the await is bounded on top of that for the race between the two.
//! - **Eviction completes before the re-sample.** The reservation discipline in
//!   `AGENTS.md` says an OS sample can never prove a not-yet-settled allocation
//!   is reflected; the mirror of that rule is that it cannot prove a
//!   not-yet-completed release is either. Each eviction is awaited, its host
//!   memory is handed back through the #1273 path, and only then is the sample
//!   re-taken.

use std::time::Instant;

use crate::model_cache::ReclaimableEntry;
use crate::state::AppState;
use std::collections::BTreeSet;

/// One cached engine this reclaim may release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReclaimTarget {
    /// The GPU whose owner thread must perform the teardown, or `None` for the
    /// shared cache, whose entries are parked and hold no device context.
    pub ordinal: Option<usize>,
    pub model: String,
    pub last_used: Instant,
}

/// What one reclaim attempt actually gave back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostReclaimOutcome {
    /// Observed increase in headroom across the whole attempt.
    pub released_bytes: u64,
    /// Models evicted, in the order they were released.
    pub evicted: Vec<String>,
    /// The sampler could not read the memory it was asked about, so this
    /// outcome carries no evidence: nothing was evicted on a missing reading
    /// and no refusal may be built on one.
    pub sample_failed: bool,
}

impl HostReclaimOutcome {
    pub(crate) fn released_anything(&self) -> bool {
        !self.evicted.is_empty()
    }

    /// `"released 9.8 GB by unloading 2 idle models"`, or `None` when nothing
    /// was reclaimable — an empty clause must never be pasted into a refusal.
    pub(crate) fn release_summary(&self) -> Option<String> {
        if self.evicted.is_empty() {
            return None;
        }
        Some(format!(
            "released {} by unloading {}",
            gb1(self.released_bytes),
            plural_models(self.evicted.len())
        ))
    }
}

/// The refusal a host-headroom shortfall carries once reclaim has run.
///
/// `required` and `available` are the post-eviction numbers on purpose: a user
/// who is told "12.40 GB available" after mold already handed back everything
/// it could is reading an actionable figure, where the pre-eviction one would
/// send them hunting for memory mold was still holding.
pub(crate) fn host_shortfall_message(
    outcome: &HostReclaimOutcome,
    required_bytes: u64,
    available_bytes: u64,
) -> String {
    let shortfall = required_bytes.saturating_sub(available_bytes);
    let tail = format!(
        "still {} short (requires {}, {} available)",
        gb1(shortfall),
        gb2(required_bytes),
        gb2(available_bytes)
    );
    match outcome.release_summary() {
        Some(summary) => format!("{summary}; {tail}"),
        None => tail,
    }
}

fn plural_models(count: usize) -> String {
    if count == 1 {
        "1 idle model".to_string()
    } else {
        format!("{count} idle models")
    }
}

fn gb1(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}

fn gb2(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
}

/// Order every candidate least-recently-used first and drop the ones this
/// request must not touch.
///
/// The order is global rather than per-cache: two workers each holding one
/// engine are one working set from host RAM's point of view, and releasing the
/// newer of them first would evict a model that is more likely to be wanted
/// again. Ties break on `(ordinal, model)` so the plan is deterministic.
pub(crate) fn plan_reclaim(
    mut candidates: Vec<ReclaimTarget>,
    requested_model: &str,
) -> Vec<ReclaimTarget> {
    candidates.retain(|candidate| candidate.model != requested_model);
    candidates.sort_by(|left, right| {
        left.last_used
            .cmp(&right.last_used)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
            .then_with(|| left.model.cmp(&right.model))
    });
    candidates
}

fn targets_from(entries: Vec<ReclaimableEntry>, ordinal: Option<usize>) -> Vec<ReclaimTarget> {
    entries
        .into_iter()
        .map(|entry| ReclaimTarget {
            ordinal,
            model: entry.model_name,
            last_used: entry.last_used,
        })
        .collect()
}

/// How long one eviction may be waited on before the reclaim gives up.
///
/// The wait is bounded because a request is on the other end of it. A real
/// teardown behind nothing is seconds — `cuMemFree` plus a safetensors unmap —
/// so this is generous by a wide margin; what it actually guards is the race
/// [`accepts_releasing_work`] cannot close, where a worker is healthy
/// when it is snapshotted and quarantined or draining by the time the scheduler
/// would lease its hard-pinned job. Expiry is not a failure to report: the
/// caller falls back to exactly today's refusal.
const EVICTION_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// The worker facts a reclaim decision turns on, read once off the atomics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WorkerReleaseState {
    pub quarantined: bool,
    pub shutting_down: bool,
    pub draining: bool,
    pub busy: bool,
}

/// Whether the scheduler would actually lease this worker a hard-pinned
/// releasing job, and whether its owner thread is free to run it.
///
/// Every arm is about a oneshot nobody would send. The scheduler refuses
/// releasing work on a quarantined device — a poisoned or fatal-CUDA context's
/// memory comes back with the process, never with an unload — and a draining or
/// shutting-down device never reports Idle, so a hard-pinned job aimed at
/// either is never dispatched and the caller waits forever. A busy worker would
/// take the job eventually, but only behind a whole generation, for memory that
/// render is about to give back anyway.
///
/// Degraded is deliberately NOT here: three consecutive failures degrade a
/// worker, a wedged model is exactly what causes those failures while still
/// holding the memory, and `DeviceSnapshot::accepts_releasing_work` admits it
/// for precisely that reason.
pub(crate) fn accepts_releasing_work(state: WorkerReleaseState) -> bool {
    !state.quarantined && !state.shutting_down && !state.draining && !state.busy
}

fn worker_release_state(worker: &crate::gpu_pool::GpuWorker) -> WorkerReleaseState {
    use std::sync::atomic::Ordering;

    WorkerReleaseState {
        quarantined: worker.poisoned.load(Ordering::SeqCst)
            || worker.fatal_cuda_error.load(Ordering::SeqCst),
        shutting_down: worker.shutdown_requested.load(Ordering::SeqCst),
        draining: worker.drain_state.load(Ordering::SeqCst) != crate::gpu_pool::DRAIN_RUNNING,
        busy: worker.pending_or_executing() > 0
            || worker
                .active_generation
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_some(),
    }
}

/// Stable IDs of owners whose current work can return host memory when it
/// settles.
///
/// These workers are deliberately not reclaim targets: an Admin unload would
/// only wait behind the render that already owns the bytes. H3 preparation
/// uses this snapshot to keep a short request queued until that transient
/// pressure is gone.
pub(crate) fn busy_worker_device_ids(state: &AppState) -> BTreeSet<String> {
    state
        .gpu_pool
        .workers
        .snapshot()
        .into_iter()
        .filter(|worker| worker_release_state(worker).busy)
        .map(|worker| crate::scheduler::worker_device_id(&worker))
        .collect()
}

/// Collect every engine this server could release right now.
/// Which memory a reclaim is for. A device reclaim releases only what sits
/// on that device — the shared cache, whose engine lives wherever the legacy
/// path put it, and that device's own worker cache — because evicting a
/// model from another GPU cannot change the reading the reclaim is chasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReclaimScope {
    Host,
    Device(usize),
}

async fn reclaim_candidates(
    state: &AppState,
    requested_model: &str,
    scope: ReclaimScope,
) -> Vec<ReclaimTarget> {
    let mut candidates = targets_from(state.model_cache.lock().await.reclaimable(), None);
    for worker in state.gpu_pool.workers.snapshot() {
        if let ReclaimScope::Device(ordinal) = scope {
            if worker.gpu.ordinal != ordinal {
                continue;
            }
        }
        if !accepts_releasing_work(worker_release_state(&worker)) {
            continue;
        }
        let entries = worker
            .model_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .reclaimable();
        candidates.extend(targets_from(entries, Some(worker.gpu.ordinal)));
    }
    plan_reclaim(candidates, requested_model)
}

/// Whether a real admission could make host headroom by releasing Mold's own
/// idle model cache.
///
/// Placement preview is a read-only authority boundary, so it may ask this
/// question but must never call [`reclaim_host_headroom`]. The answer carries
/// no byte estimate: only the completed teardown and a fresh OS sample can say
/// how many host pages an engine actually returned.
pub(crate) async fn has_reclaimable_cached_model(state: &AppState, requested_model: &str) -> bool {
    !reclaim_candidates(state, requested_model, ReclaimScope::Host)
        .await
        .is_empty()
}

/// Why one eviction did not happen.
///
/// The two arms differ in what they say about the NEXT target: a failure is
/// about this engine and the reclaim moves on, while an unanswered submission
/// says the scheduler is not dispatching this reclaim's work at all, so trying
/// another target would only wait again.
#[derive(Debug)]
enum EvictionError {
    Failed(String),
    Unanswered,
}

/// Evict one target and wait for its teardown to finish.
///
/// A worker's engine can only be destroyed on the thread that owns its CUDA
/// context, so it goes through the same `Admin` owner work
/// `DELETE /api/models/:model` uses.
///
/// The shared cache is the LEGACY (non-Scheduler-V2) path's cache and is
/// released in place, exactly as that same route already releases it. Its
/// engine may be GPU-resident rather than parked, and the legacy loader itself
/// drops such engines from async context (`model_manager::ensure_model_ready`),
/// so this is that path's own discipline rather than a new one — restricting
/// this to parked entries would instead skip the very engine holding the memory
/// on a legacy deployment. `model_load_lock` is what that path serializes on;
/// it is taken without blocking, and a held lock means a load is in flight, so
/// the entry is left alone rather than raced into the loader's `cache race`
/// error.
async fn evict_target(state: &AppState, target: &ReclaimTarget) -> Result<bool, EvictionError> {
    let Some(ordinal) = target.ordinal else {
        let Ok(_load_guard) = state.model_load_lock.try_lock() else {
            return Ok(false);
        };
        let removed = state.model_cache.lock().await.remove(&target.model);
        return Ok(removed.is_some());
    };
    let id = format!("host-reclaim-evict-{}", uuid::Uuid::new_v4());
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let work = crate::gpu_pool::OwnerWork::AdminModelUnload(Box::new(
        crate::gpu_pool::AdminModelUnloadJob {
            id: id.clone(),
            model: Some(target.model.clone()),
            evict_cached: true,
            result_tx,
        },
    ));
    state
        .scheduled_work
        .submit(
            crate::scheduler::ScheduledOwnerWork::new(id, target.model.clone(), 0, work)
                .with_hard_ordinal(Some(ordinal))
                .with_priority(mold_scheduler::PriorityClass::Admin),
        )
        .await
        .map_err(EvictionError::Failed)?;
    let evicted = tokio::time::timeout(EVICTION_WAIT, result_rx)
        .await
        .map_err(|_| EvictionError::Unanswered)?
        .map_err(|_| {
            EvictionError::Failed("host-reclaim owner worker dropped its result".to_string())
        })?
        .map_err(EvictionError::Failed)?;
    Ok(evicted.is_some())
}

/// Release cached engines until host headroom reaches `needed_headroom_bytes`,
/// or until nothing reclaimable is left.
///
/// `headroom` is supplied by the caller rather than computed here so this stays
/// a single authority over *what* to release and never becomes a second
/// authority over what "available" means — the family's own admission floor
/// answers that.
pub(crate) async fn reclaim_host_headroom(
    state: &AppState,
    requested_model: &str,
    needed_headroom_bytes: u64,
    headroom: &(dyn Fn() -> u64 + Sync),
) -> HostReclaimOutcome {
    reclaim_headroom(
        state,
        requested_model,
        needed_headroom_bytes,
        ReclaimScope::Host,
        &|| Some(headroom()),
    )
    .await
}

/// [`reclaim_host_headroom`] for one device's memory: only that device's
/// engines are released, and a sampler that cannot read the device ends the
/// attempt with `sample_failed` rather than reading as zero — an unreadable
/// device must never flush every cached model and then be refused on it.
pub(crate) async fn reclaim_device_headroom(
    state: &AppState,
    requested_model: &str,
    needed_headroom_bytes: u64,
    ordinal: usize,
    free: &(dyn Fn() -> Option<u64> + Sync),
) -> HostReclaimOutcome {
    reclaim_headroom(
        state,
        requested_model,
        needed_headroom_bytes,
        ReclaimScope::Device(ordinal),
        free,
    )
    .await
}

async fn reclaim_headroom(
    state: &AppState,
    requested_model: &str,
    needed_headroom_bytes: u64,
    scope: ReclaimScope,
    headroom: &(dyn Fn() -> Option<u64> + Sync),
) -> HostReclaimOutcome {
    let mut outcome = HostReclaimOutcome::default();
    let Some(before) = headroom() else {
        tracing::warn!(
            model = %requested_model,
            ?scope,
            "memory could not be sampled; releasing nothing on a missing reading"
        );
        outcome.sample_failed = true;
        return outcome;
    };
    if before >= needed_headroom_bytes {
        return outcome;
    }
    let targets = reclaim_candidates(state, requested_model, scope).await;
    if targets.is_empty() {
        tracing::info!(
            model = %requested_model,
            required_host_bytes = needed_headroom_bytes,
            available_host_bytes = before,
            "host headroom is short and the model cache holds nothing reclaimable"
        );
        return outcome;
    }
    tracing::info!(
        model = %requested_model,
        required_host_bytes = needed_headroom_bytes,
        available_host_bytes = before,
        reclaimable = targets.len(),
        "host headroom is short; releasing cached models before refusing"
    );
    for target in targets {
        match evict_target(state, &target).await {
            Ok(true) => {}
            Ok(false) => continue,
            Err(EvictionError::Failed(error)) => {
                tracing::warn!(
                    model = %target.model,
                    ordinal = ?target.ordinal,
                    %error,
                    "host reclaim could not evict a cached model"
                );
                continue;
            }
            Err(EvictionError::Unanswered) => {
                tracing::warn!(
                    model = %target.model,
                    ordinal = ?target.ordinal,
                    wait_secs = EVICTION_WAIT.as_secs(),
                    "host reclaim gave up waiting for an eviction; refusing with what it has"
                );
                break;
            }
        }
        // Parking an engine is not the same as handing its pages back: the
        // shared pool may still hold its component maps and glibc may still
        // hold its freed arenas. #1273's release path is what makes the next
        // sample tell the truth.
        crate::routes::release_host_memory_after_unload(state);
        outcome.evicted.push(target.model.clone());
        let Some(now) = headroom() else {
            tracing::warn!(
                model = %target.model,
                ?scope,
                "memory could not be re-sampled after an eviction; stopping the reclaim"
            );
            outcome.sample_failed = true;
            break;
        };
        outcome.released_bytes = now.saturating_sub(before);
        tracing::info!(
            model = %target.model,
            ordinal = ?target.ordinal,
            available_host_bytes = now,
            "released a cached model for host headroom"
        );
        if now >= needed_headroom_bytes {
            break;
        }
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_cache::ModelResidency;
    use std::time::Duration;

    fn entry(name: &str, age_secs: u64) -> ReclaimableEntry {
        ReclaimableEntry {
            model_name: name.to_string(),
            residency: ModelResidency::Parked,
            last_used: Instant::now() - Duration::from_secs(age_secs),
            vram_bytes: 0,
        }
    }

    #[test]
    fn a_reclaim_plan_is_least_recently_used_first_across_every_cache() {
        let mut candidates = targets_from(vec![entry("shared-old", 300)], None);
        candidates.extend(targets_from(
            vec![entry("gpu-newest", 10), entry("gpu-middle", 120)],
            Some(0),
        ));

        let planned = plan_reclaim(candidates, "minimax-h3-fl2va")
            .into_iter()
            .map(|target| (target.ordinal, target.model))
            .collect::<Vec<_>>();
        assert_eq!(
            planned,
            vec![
                (None, "shared-old".to_string()),
                (Some(0), "gpu-middle".to_string()),
                (Some(0), "gpu-newest".to_string()),
            ]
        );
    }

    /// Evicting the model being admitted would turn a warm admission into a
    /// cold reload of the very weights the request is waiting for.
    #[test]
    fn the_requested_model_is_never_a_reclaim_target() {
        let candidates = targets_from(
            vec![entry("minimax-h3-fl2va", 900), entry("flux-schnell", 30)],
            Some(0),
        );
        let planned = plan_reclaim(candidates, "minimax-h3-fl2va")
            .into_iter()
            .map(|target| target.model)
            .collect::<Vec<_>>();
        assert_eq!(planned, vec!["flux-schnell".to_string()]);
    }

    /// The refusal a user finally reads has to name what mold gave back, or
    /// "your machine cannot run this" is indistinguishable from "your machine
    /// could not run this while mold was holding 9.8 GB of it" (#1289).
    #[test]
    fn a_shortfall_refusal_names_the_bytes_released_and_the_models_unloaded() {
        let outcome = HostReclaimOutcome {
            sample_failed: false,
            released_bytes: 9_800_000_000,
            evicted: vec!["flux-schnell".into(), "sdxl-base".into()],
        };
        let message = host_shortfall_message(&outcome, 15_300_615_032, 12_400_000_000);
        assert_eq!(
            message,
            "released 9.8 GB by unloading 2 idle models; still 2.9 GB short \
             (requires 15.30 GB, 12.40 GB available)"
        );

        let single = HostReclaimOutcome {
            sample_failed: false,
            released_bytes: 1_000_000_000,
            evicted: vec!["flux-schnell".into()],
        };
        assert!(single
            .release_summary()
            .expect("one eviction still reports")
            .contains("1 idle model"));
    }

    /// Nothing reclaimable must not paste an empty clause into the refusal.
    #[test]
    fn a_shortfall_with_nothing_to_release_reads_as_a_plain_shortfall() {
        let outcome = HostReclaimOutcome::default();
        assert!(!outcome.released_anything());
        assert_eq!(outcome.release_summary(), None);
        assert_eq!(
            host_shortfall_message(&outcome, 15_300_615_032, 12_659_979_674),
            "still 2.6 GB short (requires 15.30 GB, 12.66 GB available)"
        );
    }

    struct StubEngine {
        name: String,
    }

    impl mold_inference::InferenceEngine for StubEngine {
        fn generate(
            &mut self,
            _req: &mold_core::GenerateRequest,
        ) -> anyhow::Result<mold_core::GenerateResponse> {
            unimplemented!("host reclaim never generates")
        }
        fn model_name(&self) -> &str {
            &self.name
        }
        fn is_loaded(&self) -> bool {
            false
        }
        fn load(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
        fn unload(&mut self) {}
    }

    fn stub(name: &str) -> Box<dyn mold_inference::InferenceEngine> {
        Box::new(StubEngine {
            name: name.to_string(),
        })
    }

    /// Host headroom read from a script, one entry per sample: index 0 is the
    /// pre-eviction sample and each later index is the sample taken after one
    /// completed eviction. A closure is how the caller keeps ownership of what
    /// "available" means, so the test owns it too.
    struct ScriptedHeadroom {
        samples: Vec<u64>,
        taken: std::sync::atomic::AtomicUsize,
    }

    impl ScriptedHeadroom {
        fn new(samples: &[u64]) -> Self {
            Self {
                samples: samples.to_vec(),
                taken: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn sample(&self) -> u64 {
            let index = self
                .taken
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                .min(self.samples.len() - 1);
            self.samples[index]
        }
    }

    /// The issue's exact repro (#1289): three flux-schnell stills leave FLUX
    /// resident, an H3 submit needs 15,300,615,032 host bytes, and the sample
    /// offers 12,659,979,674. Releasing the one idle engine closes the gap, so
    /// admission must be retried rather than refused — and once it fits, no
    /// further engine is touched.
    #[tokio::test]
    async fn releasing_one_idle_engine_closes_the_issues_exact_shortfall() {
        let state = AppState::for_tests();
        {
            let mut cache = state.model_cache.lock().await;
            cache.insert(stub("flux-schnell"), 0);
            cache.insert(stub("sdxl-base"), 0);
        }
        let headroom = ScriptedHeadroom::new(&[12_659_979_674, 22_459_979_674]);

        let outcome = reclaim_host_headroom(
            &state,
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
            15_300_615_032,
            &|| headroom.sample(),
        )
        .await;

        assert_eq!(outcome.evicted, vec!["flux-schnell".to_string()]);
        assert_eq!(outcome.released_bytes, 9_800_000_000);
        assert!(
            state.model_cache.lock().await.contains("sdxl-base"),
            "reclaim stops the moment the shortfall is covered"
        );
    }

    /// When even the whole cache cannot cover the gap, mold still gives the
    /// bytes back — nothing is running that could want them, the user asked for
    /// a print, and the refusal they read is then the honest post-eviction one.
    #[tokio::test]
    async fn an_uncoverable_shortfall_still_releases_everything_and_refuses_honestly() {
        let state = AppState::for_tests();
        {
            let mut cache = state.model_cache.lock().await;
            cache.insert(stub("flux-schnell"), 0);
            cache.insert(stub("sdxl-base"), 0);
        }
        let headroom = ScriptedHeadroom::new(&[8_000_000_000, 10_000_000_000, 12_400_000_000]);

        let outcome = reclaim_host_headroom(&state, "minimax-h3-fl2va", 15_300_615_032, &|| {
            headroom.sample()
        })
        .await;

        assert_eq!(
            outcome.evicted,
            vec!["flux-schnell".to_string(), "sdxl-base".to_string()]
        );
        assert_eq!(outcome.released_bytes, 4_400_000_000);
        assert!(state.model_cache.lock().await.is_empty());
        assert_eq!(
            host_shortfall_message(&outcome, 15_300_615_032, 12_400_000_000),
            "released 4.4 GB by unloading 2 idle models; still 2.9 GB short \
             (requires 15.30 GB, 12.40 GB available)"
        );
    }

    /// Headroom that already fits must not touch the cache at all — the whole
    /// point of the LRU cache is that the next print skips a cold load.
    #[tokio::test]
    async fn sufficient_headroom_evicts_nothing() {
        let state = AppState::for_tests();
        state
            .model_cache
            .lock()
            .await
            .insert(stub("flux-schnell"), 0);
        let headroom = ScriptedHeadroom::new(&[40_000_000_000]);

        let outcome = reclaim_host_headroom(&state, "minimax-h3-fl2va", 15_300_615_032, &|| {
            headroom.sample()
        })
        .await;

        assert!(!outcome.released_anything());
        assert!(state.model_cache.lock().await.contains("flux-schnell"));
    }

    /// A cache holding only the model being admitted has nothing to give, and
    /// the refusal must say so without inventing a release clause.
    #[tokio::test]
    async fn a_cache_holding_only_the_requested_model_releases_nothing() {
        let state = AppState::for_tests();
        state
            .model_cache
            .lock()
            .await
            .insert(stub("minimax-h3-fl2va"), 0);
        let headroom = ScriptedHeadroom::new(&[12_659_979_674]);

        let outcome = reclaim_host_headroom(&state, "minimax-h3-fl2va", 15_300_615_032, &|| {
            headroom.sample()
        })
        .await;

        assert!(outcome.evicted.is_empty());
        assert!(state.model_cache.lock().await.contains("minimax-h3-fl2va"));
        assert_eq!(
            host_shortfall_message(&outcome, 15_300_615_032, 12_659_979_674),
            "still 2.6 GB short (requires 15.30 GB, 12.66 GB available)"
        );
    }

    /// Every one of these is a oneshot nobody would send: the scheduler refuses
    /// releasing work on a quarantined device, and a draining or shutting-down
    /// one never reports Idle, so a hard-pinned admin unload aimed at either is
    /// never dispatched. A reclaim that queued one would hang the request it is
    /// trying to help.
    #[test]
    fn only_a_worker_that_can_be_leased_the_unload_is_a_reclaim_candidate() {
        assert!(accepts_releasing_work(WorkerReleaseState::default()));
        for state in [
            WorkerReleaseState {
                quarantined: true,
                ..Default::default()
            },
            WorkerReleaseState {
                shutting_down: true,
                ..Default::default()
            },
            WorkerReleaseState {
                draining: true,
                ..Default::default()
            },
            WorkerReleaseState {
                busy: true,
                ..Default::default()
            },
        ] {
            assert!(
                !accepts_releasing_work(state),
                "{state:?} cannot be leased a hard-pinned unload"
            );
        }
    }
}
