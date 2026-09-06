pub mod auth;
pub mod batch_transaction;
pub mod catalog_api;
pub mod catalog_credentials;
pub(crate) mod chain_execution;
pub mod chain_job_runner;
pub mod chain_limits;
mod chain_source_media;
mod cuda_peak;
pub(crate) mod dir_sync;
mod durable_admission_authority;
mod durable_disposition;
mod durable_generation_settlement;
mod gallery_authority;
mod gallery_source_media;
#[allow(dead_code)]
mod h3_admission;
mod h3_attempt;
#[cfg(any(
    test,
    feature = "h3",
    feature = "h3-private-bridge",
    feature = "h3-private-uat"
))]
mod h3_private_bridge;
mod hunyuan3d_admission;
pub mod local_h3;
pub mod test_support;
// Agent A (downloads)
pub mod device_registry;
pub mod dispatch_mode;
pub mod downloads;
mod durable_queue_feeder;
pub mod events;
pub mod execution_plan;
mod gallery_organization;
mod gallery_trash;
pub mod generation_cancel;
pub mod gpu_pool;
pub mod gpu_worker;
// MiniMax H3 is the only family whose HOST admission refuses on headroom today:
// every other family's host shortfall is a planner block that resolves when the
// work holding the RAM gives it back. The module itself is family-blind, so a
// build without H3 still compiles it and still runs its tests — it just has no
// caller yet.
#[cfg_attr(not(any(feature = "h3", feature = "h3-private-uat")), allow(dead_code))]
mod host_reclaim;
mod identity_dependencies;
/// Public so the forced-local CLI path can run the same extraction the
/// server's worker runs, at the same point in the lease. `mold-cli` builds its
/// own engine from an admitted plan and must not grow a second identity
/// lifetime.
pub mod identity_extraction;
pub mod instance;
pub mod job_registry;
pub mod job_supervisor;
pub mod logging;
mod ltx2_admission;
#[cfg(feature = "mdns")]
pub mod mdns;
mod memory_preflight;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model_cache;
pub mod model_manager;
mod paint_dependencies;
pub mod queue;
pub mod queue_journal;
pub mod queue_media;
mod queue_media_admission;
mod queue_media_ingress;
mod queue_media_lifecycle;
pub mod queue_media_runtime;
mod queue_retention;
// This dependency-free policy seam lands default-dark. The concrete
// schema/store adapter activates it atomically with queue-media admission.
#[allow(dead_code)]
mod queue_media_startup;
pub mod queue_media_store;
pub mod rate_limit;
pub mod reference_uploads;
pub mod request_id;
pub mod resources;
pub mod routes;
pub mod routes_activity;
pub mod routes_chain;
pub mod routes_chain_jobs;
pub mod routes_config;
pub mod scheduler;
mod signals;
pub mod state;
pub mod thumbnails;
pub mod variant_dependencies;
pub mod video_upscale;
mod wan_admission;
pub mod web_ui;
// The arcstats parser and credit policy are pure and unit-tested on every
// platform, but only the Linux reader calls them; the other targets would
// otherwise fail `-D warnings` on dead code (#1439).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod zfs_arc;

#[cfg(test)]
mod admission_contract_test;
#[cfg(all(test, feature = "metrics"))]
mod metrics_test;
#[cfg(test)]
mod resources_test;
#[cfg(test)]
mod routes_test;

use anyhow::Result;
use axum::{extract::DefaultBodyLimit, middleware};
use mold_core::types::GpuSelection;
use mold_core::{Config, ModelPaths};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

fn install_durable_admission_if_available(
    queue_journal: &std::sync::Arc<queue_journal::QueueJournal>,
    lifecycle: std::sync::Arc<queue_media_lifecycle::QueueMediaLifecycle>,
    queue_capacity: usize,
) -> bool {
    let admission = queue_journal
        .has_generation_v2_receipt_evidence()
        .map_err(anyhow::Error::msg)
        .and_then(|receipt_evidence_exists| {
            queue_media_admission::DurableMediaAdmission::new(
                lifecycle,
                queue_capacity,
                receipt_evidence_exists,
            )
            .map_err(anyhow::Error::new)
        });
    match admission {
        Ok(admission) => match queue_journal.install_queue_media_admission(admission) {
            Ok(()) => true,
            Err(error) => {
                tracing::error!(
                    error,
                    "canonical durable admission service was already installed"
                );
                false
            }
        },
        Err(error) => {
            tracing::error!(
                %error,
                "canonical durable admission disabled: receipt authority is unavailable"
            );
            false
        }
    }
}

use state::QueueHandle;

/// Retains Mold's compile-time H3 attention exclusion proof in server-backed
/// binaries, including the standalone Tauri desktop executable.
#[doc(hidden)]
pub fn h3_attention_release_provenance_marker() -> &'static str {
    mold_inference::h3_attention_release_provenance_marker()
}

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

fn trace_request_path<B>(request: &axum::http::Request<B>) -> &str {
    // Deliberately omit the query string: gallery media tickets are bearer
    // credentials and must never be copied into request spans or logs.
    request.uri().path()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupMode {
    GpuWorkers,
    CpuFallback,
    Maintenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StartupPlan {
    mode: StartupMode,
    start_gpu_workers: bool,
    create_cpu_engine: bool,
    start_generation_runner: bool,
    start_chain_runner: bool,
    start_legacy_cache_evictor: bool,
    start_legacy_dispatcher: bool,
    start_v2_coordinator: bool,
    observe_v2_decisions: bool,
}

/// Owns every dedicated GPU OS thread from the instant it is spawned.
///
/// `run_server` has fallible initialization after device discovery. Keeping
/// workers and join handles in one guard makes those early returns
/// transactional: dropping the guard fences every worker, wakes idle receivers,
/// and joins every thread before returning the startup error.
#[derive(Default)]
struct GpuOwnerThreads {
    workers: Vec<std::sync::Arc<gpu_pool::GpuWorker>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl GpuOwnerThreads {
    fn track(
        &mut self,
        worker: std::sync::Arc<gpu_pool::GpuWorker>,
        handle: std::thread::JoinHandle<()>,
    ) {
        self.workers.push(worker);
        self.handles.push(handle);
    }

    fn request_shutdown(&self) {
        for worker in &self.workers {
            worker.request_shutdown();
        }
    }

    fn join_all(&mut self) -> Result<()> {
        let mut failures = Vec::new();
        for handle in self.handles.drain(..) {
            let thread_name = handle
                .thread()
                .name()
                .unwrap_or("<unnamed GPU owner>")
                .to_string();
            if let Err(payload) = handle.join() {
                failures.push(format!(
                    "{thread_name}: {}",
                    panic_payload_message(payload.as_ref())
                ));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            anyhow::bail!("GPU owner thread join failed: {}", failures.join("; "))
        }
    }

    fn shutdown_and_join(mut self) -> Result<()> {
        self.request_shutdown();
        self.join_all()
    }
}

impl Drop for GpuOwnerThreads {
    fn drop(&mut self) {
        if self.handles.is_empty() {
            return;
        }
        self.request_shutdown();
        if let Err(error) = self.join_all() {
            // Drop is the startup-error fallback and cannot replace the
            // original initialization error. Do not silently discard a thread
            // panic: preserve it in the server log.
            tracing::error!(error = %format!("{error:#}"), "failed to join GPU owners during rollback");
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn classify_startup_mode(
    selection: &GpuSelection,
    discovered_count: usize,
    selected_count: usize,
    gpu_runtime_build: bool,
) -> StartupMode {
    // Explicit maintenance mode is authoritative. Fail closed even if a
    // future selection resolver regression returns devices for `none`.
    if matches!(selection, GpuSelection::None) {
        return StartupMode::Maintenance;
    }
    if selected_count > 0 {
        return StartupMode::GpuWorkers;
    }
    if discovered_count > 0 || gpu_runtime_build {
        StartupMode::Maintenance
    } else {
        StartupMode::CpuFallback
    }
}

fn startup_plan(
    selection: &GpuSelection,
    discovered_count: usize,
    selected_count: usize,
    gpu_runtime_build: bool,
    dispatch_mode: crate::dispatch_mode::DispatchMode,
) -> StartupPlan {
    let mode = classify_startup_mode(
        selection,
        discovered_count,
        selected_count,
        gpu_runtime_build,
    );
    StartupPlan {
        mode,
        start_gpu_workers: mode == StartupMode::GpuWorkers,
        create_cpu_engine: mode == StartupMode::CpuFallback,
        start_generation_runner: mode != StartupMode::Maintenance,
        start_chain_runner: mode != StartupMode::Maintenance,
        // GPU workers own their cache eviction on their CUDA owner threads.
        // Maintenance has no engine cache to sweep.
        start_legacy_cache_evictor: mode == StartupMode::CpuFallback,
        start_legacy_dispatcher: mode == StartupMode::GpuWorkers
            && !dispatch_mode.owns_v2_workers(),
        start_v2_coordinator: mode == StartupMode::GpuWorkers && dispatch_mode.owns_v2_workers(),
        observe_v2_decisions: mode == StartupMode::GpuWorkers
            && dispatch_mode.records_v2_observations(),
    }
}

pub async fn run_server(
    bind: &str,
    port: u16,
    models_dir: PathBuf,
    gpu_selection: GpuSelection,
    queue_size: usize,
) -> Result<()> {
    // Re-arm SIG_IGN for SIGPIPE. The CLI resets it to SIG_DFL in main() for
    // clean piping of short-lived commands, but for this long-running server
    // that is fatal — a single client dropping mid-write would kill the whole
    // process (issue #342). With SIG_IGN, such writes surface as EPIPE and are
    // handled per-request by hyper/axum.
    signals::ignore_sigpipe();

    let dispatch_mode = dispatch_mode::DispatchMode::from_env().map_err(anyhow::Error::msg)?;
    info!(mode = %dispatch_mode, "selected restart-time GPU dispatch mode");

    Config::install_runtime_models_dir_override(models_dir.clone());

    let mut config = Config::load_or_default();
    config.models_dir = models_dir.to_string_lossy().into_owned();
    let model_name = config.resolved_default_model();

    // ── Discover and initialize GPU workers ────────────────────────────────
    let shared_pool = std::sync::Arc::new(std::sync::Mutex::new(
        mold_inference::shared_pool::SharedPool::new(),
    ));
    // CUDA primary contexts are process-owned. Fatal driver faults signal the
    // HTTP server to stop and return an error so systemd/desktop recovery can
    // restart the process instead of retrying a poisoned context.
    let fatal_cuda_error = std::sync::Arc::new(AtomicBool::new(false));
    let fatal_cuda_shutdown = std::sync::Arc::new(tokio::sync::Notify::new());

    let discovered = mold_inference::device::discover_gpus();
    let selected = mold_inference::device::resolve_gpu_selection(&discovered, &gpu_selection)?;

    // Open persistence and project device preferences before creating any GPU
    // owner thread. A disabled device remains in the startup-selected
    // inventory (and therefore in V2's dynamic worker factory), but it must
    // not transiently own a CUDA context or receive legacy/observe work after
    // restart.
    let metadata_db = match mold_db::open_default() {
        Ok(Some(db)) => {
            info!(db = %db.path().display(), "metadata DB opened");
            std::sync::Arc::new(Some(db))
        }
        Ok(None) => {
            tracing::info!("metadata DB disabled (MOLD_DB_DISABLE set or MOLD_HOME unresolved)");
            std::sync::Arc::new(None)
        }
        Err(e) => {
            tracing::warn!(
                "failed to open metadata DB: {e:#} — gallery falls back to filesystem scan"
            );
            std::sync::Arc::new(None)
        }
    };
    // Resolved here rather than at its later assignment to `state` so the
    // queue journal can use it: the identity is `(data dir, port)`-scoped, which
    // is exactly the evidence a restarting server needs to recognise its own
    // retained queue among a peer's.
    let instance_id = instance::resolve_instance_id(metadata_db.as_ref().as_ref(), port);
    // Built before any GPU owner thread exists: a worker that quarantines
    // itself must be able to raise the retention fence, and that is the one
    // restart mold performs on its own behalf.
    let queue_journal = std::sync::Arc::new(queue_journal::QueueJournal::new(
        metadata_db.clone(),
        Config::mold_dir().as_deref(),
        &instance_id,
    ));
    if let Some(owner_uuid) = queue_journal.owner_uuid() {
        let mold_home = Config::mold_dir().ok_or_else(|| {
            anyhow::anyhow!(
                "durable queue claimed an owner but MOLD_HOME became unavailable before media reconciliation"
            )
        })?;
        let lifecycle = std::sync::Arc::new(queue_media_lifecycle::QueueMediaLifecycle::new(
            metadata_db.clone(),
            mold_home,
            owner_uuid.to_string(),
        ));
        queue_journal
            .install_queue_media_lifecycle(lifecycle.clone())
            .map_err(anyhow::Error::msg)?;
        let media_report =
            queue_media_startup::reconcile_claimed_owner(&queue_journal, lifecycle.as_ref())?;
        info!(
            durable_media_ready = media_report.durable_media_ready,
            restored = media_report.restored.len(),
            deleted = media_report.deleted.len(),
            cleared_gc_pending = media_report.cleared_gc_pending.len(),
            held_jobs = media_report.held_jobs.len(),
            issues = media_report.issues.len(),
            unclaimed_owner_roots = media_report.unclaimed_owner_roots.len(),
            "durable queue-media startup reconciliation complete"
        );
        // A count is not a diagnosis. Each reason is logged in full, once per
        // reason, so the operator gets the remedy without a source dive; the
        // same text is retained on `/api/status` for after this line ages out.
        for reason in media_report.degradation_reasons() {
            tracing::warn!(
                reason,
                "restart-safe queue media is unavailable; durable generations without \
                 request media are unaffected"
            );
        }
    }
    let generation_cancel = std::sync::Arc::new(generation_cancel::CancelRegistry::new());
    if queue_journal.is_enabled() {
        info!("durable generation queue enabled");
    }
    let device_registry =
        std::sync::Arc::new(device_registry::DeviceRegistry::from_runtime_inventory(
            discovered,
            &selected,
            metadata_db.clone(),
        ));
    for gpu in device_registry.persisted_disabled_worker_devices() {
        info!(
            gpu = gpu.ordinal,
            device_id = gpu.stable_id.as_deref().unwrap_or("unknown"),
            name = %gpu.name,
            reason = "persisted desired enablement is disabled",
            "GPU owner skipped at startup"
        );
    }
    let startup_devices = device_registry.startup_worker_devices();
    let discovered_count = device_registry.visible_device_count();
    let selected_count = device_registry.startup_allowed_count();

    let startup = startup_plan(
        &gpu_selection,
        discovered_count,
        selected_count,
        cfg!(any(feature = "cuda", feature = "metal")),
        dispatch_mode,
    );
    if startup.start_v2_coordinator {
        info!("scheduler V2 owns GPU dispatch and worker leases");
    } else if startup.observe_v2_decisions {
        info!(
            "legacy dispatcher owns GPU work; V2 decisions are observed without leases or transport"
        );
    } else if startup.start_legacy_dispatcher {
        info!("legacy dispatcher owns GPU work");
    }

    let mut workers = Vec::new();
    let mut gpu_owner_threads = GpuOwnerThreads::default();
    let mut v2_owner_handles = Vec::new();
    let (scheduler_worker_tx, scheduler_worker_rx) =
        tokio::sync::mpsc::unbounded_channel::<scheduler::WorkerEvent>();
    let (legacy_owner_event_tx, legacy_owner_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<gpu_worker::LegacyOwnerEvent>();

    // V2's capacity-one channel is a rendezvous transport after Ready.
    // Rollback mode retains the prior depth-two device-local buffer.
    let per_worker_channel_size = if startup.start_v2_coordinator { 1 } else { 2 };

    let max_cached = state::resolve_max_cached_models();
    let cache_idle_ttl = std::time::Duration::from_secs(state::resolve_cache_idle_ttl_secs());
    if startup.start_gpu_workers {
        for gpu in startup_devices {
            let (job_tx, job_rx) = std::sync::mpsc::sync_channel(per_worker_channel_size);
            let worker = std::sync::Arc::new(gpu_pool::GpuWorker {
                cuda_peak: Default::default(),
                #[cfg(test)]
                mock_device_memory: None,
                owner_epoch: 1,
                gpu: gpu.clone(),
                model_cache: std::sync::Arc::new(std::sync::Mutex::new(
                    model_cache::ModelCache::new(max_cached),
                )),
                resident_model: std::sync::Arc::new(std::sync::RwLock::new(None)),
                resident_execution_fingerprint: std::sync::Arc::new(std::sync::RwLock::new(None)),
                active_generation: std::sync::Arc::new(std::sync::RwLock::new(None)),
                model_load_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
                shared_pool: shared_pool.clone(),
                legacy_pending: AtomicUsize::new(0),
                in_flight: AtomicUsize::new(0),
                legacy_chain_waiters: Default::default(),
                consecutive_failures: AtomicUsize::new(0),
                poisoned: AtomicBool::new(false),
                fatal_cuda_error: fatal_cuda_error.clone(),
                fatal_cuda_shutdown: fatal_cuda_shutdown.clone(),
                queue_journal: queue_journal.clone(),
                generation_cancel: generation_cancel.clone(),
                shutdown_requested: AtomicBool::new(false),
                drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
                owner_thread_id: std::sync::OnceLock::new(),
                degraded_until: std::sync::RwLock::new(None),
                job_tx,
            });

            let handle = if startup.start_v2_coordinator {
                gpu_worker::spawn_gpu_thread(
                    worker.clone(),
                    job_rx,
                    scheduler_worker_tx.clone(),
                    cache_idle_ttl,
                )
            } else {
                gpu_worker::spawn_legacy_gpu_thread(
                    worker.clone(),
                    job_rx,
                    legacy_owner_event_tx.clone(),
                    cache_idle_ttl,
                )
            };
            if startup.start_v2_coordinator {
                v2_owner_handles.push((
                    scheduler::worker_device_id(&worker),
                    worker.owner_epoch,
                    handle,
                ));
            } else {
                gpu_owner_threads.track(worker.clone(), handle);
            }
            workers.push(worker);
        }
    }

    let gpu_pool = std::sync::Arc::new(gpu_pool::GpuPool {
        workers: workers.into(),
    });
    if startup.start_v2_coordinator {
        gpu_pool
            .workers
            .install_factory(
                gpu_pool::WorkerFactory {
                    registry: device_registry.clone(),
                    shared_pool: shared_pool.clone(),
                    fatal_cuda_error: fatal_cuda_error.clone(),
                    fatal_cuda_shutdown: fatal_cuda_shutdown.clone(),
                    queue_journal: queue_journal.clone(),
                    generation_cancel: generation_cancel.clone(),
                    scheduler_tx: scheduler_worker_tx.clone(),
                    owner_spawner: std::sync::Arc::new(gpu_pool::RuntimeOwnerThreadSpawner),
                    max_cached,
                    cache_idle_ttl,
                },
                v2_owner_handles,
            )
            .map_err(anyhow::Error::msg)?;
    }

    // Log discovered GPUs.
    for status in gpu_pool.gpu_status() {
        info!(
            gpu = status.ordinal,
            name = %status.name,
            vram_mb = status.vram_total_bytes / 1_000_000,
            "GPU worker ready"
        );
    }

    match startup.mode {
        StartupMode::GpuWorkers => {}
        StartupMode::CpuFallback => {
            info!("CPU-only build — server generation uses the CPU correctness path");
        }
        StartupMode::Maintenance if matches!(gpu_selection, GpuSelection::None) => {
            info!("GPU workers disabled by explicit selection; server is in maintenance mode");
        }
        StartupMode::Maintenance => {
            tracing::warn!(
                discovered = discovered_count,
                "no selected GPU worker is safe to start; generation is unavailable"
            );
        }
    }

    // Concise GPU summary for the mDNS TXT record (e.g. "2xNVIDIA GeForce RTX
    // 4090", or "cpu" when GPU-less). Computed here while `gpu_pool` is fresh.
    #[cfg(feature = "mdns")]
    let mdns_gpu_summary = {
        let names: Vec<String> = gpu_pool
            .gpu_status()
            .iter()
            .map(|s| s.name.clone())
            .collect();
        mdns::gpu_summary(&names)
    };

    // ── Create generation queue ────────────────────────────────────────────
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(queue_size.max(1));
    let queue_handle = QueueHandle::new(job_tx);
    let (scheduled_work_tx, scheduled_work_rx) = tokio::sync::mpsc::channel(queue_size.max(1));
    let (placement_preview_tx, placement_preview_rx) =
        tokio::sync::mpsc::channel(queue_size.max(1));

    // ── Create AppState ────────────────────────────────────────────────────
    let mut state = if startup.start_gpu_workers {
        if let Some(paths) = ModelPaths::resolve(&model_name, &config) {
            info!(model = %model_name, "configured model");
            info!(transformer = %paths.transformer.display());
            info!(vae = %paths.vae.display());
            if let Some(spatial_upscaler) = &paths.spatial_upscaler {
                info!(spatial_upscaler = %spatial_upscaler.display());
            }
            if let Some(t5) = &paths.t5_encoder {
                info!(t5 = %t5.display());
            }
            if let Some(clip) = &paths.clip_encoder {
                info!(clip = %clip.display());
            }
            if let Some(t5_tok) = &paths.t5_tokenizer {
                info!(t5_tok = %t5_tok.display());
            }
            if let Some(clip_tok) = &paths.clip_tokenizer {
                info!(clip_tok = %clip_tok.display());
            }
            if let Some(clip2) = &paths.clip_encoder_2 {
                info!(clip2 = %clip2.display());
            }
            if let Some(clip2_tok) = &paths.clip_tokenizer_2 {
                info!(clip2_tok = %clip2_tok.display());
            }
            for (i, te) in paths.text_encoder_files.iter().enumerate() {
                info!(text_encoder_shard = i, path = %te.display());
            }
            if let Some(text_tok) = &paths.text_tokenizer {
                info!(text_tok = %text_tok.display());
            }
            info!("multi-GPU mode defers model loading to per-GPU workers");
        } else {
            info!("no default model configured — models will be pulled on first request");
        }
        let mut state = state::AppState::empty_with_device_registry(
            config,
            queue_handle,
            gpu_pool.clone(),
            queue_size,
            device_registry.clone(),
        );
        state.shared_pool = shared_pool;
        state
    } else if startup.create_cpu_engine {
        match ModelPaths::resolve(&model_name, &config) {
            Some(paths) => {
                info!(model = %model_name, "configured model");
                info!(transformer = %paths.transformer.display());
                info!(vae = %paths.vae.display());
                if let Some(spatial_upscaler) = &paths.spatial_upscaler {
                    info!(spatial_upscaler = %spatial_upscaler.display());
                }
                if let Some(t5) = &paths.t5_encoder {
                    info!(t5 = %t5.display());
                }
                if let Some(clip) = &paths.clip_encoder {
                    info!(clip = %clip.display());
                }
                if let Some(t5_tok) = &paths.t5_tokenizer {
                    info!(t5_tok = %t5_tok.display());
                }
                if let Some(clip_tok) = &paths.clip_tokenizer {
                    info!(clip_tok = %clip_tok.display());
                }
                if let Some(clip2) = &paths.clip_encoder_2 {
                    info!(clip2 = %clip2.display());
                }
                if let Some(clip2_tok) = &paths.clip_tokenizer_2 {
                    info!(clip2_tok = %clip2_tok.display());
                }
                for (i, te) in paths.text_encoder_files.iter().enumerate() {
                    info!(text_encoder_shard = i, path = %te.display());
                }
                if let Some(text_tok) = &paths.text_tokenizer {
                    info!(text_tok = %text_tok.display());
                }

                let offload =
                    mold_inference::runtime_env::value("MOLD_OFFLOAD").is_some_and(|v| v == "1");
                let engine = mold_inference::create_engine_with_pool(
                    model_name,
                    paths,
                    &config,
                    mold_inference::LoadStrategy::Eager,
                    0,
                    offload,
                    Some(shared_pool.clone()),
                )?;
                let mut state = state::AppState::new(
                    engine,
                    config,
                    queue_handle,
                    gpu_pool.clone(),
                    queue_size,
                );
                state.shared_pool = shared_pool;
                state
            }
            None => {
                info!("no default model configured — models will be pulled on first request");
                state::AppState::empty_with_device_registry(
                    config,
                    queue_handle,
                    gpu_pool.clone(),
                    queue_size,
                    device_registry.clone(),
                )
            }
        }
    } else {
        let mut state = state::AppState::empty_with_device_registry(
            config,
            queue_handle,
            gpu_pool.clone(),
            queue_size,
            device_registry.clone(),
        );
        state.shared_pool = shared_pool;
        let reason = if matches!(gpu_selection, GpuSelection::None) {
            "generation is unavailable while GPU selection is 'none' (maintenance mode)"
        } else {
            "generation is unavailable because no safely selected GPU worker is available"
        };
        state.set_generation_unavailable(reason);
        state
    };
    state.scheduled_work = scheduler::ScheduledWorkHandle::for_runtime(
        scheduled_work_tx,
        dispatch_mode,
        startup.start_v2_coordinator,
        startup.observe_v2_decisions,
    );
    if startup.start_v2_coordinator {
        state.scheduled_work = state
            .scheduled_work
            .clone()
            .with_placement_preview(placement_preview_tx);
    }
    state.metadata_db = metadata_db;
    state.queue_journal = queue_journal.clone();
    queue_journal
        .install_event_broadcaster(state.events.clone())
        .map_err(anyhow::Error::msg)?;
    state.generation_cancel = generation_cancel.clone();
    state.device_registry = device_registry;

    // One admission/observer service is shared by both route shapes and the
    // sole durable feeder. Install it before runtime recovery or the router.
    if let Some(lifecycle) = queue_journal.queue_media_lifecycle() {
        install_durable_admission_if_available(&queue_journal, lifecycle, state.queue_capacity);
    }

    // Startup token recovery is a serving precondition. Await it before any
    // generation producer or router exists, so an HTTP admission from this
    // runtime can never have its freshly minted ownership token cleared by a
    // late feeder recovery pass. Propagating the error fails startup closed.
    durable_queue_feeder::recover_runtime(&state).await?;

    // Gallery publication recovery is a serving precondition, including when
    // SQLite is disabled. No router, gallery observer, or generation job
    // producer is started until every staged transaction is either rolled back
    // or rolled forward and the committed archive index is installed.
    {
        let config = state.config.read().await;
        if !state.is_output_disabled(&config) {
            let output_dir = config.effective_output_dir();
            drop(config);
            std::fs::create_dir_all(&output_dir)?;
            let report = batch_transaction::recover_transactions(
                &output_dir,
                &state.gallery_publication_gate,
                state.metadata_db.clone(),
            )
            .await?;
            tracing::info!(
                rolled_back = report.rolled_back,
                rolled_forward = report.rolled_forward,
                healed_committed_rows = report.healed_committed_rows,
                "gallery transaction startup recovery complete"
            );
            if let Some(lifecycle) = state.queue_journal.queue_media_lifecycle() {
                let pins = lifecycle
                    .reconcile_gallery_pins(&output_dir, &state.gallery_publication_gate)?;
                tracing::info!(
                    retained = pins.retained,
                    released = pins.released,
                    release_failures = pins.release_failures,
                    untouched = pins.untouched,
                    "gallery retained-media pin reconciliation complete"
                );
            }
        }
    }

    // Resolve the persistent instance id (ephemeral when the DB is
    // unavailable). Scoped per (data dir, port) so two servers sharing one
    // mold.db report distinct identities; the configured port is used, so an
    // ephemeral `--port 0` server shares the `.0` slot with other `--port 0`
    // runs on the same DB — its address changes every run anyway. Captured
    // for mDNS here because `state` is moved into the router before the TXT
    // records are built.
    state.instance_id = std::sync::Arc::new(instance_id);
    #[cfg(feature = "mdns")]
    let mdns_instance_id = state.instance_id.clone();

    if state.metadata_db.is_some() {
        let Some(jobs_root) = Config::mold_dir().map(|dir| dir.join("jobs")) else {
            anyhow::bail!("metadata DB opened but MOLD_HOME could not be resolved for chain jobs");
        };
        std::fs::create_dir_all(&jobs_root)?;
        let db_arc = state.metadata_db.clone();
        let reconcile_root = jobs_root.clone();
        let (paused, repaired) = tokio::task::spawn_blocking(move || {
            let Some(db) = db_arc.as_ref().as_ref() else {
                anyhow::bail!("metadata DB disappeared before chain reconcile");
            };
            chain_job_runner::startup_reconcile(db, &reconcile_root)
        })
        .await??;
        tracing::info!(
            paused,
            repaired,
            jobs_root = %jobs_root.display(),
            "chain job startup reconcile complete"
        );

        let gc_db = state.metadata_db.clone();
        let gc_root = jobs_root.clone();
        let startup_gc = tokio::task::spawn_blocking(move || {
            let Some(db) = gc_db.as_ref().as_ref() else {
                anyhow::bail!("metadata DB disappeared before chain startup GC");
            };
            chain_job_runner::startup_gc_sweep(db, &gc_root)
        })
        .await??;
        tracing::info!(
            swept_ephemeral_jobs = startup_gc.swept_ephemeral_jobs,
            pruned_artifact_dirs = startup_gc.pruned_artifact_dirs,
            jobs_root = %jobs_root.display(),
            "chain job startup GC complete"
        );

        let config_snapshot = state.config.read().await.clone();
        let output_dir = if state.is_output_disabled(&config_snapshot) {
            None
        } else {
            Some(config_snapshot.effective_output_dir())
        };
        if startup.start_chain_runner {
            let deps = chain_job_runner::RunnerDeps {
                db: state.metadata_db.clone(),
                jobs_root,
                executor: std::sync::Arc::new(chain_job_runner::ProductionStageExecutor::new(
                    state.gpu_pool.clone(),
                    state.config.clone(),
                    state.scheduled_work.clone(),
                    dispatch_mode,
                )),
                queue_probe: std::sync::Arc::new(chain_job_runner::ProductionQueueProbe::new(
                    state.queue.clone(),
                    state.gpu_pool.clone(),
                )),
                events: std::sync::Arc::new(chain_job_runner::JobEventBus::new()),
                cancel: std::sync::Arc::new(chain_job_runner::CancelRegistry::new()),
                job_locks: std::sync::Arc::new(chain_job_runner::JobMutationLocks::new()),
                claims: std::sync::Arc::new(chain_job_runner::EphemeralClaims::default()),
                output_dir,
                server_events: Some(state.events.clone()),
                gallery_publication_gate: state.gallery_publication_gate.clone(),
                dispatch_mode,
                pause: Some(state.queue_pause.clone()),
            };
            state.chain_jobs = Some(std::sync::Arc::new(chain_job_runner::spawn_runner(deps)));
        } else {
            info!("durable chain runner disabled while generation is unavailable");
        }
    }

    // Spawn the generation queue worker — processes jobs sequentially (single GPU).
    // Spawn queue worker: use multi-GPU dispatcher if GPUs are available,
    // otherwise fall back to the single-threaded queue worker.
    let worker_state = state.clone();
    let scheduler_shutdown = tokio_util::sync::CancellationToken::new();
    let uses_cooperative_gpu_dispatch =
        startup.start_v2_coordinator || startup.start_legacy_dispatcher;
    let generation_worker_handle = if startup.start_v2_coordinator {
        drop(legacy_owner_event_rx);
        Some(tokio::spawn(scheduler::run_scheduler_coordinator(
            job_rx,
            scheduled_work_rx,
            placement_preview_rx,
            scheduler_worker_rx,
            scheduler_worker_tx.clone(),
            worker_state,
            scheduler_shutdown.clone(),
        )))
    } else if startup.start_legacy_dispatcher {
        drop(scheduler_worker_rx);
        let generation_shutdown = scheduler_shutdown.clone();
        let utility_shutdown = scheduler_shutdown.clone();
        Some(tokio::spawn(async move {
            tokio::join!(
                queue::run_queue_dispatcher_until_cancelled(
                    job_rx,
                    worker_state.clone(),
                    generation_shutdown,
                ),
                queue::run_legacy_scheduled_work_dispatcher(
                    scheduled_work_rx,
                    legacy_owner_event_rx,
                    worker_state,
                    utility_shutdown,
                ),
            );
        }))
    } else {
        match startup.mode {
            StartupMode::CpuFallback => {
                drop(legacy_owner_event_rx);
                drop(scheduled_work_rx);
                drop(scheduler_worker_rx);
                Some(tokio::spawn(queue::run_queue_worker(job_rx, worker_state)))
            }
            StartupMode::Maintenance => {
                drop(legacy_owner_event_rx);
                drop(job_rx);
                drop(scheduled_work_rx);
                drop(scheduler_worker_rx);
                None
            }
            StartupMode::GpuWorkers => {
                unreachable!("GPU startup must select one dispatch owner")
            }
        }
    };

    // A SIGKILL runs no destructor, so every hard stop used to leak a
    // directory of reference media under MOLD_HOME. Swept in the same startup
    // pass that recovers the queue, because the two have the same cause.
    {
        let sweep = reference_uploads::sweep_orphaned_staging_roots(state.reference_uploads.root());
        if sweep.removed > 0 || sweep.live > 0 {
            info!(
                removed = sweep.removed,
                live = sweep.live,
                "swept reference-upload staging roots"
            );
        }
        if sweep.untracked > 0 {
            // Not deleted on purpose: without a lock their liveness cannot be
            // established, and another server may still be using them.
            info!(
                untracked = sweep.untracked,
                "reference-upload staging roots predate lock tracking and were left alone; \
                 remove them by hand once no other mold server is using this MOLD_HOME"
            );
        }
    }

    // Reclaim gallery writes that were staged but never published. The bounded
    // shutdown deadline makes an interrupted write routine, so without this the
    // partials accumulate forever.
    {
        let config = state.config.read().await;
        if !state.is_output_disabled(&config) {
            let output_dir = config.effective_output_dir();
            drop(config);
            let swept = tokio::task::spawn_blocking(move || {
                queue::sweep_stale_gallery_partials(&output_dir)
            })
            .await
            .unwrap_or(0);
            if swept > 0 {
                info!(
                    swept,
                    "removed gallery writes interrupted before publication"
                );
            }
        }
    }

    // Runtime-token recovery already completed as a startup barrier. The
    // feeder performs bounded claims and per-claim output idempotence without
    // materializing the retained backlog.
    let durable_feeder_handle = startup
        .start_generation_runner
        .then(|| durable_queue_feeder::spawn(state.clone(), scheduler_shutdown.child_token()));

    // Background idle-TTL sweeper: reclaims parked engines that haven't been
    // touched for `MOLD_CACHE_IDLE_TTL_SECS` seconds. Abort handle bound to
    // graceful shutdown like every other long-running task in this fn.
    let idle_evict_handle = if startup.start_legacy_cache_evictor {
        Some(spawn_cache_idle_evictor(
            state.model_cache.clone(),
            state.model_load_lock.clone(),
            cache_idle_ttl,
        ))
    } else {
        None
    };

    // ── Downloads UI (Agent A) ──────────────────────────────────────────────
    // Single-writer download queue driver. Bind the `JoinHandle` so we can
    // `.abort()` it when `axum::serve` returns — same pattern as the resource
    // telemetry aggregator (see commit 5e43886). Without this the task would
    // outlive graceful shutdown and keep polling its cancellation token until
    // process exit.
    let downloads_shutdown = tokio_util::sync::CancellationToken::new();
    let downloads_models_dir = state.config.read().await.resolved_models_dir();
    let downloads_driver = crate::downloads::spawn_driver(
        state.downloads.clone(),
        std::sync::Arc::new(crate::downloads::HfPullDriver),
        std::sync::Arc::new(crate::downloads::CivitaiRecipeDriver),
        downloads_models_dir,
        downloads_shutdown.clone(),
    );

    // Keep gallery observers owned by this server future. Shutdown cannot
    // complete (and an embedded desktop cannot take direct filesystem
    // authority) until both finite tasks have joined.
    let mut thumbnail_warmup_handle = None;
    let mut gallery_reconcile_handle = None;
    // Retention sweeper for the gallery trash: first pass after reconcile
    // settles the trash index, then hourly. Cancelled through a child of the
    // scheduler token (`begin_runtime_shutdown` cancels it) and joined after
    // the HTTP server drains like the other gallery observers.
    let mut trash_sweeper_handle = None;

    // Retention sweeper for HELD durable queue rows and SETTLED batch
    // summaries. Unlike the trash sweeper it does not wait on the gallery
    // reconcile: it reads the queue, not the output directory, and a held
    // row's media is released by the `generation_queue_media_retire` trigger
    // rather than by a file walk.
    let queue_sweeper_handle =
        queue_retention::spawn_queue_sweeper(state.clone(), scheduler_shutdown.child_token());

    // Ensure output directory exists and pre-generate thumbnails.
    {
        let config = state.config.read().await;
        if state.is_output_disabled(&config) {
            tracing::warn!(
                "image output is disabled (output_dir is empty) — \
                 generated images will not be saved and the TUI gallery will be empty"
            );
        } else {
            let output_dir = config.effective_output_dir();
            let _ = std::fs::create_dir_all(&output_dir);
            info!(output_dir = %output_dir.display(), "gallery output directory");
            thumbnail_warmup_handle =
                routes::spawn_thumbnail_warmup(&config, state.gallery_publication_gate.clone());

            // Async reconcile: import any existing files into the DB and
            // drop rows whose backing files are missing. Runs on a blocking
            // worker so it never stalls the request path even on large dirs.
            if state.metadata_db.is_some() {
                let db_arc = state.metadata_db.clone();
                let dir = output_dir.clone();
                let gallery_gate = state.gallery_publication_gate.clone();
                let (reconciled_tx, reconciled_rx) = tokio::sync::oneshot::channel::<()>();
                gallery_reconcile_handle = Some(tokio::spawn(async move {
                    // Reconcile mutates SQLite but only observes gallery
                    // files. The shared side keeps publication atomic while
                    // allowing listings/media reads to remain available.
                    let _reader = gallery_gate.read().await;
                    let join = tokio::task::spawn_blocking(move || {
                        if let Some(db) = db_arc.as_ref() {
                            db.reconcile(&dir)
                        } else {
                            Ok(mold_db::ReconcileStats::default())
                        }
                    })
                    .await;
                    match join {
                        Ok(Ok(stats)) => tracing::info!(
                            imported = stats.imported,
                            updated = stats.updated,
                            removed = stats.removed,
                            kept = stats.kept,
                            trashed_kept = stats.trashed_kept,
                            trashed_imported = stats.trashed_imported,
                            trashed_restored = stats.trashed_restored,
                            "metadata DB reconciled with gallery directory"
                        ),
                        Ok(Err(e)) => tracing::warn!("metadata DB reconcile failed: {e:#}"),
                        Err(e) => tracing::warn!("metadata DB reconcile task join error: {e}"),
                    }
                    let _ = reconciled_tx.send(());
                }));
                trash_sweeper_handle = Some(gallery_trash::spawn_trash_sweeper(
                    state.clone(),
                    scheduler_shutdown.child_token(),
                    Some(reconciled_rx),
                ));
            }
        }
    }

    // Load optional auth and rate-limit configuration from env vars.
    let auth_state =
        auth::load_api_keys_with_db(state.metadata_db.clone(), state.instance_id.clone())?;
    // Capture whether auth is required before `auth_state` is moved into the
    // router below — surfaced in the mDNS TXT `auth` flag.
    #[cfg(feature = "mdns")]
    let mdns_auth_required = auth_state.is_some();
    let rl_config = rate_limit::load_rate_limit_config()?;

    let cors = build_cors_layer()?;

    // Install the Prometheus metrics recorder (when feature-enabled).
    // Must happen before any middleware or handler that records metrics.
    #[cfg(feature = "metrics")]
    let prometheus_handle = metrics::install_recorder();
    #[cfg(feature = "metrics")]
    metrics::record_dispatch_mode(dispatch_mode);

    // Build the router with middleware layers.
    // Order (outermost → innermost): CORS → Trace → RequestID → Metrics → Auth → RateLimit → routes
    // All inject + enforce layers use .layer() (not .route_layer()) so they run on
    // ALL requests, including unmatched 404 paths — preventing auth/rate-limit bypass.
    // Set up graceful shutdown: fires on SIGTERM or POST /api/shutdown.
    // The public trigger feeds an arbiter that fences GPU scheduling and
    // active chain attempts *before* Axum starts draining long-lived SSE
    // responses. Waiting until `serve` returned could wedge forever on the
    // very chain stream whose inference still needed cancellation (#586).
    let (shutdown_tx, shutdown_request_rx) = tokio::sync::oneshot::channel::<()>();
    let (http_shutdown_tx, http_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    *state.shutdown_tx.lock().await = Some(shutdown_tx);
    let shutdown_scheduler = scheduler_shutdown.clone();
    let shutdown_chain_jobs = state.chain_jobs.clone();
    let shutdown_metadata_db = state.metadata_db.clone();
    let shutdown_journal = state.queue_journal.clone();
    let shutdown_generation_cancel = state.generation_cancel.clone();
    let shutdown_event_streams = state.events.clone();
    let shutdown_fatal_cuda = fatal_cuda_error.clone();
    tokio::spawn(async move {
        let _ = shutdown_request_rx.await;
        // Armed BEFORE the sequence, not after it: everything below this
        // point can block, and a deadline that only starts once the drain is
        // over bounds nothing.
        arm_shutdown_deadline(shutdown_fatal_cuda);
        begin_runtime_shutdown(
            shutdown_chain_jobs.as_deref(),
            shutdown_metadata_db.as_ref().as_ref(),
            &shutdown_scheduler,
            &shutdown_journal,
            &shutdown_generation_cancel,
            &shutdown_event_streams,
        );
        let _ = http_shutdown_tx.send(());
    });

    #[cfg(unix)]
    {
        let sigterm_state = state.clone();
        tokio::spawn(async move {
            if let Ok(mut sig) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                sig.recv().await;
                tracing::info!("received SIGTERM, initiating graceful shutdown");
                if let Some(tx) = sigterm_state.shutdown_tx.lock().await.take() {
                    let _ = tx.send(());
                }
            }
        });
    }

    // Windows has no SIGTERM, so without this arm the graceful path — and with
    // it the queue journal's retention fence, which MUST go up before the
    // scheduler is cancelled — was reachable on Windows only through
    // `POST /api/shutdown`.
    //
    // `CTRL_CLOSE` and `CTRL_SHUTDOWN` are the SIGTERM analogues. `CTRL_C` is
    // included even though the unix arm deliberately leaves SIGINT alone,
    // because a console `mold serve` is the ordinary way to run one on Windows
    // and Ctrl+C is how it is stopped; the cost is that Ctrl+C now drains
    // rather than killing instantly. `serve` calls `allow_hard_shutdown_exit`,
    // and a second Ctrl+C falls through to the OS default once this task has
    // ended, so it cannot become a hang.
    //
    // Two limits, stated rather than implied. The OS grace period for
    // CTRL_CLOSE is a few seconds, well under `DEFAULT_SHUTDOWN_ABORT_SECS`,
    // so the tail of the drain WILL be cut off — what this buys is that
    // `retain_all()` runs first, in the opening milliseconds. And these events
    // reach console-attached processes only: the GUI desktop app embeds this
    // server in a windowed process that receives none of them, so its quit and
    // machine-shutdown paths still need Tauri's own exit hooks.
    #[cfg(windows)]
    {
        let console_state = state.clone();
        tokio::spawn(async move {
            use tokio::signal::windows;

            let (Ok(mut interrupt), Ok(mut close), Ok(mut shutdown)) = (
                windows::ctrl_c(),
                windows::ctrl_close(),
                windows::ctrl_shutdown(),
            ) else {
                tracing::warn!("could not install Windows console control handlers");
                return;
            };
            let event = tokio::select! {
                _ = interrupt.recv() => "CTRL_C",
                _ = close.recv() => "CTRL_CLOSE",
                _ = shutdown.recv() => "CTRL_SHUTDOWN",
            };
            tracing::info!("received {event}, initiating graceful shutdown");
            if let Some(tx) = console_state.shutdown_tx.lock().await.take() {
                let _ = tx.send(());
            }
        });
    }

    // Spawn the resource telemetry aggregator (1 Hz). Keep the `JoinHandle`
    // bound so we can `.abort()` it when `axum::serve` returns — otherwise
    // the task outlives server shutdown and keeps ticking until process exit.
    let resources_aggregator =
        resources::spawn_aggregator(state.resources.clone(), state.device_registry.clone());

    // Start a long-lived DNS-SD browser independently of advertising. A
    // loopback-bound primary still needs to surface the server machine's LAN
    // peers to its web UI; only the MOLD_MDNS/--no-mdns gate disables browse.
    #[cfg(feature = "mdns")]
    let mdns_browser_guard = if mdns::enabled_from_env() {
        match mdns::start_browser(state.discovery.clone(), state.instance_id.clone()) {
            Ok(guard) => Some(guard),
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "mDNS peer browsing disabled");
                None
            }
        }
    } else {
        tracing::debug!("mDNS peer browsing skipped (disabled)");
        None
    };

    // Save start_time before state is moved into the router (needed for metrics).
    #[cfg(feature = "metrics")]
    let server_start_time = state.start_time;

    // The /metrics endpoint is mounted outside the auth/rate-limit stack so it
    // is always accessible for monitoring scrapers (Prometheus, Grafana Agent, etc.).
    // Recovery belongs to the server lifecycle, not router construction.
    // Tests and in-process callers may construct a router while deliberately
    // holding SQLite; doing recovery there would synchronously deadlock before
    // the request future could be spawned. Keep the dispatcher handle so the
    // server can prove it no longer owns background polling at shutdown.
    let video_upscale_dispatcher = video_upscale::recover_at_startup(&state);
    #[allow(unused_mut)]
    let mut app = routes::create_router(state)
        .merge(web_ui::router())
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(middleware::from_fn_with_state(
            rl_config,
            rate_limit::inject_rate_limit_state,
        ))
        .layer(middleware::from_fn(auth::require_api_key))
        .layer(middleware::from_fn_with_state(
            auth_state,
            auth::inject_auth_state,
        ));

    // HTTP metrics middleware sits outside auth so it observes all requests
    // (including auth failures and rate-limited responses).
    #[cfg(feature = "metrics")]
    {
        app = app.layer(middleware::from_fn(metrics::http_metrics_middleware));
    }

    #[cfg(feature = "metrics")]
    {
        let metrics_state = metrics::MetricsState {
            handle: prometheus_handle,
            start_time: server_start_time,
        };
        app = app.route(
            "/metrics",
            axum::routing::get(metrics::metrics_endpoint).with_state(metrics_state),
        );
    }

    let app = app
        .layer(middleware::from_fn(request_id::request_id_middleware))
        .layer(TraceLayer::new_for_http().make_span_with(
            |request: &axum::http::Request<axum::body::Body>| {
                tracing::debug_span!(
                    "http-request",
                    method = %request.method(),
                    path = %trace_request_path(request),
                    version = ?request.version(),
                )
            },
        ))
        .layer(cors);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    let version = mold_core::build_info::version_string();
    info!(%addr, %version, "starting mold server");

    let listener = TcpListener::bind(addr).await?;

    // ── mDNS/DNS-SD advertising ─────────────────────────────────────────────
    // Advertise this server as `_mold._tcp.local.` so desktop clients and
    // `mold server discover` find it without a configured host. Uses the real
    // bound port (supports `--port 0`), skips loopback binds, and honours the
    // MOLD_MDNS env toggle. The guard unregisters on the shutdown path below.
    #[cfg(feature = "mdns")]
    let mdns_guard = {
        let bound_port = listener.local_addr()?.port();
        if mdns::enabled_from_env() && mdns::is_advertisable(bind, bound_port) {
            let txt = mdns::build_txt_records(
                mold_core::build_info::VERSION,
                mold_core::build_info::GIT_SHA,
                mdns_auth_required,
                &mdns_gpu_summary,
                queue_size,
                &mdns_instance_id,
            );
            match mdns::register(bind, bound_port, txt) {
                Ok(guard) => Some(guard),
                Err(e) => {
                    tracing::warn!(error = %format!("{e:#}"), "mDNS advertising disabled");
                    None
                }
            }
        } else {
            tracing::debug!("mDNS advertising skipped (disabled or loopback bind)");
            None
        }
    };

    let fatal_cuda_journal = queue_journal.clone();
    let fatal_cuda_deadline = fatal_cuda_error.clone();
    let (drain_started_tx, drain_started_rx) = tokio::sync::oneshot::channel::<()>();
    // A block, so the serve future — and with it the listener — is dropped
    // before the shutdown steps below run, on every arm. (On the fatal-CUDA
    // arm this is what that arm's own comment always described; the
    // listener used to stay open, backlogging connects, until `run_server`
    // returned.)
    {
        let server = std::future::IntoFuture::into_future(
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(async move {
                let _ = http_shutdown_rx.await;
                tracing::info!("shutting down");
                let _ = drain_started_tx.send(());
            }),
        );
        tokio::pin!(server);
        let drain_grace = http_drain_grace();
        tokio::select! {
            // Biased: a drain that completes on the same tick its grace
            // elapses is a completed drain, and its result is not swallowed.
            biased;
            result = &mut server => result?,
            _ = drain_grace_elapsed(drain_started_rx, drain_grace) => {
                // Graceful shutdown waits for every in-flight request, and a
                // client the host does not control can keep one in flight for
                // as long as it likes: a webview that stops draining a paused
                // video, a request whose body never arrives. The host asked for
                // a stop with a budget it cannot enforce by ending the process,
                // so the drain is bounded here. Dropping the serve future closes
                // the listener; the connections themselves end with the runtime
                // the host tears down once this returns. Until then a handler
                // this arm abandoned may still be running against the
                // subsystems the steps below tear down — `begin_runtime_shutdown`
                // fenced scheduling before the drain began, and by definition
                // that request is stuck, so the steps do not wait for it.
                if let Some(grace) = drain_grace {
                    tracing::warn!(
                        ?grace,
                        "in-flight HTTP requests outlived the drain grace; giving up on them"
                    );
                }
            }
            _ = fatal_cuda_shutdown.notified() => {
                // Before anything discards: this path ends in `anyhow::bail!` and a
                // supervised restart, so the queue must survive it. Missing this
                // fence would make the one restart mold performs on its own behalf
                // the one that deletes every queued job.
                fatal_cuda_journal.retain_all();
                // Same bound as the operator-initiated path: this restart is the
                // one mold performs on its own behalf, and it must not hang.
                arm_shutdown_deadline(fatal_cuda_deadline);
                tracing::error!("fatal CUDA context error; stopping server for process restart");
                // Give the triggering request a brief window to receive its explicit
                // fatal-context error, then drop the server future. A normal graceful
                // shutdown could wait forever on queued SSE requests assigned to the
                // now-quarantined worker and prevent the service manager from restarting.
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        }
    }

    // Server has stopped accepting requests — cancel the downloads token so the
    // driver's `wait_for_work` arm returns, then abort the JoinHandle to ensure
    // the task is cleaned up on the same shutdown path as the HTTP server.
    // Matches the aggregator handle pattern from commit 5e43886.
    tracing::debug!(step = "mdns-advertise", "awaiting shutdown step");
    #[cfg(feature = "mdns")]
    if let Some(guard) = mdns_guard {
        guard.shutdown();
    }
    tracing::debug!(step = "mdns-browse", "awaiting shutdown step");
    #[cfg(feature = "mdns")]
    if let Some(guard) = mdns_browser_guard {
        guard.shutdown();
    }
    tracing::debug!(step = "downloads", "awaiting shutdown step");
    downloads_shutdown.cancel();
    downloads_driver.abort();
    let _ = downloads_driver.await;
    tracing::debug!(step = "video-upscale", "awaiting shutdown step");
    if let Some(runtime) = video_upscale_dispatcher {
        runtime.shutdown().await;
    }
    tracing::debug!(step = "idle-evict", "awaiting shutdown step");
    if let Some(handle) = idle_evict_handle {
        handle.abort();
        let _ = handle.await;
    }
    // Server has stopped accepting requests — stop the telemetry aggregator
    // so it doesn't outlive the server loop.
    tracing::debug!(step = "resources-aggregator", "awaiting shutdown step");
    resources_aggregator.abort();
    let _ = resources_aggregator.await;

    // These tasks may be inside spawn_blocking and therefore cannot be safely
    // detached or cancelled. Await their actual completion before returning
    // server authority to an in-process lifecycle owner.
    tracing::debug!(step = "thumbnail-warmup", "awaiting shutdown step");
    if let Some(handle) = thumbnail_warmup_handle {
        let _ = handle.await;
    }
    tracing::debug!(step = "gallery-reconcile", "awaiting shutdown step");
    if let Some(handle) = gallery_reconcile_handle {
        let _ = handle.await;
    }
    tracing::debug!(step = "scheduler-cancel", "awaiting shutdown step");
    scheduler_shutdown.cancel();
    tracing::debug!(step = "durable-feeder", "awaiting shutdown step");
    if let Some(handle) = durable_feeder_handle {
        let _ = handle.await;
    }
    tracing::debug!(step = "trash-sweeper", "awaiting shutdown step");
    if let Some(handle) = trash_sweeper_handle {
        // Its token is a child of `scheduler_shutdown`, so the loop exits on
        // the cancel above; a pass already inside `spawn_blocking` finishes.
        let _ = handle.await;
    }
    tracing::debug!(step = "queue-sweeper", "awaiting shutdown step");
    // Same child-token contract as the trash sweeper.
    let _ = queue_sweeper_handle.await;
    tracing::debug!(step = "generation-worker", "awaiting shutdown step");
    if let Some(generation_worker_handle) = generation_worker_handle {
        if !uses_cooperative_gpu_dispatch {
            // The CPU/legacy worker predates the coordinator cancellation
            // protocol. Abort and await it explicitly so in-process restart never
            // inherits a detached queue task.
            generation_worker_handle.abort();
        }
        let _ = generation_worker_handle.await;
    }
    // Also issue shutdown from the owner even if the coordinator panicked
    // before its normal teardown path.
    // The coordinator sends an explicit shutdown command to every idle owner
    // and sets the shared flag for any owner finishing a current lease.
    // Joining here makes an in-process server restart incapable of inheriting
    // detached CUDA owner threads or contexts.
    tracing::debug!(step = "gpu-owners", "awaiting shutdown step");
    let shutdown_pool = gpu_pool.clone();
    let join = tokio::task::spawn_blocking(move || {
        shutdown_pool.workers.shutdown_and_join_all();
        gpu_owner_threads.shutdown_and_join()
    });
    // Stop *awaiting* the owners at the budget so a lifecycle owner that can
    // still act gets control back. This is deliberately not the bound —
    // dropping a `spawn_blocking` handle does not cancel the blocking work,
    // and the runtime waits on it at teardown. `arm_shutdown_deadline`, armed
    // when shutdown began, is what actually ends an overrunning process.
    let budget = std::time::Duration::from_secs(resolve_shutdown_abort_secs());
    match tokio::time::timeout(budget, join).await {
        Ok(joined) => joined
            .map_err(|error| anyhow::anyhow!("failed to run GPU owner join task: {error}"))??,
        Err(_) => {
            tracing::warn!(
                budget_secs = budget.as_secs(),
                env = SHUTDOWN_ABORT_SECS_ENV,
                "a GPU owner did not return within the shutdown budget; \
                 no longer waiting — retained jobs replay after restart"
            );
        }
    }

    tracing::debug!("shutdown sequence complete");

    if fatal_cuda_error.load(std::sync::atomic::Ordering::SeqCst) {
        anyhow::bail!("fatal CUDA context error; server restart required");
    }

    Ok(())
}

/// Total budget for joining GPU owner threads at shutdown. The unit's
/// `TimeoutStopSec` is derived from this and is strictly larger, so systemd is
/// never the component that decides.
pub const SHUTDOWN_ABORT_SECS_ENV: &str = "MOLD_SHUTDOWN_ABORT_SECS";
/// Long enough for an ordinary render to reach its next checkpoint and unwind,
/// short enough that a deploy never waits on a cold 19B load.
pub const DEFAULT_SHUTDOWN_ABORT_SECS: u64 = 45;

/// Resolve the shutdown join budget, warning rather than failing on nonsense.
pub fn resolve_shutdown_abort_secs() -> u64 {
    match std::env::var(SHUTDOWN_ABORT_SECS_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(secs) if secs >= 1 => secs,
            Ok(_) => {
                tracing::warn!(
                    env = SHUTDOWN_ABORT_SECS_ENV,
                    "shutdown budget must be at least 1 second; using the default"
                );
                DEFAULT_SHUTDOWN_ABORT_SECS
            }
            Err(error) => {
                tracing::warn!(
                    env = SHUTDOWN_ABORT_SECS_ENV,
                    raw = %raw,
                    %error,
                    "ignoring unparseable shutdown budget"
                );
                DEFAULT_SHUTDOWN_ABORT_SECS
            }
        },
        Err(_) => DEFAULT_SHUTDOWN_ABORT_SECS,
    }
}

/// Whether this process may end itself when the shutdown budget expires.
///
/// Off by default: `run_server` is also embedded in the desktop app, which
/// runs it on a thread inside its own process and expects it to *return*.
/// Exiting there would take the whole application down over a slow engine
/// stop. `mold serve` and the standalone binary opt in.
static HARD_SHUTDOWN_EXIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Opt this process into ending itself if shutdown overruns its budget.
///
/// Call once, before `run_server`, from a binary whose only job is to be the
/// server. Without it the budget can only stop *waiting*, which is not the
/// same as bounding shutdown: dropping a `spawn_blocking` handle does not
/// cancel the blocking work, and the runtime blocks on it at teardown.
pub fn allow_hard_shutdown_exit() {
    HARD_SHUTDOWN_EXIT.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn hard_shutdown_exit_allowed() -> bool {
    HARD_SHUTDOWN_EXIT.load(std::sync::atomic::Ordering::SeqCst)
}

/// How long in-flight HTTP requests get once shutdown begins before the
/// server gives up on them, in milliseconds; zero means the drain is
/// unbounded, which is the standalone server's contract (its hard deadline
/// ends the process instead).
static HTTP_DRAIN_GRACE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bound the HTTP drain for a host that owns the process and cannot end it.
///
/// Axum's graceful shutdown waits for every in-flight request, and a client
/// the host does not control can keep one in flight indefinitely: a webview
/// that stops draining a paused video, a request whose body never arrives, a
/// reader that simply stopped reading a megabyte-sized response. The
/// standalone server bounds that with a deadline that ends the process; an
/// embedded server cannot, so it gives in-flight requests `grace` after the
/// shutdown signal and then stops waiting for them. Call once, before
/// `run_server`, from the host that has a stop budget to keep. A zero grace
/// is treated as one millisecond: zero is how "never set" is stored, and a
/// host that calls this has asked for a bound.
pub fn bound_http_drain(grace: std::time::Duration) {
    let millis = u64::try_from(grace.as_millis()).unwrap_or(u64::MAX).max(1);
    HTTP_DRAIN_GRACE_MS.store(millis, std::sync::atomic::Ordering::SeqCst);
}

fn http_drain_grace() -> Option<std::time::Duration> {
    match HTTP_DRAIN_GRACE_MS.load(std::sync::atomic::Ordering::SeqCst) {
        0 => None,
        millis => Some(std::time::Duration::from_millis(millis)),
    }
}

/// Resolves `grace` after the HTTP drain has begun — never before the
/// shutdown signal, and never at all when the drain is unbounded.
async fn drain_grace_elapsed(
    drain_started: tokio::sync::oneshot::Receiver<()>,
    grace: Option<std::time::Duration>,
) {
    let Some(grace) = grace else {
        std::future::pending::<()>().await;
        return;
    };
    let _ = drain_started.await;
    tokio::time::sleep(grace).await;
}

/// What expiry of the shutdown budget should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShutdownExpiry {
    /// End the process now. The status distinguishes an operator-initiated
    /// stop that merely overran from a fatal-CUDA stop that supervision must
    /// restart.
    Exit(i32),
    /// This process belongs to someone else; keep waiting and let them decide.
    KeepWaiting,
}

pub(crate) fn shutdown_expiry_action(hard_exit_allowed: bool, fatal_cuda: bool) -> ShutdownExpiry {
    if !hard_exit_allowed {
        return ShutdownExpiry::KeepWaiting;
    }
    ShutdownExpiry::Exit(if fatal_cuda { 1 } else { 0 })
}

/// Arm the hard shutdown deadline the moment shutdown begins.
///
/// This is what makes `MOLD_SHUTDOWN_ABORT_SECS` a real bound rather than an
/// aspiration. Two things in the drain are individually unbounded and neither
/// can be fixed where it sits: Axum's graceful shutdown waits for in-flight
/// responses, so an SSE stream whose generation is inside a cold model load
/// holds it open indefinitely (the cancellation wrapper covers the generate
/// call, not the load); and a `spawn_blocking` join cannot be cancelled by
/// dropping its handle — the runtime waits on it at teardown regardless. A
/// timeout placed anywhere *inside* the sequence therefore stops waiting
/// without stopping anything.
///
/// So the deadline runs beside the drain instead of within it, and ends the
/// process when it expires. That is safe precisely because of the two
/// invariants this feature already established: the queue is retained before
/// anything discards, and gallery bytes publish by rename, so a kill costs the
/// in-flight render and nothing else. The retained rows replay on restart.
fn spawn_shutdown_deadline(
    budget: std::time::Duration,
    fatal_cuda: std::sync::Arc<AtomicBool>,
    hard_exit_allowed: bool,
    on_expiry: impl FnOnce(ShutdownExpiry) + Send + 'static,
) -> Option<tokio::task::JoinHandle<()>> {
    if !hard_exit_allowed {
        tracing::debug!(
            "shutdown budget is advisory in an embedded server; the host owns the process"
        );
        return None;
    }
    Some(tokio::spawn(async move {
        tokio::time::sleep(budget).await;
        let fatal = fatal_cuda.load(std::sync::atomic::Ordering::SeqCst);
        on_expiry(shutdown_expiry_action(true, fatal));
    }))
}

/// Arm the deadline with the production expiry behaviour: log distinctly,
/// then end the process.
fn arm_shutdown_deadline(fatal_cuda: std::sync::Arc<AtomicBool>) {
    let budget = std::time::Duration::from_secs(resolve_shutdown_abort_secs());
    let allowed = hard_shutdown_exit_allowed();
    let _ = spawn_shutdown_deadline(budget, fatal_cuda, allowed, move |action| match action {
        ShutdownExpiry::Exit(status) => {
            tracing::error!(
                budget_secs = budget.as_secs(),
                env = SHUTDOWN_ABORT_SECS_ENV,
                status,
                "shutdown did not complete within its budget; ending the process now — \
                 retained generations replay on the next start"
            );
            std::process::exit(status);
        }
        ShutdownExpiry::KeepWaiting => {}
    });
}

/// Order is the correctness argument, not a preference.
///
/// The retention fence goes up first, because every later step discards jobs:
/// `scheduler_shutdown.cancel()` reaches `reject_all_unstarted`, which drops
/// each pending `GenerationJob` and therefore its journal ticket. With the
/// fence up those drops retain their rows instead of deleting them, and none
/// of the ~20 discard sites has to know durability exists.
fn begin_runtime_shutdown(
    chain_jobs: Option<&chain_job_runner::ChainJobRunnerHandle>,
    metadata_db: Option<&mold_db::MetadataDb>,
    scheduler_shutdown: &tokio_util::sync::CancellationToken,
    queue_journal: &queue_journal::QueueJournal,
    generation_cancel: &generation_cancel::CancelRegistry,
    event_streams: &events::EventBroadcaster,
) {
    queue_journal.retain_all();
    let aborted = generation_cancel.request_all();
    if aborted > 0 {
        tracing::info!(
            aborted,
            "aborting in-flight generations; they stay queued and are replayed after restart"
        );
    }
    if let Some(chain_jobs) = chain_jobs {
        if let Some(db) = metadata_db {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                .try_into()
                .unwrap_or(i64::MAX);
            match mold_db::chain_jobs::pause_queued_for_shutdown(db, now_ms) {
                Ok(paused) if paused > 0 => {
                    tracing::info!(paused, "parked queued chain jobs for restart");
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::error!(%error, "failed to park active chain jobs for restart");
                }
            }
        }
        let active_chains = chain_jobs.request_shutdown();
        tracing::info!(
            active_chains,
            "interrupted parked chain work before HTTP drain"
        );
    }
    event_streams.shutdown();
    scheduler_shutdown.cancel();
}

/// Spawn a tokio task that wakes every 60s and drops any cache entry whose
/// `last_used` is older than `ttl` (and that isn't actively GPU-resident).
/// Sweeps the legacy cache. Per-GPU caches are evicted by their dedicated
/// worker loops so CUDA-backed engines are destroyed on their owning thread.
/// Returns the `JoinHandle` so the caller can `.abort()` on shutdown.
fn spawn_cache_idle_evictor(
    legacy_cache: std::sync::Arc<tokio::sync::Mutex<model_cache::ModelCache>>,
    legacy_load_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
    ttl: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    use tokio::time::{interval, MissedTickBehavior};
    tokio::spawn(async move {
        let mut tick = interval(std::time::Duration::from_secs(60));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // First tick fires immediately; skip it so a freshly-loaded model
        // doesn't get reaped on boot before it's even been used.
        tick.tick().await;
        loop {
            tick.tick().await;

            // ── Legacy single-GPU cache ─────────────────────────────────────
            //
            // Take the legacy load lock for the full eviction window.
            {
                let _load_guard = legacy_load_lock.lock().await;
                let evicted = {
                    let mut cache = legacy_cache.lock().await;
                    cache.evict_idle(ttl)
                };
                // Drop engines OUTSIDE the cache lock — `cuMemFree` and
                // safetensor unmap during drop can block other cache users.
                drop(evicted);
            }
        }
    })
}

fn build_cors_layer() -> Result<CorsLayer> {
    let cors = match std::env::var("MOLD_CORS_ORIGIN") {
        Ok(origin) if !origin.is_empty() => {
            let origin = origin
                .parse::<axum::http::HeaderValue>()
                .map_err(|_| anyhow::anyhow!("invalid MOLD_CORS_ORIGIN value: {origin}"))?;
            CorsLayer::new()
                .allow_origin(origin)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::HEAD,
                    axum::http::Method::POST,
                    axum::http::Method::PATCH,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([
                    axum::http::header::HeaderName::from_static("x-mold-seed-used"),
                    axum::http::header::HeaderName::from_static("x-request-id"),
                    axum::http::header::HeaderName::from_static("retry-after"),
                    axum::http::header::HeaderName::from_static("x-mold-video-frames"),
                    axum::http::header::HeaderName::from_static("x-mold-video-fps"),
                    axum::http::header::HeaderName::from_static("x-mold-video-width"),
                    axum::http::header::HeaderName::from_static("x-mold-video-height"),
                    axum::http::header::HeaderName::from_static("x-mold-video-pipeline"),
                    axum::http::header::HeaderName::from_static(
                        "x-mold-video-pipeline-provenance-sha256",
                    ),
                    axum::http::header::HeaderName::from_static(
                        "x-mold-video-source-preprocessing",
                    ),
                    axum::http::header::HeaderName::from_static("x-mold-video-has-audio"),
                    axum::http::header::HeaderName::from_static("x-mold-video-duration-ms"),
                    axum::http::header::HeaderName::from_static("x-mold-video-audio-sample-rate"),
                    axum::http::header::HeaderName::from_static("x-mold-video-audio-channels"),
                    axum::http::header::HeaderName::from_static("x-mold-dimension-warning"),
                    axum::http::header::HeaderName::from_static("x-mold-request-warning"),
                    axum::http::header::HeaderName::from_static("x-mold-thumbnail-rendition"),
                ])
        }
        _ => CorsLayer::permissive(),
    };
    Ok(cors)
}

#[cfg(test)]
mod tests {
    use super::{
        begin_runtime_shutdown, build_cors_layer, classify_startup_mode, startup_plan,
        trace_request_path, GpuOwnerThreads, StartupMode,
    };
    use crate::auth::{inject_auth_state, require_api_key, ApiKeySet};
    use crate::device_registry::DeviceRegistry;
    use crate::gpu_pool::{GpuWorker, GpuWorkerCommand};
    use crate::{gpu_worker, model_cache};
    use axum::http::{header, Method, Request};
    use axum::routing::patch;
    use axum::Router;
    use mold_core::types::GpuSelection;
    use mold_inference::device::{CudaDeviceKind, DiscoveredGpu};
    use mold_inference::shared_pool::SharedPool;
    use std::collections::{BTreeSet, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::Duration;
    use tower::ServiceExt;

    #[test]
    fn runtime_shutdown_parks_chains_before_interrupting_gpu_work() {
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::chain_jobs::insert_job(
            &db,
            &mold_db::chain_jobs::ChainJobRow {
                id: "chain-in-flight".into(),
                state: mold_core::chain_job::ChainJobState::Queued,
                model: "ltx2".into(),
                request_json: "{}".into(),
                job_dir: "/tmp/chain-in-flight".into(),
                stage_count: 2,
                current_stage: 0,
                error: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                finalized_at_ms: None,
            },
        )
        .unwrap();
        let chains = crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests();
        chains.register_cancel_for_tests("chain-in-flight");
        let scheduler = tokio_util::sync::CancellationToken::new();

        let journal = Arc::new(crate::queue_journal::QueueJournal::disabled());

        let singletons = Arc::new(crate::generation_cancel::CancelRegistry::new());
        let event_streams = crate::events::EventBroadcaster::new();
        begin_runtime_shutdown(
            Some(&chains),
            Some(&db),
            &scheduler,
            &journal,
            &singletons,
            &event_streams,
        );

        assert_eq!(
            mold_db::chain_jobs::get_job(&db, "chain-in-flight")
                .unwrap()
                .unwrap()
                .state,
            mold_core::chain_job::ChainJobState::Paused,
        );
        assert!(chains.is_cancelling("chain-in-flight"));
        chains.register_cancel_for_tests("chain-claimed-during-shutdown");
        assert!(chains.is_cancelling("chain-claimed-during-shutdown"));
        assert!(scheduler.is_cancelled());
        assert!(event_streams.shutdown_token().is_cancelled());
    }

    /// The whole correctness argument for retention: the fence has to be up
    /// before anything can discard a job. `scheduler_shutdown.cancel()` is
    /// what reaches `reject_all_unstarted`, so the fence must already be
    /// visible from the moment that token resolves.
    #[tokio::test]
    async fn runtime_shutdown_retains_the_queue_before_cancelling_the_scheduler() {
        let scheduler = tokio_util::sync::CancellationToken::new();
        let journal = Arc::new(crate::queue_journal::QueueJournal::disabled());

        let observed = Arc::new(AtomicBool::new(false));
        let watcher = {
            let scheduler = scheduler.clone();
            let journal = journal.clone();
            let observed = observed.clone();
            tokio::spawn(async move {
                scheduler.cancelled().await;
                observed.store(journal.is_retaining(), std::sync::atomic::Ordering::SeqCst);
            })
        };

        begin_runtime_shutdown(
            None,
            None,
            &scheduler,
            &journal,
            &crate::generation_cancel::CancelRegistry::new(),
            &crate::events::EventBroadcaster::new(),
        );
        watcher.await.unwrap();

        assert!(
            observed.load(std::sync::atomic::Ordering::SeqCst),
            "the queue must already be retained when the scheduler starts discarding"
        );
    }

    #[test]
    fn shutdown_budget_falls_back_to_the_default_for_nonsense() {
        let _lock = crate::test_support::env_lock();
        std::env::set_var(super::SHUTDOWN_ABORT_SECS_ENV, "90");
        assert_eq!(super::resolve_shutdown_abort_secs(), 90);
        std::env::set_var(super::SHUTDOWN_ABORT_SECS_ENV, "0");
        assert_eq!(
            super::resolve_shutdown_abort_secs(),
            super::DEFAULT_SHUTDOWN_ABORT_SECS
        );
        std::env::set_var(super::SHUTDOWN_ABORT_SECS_ENV, "soon");
        assert_eq!(
            super::resolve_shutdown_abort_secs(),
            super::DEFAULT_SHUTDOWN_ABORT_SECS
        );
        std::env::remove_var(super::SHUTDOWN_ABORT_SECS_ENV);
        assert_eq!(
            super::resolve_shutdown_abort_secs(),
            super::DEFAULT_SHUTDOWN_ABORT_SECS
        );
    }

    /// The budget is only meaningful if expiry actually ends the process. A
    /// server embedded in the desktop app owns neither the process nor the
    /// decision, so it keeps waiting instead.
    #[test]
    fn shutdown_expiry_exits_only_where_the_process_is_ours_to_end() {
        use super::{shutdown_expiry_action, ShutdownExpiry};

        assert_eq!(
            shutdown_expiry_action(true, false),
            ShutdownExpiry::Exit(0),
            "an operator-initiated stop that overran is still a clean stop"
        );
        assert_eq!(
            shutdown_expiry_action(true, true),
            ShutdownExpiry::Exit(1),
            "a fatal-CUDA stop must exit non-zero so supervision restarts us"
        );
        assert_eq!(
            shutdown_expiry_action(false, false),
            ShutdownExpiry::KeepWaiting
        );
        assert_eq!(
            shutdown_expiry_action(false, true),
            ShutdownExpiry::KeepWaiting
        );
    }

    /// The property that matters: the deadline is armed when shutdown *starts*
    /// and runs independently of the drain, so a request that never completes
    /// — an SSE stream held open by a cold model load, which is exactly the
    /// incident this exists for — cannot postpone it.
    #[tokio::test(start_paused = true)]
    async fn the_shutdown_deadline_fires_even_when_the_drain_never_completes() {
        let fired = Arc::new(AtomicBool::new(false));
        let fatal = Arc::new(AtomicBool::new(false));

        // Stands in for axum's graceful drain waiting on an in-flight SSE
        // response whose generation is still loading its model.
        let blocked_drain = tokio::spawn(std::future::pending::<()>());

        let observed = fired.clone();
        let deadline =
            super::spawn_shutdown_deadline(Duration::from_secs(45), fatal, true, move |action| {
                assert_eq!(action, super::ShutdownExpiry::Exit(0));
                observed.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .expect("a process that owns its lifecycle arms the deadline");

        tokio::time::advance(Duration::from_secs(46)).await;
        deadline.await.unwrap();

        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "the deadline must not be reachable only after the drain finishes"
        );
        assert!(!blocked_drain.is_finished());
        blocked_drain.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn an_embedded_server_arms_no_deadline() {
        let fatal = Arc::new(AtomicBool::new(false));
        assert!(
            super::spawn_shutdown_deadline(Duration::from_secs(1), fatal, false, |_| panic!(
                "an embedded server must never end its host process"
            ),)
            .is_none()
        );
    }

    /// The standalone server keeps its contract: the drain waits for every
    /// in-flight request, and only the process-ending deadline bounds it.
    #[tokio::test(start_paused = true)]
    async fn an_unbounded_drain_grace_never_elapses() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let _ = started_tx.send(());
        let elapsed = tokio::time::timeout(
            Duration::from_secs(60 * 60),
            super::drain_grace_elapsed(started_rx, None),
        )
        .await;
        assert!(elapsed.is_err(), "an unbounded drain must never give up");
    }

    /// The grace is measured from the moment the drain begins, not from
    /// boot: a server that ran for an hour still gives its in-flight
    /// requests the whole grace once it is asked to stop.
    #[tokio::test(start_paused = true)]
    async fn the_drain_grace_counts_from_the_drain_not_from_boot() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
        let boot = tokio::time::Instant::now();
        let waiter = tokio::spawn(async move {
            super::drain_grace_elapsed(started_rx, Some(Duration::from_secs(3))).await;
            boot.elapsed()
        });
        tokio::time::sleep(Duration::from_secs(60 * 60)).await;
        let _ = started_tx.send(());
        let elapsed = waiter.await.unwrap();
        assert_eq!(elapsed, Duration::from_secs(60 * 60 + 3));
    }

    /// Shutdown must abort the running render, not just stop admitting new
    /// work — waiting for a video generation is structurally unbounded.
    #[test]
    fn runtime_shutdown_aborts_in_flight_generations() {
        let scheduler = tokio_util::sync::CancellationToken::new();
        let journal = Arc::new(crate::queue_journal::QueueJournal::disabled());
        let singletons = Arc::new(crate::generation_cancel::CancelRegistry::new());
        let running = singletons.token("job-1");
        assert!(!running.is_cancelled());

        begin_runtime_shutdown(
            None,
            None,
            &scheduler,
            &journal,
            &singletons,
            &crate::events::EventBroadcaster::new(),
        );

        assert!(running.is_cancelled());
    }

    /// A worker that quarantines itself is initiating the one restart mold
    /// performs on its own behalf. Without this fence that restart is also the
    /// one that deletes the entire queue.
    #[test]
    fn quarantining_a_worker_retains_the_queue() {
        let (mut worker, _rx) = owner_test_worker();
        let journal = Arc::new(crate::queue_journal::QueueJournal::disabled());
        Arc::get_mut(&mut worker).unwrap().queue_journal = journal.clone();
        assert!(!journal.is_retaining());

        gpu_worker::quarantine_poisoned_worker(&worker);

        assert!(journal.is_retaining());
    }

    #[test]
    fn invalid_cors_origin_returns_error() {
        let _lock = crate::test_support::env_lock();
        std::env::set_var("MOLD_CORS_ORIGIN", "\nnot-a-header");
        let result = build_cors_layer();
        std::env::remove_var("MOLD_CORS_ORIGIN");
        assert!(result.is_err());
    }

    #[test]
    fn valid_cors_origin_builds_layer() {
        let _lock = crate::test_support::env_lock();
        std::env::set_var("MOLD_CORS_ORIGIN", "https://example.com");
        let result = build_cors_layer();
        std::env::remove_var("MOLD_CORS_ORIGIN");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn configured_origin_preflight_allows_authenticated_device_patch() {
        let cors = {
            let _lock = crate::test_support::env_lock();
            std::env::set_var("MOLD_CORS_ORIGIN", "https://studio.example");
            let cors = build_cors_layer().unwrap();
            std::env::remove_var("MOLD_CORS_ORIGIN");
            cors
        };
        let auth_state = Some(Arc::new(ApiKeySet::new(HashSet::from([
            "correct-key".to_string()
        ]))));
        let app = Router::new()
            .route(
                "/api/devices/:id",
                patch(|| async { axum::http::StatusCode::OK }),
            )
            .layer(axum::middleware::from_fn(require_api_key))
            .layer(axum::middleware::from_fn_with_state(
                auth_state,
                inject_auth_state,
            ))
            .layer(cors);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/api/devices/cuda:test")
                    .header(header::ORIGIN, "https://studio.example")
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "PATCH")
                    .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "x-api-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(response.status().is_success());
        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&axum::http::HeaderValue::from_static(
                "https://studio.example"
            ))
        );
        let methods = response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_METHODS)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert!(methods.split(',').any(|method| method.trim() == "PATCH"));
        let headers = response
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_HEADERS)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert!(
            headers == "*"
                || headers
                    .split(',')
                    .any(|header_name| header_name.trim().eq_ignore_ascii_case("x-api-key"))
        );

        let missing_key = app
            .clone()
            .oneshot(
                Request::patch("/api/devices/cuda:test")
                    .header(header::ORIGIN, "https://studio.example")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_key.status(), axum::http::StatusCode::UNAUTHORIZED);

        let authenticated = app
            .oneshot(
                Request::patch("/api/devices/cuda:test")
                    .header(header::ORIGIN, "https://studio.example")
                    .header("x-api-key", "correct-key")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authenticated.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn trace_path_omits_gallery_bearer_ticket_query() {
        let request = axum::http::Request::builder()
            .uri("/api/gallery/image/clip.mp4?media_token=secret-ticket&expires=1900")
            .body(())
            .unwrap();

        let path = trace_request_path(&request);
        assert_eq!(path, "/api/gallery/image/clip.mp4");
        assert!(!path.contains("secret-ticket"));
    }

    #[test]
    fn explicit_none_is_maintenance_even_on_a_cuda_host() {
        assert_eq!(
            classify_startup_mode(&GpuSelection::None, 2, 0, true),
            StartupMode::Maintenance
        );
    }

    #[test]
    fn explicit_none_constructs_no_gpu_execution_components() {
        // Fail closed even if a future resolver regression accidentally hands
        // startup a selected device for the explicit maintenance selector.
        let plan = startup_plan(
            &GpuSelection::None,
            2,
            1,
            true,
            crate::dispatch_mode::DispatchMode::V2,
        );

        assert_eq!(plan.mode, StartupMode::Maintenance);
        assert!(!plan.start_gpu_workers);
        assert!(!plan.create_cpu_engine);
        assert!(!plan.start_generation_runner);
        assert!(!plan.start_chain_runner);
        assert!(!plan.start_legacy_cache_evictor);
    }

    #[test]
    fn visible_but_unusable_gpu_inventory_is_not_a_cpu_fallback() {
        assert_eq!(
            classify_startup_mode(&GpuSelection::All, 2, 0, true),
            StartupMode::Maintenance
        );
    }

    #[test]
    fn unusable_gpu_identity_constructs_no_gpu_execution_components() {
        let plan = startup_plan(
            &GpuSelection::All,
            2,
            0,
            true,
            crate::dispatch_mode::DispatchMode::V2,
        );

        assert_eq!(plan.mode, StartupMode::Maintenance);
        assert!(!plan.start_gpu_workers);
        assert!(!plan.create_cpu_engine);
        assert!(!plan.start_generation_runner);
        assert!(!plan.start_chain_runner);
        assert!(!plan.start_legacy_cache_evictor);
    }

    #[test]
    fn only_a_true_cpu_build_uses_the_legacy_cpu_fallback() {
        assert_eq!(
            classify_startup_mode(&GpuSelection::All, 0, 0, false),
            StartupMode::CpuFallback
        );
        assert_eq!(
            classify_startup_mode(&GpuSelection::All, 0, 0, true),
            StartupMode::Maintenance
        );
    }

    #[test]
    fn rollout_mode_selects_exactly_one_gpu_dispatch_owner() {
        use crate::dispatch_mode::DispatchMode;

        let default = startup_plan(
            &GpuSelection::All,
            2,
            2,
            true,
            DispatchMode::from_optional_value(None).unwrap(),
        );
        assert!(default.start_gpu_workers);
        assert!(!default.start_legacy_dispatcher);
        assert!(default.start_v2_coordinator);
        assert!(!default.observe_v2_decisions);

        let legacy = startup_plan(&GpuSelection::All, 2, 2, true, DispatchMode::Legacy);
        assert!(legacy.start_gpu_workers);
        assert!(legacy.start_legacy_dispatcher);
        assert!(!legacy.start_v2_coordinator);
        assert!(!legacy.observe_v2_decisions);

        let observe = startup_plan(&GpuSelection::All, 2, 2, true, DispatchMode::Observe);
        assert!(observe.start_gpu_workers);
        assert!(observe.start_legacy_dispatcher);
        assert!(!observe.start_v2_coordinator);
        assert!(observe.observe_v2_decisions);

        let v2 = startup_plan(&GpuSelection::All, 2, 2, true, DispatchMode::V2);
        assert!(v2.start_gpu_workers);
        assert!(!v2.start_legacy_dispatcher);
        assert!(v2.start_v2_coordinator);
        assert!(!v2.observe_v2_decisions);
    }

    #[test]
    fn maintenance_mode_never_starts_either_dispatch_owner() {
        use crate::dispatch_mode::DispatchMode;

        for dispatch_mode in [
            DispatchMode::Legacy,
            DispatchMode::Observe,
            DispatchMode::V2,
        ] {
            let plan = startup_plan(&GpuSelection::None, 2, 0, true, dispatch_mode);
            assert!(!plan.start_gpu_workers);
            assert!(!plan.start_legacy_dispatcher);
            assert!(!plan.start_v2_coordinator);
            assert!(!plan.observe_v2_decisions);
        }
    }

    fn discovered_gpu(ordinal: usize, stable_id: &str) -> DiscoveredGpu {
        DiscoveredGpu {
            ordinal,
            stable_id: Some(stable_id.to_string()),
            raw_cuda_uuid: Some([ordinal as u8; 16]),
            device_kind: Some(CudaDeviceKind::FullGpu),
            identity_error: None,
            backend: mold_core::GpuBackend::Cuda,
            name: format!("GPU {ordinal}"),
            compute_capability: Some((8, 6)),
            pci_bus_id: None,
            total_vram_bytes: 24 << 30,
            free_vram_bytes: 24 << 30,
        }
    }

    #[test]
    fn persisted_preferences_filter_restart_workers_across_dispatch_modes_and_selectors() {
        use crate::dispatch_mode::DispatchMode;

        const GPU_0: &str = "cuda:00000000000000000000000000000000";
        const GPU_1: &str = "cuda:11111111111111111111111111111111";
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let preferences = mold_db::DevicePreferences::new(&db);
        preferences.set(GPU_1, false).unwrap();
        let db = Arc::new(Some(db));
        let all = vec![discovered_gpu(0, GPU_0), discovered_gpu(1, GPU_1)];
        let registry = DeviceRegistry::from_runtime_inventory(all.clone(), &all, db.clone());

        for dispatch_mode in [
            DispatchMode::Legacy,
            DispatchMode::Observe,
            DispatchMode::V2,
        ] {
            let plan = startup_plan(
                &GpuSelection::All,
                all.len(),
                all.len(),
                true,
                dispatch_mode,
            );
            assert!(plan.start_gpu_workers);
            let worker_ids: BTreeSet<_> = registry
                .startup_worker_devices()
                .into_iter()
                .filter_map(|gpu| gpu.stable_id)
                .collect();
            assert_eq!(worker_ids, BTreeSet::from([GPU_0.to_string()]));
        }

        // Explicit startup allowlists are resolved before preferences. A
        // disabled allowed device gets no owner; an excluded device is never
        // reintroduced by its enabled-by-default preference.
        let only_disabled =
            DeviceRegistry::from_runtime_inventory(all.clone(), &all[1..], db.clone());
        assert!(only_disabled.startup_worker_devices().is_empty());
        let none = DeviceRegistry::from_runtime_inventory(all.clone(), &[], db.clone());
        assert!(none.startup_worker_devices().is_empty());

        registry.set_desired_enabled(GPU_0, false).unwrap();
        assert!(registry.startup_worker_devices().is_empty());

        assert_eq!(
            registry
                .persisted_disabled_worker_devices()
                .iter()
                .map(|gpu| (gpu.stable_id.as_deref().unwrap(), gpu.name.as_str()))
                .collect::<Vec<_>>(),
            vec![(GPU_0, "GPU 0"), (GPU_1, "GPU 1")]
        );

        // V2 retains the complete startup-selected inventory for dynamic
        // re-enable even though neither disabled device owns a worker.
        assert_eq!(
            registry
                .worker_constructions()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([GPU_0, GPU_1])
        );

        // The non-authoritative restart-recovery API writes this same
        // preference. The next Legacy/Observe boot must recreate the owner.
        registry.set_desired_enabled(GPU_0, true).unwrap();
        assert_eq!(
            registry
                .startup_worker_devices()
                .iter()
                .filter_map(|gpu| gpu.stable_id.as_deref())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([GPU_0])
        );
    }

    #[test]
    fn one_gpu_defaults_enabled_without_a_persisted_preference() {
        const GPU_0: &str = "cuda:00000000000000000000000000000000";
        let selected = vec![discovered_gpu(0, GPU_0)];
        let registry = DeviceRegistry::from_runtime_inventory(
            selected.clone(),
            &selected,
            Arc::new(Some(mold_db::MetadataDb::open_in_memory().unwrap())),
        );
        assert_eq!(registry.startup_worker_devices().len(), 1);
    }

    #[test]
    fn v2_factory_input_retains_disabled_selected_devices_not_just_startup_owners() {
        const ENABLED_GPU: &str = "cuda:00000000000000000000000000000000";
        const DISABLED_GPU: &str = "cuda:11111111111111111111111111111111";
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        mold_db::DevicePreferences::new(&db)
            .set(DISABLED_GPU, false)
            .unwrap();
        let selected = vec![
            discovered_gpu(0, ENABLED_GPU),
            discovered_gpu(1, DISABLED_GPU),
        ];
        let registry =
            DeviceRegistry::from_runtime_inventory(selected.clone(), &selected, Arc::new(Some(db)));

        assert_eq!(
            registry
                .startup_worker_devices()
                .iter()
                .filter_map(|gpu| gpu.stable_id.as_deref())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([ENABLED_GPU]),
            "only enabled devices may construct startup owners"
        );
        assert_eq!(
            registry
                .worker_constructions()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([ENABLED_GPU, DISABLED_GPU]),
            "the V2 factory must retain the full startup-selected inventory"
        );
    }

    fn owner_test_worker() -> (Arc<GpuWorker>, std::sync::mpsc::Receiver<GpuWorkerCommand>) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(1);
        (
            Arc::new(GpuWorker {
                cuda_peak: Default::default(),
                #[cfg(test)]
                mock_device_memory: None,
                owner_epoch: 1,
                gpu: DiscoveredGpu {
                    ordinal: 0,
                    stable_id: Some("cuda:00000000000000000000000000000000".to_string()),
                    raw_cuda_uuid: Some([0; 16]),
                    device_kind: Some(CudaDeviceKind::FullGpu),
                    identity_error: None,
                    backend: mold_core::GpuBackend::Cuda,
                    name: "startup-guard-test".to_string(),
                    compute_capability: Some((8, 6)),
                    pci_bus_id: None,
                    total_vram_bytes: 24 << 30,
                    free_vram_bytes: 24 << 30,
                },
                model_cache: Arc::new(Mutex::new(model_cache::ModelCache::new(1))),
                resident_model: Arc::new(RwLock::new(None)),
                resident_execution_fingerprint: Arc::new(RwLock::new(None)),
                active_generation: Arc::new(RwLock::new(None)),
                model_load_lock: Arc::new(Mutex::new(())),
                shared_pool: Arc::new(Mutex::new(SharedPool::new())),
                legacy_pending: AtomicUsize::new(0),
                in_flight: AtomicUsize::new(0),
                legacy_chain_waiters: Default::default(),
                consecutive_failures: AtomicUsize::new(0),
                poisoned: AtomicBool::new(false),
                fatal_cuda_error: Arc::new(AtomicBool::new(false)),
                fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
                queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
                generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
                shutdown_requested: AtomicBool::new(false),
                drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
                owner_thread_id: std::sync::OnceLock::new(),
                degraded_until: RwLock::new(None),
                job_tx,
            }),
            job_rx,
        )
    }

    #[test]
    fn post_owner_startup_error_requests_shutdown_and_joins_owner() {
        let (worker, job_rx) = owner_test_worker();
        let weak_worker = Arc::downgrade(&worker);
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle =
            gpu_worker::spawn_gpu_thread(worker.clone(), job_rx, event_tx, Duration::from_secs(60));
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Ready { .. })
        ));

        let mut owners = GpuOwnerThreads::default();
        owners.track(worker, handle);
        let startup_result: anyhow::Result<()> = (|| {
            let _owners = owners;
            anyhow::bail!("synthetic post-owner startup failure")
        })();

        assert!(startup_result.is_err());
        assert!(matches!(
            event_rx.blocking_recv(),
            Some(crate::scheduler::WorkerEvent::Stopped { owner_epoch: 1, .. })
        ));
        assert!(
            event_rx.blocking_recv().is_none(),
            "startup error must join the owner and drop its scheduler sender"
        );
        assert!(
            weak_worker.upgrade().is_none(),
            "no worker/channel ownership cycle may survive startup cleanup"
        );
    }

    #[test]
    fn owner_join_panic_is_reported() {
        let panicking = std::thread::Builder::new()
            .name("gpu-worker-panics-in-test".to_string())
            .spawn(|| panic!("synthetic owner panic"))
            .unwrap();
        let owners = GpuOwnerThreads {
            workers: Vec::new(),
            handles: vec![panicking],
        };

        let error = owners
            .shutdown_and_join()
            .expect_err("owner panic must be returned to the server owner");
        let message = format!("{error:#}");
        assert!(message.contains("gpu-worker-panics-in-test"));
        assert!(message.contains("synthetic owner panic"));
    }
}
