pub mod auth;
pub mod batch_attempt;
pub mod batch_parent;
pub mod batch_runtime;
pub mod batch_transaction;
pub mod catalog_api;
pub mod catalog_credentials;
pub(crate) mod chain_execution;
pub mod chain_job_runner;
pub mod chain_limits;
mod gallery_authority;
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
pub mod test_support;
// Agent A (downloads)
pub mod device_registry;
pub mod dispatch_mode;
pub mod downloads;
pub mod events;
pub mod execution_plan;
pub mod gpu_pool;
pub mod gpu_worker;
pub mod instance;
pub mod job_registry;
pub mod logging;
mod ltx2_admission;
#[cfg(feature = "mdns")]
pub mod mdns;
mod memory_preflight;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model_cache;
pub mod model_manager;
pub mod queue;
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
pub mod variant_dependencies;
mod wan_admission;
pub mod web_ui;

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
    state.device_registry = device_registry;

    let mut recovered_live_batches = None;
    // Batch recovery is a serving precondition, including when SQLite is
    // disabled. No router, gallery observer, or generation job producer is started
    // until every durable attempt is either rolled back or rolled forward.
    {
        let config = state.config.read().await;
        if !state.is_output_disabled(&config) {
            let output_dir = config.effective_output_dir();
            drop(config);
            std::fs::create_dir_all(&output_dir)?;
            let report = batch_attempt::recover_batches(
                &output_dir,
                &state.gallery_publication_gate,
                state.metadata_db.clone(),
            )
            .await?;
            if report.outcomes.iter().any(|outcome| {
                matches!(
                    outcome,
                    batch_attempt::RecoveredParentOutcome::Resumable { .. }
                )
            }) {
                recovered_live_batches = Some((output_dir.clone(), report.outcomes.clone()));
            }
            tracing::info!(
                joint_parents = report.parents_discovered,
                joint_running = report.running_attempts_retained,
                joint_rolled_forward = report.parents_rolled_forward,
                receipts_removed = report.receipts_removed,
                leases_requeued = report.leases_requeued,
                rolled_back = report.legacy_transactions.rolled_back,
                rolled_forward = report.legacy_transactions.rolled_forward,
                healed_committed_rows = report.legacy_transactions.healed_committed_rows,
                "batch transaction startup recovery complete"
            );
        }
    }

    // Resolve the persistent instance id (ephemeral when the DB is
    // unavailable). Scoped per (data dir, port) so two servers sharing one
    // mold.db report distinct identities; the configured port is used, so an
    // ephemeral `--port 0` server shares the `.0` slot with other `--port 0`
    // runs on the same DB — its address changes every run anyway. Captured
    // for mDNS here because `state` is moved into the router before the TXT
    // records are built.
    state.instance_id = std::sync::Arc::new(instance::resolve_instance_id(
        state.metadata_db.as_ref().as_ref(),
        port,
    ));
    #[cfg(feature = "mdns")]
    let mdns_instance_id = state.instance_id.clone();

    if state.metadata_db.is_some() {
        let Some(jobs_root) = Config::mold_dir().map(|dir| dir.join("jobs")) else {
            anyhow::bail!("metadata DB opened but MOLD_HOME could not be resolved for chain jobs");
        };
        std::fs::create_dir_all(&jobs_root)?;
        let db_arc = state.metadata_db.clone();
        let reconcile_root = jobs_root.clone();
        let (flipped, repaired) = tokio::task::spawn_blocking(move || {
            let Some(db) = db_arc.as_ref().as_ref() else {
                anyhow::bail!("metadata DB disappeared before chain reconcile");
            };
            chain_job_runner::startup_reconcile(db, &reconcile_root)
        })
        .await??;
        tracing::info!(
            flipped,
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

    // Recovered live parents are resumed only after their authoritative
    // dispatch owner is running, but still before the router begins serving.
    // Exact execution-equivalence drift fails closed into a durable cancelled
    // parent and private-attempt rollback.
    if let Some((output_dir, outcomes)) = recovered_live_batches {
        let report =
            batch_runtime::resume_recovered_batches(&state, &output_dir, &outcomes).await?;
        tracing::info!(
            resumed = report.resumed,
            committed = report.committed,
            rolled_back = report.rolled_back,
            "live batch restart recovery settled"
        );
    }

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
                            "metadata DB reconciled with gallery directory"
                        ),
                        Ok(Err(e)) => tracing::warn!("metadata DB reconcile failed: {e:#}"),
                        Err(e) => tracing::warn!("metadata DB reconcile task join error: {e}"),
                    }
                }));
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
    tokio::spawn(async move {
        let _ = shutdown_request_rx.await;
        begin_runtime_shutdown(shutdown_chain_jobs.as_deref(), &shutdown_scheduler);
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

    let server = std::future::IntoFuture::into_future(
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async move {
            let _ = http_shutdown_rx.await;
            tracing::info!("shutting down");
        }),
    );
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        _ = fatal_cuda_shutdown.notified() => {
            tracing::error!("fatal CUDA context error; stopping server for process restart");
            // Give the triggering request a brief window to receive its explicit
            // fatal-context error, then drop the server future. A normal graceful
            // shutdown could wait forever on queued SSE requests assigned to the
            // now-quarantined worker and prevent the service manager from restarting.
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }

    // Server has stopped accepting requests — cancel the downloads token so the
    // driver's `wait_for_work` arm returns, then abort the JoinHandle to ensure
    // the task is cleaned up on the same shutdown path as the HTTP server.
    // Matches the aggregator handle pattern from commit 5e43886.
    #[cfg(feature = "mdns")]
    if let Some(guard) = mdns_guard {
        guard.shutdown();
    }
    #[cfg(feature = "mdns")]
    if let Some(guard) = mdns_browser_guard {
        guard.shutdown();
    }
    downloads_shutdown.cancel();
    downloads_driver.abort();
    let _ = downloads_driver.await;
    if let Some(handle) = idle_evict_handle {
        handle.abort();
        let _ = handle.await;
    }
    // Server has stopped accepting requests — stop the telemetry aggregator
    // so it doesn't outlive the server loop.
    resources_aggregator.abort();
    let _ = resources_aggregator.await;

    // These tasks may be inside spawn_blocking and therefore cannot be safely
    // detached or cancelled. Await their actual completion before returning
    // server authority to an in-process lifecycle owner.
    if let Some(handle) = thumbnail_warmup_handle {
        let _ = handle.await;
    }
    if let Some(handle) = gallery_reconcile_handle {
        let _ = handle.await;
    }
    scheduler_shutdown.cancel();
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
    let shutdown_pool = gpu_pool.clone();
    tokio::task::spawn_blocking(move || {
        shutdown_pool.workers.shutdown_and_join_all();
        gpu_owner_threads.shutdown_and_join()
    })
    .await
    .map_err(|error| anyhow::anyhow!("failed to run GPU owner join task: {error}"))??;

    if fatal_cuda_error.load(std::sync::atomic::Ordering::SeqCst) {
        anyhow::bail!("fatal CUDA context error; server restart required");
    }

    Ok(())
}

fn begin_runtime_shutdown(
    chain_jobs: Option<&chain_job_runner::ChainJobRunnerHandle>,
    scheduler_shutdown: &tokio_util::sync::CancellationToken,
) {
    if let Some(chain_jobs) = chain_jobs {
        let active_chains = chain_jobs.request_shutdown();
        tracing::info!(active_chains, "cancelled chain work before HTTP drain");
    }
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

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn runtime_shutdown_cancels_chains_and_scheduler_before_http_drain() {
        let chains = crate::chain_job_runner::ChainJobRunnerHandle::inert_for_tests();
        chains.register_cancel_for_tests("chain-in-flight");
        let scheduler = tokio_util::sync::CancellationToken::new();

        begin_runtime_shutdown(Some(&chains), &scheduler);

        assert!(chains.is_cancelling("chain-in-flight"));
        chains.register_cancel_for_tests("chain-claimed-during-shutdown");
        assert!(chains.is_cancelling("chain-claimed-during-shutdown"));
        assert!(scheduler.is_cancelled());
    }

    #[test]
    fn invalid_cors_origin_returns_error() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_CORS_ORIGIN", "\nnot-a-header");
        let result = build_cors_layer();
        std::env::remove_var("MOLD_CORS_ORIGIN");
        assert!(result.is_err());
    }

    #[test]
    fn valid_cors_origin_builds_layer() {
        let _lock = ENV_LOCK.lock().unwrap();
        std::env::set_var("MOLD_CORS_ORIGIN", "https://example.com");
        let result = build_cors_layer();
        std::env::remove_var("MOLD_CORS_ORIGIN");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn configured_origin_preflight_allows_authenticated_device_patch() {
        let cors = {
            let _lock = ENV_LOCK.lock().unwrap();
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
