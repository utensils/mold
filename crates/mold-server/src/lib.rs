pub mod auth;
pub mod catalog_api;
pub mod catalog_credentials;
pub mod chain_job_runner;
pub mod chain_limits;
pub mod test_support;
// Agent A (downloads)
pub mod device_registry;
pub mod downloads;
pub mod events;
pub mod gpu_pool;
pub mod gpu_worker;
pub mod instance;
pub mod job_registry;
pub mod logging;
#[cfg(feature = "mdns")]
pub mod mdns;
mod memory_preflight;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model_cache;
pub mod model_manager;
pub mod queue;
pub mod rate_limit;
pub mod request_id;
pub mod resources;
pub mod routes;
pub mod routes_chain;
pub mod routes_chain_jobs;
pub mod routes_config;
mod signals;
pub mod state;
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

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

fn trace_request_path<B>(request: &axum::http::Request<B>) -> &str {
    // Deliberately omit the query string: gallery media tickets are bearer
    // credentials and must never be copied into request spans or logs.
    request.uri().path()
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

    let mut workers = Vec::new();
    let mut _gpu_thread_handles = Vec::new();

    // Per-worker channel is a tiny buffer: one in-flight plus one immediate
    // handoff. The global queue cap is enforced by `QueueHandle` against
    // `queue_size`; per-worker overflow triggers the dispatcher's cross-worker
    // retry path in `run_queue_dispatcher`.
    const PER_WORKER_CHANNEL_SIZE: usize = 2;

    let max_cached = state::resolve_max_cached_models();
    for gpu in &selected {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(PER_WORKER_CHANNEL_SIZE);
        let worker = std::sync::Arc::new(gpu_pool::GpuWorker {
            gpu: gpu.clone(),
            model_cache: std::sync::Arc::new(std::sync::Mutex::new(model_cache::ModelCache::new(
                max_cached,
            ))),
            resident_model: std::sync::Arc::new(std::sync::RwLock::new(None)),
            active_generation: std::sync::Arc::new(std::sync::RwLock::new(None)),
            model_load_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
            shared_pool: shared_pool.clone(),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: fatal_cuda_error.clone(),
            fatal_cuda_shutdown: fatal_cuda_shutdown.clone(),
            degraded_until: std::sync::RwLock::new(None),
            job_tx,
        });

        let handle = gpu_worker::spawn_gpu_thread(worker.clone(), job_rx);
        _gpu_thread_handles.push(handle);
        workers.push(worker);
    }

    let gpu_pool = std::sync::Arc::new(gpu_pool::GpuPool { workers });

    // Log discovered GPUs.
    for status in gpu_pool.gpu_status() {
        info!(
            gpu = status.ordinal,
            name = %status.name,
            vram_mb = status.vram_total_bytes / 1_000_000,
            "GPU worker ready"
        );
    }

    if selected.is_empty() {
        info!("no GPUs discovered — server will operate in CPU/pull-only mode");
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

    // ── Create AppState ────────────────────────────────────────────────────
    let mut state = if gpu_pool.worker_count() > 0 {
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
        let mut state = state::AppState::empty(config, queue_handle, gpu_pool.clone(), queue_size);
        state.shared_pool = shared_pool;
        state
    } else {
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

                let offload = std::env::var("MOLD_OFFLOAD").is_ok_and(|v| v == "1");
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
                state::AppState::empty(config, queue_handle, gpu_pool.clone(), queue_size)
            }
        }
    };

    // Open the gallery metadata DB (best-effort — server still runs without it).
    match mold_db::open_default() {
        Ok(Some(db)) => {
            info!(db = %db.path().display(), "metadata DB opened");
            state.metadata_db = std::sync::Arc::new(Some(db));
        }
        Ok(None) => {
            tracing::info!("metadata DB disabled (MOLD_DB_DISABLE set or MOLD_HOME unresolved)");
        }
        Err(e) => {
            tracing::warn!(
                "failed to open metadata DB: {e:#} — gallery falls back to filesystem scan"
            );
        }
    }

    // Freeze discovery-owned facts into the read registry. CUDA UUID/type
    // fields are intentionally supplied through this adapter boundary by the
    // UUID-first discovery change; the legacy baseline has no persistable
    // CUDA identity, so such workers remain visible-but-unavailable here
    // rather than inventing an ordinal identity.
    let selected_ordinals: std::collections::BTreeSet<_> =
        selected.iter().map(|gpu| gpu.ordinal).collect();
    let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let inventory = discovered
        .iter()
        .map(|gpu| {
            let telemetry_ordinal = if gpu.backend == mold_core::GpuBackend::Metal {
                Some(gpu.ordinal)
            } else {
                resources::physical_ordinal_for_worker(gpu.ordinal, visible_devices.as_deref())
            };
            device_registry::DiscoveredDevice::from_runtime_gpu(
                gpu,
                selected_ordinals.contains(&gpu.ordinal),
                telemetry_ordinal,
            )
        })
        .collect();
    state.device_registry = std::sync::Arc::new(device_registry::DeviceRegistry::new(
        std::sync::Arc::new(device_registry::StaticDeviceDiscovery::new(inventory)),
        state.metadata_db.clone(),
    ));

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
        let output_dir = if config_snapshot.is_output_disabled() {
            None
        } else {
            Some(config_snapshot.effective_output_dir())
        };
        let deps = chain_job_runner::RunnerDeps {
            db: state.metadata_db.clone(),
            jobs_root,
            executor: std::sync::Arc::new(chain_job_runner::ProductionStageExecutor::new(
                state.gpu_pool.clone(),
                config_snapshot,
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
        };
        state.chain_jobs = Some(std::sync::Arc::new(chain_job_runner::spawn_runner(deps)));
    }

    // Spawn the generation queue worker — processes jobs sequentially (single GPU).
    // Spawn queue worker: use multi-GPU dispatcher if GPUs are available,
    // otherwise fall back to the single-threaded queue worker.
    let worker_state = state.clone();
    if gpu_pool.worker_count() > 0 {
        tokio::spawn(queue::run_queue_dispatcher(job_rx, worker_state));
    } else {
        tokio::spawn(queue::run_queue_worker(job_rx, worker_state));
    }

    // Background idle-TTL sweeper: reclaims parked engines that haven't been
    // touched for `MOLD_CACHE_IDLE_TTL_SECS` seconds. Abort handle bound to
    // graceful shutdown like every other long-running task in this fn.
    let idle_evict_handle = spawn_cache_idle_evictor(
        state.model_cache.clone(),
        state.model_load_lock.clone(),
        gpu_pool.clone(),
        std::time::Duration::from_secs(state::resolve_cache_idle_ttl_secs()),
    );

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

    // Ensure output directory exists and pre-generate thumbnails.
    {
        let config = state.config.read().await;
        if config.is_output_disabled() {
            tracing::warn!(
                "image output is disabled (output_dir is empty) — \
                 generated images will not be saved and the TUI gallery will be empty"
            );
        } else {
            let output_dir = config.effective_output_dir();
            let _ = std::fs::create_dir_all(&output_dir);
            info!(output_dir = %output_dir.display(), "gallery output directory");
            routes::spawn_thumbnail_warmup(&config);

            // Async reconcile: import any existing files into the DB and
            // drop rows whose backing files are missing. Runs on a blocking
            // worker so it never stalls the request path even on large dirs.
            if state.metadata_db.is_some() {
                let db_arc = state.metadata_db.clone();
                let dir = output_dir.clone();
                tokio::spawn(async move {
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
                });
            }
        }
    }

    // Load optional auth and rate-limit configuration from env vars.
    let auth_state = auth::load_api_keys()?;
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

    // Build the router with middleware layers.
    // Order (outermost → innermost): CORS → Trace → RequestID → Metrics → Auth → RateLimit → routes
    // All inject + enforce layers use .layer() (not .route_layer()) so they run on
    // ALL requests, including unmatched 404 paths — preventing auth/rate-limit bypass.
    // Set up graceful shutdown: fires on SIGTERM or POST /api/shutdown.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    *state.shutdown_tx.lock().await = Some(shutdown_tx);

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
    let resources_aggregator = resources::spawn_aggregator(state.resources.clone());

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
            let _ = shutdown_rx.await;
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
    idle_evict_handle.abort();
    // Server has stopped accepting requests — stop the telemetry aggregator
    // so it doesn't outlive the server loop.
    resources_aggregator.abort();

    if fatal_cuda_error.load(std::sync::atomic::Ordering::SeqCst) {
        anyhow::bail!("fatal CUDA context error; server restart required");
    }

    Ok(())
}

/// Spawn a tokio task that wakes every 60s and drops any cache entry whose
/// `last_used` is older than `ttl` (and that isn't actively GPU-resident).
/// Sweeps the legacy single-GPU cache and every per-worker cache in the
/// multi-GPU pool. Returns the `JoinHandle` so the caller can `.abort()` on
/// shutdown.
///
/// After dropping evicted engines, calls `reclaim_gpu_memory` on a thread
/// bound to the relevant GPU ordinal so the freed memory actually returns to
/// the OS rather than sitting in CUDA's per-context caching allocator. Without
/// this step, `nvidia-smi` shows VRAM as still-allocated to the process even
/// after the engine struct is dropped, which is the proximate cause of "model
/// B OOMs even though model A finished and the queue went idle."
fn spawn_cache_idle_evictor(
    legacy_cache: std::sync::Arc<tokio::sync::Mutex<model_cache::ModelCache>>,
    legacy_load_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
    gpu_pool: std::sync::Arc<gpu_pool::GpuPool>,
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
            // Take the legacy load lock for the full eviction+reclaim window
            // so a generation request can't race in between us evicting and
            // reclaiming, which would slot a fresh model load into a context
            // we're about to reset.
            {
                let _load_guard = legacy_load_lock.lock().await;
                let evicted = {
                    let mut cache = legacy_cache.lock().await;
                    cache.evict_idle(ttl)
                };
                let evicted_count = evicted.len();
                // Drop engines OUTSIDE the cache lock — `cuMemFree` and
                // safetensor unmap during drop can block other cache users.
                drop(evicted);

                // Only reclaim when something was evicted (nothing to flush
                // otherwise) and when no GPU-resident engine remains
                // (`cuDevicePrimaryCtxReset` would corrupt it). The load
                // lock above guarantees no concurrent load can sneak one in
                // between this check and the reclaim.
                let legacy_active = legacy_cache.lock().await.active_model().is_some();
                if evicted_count > 0 && !legacy_active {
                    tokio::task::spawn_blocking(|| {
                        mold_inference::reclaim_gpu_memory(0);
                    })
                    .await
                    .ok();
                }
            }

            // ── Multi-GPU per-worker caches ─────────────────────────────────
            //
            // Same pattern but per-worker: hold `worker.model_load_lock` for
            // evict + drop + reclaim. Done under spawn_blocking because the
            // worker locks are std mutexes and the reclaim itself is sync.
            for worker in &gpu_pool.workers {
                let worker = worker.clone();
                let ttl = ttl;
                tokio::task::spawn_blocking(move || {
                    let _load_guard = match worker.model_load_lock.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    let evicted = {
                        let mut cache =
                            worker.model_cache.lock().unwrap_or_else(|e| e.into_inner());
                        cache.evict_idle(ttl)
                    };
                    let evicted_count = evicted.len();
                    drop(evicted);

                    let active = worker
                        .model_cache
                        .lock()
                        .map(|c| c.active_model().is_some())
                        .unwrap_or(true);
                    if evicted_count > 0 && !active {
                        // Bind the thread to the worker's ordinal so
                        // reclaim_gpu_memory's debug-assert is satisfied,
                        // then clear so the spawn_blocking thread (which
                        // returns to the tokio pool) doesn't carry a stale
                        // binding.
                        mold_inference::device::init_thread_gpu_ordinal(worker.gpu.ordinal);
                        mold_inference::reclaim_gpu_memory(worker.gpu.ordinal);
                        mold_inference::device::clear_thread_gpu_ordinal();
                    }
                })
                .await
                .ok();
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
                    axum::http::header::HeaderName::from_static("x-mold-video-has-audio"),
                    axum::http::header::HeaderName::from_static("x-mold-video-duration-ms"),
                    axum::http::header::HeaderName::from_static("x-mold-video-audio-sample-rate"),
                    axum::http::header::HeaderName::from_static("x-mold-video-audio-channels"),
                    axum::http::header::HeaderName::from_static("x-mold-dimension-warning"),
                ])
        }
        _ => CorsLayer::permissive(),
    };
    Ok(cors)
}

#[cfg(test)]
mod tests {
    use super::{build_cors_layer, trace_request_path};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
}
