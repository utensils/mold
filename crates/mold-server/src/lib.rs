pub mod auth;
pub mod chain_limits;
// Agent A (downloads)
pub mod downloads;
pub mod gpu_pool;
pub mod gpu_worker;
pub mod logging;
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
use std::sync::atomic::AtomicUsize;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use state::QueueHandle;

const MAX_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

pub async fn run_server(
    bind: &str,
    port: u16,
    models_dir: PathBuf,
    gpu_selection: GpuSelection,
    queue_size: usize,
) -> Result<()> {
    Config::install_runtime_models_dir_override(models_dir.clone());

    let mut config = Config::load_or_default();
    config.models_dir = models_dir.to_string_lossy().into_owned();
    let model_name = config.resolved_default_model();

    // ── Discover and initialize GPU workers ────────────────────────────────
    let shared_pool = std::sync::Arc::new(std::sync::Mutex::new(
        mold_inference::shared_pool::SharedPool::new(),
    ));

    let discovered = mold_inference::device::discover_gpus();
    let selected = mold_inference::device::filter_gpus(&discovered, &gpu_selection);

    if selected.is_empty() && !discovered.is_empty() {
        anyhow::bail!(
            "No GPUs matched selection {:?} (discovered: {:?})",
            gpu_selection,
            discovered.iter().map(|g| g.ordinal).collect::<Vec<_>>()
        );
    }

    let mut workers = Vec::new();
    let mut _gpu_thread_handles = Vec::new();

    // Per-worker channel is a tiny buffer: one in-flight plus one immediate
    // handoff. The global queue cap is enforced by `QueueHandle` against
    // `queue_size`; per-worker overflow triggers the dispatcher's cross-worker
    // retry path in `run_queue_dispatcher`.
    const PER_WORKER_CHANNEL_SIZE: usize = 2;

    for gpu in &selected {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(PER_WORKER_CHANNEL_SIZE);
        let worker = std::sync::Arc::new(gpu_pool::GpuWorker {
            gpu: gpu.clone(),
            model_cache: std::sync::Arc::new(std::sync::Mutex::new(model_cache::ModelCache::new(
                3,
            ))),
            active_generation: std::sync::Arc::new(std::sync::RwLock::new(None)),
            model_load_lock: std::sync::Arc::new(std::sync::Mutex::new(())),
            shared_pool: shared_pool.clone(),
            in_flight: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
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

    // Spawn the generation queue worker — processes jobs sequentially (single GPU).
    // Spawn queue worker: use multi-GPU dispatcher if GPUs are available,
    // otherwise fall back to the single-threaded queue worker.
    let worker_state = state.clone();
    if gpu_pool.worker_count() > 0 {
        tokio::spawn(queue::run_queue_dispatcher(job_rx, worker_state));
    } else {
        tokio::spawn(queue::run_queue_worker(job_rx, worker_state));
    }

    // ── Downloads UI (Agent A) ──────────────────────────────────────────────
    // Single-writer download queue driver. Bind the `JoinHandle` so we can
    // `.abort()` it when `axum::serve` returns — same pattern as the resource
    // telemetry aggregator (see commit 5e43886). Without this the task would
    // outlive graceful shutdown and keep polling its cancellation token until
    // process exit.
    let downloads_shutdown = tokio_util::sync::CancellationToken::new();
    let downloads_driver = crate::downloads::spawn_driver(
        state.downloads.clone(),
        std::sync::Arc::new(crate::downloads::HfPullDriver),
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
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    let version = mold_core::build_info::version_string();
    info!(%addr, %version, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = shutdown_rx.await;
        tracing::info!("shutting down");
    })
    .await?;

    // Server has stopped accepting requests — cancel the downloads token so the
    // driver's `wait_for_work` arm returns, then abort the JoinHandle to ensure
    // the task is cleaned up on the same shutdown path as the HTTP server.
    // Matches the aggregator handle pattern from commit 5e43886.
    downloads_shutdown.cancel();
    downloads_driver.abort();
    // Server has stopped accepting requests — stop the telemetry aggregator
    // so it doesn't outlive the server loop.
    resources_aggregator.abort();

    Ok(())
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
    use super::build_cors_layer;
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
}
