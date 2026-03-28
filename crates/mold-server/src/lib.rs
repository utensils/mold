pub mod model_manager;
pub mod queue;
pub mod routes;
pub mod state;

#[cfg(test)]
mod routes_test;

use anyhow::Result;
use mold_core::{Config, ModelPaths};
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use state::QueueHandle;

pub async fn run_server(bind: &str, port: u16, models_dir: PathBuf) -> Result<()> {
    Config::install_runtime_models_dir_override(models_dir.clone());

    let mut config = Config::load_or_default();
    config.models_dir = models_dir.to_string_lossy().into_owned();
    let model_name = config.resolved_default_model();

    // Create the generation queue channel (bounded, 16 slots).
    let (job_tx, job_rx) = tokio::sync::mpsc::channel(16);
    let queue_handle = QueueHandle::new(job_tx);

    let state = match ModelPaths::resolve(&model_name, &config) {
        Some(paths) => {
            info!(model = %model_name, "configured model");
            info!(transformer = %paths.transformer.display());
            info!(vae = %paths.vae.display());
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
            let engine = mold_inference::create_engine(
                model_name,
                paths,
                &config,
                mold_inference::LoadStrategy::Eager,
                offload,
            )?;
            state::AppState::new(engine, config, queue_handle)
        }
        None => {
            info!("no default model configured — models will be pulled on first request");
            state::AppState::empty(config, queue_handle)
        }
    };

    // Spawn the generation queue worker — processes jobs sequentially (single GPU).
    let worker_state = state.clone();
    tokio::spawn(queue::run_queue_worker(job_rx, worker_state));

    let cors = build_cors_layer()?;

    let app = routes::create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    info!(%addr, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

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
                .expose_headers([axum::http::header::HeaderName::from_static(
                    "x-mold-seed-used",
                )])
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
