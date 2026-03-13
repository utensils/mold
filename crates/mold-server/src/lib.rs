pub mod routes;
pub mod state;

#[cfg(test)]
mod routes_test;

use anyhow::{bail, Result};
use mold_core::{Config, ModelPaths};
use mold_inference::FluxEngine;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;

pub async fn run_server(bind: &str, port: u16, _models_dir: PathBuf) -> Result<()> {
    let config = Config::load_or_default();
    let model_name = config.default_model.clone();

    let paths = ModelPaths::resolve(&model_name, &config).ok_or_else(|| {
        anyhow::anyhow!(
            "no model paths configured for '{}'. Set MOLD_TRANSFORMER_PATH, MOLD_VAE_PATH, \
             MOLD_T5_PATH, MOLD_CLIP_PATH env vars or add [models.{}] to config.",
            model_name,
            model_name,
        )
    })?;

    // Resolve tokenizer paths from env vars
    let t5_tokenizer = resolve_tokenizer_path("MOLD_T5_TOKENIZER_PATH")?;
    let clip_tokenizer = resolve_tokenizer_path("MOLD_CLIP_TOKENIZER_PATH")?;

    info!(model = %model_name, "configured model paths");
    info!(transformer = %paths.transformer.display());
    info!(vae = %paths.vae.display());
    info!(t5 = %paths.t5_encoder.display());
    info!(clip = %paths.clip_encoder.display());

    let engine = FluxEngine::new(model_name, paths, t5_tokenizer, clip_tokenizer);
    let state = state::AppState::new(engine);
    let app = routes::create_router(state).layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    info!(%addr, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn resolve_tokenizer_path(env_var: &str) -> Result<PathBuf> {
    match std::env::var(env_var) {
        Ok(path) => Ok(PathBuf::from(path)),
        Err(_) => bail!("{} not set. Point it at the tokenizer.json file.", env_var),
    }
}
