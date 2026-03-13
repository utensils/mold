pub mod routes;
pub mod state;

#[cfg(test)]
mod routes_test;

use anyhow::Result;
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
            "no model paths configured for '{}'. Add [models.{}] to ~/.mold/config.toml \
             or set MOLD_TRANSFORMER_PATH / MOLD_VAE_PATH / MOLD_T5_PATH / MOLD_CLIP_PATH \
             / MOLD_T5_TOKENIZER_PATH / MOLD_CLIP_TOKENIZER_PATH env vars.",
            model_name,
            model_name,
        )
    })?;

    let model_cfg = config.model_config(&model_name);
    let is_schnell_override = model_cfg.is_schnell;

    info!(model = %model_name, "configured model");
    info!(transformer = %paths.transformer.display());
    info!(vae = %paths.vae.display());
    info!(t5 = %paths.t5_encoder.display());
    info!(clip = %paths.clip_encoder.display());
    info!(t5_tok = %paths.t5_tokenizer.display());
    info!(clip_tok = %paths.clip_tokenizer.display());

    let engine = FluxEngine::new(model_name, paths, is_schnell_override);
    let state = state::AppState::new(engine, config);
    let app = routes::create_router(state).layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    info!(%addr, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
