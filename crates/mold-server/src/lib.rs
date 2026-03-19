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

pub async fn run_server(bind: &str, port: u16, models_dir: PathBuf) -> Result<()> {
    let mut config = Config::load_or_default();
    config.models_dir = models_dir.to_string_lossy().into_owned();
    let model_name = config.default_model.clone();

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

            let engine = mold_inference::create_engine(
                model_name,
                paths,
                &config,
                mold_inference::LoadStrategy::Eager,
            )?;
            state::AppState::new(engine, config)
        }
        None => {
            info!("no default model configured — models will be pulled on first request");
            state::AppState::empty(config)
        }
    };

    let app = routes::create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    info!(%addr, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
