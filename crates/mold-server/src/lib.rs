pub mod routes;
pub mod state;

use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;

pub async fn run_server(bind: &str, port: u16, models_dir: PathBuf) -> Result<()> {
    let state = state::AppState::new(models_dir);
    let app = routes::create_router(state).layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    info!(%addr, "starting mold server");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
