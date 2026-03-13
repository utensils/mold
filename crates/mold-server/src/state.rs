use mold_inference::FluxEngine;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<FluxEngine>>,
    pub start_time: Instant,
}

impl AppState {
    pub fn new(engine: FluxEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            start_time: Instant::now(),
        }
    }
}
