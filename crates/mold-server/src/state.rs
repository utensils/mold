use mold_inference::FluxEngine;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<FluxEngine>>,
    pub start_time: Instant,
    pub models_dir: PathBuf,
}

impl AppState {
    pub fn new(models_dir: PathBuf) -> Self {
        Self {
            engine: Arc::new(Mutex::new(FluxEngine::new(models_dir.clone()))),
            start_time: Instant::now(),
            models_dir,
        }
    }
}
