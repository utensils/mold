use mold_inference::{FluxEngine, InferenceEngine};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Box<dyn InferenceEngine>>>,
    pub start_time: Instant,
}

impl AppState {
    pub fn new(engine: FluxEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Box::new(engine))),
            start_time: Instant::now(),
        }
    }

    /// For testing: create state with any InferenceEngine implementation.
    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Box::new(engine))),
            start_time: Instant::now(),
        }
    }
}
