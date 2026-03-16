use mold_core::Config;
use mold_inference::InferenceEngine;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Box<dyn InferenceEngine>>>,
    pub config: Arc<Config>,
    pub start_time: Instant,
}

impl AppState {
    pub fn new(engine: Box<dyn InferenceEngine>, config: Config) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            config: Arc::new(config),
            start_time: Instant::now(),
        }
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Box::new(engine))),
            config: Arc::new(Config::default()),
            start_time: Instant::now(),
        }
    }
}
