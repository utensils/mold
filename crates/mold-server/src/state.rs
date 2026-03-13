use mold_core::{Config, ModelPaths};
use mold_inference::{FluxEngine, InferenceEngine};
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
    pub fn new(engine: FluxEngine, config: Config) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Box::new(engine))),
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

/// Resolve model paths for hot-swapping models at runtime.
pub fn resolve_paths_for(model_name: &str, config: &Config) -> Option<(ModelPaths, Option<bool>)> {
    let paths = ModelPaths::resolve(model_name, config)?;
    let is_schnell = config.model_config(model_name).is_schnell;
    Some((paths, is_schnell))
}
