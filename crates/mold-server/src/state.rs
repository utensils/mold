use mold_core::Config;
use mold_inference::InferenceEngine;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Option<Box<dyn InferenceEngine>>>>,
    pub config: Arc<tokio::sync::RwLock<Config>>,
    pub start_time: Instant,
    /// Guards concurrent pulls — only one download at a time.
    pub pull_lock: Arc<Mutex<()>>,
}

impl AppState {
    /// Create state with a pre-loaded engine (server starts with a configured model).
    pub fn new(engine: Box<dyn InferenceEngine>, config: Config) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Some(engine))),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Create state with no engine (zero-config startup, models pulled on demand).
    pub fn empty(config: Config) -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        Self {
            engine: Arc::new(Mutex::new(Some(Box::new(engine)))),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }
}
