use mold_core::Config;
use mold_inference::InferenceEngine;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct EngineSnapshot {
    pub model_name: Option<String>,
    pub is_loaded: bool,
}

#[derive(Debug, Clone)]
pub struct ActiveGenerationSnapshot {
    pub model: String,
    pub prompt_sha256: String,
    pub started_at_unix_ms: u64,
    pub started_at: Instant,
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Option<Box<dyn InferenceEngine>>>>,
    pub engine_snapshot: Arc<tokio::sync::RwLock<EngineSnapshot>>,
    /// Uses std::sync::RwLock (not tokio) because it's only accessed from
    /// synchronous contexts (inside spawn_blocking closures and brief reads).
    /// Must never be held across an .await point.
    pub active_generation: Arc<RwLock<Option<ActiveGenerationSnapshot>>>,
    pub config: Arc<tokio::sync::RwLock<Config>>,
    pub start_time: Instant,
    /// Guards concurrent model loads and hot-swaps.
    pub model_load_lock: Arc<Mutex<()>>,
    /// Guards concurrent pulls — only one download at a time.
    pub pull_lock: Arc<Mutex<()>>,
}

impl AppState {
    /// Create state with a pre-loaded engine (server starts with a configured model).
    pub fn new(engine: Box<dyn InferenceEngine>, config: Config) -> Self {
        let snapshot = EngineSnapshot {
            model_name: Some(engine.model_name().to_string()),
            is_loaded: engine.is_loaded(),
        };
        Self {
            engine: Arc::new(Mutex::new(Some(engine))),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(snapshot)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Create state with no engine (zero-config startup, models pulled on demand).
    pub fn empty(config: Config) -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(EngineSnapshot::default())),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(config)),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }

    #[cfg(test)]
    pub fn with_engine(engine: impl InferenceEngine + 'static) -> Self {
        let snapshot = EngineSnapshot {
            model_name: Some(engine.model_name().to_string()),
            is_loaded: engine.is_loaded(),
        };
        Self {
            engine: Arc::new(Mutex::new(Some(Box::new(engine)))),
            engine_snapshot: Arc::new(tokio::sync::RwLock::new(snapshot)),
            active_generation: Arc::new(RwLock::new(None)),
            config: Arc::new(tokio::sync::RwLock::new(Config::default())),
            start_time: Instant::now(),
            model_load_lock: Arc::new(Mutex::new(())),
            pull_lock: Arc::new(Mutex::new(())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_snapshot_default_is_unloaded() {
        let snap = EngineSnapshot::default();
        assert!(snap.model_name.is_none());
        assert!(!snap.is_loaded);
    }

    #[test]
    fn active_generation_snapshot_stores_fields() {
        let snap = ActiveGenerationSnapshot {
            model: "flux-dev:q8".to_string(),
            prompt_sha256: "abc123".to_string(),
            started_at_unix_ms: 1700000000000,
            started_at: std::time::Instant::now(),
        };
        assert_eq!(snap.model, "flux-dev:q8");
        assert_eq!(snap.prompt_sha256, "abc123");
        assert_eq!(snap.started_at_unix_ms, 1700000000000);
    }
}
