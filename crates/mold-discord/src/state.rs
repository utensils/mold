use crate::cooldown::CooldownTracker;
use mold_core::{ModelInfoExtended, MoldClient};
use std::time::Instant;
use tokio::sync::RwLock;

/// Bot-wide configuration loaded from environment variables.
#[derive(Debug)]
pub struct BotConfig {
    /// Per-user cooldown between generation requests, in seconds.
    pub cooldown_seconds: u64,
}

/// Shared state accessible to all poise commands.
pub struct BotState {
    pub client: MoldClient,
    pub config: BotConfig,
    pub cooldowns: CooldownTracker,
    pub model_cache: RwLock<(Instant, Vec<ModelInfoExtended>)>,
}

impl std::fmt::Debug for BotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BotState")
            .field("config", &self.config)
            .field("host", &self.client.host())
            .finish()
    }
}

/// poise context alias.
pub type Context<'a> = poise::Context<'a, BotState, anyhow::Error>;

impl BotState {
    pub fn new(client: MoldClient, config: BotConfig) -> Self {
        Self {
            client,
            config,
            cooldowns: CooldownTracker::new(),
            model_cache: RwLock::new((Instant::now() - std::time::Duration::from_secs(60), vec![])),
        }
    }

    /// Get cached models list, refreshing if older than 30 seconds.
    pub async fn cached_models(&self) -> Vec<ModelInfoExtended> {
        let cache_ttl = std::time::Duration::from_secs(30);

        {
            let cache = self.model_cache.read().await;
            if cache.0.elapsed() < cache_ttl && !cache.1.is_empty() {
                return cache.1.clone();
            }
        }

        // Cache miss — fetch from server
        match self.client.list_models_extended().await {
            Ok(models) => {
                let mut cache = self.model_cache.write().await;
                *cache = (Instant::now(), models.clone());
                models
            }
            Err(_) => {
                // Return stale cache on error
                let cache = self.model_cache.read().await;
                cache.1.clone()
            }
        }
    }
}
