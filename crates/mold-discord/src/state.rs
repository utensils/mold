use crate::access::{AllowedRoles, BlockList};
use crate::cooldown::CooldownTracker;
use crate::quota::QuotaTracker;
use mold_core::{ModelInfoExtended, MoldClient};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Bot-wide configuration loaded from environment variables.
#[derive(Debug)]
pub struct BotConfig {
    /// Per-user cooldown between generation requests, in seconds.
    pub cooldown_seconds: u64,
    /// Role-based access control. When unrestricted, all users can generate.
    pub allowed_roles: AllowedRoles,
    /// Maximum generations per user per UTC day. `None` = unlimited.
    pub daily_quota: Option<u32>,
}

/// Shared state accessible to all poise commands.
pub struct BotState {
    pub client: MoldClient,
    pub config: BotConfig,
    pub cooldowns: CooldownTracker,
    pub quotas: QuotaTracker,
    pub block_list: BlockList,
    /// Server-reported model list + the `Instant` it was last refreshed.
    /// A background task refreshes this every `MODEL_CACHE_TTL`; callers
    /// never trigger a fetch on the hot path, so Discord's 3-second
    /// autocomplete budget is never at the mercy of server latency.
    pub model_cache: Arc<RwLock<(Instant, Vec<ModelInfoExtended>)>>,
}

/// Background-refresh interval for `model_cache`. Kept well under Discord's
/// 3-second autocomplete timeout so even a cold fetch never stalls users.
pub const MODEL_CACHE_TTL: Duration = Duration::from_secs(30);

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
            quotas: QuotaTracker::new(),
            block_list: BlockList::new(),
            model_cache: Arc::new(RwLock::new((
                Instant::now() - Duration::from_secs(60),
                vec![],
            ))),
        }
    }

    /// Read the current model cache without blocking on the network. Returns
    /// whatever is there — possibly empty when the bot just started and the
    /// background refresher hasn't completed its first fetch yet. Callers
    /// that need a fallback (e.g. autocomplete) should substitute
    /// `mold_core::manifest::visible_manifests()` in that case.
    pub async fn cached_models(&self) -> Vec<ModelInfoExtended> {
        self.model_cache.read().await.1.clone()
    }

    /// Force-refresh the model cache. Used by the background refresher and
    /// by commands like `/models` that can afford to pay a round-trip.
    pub async fn refresh_models(&self) -> Result<(), mold_core::MoldError> {
        let models = self.client.list_models_extended().await?;
        let mut cache = self.model_cache.write().await;
        *cache = (Instant::now(), models);
        Ok(())
    }

    /// Spawn a background task that refreshes `model_cache` every
    /// `MODEL_CACHE_TTL`. The first refresh runs immediately so the very
    /// first `/generate` invocation after startup already has data.
    pub fn spawn_model_cache_refresher(
        cache: Arc<RwLock<(Instant, Vec<ModelInfoExtended>)>>,
        client: MoldClient,
    ) {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(MODEL_CACHE_TTL);
            // First tick fires immediately; subsequent ticks wait the full interval.
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                match client.list_models_extended().await {
                    Ok(models) => {
                        let mut guard = cache.write().await;
                        *guard = (Instant::now(), models);
                        debug!("model cache refreshed ({} entries)", guard.1.len());
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            "model cache refresh failed; keeping previous entries",
                        );
                    }
                }
            }
        });
    }
}
