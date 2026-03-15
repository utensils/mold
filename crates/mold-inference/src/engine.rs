use anyhow::Result;
use mold_core::GenerateRequest;
use mold_core::GenerateResponse;

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    /// Load model weights. Called automatically on first generate if not yet loaded.
    fn load(&mut self) -> Result<()>;
}

/// Generate a random seed from the current system time.
pub(crate) fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
