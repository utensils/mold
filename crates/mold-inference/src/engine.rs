use anyhow::Result;
use mold_core::GenerateRequest;
use mold_core::GenerateResponse;

use crate::progress::ProgressCallback;

/// Controls how model components are loaded during inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Load all components at once, keep hot (server mode).
    #[default]
    Eager,
    /// Load-use-drop per component, minimizing peak memory (CLI one-shot mode).
    Sequential,
}

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    /// Load model weights. Called automatically on first generate if not yet loaded.
    fn load(&mut self) -> Result<()>;
    /// Unload model weights to free GPU memory. The engine remains valid and
    /// can be re-loaded by calling `load()` or generating again.
    fn unload(&mut self) {}
    /// Set a progress callback for receiving loading/inference status updates.
    /// Default implementation is a no-op for engines that don't support progress.
    fn set_on_progress(&mut self, _callback: ProgressCallback) {}
}

/// Generate a random seed from the current system time.
pub(crate) fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Generate deterministic noise on a device with a given seed.
///
/// This is the ONLY correct way to generate initial noise for denoising.
/// It sets the device seed immediately before calling `Tensor::randn`,
/// guaranteeing that the same seed always produces the same noise regardless
/// of what GPU operations (text encoding, model loading) happened before.
///
/// All pipelines MUST use this instead of calling `device.set_seed()` +
/// `Tensor::randn()` separately, which is fragile because intervening GPU
/// ops can consume RNG state.
///
/// On CPU, `set_seed` is skipped (candle's CPU backend doesn't support it).
/// In practice all inference runs on GPU so this path is rarely hit.
pub(crate) fn seeded_randn(
    seed: u64,
    shape: &[usize],
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<candle_core::Tensor> {
    // Generate noise on the target device.
    // For GPU: set_seed ensures deterministic randn.
    // For CPU: candle doesn't support set_seed, but we still get consistent
    // noise if no other threads are generating random numbers (single-threaded CLI).
    if !device.is_cpu() {
        device.set_seed(seed)?;
    }
    Ok(candle_core::Tensor::randn(0f32, 1.0, shape, device)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_randn_produces_correct_shape() {
        let dev = candle_core::Device::Cpu;
        let t = seeded_randn(42, &[1, 4, 8, 8], &dev, candle_core::DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 4, 8, 8]);
    }

    #[test]
    fn seeded_randn_respects_dtype() {
        let dev = candle_core::Device::Cpu;
        let t = seeded_randn(42, &[2, 2], &dev, candle_core::DType::BF16).unwrap();
        assert_eq!(t.dtype(), candle_core::DType::BF16);
    }
}
