use anyhow::Result;
use mold_core::GenerateRequest;
use mold_core::GenerateResponse;
use std::ops::{Deref, DerefMut};

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
    /// Clear any previously installed progress callback.
    fn clear_on_progress(&mut self) {}
    /// Return the model's resolved file paths, if available.
    /// Used by the server for pre-load memory checks on unified-memory systems.
    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        None
    }

    /// Returns a [`ChainStageRenderer`] view of this engine if the family
    /// supports chained video generation. Default is `None` — only LTX-2
    /// distilled overrides this in v1.
    ///
    /// Callers (the server chain route) invoke this once per stage to drive
    /// [`crate::chain::ChainOrchestrator::run`]; engines that don't support
    /// chaining return `None` and the caller responds with 422.
    fn as_chain_renderer(&mut self) -> Option<&mut dyn crate::chain::ChainStageRenderer> {
        None
    }
}

/// Restores an `Option<T>` slot even if the current scope unwinds.
pub(crate) struct OptionRestoreGuard<'a, T> {
    slot: &'a mut Option<T>,
    value: Option<T>,
}

impl<'a, T> OptionRestoreGuard<'a, T> {
    pub(crate) fn take(slot: &'a mut Option<T>) -> Option<Self> {
        let value = slot.take()?;
        Some(Self {
            slot,
            value: Some(value),
        })
    }
}

impl<T> Deref for OptionRestoreGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("option restore guard must hold a value")
    }
}

impl<T> DerefMut for OptionRestoreGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
            .as_mut()
            .expect("option restore guard must hold a value")
    }
}

impl<T> Drop for OptionRestoreGuard<'_, T> {
    fn drop(&mut self) {
        *self.slot = self.value.take();
    }
}

/// Select the optimal dtype for GPU inference.
///
/// Re-exported from `device::gpu_dtype` for backward compatibility.
pub(crate) fn gpu_dtype(device: &candle_core::Device) -> candle_core::DType {
    crate::device::gpu_dtype(device)
}

/// Generate a random seed from the current system time.
pub(crate) fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Tolerance for treating a CFG (classifier-free guidance) scale as "1.0,
/// disabled". When the active guidance is within this epsilon of 1.0 the
/// uncond pass adds nothing — `cond + (cond - uncond) * 0 == cond` — so the
/// pipeline can run a single conditional forward instead of batching
/// `[uncond, cond]`. Used by LCM / Lightning / Turbo (guidance-distilled)
/// workflows that ship with `cfg ≈ 1.0`.
///
/// Matches ComfyUI's short-circuit at `comfy/samplers.py:370`
/// (`if math.isclose(cond_scale, 1.0): uncond_ = None`). The default
/// `math.isclose` rel-tol is `1e-9` — ours is looser (`1e-4`) because the
/// caller-visible knob is a user-typed `f64` like `1.0` or `1.0000`.
pub(crate) const CFG_DISABLE_EPSILON: f64 = 1e-4;

/// Returns `true` when classifier-free guidance is active for the given scale,
/// i.e. when the unconditional forward pass meaningfully contributes to the
/// final noise prediction. When `false`, callers should run a single
/// conditional forward (saves ~2× denoise time).
pub(crate) fn cfg_active(guidance: f64) -> bool {
    (guidance - 1.0).abs() > CFG_DISABLE_EPSILON
}

/// Resolve the effective `cfg_plus` flag for a request.
///
/// Precedence: explicit request field > `MOLD_CFG_PLUS` env var > false.
/// Mirrors `MOLD_OFFLOAD` / `MOLD_KEEP_TE_RAM`. Shared across SD3, SDXL,
/// and SD1.5 so the wire-format and env contract stay identical.
pub(crate) fn resolve_cfg_plus(req: &GenerateRequest) -> bool {
    if let Some(explicit) = req.cfg_plus {
        return explicit;
    }
    matches!(
        std::env::var("MOLD_CFG_PLUS").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Generate deterministic noise on a device with a given seed.
///
/// This is the ONLY correct way to generate initial noise for denoising.
/// All pipelines MUST use this instead of calling `device.set_seed()` +
/// `Tensor::randn()` separately.
///
/// Noise is generated on CPU using a deterministic Rust RNG, then moved to
/// the target device. This guarantees:
/// 1. Same seed always produces identical noise (deterministic)
/// 2. Same seed produces the same noise across CUDA, Metal, and CPU backends
///    (cross-platform reproducibility)
///
/// GPU-native RNG (Metal's HybridTaus, CUDA's cuRAND) use different algorithms
/// that produce different sequences from the same seed. CPU generation avoids this.
pub(crate) fn seeded_randn(
    seed: u64,
    shape: &[usize],
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<candle_core::Tensor> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    // Generate noise on CPU with a deterministic RNG for cross-platform reproducibility.
    let mut rng = StdRng::seed_from_u64(seed);
    let elem_count: usize = shape.iter().product();
    let noise: Vec<f32> = (0..elem_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();

    let tensor = candle_core::Tensor::from_vec(noise, shape, &candle_core::Device::Cpu)?;
    Ok(tensor.to_dtype(dtype)?.to_device(device)?)
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

    #[test]
    fn seeded_randn_deterministic_same_seed() {
        let dev = candle_core::Device::Cpu;
        let a = seeded_randn(1337, &[1, 16, 8, 8], &dev, candle_core::DType::F32).unwrap();
        let b = seeded_randn(1337, &[1, 16, 8, 8], &dev, candle_core::DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "same seed must produce identical noise");
    }

    #[test]
    fn seeded_randn_different_seeds_differ() {
        let dev = candle_core::Device::Cpu;
        let a = seeded_randn(42, &[1, 4, 8, 8], &dev, candle_core::DType::F32).unwrap();
        let b = seeded_randn(43, &[1, 4, 8, 8], &dev, candle_core::DType::F32).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0, "different seeds must produce different noise");
    }

    #[test]
    fn gpu_dtype_cpu_returns_f32() {
        assert_eq!(
            gpu_dtype(&candle_core::Device::Cpu),
            candle_core::DType::F32
        );
    }

    #[test]
    fn option_restore_guard_restores_taken_value_on_drop() {
        let mut slot = Some(String::from("loaded"));
        {
            let mut guard = OptionRestoreGuard::take(&mut slot).unwrap();
            guard.push_str("-mutated");
        }
        assert_eq!(slot.as_deref(), Some("loaded-mutated"));
    }

    #[test]
    fn test_cfg_disabled_at_guidance_1_0() {
        assert!(!cfg_active(1.0), "guidance=1.0 must take the fast path");
    }

    #[test]
    fn test_cfg_disabled_just_below_1_0() {
        // LCM / Lightning workflows often expose 1.0 as a float that round-
        // trips with tiny noise; anything within the epsilon is "disabled".
        assert!(
            !cfg_active(1.0 - 1e-5),
            "guidance just under 1.0 must take the fast path"
        );
        assert!(
            !cfg_active(1.0 + 1e-5),
            "guidance just over 1.0 must take the fast path"
        );
    }

    #[test]
    fn test_cfg_enabled_at_guidance_1_5() {
        assert!(cfg_active(1.5), "guidance=1.5 must run full CFG");
    }

    #[test]
    fn test_cfg_enabled_at_guidance_7_5() {
        assert!(cfg_active(7.5), "guidance=7.5 must run full CFG");
    }

    #[test]
    fn test_cfg_enabled_just_outside_epsilon() {
        // Sanity: the boundary itself is not strictly active, but anything
        // visibly past it must engage CFG.
        assert!(
            cfg_active(1.0 + 2.0 * CFG_DISABLE_EPSILON),
            "guidance just past the epsilon must run full CFG"
        );
    }
}
