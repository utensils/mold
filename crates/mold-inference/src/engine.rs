use anyhow::Result;
use mold_core::GenerateRequest;
use mold_core::GenerateResponse;
use mold_core::GenerationReferenceMetadata;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::{Deref, DerefMut};

use crate::progress::{InferenceCancellationToken, ProgressCallback};

/// Controls how model components are loaded during inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadStrategy {
    /// Load all components at once, keep hot (server mode).
    #[default]
    Eager,
    /// Load-use-drop per component, minimizing peak memory (CLI one-shot mode).
    Sequential,
}

/// Engine-family batching contract consumed by the server-owned adaptive
/// batch planner.
///
/// A family that reports only `[1]` supports parallel singleton children but
/// does not claim native tensor batching. Sizes must be strictly increasing
/// and non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchExecutionCapability {
    pub native_batch_sizes: &'static [usize],
    pub cooperative_cancellation: bool,
}

/// One request-scoped, server-resolved media authority for generation.
///
/// This value is deliberately not serializable and its `Debug` implementation
/// exposes no private staging path. Public requests and durable metadata keep
/// only [`GenerationReferenceMetadata`]; the server opens the staged regular
/// file without following symlinks, then this binding hashes and retains that
/// exact handle for as long as the queued job is alive. Decoders must consume
/// [`Self::file`] and never reopen a pathname.
pub struct GenerationReferenceBinding {
    metadata: GenerationReferenceMetadata,
    file: File,
}

impl GenerationReferenceBinding {
    pub fn from_opened_file(
        metadata: GenerationReferenceMetadata,
        mut file: File,
        maximum_bytes: u64,
        cancellation: Option<&InferenceCancellationToken>,
    ) -> Result<Self> {
        anyhow::ensure!(
            metadata.index > 0,
            "generation reference index must be one-based"
        );
        anyhow::ensure!(
            metadata.sha256.len() == 64
                && metadata.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "generation reference digest must be SHA-256"
        );
        let file_metadata = file.metadata()?;
        anyhow::ensure!(
            file_metadata.is_file(),
            "generation reference binding requires an opened regular file"
        );
        anyhow::ensure!(
            maximum_bytes > 0 && file_metadata.len() > 0 && file_metadata.len() <= maximum_bytes,
            "generation reference binding exceeds its frozen byte limit"
        );
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        file.seek(SeekFrom::Start(0))?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        let mut observed_bytes = 0_u64;
        loop {
            if let Some(cancellation) = cancellation {
                cancellation.checkpoint()?;
            }
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            observed_bytes = observed_bytes
                .checked_add(u64::try_from(read)?)
                .ok_or_else(|| anyhow::anyhow!("generation reference byte count overflowed"))?;
            anyhow::ensure!(
                observed_bytes <= maximum_bytes,
                "generation reference binding exceeds its frozen byte limit"
            );
            digest.update(&buffer[..read]);
        }
        anyhow::ensure!(
            observed_bytes == file_metadata.len(),
            "generation reference binding changed size during verification"
        );
        if let Some(cancellation) = cancellation {
            cancellation.checkpoint()?;
        }
        let observed = format!("{:x}", digest.finalize());
        anyhow::ensure!(
            observed.eq_ignore_ascii_case(&metadata.sha256),
            "generation reference binding digest differs from staged media"
        );
        file.seek(SeekFrom::Start(0))?;
        Ok(Self { metadata, file })
    }

    pub fn metadata(&self) -> &GenerationReferenceMetadata {
        &self.metadata
    }

    /// The already-opened, content-verified media authority.
    ///
    /// Implementations may seek/read this handle directly or duplicate its
    /// descriptor for a decoder. Reopening the former staging pathname would
    /// discard the content identity proved by this binding.
    pub fn file(&self) -> &File {
        &self.file
    }

    #[cfg(test)]
    pub(crate) fn synthetic(metadata: GenerationReferenceMetadata) -> Self {
        Self {
            metadata,
            file: tempfile::tempfile().expect("create synthetic reference handle"),
        }
    }
}

impl std::fmt::Debug for GenerationReferenceBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerationReferenceBinding")
            .field("metadata", &self.metadata)
            .field("file", &"<opened regular file>")
            .finish()
    }
}

impl BatchExecutionCapability {
    /// Conservative default for adapters and test doubles that have not
    /// implemented an attempt-scoped cancellation hook.
    pub const SINGLETON_NON_COOPERATIVE: Self = Self {
        native_batch_sizes: &[1],
        cooperative_cancellation: false,
    };

    pub const SINGLETON_COOPERATIVE: Self = Self {
        native_batch_sizes: &[1],
        cooperative_cancellation: true,
    };

    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.native_batch_sizes.is_empty(),
            "native batch sizes must not be empty"
        );
        let mut previous = 0;
        for &size in self.native_batch_sizes {
            anyhow::ensure!(size > 0, "native batch sizes must be non-zero");
            anyhow::ensure!(
                size > previous,
                "native batch sizes must be strictly increasing"
            );
            previous = size;
        }
        Ok(())
    }
}

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    /// Generate with private, request-scoped reference media bound by the
    /// authenticated server ingress path. Engines fail closed by default so a
    /// reference-bearing request can never silently degrade into text-only
    /// generation.
    fn generate_with_reference_bindings(
        &mut self,
        req: &GenerateRequest,
        bindings: &[GenerationReferenceBinding],
    ) -> Result<GenerateResponse> {
        anyhow::ensure!(
            bindings.is_empty(),
            "model '{}' does not consume resolved generation references",
            self.model_name()
        );
        self.generate(req)
    }
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    /// Load model weights. Called automatically on first generate if not yet loaded.
    fn load(&mut self) -> Result<()>;
    /// Load model weights using the exact request placement selected before
    /// dispatch. Engines whose placement affects construction must override
    /// this method; the default preserves engines with no request-sensitive
    /// load behavior.
    fn load_for_request(&mut self, _req: &GenerateRequest) -> Result<()> {
        self.load()
    }
    /// Unload model weights to free GPU memory. The engine remains valid and
    /// can be re-loaded by calling `load()` or generating again.
    fn unload(&mut self) {}
    /// Set a progress callback for receiving loading/inference status updates.
    /// Default implementation is a no-op for engines that don't support progress.
    fn set_on_progress(&mut self, _callback: ProgressCallback) {}
    /// Clear any previously installed progress callback.
    fn clear_on_progress(&mut self) {}
    /// Install an attempt-scoped cooperative cancellation token. The caller
    /// may signal it from another thread; the engine observes it only at safe
    /// checkpoints on its owning worker thread.
    fn set_cancellation_token(&mut self, _token: InferenceCancellationToken) {}
    /// Remove the current attempt's cancellation token before the engine is
    /// reused for another job.
    fn clear_cancellation_token(&mut self) {}
    /// Declare this engine family's tested batch and cancellation contract.
    /// Adapters and test doubles default fail-closed; every production family
    /// overrides this explicitly after implementing safe checkpoints.
    fn batch_execution_capability(&self) -> BatchExecutionCapability {
        BatchExecutionCapability::SINGLETON_NON_COOPERATIVE
    }
    /// Return the model's resolved file paths, if available.
    /// Used by the server for pre-load memory checks on unified-memory systems.
    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        None
    }
    /// Exact load policy supplied when this engine was created, when the
    /// caller records one. Server scheduler plans use this to avoid silently
    /// reusing an engine built under a different residency contract.
    fn configured_load_strategy(&self) -> Option<LoadStrategy> {
        None
    }
    /// Exact request-controlled block-offload flag supplied at creation.
    fn configured_block_offload(&self) -> Option<bool> {
        None
    }
    /// Scheduler execution-plan fingerprint that governed this engine's
    /// successful load.
    fn configured_execution_fingerprint(&self) -> Option<&str> {
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

/// Run one attempt with a cancellation token installed, clearing it before
/// the cached engine can be reused even when inference unwinds.
pub fn with_inference_cancellation<T>(
    engine: &mut dyn InferenceEngine,
    token: InferenceCancellationToken,
    operation: impl FnOnce(&mut dyn InferenceEngine) -> T,
) -> T {
    engine.set_cancellation_token(token);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(engine)));
    engine.clear_cancellation_token();
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
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
        crate::runtime_env::value("MOLD_CFG_PLUS").as_deref(),
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
    use std::io::Write as _;

    use super::*;

    fn reference_binding_metadata(bytes: &[u8]) -> GenerationReferenceMetadata {
        let mut digest = Sha256::new();
        digest.update(bytes);
        GenerationReferenceMetadata {
            kind: mold_core::GenerationReferenceKind::Image,
            index: 1,
            name: Some("reference.png".to_string()),
            sha256: format!("{:x}", digest.finalize()),
            mime_type: "image/png".to_string(),
            width: Some(1),
            height: Some(1),
            frame_count: None,
            duration_ms: None,
            fps: None,
            has_audio: false,
            audio_duration_ms: None,
            audio_sample_count: None,
            audio_sample_rate: None,
            audio_channels: None,
            sample_rate: None,
            channels: None,
            sample_count: None,
            prepared_shape: None,
        }
    }

    fn reference_binding_file(bytes: &[u8]) -> File {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        file
    }

    #[test]
    fn reference_binding_is_bounded_and_cooperatively_cancellable() {
        let bytes = b"bounded reference bytes";
        let metadata = reference_binding_metadata(bytes);
        let binding = GenerationReferenceBinding::from_opened_file(
            metadata.clone(),
            reference_binding_file(bytes),
            bytes.len() as u64,
            None,
        )
        .unwrap();
        assert_eq!(binding.metadata(), &metadata);

        let oversized = GenerationReferenceBinding::from_opened_file(
            metadata.clone(),
            reference_binding_file(bytes),
            bytes.len() as u64 - 1,
            None,
        )
        .unwrap_err();
        assert!(oversized.to_string().contains("frozen byte limit"));

        let cancellation = InferenceCancellationToken::default();
        cancellation.cancel();
        let cancelled = GenerationReferenceBinding::from_opened_file(
            metadata,
            reference_binding_file(bytes),
            bytes.len() as u64,
            Some(&cancellation),
        )
        .unwrap_err();
        assert!(crate::progress::is_inference_cancelled(&cancelled));
    }

    #[test]
    fn batch_execution_capability_rejects_invalid_native_sizes() {
        for native_batch_sizes in [&[][..], &[0][..], &[1, 1][..], &[2, 1][..]] {
            assert!(
                BatchExecutionCapability {
                    native_batch_sizes,
                    cooperative_cancellation: true,
                }
                .validate()
                .is_err(),
                "{native_batch_sizes:?}"
            );
        }
        BatchExecutionCapability::SINGLETON_COOPERATIVE
            .validate()
            .unwrap();
        BatchExecutionCapability::SINGLETON_NON_COOPERATIVE
            .validate()
            .unwrap();
    }

    #[test]
    fn default_engine_does_not_claim_cooperative_cancellation() {
        struct MinimalEngine;
        impl InferenceEngine for MinimalEngine {
            fn generate(&mut self, _req: &GenerateRequest) -> Result<GenerateResponse> {
                unreachable!()
            }

            fn model_name(&self) -> &str {
                "minimal"
            }

            fn is_loaded(&self) -> bool {
                true
            }

            fn load(&mut self) -> Result<()> {
                Ok(())
            }
        }

        assert_eq!(
            MinimalEngine.batch_execution_capability(),
            BatchExecutionCapability::SINGLETON_NON_COOPERATIVE
        );
    }

    #[test]
    fn cancellation_scope_clears_a_cancelled_token_before_engine_reuse() {
        #[derive(Default)]
        struct ReusableEngine {
            token: Option<InferenceCancellationToken>,
        }
        impl InferenceEngine for ReusableEngine {
            fn generate(&mut self, _req: &GenerateRequest) -> Result<GenerateResponse> {
                self.token
                    .as_ref()
                    .map_or(Ok(()), InferenceCancellationToken::checkpoint)?;
                unreachable!("the test only exercises cancellation")
            }

            fn model_name(&self) -> &str {
                "reusable"
            }

            fn is_loaded(&self) -> bool {
                true
            }

            fn load(&mut self) -> Result<()> {
                Ok(())
            }

            fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
                self.token = Some(token);
            }

            fn clear_cancellation_token(&mut self) {
                self.token = None;
            }
        }

        let mut engine = ReusableEngine::default();
        let cancelled = InferenceCancellationToken::default();
        cancelled.cancel();
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "test",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap();
        let error =
            with_inference_cancellation(&mut engine, cancelled, |engine| engine.generate(&request))
                .unwrap_err();
        assert!(crate::progress::is_inference_cancelled(&error));
        assert!(engine.token.is_none());

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_inference_cancellation(&mut engine, InferenceCancellationToken::default(), |_| {
                panic!("injected inference panic")
            })
        }));
        assert!(panic.is_err());
        assert!(
            engine.token.is_none(),
            "an unwinding attempt must not poison the cached engine's next cancellation scope"
        );
    }

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
