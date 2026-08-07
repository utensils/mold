//! UpscaleEngine trait and UpscalerEngine implementation.
//!
//! Provides image upscaling via Real-ESRGAN models (RRDBNet and SRVGGNetCompact).
//! Architecture is auto-detected from safetensors state dict keys at load time.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::device::create_device;
use crate::engine::{BatchExecutionCapability, LoadStrategy};
use crate::progress::{InferenceCancellationToken, ProgressCallback, ProgressReporter};
use crate::weight_loader::load_safetensors_with_progress;

use super::arch::{detect_architecture, UpscalerArch};
use super::rrdbnet::RRDBNet;
use super::srvggnet::SRVGGNetCompact;
use super::tiling::{upscale_with_tiling, TilingConfig};

pub const UPSCALE_ACTIVATION_HEADROOM: u64 = 2 << 30;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedUpscaleArtifact {
    pub path: PathBuf,
    pub size_bytes: u64,
    /// Metadata identity/stat fence, not a content digest.
    pub identity_fingerprint: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactUpscalePlacement {
    Cpu,
    Device {
        backend: mold_core::GpuBackend,
        ordinal: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedUpscaleExecutionPlan {
    pub model_name: String,
    pub weights: ResolvedUpscaleArtifact,
    pub artifact_root: Option<PathBuf>,
    pub placement: ExactUpscalePlacement,
    pub predicted_vram_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub execution_fingerprint: String,
}

impl ResolvedUpscaleExecutionPlan {
    pub fn validate(&self) -> Result<()> {
        require_upscale_model_activation(
            &self.model_name,
            &self.weights.path,
            self.artifact_root.as_deref(),
        )?;
        let current = freeze_upscale_artifact(&self.weights.path)
            .map_err(|error| anyhow::anyhow!("upscaler artifact changed: {error}"))?;
        if current != self.weights {
            bail!("upscaler artifact changed after planning");
        }
        let (predicted_vram_peak_bytes, predicted_host_increment_bytes) =
            upscale_plan_memory_floors(&self.weights, self.placement);
        let execution_fingerprint = upscale_execution_fingerprint(
            &self.model_name,
            &self.weights,
            self.artifact_root.as_deref(),
            self.placement,
            predicted_vram_peak_bytes,
            predicted_host_increment_bytes,
        );
        if self.predicted_vram_peak_bytes != predicted_vram_peak_bytes
            || self.predicted_host_increment_bytes != predicted_host_increment_bytes
            || self.execution_fingerprint != execution_fingerprint
        {
            bail!("upscaler execution plan changed after resolution");
        }
        Ok(())
    }
}

fn require_upscale_model_activation(
    model_name: &str,
    weights_path: &Path,
    artifact_root: Option<&Path>,
) -> Result<()> {
    mold_core::require_model_activation(model_name, None)?;
    mold_core::require_model_artifact_activation(weights_path, artifact_root, None)?;
    Ok(())
}

fn upscale_artifact_identity(path: &Path, size_bytes: u64) -> Result<String> {
    let metadata = std::fs::metadata(path)?;
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_encoded_bytes());
    hash.update(size_bytes.to_le_bytes());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        hash.update(metadata.dev().to_le_bytes());
        hash.update(metadata.ino().to_le_bytes());
        hash.update(metadata.ctime().to_le_bytes());
        hash.update(metadata.ctime_nsec().to_le_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        hash.update(metadata.creation_time().to_le_bytes());
        hash.update(metadata.last_write_time().to_le_bytes());
        hash.update(metadata.file_size().to_le_bytes());
    }
    #[cfg(not(any(unix, windows)))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        hash.update(modified.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn freeze_upscale_artifact(path: &Path) -> Result<ResolvedUpscaleArtifact> {
    let path = std::fs::canonicalize(path)
        .map_err(|error| anyhow::anyhow!("could not resolve {}: {error}", path.display()))?;
    let size_bytes = std::fs::metadata(&path)?.len();
    Ok(ResolvedUpscaleArtifact {
        identity_fingerprint: upscale_artifact_identity(&path, size_bytes)?,
        path,
        size_bytes,
    })
}

fn upscale_plan_memory_floors(
    weights: &ResolvedUpscaleArtifact,
    placement: ExactUpscalePlacement,
) -> (u64, u64) {
    match placement {
        ExactUpscalePlacement::Cpu => (
            0,
            weights
                .size_bytes
                .saturating_add(UPSCALE_ACTIVATION_HEADROOM),
        ),
        ExactUpscalePlacement::Device {
            backend: mold_core::GpuBackend::Metal,
            ..
        } => (
            weights
                .size_bytes
                .saturating_add(UPSCALE_ACTIVATION_HEADROOM),
            weights
                .size_bytes
                .saturating_add(UPSCALE_ACTIVATION_HEADROOM),
        ),
        ExactUpscalePlacement::Device {
            backend: mold_core::GpuBackend::Cuda,
            ..
        } => (
            weights
                .size_bytes
                .saturating_add(UPSCALE_ACTIVATION_HEADROOM),
            weights.size_bytes,
        ),
    }
}

fn upscale_execution_fingerprint(
    model_name: &str,
    weights: &ResolvedUpscaleArtifact,
    artifact_root: Option<&Path>,
    placement: ExactUpscalePlacement,
    predicted_vram_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.upscale.execution.v1");
    hash.update(model_name.as_bytes());
    hash.update(weights.path.as_os_str().as_encoded_bytes());
    hash.update(weights.size_bytes.to_le_bytes());
    hash.update(weights.identity_fingerprint.as_bytes());
    if let Some(root) = artifact_root {
        hash.update(b"artifact-root\0");
        hash.update(root.as_os_str().as_encoded_bytes());
    }
    match placement {
        ExactUpscalePlacement::Cpu => hash.update(b"cpu"),
        ExactUpscalePlacement::Device { backend, ordinal } => {
            hash.update(match backend {
                mold_core::GpuBackend::Cuda => b"cuda".as_slice(),
                mold_core::GpuBackend::Metal => b"metal".as_slice(),
            });
            hash.update(ordinal.to_le_bytes());
        }
    }
    hash.update(predicted_vram_peak_bytes.to_le_bytes());
    hash.update(predicted_host_increment_bytes.to_le_bytes());
    format!("{:x}", hash.finalize())
}

pub fn resolve_upscale_execution_plan(
    model_name: impl Into<String>,
    weights_path: &Path,
    artifact_root: Option<&Path>,
    placement: ExactUpscalePlacement,
) -> Result<ResolvedUpscaleExecutionPlan> {
    let model_name = model_name.into();
    require_upscale_model_activation(&model_name, weights_path, artifact_root)?;
    let weights = freeze_upscale_artifact(weights_path)?;
    let artifact_root = artifact_root
        .map(|root| std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf()));
    Ok(resolve_upscale_execution_plan_from_artifact(
        model_name,
        weights,
        artifact_root,
        placement,
    ))
}

pub fn resolve_upscale_execution_plan_from_artifact(
    model_name: impl Into<String>,
    weights: ResolvedUpscaleArtifact,
    artifact_root: Option<PathBuf>,
    placement: ExactUpscalePlacement,
) -> ResolvedUpscaleExecutionPlan {
    let model_name = model_name.into();
    let (predicted_vram_peak_bytes, predicted_host_increment_bytes) =
        upscale_plan_memory_floors(&weights, placement);
    let execution_fingerprint = upscale_execution_fingerprint(
        &model_name,
        &weights,
        artifact_root.as_deref(),
        placement,
        predicted_vram_peak_bytes,
        predicted_host_increment_bytes,
    );
    ResolvedUpscaleExecutionPlan {
        model_name,
        weights,
        artifact_root,
        placement,
        predicted_vram_peak_bytes,
        predicted_host_increment_bytes,
        execution_fingerprint,
    }
}

fn create_exact_upscale_device_with<F>(
    plan: &ResolvedUpscaleExecutionPlan,
    progress: &ProgressReporter,
    gpu_factory: F,
) -> Result<Device>
where
    F: FnOnce(mold_core::GpuBackend, usize) -> Result<Device>,
{
    plan.validate()?;
    progress.checkpoint()?;
    match plan.placement {
        ExactUpscalePlacement::Cpu => {
            progress.info("Using exact CPU placement for upscaling");
            crate::device::preflight_memory_check(
                "Upscaler",
                plan.weights.size_bytes,
                UPSCALE_ACTIVATION_HEADROOM,
            )?;
            Ok(Device::Cpu)
        }
        ExactUpscalePlacement::Device { backend, ordinal } => {
            if backend == mold_core::GpuBackend::Metal {
                crate::device::preflight_memory_check(
                    "Upscaler",
                    plan.weights.size_bytes,
                    UPSCALE_ACTIVATION_HEADROOM,
                )?;
            }
            gpu_factory(backend, ordinal)
        }
    }
}

/// Trait for upscaling inference backends.
pub trait UpscaleEngine: Send + Sync {
    /// Upscale an image.
    fn upscale(&mut self, req: &mold_core::UpscaleRequest) -> Result<mold_core::UpscaleResponse>;

    /// Model name (e.g. "real-esrgan-x4plus:fp16").
    fn model_name(&self) -> &str;

    /// Whether model weights are currently loaded.
    fn is_loaded(&self) -> bool;

    /// Load model weights. Called automatically on first upscale if not yet loaded.
    fn load(&mut self) -> Result<()>;

    /// Unload model weights to free GPU memory.
    fn unload(&mut self);

    /// The upscaling factor (e.g. 2 or 4).
    fn scale_factor(&self) -> u32;

    /// Set a progress callback.
    fn set_on_progress(&mut self, callback: ProgressCallback);

    /// Clear any previously installed progress callback.
    fn clear_on_progress(&mut self);

    /// Install the current attempt's cooperative cancellation token.
    fn set_cancellation_token(&mut self, _token: InferenceCancellationToken) {}

    /// Clear the token before this engine is reused by another attempt.
    fn clear_cancellation_token(&mut self) {}

    /// Declare tested native batch sizes for standalone and post-generation
    /// upscale work. Test doubles fail closed unless they override this.
    fn batch_execution_capability(&self) -> BatchExecutionCapability {
        BatchExecutionCapability::SINGLETON_NON_COOPERATIVE
    }
}

/// Run one upscale attempt with a token installed and guarantee that a
/// cancelled token cannot poison the cached engine's next invocation.
pub fn with_upscale_cancellation<T>(
    engine: &mut dyn UpscaleEngine,
    token: InferenceCancellationToken,
    operation: impl FnOnce(&mut dyn UpscaleEngine) -> T,
) -> T {
    engine.set_cancellation_token(token);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(engine)));
    engine.clear_cancellation_token();
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Loaded model state (architecture-polymorphic).
#[allow(clippy::large_enum_variant)]
enum LoadedModel {
    SRVGGNet(SRVGGNetCompact),
    RRDBNet(RRDBNet),
}

impl LoadedModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            LoadedModel::SRVGGNet(m) => m.forward(xs),
            LoadedModel::RRDBNet(m) => m.forward(xs),
        }
    }
}

struct LoadedState {
    model: LoadedModel,
    device: Device,
    dtype: DType,
    scale: u32,
}

/// Concrete upscaler engine implementation.
pub struct UpscalerEngine {
    name: String,
    weights_path: PathBuf,
    loaded: Option<LoadedState>,
    progress: ProgressReporter,
    load_strategy: LoadStrategy,
    /// GPU ordinal this engine is pinned to. Same multi-GPU footgun as
    /// `Ltx2Engine::gpu_ordinal` — hardcoding `0` would corrupt a sibling
    /// GPU's CUDA context on unload.
    gpu_ordinal: usize,
    exact_plan: Option<ResolvedUpscaleExecutionPlan>,
}

impl UpscalerEngine {
    pub fn new(
        name: String,
        weights_path: PathBuf,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            name,
            weights_path,
            loaded: None,
            progress: ProgressReporter::default(),
            load_strategy,
            gpu_ordinal,
            exact_plan: None,
        }
    }

    pub fn from_resolved_plan(
        plan: ResolvedUpscaleExecutionPlan,
        load_strategy: LoadStrategy,
    ) -> Self {
        let gpu_ordinal = match plan.placement {
            ExactUpscalePlacement::Cpu => 0,
            ExactUpscalePlacement::Device { ordinal, .. } => ordinal,
        };
        Self {
            name: plan.model_name.clone(),
            weights_path: plan.weights.path.clone(),
            loaded: None,
            progress: ProgressReporter::default(),
            load_strategy,
            gpu_ordinal,
            exact_plan: Some(plan),
        }
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        if self.loaded.is_none() {
            self.load()?;
        }
        Ok(())
    }
}

impl UpscaleEngine for UpscalerEngine {
    fn upscale(&mut self, req: &mold_core::UpscaleRequest) -> Result<mold_core::UpscaleResponse> {
        self.progress.checkpoint()?;
        self.ensure_loaded()?;

        let start = Instant::now();
        self.progress.stage_start("Decoding input image");

        // Decode input image
        let img = image::load_from_memory(&req.image).context("failed to decode input image")?;
        let original_width = img.width();
        let original_height = img.height();
        let rgb = img.to_rgb8();

        // Convert to tensor [1, 3, H, W] in [0, 1] range
        let (h, w) = (rgb.height() as usize, rgb.width() as usize);
        let raw: Vec<f32> = rgb.into_raw().iter().map(|&v| v as f32 / 255.0).collect();
        // raw is in HWC format, need CHW
        let mut chw = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    chw[c * h * w + y * w + x] = raw[y * w * 3 + x * 3 + c];
                }
            }
        }
        let state = self.loaded.as_ref().unwrap();
        let input = Tensor::from_vec(chw, (1, 3, h, w), &state.device)?.to_dtype(state.dtype)?;

        self.progress
            .stage_done("Decoding input image", start.elapsed());
        self.progress.stage_start("Upscaling");
        let upscale_start = Instant::now();

        // Configure tiling
        let tile_size = req.tile_size.unwrap_or(512);
        let tiling_config = if tile_size == 0 {
            // Tiling disabled — use image dimensions as tile size
            TilingConfig {
                tile_size: w.max(h) as u32,
                overlap: 0,
                min_tile_size: w.max(h) as u32,
            }
        } else {
            TilingConfig {
                tile_size,
                overlap: 32,
                min_tile_size: 128,
            }
        };

        let scale = state.scale;
        let device_clone = state.device.clone();

        // Build the forward function closure — captures the loaded model state
        let loaded = self.loaded.as_ref().unwrap();
        let forward_fn = |tile: &Tensor| -> Result<Tensor> { loaded.model.forward(tile) };

        let output = upscale_with_tiling(
            &input,
            &forward_fn,
            scale,
            &tiling_config,
            &device_clone,
            &self.progress,
        )?;

        self.progress.phase_done(
            crate::ProgressPhase::Upscale,
            "Upscaling",
            upscale_start.elapsed(),
        );
        self.progress.stage_start("Encoding output");
        let encode_start = Instant::now();

        // Convert output tensor to image bytes
        let output = output.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (_, _, out_h, out_w) = output.dims4()?;

        // Clamp to [0, 1] and convert to u8
        let output = output.clamp(0.0f32, 1.0f32)?;
        let output_flat = output.flatten_all()?;
        let output_data: Vec<f32> = output_flat.to_vec1()?;

        // CHW -> HWC -> RGB8
        let mut rgb_out = vec![0u8; out_h * out_w * 3];
        for c in 0..3 {
            for y in 0..out_h {
                for x in 0..out_w {
                    rgb_out[y * out_w * 3 + x * 3 + c] =
                        (output_data[c * out_h * out_w + y * out_w + x] * 255.0).round() as u8;
                }
            }
        }

        let img_buf: image::RgbImage =
            image::ImageBuffer::from_raw(out_w as u32, out_h as u32, rgb_out)
                .context("failed to create output image buffer")?;

        let mut metadata = req.metadata.clone();
        if let Some(metadata) = metadata.as_mut() {
            metadata.apply_output_dimensions(out_w as u32, out_h as u32);
        }
        let encoded =
            crate::image::encode_rgb_image(&img_buf, req.output_format, metadata.as_ref())?;

        self.progress
            .stage_done("Encoding output", encode_start.elapsed());

        let upscale_time_ms = start.elapsed().as_millis() as u64;

        // Unload if sequential mode (CLI one-shot)
        if self.load_strategy == LoadStrategy::Sequential {
            self.unload();
        }

        Ok(mold_core::UpscaleResponse {
            image: mold_core::ImageData {
                data: encoded,
                format: req.output_format,
                width: out_w as u32,
                height: out_h as u32,
                index: 0,
            },
            upscale_time_ms,
            model: self.name.clone(),
            scale_factor: scale,
            original_width,
            original_height,
        })
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        let load_start = Instant::now();
        self.progress.stage_start("Loading upscaler model");

        let device = if let Some(plan) = &self.exact_plan {
            create_exact_upscale_device_with(plan, &self.progress, |backend, ordinal| {
                crate::device::create_exact_gpu_device(backend, ordinal, &self.progress)
            })?
        } else {
            create_device(self.gpu_ordinal, &self.progress)?
        };

        // Determine dtype: prefer F16 on GPU, F32 on CPU
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        // Read tensor names from safetensors header for architecture detection.
        // Only reads the 8-byte length prefix + JSON header, not the full file
        // (avoids a 64MB heap allocation for FP32 models before mmap).
        let tensor_names = {
            use std::io::Read as _;
            let mut f = std::fs::File::open(&self.weights_path)?;
            let mut len_buf = [0u8; 8];
            f.read_exact(&mut len_buf)?;
            let header_len = u64::from_le_bytes(len_buf) as usize;
            let mut header_buf = vec![0u8; header_len];
            f.read_exact(&mut header_buf)?;
            let header: std::collections::HashMap<String, serde_json::Value> =
                serde_json::from_slice(&header_buf)?;
            header
                .keys()
                .filter(|k| *k != "__metadata__")
                .cloned()
                .collect::<Vec<_>>()
        };
        let name_refs: Vec<&str> = tensor_names.iter().map(|s| s.as_str()).collect();
        let arch = detect_architecture(&name_refs)?;

        // Load weights via mmap VarBuilder
        let vb = load_safetensors_with_progress(
            &[&self.weights_path],
            dtype,
            &device,
            "upscaler",
            &self.progress,
        )?;
        self.progress
            .info(&format!("Detected architecture: {arch:?}"));

        let (model, scale) = match arch {
            UpscalerArch::SRVGGNetCompact {
                num_feat,
                num_conv,
                scale,
            } => {
                let m = SRVGGNetCompact::load(&vb, num_feat, num_conv, scale)?;
                (LoadedModel::SRVGGNet(m), scale)
            }
            UpscalerArch::RRDBNet {
                num_feat,
                num_grow_ch,
                num_block,
                scale,
            } => {
                let m = RRDBNet::load(&vb, num_feat, num_grow_ch, num_block, scale)?;
                (LoadedModel::RRDBNet(m), scale)
            }
        };

        self.loaded = Some(LoadedState {
            model,
            device,
            dtype,
            scale,
        });

        self.progress.phase_done(
            crate::ProgressPhase::ModelLoad,
            "Loading upscaler model",
            load_start.elapsed(),
        );
        Ok(())
    }

    fn unload(&mut self) {
        if self.loaded.is_some() {
            self.loaded = None;
            let free_after_drop = self
                .exact_plan
                .as_ref()
                .is_none_or(|plan| !matches!(plan.placement, ExactUpscalePlacement::Cpu))
                .then(|| crate::device::post_drop_free_vram_bytes(self.gpu_ordinal));
            tracing::info!(
                free_vram_bytes = ?free_after_drop,
                "Upscaler model unloaded: {}",
                self.name
            );
        }
    }

    fn scale_factor(&self) -> u32 {
        self.loaded.as_ref().map(|s| s.scale).unwrap_or(4)
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }

    fn clear_on_progress(&mut self) {
        self.progress.clear_callback();
    }

    fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.progress.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.progress.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> BatchExecutionCapability {
        BatchExecutionCapability::SINGLETON_COOPERATIVE
    }
}

/// Create an upscale engine for the given model.
pub fn create_upscale_engine(
    model_name: String,
    weights_path: PathBuf,
    artifact_root: Option<&Path>,
    load_strategy: LoadStrategy,
    gpu_ordinal: usize,
) -> Result<Box<dyn UpscaleEngine>> {
    require_upscale_model_activation(&model_name, &weights_path, artifact_root)?;
    if !weights_path.exists() {
        bail!("upscaler weights not found: {}", weights_path.display());
    }
    Ok(Box::new(UpscalerEngine::new(
        model_name,
        weights_path,
        load_strategy,
        gpu_ordinal,
    )))
}

pub fn create_upscale_engine_from_resolved_plan(
    plan: ResolvedUpscaleExecutionPlan,
    load_strategy: LoadStrategy,
) -> Result<Box<dyn UpscaleEngine>> {
    plan.validate()?;
    Ok(Box::new(UpscalerEngine::from_resolved_plan(
        plan,
        load_strategy,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[test]
    fn upscale_engine_rejects_restricted_model_before_reading_weights() {
        let missing = PathBuf::from("/definitely/missing/upscaler.safetensors");
        let error = create_upscale_engine(
            "MiniMaxH3Scheduler".into(),
            missing,
            None,
            LoadStrategy::Eager,
            0,
        )
        .err()
        .expect("restricted model must fail before filesystem access");
        assert!(
            error
                .to_string()
                .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn upscale_execution_plan_rejects_restricted_weight_path_before_metadata() {
        let missing = PathBuf::from("/definitely/missing/MiniMax-H3/upscaler.safetensors");
        let error = resolve_upscale_execution_plan(
            "ordinary-upscaler",
            &missing,
            None,
            ExactUpscalePlacement::Cpu,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn upscale_policy_ignores_h3_named_root_but_rejects_nested_h3_artifact() {
        let temp = tempfile::tempdir().unwrap();
        let artifact_root = temp.path().join("mold-uat/minimax-h3/models");
        let ordinary = artifact_root.join("real-esrgan/upscaler.safetensors");
        std::fs::create_dir_all(ordinary.parent().unwrap()).unwrap();
        std::fs::write(&ordinary, vec![0_u8; 64]).unwrap();

        resolve_upscale_execution_plan(
            "ordinary-upscaler",
            &ordinary,
            Some(&artifact_root),
            ExactUpscalePlacement::Cpu,
        )
        .expect("the storage root name must not taint an ordinary upscaler");

        let nested_h3 = artifact_root.join("MiniMax-H3/upscaler.safetensors");
        let error = resolve_upscale_execution_plan(
            "ordinary-upscaler",
            &nested_h3,
            Some(&artifact_root),
            ExactUpscalePlacement::Cpu,
        )
        .expect_err("a nested H3 artifact must remain gated");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    struct CancellationContractEngine {
        set: Arc<AtomicBool>,
        cleared: Arc<AtomicBool>,
    }

    impl UpscaleEngine for CancellationContractEngine {
        fn upscale(
            &mut self,
            _req: &mold_core::UpscaleRequest,
        ) -> Result<mold_core::UpscaleResponse> {
            unreachable!()
        }

        fn model_name(&self) -> &str {
            "contract"
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn load(&mut self) -> Result<()> {
            Ok(())
        }

        fn unload(&mut self) {}

        fn scale_factor(&self) -> u32 {
            4
        }

        fn set_on_progress(&mut self, _callback: ProgressCallback) {}

        fn clear_on_progress(&mut self) {}

        fn set_cancellation_token(&mut self, _token: InferenceCancellationToken) {
            self.set.store(true, Ordering::SeqCst);
        }

        fn clear_cancellation_token(&mut self) {
            self.cleared.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn upscale_cancellation_scope_always_clears_before_reuse() {
        let set = Arc::new(AtomicBool::new(false));
        let cleared = Arc::new(AtomicBool::new(false));
        let mut engine = CancellationContractEngine {
            set: set.clone(),
            cleared: cleared.clone(),
        };

        with_upscale_cancellation(&mut engine, InferenceCancellationToken::default(), |_| ());

        assert!(set.load(Ordering::SeqCst));
        assert!(cleared.load(Ordering::SeqCst));
        assert_eq!(
            engine.batch_execution_capability(),
            BatchExecutionCapability::SINGLETON_NON_COOPERATIVE
        );

        cleared.store(false, Ordering::SeqCst);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_upscale_cancellation(&mut engine, InferenceCancellationToken::default(), |_| {
                panic!("injected upscaler panic")
            })
        }));
        assert!(panic.is_err());
        assert!(
            cleared.load(Ordering::SeqCst),
            "an unwinding upscale must clear its attempt token before engine reuse"
        );
    }

    #[test]
    fn exact_upscale_plan_freezes_artifact_placement_and_memory() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();

        let cpu = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            ExactUpscalePlacement::Cpu,
        )
        .unwrap();
        let gpu = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            ExactUpscalePlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 7,
            },
        )
        .unwrap();

        assert_eq!(cpu.predicted_vram_peak_bytes, 0);
        assert!(cpu.predicted_host_increment_bytes > gpu.predicted_host_increment_bytes);
        assert_eq!(
            gpu.predicted_vram_peak_bytes,
            4096 + UPSCALE_ACTIVATION_HEADROOM
        );
        assert_ne!(cpu.execution_fingerprint, gpu.execution_fingerprint);
        cpu.validate().unwrap();
        gpu.validate().unwrap();
    }

    #[test]
    fn exact_upscale_plan_rejects_artifact_drift_before_device_creation() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();
        let plan = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            ExactUpscalePlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 3,
            },
        )
        .unwrap();
        std::fs::write(&weights, b"replacement").unwrap();

        let calls = std::sync::atomic::AtomicUsize::new(0);
        let error =
            create_exact_upscale_device_with(&plan, &ProgressReporter::default(), |_, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Device::Cpu)
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("artifact changed"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn exact_cpu_upscale_never_touches_gpu_and_exact_gpu_never_falls_back() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, vec![0_u8; 4096]).unwrap();
        let cpu = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            ExactUpscalePlacement::Cpu,
        )
        .unwrap();
        let gpu = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            ExactUpscalePlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 5,
            },
        )
        .unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);

        let device =
            create_exact_upscale_device_with(&cpu, &ProgressReporter::default(), |_, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                anyhow::bail!("CPU plan touched a GPU")
            })
            .unwrap();
        assert!(device.is_cpu());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let error = create_exact_upscale_device_with(
            &gpu,
            &ProgressReporter::default(),
            |backend, ordinal| {
                calls.fetch_add(1, Ordering::SeqCst);
                anyhow::bail!("exact {backend:?}:{ordinal} unavailable")
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exact Cuda:5 unavailable"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
