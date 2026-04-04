//! UpscaleEngine trait and UpscalerEngine implementation.
//!
//! Provides image upscaling via Real-ESRGAN models (RRDBNet and SRVGGNetCompact).
//! Architecture is auto-detected from safetensors state dict keys at load time.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use std::time::Instant;

use crate::device::create_device;
use crate::engine::LoadStrategy;
use crate::progress::{ProgressCallback, ProgressReporter};
use crate::weight_loader::load_safetensors_with_progress;

use super::arch::{detect_architecture, UpscalerArch};
use super::rrdbnet::RRDBNet;
use super::srvggnet::SRVGGNetCompact;
use super::tiling::{upscale_with_tiling, TilingConfig};

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
}

impl UpscalerEngine {
    pub fn new(name: String, weights_path: PathBuf, load_strategy: LoadStrategy) -> Self {
        Self {
            name,
            weights_path,
            loaded: None,
            progress: ProgressReporter::default(),
            load_strategy,
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

        self.progress
            .stage_done("Upscaling", upscale_start.elapsed());
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

        let encoded = {
            let mut buf = std::io::Cursor::new(Vec::new());
            let fmt = match req.output_format {
                mold_core::OutputFormat::Png => image::ImageFormat::Png,
                mold_core::OutputFormat::Jpeg => image::ImageFormat::Jpeg,
            };
            img_buf.write_to(&mut buf, fmt)?;
            buf.into_inner()
        };

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

        let device = create_device(&self.progress)?;

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

        self.progress
            .stage_done("Loading upscaler model", load_start.elapsed());
        Ok(())
    }

    fn unload(&mut self) {
        if self.loaded.is_some() {
            self.loaded = None;
            crate::reclaim_gpu_memory();
            tracing::info!("Upscaler model unloaded: {}", self.name);
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
}

/// Create an upscale engine for the given model.
pub fn create_upscale_engine(
    model_name: String,
    weights_path: PathBuf,
    load_strategy: LoadStrategy,
) -> Result<Box<dyn UpscaleEngine>> {
    if !weights_path.exists() {
        bail!("upscaler weights not found: {}", weights_path.display());
    }
    Ok(Box::new(UpscalerEngine::new(
        model_name,
        weights_path,
        load_strategy,
    )))
}
