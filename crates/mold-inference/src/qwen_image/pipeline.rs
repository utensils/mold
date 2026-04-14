//! Qwen-Image-2512 inference engine.
//!
//! Pipeline: Qwen2.5-VL text encoder -> QwenImageTransformer2DModel -> QwenImage VAE
//!
//! Architecture follows Z-Image closely (both from Alibaba/Tongyi):
//! - Dual-stream transformer with joint attention and 3D RoPE
//! - Flow-matching Euler discrete scheduler with dynamic shifting
//! - Drop-and-reload for text encoder to manage VRAM
//! - Both Eager and Sequential loading modes
//!
//! Key differences from Z-Image:
//! - 60 identical dual-stream blocks (no noise_refiner/context_refiner)
//! - Qwen2.5-VL text encoder (hidden_size=3584) instead of Qwen3 (2560)
//! - Custom VAE with per-channel latent normalization
//! - Official diffusers-style exponential time shift with dynamic per-image stretch

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_transformers::models::z_image::postprocess_image;
use candle_transformers::quantized_var_builder;
use mold_core::{fit_to_target_area, GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use super::quantized_transformer::QuantizedQwenImageTransformer2DModel;
use super::sampling::{image_seq_len, QwenImageScheduler};
use super::transformer::{QwenImageConfig, QwenImageTransformer2DModel};
use super::vae::QwenImageVae;
use crate::cache::{
    clear_cache, prompt_text_key, CachedTensor, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    fits_in_memory, fmt_gb, free_vram_bytes, memory_status_string, preflight_memory_check,
    qwen2_vram_threshold, should_use_gpu,
};
use crate::encoders;
use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::img_utils;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressReporter};
use crate::upscaler::tiling::{upscale_with_tiling, TilingConfig};

/// Minimum free VRAM (bytes) required to place Qwen-Image VAE on GPU.
/// The VAE weights are ~300MB; decode workspace at 1024x1024 needs ~1-2GB.
const VAE_DECODE_VRAM_THRESHOLD: u64 = 2_500_000_000;
// Use a single space rather than an empty string so the unconditional CFG path
// stays explicit after Qwen prompt templating and token windowing.
const QWEN_EMPTY_NEGATIVE_PROMPT: &str = " ";
const QWEN_NATIVE_WIDTH: usize = 1328;
const QWEN_NATIVE_HEIGHT: usize = 1328;
const QWEN_GGUF_NATIVE_CFG_HEADROOM: u64 = 14_000_000_000;
const QWEN_GGUF_MIN_CFG_HEADROOM: u64 = 3_000_000_000;
const QWEN_VAE_TILE_SIZES: [u32; 3] = [64, 32, 16];
const QWEN_IMAGE_EDIT_VAE_AREA: u32 = 1024 * 1024;
const QWEN_IMAGE_EDIT_SYSTEM_PROMPT: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Minimum free VRAM for BF16 Qwen2.5-VL 7B text encoder on GPU.
/// ~14GB model + 2GB headroom.
const QWEN2_FP16_VRAM_THRESHOLD: u64 = 16_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen2TextEncoderMode {
    Auto,
    Gpu,
    CpuStage,
    Cpu,
}

impl Qwen2TextEncoderMode {
    fn from_env() -> Self {
        match std::env::var("MOLD_QWEN2_TEXT_ENCODER_MODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "gpu" => Self::Gpu,
            "cpu-stage" => Self::CpuStage,
            "cpu_stage" => Self::CpuStage,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen2TextEncoderPlan {
    use_gpu: bool,
    use_cpu_staging: bool,
}

#[derive(Debug, Clone)]
struct ResolvedQwen2TextEncoder {
    paths: Vec<std::path::PathBuf>,
    vision_paths: Vec<std::path::PathBuf>,
    is_gguf: bool,
    variant_label: String,
    size_bytes: u64,
    auto_use_gpu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen2TextEncoderUsage {
    Sequential,
    Resident,
}

/// Check if a Qwen-Image safetensors checkpoint stores weights in FP8 (F8_E4M3).
/// Uses filename pattern first, then dtype probing as fallback.
fn safetensors_is_fp8(path: &Path) -> bool {
    // Filename-based detection
    if path.to_str().map(|s| s.contains("fp8")).unwrap_or(false) {
        return true;
    }
    // Dtype probing — try both ComfyUI and diffusers key names
    let Ok(tensors) = (unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path]) })
    else {
        return false;
    };
    for key in ["x_embedder.weight", "img_in.weight"] {
        if let Ok(t) = tensors.load(key, &Device::Cpu) {
            return t.dtype() == DType::F8E4M3;
        }
    }
    false
}

/// Check if text encoder safetensors contain FP8 weights.
/// Uses filename pattern first (reliable for known ComfyUI FP8 models),
/// then falls back to dtype probing.
fn text_encoder_is_fp8(paths: &[std::path::PathBuf]) -> bool {
    // Filename-based detection (ComfyUI FP8 models have "fp8" in name)
    if paths
        .iter()
        .any(|p| p.to_str().map(|s| s.contains("fp8")).unwrap_or(false))
    {
        return true;
    }
    // Dtype probing fallback — try common key names
    let Some(first) = paths.first() else {
        return false;
    };
    let Ok(tensors) = (unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[first]) })
    else {
        return false;
    };
    for key in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ] {
        if let Ok(t) = tensors.load(key, &Device::Cpu) {
            return t.dtype() == DType::F8E4M3;
        }
    }
    false
}

/// Loaded Qwen-Image model components, ready for inference.
struct LoadedQwenImage {
    /// Transformer wrapped in Option for drop-and-reload pattern.
    transformer: Option<QwenImageTransformer>,
    text_encoder: encoders::qwen2_text::Qwen2TextEncoder,
    vae: QwenImageVae,
    vae_path: std::path::PathBuf,
    transformer_cfg: QwenImageConfig,
    /// GPU device for transformer + denoising
    device: Device,
    /// Device where the VAE lives (may be CPU if VRAM is tight)
    vae_device: Device,
    dtype: DType,
}

#[allow(clippy::large_enum_variant)]
enum QwenImageTransformer {
    BF16(QwenImageTransformer2DModel),
    Quantized(QuantizedQwenImageTransformer2DModel),
    Offloaded(super::offload::OffloadedQwenImageTransformer),
}

#[derive(Clone)]
struct CachedPromptConditioning {
    hidden_states: CachedTensor,
    valid_len: usize,
}

impl CachedPromptConditioning {
    fn from_parts(hidden_states: &Tensor, valid_len: usize) -> Result<Self> {
        Ok(Self {
            hidden_states: CachedTensor::from_tensor(hidden_states)?,
            valid_len,
        })
    }

    fn restore(&self, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let hidden_states = self.hidden_states.restore(device, dtype)?;
        let mut mask = vec![0u8; hidden_states.dim(1)?];
        for value in &mut mask[..self.valid_len] {
            *value = 1;
        }
        let attention_mask = Tensor::from_vec(mask, (1, hidden_states.dim(1)?), device)?;
        Ok((hidden_states, attention_mask))
    }
}

fn pad_text_conditioning(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    target_len: usize,
) -> Result<(Tensor, Tensor)> {
    let seq_len = hidden_states.dim(1)?;
    if seq_len == target_len {
        return Ok((hidden_states.clone(), attention_mask.clone()));
    }
    if seq_len > target_len {
        bail!("cannot shrink text conditioning from {seq_len} to {target_len}");
    }

    let hidden_dim = hidden_states.dim(2)?;
    let pad_len = target_len - seq_len;
    let pad_hs = Tensor::zeros(
        (hidden_states.dim(0)?, pad_len, hidden_dim),
        hidden_states.dtype(),
        hidden_states.device(),
    )?;
    let pad_mask = Tensor::zeros(
        (attention_mask.dim(0)?, pad_len),
        attention_mask.dtype(),
        attention_mask.device(),
    )?;

    Ok((
        Tensor::cat(&[hidden_states, &pad_hs], 1)?,
        Tensor::cat(&[attention_mask, &pad_mask], 1)?,
    ))
}

fn align_cfg_conditioning(
    cond_hs: &Tensor,
    cond_mask: &Tensor,
    uncond_hs: &Tensor,
    uncond_mask: &Tensor,
) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
    let target_len = cond_hs.dim(1)?.max(uncond_hs.dim(1)?);
    let cond = pad_text_conditioning(cond_hs, cond_mask, target_len)?;
    let uncond = pad_text_conditioning(uncond_hs, uncond_mask, target_len)?;
    Ok((cond, uncond))
}

impl QwenImageTransformer {
    fn supports_cfg_batching(&self) -> bool {
        match self {
            Self::Quantized(model) => model.supports_cfg_batching(),
            _ => true,
        }
    }

    fn forward(
        &self,
        latents: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
            Self::Quantized(model) => {
                Ok(model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)?)
            }
            Self::Offloaded(model) => {
                model.forward(latents, t, encoder_hidden_states, encoder_attention_mask)
            }
        }
    }

    fn forward_packed(
        &self,
        packed_latents: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        match self {
            Self::BF16(model) => Ok(model.forward_packed(
                packed_latents,
                t,
                encoder_hidden_states,
                encoder_attention_mask,
                img_shapes,
            )?),
            Self::Quantized(model) => Ok(model.forward_packed(
                packed_latents,
                t,
                encoder_hidden_states,
                encoder_attention_mask,
                img_shapes,
            )?),
            Self::Offloaded(model) => model.forward_packed(
                packed_latents,
                t,
                encoder_hidden_states,
                encoder_attention_mask,
                img_shapes,
            ),
        }
    }
}

/// Qwen-Image-2512 inference engine.
pub struct QwenImageEngine {
    base: EngineBase<LoadedQwenImage>,
    prompt_cache: Mutex<LruCache<String, CachedPromptConditioning>>,
    offload: bool,
}

impl QwenImageEngine {
    fn is_edit_family(&self) -> bool {
        self.base.model_name.starts_with("qwen-image-edit")
    }

    fn should_preload_text_encoder(&self) -> bool {
        !self.is_edit_family()
    }

    fn text_encoder_load_dtype(use_gpu: bool, gpu_dtype: DType) -> DType {
        if use_gpu {
            gpu_dtype
        } else {
            // Candle CPU matmul does not support BF16 for the Qwen2.5 encoder path.
            // Keep CPU language/vision encoding in F32 and use quantized GGUF when
            // lower host residency is needed.
            DType::F32
        }
    }

    fn transformer_config(&self) -> QwenImageConfig {
        if self.is_edit_family() {
            QwenImageConfig::qwen_image_edit_2511()
        } else {
            QwenImageConfig::qwen_image_2512()
        }
    }

    fn qwen_image_edit_prompt(prompt: &str, image_count: usize) -> String {
        let picture_prefix = (0..image_count)
            .map(|idx| {
                format!(
                    "Picture {}: <|vision_start|><|image_pad|><|vision_end|>",
                    idx + 1
                )
            })
            .collect::<String>();
        format!(
            "<|im_start|>system\n{QWEN_IMAGE_EDIT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{picture_prefix}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    fn qwen_image_edit_image_dims(image: &[u8], target_area: u32) -> Result<(u32, u32)> {
        let img = image::load_from_memory(image)?;
        Ok(fit_to_target_area(
            img.width().max(1),
            img.height().max(1),
            target_area,
            16,
        ))
    }

    fn pack_latents_4d(latents: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = latents.dims4()?;
        let height_blocks = height / 2;
        let width_blocks = width / 2;
        latents
            .reshape((batch, channels, height_blocks, 2, width_blocks, 2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((batch, height_blocks * width_blocks, channels * 4))
            .map_err(Into::into)
    }

    fn unpack_latents_packed(latents: &Tensor, latent_h: usize, latent_w: usize) -> Result<Tensor> {
        let batch = latents.dim(0)?;
        latents
            .reshape((batch, latent_h / 2, latent_w / 2, 16, 2, 2))?
            .permute((0, 3, 1, 4, 2, 5))?
            .reshape((batch, 16, latent_h, latent_w))
            .map_err(Into::into)
    }

    fn img2img_source_normalize_range() -> img_utils::NormalizeRange {
        img_utils::NormalizeRange::MinusOneToOne
    }

    fn is_oom_error(err: &impl std::fmt::Display) -> bool {
        // TODO: Replace this with typed backend inspection if candle exposes
        // one. Today the fallback ladder has to key off the backend error text.
        let msg = err.to_string();
        msg.contains("OUT_OF_MEMORY")
            || msg.contains("out of memory")
            || msg.contains("cudaErrorMemoryAllocation")
    }

    fn with_cuda_oom_cpu_fallback<T, FPrimary, FFallback, FOom>(
        primary: FPrimary,
        fallback: FFallback,
        is_cuda: bool,
        sync_device: &Device,
        progress: &ProgressReporter,
        oom_message: &str,
        is_oom: FOom,
    ) -> Result<T>
    where
        FPrimary: FnOnce() -> Result<T>,
        FFallback: FnOnce() -> Result<T>,
        FOom: Fn(&anyhow::Error) -> bool,
    {
        match primary() {
            Ok(value) => Ok(value),
            Err(err) if is_cuda && is_oom(&err) => {
                progress.info(oom_message);
                sync_device.synchronize()?;
                fallback()
            }
            Err(err) => Err(err),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn with_cuda_tiled_then_cpu_fallback<T, FPrimary, FTiled, FCpu, FOom>(
        primary: FPrimary,
        tiled: FTiled,
        cpu_fallback: FCpu,
        is_cuda: bool,
        sync_device: &Device,
        progress: &ProgressReporter,
        tiled_message: &str,
        cpu_message: &str,
        is_oom: FOom,
    ) -> Result<T>
    where
        FPrimary: FnOnce() -> Result<T>,
        FTiled: FnOnce() -> Result<T>,
        FCpu: FnOnce() -> Result<T>,
        FOom: Fn(&anyhow::Error) -> bool,
    {
        match primary() {
            Ok(value) => Ok(value),
            Err(err) if is_cuda && is_oom(&err) => {
                progress.info(tiled_message);
                sync_device.synchronize()?;
                match tiled() {
                    Ok(value) => Ok(value),
                    Err(tile_err) if is_oom(&tile_err) => {
                        progress.info(cpu_message);
                        sync_device.synchronize()?;
                        cpu_fallback()
                    }
                    Err(tile_err) => Err(tile_err),
                }
            }
            Err(err) => Err(err),
        }
    }

    fn decode_vae_tiled(
        latents: &Tensor,
        vae: &QwenImageVae,
        vae_device: &Device,
        progress: &ProgressReporter,
    ) -> Result<Tensor> {
        for tile_size in QWEN_VAE_TILE_SIZES {
            let overlap = (tile_size / 4).max(4);
            progress.info(&format!(
                "Retrying VAE decode with tiled GPU decode (tile {} overlap {})",
                tile_size, overlap
            ));
            let config = TilingConfig {
                tile_size,
                overlap,
                min_tile_size: 16,
            };
            let forward = |tile: &Tensor| {
                let tile = tile.to_device(vae_device)?.to_dtype(DType::F32)?;
                vae.decode(&tile).map_err(Into::into)
            };
            // `upscale_with_tiling` is reused here because Qwen-Image VAE decode
            // is guaranteed to return 3-channel RGB. If a future VAE family
            // changes that contract, this call site needs a tiler that handles
            // arbitrary output channel counts.
            match upscale_with_tiling(latents, &forward, 8, &config, &Device::Cpu, progress) {
                Ok(image) => return Ok(image),
                Err(e) if vae_device.is_cuda() && Self::is_oom_error(&e) => {
                    if let Err(sync_err) = vae_device.synchronize() {
                        tracing::warn!(
                            "failed to synchronize CUDA device after tiled VAE OOM: {sync_err}"
                        );
                    }
                }
                Err(e) => return Err(e),
            }
        }

        bail!("tiled VAE decode still ran out of memory")
    }

    fn decode_vae_with_fallback<F>(
        latents: &Tensor,
        vae: &QwenImageVae,
        vae_device: &Device,
        sync_device: &Device,
        progress: &ProgressReporter,
        load_cpu_vae: F,
    ) -> Result<Tensor>
    where
        F: FnOnce() -> Result<QwenImageVae>,
    {
        let decode_latents = latents.to_device(vae_device)?.to_dtype(DType::F32)?;
        Self::debug_tensor_stats("latents_pre_vae", &decode_latents);
        Self::with_cuda_tiled_then_cpu_fallback(
            || vae.decode(&decode_latents).map_err(Into::into),
            || Self::decode_vae_tiled(latents, vae, vae_device, progress),
            || {
                let cpu_vae = load_cpu_vae()?;
                let cpu_latents = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                cpu_vae.decode(&cpu_latents).map_err(Into::into)
            },
            vae_device.is_cuda(),
            sync_device,
            progress,
            "VAE decode OOM on GPU — retrying with tiled GPU decode",
            "Tiled GPU VAE decode OOM — retrying on CPU",
            Self::is_oom_error,
        )
    }

    /// Encode a source image through the VAE with GPU→CPU OOM fallback.
    #[allow(clippy::too_many_arguments)]
    fn encode_vae_with_fallback(
        source_bytes: &[u8],
        width: u32,
        height: u32,
        vae: &QwenImageVae,
        vae_device: &Device,
        sync_device: &Device,
        progress: &ProgressReporter,
        load_cpu_vae: impl FnOnce() -> Result<QwenImageVae>,
    ) -> Result<Tensor> {
        progress.stage_start("Encoding source image (VAE)");
        let encode_start = Instant::now();

        // Qwen-Image VAE expects [-1, 1] normalized pixels
        let source_tensor = img_utils::decode_source_image(
            source_bytes,
            width,
            height,
            Self::img2img_source_normalize_range(),
            vae_device,
            DType::F32,
        )?;

        let result = Self::with_cuda_oom_cpu_fallback(
            || vae.encode(&source_tensor).map_err(Into::into),
            || {
                let cpu_vae = load_cpu_vae()?;
                let cpu_source = img_utils::decode_source_image(
                    source_bytes,
                    width,
                    height,
                    Self::img2img_source_normalize_range(),
                    &Device::Cpu,
                    DType::F32,
                )?;
                cpu_vae.encode(&cpu_source).map_err(Into::into)
            },
            vae_device.is_cuda(),
            sync_device,
            progress,
            "VAE encode OOM on GPU — retrying on CPU",
            Self::is_oom_error,
        );

        progress.stage_done("Encoding source image (VAE)", encode_start.elapsed());
        result
    }

    fn choose_text_encoder_source(
        preference: Option<&str>,
        is_cuda: bool,
        is_metal: bool,
        free_vram: u64,
        bf16_size_bytes: u64,
        usage: Qwen2TextEncoderUsage,
    ) -> Result<ResolvedQwen2TextEncoder> {
        match preference {
            Some(tag) if tag != "auto" && tag != "bf16" => {
                let variant = mold_core::manifest::find_qwen2_vl_variant(tag).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown Qwen2.5-VL variant '{}'. Valid: bf16, auto, q8, q6, q5, q4, q3, q2",
                        tag
                    )
                })?;
                Ok(ResolvedQwen2TextEncoder {
                    paths: vec![],
                    vision_paths: vec![],
                    is_gguf: true,
                    variant_label: variant.tag.to_string(),
                    size_bytes: variant.size_bytes,
                    auto_use_gpu: should_use_gpu(
                        is_cuda,
                        is_metal,
                        free_vram,
                        qwen2_vram_threshold(variant.size_bytes),
                    ),
                })
            }
            Some("bf16") => Ok(ResolvedQwen2TextEncoder {
                paths: vec![],
                vision_paths: vec![],
                is_gguf: false,
                variant_label: "bf16".to_string(),
                size_bytes: bf16_size_bytes,
                auto_use_gpu: should_use_gpu(
                    is_cuda,
                    is_metal,
                    free_vram,
                    QWEN2_FP16_VRAM_THRESHOLD,
                ),
            }),
            _ if is_metal => {
                for tag in ["q6", "q4"] {
                    let variant = mold_core::manifest::find_qwen2_vl_variant(tag)
                        .expect("known Metal auto qwen2 variant missing");
                    if fits_in_memory(
                        is_cuda,
                        is_metal,
                        free_vram,
                        qwen2_vram_threshold(variant.size_bytes),
                    ) {
                        return Ok(ResolvedQwen2TextEncoder {
                            paths: vec![],
                            vision_paths: vec![],
                            is_gguf: true,
                            variant_label: variant.tag.to_string(),
                            size_bytes: variant.size_bytes,
                            auto_use_gpu: true,
                        });
                    }
                }
                let fallback = mold_core::manifest::find_qwen2_vl_variant("q4")
                    .expect("known Metal fallback qwen2 variant missing");
                Ok(ResolvedQwen2TextEncoder {
                    paths: vec![],
                    vision_paths: vec![],
                    is_gguf: true,
                    variant_label: fallback.tag.to_string(),
                    size_bytes: fallback.size_bytes,
                    auto_use_gpu: true,
                })
            }
            _ => {
                let bf16_on_gpu =
                    should_use_gpu(is_cuda, is_metal, free_vram, QWEN2_FP16_VRAM_THRESHOLD);
                if bf16_on_gpu {
                    return Ok(ResolvedQwen2TextEncoder {
                        paths: vec![],
                        vision_paths: vec![],
                        is_gguf: false,
                        variant_label: "bf16".to_string(),
                        size_bytes: bf16_size_bytes,
                        auto_use_gpu: true,
                    });
                }

                if is_cuda {
                    if matches!(usage, Qwen2TextEncoderUsage::Sequential) {
                        return Ok(ResolvedQwen2TextEncoder {
                            paths: vec![],
                            vision_paths: vec![],
                            is_gguf: false,
                            variant_label: "bf16".to_string(),
                            size_bytes: bf16_size_bytes,
                            auto_use_gpu: false,
                        });
                    }

                    let fallback_tag = "q4";
                    let fallback = mold_core::manifest::find_qwen2_vl_variant(fallback_tag)
                        .expect("known CUDA fallback qwen2 variant missing");
                    return Ok(ResolvedQwen2TextEncoder {
                        paths: vec![],
                        vision_paths: vec![],
                        is_gguf: true,
                        variant_label: fallback.tag.to_string(),
                        size_bytes: fallback.size_bytes,
                        auto_use_gpu: matches!(usage, Qwen2TextEncoderUsage::Resident)
                            && fits_in_memory(
                                is_cuda,
                                is_metal,
                                free_vram,
                                qwen2_vram_threshold(fallback.size_bytes),
                            ),
                    });
                }

                Ok(ResolvedQwen2TextEncoder {
                    paths: vec![],
                    vision_paths: vec![],
                    is_gguf: false,
                    variant_label: "bf16".to_string(),
                    size_bytes: bf16_size_bytes,
                    auto_use_gpu: false,
                })
            }
        }
    }

    fn debug_tensor_stats(name: &str, tensor: &Tensor) {
        if std::env::var_os("MOLD_QWEN_DEBUG").is_none() {
            return;
        }
        let stats = || -> Result<String> {
            let t = tensor.to_dtype(DType::F32)?;
            let min = t.min_all()?.to_scalar::<f32>()?;
            let max = t.max_all()?.to_scalar::<f32>()?;
            let mean = t.mean_all()?.to_scalar::<f32>()?;
            // NaN detection: x != x is true for NaN (IEEE 754)
            let nan_count = t
                .ne(&t)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()? as u64;
            let total = t.elem_count();
            if nan_count > 0 {
                Ok(format!(
                    "[qwen-debug] {name}: min={min:.4} max={max:.4} mean={mean:.4} NaN={nan_count}/{total} ({:.1}%)",
                    nan_count as f64 / total as f64 * 100.0
                ))
            } else {
                Ok(format!(
                    "[qwen-debug] {name}: min={min:.4} max={max:.4} mean={mean:.4}"
                ))
            }
        };
        match stats() {
            Ok(msg) => eprintln!("{msg}"),
            Err(err) => eprintln!("[qwen-debug] {name}: <failed: {err}>"),
        }
    }

    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        offload: bool,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy),
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            offload,
        }
    }

    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedPromptConditioning>>,
        text_encoder: &mut encoders::qwen2_text::Qwen2TextEncoder,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cache_key = prompt_text_key(prompt);
        if let Some(cached) = prompt_cache
            .lock()
            .expect("cache poisoned")
            .get_cloned(&cache_key)
        {
            progress.cache_hit("prompt conditioning");
            return cached.restore(device, dtype);
        }

        progress.stage_start("Encoding prompt (Qwen2.5)");
        let encode_start = Instant::now();
        let (hidden_states, _attention_mask, valid_len) =
            text_encoder.encode(prompt, device, dtype)?;
        progress.stage_done("Encoding prompt (Qwen2.5)", encode_start.elapsed());

        prompt_cache.lock().expect("cache poisoned").insert(
            cache_key,
            CachedPromptConditioning::from_parts(&hidden_states, valid_len)?,
        );

        let mut mask = vec![0u8; hidden_states.dim(1)?];
        for value in &mut mask[..valid_len] {
            *value = 1;
        }
        let attention_mask = Tensor::from_vec(mask, (1, hidden_states.dim(1)?), device)?;
        Ok((hidden_states, attention_mask))
    }

    fn spill_conditioning_to_cpu(
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Ok((
            hidden_states
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?,
            attention_mask.to_device(&Device::Cpu)?,
        ))
    }

    fn maybe_spill_conditioning(
        use_cpu_staging: bool,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if use_cpu_staging {
            Self::spill_conditioning_to_cpu(hidden_states, attention_mask)
        } else {
            Ok((hidden_states, attention_mask))
        }
    }

    /// Resolve transformer shard paths.
    fn transformer_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.base.paths.transformer_shards.is_empty() {
            self.base.paths.transformer_shards.clone()
        } else {
            vec![self.base.paths.transformer.clone()]
        }
    }

    fn detect_is_quantized(&self) -> bool {
        self.base
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
    }

    /// Validate required paths exist.
    fn validate_paths(&self) -> Result<std::path::PathBuf> {
        let text_tokenizer_path =
            self.base.paths.text_tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("text tokenizer path required for Qwen-Image models")
            })?;
        if !text_tokenizer_path.exists() {
            bail!(
                "text tokenizer file not found: {}",
                text_tokenizer_path.display()
            );
        }

        let xformer_paths = self.transformer_paths();
        for path in &xformer_paths {
            if !path.exists() {
                bail!("transformer file not found: {}", path.display());
            }
        }
        if !self.base.paths.vae.exists() {
            bail!("VAE file not found: {}", self.base.paths.vae.display());
        }

        Ok(text_tokenizer_path.clone())
    }

    fn quantized_cuda_cfg_headroom(width: usize, height: usize) -> u64 {
        let native_pixels = (QWEN_NATIVE_WIDTH * QWEN_NATIVE_HEIGHT) as f64;
        let pixels = (width.max(1) * height.max(1)) as f64;
        let scaled =
            (QWEN_GGUF_NATIVE_CFG_HEADROOM as f64 * (pixels / native_pixels)).round() as u64;
        scaled.max(QWEN_GGUF_MIN_CFG_HEADROOM)
    }

    fn should_split_cfg_quantized_cuda(
        transformer_size: u64,
        free_vram: u64,
        width: usize,
        height: usize,
    ) -> bool {
        if free_vram == 0 {
            // If VRAM probing fails, bias toward the safer split-CFG path
            // instead of assuming batched CFG will fit.
            return true;
        }
        let estimated_peak =
            transformer_size.saturating_add(Self::quantized_cuda_cfg_headroom(width, height));
        estimated_peak > free_vram
    }

    /// Load transformer from disk.
    fn load_transformer(
        &self,
        device: &Device,
        dtype: DType,
        cfg: &QwenImageConfig,
        width: usize,
        height: usize,
    ) -> Result<QwenImageTransformer> {
        if self.detect_is_quantized() {
            let transformer_size = std::fs::metadata(&self.base.paths.transformer)
                .map(|m| m.len())
                .unwrap_or(0);
            let free = free_vram_bytes().unwrap_or(0);
            let split_cfg_for_memory = device.is_cuda()
                && (self.offload
                    || Self::should_split_cfg_quantized_cuda(
                        transformer_size,
                        free,
                        width,
                        height,
                    ));
            if self.offload && device.is_cuda() {
                self.base.progress.info(
                    "Quantized Qwen CUDA offload requested — using low-memory split-CFG mode until GGUF block offload lands",
                );
            } else if split_cfg_for_memory {
                let estimated_peak = transformer_size
                    .saturating_add(Self::quantized_cuda_cfg_headroom(width, height));
                self.base.progress.info(&format!(
                    "Using low-memory quantized Qwen CUDA path (est. peak {}, {} free at {}x{})",
                    fmt_gb(estimated_peak),
                    fmt_gb(free),
                    width,
                    height,
                ));
            }
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, device)?;
            Ok(QwenImageTransformer::Quantized(
                QuantizedQwenImageTransformer2DModel::new(cfg, vb, device, !split_cfg_for_memory)?,
            ))
        } else {
            let xformer_paths = self.transformer_paths();
            let is_fp8 = xformer_paths
                .first()
                .map(|p| safetensors_is_fp8(p))
                .unwrap_or(false);

            // FP8 weights stay as F8E4M3 in VRAM (~19.5GB, 1 byte/param).
            // Per-layer dequant to BF16 during forward adds ~113MB transient.
            // BF16 weights are 2 bytes/param (~40GB).
            let mem_size: u64 = xformer_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            let free = free_vram_bytes().unwrap_or(0);
            let use_offload = self.offload || crate::device::should_offload(mem_size, free);

            if is_fp8 {
                self.base
                    .progress
                    .info("Detected FP8 safetensors — loading with scale dequantization");
            }

            if use_offload {
                // Create TWO VarBuilders: GPU for blocks that fit, CPU for overflow.
                let (gpu_vb, cpu_vb) = if is_fp8 {
                    let gpu = crate::weight_loader::load_fp8_safetensors(
                        &xformer_paths,
                        device,
                        "Qwen-Image transformer (offload, GPU)",
                        &self.base.progress,
                    )?;
                    let cpu = crate::weight_loader::load_fp8_safetensors(
                        &xformer_paths,
                        &Device::Cpu,
                        "Qwen-Image transformer (offload, CPU)",
                        &self.base.progress,
                    )?;
                    (gpu, cpu)
                } else {
                    let gpu = crate::weight_loader::load_safetensors_with_progress(
                        &xformer_paths,
                        dtype,
                        device,
                        "Qwen-Image transformer (offload, GPU)",
                        &self.base.progress,
                    )?;
                    let cpu = unsafe {
                        candle_nn::VarBuilder::from_mmaped_safetensors(
                            &xformer_paths
                                .iter()
                                .map(|p| p.as_path())
                                .collect::<Vec<_>>(),
                            DType::BF16,
                            &Device::Cpu,
                        )?
                    };
                    (gpu, cpu)
                };
                Ok(QwenImageTransformer::Offloaded(
                    super::offload::OffloadedQwenImageTransformer::load(
                        gpu_vb,
                        cpu_vb,
                        cfg,
                        device,
                        &self.base.progress,
                    )?,
                ))
            } else {
                let xformer_vb = if is_fp8 {
                    crate::weight_loader::load_fp8_safetensors(
                        &xformer_paths,
                        device,
                        "Qwen-Image transformer",
                        &self.base.progress,
                    )?
                } else {
                    crate::weight_loader::load_safetensors_with_progress(
                        &xformer_paths,
                        dtype,
                        device,
                        "Qwen-Image transformer",
                        &self.base.progress,
                    )?
                };
                Ok(QwenImageTransformer::BF16(
                    QwenImageTransformer2DModel::new(cfg, xformer_vb)?,
                ))
            }
        }
    }

    /// Load VAE from disk.
    fn load_vae(&self, device: &Device, dtype: DType) -> Result<QwenImageVae> {
        Ok(QwenImageVae::load(
            &self.base.paths.vae,
            device,
            dtype,
            &self.base.progress,
        )?)
    }

    /// Load text encoder from disk.
    ///
    /// FP8 text encoders are loaded on GPU with BF16 dtype — candle's CUDA cast
    /// kernel handles F8E4M3→BF16 conversion during tensor loading.
    fn resolve_text_encoder_source(
        &self,
        gpu_device: &Device,
        free_vram: u64,
        usage: Qwen2TextEncoderUsage,
    ) -> Result<ResolvedQwen2TextEncoder> {
        let preference = std::env::var("MOLD_QWEN2_VARIANT").ok();
        self.resolve_text_encoder_source_with_preference(
            gpu_device,
            free_vram,
            usage,
            preference.as_deref(),
        )
    }

    fn resolve_text_encoder_source_with_preference(
        &self,
        gpu_device: &Device,
        free_vram: u64,
        usage: Qwen2TextEncoderUsage,
        preference: Option<&str>,
    ) -> Result<ResolvedQwen2TextEncoder> {
        let is_cuda = gpu_device.is_cuda();
        let is_metal = gpu_device.is_metal();
        let bf16_size_bytes = self
            .base
            .paths
            .text_encoder_files
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        if self.is_edit_family() {
            let mut resolved = Self::choose_text_encoder_source(
                preference,
                is_cuda,
                is_metal,
                free_vram,
                bf16_size_bytes,
                Qwen2TextEncoderUsage::Resident,
            )?;
            resolved.vision_paths = self.base.paths.text_encoder_files.clone();
            if resolved.is_gguf {
                let variant = mold_core::manifest::find_qwen2_vl_variant(&resolved.variant_label)
                    .ok_or_else(|| {
                    anyhow::anyhow!("unknown Qwen2.5-VL variant '{}'", resolved.variant_label)
                })?;
                resolved.paths = vec![
                    crate::encoders::variant_resolution::resolve_qwen2_vl_gguf_path(
                        &self.base.progress,
                        variant,
                    )?,
                ];
            } else {
                resolved.paths = self.base.paths.text_encoder_files.clone();
            }
            return Ok(resolved);
        }
        let mut resolved = Self::choose_text_encoder_source(
            preference,
            is_cuda,
            is_metal,
            free_vram,
            bf16_size_bytes,
            usage,
        )?;

        if resolved.is_gguf {
            let variant = mold_core::manifest::find_qwen2_vl_variant(&resolved.variant_label)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown Qwen2.5-VL variant '{}'", resolved.variant_label)
                })?;
            resolved.paths = vec![
                crate::encoders::variant_resolution::resolve_qwen2_vl_gguf_path(
                    &self.base.progress,
                    variant,
                )?,
            ];
        } else {
            resolved.paths = self.base.paths.text_encoder_files.clone();
        }
        resolved.vision_paths = vec![];

        match preference {
            Some(tag) if tag != "auto" && tag != "bf16" => self.base.progress.info(&format!(
                "Using quantized Qwen2.5-VL {} ({}) on {} (explicit)",
                resolved.variant_label,
                fmt_gb(resolved.size_bytes),
                if resolved.auto_use_gpu { "GPU" } else { "CPU" },
            )),
            Some("bf16") => {}
            _ if is_metal && resolved.is_gguf && resolved.variant_label == "q6" => self
                .base
                .progress
                .info(&format!(
                    "Metal auto mode selected quantized Qwen2.5-VL {} ({}) for lower memory pressure",
                    resolved.variant_label,
                    fmt_gb(resolved.size_bytes),
                )),
            _ if is_metal && resolved.is_gguf => self.base.progress.info(&format!(
                "Metal auto mode forcing quantized Qwen2.5-VL {} ({}) to avoid BF16 memory pressure",
                resolved.variant_label,
                fmt_gb(resolved.size_bytes),
            )),
            _ if is_cuda && resolved.is_gguf && resolved.auto_use_gpu => self.base.progress.info(
                &format!(
                    "CUDA auto mode selected quantized Qwen2.5-VL {} ({}) on GPU",
                    resolved.variant_label,
                    fmt_gb(resolved.size_bytes),
                ),
            ),
            _ if is_cuda && resolved.is_gguf => self.base.progress.info(&format!(
                "CUDA auto mode selected quantized Qwen2.5-VL {} ({}) on CPU to avoid large BF16 host residency",
                resolved.variant_label,
                fmt_gb(resolved.size_bytes),
            )),
            _ => {}
        }

        Ok(resolved)
    }

    fn can_keep_transformer_hot_for_vae(loaded: &LoadedQwenImage) -> bool {
        loaded.device.is_cuda()
            && loaded.vae_device.is_cuda()
            && matches!(
                loaded.transformer.as_ref(),
                Some(QwenImageTransformer::Quantized(_))
            )
    }

    fn decode_vae_gpu_only(
        latents: &Tensor,
        vae: &QwenImageVae,
        vae_device: &Device,
        sync_device: &Device,
        progress: &ProgressReporter,
    ) -> Result<Tensor> {
        let decode_latents = latents.to_device(vae_device)?.to_dtype(DType::F32)?;
        match vae.decode(&decode_latents) {
            Ok(image) => Ok(image),
            Err(e) if vae_device.is_cuda() && Self::is_oom_error(&e) => {
                progress.info(
                    "Resident-transformer VAE decode OOM on GPU — retrying with tiled GPU decode before dropping transformer",
                );
                sync_device.synchronize()?;
                Self::decode_vae_tiled(latents, vae, vae_device, progress)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn load_text_encoder(
        &self,
        resolved: &ResolvedQwen2TextEncoder,
        tokenizer_path: &std::path::PathBuf,
        device: &Device,
        dtype: DType,
        preload_weights: bool,
    ) -> Result<encoders::qwen2_text::Qwen2TextEncoder> {
        if resolved.is_gguf {
            if preload_weights {
                encoders::qwen2_text::Qwen2TextEncoder::load_gguf(
                    &resolved.paths[0],
                    tokenizer_path,
                    device,
                    dtype,
                    &resolved.vision_paths,
                    &self.base.progress,
                )
            } else {
                encoders::qwen2_text::Qwen2TextEncoder::prepare_gguf(
                    &resolved.paths[0],
                    tokenizer_path,
                    device,
                    dtype,
                    &resolved.vision_paths,
                )
            }
        } else {
            let is_fp8 = text_encoder_is_fp8(&resolved.paths);
            if is_fp8 {
                self.base
                    .progress
                    .info("Detected FP8 text encoder — loading as BF16 on GPU");
            }
            if preload_weights {
                encoders::qwen2_text::Qwen2TextEncoder::load_bf16(
                    &resolved.paths,
                    tokenizer_path,
                    device,
                    dtype,
                    self.is_edit_family(),
                    &self.base.progress,
                )
            } else {
                encoders::qwen2_text::Qwen2TextEncoder::prepare_bf16(
                    &resolved.paths,
                    tokenizer_path,
                    device,
                    dtype,
                    self.is_edit_family(),
                )
            }
        }
    }

    /// Resolve text encoder device placement and optional CPU staging.
    fn resolve_text_encoder_plan(
        &self,
        gpu_device: &Device,
        resolved: &ResolvedQwen2TextEncoder,
        free_vram: u64,
    ) -> (Qwen2TextEncoderPlan, String) {
        let is_cuda = gpu_device.is_cuda();
        let is_metal = gpu_device.is_metal();
        let plan = Self::qwen2_text_encoder_plan_for_mode(
            Qwen2TextEncoderMode::from_env(),
            is_cuda,
            is_metal,
            resolved,
        );
        let label = if plan.use_gpu { "GPU" } else { "CPU" };
        if plan.use_cpu_staging {
            self.base
                .progress
                .info("Qwen2.5 text encoder on GPU with CPU staging after encoding");
        } else if !plan.use_gpu {
            if resolved.is_gguf {
                self.base.progress.info(&format!(
                    "Qwen2.5 text encoder on CPU ({} variant {}, {} free)",
                    resolved.variant_label,
                    fmt_gb(resolved.size_bytes),
                    fmt_gb(free_vram),
                ));
            } else if is_metal || is_cuda {
                self.base.progress.info(&format!(
                    "Qwen2.5 text encoder on CPU ({} free < {} threshold)",
                    fmt_gb(free_vram),
                    fmt_gb(QWEN2_FP16_VRAM_THRESHOLD),
                ));
            }
        }
        (plan, label.to_string())
    }

    fn qwen2_text_encoder_plan_for_mode(
        mode: Qwen2TextEncoderMode,
        is_cuda: bool,
        is_metal: bool,
        resolved: &ResolvedQwen2TextEncoder,
    ) -> Qwen2TextEncoderPlan {
        match mode {
            Qwen2TextEncoderMode::Gpu => Qwen2TextEncoderPlan {
                use_gpu: is_cuda || is_metal,
                use_cpu_staging: false,
            },
            Qwen2TextEncoderMode::CpuStage => Qwen2TextEncoderPlan {
                use_gpu: is_cuda || is_metal,
                use_cpu_staging: is_cuda || is_metal,
            },
            Qwen2TextEncoderMode::Cpu => Qwen2TextEncoderPlan {
                use_gpu: false,
                use_cpu_staging: false,
            },
            Qwen2TextEncoderMode::Auto => Qwen2TextEncoderPlan {
                use_gpu: resolved.auto_use_gpu,
                use_cpu_staging: is_metal && resolved.auto_use_gpu && !resolved.is_gguf,
            },
        }
    }

    /// Load all model components (Eager mode).
    ///
    /// On error, `self.base.loaded` remains `None` — all components are assembled into
    /// local variables and only stored in `self.base.loaded` on success, so partial loads
    /// cannot leave the engine in an inconsistent state.
    pub fn load(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }

        // Sequential mode defers loading to generate_sequential()
        if self.base.load_strategy == LoadStrategy::Sequential {
            return Ok(());
        }

        tracing::info!(model = %self.base.model_name, "loading Qwen-Image model components...");

        let text_tokenizer_path = self.validate_paths()?;
        let device = crate::device::create_device(&self.base.progress)?;
        let transformer_cfg = self.transformer_config();
        let transformer_is_quantized = self.detect_is_quantized();
        // FP8 safetensors are loaded as BF16 via CPU (candle CUDA kernel bug
        // prevents direct F8E4M3→BF16 on GPU; CPU cast works fine). All paths
        // use BF16 as runtime dtype since the model trains and computes in BF16.
        let dtype = crate::engine::gpu_dtype(&device);

        // Load transformer
        let xformer_paths = self.transformer_paths();
        let xformer_label = if transformer_is_quantized {
            "Loading Qwen-Image transformer (quantized)".to_string()
        } else {
            format!(
                "Loading Qwen-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(
            &device,
            dtype,
            &transformer_cfg,
            QWEN_NATIVE_WIDTH,
            QWEN_NATIVE_HEIGHT,
        )?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!("Qwen-Image transformer loaded");

        // Decide device placement for VAE and text encoder
        let free = free_vram_bytes().unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free > 0 {
            self.base
                .progress
                .info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
        }

        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_DECODE_VRAM_THRESHOLD);
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        // Always decode in F32 — BF16 convolutions accumulate quantization noise across
        // the 4 upsampling blocks, producing visible grain. Matches diffusers' force_upcast.
        let vae_dtype = DType::F32;
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        // Load VAE
        let vae_label = format!("Loading Qwen-Image VAE ({}, F32)", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        // Load text encoder
        let resolved_text_encoder =
            self.resolve_text_encoder_source(&device, free, Qwen2TextEncoderUsage::Resident)?;
        let (te_plan, te_device_label) =
            self.resolve_text_encoder_plan(&device, &resolved_text_encoder, free);
        let te_device = if te_plan.use_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_dtype = Self::text_encoder_load_dtype(te_plan.use_gpu, dtype);

        let preload_text_encoder = self.should_preload_text_encoder();
        let te_label = if resolved_text_encoder.is_gguf {
            if preload_text_encoder {
                format!(
                    "Loading Qwen2.5 text encoder ({} GGUF, {})",
                    resolved_text_encoder.variant_label, te_device_label
                )
            } else {
                format!(
                    "Preparing Qwen2.5 text encoder ({} GGUF, {})",
                    resolved_text_encoder.variant_label, te_device_label
                )
            }
        } else if preload_text_encoder {
            format!(
                "Loading Qwen2.5 text encoder ({} shards, {})",
                resolved_text_encoder.paths.len(),
                te_device_label,
            )
        } else {
            format!(
                "Preparing Qwen2.5 text encoder ({} shards, {})",
                resolved_text_encoder.paths.len(),
                te_device_label,
            )
        };
        self.base.progress.stage_start(&te_label);
        let te_start = Instant::now();
        let text_encoder = self.load_text_encoder(
            &resolved_text_encoder,
            &text_tokenizer_path,
            &te_device,
            te_dtype,
            preload_text_encoder,
        )?;
        self.base.progress.stage_done(&te_label, te_start.elapsed());
        if preload_text_encoder {
            tracing::info!(device = %te_device_label, "Qwen2.5 text encoder loaded");
        } else {
            tracing::info!(device = %te_device_label, "Qwen2.5 text encoder prepared for staged loading");
        }

        self.base.loaded = Some(LoadedQwenImage {
            transformer: Some(transformer),
            text_encoder,
            vae,
            vae_path: self.base.paths.vae.clone(),
            transformer_cfg,
            device,
            vae_device,
            dtype,
        });

        tracing::info!(model = %self.base.model_name, "all Qwen-Image components loaded");
        Ok(())
    }

    /// Reload the transformer from disk.
    fn reload_transformer(
        &self,
        loaded: &mut LoadedQwenImage,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let transformer = self.load_transformer(
            &loaded.device,
            loaded.dtype,
            &loaded.transformer_cfg,
            width,
            height,
        )?;
        loaded.transformer = Some(transformer);
        Ok(())
    }

    /// Generate using sequential loading strategy (load-use-drop each component).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let transformer_cfg = self.transformer_config();

        let device = crate::device::create_device(&self.base.progress)?;
        let dtype = crate::engine::gpu_dtype(&device);
        let transformer_is_quantized = self.detect_is_quantized();

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        let free = free_vram_bytes().unwrap_or(0);
        let resolved_text_encoder =
            self.resolve_text_encoder_source(&device, free, Qwen2TextEncoderUsage::Sequential)?;
        let (plan, _device_label) =
            self.resolve_text_encoder_plan(&device, &resolved_text_encoder, free);
        let use_cpu_staging = plan.use_cpu_staging;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting sequential Qwen-Image generation"
        );

        self.base
            .progress
            .info("Using sequential loading (load-use-drop) to minimize peak memory");

        // --- Phase 1: Text encoding (check cache first to skip encoder load) ---
        let use_cfg = req.guidance > 1.0;
        let prompt_key = prompt_text_key(&req.prompt);
        let uncond_key = prompt_text_key(QWEN_EMPTY_NEGATIVE_PROMPT);
        let (prompt_cached, uncond_cached) = {
            let mut cache = self.prompt_cache.lock().expect("cache poisoned");
            let prompt_cached = cache.get_cloned(&prompt_key);
            let uncond_cached = if use_cfg {
                cache.get_cloned(&uncond_key)
            } else {
                None
            };
            (prompt_cached, uncond_cached)
        };
        let both_cached = prompt_cached.is_some() && (!use_cfg || uncond_cached.is_some());

        let (mut encoder_hidden_states, mut encoder_attention_mask, mut uncond_hs, mut uncond_mask) =
            if both_cached {
                self.base.progress.cache_hit("prompt conditioning");
                let cached = prompt_cached.unwrap();
                let restore_device = if use_cpu_staging {
                    &Device::Cpu
                } else {
                    &device
                };
                let restore_dtype = if use_cpu_staging { DType::F32 } else { dtype };
                let (hs, mask) = cached.restore(restore_device, restore_dtype)?;
                let (u_hs, u_mask) = if use_cfg {
                    let ucached = uncond_cached.unwrap();
                    let (u_hs, u_mask) = ucached.restore(restore_device, restore_dtype)?;
                    (Some(u_hs), Some(u_mask))
                } else {
                    (None, None)
                };
                (hs, mask, u_hs, u_mask)
            } else {
                let (te_plan, te_device_label) =
                    self.resolve_text_encoder_plan(&device, &resolved_text_encoder, free);
                let te_device = if te_plan.use_gpu {
                    device.clone()
                } else {
                    Device::Cpu
                };
                let te_dtype = Self::text_encoder_load_dtype(te_plan.use_gpu, dtype);

                let te_label = if resolved_text_encoder.is_gguf {
                    format!(
                        "Loading Qwen2.5 text encoder ({} GGUF, {})",
                        resolved_text_encoder.variant_label, te_device_label
                    )
                } else {
                    format!(
                        "Loading Qwen2.5 text encoder ({} shards, {})",
                        resolved_text_encoder.paths.len(),
                        te_device_label,
                    )
                };
                if te_plan.use_cpu_staging && device.is_metal() && !resolved_text_encoder.is_gguf {
                    self.base.progress.info(
                        "Skipping hard preflight for Qwen2.5 text encoder on Metal; sequential mode spills prompt conditioning to CPU after encoding",
                    );
                } else {
                    preflight_memory_check(
                        "Qwen2.5 text encoder",
                        resolved_text_encoder.size_bytes,
                    )?;
                }

                if let Some(status) = memory_status_string() {
                    self.base.progress.info(&status);
                }

                self.base.progress.stage_start(&te_label);
                let te_start = Instant::now();
                let mut text_encoder = self.load_text_encoder(
                    &resolved_text_encoder,
                    &text_tokenizer_path,
                    &te_device,
                    te_dtype,
                    true,
                )?;
                self.base.progress.stage_done(&te_label, te_start.elapsed());

                let (hs, mask) = Self::encode_prompt_cached(
                    &self.base.progress,
                    &self.prompt_cache,
                    &mut text_encoder,
                    &req.prompt,
                    &device,
                    dtype,
                )?;
                let (hs, mask) = Self::maybe_spill_conditioning(use_cpu_staging, hs, mask)?;

                let (u_hs, u_mask) = if use_cfg {
                    let (hs, mask) = Self::encode_prompt_cached(
                        &self.base.progress,
                        &self.prompt_cache,
                        &mut text_encoder,
                        QWEN_EMPTY_NEGATIVE_PROMPT,
                        &device,
                        dtype,
                    )?;
                    let (hs, mask) = Self::maybe_spill_conditioning(use_cpu_staging, hs, mask)?;
                    (Some(hs), Some(mask))
                } else {
                    (None, None)
                };

                drop(text_encoder);
                // Force the backend to release allocator state before transformer load.
                device.synchronize()?;
                if let Some(status) = crate::device::memory_status_string() {
                    if use_cpu_staging {
                        self.base.progress.info(&format!(
                            "Freed Qwen2.5 text encoder and spilled prompt conditioning to CPU — {status}"
                        ));
                    } else {
                        self.base
                            .progress
                            .info(&format!("Freed Qwen2.5 text encoder — {status}"));
                    }
                } else {
                    if use_cpu_staging {
                        self.base.progress.info(
                            "Freed Qwen2.5 text encoder and spilled prompt conditioning to CPU",
                        );
                    } else {
                        self.base.progress.info("Freed Qwen2.5 text encoder");
                    }
                }

                (hs, mask, u_hs, u_mask)
            };

        if use_cfg {
            let ((cond_hs, cond_mask), (neg_hs, neg_mask)) = align_cfg_conditioning(
                &encoder_hidden_states,
                &encoder_attention_mask,
                uncond_hs.as_ref().expect("unconditional prompt missing"),
                uncond_mask.as_ref().expect("unconditional mask missing"),
            )?;
            encoder_hidden_states = cond_hs;
            encoder_attention_mask = cond_mask;
            uncond_hs = Some(neg_hs);
            uncond_mask = Some(neg_mask);
        }

        // --- Phase 2: Load transformer and denoise ---
        let xformer_paths = self.transformer_paths();
        let xformer_size: u64 = xformer_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        preflight_memory_check("Qwen-Image transformer", xformer_size)?;

        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let xformer_label = if transformer_is_quantized {
            "Loading Qwen-Image transformer (quantized)".to_string()
        } else {
            format!(
                "Loading Qwen-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.base.progress.stage_start(&xformer_label);
        let xformer_start = Instant::now();
        let transformer = self.load_transformer(&device, dtype, &transformer_cfg, width, height)?;
        self.base
            .progress
            .stage_done(&xformer_label, xformer_start.elapsed());

        if use_cpu_staging {
            encoder_hidden_states = encoder_hidden_states.to_device(&device)?.to_dtype(dtype)?;
            encoder_attention_mask = encoder_attention_mask.to_device(&device)?;
            if let Some(hs) = uncond_hs.take() {
                uncond_hs = Some(hs.to_device(&device)?.to_dtype(dtype)?);
            }
            if let Some(mask) = uncond_mask.take() {
                uncond_mask = Some(mask.to_device(&device)?);
            }
            if let Some(status) = memory_status_string() {
                self.base.progress.info(&format!(
                    "Restored prompt conditioning to GPU for denoising — {status}"
                ));
            } else {
                self.base
                    .progress
                    .info("Restored prompt conditioning to GPU for denoising");
            }
        }

        // Calculate latent dimensions: image_size / 8 (VAE downsample factor)
        let vae_downsample = 8;
        let latent_h = height / vae_downsample;
        let latent_w = width / vae_downsample;
        let is_img2img = req.source_image.is_some();

        // For img2img, load VAE early to encode source image before transformer
        let (prepared_img2img_latents, inpaint_ctx) = if let Some(ref source_bytes) =
            req.source_image
        {
            let free_for_encode = free_vram_bytes().unwrap_or(0);
            let encode_on_gpu = should_use_gpu(
                device.is_cuda(),
                device.is_metal(),
                free_for_encode,
                VAE_DECODE_VRAM_THRESHOLD,
            );
            let encode_device = if encode_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let encode_label = if encode_on_gpu { "GPU" } else { "CPU" };

            let vae_label = format!("Loading Qwen-Image VAE ({}, F32) for encode", encode_label);
            self.base.progress.stage_start(&vae_label);
            let vae_start = Instant::now();
            let encode_vae = self.load_vae(&encode_device, DType::F32)?;
            self.base
                .progress
                .stage_done(&vae_label, vae_start.elapsed());

            let encoded = Self::encode_vae_with_fallback(
                source_bytes,
                req.width,
                req.height,
                &encode_vae,
                &encode_device,
                &device,
                &self.base.progress,
                || self.load_vae(&Device::Cpu, DType::F32),
            )?;
            let encoded = encoded.to_device(&device)?.to_dtype(dtype)?;
            let start_sigma = QwenImageScheduler::new_img2img(
                req.steps as usize,
                image_seq_len(latent_h, latent_w, transformer_cfg.patch_size),
                req.strength,
            )
            .0
            .initial_sigma();
            let prepared = crate::img2img::prepare_flow_match_img2img(
                &encoded,
                seed,
                &[1, 16, latent_h, latent_w],
                start_sigma,
                req.mask_image.as_deref(),
                latent_h,
                latent_w,
                &device,
                dtype,
            )?;

            // Drop early VAE to free memory before transformer load
            drop(encode_vae);
            device.synchronize()?;

            tracing::info!(
                strength = req.strength,
                "img2img: encoded source image to latents"
            );

            (Some(prepared.initial_latents), prepared.inpaint_ctx)
        } else {
            (None, None)
        };

        let image_seq_len = image_seq_len(latent_h, latent_w, transformer_cfg.patch_size);
        let (mut scheduler, num_steps) = if is_img2img {
            QwenImageScheduler::new_img2img(req.steps as usize, image_seq_len, req.strength)
        } else {
            let sched = QwenImageScheduler::new(req.steps as usize, image_seq_len);
            let n = sched.num_steps();
            (sched, n)
        };

        // Build initial latents
        let mut latents = if let Some(initial) = &prepared_img2img_latents {
            initial.clone()
        } else {
            let noise =
                crate::engine::seeded_randn(seed, &[1, 16, latent_h, latent_w], &device, dtype)?;
            (noise * scheduler.initial_sigma())?
        };

        let denoise_label = format!("Denoising ({} steps)", num_steps);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
            eprintln!(
                "[qwen-debug] cfg={} guidance={:.1} image_seq_len={} sigmas[0]={:.4} sigmas[last]={:.4} img2img={}",
                use_cfg,
                req.guidance,
                image_seq_len,
                scheduler.sigmas[0],
                scheduler.sigmas[scheduler.sigmas.len() - 1],
                is_img2img,
            );
        }

        let use_batched_cfg = use_cfg && transformer.supports_cfg_batching();
        if use_cfg && !use_batched_cfg {
            self.base.progress.info(
                "Low-memory quantized Qwen CUDA path detected — disabling CFG batching to reduce peak CUDA memory",
            );
        }

        // Pre-batch CFG inputs when the selected transformer path can handle the
        // extra batch dimension without exceeding peak memory.
        let (batched_hs, batched_mask) = if use_batched_cfg {
            let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
            let mask = Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
            (hs, mask)
        } else {
            (
                encoder_hidden_states.clone(),
                encoder_attention_mask.clone(),
            )
        };

        for step in 0..num_steps {
            let step_start = Instant::now();
            let t = scheduler.current_timestep();
            let noise_pred = if use_cfg {
                let (cond_pred, uncond_pred) = if use_batched_cfg {
                    let t_tensor =
                        Tensor::from_vec(vec![t as f32; 2], (2,), &device)?.to_dtype(dtype)?;
                    let batched_latents = Tensor::cat(&[&latents, &latents], 0)?;
                    let batched_pred = transformer.forward(
                        &batched_latents,
                        &t_tensor,
                        &batched_hs,
                        &batched_mask,
                    )?;
                    (batched_pred.narrow(0, 0, 1)?, batched_pred.narrow(0, 1, 1)?)
                } else {
                    let t_tensor =
                        Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
                    (
                        transformer.forward(
                            &latents,
                            &t_tensor,
                            &encoder_hidden_states,
                            &encoder_attention_mask,
                        )?,
                        transformer.forward(
                            &latents,
                            &t_tensor,
                            uncond_hs.as_ref().unwrap(),
                            uncond_mask.as_ref().unwrap(),
                        )?,
                    )
                };
                if step == 0 {
                    Self::debug_tensor_stats("cond_pred[0]", &cond_pred);
                    Self::debug_tensor_stats("uncond_pred[0]", &uncond_pred);
                }
                // CFG in F32 to avoid BF16 cancellation error, then norm rescale
                // to match diffusers' Qwen-Image pipeline.
                let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                let comb = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * req.guidance)?)?;
                let cond_norm = cond_f32.sqr()?.sum_keepdim(1)?.sqrt()?;
                let comb_norm = comb.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-8, f64::MAX)?;
                let rescaled = comb.broadcast_mul(&(cond_norm / comb_norm)?)?;
                rescaled.to_dtype(dtype)?
            } else {
                let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
                transformer.forward(
                    &latents,
                    &t_tensor,
                    &encoder_hidden_states,
                    &encoder_attention_mask,
                )?
            };
            if step == 0 || step == num_steps / 2 || step == num_steps - 1 {
                Self::debug_tensor_stats(&format!("noise_pred[{step}]"), &noise_pred);
                Self::debug_tensor_stats(&format!("latents[{step}]"), &latents);
            }
            latents = scheduler.step(&noise_pred, &latents)?;

            // Inpainting: blend preserved regions back at current noise level
            if let Some(ref ctx) = inpaint_ctx {
                latents = crate::img2img::apply_flow_match_inpaint(
                    &latents,
                    ctx,
                    scheduler.sigmas[step + 1],
                )?;
            }

            if std::env::var_os("MOLD_QWEN_DEBUG").is_some() {
                let n = latents
                    .ne(&latents)?
                    .to_dtype(candle_core::DType::U32)?
                    .sum_all()?
                    .to_scalar::<u32>()?;
                if n > 0 {
                    eprintln!(
                        "[qwen-nan] NaN in latents AFTER step {step}: {n}/{}",
                        latents.elem_count()
                    );
                }
            }
            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total: num_steps,
                elapsed: step_start.elapsed(),
            });
        }

        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer and embeddings
        drop(transformer);
        drop(encoder_hidden_states);
        drop(encoder_attention_mask);
        drop(uncond_hs);
        drop(uncond_mask);
        device.synchronize()?;
        self.base.progress.info("Freed Qwen-Image transformer");

        // --- Phase 3: Load VAE and decode ---
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        let free_for_vae = free_vram_bytes().unwrap_or(0);
        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_vae,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        // Always decode in F32 — BF16 convolutions accumulate quantization noise across
        // the 4 upsampling blocks, producing visible grain. Matches diffusers' force_upcast.
        let vae_dtype = DType::F32;
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        let vae_label = format!("Loading Qwen-Image VAE ({}, F32)", vae_device_label);
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = self.load_vae(&vae_device, vae_dtype)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        self.base.progress.stage_start("VAE decode");
        let vae_decode_start = Instant::now();

        let image = Self::decode_vae_with_fallback(
            &latents,
            &vae,
            &vae_device,
            &device,
            &self.base.progress,
            || self.load_vae(&Device::Cpu, DType::F32),
        )?;
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;

        self.base
            .progress
            .stage_done("VAE decode", vae_decode_start.elapsed());

        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            generation_time_ms,
            seed,
            "sequential Qwen-Image generation complete"
        );

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
        })
    }

    fn generate_edit_loaded(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let progress = &self.base.progress;
        let start = Instant::now();

        let loaded_ref = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let needs_reload = loaded_ref.transformer.is_none();
        if needs_reload {
            let mut loaded_mut = self
                .base
                .loaded
                .take()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
            progress.stage_start("Reloading Qwen-Image transformer");
            let reload_start = Instant::now();
            self.reload_transformer(&mut loaded_mut, req.width as usize, req.height as usize)?;
            progress.stage_done("Reloading Qwen-Image transformer", reload_start.elapsed());
            self.base.loaded = Some(loaded_mut);
        }

        let is_edit_family = self.is_edit_family();
        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let seed = req.seed.unwrap_or_else(rand_seed);
        let width = req.width as usize;
        let height = req.height as usize;
        let edit_images = req
            .edit_images
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qwen-image-edit requires edit_images"))?;
        let use_cfg = req.guidance > 1.0;
        let negative_prompt = req
            .negative_prompt
            .as_deref()
            .unwrap_or(QWEN_EMPTY_NEGATIVE_PROMPT);
        let formatted_prompt = Self::qwen_image_edit_prompt(&req.prompt, edit_images.len());
        let formatted_negative = Self::qwen_image_edit_prompt(negative_prompt, edit_images.len());

        tracing::info!(
            prompt = %req.prompt,
            seed,
            width,
            height,
            steps = req.steps,
            edit_images = edit_images.len(),
            "starting Qwen-Image edit generation"
        );

        if loaded.text_encoder.model.is_none() {
            progress.stage_start("Reloading Qwen2.5 encoder");
            let reload_start = Instant::now();
            loaded.text_encoder.reload(progress)?;
            progress.stage_done("Reloading Qwen2.5 encoder", reload_start.elapsed());
        }

        progress.stage_start("Encoding prompt (Qwen2.5 edit)");
        let encode_start = Instant::now();
        let (encoder_hidden_states, encoder_attention_mask, _) =
            loaded.text_encoder.encode_formatted_multimodal(
                &formatted_prompt,
                edit_images,
                &loaded.device,
                loaded.dtype,
            )?;
        progress.stage_done("Encoding prompt (Qwen2.5 edit)", encode_start.elapsed());
        let (encoder_hidden_states, encoder_attention_mask, uncond_hs, uncond_mask) = if use_cfg {
            progress.stage_start("Encoding negative prompt (Qwen2.5 edit)");
            let neg_start = Instant::now();
            let (hs, mask, _) = loaded.text_encoder.encode_formatted_multimodal(
                &formatted_negative,
                edit_images,
                &loaded.device,
                loaded.dtype,
            )?;
            progress.stage_done(
                "Encoding negative prompt (Qwen2.5 edit)",
                neg_start.elapsed(),
            );
            let ((cond_hs, cond_mask), (neg_hs, neg_mask)) = align_cfg_conditioning(
                &encoder_hidden_states,
                &encoder_attention_mask,
                &hs,
                &mask,
            )?;
            (cond_hs, cond_mask, Some(neg_hs), Some(neg_mask))
        } else {
            (encoder_hidden_states, encoder_attention_mask, None, None)
        };

        let drop_text_encoder = is_edit_family || loaded.text_encoder.on_gpu;
        if drop_text_encoder {
            loaded.text_encoder.drop_weights();
            tracing::info!(
                on_gpu = loaded.text_encoder.on_gpu,
                "Qwen2.5 text encoder dropped after edit conditioning"
            );
        }

        let mut packed_input_storage = Vec::with_capacity(edit_images.len());
        let mut img_shapes = vec![(1usize, height / 16, width / 16)];
        progress.stage_start("Encoding edit images (VAE)");
        let encode_start = Instant::now();
        for image_bytes in edit_images {
            let (vae_width, vae_height) =
                Self::qwen_image_edit_image_dims(image_bytes, QWEN_IMAGE_EDIT_VAE_AREA)?;
            let encoded = Self::encode_vae_with_fallback(
                image_bytes,
                vae_width,
                vae_height,
                &loaded.vae,
                &loaded.vae_device,
                &loaded.device,
                progress,
                || {
                    Ok(QwenImageVae::load(
                        &loaded.vae_path,
                        &Device::Cpu,
                        DType::F32,
                        progress,
                    )?)
                },
            )?
            .to_device(&loaded.device)?
            .to_dtype(loaded.dtype)?;
            img_shapes.push((1, encoded.dim(2)? / 2, encoded.dim(3)? / 2));
            packed_input_storage.push(Self::pack_latents_4d(&encoded)?);
        }
        progress.stage_done("Encoding edit images (VAE)", encode_start.elapsed());

        let packed_inputs = if packed_input_storage.is_empty() {
            None
        } else {
            let tensors = packed_input_storage.iter().collect::<Vec<_>>();
            Some(Tensor::cat(&tensors, 1)?)
        };

        let noise = crate::engine::seeded_randn(
            seed,
            &[1, 16, height / 8, width / 8],
            &loaded.device,
            loaded.dtype,
        )?;
        let mut scheduler =
            QwenImageScheduler::new(req.steps as usize, (height / 16) * (width / 16));
        let num_steps = scheduler.num_steps();
        let mut latents = Self::pack_latents_4d(&(noise * scheduler.initial_sigma())?)?;
        let output_seq_len = latents.dim(1)?;

        let denoise_label = format!("Denoising edit ({} steps)", num_steps);
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");
            let use_batched_cfg = use_cfg && transformer.supports_cfg_batching();
            let (batched_hs, batched_mask) = if use_batched_cfg {
                let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
                let mask =
                    Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
                (hs, mask)
            } else {
                (
                    encoder_hidden_states.clone(),
                    encoder_attention_mask.clone(),
                )
            };

            for step in 0..num_steps {
                let step_start = Instant::now();
                let t = scheduler.current_timestep();
                let timestep = if use_batched_cfg {
                    Tensor::from_vec(vec![t as f32; 2], (2,), &loaded.device)?
                        .to_dtype(loaded.dtype)?
                } else {
                    Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                        .to_dtype(loaded.dtype)?
                };

                let latent_model_input = if let Some(ref packed_inputs) = packed_inputs {
                    Tensor::cat(&[&latents, packed_inputs], 1)?
                } else {
                    latents.clone()
                };

                let noise_pred = if use_cfg {
                    let (cond_pred, uncond_pred) = if use_batched_cfg {
                        let batched_input =
                            Tensor::cat(&[&latent_model_input, &latent_model_input], 0)?;
                        let pred = transformer.forward_packed(
                            &batched_input,
                            &timestep,
                            &batched_hs,
                            &batched_mask,
                            &img_shapes,
                        )?;
                        (
                            pred.narrow(0, 0, 1)?.narrow(1, 0, output_seq_len)?,
                            pred.narrow(0, 1, 1)?.narrow(1, 0, output_seq_len)?,
                        )
                    } else {
                        (
                            transformer
                                .forward_packed(
                                    &latent_model_input,
                                    &timestep,
                                    &encoder_hidden_states,
                                    &encoder_attention_mask,
                                    &img_shapes,
                                )?
                                .narrow(1, 0, output_seq_len)?,
                            transformer
                                .forward_packed(
                                    &latent_model_input,
                                    &timestep,
                                    uncond_hs.as_ref().unwrap(),
                                    uncond_mask.as_ref().unwrap(),
                                    &img_shapes,
                                )?
                                .narrow(1, 0, output_seq_len)?,
                        )
                    };

                    let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                    let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                    let comb = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * req.guidance)?)?;
                    let cond_norm = cond_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
                    let comb_norm = comb
                        .sqr()?
                        .sum_keepdim(D::Minus1)?
                        .sqrt()?
                        .clamp(1e-8, f64::MAX)?;
                    comb.broadcast_mul(&(cond_norm / comb_norm)?)?
                        .to_dtype(loaded.dtype)?
                } else {
                    transformer
                        .forward_packed(
                            &latent_model_input,
                            &timestep,
                            &encoder_hidden_states,
                            &encoder_attention_mask,
                            &img_shapes,
                        )?
                        .narrow(1, 0, output_seq_len)?
                };

                latents = scheduler.step(&noise_pred, &latents)?;
                progress.emit(ProgressEvent::DenoiseStep {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        progress.stage_done(&denoise_label, denoise_start.elapsed());

        let latents = Self::unpack_latents_packed(&latents, height / 8, width / 8)?;
        let image = Self::decode_vae_with_fallback(
            &latents,
            &loaded.vae,
            &loaded.vae_device,
            &loaded.device,
            progress,
            || {
                Ok(QwenImageVae::load(
                    &loaded.vae_path,
                    &Device::Cpu,
                    DType::F32,
                    progress,
                )?)
            },
        )?;
        let image = postprocess_image(&image)?.i(0)?;
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
        })
    }
}

impl InferenceEngine for QwenImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if req.scheduler.is_some() {
            tracing::warn!(
                "scheduler selection not supported for Qwen-Image (flow-matching), ignoring"
            );
        }

        if self.is_edit_family() {
            let sequential = self.base.load_strategy == LoadStrategy::Sequential;
            if sequential && self.base.loaded.is_none() {
                let original = self.base.load_strategy;
                self.base.load_strategy = LoadStrategy::Eager;
                let load_result = self.load();
                self.base.load_strategy = original;
                load_result?;
            }
            if self.base.loaded.is_none() {
                bail!("model not loaded -- call load() first");
            }
            let result = self.generate_edit_loaded(req);
            if sequential {
                self.unload();
            }
            return result;
        }

        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.base.loaded.is_none() {
            bail!("model not loaded -- call load() first");
        }

        let progress = &self.base.progress;
        let start = Instant::now();

        // Reload transformer if it was dropped after previous VAE decode
        let loaded_ref = self
            .base
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let needs_reload = loaded_ref.transformer.is_none();
        if needs_reload {
            let mut loaded_mut = self
                .base
                .loaded
                .take()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
            progress.stage_start("Reloading Qwen-Image transformer");
            let reload_start = Instant::now();
            self.reload_transformer(&mut loaded_mut, req.width as usize, req.height as usize)?;
            progress.stage_done("Reloading Qwen-Image transformer", reload_start.elapsed());
            self.base.loaded = Some(loaded_mut);
        }

        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Qwen-Image generation"
        );

        let use_cfg = req.guidance > 1.0;
        let prompt_key = prompt_text_key(&req.prompt);
        let uncond_key = prompt_text_key(QWEN_EMPTY_NEGATIVE_PROMPT);
        let prompt_cached = self
            .prompt_cache
            .lock()
            .expect("cache poisoned")
            .get_cloned(&prompt_key);
        let uncond_cached = if use_cfg {
            self.prompt_cache
                .lock()
                .expect("cache poisoned")
                .get_cloned(&uncond_key)
        } else {
            None
        };
        let both_cached = prompt_cached.is_some() && (!use_cfg || uncond_cached.is_some());

        let (encoder_hidden_states, encoder_attention_mask, uncond_hs, uncond_mask) = if both_cached
        {
            let cached = prompt_cached.expect("prompt cache unexpectedly missing");
            progress.cache_hit("prompt conditioning");
            let (hs, mask) = cached.restore(&loaded.device, loaded.dtype)?;
            let (u_hs, u_mask) = if use_cfg {
                progress.cache_hit("unconditional conditioning");
                let ucached =
                    uncond_cached.expect("unconditional prompt cache unexpectedly missing");
                let (u_hs, u_mask) = ucached.restore(&loaded.device, loaded.dtype)?;
                (Some(u_hs), Some(u_mask))
            } else {
                (None, None)
            };
            (hs, mask, u_hs, u_mask)
        } else {
            if loaded.text_encoder.model.is_none() {
                progress.stage_start("Reloading Qwen2.5 encoder");
                let reload_start = Instant::now();
                loaded.text_encoder.reload(progress)?;
                progress.stage_done("Reloading Qwen2.5 encoder", reload_start.elapsed());
            }

            let (hs, mask) = Self::encode_prompt_cached(
                progress,
                &self.prompt_cache,
                &mut loaded.text_encoder,
                &req.prompt,
                &loaded.device,
                loaded.dtype,
            )?;

            let (u_hs, u_mask) = if use_cfg {
                let (hs, mask) = Self::encode_prompt_cached(
                    progress,
                    &self.prompt_cache,
                    &mut loaded.text_encoder,
                    QWEN_EMPTY_NEGATIVE_PROMPT,
                    &loaded.device,
                    loaded.dtype,
                )?;
                (Some(hs), Some(mask))
            } else {
                (None, None)
            };

            (hs, mask, u_hs, u_mask)
        };

        let (encoder_hidden_states, encoder_attention_mask, uncond_hs, uncond_mask) = if use_cfg {
            let ((cond_hs, cond_mask), (neg_hs, neg_mask)) = align_cfg_conditioning(
                &encoder_hidden_states,
                &encoder_attention_mask,
                uncond_hs.as_ref().expect("unconditional prompt missing"),
                uncond_mask.as_ref().expect("unconditional mask missing"),
            )?;
            (cond_hs, cond_mask, Some(neg_hs), Some(neg_mask))
        } else {
            (
                encoder_hidden_states,
                encoder_attention_mask,
                uncond_hs,
                uncond_mask,
            )
        };

        // Drop text encoder from GPU to free VRAM for denoising
        if loaded.text_encoder.on_gpu {
            loaded.text_encoder.drop_weights();
            tracing::info!("Qwen2.5 text encoder dropped from GPU");
        }

        // 3. Calculate latent dimensions
        let vae_downsample = 8;
        let latent_h = height / vae_downsample;
        let latent_w = width / vae_downsample;
        let is_img2img = req.source_image.is_some();

        // For img2img, encode source image using the pre-loaded VAE
        let (prepared_img2img_latents, inpaint_ctx) =
            if let Some(ref source_bytes) = req.source_image {
                let encoded = Self::encode_vae_with_fallback(
                    source_bytes,
                    req.width,
                    req.height,
                    &loaded.vae,
                    &loaded.vae_device,
                    &loaded.device,
                    progress,
                    || {
                        Ok(QwenImageVae::load(
                            &loaded.vae_path,
                            &Device::Cpu,
                            DType::F32,
                            progress,
                        )?)
                    },
                )?;
                let encoded = encoded.to_device(&loaded.device)?.to_dtype(loaded.dtype)?;
                let start_sigma = QwenImageScheduler::new_img2img(
                    req.steps as usize,
                    image_seq_len(latent_h, latent_w, loaded.transformer_cfg.patch_size),
                    req.strength,
                )
                .0
                .initial_sigma();
                let prepared = crate::img2img::prepare_flow_match_img2img(
                    &encoded,
                    seed,
                    &[1, 16, latent_h, latent_w],
                    start_sigma,
                    req.mask_image.as_deref(),
                    latent_h,
                    latent_w,
                    &loaded.device,
                    loaded.dtype,
                )?;

                (Some(prepared.initial_latents), prepared.inpaint_ctx)
            } else {
                (None, None)
            };

        // 4. Initialize scheduler
        let image_seq_len = image_seq_len(latent_h, latent_w, loaded.transformer_cfg.patch_size);
        let (mut scheduler, num_steps) = if is_img2img {
            QwenImageScheduler::new_img2img(req.steps as usize, image_seq_len, req.strength)
        } else {
            let sched = QwenImageScheduler::new(req.steps as usize, image_seq_len);
            let n = sched.num_steps();
            (sched, n)
        };

        // 5. Build initial latents
        let mut latents = if let Some(initial) = &prepared_img2img_latents {
            initial.clone()
        } else {
            let noise = crate::engine::seeded_randn(
                seed,
                &[1, 16, latent_h, latent_w],
                &loaded.device,
                loaded.dtype,
            )?;
            (noise * scheduler.initial_sigma())?
        };

        // 7. Denoising loop
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            let use_batched_cfg = use_cfg && transformer.supports_cfg_batching();
            if use_cfg && !use_batched_cfg {
                progress.info(
                    "Low-memory quantized Qwen CUDA path detected — disabling CFG batching to reduce peak CUDA memory",
                );
            }

            // Pre-batch CFG inputs when the selected transformer path can handle
            // the extra batch dimension without exceeding peak memory.
            let (batched_hs, batched_mask) = if use_batched_cfg {
                let hs = Tensor::cat(&[&encoder_hidden_states, uncond_hs.as_ref().unwrap()], 0)?;
                let mask =
                    Tensor::cat(&[&encoder_attention_mask, uncond_mask.as_ref().unwrap()], 0)?;
                (hs, mask)
            } else {
                (
                    encoder_hidden_states.clone(),
                    encoder_attention_mask.clone(),
                )
            };

            for step in 0..num_steps {
                let step_start = Instant::now();
                let t = scheduler.current_timestep();
                let noise_pred = if use_cfg {
                    let (cond_pred, uncond_pred) = if use_batched_cfg {
                        let t_tensor = Tensor::from_vec(vec![t as f32; 2], (2,), &loaded.device)?
                            .to_dtype(loaded.dtype)?;
                        let batched_latents = Tensor::cat(&[&latents, &latents], 0)?;
                        let batched_pred = transformer.forward(
                            &batched_latents,
                            &t_tensor,
                            &batched_hs,
                            &batched_mask,
                        )?;
                        (batched_pred.narrow(0, 0, 1)?, batched_pred.narrow(0, 1, 1)?)
                    } else {
                        let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                            .to_dtype(loaded.dtype)?;
                        (
                            transformer.forward(
                                &latents,
                                &t_tensor,
                                &encoder_hidden_states,
                                &encoder_attention_mask,
                            )?,
                            transformer.forward(
                                &latents,
                                &t_tensor,
                                uncond_hs.as_ref().unwrap(),
                                uncond_mask.as_ref().unwrap(),
                            )?,
                        )
                    };
                    // CFG in F32 + norm rescale (matches diffusers Qwen-Image pipeline)
                    let cond_f32 = cond_pred.to_dtype(DType::F32)?;
                    let uncond_f32 = uncond_pred.to_dtype(DType::F32)?;
                    let comb = (&uncond_f32 + ((&cond_f32 - &uncond_f32)? * req.guidance)?)?;
                    let cond_norm = cond_f32.sqr()?.sum_keepdim(1)?.sqrt()?;
                    let comb_norm = comb.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-8, f64::MAX)?;
                    let rescaled = comb.broadcast_mul(&(cond_norm / comb_norm)?)?;
                    rescaled.to_dtype(loaded.dtype)?
                } else {
                    let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                        .to_dtype(loaded.dtype)?;
                    transformer.forward(
                        &latents,
                        &t_tensor,
                        &encoder_hidden_states,
                        &encoder_attention_mask,
                    )?
                };
                if step == 0 {
                    Self::debug_tensor_stats("noise_pred", &noise_pred);
                }
                latents = scheduler.step(&noise_pred, &latents)?;

                // Inpainting: blend preserved regions back at current noise level
                if let Some(ref ctx) = inpaint_ctx {
                    latents = crate::img2img::apply_flow_match_inpaint(
                        &latents,
                        ctx,
                        scheduler.sigmas[step + 1],
                    )?;
                }

                progress.emit(ProgressEvent::DenoiseStep {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        progress.stage_done(&denoise_label, denoise_start.elapsed());

        // Free text embeddings
        drop(encoder_hidden_states);
        drop(encoder_attention_mask);
        drop(uncond_hs);
        drop(uncond_mask);

        // 8. VAE decode
        progress.stage_start("VAE decode");
        let vae_start = Instant::now();

        // Always decode in F32 — matches sequential path and diffusers' force_upcast.
        let keep_transformer_hot = Self::can_keep_transformer_hot_for_vae(loaded);
        let image = if keep_transformer_hot {
            match Self::decode_vae_gpu_only(
                &latents,
                &loaded.vae,
                &loaded.vae_device,
                &loaded.device,
                progress,
            ) {
                Ok(image) => {
                    progress.info(
                        "Kept quantized Qwen transformer resident across VAE decode for faster hot-path reuse",
                    );
                    image
                }
                Err(err) if Self::is_oom_error(&err) => {
                    loaded.transformer = None;
                    loaded.device.synchronize()?;
                    progress.info(
                        "Dropping Qwen-Image transformer after resident VAE decode OOM and retrying",
                    );
                    Self::decode_vae_with_fallback(
                        &latents,
                        &loaded.vae,
                        &loaded.vae_device,
                        &loaded.device,
                        progress,
                        || {
                            QwenImageVae::load(&loaded.vae_path, &Device::Cpu, DType::F32, progress)
                                .map_err(Into::into)
                        },
                    )?
                }
                Err(err) => return Err(err),
            }
        } else {
            loaded.transformer = None;
            loaded.device.synchronize()?;
            tracing::info!("Qwen-Image transformer dropped to free VRAM for VAE decode");
            Self::decode_vae_with_fallback(
                &latents,
                &loaded.vae,
                &loaded.vae_device,
                &loaded.device,
                progress,
                || {
                    QwenImageVae::load(&loaded.vae_path, &Device::Cpu, DType::F32, progress)
                        .map_err(Into::into)
                },
            )?
        };
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;

        progress.stage_done("VAE decode", vae_start.elapsed());

        // 9. Encode to output format
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.output_format,
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Qwen-Image generation complete");

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: image_bytes,
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
        })
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        QwenImageEngine::load(self)
    }

    fn unload(&mut self) {
        self.base.unload();
        clear_cache(&self.prompt_cache);
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LoadStrategy;
    use candle_core::Shape;
    use mold_core::ModelPaths;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, b"test").unwrap();
        path
    }

    fn png_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |_, _| image::Rgb([255, 0, 0]));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn qwen_image_model_paths(
        transformer: PathBuf,
        transformer_shards: Vec<PathBuf>,
        vae: PathBuf,
        text_tokenizer: Option<PathBuf>,
    ) -> ModelPaths {
        ModelPaths {
            transformer,
            transformer_shards,
            vae,
            spatial_upscaler: None,
            temporal_upscaler: None,
            distilled_lora: None,
            t5_encoder: None,
            clip_encoder: None,
            t5_tokenizer: None,
            clip_tokenizer: None,
            clip_encoder_2: None,
            clip_tokenizer_2: None,
            text_encoder_files: vec![],
            text_tokenizer,
            decoder: None,
        }
    }

    fn resolved_text_encoder(is_gguf: bool, auto_use_gpu: bool) -> ResolvedQwen2TextEncoder {
        ResolvedQwen2TextEncoder {
            paths: vec![],
            vision_paths: vec![],
            is_gguf,
            variant_label: if is_gguf {
                "q6".to_string()
            } else {
                "bf16".to_string()
            },
            size_bytes: 0,
            auto_use_gpu,
        }
    }

    fn tensor_values_u8(t: &Tensor) -> Vec<u8> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .expect("u8 tensor values")
    }

    fn tensor_values_f32(t: &Tensor) -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .expect("f32 tensor values")
    }

    #[test]
    fn safetensors_is_fp8_uses_filename_hint() {
        assert!(safetensors_is_fp8(Path::new(
            "/tmp/qwen-image-fp8.safetensors"
        )));
        assert!(!safetensors_is_fp8(Path::new(
            "/tmp/qwen-image.safetensors"
        )));
    }

    #[test]
    fn text_encoder_is_fp8_uses_filename_hint() {
        assert!(text_encoder_is_fp8(&[PathBuf::from(
            "/tmp/qwen2-text-encoder-fp8-00001-of-00002.safetensors"
        )]));
        assert!(!text_encoder_is_fp8(&[PathBuf::from(
            "/tmp/qwen2-text-encoder-00001-of-00002.safetensors"
        )]));
    }

    #[test]
    fn cached_prompt_conditioning_roundtrips_and_restores_mask() {
        let device = Device::Cpu;
        let hidden_states = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let cached = CachedPromptConditioning::from_parts(&hidden_states, 2).unwrap();

        let (restored_hs, restored_mask) = cached.restore(&device, DType::F32).unwrap();

        assert_eq!(
            tensor_values_f32(&restored_hs),
            tensor_values_f32(&hidden_states)
        );
        assert_eq!(tensor_values_u8(&restored_mask), vec![1, 1, 0]);
    }

    #[test]
    fn pad_text_conditioning_keeps_original_when_target_matches() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 1], Shape::from((1, 2)), &device).unwrap();

        let (padded_hs, padded_mask) = pad_text_conditioning(&hidden_states, &mask, 2).unwrap();

        assert_eq!(
            tensor_values_f32(&padded_hs),
            tensor_values_f32(&hidden_states)
        );
        assert_eq!(tensor_values_u8(&padded_mask), vec![1, 1]);
    }

    #[test]
    fn pad_text_conditioning_appends_zero_padding() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 0], Shape::from((1, 2)), &device).unwrap();

        let (padded_hs, padded_mask) = pad_text_conditioning(&hidden_states, &mask, 4).unwrap();

        assert_eq!(padded_hs.dims3().unwrap(), (1, 4, 2));
        assert_eq!(
            tensor_values_f32(&padded_hs),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(tensor_values_u8(&padded_mask), vec![1, 0, 0, 0]);
    }

    #[test]
    fn pad_text_conditioning_rejects_shrinking() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 1], Shape::from((1, 2)), &device).unwrap();

        let err = pad_text_conditioning(&hidden_states, &mask, 1).unwrap_err();
        assert!(err.to_string().contains("cannot shrink text conditioning"));
    }

    #[test]
    fn align_cfg_conditioning_pads_shorter_branch_to_match_longer_one() {
        let device = Device::Cpu;
        let cond_hs = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let cond_mask = Tensor::from_vec(vec![1u8, 1, 1], Shape::from((1, 3)), &device).unwrap();
        let uncond_hs = Tensor::from_vec(
            vec![7.0f32, 8.0, 9.0, 10.0],
            Shape::from((1, 2, 2)),
            &device,
        )
        .unwrap();
        let uncond_mask = Tensor::from_vec(vec![1u8, 0], Shape::from((1, 2)), &device).unwrap();

        let ((cond_hs, cond_mask), (uncond_hs, uncond_mask)) =
            align_cfg_conditioning(&cond_hs, &cond_mask, &uncond_hs, &uncond_mask).unwrap();

        assert_eq!(cond_hs.dims3().unwrap(), (1, 3, 2));
        assert_eq!(uncond_hs.dims3().unwrap(), (1, 3, 2));
        assert_eq!(tensor_values_u8(&cond_mask), vec![1, 1, 1]);
        assert_eq!(tensor_values_u8(&uncond_mask), vec![1, 0, 0]);
        assert_eq!(
            tensor_values_f32(&uncond_hs),
            vec![7.0, 8.0, 9.0, 10.0, 0.0, 0.0]
        );
    }

    #[test]
    fn qwen_image_detects_gguf_transformer() {
        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            ModelPaths {
                transformer: PathBuf::from("/tmp/qwen-image-Q4_K_S.gguf"),
                transformer_shards: vec![],
                vae: PathBuf::from("/tmp/vae.safetensors"),
                spatial_upscaler: None,
                temporal_upscaler: None,
                distilled_lora: None,
                t5_encoder: None,
                clip_encoder: None,
                t5_tokenizer: None,
                clip_tokenizer: None,
                clip_encoder_2: None,
                clip_tokenizer_2: None,
                text_encoder_files: vec![],
                text_tokenizer: Some(PathBuf::from("/tmp/tokenizer.json")),
                decoder: None,
            },
            LoadStrategy::Sequential,
            false,
        );

        assert!(engine.detect_is_quantized());
    }

    #[test]
    fn qwen_image_text_encoder_uses_gpu_on_metal() {
        let plan = QwenImageEngine::qwen2_text_encoder_plan_for_mode(
            Qwen2TextEncoderMode::Auto,
            false,
            true,
            &resolved_text_encoder(true, true),
        );
        assert!(plan.use_gpu);
        assert!(!plan.use_cpu_staging);
    }

    #[test]
    fn qwen_image_text_encoder_uses_gpu_on_cuda_with_headroom() {
        let plan = QwenImageEngine::qwen2_text_encoder_plan_for_mode(
            Qwen2TextEncoderMode::Auto,
            true,
            false,
            &resolved_text_encoder(false, true),
        );
        assert!(plan.use_gpu);
        assert!(!plan.use_cpu_staging);
    }

    #[test]
    fn qwen_image_text_encoder_uses_cpu_on_cuda_without_headroom() {
        let plan = QwenImageEngine::qwen2_text_encoder_plan_for_mode(
            Qwen2TextEncoderMode::Auto,
            true,
            false,
            &resolved_text_encoder(false, false),
        );
        assert!(!plan.use_gpu);
        assert!(!plan.use_cpu_staging);
    }

    #[test]
    fn qwen_image_cpu_safetensors_text_encoder_stays_f32() {
        assert_eq!(
            QwenImageEngine::text_encoder_load_dtype(false, DType::BF16),
            DType::F32
        );
    }

    #[test]
    fn qwen_image_cpu_gguf_text_encoder_stays_f32() {
        assert_eq!(
            QwenImageEngine::text_encoder_load_dtype(false, DType::BF16),
            DType::F32
        );
    }

    #[test]
    fn qwen_image_text_encoder_gpu_override_disables_metal_staging() {
        let plan = QwenImageEngine::qwen2_text_encoder_plan_for_mode(
            Qwen2TextEncoderMode::Gpu,
            false,
            true,
            &resolved_text_encoder(true, true),
        );
        assert!(plan.use_gpu);
        assert!(!plan.use_cpu_staging);
    }

    #[test]
    fn qwen_image_auto_prefers_q6_on_metal_with_headroom() {
        let q6 = mold_core::manifest::find_qwen2_vl_variant("q6").unwrap();
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            false,
            true,
            qwen2_vram_threshold(q6.size_bytes) + 1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q6");
        assert!(resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_falls_back_to_q4_on_metal_when_q6_does_not_fit() {
        let q4 = mold_core::manifest::find_qwen2_vl_variant("q4").unwrap();
        let q6 = mold_core::manifest::find_qwen2_vl_variant("q6").unwrap();
        let free_vram = qwen2_vram_threshold(q4.size_bytes);
        assert!(free_vram < qwen2_vram_threshold(q6.size_bytes));

        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            false,
            true,
            free_vram,
            0,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert!(resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_keeps_bf16_default_on_cuda() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            QWEN2_FP16_VRAM_THRESHOLD + 1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(!resolved.is_gguf);
        assert_eq!(resolved.variant_label, "bf16");
        assert!(resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_prefers_quantized_gpu_on_cuda_for_resident_mode_when_it_fits() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            QWEN2_FP16_VRAM_THRESHOLD - 1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert!(resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_uses_quantized_cpu_fallback_on_cuda_for_resident_mode() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert!(!resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_keeps_bf16_cpu_on_cuda_for_sequential_mode() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Sequential,
        )
        .unwrap();
        assert!(!resolved.is_gguf);
        assert_eq!(resolved.variant_label, "bf16");
        assert!(!resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_explicit_q6_respects_cpu_fallback_on_cuda() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("q6"),
            true,
            false,
            1,
            0,
            Qwen2TextEncoderUsage::Resident,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q6");
        assert!(!resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_edit_accepts_quantized_text_with_bf16_vision_sidecar() {
        let dir = temp_test_dir("qwen-image-edit-text-encoder");
        let transformer = touch(&dir, "qwen-image-edit.gguf");
        let vae = touch(&dir, "vae.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");
        let mut paths = qwen_image_model_paths(transformer, vec![], vae, Some(tokenizer));
        paths.text_encoder_files = vec![touch(&dir, "text-encoder-00001-of-00004.safetensors")];
        let engine = QwenImageEngine::new(
            "qwen-image-edit-2511:q4".to_string(),
            paths,
            LoadStrategy::Sequential,
            false,
        );

        let resolved = engine
            .resolve_text_encoder_source_with_preference(
                &Device::Cpu,
                0,
                Qwen2TextEncoderUsage::Sequential,
                Some("auto"),
            )
            .unwrap();
        assert!(!resolved.vision_paths.is_empty());

        let resolved = engine
            .resolve_text_encoder_source_with_preference(
                &Device::Cpu,
                0,
                Qwen2TextEncoderUsage::Sequential,
                Some("q4"),
            )
            .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert_eq!(resolved.vision_paths.len(), 1);

        let resolved = engine
            .resolve_text_encoder_source_with_preference(
                &Device::Cpu,
                0,
                Qwen2TextEncoderUsage::Sequential,
                Some("bf16"),
            )
            .unwrap();
        assert!(!resolved.is_gguf);
        assert_eq!(resolved.variant_label, "bf16");
        assert_eq!(resolved.vision_paths.len(), 1);
    }

    #[test]
    fn qwen_image_edit_prompt_numbers_each_picture_placeholder() {
        let prompt = QwenImageEngine::qwen_image_edit_prompt("swap materials", 3);
        assert!(prompt.contains(QWEN_IMAGE_EDIT_SYSTEM_PROMPT));
        assert!(prompt.contains("Picture 1: <|vision_start|><|image_pad|><|vision_end|>"));
        assert!(prompt.contains("Picture 2: <|vision_start|><|image_pad|><|vision_end|>"));
        assert!(prompt.contains("Picture 3: <|vision_start|><|image_pad|><|vision_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen_image_edit_image_dims_fit_target_area_with_16px_alignment() {
        let bytes = png_with_dimensions(1600, 900);
        let (width, height) =
            QwenImageEngine::qwen_image_edit_image_dims(&bytes, QWEN_IMAGE_EDIT_VAE_AREA).unwrap();
        assert_eq!((width, height), (1360, 768));
        assert_eq!(width % 16, 0);
        assert_eq!(height % 16, 0);
    }

    #[test]
    fn pack_and_unpack_latents_roundtrip() {
        let values: Vec<f32> = (0..(16 * 4 * 6)).map(|i| i as f32).collect();
        let latents = Tensor::from_vec(values.clone(), (1, 16, 4, 6), &Device::Cpu).unwrap();
        let packed = QwenImageEngine::pack_latents_4d(&latents).unwrap();
        assert_eq!(packed.dims3().unwrap(), (1, 6, 64));

        let unpacked = QwenImageEngine::unpack_latents_packed(&packed, 4, 6).unwrap();
        assert_eq!(
            unpacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            values
        );
    }

    #[test]
    fn quantized_cuda_cfg_headroom_scales_with_resolution() {
        let native = QwenImageEngine::quantized_cuda_cfg_headroom(1328, 1328);
        let reduced = QwenImageEngine::quantized_cuda_cfg_headroom(512, 512);
        assert_eq!(native, QWEN_GGUF_NATIVE_CFG_HEADROOM);
        assert_eq!(reduced, QWEN_GGUF_MIN_CFG_HEADROOM);
    }

    #[test]
    fn qwen_quantized_native_resolution_uses_split_cfg_on_24gb_cuda() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            12_300_000_000,
            24_600_000_000,
            1328,
            1328,
        ));
    }

    #[test]
    fn qwen_quantized_reduced_resolution_keeps_batched_cfg_when_it_fits() {
        assert!(!QwenImageEngine::should_split_cfg_quantized_cuda(
            12_300_000_000,
            24_600_000_000,
            512,
            512,
        ));
    }

    #[test]
    fn qwen_quantized_cfg_split_boundary_does_not_split_when_estimate_exactly_fits() {
        let headroom = QwenImageEngine::quantized_cuda_cfg_headroom(1328, 1328);
        let transformer_size = 12_300_000_000;
        let free_vram = transformer_size + headroom;
        assert!(!QwenImageEngine::should_split_cfg_quantized_cuda(
            transformer_size,
            free_vram,
            1328,
            1328,
        ));
    }

    #[test]
    fn qwen_quantized_unknown_vram_biases_to_split_cfg() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            12_300_000_000,
            0,
            1328,
            1328,
        ));
    }

    #[test]
    fn qwen_is_oom_error_matches_cuda_memory_allocation_string() {
        assert!(QwenImageEngine::is_oom_error(&"cudaErrorMemoryAllocation"));
    }

    #[test]
    fn qwen_oom_fallback_returns_primary_success_without_running_fallback() {
        let mut progress = ProgressReporter::default();
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();
        progress.set_callback(Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                messages_clone.lock().unwrap().push(message);
            }
        }));

        let fallback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fallback_called_clone = fallback_called.clone();
        let value = QwenImageEngine::with_cuda_oom_cpu_fallback(
            || Ok(7usize),
            || {
                fallback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(9usize)
            },
            true,
            &Device::Cpu,
            &progress,
            "retrying",
            |_| true,
        )
        .unwrap();

        assert_eq!(value, 7);
        assert!(!fallback_called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(messages.lock().unwrap().is_empty());
    }

    #[test]
    fn qwen_oom_fallback_retries_when_primary_ooms_on_cuda() {
        let mut progress = ProgressReporter::default();
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();
        progress.set_callback(Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                messages_clone.lock().unwrap().push(message);
            }
        }));

        let value = QwenImageEngine::with_cuda_oom_cpu_fallback(
            || Err(anyhow::anyhow!("cudaErrorMemoryAllocation")),
            || Ok(11usize),
            true,
            &Device::Cpu,
            &progress,
            "retrying",
            QwenImageEngine::is_oom_error,
        )
        .unwrap();

        assert_eq!(value, 11);
        assert_eq!(messages.lock().unwrap().as_slice(), ["retrying"]);
    }

    #[test]
    fn qwen_oom_fallback_does_not_retry_non_oom_errors() {
        let progress = ProgressReporter::default();
        let err = QwenImageEngine::with_cuda_oom_cpu_fallback(
            || Err(anyhow::anyhow!("not an oom")),
            || Ok(11usize),
            true,
            &Device::Cpu,
            &progress,
            "retrying",
            QwenImageEngine::is_oom_error,
        )
        .unwrap_err();

        assert!(err.to_string().contains("not an oom"));
    }

    #[test]
    fn qwen_tiled_fallback_returns_primary_success_without_retrying() {
        let progress = ProgressReporter::default();
        let tiled_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cpu_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tiled_called_clone = tiled_called.clone();
        let cpu_called_clone = cpu_called.clone();

        let value = QwenImageEngine::with_cuda_tiled_then_cpu_fallback(
            || Ok(5usize),
            || {
                tiled_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(7usize)
            },
            || {
                cpu_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(9usize)
            },
            true,
            &Device::Cpu,
            &progress,
            "tiled",
            "cpu",
            |_| true,
        )
        .unwrap();

        assert_eq!(value, 5);
        assert!(!tiled_called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!cpu_called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn qwen_tiled_fallback_uses_tiled_result_before_cpu() {
        let mut progress = ProgressReporter::default();
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();
        progress.set_callback(Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                messages_clone.lock().unwrap().push(message);
            }
        }));

        let cpu_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cpu_called_clone = cpu_called.clone();
        let value = QwenImageEngine::with_cuda_tiled_then_cpu_fallback(
            || Err(anyhow::anyhow!("out of memory")),
            || Ok(13usize),
            || {
                cpu_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(17usize)
            },
            true,
            &Device::Cpu,
            &progress,
            "tiled",
            "cpu",
            QwenImageEngine::is_oom_error,
        )
        .unwrap();

        assert_eq!(value, 13);
        assert!(!cpu_called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(messages.lock().unwrap().as_slice(), ["tiled"]);
    }

    #[test]
    fn qwen_tiled_fallback_uses_cpu_after_tiled_oom() {
        let mut progress = ProgressReporter::default();
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();
        progress.set_callback(Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                messages_clone.lock().unwrap().push(message);
            }
        }));

        let value = QwenImageEngine::with_cuda_tiled_then_cpu_fallback(
            || Err(anyhow::anyhow!("OUT_OF_MEMORY")),
            || Err(anyhow::anyhow!("OUT_OF_MEMORY")),
            || Ok(19usize),
            true,
            &Device::Cpu,
            &progress,
            "tiled",
            "cpu",
            QwenImageEngine::is_oom_error,
        )
        .unwrap();

        assert_eq!(value, 19);
        assert_eq!(messages.lock().unwrap().as_slice(), ["tiled", "cpu"]);
    }

    #[test]
    fn qwen_tiled_fallback_propagates_non_oom_tiled_error() {
        let progress = ProgressReporter::default();
        let err = QwenImageEngine::with_cuda_tiled_then_cpu_fallback(
            || Err(anyhow::anyhow!("out of memory")),
            || Err(anyhow::anyhow!("bad tiled decode")),
            || Ok(19usize),
            true,
            &Device::Cpu,
            &progress,
            "tiled",
            "cpu",
            QwenImageEngine::is_oom_error,
        )
        .unwrap_err();

        assert!(err.to_string().contains("bad tiled decode"));
    }

    #[test]
    fn qwen_transformer_paths_prefer_shards_when_present() {
        let dir = temp_test_dir("mold-qwen-shards");
        let shard_a = touch(&dir, "transformer-00001-of-00002.safetensors");
        let shard_b = touch(&dir, "transformer-00002-of-00002.safetensors");
        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                dir.join("transformer.safetensors"),
                vec![shard_a.clone(), shard_b.clone()],
                dir.join("vae.safetensors"),
                Some(dir.join("tokenizer.json")),
            ),
            LoadStrategy::Sequential,
            false,
        );

        assert_eq!(engine.transformer_paths(), vec![shard_a, shard_b]);

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_validate_paths_accepts_existing_files() {
        let dir = temp_test_dir("mold-qwen-validate-ok");
        let shard_a = touch(&dir, "transformer-00001-of-00002.safetensors");
        let shard_b = touch(&dir, "transformer-00002-of-00002.safetensors");
        let vae = touch(&dir, "vae.safetensors");
        let tokenizer = touch(&dir, "tokenizer.json");
        let gguf = touch(&dir, "transformer.gguf");

        let sharded = QwenImageEngine::new(
            "qwen-image:bf16".to_string(),
            qwen_image_model_paths(
                dir.join("transformer.safetensors"),
                vec![shard_a, shard_b],
                vae.clone(),
                Some(tokenizer.clone()),
            ),
            LoadStrategy::Sequential,
            false,
        );
        assert_eq!(sharded.validate_paths().unwrap(), tokenizer);
        assert!(!sharded.detect_is_quantized());

        let quantized = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(gguf, vec![], vae, Some(dir.join("tokenizer.json"))),
            LoadStrategy::Sequential,
            false,
        );
        assert!(quantized.detect_is_quantized());

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_validate_paths_requires_text_tokenizer() {
        let dir = temp_test_dir("mold-qwen-validate-missing");
        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                dir.join("vae.safetensors"),
                None,
            ),
            LoadStrategy::Sequential,
            false,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("text tokenizer path required"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_img2img_uses_minus_one_to_one_source_normalization() {
        assert_eq!(
            QwenImageEngine::img2img_source_normalize_range(),
            img_utils::NormalizeRange::MinusOneToOne
        );
    }
}
