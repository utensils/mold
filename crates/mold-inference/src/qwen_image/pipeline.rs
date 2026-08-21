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
use mold_candle::quantized as quantized_var_builder;
use mold_core::{fit_to_target_area, GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

use super::quantized_transformer::QuantizedQwenImageTransformer2DModel;
use super::sampling::{image_seq_len, shift_policy_for_model, QwenImageScheduler};
use super::transformer::{QwenImageConfig, QwenImageTransformer2DModel};
use super::vae::{self, QwenImageVae};
use crate::cache::{
    clear_cache, prompt_text_key, CachedTensor, LruCache, DEFAULT_PROMPT_CACHE_CAPACITY,
};
use crate::device::{
    effective_device_ref, fits_in_memory, fmt_gb, free_vram_bytes, memory_status_string,
    preflight_memory_check, qwen2_vram_threshold, should_use_gpu, usable_free_vram_bytes,
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
/// The Qwen-Image VAE always decodes in F32 (BF16 convolutions accumulate
/// quantization noise across the decoder), so every buffer the decode-workspace
/// arithmetic counts is four bytes an element.
const VAE_F32_BYTES: u64 = 4;
// Use a single space rather than an empty string so the unconditional CFG path
// stays explicit after Qwen prompt templating and token windowing.
const QWEN_EMPTY_NEGATIVE_PROMPT: &str = " ";
const QWEN_NATIVE_WIDTH: usize = 1328;
const QWEN_NATIVE_HEIGHT: usize = 1328;
const QWEN_GGUF_MIN_CFG_HEADROOM: u64 = 3_000_000_000;
/// Transformer geometry the denoise working set is derived from.
/// `QwenImageConfig` names the same numbers; these are the `u64` copies the
/// byte arithmetic below uses, pinned to the config by a unit test.
const QWEN_DIT_INNER_DIM: u64 = 3072;
const QWEN_DIT_HEADS: u64 = 24;
const QWEN_DIT_FF_DIM: u64 = 4 * QWEN_DIT_INNER_DIM;
/// Pixels per image token: the VAE's 8x spatial downsample times the
/// transformer's 2x patch.
const QWEN_DIT_PIXELS_PER_TOKEN_AXIS: u64 = 16;
/// Every transformer tensor in the denoise runs BF16 on CUDA.
const QWEN_DIT_BF16_BYTES: u64 = 2;
/// The batch batched true CFG puts through one forward. This whole estimate
/// exists to answer "can this request afford *batched* CFG", so it is always
/// priced at two rows.
const QWEN_CFG_BATCH: u64 = 2;

/// Largest image-token count at which batched CFG measured neutral-or-better
/// against split CFG (4096 = 1024x1024). See `should_split_cfg_quantized_cuda`
/// for the 4090 measurements; raise only with new measurements.
const QWEN_CFG_BATCH_MAX_IMAGE_TOKENS: u64 = 4096;
/// Joint-stream (`[batch, text + image, inner_dim]`) BF16 buffers alive at the
/// attention peak of one block.
///
/// Shadowed `let` bindings do not drop in Rust, so the attention forward holds
/// every intermediate to the end of the call: three generations each of q and
/// k (projection, QK-norm, RoPE) and one of v (7), their `cat`ed joint forms
/// (3), the transposed contiguous copies the matmul needs (3), the attention
/// output plus its transpose/reshape/narrow copies (4), and the block's own
/// hidden states and normalized attention inputs (3).
const QWEN_DIT_ATTENTION_LIVE_STREAMS: u64 = 20;
/// Score buffers alive inside one query chunk.
///
/// Three, not two, because this estimate prices *batched* CFG specifically —
/// and putting two different prompt lengths in one forward is exactly when
/// `qwen_image::attention::joint_key_bias` returns `Some`, which routes to
/// `attention::attention_with_bias` → `math_attention_biased_impl`. That
/// closure is `let attn_weights = QK^T * scale; let attn_weights =
/// attn_weights.broadcast_add(bias); softmax(&attn_weights).matmul(v)`, and by
/// the same no-drop rule the live-stream count above relies on, the shadowed
/// scaled matrix stays alive alongside the biased one and the softmax. The
/// unbiased path (equal prompt lengths) peaks at two, so this is the
/// conservative side of the one decision the estimate makes.
const QWEN_DIT_ATTENTION_SCORE_BUFFERS: u64 = 3;
/// Joint-stream buffers alive at the MLP peak: both hidden states, both
/// normalized MLP inputs, and the modulation temporaries around them.
const QWEN_DIT_MLP_LIVE_STREAMS: u64 = 8;
/// Feed-forward buffers alive at once: the `inner -> 4*inner` projection and
/// its GELU, which the `proj.forward(x)?.apply(gelu)` temporary keeps live
/// together.
const QWEN_DIT_MLP_LIVE_FF_BUFFERS: u64 = 2;
/// Dequantized weight copies in flight. The default CUDA arm widens each GGUF
/// weight to BF16 per forward; the largest is the feed-forward `4*inner x
/// inner` pair, and two can overlap across a statement boundary.
const QWEN_DIT_DEQUANT_WEIGHT_BUFFERS: u64 = 2;
/// Bytes per element of the F32 activation copy candle's MMQ kernels make
/// before quantizing to Q8_1 (`MOLD_QWEN_QMATMUL=1`, #1045).
const QWEN_DIT_MMQ_ACTIVATION_BYTES: u64 = 4;
/// Margin over the derived working set.
///
/// The derivation counts allocation shapes, not allocator behaviour: candle
/// hands each tensor straight to CUDA, so fragmentation across a 60-block
/// forward is real and unmodelled, and the block counts above are read off
/// today's forward rather than measured. 1.5x is the same order of slack the
/// VAE decode reserve carries.
const QWEN_CFG_HEADROOM_SAFETY_NUM: u64 = 3;
const QWEN_CFG_HEADROOM_SAFETY_DEN: u64 = 2;
const QWEN_VAE_TILE_SIZES: [u32; 3] = [64, 32, 16];
const QWEN_IMAGE_EDIT_VAE_AREA: u32 =
    mold_core::validation::QWEN_IMAGE_EDIT_SOURCE_MAX_PIXELS as u32;
const QWEN_IMAGE_EDIT_SYSTEM_PROMPT: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Minimum free VRAM for BF16 Qwen2.5-VL 7B text encoder on GPU.
/// ~14GB model + 2GB headroom.
const QWEN2_FP16_VRAM_THRESHOLD: u64 = 16_000_000_000;
/// Extra residual VRAM required before keeping Qwen2.5 on GPU after a prompt
/// cache miss. The denoise/VAE reserves cover known workspaces; this absorbs
/// allocator fragmentation and backend scratch buffers.
const QWEN2_HOT_TE_RESIDENCY_HEADROOM: u64 = 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen2TextEncoderMode {
    Auto,
    Gpu,
    CpuStage,
    Cpu,
}

impl Qwen2TextEncoderMode {
    fn from_value(value: Option<&str>) -> Self {
        match value.unwrap_or_default().to_ascii_lowercase().as_str() {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen2TextEncoderPostEncodeAction {
    KeepGpu,
    ParkCpu,
    Drop,
}

#[derive(Debug, Clone, Copy)]
struct Qwen2TextEncoderResidencyInput {
    on_gpu: bool,
    is_metal: bool,
    keep_te_ram: bool,
    prompt_cache_miss: bool,
    transformer_resident: bool,
    free_vram_bytes: u64,
    required_vram_bytes: u64,
}

/// Inputs to the edit path's post-conditioning encoder release. The edit
/// pipeline never consults free VRAM the way the hot text-to-image path does —
/// it always releases the encoder — so the only question is park or drop.
#[derive(Debug, Clone, Copy)]
struct Qwen2EditTextEncoderReleaseInput {
    on_gpu: bool,
    is_metal: bool,
    keep_te_ram: bool,
    /// The engine unloads the whole `LoadedQwenImage` as soon as this request
    /// returns (sequential edit generation).
    engine_unloads_after: bool,
}

#[derive(Debug, Clone, Copy)]
struct QwenTensorStats {
    min: f32,
    max: f32,
    mean: f32,
    nan_count: u64,
    pos_inf_count: u64,
    neg_inf_count: u64,
    total: usize,
}

/// The on-device half of a boundary check: four scalars, one transfer.
///
/// `min`/`max` are `±Inf` when the tensor holds an infinity, and NaN when it
/// holds a NaN, so [`is_clean`](Self::is_clean) needs no separate ±Inf counts —
/// the full breakdown is only ever wanted for the error message, and that path
/// pays for the complete scan anyway.
#[derive(Debug, Clone, Copy)]
struct QwenFinitenessProbe {
    nan_count: u64,
    min: f32,
    max: f32,
    mean: f32,
    total: usize,
}

impl QwenFinitenessProbe {
    fn is_clean(&self) -> bool {
        self.nan_count == 0 && self.min.is_finite() && self.max.is_finite() && self.mean.is_finite()
    }

    /// Promote a clean probe to the stats shape callers already consume. Only
    /// valid for a clean probe: the ±Inf counts are asserted zero rather than
    /// measured, which is exactly what `is_clean` established.
    fn into_stats(self) -> QwenTensorStats {
        QwenTensorStats {
            min: self.min,
            max: self.max,
            mean: self.mean,
            nan_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            total: self.total,
        }
    }
}

/// Counts full CPU-side stats downloads so a test can prove the clean boundary
/// path never takes one. Test-only: the hot path must not pay for a counter.
#[cfg(test)]
static FULL_TENSOR_STATS_DOWNLOADS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

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
}

impl CachedPromptConditioning {
    /// `Qwen2TextEncoder` always narrows its embeddings to the true token count
    /// and returns an all-ones mask, so `valid_len` is `hidden_states.dim(1)`
    /// and the cache does not need to carry a mask. The check is kept so a
    /// future encoder that starts padding fails loudly rather than silently
    /// feeding pad rows into the transformer.
    fn from_parts(hidden_states: &Tensor, valid_len: usize) -> Result<Self> {
        let seq_len = hidden_states.dim(1)?;
        if valid_len != seq_len {
            bail!("Qwen text conditioning must arrive unpadded: {valid_len} of {seq_len} tokens");
        }
        Ok(Self {
            hidden_states: CachedTensor::from_tensor(hidden_states)?,
        })
    }

    fn restore(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        self.hidden_states.restore(device, dtype)
    }
}

fn pad_text_conditioning(hidden_states: &Tensor, target_len: usize) -> Result<Tensor> {
    let seq_len = hidden_states.dim(1)?;
    if seq_len == target_len {
        return Ok(hidden_states.clone());
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

    Ok(Tensor::cat(&[hidden_states, &pad_hs], 1)?)
}

/// Pad the two CFG streams to a common length **only** when batching them into
/// one forward, and return the joint `[2, target_len]` text mask that goes with
/// the padding — `None` when both streams were already the same length.
///
/// This mirrors diffusers `pipeline_qwenimage.py:265-266`
/// (`if prompt_embeds_mask.all(): prompt_embeds_mask = None`): a mask that
/// keeps every key is not a mask, and the padding that made one necessary is
/// pure waste on every path that does not batch.
fn align_cfg_conditioning(
    cond_hs: &Tensor,
    uncond_hs: &Tensor,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    let cond_len = cond_hs.dim(1)?;
    let uncond_len = uncond_hs.dim(1)?;
    if cond_len == uncond_len {
        return Ok((cond_hs.clone(), uncond_hs.clone(), None));
    }

    let target_len = cond_len.max(uncond_len);
    let mut mask = vec![0u8; 2 * target_len];
    for value in &mut mask[..cond_len] {
        *value = 1;
    }
    for value in &mut mask[target_len..target_len + uncond_len] {
        *value = 1;
    }
    let mask = Tensor::from_vec(mask, (2, target_len), cond_hs.device())?;

    Ok((
        pad_text_conditioning(cond_hs, target_len)?,
        pad_text_conditioning(uncond_hs, target_len)?,
        Some(mask),
    ))
}

impl QwenImageTransformer {
    fn supports_cfg_batching(&self) -> bool {
        match self {
            Self::Quantized(model) => model.supports_cfg_batching(),
            _ => true,
        }
    }

    /// Re-take the quantized CUDA CFG-batching budget decision in place.
    ///
    /// Returns whether the flag moved, so the caller can report a change
    /// without printing a line on every request. Only the quantized arm
    /// carries the flag — the others always batch — so every other arm is a
    /// no-op.
    fn set_supports_cfg_batching(&mut self, supports_cfg_batching: bool) -> bool {
        match self {
            Self::Quantized(model) => {
                if model.supports_cfg_batching() == supports_cfg_batching {
                    return false;
                }
                model.set_supports_cfg_batching(supports_cfg_batching);
                true
            }
            _ => false,
        }
    }

    fn forward(
        &self,
        latents: &Tensor,
        t: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
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
        encoder_attention_mask: Option<&Tensor>,
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
    /// Per-request placement override.
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// Per-request LoRA stack. Captured at the start of `generate()`,
    /// cleared on exit. The transformer-load path consults this when
    /// constructing the `VarBuilder` so the LoRA-merged weights land
    /// before any forward pass runs.
    pending_loras: Vec<mold_core::LoraWeight>,
    /// Fingerprint of the LoRA stack baked into the transformer that is
    /// resident right now — empty both when no LoRA is merged and when no
    /// transformer is resident, which is why every rebuild decision reads
    /// residency alongside it (`qwen_transformer_rebuild_needed`).
    ///
    /// Only the resident paths (eager and the quantized stay-hot VAE
    /// decode) can elide: `generate_sequential` builds a request-local
    /// transformer and drops it, so it has nothing to reuse. Mirrors
    /// `FluxEngine::active_lora`.
    active_lora_fingerprint: Vec<QwenImageLoraFingerprint>,
    shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    qwen2_variant: Option<String>,
    qwen2_text_encoder_mode: Qwen2TextEncoderMode,
    /// Text encoder retained between sequential generations under
    /// `MOLD_KEEP_TE_RAM=1`, parked on host RAM. `None` in the default
    /// drop-and-reload mode.
    ///
    /// Sequential is the load-use-drop path — and the one the server actually
    /// selects for a quantized Qwen-Image checkpoint (`memory_preflight.rs`
    /// routes a GGUF transformer that fits straight to `Sequential`), so
    /// without this the #1044 park was unreachable for its own target
    /// configuration. The encoder is a local there rather than part of
    /// `LoadedQwenImage`, so the retention has to live on the engine.
    /// Mirrors `WanEngine::retained_encoder`.
    retained_sequential_text_encoder: Option<encoders::qwen2_text::Qwen2TextEncoder>,
}

/// One quantized CUDA split-vs-batched CFG decision plus the readings it was
/// taken from, so the caller can report the same numbers it decided on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedCfgDecision {
    split: bool,
    transformer_size: u64,
    free: u64,
}

/// Order-sensitive fingerprint of a single LoRA adapter (path-hash + scale).
#[derive(Clone, PartialEq, Eq, Debug)]
struct QwenImageLoraFingerprint {
    path_hash: u64,
    scale_bits: u64,
}

impl QwenImageLoraFingerprint {
    fn from_lora(lora: &mold_core::LoraWeight) -> Self {
        Self {
            path_hash: super::lora::lora_path_hash(&lora.path),
            scale_bits: lora.scale.to_bits(),
        }
    }
}

/// Fingerprint of an ordered LoRA stack. Equality is order-sensitive for
/// the same reason FLUX's is — the deltas commute numerically, but the
/// user-facing intent is order-driven and one redundant rebuild is cheap
/// next to a wrong merge.
fn fingerprint_stack(loras: &[mold_core::LoraWeight]) -> Vec<QwenImageLoraFingerprint> {
    loras
        .iter()
        .map(QwenImageLoraFingerprint::from_lora)
        .collect()
}

/// Resolve the effective LoRA list for a request. Mirrors `flux::pipeline::
/// effective_loras` — accepts both `lora` (singular, legacy) and `loras`
/// (plural, current). Entries with a near-zero scale are dropped.
fn effective_loras(req: &mold_core::GenerateRequest) -> Vec<mold_core::LoraWeight> {
    /// Match the FLUX threshold so the user-facing semantics are
    /// identical across families.
    const ZERO_SCALE_EPS: f64 = 1e-8;

    let raw: Vec<mold_core::LoraWeight> = if let Some(plural) = &req.loras {
        if !plural.is_empty() {
            plural.clone()
        } else {
            req.lora.iter().cloned().collect()
        }
    } else {
        req.lora.iter().cloned().collect()
    };

    raw.into_iter()
        .filter(|w| {
            let keep = w.scale.abs() > ZERO_SCALE_EPS;
            if !keep {
                tracing::debug!(
                    path = w.path.as_str(),
                    scale = w.scale,
                    "dropping zero-scale LoRA from effective Qwen-Image stack"
                );
            }
            keep
        })
        .collect()
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
        Ok(Self::qwen_image_edit_dims(
            img.width(),
            img.height(),
            target_area,
        ))
    }

    fn qwen_image_edit_dims(source_width: u32, source_height: u32, target_area: u32) -> (u32, u32) {
        let source_ratio = source_width.max(1) as f64 / source_height.max(1) as f64;
        let (mut width, mut height) =
            fit_to_target_area(source_width.max(1), source_height.max(1), target_area, 16);
        // `fit_to_target_area` rounds to the nearest grid point and can land
        // one block above the advertised area. Tighten downward, choosing the
        // axis that least perturbs the source aspect, so the model contract is
        // a ceiling rather than a guideline.
        while width.saturating_mul(height) > target_area && (width > 16 || height > 16) {
            if width == 16 {
                height = ((target_area / width) / 16).max(1) * 16;
                continue;
            }
            if height == 16 {
                width = ((target_area / height) / 16).max(1) * 16;
                continue;
            }
            let width_error = ((width - 16) as f64 / height as f64 - source_ratio).abs();
            let height_error = (width as f64 / (height - 16) as f64 - source_ratio).abs();
            if width_error <= height_error {
                width -= 16;
            } else {
                height -= 16;
            }
        }
        (width, height)
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
        prefer_tiled: bool,
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
        if is_cuda && prefer_tiled {
            progress.info("Selecting tiled GPU VAE decode proactively");
            match tiled() {
                Ok(value) => return Ok(value),
                Err(tile_err) if is_oom(&tile_err) => {
                    progress.info(cpu_message);
                    sync_device.synchronize()?;
                    return cpu_fallback();
                }
                Err(tile_err) => return Err(tile_err),
            }
        }

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

    /// Transient VRAM the VAE decode needs on top of its weights.
    ///
    /// Derived from the allocation shapes of the decode's three phases rather
    /// than from a flat per-pixel factor. The phases run one after another and
    /// each one's buffers are freed before the next allocates, so the reserve
    /// is their **max**, not their sum. Every buffer is F32; see the
    /// `qwen_vae_*_phase_bytes` helpers for the arithmetic.
    ///
    /// The term that dominates is not in the model at all — it is candle's. mold
    /// does not enable candle's `cudnn` feature, so `CudaStorage::conv2d` takes
    /// the im2col path (`USE_IM2COL_CONV2D = true`): every convolution first
    /// materialises a `[b * h_out * w_out, c_in * k_h * k_w]` column buffer,
    /// matmuls it, and copies the result out of NHWC into NCHW. For the
    /// `192 -> 96` 3x3 conv the decoder runs on the full pixel grid, that column
    /// buffer alone is `P * 192 * 9 * 4` — 12.2 GB at native 1328², an order of
    /// magnitude past every activation tensor in the same phase and the single
    /// largest allocation in the whole decode.
    ///
    /// Native 1328² therefore reserves ~15.2 GB, and the proactive gate below
    /// asks for ~17.7 GB free.
    fn qwen_vae_decode_workspace_bytes(width: u32, height: u32) -> u64 {
        let pixels = (width as u64).saturating_mul(height as u64);

        Self::qwen_vae_mid_block_phase_bytes(pixels)
            .max(Self::qwen_vae_full_res_upsample_phase_bytes(pixels))
            .max(Self::qwen_vae_final_up_block_phase_bytes(pixels))
    }

    /// Mid block, at latent resolution: `N = P / compression²` tokens of width
    /// `C = VAE_MID_BLOCK_CHANNELS`.
    ///
    /// * score buffers — two `[chunk, N]` matrices (the scores and their
    ///   softmax) = `2 * VAE_ATTENTION_CHUNK_ROWS * N * 4`. This term used to be
    ///   `2 * N * N * 4`, i.e. 6.1 GB at native 1328²; it is now 226 MB, which
    ///   is what took this phase out of contention for the peak.
    /// * operands — the `[N, 3C]` qkv projection, the three contiguous `[N, C]`
    ///   q/k/v copies, the `[N, C]` output and its transpose: `8 * C * N * 4`.
    fn qwen_vae_mid_block_phase_bytes(pixels: u64) -> u64 {
        const OPERANDS: u64 = 8;

        let compression = vae::VAE_SPATIAL_COMPRESSION as u64;
        let latent_tokens = pixels / compression.saturating_mul(compression);
        let scores = 2u64.saturating_mul(vae::VAE_ATTENTION_CHUNK_ROWS as u64);
        let operands = OPERANDS.saturating_mul(vae::VAE_MID_BLOCK_CHANNELS as u64);

        scores
            .saturating_add(operands)
            .saturating_mul(latent_tokens)
            .saturating_mul(VAE_F32_BYTES)
    }

    /// The first convolution evaluated on the full pixel grid, and the decode's
    /// peak: up-block 2's upsampler nearest-doubles its input and *then*
    /// convolves `192 -> 96` at the doubled shape. Live at once — the
    /// quarter-area input, the upsampled input, the im2col column buffer, the
    /// matmul result and the transposed copy candle makes of it.
    fn qwen_vae_full_res_upsample_phase_bytes(pixels: u64) -> u64 {
        let c_in = vae::VAE_FULL_RES_UPSAMPLE_IN_CHANNELS as u64;
        let c_out = vae::VAE_FINAL_BLOCK_CHANNELS as u64;

        let input = c_in.saturating_mul(pixels / 4);
        let upsampled = c_in.saturating_mul(pixels);
        let conv = Self::qwen_vae_conv2d_phase_bytes(pixels, c_in, c_out);

        input
            .saturating_add(upsampled)
            .saturating_mul(VAE_F32_BYTES)
            .saturating_add(conv)
    }

    /// The last up-block's three `96 -> 96` residual blocks, at full resolution:
    /// the residual chain keeps eight of those buffers alive to the end of the
    /// statement (`x`, five chained temporaries, `h`, and the `residual + h`
    /// result), and one of its convolutions is in flight on top of them.
    fn qwen_vae_final_up_block_phase_bytes(pixels: u64) -> u64 {
        const LIVE_BUFFERS: u64 = 8;

        let channels = vae::VAE_FINAL_BLOCK_CHANNELS as u64;
        let chain = LIVE_BUFFERS
            .saturating_mul(channels)
            .saturating_mul(pixels)
            .saturating_mul(VAE_F32_BYTES);

        chain.saturating_add(Self::qwen_vae_conv2d_phase_bytes(
            pixels, channels, channels,
        ))
    }

    /// What one candle im2col `conv2d` holds while it runs: the column buffer
    /// `[P, c_in * k]`, the matmul result `[P, c_out]`, and the transposed copy
    /// that reorders it into NCHW.
    fn qwen_vae_conv2d_phase_bytes(pixels: u64, in_channels: u64, out_channels: u64) -> u64 {
        let columns = in_channels.saturating_mul(vae::VAE_CONV_KERNEL_ELEMS as u64);
        let result_and_transpose = 2u64.saturating_mul(out_channels);

        columns
            .saturating_add(result_and_transpose)
            .saturating_mul(pixels)
            .saturating_mul(VAE_F32_BYTES)
    }

    /// Skip the full-resolution decode and go straight to tiles when the card
    /// cannot hold the reserve above.
    ///
    /// The machinery is unchanged. Native 1328² asks for ~17.7 GB free, which
    /// is what the im2col column buffer in the decoder's full-resolution
    /// convolutions actually costs — a card with less than that would reach the
    /// OOM-triggered fallback below and tile anyway, having paid for the failed
    /// allocation and a device synchronize first.
    fn should_proactively_tile_vae_decode(
        width: u32,
        height: u32,
        vae_is_cuda: bool,
        free_vram_bytes: u64,
    ) -> bool {
        if !vae_is_cuda || free_vram_bytes == 0 {
            return false;
        }
        let native_pixels = (QWEN_NATIVE_WIDTH * QWEN_NATIVE_HEIGHT) as u64;
        let pixels = width as u64 * height as u64;
        if pixels < native_pixels.saturating_mul(3) / 4 {
            return false;
        }
        let required = VAE_DECODE_VRAM_THRESHOLD
            .saturating_add(Self::qwen_vae_decode_workspace_bytes(width, height));
        free_vram_bytes < required
    }

    fn qwen2_text_encoder_post_encode_action(
        input: Qwen2TextEncoderResidencyInput,
    ) -> Qwen2TextEncoderPostEncodeAction {
        if !input.on_gpu {
            return Qwen2TextEncoderPostEncodeAction::Drop;
        }
        if input.prompt_cache_miss
            && input.transformer_resident
            && !input.is_metal
            && input.free_vram_bytes >= input.required_vram_bytes
        {
            return Qwen2TextEncoderPostEncodeAction::KeepGpu;
        }
        // Both dtypes park (#1044): the GGUF encoder's `QTensor` bytes move
        // host↔device losslessly, so a quantized encoder no longer has to pay
        // the 35.1 s disk reload every cold prompt. Metal is still excluded —
        // unified memory makes "host RAM" the same pool.
        if input.keep_te_ram && !input.is_metal {
            return Qwen2TextEncoderPostEncodeAction::ParkCpu;
        }
        Qwen2TextEncoderPostEncodeAction::Drop
    }

    /// Park or drop the edit path's text encoder once conditioning is done.
    ///
    /// Three gates, each of which was a real regression when it was missing:
    ///
    /// * `on_gpu` — dynamic placement can put the encoder on the CPU, where a
    ///   "park" moves nothing (`to_device` short-circuits on the same device)
    ///   and simply retains host RAM the drop used to release. With a
    ///   three-engine model-cache LRU that is up to three encoders' worth of
    ///   unreleased host RAM on a box that was already short of VRAM.
    /// * `!is_metal` — unified memory makes host RAM the same pool.
    /// * `!engine_unloads_after` — sequential edit generation calls `unload()`
    ///   as soon as the request returns, dropping the parked map microseconds
    ///   after a multi-gigabyte device→host copy paid for it.
    fn qwen2_edit_text_encoder_should_park(input: Qwen2EditTextEncoderReleaseInput) -> bool {
        input.keep_te_ram && input.on_gpu && !input.is_metal && !input.engine_unloads_after
    }

    fn qwen2_hot_text_encoder_required_vram(
        width: u32,
        height: u32,
        cfg_batch: u32,
        dtype: DType,
    ) -> u64 {
        crate::device::activation_bytes(
            width,
            height,
            cfg_batch,
            crate::device::dtype_bytes(dtype),
            crate::device::ActivationFamily::QwenImageDit,
        )
        .saturating_add(VAE_DECODE_VRAM_THRESHOLD)
        .saturating_add(Self::qwen_vae_decode_workspace_bytes(width, height))
        .saturating_add(QWEN2_HOT_TE_RESIDENCY_HEADROOM)
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
        prefer_tiled: bool,
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
            prefer_tiled,
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

        progress.phase_done(
            crate::ProgressPhase::Vae,
            "Encoding source image (VAE)",
            encode_start.elapsed(),
        );
        result
    }

    fn choose_text_encoder_source(
        preference: Option<&str>,
        is_cuda: bool,
        is_metal: bool,
        free_vram: u64,
        bf16_size_bytes: u64,
        _usage: Qwen2TextEncoderUsage,
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
                    let fallback_tag = "q4";
                    let fallback = mold_core::manifest::find_qwen2_vl_variant(fallback_tag)
                        .expect("known CUDA fallback qwen2 variant missing");
                    return Ok(ResolvedQwen2TextEncoder {
                        paths: vec![],
                        vision_paths: vec![],
                        is_gguf: true,
                        variant_label: fallback.tag.to_string(),
                        size_bytes: fallback.size_bytes,
                        auto_use_gpu: fits_in_memory(
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

    /// Four scalars reduced on the tensor's own device.
    ///
    /// This is the cheap half of the boundary check: `[nan_count, min, max,
    /// mean]` are computed with tensor ops that stay where the tensor lives
    /// and are concatenated into one 4-element vector, so a clean boundary
    /// costs one 16-byte transfer instead of a full `to_vec1` of the whole
    /// tensor — which at the decoded-image boundary is 5.3M f32 elements, run
    /// twice per image.
    ///
    /// `min`/`max` over a tensor containing NaN are backend-defined, which is
    /// fine: `nan_count` is what decides, and a probe that trips hands over to
    /// the full CPU-side stats for the message.
    fn tensor_finiteness_probe(tensor: &Tensor) -> Result<QwenFinitenessProbe> {
        let flat = tensor.to_dtype(DType::F32)?.flatten_all()?;
        let total = flat.elem_count();
        if total == 0 {
            return Ok(QwenFinitenessProbe {
                nan_count: 0,
                min: f32::NAN,
                max: f32::NAN,
                mean: f32::NAN,
                total,
            });
        }
        // `x != x` is true exactly for NaN, so the sum counts them.
        let nan = flat
            .ne(&flat)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .reshape(1)?;
        let min = flat.min(0)?.reshape(1)?;
        let max = flat.max(0)?.reshape(1)?;
        let mean = flat.mean_all()?.reshape(1)?;
        let probe = Tensor::cat(&[&nan, &min, &max, &mean], 0)?.to_vec1::<f32>()?;
        Ok(QwenFinitenessProbe {
            nan_count: probe[0] as u64,
            min: probe[1],
            max: probe[2],
            mean: probe[3],
            total,
        })
    }

    /// Full CPU-side stats: one complete GPU→CPU copy plus a scalar loop.
    ///
    /// Only reached when the cheap probe trips or `MOLD_QWEN_DEBUG` is set —
    /// the boundary validator's common path is
    /// [`tensor_finiteness_probe`].
    fn tensor_stats(tensor: &Tensor) -> Result<QwenTensorStats> {
        #[cfg(test)]
        FULL_TENSOR_STATS_DOWNLOADS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let t = tensor.to_dtype(DType::F32)?;
        let values = t.flatten_all()?.to_vec1::<f32>()?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut finite_count = 0usize;
        let mut nan_count = 0u64;
        let mut pos_inf_count = 0u64;
        let mut neg_inf_count = 0u64;
        for value in &values {
            if value.is_nan() {
                nan_count += 1;
            } else if *value == f32::INFINITY {
                pos_inf_count += 1;
            } else if *value == f32::NEG_INFINITY {
                neg_inf_count += 1;
            } else {
                min = min.min(*value);
                max = max.max(*value);
                sum += *value as f64;
                finite_count += 1;
            }
        }
        let mean = if finite_count == 0 {
            f32::NAN
        } else {
            (sum / finite_count as f64) as f32
        };
        if finite_count == 0 {
            min = f32::NAN;
            max = f32::NAN;
        }
        Ok(QwenTensorStats {
            min,
            max,
            mean,
            nan_count,
            pos_inf_count,
            neg_inf_count,
            total: values.len(),
        })
    }

    fn format_tensor_stats(name: &str, stats: QwenTensorStats) -> String {
        format!(
            "[qwen-debug] {name}: min={:.4} max={:.4} mean={:.4} NaN={}/{} ({:.1}%) +Inf={} -Inf={}",
            stats.min,
            stats.max,
            stats.mean,
            stats.nan_count,
            stats.total,
            stats.nan_count as f64 / stats.total.max(1) as f64 * 100.0,
            stats.pos_inf_count,
            stats.neg_inf_count
        )
    }

    fn near_black_image_stats(stats: QwenTensorStats) -> bool {
        if stats.nan_count > 0
            || stats.pos_inf_count > 0
            || stats.neg_inf_count > 0
            || !stats.min.is_finite()
            || !stats.max.is_finite()
            || !stats.mean.is_finite()
        {
            return false;
        }
        let scale = if stats.max <= 1.0 { 1.0 } else { 255.0 };
        stats.max <= 0.02 * scale && stats.mean <= 0.01 * scale
    }

    /// Whether a boundary has to fall back to the full CPU-side stats.
    ///
    /// Two reasons, and only two: the operator asked for the numbers, or the
    /// cheap probe found something non-finite and the error message needs the
    /// NaN/±Inf breakdown that only the full scan produces.
    fn boundary_needs_full_stats(probe: QwenFinitenessProbe, debug: bool) -> bool {
        debug || !probe.is_clean()
    }

    /// Fail loudly on a non-finite boundary tensor — this is what caught the
    /// MMQ kernel defect in #1045 — without paying for a full GPU→CPU copy
    /// when nothing is wrong.
    ///
    /// The common path is four scalars reduced on-device
    /// ([`tensor_finiteness_probe`]). The full download happens only when that
    /// probe trips, or when `MOLD_QWEN_DEBUG` asks for the numbers, so a clean
    /// render transfers 16 bytes per boundary instead of the whole tensor.
    fn validate_qwen_tensor_boundary(name: &str, tensor: &Tensor) -> Result<QwenTensorStats> {
        let probe = Self::tensor_finiteness_probe(tensor)?;
        if !Self::boundary_needs_full_stats(probe, std::env::var_os("MOLD_QWEN_DEBUG").is_some()) {
            return Ok(probe.into_stats());
        }

        let stats = Self::tensor_stats(tensor)?;
        if stats.nan_count > 0
            || stats.pos_inf_count > 0
            || stats.neg_inf_count > 0
            || !stats.min.is_finite()
            || !stats.max.is_finite()
            || !stats.mean.is_finite()
        {
            bail!(
                "Qwen diagnostic boundary '{name}' contains non-finite values: {}",
                Self::format_tensor_stats(name, stats)
            );
        }
        Ok(stats)
    }

    fn debug_tensor_stats(name: &str, tensor: &Tensor) {
        if std::env::var_os("MOLD_QWEN_DEBUG").is_none() {
            return;
        }
        match Self::tensor_stats(tensor) {
            Ok(stats) => eprintln!("{}", Self::format_tensor_stats(name, stats)),
            Err(err) => eprintln!("[qwen-debug] {name}: <failed: {err}>"),
        }
    }

    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        offload: bool,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
    ) -> Self {
        Self::new_with_preferences(
            model_name,
            paths,
            load_strategy,
            gpu_ordinal,
            offload,
            shared_pool,
            crate::runtime_env::value("MOLD_QWEN2_VARIANT"),
            crate::runtime_env::value("MOLD_QWEN2_TEXT_ENCODER_MODE"),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_preferences(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        offload: bool,
        shared_pool: Option<Arc<Mutex<crate::shared_pool::SharedPool>>>,
        qwen2_variant: Option<String>,
        qwen2_text_encoder_mode: Option<String>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            prompt_cache: Mutex::new(LruCache::new(DEFAULT_PROMPT_CACHE_CAPACITY)),
            offload,
            pending_placement: None,
            pending_loras: Vec::new(),
            active_lora_fingerprint: Vec::new(),
            shared_pool,
            qwen2_variant,
            qwen2_text_encoder_mode: Qwen2TextEncoderMode::from_value(
                qwen2_text_encoder_mode.as_deref(),
            ),
            retained_sequential_text_encoder: None,
        }
    }

    fn load_text_tokenizer(&self, tokenizer_path: &Path) -> Result<Arc<Tokenizer>> {
        if let Some(shared_pool) = &self.shared_pool {
            return shared_pool.lock().unwrap().load_tokenizer(tokenizer_path);
        }
        Tokenizer::from_file(tokenizer_path)
            .map(Arc::new)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2.5 tokenizer: {e}"))
    }

    fn encode_prompt_cached(
        progress: &ProgressReporter,
        prompt_cache: &Mutex<LruCache<String, CachedPromptConditioning>>,
        text_encoder: &mut encoders::qwen2_text::Qwen2TextEncoder,
        prompt: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
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
        progress.phase_done(
            crate::ProgressPhase::PromptEncode,
            "Encoding prompt (Qwen2.5)",
            encode_start.elapsed(),
        );

        prompt_cache.lock().expect("cache poisoned").insert(
            cache_key,
            CachedPromptConditioning::from_parts(&hidden_states, valid_len)?,
        );

        Ok(hidden_states)
    }

    fn spill_conditioning_to_cpu(hidden_states: Tensor) -> Result<Tensor> {
        Ok(hidden_states
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?)
    }

    fn maybe_spill_conditioning(use_cpu_staging: bool, hidden_states: Tensor) -> Result<Tensor> {
        if use_cpu_staging {
            Self::spill_conditioning_to_cpu(hidden_states)
        } else {
            Ok(hidden_states)
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

    /// Image tokens one CFG row carries: the VAE's 8x downsample times the
    /// transformer's 2x patch, on each axis.
    fn qwen_dit_image_tokens(width: usize, height: usize) -> u64 {
        let w = (width.max(1) as u64) / QWEN_DIT_PIXELS_PER_TOKEN_AXIS;
        let h = (height.max(1) as u64) / QWEN_DIT_PIXELS_PER_TOKEN_AXIS;
        w.max(1).saturating_mul(h.max(1))
    }

    /// Joint sequence length: image tokens plus the widest text window
    /// `Qwen2TextEncoder` can emit.
    fn qwen_dit_joint_tokens(width: usize, height: usize) -> u64 {
        Self::qwen_dit_image_tokens(width, height)
            .saturating_add(encoders::qwen2_text::MAX_SEQUENCE_LENGTH as u64)
    }

    /// One BF16 `[cfg_batch, joint_tokens, inner_dim]` buffer.
    fn qwen_dit_joint_stream_bytes(joint_tokens: u64) -> u64 {
        QWEN_CFG_BATCH
            .saturating_mul(joint_tokens)
            .saturating_mul(QWEN_DIT_INNER_DIM)
            .saturating_mul(QWEN_DIT_BF16_BYTES)
    }

    /// The widest dequantized weight the forward materializes, times the
    /// number that can be in flight at once.
    fn qwen_dit_dequant_weight_bytes() -> u64 {
        QWEN_DIT_DEQUANT_WEIGHT_BUFFERS
            .saturating_mul(QWEN_DIT_FF_DIM)
            .saturating_mul(QWEN_DIT_INNER_DIM)
            .saturating_mul(QWEN_DIT_BF16_BYTES)
    }

    /// What one block's joint attention holds while it runs: the live joint
    /// streams, the score buffers for one query chunk, and a dequantized
    /// projection weight.
    ///
    /// `query_chunk_rows` is what `crate::attention` would actually chunk the
    /// query axis by — `None` means it materializes the whole score matrix,
    /// which is the pre-#1043 behaviour the retired 14 GB constant was aimed
    /// at.
    fn qwen_dit_attention_phase_bytes(joint_tokens: u64, query_chunk_rows: Option<u64>) -> u64 {
        let rows = query_chunk_rows.unwrap_or(joint_tokens).min(joint_tokens);
        let streams = QWEN_DIT_ATTENTION_LIVE_STREAMS
            .saturating_mul(Self::qwen_dit_joint_stream_bytes(joint_tokens));
        let scores = QWEN_DIT_ATTENTION_SCORE_BUFFERS
            .saturating_mul(QWEN_CFG_BATCH)
            .saturating_mul(QWEN_DIT_HEADS)
            .saturating_mul(rows)
            .saturating_mul(joint_tokens)
            .saturating_mul(QWEN_DIT_BF16_BYTES);

        streams
            .saturating_add(scores)
            .saturating_add(Self::qwen_dit_dequant_weight_bytes())
    }

    /// What one block's feed-forward holds: the live joint streams, the
    /// `4 x inner` projection and its GELU, a dequantized weight, and the F32
    /// activation copy the opt-in MMQ arm makes before quantizing to Q8_1.
    fn qwen_dit_mlp_phase_bytes(joint_tokens: u64) -> u64 {
        let streams = QWEN_DIT_MLP_LIVE_STREAMS
            .saturating_mul(Self::qwen_dit_joint_stream_bytes(joint_tokens));
        let ff_elems = QWEN_CFG_BATCH
            .saturating_mul(joint_tokens)
            .saturating_mul(QWEN_DIT_FF_DIM);
        let ff = QWEN_DIT_MLP_LIVE_FF_BUFFERS
            .saturating_mul(ff_elems)
            .saturating_mul(QWEN_DIT_BF16_BYTES);
        let mmq_activation = ff_elems.saturating_mul(QWEN_DIT_MMQ_ACTIVATION_BYTES);

        streams
            .saturating_add(ff)
            .saturating_add(mmq_activation)
            .saturating_add(Self::qwen_dit_dequant_weight_bytes())
    }

    /// Non-weight VRAM the quantized CUDA denoise needs on top of the resident
    /// transformer, derived from the shapes one block actually allocates.
    ///
    /// The 60 blocks run one at a time and each releases before the next, so
    /// the peak is the widest single block, and inside a block the attention
    /// and feed-forward phases are sequential — hence the max of the two, not
    /// their sum, exactly as the VAE decode reserve treats its phases (#1046).
    ///
    /// This replaces a flat `14 GB scaled by pixel count`, which priced the
    /// pre-#1043 attention. That is where the order of magnitude came from:
    /// unchunked `[batch, heads, seq, seq]` BF16 score buffers are 15.8 GB at
    /// 1328² with batched CFG, and this same derivation with
    /// `query_chunk_rows = None` returns 17.7 GB. The derivation deliberately
    /// makes **no claim to reproduce** the retired constant — it does not, and
    /// the two are not even the same quantity: 17.7 GB is a phase, which only
    /// becomes a headroom after the 1.5x margin (26.6 GB). Chunking the query
    /// axis 512 rows at a time (`attention::CUDA_AUTO_QUERY_CHUNK`) drops the
    /// score term to 1.09 GB and the whole estimate to 4.59 GB at 1328², which
    /// is what makes batched CFG admissible for a q4 checkpoint on a 24 GB
    /// card at native resolution.
    ///
    /// The chunk is read from `crate::attention` rather than restated, so
    /// `MOLD_ATTN_CHUNK` moves the estimate with the allocation — including
    /// `off`, which restores the full matrix and forces split CFG back.
    fn quantized_cuda_cfg_headroom(width: usize, height: usize) -> u64 {
        let joint_tokens = Self::qwen_dit_joint_tokens(width, height);
        let chunk_rows =
            crate::attention::cuda_query_chunk_rows(joint_tokens.min(usize::MAX as u64) as usize)
                .map(|rows| rows as u64);
        Self::quantized_cuda_cfg_headroom_for_chunk(joint_tokens, chunk_rows)
    }

    /// [`quantized_cuda_cfg_headroom`] with the query chunk supplied, so the
    /// formula can be pinned without depending on process-frozen env.
    fn quantized_cuda_cfg_headroom_for_chunk(
        joint_tokens: u64,
        query_chunk_rows: Option<u64>,
    ) -> u64 {
        let peak = Self::qwen_dit_attention_phase_bytes(joint_tokens, query_chunk_rows)
            .max(Self::qwen_dit_mlp_phase_bytes(joint_tokens));
        peak.saturating_mul(QWEN_CFG_HEADROOM_SAFETY_NUM)
            .saturating_div(QWEN_CFG_HEADROOM_SAFETY_DEN)
            .max(QWEN_GGUF_MIN_CFG_HEADROOM)
    }

    fn should_split_cfg_quantized_cuda(
        is_edit_family: bool,
        transformer_size: u64,
        free_vram: u64,
        width: usize,
        height: usize,
    ) -> bool {
        // The edit path concatenates output and conditioning image tokens.
        // Batched true-CFG doubles that already irregular sequence and has
        // triggered an out-of-bounds CUDA write in GGUF q4 at 800x1312. Keep
        // quantized edits on the two-pass CFG path regardless of apparent
        // headroom; synthesis can still batch when its measured peak fits.
        if is_edit_family {
            return true;
        }
        if free_vram == 0 {
            // If VRAM probing fails, bias toward the safer split-CFG path
            // instead of assuming batched CFG will fit.
            return true;
        }
        // Batching stops being a win as the sequence grows: measured on an
        // RTX 4090 (2026-08-14, qwen-image-2512:q4, 20 steps, seed 42, this
        // branch's chunked attention), batched CFG is exactly split parity at
        // 1024² (71.4 s vs 71.5 s denoise, 4096 image tokens) and 24% SLOWER
        // at 1328² (164.9 s vs 133.2 s, 6889 tokens) — the doubled per-chunk
        // score transients outweigh the shared per-step work. Above the
        // measured-neutral point the two sequential passes win regardless of
        // how much VRAM is free.
        if Self::image_tokens(width, height) > QWEN_CFG_BATCH_MAX_IMAGE_TOKENS {
            return true;
        }
        let estimated_peak =
            transformer_size.saturating_add(Self::quantized_cuda_cfg_headroom(width, height));
        estimated_peak > free_vram
    }

    /// Image-token count of a request: latent grid (`/8`) then `2x2` patchify.
    fn image_tokens(width: usize, height: usize) -> u64 {
        ((width / 16) as u64) * ((height / 16) as u64)
    }

    /// Transformer weight bytes on disk.
    ///
    /// Two things read the same number: the weight term of the quantized CUDA
    /// CFG decision, and the VRAM that dropping a resident quantized
    /// transformer hands back to the free pool.
    fn transformer_file_bytes(&self) -> u64 {
        std::fs::metadata(&self.base.paths.transformer)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// The quantized CUDA split-vs-batched CFG decision for one request,
    /// together with the readings it was taken from (both wanted by the
    /// progress line the load path prints).
    ///
    /// `resident_transformer_bytes` is added back to the free reading because
    /// the caller may be asking *while the transformer this decision covers is
    /// still resident*: the comparison `should_split_cfg_quantized_cuda` makes
    /// is `transformer_size + headroom > free`, and on the load path `free` is
    /// read before those weights land. Adding them back is what makes a
    /// resident engine ask the same question as a fresh load at the same
    /// resolution — the two must agree, or the decision would depend on when
    /// it was taken. It is zero on the load/reload path, where nothing is
    /// resident. The add-back is skipped when the VRAM probe itself failed, so
    /// a failed probe still reads as zero free and biases to the safer split
    /// path.
    fn decide_quantized_split_cfg(
        &self,
        device: &Device,
        width: usize,
        height: usize,
        resident_transformer_bytes: u64,
    ) -> QuantizedCfgDecision {
        let transformer_size = self.transformer_file_bytes();
        let free = usable_free_vram_bytes(self.base.gpu_ordinal)
            .map(|free| free.saturating_add(resident_transformer_bytes))
            .unwrap_or(0);
        let split = device.is_cuda()
            && (self.offload
                || Self::should_split_cfg_quantized_cuda(
                    self.is_edit_family(),
                    transformer_size,
                    free,
                    width,
                    height,
                ));
        QuantizedCfgDecision {
            split,
            transformer_size,
            free,
        }
    }

    /// The CFG-batching flag a resident quantized transformer should carry for
    /// this request, or `None` when it already carries it.
    ///
    /// The flag is a **budget** decision, not a property of the built weights:
    /// `QuantizedQwenImageTransformer2DModel::new` ignores it entirely and the
    /// forward is shape-agnostic in the batch axis. So a model loaded at the
    /// native 1328² shape does not have to be rebuilt to batch CFG for a 768²
    /// request that fits — the `bool` is simply re-set in place, which is also
    /// why nothing here can oscillate: no rebuild means no second, differently
    /// sourced VRAM reading to disagree with the first.
    ///
    /// The `None` input is every non-quantized transformer (BF16, FP8,
    /// offloaded): they carry no flag, so there is nothing to move.
    fn qwen_cfg_batching_update(
        current_supports_batching: Option<bool>,
        request_splits_cfg: bool,
    ) -> Option<bool> {
        // `supports_cfg_batching` is the negation of the split decision.
        let wanted = !request_splits_cfg;
        match current_supports_batching {
            Some(current) if current != wanted => Some(wanted),
            _ => None,
        }
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
        let active_loras = &self.pending_loras;
        let has_lora = !active_loras.is_empty();
        if self.detect_is_quantized() {
            // Reserve-adjusted reading: split-CFG is a budget decision.
            // Nothing is resident here — every build site drops the old
            // transformer first — so no bytes are added back.
            let QuantizedCfgDecision {
                split: split_cfg_for_memory,
                transformer_size,
                free,
            } = self.decide_quantized_split_cfg(device, width, height, 0);
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
            let vb = if has_lora {
                let adapters = super::lora::load_lora_adapters(active_loras, &self.base.progress)?;
                let specs: Vec<super::lora::QwenImageLoraSpec<'_>> = adapters
                    .iter()
                    .zip(active_loras.iter())
                    .map(|(adapter, w)| super::lora::QwenImageLoraSpec {
                        adapter: adapter.as_ref(),
                        scale: w.scale,
                        path_hash: super::lora::lora_path_hash(&w.path),
                    })
                    .collect();
                super::lora::gguf_lora_var_builder(
                    &self.base.paths.transformer,
                    &specs,
                    device,
                    &self.base.progress,
                    None,
                )?
            } else {
                quantized_var_builder::VarBuilder::from_gguf(&self.base.paths.transformer, device)?
            };
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
            let mut mem_size: u64 = xformer_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            // `MOLD_QWEN_FP8_CACHE=1` retains a BF16 copy of every widened
            // FP8 layer (2 bytes/param on top of the 1-byte artifact), so the
            // budget admission and `should_offload` reason about must include
            // it — otherwise a card that fits plain FP8 is admitted
            // non-offloaded and OOMs on the first forward.
            if is_fp8
                && super::transformer::parse_qwen_fp8_cache(
                    crate::runtime_env::value("MOLD_QWEN_FP8_CACHE").as_deref(),
                )
            {
                mem_size = mem_size.saturating_mul(3);
            }
            // Reserve-adjusted reading: should_offload budgets against this.
            let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
            // Qwen-Image runs CFG by default; activation budget scales with
            // resolution to replace the previous fixed 3 GB heuristic.
            let activation_budget = crate::device::activation_bytes(
                width as u32,
                height as u32,
                2,
                crate::device::dtype_bytes(dtype),
                crate::device::ActivationFamily::QwenImageDit,
            );
            let use_offload =
                self.offload || crate::device::should_offload(mem_size, free, activation_budget);

            if is_fp8 {
                self.base
                    .progress
                    .info("Detected FP8 safetensors — loading with scale dequantization");
            }

            if use_offload {
                if has_lora {
                    bail!(
                        "Qwen-Image LoRA support is not yet wired through the block-offload \
                         transformer path. Disable offload (drop --offload / unset MOLD_OFFLOAD), \
                         or pick a checkpoint that fits without offload, to use LoRAs."
                    );
                }
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
                        self.base.gpu_ordinal,
                        activation_budget,
                        &self.base.progress,
                    )?,
                ))
            } else {
                let xformer_vb = if has_lora {
                    self.build_bf16_lora_var_builder(
                        &xformer_paths,
                        dtype,
                        device,
                        is_fp8,
                        active_loras,
                    )?
                } else if is_fp8 {
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

    /// Construct a `VarBuilder` for the BF16/FP8 in-memory path with a
    /// LoRA-merging `SimpleBackend` wrapping the underlying mmap (or
    /// `NativeFp8Backend`). Each `vb.get()` call delivers a tensor with
    /// `W' = W + scale·(B @ A)` already merged in.
    fn build_bf16_lora_var_builder<'a>(
        &self,
        xformer_paths: &[std::path::PathBuf],
        dtype: DType,
        device: &Device,
        is_fp8: bool,
        loras: &[mold_core::LoraWeight],
    ) -> Result<candle_nn::VarBuilder<'a>> {
        let adapters = super::lora::load_lora_adapters(loras, &self.base.progress)?;
        let specs: Vec<super::lora::QwenImageLoraSpec<'_>> = adapters
            .iter()
            .zip(loras.iter())
            .map(|(adapter, w)| super::lora::QwenImageLoraSpec {
                adapter: adapter.as_ref(),
                scale: w.scale,
                path_hash: super::lora::lora_path_hash(&w.path),
            })
            .collect();

        let path_refs: Vec<&std::path::Path> = xformer_paths.iter().map(|p| p.as_path()).collect();
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&path_refs)? };
        let inner: Box<dyn candle_nn::var_builder::SimpleBackend> = if is_fp8 {
            // FP8 path needs the `NativeFp8Backend` so F8E4M3 weights
            // stay F8E4M3 in VRAM; the LoRA wrapper merges deltas in
            // F32 and the per-layer dequant in `QwenLinear::Fp8::forward`
            // sees pre-merged weights as expected.
            self.base
                .progress
                .info("Detected FP8 safetensors — loading with LoRA-merging wrapper");
            Box::new(crate::weight_loader::NativeFp8Backend::from_mmap(tensors))
        } else {
            // candle's `MmapedSafetensors` implements `SimpleBackend`
            // directly; use it as the inner layer of the LoRA wrapper.
            Box::new(tensors)
        };

        let wrapped =
            super::lora::wrap_backend_with_lora(inner, &specs, &self.base.progress, None)?;

        let target_dtype = if is_fp8 { DType::BF16 } else { dtype };
        Ok(candle_nn::VarBuilder::from_backend(
            wrapped,
            target_dtype,
            device.clone(),
        ))
    }

    /// Load VAE from disk.
    fn load_vae(&self, device: &Device, dtype: DType) -> Result<QwenImageVae> {
        let vb = self.load_vae_var_builder(device, dtype)?;
        Ok(QwenImageVae::from_var_builder(vb, device, dtype)?)
    }

    fn load_vae_cpu_tensors(&self) -> Result<Option<Arc<HashMap<String, Tensor>>>> {
        let Some(shared_pool) = &self.shared_pool else {
            return Ok(None);
        };
        shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&self.base.paths.vae))
    }

    fn load_vae_var_builder<'a>(
        &self,
        device: &Device,
        dtype: DType,
    ) -> Result<candle_nn::VarBuilder<'a>> {
        if let Some(tensors) = self.load_vae_cpu_tensors()? {
            return Ok(encoders::park::varbuilder_from_parked(
                tensors.as_ref(),
                dtype,
                device,
            ));
        }

        crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&self.base.paths.vae),
            dtype,
            device,
            "Qwen-Image VAE",
            &self.base.progress,
        )
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
        self.resolve_text_encoder_source_with_preference(
            gpu_device,
            free_vram,
            usage,
            self.qwen2_variant.as_deref(),
        )
    }

    fn resolve_text_encoder_source_with_preference(
        &self,
        gpu_device: &Device,
        free_vram: u64,
        usage: Qwen2TextEncoderUsage,
        preference: Option<&str>,
    ) -> Result<ResolvedQwen2TextEncoder> {
        self.resolve_text_encoder_source_with_preference_using(
            gpu_device,
            free_vram,
            usage,
            preference,
            crate::encoders::variant_resolution::resolve_qwen2_vl_gguf_path,
        )
    }

    fn resolve_text_encoder_source_with_preference_using<F>(
        &self,
        gpu_device: &Device,
        free_vram: u64,
        usage: Qwen2TextEncoderUsage,
        preference: Option<&str>,
        resolve_gguf: F,
    ) -> Result<ResolvedQwen2TextEncoder>
    where
        F: FnOnce(
            &ProgressReporter,
            &mold_core::manifest::Qwen2VlVariant,
        ) -> Result<std::path::PathBuf>,
    {
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
        let is_edit_family = self.is_edit_family();
        let mut resolved = Self::choose_text_encoder_source(
            preference,
            is_cuda,
            is_metal,
            free_vram,
            bf16_size_bytes,
            if is_edit_family {
                Qwen2TextEncoderUsage::Resident
            } else {
                usage
            },
        )?;

        if resolved.is_gguf {
            let variant = mold_core::manifest::find_qwen2_vl_variant(&resolved.variant_label)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown Qwen2.5-VL variant '{}'", resolved.variant_label)
                })?;
            resolved.paths = vec![resolve_gguf(&self.base.progress, variant)?];
        } else {
            resolved.paths = self.base.paths.text_encoder_files.clone();
        }
        resolved.vision_paths = if is_edit_family {
            self.base.paths.text_encoder_files.clone()
        } else {
            vec![]
        };

        if is_edit_family {
            return Ok(resolved);
        }

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
        Self::qwen_transformer_can_stay_hot_for_vae(
            loaded.device.is_cuda(),
            loaded.vae_device.is_cuda(),
            matches!(
                loaded.transformer.as_ref(),
                Some(QwenImageTransformer::Quantized(_))
            ),
        )
    }

    fn qwen_transformer_can_stay_hot_for_vae(
        transformer_is_cuda: bool,
        vae_is_cuda: bool,
        transformer_is_quantized: bool,
    ) -> bool {
        transformer_is_cuda && vae_is_cuda && transformer_is_quantized
    }

    /// Whether this request has to rebuild the transformer.
    ///
    /// Rebuilding a Qwen-Image transformer with a LoRA stack is the
    /// expensive case: the GGUF path dequantizes, merges and re-quantizes
    /// every affected tensor across all 60 blocks. A resident transformer
    /// whose baked stack is byte-for-byte the request's stack is reused;
    /// any difference — adapter set, order, scale — invalidates it, as
    /// does having no resident transformer at all.
    ///
    /// The LoRA merge is the *only* thing baked into the built weights. The
    /// quantized transformer's `supports_cfg_batching` is deliberately not a
    /// rebuild input: it has no structural effect, so it is re-set in place
    /// per request (`qwen_cfg_batching_update`). Making it one paid a full
    /// GGUF dequantize → merge → re-quantize to flip a `bool`, and — because
    /// the check and the rebuild read free VRAM from two different,
    /// systematically unequal sources — could repeat that on every request.
    fn qwen_transformer_rebuild_needed(
        transformer_resident: bool,
        baked_lora: &[QwenImageLoraFingerprint],
        requested_lora: &[QwenImageLoraFingerprint],
    ) -> bool {
        !transformer_resident || baked_lora != requested_lora
    }

    fn decode_vae_gpu_only(
        latents: &Tensor,
        vae: &QwenImageVae,
        vae_device: &Device,
        sync_device: &Device,
        progress: &ProgressReporter,
        prefer_tiled: bool,
    ) -> Result<Tensor> {
        if vae_device.is_cuda() && prefer_tiled {
            progress.info("Selecting tiled GPU VAE decode proactively");
            return Self::decode_vae_tiled(latents, vae, vae_device, progress);
        }

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
        tokenizer: Arc<Tokenizer>,
        device: &Device,
        dtype: DType,
        preload_weights: bool,
    ) -> Result<encoders::qwen2_text::Qwen2TextEncoder> {
        if resolved.is_gguf {
            if preload_weights {
                encoders::qwen2_text::Qwen2TextEncoder::load_gguf_with_tokenizer(
                    &resolved.paths[0],
                    tokenizer_path,
                    Some(tokenizer),
                    device,
                    dtype,
                    &resolved.vision_paths,
                    &self.base.progress,
                )
            } else {
                encoders::qwen2_text::Qwen2TextEncoder::prepare_gguf_with_tokenizer(
                    &resolved.paths[0],
                    tokenizer_path,
                    Some(tokenizer),
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
                encoders::qwen2_text::Qwen2TextEncoder::load_bf16_with_tokenizer(
                    &resolved.paths,
                    tokenizer_path,
                    Some(tokenizer),
                    device,
                    dtype,
                    self.is_edit_family(),
                    &self.base.progress,
                )
            } else {
                encoders::qwen2_text::Qwen2TextEncoder::prepare_bf16_with_tokenizer(
                    &resolved.paths,
                    tokenizer_path,
                    Some(tokenizer),
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
            self.qwen2_text_encoder_mode,
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

        // Sequential mode defers loading to generate_sequential(), which
        // builds and drops a request-local transformer — nothing resident
        // to fingerprint.
        if self.base.load_strategy == LoadStrategy::Sequential {
            self.active_lora_fingerprint.clear();
            return Ok(());
        }

        tracing::info!(model = %self.base.model_name, "loading Qwen-Image model components...");

        let text_tokenizer_path = self.validate_paths()?;
        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
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

        // Decide device placement for VAE and text encoder.
        // Log raw, budget against the reserve-adjusted reading.
        let free_raw = free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free_raw > 0 {
            self.base.progress.info(&format!(
                "Free VRAM after transformer: {}",
                fmt_gb(free_raw)
            ));
        }

        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_DECODE_VRAM_THRESHOLD);
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || {
            Ok(if vae_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            })
        })?;
        let vae_on_gpu = !vae_device.is_cpu();
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
        let (te_plan, te_auto_device_label) =
            self.resolve_text_encoder_plan(&device, &resolved_text_encoder, free);
        let qwen_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| adv.qwen.clone(),
            true,
        );
        let auto_te_device = if te_plan.use_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_device =
            crate::device::resolve_device(Some(qwen_ref), || Ok(auto_te_device.clone()))?;
        let te_use_gpu = !te_device.is_cpu();
        let te_device_label: String = if te_use_gpu == te_plan.use_gpu {
            te_auto_device_label
        } else if te_use_gpu {
            "GPU".into()
        } else {
            "CPU".into()
        };
        let te_dtype = Self::text_encoder_load_dtype(te_use_gpu, dtype);

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
        let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;
        let text_encoder = self.load_text_encoder(
            &resolved_text_encoder,
            &text_tokenizer_path,
            text_tokenizer,
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

        // The transformer above was built through `load_transformer`, which
        // merges `pending_loras`; record what it carries so the first
        // generate does not immediately rebuild it.
        self.note_transformer_lora_stack();
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

    /// Reload the transformer from disk, recording the LoRA stack that the
    /// rebuilt weights carry so the next request can elide the rebuild.
    fn reload_transformer(
        &mut self,
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
        self.install_reloaded_transformer(&mut loaded.transformer, transformer);
        Ok(())
    }

    /// Install weights returned by the real loader and bind their residency
    /// authority to the exact pending LoRA stack that loader consumed.
    fn install_reloaded_transformer<T>(&mut self, slot: &mut Option<T>, transformer: T) {
        *slot = Some(transformer);
        self.note_transformer_lora_stack();
    }

    /// Record the stack that a just-built transformer carries.
    ///
    /// `load_transformer` merges `pending_loras`, so the fingerprint is
    /// always derived from that same slice — never from the request, which
    /// is what made the original defect possible. Both build sites (eager
    /// `load` and `reload_transformer`) go through here so the derivation
    /// cannot drift between them.
    fn note_transformer_lora_stack(&mut self) {
        self.active_lora_fingerprint = fingerprint_stack(&self.pending_loras);
    }

    /// Drop the resident transformer and forget the stack baked into it.
    ///
    /// The two must move together: a cleared transformer left with a stale
    /// fingerprint would let the next request elide a rebuild it genuinely
    /// needs, which is the bug this whole fingerprint exists to prevent.
    /// Every drop site (both VAE-decode drops and the changed-stack rebuild)
    /// routes through here so neither half can be forgotten alone.
    fn release_resident_transformer<T>(
        transformer: &mut Option<T>,
        active_lora_fingerprint: &mut Vec<QwenImageLoraFingerprint>,
    ) {
        *transformer = None;
        active_lora_fingerprint.clear();
    }

    fn release_edit_transformer<T>(&mut self, transformer: &mut Option<T>) {
        Self::release_resident_transformer(transformer, &mut self.active_lora_fingerprint);
    }

    fn finish_edit_generation<T>(&mut self, result: Result<T>) -> Result<T> {
        if self.base.load_strategy == LoadStrategy::Sequential {
            self.unload();
        }
        result
    }

    /// Make the resident transformer match this request's LoRA stack.
    ///
    /// Reused by the synthesis and edit generate paths. A resident
    /// transformer built with the same stack is kept as-is — that is the
    /// whole point: the quantized stay-hot path otherwise pays a full
    /// dequantize → merge → re-quantize of every LoRA-touched tensor on
    /// every request. A changed stack drops the old transformer (and
    /// synchronizes) before the rebuild so the merge is not asked to fit
    /// two transformers into VRAM at once.
    ///
    /// It also makes a resident *quantized* transformer re-decide split vs
    /// batched CFG at this request's real resolution — a 1328²-loaded engine
    /// would otherwise keep two-pass CFG for every smaller request that fits
    /// batched. That re-decision is a `bool` write on the resident model, not
    /// a rebuild: the flag has no structural effect, so paying a rebuild for
    /// it would be pure waste, and the two VRAM readings a rebuild-and-check
    /// cycle needs are systematically unequal (the drop returns more than the
    /// GGUF's disk size — dequantized norms and cached RoPE views have no
    /// on-disk counterpart), which made the rebuild able to repeat forever
    /// for any request whose estimate landed between them.
    fn ensure_transformer_for_request(&mut self, width: usize, height: usize) -> Result<()> {
        let requested = fingerprint_stack(&self.pending_loras);
        let (resident, current_cfg_batching, device) = {
            let loaded = self
                .base
                .loaded
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
            let current = match loaded.transformer.as_ref() {
                Some(QwenImageTransformer::Quantized(model)) => Some(model.supports_cfg_batching()),
                _ => None,
            };
            (loaded.transformer.is_some(), current, loaded.device.clone())
        };
        let lora_stack_changed = self.active_lora_fingerprint != requested;
        if !Self::qwen_transformer_rebuild_needed(
            resident,
            &self.active_lora_fingerprint,
            &requested,
        ) {
            // Nothing to rebuild — but a resident quantized transformer still
            // re-takes its CFG budget decision at this request's resolution.
            // Only that arm carries the flag, so only it pays for the VRAM
            // probe the decision needs.
            if current_cfg_batching.is_some() {
                let split = self
                    .decide_quantized_split_cfg(
                        &device,
                        width,
                        height,
                        // The load path reads free VRAM before the weights
                        // land; adding the resident bytes back is what makes
                        // this ask the same question at the same resolution.
                        self.transformer_file_bytes(),
                    )
                    .split;
                if let Some(supports) = Self::qwen_cfg_batching_update(current_cfg_batching, split)
                {
                    self.set_resident_cfg_batching(supports, width, height);
                }
            }
            return Ok(());
        }

        let mut loaded_mut = self
            .base
            .loaded
            .take()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        if resident {
            Self::release_resident_transformer(
                &mut loaded_mut.transformer,
                &mut self.active_lora_fingerprint,
            );
            loaded_mut.device.synchronize()?;
        }
        // Reaching here means `!resident || lora_stack_changed`, so these two
        // labels are exhaustive.
        debug_assert!(!resident || lora_stack_changed);
        let label = if resident {
            "Rebuilding Qwen-Image transformer for the requested LoRA stack"
        } else {
            "Reloading Qwen-Image transformer"
        };
        self.base.progress.stage_start(label);
        let reload_start = Instant::now();
        let result = self.reload_transformer(&mut loaded_mut, width, height);
        if result.is_ok() {
            self.base.progress.stage_done(label, reload_start.elapsed());
        }
        // Put the rest of the engine back either way: a failed rebuild
        // leaves the same transformer-less state a VAE-decode drop does,
        // rather than silently unloading every other component.
        self.base.loaded = Some(loaded_mut);
        result
    }

    /// Move the resident quantized transformer's CFG-batching flag and say so.
    ///
    /// The load path prints its split-CFG choice, so a later change to that
    /// choice has to be visible too — otherwise the log claims a mode the
    /// request did not run in.
    fn set_resident_cfg_batching(&mut self, supports: bool, width: usize, height: usize) {
        let moved = self
            .base
            .loaded
            .as_mut()
            .and_then(|loaded| loaded.transformer.as_mut())
            .is_some_and(|transformer| transformer.set_supports_cfg_batching(supports));
        if moved {
            self.base.progress.info(&format!(
                "Switching resident quantized Qwen transformer to {} CFG at {}x{}",
                if supports {
                    "batched"
                } else {
                    "low-memory split"
                },
                width,
                height,
            ));
        }
    }

    /// Generate using sequential loading strategy (load-use-drop each component).
    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let text_tokenizer_path = self.validate_paths()?;
        let transformer_cfg = self.transformer_config();
        // The checkpoint's own packaged scheduler config, not the family's.
        let shift_policy = shift_policy_for_model(&self.base.model_name);

        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.transformer.clone()),
            false,
        );
        let device = crate::device::resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
        let dtype = crate::engine::gpu_dtype(&device);
        let transformer_is_quantized = self.detect_is_quantized();

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let width = req.width as usize;
        let height = req.height as usize;
        // Reserve-adjusted reading: text-encoder source / placement is a
        // budget decision.
        let free = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
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

        let (mut encoder_hidden_states, mut uncond_hs) = if both_cached {
            self.base.progress.cache_hit("prompt conditioning");
            let cached = prompt_cached.unwrap();
            let restore_device = if use_cpu_staging {
                &Device::Cpu
            } else {
                &device
            };
            let restore_dtype = if use_cpu_staging { DType::F32 } else { dtype };
            let hs = cached.restore(restore_device, restore_dtype)?;
            let u_hs = if use_cfg {
                let ucached = uncond_cached.unwrap();
                Some(ucached.restore(restore_device, restore_dtype)?)
            } else {
                None
            };
            (hs, u_hs)
        } else {
            let (te_plan, te_auto_device_label) =
                self.resolve_text_encoder_plan(&device, &resolved_text_encoder, free);
            let qwen_ref = effective_device_ref(
                self.pending_placement.as_ref(),
                |adv| adv.qwen.clone(),
                true,
            );
            let auto_te_device = if te_plan.use_gpu {
                device.clone()
            } else {
                Device::Cpu
            };
            let te_device =
                crate::device::resolve_device(Some(qwen_ref), || Ok(auto_te_device.clone()))?;
            let te_use_gpu = !te_device.is_cpu();
            let te_device_label: String = if te_use_gpu == te_plan.use_gpu {
                te_auto_device_label
            } else if te_use_gpu {
                "GPU".into()
            } else {
                "CPU".into()
            };
            let te_dtype = Self::text_encoder_load_dtype(te_use_gpu, dtype);

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
            // Reuse the encoder retained by the previous sequential render
            // when this one was planned for the same weights, device, and
            // dtype; otherwise the retained one is stale and is dropped before
            // the fresh load so both are never resident.
            let retained = self.retained_sequential_text_encoder.take();
            let reusable = match retained {
                Some(retained)
                    if retained.matches(&resolved_text_encoder.paths, &te_device, te_dtype) =>
                {
                    Some(retained)
                }
                stale => {
                    if stale.is_some() {
                        // Visible on purpose: a silent rejection here is a
                        // 16 GB disk re-read that looks like a cold load.
                        tracing::warn!(
                            "retained Qwen2.5 encoder rejected as stale (paths/device/dtype changed); reloading from disk"
                        );
                    }
                    // Explicit: a `_` arm would keep the stale encoder's
                    // several GB alive until the end of the match, i.e.
                    // across the fresh load.
                    drop(stale);
                    None
                }
            };

            let mut text_encoder = if let Some(mut retained) = reusable {
                let label = if retained.is_parked() {
                    "Unparking Qwen2.5 encoder (CPU→GPU)"
                } else {
                    "Reloading Qwen2.5 encoder"
                };
                self.base.progress.stage_start(label);
                let unpark_start = Instant::now();
                retained.unpark_to_gpu(&self.base.progress)?;
                self.base.progress.stage_done(label, unpark_start.elapsed());
                retained
            } else {
                if te_plan.use_cpu_staging && device.is_metal() && !resolved_text_encoder.is_gguf {
                    self.base.progress.info(
                        "Skipping hard preflight for Qwen2.5 text encoder on Metal; sequential mode spills prompt conditioning to CPU after encoding",
                    );
                } else {
                    let te_activation_budget = crate::device::activation_bytes(
                        req.width,
                        req.height,
                        1,
                        crate::device::dtype_bytes(te_dtype),
                        crate::device::ActivationFamily::SmallTransformer,
                    );
                    preflight_memory_check(
                        "Qwen2.5 text encoder",
                        resolved_text_encoder.size_bytes,
                        te_activation_budget,
                    )?;
                }

                if let Some(status) = memory_status_string() {
                    self.base.progress.info(&status);
                }

                self.base.progress.stage_start(&te_label);
                let te_start = Instant::now();
                let text_tokenizer = self.load_text_tokenizer(&text_tokenizer_path)?;
                let text_encoder = self.load_text_encoder(
                    &resolved_text_encoder,
                    &text_tokenizer_path,
                    text_tokenizer,
                    &te_device,
                    te_dtype,
                    true,
                )?;
                self.base.progress.stage_done(&te_label, te_start.elapsed());
                text_encoder
            };

            let hs = Self::encode_prompt_cached(
                &self.base.progress,
                &self.prompt_cache,
                &mut text_encoder,
                &req.prompt,
                &device,
                dtype,
            )?;
            let hs = Self::maybe_spill_conditioning(use_cpu_staging, hs)?;

            let u_hs = if use_cfg {
                let hs = Self::encode_prompt_cached(
                    &self.base.progress,
                    &self.prompt_cache,
                    &mut text_encoder,
                    QWEN_EMPTY_NEGATIVE_PROMPT,
                    &device,
                    dtype,
                )?;
                Some(Self::maybe_spill_conditioning(use_cpu_staging, hs)?)
            } else {
                None
            };

            // Under `MOLD_KEEP_TE_RAM=1` the encoder moves to host RAM instead
            // of vanishing, so the next sequential render skips the disk read
            // — 35.1 s for the GGUF encoder (#1044). The device memory is
            // released either way, which is what sequential mode is for.
            // Metal is unified memory, so "host RAM" is the same pool there.
            if crate::device::keep_te_in_ram() && !te_device.is_metal() {
                text_encoder.park_to_cpu()?;
                self.retained_sequential_text_encoder = Some(text_encoder);
            } else {
                drop(text_encoder);
            }
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
                    self.base
                        .progress
                        .info("Freed Qwen2.5 text encoder and spilled prompt conditioning to CPU");
                } else {
                    self.base.progress.info("Freed Qwen2.5 text encoder");
                }
            }

            (hs, u_hs)
        };

        // Conditioning stays at its true length here. Padding the two CFG
        // streams to a common length is a property of *batching* them into one
        // forward, so it happens in the `use_batched_cfg` branch below and
        // nowhere else.

        // --- Phase 2: Load transformer and denoise ---
        let xformer_paths = self.transformer_paths();
        let xformer_size: u64 = xformer_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let xformer_activation_budget = crate::device::activation_bytes(
            req.width,
            req.height,
            if req.guidance > 1.0 { 2 } else { 1 },
            crate::device::dtype_bytes(dtype),
            crate::device::ActivationFamily::QwenImageDit,
        );
        preflight_memory_check(
            "Qwen-Image transformer",
            xformer_size,
            xformer_activation_budget,
        )?;

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
            if let Some(hs) = uncond_hs.take() {
                uncond_hs = Some(hs.to_device(&device)?.to_dtype(dtype)?);
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
            // Reserve-adjusted reading drives the encode-device decision.
            let free_for_encode = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
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
                shift_policy,
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
            QwenImageScheduler::new_img2img(
                req.steps as usize,
                image_seq_len,
                req.strength,
                shift_policy,
            )
        } else {
            let sched = QwenImageScheduler::new(req.steps as usize, image_seq_len, shift_policy);
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
        // extra batch dimension without exceeding peak memory. Only this branch
        // pads, and only when the two prompts differ in length.
        let (batched_hs, batched_mask) = if use_batched_cfg {
            let (cond_hs, neg_hs, mask) = align_cfg_conditioning(
                &encoder_hidden_states,
                uncond_hs.as_ref().expect("unconditional prompt missing"),
            )?;
            (Tensor::cat(&[&cond_hs, &neg_hs], 0)?, mask)
        } else {
            (encoder_hidden_states.clone(), None)
        };

        for step in 0..num_steps {
            self.base.progress.checkpoint()?;
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
                        batched_mask.as_ref(),
                    )?;
                    (batched_pred.narrow(0, 0, 1)?, batched_pred.narrow(0, 1, 1)?)
                } else {
                    let t_tensor =
                        Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;
                    (
                        transformer.forward(&latents, &t_tensor, &encoder_hidden_states, None)?,
                        transformer.forward(
                            &latents,
                            &t_tensor,
                            uncond_hs.as_ref().unwrap(),
                            None,
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
                transformer.forward(&latents, &t_tensor, &encoder_hidden_states, None)?
            };
            if step == 0 || step == num_steps / 2 || step == num_steps - 1 {
                Self::debug_tensor_stats(&format!("noise_pred[{step}]"), &noise_pred);
                Self::debug_tensor_stats(&format!("latents[{step}]"), &latents);
            }
            if step == 0 {
                Self::validate_qwen_tensor_boundary("noise_pred[0]", &noise_pred)?;
            }
            latents = scheduler.step(&noise_pred, &latents)?;
            if step == num_steps - 1 {
                Self::validate_qwen_tensor_boundary("latents_final", &latents)?;
            }

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

        self.base.progress.checkpoint()?;
        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Drop transformer and embeddings
        drop(transformer);
        drop(encoder_hidden_states);
        drop(uncond_hs);
        device.synchronize()?;
        self.base.progress.info("Freed Qwen-Image transformer");

        // --- Phase 3: Load VAE and decode ---
        if let Some(status) = memory_status_string() {
            self.base.progress.info(&status);
        }

        // Reserve-adjusted reading: VAE placement is a budget decision.
        let free_for_vae = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let vae_on_gpu = should_use_gpu(
            device.is_cuda(),
            device.is_metal(),
            free_for_vae,
            VAE_DECODE_VRAM_THRESHOLD,
        );
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |adv| Some(adv.vae.clone()),
            false,
        );
        let vae_device = crate::device::resolve_device(Some(vae_ref), || {
            Ok(if vae_on_gpu {
                device.clone()
            } else {
                Device::Cpu
            })
        })?;
        let vae_on_gpu = !vae_device.is_cpu();
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
        let free_for_decode = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let prefer_tiled = Self::should_proactively_tile_vae_decode(
            req.width,
            req.height,
            vae_device.is_cuda(),
            free_for_decode,
        );

        let image = Self::decode_vae_with_fallback(
            &latents,
            &vae,
            &vae_device,
            &device,
            &self.base.progress,
            prefer_tiled,
            || self.load_vae(&Device::Cpu, DType::F32),
        )?;
        Self::validate_qwen_tensor_boundary("image_pre_postprocess", &image)?;
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        let post_stats = Self::validate_qwen_tensor_boundary("image_postprocess", &image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;
        if Self::near_black_image_stats(post_stats) {
            self.base.progress.info(
                "Qwen diagnostic: decoded image is near-black after VAE postprocess; inspect MOLD_QWEN_DEBUG tensor stats to separate denoise math from VAE decode",
            );
            tracing::warn!(
                min = post_stats.min,
                max = post_stats.max,
                mean = post_stats.mean,
                "Qwen decoded image is near-black after VAE postprocess"
            );
        }

        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "VAE decode",
            vae_decode_start.elapsed(),
        );

        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.resolved_output_format(),
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
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: image_bytes,
                format: req.resolved_output_format(),
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }

    fn generate_edit_loaded(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Started before conditioning so `generation_time_ms` includes the
        // transformer reload and any request LoRA merge performed after the
        // multimodal encoder has been released.
        let start = Instant::now();

        // Edit conditioning runs the Qwen2.5-VL language/vision encoder. It is
        // materially larger than text-only conditioning and must not overlap
        // the resident diffusion transformer on a 24 GB card. Keep the loaded
        // bundle outside `self.base` while the inner routine moves through the
        // conditioning -> VAE encode -> transformer phases; putting it back on
        // every exit preserves the engine even when one phase fails.
        let mut loaded = self
            .base
            .loaded
            .take()
            .ok_or_else(|| anyhow::anyhow!("model not loaded"))?;
        let result = self.generate_edit_loaded_inner(req, start, &mut loaded);
        self.base.loaded = Some(loaded);
        result
    }

    fn generate_edit_loaded_inner(
        &mut self,
        req: &GenerateRequest,
        start: Instant,
        loaded: &mut LoadedQwenImage,
    ) -> Result<GenerateResponse> {
        // The checkpoint's own packaged scheduler config, not the family's.
        let shift_policy = shift_policy_for_model(&self.base.model_name);
        // Read before the long `&mut self.base.loaded` borrow below: the
        // sequential edit route in `generate_inner` unloads the engine the
        // moment this returns, which decides park vs drop.
        let engine_unloads_after = self.base.load_strategy == LoadStrategy::Sequential;

        let is_edit_family = self.is_edit_family();
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

        // The GGUF memory model is phase-sequential: Qwen2.5-VL conditioning,
        // edit-image VAE encoding, then diffusion. The old runtime kept the
        // transformer resident through the first two phases, so a 768x768 q4
        // edit could pass a ~16.6 GB admission estimate and still OOM on a
        // 24 GB RTX 4090 while the multimodal encoder loaded. Make runtime
        // residency match the plan before allocating any encoder weights.
        if loaded.transformer.is_some() {
            self.base
                .progress
                .info("Releasing Qwen-Image transformer before multimodal edit conditioning");
            self.release_edit_transformer(&mut loaded.transformer);
            loaded.device.synchronize()?;
        }

        if loaded.text_encoder.model.is_none() {
            let label = if loaded.text_encoder.is_parked() {
                "Unparking Qwen2.5 encoder (CPU→GPU)"
            } else {
                "Reloading Qwen2.5 encoder"
            };
            self.base.progress.stage_start(label);
            let reload_start = Instant::now();
            if loaded.text_encoder.is_parked() {
                loaded.text_encoder.unpark_to_gpu(&self.base.progress)?;
            } else {
                loaded.text_encoder.reload(&self.base.progress)?;
            }
            self.base.progress.stage_done(label, reload_start.elapsed());
        }

        self.base
            .progress
            .stage_start("Encoding prompt (Qwen2.5 edit)");
        let encode_start = Instant::now();
        let (encoder_hidden_states, _, _) = loaded.text_encoder.encode_formatted_multimodal(
            &formatted_prompt,
            edit_images,
            &loaded.device,
            loaded.dtype,
        )?;
        self.base.progress.phase_done(
            crate::ProgressPhase::PromptEncode,
            "Encoding prompt (Qwen2.5 edit)",
            encode_start.elapsed(),
        );
        let uncond_hs = if use_cfg {
            self.base
                .progress
                .stage_start("Encoding negative prompt (Qwen2.5 edit)");
            let neg_start = Instant::now();
            let (hs, _, _) = loaded.text_encoder.encode_formatted_multimodal(
                &formatted_negative,
                edit_images,
                &loaded.device,
                loaded.dtype,
            )?;
            self.base.progress.stage_done(
                "Encoding negative prompt (Qwen2.5 edit)",
                neg_start.elapsed(),
            );
            Some(hs)
        } else {
            None
        };

        let drop_text_encoder = is_edit_family || loaded.text_encoder.on_gpu;
        if drop_text_encoder {
            let park_mode =
                Self::qwen2_edit_text_encoder_should_park(Qwen2EditTextEncoderReleaseInput {
                    on_gpu: loaded.text_encoder.on_gpu,
                    is_metal: loaded.device.is_metal(),
                    keep_te_ram: crate::device::keep_te_in_ram(),
                    engine_unloads_after,
                });
            if park_mode {
                let parked_bytes = loaded.text_encoder.weights_size_bytes();
                loaded.text_encoder.park_to_cpu()?;
                tracing::info!(
                    on_gpu = loaded.text_encoder.on_gpu,
                    parked_bytes,
                    "Qwen2.5 text encoder parked to CPU host RAM after edit conditioning"
                );
            } else {
                loaded.text_encoder.drop_weights();
                tracing::info!(
                    on_gpu = loaded.text_encoder.on_gpu,
                    "Qwen2.5 text encoder dropped after edit conditioning"
                );
            }
        }

        let mut packed_input_storage = Vec::with_capacity(edit_images.len());
        let mut img_shapes = vec![(1usize, height / 16, width / 16)];
        self.base.progress.stage_start("Encoding edit images (VAE)");
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
                &self.base.progress,
                || {
                    Ok(QwenImageVae::load(
                        &loaded.vae_path,
                        &Device::Cpu,
                        DType::F32,
                        &self.base.progress,
                    )?)
                },
            )?
            .to_device(&loaded.device)?
            .to_dtype(loaded.dtype)?;
            img_shapes.push((1, encoded.dim(2)? / 2, encoded.dim(3)? / 2));
            packed_input_storage.push(Self::pack_latents_4d(&encoded)?);
        }
        self.base.progress.phase_done(
            crate::ProgressPhase::Vae,
            "Encoding edit images (VAE)",
            encode_start.elapsed(),
        );

        let packed_inputs = if packed_input_storage.is_empty() {
            None
        } else {
            let tensors = packed_input_storage.iter().collect::<Vec<_>>();
            Some(Tensor::cat(&tensors, 1)?)
        };

        self.base
            .progress
            .stage_start("Reloading Qwen-Image transformer after edit conditioning");
        let reload_start = Instant::now();
        self.reload_transformer(loaded, width, height)?;
        self.base.progress.stage_done(
            "Reloading Qwen-Image transformer after edit conditioning",
            reload_start.elapsed(),
        );

        let noise = crate::engine::seeded_randn(
            seed,
            &[1, 16, height / 8, width / 8],
            &loaded.device,
            loaded.dtype,
        )?;
        let mut scheduler = QwenImageScheduler::new(
            req.steps as usize,
            (height / 16) * (width / 16),
            shift_policy,
        );
        let num_steps = scheduler.num_steps();
        let mut latents = Self::pack_latents_4d(&(noise * scheduler.initial_sigma())?)?;
        let output_seq_len = latents.dim(1)?;

        let denoise_label = format!("Denoising edit ({} steps)", num_steps);
        self.base.progress.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");
            let use_batched_cfg = use_cfg && transformer.supports_cfg_batching();
            let (batched_hs, batched_mask) = if use_batched_cfg {
                let (cond_hs, neg_hs, mask) = align_cfg_conditioning(
                    &encoder_hidden_states,
                    uncond_hs.as_ref().expect("unconditional prompt missing"),
                )?;
                (Tensor::cat(&[&cond_hs, &neg_hs], 0)?, mask)
            } else {
                (encoder_hidden_states.clone(), None)
            };

            for step in 0..num_steps {
                self.base.progress.checkpoint()?;
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
                            batched_mask.as_ref(),
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
                                    None,
                                    &img_shapes,
                                )?
                                .narrow(1, 0, output_seq_len)?,
                            transformer
                                .forward_packed(
                                    &latent_model_input,
                                    &timestep,
                                    uncond_hs.as_ref().unwrap(),
                                    None,
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
                            None,
                            &img_shapes,
                        )?
                        .narrow(1, 0, output_seq_len)?
                };

                latents = scheduler.step(&noise_pred, &latents)?;
                self.base.progress.emit(ProgressEvent::DenoiseStep {
                    step: step + 1,
                    total: num_steps,
                    elapsed: step_start.elapsed(),
                });
            }
        }

        self.base.progress.checkpoint()?;
        self.base
            .progress
            .stage_done(&denoise_label, denoise_start.elapsed());

        // Decode is the final independent Qwen VAE phase. Admission prices
        // its workspace as a max with (not an addition to) transformer
        // residency, so release the transformer before allocating decode
        // buffers. The next edit would release this resident copy before
        // conditioning anyway.
        self.base
            .progress
            .info("Releasing Qwen-Image transformer before VAE decode");
        self.release_edit_transformer(&mut loaded.transformer);
        loaded.device.synchronize()?;

        let latents = Self::unpack_latents_packed(&latents, height / 8, width / 8)?;
        let free_for_decode = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let prefer_tiled = Self::should_proactively_tile_vae_decode(
            req.width,
            req.height,
            loaded.vae_device.is_cuda(),
            free_for_decode,
        );
        let image = Self::decode_vae_with_fallback(
            &latents,
            &loaded.vae,
            &loaded.vae_device,
            &loaded.device,
            &self.base.progress,
            prefer_tiled,
            || {
                Ok(QwenImageVae::load(
                    &loaded.vae_path,
                    &Device::Cpu,
                    DType::F32,
                    &self.base.progress,
                )?)
            },
        )?;
        let image = postprocess_image(&image)?.i(0)?;
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        Ok(GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: image_bytes,
                format: req.resolved_output_format(),
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }
}

impl QwenImageEngine {
    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
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
            return self.finish_edit_generation(result);
        }

        // Sequential mode: load-use-drop each component
        if self.base.load_strategy == LoadStrategy::Sequential {
            return self.generate_sequential(req);
        }

        // Eager mode: use pre-loaded components
        if self.base.loaded.is_none() {
            bail!("model not loaded -- call load() first");
        }

        // Started before the reload below so `generation_time_ms` keeps
        // covering the transformer rebuild and its LoRA merge, exactly as
        // FLUX's does. That cost is the whole subject of the fingerprint
        // elision — a timing that excluded it could not show the win.
        let start = Instant::now();

        // Reload the transformer when it was dropped after a previous VAE
        // decode, or when this request's LoRA stack differs from the one
        // merged into the resident transformer. An unchanged stack keeps
        // the merged weights — for GGUF that is the whole dequantize →
        // merge → re-quantize pass over every LoRA-touched tensor.
        self.ensure_transformer_for_request(req.width as usize, req.height as usize)?;

        let progress = &self.base.progress;
        let gpu_ordinal = self.base.gpu_ordinal;

        // The checkpoint's own packaged scheduler config, not the family's.
        // Read before `loaded` takes the mutable borrow of `self.base`.
        let shift_policy = shift_policy_for_model(&self.base.model_name);

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

        let (encoder_hidden_states, uncond_hs) = if both_cached {
            let cached = prompt_cached.expect("prompt cache unexpectedly missing");
            progress.cache_hit("prompt conditioning");
            let hs = cached.restore(&loaded.device, loaded.dtype)?;
            let u_hs = if use_cfg {
                progress.cache_hit("unconditional conditioning");
                let ucached =
                    uncond_cached.expect("unconditional prompt cache unexpectedly missing");
                Some(ucached.restore(&loaded.device, loaded.dtype)?)
            } else {
                None
            };
            (hs, u_hs)
        } else {
            if loaded.text_encoder.model.is_none() {
                let label = if loaded.text_encoder.is_parked() {
                    "Unparking Qwen2.5 encoder (CPU→GPU)"
                } else {
                    "Reloading Qwen2.5 encoder"
                };
                progress.stage_start(label);
                let reload_start = Instant::now();
                if loaded.text_encoder.is_parked() {
                    loaded.text_encoder.unpark_to_gpu(progress)?;
                } else {
                    loaded.text_encoder.reload(progress)?;
                }
                progress.stage_done(label, reload_start.elapsed());
            }

            let hs = Self::encode_prompt_cached(
                progress,
                &self.prompt_cache,
                &mut loaded.text_encoder,
                &req.prompt,
                &loaded.device,
                loaded.dtype,
            )?;

            let u_hs = if use_cfg {
                Some(Self::encode_prompt_cached(
                    progress,
                    &self.prompt_cache,
                    &mut loaded.text_encoder,
                    QWEN_EMPTY_NEGATIVE_PROMPT,
                    &loaded.device,
                    loaded.dtype,
                )?)
            } else {
                None
            };

            (hs, u_hs)
        };

        // Both streams keep their true length; only the batched-CFG branch in
        // the denoise loop pads, and only when the lengths differ.

        // Drop or park text encoder to free VRAM for denoising.
        if loaded.text_encoder.on_gpu {
            let free_after_encode = usable_free_vram_bytes(gpu_ordinal).unwrap_or(0);
            let required_for_residency = Self::qwen2_hot_text_encoder_required_vram(
                req.width,
                req.height,
                if req.guidance > 1.0 { 2 } else { 1 },
                loaded.dtype,
            );
            let action =
                Self::qwen2_text_encoder_post_encode_action(Qwen2TextEncoderResidencyInput {
                    on_gpu: loaded.text_encoder.on_gpu,
                    is_metal: loaded.device.is_metal(),
                    keep_te_ram: crate::device::keep_te_in_ram(),
                    prompt_cache_miss: !both_cached,
                    transformer_resident: loaded.transformer.is_some(),
                    free_vram_bytes: free_after_encode,
                    required_vram_bytes: required_for_residency,
                });
            match action {
                Qwen2TextEncoderPostEncodeAction::KeepGpu => {
                    progress.info(&format!(
                        "Keeping Qwen2.5 text encoder on GPU for hot prompt-cache misses ({} free >= {} reserve)",
                        fmt_gb(free_after_encode),
                        fmt_gb(required_for_residency)
                    ));
                    tracing::info!(
                        free_vram_bytes = free_after_encode,
                        required_vram_bytes = required_for_residency,
                        is_quantized = loaded.text_encoder.is_quantized,
                        "Qwen2.5 text encoder kept on GPU after cache miss"
                    );
                }
                Qwen2TextEncoderPostEncodeAction::ParkCpu => {
                    loaded.text_encoder.park_to_cpu()?;
                    progress.info(&format!(
                        "Parked Qwen2.5 text encoder to CPU host RAM before denoise ({} free < {} reserve)",
                        fmt_gb(free_after_encode),
                        fmt_gb(required_for_residency)
                    ));
                    tracing::info!("Qwen2.5 text encoder parked to CPU host RAM");
                }
                Qwen2TextEncoderPostEncodeAction::Drop => {
                    loaded.text_encoder.drop_weights();
                    progress.info(&format!(
                        "Dropped Qwen2.5 text encoder before denoise ({} free < {} reserve or cache hit)",
                        fmt_gb(free_after_encode),
                        fmt_gb(required_for_residency)
                    ));
                    tracing::info!("Qwen2.5 text encoder dropped from GPU");
                }
            }
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
                    shift_policy,
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
            QwenImageScheduler::new_img2img(
                req.steps as usize,
                image_seq_len,
                req.strength,
                shift_policy,
            )
        } else {
            let sched = QwenImageScheduler::new(req.steps as usize, image_seq_len, shift_policy);
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
            // the extra batch dimension without exceeding peak memory. Only this
            // branch pads, and only when the two prompts differ in length.
            let (batched_hs, batched_mask) = if use_batched_cfg {
                let (cond_hs, neg_hs, mask) = align_cfg_conditioning(
                    &encoder_hidden_states,
                    uncond_hs.as_ref().expect("unconditional prompt missing"),
                )?;
                (Tensor::cat(&[&cond_hs, &neg_hs], 0)?, mask)
            } else {
                (encoder_hidden_states.clone(), None)
            };

            for step in 0..num_steps {
                progress.checkpoint()?;
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
                            batched_mask.as_ref(),
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
                                None,
                            )?,
                            transformer.forward(
                                &latents,
                                &t_tensor,
                                uncond_hs.as_ref().unwrap(),
                                None,
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
                    transformer.forward(&latents, &t_tensor, &encoder_hidden_states, None)?
                };
                if step == 0 || step == num_steps / 2 || step == num_steps - 1 {
                    Self::debug_tensor_stats(&format!("noise_pred[{step}]"), &noise_pred);
                    Self::debug_tensor_stats(&format!("latents[{step}]"), &latents);
                }
                if step == 0 {
                    Self::validate_qwen_tensor_boundary("noise_pred[0]", &noise_pred)?;
                }
                latents = scheduler.step(&noise_pred, &latents)?;
                if step == num_steps - 1 {
                    Self::validate_qwen_tensor_boundary("latents_final", &latents)?;
                }

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

        progress.checkpoint()?;
        progress.stage_done(&denoise_label, denoise_start.elapsed());

        // Free text embeddings
        drop(encoder_hidden_states);
        drop(uncond_hs);

        // 8. VAE decode
        progress.stage_start("VAE decode");
        let vae_start = Instant::now();
        let free_for_decode = usable_free_vram_bytes(self.base.gpu_ordinal).unwrap_or(0);
        let prefer_tiled = Self::should_proactively_tile_vae_decode(
            req.width,
            req.height,
            loaded.vae_device.is_cuda(),
            free_for_decode,
        );

        // Always decode in F32 — matches sequential path and diffusers' force_upcast.
        let keep_transformer_hot = Self::can_keep_transformer_hot_for_vae(loaded);
        let image = if keep_transformer_hot {
            match Self::decode_vae_gpu_only(
                &latents,
                &loaded.vae,
                &loaded.vae_device,
                &loaded.device,
                progress,
                prefer_tiled,
            ) {
                Ok(image) => {
                    progress.info(
                        "Kept quantized Qwen transformer resident across VAE decode for faster hot-path reuse",
                    );
                    image
                }
                Err(err) if Self::is_oom_error(&err) => {
                    // No resident transformer means no baked LoRA stack.
                    Self::release_resident_transformer(
                        &mut loaded.transformer,
                        &mut self.active_lora_fingerprint,
                    );
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
                        prefer_tiled,
                        || {
                            QwenImageVae::load(&loaded.vae_path, &Device::Cpu, DType::F32, progress)
                                .map_err(Into::into)
                        },
                    )?
                }
                Err(err) => return Err(err),
            }
        } else {
            // No resident transformer means no baked LoRA stack.
            Self::release_resident_transformer(
                &mut loaded.transformer,
                &mut self.active_lora_fingerprint,
            );
            loaded.device.synchronize()?;
            tracing::info!("Qwen-Image transformer dropped to free VRAM for VAE decode");
            Self::decode_vae_with_fallback(
                &latents,
                &loaded.vae,
                &loaded.vae_device,
                &loaded.device,
                progress,
                prefer_tiled,
                || {
                    QwenImageVae::load(&loaded.vae_path, &Device::Cpu, DType::F32, progress)
                        .map_err(Into::into)
                },
            )?
        };
        Self::validate_qwen_tensor_boundary("image_pre_postprocess", &image)?;
        Self::debug_tensor_stats("image_pre_postprocess", &image);
        let image = postprocess_image(&image)?;
        let post_stats = Self::validate_qwen_tensor_boundary("image_postprocess", &image)?;
        Self::debug_tensor_stats("image_postprocess", &image);
        let image = image.i(0)?;
        if Self::near_black_image_stats(post_stats) {
            progress.info(
                "Qwen diagnostic: decoded image is near-black after VAE postprocess; inspect MOLD_QWEN_DEBUG tensor stats to separate denoise math from VAE decode",
            );
            tracing::warn!(
                min = post_stats.min,
                max = post_stats.max,
                mean = post_stats.mean,
                "Qwen decoded image is near-black after VAE postprocess"
            );
        }

        progress.phase_done(crate::ProgressPhase::Vae, "VAE decode", vae_start.elapsed());

        // 9. Encode to output format
        let output_metadata = build_output_metadata(req, seed, None);
        let image_bytes = encode_image(
            &image,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Qwen-Image generation complete");

        Ok(GenerateResponse {
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data: image_bytes,
                format: req.resolved_output_format(),
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }
}

impl InferenceEngine for QwenImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_loras(req);
        let result = self.generate_inner(req);
        self.pending_placement = None;
        self.pending_loras.clear();
        result
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

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        self.pending_loras = effective_loras(req);
        let result = QwenImageEngine::load(self);
        self.pending_placement = None;
        self.pending_loras.clear();
        result
    }

    fn unload(&mut self) {
        // A parked encoder is several GB of host RAM held by this engine.
        // Unload means "give the resources back", so the retention opt-in does
        // not survive it — an explicit unload, or the model cache evicting
        // this engine, releases it.
        self.retained_sequential_text_encoder = None;
        self.base.unload();
        clear_cache(&self.prompt_cache);
        // The fingerprint describes the transformer that just went away;
        // the next load re-applies whatever the request carries.
        self.active_lora_fingerprint.clear();
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn set_cancellation_token(&mut self, token: crate::progress::InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family("qwen-image")
            .expect("production Qwen-Image batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&mold_core::ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LoadStrategy;
    use crate::shared_pool::SharedPool;
    use candle_core::Shape;
    use mold_core::ModelPaths;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokenizers::models::bpe::BPE;

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
            low_noise_transformer: None,
            low_noise_distilled_lora: None,
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
    fn cached_prompt_conditioning_roundtrips_unpadded_conditioning() {
        let device = Device::Cpu;
        let hidden_states = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let cached = CachedPromptConditioning::from_parts(&hidden_states, 3).unwrap();

        let restored_hs = cached.restore(&device, DType::F32).unwrap();

        assert_eq!(
            tensor_values_f32(&restored_hs),
            tensor_values_f32(&hidden_states)
        );
    }

    /// `Qwen2TextEncoder` narrows to the true token count before returning, so
    /// a cached entry whose `valid_len` is short means an encoder started
    /// padding — which would feed zero rows into the transformer unmasked.
    #[test]
    fn cached_prompt_conditioning_rejects_padded_conditioning() {
        let device = Device::Cpu;
        let hidden_states = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let err = CachedPromptConditioning::from_parts(&hidden_states, 2)
            .err()
            .expect("padded conditioning must be rejected");
        assert!(err.to_string().contains("must arrive unpadded"));
    }

    #[test]
    fn pad_text_conditioning_keeps_original_when_target_matches() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();

        let padded_hs = pad_text_conditioning(&hidden_states, 2).unwrap();

        assert_eq!(
            tensor_values_f32(&padded_hs),
            tensor_values_f32(&hidden_states)
        );
    }

    #[test]
    fn pad_text_conditioning_appends_zero_padding() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();

        let padded_hs = pad_text_conditioning(&hidden_states, 4).unwrap();

        assert_eq!(padded_hs.dims3().unwrap(), (1, 4, 2));
        assert_eq!(
            tensor_values_f32(&padded_hs),
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn pad_text_conditioning_rejects_shrinking() {
        let device = Device::Cpu;
        let hidden_states =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();

        let err = pad_text_conditioning(&hidden_states, 1).unwrap_err();
        assert!(err.to_string().contains("cannot shrink text conditioning"));
    }

    /// Equal lengths mean no padding and therefore no mask, which is the state
    /// the split-CFG, no-CFG and edit paths are always in now that nothing pads
    /// ahead of the batching decision.
    #[test]
    fn align_cfg_conditioning_returns_no_mask_for_equal_lengths() {
        let device = Device::Cpu;
        let cond_hs =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], Shape::from((1, 2, 2)), &device).unwrap();
        let uncond_hs =
            Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], Shape::from((1, 2, 2)), &device).unwrap();

        let (cond, uncond, mask) = align_cfg_conditioning(&cond_hs, &uncond_hs).unwrap();

        assert!(
            mask.is_none(),
            "two equal-length streams need no mask (diffusers' `prompt_embeds_mask.all()` case)"
        );
        assert_eq!(tensor_values_f32(&cond), tensor_values_f32(&cond_hs));
        assert_eq!(tensor_values_f32(&uncond), tensor_values_f32(&uncond_hs));
    }

    #[test]
    fn align_cfg_conditioning_pads_shorter_branch_and_masks_the_padding() {
        let device = Device::Cpu;
        let cond_hs = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let uncond_hs = Tensor::from_vec(
            vec![7.0f32, 8.0, 9.0, 10.0],
            Shape::from((1, 2, 2)),
            &device,
        )
        .unwrap();

        let (cond, uncond, mask) = align_cfg_conditioning(&cond_hs, &uncond_hs).unwrap();

        assert_eq!(cond.dims3().unwrap(), (1, 3, 2));
        assert_eq!(uncond.dims3().unwrap(), (1, 3, 2));
        assert_eq!(
            tensor_values_f32(&uncond),
            vec![7.0, 8.0, 9.0, 10.0, 0.0, 0.0]
        );
        // The mask covers the batched `[cond; uncond]` pair, one row each.
        let mask = mask.expect("differing lengths must carry a mask");
        assert_eq!(mask.dims2().unwrap(), (2, 3));
        assert_eq!(tensor_values_u8(&mask), vec![1, 1, 1, 1, 1, 0]);
    }

    /// The batched-CFG bias rows must pair with the `cat([cond, uncond], 0)`
    /// batch order all the way through attention: row 0 masks nothing when
    /// cond is the longer stream, and row 1 masks exactly uncond's padding.
    /// A transposed pairing would pass every single-batch test in this crate
    /// while masking the wrong stream on every batched render.
    #[test]
    fn batched_cfg_bias_rows_pair_with_the_cond_uncond_cat_order() {
        let device = Device::Cpu;
        let cond_hs = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from((1, 3, 2)),
            &device,
        )
        .unwrap();
        let uncond_hs = Tensor::from_vec(
            vec![7.0f32, 8.0, 9.0, 10.0],
            Shape::from((1, 2, 2)),
            &device,
        )
        .unwrap();
        let (_, _, mask) = align_cfg_conditioning(&cond_hs, &uncond_hs).unwrap();
        let mask = mask.expect("differing lengths must carry a mask");

        // Joint bias over [text(3), image(2)] keys for the batch-2 forward.
        let img_seq_len = 2;
        let bias = crate::qwen_image::attention::joint_key_bias(
            Some(&mask),
            img_seq_len,
            DType::F32,
            &device,
        )
        .unwrap()
        .expect("padded text must produce a bias");
        assert_eq!(bias.dims(), &[2, 1, 1, 5]);

        // Per-batch K/V whose padded slot (text index 2 of the uncond row)
        // holds a poison value that would dominate attention if unmasked.
        let head_dim = 4;
        let total = 5;
        let mut k_data = Vec::new();
        let mut v_data = Vec::new();
        for batch in 0..2 {
            for key in 0..total {
                for d in 0..head_dim {
                    let poison = batch == 1 && key == 2;
                    k_data.push(if poison {
                        50.0
                    } else {
                        (batch * 100 + key * 10 + d) as f32 * 0.01
                    });
                    v_data.push((batch * 1000 + key * 100 + d) as f32 * 0.001);
                }
            }
        }
        let k = Tensor::from_vec(k_data, Shape::from((2, 1, total, head_dim)), &device).unwrap();
        let v = Tensor::from_vec(v_data, Shape::from((2, 1, total, head_dim)), &device).unwrap();
        let q = Tensor::rand(-1.0f32, 1.0, Shape::from((2, 1, total, head_dim)), &device).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let got = crate::attention::attention_with_bias(&q, &k, &v, scale, Some(&bias)).unwrap();

        // Row 0 (cond): nothing masked — plain attention over all 5 keys.
        let q0 = q.narrow(0, 0, 1).unwrap();
        let want0 = crate::attention::attention(
            &q0,
            &k.narrow(0, 0, 1).unwrap(),
            &v.narrow(0, 0, 1).unwrap(),
            scale,
        )
        .unwrap();
        // Row 1 (uncond): the pad key (text index 2) must be invisible —
        // reference drops it entirely.
        let keep = [0usize, 1, 3, 4];
        let q1 = q.narrow(0, 1, 1).unwrap();
        let k1 = Tensor::cat(
            &keep
                .iter()
                .map(|&i| k.narrow(0, 1, 1).unwrap().narrow(2, i, 1).unwrap())
                .collect::<Vec<_>>(),
            2,
        )
        .unwrap();
        let v1 = Tensor::cat(
            &keep
                .iter()
                .map(|&i| v.narrow(0, 1, 1).unwrap().narrow(2, i, 1).unwrap())
                .collect::<Vec<_>>(),
            2,
        )
        .unwrap();
        let want1 = crate::attention::attention(&q1, &k1, &v1, scale).unwrap();

        let diff0 = (got.narrow(0, 0, 1).unwrap() - &want0)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff1 = (got.narrow(0, 1, 1).unwrap() - &want1)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff0 < 1e-5, "cond row diverged from maskless: {diff0}");
        assert!(diff1 < 1e-5, "uncond row saw its pad key: {diff1}");
    }

    #[test]
    fn qwen_image_detects_gguf_transformer() {
        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            ModelPaths {
                low_noise_transformer: None,
                low_noise_distilled_lora: None,
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
            0,
            false,
            None,
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
    fn qwen_image_auto_prefers_quantized_gpu_on_cuda_for_sequential_mode_when_it_fits() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            QWEN2_FP16_VRAM_THRESHOLD - 1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Sequential,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert!(resolved.auto_use_gpu);
    }

    #[test]
    fn qwen_image_auto_uses_quantized_cpu_fallback_on_cuda_for_sequential_mode() {
        let resolved = QwenImageEngine::choose_text_encoder_source(
            Some("auto"),
            true,
            false,
            1,
            16_600_000_000,
            Qwen2TextEncoderUsage::Sequential,
        )
        .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
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
        let quantized_text_encoder = touch(&dir, "text-encoder-q4.gguf");
        let engine = QwenImageEngine::new(
            "qwen-image-edit-2511:q4".to_string(),
            paths,
            LoadStrategy::Sequential,
            0,
            false,
            None,
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
            .resolve_text_encoder_source_with_preference_using(
                &Device::Cpu,
                0,
                Qwen2TextEncoderUsage::Sequential,
                Some("q4"),
                |_, variant| {
                    assert_eq!(variant.tag, "q4");
                    Ok(quantized_text_encoder.clone())
                },
            )
            .unwrap();
        assert!(resolved.is_gguf);
        assert_eq!(resolved.variant_label, "q4");
        assert_eq!(resolved.paths, [quantized_text_encoder]);
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
    fn qwen_image_edit_dims_enforce_ceiling_when_one_axis_hits_the_minimum() {
        for (width, height) in [(16, 200_000), (200_000, 16)] {
            let (fitted_width, fitted_height) =
                QwenImageEngine::qwen_image_edit_dims(width, height, QWEN_IMAGE_EDIT_VAE_AREA);
            assert!(fitted_width * fitted_height <= QWEN_IMAGE_EDIT_VAE_AREA);
            assert_eq!(fitted_width % 16, 0);
            assert_eq!(fitted_height % 16, 0);
        }
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

    /// The transformer geometry the byte arithmetic restates as `u64` must
    /// stay the checkpoint's own.
    #[test]
    fn qwen_dit_working_set_constants_match_the_transformer_config() {
        let cfg = QwenImageConfig::qwen_image_2512();
        assert_eq!(QWEN_DIT_INNER_DIM, cfg.inner_dim as u64);
        assert_eq!(QWEN_DIT_HEADS, cfg.num_attention_heads as u64);
        assert_eq!(
            QWEN_DIT_PIXELS_PER_TOKEN_AXIS,
            (vae::VAE_SPATIAL_COMPRESSION * cfg.patch_size) as u64
        );
    }

    /// Both shapes are pinned exactly: the estimate decides whether a q4
    /// checkpoint batches CFG at all, so a silent drift in any buffer count is
    /// a silent change to that decision.
    #[test]
    fn quantized_cuda_cfg_headroom_is_pinned_at_the_two_common_shapes() {
        // 1328²: 6889 image + 512 text = 7401 joint tokens, chunked 512 rows.
        assert_eq!(
            QwenImageEngine::quantized_cuda_cfg_headroom_for_chunk(7401, Some(512)),
            4_591_779_840
        );
        // 1024²: 4096 + 512 = 4608 joint tokens. The derivation lands at
        // 2.94 GB, under the floor, so the floor is what ships.
        assert_eq!(
            QwenImageEngine::quantized_cuda_cfg_headroom_for_chunk(4608, Some(512)),
            QWEN_GGUF_MIN_CFG_HEADROOM
        );
        // The resolution-derived path agrees with the token-derived one.
        assert_eq!(
            QwenImageEngine::quantized_cuda_cfg_headroom(1024, 1024),
            QWEN_GGUF_MIN_CFG_HEADROOM
        );
        assert_eq!(QwenImageEngine::qwen_dit_joint_tokens(1328, 1328), 7401);
        assert_eq!(QwenImageEngine::qwen_dit_joint_tokens(1024, 1024), 4608);
    }

    /// The query chunk is the whole reason the estimate moved, and the
    /// unchunked figure is pinned exactly.
    ///
    /// It is deliberately NOT asserted to reproduce the retired 14 GB
    /// constant: it does not (17.7 GB), and it is not the same quantity — a
    /// phase becomes a headroom only after the 1.5x margin, which puts the
    /// comparable number at 26.6 GB. A range assertion wide enough to make the
    /// old claim look true is how the misstatement survived, so this pins the
    /// value.
    #[test]
    fn the_query_chunk_is_what_moved_the_attention_estimate() {
        let chunked = QwenImageEngine::qwen_dit_attention_phase_bytes(7401, Some(512));
        let unchunked = QwenImageEngine::qwen_dit_attention_phase_bytes(7401, None);

        assert_eq!(chunked, 3_061_186_560);
        assert_eq!(unchunked, 17_745_007_392);
        assert!(
            chunked < unchunked / 5,
            "chunking must be the reason the estimate moved: {chunked} vs {unchunked}"
        );
        // The comparable headroom is the phase plus the margin, and it is
        // nowhere near the constant this replaced.
        assert_eq!(
            QwenImageEngine::quantized_cuda_cfg_headroom_for_chunk(7401, None),
            26_617_511_088
        );
    }

    /// Native 1328² splits CFG by the MEASURED token cap even with the whole
    /// card free: batched CFG ran 24% slower there (164.9 s vs 133.2 s on the
    /// 4090), so free VRAM is not the deciding input past 4096 image tokens.
    #[test]
    fn qwen_quantized_native_resolution_splits_cfg_by_the_measured_token_cap() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            12_300_000_000,
            24_600_000_000,
            1328,
            1328,
        ));
    }

    /// 1024² (4096 image tokens, measured split-parity) batches when it fits.
    #[test]
    fn qwen_quantized_1024_batches_cfg_on_24gb_cuda() {
        assert!(!QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            12_300_000_000,
            24_600_000_000,
            1024,
            1024,
        ));
    }

    /// A 16 GB card still cannot, which is what keeps the decision meaningful.
    #[test]
    fn qwen_quantized_native_resolution_still_splits_cfg_on_16gb_cuda() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            12_300_000_000,
            16_000_000_000,
            1328,
            1328,
        ));
    }

    #[test]
    fn qwen_quantized_edit_always_uses_split_cfg_on_high_vram_cuda() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            true,
            13_200_000_000,
            47_000_000_000,
            800,
            1312,
        ));
    }

    #[test]
    fn qwen_quantized_reduced_resolution_keeps_batched_cfg_when_it_fits() {
        assert!(!QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            12_300_000_000,
            24_600_000_000,
            512,
            512,
        ));
    }

    #[test]
    fn qwen_quantized_cfg_split_boundary_does_not_split_when_estimate_exactly_fits() {
        // At 1024² — inside the measured token cap — the memory boundary is
        // what decides, and an exactly-fitting estimate batches.
        let headroom = QwenImageEngine::quantized_cuda_cfg_headroom(1024, 1024);
        let transformer_size = 12_300_000_000;
        let free_vram = transformer_size + headroom;
        assert!(!QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            transformer_size,
            free_vram,
            1024,
            1024,
        ));
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            transformer_size,
            free_vram - 1,
            1024,
            1024,
        ));
    }

    #[test]
    fn qwen_quantized_unknown_vram_biases_to_split_cfg() {
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
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
    fn qwen_debug_stats_counts_nan_and_inf() {
        let _guard = FULL_STATS_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tensor = Tensor::from_vec(
            vec![0.0f32, 1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            Shape::from((5,)),
            &Device::Cpu,
        )
        .unwrap();

        let stats = QwenImageEngine::tensor_stats(&tensor).unwrap();

        assert_eq!(stats.total, 5);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.pos_inf_count, 1);
        assert_eq!(stats.neg_inf_count, 1);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 1.0);
        assert_eq!(stats.mean, 0.5);
    }

    /// The two tests below read a process-global counter, so they must not
    /// interleave with each other or with anything else that calls
    /// `tensor_stats`.
    static FULL_STATS_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn full_stats_downloads() -> usize {
        FULL_TENSOR_STATS_DOWNLOADS.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn qwen_boundary_probe_reports_the_same_numbers_as_the_full_scan() {
        let _guard = FULL_STATS_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tensor = Tensor::from_vec(
            vec![-1.0f32, 0.0, 1.0, 2.0],
            Shape::from((4,)),
            &Device::Cpu,
        )
        .unwrap();

        let probe = QwenImageEngine::tensor_finiteness_probe(&tensor).unwrap();
        let stats = QwenImageEngine::tensor_stats(&tensor).unwrap();

        assert!(probe.is_clean());
        assert_eq!(probe.total, stats.total);
        assert_eq!(probe.min, stats.min);
        assert_eq!(probe.max, stats.max);
        assert_eq!(probe.mean, stats.mean);
        assert_eq!(probe.nan_count, stats.nan_count);
    }

    #[test]
    fn qwen_boundary_probe_sees_nan_and_infinities() {
        let device = Device::Cpu;
        let nan = Tensor::from_vec(vec![0.0f32, f32::NAN], Shape::from((2,)), &device).unwrap();
        let pos_inf =
            Tensor::from_vec(vec![0.0f32, f32::INFINITY], Shape::from((2,)), &device).unwrap();
        let neg_inf =
            Tensor::from_vec(vec![0.0f32, f32::NEG_INFINITY], Shape::from((2,)), &device).unwrap();

        let nan_probe = QwenImageEngine::tensor_finiteness_probe(&nan).unwrap();
        assert_eq!(nan_probe.nan_count, 1);
        assert!(!nan_probe.is_clean());
        assert!(!QwenImageEngine::tensor_finiteness_probe(&pos_inf)
            .unwrap()
            .is_clean());
        assert!(!QwenImageEngine::tensor_finiteness_probe(&neg_inf)
            .unwrap()
            .is_clean());
    }

    #[test]
    fn qwen_boundary_needs_the_full_scan_only_when_asked_or_tripped() {
        let clean = QwenFinitenessProbe {
            nan_count: 0,
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            total: 4,
        };
        let tripped = QwenFinitenessProbe {
            nan_count: 3,
            ..clean
        };

        assert!(!QwenImageEngine::boundary_needs_full_stats(clean, false));
        assert!(QwenImageEngine::boundary_needs_full_stats(clean, true));
        assert!(QwenImageEngine::boundary_needs_full_stats(tripped, false));
    }

    /// The clean boundary must never take the full CPU-side scan. Asserted on
    /// the download counter rather than on timing.
    #[test]
    fn qwen_clean_boundary_never_downloads_the_whole_tensor() {
        let _guard = FULL_STATS_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tensor =
            Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], Shape::from((4,)), &Device::Cpu).unwrap();

        let before = full_stats_downloads();
        let stats = QwenImageEngine::validate_qwen_tensor_boundary("clean", &tensor).unwrap();

        assert_eq!(
            full_stats_downloads(),
            before,
            "a finite boundary must be settled by the on-device probe alone"
        );
        assert_eq!(stats.total, 4);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.min, 0.1);
    }

    /// The fail-loud half is unchanged: an injected NaN still aborts, and the
    /// message still carries the full breakdown, which costs the scan.
    #[test]
    fn qwen_injected_nan_boundary_still_errors_with_the_full_breakdown() {
        let _guard = FULL_STATS_COUNTER_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tensor = Tensor::from_vec(
            vec![0.0f32, f32::NAN, 1.0, f32::INFINITY],
            Shape::from((4,)),
            &Device::Cpu,
        )
        .unwrap();

        let before = full_stats_downloads();
        let err = QwenImageEngine::validate_qwen_tensor_boundary("noise_pred[0]", &tensor)
            .expect_err("a NaN boundary must abort the render");

        assert!(full_stats_downloads() > before);
        let message = err.to_string();
        assert!(message.contains("noise_pred[0]"), "{message}");
        assert!(message.contains("NaN=1/4"), "{message}");
        assert!(message.contains("+Inf=1"), "{message}");
    }

    #[test]
    fn qwen_debug_stats_detects_near_black_postprocessed_image() {
        let stats = QwenTensorStats {
            min: 0.0,
            max: 0.01,
            mean: 0.004,
            nan_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            total: 1024,
        };

        assert!(QwenImageEngine::near_black_image_stats(stats));
    }

    #[test]
    fn qwen_debug_stats_does_not_flag_non_black_image() {
        let stats = QwenTensorStats {
            min: 0.0,
            max: 0.75,
            mean: 0.18,
            nan_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            total: 1024,
        };

        assert!(!QwenImageEngine::near_black_image_stats(stats));
    }

    #[test]
    fn qwen_debug_stats_formats_progress_message() {
        let stats = QwenTensorStats {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            nan_count: 2,
            pos_inf_count: 1,
            neg_inf_count: 1,
            total: 10,
        };

        let message = QwenImageEngine::format_tensor_stats("sample", stats);

        assert!(message.contains("NaN=2/10"));
        assert!(message.contains("+Inf=1"));
        assert!(message.contains("-Inf=1"));
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
            false,
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
            false,
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
            false,
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
            false,
            &Device::Cpu,
            &progress,
            "tiled",
            "cpu",
            QwenImageEngine::is_oom_error,
        )
        .unwrap_err();

        assert!(err.to_string().contains("bad tiled decode"));
    }

    /// Each decode phase is pinned separately, so a drift names the term that
    /// moved rather than only the total. The numbers are per-pixel constants:
    /// 320 B for the mid block, 8_640 B for the full-resolution upsampler,
    /// 7_296 B for the final up-block.
    #[test]
    fn qwen_vae_decode_workspace_phases_match_their_allocation_shapes() {
        // 1024²: 1_048_576 px, 16_384 latent tokens.
        //   scores    2 * 1_024 * 16_384 * 4 =   134_217_728
        //   operands  8 *   384 * 16_384 * 4 =   201_326_592
        assert_eq!(
            QwenImageEngine::qwen_vae_mid_block_phase_bytes(1_048_576),
            335_544_320
        );
        //   input       192 *   262_144 * 4 =   201_326_592
        //   upsampled   192 * 1_048_576 * 4 =   805_306_368
        //   im2col  192 * 9 * 1_048_576 * 4 = 7_247_757_312
        //   result + its transpose
        //           2 *  96 * 1_048_576 * 4 =   805_306_368
        assert_eq!(
            QwenImageEngine::qwen_vae_full_res_upsample_phase_bytes(1_048_576),
            9_059_696_640
        );
        //   residual chain 8 * 96 * 1_048_576 * 4 = 3_221_225_472
        //   im2col      96 * 9 * 1_048_576 * 4    = 3_623_878_656
        //   result + transpose 2 * 96 * 1_048_576 * 4 = 805_306_368
        assert_eq!(
            QwenImageEngine::qwen_vae_final_up_block_phase_bytes(1_048_576),
            7_650_410_496
        );

        // The phases are sequential, so the reserve is the largest of them —
        // the full-resolution upsampler, whose im2col column buffer is the
        // biggest single allocation the decode makes.
        assert_eq!(
            QwenImageEngine::qwen_vae_decode_workspace_bytes(1024, 1024),
            9_059_696_640
        );
        // 1328² (native): 1_763_584 px → 8_640 B/px.
        assert_eq!(
            QwenImageEngine::qwen_vae_decode_workspace_bytes(1328, 1328),
            15_237_365_760
        );

        // Every term is linear in pixel count, so the reserve is too.
        assert_eq!(
            QwenImageEngine::qwen_vae_decode_workspace_bytes(2048, 1024),
            2 * QwenImageEngine::qwen_vae_decode_workspace_bytes(1024, 1024)
        );
    }

    /// The workspace arithmetic reads the VAE's own channel plan rather than
    /// restating it. If `BLOCK_OUT_CHANNELS` ever changes, this is the
    /// assertion that says so in the language of the architecture, and the
    /// byte totals above are what say which term drifted.
    #[test]
    fn qwen_vae_decode_workspace_derives_from_the_vae_channel_plan() {
        assert_eq!(vae::VAE_MID_BLOCK_CHANNELS, 384);
        assert_eq!(vae::VAE_FULL_RES_UPSAMPLE_IN_CHANNELS, 192);
        assert_eq!(vae::VAE_FINAL_BLOCK_CHANNELS, 96);
        assert_eq!(vae::VAE_SPATIAL_COMPRESSION, 8);
        assert_eq!(vae::VAE_CONV_KERNEL_ELEMS, 9);
    }

    /// Dimensions are `u32`, so `pixels` reaches `(2^32 - 1)²` and every factor
    /// on top of it overflows `u64`. Upstream validation stops requests long
    /// before that, but the reserve is a gate: it saturates to "more VRAM than
    /// exists" — which tiles — rather than panicking in a debug build or
    /// wrapping to a tiny reserve in a release one.
    #[test]
    fn qwen_vae_decode_workspace_saturates_instead_of_overflowing() {
        assert_eq!(
            QwenImageEngine::qwen_vae_decode_workspace_bytes(u32::MAX, u32::MAX),
            u64::MAX
        );
        assert!(QwenImageEngine::should_proactively_tile_vae_decode(
            u32::MAX,
            u32::MAX,
            true,
            u64::MAX - 1
        ));
    }

    /// What chunking the mid-block attention bought: the unchunked score matrix
    /// was `N x N`, materialised twice, which at native 1328² is 6.1 GB — on its
    /// own more than a third of the whole reserve. The chunked phase must stay
    /// an order of magnitude below it.
    #[test]
    fn qwen_vae_mid_block_phase_is_far_below_the_unchunked_score_matrix() {
        let pixels = 1328u64 * 1328;
        let latent_tokens = pixels / 64;
        let unchunked_scores = 2 * latent_tokens * latent_tokens * 4;
        let now = QwenImageEngine::qwen_vae_mid_block_phase_bytes(pixels);

        assert_eq!(unchunked_scores, 6_074_665_088);
        assert!(
            now * 10 < unchunked_scores,
            "chunked mid block {now} must stay far below {unchunked_scores}"
        );
    }

    /// The gate reads the workspace reserve, so it moved with it: modelling the
    /// im2col column buffer the decoder's full-resolution convolutions actually
    /// allocate raised native 1328² from ~9.7 GB free to ~17.7 GB.
    #[test]
    fn qwen_proactive_tiling_engages_at_the_rederived_reserve() {
        let required = VAE_DECODE_VRAM_THRESHOLD
            + QwenImageEngine::qwen_vae_decode_workspace_bytes(1328, 1328);
        assert_eq!(required, 17_737_365_760);
        assert!(QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            true,
            required - 1
        ));
        assert!(!QwenImageEngine::should_proactively_tile_vae_decode(
            1328, 1328, true, required
        ));
        // 12.2 GB of that is one column buffer, so a card holding 9 GB free
        // cannot run the full decode — it tiles up front instead of paying for
        // the failed allocation first.
        assert!(QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            true,
            9_000_000_000
        ));
    }

    #[test]
    fn qwen_proactive_tiled_policy_selects_native_cuda_under_pressure() {
        assert!(QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            true,
            6_000_000_000
        ));
        assert!(!QwenImageEngine::should_proactively_tile_vae_decode(
            512,
            512,
            true,
            6_000_000_000
        ));
        assert!(!QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            false,
            6_000_000_000
        ));
        // A card with room for the full decode still takes it. 16 GB used to
        // be on this side of the gate, but the reserve now models the im2col
        // column buffer the decoder's full-resolution convolutions allocate —
        // 12.2 GB of the 15.2 GB total at this size — so it no longer is.
        assert!(QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            true,
            16_000_000_000
        ));
        assert!(!QwenImageEngine::should_proactively_tile_vae_decode(
            1328,
            1328,
            true,
            20_000_000_000
        ));
    }

    #[test]
    fn qwen_proactive_tiled_decode_skips_primary_full_decode() {
        let mut progress = ProgressReporter::default();
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let messages_clone = messages.clone();
        progress.set_callback(Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                messages_clone.lock().unwrap().push(message);
            }
        }));

        let primary_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let primary_called_clone = primary_called.clone();
        let value = QwenImageEngine::with_cuda_tiled_then_cpu_fallback(
            || {
                primary_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(3usize)
            },
            || Ok(7usize),
            || Ok(9usize),
            true,
            true,
            &Device::Cpu,
            &progress,
            "tiled after oom",
            "cpu",
            QwenImageEngine::is_oom_error,
        )
        .unwrap();

        assert_eq!(value, 7);
        assert!(!primary_called.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            messages.lock().unwrap().as_slice(),
            ["Selecting tiled GPU VAE decode proactively"]
        );
    }

    #[test]
    fn qwen_hot_text_encoder_keeps_gpu_after_cache_miss_with_headroom() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: false,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 10_000_000_000,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::KeepGpu);
    }

    #[test]
    fn qwen_hot_text_encoder_drops_after_cache_hit_even_with_headroom() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: false,
                prompt_cache_miss: false,
                transformer_resident: true,
                free_vram_bytes: 10_000_000_000,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::Drop);
    }

    #[test]
    fn qwen_hot_text_encoder_drops_under_transformer_pressure() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: false,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 7_999_999_999,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::Drop);
    }

    #[test]
    fn qwen_hot_text_encoder_parks_bf16_when_keep_ram_enabled() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: true,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 7_999_999_999,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::ParkCpu);
    }

    /// The GGUF encoder parks by the same rule as BF16 (#1044): its
    /// `QTensor` bytes move host↔device losslessly, so there is no longer a
    /// quantized exclusion here. The 35.1 s disk reload was the whole reason
    /// a quantized encoder used to fall through to `Drop`.
    #[test]
    fn qwen_hot_text_encoder_parks_quantized_when_keep_ram_enabled() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: true,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 7_999_999_999,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::ParkCpu);
    }

    /// Metal is unified memory: parking to "host RAM" frees nothing and only
    /// costs a copy in each direction.
    #[test]
    fn qwen_hot_text_encoder_never_parks_on_metal() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: true,
                keep_te_ram: true,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 7_999_999_999,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::Drop);
    }

    /// The retained sequential encoder is several gigabytes of host RAM held
    /// by the engine, so `unload()` — what the model-cache LRU calls when it
    /// evicts this engine — has to release it. Mirrors `WanEngine::unload`.
    #[test]
    fn qwen_unload_releases_the_retained_sequential_text_encoder() {
        let mut engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                PathBuf::from("/nonexistent/transformer.gguf"),
                vec![],
                PathBuf::from("/nonexistent/vae.safetensors"),
                Some(PathBuf::from("/nonexistent/tokenizer.json")),
            ),
            LoadStrategy::Sequential,
            0,
            false,
            None,
        );

        engine.retained_sequential_text_encoder = Some(
            encoders::qwen2_text::Qwen2TextEncoder::prepare_gguf_with_tokenizer(
                Path::new("/nonexistent/qwen2.gguf"),
                &PathBuf::from("/nonexistent/tokenizer.json"),
                Some(Arc::new(Tokenizer::new(
                    tokenizers::models::wordpiece::WordPiece::default(),
                ))),
                &Device::Cpu,
                DType::F32,
                &[],
            )
            .unwrap(),
        );

        engine.unload();

        assert!(
            engine.retained_sequential_text_encoder.is_none(),
            "unload must give the parked encoder's host RAM back"
        );
    }

    /// The edit path's release rule. `keep_te_ram` alone is not enough:
    /// parking a CPU-resident encoder retains host RAM the drop used to
    /// release, and parking right before `unload()` pays a multi-gigabyte
    /// device→host copy for a map that is discarded microseconds later.
    #[test]
    fn qwen_edit_text_encoder_parks_only_when_the_park_can_pay_off() {
        let resident_gpu_edit = Qwen2EditTextEncoderReleaseInput {
            on_gpu: true,
            is_metal: false,
            keep_te_ram: true,
            engine_unloads_after: false,
        };
        assert!(
            QwenImageEngine::qwen2_edit_text_encoder_should_park(resident_gpu_edit),
            "an opted-in GPU encoder that survives the request parks"
        );

        assert!(
            !QwenImageEngine::qwen2_edit_text_encoder_should_park(
                Qwen2EditTextEncoderReleaseInput {
                    keep_te_ram: false,
                    ..resident_gpu_edit
                }
            ),
            "parking stays opt-in"
        );
        assert!(
            !QwenImageEngine::qwen2_edit_text_encoder_should_park(
                Qwen2EditTextEncoderReleaseInput {
                    on_gpu: false,
                    ..resident_gpu_edit
                }
            ),
            "a CPU-placed encoder must be dropped, not retained in host RAM"
        );
        assert!(
            !QwenImageEngine::qwen2_edit_text_encoder_should_park(
                Qwen2EditTextEncoderReleaseInput {
                    is_metal: true,
                    ..resident_gpu_edit
                }
            ),
            "unified memory makes the park pointless"
        );
        assert!(
            !QwenImageEngine::qwen2_edit_text_encoder_should_park(
                Qwen2EditTextEncoderReleaseInput {
                    engine_unloads_after: true,
                    ..resident_gpu_edit
                }
            ),
            "a park the engine unloads immediately afterwards is pure cost"
        );
    }

    /// Without the knob, the encoder still drops — parking is a host-RAM
    /// trade, so it stays gated on `keep_te_ram`.
    #[test]
    fn qwen_hot_text_encoder_drops_quantized_without_keep_ram() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: false,
                prompt_cache_miss: true,
                transformer_resident: true,
                free_vram_bytes: 7_999_999_999,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::Drop);
    }

    #[test]
    fn qwen_hot_text_encoder_drops_when_transformer_not_resident() {
        let action = QwenImageEngine::qwen2_text_encoder_post_encode_action(
            Qwen2TextEncoderResidencyInput {
                on_gpu: true,
                is_metal: false,
                keep_te_ram: false,
                prompt_cache_miss: true,
                transformer_resident: false,
                free_vram_bytes: 10_000_000_000,
                required_vram_bytes: 8_000_000_000,
            },
        );

        assert_eq!(action, Qwen2TextEncoderPostEncodeAction::Drop);
    }

    #[test]
    fn qwen_transformer_hot_vae_eligibility_requires_quantized_cuda_components() {
        assert!(QwenImageEngine::qwen_transformer_can_stay_hot_for_vae(
            true, true, true
        ));
        assert!(!QwenImageEngine::qwen_transformer_can_stay_hot_for_vae(
            false, true, true
        ));
        assert!(!QwenImageEngine::qwen_transformer_can_stay_hot_for_vae(
            true, false, true
        ));
        assert!(!QwenImageEngine::qwen_transformer_can_stay_hot_for_vae(
            true, true, false
        ));
    }

    fn lora(path: &str, scale: f64) -> mold_core::LoraWeight {
        mold_core::LoraWeight {
            path: path.to_string(),
            scale,
            expert: None,
        }
    }

    /// Build an engine with no real weights behind it. Every path exercised
    /// through it here stops before touching the filesystem.
    fn fingerprint_test_engine(strategy: LoadStrategy) -> QwenImageEngine {
        QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                PathBuf::from("/nonexistent/transformer.gguf"),
                vec![],
                PathBuf::from("/nonexistent/vae.safetensors"),
                Some(PathBuf::from("/nonexistent/tokenizer.json")),
            ),
            strategy,
            0,
            false,
            None,
        )
    }

    /// Minimal stand-in for the resident-transformer half of the engine,
    /// used only to drive the *predicate* over a sequence of requests. It
    /// deliberately does not stand in for the engine's own bookkeeping —
    /// the write and clear sites that maintain `active_lora_fingerprint`
    /// are pinned separately, on the real engine, by the four tests below
    /// `qwen_transformer_build_records_the_pending_lora_stack`.
    struct RebuildCounter {
        resident: bool,
        baked: Vec<QwenImageLoraFingerprint>,
        builds: usize,
    }

    impl RebuildCounter {
        fn new() -> Self {
            Self {
                resident: false,
                baked: Vec::new(),
                builds: 0,
            }
        }

        /// One request: rebuild only when the engine would.
        fn request(&mut self, loras: &[mold_core::LoraWeight]) {
            let requested = fingerprint_stack(loras);
            if QwenImageEngine::qwen_transformer_rebuild_needed(
                self.resident,
                &self.baked,
                &requested,
            ) {
                self.builds += 1;
                self.baked = requested;
                self.resident = true;
            }
        }

        /// The VAE decode that drops the transformer (non-stay-hot path).
        fn drop_transformer(&mut self) {
            self.resident = false;
        }
    }

    #[test]
    fn qwen_lora_fingerprint_distinguishes_path_scale_and_order() {
        let a = lora("/loras/lightning-8.safetensors", 1.0);
        let b = lora("/loras/style.safetensors", 0.8);

        assert_eq!(
            fingerprint_stack(std::slice::from_ref(&a)),
            fingerprint_stack(std::slice::from_ref(&a))
        );
        assert_ne!(
            fingerprint_stack(std::slice::from_ref(&a)),
            fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 0.9)])
        );
        assert_ne!(
            fingerprint_stack(std::slice::from_ref(&a)),
            fingerprint_stack(std::slice::from_ref(&b))
        );
        assert_ne!(
            fingerprint_stack(&[a.clone(), b.clone()]),
            fingerprint_stack(&[b.clone(), a.clone()])
        );
        assert_ne!(
            fingerprint_stack(std::slice::from_ref(&a)),
            fingerprint_stack(&[a.clone(), b])
        );
        assert!(fingerprint_stack(&[]).is_empty());
    }

    #[test]
    fn qwen_transformer_rebuild_needed_only_when_stack_changes_or_transformer_is_gone() {
        let stack = fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);
        let rescaled = fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 0.9)]);
        let empty: Vec<QwenImageLoraFingerprint> = Vec::new();

        assert!(!QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &stack, &stack
        ));
        assert!(!QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &empty, &empty
        ));
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &stack, &rescaled
        ));
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &stack, &empty
        ));
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &empty, &stack
        ));
        // No resident transformer always rebuilds, stack notwithstanding.
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            false, &stack, &stack
        ));
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            false, &empty, &empty
        ));
    }

    #[test]
    fn qwen_cfg_batching_moves_only_when_the_flag_contradicts_this_request() {
        // Currently batched (`supports = true`), request wants batched too.
        assert_eq!(
            QwenImageEngine::qwen_cfg_batching_update(Some(true), false),
            None
        );
        // Currently split (`supports = false`), request wants split too.
        assert_eq!(
            QwenImageEngine::qwen_cfg_batching_update(Some(false), true),
            None
        );
        // Loaded split at 1328², but this smaller request batches — the whole
        // point of re-deciding per request.
        assert_eq!(
            QwenImageEngine::qwen_cfg_batching_update(Some(false), false),
            Some(true)
        );
        // Currently batched, but this request no longer fits batched.
        assert_eq!(
            QwenImageEngine::qwen_cfg_batching_update(Some(true), true),
            Some(false)
        );
        // BF16 / FP8 / offloaded transformers carry no flag.
        assert_eq!(QwenImageEngine::qwen_cfg_batching_update(None, true), None);
        assert_eq!(QwenImageEngine::qwen_cfg_batching_update(None, false), None);
    }

    /// The CFG-batching mode is a `bool` with no structural effect, so it must
    /// never reach the rebuild decision: a rebuild is a full GGUF dequantize →
    /// merge → re-quantize across every block, and because the stale check and
    /// the rebuild's own decision read free VRAM from systematically unequal
    /// sources, making it a rebuild input let one resolution rebuild on every
    /// single request, forever.
    #[test]
    fn cfg_batching_mode_never_triggers_a_qwen_transformer_rebuild() {
        let stack = fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);
        let empty: Vec<QwenImageLoraFingerprint> = Vec::new();

        // Same stack, resident transformer — elides regardless of any CFG
        // re-decision, which is handled in place.
        assert!(!QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &stack, &stack
        ));
        assert!(!QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &empty, &empty
        ));
        // The LoRA rebuild is untouched.
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            true, &stack, &empty
        ));
    }

    /// The same request, taken repeatedly against a free-VRAM reading that
    /// wobbles around the split/batched boundary, must cost zero rebuilds.
    #[test]
    fn oscillating_cfg_decisions_cost_no_qwen_transformer_rebuilds() {
        let stack = [lora("/loras/lightning-8.safetensors", 1.0)];
        let mut engine = RebuildCounter::new();
        engine.request(&stack);
        assert_eq!(engine.builds, 1, "the first load always builds");

        let mut supports = Some(false);
        for request_splits in [false, true, false, true, false] {
            if let Some(wanted) =
                QwenImageEngine::qwen_cfg_batching_update(supports, request_splits)
            {
                supports = Some(wanted);
            }
            engine.request(&stack);
        }

        assert_eq!(
            engine.builds, 1,
            "flipping the CFG mode must never rebuild the transformer"
        );
        assert_eq!(supports, Some(true), "the last request batched");
    }

    #[test]
    fn resident_quantized_transformer_re_decides_cfg_at_the_request_resolution() {
        // A 12.3 GB q4 checkpoint loaded at 1328² on a 16 GB card chose split
        // CFG. The load path read free VRAM before those weights landed, so a
        // resident engine has to add them back to ask the same question.
        let transformer_size = 12_300_000_000u64;
        let card = 16_000_000_000u64;
        let free_while_resident = card - transformer_size;

        let loaded_supports_batching = !QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            transformer_size,
            card,
            1328,
            1328,
        );
        assert!(
            !loaded_supports_batching,
            "1328² loads with split CFG on 16 GB"
        );

        // The same engine now serves 768². Read free VRAM while the
        // transformer is resident, then add its bytes back.
        let free_for_decision = free_while_resident + transformer_size;
        let request_splits = QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            transformer_size,
            free_for_decision,
            768,
            768,
        );
        assert!(!request_splits, "768² fits batched CFG on 16 GB");

        assert_eq!(
            QwenImageEngine::qwen_cfg_batching_update(
                Some(loaded_supports_batching),
                request_splits
            ),
            Some(true),
            "the split flag must move to batched for a request that fits batched"
        );

        // Forgetting the add-back is the bug this guards: the resident
        // reading alone makes even 768² look like it must split.
        assert!(QwenImageEngine::should_split_cfg_quantized_cuda(
            false,
            transformer_size,
            free_while_resident,
            768,
            768,
        ));
    }

    #[test]
    fn repeated_qwen_lora_stack_builds_the_transformer_once_on_the_hot_path() {
        let stack = [lora("/loras/lightning-8.safetensors", 1.0)];
        let mut engine = RebuildCounter::new();

        engine.request(&stack);
        engine.request(&stack);
        engine.request(&stack);

        assert_eq!(
            engine.builds, 1,
            "an unchanged LoRA stack must reuse the merged transformer"
        );
    }

    #[test]
    fn changed_qwen_lora_stack_rebuilds_the_transformer() {
        let a = [lora("/loras/lightning-8.safetensors", 1.0)];
        let rescaled = [lora("/loras/lightning-8.safetensors", 0.7)];
        let mut engine = RebuildCounter::new();

        engine.request(&a);
        engine.request(&rescaled);
        assert_eq!(engine.builds, 2, "a scale change must invalidate the merge");

        engine.request(&rescaled);
        assert_eq!(engine.builds, 2);

        let reordered = [
            lora("/loras/style.safetensors", 0.5),
            lora("/loras/lightning-8.safetensors", 1.0),
        ];
        let original_order = [
            lora("/loras/lightning-8.safetensors", 1.0),
            lora("/loras/style.safetensors", 0.5),
        ];
        engine.request(&original_order);
        assert_eq!(engine.builds, 3);
        engine.request(&reordered);
        assert_eq!(engine.builds, 4, "reordering must invalidate the merge");
    }

    #[test]
    fn qwen_empty_lora_stack_transitions_rebuild_in_both_directions() {
        let stack = [lora("/loras/lightning-8.safetensors", 1.0)];
        let mut engine = RebuildCounter::new();

        engine.request(&[]);
        assert_eq!(engine.builds, 1, "the first load always builds");
        engine.request(&[]);
        assert_eq!(engine.builds, 1, "no LoRA twice must not rebuild");
        engine.request(&stack);
        assert_eq!(engine.builds, 2, "adding a LoRA must rebuild");
        engine.request(&[]);
        assert_eq!(engine.builds, 3, "removing the LoRA must rebuild");
    }

    #[test]
    fn dropping_the_qwen_transformer_forces_a_rebuild_even_for_an_unchanged_stack() {
        let stack = [lora("/loras/lightning-8.safetensors", 1.0)];
        let mut engine = RebuildCounter::new();

        engine.request(&stack);
        // The non-stay-hot VAE decode drops the transformer, so the merge
        // cost returns on the next request; only the resident path elides.
        engine.drop_transformer();
        engine.request(&stack);

        assert_eq!(engine.builds, 2);
    }

    /// The write half of the contract, on the real engine. Both build sites
    /// (eager `load` and `reload_transformer`) call this, and it must derive
    /// the fingerprint from `pending_loras` — the slice `load_transformer`
    /// actually merges. Deriving it from anything else is the original
    /// defect's shape: a field that does not describe the resident weights.
    #[test]
    fn qwen_transformer_build_records_the_pending_lora_stack() {
        let mut engine = fingerprint_test_engine(LoadStrategy::Eager);
        let stack = vec![
            lora("/loras/lightning-8.safetensors", 1.0),
            lora("/loras/style.safetensors", 0.4),
        ];

        engine.pending_loras = stack.clone();
        engine.note_transformer_lora_stack();

        assert_eq!(
            engine.active_lora_fingerprint,
            fingerprint_stack(&stack),
            "a built transformer must record the stack merged into it"
        );
        assert!(!QwenImageEngine::qwen_transformer_rebuild_needed(
            true,
            &engine.active_lora_fingerprint,
            &fingerprint_stack(&stack),
        ));

        // An unmerged rebuild must leave nothing behind to elide against.
        engine.pending_loras.clear();
        engine.note_transformer_lora_stack();
        assert!(engine.active_lora_fingerprint.is_empty());
    }

    /// The clear half, on the code every drop site routes through. Dropping
    /// the transformer without clearing the fingerprint is exactly the state
    /// that made the stay-hot path render a new stack with the old merge, so
    /// the two mutations are pinned as one.
    #[test]
    fn releasing_the_qwen_transformer_clears_the_baked_fingerprint() {
        let mut transformer = Some(());
        let mut baked = fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);
        assert!(!baked.is_empty());

        QwenImageEngine::release_resident_transformer(&mut transformer, &mut baked);

        assert!(transformer.is_none(), "the transformer must be dropped");
        assert!(
            baked.is_empty(),
            "a dropped transformer must leave no stack to elide against"
        );
        // And the predicate now insists on a rebuild for that same stack.
        assert!(QwenImageEngine::qwen_transformer_rebuild_needed(
            false,
            &baked,
            &fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]),
        ));
    }

    #[test]
    fn qwen_edit_phase_transitions_install_pending_loras_and_release_them() {
        let stack = vec![
            lora("/loras/lightning-8.safetensors", 1.0),
            lora("/loras/style.safetensors", 0.4),
        ];
        let mut engine = fingerprint_test_engine(LoadStrategy::Eager);
        engine.pending_loras = stack.clone();

        // A resident transformer from the previous request cannot overlap
        // multimodal conditioning. Releasing it must also revoke its baked
        // stack authority.
        let mut transformer = Some(());
        engine.note_transformer_lora_stack();
        engine.release_edit_transformer(&mut transformer);
        assert!(transformer.is_none());
        assert!(engine.active_lora_fingerprint.is_empty());

        // `reload_transformer` routes the real weights through this install
        // transition, so the post-conditioning resident transformer records
        // the exact pending stack the loader merged.
        engine.install_reloaded_transformer(&mut transformer, ());
        assert!(transformer.is_some());
        assert_eq!(engine.active_lora_fingerprint, fingerprint_stack(&stack));

        // Final VAE decode is another independent phase and must revoke both
        // pieces of residency authority again.
        engine.release_edit_transformer(&mut transformer);
        assert!(transformer.is_none());
        assert!(engine.active_lora_fingerprint.is_empty());
    }

    #[test]
    fn qwen_edit_finish_keeps_eager_bundle_but_unloads_sequential_state() {
        let stack = fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);

        let mut eager = fingerprint_test_engine(LoadStrategy::Eager);
        eager.active_lora_fingerprint = stack.clone();
        eager.finish_edit_generation(Ok(())).unwrap();
        assert_eq!(eager.active_lora_fingerprint, stack);

        let mut sequential = fingerprint_test_engine(LoadStrategy::Sequential);
        sequential.active_lora_fingerprint = stack;
        sequential.finish_edit_generation(Ok(())).unwrap();
        assert!(sequential.active_lora_fingerprint.is_empty());
    }

    /// `unload()` is what the model-cache LRU calls when it evicts this
    /// engine. The transformer goes with it, so the fingerprint must too —
    /// otherwise a reloaded engine elides the merge its weights never got.
    #[test]
    fn qwen_unload_clears_the_baked_lora_fingerprint() {
        let mut engine = fingerprint_test_engine(LoadStrategy::Eager);
        engine.active_lora_fingerprint =
            fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);

        InferenceEngine::unload(&mut engine);

        assert!(
            engine.active_lora_fingerprint.is_empty(),
            "unload must forget the stack baked into the transformer it released"
        );
    }

    /// Sequential loading builds a request-local transformer inside
    /// `generate_sequential` and drops it again, so `load()` leaves nothing
    /// resident. A fingerprint surviving that would claim otherwise.
    #[test]
    fn qwen_sequential_load_leaves_no_baked_lora_fingerprint() {
        let mut engine = fingerprint_test_engine(LoadStrategy::Sequential);
        engine.active_lora_fingerprint =
            fingerprint_stack(&[lora("/loras/lightning-8.safetensors", 1.0)]);

        QwenImageEngine::load(&mut engine).expect("sequential load defers and cannot fail here");

        assert!(
            engine.active_lora_fingerprint.is_empty(),
            "sequential loading keeps no resident transformer to fingerprint"
        );
    }

    /// With nothing loaded there is no transformer to reason about, so the
    /// request must fail rather than record a stack for weights that do not
    /// exist.
    #[test]
    fn qwen_ensure_transformer_without_a_loaded_engine_is_an_error() {
        let mut engine = fingerprint_test_engine(LoadStrategy::Eager);
        engine.pending_loras = vec![lora("/loras/lightning-8.safetensors", 1.0)];

        assert!(engine.ensure_transformer_for_request(1024, 1024).is_err());
        assert!(
            engine.active_lora_fingerprint.is_empty(),
            "a failed request must not claim a stack is baked in"
        );
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
            0,
            false,
            None,
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
            0,
            false,
            None,
        );
        assert_eq!(sharded.validate_paths().unwrap(), tokenizer);
        assert!(!sharded.detect_is_quantized());

        let quantized = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(gguf, vec![], vae, Some(dir.join("tokenizer.json"))),
            LoadStrategy::Sequential,
            0,
            false,
            None,
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
            0,
            false,
            None,
        );

        let err = engine.validate_paths().unwrap_err();
        assert!(err.to_string().contains("text tokenizer path required"));

        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_image_loads_text_tokenizer_through_shared_pool() {
        let dir = temp_test_dir("mold-qwen-tokenizer-pool");
        let tokenizer_path = dir.join("tokenizer.json");
        tokenizers::Tokenizer::new(BPE::default())
            .save(&tokenizer_path, false)
            .unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_tokenizer(&tokenizer_path)
            .unwrap();

        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                dir.join("vae.safetensors"),
                Some(tokenizer_path.clone()),
            ),
            LoadStrategy::Sequential,
            0,
            false,
            Some(shared_pool),
        );

        let loaded = engine.load_text_tokenizer(&tokenizer_path).unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_image_loads_vae_tensors_through_shared_pool() {
        let dir = temp_test_dir("mold-qwen-vae-pool");
        let vae_path = dir.join("vae.safetensors");
        let weight = 1.0f32.to_le_bytes();
        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder.conv_in.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![1], &weight).unwrap(),
        );
        serialize_to_file(&tensors, &None, &vae_path).unwrap();

        let shared_pool = Arc::new(Mutex::new(SharedPool::new()));
        let pooled = shared_pool
            .lock()
            .unwrap()
            .load_safetensors_cpu_tensors(std::slice::from_ref(&vae_path))
            .unwrap()
            .unwrap();

        let engine = QwenImageEngine::new(
            "qwen-image:q4".to_string(),
            qwen_image_model_paths(
                dir.join("transformer.gguf"),
                vec![],
                vae_path.clone(),
                Some(dir.join("tokenizer.json")),
            ),
            LoadStrategy::Sequential,
            0,
            false,
            Some(shared_pool),
        );

        let loaded = engine.load_vae_cpu_tensors().unwrap().unwrap();

        assert!(Arc::ptr_eq(&pooled, &loaded));
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
