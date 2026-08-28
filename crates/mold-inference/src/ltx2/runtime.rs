#![allow(clippy::too_many_arguments)]

use crate::audio::NativeAudioTrack;

use anyhow::{bail, Context, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use image::{imageops, GenericImage, Rgb, RgbImage};
use mold_candle::ltx_video::sampling::{
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerConfig, TimeShiftType,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::backend::compute_dtype;
use super::conditioning::retake_temporal_mask;
use super::execution::SamplerMode;
use super::guidance::{
    BatchedPerturbationConfig, MultiModalGuider, MultiModalGuiderParams, Perturbation,
    PerturbationConfig, PerturbationType,
};
use super::lora;
use super::media;
use super::model::{
    audio_temporal_positions,
    audio_transformer::Ltx2AudioTransformerModel,
    cross_modal_temporal_positions, derive_stage1_render_shape, get_pixel_coords,
    scale_video_time_to_seconds, spatially_upsample_frames, temporally_upsample_frames_x2,
    video_token_positions,
    video_transformer::{
        Ltx2AvTransformer3DModel, Ltx2VideoTransformer3DModelConfig, LtxPreparedModalityStatic,
        LtxPreparedStaticInputs,
    },
    video_vae::{AutoencoderKLLtx2Video, AutoencoderKLLtx2VideoConfig},
    AudioLatentShape, AudioPatchifier, DecodedAudio, Ltx2AudioDecoder, Ltx2AudioEncoder,
    Ltx2VocoderWithBwe, SpatioTemporalScaleFactors, VideoLatentPatchifier, VideoLatentShape,
    VideoPixelShape,
};
use super::plan::{Ltx2GeneratePlan, PipelineKind};
use super::preprocess;
use super::sampler::sampler_step;
use super::text::connectors::EmbeddingsProcessorOutput;
use super::text::prompt_encoder::{NativePromptEncoder, NativePromptEncoding};
use super::tiling::{
    create_tiles, plan_spatial_decode_tiling, plan_stage2_tiling_with_policy, SpatialTilePolicy,
    Tile, TRAINED_SPATIAL_LATENT_SPAN,
};
use crate::adaptive_offload::{
    plan_adaptive_residency, AdaptiveResidencyPlan, ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
};
use crate::device::{
    dtype_bytes, fmt_gb, free_vram_bytes, ltx2_activation_budget_bytes, thread_gpu_ordinal,
    try_synchronize_device, usable_free_vram_bytes, PhaseVramProbe, PhaseVramReport,
};
use crate::engine::seeded_randn;
use crate::ltx_video::latent_upsampler::LatentUpsampler;
use crate::progress::{InferenceCancellationToken, ProgressCallback, ProgressEvent, ProgressPhase};
use crate::vae_tiling::is_out_of_memory_error;
use crate::weight_loader::{
    load_fp8_safetensors_with_callback, load_safetensors_with_progress_callback,
};
use mold_core::ltx2_weight_index::{
    Ltx2ResidentWeightForm, Ltx2TransformerWeightIndex, Ltx2WeightFormat,
};
use mold_core::{LoraWeight, Ltx2GuidanceOverrides, Ltx2SpatialUpscale, TimeRange};

pub const LTX2_VIDEO_LATENT_CHANNELS: usize = 128;
pub const LTX2_AUDIO_LATENT_CHANNELS: usize = 8;
pub const LTX2_AUDIO_MEL_BINS: usize = 16;
pub const LTX2_AUDIO_SAMPLE_RATE: usize = 16_000;
pub const LTX2_AUDIO_HOP_LENGTH: usize = 160;
pub const LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR: usize = 4;

#[derive(Debug)]
pub struct NativePreparedRun {
    pub prompt: NativePromptEncoding,
    pub debug_alt_prompt: Option<EmbeddingsProcessorOutput>,
    pub video_pixel_shape: VideoPixelShape,
    pub video_latent_shape: VideoLatentShape,
    pub audio_latent_shape: Option<AudioLatentShape>,
    pub video_positions: Tensor,
    pub audio_positions: Option<Tensor>,
    #[allow(dead_code)]
    pub cross_modal_temporal_positions: Option<(Tensor, Tensor)>,
    pub retake_mask: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct NativeRenderedVideo {
    pub frames: Vec<RgbImage>,
    /// Scene-referred linear HDR frames, present only when the plan asked for
    /// an EXR sidecar. Kept alongside the 8-bit frames rather than replacing
    /// them: the gallery artifact is still the tonemapped video.
    /// Number of EXR frames written as a sidecar during decode, if any.
    pub hdr_frames_written: Option<usize>,
    pub audio_track: Option<NativeAudioTrack>,
    pub has_audio: bool,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
}

#[derive(Debug)]
struct NativeConditioningAudio {
    latents: Tensor,
    original_track: Option<NativeAudioTrack>,
}

#[derive(Debug)]
struct NativeConditioningVideo {
    latents: Tensor,
}

#[derive(Debug, Clone)]
struct VideoTokenReplacement {
    start_token: usize,
    tokens: Tensor,
    strength: f64,
}

#[derive(Debug, Clone)]
struct VideoTokenAppendCondition {
    tokens: Tensor,
    positions: Tensor,
    strength: f64,
    /// The `(frames, height, width)` latent grid these flattened tokens came
    /// from. Only tiled stage-2 refinement needs it — a tile has to slice this
    /// condition to its own spatial region, and a flat token run cannot say
    /// which of its entries belong to which column.
    latent_grid: (usize, usize, usize),
    /// How much smaller this condition's spatial grid is than the generated
    /// one. IC-LoRA reference video is encoded at `1/df` resolution, so a tile
    /// covering generated cells `[a, b)` covers reference cells `[a/df, b/df)`
    /// (`hdr_ic_lora.py:531-537`).
    spatial_downscale_factor: usize,
}

#[derive(Debug, Clone, Default)]
struct StageVideoConditioning {
    replacements: Vec<VideoTokenReplacement>,
    appended: Vec<VideoTokenAppendCondition>,
}

impl StageVideoConditioning {
    fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.appended.is_empty()
    }
}

/// Audio latents appended to the denoised sequence as a *reference* — tokens
/// the transformer may attend to but never denoises.
///
/// The exact mirror of [`VideoTokenAppendCondition`] on the audio branch,
/// which lip dub is the first pipeline to need: upstream builds an
/// `AudioConditionByReferenceLatent` from the reference clip's own speech
/// (`lipdub.py:228-239`).
#[derive(Debug, Clone)]
struct AudioTokenAppendCondition {
    tokens: Tensor,
    positions: Tensor,
    strength: f64,
}

#[derive(Debug, Clone, Default)]
struct StageAudioConditioning {
    appended: Vec<AudioTokenAppendCondition>,
}

impl StageAudioConditioning {
    fn is_empty(&self) -> bool {
        self.appended.is_empty()
    }

    fn appended_token_count(&self) -> Result<usize> {
        let mut total = 0;
        for condition in &self.appended {
            total += condition.tokens.dim(1)?;
        }
        Ok(total)
    }
}

/// One audio latent frame, in seconds: `hop_length / sample_rate` per mel
/// frame, times the VAE's temporal downsample factor. 0.04 s at 16 kHz.
const LTX2_AUDIO_LATENT_FRAME_SECONDS: f32 = (LTX2_AUDIO_HOP_LENGTH
    * LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR) as f32
    / LTX2_AUDIO_SAMPLE_RATE as f32;

/// RoPE positions for reference audio, shifted so the whole reference sits
/// strictly *before* the clip being generated.
///
/// This is the load-bearing detail of lip dub. The reference tokens carry the
/// same seconds-valued positions the generated audio would, so without a shift
/// the model sees two overlapping soundtracks on one timeline. Upstream
/// subtracts the reference's own duration plus exactly one audio latent frame
/// (`lipdub.py:297-316`): `positions -= positions[..., -1, 1].max() + 0.04`.
/// Every position ends up negative and the last reference patch ends at
/// exactly `-0.04`, one frame before the generated audio's `0.0`.
///
/// Getting this wrong produces output that looks plausible and is out of sync,
/// so `lip_dub_audio_reference_positions_end_one_latent_frame_before_zero`
/// asserts the boundary exactly rather than within a tolerance.
fn shift_audio_reference_positions_before_zero(positions: &Tensor) -> Result<Tensor> {
    // `positions` is `(batch, 1, tokens, 2)` holding `[start, end]` seconds.
    let last_end = positions
        .i((.., .., positions.dim(2)? - 1, 1))?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);
    if !last_end.is_finite() {
        anyhow::bail!("audio reference positions contained no finite end timestamp");
    }
    positions
        .to_dtype(DType::F32)?
        .affine(1.0, -((last_end + LTX2_AUDIO_LATENT_FRAME_SECONDS) as f64))
        .map_err(Into::into)
}

/// Patchify an audio VAE latent into reference tokens with negatively shifted
/// RoPE positions. The audio analogue of
/// [`append_condition_from_video_latents`].
fn append_condition_from_audio_latents(
    latents: &Tensor,
    strength: f64,
) -> Result<AudioTokenAppendCondition> {
    let patchifier = AudioPatchifier::new(
        LTX2_AUDIO_SAMPLE_RATE,
        LTX2_AUDIO_HOP_LENGTH,
        LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        true,
        0,
    );
    let latents = latents.to_dtype(DType::F32)?;
    let (batch, channels, frames, mel_bins) = latents.dims4()?;
    let tokens = patchifier.patchify(&latents)?;
    let shape = AudioLatentShape {
        batch,
        channels,
        frames,
        mel_bins,
    };
    let positions = shift_audio_reference_positions_before_zero(
        &patchifier.get_patch_grid_bounds(shape, latents.device())?,
    )?;
    Ok(AudioTokenAppendCondition {
        tokens,
        positions,
        strength,
    })
}

fn apply_appended_audio_conditioning(
    audio_latents: &Tensor,
    audio_positions: &Tensor,
    conditioning: &StageAudioConditioning,
) -> Result<(Tensor, Tensor)> {
    if conditioning.appended.is_empty() {
        return Ok((audio_latents.clone(), audio_positions.clone()));
    }
    let mut token_parts = vec![audio_latents.clone()];
    let mut position_parts = vec![audio_positions.clone()];
    for condition in &conditioning.appended {
        token_parts.push(
            condition
                .tokens
                .to_device(audio_latents.device())?
                .to_dtype(audio_latents.dtype())?,
        );
        position_parts.push(
            condition
                .positions
                .to_device(audio_positions.device())?
                .to_dtype(audio_positions.dtype())?,
        );
    }
    let token_refs = token_parts.iter().collect::<Vec<_>>();
    let position_refs = position_parts.iter().collect::<Vec<_>>();
    Ok((
        Tensor::cat(&token_refs, 1)?,
        Tensor::cat(&position_refs, 2)?,
    ))
}

/// Re-seat the reference tokens after a sampler step so they never drift.
fn reapply_stage_audio_conditioning(
    audio_latents: &Tensor,
    base_token_count: usize,
    conditioning: &StageAudioConditioning,
) -> Result<Tensor> {
    if conditioning.appended.is_empty() {
        return Ok(audio_latents.clone());
    }
    let total_tokens = audio_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "audio token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }
    let mut parts = vec![audio_latents.narrow(1, 0, base_token_count)?];
    for condition in &conditioning.appended {
        parts.push(
            condition
                .tokens
                .to_device(audio_latents.device())?
                .to_dtype(audio_latents.dtype())?,
        );
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 1).map_err(Into::into)
}

fn strip_appended_audio_conditioning(
    audio_latents: &Tensor,
    base_token_count: usize,
) -> Result<Tensor> {
    let total_tokens = audio_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "audio token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }
    if total_tokens == base_token_count {
        return Ok(audio_latents.clone());
    }
    audio_latents
        .narrow(1, 0, base_token_count)
        .map_err(Into::into)
}

#[derive(Debug, Clone, Copy)]
struct RenderPromptInputOptions {
    include_unconditional: bool,
    include_alt: bool,
}

#[derive(Debug)]
struct RenderPromptInputs {
    cond_context: Tensor,
    uncond_context: Option<Tensor>,
    audio_shape: Option<AudioLatentShape>,
    audio_context: Option<Tensor>,
    uncond_audio_context: Option<Tensor>,
    alt_context: Option<Tensor>,
    alt_audio_context: Option<Tensor>,
    video_positions: Tensor,
    audio_positions: Option<Tensor>,
}

fn prepare_render_prompt_inputs(
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    options: RenderPromptInputOptions,
) -> Result<RenderPromptInputs> {
    let cond_context = prepared
        .prompt
        .conditional
        .video_encoding
        .to_device(device)?;
    let uncond_context = if options.include_unconditional {
        Some(
            prepared
                .prompt
                .unconditional
                .video_encoding
                .to_device(device)?,
        )
    } else {
        None
    };
    let audio_context = prepared
        .prompt
        .conditional
        .audio_encoding
        .as_ref()
        .map(|tensor| tensor.to_device(device))
        .transpose()?;
    let uncond_audio_context = if options.include_unconditional {
        prepared
            .prompt
            .unconditional
            .audio_encoding
            .as_ref()
            .map(|tensor| tensor.to_device(device))
            .transpose()?
    } else {
        None
    };
    let alt_context = if options.include_alt {
        prepared
            .debug_alt_prompt
            .as_ref()
            .map(|prompt| prompt.video_encoding.to_device(device))
            .transpose()?
    } else {
        None
    };
    let alt_audio_context = if options.include_alt {
        prepared
            .debug_alt_prompt
            .as_ref()
            .and_then(|prompt| prompt.audio_encoding.as_ref())
            .map(|tensor| tensor.to_device(device))
            .transpose()?
    } else {
        None
    };
    let video_positions = prepared.video_positions.to_device(device)?;
    let audio_positions = prepared
        .audio_positions
        .as_ref()
        .map(|tensor| tensor.to_device(device))
        .transpose()?;

    Ok(RenderPromptInputs {
        cond_context,
        uncond_context,
        audio_shape: prepared.audio_latent_shape,
        audio_context,
        uncond_audio_context,
        alt_context,
        alt_audio_context,
        video_positions,
        audio_positions,
    })
}

struct Ltx2VaeLatentStats {
    mean: Tensor,
    std: Tensor,
    broadcast_cache: Mutex<HashMap<Ltx2VaeLatentStatsBroadcastKey, (Tensor, Tensor)>>,
}

impl Ltx2VaeLatentStats {
    fn from_tensors(mean: Tensor, std: Tensor) -> Self {
        Self {
            mean,
            std,
            broadcast_cache: Mutex::new(HashMap::new()),
        }
    }

    #[cfg(test)]
    fn from_tensors_for_test(mean: Tensor, std: Tensor) -> Self {
        Self::from_tensors(mean, std)
    }

    fn load(plan: &Ltx2GeneratePlan, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&Path::new(&plan.vae_checkpoint_path)),
                dtype,
                device,
            )?
        };
        let config = ltx2_video_vae_config(plan);
        let stats_vb = if plan.vae_in_checkpoint {
            vb.pp("vae")
        } else {
            vb
        };
        let stats_vb = stats_vb.pp("per_channel_statistics");
        let mean = if stats_vb.contains_tensor("mean-of-means") {
            stats_vb.get(config.latent_channels, "mean-of-means")?
        } else {
            tracing::debug!(
                checkpoint = %plan.vae_checkpoint_path,
                "native LTX-2 VAE checkpoint missing mean-of-means statistics, falling back to config defaults"
            );
            Tensor::new(config.latents_mean.as_slice(), device)?.to_dtype(dtype)?
        };
        let std = if stats_vb.contains_tensor("std-of-means") {
            stats_vb.get(config.latent_channels, "std-of-means")?
        } else {
            tracing::debug!(
                checkpoint = %plan.vae_checkpoint_path,
                "native LTX-2 VAE checkpoint missing std-of-means statistics, falling back to config defaults"
            );
            Tensor::new(config.latents_std.as_slice(), device)?.to_dtype(dtype)?
        };
        Ok(Self::from_tensors(mean, std))
    }

    fn broadcast_tensors_for(&self, latents: &Tensor) -> Result<((Tensor, Tensor), bool)> {
        let channels = latents.dim(1)?;
        let key = Ltx2VaeLatentStatsBroadcastKey {
            channels,
            dtype: format!("{:?}", latents.dtype()),
            device: format!("{:?}", latents.device()),
        };
        if let Some((mean, std)) = self
            .broadcast_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(&key)
            .cloned()
        {
            return Ok(((mean, std), true));
        }

        let mean = self
            .mean
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?;
        let mean = if mean.dtype() == latents.dtype() {
            mean
        } else {
            mean.to_dtype(latents.dtype())?
        };
        let std = self
            .std
            .reshape((1, channels, 1, 1, 1))?
            .to_device(latents.device())?;
        let std = if std.dtype() == latents.dtype() {
            std
        } else {
            std.to_dtype(latents.dtype())?
        };
        self.broadcast_cache
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(key, (mean.clone(), std.clone()));
        Ok(((mean, std), false))
    }

    fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let ((mean, std), _) = self.broadcast_tensors_for(latents)?;
        latents
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
            .map_err(Into::into)
    }

    fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let ((mean, std), _) = self.broadcast_tensors_for(latents)?;
        latents
            .broadcast_mul(&std)?
            .broadcast_add(&mean)
            .map_err(Into::into)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Ltx2VaeLatentStatsBroadcastKey {
    channels: usize,
    dtype: String,
    device: String,
}

pub struct Ltx2RuntimeSession {
    device: Option<candle_core::Device>,
    prompt_encoder: Option<NativePromptEncoder>,
    /// Cached output of the last successful `encode_prompt_pair_with_unconditional`
    /// call. The prompt encoder is intentionally consumed during the first
    /// `prepare()` so its VRAM can be freed for the transformer (see the
    /// `take()` + drop pattern below); that leaves subsequent `prepare()`
    /// calls on the same session with no encoder. For the render-chain
    /// path every stage shares the same prompt tokens, so we cache the
    /// encoding after the first encode and reuse it on follow-up stages —
    /// no re-encode, no encoder re-load, no VRAM re-hit.
    cached_prompt_encoding: Option<CachedPromptEncoding>,
    /// Optional slot wired into `render_real_distilled_av` so
    /// `Ltx2Engine::render_chain_stage` can snapshot the pre-VAE-decode
    /// final latents and forward them to the next chain stage as a
    /// [`crate::chain::ChainTail`]. `None` outside chain flow.
    pub(crate) tail_capture: Option<std::sync::Arc<std::sync::Mutex<Option<Tensor>>>>,
    /// GPU ordinal inherited from `Ltx2Engine`. Used for the deferred CUDA
    /// device creation in `prepare()` and post-drop synchronization.
    gpu_ordinal: usize,
    /// True when this session has owned, or will lazily create, CUDA state
    /// whose pending work must be synchronized after unload drops all tensors
    /// and devices.
    cuda_state_on_unload: bool,
}

/// Remembers the last `encode_prompt_pair_with_unconditional` call so
/// successive `prepare()` calls with the same prompt can skip the encoder
/// entirely — used by the render-chain path where stages share a prompt.
struct CachedPromptEncoding {
    token_pair: super::text::gemma::EncodedPromptPair,
    encode_unconditional: bool,
    encoding: NativePromptEncoding,
    prompt_device_is_cuda: bool,
    prepared_device: candle_core::Device,
}

impl Ltx2RuntimeSession {
    pub fn new(
        device: candle_core::Device,
        prompt_encoder: NativePromptEncoder,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            cuda_state_on_unload: device.is_cuda(),
            device: Some(device),
            prompt_encoder: Some(prompt_encoder),
            cached_prompt_encoding: None,
            tail_capture: None,
            gpu_ordinal,
        }
    }

    pub fn new_deferred_cuda(prompt_encoder: NativePromptEncoder, gpu_ordinal: usize) -> Self {
        Self {
            device: None,
            prompt_encoder: Some(prompt_encoder),
            cached_prompt_encoding: None,
            tail_capture: None,
            gpu_ordinal,
            cuda_state_on_unload: true,
        }
    }

    pub(crate) fn has_cuda_state(&self) -> bool {
        self.cuda_state_on_unload
    }

    /// Arm the pre-VAE-decode latent capture slot. The distilled render
    /// path writes its `final_video_latents` into the returned slot when
    /// this is set, letting a caller drain the raw latents after a render
    /// completes. Kept after the v1.1 decoded-pixel-carryover switch in
    /// case future work (e.g. quality-diagnostic tooling) wants access
    /// to the pre-decode tensor; the production chain path no longer
    /// arms it.
    #[allow(dead_code)]
    pub(crate) fn arm_tail_capture(&mut self) -> std::sync::Arc<std::sync::Mutex<Option<Tensor>>> {
        let slot = std::sync::Arc::new(std::sync::Mutex::new(None));
        self.tail_capture = Some(std::sync::Arc::clone(&slot));
        slot
    }

    /// Disarm the latent capture slot. See [`arm_tail_capture`].
    #[allow(dead_code)]
    pub(crate) fn clear_tail_capture(&mut self) {
        self.tail_capture = None;
    }

    /// Whether the live prompt encoder is still held. False once a
    /// `prepare()` has consumed it (its VRAM is handed to the transformer),
    /// after which only a matching [`Self::can_reuse_for`] cache hit can
    /// serve another `prepare()`. Test-only observable: production code asks
    /// [`Self::can_reuse_for`], which folds this in with the cache.
    #[cfg(test)]
    pub(crate) fn has_prompt_encoder(&self) -> bool {
        self.prompt_encoder.is_some()
    }

    /// Whether this session can serve `plan` without a rebuild. Returns
    /// `true` if the encoder is still available OR the cached encoding
    /// matches the plan's prompt tokens exactly. Callers use this to decide
    /// whether to reuse a persisted runtime or drop it and build a fresh
    /// one — the only way to recover when the encoder has been consumed on
    /// a prior `prepare()` and a different prompt arrives.
    ///
    /// What reuse preserves is the cached prompt encoding and the compute
    /// device handle, which is what lets a matching plan skip the Gemma
    /// load entirely. It does NOT keep a transformer or VAE warm: neither
    /// lives in this session — `prepare()` builds them per call and hands
    /// them back in the `NativePreparedRun`.
    pub fn can_reuse_for(&self, plan: &Ltx2GeneratePlan) -> bool {
        if self.prompt_encoder.is_some() {
            return true;
        }
        let Ok(encode_unconditional) = prompt_requires_unconditional_context(plan) else {
            return false;
        };
        // Alt-prompt debug mode requires the live encoder; cache alone
        // isn't sufficient.
        if ltx_debug_alt_prompt().is_some() {
            return false;
        }
        self.cached_prompt_encoding.as_ref().is_some_and(|cached| {
            cached.encode_unconditional == encode_unconditional
                && cached.token_pair == plan.prompt_tokens
        })
    }

    #[cfg(test)]
    pub fn prepare(&mut self, plan: &Ltx2GeneratePlan) -> Result<NativePreparedRun> {
        let mut plan = plan.clone();
        self.prepare_with_progress(&mut plan, None)
    }

    pub fn prepare_with_progress(
        &mut self,
        plan: &mut Ltx2GeneratePlan,
        progress: Option<&ProgressCallback>,
    ) -> Result<NativePreparedRun> {
        let prepare_total_start = Instant::now();
        reject_oversized_axis_without_composition(plan)?;
        let encode_unconditional_prompt = prompt_requires_unconditional_context(plan)?;
        if plan.scene_embeddings_path.is_some()
            && prompt_requires_unconditional_context_for_plan(plan)?
        {
            // The saved embeddings are one conditional context; upstream's HDR
            // pipeline is distilled and never runs classifier-free guidance.
            // Reusing them as the negative would cancel guidance to nothing
            // and quietly change what the user asked for, so refuse instead.
            anyhow::bail!(
                "pre-computed control embeddings carry no negative context, but this plan \
                 needs one (guidance > 1.0). Run the control on its distilled checkpoint, \
                 or drop the guidance override."
            );
        }
        let alt_prompt_env = ltx_debug_alt_prompt();
        // Chain path fast-path: if a previous `prepare()` already encoded
        // the exact same prompt+unconditional combo, reuse those embeddings
        // instead of demanding the encoder back. Disabled when the
        // `MOLD_LTX_DEBUG_ALT_PROMPT` debug hook is active because that branch
        // still needs the live encoder.
        let cache_hit = alt_prompt_env.is_none()
            && self.cached_prompt_encoding.as_ref().is_some_and(|cached| {
                cached.encode_unconditional == encode_unconditional_prompt
                    && cached.token_pair == plan.prompt_tokens
            });
        let (prompt_device_is_cuda, prepared_device, prompt, debug_alt_prompt) = if let Some(path) =
            plan.scene_embeddings_path.as_deref()
        {
            // Pre-computed context: no Gemma load, no encode. The encoder is
            // left untaken so a later stage that does need it still finds it.
            let scene_start = Instant::now();
            let prompt = load_scene_embeddings(path)?;
            log_timing("prepare.scene_embeddings", scene_start);
            emit_phase_done(
                progress,
                ProgressPhase::PromptEncode,
                "Loading control embeddings",
                scene_start.elapsed(),
            );
            (false, candle_core::Device::Cpu, prompt, None)
        } else if cache_hit {
            let cached = self
                .cached_prompt_encoding
                .as_ref()
                .expect("cache_hit implies cached_prompt_encoding is Some");
            log_timing("prepare.prompt_pair", Instant::now());
            (
                cached.prompt_device_is_cuda,
                cached.prepared_device.clone(),
                cached.encoding.clone(),
                None,
            )
        } else {
            let mut prompt_encoder = self
                .prompt_encoder
                .take()
                .context("native LTX-2 prompt encoder is unavailable")?;
            let prompt_device_is_cuda = prompt_encoder.device().is_cuda();
            let prepared_device = if prompt_device_is_cuda || prompt_encoder.device().is_metal() {
                candle_core::Device::Cpu
            } else {
                prompt_encoder.device().clone()
            };
            let prompt_encode_start = Instant::now();
            let prompt_probe = PhaseVramProbe::enter_if("prompt_encode", prompt_device_is_cuda);
            if let Some(progress) = progress {
                progress(ProgressEvent::StageStart {
                    name: "Encoding prompt (Gemma)".to_string(),
                });
            }
            // Closure so an encode that dies of OOM is still reported before
            // the error leaves `prepare`.
            let encoded = (|| -> Result<NativePromptEncoding> {
                move_prompt_encoding_to_device(
                    prompt_encoder.encode_prompt_pair_with_progress(
                        &plan.prompt_tokens,
                        encode_unconditional_prompt,
                        progress,
                    )?,
                    &prepared_device,
                )
            })();
            log_ltx2_phase_vram_result(prompt_probe.finish(), &encoded, None, "");
            let prompt = encoded?;
            emit_phase_done(
                progress,
                ProgressPhase::PromptEncode,
                "Encoding prompt (Gemma)",
                prompt_encode_start.elapsed(),
            );
            log_timing("prepare.prompt_pair", prompt_encode_start);
            let alt_prompt_start = Instant::now();
            let debug_alt_prompt = match alt_prompt_env.clone() {
                Some(alt_prompt) => {
                    let assets =
                        super::text::gemma::GemmaAssets::discover(Path::new(&plan.gemma_root))
                            .with_context(|| {
                                format!(
                            "failed to discover Gemma assets for alternate prompt debug at '{}'",
                            plan.gemma_root
                        )
                            })?;
                    let alt_tokens =
                        assets.encode_prompt_pair(&alt_prompt, plan.negative_prompt.as_deref())?;
                    let alt_prompt = prompt_encoder
                        .encode_prompt_pair(&alt_tokens)
                        .context("failed to encode alternate debug prompt")?;
                    Some(move_embeddings_output_to_device(
                        alt_prompt.conditional,
                        &prepared_device,
                    )?)
                }
                None => None,
            };
            log_timing("prepare.alt_prompt", alt_prompt_start);
            let prompt_debug_start = Instant::now();
            if ltx_debug_enabled() {
                log_prompt_debug_stats(plan, &prompt)?;
                if let Some(alt_prompt) = debug_alt_prompt.as_ref() {
                    log_alt_prompt_debug_stats(plan, &prompt.conditional, alt_prompt)?;
                }
            }
            log_timing("prepare.prompt_debug", prompt_debug_start);
            // Cache the encoding for the next chain stage. Dropping the
            // encoder here (end of the else branch) still happens — we're
            // only holding on to the `NativePromptEncoding` output, not the
            // encoder itself, so the VRAM-free property of the original
            // take() pattern is preserved.
            self.cached_prompt_encoding = Some(CachedPromptEncoding {
                token_pair: plan.prompt_tokens.clone(),
                encode_unconditional: encode_unconditional_prompt,
                encoding: prompt.clone(),
                prompt_device_is_cuda,
                prepared_device: prepared_device.clone(),
            });
            (
                prompt_device_is_cuda,
                prepared_device,
                prompt,
                debug_alt_prompt,
            )
        };
        if let Some(bounds) = plan.auto_duration {
            let path = plan
                .duration_head_path
                .as_deref()
                .context("automatic duration was planned without a duration-head checkpoint")?;
            let duration_dtype = if prepared_device.is_cpu() {
                DType::F32
            } else {
                compute_dtype(&prepared_device)
            };
            let head = super::model::Ltx2DurationHead::from_checkpoint(
                Path::new(path),
                duration_dtype,
                &prepared_device,
            )?;
            plan.num_frames = head.predict_frames(
                Some(&prompt.conditional.video_encoding),
                prompt.conditional.audio_encoding.as_ref(),
                plan.frame_rate,
                bounds,
            )?;
            emit_info(
                progress,
                format!(
                    "Predicted caption duration: {} frames at {} fps",
                    plan.num_frames, plan.frame_rate
                ),
            );
        }
        let mut stage1_shape = derive_stage1_render_shape(
            plan.width,
            plan.height,
            plan.num_frames,
            plan.frame_rate,
            plan.spatial_upscale,
            plan.temporal_upscale,
        );
        if pipeline_uses_two_stage_spatial_refinement(plan.pipeline)
            && plan.spatial_upscale.is_none()
            && stage1_shape.width > 16
            && stage1_shape.height > 16
        {
            let implicit_x2_shape = derive_stage1_render_shape(
                plan.width,
                plan.height,
                plan.num_frames,
                plan.frame_rate,
                Some(Ltx2SpatialUpscale::X2),
                plan.temporal_upscale,
            );
            stage1_shape.width = implicit_x2_shape.width;
            stage1_shape.height = implicit_x2_shape.height;
        }
        let device_handoff_start = Instant::now();
        if prompt_device_is_cuda {
            // The conditioning handoff: the encoder's device is released and
            // the render device is (re)acquired. Measured because this is
            // where a stale encoder allocation shows up as a smaller card.
            let handoff_probe = PhaseVramProbe::enter("device_handoff");
            let handoff = (|| -> Result<()> {
                if self.device.is_none() {
                    let _ = crate::device::post_drop_free_vram_bytes(self.gpu_ordinal);
                    self.device = Some(new_native_cuda_device(self.gpu_ordinal)?);
                } else if let Some(device) = self.device.as_ref() {
                    if device.is_cuda() {
                        device.synchronize()?;
                    }
                }
                Ok(())
            })();
            log_ltx2_phase_vram_result(handoff_probe.finish(), &handoff, None, "");
            handoff?;
        }
        log_timing("prepare.device_handoff", device_handoff_start);
        let positions_start = Instant::now();
        let pixel_shape = VideoPixelShape {
            batch: 1,
            frames: stage1_shape.frames as usize,
            height: stage1_shape.height as usize,
            width: stage1_shape.width as usize,
            fps: stage1_shape.fps as f32,
        };
        let scale_factors = SpatioTemporalScaleFactors::default();
        let video_latent_shape = VideoLatentShape::from_pixel_shape(
            pixel_shape,
            LTX2_VIDEO_LATENT_CHANNELS,
            scale_factors,
        );
        let video_patchifier = VideoLatentPatchifier::new(1);
        let video_positions = scale_video_time_to_seconds(
            &get_pixel_coords(
                &video_token_positions(video_patchifier, video_latent_shape, &prepared_device)?,
                scale_factors,
                true,
            )?,
            pixel_shape.fps,
        )?;

        // The public LTX-2 checkpoints are audio-video transformers even for
        // silent exports. Keep the internal audio branch active whenever the
        // prompt encoder emitted audio conditioning so the denoiser stays on the
        // same multimodal path as upstream; export semantics remain silent
        // unless the request explicitly wants audio output.
        let prompt_has_audio_conditioning = prompt.conditional.audio_encoding.is_some()
            || prompt.unconditional.audio_encoding.is_some();
        let wants_audio_latents = if ltx_debug_disable_audio_branch_enabled() {
            false
        } else {
            plan.execution_graph.wants_audio_output
                || plan.execution_graph.uses_audio_conditioning
                || prompt_has_audio_conditioning
        };
        let (audio_latent_shape, audio_positions, cross_modal_temporal_positions) =
            if wants_audio_latents {
                let audio_shape = AudioLatentShape::from_video_pixel_shape(
                    pixel_shape,
                    LTX2_AUDIO_LATENT_CHANNELS,
                    LTX2_AUDIO_MEL_BINS,
                    LTX2_AUDIO_SAMPLE_RATE,
                    LTX2_AUDIO_HOP_LENGTH,
                    LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
                );
                let audio_patchifier = AudioPatchifier::new(
                    LTX2_AUDIO_SAMPLE_RATE,
                    LTX2_AUDIO_HOP_LENGTH,
                    LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
                    true,
                    0,
                );
                let audio_positions =
                    audio_temporal_positions(audio_patchifier, audio_shape, &prepared_device)?;
                let cross_modal =
                    cross_modal_temporal_positions(&video_positions, &audio_positions)?;
                (Some(audio_shape), Some(audio_positions), Some(cross_modal))
            } else {
                (None, None, None)
            };
        log_timing("prepare.positions", positions_start);

        let retake_mask_start = Instant::now();
        let retake_mask = plan
            .retake_range
            .as_ref()
            .map(|range| retake_temporal_mask(range, stage1_shape.fps, stage1_shape.frames))
            .transpose()?;
        log_timing("prepare.retake_mask", retake_mask_start);
        log_timing("prepare.total", prepare_total_start);

        Ok(NativePreparedRun {
            prompt,
            debug_alt_prompt,
            video_pixel_shape: pixel_shape,
            video_latent_shape,
            audio_latent_shape,
            video_positions,
            audio_positions,
            cross_modal_temporal_positions,
            retake_mask,
        })
    }

    /// Render an audio-only plan. Separate entry point from
    /// [`Self::render_native_video`] because T2A emits no frames at all — the
    /// video renderer's synthetic-placeholder fallback would happily fabricate
    /// some, which is exactly the wrong answer.
    pub fn render_native_audio(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
        progress: Option<&ProgressCallback>,
        cancellation: Option<&InferenceCancellationToken>,
    ) -> Result<NativeAudioTrack> {
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }
        if !plan.pipeline.is_audio_only() {
            bail!(
                "render_native_audio called for the LTX-2 {:?} pipeline, which renders video",
                plan.pipeline
            );
        }
        if !Path::new(&plan.checkpoint_path).is_file() {
            bail!(
                "missing LTX-2 checkpoint for text-to-audio: {}",
                plan.checkpoint_path
            );
        }
        let device = self
            .device
            .as_ref()
            .context("native LTX-2 compute device was not initialized")?;
        render_real_t2a_audio(plan, prepared, device, progress, cancellation)
    }

    pub fn render_native_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
        progress: Option<&ProgressCallback>,
        cancellation: Option<&InferenceCancellationToken>,
    ) -> Result<NativeRenderedVideo> {
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }
        if plan.pipeline.is_audio_only() {
            bail!(
                "the LTX-2 {:?} pipeline produces audio only; render it through \
                 render_native_audio",
                plan.pipeline
            );
        }
        let device = self
            .device
            .as_ref()
            .context("native LTX-2 compute device was not initialized")?;
        if let Some(rendered) =
            self.try_render_real_video(plan, prepared, device, progress, cancellation)?
        {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] render_native_video using real path pipeline={:?}",
                    plan.pipeline
                );
            }
            return Ok(rendered);
        }
        if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
            eprintln!(
                "[ltx2-debug] render_native_video falling back to placeholder path pipeline={:?}",
                plan.pipeline
            );
        }

        let summary = RenderSummary::from_prepared(prepared)?;
        let seed = plan.seed ^ 0x4c54_5832_4e41_5449;
        let mut rng = StdRng::seed_from_u64(seed);
        let phase = rng.gen_range(0.0..std::f32::consts::TAU);
        let base_width = prepared.video_pixel_shape.width as u32;
        let base_height = prepared.video_pixel_shape.height as u32;
        let base_frames = prepared.video_pixel_shape.frames as u32;
        let overlays = load_conditioning_overlays(plan, base_width, base_height, base_frames)?;

        let mut frames = Vec::with_capacity(base_frames as usize);
        for frame_idx in 0..base_frames {
            if let Some(token) = cancellation {
                token.checkpoint()?;
            }
            let mut frame = RgbImage::new(base_width, base_height);
            let t = if base_frames <= 1 {
                0.0
            } else {
                frame_idx as f32 / (base_frames - 1) as f32
            };
            let retake_strength = prepared
                .retake_mask
                .as_ref()
                .and_then(|mask| mask.get(frame_idx as usize))
                .copied()
                .unwrap_or(0.0);
            fill_background(
                &mut frame,
                t,
                phase,
                &summary,
                retake_strength,
                plan.execution_graph.uses_audio_conditioning,
                plan.execution_graph.uses_reference_video_conditioning,
            );
            apply_conditioning_overlays(&mut frame, frame_idx, base_frames, &overlays);
            frames.push(frame);
        }
        if plan.temporal_upscale.is_some() {
            frames = temporally_upsample_frames_x2(&frames, Some(plan.num_frames));
        }
        if plan.spatial_upscale.is_some() || plan.width != base_width || plan.height != base_height
        {
            frames = spatially_upsample_frames(&frames, plan.width, plan.height);
        }

        Ok(NativeRenderedVideo {
            frames,
            hdr_frames_written: None,
            audio_track: None,
            has_audio: plan.execution_graph.wants_audio_output,
            audio_sample_rate: plan.execution_graph.wants_audio_output.then_some(48_000),
            audio_channels: plan.execution_graph.wants_audio_output.then_some(2),
        })
    }

    fn try_render_real_video(
        &self,
        plan: &Ltx2GeneratePlan,
        prepared: &NativePreparedRun,
        device: &candle_core::Device,
        progress: Option<&ProgressCallback>,
        cancellation: Option<&InferenceCancellationToken>,
    ) -> Result<Option<NativeRenderedVideo>> {
        // Order matters. A checkpoint with no bytes to read means there are no
        // weights to render with — the synthetic path is the only thing left,
        // and the unit tests rely on it. But once real weights are on disk,
        // refusing the real path and quietly returning plausible-looking
        // placeholder frames is worse than any error: the user gets a file
        // that looks like a render and is not one. A checkpoint that *does*
        // carry bytes must parse or fail; see the error handling below.
        if checkpoint_has_no_weights(Path::new(&plan.checkpoint_path)) {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] real path rejected because checkpoint has no weights: {}",
                    plan.checkpoint_path
                );
            }
            return Ok(None);
        }
        if !supports_real_video_path(plan) {
            if ltx_debug_enabled() || env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
                eprintln!(
                    "[ltx2-debug] real path rejected by supports_real_video_path pipeline={:?}",
                    plan.pipeline
                );
            }
            bail!(
                "the LTX-2 {:?} pipeline cannot render this combination of inputs \
                 (spatial upscale: {:?}, temporal upscale: {:?}, source video: {}, \
                 conditioning audio: {}, reference-video conditioning: {}, retake \
                 masking: {}, LoRAs: {}). Choose a pipeline that supports them, or drop \
                 the unsupported input.",
                plan.pipeline,
                plan.spatial_upscale,
                plan.temporal_upscale,
                plan.conditioning.video_path.is_some(),
                plan.conditioning.audio_path.is_some(),
                plan.execution_graph.uses_reference_video_conditioning,
                plan.execution_graph.uses_retake_masking,
                plan.loras.len(),
            );
        }
        let render = match plan.pipeline {
            PipelineKind::Distilled => render_real_distilled_av(
                plan,
                prepared,
                device,
                progress,
                cancellation,
                self.tail_capture.as_ref(),
            ),
            PipelineKind::OneStage => {
                render_real_one_stage_av(plan, prepared, device, progress, cancellation)
            }
            PipelineKind::TwoStage
            | PipelineKind::TwoStageHq
            | PipelineKind::IcLora
            | PipelineKind::Keyframe
            | PipelineKind::A2Vid
            | PipelineKind::LipDub => {
                render_real_two_stage_av(plan, prepared, device, progress, cancellation)
            }
            PipelineKind::Retake => {
                render_real_retake_av(plan, prepared, device, progress, cancellation)
            }
            // Rejected at the top of `render_native_video`; unreachable here.
            PipelineKind::T2a => bail!("the LTX-2 text-to-audio pipeline produces no video frames"),
        };
        // Every failure from here on is a real failure. A checkpoint that is
        // present but unreadable is corrupt weights — a truncated download —
        // and `candle` surfaces the `safetensors` error transparently, so it
        // arrives as bare text like "header too small". Matching that text and
        // rendering the synthetic gradient instead produced a file with the
        // requested size, length and frame rate, containing no picture, and
        // reported it as a successful save while hiding the corruption.
        render.map(Some)
    }
}

fn move_prompt_encoding_to_device(
    prompt: NativePromptEncoding,
    device: &candle_core::Device,
) -> Result<NativePromptEncoding> {
    Ok(NativePromptEncoding {
        conditional: move_embeddings_output_to_device(prompt.conditional, device)?,
        unconditional: move_embeddings_output_to_device(prompt.unconditional, device)?,
    })
}

fn move_embeddings_output_to_device(
    output: EmbeddingsProcessorOutput,
    device: &candle_core::Device,
) -> Result<EmbeddingsProcessorOutput> {
    Ok(EmbeddingsProcessorOutput {
        video_encoding: output.video_encoding.to_device(device)?,
        audio_encoding: output
            .audio_encoding
            .map(|tensor| tensor.to_device(device))
            .transpose()?,
        attention_mask: output.attention_mask.to_device(device)?,
    })
}

#[derive(Debug, Clone)]
struct ConditioningOverlay {
    frame: u32,
    strength: f32,
    image: RgbImage,
}

#[derive(Debug, Clone, Copy)]
struct RenderSummary {
    video_mean: f32,
    video_energy: f32,
    audio_mean: f32,
    audio_energy: f32,
    negative_bias: f32,
}

impl RenderSummary {
    fn from_prepared(prepared: &NativePreparedRun) -> Result<Self> {
        let video_mean = tensor_mean(&prepared.prompt.conditional.video_encoding)?;
        let negative_bias = tensor_mean(&prepared.prompt.unconditional.video_encoding)?;
        let video_energy = tensor_energy(&prepared.video_positions)?;
        let audio_mean = prepared
            .prompt
            .conditional
            .audio_encoding
            .as_ref()
            .map(tensor_mean)
            .transpose()?
            .unwrap_or(0.0);
        let audio_energy = prepared
            .audio_positions
            .as_ref()
            .map(tensor_energy)
            .transpose()?
            .unwrap_or(0.0);
        Ok(Self {
            video_mean,
            video_energy,
            audio_mean,
            audio_energy,
            negative_bias,
        })
    }
}

fn tensor_mean(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?)
}

fn tensor_energy(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?)
}

fn emit_denoise_progress(
    progress: Option<&ProgressCallback>,
    step: usize,
    total: usize,
    elapsed: std::time::Duration,
) {
    if let Some(progress) = progress {
        progress(ProgressEvent::DenoiseStep {
            step,
            total,
            elapsed,
        });
    }
}

fn load_conditioning_overlays(
    plan: &Ltx2GeneratePlan,
    width: u32,
    height: u32,
    stage_frames: u32,
) -> Result<Vec<ConditioningOverlay>> {
    plan.conditioning
        .images
        .iter()
        .map(|image| {
            let overlay = image::open(&image.path)
                .with_context(|| {
                    format!("failed to load staged conditioning image '{}'", image.path)
                })?
                .to_rgb8();
            Ok(ConditioningOverlay {
                frame: remap_conditioning_frame(image.frame, plan.num_frames, stage_frames),
                strength: image.strength,
                image: imageops::resize(&overlay, width, height, imageops::FilterType::Triangle),
            })
        })
        .collect()
}

fn remap_conditioning_frame(source_frame: u32, source_total: u32, target_total: u32) -> u32 {
    if source_total <= 1 || target_total <= 1 {
        return 0;
    }
    let mapped = ((source_frame as u64 * (target_total - 1) as u64)
        + ((source_total - 1) / 2) as u64)
        / (source_total - 1) as u64;
    mapped.min((target_total - 1) as u64) as u32
}

fn fill_background(
    frame: &mut RgbImage,
    t: f32,
    phase: f32,
    summary: &RenderSummary,
    retake_strength: f32,
    uses_audio_conditioning: bool,
    uses_reference_video: bool,
) {
    let width = frame.width().max(1) as f32;
    let height = frame.height().max(1) as f32;
    let motion = 1.5 + summary.video_energy.abs() * 3.0;
    let audio_motion = 1.0 + summary.audio_energy.abs() * 2.0;
    let bias = summary.negative_bias.tanh() * 0.15;
    let highlight = 0.15 + retake_strength * 0.35;

    for (x, y, pixel) in frame.enumerate_pixels_mut() {
        let fx = x as f32 / width;
        let fy = y as f32 / height;
        let primary = ((fx * 6.0 + t * motion + phase).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let secondary =
            ((fy * 4.0 - t * (motion * 0.7) + phase * 0.5).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
        let ripple =
            (((fx + fy) * (3.0 + summary.audio_mean.abs()) + t * audio_motion + phase * 1.7).sin()
                * 0.5
                + 0.5)
                .clamp(0.0, 1.0);

        let mut r = primary * (200.0 + summary.video_mean.abs() * 80.0) + secondary * 32.0;
        let mut g = secondary * (180.0 + summary.audio_mean.abs() * 90.0) + ripple * 40.0;
        let mut b = ripple * 220.0 + primary * 18.0 + bias * 255.0;

        if uses_audio_conditioning && fy > 0.78 {
            let bars = ((fx * 18.0 + t * 9.0 + phase).sin() * 0.5 + 0.5) * 110.0;
            g += bars;
            b += bars * 0.35;
        }
        if uses_reference_video && fx < 0.08 {
            r += 36.0;
            b += 22.0;
        }
        if retake_strength > 0.0 && (!(0.03..=0.97).contains(&fx) || !(0.03..=0.97).contains(&fy)) {
            r += highlight * 255.0;
            g += highlight * 96.0;
        }

        *pixel = Rgb([
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        ]);
    }
}

fn apply_conditioning_overlays(
    frame: &mut RgbImage,
    frame_idx: u32,
    total_frames: u32,
    overlays: &[ConditioningOverlay],
) {
    for overlay in overlays {
        let alpha = overlay_alpha(overlay, frame_idx, total_frames);
        if alpha <= 0.0 {
            continue;
        }
        for (dst, src) in frame.pixels_mut().zip(overlay.image.pixels()) {
            let alpha = alpha.clamp(0.0, 1.0);
            let inv = 1.0 - alpha;
            *dst = Rgb([
                (dst[0] as f32 * inv + src[0] as f32 * alpha).round() as u8,
                (dst[1] as f32 * inv + src[1] as f32 * alpha).round() as u8,
                (dst[2] as f32 * inv + src[2] as f32 * alpha).round() as u8,
            ]);
        }
    }
}

fn overlay_alpha(overlay: &ConditioningOverlay, frame_idx: u32, total_frames: u32) -> f32 {
    let distance = overlay.frame.abs_diff(frame_idx) as f32;
    let spread = (total_frames.max(8) as f32 / 6.0).max(1.0);
    let falloff = (1.0 - distance / spread).clamp(0.0, 1.0);
    (overlay.strength.max(0.1) * falloff).clamp(0.0, 0.85)
}

#[cfg(feature = "cuda")]
fn new_native_cuda_device(ordinal: usize) -> Result<candle_core::Device> {
    let device = candle_core::Device::new_cuda(ordinal)?;
    let cuda = device.as_cuda_device()?;
    if cuda.is_event_tracking() {
        unsafe {
            cuda.disable_event_tracking();
        }
    }
    Ok(device)
}

#[cfg(not(feature = "cuda"))]
fn new_native_cuda_device(_ordinal: usize) -> Result<candle_core::Device> {
    anyhow::bail!("CUDA backend is unavailable in this build")
}

const DISTILLED_STAGE1_SIGMAS_NO_TERMINAL: &[f32] = &[
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
];

const DISTILLED_STAGE2_SIGMAS_NO_TERMINAL: &[f32] = &[0.909375, 0.725, 0.421875];

fn distilled_stage2_sigmas_no_terminal(_plan: &Ltx2GeneratePlan) -> &'static [f32] {
    // One authority for every distilled LTX-2 generation: current upstream
    // `STAGE_2_DISTILLED_SIGMA_VALUES` is the fixed subset beginning at
    // 0.909375. A ComfyUI fixture briefly introduced a model-name fork at
    // 0.85, but the official pipeline does not version this schedule.
    DISTILLED_STAGE2_SIGMAS_NO_TERMINAL
}

/// Refuse a resolution past the trained RoPE span on a pipeline that denoises
/// the requested shape once.
///
/// `mold_core::validation` already refuses this at admission, gated on the same
/// `refines_spatially` predicate. This is the backstop for the paths that reach
/// the engine without it — a plan reconstructed from persisted state, or a
/// pipeline forced after validation — because the failure it prevents is a
/// finished video with degraded structure and no error anywhere.
fn reject_oversized_axis_without_composition(plan: &Ltx2GeneratePlan) -> Result<()> {
    let span = mold_core::validation::LTX2_MAX_AXIS_PIXELS;
    let longest = plan.width.max(plan.height);
    if longest <= span {
        return Ok(());
    }
    if !pipeline_uses_two_stage_spatial_refinement(plan.pipeline) {
        bail!(
            "{}x{} has a {longest}px axis, past the {span}px span these checkpoints were trained \
             on, and the {:?} pipeline denoises the requested shape in one pass — there is no \
             tiled refinement to renormalize positions. Use a checkpoint that ships the spatial \
             upsampler, or render at or below {span}px on the long edge.",
            plan.width,
            plan.height,
            plan.pipeline,
        );
    }
    // Composing is not enough on its own: the ceiling belongs to the *rung*.
    // A x1.5 upscale only divides by 1.5, so a 4K output leaves stage 1 at
    // 2560px — and stage 2 tiles the refinement, never stage 1.
    mold_core::validation::validate_ltx2_stage1_span(plan.width, plan.height, plan.spatial_upscale)
        .map_err(anyhow::Error::msg)
}

/// Whether this pipeline renders stage 1 reduced and refines the upsampled
/// result.
///
/// Delegates to `Ltx2PipelineMode::refines_spatially` rather than restating the
/// set: `mold_core::validation` admits resolutions past the trained RoPE span
/// on exactly this predicate, and a copy that drifted would admit a shape this
/// function then renders in one out-of-distribution pass.
fn pipeline_uses_two_stage_spatial_refinement(pipeline: PipelineKind) -> bool {
    pipeline.wire_mode().refines_spatially()
}

fn effective_native_guidance_scale(plan: &Ltx2GeneratePlan) -> f64 {
    match plan.pipeline {
        PipelineKind::Distilled | PipelineKind::Retake => 1.0,
        _ => plan.guidance,
    }
}

fn stage_guidance_scale(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<f64> {
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::Distilled | PipelineKind::IcLora | PipelineKind::Retake, _) => 1.0,
        (PipelineKind::TwoStage, 1)
        | (PipelineKind::TwoStageHq, 1)
        | (PipelineKind::A2Vid, 1)
        | (PipelineKind::Keyframe, 1) => 1.0,
        _ => {
            let _ = denoise_pass_plan(plan, stage_index)?;
            effective_native_guidance_scale(plan)
        }
    })
}

fn stage_sampler_mode(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<SamplerMode> {
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::TwoStageHq, 0 | 1) => SamplerMode::Res2S,
        _ => denoise_pass_plan(plan, stage_index)?.sampler,
    })
}

fn multimodal_guider_requires_unconditional_context(params: &MultiModalGuiderParams) -> bool {
    (params.cfg_scale - 1.0).abs() > f64::EPSILON
}

/// The transformer block STG perturbs by default, which differs between the
/// 22B and 19B stacks because their depths differ.
fn default_stg_block(plan: &Ltx2GeneratePlan) -> usize {
    if plan.preset.name == "ltx-2.3-22b" {
        28
    } else {
        29
    }
}

/// Per-(pipeline, stage) guider constants, before any request override.
fn stage_multimodal_guider_defaults(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Option<(MultiModalGuiderParams, MultiModalGuiderParams)> {
    match (plan.pipeline, stage_index) {
        (PipelineKind::A2Vid, 0) => {
            let stg_block = default_stg_block(plan);
            Some((
                MultiModalGuiderParams {
                    cfg_scale: 3.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
                MultiModalGuiderParams::default(),
            ))
        }
        (PipelineKind::TwoStage | PipelineKind::Keyframe, 0) => {
            let stg_block = default_stg_block(plan);
            Some((
                MultiModalGuiderParams {
                    cfg_scale: 3.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
                MultiModalGuiderParams {
                    cfg_scale: 7.0,
                    stg_scale: 1.0,
                    stg_blocks: vec![stg_block],
                    rescale_scale: 0.7,
                    modality_scale: 3.0,
                    skip_step: 0,
                },
            ))
        }
        // Text-to-audio. Upstream's audio guider constants
        // (`ltx-pipelines/utils/constants.py:50-59`, `stg_blocks` retargeted
        // to 28 for 2.3 at `:78`) with `modality_scale` pinned to 1.0 —
        // audio-only generation has no video branch, so the cross-modal term
        // is meaningless and upstream disables it explicitly at
        // `ltx-pipelines/src/ltx_pipelines/t2a_one_stage.py:184`.
        (PipelineKind::T2a, 0) => Some((
            MultiModalGuiderParams::default(),
            MultiModalGuiderParams {
                cfg_scale: 7.0,
                stg_scale: 1.0,
                stg_blocks: vec![default_stg_block(plan)],
                rescale_scale: 0.7,
                modality_scale: 1.0,
                skip_step: 0,
            },
        )),
        (PipelineKind::TwoStageHq, 0) => Some((
            MultiModalGuiderParams {
                cfg_scale: 3.0,
                stg_scale: 0.0,
                stg_blocks: Vec::new(),
                rescale_scale: 0.45,
                modality_scale: 3.0,
                skip_step: 0,
            },
            MultiModalGuiderParams {
                cfg_scale: 7.0,
                stg_scale: 0.0,
                stg_blocks: Vec::new(),
                rescale_scale: 1.0,
                modality_scale: 3.0,
                skip_step: 0,
            },
        )),
        _ => None,
    }
}

/// Apply the request's guidance overrides to one guider.
///
/// A guider the pipeline left at [`MultiModalGuiderParams::default`] is
/// switched off on purpose (LTX-2's `a2-vid` audio guider), so overrides skip
/// it — tuning guidance must never turn on a pass that costs another forward
/// through the transformer. When an override enables STG on a stage that ships
/// with it off, the block list falls back to the preset's default block so the
/// perturbed pass has something to perturb.
fn apply_guidance_overrides(
    plan: &Ltx2GeneratePlan,
    params: &mut MultiModalGuiderParams,
    overrides: &Ltx2GuidanceOverrides,
) -> Result<()> {
    if *params == MultiModalGuiderParams::default() {
        return Ok(());
    }
    if let Some(stg_scale) = overrides.stg_scale {
        params.stg_scale = stg_scale;
    }
    if let Some(blocks) = &overrides.stg_blocks {
        let depth = plan.preset.transformer.num_layers;
        for block in blocks {
            let block = *block as usize;
            if block >= depth {
                anyhow::bail!(
                    "guidance_overrides.stg_blocks contains block {block}, but {} has {depth} transformer blocks (0-{})",
                    plan.preset.name,
                    depth.saturating_sub(1)
                );
            }
        }
        params.stg_blocks = blocks.iter().map(|block| *block as usize).collect();
    }
    if let Some(rescale_scale) = overrides.rescale_scale {
        params.rescale_scale = rescale_scale;
    }
    if let Some(modality_scale) = overrides.modality_scale {
        params.modality_scale = modality_scale;
    }
    if let Some(skip_step) = overrides.skip_step {
        params.skip_step = skip_step as usize;
    }
    if params.stg_scale != 0.0 && params.stg_blocks.is_empty() {
        params.stg_blocks = vec![default_stg_block(plan)];
    }
    Ok(())
}

fn stage_multimodal_guider_params(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Result<Option<(MultiModalGuiderParams, MultiModalGuiderParams)>> {
    let Some((mut video_params, mut audio_params)) =
        stage_multimodal_guider_defaults(plan, stage_index)
    else {
        return Ok(None);
    };
    if let Some(overrides) = &plan.guidance_overrides {
        apply_guidance_overrides(plan, &mut video_params, overrides)?;
        apply_guidance_overrides(plan, &mut audio_params, overrides)?;
    }
    Ok(Some((video_params, audio_params)))
}

/// Build a prompt encoding from a control adapter's saved text embeddings.
///
/// Upstream saves `video_context` and `audio_context` with
/// `safetensors.torch.save_file` and reads them back verbatim
/// (`hdr_ic_lora.py:250-256`); there is no attention mask in the file because
/// the pipeline attends to the whole fixed 1,024-token context. The
/// connector's binary mask is therefore all ones — and the saved encodings are
/// already post-connector output, so nothing further is applied to them.
///
/// The same tensors fill the unconditional slot. That slot is unread on the
/// distilled pipelines these embeddings ship for; `prepare` refuses the plans
/// where it would be read, rather than letting a duplicated context silently
/// neutralize guidance.
fn load_scene_embeddings(path: &str) -> Result<NativePromptEncoding> {
    let device = candle_core::Device::Cpu;
    let tensors = candle_core::safetensors::load(path, &device)
        .with_context(|| format!("failed to read control text embeddings from '{path}'"))?;
    let video_encoding = tensors
        .get("video_context")
        .with_context(|| format!("'{path}' has no `video_context` tensor"))?
        .clone();
    let audio_encoding = tensors.get("audio_context").cloned();

    let (batch, sequence, _) = video_encoding.dims3().with_context(|| {
        format!("`video_context` in '{path}' must be [batch, sequence, features]")
    })?;
    let attention_mask = Tensor::ones((batch, sequence), DType::U8, &device)?;

    let conditional = crate::ltx2::text::connectors::EmbeddingsProcessorOutput {
        video_encoding,
        audio_encoding,
        attention_mask,
    };
    Ok(NativePromptEncoding {
        unconditional: conditional.clone(),
        conditional,
    })
}

fn prompt_requires_unconditional_context(plan: &Ltx2GeneratePlan) -> Result<bool> {
    if ltx_debug_enabled() || ltx_debug_compare_uncond_enabled() {
        return Ok(true);
    }
    prompt_requires_unconditional_context_for_plan(plan)
}

fn prompt_requires_unconditional_context_for_plan(plan: &Ltx2GeneratePlan) -> Result<bool> {
    for stage_index in 0..plan.execution_graph.denoise_passes.len() {
        if stage_requires_unconditional_context(plan, stage_index)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn stage_requires_unconditional_context(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Result<bool> {
    if stage_guidance_scale(plan, stage_index)? > 1.0 {
        return Ok(true);
    }
    Ok(
        stage_multimodal_guider_params(plan, stage_index)?.is_some_and(
            |(video_params, audio_params)| {
                multimodal_guider_requires_unconditional_context(&video_params)
                    || multimodal_guider_requires_unconditional_context(&audio_params)
            },
        ),
    )
}

fn stage_distilled_lora_scale(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<Option<f64>> {
    let pass = denoise_pass_plan(plan, stage_index)?;
    Ok(match (plan.pipeline, stage_index) {
        (PipelineKind::TwoStageHq, 0) => Some(0.25),
        (PipelineKind::TwoStageHq, 1) => Some(0.5),
        _ if pass.apply_distilled_lora && !plan.checkpoint_is_distilled => Some(1.0),
        _ => None,
    })
}

/// Whether the native runtime can render this plan for real, as opposed to
/// falling back to `render_native_video`'s synthetic placeholder frames.
///
/// This is a statement about *conditioning shapes*, never about LoRAs. Every
/// renderer threads `stage_lora_stack` into its transformer loads, so a plan
/// carrying user LoRAs — which is what a camera-motion preset is, once the
/// server resolves `camera-control:<id>` to a path — renders normally. The
/// one exception is `native_ic_lora`, where a LoRA is *required* rather than
/// merely tolerated.
fn supports_real_video_path(plan: &Ltx2GeneratePlan) -> bool {
    let native_plain_or_image_conditioning = plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_none()
        && !plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_retake_masking;
    let native_audio_conditioning = plan.conditioning.audio_path.is_some()
        && plan.conditioning.video_path.is_none()
        && plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_retake_masking
        && plan.spatial_upscale.is_none();
    let native_retake = plan.conditioning.video_path.is_some()
        && plan.execution_graph.uses_retake_masking
        && plan.spatial_upscale.is_none()
        && plan.temporal_upscale.is_none();
    let native_ic_lora = plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_some()
        && plan.execution_graph.uses_reference_video_conditioning
        && !plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_retake_masking
        && !plan.loras.is_empty()
        && plan.spatial_upscale.is_none();
    // Lip dub is the one pipeline that conditions on audio it was never handed:
    // the reference clip supplies both the pixels and the voice, so a separate
    // `audio_path` is a contradiction rather than a requirement.
    let native_lip_dub = plan.conditioning.audio_path.is_none()
        && plan.conditioning.video_path.is_some()
        && plan.execution_graph.uses_reference_video_conditioning
        && plan.execution_graph.uses_audio_conditioning
        && !plan.execution_graph.uses_retake_masking
        && !plan.loras.is_empty()
        && plan.spatial_upscale.is_none()
        && plan.temporal_upscale.is_none();
    match plan.pipeline {
        PipelineKind::Distilled => native_plain_or_image_conditioning,
        PipelineKind::OneStage => {
            native_plain_or_image_conditioning
                && plan.spatial_upscale.is_none()
                && plan.temporal_upscale.is_none()
        }
        PipelineKind::TwoStage | PipelineKind::TwoStageHq | PipelineKind::Keyframe => {
            native_plain_or_image_conditioning
        }
        PipelineKind::A2Vid => native_audio_conditioning,
        PipelineKind::IcLora => native_ic_lora,
        PipelineKind::LipDub => native_lip_dub,
        PipelineKind::Retake => native_retake,
        // T2A produces no frames at all. It never reaches the video renderer —
        // `render_native_video` rejects it up-front and callers route to
        // `render_native_audio` instead.
        PipelineKind::T2a => false,
    }
}

fn denoise_pass_plan(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
) -> Result<&crate::ltx2::execution::DenoisePassPlan> {
    plan.execution_graph
        .denoise_passes
        .get(stage_index)
        .with_context(|| {
            format!(
                "missing LTX-2 denoise pass plan for stage {}",
                stage_index + 1
            )
        })
}

/// What the second denoise pass does with the audio the first one produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage2AudioPolicy {
    /// Re-noise it to the stage-2 sigma and denoise it again alongside the
    /// upscaled video. Every two-stage pipeline except lip dub.
    Refine,
    /// Carry it through untouched and export *it* rather than stage 2's copy.
    ///
    /// Lip dub's second pass exists to sharpen the picture. Upstream marks its
    /// audio `frozen=True, noise_scale=0.0,
    /// initial_latent=s1_audio_latent` (`lipdub.py:283-289`) and decodes the
    /// stage-1 latent for the soundtrack (`lipdub.py:293`) — re-denoising a
    /// finished dub can only move it away from the mouth already rendered
    /// against it.
    Frozen,
}

fn stage2_audio_policy(pipeline: PipelineKind) -> Stage2AudioPolicy {
    match pipeline {
        PipelineKind::LipDub => Stage2AudioPolicy::Frozen,
        _ => Stage2AudioPolicy::Refine,
    }
}

/// Whether this pipeline appends reference speech as negatively positioned
/// audio tokens, the way lip dub conditions on the voice it is imitating.
fn appends_audio_reference(pipeline: PipelineKind) -> bool {
    matches!(pipeline, PipelineKind::LipDub)
}

fn stage_lora_stack(plan: &Ltx2GeneratePlan, stage_index: usize) -> Result<Vec<LoraWeight>> {
    // Generic IC-LoRA builds its second stage with `loras=()`
    // (`ic_lora.py:99-103`) — the adapter has done its job by then. Lip dub
    // deliberately does not: one `DiffusionStage` carrying the adapter runs
    // both passes (`lipdub.py:96-106`), because stage 2 is still re-timing a
    // mouth and would drift off the reference without it.
    if matches!(plan.pipeline, PipelineKind::IcLora) && stage_index > 0 {
        return Ok(Vec::new());
    }
    let mut loras = plan.loras.clone();
    if let Some(scale) = stage_distilled_lora_scale(plan, stage_index)? {
        let path = plan
            .distilled_lora_path
            .clone()
            .context("native LTX-2 two-stage runtime requires a distilled LoRA asset")?;
        loras.push(LoraWeight {
            path,
            scale,
            expert: None,
        });
    }
    Ok(loras)
}

fn stage_sigmas_no_terminal(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
    device: &candle_core::Device,
) -> Result<Vec<f32>> {
    let pass = denoise_pass_plan(plan, stage_index)?;
    if stage_index == 1
        && matches!(
            plan.pipeline,
            PipelineKind::TwoStage | PipelineKind::TwoStageHq
        )
    {
        return Ok(distilled_stage2_sigmas_no_terminal(plan).to_vec());
    }
    if pass.uses_distilled_checkpoint {
        return Ok(match stage_index {
            0 => DISTILLED_STAGE1_SIGMAS_NO_TERMINAL.to_vec(),
            1 => distilled_stage2_sigmas_no_terminal(plan).to_vec(),
            _ => anyhow::bail!("unsupported distilled denoise stage {}", stage_index + 1),
        });
    }

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(ltx2_scheduler_config())?;
    scheduler.set_timesteps(
        Some(plan.num_inference_steps as usize),
        device,
        None,
        None,
        None,
    )?;
    let sigmas = scheduler.sigmas().to_device(&candle_core::Device::Cpu)?;
    let sigmas = sigmas.to_vec1::<f32>()?;
    Ok(sigmas[..sigmas.len().saturating_sub(1)].to_vec())
}

#[derive(Debug, Clone)]
struct StageRenderContext {
    #[allow(dead_code)]
    stage_index: usize,
    guidance_scale: f64,
    sampler_mode: SamplerMode,
    sigmas_no_terminal: Vec<f32>,
    loras: Vec<LoraWeight>,
    multimodal_guidance: Option<(MultiModalGuiderParams, MultiModalGuiderParams)>,
    requires_unconditional_context: bool,
}

fn prepare_stage_context(
    plan: &Ltx2GeneratePlan,
    stage_index: usize,
    device: &candle_core::Device,
) -> Result<StageRenderContext> {
    Ok(StageRenderContext {
        stage_index,
        guidance_scale: stage_guidance_scale(plan, stage_index)?,
        sampler_mode: stage_sampler_mode(plan, stage_index)?,
        sigmas_no_terminal: stage_sigmas_no_terminal(plan, stage_index, device)?,
        loras: stage_lora_stack(plan, stage_index)?,
        multimodal_guidance: stage_multimodal_guider_params(plan, stage_index)?,
        requires_unconditional_context: stage_requires_unconditional_context(plan, stage_index)?,
    })
}

fn video_latent_shape_from_tensor(latents: &Tensor) -> Result<VideoLatentShape> {
    let (batch, channels, frames, height, width) = latents.dims5()?;
    Ok(VideoLatentShape {
        batch,
        channels,
        frames,
        height,
        width,
    })
}

fn pixel_shape_for_video_latents(latent_shape: VideoLatentShape, fps: u32) -> VideoPixelShape {
    let pixel_shape = latent_shape.upscale(SpatioTemporalScaleFactors::default());
    VideoPixelShape {
        batch: pixel_shape.batch,
        frames: pixel_shape.frames,
        height: pixel_shape.height,
        width: pixel_shape.width,
        fps: fps as f32,
    }
}

#[allow(dead_code)]
fn source_image_only_conditioning(plan: &Ltx2GeneratePlan) -> bool {
    matches!(plan.conditioning.images.as_slice(), [image] if image.frame == 0)
        && !plan.execution_graph.uses_keyframe_conditioning
}

#[allow(dead_code)]
fn keyframe_only_conditioning(plan: &Ltx2GeneratePlan) -> bool {
    !plan.conditioning.images.is_empty()
        && plan.conditioning.images.iter().all(|image| image.frame > 0)
        && plan.execution_graph.uses_keyframe_conditioning
}

fn offset_video_time_positions(pixel_coords: &Tensor, frame_offset: u32) -> Result<Tensor> {
    let temporal = pixel_coords
        .i((.., 0..1, .., ..))?
        .affine(1.0, frame_offset as f64)?;
    let height_width = pixel_coords.i((.., 1.., .., ..))?;
    Tensor::cat(&[temporal, height_width], 1).map_err(Into::into)
}

fn scale_video_spatial_positions(positions: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 1 {
        return Ok(positions.clone());
    }
    let temporal = positions.i((.., 0..1, .., ..))?;
    let height = positions
        .i((.., 1..2, .., ..))?
        .affine(factor as f64, 0.0)?;
    let width = positions
        .i((.., 2..3, .., ..))?
        .affine(factor as f64, 0.0)?;
    Tensor::cat(&[temporal, height, width], 1).map_err(Into::into)
}

fn append_condition_from_video_latents(
    latents: &Tensor,
    pixel_shape: VideoPixelShape,
    frame_offset: u32,
    spatial_position_scale: usize,
    strength: f64,
) -> Result<VideoTokenAppendCondition> {
    let patchifier = VideoLatentPatchifier::new(1);
    let tokens = patchifier.patchify(&latents.to_dtype(DType::F32)?)?;
    let latent_shape = video_latent_shape_from_tensor(latents)?;
    let latent_coords = patchifier.get_patch_grid_bounds(latent_shape, latents.device())?;
    let pixel_coords =
        get_pixel_coords(&latent_coords, SpatioTemporalScaleFactors::default(), true)?;
    let positions = scale_video_spatial_positions(
        &scale_video_time_to_seconds(
            &offset_video_time_positions(&pixel_coords, frame_offset)?,
            pixel_shape.fps,
        )?,
        spatial_position_scale,
    )?
    .to_dtype(DType::F32)?;
    Ok(VideoTokenAppendCondition {
        tokens,
        positions,
        strength,
        latent_grid: (latent_shape.frames, latent_shape.height, latent_shape.width),
        spatial_downscale_factor: spatial_position_scale,
    })
}

/// Whether a stage has any visual conditioning to ingest at all.
///
/// Also the probe predicate: a run with no conditioning does no GPU work here
/// and must not emit a phase line claiming it did.
fn stage_conditioning_is_empty(plan: &Ltx2GeneratePlan, include_reference_video: bool) -> bool {
    plan.conditioning.images.is_empty()
        && plan.conditioning.latents.is_empty()
        && !include_reference_video
}

fn maybe_load_stage_video_conditioning(
    plan: &Ltx2GeneratePlan,
    pixel_shape: VideoPixelShape,
    device: &candle_core::Device,
    dtype: DType,
    include_reference_video: bool,
    progress: Option<&ProgressCallback>,
) -> Result<StageVideoConditioning> {
    if stage_conditioning_is_empty(plan, include_reference_video) {
        return Ok(StageVideoConditioning::default());
    }
    // The image-to-video source encode: a VAE load plus one encode pass, which
    // is where a conditioned run first outgrows an unconditioned one.
    let probe = PhaseVramProbe::enter_if(
        format!(
            "conditioning_encode[{}x{}x{}]",
            pixel_shape.width, pixel_shape.height, pixel_shape.frames
        ),
        device.is_cuda(),
    );
    let result = maybe_load_stage_video_conditioning_inner(
        plan,
        pixel_shape,
        device,
        dtype,
        include_reference_video,
        progress,
    );
    log_ltx2_phase_vram_result(probe.finish(), &result, None, "");
    result
}

fn maybe_load_stage_video_conditioning_inner(
    plan: &Ltx2GeneratePlan,
    pixel_shape: VideoPixelShape,
    device: &candle_core::Device,
    dtype: DType,
    include_reference_video: bool,
    progress: Option<&ProgressCallback>,
) -> Result<StageVideoConditioning> {
    if stage_conditioning_is_empty(plan, include_reference_video) {
        return Ok(StageVideoConditioning::default());
    }

    // The VAE is needed for staged images, reference video ingest, and —
    // on chain continuations — re-encoding the emitting stage's trailing
    // RGB frames into a proper-slot-semantics conditioning latent. Every
    // StagedLatent now carries RGB frames, so any non-empty
    // plan.conditioning.latents implies a VAE load.
    let need_vae = !plan.conditioning.images.is_empty()
        || include_reference_video
        || !plan.conditioning.latents.is_empty();
    let mut vae = if need_vae {
        Some(load_ltx2_video_vae(plan, device, dtype, progress)?)
    } else {
        None
    };

    let patchifier = VideoLatentPatchifier::new(1);
    let mut conditioning = StageVideoConditioning::default();
    for image in &plan.conditioning.images {
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever plan.conditioning.images is non-empty",
        );
        // Upstream parity (#1055): oriented sRGB decode + the checkpoint
        // generation's H.264 round-trip happen once at native resolution,
        // then each stage fits the same preprocessed image to its own
        // conditioning canvas. Materialization guarantees the profile is
        // resolved whenever images are staged.
        let profile = plan.image_preprocessing.with_context(|| {
            format!(
                "staged LTX-2 conditioning image '{}' has no preprocessing profile; \
                 materialization must resolve the checkpoint generation first",
                image.path
            )
        })?;
        let native = preprocess::cached_native_conditioning_image(&image.path, &profile)?;
        let decoded = preprocess::fit_conditioning_image(
            &native,
            pixel_shape.width as u32,
            pixel_shape.height as u32,
            device,
            dtype,
        )?;
        let video = decoded.unsqueeze(2)?;
        let latents = vae.encode(&video).with_context(|| {
            format!(
                "failed to encode native LTX-2 conditioning image '{}'",
                image.path
            )
        })?;
        let tokens = patchifier.patchify(&latents.to_dtype(DType::F32)?)?;
        let use_guiding_latent = matches!(plan.pipeline, PipelineKind::Keyframe);
        if image.frame == 0 && !use_guiding_latent {
            conditioning.replacements.push(VideoTokenReplacement {
                start_token: 0,
                tokens,
                strength: image.strength as f64,
            });
        } else {
            conditioning
                .appended
                .push(append_condition_from_video_latents(
                    &latents,
                    pixel_shape,
                    image.frame,
                    1,
                    image.strength as f64,
                )?);
        }
    }
    // Chain carryover: every StagedLatent is a contiguous RGB window from
    // the end of the emitting stage. Re-encoding on the receiving side
    // (rather than slicing the emitting stage's final latent tensor) keeps
    // slot semantics aligned with the receiving clip's time axis — slot 0
    // is a proper causal 1-pixel encoding, slot 1+ are proper 8-pixel
    // continuation encodings, with no ambiguity about which latent slot
    // corresponds to which pixel-frame range.
    for staged in &plan.conditioning.latents {
        if staged.tail_rgb_frames.is_empty() {
            anyhow::bail!(
                "StagedLatent has an empty tail_rgb_frames; at least one frame is required"
            );
        }
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever plan.conditioning.latents is non-empty",
        );
        // Tail frames arrive at the emitting stage's final decoded resolution
        // (chain.width × chain.height). The distilled pipeline's stage 1 runs
        // at a reduced resolution via `derive_stage1_render_shape`'s implicit
        // X2 downsample, so `pixel_shape` here can be half the tail's pixel
        // dims. VAE-encoding at the tail's native size would produce latents
        // at the wrong spatial grid for stage 1 — the replacement's token
        // count would mismatch (typically exceed) the stage 1 total. Resize
        // to `pixel_shape` first so stage 1 and stage 2 each see tokens on
        // their own grid. No-op when dims already match (stage 2 case).
        let resized_frames = resize_tail_frames_to_pixel_shape(
            &staged.tail_rgb_frames,
            pixel_shape.width as u32,
            pixel_shape.height as u32,
        );
        let video = video_tensor_from_frames(&resized_frames, device, dtype)
            .context("encode chain tail RGB frames into pixel tensor for carryover")?;
        let latents = vae
            .encode(&video)
            .context("failed to encode chain tail RGB frames through the LTX-2 video VAE")?
            .to_dtype(DType::F32)?;
        let use_guiding_latent = matches!(plan.pipeline, PipelineKind::Keyframe);
        if staged.frame == 0 && !use_guiding_latent {
            let tokens = patchifier.patchify(&latents)?;
            conditioning.replacements.push(VideoTokenReplacement {
                start_token: 0,
                tokens,
                strength: staged.strength as f64,
            });
        } else {
            conditioning
                .appended
                .push(append_condition_from_video_latents(
                    &latents,
                    pixel_shape,
                    staged.frame,
                    1,
                    staged.strength as f64,
                )?);
        }
    }
    if include_reference_video {
        let vae = vae.as_mut().expect(
            "need_vae guarantees the VAE is loaded whenever include_reference_video is true",
        );
        let video_path = plan.conditioning.video_path.as_ref().with_context(|| {
            format!(
                "native {:?} stage requested reference video conditioning without a staged source_video",
                plan.pipeline
            )
        })?;
        let reference_downscale_factor = lora::reference_video_downscale_factor(&plan.loras)?;
        if !pixel_shape.width.is_multiple_of(reference_downscale_factor)
            || !pixel_shape
                .height
                .is_multiple_of(reference_downscale_factor)
        {
            anyhow::bail!(
                "native LTX-2 IC-LoRA output dimensions ({}x{}) must be divisible by reference_downscale_factor ({reference_downscale_factor})",
                pixel_shape.width,
                pixel_shape.height
            );
        }
        let ref_width = pixel_shape.width / reference_downscale_factor;
        let ref_height = pixel_shape.height / reference_downscale_factor;
        let (_metadata, decoded_frames) = media::decode_video_frames(Path::new(video_path))?;
        // A chain stage conditions on its own temporal window of the shared
        // reference: skip everything before the stage's stitched-timeline
        // start, then cap at the render length. Zero for single renders, so
        // the existing behaviour is the identity case. Upstream slices
        // reference conditioning per temporal tile the same way
        // (hdr_ic_lora.py:521-541).
        let offset = plan.reference_frame_offset as usize;
        if offset > 0 && decoded_frames.len() <= offset {
            anyhow::bail!(
                "reference video '{video_path}' has {} frames but this chain stage begins at \
                 stitched frame {offset}; the reference must cover the full requested duration",
                decoded_frames.len(),
            );
        }
        let mut frames: Vec<RgbImage> = decoded_frames.into_iter().skip(offset).collect();
        if frames.len() > pixel_shape.frames {
            frames.truncate(pixel_shape.frames);
        }
        let resized = frames
            .into_iter()
            .map(|frame| {
                if frame.width() == ref_width as u32 && frame.height() == ref_height as u32 {
                    frame
                } else {
                    imageops::resize(
                        &frame,
                        ref_width as u32,
                        ref_height as u32,
                        imageops::FilterType::Lanczos3,
                    )
                }
            })
            .collect::<Vec<_>>();
        let video = video_tensor_from_frames(&resized, device, dtype)?;
        let latents = vae.encode(&video).with_context(|| {
            format!(
                "failed to encode native LTX-2 IC-LoRA reference video '{}'",
                video_path
            )
        })?;
        conditioning
            .appended
            .push(append_condition_from_video_latents(
                &latents,
                pixel_shape,
                0,
                reference_downscale_factor,
                1.0,
            )?);
    }
    drop(vae);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(conditioning)
}

fn apply_video_token_replacements(
    video_latents: &Tensor,
    replacements: &[VideoTokenReplacement],
) -> Result<Tensor> {
    let mut patched = video_latents.clone();
    for replacement in replacements {
        let total_tokens = patched.dim(1)?;
        let replacement_tokens = replacement
            .tokens
            .to_device(patched.device())?
            .to_dtype(patched.dtype())?;
        let count = replacement_tokens.dim(1)?;
        if replacement.start_token + count > total_tokens {
            anyhow::bail!(
                "conditioning replacement exceeds video token count: start={} count={} total={total_tokens}",
                replacement.start_token,
                count
            );
        }
        let current = patched.narrow(1, replacement.start_token, count)?;
        let blended = if replacement.strength <= 0.0 {
            current
        } else if replacement.strength >= 1.0 {
            replacement_tokens
        } else {
            current
                .affine(1.0 - replacement.strength, 0.0)?
                .broadcast_add(&replacement_tokens.affine(replacement.strength, 0.0)?)?
        };
        let mut parts = Vec::with_capacity(3);
        if replacement.start_token != 0 {
            parts.push(patched.narrow(1, 0, replacement.start_token)?);
        }
        parts.push(blended);
        let end = replacement.start_token + count;
        if end < total_tokens {
            parts.push(patched.narrow(1, end, total_tokens - end)?);
        }
        let refs = parts.iter().collect::<Vec<_>>();
        patched = Tensor::cat(&refs, 1)?;
    }
    Ok(patched)
}

/// Build the "clean reference" tensor used by the denoise mask blend at every
/// step. For replacement-based conditioning (e.g. i2v source image) with
/// `strength < 1.0`, `video_latents` already holds `noise*(1-s) + source*s` at
/// the replacement positions. If we reuse that as the clean target, the
/// denoise-mask blend pulls those tokens toward a noisy ghost of the image at
/// every step — the first latent frame never converges to the pure source.
///
/// Re-applying the replacements with strength 1.0 overwrites those positions
/// with the pure source tokens. Appended conditions likewise use their clean
/// tokens here even when their initial latent is softly noised (#1080).
fn clean_latents_for_conditioning(
    video_latents: &Tensor,
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
) -> Result<Tensor> {
    let hard_replacements: Vec<VideoTokenReplacement> = conditioning
        .replacements
        .iter()
        .map(|replacement| VideoTokenReplacement {
            start_token: replacement.start_token,
            tokens: replacement.tokens.clone(),
            strength: 1.0,
        })
        .collect();
    let base = video_latents.narrow(1, 0, base_token_count)?;
    let base = apply_video_token_replacements(&base, &hard_replacements)?;
    if conditioning.appended.is_empty() {
        return Ok(base);
    }

    let mut parts = vec![base];
    for condition in &conditioning.appended {
        parts.push(
            condition
                .tokens
                .to_device(video_latents.device())?
                .to_dtype(video_latents.dtype())?,
        );
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 1).map_err(Into::into)
}

fn apply_appended_video_conditioning(
    video_latents: &Tensor,
    video_positions: &Tensor,
    appended: &[VideoTokenAppendCondition],
    noise_seed: u64,
) -> Result<(Tensor, Tensor)> {
    if appended.is_empty() {
        return Ok((video_latents.clone(), video_positions.clone()));
    }

    let mut soft_appended_token_count = 0;
    for condition in appended.iter().filter(|condition| condition.strength < 1.0) {
        soft_appended_token_count += condition.tokens.dim(1)?;
    }
    let (batch, _, channels) = video_latents.dims3()?;
    let appended_noise = if soft_appended_token_count == 0 {
        None
    } else {
        Some(seeded_randn(
            noise_seed ^ APPENDED_VIDEO_NOISE_SALT,
            &[batch, soft_appended_token_count, channels],
            video_latents.device(),
            video_latents.dtype(),
        )?)
    };

    let mut token_parts = vec![video_latents.clone()];
    let mut position_parts = vec![video_positions.clone()];
    let mut noise_offset = 0;
    for condition in appended {
        let clean = condition
            .tokens
            .to_device(video_latents.device())?
            .to_dtype(video_latents.dtype())?;
        let token_count = clean.dim(1)?;
        let tokens = if condition.strength >= 1.0 {
            clean
        } else {
            let noise = appended_noise
                .as_ref()
                .context("soft appended video conditioning requires a noise stream")?
                .narrow(1, noise_offset, token_count)?;
            noise_offset += token_count;
            if condition.strength <= 0.0 {
                noise
            } else {
                noise
                    .affine(1.0 - condition.strength, 0.0)?
                    .broadcast_add(&clean.affine(condition.strength, 0.0)?)?
            }
        };
        token_parts.push(tokens);
        position_parts.push(
            condition
                .positions
                .to_device(video_positions.device())?
                .to_dtype(video_positions.dtype())?,
        );
    }
    let token_refs = token_parts.iter().collect::<Vec<_>>();
    let position_refs = position_parts.iter().collect::<Vec<_>>();
    Ok((
        Tensor::cat(&token_refs, 1)?,
        Tensor::cat(&position_refs, 2)?,
    ))
}

fn apply_stage_video_conditioning(
    video_latents: &Tensor,
    video_positions: &Tensor,
    conditioning: &StageVideoConditioning,
    appended_noise_seed: u64,
) -> Result<(Tensor, Tensor)> {
    let replaced = apply_video_token_replacements(video_latents, &conditioning.replacements)?;
    apply_appended_video_conditioning(
        &replaced,
        video_positions,
        &conditioning.appended,
        appended_noise_seed,
    )
}

fn reapply_stage_video_conditioning(
    video_latents: &Tensor,
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
) -> Result<Tensor> {
    let total_tokens = video_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "video token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }

    let base = video_latents.narrow(1, 0, base_token_count)?;
    let hard_replacements = conditioning
        .replacements
        .iter()
        .filter(|replacement| replacement.strength >= 1.0)
        .cloned()
        .collect::<Vec<_>>();
    let base = apply_video_token_replacements(&base, &hard_replacements)?;
    if conditioning.appended.is_empty() {
        return Ok(base);
    }

    let mut parts = vec![base];
    let mut appended_offset = base_token_count;
    for condition in &conditioning.appended {
        // Appended conditioning tokens must remain present for the whole
        // denoise loop. Hard tokens stay pinned to their source; soft tokens
        // keep the sampler's evolved slice. Dropping either kind would
        // desynchronize the cached clean latents, mask, and positions.
        let token_count = condition.tokens.dim(1)?;
        let tokens = if condition.strength >= 1.0 {
            condition
                .tokens
                .to_device(video_latents.device())?
                .to_dtype(video_latents.dtype())?
        } else {
            video_latents.narrow(1, appended_offset, token_count)?
        };
        parts.push(tokens);
        appended_offset += token_count;
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 1).map_err(Into::into)
}

fn strip_appended_video_conditioning(
    video_latents: &Tensor,
    base_token_count: usize,
) -> Result<Tensor> {
    let total_tokens = video_latents.dim(1)?;
    if total_tokens < base_token_count {
        anyhow::bail!(
            "video token count ({total_tokens}) is smaller than base token count ({base_token_count})"
        );
    }
    if total_tokens == base_token_count {
        return Ok(video_latents.clone());
    }
    video_latents
        .narrow(1, 0, base_token_count)
        .map_err(Into::into)
}

fn build_video_conditioning_denoise_mask(
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut values = vec![1.0f32; base_token_count];
    for replacement in &conditioning.replacements {
        let count = replacement.tokens.dim(1)?;
        let end = replacement.start_token + count;
        if end > base_token_count {
            anyhow::bail!(
                "conditioning replacement exceeds base token count: start={} count={} total={base_token_count}",
                replacement.start_token,
                count
            );
        }
        values[replacement.start_token..end].fill((1.0 - replacement.strength) as f32);
    }
    for condition in &conditioning.appended {
        values.extend(std::iter::repeat_n(
            (1.0 - condition.strength) as f32,
            condition.tokens.dim(1)?,
        ));
    }
    Tensor::from_vec(values.clone(), (1, values.len()), device).map_err(Into::into)
}

fn append_conditioning_attention_mask(
    existing_mask: Option<&Tensor>,
    num_noisy_tokens: usize,
    num_existing_tokens: usize,
    num_new_tokens: usize,
    batch_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let top_left = match existing_mask {
        Some(mask) => mask.to_device(device)?.to_dtype(DType::F32)?,
        None => Tensor::ones(
            (batch_size, num_existing_tokens, num_existing_tokens),
            DType::F32,
            device,
        )?,
    };
    let previous_ref_tokens = num_existing_tokens.saturating_sub(num_noisy_tokens);
    let noisy_to_new = Tensor::ones(
        (batch_size, num_noisy_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;
    let prev_ref_to_new = Tensor::zeros(
        (batch_size, previous_ref_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;
    let top_right = Tensor::cat(&[&noisy_to_new, &prev_ref_to_new], 1)?;

    let new_to_noisy = Tensor::ones(
        (batch_size, num_new_tokens, num_noisy_tokens),
        DType::F32,
        device,
    )?;
    let new_to_prev_ref = Tensor::zeros(
        (batch_size, num_new_tokens, previous_ref_tokens),
        DType::F32,
        device,
    )?;
    let bottom_left = Tensor::cat(&[&new_to_noisy, &new_to_prev_ref], 2)?;
    let bottom_right = Tensor::ones(
        (batch_size, num_new_tokens, num_new_tokens),
        DType::F32,
        device,
    )?;

    let top = Tensor::cat(&[&top_left, &top_right], 2)?;
    let bottom = Tensor::cat(&[&bottom_left, &bottom_right], 2)?;
    Tensor::cat(&[&top, &bottom], 1).map_err(Into::into)
}

/// Denoise mask over `base + appended` audio tokens.
///
/// `base_denoise` is the mask the stage wants for the audio it is generating —
/// all ones on stage 1, all zeros on stage 2 where the audio is frozen. The
/// appended reference tokens are always pinned at their conditioning strength,
/// exactly like their video counterparts.
fn build_audio_conditioning_denoise_mask(
    audio_shape: AudioLatentShape,
    base_denoise: Option<&Tensor>,
    conditioning: &StageAudioConditioning,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if conditioning.is_empty() {
        return base_denoise
            .map(|mask| mask.to_device(device)?.to_dtype(DType::F32))
            .transpose()
            .map_err(Into::into);
    }
    let base = match base_denoise {
        Some(mask) => mask.to_device(device)?.to_dtype(DType::F32)?,
        None => Tensor::ones((audio_shape.batch, audio_shape.frames), DType::F32, device)?,
    };
    let mut values = Vec::with_capacity(conditioning.appended_token_count()?);
    for condition in &conditioning.appended {
        values.extend(std::iter::repeat_n(
            (1.0 - condition.strength) as f32,
            condition.tokens.dim(1)?,
        ));
    }
    let appended_tokens = values.len();
    let appended = Tensor::from_vec(values, (1, appended_tokens), device)?
        .broadcast_as((audio_shape.batch, appended_tokens))?
        .contiguous()?;
    Ok(Some(Tensor::cat(&[&base, &appended], 1)?))
}

fn build_audio_conditioning_self_attention_mask(
    base_token_count: usize,
    conditioning: &StageAudioConditioning,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if conditioning.appended.is_empty() {
        return Ok(None);
    }
    let batch_size = conditioning
        .appended
        .first()
        .context("appended audio conditioning unexpectedly empty")?
        .tokens
        .dim(0)?;
    let mut existing_mask = None;
    let mut existing_tokens = base_token_count;
    for condition in &conditioning.appended {
        existing_mask = Some(append_conditioning_attention_mask(
            existing_mask.as_ref(),
            base_token_count,
            existing_tokens,
            condition.tokens.dim(1)?,
            batch_size,
            device,
        )?);
        existing_tokens += condition.tokens.dim(1)?;
    }
    Ok(existing_mask)
}

fn build_video_conditioning_self_attention_mask(
    base_token_count: usize,
    conditioning: &StageVideoConditioning,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if conditioning.appended.is_empty() {
        return Ok(None);
    }
    let batch_size = conditioning
        .appended
        .first()
        .context("appended conditioning unexpectedly empty")?
        .tokens
        .dim(0)?;
    let mut existing_mask = None;
    let mut existing_tokens = base_token_count;
    for condition in &conditioning.appended {
        existing_mask = Some(append_conditioning_attention_mask(
            existing_mask.as_ref(),
            base_token_count,
            existing_tokens,
            condition.tokens.dim(1)?,
            batch_size,
            device,
        )?);
        existing_tokens += condition.tokens.dim(1)?;
    }
    Ok(existing_mask)
}

fn maybe_apply_temporal_upsampler(
    plan: &Ltx2GeneratePlan,
    latents: &Tensor,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    if plan.temporal_upscale.is_none() {
        return Ok(latents.clone());
    }
    let temporal_upsampler_path = plan
        .temporal_upsampler_path
        .as_ref()
        .context("native LTX-2 temporal upscaling requires a temporal upsampler asset")?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let upsampler = LatentUpsampler::load(Path::new(temporal_upsampler_path), dtype, device)?;
    let upsampled = latent_stats
        .normalize(&upsampler.forward(&latent_stats.denormalize(&latents.to_dtype(dtype)?)?)?)?;
    drop(upsampler);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(upsampled)
}

/// The video-modality state entering the denoise loop: conditioned start
/// latents, widened positions, the per-step clean target, and the denoise
/// mask.
struct VideoStageInit {
    latents: Tensor,
    positions: Tensor,
    clean: Tensor,
    denoise_mask: Tensor,
}

/// Keep appended conditioning noise independent from the main video latent
/// stream while remaining deterministic across CPU, Metal, and CUDA.
const APPENDED_VIDEO_NOISE_SALT: u64 = 0x4150_5045_4e44_4c54;

/// Build the initial conditioned video latents exactly once (#1055).
///
/// Upstream constructs the noisy state with a single lerp
/// (`noisers.py:32-33` @ fd4ded7): a conditioned token starts as
/// `s·C + (1-s)·N` and its per-token timestep says `(1-s)·σ` — the two
/// must agree. `apply_stage_video_conditioning` performs that blend for soft
/// replacements and soft appended tokens, so re-blending the
/// result toward the clean target — which this function's pre-fix caller
/// did for every conditioned render — double-counted the source:
/// `(2s-s²)·C + (1-s)²·N` under a timestep still claiming `(1-s)` noise.
/// That latent/timestep mismatch suppressed I2V motion at soft strengths.
///
/// The freeze blend itself is still upstream's lerp and remains
/// load-bearing when the caller supplies an *explicit* clean tensor over
/// pure-noise start latents (retake's temporal window; lip-dub's frozen
/// audio mirrors this on the audio side): there the blend is what installs
/// the source into the frozen (mask = 0) regions, and it runs exactly
/// once. For conditioning-derived cleans it is removed rather than kept:
/// hard replacements and hard appended tokens already equal their clean
/// values (identity blend), while soft conditioning is already correctly
/// noised once.
fn initialize_video_stage_latents(
    patched_start_latents: &Tensor,
    video_positions: &Tensor,
    conditioning: &StageVideoConditioning,
    appended_noise_seed: u64,
    clean_override: Option<&Tensor>,
    mask_override: Option<&Tensor>,
    base_token_count: usize,
    device: &candle_core::Device,
) -> Result<VideoStageInit> {
    let (latents, positions) = apply_stage_video_conditioning(
        patched_start_latents,
        video_positions,
        conditioning,
        appended_noise_seed,
    )?;
    let clean = match clean_override {
        Some(clean) => clean.clone(),
        None => clean_latents_for_conditioning(&latents, base_token_count, conditioning)?,
    };
    let denoise_mask = match mask_override {
        Some(mask) => mask.to_device(device)?.to_dtype(DType::F32)?,
        None => build_video_conditioning_denoise_mask(base_token_count, conditioning, device)?,
    };
    let latents = match clean_override {
        Some(_) => blend_conditioned_denoised(&latents, &clean, &denoise_mask)?,
        None => latents,
    };
    Ok(VideoStageInit {
        latents,
        positions,
        clean,
        denoise_mask,
    })
}

fn blend_conditioned_denoised(
    denoised: &Tensor,
    clean_latents: &Tensor,
    denoise_mask: &Tensor,
) -> Result<Tensor> {
    let mask = denoise_mask
        .to_device(denoised.device())?
        .to_dtype(denoised.dtype())?;
    let mask = mask.unsqueeze(2)?;
    let clean = clean_latents
        .to_device(denoised.device())?
        .to_dtype(denoised.dtype())?;
    let inverse = Tensor::ones_like(&mask)?.broadcast_sub(&mask)?;
    denoised
        .broadcast_mul(&mask)?
        .broadcast_add(&clean.broadcast_mul(&inverse)?)
        .map_err(Into::into)
}

/// Whether the checkpoint carries no weights at all — absent, not a regular
/// file, or present but zero-length.
///
/// Zero-length is the signature of a download that never landed, and it is
/// what the unit tests write as a stand-in for real weights. It is deliberately
/// *not* the same as a short-but-non-empty file: those bytes are a truncated
/// real checkpoint, and quietly rendering the synthetic gradient for them hands
/// the user a correctly sized video with no picture while hiding the
/// corruption.
fn checkpoint_has_no_weights(path: &Path) -> bool {
    stat_reports_no_weights(
        std::fs::metadata(path)
            .map(|metadata| (metadata.is_file(), metadata.len()))
            .map_err(|err| err.kind()),
    )
}

/// Classify a checkpoint's stat result as "no weights to read".
///
/// Split from the filesystem call so the permission and I/O cases are testable:
/// the suite runs under a mapped root user, where a `chmod`-based fixture would
/// not actually deny anything.
///
/// Only [`std::io::ErrorKind::NotFound`] means absent. A permission or
/// transient I/O error is a real failure and must fall through to the load,
/// which reports it with the checkpoint named — treating it as "no weights"
/// would hide it behind exactly the plausible-looking gradient this module
/// exists to stop producing.
fn stat_reports_no_weights(stat: std::result::Result<(bool, u64), std::io::ErrorKind>) -> bool {
    match stat {
        Ok((is_file, len)) => !is_file || len == 0,
        Err(std::io::ErrorKind::NotFound) => true,
        Err(_) => false,
    }
}

fn render_real_distilled_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
    tail_capture: Option<&std::sync::Arc<std::sync::Mutex<Option<Tensor>>>>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: false,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    // Upstream LTX-2 diffusion stages pass connector outputs directly as the
    // text context and leave `context_mask=None` in the transformer modality
    // wrapper. The connector has already packed padded tokens into registers
    // and zeroed masked positions, so feeding the binary mask back into text
    // cross-attention here over-constrains the prompt path and does not match
    // the published inference stack.
    let cond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        log_tensor_stats("video_context", &prompt_inputs.cond_context)?;
        if let Some(audio_context) = prompt_inputs.audio_context.as_ref() {
            log_tensor_stats("audio_context", audio_context)?;
        }
        log_tensor_stats("initial_video_latents", &stage1_video_noise)?;
        if let Some(stage1_audio_noise) = stage1_audio_noise.as_ref() {
            log_tensor_stats("initial_audio_latents", stage1_audio_noise)?;
        }
    }

    let dtype = compute_dtype(device);
    let stage1_guidance_scale = stage_guidance_scale(plan, 0)?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
        progress,
    )?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage1 transformer");
    }
    let stage1_transformer_load_start = Instant::now();
    let stage1_stage_shape = Ltx2StageShape::from_pixel_shape(plan, prepared.video_pixel_shape);
    let stage1_loras = stage_lora_stack(plan, 0)?;
    let stage1_transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage1_stage_shape,
        device,
        &stage1_loras,
        None,
        progress,
    )?;
    log_timing(
        "distilled.stage1.transformer_load",
        stage1_transformer_load_start,
    );
    if debug_enabled {
        log_debug_vram("after_stage1_transformer_load");
    }
    let stage1_denoise_start = Instant::now();
    let ((stage1_video_latents, stage1_audio_latents), stage1_transformer) =
        run_denoise_stage_with_oom_recovery(
            "distilled.stage1",
            stage1_transformer,
            device,
            |budget| {
                load_ltx2_av_transformer_with_loras(
                    plan,
                    stage1_stage_shape,
                    device,
                    &stage1_loras,
                    Some(budget),
                    progress,
                )
            },
            |transformer| {
                run_real_distilled_stage(
                    transformer,
                    prepared.video_latent_shape,
                    audio_shape,
                    &stage1_video_noise,
                    &stage1_video_conditioning,
                    plan.seed,
                    None,
                    stage1_audio_noise.as_ref(),
                    None,
                    &StageAudioConditioning::default(),
                    &prompt_inputs.video_positions,
                    prompt_inputs.audio_positions.as_ref(),
                    &prompt_inputs.cond_context,
                    None,
                    prompt_inputs.alt_context.as_ref(),
                    prompt_inputs.audio_context.as_ref(),
                    None,
                    prompt_inputs.alt_audio_context.as_ref(),
                    cond_mask,
                    None,
                    alt_mask,
                    None,
                    stage1_guidance_scale,
                    DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
                    stage_sampler_mode(plan, 0)?,
                    Some(&stage1_video_noise),
                    stage1_audio_noise.as_ref(),
                    None,
                    None,
                    Some("distilled.stage1"),
                    debug_enabled.then_some("stage1"),
                    progress,
                    cancellation,
                )
            },
            progress,
        )?;
    log_timing("distilled.stage1.denoise", stage1_denoise_start);
    if debug_enabled {
        log_debug_vram("after_stage1_denoise");
    }
    drop(stage1_transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_stage1_transformer_drop");
    }
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
        maybe_write_debug_stage_video(
            "stage1",
            &mut debug_vae,
            &stage1_video_latents,
            prepared.video_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let spatial_upsampler_path = plan
        .spatial_upsampler_path
        .as_ref()
        .context("native distilled LTX-2 inference requires a spatial upsampler asset")?;
    let stage1_upsample_start = Instant::now();
    let upsampler = LatentUpsampler::load(Path::new(spatial_upsampler_path), dtype, device)?;
    let stage2_clean_video_latents = latent_stats.normalize(
        &upsampler.forward(&latent_stats.denormalize(&stage1_video_latents.to_dtype(dtype)?)?)?,
    )?;
    drop(upsampler);
    device.synchronize()?;
    log_timing("distilled.stage1.spatial_upsample", stage1_upsample_start);
    if debug_enabled {
        log_debug_vram("after_stage1_upsample");
    }
    let requested_pixel_shape = VideoPixelShape {
        batch: 1,
        frames: plan.num_frames as usize,
        height: plan.height as usize,
        width: plan.width as usize,
        fps: plan.frame_rate as f32,
    };
    let stage2_video_latent_shape = video_latent_shape_from_tensor(&stage2_clean_video_latents)?;
    let stage2_pixel_shape =
        pixel_shape_for_video_latents(stage2_video_latent_shape, plan.frame_rate);
    let stage2_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        stage2_pixel_shape,
        device,
        dtype,
        false,
        progress,
    )?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
        maybe_write_debug_stage_video(
            "stage1-upscaled",
            &mut debug_vae,
            &stage2_clean_video_latents,
            stage2_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let stage2_tiles = plan_stage2_tiles(stage2_video_latent_shape)?;
    let stage2_is_tiled = stage2_tiles.len() > 1;
    let stage2_video_noise_seed = plan.seed ^ 0x5354_4147_4532_4c54;
    let stage2_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_3254_4c58,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };
    let stage2_sigmas = distilled_stage2_sigmas_no_terminal(plan);
    let stage2_sigma = stage2_sigmas[0];
    let stage2_clean_video_latents_f32 = stage2_clean_video_latents.to_dtype(DType::F32)?;
    let stage2_audio_start = match (stage1_audio_latents.as_ref(), stage2_audio_noise.as_ref()) {
        (Some(stage1_audio_latents), Some(stage2_audio_noise)) => {
            Some(mix_clean_latents_with_noise(
                &stage1_audio_latents.to_dtype(DType::F32)?,
                stage2_audio_noise,
                stage2_sigma,
            )?)
        }
        _ => None,
    };
    let stage2_audio = if stage2_is_tiled {
        Stage2AudioInputs::video_only(audio_shape.is_some())
    } else {
        Stage2AudioInputs {
            shape: audio_shape,
            start: stage2_audio_start.as_ref(),
            noise: stage2_audio_noise.as_ref(),
            positions: prompt_inputs.audio_positions.as_ref(),
            context: prompt_inputs.audio_context.as_ref(),
            alt_context: prompt_inputs.alt_audio_context.as_ref(),
            ..Stage2AudioInputs::default()
        }
    };
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage2 transformer");
    }
    let stage2_transformer_load_start = Instant::now();
    let stage2_stage_shape = Ltx2StageShape::from_pixel_shape(
        plan,
        stage2_forward_pixel_shape(&stage2_tiles, stage2_pixel_shape),
    );
    let stage2_loras = stage_lora_stack(plan, 1)?;
    let stage2_transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage2_stage_shape,
        device,
        &stage2_loras,
        None,
        progress,
    )?;
    log_timing(
        "distilled.stage2.transformer_load",
        stage2_transformer_load_start,
    );
    if debug_enabled {
        log_debug_vram("after_stage2_transformer_load");
    }
    let stage2_denoise_start = Instant::now();
    let ((latents, audio_latents), stage2_transformer) = run_denoise_stage_with_oom_recovery(
        "distilled.stage2",
        stage2_transformer,
        device,
        |budget| {
            load_ltx2_av_transformer_with_loras(
                plan,
                stage2_stage_shape,
                device,
                &stage2_loras,
                Some(budget),
                progress,
            )
        },
        |transformer| {
            let mut refined_audio = None;
            // The distilled pipeline appends no audio reference at either
            // stage; the two-stage path is where lip-dub's carry lives.
            let stage2_audio_conditioning = StageAudioConditioning::default();
            let video = TiledStage2Pass {
                tiles: &stage2_tiles,
                full_shape: stage2_video_latent_shape,
                clean_latents: &stage2_clean_video_latents_f32,
                noise_seed: stage2_video_noise_seed,
                sigma: stage2_sigma,
                fps: plan.frame_rate as f32,
                conditioning: &stage2_video_conditioning,
                device,
            }
            .run(|request| {
                let (video, audio) = run_real_distilled_stage(
                    transformer,
                    request.latent_shape,
                    stage2_audio.shape,
                    &request.start_latents,
                    &request.conditioning,
                    plan.seed,
                    None,
                    stage2_audio.start,
                    None,
                    &stage2_audio_conditioning,
                    &request.positions,
                    stage2_audio.positions,
                    &prompt_inputs.cond_context,
                    None,
                    prompt_inputs.alt_context.as_ref(),
                    stage2_audio.context,
                    stage2_audio.uncond_context,
                    stage2_audio.alt_context,
                    cond_mask,
                    None,
                    alt_mask,
                    None,
                    stage_guidance_scale(plan, 1)?,
                    stage2_sigmas,
                    stage_sampler_mode(plan, 1)?,
                    Some(&request.sampler_noise),
                    stage2_audio.noise,
                    None,
                    stage2_audio.denoise_mask,
                    Some("distilled.stage2"),
                    debug_enabled.then_some(stage2_tile_debug_label(request.index)),
                    progress,
                    cancellation,
                )?;
                refined_audio = audio;
                Ok(video)
            })?;
            Ok((
                video,
                stage2_carried_audio(
                    refined_audio,
                    stage1_audio_latents.as_ref(),
                    stage2_is_tiled,
                ),
            ))
        },
        progress,
    )?;
    log_timing("distilled.stage2.denoise", stage2_denoise_start);
    if debug_enabled {
        log_debug_vram("after_stage2_denoise");
    }
    drop(stage2_transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_stage2_transformer_drop");
    }
    let latents = maybe_apply_temporal_upsampler(plan, &latents, device, dtype)?;
    if debug_enabled && plan.temporal_upscale.is_some() {
        log_debug_vram("after_temporal_upsample");
    }
    if debug_enabled {
        log_tensor_stats("final_video_latents", &latents)?;
    }
    let mut vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
    // Chain-stage hook: capture the pre-decode F32 latents so
    // `Ltx2Engine::render_chain_stage` can narrow the tail off for the next
    // stage's conditioning. Cheap shallow clone (candle tensors are
    // Arc-backed). A poisoned mutex is ignored here — the outer caller
    // detects an empty slot and emits a clear error.
    if let Some(slot) = tail_capture {
        if let Ok(mut guard) = slot.lock() {
            *guard = Some(latents.clone());
        }
    }
    let decoded = decode_video_frames_with_telemetry(
        "distilled",
        &mut vae,
        &latents,
        requested_pixel_shape,
        dtype,
        device,
        debug_enabled,
        plan_hdr_exr_target(plan).as_ref(),
        progress,
    )?;
    let frames = decoded.frames;
    let hdr_frames_written = decoded.hdr_frames_written;
    drop(vae);
    let audio_render_start = Instant::now();
    let audio_track = maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?;
    log_timing("distilled.render_audio", audio_render_start);
    drop(latents);
    drop(audio_latents);
    drop(stage2_audio_start);
    drop(stage2_audio_noise);
    drop(stage2_clean_video_latents_f32);
    drop(stage2_clean_video_latents);
    drop(stage2_tiles);
    drop(stage1_audio_latents);
    drop(stage1_video_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        hdr_frames_written,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

// ── Tiled stage-2 refinement ─────────────────────────────────────────────────
//
// Past 2048 px on an axis the checkpoints' RoPE runs outside the span it was
// trained on: the picture still renders, it just stops being structurally
// sound. Upstream's answer is to refine in overlapping latent tiles, each
// denoised at a shape the model handles, with positions renormalized so every
// tile looks like a sequence starting at zero
// (`hdr_ic_lora.py:493-563`). The tile arithmetic lives in `tiling.rs`; this is
// the execution.
//
// Everything here is inert unless `tiling::plan_stage2_tiling_with_policy`
// returns more than one tile, which `Auto` can only do past the trained span —
// a resolution `mold_core::validation` does not admit today.

/// Everything stage 2 feeds the audio branch.
///
/// A tiled stage 2 is video-only. That is upstream's behaviour, not a
/// simplification of it — `hdr_ic_lora.py:504-507` states it outright:
///
/// > Each tile calls `stage_2.run()` with a tile-sized `ModalitySpec` for
/// > video only (audio is omitted entirely for HDR).
///
/// The reason holds independently: a spatial tile carries no statement about
/// an audio track, so refining one once per tile would denoise the same track
/// N times with no defensible way to recombine the results. Stage 1's audio is
/// carried through instead.
///
/// The branch is switched off as a unit — shape, latents, noise, positions,
/// and context together, never half of it. Half of it is a real failure, not a
/// tidiness argument: a static input batch carrying audio context for a
/// forward pass with no audio latents makes the transformer reject the step
/// with "audio hidden states, static inputs, sigma, and timesteps must be
/// provided together".
#[derive(Default)]
struct Stage2AudioInputs<'a> {
    shape: Option<AudioLatentShape>,
    start: Option<&'a Tensor>,
    noise: Option<&'a Tensor>,
    positions: Option<&'a Tensor>,
    context: Option<&'a Tensor>,
    uncond_context: Option<&'a Tensor>,
    alt_context: Option<&'a Tensor>,
    denoise_mask: Option<&'a Tensor>,
}

impl Stage2AudioInputs<'_> {
    fn video_only(had_audio: bool) -> Self {
        if had_audio {
            tracing::info!(
                target: LTX2_VRAM_TARGET,
                "[ltx2-vram] tiled stage 2 refines video only; the audio track from stage 1 is \
                 carried through unrefined"
            );
        }
        Self::default()
    }
}

/// Pair [`Stage2AudioInputs`]: what actually leaves stage 2.
fn stage2_carried_audio(
    refined: Option<Tensor>,
    stage1_audio: Option<&Tensor>,
    tiled: bool,
) -> Option<Tensor> {
    if tiled {
        return stage1_audio.cloned();
    }
    refined
}

/// Per-tile debug prefix, so `MOLD_LTX_DEBUG` traces stay attributable when a
/// stage runs more than once.
fn stage2_tile_debug_label(index: usize) -> &'static str {
    const LABELS: [&str; 8] = [
        "stage2",
        "stage2.t1",
        "stage2.t2",
        "stage2.t3",
        "stage2.t4",
        "stage2.t5",
        "stage2.t6",
        "stage2.t7",
    ];
    LABELS[index.min(LABELS.len() - 1)]
}

/// Resolve `MOLD_LTX2_SPATIAL_TILE` (the `--spatial-tile` knob).
fn spatial_tile_policy() -> Result<SpatialTilePolicy> {
    match crate::runtime_env::value("MOLD_LTX2_SPATIAL_TILE") {
        Some(value) => SpatialTilePolicy::parse(&value)
            .context("invalid MOLD_LTX2_SPATIAL_TILE / --spatial-tile value"),
        None => Ok(SpatialTilePolicy::Auto),
    }
}

/// Resolve the stage-2 tile layout for `shape` and report what it decided.
///
/// Reported once per stage-2 pass, always: at these sizes the failure mode is
/// degraded structure rather than an error, so which of the two paths ran has
/// to be correlatable with the output after the fact.
fn plan_stage2_tiles(shape: VideoLatentShape) -> Result<Vec<Tile>> {
    plan_stage2_tiles_with_policy(shape, spatial_tile_policy()?)
}

/// [`plan_stage2_tiles`] with the policy supplied rather than read from the
/// environment, so the refusal below is testable without mutating a process
/// the rest of the suite shares.
fn plan_stage2_tiles_with_policy(
    shape: VideoLatentShape,
    policy: SpatialTilePolicy,
) -> Result<Vec<Tile>> {
    let config = plan_stage2_tiling_with_policy(shape.frames, shape.height, shape.width, policy);
    let tiles = create_tiles(shape.frames, shape.height, shape.width, config)?;
    let past_trained_span = shape.height.max(shape.width) > TRAINED_SPATIAL_LATENT_SPAN;
    if tiles.len() > 1 {
        tracing::info!(
            target: LTX2_VRAM_TARGET,
            "[ltx2-vram] stage 2 refining latent {}x{} as {} tiles ({}x{}), each denoised \
             inside the {}-cell span these checkpoints were trained on",
            shape.width,
            shape.height,
            tiles.len(),
            config.width.num_tiles,
            config.height.num_tiles,
            TRAINED_SPATIAL_LATENT_SPAN,
        );
    } else if past_trained_span {
        // Refusing beats rendering. The failure mode here is not an error but
        // a plausible-looking video with degraded large-scale structure, and a
        // user who turned tiling off to save time would have no way to tell
        // that is what they got. Nothing that was renderable before this
        // ceiling rose can reach this branch: `LTX2_MAX_AXIS_PIXELS` did not
        // admit an oversized axis at all.
        bail!(
            "stage 2 has to refine a {}x{} latent, past the {}-cell span these checkpoints were \
             trained on, but spatial tiling is disabled. Positions past the span are out of \
             distribution and the render would be quietly degraded rather than fail. Drop \
             --spatial-tile off (or MOLD_LTX2_SPATIAL_TILE) to let auto-tiling handle it, or \
             render at or below {}px on the long edge.",
            shape.width,
            shape.height,
            TRAINED_SPATIAL_LATENT_SPAN,
            mold_core::validation::LTX2_MAX_AXIS_PIXELS,
        );
    }
    Ok(tiles)
}

/// The shape stage 2 actually pushes through the transformer in one pass.
///
/// Residency planning budgets activations from the stage shape, so a tiled
/// refinement has to declare its largest *tile* rather than the whole frame —
/// otherwise the plan reserves for a forward pass that never happens and
/// streams blocks it had room to keep. A single full-cover tile returns the
/// full shape unchanged.
fn stage2_forward_pixel_shape(tiles: &[Tile], full: VideoPixelShape) -> VideoPixelShape {
    let mut largest = full;
    let mut largest_tokens = 0usize;
    for tile in tiles {
        let tokens = tile.token_count();
        if tokens <= largest_tokens {
            continue;
        }
        largest_tokens = tokens;
        let (width, height, frames) = tile.pixel_shape();
        largest = VideoPixelShape {
            batch: full.batch,
            frames,
            height,
            width,
            fps: full.fps,
        };
    }
    largest
}

/// One tile's fully-prepared stage-2 inputs, handed to the denoiser.
struct Stage2TileRequest {
    /// Enumeration index, which is also what seeds this tile's noise.
    index: usize,
    latent_shape: VideoLatentShape,
    /// Noised tile input, `[B, C, f, h, w]`.
    start_latents: Tensor,
    /// The same noise, kept for the sampler's stochastic steps.
    sampler_noise: Tensor,
    /// Positions built at the tile's own pixel shape, so they start at zero.
    positions: Tensor,
    conditioning: StageVideoConditioning,
}

/// Everything a tiled stage-2 pass needs that does not vary per tile.
struct TiledStage2Pass<'a> {
    tiles: &'a [Tile],
    full_shape: VideoLatentShape,
    /// The stage-1 upscaled latent, `[B, C, F, H, W]`, f32.
    clean_latents: &'a Tensor,
    /// Base seed for per-tile noise; tile `i` uses `seed + i`
    /// (`hdr_ic_lora.py:547`).
    noise_seed: u64,
    /// Stage-2's first sigma — how much noise the refinement starts from.
    sigma: f32,
    fps: f32,
    conditioning: &'a StageVideoConditioning,
    device: &'a candle_core::Device,
}

impl TiledStage2Pass<'_> {
    /// Denoise every tile and recombine them with the trapezoidal window.
    ///
    /// A single full-cover tile short-circuits to its own result: an untiled
    /// refinement must stay bit-identical to what it was before tiling
    /// existed, and an accumulate-then-unpatchify round trip is not something
    /// to take on trust at that boundary.
    fn run<F>(&self, mut denoise_tile: F) -> Result<Tensor>
    where
        F: FnMut(Stage2TileRequest) -> Result<Tensor>,
    {
        let patchifier = VideoLatentPatchifier::new(1);
        let mut accumulated: Option<Tensor> = None;

        for (index, tile) in self.tiles.iter().enumerate() {
            let request = self.prepare_tile(index, tile)?;
            let is_only_tile = self.tiles.len() == 1;
            let denoised = denoise_tile(request)?;
            if is_only_tile {
                return Ok(denoised);
            }

            let tokens = patchifier.patchify(&denoised)?;
            let (_, token_count, _) = tokens.dims3()?;
            let window = Tensor::from_vec(tile.blend_window(), (1, token_count, 1), self.device)?
                .to_dtype(tokens.dtype())?;
            let weighted = tokens.broadcast_mul(&window)?;

            let indices = tile.token_indices(self.full_shape.height, self.full_shape.width);
            let indices = Tensor::from_vec(
                indices
                    .iter()
                    .map(|index| *index as u32)
                    .collect::<Vec<_>>(),
                token_count,
                self.device,
            )?;

            let target = match accumulated.take() {
                Some(target) => target,
                None => Tensor::zeros(
                    (
                        self.full_shape.batch,
                        patchifier.get_token_count(self.full_shape),
                        tokens.dim(2)?,
                    ),
                    tokens.dtype(),
                    self.device,
                )?,
            };
            accumulated = Some(target.index_add(&indices, &weighted, 1)?);
        }

        let accumulated =
            accumulated.context("a stage-2 tile plan must contain at least one tile")?;
        patchifier.unpatchify(&accumulated, self.full_shape)
    }

    fn prepare_tile(&self, index: usize, tile: &Tile) -> Result<Stage2TileRequest> {
        let latent_shape = VideoLatentShape {
            batch: self.full_shape.batch,
            channels: self.full_shape.channels,
            frames: tile.frames.len(),
            height: tile.height.len(),
            width: tile.width.len(),
        };
        let clean = self
            .clean_latents
            .narrow(2, tile.frames.start, tile.frames.len())?
            .narrow(3, tile.height.start, tile.height.len())?
            .narrow(4, tile.width.start, tile.width.len())?
            .contiguous()?;
        // Per-tile noise, not a slice of one full-shape draw: upstream reseeds
        // the generator per tile, and a slice would make the result depend on
        // the layout of tiles it is not part of.
        let sampler_noise = seeded_randn(
            self.noise_seed.wrapping_add(index as u64),
            &[
                latent_shape.batch,
                latent_shape.channels,
                latent_shape.frames,
                latent_shape.height,
                latent_shape.width,
            ],
            self.device,
            DType::F32,
        )?;
        let start_latents = mix_clean_latents_with_noise(&clean, &sampler_noise, self.sigma)?;

        let (pixel_width, pixel_height, pixel_frames) = tile.pixel_shape();
        let positions = build_video_positions(
            VideoPixelShape {
                batch: latent_shape.batch,
                frames: pixel_frames,
                height: pixel_height,
                width: pixel_width,
                fps: self.fps,
            },
            self.device,
        )?;

        Ok(Stage2TileRequest {
            index,
            latent_shape,
            start_latents,
            sampler_noise,
            positions,
            conditioning: conditioning_for_tile(self.conditioning, self.full_shape, tile)?,
        })
    }
}

/// Slice stage conditioning down to one tile's spatial region.
///
/// Token replacements are a contiguous run of the full grid's tokens, so the
/// tile keeps whichever of its own tokens land inside that run. Appended
/// conditions carry their own latent grid, which is sliced by the tile's box
/// scaled into that grid's resolution, and their positions are rebased onto
/// the tile's origin — the generated tokens' positions already start at zero,
/// and conditioning that stayed absolute would sit somewhere else entirely.
fn conditioning_for_tile(
    conditioning: &StageVideoConditioning,
    full_shape: VideoLatentShape,
    tile: &Tile,
) -> Result<StageVideoConditioning> {
    if conditioning.is_empty() {
        return Ok(StageVideoConditioning::default());
    }
    if tile.frames.len() != full_shape.frames {
        // `plan_stage2_tiling*` never splits time, so this is a fence rather
        // than a case: a tile that held only part of the timeline would have
        // to decide which keyframe conditions still apply to it.
        bail!(
            "stage-2 conditioning cannot be sliced across a temporal tile ({} of {} frames)",
            tile.frames.len(),
            full_shape.frames
        );
    }
    let covers_everything = tile.height.len() == full_shape.height
        && tile.width.len() == full_shape.width
        && tile.frames.len() == full_shape.frames;
    if covers_everything {
        return Ok(conditioning.clone());
    }

    let token_indices = tile.token_indices(full_shape.height, full_shape.width);
    let mut replacements = Vec::with_capacity(conditioning.replacements.len());
    for replacement in &conditioning.replacements {
        let count = replacement.tokens.dim(1)?;
        let range = replacement.start_token..replacement.start_token + count;
        let mut local_start = None;
        let mut rows: Vec<u32> = Vec::new();
        for (local, global) in token_indices.iter().enumerate() {
            if !range.contains(global) {
                continue;
            }
            match local_start {
                None => local_start = Some(local),
                Some(start) if local == start + rows.len() => {}
                Some(start) => bail!(
                    "stage-2 conditioning replacement is not contiguous within a tile \
                     (token {local} follows {})",
                    start + rows.len() - 1
                ),
            }
            rows.push((global - replacement.start_token) as u32);
        }
        let Some(local_start) = local_start else {
            continue;
        };
        let selector = Tensor::from_vec(rows.clone(), rows.len(), replacement.tokens.device())?;
        replacements.push(VideoTokenReplacement {
            start_token: local_start,
            tokens: replacement.tokens.index_select(&selector, 1)?,
            strength: replacement.strength,
        });
    }

    let mut appended = Vec::with_capacity(conditioning.appended.len());
    for condition in &conditioning.appended {
        appended.push(slice_append_condition_to_tile(condition, tile)?);
    }

    Ok(StageVideoConditioning {
        replacements,
        appended,
    })
}

fn slice_append_condition_to_tile(
    condition: &VideoTokenAppendCondition,
    tile: &Tile,
) -> Result<VideoTokenAppendCondition> {
    let (grid_frames, grid_height, grid_width) = condition.latent_grid;
    let factor = condition.spatial_downscale_factor.max(1);
    let (batch, token_count, channels) = condition.tokens.dims3()?;
    if grid_frames * grid_height * grid_width != token_count {
        bail!(
            "appended conditioning grid {grid_frames}x{grid_height}x{grid_width} does not \
             describe its {token_count} tokens"
        );
    }
    // Upstream floors both ends (`hdr_ic_lora.py:534-536`); with a reference
    // grid at `1/df` of the generated one and dimensions validated divisible
    // by `df`, the two ends land on real cell boundaries.
    let height_start = (tile.height.start / factor).min(grid_height);
    let height_end = (tile.height.end / factor).min(grid_height);
    let width_start = (tile.width.start / factor).min(grid_width);
    let width_end = (tile.width.end / factor).min(grid_width);
    if height_start >= height_end || width_start >= width_end {
        bail!(
            "appended conditioning grid {grid_height}x{grid_width} has nothing under tile \
             rows {}..{} cols {}..{}",
            tile.height.start,
            tile.height.end,
            tile.width.start,
            tile.width.end
        );
    }
    let (tile_height, tile_width) = (height_end - height_start, width_end - width_start);

    let tokens = condition
        .tokens
        .reshape((batch, grid_frames, grid_height, grid_width, channels))?
        .narrow(2, height_start, tile_height)?
        .narrow(3, width_start, tile_width)?
        .contiguous()?
        .reshape((batch, grid_frames * tile_height * tile_width, channels))?;

    // Positions are `[B, 3, tokens, 2]` in `(time, height, width)` order.
    let (position_batch, axes, position_tokens, bounds) = condition.positions.dims4()?;
    if position_tokens != token_count {
        bail!("appended conditioning carries {position_tokens} positions for {token_count} tokens");
    }
    let positions = condition
        .positions
        .reshape((
            position_batch,
            axes,
            grid_frames,
            grid_height,
            grid_width,
            bounds,
        ))?
        .narrow(3, height_start, tile_height)?
        .narrow(4, width_start, tile_width)?
        .contiguous()?
        .reshape((
            position_batch,
            axes,
            grid_frames * tile_height * tile_width,
            bounds,
        ))?;
    // Rebase onto the tile's origin. `scale_video_spatial_positions` already
    // expressed these in full-resolution pixels, and the tile's own generated
    // tokens now start at zero, so the offset is the tile's pixel origin —
    // exactly the `gen_pos.amin` upstream subtracts from every kept position
    // (`modality_tiling.py:116-120`). Time is untouched: time is never tiled.
    let stride = crate::ltx2::tiling::LATENT_PIXEL_STRIDE;
    let positions = Tensor::cat(
        &[
            positions.i((.., 0..1, .., ..))?,
            positions
                .i((.., 1..2, .., ..))?
                .affine(1.0, -((tile.height.start * stride) as f64))?,
            positions
                .i((.., 2..3, .., ..))?
                .affine(1.0, -((tile.width.start * stride) as f64))?,
        ],
        1,
    )?;

    Ok(VideoTokenAppendCondition {
        tokens,
        positions,
        strength: condition.strength,
        latent_grid: (grid_frames, tile_height, tile_width),
        spatial_downscale_factor: condition.spatial_downscale_factor,
    })
}

fn render_real_two_stage_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: true,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let uncond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let dtype = compute_dtype(device);
    let conditioned_audio = maybe_load_native_conditioning_audio(plan, audio_shape, device, dtype)?;
    let frozen_audio_denoise_mask = conditioned_audio
        .as_ref()
        .map(|_| {
            build_frozen_audio_denoise_mask(
                audio_shape.context("frozen audio conditioning requires an audio latent shape")?,
                device,
            )
        })
        .transpose()?;
    let stage1_audio_noise = if conditioned_audio.is_some() {
        None
    } else {
        match audio_shape {
            Some(audio_shape) => Some(seeded_randn(
                plan.seed ^ 0x4155_4449_4f4c_5458,
                &[
                    audio_shape.batch,
                    audio_shape.channels,
                    audio_shape.frames,
                    audio_shape.mel_bins,
                ],
                device,
                DType::F32,
            )?),
            None => None,
        }
    };
    let stage1_context = prepare_stage_context(plan, 0, device)?;
    let latent_stats = Ltx2VaeLatentStats::load(plan, device, dtype)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        matches!(plan.pipeline, PipelineKind::IcLora | PipelineKind::LipDub),
        progress,
    )?;
    // Lip dub hands the model the reference clip's own speech as negatively
    // positioned tokens sitting entirely before the generated audio, so the
    // dub inherits the speaker's voice rather than inventing one
    // (`lipdub.py:228-239`).
    let lip_dub_reference_audio = matches!(plan.pipeline, PipelineKind::LipDub)
        .then(|| load_lip_dub_reference_audio_latents(plan, device, dtype))
        .transpose()?;
    let stage1_audio_conditioning = match lip_dub_reference_audio.as_ref() {
        Some(latents) => StageAudioConditioning {
            appended: vec![append_condition_from_audio_latents(latents, 1.0)?],
        },
        None => StageAudioConditioning::default(),
    };
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage1 transformer");
    }
    let stage1_transformer_load_start = Instant::now();
    let stage1_stage_shape = Ltx2StageShape::from_pixel_shape(plan, prepared.video_pixel_shape);
    let stage1_transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage1_stage_shape,
        device,
        &stage1_context.loras,
        None,
        progress,
    )?;
    log_timing(
        "two_stage.stage1.transformer_load",
        stage1_transformer_load_start,
    );
    let stage1_audio_start = conditioned_audio
        .as_ref()
        .map(|audio| &audio.latents)
        .or(stage1_audio_noise.as_ref());
    let stage1_denoise_start = Instant::now();
    let ((stage1_video_latents, stage1_audio_latents), stage1_transformer) =
        run_denoise_stage_with_oom_recovery(
            "two_stage.stage1",
            stage1_transformer,
            device,
            |budget| {
                load_ltx2_av_transformer_with_loras(
                    plan,
                    stage1_stage_shape,
                    device,
                    &stage1_context.loras,
                    Some(budget),
                    progress,
                )
            },
            |transformer| {
                run_real_distilled_stage(
                    transformer,
                    prepared.video_latent_shape,
                    audio_shape,
                    &stage1_video_noise,
                    &stage1_video_conditioning,
                    plan.seed,
                    None,
                    stage1_audio_start,
                    None,
                    &stage1_audio_conditioning,
                    &prompt_inputs.video_positions,
                    prompt_inputs.audio_positions.as_ref(),
                    &prompt_inputs.cond_context,
                    stage1_context
                        .requires_unconditional_context
                        .then_some(prompt_inputs.uncond_context.as_ref())
                        .flatten(),
                    prompt_inputs.alt_context.as_ref(),
                    prompt_inputs.audio_context.as_ref(),
                    stage1_context
                        .requires_unconditional_context
                        .then_some(prompt_inputs.uncond_audio_context.as_ref())
                        .flatten(),
                    prompt_inputs.alt_audio_context.as_ref(),
                    cond_mask,
                    if stage1_context.requires_unconditional_context {
                        uncond_mask
                    } else {
                        None
                    },
                    alt_mask,
                    stage1_context.multimodal_guidance.clone(),
                    stage1_context.guidance_scale,
                    &stage1_context.sigmas_no_terminal,
                    stage1_context.sampler_mode,
                    Some(&stage1_video_noise),
                    stage1_audio_noise.as_ref(),
                    None,
                    frozen_audio_denoise_mask.as_ref(),
                    Some("two_stage.stage1"),
                    debug_enabled.then_some("stage1"),
                    progress,
                    cancellation,
                )
            },
            progress,
        )?;
    log_timing("two_stage.stage1.denoise", stage1_denoise_start);
    drop(stage1_transformer);
    device.synchronize()?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
        maybe_write_debug_stage_video(
            "stage1",
            &mut debug_vae,
            &stage1_video_latents,
            prepared.video_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }

    let spatial_upsampler_path = plan
        .spatial_upsampler_path
        .as_ref()
        .context("native LTX-2 two-stage inference requires a spatial upsampler asset")?;
    let stage1_upsample_start = Instant::now();
    let upsampler = LatentUpsampler::load(Path::new(spatial_upsampler_path), dtype, device)?;
    let stage2_clean_video_latents = latent_stats.normalize(
        &upsampler.forward(&latent_stats.denormalize(&stage1_video_latents.to_dtype(dtype)?)?)?,
    )?;
    drop(upsampler);
    device.synchronize()?;
    log_timing("two_stage.stage1.spatial_upsample", stage1_upsample_start);

    let requested_pixel_shape = VideoPixelShape {
        batch: 1,
        frames: plan.num_frames as usize,
        height: plan.height as usize,
        width: plan.width as usize,
        fps: plan.frame_rate as f32,
    };
    let stage2_video_latent_shape = video_latent_shape_from_tensor(&stage2_clean_video_latents)?;
    let stage2_pixel_shape =
        pixel_shape_for_video_latents(stage2_video_latent_shape, plan.frame_rate);
    let stage2_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        stage2_pixel_shape,
        device,
        dtype,
        plan.pipeline.keeps_reference_video_in_stage_two(),
        progress,
    )?;
    if env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX").is_some() {
        let mut debug_vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
        maybe_write_debug_stage_video(
            "stage1-upscaled",
            &mut debug_vae,
            &stage2_clean_video_latents,
            stage2_pixel_shape,
            dtype,
        )?;
        drop(debug_vae);
        device.synchronize()?;
    }
    let stage2_tiles = plan_stage2_tiles(stage2_video_latent_shape)?;
    let stage2_is_tiled = stage2_tiles.len() > 1;
    let stage2_video_noise_seed = plan.seed ^ 0x5354_4147_4532_4c54;
    let stage2_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_3254_4c58,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };
    let stage2_context = prepare_stage_context(plan, 1, device)?;
    let stage2_sigma = *stage2_context
        .sigmas_no_terminal
        .first()
        .context("stage2 sigma schedule must contain at least one step")?;
    let stage2_clean_video_latents_f32 = stage2_clean_video_latents.to_dtype(DType::F32)?;
    let stage2_audio_policy = stage2_audio_policy(plan.pipeline);
    let stage2_audio_is_frozen = stage2_audio_policy == Stage2AudioPolicy::Frozen;
    let stage2_audio_start = match (stage1_audio_latents.as_ref(), stage2_audio_noise.as_ref()) {
        (Some(stage1_audio_latents), _) if stage2_audio_is_frozen => {
            Some(stage1_audio_latents.to_dtype(DType::F32)?)
        }
        (Some(stage1_audio_latents), Some(stage2_audio_noise)) => {
            Some(mix_clean_latents_with_noise(
                &stage1_audio_latents.to_dtype(DType::F32)?,
                stage2_audio_noise,
                stage2_sigma,
            )?)
        }
        _ => None,
    };
    // The reference the refinement attends to is the audio stage 1 *produced*,
    // not the original clip's (`lipdub.py:267`): stage 2 is re-timing a mouth
    // against the dub, so pointing it at the source speech would fight the dub
    // it is supposed to sharpen.
    let stage2_audio_conditioning = match stage1_audio_latents.as_ref() {
        Some(stage1_audio_latents) if appends_audio_reference(plan.pipeline) => {
            StageAudioConditioning {
                appended: vec![append_condition_from_audio_latents(
                    stage1_audio_latents,
                    1.0,
                )?],
            }
        }
        _ => StageAudioConditioning::default(),
    };
    let stage2_frozen_audio_denoise_mask = if stage2_audio_is_frozen {
        Some(build_frozen_audio_denoise_mask(
            audio_shape.context("a frozen stage-2 audio pass requires an audio latent shape")?,
            device,
        )?)
    } else {
        frozen_audio_denoise_mask.clone()
    };

    let stage2_audio = if stage2_is_tiled {
        Stage2AudioInputs::video_only(audio_shape.is_some())
    } else {
        Stage2AudioInputs {
            shape: audio_shape,
            start: stage2_audio_start.as_ref(),
            noise: stage2_audio_noise.as_ref(),
            positions: prompt_inputs.audio_positions.as_ref(),
            context: prompt_inputs.audio_context.as_ref(),
            uncond_context: stage2_context
                .requires_unconditional_context
                .then_some(prompt_inputs.uncond_audio_context.as_ref())
                .flatten(),
            alt_context: prompt_inputs.alt_audio_context.as_ref(),
            denoise_mask: stage2_frozen_audio_denoise_mask.as_ref(),
        }
    };
    if debug_enabled {
        eprintln!("[ltx2-debug] loading stage2 transformer");
    }
    let stage2_transformer_load_start = Instant::now();
    let stage2_stage_shape = Ltx2StageShape::from_pixel_shape(
        plan,
        stage2_forward_pixel_shape(&stage2_tiles, stage2_pixel_shape),
    );
    let stage2_transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage2_stage_shape,
        device,
        &stage2_context.loras,
        None,
        progress,
    )?;
    log_timing(
        "two_stage.stage2.transformer_load",
        stage2_transformer_load_start,
    );
    let stage2_denoise_start = Instant::now();
    // A tiled stage 2 is video-only, so it carries no audio reference either —
    // the same rule as `Stage2AudioInputs::video_only`. Resolved before the
    // dispatch closure because the closure is `FnMut` and may run more than
    // once, so it cannot consume this.
    let stage2_audio_conditioning = if stage2_is_tiled {
        StageAudioConditioning::default()
    } else {
        stage2_audio_conditioning
    };
    let ((latents, audio_latents), stage2_transformer) = run_denoise_stage_with_oom_recovery(
        "two_stage.stage2",
        stage2_transformer,
        device,
        |budget| {
            load_ltx2_av_transformer_with_loras(
                plan,
                stage2_stage_shape,
                device,
                &stage2_context.loras,
                Some(budget),
                progress,
            )
        },
        |transformer| {
            let mut refined_audio = None;
            let video = TiledStage2Pass {
                tiles: &stage2_tiles,
                full_shape: stage2_video_latent_shape,
                clean_latents: &stage2_clean_video_latents_f32,
                noise_seed: stage2_video_noise_seed,
                sigma: stage2_sigma,
                fps: plan.frame_rate as f32,
                conditioning: &stage2_video_conditioning,
                device,
            }
            .run(|request| {
                let (video, audio) = run_real_distilled_stage(
                    transformer,
                    request.latent_shape,
                    stage2_audio.shape,
                    &request.start_latents,
                    &request.conditioning,
                    plan.seed,
                    None,
                    stage2_audio.start,
                    None,
                    &stage2_audio_conditioning,
                    &request.positions,
                    stage2_audio.positions,
                    &prompt_inputs.cond_context,
                    stage2_context
                        .requires_unconditional_context
                        .then_some(prompt_inputs.uncond_context.as_ref())
                        .flatten(),
                    prompt_inputs.alt_context.as_ref(),
                    stage2_audio.context,
                    stage2_audio.uncond_context,
                    stage2_audio.alt_context,
                    cond_mask,
                    if stage2_context.requires_unconditional_context {
                        uncond_mask
                    } else {
                        None
                    },
                    alt_mask,
                    stage2_context.multimodal_guidance.clone(),
                    stage2_context.guidance_scale,
                    &stage2_context.sigmas_no_terminal,
                    stage2_context.sampler_mode,
                    Some(&request.sampler_noise),
                    stage2_audio.noise,
                    None,
                    stage2_audio.denoise_mask,
                    Some("two_stage.stage2"),
                    debug_enabled.then_some(stage2_tile_debug_label(request.index)),
                    progress,
                    cancellation,
                )?;
                refined_audio = audio;
                Ok(video)
            })?;
            Ok((
                video,
                stage2_carried_audio(
                    refined_audio,
                    stage1_audio_latents.as_ref(),
                    stage2_is_tiled,
                ),
            ))
        },
        progress,
    )?;
    log_timing("two_stage.stage2.denoise", stage2_denoise_start);
    drop(stage2_transformer);
    device.synchronize()?;
    let latents = maybe_apply_temporal_upsampler(plan, &latents, device, dtype)?;

    let mut vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
    let decoded = decode_video_frames_with_telemetry(
        "two_stage",
        &mut vae,
        &latents,
        requested_pixel_shape,
        dtype,
        device,
        false,
        plan_hdr_exr_target(plan).as_ref(),
        progress,
    )?;
    let frames = decoded.frames;
    let hdr_frames_written = decoded.hdr_frames_written;
    drop(vae);
    let audio_render_start = Instant::now();
    let audio_track = if let Some(conditioned_audio) = conditioned_audio.as_ref() {
        conditioned_audio.original_track.clone()
    } else if stage2_audio_is_frozen {
        // Stage 2 never denoised the audio, so its "output" is the frozen copy
        // that went in. Upstream decodes `s1_audio_latent` (`lipdub.py:293`);
        // decoding stage 2's would round-trip the same samples through an
        // extra blend for nothing.
        maybe_render_native_audio_track(plan, stage1_audio_latents.as_ref(), device, dtype)?
    } else {
        maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?
    };
    log_timing("two_stage.render_audio", audio_render_start);
    drop(latents);
    drop(audio_latents);
    drop(stage2_audio_start);
    drop(stage2_audio_noise);
    drop(stage2_clean_video_latents_f32);
    drop(stage2_clean_video_latents);
    drop(stage2_tiles);
    drop(stage1_audio_latents);
    drop(stage1_video_latents);
    drop(stage1_audio_noise);
    drop(stage2_frozen_audio_denoise_mask);
    drop(frozen_audio_denoise_mask);
    drop(lip_dub_reference_audio);
    drop(conditioned_audio);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = uncond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    drop(latent_stats);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        hdr_frames_written,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

fn render_real_one_stage_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: true,
            include_alt: true,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let uncond_mask: Option<&Tensor> = None;
    let alt_mask: Option<&Tensor> = None;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        log_tensor_stats("video_context", &prompt_inputs.cond_context)?;
        if let Some(audio_context) = prompt_inputs.audio_context.as_ref() {
            log_tensor_stats("audio_context", audio_context)?;
        }
        log_tensor_stats("initial_video_latents", &stage1_video_noise)?;
        if let Some(stage1_audio_noise) = stage1_audio_noise.as_ref() {
            log_tensor_stats("initial_audio_latents", stage1_audio_noise)?;
        }
    }

    let dtype = compute_dtype(device);
    let stage1_guidance_scale = stage_guidance_scale(plan, 0)?;
    let stage1_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
        progress,
    )?;
    if debug_enabled {
        eprintln!("[ltx2-debug] loading one-stage transformer");
    }
    let stage_shape = Ltx2StageShape::from_pixel_shape(plan, prepared.video_pixel_shape);
    let stage_loras = stage_lora_stack(plan, 0)?;
    let transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage_shape,
        device,
        &stage_loras,
        None,
        progress,
    )?;
    if debug_enabled {
        log_debug_vram("after_one_stage_transformer_load");
    }
    let stage1_requires_uncond = stage_requires_unconditional_context(plan, 0)?;
    let ((latents, stage1_audio_latents), transformer) = run_denoise_stage_with_oom_recovery(
        "one_stage",
        transformer,
        device,
        |budget| {
            load_ltx2_av_transformer_with_loras(
                plan,
                stage_shape,
                device,
                &stage_loras,
                Some(budget),
                progress,
            )
        },
        |transformer| {
            run_real_distilled_stage(
                transformer,
                prepared.video_latent_shape,
                audio_shape,
                &stage1_video_noise,
                &stage1_video_conditioning,
                plan.seed,
                None,
                stage1_audio_noise.as_ref(),
                None,
                &StageAudioConditioning::default(),
                &prompt_inputs.video_positions,
                prompt_inputs.audio_positions.as_ref(),
                &prompt_inputs.cond_context,
                stage1_requires_uncond
                    .then_some(prompt_inputs.uncond_context.as_ref())
                    .flatten(),
                prompt_inputs.alt_context.as_ref(),
                prompt_inputs.audio_context.as_ref(),
                stage1_requires_uncond
                    .then_some(prompt_inputs.uncond_audio_context.as_ref())
                    .flatten(),
                prompt_inputs.alt_audio_context.as_ref(),
                cond_mask,
                if stage1_requires_uncond {
                    uncond_mask
                } else {
                    None
                },
                alt_mask,
                None,
                stage1_guidance_scale,
                DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
                stage_sampler_mode(plan, 0)?,
                Some(&stage1_video_noise),
                stage1_audio_noise.as_ref(),
                None,
                None,
                Some("one_stage"),
                debug_enabled.then_some("one-stage"),
                progress,
                cancellation,
            )
        },
        progress,
    )?;
    if debug_enabled {
        log_debug_vram("after_one_stage_denoise");
        log_tensor_stats("final_video_latents", &latents)?;
    }
    drop(transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_one_stage_transformer_drop");
    }

    let mut vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
    let decoded = decode_video_frames_with_telemetry(
        "one_stage",
        &mut vae,
        &latents,
        prepared.video_pixel_shape,
        dtype,
        device,
        debug_enabled,
        plan_hdr_exr_target(plan).as_ref(),
        progress,
    )?;
    let frames = decoded.frames;
    let hdr_frames_written = decoded.hdr_frames_written;
    drop(vae);
    let audio_track =
        maybe_render_native_audio_track(plan, stage1_audio_latents.as_ref(), device, dtype)?;
    drop(latents);
    drop(stage1_audio_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    let _ = cond_mask;
    let _ = uncond_mask;
    let _ = alt_mask;
    drop(prompt_inputs);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        hdr_frames_written,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

fn render_real_retake_av(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<NativeRenderedVideo> {
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: false,
            include_alt: false,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape;
    let cond_mask: Option<&Tensor> = None;
    let dtype = compute_dtype(device);
    let retake_range = plan
        .retake_range
        .as_ref()
        .context("native LTX-2 retake requires a retake_range")?;
    let source_video = maybe_load_native_conditioning_video(
        plan,
        prepared.video_pixel_shape,
        prepared.video_latent_shape,
        device,
        dtype,
        progress,
    )?
    .context("native LTX-2 retake requires a source_video")?;
    let stage_video_conditioning = maybe_load_stage_video_conditioning(
        plan,
        prepared.video_pixel_shape,
        device,
        dtype,
        false,
        progress,
    )?;
    let video_retake_mask =
        build_temporal_token_denoise_mask(retake_range, &prompt_inputs.video_positions, device)?;
    let stage1_video_noise = seeded_randn(
        plan.seed,
        &[
            prepared.video_latent_shape.batch,
            prepared.video_latent_shape.channels,
            prepared.video_latent_shape.frames,
            prepared.video_latent_shape.height,
            prepared.video_latent_shape.width,
        ],
        device,
        DType::F32,
    )?;
    let conditioned_audio = maybe_load_native_conditioning_audio(plan, audio_shape, device, dtype)?;
    let audio_retake_mask = match (
        retake_range,
        prompt_inputs.audio_positions.as_ref(),
        conditioned_audio.as_ref(),
    ) {
        (range, Some(audio_positions), Some(_)) => Some(build_temporal_token_denoise_mask(
            range,
            audio_positions,
            device,
        )?),
        _ => None,
    };
    let stage1_audio_noise = match audio_shape {
        Some(audio_shape) => Some(seeded_randn(
            plan.seed ^ 0x4155_4449_4f4c_5458,
            &[
                audio_shape.batch,
                audio_shape.channels,
                audio_shape.frames,
                audio_shape.mel_bins,
            ],
            device,
            DType::F32,
        )?),
        None => None,
    };

    if debug_enabled {
        eprintln!("[ltx2-debug] loading retake transformer");
    }
    let stage_shape = Ltx2StageShape::from_pixel_shape(plan, prepared.video_pixel_shape);
    let stage_loras = stage_lora_stack(plan, 0)?;
    let transformer = load_ltx2_av_transformer_with_loras(
        plan,
        stage_shape,
        device,
        &stage_loras,
        None,
        progress,
    )?;
    let ((latents, audio_latents), transformer) = run_denoise_stage_with_oom_recovery(
        "retake",
        transformer,
        device,
        |budget| {
            load_ltx2_av_transformer_with_loras(
                plan,
                stage_shape,
                device,
                &stage_loras,
                Some(budget),
                progress,
            )
        },
        |transformer| {
            run_real_distilled_stage(
                transformer,
                prepared.video_latent_shape,
                audio_shape,
                &stage1_video_noise,
                &stage_video_conditioning,
                plan.seed,
                Some(&source_video.latents),
                stage1_audio_noise.as_ref(),
                conditioned_audio.as_ref().map(|audio| &audio.latents),
                &StageAudioConditioning::default(),
                &prompt_inputs.video_positions,
                prompt_inputs.audio_positions.as_ref(),
                &prompt_inputs.cond_context,
                None,
                None,
                prompt_inputs.audio_context.as_ref(),
                None,
                None,
                cond_mask,
                None,
                None,
                None,
                stage_guidance_scale(plan, 0)?,
                DISTILLED_STAGE1_SIGMAS_NO_TERMINAL,
                stage_sampler_mode(plan, 0)?,
                Some(&stage1_video_noise),
                stage1_audio_noise.as_ref(),
                Some(&video_retake_mask),
                audio_retake_mask.as_ref(),
                Some("retake.stage1"),
                debug_enabled.then_some("retake"),
                progress,
                cancellation,
            )
        },
        progress,
    )?;
    drop(transformer);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let mut vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
    let decoded = decode_video_frames_with_telemetry(
        "retake",
        &mut vae,
        &latents,
        prepared.video_pixel_shape,
        dtype,
        device,
        debug_enabled,
        plan_hdr_exr_target(plan).as_ref(),
        progress,
    )?;
    let frames = decoded.frames;
    let hdr_frames_written = decoded.hdr_frames_written;
    drop(vae);
    let audio_track = maybe_render_native_audio_track(plan, audio_latents.as_ref(), device, dtype)?;
    drop(latents);
    drop(audio_latents);
    drop(stage1_audio_noise);
    drop(stage1_video_noise);
    drop(audio_retake_mask);
    drop(video_retake_mask);
    drop(conditioned_audio);
    drop(source_video);
    let _ = cond_mask;
    drop(prompt_inputs);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let has_audio = audio_track.is_some();
    let audio_sample_rate = audio_track.as_ref().map(|track| track.sample_rate);
    let audio_channels = audio_track.as_ref().map(|track| u32::from(track.channels));

    Ok(NativeRenderedVideo {
        frames,
        hdr_frames_written,
        audio_track,
        has_audio,
        audio_sample_rate,
        audio_channels,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_real_distilled_stage(
    transformer: &Ltx2AvTransformer3DModel,
    video_shape: VideoLatentShape,
    audio_shape: Option<AudioLatentShape>,
    video_start_latents: &Tensor,
    video_conditioning: &StageVideoConditioning,
    request_seed: u64,
    video_clean_latents: Option<&Tensor>,
    audio_start_latents: Option<&Tensor>,
    audio_clean_latents: Option<&Tensor>,
    audio_conditioning: &StageAudioConditioning,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    cond_context: &Tensor,
    uncond_context: Option<&Tensor>,
    alt_context: Option<&Tensor>,
    audio_context: Option<&Tensor>,
    uncond_audio_context: Option<&Tensor>,
    alt_audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    alt_mask: Option<&Tensor>,
    multimodal_guidance: Option<(MultiModalGuiderParams, MultiModalGuiderParams)>,
    guidance_scale: f64,
    sigmas_no_terminal: &[f32],
    sampler_mode: SamplerMode,
    video_sampler_noise: Option<&Tensor>,
    audio_sampler_noise: Option<&Tensor>,
    video_denoise_mask: Option<&Tensor>,
    audio_denoise_mask: Option<&Tensor>,
    timing_label: Option<&str>,
    debug_stage: Option<&str>,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<(Tensor, Option<Tensor>)> {
    let device = video_start_latents.device().clone();
    let video_patchifier = VideoLatentPatchifier::new(1);
    let audio_patchifier = AudioPatchifier::new(
        LTX2_AUDIO_SAMPLE_RATE,
        LTX2_AUDIO_HOP_LENGTH,
        LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        true,
        0,
    );
    let mut run_sigmas = sigmas_no_terminal.to_vec();
    run_sigmas.push(0.0);
    if let Some(progress) = progress {
        progress(ProgressEvent::StageStart {
            name: format!("Denoising ({} steps)", run_sigmas.len().saturating_sub(1)),
        });
    }
    let base_video_token_count = video_patchifier.get_token_count(video_shape);
    let clean_video_override = video_clean_latents
        .map(|latents| video_patchifier.patchify(latents))
        .transpose()?;
    let init = initialize_video_stage_latents(
        &video_patchifier.patchify(video_start_latents)?,
        video_positions,
        video_conditioning,
        request_seed,
        clean_video_override.as_ref(),
        video_denoise_mask,
        base_video_token_count,
        &device,
    )?;
    let mut video_latents = init.latents;
    let conditioned_video_positions = init.positions;
    let clean_video_latents = init.clean;
    let video_denoise_mask = init.denoise_mask;
    let video_self_attention_mask = build_video_conditioning_self_attention_mask(
        base_video_token_count,
        video_conditioning,
        &device,
    )?;
    let video_positions = &conditioned_video_positions;
    let uses_video_freeze_mask = clean_video_override.is_some() || !video_conditioning.is_empty();
    let video_sampler_noise = video_sampler_noise
        .map(|noise| video_patchifier.patchify(noise))
        .transpose()?;
    let base_audio_token_count = audio_shape.map(|shape| audio_patchifier.get_token_count(shape));
    if !audio_conditioning.is_empty() && audio_shape.is_none() {
        anyhow::bail!("appended audio conditioning requires an audio latent shape");
    }
    let mut audio_latents = match (audio_shape, audio_start_latents) {
        (Some(_), Some(latents)) => Some(audio_patchifier.patchify(latents)?),
        _ => None,
    };
    let has_explicit_audio_clean = audio_shape.is_some() && audio_clean_latents.is_some();
    let clean_audio_latents = match (audio_shape, audio_clean_latents) {
        (Some(_), Some(latents)) => Some(audio_patchifier.patchify(latents)?),
        _ => audio_latents.clone(),
    };
    let audio_sampler_noise = match (audio_shape, audio_sampler_noise) {
        (Some(_), Some(noise)) => Some(audio_patchifier.patchify(noise)?),
        _ => None,
    };
    // Reference tokens ride along with the denoised sequence: appended to the
    // latents, to the clean target the freeze blend restores every step, and
    // to the RoPE positions, with a self-attention mask that lets the noisy
    // tokens see them without letting them see each other.
    let mut conditioned_audio_positions = audio_positions.cloned();
    let mut audio_self_attention_mask = None;
    if let (Some(audio_shape), Some(audio_positions)) = (audio_shape, audio_positions) {
        if !audio_conditioning.is_empty() {
            let latents = audio_latents
                .as_ref()
                .context("appended audio conditioning requires audio latents to append to")?;
            let (appended_latents, appended_positions) =
                apply_appended_audio_conditioning(latents, audio_positions, audio_conditioning)?;
            audio_latents = Some(appended_latents);
            conditioned_audio_positions = Some(appended_positions);
            audio_self_attention_mask = build_audio_conditioning_self_attention_mask(
                audio_shape.frames,
                audio_conditioning,
                &device,
            )?;
        }
    }
    let clean_audio_latents = match (clean_audio_latents, audio_positions) {
        (Some(clean), Some(positions)) if !audio_conditioning.is_empty() => {
            Some(apply_appended_audio_conditioning(&clean, positions, audio_conditioning)?.0)
        }
        (clean, _) => clean,
    };
    let audio_denoise_mask = match audio_shape {
        Some(audio_shape) => build_audio_conditioning_denoise_mask(
            audio_shape,
            audio_denoise_mask,
            audio_conditioning,
            &device,
        )?,
        // No audio latent shape means no reference tokens to widen the mask
        // for; pass whatever the caller supplied through untouched.
        None => audio_denoise_mask
            .map(|mask| mask.to_device(&device)?.to_dtype(DType::F32))
            .transpose()?,
    };
    let audio_positions = conditioned_audio_positions.as_ref();
    // The audio pre-loop freeze blend mirrors the video one inside
    // `initialize_video_stage_latents`: it installs explicit caller-supplied
    // clean latents (retake windows, lip-dub's frozen stage-2 audio) into the
    // masked regions of the fresh start latents. Without an explicit clean
    // tensor it is an exact identity — the derived clean *is* the start
    // latents, appended reference tokens included — so it runs only for the
    // explicit case (#1055: the equivalent video blend double-counted soft
    // source strength when applied to conditioning-derived cleans).
    if has_explicit_audio_clean {
        if let Some(blended_audio_latents) = match (
            audio_latents.as_ref(),
            clean_audio_latents.as_ref(),
            audio_denoise_mask.as_ref(),
        ) {
            (Some(audio_latents), Some(clean_audio_latents), Some(audio_denoise_mask)) => Some(
                blend_conditioned_denoised(audio_latents, clean_audio_latents, audio_denoise_mask)?,
            ),
            _ => None,
        } {
            audio_latents = Some(blended_audio_latents);
        }
    }
    let use_cfg = guidance_scale > 1.0;
    let multimodal_guiders = multimodal_guidance.map(|(video_params, audio_params)| {
        (
            MultiModalGuider::new(video_params, uncond_context.cloned()),
            MultiModalGuider::new(audio_params, uncond_audio_context.cloned()),
        )
    });
    let multimodal_static_batch = match multimodal_guiders.as_ref() {
        Some((video_guider, audio_guider)) => Some(prepare_static_multimodal_guidance_batch(
            transformer,
            cond_context,
            audio_context,
            cond_mask,
            uncond_mask,
            video_self_attention_mask.as_ref(),
            audio_self_attention_mask.as_ref(),
            video_positions,
            audio_positions,
            video_guider,
            audio_guider,
        )?),
        None => None,
    };
    let cond_static_inputs = if multimodal_guiders.is_none() {
        Some(transformer.prepare_static_inputs(
            cond_context,
            audio_context,
            cond_mask,
            cond_mask,
            video_self_attention_mask.as_ref(),
            audio_self_attention_mask.as_ref(),
            video_positions,
            audio_positions,
        )?)
    } else {
        None
    };
    let uncond_static_inputs = if multimodal_guiders.is_none() {
        match (uncond_context, uncond_audio_context) {
            (Some(uncond_context), uncond_audio_context) => {
                Some(transformer.prepare_static_inputs(
                    uncond_context,
                    uncond_audio_context,
                    uncond_mask,
                    uncond_mask,
                    video_self_attention_mask.as_ref(),
                    audio_self_attention_mask.as_ref(),
                    video_positions,
                    audio_positions,
                )?)
            }
            (None, _) => None,
        }
    } else {
        None
    };
    let alt_static_inputs = if multimodal_guiders.is_none() {
        match (alt_context, alt_audio_context) {
            (Some(alt_context), alt_audio_context) => Some(transformer.prepare_static_inputs(
                alt_context,
                alt_audio_context,
                alt_mask,
                alt_mask,
                video_self_attention_mask.as_ref(),
                audio_self_attention_mask.as_ref(),
                video_positions,
                audio_positions,
            )?),
            (None, _) => None,
        }
    } else {
        None
    };
    let mut step_setup_secs = 0.0;
    let mut transformer_secs = 0.0;
    let mut update_secs = 0.0;

    for (step_idx, sigma) in run_sigmas
        .iter()
        .copied()
        .take(run_sigmas.len().saturating_sub(1))
        .enumerate()
    {
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }
        let step_start = Instant::now();
        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6} entering");
        }
        let step_setup_start = Instant::now();
        let video_sigma = Tensor::full(sigma, (video_latents.dim(0)?,), &device)?;
        let video_timestep = timestep_from_sigma_and_mask(
            sigma,
            video_latents.dim(0)?,
            uses_video_freeze_mask.then_some(&video_denoise_mask),
            &device,
        )?;
        let audio_sigma = if let Some(audio_latents_ref) = audio_latents.as_ref() {
            Some(Tensor::full(sigma, (audio_latents_ref.dim(0)?,), &device)?)
        } else {
            None
        };
        let audio_timestep = if let Some(audio_latents_ref) = audio_latents.as_ref() {
            Some(timestep_from_sigma_and_mask(
                sigma,
                audio_latents_ref.dim(0)?,
                audio_denoise_mask.as_ref(),
                &device,
            )?)
        } else {
            None
        };
        step_setup_secs += step_setup_start.elapsed().as_secs_f64();
        let transformer_start = Instant::now();
        let (mut video_denoised, audio_denoised, video_velocity): (
            Tensor,
            Option<Tensor>,
            Option<Tensor>,
        ) = if let Some((video_guider, audio_guider)) = multimodal_guiders.as_ref() {
            let static_batch = multimodal_static_batch
                .as_ref()
                .context("missing static multimodal guidance batch")?;
            let (video_denoised, audio_denoised) = multimodal_guided_denoise_step(
                transformer,
                &video_latents,
                audio_latents.as_ref(),
                static_batch,
                &video_sigma,
                &video_timestep,
                audio_sigma.as_ref(),
                audio_timestep.as_ref(),
                video_guider,
                audio_guider,
                step_idx,
                progress,
            )?;
            (video_denoised, audio_denoised, None)
        } else if let Some(audio_latents_ref) = audio_latents.as_ref() {
            if use_cfg {
                let uncond_static_inputs = uncond_static_inputs
                    .as_ref()
                    .context("missing unconditional static inputs for CFG")?;
                let cond_static_inputs = cond_static_inputs
                    .as_ref()
                    .context("missing conditional static inputs for multimodal stage")?;
                let (uncond_video_velocity, uncond_audio_velocity) = transformer
                    .forward_with_static_inputs_and_progress(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        uncond_static_inputs,
                        None,
                        "Evaluating transformer (unconditional)",
                        progress,
                    )?;
                let (cond_video_velocity, cond_audio_velocity) = transformer
                    .forward_with_static_inputs_and_progress(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        cond_static_inputs,
                        None,
                        "Evaluating transformer (conditional)",
                        progress,
                    )?;
                let uncond_audio_velocity = uncond_audio_velocity
                    .context("audio branch unexpectedly returned no unconditional output")?;
                let cond_audio_velocity = cond_audio_velocity
                    .context("audio branch unexpectedly returned no conditional output")?;
                (
                    denoised_from_velocity(
                        &video_latents,
                        &guided_velocity_from_cfg(
                            &video_latents,
                            &cond_video_velocity,
                            &uncond_video_velocity,
                            sigma,
                            guidance_scale,
                        )?,
                        sigma,
                    )?,
                    Some(denoised_from_velocity(
                        audio_latents_ref,
                        &guided_velocity_from_cfg(
                            audio_latents_ref,
                            &cond_audio_velocity,
                            &uncond_audio_velocity,
                            sigma,
                            guidance_scale,
                        )?,
                        sigma,
                    )?),
                    Some(cond_video_velocity),
                )
            } else {
                let cond_static_inputs = cond_static_inputs
                    .as_ref()
                    .context("missing conditional static inputs for multimodal stage")?;
                let (cond_video_velocity, cond_audio_velocity) = transformer
                    .forward_with_static_inputs_and_progress(
                        &video_latents,
                        Some(audio_latents_ref),
                        &video_sigma,
                        &video_timestep,
                        audio_sigma.as_ref(),
                        audio_timestep.as_ref(),
                        cond_static_inputs,
                        None,
                        "Evaluating transformer (conditional)",
                        progress,
                    )?;
                if ltx_debug_compare_uncond_enabled() && step_idx == 0 {
                    if let Some(uncond_static_inputs) = uncond_static_inputs.as_ref() {
                        let (uncond_video_velocity, uncond_audio_velocity) = transformer
                            .forward_with_static_inputs_and_progress(
                                &video_latents,
                                Some(audio_latents_ref),
                                &video_sigma,
                                &video_timestep,
                                audio_sigma.as_ref(),
                                audio_timestep.as_ref(),
                                uncond_static_inputs,
                                None,
                                "Evaluating transformer (debug unconditional)",
                                progress,
                            )?;
                        log_distilled_prompt_sensitivity(
                            debug_stage,
                            step_idx,
                            sigma,
                            &video_latents,
                            &cond_video_velocity,
                            &uncond_video_velocity,
                            Some(audio_latents_ref),
                            cond_audio_velocity.as_ref(),
                            uncond_audio_velocity.as_ref(),
                        )?;
                    }
                }
                if step_idx == 0 {
                    if let Some(alt_static_inputs) = alt_static_inputs.as_ref() {
                        let (alt_video_velocity, alt_audio_velocity) = transformer
                            .forward_with_static_inputs_and_progress(
                                &video_latents,
                                Some(audio_latents_ref),
                                &video_sigma,
                                &video_timestep,
                                audio_sigma.as_ref(),
                                audio_timestep.as_ref(),
                                alt_static_inputs,
                                None,
                                "Evaluating transformer (debug alternate prompt)",
                                progress,
                            )?;
                        log_distilled_alternate_prompt_sensitivity(
                            debug_stage,
                            step_idx,
                            sigma,
                            &video_latents,
                            &cond_video_velocity,
                            &alt_video_velocity,
                            Some(audio_latents_ref),
                            cond_audio_velocity.as_ref(),
                            alt_audio_velocity.as_ref(),
                        )?;
                    }
                }
                (
                    denoised_from_velocity(&video_latents, &cond_video_velocity, sigma)?,
                    cond_audio_velocity
                        .as_ref()
                        .map(|velocity| denoised_from_velocity(audio_latents_ref, velocity, sigma))
                        .transpose()?,
                    Some(cond_video_velocity),
                )
            }
        } else if use_cfg {
            let uncond_static_inputs = uncond_static_inputs
                .as_ref()
                .context("missing unconditional static inputs for CFG")?;
            let cond_static_inputs = cond_static_inputs
                .as_ref()
                .context("missing conditional static inputs for video stage")?;
            let (uncond_video_velocity, _) = transformer.forward_with_static_inputs_and_progress(
                &video_latents,
                None,
                &video_sigma,
                &video_timestep,
                None,
                None,
                uncond_static_inputs,
                None,
                "Evaluating transformer (unconditional)",
                progress,
            )?;
            let (cond_video_velocity, _) = transformer.forward_with_static_inputs_and_progress(
                &video_latents,
                None,
                &video_sigma,
                &video_timestep,
                None,
                None,
                cond_static_inputs,
                None,
                "Evaluating transformer (conditional)",
                progress,
            )?;
            (
                denoised_from_velocity(
                    &video_latents,
                    &guided_velocity_from_cfg(
                        &video_latents,
                        &cond_video_velocity,
                        &uncond_video_velocity,
                        sigma,
                        guidance_scale,
                    )?,
                    sigma,
                )?,
                None,
                Some(cond_video_velocity),
            )
        } else {
            let cond_static_inputs = cond_static_inputs
                .as_ref()
                .context("missing conditional static inputs for video stage")?;
            let (cond_video_velocity, _cond_audio_velocity) = transformer
                .forward_with_static_inputs_and_progress(
                    &video_latents,
                    None,
                    &video_sigma,
                    &video_timestep,
                    None,
                    None,
                    cond_static_inputs,
                    None,
                    "Evaluating transformer (conditional)",
                    progress,
                )?;
            if ltx_debug_compare_uncond_enabled() && step_idx == 0 {
                if let Some(uncond_static_inputs) = uncond_static_inputs.as_ref() {
                    let (uncond_video_velocity, _) = transformer
                        .forward_with_static_inputs_and_progress(
                            &video_latents,
                            None,
                            &video_sigma,
                            &video_timestep,
                            None,
                            None,
                            uncond_static_inputs,
                            None,
                            "Evaluating transformer (debug unconditional)",
                            progress,
                        )?;
                    log_distilled_prompt_sensitivity(
                        debug_stage,
                        step_idx,
                        sigma,
                        &video_latents,
                        &cond_video_velocity,
                        &uncond_video_velocity,
                        None,
                        None,
                        None,
                    )?;
                }
            }
            if step_idx == 0 {
                if let Some(alt_static_inputs) = alt_static_inputs.as_ref() {
                    let (alt_video_velocity, _) = transformer
                        .forward_with_static_inputs_and_progress(
                            &video_latents,
                            None,
                            &video_sigma,
                            &video_timestep,
                            None,
                            None,
                            alt_static_inputs,
                            None,
                            "Evaluating transformer (debug alternate prompt)",
                            progress,
                        )?;
                    log_distilled_alternate_prompt_sensitivity(
                        debug_stage,
                        step_idx,
                        sigma,
                        &video_latents,
                        &cond_video_velocity,
                        &alt_video_velocity,
                        None,
                        None,
                        None,
                    )?;
                }
            }
            (
                denoised_from_velocity(&video_latents, &cond_video_velocity, sigma)?,
                None,
                Some(cond_video_velocity),
            )
        };
        transformer_secs += transformer_start.elapsed().as_secs_f64();
        let update_start = Instant::now();
        // Keep the hot denoise loop fully device-side unless step-level debug
        // inspection is explicitly enabled.
        if should_inspect_step_velocity(debug_stage) {
            let stage =
                debug_stage.expect("debug stage should be present when inspection is enabled");
            let video_velocity = video_velocity
                .as_ref()
                .context("video velocity missing for debug inspection")?;
            let video_velocity = video_velocity.to_dtype(DType::F32)?;
            log_tensor_stats("video_velocity", &video_velocity)?;
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
        }
        if uses_video_freeze_mask {
            video_denoised = blend_conditioned_denoised(
                &video_denoised,
                &clean_video_latents,
                &video_denoise_mask,
            )?;
        }
        video_latents = sampler_step(
            sampler_mode,
            &video_latents,
            &video_denoised,
            &run_sigmas,
            step_idx,
            video_sampler_noise.as_ref(),
            "video sampler noise missing for Res2S stage",
        )?;
        if !video_conditioning.is_empty() {
            video_latents = reapply_stage_video_conditioning(
                &video_latents,
                base_video_token_count,
                video_conditioning,
            )?;
        }

        if let (Some(audio_latents), Some(audio_velocity)) =
            (audio_latents.as_mut(), audio_denoised.as_ref())
        {
            let audio_velocity = if let (Some(clean_audio_latents), Some(audio_denoise_mask)) =
                (clean_audio_latents.as_ref(), audio_denoise_mask.as_ref())
            {
                blend_conditioned_denoised(audio_velocity, clean_audio_latents, audio_denoise_mask)?
            } else {
                audio_velocity.clone()
            };
            *audio_latents = sampler_step(
                sampler_mode,
                audio_latents,
                &audio_velocity,
                &run_sigmas,
                step_idx,
                audio_sampler_noise.as_ref(),
                "audio sampler noise missing for Res2S stage",
            )?;
            if let Some(base_audio_token_count) = base_audio_token_count {
                *audio_latents = reapply_stage_audio_conditioning(
                    audio_latents,
                    base_audio_token_count,
                    audio_conditioning,
                )?;
            }
        }
        update_secs += update_start.elapsed().as_secs_f64();
        emit_denoise_progress(
            progress,
            step_idx + 1,
            run_sigmas.len() - 1,
            step_start.elapsed(),
        );
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }

        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
            log_tensor_stats("step_video_latents", &video_latents)?;
            if let Some(audio_latents) = audio_latents.as_ref() {
                log_tensor_stats("step_audio_latents", audio_latents)?;
            }
            log_tensor_stats("video_x0", &video_denoised)?;
            if let (Some(audio_latents), Some(audio_denoised)) =
                (audio_latents.as_ref(), audio_denoised.as_ref())
            {
                log_tensor_stats("audio_x0", audio_denoised)?;
                let audio_velocity = velocity_from_denoised(audio_latents, audio_denoised, sigma)?;
                log_tensor_stats("audio_velocity", &audio_velocity)?;
            }
        }
    }

    if let Some(token) = cancellation {
        token.checkpoint()?;
    }
    let video_latents = strip_appended_video_conditioning(&video_latents, base_video_token_count)?;
    let video_latents = video_patchifier.unpatchify(&video_latents, video_shape)?;
    let audio_latents = match (audio_latents, audio_shape) {
        (Some(latents), Some(shape)) => Some(audio_patchifier.unpatchify(
            &strip_appended_audio_conditioning(&latents, shape.frames)?,
            shape,
        )?),
        _ => None,
    };
    if debug_stage.is_some() {
        log_tensor_stats("final_patched_latents", &video_latents)?;
    }
    if device.is_cuda() {
        device.synchronize()?;
    }
    if let Some(timing_label) = timing_label {
        log_elapsed_secs(&format!("{timing_label}.step_setup_total"), step_setup_secs);
        log_elapsed_secs(
            &format!("{timing_label}.transformer_total"),
            transformer_secs,
        );
        log_elapsed_secs(&format!("{timing_label}.update_total"), update_secs);
    }
    Ok((video_latents, audio_latents))
}

/// Upstream's non-distilled step count for the audio schedule: 40 for LTX-2.0
/// (`ltx-pipelines/src/ltx_pipelines/utils/constants.py:39`) and 30 for
/// LTX-2.3 (`:76`).
fn t2a_default_steps(plan: &Ltx2GeneratePlan) -> u32 {
    if plan.preset.name == "ltx-2.3-22b" {
        30
    } else {
        40
    }
}

/// The step count a T2A run actually uses.
///
/// The LTX-2 family default is 8, tuned for the *distilled* video ladder.
/// T2A runs the plain flow-match scheduler, where 8 steps leave the audio
/// latents far from the data manifold and the vocoder renders hiss. Raise a
/// too-small request to the preset default and say so — a caller who asks for
/// more than the default keeps their number.
fn t2a_effective_steps(plan: &Ltx2GeneratePlan, progress: Option<&ProgressCallback>) -> u32 {
    let minimum = t2a_default_steps(plan);
    if plan.num_inference_steps >= minimum {
        return plan.num_inference_steps;
    }
    emit_info(
        progress,
        format!(
            "LTX-2 text-to-audio runs the non-distilled schedule; raising {} steps to {} \
             (the {} default). Request more steps explicitly to override.",
            plan.num_inference_steps, minimum, plan.preset.name
        ),
    );
    minimum
}

/// The per-guider batch layout for an audio-only guided step.
///
/// Same trick as [`StaticMultimodalGuidanceBatch`]: the conditional,
/// unconditional and STG-perturbed passes are stacked on the batch axis and
/// evaluated in one transformer call. There is no modality slot — a modality
/// pass isolates the audio↔video cross-attention, and audio-only has none.
struct StaticAudioGuidanceBatch {
    perturbations: BatchedPerturbationConfig,
    repeat_count: usize,
    cond_index: usize,
    uncond_index: Option<usize>,
    perturbed_index: Option<usize>,
    static_inputs: LtxPreparedModalityStatic,
}

fn prepare_static_audio_guidance_batch(
    transformer: &Ltx2AudioTransformerModel,
    audio_context: &Tensor,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    audio_positions: &Tensor,
    audio_guider: &MultiModalGuider,
) -> Result<StaticAudioGuidanceBatch> {
    let mut contexts = vec![audio_context.clone()];
    let mut masks = vec![cond_mask.cloned()];
    let mut perturbations = vec![PerturbationConfig::empty()];
    let cond_index = 0usize;
    let mut uncond_index = None;
    let mut perturbed_index = None;

    if audio_guider.do_unconditional_generation() {
        let negative_context = audio_guider
            .negative_context
            .as_ref()
            .context("missing unconditional audio context for text-to-audio guidance")?;
        contexts.push(negative_context.clone());
        masks.push(uncond_mask.cloned());
        perturbations.push(PerturbationConfig::empty());
        uncond_index = Some(perturbations.len() - 1);
    }
    if audio_guider.do_perturbed_generation() {
        contexts.push(audio_context.clone());
        masks.push(cond_mask.cloned());
        perturbations.push(PerturbationConfig::new(vec![Perturbation::new(
            PerturbationType::SkipAudioSelfAttention,
            Some(audio_guider.params.stg_blocks.clone()),
        )]));
        perturbed_index = Some(perturbations.len() - 1);
    }

    let repeat_count = perturbations.len();
    let batched_context = Tensor::cat(&contexts.iter().collect::<Vec<_>>(), 0)?;
    let batched_mask = cat_optional_batches(&masks)?;
    let batched_positions = repeat_batch(audio_positions, repeat_count)?;
    let static_inputs = transformer.prepare_static_inputs(
        &batched_context,
        batched_mask.as_ref(),
        None,
        &batched_positions,
    )?;

    Ok(StaticAudioGuidanceBatch {
        perturbations: BatchedPerturbationConfig::new(perturbations),
        repeat_count,
        cond_index,
        uncond_index,
        perturbed_index,
        static_inputs,
    })
}

fn audio_guided_denoise_step(
    transformer: &Ltx2AudioTransformerModel,
    audio_latents: &Tensor,
    static_batch: &StaticAudioGuidanceBatch,
    audio_sigma: &Tensor,
    audio_timestep: &Tensor,
    audio_guider: &MultiModalGuider,
    step_idx: usize,
    progress: Option<&ProgressCallback>,
) -> Result<Tensor> {
    let batch = audio_latents.dim(0)?;
    let batched_latents = repeat_batch(audio_latents, static_batch.repeat_count)?;
    let batched_sigma = repeat_batch(audio_sigma, static_batch.repeat_count)?;
    let batched_timestep = repeat_batch(audio_timestep, static_batch.repeat_count)?;

    let all_velocity = transformer.forward_with_static_inputs_and_progress(
        &batched_latents,
        &batched_sigma,
        &batched_timestep,
        &static_batch.static_inputs,
        Some(&static_batch.perturbations),
        "Evaluating audio transformer",
        progress,
    )?;

    let cond = denoised_from_velocity_with_sigma(
        audio_latents,
        &split_batch_chunk(&all_velocity, static_batch.cond_index, batch)?,
        audio_timestep,
    )?;
    if audio_guider.should_skip_step(step_idx) {
        return Ok(cond);
    }
    let uncond = match static_batch.uncond_index {
        Some(index) => denoised_from_velocity_with_sigma(
            audio_latents,
            &split_batch_chunk(&all_velocity, index, batch)?,
            audio_timestep,
        )?,
        None => cond.clone(),
    };
    let perturbed = match static_batch.perturbed_index {
        Some(index) => denoised_from_velocity_with_sigma(
            audio_latents,
            &split_batch_chunk(&all_velocity, index, batch)?,
            audio_timestep,
        )?,
        None => cond.clone(),
    };
    // `modality` collapses onto `cond`: with `modality_scale == 1.0` the term
    // is multiplied by zero, and audio-only has no isolated modality pass to
    // produce anyway.
    audio_guider.calculate(&cond, &uncond, &perturbed, &cond)
}

/// One audio-only denoise: the sibling of [`run_real_distilled_stage`] with
/// the video branch, the conditioning masks and the two-stage plumbing gone.
#[allow(clippy::too_many_arguments)]
fn run_real_audio_only_stage(
    transformer: &Ltx2AudioTransformerModel,
    audio_shape: AudioLatentShape,
    audio_start_latents: &Tensor,
    audio_positions: &Tensor,
    audio_context: &Tensor,
    uncond_audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    audio_guider_params: MultiModalGuiderParams,
    sigmas_no_terminal: &[f32],
    sampler_mode: SamplerMode,
    audio_sampler_noise: Option<&Tensor>,
    timing_label: Option<&str>,
    debug_stage: Option<&str>,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<Tensor> {
    let device = audio_start_latents.device().clone();
    let audio_patchifier = AudioPatchifier::new(
        LTX2_AUDIO_SAMPLE_RATE,
        LTX2_AUDIO_HOP_LENGTH,
        LTX2_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        true,
        0,
    );
    let mut run_sigmas = sigmas_no_terminal.to_vec();
    run_sigmas.push(0.0);
    if let Some(progress) = progress {
        progress(ProgressEvent::StageStart {
            name: format!("Denoising ({} steps)", run_sigmas.len().saturating_sub(1)),
        });
    }

    let mut audio_latents = audio_patchifier.patchify(audio_start_latents)?;
    let audio_sampler_noise = audio_sampler_noise
        .map(|noise| audio_patchifier.patchify(noise))
        .transpose()?;

    let audio_guider = MultiModalGuider::new(audio_guider_params, uncond_audio_context.cloned());
    let static_batch = prepare_static_audio_guidance_batch(
        transformer,
        audio_context,
        cond_mask,
        uncond_mask,
        audio_positions,
        &audio_guider,
    )?;

    let mut transformer_secs = 0.0;
    for (step_idx, sigma) in run_sigmas
        .iter()
        .copied()
        .take(run_sigmas.len().saturating_sub(1))
        .enumerate()
    {
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }
        let step_start = Instant::now();
        // Both stay rank-1 `[batch]`, matching the unmasked branch of
        // `timestep_from_sigma_and_mask`. A rank-2 timestep would make
        // `sigma_scale_for_sample` try to broadcast one value across the token
        // axis and fail on the first step. There is no conditioning here, so
        // there is no per-token denoise mask to carry either.
        let audio_sigma = Tensor::full(sigma, (audio_latents.dim(0)?,), &device)?;
        let audio_timestep = audio_sigma.clone();

        let transformer_start = Instant::now();
        let denoised = audio_guided_denoise_step(
            transformer,
            &audio_latents,
            &static_batch,
            &audio_sigma,
            &audio_timestep,
            &audio_guider,
            step_idx,
            progress,
        )?;
        transformer_secs += transformer_start.elapsed().as_secs_f64();

        audio_latents = sampler_step(
            sampler_mode,
            &audio_latents,
            &denoised,
            &run_sigmas,
            step_idx,
            audio_sampler_noise.as_ref(),
            "audio sampler noise missing for Res2S stage",
        )?;

        emit_denoise_progress(
            progress,
            step_idx + 1,
            run_sigmas.len() - 1,
            step_start.elapsed(),
        );
        if let Some(stage) = debug_stage {
            eprintln!("[ltx2-debug] {stage} step={step_idx} sigma={sigma:.6}");
            log_tensor_stats("step_audio_latents", &audio_latents)?;
            log_tensor_stats("audio_x0", &denoised)?;
        }
        if let Some(token) = cancellation {
            token.checkpoint()?;
        }
    }

    if let Some(token) = cancellation {
        token.checkpoint()?;
    }
    let audio_latents = audio_patchifier.unpatchify(&audio_latents, audio_shape)?;
    if device.is_cuda() {
        device.synchronize()?;
    }
    if let Some(timing_label) = timing_label {
        log_elapsed_secs(
            &format!("{timing_label}.transformer_total"),
            transformer_secs,
        );
    }
    Ok(audio_latents)
}

/// Load the `audio_*` half of an LTX-2 checkpoint as a standalone denoiser.
///
/// Always eager: the audio branch is roughly a quarter of the per-block
/// parameters and drops both cross-modal attentions, so the whole thing sits
/// in a few GB — block streaming would buy nothing and cost a host round-trip
/// per layer per step.
fn load_ltx2_audio_transformer(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    loras: &[LoraWeight],
    progress: Option<&ProgressCallback>,
) -> Result<Ltx2AudioTransformerModel> {
    let probe = PhaseVramProbe::enter_if("audio_transformer_load".to_string(), device.is_cuda());
    let result = (|| -> Result<Ltx2AudioTransformerModel> {
        let config = ltx2_video_transformer_config(plan);
        let lora_registry = super::lora::load_lora_registry(loras)?;
        let checkpoint_path = Path::new(&plan.checkpoint_path);
        let checkpoint_is_nvfp4 = super::nvfp4::checkpoint_is_nvfp4(checkpoint_path);
        let checkpoint_is_convrot =
            !checkpoint_is_nvfp4 && super::convrot::checkpoint_is_convrot_w4a4(checkpoint_path);
        let weight_index = Ltx2TransformerWeightIndex::read(checkpoint_path).ok();
        let checkpoint_is_fp8 =
            !checkpoint_is_nvfp4 && ltx2_checkpoint_is_fp8(plan, weight_index.as_ref());
        let vb = if checkpoint_is_nvfp4 {
            let backend = super::nvfp4::Ltx2Nvfp4Backend::from_path(checkpoint_path)?;
            VarBuilder::from_backend(Box::new(backend), compute_dtype(device), device.clone())
        } else if checkpoint_is_convrot {
            let backend =
                super::convrot::Ltx2ConvRotBackend::from_path_for_device(checkpoint_path, device)?;
            VarBuilder::from_backend(Box::new(backend), compute_dtype(device), device.clone())
        } else if checkpoint_is_fp8 {
            load_fp8_safetensors_with_callback(
                std::slice::from_ref(&checkpoint_path),
                device,
                "LTX-2 audio transformer",
                progress,
            )?
        } else {
            let dtype = transformer_weight_dtype(plan, device);
            load_safetensors_with_progress_callback(
                std::slice::from_ref(&checkpoint_path),
                dtype,
                device,
                "LTX-2 audio transformer",
                progress,
            )?
        };
        let vb = if checkpoint_is_nvfp4 || checkpoint_is_convrot {
            vb
        } else {
            vb.rename_f(remap_ltx2_transformer_key)
        };
        Ok(Ltx2AudioTransformerModel::new(&config, vb, lora_registry)?)
    })();
    log_ltx2_phase_vram_result(probe.finish(), &result, None, "");
    result
}

/// Render an audio-only (text-to-audio) request end to end.
pub(crate) fn render_real_t2a_audio(
    plan: &Ltx2GeneratePlan,
    prepared: &NativePreparedRun,
    device: &candle_core::Device,
    progress: Option<&ProgressCallback>,
    cancellation: Option<&InferenceCancellationToken>,
) -> Result<NativeAudioTrack> {
    if let Some(overrides) = plan.guidance_overrides.as_ref() {
        if let Some(modality_scale) = overrides.modality_scale {
            if (modality_scale - 1.0).abs() > f64::EPSILON {
                bail!(
                    "guidance_overrides.modality_scale must be 1.0 for the LTX-2 text-to-audio \
                     pipeline (got {modality_scale}): audio-only generation has no video \
                     modality for cross-modal guidance to act on"
                );
            }
        }
    }
    let debug_enabled = ltx_debug_enabled();
    let prompt_inputs = prepare_render_prompt_inputs(
        prepared,
        device,
        RenderPromptInputOptions {
            include_unconditional: true,
            include_alt: false,
        },
    )?;
    let audio_shape = prompt_inputs.audio_shape.context(
        "LTX-2 text-to-audio requires audio latents, but the prepared run produced none",
    )?;
    let audio_positions = prompt_inputs
        .audio_positions
        .as_ref()
        .context("LTX-2 text-to-audio requires audio positions")?;
    let audio_context = prompt_inputs.audio_context.as_ref().context(
        "LTX-2 text-to-audio requires audio prompt conditioning; this checkpoint's text encoder \
         produced none",
    )?;

    let noise = seeded_randn(
        plan.seed ^ 0x4155_4449_4f4c_5458,
        &[
            audio_shape.batch,
            audio_shape.channels,
            audio_shape.frames,
            audio_shape.mel_bins,
        ],
        device,
        DType::F32,
    )?;
    if debug_enabled {
        log_tensor_stats("audio_context", audio_context)?;
        log_tensor_stats("initial_audio_latents", &noise)?;
    }

    let (_, audio_guider_params) = stage_multimodal_guider_params(plan, 0)?
        .context("LTX-2 text-to-audio requires multimodal guider parameters")?;
    let steps = t2a_effective_steps(plan, progress);
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(ltx2_scheduler_config())?;
    scheduler.set_timesteps(Some(steps as usize), device, None, None, None)?;
    let sigmas = scheduler
        .sigmas()
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    let sigmas_no_terminal = &sigmas[..sigmas.len().saturating_sub(1)];

    let dtype = compute_dtype(device);
    let stage_loras = stage_lora_stack(plan, 0)?;
    let transformer = load_ltx2_audio_transformer(plan, device, &stage_loras, progress)?;
    if debug_enabled {
        log_debug_vram("after_t2a_transformer_load");
    }

    let audio_latents = run_real_audio_only_stage(
        &transformer,
        audio_shape,
        &noise,
        audio_positions,
        audio_context,
        prompt_inputs.uncond_audio_context.as_ref(),
        None,
        None,
        audio_guider_params,
        sigmas_no_terminal,
        stage_sampler_mode(plan, 0)?,
        None,
        Some("t2a"),
        debug_enabled.then_some("t2a"),
        progress,
        cancellation,
    )?;
    drop(transformer);
    device.synchronize()?;
    if debug_enabled {
        log_debug_vram("after_t2a_transformer_drop");
        log_tensor_stats("final_audio_latents", &audio_latents)?;
    }

    let track = render_native_audio_track(plan, &audio_latents, device, dtype)?.context(
        "LTX-2 text-to-audio produced an empty waveform; the checkpoint's vocoder returned no \
         samples",
    )?;
    drop(audio_latents);
    drop(noise);
    drop(prompt_inputs);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(track)
}

fn build_video_positions(
    pixel_shape: VideoPixelShape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let scale_factors = SpatioTemporalScaleFactors::default();
    let latent_shape =
        VideoLatentShape::from_pixel_shape(pixel_shape, LTX2_VIDEO_LATENT_CHANNELS, scale_factors);
    let video_patchifier = VideoLatentPatchifier::new(1);
    scale_video_time_to_seconds(
        &get_pixel_coords(
            &video_token_positions(video_patchifier, latent_shape, device)?,
            scale_factors,
            true,
        )?,
        pixel_shape.fps,
    )
}

fn mix_clean_latents_with_noise(
    clean_latents: &Tensor,
    noise: &Tensor,
    noise_scale: f32,
) -> Result<Tensor> {
    let noise_scale = noise_scale as f64;
    let clean_scale = 1.0 - noise_scale;
    clean_latents
        .affine(clean_scale, 0.0)?
        .broadcast_add(&noise.affine(noise_scale, 0.0)?)
        .map_err(Into::into)
}

fn should_inspect_step_velocity(debug_stage: Option<&str>) -> bool {
    debug_stage.is_some()
}

/// Fork the decoded VAE tensor into scene-referred linear HDR.
///
/// `decoded_video_to_frames` is the single float→u8 conversion in the whole
/// LTX-2 path, so this is the one place HDR can diverge — everything past it
/// is `RgbImage`. Keeps full float precision and does not resize: an EXR is
/// for compositing, and resampling linear light after the grade defeats it.
/// The EXR sidecar target for this plan, if one was requested.
///
/// Resolved once per decode so the four render paths cannot disagree about
/// where the sequence goes or at what precision.
fn plan_hdr_exr_target(plan: &Ltx2GeneratePlan) -> Option<HdrExrTarget> {
    let dir = plan.hdr_exr_dir.as_deref()?;
    let precision = if plan.hdr_exr_full_float {
        crate::ltx2::exr::ExrPrecision::Full
    } else {
        crate::ltx2::exr::ExrPrecision::Half
    };
    Some(HdrExrTarget {
        dir: std::path::PathBuf::from(dir),
        precision,
        window: plan.hdr_exr_window,
    })
}

/// Resolved EXR sidecar destination for one decode: where the frames go, at
/// what precision, and — for chain stages — which window of the decoded clip
/// lands at which global indices. `window: None` is the single-render
/// identity (every decoded frame, numbered from zero).
#[derive(Debug, Clone)]
pub(crate) struct HdrExrTarget {
    pub(crate) dir: std::path::PathBuf,
    pub(crate) precision: crate::ltx2::exr::ExrPrecision,
    pub(crate) window: Option<crate::chain::ExrStageWindow>,
}

/// Convert the decoded tensor to scene-referred linear HDR and write it out
/// one frame at a time, returning how many frames landed.
///
/// Deliberately streaming rather than returning a `Vec<HdrFrame>`. Each frame
/// holds `width * height * 3` `f32` samples, so buffering the clip scales with
/// its length: 25 MB per frame at 1920x1088, which is 3 GB for a 5-second
/// render and 12 GB at LTX-2's 20-second ceiling — on a card that is already
/// holding the model. Only one frame is live at a time here.
fn write_hdr_frames_streaming(video: &Tensor, target: &HdrExrTarget) -> Result<usize> {
    let dir = target.dir.as_path();
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create HDR EXR directory '{}'", dir.display()))?;
    let video = video.to_dtype(DType::F32)?.i(0)?;
    let count = video.dim(1)?;
    // A chain stage writes only its window: local frames
    // `[skip, skip + write_count)` land at global indices
    // `[start, start + write_count)`. The identity (no window) keeps the
    // single-render behaviour bit-for-bit: every frame, numbered from zero.
    // Skipped frames are never materialized, so peak memory stays at one
    // frame either way.
    let (skip, start, limit) = match target.window {
        Some(window) => (
            window.skip_leading as usize,
            window.start_index as usize,
            window.write_count as usize,
        ),
        None => (0, 0, count),
    };
    let end = count.min(skip.saturating_add(limit));
    let mut written = 0usize;
    for index in skip..end {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let (height, width, channels) = frame.dims3()?;
        if channels != 3 {
            anyhow::bail!("expected decoded LTX-2 frame to have 3 channels, got {channels}");
        }
        let samples: Vec<f32> = frame.flatten_all()?.to_vec1()?;
        let rgb = samples
            .into_iter()
            .map(crate::ltx2::hdr::vae_output_to_linear_hdr)
            .collect();
        let hdr_frame = crate::ltx2::exr::HdrFrame { width, height, rgb };
        crate::ltx2::exr::write_exr_frame(
            &crate::ltx2::exr::exr_frame_path(dir, start + (index - skip)),
            &hdr_frame,
            target.precision,
        )?;
        written += 1;
    }
    Ok(written)
}

fn decoded_video_to_frames(video: &Tensor, pixel_shape: VideoPixelShape) -> Result<Vec<RgbImage>> {
    let video =
        ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let video = video.i(0)?;

    let mut frames = Vec::with_capacity(video.dim(1)?);
    for index in 0..video.dim(1)? {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let (decoded_height, decoded_width, decoded_channels) = frame.dims3()?;
        if decoded_channels != 3 {
            anyhow::bail!(
                "expected decoded LTX-2 frame to have 3 channels, got {decoded_channels}"
            );
        }
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let rgb = RgbImage::from_raw(decoded_width as u32, decoded_height as u32, data)
            .context("failed to build an RGB frame from the decoded LTX-2 tensor")?;
        let rgb = if decoded_width != pixel_shape.width || decoded_height != pixel_shape.height {
            imageops::resize(
                &rgb,
                pixel_shape.width as u32,
                pixel_shape.height as u32,
                imageops::FilterType::Triangle,
            )
        } else {
            rgb
        };
        frames.push(rgb);
    }
    Ok(frames)
}

fn maybe_render_native_audio_track(
    plan: &Ltx2GeneratePlan,
    audio_latents: Option<&Tensor>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeAudioTrack>> {
    if !plan.execution_graph.wants_audio_output {
        return Ok(None);
    }
    let audio_latents = audio_latents.context(
        "native LTX-2 audio output requested but the denoiser produced no audio latents",
    )?;
    render_native_audio_track(plan, audio_latents, device, dtype)
}

/// Audio VAE → vocoder → interleaved f32 samples. Shared by the joint AV
/// pipelines (which attach the result as an MP4 track) and by T2A (where it
/// is the entire artifact).
fn render_native_audio_track(
    plan: &Ltx2GeneratePlan,
    audio_latents: &Tensor,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeAudioTrack>> {
    let audio_checkpoint = plan
        .audio_components_path
        .as_deref()
        .unwrap_or(&plan.checkpoint_path);
    let decoder =
        Ltx2AudioDecoder::load_from_checkpoint(Path::new(audio_checkpoint), dtype, device)?;
    let mel_spec = decoder.decode(&audio_latents.to_dtype(dtype)?)?;
    drop(decoder);
    if device.is_cuda() {
        device.synchronize()?;
    }

    let vocoder = Ltx2VocoderWithBwe::load_from_checkpoint(Path::new(audio_checkpoint), device)?;
    let output_sample_rate = vocoder.config.output_sample_rate as u32;
    let waveform = vocoder.forward(&mel_spec.to_dtype(DType::F32)?)?;
    drop(vocoder);
    drop(mel_spec);
    if device.is_cuda() {
        device.synchronize()?;
    }
    waveform_to_audio_track(&waveform, output_sample_rate)
}

/// Resize `tail_rgb_frames` to the current stage's `pixel_shape` so the
/// VAE encodes them onto the grid stage 1 or stage 2 expects. Stage 1 of
/// the distilled pipeline runs at an implicitly X2-downsampled resolution
/// (see `derive_stage1_render_shape`), while the chain carryover tail is
/// always captured at the emitting stage's final decoded resolution. Without
/// this resize the staged-latent replacement path produces a token count on
/// a full-res grid that overflows stage 1's half-res grid.
///
/// No-op when the source dimensions already match the target (stage 2 case).
fn resize_tail_frames_to_pixel_shape(
    tail_rgb_frames: &[RgbImage],
    target_width: u32,
    target_height: u32,
) -> Vec<RgbImage> {
    tail_rgb_frames
        .iter()
        .map(|frame| {
            if frame.width() == target_width && frame.height() == target_height {
                frame.clone()
            } else {
                imageops::resize(
                    frame,
                    target_width,
                    target_height,
                    imageops::FilterType::Lanczos3,
                )
            }
        })
        .collect()
}

fn video_tensor_from_frames(
    frames: &[RgbImage],
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let first = frames
        .first()
        .context("native LTX-2 source video conditioning requires at least one frame")?;
    let width = first.width() as usize;
    let height = first.height() as usize;
    let frame_count = frames.len();
    let mut data = Vec::with_capacity(frame_count * width * height * 3);

    for channel in 0..3usize {
        for frame in frames {
            if frame.width() as usize != width || frame.height() as usize != height {
                anyhow::bail!("native LTX-2 source video frames do not share a common size");
            }
            for pixel in frame.pixels() {
                data.push((pixel[channel] as f32 / 127.5) - 1.0);
            }
        }
    }

    Tensor::from_vec(data, (1, 3, frame_count, height, width), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn conform_video_latent_length(
    latents: &Tensor,
    expected_shape: VideoLatentShape,
) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = latents.dims5()?;
    if batch != expected_shape.batch
        || channels != expected_shape.channels
        || height != expected_shape.height
        || width != expected_shape.width
    {
        anyhow::bail!(
            "native LTX-2 source video latent shape mismatch: got [{batch}, {channels}, {frames}, {height}, {width}], expected [{}, {}, {}, {}, {}]",
            expected_shape.batch,
            expected_shape.channels,
            expected_shape.frames,
            expected_shape.height,
            expected_shape.width
        );
    }
    if frames == expected_shape.frames {
        return Ok(latents.clone());
    }
    if frames > expected_shape.frames {
        return latents
            .narrow(2, 0, expected_shape.frames)
            .map_err(Into::into);
    }
    let pad_frames = expected_shape.frames - frames;
    let pad = Tensor::zeros(
        (batch, channels, pad_frames, height, width),
        latents.dtype(),
        latents.device(),
    )?;
    Tensor::cat(&[latents, &pad], 2).map_err(Into::into)
}

fn maybe_load_native_conditioning_video(
    plan: &Ltx2GeneratePlan,
    pixel_shape: VideoPixelShape,
    latent_shape: VideoLatentShape,
    device: &candle_core::Device,
    dtype: DType,
    progress: Option<&ProgressCallback>,
) -> Result<Option<NativeConditioningVideo>> {
    let Some(video_path) = plan.conditioning.video_path.as_ref() else {
        return Ok(None);
    };
    let (metadata, frames) = media::decode_video_frames(Path::new(video_path))?;
    if metadata.fps != pixel_shape.fps.round() as u32 {
        anyhow::bail!(
            "native LTX-2 source video FPS mismatch: source={} expected={}",
            metadata.fps,
            pixel_shape.fps.round() as u32
        );
    }
    let resized = frames
        .into_iter()
        .map(|frame| {
            if frame.width() == pixel_shape.width as u32
                && frame.height() == pixel_shape.height as u32
            {
                frame
            } else {
                imageops::resize(
                    &frame,
                    pixel_shape.width as u32,
                    pixel_shape.height as u32,
                    imageops::FilterType::Lanczos3,
                )
            }
        })
        .collect::<Vec<_>>();
    let video = video_tensor_from_frames(&resized, device, dtype)?;
    let vae = load_ltx2_video_vae(plan, device, dtype, progress)?;
    let latents = conform_video_latent_length(&vae.encode(&video)?, latent_shape)?;
    drop(vae);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(Some(NativeConditioningVideo { latents }))
}

/// Encode the lip-dub reference clip's own soundtrack into audio VAE latents.
///
/// Deliberately *not* [`maybe_load_native_conditioning_audio`]: that path
/// conforms the latent to the render's audio shape and keeps the decoded track
/// as the output soundtrack, both of which are wrong here. These latents are
/// reference tokens at whatever natural length the clip has — upstream encodes
/// the full stream with no duration cap and no conforming
/// (`lipdub.py:166-171`) — and the exported audio is the *generated* dub.
fn load_lip_dub_reference_audio_latents(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let video_path = plan
        .conditioning
        .video_path
        .as_ref()
        .context("the LTX-2 lip-dub pipeline requires a reference video")?;
    let decoded_audio =
        DecodedAudio::from_file(Path::new(video_path), None)?.with_context(|| {
            format!(
                "lip-dub reference video '{video_path}' has no audio stream; the pipeline \
                 re-voices existing speech, so the reference must contain some"
            )
        })?;
    let encoder =
        Ltx2AudioEncoder::load_from_checkpoint(Path::new(&plan.checkpoint_path), dtype, device)?;
    let latents = encoder.encode_audio(&decoded_audio)?;
    drop(encoder);
    if device.is_cuda() {
        device.synchronize()?;
    }
    Ok(latents)
}

fn maybe_load_native_conditioning_audio(
    plan: &Ltx2GeneratePlan,
    audio_shape: Option<AudioLatentShape>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Option<NativeConditioningAudio>> {
    let explicit_audio_path = plan.conditioning.audio_path.as_ref();
    let audio_path = explicit_audio_path.or({
        if plan.execution_graph.uses_retake_masking {
            plan.conditioning.video_path.as_ref()
        } else {
            None
        }
    });
    let Some(audio_path) = audio_path else {
        return Ok(None);
    };
    let audio_shape = audio_shape.context(
        "native LTX-2 audio conditioning requested but the prepared run has no audio latent shape",
    )?;
    let max_duration = plan.num_frames as f32 / plan.frame_rate.max(1) as f32;
    let decoded_audio = match DecodedAudio::from_file(Path::new(audio_path), Some(max_duration))? {
        Some(decoded_audio) => decoded_audio,
        None if explicit_audio_path.is_none() && plan.execution_graph.uses_retake_masking => {
            return Ok(None);
        }
        None => {
            return Err(anyhow::anyhow!(
                "source audio '{}' did not contain a decodable audio stream",
                audio_path
            ));
        }
    };
    let encoder =
        Ltx2AudioEncoder::load_from_checkpoint(Path::new(&plan.checkpoint_path), dtype, device)?;
    let latents = conform_audio_latent_length(&encoder.encode_audio(&decoded_audio)?, audio_shape)?;
    drop(encoder);
    if device.is_cuda() {
        device.synchronize()?;
    }
    let original_track = if plan.execution_graph.wants_audio_output {
        native_audio_track_from_decoded_audio(&decoded_audio)?
    } else {
        None
    };
    Ok(Some(NativeConditioningAudio {
        latents,
        original_track,
    }))
}

fn build_temporal_token_denoise_mask(
    range: &TimeRange,
    positions: &Tensor,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let temporal = positions
        .i((.., 0, .., ..))?
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let (batch, tokens, _) = temporal.dims3()?;
    let mut values = Vec::with_capacity(batch * tokens);
    for batch_item in temporal.to_vec3::<f32>()? {
        for bounds in batch_item {
            let start = bounds.first().copied().unwrap_or_default();
            let end = bounds.get(1).copied().unwrap_or(start);
            let active = end > range.start_seconds && start < range.end_seconds;
            values.push(if active { 1.0f32 } else { 0.0f32 });
        }
    }
    Tensor::from_vec(values, (batch, tokens), device).map_err(Into::into)
}

fn conform_audio_latent_length(
    latents: &Tensor,
    expected_shape: AudioLatentShape,
) -> Result<Tensor> {
    let (batch, channels, frames, mel_bins) = latents.dims4()?;
    if batch != expected_shape.batch
        || channels != expected_shape.channels
        || mel_bins != expected_shape.mel_bins
    {
        anyhow::bail!(
            "native LTX-2 source audio latent shape mismatch: got [{batch}, {channels}, {frames}, {mel_bins}], expected [{}, {}, {}, {}]",
            expected_shape.batch,
            expected_shape.channels,
            expected_shape.frames,
            expected_shape.mel_bins
        );
    }
    if frames == expected_shape.frames {
        return Ok(latents.clone());
    }
    if frames > expected_shape.frames {
        return latents
            .narrow(2, 0, expected_shape.frames)
            .map_err(Into::into);
    }
    let pad_frames = expected_shape.frames - frames;
    let pad = Tensor::zeros(
        (batch, channels, pad_frames, mel_bins),
        latents.dtype(),
        latents.device(),
    )?;
    Tensor::cat(&[latents, &pad], 2).map_err(Into::into)
}

fn build_frozen_audio_denoise_mask(
    audio_shape: AudioLatentShape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    Tensor::zeros((audio_shape.batch, audio_shape.frames), DType::F32, device).map_err(Into::into)
}

fn timestep_from_sigma_and_mask(
    sigma: f32,
    batch_size: usize,
    denoise_mask: Option<&Tensor>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let sigma_tensor = Tensor::full(sigma, (batch_size,), device)?;
    match denoise_mask {
        Some(mask) => mask.affine(sigma as f64, 0.0).map_err(Into::into),
        None => Ok(sigma_tensor),
    }
}

fn native_audio_track_from_decoded_audio(
    decoded_audio: &DecodedAudio,
) -> Result<Option<NativeAudioTrack>> {
    let channels = decoded_audio.channel_count();
    let samples_per_channel = decoded_audio.sample_count();
    if channels == 0 || samples_per_channel == 0 {
        return Ok(None);
    }
    let mut interleaved_samples = Vec::with_capacity(channels * samples_per_channel);
    for sample_idx in 0..samples_per_channel {
        for channel in &decoded_audio.channels {
            interleaved_samples.push(channel[sample_idx]);
        }
    }
    Ok(Some(NativeAudioTrack {
        interleaved_samples,
        sample_rate: decoded_audio.sample_rate as u32,
        channels: channels as u16,
    }))
}

fn waveform_to_audio_track(
    waveform: &Tensor,
    sample_rate: u32,
) -> Result<Option<NativeAudioTrack>> {
    let waveform = waveform
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let (batch, channels, samples_per_channel) = waveform.dims3()?;
    if batch == 0 || channels == 0 || samples_per_channel == 0 {
        return Ok(None);
    }
    let channel_vectors = waveform.i(0)?.to_vec2::<f32>()?;
    let mut interleaved_samples = Vec::with_capacity(channels * samples_per_channel);
    for sample_idx in 0..samples_per_channel {
        for channel in &channel_vectors {
            interleaved_samples.push(channel[sample_idx]);
        }
    }
    Ok(Some(NativeAudioTrack {
        interleaved_samples,
        sample_rate,
        channels: channels as u16,
    }))
}

fn maybe_write_debug_stage_video(
    stage: &str,
    vae: &mut AutoencoderKLLtx2Video,
    latents: &Tensor,
    pixel_shape: VideoPixelShape,
    dtype: DType,
) -> Result<()> {
    let Some(prefix) = env::var_os("MOLD_LTX2_DEBUG_STAGE_PREFIX") else {
        return Ok(());
    };

    let decode_latents = latents.to_dtype(dtype)?;
    configure_ltx2_vae_decode_memory_mode(vae, &decode_latents, decode_latents.device())?;
    let (_decoded, video) = vae.decode(&decode_latents, None, false, false)?;
    let frames = decoded_video_to_frames(&video, pixel_shape)?;
    let prefix = prefix.to_string_lossy();
    let first_frame_path = std::path::PathBuf::from(format!("{prefix}-{stage}-first-frame.png"));
    let contact_sheet_path =
        std::path::PathBuf::from(format!("{prefix}-{stage}-contact-sheet.png"));
    if let Some(first) = frames.first() {
        first.save(&first_frame_path)?;
    }
    write_contact_sheet_from_frames(&frames, &contact_sheet_path)?;
    eprintln!(
        "[ltx2-debug] wrote stage video: stage={stage} first_frame={} contact_sheet={}",
        first_frame_path.display(),
        contact_sheet_path.display()
    );
    Ok(())
}

fn write_contact_sheet_from_frames(
    frames: &[RgbImage],
    output_png: &std::path::Path,
) -> Result<()> {
    if frames.is_empty() {
        return Ok(());
    }

    let columns = 3usize;
    let rows = frames.len().div_ceil(columns);
    let frame_width = frames[0].width();
    let frame_height = frames[0].height();
    let mut sheet = RgbImage::from_pixel(
        frame_width * columns as u32,
        frame_height * rows as u32,
        Rgb([0, 0, 0]),
    );

    for (index, frame) in frames.iter().enumerate() {
        let x = (index % columns) as u32 * frame_width;
        let y = (index / columns) as u32 * frame_height;
        sheet.copy_from(frame, x, y)?;
    }

    sheet.save(output_png)?;
    Ok(())
}

fn repeat_batch(tensor: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats <= 1 {
        return Ok(tensor.clone());
    }
    let parts = (0..repeats).map(|_| tensor.clone()).collect::<Vec<_>>();
    let refs = parts.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

fn cat_optional_batches(parts: &[Option<Tensor>]) -> Result<Option<Tensor>> {
    if parts.iter().all(Option::is_none) {
        return Ok(None);
    }
    if !parts.iter().all(Option::is_some) {
        anyhow::bail!("batched optional tensors must be either all present or all absent");
    }
    let tensors = parts.iter().flatten().collect::<Vec<_>>();
    Tensor::cat(&tensors, 0).map(Some).map_err(Into::into)
}

struct StaticMultimodalGuidanceBatch {
    batched_video_context: Tensor,
    batched_audio_context: Option<Tensor>,
    batched_video_mask: Option<Tensor>,
    batched_audio_mask: Option<Tensor>,
    perturbations: BatchedPerturbationConfig,
    repeat_count: usize,
    cond_index: usize,
    uncond_index: Option<usize>,
    perturbed_index: Option<usize>,
    modality_index: Option<usize>,
    static_inputs: Option<LtxPreparedStaticInputs>,
}

fn build_static_multimodal_guidance_batch(
    cond_context: &Tensor,
    audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    video_guider: &MultiModalGuider,
    audio_guider: &MultiModalGuider,
) -> Result<StaticMultimodalGuidanceBatch> {
    let mut video_contexts = vec![cond_context.clone()];
    let mut audio_contexts = vec![audio_context.cloned()];
    let mut video_masks = vec![cond_mask.cloned()];
    let mut audio_masks = vec![cond_mask.cloned()];
    let mut perturbations = vec![PerturbationConfig::empty()];
    let cond_index = 0usize;
    let mut uncond_index = None;
    let mut perturbed_index = None;
    let mut modality_index = None;

    if video_guider.do_unconditional_generation() || audio_guider.do_unconditional_generation() {
        let negative_video_context = video_guider
            .negative_context
            .as_ref()
            .context("missing unconditional video context for multimodal guidance")?;
        video_contexts.push(negative_video_context.clone());
        audio_contexts.push(
            audio_guider
                .negative_context
                .clone()
                .or_else(|| audio_context.cloned()),
        );
        video_masks.push(uncond_mask.cloned());
        audio_masks.push(uncond_mask.cloned());
        perturbations.push(PerturbationConfig::empty());
        uncond_index = Some(perturbations.len() - 1);
    }

    if video_guider.do_perturbed_generation() || audio_guider.do_perturbed_generation() {
        let mut stg_perturbations = Vec::new();
        if video_guider.do_perturbed_generation() {
            stg_perturbations.push(Perturbation::new(
                PerturbationType::SkipVideoSelfAttention,
                Some(video_guider.params.stg_blocks.clone()),
            ));
        }
        if audio_guider.do_perturbed_generation() {
            stg_perturbations.push(Perturbation::new(
                PerturbationType::SkipAudioSelfAttention,
                Some(audio_guider.params.stg_blocks.clone()),
            ));
        }
        video_contexts.push(cond_context.clone());
        audio_contexts.push(audio_context.cloned());
        video_masks.push(cond_mask.cloned());
        audio_masks.push(cond_mask.cloned());
        perturbations.push(PerturbationConfig::new(stg_perturbations));
        perturbed_index = Some(perturbations.len() - 1);
    }
    if video_guider.do_isolated_modality_generation()
        || audio_guider.do_isolated_modality_generation()
    {
        video_contexts.push(cond_context.clone());
        audio_contexts.push(audio_context.cloned());
        video_masks.push(cond_mask.cloned());
        audio_masks.push(cond_mask.cloned());
        perturbations.push(PerturbationConfig::new(vec![
            Perturbation::new(PerturbationType::SkipA2VCrossAttention, None),
            Perturbation::new(PerturbationType::SkipV2ACrossAttention, None),
        ]));
        modality_index = Some(perturbations.len() - 1);
    }

    let repeat_count = perturbations.len();
    let batched_video_context = Tensor::cat(&video_contexts.iter().collect::<Vec<_>>(), 0)?;
    let batched_audio_context = cat_optional_batches(&audio_contexts)?;
    let batched_video_mask = cat_optional_batches(&video_masks)?;
    let batched_audio_mask = cat_optional_batches(&audio_masks)?;
    Ok(StaticMultimodalGuidanceBatch {
        batched_video_context,
        batched_audio_context,
        batched_video_mask,
        batched_audio_mask,
        perturbations: BatchedPerturbationConfig::new(perturbations),
        repeat_count,
        cond_index,
        uncond_index,
        perturbed_index,
        modality_index,
        static_inputs: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_static_multimodal_guidance_batch(
    transformer: &Ltx2AvTransformer3DModel,
    cond_context: &Tensor,
    audio_context: Option<&Tensor>,
    cond_mask: Option<&Tensor>,
    uncond_mask: Option<&Tensor>,
    video_self_attention_mask: Option<&Tensor>,
    audio_self_attention_mask: Option<&Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    video_guider: &MultiModalGuider,
    audio_guider: &MultiModalGuider,
) -> Result<StaticMultimodalGuidanceBatch> {
    let mut batch = build_static_multimodal_guidance_batch(
        cond_context,
        audio_context,
        cond_mask,
        uncond_mask,
        video_guider,
        audio_guider,
    )?;
    let batched_video_self_attention_mask = video_self_attention_mask
        .map(|mask| repeat_batch(mask, batch.repeat_count))
        .transpose()?;
    let batched_audio_self_attention_mask = audio_self_attention_mask
        .map(|mask| repeat_batch(mask, batch.repeat_count))
        .transpose()?;
    let batched_video_positions = repeat_batch(video_positions, batch.repeat_count)?;
    let batched_audio_positions = audio_positions
        .map(|positions| repeat_batch(positions, batch.repeat_count))
        .transpose()?;
    let static_inputs = transformer.prepare_static_inputs(
        &batch.batched_video_context,
        batch.batched_audio_context.as_ref(),
        batch.batched_video_mask.as_ref(),
        batch.batched_audio_mask.as_ref(),
        batched_video_self_attention_mask.as_ref(),
        batched_audio_self_attention_mask.as_ref(),
        &batched_video_positions,
        batched_audio_positions.as_ref(),
    )?;
    batch.static_inputs = Some(static_inputs);
    Ok(batch)
}

fn split_batch_chunk(tensor: &Tensor, index: usize, chunk: usize) -> Result<Tensor> {
    tensor.narrow(0, index * chunk, chunk).map_err(Into::into)
}

fn sigma_scale_for_sample(sample: &Tensor, sigma: &Tensor) -> Result<Tensor> {
    match sigma.rank() {
        1 => sigma
            .reshape((sample.dim(0)?, 1, 1))?
            .to_device(sample.device())?
            .to_dtype(sample.dtype())
            .map_err(Into::into),
        2 => sigma
            .reshape((sample.dim(0)?, sample.dim(1)?, 1))?
            .to_device(sample.device())?
            .to_dtype(sample.dtype())
            .map_err(Into::into),
        other => anyhow::bail!("expected sigma rank 1 or 2, got rank {other}"),
    }
}

fn denoised_from_velocity_with_sigma(
    sample: &Tensor,
    velocity: &Tensor,
    sigma: &Tensor,
) -> Result<Tensor> {
    let sigma = sigma_scale_for_sample(sample, sigma)?;
    let velocity = if velocity.dtype() == sample.dtype() {
        velocity.clone()
    } else {
        velocity.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&velocity.broadcast_mul(&sigma)?)
        .map_err(Into::into)
}

fn multimodal_guided_denoise_step(
    transformer: &Ltx2AvTransformer3DModel,
    video_latents: &Tensor,
    audio_latents: Option<&Tensor>,
    static_batch: &StaticMultimodalGuidanceBatch,
    video_sigma: &Tensor,
    video_timestep: &Tensor,
    audio_sigma: Option<&Tensor>,
    audio_timestep: Option<&Tensor>,
    video_guider: &MultiModalGuider,
    audio_guider: &MultiModalGuider,
    step_idx: usize,
    progress: Option<&ProgressCallback>,
) -> Result<(Tensor, Option<Tensor>)> {
    let video_skip = video_guider.should_skip_step(step_idx);
    let audio_skip = audio_guider.should_skip_step(step_idx);

    let batch = video_latents.dim(0)?;
    let batched_video_latents = repeat_batch(video_latents, static_batch.repeat_count)?;
    let batched_video_sigma = repeat_batch(video_sigma, static_batch.repeat_count)?;
    let batched_video_timestep = repeat_batch(video_timestep, static_batch.repeat_count)?;
    let batched_audio_latents = audio_latents
        .map(|latents| repeat_batch(latents, static_batch.repeat_count))
        .transpose()?;
    let batched_audio_sigma = audio_sigma
        .map(|sigma| repeat_batch(sigma, static_batch.repeat_count))
        .transpose()?;
    let batched_audio_timestep = audio_timestep
        .map(|timestep| repeat_batch(timestep, static_batch.repeat_count))
        .transpose()?;
    let static_inputs = static_batch
        .static_inputs
        .as_ref()
        .context("missing prepared static multimodal guidance inputs")?;

    let (all_video_velocity, all_audio_velocity) = transformer
        .forward_with_static_inputs_and_progress(
            &batched_video_latents,
            batched_audio_latents.as_ref(),
            &batched_video_sigma,
            &batched_video_timestep,
            batched_audio_sigma.as_ref(),
            batched_audio_timestep.as_ref(),
            static_inputs,
            Some(&static_batch.perturbations),
            "Evaluating transformer",
            progress,
        )?;

    let cond_video = denoised_from_velocity_with_sigma(
        video_latents,
        &split_batch_chunk(&all_video_velocity, static_batch.cond_index, batch)?,
        video_timestep,
    )?;
    let uncond_video = if let Some(index) = static_batch.uncond_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let perturbed_video = if let Some(index) = static_batch.perturbed_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let modality_video = if let Some(index) = static_batch.modality_index {
        denoised_from_velocity_with_sigma(
            video_latents,
            &split_batch_chunk(&all_video_velocity, index, batch)?,
            video_timestep,
        )?
    } else {
        cond_video.clone()
    };
    let video_denoised = if video_skip {
        cond_video.clone()
    } else {
        video_guider.calculate(
            &cond_video,
            &uncond_video,
            &perturbed_video,
            &modality_video,
        )?
    };

    let audio_denoised = match (audio_latents, all_audio_velocity.as_ref(), audio_timestep) {
        (Some(audio_latents), Some(all_audio_velocity), Some(audio_timestep)) => {
            let cond_audio = denoised_from_velocity_with_sigma(
                audio_latents,
                &split_batch_chunk(all_audio_velocity, static_batch.cond_index, batch)?,
                audio_timestep,
            )?;
            let uncond_audio = if let Some(index) = static_batch.uncond_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            let perturbed_audio = if let Some(index) = static_batch.perturbed_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            let modality_audio = if let Some(index) = static_batch.modality_index {
                denoised_from_velocity_with_sigma(
                    audio_latents,
                    &split_batch_chunk(all_audio_velocity, index, batch)?,
                    audio_timestep,
                )?
            } else {
                cond_audio.clone()
            };
            Some(if audio_skip {
                cond_audio
            } else {
                audio_guider.calculate(
                    &cond_audio,
                    &uncond_audio,
                    &perturbed_audio,
                    &modality_audio,
                )?
            })
        }
        _ => None,
    };

    Ok((video_denoised, audio_denoised))
}

fn convert_velocity_to_x0(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&velocity.to_dtype(DType::F32)?.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn convert_x0_to_velocity(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma.abs() <= f32::EPSILON {
        anyhow::bail!("cannot convert x0 to velocity at zero sigma");
    }
    sample
        .to_dtype(DType::F32)?
        .broadcast_sub(&denoised.to_dtype(DType::F32)?)?
        .affine(1.0 / sigma as f64, 0.0)
        .map_err(Into::into)
}

fn guided_velocity_from_cfg(
    sample: &Tensor,
    conditional_velocity: &Tensor,
    unconditional_velocity: &Tensor,
    sigma: f32,
    guidance_scale: f64,
) -> Result<Tensor> {
    if guidance_scale <= 1.0 {
        return Ok(conditional_velocity.clone());
    }
    let conditional_x0 = convert_velocity_to_x0(sample, conditional_velocity, sigma)?;
    let unconditional_x0 = convert_velocity_to_x0(sample, unconditional_velocity, sigma)?;
    let guidance_delta = conditional_x0
        .broadcast_sub(&unconditional_x0)?
        .affine(guidance_scale - 1.0, 0.0)?;
    let guided_x0 = conditional_x0.broadcast_add(&guidance_delta)?;
    convert_x0_to_velocity(sample, &guided_x0, sigma)
}

/// Load the transformer and record what the load actually cost.
///
/// The residency plan is the prediction for this phase, so the measured peak
/// and `peak_bytes()` land on one line — the comparison issue #641 needed and
/// nobody had.
fn load_ltx2_av_transformer_with_loras(
    plan: &Ltx2GeneratePlan,
    stage: Ltx2StageShape,
    device: &candle_core::Device,
    loras: &[LoraWeight],
    vram_budget_override: Option<u64>,
    progress: Option<&ProgressCallback>,
) -> Result<Ltx2AvTransformer3DModel> {
    // Clear first: a streaming or eager build must not be reported against the
    // previous stage's adaptive plan.
    record_ltx2_residency_plan(None);
    // The shape is part of the label: a two-stage render loads twice at two
    // different shapes, and two identically named lines would be useless.
    let probe = PhaseVramProbe::enter_if(
        format!(
            "transformer_load[{}x{}x{}]",
            stage.width, stage.height, stage.frames
        ),
        device.is_cuda(),
    );
    let result = load_ltx2_av_transformer_with_loras_inner(
        plan,
        stage,
        device,
        loras,
        vram_budget_override,
        progress,
    );
    let residency = last_ltx2_residency_plan();
    let report = probe.finish_with_predicted(residency.as_ref().map(|plan| plan.peak_bytes()));
    log_ltx2_phase_vram_result(
        report,
        &result,
        residency.as_ref(),
        &ltx2_residency_detail(residency.as_ref()),
    );
    result
}

/// Open the checkpoint the selected backend expects. Split out so the caller
/// can attach one piece of context — naming the file — to every backend's
/// failure.
fn ltx2_transformer_var_builder<'a>(
    plan: &Ltx2GeneratePlan,
    checkpoint_path: &Path,
    device: &candle_core::Device,
    checkpoint_is_nvfp4: bool,
    checkpoint_is_convrot: bool,
    checkpoint_is_fp8: bool,
    progress: Option<&ProgressCallback>,
) -> Result<VarBuilder<'a>> {
    if checkpoint_is_nvfp4 {
        let backend = super::nvfp4::Ltx2Nvfp4Backend::from_path(checkpoint_path)?;
        return Ok(VarBuilder::from_backend(
            Box::new(backend),
            compute_dtype(device),
            device.clone(),
        ));
    }
    if checkpoint_is_convrot {
        let backend =
            super::convrot::Ltx2ConvRotBackend::from_path_for_device(checkpoint_path, device)?;
        return Ok(VarBuilder::from_backend(
            Box::new(backend),
            compute_dtype(device),
            device.clone(),
        ));
    }
    if checkpoint_is_fp8 {
        return load_fp8_safetensors_with_callback(
            std::slice::from_ref(&checkpoint_path),
            device,
            "LTX-2 transformer",
            progress,
        );
    }
    let dtype = transformer_weight_dtype(plan, device);
    load_safetensors_with_progress_callback(
        std::slice::from_ref(&checkpoint_path),
        dtype,
        device,
        "LTX-2 transformer",
        progress,
    )
}

fn load_ltx2_av_transformer_with_loras_inner(
    plan: &Ltx2GeneratePlan,
    stage: Ltx2StageShape,
    device: &candle_core::Device,
    loras: &[LoraWeight],
    vram_budget_override: Option<u64>,
    progress: Option<&ProgressCallback>,
) -> Result<Ltx2AvTransformer3DModel> {
    let force_streaming = ltx2_force_streaming_enabled();
    let force_eager = crate::runtime_env::value("MOLD_LTX2_FORCE_EAGER").is_some();
    let config = ltx2_video_transformer_config(plan);
    let lora_registry = super::lora::load_lora_registry(loras)?;
    // Absence of the synthetic gradient is not proof a LoRA was applied, so say
    // what actually resolved: the ordered stack, each adapter's scale, and how
    // many transformer layers it landed on.
    if ltx_debug_enabled() {
        let layers = lora_registry
            .as_ref()
            .map(|registry| registry.layer_count())
            .unwrap_or(0);
        eprintln!(
            "[ltx2-debug] lora stack for {}x{}x{}: {} adapter(s) -> {} layer(s)",
            stage.width,
            stage.height,
            stage.frames,
            loras.len(),
            layers
        );
        for (index, lora) in loras.iter().enumerate() {
            eprintln!(
                "[ltx2-debug]   lora[{index}] scale={} path={}",
                lora.scale, lora.path
            );
        }
    }
    let checkpoint_path = Path::new(&plan.checkpoint_path);
    let checkpoint_is_nvfp4 = super::nvfp4::checkpoint_is_nvfp4(checkpoint_path);
    let checkpoint_is_convrot =
        !checkpoint_is_nvfp4 && super::convrot::checkpoint_is_convrot_w4a4(checkpoint_path);
    // ConvRot on CUDA stays resident in its packed form and is priced that
    // way; on Metal/CPU the compatibility backend reconstructs BF16 weights,
    // so blocks keep streaming there rather than pricing compact bytes as
    // widened residency.
    let force_streaming = ltx2_effective_force_streaming(
        force_streaming,
        checkpoint_is_convrot,
        Ltx2Accelerator::of(device),
    );
    // One header pass feeds both the fp8 probe and the residency sizing. The
    // index is the same authority admission priced this job with, so the
    // plan the engine builds cannot disagree with the grant it was given.
    let weight_index = Ltx2TransformerWeightIndex::read(checkpoint_path).ok();
    let checkpoint_is_fp8 =
        !checkpoint_is_nvfp4 && ltx2_checkpoint_is_fp8(plan, weight_index.as_ref());
    // `candle` re-exports the `safetensors` error transparently, so a corrupt
    // or truncated checkpoint arrives as bare text ("header too small") with
    // nothing identifying the file. Name it here: this is the error the user
    // now sees instead of a silently synthesized gradient.
    let vb = ltx2_transformer_var_builder(
        plan,
        checkpoint_path,
        device,
        checkpoint_is_nvfp4,
        checkpoint_is_convrot,
        checkpoint_is_fp8,
        progress,
    )
    .with_context(|| {
        format!(
            "failed to load LTX-2 transformer checkpoint {}; the file may be corrupt or \
             incompletely downloaded — re-pull the model and retry",
            checkpoint_path.display()
        )
    })?;
    let vb = if checkpoint_is_nvfp4 || checkpoint_is_convrot {
        vb
    } else {
        vb.rename_f(remap_ltx2_transformer_key)
    };
    if select_ltx2_transformer_residency_mode(
        Ltx2Accelerator::of(device),
        checkpoint_is_fp8,
        force_eager,
        force_streaming,
        false,
        0,
    ) == Ltx2TransformerResidencyMode::Eager
    {
        Ok(Ltx2AvTransformer3DModel::new(&config, vb, lora_registry)?)
    } else if device.is_cuda() && !force_streaming {
        let gpu_ordinal = thread_gpu_ordinal().unwrap_or(0);
        let free_vram = match vram_budget_override {
            Some(budget) => budget,
            None => usable_free_vram_bytes(gpu_ordinal).unwrap_or(0),
        };
        let weights = weight_index
            .as_ref()
            .context("LTX-2 checkpoint header is required for adaptive residency")
            .and_then(|index| {
                ltx2_transformer_weight_sizes(
                    index,
                    config.num_layers,
                    compute_dtype(device),
                    Ltx2ResidentWeightForm::for_convrot_backend(device.is_cuda()),
                )
            });
        match weights {
            Ok(weights)
                if select_ltx2_transformer_residency_mode(
                    Ltx2Accelerator::of(device),
                    checkpoint_is_fp8,
                    force_eager,
                    force_streaming,
                    weights.blocks.iter().any(|size| *size > 0),
                    free_vram,
                ) == Ltx2TransformerResidencyMode::Adaptive =>
            {
                let mut residency_plan =
                    ltx2_adaptive_transformer_plan(plan, stage, &weights, free_vram);
                emit_info(
                    progress,
                    format!(
                        "LTX-2 adaptive offload: {} resident / {} streamed blocks (resident {}, streamed {} per denoise pass, non-block weights {}, reserve {})",
                        residency_plan.resident_count(),
                        residency_plan.streamed_count(),
                        fmt_gb(residency_plan.resident_bytes),
                        fmt_gb(residency_plan.streamed_bytes),
                        fmt_gb(residency_plan.fixed_resident_bytes),
                        fmt_gb(residency_plan.reserved_bytes()),
                    ),
                );
                loop {
                    // Record before each attempt so the phase report — and an
                    // OOM diagnosis — describes the plan actually attempted,
                    // including every demotion rung.
                    record_ltx2_residency_plan(Some(residency_plan.clone()));
                    match Ltx2AvTransformer3DModel::new_adaptive(
                        &config,
                        vb.clone(),
                        lora_registry.clone(),
                        residency_plan.clone(),
                    ) {
                        Ok(transformer) => break Ok(transformer),
                        Err(err)
                            if device.is_cuda()
                                && residency_plan.resident_count() > 0
                                && !ltx2_error_is_fatal_cuda(&err)
                                && is_out_of_memory_error(&err) =>
                        {
                            emit_info(
                                progress,
                                format!(
                                    "LTX-2 adaptive offload: resident allocation OOM at {} resident blocks; retrying with fewer resident blocks",
                                    residency_plan.resident_count()
                                ),
                            );
                            try_synchronize_device(gpu_ordinal)?;
                            if !residency_plan.demote_largest_resident(&weights.blocks) {
                                return Err(err.into());
                            }
                        }
                        Err(err) => return Err(err.into()),
                    }
                }
            }
            Ok(_) | Err(_) => Ok(Ltx2AvTransformer3DModel::new_streaming(
                &config,
                vb,
                lora_registry,
            )?),
        }
    } else {
        Ok(Ltx2AvTransformer3DModel::new_streaming(
            &config,
            vb,
            lora_registry,
        )?)
    }
}

fn emit_info(progress: Option<&ProgressCallback>, message: String) {
    if let Some(progress) = progress {
        progress(ProgressEvent::Info { message });
    } else {
        tracing::info!("{message}");
    }
}

fn emit_phase_done(
    progress: Option<&ProgressCallback>,
    phase: ProgressPhase,
    name: &str,
    elapsed: std::time::Duration,
) {
    if let Some(callback) = progress {
        callback(ProgressEvent::PhaseDone {
            phase,
            name: name.to_string(),
            elapsed,
        });
    }
}

/// The shape a single denoise stage actually renders at.
///
/// Two-stage and Distilled pipelines render stage 1 at a fraction of the
/// requested resolution (`derive_stage1_render_shape`) and only stage 2 at the
/// full frame, so budgeting every stage against `plan.width`/`plan.height`
/// over-charges stage 1 by 4× at the usual ×2 spatial upsample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ltx2StageShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    /// Whether the run carries per-token conditioning (source image,
    /// keyframes, source video, or an extend carryover), which materializes
    /// the per-token AdaLN modulation instead of one broadcast row.
    pub(crate) conditioned: bool,
}

impl Ltx2StageShape {
    /// The shape a stage renders, taken from the stage's own pixel shape.
    /// For a one-stage pipeline that is the requested frame; for a two-stage
    /// or Distilled one it is the reduced stage-1 render and then the
    /// upsampled stage-2 frame.
    fn from_pixel_shape(plan: &Ltx2GeneratePlan, shape: VideoPixelShape) -> Self {
        Self {
            width: shape.width as u32,
            height: shape.height as u32,
            frames: shape.frames as u32,
            conditioned: plan_is_conditioned(plan),
        }
    }
}

fn plan_is_conditioned(plan: &Ltx2GeneratePlan) -> bool {
    !plan.conditioning.images.is_empty()
        || !plan.conditioning.latents.is_empty()
        || plan.conditioning.video_path.is_some()
}

fn ltx2_video_activation_budget(stage: Ltx2StageShape, adaln_dim: Option<u64>) -> u64 {
    ltx2_activation_budget_bytes(
        stage.width,
        stage.height,
        stage.frames,
        stage.conditioned,
        adaln_dim,
    )
}

/// The VRAM the adaptive planner may spend on this transformer.
///
/// Sampled free VRAM is an upper bound, never an entitlement: when the
/// scheduler admitted the job at a predicted peak the engine must size itself
/// against that grant, otherwise a plan admitted at 11.5 GB quietly fills a
/// 24 GB card and dies at the first denoise step.
fn ltx2_transformer_vram_budget(grant: Option<u64>, usable_free_vram: u64) -> u64 {
    match grant {
        Some(grant) => grant.min(usable_free_vram),
        None => usable_free_vram,
    }
}

fn ltx2_adaptive_transformer_plan(
    plan: &Ltx2GeneratePlan,
    stage: Ltx2StageShape,
    weights: &Ltx2TransformerWeightSizes,
    free_vram: u64,
) -> AdaptiveResidencyPlan {
    // The per-forward transient (one dequantized linear) sits beside the
    // resident weights for the whole denoise, and a packed-resident ConvRot
    // forward additionally rotates and re-quantizes its activation — both are
    // reserved here exactly as admission reserves them.
    let activation = ltx2_video_activation_budget(stage, weights.adaln_dim).saturating_add(
        if weights.int8_packed {
            mold_core::ltx2_weight_index::ltx2_int8_w8a8_workspace_bytes(
                crate::device::ltx2_token_count(stage.width, stage.height, stage.frames),
            )
        } else {
            0
        },
    );
    plan_adaptive_residency(
        &weights.blocks,
        ltx2_transformer_vram_budget(plan.vram_grant_bytes, free_vram),
        weights
            .non_block_bytes
            .saturating_add(weights.transient_bytes),
        activation,
        ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
    )
}

/// Shrink the VRAM budget for a denoise-stage OOM retry.
///
/// Each rung gives the planner a quarter less to work with, which demotes
/// blocks *and* keeps the activation reserve intact. Returns `None` once the
/// budget is below the point where any residency is worth attempting — the
/// caller then rebuilds in full-streaming mode, which is what a budget under
/// the base reserve produces anyway.
fn ltx2_denoise_retry_vram_budget(previous: u64) -> Option<u64> {
    const FLOOR: u64 = ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM;
    let next = previous / 4 * 3;
    (next > FLOOR).then_some(next)
}

/// Detect CUDA errors that invalidate the process-owned context.
///
/// Mirrors `mold-server`'s `is_fatal_cuda_error` (that crate isn't a
/// dependency of `mold-inference`). These must never be retried: `CLAUDE.md`
/// requires the worker be quarantined and the process restarted, because
/// candle/cudarc objects still hold primary-context handles.
fn ltx2_error_is_fatal_cuda(err: &impl std::fmt::Display) -> bool {
    let message = err.to_string();
    [
        "CUDA_ERROR_ILLEGAL_ADDRESS",
        "CUDA_ERROR_ECC_UNCORRECTABLE",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA_ERROR_ASSERT",
        "CUDA_ERROR_MISALIGNED_ADDRESS",
        "CUDA_ERROR_HARDWARE_STACK_ERROR",
        "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        "CUDA_ERROR_INVALID_PC",
        "CUDA_ERROR_LAUNCH_TIMEOUT",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

/// Whether a denoise-stage failure is a recoverable OOM.
///
/// A fatal context error wins over the OOM substrings — an illegal address can
/// surface alongside an allocation message, and retrying a dead context is the
/// one thing we must never do.
fn ltx2_denoise_error_is_recoverable_oom(err: &anyhow::Error) -> bool {
    !ltx2_error_is_fatal_cuda(&format!("{err:#}")) && is_out_of_memory_error(&format!("{err:#}"))
}

/// Maximum denoise-stage rebuild attempts after the first OOM.
const LTX2_DENOISE_OOM_MAX_RETRIES: usize = 2;

/// Run one denoise stage, rebuilding the transformer with a smaller VRAM
/// budget if it hits CUDA OOM.
///
/// The transformer is moved in and returned so the failing engine can actually
/// be dropped (and the device synchronized) before the retry allocates
/// anything — the whole point of the ladder. Construction already had an OOM
/// ladder; the denoise loop, which is where the 24 GB card actually dies, had
/// none.
fn run_denoise_stage_with_oom_recovery<T>(
    label: &str,
    transformer: Ltx2AvTransformer3DModel,
    device: &candle_core::Device,
    mut rebuild: impl FnMut(u64) -> Result<Ltx2AvTransformer3DModel>,
    mut run: impl FnMut(&Ltx2AvTransformer3DModel) -> Result<T>,
    progress: Option<&ProgressCallback>,
) -> Result<(T, Ltx2AvTransformer3DModel)> {
    let gpu_ordinal = thread_gpu_ordinal().unwrap_or(0);
    let mut transformer = transformer;
    let mut budget: Option<u64> = None;
    for attempt in 0..=LTX2_DENOISE_OOM_MAX_RETRIES {
        // One line per stage per attempt — never per denoise step.
        let probe = PhaseVramProbe::enter_if(format!("{label}.denoise"), device.is_cuda());
        let outcome = run(&transformer);
        let residency = last_ltx2_residency_plan();
        let report = probe.finish_with_predicted(residency.as_ref().map(|plan| plan.peak_bytes()));
        log_ltx2_phase_vram_result(
            report,
            &outcome,
            residency.as_ref(),
            &ltx2_residency_detail(residency.as_ref()),
        );
        match outcome {
            Ok(value) => return Ok((value, transformer)),
            Err(err) => {
                if !device.is_cuda()
                    || attempt == LTX2_DENOISE_OOM_MAX_RETRIES
                    || !ltx2_denoise_error_is_recoverable_oom(&err)
                {
                    return Err(err);
                }
                // Drop first: free VRAM sampled while the failing engine is
                // still resident describes what is left over, not what the
                // rebuild may spend.
                drop(transformer);
                try_synchronize_device(gpu_ordinal)?;
                let ceiling = match budget {
                    Some(previous) => previous,
                    None => usable_free_vram_bytes(gpu_ordinal).unwrap_or(0),
                };
                let Some(next_budget) = ltx2_denoise_retry_vram_budget(ceiling) else {
                    return Err(err);
                };
                budget = Some(next_budget);
                emit_info(
                    progress,
                    format!(
                        "LTX-2 {label}: denoise ran out of VRAM; rebuilding the transformer \
                         within {} (attempt {} of {})",
                        fmt_gb(next_budget),
                        attempt + 2,
                        LTX2_DENOISE_OOM_MAX_RETRIES + 1,
                    ),
                );
                transformer = rebuild(next_budget)?;
            }
        }
    }
    unreachable!("the loop returns on the final attempt")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ltx2TransformerResidencyMode {
    Eager,
    Streaming,
    Adaptive,
}

/// The accelerator a transformer is being loaded onto. Residency is not a
/// CUDA-or-not decision: CUDA can page blocks against a measured VRAM budget,
/// Metal shares one unified pool with the host and cannot, and CPU has no
/// device memory to be resident in at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ltx2Accelerator {
    Cuda,
    Metal,
    Other,
}

impl Ltx2Accelerator {
    fn of(device: &candle_core::Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Other
        }
    }
}

/// `MOLD_LTX2_FORCE_STREAMING` used to be read with `is_some()`, so setting it
/// to `0` or `false` switched streaming *on* — the opposite of what the value
/// says, and unlike the `MOLD_OFFLOAD` alias sitting next to it. Both are
/// parsed the same way now.
fn ltx2_force_streaming_from_values(force_streaming: Option<&str>, offload: Option<&str>) -> bool {
    fn truthy(value: Option<&str>) -> bool {
        matches!(
            value.map(|raw| raw.trim().to_ascii_lowercase()).as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }
    truthy(force_streaming) || truthy(offload)
}

fn ltx2_force_streaming_enabled() -> bool {
    ltx2_force_streaming_from_values(
        crate::runtime_env::value("MOLD_LTX2_FORCE_STREAMING").as_deref(),
        crate::runtime_env::value("MOLD_OFFLOAD").as_deref(),
    )
}

fn ltx2_effective_force_streaming(
    configured: bool,
    checkpoint_is_convrot: bool,
    accelerator: Ltx2Accelerator,
) -> bool {
    // ConvRot forces streaming only where no packed-resident arm exists:
    // CUDA holds packed U8 blocks resident (`LtxLinear::ConvRotPacked`),
    // while Metal and CPU still widen every block to BF16 on materialize, so
    // a resident block there would cost the widened figure — the exact 2x
    // the old unconditional rule existed to avoid.
    configured || (checkpoint_is_convrot && accelerator != Ltx2Accelerator::Cuda)
}

fn select_ltx2_transformer_residency_mode(
    accelerator: Ltx2Accelerator,
    checkpoint_is_fp8: bool,
    force_eager: bool,
    force_streaming: bool,
    has_block_sizes: bool,
    free_vram: u64,
) -> Ltx2TransformerResidencyMode {
    if force_streaming {
        return Ltx2TransformerResidencyMode::Streaming;
    }
    // An explicit eager request is honoured on every accelerator that has
    // device memory to be resident in. On Metal this is the only way to
    // measure what streaming costs: it re-materialises all 48 blocks per
    // denoise pass out of a pool the host already shares, so the copy buys
    // nothing there. The *default* stays streaming on Metal — a resident
    // transformer has to be sized against the prompt encoder first.
    if checkpoint_is_fp8
        && force_eager
        && matches!(accelerator, Ltx2Accelerator::Cuda | Ltx2Accelerator::Metal)
    {
        return Ltx2TransformerResidencyMode::Eager;
    }
    // Only CUDA can page blocks against a measured free-VRAM budget.
    if accelerator != Ltx2Accelerator::Cuda {
        return Ltx2TransformerResidencyMode::Streaming;
    }
    if has_block_sizes && free_vram > 0 {
        Ltx2TransformerResidencyMode::Adaptive
    } else {
        Ltx2TransformerResidencyMode::Streaming
    }
}

fn load_ltx2_video_vae(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    dtype: DType,
    progress: Option<&ProgressCallback>,
) -> Result<AutoencoderKLLtx2Video> {
    let probe = PhaseVramProbe::enter_if("vae_load", device.is_cuda());
    let result = load_ltx2_video_vae_inner(plan, device, dtype, progress);
    log_ltx2_phase_vram_result(probe.finish(), &result, None, "");
    result
}

fn load_ltx2_video_vae_inner(
    plan: &Ltx2GeneratePlan,
    device: &candle_core::Device,
    dtype: DType,
    progress: Option<&ProgressCallback>,
) -> Result<AutoencoderKLLtx2Video> {
    let vb = load_safetensors_with_progress_callback(
        std::slice::from_ref(&Path::new(&plan.vae_checkpoint_path)),
        dtype,
        device,
        "LTX-2 VAE",
        progress,
    )?;
    let vb = if plan.vae_in_checkpoint {
        vb.pp("vae")
    } else {
        vb
    };
    let config = ltx2_video_vae_config(plan);
    let is_diffusion_decoder = !plan.vae_in_checkpoint
        && matches!(
            mold_core::ltx25_probe::probe_ltx25_video_vae(Path::new(&plan.vae_checkpoint_path)),
            Ok(mold_core::ltx25_probe::Ltx25VideoVaeKind::Diffusion)
        );
    if is_diffusion_decoder {
        Ok(AutoencoderKLLtx2Video::new_diffusion(config, vb)?)
    } else {
        Ok(AutoencoderKLLtx2Video::new(config, vb)?)
    }
}

/// Decode video latents to frames as one measured, reported VAE phase.
///
/// Every pipeline decodes through here. Besides the memory telemetry this
/// closes a real gap: the one-stage and retake paths emitted neither a
/// `ProgressPhase::Vae` nor a decode timing, so the scheduler never learned a
/// `vae_ms` for them and planned their decode as free.
/// A decoded pass: 8-bit frames always, linear HDR frames when asked for.
struct DecodedVideo {
    frames: Vec<RgbImage>,
    /// How many EXR frames the decode wrote, when a sidecar was requested.
    /// A count rather than the pixels: the frames are already on disk.
    hdr_frames_written: Option<usize>,
}

fn decode_video_frames_with_telemetry(
    pipeline: &str,
    vae: &mut AutoencoderKLLtx2Video,
    latents: &Tensor,
    pixel_shape: VideoPixelShape,
    dtype: DType,
    device: &candle_core::Device,
    debug_enabled: bool,
    hdr_exr: Option<&HdrExrTarget>,
    progress: Option<&ProgressCallback>,
) -> Result<DecodedVideo> {
    let decode_start = Instant::now();
    let probe = PhaseVramProbe::enter_if(format!("{pipeline}.vae_decode"), device.is_cuda());
    // Closure so a decode that dies of OOM is reported before it propagates.
    let decoded = (|| -> Result<DecodedVideo> {
        let decode_latents = latents.to_dtype(dtype)?;
        configure_ltx2_vae_decode_memory_mode(vae, &decode_latents, device)?;
        let (_dec_output, video) = vae.decode(&decode_latents, None, false, false)?;
        if debug_enabled {
            log_tensor_stats("decoded_video", &video)?;
        }
        // The EXR sidecar is written here, from the same tensor, because the
        // 8-bit conversion below is lossy and HDR cannot be recovered from the
        // frames afterwards. Streaming it frame-by-frame keeps peak memory at
        // one frame instead of the whole clip.
        let hdr_frames_written = hdr_exr
            .map(|target| write_hdr_frames_streaming(&video, target))
            .transpose()?;
        let frames = decoded_video_to_frames(&video, pixel_shape)?;
        if device.is_cuda() {
            device.synchronize()?;
        }
        drop(video);
        Ok(DecodedVideo {
            frames,
            hdr_frames_written,
        })
    })();
    log_ltx2_phase_vram_result(probe.finish(), &decoded, None, "");
    if decoded.is_ok() {
        let elapsed = decode_start.elapsed();
        emit_phase_done(
            progress,
            ProgressPhase::Vae,
            "Decoding video frames",
            elapsed,
        );
        log_elapsed_secs(&format!("{pipeline}.decode_video"), elapsed.as_secs_f64());
    }
    decoded
}

fn configure_ltx2_vae_decode_memory_mode(
    vae: &mut AutoencoderKLLtx2Video,
    latents: &Tensor,
    device: &candle_core::Device,
) -> Result<()> {
    vae.use_framewise_decoding = should_use_ltx2_framewise_decode(vae, latents, device)?;
    if vae.use_framewise_decoding {
        tracing::info!(
            "LTX-2 VAE decode using temporal chunks; projected full decode workspace {} exceeds memory budget",
            fmt_gb(projected_ltx2_vae_decode_workspace_bytes(vae, latents)?)
        );
    }
    // Temporal chunking bounds how many frames are in flight, not how large
    // one frame is. Past the trained span a single frame is the problem, so
    // the same memory verdict also splits it spatially.
    let (_, _, _, latent_height, latent_width) = latents.dims5()?;
    vae.spatial_decode_tiling = plan_spatial_decode_tiling(
        latent_height,
        latent_width,
        spatial_tile_policy()?,
        vae.use_framewise_decoding,
    );
    if let Some(tiling) = vae.spatial_decode_tiling {
        tracing::info!(
            "LTX-2 VAE decode using {}-px spatial tiles with {}-px overlap",
            tiling.tile_cells * crate::ltx2::tiling::LATENT_PIXEL_STRIDE,
            tiling.overlap_cells * crate::ltx2::tiling::LATENT_PIXEL_STRIDE,
        );
    }
    Ok(())
}

fn should_use_ltx2_framewise_decode(
    vae: &AutoencoderKLLtx2Video,
    latents: &Tensor,
    device: &candle_core::Device,
) -> Result<bool> {
    if crate::runtime_env::value("MOLD_LTX2_VAE_FORCE_FULL_DECODE").is_some() {
        return Ok(false);
    }
    if crate::runtime_env::value("MOLD_LTX2_VAE_FORCE_FRAMEWISE").is_some() {
        return Ok(true);
    }
    if default_ltx2_framewise_decode(device) {
        return Ok(true);
    }
    if !device.is_cuda() {
        return Ok(false);
    }

    let projected = projected_ltx2_vae_decode_workspace_bytes(vae, latents)?;
    let gpu_ordinal = thread_gpu_ordinal().unwrap_or(0);
    let Some(free_vram) = usable_free_vram_bytes(gpu_ordinal) else {
        return Ok(false);
    };
    Ok(projected.saturating_add(ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM) > free_vram)
}

fn default_ltx2_framewise_decode(device: &candle_core::Device) -> bool {
    // Apple unified memory is shared with the application and window server;
    // a full temporal decode can pressure the Metal driver even when tensor
    // accounting looks flat. Bound the default to one temporal chunk, while
    // retaining the explicit full-decode override for diagnostics.
    device.is_metal()
}

fn projected_ltx2_vae_decode_workspace_bytes(
    vae: &AutoencoderKLLtx2Video,
    latents: &Tensor,
) -> Result<u64> {
    let (batch, _channels, latent_frames, latent_height, latent_width) = latents.dims5()?;
    let temporal_scale = vae.temporal_compression_ratio().max(1);
    let spatial_scale = vae.spatial_compression_ratio().max(1);
    let frames = if latent_frames == 0 {
        0
    } else {
        (latent_frames - 1)
            .saturating_mul(temporal_scale)
            .saturating_add(1)
    };
    let height = latent_height.saturating_mul(spatial_scale);
    let width = latent_width.saturating_mul(spatial_scale);
    let output_channels = vae.config().out_channels;
    let sample_bytes = [batch, output_channels, frames, height, width]
        .into_iter()
        .try_fold(1u64, |acc, value| acc.checked_mul(value as u64))
        .and_then(|elements| elements.checked_mul(dtype_bytes(latents.dtype()) as u64))
        .context("LTX-2 VAE decode byte estimate overflowed")?;
    Ok(sample_bytes.saturating_mul(8))
}

fn ltx2_video_transformer_config(plan: &Ltx2GeneratePlan) -> Ltx2VideoTransformer3DModelConfig {
    let cross_attention_adaln = plan.preset.transformer.cross_attention_adaln
        && !ltx_debug_disable_cross_attention_adaln_enabled();
    let apply_gated_attention = plan.preset.transformer.apply_gated_attention
        && !ltx_debug_disable_transformer_gated_attention_enabled();
    Ltx2VideoTransformer3DModelConfig {
        in_channels: plan.preset.transformer.in_channels,
        out_channels: plan.preset.transformer.out_channels,
        patch_size: 1,
        patch_size_t: 1,
        num_attention_heads: plan.preset.transformer.num_attention_heads,
        attention_head_dim: plan.preset.transformer.attention_head_dim,
        cross_attention_dim: plan.preset.transformer.cross_attention_dim,
        num_layers: plan.preset.transformer.num_layers,
        qk_norm: "rms_norm".to_string(),
        norm_elementwise_affine: false,
        norm_eps: 1e-6,
        caption_channels: plan.preset.video_connector_inner_dim(),
        caption_projection_in_transformer: matches!(
            plan.preset.caption_projection,
            crate::ltx2::preset::CaptionProjectionPlacement::Transformer
        ),
        attention_bias: true,
        attention_out_bias: true,
        positional_embedding_theta: 10_000.0,
        positional_embedding_max_pos: vec![20, 2048, 2048],
        use_middle_indices_grid: true,
        rope_type: crate::ltx2::model::LtxRopeType::Split,
        double_precision_rope: true,
        audio_num_attention_heads: plan.preset.transformer.audio_num_attention_heads,
        audio_attention_head_dim: plan.preset.transformer.audio_attention_head_dim,
        audio_in_channels: plan.preset.transformer.audio_in_channels,
        audio_out_channels: plan.preset.transformer.audio_out_channels,
        audio_cross_attention_dim: plan.preset.transformer.audio_cross_attention_dim,
        audio_positional_embedding_max_pos: vec![20],
        apply_gated_attention,
        // Public LTX-2 checkpoints set this to 1000.0, which keeps the AV gate
        // branch on the same sigma*1000 scale as the main timestep embedding.
        av_ca_timestep_scale_multiplier: 1000.0,
        cross_attention_adaln,
        video_ff_bias: plan.preset.transformer.video_ff_bias,
        audio_ff_bias: plan.preset.transformer.audio_ff_bias,
        use_keyframes_abs_pos_embedding: plan.preset.transformer.use_keyframes_abs_pos_embedding,
        streaming_prefetch_count: plan.streaming_prefetch_count.unwrap_or(1) as usize,
    }
}

fn transformer_weight_dtype(_plan: &Ltx2GeneratePlan, device: &candle_core::Device) -> DType {
    // Public LTX-2 FP8 manifests keep transformer weights in float8 storage but
    // run the native Rust matmuls in the normal compute dtype after applying the
    // checkpoint-provided per-tensor weight scales.
    compute_dtype(device)
}

fn ltx2_checkpoint_is_fp8(
    plan: &Ltx2GeneratePlan,
    weight_index: Option<&Ltx2TransformerWeightIndex>,
) -> bool {
    if plan.checkpoint_path.to_ascii_lowercase().contains("fp8") {
        return true;
    }
    weight_index.is_some_and(Ltx2TransformerWeightIndex::is_fp8)
}

fn ltx2_video_vae_config(plan: &Ltx2GeneratePlan) -> AutoencoderKLLtx2VideoConfig {
    if plan.preset.uses_ltx2_22b_video_vae {
        AutoencoderKLLtx2VideoConfig::ltx2_22b()
    } else {
        AutoencoderKLLtx2VideoConfig::default()
    }
}

fn ltx2_scheduler_config() -> FlowMatchEulerDiscreteSchedulerConfig {
    FlowMatchEulerDiscreteSchedulerConfig {
        num_train_timesteps: 1000,
        shift: 1.0,
        use_dynamic_shifting: false,
        base_shift: Some(0.5),
        max_shift: Some(1.15),
        base_image_seq_len: Some(256),
        max_image_seq_len: Some(4096),
        invert_sigmas: false,
        shift_terminal: None,
        use_karras_sigmas: false,
        use_exponential_sigmas: false,
        use_beta_sigmas: false,
        time_shift_type: TimeShiftType::Exponential,
        stochastic_sampling: false,
    }
}

fn remap_ltx2_transformer_key(name: &str) -> String {
    super::nvfp4::remap_ltx2_transformer_key(name)
}

/// Transformer weight byte totals the residency planner works from.
///
/// Derived from [`Ltx2TransformerWeightIndex`] — the same header index
/// admission prices the job with — through
/// [`ltx2_transformer_weight_sizes`], so the engine's plan and the
/// scheduler's grant read one set of numbers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ltx2TransformerWeightSizes {
    /// Per-block device bytes while resident, indexed by transformer block.
    pub blocks: Vec<usize>,
    /// Transformer tensors outside every block — `patchify_proj`,
    /// `adaln_single.linear`, `caption_projection`, the audio/video
    /// connectors, and `proj_out`. `new_with_block_source` allocates these on
    /// the GPU *after* every resident block, so they are unconditionally
    /// resident and must be reserved, not discovered.
    pub non_block_bytes: u64,
    /// The checkpoint's `adaln_single.linear` output width, which sets the
    /// per-token AdaLN cost of a conditioned render. Not a constant across
    /// LTX-2: the 19B ships six components (24,576) and LTX-2.3's 22B ships
    /// nine (36,864). `None` when the tensor is absent from the header.
    pub adaln_dim: Option<u64>,
    /// Per-forward scratch beside the resident weights (one dequantized
    /// linear for a quantized checkpoint, zero for dense/float8). Carried so
    /// admission and the planner can reserve the same figure.
    pub transient_bytes: u64,
    /// Blocks are resident in the packed INT8 ConvRot form, so each forward
    /// also needs the token-scaled W8A8 workspace
    /// (`mold_core::ltx2_weight_index::ltx2_int8_w8a8_workspace_bytes`).
    pub int8_packed: bool,
}

/// Size a transformer for residency planning at the given compute dtype.
///
/// `num_layers` is the model config's block count: blocks the header does
/// not carry are reported as zero bytes, blocks beyond it are ignored. INT8
/// ConvRot is priced fully widened (every ConvRot arm reconstructs BF16
/// weights on the device), float8 and GGUF at their packed bytes, dense
/// checkpoints at the compute dtype — the index owns that rule. NVFP4 is
/// refused here on purpose: its packed streaming path is not modelled by the
/// adaptive planner, and refusing keeps it on the streaming fallback it has
/// always taken.
pub fn ltx2_transformer_weight_sizes(
    index: &Ltx2TransformerWeightIndex,
    num_layers: usize,
    compute_dtype: DType,
    form: Ltx2ResidentWeightForm,
) -> Result<Ltx2TransformerWeightSizes> {
    if index.format() == Ltx2WeightFormat::Nvfp4 {
        bail!("adaptive residency is not modelled for NVFP4 checkpoints");
    }
    let elem_size = compute_dtype.size_in_bytes() as u64;
    let mut blocks = vec![0usize; num_layers];
    for (slot, bytes) in blocks
        .iter_mut()
        .zip(index.resident_block_bytes_for(elem_size, form))
    {
        *slot = usize::try_from(bytes).context("LTX-2 block size overflows usize")?;
    }
    Ok(Ltx2TransformerWeightSizes {
        blocks,
        non_block_bytes: index.resident_non_block_bytes(elem_size),
        adaln_dim: index.adaln_dim(),
        transient_bytes: index.transient_bytes(),
        int8_packed: index.is_convrot() && form == Ltx2ResidentWeightForm::Packed,
    })
}

/// Read-and-size in one call for tests that only care about the sizing
/// result; the load path reads the index once and reuses it.
#[cfg(test)]
fn ltx2_transformer_block_sizes_from_safetensors(
    path: &Path,
    num_layers: usize,
    compute_dtype: DType,
) -> Result<Ltx2TransformerWeightSizes> {
    let index = Ltx2TransformerWeightIndex::read(path)?;
    ltx2_transformer_weight_sizes(
        &index,
        num_layers,
        compute_dtype,
        Ltx2ResidentWeightForm::Widened,
    )
}

fn denoised_from_velocity(sample: &Tensor, velocity: &Tensor, sigma: f32) -> Result<Tensor> {
    let velocity = if velocity.dtype() == sample.dtype() {
        velocity.clone()
    } else {
        velocity.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&velocity.affine(sigma as f64, 0.0)?)
        .map_err(Into::into)
}

fn velocity_from_denoised(sample: &Tensor, denoised: &Tensor, sigma: f32) -> Result<Tensor> {
    if sigma == 0.0 {
        return Tensor::zeros_like(sample).map_err(Into::into);
    }
    let denoised = if denoised.dtype() == sample.dtype() {
        denoised.clone()
    } else {
        denoised.to_dtype(sample.dtype())?
    };
    sample
        .broadcast_sub(&denoised)?
        .affine(1.0 / sigma as f64, 0.0)
        .map_err(Into::into)
}

fn ltx_debug_enabled() -> bool {
    env::var_os("MOLD_LTX_DEBUG").is_some()
}

fn ltx_debug_timings_enabled() -> bool {
    env::var_os("MOLD_LTX2_DEBUG_TIMINGS").is_some()
}

// ── Phase VRAM telemetry ─────────────────────────────────────────────────────
//
// LTX-2 requests run for minutes, so a handful of always-on lines per request
// is free — and issue #641 (a 24 GB card admitted, loaded for two minutes, then
// OOMed) was expensive precisely because nothing recorded what each phase
// actually cost. Every line is diagnostics: the scheduler's frozen grant stays
// the memory authority, and no probe result is ever fed back into admission.

const LTX2_VRAM_TARGET: &str = "mold::ltx2::vram";

thread_local! {
    /// The residency plan the most recent transformer build on this thread
    /// used. Stashed rather than threaded through the call graph because the
    /// plan is chosen deep inside the loader while the phases that need it for
    /// diagnosis (load, denoise) sit above it.
    static LAST_RESIDENCY_PLAN: std::cell::RefCell<Option<AdaptiveResidencyPlan>> =
        const { std::cell::RefCell::new(None) };
}

fn record_ltx2_residency_plan(plan: Option<AdaptiveResidencyPlan>) {
    LAST_RESIDENCY_PLAN.with(|slot| *slot.borrow_mut() = plan);
}

fn last_ltx2_residency_plan() -> Option<AdaptiveResidencyPlan> {
    LAST_RESIDENCY_PLAN.with(|slot| slot.borrow().clone())
}

/// The full residency decision, so an OOM report needs no reproduction.
fn ltx2_residency_summary(plan: Option<&AdaptiveResidencyPlan>) -> String {
    let Some(plan) = plan else {
        return "residency=unknown".to_string();
    };
    format!(
        "residency: resident_count={} streamed_count={} resident_bytes={} \
         activation_budget={} runtime_headroom={} largest_streamed_block={} \
         fixed_resident_bytes={} planned_peak={}",
        plan.resident_count(),
        plan.streamed_count(),
        fmt_gb(plan.resident_bytes),
        fmt_gb(plan.activation_budget),
        fmt_gb(plan.runtime_headroom),
        fmt_gb(plan.largest_streamed_block),
        fmt_gb(plan.fixed_resident_bytes),
        fmt_gb(plan.peak_bytes()),
    )
}

/// Plan detail carried beside a measured peak on the healthy path, so
/// predicted-versus-actual is one line rather than two.
fn ltx2_residency_detail(plan: Option<&AdaptiveResidencyPlan>) -> String {
    match plan {
        Some(plan) => format!(
            " resident_blocks={} fixed_resident={}",
            plan.resident_count(),
            fmt_gb(plan.fixed_resident_bytes),
        ),
        None => String::new(),
    }
}

fn log_ltx2_phase_vram(report: &PhaseVramReport, detail: &str) {
    tracing::info!(target: LTX2_VRAM_TARGET, "[ltx2-vram] {report}{detail}");
}

/// Report one finished phase, escalating to an error line that carries the
/// whole residency plan when the phase died of CUDA OOM. Logged before the
/// error propagates so the diagnosis survives whatever the caller does next.
fn log_ltx2_phase_vram_result<T>(
    report: PhaseVramReport,
    result: &Result<T>,
    plan: Option<&AdaptiveResidencyPlan>,
    detail: &str,
) {
    match result {
        Err(error) if is_out_of_memory_error(&format!("{error:#}")) => {
            tracing::error!(
                target: LTX2_VRAM_TARGET,
                "[ltx2-vram] {report}{detail} OUT-OF-MEMORY {} error={error:#}",
                ltx2_residency_summary(plan),
            );
        }
        _ => log_ltx2_phase_vram(&report, detail),
    }
}

fn log_debug_vram(label: &str) {
    // Read the GPU this thread is actually bound to. Hard-coding ordinal 0
    // reported another card's free memory on every multi-GPU host, which makes
    // the whole triage trace fiction.
    let ordinal = thread_gpu_ordinal().unwrap_or(0);
    if let Some(free) = free_vram_bytes(ordinal) {
        eprintln!(
            "[ltx2-debug] {label} gpu={ordinal} free_vram={}",
            fmt_gb(free)
        );
    } else {
        eprintln!("[ltx2-debug] {label} gpu={ordinal} free_vram=unavailable");
    }
}

fn ltx_debug_compare_uncond_enabled() -> bool {
    crate::runtime_env::value("MOLD_LTX_DEBUG_COMPARE_UNCOND").is_some()
}

fn ltx_debug_alt_prompt() -> Option<String> {
    crate::runtime_env::value("MOLD_LTX_DEBUG_ALT_PROMPT")
        .map(|prompt| prompt.trim().to_string())
        .filter(|prompt| !prompt.is_empty())
}

fn ltx_debug_disable_audio_branch_enabled() -> bool {
    crate::runtime_env::value("MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH").is_some()
}

fn ltx_debug_disable_cross_attention_adaln_enabled() -> bool {
    crate::runtime_env::value("MOLD_LTX_DEBUG_DISABLE_CROSS_ATTENTION_ADALN").is_some()
}

fn ltx_debug_disable_transformer_gated_attention_enabled() -> bool {
    crate::runtime_env::value("MOLD_LTX2_DEBUG_DISABLE_TRANSFORMER_GATED_ATTENTION").is_some()
}

fn ltx_debug_log_file() -> &'static Mutex<Option<std::fs::File>> {
    static LOG_FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    LOG_FILE.get_or_init(|| {
        let path = env::var_os("MOLD_LTX_DEBUG_FILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp/mold-ltx2-debug.log"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok();
        Mutex::new(file)
    })
}

fn log_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
    let tensor = tensor.to_dtype(DType::F32)?;
    let mean = tensor.flatten_all()?.mean_all()?.to_scalar::<f32>()?;
    let abs_mean = tensor
        .flatten_all()?
        .abs()?
        .mean_all()?
        .to_scalar::<f32>()?;
    let sq_mean = tensor
        .flatten_all()?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?;
    let std = (sq_mean - mean * mean).max(0.0).sqrt();
    let line = format!(
        "[ltx2-debug] {name}: shape={:?} mean={mean:.6} abs_mean={abs_mean:.6} rms={:.6} std={std:.6}",
        tensor.dims(),
        sq_mean.sqrt(),
    );
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    Ok(())
}

fn log_timing(label: &str, start: Instant) {
    if !ltx_debug_timings_enabled() {
        return;
    }
    eprintln!(
        "[ltx2-timing] {label} {:.3}s",
        start.elapsed().as_secs_f64()
    );
}

fn log_elapsed_secs(label: &str, elapsed_secs: f64) {
    if !ltx_debug_timings_enabled() {
        return;
    }
    eprintln!("[ltx2-timing] {label} {elapsed_secs:.3}s");
}

fn log_prompt_debug_stats(plan: &Ltx2GeneratePlan, prompt: &NativePromptEncoding) -> Result<()> {
    let cond = &plan.prompt_tokens.conditional;
    let uncond = &plan.prompt_tokens.unconditional;
    let cond_tail = cond
        .input_ids
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let cond_tail_mask = cond
        .attention_mask
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let uncond_tail = uncond
        .input_ids
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let uncond_tail_mask = uncond
        .attention_mask
        .iter()
        .rev()
        .take(12)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let token_line = format!(
        "[ltx2-debug] prompt_tokens cond_valid={} uncond_valid={} cond_tail_ids={cond_tail:?} cond_tail_mask={cond_tail_mask:?} uncond_tail_ids={uncond_tail:?} uncond_tail_mask={uncond_tail_mask:?}",
        cond.valid_len(),
        uncond.valid_len(),
    );
    eprintln!("{token_line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{token_line}");
        }
    }

    log_tensor_stats("cond_video_context", &prompt.conditional.video_encoding)?;
    log_tensor_stats("uncond_video_context", &prompt.unconditional.video_encoding)?;
    log_tensor_pair_stats(
        "video_context",
        &prompt.conditional.video_encoding,
        &prompt.unconditional.video_encoding,
    )?;

    let cond_video_mask_valid = prompt
        .conditional
        .attention_mask
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let uncond_video_mask_valid = prompt
        .unconditional
        .attention_mask
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let mask_line = format!(
        "[ltx2-debug] prompt_masks cond_valid_tokens={cond_video_mask_valid:.0} uncond_valid_tokens={uncond_video_mask_valid:.0}"
    );
    eprintln!("{mask_line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{mask_line}");
        }
    }

    if let (Some(cond_audio), Some(uncond_audio)) = (
        prompt.conditional.audio_encoding.as_ref(),
        prompt.unconditional.audio_encoding.as_ref(),
    ) {
        log_tensor_stats("cond_audio_context", cond_audio)?;
        log_tensor_stats("uncond_audio_context", uncond_audio)?;
        log_tensor_pair_stats("audio_context", cond_audio, uncond_audio)?;
    }

    Ok(())
}

fn log_alt_prompt_debug_stats(
    plan: &Ltx2GeneratePlan,
    primary: &EmbeddingsProcessorOutput,
    alternate: &EmbeddingsProcessorOutput,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let alt_prompt = ltx_debug_alt_prompt().unwrap_or_else(|| "<unset>".to_string());
    let line = format!(
        "[ltx2-debug] alternate_prompt primary={:?} alternate={alt_prompt:?}",
        plan.prompt
    );
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    log_tensor_pair_stats(
        "alt_prompt_video_context",
        &primary.video_encoding,
        &alternate.video_encoding,
    )?;
    if let (Some(primary_audio), Some(alternate_audio)) = (
        primary.audio_encoding.as_ref(),
        alternate.audio_encoding.as_ref(),
    ) {
        log_tensor_pair_stats("alt_prompt_audio_context", primary_audio, alternate_audio)?;
    }
    Ok(())
}

fn log_tensor_pair_stats(name: &str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    let delta = lhs.broadcast_sub(rhs)?;
    log_tensor_stats(&format!("{name}_delta"), &delta)?;
    let cosine = tensor_cosine_similarity(lhs, rhs)?;
    let l2 = tensor_l2_distance(lhs, rhs)?;
    let line = format!("[ltx2-debug] {name}_pair cosine={cosine:.6} l2={l2:.6}");
    eprintln!("{line}");
    if let Ok(mut guard) = ltx_debug_log_file().lock() {
        if let Some(file) = guard.as_mut() {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
    Ok(())
}

fn log_distilled_prompt_sensitivity(
    stage: Option<&str>,
    step_idx: usize,
    sigma: f32,
    video_sample: &Tensor,
    conditional_video_velocity: &Tensor,
    unconditional_video_velocity: &Tensor,
    audio_sample: Option<&Tensor>,
    conditional_audio_velocity: Option<&Tensor>,
    unconditional_audio_velocity: Option<&Tensor>,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let prefix = format!(
        "{}_step{step_idx}_sigma{sigma:.6}",
        stage.unwrap_or("stage")
    );
    log_tensor_pair_stats(
        &format!("{prefix}_video_velocity_cond_vs_uncond"),
        conditional_video_velocity,
        unconditional_video_velocity,
    )?;
    let conditional_video_x0 =
        convert_velocity_to_x0(video_sample, conditional_video_velocity, sigma)?;
    let unconditional_video_x0 =
        convert_velocity_to_x0(video_sample, unconditional_video_velocity, sigma)?;
    log_tensor_pair_stats(
        &format!("{prefix}_video_x0_cond_vs_uncond"),
        &conditional_video_x0,
        &unconditional_video_x0,
    )?;

    if let (
        Some(audio_sample),
        Some(conditional_audio_velocity),
        Some(unconditional_audio_velocity),
    ) = (
        audio_sample,
        conditional_audio_velocity,
        unconditional_audio_velocity,
    ) {
        log_tensor_pair_stats(
            &format!("{prefix}_audio_velocity_cond_vs_uncond"),
            conditional_audio_velocity,
            unconditional_audio_velocity,
        )?;
        let conditional_audio_x0 =
            convert_velocity_to_x0(audio_sample, conditional_audio_velocity, sigma)?;
        let unconditional_audio_x0 =
            convert_velocity_to_x0(audio_sample, unconditional_audio_velocity, sigma)?;
        log_tensor_pair_stats(
            &format!("{prefix}_audio_x0_cond_vs_uncond"),
            &conditional_audio_x0,
            &unconditional_audio_x0,
        )?;
    }

    Ok(())
}

fn log_distilled_alternate_prompt_sensitivity(
    stage: Option<&str>,
    step_idx: usize,
    sigma: f32,
    video_sample: &Tensor,
    primary_video_velocity: &Tensor,
    alternate_video_velocity: &Tensor,
    audio_sample: Option<&Tensor>,
    primary_audio_velocity: Option<&Tensor>,
    alternate_audio_velocity: Option<&Tensor>,
) -> Result<()> {
    if !ltx_debug_enabled() {
        return Ok(());
    }
    let prefix = format!(
        "{}_step{step_idx}_sigma{sigma:.6}",
        stage.unwrap_or("stage")
    );
    log_tensor_pair_stats(
        &format!("{prefix}_video_velocity_prompt_vs_alt"),
        primary_video_velocity,
        alternate_video_velocity,
    )?;
    let primary_video_x0 = convert_velocity_to_x0(video_sample, primary_video_velocity, sigma)?;
    let alternate_video_x0 = convert_velocity_to_x0(video_sample, alternate_video_velocity, sigma)?;
    log_tensor_pair_stats(
        &format!("{prefix}_video_x0_prompt_vs_alt"),
        &primary_video_x0,
        &alternate_video_x0,
    )?;

    if let (Some(audio_sample), Some(primary_audio_velocity), Some(alternate_audio_velocity)) = (
        audio_sample,
        primary_audio_velocity,
        alternate_audio_velocity,
    ) {
        log_tensor_pair_stats(
            &format!("{prefix}_audio_velocity_prompt_vs_alt"),
            primary_audio_velocity,
            alternate_audio_velocity,
        )?;
        let primary_audio_x0 = convert_velocity_to_x0(audio_sample, primary_audio_velocity, sigma)?;
        let alternate_audio_x0 =
            convert_velocity_to_x0(audio_sample, alternate_audio_velocity, sigma)?;
        log_tensor_pair_stats(
            &format!("{prefix}_audio_x0_prompt_vs_alt"),
            &primary_audio_x0,
            &alternate_audio_x0,
        )?;
    }

    Ok(())
}

fn tensor_cosine_similarity(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let rhs = rhs
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?;
    let lhs_flat = lhs.flatten_all()?;
    let rhs_flat = rhs.flatten_all()?;
    let dot = lhs_flat
        .broadcast_mul(&rhs_flat)?
        .sum_all()?
        .to_scalar::<f32>()?;
    let lhs_norm = lhs_flat
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt()
        .max(1e-12);
    let rhs_norm = rhs_flat
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt()
        .max(1e-12);
    Ok(dot / (lhs_norm * rhs_norm))
}

fn tensor_l2_distance(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    Ok(lhs
        .broadcast_sub(rhs)?
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use candle_core::{DType, Device, Tensor};

    #[cfg(feature = "metal")]
    #[test]
    fn metal_defaults_to_framewise_vae_decode() {
        let device = Device::new_metal(0).unwrap();
        assert!(super::default_ltx2_framewise_decode(&device));
        assert!(!super::default_ltx2_framewise_decode(&Device::Cpu));
    }
    use candle_nn::VarBuilder;
    use mold_core::{
        GenerateRequest, LoraWeight, Ltx2SpatialUpscale, Ltx2TemporalUpscale, OutputFormat,
        TimeRange,
    };

    use super::{
        apply_stage_video_conditioning, apply_video_token_replacements,
        build_static_multimodal_guidance_batch, build_video_conditioning_self_attention_mask,
        clean_latents_for_conditioning, convert_velocity_to_x0, convert_x0_to_velocity,
        decoded_video_to_frames, effective_native_guidance_scale, emit_denoise_progress,
        guided_velocity_from_cfg, initialize_video_stage_latents, keyframe_only_conditioning,
        ltx2_video_transformer_config, plan_stage2_tiles_with_policy,
        reapply_stage_video_conditioning, resize_tail_frames_to_pixel_shape,
        should_inspect_step_velocity, source_image_only_conditioning, stage2_carried_audio,
        strip_appended_video_conditioning, timestep_from_sigma_and_mask,
        write_hdr_frames_streaming, HdrExrTarget, Ltx2RuntimeSession, Ltx2VaeLatentStats,
        Stage2AudioPolicy, StageAudioConditioning, StageVideoConditioning, TiledStage2Pass,
        VideoTokenAppendCondition, VideoTokenReplacement, APPENDED_VIDEO_NOISE_SALT,
        LTX2_AUDIO_LATENT_CHANNELS, LTX2_AUDIO_MEL_BINS, LTX2_VIDEO_LATENT_CHANNELS,
    };
    use crate::engine::seeded_randn;
    use crate::ltx2::conditioning::{self, StagedConditioning};
    use crate::ltx2::model::VideoLatentShape;
    use crate::ltx2::model::VideoPixelShape;
    use crate::ltx2::plan::{Ltx2GeneratePlan, PipelineKind};
    use mold_core::ltx2_weight_index::Ltx2TransformerWeightIndex;

    /// `[1, 3, frames, 2, 2]` CPU tensor whose every sample in frame `f` is
    /// `values[f]`, so a written EXR identifies which decoded frame it came
    /// from by value alone.
    fn per_frame_constant_video(values: &[f32]) -> Tensor {
        let mut data = Vec::with_capacity(3 * values.len() * 4);
        for _channel in 0..3 {
            for &value in values {
                data.extend_from_slice(&[value; 4]);
            }
        }
        Tensor::from_vec(data, (1, 3, values.len(), 2, 2), &Device::Cpu).unwrap()
    }

    fn read_exr_first_sample(path: &std::path::Path) -> f32 {
        let read = exr::prelude::read_first_rgba_layer_from_file(
            path,
            |resolution, _| {
                vec![(0f32, 0f32, 0f32, 0f32); resolution.width() * resolution.height()]
            },
            |pixels: &mut Vec<(f32, f32, f32, f32)>,
             position,
             (r, g, b, a): (f32, f32, f32, f32)| {
                pixels[position.y() * 2 + position.x()] = (r, g, b, a);
            },
        )
        .unwrap();
        read.layer_data.channel_data.pixels[0].0
    }

    /// A chain stage writes exactly its window — local frames
    /// `[skip, skip + count)` at global indices `[start, start + count)` —
    /// and nothing else. This is the per-stage half of issue #688's "EXR
    /// count equals the stitched frame count and no index written twice".
    #[test]
    fn hdr_streaming_writer_honours_the_stage_window() {
        let dir = tempfile::tempdir().unwrap();
        let values = [-0.8f32, -0.4, 0.0, 0.4, 0.8, 0.9, 1.0];
        let video = per_frame_constant_video(&values);
        let target = HdrExrTarget {
            dir: dir.path().to_path_buf(),
            precision: crate::ltx2::exr::ExrPrecision::Full,
            window: Some(crate::chain::ExrStageWindow {
                skip_leading: 2,
                start_index: 5,
                write_count: 3,
            }),
        };
        let written = write_hdr_frames_streaming(&video, &target).unwrap();
        assert_eq!(written, 3);

        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["frame_00005.exr", "frame_00006.exr", "frame_00007.exr"],
            "skipped local frames must produce no files and indices must be global",
        );
        // frame_00005 is local frame 2 (value 0.0) through the LogC3 inverse.
        for (global, local) in [(5usize, 2usize), (6, 3), (7, 4)] {
            let expected = crate::ltx2::hdr::vae_output_to_linear_hdr(values[local]);
            let actual =
                read_exr_first_sample(&crate::ltx2::exr::exr_frame_path(dir.path(), global));
            assert!(
                (actual - expected).abs() < 1e-6,
                "frame_{global:05} should hold local frame {local}'s value \
                 (expected {expected}, got {actual})",
            );
        }
    }

    /// No window is the single-render identity: every decoded frame, numbered
    /// from zero — bit-for-bit the pre-#688 behaviour.
    #[test]
    fn hdr_streaming_writer_default_window_is_the_identity() {
        let dir = tempfile::tempdir().unwrap();
        let values = [-0.5f32, 0.0, 0.5];
        let video = per_frame_constant_video(&values);
        let target = HdrExrTarget {
            dir: dir.path().to_path_buf(),
            precision: crate::ltx2::exr::ExrPrecision::Full,
            window: None,
        };
        let written = write_hdr_frames_streaming(&video, &target).unwrap();
        assert_eq!(written, 3);
        for (index, &value) in values.iter().enumerate() {
            let expected = crate::ltx2::hdr::vae_output_to_linear_hdr(value);
            let actual =
                read_exr_first_sample(&crate::ltx2::exr::exr_frame_path(dir.path(), index));
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    /// A window reaching past the decoded clip writes only what exists —
    /// the cap arithmetic upstream should prevent this, but the writer must
    /// not index out of bounds if it ever regresses.
    #[test]
    fn hdr_streaming_writer_clamps_the_window_to_the_clip() {
        let dir = tempfile::tempdir().unwrap();
        let video = per_frame_constant_video(&[0.0, 0.1, 0.2]);
        let target = HdrExrTarget {
            dir: dir.path().to_path_buf(),
            precision: crate::ltx2::exr::ExrPrecision::Half,
            window: Some(crate::chain::ExrStageWindow {
                skip_leading: 1,
                start_index: 10,
                write_count: 99,
            }),
        };
        let written = write_hdr_frames_streaming(&video, &target).unwrap();
        assert_eq!(written, 2, "only local frames 1..3 exist");
        assert!(crate::ltx2::exr::exr_frame_path(dir.path(), 10).exists());
        assert!(crate::ltx2::exr::exr_frame_path(dir.path(), 11).exists());
    }
    use crate::ltx2::preset::preset_for_model;
    use crate::ltx2::text::connectors::PaddingSide;
    use crate::ltx2::text::encoder::{GemmaConfig, GemmaHiddenStateEncoder};
    use crate::ltx2::text::gemma::{EncodedPromptPair, PromptTokens};
    use crate::ltx2::text::prompt_encoder::{
        build_embeddings_processor, ConnectorSpec, NativePromptEncoder,
    };
    use crate::ltx2::tiling::{create_tiles, DimensionTiling, SpatialTilePolicy, TileCountConfig};
    use crate::progress::{
        InferenceCancellationToken, ProgressCallback, ProgressEvent, ProgressPhase,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

    fn req(model: &str, format: OutputFormat, enable_audio: Option<bool>) -> GenerateRequest {
        GenerateRequest {
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "test".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 3.0,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(format),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(97),
            fps: Some(24),
            upscale_model: None,
            gif_preview: false,
            enable_audio,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    fn prompt_pair() -> EncodedPromptPair {
        EncodedPromptPair {
            conditional: PromptTokens {
                input_ids: vec![0, 0, 5],
                attention_mask: vec![0, 0, 1],
            },
            unconditional: PromptTokens {
                input_ids: vec![0, 0, 0],
                attention_mask: vec![0, 0, 0],
            },
            pad_token_id: 0,
            eos_token_id: Some(1),
            max_length: 3,
        }
    }

    fn tiny_gemma_config() -> GemmaConfig {
        GemmaConfig {
            attention_bias: false,
            head_dim: 4,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            vocab_size: 16,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 4,
            sliding_window: 4,
            sliding_window_pattern: 2,
            max_position_embeddings: 32,
        }
    }

    fn zero_gemma_var_builder(cfg: &GemmaConfig) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let (rows, cols) = match name {
                    "self_attn.q_proj" => (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                    "self_attn.k_proj" | "self_attn.v_proj" => {
                        (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                    }
                    "self_attn.o_proj" => (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                    "mlp.gate_proj" | "mlp.up_proj" => (cfg.intermediate_size, cfg.hidden_size),
                    "mlp.down_proj" => (cfg.hidden_size, cfg.intermediate_size),
                    _ => unreachable!(),
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for name in [
                "self_attn.q_norm",
                "self_attn.k_norm",
                "input_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_attention_layernorm",
            ] {
                let dim = if name.contains("q_norm") || name.contains("k_norm") {
                    cfg.head_dim
                } else {
                    cfg.hidden_size
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        tensors.insert(
            "model.norm.weight".to_string(),
            Tensor::zeros(cfg.hidden_size, DType::F32, &Device::Cpu).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn zero_connector_source_var_builder() -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.weight".to_string(),
            Tensor::zeros((8, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.bias".to_string(),
            Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.weight".to_string(),
            Tensor::zeros((4, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.bias".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        for (prefix, dim) in [
            ("model.diffusion_model.video_embeddings_connector", 8usize),
            ("model.diffusion_model.audio_embeddings_connector", 4usize),
        ] {
            for linear_name in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.weight"),
                    Tensor::zeros((dim, dim), DType::F32, &Device::Cpu).unwrap(),
                );
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.bias"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for norm_name in ["attn1.q_norm", "attn1.k_norm"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{norm_name}.weight"),
                    Tensor::ones(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.weight"),
                Tensor::zeros((dim * 4, dim), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.bias"),
                Tensor::zeros(dim * 4, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.weight"),
                Tensor::zeros((dim, dim * 4), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.bias"),
                Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.learnable_registers"),
                Tensor::zeros((128, dim), DType::F32, &Device::Cpu).unwrap(),
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn runtime_session() -> Ltx2RuntimeSession {
        let cfg = tiny_gemma_config();
        let gemma = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        let prompt_encoder = NativePromptEncoder::new(
            gemma,
            build_embeddings_processor(
                zero_connector_source_var_builder(),
                crate::ltx2::preset::GemmaFeatureExtractorKind::V2DualAv,
                cfg.hidden_size,
                cfg.num_hidden_layers,
                8,
                Some(4),
                ConnectorSpec {
                    prefix: "model.diffusion_model.video_embeddings_connector.",
                    num_attention_heads: 2,
                    attention_head_dim: 4,
                    num_layers: 1,
                    apply_gated_attention: false,
                    positional_embedding_theta: 10_000.0,
                    positional_embedding_max_pos: &[32],
                    rope_type: crate::ltx2::model::LtxRopeType::Split,
                    double_precision_rope: true,
                    num_learnable_registers: Some(128),
                },
                Some(ConnectorSpec {
                    prefix: "model.diffusion_model.audio_embeddings_connector.",
                    num_attention_heads: 1,
                    attention_head_dim: 4,
                    num_layers: 1,
                    apply_gated_attention: false,
                    positional_embedding_theta: 10_000.0,
                    positional_embedding_max_pos: &[32],
                    rope_type: crate::ltx2::model::LtxRopeType::Split,
                    double_precision_rope: true,
                    num_learnable_registers: Some(128),
                }),
            )
            .unwrap(),
            PaddingSide::Left,
        );
        Ltx2RuntimeSession::new(candle_core::Device::Cpu, prompt_encoder, 0)
    }

    fn build_plan(
        req: &GenerateRequest,
        preset: crate::ltx2::preset::Ltx2ModelPreset,
        conditioning: StagedConditioning,
    ) -> Ltx2GeneratePlan {
        let loras = crate::ltx2::lora::normalize_loras(req);
        let graph = crate::ltx2::execution::build_execution_graph(
            req,
            PipelineKind::Distilled,
            &conditioning,
            &preset,
            loras.len(),
        );
        Ltx2GeneratePlan {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            hdr_exr_window: None,
            reference_frame_offset: 0,
            scene_embeddings_path: None,
            guidance_overrides: None,
            vram_grant_bytes: None,
            pipeline: PipelineKind::Distilled,
            preset,
            checkpoint_is_distilled: req.model.contains("distilled"),
            execution_graph: graph,
            checkpoint_path: "/tmp/ltx2.safetensors".to_string(),
            vae_checkpoint_path: "/tmp/ltx2.safetensors".to_string(),
            vae_in_checkpoint: true,
            audio_components_path: None,
            text_projection_path: None,
            distilled_checkpoint_path: None,
            distilled_lora_path: None,
            spatial_upsampler_path: None,
            temporal_upsampler_path: None,
            duration_head_path: None,
            auto_duration: None,
            gemma_root: "/tmp/gemma".to_string(),
            output_path: "/tmp/output.mp4".to_string(),
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            prompt_tokens: prompt_pair(),
            seed: 7,
            width: req.width,
            height: req.height,
            num_frames: req.frames.unwrap(),
            frame_rate: req.fps.unwrap(),
            num_inference_steps: req.steps,
            guidance: req.guidance,
            quantization: Some("fp8-cast".to_string()),
            streaming_prefetch_count: Some(2),
            conditioning,
            image_preprocessing: None,
            loras,
            retake_range: req.retake_range.clone(),
            spatial_upscale: req.spatial_upscale,
            temporal_upscale: req.temporal_upscale,
        }
    }

    /// A T2A plan for `model`, with the pipeline and execution graph the
    /// audio-only route actually builds.
    fn t2a_plan(model: &str) -> Ltx2GeneratePlan {
        let mut request = req(model, OutputFormat::Wav, None);
        request.pipeline = Some(mold_core::Ltx2PipelineMode::T2a);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&request, temp_dir.path()).unwrap();
        let preset = preset_for_model(model).unwrap();
        let loras = crate::ltx2::lora::normalize_loras(&request);
        let graph = crate::ltx2::execution::build_execution_graph(
            &request,
            PipelineKind::T2a,
            &conditioning,
            &preset,
            loras.len(),
        );
        let mut plan = build_plan(&request, preset, conditioning);
        plan.pipeline = PipelineKind::T2a;
        plan.execution_graph = graph;
        plan
    }

    /// Upstream's audio guider constants, with `modality_scale` pinned to 1.0
    /// because audio-only has no video branch for the cross-modal term to act
    /// on (`t2a_one_stage.py:184`). The STG block differs by checkpoint depth.
    #[test]
    fn t2a_audio_guider_matches_upstream_constants() {
        for (model, expected_stg_block) in
            [("ltx-2.3-22b-dev:fp8", 28usize), ("ltx-2-19b-dev:fp8", 29)]
        {
            let plan = t2a_plan(model);
            let (video, audio) = super::stage_multimodal_guider_params(&plan, 0)
                .unwrap()
                .expect("t2a must define guider params");

            assert_eq!(
                video,
                super::MultiModalGuiderParams::default(),
                "{model}: the video guider is inert for audio-only"
            );
            assert_eq!(audio.cfg_scale, 7.0, "{model}");
            assert_eq!(audio.stg_scale, 1.0, "{model}");
            assert_eq!(audio.rescale_scale, 0.7, "{model}");
            assert_eq!(audio.skip_step, 0, "{model}");
            assert_eq!(audio.stg_blocks, vec![expected_stg_block], "{model}");
            assert_eq!(
                audio.modality_scale, 1.0,
                "{model}: cross-modal guidance is meaningless without a video branch"
            );
        }
    }

    /// cfg 7.0 means the unconditional pass is required — if this went false
    /// the prompt encoder would drop the negative encoding and the guided step
    /// would silently degrade to a plain conditional one.
    #[test]
    fn t2a_requires_the_unconditional_context() {
        let plan = t2a_plan("ltx-2.3-22b-dev:fp8");
        assert!(super::stage_requires_unconditional_context(&plan, 0).unwrap());
    }

    /// The LTX-2 family default is 8 steps, tuned for the distilled *video*
    /// ladder. T2A runs the plain flow-match scheduler, where 8 steps leave
    /// the latents far from the manifold and the vocoder renders hiss. The
    /// raise is disclosed, and a larger request is left alone.
    #[test]
    fn t2a_raises_too_few_steps_to_the_preset_default_and_says_so() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let sink = messages.clone();
        let progress: ProgressCallback = Box::new(move |event| {
            if let ProgressEvent::Info { message } = event {
                sink.lock().unwrap().push(message);
            }
        });

        let mut plan = t2a_plan("ltx-2.3-22b-dev:fp8");
        plan.num_inference_steps = 8;
        assert_eq!(super::t2a_effective_steps(&plan, Some(&progress)), 30);
        let logged = messages.lock().unwrap().join("\n");
        assert!(logged.contains("30"), "raise must be disclosed: {logged}");

        let mut plan_19b = t2a_plan("ltx-2-19b-dev:fp8");
        plan_19b.num_inference_steps = 8;
        assert_eq!(super::t2a_effective_steps(&plan_19b, None), 40);

        plan.num_inference_steps = 60;
        let before = messages.lock().unwrap().len();
        assert_eq!(super::t2a_effective_steps(&plan, Some(&progress)), 60);
        assert_eq!(
            messages.lock().unwrap().len(),
            before,
            "a caller asking for more than the default keeps their number, silently"
        );
    }

    /// End-to-end CPU run of the audio-only denoise: patchify, the guided
    /// steps, and unpatchify back to `[batch, channels, frames, mel_bins]`.
    ///
    /// This is the shape contract the first GPU render would otherwise
    /// discover: a rank-2 audio timestep makes `sigma_scale_for_sample`
    /// broadcast one value across the token axis and fail on step 0, and a
    /// mismatched guidance batch layout silently reads the wrong chunk.
    #[test]
    fn audio_only_stage_denoises_and_restores_the_latent_shape() {
        use crate::ltx2::model::audio_transformer::Ltx2AudioTransformerModel;
        use crate::ltx2::model::video_transformer::tests::{
            av_transformer_var_builder_with_options, tiny_av_config,
        };
        use crate::ltx2::model::{AudioLatentShape, AudioPatchifier};

        let config = tiny_av_config();
        let vb = av_transformer_var_builder_with_options(config.clone(), false);
        let transformer = Ltx2AudioTransformerModel::new(&config, vb, None).unwrap();

        // `audio_in_channels` is 2 in the tiny config, and patchify folds
        // `channels * mel_bins` into the token feature axis.
        let shape = AudioLatentShape {
            batch: 1,
            channels: 2,
            frames: 5,
            mel_bins: 1,
        };
        let device = Device::Cpu;
        let noise = Tensor::rand(
            -1.0f32,
            1.0,
            (shape.batch, shape.channels, shape.frames, shape.mel_bins),
            &device,
        )
        .unwrap();
        let positions = AudioPatchifier::new(16_000, 160, 4, true, 0)
            .get_patch_grid_bounds(shape, &device)
            .unwrap();
        let context = Tensor::rand(-1.0f32, 1.0, (1, 4, config.caption_channels), &device).unwrap();
        let uncond = Tensor::rand(-1.0f32, 1.0, (1, 4, config.caption_channels), &device).unwrap();

        let out = super::run_real_audio_only_stage(
            &transformer,
            shape,
            &noise,
            &positions,
            &context,
            Some(&uncond),
            None,
            None,
            super::MultiModalGuiderParams {
                cfg_scale: 7.0,
                stg_scale: 1.0,
                stg_blocks: vec![0],
                rescale_scale: 0.7,
                modality_scale: 1.0,
                skip_step: 0,
            },
            &[1.0, 0.5],
            crate::ltx2::execution::SamplerMode::Euler,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            out.dims(),
            &[shape.batch, shape.channels, shape.frames, shape.mel_bins]
        );
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()),
            "denoised audio latents must be finite"
        );
    }

    /// The exact mechanism behind the rank-2 timestep bug, pinned on its own.
    ///
    /// `sigma_scale_for_sample` branches on rank: a rank-1 `[batch]` sigma is
    /// one value per sample, while a rank-2 sigma is one value per *token* and
    /// is reshaped to `[batch, tokens, 1]`. Handing it a `[batch, 1]` tensor —
    /// which reads like "one sigma for this batch" — asks it to spread one
    /// element across every token, and it fails. T2A therefore has to build a
    /// rank-1 timestep, matching the unmasked branch of
    /// `timestep_from_sigma_and_mask`.
    #[test]
    fn audio_sigma_scaling_requires_a_rank_one_per_batch_timestep() {
        let device = Device::Cpu;
        let sample = Tensor::zeros((1, 126, 128), DType::F32, &device).unwrap();

        let rank_one = Tensor::full(0.5f32, (1,), &device).unwrap();
        let scaled = super::sigma_scale_for_sample(&sample, &rank_one).unwrap();
        assert_eq!(scaled.dims(), &[1, 1, 1]);

        // The shape the first implementation produced.
        let rank_two = Tensor::full(0.5f32, (1, 1), &device).unwrap();
        assert!(
            super::sigma_scale_for_sample(&sample, &rank_two).is_err(),
            "a [batch, 1] sigma must not silently broadcast across 126 tokens"
        );
    }

    /// Step 0 specifically. The rank-2 timestep failed on the very first
    /// denoise step, so a one-step schedule is the smallest run that proves
    /// the guided path is wired correctly — no later step can mask it, and no
    /// later step is needed to expose it.
    #[test]
    fn audio_only_stage_completes_its_first_denoise_step() {
        use crate::ltx2::model::audio_transformer::Ltx2AudioTransformerModel;
        use crate::ltx2::model::video_transformer::tests::{
            av_transformer_var_builder_with_options, tiny_av_config,
        };
        use crate::ltx2::model::{AudioLatentShape, AudioPatchifier};

        let config = tiny_av_config();
        let vb = av_transformer_var_builder_with_options(config.clone(), false);
        let transformer = Ltx2AudioTransformerModel::new(&config, vb, None).unwrap();
        let shape = AudioLatentShape {
            batch: 1,
            channels: 2,
            frames: 5,
            mel_bins: 1,
        };
        let device = Device::Cpu;
        let noise = Tensor::rand(
            -1.0f32,
            1.0,
            (shape.batch, shape.channels, shape.frames, shape.mel_bins),
            &device,
        )
        .unwrap();
        let positions = AudioPatchifier::new(16_000, 160, 4, true, 0)
            .get_patch_grid_bounds(shape, &device)
            .unwrap();
        let context = Tensor::rand(-1.0f32, 1.0, (1, 4, config.caption_channels), &device).unwrap();
        let uncond = Tensor::rand(-1.0f32, 1.0, (1, 4, config.caption_channels), &device).unwrap();

        // Exactly one step: `sigmas_no_terminal` of length 1 becomes the
        // schedule `[1.0, 0.0]`, so the loop body runs once, at step 0.
        let out = super::run_real_audio_only_stage(
            &transformer,
            shape,
            &noise,
            &positions,
            &context,
            Some(&uncond),
            None,
            None,
            super::MultiModalGuiderParams {
                cfg_scale: 7.0,
                stg_scale: 1.0,
                stg_blocks: vec![0],
                rescale_scale: 0.7,
                modality_scale: 1.0,
                skip_step: 0,
            },
            &[1.0],
            crate::ltx2::execution::SamplerMode::Euler,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("the first denoise step must complete");
        assert_eq!(
            out.dims(),
            &[shape.batch, shape.channels, shape.frames, shape.mel_bins]
        );
    }

    /// T2A emits no frames. `render_native_video` must refuse it outright
    /// rather than fall through to the synthetic-placeholder path, which would
    /// happily fabricate a video for an audio request.
    #[test]
    fn t2a_never_reaches_the_video_renderer() {
        let plan = t2a_plan("ltx-2.3-22b-dev:fp8");
        assert!(!super::supports_real_video_path(&plan));
    }

    /// A Distilled FP8 plan at an explicit render shape.
    fn ltx2_plan_at(width: u32, height: u32, frames: u32) -> Ltx2GeneratePlan {
        let mut request = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        request.width = width;
        request.height = height;
        request.frames = Some(frames);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&request, temp_dir.path()).unwrap();
        let preset = preset_for_model(&request.model).unwrap();
        build_plan(&request, preset, conditioning)
    }

    #[test]
    fn distilled_stage2_sigmas_share_the_upstream_authority() {
        let ltx25 = req(
            "ltx-2.5-22b-distilled:int8-conv",
            OutputFormat::Mp4,
            Some(false),
        );
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&ltx25, temp_dir.path()).unwrap();
        let ltx25_plan = build_plan(
            &ltx25,
            preset_for_model(&ltx25.model).unwrap(),
            conditioning,
        );
        assert_eq!(
            super::distilled_stage2_sigmas_no_terminal(&ltx25_plan),
            &[0.909375, 0.725, 0.421875]
        );

        let ltx23 = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&ltx23, temp_dir.path()).unwrap();
        let ltx23_plan = build_plan(
            &ltx23,
            preset_for_model(&ltx23.model).unwrap(),
            conditioning,
        );
        assert_eq!(
            super::distilled_stage2_sigmas_no_terminal(&ltx23_plan),
            &[0.909375, 0.725, 0.421875]
        );
    }

    /// The engine's backstop for a resolution admission should already have
    /// refused. A one-pass pipeline past the trained span has no tiled
    /// refinement to renormalize positions, so it must error rather than
    /// render something plausible-looking and wrong.
    #[test]
    fn a_one_pass_pipeline_refuses_an_axis_past_the_trained_span() {
        let mut plan = ltx2_plan_at(3_840, 2_176, 25);
        plan.pipeline = PipelineKind::OneStage;
        let err = super::reject_oversized_axis_without_composition(&plan)
            .expect_err("a single-pass render cannot hold a 3840px axis");
        let message = err.to_string();
        assert!(
            message.contains("3840") && message.contains("spatial upsampler"),
            "the refusal must name the axis and the way out, got: {message}"
        );

        // The refining pipelines compose, so the same shape is fine.
        for pipeline in [
            PipelineKind::Distilled,
            PipelineKind::TwoStage,
            PipelineKind::TwoStageHq,
            PipelineKind::IcLora,
            PipelineKind::Keyframe,
            PipelineKind::A2Vid,
        ] {
            plan.pipeline = pipeline;
            assert!(
                super::reject_oversized_axis_without_composition(&plan).is_ok(),
                "{pipeline:?} renders stage 1 halved and refines it over tiles"
            );
        }

        // And nothing inside the span is touched, on any pipeline.
        let mut small = ltx2_plan_at(1_920, 1_088, 25);
        for pipeline in [
            PipelineKind::OneStage,
            PipelineKind::Retake,
            PipelineKind::LipDub,
        ] {
            small.pipeline = pipeline;
            assert!(super::reject_oversized_axis_without_composition(&small).is_ok());
        }
    }

    /// Composing is not enough on its own — the ceiling belongs to the rung.
    /// x1.5 divides by 1.5, so the same 4K output leaves stage 1 at 2560px,
    /// and stage 2 tiles the refinement rather than stage 1.
    #[test]
    fn a_refining_pipeline_still_refuses_a_rung_that_cannot_halve_the_target() {
        let mut plan = ltx2_plan_at(3_840, 2_176, 25);
        plan.pipeline = PipelineKind::TwoStage;

        plan.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        let message = super::reject_oversized_axis_without_composition(&plan)
            .expect_err("x1.5 cannot bring 3840 back inside the span")
            .to_string();
        assert!(message.contains("2560"), "got: {message}");

        // x2 halves it, and an absent rung means the runtime's implicit x2.
        plan.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        assert!(super::reject_oversized_axis_without_composition(&plan).is_ok());
        plan.spatial_upscale = None;
        assert!(super::reject_oversized_axis_without_composition(&plan).is_ok());

        // x1.5 is fine at its own ceiling.
        let mut within = ltx2_plan_at(3_072, 1_728, 25);
        within.pipeline = PipelineKind::TwoStage;
        within.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        assert!(super::reject_oversized_axis_without_composition(&within).is_ok());
    }

    fn stage_shape(
        plan: &Ltx2GeneratePlan,
        width: u32,
        height: u32,
        frames: u32,
    ) -> super::Ltx2StageShape {
        super::Ltx2StageShape::from_pixel_shape(
            plan,
            VideoPixelShape {
                batch: 1,
                frames: frames as usize,
                height: height as usize,
                width: width as usize,
                fps: plan.frame_rate as f32,
            },
        )
    }

    /// The 19B FP8 block set as measured from the checkpoint header: 6 BF16
    /// blocks (block 0 included) and 42 FP8 blocks.
    fn ltx2_19b_fp8_blocks() -> Vec<usize> {
        let mut blocks = vec![772_284_416usize; 6];
        blocks.extend(std::iter::repeat_n(386_408_672usize, 42));
        blocks
    }

    #[test]
    fn emit_denoise_progress_reports_progress_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let callback: ProgressCallback = Box::new(move |event| {
            sink.lock().unwrap().push(event);
        });

        emit_denoise_progress(Some(&callback), 3, 8, std::time::Duration::from_millis(12));

        let events = events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [ProgressEvent::DenoiseStep {
                step: 3,
                total: 8,
                ..
            }]
        ));
    }

    #[test]
    fn ltx2_vae_latent_stats_cache_reuses_broadcast_tensors() {
        let device = candle_core::Device::Cpu;
        let stats = Ltx2VaeLatentStats::from_tensors_for_test(
            Tensor::new(&[1.0f32, 2.0], &device).unwrap(),
            Tensor::new(&[2.0f32, 4.0], &device).unwrap(),
        );
        let latents = Tensor::from_vec(vec![3.0f32, 10.0], (1, 2, 1, 1, 1), &device).unwrap();

        let ((mean, std), first_hit) = stats.broadcast_tensors_for(&latents).unwrap();
        let ((mean_again, std_again), second_hit) = stats.broadcast_tensors_for(&latents).unwrap();
        let normalized = stats.normalize(&latents).unwrap();

        assert!(!first_hit);
        assert!(second_hit);
        assert_eq!(mean.dims5().unwrap(), (1, 2, 1, 1, 1));
        assert_eq!(std.dims5().unwrap(), (1, 2, 1, 1, 1));
        assert_eq!(format!("{:?}", mean_again.device()), format!("{device:?}"));
        assert_eq!(std_again.dtype(), DType::F32);
        assert_eq!(
            normalized.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
    }

    fn rebuild_execution_graph(plan: &mut Ltx2GeneratePlan, req: &GenerateRequest) {
        plan.execution_graph = crate::ltx2::execution::build_execution_graph(
            req,
            plan.pipeline,
            &plan.conditioning,
            &plan.preset,
            plan.loras.len(),
        );
    }

    #[derive(Clone, Copy)]
    struct Ltx2ParityCase {
        workflow: &'static str,
        model: &'static str,
        pipeline: PipelineKind,
        enable_audio: Option<bool>,
        seed: u64,
        configure: fn(&mut GenerateRequest),
    }

    impl Ltx2ParityCase {
        fn apply(self, req: &mut GenerateRequest) {
            (self.configure)(req);
        }
    }

    fn parity_noop(_req: &mut GenerateRequest) {}

    fn parity_source_image(req: &mut GenerateRequest) {
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    fn parity_audio_file(req: &mut GenerateRequest) {
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
    }

    fn parity_keyframes(req: &mut GenerateRequest) {
        req.keyframes = Some(vec![
            mold_core::KeyframeCondition {
                frame: 8,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
                name: None,
            },
            mold_core::KeyframeCondition {
                frame: 48,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
                name: None,
            },
        ]);
    }

    fn parity_retake(req: &mut GenerateRequest) {
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.retake_range = Some(TimeRange {
            start_seconds: 0.5,
            end_seconds: 1.25,
        });
    }

    fn parity_ic_lora(req: &mut GenerateRequest) {
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
    }

    fn parity_spatial_x2(req: &mut GenerateRequest) {
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
    }

    fn parity_spatial_x1_5(req: &mut GenerateRequest) {
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
    }

    fn parity_temporal_x2(req: &mut GenerateRequest) {
        req.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
    }

    fn ltx2_native_parity_matrix() -> Vec<Ltx2ParityCase> {
        vec![
            Ltx2ParityCase {
                workflow: "text-audio-video-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(true),
                seed: 424_301,
                configure: parity_noop,
            },
            Ltx2ParityCase {
                workflow: "fixed-seed-cuda-reference",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(true),
                seed: 424_303,
                configure: parity_noop,
            },
            Ltx2ParityCase {
                workflow: "image-to-video-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(false),
                seed: 424_311,
                configure: parity_source_image,
            },
            Ltx2ParityCase {
                workflow: "image-to-video-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(false),
                seed: 424_312,
                configure: parity_source_image,
            },
            Ltx2ParityCase {
                workflow: "audio-to-video-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::A2Vid,
                enable_audio: Some(true),
                seed: 424_321,
                configure: parity_audio_file,
            },
            Ltx2ParityCase {
                workflow: "audio-to-video-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::A2Vid,
                enable_audio: Some(true),
                seed: 424_322,
                configure: parity_audio_file,
            },
            Ltx2ParityCase {
                workflow: "keyframe-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::Keyframe,
                enable_audio: Some(false),
                seed: 424_331,
                configure: parity_keyframes,
            },
            Ltx2ParityCase {
                workflow: "keyframe-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Keyframe,
                enable_audio: Some(false),
                seed: 424_332,
                configure: parity_keyframes,
            },
            Ltx2ParityCase {
                workflow: "retake-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::Retake,
                enable_audio: Some(true),
                seed: 424_341,
                configure: parity_retake,
            },
            Ltx2ParityCase {
                workflow: "retake-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Retake,
                enable_audio: Some(true),
                seed: 424_342,
                configure: parity_retake,
            },
            Ltx2ParityCase {
                workflow: "public-ic-lora-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::IcLora,
                enable_audio: Some(true),
                seed: 424_351,
                configure: parity_ic_lora,
            },
            Ltx2ParityCase {
                workflow: "two-stage-dev-19b",
                model: "ltx-2-19b-dev:fp8",
                pipeline: PipelineKind::TwoStage,
                enable_audio: Some(false),
                seed: 424_361,
                configure: parity_noop,
            },
            Ltx2ParityCase {
                workflow: "two-stage-dev-22b",
                model: "ltx-2.3-22b-dev:fp8",
                pipeline: PipelineKind::TwoStage,
                enable_audio: Some(false),
                seed: 424_362,
                configure: parity_noop,
            },
            Ltx2ParityCase {
                workflow: "two-stage-hq-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::TwoStageHq,
                enable_audio: Some(false),
                seed: 424_363,
                configure: parity_noop,
            },
            Ltx2ParityCase {
                workflow: "spatial-x2-19b",
                model: "ltx-2-19b-dev:fp8",
                pipeline: PipelineKind::TwoStage,
                enable_audio: Some(false),
                seed: 424_371,
                configure: parity_spatial_x2,
            },
            Ltx2ParityCase {
                workflow: "spatial-x1.5-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(false),
                seed: 424_372,
                configure: parity_spatial_x1_5,
            },
            Ltx2ParityCase {
                workflow: "temporal-x2-19b",
                model: "ltx-2-19b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(false),
                seed: 424_381,
                configure: parity_temporal_x2,
            },
            Ltx2ParityCase {
                workflow: "temporal-x2-22b",
                model: "ltx-2.3-22b-distilled:fp8",
                pipeline: PipelineKind::Distilled,
                enable_audio: Some(false),
                seed: 424_382,
                configure: parity_temporal_x2,
            },
        ]
    }

    #[test]
    fn runtime_prepare_tracks_audio_and_video_latent_shapes() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = events.clone();
        let progress: ProgressCallback = Box::new(move |event| {
            captured.lock().unwrap().push(event);
        });
        let prepared = session
            .prepare_with_progress(&mut plan, Some(&progress))
            .unwrap();

        assert_eq!(prepared.video_pixel_shape.frames, 97);
        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(
            prepared.video_latent_shape.channels,
            LTX2_VIDEO_LATENT_CHANNELS
        );
        assert_eq!(prepared.video_latent_shape.frames, 13);
        assert_eq!(
            prepared.video_positions.dims4().unwrap(),
            (1, 3, 13 * 11 * 19, 2)
        );
        assert_eq!(
            prepared.audio_latent_shape.unwrap().channels,
            LTX2_AUDIO_LATENT_CHANNELS
        );
        assert!(prepared.audio_positions.is_some());
        assert!(prepared.cross_modal_temporal_positions.is_some());
        assert_eq!(
            prepared.prompt.conditional.video_encoding.dims3().unwrap(),
            (1, 3, 8)
        );

        let rendered = session
            .render_native_video(&plan, &prepared, Some(&progress), None)
            .unwrap();
        assert_eq!(rendered.frames.len(), 97);
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            ProgressEvent::PhaseDone {
                phase: ProgressPhase::PromptEncode,
                ..
            }
        )));
        assert!(
            !events.iter().any(|event| matches!(
                event,
                ProgressEvent::PhaseDone {
                    phase: ProgressPhase::Vae,
                    ..
                }
            )),
            "placeholder rendering must not invent a VAE timing sample"
        );
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
        assert!(rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, Some(48_000));
        assert_eq!(rendered.audio_channels, Some(2));
    }

    #[test]
    fn runtime_render_rejects_a_cancelled_attempt_before_work() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);
        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let cancellation = InferenceCancellationToken::default();
        cancellation.cancel();

        let error = session
            .render_native_video(&plan, &prepared, None, Some(&cancellation))
            .unwrap_err();

        assert!(crate::progress::is_inference_cancelled(&error));
    }

    #[test]
    fn runtime_prepare_keeps_av_audio_latents_for_silent_outputs() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Gif, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert!(prepared.audio_latent_shape.is_some());
        assert!(prepared.audio_positions.is_some());
        assert!(prepared.cross_modal_temporal_positions.is_some());

        let rendered = session
            .render_native_video(&plan, &prepared, None, None)
            .unwrap();
        assert_eq!(rendered.frames.len(), 97);
        assert!(!rendered.has_audio);
        assert_eq!(rendered.audio_sample_rate, None);
        assert_eq!(rendered.audio_channels, None);
    }

    /// A checkpoint that is present but unreadable is corrupt real weights —
    /// a truncated or interrupted download — not a test placeholder. Swapping
    /// it for the synthetic gradient hands the user a file with the requested
    /// size, length and frame rate that contains no picture, and reports it as
    /// a successful save, while hiding the corruption that caused it.
    #[test]
    fn corrupt_checkpoint_fails_loudly_instead_of_rendering_the_synthetic_gradient() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);

        // Four bytes is a present, non-empty file that is too short to hold a
        // safetensors header length. Candle surfaces the `safetensors` error
        // transparently, so this arrives verbatim as "header too small" —
        // exactly the text the old placeholder heuristic matched on.
        let checkpoint = temp_dir.path().join("ltx2-truncated.safetensors");
        std::fs::write(&checkpoint, [0u8; 4]).unwrap();
        plan.checkpoint_path = checkpoint.to_string_lossy().into_owned();
        plan.vae_checkpoint_path = plan.checkpoint_path.clone();

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let error = session
            .render_native_video(&plan, &prepared, None, None)
            .expect_err("a corrupt checkpoint must fail, not render placeholder frames");

        // The message has to name the checkpoint: "header too small" alone
        // gives the user nothing to act on.
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("ltx2-truncated.safetensors"),
            "error must name the offending checkpoint, got: {rendered}"
        );
    }

    /// The boundary the corrupt-checkpoint fix turns on: no bytes means no
    /// weights (a download that never landed, and what the tests stub), while
    /// any bytes at all are a checkpoint that must parse or fail.
    #[test]
    fn checkpoint_has_no_weights_separates_empty_from_truncated() {
        let temp_dir = tempfile::tempdir().unwrap();

        let missing = temp_dir.path().join("absent.safetensors");
        assert!(super::checkpoint_has_no_weights(&missing));

        let empty = temp_dir.path().join("empty.safetensors");
        std::fs::write(&empty, []).unwrap();
        assert!(super::checkpoint_has_no_weights(&empty));

        let truncated = temp_dir.path().join("truncated.safetensors");
        std::fs::write(&truncated, [0u8; 4]).unwrap();
        assert!(
            !super::checkpoint_has_no_weights(&truncated),
            "a short but non-empty checkpoint is corrupt weights, not an absent download"
        );

        assert!(super::checkpoint_has_no_weights(temp_dir.path()));
    }

    /// An unreadable checkpoint is not an absent one. A permission or transient
    /// I/O error must reach the loader — which names the file — rather than
    /// being classified as "nothing downloaded" and rendered as a gradient.
    #[test]
    fn stat_reports_no_weights_only_treats_not_found_as_absent() {
        use std::io::ErrorKind;

        assert!(super::stat_reports_no_weights(Err(ErrorKind::NotFound)));
        assert!(super::stat_reports_no_weights(Ok((true, 0))));
        assert!(super::stat_reports_no_weights(Ok((false, 4096))));
        assert!(!super::stat_reports_no_weights(Ok((true, 4))));

        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::Other,
            ErrorKind::InvalidInput,
        ] {
            assert!(
                !super::stat_reports_no_weights(Err(kind)),
                "{kind:?} is a real failure and must not select the placeholder path"
            );
        }
    }

    #[test]
    fn runtime_prepare_derives_retake_mask_from_request_range() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.retake_range = Some(TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.25,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let mask = prepared.retake_mask.unwrap();

        assert_eq!(mask.len(), 97);
        assert!(mask[..24].iter().all(|value| *value == 0.0));
        assert!(mask[24..54].iter().all(|value| *value == 1.0));
        assert!(mask[54..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_temporal_upscale() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.frames = Some(17);
        req.fps = Some(12);
        req.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let rendered = session
            .render_native_video(&plan, &prepared, None, None)
            .unwrap();

        assert_eq!(prepared.video_pixel_shape.frames, 9);
        assert_eq!(prepared.video_pixel_shape.fps as u32, 6);
        assert_eq!(rendered.frames.len(), 17);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_spatial_upscale() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();
        let rendered = session
            .render_native_video(&plan, &prepared, None, None)
            .unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn runtime_prepare_uses_stage_one_shape_for_x1_5_spatial_upscale() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 800);
        assert_eq!(prepared.video_pixel_shape.height, 480);
        assert_eq!(prepared.video_latent_shape.width, 25);
        assert_eq!(prepared.video_latent_shape.height, 15);
    }

    #[test]
    fn runtime_prepare_uses_half_resolution_shape_for_distilled_pipeline() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
    }

    #[test]
    fn runtime_prepare_aligns_implicit_two_stage_shape_to_latent_grid_for_odd_sizes() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.width = 608;
        req.height = 352;
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 320);
        assert_eq!(prepared.video_pixel_shape.height, 192);
        assert_eq!(prepared.video_latent_shape.width, 10);
        assert_eq!(prepared.video_latent_shape.height, 6);
    }

    #[test]
    fn runtime_prepare_aligns_explicit_x2_spatial_upscale_shape_to_latent_grid_for_odd_sizes() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.width = 608;
        req.height = 352;
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 320);
        assert_eq!(prepared.video_pixel_shape.height, 192);
        assert_eq!(prepared.video_latent_shape.width, 10);
        assert_eq!(prepared.video_latent_shape.height, 6);
    }

    #[test]
    fn runtime_render_native_video_accepts_bf16_prompt_encodings() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        let mut prepared = session.prepare(&plan).unwrap();
        prepared.prompt.conditional.video_encoding = prepared
            .prompt
            .conditional
            .video_encoding
            .to_dtype(DType::BF16)
            .unwrap();
        prepared.prompt.unconditional.video_encoding = prepared
            .prompt
            .unconditional
            .video_encoding
            .to_dtype(DType::BF16)
            .unwrap();
        prepared.prompt.conditional.audio_encoding = prepared
            .prompt
            .conditional
            .audio_encoding
            .take()
            .map(|tensor| tensor.to_dtype(DType::BF16).unwrap());

        let rendered = session
            .render_native_video(&plan, &prepared, None, None)
            .unwrap();

        assert_eq!(rendered.frames.len(), 97);
        assert_eq!(rendered.frames[0].dimensions(), (1216, 704));
    }

    #[test]
    fn decoded_video_to_frames_resizes_decoded_shape_to_requested_pixels() {
        let video = Tensor::zeros((1, 3, 2, 320, 544), DType::F32, &Device::Cpu).unwrap();
        let pixel_shape = VideoPixelShape {
            batch: 1,
            frames: 2,
            height: 352,
            width: 608,
            fps: 12.0,
        };

        let frames = decoded_video_to_frames(&video, pixel_shape).unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].dimensions(), (608, 352));
        assert_eq!(frames[1].dimensions(), (608, 352));
    }

    #[test]
    fn runtime_video_transformer_config_matches_upstream_av_gate_timestep_scale() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let config = ltx2_video_transformer_config(&plan);

        assert_eq!(config.av_ca_timestep_scale_multiplier, 1000.0);
        assert_eq!(config.streaming_prefetch_count, 2);
    }

    #[test]
    fn velocity_x0_roundtrip_preserves_sample_velocity_pair() {
        let sample = Tensor::new(&[[10.0f32, 4.0]], &Device::Cpu).unwrap();
        let velocity = Tensor::new(&[[2.0f32, -1.0]], &Device::Cpu).unwrap();
        let sigma = 0.5f32;

        let x0 = convert_velocity_to_x0(&sample, &velocity, sigma).unwrap();
        let roundtrip = convert_x0_to_velocity(&sample, &x0, sigma).unwrap();

        let values = roundtrip.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-5);
        assert!((values[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn cfg_guidance_is_applied_in_x0_space_before_velocity_conversion() {
        let sample = Tensor::new(&[[10.0f32]], &Device::Cpu).unwrap();
        let conditional_velocity = Tensor::new(&[[2.0f32]], &Device::Cpu).unwrap();
        let unconditional_velocity = Tensor::new(&[[4.0f32]], &Device::Cpu).unwrap();

        let guided = guided_velocity_from_cfg(
            &sample,
            &conditional_velocity,
            &unconditional_velocity,
            0.5,
            3.0,
        )
        .unwrap();
        let value = guided.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

        assert!((value + 2.0).abs() < 1e-5);
    }

    #[test]
    fn denoiser_helpers_cast_velocity_and_denoised_to_sample_dtype() {
        let sample = Tensor::new(&[[[10.0f32, 4.0]]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let velocity = Tensor::new(&[[[2.0f32, -1.0]]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let sigma = Tensor::new(&[[0.5f32]], &Device::Cpu).unwrap();

        let denoised =
            super::denoised_from_velocity_with_sigma(&sample, &velocity, &sigma).unwrap();
        let restored = super::velocity_from_denoised(&sample, &denoised, 0.5).unwrap();

        assert_eq!(denoised.dtype(), DType::F32);
        assert_eq!(restored.dtype(), DType::F32);
        let values = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-3);
        assert!((values[1] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn step_velocity_inspection_is_debug_only() {
        assert!(!should_inspect_step_velocity(None));
        assert!(should_inspect_step_velocity(Some("stage1")));
    }

    #[test]
    fn distilled_runtime_forces_simple_denoiser_guidance() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert_eq!(plan.guidance, 3.0);
        assert_eq!(effective_native_guidance_scale(&plan), 1.0);
    }

    #[test]
    fn non_distilled_runtime_preserves_requested_guidance() {
        let req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.guidance = 4.5;

        assert_eq!(effective_native_guidance_scale(&plan), 4.5);
    }

    #[test]
    fn distilled_runtime_skips_unconditional_prompt_encoding() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn ic_lora_runtime_skips_unconditional_prompt_encoding() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn two_stage_runtime_keeps_unconditional_prompt_encoding_for_multimodal_guidance() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn a2vid_runtime_keeps_unconditional_prompt_encoding_for_multimodal_guidance() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::prompt_requires_unconditional_context_for_plan(&plan).unwrap());
    }

    #[test]
    fn stage_unconditional_context_follows_multimodal_guidance_at_guidance_one() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.guidance = 1.0;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::stage_requires_unconditional_context(&plan, 0).unwrap());
        assert!(!super::stage_requires_unconditional_context(&plan, 1).unwrap());
    }

    #[test]
    fn multimodal_guidance_batch_prebuilds_static_contexts_once() {
        let device = Device::Cpu;
        let cond_video = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let uncond_video = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
        let cond_audio = Tensor::zeros((1, 4, 5), DType::F32, &device).unwrap();
        let video_guider = crate::ltx2::guidance::MultiModalGuider::new(
            crate::ltx2::guidance::MultiModalGuiderParams {
                cfg_scale: 4.0,
                stg_scale: 1.5,
                stg_blocks: vec![2, 3],
                modality_scale: 1.25,
                ..Default::default()
            },
            Some(uncond_video),
        );
        let audio_guider = crate::ltx2::guidance::MultiModalGuider::new(
            crate::ltx2::guidance::MultiModalGuiderParams {
                stg_scale: 0.5,
                stg_blocks: vec![7],
                ..Default::default()
            },
            None,
        );

        let batch = build_static_multimodal_guidance_batch(
            &cond_video,
            Some(&cond_audio),
            None,
            None,
            &video_guider,
            &audio_guider,
        )
        .unwrap();

        assert_eq!(batch.repeat_count, 4);
        assert_eq!(batch.cond_index, 0);
        assert_eq!(batch.uncond_index, Some(1));
        assert_eq!(batch.perturbed_index, Some(2));
        assert_eq!(batch.modality_index, Some(3));
        assert_eq!(batch.batched_video_context.dims3().unwrap(), (4, 2, 3));
        assert_eq!(
            batch
                .batched_audio_context
                .as_ref()
                .unwrap()
                .dims3()
                .unwrap(),
            (4, 4, 5)
        );
        assert_eq!(
            batch.perturbations.mask_values(
                crate::ltx2::guidance::PerturbationType::SkipVideoSelfAttention,
                2
            ),
            vec![1.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn multimodal_guidance_batch_omits_unneeded_optional_contexts() {
        let device = Device::Cpu;
        let cond_video = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let video_guider = crate::ltx2::guidance::MultiModalGuider::new(
            crate::ltx2::guidance::MultiModalGuiderParams::default(),
            None,
        );
        let audio_guider = crate::ltx2::guidance::MultiModalGuider::new(
            crate::ltx2::guidance::MultiModalGuiderParams::default(),
            None,
        );

        let batch = build_static_multimodal_guidance_batch(
            &cond_video,
            None,
            None,
            None,
            &video_guider,
            &audio_guider,
        )
        .unwrap();

        assert_eq!(batch.repeat_count, 1);
        assert_eq!(batch.uncond_index, None);
        assert_eq!(batch.batched_video_context.dims3().unwrap(), (1, 2, 3));
        assert!(batch.batched_audio_context.is_none());
        assert!(batch.batched_video_mask.is_none());
        assert!(batch.batched_audio_mask.is_none());
    }

    #[test]
    fn runtime_session_prepare_consumes_prompt_encoder() {
        // The encoder is still consumed on first prepare() — the encoder
        // slot moves out to free VRAM for the transformer. But same-prompt
        // follow-up calls now short-circuit through `cached_prompt_encoding`
        // so chain stages that replicate the prompt can reuse the session
        // instead of erroring on a consumed encoder.
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        session.prepare(&plan).unwrap();

        // Encoder slot is empty post-take.
        assert!(session.prompt_encoder.is_none());
        // But `can_reuse_for` reports true because the cached encoding
        // matches the incoming plan's prompt tokens.
        assert!(session.can_reuse_for(&plan));
        // Same-prompt re-prepare succeeds from the cache.
        session
            .prepare(&plan)
            .expect("same-prompt cache hit must succeed");
    }

    #[test]
    fn runtime_session_prepare_rejects_encoder_reuse_with_different_prompt() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        let mut session = runtime_session();
        session.prepare(&plan).unwrap();

        // Mutate the plan's prompt tokens so the cache key misses.
        let mut plan_alt = plan.clone();
        plan_alt.prompt_tokens.conditional.input_ids[0] =
            plan_alt.prompt_tokens.conditional.input_ids[0].wrapping_add(1);

        // can_reuse_for must report false for a fresh prompt because the
        // encoder has already been consumed.
        assert!(!session.can_reuse_for(&plan_alt));
        // And prepare() with the new plan fails explicitly so the caller
        // knows to drop the session and rebuild.
        assert!(session.prepare(&plan_alt).is_err());
    }

    #[test]
    fn remap_ltx2_transformer_key_rewrites_only_exact_path_segments() {
        assert_eq!(
            super::remap_ltx2_transformer_key("proj_in.weight"),
            "model.diffusion_model.patchify_proj.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.norm_q.weight"),
            "model.diffusion_model.blocks.0.q_norm.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.patchify_proj_in.weight"),
            "model.diffusion_model.blocks.0.patchify_proj_in.weight"
        );
        assert_eq!(
            super::remap_ltx2_transformer_key("blocks.0.norm_q_extra.weight"),
            "model.diffusion_model.blocks.0.norm_q_extra.weight"
        );
    }

    #[test]
    fn ltx2_block_size_discovery_groups_transformer_tensors_after_remap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ltx2-blocks.safetensors");
        let block0 = vec![0u8; 4 * SafeDtype::F32.size()];
        let block1_a = vec![0u8; 6 * SafeDtype::F16.size()];
        let block1_b = vec![0u8; 2 * SafeDtype::F32.size()];
        let non_block = vec![0u8; 3 * SafeDtype::F32.size()];
        let mut tensors = HashMap::new();
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &block0).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.transformer_blocks.1.ff.net.0.proj.weight".to_string(),
            TensorView::new(SafeDtype::F16, vec![2, 3], &block1_a).unwrap(),
        );
        tensors.insert(
            "blocks.1.norm_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2], &block1_b).unwrap(),
        );
        tensors.insert(
            "caption_projection.linear_1.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![3], &non_block).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let index = Ltx2TransformerWeightIndex::read(&path).unwrap();
        let sizes =
            super::ltx2_transformer_block_sizes_from_safetensors(&path, 3, DType::F32).unwrap();

        assert_eq!(index.block_bytes_at_rest(), vec![16, 20]);
        // Resident bytes are priced at the compute dtype: block 1's F16
        // tensor widens to F32 on a CPU device (6 × 4 + 2 × 4).
        assert_eq!(sizes.blocks, vec![16, 32, 0]);
        assert_eq!(sizes.transient_bytes, 0);
    }

    /// Everything under the transformer that isn't a block — `patchify_proj`,
    /// `adaln_single.linear`, `caption_projection`, the connectors, `proj_out`
    /// — is allocated on the GPU after every resident block. The 19B FP8
    /// checkpoint carries 2.1 GB of it; discarding the total is how the
    /// planner overshot a 24 GB card by exactly that much.
    #[test]
    fn block_sizes_from_safetensors_reports_non_block_total() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ltx2-non-block.safetensors");
        let block0 = vec![0u8; 4 * SafeDtype::F32.size()];
        let patchify = vec![0u8; 6 * SafeDtype::F32.size()];
        let adaln = vec![0u8; 8 * SafeDtype::F32.size()];
        let vae = vec![0u8; 32 * SafeDtype::F32.size()];
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &block0).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.patchify_proj.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 3], &patchify).unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.adaln_single.linear.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 4], &adaln).unwrap(),
        );
        // A combined export also carries the VAE; it is not transformer
        // residency and must not be charged to it.
        tensors.insert(
            "vae.encoder.conv_in.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![8, 4], &vae).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        let sizes =
            super::ltx2_transformer_block_sizes_from_safetensors(&path, 1, DType::F32).unwrap();

        assert_eq!(sizes.blocks, vec![16]);
        assert_eq!(
            sizes.non_block_bytes,
            24 + 32,
            "patchify_proj + adaln_single must be reported, vae must not"
        );
    }

    /// INT8 ConvRot is priced at the BF16 size the loader actually
    /// materializes on the device — the raw I8 span is exactly half of it,
    /// which is how the 2.5 int8-conv pack was under-counted by 2×. The
    /// figures come from the golden 2.5 distilled int8-convrot header
    /// (`crates/mold-core/testdata/ltx25`): blocks 0 and 47 are complete,
    /// 1..46 were cut, and the AdaLN width is the exact video key, not the
    /// `prompt_adaln_single` look-alike.
    #[test]
    fn ltx2_int8_convrot_blocks_are_sized_widened_from_the_shared_index() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mold-core/testdata/ltx25/distilled-int8-convrot.header.safetensors");
        let index = Ltx2TransformerWeightIndex::read(&path).unwrap();
        assert_eq!(index.block_bytes_at_rest()[0], 388_065_632);

        let sizes = super::ltx2_transformer_weight_sizes(
            &index,
            48,
            DType::BF16,
            mold_core::ltx2_weight_index::Ltx2ResidentWeightForm::Widened,
        )
        .unwrap();

        assert_eq!(sizes.blocks.len(), 48);
        assert!(!sizes.int8_packed);
        assert_eq!(sizes.blocks[0], 773_349_760);
        assert_eq!(sizes.blocks[47], 773_349_760);
        assert_eq!(sizes.blocks[1], 0);
        assert_eq!(sizes.non_block_bytes, 4_887_262_720);
        assert_eq!(sizes.adaln_dim, Some(36_864));
        assert_eq!(sizes.transient_bytes, 134_217_728);
    }

    /// On CUDA the ConvRot blocks stay packed (`LtxLinear::ConvRotPacked`):
    /// a resident block costs its at-rest bytes, roughly half the widened
    /// figure, and the sizes flag the token-scaled W8A8 workspace.
    #[test]
    fn ltx2_int8_convrot_blocks_are_sized_packed_on_cuda() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mold-core/testdata/ltx25/distilled-int8-convrot.header.safetensors");
        let index = Ltx2TransformerWeightIndex::read(&path).unwrap();
        let packed = super::ltx2_transformer_weight_sizes(
            &index,
            48,
            DType::BF16,
            mold_core::ltx2_weight_index::Ltx2ResidentWeightForm::Packed,
        )
        .unwrap();
        let widened = super::ltx2_transformer_weight_sizes(
            &index,
            48,
            DType::BF16,
            mold_core::ltx2_weight_index::Ltx2ResidentWeightForm::Widened,
        )
        .unwrap();

        assert!(packed.int8_packed);
        assert_eq!(
            packed.blocks[0] as u64,
            index.block_bytes_packed(2)[0],
            "packed residency prices the at-rest form"
        );
        assert!(
            (packed.blocks[0] as f64) < 0.55 * widened.blocks[0] as f64,
            "packed ({}) must be roughly half of widened ({})",
            packed.blocks[0],
            widened.blocks[0]
        );
        // Non-block weights widen on every backend (the quantized ones are
        // the prompt encoder's connectors).
        assert_eq!(packed.non_block_bytes, widened.non_block_bytes);
        assert_eq!(packed.transient_bytes, widened.transient_bytes);
    }

    /// The adaptive plan reserves the per-forward transient beside the fixed
    /// weights, and the W8A8 workspace beside the activations — packed
    /// ConvRot only.
    #[test]
    fn ltx2_adaptive_plan_reserves_transient_and_w8a8_workspace() {
        let stage = super::Ltx2StageShape {
            width: 1216,
            height: 704,
            frames: 121,
            conditioned: false,
        };
        let weights = |int8_packed| super::Ltx2TransformerWeightSizes {
            blocks: vec![400_000_000; 4],
            non_block_bytes: 1_000_000_000,
            adaln_dim: Some(36_864),
            transient_bytes: 134_217_728,
            int8_packed,
        };
        let plan = |int8_packed| {
            super::ltx2_adaptive_transformer_plan(
                &ltx2_plan_at(1216, 704, 121),
                stage,
                &weights(int8_packed),
                24_000_000_000,
            )
        };
        let dense = plan(false);
        let packed = plan(true);
        assert_eq!(
            dense.fixed_resident_bytes,
            1_000_000_000 + 134_217_728,
            "transient is reserved with the fixed weights"
        );
        let workspace = mold_core::ltx2_weight_index::ltx2_int8_w8a8_workspace_bytes(
            crate::device::ltx2_token_count(1216, 704, 121),
        );
        assert_eq!(
            packed.reserved_bytes(),
            dense.reserved_bytes() + workspace,
            "packed ConvRot additionally reserves the W8A8 forward workspace"
        );
    }

    /// GGUF blocks stay at their packed ggml size; NVFP4 keeps refusing so it
    /// stays on the streaming path it always took.
    #[test]
    fn ltx2_gguf_blocks_are_sized_packed_and_nvfp4_stays_unmodelled() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mold-core/testdata/ltx25/distilled-q4-k-m.header.gguf");
        let index = Ltx2TransformerWeightIndex::read(&path).unwrap();
        let sizes = super::ltx2_transformer_weight_sizes(
            &index,
            48,
            DType::BF16,
            mold_core::ltx2_weight_index::Ltx2ResidentWeightForm::Packed,
        )
        .unwrap();
        assert_eq!(index.block_bytes_at_rest()[0], 249_582_336);
        assert_eq!(sizes.blocks[0], 249_164_160);
        assert_eq!(sizes.blocks.iter().sum::<usize>(), 12_603_951_104);
        assert_eq!(sizes.adaln_dim, Some(36_864));
        assert_eq!(sizes.transient_bytes, 402_653_184);
        assert!(!super::ltx2_checkpoint_is_fp8(
            &ltx2_plan_at(512, 512, 9),
            Some(&index)
        ));

        let nvfp4 = br#"{"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight":{"dtype":"U8","shape":[4096,2048],"data_offsets":[0,8388608]},"model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale_2":{"dtype":"F32","shape":[],"data_offsets":[8388608,8388612]}}"#;
        let index = Ltx2TransformerWeightIndex::from_safetensors_header(nvfp4).unwrap();
        assert!(super::ltx2_transformer_weight_sizes(
            &index,
            1,
            DType::BF16,
            mold_core::ltx2_weight_index::Ltx2ResidentWeightForm::Widened,
        )
        .is_err());
    }

    /// The residency planner must reserve those non-block weights instead of
    /// allocating them after a plan that already filled the card.
    #[test]
    fn ltx2_adaptive_plan_reserves_non_block_transformer_weights() {
        let plan = ltx2_plan_at(1024, 1024, 97);
        let stage = stage_shape(&plan, 1024, 1024, 97);
        let weights = super::Ltx2TransformerWeightSizes {
            int8_packed: false,
            blocks: ltx2_19b_fp8_blocks(),
            non_block_bytes: 2_107_091_456,
            adaln_dim: Some(24_576),
            transient_bytes: 0,
        };
        const FREE_VRAM: u64 = 25_339_395_072;

        let residency = super::ltx2_adaptive_transformer_plan(&plan, stage, &weights, FREE_VRAM);

        assert_eq!(residency.fixed_resident_bytes, weights.non_block_bytes);
        let true_demand = residency.resident_bytes
            + weights.non_block_bytes
            + super::ltx2_video_activation_budget(stage, weights.adaln_dim)
            + super::ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM
            + residency.largest_streamed_block;
        assert!(
            true_demand <= FREE_VRAM,
            "true demand {true_demand} must fit {FREE_VRAM}"
        );
    }

    /// Stage 1 of a Distilled 1024² plan renders at 512²; charging it the
    /// full-frame budget reserved 4× the tokens it actually uses.
    #[test]
    fn ltx2_activation_budget_uses_stage_shape() {
        let plan = ltx2_plan_at(1024, 1024, 97);
        let stage1 = stage_shape(&plan, 512, 512, 97);
        let stage2 = stage_shape(&plan, 1024, 1024, 97);

        let stage1_budget = super::ltx2_video_activation_budget(stage1, None);
        let stage2_budget = super::ltx2_video_activation_budget(stage2, None);

        assert_eq!(
            stage1_budget,
            super::ltx2_activation_budget_bytes(512, 512, 97, stage1.conditioned, None)
        );
        assert!(
            stage1_budget < stage2_budget,
            "half-resolution stage 1 ({stage1_budget}) must cost less than stage 2 ({stage2_budget})"
        );
    }

    /// Sampled free VRAM is a ceiling, not an entitlement. A job admitted at a
    /// smaller peak must size itself to that grant.
    #[test]
    fn ltx2_transformer_budget_respects_scheduler_grant() {
        assert_eq!(super::ltx2_transformer_vram_budget(None, 25_000), 25_000);
        assert_eq!(
            super::ltx2_transformer_vram_budget(Some(11_000), 25_000),
            11_000
        );
        // A grant above what the card actually has is still bounded by the card.
        assert_eq!(
            super::ltx2_transformer_vram_budget(Some(40_000), 25_000),
            25_000
        );
    }

    /// The grant has to change the plan, not just the arithmetic.
    #[test]
    fn ltx2_grant_produces_fewer_resident_blocks_than_free_vram() {
        const FREE_VRAM: u64 = 25_339_395_072;
        let weights = super::Ltx2TransformerWeightSizes {
            int8_packed: false,
            blocks: ltx2_19b_fp8_blocks(),
            non_block_bytes: 2_107_091_456,
            adaln_dim: Some(24_576),
            transient_bytes: 0,
        };
        let ungranted = ltx2_plan_at(1024, 1024, 97);
        let stage = stage_shape(&ungranted, 1024, 1024, 97);
        let mut granted = ungranted.clone();
        granted.vram_grant_bytes = Some(16_000_000_000);

        let without = super::ltx2_adaptive_transformer_plan(&ungranted, stage, &weights, FREE_VRAM);
        let with = super::ltx2_adaptive_transformer_plan(&granted, stage, &weights, FREE_VRAM);

        assert!(
            with.resident_count() < without.resident_count(),
            "grant of 16 GB must keep fewer than the {} blocks a 25 GB card allows, got {}",
            without.resident_count(),
            with.resident_count()
        );
        assert!(with.peak_bytes() <= 16_000_000_000);
    }

    /// The denoise ladder shrinks the budget and eventually gives up instead
    /// of looping forever.
    #[test]
    fn ltx2_denoise_retry_vram_budget_shrinks_then_stops() {
        let first = super::ltx2_denoise_retry_vram_budget(24_000_000_000).unwrap();
        assert!(first < 24_000_000_000);
        let second = super::ltx2_denoise_retry_vram_budget(first).unwrap();
        assert!(second < first);
        assert_eq!(super::ltx2_denoise_retry_vram_budget(0), None);
        assert_eq!(
            super::ltx2_denoise_retry_vram_budget(super::ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM),
            None
        );
    }

    /// A fatal CUDA fault must never be treated as a recoverable OOM — the
    /// quarantine-and-stop rule is unchanged.
    #[test]
    fn ltx2_denoise_never_retries_a_fatal_cuda_error() {
        let oom = anyhow::anyhow!("DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)");
        assert!(super::ltx2_denoise_error_is_recoverable_oom(&oom));

        let fatal = anyhow::anyhow!(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS) while allocating; out of memory"
        );
        assert!(super::ltx2_error_is_fatal_cuda(&format!("{fatal:#}")));
        assert!(!super::ltx2_denoise_error_is_recoverable_oom(&fatal));

        let ordinary = anyhow::anyhow!("shape mismatch");
        assert!(!super::ltx2_denoise_error_is_recoverable_oom(&ordinary));
    }

    #[test]
    fn ltx2_transformer_residency_defaults_cuda_fp8_to_adaptive() {
        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Cuda,
                true,
                false,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Adaptive
        );
    }

    /// Metal reached `Streaming` through a blanket "not CUDA" arm that ran
    /// *before* the eager one, so `MOLD_LTX2_FORCE_EAGER` was dead code on
    /// Apple Silicon and a resident transformer could not even be measured —
    /// while streaming re-materialises all 48 blocks (20.86 GB) from the mmap
    /// on every denoise pass, for a pool the host already shares. The default
    /// deliberately stays `Streaming`; this only makes the escape hatch real.
    #[test]
    fn ltx2_transformer_residency_honours_an_explicit_eager_request_on_metal() {
        let mode = |force_eager, force_streaming| {
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Metal,
                true,
                force_eager,
                force_streaming,
                true,
                24_000_000_000,
            )
        };

        assert_eq!(
            mode(true, false),
            super::Ltx2TransformerResidencyMode::Eager
        );
        assert_eq!(
            mode(false, false),
            super::Ltx2TransformerResidencyMode::Streaming,
            "Metal must keep streaming by default until a resident window is sized"
        );
        assert_eq!(
            mode(true, true),
            super::Ltx2TransformerResidencyMode::Streaming,
            "an explicit streaming request still outranks an eager one"
        );

        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Metal,
                false,
                true,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Streaming,
            "eager residency is an FP8-checkpoint contract, not a Metal one"
        );

        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Other,
                true,
                true,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Streaming,
            "CPU has no device memory to be resident in"
        );

        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Metal,
                true,
                false,
                false,
                false,
                0
            ),
            super::Ltx2TransformerResidencyMode::Streaming,
            "Metal never reaches the CUDA adaptive plan"
        );
    }

    /// `MOLD_LTX2_FORCE_STREAMING` was read with `is_some()`, so `0` and
    /// `false` both switched streaming on — the opposite of the value, and
    /// unlike the `MOLD_OFFLOAD` alias parsed right beside it.
    #[test]
    fn ltx2_force_streaming_reads_its_value_rather_than_its_presence() {
        let from = super::ltx2_force_streaming_from_values;

        assert!(from(Some("1"), None));
        assert!(from(Some("true"), None));
        assert!(from(Some("YES"), None));
        assert!(from(Some(" on "), None));
        assert!(from(None, Some("1")));

        assert!(!from(Some("0"), None));
        assert!(!from(Some("false"), None));
        assert!(!from(Some(""), None));
        assert!(!from(None, Some("0")));
        assert!(!from(None, None));
    }

    /// The budget still scales with every simultaneously live latent frame —
    /// 97 pixel frames produce 13 of them — but the per-frame cost is now the
    /// real token slope rather than the pixel-area heuristic.
    #[test]
    fn ltx2_adaptive_residency_reserves_every_live_latent_frame() {
        let plan = ltx2_plan_at(1024, 1024, 97);
        let budget = |frames| {
            super::ltx2_video_activation_budget(stage_shape(&plan, 1024, 1024, frames), None)
        };

        // 1 pixel frame → 1 latent frame; 9 → 2; 97 → 13.
        let one_latent_frame = budget(9) - budget(1);
        assert!(one_latent_frame > 0);
        assert_eq!(
            budget(97) - budget(1),
            one_latent_frame * 12,
            "97 pixel frames produce 13 simultaneously live latent frames"
        );
    }

    #[test]
    fn ltx2_transformer_residency_force_streaming_wins() {
        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Cuda,
                true,
                true,
                true,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Streaming
        );
    }

    #[test]
    fn ltx2_convrot_forces_streaming_for_reconstructed_bf16_weights() {
        // ConvRot no longer forces streaming on CUDA — packed residency
        // exists there — but still does on Metal and CPU, which widen.
        assert!(!super::ltx2_effective_force_streaming(
            false,
            true,
            super::Ltx2Accelerator::Cuda
        ));
        assert!(super::ltx2_effective_force_streaming(
            false,
            true,
            super::Ltx2Accelerator::Metal
        ));
        assert!(super::ltx2_effective_force_streaming(
            false,
            true,
            super::Ltx2Accelerator::Other
        ));
        assert!(super::ltx2_effective_force_streaming(
            true,
            false,
            super::Ltx2Accelerator::Cuda
        ));
        assert!(!super::ltx2_effective_force_streaming(
            false,
            false,
            super::Ltx2Accelerator::Cuda
        ));
    }

    #[test]
    fn ltx2_transformer_residency_force_eager_is_explicit_cuda_fp8_only() {
        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Cuda,
                true,
                true,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Eager
        );
        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Cuda,
                false,
                true,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Adaptive
        );
        assert_eq!(
            super::select_ltx2_transformer_residency_mode(
                super::Ltx2Accelerator::Other,
                true,
                true,
                false,
                true,
                24_000_000_000
            ),
            super::Ltx2TransformerResidencyMode::Streaming
        );
    }

    #[test]
    fn one_stage_runtime_keeps_requested_full_resolution_shape() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 1216);
        assert_eq!(prepared.video_pixel_shape.height, 704);
        assert_eq!(prepared.video_latent_shape.width, 38);
        assert_eq!(prepared.video_latent_shape.height, 22);
    }

    #[test]
    fn retake_runtime_keeps_requested_full_resolution_shape() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.retake_range = Some(TimeRange {
            start_seconds: 1.0,
            end_seconds: 2.0,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Retake;
        rebuild_execution_graph(&mut plan, &req);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 1216);
        assert_eq!(prepared.video_pixel_shape.height, 704);
        assert_eq!(prepared.video_latent_shape.width, 38);
        assert_eq!(prepared.video_latent_shape.height, 22);
    }

    #[test]
    fn ic_lora_runtime_keeps_requested_stage1_shape() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        let mut session = runtime_session();
        let prepared = session.prepare(&plan).unwrap();

        assert_eq!(prepared.video_pixel_shape.width, 608);
        assert_eq!(prepared.video_pixel_shape.height, 352);
        assert_eq!(prepared.video_latent_shape.width, 19);
        assert_eq!(prepared.video_latent_shape.height, 11);
    }

    #[test]
    fn supports_real_video_path_accepts_plain_silent_one_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_plain_audio_one_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_source_image_distilled_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert!(source_image_only_conditioning(&plan));
        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_source_image_distilled_lora_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/camera-control.safetensors".to_string(),
            scale: 0.63,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert!(source_image_only_conditioning(&plan));
        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();
        assert_eq!(stage1_loras, plan.loras);
        assert_eq!(stage2_loras, plan.loras);
        assert_eq!(stage1_loras[0].scale, 0.63);
        assert_eq!(stage2_loras[0].scale, 0.63);
        assert!(super::supports_real_video_path(&plan));
    }

    /// The exact combination users reported as a rainbow gradient: a distilled
    /// image-to-video run carrying a *stack* of two camera-control adapters.
    /// Both must survive into both denoising stages in request order — dropping
    /// one silently changes the motion, and refusing the plan used to swap the
    /// whole render for synthetic frames.
    #[test]
    fn supports_real_video_path_accepts_two_stacked_loras_with_a_source_image() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.source_image = Some(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        req.loras = Some(vec![
            LoraWeight {
                path: "/tmp/camera-control-dolly-in.safetensors".to_string(),
                scale: 0.8,

                expert: None,
            },
            LoraWeight {
                path: "/tmp/camera-control-jib-up.safetensors".to_string(),
                scale: 0.5,

                expert: None,
            },
        ]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let plan = build_plan(&req, preset, conditioning);

        assert!(source_image_only_conditioning(&plan));
        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();
        assert_eq!(stage1_loras, plan.loras);
        assert_eq!(stage2_loras, plan.loras);
        assert_eq!(stage1_loras.len(), 2);
        assert_eq!(stage1_loras[0].scale, 0.8);
        assert_eq!(stage1_loras[1].scale, 0.5);
        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_keyframe_two_stage_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.keyframes = Some(vec![
            mold_core::KeyframeCondition {
                frame: 8,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
                name: None,
            },
            mold_core::KeyframeCondition {
                frame: 48,
                image: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
                name: None,
            },
        ]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Keyframe;
        rebuild_execution_graph(&mut plan, &req);

        assert!(keyframe_only_conditioning(&plan));
        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_retake_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.retake_range = Some(TimeRange {
            start_seconds: 0.5,
            end_seconds: 1.25,
        });
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Retake;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn temporal_token_denoise_mask_marks_only_overlapping_tokens() {
        let positions = Tensor::from_vec(
            vec![0.0f32, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0],
            (1, 1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let range = TimeRange {
            start_seconds: 0.75,
            end_seconds: 1.6,
        };

        let mask =
            super::build_temporal_token_denoise_mask(&range, &positions, &Device::Cpu).unwrap();

        assert_eq!(
            mask.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 1.0, 1.0, 1.0]]
        );
    }

    #[test]
    fn timestep_from_sigma_and_mask_defaults_to_full_sigma_without_mask() {
        let timestep = super::timestep_from_sigma_and_mask(0.75, 2, None, &Device::Cpu).unwrap();

        assert_eq!(timestep.to_vec1::<f32>().unwrap(), vec![0.75, 0.75]);
    }

    #[test]
    fn timestep_from_sigma_and_mask_scales_per_token_when_masked() {
        let mask = Tensor::from_vec(vec![0.0f32, 0.25, 1.0], (1, 3), &Device::Cpu).unwrap();

        let timestep =
            super::timestep_from_sigma_and_mask(0.8, 1, Some(&mask), &Device::Cpu).unwrap();

        assert_eq!(
            timestep.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 0.2, 0.8]]
        );
    }

    #[test]
    fn resize_tail_frames_to_pixel_shape_downscales_for_stage1_half_res_grid() {
        use image::RgbImage;

        // Simulate a chain carryover: the emitting stage decoded 4 pixel
        // frames at full output resolution (1024×1024) and the receiving
        // stage's distilled pipeline will run stage 1 at the implicit X2
        // downsampled resolution (512×512). Without the resize, VAE-
        // encoding these tail frames at 1024×1024 produces a 32×32 spatial
        // grid per latent frame — which exceeds stage 1's 16×16 grid and
        // triggers the "conditioning replacement exceeds video token
        // count" bail in `apply_video_token_replacements`.
        let full_res_tail: Vec<RgbImage> = (0..4).map(|_| RgbImage::new(1024, 1024)).collect();

        let resized = resize_tail_frames_to_pixel_shape(&full_res_tail, 512, 512);
        assert_eq!(resized.len(), 4);
        for frame in &resized {
            assert_eq!(frame.width(), 512);
            assert_eq!(frame.height(), 512);
        }
    }

    #[test]
    fn resize_tail_frames_to_pixel_shape_is_noop_when_dims_match() {
        use image::RgbImage;

        // Stage 2 of the distilled pipeline passes the full-resolution
        // pixel_shape; the tail frames are already at that resolution, so
        // the resize must be a cheap clone — no filtering artifacts from
        // resampling a frame onto itself.
        let frame = RgbImage::from_pixel(1024, 1024, image::Rgb([200, 50, 120]));
        let tail = vec![frame.clone(), frame.clone()];

        let resized = resize_tail_frames_to_pixel_shape(&tail, 1024, 1024);
        assert_eq!(resized.len(), 2);
        for (original, passed_through) in tail.iter().zip(resized.iter()) {
            assert_eq!(passed_through.width(), 1024);
            assert_eq!(passed_through.height(), 1024);
            // Pixel-exact equality proves no resampling happened.
            assert_eq!(passed_through.as_raw(), original.as_raw());
        }
    }

    #[test]
    fn apply_video_token_replacements_blends_source_tokens_into_sequence() {
        let latents = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let replacement_tokens =
            Tensor::from_vec(vec![10.0f32, 20.0], (1, 1, 2), &Device::Cpu).unwrap();
        let replacement = VideoTokenReplacement {
            start_token: 1,
            tokens: replacement_tokens,
            strength: 0.25,
        };

        let replaced = apply_video_token_replacements(&latents, &[replacement]).unwrap();
        let values = replaced.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(values, vec![0.0, 1.0, 4.0, 7.25, 4.0, 5.0]);
    }

    /// #1055 witness: the initial conditioned latent handed to the first
    /// transformer pass must hold exactly `s·C + (1-s)·N` at replacement
    /// tokens — the single upstream noiser blend — while the denoise mask
    /// (and therefore the per-token timestep) says `1-s`. Before the fix
    /// the pre-loop freeze blend re-applied the clean target, producing
    /// `(2s-s²)·C + (1-s)²·N` under an unchanged `(1-s)·σ` timestep.
    #[test]
    fn initialized_latents_hold_single_blend_at_soft_strengths() {
        let noise = Tensor::from_vec(
            vec![2.0f32, -4.0, 6.0, -8.0, 10.0, -12.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let positions = Tensor::zeros((1, 3, 3, 2), DType::F32, &Device::Cpu).unwrap();
        for strength in [0.0f64, 0.25, 0.75, 1.0] {
            let source = Tensor::from_vec(vec![100.0f32, 200.0], (1, 1, 2), &Device::Cpu).unwrap();
            let conditioning = StageVideoConditioning {
                replacements: vec![VideoTokenReplacement {
                    start_token: 0,
                    tokens: source.clone(),
                    strength,
                }],
                appended: vec![],
            };
            let init = initialize_video_stage_latents(
                &noise,
                &positions,
                &conditioning,
                42,
                None,
                None,
                3,
                &Device::Cpu,
            )
            .unwrap();
            let values = init
                .latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let s = strength as f32;
            // Replacement token (index 0..2): exactly s·C + (1-s)·N.
            assert_eq!(
                &values[..2],
                &[s * 100.0 + (1.0 - s) * 2.0, s * 200.0 + (1.0 - s) * -4.0],
                "strength {strength}: initialized latent is not the single blend"
            );
            // Unconditioned tokens keep the pure start noise.
            assert_eq!(&values[2..], &[6.0, -8.0, 10.0, -12.0]);
            // Clean target stays the pure source.
            let clean = init.clean.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(&clean[..2], &[100.0, 200.0]);
            // Mask (→ timestep scale) agrees with the actual noise fraction.
            let mask = init
                .denoise_mask
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(mask, vec![1.0 - s, 1.0, 1.0]);
        }
    }

    /// The per-token timestep derived from the init's mask must describe
    /// the noise actually present in the initialized latent: `(1-s)·σ`
    /// for a latent whose noise coefficient is exactly `(1-s)`.
    #[test]
    fn initialized_timesteps_match_actual_noise_fraction() {
        let noise = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::zeros((1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let source = Tensor::from_vec(vec![0.0f32, 0.0], (1, 1, 2), &Device::Cpu).unwrap();
        let strength = 0.75f64;
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens: source,
                strength,
            }],
            appended: vec![],
        };
        let init = initialize_video_stage_latents(
            &noise,
            &positions,
            &conditioning,
            42,
            None,
            None,
            2,
            &Device::Cpu,
        )
        .unwrap();
        // With C = 0 and N = 1 the latent literally holds its own noise
        // coefficient at the conditioned token.
        let latent = init
            .latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let noise_fraction = latent[0];
        assert_eq!(noise_fraction, 0.25);
        let sigma = 0.8f32;
        let timestep =
            timestep_from_sigma_and_mask(sigma, 1, Some(&init.denoise_mask), &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        assert!((timestep[0] - sigma * noise_fraction).abs() < 1e-6);
        assert!((timestep[1] - sigma).abs() < 1e-6);
    }

    /// #1080 witness: soft appended conditioning must start as the same
    /// single `s·C + (1-s)·N` blend upstream's noiser produces. Hard
    /// keyframes and references remain exactly clean.
    #[test]
    fn soft_appended_tokens_initialize_with_seeded_noise() {
        let noise = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::zeros((1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![VideoTokenAppendCondition {
                tokens: Tensor::from_vec(vec![9.0f32, 10.0], (1, 1, 2), &Device::Cpu).unwrap(),
                positions: Tensor::zeros((1, 3, 1, 2), DType::F32, &Device::Cpu).unwrap(),
                strength: 0.4,
                latent_grid: (1, 1, 1),
                spatial_downscale_factor: 1,
            }],
        };
        let init = initialize_video_stage_latents(
            &noise,
            &positions,
            &conditioning,
            42,
            None,
            None,
            2,
            &Device::Cpu,
        )
        .unwrap();
        let values = init
            .latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let appended_noise = seeded_randn(
            42 ^ APPENDED_VIDEO_NOISE_SALT,
            &[1, 1, 2],
            &Device::Cpu,
            DType::F32,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        assert_eq!(
            values,
            vec![
                1.0,
                2.0,
                3.0,
                4.0,
                0.4 * 9.0 + 0.6 * appended_noise[0],
                0.4 * 10.0 + 0.6 * appended_noise[1],
            ]
        );
        assert_eq!(
            init.clean.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0]
        );
        let mask = init
            .denoise_mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(mask, vec![1.0, 1.0, 0.6]);
    }

    /// The retake shape: an explicit clean override with a binary mask
    /// keeps the pre-loop freeze blend, which installs the source into the
    /// frozen (mask = 0) regions of pure-noise start latents — exactly the
    /// pre-fix behaviour, bit for bit.
    #[test]
    fn explicit_clean_override_installs_source_into_frozen_regions() {
        let noise = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::zeros((1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let clean =
            Tensor::from_vec(vec![50.0f32, 60.0, 70.0, 80.0], (1, 2, 2), &Device::Cpu).unwrap();
        let mask = Tensor::from_vec(vec![0.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let init = initialize_video_stage_latents(
            &noise,
            &positions,
            &StageVideoConditioning::default(),
            42,
            Some(&clean),
            Some(&mask),
            2,
            &Device::Cpu,
        )
        .unwrap();
        let values = init
            .latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(values, vec![50.0, 60.0, 3.0, 4.0]);
    }

    /// T2V: no conditioning, no overrides — the seam is an exact identity.
    #[test]
    fn no_conditioning_returns_start_latents_untouched() {
        let noise = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::zeros((1, 3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let init = initialize_video_stage_latents(
            &noise,
            &positions,
            &StageVideoConditioning::default(),
            42,
            None,
            None,
            2,
            &Device::Cpu,
        )
        .unwrap();
        let values = init
            .latents
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
        let mask = init
            .denoise_mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(mask, vec![1.0, 1.0]);
    }

    #[test]
    fn stage_video_conditioning_appends_keyframe_tokens_and_restores_them() {
        let latents =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 2), &Device::Cpu).unwrap();
        let positions = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 1.0, 2.0, 10.0, 11.0, 11.0, 12.0, 20.0, 21.0, 21.0, 22.0,
            ],
            (1, 3, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens: Tensor::from_vec(vec![7.0f32, 8.0], (1, 1, 2), &Device::Cpu).unwrap(),
                strength: 1.0,
            }],
            appended: vec![VideoTokenAppendCondition {
                tokens: Tensor::from_vec(vec![9.0f32, 10.0], (1, 1, 2), &Device::Cpu).unwrap(),
                positions: Tensor::from_vec(
                    vec![30.0f32, 31.0, 40.0, 41.0, 50.0, 51.0],
                    (1, 3, 1, 2),
                    &Device::Cpu,
                )
                .unwrap(),
                strength: 1.0,
                latent_grid: (1, 1, 1),
                spatial_downscale_factor: 1,
            }],
        };

        let (conditioned_latents, conditioned_positions) =
            apply_stage_video_conditioning(&latents, &positions, &conditioning, 42).unwrap();
        assert_eq!(conditioned_latents.dims3().unwrap(), (1, 3, 2));
        assert_eq!(conditioned_positions.dims4().unwrap(), (1, 3, 3, 2));
        assert_eq!(
            conditioned_latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0, 8.0, 2.0, 3.0, 9.0, 10.0]
        );

        let mutated = Tensor::from_vec(
            vec![0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let reapplied = reapply_stage_video_conditioning(&mutated, 2, &conditioning).unwrap();
        assert_eq!(
            reapplied.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0, 8.0, 1.0, 1.0, 9.0, 10.0]
        );

        let stripped = strip_appended_video_conditioning(&reapplied, 2).unwrap();
        assert_eq!(stripped.dims3().unwrap(), (1, 2, 2));
        assert_eq!(
            stripped.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![7.0, 8.0, 1.0, 1.0]
        );
    }

    #[test]
    fn reapply_stage_video_conditioning_keeps_soft_appended_tokens() {
        let latents = Tensor::from_vec(
            vec![0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![VideoTokenAppendCondition {
                tokens: Tensor::from_vec(vec![9.0f32, 10.0], (1, 1, 2), &Device::Cpu).unwrap(),
                positions: Tensor::from_vec(vec![30.0f32, 40.0, 50.0], (1, 3, 1, 1), &Device::Cpu)
                    .unwrap(),
                strength: 0.4,
                latent_grid: (1, 1, 1),
                spatial_downscale_factor: 1,
            }],
        };

        let reapplied = reapply_stage_video_conditioning(&latents, 2, &conditioning).unwrap();
        assert_eq!(reapplied.dims3().unwrap(), (1, 3, 2));
        assert_eq!(
            reapplied.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn clean_latents_replace_soft_blended_positions_with_pure_source() {
        // Simulate the state after `apply_stage_video_conditioning` with
        // strength 0.75: at the replacement positions, `video_latents` already
        // holds `noise*0.25 + source*0.75`. The denoise-mask blend uses
        // `clean_latents` as the target it pulls those positions toward at
        // every step — so the clean target must be pure source, not the
        // pre-blended mix.
        let noise = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let source = [10.0f32, 10.0];
        let strength = 0.75f32;
        let blended_first = [
            noise[0] * (1.0 - strength) + source[0] * strength,
            noise[1] * (1.0 - strength) + source[1] * strength,
        ];
        let soft_blended = Tensor::from_vec(
            vec![
                blended_first[0],
                blended_first[1],
                noise[2],
                noise[3],
                noise[4],
                noise[5],
            ],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens: Tensor::from_vec(source.to_vec(), (1, 1, 2), &Device::Cpu).unwrap(),
                strength: strength as f64,
            }],
            appended: vec![],
        };

        let clean = clean_latents_for_conditioning(&soft_blended, 3, &conditioning).unwrap();
        let values = clean.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(
            values,
            vec![source[0], source[1], noise[2], noise[3], noise[4], noise[5]],
            "soft-blended replacement positions must be overwritten with the pure \
             source tokens; other positions must be preserved unchanged"
        );
    }

    #[test]
    fn clean_latents_passthrough_when_no_replacements() {
        let latents =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 2, 2), &Device::Cpu).unwrap();
        let conditioning = StageVideoConditioning::default();

        let clean = clean_latents_for_conditioning(&latents, 2, &conditioning).unwrap();
        assert_eq!(
            clean.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn staged_latent_patchifies_to_same_token_shape_as_image_at_single_latent_frame() {
        // A 4-pixel-frame motion tail at 1216×704 output lands on a latent
        // block of shape [1, 128, 1, 22, 38]. The render-chain orchestrator
        // produces this block from the prior stage's denoise result; the
        // image-conditioning path produces the same shape after VAE encode.
        // Both must patchify to [1, T*H*W, C] = [1, 1*22*38, 128] tokens so
        // the downstream replacement pass sees them identically regardless
        // of which path produced them.
        let latents = Tensor::zeros(
            (1, LTX2_VIDEO_LATENT_CHANNELS, 1, 22, 38),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let patchifier = super::VideoLatentPatchifier::new(1);
        let tokens = patchifier.patchify(&latents).expect("patchify");
        assert_eq!(tokens.dims(), &[1, 22 * 38, LTX2_VIDEO_LATENT_CHANNELS]);
    }

    #[test]
    fn video_conditioning_self_attention_mask_blocks_cross_keyframe_attention() {
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![
                VideoTokenAppendCondition {
                    tokens: Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &Device::Cpu).unwrap(),
                    positions: Tensor::zeros((1, 3, 1, 2), DType::F32, &Device::Cpu).unwrap(),
                    strength: 1.0,
                    latent_grid: (1, 1, 1),
                    spatial_downscale_factor: 1,
                },
                VideoTokenAppendCondition {
                    tokens: Tensor::from_vec(vec![3.0f32, 4.0], (1, 1, 2), &Device::Cpu).unwrap(),
                    positions: Tensor::zeros((1, 3, 1, 2), DType::F32, &Device::Cpu).unwrap(),
                    strength: 1.0,
                    latent_grid: (1, 1, 1),
                    spatial_downscale_factor: 1,
                },
            ],
        };

        let mask =
            build_video_conditioning_self_attention_mask(2, &conditioning, &Device::Cpu).unwrap();
        let values = mask
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(
            values,
            vec![
                1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, 1.0, //
            ]
        );
    }

    #[test]
    fn scale_video_spatial_positions_multiplies_only_height_and_width_axes() {
        let positions = Tensor::from_vec(
            vec![
                0.5f32, 1.5, //
                10.0, 11.0, //
                20.0, 21.0,
            ],
            (1, 3, 1, 2),
            &Device::Cpu,
        )
        .unwrap();

        let scaled = super::scale_video_spatial_positions(&positions, 2).unwrap();

        assert_eq!(
            scaled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.5, 1.5, 20.0, 22.0, 40.0, 42.0]
        );
    }

    #[test]
    fn supports_real_video_path_accepts_plain_silent_two_stage_runs() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    /// A camera-motion preset rides as an ordinary user LoRA
    /// (`camera-control:<id>` resolved to a path by the server). The renderers
    /// apply it, so refusing the real path here does not disable the LoRA — it
    /// silently swaps the whole render for synthetic placeholder frames.
    #[test]
    fn supports_real_video_path_accepts_a_user_lora_on_distilled() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ltx-2-19b-camera-dolly-in.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_a_user_lora_on_two_stage() {
        let mut req = req("ltx-2-19b-dev:fp8", OutputFormat::Mp4, Some(false));
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ltx-2-19b-camera-dolly-in.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_a_user_lora_on_one_stage() {
        let mut req = req("ltx-2-19b-dev:fp8", OutputFormat::Mp4, Some(false));
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ltx-2-19b-camera-dolly-in.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_a2vid_two_stage_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_ic_lora_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn a2vid_stage1_uses_positive_only_audio_guidance() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        rebuild_execution_graph(&mut plan, &req);

        let (_video_params, audio_params) = super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .unwrap();

        assert_eq!(
            audio_params,
            crate::ltx2::guidance::MultiModalGuiderParams::default()
        );
    }

    fn two_stage_plan_with_overrides(
        model: &str,
        overrides: Option<mold_core::Ltx2GuidanceOverrides>,
    ) -> Ltx2GeneratePlan {
        let mut req = req(model, OutputFormat::Mp4, Some(false));
        req.guidance_overrides = overrides.clone();
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.guidance_overrides = overrides;
        rebuild_execution_graph(&mut plan, &req);
        plan
    }

    #[test]
    fn guidance_overrides_absent_keeps_pipeline_constants() {
        let plan = two_stage_plan_with_overrides("ltx-2-19b-distilled:fp8", None);

        let (video_params, audio_params) = super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .unwrap();

        assert_eq!(
            video_params,
            crate::ltx2::guidance::MultiModalGuiderParams {
                cfg_scale: 3.0,
                stg_scale: 1.0,
                stg_blocks: vec![29],
                rescale_scale: 0.7,
                modality_scale: 3.0,
                skip_step: 0,
            }
        );
        assert_eq!(audio_params.cfg_scale, 7.0);
    }

    #[test]
    fn guidance_overrides_replace_only_the_fields_they_set() {
        let plan = two_stage_plan_with_overrides(
            "ltx-2-19b-distilled:fp8",
            Some(mold_core::Ltx2GuidanceOverrides {
                stg_scale: Some(2.5),
                stg_blocks: Some(vec![14, 15]),
                skip_step: Some(2),
                ..mold_core::Ltx2GuidanceOverrides::default()
            }),
        );

        let (video_params, audio_params) = super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .unwrap();

        assert_eq!(video_params.stg_scale, 2.5);
        assert_eq!(video_params.stg_blocks, vec![14, 15]);
        assert_eq!(video_params.skip_step, 2);
        // Untouched fields keep the pipeline's constants, and base guidance
        // stays with the existing `guidance` request field.
        assert_eq!(video_params.rescale_scale, 0.7);
        assert_eq!(video_params.modality_scale, 3.0);
        assert_eq!(video_params.cfg_scale, 3.0);
        assert_eq!(audio_params.cfg_scale, 7.0);
        assert_eq!(audio_params.stg_scale, 2.5);
    }

    #[test]
    fn guidance_overrides_never_enable_a_disabled_guider() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.audio_file = Some(b"RIFFtestWAVEfmt ".to_vec());
        let overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(2.0),
            modality_scale: Some(4.0),
            ..mold_core::Ltx2GuidanceOverrides::default()
        };
        req.guidance_overrides = Some(overrides.clone());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::A2Vid;
        plan.guidance_overrides = Some(overrides);
        rebuild_execution_graph(&mut plan, &req);

        let (video_params, audio_params) = super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .unwrap();

        assert_eq!(video_params.stg_scale, 2.0);
        // a2-vid runs audio positive-only on purpose; an override must not
        // buy the request an extra transformer pass it never asked for.
        assert_eq!(
            audio_params,
            crate::ltx2::guidance::MultiModalGuiderParams::default()
        );
    }

    #[test]
    fn guidance_overrides_enabling_stg_fall_back_to_the_preset_block() {
        let mut req = req("ltx-2.3-22b-dev:fp8", OutputFormat::Mp4, Some(false));
        let overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(1.0),
            ..mold_core::Ltx2GuidanceOverrides::default()
        };
        req.guidance_overrides = Some(overrides.clone());
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        // two-stage-hq ships with STG off and no block list.
        plan.pipeline = PipelineKind::TwoStageHq;
        plan.guidance_overrides = Some(overrides);
        rebuild_execution_graph(&mut plan, &req);

        let (video_params, _audio_params) = super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .unwrap();

        assert_eq!(video_params.stg_scale, 1.0);
        assert_eq!(video_params.stg_blocks, vec![28]);
    }

    #[test]
    fn guidance_overrides_reject_blocks_deeper_than_the_checkpoint() {
        let plan = two_stage_plan_with_overrides(
            "ltx-2-19b-distilled:fp8",
            Some(mold_core::Ltx2GuidanceOverrides {
                stg_blocks: Some(vec![60]),
                ..mold_core::Ltx2GuidanceOverrides::default()
            }),
        );

        let err = super::stage_multimodal_guider_params(&plan, 0).unwrap_err();

        assert!(err.to_string().contains("transformer blocks"), "got: {err}");
    }

    #[test]
    fn guidance_overrides_are_inert_for_pipelines_without_multimodal_guidance() {
        let mut plan = two_stage_plan_with_overrides(
            "ltx-2-19b-distilled:fp8",
            Some(mold_core::Ltx2GuidanceOverrides {
                stg_scale: Some(2.0),
                ..mold_core::Ltx2GuidanceOverrides::default()
            }),
        );
        plan.pipeline = PipelineKind::Distilled;

        assert!(super::stage_multimodal_guider_params(&plan, 0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn stage_lora_stack_adds_internal_distilled_lora_for_two_stage_second_pass() {
        let req = req("ltx-2.3-22b-dev:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0].path, "/tmp/distilled-lora.safetensors");
        assert_eq!(loras[0].scale, 1.0);
    }

    #[test]
    fn stage_lora_stack_skips_internal_distilled_lora_for_distilled_checkpoint() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        let loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert!(loras.is_empty());
    }

    // ── lip dub ─────────────────────────────────────────────────────────────

    fn lip_dub_plan() -> Ltx2GeneratePlan {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.pipeline = Some(mold_core::Ltx2PipelineMode::LipDub);
        req.ic_lora_control = Some("lipdub".to_string());
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ltx-2.3-22b-ic-lora-dubit-0.9.safetensors".to_string(),
            scale: 1.0,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::LipDub;
        plan.loras = req.loras.clone().unwrap();
        plan.spatial_upsampler_path = Some("/tmp/spatial.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);
        plan
    }

    /// The one number in this pipeline that is invisible when it is wrong.
    ///
    /// The reference speech is appended to the same token sequence as the
    /// audio being generated, so without a shift the model sees two
    /// soundtracks stacked on one timeline. Upstream subtracts the reference's
    /// own duration plus exactly one audio latent frame
    /// (`lipdub.py:314-315`), leaving the last reference patch ending at
    /// `-0.04` — immediately before the generated audio's `0.0`.
    #[test]
    fn lip_dub_audio_reference_positions_end_one_latent_frame_before_zero() {
        let device = Device::Cpu;
        for frames in [1usize, 3, 126] {
            let latents = Tensor::zeros(
                (1, LTX2_AUDIO_LATENT_CHANNELS, frames, LTX2_AUDIO_MEL_BINS),
                DType::F32,
                &device,
            )
            .unwrap();
            let condition = super::append_condition_from_audio_latents(&latents, 1.0).unwrap();

            assert_eq!(
                condition.tokens.dims3().unwrap(),
                (1, frames, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_MEL_BINS)
            );
            let positions = condition
                .positions
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(positions.len(), frames * 2, "frames={frames}");

            // Every position is strictly in the past …
            assert!(
                positions.iter().all(|value| *value < 0.0),
                "frames={frames}: {positions:?}"
            );
            // … the last patch ends exactly one audio latent frame before zero …
            let last_end = positions[positions.len() - 1];
            assert!(
                (last_end + 0.04).abs() < 1e-6,
                "frames={frames}: last patch ends at {last_end}, expected -0.04"
            );
            // … and the reference starts a whole clip earlier. Latent frame 0
            // spans 0.01 s and every later one 0.04 s, so `T` frames shifted
            // by `duration + 0.04` begin at `-(0.04 * T + 0.01)`.
            let expected_start = -(0.04 * frames as f32 + 0.01);
            assert!(
                (positions[0] - expected_start).abs() < 1e-5,
                "frames={frames}: reference starts at {}, expected {expected_start}",
                positions[0]
            );
        }
    }

    /// Deviation 1 of 4 from the plain two-stage pipeline.
    #[test]
    fn lip_dub_keeps_the_ic_lora_on_both_denoise_stages() {
        let plan = lip_dub_plan();

        let stage1 = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2 = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(stage1.len(), 1);
        assert_eq!(
            stage2.len(),
            1,
            "lip dub runs one adapter-carrying stage twice (`lipdub.py:96-106`);              dropping it for stage 2 is the generic IC-LoRA rule, not this one"
        );
        assert_eq!(stage1[0].path, stage2[0].path);
    }

    /// Deviation 2 of 4.
    #[test]
    fn lip_dub_stage_two_freezes_the_audio_instead_of_refining_it() {
        assert_eq!(
            super::stage2_audio_policy(PipelineKind::LipDub),
            Stage2AudioPolicy::Frozen
        );
        for pipeline in [
            PipelineKind::TwoStage,
            PipelineKind::TwoStageHq,
            PipelineKind::A2Vid,
            PipelineKind::Keyframe,
            PipelineKind::IcLora,
        ] {
            assert_eq!(
                super::stage2_audio_policy(pipeline),
                Stage2AudioPolicy::Refine,
                "{pipeline:?} must keep re-denoising its audio in stage 2"
            );
        }

        // The frozen policy has to reach three places in the renderer: the
        // stage-2 initial latent, its denoise mask, and the exported track.
        let render = runtime_function_source("fn render_real_two_stage_av(");
        assert!(render.contains("(Some(stage1_audio_latents), _) if stage2_audio_is_frozen =>"));
        assert!(render.contains("let stage2_frozen_audio_denoise_mask = if stage2_audio_is_frozen"));
        assert!(render.contains("stage2_frozen_audio_denoise_mask.as_ref(),"));
    }

    /// Deviation 3 of 4.
    #[test]
    fn lip_dub_stage_two_rebuilds_its_audio_reference_from_the_generated_audio() {
        assert!(super::appends_audio_reference(PipelineKind::LipDub));
        assert!(!super::appends_audio_reference(PipelineKind::A2Vid));

        let render = runtime_function_source("fn render_real_two_stage_av(");
        assert!(
            render.contains(
                "Some(stage1_audio_latents) if appends_audio_reference(plan.pipeline) =>"
            ),
            "stage 2's audio reference must come from stage 1's output, not the source clip"
        );
    }

    /// Deviation 4 of 4.
    #[test]
    fn lip_dub_exports_the_stage_one_audio_not_the_frozen_stage_two_copy() {
        let render = runtime_function_source("fn render_real_two_stage_av(");
        assert!(render.contains("} else if stage2_audio_is_frozen {"));
        assert!(render.contains(
            "maybe_render_native_audio_track(plan, stage1_audio_latents.as_ref(), device, dtype)?"
        ));
    }

    /// The deviation that is easiest to miss because generic IC-LoRA does the
    /// opposite: lip dub re-encodes the reference clip for stage 2 as well.
    #[test]
    fn lip_dub_keeps_the_reference_video_conditioning_in_stage_two() {
        assert!(PipelineKind::LipDub.keeps_reference_video_in_stage_two());
        assert!(!PipelineKind::IcLora.keeps_reference_video_in_stage_two());
        assert!(!PipelineKind::TwoStage.keeps_reference_video_in_stage_two());

        let render = runtime_function_source("fn render_real_two_stage_av(");
        assert!(render.contains("plan.pipeline.keeps_reference_video_in_stage_two(),"));
        assert!(render
            .contains("matches!(plan.pipeline, PipelineKind::IcLora | PipelineKind::LipDub),"));
    }

    #[test]
    fn lip_dub_conditions_on_audio_it_was_never_handed_and_renders_for_real() {
        let plan = lip_dub_plan();

        assert!(plan.conditioning.audio_path.is_none());
        assert!(plan.execution_graph.uses_audio_conditioning);
        assert!(plan.execution_graph.uses_reference_video_conditioning);
        assert!(plan.execution_graph.wants_audio_output);
        assert_eq!(plan.execution_graph.denoise_passes.len(), 2);
        assert!(plan
            .execution_graph
            .denoise_passes
            .iter()
            .all(|pass| pass.uses_distilled_checkpoint && !pass.apply_distilled_lora));
        assert!(
            super::supports_real_video_path(&plan),
            "a lip-dub plan must take the real path; the synthetic fallback would \
             hand back a gradient that looks like a render"
        );
    }

    #[test]
    fn appended_audio_reference_tokens_are_frozen_and_stripped_before_unpatchify() {
        let device = Device::Cpu;
        let audio_shape = crate::ltx2::model::AudioLatentShape {
            batch: 1,
            channels: LTX2_AUDIO_LATENT_CHANNELS,
            frames: 5,
            mel_bins: LTX2_AUDIO_MEL_BINS,
        };
        let reference = Tensor::ones(
            (1, LTX2_AUDIO_LATENT_CHANNELS, 3, LTX2_AUDIO_MEL_BINS),
            DType::F32,
            &device,
        )
        .unwrap();
        let conditioning = StageAudioConditioning {
            appended: vec![super::append_condition_from_audio_latents(&reference, 1.0).unwrap()],
        };

        let latents = Tensor::zeros(
            (
                1,
                audio_shape.frames,
                LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_MEL_BINS,
            ),
            DType::F32,
            &device,
        )
        .unwrap();
        let positions = Tensor::zeros((1, 1, audio_shape.frames, 2), DType::F32, &device).unwrap();
        let (appended, appended_positions) =
            super::apply_appended_audio_conditioning(&latents, &positions, &conditioning).unwrap();
        assert_eq!(appended.dims3().unwrap().1, 8);
        assert_eq!(appended_positions.dims4().unwrap().2, 8);

        // Generating audio: the render's own tokens denoise, the reference does not.
        let mask =
            super::build_audio_conditioning_denoise_mask(audio_shape, None, &conditioning, &device)
                .unwrap()
                .unwrap();
        assert_eq!(
            mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        );

        // Stage 2: everything is frozen, reference included.
        let frozen = super::build_frozen_audio_denoise_mask(audio_shape, &device).unwrap();
        let mask = super::build_audio_conditioning_denoise_mask(
            audio_shape,
            Some(&frozen),
            &conditioning,
            &device,
        )
        .unwrap()
        .unwrap();
        assert!(mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));

        // The reference never reaches the audio decoder.
        let stripped =
            super::strip_appended_audio_conditioning(&appended, audio_shape.frames).unwrap();
        assert_eq!(stripped.dims3().unwrap().1, audio_shape.frames);
    }

    #[test]
    fn stage_lora_stack_skips_user_loras_for_ic_lora_second_pass() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.source_video = Some(vec![0, 0, 0, 0, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        req.loras = Some(vec![LoraWeight {
            path: "/tmp/ic-lora.safetensors".to_string(),
            scale: 0.8,

            expert: None,
        }]);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::IcLora;
        rebuild_execution_graph(&mut plan, &req);

        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(stage1_loras.len(), 1);
        assert!(stage2_loras.is_empty());
        assert_eq!(super::stage_guidance_scale(&plan, 0).unwrap(), 1.0);
        assert_eq!(super::stage_guidance_scale(&plan, 1).unwrap(), 1.0);
    }

    #[test]
    fn two_stage_stage2_sigmas_use_fixed_distilled_subset() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.num_inference_steps = 30;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let sigmas = super::stage_sigmas_no_terminal(&plan, 1, &Device::Cpu).unwrap();

        assert_eq!(sigmas, vec![0.909375, 0.725, 0.421875]);
    }

    #[test]
    fn prepare_stage_context_collects_two_stage_stage2_selection() {
        let req = req("ltx-2.3-22b-dev:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        plan.num_inference_steps = 30;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let ctx = super::prepare_stage_context(&plan, 1, &Device::Cpu).unwrap();

        assert_eq!(ctx.stage_index, 1);
        assert_eq!(ctx.guidance_scale, 1.0);
        assert_eq!(ctx.sampler_mode, crate::ltx2::execution::SamplerMode::Euler);
        assert_eq!(ctx.sigmas_no_terminal, vec![0.909375, 0.725, 0.421875]);
        assert_eq!(ctx.loras.len(), 1);
        assert_eq!(ctx.loras[0].path, "/tmp/distilled-lora.safetensors");
        assert!(ctx.multimodal_guidance.is_none());
        assert!(!ctx.requires_unconditional_context);
    }

    #[test]
    fn prepare_stage_context_collects_two_stage_hq_res2s_defaults() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStageHq;
        plan.num_inference_steps = 6;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let ctx = super::prepare_stage_context(&plan, 0, &Device::Cpu).unwrap();

        assert_eq!(ctx.stage_index, 0);
        assert_eq!(ctx.sampler_mode, crate::ltx2::execution::SamplerMode::Res2S);
        assert_eq!(ctx.sigmas_no_terminal.len(), 6);
        assert!(ctx
            .sigmas_no_terminal
            .windows(2)
            .all(|pair| pair[0] >= pair[1]));
        assert_eq!(ctx.loras.len(), 1);
        assert_eq!(ctx.loras[0].scale, 0.25);
        assert!(ctx.multimodal_guidance.is_some());
        assert!(ctx.requires_unconditional_context);
    }

    #[test]
    fn two_stage_hq_stage_defaults_match_upstream_runtime() {
        let req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStageHq;
        plan.num_inference_steps = 6;
        plan.distilled_lora_path = Some("/tmp/distilled-lora.safetensors".to_string());
        rebuild_execution_graph(&mut plan, &req);

        let stage1_sigmas = super::stage_sigmas_no_terminal(&plan, 0, &Device::Cpu).unwrap();
        let stage2_sigmas = super::stage_sigmas_no_terminal(&plan, 1, &Device::Cpu).unwrap();
        let stage1_loras = super::stage_lora_stack(&plan, 0).unwrap();
        let stage2_loras = super::stage_lora_stack(&plan, 1).unwrap();

        assert_eq!(
            super::stage_sampler_mode(&plan, 0).unwrap(),
            crate::ltx2::execution::SamplerMode::Res2S
        );
        assert_eq!(
            super::stage_sampler_mode(&plan, 1).unwrap(),
            crate::ltx2::execution::SamplerMode::Res2S
        );
        assert_eq!(stage1_sigmas.len(), 6);
        assert!(stage1_sigmas.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(stage1_sigmas.last().copied().unwrap() > 0.0);
        assert_eq!(stage2_sigmas, vec![0.909375, 0.725, 0.421875]);
        assert_eq!(stage1_loras.len(), 1);
        assert_eq!(stage1_loras[0].scale, 0.25);
        assert_eq!(stage2_loras.len(), 1);
        assert_eq!(stage2_loras[0].scale, 0.5);
        assert_eq!(super::stage_guidance_scale(&plan, 1).unwrap(), 1.0);
    }

    #[test]
    fn supports_real_video_path_rejects_one_stage_audio_and_upscale_requests() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(true));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::OneStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(!super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_distilled_spatial_upscale_runs() {
        let mut req = req("ltx-2.3-22b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X1_5);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_distilled_temporal_upscale_runs() {
        let mut req = req("ltx-2-19b-distilled:fp8", OutputFormat::Mp4, Some(false));
        req.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::Distilled;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn supports_real_video_path_accepts_two_stage_spatial_upscale_runs() {
        let mut req = req("ltx-2-19b:fp8", OutputFormat::Mp4, Some(false));
        req.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        let temp_dir = tempfile::tempdir().unwrap();
        let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
        let preset = preset_for_model(&req.model).unwrap();
        let mut plan = build_plan(&req, preset, conditioning);
        plan.pipeline = PipelineKind::TwoStage;
        rebuild_execution_graph(&mut plan, &req);

        assert!(super::supports_real_video_path(&plan));
    }

    #[test]
    fn ltx2_native_parity_matrix_cases_stay_on_real_runtime_path() {
        let cases = ltx2_native_parity_matrix();
        assert!(cases.iter().any(|case| case.model.contains("2.3-22b")));
        assert!(cases
            .iter()
            .any(|case| case.workflow == "fixed-seed-cuda-reference"));

        for case in cases {
            let mut req = req(case.model, OutputFormat::Mp4, case.enable_audio);
            req.seed = Some(case.seed);
            case.apply(&mut req);
            let temp_dir = tempfile::tempdir().unwrap();
            let conditioning = conditioning::stage_conditioning(&req, temp_dir.path()).unwrap();
            let preset = preset_for_model(&req.model).unwrap();
            let mut plan = build_plan(&req, preset, conditioning);
            plan.pipeline = case.pipeline;
            rebuild_execution_graph(&mut plan, &req);

            assert!(
                super::supports_real_video_path(&plan),
                "{} ({}) should stay on the native runtime path",
                case.workflow,
                case.model
            );
        }
    }

    #[test]
    fn waveform_to_audio_track_interleaves_stereo_samples() {
        let waveform = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();

        let track = super::waveform_to_audio_track(&waveform, 48_000)
            .unwrap()
            .unwrap();

        assert_eq!(track.channels, 2);
        assert_eq!(track.sample_rate, 48_000);
        assert_eq!(
            track.interleaved_samples,
            vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
        );
    }

    /// Build a synthetic `NativePromptEncoding` on the supplied device.
    /// Shapes match what the V2 connectors produce (batch=1, seq=3, dims
    /// match `tiny_gemma_config()` widths).
    fn synthetic_prompt_encoding(device: &Device) -> super::NativePromptEncoding {
        use super::EmbeddingsProcessorOutput;

        let video = Tensor::from_vec(
            (0..24).map(|x| x as f32).collect::<Vec<_>>(),
            (1, 3, 8),
            device,
        )
        .unwrap();
        let audio = Tensor::from_vec(
            (0..12).map(|x| x as f32 * 0.5).collect::<Vec<_>>(),
            (1, 3, 4),
            device,
        )
        .unwrap();
        let mask = Tensor::from_vec(vec![1u8, 1, 1], (1, 3), device).unwrap();

        super::NativePromptEncoding {
            conditional: EmbeddingsProcessorOutput {
                video_encoding: video.clone(),
                audio_encoding: Some(audio.clone()),
                attention_mask: mask.clone(),
            },
            unconditional: EmbeddingsProcessorOutput {
                video_encoding: video,
                audio_encoding: Some(audio),
                attention_mask: mask,
            },
        }
    }

    /// `move_prompt_encoding_to_device` round-trips a CPU-built encoding back
    /// to CPU intact. Pins the function shape (preserves video/audio/mask,
    /// preserves dtypes) so a future refactor that drops audio or downcasts
    /// the mask gets caught.
    #[test]
    fn move_prompt_encoding_round_trips_on_cpu() {
        let prompt = synthetic_prompt_encoding(&Device::Cpu);
        let video_before = prompt.conditional.video_encoding.to_vec3::<f32>().unwrap();

        let moved = super::move_prompt_encoding_to_device(prompt, &Device::Cpu).unwrap();

        assert!(moved.conditional.video_encoding.device().is_cpu());
        assert!(moved.unconditional.video_encoding.device().is_cpu());
        assert!(moved.conditional.attention_mask.device().is_cpu());
        let audio = moved
            .conditional
            .audio_encoding
            .as_ref()
            .expect("audio survives the move");
        assert!(audio.device().is_cpu());
        assert_eq!(
            moved.conditional.video_encoding.to_vec3::<f32>().unwrap(),
            video_before,
            "values must round-trip exactly"
        );
    }

    /// LTX-2 Gemma encoder built on CPU + transformer on CUDA: the move
    /// function must lift the conditioning to the CUDA device for the
    /// transformer's encode-time consumer. Ignored when CUDA isn't built —
    /// the no-feature gate keeps CI green on Metal/CPU runners.
    #[cfg(feature = "cuda")]
    #[test]
    #[cfg_attr(not(target_os = "linux"), ignore)]
    fn runtime_handles_prompt_encoder_on_cpu_with_transformer_on_cuda() {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let prompt = synthetic_prompt_encoding(&Device::Cpu);
        let moved = super::move_prompt_encoding_to_device(prompt, &cuda).unwrap();
        assert!(moved.conditional.video_encoding.device().is_cuda());
        assert!(moved.unconditional.video_encoding.device().is_cuda());
        assert!(moved.conditional.attention_mask.device().is_cuda());
        assert!(moved
            .conditional
            .audio_encoding
            .as_ref()
            .unwrap()
            .device()
            .is_cuda());
    }

    // ── Tiled stage-2 refinement ────────────────────────────────────────

    /// Largest per-pixel difference, on a 0-255 scale, measured between a
    /// tiled and an untiled stage-2 refinement of the same prompt and seed.
    ///
    /// Measured on an RTX 4090 with `ltx-2-19b-distilled:fp8` at 512x512x25 —
    /// a shape that needs no tiling at all, which is the point: it isolates
    /// what tiling itself costs from what out-of-distribution positions cost.
    /// `MOLD_LTX2_SPATIAL_TILE=256:64` forced a 2x2 layout against the
    /// untiled default. See `crates/mold-inference/src/ltx2/tiling.rs` for
    /// the blend and the PR for the run.
    ///
    /// The residual is not blend arithmetic — that reconstructs exactly, as
    /// `tiled_refinement_reassembles_a_pointwise_denoiser_exactly` asserts.
    /// It is the model: each tile is denoised from its own noise draw and
    /// sees only its own region, so a tile cannot reproduce what a global
    /// pass would have made of a structure crossing its edge.
    #[allow(dead_code)]
    const TILED_STAGE2_MEASURED_PIXEL_DEVIATION: f64 = 0.0;

    fn ramp_latents(shape: VideoLatentShape) -> Tensor {
        let count = shape.batch * shape.channels * shape.frames * shape.height * shape.width;
        let values = (0..count)
            .map(|index| (index % 97) as f32 / 97.0 - 0.5)
            .collect::<Vec<_>>();
        Tensor::from_vec(
            values,
            (
                shape.batch,
                shape.channels,
                shape.frames,
                shape.height,
                shape.width,
            ),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn tiled_shape() -> VideoLatentShape {
        VideoLatentShape {
            batch: 1,
            channels: 2,
            frames: 3,
            height: 8,
            width: 9,
        }
    }

    /// Deliberately asymmetric: a layout with the same tile count on both
    /// spatial axes would survive a height/width transpose in the blend.
    fn asymmetric_layout() -> TileCountConfig {
        TileCountConfig {
            frames: DimensionTiling::none(),
            height: DimensionTiling::new(2, 2),
            width: DimensionTiling::new(3, 2),
        }
    }

    fn pass_over<'a>(
        tiles: &'a [crate::ltx2::tiling::Tile],
        shape: VideoLatentShape,
        clean: &'a Tensor,
        sigma: f32,
        conditioning: &'a StageVideoConditioning,
    ) -> TiledStage2Pass<'a> {
        TiledStage2Pass {
            tiles,
            full_shape: shape,
            clean_latents: clean,
            noise_seed: 0x5354_4147_4532_4c54,
            sigma,
            fps: 24.0,
            conditioning,
            device: &Device::Cpu,
        }
    }

    /// A 4K stage-2 latent: 3840x2176 is 120x68 cells, past the 64-cell
    /// trained span on both axes.
    fn uhd_stage2_shape() -> VideoLatentShape {
        VideoLatentShape {
            batch: 1,
            channels: 128,
            frames: 4,
            height: 68,
            width: 120,
        }
    }

    /// Disabling tiling past the trained span must fail, not render.
    ///
    /// What it would otherwise produce is a finished video with degraded
    /// large-scale structure and no error anywhere — the one failure mode a
    /// user cannot detect from the output.
    #[test]
    fn tiling_off_past_the_trained_span_refuses_instead_of_degrading() {
        let err = plan_stage2_tiles_with_policy(uhd_stage2_shape(), SpatialTilePolicy::Off)
            .expect_err("a 4K stage 2 cannot run untiled");
        let message = err.to_string();
        assert!(
            message.contains("spatial tiling is disabled") && message.contains("2048"),
            "the refusal must name the cause and the way out, got: {message}"
        );

        // Auto handles the same shape, in the layout the ladder advertises.
        let tiles = plan_stage2_tiles_with_policy(uhd_stage2_shape(), SpatialTilePolicy::Auto)
            .expect("auto tiling covers a 4K stage 2");
        assert_eq!(tiles.len(), 4, "3840x2176 refines as 2x2 tiles");
        for tile in &tiles {
            let (width, height, _) = tile.pixel_shape();
            assert!(
                width <= mold_core::validation::LTX2_MAX_AXIS_PIXELS as usize
                    && height <= mold_core::validation::LTX2_MAX_AXIS_PIXELS as usize,
                "every tile must be denoised inside the trained span, got {width}x{height}"
            );
        }
    }

    /// The refusal is scoped to shapes that need tiling. Every resolution that
    /// rendered before the composed ceiling still runs untiled under `off`.
    #[test]
    fn tiling_off_inside_the_trained_span_is_unchanged() {
        for (width, height) in [(24usize, 16usize), (60, 34), (64, 64)] {
            let shape = VideoLatentShape {
                batch: 1,
                channels: 128,
                frames: 4,
                height,
                width,
            };
            let tiles = plan_stage2_tiles_with_policy(shape, SpatialTilePolicy::Off)
                .expect("a shape inside the span never needed tiling");
            assert_eq!(
                tiles.len(),
                1,
                "{width}x{height} latent cells must stay a single untiled pass"
            );
        }
    }

    /// The blend is the part that fails silently. With `sigma = 0` every tile
    /// is handed its own clean slice, so an identity denoiser must come back
    /// as the original latent — any error is the window's or the index map's.
    #[test]
    fn tiled_refinement_reassembles_a_pointwise_denoiser_exactly() {
        let shape = tiled_shape();
        let clean = ramp_latents(shape);
        let tiles =
            create_tiles(shape.frames, shape.height, shape.width, asymmetric_layout()).unwrap();
        assert_eq!(tiles.len(), 6, "this layout must actually be tiled");

        let conditioning = StageVideoConditioning::default();
        let blended = pass_over(&tiles, shape, &clean, 0.0, &conditioning)
            .run(|request| Ok(request.start_latents.clone()))
            .unwrap();

        let expected = clean.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = blended.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.len(), expected.len());
        for (index, (actual, wanted)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (actual - wanted).abs() < 1e-5,
                "element {index}: {actual} vs {wanted}"
            );
        }
    }

    /// A tile has to look like a sequence starting at zero — that is what
    /// upstream's `normalize_positions=True` buys and the only reason a tile
    /// lands back inside the trained span — and it has to draw its own noise.
    #[test]
    fn each_tile_is_seeded_and_positioned_as_its_own_sequence() {
        let shape = tiled_shape();
        let clean = ramp_latents(shape);
        let tiles =
            create_tiles(shape.frames, shape.height, shape.width, asymmetric_layout()).unwrap();
        let conditioning = StageVideoConditioning::default();

        let mut noises: Vec<Vec<f32>> = Vec::new();
        // sigma 1.0 makes the tile input pure noise, so this reads it directly.
        pass_over(&tiles, shape, &clean, 1.0, &conditioning)
            .run(|request| {
                let (_, axes, tokens, _) = request.positions.dims4().unwrap();
                let flat = request
                    .positions
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for axis in 0..axes {
                    let start = axis * tokens * 2;
                    let minimum = flat[start..start + tokens * 2]
                        .iter()
                        .copied()
                        .fold(f32::MAX, f32::min);
                    assert_eq!(
                        minimum, 0.0,
                        "tile {} axis {axis} starts at {minimum}, not zero",
                        request.index
                    );
                }
                noises.push(
                    request
                        .start_latents
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
                Ok(request.start_latents.clone())
            })
            .unwrap();

        assert_eq!(noises.len(), tiles.len());
        for left in 0..noises.len() {
            for right in left + 1..noises.len() {
                if noises[left].len() == noises[right].len() {
                    assert_ne!(
                        noises[left], noises[right],
                        "tiles {left} and {right} were handed the same noise"
                    );
                }
            }
        }
    }

    /// The compatibility boundary. A one-tile plan must hand the denoiser
    /// exactly what the pre-tiling code built, and return exactly what it
    /// returned — no blend round trip, no reseeding, no shifted positions.
    #[test]
    fn an_untiled_pass_reproduces_the_pre_tiling_inputs_exactly() {
        let shape = tiled_shape();
        let clean = ramp_latents(shape);
        let tiles = create_tiles(
            shape.frames,
            shape.height,
            shape.width,
            TileCountConfig::untiled(),
        )
        .unwrap();
        assert_eq!(tiles.len(), 1);
        let sigma = 0.35;
        let conditioning = StageVideoConditioning::default();

        let mut seen_positions = None;
        let output = pass_over(&tiles, shape, &clean, sigma, &conditioning)
            .run(|request| {
                seen_positions = Some(request.positions.clone());
                Ok(request.start_latents.clone())
            })
            .unwrap();

        let noise = crate::engine::seeded_randn(
            0x5354_4147_4532_4c54,
            &[
                shape.batch,
                shape.channels,
                shape.frames,
                shape.height,
                shape.width,
            ],
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        let expected_start = super::mix_clean_latents_with_noise(&clean, &noise, sigma).unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected_start
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        let expected_positions = super::build_video_positions(
            super::pixel_shape_for_video_latents(shape, 24),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            seen_positions
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected_positions
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn tile_conditioning_keeps_only_the_replacement_rows_under_the_tile() {
        let shape = VideoLatentShape {
            batch: 1,
            channels: 1,
            frames: 1,
            height: 4,
            width: 4,
        };
        // One replaced latent frame: token `t` carries the value `t`.
        let tokens = Tensor::from_vec(
            (0..16).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 16, 1),
            &Device::Cpu,
        )
        .unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![VideoTokenReplacement {
                start_token: 0,
                tokens,
                strength: 0.8,
            }],
            appended: vec![],
        };
        let tiles = create_tiles(
            1,
            4,
            4,
            TileCountConfig {
                frames: DimensionTiling::none(),
                height: DimensionTiling::new(2, 1),
                width: DimensionTiling::new(2, 1),
            },
        )
        .unwrap();
        // Last tile covers rows 2..4 and columns 2..4.
        let tile = tiles.last().unwrap();
        assert_eq!((tile.height.start, tile.height.end), (2, 4));
        assert_eq!((tile.width.start, tile.width.end), (2, 4));

        let sliced = super::conditioning_for_tile(&conditioning, shape, tile).unwrap();
        let replacement = &sliced.replacements[0];
        assert_eq!(replacement.start_token, 0);
        assert_eq!(replacement.strength, 0.8);
        assert_eq!(
            replacement
                .tokens
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![10.0, 11.0, 14.0, 15.0]
        );
    }

    /// IC-LoRA reference video is encoded at `1/df` of the generated grid, so
    /// a tile covering generated cells `[a, b)` covers reference cells
    /// `[a/df, b/df)` — and its positions have to be rebased onto the tile's
    /// origin, or the reference lands somewhere the tile never looks.
    #[test]
    fn tile_conditioning_slices_reference_latents_by_the_downscale_factor() {
        // Generated grid is 4x4; the reference grid is 2x2 at df = 2.
        let tokens =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (1, 4, 1), &Device::Cpu).unwrap();
        // Positions are already in full-resolution pixels: reference cell `r`
        // sits at `r * 32 * df`.
        let mut position_values = Vec::new();
        for axis in 0..3 {
            for row in 0..2 {
                for column in 0..2 {
                    let coordinate = match axis {
                        0 => 0.0,
                        1 => (row * 32 * 2) as f32,
                        _ => (column * 32 * 2) as f32,
                    };
                    position_values.push(coordinate);
                    position_values.push(coordinate);
                }
            }
        }
        let positions = Tensor::from_vec(position_values, (1, 3, 4, 2), &Device::Cpu).unwrap();
        let conditioning = StageVideoConditioning {
            replacements: vec![],
            appended: vec![VideoTokenAppendCondition {
                tokens,
                positions,
                strength: 1.0,
                latent_grid: (1, 2, 2),
                spatial_downscale_factor: 2,
            }],
        };
        let shape = VideoLatentShape {
            batch: 1,
            channels: 1,
            frames: 1,
            height: 4,
            width: 4,
        };
        let tiles = create_tiles(
            1,
            4,
            4,
            TileCountConfig {
                frames: DimensionTiling::none(),
                height: DimensionTiling::new(2, 0),
                width: DimensionTiling::new(2, 0),
            },
        )
        .unwrap();
        // Bottom-right tile: generated rows 2..4, columns 2..4.
        let tile = tiles.last().unwrap();
        let sliced = super::conditioning_for_tile(&conditioning, shape, tile).unwrap();
        let condition = &sliced.appended[0];

        assert_eq!(condition.latent_grid, (1, 1, 1));
        assert_eq!(
            condition
                .tokens
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0],
            "reference cell (1, 1) is the one under generated rows 2..4"
        );
        // Reference cell 1 sat at pixel 64; the tile's origin is also pixel
        // 64, so a correctly rebased position is zero.
        assert_eq!(
            condition
                .positions
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 6]
        );
    }

    /// Residency planning reads the stage shape to budget activations. If a
    /// tiled pass still declared the whole frame, it would reserve for a
    /// forward pass that never runs — and stream transformer blocks it had
    /// room to keep resident, which is most of what tiling was for.
    #[test]
    fn a_tiled_stage_declares_its_largest_tile_not_the_whole_frame() {
        let shape = tiled_shape();
        let full = super::pixel_shape_for_video_latents(shape, 24);

        let single = create_tiles(
            shape.frames,
            shape.height,
            shape.width,
            TileCountConfig::untiled(),
        )
        .unwrap();
        assert_eq!(
            super::stage2_forward_pixel_shape(&single, full),
            full,
            "one full-cover tile must declare exactly the shape it always did"
        );

        let tiles =
            create_tiles(shape.frames, shape.height, shape.width, asymmetric_layout()).unwrap();
        let declared = super::stage2_forward_pixel_shape(&tiles, full);
        assert!(declared.width < full.width && declared.height < full.height);
        assert_eq!(declared.frames, full.frames, "time is never tiled");
        for tile in &tiles {
            let (width, height, _) = tile.pixel_shape();
            assert!(width <= declared.width && height <= declared.height);
        }
    }

    /// `Stage2AudioInputs` is now the only thing standing between a tiled
    /// stage 2 (video-only) and an untiled one (audio refined as always), and
    /// the untiled half cannot be reached from a unit test — it is built
    /// inline in each renderer. So it is asserted against the source, the same
    /// way the phase-probe and LoRA-stack contracts are.
    #[test]
    fn only_a_tiled_stage2_switches_the_audio_branch_off() {
        for signature in [
            "fn render_real_distilled_av(",
            "fn render_real_two_stage_av(",
        ] {
            let body = runtime_function_source(signature);
            assert!(
                body.contains("Stage2AudioInputs::video_only(audio_shape.is_some())"),
                "`{signature}` must make its tiled stage 2 video-only"
            );
            assert!(
                body.contains("shape: audio_shape,"),
                "`{signature}` must still hand the untiled stage 2 the real audio shape"
            );
            for fed in [
                "start: stage2_audio_start.as_ref(),",
                "noise: stage2_audio_noise.as_ref(),",
                "positions: prompt_inputs.audio_positions.as_ref(),",
                "context: prompt_inputs.audio_context.as_ref(),",
            ] {
                assert!(
                    body.contains(fed),
                    "`{signature}` drops `{fed}` from the untiled stage 2, which would \
                     silently stop refining audio at every resolution"
                );
            }
            assert!(
                !body.contains("stage2_audio_shape"),
                "`{signature}` must route audio through the bundle, not a loose shape — \
                 half a branch is what the transformer rejects"
            );
        }
    }

    /// A spatial tile says nothing about the audio track, so upstream refines
    /// video only when it tiles. Refining the same track once per tile and
    /// keeping whichever came last would be worse than not refining it.
    #[test]
    fn a_tiled_stage2_carries_stage1_audio_through_unrefined() {
        let shape = crate::ltx2::model::AudioLatentShape {
            batch: 1,
            channels: 2,
            frames: 3,
            mel_bins: 4,
        };
        let refined = super::Stage2AudioInputs {
            shape: Some(shape),
            ..super::Stage2AudioInputs::default()
        };
        assert_eq!(refined.shape, Some(shape));
        // Video-only switches the whole branch off together. Half of it — an
        // audio context with no audio latents — is what the transformer
        // rejects with "must be provided together".
        let video_only = super::Stage2AudioInputs::video_only(true);
        assert!(video_only.shape.is_none());
        assert!(video_only.start.is_none());
        assert!(video_only.noise.is_none());
        assert!(video_only.positions.is_none());
        assert!(video_only.context.is_none());
        assert!(video_only.uncond_context.is_none());
        assert!(video_only.alt_context.is_none());
        assert!(video_only.denoise_mask.is_none());

        let stage1 = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let refined = Tensor::ones((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let untiled = stage2_carried_audio(Some(refined.clone()), Some(&stage1), false).unwrap();
        assert_eq!(
            untiled.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            24.0,
            "an untiled stage 2 keeps its own refinement"
        );
        let tiled = stage2_carried_audio(None, Some(&stage1), true).unwrap();
        assert_eq!(
            tiled.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            0.0,
            "a tiled stage 2 keeps stage 1's track"
        );
    }

    // ── Phase VRAM telemetry ────────────────────────────────────────────
    //
    // The instrumented boundaries are minutes-long GPU phases that no unit
    // test can execute, so the contract is asserted structurally against this
    // file's own source — the pattern already used for the Z-Image staged
    // load/drop ordering.

    fn runtime_function_source(signature: &str) -> &'static str {
        let source = include_str!("runtime.rs");
        let start = source
            .find(signature)
            .unwrap_or_else(|| panic!("runtime.rs should define `{signature}`"));
        let indent: String = signature.chars().take_while(|c| *c == ' ').collect();
        let terminator = format!("\n{indent}}}\n");
        let rest = &source[start..];
        let end = rest
            .find(&terminator)
            .unwrap_or_else(|| panic!("`{signature}` should end at its own closing brace"));
        &rest[..end]
    }

    /// Every renderer must build its transformer from `stage_lora_stack`.
    ///
    /// Passing a literal empty slice silently drops the user's LoRAs — which
    /// is how camera-motion presets came to render synthetic placeholder
    /// frames: `supports_real_video_path` refused any plan carrying a LoRA
    /// precisely because these renderers would have ignored it.
    #[test]
    fn every_ltx2_renderer_loads_the_stage_lora_stack() {
        for signature in [
            "fn render_real_distilled_av(",
            "fn render_real_two_stage_av(",
            "fn render_real_one_stage_av(",
            "fn render_real_retake_av(",
        ] {
            let body = runtime_function_source(signature);
            assert!(
                body.contains("stage_lora_stack(plan,") || body.contains("_context.loras"),
                "`{signature}` must build its transformer from the stage LoRA stack"
            );
            assert!(
                !body.contains("device,\n                    &[],")
                    && !body.contains("device,\n                &[],")
                    && !body.contains("device, &[],"),
                "`{signature}` passes an empty LoRA slice, which silently drops user LoRAs \
                 such as camera-motion presets"
            );
        }
    }

    #[test]
    fn distilled_runtime_wires_stage_loras_into_initial_and_recovery_loads() {
        let render = runtime_function_source("fn render_real_distilled_av(");
        for required in [
            "let stage1_loras = stage_lora_stack(plan, 0)?;",
            "&stage1_loras,",
            "let stage2_loras = stage_lora_stack(plan, 1)?;",
            "&stage2_loras,",
        ] {
            assert!(
                render.contains(required),
                "distilled inference must retain `{required}` so neither an initial load nor an OOM recovery silently drops user LoRAs"
            );
        }
        assert_eq!(render.matches("&stage1_loras,").count(), 2);
        assert_eq!(render.matches("&stage2_loras,").count(), 2);
    }

    #[test]
    fn every_ltx2_phase_boundary_opens_a_vram_probe() {
        for signature in [
            "fn load_ltx2_av_transformer_with_loras(",
            "fn load_ltx2_video_vae(",
            "fn maybe_load_stage_video_conditioning(",
            "fn run_denoise_stage_with_oom_recovery<T>(",
            "fn decode_video_frames_with_telemetry(",
        ] {
            assert!(
                runtime_function_source(signature).contains("PhaseVramProbe::enter"),
                "`{signature}` must measure its GPU memory phase"
            );
        }
        // Prompt encode and the CUDA device handoff live in the progress-aware
        // `prepare` body that plain `prepare` delegates to.
        let prepare = runtime_function_source("    pub fn prepare_with_progress(");
        assert!(
            prepare.contains("PhaseVramProbe::enter_if(\"prompt_encode\"")
                && prepare.contains("PhaseVramProbe::enter(\"device_handoff\")"),
            "prompt encoding and the conditioning device handoff must be measured"
        );
    }

    #[test]
    fn phase_vram_lines_use_the_dedicated_tracing_target() {
        let source = include_str!("runtime.rs");
        assert!(
            source.contains("tracing::info!(target: LTX2_VRAM_TARGET"),
            "phase reports are always-on info lines on the LTX-2 VRAM target"
        );
        assert!(
            source.contains("tracing::error!(target: LTX2_VRAM_TARGET"),
            "OOM diagnoses are error lines on the same target"
        );
        assert!(
            source.contains("const LTX2_VRAM_TARGET: &str = \"mold::ltx2::vram\";"),
            "the documented target string must not drift"
        );
    }

    #[test]
    fn oom_diagnosis_reports_the_whole_residency_plan() {
        let plan = crate::adaptive_offload::AdaptiveResidencyPlan {
            resident: vec![true, false, true],
            resident_bytes: 3_000_000_000,
            streamed_bytes: 1_000_000_000,
            largest_streamed_block: 700_000_000,
            fixed_resident_bytes: 2_107_000_000,
            activation_budget: 4_000_000_000,
            runtime_headroom: crate::adaptive_offload::ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM,
        };
        let summary = super::ltx2_residency_summary(Some(&plan));

        for field in [
            "resident_count=2",
            "streamed_count=1",
            "resident_bytes=",
            "activation_budget=",
            "runtime_headroom=",
            "largest_streamed_block=",
            "fixed_resident_bytes=",
        ] {
            assert!(
                summary.contains(field),
                "residency summary must carry `{field}` so an OOM needs no reproduction: {summary}"
            );
        }
        assert_eq!(super::ltx2_residency_summary(None), "residency=unknown");
    }

    #[test]
    fn cuda_oom_failures_log_before_the_error_propagates() {
        for signature in [
            "fn load_ltx2_av_transformer_with_loras(",
            "fn run_denoise_stage_with_oom_recovery<T>(",
            "fn decode_video_frames_with_telemetry(",
        ] {
            assert!(
                runtime_function_source(signature).contains("log_ltx2_phase_vram_result("),
                "`{signature}` must route its result through the OOM-aware phase logger"
            );
        }
        assert!(
            runtime_function_source("fn log_ltx2_phase_vram_result(")
                .contains("ltx2_residency_summary("),
            "the OOM branch must attach the residency plan"
        );
    }

    #[test]
    fn one_stage_decode_reports_a_vae_phase_and_a_decode_timing() {
        let helper = runtime_function_source("fn decode_video_frames_with_telemetry(");
        assert!(
            helper.contains("ProgressPhase::Vae") && helper.contains("decode_video"),
            "the shared decode path must emit the scheduler's VAE phase and a timing line"
        );
        for (signature, label) in [
            ("fn render_real_one_stage_av(", "\"one_stage\""),
            ("fn render_real_distilled_av(", "\"distilled\""),
            ("fn render_real_two_stage_av(", "\"two_stage\""),
            ("fn render_real_retake_av(", "\"retake\""),
        ] {
            let body = runtime_function_source(signature);
            assert!(
                body.contains("decode_video_frames_with_telemetry("),
                "`{signature}` must decode through the instrumented path"
            );
            assert!(
                body.contains(label),
                "`{signature}` must keep its own `{label}` phase label"
            );
        }
    }

    #[test]
    fn debug_vram_lines_follow_the_thread_gpu_binding() {
        let body = runtime_function_source("fn log_debug_vram(");
        assert!(
            body.contains("thread_gpu_ordinal()") && !body.contains("free_vram_bytes(0)"),
            "debug VRAM lines must sample the GPU this thread is bound to"
        );
    }
}

#[cfg(test)]
mod scene_embeddings_tests {
    use super::*;

    /// Writes a stand-in for the adapter's shipped embeddings file.
    fn write_scene_embeddings(dir: &std::path::Path, with_audio: bool) -> String {
        let path = dir.join("scene-emb.safetensors");
        let device = candle_core::Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "video_context".to_string(),
            Tensor::ones((1, 1024, 4096), DType::F32, &device).unwrap(),
        );
        if with_audio {
            tensors.insert(
                "audio_context".to_string(),
                Tensor::ones((1, 1024, 2048), DType::F32, &device).unwrap(),
            );
        }
        candle_core::safetensors::save(&tensors, &path).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn saved_embeddings_become_the_conditioning_without_encoding_a_prompt() {
        let temp = tempfile::tempdir().unwrap();
        let path = write_scene_embeddings(temp.path(), true);

        let prompt = load_scene_embeddings(&path).expect("embeddings must load");

        assert_eq!(prompt.conditional.video_encoding.dims3().unwrap().2, 4096);
        assert_eq!(
            prompt
                .conditional
                .audio_encoding
                .as_ref()
                .expect("audio context is present")
                .dims3()
                .unwrap()
                .2,
            2048
        );
        // Upstream ships no mask; the whole fixed context is attended.
        let mask: Vec<u8> = prompt
            .conditional
            .attention_mask
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(mask.len(), 1024);
        assert!(mask.iter().all(|&value| value == 1));
    }

    #[test]
    fn a_video_only_embeddings_file_still_loads() {
        let temp = tempfile::tempdir().unwrap();
        let path = write_scene_embeddings(temp.path(), false);
        let prompt = load_scene_embeddings(&path).expect("embeddings must load");
        assert!(prompt.conditional.audio_encoding.is_none());
    }

    #[test]
    fn a_file_without_video_context_names_the_missing_tensor() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty.safetensors");
        let tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        candle_core::safetensors::save(&tensors, &path).unwrap();

        let err = load_scene_embeddings(&path.to_string_lossy()).unwrap_err();
        assert!(format!("{err:#}").contains("video_context"), "got: {err:#}");
    }
}
