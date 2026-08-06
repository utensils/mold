//! Wan 2.1 / 2.2 text-to-video engine.
//!
//! Ties the four Wan layers together: UMT5-XXL prompt encoding, the DiT, the
//! FlowUniPC sampler, and the causal 3-D VAE. Text-to-video only — image and
//! video conditioning (I2V, TI2V) arrive in a later layer and are rejected
//! here rather than silently ignored.
//!
//! Like the other video families this engine is *sequential by construction*:
//! every component is loaded inside `generate` and dropped as soon as it is
//! done. That is not a simplification, it is a VRAM requirement — UMT5-XXL is
//! 11.4 GB at fp16 and TI2V-5B's transformer is another 10 GB, so the encoder
//! has to be gone before the first denoise step. `load()` is therefore a
//! no-op, matching `LtxVideoEngine`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, IndexOp, Tensor};
use mold_core::{GenerateRequest, GenerateResponse, ModelPaths, OutputFormat, VideoData};

use crate::engine::{gpu_dtype, rand_seed, seeded_randn, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::ltx_video::video_enc;
use crate::progress::{ProgressCallback, ProgressEvent, ProgressPhase};
use crate::shared_pool::SharedPool;
use crate::wan::model::transformer::{WanTransformer, WanTransformerConfig};
use crate::wan::model::vae::{WanVaeConfig, WanVideoVae};
use crate::wan::sampler::{apply_cfg, FlowUniPc, WanSchedule, WanScheduleConfig};
use crate::wan::text::umt5::WanTextEncoder;

/// ComfyUI ships flow shift 8.0 in both its Wan 2.1 and Wan 2.2 templates.
/// Upstream's own CLI defaults differ per task (5.0 for 1.3B, 5.0 for TI2V),
/// but the ComfyUI recipe is the one the community's prompts are tuned
/// against, and it is what the manifest defaults already mirror.
const DEFAULT_FLOW_SHIFT: f64 = 8.0;

/// Override for [`DEFAULT_FLOW_SHIFT`]. Validated, not silently ignored.
const FLOW_SHIFT_ENV: &str = "MOLD_WAN_SHIFT";

/// Temporal compression is 4x for both VAE generations.
const VAE_TEMPORAL_COMPRESSION: usize = 4;

/// Which VAE generation a checkpoint pairs with. This decides latent channel
/// count, spatial compression, and the frame/fps defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanVaeGeneration {
    /// 16-channel, 8x8 spatial — Wan 2.1 (1.3B, 14B, A14B).
    V2_1,
    /// 48-channel, 16x16 spatial — Wan 2.2 TI2V-5B only.
    V2_2,
}

impl WanVaeGeneration {
    fn config(self) -> WanVaeConfig {
        match self {
            Self::V2_1 => WanVaeConfig::v2_1(),
            Self::V2_2 => WanVaeConfig::v2_2(),
        }
    }

    /// `(frames, fps)` when the request carries neither. These mirror the
    /// manifest defaults, which the CLI and server normally plumb through.
    fn default_timing(self) -> (u32, u32) {
        match self {
            Self::V2_1 => (81, 16),
            Self::V2_2 => (121, 24),
        }
    }
}

/// Read a safetensors header without materializing any weights.
fn header_shapes(path: &Path) -> Result<Vec<(String, Vec<usize>)>> {
    let st = unsafe { MmapedSafetensors::new(path) }
        .with_context(|| format!("open Wan checkpoint at {}", path.display()))?;
    Ok(st
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}

/// Detect the VAE generation from the checkpoint's own key layout.
///
/// Wan 2.2 nests its stages (`decoder.upsamples.{s}.upsamples.{j}`) while 2.1
/// keeps one flat `nn.Sequential`. ComfyUI uses exactly this probe
/// (`comfy/sd.py:770-775`); it beats matching on the file name, which varies
/// between repacks, or on the model name, which a `--vae` override can
/// contradict.
pub(crate) fn detect_vae_generation(path: &Path) -> Result<WanVaeGeneration> {
    let shapes = header_shapes(path)?;
    let nested = shapes
        .iter()
        .any(|(name, _)| name.starts_with("decoder.upsamples.0.upsamples."));
    if nested {
        return Ok(WanVaeGeneration::V2_2);
    }
    if shapes
        .iter()
        .any(|(name, _)| name == "decoder.middle.0.residual.0.gamma")
    {
        return Ok(WanVaeGeneration::V2_1);
    }
    bail!(
        "{} does not look like a Wan VAE (no decoder.middle.0.residual.0.gamma)",
        path.display()
    )
}

/// Derive the DiT config from the checkpoint's tensor shapes.
///
/// Shape-driven rather than name-driven, the way ComfyUI detects its models: a
/// repack, a community fine-tune, or a `--transformer` override can all carry
/// a name this code has never seen, but the shapes are the architecture.
///
/// - `patch_embedding.weight` `[dim, in_dim, pt, ph, pw]` gives the width, the
///   latent channel count, and the patch size.
/// - `blocks.{i}.ffn.0.weight` `[ffn_dim, dim]` gives the MLP width.
/// - `text_embedding.0.weight` / `time_embedding.0.weight` give the two
///   conditioning widths.
/// - `head.head.weight` `[out_dim * patch, dim]` gives the output channels.
/// - The highest `blocks.{i}.` index gives the depth.
pub(crate) fn detect_transformer_config(path: &Path) -> Result<WanTransformerConfig> {
    let shapes = header_shapes(path)?;
    let find = |key: &str| -> Result<&Vec<usize>> {
        shapes
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, shape)| shape)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} is missing `{key}` — not a Wan DiT checkpoint in the original key layout",
                    path.display()
                )
            })
    };

    let patch = find("patch_embedding.weight")?;
    if patch.len() != 5 {
        bail!(
            "Wan DiT: patch_embedding.weight must be 5-D [dim, in_dim, pt, ph, pw], got {patch:?}"
        );
    }
    let dim = patch[0];
    let in_dim = patch[1];
    let patch_size = (patch[2], patch[3], patch[4]);
    let patch_elems = patch_size.0 * patch_size.1 * patch_size.2;

    let ffn_dim = find("blocks.0.ffn.0.weight")?[0];
    let text_dim = find("text_embedding.0.weight")?[1];
    let freq_dim = find("time_embedding.0.weight")?[1];
    let head = find("head.head.weight")?;
    if !head[0].is_multiple_of(patch_elems) {
        bail!(
            "Wan DiT: head.head.weight rows {} are not a multiple of the {patch_elems}-element \
             patch",
            head[0]
        );
    }
    let out_dim = head[0] / patch_elems;

    let num_layers = shapes
        .iter()
        .filter_map(|(name, _)| name.strip_prefix("blocks."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|index| index.parse::<usize>().ok())
        .max()
        .map(|highest| highest + 1)
        .ok_or_else(|| anyhow::anyhow!("{} has no transformer blocks", path.display()))?;

    // Every shipped Wan variant uses a 128-wide head; the checkpoint does not
    // record the head count directly, so derive it from that invariant.
    if !dim.is_multiple_of(128) {
        bail!("Wan DiT: model width {dim} is not a multiple of the 128-wide attention head");
    }
    let num_heads = dim / 128;

    let config = WanTransformerConfig {
        dim,
        ffn_dim,
        num_heads,
        num_layers,
        in_dim,
        out_dim,
        text_dim,
        freq_dim,
        patch_size,
        eps: 1e-6,
        rope_max_seq_len: WanTransformerConfig::t2v_1_3b().rope_max_seq_len,
    };
    Ok(config)
}

/// Resolve the flow shift, honouring `MOLD_WAN_SHIFT`.
fn resolve_flow_shift() -> Result<f64> {
    let Ok(raw) = std::env::var(FLOW_SHIFT_ENV) else {
        return Ok(DEFAULT_FLOW_SHIFT);
    };
    let parsed: f64 = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{FLOW_SHIFT_ENV} must be a number, got {raw:?}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        bail!("{FLOW_SHIFT_ENV} must be finite and positive, got {parsed}");
    }
    Ok(parsed)
}

/// Reject the conditioning inputs this layer cannot honour.
///
/// Silently dropping them would render a plain text-to-video clip and look
/// like the conditioning simply had no effect.
fn reject_unsupported_conditioning(req: &GenerateRequest) -> Result<()> {
    let unsupported = [
        (
            req.source_image.is_some() || req.source_image_name.is_some(),
            "source_image",
        ),
        (
            req.source_video.is_some() || req.source_video_path.is_some(),
            "source_video",
        ),
        (
            req.extend_video.is_some() || req.extend_video_path.is_some(),
            "extend_video",
        ),
        (req.keyframes.is_some(), "keyframes"),
    ];
    for (present, field) in unsupported {
        if present {
            bail!(
                "{field} is not yet supported for Wan — this layer ships text-to-video only; \
                 image and video conditioning land with I2V/TI2V support"
            );
        }
    }
    Ok(())
}

/// Whether a request needs the unconditional pass.
///
/// At guidance <= 1 the CFG combination reduces to the conditional prediction,
/// so the second forward is pure waste. Skipping it is what makes the 4-step
/// Lightning recipe fast.
pub(crate) fn needs_cfg_pass(guidance: f64) -> bool {
    guidance > 1.0
}

pub struct WanEngine {
    base: EngineBase<()>,
    shared_pool: Option<Arc<Mutex<SharedPool>>>,
    pending_placement: Option<mold_core::types::DevicePlacement>,
}

impl WanEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
        shared_pool: Option<Arc<Mutex<SharedPool>>>,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            shared_pool,
            pending_placement: None,
        }
    }

    /// UMT5 weight shards. The manifest ships one fp16 safetensors, but the
    /// multi-shard field is the general contract.
    fn text_encoder_paths(&self) -> Result<Vec<PathBuf>> {
        let paths = &self.base.paths;
        if !paths.text_encoder_files.is_empty() {
            return Ok(paths.text_encoder_files.clone());
        }
        // Fall back to the generic T5 slot so a hand-configured model that
        // routes UMT5 through `t5_encoder` still loads.
        paths
            .t5_encoder
            .as_ref()
            .map(|path| vec![path.clone()])
            .ok_or_else(|| {
                anyhow::anyhow!("Wan: no UMT5 encoder weights configured for this model")
            })
    }

    fn tokenizer_path(&self) -> Result<PathBuf> {
        let paths = &self.base.paths;
        paths
            .text_tokenizer
            .clone()
            .or_else(|| paths.t5_tokenizer.clone())
            .ok_or_else(|| anyhow::anyhow!("Wan: no UMT5 tokenizer configured for this model"))
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        reject_unsupported_conditioning(req)?;

        let progress = &self.base.progress;
        let paths = &self.base.paths;

        // ------------------------------------------------------------------
        // Shape-driven configuration
        // ------------------------------------------------------------------
        let vae_generation = detect_vae_generation(&paths.vae)?;
        let vae_config = vae_generation.config();
        let transformer_config = detect_transformer_config(&paths.transformer)?;
        if transformer_config.in_dim != vae_config.z_dim {
            bail!(
                "Wan: the transformer expects {} latent channels but the VAE produces {} — the \
                 checkpoint and VAE generations do not match",
                transformer_config.in_dim,
                vae_config.z_dim
            );
        }

        let (default_frames, default_fps) = vae_generation.default_timing();
        let num_frames = req.frames.unwrap_or(default_frames);
        let fps = req.fps.unwrap_or(default_fps);
        let steps = req.steps;
        let guidance = req.guidance;
        let seed = req.seed.unwrap_or_else(rand_seed);
        let (width, height) = (req.width, req.height);

        if num_frames == 0 || !(num_frames as usize - 1).is_multiple_of(VAE_TEMPORAL_COMPRESSION) {
            bail!(
                "Wan requires a 4n+1 frame count (1, 5, 9, ... 81, 121), got {}",
                num_frames
            );
        }
        // The DiT patches the latent 2x2 on top of the VAE's spatial stride,
        // so the pixel grid must clear both.
        let spatial_grid = vae_config.spatial_compression() * transformer_config.patch_size.1;
        if !width.is_multiple_of(spatial_grid as u32) || !height.is_multiple_of(spatial_grid as u32)
        {
            bail!(
                "Wan requires width and height to be multiples of {spatial_grid}, got {width}x{height}"
            );
        }

        let latent_frames = (num_frames as usize - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let latent_h = height as usize / vae_config.spatial_compression();
        let latent_w = width as usize / vae_config.spatial_compression();
        let shift = resolve_flow_shift()?;
        let needs_cfg = needs_cfg_pass(guidance);

        let device = crate::device::create_device(self.base.gpu_ordinal, progress)?;
        let dtype = gpu_dtype(&device);

        progress.info(&format!(
            "Wan: {width}x{height} x {num_frames} frames @ {fps} fps, {steps} steps, \
             guidance {guidance:.1}, shift {shift:.1}, seed {seed}"
        ));
        if !needs_cfg {
            progress.info("Guidance <= 1: running one forward per step (no CFG pass)");
        }

        // ------------------------------------------------------------------
        // 1. Prompt encoding, then drop the encoder before denoise
        // ------------------------------------------------------------------
        progress.stage_start("Loading UMT5-XXL encoder");
        let encoder_start = Instant::now();
        let tokenizer_path = self.tokenizer_path()?;
        let tokenizer = match &self.shared_pool {
            Some(pool) => pool.lock().unwrap().load_tokenizer(&tokenizer_path)?,
            None => Arc::new(
                tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("Wan: loading UMT5 tokenizer failed: {e}"))?,
            ),
        };
        let text_device = crate::device::resolve_device(
            Some(
                self.pending_placement
                    .as_ref()
                    .map(|placement| placement.text_encoders.clone())
                    .unwrap_or_default(),
            ),
            || Ok(device.clone()),
        )?;
        let mut encoder = WanTextEncoder::load_with_tokenizer(
            &self.text_encoder_paths()?,
            &text_device,
            dtype,
            tokenizer,
        )?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading UMT5-XXL encoder",
            encoder_start.elapsed(),
        );

        progress.stage_start("Encoding prompt");
        let encode_start = Instant::now();
        let negative = req
            .negative_prompt
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or(mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT);
        let prompts: Vec<&str> = if needs_cfg {
            vec![req.prompt.as_str(), negative]
        } else {
            vec![req.prompt.as_str()]
        };
        let embeds = encoder
            .encode(&prompts)?
            .to_device(&device)?
            .to_dtype(dtype)?;
        let cond_embeds = embeds.narrow(0, 0, 1)?.contiguous()?;
        let uncond_embeds = if needs_cfg {
            Some(embeds.narrow(0, 1, 1)?.contiguous()?)
        } else {
            None
        };
        drop(embeds);
        progress.phase_done(
            ProgressPhase::PromptEncode,
            "Encoding prompt",
            encode_start.elapsed(),
        );

        // The encoder is 11.4 GB at fp16; it must be gone before the DiT loads.
        encoder.drop_weights();
        drop(encoder);
        device.synchronize()?;
        progress.info("UMT5 encoder dropped, VRAM freed");

        // ------------------------------------------------------------------
        // 2. Denoise
        // ------------------------------------------------------------------
        progress.stage_start("Loading Wan transformer");
        let transformer_start = Instant::now();
        let transformer = WanTransformer::from_safetensors_file(
            &paths.transformer,
            transformer_config.clone(),
            &device,
            dtype,
        )?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading Wan transformer",
            transformer_start.elapsed(),
        );

        let schedule = WanSchedule::new(WanScheduleConfig::new(steps as usize, shift))?;
        let mut solver = FlowUniPc::new(schedule.clone());
        let mut latents = seeded_randn(
            seed,
            &[1, vae_config.z_dim, latent_frames, latent_h, latent_w],
            &device,
            DType::F32,
        )?
        .to_dtype(dtype)?;

        // Hoisted: the rotation tables depend only on the latent grid, which
        // is fixed for the whole run.
        let rope = transformer.rope_freqs_for(&latents)?;

        progress.stage_start("Denoising");
        for (index, timestep) in schedule.timesteps.iter().enumerate() {
            progress.checkpoint()?;
            let step_start = Instant::now();
            let timestep_tensor =
                Tensor::from_vec(vec![*timestep as f32], 1, &device)?.to_dtype(dtype)?;

            let cond =
                transformer.forward_with_rope(&latents, &timestep_tensor, &cond_embeds, &rope)?;
            let velocity = match &uncond_embeds {
                Some(uncond_embeds) => {
                    let uncond = transformer.forward_with_rope(
                        &latents,
                        &timestep_tensor,
                        uncond_embeds,
                        &rope,
                    )?;
                    apply_cfg(&cond, &uncond, guidance)?
                }
                None => cond,
            };
            latents = solver.step(&velocity, index, &latents)?;

            progress.emit(ProgressEvent::DenoiseStep {
                step: index + 1,
                total: steps as usize,
                elapsed: step_start.elapsed(),
            });
        }
        progress.checkpoint()?;
        drop(transformer);
        device.synchronize()?;

        // ------------------------------------------------------------------
        // 3. VAE decode
        // ------------------------------------------------------------------
        progress.stage_start("Loading Wan VAE");
        let vae_start = Instant::now();
        let vae = WanVideoVae::from_safetensors(&paths.vae, vae_config, &device, dtype)?;
        progress.phase_done(
            ProgressPhase::ModelLoad,
            "Loading Wan VAE",
            vae_start.elapsed(),
        );

        progress.stage_start("Decoding video frames");
        let decode_start = Instant::now();
        let video = vae.decode(&latents)?;
        drop(vae);
        device.synchronize()?;
        progress.phase_done(
            ProgressPhase::Vae,
            "Decoding video frames",
            decode_start.elapsed(),
        );

        // ------------------------------------------------------------------
        // 4. Encode the artifact
        // ------------------------------------------------------------------
        let output_format = if req.resolved_output_format().is_video() {
            req.resolved_output_format()
        } else {
            OutputFormat::Apng
        };
        let format_name = output_format.extension().to_uppercase();
        progress.stage_start(&format!("Encoding {format_name}"));
        let encode_start = Instant::now();

        let frames = video_frames_to_images(&video, width, height)?;
        let frame_count = frames.len() as u32;
        let video_bytes = match output_format {
            OutputFormat::Apng => {
                let metadata = video_enc::VideoMetadata {
                    prompt: req.prompt.clone(),
                    model: self.base.model_name.clone(),
                    seed,
                    steps,
                    guidance,
                    width,
                    height,
                    frames: frame_count,
                    fps,
                };
                video_enc::encode_apng(&frames, fps, Some(&metadata))?
            }
            OutputFormat::Gif => video_enc::encode_gif(&frames, fps)?,
            #[cfg(feature = "webp")]
            OutputFormat::Webp => video_enc::encode_webp(&frames, fps)?,
            #[cfg(feature = "mp4")]
            OutputFormat::Mp4 => video_enc::encode_mp4(&frames, fps)?,
            #[cfg(not(feature = "webp"))]
            OutputFormat::Webp => {
                bail!("WebP output requires the 'webp' feature — rebuild with --features webp")
            }
            #[cfg(not(feature = "mp4"))]
            OutputFormat::Mp4 => {
                bail!("MP4 output requires the 'mp4' feature — rebuild with --features mp4")
            }
            _ => bail!("{format_name} is not a supported video output format"),
        };
        let thumbnail = video_enc::first_frame_png(&frames)?;
        let gif_preview = if req.gif_preview {
            if output_format == OutputFormat::Gif {
                video_bytes.clone()
            } else {
                video_enc::encode_gif(&frames, fps)?
            }
        } else {
            Vec::new()
        };
        progress.stage_done(&format!("Encoding {format_name}"), encode_start.elapsed());

        let generation_time_ms = start.elapsed().as_millis() as u64;
        progress.info(&format!(
            "Done: {frame_count} frames, {:.1}s total",
            generation_time_ms as f64 / 1000.0
        ));

        Ok(GenerateResponse {
            audio: None,
            images: vec![],
            video: Some(VideoData {
                data: video_bytes,
                format: output_format,
                width,
                height,
                frames: frame_count,
                fps,
                // `pipeline` is LTX-2's provenance slot; Wan has no pipeline
                // variants to record.
                pipeline: None,
                thumbnail,
                gif_preview,
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            }),
            generation_time_ms,
            model: self.base.model_name.clone(),
            seed_used: seed,
            gpu: None,
        })
    }
}

/// `[1, 3, F, H, W]` in `[-1, 1]` to RGB frames, resampling if the VAE's
/// output grid differs from what the caller asked for.
fn video_frames_to_images(video: &Tensor, width: u32, height: u32) -> Result<Vec<image::RgbImage>> {
    let (_, channels, frame_count, decoded_h, decoded_w) = video.dims5()?;
    if channels != 3 {
        bail!("Wan VAE decoded {channels} channels, expected 3");
    }
    let bytes = ((video.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .i(0)?;
    let mut frames = Vec::with_capacity(frame_count);
    for index in 0..frame_count {
        let frame = bytes
            .i((.., index, .., ..))?
            .contiguous()?
            .permute((1, 2, 0))?;
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let mut rgb = image::RgbImage::from_raw(decoded_w as u32, decoded_h as u32, data)
            .ok_or_else(|| anyhow::anyhow!("Wan: could not build frame {index}"))?;
        if decoded_w as u32 != width || decoded_h as u32 != height {
            rgb =
                image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Triangle);
        }
        frames.push(rgb);
    }
    Ok(frames)
}

impl crate::engine::InferenceEngine for WanEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.pending_placement = req.placement.clone();
        let result = self.generate_inner(req);
        self.pending_placement = None;
        result
    }

    fn model_name(&self) -> &str {
        &self.base.model_name
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        // Components are loaded and dropped inside `generate`; see the module
        // docs for why the 11.4 GB encoder cannot stay resident.
        Ok(())
    }

    fn unload(&mut self) {
        self.base.unload();
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
        crate::batch_execution_capability_for_family("wan")
            .expect("production Wan batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InferenceEngine;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use std::collections::HashMap;

    fn dummy_paths() -> ModelPaths {
        ModelPaths {
            transformer: PathBuf::from("/tmp/wan-transformer"),
            transformer_shards: vec![],
            vae: PathBuf::from("/tmp/wan-vae"),
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
            text_tokenizer: None,
            decoder: None,
        }
    }

    fn request() -> GenerateRequest {
        GenerateRequest {
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: "wan21-t2v-1.3b:bf16".to_string(),
            width: 832,
            height: 480,
            steps: 4,
            guidance: 6.0,
            seed: Some(7),
            batch_size: 1,
            output_format: None,
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
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
            frames: Some(81),
            fps: Some(16),
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
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
        }
    }

    /// The registry force-constructs every family with paths that do not
    /// exist; construction must not touch the filesystem.
    #[test]
    fn constructs_without_weights() {
        let engine = WanEngine::new(
            "wan21-t2v-1.3b:bf16".into(),
            dummy_paths(),
            LoadStrategy::Eager,
            0,
            None,
        );
        assert_eq!(engine.model_name(), "wan21-t2v-1.3b:bf16");
        assert!(
            !engine.is_loaded(),
            "an eagerly-strategised engine holds no weights until load"
        );
        assert!(engine.model_paths().is_some());
        assert_eq!(
            engine.batch_execution_capability(),
            crate::batch_execution_capability_for_family("wan").unwrap()
        );
    }

    /// Under the CLI's sequential strategy the engine reports ready because
    /// it loads on demand — the same contract `LtxVideoEngine` publishes.
    #[test]
    fn sequential_strategy_reports_ready() {
        let mut engine = WanEngine::new(
            "wan21-t2v-1.3b:bf16".into(),
            dummy_paths(),
            LoadStrategy::Sequential,
            0,
            None,
        );
        engine.load().unwrap();
        assert!(engine.is_loaded());
    }

    #[test]
    fn conditioning_inputs_are_rejected_with_a_clear_message() {
        for mutate in [
            (|req: &mut GenerateRequest| req.source_image = Some(vec![1, 2, 3])) as fn(&mut _),
            |req: &mut GenerateRequest| req.source_video = Some(vec![1, 2, 3]),
            |req: &mut GenerateRequest| req.extend_video = Some(vec![1, 2, 3]),
            |req: &mut GenerateRequest| req.source_image_name = Some("cat.png".into()),
        ] {
            let mut req = request();
            mutate(&mut req);
            let error = reject_unsupported_conditioning(&req)
                .expect_err("conditioning must be refused")
                .to_string();
            assert!(
                error.contains("not yet supported for Wan"),
                "unexpected error: {error}"
            );
            assert!(
                error.contains("text-to-video only"),
                "the error must say what the engine does support: {error}"
            );
        }
        // A plain text-to-video request passes.
        reject_unsupported_conditioning(&request()).unwrap();
    }

    /// The Lightning recipe runs at guidance 1.0 and must not pay for a second
    /// forward per step.
    #[test]
    fn cfg_pass_is_skipped_at_or_below_unit_guidance() {
        assert!(!needs_cfg_pass(1.0));
        assert!(!needs_cfg_pass(0.0));
        assert!(needs_cfg_pass(1.0001));
        assert!(needs_cfg_pass(5.0));
        assert!(needs_cfg_pass(6.0));
    }

    #[test]
    fn flow_shift_defaults_and_validates() {
        // The env var is process-global; this test owns it for its duration.
        let previous = std::env::var(FLOW_SHIFT_ENV).ok();
        unsafe { std::env::remove_var(FLOW_SHIFT_ENV) };
        assert_eq!(resolve_flow_shift().unwrap(), DEFAULT_FLOW_SHIFT);

        unsafe { std::env::set_var(FLOW_SHIFT_ENV, "3.5") };
        assert_eq!(resolve_flow_shift().unwrap(), 3.5);

        for bad in ["", "abc", "0", "-2", "inf"] {
            unsafe { std::env::set_var(FLOW_SHIFT_ENV, bad) };
            assert!(resolve_flow_shift().is_err(), "{bad:?} must be rejected");
        }

        match previous {
            Some(value) => unsafe { std::env::set_var(FLOW_SHIFT_ENV, value) },
            None => unsafe { std::env::remove_var(FLOW_SHIFT_ENV) },
        }
    }

    #[test]
    fn vae_generations_carry_their_own_geometry_and_timing() {
        assert_eq!(WanVaeGeneration::V2_1.config().z_dim, 16);
        assert_eq!(WanVaeGeneration::V2_1.config().spatial_compression(), 8);
        assert_eq!(WanVaeGeneration::V2_1.default_timing(), (81, 16));

        assert_eq!(WanVaeGeneration::V2_2.config().z_dim, 48);
        assert_eq!(WanVaeGeneration::V2_2.config().spatial_compression(), 16);
        assert_eq!(WanVaeGeneration::V2_2.default_timing(), (121, 24));
    }

    /// Write a header-only safetensors file carrying the given shapes, so the
    /// detection probes can be exercised without real weights.
    fn write_header(path: &Path, shapes: &[(&str, &[usize])]) {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, shape) in shapes {
            tensors.insert(
                (*name).to_string(),
                Tensor::zeros(*shape, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    #[test]
    fn transformer_config_is_detected_from_checkpoint_shapes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("wan.safetensors");
        // The 1.3B geometry: dim 1536, ffn 8960, 30 layers, in/out 16.
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![1536, 16, 1, 2, 2]),
            ("blocks.0.ffn.0.weight".into(), vec![8960, 1536]),
            ("text_embedding.0.weight".into(), vec![1536, 4096]),
            ("time_embedding.0.weight".into(), vec![1536, 256]),
            ("head.head.weight".into(), vec![64, 1536]),
        ];
        for layer in 0..30 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 1536]));
        }
        let borrowed: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&path, &borrowed);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::t2v_1_3b());
    }

    #[test]
    fn transformer_detection_reads_ti2v_geometry() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("ti2v.safetensors");
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedding.weight".into(), vec![3072, 48, 1, 2, 2]),
            ("blocks.0.ffn.0.weight".into(), vec![14336, 3072]),
            ("text_embedding.0.weight".into(), vec![3072, 4096]),
            ("time_embedding.0.weight".into(), vec![3072, 256]),
            // out_dim 48 x patch 4.
            ("head.head.weight".into(), vec![192, 3072]),
        ];
        for layer in 0..30 {
            shapes.push((format!("blocks.{layer}.modulation"), vec![1, 6, 3072]));
        }
        let borrowed: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();
        write_header(&path, &borrowed);

        let config = detect_transformer_config(&path).unwrap();
        assert_eq!(config, WanTransformerConfig::ti2v_5b());
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn transformer_detection_rejects_a_foreign_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("not-wan.safetensors");
        write_header(&path, &[("some.other.weight", &[16, 16])]);
        let error = detect_transformer_config(&path).unwrap_err().to_string();
        assert!(error.contains("patch_embedding.weight"), "{error}");
    }

    /// The VAE generation must come from the checkpoint's key layout, since
    /// the 2.2 nesting is the only reliable discriminator.
    #[test]
    fn vae_generation_is_detected_from_the_key_layout() {
        let temp = tempfile::tempdir().unwrap();

        let v21 = temp.path().join("wan_2.1_vae.safetensors");
        write_header(
            &v21,
            &[
                ("decoder.middle.0.residual.0.gamma", &[384, 1, 1, 1]),
                (
                    "decoder.upsamples.0.residual.2.weight",
                    &[384, 384, 3, 3, 3],
                ),
            ],
        );
        assert_eq!(detect_vae_generation(&v21).unwrap(), WanVaeGeneration::V2_1);

        let v22 = temp.path().join("wan2.2_vae.safetensors");
        write_header(
            &v22,
            &[
                ("decoder.middle.0.residual.0.gamma", &[1024, 1, 1, 1]),
                (
                    "decoder.upsamples.0.upsamples.0.residual.2.weight",
                    &[1024, 1024, 3, 3, 3],
                ),
            ],
        );
        assert_eq!(detect_vae_generation(&v22).unwrap(), WanVaeGeneration::V2_2);

        let foreign = temp.path().join("foreign.safetensors");
        write_header(&foreign, &[("encoder.conv_in.weight", &[4, 4])]);
        assert!(detect_vae_generation(&foreign).is_err());
    }

    /// End-to-end on CPU at toy widths: tiny DiT + tiny VAE + a real
    /// FlowUniPC schedule, exercising the whole denoise/decode path including
    /// the CFG branch and the artifact encode.
    fn tiny_engine_run(guidance: f64, steps: u32) -> Vec<image::RgbImage> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let vae_config = WanVaeConfig::tiny_v2_1();
        let varmap = VarMap::new();
        let vae = WanVideoVae::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            vae_config.clone(),
            &device,
            dtype,
        )
        .unwrap();

        let transformer_config = WanTransformerConfig {
            in_dim: vae_config.z_dim,
            out_dim: vae_config.z_dim,
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let transformer_map = VarMap::new();
        let transformer = WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&transformer_map, dtype, &device),
            transformer_config.clone(),
        )
        .unwrap();

        // 5 pixel frames -> 2 latent frames; 32x32 pixels -> 4x4 latent.
        let (frames, width, height) = (5usize, 32u32, 32u32);
        let latent_frames = (frames - 1) / VAE_TEMPORAL_COMPRESSION + 1;
        let latent_h = height as usize / vae_config.spatial_compression();
        let latent_w = width as usize / vae_config.spatial_compression();

        let context = Tensor::zeros((1, 6, 32), dtype, &device).unwrap();
        let schedule = WanSchedule::new(WanScheduleConfig::new(steps as usize, 8.0)).unwrap();
        let mut solver = FlowUniPc::new(schedule.clone());
        let mut latents = seeded_randn(
            7,
            &[1, vae_config.z_dim, latent_frames, latent_h, latent_w],
            &device,
            dtype,
        )
        .unwrap();
        let rope = transformer.rope_freqs_for(&latents).unwrap();

        for (index, timestep) in schedule.timesteps.iter().enumerate() {
            let t = Tensor::from_vec(vec![*timestep as f32], 1, &device).unwrap();
            let cond = transformer
                .forward_with_rope(&latents, &t, &context, &rope)
                .unwrap();
            let velocity = if needs_cfg_pass(guidance) {
                let uncond = transformer
                    .forward_with_rope(&latents, &t, &context, &rope)
                    .unwrap();
                apply_cfg(&cond, &uncond, guidance).unwrap()
            } else {
                cond
            };
            latents = solver.step(&velocity, index, &latents).unwrap();
        }

        let video = vae.decode(&latents).unwrap();
        assert_eq!(
            video.dims(),
            &[1, 3, frames, height as usize, width as usize]
        );
        video_frames_to_images(&video, width, height).unwrap()
    }

    #[test]
    fn tiny_end_to_end_denoise_and_decode_produces_frames() {
        let frames = tiny_engine_run(5.0, 4);
        assert_eq!(frames.len(), 5);
        for frame in &frames {
            assert_eq!(frame.dimensions(), (32, 32));
        }

        // The artifact encoders must accept what the pipeline produces.
        let apng = video_enc::encode_apng(&frames, 16, None).unwrap();
        assert!(!apng.is_empty());
        let thumbnail = video_enc::first_frame_png(&frames).unwrap();
        assert!(!thumbnail.is_empty());
    }

    /// The single-pass path must reach the same shapes as the CFG path — the
    /// guidance branch changes the arithmetic, not the plumbing.
    #[test]
    fn single_pass_guidance_runs_the_same_pipeline() {
        let frames = tiny_engine_run(1.0, 4);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].dimensions(), (32, 32));
    }
}
