//! Qwen Image 2.1 inference engine wiring.
//!
//! The component implementations in this directory deliberately preserve the
//! checkpoint's native layout: a Qwen3-VL language submodel conditions a
//! 32-block causal-condition transformer over unpatched 64-channel latents,
//! followed by the 2.1 decoder. This engine keeps that sequence intact for
//! both eager and sequential residency modes.
//!
//! A request with ordered reference images (`edit_images`) adds three things
//! (diffusers `pipeline_qwenimage21.py`, cited `P:`): the Qwen3-VL vision
//! tower conditions the prompt encoding (`P:233-327`), the VAE encoder turns
//! each reference into a condition block laid into the joint sequence where
//! its image slots were (`P:423-476`), and the output keeps its alpha plane
//! when a reference carries transparency.

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::z_image::postprocess_image;
use mold_candle::qwen3_vl::Qwen3VlVisionModel;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat};
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use super::layout::QwenImage21JointLayout;
use super::reference::{
    encode_prompt_with_images, encode_vision, load_vision_tower, prepare_reference,
    PreparedReference, VisionFeatures,
};
use super::scheduler::{scheduler_for, transformer_timestep, ScheduleKind};
use super::transformer::QwenImage21Transformer;
use super::vae::QwenImage21Vae;
use super::vae_encoder::QwenImage21VaeEncoder;
use super::{
    encode_t2i_prompts, QwenImage21TextConditioning, QWEN_IMAGE_21_CANVAS_ALIGNMENT,
    QWEN_IMAGE_21_LATENT_CHANNELS, QWEN_IMAGE_21_VAE_SCALE_FACTOR,
};
use crate::device::{effective_device_ref, resolve_device};
use crate::encoders::qwen3::Qwen3Encoder;
use crate::encoders::qwen3_bf16::Qwen3BF16Config;
use crate::engine::{cfg_active, rand_seed, seeded_randn, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{
    alpha_output_for_request, build_output_metadata, encode_image_with_alpha, AlphaOutput,
};
use crate::progress::{ProgressCallback, ProgressEvent, ProgressPhase, ProgressReporter};

/// Components kept resident by an eager Qwen Image 2.1 engine.
struct LoadedQwenImage21 {
    transformer: QwenImage21Transformer,
    text_encoder: Qwen3Encoder,
    vae: QwenImage21Vae,
    /// Loaded on the first reference-conditioned request, then kept.
    vision: Option<Qwen3VlVisionModel>,
    /// Loaded on the first reference-conditioned request, then kept.
    vae_encoder: Option<QwenImage21VaeEncoder>,
    text_paths: Vec<PathBuf>,
    vae_path: PathBuf,
    device: Device,
    text_device: Device,
    vae_device: Device,
    dtype: DType,
    text_dtype: DType,
    vae_dtype: DType,
}

/// Native inference engine for `Qwen/Qwen-Image-2.1`.
pub struct QwenImage21Engine {
    base: EngineBase<LoadedQwenImage21>,
    /// Placement is request-scoped because it affects component construction.
    pending_placement: Option<mold_core::types::DevicePlacement>,
    /// Parity tests inject upstream's exact initial latents: torch's RNG is
    /// not mold's ChaCha stream, so a seed cannot reproduce them.
    #[cfg(test)]
    injected_latents: Option<Tensor>,
}

/// Whether the denoise loop rounds the transformer timestep through the
/// working dtype as upstream does. It moves pixels (BF16: 900 -> 0.8984375),
/// so it belongs to the execution path: `Qwen21ExecPath::legacy()` and Metal
/// keep v0.32's unrounded value. Until the exec path supplies it, every render
/// keeps v0.32's behaviour.
const ROUND_TIMESTEP_TO_DTYPE: bool = false;

/// The condition blocks of a reference-conditioned request: each
/// reference's latent grid and their packed, normalized latents
/// `[1, Σ h·w, 64]` in reference order, shared by both CFG branches.
pub(crate) struct ConditionBlocks {
    pub(crate) shapes: Vec<(usize, usize)>,
    pub(crate) latents: Tensor,
}

/// How a denoise begins.
pub(crate) struct DenoiseStart {
    pub(crate) seed: u64,
    /// Upstream's `latents=` (parity tests), else seeded noise.
    pub(crate) initial_latents: Option<Tensor>,
    pub(crate) round_timestep_to_dtype: bool,
    /// Condition-image blocks; `None` is text-to-image.
    pub(crate) condition: Option<ConditionBlocks>,
}

/// The normalized timestep the transformer receives. Unrounded is v0.32's
/// `scheduler_timestep / 1000` in f64; rounded is upstream's
/// `t.to(latents.dtype) / 1000` ([`transformer_timestep`]).
pub(crate) fn step_timestep(
    scheduler_timestep: f64,
    sigma: f64,
    dtype: DType,
    round_to_dtype: bool,
) -> f64 {
    if round_to_dtype {
        transformer_timestep(sigma, dtype)
    } else {
        scheduler_timestep / 1000.0
    }
}

/// The positive prompt the encoder reads: the model card's RGBA recipe
/// around the user's text when a transparent background was asked for
/// (`mold_core::transparency`), else the text unchanged. The negative prompt
/// is never wrapped.
pub(crate) fn positive_prompt(req: &GenerateRequest) -> Cow<'_, str> {
    if req.transparent_background == Some(true) {
        mold_core::transparency::apply_rgba_prompt_recipe(&req.prompt)
    } else {
        Cow::Borrowed(req.prompt.as_str())
    }
}

/// The request warning when a kept alpha plane had to be flattened.
pub(crate) fn alpha_warning(alpha: AlphaOutput, format: OutputFormat) -> Option<&'static str> {
    (alpha == AlphaOutput::Keep && format == OutputFormat::Jpeg).then_some(
        "A Qwen Image 2.1 reference carries transparency, which JPEG cannot store; the output was composited over white. Choose PNG or WebP to keep the alpha channel.",
    )
}

impl QwenImage21Engine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
            pending_placement: None,
            #[cfg(test)]
            injected_latents: None,
        }
    }

    /// Use `latents` `[1, target_tokens, 64]` as the next render's initial
    /// latents, exactly as upstream's `latents=` argument (no sigma scaling:
    /// the schedule starts at sigma 1).
    #[cfg(test)]
    pub(crate) fn inject_initial_latents(&mut self, latents: Tensor) {
        self.injected_latents = Some(latents);
    }

    fn take_initial_latents(&mut self) -> Option<Tensor> {
        #[cfg(test)]
        {
            self.injected_latents.take()
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    fn uses_sequential_generate_path(&self) -> bool {
        self.base.load_strategy == LoadStrategy::Sequential
    }

    fn transformer_paths(&self) -> Vec<PathBuf> {
        if self.base.paths.transformer_shards.is_empty() {
            vec![self.base.paths.transformer.clone()]
        } else {
            self.base.paths.transformer_shards.clone()
        }
    }

    /// Resolve and validate the exact artifacts this native engine consumes.
    fn validate_paths(&self) -> Result<(Vec<PathBuf>, Vec<PathBuf>, PathBuf, PathBuf)> {
        let transformer_paths = self.transformer_paths();
        if transformer_paths.is_empty() || transformer_paths.iter().any(|path| !path.is_file()) {
            bail!(
                "Qwen Image 2.1 transformer shards are incomplete: {:?}",
                transformer_paths
            );
        }
        let text_paths = self.base.paths.text_encoder_files.clone();
        if text_paths.is_empty() || text_paths.iter().any(|path| !path.is_file()) {
            bail!(
                "Qwen Image 2.1 Qwen3-VL text encoder shards are incomplete: {:?}",
                text_paths
            );
        }
        let tokenizer = self
            .base
            .paths
            .text_tokenizer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qwen Image 2.1 tokenizer is missing"))?;
        if !tokenizer.is_file() {
            bail!(
                "Qwen Image 2.1 tokenizer is missing: {}",
                tokenizer.display()
            );
        }
        let vae = self.base.paths.vae.clone();
        if !vae.is_file() {
            bail!("Qwen Image 2.1 VAE is missing: {}", vae.display());
        }
        Ok((transformer_paths, text_paths, tokenizer, vae))
    }

    fn resolve_devices(&self) -> Result<(Device, Device, Device)> {
        let transformer_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |advanced| Some(advanced.transformer.clone()),
            false,
        );
        let device = resolve_device(Some(transformer_ref), || {
            crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)
        })?;
        let text_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |advanced| advanced.qwen.clone(),
            true,
        );
        let text_device = resolve_device(Some(text_ref), || Ok(device.clone()))?;
        let vae_ref = effective_device_ref(
            self.pending_placement.as_ref(),
            |advanced| Some(advanced.vae.clone()),
            false,
        );
        let vae_device = resolve_device(Some(vae_ref), || Ok(device.clone()))?;
        Ok((device, text_device, vae_device))
    }

    /// Resolve which Qwen3-VL-8B language model to load (the BF16 shards or an
    /// official GGUF, per `MOLD_QWEN3_VARIANT` and what the card has left),
    /// and load it. `free_vram` is measured AFTER the transformer and VAE are
    /// resident on an eager engine, before anything on a sequential one —
    /// the same inputs `text_encoder_residency::plan` gives the planner.
    fn load_text_encoder(
        bf16_paths: &[PathBuf],
        tokenizer: &PathBuf,
        device: &Device,
        dtype: DType,
        free_vram: u64,
        progress: &ProgressReporter,
    ) -> Result<Qwen3Encoder> {
        let preference = crate::runtime_env::value("MOLD_QWEN3_VARIANT");
        let (paths, is_gguf, on_gpu, _label) =
            crate::encoders::variant_resolution::resolve_qwen3_variant(
                progress,
                preference.as_deref(),
                device,
                free_vram,
                bf16_paths,
                !bf16_paths.is_empty(),
                false,
                crate::encoders::variant_resolution::Qwen3Size::Vl8b,
            )?;
        let device = if on_gpu { device.clone() } else { Device::Cpu };
        if is_gguf {
            Qwen3Encoder::load_gguf(
                &paths[0],
                tokenizer,
                &device,
                &Qwen3BF16Config::qwen3_image_21_text_encoder(),
            )
        } else {
            Qwen3Encoder::load_bf16(
                &paths,
                tokenizer,
                &device,
                if device.is_cpu() {
                    crate::engine::gpu_dtype(&device)
                } else {
                    dtype
                },
                &Qwen3BF16Config::qwen3_image_21_text_encoder(),
                progress,
            )
        }
    }

    /// Free device bytes on the text encoder's device, or zero when it cannot
    /// be measured (CPU, or an unmeasurable card — both resolve as today).
    fn free_vram_for(device: &Device, ordinal: usize) -> u64 {
        if device.is_cuda() {
            crate::device::usable_free_vram_bytes(ordinal).unwrap_or(0)
        } else if device.is_metal() {
            crate::device::available_system_memory_bytes().unwrap_or(0)
        } else {
            0
        }
    }

    /// After an eager encode, apply the ONE residency decision
    /// (`text_encoder_residency::decide`) the planner priced: keep the encoder,
    /// park it in host RAM, or drop it for the denoise. The decision is
    /// returned so the caller applies its transformer-for-decode half too.
    fn settle_text_encoder_residency(
        progress: &ProgressReporter,
        paths: &ModelPaths,
        text_encoder: &mut Qwen3Encoder,
        req: &GenerateRequest,
        ordinal: usize,
        vae_dtype: DType,
    ) -> Result<super::text_encoder_residency::Qwen21TeDecision> {
        use super::text_encoder_residency as residency;
        let device = if !text_encoder.on_gpu || text_encoder.device.is_cpu() {
            residency::TeDevice::Cpu
        } else if text_encoder.device.is_metal() {
            residency::TeDevice::Metal
        } else {
            residency::TeDevice::Cuda
        };
        let transformer_bytes = residency::transformer_device_bytes(paths);
        let vae_bytes = std::fs::metadata(&paths.vae).map_or(0, |metadata| metadata.len());
        let text_encoder_bytes =
            residency::text_encoder_device_bytes(text_encoder.encoder_paths()).unwrap_or(0);
        let resident_now = transformer_bytes.saturating_add(vae_bytes).saturating_add(
            if text_encoder.model.is_some() {
                text_encoder_bytes
            } else {
                0
            },
        );
        let usable_free_bytes = match device {
            residency::TeDevice::Cuda => crate::device::usable_free_vram_bytes(ordinal)
                .map_or(0, |free| free.saturating_add(resident_now)),
            _ => 0,
        };
        let (denoise_workspace_bytes, decode_peak_bytes) = residency::render_workspace_bytes(
            req.width,
            req.height,
            1,
            crate::device::dtype_bytes(vae_dtype),
        );
        let decision = residency::decide(&residency::Qwen21TeBudget {
            device,
            usable_free_bytes,
            transformer_bytes,
            vae_bytes,
            text_encoder_bytes,
            denoise_workspace_bytes,
            decode_peak_bytes,
            host_total_bytes: crate::flux::pinned::total_system_ram_bytes().unwrap_or(0),
            host_available_bytes: crate::device::available_host_ram_bytes().unwrap_or(0),
            already_parked_bytes: text_encoder.parked_bytes(),
            keep_te_ram: crate::device::keep_te_ram_mode(),
        });
        match decision.residency {
            residency::Qwen21TeResidency::Resident => {}
            residency::Qwen21TeResidency::ParkHost => {
                progress.info(&format!(
                    "Parking Qwen3-VL text encoder: {}",
                    decision.reason
                ));
                text_encoder.park_to_cpu()?;
            }
            residency::Qwen21TeResidency::Drop => {
                progress.info(&format!(
                    "Dropping Qwen3-VL text encoder: {}",
                    decision.reason
                ));
                text_encoder.drop_weights();
            }
        }
        Ok(decision)
    }

    fn load_vision(
        progress: &ProgressReporter,
        paths: &[PathBuf],
        device: &Device,
        dtype: DType,
    ) -> Result<Qwen3VlVisionModel> {
        let label = format!("Loading Qwen3-VL vision tower ({})", device_label(device));
        progress.stage_start(&label);
        let start = Instant::now();
        let tower = load_vision_tower(paths, device, dtype, progress)?;
        progress.stage_done(&label, start.elapsed());
        Ok(tower)
    }

    fn load_vae_encoder(
        progress: &ProgressReporter,
        path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> Result<QwenImage21VaeEncoder> {
        let label = format!(
            "Loading Qwen Image 2.1 VAE encoder ({})",
            device_label(device)
        );
        progress.stage_start(&label);
        let start = Instant::now();
        let encoder = QwenImage21VaeEncoder::load(path, device, dtype, progress)?;
        progress.stage_done(&label, start.elapsed());
        Ok(encoder)
    }

    /// Load all components for the eager path.
    pub fn load(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }
        if self.uses_sequential_generate_path() {
            return Ok(());
        }

        let (transformer_paths, text_paths, tokenizer, vae_path) = self.validate_paths()?;
        let (device, text_device, vae_device) = self.resolve_devices()?;
        let dtype = super::transformer_dtype(&device);
        let text_dtype = crate::engine::gpu_dtype(&text_device);
        let vae_dtype = crate::engine::gpu_dtype(&vae_device);

        let transformer_label = format!(
            "Loading Qwen Image 2.1 transformer ({} shards)",
            transformer_paths.len()
        );
        self.base.progress.stage_start(&transformer_label);
        let transformer_start = Instant::now();
        let transformer =
            QwenImage21Transformer::load(&transformer_paths, &device, dtype, &self.base.progress)?;
        self.base
            .progress
            .stage_done(&transformer_label, transformer_start.elapsed());

        let vae_label = format!("Loading Qwen Image 2.1 VAE ({})", device_label(&vae_device));
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = QwenImage21Vae::load(&vae_path, &vae_device, vae_dtype, &self.base.progress)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());

        let text_label = format!(
            "Loading Qwen3-VL text encoder ({} shards, {})",
            text_paths.len(),
            device_label(&text_device)
        );
        self.base.progress.stage_start(&text_label);
        let text_start = Instant::now();
        let text_encoder = Self::load_text_encoder(
            &text_paths,
            &tokenizer,
            &text_device,
            text_dtype,
            Self::free_vram_for(&text_device, self.base.gpu_ordinal),
            &self.base.progress,
        )?;
        self.base
            .progress
            .stage_done(&text_label, text_start.elapsed());

        self.base.loaded = Some(LoadedQwenImage21 {
            transformer,
            text_encoder,
            vae,
            vision: None,
            vae_encoder: None,
            text_paths,
            vae_path,
            device,
            text_device,
            vae_device,
            dtype,
            text_dtype,
            vae_dtype,
        });
        Ok(())
    }

    /// Keep direct callers honest too; the server validates the same contracts
    /// earlier, but an inference engine must never silently omit input media.
    fn validate_request(req: &GenerateRequest) -> Result<()> {
        anyhow::ensure!(
            req.batch_size == 1,
            "Qwen Image 2.1 currently supports one image per request"
        );
        anyhow::ensure!(
            req.steps > 0,
            "Qwen Image 2.1 requires at least one denoise step"
        );
        anyhow::ensure!(
            req.width > 0
                && req.height > 0
                && (req.width as usize).is_multiple_of(QWEN_IMAGE_21_CANVAS_ALIGNMENT)
                && (req.height as usize).is_multiple_of(QWEN_IMAGE_21_CANVAS_ALIGNMENT),
            "Qwen Image 2.1 width and height must be positive multiples of {QWEN_IMAGE_21_CANVAS_ALIGNMENT}"
        );
        if let Some(images) = &req.edit_images {
            let max = mold_core::validation::QWEN_IMAGE21_MAX_REFERENCE_IMAGES as usize;
            anyhow::ensure!(
                (1..=max).contains(&images.len()),
                "Qwen Image 2.1 takes 1 to {max} reference images, got {}",
                images.len()
            );
        }
        let other_media = req.references.is_some()
            || req.source_image.is_some()
            || req.source_image_name.is_some()
            || req.id_image.is_some()
            || req.id_image_name.is_some()
            || req.id_images.is_some()
            || req.id_image_names.is_some()
            || req.mask_image.is_some()
            || req.control_image.is_some()
            || req.audio_file.is_some()
            || req.audio_file_path.is_some()
            || req.source_video.is_some()
            || req.source_video_path.is_some()
            || req.extend_video.is_some()
            || req.extend_video_path.is_some()
            || req.keyframes.is_some();
        anyhow::ensure!(
            !other_media && req.control_model.is_none() && req.mesh.is_none(),
            "Qwen Image 2.1 conditions on ordered reference images (edit_images) only; source, mask, identity, control, audio, video, and mesh inputs are not supported"
        );
        anyhow::ensure!(
            req.caller_lora_stack().is_empty(),
            "Qwen Image 2.1 LoRA adapters are not implemented"
        );
        anyhow::ensure!(
            mold_core::manifest::qwen_image21_turbo_schedule(&req.model).is_none(),
            "Qwen Image 2.1 turbo tiers need their distilled adapter, which this build cannot apply yet"
        );
        let format = req.resolved_output_format();
        anyhow::ensure!(
            matches!(
                format,
                OutputFormat::Png | OutputFormat::Jpeg | OutputFormat::Webp
            ),
            "Qwen Image 2.1 supports PNG, WebP and JPEG output"
        );
        anyhow::ensure!(
            !(req.transparent_background == Some(true) && format == OutputFormat::Jpeg),
            "Qwen Image 2.1 cannot deliver a transparent background as JPEG; choose PNG or WebP"
        );
        Ok(())
    }

    /// Decode and resize every reference, in request order.
    fn prepare_references(
        progress: &ProgressReporter,
        req: &GenerateRequest,
    ) -> Result<Vec<PreparedReference>> {
        let Some(images) = req.edit_images.as_deref() else {
            return Ok(Vec::new());
        };
        let label = format!("Preparing {} reference image(s)", images.len());
        progress.stage_start(&label);
        let start = Instant::now();
        let mut references = Vec::with_capacity(images.len());
        for bytes in images {
            progress.checkpoint()?;
            references.push(prepare_reference(bytes)?);
        }
        progress.stage_done(&label, start.elapsed());
        Ok(references)
    }

    fn run_vision(
        progress: &ProgressReporter,
        tower: &Qwen3VlVisionModel,
        references: &[PreparedReference],
        device: &Device,
    ) -> Result<VisionFeatures> {
        let label = "Encoding reference images (Qwen3-VL vision)";
        progress.stage_start(label);
        let start = Instant::now();
        let features = encode_vision(
            tower,
            references,
            device,
            &mut || Ok(progress.checkpoint()?),
        )?;
        progress.stage_done(label, start.elapsed());
        Ok(features)
    }

    /// VAE-encode every reference into its condition block (`P:423-476`).
    fn encode_references(
        progress: &ProgressReporter,
        encoder: &QwenImage21VaeEncoder,
        references: &[PreparedReference],
        vae: (&Device, DType),
        target: (&Device, DType),
    ) -> Result<ConditionBlocks> {
        let label = "Encoding reference images (VAE)";
        progress.stage_start(label);
        let start = Instant::now();
        let mut blocks = Vec::with_capacity(references.len());
        let mut shapes = Vec::with_capacity(references.len());
        for reference in references {
            progress.checkpoint()?;
            let input = reference.vae_input(vae.0, vae.1)?;
            let packed = {
                let _conv = crate::conv_policy::ConvScope::for_family("qwen-image21");
                encoder.encode_packed(&input)?
            };
            blocks.push(packed.to_device(target.0)?.to_dtype(target.1)?);
            shapes.push(reference.latent_shape());
        }
        let latents = Tensor::cat(&blocks, 1)?;
        progress.phase_done(ProgressPhase::Vae, label, start.elapsed());
        Ok(ConditionBlocks { shapes, latents })
    }

    fn encode_conditioning(
        progress: &ProgressReporter,
        text_encoder: &mut Qwen3Encoder,
        req: &GenerateRequest,
        vision: Option<&VisionFeatures>,
        target: (&Device, DType),
    ) -> Result<(
        QwenImage21TextConditioning,
        Option<QwenImage21TextConditioning>,
    )> {
        let (target_device, target_dtype) = target;
        let label = "Encoding prompt (Qwen3-VL)";
        progress.stage_start(label);
        let start = Instant::now();
        let mut encode = |prompt: &str| -> Result<QwenImage21TextConditioning> {
            let conditioning = match vision {
                Some(vision) => encode_prompt_with_images(text_encoder, vision, prompt)?,
                None => encode_t2i_prompts(text_encoder, &[prompt.to_string()])?,
            };
            conditioning.to_device_dtype(target_device, target_dtype)
        };
        let conditional = encode(&positive_prompt(req))?;

        let unconditional = if cfg_active(req.guidance) {
            match req.negative_prompt.as_ref() {
                Some(negative_prompt) => Some(encode(negative_prompt)?),
                None => {
                    progress.info(
                        "Qwen Image 2.1 guidance is above 1, but no negative prompt was supplied; using the native conditional-only path",
                    );
                    None
                }
            }
        } else {
            if req.negative_prompt.is_some() {
                progress.info(
                    "Qwen Image 2.1 ignores the negative prompt when guidance is at or below 1",
                );
            }
            None
        };
        progress.phase_done(ProgressPhase::PromptEncode, label, start.elapsed());
        Ok((conditional, unconditional))
    }

    fn denoise(
        progress: &ProgressReporter,
        req: &GenerateRequest,
        transformer: &QwenImage21Transformer,
        conditioning: &QwenImage21TextConditioning,
        negative_conditioning: Option<&QwenImage21TextConditioning>,
        compute: (&Device, DType),
        start: DenoiseStart,
    ) -> Result<(Tensor, usize, usize)> {
        let (device, dtype) = compute;
        let DenoiseStart {
            seed,
            initial_latents,
            round_timestep_to_dtype,
            condition,
        } = start;
        let latent_height = req.height as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
        let latent_width = req.width as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
        let latent_tokens = latent_height * latent_width;
        // `mu` reads the TARGET tokens only (`P:724`): references never move
        // the schedule.
        let (mut scheduler, schedule_warning) = scheduler_for(
            ScheduleKind::for_model(&req.model),
            req.steps as usize,
            latent_tokens,
        );
        if let Some(warning) = schedule_warning {
            progress.info(&warning);
        }
        let mut latents = match initial_latents {
            Some(latents) => {
                anyhow::ensure!(
                    latents.dims() == [1, latent_tokens, QWEN_IMAGE_21_LATENT_CHANNELS],
                    "Qwen Image 2.1 injected latents {:?} do not match the {latent_tokens}-token canvas",
                    latents.dims()
                );
                latents.to_device(device)?.to_dtype(dtype)?
            }
            None => {
                let noise = seeded_randn(
                    seed,
                    &[1, latent_tokens, QWEN_IMAGE_21_LATENT_CHANNELS],
                    device,
                    dtype,
                )?;
                (noise * scheduler.initial_sigma())?
            }
        };

        let total = scheduler.num_steps();
        let label = format!("Denoising ({total} steps)");
        progress.stage_start(&label);
        let denoise_start = Instant::now();
        let branches: Vec<&QwenImage21TextConditioning> = std::iter::once(conditioning)
            .chain(negative_conditioning)
            .collect();
        // Each CFG branch has its own layout (a different text length), and
        // both share the condition latents (`P:689-696`).
        let layouts = branches
            .iter()
            .map(|branch| match &condition {
                Some(condition) => QwenImage21JointLayout::build(
                    &branch.image_slots[0],
                    &branch.valid_tokens,
                    &condition.shapes,
                    (latent_height, latent_width),
                ),
                None => QwenImage21JointLayout::text_to_image(
                    &branch.valid_tokens,
                    (latent_height, latent_width),
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        let prefixes: Vec<usize> = layouts.iter().map(|layout| layout.prefix_len()).collect();
        let decisions = super::PrefixCachePolicy::resolve_from_env(
            &prefixes,
            condition.is_some(),
            conditioning.batch_size(),
            dtype,
        );
        if decisions.contains(&super::PrefixCacheDecision::Recompute) {
            progress.info(
                "Qwen Image 2.1 recomputes its prompt prefix every step (prefix KV cache off or over budget).",
            );
        }
        let mut prepared = branches
            .into_iter()
            .zip(layouts)
            .zip(&decisions)
            .map(|((branch, layout), decision)| {
                transformer.prepare(
                    branch,
                    layout,
                    condition.as_ref().map(|c| c.latents.clone()),
                    *decision,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for step in 0..total {
            progress.checkpoint()?;
            let step_start = Instant::now();
            let timestep = step_timestep(
                scheduler.current_timestep(),
                scheduler.current_sigma(),
                dtype,
                round_timestep_to_dtype,
            );
            let conditional_prediction = prepared[0].forward(&latents, timestep)?;
            let prediction = if let Some(negative) = prepared.get_mut(1) {
                progress.checkpoint()?;
                let negative_prediction = negative.forward(&latents, timestep)?;
                (&negative_prediction
                    + ((&conditional_prediction - &negative_prediction)? * req.guidance)?)?
            } else {
                conditional_prediction
            };
            transformer.ensure_finite(&prediction, step)?;
            latents = scheduler.step(&prediction, &latents)?;
            progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total,
                elapsed: step_start.elapsed(),
            });
        }
        progress.checkpoint()?;
        progress.stage_done(&label, denoise_start.elapsed());
        Ok((latents, latent_height, latent_width))
    }

    fn decode_rgba(
        progress: &ProgressReporter,
        vae: &QwenImage21Vae,
        latents: &Tensor,
        latent_height: usize,
        latent_width: usize,
        vae_device: &Device,
        vae_dtype: DType,
    ) -> Result<Tensor> {
        let label = "VAE decode";
        progress.stage_start(label);
        let start = Instant::now();
        let latents = latents.to_device(vae_device)?.to_dtype(vae_dtype)?;
        // The VAE is the family's only convolution, so the FastStill conv
        // scope wraps the decode alone (as FLUX does). The dispatch counter,
        // not the resolved policy, is the receipt of what actually ran.
        let cudnn_dispatches_before = crate::conv_policy::cudnn_dispatch_count();
        let decoded = {
            let _conv = crate::conv_policy::ConvScope::for_family("qwen-image21");
            let decoded = vae.decode_packed(&latents, latent_height, latent_width)?;
            vae_device.synchronize()?;
            decoded
        };
        crate::conv_policy::report_vae_decode_backend("qwen-image21", cudnn_dispatches_before);
        // The checkpoint decodes RGBA (`conv_out` has four channels, V:1137);
        // keep all four and let the output-alpha rule decide what is
        // published.
        let image = postprocess_image(&decoded)?.i(0)?;
        progress.phase_done(ProgressPhase::Vae, label, start.elapsed());
        Ok(image)
    }

    /// Encode the decoded `[4, H, W]` render. The alpha rule is
    /// `image::alpha_output_for_request` — keep alpha iff transparency was
    /// requested or a reference carries it, otherwise drop it so a plain
    /// render's RGB bytes are exactly v0.32's — never `Infer`: an opaque
    /// render still decodes edge alpha below 255.
    fn response(
        req: &GenerateRequest,
        rgba: &Tensor,
        seed: u64,
        started: Instant,
    ) -> Result<GenerateResponse> {
        let format = req.resolved_output_format();
        let alpha = alpha_output_for_request(req);
        let output_metadata = build_output_metadata(req, seed, None);
        let data = encode_image_with_alpha(
            rgba,
            format,
            req.width,
            req.height,
            output_metadata.as_ref(),
            alpha,
        )?;
        Ok(GenerateResponse {
            mesh: None,
            request_warnings: alpha_warning(alpha, format)
                .map(str::to_string)
                .into_iter()
                .collect(),
            audio: None,
            images: vec![ImageData {
                data,
                format,
                width: req.width,
                height: req.height,
                index: 0,
            }],
            generation_time_ms: started.elapsed().as_millis() as u64,
            model: req.model.clone(),
            seed_used: seed,
            video: None,
            gpu: None,
        })
    }

    fn generate_sequential(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let (transformer_paths, text_paths, tokenizer, vae_path) = self.validate_paths()?;
        let (device, text_device, vae_device) = self.resolve_devices()?;
        let dtype = super::transformer_dtype(&device);
        let text_dtype = crate::engine::gpu_dtype(&text_device);
        let vae_dtype = crate::engine::gpu_dtype(&vae_device);
        let started = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let progress = &self.base.progress;

        progress.info("Using sequential Qwen Image 2.1 loading (Qwen3-VL -> transformer -> VAE)");
        let references = Self::prepare_references(progress, req)?;

        // Phase 1: the Qwen3-VL text encoder (and, with references, its vision
        // tower) encode both branches, then leave the card.
        let text_label = format!(
            "Loading Qwen3-VL text encoder ({} shards, {})",
            text_paths.len(),
            device_label(&text_device)
        );
        progress.stage_start(&text_label);
        let text_start = Instant::now();
        let mut text_encoder = Self::load_text_encoder(
            &text_paths,
            &tokenizer,
            &text_device,
            text_dtype,
            Self::free_vram_for(&text_device, self.base.gpu_ordinal),
            progress,
        )?;
        progress.stage_done(&text_label, text_start.elapsed());
        let vision = if references.is_empty() {
            None
        } else {
            let tower = Self::load_vision(progress, &text_paths, &text_device, text_dtype)?;
            let features = Self::run_vision(progress, &tower, &references, &text_device)?;
            drop(tower);
            Some(features)
        };
        let (conditioning, negative_conditioning) = Self::encode_conditioning(
            progress,
            &mut text_encoder,
            req,
            vision.as_ref(),
            (&device, dtype),
        )?;
        drop(vision);
        drop(text_encoder);
        text_device.synchronize()?;

        // Phase 2: the VAE encoder turns the references into condition blocks.
        let condition = if references.is_empty() {
            None
        } else {
            let encoder = Self::load_vae_encoder(progress, &vae_path, &vae_device, vae_dtype)?;
            let blocks = Self::encode_references(
                progress,
                &encoder,
                &references,
                (&vae_device, vae_dtype),
                (&device, dtype),
            )?;
            drop(encoder);
            vae_device.synchronize()?;
            Some(blocks)
        };

        // Phase 3: denoise.
        let transformer_label = format!(
            "Loading Qwen Image 2.1 transformer ({} shards)",
            transformer_paths.len()
        );
        progress.stage_start(&transformer_label);
        let transformer_start = Instant::now();
        let transformer =
            QwenImage21Transformer::load(&transformer_paths, &device, dtype, progress)?;
        progress.stage_done(&transformer_label, transformer_start.elapsed());
        let initial_latents = self.take_initial_latents();
        let progress = &self.base.progress;
        let (latents, latent_height, latent_width) = Self::denoise(
            progress,
            req,
            &transformer,
            &conditioning,
            negative_conditioning.as_ref(),
            (&device, dtype),
            DenoiseStart {
                seed,
                initial_latents,
                round_timestep_to_dtype: ROUND_TIMESTEP_TO_DTYPE,
                condition,
            },
        )?;
        drop(transformer);
        drop(conditioning);
        drop(negative_conditioning);
        device.synchronize()?;

        // Phase 4: decode.
        let vae_label = format!("Loading Qwen Image 2.1 VAE ({})", device_label(&vae_device));
        progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = QwenImage21Vae::load(&vae_path, &vae_device, vae_dtype, progress)?;
        progress.stage_done(&vae_label, vae_start.elapsed());
        let image = Self::decode_rgba(
            progress,
            &vae,
            &latents,
            latent_height,
            latent_width,
            &vae_device,
            vae_dtype,
        )?;
        Self::response(req, &image, seed, started)
    }

    fn generate_eager(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if self.base.loaded.is_none() {
            self.load()?;
        }
        let started = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        let initial_latents = self.take_initial_latents();
        let progress = &self.base.progress;
        let references = Self::prepare_references(progress, req)?;
        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen Image 2.1 was not loaded"))?;
        // A previous request may have parked or dropped the encoder; this
        // restores it (a host→device copy, or a reload) and is a no-op when
        // it stayed resident.
        loaded.text_encoder.unpark_to_gpu(progress)?;
        if !references.is_empty() {
            if loaded.vision.is_none() {
                loaded.vision = Some(Self::load_vision(
                    progress,
                    &loaded.text_paths,
                    &loaded.text_device,
                    loaded.text_dtype,
                )?);
            }
            if loaded.vae_encoder.is_none() {
                loaded.vae_encoder = Some(Self::load_vae_encoder(
                    progress,
                    &loaded.vae_path,
                    &loaded.vae_device,
                    loaded.vae_dtype,
                )?);
            }
        }
        let vision = match &loaded.vision {
            Some(tower) if !references.is_empty() => Some(Self::run_vision(
                progress,
                tower,
                &references,
                &loaded.text_device,
            )?),
            _ => None,
        };
        let (conditioning, negative_conditioning) = Self::encode_conditioning(
            progress,
            &mut loaded.text_encoder,
            req,
            vision.as_ref(),
            (&loaded.device, loaded.dtype),
        )?;
        drop(vision);
        let condition = match &loaded.vae_encoder {
            Some(encoder) if !references.is_empty() => Some(Self::encode_references(
                progress,
                encoder,
                &references,
                (&loaded.vae_device, loaded.vae_dtype),
                (&loaded.device, loaded.dtype),
            )?),
            _ => None,
        };
        let residency = Self::settle_text_encoder_residency(
            progress,
            &self.base.paths,
            &mut loaded.text_encoder,
            req,
            self.base.gpu_ordinal,
            loaded.vae_dtype,
        )?;
        let (latents, latent_height, latent_width) = Self::denoise(
            progress,
            req,
            &loaded.transformer,
            &conditioning,
            negative_conditioning.as_ref(),
            (&loaded.device, loaded.dtype),
            DenoiseStart {
                seed,
                initial_latents,
                round_timestep_to_dtype: ROUND_TIMESTEP_TO_DTYPE,
                condition,
            },
        )?;
        // The decode's peak may not fit beside the transformer (2K): park it
        // to host RAM for the decode and restore it after, or release it and
        // let the next request reload — the decision's second half.
        use super::text_encoder_residency::TransformerDecode;
        match residency.transformer_decode {
            TransformerDecode::Resident => {}
            TransformerDecode::ParkHost => {
                progress.info(&format!(
                    "Parking Qwen Image 2.1 transformer: {}",
                    residency.reason
                ));
                loaded.transformer.move_to_device(&Device::Cpu)?;
                loaded.device.synchronize()?;
            }
            TransformerDecode::Drop => {
                progress.info(&format!(
                    "Releasing Qwen Image 2.1 transformer: {}",
                    residency.reason
                ));
            }
        }
        if residency.transformer_decode == TransformerDecode::Drop {
            // Nothing survives this request: the next one loads afresh.
            let loaded = self
                .base
                .loaded
                .take()
                .ok_or_else(|| anyhow::anyhow!("Qwen Image 2.1 was not loaded"))?;
            let LoadedQwenImage21 {
                transformer,
                vae,
                vae_device,
                vae_dtype,
                device,
                ..
            } = loaded;
            drop(transformer);
            device.synchronize()?;
            let image = Self::decode_rgba(
                &self.base.progress,
                &vae,
                &latents,
                latent_height,
                latent_width,
                &vae_device,
                vae_dtype,
            )?;
            return Self::response(req, &image, seed, started);
        }
        let decoded = Self::decode_rgba(
            progress,
            &loaded.vae,
            &latents,
            latent_height,
            latent_width,
            &loaded.vae_device,
            loaded.vae_dtype,
        );
        if residency.transformer_decode == TransformerDecode::ParkHost {
            // Restore even when the decode failed, so the engine stays usable.
            loaded.transformer.move_to_device(&loaded.device)?;
        }
        let image = decoded?;
        Self::response(req, &image, seed, started)
    }
}

impl InferenceEngine for QwenImage21Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        Self::validate_request(req)?;
        self.pending_placement = req.placement.clone();
        let result = if self.uses_sequential_generate_path() {
            self.generate_sequential(req)
        } else {
            self.generate_eager(req)
        };
        self.pending_placement = None;
        result
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        QwenImage21Engine::load(self)
    }

    fn load_for_request(&mut self, req: &GenerateRequest) -> Result<()> {
        self.pending_placement = req.placement.clone();
        let result = QwenImage21Engine::load(self);
        self.pending_placement = None;
        result
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
        crate::batch_execution_capability_for_family("qwen-image21")
            .expect("Qwen Image 2.1 batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }

    fn configured_load_strategy(&self) -> Option<LoadStrategy> {
        Some(self.base.load_strategy)
    }

    fn configured_block_offload(&self) -> Option<bool> {
        Some(false)
    }
}

fn device_label(device: &Device) -> &'static str {
    if device.is_cpu() {
        "CPU"
    } else {
        "GPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a red ceramic teapot",
            "model": "qwen-image-2.1:bf16",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 1.0
        }))
        .unwrap()
    }

    fn png(alpha: u8) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, alpha]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn text_to_image_drops_alpha_and_publishes_the_v032_rgb_bytes() {
        for format in [OutputFormat::Png, OutputFormat::Jpeg, OutputFormat::Webp] {
            let mut req = request();
            req.output_format = Some(format);
            assert_eq!(alpha_output_for_request(&req), AlphaOutput::Drop);
            assert!(alpha_warning(AlphaOutput::Drop, format).is_none());
        }
        // v0.32 wrote the decoded batch's first three channels as RGB.
        let rgba = Tensor::from_vec(
            vec![10u8, 200, 20, 100, 30, 50, 255, 128],
            (4, 1, 2),
            &Device::Cpu,
        )
        .unwrap();
        let legacy = crate::image::encode_image(
            &rgba.narrow(0, 0, 3).unwrap(),
            OutputFormat::Png,
            2,
            1,
            None,
        )
        .unwrap();
        let dropped =
            encode_image_with_alpha(&rgba, OutputFormat::Png, 2, 1, None, AlphaOutput::Drop)
                .unwrap();
        assert_eq!(dropped, legacy);
    }

    #[test]
    fn transparency_and_alpha_references_keep_alpha() {
        let mut req = request();
        req.transparent_background = Some(true);
        assert_eq!(alpha_output_for_request(&req), AlphaOutput::Keep);
        let mut req = request();
        req.edit_images = Some(vec![png(255), png(128)]);
        assert_eq!(alpha_output_for_request(&req), AlphaOutput::Keep);
        assert!(alpha_warning(AlphaOutput::Keep, OutputFormat::Jpeg).is_some());
        assert!(alpha_warning(AlphaOutput::Keep, OutputFormat::Png).is_none());
        let mut req = request();
        req.edit_images = Some(vec![png(255)]);
        assert_eq!(alpha_output_for_request(&req), AlphaOutput::Drop);
    }

    #[test]
    fn only_the_positive_prompt_takes_the_rgba_recipe() {
        let mut req = request();
        assert_eq!(positive_prompt(&req), "a red ceramic teapot");
        req.transparent_background = Some(true);
        assert_eq!(
            positive_prompt(&req),
            "This is an RGBA image with transparency. a red ceramic teapot. The image has alpha channel and the background is transparent."
        );
    }

    #[test]
    fn timestep_rounding_is_opt_in_and_default_keeps_v032() {
        const { assert!(!ROUND_TIMESTEP_TO_DTYPE) };
        assert_eq!(step_timestep(900.0, 0.9, DType::BF16, false), 0.9);
        assert_eq!(step_timestep(900.0, 0.9, DType::BF16, true), 0.8984375);
    }

    #[test]
    fn injected_latents_are_consumed_by_exactly_one_render() {
        let mut engine = QwenImage21Engine::new(
            "qwen-image-2.1:bf16".to_string(),
            ModelPaths {
                low_noise_transformer: None,
                low_noise_distilled_lora: None,
                transformer: PathBuf::from("/nonexistent/transformer"),
                transformer_shards: vec![],
                vae: PathBuf::from("/nonexistent/vae"),
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
            },
            LoadStrategy::Eager,
            0,
        );
        assert!(engine.take_initial_latents().is_none());
        engine.inject_initial_latents(Tensor::zeros((1, 4, 64), DType::F32, &Device::Cpu).unwrap());
        assert_eq!(engine.take_initial_latents().unwrap().dims(), &[1, 4, 64]);
        assert!(engine.take_initial_latents().is_none());
    }

    #[test]
    fn contract_accepts_native_canvas_references_and_webp() {
        QwenImage21Engine::validate_request(&request()).unwrap();
        let mut req = request();
        req.edit_images = Some(vec![png(255); 10]);
        req.output_format = Some(OutputFormat::Webp);
        req.transparent_background = Some(true);
        QwenImage21Engine::validate_request(&req).unwrap();
    }

    #[test]
    fn contract_rejects_non_native_canvas() {
        let mut req = request();
        req.width = 1008;
        assert!(QwenImage21Engine::validate_request(&req)
            .unwrap_err()
            .to_string()
            .contains("multiples of 32"));
    }

    #[test]
    fn contract_bounds_the_reference_count() {
        for count in [0, 11] {
            let mut req = request();
            req.edit_images = Some(vec![png(255); count]);
            assert!(QwenImage21Engine::validate_request(&req)
                .unwrap_err()
                .to_string()
                .contains("1 to 10 reference images"));
        }
    }

    #[test]
    fn contract_refuses_other_media() {
        let mut req = request();
        req.source_image = Some(png(255));
        assert!(QwenImage21Engine::validate_request(&req)
            .unwrap_err()
            .to_string()
            .contains("edit_images"));
        let mut req = request();
        req.mask_image = Some(png(255));
        assert!(QwenImage21Engine::validate_request(&req).is_err());
    }

    #[test]
    fn contract_refuses_transparent_jpeg() {
        let mut req = request();
        req.output_format = Some(OutputFormat::Jpeg);
        QwenImage21Engine::validate_request(&req).unwrap();
        req.transparent_background = Some(true);
        assert!(QwenImage21Engine::validate_request(&req)
            .unwrap_err()
            .to_string()
            .contains("JPEG"));
    }
}
