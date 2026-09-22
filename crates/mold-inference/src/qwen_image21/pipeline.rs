//! Qwen Image 2.1 inference engine wiring.
//!
//! The component implementations in this directory deliberately preserve the
//! checkpoint's native layout: a Qwen3-VL language submodel conditions a
//! 32-block causal-condition transformer over unpatched 64-channel latents,
//! followed by the 2.1 decoder. This engine keeps that sequence intact for
//! both eager and sequential residency modes.

use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::z_image::postprocess_image;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat};
use std::path::PathBuf;
use std::time::Instant;

use super::scheduler::{QwenImage21Scheduler, QwenShiftPolicy};
use super::transformer::QwenImage21Transformer;
use super::vae::QwenImage21Vae;
use super::{
    encode_t2i_prompts, QwenImage21TextConditioning, QWEN_IMAGE_21_CANVAS_ALIGNMENT,
    QWEN_IMAGE_21_LATENT_CHANNELS, QWEN_IMAGE_21_VAE_SCALE_FACTOR,
};
use crate::device::{effective_device_ref, resolve_device};
use crate::encoders::qwen3::Qwen3Encoder;
use crate::encoders::qwen3_bf16::Qwen3BF16Config;
use crate::engine::{cfg_active, rand_seed, seeded_randn, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressEvent, ProgressPhase, ProgressReporter};

/// Components kept resident by an eager Qwen Image 2.1 engine.
struct LoadedQwenImage21 {
    transformer: QwenImage21Transformer,
    text_encoder: Qwen3Encoder,
    vae: QwenImage21Vae,
    device: Device,
    vae_device: Device,
    dtype: DType,
    vae_dtype: DType,
}

/// Native inference engine for `Qwen/Qwen-Image-2.1`.
///
/// Text-to-image is the complete first runtime path. The upstream checkpoint
/// can also consume image slots through Qwen3-VL's vision tower, but carrying
/// a source image into this engine without that tower would be a wrong render,
/// so those requests are rejected before weights are loaded.
pub struct QwenImage21Engine {
    base: EngineBase<LoadedQwenImage21>,
    /// Placement is request-scoped because it affects component construction.
    pending_placement: Option<mold_core::types::DevicePlacement>,
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

    fn load_text_encoder(
        paths: &[PathBuf],
        tokenizer: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &ProgressReporter,
    ) -> Result<Qwen3Encoder> {
        Qwen3Encoder::load_bf16(
            paths,
            tokenizer,
            device,
            dtype,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
            progress,
        )
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
            &self.base.progress,
        )?;
        self.base
            .progress
            .stage_done(&text_label, text_start.elapsed());

        self.base.loaded = Some(LoadedQwenImage21 {
            transformer,
            text_encoder,
            vae,
            device,
            vae_device,
            dtype,
            vae_dtype,
        });
        Ok(())
    }

    /// Keep direct callers honest too; the server validates the same contracts
    /// earlier, but an inference engine must never silently omit input media.
    fn validate_text_to_image_request(req: &GenerateRequest) -> Result<()> {
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
        anyhow::ensure!(
            !req.has_durable_media_inputs()
                && req.control_model.is_none()
                && req.mesh.is_none(),
            "Qwen Image 2.1 native support is text-to-image only; reference, source, edit, control, and mesh inputs are not implemented"
        );
        anyhow::ensure!(
            req.caller_lora_stack().is_empty(),
            "Qwen Image 2.1 LoRA adapters are not implemented"
        );
        anyhow::ensure!(
            matches!(
                req.resolved_output_format(),
                OutputFormat::Png | OutputFormat::Jpeg
            ),
            "Qwen Image 2.1 supports PNG and JPEG output"
        );
        Ok(())
    }

    fn encode_conditioning(
        progress: &ProgressReporter,
        text_encoder: &mut Qwen3Encoder,
        req: &GenerateRequest,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(
        QwenImage21TextConditioning,
        Option<QwenImage21TextConditioning>,
    )> {
        let label = "Encoding prompt (Qwen3-VL)";
        progress.stage_start(label);
        let start = Instant::now();
        let conditional = encode_t2i_prompts(text_encoder, std::slice::from_ref(&req.prompt))?
            .to_device_dtype(target_device, target_dtype)?;

        let unconditional = if cfg_active(req.guidance) {
            match req.negative_prompt.as_ref() {
                Some(negative_prompt) => Some(
                    encode_t2i_prompts(text_encoder, std::slice::from_ref(negative_prompt))?
                        .to_device_dtype(target_device, target_dtype)?,
                ),
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
        seed: u64,
    ) -> Result<(Tensor, usize, usize)> {
        let (device, dtype) = compute;
        let latent_height = req.height as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
        let latent_width = req.width as usize / QWEN_IMAGE_21_VAE_SCALE_FACTOR;
        let latent_tokens = latent_height * latent_width;
        let mut scheduler = QwenImage21Scheduler::new(
            req.steps as usize,
            latent_tokens,
            QwenShiftPolicy::DynamicResolution,
        );
        let noise = seeded_randn(
            seed,
            &[1, latent_tokens, QWEN_IMAGE_21_LATENT_CHANNELS],
            device,
            dtype,
        )?;
        let mut latents = (noise * scheduler.initial_sigma())?;

        let total = scheduler.num_steps();
        let label = format!("Denoising ({total} steps)");
        progress.stage_start(&label);
        let denoise_start = Instant::now();
        let mut conditional = transformer.prepare_t2i(conditioning, latent_height, latent_width);
        let mut negative = negative_conditioning
            .map(|conditioning| transformer.prepare_t2i(conditioning, latent_height, latent_width));
        if conditioning.sequence_length() > super::PREFIX_CACHE_MAX_TOKENS
            || negative_conditioning
                .is_some_and(|c| c.sequence_length() > super::PREFIX_CACHE_MAX_TOKENS)
        {
            progress.info(
                "Long text prefixes render in full without KV caching (512-token retention limit).",
            );
        }
        for step in 0..total {
            progress.checkpoint()?;
            let step_start = Instant::now();
            // The diffusion transformer takes normalized `[0, 1]` time;
            // the packaged scheduler exposes the usual `[0, 1000]` values.
            let timestep = scheduler.current_timestep() / 1000.0;
            let conditional_prediction = conditional.forward(&latents, timestep)?;
            let prediction = if let Some(negative) = &mut negative {
                progress.checkpoint()?;
                let negative_prediction = negative.forward(&latents, timestep)?;
                (&negative_prediction
                    + ((&conditional_prediction - &negative_prediction)? * req.guidance)?)?
            } else {
                conditional_prediction
            };
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

    fn decode_rgb(
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
        let decoded = vae.decode_packed(&latents, latent_height, latent_width)?;
        // The checkpoint decodes RGBA. Mold's image artifact contract is RGB,
        // so preserve the trained RGB channels and do not pretend to publish a
        // separate alpha-capable format.
        let image = postprocess_image(&decoded)?.narrow(1, 0, 3)?.i(0)?;
        progress.phase_done(ProgressPhase::Vae, label, start.elapsed());
        Ok(image)
    }

    fn response(
        req: &GenerateRequest,
        image: &Tensor,
        seed: u64,
        started: Instant,
    ) -> Result<GenerateResponse> {
        let output_metadata = build_output_metadata(req, seed, None);
        let data = encode_image(
            image,
            req.resolved_output_format(),
            req.width,
            req.height,
            output_metadata.as_ref(),
        )?;
        Ok(GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![ImageData {
                data,
                format: req.resolved_output_format(),
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

        self.base
            .progress
            .info("Using sequential Qwen Image 2.1 loading (Qwen3-VL -> transformer -> VAE)");

        let text_label = format!(
            "Loading Qwen3-VL text encoder ({} shards, {})",
            text_paths.len(),
            device_label(&text_device)
        );
        self.base.progress.stage_start(&text_label);
        let text_start = Instant::now();
        let mut text_encoder = Self::load_text_encoder(
            &text_paths,
            &tokenizer,
            &text_device,
            text_dtype,
            &self.base.progress,
        )?;
        self.base
            .progress
            .stage_done(&text_label, text_start.elapsed());
        let (conditioning, negative_conditioning) =
            Self::encode_conditioning(&self.base.progress, &mut text_encoder, req, &device, dtype)?;
        drop(text_encoder);
        text_device.synchronize()?;

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
        let (latents, latent_height, latent_width) = Self::denoise(
            &self.base.progress,
            req,
            &transformer,
            &conditioning,
            negative_conditioning.as_ref(),
            (&device, dtype),
            seed,
        )?;
        drop(transformer);
        drop(conditioning);
        drop(negative_conditioning);
        device.synchronize()?;

        let vae_label = format!("Loading Qwen Image 2.1 VAE ({})", device_label(&vae_device));
        self.base.progress.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae = QwenImage21Vae::load(&vae_path, &vae_device, vae_dtype, &self.base.progress)?;
        self.base
            .progress
            .stage_done(&vae_label, vae_start.elapsed());
        let image = Self::decode_rgb(
            &self.base.progress,
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
        let progress = &self.base.progress;
        let loaded = self
            .base
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen Image 2.1 was not loaded"))?;
        let (conditioning, negative_conditioning) = Self::encode_conditioning(
            progress,
            &mut loaded.text_encoder,
            req,
            &loaded.device,
            loaded.dtype,
        )?;
        let (latents, latent_height, latent_width) = Self::denoise(
            progress,
            req,
            &loaded.transformer,
            &conditioning,
            negative_conditioning.as_ref(),
            (&loaded.device, loaded.dtype),
            seed,
        )?;
        let image = Self::decode_rgb(
            progress,
            &loaded.vae,
            &latents,
            latent_height,
            latent_width,
            &loaded.vae_device,
            loaded.vae_dtype,
        )?;
        Self::response(req, &image, seed, started)
    }
}

impl InferenceEngine for QwenImage21Engine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        Self::validate_text_to_image_request(req)?;
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

    #[test]
    fn text_to_image_contract_accepts_native_canvas() {
        QwenImage21Engine::validate_text_to_image_request(&request()).unwrap();
    }

    #[test]
    fn text_to_image_contract_rejects_non_native_canvas() {
        let mut req = request();
        req.width = 1008;
        assert!(QwenImage21Engine::validate_text_to_image_request(&req)
            .unwrap_err()
            .to_string()
            .contains("multiples of 32"));
    }

    #[test]
    fn text_to_image_contract_refuses_reference_media() {
        let mut req = request();
        req.edit_images = Some(vec![vec![1, 2, 3]]);
        assert!(QwenImage21Engine::validate_text_to_image_request(&req)
            .unwrap_err()
            .to_string()
            .contains("text-to-image only"));
    }

    #[test]
    fn text_to_image_contract_refuses_unsupported_delivery() {
        let mut req = request();
        req.output_format = Some(OutputFormat::Webp);
        assert!(QwenImage21Engine::validate_text_to_image_request(&req)
            .unwrap_err()
            .to_string()
            .contains("PNG and JPEG"));
    }
}
