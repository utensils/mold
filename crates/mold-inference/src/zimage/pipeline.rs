use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, get_noise, postprocess_image, AutoEncoderKL, Config,
    FlowMatchEulerDiscreteScheduler, SchedulerConfig, TextEncoderConfig, VaeConfig,
    ZImageTextEncoder, ZImageTransformer2DModel,
};
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use crate::engine::{rand_seed, InferenceEngine};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent};

/// Z-Image scheduler shift constants (from reference implementation).
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Loaded Z-Image model components, ready for inference.
struct LoadedZImage {
    transformer: ZImageTransformer2DModel,
    text_encoder: ZImageTextEncoder,
    vae: AutoEncoderKL,
    tokenizer: tokenizers::Tokenizer,
    transformer_cfg: Config,
    device: Device,
    dtype: DType,
}

/// Z-Image inference engine backed by candle's z_image module.
pub struct ZImageEngine {
    loaded: Option<LoadedZImage>,
    model_name: String,
    paths: ModelPaths,
    on_progress: Option<ProgressCallback>,
}

/// Format a user prompt for Qwen3 chat template.
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

impl ZImageEngine {
    pub fn new(model_name: String, paths: ModelPaths) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            on_progress: None,
        }
    }

    fn emit(&self, event: ProgressEvent) {
        if let Some(cb) = &self.on_progress {
            cb(event);
        }
    }

    fn stage_start(&self, name: &str) {
        self.emit(ProgressEvent::StageStart {
            name: name.to_string(),
        });
    }

    fn stage_done(&self, name: &str, elapsed: std::time::Duration) {
        self.emit(ProgressEvent::StageDone {
            name: name.to_string(),
            elapsed,
        });
    }

    fn info(&self, message: &str) {
        self.emit(ProgressEvent::Info {
            message: message.to_string(),
        });
    }

    /// Resolve transformer shard paths: use `transformer_shards` if non-empty,
    /// otherwise treat `transformer` as a single file.
    fn transformer_paths(&self) -> Vec<std::path::PathBuf> {
        if !self.paths.transformer_shards.is_empty() {
            self.paths.transformer_shards.clone()
        } else {
            vec![self.paths.transformer.clone()]
        }
    }

    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        tracing::info!(model = %self.model_name, "loading Z-Image model components...");

        // Check if this is a GGUF quantized model (not yet supported)
        let is_gguf = self
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        if is_gguf {
            bail!(
                "GGUF quantized Z-Image not yet supported — use :bf16 tag instead. \
                 Run: mold pull z-image-turbo:bf16"
            );
        }

        // Validate required paths
        if self.paths.text_encoder_files.is_empty() {
            bail!("text encoder paths required for Z-Image models");
        }
        let text_tokenizer_path =
            self.paths.text_tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("text tokenizer path required for Z-Image models")
            })?;

        let xformer_paths = self.transformer_paths();

        // Validate all files exist
        for path in &xformer_paths {
            if !path.exists() {
                bail!("transformer file not found: {}", path.display());
            }
        }
        if !self.paths.vae.exists() {
            bail!("VAE file not found: {}", self.paths.vae.display());
        }
        for path in &self.paths.text_encoder_files {
            if !path.exists() {
                bail!("text encoder file not found: {}", path.display());
            }
        }
        if !text_tokenizer_path.exists() {
            bail!(
                "text tokenizer file not found: {}",
                text_tokenizer_path.display()
            );
        }

        // Select device
        let device = if candle_core::utils::cuda_is_available() {
            self.info("CUDA detected, using GPU");
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            self.info("Metal detected, using GPU");
            Device::new_metal(0)?
        } else {
            self.info("No GPU detected, using CPU");
            Device::Cpu
        };

        let dtype = device.bf16_default_to_f32();

        // Load transformer
        let xformer_label = format!(
            "Loading Z-Image transformer ({} shards)",
            xformer_paths.len()
        );
        self.stage_start(&xformer_label);
        let xformer_start = Instant::now();

        let transformer_cfg = Config::z_image_turbo();
        let xformer_path_strs: Vec<&str> = xformer_paths
            .iter()
            .map(|p| p.to_str().expect("non-UTF8 path"))
            .collect();
        let xformer_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&xformer_path_strs, dtype, &device)? };
        let transformer = ZImageTransformer2DModel::new(&transformer_cfg, xformer_vb)?;

        self.stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!("Z-Image transformer loaded");

        // Load VAE
        self.stage_start("Loading VAE");
        let vae_start = Instant::now();
        let vae_cfg = VaeConfig::z_image();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.paths.vae.to_str().expect("non-UTF8 path")],
                dtype,
                &device,
            )?
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vae_vb)?;
        self.stage_done("Loading VAE", vae_start.elapsed());
        tracing::info!("Z-Image VAE loaded");

        // Load text encoder (Qwen3)
        let te_label = format!(
            "Loading Qwen3 text encoder ({} shards)",
            self.paths.text_encoder_files.len()
        );
        self.stage_start(&te_label);
        let te_start = Instant::now();

        let te_cfg = TextEncoderConfig::z_image();
        let te_path_strs: Vec<&str> = self
            .paths
            .text_encoder_files
            .iter()
            .map(|p| p.to_str().expect("non-UTF8 path"))
            .collect();
        let te_vb = unsafe { VarBuilder::from_mmaped_safetensors(&te_path_strs, dtype, &device)? };
        let text_encoder = ZImageTextEncoder::new(&te_cfg, te_vb)?;

        self.stage_done(&te_label, te_start.elapsed());
        tracing::info!("Qwen3 text encoder loaded");

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(text_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        self.loaded = Some(LoadedZImage {
            transformer,
            text_encoder,
            vae,
            tokenizer,
            transformer_cfg,
            device,
            dtype,
        });

        tracing::info!(model = %self.model_name, "all Z-Image components loaded successfully");
        Ok(())
    }
}

impl InferenceEngine for ZImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let loaded = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        loaded.device.set_seed(seed)?;

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed, width, height,
            steps = req.steps,
            "starting Z-Image generation"
        );

        // 1. Tokenize prompt with Qwen3 chat template
        self.stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();

        let formatted_prompt = format_prompt_for_qwen3(&req.prompt);
        let tokens = loaded
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let input_ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &loaded.device)?;

        // 2. Encode text
        let cap_feats = loaded.text_encoder.forward(&input_ids)?;
        let cap_mask = Tensor::ones((1, tokens.len()), DType::U8, &loaded.device)?;

        self.stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());
        tracing::info!(token_count = tokens.len(), "text encoding complete");

        // 3. Calculate latent dimensions: 2 * (image_size / 16)
        let vae_align = 16;
        let latent_h = 2 * (height / vae_align);
        let latent_w = 2 * (width / vae_align);

        // 4. Calculate scheduler shift
        let patch_size = loaded.transformer_cfg.all_patch_size[0];
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );

        // 5. Initialize scheduler
        let scheduler_cfg = SchedulerConfig::z_image_turbo();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);
        scheduler.set_timesteps(req.steps as usize, Some(mu));

        // 6. Generate initial noise: (B, 16, latent_h, latent_w) → add frame dim → (B, 16, 1, latent_h, latent_w)
        let mut latents =
            get_noise(1, 16, latent_h, latent_w, &loaded.device)?.to_dtype(loaded.dtype)?;
        latents = latents.unsqueeze(2)?;

        // 7. Denoising loop
        let num_steps = req.steps as usize;
        let denoise_label = format!("Denoising ({} steps)", num_steps);
        self.stage_start(&denoise_label);
        let denoise_start = Instant::now();

        for _step in 0..num_steps {
            let t = scheduler.current_timestep_normalized();
            let t_tensor =
                Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?.to_dtype(loaded.dtype)?;

            // Forward pass through transformer
            let noise_pred = loaded
                .transformer
                .forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

            // Negate prediction (Z-Image specific)
            let noise_pred = noise_pred.neg()?;

            // Remove frame dimension for scheduler: (B, C, 1, H, W) → (B, C, H, W)
            let noise_pred_4d = noise_pred.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;

            // Scheduler step
            let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;

            // Add back frame dimension
            latents = prev_latents.unsqueeze(2)?;
        }

        self.stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete");

        // 8. VAE decode
        self.stage_start("VAE decode");
        let vae_start = Instant::now();

        // Remove frame dimension: (B, C, 1, H, W) → (B, C, H, W)
        let latents = latents.squeeze(2)?;
        let image = loaded.vae.decode(&latents)?;

        // Post-process: [-1, 1] → [0, 255] (candle z_image utility)
        let image = postprocess_image(&image)?;
        let image = image.i(0)?; // Remove batch dimension → [3, H, W]

        self.stage_done("VAE decode", vae_start.elapsed());

        // 9. Encode to output format
        let image_bytes = encode_image(&image, req.output_format, req.width, req.height)?;

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "Z-Image generation complete");

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
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn load(&mut self) -> Result<()> {
        ZImageEngine::load(self)
    }

    fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.on_progress = Some(callback);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_formatting() {
        let result = format_prompt_for_qwen3("a cat");
        assert_eq!(
            result,
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn latent_dimensions() {
        // 1024px → 2 * (1024 / 16) = 128
        assert_eq!(2 * (1024 / 16), 128);
        // 512px → 2 * (512 / 16) = 64
        assert_eq!(2 * (512 / 16), 64);
        // 768px → 2 * (768 / 16) = 96
        assert_eq!(2 * (768 / 16), 96);
    }
}
