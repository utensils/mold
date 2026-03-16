use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, get_noise, postprocess_image, AutoEncoderKL, Config,
    FlowMatchEulerDiscreteScheduler, SchedulerConfig, TextEncoderConfig, VaeConfig,
    ZImageTextEncoder, ZImageTransformer2DModel,
};
use candle_transformers::quantized_var_builder;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths};
use std::time::Instant;

use super::quantized_transformer::QuantizedZImageTransformer2DModel;
use super::transformer::ZImageTransformer;
use crate::device::{fmt_gb, free_vram_bytes, should_use_gpu};
use crate::engine::{rand_seed, InferenceEngine};
use crate::image::encode_image;
use crate::progress::{ProgressCallback, ProgressEvent};

/// Z-Image scheduler shift constants (from reference implementation).
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Minimum free VRAM (bytes) required to place Z-Image VAE on GPU.
/// The VAE itself is small (~160MB), but decode at 1024x1024 needs ~6GB workspace
/// for conv2d im2col expansions through the upsampling blocks.
const VAE_DECODE_VRAM_THRESHOLD: u64 = 6_500_000_000;

/// Minimum free VRAM (bytes) required to place Qwen3 text encoder on GPU.
/// The encoder is ~8GB BF16. We also need headroom for denoising activations (~8GB)
/// and VAE decode workspace (~6GB) that share the same VRAM. On a 24GB card with Q8
/// (~17GB free), this threshold ensures the text encoder stays on CPU. On 48GB+ cards,
/// it loads on GPU for fast encoding (~0.1s vs ~14s on CPU).
const QWEN3_VRAM_THRESHOLD: u64 = 22_000_000_000;

/// Loaded Z-Image model components, ready for inference.
struct LoadedZImage {
    /// Transformer is wrapped in Option so it can be dropped to free VRAM for VAE decode,
    /// then reloaded from disk for the next generation (similar to FLUX's T5/CLIP offload).
    transformer: Option<ZImageTransformer>,
    text_encoder: ZImageTextEncoder,
    vae: AutoEncoderKL,
    tokenizer: tokenizers::Tokenizer,
    transformer_cfg: Config,
    /// GPU device for transformer + denoising
    device: Device,
    /// Device where the text encoder lives (may be CPU if VRAM is tight)
    text_encoder_device: Device,
    /// Device where the VAE lives (may be CPU if VRAM is extremely tight)
    vae_device: Device,
    dtype: DType,
    /// Whether the transformer is a GGUF quantized model (needed for reload).
    is_quantized: bool,
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

        let is_gguf = self
            .paths
            .transformer
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

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
        let transformer_cfg = Config::z_image_turbo();

        // Load transformer — GGUF quantized or BF16 safetensors
        let xformer_label = if is_gguf {
            "Loading Z-Image transformer (GPU, quantized)".to_string()
        } else {
            format!(
                "Loading Z-Image transformer ({} shards)",
                xformer_paths.len()
            )
        };
        self.stage_start(&xformer_label);
        let xformer_start = Instant::now();

        let transformer = if is_gguf {
            tracing::info!(
                path = %self.paths.transformer.display(),
                "loading quantized Z-Image transformer from GGUF..."
            );
            let vb =
                quantized_var_builder::VarBuilder::from_gguf(&self.paths.transformer, &device)?;
            ZImageTransformer::Quantized(QuantizedZImageTransformer2DModel::new(
                &transformer_cfg,
                dtype,
                vb,
            )?)
        } else {
            let xformer_path_strs: Vec<&str> = xformer_paths
                .iter()
                .map(|p| p.to_str().expect("non-UTF8 path"))
                .collect();
            let xformer_vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&xformer_path_strs, dtype, &device)? };
            ZImageTransformer::BF16(ZImageTransformer2DModel::new(&transformer_cfg, xformer_vb)?)
        };

        self.stage_done(&xformer_label, xformer_start.elapsed());
        tracing::info!(quantized = is_gguf, "Z-Image transformer loaded");

        // --- Decide where to place VAE and Qwen3 text encoder based on remaining VRAM ---
        let free = free_vram_bytes().unwrap_or(0);
        let is_cuda = device.is_cuda();
        let is_metal = device.is_metal();
        if free > 0 {
            self.info(&format!("Free VRAM after transformer: {}", fmt_gb(free)));
            tracing::info!(free_vram = free, "free VRAM after loading transformer");
        }

        // VAE decode at 1024x1024 needs ~6GB workspace for conv2d im2col.
        // On tight VRAM, load VAE on CPU to guarantee decode succeeds.
        let vae_on_gpu = should_use_gpu(is_cuda, is_metal, free, VAE_DECODE_VRAM_THRESHOLD);
        let vae_device = if vae_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let vae_dtype = if vae_on_gpu { dtype } else { DType::F32 };
        let vae_device_label = if vae_on_gpu { "GPU" } else { "CPU" };

        if !vae_on_gpu && is_cuda {
            self.info(&format!(
                "VAE on CPU ({} free < {} threshold for decode workspace)",
                fmt_gb(free),
                fmt_gb(VAE_DECODE_VRAM_THRESHOLD),
            ));
        }

        // Load VAE
        let vae_label = format!("Loading VAE ({})", vae_device_label);
        self.stage_start(&vae_label);
        let vae_start = Instant::now();
        let vae_cfg = VaeConfig::z_image();
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.paths.vae.to_str().expect("non-UTF8 path")],
                vae_dtype,
                &vae_device,
            )?
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vae_vb)?;
        self.stage_done(&vae_label, vae_start.elapsed());
        tracing::info!(device = vae_device_label, "Z-Image VAE loaded");

        // Text encoder placement: dynamic based on free VRAM.
        // The Qwen3 text encoder is ~8GB BF16. We need enough VRAM to also hold
        // denoising activations (~8GB) and VAE decode workspace (~6GB), so the threshold
        // is high (~22GB). On a 24GB card this means CPU; on 48GB+ cards it goes on GPU
        // for fast encoding (~0.1s vs ~14s on CPU). Metal always uses GPU (unified memory).
        let te_on_gpu = should_use_gpu(is_cuda, is_metal, free, QWEN3_VRAM_THRESHOLD);
        let te_device = if te_on_gpu {
            device.clone()
        } else {
            Device::Cpu
        };
        let te_dtype = if te_on_gpu { dtype } else { DType::F32 };
        let te_device_label = if te_on_gpu { "GPU" } else { "CPU" };

        if !te_on_gpu && is_cuda {
            self.info(&format!(
                "Qwen3 text encoder on CPU ({} free < {} needed for encoder + activations)",
                fmt_gb(free),
                fmt_gb(QWEN3_VRAM_THRESHOLD),
            ));
        }

        // Load text encoder (Qwen3)
        let te_label = format!(
            "Loading Qwen3 text encoder ({} shards, {})",
            self.paths.text_encoder_files.len(),
            te_device_label,
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
        let te_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&te_path_strs, te_dtype, &te_device)? };
        let text_encoder = ZImageTextEncoder::new(&te_cfg, te_vb)?;

        self.stage_done(&te_label, te_start.elapsed());
        tracing::info!(device = te_device_label, "Qwen3 text encoder loaded");

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(text_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;

        self.loaded = Some(LoadedZImage {
            transformer: Some(transformer),
            text_encoder,
            vae,
            tokenizer,
            transformer_cfg,
            device,
            text_encoder_device: te_device,
            vae_device,
            dtype,
            is_quantized: is_gguf,
        });

        tracing::info!(model = %self.model_name, "all Z-Image components loaded successfully");
        Ok(())
    }

    /// Reload the transformer from disk (called when it was dropped to free VRAM for VAE decode).
    fn reload_transformer(&self, loaded: &mut LoadedZImage) -> Result<()> {
        let xformer_paths = self.transformer_paths();
        if loaded.is_quantized {
            let vb = quantized_var_builder::VarBuilder::from_gguf(
                &self.paths.transformer,
                &loaded.device,
            )?;
            loaded.transformer = Some(ZImageTransformer::Quantized(
                QuantizedZImageTransformer2DModel::new(&loaded.transformer_cfg, loaded.dtype, vb)?,
            ));
        } else {
            let xformer_path_strs: Vec<&str> = xformer_paths
                .iter()
                .map(|p| p.to_str().expect("non-UTF8 path"))
                .collect();
            let xformer_vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &xformer_path_strs,
                    loaded.dtype,
                    &loaded.device,
                )?
            };
            loaded.transformer = Some(ZImageTransformer::BF16(ZImageTransformer2DModel::new(
                &loaded.transformer_cfg,
                xformer_vb,
            )?));
        }
        Ok(())
    }
}

impl InferenceEngine for ZImageEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        if self.loaded.is_none() {
            bail!("model not loaded — call load() first");
        }

        // Extract progress callback so we can emit events while loaded is mutably borrowed
        let on_progress = &self.on_progress;
        let emit = |event: ProgressEvent| {
            if let Some(cb) = on_progress {
                cb(event);
            }
        };
        let stage_start = |name: &str| {
            emit(ProgressEvent::StageStart {
                name: name.to_string(),
            });
        };
        let stage_done = |name: &str, elapsed: std::time::Duration| {
            emit(ProgressEvent::StageDone {
                name: name.to_string(),
                elapsed,
            });
        };

        let start = Instant::now();

        // Reload transformer if it was dropped (offloaded) after previous VAE decode
        let needs_reload = self.loaded.as_ref().unwrap().transformer.is_none();
        if needs_reload {
            {
                let mut loaded_mut = self.loaded.take().unwrap();
                let xformer_label = if loaded_mut.is_quantized {
                    "Reloading Z-Image transformer (GPU, quantized)"
                } else {
                    "Reloading Z-Image transformer (GPU, BF16)"
                };
                stage_start(xformer_label);
                let reload_start = Instant::now();
                self.reload_transformer(&mut loaded_mut)?;
                stage_done(xformer_label, reload_start.elapsed());
                self.loaded = Some(loaded_mut);
            }
        }

        let loaded = self.loaded.as_mut().unwrap();
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
        stage_start("Encoding prompt (Qwen3)");
        let encode_start = Instant::now();

        let formatted_prompt = format_prompt_for_qwen3(&req.prompt);
        let tokens = loaded
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        // Create input_ids on the text encoder's device (may be CPU)
        let input_ids = Tensor::from_vec(
            tokens.clone(),
            (1, tokens.len()),
            &loaded.text_encoder_device,
        )?;

        // 2. Encode text (on text encoder device, then move to GPU)
        let cap_feats = loaded.text_encoder.forward(&input_ids)?;
        let cap_feats = cap_feats
            .to_device(&loaded.device)?
            .to_dtype(loaded.dtype)?;
        let cap_mask = Tensor::ones((1, tokens.len()), DType::U8, &loaded.device)?;

        stage_done("Encoding prompt (Qwen3)", encode_start.elapsed());
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
        stage_start(&denoise_label);
        let denoise_start = Instant::now();

        // Scope the transformer borrow so it can be dropped before VAE decode
        {
            let transformer = loaded
                .transformer
                .as_ref()
                .expect("transformer must be loaded for denoising");

            for _step in 0..num_steps {
                let t = scheduler.current_timestep_normalized();
                let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &loaded.device)?
                    .to_dtype(loaded.dtype)?;

                // Forward pass through transformer
                let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

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
        }

        stage_done(&denoise_label, denoise_start.elapsed());
        tracing::info!("denoising complete");

        // Free text embeddings — no longer needed after denoising
        drop(cap_feats);
        drop(cap_mask);

        // Drop the transformer weights from GPU to free VRAM for VAE decode.
        // The transformer (~6.6GB for Q8) is only needed during denoising.
        // It will be reloaded from disk on the next generate() call.
        loaded.transformer = None;
        tracing::info!("Z-Image transformer dropped from GPU to free VRAM for VAE decode");

        // 8. VAE decode (may be on CPU if VRAM is tight)
        stage_start("VAE decode");
        let vae_start = Instant::now();

        // Remove frame dimension: (B, C, 1, H, W) → (B, C, H, W)
        // Move latents to VAE device (may be CPU)
        let latents = latents
            .squeeze(2)?
            .to_device(&loaded.vae_device)?
            .to_dtype(if loaded.vae_device.is_cpu() {
                DType::F32
            } else {
                loaded.dtype
            })?;
        let image = loaded.vae.decode(&latents)?;

        // Post-process: [-1, 1] → [0, 255] (candle z_image utility)
        let image = postprocess_image(&image)?;
        let image = image.i(0)?; // Remove batch dimension → [3, H, W]

        stage_done("VAE decode", vae_start.elapsed());

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
    use crate::device::should_use_gpu;

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

    // --- VRAM threshold decision tests ---

    #[test]
    fn qwen3_on_cpu_on_24gb_with_q8() {
        // Q8 transformer (6.6GB) on 24GB card → ~17GB free
        // Text encoder should go to CPU (17GB < 22GB threshold)
        assert!(!should_use_gpu(
            true,
            false,
            17_000_000_000,
            QWEN3_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_cpu_on_24gb_with_q4() {
        // Q4 transformer (3.9GB) on 24GB card → ~19GB free
        // Still on CPU (19GB < 22GB threshold) — need room for denoising + VAE
        assert!(!should_use_gpu(
            true,
            false,
            19_000_000_000,
            QWEN3_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_gpu_on_48gb_card() {
        // Q8 transformer (6.6GB) on 48GB card → ~40GB free
        // Text encoder should go to GPU for fast encoding
        assert!(should_use_gpu(
            true,
            false,
            40_000_000_000,
            QWEN3_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn qwen3_on_gpu_on_metal() {
        // Metal uses unified memory — always GPU regardless of free VRAM
        assert!(should_use_gpu(false, true, 0, QWEN3_VRAM_THRESHOLD));
    }

    #[test]
    fn vae_on_gpu_when_plenty_of_vram() {
        // Q8 transformer on 24GB, no other processes → ~17GB free
        assert!(should_use_gpu(
            true,
            false,
            17_000_000_000,
            VAE_DECODE_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn vae_on_cpu_when_vram_tight() {
        // Q8 transformer on 24GB + 12GB used by other process → ~5.4GB free
        assert!(!should_use_gpu(
            true,
            false,
            5_400_000_000,
            VAE_DECODE_VRAM_THRESHOLD
        ));
    }

    #[test]
    fn vae_on_gpu_on_metal() {
        assert!(should_use_gpu(false, true, 0, VAE_DECODE_VRAM_THRESHOLD));
    }

    // --- Threshold sanity checks ---

    #[test]
    fn qwen3_threshold_exceeds_encoder_size() {
        // Qwen3 text encoder is ~8GB; threshold must be much higher
        // to account for denoising activations + VAE workspace
        assert!(QWEN3_VRAM_THRESHOLD > 8_000_000_000);
    }

    #[test]
    fn qwen3_threshold_reasonable_for_large_cards() {
        // Should not be so high that even 48GB cards can't use GPU
        assert!(QWEN3_VRAM_THRESHOLD < 40_000_000_000);
    }

    #[test]
    fn vae_threshold_accounts_for_decode_workspace() {
        // VAE weights are ~160MB, but im2col workspace needs much more
        assert!(VAE_DECODE_VRAM_THRESHOLD > 160_000_000);
        assert!(VAE_DECODE_VRAM_THRESHOLD < 15_000_000_000);
    }
}
