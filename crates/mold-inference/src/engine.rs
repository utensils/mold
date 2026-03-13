use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, flux, t5};
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn model_name(&self) -> &str;
    fn is_loaded(&self) -> bool;
}

/// Loaded FLUX model components, ready for inference.
/// T5 and CLIP run on CPU to save VRAM; FLUX transformer and VAE run on GPU.
struct LoadedFlux {
    flux_model: flux::model::Flux,
    t5_model: t5::T5EncoderModel,
    t5_tokenizer: Tokenizer,
    clip_model: clip::text_model::ClipTextTransformer,
    clip_tokenizer: Tokenizer,
    vae: flux::autoencoder::AutoEncoder,
    /// CPU device for text encoders (T5 + CLIP)
    cpu: Device,
    /// GPU device for FLUX transformer + VAE
    device: Device,
    dtype: DType,
    is_schnell: bool,
}

/// FLUX inference engine backed by candle.
pub struct FluxEngine {
    loaded: Option<LoadedFlux>,
    model_name: String,
    paths: ModelPaths,
    t5_tokenizer_path: PathBuf,
    clip_tokenizer_path: PathBuf,
}

impl FluxEngine {
    /// Create a new FluxEngine. Does not load models until `load()` is called.
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        t5_tokenizer_path: PathBuf,
        clip_tokenizer_path: PathBuf,
    ) -> Self {
        Self {
            loaded: None,
            model_name,
            paths,
            t5_tokenizer_path,
            clip_tokenizer_path,
        }
    }

    /// Load all model components into GPU memory.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.is_some() {
            return Ok(());
        }

        let is_schnell = self.model_name.contains("schnell");
        tracing::info!(model = %self.model_name, "loading FLUX model components...");

        let cpu = Device::Cpu;
        let device = Device::cuda_if_available(0)?;
        let gpu_dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        tracing::info!("GPU device: {:?}, GPU dtype: {:?}", device, gpu_dtype);
        tracing::info!("T5/CLIP will load on CPU (F32) to save VRAM");

        // Load T5 encoder on CPU (9.2GB in F32, too large for GPU alongside FLUX)
        tracing::info!(path = %self.paths.t5_encoder.display(), "loading T5 encoder on CPU...");
        let t5_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.t5_encoder),
                DType::F32,
                &cpu,
            )?
        };
        let t5_config_text = r#"{
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
            "initializer_factor": 1.0,
            "is_encoder_decoder": true,
            "is_gated_act": true,
            "model_type": "t5",
            "num_decoder_layers": 24,
            "num_heads": 64,
            "num_layers": 24,
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "vocab_size": 32128,
            "dense_act_fn": "gelu_new"
        }"#;
        let t5_config: t5::Config = serde_json::from_str(t5_config_text)?;
        let t5_model = t5::T5EncoderModel::load(t5_vb, &t5_config)?;
        tracing::info!("T5 encoder loaded on CPU");

        // Load T5 tokenizer
        let t5_tokenizer = Tokenizer::from_file(&self.t5_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load T5 tokenizer: {e}"))?;

        // Load CLIP encoder on CPU (small, but keeps all text encoding on CPU)
        tracing::info!(path = %self.paths.clip_encoder.display(), "loading CLIP encoder on CPU...");
        let clip_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.clip_encoder),
                DType::F32,
                &cpu,
            )?
        };
        let clip_config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let clip_model =
            clip::text_model::ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config)?;
        tracing::info!("CLIP encoder loaded on CPU");

        // Load CLIP tokenizer
        let clip_tokenizer = Tokenizer::from_file(&self.clip_tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP tokenizer: {e}"))?;

        // Load FLUX transformer on GPU (23GB BF16 — fits in 24GB VRAM)
        tracing::info!(path = %self.paths.transformer.display(), "loading FLUX transformer on GPU...");
        let flux_cfg = if is_schnell {
            flux::model::Config::schnell()
        } else {
            flux::model::Config::dev()
        };
        let flux_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.transformer),
                gpu_dtype,
                &device,
            )?
        };
        let flux_model = flux::model::Flux::new(&flux_cfg, flux_vb)?;
        tracing::info!("FLUX transformer loaded on GPU");

        // Load VAE on GPU (small, ~300MB)
        tracing::info!(path = %self.paths.vae.display(), "loading VAE on GPU...");
        let vae_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.paths.vae),
                gpu_dtype,
                &device,
            )?
        };
        let vae_cfg = if is_schnell {
            flux::autoencoder::Config::schnell()
        } else {
            flux::autoencoder::Config::dev()
        };
        let vae = flux::autoencoder::AutoEncoder::new(&vae_cfg, vae_vb)?;
        tracing::info!("VAE loaded on GPU");

        self.loaded = Some(LoadedFlux {
            flux_model,
            t5_model,
            t5_tokenizer,
            clip_model,
            clip_tokenizer,
            vae,
            cpu,
            device,
            dtype: gpu_dtype,
            is_schnell,
        });

        tracing::info!(model = %self.model_name, "all model components loaded successfully");
        Ok(())
    }
}

impl InferenceEngine for FluxEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("model not loaded — call load() first"))?;

        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);
        loaded.device.set_seed(seed)?;

        let width = req.width as usize;
        let height = req.height as usize;

        tracing::info!(
            prompt = %req.prompt,
            seed,
            width,
            height,
            steps = req.steps,
            "starting generation"
        );

        // 1. Encode prompt with T5 (on CPU, then move to GPU)
        let t5_emb = {
            let mut tokens = loaded
                .t5_tokenizer
                .encode(req.prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {e}"))?
                .get_ids()
                .to_vec();
            tokens.resize(256, 0);
            let input_ids = Tensor::new(&tokens[..], &loaded.cpu)?.unsqueeze(0)?;
            let emb = loaded.t5_model.forward(&input_ids)?;
            // Move to GPU and cast to GPU dtype for FLUX transformer
            emb.to_device(&loaded.device)?.to_dtype(loaded.dtype)?
        };
        tracing::info!("T5 encoding complete (moved to GPU)");

        // 2. Encode prompt with CLIP (on CPU, then move to GPU)
        let clip_emb = {
            let tokens = loaded
                .clip_tokenizer
                .encode(req.prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
                .get_ids()
                .to_vec();
            let input_ids = Tensor::new(&tokens[..], &loaded.cpu)?.unsqueeze(0)?;
            let emb = loaded.clip_model.forward(&input_ids)?;
            // Move to GPU and cast to GPU dtype for FLUX transformer
            emb.to_device(&loaded.device)?.to_dtype(loaded.dtype)?
        };
        tracing::info!("CLIP encoding complete (moved to GPU)");

        // 3. Generate initial noise
        let img =
            flux::sampling::get_noise(1, height, width, &loaded.device)?.to_dtype(loaded.dtype)?;

        // 4. Build sampling state
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;

        // 5. Get timestep schedule
        let timesteps = if loaded.is_schnell {
            flux::sampling::get_schedule(req.steps as usize, None)
        } else {
            flux::sampling::get_schedule(req.steps as usize, Some((state.img.dim(1)?, 0.5, 1.15)))
        };

        tracing::info!(steps = timesteps.len(), "running denoising loop...");

        // 6. Denoise
        let img = flux::sampling::denoise(
            &loaded.flux_model,
            &state.img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &state.vec,
            &timesteps,
            4.0,
        )?;

        // 7. Unpack latent to spatial
        let img = flux::sampling::unpack(&img, height, width)?;
        tracing::info!("denoising complete, decoding VAE...");

        // 8. Decode with VAE
        let img = loaded.vae.decode(&img)?;

        // 9. Convert to u8 image: clamp to [-1, 1], map to [0, 255]
        let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img = img.i(0)?; // remove batch dim: [3, H, W]

        tracing::info!("VAE decode complete, encoding output image...");

        // 10. Convert candle tensor to image bytes
        let (c, h, w) = img.dims3()?;
        if c != 3 {
            bail!("expected 3 channels, got {c}");
        }
        let img_data = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        let rgb_image = image::RgbImage::from_raw(w as u32, h as u32, img_data)
            .ok_or_else(|| anyhow::anyhow!("failed to create image from tensor data"))?;

        let mut buf = std::io::Cursor::new(Vec::new());
        match req.output_format {
            OutputFormat::Png => rgb_image.write_to(&mut buf, image::ImageFormat::Png)?,
            OutputFormat::Jpeg => rgb_image.write_to(&mut buf, image::ImageFormat::Jpeg)?,
        }

        let generation_time_ms = start.elapsed().as_millis() as u64;
        tracing::info!(generation_time_ms, seed, "generation complete");

        Ok(GenerateResponse {
            images: vec![ImageData {
                data: buf.into_inner(),
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
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
