//! Hunyuan3D's pinned SD2.1 InstructPix2Pix lighting-removal pre-stage.
//!
//! Tencent invokes diffusers 0.30.1 with an empty prompt, Euler ancestral,
//! seed 42, text guidance 1.0 and image guidance 1.5. Diffusers enables its
//! three-branch guidance only when text guidance is greater than one, so the
//! published call executes ONE conditional branch; image guidance is inert.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use candle_transformers::models::stable_diffusion::{
    schedulers::PredictionType, StableDiffusionConfig,
};
use image::RgbaImage;
use mold_core::{
    GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat, Scheduler,
};
use std::sync::Arc;
use std::time::Instant;

use crate::engine::{seeded_randn, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::image::{build_output_metadata, encode_image};
use crate::progress::{ProgressCallback, ProgressEvent};

const SIZE: u32 = 512;
const STEPS: u32 = 50;
const SEED: u64 = 42;
const VAE_SCALE: f64 = 0.18215;

struct LoadedDelight {
    unet: candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    clip: ClipTextTransformer,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: Device,
    dtype: DType,
    config: StableDiffusionConfig,
}

pub struct DelightEngine {
    base: EngineBase<LoadedDelight>,
}

impl DelightEngine {
    pub fn new(model_name: String, paths: ModelPaths, strategy: LoadStrategy, gpu: usize) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, strategy, gpu),
        }
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }
        for (label, path) in [
            ("UNet", &self.base.paths.transformer),
            ("VAE", &self.base.paths.vae),
            (
                "CLIP",
                self.base
                    .paths
                    .clip_encoder
                    .as_ref()
                    .context("delight CLIP path is missing")?,
            ),
            (
                "tokenizer",
                self.base
                    .paths
                    .clip_tokenizer
                    .as_ref()
                    .context("delight tokenizer path is missing")?,
            ),
        ] {
            if !path.exists() {
                bail!("delight {label} file not found: {}", path.display());
            }
        }
        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        let dtype = if crate::device::is_gpu(&device) {
            DType::F16
        } else {
            DType::F32
        };
        let config = StableDiffusionConfig::v2_1(None, Some(SIZE as usize), Some(SIZE as usize));
        self.base.progress.stage_start("Loading delight UNet");
        let unet = config.build_unet(&self.base.paths.transformer, &device, 8, false, dtype)?;
        let vae = config.build_vae(&self.base.paths.vae, &device, dtype)?;
        let clip = candle_transformers::models::stable_diffusion::build_clip_transformer(
            &config.clip,
            self.base.paths.clip_encoder.as_ref().unwrap(),
            &device,
            DType::F32,
        )?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_file(self.base.paths.clip_tokenizer.as_ref().unwrap())
                .map_err(|e| anyhow::anyhow!("load delight tokenizer: {e}"))?,
        );
        self.base.loaded = Some(LoadedDelight {
            unet,
            vae,
            clip,
            tokenizer,
            device,
            dtype,
            config,
        });
        Ok(())
    }

    fn tokens(
        tokenizer: &tokenizers::Tokenizer,
        max_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let encoding = tokenizer
            .encode("", true)
            .map_err(|e| anyhow::anyhow!("tokenize delight prompt: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(max_len);
        // Tencent's tokenizer_config pins `pad_token` to `!`, vocabulary id 0.
        ids.resize(max_len, 0);
        Ok(
            Tensor::new(ids.into_iter().map(i64::from).collect::<Vec<_>>(), device)?
                .unsqueeze(0)?,
        )
    }

    fn preprocess(bytes: &[u8]) -> Result<(RgbaImage, Tensor)> {
        let source = image::load_from_memory(bytes).context("decode delight source image")?;
        let resized = source
            // Pillow's `Image.resize((512, 512))` defaults to nearest-neighbour.
            .resize_exact(SIZE, SIZE, image::imageops::FilterType::Nearest)
            .to_rgba8();
        let mut rgba = resized;
        erode_alpha(&mut rgba);
        let mut rgb = Vec::with_capacity((SIZE * SIZE * 3) as usize);
        for pixel in rgba.pixels_mut() {
            if pixel[3] == 0 {
                pixel.0[..3].fill(255);
            }
            rgb.extend(pixel.0[..3].iter().map(|&v| f32::from(v) / 127.5 - 1.0));
        }
        let tensor = Tensor::from_vec(rgb, (SIZE as usize, SIZE as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        Ok((rgba, tensor))
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        req.source_image
            .as_deref()
            .context("delight requires source_image")?;
        if req.width != SIZE
            || req.height != SIZE
            || req.steps != STEPS
            || req.seed != Some(SEED)
            || req.scheduler.unwrap_or(Scheduler::EulerAncestral) != Scheduler::EulerAncestral
            || (req.guidance - 1.0).abs() > f64::EPSILON
        {
            bail!("delight uses the fixed 512x512, 50-step Euler ancestral, seed-42, guidance-1 recipe");
        }
        self.ensure_loaded()?;
        generate_loaded(self.base.loaded.as_ref().unwrap(), req, &self.base.progress)
    }
}

fn generate_loaded(
    loaded: &LoadedDelight,
    req: &GenerateRequest,
    progress: &crate::progress::ProgressReporter,
) -> Result<GenerateResponse> {
    let source = req
        .source_image
        .as_deref()
        .context("delight requires source_image")?;
    let started = Instant::now();
    let (target, source_tensor) = DelightEngine::preprocess(source)?;
    let tokens = DelightEngine::tokens(
        &loaded.tokenizer,
        loaded.config.clip.max_position_embeddings,
        &loaded.device,
    )?;
    let context = loaded.clip.forward(&tokens)?.to_dtype(loaded.dtype)?;
    // InstructPix2Pix concatenates the VAE posterior mode itself. Diffusers
    // applies the VAE scaling factor only to the denoised output latents
    // before decode; scaling this conditioning latent makes the source image
    // nearly invisible to the eight-channel UNet.
    let image_latents = prepare_image_latents(
        loaded
            .vae
            .encode(
                &source_tensor
                    .to_device(&loaded.device)?
                    .to_dtype(loaded.dtype)?,
            )?
            .mode()?,
    )?;
    let mut scheduler = crate::scheduler::build_scheduler(
        Scheduler::EulerAncestral,
        STEPS as usize,
        PredictionType::VPrediction,
        false,
    )?;
    let mut latents = (seeded_randn(SEED, &[1, 4, 64, 64], &loaded.device, DType::F32)?
        * scheduler.init_noise_sigma())?
    .to_dtype(loaded.dtype)?;
    for (index, &t) in scheduler.timesteps().to_vec().iter().enumerate() {
        progress.checkpoint()?;
        let scaled = scheduler.scale_model_input(latents.clone(), t)?;
        let input = Tensor::cat(&[&scaled, &image_latents], 1)?;
        let prediction = loaded.unet.forward(&input, t as f64, &context)?;
        // Euler ancestral injects fresh noise at every step. Reset the GPU RNG
        // from the request seed so a hot server repeats the fixed recipe.
        seed_ancestral_noise(&loaded.device, SEED.wrapping_add(index as u64 + 1))?;
        latents = scheduler.step(&prediction, t, &latents)?;
        progress.emit(ProgressEvent::DenoiseStep {
            step: index + 1,
            total: STEPS as usize,
            elapsed: started.elapsed(),
        });
    }
    let decoded = loaded.vae.decode(&(latents / VAE_SCALE)?)?;
    let decoded = ((((decoded / 2.)? + 0.5)?.clamp(0f32, 1f32)? * 255.)?)
        .to_dtype(DType::U8)?
        .squeeze(0)?;
    let corrected = recorrect_and_composite(&decoded, &target)?;
    let metadata = build_output_metadata(req, SEED, Some(Scheduler::EulerAncestral));
    let data = encode_image(&corrected, OutputFormat::Png, SIZE, SIZE, metadata.as_ref())?;
    Ok(GenerateResponse {
        images: vec![ImageData {
            data,
            format: OutputFormat::Png,
            width: SIZE,
            height: SIZE,
            index: 0,
        }],
        mesh: None,
        audio: None,
        video: None,
        gpu: None,
        request_warnings: Vec::new(),
        generation_time_ms: started.elapsed().as_millis() as u64,
        model: req.model.clone(),
        seed_used: SEED,
    })
}

fn prepare_image_latents(posterior_mode: Tensor) -> Result<Tensor> {
    Ok(posterior_mode)
}

fn seed_ancestral_noise(device: &Device, seed: u64) -> Result<()> {
    // Candle's CPU backend has no seedable generator. Delight is a GPU model;
    // retain the CPU fallback for tests and diagnostic execution.
    if !matches!(device, Device::Cpu) {
        device.set_seed(seed)?;
    }
    Ok(())
}

pub(crate) fn delight_rgba(
    paths: &ModelPaths,
    gpu_ordinal: usize,
    source: &RgbaImage,
    progress: &crate::progress::ProgressReporter,
) -> Result<RgbaImage> {
    let request = delight_request(source)?;
    let loaded = load_delight(paths, gpu_ordinal, progress)?;
    let response = generate_loaded(&loaded, &request, progress)?;
    let image = response
        .images
        .into_iter()
        .next()
        .context("delight returned no image")?;
    Ok(image::load_from_memory(&image.data)
        .context("decode delighted intermediate")?
        .to_rgba8())
}

fn delight_request(source: &RgbaImage) -> Result<GenerateRequest> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(source.clone())
        .write_to(&mut bytes, image::ImageFormat::Png)
        .context("encode matted input for delight")?;
    let mut request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": "",
        "model": mold_core::manifest::HUNYUAN3D_DELIGHT_MANIFEST,
        "width": SIZE,
        "height": SIZE,
        "steps": STEPS,
        "guidance": 1.0,
        "seed": SEED,
        "scheduler": "euler-ancestral",
        "output_format": "png"
    }))?;
    // The wire schema accepts base64 strings, not JSON arrays of byte values.
    // This internal PNG already has the typed representation we need.
    request.source_image = Some(bytes.into_inner());
    Ok(request)
}

fn load_delight(
    paths: &ModelPaths,
    gpu_ordinal: usize,
    progress: &crate::progress::ProgressReporter,
) -> Result<LoadedDelight> {
    for (label, path) in [
        ("UNet", &paths.transformer),
        ("VAE", &paths.vae),
        (
            "CLIP",
            paths
                .clip_encoder
                .as_ref()
                .context("delight CLIP path is missing")?,
        ),
        (
            "tokenizer",
            paths
                .clip_tokenizer
                .as_ref()
                .context("delight tokenizer path is missing")?,
        ),
    ] {
        if !path.exists() {
            bail!("delight {label} file not found: {}", path.display());
        }
    }
    let device = crate::device::create_device(gpu_ordinal, progress)?;
    let dtype = if crate::device::is_gpu(&device) {
        DType::F16
    } else {
        DType::F32
    };
    let config = StableDiffusionConfig::v2_1(None, Some(SIZE as usize), Some(SIZE as usize));
    progress.stage_start("Loading delight UNet");
    let unet = config.build_unet(&paths.transformer, &device, 8, false, dtype)?;
    let vae = config.build_vae(&paths.vae, &device, dtype)?;
    let clip = candle_transformers::models::stable_diffusion::build_clip_transformer(
        &config.clip,
        paths.clip_encoder.as_ref().unwrap(),
        &device,
        DType::F32,
    )?;
    let tokenizer = Arc::new(
        tokenizers::Tokenizer::from_file(paths.clip_tokenizer.as_ref().unwrap())
            .map_err(|e| anyhow::anyhow!("load delight tokenizer: {e}"))?,
    );
    Ok(LoadedDelight {
        unet,
        vae,
        clip,
        tokenizer,
        device,
        dtype,
        config,
    })
}

fn erode_alpha(image: &mut RgbaImage) {
    let source = image.clone();
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut alpha = 255;
            for oy in y.saturating_sub(1)..=(y + 1).min(SIZE - 1) {
                for ox in x.saturating_sub(1)..=(x + 1).min(SIZE - 1) {
                    alpha = alpha.min(source.get_pixel(ox, oy)[3]);
                }
            }
            image.get_pixel_mut(x, y)[3] = alpha;
        }
    }
}

fn recorrect_and_composite(decoded: &Tensor, target: &RgbaImage) -> Result<Tensor> {
    let planar = decoded.to_device(&Device::Cpu)?.to_vec3::<u8>()?;
    let count = (SIZE * SIZE) as usize;
    let mut source = vec![[0f32; 3]; count];
    for c in 0..3 {
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                source[y * SIZE as usize + x][c] = f32::from(planar[c][y][x]) / 255.;
            }
        }
    }
    let mask = target.pixels().map(|p| p[3] > 127).collect::<Vec<_>>();
    let n = mask.iter().filter(|&&v| v).count();
    let mut corrected = source.clone();
    if n > 1 {
        for c in 0..3 {
            let (sm, tm) =
                mask.iter()
                    .enumerate()
                    .filter(|(_, m)| **m)
                    .fold((0., 0.), |(s, t), (i, _)| {
                        (
                            s + source[i][c],
                            t + f32::from(target.as_raw()[i * 4 + c]) / 255.,
                        )
                    });
            let (sm, tm) = (sm / n as f32, tm / n as f32);
            let (sv, tv) =
                mask.iter()
                    .enumerate()
                    .filter(|(_, m)| **m)
                    .fold((0., 0.), |(s, t), (i, _)| {
                        let target_value = f32::from(target.as_raw()[i * 4 + c]) / 255.;
                        (
                            s + (source[i][c] - sm).powi(2),
                            t + (target_value - tm).powi(2),
                        )
                    });
            let ss = (sv / (n - 1) as f32).sqrt();
            let ts = (tv / (n - 1) as f32).sqrt();
            if ss > 1e-8 {
                for pixel in &mut corrected {
                    pixel[c] = ((pixel[c] - 0.95 * sm) * ts / ss + 0.95 * tm).clamp(0., 1.);
                }
            }
        }
        let mse = |pixels: &[[f32; 3]]| {
            pixels
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    (0..3)
                        .map(|c| {
                            let t = f32::from(target.as_raw()[i * 4 + c]) / 255.;
                            (p[c] - t).powi(2)
                        })
                        .sum::<f32>()
                })
                .sum::<f32>()
                / (count * 3) as f32
        };
        if mse(&source) < mse(&corrected) {
            corrected = source;
        }
    }
    let mut hwc = Vec::with_capacity(count * 3);
    for (i, pixel) in corrected.iter().enumerate() {
        let a = f32::from(target.as_raw()[i * 4 + 3]) / 255.;
        for &v in pixel {
            hwc.push(((v * a + 1. - a) * 255.) as u8);
        }
    }
    Ok(
        Tensor::from_vec(hwc, (SIZE as usize, SIZE as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?,
    )
}

impl InferenceEngine for DelightEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.generate_inner(req)
    }
    fn model_name(&self) -> &str {
        self.base.model_name()
    }
    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }
    fn load(&mut self) -> Result<()> {
        self.ensure_loaded()
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
        crate::batch_execution_capability_for_family(mold_core::manifest::HUNYUAN3D_DELIGHT_FAMILY)
            .expect("delight capability")
    }
    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delight_request_preserves_rgba_source_and_pinned_recipe() {
        let source = RgbaImage::from_fn(3, 2, |x, y| {
            image::Rgba([x as u8 * 70, y as u8 * 90, 123, (x + y) as u8 * 50])
        });
        let request = delight_request(&source).expect("build internal delight request");
        let bytes = request.source_image.as_ref().expect("source PNG");
        assert_eq!(image::load_from_memory(bytes).unwrap().to_rgba8(), source);
        assert_eq!(request.prompt, "");
        assert_eq!(
            request.model,
            mold_core::manifest::HUNYUAN3D_DELIGHT_MANIFEST
        );
        assert_eq!((request.width, request.height), (SIZE, SIZE));
        assert_eq!(request.steps, STEPS);
        assert_eq!(request.guidance, 1.0);
        assert_eq!(request.seed, Some(SEED));
        assert_eq!(request.scheduler, Some(Scheduler::EulerAncestral));
        assert_eq!(request.output_format, Some(OutputFormat::Png));
        let wire = serde_json::to_value(&request).unwrap();
        assert!(wire["source_image"].is_string());
        let restored: GenerateRequest = serde_json::from_value(wire).unwrap();
        assert_eq!(restored.source_image, request.source_image);
    }

    #[test]
    fn alpha_erosion_matches_a_three_by_three_min_filter() {
        let mut image = RgbaImage::from_pixel(SIZE, SIZE, image::Rgba([0, 0, 0, 255]));
        image.get_pixel_mut(4, 4)[3] = 0;
        erode_alpha(&mut image);
        for y in 3..=5 {
            for x in 3..=5 {
                assert_eq!(image.get_pixel(x, y)[3], 0);
            }
        }
        assert_eq!(image.get_pixel(2, 2)[3], 255);
    }

    #[test]
    fn instruct_pix2pix_conditioning_uses_the_unscaled_vae_mode() {
        let posterior_mode = Tensor::new(&[[[[1f32, -2.], [3., -4.]]]], &Device::Cpu).unwrap();
        let conditioning = prepare_image_latents(posterior_mode.clone()).unwrap();
        assert_eq!(
            conditioning
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            posterior_mode
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn ancestral_seed_hook_accepts_the_cpu_diagnostic_fallback() {
        let device = Device::Cpu;
        seed_ancestral_noise(&device, 42).unwrap();
    }
}
