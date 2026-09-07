//! Hidden, scheduler-owned U²-Net worker used by durable mesh workflows.

use anyhow::{bail, Context, Result};
use image::ImageFormat;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, ModelPaths, OutputFormat};
use std::io::Cursor;
use std::time::Instant;

use crate::engine::{InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::progress::{InferenceCancellationToken, ProgressCallback};

pub struct MattingEngine {
    base: EngineBase<super::background_matting::U2Net>,
}

impl MattingEngine {
    pub fn new(model_name: String, paths: ModelPaths, strategy: LoadStrategy, gpu: usize) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, strategy, gpu),
        }
    }

    fn ensure_loaded(&mut self) -> Result<()> {
        if self.base.loaded.is_some() {
            return Ok(());
        }
        let path = &self.base.paths.transformer;
        if !path.exists() {
            bail!("matting graph not found: {}", path.display());
        }
        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        self.base
            .progress
            .stage_start("Loading U²-Net matting graph");
        self.base.loaded = Some(super::background_matting::U2Net::load(path, &device)?);
        Ok(())
    }
}

impl InferenceEngine for MattingEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let started = Instant::now();
        let bytes = req
            .source_image
            .as_deref()
            .context("matting requires source_image")?;
        if req
            .output_format
            .is_some_and(|format| format != OutputFormat::Png)
        {
            bail!("matting output must be PNG");
        }
        self.ensure_loaded()?;
        self.base.progress.checkpoint()?;
        self.base.progress.stage_start("Removing background");
        let source = image::load_from_memory(bytes)
            .context("decode matting source image")?
            .to_rgba8();
        let width = source.width();
        let height = source.height();
        let force = req.model == mold_core::manifest::HUNYUAN3D_MATTING_FORCE_MANIFEST;
        let matte = if !force && super::background_matting::has_useful_alpha(&source) {
            source
        } else {
            self.base.loaded.as_ref().unwrap().matte(&source)?
        };
        self.base.progress.checkpoint()?;
        let mut output = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(matte)
            .write_to(&mut output, ImageFormat::Png)
            .context("encode matted PNG")?;
        self.base
            .progress
            .stage_done("Removing background", started.elapsed());
        Ok(GenerateResponse {
            images: vec![ImageData {
                data: output.into_inner(),
                format: OutputFormat::Png,
                width,
                height,
                index: 0,
            }],
            mesh: None,
            audio: None,
            video: None,
            gpu: None,
            request_warnings: Vec::new(),
            generation_time_ms: started.elapsed().as_millis() as u64,
            model: req.model.clone(),
            seed_used: req.seed.unwrap_or_default(),
        })
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
    fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }
    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }
    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family(mold_core::manifest::HUNYUAN3D_MATTING_FAMILY)
            .expect("matting capability")
    }
    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }
}
