use anyhow::Result;
use mold_core::{GenerateRequest, GenerateResponse, ImageData, OutputFormat};
use std::collections::HashMap;
use std::time::Instant;

/// Trait for inference backends.
pub trait InferenceEngine: Send + Sync {
    fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse>;
    fn load_model(&mut self, model: &str) -> Result<()>;
    fn unload_model(&mut self, model: &str) -> Result<()>;
    fn loaded_models(&self) -> Vec<String>;
}

/// FLUX inference engine backed by candle.
///
/// Currently stubbed — returns placeholder images.
/// Real implementation will use candle-transformers FLUX pipeline.
pub struct FluxEngine {
    models_dir: std::path::PathBuf,
    loaded: HashMap<String, LoadedModel>,
}

struct LoadedModel {
    #[allow(dead_code)]
    name: String,
}

impl FluxEngine {
    pub fn new(models_dir: std::path::PathBuf) -> Self {
        Self {
            models_dir,
            loaded: HashMap::new(),
        }
    }
}

impl InferenceEngine for FluxEngine {
    fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse> {
        // Stub: generate a small placeholder image
        let start = Instant::now();
        let seed = req.seed.unwrap_or_else(rand_seed);

        let mut images = Vec::with_capacity(req.batch_size as usize);
        for i in 0..req.batch_size {
            // Create a simple gradient placeholder image
            let img = image::RgbImage::from_fn(req.width, req.height, |x, y| {
                image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
            });

            let mut buf = std::io::Cursor::new(Vec::new());
            match req.output_format {
                OutputFormat::Png => {
                    img.write_to(&mut buf, image::ImageFormat::Png)?;
                }
                OutputFormat::Jpeg => {
                    img.write_to(&mut buf, image::ImageFormat::Jpeg)?;
                }
            }

            images.push(ImageData {
                data: buf.into_inner(),
                format: req.output_format,
                width: req.width,
                height: req.height,
                index: i,
            });
        }

        Ok(GenerateResponse {
            images,
            generation_time_ms: start.elapsed().as_millis() as u64,
            model: req.model.clone(),
            seed_used: seed,
        })
    }

    fn load_model(&mut self, model: &str) -> Result<()> {
        // Stub: just mark the model as loaded
        tracing::info!(model, "loading model (stub)");
        let _model_path = self.models_dir.join(model);
        self.loaded.insert(
            model.to_string(),
            LoadedModel {
                name: model.to_string(),
            },
        );
        Ok(())
    }

    fn unload_model(&mut self, model: &str) -> Result<()> {
        self.loaded.remove(model);
        tracing::info!(model, "unloaded model");
        Ok(())
    }

    fn loaded_models(&self) -> Vec<String> {
        self.loaded.keys().cloned().collect()
    }
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
