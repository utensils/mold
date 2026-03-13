use anyhow::Result;
use reqwest::Client;

use crate::types::{GenerateRequest, GenerateResponse, ImageData, ModelInfo, ServerStatus};

pub struct MoldClient {
    base_url: String,
    client: Client,
}

impl MoldClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    pub fn from_env() -> Self {
        let base_url =
            std::env::var("MOLD_HOST").unwrap_or_else(|_| "http://localhost:7680".to_string());
        Self::new(&base_url)
    }

    /// Generate an image. Returns raw image bytes (PNG or JPEG).
    /// The server returns raw bytes, not JSON — callers are responsible for
    /// writing the bytes to disk or further processing.
    pub async fn generate_raw(&self, req: &GenerateRequest) -> Result<Vec<u8>> {
        let bytes = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(req)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec();
        Ok(bytes)
    }

    /// Generate an image and return a minimal response wrapping the raw bytes.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse> {
        let seed = req.seed.unwrap_or(0);
        let width = req.width;
        let height = req.height;
        let model = req.model.clone();
        let format = req.output_format;

        let start = std::time::Instant::now();
        let data = self.generate_raw(&req).await?;
        let generation_time_ms = start.elapsed().as_millis() as u64;

        Ok(GenerateResponse {
            images: vec![ImageData {
                data,
                format,
                width,
                height,
                index: 0,
            }],
            generation_time_ms,
            model,
            seed_used: seed,
        })
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let resp = self
            .client
            .get(format!("{}/api/models", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<Vec<ModelInfo>>()
            .await?;
        Ok(resp)
    }

    pub async fn server_status(&self) -> Result<ServerStatus> {
        let resp = self
            .client
            .get(format!("{}/api/status", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<ServerStatus>()
            .await?;
        Ok(resp)
    }
}
