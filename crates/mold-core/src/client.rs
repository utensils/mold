use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;

use crate::types::{GenerateRequest, GenerateResponse, ImageData, ModelInfo, ServerStatus};

/// Extended model info returned by /api/models, includes generation defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    #[serde(flatten)]
    pub defaults: ModelDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelDefaults {
    pub default_steps: u32,
    pub default_guidance: f64,
    pub default_width: u32,
    pub default_height: u32,
    pub description: String,
}

// Delegate the basic ModelInfo fields for ergonomic access.
impl std::ops::Deref for ModelInfoExtended {
    type Target = ModelInfo;
    fn deref(&self) -> &Self::Target {
        &self.info
    }
}

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
        let models = self.list_models_extended().await?;
        Ok(models.into_iter().map(|m| m.info).collect())
    }

    pub async fn list_models_extended(&self) -> Result<Vec<ModelInfoExtended>> {
        let resp = self
            .client
            .get(format!("{}/api/models", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<Vec<ModelInfoExtended>>()
            .await?;
        Ok(resp)
    }

    /// Check whether an error is a connection error (e.g. "connection refused").
    /// Useful for deciding whether to fall back to local inference.
    pub fn is_connection_error(err: &anyhow::Error) -> bool {
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() {
            return reqwest_err.is_connect();
        }
        false
    }

    /// Check whether an error is a 404 "model not found" from the server.
    /// Useful for triggering client-side pull when the server doesn't have the model.
    pub fn is_model_not_found(err: &anyhow::Error) -> bool {
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() {
            return reqwest_err.status() == Some(reqwest::StatusCode::NOT_FOUND);
        }
        false
    }

    pub fn host(&self) -> &str {
        &self.base_url
    }

    pub async fn unload_model(&self) -> Result<String> {
        let resp = self
            .client
            .delete(format!("{}/api/models/unload", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .text()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_trims_trailing_slash() {
        let client = MoldClient::new("http://localhost:7680/");
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn test_new_no_slash_unchanged() {
        let client = MoldClient::new("http://localhost:7680");
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn test_new_multiple_slashes() {
        let client = MoldClient::new("http://localhost:7680///");
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn test_from_env_uses_mold_host() {
        // Use a unique value so parallel tests don't collide
        let unique_url = "http://test-host-env:9999";
        unsafe { std::env::set_var("MOLD_HOST", unique_url) };
        let client = MoldClient::from_env();
        assert_eq!(client.host(), unique_url);
        unsafe { std::env::remove_var("MOLD_HOST") };
    }

    #[test]
    fn test_from_env_default_when_unset() {
        unsafe { std::env::remove_var("MOLD_HOST") };
        let client = MoldClient::from_env();
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn test_is_connection_error_non_connect() {
        // A generic anyhow error is not a connection error
        let err = anyhow::anyhow!("something went wrong");
        assert!(!MoldClient::is_connection_error(&err));
    }
}
