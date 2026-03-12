use anyhow::Result;
use reqwest::Client;

use crate::types::{GenerateRequest, GenerateResponse, LoadModelRequest, ModelInfo, ServerStatus};

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

    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse> {
        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<GenerateResponse>()
            .await?;
        Ok(resp)
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

    pub async fn load_model(&self, model: &str) -> Result<()> {
        self.client
            .post(format!("{}/api/models/load", self.base_url))
            .json(&LoadModelRequest {
                model: model.to_string(),
            })
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn unload_model(&self, model: &str) -> Result<()> {
        self.client
            .delete(format!("{}/api/models/{}", self.base_url, model))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}
