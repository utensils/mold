use anyhow::Result;
use base64::Engine as _;
use reqwest::Client;
use serde::Deserialize;

use crate::types::{
    GenerateRequest, GenerateResponse, ImageData, ModelInfo, ServerStatus, SseCompleteEvent,
    SseErrorEvent, SseProgressEvent,
};

/// Extended model info returned by /api/models, includes generation defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfoExtended {
    #[serde(flatten)]
    pub info: ModelInfo,
    #[serde(flatten)]
    pub defaults: ModelDefaults,
    /// Whether the model is downloaded on the server.
    #[serde(default)]
    pub downloaded: bool,
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
            base_url: normalize_host(base_url),
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
    /// Useful for triggering a server-side pull when the model isn't downloaded.
    pub fn is_model_not_found(err: &anyhow::Error) -> bool {
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() {
            return reqwest_err.status() == Some(reqwest::StatusCode::NOT_FOUND);
        }
        // SSE streaming returns ModelNotFoundError instead of reqwest status errors
        err.downcast_ref::<ModelNotFoundError>().is_some()
    }

    /// Generate an image via SSE streaming, receiving progress events.
    ///
    /// Returns:
    /// - `Ok(Some(response))` — streaming succeeded
    /// - `Ok(None)` — server doesn't support SSE (endpoint returned 404 with empty body)
    /// - `Err(e)` — generation error, model not found, or connection error
    pub async fn generate_stream(
        &self,
        req: &GenerateRequest,
        progress_tx: tokio::sync::mpsc::UnboundedSender<SseProgressEvent>,
    ) -> Result<Option<GenerateResponse>> {
        let mut resp = self
            .client
            .post(format!("{}/api/generate/stream", self.base_url))
            .json(req)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                // Axum returns empty 404 for unmatched routes — server doesn't support SSE
                return Ok(None);
            }
            // Non-empty 404 = model not found
            return Err(ModelNotFoundError(body).into());
        }

        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("validation error: {body}");
        }

        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        // Parse SSE events from chunked response body
        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE events (delimited by double newline)
            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                let mut event_type = String::new();
                let mut data = String::new();
                for line in event_text.lines() {
                    if line.starts_with(':') {
                        continue; // SSE comment (keep-alive ping)
                    }
                    if let Some(t) = line.strip_prefix("event:") {
                        event_type = t.trim().to_string();
                    } else if let Some(d) = line.strip_prefix("data:") {
                        data = d.trim().to_string();
                    }
                }

                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<SseProgressEvent>(&data) {
                            let _ = progress_tx.send(p);
                        }
                    }
                    "complete" => {
                        let complete: SseCompleteEvent = serde_json::from_str(&data)?;
                        let image_data =
                            base64::engine::general_purpose::STANDARD.decode(&complete.image)?;
                        return Ok(Some(GenerateResponse {
                            images: vec![ImageData {
                                data: image_data,
                                format: complete.format,
                                width: complete.width,
                                height: complete.height,
                                index: 0,
                            }],
                            generation_time_ms: complete.generation_time_ms,
                            model: req.model.clone(),
                            seed_used: complete.seed_used,
                        }));
                    }
                    "error" => {
                        let error: SseErrorEvent = serde_json::from_str(&data)?;
                        anyhow::bail!("server error: {}", error.message);
                    }
                    _ => {}
                }
            }
        }

        anyhow::bail!("SSE stream ended without complete event")
    }

    /// Ask the server to pull (download) a model. Blocks until the download
    /// completes on the server side. The server updates its in-memory config
    /// so subsequent generate/load requests can find the model.
    pub async fn pull_model(&self, model: &str) -> Result<String> {
        let resp = self
            .client
            .post(format!("{}/api/models/pull", self.base_url))
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        Ok(resp)
    }

    /// Pull a model via SSE streaming, receiving download progress events.
    ///
    /// Sends `Accept: text/event-stream` to request SSE from the server.
    /// Falls back to blocking pull if the server doesn't support SSE.
    pub async fn pull_model_stream(
        &self,
        model: &str,
        progress_tx: tokio::sync::mpsc::UnboundedSender<SseProgressEvent>,
    ) -> Result<()> {
        let mut resp = self
            .client
            .post(format!("{}/api/models/pull", self.base_url))
            .header("Accept", "text/event-stream")
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await?;

        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        // Check if server returned SSE or plain text
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("text/event-stream") {
            // Old server — blocking pull, no progress. Just consume the response.
            let _ = resp.text().await?;
            return Ok(());
        }

        // Parse SSE events (same pattern as generate_stream)
        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let event_text = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                let mut event_type = String::new();
                let mut data = String::new();
                for line in event_text.lines() {
                    if line.starts_with(':') {
                        continue;
                    }
                    if let Some(t) = line.strip_prefix("event:") {
                        event_type = t.trim().to_string();
                    } else if let Some(d) = line.strip_prefix("data:") {
                        data = d.trim().to_string();
                    }
                }

                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<SseProgressEvent>(&data) {
                            // PullComplete signals end of pull
                            let is_done = matches!(p, SseProgressEvent::PullComplete { .. });
                            let _ = progress_tx.send(p);
                            if is_done {
                                return Ok(());
                            }
                        }
                    }
                    "error" => {
                        let error: SseErrorEvent = serde_json::from_str(&data)?;
                        anyhow::bail!("server error: {}", error.message);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
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

/// Normalize a host string into a full URL.
///
/// Accepts:
/// - Bare hostname: `hal9000` → `http://hal9000:7680`
/// - Host with port: `hal9000:8080` → `http://hal9000:8080`
/// - Full URL: `http://hal9000:7680` → unchanged
/// - URL without port: `http://hal9000` → unchanged (uses scheme default 80/443)
fn normalize_host(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.contains("://") {
        trimmed.to_string()
    } else if trimmed.contains(':') {
        format!("http://{trimmed}")
    } else {
        format!("http://{trimmed}:7680")
    }
}

/// Error indicating a model was not found on the server (404 with body).
/// Detected by [`MoldClient::is_model_not_found`].
#[derive(Debug)]
pub struct ModelNotFoundError(pub String);

impl std::fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ModelNotFoundError {}

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

    #[test]
    fn test_is_model_not_found_via_custom_error() {
        let err: anyhow::Error =
            ModelNotFoundError("model 'test' is not downloaded".to_string()).into();
        assert!(MoldClient::is_model_not_found(&err));
    }

    #[test]
    fn test_is_model_not_found_generic_error() {
        let err = anyhow::anyhow!("something else");
        assert!(!MoldClient::is_model_not_found(&err));
    }

    #[test]
    fn test_normalize_bare_hostname() {
        let client = MoldClient::new("hal9000");
        assert_eq!(client.host(), "http://hal9000:7680");
    }

    #[test]
    fn test_normalize_hostname_with_port() {
        let client = MoldClient::new("hal9000:8080");
        assert_eq!(client.host(), "http://hal9000:8080");
    }

    #[test]
    fn test_normalize_full_url_unchanged() {
        let client = MoldClient::new("http://hal9000:7680");
        assert_eq!(client.host(), "http://hal9000:7680");
    }

    #[test]
    fn test_normalize_https_no_port() {
        let client = MoldClient::new("https://hal9000");
        assert_eq!(client.host(), "https://hal9000");
    }

    #[test]
    fn test_normalize_http_no_port() {
        let client = MoldClient::new("http://hal9000");
        assert_eq!(client.host(), "http://hal9000");
    }

    #[test]
    fn test_normalize_localhost() {
        let client = MoldClient::new("localhost");
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn test_normalize_whitespace_trimmed() {
        let client = MoldClient::new("  hal9000  ");
        assert_eq!(client.host(), "http://hal9000:7680");
    }

    #[test]
    fn test_normalize_ip_address() {
        let client = MoldClient::new("192.168.1.100");
        assert_eq!(client.host(), "http://192.168.1.100:7680");
    }

    #[test]
    fn test_normalize_ip_with_port() {
        let client = MoldClient::new("192.168.1.100:9090");
        assert_eq!(client.host(), "http://192.168.1.100:9090");
    }
}
