use crate::chain::{ChainProgressEvent, ChainRequest, ChainResponse, SseChainCompleteEvent};
use crate::chain_job::{
    ChainJobDetail, ChainJobListing, ChainJobSummary, CreateChainJobResponse, GcOutcome,
    RetakeRequest,
};
use crate::error::MoldError;
use crate::types::{
    AudioData, DeviceState, ExpandRequest, ExpandResponse, GalleryImage, GenerateRequest,
    GenerateResponse, ImageData, LoraInfo, ModelInfo, ModelInfoExtended, OutputFormat,
    QueueListingWire, ServerStatus, SseCompleteEvent, SseErrorEvent, SseProgressEvent, VideoData,
};
use anyhow::{Context, Result};
use base64::Engine as _;
use reqwest::{Client, StatusCode};

#[derive(Clone)]
pub struct MoldClient {
    base_url: String,
    client: Client,
}

impl MoldClient {
    pub fn new(base_url: &str) -> Self {
        let client = build_client(None);
        Self {
            base_url: normalize_host(base_url),
            client,
        }
    }

    /// Create a client with an explicit API key for authentication.
    pub fn with_api_key(base_url: &str, api_key: String) -> Self {
        let client = build_client(Some(&api_key));
        Self {
            base_url: normalize_host(base_url),
            client,
        }
    }

    pub fn from_env() -> Self {
        let base_url =
            std::env::var("MOLD_HOST").unwrap_or_else(|_| "http://localhost:7680".to_string());
        let api_key = std::env::var("MOLD_API_KEY").ok().filter(|k| !k.is_empty());
        let client = build_client(api_key.as_deref());
        Self {
            base_url: normalize_host(&base_url),
            client,
        }
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

    /// Generate an image or video and return the response wrapping the raw bytes.
    ///
    /// For video responses the server sends `x-mold-video-*` metadata headers
    /// alongside the raw video bytes so we can reconstruct [`VideoData`].
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateResponse> {
        let fallback_seed = req.seed.unwrap_or(0);
        let width = req.width;
        let height = req.height;
        let model = req.model.clone();
        let format = req.resolved_output_format();

        let start = std::time::Instant::now();
        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?
            .error_for_status()?;

        // Read the seed the server actually used from the response header.
        // Fall back to the request seed for backward compat with older servers.
        let seed_used = resp
            .headers()
            .get("x-mold-seed-used")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(fallback_seed);
        let gpu = resp
            .headers()
            .get("x-mold-gpu")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<usize>().ok());

        // Detect an audio-only response before the video probe: an audio print
        // has no frames, so a video-shaped probe falls through to the image
        // branch and hands the caller WAV bytes labelled as an image.
        let audio_meta = parse_audio_headers(resp.headers());
        // Detect video response via x-mold-video-frames header
        let video_meta = parse_video_headers(resp.headers());

        let data = resp.bytes().await?.to_vec();
        let generation_time_ms = start.elapsed().as_millis() as u64;

        if let Some(meta) = audio_meta {
            return Ok(GenerateResponse {
                audio: Some(AudioData {
                    data,
                    // The request's format only stands in for an older server
                    // that predates the header; an audio-only response is
                    // never the still-image default the request may carry.
                    format: meta.format.unwrap_or(if format.is_audio() {
                        format
                    } else {
                        OutputFormat::Wav
                    }),
                    sample_rate: meta.sample_rate,
                    channels: meta.channels,
                    duration_ms: meta.duration_ms,
                    // The waveform tile cannot ride along in a body that is
                    // already the WAV. Only its recorded size travels.
                    thumbnail: Vec::new(),
                    thumbnail_width: meta.thumbnail_width,
                    thumbnail_height: meta.thumbnail_height,
                }),
                images: Vec::new(),
                video: None,
                generation_time_ms,
                model,
                seed_used,
                gpu,
            });
        }

        let video = video_meta.map(|meta| VideoData {
            data: data.clone(),
            format,
            width: meta.width.unwrap_or(width),
            height: meta.height.unwrap_or(height),
            frames: meta.frames,
            fps: meta.fps,
            pipeline: meta.pipeline,
            thumbnail: Vec::new(),
            gif_preview: Vec::new(),
            has_audio: meta.has_audio,
            duration_ms: meta.duration_ms,
            audio_sample_rate: meta.audio_sample_rate,
            audio_channels: meta.audio_channels,
        });

        // For video responses, images is empty — the payload lives in `video`.
        let images = if video.is_some() {
            Vec::new()
        } else {
            vec![ImageData {
                data,
                format,
                width,
                height,
                index: 0,
            }]
        };

        Ok(GenerateResponse {
            audio: None,
            images,
            generation_time_ms,
            model,
            seed_used,
            video,
            gpu,
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

    /// List installed LoRAs from the server, optionally filtered to a model's family.
    pub async fn list_loras(&self, model: Option<&str>) -> Result<Vec<LoraInfo>> {
        match self.list_loras_endpoint(model).await {
            Ok(loras) => Ok(loras),
            Err(err) if should_fallback_loras_endpoint(&err) => self
                .list_loras_from_installed_catalog(model)
                .await
                .with_context(|| {
                    format!(
                        "failed to list LoRAs via /api/loras ({err}); fallback to /api/catalog/installed also failed"
                    )
                }),
            Err(err) => Err(err),
        }
    }

    async fn list_loras_endpoint(&self, model: Option<&str>) -> Result<Vec<LoraInfo>> {
        let req = self.client.get(format!("{}/api/loras", self.base_url));
        let req = if let Some(model) = model {
            req.query(&[("model", model)])
        } else {
            req
        };
        let resp = req
            .send()
            .await?
            .error_for_status()?
            .json::<Vec<LoraInfo>>()
            .await?;
        Ok(resp)
    }

    async fn list_loras_from_installed_catalog(
        &self,
        model: Option<&str>,
    ) -> Result<Vec<LoraInfo>> {
        let family = model.and_then(lora_family_for_model_filter);
        let mut req = self
            .client
            .get(format!("{}/api/catalog/installed", self.base_url))
            .query(&[("kind", "lora")]);
        if let Some(family) = family.as_deref() {
            req = req.query(&[("family", family)]);
        }

        let resp = req
            .send()
            .await?
            .error_for_status()?
            .json::<crate::catalog_wire::InstalledCatalogResponse>()
            .await?;
        let family = family.as_deref();
        let mut loras = resp
            .entries
            .into_iter()
            .filter_map(installed_entry_into_lora_info)
            .filter(|lora| family.is_none_or(|family| lora.family == family))
            .collect::<Vec<_>>();
        loras.sort_by(|a, b| {
            b.added_at
                .cmp(&a.added_at)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(loras)
    }

    /// Check whether an error is a connection error (e.g. "connection refused").
    /// Useful for deciding whether to fall back to local inference.
    pub fn is_connection_error(err: &anyhow::Error) -> bool {
        // Check for MoldError::Client variant
        if let Some(mold_err) = err.downcast_ref::<MoldError>() {
            if matches!(mold_err, MoldError::Client(_)) {
                return true;
            }
        }
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() {
            return reqwest_err.is_connect();
        }
        false
    }

    /// Check whether an error is a 404 "model not found" from the server.
    /// Useful for triggering a server-side pull when the model isn't downloaded.
    pub fn is_model_not_found(err: &anyhow::Error) -> bool {
        // Check for MoldError::ModelNotFound variant
        if let Some(mold_err) = err.downcast_ref::<MoldError>() {
            if matches!(mold_err, MoldError::ModelNotFound(_)) {
                return true;
            }
        }
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
            return Err(MoldError::ModelNotFound(body).into());
        }

        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            let body = resp.text().await.unwrap_or_default();
            return Err(MoldError::Validation(api_error_detail(&body)).into());
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

            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (event_type, data) = parse_sse_event(&event_text);
                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<SseProgressEvent>(&data) {
                            let _ = progress_tx.send(p);
                        }
                    }
                    "complete" => {
                        let complete: SseCompleteEvent = serde_json::from_str(&data)?;
                        let payload =
                            base64::engine::general_purpose::STANDARD.decode(&complete.image)?;
                        let b64 = base64::engine::general_purpose::STANDARD;
                        // Use server-provided model name (source of truth);
                        // fall back to request model for backwards compat with
                        // older servers that don't include it.
                        let model = if complete.model.is_empty() {
                            req.model.clone()
                        } else {
                            complete.model
                        };

                        // Detect an audio-only response via `audio_sample_rate`.
                        // Checked before video: an audio print has no frames,
                        // so the video probe below would fall through to the
                        // image branch and hand the caller WAV bytes labelled
                        // as an image.
                        if let Some(sample_rate) = complete.audio_sample_rate {
                            let thumbnail = complete
                                .audio_thumbnail
                                .as_deref()
                                .and_then(|s| b64.decode(s).ok())
                                .unwrap_or_default();
                            return Ok(Some(GenerateResponse {
                                images: Vec::new(),
                                video: None,
                                audio: Some(AudioData {
                                    data: payload,
                                    format: complete.format,
                                    sample_rate,
                                    channels: complete.audio_channels.unwrap_or(1),
                                    duration_ms: complete.audio_duration_ms.unwrap_or(0),
                                    thumbnail,
                                    thumbnail_width: complete.width,
                                    thumbnail_height: complete.height,
                                }),
                                generation_time_ms: complete.generation_time_ms,
                                model,
                                seed_used: complete.seed_used,
                                gpu: complete.gpu,
                            }));
                        }

                        // Detect video response via video_frames field
                        let (images, video) = if let (Some(frames), Some(fps)) =
                            (complete.video_frames, complete.video_fps)
                        {
                            let thumbnail = complete
                                .video_thumbnail
                                .as_deref()
                                .and_then(|s| b64.decode(s).ok())
                                .unwrap_or_default();
                            let gif_preview = complete
                                .video_gif_preview
                                .as_deref()
                                .and_then(|s| b64.decode(s).ok())
                                .unwrap_or_default();
                            let vd = VideoData {
                                data: payload,
                                format: complete.format,
                                width: complete.width,
                                height: complete.height,
                                frames,
                                fps,
                                pipeline: complete.metadata.as_ref().and_then(|m| m.pipeline),
                                thumbnail,
                                gif_preview,
                                has_audio: complete.video_has_audio,
                                duration_ms: complete.video_duration_ms,
                                audio_sample_rate: complete.video_audio_sample_rate,
                                audio_channels: complete.video_audio_channels,
                            };
                            (Vec::new(), Some(vd))
                        } else {
                            let img = ImageData {
                                data: payload,
                                format: complete.format,
                                width: complete.width,
                                height: complete.height,
                                index: 0,
                            };
                            (vec![img], None)
                        };

                        return Ok(Some(GenerateResponse {
                            audio: None,
                            images,
                            generation_time_ms: complete.generation_time_ms,
                            model,
                            seed_used: complete.seed_used,
                            video,
                            gpu: complete.gpu,
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

    /// Submit a chained video generation request (non-streaming).
    ///
    /// The server normalises the auto-expand form into stages, runs each
    /// stage sequentially with motion-tail latent carryover, stitches the
    /// result into a single video, and returns a [`ChainResponse`]. Large
    /// chains take minutes — prefer [`Self::generate_chain_stream`] for
    /// interactive clients that want progress updates.
    pub async fn generate_chain(&self, req: &ChainRequest) -> Result<ChainResponse> {
        let resp = self
            .client
            .post(format!("{}/api/generate/chain", self.base_url))
            .json(req)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                anyhow::bail!("chain endpoint not found — server predates render-chain v1");
            }
            return Err(MoldError::ModelNotFound(body).into());
        }
        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            let body = resp.text().await.unwrap_or_default();
            return Err(MoldError::Validation(api_error_detail(&body)).into());
        }
        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        let chain: ChainResponse = resp.json().await?;
        Ok(chain)
    }

    /// Submit a chained video generation request with SSE progress streaming.
    ///
    /// Returns:
    /// - `Ok(Some(response))` — streaming succeeded and the `complete` event
    ///   carried the stitched video.
    /// - `Ok(None)` — server doesn't have the chain endpoint (empty 404).
    ///   Callers can fall back to [`Self::generate_chain`] or error.
    /// - `Err(_)` — validation, model-not-found, or mid-stream server error.
    pub async fn generate_chain_stream(
        &self,
        req: &ChainRequest,
        progress_tx: tokio::sync::mpsc::UnboundedSender<ChainProgressEvent>,
    ) -> Result<Option<ChainResponse>> {
        let mut resp = self
            .client
            .post(format!("{}/api/generate/chain/stream", self.base_url))
            .json(req)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                return Ok(None);
            }
            return Err(MoldError::ModelNotFound(body).into());
        }
        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            let body = resp.text().await.unwrap_or_default();
            return Err(MoldError::Validation(api_error_detail(&body)).into());
        }
        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        let b64 = base64::engine::general_purpose::STANDARD;
        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (event_type, data) = parse_sse_event(&event_text);
                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<ChainProgressEvent>(&data) {
                            let _ = progress_tx.send(p);
                        }
                    }
                    "complete" => {
                        let complete: SseChainCompleteEvent = serde_json::from_str(&data)?;
                        let payload = b64.decode(&complete.video)?;
                        let thumbnail = complete
                            .thumbnail
                            .as_deref()
                            .and_then(|s| b64.decode(s).ok())
                            .unwrap_or_default();
                        let gif_preview = complete
                            .gif_preview
                            .as_deref()
                            .and_then(|s| b64.decode(s).ok())
                            .unwrap_or_default();
                        let video = VideoData {
                            data: payload,
                            format: complete.format,
                            width: complete.width,
                            height: complete.height,
                            frames: complete.frames,
                            fps: complete.fps,
                            pipeline: complete.metadata.as_ref().and_then(|m| m.pipeline),
                            thumbnail,
                            gif_preview,
                            has_audio: complete.has_audio,
                            duration_ms: complete.duration_ms,
                            audio_sample_rate: complete.audio_sample_rate,
                            audio_channels: complete.audio_channels,
                        };
                        return Ok(Some(ChainResponse {
                            video,
                            stage_count: complete.stage_count,
                            gpu: complete.gpu,
                            script: complete.script,
                            vram_estimate: complete.vram_estimate,
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

        anyhow::bail!("chain SSE stream ended without complete event")
    }

    pub async fn create_chain_job(&self, req: &ChainRequest) -> Result<CreateChainJobResponse> {
        let resp = self
            .client
            .post(format!("{}/api/chain-jobs", self.base_url))
            .json(req)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<CreateChainJobResponse>()
            .await?)
    }

    pub async fn list_chain_jobs(&self) -> Result<ChainJobListing> {
        let resp = self
            .client
            .get(format!("{}/api/chain-jobs", self.base_url))
            .send()
            .await?;
        let mut listing = error_for_status_with_body(resp)
            .await?
            .json::<ChainJobListing>()
            .await?;
        // Older servers exposed the ephemeral runner records used by
        // automatic long one-shot videos. They are not authored sequences and
        // must stay out of `mold jobs list` even across a mixed-version pair.
        listing.jobs.retain(|job| !job.ephemeral);
        Ok(listing)
    }

    pub async fn get_chain_job(&self, id: &str) -> Result<ChainJobDetail> {
        let resp = self
            .client
            .get(format!(
                "{}/api/chain-jobs/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<ChainJobDetail>()
            .await?)
    }

    pub async fn resume_chain_job(&self, id: &str) -> Result<ChainJobSummary> {
        let resp = self
            .client
            .post(format!(
                "{}/api/chain-jobs/{}/resume",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<ChainJobSummary>()
            .await?)
    }

    pub async fn retake_chain_job(&self, id: &str, req: &RetakeRequest) -> Result<ChainJobSummary> {
        let resp = self
            .client
            .post(format!(
                "{}/api/chain-jobs/{}/retake",
                self.base_url,
                encode_path_segment(id)
            ))
            .json(req)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<ChainJobSummary>()
            .await?)
    }

    pub async fn cancel_chain_job(&self, id: &str) -> Result<ChainJobSummary> {
        let resp = self
            .client
            .post(format!(
                "{}/api/chain-jobs/{}/cancel",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<ChainJobSummary>()
            .await?)
    }

    pub async fn delete_chain_job(&self, id: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/chain-jobs/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    pub async fn gc_chain_jobs(&self) -> Result<GcOutcome> {
        let resp = self
            .client
            .post(format!("{}/api/chain-jobs/gc", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<GcOutcome>()
            .await?)
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

    /// Request graceful server shutdown.
    pub async fn shutdown_server(&self) -> Result<()> {
        self.client
            .post(format!("{}/api/shutdown", self.base_url))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
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
            // Drop the sender so the receiver's recv() returns None instead of blocking.
            drop(progress_tx);
            let _ = resp.text().await?;
            return Ok(());
        }

        // Parse SSE events (same pattern as generate_stream)
        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (event_type, data) = parse_sse_event(&event_text);
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
        self.unload_model_target(None, None).await
    }

    pub async fn unload_model_target(
        &self,
        model: Option<&str>,
        gpu: Option<usize>,
    ) -> Result<String> {
        let req = serde_json::json!({
            "model": model,
            "gpu": gpu,
        });
        let builder = self
            .client
            .delete(format!("{}/api/models/unload", self.base_url));
        let builder = if model.is_some() || gpu.is_some() {
            builder.json(&req)
        } else {
            builder
        };
        let resp = builder.send().await?.error_for_status()?.text().await?;
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

    /// Feature-detect optional server contracts before rendering controls.
    pub async fn server_capabilities(&self) -> Result<crate::ServerCapabilities> {
        let resp = self
            .client
            .get(format!("{}/api/capabilities", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<crate::ServerCapabilities>()
            .await?;
        Ok(resp)
    }

    /// Read the server's stable, runtime-visible device inventory.
    pub async fn devices(&self) -> Result<DeviceState> {
        let resp = self
            .client
            .get(format!("{}/api/devices", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<DeviceState>()
            .await?;
        Ok(resp)
    }

    /// Alias used by clients that preflight optional administrative actions.
    pub async fn capabilities(&self) -> Result<crate::ServerCapabilities> {
        self.server_capabilities().await
    }

    /// Enable or disable one stable device. The server may return a draining
    /// or starting state while the owner transition completes.
    pub async fn set_device_enabled(
        &self,
        device_id: &str,
        enabled: bool,
    ) -> Result<crate::DeviceInfo> {
        let response = self
            .client
            .patch(format!(
                "{}/api/devices/{}",
                self.base_url,
                encode_path_segment(device_id)
            ))
            .json(&crate::DeviceMutationRequest { enabled })
            .send()
            .await?;
        let response = error_for_status_with_body(response)
            .await?
            .json::<crate::DeviceInfo>()
            .await?;
        Ok(response)
    }

    /// Snapshot the server's generation queue (`GET /api/queue`).
    ///
    /// The listing covers everything currently queued or running — completed
    /// jobs live in the gallery, not here. Entries are wire-shaped twins of
    /// the server's `job_registry::JobEntry`; see [`QueueListingWire`] for
    /// the forward-compat rules (plain-string `state`, defaulted additive
    /// fields), which let this client talk to both older and newer servers.
    pub async fn list_queue(&self) -> Result<QueueListingWire> {
        let resp = self
            .client
            .get(format!("{}/api/queue", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<QueueListingWire>()
            .await?;
        Ok(resp)
    }

    /// Cancel a still-queued job (`DELETE /api/queue/{id}`).
    ///
    /// The server answers `204` on success, `404` for unknown ids, and `409`
    /// when the job is already running on a GPU worker (running jobs are not
    /// cancelable — there is no safe preemption point). Non-2xx responses
    /// surface as errors carrying the response body text, so the 409 reason
    /// reaches the caller verbatim.
    pub async fn cancel_queue_job(&self, id: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/queue/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// List gallery images from the server's output directory.
    pub async fn list_gallery(&self) -> Result<Vec<GalleryImage>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<Vec<GalleryImage>>()
            .await?;
        Ok(resp)
    }

    /// Download a gallery image by filename.
    pub async fn get_gallery_image(&self, filename: &str) -> Result<Vec<u8>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery/image/{filename}", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(resp.to_vec())
    }

    /// Delete a gallery image on the server.
    pub async fn delete_gallery_image(&self, filename: &str) -> Result<()> {
        self.client
            .delete(format!("{}/api/gallery/image/{filename}", self.base_url))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Download a cached animated GIF preview for a video gallery entry.
    ///
    /// Returns `Ok(None)` when the server responds with 404 (no preview
    /// has been generated for this filename yet). Callers are expected to
    /// fall back to the full `get_gallery_image` path in that case.
    pub async fn get_gallery_preview(&self, filename: &str) -> Result<Option<Vec<u8>>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery/preview/{filename}", self.base_url))
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let bytes = resp.error_for_status()?.bytes().await?;
        Ok(Some(bytes.to_vec()))
    }

    /// Download a gallery thumbnail by filename. Smaller/faster than full image.
    pub async fn get_gallery_thumbnail(&self, filename: &str) -> Result<Vec<u8>> {
        let resp = self
            .client
            .get(format!(
                "{}/api/gallery/thumbnail/{filename}",
                self.base_url
            ))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(resp.to_vec())
    }

    /// Expand a prompt using the server's LLM prompt expansion endpoint.
    pub async fn expand_prompt(&self, req: &ExpandRequest) -> Result<ExpandResponse> {
        let resp = self
            .client
            .post(format!("{}/api/expand", self.base_url))
            .json(req)
            .send()
            .await?
            .error_for_status()?
            .json::<ExpandResponse>()
            .await?;
        Ok(resp)
    }

    /// Upscale an image using a super-resolution model on the server.
    pub async fn upscale(&self, req: &crate::UpscaleRequest) -> Result<crate::UpscaleResponse> {
        let resp = self
            .client
            .post(format!("{}/api/upscale", self.base_url))
            .json(req)
            .send()
            .await?
            .error_for_status()?
            .json::<crate::UpscaleResponse>()
            .await?;
        Ok(resp)
    }

    /// Upscale an image via SSE streaming -- progress events are sent to `progress_tx`,
    /// returns the final `UpscaleResponse` on success.
    pub async fn upscale_stream(
        &self,
        req: &crate::UpscaleRequest,
        progress_tx: tokio::sync::mpsc::UnboundedSender<SseProgressEvent>,
    ) -> Result<Option<crate::UpscaleResponse>> {
        let mut resp = self
            .client
            .post(format!("{}/api/upscale/stream", self.base_url))
            .json(req)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                return Ok(None); // server doesn't support SSE upscale
            }
            return Err(MoldError::ModelNotFound(body).into());
        }

        if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
            let body = resp.text().await.unwrap_or_default();
            return Err(MoldError::Validation(api_error_detail(&body)).into());
        }

        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (event_type, data) = parse_sse_event(&event_text);
                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<SseProgressEvent>(&data) {
                            let _ = progress_tx.send(p);
                        }
                    }
                    "complete" => {
                        let complete: crate::SseUpscaleCompleteEvent = serde_json::from_str(&data)?;
                        let image_data =
                            base64::engine::general_purpose::STANDARD.decode(&complete.image)?;
                        return Ok(Some(crate::UpscaleResponse {
                            image: crate::ImageData {
                                data: image_data,
                                format: complete.format,
                                width: complete.original_width * complete.scale_factor,
                                height: complete.original_height * complete.scale_factor,
                                index: 0,
                            },
                            upscale_time_ms: complete.upscale_time_ms,
                            model: complete.model,
                            scale_factor: complete.scale_factor,
                            original_width: complete.original_width,
                            original_height: complete.original_height,
                        }));
                    }
                    "error" => {
                        let error: crate::SseErrorEvent = serde_json::from_str(&data)?;
                        anyhow::bail!("server error: {}", error.message);
                    }
                    _ => {}
                }
            }
        }

        anyhow::bail!("SSE stream ended without complete event")
    }
}

/// Client policy: which installed-catalog entries surface as usable LoRAs.
fn installed_entry_into_lora_info(
    entry: crate::catalog_wire::InstalledCatalogEntry,
) -> Option<LoraInfo> {
    if entry.kind != "lora" || !entry.installed {
        return None;
    }
    Some(LoraInfo {
        id: entry.id,
        name: entry.name,
        family: entry.family,
        author: entry.author,
        path: entry.primary_path?,
        trained_words: entry.trained_words,
        size_bytes: entry.size_bytes,
        thumbnail_url: entry.thumbnail_url,
        added_at: entry.added_at,
    })
}

fn should_fallback_loras_endpoint(err: &anyhow::Error) -> bool {
    let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() else {
        return false;
    };
    reqwest_err.is_decode()
        || reqwest_err.status().is_some_and(|status| {
            matches!(
                status,
                StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED
            )
        })
}

async fn error_for_status_with_body(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_client_error() || resp.status().is_server_error() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("server error {status}: {body}");
    }
    Ok(resp)
}

fn api_error_detail(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .or_else(|| value.get("message"))
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| body.trim().to_string())
}

fn lora_family_for_model_filter(model: &str) -> Option<String> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }
    let canonical = crate::manifest::resolve_model_name(model);
    crate::manifest::find_manifest(&canonical)
        .or_else(|| crate::manifest::find_manifest(model))
        .map(|manifest| catalog_lora_family_filter(&manifest.family))
        .or_else(|| {
            let config = crate::Config::load_or_default();
            config
                .models
                .get(model)
                .or_else(|| config.models.get(&canonical))
                .and_then(|model| model.family.as_deref().map(catalog_lora_family_filter))
        })
}

fn catalog_lora_family_filter(family: &str) -> String {
    match family {
        "qwen-image-edit" | "qwen_image_edit" => "qwen-image".to_string(),
        other => other.to_string(),
    }
}

/// Parsed video metadata from `x-mold-video-*` response headers.
struct VideoMeta {
    frames: u32,
    fps: u32,
    width: Option<u32>,
    height: Option<u32>,
    pipeline: Option<crate::Ltx2PipelineMode>,
    has_audio: bool,
    duration_ms: Option<u64>,
    audio_sample_rate: Option<u32>,
    audio_channels: Option<u32>,
}

/// Parse video metadata from HTTP response headers.
/// Returns `Some` when `x-mold-video-frames` is present, indicating a video response.
fn parse_video_headers(headers: &reqwest::header::HeaderMap) -> Option<VideoMeta> {
    let frames = headers
        .get("x-mold-video-frames")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())?;
    let fps = headers
        .get("x-mold-video-fps")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(24);
    let width = headers
        .get("x-mold-video-width")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let height = headers
        .get("x-mold-video-height")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let pipeline = headers
        .get("x-mold-video-pipeline")
        .and_then(|v| v.to_str().ok())
        .and_then(|value| serde_json::from_value(serde_json::Value::String(value.into())).ok());
    let has_audio = headers
        .get("x-mold-video-has-audio")
        .and_then(|v| v.to_str().ok())
        .map(|s| s == "1")
        .unwrap_or(false);
    let duration_ms = headers
        .get("x-mold-video-duration-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let audio_sample_rate = headers
        .get("x-mold-video-audio-sample-rate")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let audio_channels = headers
        .get("x-mold-video-audio-channels")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    Some(VideoMeta {
        frames,
        fps,
        width,
        height,
        pipeline,
        has_audio,
        duration_ms,
        audio_sample_rate,
        audio_channels,
    })
}

struct AudioMeta {
    format: Option<OutputFormat>,
    sample_rate: u32,
    channels: u32,
    duration_ms: u64,
    thumbnail_width: u32,
    thumbnail_height: u32,
}

/// Parse audio metadata from HTTP response headers.
///
/// Returns `Some` when `x-mold-audio-sample-rate` is present, indicating an
/// audio-only response. Probed before the video headers for the same reason
/// the SSE branch probes `audio_sample_rate` first: an audio print has no
/// frames, so a video-shaped probe would fall through to the image branch and
/// hand the caller WAV bytes labelled as an image.
fn parse_audio_headers(headers: &reqwest::header::HeaderMap) -> Option<AudioMeta> {
    let read = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
    };
    let sample_rate = read("x-mold-audio-sample-rate")? as u32;
    Some(AudioMeta {
        // Stated by the server, because a request that omitted
        // `output_format` was normalised server-side and is no evidence of
        // what came back.
        format: headers
            .get("x-mold-audio-format")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<OutputFormat>().ok()),
        sample_rate,
        channels: read("x-mold-audio-channels").unwrap_or(1) as u32,
        duration_ms: read("x-mold-audio-duration-ms").unwrap_or(0),
        thumbnail_width: read("x-mold-audio-thumbnail-width").unwrap_or(0) as u32,
        thumbnail_height: read("x-mold-audio-thumbnail-height").unwrap_or(0) as u32,
    })
}

fn next_sse_event(buffer: &mut String) -> Option<String> {
    for separator in ["\r\n\r\n", "\n\n"] {
        if let Some(pos) = buffer.find(separator) {
            let event_text = buffer[..pos].to_string();
            *buffer = buffer[pos + separator.len()..].to_string();
            return Some(event_text);
        }
    }
    None
}

fn parse_sse_event(event_text: &str) -> (String, String) {
    let mut event_type = String::new();
    let mut data_lines = Vec::new();
    for line in event_text.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(t) = line.strip_prefix("event:") {
            event_type = t.trim().to_string();
        } else if let Some(d) = line.strip_prefix("data:") {
            data_lines.push(d.trim().to_string());
        }
    }
    (event_type, data_lines.join("\n"))
}

/// Build a reqwest Client, optionally with a default `X-Api-Key` header.
fn build_client(api_key: Option<&str>) -> Client {
    let mut builder = Client::builder();
    if let Some(key) = api_key {
        let mut headers = reqwest::header::HeaderMap::new();
        match reqwest::header::HeaderValue::from_str(key) {
            Ok(val) => {
                headers.insert("x-api-key", val);
            }
            Err(_) => {
                eprintln!(
                    "warning: MOLD_API_KEY contains characters invalid for an HTTP header; \
                     authentication header will not be sent"
                );
            }
        }
        builder = builder.default_headers(headers);
    }
    builder.build().unwrap_or_else(|_| Client::new())
}

/// Normalize a host string into a full URL.
///
/// Accepts:
/// - Bare hostname: `hal9000` → `http://hal9000:7680`
/// - Host with port: `hal9000:8080` → `http://hal9000:8080`
/// - Full URL: `http://hal9000:7680` → unchanged
/// - URL without port: `http://hal9000` → unchanged (uses scheme default 80/443)
pub fn normalize_host(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.contains("://") {
        trimmed.to_string()
    } else if trimmed.contains(':') {
        format!("http://{trimmed}")
    } else {
        format!("http://{trimmed}:7680")
    }
}

fn encode_path_segment(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
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
    use crate::test_support::ENV_LOCK;

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
    fn test_from_env_mold_host() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Single test to avoid env var races between parallel tests
        unsafe { std::env::remove_var("MOLD_HOST") };
        let client = MoldClient::from_env();
        assert_eq!(client.host(), "http://localhost:7680");

        let unique_url = "http://test-host-env:9999";
        unsafe { std::env::set_var("MOLD_HOST", unique_url) };
        let client = MoldClient::from_env();
        assert_eq!(client.host(), unique_url);
        unsafe { std::env::remove_var("MOLD_HOST") };
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

    #[test]
    fn test_is_model_not_found_via_mold_error() {
        let err: anyhow::Error =
            MoldError::ModelNotFound("model 'test' is not downloaded".to_string()).into();
        assert!(MoldClient::is_model_not_found(&err));
    }

    #[test]
    fn test_is_connection_error_via_mold_error() {
        let err: anyhow::Error = MoldError::Client("connection refused".to_string()).into();
        assert!(MoldClient::is_connection_error(&err));
    }

    #[tokio::test]
    async fn list_loras_falls_back_to_installed_catalog_for_older_servers() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/loras"))
            .and(query_param("model", "flux-dev:q8"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html")
                    .set_body_string("<!doctype html><html></html>"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/catalog/installed"))
            .and(query_param("kind", "lora"))
            .and(query_param("family", "flux"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entries": [
                    {
                        "id": "cv:827325",
                        "name": "Flux Skin Texture",
                        "family": "flux",
                        "author": null,
                        "primary_path": "/models/cv-827325/fluxRealSkin-V2.safetensors",
                        "trained_words": ["realskin"],
                        "size_bytes": 167938890,
                        "thumbnail_url": null,
                        "added_at": 1778268326,
                        "installed": true,
                        "kind": "lora"
                    }
                ],
                "page": 1,
                "page_size": 1,
                "total": 1
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let loras = client.list_loras(Some("flux-dev:q8")).await.unwrap();

        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0].id, "cv:827325");
        assert_eq!(
            loras[0].path,
            "/models/cv-827325/fluxRealSkin-V2.safetensors"
        );
        assert_eq!(loras[0].trained_words, ["realskin"]);
    }

    #[tokio::test]
    async fn devices_fetches_and_parses_the_stable_inventory() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/devices"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "devices": [{
                    "id": "cuda:0123456789abcdef0123456789abcdef",
                    "backend": "cuda",
                    "ordinal": 0,
                    "device_kind": "full_gpu",
                    "nvml_uuid": null,
                    "physical_uuid": null,
                    "mig_uuid": null,
                    "mig_parent_uuid": null,
                    "mig_profile": null,
                    "name": "test gpu",
                    "pci_bus_id": null,
                    "compute_capability": "8.6",
                    "memory": {
                        "total_bytes": 24_000_000_000_u64,
                        "used_bytes": null,
                        "mold_used_bytes": null,
                        "other_used_bytes": null
                    },
                    "telemetry": {
                        "utilization_percent": null,
                        "temperature_c": null,
                        "power_w": null
                    },
                    "desired_enabled": true,
                    "admin_state": "enabled",
                    "health": "healthy",
                    "activity": "idle",
                    "schedulable": true,
                    "unschedulable_reason": null,
                    "loaded_models": [],
                    "active_work_id": null,
                    "planned_work_ids": []
                }],
                "plan_version": 0
            })))
            .mount(&server)
            .await;

        let devices = MoldClient::new(&server.uri()).devices().await.unwrap();
        assert_eq!(
            devices.devices[0].id,
            "cuda:0123456789abcdef0123456789abcdef"
        );
        assert_eq!(devices.devices[0].device_kind, crate::DeviceKind::FullGpu);
        assert_eq!(devices.devices[0].memory.used_bytes, None);
    }

    #[tokio::test]
    async fn capabilities_defaults_missing_device_lifecycle_to_unavailable() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "gallery": { "can_delete": true },
                "catalog": { "available": false, "families": [], "sort": [] }
            })))
            .mount(&server)
            .await;

        let capabilities = MoldClient::new(&server.uri()).capabilities().await.unwrap();
        assert!(!capabilities.devices.lifecycle);
    }

    #[tokio::test]
    async fn set_device_enabled_preserves_the_server_error_body() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/api/devices/cuda%3Adevice-1"))
            .and(body_json(serde_json::json!({ "enabled": true })))
            .respond_with(
                ResponseTemplate::new(409)
                    .set_body_string("device is startup-excluded and requires a restart"),
            )
            .mount(&server)
            .await;

        let error = MoldClient::new(&server.uri())
            .set_device_enabled("cuda:device-1", true)
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("409 Conflict"));
        assert!(message.contains("startup-excluded"));
        assert!(message.contains("requires a restart"));
    }

    #[tokio::test]
    async fn set_device_enabled_encodes_the_stable_id_and_sends_auth() {
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/api/devices/cuda%3Aparent%2Fgpu"))
            .and(header("x-api-key", "sekrit"))
            .and(body_json(serde_json::json!({ "enabled": false })))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "cuda:parent/gpu",
                "backend": "cuda",
                "ordinal": 1,
                "device_kind": "full_gpu",
                "name": "GPU 1",
                "memory": {},
                "telemetry": {},
                "desired_enabled": false,
                "admin_state": "draining",
                "health": "healthy",
                "activity": "generating",
                "schedulable": false,
                "loaded_models": [],
                "planned_work_ids": []
            })))
            .mount(&server)
            .await;

        let device = MoldClient::with_api_key(&server.uri(), "sekrit".to_string())
            .set_device_enabled("cuda:parent/gpu", false)
            .await
            .unwrap();
        assert_eq!(device.id, "cuda:parent/gpu");
        assert_eq!(device.admin_state, crate::DeviceAdminState::Draining);
        assert!(!device.desired_enabled);
    }

    // ── Queue endpoints ──────────────────────────────────────────────────

    #[tokio::test]
    async fn list_queue_parses_the_wrapped_entries_listing() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entries": [
                    {
                        "id": "job-1",
                        "model": "flux-dev:q8",
                        "state": "running",
                        "started_at_unix_ms": 1_711_305_600_000_u64,
                        "position": 0,
                        "gpu": 0
                    },
                    {
                        "id": "job-2",
                        "model": "sdxl:q8",
                        "state": "queued",
                        "started_at_unix_ms": 1_711_305_601_000_u64,
                        "position": 1
                    }
                ]
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let listing = client.list_queue().await.unwrap();

        assert_eq!(listing.entries.len(), 2);
        assert_eq!(listing.entries[0].id, "job-1");
        assert_eq!(listing.entries[0].state, "running");
        assert_eq!(listing.entries[0].gpu, Some(0));
        assert_eq!(listing.entries[1].state, "queued");
        assert_eq!(listing.entries[1].gpu, None);
        assert_eq!(listing.entries[1].position, 1);
    }

    #[tokio::test]
    async fn list_queue_sends_the_api_key_header() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .and(header("x-api-key", "sekrit"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "entries": [] })),
            )
            .mount(&server)
            .await;

        let client = MoldClient::with_api_key(&server.uri(), "sekrit".to_string());
        let listing = client.list_queue().await.unwrap();
        assert!(listing.entries.is_empty());
    }

    #[tokio::test]
    async fn cancel_queue_job_succeeds_on_no_content() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/queue/job-1"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        client.cancel_queue_job("job-1").await.unwrap();
    }

    #[tokio::test]
    async fn cancel_queue_job_surfaces_the_409_body_for_running_jobs() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/queue/job-1"))
            .respond_with(ResponseTemplate::new(409).set_body_json(serde_json::json!({
                "error": "queue job job-1 is already running; only queued jobs can be cancelled"
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let err = client.cancel_queue_job("job-1").await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("409"), "status missing from error: {msg}");
        assert!(
            msg.contains("already running"),
            "body text missing from error: {msg}"
        );
    }

    #[test]
    fn qwen_edit_lora_fallback_uses_qwen_image_catalog_family() {
        assert_eq!(
            lora_family_for_model_filter("qwen-image-edit-2511:q4"),
            Some("qwen-image".to_string())
        );
    }

    #[test]
    fn api_error_detail_extracts_actionable_server_json() {
        assert_eq!(
            api_error_detail(
                r#"{"error":"Qwen Image Edit needs a Target image.","code":"VALIDATION_ERROR"}"#
            ),
            "Qwen Image Edit needs a Target image."
        );
        assert_eq!(
            api_error_detail("plain validation failure"),
            "plain validation failure"
        );
    }

    #[test]
    fn parse_sse_event_joins_multiline_data() {
        let (event_type, data) =
            parse_sse_event("event: progress\ndata: {\"a\":1}\ndata: {\"b\":2}");
        assert_eq!(event_type, "progress");
        assert_eq!(data, "{\"a\":1}\n{\"b\":2}");
    }

    #[test]
    fn next_sse_event_supports_crlf_delimiters() {
        let mut buffer = "event: progress\r\ndata: {\"ok\":true}\r\n\r\nrest".to_string();
        let event = next_sse_event(&mut buffer).expect("expected one event");
        assert!(event.contains("event: progress"));
        assert_eq!(buffer, "rest");
    }

    // ── Audio header parsing tests ───────────────────────────────────────

    #[test]
    fn parse_audio_headers_returns_none_for_a_still_or_a_clip() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert!(parse_audio_headers(&headers).is_none());

        // A clip that happens to carry an audio track is still a video: only
        // the audio-only headers may promote a response to `AudioData`.
        headers.insert("x-mold-video-frames", "97".parse().unwrap());
        headers.insert("x-mold-video-audio-sample-rate", "48000".parse().unwrap());
        assert!(parse_audio_headers(&headers).is_none());
    }

    #[test]
    fn parse_audio_headers_reads_the_audio_only_shape() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-audio-format", "wav".parse().unwrap());
        headers.insert("x-mold-audio-sample-rate", "24000".parse().unwrap());
        headers.insert("x-mold-audio-channels", "2".parse().unwrap());
        headers.insert("x-mold-audio-duration-ms", "5010".parse().unwrap());
        headers.insert("x-mold-audio-thumbnail-width", "640".parse().unwrap());
        headers.insert("x-mold-audio-thumbnail-height", "360".parse().unwrap());

        let meta = parse_audio_headers(&headers).expect("should detect audio");
        assert_eq!(meta.format, Some(OutputFormat::Wav));
        assert_eq!(meta.sample_rate, 24_000);
        assert_eq!(meta.channels, 2);
        assert_eq!(meta.duration_ms, 5_010);
        assert_eq!(meta.thumbnail_width, 640);
        assert_eq!(meta.thumbnail_height, 360);
    }

    /// An older server that grew the sample-rate header before the rest must
    /// still produce a usable response rather than none at all.
    #[test]
    fn parse_audio_headers_defaults_the_optional_fields() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-audio-sample-rate", "48000".parse().unwrap());
        let meta = parse_audio_headers(&headers).expect("should detect audio");
        assert_eq!(
            meta.format, None,
            "the caller falls back to an audio format"
        );
        assert_eq!(meta.sample_rate, 48_000);
        assert_eq!(meta.channels, 1);
        assert_eq!(meta.duration_ms, 0);
        assert_eq!(meta.thumbnail_width, 0);
        assert_eq!(meta.thumbnail_height, 0);
    }

    // ── Video header parsing tests ───────────────────────────────────────

    #[test]
    fn parse_video_headers_returns_none_without_frames() {
        let headers = reqwest::header::HeaderMap::new();
        assert!(parse_video_headers(&headers).is_none());
    }

    #[test]
    fn parse_video_headers_returns_some_with_frames() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-video-frames", "33".parse().unwrap());
        headers.insert("x-mold-video-fps", "12".parse().unwrap());
        headers.insert("x-mold-video-width", "832".parse().unwrap());
        headers.insert("x-mold-video-height", "480".parse().unwrap());
        headers.insert("x-mold-video-pipeline", "two-stage".parse().unwrap());

        let meta = parse_video_headers(&headers).expect("should detect video");
        assert_eq!(meta.frames, 33);
        assert_eq!(meta.fps, 12);
        assert_eq!(meta.width, Some(832));
        assert_eq!(meta.height, Some(480));
        assert_eq!(meta.pipeline, Some(crate::Ltx2PipelineMode::TwoStage));
        assert!(!meta.has_audio);
        assert!(meta.duration_ms.is_none());
    }

    #[test]
    fn parse_video_headers_with_audio_metadata() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-video-frames", "17".parse().unwrap());
        headers.insert("x-mold-video-fps", "24".parse().unwrap());
        headers.insert("x-mold-video-has-audio", "1".parse().unwrap());
        headers.insert("x-mold-video-duration-ms", "2750".parse().unwrap());
        headers.insert("x-mold-video-audio-sample-rate", "44100".parse().unwrap());
        headers.insert("x-mold-video-audio-channels", "2".parse().unwrap());

        let meta = parse_video_headers(&headers).expect("should detect video");
        assert_eq!(meta.frames, 17);
        assert_eq!(meta.fps, 24);
        assert!(meta.has_audio);
        assert_eq!(meta.duration_ms, Some(2750));
        assert_eq!(meta.audio_sample_rate, Some(44100));
        assert_eq!(meta.audio_channels, Some(2));
    }

    #[test]
    fn parse_video_headers_fps_defaults_to_24() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-video-frames", "10".parse().unwrap());
        // No fps header — should default to 24

        let meta = parse_video_headers(&headers).expect("should detect video");
        assert_eq!(meta.fps, 24);
    }

    #[test]
    fn parse_video_headers_has_audio_absent_is_false() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-video-frames", "10".parse().unwrap());
        // No has-audio header

        let meta = parse_video_headers(&headers).expect("should detect video");
        assert!(!meta.has_audio);
    }
}
