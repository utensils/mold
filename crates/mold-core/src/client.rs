use crate::chain::{ChainProgressEvent, ChainRequest};
use crate::chain_job::{
    ChainJobDetail, ChainJobListing, ChainJobSummary, CreateChainJobResponse, GcOutcome,
    RetakeRequest,
};
use crate::error::MoldError;
use crate::queue_progress::QueueJobProgress;
use crate::types::{
    AudioData, Collection, CollectionCreateRequest, CollectionDetail, CollectionItemsRequest,
    CollectionUpdateRequest, DeviceState, EmptyTrashResult, ExpandRequest, ExpandResponse,
    GalleryBulkMutationRequest, GalleryBulkMutationResult, GalleryImage, GalleryOrganizeRequest,
    GalleryPatchRequest, GenerateRequest, GenerateResponse, GenerationBatchAdmissionRequest,
    GenerationBatchStatus, GenerationBatchStatusRequest, GenerationBatchStatusResponse,
    GenerationRetryRequest, ImageData, LoraInfo, MeshData, ModelInfo, ModelInfoExtended,
    OutputFormat, QueueListingWire, ReferenceUploadCompleteResponse, ReferenceUploadSessionRequest,
    ReferenceUploadSessionResponse, ServerStatus, SseCompleteEvent, SseErrorEvent,
    SseProgressEvent, TagCount, TagRenameRequest, TrashFilenamesRequest, TrashSweepResult,
    VideoData,
};
use anyhow::{Context, Result};
use base64::Engine as _;
use reqwest::{Client, StatusCode};
use std::io::{Seek, SeekFrom};
use std::path::Path;
use tokio_util::io::ReaderStream;

/// Body for `POST /api/models/pull`.
///
/// `accept_licenses` is omitted when empty so requests to older servers —
/// which reject unknown fields on some builds and would otherwise see a field
/// they never asked for — stay byte-identical to what they got before.
fn pull_body(
    model: &str,
    accept_licenses: &[crate::types::LicenseAcceptance],
) -> serde_json::Value {
    let mut body = serde_json::json!({ "model": model });
    if !accept_licenses.is_empty() {
        body["accept_licenses"] = serde_json::json!(accept_licenses);
    }
    body
}

const REFERENCE_UPLOAD_HANDLE_HEADER: &str = "x-mold-reference-upload";
const REFERENCE_UPLOAD_SESSION_HEADER: &str = "x-mold-reference-upload-session";

#[derive(Clone)]
pub struct MoldClient {
    base_url: String,
    client: Client,
    api_key_configured: bool,
}

fn require_direct_singleton(req: &GenerateRequest) -> Result<()> {
    if req.batch_size != 1 {
        return Err(MoldError::Validation(
            "direct generation accepts batch_size = 1; use durable generation-batch admission for multiple outputs"
                .to_string(),
        )
        .into());
    }
    Ok(())
}

/// Explain a TERMINAL error frame in which the host said it kept the job.
///
/// This is the graceful-restart path — an operator restarting the server —
/// and it is the scenario the durable queue exists for, so it must never read
/// as an ordinary failure. The frame is per-job authoritative (the server sets
/// `retained` only for a job it actually journalled), so no probe is consulted
/// and none is needed. The server's own message is preserved as the cause.
fn retained_frame_error(message: &str, job_id: Option<&str>, host: &str) -> anyhow::Error {
    let note = match job_id {
        Some(id) if !id.is_empty() => {
            format!("job {id} is retained on {host} and will finish there")
        }
        _ => format!("this generation is retained on {host} and will finish there"),
    };
    tracing::warn!(job_id = job_id.unwrap_or(""), host = %host, "{note}");
    anyhow::anyhow!("server error: {message}").context(note)
}

/// Explain a stream that ended without a terminal event.
///
/// The host journalled this job before it acknowledged it — that is the only
/// admission path there is — so it runs whether or not a client is attached,
/// and replays across a restart what it could not finish. The job the caller
/// just lost sight of is still going to render there. Say so.
///
/// The original transport error is preserved as the cause, so
/// [`MoldClient::is_connection_error`] still reports `false` for a mid-body
/// death and the CLI surfaces it instead of silently re-rendering locally.
fn annotate_lost_stream(err: anyhow::Error, job_id: Option<&str>, host: &str) -> anyhow::Error {
    let Some(job_id) = job_id else {
        return err;
    };
    let note = format!("job {job_id} is retained on {host} and will finish there");
    tracing::warn!(job_id = %job_id, host = %host, "{note}");
    err.context(note)
}

/// Terminal outcome of one durable chain job, as observed from its event
/// stream.
#[derive(Debug, Clone)]
pub struct ChainJobOutcome {
    pub state: crate::chain_job::ChainJobState,
    pub error: Option<String>,
    /// Gallery filename of the stitched print, from the `Finalized` event.
    pub output: Option<String>,
}

impl MoldClient {
    pub fn new(base_url: &str) -> Self {
        let (client, api_key_configured) = build_client(None);
        Self {
            base_url: normalize_host(base_url),
            client,
            api_key_configured,
        }
    }

    /// Create a client with an explicit API key for authentication.
    pub fn with_api_key(base_url: &str, api_key: String) -> Self {
        let (client, api_key_configured) = build_client(Some(&api_key));
        Self {
            base_url: normalize_host(base_url),
            client,
            api_key_configured,
        }
    }

    pub fn from_env() -> Self {
        let base_url =
            std::env::var("MOLD_HOST").unwrap_or_else(|_| "http://localhost:7680".to_string());
        let api_key = std::env::var("MOLD_API_KEY").ok().filter(|k| !k.is_empty());
        let (client, api_key_configured) = build_client(api_key.as_deref());
        Self {
            base_url: normalize_host(&base_url),
            client,
            api_key_configured,
        }
    }

    /// Whether this client actually installed a non-empty API-key header.
    pub fn has_api_key(&self) -> bool {
        self.api_key_configured
    }

    /// Create a request-bound MiniMax H3 reference upload session.
    ///
    /// The server applies authentication and legal activation before it
    /// allocates staging. Error bodies (including HTTP 451 policy details) are
    /// retained verbatim for callers instead of being collapsed to a generic
    /// reqwest status error.
    pub async fn create_reference_upload_session(
        &self,
        request: &ReferenceUploadSessionRequest,
    ) -> Result<ReferenceUploadSessionResponse> {
        let mut wire_request = request.clone();
        wire_request.request =
            crate::prompt_text::protect_generate_request_for_wire(&request.request);
        let response = self
            .client
            .post(format!(
                "{}/api/generate/reference-upload-sessions",
                self.base_url
            ))
            .json(&wire_request)
            .send()
            .await?;
        Ok(error_for_status_with_body(response)
            .await?
            .json::<ReferenceUploadSessionResponse>()
            .await?)
    }

    /// Stream one local reference file through a one-use bearer handle.
    ///
    /// The handle stays in a header, while the opened file's exact length is
    /// sent as `Content-Length`. `ReaderStream` keeps large video references
    /// out of process-sized buffers; the server independently hashes, probes,
    /// and checks the declared descriptor before admission.
    pub async fn upload_reference_file(
        &self,
        handle: &str,
        path: &Path,
        mime_type: &str,
    ) -> Result<ReferenceUploadCompleteResponse> {
        let file = tokio::fs::File::open(path)
            .await
            .with_context(|| format!("failed to open reference '{}'", path.display()))?;
        let metadata = file
            .metadata()
            .await
            .with_context(|| format!("failed to inspect reference '{}'", path.display()))?;
        anyhow::ensure!(
            metadata.is_file() && metadata.len() > 0,
            "reference upload source is not a non-empty regular file: {}",
            path.display()
        );
        self.upload_reference_body(
            handle,
            mime_type,
            metadata.len(),
            reqwest::Body::wrap_stream(ReaderStream::new(file)),
        )
        .await
    }

    /// Stream an already-open reference file through a one-use bearer handle.
    ///
    /// Keeping the descriptor probe and upload tied to clones of the same file
    /// descriptor prevents a path or symlink replacement between inspection and
    /// upload. The local pathname never enters the request or this method's
    /// errors.
    pub async fn upload_reference_open_file(
        &self,
        handle: &str,
        mut file: std::fs::File,
        mime_type: &str,
    ) -> Result<ReferenceUploadCompleteResponse> {
        let metadata = file.metadata().context("failed to inspect reference")?;
        anyhow::ensure!(
            metadata.is_file() && metadata.len() > 0,
            "reference upload source is not a non-empty regular file"
        );
        file.seek(SeekFrom::Start(0))
            .context("failed to rewind reference")?;
        let file = tokio::fs::File::from_std(file);
        self.upload_reference_body(
            handle,
            mime_type,
            metadata.len(),
            reqwest::Body::wrap_stream(ReaderStream::new(file)),
        )
        .await
    }

    /// Upload one bounded in-memory attachment through a one-use bearer handle.
    ///
    /// Discord already owns attachment bytes in memory after its download-size
    /// gate. Taking ownership here guarantees the uploaded body is exactly the
    /// body that was hashed and probed, without a second URL fetch.
    pub async fn upload_reference_bytes(
        &self,
        handle: &str,
        bytes: Vec<u8>,
        mime_type: &str,
    ) -> Result<ReferenceUploadCompleteResponse> {
        anyhow::ensure!(!bytes.is_empty(), "reference upload source is empty");
        let length = u64::try_from(bytes.len()).context("reference upload is too large")?;
        self.upload_reference_body(handle, mime_type, length, reqwest::Body::from(bytes))
            .await
    }

    async fn upload_reference_body(
        &self,
        handle: &str,
        mime_type: &str,
        content_length: u64,
        body: reqwest::Body,
    ) -> Result<ReferenceUploadCompleteResponse> {
        let response = self
            .client
            .put(format!("{}/api/generate/reference-upload", self.base_url))
            .header(REFERENCE_UPLOAD_HANDLE_HEADER, handle)
            .header(reqwest::header::CONTENT_TYPE, mime_type)
            .header(reqwest::header::CONTENT_LENGTH, content_length)
            .body(body)
            .send()
            .await?;
        Ok(error_for_status_with_body(response)
            .await?
            .json::<ReferenceUploadCompleteResponse>()
            .await?)
    }

    /// Best-effort cleanup for an unconsumed reference upload session.
    pub async fn cancel_reference_upload_session(&self, handle: &str) -> Result<()> {
        let response = self
            .client
            .delete(format!(
                "{}/api/generate/reference-upload-sessions",
                self.base_url
            ))
            .header(REFERENCE_UPLOAD_SESSION_HEADER, handle)
            .send()
            .await?;
        error_for_status_with_body(response).await?;
        Ok(())
    }

    /// Generate an image. Returns raw image bytes (PNG or JPEG).
    /// The server returns raw bytes, not JSON — callers are responsible for
    /// writing the bytes to disk or further processing.
    pub async fn generate_raw(&self, req: &GenerateRequest) -> Result<Vec<u8>> {
        require_direct_singleton(req)?;
        let wire_req = crate::prompt_text::protect_generate_request_for_wire(req);
        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&wire_req)
            .send()
            .await?;
        let bytes = require_direct_media_response(response)
            .await?
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
        require_direct_singleton(&req)?;
        let fallback_seed = req.seed.unwrap_or(0);
        let width = req.width;
        let height = req.height;
        let model = req.model.clone();
        let format = req.resolved_output_format();
        let wire_req = crate::prompt_text::protect_generate_request_for_wire(&req);

        let start = std::time::Instant::now();
        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&wire_req)
            .send()
            .await?;
        let resp = require_direct_media_response(resp).await?;

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

        // Probe order is mesh, then audio, then video, narrowest first. Each
        // of these artifacts is missing whatever the next probe keys on, so a
        // wider probe running first would fall through to the image branch and
        // hand the caller non-raster bytes labelled as a picture.
        let mesh_meta = parse_mesh_headers(resp.headers());
        let audio_meta = parse_audio_headers(resp.headers());
        // Detect video response via x-mold-video-frames header
        let video_meta = parse_video_headers(resp.headers());
        // Advisories about the accepted request (retimings, a filing the host
        // could not apply). Read here so every branch below carries them.
        let request_warnings = parse_request_warnings(resp.headers());

        let data = resp.bytes().await?.to_vec();
        let generation_time_ms = start.elapsed().as_millis() as u64;

        if let Some(meta) = mesh_meta {
            return Ok(GenerateResponse {
                mesh: Some(MeshData {
                    data,
                    format: meta.format.unwrap_or(if format.is_mesh() {
                        format
                    } else {
                        OutputFormat::Glb
                    }),
                    vertex_count: meta.vertex_count,
                    face_count: meta.face_count,
                    bounds_min: meta.bounds_min,
                    bounds_max: meta.bounds_max,
                    textured: meta.textured,
                    // The poster cannot ride along in a body that is already
                    // the GLB. Only its recorded size travels.
                    poster: Vec::new(),
                    poster_width: meta.poster_width,
                    poster_height: meta.poster_height,
                }),
                images: Vec::new(),
                video: None,
                audio: None,
                generation_time_ms,
                model,
                seed_used,
                gpu,
                request_warnings,
            });
        }

        if let Some(meta) = audio_meta {
            return Ok(GenerateResponse {
                mesh: None,
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
                request_warnings,
            });
        }

        let video = video_meta.map(|meta| VideoData {
            video_only: meta.video_only,
            attention_path: meta.attention_path,
            int8_arm: meta.int8_arm,
            data: data.clone(),
            format,
            width: meta.width.unwrap_or(width),
            height: meta.height.unwrap_or(height),
            frames: meta.frames,
            fps: meta.fps,
            pipeline: meta.pipeline,
            pipeline_provenance_sha256: meta.pipeline_provenance_sha256,
            source_preprocessing: meta.source_preprocessing,
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
            mesh: None,
            images,
            generation_time_ms,
            model,
            seed_used,
            video,
            gpu,
            request_warnings,
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

    /// Generate one output via the durable `/api/generate/stream` facade,
    /// receiving progress events.
    ///
    /// A host that does not serve the route is an error naming what it lacks:
    /// this is the only singleton path, so there is nothing to degrade to.
    pub async fn generate_stream(
        &self,
        req: &GenerateRequest,
        progress_tx: tokio::sync::mpsc::UnboundedSender<SseProgressEvent>,
    ) -> Result<GenerateResponse> {
        require_direct_singleton(req)?;
        let wire_req = crate::prompt_text::protect_generate_request_for_wire(req);
        let mut resp = self
            .client
            .post(format!("{}/api/generate/stream", self.base_url))
            .json(&wire_req)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                // Axum returns an empty 404 for an unmatched route.
                anyhow::bail!("{} does not serve POST /api/generate/stream", self.base_url);
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

        // Advisories ride the response headers, which arrive before the first
        // SSE frame — captured now because `resp` is consumed chunk by chunk
        // below and the headers are not reachable from a completion event.
        let request_warnings = parse_request_warnings(resp.headers());

        // Parse SSE events from chunked response body
        let mut buffer = String::new();
        // The server-assigned job id, latched from the first `queued` event.
        // It exists only once the job is admitted — and an admitted job is a
        // journalled one.
        let mut retained: Option<String> = None;
        loop {
            let chunk = match resp.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => break,
                Err(err) => {
                    return Err(annotate_lost_stream(
                        anyhow::Error::new(err),
                        retained.as_deref(),
                        &self.base_url,
                    ));
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (event_type, data) = parse_sse_event(&event_text);
                match event_type.as_str() {
                    "progress" => {
                        if let Ok(p) = serde_json::from_str::<SseProgressEvent>(&data) {
                            if let SseProgressEvent::Queued { id, .. } = &p {
                                if !id.is_empty() && retained.is_none() {
                                    retained = Some(id.clone());
                                }
                            }
                            let _ = progress_tx.send(p);
                        }
                    }
                    "complete" => {
                        let complete: SseCompleteEvent = serde_json::from_str(&data)?;
                        // The render's own advisories live here, not in the
                        // headers captured above — they were written before
                        // the job ran. Both channels, or the caller sees only
                        // half of what the server told it.
                        let request_warnings = merge_completion_warnings(
                            &request_warnings,
                            &complete.request_warnings,
                        );
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

                        // Narrowest probe first: mesh, then audio, then
                        // video. A mesh has no sample rate and no frames and
                        // an audio print has no frames, so running a wider
                        // probe first drops each of them into the image branch
                        // and hands the caller non-raster bytes labelled as a
                        // picture.
                        if let Some(vertices) = complete.mesh_vertices {
                            let poster = complete
                                .mesh_poster
                                .as_deref()
                                .and_then(|s| b64.decode(s).ok())
                                .unwrap_or_default();
                            return Ok(GenerateResponse {
                                images: Vec::new(),
                                video: None,
                                audio: None,
                                mesh: Some(MeshData {
                                    data: payload,
                                    format: complete.format,
                                    vertex_count: vertices,
                                    face_count: complete.mesh_faces.unwrap_or(0),
                                    bounds_min: complete.mesh_bounds_min.unwrap_or([0.0; 3]),
                                    bounds_max: complete.mesh_bounds_max.unwrap_or([0.0; 3]),
                                    textured: complete.mesh_textured,
                                    poster,
                                    // `width`/`height` describe the poster on
                                    // a mesh event, exactly as they describe
                                    // the waveform on an audio one.
                                    poster_width: complete.width,
                                    poster_height: complete.height,
                                }),
                                generation_time_ms: complete.generation_time_ms,
                                model,
                                seed_used: complete.seed_used,
                                gpu: complete.gpu,
                                request_warnings,
                            });
                        }

                        if let Some(sample_rate) = complete.audio_sample_rate {
                            let thumbnail = complete
                                .audio_thumbnail
                                .as_deref()
                                .and_then(|s| b64.decode(s).ok())
                                .unwrap_or_default();
                            return Ok(GenerateResponse {
                                images: Vec::new(),
                                video: None,
                                mesh: None,
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
                                request_warnings,
                            });
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
                                video_only: complete
                                    .metadata
                                    .as_ref()
                                    .and_then(|metadata| metadata.video_only),
                                attention_path: complete
                                    .metadata
                                    .as_ref()
                                    .and_then(|metadata| metadata.attention_path.clone()),
                                int8_arm: complete
                                    .metadata
                                    .as_ref()
                                    .and_then(|metadata| metadata.int8_arm.clone()),
                                data: payload,
                                format: complete.format,
                                width: complete.width,
                                height: complete.height,
                                frames,
                                fps,
                                pipeline: complete.metadata.as_ref().and_then(|m| m.pipeline),
                                pipeline_provenance_sha256: complete.metadata.as_ref().and_then(
                                    |metadata| metadata.pipeline_provenance_sha256.clone(),
                                ),
                                source_preprocessing: complete
                                    .metadata
                                    .as_ref()
                                    .and_then(|metadata| metadata.source_preprocessing.clone()),
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

                        return Ok(GenerateResponse {
                            audio: None,
                            mesh: None,
                            images,
                            generation_time_ms: complete.generation_time_ms,
                            model,
                            seed_used: complete.seed_used,
                            video,
                            gpu: complete.gpu,
                            request_warnings,
                        });
                    }
                    "error" => {
                        let error: SseErrorEvent = serde_json::from_str(&data)?;
                        let job_id = retained.take();
                        if error.retained {
                            return Err(retained_frame_error(
                                &error.message,
                                job_id.as_deref(),
                                &self.base_url,
                            ));
                        }
                        // Durable admission accepts before it resolves a
                        // checkpoint, so "this model is not here" arrives as a
                        // terminal frame where the attached path answered 404.
                        // Re-typed here so `classify_generate_error` still
                        // reaches `PullModelAndRetry` and `mold run` still
                        // offers the pull.
                        if error.code.as_deref().is_some_and(|code| {
                            code == crate::types::SSE_ERROR_CODE_MODEL_NOT_FOUND
                                || code == crate::types::SSE_ERROR_CODE_UNKNOWN_MODEL
                        }) {
                            return Err(MoldError::ModelNotFound(error.message).into());
                        }
                        // A definitive server failure promises nothing.
                        anyhow::bail!("server error: {}", error.message);
                    }
                    _ => {}
                }
            }
        }

        Err(annotate_lost_stream(
            anyhow::anyhow!("SSE stream ended without complete event"),
            retained.as_deref(),
            &self.base_url,
        ))
    }

    /// Follow one durable chain job to settlement, forwarding its stage
    /// progress as [`ChainProgressEvent`] so the CLI and TUI renderers see the
    /// same shape they always have.
    ///
    /// The stream opens with a snapshot, so a caller that attaches after some
    /// stages have already run still learns the stage count and where the job
    /// is — which the old fire-and-forget chain endpoint could not do.
    pub async fn stream_chain_job_events(
        &self,
        job_id: &str,
        progress_tx: tokio::sync::mpsc::UnboundedSender<ChainProgressEvent>,
    ) -> Result<ChainJobOutcome> {
        use crate::chain_job::{ChainJobEvent, ChainJobState};

        let mut resp = self
            .client
            .get(format!(
                "{}/api/chain-jobs/{}/events",
                self.base_url, job_id
            ))
            .send()
            .await?;
        if resp.status().is_client_error() || resp.status().is_server_error() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("server error {status}: {body}");
        }

        let mut outcome = ChainJobOutcome {
            state: ChainJobState::Running,
            error: None,
            output: None,
        };
        let mut buffer = String::new();
        while let Some(chunk) = resp.chunk().await? {
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(event_text) = next_sse_event(&mut buffer) {
                let (_, data) = parse_sse_event(&event_text);
                let Ok(event) = serde_json::from_str::<ChainJobEvent>(&data) else {
                    continue;
                };
                match event {
                    ChainJobEvent::Snapshot { job } => {
                        outcome.state = job.summary.state;
                        outcome.error = job.summary.error.clone();
                        let _ = progress_tx.send(ChainProgressEvent::ChainStart {
                            stage_count: job.summary.stage_count,
                            estimated_total_frames: job
                                .stages
                                .iter()
                                .filter_map(|stage| stage.frames_emitted)
                                .sum(),
                        });
                        if crate::chain_job::settled(job.summary.state) {
                            // A job that settled before we subscribed carries
                            // its print in the manifest, not in a frame.
                            outcome.output = job
                                .finalizes
                                .last()
                                .and_then(|record| record.gallery_filename.clone());
                            return Ok(outcome);
                        }
                    }
                    ChainJobEvent::StageStart { stage_idx } => {
                        let _ = progress_tx.send(ChainProgressEvent::StageStart { stage_idx });
                    }
                    ChainJobEvent::DenoiseStep {
                        stage_idx,
                        step,
                        total,
                    } => {
                        let _ = progress_tx.send(ChainProgressEvent::DenoiseStep {
                            stage_idx,
                            step,
                            total,
                        });
                    }
                    ChainJobEvent::StageDone {
                        stage_idx,
                        frames_emitted,
                        ..
                    } => {
                        let _ = progress_tx.send(ChainProgressEvent::StageDone {
                            stage_idx,
                            frames_emitted,
                        });
                    }
                    ChainJobEvent::Finalizing { total_frames } => {
                        let _ = progress_tx.send(ChainProgressEvent::Stitching { total_frames });
                    }
                    ChainJobEvent::Finalized {
                        gallery_filename, ..
                    } => outcome.output = gallery_filename,
                    ChainJobEvent::StateChanged { state, error } => {
                        outcome.state = state;
                        if error.is_some() {
                            outcome.error = error;
                        }
                        if crate::chain_job::settled(state) {
                            return Ok(outcome);
                        }
                    }
                    ChainJobEvent::Yielded { .. } => {}
                }
            }
        }
        // The runner closes the broadcast when the job settles, so a stream
        // that ends without a terminal frame means the state changed while
        // nobody was subscribed. Ask.
        outcome.state = self
            .get_chain_job(job_id)
            .await
            .map(|detail| detail.summary.state)
            .unwrap_or(outcome.state);
        Ok(outcome)
    }

    pub async fn create_chain_job(&self, req: &ChainRequest) -> Result<CreateChainJobResponse> {
        let wire_req = crate::prompt_text::protect_chain_request_for_wire(req);
        let resp = self
            .client
            .post(format!("{}/api/chain-jobs", self.base_url))
            .json(&wire_req)
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
        let wire_req = crate::prompt_text::protect_retake_request_for_wire(req);
        let resp = self
            .client
            .post(format!(
                "{}/api/chain-jobs/{}/retake",
                self.base_url,
                encode_path_segment(id)
            ))
            .json(&wire_req)
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
        self.pull_model_accepting(model, &[]).await
    }

    /// Pull a model, recording third-party license acceptances ON THE SERVER
    /// first.
    ///
    /// Acceptance is per Mold data root, so a client that recorded it locally
    /// has told the wrong machine. The ids ride the request instead, and the
    /// server writes them into its own root before the pull starts.
    pub async fn pull_model_accepting(
        &self,
        model: &str,
        accept_licenses: &[crate::types::LicenseAcceptance],
    ) -> Result<String> {
        let resp = self
            .client
            .post(format!("{}/api/models/pull", self.base_url))
            .json(&pull_body(model, accept_licenses))
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        Ok(resp)
    }

    /// List this server's third-party licenses and their acceptance state.
    pub async fn list_licenses(&self) -> Result<Vec<crate::types::ThirdPartyLicenseStatus>> {
        let listing: crate::types::LicenseListing = self
            .client
            .get(format!("{}/api/licenses", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(listing.licenses)
    }

    /// Read-only dependency and device plan for an exact generation request.
    /// Newer clients use the additive pending-license metadata to pause for
    /// consent before admission starts any download.
    pub async fn preview_generation_placement(
        &self,
        request: crate::types::GenerateRequest,
        copies: u32,
    ) -> Result<crate::types::GenerationPlacementPreview> {
        let request = crate::prompt_text::protect_generate_request_for_wire(&request);
        let preview = self
            .client
            .post(format!("{}/api/generate/placement-preview", self.base_url))
            .json(&crate::types::GenerationPlacementPreviewRequest { request, copies })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(preview)
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
        self.pull_model_stream_accepting(model, &[], progress_tx)
            .await
    }

    /// [`Self::pull_model_stream`], recording license acceptances on the
    /// server before the pull starts. See [`Self::pull_model_accepting`].
    pub async fn pull_model_stream_accepting(
        &self,
        model: &str,
        accept_licenses: &[crate::types::LicenseAcceptance],
        progress_tx: tokio::sync::mpsc::UnboundedSender<SseProgressEvent>,
    ) -> Result<()> {
        let mut resp = self
            .client
            .post(format!("{}/api/models/pull", self.base_url))
            .header("Accept", "text/event-stream")
            .json(&pull_body(model, accept_licenses))
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

    /// Durably admit one ordered set of independently executing singleton
    /// requests. `client_batch_id` is the caller's idempotency key.
    pub async fn admit_generation_batch(
        &self,
        request: &GenerationBatchAdmissionRequest,
    ) -> Result<GenerationBatchStatus> {
        let response = self
            .client
            .post(format!("{}/api/generation-batches", self.base_url))
            .json(request)
            .send()
            .await?;
        Ok(error_for_status_with_body(response)
            .await?
            .json::<GenerationBatchStatus>()
            .await?)
    }

    /// Resolve an ambiguous admission response using the caller-owned
    /// idempotency key. A genuine 404 is represented as `None`.
    pub async fn generation_batch_by_client_id(
        &self,
        client_batch_id: &str,
    ) -> Result<Option<GenerationBatchStatus>> {
        let response = self
            .client
            .get(format!(
                "{}/api/generation-batches/by-client/{}",
                self.base_url,
                encode_path_segment(client_batch_id)
            ))
            .send()
            .await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(
            error_for_status_with_body(response)
                .await?
                .json::<GenerationBatchStatus>()
                .await?,
        ))
    }

    /// Read one durable generation batch by server-assigned id.
    pub async fn generation_batch(&self, id: &str) -> Result<Option<GenerationBatchStatus>> {
        let response = self
            .client
            .get(format!(
                "{}/api/generation-batches/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(
            error_for_status_with_body(response)
                .await?
                .json::<GenerationBatchStatus>()
                .await?,
        ))
    }

    /// Reconcile a bounded set of durable generation batches in one request.
    pub async fn generation_batch_statuses(
        &self,
        request: &GenerationBatchStatusRequest,
    ) -> Result<GenerationBatchStatusResponse> {
        let response = self
            .client
            .post(format!("{}/api/generation-batches/status", self.base_url))
            .json(request)
            .send()
            .await?;
        Ok(error_for_status_with_body(response)
            .await?
            .json::<GenerationBatchStatusResponse>()
            .await?)
    }

    /// Return a retryable held generation child to the durable queue.
    pub async fn retry_queue_job(&self, request: &GenerationRetryRequest) -> Result<()> {
        let response = self
            .client
            .post(format!(
                "{}/api/queue/{}/retry",
                self.base_url,
                encode_path_segment(&request.job_id)
            ))
            .json(request)
            .send()
            .await?;
        error_for_status_with_body(response).await?;
        Ok(())
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

    /// Snapshot the server's live generation window (`GET /api/queue`).
    ///
    /// Current hosts return the capacity-sized live window; older hosts that
    /// omit `queue_capacity` retain the legacy full-list behavior. Entries are wire-shaped twins of
    /// the server's `job_registry::JobEntry`; see [`QueueListingWire`] for
    /// the forward-compat rules (plain-string `state`, defaulted additive
    /// fields), which let this client talk to both older and newer servers.
    pub async fn list_queue(&self) -> Result<QueueListingWire> {
        let status = self.server_status().await?;
        self.list_queue_for_capacity(status.queue_capacity).await
    }

    /// Read a queue page using capacity already observed from this exact host.
    /// `None` is the explicit legacy-host fallback; current callers must pass
    /// the positive `queue_capacity` advertised by `/api/status`.
    pub async fn list_queue_for_capacity(
        &self,
        queue_capacity: Option<usize>,
    ) -> Result<QueueListingWire> {
        match queue_capacity {
            Some(0) => anyhow::bail!("server reported an invalid zero queue capacity"),
            Some(limit) => self.list_queue_page(limit, None).await,
            None => self.fetch_queue_listing(None, None).await,
        }
    }

    /// Continue an explicitly requested durable queue snapshot. This is for
    /// user-driven pagination, never for health polling.
    pub async fn list_queue_page(
        &self,
        limit: usize,
        cursor: Option<&str>,
    ) -> Result<QueueListingWire> {
        if limit == 0 {
            anyhow::bail!("queue page limit must be positive");
        }
        self.fetch_queue_listing(Some(limit), cursor).await
    }

    /// Walk every durable continuation page and return the whole queue.
    ///
    /// [`Self::list_queue`] returns ONE bounded page, which is right for a
    /// poller reconciling live cards. It is wrong for anything exhaustive: a
    /// backlog longer than the host's `queue_capacity` silently loses its
    /// tail, so "nothing is held" and "that job is not held" become answers
    /// about the first page rather than about the queue. Reserved for
    /// explicit operator actions; periodic consumers keep polling one page.
    ///
    /// `page` and `live_only_entries` are dropped from the result: they
    /// describe one page's position in a walk that is now finished, and
    /// keeping either would invite a caller to page again from the middle.
    ///
    /// Rows are deduplicated by id across the whole walk. The server repeats
    /// the bounded `live_only_entries` set on EVERY explicit page — it is the
    /// registry's non-durable overlay, not a slice of the durable order — and
    /// `fetch_queue_listing` folds it into `entries` per page, so appending
    /// blindly would list each live-only job once per continuation.
    pub async fn list_queue_all(&self) -> Result<QueueListingWire> {
        let mut listing = self.list_queue().await?;
        let plan = listing.plan.take();
        let mut entries: Vec<crate::QueueJobEntryWire> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        let mut seen_cursors = std::collections::HashSet::new();
        loop {
            entries.extend(
                std::mem::take(&mut listing.entries)
                    .into_iter()
                    .filter(|entry| seen_ids.insert(entry.id.clone())),
            );
            let Some(cursor) = listing
                .page
                .as_ref()
                .and_then(|page| page.next_cursor.clone())
            else {
                break;
            };
            let limit = listing.page.as_ref().map_or(0, |page| page.limit);
            if !seen_cursors.insert(cursor.clone()) {
                anyhow::bail!("host repeated a queue continuation cursor");
            }
            listing = self.list_queue_page(limit, Some(&cursor)).await?;
        }
        // A walk of the durable order restates position per page, so the
        // merged sequence is the authority for where each row sits — held
        // rows included in the walk, excluded from the count.
        crate::queue_wait::assign_listed_positions(&mut entries);
        Ok(QueueListingWire {
            entries,
            live_only_entries: Vec::new(),
            plan,
            page: None,
        })
    }

    /// Resolve one exact job through bounded pages. This is reserved for an
    /// explicit per-job action/probe; periodic consumers use only
    /// [`Self::list_queue`] and never walk the durable journal.
    pub async fn find_queue_job(&self, id: &str) -> Result<Option<crate::QueueJobEntryWire>> {
        let mut listing = self.list_queue().await?;
        let mut seen_cursors = std::collections::HashSet::new();
        loop {
            if let Some(entry) = listing.entries.into_iter().find(|entry| entry.id == id) {
                return Ok(Some(entry));
            }
            let Some(page) = listing.page else {
                return Ok(None);
            };
            let Some(cursor) = page.next_cursor else {
                return Ok(None);
            };
            if !seen_cursors.insert(cursor.clone()) {
                anyhow::bail!("host repeated a queue continuation cursor");
            }
            listing = self.list_queue_page(page.limit, Some(&cursor)).await?;
        }
    }

    async fn fetch_queue_listing(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<QueueListingWire> {
        let mut request = self.client.get(format!("{}/api/queue", self.base_url));
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit.to_string())]);
        }
        if let Some(cursor) = cursor {
            request = request.query(&[("cursor", cursor)]);
        }
        let mut listing = request
            .send()
            .await?
            .error_for_status()?
            .json::<QueueListingWire>()
            .await?;
        listing.merge_live_only_entries();
        Ok(listing)
    }

    /// Read one live job's folded progress snapshot
    /// (`GET /api/queue/{id}/preview`).
    ///
    /// The outer `Option` is live-row existence — a `404` means the row has
    /// left the queue and is reported as `Ok(None)` rather than an error,
    /// because that is the ordinary end of a render rather than a fault.
    pub async fn queue_job_progress(&self, id: &str) -> Result<Option<QueueJobProgress>> {
        let resp = self
            .client
            .get(format!(
                "{}/api/queue/{}/preview",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(resp
            .error_for_status()?
            .json::<Option<QueueJobProgress>>()
            .await?)
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

    /// Read ONE queued job with the planner's own work item for it
    /// (`GET /api/queue/{id}`).
    ///
    /// Unlike [`Self::find_queue_job`], which scans the paged listing, this
    /// asks the host directly — so it answers for a durable row deeper than
    /// the runtime window, and it carries the request settings the
    /// payload-free listing projection deliberately omits. A job that has
    /// left the queue is `Ok(None)`, never an error: finishing is not a
    /// failure.
    pub async fn queue_job(&self, id: &str) -> Result<Option<crate::QueueJobDetailWire>> {
        let resp = self
            .client
            .get(format!(
                "{}/api/queue/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(
            error_for_status_with_body(resp)
                .await?
                .json::<crate::QueueJobDetailWire>()
                .await?,
        ))
    }

    /// Reorder one queued job (`PATCH /api/queue/{id}`), returning the row as
    /// the server re-projected it. Positions past the tail are clamped by the
    /// server, so the returned `position` is the authority, not the request.
    pub async fn move_queue_job(
        &self,
        id: &str,
        position: usize,
    ) -> Result<crate::QueueJobEntryWire> {
        let resp = self
            .client
            .patch(format!(
                "{}/api/queue/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .json(&serde_json::json!({ "position": position }))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<crate::QueueJobEntryWire>()
            .await?)
    }

    /// Cancel every still-queued job on this host (`DELETE /api/queue`).
    ///
    /// Running work is deliberately left alone; the count is the number of
    /// queued rows the server removed.
    pub async fn cancel_all_queue_jobs(&self) -> Result<usize> {
        let resp = self
            .client
            .delete(format!("{}/api/queue", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<crate::QueueCancelAllResult>()
            .await?
            .cancelled)
    }

    /// Cancel every non-terminal child of one durable batch
    /// (`DELETE /api/generation-batches/{id}`), returning the authoritative
    /// child states the server settled on.
    pub async fn cancel_generation_batch(&self, id: &str) -> Result<GenerationBatchStatus> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/generation-batches/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<GenerationBatchStatus>()
            .await?)
    }

    /// Hold dispatch of new generation jobs (`POST /api/queue/pause`). The
    /// job already on a worker finishes. Returns the resulting pause state.
    pub async fn pause_queue(&self) -> Result<bool> {
        self.set_queue_paused("pause").await
    }

    /// Resume dispatch (`POST /api/queue/resume`). Returns the resulting
    /// pause state, so a caller reports what the host settled on rather than
    /// what it asked for.
    pub async fn resume_queue(&self) -> Result<bool> {
        self.set_queue_paused("resume").await
    }

    async fn set_queue_paused(&self, action: &str) -> Result<bool> {
        let resp = self
            .client
            .post(format!("{}/api/queue/{action}", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<crate::QueuePauseState>()
            .await?
            .paused)
    }

    /// Pause only one waiting generation row, leaving host dispatch and every
    /// sibling runnable.
    pub async fn pause_queue_job(&self, id: &str) -> Result<()> {
        self.set_queue_job_paused(id, "pause").await
    }

    /// Return only one paused generation row to the runnable queue.
    pub async fn resume_queue_job(&self, id: &str) -> Result<()> {
        self.set_queue_job_paused(id, "resume").await
    }

    async fn set_queue_job_paused(&self, id: &str, action: &str) -> Result<()> {
        let resp = self
            .client
            .post(format!(
                "{}/api/queue/{}/{}",
                self.base_url,
                encode_path_segment(id),
                action
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Run the held-row retention sweep now (`POST /api/queue/held/sweep`).
    pub async fn sweep_held_queue(&self) -> Result<crate::HeldSweepResult> {
        let resp = self
            .client
            .post(format!("{}/api/queue/held/sweep", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<crate::HeldSweepResult>()
            .await?)
    }

    /// Run the settled-batch retention sweep now
    /// (`POST /api/generation-batches/sweep`).
    pub async fn sweep_settled_batches(&self) -> Result<crate::SettledBatchSweepResult> {
        let resp = self
            .client
            .post(format!("{}/api/generation-batches/sweep", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<crate::SettledBatchSweepResult>()
            .await?)
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

    /// Read one gallery row's metadata without transferring the whole index.
    ///
    /// Falls back to a full listing for a host that predates `?filename=`:
    /// an older server ignores the unknown query parameter and returns
    /// everything, which is exactly the pre-filter behaviour, so the local
    /// `find` below stays correct either way.
    pub async fn gallery_item(&self, filename: &str) -> Result<Option<GalleryImage>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery", self.base_url))
            .query(&[("filename", filename)])
            .send()
            .await?
            .error_for_status()?
            .json::<Vec<GalleryImage>>()
            .await?;
        Ok(resp.into_iter().find(|item| item.filename == filename))
    }

    /// Download a gallery image by filename.
    pub async fn get_gallery_image(&self, filename: &str) -> Result<Vec<u8>> {
        let resp = self
            .client
            .get(format!(
                "{}/api/gallery/image/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
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
            .delete(format!(
                "{}/api/gallery/image/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// List one gallery view: `"library"` (live prints, the default the
    /// bare `GET /api/gallery` serves) or `"trash"` (prints waiting in the
    /// host's trash, each carrying `trashed_at` / `purge_at`). Older servers
    /// ignore the query and return the live listing.
    pub async fn list_gallery_view(&self, view: &str) -> Result<Vec<GalleryImage>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery", self.base_url))
            .query(&[("view", view)])
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<Vec<GalleryImage>>()
            .await?)
    }

    /// Move one print to the host's trash (`DELETE /api/gallery/image/:name`
    /// without `permanent`). On older servers without a trash this deletes
    /// outright — check `capabilities.gallery.trash` first when that matters.
    pub async fn trash_gallery_image(&self, filename: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/gallery/image/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Permanently delete one print, bypassing the trash
    /// (`DELETE /api/gallery/image/:name?permanent=true`). Works on live and
    /// already-trashed prints alike.
    pub async fn delete_gallery_image_forever(&self, filename: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/gallery/image/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
            .query(&[("permanent", "true")])
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Move several prints to the trash in one call (`POST /api/gallery/trash`).
    pub async fn trash_gallery_images(&self, filenames: &[String]) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/trash", self.base_url))
            .json(&TrashFilenamesRequest {
                filenames: filenames.to_vec(),
            })
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Restore trashed prints to the live gallery
    /// (`POST /api/gallery/trash/restore`). A live print already using one of
    /// the names surfaces as a `409` `GALLERY_RESTORE_CONFLICT` error.
    pub async fn restore_trashed(&self, filenames: &[String]) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/trash/restore", self.base_url))
            .json(&TrashFilenamesRequest {
                filenames: filenames.to_vec(),
            })
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Permanently purge every trashed print (`DELETE /api/gallery/trash`).
    pub async fn empty_trash(&self) -> Result<EmptyTrashResult> {
        let resp = self
            .client
            .delete(format!("{}/api/gallery/trash", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<EmptyTrashResult>()
            .await?)
    }

    /// Run the retention sweep now (`POST /api/gallery/trash/sweep`): purges
    /// trashed prints older than `gallery.trash_retention_days` and reports
    /// how many remain.
    pub async fn sweep_trash(&self) -> Result<TrashSweepResult> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/trash/sweep", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<TrashSweepResult>()
            .await?)
    }

    /// Edit one print's title / favorite / tags
    /// (`PATCH /api/gallery/image/:name`); returns the updated row.
    pub async fn patch_gallery_image(
        &self,
        filename: &str,
        patch: &GalleryPatchRequest,
    ) -> Result<GalleryImage> {
        let resp = self
            .client
            .patch(format!(
                "{}/api/gallery/image/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
            .json(patch)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<GalleryImage>()
            .await?)
    }

    /// Apply one organization edit to many prints (`POST /api/gallery/organize`).
    pub async fn organize_gallery(&self, req: &GalleryOrganizeRequest) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/organize", self.base_url))
            .json(req)
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Apply one replay-safe bulk organization mutation
    /// (`POST /api/gallery/mutations`).
    pub async fn mutate_gallery_bulk(
        &self,
        req: &GalleryBulkMutationRequest,
    ) -> Result<GalleryBulkMutationResult> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/mutations", self.base_url))
            .json(req)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<GalleryBulkMutationResult>()
            .await?)
    }

    /// List the host's collections (`GET /api/gallery/collections`).
    pub async fn list_collections(&self) -> Result<Vec<Collection>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery/collections", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<Vec<Collection>>()
            .await?)
    }

    /// Read one collection and its ordered member filenames
    /// (`GET /api/gallery/collections/:id`).
    pub async fn get_collection(&self, id: &str) -> Result<CollectionDetail> {
        let resp = self
            .client
            .get(format!(
                "{}/api/gallery/collections/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<CollectionDetail>()
            .await?)
    }

    /// Create a collection (`POST /api/gallery/collections`).
    pub async fn create_collection(&self, req: &CollectionCreateRequest) -> Result<Collection> {
        let resp = self
            .client
            .post(format!("{}/api/gallery/collections", self.base_url))
            .json(req)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<Collection>()
            .await?)
    }

    /// Rename / describe / re-cover a collection
    /// (`PATCH /api/gallery/collections/:id`); returns the updated record.
    pub async fn update_collection(
        &self,
        id: &str,
        req: &CollectionUpdateRequest,
    ) -> Result<Collection> {
        let resp = self
            .client
            .patch(format!(
                "{}/api/gallery/collections/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .json(req)
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<Collection>()
            .await?)
    }

    /// Delete a collection (`DELETE /api/gallery/collections/:id`). Member
    /// prints are never trashed or deleted.
    pub async fn delete_collection(&self, id: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/gallery/collections/{}",
                self.base_url,
                encode_path_segment(id)
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Add / remove prints in a collection
    /// (`PUT /api/gallery/collections/:id/items`).
    pub async fn set_collection_items(&self, id: &str, req: &CollectionItemsRequest) -> Result<()> {
        let resp = self
            .client
            .put(format!(
                "{}/api/gallery/collections/{}/items",
                self.base_url,
                encode_path_segment(id)
            ))
            .json(req)
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// List tags with live-print counts (`GET /api/gallery/tags`).
    pub async fn list_tags(&self) -> Result<Vec<TagCount>> {
        let resp = self
            .client
            .get(format!("{}/api/gallery/tags", self.base_url))
            .send()
            .await?;
        Ok(error_for_status_with_body(resp)
            .await?
            .json::<Vec<TagCount>>()
            .await?)
    }

    /// Rename a tag everywhere it is used (`PATCH /api/gallery/tags/:name`).
    pub async fn rename_tag(&self, name: &str, new_name: &str) -> Result<()> {
        let resp = self
            .client
            .patch(format!(
                "{}/api/gallery/tags/{}",
                self.base_url,
                encode_path_segment(name)
            ))
            .json(&TagRenameRequest {
                name: new_name.to_string(),
            })
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
        Ok(())
    }

    /// Delete a tag, detaching it from every print (`DELETE /api/gallery/tags/:name`).
    pub async fn delete_tag(&self, name: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!(
                "{}/api/gallery/tags/{}",
                self.base_url,
                encode_path_segment(name)
            ))
            .send()
            .await?;
        error_for_status_with_body(resp).await?;
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
            .get(format!(
                "{}/api/gallery/preview/{}",
                self.base_url,
                encode_path_segment(filename)
            ))
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
                "{}/api/gallery/thumbnail/{}",
                self.base_url,
                encode_path_segment(filename)
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
        let wire_req = crate::prompt_text::protect_expand_request_for_wire(req);
        let resp = self
            .client
            .post(format!("{}/api/expand", self.base_url))
            .json(&wire_req)
            .send()
            .await?
            .error_for_status()?
            .json::<ExpandResponse>()
            .await?;
        Ok(resp)
    }

    /// Generate subject-preserving prompt alternatives. A distinct endpoint is
    /// intentional: older hosts return 404 instead of silently expanding.
    pub async fn remix_prompt(&self, req: &crate::RemixRequest) -> Result<crate::RemixResponse> {
        let wire_req = crate::prompt_text::protect_remix_request_for_wire(req);
        let resp = self
            .client
            .post(format!("{}/api/remix", self.base_url))
            .json(&wire_req)
            .send()
            .await?
            .error_for_status()?
            .json::<crate::RemixResponse>()
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

/// True only when an HTTP endpoint is absent on an otherwise responding
/// older server. Callers may use this for additive API compatibility, but
/// must not turn authentication, transport, or server failures into a legacy
/// fallback that could mutate state without a successful preflight.
pub fn is_missing_endpoint_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<ServerResponseError>()
        .map(|error| error.status)
        .or_else(|| {
            err.downcast_ref::<reqwest::Error>()
                .and_then(reqwest::Error::status)
        })
        .is_some_and(|status| {
            matches!(
                status,
                reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
            )
        })
}

/// True when retrying an idempotent read or reconciling an ambiguously
/// committed mutation is safe. HTTP response failures are wrapped so their
/// status and body remain actionable; keep the classification here rather
/// than making callers guess from formatted error text.
pub fn is_transient_request_error(err: &anyhow::Error) -> bool {
    if MoldClient::is_connection_error(err) {
        return true;
    }
    err.chain().any(|cause| {
        cause
            .downcast_ref::<ServerResponseError>()
            .is_some_and(|error| error.status.is_server_error() || error.status.as_u16() == 429)
            || cause.downcast_ref::<reqwest::Error>().is_some_and(|error| {
                error.is_timeout()
                    || error.is_connect()
                    || error.is_body()
                    || error.is_decode()
                    || error.is_request()
                    || error
                        .status()
                        .is_some_and(|status| status.is_server_error() || status.as_u16() == 429)
            })
    })
}

#[derive(Debug)]
struct ServerResponseError {
    status: StatusCode,
    body: String,
}

impl std::fmt::Display for ServerResponseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "server error {}: {}", self.status, self.body)
    }
}

impl std::error::Error for ServerResponseError {}

async fn error_for_status_with_body(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_client_error() || resp.status().is_server_error() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(ServerResponseError { status, body }.into());
    }
    Ok(resp)
}

/// Read a `/api/generate` answer as the media it promises, or say exactly
/// which of the three non-media answers arrived.
///
/// `202` is the attached observer detaching after commit — the print is
/// still queued and must be reconciled, never resubmitted. A `200` carrying
/// JSON is the idempotent replay of a `client_batch_id` this host already
/// admitted: the batch status, with the gallery filename when the print is
/// done. A non-empty `404` is the singleton contract's own "model not here"
/// (`MODEL_NOT_FOUND` / `UNKNOWN_MODEL`), surfaced as
/// [`MoldError::ModelNotFound`] so [`MoldClient::is_model_not_found`] and the
/// CLI's auto-pull read it as they always have.
async fn require_direct_media_response(resp: reqwest::Response) -> Result<reqwest::Response> {
    match resp.status() {
        StatusCode::ACCEPTED => {
            let status = resp.json::<GenerationBatchStatus>().await?;
            anyhow::bail!(
                "durable generation was accepted but its direct observer detached; reconcile batch {} or client operation {}",
                status.id,
                status.client_batch_id
            );
        }
        StatusCode::OK if response_is_json(&resp) => {
            let status = resp.json::<GenerationBatchStatus>().await?;
            let filename = status
                .children
                .first()
                .and_then(|child| child.result.as_ref())
                .and_then(|result| result.filename.as_deref());
            match filename {
                Some(filename) => anyhow::bail!(
                    "client operation {} was already admitted as batch {}; its print is gallery file {filename}",
                    status.client_batch_id,
                    status.id
                ),
                None => anyhow::bail!(
                    "client operation {} was already admitted as batch {}; reconcile it instead of resubmitting",
                    status.client_batch_id,
                    status.id
                ),
            }
        }
        StatusCode::NOT_FOUND => {
            let body = resp.text().await.unwrap_or_default();
            if body.is_empty() {
                anyhow::bail!("the server does not serve POST /api/generate");
            }
            Err(MoldError::ModelNotFound(body).into())
        }
        _ => error_for_status_with_body(resp).await,
    }
}

fn response_is_json(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"))
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
    pipeline_provenance_sha256: Option<String>,
    source_preprocessing: Option<crate::Ltx2SourcePreprocessing>,
    attention_path: Option<String>,
    int8_arm: Option<String>,
    video_only: Option<bool>,
    has_audio: bool,
    duration_ms: Option<u64>,
    audio_sample_rate: Option<u32>,
    audio_channels: Option<u32>,
}

/// Read the `x-mold-request-warning` advisories off a response.
///
/// The server accepted the request and rendered the print; these say what it
/// had to adjust or drop along the way — a lip-dub retiming, a filing a host
/// with no metadata database could not apply, a collection deleted between
/// listing and Generate. A terminal client that never reads this header turns
/// "never a silent drop" into exactly that, which is why this is parsed on
/// every response path rather than only where a warning is expected.
///
/// **Header values are never split.** The server joins several advisories
/// with `"; "`, but its own advisory prose contains that sequence — "…were
/// not applied; the print was generated and saved normally" — so splitting on
/// the separator would shred one advisory into two half-sentences, each
/// rendered as its own warning line. A joined line reads correctly as prose
/// because the semicolons are punctuation; two fragments do not. Values are
/// therefore taken whole.
///
/// `get_all` rather than `get` so that a server which one day emits one
/// header per advisory — the lossless HTTP idiom, and the fix if this ever
/// needs real structure — is read correctly without a client change.
fn parse_request_warnings(headers: &reqwest::header::HeaderMap) -> Vec<String> {
    headers
        .get_all("x-mold-request-warning")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Fold the advisories a completion event carried into the ones the response
/// headers carried.
///
/// A streaming render has two delivery channels and needs both. The headers
/// arrive before the first SSE frame, so they can only carry what admission
/// already knew — a retimed lip dub, a filing the host could not apply. An
/// advisory the RENDER produced, such as which of several detected faces the
/// identity extractor conditioned on, is decided while the job is being
/// prepared, long after those headers were written; it can only travel in the
/// completion event. Reading one channel and not the other is how `mold run`
/// silently swallowed the multi-face notice (#1223).
///
/// Header advisories keep their position, because they describe what happened
/// first. Duplicates are dropped: the server may repeat an advisory in the
/// completion event that it already sent as a header, and a caller should see
/// it once.
fn merge_completion_warnings(from_headers: &[String], from_completion: &[String]) -> Vec<String> {
    let mut merged = from_headers.to_vec();
    for warning in from_completion {
        let warning = warning.trim();
        if warning.is_empty() || merged.iter().any(|held| held == warning) {
            continue;
        }
        merged.push(warning.to_owned());
    }
    merged
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
    let pipeline_provenance_sha256 = headers
        .get("x-mold-video-pipeline-provenance-sha256")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let source_preprocessing = headers
        .get("x-mold-video-source-preprocessing")
        .and_then(|value| value.to_str().ok())
        .and_then(|json| serde_json::from_str(json).ok());
    let attention_path = headers
        .get("x-mold-video-attention-path")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let int8_arm = headers
        .get("x-mold-video-int8-arm")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let video_only = headers
        .get("x-mold-video-video-only")
        .and_then(|v| v.to_str().ok())
        .map(|s| s == "1")
        .filter(|only| *only);
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
        pipeline_provenance_sha256,
        source_preprocessing,
        attention_path,
        int8_arm,
        video_only,
        has_audio,
        duration_ms,
        audio_sample_rate,
        audio_channels,
    })
}

struct MeshMeta {
    format: Option<OutputFormat>,
    vertex_count: u32,
    face_count: u32,
    textured: bool,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    poster_width: u32,
    poster_height: u32,
}

/// Parse mesh metadata from HTTP response headers.
///
/// Returns `Some` when `x-mold-mesh-vertices` is present. Probed FIRST, ahead
/// of audio and video, for the same reason audio is probed ahead of video: a
/// mesh response has no frames and no sample rate, so both of those probes
/// fall through to the image branch and would hand the caller glTF bytes
/// labelled as a picture.
fn parse_mesh_headers(headers: &reqwest::header::HeaderMap) -> Option<MeshMeta> {
    let read = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
    };
    let read_bounds =
        |name: &str| parse_bounds(headers.get(name).and_then(|value| value.to_str().ok()));
    let vertex_count = read("x-mold-mesh-vertices")? as u32;
    Some(MeshMeta {
        // Stated by the server: a request that omitted `output_format` was
        // normalised server-side and is no evidence of what came back.
        format: headers
            .get("x-mold-mesh-format")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<OutputFormat>().ok()),
        vertex_count,
        face_count: read("x-mold-mesh-faces").unwrap_or(0) as u32,
        textured: headers
            .get("x-mold-mesh-textured")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(false),
        bounds_min: read_bounds("x-mold-mesh-bounds-min"),
        bounds_max: read_bounds("x-mold-mesh-bounds-max"),
        poster_width: read("x-mold-mesh-poster-width").unwrap_or(0) as u32,
        poster_height: read("x-mold-mesh-poster-height").unwrap_or(0) as u32,
    })
}

/// Parse an `x, y, z` bounds header. An absent or malformed value is the
/// origin — an older server that does not send the header is not an error,
/// and a caller reads a degenerate box rather than a wrong one.
fn parse_bounds(raw: Option<&str>) -> [f32; 3] {
    let Some(raw) = raw else {
        return [0.0; 3];
    };
    let mut out = [0.0f32; 3];
    let mut seen = 0usize;
    for (slot, part) in out.iter_mut().zip(raw.split(',')) {
        match part.trim().parse::<f32>() {
            Ok(value) if value.is_finite() => {
                *slot = value;
                seen += 1;
            }
            _ => return [0.0; 3],
        }
    }
    if seen == 3 {
        out
    } else {
        [0.0; 3]
    }
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
fn build_client(api_key: Option<&str>) -> (Client, bool) {
    let mut builder = Client::builder();
    let mut api_key_configured = false;
    if let Some(key) = api_key {
        let mut headers = reqwest::header::HeaderMap::new();
        match reqwest::header::HeaderValue::from_str(key) {
            Ok(val) if !key.trim().is_empty() => {
                headers.insert("x-api-key", val);
                api_key_configured = true;
            }
            _ => {
                eprintln!(
                    "warning: MOLD_API_KEY contains characters invalid for an HTTP header; \
                     authentication header will not be sent"
                );
            }
        }
        builder = builder.default_headers(headers);
    }
    match builder.build() {
        Ok(client) => (client, api_key_configured),
        Err(_) => (Client::new(), false),
    }
}

/// Normalize a host string into a full URL.
///
/// Accepts:
/// - Bare hostname: `hal9000` → `http://hal9000:7680`
/// - Host with port: `hal9000:8080` → `http://hal9000:8080`
/// - Full URL: `http://hal9000:7680` → unchanged
/// - URL without port: `http://hal9000` → unchanged (uses scheme default 80/443)
fn has_http_scheme(input: &str) -> bool {
    input
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        || input
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"))
}

pub fn normalize_host(input: &str) -> String {
    let trimmed = input.trim().trim_end_matches('/');
    let has_scheme = has_http_scheme(trimmed);
    let bare_ipv6 = !has_scheme
        && !trimmed.starts_with('[')
        && trimmed.bytes().filter(|byte| *byte == b':').count() > 1;
    let candidate = if has_scheme {
        trimmed.to_string()
    } else if bare_ipv6 {
        format!("http://[{trimmed}]")
    } else {
        format!("http://{trimmed}")
    };
    if let Ok(mut url) = reqwest::Url::parse(&candidate) {
        if !has_scheme && url.port().is_none() {
            let _ = url.set_port(Some(7680));
        }
        url.to_string().trim_end_matches('/').to_string()
    } else {
        candidate
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
    #[test]
    fn mesh_bounds_survive_the_wire_and_degrade_to_the_origin() {
        assert_eq!(parse_bounds(Some("-1.5, 0, 2.25")), [-1.5, 0.0, 2.25]);
        // An older server sends no header at all. That is not an error; the
        // caller reads a degenerate box rather than a wrong one.
        assert_eq!(parse_bounds(None), [0.0; 3]);
        // Partial, malformed and non-finite values all collapse the whole
        // triple rather than yielding a plausible-looking mixture.
        assert_eq!(parse_bounds(Some("1,2")), [0.0; 3]);
        assert_eq!(parse_bounds(Some("1,2,three")), [0.0; 3]);
        assert_eq!(parse_bounds(Some("1,2,NaN")), [0.0; 3]);
        assert_eq!(parse_bounds(Some("1,2,inf")), [0.0; 3]);
    }

    #[test]
    fn mesh_headers_carry_the_bounds_the_meshdata_promises() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-mesh-vertices", "24576".parse().unwrap());
        headers.insert("x-mold-mesh-faces", "49152".parse().unwrap());
        headers.insert("x-mold-mesh-bounds-min", "-1,-1,-1".parse().unwrap());
        headers.insert("x-mold-mesh-bounds-max", "1,1,1".parse().unwrap());
        let meta = parse_mesh_headers(&headers).expect("a mesh response");
        assert_eq!(meta.vertex_count, 24576);
        assert_eq!(meta.face_count, 49152);
        assert_eq!(meta.bounds_min, [-1.0, -1.0, -1.0]);
        assert_eq!(meta.bounds_max, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_still_or_a_clip_is_never_read_as_a_mesh() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert!(parse_mesh_headers(&headers).is_none());
        // A clip carries neither vertices nor faces; only the vertex header
        // may promote a response to `MeshData`.
        headers.insert("x-mold-video-frames", "97".parse().unwrap());
        assert!(parse_mesh_headers(&headers).is_none());
    }

    use super::*;
    use crate::test_support::ENV_LOCK;

    #[tokio::test]
    async fn gallery_media_helpers_encode_reserved_filename_characters() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let encoded = "odd%20%23%20100%25-%E6%97%A5%E6%9C%AC.png";
        for (method_name, route, body) in [
            ("GET", "image", b"image".as_slice()),
            ("GET", "preview", b"preview".as_slice()),
            ("GET", "thumbnail", b"thumbnail".as_slice()),
        ] {
            Mock::given(method(method_name))
                .and(path(format!("/api/gallery/{route}/{encoded}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
                .expect(1)
                .mount(&server)
                .await;
        }
        Mock::given(method("DELETE"))
            .and(path(format!("/api/gallery/image/{encoded}")))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let filename = "odd # 100%-日本.png";
        assert_eq!(client.get_gallery_image(filename).await.unwrap(), b"image");
        assert_eq!(
            client.get_gallery_preview(filename).await.unwrap().unwrap(),
            b"preview"
        );
        assert_eq!(
            client.get_gallery_thumbnail(filename).await.unwrap(),
            b"thumbnail"
        );
        client.delete_gallery_image(filename).await.unwrap();
    }

    #[test]
    fn transient_request_errors_include_wrapped_rate_limits_and_server_failures() {
        for status in [StatusCode::TOO_MANY_REQUESTS, StatusCode::BAD_GATEWAY] {
            let error = anyhow::Error::new(ServerResponseError {
                status,
                body: "retry later".into(),
            });
            assert!(is_transient_request_error(&error), "status {status}");
        }

        let conflict = anyhow::Error::new(ServerResponseError {
            status: StatusCode::CONFLICT,
            body: "authority mismatch".into(),
        });
        assert!(!is_transient_request_error(&conflict));
    }

    fn reference_session_request() -> ReferenceUploadSessionRequest {
        let request = serde_json::from_value(serde_json::json!({
            "prompt": "match the reference",
            "model": crate::minimax_h3::REF2VA_COMFY,
            "width": crate::minimax_h3::DEFAULT_WIDTH,
            "height": crate::minimax_h3::DEFAULT_HEIGHT,
            "steps": crate::minimax_h3::DEFAULT_STEPS,
            "guidance": 0.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "mp4",
            "strength": 1.0,
            "frames": crate::minimax_h3::MIN_FRAMES,
            "fps": crate::minimax_h3::FIXED_FPS,
            "enable_audio": true,
            "references": [{
                "kind": "image",
                "media": { "authority": "descriptor" },
                "provenance": {
                    "name": "reference.png",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
                },
                "mime_type": "image/png",
                "width": 1,
                "height": 1
            }]
        }))
        .unwrap();
        ReferenceUploadSessionRequest {
            request,
            upload_references: vec![1],
        }
    }

    #[test]
    fn test_new_trims_trailing_slash() {
        let client = MoldClient::new("http://localhost:7680/");
        assert_eq!(client.host(), "http://localhost:7680");
    }

    #[test]
    fn api_key_state_tracks_only_an_installed_header() {
        assert!(!MoldClient::new("http://localhost:7680").has_api_key());
        assert!(
            MoldClient::with_api_key("http://localhost:7680", "sekrit".to_string()).has_api_key()
        );
        assert!(!MoldClient::with_api_key("http://localhost:7680", "".to_string()).has_api_key());
        assert!(
            !MoldClient::with_api_key("http://localhost:7680", "bad\nkey".to_string())
                .has_api_key()
        );
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
    fn test_normalize_scheme_case_insensitively() {
        let client = MoldClient::new("HTTP://hal9000");
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
        let client = MoldClient::new("100.123.198.98");
        assert_eq!(client.host(), "http://100.123.198.98:7680");
    }

    #[test]
    fn test_normalize_ip_with_port() {
        let client = MoldClient::new("192.168.1.100:9090");
        assert_eq!(client.host(), "http://192.168.1.100:9090");
    }

    #[test]
    fn test_normalize_bare_ipv6() {
        let client = MoldClient::new("::1");
        assert_eq!(client.host(), "http://[::1]:7680");
    }

    #[test]
    fn test_normalize_host_is_idempotent() {
        for input in [
            "100.123.198.98",
            "100.123.198.98:9000",
            "http://100.123.198.98",
            "https://studio.tailnet.ts.net",
            "https://studio.tailnet.ts.net:443",
            "::1",
        ] {
            let once = normalize_host(input);
            assert_eq!(normalize_host(&once), once, "{input}");
        }
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
    async fn reference_session_preserves_authenticated_http_451_body() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate/reference-upload-sessions"))
            .and(header("x-api-key", "sekrit"))
            .respond_with(ResponseTemplate::new(451).set_body_json(serde_json::json!({
                "error": "MiniMax H3 legal activation is unavailable",
                "code": crate::MINIMAX_H3_AUTHORIZATION_REQUIRED
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = MoldClient::with_api_key(&server.uri(), "sekrit".to_string())
            .create_reference_upload_session(&reference_session_request())
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("451 Unavailable For Legal Reasons"));
        assert!(message.contains(crate::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[tokio::test]
    async fn reference_file_streams_with_secret_headers_and_cancels_by_session_header() {
        use wiremock::matchers::{body_string, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/api/generate/reference-upload"))
            .and(header("x-api-key", "sekrit"))
            .and(header(REFERENCE_UPLOAD_HANDLE_HEADER, "mru_secret"))
            .and(header("content-type", "image/png"))
            .and(header("content-length", "5"))
            .and(body_string("bytes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "instance_id": "server-1",
                "reference": 1,
                "metadata": {
                    "kind": "image",
                    "index": 1,
                    "name": "reference.png",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "mime_type": "image/png",
                    "width": 1,
                    "height": 1
                },
                "request_scope_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "session_complete": true
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/api/generate/reference-upload"))
            .and(header("x-api-key", "sekrit"))
            .and(header(REFERENCE_UPLOAD_HANDLE_HEADER, "mru_open"))
            .and(header("content-type", "image/png"))
            .and(header("content-length", "15"))
            .and(body_string("reference-bytes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "instance_id": "server-1",
                "reference": 1,
                "metadata": {
                    "kind": "image",
                    "index": 1,
                    "name": "reference.png",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "mime_type": "image/png",
                    "width": 16,
                    "height": 16
                },
                "request_scope_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "session_complete": true
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/api/generate/reference-upload"))
            .and(header("x-api-key", "sekrit"))
            .and(header(REFERENCE_UPLOAD_HANDLE_HEADER, "mru_bytes"))
            .and(header("content-type", "audio/wav"))
            .and(header("content-length", "5"))
            .and(body_string("bytes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "instance_id": "server-1",
                "reference": 2,
                "metadata": {
                    "kind": "audio",
                    "index": 2,
                    "name": "reference.wav",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "mime_type": "audio/wav",
                    "duration_ms": 2000,
                    "sample_rate": 32000,
                    "channels": 2
                },
                "request_scope_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                "session_complete": true
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/api/generate/reference-upload-sessions"))
            .and(header("x-api-key", "sekrit"))
            .and(header(
                REFERENCE_UPLOAD_SESSION_HEADER,
                "mrs_session_secret",
            ))
            .respond_with(ResponseTemplate::new(204))
            .expect(1)
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.png");
        let open_path = dir.path().join("reference-open.png");
        std::fs::write(&path, b"bytes").unwrap();
        std::fs::write(&open_path, b"reference-bytes").unwrap();
        let client = MoldClient::with_api_key(&server.uri(), "sekrit".to_string());
        let completed = client
            .upload_reference_file("mru_secret", &path, "image/png")
            .await
            .unwrap();
        assert_eq!(completed.instance_id, "server-1");
        assert_eq!(completed.reference, 1);
        let completed = client
            .upload_reference_open_file(
                "mru_open",
                std::fs::File::open(&open_path).unwrap(),
                "image/png",
            )
            .await
            .unwrap();
        assert_eq!(completed.instance_id, "server-1");
        assert_eq!(completed.reference, 1);
        let completed = client
            .upload_reference_bytes("mru_bytes", b"bytes".to_vec(), "audio/wav")
            .await
            .unwrap();
        assert_eq!(completed.instance_id, "server-1");
        assert_eq!(completed.reference, 2);
        client
            .cancel_reference_upload_session("mrs_session_secret")
            .await
            .unwrap();
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
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.20.0",
                "models_loaded": [],
                "gpu_info": null,
                "uptime_secs": 1,
                "queue_capacity": 2
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .and(query_param("limit", "2"))
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
                ],
                "live_only_entries": [{
                    "id": "job-live",
                    "model": "minimax-h3:nvfp4",
                    "state": "running",
                    "started_at_unix_ms": 1_711_305_602_000_u64,
                    "position": 0,
                    "durable": false
                }],
                "page": { "limit": 2, "offset": 0, "returned": 2 }
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let listing = client.list_queue().await.unwrap();

        assert_eq!(listing.entries.len(), 3);
        assert_eq!(listing.entries[0].id, "job-1");
        assert_eq!(listing.entries[0].state, "running");
        assert_eq!(listing.entries[0].gpu, Some(0));
        assert_eq!(listing.entries[1].state, "queued");
        assert_eq!(listing.entries[1].gpu, None);
        assert_eq!(listing.entries[1].position, 1);
        assert_eq!(listing.entries[2].id, "job-live");
        assert_eq!(listing.entries[2].durable, Some(false));
        assert_eq!(listing.page.as_ref().map(|page| page.limit), Some(2));
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
        let listing = client.list_queue_for_capacity(None).await.unwrap();
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

    #[tokio::test]
    async fn list_queue_carries_the_hold_and_batch_identity_a_retry_needs() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entries": [{
                    "id": "job-held",
                    "model": "flux-dev:q8",
                    "state": "held",
                    "started_at_unix_ms": 1_711_305_600_000_u64,
                    "position": 0,
                    "held_reason": "dependency download failed",
                    "error": "dependency download failed",
                    "retryable": true,
                    "replayed": true,
                    "dispatch_attempts": 2,
                    "batch_id": "batch-1",
                    "client_batch_id": "client-1",
                    "batch_index": 3
                }]
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let listing = client.list_queue_for_capacity(None).await.unwrap();
        let row = &listing.entries[0];
        assert_eq!(row.error.as_deref(), Some("dependency download failed"));
        assert_eq!(row.retryable, Some(true));
        assert_eq!(row.replayed, Some(true));
        assert_eq!(row.dispatch_attempts, Some(2));
        let retry = row.retry_request("instance-a").expect("batch identity");
        assert_eq!(
            retry,
            crate::GenerationRetryRequest {
                instance_id: "instance-a".into(),
                batch_id: "batch-1".into(),
                client_batch_id: "client-1".into(),
                job_id: "job-held".into(),
            }
        );
    }

    #[test]
    fn a_row_with_no_batch_composes_no_retry_authority() {
        // Half an authority is worse than none: the server rejects a
        // mismatched body with a 409 the user cannot act on.
        let row = crate::QueueJobEntryWire {
            id: "solo".into(),
            batch_id: Some("batch-1".into()),
            ..Default::default()
        };
        assert!(row.retry_request("instance-a").is_none());
        assert!(crate::QueueJobEntryWire::default()
            .retry_request("instance-a")
            .is_none());
    }

    #[tokio::test]
    async fn list_queue_all_walks_every_continuation_page() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "version": "0.25.0",
                "models_loaded": [],
                "gpu_info": null,
                "uptime_secs": 1,
                "queue_capacity": 1
            })))
            .mount(&server)
            .await;
        let row = |id: &str| {
            serde_json::json!({
                "id": id,
                "model": "flux-dev:q8",
                "state": "held",
                "started_at_unix_ms": 1_u64,
                "position": 0
            })
        };
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .and(query_param("cursor", "page-2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entries": [row("tail")],
                "live_only_entries": [row("live")],
                "page": { "limit": 1, "offset": 1, "returned": 1 }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/queue"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entries": [row("head")],
                // Repeated on every explicit page, as the server does.
                "live_only_entries": [row("live")],
                "plan": { "plan_version": 1, "state_version": 1, "optimizer_state": "idle", "work_items": [] },
                "page": { "limit": 1, "offset": 0, "returned": 1, "next_cursor": "page-2" }
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        // One page is what a poller wants and is deliberately not the whole
        // queue: it carries the durable head plus the live-only overlay, and
        // the tail row is invisible to it.
        let one_page = client.list_queue().await.unwrap();
        assert_eq!(
            one_page
                .entries
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["head", "live"]
        );

        let all = client.list_queue_all().await.unwrap();
        assert_eq!(
            all.entries
                .iter()
                .map(|e| e.id.as_str())
                .collect::<Vec<_>>(),
            vec!["head", "live", "tail"],
            "the repeated live-only overlay must be listed once"
        );
        assert_eq!(
            all.entries
                .iter()
                .map(|entry| entry.position)
                .collect::<Vec<_>>(),
            vec![0, 0, 0],
            "the merged sequence is the authority for position; every row in this fixture is held, so none takes a place in line"
        );
        assert!(all.plan.is_some(), "the first page's plan is retained");
        assert!(
            all.page.is_none(),
            "a finished walk must not offer a middle to resume from"
        );
    }

    #[tokio::test]
    async fn queue_job_reads_one_row_with_its_work_item() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/queue/job-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "job": {
                    "id": "job-1",
                    "model": "flux-dev:q8",
                    "state": "queued",
                    "started_at_unix_ms": 1_711_305_600_000_u64,
                    "position": 2
                },
                "work_item": {
                    "work_id": "job-1",
                    "parent_id": "job-1",
                    "work_kind": "generation",
                    "activity_phase": "denoise",
                    "estimate_confidence": "high",
                    "priority_class": "user",
                    "queue_rank": 4,
                    "bypass_count": 0,
                    "blocked_reason": "insufficient_vram"
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/queue/missing"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "queue job missing not found"
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let detail = client.queue_job("job-1").await.unwrap().expect("job");
        assert_eq!(detail.job.id, "job-1");
        assert_eq!(detail.job.position, 2);
        assert_eq!(
            detail
                .work_item
                .as_ref()
                .and_then(|item| item.blocked_reason.clone()),
            Some(crate::QueueBlockedReason::InsufficientVram)
        );
        // A job that left the queue is an answer, not an error.
        assert!(client.queue_job("missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn queue_lifecycle_calls_report_the_server_s_own_numbers() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/queue"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"cancelled": 4})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/queue/pause"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"paused": true})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/queue/resume"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"paused": false})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/queue/held/sweep"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"purged": 2, "remaining": 5, "media_deferred": 1}),
            ))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches/sweep"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"purged": 3, "remaining": 6})),
            )
            .mount(&server)
            .await;
        Mock::given(method("PATCH"))
            .and(path("/api/queue/job-1"))
            .and(body_json(serde_json::json!({"position": 0})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "job-1",
                "model": "flux-dev:q8",
                "state": "queued",
                "started_at_unix_ms": 1_u64,
                "position": 0
            })))
            .mount(&server)
            .await;
        for action in ["pause", "resume"] {
            Mock::given(method("POST"))
                .and(path(format!("/api/queue/job-1/{action}")))
                .respond_with(ResponseTemplate::new(204))
                .mount(&server)
                .await;
        }
        Mock::given(method("DELETE"))
            .and(path("/api/generation-batches/batch-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": "client-1",
                "instance_id": "instance-a",
                "durable": true,
                "children": []
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        assert_eq!(client.cancel_all_queue_jobs().await.unwrap(), 4);
        assert!(client.pause_queue().await.unwrap());
        assert!(!client.resume_queue().await.unwrap());
        client.pause_queue_job("job-1").await.unwrap();
        client.resume_queue_job("job-1").await.unwrap();
        let held = client.sweep_held_queue().await.unwrap();
        assert_eq!(
            (held.purged, held.remaining, held.media_deferred),
            (2, 5, 1)
        );
        let batches = client.sweep_settled_batches().await.unwrap();
        assert_eq!((batches.purged, batches.remaining), (3, 6));
        let moved = client.move_queue_job("job-1", 0).await.unwrap();
        assert_eq!(moved.position, 0);
        let cancelled = client.cancel_generation_batch("batch-1").await.unwrap();
        assert_eq!(cancelled.id, "batch-1");
    }

    #[tokio::test]
    async fn queue_lifecycle_errors_carry_the_server_s_own_sentence() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path("/api/queue/job-1"))
            .respond_with(ResponseTemplate::new(409).set_body_json(serde_json::json!({
                "error": "queue job job-1 is already running"
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let err = client.move_queue_job("job-1", 0).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("409"), "{msg}");
        assert!(msg.contains("already running"), "{msg}");
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
        // Absent provenance headers (older server) read as unrecorded.
        assert!(meta.attention_path.is_none());
        assert!(meta.int8_arm.is_none());
        assert!(meta.video_only.is_none());
    }

    /// Runtime provenance is output authority: the raw route carries what
    /// actually ran as response headers, and the client save records them.
    #[test]
    fn parse_video_headers_reads_runtime_provenance() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-video-frames", "9".parse().unwrap());
        headers.insert(
            "x-mold-video-attention-path",
            "ltx2-bf16-math".parse().unwrap(),
        );
        headers.insert("x-mold-video-int8-arm", "native-w8a8".parse().unwrap());
        headers.insert("x-mold-video-video-only", "1".parse().unwrap());

        let meta = parse_video_headers(&headers).expect("should detect video");
        assert_eq!(meta.attention_path.as_deref(), Some("ltx2-bf16-math"));
        assert_eq!(meta.int8_arm.as_deref(), Some("native-w8a8"));
        assert_eq!(meta.video_only, Some(true));
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

    /// A server that answers `GET /api/queue` with `queue`, then serves one
    /// `/api/generate/stream` request by emitting a `queued` SSE event with
    /// `job_id` and dropping the connection mid-body (an under-delivered
    /// `Content-Length`), exactly like a process that exits while a client is
    /// attached.
    async fn spawn_dying_stream_server(queue: serde_json::Value, job_id: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let queue = queue.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buf = [0u8; 1024];
                    // Read until the end of the request head; the generate
                    // body follows on the same read for these small requests.
                    while let Ok(read) = socket.read(&mut buf).await {
                        if read == 0 {
                            return;
                        }
                        request.extend_from_slice(&buf[..read]);
                        if request.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let head = String::from_utf8_lossy(&request).to_string();
                    if head.starts_with("GET /api/status") {
                        let body = serde_json::json!({
                            "version": "0.20.0",
                            "models_loaded": [],
                            "gpu_info": null,
                            "uptime_secs": 1,
                            "queue_capacity": 1
                        })
                        .to_string();
                        let _ = socket
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                                    body.len()
                                )
                                .as_bytes(),
                            )
                            .await;
                        let _ = socket.flush().await;
                        return;
                    }
                    if head.starts_with("GET /api/queue") {
                        let body = queue.to_string();
                        let _ = socket
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                                    body.len()
                                )
                                .as_bytes(),
                            )
                            .await;
                        let _ = socket.flush().await;
                        return;
                    }
                    let frame = format!(
                        "event: progress\ndata: {{\"type\":\"queued\",\"position\":0,\"id\":\"{job_id}\"}}\n\n"
                    );
                    // Promise far more body than is delivered, then hang up.
                    let _ = socket
                        .write_all(
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 40960\r\n\r\n{frame}"
                            )
                            .as_bytes(),
                        )
                        .await;
                    let _ = socket.flush().await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        base
    }

    #[tokio::test]
    async fn a_replayed_direct_generation_names_its_batch_and_print() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": "6c2f3a7e-2d1b-4c58-8a0e-9f1d2b3c4d5e",
                "instance_id": "server-1",
                "durable": true,
                "children": [{
                    "index": 0,
                    "job_id": "job-1",
                    "state": "complete",
                    "created_at_ms": 1,
                    "updated_at_ms": 2,
                    "result": { "filename": "mold-print.png" }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = MoldClient::new(&server.uri())
            .generate(stream_request())
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("batch-1"), "{message}");
        assert!(message.contains("mold-print.png"), "{message}");
        assert!(!MoldClient::is_model_not_found(&error));
    }

    #[tokio::test]
    async fn a_direct_404_with_a_body_is_a_model_not_found() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "model 'flux-schnell:q8' is not downloaded",
                "code": crate::SSE_ERROR_CODE_MODEL_NOT_FOUND
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = MoldClient::new(&server.uri())
            .generate(stream_request())
            .await
            .unwrap_err();
        assert!(MoldClient::is_model_not_found(&error), "{error}");
        assert!(error.to_string().contains("flux-schnell:q8"), "{error}");
    }

    fn stream_request() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "a lighthouse in a storm",
            "model": "z-image-turbo:q8",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "guidance": 1.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "png",
            "strength": 1.0
        }))
        .unwrap()
    }

    /// A one-shot SSE server that emits the given response headers and then a
    /// single complete event.
    async fn spawn_completing_stream_server(
        header_warnings: &'static [&'static str],
        complete: serde_json::Value,
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            while let Ok(read) = socket.read(&mut buf).await {
                if read == 0 {
                    return;
                }
                request.extend_from_slice(&buf[..read]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let advisories = header_warnings
                .iter()
                .map(|warning| format!("x-mold-request-warning: {warning}\r\n"))
                .collect::<String>();
            let body = format!("event: complete\ndata: {complete}\n\n");
            let _ = socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{advisories}\
                         Content-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
            let _ = socket.flush().await;
        });
        base
    }

    fn complete_event(warnings: serde_json::Value) -> serde_json::Value {
        let png = base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G']);
        serde_json::json!({
            "image": png,
            "format": "png",
            "width": 64,
            "height": 64,
            "seed_used": 7,
            "generation_time_ms": 12,
            "model": "flux-dev:q4",
            "request_warnings": warnings,
        })
    }

    /// The identity extractor decides which of several detected faces to
    /// condition on while the job is being prepared — after the response
    /// headers were written, so the only channel it can travel on is the
    /// completion event. Reading headers alone is how `mold run` swallowed
    /// the notice entirely (#1223).
    #[tokio::test]
    async fn streaming_surfaces_an_advisory_the_render_produced() {
        let identity =
            "3 faces were detected in the identity image; conditioning on the largest one";
        let base = spawn_completing_stream_server(
            &["the requested collection was dropped"],
            complete_event(serde_json::json!([identity])),
        )
        .await;

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let response = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect("the stream completes");

        assert_eq!(
            response.request_warnings,
            vec![
                "the requested collection was dropped".to_string(),
                identity.to_string(),
            ],
            "both channels must reach the caller, admission's first"
        );
    }

    /// An older server omits the field entirely, and an ordinary render sends
    /// an empty one. Neither may disturb the header advisories.
    #[tokio::test]
    async fn a_completion_without_advisories_keeps_the_header_ones() {
        let mut event = complete_event(serde_json::json!([]));
        event.as_object_mut().unwrap().remove("request_warnings");
        let base = spawn_completing_stream_server(&["a lip dub was retimed"], event).await;

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let response = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect("the stream completes");

        assert_eq!(
            response.request_warnings,
            vec!["a lip dub was retimed".to_string()]
        );
    }

    /// A server may repeat a header advisory in the completion event. The
    /// caller should see it once, and prose containing `"; "` is never split.
    #[test]
    fn merging_completion_advisories_dedupes_and_never_splits_prose() {
        let prose = "Tags were not applied; the print was generated and saved normally.";
        assert_eq!(
            super::merge_completion_warnings(
                &[prose.to_string()],
                &[prose.to_string(), "  ".to_string(), " kept ".to_string()],
            ),
            vec![prose.to_string(), "kept".to_string()]
        );
        assert!(super::merge_completion_warnings(&[], &[]).is_empty());
    }

    #[tokio::test]
    async fn mid_stream_death_reports_the_job_retained_on_a_durable_host() {
        let base = spawn_dying_stream_server(
            serde_json::json!({ "entries": [{
                "id": "job-77",
                "model": "z-image-turbo:q8",
                "state": "queued",
                "started_at_unix_ms": 0,
                "position": 0,
                "durable": true
            }] }),
            "job-77",
        )
        .await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let err = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect_err("the stream dies mid-body");

        // The queued id reached the caller and the retention note names it.
        assert!(matches!(
            rx.try_recv(),
            Ok(SseProgressEvent::Queued { ref id, .. }) if id == "job-77"
        ));
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("job job-77 is retained on")
                && rendered.contains("will finish there"),
            "expected a retention note, got: {rendered}"
        );

        // Crucially, this is NOT a connect error: the CLI must surface it
        // rather than silently re-rendering the same job locally.
        assert!(!MoldClient::is_connection_error(&err));
        assert!(!MoldClient::is_model_not_found(&err));
        assert_eq!(
            crate::control::classify_generate_error(&err),
            crate::control::GenerateServerAction::SurfaceError
        );
    }

    /// A lost stream now always promises retention, because an admitted job
    /// is a journalled one by construction: `/api/generate/stream` admits
    /// through the durable queue, so there is no "the host took it but will
    /// not replay it" case left to guard against. The old per-job
    /// `durable: false` probe went with the attached path that produced it.
    #[tokio::test]
    async fn mid_stream_death_promises_the_job_finishes_on_the_host() {
        let base = spawn_dying_stream_server(
            serde_json::json!({ "entries": [{
                "id": "job-77",
                "model": "z-image-turbo:q8",
                "state": "queued",
                "started_at_unix_ms": 0,
                "position": 0,
                "durable": true
            }] }),
            "job-77",
        )
        .await;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let err = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect_err("the stream dies mid-body");

        let rendered = format!("{err:#}");
        assert!(rendered.contains("retained"), "{rendered}");
        assert!(rendered.contains("job-77"), "{rendered}");
        // Still not a connect error: the CLI surfaces it rather than silently
        // re-rendering the same job locally.
        assert!(!MoldClient::is_connection_error(&err));
        assert_eq!(
            crate::control::classify_generate_error(&err),
            crate::control::GenerateServerAction::SurfaceError
        );
    }

    /// A server that emits a `queued` event and then a TERMINAL error frame —
    /// the graceful-restart shape PR 1 produces, where the host keeps the job.
    async fn spawn_retained_frame_server(job_id: &'static str, frame: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            while let Ok(read) = socket.read(&mut buf).await {
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..read]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let body = format!(
                "event: progress\ndata: {{\"type\":\"queued\",\"position\":0,\"id\":\"{job_id}\"}}\n\nevent: error\ndata: {frame}\n\n"
            );
            let _ = socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
            let _ = socket.flush().await;
            let _ = socket.shutdown().await;
        });
        base
    }

    #[tokio::test]
    async fn a_retained_terminal_frame_reports_the_job_as_kept() {
        // The graceful restart: an operator runs `systemctl restart mold`. The
        // host sends an explicit terminal frame saying it KEPT this job, which
        // is the exact scenario the durable queue exists for.
        let base = spawn_retained_frame_server(
            "job-graceful",
            r#"{"message":"mold is restarting; this generation was kept in the queue","retained":true,"code":"server_restarting"}"#,
        )
        .await;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let err = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect_err("a terminal error frame is still an error");

        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("job job-graceful is retained on")
                && rendered.contains("will finish there"),
            "expected the retention note, got: {rendered}"
        );
        assert!(
            rendered.contains("mold is restarting"),
            "the server's own message must survive: {rendered}"
        );
        assert_eq!(
            crate::control::classify_generate_error(&err),
            crate::control::GenerateServerAction::SurfaceError
        );
    }

    #[tokio::test]
    async fn a_retained_frame_without_a_job_id_still_names_the_host() {
        let base = spawn_retained_frame_server(
            "",
            r#"{"message":"mold is restarting","retained":true,"code":"server_restarting"}"#,
        )
        .await;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let err = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect_err("a terminal error frame is still an error");

        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("retained on") && rendered.contains("will finish there"),
            "expected a host-scoped retention note, got: {rendered}"
        );
    }

    #[tokio::test]
    async fn an_ordinary_terminal_frame_promises_nothing() {
        let base =
            spawn_retained_frame_server("job-failed", r#"{"message":"host ran out of memory"}"#)
                .await;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let err = MoldClient::new(&base)
            .generate_stream(&stream_request(), tx)
            .await
            .expect_err("the server reported a failure");

        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("retained"),
            "a definitive server failure must not promise retention: {rendered}"
        );
        assert!(rendered.contains("host ran out of memory"));
    }

    // ── Library organization + trash ────────────────────────────────────

    fn trashed_row_json() -> serde_json::Value {
        serde_json::json!({
            "filename": "mold-flux-dev-q4-1700000000000~smurf-village.png",
            "metadata": {
                "prompt": "smurf village at dusk",
                "title": "Smurf village",
                "model": "flux-dev:q4",
                "seed": 7,
                "steps": 20,
                "guidance": 3.5,
                "width": 1024,
                "height": 1024,
                "version": "test"
            },
            "timestamp": 1_700_000_000_u64,
            "format": "png",
            "size_bytes": 123_456_u64,
            "title": "Smurf village",
            "tags": ["smurfs"],
            "favorite": true,
            "collections": ["col-1"],
            "trashed_at": 1_700_000_100_u64,
            "purge_at": 1_702_592_100_u64
        })
    }

    #[tokio::test]
    async fn list_gallery_view_sends_the_view_query_and_parses_trash_rows() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/gallery"))
            .and(query_param("view", "trash"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!([trashed_row_json()])),
            )
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let rows = client.list_gallery_view("trash").await.unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.title.as_deref(), Some("Smurf village"));
        assert_eq!(row.tags, vec!["smurfs".to_string()]);
        assert!(row.favorite);
        assert_eq!(row.collections, vec!["col-1".to_string()]);
        assert_eq!(row.trashed_at, Some(1_700_000_100));
        assert_eq!(row.purge_at, Some(1_702_592_100));
        assert_eq!(row.size_bytes, Some(123_456));
    }

    #[tokio::test]
    async fn trash_gallery_image_deletes_without_the_permanent_flag() {
        use wiremock::matchers::{method, path, query_param_is_missing};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/gallery/image/cat.png"))
            .and(query_param_is_missing("permanent"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        client.trash_gallery_image("cat.png").await.unwrap();
    }

    #[tokio::test]
    async fn delete_gallery_image_forever_sends_permanent_true() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/gallery/image/cat.png"))
            .and(query_param("permanent", "true"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        client
            .delete_gallery_image_forever("cat.png")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn bulk_trash_and_restore_post_the_filename_list() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let body = serde_json::json!({ "filenames": ["a.png", "b.png"] });
        Mock::given(method("POST"))
            .and(path("/api/gallery/trash"))
            .and(body_json(body.clone()))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/gallery/trash/restore"))
            .and(body_json(body))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let names = vec!["a.png".to_string(), "b.png".to_string()];
        client.trash_gallery_images(&names).await.unwrap();
        client.restore_trashed(&names).await.unwrap();
    }

    #[tokio::test]
    async fn restore_trashed_surfaces_the_409_conflict_body() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/gallery/trash/restore"))
            .respond_with(ResponseTemplate::new(409).set_body_json(serde_json::json!({
                "error": "a live print named cat.png already exists",
                "code": "GALLERY_RESTORE_CONFLICT"
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let err = client
            .restore_trashed(&["cat.png".to_string()])
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("409"), "status missing: {msg}");
        assert!(
            msg.contains("GALLERY_RESTORE_CONFLICT"),
            "body missing: {msg}"
        );
    }

    #[tokio::test]
    async fn empty_and_sweep_trash_parse_their_counts() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/api/gallery/trash"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "purged": 7 })),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/gallery/trash/sweep"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "purged": 2,
                "remaining": 5
            })))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        assert_eq!(client.empty_trash().await.unwrap().purged, 7);
        let sweep = client.sweep_trash().await.unwrap();
        assert_eq!(sweep.purged, 2);
        assert_eq!(sweep.remaining, 5);
    }

    #[tokio::test]
    async fn patch_gallery_image_sends_the_patch_and_returns_the_row() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("PATCH"))
            .and(path(
                "/api/gallery/image/mold-flux-dev-q4-1700000000000~smurf-village.png",
            ))
            .and(body_json(
                serde_json::json!({ "title": "Smurf village", "favorite": true }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(trashed_row_json()))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let patch = GalleryPatchRequest {
            title: Some("Smurf village".into()),
            favorite: Some(true),
            ..Default::default()
        };
        let row = client
            .patch_gallery_image("mold-flux-dev-q4-1700000000000~smurf-village.png", &patch)
            .await
            .unwrap();
        assert_eq!(row.title.as_deref(), Some("Smurf village"));
        assert!(row.favorite);
    }

    #[tokio::test]
    async fn organize_gallery_posts_the_bulk_body() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/gallery/organize"))
            .and(body_json(serde_json::json!({
                "filenames": ["a.png"],
                "add_tags": ["smurfs"],
                "add_to_collections": ["col-1"]
            })))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let req = GalleryOrganizeRequest {
            filenames: vec!["a.png".into()],
            add_tags: Some(vec!["smurfs".into()]),
            add_to_collections: Some(vec!["col-1".into()]),
            ..Default::default()
        };
        client.organize_gallery(&req).await.unwrap();
    }

    #[tokio::test]
    async fn collections_crud_round_trips_the_wire_types() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let collection = serde_json::json!({
            "id": "col-1",
            "name": "Smurfs",
            "slug": "smurfs",
            "count": 3,
            "created_at": 1_700_000_000_u64,
            "updated_at": 1_700_000_050_u64
        });
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/gallery/collections"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([collection])))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/gallery/collections"))
            .and(body_json(serde_json::json!({ "name": "Smurfs" })))
            .respond_with(ResponseTemplate::new(201).set_body_json(collection.clone()))
            .mount(&server)
            .await;
        Mock::given(method("PATCH"))
            .and(path("/api/gallery/collections/col-1"))
            .and(body_json(serde_json::json!({ "name": "Smurfs 2" })))
            .respond_with(ResponseTemplate::new(200).set_body_json(collection.clone()))
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/api/gallery/collections/col-1/items"))
            .and(body_json(
                serde_json::json!({ "add": ["a.png"], "remove": [] }),
            ))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/api/gallery/collections/col-1"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let listed = client.list_collections().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].slug, "smurfs");
        assert_eq!(listed[0].count, 3);
        let created = client
            .create_collection(&CollectionCreateRequest {
                name: "Smurfs".into(),
                description: None,
            })
            .await
            .unwrap();
        assert_eq!(created.id, "col-1");
        let updated = client
            .update_collection(
                "col-1",
                &CollectionUpdateRequest {
                    name: Some("Smurfs 2".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.id, "col-1");
        client
            .set_collection_items(
                "col-1",
                &CollectionItemsRequest {
                    add: vec!["a.png".into()],
                    remove: vec![],
                },
            )
            .await
            .unwrap();
        client.delete_collection("col-1").await.unwrap();
    }

    #[tokio::test]
    async fn tag_listing_rename_and_delete_encode_the_tag_name() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/gallery/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                { "name": "smurfs", "count": 4 },
                { "name": "sci fi", "count": 1 }
            ])))
            .mount(&server)
            .await;
        Mock::given(method("PATCH"))
            .and(path("/api/gallery/tags/sci%20fi"))
            .and(body_json(serde_json::json!({ "name": "scifi" })))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/api/gallery/tags/sci%20fi"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let tags = client.list_tags().await.unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].name, "smurfs");
        assert_eq!(tags[0].count, 4);
        client.rename_tag("sci fi", "scifi").await.unwrap();
        client.delete_tag("sci fi").await.unwrap();
    }

    // ── request advisories (`x-mold-request-warning`) ───────────────────

    fn warning_headers(value: &str) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-mold-request-warning", value.parse().unwrap());
        headers
    }

    /// The load-bearing one. Both filing advisories the server actually
    /// emits contain `"; "` in their prose, so a client that split on the
    /// server's join separator would render each of them as two dangling
    /// half-sentences. The value is taken whole.
    #[test]
    fn an_advisory_containing_the_join_separator_is_never_split() {
        for advisory in [
            // `resolve_request_filing`, DB-disabled host.
            "this host has no metadata database, so the requested collection and 2 tags \
             were not applied; the print was generated and saved normally",
            // `resolve_collection_reference`, unresolvable id.
            "collection 'col-1' no longer exists on this host, so the print was not filed \
             into it; its tags and everything else were applied normally",
        ] {
            assert_eq!(
                super::parse_request_warnings(&warning_headers(advisory)),
                vec![advisory.to_string()],
                "one advisory must stay one line"
            );
        }
    }

    /// Several advisories arrive joined on one line and stay one line — they
    /// read as prose, which two fragments would not. A server that one day
    /// emits one header per advisory is read as a real list without a client
    /// change.
    #[test]
    fn repeated_headers_are_read_as_separate_advisories() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.append("x-mold-request-warning", "first advisory".parse().unwrap());
        headers.append("x-mold-request-warning", "second advisory".parse().unwrap());
        assert_eq!(
            super::parse_request_warnings(&headers),
            vec!["first advisory".to_string(), "second advisory".to_string()]
        );

        // Today's joined form is one entry, verbatim.
        assert_eq!(
            super::parse_request_warnings(&warning_headers("first; second")),
            vec!["first; second".to_string()]
        );
    }

    #[test]
    fn absent_or_empty_request_warnings_yield_nothing() {
        assert!(super::parse_request_warnings(&reqwest::header::HeaderMap::new()).is_empty());
        for value in ["", "   "] {
            assert!(
                super::parse_request_warnings(&warning_headers(value)).is_empty(),
                "{value:?} must not render as a blank advisory"
            );
        }
    }

    /// The header is the whole point: a terminal client that does not read it
    /// turns "never a silent drop" into exactly that. This drives the real
    /// client against a real response.
    #[tokio::test]
    async fn generate_surfaces_the_request_warning_header() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .insert_header("x-mold-seed-used", "42")
                    .insert_header(
                        "x-mold-request-warning",
                        "collection 'col-1' no longer exists on this host, so the print was \
                         not filed into it; its tags and everything else were applied normally",
                    )
                    .set_body_bytes(b"fake-png".to_vec()),
            )
            .expect(1)
            .mount(&server)
            .await;

        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        let response = MoldClient::new(&server.uri())
            .generate(request)
            .await
            .unwrap();

        assert_eq!(response.seed_used, 42);
        assert_eq!(
            response.request_warnings,
            vec![
                "collection 'col-1' no longer exists on this host, so the print was not filed \
                 into it; its tags and everything else were applied normally"
                    .to_string()
            ],
            "the advisory reaches the caller whole"
        );
    }

    /// An ordinary generation carries no advisory, so the field stays empty
    /// and nothing is printed. Guards against the reporting path becoming
    /// per-print noise.
    #[tokio::test]
    async fn an_unwarned_generate_carries_no_advisories() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .insert_header("x-mold-seed-used", "7")
                    .set_body_bytes(b"fake-png".to_vec()),
            )
            .mount(&server)
            .await;

        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        let response = MoldClient::new(&server.uri())
            .generate(request)
            .await
            .unwrap();
        assert!(response.request_warnings.is_empty());
        // …and it stays off the wire when the response is re-serialized.
        let wire = serde_json::to_value(&response).unwrap();
        assert!(wire.get("request_warnings").is_none(), "{wire}");
    }

    #[tokio::test]
    async fn direct_generation_clients_reject_batches_before_network() {
        let server = wiremock::MockServer::start().await;
        let client = MoldClient::new(&server.uri());
        let mut request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"two cats","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        request.batch_size = 2;

        let raw = client.generate_raw(&request).await.unwrap_err().to_string();
        let blocking = client
            .generate(request.clone())
            .await
            .unwrap_err()
            .to_string();
        let (progress, _rx) = tokio::sync::mpsc::unbounded_channel();
        let streaming = client
            .generate_stream(&request, progress)
            .await
            .unwrap_err()
            .to_string();

        for error in [raw, blocking, streaming] {
            assert!(error.contains("batch_size = 1"), "{error}");
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn direct_generation_surfaces_detached_durable_authority() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "batch-detached",
                "client_batch_id": "operation-detached",
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "job-1",
                    "state": "accepted",
                    "created_at_ms": 10,
                    "updated_at_ms": 11
                }]
            })))
            .mount(&server)
            .await;
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();

        let error = MoldClient::new(&server.uri())
            .generate_raw(&request)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("batch-detached"), "{error}");
        assert!(error.contains("operation-detached"), "{error}");
    }

    #[tokio::test]
    async fn durable_generation_batch_methods_preserve_typed_ids_and_routes() {
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        let admission = GenerationBatchAdmissionRequest {
            client_batch_id: "00000000-0000-4000-8000-000000000001".into(),
            requests: vec![request],
        };
        let status = serde_json::json!({
            "id": "batch-1",
            "client_batch_id": admission.client_batch_id,
            "instance_id": "instance-1",
            "durable": true,
            "children": [{
                "index": 1,
                "job_id": "job-1",
                "state": "accepted",
                "created_at_ms": 1,
                "updated_at_ms": 1
            }]
        });
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .and(body_json(&admission))
            .respond_with(ResponseTemplate::new(202).set_body_json(&status))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!(
                "/api/generation-batches/by-client/{}",
                admission.client_batch_id
            )))
            .respond_with(ResponseTemplate::new(200).set_body_json(&status))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/queue/job-1/retry"))
            .and(body_json(serde_json::json!({
                "instance_id": "instance-1",
                "batch_id": "batch-1",
                "client_batch_id": admission.client_batch_id,
                "job_id": "job-1"
            })))
            .respond_with(ResponseTemplate::new(202))
            .expect(1)
            .mount(&server)
            .await;

        let client = MoldClient::new(&server.uri());
        let admitted = client.admit_generation_batch(&admission).await.unwrap();
        assert_eq!(admitted.id, "batch-1");
        assert_eq!(
            client
                .generation_batch_by_client_id(&admission.client_batch_id)
                .await
                .unwrap()
                .unwrap()
                .children[0]
                .job_id,
            "job-1"
        );
        client
            .retry_queue_job(&GenerationRetryRequest {
                instance_id: "instance-1".into(),
                batch_id: "batch-1".into(),
                client_batch_id: admission.client_batch_id.clone(),
                job_id: "job-1".into(),
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn generation_batch_missing_route_preserves_compatibility_status() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(ResponseTemplate::new(405).set_body_string("old host"))
            .expect(1)
            .mount(&server)
            .await;
        let request: GenerateRequest = serde_json::from_str(
            r#"{"prompt":"a cat","model":"flux-dev:q4","width":64,"height":64,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        let error = MoldClient::new(&server.uri())
            .admit_generation_batch(&GenerationBatchAdmissionRequest {
                client_batch_id: "client-1".into(),
                requests: vec![request],
            })
            .await
            .unwrap_err();
        assert!(super::is_missing_endpoint_error(&error));
        assert!(error.to_string().contains("405 Method Not Allowed"));
    }
}
