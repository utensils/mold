//! Durable framewise video upscaling using the native Real-ESRGAN engine.
//!
//! ffmpeg is a bounded codec bridge only: decoded RGB frames cross one at a
//! time and every super-resolution operation stays in the native Candle path.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use image::ImageEncoder as _;
use mold_core::{
    CreateVideoUpscaleJobRequest, OutputFormat, UpscaleRequest, VideoUpscaleJob,
    VideoUpscaleJobState, VideoUpscaleMediaFacts, VideoUpscaleSource,
    VIDEO_UPSCALE_CONTRACT_VERSION, VIDEO_UPSCALE_DISCLOSURE,
};
use mold_db::video_upscale_jobs::{self as jobs, StoredVideoUpscaleJob};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    convert::Infallible,
    fs,
    io::{BufRead, Read, Write},
    path::{Path as FsPath, PathBuf},
    process::{Command, Stdio},
};

use crate::{routes::ApiError, state::AppState};

const CHECKPOINT_FRAMES: u64 = 16;
const MAX_SOURCE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_OUTPUT_PIXELS: u64 = 35_389_440;
const MAX_FRAME_BYTES: u64 = 128 * 1024 * 1024;
const DISK_SAFETY_BYTES: u64 = 1024 * 1024 * 1024;
const SOURCE_DIGEST_FILE: &str = "source.sha256";
const SOURCE_METADATA_FILE: &str = "source-metadata.json";

fn now_ms() -> i64 {
    mold_core::time::now_epoch_ms_u64() as i64
}
fn db(state: &AppState) -> Result<&mold_db::MetadataDb, ApiError> {
    state.metadata_db.as_ref().as_ref().ok_or_else(|| {
        ApiError::internal_with_status(
            "Framewise upscale requires the durable metadata database",
            StatusCode::SERVICE_UNAVAILABLE,
        )
    })
}

pub struct VideoUpscaleRuntime {
    shutdown: tokio_util::sync::CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl VideoUpscaleRuntime {
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let _ = self.handle.await;
    }
}

pub fn recover_at_startup(state: &AppState) -> Option<VideoUpscaleRuntime> {
    let db = state.metadata_db.as_ref().as_ref()?;
    match jobs::pause_unfinished_for_recovery(db, now_ms()) {
        Ok(count) if count > 0 => tracing::info!(
            count,
            "paused unfinished framewise upscale jobs for explicit recovery"
        ),
        Ok(_) => {}
        Err(error) => tracing::error!(%error, "failed to recover framewise upscale jobs"),
    }

    // A generated video is published from either the async single-worker path
    // or a dedicated GPU owner thread. Both can durably enqueue the follow-up
    // in SQLite; this lightweight dispatcher supplies the AppState needed to
    // run it without making generation completion depend on a connected UI.
    let shutdown = tokio_util::sync::CancellationToken::new();
    let handle = tokio::spawn(run_dispatcher(
        state.clone(),
        shutdown.clone(),
        state.events.shutdown_token(),
        std::sync::Arc::new(|state, id| async move { run_tracked_job(state, id).await }),
    ));
    Some(VideoUpscaleRuntime { shutdown, handle })
}

async fn run_dispatcher<F, Fut>(
    state: AppState,
    lifecycle_shutdown: tokio_util::sync::CancellationToken,
    server_shutdown: tokio_util::sync::CancellationToken,
    runner: std::sync::Arc<F>,
) where
    F: Fn(AppState, String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
    let mut active = HashSet::new();
    let mut tasks = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            _ = lifecycle_shutdown.cancelled() => break,
            _ = server_shutdown.cancelled() => break,
            completed = tasks.join_next(), if !tasks.is_empty() => {
                match completed {
                    Some(Ok(id)) => { active.remove(&id); }
                    Some(Err(error)) => tracing::error!(%error, "framewise upscale task panicked"),
                    None => {}
                }
            }
            _ = interval.tick() => {
                let Some(database) = state.metadata_db.as_ref().as_ref() else {
                    break;
                };
                let queued = match jobs::list_queued_ids(database) {
                    Ok(ids) => ids,
                    Err(error) => {
                        tracing::warn!(%error, "could not scan queued framewise upscale jobs");
                        continue;
                    }
                };
                for id in queued {
                    if active.insert(id.clone()) {
                        let state = state.clone();
                        let runner = runner.clone();
                        tasks.spawn(async move {
                            runner(state, id.clone()).await;
                            id
                        });
                    }
                }
            }
        }
    }

    // Parking is the cancellation signal observed by run_job at frame
    // boundaries. Finalizing rows have crossed the no-cancel publication
    // boundary, so leave them claimed and drain them through completion. Drain
    // every tracked task before returning so an embedded restart cannot inherit
    // old DB, GPU, ffmpeg, or gallery owners.
    if let Some(database) = state.metadata_db.as_ref().as_ref() {
        if let Err(error) = jobs::pause_interruptible_for_shutdown(database, now_ms()) {
            tracing::error!(%error, "failed to park framewise upscale jobs for shutdown");
        }
    }
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            tracing::error!(%error, "framewise upscale task panicked during shutdown");
        }
    }
}

fn scale_for(model: &str) -> u32 {
    let lower = model.to_ascii_lowercase();
    if lower.contains("x2plus") || lower.contains("-x2") {
        2
    } else {
        4
    }
}

fn hash_pinned_source(source: &FsPath) -> anyhow::Result<(u64, String)> {
    let metadata = fs::metadata(source)?;
    anyhow::ensure!(metadata.is_file(), "source is not a regular file");
    anyhow::ensure!(
        metadata.len() > 0 && metadata.len() <= MAX_SOURCE_BYTES,
        "source exceeds the 2 GiB durable-job limit"
    );
    let mut input = fs::File::open(source)?;
    let mut digest = Sha256::new();
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        copied = copied
            .checked_add(read as u64)
            .ok_or_else(|| anyhow::anyhow!("source byte count overflow"))?;
        anyhow::ensure!(
            copied <= MAX_SOURCE_BYTES,
            "source changed beyond the 2 GiB limit"
        );
        digest.update(&buffer[..read]);
    }
    Ok((copied, format!("{:x}", digest.finalize())))
}

fn source_sha256(path: &FsPath) -> anyhow::Result<String> {
    let mut source = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_image_upscale_dimensions(width: u32, height: u32, scale: u32) -> Result<(), ApiError> {
    let output_pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(u64::from(scale).pow(2)))
        .ok_or_else(|| ApiError::validation("upscaled image dimensions overflow"))?;
    if output_pixels > MAX_OUTPUT_PIXELS {
        return Err(ApiError::validation(format!(
            "upscaled image would contain {output_pixels} pixels; the limit is {MAX_OUTPUT_PIXELS}"
        )));
    }
    Ok(())
}

fn validate_resource_bounds(
    facts: &VideoUpscaleMediaFacts,
    scale: u32,
    source_bytes: u64,
    available_bytes: u64,
) -> anyhow::Result<()> {
    let source_pixels = u64::from(facts.width)
        .checked_mul(u64::from(facts.height))
        .ok_or_else(|| anyhow::anyhow!("source pixel count overflow"))?;
    let output_pixels = source_pixels
        .checked_mul(u64::from(scale).pow(2))
        .ok_or_else(|| anyhow::anyhow!("output pixel count overflow"))?;
    anyhow::ensure!(
        output_pixels <= MAX_OUTPUT_PIXELS,
        "output would contain {output_pixels} pixels per frame; the limit is {MAX_OUTPUT_PIXELS}"
    );
    let source_frame_bytes = source_pixels
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("source frame byte count overflow"))?;
    let output_frame_bytes = output_pixels
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("output frame byte count overflow"))?;
    anyhow::ensure!(
        source_frame_bytes <= MAX_FRAME_BYTES && output_frame_bytes <= MAX_FRAME_BYTES,
        "decoded frame memory exceeds the 128 MiB per-frame limit"
    );
    let scaled_media = source_bytes
        .checked_mul(u64::from(scale).pow(2))
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("workspace estimate overflow"))?;
    let checkpoint_workspace = output_frame_bytes
        .checked_mul(CHECKPOINT_FRAMES)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("checkpoint workspace estimate overflow"))?;
    let required = source_bytes
        .checked_add(scaled_media)
        .and_then(|bytes| bytes.checked_add(checkpoint_workspace))
        .and_then(|bytes| bytes.checked_add(DISK_SAFETY_BYTES))
        .ok_or_else(|| anyhow::anyhow!("workspace estimate overflow"))?;
    anyhow::ensure!(
        available_bytes >= required,
        "insufficient disk for durable Framewise upscale: need about {required} bytes, have {available_bytes}"
    );
    Ok(())
}

fn write_synced(path: &FsPath, bytes: &[u8]) -> anyhow::Result<()> {
    let partial = path.with_extension("partial");
    let _ = fs::remove_file(&partial);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(partial, path)?;
    Ok(())
}

fn prepare_pinned_source(
    pinned_source: &FsPath,
    work_dir: &FsPath,
    scale: u32,
) -> anyhow::Result<VideoUpscaleMediaFacts> {
    let (source_bytes, digest) = hash_pinned_source(pinned_source)?;
    let facts = probe(pinned_source)?;
    validate_resource_bounds(&facts, scale, source_bytes, fs2::available_space(work_dir)?)?;
    let digest_path = work_dir.join(SOURCE_DIGEST_FILE);
    if digest_path.exists() {
        anyhow::ensure!(
            fs::read_to_string(&digest_path)?.trim() == digest,
            "durable Framewise upscale source identity changed during preparation"
        );
    } else {
        write_synced(&digest_path, digest.as_bytes())?;
    }
    crate::dir_sync::sync_directory(work_dir)?;
    Ok(facts)
}

fn verify_pinned_source(stored: &StoredVideoUpscaleJob) -> anyhow::Result<()> {
    let expected = fs::read_to_string(stored.work_dir.join(SOURCE_DIGEST_FILE))?;
    let actual = source_sha256(&stored.source_path)?;
    anyhow::ensure!(
        expected.trim() == actual,
        "durable Framewise upscale source identity changed"
    );
    Ok(())
}

fn source_metadata(stored: &StoredVideoUpscaleJob) -> anyhow::Result<mold_core::OutputMetadata> {
    Ok(serde_json::from_slice(&fs::read(
        stored.work_dir.join(SOURCE_METADATA_FILE),
    )?)?)
}

async fn run_tracked_job(state: AppState, id: String) {
    if let Err(error) = run_job(state.clone(), &id).await {
        let current = state
            .metadata_db
            .as_ref()
            .as_ref()
            .and_then(|db| jobs::get(db, &id).ok().flatten())
            .map(|stored| stored.job.state);
        let terminal = current
            .is_some_and(|state| state == VideoUpscaleJobState::Paused || state.is_terminal());
        if !terminal {
            if let Some(db) = state.metadata_db.as_ref().as_ref() {
                if current == Some(VideoUpscaleJobState::Finalizing) {
                    let _ = jobs::pause_after_error(db, &id, &format!("{error:#}"), now_ms());
                } else {
                    let _ = jobs::fail(db, &id, &format!("{error:#}"), now_ms());
                }
            }
        }
        tracing::warn!(job_id = %id, %error, "framewise upscale stopped");
    }
}

async fn create_library_job(
    state: &AppState,
    request: CreateVideoUpscaleJobRequest,
) -> Result<VideoUpscaleJob, ApiError> {
    if let Some(reason) = state
        .generation_unavailable_reason
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return Err(ApiError::generation_unavailable(reason));
    }
    mold_core::validate_create_video_upscale_job(&request).map_err(ApiError::validation)?;
    let model = mold_core::manifest::resolve_model_name(&request.model);
    let manifest = mold_core::manifest::find_manifest(&model)
        .ok_or_else(|| ApiError::not_found(format!("unknown upscaler model '{model}'")))?;
    if !mold_core::manifest::UPSCALER_FAMILIES.contains(&manifest.family.as_str()) {
        return Err(ApiError::validation("model is not a native image upscaler"));
    }
    mold_core::require_registered_manifest_activation(manifest)
        .map_err(ApiError::model_activation)?;
    let output_dir = state.config.read().await.effective_output_dir();
    match &request.source {
        VideoUpscaleSource::Library { filename } => enqueue_gallery_video_job(
            &output_dir,
            db(state)?,
            filename,
            model,
            request.tile_size,
        ),
        VideoUpscaleSource::Upload { .. } => Err(ApiError::structured(
            "Video upload handles are not advertised by this first capability; import the video into Library first",
            "VIDEO_UPSCALE_UPLOAD_UNAVAILABLE", StatusCode::NOT_IMPLEMENTED, None, None)),
    }
}

pub(crate) fn enqueue_gallery_video_job(
    output_dir: &FsPath,
    database: &mold_db::MetadataDb,
    filename: &str,
    model: String,
    tile_size: Option<u32>,
) -> Result<VideoUpscaleJob, ApiError> {
    let source_metadata = database
        .get(output_dir, filename)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .ok_or_else(|| ApiError::not_found("library source is not a committed gallery item"))?
        .metadata;
    let candidate = output_dir.join(filename);
    let root = output_dir
        .canonicalize()
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let source_path = candidate
        .canonicalize()
        .map_err(|_| ApiError::not_found("library source media is missing"))?;
    if !source_path.starts_with(&root) || !source_path.is_file() {
        return Err(ApiError::validation(
            "library source escaped gallery authority",
        ));
    }
    let id = format!("vup-{}", uuid::Uuid::new_v4());
    let work_dir = output_dir.join(".mold-video-upscale-jobs").join(&id);
    fs::create_dir_all(&work_dir).map_err(|e| ApiError::internal(e.to_string()))?;
    let extension = source_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("mp4");
    let pinned_source = work_dir.join(format!("source.{extension}"));
    let admission = (|| -> anyhow::Result<()> {
        fs::hard_link(&source_path, &pinned_source)?;
        write_synced(
            &work_dir.join(SOURCE_METADATA_FILE),
            &serde_json::to_vec(&source_metadata)?,
        )?;
        crate::dir_sync::sync_directory(&work_dir)?;
        if let Some(parent) = work_dir.parent() {
            crate::dir_sync::sync_directory(parent)?;
        }
        Ok(())
    })();
    if let Err(error) = admission {
        let _ = fs::remove_dir_all(&work_dir);
        return Err(ApiError::internal(format!(
            "failed to pin Gallery source into durable Framewise job: {error}"
        )));
    }
    let now = now_ms();
    let stored = StoredVideoUpscaleJob {
        job: VideoUpscaleJob {
            contract_version: VIDEO_UPSCALE_CONTRACT_VERSION,
            id: id.clone(),
            state: VideoUpscaleJobState::Queued,
            source: VideoUpscaleSource::Library {
                filename: filename.to_string(),
            },
            scale_factor: scale_for(&model),
            model,
            tile_size,
            completed_frames: 0,
            total_frames: 0,
            source_facts: None,
            output_facts: None,
            output_filename: None,
            error: None,
            created_at_ms: now,
            updated_at_ms: now,
            disclosure: VIDEO_UPSCALE_DISCLOSURE.into(),
        },
        // A same-filesystem hard link pins the admitted Gallery bytes with a
        // cheap metadata operation before durable acknowledgement. Hashing,
        // probing, and bounds checks remain on the blocking preparation pool.
        source_path: pinned_source,
        work_dir,
    };
    if let Err(error) = jobs::insert(database, &stored) {
        let _ = fs::remove_dir_all(&stored.work_dir);
        return Err(ApiError::internal(error.to_string()));
    }
    Ok(stored.job)
}

pub(crate) async fn create_generated_video_job(
    state: &AppState,
    filename: String,
    model: String,
) -> Result<VideoUpscaleJob, ApiError> {
    let job = create_library_job(
        state,
        CreateVideoUpscaleJobRequest {
            source: VideoUpscaleSource::Library { filename },
            model,
            tile_size: None,
        },
    )
    .await?;
    Ok(job)
}

pub async fn create_job(
    State(state): State<AppState>,
    Json(request): Json<CreateVideoUpscaleJobRequest>,
) -> Result<(StatusCode, Json<VideoUpscaleJob>), ApiError> {
    let job = create_library_job(&state, request).await?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct CreateGalleryImageUpscaleRequest {
    filename: String,
    model: String,
    #[serde(default)]
    tile_size: Option<u32>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct GalleryImageUpscaleResponse {
    filename: String,
    model: String,
    scale_factor: u32,
}

pub async fn upscale_gallery_image(
    State(state): State<AppState>,
    Json(request): Json<CreateGalleryImageUpscaleRequest>,
) -> Result<(StatusCode, Json<GalleryImageUpscaleResponse>), ApiError> {
    let filename = request.filename.trim();
    if filename != request.filename
        || filename.is_empty()
        || filename.contains('/')
        || filename.contains('\\')
    {
        return Err(ApiError::validation(
            "image upscale source must be one exact gallery filename",
        ));
    }
    let extension = FsPath::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp") {
        return Err(ApiError::validation(
            "Library image upscale supports PNG, JPEG, and WebP sources",
        ));
    }
    if let Some(tile_size) = request.tile_size {
        if tile_size != 0 && tile_size < 64 {
            return Err(ApiError::validation(
                "tile_size must be 0 (disabled) or at least 64",
            ));
        }
    }
    let model = mold_core::manifest::resolve_model_name(&request.model);
    let manifest = mold_core::manifest::find_manifest(&model)
        .ok_or_else(|| ApiError::not_found(format!("unknown upscaler model '{model}'")))?;
    if !mold_core::manifest::UPSCALER_FAMILIES.contains(&manifest.family.as_str()) {
        return Err(ApiError::validation("model is not a native image upscaler"));
    }
    mold_core::require_registered_manifest_activation(manifest)
        .map_err(ApiError::model_activation)?;

    let output_dir = state.config.read().await.effective_output_dir();
    let database = db(&state)?;
    let mut metadata = database
        .get(&output_dir, filename)
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::not_found("library source is not a committed gallery item"))?
        .metadata;
    let root = output_dir
        .canonicalize()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let source_path = output_dir
        .join(filename)
        .canonicalize()
        .map_err(|_| ApiError::not_found("library source media is missing"))?;
    if !source_path.starts_with(&root) || !source_path.is_file() {
        return Err(ApiError::validation(
            "library source escaped gallery authority",
        ));
    }
    let source_bytes = fs::read(&source_path)
        .map_err(|_| ApiError::not_found("library source media is missing"))?;
    if source_bytes.len() as u64 > 64 * 1024 * 1024 {
        return Err(ApiError::validation(
            "Library image exceeds the 64 MiB upscale limit",
        ));
    }
    let scale = scale_for(&model);
    let (source_width, source_height) =
        image::ImageReader::new(std::io::Cursor::new(&source_bytes))
            .with_guessed_format()
            .map_err(|error| ApiError::validation(format!("invalid source image: {error}")))?
            .into_dimensions()
            .map_err(|error| ApiError::validation(format!("invalid source image: {error}")))?;
    validate_image_upscale_dimensions(source_width, source_height, scale)?;
    // Full decoding happens only after the header dimensions prove that the
    // expanded image fits the same per-frame policy as video upscaling.
    let source = image::load_from_memory(&source_bytes)
        .map_err(|error| ApiError::validation(format!("invalid source image: {error}")))?;
    if crate::model_manager::upscaler_model_needs_pull(&*state.config.read().await, &model) {
        crate::model_manager::pull_model(&state, &model, None).await?;
    }
    let weights = state
        .config
        .read()
        .await
        .models
        .get(&model)
        .and_then(|config| config.transformer.as_ref())
        .map(PathBuf::from)
        .ok_or_else(|| ApiError::internal("upscaler model is not configured"))?;
    let upscaled = upscale_frame(&state, &model, &weights, source_bytes, request.tile_size).await?;
    let output = image::load_from_memory(&upscaled)
        .map_err(|error| ApiError::internal(format!("invalid upscaler output: {error}")))?;
    metadata.upscale_model = Some(model.clone());
    metadata.generation_width = Some(source.width());
    metadata.generation_height = Some(source.height());
    metadata.width = output.width();
    metadata.height = output.height();
    metadata.output_format = Some(OutputFormat::Png);
    metadata.job_id = None;
    let image = mold_core::ImageData {
        data: upscaled,
        format: OutputFormat::Png,
        width: output.width(),
        height: output.height(),
        index: 0,
    };
    let saved = crate::queue::save_image_to_dir_with_suffix(
        &output_dir,
        &image,
        &model,
        1,
        Some("upscaled"),
        Some(&metadata),
        None,
        Some(database),
        Some(&state.events),
        &state.gallery_publication_gate,
    )
    .ok_or_else(|| ApiError::internal("upscaled image could not be published"))?;
    Ok((
        StatusCode::CREATED,
        Json(GalleryImageUpscaleResponse {
            filename: saved,
            model,
            scale_factor: scale,
        }),
    ))
}

pub async fn list_jobs(
    State(state): State<AppState>,
) -> Result<Json<Vec<VideoUpscaleJob>>, ApiError> {
    Ok(Json(
        jobs::list(db(&state)?).map_err(|e| ApiError::internal(e.to_string()))?,
    ))
}

pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VideoUpscaleJob>, ApiError> {
    jobs::get(db(&state)?, &id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map(|stored| Json(stored.job))
        .ok_or_else(|| ApiError::not_found("framewise upscale job not found"))
}

async fn transition_job(
    state: &AppState,
    id: &str,
    expected: &[VideoUpscaleJobState],
    next: VideoUpscaleJobState,
) -> Result<VideoUpscaleJob, ApiError> {
    if !jobs::transition(db(state)?, id, expected, next, now_ms())
        .map_err(|e| ApiError::internal(e.to_string()))?
    {
        return Err(ApiError::structured(
            "framewise upscale job is not in a compatible state",
            "VIDEO_UPSCALE_STATE_CONFLICT",
            StatusCode::CONFLICT,
            None,
            None,
        ));
    }
    Ok(jobs::get(db(state)?, id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .expect("transitioned row exists")
        .job)
}

pub async fn pause_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VideoUpscaleJob>, ApiError> {
    Ok(Json(
        transition_job(
            &state,
            &id,
            &[VideoUpscaleJobState::Queued, VideoUpscaleJobState::Running],
            VideoUpscaleJobState::Paused,
        )
        .await?,
    ))
}

pub async fn resume_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VideoUpscaleJob>, ApiError> {
    let job = transition_job(
        &state,
        &id,
        &[VideoUpscaleJobState::Paused],
        VideoUpscaleJobState::Queued,
    )
    .await?;
    Ok(Json(job))
}

pub async fn cancel_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<VideoUpscaleJob>, ApiError> {
    let before = jobs::get(db(&state)?, &id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .ok_or_else(|| ApiError::not_found("framewise upscale job not found"))?;
    let job = transition_job(
        &state,
        &id,
        &[
            VideoUpscaleJobState::Queued,
            VideoUpscaleJobState::Running,
            VideoUpscaleJobState::Paused,
        ],
        VideoUpscaleJobState::Cancelled,
    )
    .await?;
    // No process owns a queued/paused job's files. A running job observes
    // cancellation at its next frame boundary and cleans up there.
    if before.job.state != VideoUpscaleJobState::Running {
        let _ = fs::remove_dir_all(before.work_dir);
    }
    Ok(Json(job))
}

pub async fn job_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if jobs::get(db(&state)?, &id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .is_none()
    {
        return Err(ApiError::not_found("framewise upscale job not found"));
    }
    let stream = async_stream::stream! {
        let mut last = None;
        loop {
            let current = state.metadata_db.as_ref().as_ref().and_then(|db| jobs::get(db, &id).ok().flatten()).map(|s| s.job);
            let Some(job) = current else { break };
            let json = serde_json::to_string(&job).unwrap_or_else(|_| "{}".into());
            if last.as_ref() != Some(&json) {
                yield Ok(Event::default().event("status").data(json.clone()));
                last = Some(json);
            }
            if job.state.is_terminal() { break }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Debug, Deserialize)]
struct Probe {
    streams: Vec<ProbeStream>,
    format: ProbeFormat,
    #[serde(default)]
    chapters: Vec<serde_json::Value>,
}
#[derive(Debug, Deserialize)]
struct ProbeFormat {
    format_name: String,
    duration: Option<String>,
}
#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: String,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    pix_fmt: Option<String>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
    duration: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
    color_transfer: Option<String>,
}

fn rational(value: &str) -> anyhow::Result<(u64, u64)> {
    let (n, d) = value
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("invalid frame rate {value:?}"))?;
    let (n, d) = (n.parse::<u64>()?, d.parse::<u64>()?);
    anyhow::ensure!(n > 0 && d > 0, "invalid zero frame rate");
    Ok((n, d))
}

fn probe(path: &FsPath) -> anyhow::Result<VideoUpscaleMediaFacts> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| anyhow::anyhow!("ffprobe is required for Framewise upscale: {e}"))?;
    anyhow::ensure!(
        output.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let probe: Probe = serde_json::from_slice(&output.stdout)?;
    let video: Vec<_> = probe
        .streams
        .iter()
        .filter(|s| s.codec_type == "video")
        .collect();
    let audio: Vec<_> = probe
        .streams
        .iter()
        .filter(|s| s.codec_type == "audio")
        .collect();
    let subtitles = probe
        .streams
        .iter()
        .filter(|s| s.codec_type == "subtitle")
        .count();
    anyhow::ensure!(video.len() == 1, "exactly one video track is required");
    anyhow::ensure!(
        audio.len() <= 1,
        "multiple audio tracks are not preserved by this MVP"
    );
    anyhow::ensure!(
        subtitles == 0 && probe.chapters.is_empty(),
        "subtitles and chapters are not preserved by this MVP"
    );
    let video = video[0];
    let fps = video
        .avg_frame_rate
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("missing average frame rate"))?;
    let nominal = video
        .r_frame_rate
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("missing nominal frame rate"))?;
    anyhow::ensure!(
        rational(fps)? == rational(nominal)?,
        "variable-frame-rate sources are not supported"
    );
    let pix_fmt = video.pix_fmt.as_deref().unwrap_or_default();
    let transfer = video.color_transfer.as_deref().unwrap_or_default();
    anyhow::ensure!(
        !pix_fmt.contains("10")
            && !pix_fmt.contains("12")
            && !matches!(transfer, "smpte2084" | "arib-std-b67"),
        "HDR and high-bit-depth video are not supported"
    );
    let frame_count = video
        .nb_frames
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("source must report an exact frame count"))?
        .parse::<u64>()?;
    anyhow::ensure!(
        frame_count > 0 && frame_count <= 200_000,
        "source frame count is outside the supported range"
    );
    validate_constant_frame_timestamps(path, fps, frame_count)?;
    let duration = video
        .duration
        .as_deref()
        .or(probe.format.duration.as_deref())
        .ok_or_else(|| anyhow::anyhow!("missing source duration"))?
        .parse::<f64>()?;
    let audio_codec = audio.first().and_then(|s| s.codec_name.clone());
    if let Some(codec) = audio_codec.as_deref() {
        anyhow::ensure!(
            matches!(codec, "aac" | "mp3" | "ac3" | "eac3" | "alac"),
            "primary audio codec {codec:?} cannot be copied safely into MP4"
        );
    }
    Ok(VideoUpscaleMediaFacts {
        container: probe.format.format_name,
        video_codec: video.codec_name.clone().unwrap_or_default(),
        width: video
            .width
            .ok_or_else(|| anyhow::anyhow!("missing width"))?,
        height: video
            .height
            .ok_or_else(|| anyhow::anyhow!("missing height"))?,
        frame_count,
        fps: fps.into(),
        duration_ms: (duration * 1000.0).round() as u64,
        primary_audio_codec: audio_codec,
        primary_audio_sample_rate: audio
            .first()
            .and_then(|s| s.sample_rate.as_deref())
            .and_then(|s| s.parse().ok()),
        primary_audio_channels: audio.first().and_then(|s| s.channels),
    })
}

fn validate_constant_frame_timestamps(
    path: &FsPath,
    fps: &str,
    expected_frames: u64,
) -> anyhow::Result<()> {
    let (numerator, denominator) = rational(fps)?;
    let expected_delta = denominator as f64 / numerator as f64;
    let tolerance = (expected_delta * 0.001).max(0.000_002);
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("ffprobe timestamp stdout missing"))?;
    let mut previous: Option<f64> = None;
    let mut seen = 0_u64;
    let mut variable_rate = false;
    for line in std::io::BufReader::new(stdout).lines() {
        // Some ffprobe builds append an empty CSV column when frame side data
        // is present (for example `0.000000,`). Only the requested first field
        // is the timestamp.
        let line = line?;
        let timestamp = line
            .split(',')
            .next()
            .unwrap_or_default()
            .trim()
            .parse::<f64>()?;
        if let Some(previous) = previous {
            variable_rate |= (timestamp - previous - expected_delta).abs() > tolerance;
        }
        previous = Some(timestamp);
        seen += 1;
    }
    let output = child.wait_with_output()?;
    anyhow::ensure!(
        output.status.success(),
        "ffprobe frame-timestamp scan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    anyhow::ensure!(
        !variable_rate,
        "variable-frame-rate sources are not supported"
    );
    anyhow::ensure!(
        seen == expected_frames,
        "source frame count changed while it was being admitted"
    );
    Ok(())
}

fn verify_preserved_facts(
    source: &VideoUpscaleMediaFacts,
    output: &VideoUpscaleMediaFacts,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        output.frame_count == source.frame_count && output.fps == source.fps,
        "output did not preserve frame count/FPS"
    );
    anyhow::ensure!(
        output.primary_audio_codec == source.primary_audio_codec
            && output.primary_audio_sample_rate == source.primary_audio_sample_rate
            && output.primary_audio_channels == source.primary_audio_channels,
        "output did not preserve primary audio facts"
    );
    let (n, d) = rational(&source.fps)?;
    let frame_ms = (1000 * d).div_ceil(n);
    anyhow::ensure!(
        output.duration_ms.abs_diff(source.duration_ms) <= frame_ms + 2,
        "output duration drift exceeded one frame"
    );
    Ok(())
}

fn current_state(db: &mold_db::MetadataDb, id: &str) -> anyhow::Result<VideoUpscaleJobState> {
    jobs::get(db, id)?
        .map(|stored| stored.job.state)
        .ok_or_else(|| anyhow::anyhow!("job disappeared"))
}

fn should_stop(db: &mold_db::MetadataDb, id: &str, work_dir: &FsPath) -> anyhow::Result<bool> {
    let state = current_state(db, id)?;
    if state == VideoUpscaleJobState::Cancelled {
        let _ = fs::remove_dir_all(work_dir);
    }
    Ok(state != VideoUpscaleJobState::Running)
}

fn spawn_decoder(source: &FsPath, start: u64, count: u64) -> anyhow::Result<std::process::Child> {
    let filter = format!(
        "trim=start_frame={start}:end_frame={},setpts=PTS-STARTPTS",
        start + count
    );
    Ok(Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(source)
        .args([
            "-map", "0:v:0", "-vf", &filter, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?)
}

fn spawn_encoder(
    path: &FsPath,
    width: u32,
    height: u32,
    fps: &str,
) -> anyhow::Result<std::process::Child> {
    Ok(Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{width}x{height}"),
            "-r",
            fps,
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?)
}

async fn upscale_frame(
    state: &AppState,
    model: &str,
    weights: &FsPath,
    image: Vec<u8>,
    tile_size: Option<u32>,
) -> Result<Vec<u8>, ApiError> {
    let request = UpscaleRequest {
        model: model.into(),
        image,
        output_format: OutputFormat::Png,
        tile_size,
        metadata: None,
    };
    let response = if state.scheduled_work.v2_authoritative() || state.gpu_pool.worker_count() > 0 {
        crate::routes::schedule_standalone_upscale(
            state,
            model.into(),
            weights.into(),
            request,
            None,
        )
        .await?
    } else {
        let cache = state.upscaler_cache.clone();
        let model = model.to_string();
        let weights = weights.to_path_buf();
        let artifact_root = state.config.read().await.resolved_models_dir();
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache
                .as_ref()
                .is_none_or(|engine| engine.model_name() != model)
            {
                *cache = Some(mold_inference::create_upscale_engine(
                    model,
                    weights,
                    Some(&artifact_root),
                    mold_inference::LoadStrategy::Eager,
                    0,
                )?);
            }
            cache.as_mut().unwrap().upscale(&request)
        })
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
        .map_err(|e| ApiError::internal(e.to_string()))?
    };
    Ok(response.image.data)
}

async fn run_job(state: AppState, id: &str) -> anyhow::Result<()> {
    let database = db(&state).map_err(|e| anyhow::anyhow!(e.error))?;
    if !jobs::transition(
        database,
        id,
        &[VideoUpscaleJobState::Queued],
        VideoUpscaleJobState::Running,
        now_ms(),
    )? {
        return Ok(());
    }
    let mut stored = jobs::get(database, id)?.ok_or_else(|| anyhow::anyhow!("job disappeared"))?;
    let facts = match stored.job.source_facts.clone() {
        Some(facts) => {
            verify_pinned_source(&stored)?;
            facts
        }
        None => {
            let pinned_source = stored.source_path.clone();
            let work_dir = stored.work_dir.clone();
            let scale = stored.job.scale_factor;
            let prepared = tokio::task::spawn_blocking(move || {
                prepare_pinned_source(&pinned_source, &work_dir, scale)
                    .map(|facts| (pinned_source, facts))
            })
            .await
            .map_err(|error| anyhow::anyhow!("Framewise preparation task failed: {error}"))??;
            if !jobs::finish_preparation(database, id, &prepared.0, &prepared.1, now_ms())? {
                if jobs::get(database, id)?
                    .is_some_and(|stored| stored.job.state == VideoUpscaleJobState::Cancelled)
                {
                    let _ = fs::remove_dir_all(&stored.work_dir);
                }
                return Ok(());
            }
            stored = jobs::get(database, id)?
                .ok_or_else(|| anyhow::anyhow!("prepared job disappeared"))?;
            verify_pinned_source(&stored)?;
            prepared.1
        }
    };
    let model = stored.job.model.clone();
    let needs_pull = {
        let config = state.config.read().await;
        crate::model_manager::upscaler_model_needs_pull(&config, &model)
    };
    if needs_pull {
        crate::model_manager::pull_model(&state, &model, None)
            .await
            .map_err(|e| anyhow::anyhow!(e.error))?;
    }
    let weights = state
        .config
        .read()
        .await
        .models
        .get(&model)
        .and_then(|c| c.transformer.as_ref())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("upscaler model is not configured"))?;
    let out_width = facts
        .width
        .checked_mul(stored.job.scale_factor)
        .ok_or_else(|| anyhow::anyhow!("output width overflow"))?;
    let out_height = facts
        .height
        .checked_mul(stored.job.scale_factor)
        .ok_or_else(|| anyhow::anyhow!("output height overflow"))?;
    anyhow::ensure!(
        out_width % 2 == 0 && out_height % 2 == 0,
        "H.264 output dimensions must be even"
    );
    let frame_bytes = usize::try_from(u64::from(facts.width) * u64::from(facts.height) * 3)?;
    let mut completed = stored.job.completed_frames;
    for start in (0..completed).step_by(CHECKPOINT_FRAMES as usize) {
        if !stored
            .work_dir
            .join(format!("chunk-{start:012}.mp4"))
            .is_file()
        {
            completed = start;
            jobs::update_progress(database, id, completed, now_ms())?;
            break;
        }
    }
    while completed < facts.frame_count {
        if should_stop(database, id, &stored.work_dir)? {
            return Ok(());
        }
        let count = CHECKPOINT_FRAMES.min(facts.frame_count - completed);
        let partial = stored
            .work_dir
            .join(format!("chunk-{completed:012}.partial.mp4"));
        let chunk = stored.work_dir.join(format!("chunk-{completed:012}.mp4"));
        // A crash may leave an uncommitted partial, or a renamed chunk whose
        // progress transaction never committed. Re-render that checkpoint.
        let _ = fs::remove_file(&partial);
        let _ = fs::remove_file(&chunk);
        let mut decoder = spawn_decoder(&stored.source_path, completed, count)?;
        let mut encoder = spawn_encoder(&partial, out_width, out_height, &facts.fps)?;
        let mut decoder_out = decoder
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("decoder stdout missing"))?;
        let mut encoder_in = encoder
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("encoder stdin missing"))?;
        let mut raw = vec![0u8; frame_bytes];
        for _ in 0..count {
            if should_stop(database, id, &stored.work_dir)? {
                let _ = decoder.kill();
                let _ = encoder.kill();
                let _ = fs::remove_file(&partial);
                return Ok(());
            }
            decoder_out.read_exact(&mut raw)?;
            let mut png = Vec::new();
            image::codecs::png::PngEncoder::new(&mut png).write_image(
                &raw,
                facts.width,
                facts.height,
                image::ExtendedColorType::Rgb8,
            )?;
            let upscaled = upscale_frame(&state, &model, &weights, png, stored.job.tile_size)
                .await
                .map_err(|e| anyhow::anyhow!(e.error))?;
            let rgb = image::load_from_memory(&upscaled)?.to_rgb8();
            anyhow::ensure!(
                rgb.width() == out_width && rgb.height() == out_height,
                "upscaler returned unexpected dimensions"
            );
            encoder_in.write_all(rgb.as_raw())?;
        }
        drop(encoder_in);
        drop(decoder_out);
        let decoder_output = decoder.wait_with_output()?;
        let encoder_output = encoder.wait_with_output()?;
        anyhow::ensure!(
            decoder_output.status.success(),
            "frame decode failed: {}",
            String::from_utf8_lossy(&decoder_output.stderr)
        );
        anyhow::ensure!(
            encoder_output.status.success(),
            "chunk encode failed: {}",
            String::from_utf8_lossy(&encoder_output.stderr)
        );
        fs::rename(&partial, &chunk)?;
        completed += count;
        jobs::update_progress(database, id, completed, now_ms())?;
    }
    if should_stop(database, id, &stored.work_dir)? {
        return Ok(());
    }
    stored = jobs::get(database, id)?.ok_or_else(|| anyhow::anyhow!("job disappeared"))?;
    let concat_list = stored.work_dir.join("chunks.txt");
    let mut list = fs::File::create(&concat_list)?;
    for start in (0..facts.frame_count).step_by(CHECKPOINT_FRAMES as usize) {
        writeln!(list, "file 'chunk-{start:012}.mp4'")?;
    }
    list.sync_all()?;
    let video_only = stored.work_dir.join("video-only.mp4");
    let concat = Command::new("ffmpeg")
        .current_dir(&stored.work_dir)
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "chunks.txt",
            "-c",
            "copy",
        ])
        .arg(&video_only)
        .output()?;
    anyhow::ensure!(
        concat.status.success(),
        "chunk assembly failed: {}",
        String::from_utf8_lossy(&concat.stderr)
    );
    if should_stop(database, id, &stored.work_dir)? {
        return Ok(());
    }
    let final_path = stored.work_dir.join("final.mp4");
    let mut mux = Command::new("ffmpeg");
    mux.args(["-v", "error", "-y", "-i"]).arg(&video_only);
    if facts.primary_audio_codec.is_some() {
        mux.args(["-i"]).arg(&stored.source_path).args([
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
        ]);
    } else {
        mux.args(["-map", "0:v:0", "-c", "copy", "-movflags", "+faststart"]);
    }
    let muxed = mux.arg(&final_path).output()?;
    anyhow::ensure!(
        muxed.status.success(),
        "audio mux failed: {}",
        String::from_utf8_lossy(&muxed.stderr)
    );
    let output_facts = probe(&final_path)?;
    verify_preserved_facts(&facts, &output_facts)?;
    if should_stop(database, id, &stored.work_dir)? {
        return Ok(());
    }
    // Once finalization is claimed, cancellation can no longer race gallery
    // publication. A crash pauses this state at startup; Resume replays the
    // deterministic staged artifact and idempotent publication path.
    if !jobs::transition(
        database,
        id,
        &[VideoUpscaleJobState::Running],
        VideoUpscaleJobState::Finalizing,
        now_ms(),
    )? {
        return Ok(());
    }
    let (n, d) = rational(&facts.fps)?;
    let output_dir = state.config.read().await.effective_output_dir();
    let source_name = match &stored.job.source {
        VideoUpscaleSource::Library { filename } => filename.clone(),
        VideoUpscaleSource::Upload { .. } => "upload.mp4".into(),
    };
    let mut metadata = source_metadata(&stored)?;
    metadata.upscale_model = Some(model.clone());
    metadata.job_id = Some(id.into());
    metadata.source_video_path = Some(source_name.clone());
    metadata.width = out_width;
    metadata.height = out_height;
    metadata.generation_width = Some(facts.width);
    metadata.generation_height = Some(facts.height);
    metadata.frames = u32::try_from(facts.frame_count).ok();
    metadata.fps = u32::try_from(n.saturating_add(d / 2) / d).ok();
    metadata.output_format = Some(OutputFormat::Mp4);
    let stem = FsPath::new(&source_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video");
    let filename = format!(
        "{stem}-framewise-upscaled-{}.mp4",
        &id[id.len().saturating_sub(8)..]
    );
    crate::queue::publish_video_path_to_dir_named(
        &output_dir,
        &filename,
        &final_path,
        OutputFormat::Mp4,
        &metadata,
        None,
        Some(database),
        Some(&state.events),
        &state.gallery_publication_gate,
    )?;
    anyhow::ensure!(
        jobs::complete(database, id, &filename, &output_facts, now_ms())?,
        "job left finalizing state during publication"
    );
    let _ = fs::remove_dir_all(&stored.work_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::ModelConfig;
    use std::sync::Mutex;

    static FFMPEG_FIXTURE_LOCK: Mutex<()> = Mutex::new(());

    fn ffmpeg_fixture_tools_available() -> bool {
        ["ffmpeg", "ffprobe"].into_iter().all(|tool| {
            Command::new(tool)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
        })
    }

    #[tokio::test]
    async fn shutdown_parks_and_joins_active_jobs_before_restart() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.mp4");
        fs::write(&source, b"source-placeholder").unwrap();
        let database = mold_db::MetadataDb::open_in_memory().unwrap();
        let metadata = mold_db::metadata_io::synthesize_from_filename("source.mp4", 1);
        let record = mold_db::GenerationRecord::from_save(
            temp.path(),
            "source.mp4",
            OutputFormat::Mp4,
            metadata,
            mold_db::RecordSource::Server,
            1,
        );
        database.upsert(&record).unwrap();

        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(database));
        let lifecycle_shutdown = tokio_util::sync::CancellationToken::new();
        let server_shutdown = tokio_util::sync::CancellationToken::new();
        let started = Arc::new(tokio::sync::Notify::new());
        let stopped = Arc::new(tokio::sync::Notify::new());
        let runs = Arc::new(AtomicUsize::new(0));
        let runner = {
            let started = started.clone();
            let stopped = stopped.clone();
            let runs = runs.clone();
            Arc::new(move |state: AppState, id: String| {
                let started = started.clone();
                let stopped = stopped.clone();
                let runs = runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    let database = state.metadata_db.as_ref().as_ref().unwrap();
                    assert!(jobs::transition(
                        database,
                        &id,
                        &[VideoUpscaleJobState::Queued],
                        VideoUpscaleJobState::Running,
                        now_ms(),
                    )
                    .unwrap());
                    started.notify_one();
                    loop {
                        if jobs::get(database, &id).unwrap().unwrap().job.state
                            == VideoUpscaleJobState::Paused
                        {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    stopped.notify_one();
                }
            })
        };
        let handle = tokio::spawn(run_dispatcher(
            state.clone(),
            lifecycle_shutdown.clone(),
            server_shutdown,
            runner,
        ));
        let runtime = VideoUpscaleRuntime {
            shutdown: lifecycle_shutdown,
            handle,
        };
        let job = enqueue_gallery_video_job(
            temp.path(),
            state.metadata_db.as_ref().as_ref().unwrap(),
            "source.mp4",
            "real-esrgan-x4plus:fp16".into(),
            None,
        )
        .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(2), started.notified())
            .await
            .expect("queued job must start");
        runtime.shutdown().await;
        tokio::time::timeout(std::time::Duration::from_secs(2), stopped.notified())
            .await
            .expect("shutdown must join the parked active job");
        assert_eq!(
            jobs::get(state.metadata_db.as_ref().as_ref().unwrap(), &job.id)
                .unwrap()
                .unwrap()
                .job
                .state,
            VideoUpscaleJobState::Paused
        );
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        let restarted = recover_at_startup(&state).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        restarted.shutdown().await;
        assert_eq!(
            runs.load(Ordering::SeqCst),
            1,
            "a restart must not auto-resume parked work"
        );
    }

    #[tokio::test]
    async fn shutdown_drains_claimed_finalization_before_returning() {
        use std::sync::Arc;

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.mp4");
        fs::write(&source, b"source-placeholder").unwrap();
        let database = mold_db::MetadataDb::open_in_memory().unwrap();
        let metadata = mold_db::metadata_io::synthesize_from_filename("source.mp4", 1);
        let record = mold_db::GenerationRecord::from_save(
            temp.path(),
            "source.mp4",
            OutputFormat::Mp4,
            metadata,
            mold_db::RecordSource::Server,
            1,
        );
        database.upsert(&record).unwrap();

        let mut state = AppState::for_tests();
        state.metadata_db = Arc::new(Some(database));
        let lifecycle_shutdown = tokio_util::sync::CancellationToken::new();
        let finalizing = Arc::new(tokio::sync::Notify::new());
        let allow_completion = Arc::new(tokio::sync::Notify::new());
        let runner = {
            let finalizing = finalizing.clone();
            let allow_completion = allow_completion.clone();
            Arc::new(move |state: AppState, id: String| {
                let finalizing = finalizing.clone();
                let allow_completion = allow_completion.clone();
                async move {
                    let database = state.metadata_db.as_ref().as_ref().unwrap();
                    assert!(jobs::transition(
                        database,
                        &id,
                        &[VideoUpscaleJobState::Queued],
                        VideoUpscaleJobState::Running,
                        now_ms(),
                    )
                    .unwrap());
                    assert!(jobs::transition(
                        database,
                        &id,
                        &[VideoUpscaleJobState::Running],
                        VideoUpscaleJobState::Finalizing,
                        now_ms(),
                    )
                    .unwrap());
                    finalizing.notify_one();
                    allow_completion.notified().await;
                    let facts = VideoUpscaleMediaFacts {
                        container: "mov,mp4,m4a,3gp,3g2,mj2".into(),
                        video_codec: "h264".into(),
                        width: 64,
                        height: 64,
                        frame_count: 5,
                        fps: "5/1".into(),
                        duration_ms: 1_000,
                        primary_audio_codec: None,
                        primary_audio_sample_rate: None,
                        primary_audio_channels: None,
                    };
                    assert!(jobs::complete(database, &id, "out.mp4", &facts, now_ms()).unwrap());
                }
            })
        };
        let handle = tokio::spawn(run_dispatcher(
            state.clone(),
            lifecycle_shutdown.clone(),
            tokio_util::sync::CancellationToken::new(),
            runner,
        ));
        let runtime = VideoUpscaleRuntime {
            shutdown: lifecycle_shutdown,
            handle,
        };
        let job = enqueue_gallery_video_job(
            temp.path(),
            state.metadata_db.as_ref().as_ref().unwrap(),
            "source.mp4",
            "real-esrgan-x4plus:fp16".into(),
            None,
        )
        .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(2), finalizing.notified())
            .await
            .expect("job must claim finalization");
        let shutdown = tokio::spawn(runtime.shutdown());
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!shutdown.is_finished(), "shutdown must await finalization");
        assert_eq!(
            jobs::get(state.metadata_db.as_ref().as_ref().unwrap(), &job.id)
                .unwrap()
                .unwrap()
                .job
                .state,
            VideoUpscaleJobState::Finalizing,
            "normal shutdown must not park a claimed publication"
        );

        allow_completion.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(2), shutdown)
            .await
            .expect("shutdown must return after finalization")
            .unwrap();
        assert_eq!(
            jobs::get(state.metadata_db.as_ref().as_ref().unwrap(), &job.id)
                .unwrap()
                .unwrap()
                .job
                .state,
            VideoUpscaleJobState::Completed
        );
    }

    #[test]
    fn rational_rate_is_exact() {
        assert_eq!(rational("24000/1001").unwrap(), (24000, 1001));
    }
    #[test]
    fn scale_is_derived_conservatively() {
        assert_eq!(scale_for("real-esrgan-x2plus:fp16"), 2);
        assert_eq!(scale_for("real-esrgan-x4plus:fp16"), 4);
    }

    #[test]
    fn first_use_and_stale_paths_trigger_upscaler_acquisition() {
        let model = "real-esrgan-x4plus:fp16";
        let mut config = mold_core::Config::default();
        assert!(crate::model_manager::upscaler_model_needs_pull(
            &config, model
        ));

        let temp = tempfile::tempdir().unwrap();
        config.models.insert(
            model.into(),
            ModelConfig {
                transformer: Some(
                    temp.path()
                        .join("missing.safetensors")
                        .display()
                        .to_string(),
                ),
                ..Default::default()
            },
        );
        assert!(crate::model_manager::upscaler_model_needs_pull(
            &config, model
        ));

        let weights = temp.path().join("installed.safetensors");
        fs::write(&weights, b"weights").unwrap();
        config.models.get_mut(model).unwrap().transformer = Some(weights.display().to_string());
        assert!(!crate::model_manager::upscaler_model_needs_pull(
            &config, model
        ));
    }

    #[test]
    fn enqueue_persists_intent_without_copying_or_probing_source_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let gallery = temp.path();
        let source = gallery.join("source.mp4");
        fs::write(&source, b"large-source-placeholder").unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let metadata = mold_db::metadata_io::synthesize_from_filename("source.mp4", 1);
        let record = mold_db::GenerationRecord::from_save(
            gallery,
            "source.mp4",
            OutputFormat::Mp4,
            metadata,
            mold_db::RecordSource::Server,
            1,
        );
        db.upsert(&record).unwrap();

        let job = enqueue_gallery_video_job(
            gallery,
            &db,
            "source.mp4",
            "real-esrgan-x4plus:fp16".into(),
            None,
        )
        .unwrap();
        let stored = jobs::get(&db, &job.id).unwrap().unwrap();
        assert_eq!(stored.job.state, VideoUpscaleJobState::Queued);
        assert_eq!(stored.job.total_frames, 0);
        assert!(stored.job.source_facts.is_none());
        assert!(stored.source_path.starts_with(&stored.work_dir));
        assert_ne!(stored.source_path, source.canonicalize().unwrap());
        assert_eq!(
            fs::read(&stored.source_path).unwrap(),
            fs::read(source).unwrap()
        );
        assert!(!stored.work_dir.join(SOURCE_DIGEST_FILE).exists());
        assert!(stored.work_dir.join(SOURCE_METADATA_FILE).is_file());
    }

    #[test]
    fn image_dimensions_are_rejected_before_full_decode() {
        assert!(validate_image_upscale_dimensions(1024, 1024, 4).is_ok());
        let error = validate_image_upscale_dimensions(32_000, 32_000, 4).unwrap_err();
        assert!(error.error.contains("pixels"));
    }

    #[test]
    fn probe_reports_exact_cfr_and_primary_audio_facts() {
        let _fixture_guard = FFMPEG_FIXTURE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !ffmpeg_fixture_tools_available() {
            eprintln!("skipping ffmpeg integration fixture: ffmpeg/ffprobe unavailable");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("fixture.mp4");
        let output = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-filter_threads",
                "1",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=5:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-threads",
                "1",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let facts = probe(&path).unwrap();
        assert_eq!(facts.frame_count, 5);
        assert_eq!(facts.fps, "5/1");
        assert_eq!(facts.primary_audio_codec.as_deref(), Some("aac"));
        assert_eq!(facts.primary_audio_sample_rate, Some(48_000));
        assert_eq!(facts.primary_audio_channels, Some(1));
        assert!(facts.duration_ms.abs_diff(1_000) <= 200);
    }

    #[test]
    fn probe_rejects_vfr_timestamps_even_when_container_is_supported() {
        let _fixture_guard = FFMPEG_FIXTURE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !ffmpeg_fixture_tools_available() {
            eprintln!("skipping ffmpeg integration fixture: ffmpeg/ffprobe unavailable");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vfr.mp4");
        let output = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-filter_threads",
                "1",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=16x16:rate=10:duration=1",
                "-vf",
                "setpts=if(lt(N\\,5)\\,N/(10*TB)\\,0.5+(N-5)/(5*TB))",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(probe(&path)
            .unwrap_err()
            .to_string()
            .contains("variable-frame-rate"));
    }

    #[test]
    fn publication_guard_rejects_fact_loss() {
        let source = VideoUpscaleMediaFacts {
            container: "mp4".into(),
            video_codec: "h264".into(),
            width: 16,
            height: 16,
            frame_count: 5,
            fps: "5/1".into(),
            duration_ms: 1000,
            primary_audio_codec: Some("aac".into()),
            primary_audio_sample_rate: Some(48_000),
            primary_audio_channels: Some(1),
        };
        let mut output = source.clone();
        output.primary_audio_codec = None;
        assert!(verify_preserved_facts(&source, &output)
            .unwrap_err()
            .to_string()
            .contains("primary audio"));
        output = source.clone();
        output.frame_count = 4;
        assert!(verify_preserved_facts(&source, &output)
            .unwrap_err()
            .to_string()
            .contains("frame count/FPS"));
    }

    #[test]
    fn publication_creates_a_new_provenance_bearing_gallery_row() {
        let temp = tempfile::tempdir().unwrap();
        let gallery = temp.path().join("gallery");
        let work = gallery.join(".mold-video-upscale-jobs/vup-test");
        fs::create_dir_all(&work).unwrap();
        let staged = work.join("final.mp4");
        fs::write(&staged, b"encoded-video-fixture").unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let mut metadata = mold_db::metadata_io::synthesize_from_filename("source.mp4", 1);
        metadata.upscale_model = Some("real-esrgan-x4plus:fp16".into());
        metadata.job_id = Some("vup-test".into());
        metadata.source_video_path = Some("source.mp4".into());
        crate::queue::publish_video_path_to_dir_named(
            &gallery,
            "source-framewise-upscaled-test.mp4",
            &staged,
            OutputFormat::Mp4,
            &metadata,
            None,
            Some(&db),
            None,
            &crate::batch_transaction::GalleryPublicationGate::default(),
        )
        .unwrap();
        assert_eq!(
            fs::read(gallery.join("source-framewise-upscaled-test.mp4")).unwrap(),
            b"encoded-video-fixture"
        );
        let record = db
            .get(&gallery, "source-framewise-upscaled-test.mp4")
            .unwrap()
            .unwrap();
        assert_eq!(record.metadata.job_id.as_deref(), Some("vup-test"));
        assert_eq!(
            record.metadata.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
        assert_eq!(
            record.metadata.source_video_path.as_deref(),
            Some("source.mp4")
        );
    }
}
