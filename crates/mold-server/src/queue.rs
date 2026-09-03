use std::collections::{BTreeSet, VecDeque};
use std::io::Write as _;
use std::sync::Arc;

use anyhow::Context as _;
use base64::Engine as _;
use mold_core::{
    ImageData, OutputFormat, OutputMetadata, SseCompleteEvent, SseErrorEvent, SseProgressEvent,
};
use mold_db::{MetadataDb, RecordSource};
use sha2::{Digest, Sha256};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::Notify;

use crate::durable_disposition::DurableDisposition;
use crate::durable_generation_settlement;
use crate::gpu_pool::GpuJob;
use crate::model_manager;
use crate::state::{
    ActiveGenerationSnapshot, AppState, GenerationJob, GenerationJobResult, SseCompletionPayload,
    SseMessage,
};

/// Convert an inference-crate progress event to an SSE wire event.
fn progress_to_sse(event: mold_inference::ProgressEvent) -> SseProgressEvent {
    event.into()
}

/// Forward an engine progress event to the job's channel. The channel is
/// the registry relay (`job_registry::progress_relay`), so the fold into the
/// `/api/queue/{id}/preview` snapshot happens there for every producer alike.
fn forward_generation_progress(
    tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    event: mold_inference::ProgressEvent,
) {
    if let Some(tx) = tx {
        let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
    }
}

/// Strips backtrace frames from candle error messages.
///
/// Renders the full anyhow cause chain (`{:#}`) so wrappers like
/// `with_context("mmap single-file checkpoint at …")` carry their root cause
/// through to the wire — otherwise users see the outer wrapper only.
pub(crate) fn clean_error_message(e: &anyhow::Error) -> String {
    let full = format!("{e:#}");
    let mut lines: Vec<&str> = Vec::new();
    for line in full.lines() {
        let trimmed = line.trim_start();
        if (trimmed.starts_with("0:") || trimmed.starts_with("1:"))
            && trimmed.len() > 3
            && trimmed
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit())
        {
            break;
        }
        if trimmed.len() > 2
            && trimmed.as_bytes()[0].is_ascii_digit()
            && trimmed.contains("::")
            && trimmed.contains("at ")
        {
            break;
        }
        lines.push(line);
    }
    let msg = lines.join("\n").trim().to_string();
    if msg.is_empty() {
        format!("{}", e.root_cause())
    } else {
        msg
    }
}

fn set_active_generation(state: &AppState, model: &str, prompt: &str) {
    let prompt_sha256 = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    let started_at_unix_ms = mold_core::time::now_epoch_ms_u64();

    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = Some(ActiveGenerationSnapshot {
        model: model.to_string(),
        prompt_sha256,
        started_at_unix_ms,
        started_at: Instant::now(),
    });
}

fn clear_active_generation(state: &AppState) {
    let mut active = state
        .active_generation
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *active = None;
}

/// Test-facing single-image wrapper around the shared output persistence path.
///
/// Errors writing to disk are logged and skipped. DB errors are also logged
/// but do not fail the save — the file is the source of truth.
///
/// Shared between the legacy single-GPU `process_job` (this file) and the
/// per-GPU worker (`gpu_worker.rs`). Keep these on one helper so the DB
/// upsert can never silently regress on one path while the other keeps
/// working.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn save_image_to_dir(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
    metadata: Option<&OutputMetadata>,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
) {
    let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();
    save_image_to_dir_with_suffix(
        dir,
        img,
        model,
        batch_size,
        None,
        metadata,
        generation_time_ms,
        db,
        events,
        &gallery_gate,
    );
}

/// Gallery filename for one saved output: the legacy
/// `mold-{model}-{ts}[-{idx}]` stem, then an optional `-{suffix}` (original /
/// upscaled), then the title slug as `~{slug}` so it is always the LAST stem
/// component (`mold_core::strip_title_slug` cuts at the final `~`), then the
/// extension. Untitled, unsuffixed prints are byte-identical to
/// `mold_core::default_output_filename`.
pub(crate) fn titled_output_filename(
    model: &str,
    timestamp_ms: u64,
    ext: &str,
    batch_size: u32,
    index: u32,
    suffix: Option<&str>,
    title_slug: Option<&str>,
) -> String {
    let legacy = mold_core::default_output_filename(model, timestamp_ms, ext, batch_size, index);
    let dot_ext = format!(".{ext}");
    let mut stem = legacy
        .strip_suffix(dot_ext.as_str())
        .unwrap_or(legacy.as_str())
        .to_string();
    if let Some(suffix) = suffix {
        stem = format!("{stem}-{suffix}");
    }
    match title_slug.filter(|slug| !slug.is_empty()) {
        Some(slug) => format!(
            "{stem}{}{slug}{dot_ext}",
            mold_core::print_title::TITLE_SLUG_SEPARATOR
        ),
        None => format!("{stem}{dot_ext}"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn save_image_to_dir_with_suffix(
    dir: &std::path::Path,
    img: &mold_core::ImageData,
    model: &str,
    batch_size: u32,
    suffix: Option<&str>,
    metadata: Option<&OutputMetadata>,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return None;
    }
    let timestamp_ms = mold_core::time::now_epoch_ms_u64();
    let ext = img.format.to_string();
    // A titled print carries a lossy `~slug` in its stem
    // (`mold-{model}-{ts}[-{idx}]~{slug}.{ext}`); untitled prints keep the
    // byte-identical legacy name. `synthesize_from_filename` strips the slug.
    let title_slug = metadata
        .and_then(|meta| meta.title.as_deref())
        .and_then(mold_core::title_slug);
    let filename = titled_output_filename(
        model,
        timestamp_ms,
        &ext,
        batch_size,
        img.index,
        suffix,
        title_slug.as_deref(),
    );
    let (filename, path, reservation) =
        match write_gallery_bytes_no_replace(dir, &filename, &img.data) {
            Ok(saved) => saved,
            Err(e) => {
                tracing::warn!("failed to save image to {}: {e}", dir.display());
                return None;
            }
        };
    tracing::info!("saved image to {}", path.display());
    let image_row = if let Some(meta) = metadata {
        let params = mold_db::persist::OutputRecordParams {
            format: img.format,
            metadata: meta,
            source: RecordSource::Server,
            generation_time_ms,
            backend: Some(mold_inference::compiled_backend_label()),
        };
        let record = mold_db::persist::build_saved_output_record(dir, &filename, &path, &params);
        let record = match crate::batch_transaction::archive_ordinary_gallery_record(
            dir,
            &path,
            record,
            gallery_gate,
            reservation.authority(),
        ) {
            Ok(record) => record,
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                let _ = crate::batch_transaction::sync_ordinary_gallery_directory(dir);
                drop(reservation);
                tracing::error!(
                    file = %path.display(),
                    %error,
                    "gallery archive failed; rolled back unpublished image"
                );
                return None;
            }
        };
        drop(reservation);
        let seeded = db.and_then(|db| upsert_and_report_filing(db, &record));
        Some(Box::new(gallery_image_with_filing(
            &record,
            seeded.as_ref(),
        )))
    } else {
        drop(reservation);
        None
    };
    // Emit even without a DB row — `image: None` tells clients to refetch
    // `/api/gallery` instead of inserting in place.
    if let Some(events) = events {
        let seeded_filing = image_row
            .as_ref()
            .is_some_and(|image| !image.tags.is_empty() || !image.collections.is_empty());
        let announced = image_row.clone();
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: image_row,
        });
        if seeded_filing {
            announce_seeded_filing(events, &filename, announced);
        }
    }
    Some(filename)
}

/// Upsert a publication's gallery row and report the creation-time filing
/// seeded onto it. Errors are logged and swallowed exactly as before — the
/// file on disk is the source of truth, and a finished render is never lost
/// over its filing.
fn upsert_and_report_filing(
    db: &MetadataDb,
    record: &mold_db::GenerationRecord,
) -> Option<mold_db::organization::SeededOrganization> {
    match db.upsert_reporting_organization(record) {
        Ok((_, seeded)) => Some(seeded),
        Err(error) => {
            tracing::warn!(
                "metadata DB upsert failed for {}: {error:#}",
                record.filename
            );
            None
        }
    }
}

/// Project a just-published record into its gallery row, folding in the
/// filing that was seeded with it.
///
/// `GenerationRecord::to_gallery_image` leaves `tags` / `collections` empty
/// because organization normally arrives through the server's post-overlay
/// enrichment. A print that was filed at creation already knows both, so the
/// `gallery_added` row carries them and a client can insert in place without
/// a refetch.
fn gallery_image_with_filing(
    record: &mold_db::GenerationRecord,
    seeded: Option<&mold_db::organization::SeededOrganization>,
) -> mold_core::GalleryImage {
    let mut image = record.to_gallery_image();
    if let Some(seeded) = seeded {
        image.tags = seeded.tags.clone();
        image.collections = seeded.collection_id.clone().into_iter().collect();
    }
    // Every `gallery_added` / `gallery_updated` row this module publishes is
    // built here, so this is where a mesh row picks up the poster renderer's
    // revision — the same value `/api/gallery` stamps at its listing exit. A
    // client keys its tile cache on `media_version`, and an event row that
    // disagreed with the listing would make it refetch the tile a moment
    // after inserting it.
    crate::thumbnails::stamp_poster_revision(&mut image);
    image
}

/// Announce a print that arrived already filed.
///
/// `gallery_updated` carries the organized row so a client that already
/// inserted the `gallery_added` row picks up its tags and membership.
/// `gallery_collections_changed` follows whenever the print joined a
/// collection — that event's contract is "created, renamed, re-covered,
/// deleted, **or had its membership changed**", and the shelf's item counts
/// move on a join whether or not the collection is new.
fn announce_seeded_filing(
    events: &crate::events::EventBroadcaster,
    filename: &str,
    image: Option<Box<mold_core::GalleryImage>>,
) {
    let joined_collection = image
        .as_ref()
        .map(|image| !image.collections.is_empty())
        .unwrap_or(false);
    events.publish(mold_core::ServerEvent::GalleryUpdated {
        filename: filename.to_string(),
        image,
    });
    if joined_collection {
        events.publish(mold_core::ServerEvent::GalleryCollectionsChanged {});
    }
}

/// Gallery filenames a generation's outputs were saved under, threaded into
/// the SSE complete event so mirroring clients keep the same identity.
#[derive(Debug, Default, Clone)]
pub(crate) struct SavedOutputNames {
    /// The payload the complete event carries (upscaled when upscaling ran).
    pub output: Option<String>,
    /// The pre-upscale original, when one was saved separately.
    pub original: Option<String>,
}

impl SavedOutputNames {
    /// The one projection of a completed child's terminal result.
    ///
    /// The facts come from the same response the SSE complete event carries,
    /// so an attached observer and a durable child describe one render.
    pub(crate) fn terminal_json(&self, response: &mold_core::GenerateResponse) -> String {
        serde_json::to_string(&mold_core::GenerationBatchResult {
            filename: self.output.clone(),
            original_filename: self.original.clone(),
            seed: Some(response.seed_used),
            generation_time_ms: Some(response.generation_time_ms),
            gpu: response.gpu,
        })
        .unwrap_or_default()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn save_generated_image_outputs(
    dir: &std::path::Path,
    original: Option<&ImageData>,
    output: &ImageData,
    model: &str,
    batch_size: u32,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> SavedOutputNames {
    let mut names = SavedOutputNames::default();
    if let Some(original) = original {
        let mut original_metadata = metadata.clone();
        apply_output_dimensions_to_metadata(&mut original_metadata, original);
        names.original = save_image_to_dir_with_suffix(
            dir,
            original,
            model,
            batch_size,
            Some("original"),
            Some(&original_metadata),
            generation_time_ms,
            db,
            events,
            gallery_gate,
        );
    }
    let mut output_metadata = metadata.clone();
    apply_output_dimensions_to_metadata(&mut output_metadata, output);
    names.output = save_image_to_dir_with_suffix(
        dir,
        output,
        model,
        batch_size,
        original.map(|_| "upscaled"),
        Some(&output_metadata),
        generation_time_ms,
        db,
        events,
        gallery_gate,
    );
    names
}

/// Save a video file to disk and (best-effort) record its metadata row.
/// Mirrors `save_image_to_dir` for the video-output path. See that helper
/// for the multi-path-callers note.
///
/// When `gif_preview` is non-empty, also persists
/// `$MOLD_HOME/cache/previews/<filename>.preview.gif`. The gallery preview
/// endpoint (`GET /api/gallery/preview/:filename`) streams from that path
/// so remote TUI clients can animate the detail pane without re-fetching
/// the full MP4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_video_to_dir(
    dir: &std::path::Path,
    bytes: &[u8],
    gif_preview: &[u8],
    format: OutputFormat,
    model: &str,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Option<String> {
    save_video_to_dir_with_sidecar(
        dir,
        bytes,
        gif_preview,
        format,
        model,
        metadata,
        generation_time_ms,
        db,
        events,
        gallery_gate,
        None,
    )
}

/// [`save_video_to_dir`] with a hook that runs once the final filename is
/// known and BEFORE `GalleryAdded` is published.
///
/// A sidecar tile nothing downstream can regenerate has to exist by the time
/// the event goes out: a client that fetches the tile on the event would
/// otherwise race the writer and cache whatever placeholder it got. Filename
/// allocation, the durable archive record, the rollback, and the event all
/// stay here rather than being duplicated per media kind.
#[allow(clippy::too_many_arguments)]
fn save_video_to_dir_with_sidecar(
    dir: &std::path::Path,
    bytes: &[u8],
    gif_preview: &[u8],
    format: OutputFormat,
    model: &str,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
    before_publish: Option<&dyn Fn(&str)>,
) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("failed to create output dir {}: {e}", dir.display());
        return None;
    }
    let ts = mold_core::time::now_epoch_ms_u64();
    let ext = format.extension();
    // A titled video carries the same lossy `~slug` an image does — a
    // sequence's stitched print reaches this path, and a title the user typed
    // must reach the filename on both media kinds or the two disagree.
    let desired = titled_output_filename(
        model,
        ts,
        ext,
        1,
        0,
        None,
        metadata
            .title
            .as_deref()
            .and_then(mold_core::title_slug)
            .as_deref(),
    );
    let (filename, path, reservation) = match write_gallery_bytes_no_replace(dir, &desired, bytes) {
        Ok(saved) => saved,
        Err(e) => {
            tracing::error!("failed to save video to {}: {e}", dir.display());
            return None;
        }
    };
    let params = mold_db::persist::OutputRecordParams {
        format,
        metadata,
        source: RecordSource::Server,
        generation_time_ms,
        backend: Some(mold_inference::compiled_backend_label()),
    };
    let record = mold_db::persist::build_saved_output_record(dir, &filename, &path, &params);
    let record = match crate::batch_transaction::archive_ordinary_gallery_record(
        dir,
        &path,
        record,
        gallery_gate,
        reservation.authority(),
    ) {
        Ok(record) => record,
        Err(error) => {
            let _ = std::fs::remove_file(&path);
            let _ = crate::batch_transaction::sync_ordinary_gallery_directory(dir);
            drop(reservation);
            tracing::error!(
                file = %path.display(),
                %error,
                "gallery archive failed; rolled back unpublished video"
            );
            return None;
        }
    };
    drop(reservation);
    if !gif_preview.is_empty() {
        save_video_preview_gif(&filename, gif_preview);
    }
    if let Some(before_publish) = before_publish {
        before_publish(&filename);
    }
    let seeded = db.and_then(|db| upsert_and_report_filing(db, &record));
    let image_row = Some(Box::new(gallery_image_with_filing(
        &record,
        seeded.as_ref(),
    )));
    if let Some(events) = events {
        let seeded_filing = seeded.map(|seeded| !seeded.is_empty()).unwrap_or(false);
        let announced = image_row.clone();
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.clone(),
            image: image_row,
        });
        if seeded_filing {
            announce_seeded_filing(events, &filename, announced);
        }
    }
    Some(filename)
}

/// Save an audio-only output plus its waveform thumbnail.
///
/// The bytes go through the same gallery publication path as video, then the
/// waveform PNG is written straight into the server thumbnail cache: nothing
/// downstream can decode a raster frame out of a WAV, so the tile has to be
/// persisted here or the gallery would only ever show a placeholder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_audio_to_dir(
    dir: &std::path::Path,
    bytes: &[u8],
    thumbnail_png: &[u8],
    format: OutputFormat,
    model: &str,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Option<String> {
    let filename = save_video_to_dir(
        dir,
        bytes,
        &[],
        format,
        model,
        metadata,
        generation_time_ms,
        db,
        events,
        gallery_gate,
    )?;
    if !thumbnail_png.is_empty() {
        save_audio_waveform_thumbnail(&filename, thumbnail_png);
    }
    Some(filename)
}

/// Publish a mesh into the gallery.
///
/// Shaped exactly like [`save_audio_to_dir`], and for the same reason: the
/// primary write is an ordinary single-file gallery publication, and the only
/// extra work is a sidecar tile that must exist at save time because nothing
/// downstream can render one on demand. A glTF buffer has no raster frame,
/// so without the poster the gallery would show a placeholder forever.
///
/// `save_video_to_dir_with_sidecar` is reused for the primary write rather
/// than duplicated: it already owns filename allocation, the durable archive
/// record, the rollback that deletes the file when the archive fails, and the
/// `GalleryAdded` event. None of that is raster-specific, and its hook is
/// what lets the poster land before the event rather than after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_mesh_to_dir(
    dir: &std::path::Path,
    bytes: &[u8],
    poster_png: &[u8],
    format: OutputFormat,
    model: &str,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> Option<String> {
    save_video_to_dir_with_sidecar(
        dir,
        bytes,
        // No GIF preview: there is no motion to preview, and the poster
        // sidecar is what every grid actually lays out.
        &[],
        format,
        model,
        metadata,
        generation_time_ms,
        db,
        events,
        gallery_gate,
        // The poster lands BEFORE `GalleryAdded`: a client fetching the tile
        // on the event used to race this write and could be answered the
        // placeholder for a print whose poster existed a millisecond later.
        Some(&|filename: &str| {
            if !poster_png.is_empty() {
                save_mesh_poster_thumbnail(filename, poster_png);
            }
        }),
    )
}

pub(crate) fn save_mesh_poster_thumbnail(filename: &str, png_bytes: &[u8]) {
    let thumb_dir = mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails");
    save_mesh_poster_thumbnail_to(&thumb_dir, filename, png_bytes);
}

/// Testable inner of [`save_mesh_poster_thumbnail`] with an explicit cache
/// directory, so unit tests don't race on `MOLD_HOME`.
fn save_mesh_poster_thumbnail_to(thumb_dir: &std::path::Path, filename: &str, png_bytes: &[u8]) {
    // One writer for both sidecar names, shared with the on-demand render in
    // `crate::thumbnails`, so save time and fetch time cannot disagree about
    // where a poster lives or write a tile a reader can catch half-finished.
    if let Err(error) =
        crate::thumbnails::write_mesh_poster_sidecars(thumb_dir, filename, png_bytes)
    {
        tracing::warn!(
            file = %filename,
            error = %format!("{error:#}"),
            "failed to write the mesh poster thumbnail"
        );
    }
}

pub(crate) fn save_audio_waveform_thumbnail(filename: &str, png_bytes: &[u8]) {
    let thumb_dir = mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("thumbnails");
    save_audio_waveform_thumbnail_to(&thumb_dir, filename, png_bytes);
}

/// Testable inner of [`save_audio_waveform_thumbnail`] with an explicit cache
/// directory, so unit tests don't race on `MOLD_HOME`.
fn save_audio_waveform_thumbnail_to(thumb_dir: &std::path::Path, filename: &str, png_bytes: &[u8]) {
    if let Err(e) = std::fs::create_dir_all(thumb_dir) {
        tracing::warn!(
            "failed to create thumbnail cache dir {}: {e}",
            thumb_dir.display()
        );
        return;
    }
    for thumb_path in mold_core::media_paths::audio_waveform_thumbnail_paths(thumb_dir, filename) {
        if let Err(e) = std::fs::write(&thumb_path, png_bytes) {
            tracing::warn!(
                "failed to write waveform thumbnail {}: {e}",
                thumb_path.display()
            );
        }
    }
}

/// Idempotently publish a video under one caller-owned gallery filename.
///
/// Durable chain finalization uses an attempt-derived filename so replaying a
/// crash window upserts the same gallery row instead of allocating another
/// timestamped output. An existing file is accepted only when its bytes match.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_video_to_dir_named(
    dir: &std::path::Path,
    filename: &str,
    bytes: &[u8],
    format: OutputFormat,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> anyhow::Result<String> {
    let filename_path = std::path::Path::new(filename);
    if filename_path.components().count() != 1
        || !matches!(
            filename_path.components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        anyhow::bail!("gallery filename must be one normal path component");
    }
    std::fs::create_dir_all(dir)?;
    let authority = crate::batch_transaction::acquire_gallery_bookkeeping_lock(dir)?;
    let path = dir.join(filename);
    let created = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(error.into());
            }
            crate::batch_transaction::sync_ordinary_gallery_directory(dir)?;
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(&path)?;
            if existing != bytes {
                anyhow::bail!(
                    "gallery replay target '{}' exists with different bytes",
                    path.display()
                );
            }
            false
        }
        Err(error) => return Err(error.into()),
    };
    let params = mold_db::persist::OutputRecordParams {
        format,
        metadata,
        source: RecordSource::Server,
        generation_time_ms,
        backend: Some(mold_inference::compiled_backend_label()),
    };
    let index = gallery_gate.committed_archive_index_while_locked(dir, &authority)?;
    let record = if let Some(existing) = index.get(filename) {
        anyhow::ensure!(
            existing.record().format == format
                && existing.record().metadata == *metadata
                && !existing.record().metadata_synthetic,
            "gallery replay target '{}' exists with different archived metadata",
            path.display()
        );
        existing.record().clone()
    } else {
        let record = mold_db::persist::build_saved_output_record(dir, filename, &path, &params);
        match crate::batch_transaction::archive_ordinary_gallery_record(
            dir,
            &path,
            record,
            gallery_gate,
            &authority,
        ) {
            Ok(record) => record,
            Err(error) => {
                if created {
                    let _ = std::fs::remove_file(&path);
                    let _ = crate::batch_transaction::sync_ordinary_gallery_directory(dir);
                }
                return Err(error).context("archiving durable chain gallery publication");
            }
        }
    };
    drop(authority);
    // Unlike the ordinary paths, a durable chain's DB failure is fatal: the
    // caller settles the job on this returning, so a silently missing row
    // would let a replay re-publish the same take.
    let seeded = match db {
        Some(db) => Some(
            db.upsert_reporting_organization(&record)
                .context("recording durable chain gallery metadata")?
                .1,
        ),
        None => None,
    };
    if let Some(events) = events {
        let seeded_filing = seeded.as_ref().map(|s| !s.is_empty()).unwrap_or(false);
        let image_row = Some(Box::new(gallery_image_with_filing(
            &record,
            seeded.as_ref(),
        )));
        let announced = image_row.clone();
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.to_string(),
            image: image_row,
        });
        if seeded_filing {
            announce_seeded_filing(events, filename, announced);
        }
    }
    Ok(filename.to_string())
}

/// Publish an already-encoded video without materializing the whole file in
/// memory. The staged file must share the gallery filesystem; framewise video
/// jobs deliberately stage beneath the output directory to guarantee that.
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_video_path_to_dir_named(
    dir: &std::path::Path,
    filename: &str,
    staged: &std::path::Path,
    format: OutputFormat,
    metadata: &OutputMetadata,
    generation_time_ms: Option<i64>,
    db: Option<&MetadataDb>,
    events: Option<&crate::events::EventBroadcaster>,
    gallery_gate: &crate::batch_transaction::GalleryPublicationGate,
) -> anyhow::Result<String> {
    let filename_path = std::path::Path::new(filename);
    anyhow::ensure!(
        filename_path.components().count() == 1
            && matches!(
                filename_path.components().next(),
                Some(std::path::Component::Normal(_))
            ),
        "gallery filename must be one normal path component"
    );
    std::fs::create_dir_all(dir)?;
    let authority = crate::batch_transaction::acquire_gallery_bookkeeping_lock(dir)?;
    let path = dir.join(filename);
    let created = match std::fs::hard_link(staged, &path) {
        Ok(()) => {
            crate::batch_transaction::sync_ordinary_gallery_directory(dir)?;
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fn digest(path: &std::path::Path) -> anyhow::Result<[u8; 32]> {
                use std::io::Read as _;
                let mut file = std::fs::File::open(path)?;
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 64 * 1024];
                loop {
                    let read = file.read(&mut buffer)?;
                    if read == 0 {
                        break;
                    }
                    hash.update(&buffer[..read]);
                }
                Ok(hash.finalize().into())
            }
            anyhow::ensure!(
                digest(&path)? == digest(staged)?,
                "gallery replay target '{}' exists with different bytes",
                path.display()
            );
            false
        }
        Err(error) => return Err(error).context("linking staged video into gallery"),
    };
    let params = mold_db::persist::OutputRecordParams {
        format,
        metadata,
        source: RecordSource::Server,
        generation_time_ms,
        backend: Some(mold_inference::compiled_backend_label()),
    };
    let index = gallery_gate.committed_archive_index_while_locked(dir, &authority)?;
    let record = if let Some(existing) = index.get(filename) {
        anyhow::ensure!(
            existing.record().format == format
                && existing.record().metadata == *metadata
                && !existing.record().metadata_synthetic,
            "gallery replay target '{}' exists with different archived metadata",
            path.display()
        );
        existing.record().clone()
    } else {
        let record = mold_db::persist::build_saved_output_record(dir, filename, &path, &params);
        match crate::batch_transaction::archive_ordinary_gallery_record(
            dir,
            &path,
            record,
            gallery_gate,
            &authority,
        ) {
            Ok(record) => record,
            Err(error) => {
                if created {
                    let _ = std::fs::remove_file(&path);
                    let _ = crate::batch_transaction::sync_ordinary_gallery_directory(dir);
                }
                return Err(error).context("archiving framewise upscale publication");
            }
        }
    };
    drop(authority);
    let seeded = db
        .map(|db| {
            db.upsert_reporting_organization(&record)
                .context("recording framewise upscale gallery metadata")
        })
        .transpose()?
        .map(|(_, seeded)| seeded);
    if let Some(events) = events {
        let seeded_filing = seeded.as_ref().is_some_and(|seeded| !seeded.is_empty());
        let image_row = Some(Box::new(gallery_image_with_filing(
            &record,
            seeded.as_ref(),
        )));
        let announced = image_row.clone();
        events.publish(mold_core::ServerEvent::GalleryAdded {
            filename: filename.to_string(),
            image: image_row,
        });
        if seeded_filing {
            announce_seeded_filing(events, filename, announced);
        }
    }
    Ok(filename.to_string())
}

fn write_gallery_bytes_no_replace(
    dir: &std::path::Path,
    desired: &str,
    bytes: &[u8],
) -> anyhow::Result<(
    String,
    std::path::PathBuf,
    crate::batch_transaction::GalleryNameReservation,
)> {
    write_gallery_bytes_no_replace_with_directory_sync(
        dir,
        desired,
        bytes,
        &crate::batch_transaction::sync_ordinary_gallery_directory,
    )
}

fn write_gallery_bytes_no_replace_with_directory_sync(
    dir: &std::path::Path,
    desired: &str,
    bytes: &[u8],
    sync_directory: &dyn Fn(&std::path::Path) -> anyhow::Result<()>,
) -> anyhow::Result<(
    String,
    std::path::PathBuf,
    crate::batch_transaction::GalleryNameReservation,
)> {
    let reservation = crate::batch_transaction::reserve_gallery_final_name_with_directory_sync(
        dir,
        desired,
        sync_directory,
    )?;
    let filename = reservation.final_name().to_owned();
    let path = dir.join(&filename);
    // Stage under `<final>.partial` and publish by rename. Writing the final
    // name in place means a kill mid-write leaves a truncated file at a real
    // gallery name, which the next boot's `db.reconcile` imports as a valid
    // print — and the shutdown path deliberately bounds itself and exits, so
    // that kill is a routine event rather than a hypothetical one.
    let staged = dir.join(format!("{filename}{GALLERY_PARTIAL_SUFFIX}"));
    // Any leftover from an earlier interrupted write at this exact name: the
    // reservation says the name is ours, so a stale sibling is ours to drop.
    let _ = std::fs::remove_file(&staged);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(&staged);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = publish_staged_no_replace(&staged, &path) {
        let _ = std::fs::remove_file(&staged);
        return Err(error.into());
    }
    sync_directory(dir)?;
    Ok((filename, path, reservation))
}

/// Move staged bytes onto their final gallery name, atomically, without ever
/// replacing a file somebody else put there.
///
/// Plain `rename` replaces its destination on Unix, which would silently
/// overwrite a name another writer took between reservation and publication.
/// `hard_link` refuses correctly but demands a filesystem with links, and
/// exFAT and some SMB mounts do not have them — requiring links made every
/// save on such a gallery fail, discard its bytes, and hold the durable job
/// forever. So: use the platform's real no-replace rename where there is one,
/// and fall back to reserving the name where there is not.
fn publish_staged_no_replace(
    staged: &std::path::Path,
    final_path: &std::path::Path,
) -> std::io::Result<()> {
    #[cfg(test)]
    if FORCE_PUBLISH_FALLBACK.load(std::sync::atomic::Ordering::SeqCst) {
        return publish_by_reserving_final_name(staged, final_path);
    }
    match platform_rename_no_replace(staged, final_path) {
        Some(Ok(())) => Ok(()),
        // The destination exists — the contract firing, not a platform gap.
        Some(Err(error)) if error.kind() == std::io::ErrorKind::AlreadyExists => Err(error),
        Some(Err(error)) if !is_unsupported_operation(&error) => Err(error),
        // No primitive, or this filesystem does not implement it.
        _ => publish_by_reserving_final_name(staged, final_path),
    }
}

/// Test seam: behave as a filesystem with no no-replace rename, which is what
/// exFAT and some SMB mounts are. Setting it is harmless to any test running
/// concurrently — the fallback satisfies exactly the same contract, so every
/// other gallery assertion holds under either path.
#[cfg(test)]
static FORCE_PUBLISH_FALLBACK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `Some` when the platform has a no-replace rename; `None` when it has none
/// to try.
fn platform_rename_no_replace(
    staged: &std::path::Path,
    final_path: &std::path::Path,
) -> Option<std::io::Result<()>> {
    #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
    {
        use std::os::unix::ffi::OsStrExt;
        let (Ok(from), Ok(to)) = (
            std::ffi::CString::new(staged.as_os_str().as_bytes()),
            std::ffi::CString::new(final_path.as_os_str().as_bytes()),
        ) else {
            return None;
        };
        // SAFETY: both paths are NUL-terminated C strings that outlive the call.
        let result = unsafe {
            #[cfg(target_os = "macos")]
            {
                libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL)
            }
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            {
                libc::renameat2(
                    libc::AT_FDCWD,
                    from.as_ptr(),
                    libc::AT_FDCWD,
                    to.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            }
        };
        if result == 0 {
            Some(Ok(()))
        } else {
            Some(Err(std::io::Error::last_os_error()))
        }
    }
    #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
    {
        let _ = (staged, final_path);
        None
    }
}

/// Whether the error means "this platform or filesystem cannot do that",
/// rather than a genuine failure to publish.
fn is_unsupported_operation(error: &std::io::Error) -> bool {
    // Compared rather than pattern-matched: Linux defines ENOTSUP and
    // EOPNOTSUPP as the same value, so listing both as patterns makes the
    // second arm unreachable and `-D warnings` rejects it. macOS keeps them
    // distinct, which is why the pattern form compiled locally and failed only
    // on Linux CI.
    let Some(code) = error.raw_os_error() else {
        return false;
    };
    code == libc::ENOSYS
        || code == libc::ENOTSUP
        || code == libc::EINVAL
        || code == libc::EOPNOTSUPP
}

/// The universal fallback: reserve the final name with an atomic `create_new`,
/// then rename our own staged bytes over our own reservation.
///
/// `create_new` is the no-replace guarantee — it fails with `AlreadyExists` if
/// anybody else holds the name — and the rename that follows only ever
/// replaces the empty placeholder this call just made. A crash in between
/// leaves a zero-byte file, which the gallery's existing size floor already
/// filters out, so it is never mistaken for a print.
fn publish_by_reserving_final_name(
    staged: &std::path::Path,
    final_path: &std::path::Path,
) -> std::io::Result<()> {
    let placeholder = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(final_path)?;
    drop(placeholder);
    if let Err(error) = std::fs::rename(staged, final_path) {
        let _ = std::fs::remove_file(final_path);
        return Err(error);
    }
    Ok(())
}

/// Suffix of a gallery write that has been staged but not yet published.
const GALLERY_PARTIAL_SUFFIX: &str = ".partial";

/// Remove staged gallery writes no process is still making.
///
/// A kill between staging and publication leaves `<final>.partial` behind, and
/// the bounded shutdown deadline turns that from a rare event into a routine
/// one. Nothing else reclaims them and later generations take fresh
/// timestamped names, so without this repeated interruptions consume gallery
/// disk permanently.
///
/// Safe against a concurrent writer — including one in another server sharing
/// this gallery — because it holds the gallery bookkeeping lock, which every
/// writer already holds for its whole reserve-write-publish window. A partial
/// visible while we hold that lock cannot be in flight.
pub(crate) fn sweep_stale_gallery_partials(dir: &std::path::Path) -> usize {
    let _bookkeeping = match crate::batch_transaction::acquire_gallery_bookkeeping_lock(dir) {
        Ok(guard) => guard,
        Err(error) => {
            tracing::warn!(
                dir = %dir.display(),
                error = %format!("{error:#}"),
                "skipping stale gallery partial sweep: bookkeeping lock unavailable"
            );
            return 0;
        }
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !entry
            .file_name()
            .to_string_lossy()
            .ends_with(GALLERY_PARTIAL_SUFFIX)
        {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                tracing::info!(
                    file = %path.display(),
                    "removed a gallery write interrupted before publication"
                );
                removed += 1;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                file = %path.display(),
                %error,
                "could not remove an interrupted gallery write"
            ),
        }
    }
    removed
}

fn requested_post_upscale_model(req: &mold_core::GenerateRequest) -> Option<&str> {
    req.upscale_model
        .as_deref()
        .map(str::trim)
        .filter(|m| !m.is_empty())
}

fn post_upscale_model_to_pull(
    config: &mold_core::Config,
    req: &mold_core::GenerateRequest,
) -> Result<Option<String>, String> {
    let Some(requested) = requested_post_upscale_model(req) else {
        return Ok(None);
    };
    let model_name = mold_core::manifest::resolve_model_name(requested);
    if !model_manager::upscaler_model_needs_pull(config, &model_name) {
        return Ok(None);
    }
    if mold_core::manifest::find_manifest(&model_name).is_none() {
        return Err(format!("unknown upscaler model '{model_name}'"));
    }
    Ok(Some(model_name))
}

pub(crate) async fn ensure_post_upscale_model_downloaded(
    state: &AppState,
    req: &mold_core::GenerateRequest,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<(), String> {
    let model_to_pull = {
        let config = state.config.read().await;
        post_upscale_model_to_pull(&config, req)?
    };
    let Some(model_name) = model_to_pull else {
        return Ok(());
    };

    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Downloading upscaler {model_name}"),
        }));
    }
    let progress = progress_tx.cloned().map(|tx| {
        Arc::new(move |event: mold_core::download::DownloadProgressEvent| {
            let event = match event {
                mold_core::download::DownloadProgressEvent::Status { message } => {
                    SseProgressEvent::Info { message }
                }
                mold_core::download::DownloadProgressEvent::FileStart {
                    filename,
                    file_index,
                    total_files,
                    size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadProgress {
                    filename,
                    file_index,
                    total_files,
                    bytes_downloaded: 0,
                    bytes_total: size_bytes,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
                mold_core::download::DownloadProgressEvent::FileProgress {
                    filename,
                    file_index,
                    bytes_downloaded,
                    bytes_total,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadProgress {
                    filename,
                    file_index,
                    total_files: 0,
                    bytes_downloaded,
                    bytes_total,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
                mold_core::download::DownloadProgressEvent::FileDone {
                    filename,
                    file_index,
                    total_files,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                } => SseProgressEvent::DownloadDone {
                    filename,
                    file_index,
                    total_files,
                    batch_bytes_downloaded,
                    batch_bytes_total,
                    batch_elapsed_ms,
                },
            };
            let _ = tx.send(SseMessage::Progress(event));
        }) as model_manager::DownloadProgressCallback
    });
    model_manager::pull_model(state, &model_name, progress)
        .await
        .map_err(|e| format!("failed to pull upscaler model: {}", e.error))?;
    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::PullComplete {
            model: model_name,
        }));
    }
    Ok(())
}

async fn ensure_legacy_post_upscale_model_downloaded(
    state: &AppState,
    req: &mold_core::GenerateRequest,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
    #[cfg(test)] hook: Option<&LegacyUpscalePreparationHook>,
) -> Result<(), String> {
    #[cfg(test)]
    if let Some(hook) = hook {
        hook.started.notify_one();
        hook.resume.notified().await;
    }
    ensure_post_upscale_model_downloaded(state, req, progress_tx).await
}

pub(crate) fn apply_output_dimensions_to_metadata(metadata: &mut OutputMetadata, img: &ImageData) {
    metadata.apply_output_dimensions(img.width, img.height);
}

pub(crate) fn apply_upscale_response_to_image_generation(
    req: &mold_core::GenerateRequest,
    response: &mut mold_core::GenerateResponse,
    original: ImageData,
    upscaled: mold_core::UpscaleResponse,
) -> anyhow::Result<ImageData> {
    if response.video.is_some() || requested_post_upscale_model(req).is_none() {
        return Ok(original);
    }
    if upscaled.image.data.is_empty() {
        anyhow::bail!("upscaler returned an empty image");
    }
    response.generation_time_ms = response
        .generation_time_ms
        .saturating_add(upscaled.upscale_time_ms);
    Ok(ImageData {
        index: original.index,
        ..upscaled.image
    })
}

pub(crate) fn settle_post_generation_upscale(
    original: ImageData,
    result: Result<ImageData, String>,
) -> (ImageData, Option<ImageData>, Option<String>) {
    match result {
        Ok(upscaled) => (upscaled, Some(original), None),
        Err(error) => (original, None, Some(error)),
    }
}

async fn upscale_generated_image_on_single_worker(
    state: &AppState,
    req: &mold_core::GenerateRequest,
    seed_used: u64,
    img: ImageData,
    progress_tx: Option<&tokio::sync::mpsc::UnboundedSender<SseMessage>>,
) -> Result<ImageData, String> {
    let Some(upscale_model) = requested_post_upscale_model(req).map(str::to_string) else {
        return Ok(img);
    };
    let model_name = mold_core::manifest::resolve_model_name(&upscale_model);
    if let Some(tx) = progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::StageStart {
            name: format!("Loading upscaler {model_name}"),
        }));
    }

    let needs_pull = {
        let config = state.config.read().await;
        config
            .models
            .get(&model_name)
            .and_then(|c| c.transformer.as_ref())
            .is_none()
    };
    if needs_pull {
        if mold_core::manifest::find_manifest(&model_name).is_none() {
            return Err(format!("unknown upscaler model '{model_name}'"));
        }
        model_manager::pull_model(state, &model_name, None)
            .await
            .map_err(|e| format!("failed to pull upscaler model: {}", e.error))?;
    }

    let (weights_path, artifact_root) = {
        let config = state.config.read().await;
        (
            config
                .models
                .get(&model_name)
                .and_then(|c| c.transformer.as_ref())
                .map(std::path::PathBuf::from),
            config.resolved_models_dir(),
        )
    };
    let weights_path = weights_path
        .ok_or_else(|| format!("upscaler model '{model_name}' not configured after pull"))?;

    let upscale_req = mold_core::UpscaleRequest {
        model: model_name.clone(),
        image: img.data.clone(),
        output_format: img.format,
        tile_size: None,
        metadata: Some(OutputMetadata::from_generate_request(
            req,
            seed_used,
            None,
            mold_core::build_info::version_string(),
        )),
    };
    let upscaler_cache = state.upscaler_cache.clone();
    let progress_tx_for_blocking = progress_tx.cloned();
    let upscaled =
        tokio::task::spawn_blocking(move || -> anyhow::Result<mold_core::UpscaleResponse> {
            let mut cache = upscaler_cache.lock().unwrap_or_else(|e| e.into_inner());
            let needs_new = cache.as_ref().is_none_or(|e| e.model_name() != model_name);
            if needs_new {
                let new_engine = mold_inference::create_upscale_engine(
                    model_name.clone(),
                    weights_path,
                    Some(&artifact_root),
                    mold_inference::LoadStrategy::Eager,
                    0,
                )?;
                if let Some(mut old_engine) = cache.take() {
                    old_engine.unload();
                }
                *cache = Some(new_engine);
            }
            let engine = cache.as_mut().unwrap();
            if let Some(tx) = progress_tx_for_blocking {
                engine.set_on_progress(Box::new(move |event| {
                    let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
                }));
            }
            let result = engine.upscale(&upscale_req);
            engine.clear_on_progress();
            result
        })
        .await
        .map_err(|e| format!("upscale task failed: {e}"))?
        .map_err(|e| format!("upscale failed: {e}"))?;

    let mut response = mold_core::GenerateResponse {
        request_warnings: Vec::new(),
        audio: None,
        mesh: None,
        images: vec![],
        video: None,
        generation_time_ms: 0,
        model: req.model.clone(),
        seed_used: req.seed.unwrap_or(0),
        gpu: None,
    };
    apply_upscale_response_to_image_generation(req, &mut response, img, upscaled)
        .map_err(|e| format!("upscale failed: {e}"))
}

/// Persist a video's `.preview.gif` sidecar to the server's preview cache
/// (`$MOLD_HOME/cache/previews/<filename>.preview.gif`). Best-effort —
/// warnings log and return so a failure here never fails the save path.
///
/// Shared with the multi-GPU worker path (`gpu_worker::process_job`) so
/// video outputs land a preview regardless of which save flow wrote the
/// MP4; otherwise `/api/gallery/preview/:filename` would 404 whenever the
/// server is running with GPU workers enabled.
pub(crate) fn save_video_preview_gif(filename: &str, gif_bytes: &[u8]) {
    let preview_dir = mold_core::Config::mold_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".mold"))
        .join("cache")
        .join("previews");
    save_video_preview_gif_to(&preview_dir, filename, gif_bytes);
}

/// Testable inner of [`save_video_preview_gif`] that accepts an explicit
/// preview directory (lets unit tests exercise the write path without
/// racing on the `MOLD_HOME` env var).
fn save_video_preview_gif_to(preview_dir: &std::path::Path, filename: &str, gif_bytes: &[u8]) {
    if let Err(e) = std::fs::create_dir_all(preview_dir) {
        tracing::warn!(
            "failed to create preview cache dir {}: {e}",
            preview_dir.display()
        );
        return;
    }
    let preview_path = preview_dir.join(mold_core::media_paths::preview_gif_filename(filename));
    if let Err(e) = std::fs::write(&preview_path, gif_bytes) {
        tracing::warn!(
            "failed to write preview gif {}: {e}",
            preview_path.display()
        );
    }
}

/// Build the SSE `complete` wire event from a finished generation response.
///
/// Video responses encode the actual video bytes (MP4/GIF/APNG/WebP) as the
/// payload and populate every `video_*` metadata field; image responses
/// encode the image bytes with the video fields cleared. `img` is the
/// `ImageData` chosen by the caller — either the first generated image or an
/// `ImageData` synthesized from the video thumbnail (the single-primary-image
/// shape that the internal `GenerationJobResult` still expects).
///
/// Shared between the single-GPU path (`process_job` in this file) and the
/// multi-GPU path (`gpu_worker::process_job`) so the two can never drift on
/// which `video_*` fields are populated. Before this helper existed the
/// multi-GPU worker always encoded the thumbnail PNG as the payload and
/// hard-coded every `video_*` field to `None`, which silently degraded every
/// LTX-Video / LTX-2 generation into an image response on hosts with at
/// least one GPU worker.
pub(crate) fn build_sse_complete_event(
    response: &mold_core::GenerateResponse,
    img: &mold_core::ImageData,
    original: Option<&mold_core::ImageData>,
    metadata: Option<&OutputMetadata>,
    saved: &SavedOutputNames,
    payload: SseCompletionPayload,
) -> SseCompleteEvent {
    let b64 = base64::engine::general_purpose::STANDARD;
    let include_media = payload == SseCompletionPayload::Full;
    // Mirror exactly what the save path records: video metadata is used
    // as-built, image metadata gets the payload's actual dimensions.
    let event_metadata = metadata.map(|meta| {
        let mut meta = meta.clone();
        // Only a raster print's metadata records the payload's own
        // dimensions. A clip already carries them, and neither audio nor a
        // mesh has any — `img` there is the sidecar tile, not the artifact.
        if response.video.is_none() && response.audio.is_none() && response.mesh.is_none() {
            apply_output_dimensions_to_metadata(&mut meta, img);
        }
        Box::new(meta)
    });
    if let Some(ref mesh) = response.mesh {
        return SseCompleteEvent {
            request_warnings: response.request_warnings.clone(),
            image: if include_media {
                b64.encode(&mesh.data)
            } else {
                String::new()
            },
            format: mesh.format,
            // As with audio, the raster dimensions a client lays the tile out
            // with are the POSTER's. A mesh has none of its own.
            width: mesh.poster_width,
            height: mesh.poster_height,
            original_image: None,
            original_width: None,
            original_height: None,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
            mesh_vertices: Some(mesh.vertex_count),
            mesh_faces: Some(mesh.face_count),
            mesh_textured: mesh.textured,
            mesh_poster: include_media.then(|| b64.encode(&mesh.poster)),
            mesh_bounds_min: Some(mesh.bounds_min),
            mesh_bounds_max: Some(mesh.bounds_max),
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: None,
            metadata: event_metadata,
        };
    }
    if let Some(ref audio) = response.audio {
        return SseCompleteEvent {
            request_warnings: response.request_warnings.clone(),
            image: if include_media {
                b64.encode(&audio.data)
            } else {
                String::new()
            },
            format: audio.format,
            // The raster dimensions clients lay the tile out with are the
            // waveform thumbnail's, not the audio's — audio has none.
            width: img.width,
            height: img.height,
            original_image: None,
            original_width: None,
            original_height: None,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
            audio_sample_rate: Some(audio.sample_rate),
            audio_channels: Some(audio.channels),
            audio_duration_ms: Some(audio.duration_ms),
            audio_thumbnail: include_media.then(|| b64.encode(&audio.thumbnail)),
            mesh_vertices: None,
            mesh_faces: None,
            mesh_textured: false,
            mesh_poster: None,
            mesh_bounds_min: None,
            mesh_bounds_max: None,
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: None,
            metadata: event_metadata,
        };
    }
    if let Some(ref video) = response.video {
        SseCompleteEvent {
            request_warnings: response.request_warnings.clone(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
            image: if include_media {
                b64.encode(&video.data)
            } else {
                String::new()
            },
            format: video.format,
            width: video.width,
            height: video.height,
            original_image: None,
            original_width: None,
            original_height: None,
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: Some(video.frames),
            video_fps: Some(video.fps),
            video_thumbnail: include_media.then(|| b64.encode(&video.thumbnail)),
            video_gif_preview: if !include_media || video.gif_preview.is_empty() {
                None
            } else {
                Some(b64.encode(&video.gif_preview))
            },
            video_has_audio: video.has_audio,
            video_duration_ms: video.duration_ms,
            video_audio_sample_rate: video.audio_sample_rate,
            video_audio_channels: video.audio_channels,
            mesh_vertices: None,
            mesh_faces: None,
            mesh_textured: false,
            mesh_poster: None,
            mesh_bounds_min: None,
            mesh_bounds_max: None,
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: None,
            metadata: event_metadata,
        }
    } else {
        SseCompleteEvent {
            request_warnings: response.request_warnings.clone(),
            audio_sample_rate: None,
            audio_channels: None,
            audio_duration_ms: None,
            audio_thumbnail: None,
            image: if include_media {
                b64.encode(&img.data)
            } else {
                String::new()
            },
            format: img.format,
            width: img.width,
            height: img.height,
            original_image: include_media
                .then(|| original.map(|image| b64.encode(&image.data)))
                .flatten(),
            original_width: original.map(|image| image.width),
            original_height: original.map(|image| image.height),
            seed_used: response.seed_used,
            generation_time_ms: response.generation_time_ms,
            model: response.model.clone(),
            video_frames: None,
            video_fps: None,
            video_thumbnail: None,
            video_gif_preview: None,
            video_has_audio: false,
            video_duration_ms: None,
            video_audio_sample_rate: None,
            video_audio_channels: None,
            mesh_vertices: None,
            mesh_faces: None,
            mesh_textured: false,
            mesh_poster: None,
            mesh_bounds_min: None,
            mesh_bounds_max: None,
            gpu: response.gpu,
            filename: saved.output.clone(),
            original_filename: saved.original.clone(),
            metadata: event_metadata,
        }
    }
}

pub(crate) fn build_sse_completion_message(
    response: &mold_core::GenerateResponse,
    img: &mold_core::ImageData,
    original: Option<&mold_core::ImageData>,
    metadata: Option<&OutputMetadata>,
    saved: &SavedOutputNames,
    payload: SseCompletionPayload,
) -> SseMessage {
    if payload == SseCompletionPayload::MetadataOnly && saved.output.is_none() {
        return SseMessage::Error(SseErrorEvent::failed(
            "generation completed but the output could not be saved for streaming".to_string(),
        ));
    }
    SseMessage::Complete(Box::new(build_sse_complete_event(
        response, img, original, metadata, saved, payload,
    )))
}

/// Dispatch gate shared through `AppState`, toggled by `POST /api/queue/pause`
/// and `POST /api/queue/resume`. When paused the dispatch loops stop pulling
/// *new* jobs off the channel; the job already running on a worker finishes
/// untouched. Cheap to poll (a single relaxed-ish atomic) so it can sit at the
/// top of every loop iteration.
pub struct QueuePause {
    paused: AtomicBool,
    /// Wakes every gated dispatch loop on resume. Resume calls
    /// `notify_waiters()` so *all* loops (single- and multi-GPU) proceed, not
    /// just one.
    notify: Notify,
    #[cfg(test)]
    waiters: AtomicUsize,
    #[cfg(test)]
    waiter_notify: Notify,
}

impl QueuePause {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            paused: AtomicBool::new(false),
            notify: Notify::new(),
            #[cfg(test)]
            waiters: AtomicUsize::new(0),
            #[cfg(test)]
            waiter_notify: Notify::new(),
        })
    }

    /// Pause new-job dispatch. Returns `true` iff this call flipped the state
    /// (was running); idempotent repeat pauses return `false` so the route can
    /// suppress a duplicate `queue_paused` event.
    pub fn pause(&self) -> bool {
        !self.paused.swap(true, Ordering::SeqCst)
    }

    /// Resume dispatch and wake every gated loop. Returns `true` iff this call
    /// flipped the state (was paused).
    pub fn resume(&self) -> bool {
        let was_paused = self.paused.swap(false, Ordering::SeqCst);
        if was_paused {
            self.notify.notify_waiters();
        }
        was_paused
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Blocking-context variant of [`Self::wait_if_paused`] for the chain
    /// runner's stage loop (which runs under `spawn_blocking`). Polls at a
    /// coarse interval — it only ever runs between stages while the queue
    /// is paused — and returns early when `should_abort` reports true so a
    /// cancel lands while the queue is held.
    pub fn wait_if_paused_blocking(&self, should_abort: &dyn Fn() -> bool) {
        while self.is_paused() && !should_abort() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Park the caller while paused, returning as soon as dispatch is resumed
    /// (immediately when not paused). Registers the wakeup *before* the second
    /// flag check so a concurrent `resume()`'s `notify_waiters()` can't slip
    /// between the check and the await — the classic lost-wakeup race — and
    /// re-loops in case of a spurious wake.
    pub async fn wait_if_paused(&self) {
        while self.paused.load(Ordering::SeqCst) {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.paused.load(Ordering::SeqCst) {
                break;
            }
            #[cfg(test)]
            {
                self.waiters.fetch_add(1, Ordering::SeqCst);
                self.waiter_notify.notify_waiters();
            }
            notified.await;
            #[cfg(test)]
            self.waiters.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[cfg(test)]
    pub async fn wait_until_blocked(&self) {
        while self.waiters.load(Ordering::SeqCst) == 0 {
            let notified = self.waiter_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.waiters.load(Ordering::SeqCst) > 0 {
                break;
            }
            notified.await;
        }
    }
}

/// Runs the generation queue worker loop. Processes one job at a time (FIFO),
/// but uses a small bounded lookahead buffer to prefer jobs whose model is
/// already loaded — minimizing model swaps when the queue interleaves models.
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_worker(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    tracing::debug!("generation queue worker started");
    let buffer_size = resolve_lookahead_buffer();
    let max_deferrals = resolve_max_deferrals();
    let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(buffer_size);

    loop {
        // Hold new-job dispatch while paused. A job already running finishes
        // untouched — this only gates the pull of the *next* job.
        state.queue_pause.wait_if_paused().await;
        if buffer.is_empty() {
            match job_rx.recv().await {
                Some(j) => buffer.push_back(BufferedJob::new(j)),
                None => break,
            }
        }
        // Top up the buffer without blocking — drain the channel up to capacity.
        top_up_buffer(&mut buffer, &mut job_rx, buffer_size);
        // Re-check after the recv: a pause that landed while this loop was
        // parked waiting for work must hold the job that woke it, not leak
        // it into dispatch.
        state.queue_pause.wait_if_paused().await;

        // Honor user reorders (`PATCH /api/queue/:id {position}`) before the
        // model-swap picker runs — the registry is the single source of truth
        // for order, so aligning the buffer to it makes a reorder change real
        // dispatch order rather than only the `GET /api/queue` snapshot.
        align_buffer_to_registry_order(&mut buffer, &state.job_registry.queued_ids_in_order());

        let loaded = single_gpu_loaded_models(&state).await;
        let job = pick_next_job(&mut buffer, &loaded, max_deferrals);
        let job_id = job.id.clone();

        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
        process_job(&state, job).await;
        state.queue.decrement();
        // Drop the registry entry on every terminal path — the worker
        // here doesn't own a drop guard, so we do it inline alongside
        // the queue counter decrement.
        state.job_registry.remove(&job_id);
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
    }
    tracing::info!("generation queue worker shutting down");
}

async fn single_gpu_loaded_models(state: &AppState) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    let cache = state.model_cache.lock().await;
    if let Some(name) = cache.active_model() {
        set.insert(name.to_string());
    }
    set
}

/// Build the set of "currently loaded somewhere" model names across every
/// worker in the multi-GPU pool. A worker counts the model as loaded if
/// either it's in the worker's cache as Gpu-resident OR it's the worker's
/// `active_generation` (covering the take-and-restore window where the
/// cache entry briefly disappears).
fn multi_gpu_loaded_models(state: &AppState) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    for worker in state.gpu_pool.worker_snapshot() {
        if let Ok(active_gen) = worker.active_generation.read() {
            if let Some(g) = active_gen.as_ref() {
                set.insert(g.model.clone());
            }
        }
        if let Ok(resident) = worker.resident_model.read() {
            if let Some(name) = resident.as_ref() {
                set.insert(crate::gpu_pool::resident_model_display_name(name).to_string());
            }
        }
    }
    set
}

/// In-flight wrapper that tracks how many times the picker has skipped this
/// job. Once the count exceeds `max_deferrals`, the picker force-dispatches
/// it to bound starvation.
pub(crate) struct BufferedJob {
    pub(crate) job: GenerationJob,
    pub(crate) deferred: usize,
}

impl BufferedJob {
    fn new(job: GenerationJob) -> Self {
        Self { job, deferred: 0 }
    }
}

/// Drain the receive channel into the lookahead buffer, capped at
/// `buffer_size`. Returns when the buffer is full or the channel has no
/// immediately-available jobs (the receiver is unchanged on `Empty`). Pure
/// helper extracted so tests can lock in the cap as a load-bearing invariant
/// without spinning up the full async dispatcher.
pub(crate) fn top_up_buffer(
    buffer: &mut VecDeque<BufferedJob>,
    job_rx: &mut tokio::sync::mpsc::Receiver<GenerationJob>,
    buffer_size: usize,
) {
    while buffer.len() < buffer_size {
        match job_rx.try_recv() {
            Ok(j) => buffer.push_back(BufferedJob::new(j)),
            Err(_) => break,
        }
    }
}

/// Pure picker for the lookahead buffer. Selects the buffered job whose
/// model is already loaded somewhere in `loaded`; ties broken by arrival
/// order (front of the deque wins). The head's `deferred` count bounds
/// starvation: if the head has been skipped `max_deferrals` times, it wins
/// regardless of `loaded` membership.
///
/// The returned job is removed from the buffer; remaining buffered jobs that
/// were skipped have their `deferred` count incremented. Increments
/// `mold_queue_reorders_total` whenever a non-head job is picked.
pub(crate) fn pick_next_job(
    buffer: &mut VecDeque<BufferedJob>,
    loaded: &std::collections::HashSet<String>,
    max_deferrals: usize,
) -> GenerationJob {
    debug_assert!(
        !buffer.is_empty(),
        "pick_next_job requires non-empty buffer"
    );

    // Force-dispatch the head if it's hit the starvation budget.
    if let Some(head) = buffer.pop_front_if(|head| head.deferred >= max_deferrals) {
        return head.job;
    }

    // Find the front-most buffered job whose model is already loaded.
    let pick_idx = buffer
        .iter()
        .position(|b| loaded.contains(&b.job.request.model))
        .unwrap_or(0);

    if pick_idx > 0 {
        for (i, b) in buffer.iter_mut().enumerate() {
            if i < pick_idx {
                b.deferred += 1;
            }
        }
        let model = buffer[pick_idx].job.request.model.clone();
        tracing::debug!(
            picked_model = %model,
            head_model = %buffer.front().map(|b| b.job.request.model.as_str()).unwrap_or(""),
            picked_index = pick_idx,
            "queue reorder picked non-head job"
        );
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_reorder();
    }

    buffer.remove(pick_idx).expect("pick_idx in range").job
}

/// Reorder the lookahead `buffer` to follow the registry's queued order — the
/// single source of truth that `PATCH /api/queue/:id {position}` mutates.
///
/// The dispatch loops pull jobs off the channel in submission order into a
/// bounded buffer and hand it to [`pick_next_job`], which treats the buffer
/// front as highest priority. Without this step a `reorder_queued` would only
/// change the `GET /api/queue` snapshot while the loop kept consuming its FIFO
/// buffer, so a user's "run this next" would be silently ignored. Aligning the
/// buffer to `order` here makes the reorder drive real dispatch order, while
/// the model-swap picker still runs on top of the (now user-ordered) sequence,
/// bounded by the starvation budget.
///
/// Jobs are stably reordered by their index in `order`; a job whose id isn't
/// present — an empty-id test job, one cancelled out of the registry while it
/// still holds a buffer slot, or one not yet registered — keeps its relative
/// arrival order behind every registry-tracked job, so nothing untracked is
/// promoted ahead of tracked work. When the buffer already matches the
/// registry (the no-reorder steady state) this is a stable no-op that
/// preserves each job's `deferred` starvation count.
pub(crate) fn align_buffer_to_registry_order(buffer: &mut VecDeque<BufferedJob>, order: &[String]) {
    if buffer.len() < 2 {
        return;
    }
    let rank: std::collections::HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    // `sort_by_key` is stable, so jobs sharing the fallback rank (unknown ids)
    // keep their arrival order relative to one another.
    let mut items: Vec<BufferedJob> = buffer.drain(..).collect();
    items.sort_by_key(|b| {
        if b.job.id.is_empty() {
            usize::MAX
        } else {
            rank.get(b.job.id.as_str()).copied().unwrap_or(usize::MAX)
        }
    });
    buffer.extend(items);
}

pub(crate) const DEFAULT_LOOKAHEAD_BUFFER: usize = 8;
pub(crate) const DEFAULT_MAX_DEFERRALS: usize = 3;
pub(crate) const LOOKAHEAD_BUFFER_ENV: &str = "MOLD_QUEUE_LOOKAHEAD_BUFFER";
pub(crate) const MAX_DEFERRALS_ENV: &str = "MOLD_QUEUE_MAX_DEFERRALS";
const LOOKAHEAD_BUFFER_LOWER: usize = 1;
const LOOKAHEAD_BUFFER_UPPER: usize = 64;
const MAX_DEFERRALS_UPPER: usize = 32;

/// Resolve the lookahead buffer size from env, falling back to the default.
/// Out-of-range or unparseable values log a warning and use the default —
/// matching the warn-then-default pattern of `resolve_max_cached_models`.
pub(crate) fn resolve_lookahead_buffer() -> usize {
    match std::env::var(LOOKAHEAD_BUFFER_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if (LOOKAHEAD_BUFFER_LOWER..=LOOKAHEAD_BUFFER_UPPER).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    env = LOOKAHEAD_BUFFER_ENV,
                    value = n,
                    lower = LOOKAHEAD_BUFFER_LOWER,
                    upper = LOOKAHEAD_BUFFER_UPPER,
                    "ignoring out-of-range queue lookahead buffer; using default"
                );
                DEFAULT_LOOKAHEAD_BUFFER
            }
            Err(e) => {
                tracing::warn!(
                    env = LOOKAHEAD_BUFFER_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable queue lookahead buffer; using default"
                );
                DEFAULT_LOOKAHEAD_BUFFER
            }
        },
        Err(_) => DEFAULT_LOOKAHEAD_BUFFER,
    }
}

/// Resolve the max-deferrals starvation budget from env. Out-of-range or
/// unparseable values log a warning and use the default.
pub(crate) fn resolve_max_deferrals() -> usize {
    match std::env::var(MAX_DEFERRALS_ENV) {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if n <= MAX_DEFERRALS_UPPER => n,
            Ok(n) => {
                tracing::warn!(
                    env = MAX_DEFERRALS_ENV,
                    value = n,
                    upper = MAX_DEFERRALS_UPPER,
                    "ignoring out-of-range queue max-deferrals; using default"
                );
                DEFAULT_MAX_DEFERRALS
            }
            Err(e) => {
                tracing::warn!(
                    env = MAX_DEFERRALS_ENV,
                    raw = %raw,
                    error = %e,
                    "ignoring unparseable queue max-deferrals; using default"
                );
                DEFAULT_MAX_DEFERRALS
            }
        },
        Err(_) => DEFAULT_MAX_DEFERRALS,
    }
}

async fn process_job(state: &AppState, mut job: GenerationJob) {
    // Check if client already disconnected before doing any work
    if job.should_cancel_for_observer_disconnect() {
        tracing::debug!("skipping queued job — client disconnected");
        return;
    }

    // The single-worker path owns the same durable dispatch transition as a
    // GPU owner thread. In particular, feeder tickets start queued with an
    // exact runtime token and must become running before terminal settlement
    // can CAS and delete the row. Serialize that DB transition with PATCH and
    // cancellation, but complete SQLite work before taking the scheduler
    // fence so lock contention cannot freeze unrelated grants.
    let _durable_transition = state.queue_journal.lock_durable_transition().await;
    let dispatch_claim = claim_single_worker_dispatch(job.journal.as_ref());

    // DELETE takes this same fence for the bounded registry transition. The
    // durable gate remains held, so cancellation can only win before the DB
    // claim or after the exact attempt token is installed.
    let scheduler_mutation = state.scheduler_mutation_fence.lock().await;
    if let Some(claim) = dispatch_claim {
        match claim {
            crate::queue_journal::DispatchClaim::Exhausted { attempts, cap } => {
                drop(scheduler_mutation);
                let model = job.request.model.clone();
                durable_generation_settlement::refuse_exhausted_dispatch_async(
                    job, &model, attempts, cap,
                )
                .await;
                return;
            }
            crate::queue_journal::DispatchClaim::Fenced => {
                drop(scheduler_mutation);
                let job_id = job.id.clone();
                durable_generation_settlement::refuse_fenced_dispatch(job, &job_id);
                return;
            }
            crate::queue_journal::DispatchClaim::Cancelled => {
                drop(scheduler_mutation);
                finish_single_worker_cancelled(job, true).await;
                return;
            }
            crate::queue_journal::DispatchClaim::Granted
            | crate::queue_journal::DispatchClaim::Untracked => {}
        }
    }

    // Register every ordinary singleton with the shutdown registry and expose
    // that exact token through the public job registry. The guard is created
    // before publication and lives across every await/return below, so a
    // finished attempt can never leave a token behind for a later job id.
    let attempt_cancellation = state.generation_cancel.token(&job.id);
    let _attempt_cancellation_guard = SingleWorkerCancelGuard {
        registry: state.generation_cancel.as_ref(),
        job_id: job.id.clone(),
    };
    let cancellation_installed = state
        .job_registry
        .install_running_cancellation(&job.id, attempt_cancellation.clone());
    if !job.id.is_empty() && !cancellation_installed {
        // A queued DELETE won before this critical section. It already made
        // the durable row terminal; refuse to run or publish the stale payload.
        attempt_cancellation.cancel();
        drop(scheduler_mutation);
        finish_single_worker_cancelled(job, true).await;
        return;
    }

    // Single-GPU path: there's only one slot. `gpu=None` keeps the wire
    // shape consistent with multi-GPU even when we don't know the ordinal.
    state.job_registry.mark_running(&job.id, None);
    drop(scheduler_mutation);
    drop(_durable_transition);

    if attempt_cancellation.is_cancelled() {
        let user_requested = state.job_registry.cancel_requested(&job.id);
        finish_single_worker_cancelled(job, user_requested).await;
        return;
    }

    // The single-worker loop is the execution-slot lease. Keep the durable
    // bundle opaque through all queueing and cancellation-before-start paths,
    // then hydrate off Tokio immediately before reference/model preparation.
    let job_id = job.id.clone();
    let hydrated_media_lease = if let Some(deferred) = job.deferred_media.take() {
        let expected_job_id = job_id.clone();
        let mut request = job.request.clone();
        match tokio::task::spawn_blocking(move || {
            let result = deferred.hydrate_into(&expected_job_id, &mut request);
            (request, result)
        })
        .await
        {
            Ok((request, Ok(lease))) => {
                job.request = request;
                Some(lease)
            }
            Ok((_request, Err(error))) => {
                durable_generation_settlement::fail_hydration_async(job, &job_id, error).await;
                return;
            }
            Err(_) => {
                durable_generation_settlement::fail_hydration_async(
                    job,
                    &job_id,
                    crate::queue_media_runtime::DeferredQueueMediaError::worker_failure(),
                )
                .await;
                return;
            }
        }
    } else {
        None
    };
    // The ordered references are bound from THIS hydration, under this lease;
    // no job field carries a reference set across admission or dispatch.
    let references = match hydrated_media_lease
        .as_ref()
        .map(|lease| lease.references(&job.request))
        .transpose()
    {
        Ok(references) => references.flatten(),
        Err(error) => {
            durable_generation_settlement::fail_hydration_async(job, &job_id, error).await;
            return;
        }
    };
    let request = match hydrated_media_lease {
        Some(lease) => {
            crate::queue_media_runtime::AttemptQueueMediaRequest::hydrated(&mut job.request, lease)
        }
        None => crate::queue_media_runtime::AttemptQueueMediaRequest::plain(&job.request),
    };

    if attempt_cancellation.is_cancelled() {
        let user_requested = state.job_registry.cancel_requested(&job.id);
        drop(request);
        finish_single_worker_cancelled(job, user_requested).await;
        return;
    }

    // Send "now processing" event (position 0). `id` echoes the
    // server-assigned UUID so reconnecting clients can match progress
    // updates to their persisted card.
    if let Some(ref tx) = job.progress_tx {
        let _ = tx.send(SseMessage::Progress(SseProgressEvent::Queued {
            position: 0,
            id: job.id.clone(),
        }));
    }

    // Reference binding verifies up to one GiB of staged media. Keep that I/O
    // off Tokio's async worker. The opened handles outlive the set; the
    // staging itself stays alive through the hydrated request guard.
    let reference_binding_result = if request.references.is_none() && references.is_none() {
        Ok(Vec::new())
    } else {
        let request = request.zeroizing_clone();
        let cancellation = attempt_cancellation.clone();
        match tokio::task::spawn_blocking(move || {
            crate::reference_uploads::inference_bindings_for_request(
                &request,
                references.as_ref(),
                Some(&cancellation),
            )
        })
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!(
                "reference binding worker did not complete safely"
            )),
        }
    };
    let reference_bindings = match reference_binding_result {
        Ok(bindings) => bindings,
        Err(error) => {
            if mold_inference::is_inference_cancelled(&error) {
                let user_requested = state.job_registry.cancel_requested(&job.id);
                drop(request);
                finish_single_worker_cancelled(job, user_requested).await;
                return;
            }
            let err_msg = request
                .redact_staging_paths(format!("generation reference binding error: {error:#}"));
            drop(request);
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: true },
                err_msg,
            )
            .await;
            return;
        }
    };

    // 1. Ensure model is ready (with progress forwarding)
    let progress_callback = job.progress_tx.as_ref().map(|tx| {
        let tx = tx.clone();
        Arc::new(move |event: mold_inference::ProgressEvent| {
            let _ = tx.send(SseMessage::Progress(progress_to_sse(event)));
        }) as model_manager::EngineProgressCallback
    });

    let activation_hint = model_manager::activation_hint_for_request(state, &request).await;
    let request_has_lora = model_manager::request_has_effective_lora(&request);
    if let Err(api_err) = model_manager::ensure_model_ready(
        state,
        &request.model,
        progress_callback,
        activation_hint,
        request_has_lora,
    )
    .await
    {
        let err_msg = request.redact_staging_paths(api_err.error.clone());
        drop(request);
        durable_generation_settlement::fail_async(
            job,
            DurableDisposition::Hold { retryable: true },
            err_msg,
        )
        .await;
        return;
    }

    if attempt_cancellation.is_cancelled() {
        let user_requested = state.job_registry.cancel_requested(&job.id);
        drop(request);
        finish_single_worker_cancelled(job, user_requested).await;
        return;
    }

    // 2. Low-memory warning (MPS/unified memory only — observability aid)
    #[cfg(target_os = "macos")]
    if let Some(available) = mold_inference::device::available_system_memory_bytes() {
        if available < 1_000_000_000 {
            tracing::warn!(
                available_mb = available / 1_000_000,
                "low memory before inference — system may become unstable"
            );
        }
    }

    // 3. Take the engine out of the cache so the cache mutex stays free during
    //    generation. Mirrors the multi-GPU `gpu_worker::process_job` pattern —
    //    holding the cache lock through inference would block /api/models,
    //    /api/cache, and any concurrent gallery/admin reads.
    let taken = {
        let mut cache = state.model_cache.lock().await;
        cache.take(&request.model)
    };
    let Some(mut cached_engine) = taken else {
        drop(request);
        durable_generation_settlement::fail_async(
            job,
            DurableDisposition::Hold { retryable: true },
            "no engine available after model readiness check",
        )
        .await;
        return;
    };

    let active_gen = state.active_generation.clone();
    let gen_req = request.zeroizing_clone();
    let progress_tx = job.progress_tx.clone();
    let inference_cancellation = attempt_cancellation.clone();

    set_active_generation(state, &request.model, &request.prompt);

    // Install progress capture before crossing into spawn_blocking. The live
    // registry snapshot is useful even when the submitting SSE receiver is
    // gone, so queue selection can still follow the render from another client.
    cached_engine.engine.set_on_progress(Box::new(move |event| {
        forward_generation_progress(progress_tx.as_ref(), event);
    }));

    #[cfg(feature = "metrics")]
    let inference_start = Instant::now();
    // RSS sample taken just before inference; the post-inference sample below
    // logs the per-job delta so RAM growth can be attributed to a specific
    // generation rather than tracked at process granularity.
    let rss_before = crate::resources::ram_snapshot_from_system().used_by_mold;
    // Run generation on the blocking pool. Move the engine in, return it back
    // out (alongside the result + any panic payload) so we can restore it to
    // the cache in async context regardless of outcome.
    let join_result = tokio::task::spawn_blocking(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mold_inference::with_inference_cancellation(
                &mut *cached_engine.engine,
                inference_cancellation,
                |engine| engine.generate_with_reference_bindings(&gen_req, &reference_bindings),
            )
        }));
        cached_engine.engine.clear_on_progress();
        (cached_engine, result)
    })
    .await;

    let rss_after = crate::resources::ram_snapshot_from_system().used_by_mold;
    let rss_delta = rss_after as i64 - rss_before as i64;
    tracing::info!(
        model = %request.model,
        rss_before_mb = rss_before / 1_000_000,
        rss_after_mb = rss_after / 1_000_000,
        rss_delta_mb = rss_delta / 1_000_000,
        "generation memory delta"
    );

    #[cfg(feature = "metrics")]
    let inference_duration = inference_start.elapsed().as_secs_f64();

    // Restore the engine to the cache as soon as the blocking task joins —
    // even panics must restore so the cache isn't left with a hole. If the
    // tokio task itself failed (JoinError), the engine is gone — restoration
    // is impossible. Without `clear_in_flight` the model name would leak
    // forever in `in_flight`, so `ensure_model_ready` keeps fast-pathing
    // through `contains()` while every subsequent `take()` returns `None`,
    // permanently jamming this model. Clear the marker so the cache will
    // legitimately re-load the engine on the next request.
    let result = match join_result {
        Ok((cached_engine, panic_or_result)) => {
            let superseded = {
                let mut cache = state.model_cache.lock().await;
                cache.restore(cached_engine).superseded
            };
            if let Some(superseded) = superseded {
                // The legacy dispatcher has no scheduler owner thread. Keep
                // stale teardown outside the async cache mutex; authoritative
                // CUDA dispatch uses the owner-safe gpu_worker path above.
                drop(superseded);
            }
            clear_active_generation(state);
            Ok(panic_or_result)
        }
        Err(join_err) => {
            {
                let mut cache = state.model_cache.lock().await;
                cache.clear_in_flight(&request.model);
            }
            clear_active_generation(state);
            Err(join_err)
        }
    };

    match result {
        Ok(Ok(Ok(mut response))) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation(&request.model, inference_duration);

            if response.images.is_empty()
                && response.video.is_none()
                && response.audio.is_none()
                && response.mesh.is_none()
            {
                drop(request);
                durable_generation_settlement::fail_async(
                    job,
                    DurableDisposition::Hold { retryable: true },
                    "generation error: engine returned no images, video, audio, or mesh",
                )
                .await;
                return;
            }
            // For video-only responses, synthesize an ImageData from the thumbnail
            // so the existing queue/SSE pipeline can handle it.
            let mut img = if !response.images.is_empty() {
                response.images.remove(0)
            } else if let Some(ref video) = response.video {
                ImageData {
                    data: video.thumbnail.clone(),
                    format: OutputFormat::Png,
                    width: video.width,
                    height: video.height,
                    index: 0,
                }
            } else if let Some(ref audio) = response.audio {
                // The waveform PNG stands in for the raster payload the queue
                // and SSE pipeline expect. Its dimensions are the tile's, and
                // `build_sse_complete_event` re-reads the real audio bytes.
                ImageData {
                    data: audio.thumbnail.clone(),
                    format: OutputFormat::Png,
                    width: audio.thumbnail_width,
                    height: audio.thumbnail_height,
                    index: 0,
                }
            } else if let Some(ref mesh) = response.mesh {
                // The poster PNG stands in for the raster payload, exactly as
                // the waveform does for audio. The real glTF bytes are re-read
                // from `response.mesh` by `build_sse_complete_event`.
                ImageData {
                    data: mesh.poster.clone(),
                    format: OutputFormat::Png,
                    width: mesh.poster_width,
                    height: mesh.poster_height,
                    index: 0,
                }
            } else {
                unreachable!("checked above");
            };
            let mut original_img = None;
            // Post-generation upscaling is a RASTER operation. Every non-image
            // kind reaches here with `img` holding a sidecar tile, and
            // upscaling that would replace the artifact with a bigger picture
            // of its own thumbnail.
            if response.video.is_none()
                && response.audio.is_none()
                && response.mesh.is_none()
                && requested_post_upscale_model(&request).is_some()
            {
                let upscale_result = upscale_generated_image_on_single_worker(
                    state,
                    &request,
                    response.seed_used,
                    img.clone(),
                    job.progress_tx.as_ref(),
                )
                .await;
                let (output, preserved_original, upscale_error) =
                    settle_post_generation_upscale(img, upscale_result);
                img = output;
                original_img = preserved_original;
                if let Some(error) = upscale_error {
                    let error = request.redact_staging_paths(error);
                    tracing::warn!(%error, "post-generation upscale failed; keeping original image");
                }
            }

            // Publication is a lifecycle claim, not merely the next line of
            // the happy path. DELETE and shutdown cancellation use the same
            // registry lock/token, so whichever won before this point decides
            // whether the bytes may become visible or the durable row may be
            // marked complete.
            match state.job_registry.claim_completion(&job.id) {
                crate::job_registry::CompletionClaim::Claimed => {}
                crate::job_registry::CompletionClaim::UserCancelled => {
                    drop(request);
                    finish_single_worker_cancelled(job, true).await;
                    return;
                }
                crate::job_registry::CompletionClaim::AttemptCancelled => {
                    drop(request);
                    finish_single_worker_cancelled(job, false).await;
                    return;
                }
            }

            // Save to output directory if configured.
            // Builds OutputMetadata from the request + the engine's actual
            // seed_used so the DB and embedded chunks agree. Awaited (still
            // off the async loop via spawn_blocking) so the complete event
            // below can carry the saved gallery filenames.
            let mut metadata = request.output_metadata(
                response.seed_used,
                None,
                mold_core::build_info::version_string(),
            );
            // The replay idempotence key, exactly as the GPU worker stamps it.
            // Without it a print saved by this path is unrecognisable to boot
            // replay, which would re-render it into a duplicate — and output
            // filenames are wall-clock, so nothing downstream could merge them.
            metadata.job_id = Some(job.id.clone());
            if let Some(video) = response.video.as_ref() {
                metadata.apply_video_output(video);
                // The source print is not itself upscaled. Its requested
                // model becomes authority for the durable follow-up below.
                metadata.upscale_model = None;
            }
            let video_upscale_model = response.video.as_ref().and_then(|_| {
                requested_post_upscale_model(&request).map(mold_core::manifest::resolve_model_name)
            });
            let mut saved_names = SavedOutputNames::default();
            if let Some(ref dir) = job.output_dir {
                let _gallery_writer = state.gallery_publication_gate.write().await;
                let dir = dir.clone();
                let model = request.model.clone();
                let batch_size = request.batch_size;
                let generation_time_ms = response.generation_time_ms as i64;
                let db = state.metadata_db.clone();
                let events = state.events.clone();
                let gallery_gate = state.gallery_publication_gate.clone();
                let save_task = if let Some(ref mesh) = response.mesh {
                    let mesh_data = mesh.data.clone();
                    let mesh_poster = mesh.poster.clone();
                    let mesh_format = mesh.format;
                    // A mesh has no raster of its own; record the poster
                    // tile's size so the gallery grid has a real aspect ratio
                    // rather than the request's (meaningless) canvas.
                    let mut mesh_metadata = metadata.clone();
                    mesh_metadata.apply_output_dimensions(mesh.poster_width, mesh.poster_height);
                    tokio::task::spawn_blocking(move || SavedOutputNames {
                        output: save_mesh_to_dir(
                            &dir,
                            &mesh_data,
                            &mesh_poster,
                            mesh_format,
                            &model,
                            &mesh_metadata,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                            &gallery_gate,
                        ),
                        original: None,
                    })
                } else if let Some(ref audio) = response.audio {
                    let audio_data = audio.data.clone();
                    let audio_thumbnail = audio.thumbnail.clone();
                    let audio_format = audio.format;
                    // Audio has no raster of its own; record the waveform
                    // tile's size so the gallery grid has a real aspect ratio
                    // instead of the request's (meaningless) video dimensions.
                    let mut audio_metadata = metadata.clone();
                    audio_metadata
                        .apply_output_dimensions(audio.thumbnail_width, audio.thumbnail_height);
                    tokio::task::spawn_blocking(move || SavedOutputNames {
                        output: save_audio_to_dir(
                            &dir,
                            &audio_data,
                            &audio_thumbnail,
                            audio_format,
                            &model,
                            &audio_metadata,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                            &gallery_gate,
                        ),
                        original: None,
                    })
                } else if let Some(ref video) = response.video {
                    let video_data = video.data.clone();
                    let video_gif_preview = video.gif_preview.clone();
                    let video_format = video.format;
                    let video_metadata = metadata.clone();
                    tokio::task::spawn_blocking(move || SavedOutputNames {
                        output: save_video_to_dir(
                            &dir,
                            &video_data,
                            &video_gif_preview,
                            video_format,
                            &model,
                            &video_metadata,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                            &gallery_gate,
                        ),
                        original: None,
                    })
                } else {
                    let img_clone = img.clone();
                    let original_clone = original_img.clone();
                    let metadata_clone = metadata.clone();
                    tokio::task::spawn_blocking(move || {
                        save_generated_image_outputs(
                            &dir,
                            original_clone.as_ref(),
                            &img_clone,
                            &model,
                            batch_size,
                            &metadata_clone,
                            Some(generation_time_ms),
                            db.as_ref().as_ref(),
                            Some(&events),
                            &gallery_gate,
                        )
                    })
                };
                saved_names = save_task.await.unwrap_or_default();
            }

            drop(request);
            // Persist the requested video follow-up before reporting the
            // generation successful. Preparation itself runs on the blocking
            // pool from the Framewise dispatcher.
            if let (Some(model), Some(filename)) = (video_upscale_model, saved_names.output.clone())
            {
                if let Err(error) =
                    crate::video_upscale::create_generated_video_job(state, filename, model).await
                {
                    let message = format!(
                        "video was published, but its requested Framewise upscale could not be queued: {}",
                        error.error
                    );
                    tracing::error!(job_id = %job.id, %message);
                    durable_generation_settlement::fail_async(
                        job,
                        DurableDisposition::Retain,
                        message,
                    )
                    .await;
                    return;
                }
            }

            // Settle the durable row on what actually reached the gallery,
            // mirroring the GPU worker. Left to the ticket's ordinary drop,
            // a render finishing during shutdown would keep its row behind the
            // retention fence and replay into a duplicate print; and a failed
            // publication would delete the row, losing a replayed job outright
            // since the gallery file is its only delivery.
            let job_id = job.id.clone();
            let output_dir = job.output_dir.clone();
            let gallery_gate = state.gallery_publication_gate.clone();
            let completion_payload = job.completion_payload;
            let mut channels =
                durable_generation_settlement::IntoSettlementChannels::into_settlement_channels(
                    job,
                );
            if durable_generation_settlement::settle_publication_async(
                &mut channels,
                &job_id,
                output_dir.as_deref(),
                &state.job_registry,
                &saved_names,
                &response,
                &gallery_gate,
            )
            .await
            .is_err()
            {
                return;
            }
            let completion = channels.progress_tx.is_some().then(|| {
                build_sse_completion_message(
                    &response,
                    &img,
                    original_img.as_ref(),
                    Some(&metadata),
                    &saved_names,
                    completion_payload,
                )
            });
            channels.complete(
                completion,
                GenerationJobResult {
                    image: img,
                    response,
                },
            );
        }
        Ok(Ok(Err(e))) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            if mold_inference::is_inference_cancelled(&e) {
                let user_requested = state.job_registry.cancel_requested(&job.id);
                drop(request);
                finish_single_worker_cancelled(job, user_requested).await;
                return;
            }
            let err_msg = request
                .redact_staging_paths(format!("generation error: {}", clean_error_message(&e)));
            tracing::error!(%err_msg, "generation failed");
            drop(request);
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: true },
                err_msg,
            )
            .await;
        }
        Ok(Err(panic_payload)) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            let msg = panic_payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            let err_msg = request.redact_staging_paths(format!("inference panicked: {msg}"));
            tracing::error!(%err_msg, "inference panicked");
            drop(request);
            // A panic is never auto-replayed and never user-retryable
            // unchanged — the same answer the GPU-owner path gives, where the
            // quarantine fence then outranks the hold and retains the row for
            // the restart. Here nothing restarts, so the hold stands: settled,
            // visible, and fenced off from `POST /api/queue/:id/retry`.
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: false },
                err_msg,
            )
            .await;
        }
        Err(join_err) => {
            #[cfg(feature = "metrics")]
            crate::metrics::record_generation_error(&request.model);

            *active_gen.write().unwrap_or_else(|e| e.into_inner()) = None;
            tracing::error!("inference task join error: {join_err:?}");
            drop(request);
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: true },
                "inference task failed",
            )
            .await;
        }
    }
}

fn claim_single_worker_dispatch(
    ticket: Option<&crate::queue_journal::QueueTicket>,
) -> Option<crate::queue_journal::DispatchClaim> {
    ticket.map(crate::queue_journal::QueueTicket::claim_dispatch)
}

/// Unregister an ordinary single-worker attempt from the shutdown registry on
/// every exit path, including model-load errors and panics restored through
/// the blocking-task boundary.
struct SingleWorkerCancelGuard<'a> {
    registry: &'a crate::generation_cancel::CancelRegistry,
    job_id: String,
}

impl Drop for SingleWorkerCancelGuard<'_> {
    fn drop(&mut self) {
        self.registry.unregister(&self.job_id);
    }
}

/// Settle an attempt whose cancellation authority won before publication.
/// Explicit user cancellation is terminal; shutdown/attempt cancellation is
/// retained for feeder replay. Neither path can publish gallery bytes or mark
/// a durable batch child complete.
async fn finish_single_worker_cancelled(job: GenerationJob, user_requested: bool) {
    let model = job.request.model.clone();
    durable_generation_settlement::finish_cancelled_async(job, &model, user_requested).await;
}

// ── Multi-GPU queue dispatcher ──────────────────────────────────────────────

/// Runs the multi-GPU dispatch loop. Routes each generation job to the best
/// GPU worker per `select_worker`'s tier order: model loaded + idle > idle
/// empty GPU that fits > model loaded but busy > idle empty GPU that does
/// not fit > most-headroom fallback (evict LRU there).
/// Uses a small lookahead buffer so an interleaved queue (`[A, B, A, B]`)
/// doesn't force a sibling worker to swap models when one already has the
/// right one warm.
///
/// Exits when the sender half of the channel is dropped (server shutdown).
pub async fn run_queue_dispatcher(
    job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
) {
    run_queue_dispatcher_until_cancelled(job_rx, state, tokio_util::sync::CancellationToken::new())
        .await;
}

pub async fn run_queue_dispatcher_until_cancelled(
    job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) {
    tracing::debug!("multi-GPU queue dispatcher started");
    let buffer_size = resolve_lookahead_buffer();
    let max_deferrals = resolve_max_deferrals();
    run_queue_dispatcher_with_tuning(job_rx, state, buffer_size, max_deferrals, shutdown).await;
}

const LEGACY_UNAVAILABLE_INITIAL_WAIT: std::time::Duration = std::time::Duration::from_millis(250);
const LEGACY_UNAVAILABLE_MAX_WAIT: std::time::Duration = std::time::Duration::from_secs(2);
const LEGACY_GENERATION_AUTHORITY_POLL: std::time::Duration = std::time::Duration::from_millis(50);

struct LegacyUnavailableBackoff {
    next_wait: std::time::Duration,
    warning_emitted: bool,
}

impl LegacyUnavailableBackoff {
    fn new() -> Self {
        Self {
            next_wait: LEGACY_UNAVAILABLE_INITIAL_WAIT,
            warning_emitted: false,
        }
    }

    /// Return whether this is the transition into unavailable state. The
    /// caller logs only on that transition instead of once per retry tick.
    fn enter_unavailable(&mut self) -> bool {
        !std::mem::replace(&mut self.warning_emitted, true)
    }

    fn take_wait(&mut self) -> std::time::Duration {
        let wait = self.next_wait;
        self.next_wait = self
            .next_wait
            .saturating_mul(2)
            .min(LEGACY_UNAVAILABLE_MAX_WAIT);
        wait
    }
}

async fn wait_for_legacy_worker_retry(
    shutdown: &tokio_util::sync::CancellationToken,
    wait: std::time::Duration,
) -> bool {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => false,
        _ = tokio::time::sleep(wait) => true,
    }
}

async fn wait_for_legacy_generation_retry(
    shutdown: &tokio_util::sync::CancellationToken,
    registry_mutation: &tokio::sync::Notify,
    wait: std::time::Duration,
) -> bool {
    // Notify is shared with the scheduler and dependency waiters. Bound the
    // fallback so another legitimate subscriber consuming the permit cannot
    // delay cancellation authority behind the worker backoff.
    let wait = wait.min(LEGACY_GENERATION_AUTHORITY_POLL);
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => false,
        _ = registry_mutation.notified() => true,
        _ = tokio::time::sleep(wait) => true,
    }
}

pub async fn run_legacy_scheduled_work_dispatcher(
    mut scheduled_work_rx: tokio::sync::mpsc::Receiver<crate::scheduler::ScheduledOwnerWork>,
    mut owner_event_rx: tokio::sync::mpsc::UnboundedReceiver<crate::gpu_worker::LegacyOwnerEvent>,
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
) {
    let mut scheduled_closed = false;
    let mut followups_closed = false;
    loop {
        if !wait_for_legacy_dispatch(&state, &shutdown).await {
            break;
        }
        if scheduled_closed && followups_closed {
            break;
        }
        let work = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            work = scheduled_work_rx.recv(), if !scheduled_closed => {
                match work {
                    Some(work) => Some(work),
                    None => {
                        scheduled_closed = true;
                        None
                    }
                }
            }
            event = owner_event_rx.recv(), if !followups_closed => {
                match event {
                    Some(crate::gpu_worker::LegacyOwnerEvent::FollowupReady(work)) => Some(*work),
                    None => {
                        followups_closed = true;
                        None
                    }
                }
            }
        };
        let Some(work) = work else {
            continue;
        };
        if !dispatch_legacy_scheduled_work(&state, work, &shutdown).await {
            break;
        }
    }
    scheduled_work_rx.close();
    while let Some(work) = scheduled_work_rx.recv().await {
        work.work.reject(legacy_dispatch_stop_message(&state));
    }
    while let Ok(crate::gpu_worker::LegacyOwnerEvent::FollowupReady(work)) =
        owner_event_rx.try_recv()
    {
        work.work.reject(legacy_dispatch_stop_message(&state));
    }
    tracing::info!("legacy GPU utility dispatcher shutting down");
}

async fn dispatch_legacy_scheduled_work(
    state: &AppState,
    mut work: crate::scheduler::ScheduledOwnerWork,
    shutdown: &tokio_util::sync::CancellationToken,
) -> bool {
    if let Err(error) = freeze_legacy_post_upscale_candidates(state, &mut work) {
        work.work.reject(error);
        return true;
    }
    let mut pending = Some(work);
    let mut skip = Vec::new();
    let mut unavailable = LegacyUnavailableBackoff::new();
    loop {
        if !wait_for_legacy_dispatch(state, shutdown).await {
            pending
                .take()
                .expect("legacy scheduled work remains pending")
                .work
                .reject(legacy_dispatch_stop_message(state));
            return false;
        }
        let Some(current) = pending.as_ref() else {
            return true;
        };
        if current.work.is_cancelled() {
            return true;
        }
        if state
            .gpu_pool
            .workers
            .iter()
            .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            pending
                .take()
                .expect("legacy scheduled work remains pending")
                .work
                .reject("fatal CUDA error requires server restart".to_string());
            return false;
        }
        let worker = if let Some(ordinal) = current.hard_ordinal {
            state.gpu_pool.worker_by_ordinal(ordinal)
        } else {
            state.gpu_pool.select_worker_excluding(
                &current.model_fingerprint,
                current.estimated_vram_bytes,
                &skip,
            )
        };
        let Some(worker) = worker else {
            if current.hard_ordinal.is_some() || state.gpu_pool.worker_count() == 0 {
                pending
                    .take()
                    .expect("legacy scheduled work remains pending")
                    .work
                    .reject("requested GPU is unavailable".to_string());
                return true;
            }
            skip.clear();
            if !wait_for_legacy_worker_retry(shutdown, unavailable.take_wait()).await {
                pending
                    .take()
                    .expect("legacy scheduled work remains pending")
                    .work
                    .reject(legacy_dispatch_stop_message(state));
                return false;
            }
            continue;
        };

        let observation = build_observed_dispatch(
            state,
            ObservedDispatchInput {
                work_id: &current.id,
                work_kind: current.work.kind(),
                model_fingerprint: &current.model_fingerprint,
                estimated_vram_bytes: current.estimated_vram_bytes,
                estimated_host_ram_bytes: current.estimated_host_ram_bytes,
                request: None,
                hard_ordinal: current.hard_ordinal,
            },
            &worker,
        );
        let mut current = pending.take().expect("legacy scheduled work is present");
        if !current.utility_plans.is_empty() {
            let selected = current.utility_plans.iter().find(|plan| {
                matches!(
                    plan.placement(),
                    crate::gpu_pool::UtilityPlacement::Device { backend, ordinal }
                        if backend == worker.gpu.backend && ordinal == worker.gpu.ordinal
                )
            });
            let Some(selected) = selected.cloned() else {
                current.work.reject(format!(
                    "legacy GPU owner {} had no exact utility execution plan",
                    worker.gpu.ordinal
                ));
                return true;
            };
            if let Err(error) = current.work.install_utility_plan(selected) {
                current.work.reject(error);
                return true;
            }
        }
        let retry = crate::gpu_pool::OwnerWorkRetry {
            model_fingerprint: current.model_fingerprint,
            estimated_vram_bytes: current.estimated_vram_bytes,
            estimated_host_ram_bytes: current.estimated_host_ram_bytes,
            hard_ordinal: current.hard_ordinal,
            priority: current.priority,
            preferred_ordinal: current.preferred_ordinal,
            candidate_plans: current.candidate_plans,
            queue_rank: 0,
            ready_at_ms: 0,
            bypass_count: 0,
            warm_wait_started_ms: None,
            retry_not_before_ms: None,
            utility_plans: current.utility_plans.clone(),
        };
        let fence = crate::scheduler::LeaseFence {
            work_id: current.id,
            device_id: crate::scheduler::worker_device_id(&worker),
            owner_epoch: worker.owner_epoch,
            state_version: 0,
            plan_version: 0,
            worker_generation: 0,
            memory_sample_generation: 0,
            memory_ledger_sequence: 0,
        };
        worker.reserve_legacy_transport();
        let grant = Box::new(crate::gpu_pool::LeaseGrant {
            fence,
            work: current.work,
            retry: Some(retry),
        });
        match worker.try_send_job(grant) {
            Ok(()) => {
                if let Some(observation) = observation {
                    state.scheduled_work.observations().record(observation);
                }
                return true;
            }
            Err(error) => {
                worker.settle_legacy_transport();
                let grant = match error {
                    std::sync::mpsc::TrySendError::Full(grant)
                    | std::sync::mpsc::TrySendError::Disconnected(grant) => *grant,
                };
                let retry = grant
                    .retry
                    .expect("legacy scheduled work always carries retry metadata");
                pending = Some(crate::scheduler::ScheduledOwnerWork {
                    id: grant.fence.work_id,
                    model_fingerprint: retry.model_fingerprint,
                    estimated_vram_bytes: retry.estimated_vram_bytes,
                    estimated_host_ram_bytes: retry.estimated_host_ram_bytes,
                    hard_ordinal: retry.hard_ordinal,
                    priority: retry.priority,
                    preferred_ordinal: retry.preferred_ordinal,
                    candidate_plans: retry.candidate_plans,
                    utility_plans: retry.utility_plans,
                    work: grant.work,
                });
                if pending
                    .as_ref()
                    .is_some_and(|pending| pending.hard_ordinal.is_none())
                {
                    skip.push(worker.gpu.ordinal);
                    if skip.len() >= state.gpu_pool.worker_count().max(1) {
                        skip.clear();
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
    }
}

fn freeze_legacy_post_upscale_candidates(
    state: &AppState,
    work: &mut crate::scheduler::ScheduledOwnerWork,
) -> Result<(), String> {
    if !matches!(&work.work, crate::gpu_pool::OwnerWork::PostUpscale(_)) {
        return Ok(());
    }
    #[cfg(feature = "expand")]
    let base = work.utility_plans.iter().find_map(|plan| match plan {
        crate::gpu_pool::UtilityExecutionPlan::Upscale(plan) => Some(plan),
        crate::gpu_pool::UtilityExecutionPlan::PromptExpansion(_) => None,
    });
    #[cfg(not(feature = "expand"))]
    let base = work.utility_plans.first().map(|plan| match plan {
        crate::gpu_pool::UtilityExecutionPlan::Upscale(plan) => plan,
    });
    let base = base.ok_or_else(|| {
        "post-generation upscaling lacked a frozen artifact candidate".to_string()
    })?;
    let placements = std::iter::once(crate::gpu_pool::UtilityPlacement::Cpu).chain(
        state
            .gpu_pool
            .schedulable_workers()
            .into_iter()
            .map(|worker| crate::gpu_pool::UtilityPlacement::Device {
                backend: worker.gpu.backend,
                ordinal: worker.gpu.ordinal,
            }),
    );
    work.utility_plans = crate::scheduler::upscale_utility_candidates(
        &base.model_name,
        &base.weights,
        base.artifact_root.as_deref(),
        placements,
    );
    Ok(())
}

async fn wait_for_legacy_dispatch(
    state: &AppState,
    shutdown: &tokio_util::sync::CancellationToken,
) -> bool {
    if shutdown.is_cancelled()
        || state
            .gpu_pool
            .workers
            .iter()
            .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
    {
        return false;
    }
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => false,
        _ = state.queue_pause.wait_if_paused() => {
            !state.gpu_pool.workers.iter().any(|worker| {
                worker.fatal_cuda_error.load(Ordering::SeqCst)
            })
        }
    }
}

fn legacy_dispatch_stop_message(state: &AppState) -> String {
    if state
        .gpu_pool
        .workers
        .iter()
        .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
    {
        "fatal CUDA error requires server restart".to_string()
    } else {
        "GPU work was not started because the server is shutting down".to_string()
    }
}

async fn run_queue_dispatcher_with_tuning(
    job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
    buffer_size: usize,
    max_deferrals: usize,
    shutdown: tokio_util::sync::CancellationToken,
) {
    run_queue_dispatcher_with_tuning_inner(
        job_rx,
        state,
        buffer_size,
        max_deferrals,
        shutdown,
        #[cfg(test)]
        None,
    )
    .await;
}

#[cfg(test)]
pub(crate) struct LegacyGenerationDispatchHook {
    pub(crate) selected: Arc<tokio::sync::Notify>,
    pub(crate) resume: Arc<tokio::sync::Notify>,
    pub(crate) upscale_preparation: Option<LegacyUpscalePreparationHook>,
}

#[cfg(test)]
pub(crate) struct LegacyUpscalePreparationHook {
    pub(crate) started: Arc<tokio::sync::Notify>,
    pub(crate) resume: Arc<tokio::sync::Notify>,
}

async fn run_queue_dispatcher_with_tuning_inner(
    mut job_rx: tokio::sync::mpsc::Receiver<GenerationJob>,
    state: AppState,
    buffer_size: usize,
    max_deferrals: usize,
    shutdown: tokio_util::sync::CancellationToken,
    #[cfg(test)] mut before_final_dispatch: Option<LegacyGenerationDispatchHook>,
) {
    let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(buffer_size);

    'dispatcher: loop {
        if shutdown.is_cancelled()
            || state
                .gpu_pool
                .workers
                .iter()
                .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
        {
            break;
        }
        // Hold new-job dispatch while paused; in-flight worker jobs continue.
        if !wait_for_legacy_dispatch(&state, &shutdown).await {
            break;
        }
        if buffer.is_empty() {
            match tokio::select! {
                biased;
                _ = shutdown.cancelled() => None,
                job = job_rx.recv() => job,
            } {
                Some(j) => buffer.push_back(BufferedJob::new(j)),
                None => break,
            }
        }
        top_up_buffer(&mut buffer, &mut job_rx, buffer_size);
        // Re-check after the recv: a pause that landed while this loop was
        // parked waiting for work must hold the job that woke it, not leak
        // it into dispatch.
        if !wait_for_legacy_dispatch(&state, &shutdown).await {
            break;
        }

        // Honor user reorders (`PATCH /api/queue/:id {position}`) before the
        // model-swap picker runs — the registry is the single source of truth
        // for order, so aligning the buffer to it makes a reorder change real
        // dispatch order rather than only the `GET /api/queue` snapshot.
        // Reorder is within-host; per-lane `target_gpu` semantics are applied
        // later when a worker is selected for the picked job.
        let loaded = multi_gpu_loaded_models(&state);
        // Selection is a revocable read. Capture the exact bounded registry
        // order and target while holding the same scheduler fence used by
        // PATCH/cancellation. Downloads and worker availability waits happen
        // after this guard is released; the final send revalidates both facts
        // before it can promote the registry row to Running.
        let (job, selected_order, selected_target) = {
            let _scheduler_mutation = state.scheduler_mutation_fence.lock().await;
            let snapshot = state.job_registry.scheduler_snapshot();
            let order = snapshot
                .iter()
                .filter(|entry| entry.state == crate::job_registry::JobLifecycle::Queued)
                .map(|entry| entry.id.clone())
                .collect::<Vec<_>>();
            align_buffer_to_registry_order(&mut buffer, &order);
            let job = pick_next_job(&mut buffer, &loaded, max_deferrals);
            let target = snapshot
                .iter()
                .find(|entry| entry.id == job.id)
                .map(|entry| entry.target_gpu);
            (job, order, target)
        };

        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());

        let job_id = job.id.clone();
        let model_name = job.request.model.clone();
        let estimated_vram = estimate_model_vram(&model_name);
        let tracked = selected_target.is_some();
        if !job_id.is_empty() && !tracked {
            // Every real generation id is registry-authoritative. A buffered
            // row missing from the snapshot was cancelled after entering the
            // transport; treating it like an old id-less test job can execute
            // work after DELETE already returned 204.
            let _durable_transition = state.queue_journal.lock_durable_transition().await;
            reject_cancelled_generation_job(&state, job).await;
            continue;
        }

        let shape_bucket = crate::gpu_pool::oom_shape_bucket_with_projection(
            &job.request,
            job.deferred_media.as_ref().map(|media| media.projection()),
        );
        if let Some(err_msg) =
            crate::gpu_pool::model_unschedulable_message(&model_name, Some(&shape_bucket))
        {
            tracing::warn!(model = %model_name, "{err_msg}");
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: false },
                err_msg,
            )
            .await;
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        let preferred_gpu = match legacy_generation_preferred_gpu(
            &state,
            &job_id,
            job.request.placement.as_ref(),
        ) {
            Ok(ordinal) => ordinal,
            Err(err_msg) => {
                tracing::warn!(model = %model_name, "{err_msg}");
                durable_generation_settlement::fail_async(
                    job,
                    DurableDisposition::Hold { retryable: false },
                    err_msg,
                )
                .await;
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                #[cfg(feature = "metrics")]
                crate::metrics::record_queue_depth(state.queue.pending());
                continue;
            }
        };
        if job.should_cancel_for_observer_disconnect() {
            tracing::debug!(model = %model_name, "skipping queued multi-GPU job — client disconnected");
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        // Multi-GPU workers are synchronous threads and cannot pull missing
        // assets themselves. Resolve a first-use post-generation upscaler at
        // this async boundary, on the server/host that accepted the job,
        // before handing it to the selected GPU.
        let registry_mutation = state.job_registry.mutation_notifier();
        let mut upscale_download = Box::pin(ensure_legacy_post_upscale_model_downloaded(
            &state,
            &job.request,
            job.progress_tx.as_ref(),
            #[cfg(test)]
            before_final_dispatch
                .as_ref()
                .and_then(|hook| hook.upscale_preparation.as_ref()),
        ));
        let upscale_result = loop {
            let mutation = registry_mutation.notified();
            tokio::pin!(mutation);
            if tracked
                && state.job_registry.scheduler_lifecycle(&job_id)
                    != Some(crate::job_registry::JobLifecycle::Queued)
            {
                drop(upscale_download);
                let _durable_transition = state.queue_journal.lock_durable_transition().await;
                if state.job_registry.scheduler_lifecycle(&job_id)
                    != Some(crate::job_registry::JobLifecycle::Queued)
                {
                    reject_cancelled_generation_job(&state, job).await;
                    continue 'dispatcher;
                }
                upscale_download = Box::pin(ensure_legacy_post_upscale_model_downloaded(
                    &state,
                    &job.request,
                    job.progress_tx.as_ref(),
                    #[cfg(test)]
                    before_final_dispatch
                        .as_ref()
                        .and_then(|hook| hook.upscale_preparation.as_ref()),
                ));
            }
            tokio::select! {
                biased;
                result = &mut upscale_download => break result,
                _ = shutdown.cancelled() => {
                    drop(upscale_download);
                    reject_generation_job(&state, job, legacy_dispatch_stop_message(&state)).await;
                    break 'dispatcher;
                }
                _ = &mut mutation => {}
                _ = tokio::time::sleep(LEGACY_GENERATION_AUTHORITY_POLL) => {}
            }
        };
        drop(upscale_download);
        if tracked
            && state.job_registry.scheduler_lifecycle(&job_id)
                != Some(crate::job_registry::JobLifecycle::Queued)
        {
            let _durable_transition = state.queue_journal.lock_durable_transition().await;
            if state.job_registry.scheduler_lifecycle(&job_id)
                != Some(crate::job_registry::JobLifecycle::Queued)
            {
                reject_cancelled_generation_job(&state, job).await;
                continue 'dispatcher;
            }
        }
        if let Err(err_msg) = upscale_result {
            tracing::warn!(
                model = %model_name,
                upscaler = ?job.request.upscale_model,
                "{err_msg}"
            );
            durable_generation_settlement::fail_async(
                job,
                DurableDisposition::Hold { retryable: true },
                err_msg,
            )
            .await;
            state.queue.decrement();
            state.job_registry.remove(&job_id);
            #[cfg(feature = "metrics")]
            crate::metrics::record_queue_depth(state.queue.pending());
            continue;
        }

        // Build the GpuJob once; the retry loop moves it between attempts.
        let mut gpu_job = Some(GpuJob {
            id: job.id.clone(),
            durable_queue_rank: job.durable_queue_rank,
            model: model_name.clone(),
            request: job.request,
            deferred_media: job.deferred_media,
            completion_payload: job.completion_payload,
            progress_tx: job.progress_tx,
            result_tx: job.result_tx,
            output_dir: job.output_dir,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            gallery_publication_gate: state.gallery_publication_gate.clone(),
            queue: state.queue.clone(),
            registry: state.job_registry.clone(),
            events: state.events.clone(),
            execution_plan: None,
            prepared_execution_inputs: None,
            #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
            h3_prepared_attempt: None,
            lease: None,
            journal: job.journal,
        });

        let mut skip: Vec<usize> = if preferred_gpu.is_none() {
            let failed = crate::gpu_pool::failed_ordinals_for_model(&model_name);
            if failed.len() < state.gpu_pool.worker_count() {
                failed
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let mut dispatched = false;
        let mut unavailable = LegacyUnavailableBackoff::new();

        #[cfg(test)]
        if let Some(hook) = before_final_dispatch.take() {
            hook.selected.notify_one();
            hook.resume.notified().await;
        }

        while !dispatched {
            if shutdown.is_cancelled()
                || state
                    .gpu_pool
                    .workers
                    .iter()
                    .any(|worker| worker.fatal_cuda_error.load(Ordering::SeqCst))
            {
                if let Some(job) = gpu_job.take() {
                    crate::gpu_pool::OwnerWork::Generation(Box::new(job))
                        .reject(legacy_dispatch_stop_message(&state));
                }
                break 'dispatcher;
            }
            if gpu_job
                .as_ref()
                .is_some_and(GpuJob::should_cancel_for_observer_disconnect)
            {
                tracing::debug!(
                    model = %model_name,
                    "dropping queued multi-GPU job before dispatch — client disconnected"
                );
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                break;
            }

            if tracked
                && state.job_registry.scheduler_lifecycle(&job_id)
                    != Some(crate::job_registry::JobLifecycle::Queued)
            {
                // Cancellation removes the registry row before its durable
                // transaction finishes. Wait for that transaction boundary,
                // then release runtime capacity and settle the observer.
                let _durable_transition = state.queue_journal.lock_durable_transition().await;
                if state.job_registry.scheduler_lifecycle(&job_id)
                    != Some(crate::job_registry::JobLifecycle::Queued)
                {
                    let pending = gpu_job
                        .take()
                        .expect("cancelled selected job remains pending");
                    reject_cancelled_generation_job(
                        &state,
                        generation_from_legacy_gpu_job(pending),
                    )
                    .await;
                    continue 'dispatcher;
                }
            }

            let worker = if let Some(ordinal) = preferred_gpu {
                state.gpu_pool.worker_by_ordinal(ordinal)
            } else {
                state
                    .gpu_pool
                    .select_worker_excluding(&model_name, estimated_vram, &skip)
            };

            let Some(worker) = worker else {
                if preferred_gpu.is_none() && state.gpu_pool.worker_count() > 0 {
                    if unavailable.enter_unavailable() {
                        tracing::warn!(
                            model = %model_name,
                            "all GPU workers are temporarily unavailable; keeping job queued"
                        );
                    }
                    let registry_mutation = state.job_registry.mutation_notifier();
                    if !wait_for_legacy_generation_retry(
                        &shutdown,
                        &registry_mutation,
                        unavailable.take_wait(),
                    )
                    .await
                    {
                        if let Some(job) = gpu_job.take() {
                            crate::gpu_pool::OwnerWork::Generation(Box::new(job))
                                .reject(legacy_dispatch_stop_message(&state));
                        }
                        break 'dispatcher;
                    }
                    continue;
                }
                let rejected = gpu_job
                    .take()
                    .expect("gpu_job retained after failed dispatch");
                let err_msg = if state.gpu_pool.worker_count() == 0 {
                    format!("no GPU available for model {model_name}")
                } else if let Some(ordinal) = preferred_gpu {
                    format!("gpu:{ordinal} is not available for model {model_name}")
                } else {
                    format!("no GPU worker available for model {model_name}")
                };
                tracing::error!(model = %model_name, "{err_msg}");
                if let Some(tx) = rejected.progress_tx {
                    let _ = tx.send(SseMessage::Error(SseErrorEvent::failed(err_msg.clone())));
                }
                let _ = rejected.result_tx.send(Err(err_msg));
                state.queue.decrement();
                state.job_registry.remove(&job_id);
                break;
            };

            let observed_dispatch = build_observed_dispatch(
                &state,
                ObservedDispatchInput {
                    work_id: &job_id,
                    work_kind: mold_scheduler::WorkKind::Generation,
                    model_fingerprint: &model_name,
                    estimated_vram_bytes: estimated_vram,
                    estimated_host_ram_bytes: 0,
                    request: gpu_job.as_ref().map(|job| &job.request),
                    hard_ordinal: preferred_gpu,
                },
                &worker,
            );

            // PATCH and cancellation hold the durable-transition gate across
            // their durable commit and bounded registry acknowledgement. Take
            // it before the scheduler fence so the final validation + send is
            // linear with the complete mutation protocol, while doing no DB
            // or awaited I/O beneath the scheduler fence.
            let durable_transition = state.queue_journal.lock_durable_transition().await;
            let scheduler_mutation = state.scheduler_mutation_fence.lock().await;
            let current_snapshot = state.job_registry.scheduler_snapshot();
            let current_order = current_snapshot
                .iter()
                .filter(|entry| entry.state == crate::job_registry::JobLifecycle::Queued)
                .map(|entry| entry.id.clone())
                .collect::<Vec<_>>();
            let current_target = current_snapshot
                .iter()
                .find(|entry| entry.id == job_id)
                .map(|entry| entry.target_gpu);
            if tracked
                && (current_order != selected_order
                    || current_target != selected_target
                    || current_snapshot.iter().any(|entry| {
                        entry.id == job_id
                            && entry.state != crate::job_registry::JobLifecycle::Queued
                    }))
            {
                drop(scheduler_mutation);
                drop(durable_transition);
                let pending = gpu_job.take().expect("stale selected job remains pending");
                if current_snapshot.iter().any(|entry| entry.id == job_id) {
                    buffer.push_front(BufferedJob::new(generation_from_legacy_gpu_job(pending)));
                } else {
                    reject_generation_job(
                        &state,
                        generation_from_legacy_gpu_job(pending),
                        "Cancelled".to_string(),
                    )
                    .await;
                }
                continue 'dispatcher;
            }
            let fresh_preferred_gpu = match legacy_generation_preferred_gpu(
                &state,
                &job_id,
                gpu_job
                    .as_ref()
                    .and_then(|job| job.request.placement.as_ref()),
            ) {
                Ok(target) => target,
                Err(error) => {
                    drop(scheduler_mutation);
                    drop(durable_transition);
                    reject_generation_job(
                        &state,
                        generation_from_legacy_gpu_job(
                            gpu_job
                                .take()
                                .expect("invalid selected job remains pending"),
                        ),
                        error,
                    )
                    .await;
                    continue 'dispatcher;
                }
            };
            if fresh_preferred_gpu != preferred_gpu {
                drop(scheduler_mutation);
                drop(durable_transition);
                buffer.push_front(BufferedJob::new(generation_from_legacy_gpu_job(
                    gpu_job
                        .take()
                        .expect("retargeted selected job remains pending"),
                )));
                continue 'dispatcher;
            }

            // Reserve rollback transport capacity before sending. Execution
            // ownership remains the legacy owner's binary claim after dequeue.
            worker.reserve_legacy_transport();
            let pending = gpu_job.take().expect("gpu_job present in retry loop");
            let lease = crate::scheduler::LeaseFence {
                work_id: pending.id.clone(),
                device_id: crate::scheduler::worker_device_id(&worker),
                owner_epoch: worker.owner_epoch,
                state_version: 0,
                plan_version: 0,
                worker_generation: 1,
                memory_sample_generation: 0,
                memory_ledger_sequence: 0,
            };
            let grant = crate::gpu_pool::LeaseGrant {
                fence: lease,
                work: crate::gpu_pool::OwnerWork::Generation(Box::new(pending)),
                retry: None,
            };
            let mut disconnected = false;
            let dispatch = if tracked {
                state.job_registry.dispatch_if_queued(
                    &job_id,
                    worker.gpu.ordinal,
                    Box::new(grant),
                    |grant| {
                        worker.try_send_job(grant).map_err(|error| match error {
                            std::sync::mpsc::TrySendError::Full(grant) => grant,
                            std::sync::mpsc::TrySendError::Disconnected(grant) => {
                                disconnected = true;
                                grant
                            }
                        })
                    },
                )
            } else {
                worker
                    .try_send_job(Box::new(grant))
                    .map(|()| None)
                    .map_err(|error| match error {
                        std::sync::mpsc::TrySendError::Full(grant) => {
                            crate::job_registry::DispatchAttemptError::Transport(grant)
                        }
                        std::sync::mpsc::TrySendError::Disconnected(grant) => {
                            disconnected = true;
                            crate::job_registry::DispatchAttemptError::Transport(grant)
                        }
                    })
            };
            drop(scheduler_mutation);
            drop(durable_transition);
            match dispatch {
                Ok(_) => {
                    if let Some(observation) = observed_dispatch {
                        state.scheduled_work.observations().record(observation);
                    }
                    dispatched = true;
                }
                Err(crate::job_registry::DispatchAttemptError::Claim(_, grant)) => {
                    worker.settle_legacy_transport();
                    let pending = generation_from_legacy_grant(*grant);
                    if state.job_registry.scheduler_lifecycle(&job_id).is_some() {
                        buffer
                            .push_front(BufferedJob::new(generation_from_legacy_gpu_job(pending)));
                    } else {
                        reject_generation_job(
                            &state,
                            generation_from_legacy_gpu_job(pending),
                            "Cancelled".to_string(),
                        )
                        .await;
                    }
                    continue 'dispatcher;
                }
                Err(crate::job_registry::DispatchAttemptError::Transport(grant))
                    if !disconnected =>
                {
                    worker.settle_legacy_transport();
                    gpu_job = Some(generation_from_legacy_grant(*grant));
                    if preferred_gpu.is_none() {
                        skip.push(worker.gpu.ordinal);
                        if skip.len() >= state.gpu_pool.worker_count().max(1) {
                            skip.clear();
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                    } else {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                }
                Err(crate::job_registry::DispatchAttemptError::Transport(grant)) => {
                    worker.settle_legacy_transport();
                    tracing::warn!(
                        gpu = worker.gpu.ordinal,
                        "GPU worker disconnected — retrying dispatch"
                    );
                    gpu_job = Some(generation_from_legacy_grant(*grant));
                    if preferred_gpu.is_none() {
                        skip.push(worker.gpu.ordinal);
                    } else {
                        let rejected = gpu_job.take().expect("gpu_job retained after disconnect");
                        durable_generation_settlement::fail_async(
                            rejected,
                            DurableDisposition::Hold { retryable: true },
                            format!(
                                "gpu:{} disconnected while dispatching model {model_name}",
                                worker.gpu.ordinal
                            ),
                        )
                        .await;
                        state.queue.decrement();
                        state.job_registry.remove(&job_id);
                        break;
                    }
                }
            }
        }
        #[cfg(feature = "metrics")]
        crate::metrics::record_queue_depth(state.queue.pending());
    }
    for buffered in buffer {
        reject_generation_job(&state, buffered.job, legacy_dispatch_stop_message(&state)).await;
    }
    job_rx.close();
    while let Some(job) = job_rx.recv().await {
        reject_generation_job(&state, job, legacy_dispatch_stop_message(&state)).await;
    }
    tracing::info!("multi-GPU queue dispatcher shutting down");
}

pub(crate) fn legacy_generation_preferred_gpu(
    state: &AppState,
    job_id: &str,
    placement: Option<&mold_core::DevicePlacement>,
) -> Result<Option<usize>, String> {
    let placement_gpu = state.gpu_pool.resolve_explicit_placement_gpu(placement)?;
    Ok(state
        .job_registry
        .target_gpu(job_id)
        .flatten()
        .or(placement_gpu))
}

async fn reject_generation_job(state: &AppState, job: GenerationJob, message: String) {
    let job_id = job.id.clone();
    durable_generation_settlement::fail_async(job, DurableDisposition::Retain, message).await;
    state.queue.decrement();
    state.job_registry.remove(&job_id);
}

async fn reject_cancelled_generation_job(state: &AppState, job: GenerationJob) {
    let job_id = job.id.clone();
    let model = job.request.model.clone();
    durable_generation_settlement::finish_cancelled_async(job, &model, true).await;
    state.queue.decrement();
    state.job_registry.remove(&job_id);
}

fn generation_from_legacy_grant(grant: crate::gpu_pool::LeaseGrant) -> GpuJob {
    match grant.work {
        crate::gpu_pool::OwnerWork::Generation(job) => *job,
        work => panic!("legacy dispatcher received {:?}", work.kind()),
    }
}

fn generation_from_legacy_gpu_job(job: GpuJob) -> GenerationJob {
    GenerationJob {
        id: job.id,
        durable_queue_rank: job.durable_queue_rank,
        request: job.request,
        deferred_media: job.deferred_media,
        completion_payload: job.completion_payload,
        progress_tx: job.progress_tx,
        result_tx: job.result_tx,
        output_dir: job.output_dir,
        journal: job.journal,
        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
        h3_private_ingress_grant: None,
    }
}

pub(crate) struct ObservedDispatchInput<'a> {
    pub work_id: &'a str,
    pub work_kind: mold_scheduler::WorkKind,
    pub model_fingerprint: &'a str,
    pub estimated_vram_bytes: u64,
    pub estimated_host_ram_bytes: u64,
    pub request: Option<&'a mold_core::GenerateRequest>,
    pub hard_ordinal: Option<usize>,
}

pub(crate) fn build_observed_dispatch(
    state: &AppState,
    input: ObservedDispatchInput<'_>,
    legacy_worker: &crate::gpu_pool::GpuWorker,
) -> Option<crate::dispatch_mode::DispatchObservation> {
    let ObservedDispatchInput {
        work_id,
        work_kind,
        model_fingerprint,
        estimated_vram_bytes,
        estimated_host_ram_bytes,
        request,
        hard_ordinal,
    } = input;
    if !state.scheduled_work.observes_v2_decisions() {
        return None;
    }
    let now_ms = crate::scheduler::monotonic_ms();

    let resource_snapshot = state.resources.latest();
    let sampled_free = resource_snapshot
        .as_ref()
        .map(|snapshot| {
            snapshot
                .gpus
                .iter()
                .map(|gpu| {
                    (
                        (gpu.backend, gpu.ordinal),
                        (
                            gpu.vram_total.saturating_sub(gpu.vram_used),
                            gpu.vram_used_by_mold,
                        ),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let devices = state
        .gpu_pool
        .workers
        .iter()
        .map(|worker| {
            let device_id = crate::scheduler::worker_device_id(&worker);
            let sampled_available_vram = sampled_free
                .get(&(worker.gpu.backend, worker.gpu.ordinal))
                .map(|(free, _)| *free);
            let health = if worker.poisoned.load(Ordering::SeqCst)
                || worker.fatal_cuda_error.load(Ordering::SeqCst)
            {
                mold_scheduler::DeviceHealth::Poisoned
            } else if sampled_available_vram.is_none() {
                // Observe mode has no reservation ledger of its own, so a
                // missing current sample is unavailable comparison data, not
                // permission to substitute discovery-time free memory.
                mold_scheduler::DeviceHealth::Degraded
            } else if worker.consecutive_failures.load(Ordering::SeqCst) >= 3
                && worker
                    .degraded_until
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .is_some_and(|until| Instant::now() < until)
            {
                mold_scheduler::DeviceHealth::Degraded
            } else {
                mold_scheduler::DeviceHealth::Healthy
            };
            let activity = if worker.in_flight.load(Ordering::SeqCst) == 0 {
                mold_scheduler::DeviceActivity::Idle
            } else {
                mold_scheduler::DeviceActivity::Busy
            };
            let sampled_available_vram_bytes = sampled_available_vram
                // Observe comparisons are post-startup telemetry. Unlike the
                // authoritative coordinator's explicitly documented bootstrap
                // allowance, they must not turn a stale discovery sample into
                // a claimed current placement.
                .unwrap_or(0);
            let measured_cache_bytes = worker
                .model_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .active_vram_bytes();
            let reclaimable_cache_bytes = sampled_free
                .get(&(worker.gpu.backend, worker.gpu.ordinal))
                .and_then(|(_, used_by_mold)| *used_by_mold)
                .map(|used_by_mold| measured_cache_bytes.min(used_by_mold))
                .unwrap_or(0);
            let available_vram_bytes = crate::scheduler::effective_available_vram_bytes(
                sampled_available_vram_bytes,
                reclaimable_cache_bytes,
                worker.gpu.total_vram_bytes,
            );
            let warm = worker
                .resident_execution_fingerprint
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .into_iter()
                .map(mold_scheduler::ExecutionFingerprint::new)
                .collect::<BTreeSet<_>>();
            mold_scheduler::DeviceSnapshot {
                id: mold_scheduler::DeviceId::new(device_id),
                backend: match worker.gpu.backend {
                    mold_core::GpuBackend::Metal => mold_scheduler::Backend::Metal,
                    _ => mold_scheduler::Backend::Cuda,
                },
                admin_state: mold_scheduler::DeviceAdminState::Enabled,
                health,
                activity,
                // Observe mode uses the same documented static fallback as
                // authoritative planning until learned Phase E estimates
                // exist. It does not fabricate an observe-only duration.
                available_at_ms: (activity == mold_scheduler::DeviceActivity::Busy).then(|| {
                    now_ms.saturating_add(
                        mold_scheduler::static_timing_for(work_kind).predicted_run_ms,
                    )
                }),
                worker_generation: 0,
                available_vram_bytes,
                warm_execution_fingerprints: warm,
            }
        })
        .collect::<Vec<_>>();

    let candidates = if let Some(request) = request {
        let device_facts = devices
            .iter()
            .filter(|device| device.is_schedulable())
            .filter_map(|device| {
                let worker = state.gpu_pool.workers.iter().find(|worker| {
                    crate::scheduler::worker_device_id(worker) == device.id.as_str()
                })?;
                Some(crate::execution_plan::DeviceFact {
                    id: device.id.to_string(),
                    ordinal: worker.gpu.ordinal,
                    backend: worker.gpu.backend,
                    compute_capability: worker.gpu.compute_capability,
                    available_vram_bytes: device.available_vram_bytes,
                })
            })
            .collect::<Vec<_>>();
        let config = match state.config.try_read() {
            Ok(config) => config,
            Err(_) => {
                tracing::debug!(
                    work_id,
                    "configuration changed while computing read-only V2 observation"
                );
                return None;
            }
        };
        let offload_requested = matches!(
            mold_inference::runtime_env::value("MOLD_OFFLOAD").as_deref(),
            Some("1") | Some("true") | Some("yes")
        );
        match crate::execution_plan::resolve_execution_plans(
            &config,
            request,
            &device_facts,
            offload_requested,
        ) {
            Ok(plans) => {
                let failed = crate::gpu_pool::failed_ordinals_for_model(model_fingerprint);
                plans
                    .into_iter()
                    .filter(|plan| !failed.contains(&plan.device_ordinal))
                    .map(|plan| {
                        mold_scheduler::CandidatePlacement::new(
                            plan.device_id,
                            plan.execution_fingerprint,
                            plan.predicted_host_increment_bytes,
                        )
                        .with_execution_equivalence(plan.execution_equivalence_fingerprint)
                        .with_vram(plan.predicted_vram_peak_bytes)
                        .with_device_available_vram(plan.admitted_available_vram_bytes)
                        .with_static_timing(mold_scheduler::WorkKind::Generation)
                    })
                    .collect()
            }
            Err(error) => {
                tracing::debug!(
                    work_id,
                    error = %error,
                    "generation has no valid read-only V2 execution plan"
                );
                Vec::new()
            }
        }
    } else {
        state
            .gpu_pool
            .workers
            .iter()
            .map(|worker| {
                mold_scheduler::CandidatePlacement::new(
                    crate::scheduler::worker_device_id(&worker),
                    model_fingerprint,
                    estimated_host_ram_bytes,
                )
                .with_vram(estimated_vram_bytes)
                .with_device_available_vram(
                    devices
                        .iter()
                        .find(|device| {
                            device.id.as_str() == crate::scheduler::worker_device_id(&worker)
                        })
                        .map_or(0, |device| device.available_vram_bytes),
                )
                .with_static_timing(work_kind)
            })
            .collect::<Vec<_>>()
    };
    let mut work = mold_scheduler::WorkSnapshot::new(work_id, 0, candidates);
    work.kind = work_kind;
    if let Some(ordinal) = hard_ordinal {
        let hard_device_id = state
            .gpu_pool
            .worker_by_ordinal(ordinal)
            .map(|worker| crate::scheduler::worker_device_id(&worker))
            .unwrap_or_else(|| format!("unavailable:gpu:{ordinal}"));
        work = work.with_hard_device(hard_device_id);
    }
    let work_id = mold_scheduler::WorkId::new(work_id);
    let host_memory_headroom = resource_snapshot.as_ref().map_or(0, |snapshot| {
        let available = snapshot.system_ram.available.unwrap_or_else(|| {
            snapshot
                .system_ram
                .total
                .saturating_sub(snapshot.system_ram.used)
        });
        let safety_floor = (snapshot.system_ram.total.saturating_mul(15) / 100).max(8 << 30);
        available.saturating_sub(safety_floor)
    });
    // Work that frees memory is not asked to prove it has memory — see
    // `WorkKind::releases_resources`. This mirrors the planner's own exemption
    // so a preview never reports an unload as unschedulable.
    let releases_resources = work_kind.releases_resources();
    let mut eligible_idle_device_ids =
        work.candidate_placements
            .iter()
            .filter(|candidate| {
                releases_resources || candidate.incremental_host_ram_bytes <= host_memory_headroom
            })
            .filter_map(|candidate| {
                let device = devices
                    .iter()
                    .find(|device| device.id == candidate.device_id)?;
                let worker = state.gpu_pool.workers.iter().find(|worker| {
                    crate::scheduler::worker_device_id(worker) == device.id.as_str()
                })?;
                (device.is_idle()
                    && (releases_resources
                        || candidate.predicted_vram_bytes <= device.available_vram_bytes)
                    && hard_ordinal.is_none_or(|ordinal| ordinal == worker.gpu.ordinal))
                .then(|| device.id.to_string())
            })
            .collect::<Vec<_>>();
    eligible_idle_device_ids.sort();
    let plan = match mold_scheduler::Planner::default().plan(&mold_scheduler::PlannerSnapshot::new(
        1,
        1,
        now_ms,
        host_memory_headroom,
        devices,
        vec![work],
    )) {
        Ok(plan) => plan,
        Err(error) => {
            // Observation must never become dispatch authority. An invalid
            // comparison snapshot is telemetry loss, not permission to delay
            // or reject the legacy decision.
            tracing::warn!(
                work_id = %work_id,
                error = %error,
                "could not compute read-only V2 dispatch observation"
            );
            return None;
        }
    };
    let v2_device_id = plan
        .immediate_leases
        .iter()
        .find(|lease| lease.work_id == work_id)
        .map(|lease| lease.device_id.to_string());
    let blocked_reason = plan.blocked_reason(&work_id).copied();
    let legacy_device_id = crate::scheduler::worker_device_id(legacy_worker);
    let legacy_setup_warm = legacy_worker
        .resident_model
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_deref()
        == Some(model_fingerprint);
    let v2_setup_warm = v2_device_id.as_deref().and_then(|device_id| {
        state
            .gpu_pool
            .workers
            .iter()
            .find(|worker| crate::scheduler::worker_device_id(worker) == device_id)
            .map(|worker| {
                worker
                    .resident_model
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .as_deref()
                    == Some(model_fingerprint)
            })
    });
    Some(crate::dispatch_mode::DispatchObservation {
        work_id: work_id.to_string(),
        work_kind,
        legacy_device_id,
        v2_device_id,
        blocked_reason,
        eligible_idle_device_ids,
        legacy_setup_warm,
        v2_setup_warm,
    })
}

/// Rough VRAM estimate for a model (used for placement decisions).
pub fn estimate_model_vram(model_name: &str) -> u64 {
    // Use a simple heuristic based on model name patterns.
    // Quantized models are smaller; BF16/FP16 are larger.
    let lower = model_name.to_lowercase();
    if lower.contains("flux2")
        && lower.contains("9b")
        && (lower.contains(":bf16") || lower.contains(":fp16"))
    {
        32_000_000_000 // Klein-9B BF16 needs a 32GB-class card in practice.
    } else if lower.contains(":q4") {
        6_000_000_000 // ~6GB
    } else if lower.contains(":q8") || lower.contains(":fp8") {
        12_000_000_000 // ~12GB
    } else if lower.contains(":bf16") || lower.contains(":fp16") {
        24_000_000_000 // ~24GB
    } else if lower.contains("sd15") || lower.contains("sd1.5") {
        4_000_000_000 // ~4GB
    } else {
        // SDXL (~8GB) and other models default to 8GB.
        8_000_000_000
    }
}

#[cfg(test)]
mod tests {
    /// Every `gallery_added` / `gallery_updated` row this module publishes is
    /// built by `gallery_image_with_filing`, so that is where a mesh row
    /// picks up the poster renderer's revision.
    ///
    /// A client inserts the event row in place and keys its tile cache on
    /// `media_version`. If the event disagreed with `/api/gallery`'s listing,
    /// a freshly generated mesh print would cache one tile on the event and
    /// then discard it on the next listing refresh.
    #[test]
    fn an_announced_mesh_row_carries_the_poster_revision() {
        fn record(filename: &str, format: mold_core::OutputFormat) -> mold_db::GenerationRecord {
            let mut record = mold_db::GenerationRecord::from_save(
                std::path::Path::new("/prints"),
                filename,
                format,
                mold_db::metadata_io::synthesize_from_filename(filename, 1_700_000_000),
                mold_db::RecordSource::Server,
                1_700_000_000_000,
            );
            record.file_mtime_ms = Some(1_700_000_000_000);
            record.file_size_bytes = Some(4096);
            record
        }

        let mesh = super::gallery_image_with_filing(
            &record("chair.glb", mold_core::OutputFormat::Glb),
            None,
        );
        let raster = super::gallery_image_with_filing(
            &record("cat.png", mold_core::OutputFormat::Png),
            None,
        );

        assert_eq!(
            mesh.media_version.as_deref(),
            Some("1700000000000:4096:p2"),
            "the announced mesh row does not carry the poster revision"
        );
        assert_eq!(raster.media_version.as_deref(), Some("1700000000000:4096"));
    }

    #[test]
    fn durable_media_hydrates_only_after_the_single_worker_slot_and_cancel_fences() {
        let source = include_str!("queue.rs");
        let start = source
            .find("async fn process_job(")
            .expect("single-worker generation handler");
        let end = source[start..]
            .find("\nfn claim_single_worker_dispatch(")
            .map(|offset| start + offset)
            .expect("single-worker generation boundary");
        let body = &source[start..end];
        let install = body
            .find("install_running_cancellation")
            .expect("attempt cancellation installation");
        let slot = body.find("mark_running").expect("single-worker slot claim");
        let hydrate = body
            .find("deferred.hydrate_into")
            .expect("slot-bound durable hydration");
        let binding = body
            .find("inference_bindings_for_request")
            .expect("reference preparation");
        let guard = body
            .find("AttemptQueueMediaRequest::hydrated")
            .expect("attempt-scoped hydrated request guard");
        let metadata = body
            .find("request.output_metadata")
            .expect("guard-aware output metadata");
        let cancel_checks = body
            .match_indices("attempt_cancellation.is_cancelled()")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        assert!(install < slot && slot < cancel_checks[0]);
        assert!(cancel_checks[0] < hydrate);
        assert!(hydrate < guard && guard < cancel_checks[1]);
        assert!(cancel_checks[1] < binding);
        assert!(binding < metadata);
        assert!(body.matches("request.zeroizing_clone()").count() >= 2);
        assert!(body[hydrate..].find("job.request.clone()").is_none());
        assert!(body.contains("request.redact_staging_paths"));
        assert!(body.contains("drop(request);\n                    finish_single_worker_cancelled"));
    }
    use super::*;
    use crate::gpu_pool::{GpuPool, GpuWorker};
    use crate::model_cache::ModelCache;
    use crate::state::QueueHandle;
    use mold_core::{GenerateRequest, ImageData, ModelConfig, OutputFormat};
    use mold_db::MetadataDb;
    use mold_inference::device::DiscoveredGpu;
    use mold_inference::shared_pool::SharedPool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex, RwLock};
    use tempfile::TempDir;

    #[test]
    fn durable_terminal_result_carries_saved_gallery_names_and_terminal_facts() {
        let response = mold_core::GenerateResponse {
            mesh: None,
            images: Vec::new(),
            video: None,
            audio: None,
            generation_time_ms: 12_345,
            model: "flux-dev:q8".to_string(),
            seed_used: 99,
            gpu: Some(1),
            request_warnings: Vec::new(),
        };
        let result = SavedOutputNames {
            output: Some("print.png".to_string()),
            original: Some("print_original.png".to_string()),
        }
        .terminal_json(&response);
        let parsed: mold_core::GenerationBatchResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.filename.as_deref(), Some("print.png"));
        assert_eq!(
            parsed.original_filename.as_deref(),
            Some("print_original.png")
        );
        assert_eq!(parsed.seed, Some(99));
        assert_eq!(parsed.generation_time_ms, Some(12_345));
        assert_eq!(parsed.gpu, Some(1));
    }

    #[test]
    fn a_result_settled_without_terminal_facts_still_reads_back() {
        let parsed: mold_core::GenerationBatchResult =
            serde_json::from_str(r#"{"filename":"done.png"}"#).unwrap();
        assert_eq!(parsed.filename.as_deref(), Some("done.png"));
        assert_eq!(parsed.seed, None);
        assert_eq!(parsed.generation_time_ms, None);
        assert_eq!(parsed.gpu, None);
    }

    fn claimed_single_worker_ticket(
        id: &str,
    ) -> (
        Arc<crate::queue_journal::QueueJournal>,
        Arc<Option<MetadataDb>>,
        crate::queue_journal::QueueTicket,
    ) {
        let root = tempfile::tempdir().unwrap().keep();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(&root),
            "single-worker-claim-test",
        ));
        let request = fake_request("mock-model");
        journal
            .record_batch(crate::queue_journal::BatchJournalAdmission {
                id: "batch",
                client_batch_id: "client",
                request_sha256: "sha",
                children: &[crate::queue_journal::JournalAdmission {
                    id,
                    request: &request,
                    output_dir: Some(root.as_path()),
                    target_gpu: None,
                    target_device_id: None,
                    completion_payload: SseCompletionPayload::MetadataOnly,
                    batch_child: false,
                }],
            })
            .unwrap();
        let claim = journal.claim_next_feeder().unwrap().unwrap();
        assert_eq!(claim.row.id, id);
        let ticket = journal.attach_claimed(id, claim.claim_token);
        (journal, db, ticket)
    }

    /// One panic policy for both worker paths.
    ///
    /// A panic is never auto-replayed and never user-retryable unchanged. The
    /// GPU owner adds a quarantine fence on top, which `settle_one` honours
    /// over the hold; nothing here restarts, so the hold is what the row
    /// keeps. The two arms used to disagree in the source, with a comment
    /// explaining why — this asserts they no longer do.
    #[tokio::test]
    async fn a_single_worker_inference_panic_holds_the_row_non_retryably() {
        let (journal, db, ticket) = claimed_single_worker_ticket("panicked");
        assert_eq!(
            claim_single_worker_dispatch(Some(&ticket)),
            Some(crate::queue_journal::DispatchClaim::Granted)
        );
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        durable_generation_settlement::fail_async(
            durable_generation_settlement::SettlementChannels {
                journal: Some(ticket),
                progress_tx: Some(progress_tx),
                result_tx: Some(result_tx),
            },
            DurableDisposition::Hold { retryable: false },
            "inference panicked: boom".to_string(),
        )
        .await;

        let row = mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "panicked")
            .unwrap()
            .expect("a panicked row is parked, never deleted");
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Held);
        let child = mold_db::generation_batches::get_durable(
            db.as_ref().as_ref().unwrap(),
            journal.owner_uuid().unwrap(),
            "batch",
        )
        .unwrap()
        .unwrap()
        .children
        .remove(0);
        assert!(
            !child.retryable,
            "`POST /api/queue/:id/retry` must not re-run a request that panicked"
        );
        match progress_rx.try_recv().expect("one terminal frame") {
            SseMessage::Error(event) => {
                assert!(!event.retained);
                assert_eq!(event.code, None);
            }
            _ => panic!("expected a terminal error frame"),
        }
        assert!(result_rx.await.unwrap().is_err());

        for (file, source) in [
            ("queue.rs", include_str!("queue.rs")),
            ("gpu_worker.rs", include_str!("gpu_worker.rs")),
        ] {
            let arm = source
                .find("\"inference panicked")
                .expect("both worker paths report an inference panic");
            let tail = &source[arm..arm + 800];
            assert!(
                tail.contains("DurableDisposition::Hold { retryable: false }"),
                "{file} must answer a panic with a non-retryable hold"
            );
        }
    }

    #[test]
    fn single_worker_claim_transitions_running_then_terminal_complete() {
        let (journal, db, ticket) = claimed_single_worker_ticket("claimed");
        assert_eq!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "claimed")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Queued
        );
        assert_eq!(
            claim_single_worker_dispatch(Some(&ticket)),
            Some(crate::queue_journal::DispatchClaim::Granted)
        );
        assert_eq!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "claimed")
                .unwrap()
                .unwrap()
                .state,
            mold_db::generation_queue::QueueRowState::Running
        );
        ticket.complete_with_result(Some(r#"{"filename":"claimed.png"}"#));
        assert!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "claimed")
                .unwrap()
                .is_none()
        );
        let batch = journal.generation_batch("batch").unwrap();
        assert_eq!(batch.children[0].state, "complete");
    }

    #[test]
    fn single_worker_refuses_a_stale_claim_without_deleting_its_row() {
        let (journal, db, ticket) = claimed_single_worker_ticket("stale");
        let recovery = journal.recover_feeder_runtime().unwrap();
        assert_eq!(recovery.claims_cleared, 1);
        assert_eq!(
            claim_single_worker_dispatch(Some(&ticket)),
            Some(crate::queue_journal::DispatchClaim::Fenced)
        );
        drop(ticket);
        let row = mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "stale")
            .unwrap()
            .unwrap();
        assert_eq!(row.state, mold_db::generation_queue::QueueRowState::Paused);
        assert_eq!(
            journal.generation_batch("batch").unwrap().children[0].state,
            "paused"
        );
    }

    #[test]
    fn single_worker_keeps_unclaimed_legacy_ticket_behavior() {
        let root = tempfile::tempdir().unwrap().keep();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(&root),
            "single-worker-legacy-test",
        ));
        let request = fake_request("mock-model");
        let request_json = serde_json::to_string(&request).unwrap();
        mold_db::generation_queue::insert(
            db.as_ref().as_ref().unwrap(),
            &mold_db::generation_queue::GenerationQueueRow {
                id: "legacy".to_string(),
                owner_uuid: journal.owner_uuid().unwrap().to_string(),
                state: mold_db::generation_queue::QueueRowState::Queued,
                model: request.model,
                request_json,
                media_set_id: None,
                admission_authority: None,
                output_dir: root,
                target_gpu: None,
                target_device_id: None,
                completion_payload: "full".to_string(),
                seed_pinned: false,
                dispatch_attempts: 0,
                replay_seen: 0,
                held_reason: None,
                created_at_ms: 1,
                updated_at_ms: 1,
                started_at_ms: None,
            },
        )
        .unwrap();
        let ticket = journal.attach("legacy");
        assert_eq!(
            claim_single_worker_dispatch(Some(&ticket)),
            Some(crate::queue_journal::DispatchClaim::Granted)
        );
        ticket.discard();
        assert!(
            mold_db::generation_queue::get(db.as_ref().as_ref().unwrap(), "legacy")
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn single_worker_user_cancellation_emits_canonical_cancelled_observer() {
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = GenerationJob {
            id: "single-worker-dispatch-cancelled".to_string(),
            durable_queue_rank: None,
            request: fake_request("mock-model"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: Some(progress_tx),
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };

        finish_single_worker_cancelled(job, true).await;

        let SseMessage::Error(error) = progress_rx.recv().await.unwrap() else {
            panic!("user cancellation must emit a terminal SSE error frame");
        };
        assert_eq!(
            error.code.as_deref(),
            Some(mold_core::SSE_ERROR_CODE_QUEUED_CANCELLED)
        );
        assert!(matches!(
            result_rx.await.unwrap(),
            Err(ref message) if message == "Cancelled"
        ));
    }

    /// The single worker hydrates its durable media under its own lease and
    /// binds the ordered references from THAT hydration: a sealed file whose
    /// bytes no longer match its descriptor's digest is refused before any
    /// model is prepared, and the refusal names the binding, not a lost set.
    #[cfg(unix)]
    #[tokio::test]
    async fn single_worker_binds_references_from_its_own_hydration() {
        use sha2::{Digest as _, Sha256};

        let home = tempfile::tempdir().unwrap();
        let staging = tempfile::tempdir().unwrap();
        let verified = staging.path().join("verified.media");
        let tampered = staging.path().join("tampered.media");
        std::fs::write(&verified, b"verified-reference-bytes").unwrap();
        std::fs::write(&tampered, b"bytes that were swapped after probing").unwrap();
        let descriptor = |name: &str, digest_of: &[u8]| {
            serde_json::json!({
                "kind": "image",
                "media": { "authority": "descriptor" },
                "provenance": { "name": name, "sha256": format!("{:x}", Sha256::digest(digest_of)) },
                "mime_type": "image/png",
                "width": 1024,
                "height": 768
            })
        };
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "single worker references",
            "model": mold_core::minimax_h3::REF2VA_COMFY,
            "width": mold_core::minimax_h3::DEFAULT_WIDTH,
            "height": mold_core::minimax_h3::DEFAULT_HEIGHT,
            "steps": 4,
            "guidance": 0.0,
            "seed": 7,
            "batch_size": 1,
            "output_format": "mp4",
            "references": [
                descriptor("verified.png", b"verified-reference-bytes"),
                descriptor("tampered.png", b"the bytes the descriptor was probed from"),
            ]
        }))
        .unwrap();
        let staged = crate::reference_uploads::StagedReferences::from_files_for_test(
            &request,
            vec![verified, tampered],
        );
        let (deferred, request_json) = crate::queue_media_runtime::seal_request_for_test(
            home.path(),
            "single-worker-references",
            request,
            Some(&staged),
        );
        drop(staged);

        let state = crate::state::AppState::for_tests();
        state.job_registry.register(
            "single-worker-references",
            mold_core::minimax_h3::REF2VA_COMFY,
        );
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = GenerationJob {
            id: "single-worker-references".to_string(),
            durable_queue_rank: None,
            request: serde_json::from_str(&request_json).unwrap(),
            deferred_media: Some(deferred),
            completion_payload: SseCompletionPayload::Full,
            progress_tx: Some(progress_tx),
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };

        process_job(&state, job).await;

        let error = match result_rx.await.unwrap() {
            Err(error) => error,
            Ok(_) => panic!("a tampered reference must not render"),
        };
        assert!(
            error.contains("generation reference binding error"),
            "{error}"
        );
        assert!(error.contains("resolved reference 2"), "{error}");
        assert!(!error.contains(home.path().to_string_lossy().as_ref()));
        let mut saw_terminal_error = false;
        while let Ok(message) = progress_rx.try_recv() {
            if matches!(message, SseMessage::Error(_)) {
                saw_terminal_error = true;
            }
        }
        assert!(saw_terminal_error);
    }

    #[test]
    fn legacy_unavailable_backoff_warns_once_and_caps_retry_frequency() {
        let mut backoff = LegacyUnavailableBackoff::new();
        assert!(backoff.enter_unavailable());
        assert!(!backoff.enter_unavailable());
        assert_eq!(backoff.take_wait(), std::time::Duration::from_millis(250));
        assert_eq!(backoff.take_wait(), std::time::Duration::from_millis(500));
        assert_eq!(backoff.take_wait(), std::time::Duration::from_secs(1));
        assert_eq!(backoff.take_wait(), std::time::Duration::from_secs(2));
        assert_eq!(backoff.take_wait(), std::time::Duration::from_secs(2));
    }

    #[test]
    fn legacy_restart_without_top_level_identity_drops_missing_component_device_identity() {
        let (worker, _worker_rx) = test_worker(7, 1);
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            QueueHandle::new(queue_tx),
            Arc::new(GpuPool {
                workers: vec![worker].into(),
            }),
            1,
        );
        let mut request = fake_request("mock-model");
        request.placement = Some(mold_core::DevicePlacement {
            text_encoders: mold_core::DeviceRef::Cpu,
            advanced: Some(mold_core::AdvancedPlacement {
                transformer: mold_core::DeviceRef::gpu(7),
                vae: mold_core::DeviceRef::device("cuda:removed"),
                ..mold_core::AdvancedPlacement::default()
            }),
        });

        let replay_target =
            crate::queue_journal::resolve_replay_affinity(&mut request, Some(7), None, |_| None);
        state
            .job_registry
            .register_job("legacy-replay", "mock-model", replay_target, None, None);

        assert_eq!(
            legacy_generation_preferred_gpu(&state, "legacy-replay", request.placement.as_ref())
                .unwrap(),
            None
        );
        let placement = request.placement.unwrap();
        assert_eq!(placement.text_encoders, mold_core::DeviceRef::Cpu);
        let advanced = placement.advanced.unwrap();
        assert_eq!(advanced.transformer, mold_core::DeviceRef::Auto);
        assert_eq!(advanced.vae, mold_core::DeviceRef::Auto);
    }

    #[tokio::test]
    async fn legacy_unavailable_wait_is_immediately_shutdown_cancellable() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        shutdown.cancel();
        assert!(!wait_for_legacy_worker_retry(&shutdown, std::time::Duration::from_secs(60)).await);
    }

    #[tokio::test]
    async fn legacy_generation_unavailable_wait_wakes_for_registry_cancellation() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let mutation = tokio::sync::Notify::new();
        mutation.notify_one();
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            wait_for_legacy_generation_retry(
                &shutdown,
                &mutation,
                std::time::Duration::from_secs(60),
            ),
        )
        .await
        .expect("registry cancellation must bypass worker backoff");
    }

    /// A `GenerateRequest` with the bare minimum fields populated — enough to
    /// hand to `OutputMetadata::from_generate_request` in tests.
    fn fake_request(model: &str) -> GenerateRequest {
        GenerateRequest {
            mesh: None,
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            prompt: "a cat".to_string(),
            negative_prompt: None,
            model: model.to_string(),
            width: 512,
            height: 512,
            steps: 4,
            guidance: 3.5,
            seed: Some(7),
            batch_size: 1,
            output_format: Some(OutputFormat::Png),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 0.75,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: None,
            fps: None,
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    fn fake_image() -> ImageData {
        ImageData {
            // PNG magic bytes — the helpers don't validate, but this keeps
            // the on-disk file from being trivially mistaken for empty.
            data: vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            format: OutputFormat::Png,
            width: 512,
            height: 512,
            index: 0,
        }
    }

    #[test]
    fn multi_gpu_dispatch_identifies_missing_post_upscaler_for_auto_pull() {
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());

        assert_eq!(
            post_upscale_model_to_pull(&mold_core::Config::default(), &req).unwrap(),
            Some("real-esrgan-x4plus:fp16".to_string())
        );

        let tmp = TempDir::new().unwrap();
        let weights = tmp.path().join("realesrgan.safetensors");
        std::fs::write(&weights, b"test weights").unwrap();
        let mut config = mold_core::Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(weights.display().to_string()),
                ..Default::default()
            },
        );
        assert_eq!(post_upscale_model_to_pull(&config, &req).unwrap(), None);

        config
            .models
            .get_mut("real-esrgan-x4plus:fp16")
            .unwrap()
            .transformer = Some(tmp.path().join("missing.safetensors").display().to_string());
        assert_eq!(
            post_upscale_model_to_pull(&config, &req).unwrap(),
            Some("real-esrgan-x4plus:fp16".to_string()),
            "stale config paths should trigger a repair pull"
        );
    }

    fn test_worker(
        ordinal: usize,
        channel_size: usize,
    ) -> (
        Arc<GpuWorker>,
        std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
    ) {
        let (job_tx, job_rx) = std::sync::mpsc::sync_channel(channel_size);
        let worker = Arc::new(GpuWorker {
            owner_epoch: 1,
            gpu: DiscoveredGpu {
                ordinal,
                stable_id: Some(format!("cuda:{ordinal:032x}")),
                raw_cuda_uuid: Some((ordinal as u128).to_be_bytes()),
                device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
                identity_error: None,
                backend: mold_core::types::GpuBackend::Cuda,
                name: format!("gpu{ordinal}"),
                compute_capability: Some((8, 6)),
                pci_bus_id: None,
                total_vram_bytes: 24_000_000_000,
                free_vram_bytes: 24_000_000_000,
            },
            model_cache: Arc::new(Mutex::new(ModelCache::new(3))),
            resident_model: Arc::new(RwLock::new(None)),
            resident_execution_fingerprint: Arc::new(RwLock::new(None)),
            active_generation: Arc::new(RwLock::new(None)),
            model_load_lock: Arc::new(Mutex::new(())),
            shared_pool: Arc::new(Mutex::new(SharedPool::new())),
            legacy_pending: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            legacy_chain_waiters: Default::default(),
            consecutive_failures: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            fatal_cuda_error: Arc::new(AtomicBool::new(false)),
            fatal_cuda_shutdown: Arc::new(tokio::sync::Notify::new()),
            queue_journal: Arc::new(crate::queue_journal::QueueJournal::disabled()),
            generation_cancel: Arc::new(crate::generation_cancel::CancelRegistry::new()),
            shutdown_requested: AtomicBool::new(false),
            drain_state: std::sync::atomic::AtomicU8::new(crate::gpu_pool::DRAIN_RUNNING),
            owner_thread_id: std::sync::OnceLock::new(),
            degraded_until: RwLock::new(None),
            job_tx,
        });
        (worker, job_rx)
    }

    #[test]
    fn observe_planning_is_read_only_until_legacy_transport_accepts_work() {
        let (worker, worker_rx) = test_worker(0, 1);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 0,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu0".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24_000_000_000,
                vram_used: 0,
                vram_used_by_mold: Some(0),
                vram_used_by_other: Some(0),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: None,
                reclaimable_zfs_arc: None,
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        });

        let observation = build_observed_dispatch(
            &state,
            ObservedDispatchInput {
                work_id: "observe-read-only",
                work_kind: mold_scheduler::WorkKind::AdminModelLoad,
                model_fingerprint: "flux-dev:q4",
                estimated_vram_bytes: 6_000_000_000,
                estimated_host_ram_bytes: 1 << 30,
                request: None,
                hard_ordinal: None,
            },
            &worker,
        )
        .expect("observe mode should compute a comparison");

        assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        assert_eq!(state.scheduled_work.observations().snapshot().total, 0);
        assert_eq!(
            observation.v2_device_id.as_deref(),
            worker.gpu.stable_id.as_deref()
        );

        state.scheduled_work.observations().record(observation);
        let snapshot = state.scheduled_work.observations().snapshot();
        assert_eq!(snapshot.total, 1);
        assert_eq!(snapshot.matched, 1);
    }

    #[test]
    fn observe_generation_never_fabricates_a_candidate_without_an_execution_plan() {
        let (worker, worker_rx) = test_worker(0, 1);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 0,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu0".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24_000_000_000,
                vram_used: 0,
                vram_used_by_mold: Some(0),
                vram_used_by_other: Some(0),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: None,
                reclaimable_zfs_arc: None,
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        });
        let request = fake_request("missing-model-with-no-artifacts");

        let observation = build_observed_dispatch(
            &state,
            ObservedDispatchInput {
                work_id: "missing-plan",
                work_kind: mold_scheduler::WorkKind::Generation,
                model_fingerprint: &request.model,
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 0,
                request: Some(&request),
                hard_ordinal: None,
            },
            &worker,
        )
        .expect("invalid placement should still produce blocked telemetry");

        assert_eq!(observation.v2_device_id, None);
        assert!(observation.blocked_reason.is_some());
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn observe_busy_availability_is_absolute_and_preserves_warm_wait_economics() {
        let (busy, _busy_rx) = test_worker(0, 1);
        let (idle, _idle_rx) = test_worker(1, 1);
        busy.in_flight.store(1, Ordering::SeqCst);
        *busy
            .resident_execution_fingerprint
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some("utility-exec".to_string());

        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![busy.clone(), idle.clone()].into(),
        });
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 0,
            gpus: [0, 1]
                .into_iter()
                .map(|ordinal| mold_core::GpuSnapshot {
                    ordinal,
                    name: format!("gpu{ordinal}"),
                    backend: mold_core::GpuBackend::Cuda,
                    vram_total: 24_000_000_000,
                    vram_used: 0,
                    vram_used_by_mold: Some(0),
                    vram_used_by_other: Some(0),
                    gpu_utilization: Some(0),
                })
                .collect(),
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: Some(56 << 30),
                reclaimable_zfs_arc: None,
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        });

        let observation = build_observed_dispatch(
            &state,
            ObservedDispatchInput {
                work_id: "observe-absolute-busy-time",
                work_kind: mold_scheduler::WorkKind::Generation,
                model_fingerprint: "utility-exec",
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 0,
                request: None,
                hard_ordinal: None,
            },
            &idle,
        )
        .expect("observe comparison");

        assert_eq!(
            observation.v2_device_id.as_deref(),
            idle.gpu.stable_id.as_deref(),
            "a 30s busy remainder plus the run must not be treated as an absolute timestamp"
        );
    }

    #[test]
    fn observe_utility_requires_current_telemetry_and_never_truncates_vram_demand() {
        let (worker, worker_rx) = test_worker(0, 1);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );

        let without_sample = build_observed_dispatch(
            &state,
            ObservedDispatchInput {
                work_id: "no-current-sample",
                work_kind: mold_scheduler::WorkKind::AdminModelLoad,
                model_fingerprint: "utility",
                estimated_vram_bytes: 1,
                estimated_host_ram_bytes: 0,
                request: None,
                hard_ordinal: None,
            },
            &worker,
        )
        .expect("missing telemetry should produce blocked comparison data");
        assert_eq!(without_sample.v2_device_id, None);
        assert!(without_sample.blocked_reason.is_some());

        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 0,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu0".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24 << 30,
                vram_used: 0,
                vram_used_by_mold: Some(0),
                vram_used_by_other: Some(0),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: None,
                reclaimable_zfs_arc: None,
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        });
        let oversized = build_observed_dispatch(
            &state,
            ObservedDispatchInput {
                work_id: "oversized-utility",
                work_kind: mold_scheduler::WorkKind::AdminModelLoad,
                model_fingerprint: "utility",
                estimated_vram_bytes: 30 << 30,
                estimated_host_ram_bytes: 0,
                request: None,
                hard_ordinal: None,
            },
            &worker,
        )
        .expect("oversized work should produce blocked comparison data");
        assert_eq!(oversized.v2_device_id, None);
        assert!(oversized.blocked_reason.is_some());
        assert_eq!(worker.pending_or_executing(), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn legacy_and_v2_modes_never_run_the_observe_hook() {
        for mode in [
            crate::dispatch_mode::DispatchMode::Legacy,
            crate::dispatch_mode::DispatchMode::V2,
        ] {
            let (worker, worker_rx) = test_worker(0, 1);
            let mut state = empty_test_state(mold_core::Config::default());
            state.gpu_pool = Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            });
            let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
            state.scheduled_work =
                crate::scheduler::ScheduledWorkHandle::for_mode(scheduled_tx, mode);

            assert!(build_observed_dispatch(
                &state,
                ObservedDispatchInput {
                    work_id: "disabled-observer",
                    work_kind: mold_scheduler::WorkKind::Generation,
                    model_fingerprint: "flux-dev:q4",
                    estimated_vram_bytes: 6_000_000_000,
                    estimated_host_ram_bytes: 0,
                    request: None,
                    hard_ordinal: None,
                },
                &worker,
            )
            .is_none());
            assert_eq!(worker.in_flight.load(Ordering::SeqCst), 0);
            assert!(matches!(
                worker_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ));
        }
    }

    #[tokio::test]
    async fn observe_records_only_after_legacy_transport_accepts_work() {
        let (worker, worker_rx) = test_worker(0, 1);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(
            scheduled_tx,
            crate::dispatch_mode::DispatchMode::Observe,
        );
        state.resources.publish(mold_core::ResourceSnapshot {
            hostname: "test".to_string(),
            timestamp: 0,
            gpus: vec![mold_core::GpuSnapshot {
                ordinal: 0,
                name: "gpu0".to_string(),
                backend: mold_core::GpuBackend::Cuda,
                vram_total: 24_000_000_000,
                vram_used: 0,
                vram_used_by_mold: Some(0),
                vram_used_by_other: Some(0),
                gpu_utilization: Some(0),
            }],
            system_ram: mold_core::RamSnapshot {
                total: 64 << 30,
                used: 8 << 30,
                available: None,
                reclaimable_zfs_arc: None,
                used_by_mold: 0,
                used_by_other: 8 << 30,
            },
            cpu: None,
        });

        worker.reserve_legacy_transport();
        worker
            .try_send_job(Box::new(crate::gpu_pool::LeaseGrant {
                fence: crate::scheduler::LeaseFence {
                    work_id: "filler".to_string(),
                    device_id: crate::scheduler::worker_device_id(&worker),
                    owner_epoch: worker.owner_epoch,
                    state_version: 0,
                    plan_version: 0,
                    worker_generation: 0,
                    memory_sample_generation: 0,
                    memory_ledger_sequence: 0,
                },
                work: crate::gpu_pool::OwnerWork::Probe {
                    id: "filler".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(|| {}),
                },
                retry: None,
            }))
            .unwrap();

        let work = crate::scheduler::ScheduledOwnerWork::new(
            "observed-accepted",
            "flux-dev:q4",
            6_000_000_000,
            crate::gpu_pool::OwnerWork::Probe {
                id: "observed-accepted".to_string(),
                kind: mold_scheduler::WorkKind::AdminModelLoad,
                run: Box::new(|| {}),
            },
        );
        let dispatch_state = state.clone();
        let dispatch = tokio::spawn(async move {
            dispatch_legacy_scheduled_work(
                &dispatch_state,
                work,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert_eq!(
            state.scheduled_work.observations().snapshot().total,
            0,
            "a full legacy transport must not create a fake observation"
        );

        let filler = worker_rx.recv().expect("remove full-channel filler");
        drop(filler);
        worker.settle_legacy_transport();
        let accepted = tokio::task::spawn_blocking(move || {
            worker_rx.recv_timeout(std::time::Duration::from_secs(1))
        })
        .await
        .unwrap()
        .expect("legacy transport should accept after capacity opens");
        let accepted_id = match accepted {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant.work.id().to_string(),
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        assert_eq!(accepted_id, "observed-accepted");
        dispatch.await.unwrap();
        let snapshot = state.scheduled_work.observations().snapshot();
        assert_eq!(snapshot.total, 1);
        assert_eq!(snapshot.matched, 1);
        worker.settle_legacy_transport();
    }

    #[tokio::test]
    async fn legacy_scheduled_dispatch_honors_the_shared_pause_gate() {
        let (worker, worker_rx) = test_worker(0, 2);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![worker.clone()].into(),
        });
        let (scheduled_tx, scheduled_rx) = tokio::sync::mpsc::channel(2);
        let (owner_event_tx, owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        state.queue_pause.pause();
        scheduled_tx
            .send(crate::scheduler::ScheduledOwnerWork::new(
                "paused-utility",
                "utility",
                1,
                crate::gpu_pool::OwnerWork::Probe {
                    id: "paused-utility".to_string(),
                    kind: mold_scheduler::WorkKind::AdminModelLoad,
                    run: Box::new(|| {}),
                },
            ))
            .await
            .unwrap();

        let shutdown = tokio_util::sync::CancellationToken::new();
        let task = tokio::spawn(run_legacy_scheduled_work_dispatcher(
            scheduled_rx,
            owner_event_rx,
            state.clone(),
            shutdown.clone(),
        ));
        state.queue_pause.wait_until_blocked().await;
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));

        state.queue_pause.resume();
        let command = tokio::task::spawn_blocking(move || {
            worker_rx.recv_timeout(std::time::Duration::from_secs(1))
        })
        .await
        .unwrap()
        .expect("utility should dispatch only after resume");
        assert!(matches!(
            command,
            crate::gpu_pool::GpuWorkerCommand::Grant(_)
        ));
        worker.settle_legacy_transport();
        shutdown.cancel();
        drop(scheduled_tx);
        drop(owner_event_tx);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn legacy_post_upscale_moves_to_idle_sibling_with_same_frozen_artifact_and_f0_fallback() {
        let root = tempfile::tempdir().unwrap();
        let weights = root.path().join("upscaler.safetensors");
        std::fs::write(&weights, b"frozen-but-not-a-real-upscaler").unwrap();
        let cpu_plan = mold_inference::upscaler::resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &weights,
            None,
            mold_inference::upscaler::ExactUpscalePlacement::Cpu,
        )
        .unwrap();
        let frozen_artifact = cpu_plan.weights.clone();

        let (origin, _origin_rx) = test_worker(0, 1);
        let (sibling, sibling_rx) = test_worker(1, 1);
        origin.in_flight.store(1, Ordering::SeqCst);
        let mut state = empty_test_state(mold_core::Config::default());
        state.gpu_pool = Arc::new(GpuPool {
            workers: vec![origin.clone(), sibling.clone()].into(),
        });

        let output_dir = root.path().join("gallery");
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let (queue_tx, _queue_rx) = tokio::sync::mpsc::channel(1);
        let mut request = fake_request("flux-dev:q4");
        request.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let original = fake_image();
        let generation = crate::gpu_pool::GpuJob {
            id: "legacy-sibling-post-upscale".to_string(),
            durable_queue_rank: None,
            model: request.model.clone(),
            request,
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: Some(output_dir.clone()),
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            gallery_publication_gate: state.gallery_publication_gate.clone(),
            queue: QueueHandle::new(queue_tx),
            registry: state.job_registry.clone(),
            events: state.events.clone(),
            execution_plan: None,
            prepared_execution_inputs: None,
            #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
            h3_prepared_attempt: None,
            lease: None,
            journal: None,
        };
        let response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: Vec::new(),
            video: None,
            generation_time_ms: 1,
            model: generation.model.clone(),
            seed_used: 7,
            gpu: Some(origin.gpu.ordinal),
        };
        let work = crate::scheduler::ScheduledOwnerWork::new(
            "legacy-sibling-post-upscale",
            "real-esrgan-x4plus:fp16",
            1,
            crate::gpu_pool::OwnerWork::PostUpscale(Box::new(
                crate::gpu_pool::PostGenerationUpscaleJob {
                    id: "legacy-sibling-post-upscale".to_string(),
                    generation: Box::new(generation),
                    response,
                    image: original.clone(),
                    output_metadata: None,
                    cancellation: mold_inference::InferenceCancellationToken::default(),
                    execution_plan: None,
                },
            )),
        )
        .with_utility_plans(vec![
            crate::gpu_pool::UtilityExecutionPlan::Upscale(cpu_plan.clone()),
            crate::gpu_pool::UtilityExecutionPlan::Upscale(
                mold_inference::upscaler::resolve_upscale_execution_plan_from_artifact(
                    cpu_plan.model_name.clone(),
                    cpu_plan.weights.clone(),
                    cpu_plan.artifact_root.clone(),
                    mold_inference::upscaler::ExactUpscalePlacement::Device {
                        backend: origin.gpu.backend,
                        ordinal: origin.gpu.ordinal,
                    },
                ),
            ),
        ]);

        // Drift after the dependency boundary. Every sibling candidate must
        // retain the original artifact identity and fail closed at execution.
        std::fs::write(&weights, b"changed-after-freeze").unwrap();
        assert!(
            dispatch_legacy_scheduled_work(
                &state,
                work,
                &tokio_util::sync::CancellationToken::new(),
            )
            .await
        );
        let command = sibling_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("idle sibling must receive the post-upscale followup");
        let grant = match command {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => grant,
            crate::gpu_pool::GpuWorkerCommand::Drain => panic!("unexpected drain"),
            crate::gpu_pool::GpuWorkerCommand::Shutdown => panic!("unexpected shutdown"),
        };
        let selected = match &grant.work {
            crate::gpu_pool::OwnerWork::PostUpscale(job) => job.execution_plan.as_ref().unwrap(),
            _ => panic!("expected post-upscale work"),
        };
        assert_eq!(
            selected.placement,
            mold_inference::upscaler::ExactUpscalePlacement::Device {
                backend: sibling.gpu.backend,
                ordinal: sibling.gpu.ordinal,
            }
        );
        assert_eq!(selected.weights, frozen_artifact);
        assert!(
            selected.validate().is_err(),
            "artifact drift must invalidate"
        );

        let (owner_event_tx, _owner_event_rx) = tokio::sync::mpsc::unbounded_channel();
        let owner = crate::gpu_worker::spawn_legacy_gpu_thread(
            sibling.clone(),
            sibling_rx,
            owner_event_tx,
            std::time::Duration::from_secs(60),
        );
        sibling.try_send_job(grant).unwrap();
        let completed = tokio::time::timeout(std::time::Duration::from_secs(2), result_rx)
            .await
            .expect("post-upscale fallback must settle")
            .expect("result sender must settle")
            .expect("F0 keeps the original when exact upscale invalidates");
        assert_eq!(completed.image.data, original.data);
        assert_eq!(completed.image.width, original.width);
        assert!(
            std::fs::read_dir(&output_dir)
                .unwrap()
                .any(|entry| entry.unwrap().path().is_file()),
            "F0 must still publish the original to the gallery"
        );
        sibling.request_shutdown();
        owner.join().unwrap();
    }

    fn recv_worker_job(
        rx: &std::sync::mpsc::Receiver<crate::gpu_pool::GpuWorkerCommand>,
        timeout: std::time::Duration,
    ) -> Result<crate::gpu_pool::GpuJob, std::sync::mpsc::RecvTimeoutError> {
        match rx.recv_timeout(timeout)? {
            crate::gpu_pool::GpuWorkerCommand::Grant(grant) => {
                Ok(generation_from_legacy_grant(*grant))
            }
            crate::gpu_pool::GpuWorkerCommand::Shutdown => {
                panic!("unexpected worker shutdown command")
            }
            crate::gpu_pool::GpuWorkerCommand::Drain => {
                panic!("unexpected worker drain command")
            }
        }
    }

    fn empty_test_state(config: mold_core::Config) -> crate::state::AppState {
        crate::state::AppState::empty(
            config,
            QueueHandle::new(tokio::sync::mpsc::channel(1).0),
            crate::state::AppState::empty_gpu_pool(),
            200,
        )
    }

    #[test]
    fn save_image_to_dir_writes_file_and_creates_missing_dir() {
        let tmp = TempDir::new().unwrap();
        let nested = tmp.path().join("sub/output");
        assert!(!nested.exists());

        save_image_to_dir(
            &nested,
            &fake_image(),
            "flux-dev:q4",
            1,
            None,
            None,
            None,
            None,
        );

        assert!(nested.exists(), "save should mkdir -p");
        let entries: Vec<_> = std::fs::read_dir(&nested).unwrap().collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].as_ref().unwrap().file_name();
        let name_str = name.to_string_lossy();
        // Filename uses model-with-colon-replaced-by-dash + ms timestamp + .png.
        assert!(name_str.starts_with("mold-flux-dev-q4-"), "{name_str}");
        assert!(name_str.ends_with(".png"), "{name_str}");
    }

    #[test]
    fn ordinary_gallery_save_never_takes_a_batch_reserved_name() {
        let tmp = TempDir::new().unwrap();
        let reservations = tmp
            .path()
            .join(crate::batch_transaction::TRANSACTION_DIR)
            .join("reservations");
        std::fs::create_dir_all(&reservations).unwrap();
        std::fs::write(reservations.join("same.png.reserve"), b"reserved").unwrap();

        let (filename, path, _reservation) =
            write_gallery_bytes_no_replace(tmp.path(), "same.png", b"ordinary").unwrap();

        assert_eq!(filename, "same-1.png");
        assert_eq!(std::fs::read(path).unwrap(), b"ordinary");
        assert!(!tmp.path().join("same.png").exists());
    }

    /// Gallery bytes are staged under `<final>.partial` and published by
    /// rename, so a kill mid-write can never leave a truncated file at a real
    /// gallery name for `db.reconcile` to import as a valid print. Blocking
    /// the staging path is the cheapest way to prove the final name is only
    /// ever created by the publish step.
    #[test]
    fn ordinary_gallery_save_publishes_by_rename_and_never_writes_the_final_name_directly() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("ordinary.png.partial")).unwrap();

        let outcome =
            write_gallery_bytes_no_replace(tmp.path(), "ordinary.png", b"generated output");
        assert!(
            outcome.is_err(),
            "staging must fail when the partial path is unusable"
        );
        assert!(
            !tmp.path().join("ordinary.png").exists(),
            "a failed write must leave nothing at the gallery name"
        );
    }

    /// The helper is `no_replace` in its name and in its contract. Publishing
    /// by plain `rename` quietly broke that on Unix, where rename REPLACES the
    /// destination: an external gallery writer that created the reserved
    /// filename between reservation and publication would be overwritten
    /// instead of refused.
    #[test]
    fn ordinary_gallery_save_refuses_to_replace_a_name_that_appeared_underneath_it() {
        let tmp = TempDir::new().unwrap();
        let reservations = tmp
            .path()
            .join(crate::batch_transaction::TRANSACTION_DIR)
            .join("reservations");
        std::fs::create_dir_all(&reservations).unwrap();
        // Reserve the name so the helper picks it, then have somebody else
        // create the file after the reservation and before publication.
        let planted = tmp.path().join("ordinary.png");
        std::fs::write(&planted, b"written by someone else").unwrap();

        let outcome = write_gallery_bytes_no_replace(tmp.path(), "ordinary.png", b"ours");

        if let Ok((filename, _, _)) = outcome {
            assert_ne!(
                filename, "ordinary.png",
                "a taken name must never be published over"
            );
        }
        assert_eq!(
            std::fs::read(&planted).unwrap(),
            b"written by someone else",
            "the other writer's bytes must survive"
        );
    }

    /// A filesystem without hard-link support — exFAT, some SMB mounts — is a
    /// perfectly ordinary place to keep a gallery. Requiring links there made
    /// every save fail, discard its bytes, and hold the durable job forever,
    /// so the server rendered and then threw the result away. The fallback is
    /// what such a host runs, and it has to be correct on its own.
    #[test]
    fn the_link_free_publish_fallback_is_atomic_and_still_refuses_to_replace() {
        let tmp = TempDir::new().unwrap();
        let staged = tmp.path().join("out.png.partial");
        let published = tmp.path().join("out.png");
        std::fs::write(&staged, b"generated bytes").unwrap();

        publish_by_reserving_final_name(&staged, &published).expect("publishes without links");
        assert_eq!(std::fs::read(&published).unwrap(), b"generated bytes");
        assert!(!staged.exists());

        // A name somebody else already holds is refused, not overwritten.
        let other = tmp.path().join("taken.png.partial");
        std::fs::write(&other, b"ours").unwrap();
        std::fs::write(tmp.path().join("taken.png"), b"theirs").unwrap();
        let refused = publish_by_reserving_final_name(&other, &tmp.path().join("taken.png"))
            .expect_err("a taken name must be refused");
        assert_eq!(refused.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(tmp.path().join("taken.png")).unwrap(),
            b"theirs"
        );
    }

    /// The proof that matters for a link-less host: not that a fallback
    /// function exists, but that a real gallery save completes through it. A
    /// broken fallback would park every job on exactly the filesystems it was
    /// added for, so this drives the whole helper with the platform primitive
    /// forced off.
    #[test]
    fn a_filesystem_without_no_replace_rename_still_publishes_and_still_refuses() {
        let tmp = TempDir::new().unwrap();
        FORCE_PUBLISH_FALLBACK.store(true, Ordering::SeqCst);

        let saved = write_gallery_bytes_no_replace(tmp.path(), "ordinary.png", b"generated output");

        let published = match saved {
            Ok((filename, path, _reservation)) => {
                assert_eq!(filename, "ordinary.png");
                path
            }
            Err(error) => {
                FORCE_PUBLISH_FALLBACK.store(false, Ordering::SeqCst);
                panic!("a link-less filesystem must still publish: {error:#}");
            }
        };
        assert_eq!(
            std::fs::read(&published).unwrap(),
            b"generated output",
            "the whole file must land, not a placeholder"
        );
        assert!(
            !tmp.path().join("ordinary.png.partial").exists(),
            "staging must not be left behind"
        );

        // And the no-replace contract still holds on that path.
        std::fs::write(tmp.path().join("taken.png"), b"someone else's").unwrap();
        let refused = write_gallery_bytes_no_replace(tmp.path(), "taken.png", b"ours");
        FORCE_PUBLISH_FALLBACK.store(false, Ordering::SeqCst);
        if let Ok((filename, _, _)) = refused {
            assert_ne!(
                filename, "taken.png",
                "a taken name must not be published over"
            );
        }
        assert_eq!(
            std::fs::read(tmp.path().join("taken.png")).unwrap(),
            b"someone else's"
        );
    }

    /// Whatever primitive the host supports, the contract is the same.
    #[test]
    fn publishing_refuses_an_existing_destination_on_this_host() {
        let tmp = TempDir::new().unwrap();
        let staged = tmp.path().join("out.png.partial");
        std::fs::write(&staged, b"ours").unwrap();
        let published = tmp.path().join("out.png");
        std::fs::write(&published, b"theirs").unwrap();

        let refused = publish_staged_no_replace(&staged, &published).expect_err("must not replace");
        assert_eq!(refused.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&published).unwrap(), b"theirs");
    }

    /// A kill mid-write leaves `<final>.partial` behind, and the bounded
    /// shutdown deadline makes that a routine event rather than a rare one.
    /// Nothing else reclaims them and later generations take fresh timestamped
    /// names, so repeated interruptions would consume gallery disk forever.
    #[test]
    fn stale_gallery_partials_are_reclaimed_at_startup() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("mold-1.png.partial"), b"half a print").unwrap();
        std::fs::write(tmp.path().join("mold-2.mp4.partial"), b"half a clip").unwrap();
        // Real prints and unrelated files are not partials.
        std::fs::write(tmp.path().join("mold-3.png"), b"a real print").unwrap();
        std::fs::write(tmp.path().join("notes.txt"), b"not ours").unwrap();

        assert_eq!(sweep_stale_gallery_partials(tmp.path()), 2);

        assert!(!tmp.path().join("mold-1.png.partial").exists());
        assert!(!tmp.path().join("mold-2.mp4.partial").exists());
        assert!(tmp.path().join("mold-3.png").is_file());
        assert!(tmp.path().join("notes.txt").is_file());
        assert_eq!(
            sweep_stale_gallery_partials(tmp.path()),
            0,
            "sweeping is idempotent"
        );
    }

    #[test]
    fn ordinary_gallery_save_leaves_no_staging_file_behind() {
        let tmp = TempDir::new().unwrap();
        let (filename, path, _reservation) =
            write_gallery_bytes_no_replace(tmp.path(), "ordinary.png", b"generated output")
                .unwrap();

        assert_eq!(filename, "ordinary.png");
        assert_eq!(std::fs::read(path).unwrap(), b"generated output");
        assert!(!tmp.path().join("ordinary.png.partial").exists());
    }

    #[test]
    fn ordinary_gallery_save_keeps_output_when_directory_sync_is_unsupported() {
        let tmp = TempDir::new().unwrap();
        let sync_attempts = AtomicUsize::new(0);
        let unsupported_sync = |path: &std::path::Path| {
            sync_attempts.fetch_add(1, Ordering::SeqCst);
            crate::batch_transaction::tolerate_unsupported_ordinary_directory_sync(
                path,
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "injected unsupported directory fsync",
                )
                .into()),
            )
        };

        let (filename, path, _reservation) = write_gallery_bytes_no_replace_with_directory_sync(
            tmp.path(),
            "ordinary.png",
            b"generated output",
            &unsupported_sync,
        )
        .unwrap();

        assert_eq!(filename, "ordinary.png");
        assert_eq!(std::fs::read(path).unwrap(), b"generated output");
        assert_eq!(
            sync_attempts.load(Ordering::SeqCst),
            2,
            "reservation and gallery directories both use the explicit best-effort policy"
        );
    }

    #[test]
    fn save_image_to_dir_includes_batch_index_when_batch_size_gt_1() {
        let tmp = TempDir::new().unwrap();
        let mut img = fake_image();
        img.index = 3;
        img.format = OutputFormat::Jpeg;
        img.data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic

        save_image_to_dir(tmp.path(), &img, "sdxl", 4, None, None, None, None);

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
            .collect();
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(
            name.contains("-3.jpeg"),
            "expected batch index suffix: {name}"
        );
    }

    #[test]
    fn titled_output_filename_places_slug_last_in_the_stem() {
        assert_eq!(
            titled_output_filename("flux-dev:q4", 1_700_000_000_000, "png", 1, 0, None, None),
            mold_core::default_output_filename("flux-dev:q4", 1_700_000_000_000, "png", 1, 0),
            "untitled prints keep the byte-identical legacy name"
        );
        assert_eq!(
            titled_output_filename(
                "flux-dev:q4",
                1_700_000_000_000,
                "png",
                1,
                0,
                None,
                Some("smurf-04")
            ),
            "mold-flux-dev-q4-1700000000000~smurf-04.png"
        );
        assert_eq!(
            titled_output_filename(
                "flux-dev:q4",
                1_700_000_000_000,
                "png",
                4,
                2,
                Some("upscaled"),
                Some("smurf-04")
            ),
            "mold-flux-dev-q4-1700000000000-2-upscaled~smurf-04.png",
            "suffix variants carry the slug after the suffix"
        );
        // The stem round-trips through the synthesizer: the slug is cut
        // before the model is recovered.
        let synthesized = mold_db::metadata_io::synthesize_from_filename(
            "mold-flux-dev-q4-1700000000000~smurf-04.png",
            0,
        );
        assert_eq!(synthesized.model, "flux-dev-q4");
    }

    #[test]
    fn save_image_to_dir_folds_title_into_filename_and_seeds_row_title() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut req = fake_request("flux-dev:q4");
        req.title = Some("Smurf 04!".to_string());
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        assert_eq!(meta.title.as_deref(), Some("Smurf 04!"));

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            None,
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        let rec = &rows[0];
        assert!(
            rec.filename.ends_with("~smurf-04.png"),
            "titled filename carries the slug: {}",
            rec.filename
        );
        assert!(tmp.path().join(&rec.filename).is_file());
        assert_eq!(rec.title.as_deref(), Some("Smurf 04!"), "row title seeded");
        assert_eq!(rec.metadata.title.as_deref(), Some("Smurf 04!"));
        let image = rec.to_gallery_image();
        assert_eq!(image.title.as_deref(), Some("Smurf 04!"));
    }

    #[test]
    fn save_image_to_dir_upserts_metadata_row_when_db_provided() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("flux-dev:q4");
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            None,
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1, "exactly one DB row for the saved file");
        let rec = &rows[0];
        assert_eq!(rec.metadata.prompt, "a cat");
        assert_eq!(rec.metadata.seed, 42);
        assert_eq!(rec.metadata.version, "test-version");
        assert_eq!(rec.format, OutputFormat::Png);
        assert_eq!(rec.generation_time_ms, Some(1234));
        // stat_from_disk should have populated the size from the actual file.
        assert!(rec.file_size_bytes.unwrap_or(0) > 0);
    }

    /// A print that arrived filed lands tagged and in its collection, and
    /// the `gallery_added` row carries both so a client can insert in place
    /// without refetching.
    #[test]
    fn publication_applies_creation_time_filing_and_announces_it() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        let mut req = fake_request("flux-dev:q4");
        req.title = Some("Smurf Village".into());
        req.tags = Some(vec!["smurfs".into(), "village".into()]);
        req.collection = Some(mold_core::CollectionRef::by_name("Sequences"));
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            Some(&events),
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        let filename = rows[0].filename.clone();
        // The title still folds into the filename, unchanged by filing.
        assert!(filename.contains("~smurf-village"), "{filename}");

        let org = db
            .print_organization(tmp.path(), &filename)
            .unwrap()
            .unwrap();
        assert_eq!(org.tags, vec!["smurfs".to_string(), "village".to_string()]);
        let collections = db.list_collections().unwrap();
        assert_eq!(collections.len(), 1);
        assert_eq!(collections[0].name, "Sequences");

        // gallery_added carries the filing, then gallery_updated repeats the
        // organized row and gallery_collections_changed follows the join.
        let added = rx.try_recv().expect("gallery_added");
        match added {
            mold_core::ServerEvent::GalleryAdded {
                image: Some(image), ..
            } => {
                assert_eq!(
                    image.tags,
                    vec!["smurfs".to_string(), "village".to_string()]
                );
                assert_eq!(image.collections, vec![collections[0].id.clone()]);
                assert_eq!(image.title.as_deref(), Some("Smurf Village"));
            }
            other => panic!("expected gallery_added with a row, got {other:?}"),
        }
        assert!(matches!(
            rx.try_recv().expect("gallery_updated"),
            mold_core::ServerEvent::GalleryUpdated { image: Some(_), .. }
        ));
        assert!(matches!(
            rx.try_recv().expect("gallery_collections_changed"),
            mold_core::ServerEvent::GalleryCollectionsChanged {}
        ));
    }

    /// An unfiled print — the overwhelmingly common case — emits exactly the
    /// one event it always did. The organization events must not become
    /// per-print noise.
    #[test]
    fn an_unfiled_publication_emits_only_gallery_added() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        let req = fake_request("flux-dev:q4");
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            None,
            Some(&db),
            Some(&events),
        );

        assert!(matches!(
            rx.try_recv().expect("gallery_added"),
            mold_core::ServerEvent::GalleryAdded { .. }
        ));
        assert!(
            rx.try_recv().is_err(),
            "an unfiled print must emit no organization events"
        );
        assert!(db.list_tags().unwrap().is_empty());
        assert!(db.list_collections().unwrap().is_empty());
    }

    #[test]
    fn save_generated_image_outputs_persists_original_and_upscaled_dimensions() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        let original = fake_image();
        let mut upscaled = fake_image();
        upscaled.width = 2048;
        upscaled.height = 2048;
        upscaled.data = vec![4, 5, 6];
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();

        save_generated_image_outputs(
            tmp.path(),
            Some(&original),
            &upscaled,
            "flux-dev:q4",
            1,
            &meta,
            Some(1234),
            Some(&db),
            None,
            &gallery_gate,
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 2);
        let original_row = rows
            .iter()
            .find(|row| row.filename.contains("-original."))
            .expect("original row");
        let upscaled_row = rows
            .iter()
            .find(|row| row.filename.contains("-upscaled."))
            .expect("upscaled row");
        assert_eq!(
            (original_row.metadata.width, original_row.metadata.height),
            (512, 512)
        );
        assert_eq!(
            (upscaled_row.metadata.width, upscaled_row.metadata.height),
            (2048, 2048)
        );
        assert_eq!(upscaled_row.metadata.generation_width, Some(512));
        assert_eq!(upscaled_row.metadata.generation_height, Some(512));
    }

    #[test]
    fn save_image_to_dir_skips_db_when_metadata_is_none() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            None, // ← metadata absent
            Some(1234),
            Some(&db),
            None,
        );

        // File still on disk, but no DB row recorded — both gates must hold
        // for the upsert to fire.
        assert_eq!(
            std::fs::read_dir(tmp.path())
                .unwrap()
                .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
                .count(),
            1
        );
        assert_eq!(db.list(None).unwrap().len(), 0);
    }

    #[test]
    fn save_image_to_dir_invalid_path_does_not_panic() {
        // /dev/null is a file, not a directory — create_dir_all should fail
        // and the helper must log + return cleanly rather than panic.
        save_image_to_dir(
            std::path::Path::new("/dev/null/cant-mkdir-here"),
            &fake_image(),
            "test",
            1,
            None,
            None,
            None,
            None,
        );
    }

    #[test]
    fn save_image_to_dir_emits_gallery_added_with_row_when_db_records() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("flux-dev:q4");
        let meta = OutputMetadata::from_generate_request(&req, 42, None, "test-version");
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            Some(&meta),
            Some(1234),
            Some(&db),
            Some(&events),
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { filename, image } => {
                assert!(filename.ends_with(".png"), "{filename}");
                let img = image.expect("DB recorded — event must carry the gallery row");
                assert_eq!(img.filename, filename);
                assert_eq!(img.metadata.prompt, "a cat");
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_image_to_dir_emits_gallery_added_without_row_when_db_absent() {
        let tmp = TempDir::new().unwrap();
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            tmp.path(),
            &fake_image(),
            "flux-dev:q4",
            1,
            None,
            None,
            None, // no DB
            Some(&events),
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { image, .. } => {
                assert!(image.is_none(), "no DB → clients must refetch");
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_image_to_dir_emits_nothing_on_write_failure() {
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();

        save_image_to_dir(
            std::path::Path::new("/dev/null/cant-mkdir-here"),
            &fake_image(),
            "test",
            1,
            None,
            None,
            None,
            Some(&events),
        );

        assert!(
            rx.try_recv().is_err(),
            "failed save must not announce a gallery entry"
        );
    }

    #[test]
    fn save_video_to_dir_emits_gallery_added() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");
        let events = crate::events::EventBroadcaster::new();
        let mut rx = events.subscribe();
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();

        save_video_to_dir(
            tmp.path(),
            b"fake mp4 bytes",
            b"",
            OutputFormat::Mp4,
            "ltx-video:fp16",
            &meta,
            Some(5000),
            Some(&db),
            Some(&events),
            &gallery_gate,
        );

        match rx.try_recv().unwrap() {
            mold_core::ServerEvent::GalleryAdded { filename, image } => {
                assert!(filename.ends_with(".mp4"), "{filename}");
                assert!(image.is_some());
            }
            other => panic!("expected gallery_added, got {other:?}"),
        }
    }

    #[test]
    fn save_video_to_dir_writes_mp4_and_records_metadata() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut req = fake_request("ltx-video:fp16");
        req.frames = Some(25);
        req.fps = Some(24);
        req.keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 24,
            image: b"private closing-frame bytes".to_vec(),
            name: Some("closing-frame.png".to_string()),
        }]);
        let meta = OutputMetadata::from_generate_request(&req, 99, None, "test-version");

        // Minimal MP4-ish bytes: an `ftyp` box header. The helper writes
        // bytes verbatim — content validation happens at gallery scan time.
        let bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom".to_vec();
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();

        save_video_to_dir(
            tmp.path(),
            &bytes,
            b"",
            OutputFormat::Mp4,
            "ltx-video:fp16",
            &meta,
            Some(5000),
            Some(&db),
            None,
            &gallery_gate,
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
            .collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(name.starts_with("mold-ltx-video-fp16-"), "{name}");
        assert!(name.ends_with(".mp4"), "{name}");

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].format, OutputFormat::Mp4);
        assert_eq!(rows[0].metadata.frames, Some(25));
        assert_eq!(rows[0].metadata.fps, Some(24));
        assert_eq!(rows[0].metadata.keyframes, meta.keyframes);
        assert_eq!(rows[0].metadata.keyframes.as_ref().unwrap()[0].frame, 24);
        assert_eq!(
            rows[0].metadata.keyframes.as_ref().unwrap()[0]
                .name
                .as_deref(),
            Some("closing-frame.png")
        );
        assert_eq!(rows[0].generation_time_ms, Some(5000));
        let archived = gallery_gate.committed_archive_index(tmp.path()).unwrap();
        assert_eq!(
            archived.get(&name).unwrap().record().metadata.frames,
            Some(25),
            "DB-enabled publications must retain archive authority for a later DB-disabled restart"
        );
    }

    #[test]
    fn synthetic_h3_video_publishes_synchronized_sse_and_gallery_metadata() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let mut request = fake_request(mold_core::minimax_h3::FL2VA_COMFY);
        request.width = 1344;
        request.height = 768;
        request.frames = Some(124);
        request.fps = Some(24);
        request.output_format = Some(OutputFormat::Mp4);
        request.enable_audio = Some(true);

        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: b"synthetic-h3-mp4-with-synchronized-audio".to_vec(),
            format: OutputFormat::Mp4,
            width: 1344,
            height: 768,
            frames: 124,
            fps: 24,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: b"synthetic-h3-thumbnail".to_vec(),
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(5_167),
            audio_sample_rate: Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ),
            audio_channels: Some(mold_core::minimax_h3::AUDIO_CHANNELS),
        };
        let response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            images: Vec::new(),
            video: Some(video.clone()),
            audio: None,
            generation_time_ms: 12_345,
            model: mold_core::minimax_h3::FL2VA_COMFY.to_string(),
            seed_used: 42,
            gpu: Some(0),
        };
        let mut metadata =
            OutputMetadata::from_generate_request(&request, response.seed_used, None, "test");
        metadata.apply_video_output(&video);
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();
        let filename = save_video_to_dir(
            tmp.path(),
            &video.data,
            &video.gif_preview,
            video.format,
            &request.model,
            &metadata,
            Some(response.generation_time_ms as i64),
            Some(&db),
            None,
            &gallery_gate,
        )
        .expect("synthetic publication should allocate one gallery filename");
        assert_eq!(
            std::fs::read(tmp.path().join(&filename)).unwrap(),
            video.data
        );

        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].metadata.model, mold_core::minimax_h3::FL2VA_COMFY);
        assert_eq!(rows[0].metadata.seed, 42);
        assert_eq!(rows[0].metadata.frames, Some(124));
        assert_eq!(rows[0].metadata.fps, Some(24));
        assert_eq!(rows[0].metadata.enable_audio, Some(true));

        let thumbnail = ImageData {
            data: video.thumbnail.clone(),
            format: OutputFormat::Png,
            width: video.width,
            height: video.height,
            index: 0,
        };
        let message = build_sse_completion_message(
            &response,
            &thumbnail,
            None,
            Some(&metadata),
            &SavedOutputNames {
                output: Some(filename.clone()),
                original: None,
            },
            SseCompletionPayload::MetadataOnly,
        );
        let SseMessage::Complete(event) = message else {
            panic!("saved synthetic H3 video must complete over SSE")
        };
        assert_eq!(event.filename.as_deref(), Some(filename.as_str()));
        assert_eq!(event.model, mold_core::minimax_h3::FL2VA_COMFY);
        assert_eq!(event.format, OutputFormat::Mp4);
        assert_eq!(event.video_frames, Some(124));
        assert_eq!(event.video_fps, Some(24));
        assert!(event.video_has_audio);
        assert_eq!(
            event.video_audio_sample_rate,
            Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ)
        );
        assert_eq!(
            event.video_audio_channels,
            Some(mold_core::minimax_h3::AUDIO_CHANNELS)
        );
        assert_eq!(event.metadata.as_ref().map(|value| value.seed), Some(42));
        assert!(
            event.image.is_empty(),
            "metadata-only SSE must not duplicate MP4 bytes"
        );
    }

    #[test]
    fn synthetic_h3_ref2va_publication_preserves_ordered_redacted_provenance() {
        use mold_core::{
            GenerationReference, GenerationReferenceAuthority, GenerationReferenceKind,
            GenerationReferenceProvenance,
        };

        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let provenance = |name: &str, byte: u8| GenerationReferenceProvenance {
            name: Some(name.to_string()),
            sha256: Some(format!("{byte:02x}").repeat(32)),
            crop: None,
        };
        let mut request = fake_request(mold_core::minimax_h3::REF2VA_COMFY);
        request.width = 1344;
        request.height = 768;
        request.frames = Some(124);
        request.fps = Some(24);
        request.output_format = Some(OutputFormat::Mp4);
        request.enable_audio = Some(true);
        request.guidance = 0.0;
        request.strength = 1.0;
        request.references = Some(vec![
            GenerationReference::Video {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("motion.mp4", 1),
                mime_type: "video/mp4".to_string(),
                width: 1280,
                height: 720,
                frame_count: Some(48),
                duration_ms: 2_000,
                fps: 24.0,
                has_audio: true,
                audio_duration_ms: Some(2_000),
                audio_sample_count: Some(96_000),
                audio_sample_rate: Some(48_000),
                audio_channels: Some(2),
            },
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("portrait.png", 2),
                mime_type: "image/png".to_string(),
                width: 1024,
                height: 768,
            },
            GenerationReference::Audio {
                media: GenerationReferenceAuthority::Descriptor,
                provenance: provenance("voice.wav", 3),
                mime_type: "audio/wav".to_string(),
                duration_ms: 2_000,
                sample_rate: 48_000,
                channels: 1,
                sample_count: Some(96_000),
            },
        ]);
        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: b"synthetic-ref2va-mp4-with-synchronized-audio".to_vec(),
            format: OutputFormat::Mp4,
            width: request.width,
            height: request.height,
            frames: request.frames.unwrap(),
            fps: request.fps.unwrap(),
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: b"synthetic-ref2va-thumbnail".to_vec(),
            gif_preview: Vec::new(),
            has_audio: true,
            duration_ms: Some(5_167),
            audio_sample_rate: Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ),
            audio_channels: Some(mold_core::minimax_h3::AUDIO_CHANNELS),
        };
        let response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            images: Vec::new(),
            video: Some(video.clone()),
            audio: None,
            generation_time_ms: 12_345,
            model: request.model.clone(),
            seed_used: 42,
            gpu: Some(0),
        };
        let mut metadata = OutputMetadata::from_generate_request(&request, 42, None, "test");
        metadata.apply_video_output(&video);
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();
        let filename = save_video_to_dir(
            tmp.path(),
            &video.data,
            &video.gif_preview,
            video.format,
            &request.model,
            &metadata,
            Some(response.generation_time_ms as i64),
            Some(&db),
            None,
            &gallery_gate,
        )
        .unwrap();

        let rows = db.list(Some(tmp.path())).unwrap();
        let references = rows[0].metadata.references.as_deref().unwrap();
        assert_eq!(
            references.iter().map(|item| item.kind).collect::<Vec<_>>(),
            [
                GenerationReferenceKind::Video,
                GenerationReferenceKind::Image,
                GenerationReferenceKind::Audio,
            ]
        );
        assert_eq!(references[0].index, 1);
        assert!(references[0].has_audio);
        assert_eq!(references[0].audio_sample_rate, Some(48_000));
        assert_eq!(references[2].index, 3);
        let saved_fingerprint = mold_core::generation_reference_fingerprint(references);
        let mut swapped_request = request.clone();
        swapped_request
            .references
            .as_mut()
            .expect("ordered references")
            .swap(0, 1);
        let swapped_metadata =
            OutputMetadata::from_generate_request(&swapped_request, 42, None, "test");
        let swapped_fingerprint = mold_core::generation_reference_fingerprint(
            swapped_metadata
                .references
                .as_deref()
                .expect("swapped ordered metadata"),
        );
        assert_ne!(saved_fingerprint, swapped_fingerprint);
        let durable_json = serde_json::to_string(&rows[0].metadata).unwrap();
        for secret in ["authority", "handle", "server_path", "/synthetic/"] {
            assert!(!durable_json.contains(secret));
        }

        let thumbnail = ImageData {
            data: video.thumbnail.clone(),
            format: OutputFormat::Png,
            width: video.width,
            height: video.height,
            index: 0,
        };
        let SseMessage::Complete(event) = build_sse_completion_message(
            &response,
            &thumbnail,
            None,
            Some(&metadata),
            &SavedOutputNames {
                output: Some(filename),
                original: None,
            },
            SseCompletionPayload::MetadataOnly,
        ) else {
            panic!("saved synthetic Ref2VA output must complete over SSE")
        };
        assert_eq!(
            event.video_audio_sample_rate,
            Some(mold_core::minimax_h3::AUDIO_SAMPLE_RATE_HZ)
        );
        assert_eq!(
            event.video_audio_channels,
            Some(mold_core::minimax_h3::AUDIO_CHANNELS)
        );
        assert_eq!(
            event
                .metadata
                .as_ref()
                .and_then(|value| value.references.as_ref())
                .map(|items| items.iter().map(|item| item.index).collect::<Vec<_>>()),
            Some(vec![1, 2, 3])
        );
        assert!(event.image.is_empty());
    }

    #[test]
    fn named_video_replay_keeps_one_gallery_file_and_row() {
        let tmp = TempDir::new().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 99, None, "test-version");
        let filename = "chain-01TEST-take-1.mp4";
        let bytes = b"stable chain bytes";
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();

        for _ in 0..2 {
            assert_eq!(
                save_video_to_dir_named(
                    tmp.path(),
                    filename,
                    bytes,
                    OutputFormat::Mp4,
                    &meta,
                    None,
                    Some(&db),
                    None,
                    &gallery_gate,
                )
                .unwrap(),
                filename
            );
        }

        assert_eq!(
            std::fs::read_dir(tmp.path())
                .unwrap()
                .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
                .count(),
            1
        );
        let rows = db.list(Some(tmp.path())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].filename, filename);
        assert_eq!(std::fs::read(tmp.path().join(filename)).unwrap(), bytes);
    }

    #[test]
    fn save_video_to_dir_without_db_still_writes_file() {
        let tmp = TempDir::new().unwrap();
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();

        save_video_to_dir(
            tmp.path(),
            b"fake gif bytes",
            b"",
            OutputFormat::Gif,
            "ltx-video:fp16",
            &meta,
            None,
            None,
            None,
            &gallery_gate,
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter(|entry| entry.as_ref().is_ok_and(|entry| entry.path().is_file()))
            .collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_string_lossy()
            .to_string();
        assert!(name.ends_with(".gif"), "{name}");
        let archived = gallery_gate.committed_archive_index(tmp.path()).unwrap();
        assert_eq!(archived.get(&name).unwrap().record().metadata, meta);

        let restarted_gate = crate::batch_transaction::GalleryPublicationGate::default();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(crate::batch_transaction::recover_transactions(
                tmp.path(),
                &restarted_gate,
                Arc::new(None),
            ))
            .unwrap();
        assert_eq!(
            restarted_gate
                .committed_archive_index(tmp.path())
                .unwrap()
                .get(&name)
                .unwrap()
                .record()
                .metadata,
            meta
        );
    }

    #[test]
    fn save_video_to_dir_invalid_path_does_not_panic() {
        let req = fake_request("ltx-video:fp16");
        let meta = OutputMetadata::from_generate_request(&req, 1, None, "v");
        let gallery_gate = crate::batch_transaction::GalleryPublicationGate::default();
        save_video_to_dir(
            std::path::Path::new("/dev/null/nope"),
            b"x",
            b"",
            OutputFormat::Mp4,
            "test",
            &meta,
            None,
            None,
            None,
            &gallery_gate,
        );
    }

    /// `save_video_preview_gif_to` must write to
    /// `<preview_dir>/<filename>.preview.gif` — the exact location
    /// `GET /api/gallery/preview/:filename` streams from. Without this
    /// sidecar the preview endpoint would 404 on every real generation
    /// and the TUI detail pane would only ever see the PNG thumbnail
    /// fallback.
    #[test]
    fn save_video_preview_gif_writes_to_preview_cache() {
        let td = tempfile::tempdir().unwrap();
        let preview_dir = td.path().join("cache").join("previews");

        const GIF: &[u8] = b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x3b";
        save_video_preview_gif_to(&preview_dir, "ltx2-42.mp4", GIF);

        let expected = preview_dir.join("ltx2-42.mp4.preview.gif");
        assert!(
            expected.is_file(),
            "preview gif should land at {}",
            expected.display()
        );
        assert_eq!(std::fs::read(&expected).unwrap(), GIF);
    }

    fn fake_audio(sample_rate: u32) -> mold_core::AudioData {
        mold_core::AudioData {
            data: b"RIFF\x00\x00\x00\x00WAVEfmt ".to_vec(),
            format: OutputFormat::Wav,
            sample_rate,
            channels: 2,
            duration_ms: 5_040,
            thumbnail: vec![0x89, 0x50, 0x4E, 0x47],
            thumbnail_width: 640,
            thumbnail_height: 360,
        }
    }

    /// An SSE render has no response headers, so `x-mold-request-warning` —
    /// the JSON path's whole delivery mechanism for advisories — reaches
    /// nobody streaming. The identity extraction's "several faces, largest
    /// used" is decided during admission and is exactly the kind of thing the
    /// person who supplied the photograph needs, so the complete event carries
    /// it too (#1223).
    #[test]
    fn build_sse_complete_event_carries_the_renders_request_warnings() {
        let identity =
            "3 faces were detected in the identity image; conditioning on the largest one";
        let image = ImageData {
            data: vec![0x89, 0x50, 0x4E, 0x47],
            format: OutputFormat::Png,
            width: 64,
            height: 64,
            index: 0,
        };
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: vec![identity.to_string()],
            audio: None,
            images: vec![image.clone()],
            video: None,
            generation_time_ms: 1,
            model: "flux-dev:q4".to_string(),
            seed_used: 7,
            gpu: Some(0),
        };

        let event = build_sse_complete_event(
            &resp,
            &image,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert_eq!(event.request_warnings, vec![identity.to_string()]);

        // A metadata-only payload drops the media, never the advisory: a
        // client that asked for no bytes still asked for the truth.
        let lean = build_sse_complete_event(
            &resp,
            &image,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::MetadataOnly,
        );
        assert_eq!(lean.request_warnings, vec![identity.to_string()]);

        // And an ordinary render says nothing, so the field serializes away.
        let mut quiet = resp.clone();
        quiet.request_warnings.clear();
        let quiet = build_sse_complete_event(
            &quiet,
            &image,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert!(quiet.request_warnings.is_empty());
        assert!(!serde_json::to_string(&quiet)
            .unwrap()
            .contains("request_warnings"));
    }

    /// An audio-only response must arrive as audio, not as a degraded image:
    /// the payload is the WAV bytes, the format says `wav`, the waveform is a
    /// separate field, and every `video_*` field stays empty so no client
    /// tries to seek frames in it.
    #[test]
    fn build_sse_complete_event_audio_carries_wav_payload_and_no_video_fields() {
        let audio = fake_audio(48_000);
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: Some(audio.clone()),
            images: vec![],
            video: None,
            generation_time_ms: 4321,
            model: "ltx-2.3-22b-dev:fp8".to_string(),
            seed_used: 11,
            gpu: Some(1),
        };
        let waveform_img = ImageData {
            data: audio.thumbnail.clone(),
            format: OutputFormat::Png,
            width: audio.thumbnail_width,
            height: audio.thumbnail_height,
            index: 0,
        };

        let event = build_sse_complete_event(
            &resp,
            &waveform_img,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );

        let b64 = base64::engine::general_purpose::STANDARD;
        assert_eq!(event.image, b64.encode(&audio.data));
        assert_eq!(event.format, OutputFormat::Wav);
        assert_eq!(event.audio_sample_rate, Some(48_000));
        assert_eq!(event.audio_channels, Some(2));
        assert_eq!(event.audio_duration_ms, Some(5_040));
        assert_eq!(event.audio_thumbnail, Some(b64.encode(&audio.thumbnail)));
        assert_eq!(event.width, 640);
        assert_eq!(event.height, 360);
        assert_eq!(event.video_frames, None);
        assert_eq!(event.video_fps, None);
        assert_eq!(event.video_thumbnail, None);
        assert!(!event.video_has_audio);
        assert_eq!(event.video_duration_ms, None);
        assert_eq!(event.gpu, Some(1));
    }

    #[test]
    fn build_sse_complete_event_audio_omits_media_for_metadata_only_payloads() {
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: Some(fake_audio(24_000)),
            images: vec![],
            video: None,
            generation_time_ms: 1,
            model: "ltx-2-19b-dev:fp8".to_string(),
            seed_used: 2,
            gpu: None,
        };
        let waveform_img = ImageData {
            data: vec![],
            format: OutputFormat::Png,
            width: 640,
            height: 360,
            index: 0,
        };
        let event = build_sse_complete_event(
            &resp,
            &waveform_img,
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::MetadataOnly,
        );
        assert!(event.image.is_empty());
        assert_eq!(event.audio_thumbnail, None);
        // Shape metadata still travels — only the bytes are withheld.
        assert_eq!(event.audio_sample_rate, Some(24_000));
    }

    /// `.wav` has no raster frame, so neither the server's on-demand
    /// thumbnailer nor the TUI's `image::open` can build a tile. The waveform
    /// PNG has to land in the cache at save time or the gallery shows a
    /// placeholder forever.
    #[test]
    fn save_audio_waveform_thumbnail_writes_both_cache_names() {
        let td = TempDir::new().unwrap();
        let thumb_dir = td.path().join("cache").join("thumbnails");
        const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";

        save_audio_waveform_thumbnail_to(&thumb_dir, "mold-ltx2-42.wav", PNG);

        // Server route naming.
        assert_eq!(
            std::fs::read(thumb_dir.join("mold-ltx2-42.wav.png")).unwrap(),
            PNG
        );
        // TUI cache naming.
        assert_eq!(
            std::fs::read(thumb_dir.join("mold-ltx2-42.wav.thumb.png")).unwrap(),
            PNG
        );
    }

    #[test]
    fn build_sse_complete_event_video_carries_mp4_payload_and_metadata() {
        // Regression guard for the multi-GPU bug: if `response.video` is set,
        // the SSE complete event must encode the actual video bytes and
        // populate every `video_*` field so the client can reconstruct a
        // `VideoData`. Before the shared helper, `gpu_worker.rs` encoded the
        // thumbnail PNG and hard-coded every `video_*` field to `None`,
        // silently degrading every LTX-Video / LTX-2 response to an image.
        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: vec![0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p'],
            format: OutputFormat::Mp4,
            width: 768,
            height: 512,
            frames: 25,
            fps: 24,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: vec![0x89, 0x50, 0x4E, 0x47],
            gif_preview: vec![b'G', b'I', b'F', b'8'],
            has_audio: true,
            duration_ms: Some(1040),
            audio_sample_rate: Some(44100),
            audio_channels: Some(2),
        };
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(video.clone()),
            generation_time_ms: 1234,
            model: "ltx-2-19b-distilled:fp8".to_string(),
            seed_used: 7,
            gpu: Some(0),
        };
        // The `img` the caller synthesizes from the video thumbnail — must be
        // ignored for the video branch.
        let thumb_img = ImageData {
            data: video.thumbnail.clone(),
            format: OutputFormat::Png,
            width: video.width,
            height: video.height,
            index: 0,
        };
        let mut req = fake_request("minimax-h3-ref2va:bf16");
        req.frames = Some(124);
        req.keyframes = Some(vec![mold_core::KeyframeCondition {
            frame: 123,
            image: b"private closing-frame bytes".to_vec(),
            name: Some("closing-frame.png".to_string()),
        }]);
        let metadata =
            OutputMetadata::from_generate_request(&req, resp.seed_used, None, "test-version");

        let event = build_sse_complete_event(
            &resp,
            &thumb_img,
            None,
            Some(&metadata),
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );

        let b64 = base64::engine::general_purpose::STANDARD;
        assert_eq!(event.image, b64.encode(&video.data));
        assert_eq!(event.format, OutputFormat::Mp4);
        assert_eq!(event.video_frames, Some(25));
        assert_eq!(event.video_fps, Some(24));
        assert_eq!(event.video_thumbnail, Some(b64.encode(&video.thumbnail)));
        assert_eq!(
            event.video_gif_preview,
            Some(b64.encode(&video.gif_preview))
        );
        assert!(event.video_has_audio);
        assert_eq!(event.video_duration_ms, Some(1040));
        assert_eq!(event.gpu, Some(0));
        let event_metadata = event.metadata.expect("metadata rides the complete event");
        assert_eq!(event_metadata.keyframes, metadata.keyframes);
        assert_eq!(event_metadata.keyframes.as_ref().unwrap()[0].frame, 123);
        assert_eq!(
            event_metadata.keyframes.as_ref().unwrap()[0]
                .name
                .as_deref(),
            Some("closing-frame.png")
        );

        let saved = SavedOutputNames {
            output: Some("generated-video.mp4".to_string()),
            original: None,
        };
        let metadata_only = build_sse_complete_event(
            &resp,
            &thumb_img,
            None,
            Some(&metadata),
            &saved,
            SseCompletionPayload::MetadataOnly,
        );
        assert!(metadata_only.image.is_empty());
        assert!(metadata_only.video_thumbnail.is_none());
        assert!(metadata_only.video_gif_preview.is_none());
        assert_eq!(metadata_only.video_frames, Some(25));
        assert_eq!(
            metadata_only.filename.as_deref(),
            Some("generated-video.mp4")
        );
        assert_eq!(
            metadata_only.metadata.unwrap().keyframes,
            metadata.keyframes
        );
    }

    #[test]
    fn build_sse_complete_event_video_empty_gif_preview_omits_field() {
        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: vec![0x00, 0x00, 0x00, 0x18],
            format: OutputFormat::Mp4,
            width: 256,
            height: 256,
            frames: 17,
            fps: 12,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: vec![0x89, 0x50],
            gif_preview: Vec::new(),
            has_audio: false,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(video),
            generation_time_ms: 0,
            model: "m".to_string(),
            seed_used: 0,
            gpu: None,
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
    }

    #[test]
    fn build_sse_complete_event_image_clears_all_video_fields() {
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-schnell:q8".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert_eq!(event.format, OutputFormat::Png);
        assert!(event.video_frames.is_none());
        assert!(event.video_fps.is_none());
        assert!(event.video_thumbnail.is_none());
        assert!(event.video_gif_preview.is_none());
        assert!(!event.video_has_audio);
        assert!(event.video_duration_ms.is_none());
    }

    #[test]
    fn build_sse_complete_event_carries_saved_names_and_recorded_metadata() {
        let mut req = fake_request("flux-dev:q4");
        req.batch_id = Some("prepared-batch-1".to_string());
        req.batch_index = Some(2);
        req.batch_count = Some(3);
        let resp = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let metadata =
            OutputMetadata::from_generate_request(&req, resp.seed_used, None, "test-version");
        let saved = SavedOutputNames {
            output: Some("flux-dev-q4-123.png".to_string()),
            original: Some("flux-dev-q4-123-original.png".to_string()),
        };
        let event = build_sse_complete_event(
            &resp,
            &fake_image(),
            None,
            Some(&metadata),
            &saved,
            SseCompletionPayload::Full,
        );
        assert_eq!(event.filename.as_deref(), Some("flux-dev-q4-123.png"));
        assert_eq!(
            event.original_filename.as_deref(),
            Some("flux-dev-q4-123-original.png")
        );
        // The event metadata mirrors what the save path records: the
        // payload's actual dimensions, not the request's.
        let meta = event.metadata.expect("metadata rides the complete event");
        assert_eq!(meta.seed, 5);
        assert_eq!(meta.width, fake_image().width);
        assert_eq!(meta.height, fake_image().height);
        assert_eq!(meta.batch_id.as_deref(), Some("prepared-batch-1"));
        assert_eq!(meta.batch_index, Some(2));
        assert_eq!(meta.batch_count, Some(3));

        let metadata_only = build_sse_complete_event(
            &resp,
            &fake_image(),
            Some(&fake_image()),
            Some(&metadata),
            &saved,
            SseCompletionPayload::MetadataOnly,
        );
        assert!(metadata_only.image.is_empty());
        assert!(metadata_only.original_image.is_none());
        assert_eq!(
            metadata_only.filename.as_deref(),
            Some("flux-dev-q4-123.png")
        );
        assert!(metadata_only.metadata.is_some());
    }

    #[test]
    fn metadata_only_completion_fails_when_the_output_was_not_saved() {
        let response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![fake_image()],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let message = build_sse_completion_message(
            &response,
            &fake_image(),
            None,
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::MetadataOnly,
        );
        match message {
            SseMessage::Error(error) => assert!(error.message.contains("could not be saved")),
            _ => panic!("metadata-only completion without a file must be an SSE error"),
        }
    }

    #[test]
    fn post_generation_upscale_replaces_image_response_dimensions() {
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let mut response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: None,
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let img = fake_image();
        let upscaled = mold_core::UpscaleResponse {
            image: ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 2048,
                height: 2048,
                index: 0,
            },
            upscale_time_ms: 42,
            model: "real-esrgan-x4plus:fp16".to_string(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
        };

        let next = apply_upscale_response_to_image_generation(&req, &mut response, img, upscaled)
            .expect("image upscale should apply");
        let event = build_sse_complete_event(
            &response,
            &next,
            Some(&fake_image()),
            None,
            &SavedOutputNames::default(),
            SseCompletionPayload::Full,
        );
        assert!(event.original_image.is_some());
        assert_eq!(event.original_width, Some(512));
        assert_eq!(event.original_height, Some(512));
        let mut metadata =
            OutputMetadata::from_generate_request(&req, response.seed_used, None, "test-version");
        apply_output_dimensions_to_metadata(&mut metadata, &next);

        assert_eq!(next.width, 2048);
        assert_eq!(next.height, 2048);
        assert_eq!(event.width, 2048);
        assert_eq!(event.height, 2048);
        assert_eq!(metadata.width, 2048);
        assert_eq!(metadata.height, 2048);
        assert_eq!(metadata.generation_width, Some(512));
        assert_eq!(metadata.generation_height, Some(512));
        assert_eq!(
            metadata.upscale_model.as_deref(),
            Some("real-esrgan-x4plus:fp16")
        );
    }

    #[test]
    fn failed_post_generation_upscale_keeps_only_the_original_output() {
        let original = fake_image();
        let (output, preserved_original, error) = settle_post_generation_upscale(
            original.clone(),
            Err("upscaler unavailable".to_string()),
        );

        assert_eq!(output.data, original.data);
        assert!(preserved_original.is_none());
        assert_eq!(error.as_deref(), Some("upscaler unavailable"));
    }

    #[test]
    fn post_generation_upscale_skips_video_responses() {
        let mut req = fake_request("ltx-video:fp16");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let video = mold_core::VideoData {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            data: vec![0, 0, 0, 24],
            format: OutputFormat::Mp4,
            width: 512,
            height: 512,
            frames: 25,
            fps: 24,
            pipeline: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            thumbnail: vec![9, 9],
            gif_preview: vec![],
            has_audio: false,
            duration_ms: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        let mut response = mold_core::GenerateResponse {
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
            images: vec![],
            video: Some(video),
            generation_time_ms: 100,
            model: "ltx-video:fp16".to_string(),
            seed_used: 5,
            gpu: None,
        };
        let img = fake_image();
        let upscaled = mold_core::UpscaleResponse {
            image: ImageData {
                data: vec![1, 2, 3],
                format: OutputFormat::Png,
                width: 2048,
                height: 2048,
                index: 0,
            },
            upscale_time_ms: 42,
            model: "real-esrgan-x4plus:fp16".to_string(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
        };

        let next = apply_upscale_response_to_image_generation(&req, &mut response, img, upscaled)
            .expect("video upscale should be skipped");

        assert_eq!(next.width, 512);
        assert_eq!(next.height, 512);
        assert!(response.video.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_noops_without_model() {
        let state = empty_test_state(mold_core::Config::default());
        let req = fake_request("flux-dev:q4");

        let next = upscale_generated_image_on_single_worker(&state, &req, 5, fake_image(), None)
            .await
            .expect("missing upscale model should leave the image unchanged");

        assert_eq!(next.width, 512);
        assert_eq!(next.height, 512);
        assert_eq!(next.index, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_rejects_unknown_upscaler_manifest() {
        let state = empty_test_state(mold_core::Config::default());
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("definitely-not-a-real-upscaler:fp16".to_string());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

        let err = upscale_generated_image_on_single_worker(
            &state,
            &req,
            5,
            fake_image(),
            Some(&progress_tx),
        )
        .await
        .expect_err("unknown upscalers should fail before generation completes");

        assert!(err.contains("unknown upscaler model"), "got: {err}");
        let first_progress = progress_rx
            .try_recv()
            .expect("loading stage should be emitted before validation fails");
        assert!(matches!(
            first_progress,
            SseMessage::Progress(SseProgressEvent::StageStart { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_post_upscale_surfaces_missing_weights_path() {
        let tmp = TempDir::new().unwrap();
        let missing_weights = tmp.path().join("missing-upscaler.safetensors");
        let mut config = mold_core::Config::default();
        config.models.insert(
            "real-esrgan-x4plus:fp16".to_string(),
            ModelConfig {
                transformer: Some(missing_weights.display().to_string()),
                ..Default::default()
            },
        );
        let state = empty_test_state(config);
        let mut req = fake_request("flux-dev:q4");
        req.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();

        let err = upscale_generated_image_on_single_worker(
            &state,
            &req,
            5,
            fake_image(),
            Some(&progress_tx),
        )
        .await
        .expect_err("missing weight files should be surfaced");

        assert!(err.contains("upscale failed"), "got: {err}");
        assert!(err.contains("upscaler weights not found"), "got: {err}");
        let first_progress = progress_rx
            .try_recv()
            .expect("loading stage should be emitted before loading fails");
        assert!(matches!(
            first_progress,
            SseMessage::Progress(SseProgressEvent::StageStart { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_waits_for_worker_capacity_instead_of_rejecting() {
        let (worker, worker_rx) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            8,
        );

        let (filler_result_tx, _filler_result_rx) = tokio::sync::oneshot::channel();
        let filler_job = crate::gpu_pool::GpuJob {
            id: String::new(),
            durable_queue_rank: None,
            model: "busy-model".to_string(),
            request: fake_request("busy-model"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx: filler_result_tx,
            output_dir: None,
            config: state.config.clone(),
            metadata_db: state.metadata_db.clone(),
            gallery_publication_gate: state.gallery_publication_gate.clone(),
            queue: state.queue.clone(),
            registry: state.job_registry.clone(),
            events: state.events.clone(),
            execution_plan: None,
            prepared_execution_inputs: None,
            #[cfg(any(test, feature = "h3-private-bridge", feature = "h3-private-uat"))]
            h3_prepared_attempt: None,
            lease: None,
            journal: None,
        };
        worker.send_job(filler_job).unwrap();

        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning(
            job_rx,
            state.clone(),
            8,
            DEFAULT_MAX_DEFERRALS,
            tokio_util::sync::CancellationToken::new(),
        ));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            durable_queue_rank: None,
            request: fake_request("flux-dev:q4"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            result_rx.try_recv().is_err(),
            "dispatcher should keep the job pending while all worker channels are full"
        );

        let _filler = worker_rx
            .recv()
            .expect("filler job should occupy the local channel");
        let dispatched = recv_worker_job(&worker_rx, std::time::Duration::from_secs(1))
            .expect("queued job should dispatch once capacity is available");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_waits_for_degraded_worker_recovery_instead_of_rejecting() {
        let (worker, worker_rx) = test_worker(0, 1);
        worker.consecutive_failures.store(3, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() =
            Some(Instant::now() + std::time::Duration::from_secs(60));

        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            8,
        );
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, mut result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            durable_queue_rank: None,
            request: fake_request("flux-dev:q4"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        queue.submit(job, 8).await.unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            result_rx.try_recv().is_err(),
            "dispatcher should keep the job pending while all workers are degraded"
        );
        assert!(
            worker_rx.try_recv().is_err(),
            "degraded worker must not receive work before recovery"
        );

        worker.consecutive_failures.store(0, Ordering::SeqCst);
        *worker.degraded_until.write().unwrap() = None;

        let dispatched = recv_worker_job(&worker_rx, std::time::Duration::from_secs(1))
            .expect("queued job should dispatch once a worker recovers");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    /// Regression for the take-and-restore refactor in `process_job`: when
    /// the engine vanishes from the cache between `ensure_model_ready` and
    /// `cache.take()`, the take path must produce `None` (handled with a
    /// clean error in `process_job`) rather than panicking. The pure cache
    /// invariant — `take()` on an absent model returns `None` — is what
    /// keeps the take-and-restore safe.
    #[tokio::test]
    async fn cache_take_on_vanished_engine_returns_none_not_panic() {
        use crate::model_cache::ModelCache;
        use mold_core::GenerateResponse;
        use mold_inference::InferenceEngine;

        struct StubEngine(&'static str);
        impl InferenceEngine for StubEngine {
            fn generate(&mut self, _r: &GenerateRequest) -> anyhow::Result<GenerateResponse> {
                unimplemented!()
            }
            fn model_name(&self) -> &str {
                self.0
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn load(&mut self) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let mut cache = ModelCache::new(3);
        // Cache empty (engine never inserted, or evicted/removed by a
        // concurrent admin call between `ensure_model_ready` and `take`).
        assert!(cache.take("vanished-model").is_none());

        // After a take of a present engine, a subsequent take of the same
        // name must also return None — guards against double-take in the
        // restore path.
        cache.insert(Box::new(StubEngine("present-model")), 0);
        let first = cache.take("present-model");
        assert!(first.is_some());
        assert!(
            cache.take("present-model").is_none(),
            "double-take must return None"
        );
    }

    fn buf_job(model: &str) -> BufferedJob {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        BufferedJob::new(crate::state::GenerationJob {
            id: String::new(),
            durable_queue_rank: None,
            request: fake_request(model),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx: tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        })
    }

    fn buf_job_with_id(id: &str, model: &str) -> BufferedJob {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        BufferedJob::new(crate::state::GenerationJob {
            id: id.to_string(),
            durable_queue_rank: None,
            request: fake_request(model),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx: tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        })
    }

    #[test]
    fn align_buffer_reorders_to_match_registry_queued_order() {
        use std::collections::VecDeque;
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        for id in ["a", "b", "c"] {
            buffer.push_back(buf_job_with_id(id, &format!("model-{id}")));
        }
        // Registry moved c to the front, then a, then b.
        let order = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        align_buffer_to_registry_order(&mut buffer, &order);
        let ids: Vec<&str> = buffer.iter().map(|b| b.job.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "a", "b"]);
    }

    #[test]
    fn align_buffer_is_a_noop_when_already_in_registry_order() {
        use std::collections::VecDeque;
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job_with_id("a", "model-a"));
        // Give the middle job a non-zero deferral count so we can prove the
        // no-op path preserves per-job starvation accounting.
        let mut b = buf_job_with_id("b", "model-b");
        b.deferred = 2;
        buffer.push_back(b);
        buffer.push_back(buf_job_with_id("c", "model-c"));
        let order = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        align_buffer_to_registry_order(&mut buffer, &order);
        let ids: Vec<&str> = buffer.iter().map(|b| b.job.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
        assert_eq!(
            buffer[1].deferred, 2,
            "a no-op align must preserve the deferred starvation count"
        );
    }

    #[test]
    fn align_buffer_keeps_unregistered_jobs_in_arrival_order_at_the_back() {
        use std::collections::VecDeque;
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        // "x" and "y" aren't in the registry order (e.g. cancelled out but
        // still holding a buffer slot); "b" and "a" are.
        for id in ["x", "b", "y", "a"] {
            buffer.push_back(buf_job_with_id(id, "m"));
        }
        let order = vec!["a".to_string(), "b".to_string()];
        align_buffer_to_registry_order(&mut buffer, &order);
        let ids: Vec<&str> = buffer.iter().map(|b| b.job.id.as_str()).collect();
        // Registry-tracked jobs first in registry order (a, b), then the
        // untracked ones in their original arrival order (x, y).
        assert_eq!(ids, vec!["a", "b", "x", "y"]);
    }

    #[test]
    fn align_buffer_leaves_empty_id_jobs_untouched() {
        // Tests that submit `GenerationJob`s directly (empty ids) never
        // register in the registry, so the align pass must be a stable no-op
        // that preserves the model-swap picker's interleaving assumptions.
        use std::collections::VecDeque;
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        for model in ["a", "b", "a", "b"] {
            buffer.push_back(buf_job(model));
        }
        align_buffer_to_registry_order(&mut buffer, &[]);
        let models: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(models, vec!["a", "b", "a", "b"]);
    }

    #[test]
    fn pick_next_job_picks_head_when_head_model_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        let loaded: HashSet<String> = ["a".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
        assert_eq!(
            buffer.front().unwrap().deferred,
            0,
            "head shouldn't be deferred when picker chose the head itself"
        );
    }

    #[test]
    fn pick_next_job_picks_non_head_when_only_non_head_model_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        let loaded: HashSet<String> = ["b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "b");
        assert_eq!(buffer.len(), 2);
        // The head ("a") was skipped once and now sits at deferral=1.
        assert_eq!(buffer.front().unwrap().job.request.model, "a");
        assert_eq!(buffer.front().unwrap().deferred, 1);
    }

    #[test]
    fn pick_next_job_force_dispatches_head_after_max_deferrals() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        let mut head = buf_job("a");
        head.deferred = 3;
        buffer.push_back(head);
        buffer.push_back(buf_job("b"));
        // Even though only `b` is loaded, head ("a") has hit the budget and wins.
        let loaded: HashSet<String> = ["b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
    }

    #[test]
    fn pick_next_job_falls_back_to_head_when_nothing_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = HashSet::new();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(picked.request.model, "a");
    }

    /// Fix D: with `max_deferrals = 0`, every reorder would exceed the
    /// budget on the very first skip, so the picker degenerates to FIFO —
    /// the head wins regardless of which model is loaded.
    #[test]
    fn pick_next_job_max_deferrals_zero_picks_head_even_when_non_head_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("b")); // head
        buffer.push_back(buf_job("a")); // non-head
        let loaded: HashSet<String> = ["a".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 0);
        assert_eq!(
            picked.request.model, "b",
            "max_deferrals=0 must force FIFO — head must win even when only the non-head model is loaded"
        );
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "a");
    }

    /// Fix D: with `max_deferrals = 0` and an empty `loaded` set, the head
    /// is the only candidate anyway. Locks in the FIFO behaviour when
    /// nothing is warm.
    #[test]
    fn pick_next_job_max_deferrals_zero_with_empty_loaded_picks_head() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a")); // head
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = HashSet::new();
        let picked = pick_next_job(&mut buffer, &loaded, 0);
        assert_eq!(picked.request.model, "a");
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front().unwrap().job.request.model, "b");
    }

    /// Fix E: when both head and a non-head match `loaded`, the picker must
    /// pick the front-most match — i.e. the first `A` in `[A, B, A, B]`
    /// when both `A` and `B` are loaded. Locks in arrival-order stability
    /// across multiple matching jobs.
    #[test]
    fn pick_next_job_picks_front_most_match_when_multiple_loaded() {
        use std::collections::{HashSet, VecDeque};
        let mut buffer: VecDeque<BufferedJob> = VecDeque::new();
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        buffer.push_back(buf_job("a"));
        buffer.push_back(buf_job("b"));
        let loaded: HashSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
        let picked = pick_next_job(&mut buffer, &loaded, 3);
        assert_eq!(
            picked.request.model, "a",
            "front-most match wins (the first `a`), not the loaded model with the most copies later in the buffer"
        );
        // Three jobs remain: [b, a, b]; head was the picked first `a` so the
        // new head is the original-index-1 `b`. Nothing was deferred because
        // the picker chose the head itself.
        assert_eq!(buffer.len(), 3);
        let remaining: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(remaining, vec!["b", "a", "b"]);
        assert_eq!(buffer.front().unwrap().deferred, 0);
    }

    /// Integration: an interleaved `[A, B, A, B]` queue dispatched against a
    /// single worker that has model `A` warm should reorder so both `A` jobs
    /// run first, then both `B` jobs — minimizing model swaps from 4 → 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_reorders_interleaved_jobs_to_minimize_swaps() {
        let (worker, worker_rx) = test_worker(0, 8);
        // Pre-mark the worker as having model "a" loaded so the picker
        // recognises it as warm.
        {
            let mut cache = worker.model_cache.lock().unwrap();
            struct Engine(&'static str);
            impl mold_inference::InferenceEngine for Engine {
                fn generate(
                    &mut self,
                    _r: &GenerateRequest,
                ) -> anyhow::Result<mold_core::GenerateResponse> {
                    unimplemented!()
                }
                fn model_name(&self) -> &str {
                    self.0
                }
                fn is_loaded(&self) -> bool {
                    true
                }
                fn load(&mut self) -> anyhow::Result<()> {
                    Ok(())
                }
            }
            cache.insert(Box::new(Engine("a")), 0);
        }
        worker.set_resident_model(Some("a"));

        let (job_tx, job_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            8,
        );

        // Submit [a, b, a, b] BEFORE the dispatcher spins up so the buffer
        // top-up sees all four at once.
        let mut result_rxs = Vec::new();
        for model in ["a", "b", "a", "b"] {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let job = crate::state::GenerationJob {
                id: String::new(),
                durable_queue_rank: None,
                request: fake_request(model),
                deferred_media: None,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
                journal: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
            };
            queue.submit(job, 8).await.unwrap();
            result_rxs.push(rx);
        }

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let mut order = Vec::new();
        for _ in 0..4 {
            let dispatched = recv_worker_job(&worker_rx, std::time::Duration::from_secs(2))
                .expect("worker should receive the dispatched job");
            order.push(dispatched.model);
        }
        drop(job_tx);
        dispatcher.abort();

        assert_eq!(
            order,
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
            ],
            "lookahead reorder should batch all `a` jobs together before swapping to `b`"
        );
    }

    /// A `PATCH /api/queue/:id {position}` reorder must change *real* dispatch
    /// order, not just the `GET /api/queue` snapshot. Pause the queue so the
    /// dispatcher parks before pulling anything, submit A, B, C (all buffered
    /// together on resume), move C to the front of the registry, then resume —
    /// the worker must receive C first, then A, B. Distinct models with nothing
    /// warm keep the model-swap picker on the buffer head, so the registry
    /// reorder is the only thing that can change the order.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_honors_registry_reorder_in_real_dispatch() {
        let (worker, worker_rx) = test_worker(0, 8);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(8);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            8,
        );

        // Pause *before* the dispatcher exists so its first `wait_if_paused`
        // parks it ahead of the pre-recv gate — nothing is pulled off the
        // channel until we resume, so all three jobs buffer together.
        state.queue_pause.pause();
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        // Register + submit A, B, C with matching ids so the dispatcher's
        // registry lookups line up with the queue payloads.
        let mut result_rxs = Vec::new();
        for id in ["a", "b", "c"] {
            state
                .job_registry
                .register_job(id, format!("model-{id}"), None, None, None);
            let (tx, rx) = tokio::sync::oneshot::channel();
            let job = crate::state::GenerationJob {
                id: id.to_string(),
                durable_queue_rank: None,
                request: fake_request(&format!("model-{id}")),
                deferred_media: None,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
                journal: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
            };
            queue.submit(job, 8).await.unwrap();
            result_rxs.push(rx);
        }

        // Move C to the front of the registry — the single source of truth for
        // dispatch order.
        state.job_registry.reorder_queued("c", 0).unwrap();

        // Resume: the dispatcher drains A, B, C (still submission-ordered in the
        // channel), aligns its buffer to the registry, and dispatches.
        state.queue_pause.resume();

        let mut order = Vec::new();
        for _ in 0..3 {
            let dispatched = recv_worker_job(&worker_rx, std::time::Duration::from_secs(2))
                .expect("worker should receive the dispatched job");
            order.push(dispatched.model);
        }
        drop(job_tx);
        dispatcher.abort();

        assert_eq!(
            order,
            vec![
                "model-c".to_string(),
                "model-a".to_string(),
                "model-b".to_string(),
            ],
            "registry reorder must drive real dispatch order, not just the snapshot"
        );
    }

    /// Fix F: the `top_up_buffer` helper must never grow the buffer past
    /// `buffer_size`, no matter how many jobs are sitting in the channel.
    /// This is the load-bearing invariant that bounds the working set the
    /// picker considers — without it a burst submission could let the
    /// dispatcher reorder across the entire pending queue, defeating the
    /// fairness guarantees the `deferred` counter is built around.
    #[tokio::test]
    async fn top_up_buffer_never_exceeds_capacity() {
        use std::collections::VecDeque;
        let (job_tx, mut job_rx) = tokio::sync::mpsc::channel::<GenerationJob>(32);

        // Submit 10 jobs into the channel synchronously so the buffer's top-up
        // call sees them all immediately available via try_recv.
        for i in 0..10 {
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let job = GenerationJob {
                id: String::new(),
                durable_queue_rank: None,
                request: fake_request(&format!("model-{i}")),
                deferred_media: None,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
                journal: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
            };
            job_tx.send(job).await.unwrap();
        }

        // buffer_size = 4 — top_up must stop at 4 even with 10 in the channel.
        let mut buffer: VecDeque<BufferedJob> = VecDeque::with_capacity(4);
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(
            buffer.len(),
            4,
            "top_up_buffer must cap at buffer_size, leaving the rest in the channel"
        );

        // Drain the four buffered jobs, then top up again; the next call must
        // pull only the next four from the channel (FIFO order preserved).
        while buffer.pop_front().is_some() {}
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(buffer.len(), 4);
        let names: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["model-4", "model-5", "model-6", "model-7"],
            "second top-up must drain the next FIFO window from the channel"
        );

        // Drop sender so the channel reports closed; remaining 2 jobs still
        // arrive via try_recv before the channel goes dry.
        drop(job_tx);
        while buffer.pop_front().is_some() {}
        top_up_buffer(&mut buffer, &mut job_rx, 4);
        assert_eq!(
            buffer.len(),
            2,
            "top_up_buffer drains the channel tail when fewer jobs than capacity remain"
        );
        let names: Vec<&str> = buffer
            .iter()
            .map(|b| b.job.request.model.as_str())
            .collect();
        assert_eq!(names, vec!["model-8", "model-9"]);
    }

    /// Same invariant, but reached via the dispatcher loop (integration). A
    /// burst of N > buffer_size jobs must still dispatch in FIFO order with
    /// no jobs lost — the buffer cap can't drop traffic, only delay it. We
    /// drain the worker channel as fast as the dispatcher fills it, so the
    /// test exercises buffer rotation rather than worker-channel back-pressure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_dispatches_all_jobs_when_submission_exceeds_buffer() {
        let (worker, worker_rx) = test_worker(0, 4);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(32);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker.clone()].into(),
            }),
            32,
        );

        // Drain the worker channel concurrently and decrement in_flight as
        // a real worker would, so the dispatcher's worker-selection sees the
        // worker as idle for each subsequent send (otherwise `in_flight`
        // grows unbounded and the worker never re-classifies as eligible
        // when the sync-channel fills).
        let drain_worker = worker.clone();
        let drainer = std::thread::spawn(move || {
            let mut order = Vec::new();
            while order.len() < 10 {
                match recv_worker_job(&worker_rx, std::time::Duration::from_secs(5)) {
                    Ok(j) => {
                        drain_worker.in_flight.fetch_sub(1, Ordering::SeqCst);
                        order.push(j.model);
                    }
                    Err(e) => panic!("drain stalled at {:?}: {e:?}", order),
                }
            }
            order
        });

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        // Submit AFTER the dispatcher and drainer are running so we exercise
        // the live top-up loop rather than a one-shot drain of a pre-filled
        // channel. Hold result_rx values past the dispatch — the dispatcher
        // skips jobs whose result_tx is closed, which would otherwise drop
        // every job before it reaches the worker channel.
        let mut held_rxs = Vec::new();
        for i in 0..10 {
            let (tx, rx) = tokio::sync::oneshot::channel();
            held_rxs.push(rx);
            let job = crate::state::GenerationJob {
                id: String::new(),
                durable_queue_rank: None,
                request: fake_request(&format!("model-{i}")),
                deferred_media: None,
                completion_payload: SseCompletionPayload::Full,
                progress_tx: None,
                result_tx: tx,
                output_dir: None,
                journal: None,
                #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                h3_private_ingress_grant: None,
            };
            queue.submit(job, 32).await.unwrap();
        }

        let order = drainer.join().expect("drainer thread panic");
        drop(job_tx);
        dispatcher.abort();

        let expected: Vec<String> = (0..10).map(|i| format!("model-{i}")).collect();
        assert_eq!(
            order, expected,
            "10 distinct jobs must come out in FIFO across buffer rotations"
        );
    }

    fn with_queue_env<R>(name: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _g = crate::test_support::env_lock();
        let prev = std::env::var(name).ok();
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
        out
    }

    #[test]
    fn resolve_lookahead_buffer_uses_default_when_env_missing() {
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, None, resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_lookahead_buffer_honors_env_within_range() {
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("4"), resolve_lookahead_buffer);
        assert_eq!(n, 4);
    }

    #[test]
    fn resolve_lookahead_buffer_falls_back_when_out_of_range() {
        // 0 is below the 1 lower bound; 999 is above the 64 upper bound.
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("0"), resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
        let n = with_queue_env(LOOKAHEAD_BUFFER_ENV, Some("999"), resolve_lookahead_buffer);
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_lookahead_buffer_falls_back_when_unparseable() {
        let n = with_queue_env(
            LOOKAHEAD_BUFFER_ENV,
            Some("not-a-number"),
            resolve_lookahead_buffer,
        );
        assert_eq!(n, DEFAULT_LOOKAHEAD_BUFFER);
    }

    #[test]
    fn resolve_max_deferrals_uses_default_when_env_missing() {
        let n = with_queue_env(MAX_DEFERRALS_ENV, None, resolve_max_deferrals);
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[test]
    fn resolve_max_deferrals_honors_env_within_range() {
        // 0 is the in-range "FIFO" sentinel, 32 is the upper edge.
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("0"), resolve_max_deferrals);
        assert_eq!(n, 0);
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("32"), resolve_max_deferrals);
        assert_eq!(n, 32);
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("5"), resolve_max_deferrals);
        assert_eq!(n, 5);
    }

    #[test]
    fn resolve_max_deferrals_falls_back_when_out_of_range() {
        let n = with_queue_env(MAX_DEFERRALS_ENV, Some("999"), resolve_max_deferrals);
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[test]
    fn resolve_max_deferrals_falls_back_when_unparseable() {
        let n = with_queue_env(
            MAX_DEFERRALS_ENV,
            Some("not-a-number"),
            resolve_max_deferrals,
        );
        assert_eq!(n, DEFAULT_MAX_DEFERRALS);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_honors_explicit_placement_gpu() {
        let (worker0, rx0) = test_worker(0, 1);
        let (worker1, rx1) = test_worker(1, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0, worker1].into(),
            }),
            8,
        );

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state));

        let mut request = fake_request("flux-dev:q4");
        request.placement = Some(mold_core::types::DevicePlacement {
            text_encoders: mold_core::types::DeviceRef::Auto,
            advanced: Some(mold_core::types::AdvancedPlacement {
                transformer: mold_core::types::DeviceRef::gpu(1),
                ..mold_core::types::AdvancedPlacement::default()
            }),
        });

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: String::new(),
            durable_queue_rank: None,
            request,
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        let dispatched = recv_worker_job(&rx1, std::time::Duration::from_secs(1))
            .expect("explicit placement should route to gpu 1");
        assert_eq!(dispatched.model, "flux-dev:q4");
        assert!(rx0.try_recv().is_err(), "gpu 0 should not receive the job");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_dispatcher_records_auto_selected_gpu_before_worker_starts() {
        let (worker0, rx0) = test_worker(0, 1);
        let (worker1, rx1) = test_worker(1, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0, worker1].into(),
            }),
            8,
        );
        state.job_registry.register("auto-job", "flux-dev:q4");

        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "auto-job".to_string(),
            durable_queue_rank: None,
            request: fake_request("flux-dev:q4"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        let (dispatched, ordinal) = match recv_worker_job(&rx0, std::time::Duration::from_secs(1)) {
            Ok(job) => (job, 0),
            Err(_) => (
                recv_worker_job(&rx1, std::time::Duration::from_secs(1))
                    .expect("auto job should dispatch to one GPU"),
                1,
            ),
        };
        assert_eq!(dispatched.model, "flux-dev:q4");
        let entry = state.job_registry.entry("auto-job").unwrap();
        assert_eq!(entry.state, crate::job_registry::JobLifecycle::Running);
        assert_eq!(entry.gpu, Some(ordinal));
        assert_eq!(entry.target_gpu, None);

        drop(job_tx);
        dispatcher.abort();
    }

    async fn legacy_dispatch_retargets_after_blocked_patch(
        mode: crate::dispatch_mode::DispatchMode,
    ) {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("gallery");
        std::fs::create_dir_all(&output).unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let (worker0, rx0) = test_worker(0, 2);
        let (worker1, rx1) = test_worker(1, 2);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let mut state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0.clone(), worker1.clone()].into(),
            }),
            4,
        );
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db.clone(),
            Some(root.path()),
            "legacy-dispatch-race",
        ));
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(scheduled_tx, mode);

        let id = format!("{}-blocked-patch", mode.as_str());
        let request = fake_request("flux-dev:q4");
        state
            .job_registry
            .register_with_target_gpu(&id, &request.model, Some(0));
        let ticket = state
            .queue_journal
            .record_for_test(crate::queue_journal::JournalAdmission {
                id: &id,
                request: &request,
                output_dir: Some(&output),
                target_gpu: Some(0),
                target_device_id: None,
                completion_payload: SseCompletionPayload::Full,
                batch_child: false,
            })
            .expect("durable test job");
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        queue
            .submit(
                GenerationJob {
                    id: id.clone(),
                    durable_queue_rank: None,
                    request,
                    deferred_media: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: Some(output),
                    journal: Some(ticket),
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
                },
                4,
            )
            .await
            .unwrap();

        let selected = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning_inner(
            job_rx,
            state.clone(),
            4,
            DEFAULT_MAX_DEFERRALS,
            tokio_util::sync::CancellationToken::new(),
            Some(LegacyGenerationDispatchHook {
                selected: selected.clone(),
                resume: resume.clone(),
                upscale_preparation: None,
            }),
        ));
        selected.notified().await;

        // Hold the actual SQLite connection after selection. PATCH installs
        // its exact runtime blocker, then waits in the DB without owning the
        // scheduler fence.
        let locked_db = db.clone();
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(0);
        let blocker = tokio::task::spawn_blocking(move || {
            locked_db.as_ref().as_ref().unwrap().with_conn(|_| {
                locked_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            })
        });
        locked_rx.await.unwrap();

        let patch_state = state.clone();
        let patch_id = id.clone();
        let patch = tokio::spawn(async move {
            let _durable = patch_state.queue_journal.lock_durable_transition().await;
            let token = {
                let _scheduler = patch_state.scheduler_mutation_fence.lock().await;
                patch_state
                    .job_registry
                    .begin_queue_patch(&patch_id)
                    .unwrap()
            };
            let journal = patch_state.queue_journal.clone();
            let durable_id = patch_id.clone();
            tokio::task::spawn_blocking(move || {
                journal.patch_owned_any_queued(&durable_id, Some(Some(1)), None, None)
            })
            .await
            .unwrap()
            .unwrap();
            let _scheduler = patch_state.scheduler_mutation_fence.lock().await;
            assert!(patch_state
                .job_registry
                .queue_patch_token_matches(&patch_id, token));
            patch_state
                .job_registry
                .set_target_gpu(&patch_id, Some(1))
                .unwrap();
            patch_state
                .job_registry
                .finish_queue_patch(&patch_id, token);
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while state.job_registry.scheduler_lifecycle(&id)
                != Some(crate::job_registry::JobLifecycle::Held)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("PATCH installs its dispatch blocker before the DB commit");

        resume.notify_one();
        assert!(rx0
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err());
        assert!(rx1
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err());

        state.job_registry.register("unrelated-live", "model-live");
        let scheduler = state
            .scheduler_mutation_fence
            .try_lock()
            .expect("DB-blocked PATCH leaves unrelated scheduler work live");
        state
            .job_registry
            .dispatch_if_queued("unrelated-live", 0, (), |_| Ok(()))
            .unwrap();
        drop(scheduler);

        release_tx.send(()).unwrap();
        blocker.await.unwrap().unwrap();
        patch.await.unwrap();
        let dispatched = recv_worker_job(&rx1, std::time::Duration::from_secs(2))
            .expect("the freshly retargeted job must reach gpu 1");
        assert_eq!(dispatched.id, id);
        assert!(
            rx0.try_recv().is_err(),
            "gpu 0 must never see the stale target"
        );
        let entry = state.job_registry.entry(&id).unwrap();
        assert_eq!(entry.state, crate::job_registry::JobLifecycle::Running);
        assert_eq!(entry.gpu, Some(1));
        assert_eq!(
            state.job_registry.entry("unrelated-live").unwrap().state,
            crate::job_registry::JobLifecycle::Running
        );

        worker1.settle_legacy_transport();
        state.job_registry.remove(&id);
        state.job_registry.remove("unrelated-live");
        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_and_observe_dispatch_never_send_a_db_blocked_stale_patch_target() {
        for mode in [
            crate::dispatch_mode::DispatchMode::Legacy,
            crate::dispatch_mode::DispatchMode::Observe,
        ] {
            legacy_dispatch_retargets_after_blocked_patch(mode).await;
        }
    }

    async fn legacy_dispatch_drops_cancelled_selection_and_runs_unrelated_work(
        mode: crate::dispatch_mode::DispatchMode,
    ) {
        let (worker0, rx0) = test_worker(0, 4);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let mut state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0.clone()].into(),
            }),
            4,
        );
        let (scheduled_tx, _scheduled_rx) = tokio::sync::mpsc::channel(1);
        state.scheduled_work = crate::scheduler::ScheduledWorkHandle::for_mode(scheduled_tx, mode);

        let mut results = Vec::new();
        for id in ["cancel-stale", "unrelated-next"] {
            state.job_registry.register(id, "flux-dev:q4");
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            results.push(result_rx);
            queue
                .submit(
                    GenerationJob {
                        id: id.to_string(),
                        durable_queue_rank: None,
                        request: fake_request("flux-dev:q4"),
                        deferred_media: None,
                        completion_payload: SseCompletionPayload::Full,
                        progress_tx: None,
                        result_tx,
                        output_dir: None,
                        journal: None,
                        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                        h3_private_ingress_grant: None,
                    },
                    4,
                )
                .await
                .unwrap();
        }

        let selected = Arc::new(tokio::sync::Notify::new());
        let resume = Arc::new(tokio::sync::Notify::new());
        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning_inner(
            job_rx,
            state.clone(),
            4,
            DEFAULT_MAX_DEFERRALS,
            tokio_util::sync::CancellationToken::new(),
            Some(LegacyGenerationDispatchHook {
                selected: selected.clone(),
                resume: resume.clone(),
                upscale_preparation: None,
            }),
        ));
        selected.notified().await;
        {
            let _scheduler = state.scheduler_mutation_fence.lock().await;
            state.job_registry.cancel_queued("cancel-stale").unwrap();
        }
        resume.notify_one();

        let dispatched = recv_worker_job(&rx0, std::time::Duration::from_secs(2))
            .expect("the unrelated queued job remains live");
        assert_eq!(dispatched.id, "unrelated-next");
        assert_eq!(
            state.job_registry.entry("unrelated-next").unwrap().state,
            crate::job_registry::JobLifecycle::Running
        );
        assert!(state.job_registry.entry("cancel-stale").is_none());
        let cancelled = tokio::time::timeout(std::time::Duration::from_secs(1), &mut results[0])
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(cancelled, Err(message) if message == "Cancelled"));

        worker0.settle_legacy_transport();
        state.job_registry.remove("unrelated-next");
        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_and_observe_dispatch_drop_cancelled_stale_selection_without_stalling() {
        for mode in [
            crate::dispatch_mode::DispatchMode::Legacy,
            crate::dispatch_mode::DispatchMode::Observe,
        ] {
            legacy_dispatch_drops_cancelled_selection_and_runs_unrelated_work(mode).await;
        }
    }

    async fn legacy_upscale_preparation_outcome(cancel: bool) {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("gallery");
        std::fs::create_dir_all(&output).unwrap();
        let db = Arc::new(Some(MetadataDb::open_in_memory().unwrap()));
        let (worker, worker_rx) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(1);
        let queue = QueueHandle::new(job_tx.clone());
        let mut state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker].into(),
            }),
            1,
        );
        state.metadata_db = db.clone();
        state.queue_journal = Arc::new(crate::queue_journal::QueueJournal::new(
            db,
            Some(root.path()),
            "legacy-upscale-preparation",
        ));

        let id = if cancel {
            "cancel-upscale-preparation"
        } else {
            "shutdown-upscale-preparation"
        };
        let mut request = fake_request("flux-dev:q4");
        request.upscale_model = Some("real-esrgan-x4plus:fp16".to_string());
        state
            .job_registry
            .register_with_target_gpu(id, &request.model, Some(0));
        let ticket = state
            .queue_journal
            .record_for_test(crate::queue_journal::JournalAdmission {
                id,
                request: &request,
                output_dir: Some(&output),
                target_gpu: Some(0),
                target_device_id: None,
                completion_payload: SseCompletionPayload::Full,
                batch_child: false,
            })
            .expect("durable test job");
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        queue
            .submit(
                GenerationJob {
                    id: id.to_string(),
                    durable_queue_rank: None,
                    request,
                    deferred_media: None,
                    completion_payload: SseCompletionPayload::Full,
                    progress_tx: None,
                    result_tx,
                    output_dir: Some(output),
                    journal: Some(ticket),
                    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                    h3_private_ingress_grant: None,
                },
                1,
            )
            .await
            .unwrap();

        let upscale_started = Arc::new(tokio::sync::Notify::new());
        let upscale_resume = Arc::new(tokio::sync::Notify::new());
        let shutdown = tokio_util::sync::CancellationToken::new();
        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning_inner(
            job_rx,
            state.clone(),
            1,
            DEFAULT_MAX_DEFERRALS,
            shutdown.clone(),
            Some(LegacyGenerationDispatchHook {
                selected: Arc::new(tokio::sync::Notify::new()),
                resume: Arc::new(tokio::sync::Notify::new()),
                upscale_preparation: Some(LegacyUpscalePreparationHook {
                    started: upscale_started.clone(),
                    resume: upscale_resume,
                }),
            }),
        ));
        upscale_started.notified().await;

        if cancel {
            assert!(state.queue_journal.cancel_id(id).unwrap());
            let _scheduler = state.scheduler_mutation_fence.lock().await;
            assert!(state.job_registry.cancel_queued(id).is_ok());
        } else {
            shutdown.cancel();
        }

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), result_rx)
            .await
            .expect("the blocked upscaler preparation settles promptly")
            .unwrap();
        shutdown.cancel();
        dispatcher.await.unwrap();
        assert!(worker_rx.try_recv().is_err());
        if cancel {
            assert!(matches!(result, Err(message) if message == "Cancelled"));
            assert!(state
                .queue_journal
                .list_all()
                .iter()
                .all(|row| row.id != id));
        } else {
            assert!(matches!(result, Err(message) if message.contains("shutting down")));
            assert!(state
                .queue_journal
                .list_all()
                .iter()
                .any(|row| row.id == id));
        }
        drop(job_tx);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_upscale_preparation_distinguishes_cancel_from_shutdown_retention() {
        legacy_upscale_preparation_outcome(true).await;
        legacy_upscale_preparation_outcome(false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paused_dispatcher_holds_new_jobs_until_resumed() {
        let (worker0, rx0) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0].into(),
            }),
            8,
        );

        // Pause before the dispatcher runs — a submitted job must stay queued.
        assert!(state.queue_pause.pause());
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));

        state.job_registry.register("paused-job", "flux-dev:q4");
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "paused-job".to_string(),
            durable_queue_rank: None,
            request: fake_request("flux-dev:q4"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        // While paused the worker never receives the job.
        assert!(
            rx0.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "paused dispatcher must not hand a job to a worker"
        );

        // Resume → the queued job dispatches.
        assert!(state.queue_pause.resume());
        let dispatched = recv_worker_job(&rx0, std::time::Duration::from_secs(1))
            .expect("resumed dispatcher should dispatch the queued job");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }

    #[tokio::test]
    async fn cancelling_legacy_dispatcher_rejects_all_accepted_generation_work() {
        let (worker, worker_rx) = test_worker(0, 2);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx);
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker].into(),
            }),
            8,
        );
        state.queue_pause.pause();
        let shutdown = tokio_util::sync::CancellationToken::new();
        let dispatcher = tokio::spawn(run_queue_dispatcher_with_tuning(
            job_rx,
            state.clone(),
            8,
            DEFAULT_MAX_DEFERRALS,
            shutdown.clone(),
        ));

        let mut results = Vec::new();
        for id in ["shutdown-queued-1", "shutdown-queued-2"] {
            let (result_tx, result_rx) = tokio::sync::oneshot::channel();
            state.job_registry.register(id, "flux-dev:q4");
            queue
                .submit(
                    crate::state::GenerationJob {
                        id: id.to_string(),
                        durable_queue_rank: None,
                        request: fake_request("flux-dev:q4"),
                        deferred_media: None,
                        completion_payload: SseCompletionPayload::Full,
                        progress_tx: None,
                        result_tx,
                        output_dir: None,
                        journal: None,
                        #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
                        h3_private_ingress_grant: None,
                    },
                    8,
                )
                .await
                .unwrap();
            results.push(result_rx);
        }
        state.queue_pause.wait_until_blocked().await;
        shutdown.cancel();
        dispatcher.await.unwrap();

        for result in results {
            let error = match result.await.expect("accepted generation must settle") {
                Ok(_) => panic!("unstarted generation must be rejected"),
                Err(error) => error,
            };
            assert!(error.contains("shutting down"));
        }
        assert_eq!(queue.pending(), 0);
        assert_eq!(state.job_registry.len(), 0);
        assert!(matches!(
            worker_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_while_dispatcher_is_parked_on_an_empty_queue_still_holds_the_next_job() {
        // The subtle ordering: the dispatcher passes the top-of-loop gate,
        // then parks in job_rx.recv() on an EMPTY queue. A pause that lands
        // while it is parked must hold the very job whose arrival wakes it —
        // without the post-recv re-check, that job leaks into dispatch.
        let (worker0, rx0) = test_worker(0, 1);
        let (job_tx, job_rx) = tokio::sync::mpsc::channel(4);
        let queue = QueueHandle::new(job_tx.clone());
        let state = crate::state::AppState::empty(
            mold_core::Config::default(),
            queue.clone(),
            Arc::new(GpuPool {
                workers: vec![worker0].into(),
            }),
            8,
        );

        // Dispatcher starts UNPAUSED and parks waiting for work.
        let dispatcher = tokio::spawn(run_queue_dispatcher(job_rx, state.clone()));
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pause lands while it is parked, then a job arrives.
        assert!(state.queue_pause.pause());
        state.job_registry.register("parked-job", "flux-dev:q4");
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = crate::state::GenerationJob {
            id: "parked-job".to_string(),
            durable_queue_rank: None,
            request: fake_request("flux-dev:q4"),
            deferred_media: None,
            completion_payload: SseCompletionPayload::Full,
            progress_tx: None,
            result_tx,
            output_dir: None,
            journal: None,
            #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
            h3_private_ingress_grant: None,
        };
        let _position = queue.submit(job, 8).await.unwrap();

        assert!(
            rx0.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a job arriving while paused must not wake straight into dispatch"
        );

        assert!(state.queue_pause.resume());
        let dispatched = recv_worker_job(&rx0, std::time::Duration::from_secs(1))
            .expect("resume should release the held job");
        assert_eq!(dispatched.model, "flux-dev:q4");

        drop(job_tx);
        dispatcher.abort();
    }
}

#[cfg(test)]
mod queue_pause_tests {
    use super::QueuePause;
    use std::time::Duration;

    #[test]
    fn pause_and_resume_report_state_transitions() {
        let gate = QueuePause::new();
        assert!(!gate.is_paused());
        assert!(gate.pause(), "first pause flips state");
        assert!(gate.is_paused());
        assert!(!gate.pause(), "second pause is a no-op transition");
        assert!(gate.resume(), "first resume flips state");
        assert!(!gate.is_paused());
        assert!(!gate.resume(), "second resume is a no-op transition");
    }

    #[tokio::test]
    async fn wait_if_paused_returns_immediately_when_not_paused() {
        let gate = QueuePause::new();
        // Not paused → the await resolves without needing a resume.
        tokio::time::timeout(Duration::from_secs(1), gate.wait_if_paused())
            .await
            .expect("wait_if_paused must not block when the gate is open");
    }

    #[tokio::test]
    async fn wait_if_paused_blocks_until_resumed() {
        let gate = QueuePause::new();
        assert!(gate.pause());

        let waiter = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.wait_if_paused().await })
        };

        // While paused the waiter must stay parked.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "waiter must block while paused");

        // Resume wakes it via notify_waiters().
        assert!(gate.resume());
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter must unblock within the timeout after resume")
            .expect("waiter task must not panic");
    }

    #[tokio::test]
    async fn resume_wakes_every_gated_waiter() {
        // notify_waiters (not notify_one) so all dispatch loops proceed.
        let gate = QueuePause::new();
        assert!(gate.pause());

        let waiters: Vec<_> = (0..3)
            .map(|_| {
                let gate = gate.clone();
                tokio::spawn(async move { gate.wait_if_paused().await })
            })
            .collect();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(gate.resume());

        for waiter in waiters {
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .expect("every gated waiter must wake on a single resume")
                .expect("waiter task must not panic");
        }
    }
}
