//! Gallery trash: move-to-`.trash/`, restore, permanent delete, empty, and
//! the retention sweeper.
//!
//! Every primitive here moves or unlinks gallery bytes, so each runs on the
//! blocking pool under the WRITE side of the gallery publication gate —
//! exactly like the historical hard delete. The committed-archive authority
//! is kept honest through `batch_transaction::trash_committed_archive_filename`
//! / `restore_trashed_archive_filename`: a trashed committed print retires
//! into `retired_entries` (never collected, because no projection epoch is
//! recorded) and a restore re-homes that exact identity; a permanent delete
//! of a trashed print acknowledges the retirement projection like today's
//! delete does.
//!
//! With the metadata DB disabled there is no trash index, so
//! `DELETE /api/gallery/image/:filename` stays a hard delete and
//! `GET /api/capabilities` advertises `gallery.trash.enabled = false`.
//!
//! Tombstones and the `trashed_at_ms` flag are owned by `mold_db::trash`;
//! this module only orchestrates the order: archive + bytes → tombstone →
//! row flag, so a crash anywhere leaves something reconcile can repair
//! (see `mold_db::reconcile`'s trash passes).

use std::path::{Path, PathBuf};

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    Json,
};
use mold_core::{
    EmptyTrashResult, GalleryImage, ServerEvent, TrashFilenamesRequest, TrashSweepResult,
};
use mold_db::MetadataDb;
use serde::Deserialize;

use crate::batch_transaction::{
    self, ArchiveDeleteDisposition, GalleryPublicationGate, RestoreArchiveDisposition,
    TrashArchiveDisposition,
};
use crate::gallery_organization::{
    clean_gallery_filename, current_retention_days, enriched_gallery_image, gallery_output_dir,
    require_metadata_db,
};
use crate::routes::ApiError;
use crate::state::AppState;

pub(crate) const GALLERY_NOT_FOUND: &str = "GALLERY_NOT_FOUND";
pub(crate) const GALLERY_NOT_TRASHED: &str = "GALLERY_NOT_TRASHED";
pub(crate) const GALLERY_RESTORE_CONFLICT: &str = "GALLERY_RESTORE_CONFLICT";
pub(crate) const GALLERY_DELETE_IDENTITY_CHANGED: &str = "GALLERY_DELETE_IDENTITY_CHANGED";

/// Interval between retention sweeps.
pub(crate) const TRASH_SWEEP_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(60 * 60);

/// `?permanent=true` on `DELETE /api/gallery/image/:filename`.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct GalleryDeleteQuery {
    #[serde(default)]
    pub(crate) permanent: Option<bool>,
}

/// Where a gallery filename's bytes live right now: the live path when it
/// exists, else `<dir>/.trash/<name>`. Media, thumbnail, and preview routes
/// resolve through this so a trashed print still renders.
pub(crate) fn resolve_gallery_media_source(dir: &Path, name: &str) -> PathBuf {
    let live = dir.join(name);
    if live.is_file() {
        return live;
    }
    let trashed = batch_transaction::gallery_trash_dir(dir).join(name);
    if trashed.is_file() {
        trashed
    } else {
        live
    }
}

fn not_found(message: impl Into<String>) -> ApiError {
    ApiError::with_code(message, GALLERY_NOT_FOUND, StatusCode::NOT_FOUND)
}

fn internal(context: &str, error: impl std::fmt::Display) -> ApiError {
    ApiError::internal(format!("{context}: {error}"))
}

fn remove_cached_sidecars(name: &str) {
    // Both legacy no-suffix and current `.png`-suffixed thumbnail layouts,
    // plus the animated preview so `/api/gallery/preview/:filename` does
    // not keep serving a purged clip.
    let thumb_dir = crate::routes::server_thumbnail_dir();
    let _ = std::fs::remove_file(thumb_dir.join(name));
    let _ = std::fs::remove_file(thumb_dir.join(format!("{name}.png")));
    let _ = std::fs::remove_file(
        crate::routes::server_preview_gif_dir()
            .join(mold_core::media_paths::preview_gif_filename(name)),
    );
}

/// What a single `DELETE` / trash primitive did, so the handler can publish
/// the matching event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrashOutcome {
    /// Moved to the trash — publish `gallery_trashed`.
    Trashed,
    /// Already in the trash — nothing to announce.
    AlreadyTrashed,
    /// Neither live bytes nor trashed bytes exist; the stale row (if any)
    /// was dropped — publish `gallery_removed`.
    Vanished,
}

/// Import a live file that has no DB row yet (reconcile has not seen it)
/// so the trash index has something to flag.
fn ensure_row_for_live_file(db: &MetadataDb, dir: &Path, name: &str, live_path: &Path) -> bool {
    let Some(format) = mold_db::metadata_io::format_from_path(Path::new(name)) else {
        return false;
    };
    let metadata = std::fs::metadata(live_path).ok();
    let (mtime_ms, size_bytes) = metadata
        .as_ref()
        .map(|m| {
            let size = Some(m.len() as i64);
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64);
            (mtime, size)
        })
        .unwrap_or((None, None));
    let timestamp_secs = mtime_ms.map(|ms| (ms / 1000).max(0)).unwrap_or(0) as u64;
    let (output_metadata, synthetic) =
        mold_db::metadata_io::read_or_synthesize(live_path, format, name, timestamp_secs);
    let mut record = mold_db::GenerationRecord::from_save(
        dir,
        name,
        format,
        output_metadata,
        mold_db::RecordSource::Backfill,
        mold_core::time::now_epoch_ms(),
    );
    record.file_mtime_ms = mtime_ms;
    record.file_size_bytes = size_bytes;
    record.metadata_synthetic = synthetic;
    db.upsert(&record).is_ok()
}

/// Move one print to the trash. Caller holds the gallery writer.
pub(crate) fn trash_print_blocking(
    dir: &Path,
    name: &str,
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    now_ms: i64,
) -> Result<TrashOutcome, ApiError> {
    let row = db
        .get(dir, name)
        .map_err(|e| internal("metadata DB read failed", format!("{e:#}")))?;
    if row.as_ref().is_some_and(|r| r.trashed_at_ms.is_some()) {
        return Ok(TrashOutcome::AlreadyTrashed);
    }
    let live_path = dir.join(name);
    let trash_dir = batch_transaction::gallery_trash_dir(dir);
    let live_exists = live_path.is_file();
    let already_in_trash = trash_dir.join(name).is_file();
    if row.is_none() && !live_exists {
        // No row and no live bytes: nothing to trash. (A tombstoned orphan
        // in `.trash/` is reconcile's to import, not a trash target.)
        return Err(not_found(format!("gallery print not found: {name}")));
    }
    if row.is_some() && !live_exists && !already_in_trash {
        // The bytes are gone from both places: this is a delete of a print
        // that no longer exists. Drop the stale row instead of inventing a
        // trash entry reconcile would discard anyway.
        let _ = db.delete(dir, name);
        return Ok(TrashOutcome::Vanished);
    }
    if row.is_none() && live_exists && !ensure_row_for_live_file(db, dir, name, &live_path) {
        return Err(ApiError::internal(format!(
            "could not record {name} in the metadata DB before trashing it"
        )));
    }

    let disposition = batch_transaction::trash_committed_archive_filename(dir, name, gate)
        .map_err(|e| {
            internal(
                "failed to retire committed gallery metadata before trashing",
                format!("{e:#}"),
            )
        })?;
    match disposition {
        TrashArchiveDisposition::PreservedReplacement => {
            return Err(ApiError::with_code(
                "gallery file changed since publication; the replacement was preserved and quarantined",
                GALLERY_DELETE_IDENTITY_CHANGED,
                StatusCode::CONFLICT,
            ));
        }
        TrashArchiveDisposition::Moved => {}
        TrashArchiveDisposition::NoArchive => {
            if live_path.is_file() {
                batch_transaction::move_gallery_file_to_trash(dir, &live_path, name)
                    .map_err(|e| internal("failed to move print to the trash", format!("{e:#}")))?;
            }
            gate.retire_committed_filename(name);
        }
    }

    let tombstone = db
        .build_tombstone(dir, name, now_ms)
        .map_err(|e| internal("failed to build trash tombstone", format!("{e:#}")))?;
    if let Some(tombstone) = tombstone {
        mold_db::trash::write_tombstone(&trash_dir, &tombstone)
            .map_err(|e| internal("failed to write trash tombstone", format!("{e:#}")))?;
    }
    db.mark_trashed(dir, name, now_ms)
        .map_err(|e| internal("failed to flag print as trashed", format!("{e:#}")))?;
    // Thumbnails and previews are deliberately kept: the trash view renders
    // them through the same routes, resolved into `.trash/`.
    Ok(TrashOutcome::Trashed)
}

/// Restore one trashed print. Caller holds the gallery writer. Returns the
/// enriched live row.
pub(crate) fn restore_print_blocking(
    dir: &Path,
    name: &str,
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    retention_days: u32,
) -> Result<Option<GalleryImage>, ApiError> {
    let row = db
        .get(dir, name)
        .map_err(|e| internal("metadata DB read failed", format!("{e:#}")))?
        .ok_or_else(|| not_found(format!("gallery print not found: {name}")))?;
    if row.trashed_at_ms.is_none() {
        return Err(ApiError::with_code(
            format!("{name} is not in the trash"),
            GALLERY_NOT_TRASHED,
            StatusCode::CONFLICT,
        ));
    }
    let trash_dir = batch_transaction::gallery_trash_dir(dir);
    let trash_path = trash_dir.join(name);
    if !trash_path.is_file() {
        return Err(not_found(format!(
            "trashed bytes for {name} are missing from the gallery trash"
        )));
    }
    let live_path = dir.join(name);
    if std::fs::symlink_metadata(&live_path).is_ok() {
        return Err(ApiError::with_code(
            format!("a live gallery file already exists at {name}; restore would overwrite it"),
            GALLERY_RESTORE_CONFLICT,
            StatusCode::CONFLICT,
        ));
    }
    let disposition =
        batch_transaction::restore_trashed_archive_filename(dir, name, gate, &trash_path).map_err(
            |e| {
                internal(
                    "failed to restore committed gallery metadata",
                    format!("{e:#}"),
                )
            },
        )?;
    match disposition {
        RestoreArchiveDisposition::Conflict => {
            return Err(ApiError::with_code(
                format!("trashed bytes for {name} no longer match their published identity"),
                GALLERY_RESTORE_CONFLICT,
                StatusCode::CONFLICT,
            ));
        }
        RestoreArchiveDisposition::Restored => {}
        RestoreArchiveDisposition::NoArchive => {
            batch_transaction::move_gallery_file_from_trash(dir, &trash_path, name)
                .map_err(|e| internal("failed to move print out of the trash", format!("{e:#}")))?;
            gate.unretire_committed_filename(name);
        }
    }
    if let Err(error) = mold_db::trash::remove_tombstone(&trash_dir, name) {
        tracing::warn!(file = %name, %error, "restored print but could not remove its tombstone");
    }
    db.mark_restored(dir, name)
        .map_err(|e| internal("failed to clear the trashed flag", format!("{e:#}")))?;
    let index = gate
        .committed_archive_index(dir)
        .map_err(|e| internal("gallery archive read failed", format!("{e:#}")))?;
    enriched_gallery_image(db, Some(&index), dir, name, retention_days)
        .map_err(|e| internal("gallery enrichment failed", format!("{e:#}")))
}

/// Permanently delete a TRASHED print: bytes, tombstone, cached sidecars,
/// row, and the committed-archive retirement projection. Caller holds the
/// gallery writer.
pub(crate) fn purge_trashed_print_blocking(
    dir: &Path,
    name: &str,
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
) -> Result<(), ApiError> {
    let trash_dir = batch_transaction::gallery_trash_dir(dir);
    let trash_path = trash_dir.join(name);
    match std::fs::remove_file(&trash_path) {
        Ok(()) => {
            batch_transaction::sync_ordinary_gallery_directory(&trash_dir)
                .map_err(|e| internal("failed to make trash purge durable", format!("{e:#}")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(internal("failed to delete trashed print", error));
        }
    }
    if let Err(error) = mold_db::trash::remove_tombstone(&trash_dir, name) {
        tracing::warn!(file = %name, %error, "purged print but could not remove its tombstone");
    }
    remove_cached_sidecars(name);
    let projection_complete = match db.delete(dir, name) {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(file = %name, %error, "metadata DB delete failed after purge");
            false
        }
    };
    if projection_complete {
        gate.acknowledge_retirement_projections(dir, [name.to_owned()])
            .map_err(|e| {
                internal(
                    "failed to checkpoint gallery deletion projection",
                    format!("{e:#}"),
                )
            })?;
    }
    Ok(())
}

/// The historical hard delete of a LIVE print (bytes, sidecars, row,
/// archive tombstone). Caller holds the gallery writer. This is the whole
/// behaviour of `DELETE /api/gallery/image/:filename` when the metadata DB
/// is disabled, and of `?permanent=true` on a live row.
pub(crate) fn hard_delete_live_print_blocking(
    dir: &Path,
    name: &str,
    db: Option<&MetadataDb>,
    gate: &GalleryPublicationGate,
) -> Result<(), ApiError> {
    let path = dir.join(name);
    let archive_disposition = batch_transaction::tombstone_committed_archive_filename(
        dir, name, gate,
    )
    .map_err(|error| {
        ApiError::internal(format!(
            "failed to retire committed gallery metadata before delete: {error:#}"
        ))
    })?;
    if archive_disposition == ArchiveDeleteDisposition::PreservedReplacement {
        return Err(ApiError::with_code(
            "gallery file changed since publication; the replacement was preserved and quarantined",
            GALLERY_DELETE_IDENTITY_CHANGED,
            StatusCode::CONFLICT,
        ));
    }
    if path.is_file() {
        std::fs::remove_file(&path)
            .map_err(|e| ApiError::internal(format!("failed to delete image: {e}")))?;
        batch_transaction::sync_ordinary_gallery_directory(dir).map_err(|error| {
            ApiError::internal(format!(
                "failed to make gallery deletion durable: {error:#}"
            ))
        })?;
    }
    if archive_disposition == ArchiveDeleteDisposition::NoArchive {
        gate.retire_committed_filename(name);
    }
    remove_cached_sidecars(name);

    // Drop the matching metadata row if the DB is enabled. Errors here are
    // logged — they don't roll back the disk delete since the file is the
    // source of truth and reconciliation will re-sync on the next restart.
    let projection_complete = if let Some(db) = db {
        match db.delete(dir, name) {
            Ok(true) => true,
            Ok(false) => {
                tracing::debug!("delete: no metadata row for {}", dir.join(name).display());
                true
            }
            Err(e) => {
                tracing::warn!(
                    "metadata DB delete failed for {}: {e:#}",
                    dir.join(name).display()
                );
                false
            }
        }
    } else {
        true
    };
    if projection_complete && archive_disposition == ArchiveDeleteDisposition::SafeToUnlink {
        gate.acknowledge_retirement_projections(dir, [name.to_owned()])
            .map_err(|error| {
                ApiError::internal(format!(
                    "failed to checkpoint gallery deletion projection: {error:#}"
                ))
            })?;
    }
    Ok(())
}

// ── DELETE /api/gallery/image/:filename ─────────────────────────────────────

/// Move a gallery print to the trash, or (`?permanent=true`, or when the
/// metadata DB is disabled) delete it for good along with its cached
/// thumbnail and preview.
///
/// Destructive, but always enabled — pair with the `MOLD_API_KEY` middleware
/// when the server is exposed beyond localhost.
#[utoipa::path(
    delete,
    path = "/api/gallery/image/{filename}",
    tag = "gallery",
    params(
        ("filename" = String, Path, description = "Gallery filename"),
        ("permanent" = Option<bool>, Query, description = "Delete for good instead of moving to the trash"),
    ),
    responses(
        (status = 204, description = "Trashed (gallery_trashed) or permanently removed (gallery_removed)"),
        (status = 404, description = "Output disabled, or no such print"),
        (status = 409, description = "The live file changed since publication; preserved and quarantined"),
        (status = 422, description = "Invalid filename"),
    )
)]
pub(crate) async fn delete_gallery_image(
    State(state): State<AppState>,
    AxumPath(filename): AxumPath<String>,
    Query(query): Query<GalleryDeleteQuery>,
) -> Result<StatusCode, ApiError> {
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let dir = gallery_output_dir(&state).await?;
    let name = clean_gallery_filename(&filename)?;
    let permanent = query.permanent.unwrap_or(false);
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let task_name = name.clone();
    let event = tokio::task::spawn_blocking(move || -> Result<Option<ServerEvent>, ApiError> {
        let Some(db) = db.as_ref().as_ref() else {
            hard_delete_live_print_blocking(&dir, &task_name, None, &gate)?;
            return Ok(Some(ServerEvent::GalleryRemoved {
                filename: task_name,
            }));
        };
        if permanent {
            let trashed = db
                .get(&dir, &task_name)
                .map_err(|e| internal("metadata DB read failed", format!("{e:#}")))?
                .is_some_and(|row| row.trashed_at_ms.is_some());
            if trashed {
                purge_trashed_print_blocking(&dir, &task_name, db, &gate)?;
            } else {
                hard_delete_live_print_blocking(&dir, &task_name, Some(db), &gate)?;
            }
            return Ok(Some(ServerEvent::GalleryRemoved {
                filename: task_name,
            }));
        }
        let now_ms = mold_core::time::now_epoch_ms();
        Ok(
            match trash_print_blocking(&dir, &task_name, db, &gate, now_ms)? {
                TrashOutcome::Trashed => Some(ServerEvent::GalleryTrashed {
                    filename: task_name,
                }),
                TrashOutcome::AlreadyTrashed => None,
                TrashOutcome::Vanished => Some(ServerEvent::GalleryRemoved {
                    filename: task_name,
                }),
            },
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery delete task failed: {e}")))??;
    if let Some(event) = event {
        state.events.publish(event);
    }
    Ok(StatusCode::NO_CONTENT)
}

// ── POST /api/gallery/trash ─────────────────────────────────────────────────

fn clean_filenames(request: &TrashFilenamesRequest) -> Result<Vec<String>, ApiError> {
    if request.filenames.is_empty() {
        return Err(ApiError::validation("filenames must not be empty"));
    }
    request
        .filenames
        .iter()
        .map(|f| clean_gallery_filename(f))
        .collect()
}

fn name_failure(name: &str, error: ApiError) -> ApiError {
    ApiError::with_code(
        format!("{name}: {}", error.error),
        error.code.clone(),
        error.status(),
    )
}

/// Move several prints to the trash. Stops at the first failure (naming the
/// filename); prints trashed before it stay trashed and are announced.
#[utoipa::path(
    post,
    path = "/api/gallery/trash",
    tag = "gallery",
    request_body = TrashFilenamesRequest,
    responses(
        (status = 204, description = "Every listed print is in the trash (gallery_trashed per file)"),
        (status = 404, description = "A print does not exist; earlier ones were trashed"),
        (status = 409, description = "A live file changed since publication"),
        (status = 422, description = "Empty list or invalid filename"),
        (status = 501, description = "Metadata DB disabled — trash unavailable"),
    )
)]
pub(crate) async fn trash_gallery_files(
    State(state): State<AppState>,
    Json(request): Json<TrashFilenamesRequest>,
) -> Result<StatusCode, ApiError> {
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let gate = state.gallery_publication_gate.clone();
    let (events, failure) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<ServerEvent>, Option<ApiError>), ApiError> {
            let db = require_metadata_db(&db)?;
            let now_ms = mold_core::time::now_epoch_ms();
            let mut events = Vec::new();
            for name in names {
                match trash_print_blocking(&dir, &name, db, &gate, now_ms) {
                    Ok(TrashOutcome::Trashed) => {
                        events.push(ServerEvent::GalleryTrashed { filename: name })
                    }
                    Ok(TrashOutcome::AlreadyTrashed) => {}
                    Ok(TrashOutcome::Vanished) => {
                        events.push(ServerEvent::GalleryRemoved { filename: name })
                    }
                    Err(error) => return Ok((events, Some(name_failure(&name, error)))),
                }
            }
            Ok((events, None))
        },
    )
    .await
    .map_err(|e| ApiError::internal(format!("gallery trash task failed: {e}")))??;
    for event in events {
        state.events.publish(event);
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(StatusCode::NO_CONTENT),
    }
}

// ── POST /api/gallery/trash/restore ─────────────────────────────────────────

/// Restore prints from the trash to the live gallery. Stops at the first
/// failure (naming the filename); earlier restores stand.
#[utoipa::path(
    post,
    path = "/api/gallery/trash/restore",
    tag = "gallery",
    request_body = TrashFilenamesRequest,
    responses(
        (status = 204, description = "Every listed print is live again (gallery_restored per file)"),
        (status = 404, description = "A print has no row or its trashed bytes are missing"),
        (status = 409, description = "A live file already claims the name, or the bytes no longer match their published identity"),
        (status = 422, description = "Empty list or invalid filename"),
        (status = 501, description = "Metadata DB disabled — trash unavailable"),
    )
)]
pub(crate) async fn restore_gallery_files(
    State(state): State<AppState>,
    Json(request): Json<TrashFilenamesRequest>,
) -> Result<StatusCode, ApiError> {
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let retention = current_retention_days(&state).await;
    let gate = state.gallery_publication_gate.clone();
    let (events, failure) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<ServerEvent>, Option<ApiError>), ApiError> {
            let db = require_metadata_db(&db)?;
            let mut events = Vec::new();
            for name in names {
                match restore_print_blocking(&dir, &name, db, &gate, retention) {
                    Ok(image) => events.push(ServerEvent::GalleryRestored {
                        filename: name,
                        image: image.map(Box::new),
                    }),
                    Err(error) => return Ok((events, Some(name_failure(&name, error)))),
                }
            }
            Ok((events, None))
        },
    )
    .await
    .map_err(|e| ApiError::internal(format!("gallery restore task failed: {e}")))??;
    for event in events {
        state.events.publish(event);
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(StatusCode::NO_CONTENT),
    }
}

// ── POST /api/gallery/trash/delete-forever ────────────────────────────────

/// Permanently delete several live or trashed prints through one host call.
/// Stops at the first failure and names it; earlier removals stay removed.
#[utoipa::path(
    post,
    path = "/api/gallery/trash/delete-forever",
    tag = "gallery",
    request_body = TrashFilenamesRequest,
    responses(
        (status = 204, description = "Every listed print was permanently removed"),
        (status = 404, description = "A print does not exist"),
        (status = 409, description = "A live file changed since publication"),
        (status = 422, description = "Empty list or invalid filename"),
    )
)]
pub(crate) async fn delete_gallery_files_forever(
    State(state): State<AppState>,
    Json(request): Json<TrashFilenamesRequest>,
) -> Result<StatusCode, ApiError> {
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let (removed, failure) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<String>, Option<ApiError>), ApiError> {
            let mut removed = Vec::new();
            for name in names {
                let outcome = if let Some(db) = db.as_ref().as_ref() {
                    let trashed = db
                        .get(&dir, &name)
                        .map_err(|error| internal("metadata DB read failed", format!("{error:#}")))?
                        .is_some_and(|row| row.trashed_at_ms.is_some());
                    if trashed {
                        purge_trashed_print_blocking(&dir, &name, db, &gate)
                    } else {
                        hard_delete_live_print_blocking(&dir, &name, Some(db), &gate)
                    }
                } else {
                    hard_delete_live_print_blocking(&dir, &name, None, &gate)
                };
                match outcome {
                    Ok(()) => removed.push(name),
                    Err(error) => return Ok((removed, Some(name_failure(&name, error)))),
                }
            }
            Ok((removed, None))
        },
    )
    .await
    .map_err(|error| ApiError::internal(format!("gallery bulk delete task failed: {error}")))??;
    for filename in removed {
        state
            .events
            .publish(ServerEvent::GalleryRemoved { filename });
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(StatusCode::NO_CONTENT),
    }
}

// ── DELETE /api/gallery/trash ───────────────────────────────────────────────

/// Purge every trashed print now.
#[utoipa::path(
    delete,
    path = "/api/gallery/trash",
    tag = "gallery",
    responses(
        (status = 200, description = "How many prints were purged (gallery_removed per file)", body = EmptyTrashResult),
        (status = 404, description = "Output disabled"),
        (status = 501, description = "Metadata DB disabled — trash unavailable"),
    )
)]
pub(crate) async fn empty_gallery_trash(
    State(state): State<AppState>,
) -> Result<Json<EmptyTrashResult>, ApiError> {
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let dir = gallery_output_dir(&state).await?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let gate = state.gallery_publication_gate.clone();
    let purged = tokio::task::spawn_blocking(move || -> Result<Vec<String>, ApiError> {
        let db = require_metadata_db(&db)?;
        let rows = db
            .list_trashed(Some(&dir))
            .map_err(|e| internal("metadata DB read failed", format!("{e:#}")))?;
        let mut purged = Vec::new();
        for row in rows {
            purge_trashed_print_blocking(&dir, &row.filename, db, &gate)?;
            purged.push(row.filename);
        }
        Ok(purged)
    })
    .await
    .map_err(|e| ApiError::internal(format!("empty trash task failed: {e}")))??;
    let count = purged.len() as u64;
    for filename in purged {
        state
            .events
            .publish(ServerEvent::GalleryRemoved { filename });
    }
    Ok(Json(EmptyTrashResult { purged: count }))
}

// ── POST /api/gallery/trash/sweep + the sweeper ─────────────────────────────

/// Run one retention sweep now.
#[utoipa::path(
    post,
    path = "/api/gallery/trash/sweep",
    tag = "gallery",
    responses(
        (status = 200, description = "Purged and remaining counts (gallery_removed per purged file)", body = TrashSweepResult),
        (status = 404, description = "Output disabled"),
        (status = 501, description = "Metadata DB disabled — trash unavailable"),
    )
)]
pub(crate) async fn sweep_gallery_trash(
    State(state): State<AppState>,
) -> Result<Json<TrashSweepResult>, ApiError> {
    gallery_output_dir(&state).await?;
    require_metadata_db(&state.metadata_db)?;
    let result = sweep_trash_once(&state)
        .await
        .map_err(|e| ApiError::internal(format!("trash sweep failed: {e:#}")))?;
    Ok(Json(result))
}

/// One retention pass: purge every trashed print whose
/// `gallery.trash_retention_days` (read fresh from the live config; `0`
/// keeps forever) has elapsed. Takes the gallery writer for the pass and
/// publishes `gallery_removed` for each purged print.
pub(crate) async fn sweep_trash_once(state: &AppState) -> anyhow::Result<TrashSweepResult> {
    let (dir, retention) = {
        let config = state.config.read().await;
        if state.is_output_disabled(&config) {
            return Ok(TrashSweepResult::default());
        }
        (
            config.effective_output_dir(),
            config.gallery.effective_trash_retention_days(),
        )
    };
    let db = state.metadata_db.clone();
    if db.as_ref().is_none() {
        return Ok(TrashSweepResult::default());
    }
    let gate = state.gallery_publication_gate.clone();
    let _gallery_writer = state.gallery_publication_gate.write().await;
    let (purged, remaining) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<String>, u64)> {
            let Some(db) = db.as_ref().as_ref() else {
                return Ok((Vec::new(), 0));
            };
            let now_ms = mold_core::time::now_epoch_ms();
            let expired = db.expired_trashed(&dir, retention, now_ms)?;
            let mut purged = Vec::with_capacity(expired.len());
            for row in expired {
                match purge_trashed_print_blocking(&dir, &row.filename, db, &gate) {
                    Ok(()) => purged.push(row.filename),
                    Err(error) => tracing::warn!(
                        file = %row.filename,
                        error = %error.error,
                        "trash sweep could not purge an expired print"
                    ),
                }
            }
            let remaining = db.list_trashed(Some(&dir))?.len() as u64;
            Ok((purged, remaining))
        })
        .await??;
    let count = purged.len() as u64;
    for filename in purged {
        state
            .events
            .publish(ServerEvent::GalleryRemoved { filename });
    }
    Ok(TrashSweepResult {
        purged: count,
        remaining,
    })
}

/// Background retention sweeper: one pass at startup (after `ready`
/// resolves — the gallery reconcile handle — so the trash index is settled
/// first), then hourly, until `shutdown` is cancelled.
pub(crate) fn spawn_trash_sweeper(
    state: AppState,
    shutdown: tokio_util::sync::CancellationToken,
    ready: Option<tokio::sync::oneshot::Receiver<()>>,
) -> tokio::task::JoinHandle<()> {
    use tokio::time::{interval, MissedTickBehavior};
    tokio::spawn(async move {
        if let Some(ready) = ready {
            tokio::select! {
                _ = ready => {}
                _ = shutdown.cancelled() => return,
            }
        }
        let mut tick = interval(TRASH_SWEEP_INTERVAL);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = tick.tick() => {}
            }
            match sweep_trash_once(&state).await {
                Ok(result) if result.purged > 0 => tracing::info!(
                    purged = result.purged,
                    remaining = result.remaining,
                    "gallery trash sweep purged expired prints"
                ),
                Ok(result) => tracing::debug!(
                    remaining = result.remaining,
                    "gallery trash sweep found nothing to purge"
                ),
                Err(error) => tracing::warn!(%error, "gallery trash sweep failed"),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_source_prefers_live_then_trash() {
        let dir = tempfile::tempdir().unwrap();
        let trash = batch_transaction::gallery_trash_dir(dir.path());
        std::fs::create_dir_all(&trash).unwrap();
        std::fs::write(trash.join("only-trashed.png"), b"t").unwrap();
        std::fs::write(dir.path().join("live.png"), b"l").unwrap();
        std::fs::write(trash.join("live.png"), b"stale").unwrap();

        assert_eq!(
            resolve_gallery_media_source(dir.path(), "live.png"),
            dir.path().join("live.png")
        );
        assert_eq!(
            resolve_gallery_media_source(dir.path(), "only-trashed.png"),
            trash.join("only-trashed.png")
        );
        // Neither exists: the live path is returned so callers 404 on it.
        assert_eq!(
            resolve_gallery_media_source(dir.path(), "missing.png"),
            dir.path().join("missing.png")
        );
    }
}
