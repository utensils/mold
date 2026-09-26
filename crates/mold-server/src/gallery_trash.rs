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

/// `?view=trash` on the media, thumbnail, and preview routes: ask for the
/// `.trash/` bytes even when a NEW live file has since landed under the same
/// name. The native `mold-local:` protocol already took the same switch; the
/// HTTP routes resolved live-first with no way to ask otherwise, so a Trash
/// row on a remote machine could show its live twin's pixels.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct GalleryMediaQuery {
    #[serde(default)]
    pub(crate) view: Option<String>,
}

impl GalleryMediaQuery {
    pub(crate) fn reads_trash(&self) -> Result<bool, ApiError> {
        Ok(crate::routes::GalleryView::parse(self.view.as_deref())?
            == crate::routes::GalleryView::Trash)
    }
}

/// Where a gallery filename's bytes live for the view the client asked for:
/// the trash view answers ONLY `<dir>/.trash/<name>` (so a missing trashed
/// file 404s rather than falling back to a live twin); the library view is
/// [`resolve_gallery_media_source`]'s live-then-trash answer.
pub(crate) fn resolve_gallery_media_for_view(dir: &Path, name: &str, from_trash: bool) -> PathBuf {
    if from_trash {
        batch_transaction::gallery_trash_dir(dir).join(name)
    } else {
        resolve_gallery_media_source(dir, name)
    }
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
    // A mesh poster is not at `<name>.png`: its name carries the poster
    // renderer's revision. Purging through the one place that name is defined
    // is what keeps a purge from leaving a poster behind for every 3-D print
    // ever deleted.
    let _ = std::fs::remove_file(mold_core::media_paths::mesh_poster_thumbnail_path(
        &thumb_dir, name,
    ));
    // `<name>.thumb.png` is the retired terminal app's sidecar, for a mesh
    // poster or an audio waveform alike. Nothing writes one any more and the
    // orphan sweeper clears the backlog, but a purge must still take the one
    // belonging to THIS print: after the file is gone the sweeper can still
    // reach it, yet leaving dead bytes behind until the next pass is exactly
    // what a permanent delete promises not to do. A raster print has neither
    // name, so both removals are no-ops for one.
    let _ = std::fs::remove_file(thumb_dir.join(format!("{name}.thumb.png")));
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

/// "Save every result" off: move a just-published print (and its
/// pre-upscale original, when one was kept) straight to the trash. Caller
/// holds the gallery writer, exactly as for a user's own Delete. A failure
/// is logged and never fails the render: the print reached the gallery, and
/// a stray live copy is the lesser wrong.
pub(crate) fn trash_published_outputs_blocking(
    dir: &Path,
    saved: &crate::queue::SavedOutputNames,
    db: Option<&MetadataDb>,
    gate: &GalleryPublicationGate,
    events: Option<&crate::events::EventBroadcaster>,
) {
    let Some(db) = db else {
        tracing::warn!("save_to_gallery=false: no metadata DB, the print stays live");
        return;
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    for name in [saved.output.as_deref(), saved.original.as_deref()]
        .into_iter()
        .flatten()
    {
        match trash_print_blocking(dir, name, db, gate, now_ms) {
            // The SAME event a user's own Delete publishes, mapped from the
            // same outcome: `gallery_trashed` means "moved, recoverable" and
            // is what puts the row in a client's Trash scope, while
            // `gallery_removed` means "gone" and drops it from both scopes.
            // Announcing a removal here made an opt-out print vanish from
            // desktop state as if it had been purged.
            Ok(TrashOutcome::Trashed) => {
                if let Some(events) = events {
                    events.publish(mold_core::ServerEvent::GalleryTrashed {
                        filename: name.to_string(),
                    });
                }
            }
            // Already trashed: nothing moved, so nothing to announce.
            Ok(TrashOutcome::AlreadyTrashed) => {}
            // The bytes were gone from both places and the stale row was
            // dropped — that IS a removal.
            Ok(TrashOutcome::Vanished) => {
                if let Some(events) = events {
                    events.publish(mold_core::ServerEvent::GalleryRemoved {
                        filename: name.to_string(),
                    });
                }
            }
            Err(error) => {
                tracing::warn!(filename = name, error = %error.error, "save_to_gallery=false: could not move the print to the trash");
            }
        }
    }
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
            gate.retire_committed_filename(dir, name);
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
            gate.unretire_committed_filename(dir, name);
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
    media_lifecycle: Option<&crate::queue_media_lifecycle::QueueMediaLifecycle>,
) -> Result<(), ApiError> {
    batch_transaction::retire_trashed_archive_filename(dir, name, gate).map_err(|e| {
        internal(
            "failed to retire trashed gallery authority",
            format!("{e:#}"),
        )
    })?;
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
        Ok(_) => match remove_gallery_media_projection(db, dir, name) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(file = %name, %error, "retained-media DB projection removal failed after purge");
                false
            }
        },
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
        release_retired_media_pins(dir, name, gate, media_lifecycle)?;
    }
    Ok(())
}

/// Revalidate a row selected by an earlier empty/sweep snapshot while the
/// caller holds the gallery writer. A restore may have won between listing
/// and this item; in that case purging the stale candidate would delete its
/// DB row and retained-media pins while leaving its restored live bytes.
fn purge_if_still_trashed_blocking(
    dir: &Path,
    name: &str,
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    media_lifecycle: Option<&crate::queue_media_lifecycle::QueueMediaLifecycle>,
    expired_by: Option<(u32, i64)>,
) -> Result<bool, ApiError> {
    let row = db
        .get(dir, name)
        .map_err(|error| internal("metadata DB read failed", format!("{error:#}")))?;
    let Some(trashed_at) = row.and_then(|row| row.trashed_at_ms) else {
        return Ok(false);
    };
    if let Some((retention, now_ms)) = expired_by {
        let expired = mold_db::trash::purge_at_ms(trashed_at, retention)
            .is_some_and(|purge_at| purge_at <= now_ms);
        if !expired {
            return Ok(false);
        }
    }
    purge_trashed_print_blocking(dir, name, db, gate, media_lifecycle)?;
    Ok(true)
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
    media_lifecycle: Option<&crate::queue_media_lifecycle::QueueMediaLifecycle>,
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
        gate.retire_committed_filename(dir, name);
    }
    remove_cached_sidecars(name);

    // Drop the matching metadata row if the DB is enabled. Errors here are
    // logged — they don't roll back the disk delete since the file is the
    // source of truth and reconciliation will re-sync on the next restart.
    let projection_complete = if let Some(db) = db {
        match db.delete(dir, name) {
            Ok(true) => match remove_gallery_media_projection(db, dir, name) {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(file = %name, %error, "retained-media DB projection removal failed after delete");
                    false
                }
            },
            Ok(false) => {
                tracing::debug!("delete: no metadata row for {}", dir.join(name).display());
                match remove_gallery_media_projection(db, dir, name) {
                    Ok(()) => true,
                    Err(error) => {
                        tracing::warn!(file = %name, %error, "retained-media DB projection removal failed after delete");
                        false
                    }
                }
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
        release_retired_media_pins(dir, name, gate, media_lifecycle)?;
    }
    Ok(())
}

fn remove_gallery_media_projection(db: &MetadataDb, dir: &Path, name: &str) -> anyhow::Result<()> {
    let canonical = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    mold_db::gallery_media::replace_for_item(db, &canonical.to_string_lossy(), name, &[])
}

fn release_retired_media_pins(
    dir: &Path,
    name: &str,
    gate: &GalleryPublicationGate,
    lifecycle: Option<&crate::queue_media_lifecycle::QueueMediaLifecycle>,
) -> Result<(), ApiError> {
    let pins = gate
        .finalize_retained_media_release(dir, [name.to_owned()])
        .map_err(|error| {
            internal(
                "failed to commit retained source-media release",
                format!("{error:#}"),
            )
        })?;
    let Some(lifecycle) = lifecycle else {
        // Authority already excludes these pins. A future complete startup
        // scan may collect the encrypted orphans; never guess without one.
        return Ok(());
    };
    for pin in pins {
        if let Err(error) = lifecycle.release_gallery_pin(pin.media_set, pin.pin_id) {
            tracing::warn!(file = %name, %error, "retained source-media pin release will be reconciled on startup");
        }
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
    let dir = gallery_output_dir(&state).await?;
    let name = clean_gallery_filename(&filename)?;
    let permanent = query.permanent.unwrap_or(false);
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let media_lifecycle = state.queue_journal.queue_media_lifecycle();
    let task_name = name.clone();
    let gallery_writer = state.gallery_publication_gate.write().await;
    let event = tokio::task::spawn_blocking(move || -> Result<Option<ServerEvent>, ApiError> {
        let _gallery_writer = gallery_writer;
        let Some(db) = db.as_ref().as_ref() else {
            hard_delete_live_print_blocking(
                &dir,
                &task_name,
                None,
                &gate,
                media_lifecycle.as_deref(),
            )?;
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
                purge_trashed_print_blocking(
                    &dir,
                    &task_name,
                    db,
                    &gate,
                    media_lifecycle.as_deref(),
                )?;
            } else {
                hard_delete_live_print_blocking(
                    &dir,
                    &task_name,
                    Some(db),
                    &gate,
                    media_lifecycle.as_deref(),
                )?;
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

/// Bound the time for which one blocking worker excludes gallery readers and
/// bound the recovery/rollback set of one authority commit.
const GALLERY_TRASH_CHUNK_SIZE: usize = 16;

#[cfg(test)]
pub(crate) struct TrashChunkBoundaryHook {
    output_dir: PathBuf,
    reached: tokio::sync::Notify,
    resume: tokio::sync::Notify,
}

#[cfg(test)]
fn trash_chunk_boundary_hook(
) -> &'static std::sync::Mutex<Option<std::sync::Arc<TrashChunkBoundaryHook>>> {
    static HOOK: std::sync::OnceLock<
        std::sync::Mutex<Option<std::sync::Arc<TrashChunkBoundaryHook>>>,
    > = std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(crate) fn install_trash_chunk_boundary_hook(
    output_dir: &Path,
) -> std::sync::Arc<TrashChunkBoundaryHook> {
    let hook = std::sync::Arc::new(TrashChunkBoundaryHook {
        output_dir: std::fs::canonicalize(output_dir).unwrap_or_else(|_| output_dir.to_path_buf()),
        reached: tokio::sync::Notify::new(),
        resume: tokio::sync::Notify::new(),
    });
    *trash_chunk_boundary_hook()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(hook.clone());
    hook
}

#[cfg(test)]
impl TrashChunkBoundaryHook {
    pub(crate) async fn wait_until_reached(&self) {
        self.reached.notified().await;
    }

    pub(crate) fn resume(&self) {
        self.resume.notify_one();
        trash_chunk_boundary_hook()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
    }
}

#[cfg(test)]
async fn pause_at_trash_chunk_boundary(output_dir: &Path) {
    let hook = trash_chunk_boundary_hook()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone();
    let canonical = std::fs::canonicalize(output_dir).unwrap_or_else(|_| output_dir.to_path_buf());
    if let Some(hook) = hook.filter(|hook| hook.output_dir == canonical) {
        hook.reached.notify_one();
        hook.resume.notified().await;
    }
}

fn trash_gallery_chunk_blocking(
    dir: &Path,
    names: &[String],
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    events: &crate::events::EventBroadcaster,
) -> Result<Option<ApiError>, ApiError> {
    enum PreparedAction {
        AlreadyTrashed,
        Vanished(String),
        Candidate(String),
    }

    let now_ms = mold_core::time::now_epoch_ms();
    let trash_dir = batch_transaction::gallery_trash_dir(dir);
    let mut candidates = Vec::with_capacity(names.len());
    let mut actions = Vec::with_capacity(names.len());
    let mut tombstones = std::collections::BTreeMap::new();
    let mut preflight_failure = None;

    // Perform every fallible DB read in request order before moving the
    // candidate prefix. That keeps the documented stop-at-first-error shape
    // while allowing its archive mutations to share one durable commit.
    for name in names {
        let row = match db.get(dir, name) {
            Ok(row) => row,
            Err(error) => {
                preflight_failure = Some(name_failure(
                    name,
                    internal("metadata DB read failed", format!("{error:#}")),
                ));
                break;
            }
        };
        if row.as_ref().is_some_and(|row| row.trashed_at_ms.is_some()) {
            actions.push(PreparedAction::AlreadyTrashed);
            continue;
        }
        let live_path = dir.join(name);
        let live_exists = live_path.is_file();
        let already_in_trash = trash_dir.join(name).is_file();
        if row.is_none() && !live_exists {
            preflight_failure = Some(name_failure(
                name,
                not_found(format!("gallery print not found: {name}")),
            ));
            break;
        }
        if row.is_some() && !live_exists && !already_in_trash {
            // Defer this mutation until the authority prefix before it has
            // succeeded. Preflight must never delete a later row past an
            // earlier identity conflict.
            actions.push(PreparedAction::Vanished(name.clone()));
            continue;
        }
        if row.is_none() && live_exists && !ensure_row_for_live_file(db, dir, name, &live_path) {
            preflight_failure = Some(name_failure(
                name,
                ApiError::internal(format!(
                    "could not record {name} in the metadata DB before trashing it"
                )),
            ));
            break;
        }
        match db.build_tombstone(dir, name, now_ms) {
            Ok(tombstone) => {
                if let Some(tombstone) = tombstone {
                    tombstones.insert(name.clone(), tombstone);
                }
                candidates.push(name.clone());
                actions.push(PreparedAction::Candidate(name.clone()));
            }
            Err(error) => {
                preflight_failure = Some(name_failure(
                    name,
                    internal("failed to build trash tombstone", format!("{error:#}")),
                ));
                break;
            }
        }
    }

    let outcome = if candidates.is_empty() {
        batch_transaction::TrashArchiveBatchOutcome {
            completed: Vec::new(),
            failure: None,
        }
    } else {
        batch_transaction::trash_committed_archive_filenames(dir, &candidates, gate).map_err(
            |error| {
                internal(
                    "failed to retire committed gallery metadata before trashing",
                    format!("{error:#}"),
                )
            },
        )?
    };
    let mut finalization_failure = None;
    let mut completed = outcome.completed.into_iter();
    for action in actions {
        let PreparedAction::Candidate(name) = action else {
            match action {
                PreparedAction::AlreadyTrashed => {}
                PreparedAction::Vanished(name) => {
                    let _ = db.delete(dir, &name);
                    events.publish(ServerEvent::GalleryRemoved { filename: name });
                }
                PreparedAction::Candidate(_) => unreachable!(),
            }
            continue;
        };
        let Some((completed_name, disposition)) = completed.next() else {
            break;
        };
        debug_assert_eq!(completed_name, name);
        if disposition == TrashArchiveDisposition::PreservedReplacement {
            finalization_failure = Some(name_failure(
                &name,
                ApiError::with_code(
                    "gallery file changed since publication; the replacement was preserved and quarantined",
                    GALLERY_DELETE_IDENTITY_CHANGED,
                    StatusCode::CONFLICT,
                ),
            ));
            break;
        }

        // Once the authority commit says the whole prefix is retired, finish
        // every item in that prefix even if a sidecar/DB write fails. Stopping
        // here would knowingly leave later moved bytes unindexed until a
        // restart reconcile. Preserve the first error for the HTTP response.
        if let Some(tombstone) = tombstones.remove(&name) {
            if let Err(error) = mold_db::trash::write_tombstone(&trash_dir, &tombstone) {
                finalization_failure.get_or_insert_with(|| {
                    name_failure(
                        &name,
                        internal("failed to write trash tombstone", format!("{error:#}")),
                    )
                });
                continue;
            }
        }
        match db.mark_trashed(dir, &name, now_ms) {
            Ok(_) => events.publish(ServerEvent::GalleryTrashed { filename: name }),
            Err(error) => {
                finalization_failure.get_or_insert_with(|| {
                    name_failure(
                        &name,
                        internal("failed to flag print as trashed", format!("{error:#}")),
                    )
                });
            }
        }
    }
    if let Some((name, error)) = outcome.failure {
        return Ok(Some(name_failure(
            &name,
            internal("failed to move print to the trash", format!("{error:#}")),
        )));
    }
    Ok(finalization_failure.or(preflight_failure))
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
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let gate = state.gallery_publication_gate.clone();
    let started = std::time::Instant::now();
    let total = names.len();
    let mut completed = 0usize;
    for chunk in names.chunks(GALLERY_TRASH_CHUNK_SIZE) {
        let chunk = chunk.to_vec();
        let chunk_len = chunk.len();
        let chunk_dir = dir.clone();
        let chunk_db = db.clone();
        let chunk_gate = gate.clone();
        let chunk_events = state.events.clone();
        let gallery_writer = gate.write().await;
        let failure = tokio::task::spawn_blocking(move || {
            // Owned by the blocking task, not the HTTP future. If its client
            // times out and Axum drops the waiter, the detached filesystem
            // work remains protected until it actually finishes.
            let _gallery_writer = gallery_writer;
            let db = require_metadata_db(&chunk_db)?;
            trash_gallery_chunk_blocking(&chunk_dir, &chunk, db, &chunk_gate, &chunk_events)
        })
        .await
        .map_err(|e| ApiError::internal(format!("gallery trash task failed: {e}")))??;
        if failure.is_none() {
            completed += chunk_len;
        }
        tracing::info!(
            completed,
            total,
            attempted = chunk_len,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "gallery bulk trash progress"
        );
        if let Some(error) = failure {
            return Err(error);
        }
        #[cfg(test)]
        if completed < total {
            pause_at_trash_chunk_boundary(&dir).await;
        }
        // Give queued readers a chance before the next writer is acquired.
        tokio::task::yield_now().await;
    }
    tracing::info!(
        total,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "gallery bulk trash complete"
    );
    Ok(StatusCode::NO_CONTENT)
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
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let retention = current_retention_days(&state).await;
    let gate = state.gallery_publication_gate.clone();
    let started = std::time::Instant::now();
    let total = names.len();
    for (offset, name) in names.into_iter().enumerate() {
        let item_dir = dir.clone();
        let item_db = db.clone();
        let item_gate = gate.clone();
        let item_events = state.events.clone();
        let item_name = name.clone();
        let gallery_writer = gate.write().await;
        let result = tokio::task::spawn_blocking(move || {
            let _gallery_writer = gallery_writer;
            let db = require_metadata_db(&item_db)?;
            let image = restore_print_blocking(&item_dir, &item_name, db, &item_gate, retention)?;
            item_events.publish(ServerEvent::GalleryRestored {
                filename: item_name,
                image: image.map(Box::new),
            });
            Ok::<_, ApiError>(())
        })
        .await
        .map_err(|error| ApiError::internal(format!("gallery restore task failed: {error}")))?;
        if let Err(error) = result {
            return Err(name_failure(&name, error));
        }
        if (offset + 1) % GALLERY_TRASH_CHUNK_SIZE == 0 || offset + 1 == total {
            tracing::info!(
                completed = offset + 1,
                total,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "gallery bulk restore progress"
            );
        }
        tokio::task::yield_now().await;
    }
    Ok(StatusCode::NO_CONTENT)
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
    let dir = gallery_output_dir(&state).await?;
    let names = clean_filenames(&request)?;
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let media_lifecycle = state.queue_journal.queue_media_lifecycle();
    let started = std::time::Instant::now();
    let total = names.len();
    for (offset, name) in names.into_iter().enumerate() {
        let item_dir = dir.clone();
        let item_db = db.clone();
        let item_gate = gate.clone();
        let item_media_lifecycle = media_lifecycle.clone();
        let item_events = state.events.clone();
        let item_name = name.clone();
        let gallery_writer = gate.write().await;
        let result = tokio::task::spawn_blocking(move || {
            let _gallery_writer = gallery_writer;
            if let Some(db) = item_db.as_ref().as_ref() {
                let trashed = db
                    .get(&item_dir, &item_name)
                    .map_err(|error| internal("metadata DB read failed", format!("{error:#}")))?
                    .is_some_and(|row| row.trashed_at_ms.is_some());
                if trashed {
                    purge_trashed_print_blocking(
                        &item_dir,
                        &item_name,
                        db,
                        &item_gate,
                        item_media_lifecycle.as_deref(),
                    )?;
                } else {
                    hard_delete_live_print_blocking(
                        &item_dir,
                        &item_name,
                        Some(db),
                        &item_gate,
                        item_media_lifecycle.as_deref(),
                    )?;
                }
            } else {
                hard_delete_live_print_blocking(
                    &item_dir,
                    &item_name,
                    None,
                    &item_gate,
                    item_media_lifecycle.as_deref(),
                )?;
            }
            item_events.publish(ServerEvent::GalleryRemoved {
                filename: item_name,
            });
            Ok::<_, ApiError>(())
        })
        .await
        .map_err(|error| ApiError::internal(format!("gallery bulk delete task failed: {error}")))?;
        if let Err(error) = result {
            return Err(name_failure(&name, error));
        }
        if (offset + 1) % GALLERY_TRASH_CHUNK_SIZE == 0 || offset + 1 == total {
            tracing::info!(
                completed = offset + 1,
                total,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "gallery permanent delete progress"
            );
        }
        tokio::task::yield_now().await;
    }
    Ok(StatusCode::NO_CONTENT)
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
    let dir = gallery_output_dir(&state).await?;
    let db = state.metadata_db.clone();
    require_metadata_db(&db)?;
    let gate = state.gallery_publication_gate.clone();
    let media_lifecycle = state.queue_journal.queue_media_lifecycle();
    let list_dir = dir.clone();
    let list_db = db.clone();
    let rows = tokio::task::spawn_blocking(move || {
        let db = require_metadata_db(&list_db)?;
        db.list_trashed(Some(&list_dir))
            .map_err(|error| internal("metadata DB read failed", format!("{error:#}")))
    })
    .await
    .map_err(|error| ApiError::internal(format!("empty trash listing failed: {error}")))??;
    let total = rows.len();
    let started = std::time::Instant::now();
    let mut purged = 0u64;
    for (offset, row) in rows.into_iter().enumerate() {
        let item_dir = dir.clone();
        let item_db = db.clone();
        let item_gate = gate.clone();
        let item_media_lifecycle = media_lifecycle.clone();
        let item_events = state.events.clone();
        let filename = row.filename;
        let gallery_writer = gate.write().await;
        let did_purge = tokio::task::spawn_blocking(move || -> Result<bool, ApiError> {
            let _gallery_writer = gallery_writer;
            let db = require_metadata_db(&item_db)?;
            let purged = purge_if_still_trashed_blocking(
                &item_dir,
                &filename,
                db,
                &item_gate,
                item_media_lifecycle.as_deref(),
                None,
            )?;
            if purged {
                item_events.publish(ServerEvent::GalleryRemoved { filename });
            }
            Ok(purged)
        })
        .await
        .map_err(|error| ApiError::internal(format!("empty trash task failed: {error}")))??;
        if did_purge {
            purged += 1;
        }
        if (offset + 1) % GALLERY_TRASH_CHUNK_SIZE == 0 || offset + 1 == total {
            tracing::info!(
                completed = offset + 1,
                total,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "empty gallery trash progress"
            );
        }
        tokio::task::yield_now().await;
    }
    Ok(Json(EmptyTrashResult { purged }))
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
/// keeps forever) has elapsed. Each expired print owns the writer only for
/// its blocking purge, so listings can run between a large sweep's items.
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
    let media_lifecycle = state.queue_journal.queue_media_lifecycle();
    let list_dir = dir.clone();
    let list_db = db.clone();
    let expired = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let Some(db) = list_db.as_ref().as_ref() else {
            return Ok(Vec::new());
        };
        db.expired_trashed(&list_dir, retention, mold_core::time::now_epoch_ms())
    })
    .await??;
    let mut count = 0u64;
    for row in expired {
        let item_dir = dir.clone();
        let item_db = db.clone();
        let item_gate = gate.clone();
        let item_media_lifecycle = media_lifecycle.clone();
        let item_events = state.events.clone();
        let filename = row.filename;
        let log_name = filename.clone();
        let gallery_writer = gate.write().await;
        let result = tokio::task::spawn_blocking(move || -> Result<bool, ApiError> {
            let _gallery_writer = gallery_writer;
            let db = require_metadata_db(&item_db)?;
            let purged = purge_if_still_trashed_blocking(
                &item_dir,
                &filename,
                db,
                &item_gate,
                item_media_lifecycle.as_deref(),
                Some((retention, mold_core::time::now_epoch_ms())),
            )?;
            if purged {
                item_events.publish(ServerEvent::GalleryRemoved { filename });
            }
            Ok(purged)
        })
        .await?;
        match result {
            Ok(true) => count += 1,
            Ok(false) => {}
            Err(error) => tracing::warn!(
                file = %log_name,
                error = %error.error,
                "trash sweep could not purge an expired print"
            ),
        }
        tokio::task::yield_now().await;
    }
    let remaining_dir = dir.clone();
    let remaining_db = db.clone();
    let remaining = tokio::task::spawn_blocking(move || -> anyhow::Result<u64> {
        let Some(db) = remaining_db.as_ref().as_ref() else {
            return Ok(0);
        };
        Ok(db.list_trashed(Some(&remaining_dir))?.len() as u64)
    })
    .await??;
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

    #[tokio::test]
    async fn detached_blocking_mutation_keeps_its_owned_gallery_writer() {
        let gate = GalleryPublicationGate::default();
        let writer = gate.write().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let waiter = tokio::task::spawn_blocking(move || {
            let _writer = writer;
            let _ = started_tx.send(());
            release_rx.recv().unwrap();
        });
        started_rx.await.unwrap();
        waiter.abort();

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), gate.read())
                .await
                .is_err(),
            "dropping the waiter must not expose an active blocking mutation"
        );
        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), gate.read())
            .await
            .expect("the writer is released when the blocking mutation ends");
    }

    #[test]
    fn stale_empty_snapshot_does_not_purge_a_restored_print() {
        let dir = tempfile::tempdir().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let live = dir.path().join("restored.png");
        std::fs::write(&live, b"restored").unwrap();
        assert!(ensure_row_for_live_file(
            &db,
            dir.path(),
            "restored.png",
            &live
        ));
        db.mark_trashed(dir.path(), "restored.png", 1).unwrap();
        let trash = batch_transaction::ensure_gallery_trash_dir(dir.path()).unwrap();
        std::fs::rename(&live, trash.join("restored.png")).unwrap();

        // Empty Trash listed the row, then Restore won before Empty acquired
        // this item's writer.
        std::fs::rename(trash.join("restored.png"), &live).unwrap();
        db.mark_restored(dir.path(), "restored.png").unwrap();
        let gate = GalleryPublicationGate::default();
        assert!(!purge_if_still_trashed_blocking(
            dir.path(),
            "restored.png",
            &db,
            &gate,
            None,
            None,
        )
        .unwrap());
        assert_eq!(std::fs::read(live).unwrap(), b"restored");
        assert!(db.get(dir.path(), "restored.png").unwrap().is_some());
    }

    #[test]
    fn stale_sweep_snapshot_rechecks_the_current_retention_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let live = dir.path().join("reaged.png");
        std::fs::write(&live, b"reaged").unwrap();
        assert!(ensure_row_for_live_file(
            &db,
            dir.path(),
            "reaged.png",
            &live
        ));
        let trash = batch_transaction::ensure_gallery_trash_dir(dir.path()).unwrap();
        std::fs::rename(&live, trash.join("reaged.png")).unwrap();
        let now = mold_core::time::now_epoch_ms();
        db.mark_trashed(dir.path(), "reaged.png", now).unwrap();
        let gate = GalleryPublicationGate::default();
        assert!(!purge_if_still_trashed_blocking(
            dir.path(),
            "reaged.png",
            &db,
            &gate,
            None,
            Some((30, now)),
        )
        .unwrap());
        assert!(trash.join("reaged.png").is_file());
        assert!(db.get(dir.path(), "reaged.png").unwrap().is_some());
    }

    #[test]
    fn permanent_delete_projection_removes_retained_media_bindings() {
        let dir = tempfile::tempdir().unwrap();
        let db = MetadataDb::open_in_memory().unwrap();
        let canonical = std::fs::canonicalize(dir.path()).unwrap();
        mold_db::gallery_media::replace_for_item(
            &db,
            &canonical.to_string_lossy(),
            "print.png",
            &[mold_db::gallery_media::GalleryMediaBinding {
                output_dir: canonical.to_string_lossy().into_owned(),
                filename: "print.png".into(),
                pin_id: "a".repeat(64),
                media_set_id: "0".repeat(32),
                owner_uuid: "owner".into(),
                job_id: "job".into(),
            }],
        )
        .unwrap();

        remove_gallery_media_projection(&db, dir.path(), "print.png").unwrap();
        assert!(mold_db::gallery_media::list_for_item(
            &db,
            &canonical.to_string_lossy(),
            "print.png"
        )
        .unwrap()
        .is_empty());
    }

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

    /// `?view=trash` reads the trash even when a live twin exists, and never
    /// falls back to that twin when the trashed file is gone.
    #[test]
    fn trash_view_reads_only_the_trash() {
        let dir = tempfile::tempdir().unwrap();
        let trash = batch_transaction::gallery_trash_dir(dir.path());
        std::fs::create_dir_all(&trash).unwrap();
        std::fs::write(dir.path().join("twin.png"), b"new live").unwrap();
        std::fs::write(trash.join("twin.png"), b"old trashed").unwrap();

        assert_eq!(
            resolve_gallery_media_for_view(dir.path(), "twin.png", true),
            trash.join("twin.png")
        );
        assert_eq!(
            resolve_gallery_media_for_view(dir.path(), "twin.png", false),
            dir.path().join("twin.png")
        );
        assert_eq!(
            resolve_gallery_media_for_view(dir.path(), "gone.png", true),
            trash.join("gone.png")
        );
        assert!(!resolve_gallery_media_for_view(dir.path(), "gone.png", true).is_file());

        let query = |view: Option<&str>| GalleryMediaQuery {
            view: view.map(str::to_string),
        };
        assert!(!query(None).reads_trash().unwrap());
        assert!(!query(Some("library")).reads_trash().unwrap());
        assert!(query(Some("trash")).reads_trash().unwrap());
        assert!(query(Some("wat")).reads_trash().is_err());
    }
}
