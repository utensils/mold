//! `GET /api/gallery/changes` — the compacted gallery change feed — and the
//! committed-archive projector that keeps it complete.
//!
//! The feed reads the v43 dirty set (`mold_db::gallery_changes`) in seq
//! order and answers each dirty name with the row the listing would show for
//! it NOW ([`crate::gallery_rows::build_feed_rows`]: library, else trash,
//! else `removed`). SQLite triggers mark every DB-side change; the one
//! writer they cannot see is the committed gallery archive, whose overlay
//! can add, replace, retire, or quarantine a print with no DB write at all.
//! [`project_archive_changes`] closes that gap at feed-read time: every
//! commit this process lands is logged with the names it touched
//! (`gallery_authority::archive_changes_between`), and a generation the log
//! cannot account for — another process's commit, a crash, a cold start —
//! re-marks every name. Over-marking costs clients a re-read; a missed mark
//! would lose a change forever.
//!
//! A cursor is opaque to clients: `epoch . dir . row format . seq`, encoded.
//! A cursor that names another database, directory, or row format, or a seq
//! outside `[min_seq, max_seq]`, gets `reset: true` and a page from the
//! beginning.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    Json,
};
use base64::Engine as _;
use mold_core::{GalleryChange, GalleryChangesPage, GALLERY_CHANGES_PROTOCOL_VERSION};
use mold_db::gallery_changes::ArchiveProjection;
use mold_db::MetadataDb;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::batch_transaction::{CommittedArchiveSnapshot, GalleryPublicationGate};
use crate::routes::ApiError;
use crate::state::AppState;

/// The version of how a [`mold_core::GalleryImage`] is derived from the DB
/// row, the archive, and organization. Bump it whenever that derivation
/// changes for EXISTING prints (a new field, a different title fallback,
/// …): every cursor then resets and clients rebuild, because no seq records
/// a change the code made.
pub(crate) const ROW_FORMAT: u32 = 1;

pub(crate) const DEFAULT_LIMIT: usize = 500;
pub(crate) const MAX_LIMIT: usize = 2000;

pub(crate) const GALLERY_FEED_NOT_READY: &str = "GALLERY_FEED_NOT_READY";
pub(crate) const GALLERY_FEED_UNSUPPORTED: &str = "GALLERY_FEED_UNSUPPORTED";

/// The row-format discriminator carried in every cursor and validator:
/// [`ROW_FORMAT`] plus the mesh poster revision every row's `media_version`
/// is stamped with, so a poster bump resets cursors too.
pub(crate) fn row_format_token() -> String {
    let digest = Sha256::digest(format!(
        "{ROW_FORMAT}|{}",
        crate::thumbnails::MESH_POSTER_REVISION_SUFFIX
    ));
    format!("{digest:x}")[..8].to_string()
}

/// A short, stable digest of the canonical output directory.
pub(crate) fn dir_token(dir: &Path) -> String {
    let digest = Sha256::digest(mold_db::canonical_dir_string(dir).as_bytes());
    format!("{digest:x}")[..16].to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Cursor {
    epoch: String,
    dir: String,
    format: String,
    seq: i64,
}

const CURSOR_TAG: &str = "g1";

impl Cursor {
    fn encode(&self) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(format!(
            "{CURSOR_TAG}.{}.{}.{}.{}",
            self.epoch, self.dir, self.format, self.seq
        ))
    }

    /// `None` for anything this server did not mint.
    fn decode(raw: &str) -> Option<Self> {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(raw.trim())
            .ok()?;
        let text = String::from_utf8(bytes).ok()?;
        let mut parts = text.split('.');
        let (tag, epoch, dir, format, seq) = (
            parts.next()?,
            parts.next()?,
            parts.next()?,
            parts.next()?,
            parts.next()?,
        );
        if parts.next().is_some() || tag != CURSOR_TAG || epoch.is_empty() {
            return None;
        }
        Some(Self {
            epoch: epoch.to_string(),
            dir: dir.to_string(),
            format: format.to_string(),
            seq: seq.parse().ok().filter(|seq: &i64| *seq >= 0)?,
        })
    }
}

/// Whether startup reconcile has finished for an output directory. The feed
/// answers `503 GALLERY_FEED_NOT_READY` until it has: before it, the DB can
/// be missing rows the listing's filesystem fallback would show.
#[derive(Debug, Default)]
pub struct GalleryFeedReadiness {
    inner: std::sync::Mutex<ReadinessInner>,
}

#[derive(Debug, Default)]
struct ReadinessInner {
    ready: HashSet<PathBuf>,
    reconciling: HashSet<PathBuf>,
}

fn readiness_key(dir: &Path) -> PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

impl GalleryFeedReadiness {
    fn lock(&self) -> std::sync::MutexGuard<'_, ReadinessInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// A reconcile of `dir` has started.
    pub(crate) fn begin(&self, dir: &Path) {
        self.lock().reconciling.insert(readiness_key(dir));
    }

    /// The reconcile of `dir` has finished (successfully or not: a failed
    /// reconcile leaves the DB exactly as authoritative as the listing
    /// treats it).
    pub(crate) fn finish(&self, dir: &Path) {
        let key = readiness_key(dir);
        let mut inner = self.lock();
        inner.reconciling.remove(&key);
        inner.ready.insert(key);
    }

    pub(crate) fn is_ready(&self, dir: &Path) -> bool {
        self.lock().ready.contains(&readiness_key(dir))
    }

    /// Claim the reconcile of a directory nobody has reconciled or is
    /// reconciling — an `output_dir` changed through the config API after
    /// startup. True exactly once per such directory.
    fn claim_reconcile(&self, dir: &Path) -> bool {
        let key = readiness_key(dir);
        let mut inner = self.lock();
        if inner.ready.contains(&key) || inner.reconciling.contains(&key) {
            return false;
        }
        inner.reconciling.insert(key);
        true
    }
}

/// Reconcile a directory the server did not reconcile at startup, then mark
/// it ready. Same work and lock as the startup reconcile.
fn spawn_reconcile(state: &AppState, dir: PathBuf) {
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let readiness = state.gallery_feed_readiness.clone();
    tokio::spawn(async move {
        let _reader = gate.read().await;
        let reconcile_dir = dir.clone();
        let outcome = tokio::task::spawn_blocking(move || match db.as_ref() {
            Some(db) => db.reconcile(&reconcile_dir).map(|_| ()),
            None => Ok(()),
        })
        .await;
        match outcome {
            Ok(Ok(())) => {
                tracing::info!(dir = %dir.display(), "reconciled gallery for the change feed")
            }
            Ok(Err(error)) => {
                tracing::warn!(dir = %dir.display(), "gallery reconcile for the change feed failed: {error:#}")
            }
            Err(error) => {
                tracing::warn!(dir = %dir.display(), %error, "gallery reconcile task failed")
            }
        }
        readiness.finish(&dir);
    });
}

/// Every name the committed archive has an opinion about.
fn archive_names(snapshot: &CommittedArchiveSnapshot) -> Vec<String> {
    let index = &snapshot.index;
    let mut names = index
        .entries
        .keys()
        .chain(index.retired_entries.keys())
        .chain(index.retired_names.iter())
        .chain(index.quarantined_names.iter())
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

/// Serializes projections within this process; the DB transaction is what
/// makes concurrent projectors in different processes safe.
static PROJECTION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Bring the dirty set up to date with the committed archive in `snapshot`.
///
/// Must run BEFORE a feed page is read, so every archive change committed up
/// to `snapshot` has a seq above every cursor handed out before it.
pub(crate) fn project_archive_changes(
    db: &MetadataDb,
    dir: &Path,
    snapshot: &CommittedArchiveSnapshot,
) -> anyhow::Result<ArchiveProjection> {
    // No authority store yet: the archive lists nothing, and there is no
    // generation to record.
    let Some(generation) = snapshot.generation else {
        return Ok(ArchiveProjection::Unchanged);
    };
    let _serial = PROJECTION_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let stored = mold_db::gallery_changes::projected_archive_generation(db, dir)?;
    if stored == Some(generation) {
        return Ok(ArchiveProjection::Unchanged);
    }
    let changed = stored
        .filter(|from| *from < generation)
        .and_then(|from| {
            crate::gallery_authority::archive_changes_between(&snapshot.root, from, generation)
        })
        .map(|names| names.into_iter().collect::<Vec<_>>());
    let outcome = mold_db::gallery_changes::project_archive(
        db,
        dir,
        stored,
        generation,
        changed.as_deref(),
        || archive_names(snapshot),
    )?;
    crate::gallery_authority::prune_archive_commit_log(&snapshot.root, generation);
    if let ArchiveProjection::Marked {
        marked,
        everything: true,
    } = outcome
    {
        tracing::info!(
            marked,
            generation,
            previous = ?stored,
            "gallery change feed re-marked every name: the archive generation was not covered by this process's commit log"
        );
    }
    Ok(outcome)
}

/// One feed page for `dir`. The caller holds the gallery publication READER.
pub(crate) fn feed_page(
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    dir: &Path,
    since: Option<&str>,
    limit: usize,
    retention_days: u32,
    instance_id: &str,
) -> anyhow::Result<GalleryChangesPage> {
    let limit = limit.clamp(1, MAX_LIMIT);
    let snapshot = gate.committed_archive_snapshot(dir)?;
    project_archive_changes(db, dir, &snapshot)?;
    let dir_token = dir_token(dir);
    let format = row_format_token();

    let cursor = since
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .map(Cursor::decode);
    let (mut since_seq, mut reset) = match &cursor {
        None => (0, false),
        Some(None) => (0, true),
        Some(Some(cursor)) if cursor.dir != dir_token || cursor.format != format => (0, true),
        Some(Some(cursor)) => (cursor.seq, false),
    };
    let mut page = mold_db::gallery_changes::changes_page(db, dir, since_seq, limit)?;
    if let Some(Some(cursor)) = &cursor {
        if !reset
            && (cursor.epoch != page.epoch
                || cursor.seq < page.min_seq
                || cursor.seq > page.max_seq)
        {
            reset = true;
            since_seq = 0;
            page = mold_db::gallery_changes::changes_page(db, dir, 0, limit)?;
        }
    }

    let names = page
        .changes
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut rows =
        crate::gallery_rows::build_feed_rows(db, &snapshot.index, dir, &names, retention_days)?;
    let changes = names
        .into_iter()
        .map(|filename| {
            let image = rows.remove(&filename).flatten();
            GalleryChange {
                removed: image.is_none(),
                filename,
                image,
            }
        })
        .collect();
    let cursor = Cursor {
        epoch: page.epoch.clone(),
        dir: dir_token,
        format,
        seq: page.next_cursor(since_seq),
    };
    Ok(GalleryChangesPage {
        protocol_version: GALLERY_CHANGES_PROTOCOL_VERSION,
        instance_id: instance_id.to_string(),
        cursor: cursor.encode(),
        more: page.more,
        reset,
        changes,
    })
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChangesQuery {
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

fn unsupported(message: &str) -> ApiError {
    ApiError::with_code(message, GALLERY_FEED_UNSUPPORTED, StatusCode::NOT_FOUND)
}

/// `GET /api/gallery/changes?since=<cursor>&limit=<n>`.
pub(crate) async fn gallery_changes(
    State(state): State<AppState>,
    Query(query): Query<ChangesQuery>,
) -> Result<(HeaderMap, Json<GalleryChangesPage>), ApiError> {
    // Same exclusion as the listing: never read half of a batch publication.
    let _reader = state.gallery_publication_gate.read().await;
    if state.metadata_db.is_none() {
        return Err(unsupported(
            "the gallery change feed needs the metadata DB (MOLD_DB_DISABLE is set)",
        ));
    }
    let config = state.config.read().await;
    if state.is_output_disabled(&config) {
        return Err(unsupported("image output is disabled"));
    }
    let dir = config.effective_output_dir();
    let retention_days = config.gallery.effective_trash_retention_days();
    drop(config);
    if !state.gallery_feed_readiness.is_ready(&dir) {
        if dir.is_dir() && state.gallery_feed_readiness.claim_reconcile(&dir) {
            spawn_reconcile(&state, dir.clone());
        }
        return Err(ApiError::with_code(
            "the gallery is still being reconciled; retry shortly",
            GALLERY_FEED_NOT_READY,
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    if !dir.is_dir() {
        // Nothing to read yet (the listing answers an empty list here);
        // never create the directory from a read.
        return Err(ApiError::with_code(
            "the gallery output directory does not exist yet",
            GALLERY_FEED_NOT_READY,
            StatusCode::SERVICE_UNAVAILABLE,
        ));
    }
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT);
    let since = query.since;
    let instance_id = state.instance_id.to_string();
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let page = tokio::task::spawn_blocking(move || {
        let db = db
            .as_ref()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("metadata DB is unavailable"))?;
        feed_page(
            db,
            &gate,
            &dir,
            since.as_deref(),
            limit,
            retention_days,
            &instance_id,
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery change feed task failed: {e}")))?
    .map_err(|e| ApiError::internal(format!("gallery change feed failed: {e:#}")))?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok((headers, Json(page)))
}

/// The cheap validator for `GET /api/gallery`: everything the listing's
/// body is a function of, read WITHOUT building it.
///
/// - `epoch` and `max_seq` cover every DB-side change (triggers bump the seq
///   in the same transaction as the change, for every writer).
/// - the committed-archive `generation` covers the archive overlay
///   (retire, quarantine, archive-only entries, records that win over rows).
/// - `retention` covers `purge_at`; `view` and the directory pick the body;
///   the row-format token and the crate version cover derivations the code
///   changes without any data changing.
///
/// `None` whenever the listing would not come from the DB (feed not ready,
/// directory not indexed), so the caller keeps the body-hash ETag there.
pub(crate) fn listing_validator(
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    dir: &Path,
    library_view: bool,
    retention_days: u32,
) -> anyhow::Result<Option<String>> {
    if library_view && !db.has_rows_for_dir(dir)? {
        return Ok(None);
    }
    let state = mold_db::gallery_changes::feed_state(db)?;
    let generation = gate.committed_archive_generation(dir)?;
    // Asset inventories are indexed lazily WHILE a listing is built, with no
    // seq (a trigger there would make reads write changes). A scan that
    // failed once and succeeds later changes `assets` with no other change,
    // so the count of recorded scans is part of the validator: every
    // successful index adds a row (a media change that replaces one also
    // bumps the seq through `generations`).
    let asset_scans: i64 = db.with_conn(|conn| {
        Ok(
            conn.query_row("SELECT COUNT(*) FROM generation_asset_scans", [], |r| {
                r.get(0)
            })?,
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "{}|{}|{}|{:?}|{}|{}|{}|{}|{}",
            state.epoch,
            state.max_seq,
            asset_scans,
            generation,
            if library_view { "library" } else { "trash" },
            row_format_token(),
            retention_days,
            mold_db::canonical_dir_string(dir),
            env!("CARGO_PKG_VERSION"),
        )
        .as_bytes(),
    );
    Ok(Some(format!(
        "W/\"g1-{}\"",
        &format!("{:x}", hasher.finalize())[..32]
    )))
}

/// Weak `If-None-Match` comparison (RFC 9110 §13.1.2).
pub(crate) fn if_none_match_contains(headers: &HeaderMap, etag: &str) -> bool {
    let opaque = |tag: &str| tag.trim().trim_start_matches("W/").to_string();
    let wanted = opaque(etag);
    headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|candidate| opaque(candidate) == wanted)
        })
}

/// Replay helper shared by the tests: apply one page to a replica.
#[cfg(test)]
pub(crate) fn apply_page(
    replica: &mut std::collections::BTreeMap<String, serde_json::Value>,
    page: &GalleryChangesPage,
) {
    if page.reset {
        replica.clear();
    }
    for change in &page.changes {
        match &change.image {
            Some(image) => {
                replica.insert(
                    change.filename.clone(),
                    serde_json::to_value(image).unwrap(),
                );
            }
            None => {
                replica.remove(&change.filename);
            }
        }
    }
}

#[cfg(test)]
#[path = "gallery_changes_api_tests.rs"]
mod tests;
