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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    Json,
};
use base64::Engine as _;
use mold_core::{GalleryChange, GalleryChangesPage, GALLERY_CHANGES_PROTOCOL_VERSION};
use mold_db::gallery_changes::{ArchiveMark, ArchiveProjection};
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

/// Whether reconcile has finished for an output directory, against the
/// metadata DB as it is NOW. The feed answers `503 GALLERY_FEED_NOT_READY`
/// until it has: before it, the DB can be missing rows the listing's
/// filesystem fallback would show.
///
/// Readiness is recorded together with the feed `epoch` the reconcile ran
/// against. A corruption rebuild recreates `mold.db` (new epoch, rows not yet
/// re-read from disk), so a stale epoch is NOT ready and re-claims a
/// reconcile; otherwise the feed would reset clients onto an archive-only
/// view and they would drop every DB-only print.
#[derive(Debug, Default)]
pub struct GalleryFeedReadiness {
    inner: std::sync::Mutex<ReadinessInner>,
}

#[derive(Debug, Default)]
struct ReadinessInner {
    /// Directory key → the epoch its reconcile finished against.
    ready: HashMap<String, String>,
    reconciling: HashSet<String>,
}

/// One key per directory whether or not it exists yet: canonical when it
/// does, else its canonical parent joined with its name, so a directory
/// created after `begin` keeps its key.
fn readiness_key(dir: &Path) -> String {
    if let Ok(canonical) = std::fs::canonicalize(dir) {
        return canonical.to_string_lossy().into_owned();
    }
    match (dir.parent(), dir.file_name()) {
        (Some(parent), Some(name)) => std::fs::canonicalize(parent)
            .map(|parent| parent.join(name))
            .unwrap_or_else(|_| dir.to_path_buf())
            .to_string_lossy()
            .into_owned(),
        _ => dir.to_string_lossy().into_owned(),
    }
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

    /// The reconcile of `dir` has finished against the DB whose feed epoch
    /// is `epoch` (successfully or not: a failed reconcile leaves the DB
    /// exactly as authoritative as the listing treats it).
    pub(crate) fn finish(&self, dir: &Path, epoch: String) {
        let key = readiness_key(dir);
        let mut inner = self.lock();
        inner.reconciling.remove(&key);
        inner.ready.insert(key, epoch);
    }

    /// [`Self::finish`] with the epoch read from `db` now. Blocking.
    pub(crate) fn finish_with_db(&self, db: Option<&MetadataDb>, dir: &Path) {
        let epoch = db
            .and_then(|db| mold_db::gallery_changes::feed_state(db).ok())
            .map(|state| state.epoch)
            .unwrap_or_default();
        self.finish(dir, epoch);
    }

    pub(crate) fn is_ready(&self, dir: &Path, epoch: &str) -> bool {
        !epoch.is_empty()
            && self
                .lock()
                .ready
                .get(&readiness_key(dir))
                .is_some_and(|ready| ready == epoch)
    }

    /// Claim the reconcile of a directory that is not ready for `epoch` and
    /// that nobody is reconciling — an `output_dir` changed after startup, or
    /// a DB rebuilt after corruption. True once per such claim.
    fn claim_reconcile(&self, dir: &Path, epoch: &str) -> bool {
        let key = readiness_key(dir);
        let mut inner = self.lock();
        if inner.reconciling.contains(&key)
            || inner.ready.get(&key).is_some_and(|ready| ready == epoch)
        {
            return false;
        }
        inner.reconciling.insert(key);
        true
    }
}

/// Reconcile a directory that is not ready for the current DB, then mark it
/// ready. Same work and lock as the startup reconcile.
fn spawn_reconcile(state: &AppState, dir: PathBuf) {
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let readiness = state.gallery_feed_readiness.clone();
    tokio::spawn(async move {
        let _reader = gate.read().await;
        let reconcile_dir = dir.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let db = db.as_ref().as_ref();
            let outcome = match db {
                Some(db) if reconcile_dir.is_dir() => db.reconcile(&reconcile_dir).map(|_| ()),
                _ => Ok(()),
            };
            // A directory that does not exist yet has nothing to reconcile:
            // it is ready, and empty.
            readiness.finish_with_db(db, &reconcile_dir);
            outcome
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
    });
}

/// Every name the committed archive has an opinion about.
pub(crate) fn archive_names(snapshot: &CommittedArchiveSnapshot) -> Vec<String> {
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
    let current = ArchiveMark {
        generation,
        store_id: snapshot.store_id.clone(),
    };
    let _serial = PROJECTION_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let stored = mold_db::gallery_changes::projected_archive(db, dir)?;
    if stored.as_ref() == Some(&current) {
        return Ok(ArchiveProjection::Unchanged);
    }
    let changed = stored
        .as_ref()
        .filter(|from| from.generation < generation && from.store_id == current.store_id)
        .and_then(|from| {
            crate::gallery_authority::archive_changes_between(
                &snapshot.root,
                from.generation,
                generation,
            )
        })
        .map(|names| names.into_iter().collect::<Vec<_>>());
    let outcome = mold_db::gallery_changes::project_archive(
        db,
        dir,
        stored.as_ref(),
        &current,
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
            previous = ?stored.map(|mark| mark.generation),
            "gallery change feed re-marked every name: the archive generation was not covered by this process's commit log"
        );
    }
    Ok(outcome)
}

/// Test seam: runs between the projection and the page read of the next
/// `feed_page` for a directory, standing in for another process that
/// commits (and projects) in exactly that window.
#[cfg(test)]
type FeedHook = Box<dyn FnOnce() + Send>;

#[cfg(test)]
static FEED_HOOKS: std::sync::Mutex<Vec<(String, FeedHook)>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
pub(crate) fn install_feed_hook(dir: &Path, hook: impl FnOnce() + Send + 'static) {
    FEED_HOOKS
        .lock()
        .unwrap()
        .push((mold_db::canonical_dir_string(dir), Box::new(hook)));
}

#[cfg(test)]
fn run_feed_hook(dir: &Path) {
    let key = mold_db::canonical_dir_string(dir);
    let hook = {
        let mut hooks = FEED_HOOKS.lock().unwrap();
        hooks
            .iter()
            .position(|(dir, _)| *dir == key)
            .map(|at| hooks.remove(at).1)
    };
    if let Some(hook) = hook {
        hook();
    }
}

/// How many times a page is re-read because the archive moved under it.
const MAX_ARCHIVE_RETRIES: usize = 8;

/// The archive kept moving for every retry of one page.
#[derive(Debug)]
pub(crate) struct FeedBusy;

impl std::fmt::Display for FeedBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the gallery archive kept changing while a change-feed page was read")
    }
}

impl std::error::Error for FeedBusy {}

/// One feed page for `dir`. The caller holds the gallery publication READER.
///
/// The rows are built from the archive snapshot that was projected BEFORE
/// the page was read, which is only sound if that snapshot is still the
/// committed one AFTER the read: another process sharing `mold.db` can
/// commit and project in between, putting a mark for a change the snapshot
/// predates into this page — and the cursor would then carry the client
/// past it for good. So the archive identity is re-read after the page, and
/// a moved archive re-projects and re-reads (bounded; [`FeedBusy`] after).
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
    let dir_token = dir_token(dir);
    let format = row_format_token();
    let cursor = since
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .map(Cursor::decode);
    for _ in 0..MAX_ARCHIVE_RETRIES {
        let exists = dir.is_dir();
        let snapshot = if exists {
            gate.committed_archive_snapshot(dir)?
        } else {
            // Nothing published yet: an empty archive, and nothing to lock.
            CommittedArchiveSnapshot {
                root: dir.to_path_buf(),
                generation: None,
                store_id: String::new(),
                index: Default::default(),
            }
        };
        project_archive_changes(db, dir, &snapshot)?;
        #[cfg(test)]
        run_feed_hook(dir);

        // A client without a cursor has nothing to merge into: it REPLACES
        // its replica, so the snapshot's first page says `reset`.
        let (mut since_seq, mut reset) = match &cursor {
            None | Some(None) => (0, true),
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
        if exists {
            let now = gate.committed_archive_identity(dir)?;
            if now != (snapshot.generation, snapshot.store_id.clone()) {
                tracing::debug!(
                    from = ?snapshot.generation,
                    to = ?now.0,
                    "gallery archive moved while a change-feed page was read; re-reading"
                );
                continue;
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
        return Ok(GalleryChangesPage {
            protocol_version: GALLERY_CHANGES_PROTOCOL_VERSION,
            instance_id: instance_id.to_string(),
            cursor: cursor.encode(),
            more: page.more,
            reset,
            changes,
        });
    }
    Err(FeedBusy.into())
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

fn not_ready(message: &str) -> ApiError {
    ApiError::with_code(
        message,
        GALLERY_FEED_NOT_READY,
        StatusCode::SERVICE_UNAVAILABLE,
    )
}

enum FeedAnswer {
    Page(GalleryChangesPage),
    /// Not reconciled for the current DB; `claimed` means this request
    /// starts the reconcile.
    NotReady {
        claimed: bool,
    },
    Busy,
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
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT);
    let since = query.since;
    let instance_id = state.instance_id.to_string();
    let db = state.metadata_db.clone();
    let gate = state.gallery_publication_gate.clone();
    let readiness = state.gallery_feed_readiness.clone();
    let blocking_dir = dir.clone();
    let answer = tokio::task::spawn_blocking(move || -> anyhow::Result<FeedAnswer> {
        let db = db
            .as_ref()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("metadata DB is unavailable"))?;
        let epoch = mold_db::gallery_changes::feed_state(db)?.epoch;
        if !readiness.is_ready(&blocking_dir, &epoch) {
            return Ok(FeedAnswer::NotReady {
                claimed: readiness.claim_reconcile(&blocking_dir, &epoch),
            });
        }
        match feed_page(
            db,
            &gate,
            &blocking_dir,
            since.as_deref(),
            limit,
            retention_days,
            &instance_id,
        ) {
            Ok(page) => Ok(FeedAnswer::Page(page)),
            Err(error) if error.is::<FeedBusy>() => Ok(FeedAnswer::Busy),
            Err(error) => Err(error),
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("gallery change feed task failed: {e}")))?
    .map_err(|e| ApiError::internal(format!("gallery change feed failed: {e:#}")))?;
    let page = match answer {
        FeedAnswer::Page(page) => page,
        FeedAnswer::NotReady { claimed } => {
            if claimed {
                spawn_reconcile(&state, dir);
            }
            return Err(not_ready(
                "the gallery is still being reconciled; retry shortly",
            ));
        }
        FeedAnswer::Busy => {
            return Err(not_ready("the gallery is changing quickly; retry shortly"));
        }
    };
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
///   in the same transaction as the change, for every writer) and the
///   lazily indexed asset inventories: a GLB inventory write, and a bounded
///   number of index failures, mark the print (`generation_assets`).
/// - the committed-archive `generation` and store identity cover the
///   archive overlay (retire, quarantine, archive-only entries, records that
///   win over rows), including a store recreated at the same generation.
/// - `retention` covers `purge_at`; `view` and the directory pick the body;
///   the row-format token and the crate version cover derivations the code
///   changes without any data changing.
///
/// `None` whenever the listing would not come from the DB (not reconciled
/// for this DB, directory not indexed), so the caller keeps the body-hash
/// ETag there.
pub(crate) fn listing_validator(
    db: &MetadataDb,
    gate: &GalleryPublicationGate,
    readiness: &GalleryFeedReadiness,
    dir: &Path,
    library_view: bool,
    retention_days: u32,
) -> anyhow::Result<Option<String>> {
    let state = mold_db::gallery_changes::feed_state(db)?;
    if !readiness.is_ready(dir, &state.epoch) {
        return Ok(None);
    }
    if library_view && !db.has_rows_for_dir(dir)? {
        return Ok(None);
    }
    let (generation, store_id) = gate.committed_archive_identity(dir)?;
    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "{}|{}|{:?}|{}|{}|{}|{}|{}|{}",
            state.epoch,
            state.max_seq,
            generation,
            store_id,
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
