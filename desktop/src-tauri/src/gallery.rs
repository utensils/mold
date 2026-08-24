use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock, Mutex, OnceLock,
    },
    time::Duration,
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::http::{header, Request, Response, StatusCode};
use tauri::Manager;
use tokio::sync::Semaphore;

use crate::commands::{AppState, LocalServer, LocalServerInfo, SettingsStore};

const MAX_SOURCE_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_GALLERY_THUMBNAIL_BYTES: usize = 8 * 1024 * 1024;
const MAX_CONCURRENT_GALLERY_THUMBNAILS: usize = 16;
/// Full-size gallery media (the Library lightbox, the source picker) also
/// travels through the native client — see `fetch_gallery_media`. The cap
/// bounds what one open print may hold in memory; larger files fall back to
/// the webview's streaming path.
const MAX_GALLERY_MEDIA_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONCURRENT_GALLERY_MEDIA: usize = 3;
const GALLERY_IMPORT_CONTENT_TYPE: &str = "application/vnd.mold.gallery-import";

static GALLERY_THUMBNAIL_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
static GALLERY_THUMBNAIL_PERMITS: Semaphore =
    Semaphore::const_new(MAX_CONCURRENT_GALLERY_THUMBNAILS);
static GALLERY_MEDIA_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
static GALLERY_MEDIA_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_GALLERY_MEDIA);
static ACTIVE_THUMBNAIL_REQUESTS: LazyLock<Mutex<HashMap<String, Arc<ThumbnailCancellation>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Default)]
struct ThumbnailCancellation {
    cancelled: AtomicBool,
    notify: tokio::sync::Notify,
}

impl ThumbnailCancellation {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn check(&self) -> Result<(), String> {
        if self.cancelled.load(Ordering::Acquire) {
            Err("Gallery thumbnail request cancelled.".into())
        } else {
            Ok(())
        }
    }
}

struct ActiveThumbnailRequest {
    id: String,
    cancellation: Arc<ThumbnailCancellation>,
}

impl ActiveThumbnailRequest {
    fn register(id: String) -> Result<Self, String> {
        let cancellation = Arc::new(ThumbnailCancellation::default());
        ACTIVE_THUMBNAIL_REQUESTS
            .lock()
            .map_err(|_| "The gallery thumbnail cancellation registry is unavailable.".to_string())?
            .insert(id.clone(), cancellation.clone());
        Ok(Self { id, cancellation })
    }
}

impl Drop for ActiveThumbnailRequest {
    fn drop(&mut self) {
        let Ok(mut requests) = ACTIVE_THUMBNAIL_REQUESTS.lock() else {
            return;
        };
        if requests
            .get(&self.id)
            .is_some_and(|active| Arc::ptr_eq(active, &self.cancellation))
        {
            requests.remove(&self.id);
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ImportedSourceImage {
    filename: String,
    base64: String,
    width: u32,
    height: u32,
    sha256: String,
    metadata: Option<mold_core::OutputMetadata>,
}

fn import_source_image_from_path(path: &std::path::Path) -> Result<ImportedSourceImage, String> {
    use base64::Engine;

    let size = std::fs::metadata(path)
        .map_err(|error| format!("Couldn't inspect the dropped image: {error}"))?
        .len();
    if size > MAX_SOURCE_IMAGE_BYTES {
        return Err("Drop an image no larger than 64 MiB.".into());
    }

    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "The dropped image has no valid filename.".to_string())?
        .to_string();
    let reader = image::ImageReader::open(path)
        .map_err(|error| format!("Couldn't open the dropped image: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("Couldn't identify the dropped image: {error}"))?;
    let format = match reader.format() {
        Some(image::ImageFormat::Png) => mold_core::OutputFormat::Png,
        Some(image::ImageFormat::Jpeg) => mold_core::OutputFormat::Jpeg,
        _ => return Err("Drop a PNG or JPEG image.".into()),
    };
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| format!("Couldn't decode the dropped image: {error}"))?;

    // Read through a hard cap as well as checking metadata so a file that grows
    // between validation and ingestion cannot force an unbounded allocation.
    let mut bytes = Vec::with_capacity(size as usize);
    std::fs::File::open(path)
        .map_err(|error| format!("Couldn't read the dropped image: {error}"))?
        .take(MAX_SOURCE_IMAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Couldn't read the dropped image: {error}"))?;
    if bytes.len() as u64 > MAX_SOURCE_IMAGE_BYTES {
        return Err("Drop an image no larger than 64 MiB.".into());
    }
    let metadata = mold_db::metadata_io::read_embedded(path, format);
    let sha256 = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    };
    Ok(ImportedSourceImage {
        filename,
        base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        width,
        height,
        sha256,
        metadata,
    })
}

/// Read an OS-dropped still and its embedded Mold generation metadata. The
/// command validates the file's decoded format instead of trusting its suffix.
#[tauri::command]
pub async fn import_source_image(
    app: tauri::AppHandle,
    path: String,
) -> Result<ImportedSourceImage, String> {
    let source_path = std::path::PathBuf::from(path);
    let app_for_task = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let imported = import_source_image_from_path(&source_path)?;
        // Path provenance is a best-effort restore aid; a read-only app-data
        // directory must not prevent the image from being attached now.
        let _ = crate::source_stash::remember_source_path(
            &app_for_task,
            &imported.sha256,
            &source_path,
        );
        Ok(imported)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn output_dir() -> Option<std::path::PathBuf> {
    let config = mold_core::Config::load_or_default();
    (!config.is_output_disabled()).then(|| config.effective_output_dir())
}

/// A gallery filename is a basename, so anything that can address a second
/// location is refused.
///
/// The colon is **Windows-only, deliberately**. There it reaches a second
/// location twice over: `Path::join` reads `C:evil.png` as drive-relative and
/// resolves it against that drive's working directory instead of the gallery,
/// and `name.png:hidden` names an NTFS alternate data stream. On macOS and
/// Linux a colon is an ordinary filename byte, and the gallery is NOT limited
/// to names mold generated — `mold_db::reconcile` imports whatever it finds by
/// extension, so an existing install can hold a user's `beach: sunset.png`.
/// Rejecting it everywhere would turn that row into a tile that cannot be
/// opened, saved, revealed, or deleted.
pub(crate) fn valid_filename(filename: &str) -> bool {
    #[cfg(windows)]
    if filename.contains(':') {
        return false;
    }
    !filename.is_empty()
        && filename != "."
        && filename != ".."
        && !filename.contains('/')
        && !filename.contains('\\')
        && !filename.contains('\0')
}

fn scan(dir: &std::path::Path) -> Vec<mold_core::GalleryImage> {
    let mut images: Vec<_> = mold_db::scan::scan_output_dir(dir)
        .filter_map(|item| match item {
            mold_db::scan::ScanItem::Valid(file) => Some(file),
            _ => None,
        })
        .map(|file| {
            let timestamp = file.timestamp_secs();
            let size_bytes = file.size_u64();
            let (metadata, synthetic) = mold_db::metadata_io::read_or_synthesize(
                &file.path,
                file.format,
                &file.filename,
                timestamp,
            );
            mold_core::GalleryImage {
                filename: file.filename,
                metadata,
                timestamp,
                format: Some(file.format),
                size_bytes: Some(size_bytes),
                metadata_synthetic: synthetic,
                title: None,
                tags: Vec::new(),
                favorite: false,
                collections: Vec::new(),
                trashed_at: None,
                purge_at: None,
            }
        })
        .collect();
    images.sort_by_key(|image| std::cmp::Reverse(image.timestamp));
    images
}

enum LocalGalleryAuthority<'a> {
    Server(LocalServerInfo),
    Offline(tokio::sync::MutexGuard<'a, LocalServer>),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalGallerySnapshot {
    images: Vec<mold_core::GalleryImage>,
    target: Option<LocalGalleryTarget>,
    /// The retention this device's OFFLINE `.trash/` sweep would apply, so
    /// the Trash banner can name This Mac while no capability snapshot
    /// exists. `None` whenever a running local server is the authority (its
    /// `/api/capabilities` carries the value) and on the live listing.
    #[serde(skip_serializing_if = "Option::is_none")]
    retention_days: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalGalleryTarget {
    base_url: String,
    api_key: String,
}

impl LocalGalleryTarget {
    fn from_server(info: &LocalServerInfo) -> Result<Self, String> {
        Ok(Self {
            base_url: info.base_url.clone(),
            api_key: api_key(info)?.to_string(),
        })
    }
}

async fn local_gallery_authority(state: &AppState) -> LocalGalleryAuthority<'_> {
    let guard = state.local_server.lock().await;
    match guard.info(&state.local_api_key) {
        Some(info) => LocalGalleryAuthority::Server(info),
        None => LocalGalleryAuthority::Offline(guard),
    }
}

fn server_url(info: &LocalServerInfo, path: &str) -> String {
    format!("{}{}", info.base_url.trim_end_matches('/'), path)
}

fn api_key(info: &LocalServerInfo) -> Result<&str, String> {
    info.api_key
        .as_deref()
        .ok_or_else(|| "The local gallery server has no API key.".to_string())
}

async fn response_error(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value["error"]
                .as_str()
                .or_else(|| value["message"].as_str())
                .map(str::to_string)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("Local gallery server returned {status}."))
}

#[tauri::command]
pub async fn local_gallery_list(
    state: tauri::State<'_, AppState>,
) -> Result<LocalGallerySnapshot, String> {
    let authority = local_gallery_authority(&state).await;
    if let LocalGalleryAuthority::Server(info) = authority {
        let target = LocalGalleryTarget::from_server(&info)?;
        let response = reqwest::Client::new()
            .get(server_url(&info, "/api/gallery"))
            .header("X-Api-Key", api_key(&info)?)
            .send()
            .await
            .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
        if !response.status().is_success() {
            return Err(response_error(response).await);
        }
        let images = response
            .json()
            .await
            .map_err(|error| format!("Invalid local gallery response: {error}"))?;
        return Ok(LocalGallerySnapshot {
            images,
            target: Some(target),
            retention_days: None,
        });
    }
    let LocalGalleryAuthority::Offline(_guard) = authority else {
        unreachable!()
    };
    let Some(dir) = output_dir() else {
        return Ok(LocalGallerySnapshot {
            images: Vec::new(),
            target: None,
            retention_days: None,
        });
    };
    let images = tauri::async_runtime::spawn_blocking(move || {
        offline_live_images(&dir, mold_db::global_db())
    })
    .await
    .map_err(|error| error.to_string())??;
    Ok(LocalGallerySnapshot {
        images,
        target: None,
        retention_days: None,
    })
}

/// The trashed prints of this device's gallery (`GET /api/gallery?view=trash`
/// while the local server runs; the `.trash/` rows otherwise). Same snapshot
/// shape as [`local_gallery_list`] so the store reuses one reader.
#[tauri::command]
pub async fn local_gallery_trash_list(
    state: tauri::State<'_, AppState>,
) -> Result<LocalGallerySnapshot, String> {
    let authority = local_gallery_authority(&state).await;
    if let LocalGalleryAuthority::Server(info) = authority {
        let target = LocalGalleryTarget::from_server(&info)?;
        let response = reqwest::Client::new()
            .get(server_url(&info, "/api/gallery?view=trash"))
            .header("X-Api-Key", api_key(&info)?)
            .send()
            .await
            .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
        if !response.status().is_success() {
            return Err(response_error(response).await);
        }
        let images = response
            .json()
            .await
            .map_err(|error| format!("Invalid local gallery response: {error}"))?;
        return Ok(LocalGallerySnapshot {
            images,
            target: Some(target),
            retention_days: None,
        });
    }
    let LocalGalleryAuthority::Offline(_guard) = authority else {
        unreachable!()
    };
    let Some(dir) = output_dir() else {
        // Output disabled: there is no offline trash to keep, so no
        // retention to advertise either.
        return Ok(LocalGallerySnapshot {
            images: Vec::new(),
            target: None,
            retention_days: None,
        });
    };
    let retention_days = offline_trash_retention_days();
    let images = tauri::async_runtime::spawn_blocking(move || {
        offline_trash_images(&dir, mold_db::global_db(), retention_days)
    })
    .await
    .map_err(|error| error.to_string())??;
    Ok(LocalGallerySnapshot {
        images,
        target: None,
        retention_days: Some(retention_days),
    })
}

// ── Offline trash (the lifecycle guard proves no local server runs) ───────
//
// These helpers take the gallery directory and the metadata DB explicitly so
// the commands above stay thin and the trash arithmetic is testable against
// a tempdir + in-memory DB. They mirror the server's `gallery_trash.rs`
// primitives: move the bytes to `<output_dir>/.trash/<filename>`, write the
// tombstone, flag the row (`mold_db::trash`). Never call them while the
// server runs — the `Server` arm of every command routes over HTTP.

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The retention the embedded engine would apply, read through the DB
/// overlay like every other local surface (`Config::load_or_default`).
fn offline_trash_retention_days() -> u32 {
    mold_core::Config::load_or_default()
        .gallery
        .effective_trash_retention_days()
}

/// `dir/filename` when it exists AND canonicalizes inside `dir`; `None` when
/// it does not exist; an error for anything escaping the directory.
fn contained_file(
    dir: &std::path::Path,
    filename: &str,
) -> Result<Option<std::path::PathBuf>, String> {
    let path = dir.join(filename);
    if !path.exists() {
        return Ok(None);
    }
    let root = dir.canonicalize().map_err(|error| error.to_string())?;
    let candidate = path.canonicalize().map_err(|error| error.to_string())?;
    if !candidate.starts_with(&root) || !candidate.is_file() {
        return Err("Invalid gallery filename.".into());
    }
    Ok(Some(candidate))
}

/// Where a print's bytes live offline. A live-view lookup prefers the live
/// file, then its `.trash/` copy (trashed rows keep their media reachable —
/// thumbnails, the Trash lightbox — exactly as the server resolves trashed
/// rows into `.trash/`). A trash-view lookup (`from_trash`) prefers the
/// `.trash/` copy, so a trashed print is never shadowed by a NEW live file
/// that later landed under the same name.
fn offline_media_path(
    dir: &std::path::Path,
    filename: &str,
    from_trash: bool,
) -> Result<Option<std::path::PathBuf>, String> {
    let trash_dir = mold_db::trash::trash_dir(dir);
    let (first, second) = if from_trash {
        (trash_dir.as_path(), dir)
    } else {
        (dir, trash_dir.as_path())
    };
    if let Some(found) = contained_file(first, filename)? {
        return Ok(Some(found));
    }
    contained_file(second, filename)
}

/// Live prints only: DB rows with `trashed_at_ms IS NULL`, else a disk scan
/// (which never descends into `.trash/`).
fn offline_live_images(
    dir: &std::path::Path,
    db: Option<&mold_db::MetadataDb>,
) -> Result<Vec<mold_core::GalleryImage>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    if let Some(db) = db {
        let rows = db
            .list_live(Some(dir))
            .map_err(|error| format!("{error:#}"))?;
        if !rows.is_empty() {
            return Ok(rows.iter().map(|row| row.to_gallery_image()).collect());
        }
    }
    Ok(scan(dir))
}

/// Trashed prints: DB rows with `trashed_at_ms` set (purge stamps derived
/// from `retention_days`), unioned with the tombstoned files in `.trash/`
/// that have no row — a DB-less install, or an entry whose row was stolen
/// by a same-name replacement — so everything restorable is listed.
fn offline_trash_images(
    dir: &std::path::Path,
    db: Option<&mold_db::MetadataDb>,
    retention_days: u32,
) -> Result<Vec<mold_core::GalleryImage>, String> {
    let trash_dir = mold_db::trash::trash_dir(dir);
    if !trash_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut listed: Vec<mold_core::GalleryImage> = Vec::new();
    let mut covered = std::collections::HashSet::new();
    if let Some(db) = db {
        let rows = db
            .list_trashed(Some(dir))
            .map_err(|error| format!("{error:#}"))?;
        for row in &rows {
            covered.insert(row.filename.clone());
            let mut image = row.to_gallery_image();
            image.purge_at = row
                .trashed_at_ms
                .and_then(|t| mold_db::trash::purge_at_ms(t, retention_days))
                .map(|ms| (ms / 1000) as u64);
            listed.push(image);
        }
    }
    let mut images: Vec<_> = scan(&trash_dir)
        .into_iter()
        .filter(|image| !covered.contains(&image.filename))
        .map(|mut image| {
            let tombstone = mold_db::trash::read_tombstone(&mold_db::trash::tombstone_path(
                &trash_dir,
                &image.filename,
            ))
            .ok();
            if let Some(tombstone) = tombstone {
                if let Some(metadata) = tombstone
                    .metadata_json
                    .as_deref()
                    .and_then(|json| serde_json::from_str(json).ok())
                {
                    image.metadata = metadata;
                    image.metadata_synthetic = false;
                }
                image.title = tombstone.title.or(image.title);
                image.favorite = tombstone.favorite;
                image.tags = tombstone.tags;
                image.trashed_at = Some((tombstone.trashed_at_ms / 1000) as u64);
                image.purge_at =
                    mold_db::trash::purge_at_ms(tombstone.trashed_at_ms, retention_days)
                        .map(|ms| (ms / 1000) as u64);
            } else {
                image.trashed_at = Some(image.timestamp);
                image.purge_at =
                    mold_db::trash::purge_at_ms(image.timestamp as i64 * 1000, retention_days)
                        .map(|ms| (ms / 1000) as u64);
            }
            image
        })
        .collect();
    images.append(&mut listed);
    images.sort_by_key(|image| std::cmp::Reverse(image.trashed_at.unwrap_or(image.timestamp)));
    Ok(images)
}

/// First free name for an incoming trash entry: `<name>`, then
/// `<stem>-2.<ext>`, `-3`, … Free means no trashed bytes, no tombstone, no
/// live file, and (past the original name, whose row is the one being
/// re-keyed) no DB row already claim it. An existing trash entry must NEVER
/// be renamed over — that silently destroys the previously trashed bytes
/// AND their tombstone.
fn unique_trash_name(
    dir: &std::path::Path,
    trash_dir: &std::path::Path,
    db: &mold_db::MetadataDb,
    filename: &str,
) -> Result<String, String> {
    let taken = |name: &str| -> Result<bool, String> {
        if trash_dir.join(name).exists() || mold_db::trash::tombstone_path(trash_dir, name).exists()
        {
            return Ok(true);
        }
        if name == filename {
            return Ok(false);
        }
        // A de-conflicted candidate must also dodge live files and other
        // rows: the rename re-keys the row, and a restore lands at
        // `dir/<name>`.
        if dir.join(name).exists() {
            return Ok(true);
        }
        Ok(db
            .get(dir, name)
            .map_err(|error| format!("{error:#}"))?
            .is_some())
    };
    if !taken(filename)? {
        return Ok(filename.to_string());
    }
    let (stem, ext) = match filename.rsplit_once('.') {
        Some((s, e)) => (s, e),
        None => (filename, ""),
    };
    for n in 2.. {
        let name = if ext.is_empty() {
            format!("{stem}-{n}")
        } else {
            format!("{stem}-{n}.{ext}")
        };
        if !taken(&name)? {
            return Ok(name);
        }
    }
    unreachable!("the counter loop always yields a free name")
}

/// Re-key a `generations` row in place — same rowid, so its tag and
/// collection links survive. `mold_db` exposes no rename primitive and the
/// desktop's offline trash de-confliction is the only caller that needs
/// one, so this goes through the public `with_conn` escape hatch rather
/// than growing the crate's API for it.
fn rename_generation_row(
    db: &mold_db::MetadataDb,
    dir: &std::path::Path,
    from: &str,
    to: &str,
) -> Result<(), String> {
    let dir_key = mold_db::canonical_dir_string(dir);
    db.with_conn(|conn| {
        conn.execute(
            "UPDATE generations SET filename = ?1 WHERE output_dir = ?2 AND filename = ?3",
            [to, dir_key.as_str(), from],
        )?;
        Ok(())
    })
    .map_err(|error| format!("{error:#}"))
}

/// Move a live print into `.trash/` with its tombstone and flag the row.
/// Without a DB there is no row to flag and no retention sweeper to honour,
/// so the file is removed outright — today's behaviour.
///
/// A `.trash/<filename>` that already exists (an earlier print was trashed,
/// then a new live file landed under the same name and is being trashed
/// too) is a distinct print that must survive: the INCOMING file is
/// de-conflicted to `<stem>-2.<ext>` (first free suffix) and its tombstone,
/// DB row, and trash listing all carry that new name, so both prints keep
/// their bytes and both stay individually restorable.
fn offline_trash(
    dir: &std::path::Path,
    db: Option<&mold_db::MetadataDb>,
    filename: &str,
    now_ms: i64,
) -> Result<(), String> {
    let Some(db) = db else {
        if let Some(path) = contained_file(dir, filename)? {
            std::fs::remove_file(&path).map_err(|error| error.to_string())?;
        }
        return Ok(());
    };
    let trash_dir = mold_db::trash::trash_dir(dir);
    if let Some(path) = contained_file(dir, filename)? {
        // The row under this name belongs to the INCOMING live file only
        // when it is not already flagged trashed — a trashed row is the
        // previously trashed print's and must stay untouched.
        let has_live_row = db
            .get(dir, filename)
            .map_err(|error| format!("{error:#}"))?
            .is_some_and(|row| row.trashed_at_ms.is_none());
        let trash_name = unique_trash_name(dir, &trash_dir, db, filename)?;
        let fallback = || mold_db::trash::Tombstone {
            version: mold_db::trash::TOMBSTONE_VERSION,
            filename: filename.to_string(),
            trashed_at_ms: now_ms,
            original_dir: dir.display().to_string(),
            title: None,
            favorite: false,
            tags: Vec::new(),
            collections: Vec::new(),
            metadata_json: None,
        };
        let mut tombstone = if has_live_row {
            db.build_tombstone(dir, filename, now_ms)
                .map_err(|error| format!("{error:#}"))?
                .unwrap_or_else(fallback)
        } else {
            // Without a live row (or with a stolen one) the row under this
            // name describes some OTHER print; record only what we know.
            fallback()
        };
        // The tombstone names the trash entry (bytes sit at
        // `.trash/<tombstone.filename>`); everything else it carries — the
        // row's organization and metadata — is the incoming print's.
        tombstone.filename = trash_name.clone();
        mold_db::trash::write_tombstone(&trash_dir, &tombstone)
            .map_err(|error| format!("{error:#}"))?;
        std::fs::rename(&path, trash_dir.join(&trash_name))
            .map_err(|error| format!("Couldn't move {filename} to the trash: {error}"))?;
        if trash_name != filename && has_live_row {
            rename_generation_row(db, dir, filename, &trash_name)?;
        }
        db.mark_trashed(dir, &trash_name, now_ms)
            .map_err(|error| format!("{error:#}"))?;
        return Ok(());
    }
    // A row without a file still settles: the flag is what hides it from
    // the Library and what the sweeper reads.
    db.mark_trashed(dir, filename, now_ms)
        .map_err(|error| format!("{error:#}"))?;
    Ok(())
}

/// Move a trashed print back. A live file with the same name is a conflict
/// (the server answers 409 `GALLERY_RESTORE_CONFLICT`); never overwrite it.
fn offline_restore(
    dir: &std::path::Path,
    db: Option<&mold_db::MetadataDb>,
    filename: &str,
) -> Result<(), String> {
    let trash_dir = mold_db::trash::trash_dir(dir);
    let trashed = contained_file(&trash_dir, filename)?;
    if contained_file(dir, filename)?.is_some() {
        return Err(format!(
            "Couldn't restore {filename}: a print with that name is already in the library."
        ));
    }
    let Some(trashed) = trashed else {
        return Err(format!("{filename} is not in the trash."));
    };
    std::fs::rename(&trashed, dir.join(filename))
        .map_err(|error| format!("Couldn't restore {filename}: {error}"))?;
    mold_db::trash::remove_tombstone(&trash_dir, filename).map_err(|error| format!("{error:#}"))?;
    if let Some(db) = db {
        db.mark_restored(dir, filename)
            .map_err(|error| format!("{error:#}"))?;
    }
    Ok(())
}

/// Remove a print's bytes for good — from `.trash/` when it was trashed,
/// from the live gallery otherwise — plus its tombstone and row.
fn offline_delete_forever(
    dir: &std::path::Path,
    db: Option<&mold_db::MetadataDb>,
    filename: &str,
) -> Result<(), String> {
    let trash_dir = mold_db::trash::trash_dir(dir);
    if let Some(trashed) = contained_file(&trash_dir, filename)? {
        std::fs::remove_file(&trashed).map_err(|error| error.to_string())?;
    } else if let Some(live) = contained_file(dir, filename)? {
        std::fs::remove_file(&live).map_err(|error| error.to_string())?;
    }
    if trash_dir.is_dir() {
        mold_db::trash::remove_tombstone(&trash_dir, filename)
            .map_err(|error| format!("{error:#}"))?;
    }
    if let Some(db) = db {
        let _ = db.delete(dir, filename);
    }
    Ok(())
}

/// First free name for a new save: `name.png`, then `name-2.png`, `-3`, …
/// Pure over the `exists` probe so it's testable without hitting a real dir.
fn unique_output_path(dir: &std::path::Path, filename: &str) -> std::path::PathBuf {
    let candidate = dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let (stem, ext) = match filename.rsplit_once('.') {
        Some((s, e)) => (s, e),
        None => (filename, ""),
    };
    for n in 2.. {
        let name = if ext.is_empty() {
            format!("{stem}-{n}")
        } else {
            format!("{stem}-{n}.{ext}")
        };
        let candidate = dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("the counter loop always yields a free name")
}

/// True when `path` holds exactly `bytes` — size gate first, then a chunked
/// streaming compare so a large video is never buffered twice in memory.
fn file_matches_bytes(path: &std::path::Path, bytes: &[u8]) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() != bytes.len() as u64 {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    let mut offset = 0usize;
    let mut buf = [0u8; 1 << 16];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return offset == bytes.len(),
            Ok(n) => {
                if bytes.get(offset..offset + n) != Some(&buf[..n]) {
                    return false;
                }
                offset += n;
            }
            Err(_) => return false,
        }
    }
}

/// Write generated bytes into this Mac's output dir — auto-save of remote
/// results and the gallery's "Save to this Mac". The bytes are the encoded
/// output file, so the local gallery reads embedded provenance back out of
/// it; `metadata` optionally carries the origin server's recorded metadata
/// for formats that embed nothing (video). Returns the saved filename.
///
/// Saving the same print twice is idempotent: when the target name already
/// holds a byte-identical file, no `-2` copy is minted — the existing file
/// is re-recorded and its name returned, preserving cross-host identity.
fn save_output_bytes_offline(
    filename: String,
    bytes: Vec<u8>,
    metadata: Option<Box<mold_core::OutputMetadata>>,
) -> Result<String, String> {
    let dir = output_dir().ok_or_else(|| "Local output is disabled.".to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let existing = dir.join(&filename);
    // Idempotence requires byte-identical content, not just a matching
    // name and length — a different print under the same name must not
    // be silently dropped or re-recorded with the wrong provenance.
    let path = if file_matches_bytes(&existing, &bytes) {
        existing
    } else {
        let path = unique_output_path(&dir, &filename);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        path
    };

    let saved_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&filename)
        .to_string();
    // Best-effort DB row so the file shows in the local gallery
    // immediately instead of waiting for the next reconcile walk.
    // Embedded metadata wins (it is the file's own record); the caller's
    // wire metadata covers formats that embed nothing; filename
    // synthesis remains the last resort.
    if let (Some(db), Some(format)) = (
        mold_db::global_db(),
        mold_db::metadata_io::format_from_path(&path),
    ) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let row_metadata = match mold_db::metadata_io::read_embedded(&path, format) {
            Some(embedded) => embedded,
            None => match metadata {
                Some(provided) => *provided,
                None => mold_db::metadata_io::synthesize_from_filename(&saved_name, timestamp),
            },
        };
        let _ = mold_db::persist::record_saved_output(
            db,
            &dir,
            &saved_name,
            &path,
            &mold_db::persist::OutputRecordParams {
                format,
                metadata: &row_metadata,
                source: mold_db::RecordSource::Backfill,
                generation_time_ms: None,
                backend: None,
            },
        );
    }
    Ok(saved_name)
}

#[derive(Serialize)]
struct GalleryImportDescriptor<'a> {
    metadata: &'a mold_core::OutputMetadata,
    metadata_synthetic: bool,
}

#[derive(Deserialize)]
struct GalleryImportResponse {
    filename: String,
}

fn gallery_import_metadata(
    filename: String,
    bytes: &[u8],
    metadata: Option<Box<mold_core::OutputMetadata>>,
) -> (mold_core::OutputMetadata, bool) {
    // Match the offline authority: provenance embedded in the exact bytes
    // wins over separately transported event/DB metadata. Sending a
    // descriptor that differs even slightly from an embedded record is
    // correctly rejected by the server as an immutable import conflict.
    let format = mold_db::metadata_io::format_from_path(std::path::Path::new(&filename));
    let embedded = format
        .filter(|format| {
            matches!(
                format,
                mold_core::OutputFormat::Png
                    | mold_core::OutputFormat::Jpeg
                    | mold_core::OutputFormat::Gif
                    | mold_core::OutputFormat::Apng
            )
        })
        .and_then(|format| {
            let mut temp = tempfile::NamedTempFile::new().ok()?;
            std::io::Write::write_all(&mut temp, bytes).ok()?;
            mold_db::metadata_io::read_embedded(temp.path(), format)
        });
    match embedded.or(metadata.map(|metadata| *metadata)) {
        Some(metadata) => (metadata, false),
        None => {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            (
                mold_db::metadata_io::synthesize_from_filename(&filename, timestamp),
                true,
            )
        }
    }
}

async fn save_output_bytes_server(
    info: LocalServerInfo,
    filename: String,
    bytes: Vec<u8>,
    metadata: Option<Box<mold_core::OutputMetadata>>,
) -> Result<String, String> {
    let (metadata, metadata_synthetic) =
        gallery_import_metadata(filename.clone(), &bytes, metadata);
    let descriptor = serde_json::to_vec(&GalleryImportDescriptor {
        metadata: &metadata,
        metadata_synthetic,
    })
    .map_err(|error| format!("Couldn't encode gallery metadata: {error}"))?;
    let descriptor_len = u32::try_from(descriptor.len())
        .map_err(|_| "Gallery metadata is too large.".to_string())?;
    let file_len =
        u64::try_from(bytes.len()).map_err(|_| "Gallery file is too large.".to_string())?;
    let mut prefix = Vec::with_capacity(12 + descriptor.len());
    prefix.extend_from_slice(&descriptor_len.to_be_bytes());
    prefix.extend_from_slice(&file_len.to_be_bytes());
    prefix.extend_from_slice(&descriptor);
    let body = reqwest::Body::wrap_stream(futures_util::stream::iter([
        Ok::<Vec<u8>, std::io::Error>(prefix),
        Ok(bytes),
    ]));
    let encoded =
        percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
    let response = reqwest::Client::new()
        .put(server_url(&info, &format!("/api/gallery/import/{encoded}")))
        .header("X-Api-Key", api_key(&info)?)
        .header(reqwest::header::CONTENT_TYPE, GALLERY_IMPORT_CONTENT_TYPE)
        .body(body)
        .send()
        .await
        .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
    if !response.status().is_success() {
        return Err(response_error(response).await);
    }
    response
        .json::<GalleryImportResponse>()
        .await
        .map(|response| response.filename)
        .map_err(|error| format!("Invalid local gallery import response: {error}"))
}

#[tauri::command]
pub async fn save_output_bytes(
    state: tauri::State<'_, AppState>,
    filename: String,
    data_b64: String,
    metadata: Option<Box<mold_core::OutputMetadata>>,
) -> Result<String, String> {
    if !valid_filename(&filename) {
        return Err("Invalid filename.".into());
    }
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64.as_bytes())
        .map_err(|e| format!("Invalid image data: {e}"))?;
    match local_gallery_authority(&state).await {
        LocalGalleryAuthority::Server(info) => {
            save_output_bytes_server(info, filename, bytes, metadata).await
        }
        LocalGalleryAuthority::Offline(_guard) => {
            save_output_bytes_offline(filename, bytes, metadata)
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaSaveTarget {
    base_url: String,
    api_key: Option<String>,
}

/// Fetch one bounded gallery file through the native HTTP client. WebKit's
/// per-host connection pool is shared with every held-open generation SSE
/// stream to that host, so media elements pointed straight at a busy remote
/// host can queue indefinitely; the native client sidesteps the webview's
/// pool entirely (the same route `fetch_gallery_thumbnail` took in #1132).
async fn fetch_gallery_bytes(
    client: &reqwest::Client,
    target: &MediaSaveTarget,
    api_path: &str,
    max_bytes: usize,
    what: &str,
    cancellation: Option<&ThumbnailCancellation>,
) -> Result<(Vec<u8>, String), String> {
    if let Some(cancellation) = cancellation {
        cancellation.check()?;
    }
    let url = format!("{}{api_path}", target.base_url.trim_end_matches('/'));
    let mut request = client.get(url);
    if let Some(key) = target.api_key.as_deref().filter(|key| !key.is_empty()) {
        request = request.header("X-Api-Key", key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("Couldn't reach the gallery host: {error}"))?;
    if !response.status().is_success() {
        return Err(response_error(response).await);
    }
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(format!("The gallery {what} is unexpectedly large."));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    let mut bytes = Vec::with_capacity(response.content_length().unwrap_or(0) as usize);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if let Some(cancellation) = cancellation {
            cancellation.check()?;
        }
        let chunk = chunk.map_err(|error| format!("The {what} transfer failed: {error}"))?;
        if bytes.len().saturating_add(chunk.len()) > max_bytes {
            return Err(format!("The gallery {what} is unexpectedly large."));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok((bytes, content_type))
}

#[tauri::command]
pub async fn fetch_gallery_thumbnail(
    target: MediaSaveTarget,
    filename: String,
    request_id: String,
) -> Result<tauri::ipc::Response, String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    if request_id.is_empty() || request_id.len() > 128 {
        return Err("Invalid gallery thumbnail request id.".into());
    }
    let active = ActiveThumbnailRequest::register(request_id)?;
    active.cancellation.check()?;
    let _permit = tokio::select! {
        permit = tokio::time::timeout(
            Duration::from_secs(5),
            GALLERY_THUMBNAIL_PERMITS.acquire()
        ) => permit
            .map_err(|_| "The gallery thumbnail queue is busy; retrying may help.".to_string())?
            .map_err(|_| "The gallery thumbnail service is unavailable.".to_string())?,
        _ = active.cancellation.notify.notified() => {
            active.cancellation.check()?;
            return Err("Gallery thumbnail request cancelled.".into());
        }
    };
    active.cancellation.check()?;
    let encoded =
        percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
    let client = GALLERY_THUMBNAIL_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(15))
            // The per-host key must never follow a redirect off the host.
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(MAX_CONCURRENT_GALLERY_THUMBNAILS)
            .build()
            .expect("static gallery HTTP client settings must be valid")
    });
    let thumbnail_path = format!("/api/gallery/thumbnail/{encoded}");
    let fetch = fetch_gallery_bytes(
        client,
        &target,
        &thumbnail_path,
        MAX_GALLERY_THUMBNAIL_BYTES,
        "thumbnail",
        Some(&active.cancellation),
    );
    tokio::pin!(fetch);
    let (bytes, _content_type) = tokio::select! {
        result = &mut fetch => result?,
        _ = active.cancellation.notify.notified() => {
            active.cancellation.check()?;
            return Err("Gallery thumbnail request cancelled.".into());
        }
    };
    active.cancellation.check()?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
pub fn cancel_gallery_thumbnail(request_id: String) {
    if let Ok(requests) = ACTIVE_THUMBNAIL_REQUESTS.lock() {
        if let Some(cancellation) = requests.get(&request_id) {
            cancellation.cancel();
        }
    }
}

/// Full-size gallery media for a host-backed print, returned as raw bytes
/// (`tauri::ipc::Response` → an `ArrayBuffer` in the webview, no base64).
/// The frontend turns it into an object URL; the file's MIME type rides on
/// the filename extension there. Too-large files and transport failures
/// surface as errors so the caller can fall back to the streaming URL.
#[tauri::command]
pub async fn fetch_gallery_media(
    target: MediaSaveTarget,
    filename: String,
) -> Result<tauri::ipc::Response, String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    // A short wait: the caller falls back to the streaming URL, so the print
    // the user is looking at must not queue behind stale prev/next fetches.
    let _permit = tokio::time::timeout(Duration::from_secs(5), GALLERY_MEDIA_PERMITS.acquire())
        .await
        .map_err(|_| "The gallery media queue is busy; retrying may help.".to_string())?
        .map_err(|_| "The gallery media service is unavailable.".to_string())?;
    let encoded =
        percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
    let client = GALLERY_MEDIA_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(MAX_CONCURRENT_GALLERY_MEDIA)
            .build()
            .expect("static gallery HTTP client settings must be valid")
    });
    let (bytes, _content_type) = fetch_gallery_bytes(
        client,
        &target,
        &format!("/api/gallery/image/{encoded}"),
        MAX_GALLERY_MEDIA_BYTES,
        "file",
        None,
    )
    .await?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedMedia {
    filename: String,
    path: String,
    directory: String,
}

fn effective_media_save_dir(
    app: &tauri::AppHandle,
    store: &SettingsStore,
) -> Result<std::path::PathBuf, String> {
    let configured = store
        .current
        .lock()
        .map_err(|_| "Settings are temporarily unavailable.".to_string())?
        .media_save_dir
        .clone();
    let dir = match configured {
        Some(path) if !path.trim().is_empty() => std::path::PathBuf::from(path),
        _ => app
            .path()
            .download_dir()
            .map_err(|error| format!("Couldn't find the Downloads folder: {error}"))?,
    };
    if !dir.is_dir() {
        return Err(format!(
            "The save folder no longer exists: {}. Choose a new folder in Settings.",
            dir.display()
        ));
    }
    Ok(dir)
}

fn available_media_path(dir: &std::path::Path, filename: &str, index: u32) -> std::path::PathBuf {
    if index == 0 {
        return dir.join(filename);
    }
    let path = std::path::Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("Mold export");
    let suffix = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    dir.join(format!("{stem} ({index}){suffix}"))
}

fn persist_media(
    mut temp: tempfile::NamedTempFile,
    dir: &std::path::Path,
    filename: &str,
) -> Result<SavedMedia, String> {
    temp.as_file_mut()
        .sync_all()
        .map_err(|error| format!("Couldn't finish saving {filename}: {error}"))?;
    for index in 0..10_000 {
        let path = available_media_path(dir, filename, index);
        match temp.persist_noclobber(&path) {
            Ok(_) => {
                let saved_filename = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(filename)
                    .to_string();
                return Ok(SavedMedia {
                    filename: saved_filename,
                    path: path.display().to_string(),
                    directory: dir.display().to_string(),
                });
            }
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                temp = error.file;
            }
            Err(error) => return Err(format!("Couldn't save {filename}: {}", error.error)),
        }
    }
    Err(format!("Couldn't find an available name for {filename}."))
}

#[tauri::command]
pub fn media_save_directory(
    app: tauri::AppHandle,
    store: tauri::State<'_, SettingsStore>,
) -> Result<String, String> {
    effective_media_save_dir(&app, &store).map(|path| path.display().to_string())
}

#[tauri::command]
pub fn reveal_saved_media(
    app: tauri::AppHandle,
    store: tauri::State<'_, SettingsStore>,
    path: String,
) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let dir = effective_media_save_dir(&app, &store)?
        .canonicalize()
        .map_err(|error| format!("Couldn't open the save folder: {error}"))?;
    let path = std::path::PathBuf::from(path)
        .canonicalize()
        .map_err(|_| "The saved file is no longer on disk.".to_string())?;
    if path.parent() != Some(dir.as_path()) || !path.is_file() {
        return Err("That file isn't in the configured save folder.".into());
    }
    app.opener()
        .reveal_item_in_dir(&path)
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub async fn save_media_bytes(
    app: tauri::AppHandle,
    store: tauri::State<'_, SettingsStore>,
    filename: String,
    data_b64: String,
) -> Result<SavedMedia, String> {
    use base64::Engine;
    if !valid_filename(&filename) {
        return Err("Invalid filename.".into());
    }
    let dir = effective_media_save_dir(&app, &store)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64.as_bytes())
        .map_err(|error| format!("Invalid media data: {error}"))?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut temp = tempfile::Builder::new()
            .prefix(".mold-save-")
            .tempfile_in(&dir)
            .map_err(|error| format!("Couldn't create a file in {}: {error}", dir.display()))?;
        std::io::Write::write_all(&mut temp, &bytes)
            .map_err(|error| format!("Couldn't save {filename}: {error}"))?;
        persist_media(temp, &dir, &filename)
    })
    .await
    .map_err(|error| error.to_string())?
}

// A Tauri command's parameters are its wire contract — bundling them into a
// struct would change every call site for a lint about human ergonomics.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn save_gallery_media(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    store: tauri::State<'_, SettingsStore>,
    target: Option<MediaSaveTarget>,
    filename: String,
    output_filename: String,
    export_options: Option<serde_json::Value>,
    from_trash: Option<bool>,
) -> Result<SavedMedia, String> {
    if !valid_filename(&filename) || !valid_filename(&output_filename) {
        return Err("Invalid filename.".into());
    }
    let from_trash = from_trash.unwrap_or(false);
    let dir = effective_media_save_dir(&app, &store)?;
    let target = match target {
        Some(target) => target,
        None if export_options.is_some() => {
            return Err("The video's host is no longer connected.".into());
        }
        None => match local_gallery_authority(&state).await {
            LocalGalleryAuthority::Server(info) => MediaSaveTarget {
                base_url: info.base_url,
                api_key: info.api_key,
            },
            LocalGalleryAuthority::Offline(_guard) => {
                let source_dir =
                    output_dir().ok_or_else(|| "This device's gallery is disabled.".to_string())?;
                // Trash-view rows resolve into `.trash/` (and either view
                // falls through to the other location) so Save keeps
                // working on a trashed print.
                let source = offline_media_path(&source_dir, &filename, from_trash)?
                    .ok_or_else(|| "The gallery file is no longer on disk.".to_string())?;
                let mut input = std::fs::File::open(&source)
                    .map_err(|error| format!("Couldn't read {filename}: {error}"))?;
                let mut temp = tempfile::Builder::new()
                    .prefix(".mold-save-")
                    .tempfile_in(&dir)
                    .map_err(|error| {
                        format!("Couldn't create a file in {}: {error}", dir.display())
                    })?;
                std::io::copy(&mut input, &mut temp)
                    .map_err(|error| format!("Couldn't save {output_filename}: {error}"))?;
                return persist_media(temp, &dir, &output_filename);
            }
        },
    };
    let encoded =
        percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
    let path = if export_options.is_some() {
        format!("/api/gallery/export/{encoded}")
    } else {
        format!("/api/gallery/image/{encoded}")
    };
    let url = format!("{}{}", target.base_url.trim_end_matches('/'), path);
    let client = reqwest::Client::new();
    let mut request = if let Some(options) = export_options {
        client.post(url).json(&options)
    } else {
        client.get(url)
    };
    if let Some(key) = target.api_key.filter(|key| !key.is_empty()) {
        request = request.header("X-Api-Key", key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("Couldn't reach the media host: {error}"))?;
    if !response.status().is_success() {
        return Err(response_error(response).await);
    }
    let mut temp = tempfile::Builder::new()
        .prefix(".mold-save-")
        .tempfile_in(&dir)
        .map_err(|error| format!("Couldn't create a file in {}: {error}", dir.display()))?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|error| format!("The media transfer was interrupted: {error}"))?;
        std::io::Write::write_all(&mut temp, &chunk)
            .map_err(|error| format!("Couldn't save {output_filename}: {error}"))?;
    }
    persist_media(temp, &dir, &output_filename)
}

#[tauri::command]
pub async fn local_output_file_path(
    state: tauri::State<'_, AppState>,
    filename: String,
    from_trash: Option<bool>,
) -> Result<Option<String>, String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let from_trash = from_trash.unwrap_or(false);
    let Some(dir) = output_dir() else {
        return Ok(None);
    };
    match local_gallery_authority(&state).await {
        LocalGalleryAuthority::Server(info) => {
            let encoded = percent_encoding::utf8_percent_encode(
                &filename,
                percent_encoding::NON_ALPHANUMERIC,
            );
            let response = reqwest::Client::new()
                .get(server_url(&info, &format!("/api/gallery/image/{encoded}")))
                .header("X-Api-Key", api_key(&info)?)
                .header(reqwest::header::RANGE, "bytes=0-0")
                .send()
                .await
                .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Ok(None);
            }
            if !response.status().is_success() {
                return Err(response_error(response).await);
            }
            // While the server runs, existence was proven over HTTP; the
            // path is joined without touching the filesystem (the running
            // server owns direct access). Trash-view rows live under the
            // server's own `.trash/` layout (`mold_db::trash::trash_dir`).
            let path = if from_trash {
                mold_db::trash::trash_dir(&dir).join(filename)
            } else {
                dir.join(filename)
            };
            Ok(Some(path.display().to_string()))
        }
        LocalGalleryAuthority::Offline(_guard) => {
            // The guard proves no local server is starting or running, so
            // direct filesystem inspection is safe under the singular
            // desktop gallery authority. A trashed print resolves into
            // `.trash/` so Reveal still works from the Trash view.
            if !dir.is_dir() {
                return Ok(None);
            }
            Ok(offline_media_path(&dir, &filename, from_trash)
                .unwrap_or(None)
                .map(|path| path.display().to_string()))
        }
    }
}

#[tauri::command]
pub async fn local_gallery_delete(
    state: tauri::State<'_, AppState>,
    filename: String,
) -> Result<(), String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let authority = local_gallery_authority(&state).await;
    if let LocalGalleryAuthority::Server(info) = authority {
        let encoded =
            percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
        let response = reqwest::Client::new()
            .delete(server_url(&info, &format!("/api/gallery/image/{encoded}")))
            .header("X-Api-Key", api_key(&info)?)
            .send()
            .await
            .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
        if !response.status().is_success() {
            return Err(response_error(response).await);
        }
        return Ok(());
    }
    let LocalGalleryAuthority::Offline(_guard) = authority else {
        unreachable!()
    };
    // Offline, delete means "move to this device's trash" exactly as the
    // server does; `local_gallery_delete_forever` is the permanent path.
    let dir = output_dir().ok_or_else(|| "Local gallery is disabled.".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        offline_trash(&dir, mold_db::global_db(), &filename, now_ms())
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Bring a trashed print back (`POST /api/gallery/trash/restore` while the
/// local server runs; the `.trash/` move otherwise). A name conflict is an
/// error, never an overwrite.
#[tauri::command]
pub async fn local_gallery_restore(
    state: tauri::State<'_, AppState>,
    filename: String,
) -> Result<(), String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let authority = local_gallery_authority(&state).await;
    if let LocalGalleryAuthority::Server(info) = authority {
        let response = reqwest::Client::new()
            .post(server_url(&info, "/api/gallery/trash/restore"))
            .header("X-Api-Key", api_key(&info)?)
            .json(&serde_json::json!({ "filenames": [filename] }))
            .send()
            .await
            .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
        if !response.status().is_success() {
            return Err(response_error(response).await);
        }
        return Ok(());
    }
    let LocalGalleryAuthority::Offline(_guard) = authority else {
        unreachable!()
    };
    let dir = output_dir().ok_or_else(|| "Local gallery is disabled.".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        offline_restore(&dir, mold_db::global_db(), &filename)
    })
    .await
    .map_err(|error| error.to_string())?
}

/// Delete a print for good (`DELETE /api/gallery/image/:filename?permanent=true`
/// while the local server runs; the `.trash/` purge otherwise).
#[tauri::command]
pub async fn local_gallery_delete_forever(
    state: tauri::State<'_, AppState>,
    filename: String,
) -> Result<(), String> {
    if !valid_filename(&filename) {
        return Err("Invalid gallery filename.".into());
    }
    let authority = local_gallery_authority(&state).await;
    if let LocalGalleryAuthority::Server(info) = authority {
        let encoded =
            percent_encoding::utf8_percent_encode(&filename, percent_encoding::NON_ALPHANUMERIC);
        let response = reqwest::Client::new()
            .delete(server_url(
                &info,
                &format!("/api/gallery/image/{encoded}?permanent=true"),
            ))
            .header("X-Api-Key", api_key(&info)?)
            .send()
            .await
            .map_err(|error| format!("Couldn't reach the local gallery server: {error}"))?;
        if !response.status().is_success() {
            return Err(response_error(response).await);
        }
        return Ok(());
    }
    let LocalGalleryAuthority::Offline(_guard) = authority else {
        unreachable!()
    };
    let dir = output_dir().ok_or_else(|| "Local gallery is disabled.".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        offline_delete_forever(&dir, mold_db::global_db(), &filename)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn content_type(filename: &str) -> &'static str {
    match filename
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" | "apng" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

fn error_response(status: StatusCode, message: &str) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(message.as_bytes().to_vec())
        .expect("valid protocol response")
}

pub fn protocol_response(state: &AppState, request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let guard = match state.local_server.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "The local gallery server is changing state.",
            )
        }
    };
    if guard.info(&state.local_api_key).is_some() {
        return error_response(
            StatusCode::CONFLICT,
            "The running local gallery is available through its authenticated HTTP API.",
        );
    }
    protocol_response_offline(request)
}

fn protocol_response_offline(request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let encoded = request.uri().path().trim_start_matches('/');
    let filename = match percent_encoding::percent_decode_str(encoded).decode_utf8() {
        Ok(filename) if valid_filename(&filename) => filename.into_owned(),
        _ => return error_response(StatusCode::BAD_REQUEST, "Invalid gallery filename."),
    };
    let Some(dir) = output_dir() else {
        return error_response(StatusCode::NOT_FOUND, "Local gallery is disabled.");
    };
    // Live file first, then its `.trash/` copy (trashed prints keep their
    // thumbnails and lightbox media, mirroring the server's path resolution).
    // A `?view=trash` URL flips that preference so a Trash-view row is never
    // shadowed by a newer live file under the same name.
    let from_trash = request
        .uri()
        .query()
        .is_some_and(|query| query.split('&').any(|pair| pair == "view=trash"));
    let safe_path = match offline_media_path(&dir, &filename, from_trash) {
        Ok(Some(path)) => path,
        _ => return error_response(StatusCode::NOT_FOUND, "Gallery file not found."),
    };
    let Ok(mut file) = std::fs::File::open(&safe_path) else {
        return error_response(StatusCode::NOT_FOUND, "Gallery file not found.");
    };
    let Ok(metadata) = file.metadata() else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read gallery file.",
        );
    };
    let total = metadata.len();
    let range = match request.headers().get(header::RANGE) {
        Some(value) => match value
            .to_str()
            .map_err(|_| ())
            .and_then(|value| parse_byte_range(value, total))
        {
            Ok(range) => range,
            Err(()) => {
                return Response::builder()
                    .status(StatusCode::RANGE_NOT_SATISFIABLE)
                    .header(header::CONTENT_RANGE, format!("bytes */{total}"))
                    .header(header::ACCEPT_RANGES, "bytes")
                    .body(Vec::new())
                    .expect("valid range error response")
            }
        },
        None => None,
    };
    let (status, start, end) = range
        .map(|(start, end)| (StatusCode::PARTIAL_CONTENT, start, end))
        .unwrap_or((StatusCode::OK, 0, total.saturating_sub(1)));
    let length = if total == 0 { 0 } else { end - start + 1 };
    if file.seek(SeekFrom::Start(start)).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not seek gallery file.",
        );
    }
    let mut body = vec![0; length as usize];
    if file.read_exact(&mut body).is_err() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Could not read gallery file.",
        );
    }
    let mut response = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type(&filename))
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, length.to_string())
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    if status == StatusCode::PARTIAL_CONTENT {
        response = response.header(
            header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{total}"),
        );
    }
    response.body(body).expect("valid protocol response")
}

fn parse_byte_range(value: &str, total: u64) -> Result<Option<(u64, u64)>, ()> {
    let Some(spec) = value.strip_prefix("bytes=") else {
        return Ok(None);
    };
    if total == 0 {
        return Err(());
    }
    let first = spec.split(',').next().ok_or(())?.trim();
    let (start, end) = first.split_once('-').ok_or(())?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().map_err(|_| ())?;
        if suffix == 0 {
            return Err(());
        }
        let length = suffix.min(total);
        return Ok(Some((total - length, total - 1)));
    }
    let start = start.parse::<u64>().map_err(|_| ())?;
    if start >= total {
        return Err(());
    }
    let end = if end.is_empty() {
        total - 1
    } else {
        end.parse::<u64>().map_err(|_| ())?.min(total - 1)
    };
    if end < start {
        return Err(());
    }
    Ok(Some((start, end)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::Conn;
    use std::sync::Arc;

    fn test_app_state(dir: &tempfile::TempDir) -> Arc<AppState> {
        Arc::new(AppState {
            conn: tokio::sync::Mutex::new(Conn::Off),
            local_server: tokio::sync::Mutex::new(LocalServer::Off),
            local_api_key: "desktop-test-key".into(),
            secrets: crate::secrets::SecretStore::new(dir.path().to_path_buf()),
        })
    }

    #[test]
    fn imports_a_valid_png_as_base64() {
        use base64::Engine;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.png");
        image::RgbImage::from_pixel(2, 1, image::Rgb([12, 34, 56]))
            .save(&path)
            .unwrap();

        let imported = import_source_image_from_path(&path).unwrap();
        assert_eq!(imported.filename, "source.png");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(imported.base64)
                .unwrap(),
            std::fs::read(path).unwrap()
        );
        assert!(imported.metadata.is_none());
    }

    #[test]
    fn rejects_non_image_drops() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"not an image").unwrap();

        assert_eq!(
            import_source_image_from_path(&path).unwrap_err(),
            "Drop a PNG or JPEG image."
        );
    }

    #[test]
    fn rejects_oversized_drops_before_reading_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oversized.png");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_SOURCE_IMAGE_BYTES + 1).unwrap();

        assert_eq!(
            import_source_image_from_path(&path).unwrap_err(),
            "Drop an image no larger than 64 MiB."
        );
    }

    #[test]
    fn file_matches_bytes_requires_identical_content() {
        let dir = std::env::temp_dir().join(format!("mold-fmb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.bin");
        std::fs::write(&path, b"hello world").unwrap();

        assert!(file_matches_bytes(&path, b"hello world"));
        // Same length, different content — the size gate alone must not pass.
        assert!(!file_matches_bytes(&path, b"hello_world"));
        assert!(!file_matches_bytes(&path, b"hello"));
        assert!(!file_matches_bytes(
            &dir.join("missing.bin"),
            b"hello world"
        ));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gallery_protocol_rejects_path_traversal() {
        assert!(!valid_filename("../secrets.json"));
        assert!(!valid_filename("nested/image.png"));
        assert!(!valid_filename("nested\\image.png"));
        assert!(valid_filename("mold-flux-1.png"));

        // `Path::join` resolves a drive-relative name against that drive's own
        // working directory, leaving the gallery root behind entirely, and a
        // `:` suffix names an NTFS alternate data stream.
        #[cfg(windows)]
        {
            assert!(!valid_filename("C:evil.png"));
            assert!(!valid_filename("mold-flux-1.png:hidden"));
        }
        // On unix a colon is an ordinary filename byte, and reconcile imports
        // files mold did not name. Refusing it there would strand real rows.
        #[cfg(unix)]
        assert!(valid_filename("beach: sunset.png"));
    }

    /// On Windows `valid_filename` rejects `:`, and model tags are full of
    /// them (`flux-dev:q8`). That is only safe because
    /// `default_output_filename` replaces the colon before it reaches a
    /// filename — so the two rules have to be pinned together, or a future
    /// generator change silently makes every print of a tagged model
    /// unreachable through this guard.
    ///
    /// Resolved manifest identities only. A raw `hf:owner/repo` id still
    /// carries its slash through the generator, which is a separate
    /// pre-existing question about a path that does not reach gallery naming.
    #[test]
    fn every_filename_mold_generates_passes_the_guard() {
        let tagged = [
            "flux-dev:q8",
            "flux-dev:q4",
            "z-image:turbo",
            "sdxl:base",
            "flux2-klein:q4",
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
        ];
        for model in tagged {
            for (batch, index) in [(1u32, 0u32), (4, 2)] {
                for ext in ["png", "jpg", "mp4"] {
                    let plain =
                        mold_core::default_output_filename(model, 1_700_000_000, ext, batch, index);
                    assert!(valid_filename(&plain), "generator produced {plain}");

                    let slug = mold_core::title_slug("A Titled Print!");
                    let titled = mold_core::default_output_filename_titled(
                        model,
                        1_700_000_000,
                        ext,
                        batch,
                        index,
                        slug.as_deref(),
                    );
                    assert!(valid_filename(&titled), "generator produced {titled}");
                }
            }
        }
    }

    #[test]
    fn server_import_descriptor_prefers_metadata_embedded_in_exact_bytes() {
        let embedded = r#"{"prompt":"embedded owl","model":"ltx-video","seed":7,"steps":30,"guidance":3.0,"width":64,"height":64,"version":"test"}"#;
        let mut bytes = b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x21\xFE".to_vec();
        let comment = format!("mold:parameters {embedded}");
        bytes.push(comment.len() as u8);
        bytes.extend_from_slice(comment.as_bytes());
        bytes.extend_from_slice(&[0, 0x3B]);
        let mut transported =
            mold_db::metadata_io::synthesize_from_filename("clip.gif", 1_700_000_000);
        transported.prompt = "stale transported prompt".into();

        let (resolved, synthetic) =
            gallery_import_metadata("clip.gif".into(), &bytes, Some(Box::new(transported)));

        assert_eq!(resolved.prompt, "embedded owl");
        assert!(!synthetic);
    }

    #[test]
    fn parses_open_ended_and_suffix_byte_ranges() {
        assert_eq!(parse_byte_range("bytes=5-", 10), Ok(Some((5, 9))));
        assert_eq!(parse_byte_range("bytes=-4", 10), Ok(Some((6, 9))));
        assert_eq!(parse_byte_range("bytes=-40", 10), Ok(Some((0, 9))));
    }

    #[test]
    fn rejects_unsatisfiable_or_malformed_byte_ranges() {
        assert_eq!(parse_byte_range("bytes=10-", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=8-2", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=-0", 10), Err(()));
        assert_eq!(parse_byte_range("bytes=nope", 10), Err(()));
    }

    #[test]
    fn unique_output_path_counts_up_past_collisions() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "print.png"),
            dir.path().join("print.png")
        );
        std::fs::write(dir.path().join("print.png"), b"x").unwrap();
        std::fs::write(dir.path().join("print-2.png"), b"x").unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "print.png"),
            dir.path().join("print-3.png")
        );
        // Extension-less names still get numbered rather than clobbered.
        std::fs::write(dir.path().join("raw"), b"x").unwrap();
        assert_eq!(
            unique_output_path(dir.path(), "raw"),
            dir.path().join("raw-2")
        );
    }

    #[test]
    fn media_save_paths_keep_extensions_and_never_clobber() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            available_media_path(dir.path(), "clip.gif", 0),
            dir.path().join("clip.gif")
        );
        assert_eq!(
            available_media_path(dir.path(), "clip.gif", 2),
            dir.path().join("clip (2).gif")
        );

        std::fs::write(dir.path().join("clip.gif"), b"old").unwrap();
        let mut temp = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
        std::io::Write::write_all(&mut temp, b"new").unwrap();
        let saved = persist_media(temp, dir.path(), "clip.gif").unwrap();
        assert_eq!(saved.filename, "clip (1).gif");
        assert_eq!(std::fs::read(dir.path().join("clip.gif")).unwrap(), b"old");
        assert_eq!(
            std::fs::read(dir.path().join("clip (1).gif")).unwrap(),
            b"new"
        );
    }

    #[tokio::test]
    async fn offline_gallery_authority_blocks_off_to_starting_transition() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_app_state(&dir);
        let authority = local_gallery_authority(&state).await;
        assert!(matches!(authority, LocalGalleryAuthority::Offline(_)));

        let transition_state = state.clone();
        let mut transition = tokio::spawn(async move {
            let mut local = transition_state.local_server.lock().await;
            *local = LocalServer::External {
                base_url: "http://127.0.0.1:49152".into(),
            };
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), &mut transition)
                .await
                .is_err(),
            "server transition must wait for the entire direct gallery operation"
        );

        drop(authority);
        tokio::time::timeout(std::time::Duration::from_secs(1), transition)
            .await
            .unwrap()
            .unwrap();
        let authority = local_gallery_authority(&state).await;
        let LocalGalleryAuthority::Server(info) = authority else {
            panic!("completed transition must route through HTTP");
        };
        assert_eq!(info.api_key.as_deref(), Some("desktop-test-key"));
    }

    /// Seed a png in `dir` with a DB row (the offline list reads rows first).
    fn seed_print(
        db: &mold_db::MetadataDb,
        dir: &std::path::Path,
        filename: &str,
    ) -> std::path::PathBuf {
        let path = dir.join(filename);
        image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
            .save(&path)
            .unwrap();
        let req: mold_core::GenerateRequest = serde_json::from_str(
            r#"{"prompt":"an offline owl","model":"flux-dev:q4","width":2,"height":2,"steps":1,"guidance":1.0}"#,
        )
        .unwrap();
        let metadata = mold_core::OutputMetadata::from_generate_request(&req, 9, None, "test");
        let mut rec = mold_db::GenerationRecord::from_save(
            dir,
            filename,
            mold_core::OutputFormat::Png,
            metadata,
            mold_db::RecordSource::Server,
            1_700_000_000_000,
        );
        rec.stat_from_disk(&path);
        db.upsert(&rec).unwrap();
        path
    }

    #[test]
    fn offline_delete_moves_the_print_to_trash_and_hides_it_from_the_live_list() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let live = seed_print(&db, dir.path(), "mold-flux-dev-1700000000.png");
        seed_print(&db, dir.path(), "mold-flux-dev-1700000001.png");
        assert_eq!(offline_live_images(dir.path(), Some(&db)).unwrap().len(), 2);

        offline_trash(
            dir.path(),
            Some(&db),
            "mold-flux-dev-1700000000.png",
            1_000_000,
        )
        .unwrap();

        let trash_dir = mold_db::trash::trash_dir(dir.path());
        assert!(!live.exists(), "the live file must move, not be copied");
        assert!(trash_dir.join("mold-flux-dev-1700000000.png").is_file());
        let tombstone = mold_db::trash::read_tombstone(&mold_db::trash::tombstone_path(
            &trash_dir,
            "mold-flux-dev-1700000000.png",
        ))
        .unwrap();
        assert_eq!(tombstone.trashed_at_ms, 1_000_000);
        assert_eq!(tombstone.filename, "mold-flux-dev-1700000000.png");

        let live_rows = offline_live_images(dir.path(), Some(&db)).unwrap();
        assert_eq!(
            live_rows
                .iter()
                .map(|i| i.filename.as_str())
                .collect::<Vec<_>>(),
            ["mold-flux-dev-1700000001.png"]
        );
        let trashed = offline_trash_images(dir.path(), Some(&db), 30).unwrap();
        assert_eq!(trashed.len(), 1);
        assert_eq!(trashed[0].filename, "mold-flux-dev-1700000000.png");
        assert_eq!(trashed[0].trashed_at, Some(1_000));
        assert_eq!(
            trashed[0].purge_at,
            Some(1_000 + 30 * 24 * 60 * 60),
            "purge_at derives from the device's retention"
        );
        // Keep-forever advertises no purge stamp.
        assert_eq!(
            offline_trash_images(dir.path(), Some(&db), 0).unwrap()[0].purge_at,
            None
        );
        // Trashed media stays reachable for thumbnails / the Trash lightbox.
        assert_eq!(
            offline_media_path(dir.path(), "mold-flux-dev-1700000000.png", false)
                .unwrap()
                .unwrap()
                .canonicalize()
                .unwrap(),
            trash_dir
                .join("mold-flux-dev-1700000000.png")
                .canonicalize()
                .unwrap()
        );
    }

    /// A distinguishable PNG that passes the gallery scan's validity gate
    /// (`is_valid_gallery_file`: ≥256 bytes, decodable, not solid-color) —
    /// raw junk bytes or a tiny solid swatch won't list from a disk scan.
    fn png_bytes(rgb: [u8; 3]) -> Vec<u8> {
        let path = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .unwrap()
            .into_temp_path();
        image::RgbImage::from_fn(64, 64, |x, y| {
            if ((x / 8) + (y / 8)) % 2 == 0 {
                image::Rgb(rgb)
            } else {
                image::Rgb([255 - rgb[0], 255 - rgb[1], 255 - rgb[2]])
            }
        })
        .save(&path)
        .unwrap();
        std::fs::read(&path).unwrap()
    }

    #[test]
    fn offline_trash_keeps_both_prints_on_a_same_name_collision() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let name = "mold-flux-dev-1700000000.png";
        let live = seed_print(&db, dir.path(), name);
        let original_bytes = std::fs::read(&live).unwrap();
        offline_trash(dir.path(), Some(&db), name, 1_000).unwrap();

        // A NEW print lands under the same filename (out-of-band — it has
        // no row of its own) and is trashed too.
        let replacement_bytes = png_bytes([10, 20, 30]);
        assert_ne!(replacement_bytes, original_bytes);
        std::fs::write(&live, &replacement_bytes).unwrap();
        offline_trash(dir.path(), Some(&db), name, 2_000).unwrap();

        let trash_dir = mold_db::trash::trash_dir(dir.path());
        // The ORIGINAL trashed bytes and tombstone survive untouched.
        assert_eq!(
            std::fs::read(trash_dir.join(name)).unwrap(),
            original_bytes,
            "a same-name collision must never overwrite previously trashed bytes"
        );
        let original_tombstone =
            mold_db::trash::read_tombstone(&mold_db::trash::tombstone_path(&trash_dir, name))
                .unwrap();
        assert_eq!(original_tombstone.trashed_at_ms, 1_000);

        // The INCOMING print was de-conflicted to `<stem>-2.png` with a
        // consistent tombstone + listing under the new name.
        let deconflicted = "mold-flux-dev-1700000000-2.png";
        assert_eq!(
            std::fs::read(trash_dir.join(deconflicted)).unwrap(),
            replacement_bytes
        );
        let tombstone = mold_db::trash::read_tombstone(&mold_db::trash::tombstone_path(
            &trash_dir,
            deconflicted,
        ))
        .unwrap();
        assert_eq!(tombstone.filename, deconflicted);
        assert_eq!(tombstone.trashed_at_ms, 2_000);
        let trashed = offline_trash_images(dir.path(), Some(&db), 30).unwrap();
        let mut names: Vec<_> = trashed.iter().map(|i| i.filename.as_str()).collect();
        names.sort();
        assert_eq!(names, [deconflicted, name]);

        // Both remain individually restorable; the de-conflicted print
        // restores under its new name without touching the original.
        offline_restore(dir.path(), Some(&db), deconflicted).unwrap();
        assert_eq!(
            std::fs::read(dir.path().join(deconflicted)).unwrap(),
            replacement_bytes
        );
        assert_eq!(std::fs::read(trash_dir.join(name)).unwrap(), original_bytes);
        offline_restore(dir.path(), Some(&db), name).unwrap();
        assert_eq!(
            std::fs::read(dir.path().join(name)).unwrap(),
            original_bytes
        );
        assert!(offline_trash_images(dir.path(), Some(&db), 30)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn offline_trash_re_keys_the_live_row_when_its_name_is_deconflicted() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let name = "mold-flux-dev-1700000000.png";
        // An earlier print already sits in the trash under this name — as a
        // tombstoned file without a row (e.g. trashed while the DB was
        // unavailable).
        let trash_dir = mold_db::trash::trash_dir(dir.path());
        std::fs::create_dir_all(&trash_dir).unwrap();
        let original_bytes = png_bytes([200, 0, 0]);
        std::fs::write(trash_dir.join(name), &original_bytes).unwrap();
        mold_db::trash::write_tombstone(
            &trash_dir,
            &mold_db::trash::Tombstone {
                version: mold_db::trash::TOMBSTONE_VERSION,
                filename: name.to_string(),
                trashed_at_ms: 500,
                original_dir: dir.path().display().to_string(),
                title: None,
                favorite: false,
                tags: Vec::new(),
                collections: Vec::new(),
                metadata_json: None,
            },
        )
        .unwrap();

        // A live print with a real row is trashed under the same name.
        seed_print(&db, dir.path(), name);
        offline_trash(dir.path(), Some(&db), name, 2_000).unwrap();

        let deconflicted = "mold-flux-dev-1700000000-2.png";
        assert_eq!(std::fs::read(trash_dir.join(name)).unwrap(), original_bytes);
        assert!(trash_dir.join(deconflicted).is_file());
        // The live row followed its bytes: re-keyed in place (same rowid,
        // organization intact) and flagged trashed under the new name.
        assert!(db.get(dir.path(), name).unwrap().is_none());
        let row = db.get(dir.path(), deconflicted).unwrap().unwrap();
        assert_eq!(row.trashed_at_ms, Some(2_000));
        let trashed = offline_trash_images(dir.path(), Some(&db), 30).unwrap();
        let mut names: Vec<_> = trashed.iter().map(|i| i.filename.as_str()).collect();
        names.sort();
        assert_eq!(names, [deconflicted, name]);
    }

    #[test]
    fn offline_media_path_prefers_trash_for_the_trash_view() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let name = "mold-flux-dev-1700000000.png";
        let live = seed_print(&db, dir.path(), name);
        let original_bytes = std::fs::read(&live).unwrap();
        offline_trash(dir.path(), Some(&db), name, 1_000).unwrap();
        // A new live file lands under the same name.
        std::fs::write(&live, b"replacement bytes").unwrap();

        let trash_path = offline_media_path(dir.path(), name, true).unwrap().unwrap();
        assert_eq!(
            trash_path.canonicalize().unwrap(),
            mold_db::trash::trash_dir(dir.path())
                .join(name)
                .canonicalize()
                .unwrap(),
            "the trash view must read the tombstoned bytes, not the replacement"
        );
        assert_eq!(std::fs::read(&trash_path).unwrap(), original_bytes);
        let live_path = offline_media_path(dir.path(), name, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            live_path.canonicalize().unwrap(),
            live.canonicalize().unwrap()
        );
        // Either view falls back to the other location when its preferred
        // copy is missing.
        std::fs::remove_file(&live).unwrap();
        assert_eq!(
            offline_media_path(dir.path(), name, false)
                .unwrap()
                .unwrap()
                .canonicalize()
                .unwrap(),
            trash_path.canonicalize().unwrap()
        );
    }

    #[test]
    fn offline_restore_moves_the_print_back_and_refuses_a_name_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        let live = seed_print(&db, dir.path(), "mold-flux-dev-1700000000.png");
        offline_trash(dir.path(), Some(&db), "mold-flux-dev-1700000000.png", 5).unwrap();

        // A new live print lands under the same name → restore must refuse.
        std::fs::write(&live, b"newer bytes").unwrap();
        let err =
            offline_restore(dir.path(), Some(&db), "mold-flux-dev-1700000000.png").unwrap_err();
        assert!(err.contains("already in the library"), "{err}");
        assert_eq!(std::fs::read(&live).unwrap(), b"newer bytes");
        std::fs::remove_file(&live).unwrap();

        offline_restore(dir.path(), Some(&db), "mold-flux-dev-1700000000.png").unwrap();
        assert!(live.is_file());
        let trash_dir = mold_db::trash::trash_dir(dir.path());
        assert!(!trash_dir.join("mold-flux-dev-1700000000.png").exists());
        assert!(
            !mold_db::trash::tombstone_path(&trash_dir, "mold-flux-dev-1700000000.png").exists()
        );
        assert_eq!(offline_live_images(dir.path(), Some(&db)).unwrap().len(), 1);
        assert!(offline_trash_images(dir.path(), Some(&db), 30)
            .unwrap()
            .is_empty());
        assert!(
            offline_restore(dir.path(), Some(&db), "mold-flux-dev-1700000000.png").is_err(),
            "restoring a live print is not in the trash"
        );
    }

    #[test]
    fn offline_delete_forever_purges_trash_tombstone_and_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        seed_print(&db, dir.path(), "mold-flux-dev-1700000000.png");
        offline_trash(dir.path(), Some(&db), "mold-flux-dev-1700000000.png", 5).unwrap();

        offline_delete_forever(dir.path(), Some(&db), "mold-flux-dev-1700000000.png").unwrap();
        let trash_dir = mold_db::trash::trash_dir(dir.path());
        assert!(!trash_dir.join("mold-flux-dev-1700000000.png").exists());
        assert!(
            !mold_db::trash::tombstone_path(&trash_dir, "mold-flux-dev-1700000000.png").exists()
        );
        assert!(db
            .get(dir.path(), "mold-flux-dev-1700000000.png")
            .unwrap()
            .is_none());
        assert!(offline_trash_images(dir.path(), Some(&db), 30)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn offline_delete_without_a_db_keeps_todays_hard_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mold-flux-dev-1700000000.png");
        image::RgbImage::from_pixel(2, 2, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        offline_trash(dir.path(), None, "mold-flux-dev-1700000000.png", 5).unwrap();
        assert!(!path.exists());
        assert!(!mold_db::trash::trash_dir(dir.path()).exists());
    }

    #[test]
    fn offline_trash_never_escapes_the_gallery_directory() {
        let root = tempfile::tempdir().unwrap();
        let db = mold_db::MetadataDb::open_in_memory().unwrap();
        // The gallery is a SUBDIRECTORY so a relative escape from it actually
        // reaches the victim. `contained_file` is a containment guard, not a
        // name guard — a name that resolves to nothing is a no-op, so pointing
        // the escape at a file that does not exist would pass with the guard
        // removed and prove nothing.
        let dir = root.path().join("gallery");
        std::fs::create_dir(&dir).unwrap();
        let victim = root.path().join("victim.png");
        std::fs::write(&victim, b"x").unwrap();

        // `/` is a separator everywhere, so this escape really reaches the
        // victim and must be refused on every platform.
        assert!(
            offline_trash(&dir, Some(&db), "../victim.png", 5).is_err(),
            "../victim.png was not refused"
        );

        // `\` is a separator only on Windows. There the escape reaches the
        // victim and is refused; on unix it is a single ordinary filename that
        // resolves to nothing, so `contained_file` answers `None` and the call
        // is a harmless no-op. Spelled as two explicit expectations rather than
        // one boolean, because the boolean form read as if it covered both and
        // in fact asserted nothing on unix.
        let backslash = offline_trash(&dir, Some(&db), "..\\victim.png", 5);
        #[cfg(windows)]
        assert!(backslash.is_err(), "..\\victim.png was not refused");
        #[cfg(unix)]
        assert!(
            backslash.is_ok(),
            "a backslash name is one component on unix, so this should be a no-op"
        );

        // The symlink arm needs a privilege Windows does not grant by default.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&victim, dir.join("link.png")).unwrap();
            assert!(offline_trash(&dir, Some(&db), "link.png", 5).is_err());
        }

        assert!(victim.exists(), "the victim was moved out of its directory");
    }

    #[tokio::test]
    async fn native_protocol_never_reads_files_while_server_is_running() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_app_state(&dir);
        *state.local_server.lock().await = LocalServer::External {
            base_url: "http://127.0.0.1:49152".into(),
        };
        let response = protocol_response(
            &state,
            Request::get("mold-local://localhost/anything.mp4")
                .body(Vec::new())
                .unwrap(),
        );
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(String::from_utf8(response.into_body())
            .unwrap()
            .contains("authenticated HTTP API"));
    }
}
