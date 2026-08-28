use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::app::{GalleryEntry, GalleryOrigin};
use crate::hosts::HostRegistry;

/// Returns the default output directory: ~/.mold/output/
pub fn default_gallery_dir() -> PathBuf {
    mold_core::Config::load_or_default().effective_output_dir()
}

/// Returns the image cache directory for server-fetched images: ~/.mold/cache/images/
pub fn image_cache_dir() -> PathBuf {
    mold_core::Config::mold_dir()
        .unwrap_or_else(|| PathBuf::from(".mold"))
        .join("cache")
        .join("images")
}

/// Cache filename for a server-fetched gallery file, scoped by the origin
/// host so same-named files on two machines can never collide in the
/// local cache: `<host_id>__<filename>`. Local entries never enter this
/// cache — they are read straight from the output directory.
pub fn host_cache_filename(host_id: &str, filename: &str) -> String {
    format!("{host_id}__{filename}")
}

/// Local cache path for a server-fetched gallery file (host-scoped).
pub fn cached_image_path(host_id: &str, filename: &str) -> PathBuf {
    image_cache_dir().join(host_cache_filename(host_id, filename))
}

/// Scan one remote host's gallery via its API. Returns `None` when the
/// host is unreachable (or `origin` has no URL) so the caller can count
/// it as offline without failing the merged scan. Honors the host's
/// saved API key.
pub async fn scan_images_from_server(origin: &GalleryOrigin) -> Option<Vec<GalleryEntry>> {
    let url = origin.url.as_deref()?;
    let api_key = crate::hosts::api_key_for(&origin.host_id);
    let client = crate::hosts::client_for(url, api_key.as_deref());
    let images = client.list_gallery().await.ok()?;

    Some(
        images
            .into_iter()
            .map(|img| GalleryEntry {
                path: PathBuf::from(&img.filename),
                metadata: img.metadata,
                generation_time_ms: None,
                timestamp: img.timestamp,
                server_url: Some(url.to_string()),
                title: img.title,
                origins: vec![origin.clone()],
            })
            .collect(),
    )
}

/// Fetch an image from a host and cache it locally under the host-scoped
/// cache name. Returns the local cache path if successful.
pub async fn fetch_and_cache_image(
    server_url: &str,
    host_id: &str,
    filename: &str,
) -> Option<PathBuf> {
    let cached_path = cached_image_path(host_id, filename);

    // Return cached copy if it exists
    if cached_path.is_file() {
        return Some(cached_path);
    }

    let api_key = crate::hosts::api_key_for(host_id);
    let client = crate::hosts::client_for(server_url, api_key.as_deref());
    let data = client.get_gallery_image(filename).await.ok()?;

    std::fs::create_dir_all(image_cache_dir()).ok()?;
    std::fs::write(&cached_path, &data).ok()?;
    Some(cached_path)
}

/// Local cache path for a server-provided animated GIF preview of a video
/// gallery entry. Host-scoped like the full-file cache, and keeps the
/// server's `<filename>.preview.gif` suffix so the two never collide.
pub fn preview_cache_path(host_id: &str, filename: &str) -> PathBuf {
    image_cache_dir().join(host_cache_filename(
        host_id,
        &mold_core::media_paths::preview_gif_filename(filename),
    ))
}

/// Fetch the cached GIF preview for a video gallery entry from a host
/// and persist it under `preview_cache_path`.
///
/// Returns `Some(bytes)` on a 200 response (and updates the on-disk cache);
/// returns `None` when the server returns 404 or the request fails —
/// callers fall back to `fetch_and_cache_image` for the raw file in that
/// case.
pub async fn fetch_and_cache_preview(
    server_url: &str,
    host_id: &str,
    filename: &str,
) -> Option<Vec<u8>> {
    let cached = preview_cache_path(host_id, filename);
    if cached.is_file() {
        if let Ok(data) = std::fs::read(&cached) {
            if !data.is_empty() {
                return Some(data);
            }
        }
    }

    let api_key = crate::hosts::api_key_for(host_id);
    let client = crate::hosts::client_for(server_url, api_key.as_deref());
    let data = match client.get_gallery_preview(filename).await {
        Ok(Some(bytes)) => bytes,
        _ => return None,
    };

    if let Some(parent) = cached.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&cached, &data);
    Some(data)
}

/// Whether `filename`'s extension looks like one of the video formats mold
/// can emit. Used by the TUI to decide whether to try the GIF preview
/// endpoint before falling back to the raw file.
pub fn is_video_filename(filename: &str) -> bool {
    matches!(
        Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "webm" | "mov" | "mkv")
    )
}

/// Result of the local-disk scan.
#[derive(Debug, Default)]
pub(crate) struct LocalScan {
    /// Live (non-trashed) prints, newest first.
    pub entries: Vec<GalleryEntry>,
    /// True when the metadata DB answered — the only case in which a
    /// local delete can become a "move to `<output_dir>/.trash/`" with a
    /// tombstone and a flagged row. Without the DB the TUI hard-deletes,
    /// exactly as before trash existed.
    pub trash_available: bool,
}

/// Scan for mold-generated images in the local output directory.
///
/// Prefers the SQLite metadata DB (populated by both the CLI and server
/// when they save a file) — falls back to the on-disk walk that reads
/// embedded `mold:parameters` chunks when the DB is disabled, empty, or
/// unreadable. The DB path keeps the gallery accurate for formats with no
/// embedded metadata (gif/webp/mp4) without requiring per-file parsing on
/// every refresh.
pub fn scan_images_local() -> Vec<GalleryEntry> {
    scan_local().entries
}

/// [`scan_images_local`] plus whether the metadata DB was available. Only
/// **live** rows are listed (`MetadataDb::list_live`): a trashed print
/// keeps its row and its bytes under `.trash/`, but it is not part of the
/// Library until restored.
pub(crate) fn scan_local() -> LocalScan {
    let output_dir = default_gallery_dir();

    if !output_dir.is_dir() {
        return LocalScan::default();
    }

    if let Ok(Some(db)) = mold_db::open_default() {
        // Local-only workflows (TUI without `mold serve`) don't get the
        // server's startup reconciliation pass — sync the DB to disk on
        // every refresh so files added/removed outside the CLI are
        // reflected. The walk is bounded by the gallery directory size
        // and matches what the legacy filesystem scan paid per call.
        if let Err(e) = db.reconcile(&output_dir) {
            tracing::warn!("local TUI gallery reconcile failed: {e:#}");
        }
        if let Ok(rows) = db.list_live(Some(&output_dir)) {
            if !rows.is_empty() {
                let entries: Vec<GalleryEntry> = rows
                    .into_iter()
                    .map(|r| GalleryEntry {
                        path: PathBuf::from(&r.output_dir).join(&r.filename),
                        metadata: r.metadata,
                        generation_time_ms: r.generation_time_ms.map(|n| n as u64),
                        timestamp: r
                            .file_mtime_ms
                            .or(Some(r.created_at_ms))
                            .map(|ms| (ms / 1000) as u64)
                            .unwrap_or(0),
                        server_url: None,
                        title: r.title,
                        origins: vec![GalleryOrigin::local()],
                    })
                    .collect();
                return LocalScan {
                    entries,
                    trash_available: true,
                };
            }
        }
        // The DB is up but holds no row for this directory (fresh
        // install, or every row is trashed): walk the disk for the
        // listing, but a delete can still go through the DB-backed
        // trash, which records the row itself.
        let mut scan = scan_local_disk(&output_dir);
        scan.trash_available = true;
        return scan;
    }

    scan_local_disk(&output_dir)
}

/// The DB-less listing: walk `output_dir` and recover metadata per file.
fn scan_local_disk(output_dir: &Path) -> LocalScan {
    // Fallback filesystem walk (DB disabled/unavailable): the shared
    // mold-db walker applies the same validity guards the server gallery
    // and reconcile use, and the shared metadata leaf recovers embedded
    // PNG/JPEG/GIF metadata or synthesizes from the filename (with real
    // raster dims) for everything else.
    let mut entries: Vec<GalleryEntry> = mold_db::scan::scan_output_dir(output_dir)
        .filter_map(|item| match item {
            mold_db::scan::ScanItem::Valid(file) => Some(file),
            _ => None,
        })
        .map(|file| {
            let timestamp = file.timestamp_secs();
            let (metadata, _synthetic) = mold_db::metadata_io::read_or_synthesize(
                &file.path,
                file.format,
                &file.filename,
                timestamp,
            );
            GalleryEntry {
                path: file.path,
                metadata,
                generation_time_ms: None,
                timestamp,
                server_url: None,
                title: None,
                origins: vec![GalleryOrigin::local()],
            }
        })
        .collect();

    entries.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
    LocalScan {
        entries,
        trash_available: false,
    }
}

// ── Multi-host merged scan ──────────────────────────────────────────

/// Result of a merged all-hosts gallery scan.
#[derive(Debug, Default)]
pub struct MergedScan {
    /// Deduplicated entries, newest first.
    pub entries: Vec<GalleryEntry>,
    /// Number of remote sources that failed to answer — surfaced in the
    /// Library header so the merged view is honest about missing hosts.
    pub offline_hosts: usize,
    /// True when this machine's prints came from the DB-backed local scan,
    /// so a local delete can move the file to `<output_dir>/.trash/`
    /// (see [`LocalScan::trash_available`]). False when the local source
    /// was read through a loopback server (that host's capabilities
    /// answer instead) or when the DB was unavailable.
    pub local_trash_available: bool,
}

/// One source's contribution to a merged scan. `entries: None` means the
/// source failed (offline host) — it contributes nothing but doesn't
/// break the scan.
#[derive(Debug)]
pub(crate) struct SourceScan {
    pub origin: GalleryOrigin,
    pub entries: Option<Vec<GalleryEntry>>,
    /// Only meaningful for the disk-backed this-machine source: whether
    /// the metadata DB answered, i.e. whether local trash is available.
    pub local_trash_available: bool,
}

/// Whether a URL points at this machine (`localhost` / `127.*` / `::1` /
/// `0.0.0.0`). The connected loopback server's gallery is authoritative
/// for the local machine — it must count as ONE machine, not two.
pub(crate) fn is_loopback_url(url: &str) -> bool {
    let host_port = crate::hosts::host_port_label(url);
    let host = match host_port.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !h.is_empty() => h,
        _ => host_port.as_str(),
    };
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    host == "localhost" || host == "::1" || host == "0.0.0.0" || host.starts_with("127.")
}

/// Build the source list for a merged scan: this machine first, then the
/// connected `--host` server (when it is a different box), then every
/// registered Machines host — deduplicated by URL and host id so one box
/// reached two ways is scanned once.
///
/// A connected **loopback** server collapses into the this-machine
/// source: its gallery is authoritative for the local output dir (the
/// pre-merge behavior), so this machine is scanned through it over HTTP —
/// with a directory-walk fallback if the server stops answering.
pub(crate) fn scan_sources(
    server_url: Option<&str>,
    registry: &HostRegistry,
) -> Vec<GalleryOrigin> {
    let mut out = vec![match server_url.filter(|u| is_loopback_url(u)) {
        Some(url) => GalleryOrigin {
            host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
            url: Some(url.to_string()),
            name: crate::hosts::local_display_name().to_string(),
        },
        None => GalleryOrigin::local(),
    }];

    let mut candidates = Vec::new();
    if let Some(url) = server_url.filter(|u| !is_loopback_url(u)) {
        // Prefer the registry's identity for the connected server when it
        // is also a registered host — same id ⇒ same API key + name.
        let origin = registry
            .hosts
            .iter()
            .find(|h| h.url == url)
            .map(GalleryOrigin::from_host_entry)
            .unwrap_or_else(|| GalleryOrigin::remote_from_url(url));
        candidates.push(origin);
    }
    for host in registry.hosts.iter().filter(|host| host.connected) {
        candidates.push(GalleryOrigin::from_host_entry(host));
    }

    for origin in candidates {
        let dup = out
            .iter()
            .any(|o| o.host_id == origin.host_id || (o.url.is_some() && o.url == origin.url));
        if !dup {
            out.push(origin);
        }
    }
    out
}

/// Scan every source concurrently and merge the results. Local scanning
/// runs on the blocking pool; each remote host gets its own task so one
/// slow host can't serialize the others. A failed host contributes
/// nothing and is counted in `offline_hosts`; the this-machine source
/// never goes offline — if its loopback server stops answering, it falls
/// back to walking the output directory.
pub async fn scan_all_hosts(sources: Vec<GalleryOrigin>) -> MergedScan {
    let mut handles = Vec::with_capacity(sources.len());
    for origin in sources {
        handles.push(tokio::spawn(async move {
            if origin.is_this_machine() {
                if origin.url.is_some() {
                    if let Some(entries) = scan_images_from_server(&origin).await {
                        return SourceScan {
                            origin,
                            entries: Some(entries),
                            local_trash_available: false,
                        };
                    }
                }
                let scan = tokio::task::spawn_blocking(scan_local)
                    .await
                    .unwrap_or_default();
                SourceScan {
                    origin: GalleryOrigin::local(),
                    entries: Some(scan.entries),
                    local_trash_available: scan.trash_available,
                }
            } else {
                let entries = scan_images_from_server(&origin).await;
                SourceScan {
                    origin,
                    entries,
                    local_trash_available: false,
                }
            }
        }));
    }

    let mut scans = Vec::with_capacity(handles.len());
    for handle in handles {
        if let Ok(scan) = handle.await {
            scans.push(scan);
        }
    }
    merge_scans(scans)
}

/// Merge per-source scans into one deduplicated list.
///
/// Print identity is the **filename** — the same rule the desktop's
/// unified gallery uses as its primary collapse key (auto-saved remote
/// outputs keep the origin's gallery filename, so cross-host copies of
/// one print share a name). The desktop's seed + byte-size fallback for
/// legacy copies is deliberately not implemented here: the TUI's local
/// scan rows don't carry byte sizes, so filename-collapse is the whole
/// rule.
///
/// Sources are merged in order with the local source first, so when a
/// print exists both locally and on a host the **local copy is primary**
/// (read from disk, no fetch); otherwise the first host that listed it
/// wins. Every other machine that holds a copy is appended to the
/// entry's `origins` so the details panel can say where it exists and
/// delete can route to all of them.
pub(crate) fn merge_scans(scans: Vec<SourceScan>) -> MergedScan {
    let mut entries: Vec<GalleryEntry> = Vec::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut offline_hosts = 0usize;
    let mut local_trash_available = false;

    for scan in scans {
        local_trash_available |= scan.local_trash_available;
        let Some(source_entries) = scan.entries else {
            if !scan.origin.is_this_machine() {
                offline_hosts += 1;
            }
            continue;
        };
        for entry in source_entries {
            let name = entry.filename();
            match by_name.get(&name) {
                Some(&i) => {
                    let existing = &mut entries[i];
                    for origin in entry.origins {
                        if !existing.origins.contains(&origin) {
                            existing.origins.push(origin);
                        }
                    }
                    if existing.generation_time_ms.is_none() {
                        existing.generation_time_ms = entry.generation_time_ms;
                    }
                }
                None => {
                    by_name.insert(name, entries.len());
                    entries.push(entry);
                }
            }
        }
    }

    // Newest first across all machines. Stable sort keeps source order
    // for equal timestamps.
    entries.sort_by_key(|e| std::cmp::Reverse(e.timestamp));
    MergedScan {
        entries,
        offline_hosts,
        local_trash_available,
    }
}

// ── Library filter + header ─────────────────────────────────────────

/// Case-insensitive filter over prompt, model, and filename. Returns the
/// indices of matching entries — thumbnail caches stay keyed by the
/// underlying entry index, so filtering never invalidates them.
pub(crate) fn filter_entries(entries: &[GalleryEntry], query: &str) -> Vec<usize> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return (0..entries.len()).collect();
    }
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.metadata.prompt.to_lowercase().contains(&q)
                || e.metadata.model.to_lowercase().contains(&q)
                || e.filename().to_lowercase().contains(&q)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Compose the Library panel's header hint.
///
/// - `"{n} prints"` when everything shown is local,
/// - `"{n} prints · {host}"` when everything came from exactly one
///   remote machine,
/// - `"{n} prints · all machines"` once prints span more than one
///   machine,
/// - with an active filter: `"{k}/{n} prints · /{query}"`,
/// - and an honest `"· {k} host(s) offline"` suffix when some hosts
///   failed to answer the scan.
pub(crate) fn library_hint(
    entries: &[GalleryEntry],
    filter: &str,
    filtered_count: usize,
    offline_hosts: usize,
) -> Option<String> {
    if entries.is_empty() {
        return None;
    }

    let mut base = if !filter.trim().is_empty() {
        format!("{filtered_count}/{} prints · /{}", entries.len(), filter)
    } else {
        let mut has_local = false;
        let mut remote_names: Vec<String> = Vec::new();
        for entry in entries {
            for origin in entry.owning_origins() {
                if origin.is_this_machine() {
                    has_local = true;
                } else if !remote_names.contains(&origin.name) {
                    remote_names.push(origin.name.clone());
                }
            }
        }
        let n = entries.len();
        let noun = if n == 1 { "print" } else { "prints" };
        if remote_names.is_empty() {
            format!("{n} {noun}")
        } else if !has_local && remote_names.len() == 1 {
            format!("{n} {noun} · {}", remote_names[0])
        } else {
            format!("{n} {noun} · all machines")
        }
    };

    if offline_hosts > 0 {
        let plural = if offline_hosts == 1 { "host" } else { "hosts" };
        base.push_str(&format!(" · {offline_hosts} {plural} offline"));
    }
    Some(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosts::HostEntry;

    fn meta(prompt: &str, model: &str) -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: prompt.to_string(),
            negative_prompt: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
            model: model.to_string(),
            seed: 1,
            steps: 4,
            guidance: 0.0,
            width: 64,
            height: 64,
            generation_width: Some(64),
            generation_height: Some(64),
            strength: None,
            source_image_name: None,
            source_image_sha256: None,
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: None,
            lora_scale: None,
            loras: None,
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "test".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    fn local_entry(name: &str, ts: u64) -> GalleryEntry {
        GalleryEntry {
            path: PathBuf::from(name),
            metadata: meta("a cat", "flux2-klein:q8"),
            generation_time_ms: None,
            timestamp: ts,
            server_url: None,
            title: None,
            origins: vec![GalleryOrigin::local()],
        }
    }

    fn remote_origin(id: &str, name: &str) -> GalleryOrigin {
        GalleryOrigin {
            host_id: id.to_string(),
            url: Some(format!("http://{name}:7680")),
            name: name.to_string(),
        }
    }

    fn remote_entry(name: &str, ts: u64, origin: &GalleryOrigin) -> GalleryEntry {
        GalleryEntry {
            path: PathBuf::from(name),
            metadata: meta("a cat", "flux2-klein:q8"),
            generation_time_ms: None,
            timestamp: ts,
            server_url: origin.url.clone(),
            title: None,
            origins: vec![origin.clone()],
        }
    }

    fn host(id: &str, url: &str) -> HostEntry {
        HostEntry {
            id: id.to_string(),
            url: url.to_string(),
            name: Some(id.split('-').next().unwrap_or(id).to_string()),
            instance_id: None,
            connected: true,
        }
    }

    // ── merged-scan dedupe rules ────────────────────────────────

    #[test]
    fn merge_collapses_same_filename_across_hosts_prefers_local() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let scans = vec![
            SourceScan {
                origin: GalleryOrigin::local(),
                entries: Some(vec![local_entry("print-1.png", 100)]),
                local_trash_available: false,
            },
            SourceScan {
                origin: hal.clone(),
                entries: Some(vec![
                    remote_entry("print-1.png", 100, &hal),
                    remote_entry("print-2.png", 50, &hal),
                ]),
                local_trash_available: false,
            },
        ];
        let merged = merge_scans(scans);
        assert_eq!(merged.entries.len(), 2, "print-1 collapses by filename");
        let first = &merged.entries[0];
        assert_eq!(first.filename(), "print-1.png");
        assert!(
            first.server_url.is_none(),
            "local copy is primary — read from disk, no fetch"
        );
        assert_eq!(first.origins.len(), 2, "both machines are recorded");
        assert!(first.origins[0].is_local());
        assert_eq!(first.origins[1].host_id, "hal9000-7680");
        assert_eq!(merged.offline_hosts, 0);
    }

    #[test]
    fn merge_keeps_distinct_filenames_and_sorts_newest_first() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let scans = vec![
            SourceScan {
                origin: GalleryOrigin::local(),
                entries: Some(vec![local_entry("old.png", 10)]),
                local_trash_available: false,
            },
            SourceScan {
                origin: hal.clone(),
                entries: Some(vec![remote_entry("new.png", 99, &hal)]),
                local_trash_available: false,
            },
        ];
        let merged = merge_scans(scans);
        assert_eq!(merged.entries.len(), 2);
        assert_eq!(merged.entries[0].filename(), "new.png");
        assert_eq!(merged.entries[1].filename(), "old.png");
    }

    #[test]
    fn merge_counts_offline_hosts_without_breaking_the_scan() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let bender = remote_origin("bender-7680", "bender");
        let scans = vec![
            SourceScan {
                origin: GalleryOrigin::local(),
                entries: Some(vec![local_entry("a.png", 1)]),
                local_trash_available: false,
            },
            SourceScan {
                origin: hal,
                entries: None, // offline
                local_trash_available: false,
            },
            SourceScan {
                origin: bender,
                entries: None, // offline
                local_trash_available: false,
            },
        ];
        let merged = merge_scans(scans);
        assert_eq!(merged.entries.len(), 1, "local prints survive dead hosts");
        assert_eq!(merged.offline_hosts, 2);
    }

    #[test]
    fn merge_dedupes_same_print_listed_by_two_hosts() {
        // A print auto-saved to two machines (no local copy): first host
        // wins as primary, second is appended as an extra origin.
        let hal = remote_origin("hal9000-7680", "hal9000");
        let bender = remote_origin("bender-7680", "bender");
        let scans = vec![
            SourceScan {
                origin: hal.clone(),
                entries: Some(vec![remote_entry("shared.png", 5, &hal)]),
                local_trash_available: false,
            },
            SourceScan {
                origin: bender.clone(),
                entries: Some(vec![remote_entry("shared.png", 5, &bender)]),
                local_trash_available: false,
            },
        ];
        let merged = merge_scans(scans);
        assert_eq!(merged.entries.len(), 1);
        let entry = &merged.entries[0];
        assert_eq!(entry.server_url.as_deref(), Some("http://hal9000:7680"));
        assert_eq!(entry.origins.len(), 2);
    }

    // ── source list ─────────────────────────────────────────────

    #[test]
    fn scan_sources_this_machine_first_then_server_then_registry() {
        let mut registry = HostRegistry::default();
        registry
            .add(host("hal9000-7680", "http://hal9000:7680"))
            .unwrap();
        let sources = scan_sources(Some("http://bender:7680"), &registry);
        assert_eq!(sources.len(), 3);
        assert!(sources[0].is_this_machine());
        assert!(sources[0].url.is_none());
        assert_eq!(sources[1].url.as_deref(), Some("http://bender:7680"));
        assert_eq!(sources[2].host_id, "hal9000-7680");
    }

    #[test]
    fn scan_sources_loopback_server_collapses_into_this_machine() {
        // `mold tui` with a local `mold serve` on :7680 is ONE machine —
        // the loopback server becomes the this-machine source (scanned
        // over HTTP, authoritative for the local output dir), never a
        // second "machine" that would flip the header to "all machines".
        for url in [
            "http://localhost:7680",
            "http://127.0.0.1:7680",
            "http://[::1]:7680",
        ] {
            let sources = scan_sources(Some(url), &HostRegistry::default());
            assert_eq!(sources.len(), 1, "{url}");
            assert!(sources[0].is_this_machine(), "{url}");
            assert_eq!(sources[0].url.as_deref(), Some(url));
            assert_eq!(sources[0].name, crate::hosts::local_display_name());
        }
    }

    #[test]
    fn is_loopback_url_classifies_hosts() {
        assert!(is_loopback_url("http://localhost:7680"));
        assert!(is_loopback_url("http://127.0.0.1:7680"));
        assert!(is_loopback_url("http://[::1]:7680"));
        assert!(is_loopback_url("http://0.0.0.0:7680"));
        assert!(!is_loopback_url("http://hal9000:7680"));
        assert!(!is_loopback_url("http://192.168.1.5:7680"));
        assert!(!is_loopback_url("https://mold.example.com"));
    }

    #[test]
    fn scan_sources_dedupes_connected_server_against_registry() {
        // The connected --host server is also a registered machine: scan
        // it once, under its registry identity (so its API key applies).
        let mut registry = HostRegistry::default();
        registry
            .add(host("hal9000-7680", "http://hal9000:7680"))
            .unwrap();
        let sources = scan_sources(Some("http://hal9000:7680"), &registry);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[1].host_id, "hal9000-7680");
        assert_eq!(sources[1].name, "hal9000");
    }

    #[test]
    fn scan_sources_no_server_no_hosts_is_local_only() {
        let sources = scan_sources(None, &HostRegistry::default());
        assert_eq!(sources.len(), 1);
        assert!(sources[0].is_local());
    }

    // ── host-scoped cache paths ─────────────────────────────────

    #[test]
    #[serial_test::serial(mold_env)]
    fn cached_image_path_is_scoped_by_host_id() {
        // Same filename on two machines must land in two cache slots —
        // otherwise the first host fetched poisons the other's preview.
        let a = cached_image_path("hal9000-7680", "print.png");
        let b = cached_image_path("bender-7680", "print.png");
        assert_ne!(a, b);
        assert!(a
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "hal9000-7680__print.png"));
        assert!(a.starts_with(image_cache_dir()));
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn preview_cache_path_is_host_scoped_and_keeps_server_suffix() {
        // The server names previews `<filename>.preview.gif`; the local
        // cache keeps that suffix, prefixed by the origin host id.
        let path = preview_cache_path("hal9000-7680", "ltx2-1234.mp4");
        assert!(
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "hal9000-7680__ltx2-1234.mp4.preview.gif"),
            "unexpected preview cache filename: {}",
            path.display()
        );
        assert!(path.starts_with(image_cache_dir()));
    }

    // ── filter ──────────────────────────────────────────────────

    #[test]
    fn filter_entries_matches_prompt_model_filename() {
        let mut a = local_entry("sunset-beach.png", 3);
        a.metadata.prompt = "A golden sunset".into();
        a.metadata.model = "flux2-klein:q8".into();
        let mut b = local_entry("cat-01.png", 2);
        b.metadata.prompt = "a tabby cat".into();
        b.metadata.model = "z-image:q8".into();
        let entries = vec![a, b];

        assert_eq!(filter_entries(&entries, "SUNSET"), vec![0], "prompt match");
        assert_eq!(filter_entries(&entries, "z-image"), vec![1], "model match");
        assert_eq!(
            filter_entries(&entries, "cat-01"),
            vec![1],
            "filename match"
        );
        assert_eq!(filter_entries(&entries, ""), vec![0, 1], "empty = all");
        assert_eq!(filter_entries(&entries, "  "), vec![0, 1]);
        assert!(filter_entries(&entries, "no-such").is_empty());
    }

    // ── header hint ─────────────────────────────────────────────

    #[test]
    fn library_hint_local_only_is_plain_prints() {
        let entries = vec![local_entry("a.png", 1), local_entry("b.png", 2)];
        assert_eq!(
            library_hint(&entries, "", 2, 0).as_deref(),
            Some("2 prints")
        );
    }

    #[test]
    fn library_hint_single_remote_source_names_the_host() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let entries = vec![remote_entry("a.png", 1, &hal)];
        assert_eq!(
            library_hint(&entries, "", 1, 0).as_deref(),
            Some("1 print · hal9000")
        );
        let two = vec![
            remote_entry("a.png", 1, &hal),
            remote_entry("b.png", 2, &hal),
        ];
        assert_eq!(
            library_hint(&two, "", 2, 0).as_deref(),
            Some("2 prints · hal9000")
        );
    }

    #[test]
    fn library_hint_multiple_machines_says_all_machines() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let entries = vec![local_entry("a.png", 2), remote_entry("b.png", 1, &hal)];
        assert_eq!(
            library_hint(&entries, "", 2, 0).as_deref(),
            Some("2 prints · all machines")
        );
    }

    #[test]
    fn library_hint_appends_offline_host_count() {
        let entries = vec![local_entry("a.png", 1)];
        assert_eq!(
            library_hint(&entries, "", 1, 1).as_deref(),
            Some("1 print · 1 host offline")
        );
        assert_eq!(
            library_hint(&entries, "", 1, 2).as_deref(),
            Some("1 print · 2 hosts offline")
        );
    }

    #[test]
    fn library_hint_filter_shows_match_ratio_and_query() {
        let entries = vec![local_entry("a.png", 1), local_entry("b.png", 2)];
        assert_eq!(
            library_hint(&entries, "cat", 1, 0).as_deref(),
            Some("1/2 prints · /cat")
        );
    }

    #[test]
    fn library_hint_empty_gallery_is_none() {
        assert_eq!(library_hint(&[], "", 0, 3), None);
    }

    #[test]
    fn library_hint_loopback_server_counts_as_local() {
        // Prints listed by the connected loopback server carry a
        // this-machine origin WITH a url — they must read as local, not
        // as a remote host named localhost.
        let origin = GalleryOrigin {
            host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
            url: Some("http://localhost:7680".into()),
            name: crate::hosts::local_display_name().to_string(),
        };
        let entries = vec![remote_entry("a.png", 1, &origin)];
        assert_eq!(library_hint(&entries, "", 1, 0).as_deref(), Some("1 print"));
    }

    // ── legacy invariants ───────────────────────────────────────

    #[test]
    #[serial_test::serial(mold_env)]
    fn default_gallery_dir_contains_output() {
        let dir = default_gallery_dir();
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains("output"),
            "default_gallery_dir should contain 'output': {dir_str}"
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn default_gallery_dir_under_mold() {
        let dir = default_gallery_dir();
        let mold_dir = mold_core::Config::mold_dir().expect("mold dir should resolve in tests");
        assert!(
            dir.starts_with(&mold_dir),
            "default_gallery_dir should be under mold dir: {} (mold dir: {})",
            dir.display(),
            mold_dir.display()
        );
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn scan_images_local_returns_empty_for_nonexistent_dir() {
        let entries = scan_images_local();
        let _ = entries;
    }

    /// A PNG that passes the shared gallery validity guards.
    fn write_valid_png(path: &Path) {
        let img = image::RgbaImage::from_fn(64, 64, |x, y| {
            image::Rgba([
                (x * 37 % 251) as u8,
                (y * 91 % 241) as u8,
                ((x ^ y) * 13 % 233) as u8,
                255,
            ])
        });
        image::DynamicImage::ImageRgba8(img).save(path).unwrap();
    }

    /// Run `f` with `MOLD_OUTPUT_DIR` pinned to `dir` (inside the isolated
    /// env, which already holds the env mutex), restoring it afterwards.
    fn with_output_dir<F: FnOnce()>(dir: &Path, f: F) {
        let prev = std::env::var("MOLD_OUTPUT_DIR").ok();
        std::env::set_var("MOLD_OUTPUT_DIR", dir);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match prev {
            Some(v) => std::env::set_var("MOLD_OUTPUT_DIR", v),
            None => std::env::remove_var("MOLD_OUTPUT_DIR"),
        }
        if let Err(payload) = res {
            std::panic::resume_unwind(payload);
        }
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn scan_local_lists_only_live_rows_and_carries_titles() {
        // A trashed print keeps its row (flagged) and its bytes under
        // `.trash/`, but the Library must not list it; a live print's
        // editable title rides along so the grid/details can show it.
        crate::test_env::with_isolated_env(|home| {
            let gallery = home.join("output");
            std::fs::create_dir_all(&gallery).unwrap();
            write_valid_png(&gallery.join("mold-live-0001.png"));
            write_valid_png(&gallery.join("mold-binned-0002.png"));
            let db = mold_db::open_default().unwrap().expect("isolated DB");
            db.reconcile(&gallery).unwrap();
            db.set_title(&gallery, "mold-live-0001.png", Some("Lighthouse study"))
                .unwrap();
            crate::gallery_trash::trash_local_print(
                &db,
                &gallery.join("mold-binned-0002.png"),
                1_700_000_000_000,
            )
            .unwrap();

            with_output_dir(&gallery, || {
                let scan = scan_local();
                assert!(scan.trash_available, "DB answered ⇒ trash available");
                let names: Vec<String> = scan.entries.iter().map(|e| e.filename()).collect();
                assert_eq!(names, vec!["mold-live-0001.png".to_string()]);
                assert_eq!(
                    scan.entries[0].title.as_deref(),
                    Some("Lighthouse study"),
                    "title carried from the gallery row"
                );
                assert!(scan.entries[0].origins[0].is_local());

                // Nothing live at all: the disk walk finds nothing either
                // (`.trash/` is a directory, below the depth-1 walk) —
                // but the trash stays available for the next delete.
                crate::gallery_trash::trash_local_print(
                    &db,
                    &gallery.join("mold-live-0001.png"),
                    1_700_000_000_001,
                )
                .unwrap();
                let scan = scan_local();
                assert!(scan.entries.is_empty(), "{:?}", scan.entries);
                assert!(scan.trash_available);
            });
        });
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn scan_local_without_the_db_walks_the_disk_and_offers_no_trash() {
        crate::test_env::with_isolated_env(|home| {
            std::env::set_var("MOLD_DB_DISABLE", "1");
            let gallery = home.join("output");
            std::fs::create_dir_all(&gallery).unwrap();
            write_valid_png(&gallery.join("mold-disk-0001.png"));
            with_output_dir(&gallery, || {
                let scan = scan_local();
                assert_eq!(scan.entries.len(), 1);
                assert_eq!(scan.entries[0].title, None);
                assert!(
                    !scan.trash_available,
                    "no DB ⇒ a local delete stays a hard delete"
                );
            });
            std::env::remove_var("MOLD_DB_DISABLE");
        });
    }

    #[test]
    fn merge_scans_keeps_local_trash_availability_from_the_local_source() {
        let hal = remote_origin("hal9000-7680", "hal9000");
        let merged = merge_scans(vec![
            SourceScan {
                origin: GalleryOrigin::local(),
                entries: Some(vec![local_entry("a.png", 1)]),
                local_trash_available: true,
            },
            SourceScan {
                origin: hal.clone(),
                entries: Some(vec![remote_entry("b.png", 2, &hal)]),
                local_trash_available: false,
            },
        ]);
        assert!(merged.local_trash_available);
        let merged = merge_scans(vec![SourceScan {
            origin: GalleryOrigin::local(),
            entries: Some(vec![local_entry("a.png", 1)]),
            local_trash_available: false,
        }]);
        assert!(!merged.local_trash_available);
    }

    #[test]
    fn is_video_filename_matches_known_extensions() {
        assert!(is_video_filename("out.mp4"));
        assert!(is_video_filename("OUT.MP4"));
        assert!(is_video_filename("clip.webm"));
        assert!(is_video_filename("clip.mov"));
        assert!(is_video_filename("clip.mkv"));

        // Raster formats that the preview endpoint shouldn't be tried on.
        assert!(!is_video_filename("frame.png"));
        assert!(!is_video_filename("frame.jpg"));
        assert!(!is_video_filename("frame.gif"));
        assert!(!is_video_filename("frame.apng"));
        assert!(!is_video_filename("frame.webp"));
        assert!(!is_video_filename("no_extension"));
    }
}
