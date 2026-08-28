//! Persistent, content-addressed thumbnail cache — the desktop's equivalent
//! of Lightroom's preview cache. Every Library tile the app has ever shown
//! stays on disk under the app data dir, so a cold launch paints the grid
//! from local files and never asks a host for a thumbnail it already holds.
//!
//! Key: `sha256(origin \0 filename \0 media_version \0 size)`. `origin` is
//! `"local"` for This device (online and offline read the same files, so
//! they share one key) or a digest of the remote's base URL — deliberately
//! NOT its API key, so rotating a key never invalidates a cache. The wire's
//! `media_version` (`mtime_ms:size`) is part of the key, which is why a hit
//! never revalidates: a changed file gets a new version, hence a new key,
//! and the old entry ages out. That beats an `If-None-Match` round trip,
//! and the server's own ETag (`mtime_nanos-len`) cannot be derived from the
//! wire version anyway.
//!
//! Layout is `<root>/<digest[..2]>/<digest>.bin`, written temp-file +
//! rename (atomic on one filesystem). Bounded by bytes AND file count with
//! LRU eviction by mtime (touched on read, at most hourly, like
//! `source_stash.rs`). `get` sniffs the magic bytes and deletes-and-misses
//! on anything unrecognised, so a torn or foreign file is never served.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Weak},
    time::{Duration, SystemTime},
};

pub const CACHE_DIR: &str = "thumbnail-cache/v1";
pub const MAX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_CACHE_FILES: usize = 20_000;
/// One tile; a 512 px JPEG is tens of KB, a PNG a few hundred KB.
pub const MAX_ENTRY_BYTES: usize = 2 * 1024 * 1024;
const TOUCH_INTERVAL: Duration = Duration::from_secs(60 * 60);
const STALE_TMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// The two retina-aware tile sizes the grid asks for. Bounding the set keeps
/// the cache from holding one entry per slider position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SizeTier {
    S256,
    S512,
}

impl SizeTier {
    pub fn pixels(self) -> u32 {
        match self {
            SizeTier::S256 => 256,
            SizeTier::S512 => 512,
        }
    }
}

impl TryFrom<u32> for SizeTier {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            256 => Ok(SizeTier::S256),
            512 => Ok(SizeTier::S512),
            other => Err(format!(
                "Unsupported thumbnail size {other}; use 256 or 512."
            )),
        }
    }
}

/// Immutable identity of one tile.
#[derive(Clone, Copy, Debug)]
pub struct ThumbKey<'a> {
    pub origin: &'a str,
    pub filename: &'a str,
    pub media_version: &'a str,
    pub size: SizeTier,
}

impl ThumbKey<'_> {
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.origin.as_bytes());
        hasher.update(b"\0");
        hasher.update(self.filename.as_bytes());
        hasher.update(b"\0");
        hasher.update(self.media_version.as_bytes());
        hasher.update(b"\0");
        hasher.update(self.size.pixels().to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// The origin half of a key: This device passes through as `local`; a remote
/// host is the first 16 hex digits of its base URL's digest. The API key is
/// never part of it.
pub fn origin_for(cache_key: &str, base_url: Option<&str>) -> String {
    if cache_key == "local" {
        return "local".to_string();
    }
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(
        base_url
            .unwrap_or(cache_key)
            .trim_end_matches('/')
            .as_bytes(),
    );
    let hex = format!("{:x}", hasher.finalize());
    hex[..16].to_string()
}

/// An origin can only be `local` or hex, so it can never traverse a path.
pub fn valid_origin(origin: &str) -> bool {
    origin == "local"
        || (!origin.is_empty()
            && origin.len() <= 64
            && origin.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
}

pub struct CachedThumb {
    pub bytes: Vec<u8>,
    pub content_type: &'static str,
}

/// Recognise the formats a thumbnail can legitimately be; anything else is
/// refused rather than handed to the webview.
pub fn sniff_content_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF8") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        let head = &bytes[..bytes.len().min(256)];
        let text = std::str::from_utf8(head).ok()?;
        let trimmed = text.trim_start_matches(['\u{feff}', ' ', '\n', '\r', '\t']);
        if trimmed.starts_with("<svg") || trimmed.starts_with("<?xml") {
            Some("image/svg+xml")
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CacheLimits {
    pub max_bytes: u64,
    pub max_files: usize,
}

struct CacheIndex {
    total_bytes: u64,
    files: HashMap<String, (u64, SystemTime)>,
}

pub struct ThumbnailCache {
    root: PathBuf,
    limits: CacheLimits,
    index: Mutex<Option<CacheIndex>>,
    flights: Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

impl ThumbnailCache {
    pub fn new(root: PathBuf) -> Self {
        Self::with_limits(
            root,
            CacheLimits {
                max_bytes: MAX_CACHE_BYTES,
                max_files: MAX_CACHE_FILES,
            },
        )
    }

    pub fn with_limits(root: PathBuf, limits: CacheLimits) -> Self {
        Self {
            root,
            limits,
            index: Mutex::new(None),
            flights: Mutex::new(HashMap::new()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path_for(&self, digest: &str) -> PathBuf {
        self.root.join(&digest[..2]).join(format!("{digest}.bin"))
    }

    /// Stat only — the prewarm planner's probe.
    pub fn contains(&self, digest: &str) -> bool {
        valid_digest(digest) && self.path_for(digest).is_file()
    }

    /// The bytes and sniffed type of a cached tile. Anything unrecognised is
    /// deleted and reported as a miss.
    pub fn get(&self, digest: &str) -> Result<Option<CachedThumb>, String> {
        if !valid_digest(digest) {
            return Err("Invalid thumbnail cache key.".into());
        }
        let path = self.path_for(digest);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("Couldn't read the thumbnail cache: {error}")),
        };
        let Some(content_type) = sniff_content_type(&bytes)
            .filter(|_| !bytes.is_empty() && bytes.len() <= MAX_ENTRY_BYTES)
        else {
            self.remove(digest);
            return Ok(None);
        };
        self.touch(digest, &path);
        Ok(Some(CachedThumb {
            bytes,
            content_type,
        }))
    }

    /// Coarse LRU: bump the mtime at most once an hour per entry.
    fn touch(&self, digest: &str, path: &Path) {
        let now = SystemTime::now();
        let stale = match std::fs::metadata(path).and_then(|m| m.modified()) {
            Ok(modified) => now.duration_since(modified).unwrap_or_default() >= TOUCH_INTERVAL,
            Err(_) => true,
        };
        if !stale {
            return;
        }
        let _ = filetime::set_file_mtime(path, filetime::FileTime::from_system_time(now));
        if let Ok(mut guard) = self.index.lock() {
            if let Some(index) = guard.as_mut() {
                if let Some(entry) = index.files.get_mut(digest) {
                    entry.1 = now;
                }
            }
        }
    }

    /// Store one tile atomically, then prune to the byte/count budget.
    pub fn put(&self, digest: &str, bytes: &[u8]) -> Result<(), String> {
        if !valid_digest(digest) {
            return Err("Invalid thumbnail cache key.".into());
        }
        if bytes.is_empty() || bytes.len() > MAX_ENTRY_BYTES {
            return Err("The thumbnail is empty or too large to cache.".into());
        }
        if sniff_content_type(bytes).is_none() {
            return Err("The thumbnail is not a recognised image.".into());
        }
        let path = self.path_for(digest);
        let dir = path
            .parent()
            .ok_or_else(|| "The thumbnail cache path has no parent.".to_string())?;
        std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = dir.join(format!("{digest}.{nanos}.tmp"));
        std::fs::write(&tmp, bytes).map_err(|error| error.to_string())?;
        if let Err(error) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(error.to_string());
        }
        let now = SystemTime::now();
        {
            let mut guard = self.index.lock().map_err(|_| "cache index poisoned")?;
            let index = self.ensure_index(&mut guard)?;
            if let Some((previous, _)) = index
                .files
                .insert(digest.to_string(), (bytes.len() as u64, now))
            {
                index.total_bytes = index.total_bytes.saturating_sub(previous);
            }
            index.total_bytes += bytes.len() as u64;
        }
        self.prune();
        Ok(())
    }

    pub fn remove(&self, digest: &str) {
        if !valid_digest(digest) {
            return;
        }
        let _ = std::fs::remove_file(self.path_for(digest));
        if let Ok(mut guard) = self.index.lock() {
            if let Some(index) = guard.as_mut() {
                if let Some((len, _)) = index.files.remove(digest) {
                    index.total_bytes = index.total_bytes.saturating_sub(len);
                }
            }
        }
    }

    /// Evict least-recently-used entries until both budgets hold, and reap
    /// abandoned temp files older than a day.
    pub fn prune(&self) {
        let Ok(mut guard) = self.index.lock() else {
            return;
        };
        let Ok(index) = self.ensure_index(&mut guard) else {
            return;
        };
        if index.total_bytes <= self.limits.max_bytes && index.files.len() <= self.limits.max_files
        {
            return;
        }
        let mut by_age: Vec<(String, u64, SystemTime)> = index
            .files
            .iter()
            .map(|(digest, (len, mtime))| (digest.clone(), *len, *mtime))
            .collect();
        by_age.sort_by_key(|(_, _, mtime)| *mtime);
        for (digest, len, _) in by_age {
            if index.total_bytes <= self.limits.max_bytes
                && index.files.len() <= self.limits.max_files
            {
                break;
            }
            let _ = std::fs::remove_file(self.path_for(&digest));
            index.files.remove(&digest);
            index.total_bytes = index.total_bytes.saturating_sub(len);
        }
    }

    /// Bytes and files currently accounted for.
    pub fn usage(&self) -> Result<(u64, usize), String> {
        let mut guard = self.index.lock().map_err(|_| "cache index poisoned")?;
        let index = self.ensure_index(&mut guard)?;
        Ok((index.total_bytes, index.files.len()))
    }

    /// One in-flight fetch per digest: the second tile asking for the same
    /// bytes waits for the first rather than fetching again (the same
    /// weak-map pattern as the server's `thumbnail_singleflight`).
    pub fn singleflight(&self, digest: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut flights = self.flights.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = flights.get(digest).and_then(Weak::upgrade) {
            return existing;
        }
        let fresh = Arc::new(tokio::sync::Mutex::new(()));
        flights.insert(digest.to_string(), Arc::downgrade(&fresh));
        flights.retain(|_, weak| weak.strong_count() > 0);
        fresh
    }

    /// Build the index lazily with one walk of the two-level layout; also
    /// the moment stale temp files are reaped.
    fn ensure_index<'a>(
        &self,
        guard: &'a mut Option<CacheIndex>,
    ) -> Result<&'a mut CacheIndex, String> {
        if guard.is_none() {
            let mut index = CacheIndex {
                total_bytes: 0,
                files: HashMap::new(),
            };
            let now = SystemTime::now();
            if self.root.is_dir() {
                for shard in std::fs::read_dir(&self.root).map_err(|e| e.to_string())? {
                    let shard = shard.map_err(|e| e.to_string())?.path();
                    if !shard.is_dir() {
                        continue;
                    }
                    for entry in std::fs::read_dir(&shard).map_err(|e| e.to_string())? {
                        let entry = entry.map_err(|e| e.to_string())?;
                        let path = entry.path();
                        let Ok(metadata) = entry.metadata() else {
                            continue;
                        };
                        if !metadata.is_file() {
                            continue;
                        }
                        let name = entry.file_name();
                        let name = name.to_string_lossy();
                        if name.ends_with(".tmp") {
                            let age = metadata
                                .modified()
                                .ok()
                                .and_then(|m| now.duration_since(m).ok())
                                .unwrap_or(STALE_TMP_AGE);
                            if age >= STALE_TMP_AGE {
                                let _ = std::fs::remove_file(&path);
                            }
                            continue;
                        }
                        let Some(digest) = name.strip_suffix(".bin") else {
                            continue;
                        };
                        if !valid_digest(digest) {
                            continue;
                        }
                        let mtime = metadata.modified().unwrap_or(now);
                        index.total_bytes += metadata.len();
                        index
                            .files
                            .insert(digest.to_string(), (metadata.len(), mtime));
                    }
                }
            }
            *guard = Some(index);
        }
        Ok(guard.as_mut().expect("index populated above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3];
    const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 0, 1, 2, 3];

    fn key<'a>(
        origin: &'a str,
        filename: &'a str,
        version: &'a str,
        size: SizeTier,
    ) -> ThumbKey<'a> {
        ThumbKey {
            origin,
            filename,
            media_version: version,
            size,
        }
    }

    fn cache_in(dir: &tempfile::TempDir) -> ThumbnailCache {
        ThumbnailCache::new(dir.path().join("thumbs"))
    }

    #[test]
    fn miss_then_put_then_hit_round_trips_bytes_and_type() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(&dir);
        let digest = key("local", "a.png", "1:10", SizeTier::S256).digest();
        assert!(cache.get(&digest).unwrap().is_none());
        assert!(!cache.contains(&digest));

        cache.put(&digest, PNG).unwrap();
        assert!(cache.contains(&digest));
        let hit = cache.get(&digest).unwrap().unwrap();
        assert_eq!(hit.bytes, PNG);
        assert_eq!(hit.content_type, "image/png");

        let shard = cache.root().join(&digest[..2]);
        let leftovers: Vec<_> = std::fs::read_dir(&shard)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no temp file left behind: {leftovers:?}"
        );
    }

    #[test]
    fn version_and_tier_are_part_of_the_key_but_the_api_key_is_not() {
        let a = key("local", "a.png", "1:10", SizeTier::S256).digest();
        let b = key("local", "a.png", "2:10", SizeTier::S256).digest();
        let c = key("local", "a.png", "1:10", SizeTier::S512).digest();
        let d = key("local", "b.png", "1:10", SizeTier::S256).digest();
        assert_ne!(a, b, "a changed media_version is a different tile");
        assert_ne!(a, c, "a different tier is a different tile");
        assert_ne!(a, d);
        assert_eq!(a, key("local", "a.png", "1:10", SizeTier::S256).digest());

        // The origin derives from the base URL alone; keys never enter it.
        let plato = origin_for("plato-7680", Some("http://plato:7680"));
        assert_eq!(plato, origin_for("plato-7680", Some("http://plato:7680/")));
        assert_ne!(plato, origin_for("hal", Some("http://hal9000:7680")));
        assert_eq!(origin_for("local", Some("http://127.0.0.1:1")), "local");
        assert!(valid_origin(&plato));
        assert!(valid_origin("local"));
        assert!(!valid_origin("../etc"));
        assert!(!valid_origin(""));
    }

    #[test]
    fn prunes_least_recently_used_by_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::with_limits(
            dir.path().join("thumbs"),
            CacheLimits {
                max_bytes: 3 * PNG.len() as u64,
                max_files: 100,
            },
        );
        let digests: Vec<String> = (0..5)
            .map(|i| key("local", &format!("{i}.png"), "1:1", SizeTier::S256).digest())
            .collect();
        for (i, digest) in digests.iter().enumerate() {
            cache.put(digest, PNG).unwrap();
            // Distinct mtimes so LRU order is unambiguous.
            let when = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000 + i as u64);
            filetime::set_file_mtime(
                cache.path_for(digest),
                filetime::FileTime::from_system_time(when),
            )
            .unwrap();
            if let Ok(mut guard) = cache.index.lock() {
                guard.as_mut().unwrap().files.get_mut(digest).unwrap().1 = when;
            }
        }
        cache.prune();
        let (bytes, files) = cache.usage().unwrap();
        assert_eq!(files, 3);
        assert_eq!(bytes, 3 * PNG.len() as u64);
        assert!(!cache.contains(&digests[0]));
        assert!(!cache.contains(&digests[1]));
        assert!(cache.contains(&digests[4]));
    }

    #[test]
    fn prunes_by_file_count() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::with_limits(
            dir.path().join("thumbs"),
            CacheLimits {
                max_bytes: u64::MAX,
                max_files: 2,
            },
        );
        for i in 0..4 {
            let digest = key("local", &format!("{i}.png"), "1:1", SizeTier::S256).digest();
            cache.put(&digest, JPEG).unwrap();
        }
        assert_eq!(cache.usage().unwrap().1, 2);
    }

    #[test]
    fn corrupt_entry_is_deleted_and_missed() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(&dir);
        let digest = key("local", "a.png", "1:10", SizeTier::S256).digest();
        let path = cache.path_for(&digest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not an image at all").unwrap();
        assert!(cache.get(&digest).unwrap().is_none());
        assert!(!path.exists());
        // And a put refuses foreign bytes outright.
        assert!(cache.put(&digest, b"garbage").is_err());
        assert!(cache.put(&digest, &[]).is_err());
    }

    #[test]
    fn rebuilt_index_counts_existing_files_and_reaps_stale_tmp_only() {
        let dir = tempfile::tempdir().unwrap();
        let digest = key("local", "a.png", "1:10", SizeTier::S256).digest();
        {
            let cache = cache_in(&dir);
            cache.put(&digest, PNG).unwrap();
        }
        let cache = cache_in(&dir);
        let shard = cache.root().join(&digest[..2]);
        let stale = shard.join(format!("{digest}.1.tmp"));
        let fresh = shard.join(format!("{digest}.2.tmp"));
        std::fs::write(&stale, b"x").unwrap();
        std::fs::write(&fresh, b"x").unwrap();
        let old = SystemTime::now() - STALE_TMP_AGE - Duration::from_secs(60);
        filetime::set_file_mtime(&stale, filetime::FileTime::from_system_time(old)).unwrap();

        let (bytes, files) = cache.usage().unwrap();
        assert_eq!(files, 1);
        assert_eq!(bytes, PNG.len() as u64);
        assert!(!stale.exists(), "stale temp files are reaped");
        assert!(fresh.exists(), "a temp file mid-write is left alone");
    }

    #[test]
    fn singleflight_shares_one_lock_per_digest() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(&dir);
        let a = cache.singleflight("d1");
        let b = cache.singleflight("d1");
        let c = cache.singleflight("d2");
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
        drop(a);
        drop(b);
        let d = cache.singleflight("d1");
        assert_eq!(
            Arc::strong_count(&d),
            1,
            "a released flight is minted fresh"
        );
    }

    #[test]
    fn sniffs_every_thumbnail_format_and_nothing_else() {
        assert_eq!(sniff_content_type(PNG), Some("image/png"));
        assert_eq!(sniff_content_type(JPEG), Some("image/jpeg"));
        assert_eq!(sniff_content_type(b"GIF89a...."), Some("image/gif"));
        assert_eq!(
            sniff_content_type(b"RIFF\0\0\0\0WEBPVP8 "),
            Some("image/webp")
        );
        assert_eq!(
            sniff_content_type(b"<svg xmlns=\"x\"/>"),
            Some("image/svg+xml")
        );
        assert_eq!(
            sniff_content_type(b"  <?xml version=\"1.0\"?><svg/>"),
            Some("image/svg+xml")
        );
        assert_eq!(sniff_content_type(b"<html></html>"), None);
        assert_eq!(sniff_content_type(b""), None);
        assert_eq!(sniff_content_type(b"\x7fELF"), None);
    }

    #[test]
    fn rejects_traversal_shaped_digests() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(&dir);
        assert!(cache.get("../../etc/passwd").is_err());
        assert!(cache.put("..", PNG).is_err());
        assert!(!cache.contains("zz"));
    }
}
