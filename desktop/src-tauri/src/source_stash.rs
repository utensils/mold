//! Local stash of img2img source images, keyed by the SHA-256 of the exact
//! bytes that shipped in the request — the same hash the server records as
//! `OutputMetadata::source_image_sha256`. Reuse settings reads the stash
//! first (covers uploads and canvas-fitted sources that exist nowhere else),
//! then falls back to a gallery filename match.
//!
//! Integrity: the key IS the content hash, and both directions enforce it —
//! `put` recomputes the digest of the decoded bytes and rejects a client
//! whose hash doesn't match; `get` re-verifies the digest of what it read
//! and deletes-and-misses on mismatch, so a torn write can never be
//! restored as an image. Writes are temp-file + rename (atomic on one
//! filesystem). Reads and repeat puts bump mtime so pruning is true LRU,
//! not FIFO-by-first-write.

use std::path::{Path, PathBuf};

const STASH_DIR: &str = "source-stash";
const SOURCE_PATHS_DIR: &str = "source-paths";
const MAX_SOURCE_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SOURCE_PATHS: usize = 256;
/// Upper bound on stashed sources; least-recently-used pruned past this. At
/// the 1–10 MB typical source size this caps the stash around 64–640 MB.
const MAX_STASH_FILES: usize = 64;

/// The hash doubles as the filename — accept only exact hex SHA-256 so a
/// malicious value can't traverse out of the stash dir.
fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn touch(path: &Path) {
    let _ = filetime::set_file_mtime(path, filetime::FileTime::now());
}

fn stash_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join(STASH_DIR);
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    Ok(dir)
}

fn source_paths_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join(SOURCE_PATHS_DIR);
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    Ok(dir)
}

/// Remember where a native desktop picker/drop found these exact bytes. The
/// path stays local to this app and is keyed by content hash; it is never sent
/// to a server or embedded in gallery metadata.
pub(crate) fn remember_source_path(
    app: &tauri::AppHandle,
    sha256: &str,
    path: &Path,
) -> Result<(), String> {
    use std::os::unix::ffi::OsStrExt;

    if !is_valid_sha256_hex(sha256) {
        return Err("invalid sha256 key".to_string());
    }
    let dir = source_paths_dir(app)?;
    let destination = dir.join(sha256);
    let tmp = dir.join(format!("{sha256}.tmp"));
    std::fs::write(&tmp, path.as_os_str().as_bytes()).map_err(|error| error.to_string())?;
    std::fs::rename(&tmp, destination).map_err(|error| error.to_string())?;
    prune_oldest(&dir, MAX_SOURCE_PATHS);
    Ok(())
}

fn remembered_source_read(dir: &Path, sha256: &str) -> Result<Option<Vec<u8>>, String> {
    use std::os::unix::ffi::OsStringExt;

    let pointer = dir.join(sha256);
    let path_bytes = match std::fs::read(&pointer) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    let path = PathBuf::from(std::ffi::OsString::from_vec(path_bytes));
    let mut bytes = Vec::new();
    let read_result = std::fs::File::open(path).and_then(|file| {
        use std::io::Read;
        file.take(MAX_SOURCE_IMAGE_BYTES + 1)
            .read_to_end(&mut bytes)
    });
    match read_result {
        Ok(_) if bytes.len() as u64 <= MAX_SOURCE_IMAGE_BYTES => {}
        Ok(_) => {
            let _ = std::fs::remove_file(pointer);
            return Ok(None);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let _ = std::fs::remove_file(pointer);
            return Ok(None);
        }
        Err(error) => return Err(error.to_string()),
    }
    if sha256_hex(&bytes) != sha256 {
        let _ = std::fs::remove_file(pointer);
        return Ok(None);
    }
    Ok(Some(bytes))
}

/// Remove oldest-by-mtime entries until at most `keep` remain. A concurrent
/// put's mid-write `.tmp` must neither count toward the cap (it isn't an
/// entry yet) nor be deleted out from under the writer; crash-orphaned tmps
/// are reaped separately once they're clearly abandoned.
fn prune_oldest(dir: &Path, keep: usize) {
    const ORPHANED_TMP_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tmp") {
            if mtime.elapsed().is_ok_and(|age| age > ORPHANED_TMP_AGE) {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        files.push((mtime, path));
    }
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|(mtime, _)| *mtime);
    for (_, path) in files.iter().take(files.len() - keep) {
        let _ = std::fs::remove_file(path);
    }
}

/// Verified, atomic write: rejects a key that doesn't hash the payload, and
/// a crash mid-write leaves only an ignorable `.tmp`, never a corrupt entry.
fn stash_write(dir: &Path, sha256: &str, bytes: &[u8]) -> Result<(), String> {
    if sha256_hex(bytes) != sha256 {
        return Err("sha256 key does not match the payload".to_string());
    }
    let path = dir.join(sha256);
    if path.exists() {
        // Content-addressed: same key = same verified bytes. Refresh LRU.
        touch(&path);
        return Ok(());
    }
    let tmp = dir.join(format!("{sha256}.tmp"));
    std::fs::write(&tmp, bytes).map_err(|error| error.to_string())?;
    if let Err(error) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        // Two concurrent puts of the same content race the rename; if the
        // destination exists, the OTHER writer won and the store succeeded.
        if !path.exists() {
            return Err(error.to_string());
        }
    }
    Ok(())
}

/// Verified read: a stored file whose digest no longer matches its name
/// (torn write, disk corruption) is deleted and reported as a miss.
fn stash_read(dir: &Path, sha256: &str) -> Result<Option<Vec<u8>>, String> {
    let path = dir.join(sha256);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if sha256_hex(&bytes) != sha256 {
        let _ = std::fs::remove_file(&path);
        return Ok(None);
    }
    touch(&path);
    Ok(Some(bytes))
}

#[tauri::command]
pub async fn source_stash_put(
    app: tauri::AppHandle,
    sha256: String,
    data_b64: String,
) -> Result<(), String> {
    if !is_valid_sha256_hex(&sha256) {
        return Err("invalid sha256 key".to_string());
    }
    let dir = stash_dir(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .map_err(|error| error.to_string())?;
        stash_write(&dir, &sha256, &bytes)?;
        prune_oldest(&dir, MAX_STASH_FILES);
        Ok(())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn source_stash_get(
    app: tauri::AppHandle,
    sha256: String,
) -> Result<Option<String>, String> {
    if !is_valid_sha256_hex(&sha256) {
        return Err("invalid sha256 key".to_string());
    }
    let dir = stash_dir(&app)?;
    let source_paths = source_paths_dir(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        use base64::Engine;
        let bytes = match stash_read(&dir, &sha256)? {
            Some(bytes) => Some(bytes),
            None => remembered_source_read(&source_paths, &sha256)?,
        };
        Ok(bytes.map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes)))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha_keys_are_strict_hex_so_they_cannot_escape_the_stash_dir() {
        assert!(is_valid_sha256_hex(&"a".repeat(64)));
        assert!(is_valid_sha256_hex(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!is_valid_sha256_hex("../../../etc/passwd"));
        assert!(!is_valid_sha256_hex(&"a".repeat(63)));
        assert!(!is_valid_sha256_hex(&"g".repeat(64)));
        assert!(!is_valid_sha256_hex(""));
    }

    #[test]
    fn put_rejects_a_key_that_does_not_hash_the_payload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = stash_write(dir.path(), &"a".repeat(64), b"bytes").unwrap_err();
        assert!(err.contains("does not match"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn round_trips_and_reads_verify_the_digest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = b"real-image-bytes".to_vec();
        let sha = sha256_hex(&bytes);
        stash_write(dir.path(), &sha, &bytes).unwrap();
        assert_eq!(stash_read(dir.path(), &sha).unwrap(), Some(bytes.clone()));

        // Corrupt the entry on disk (torn write): the read self-heals to a
        // miss and removes the lying file instead of restoring garbage.
        std::fs::write(dir.path().join(&sha), b"truncated").unwrap();
        assert_eq!(stash_read(dir.path(), &sha).unwrap(), None);
        assert!(!dir.path().join(&sha).exists());
    }

    #[test]
    fn prune_keeps_the_newest_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..5 {
            let path = dir.path().join(format!("{i:064}"));
            std::fs::write(&path, b"x").unwrap();
            // Distinct mtimes without sleeping.
            let t = filetime::FileTime::from_unix_time(1_000 + i, 0);
            filetime::set_file_mtime(&path, t).unwrap();
        }
        prune_oldest(dir.path(), 2);
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec![format!("{:064}", 3), format!("{:064}", 4)]);
    }

    #[test]
    fn remembered_source_reloads_only_while_the_original_bytes_still_match() {
        use std::os::unix::ffi::OsStrExt;

        let dir = tempfile::tempdir().expect("pointer dir");
        let source_dir = tempfile::tempdir().expect("source dir");
        let source = source_dir.path().join("target.png");
        let bytes = b"target bytes";
        std::fs::write(&source, bytes).unwrap();
        let sha = sha256_hex(bytes);
        std::fs::write(dir.path().join(&sha), source.as_os_str().as_bytes()).unwrap();

        assert_eq!(
            remembered_source_read(dir.path(), &sha).unwrap(),
            Some(bytes.to_vec())
        );

        std::fs::write(&source, b"changed").unwrap();
        assert_eq!(remembered_source_read(dir.path(), &sha).unwrap(), None);
        assert!(!dir.path().join(sha).exists());
    }

    #[test]
    fn prune_ignores_in_flight_tmp_files_but_reaps_abandoned_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..2 {
            let path = dir.path().join(format!("{i:064}"));
            std::fs::write(&path, b"x").unwrap();
            filetime::set_file_mtime(&path, filetime::FileTime::from_unix_time(1_000 + i, 0))
                .unwrap();
        }
        // A fresh mid-write tmp: not counted toward the cap, never deleted.
        let live_tmp = dir.path().join(format!("{}.tmp", "b".repeat(64)));
        std::fs::write(&live_tmp, b"partial").unwrap();
        // A crash orphan from days ago: reaped.
        let orphan = dir.path().join(format!("{}.tmp", "c".repeat(64)));
        std::fs::write(&orphan, b"partial").unwrap();
        filetime::set_file_mtime(&orphan, filetime::FileTime::from_unix_time(1, 0)).unwrap();

        prune_oldest(dir.path(), 2);

        assert!(live_tmp.exists(), "an in-flight tmp must survive pruning");
        assert!(!orphan.exists(), "an abandoned tmp must be reaped");
        // Both real entries kept — the tmps never counted toward the cap.
        assert!(dir.path().join(format!("{:064}", 0)).exists());
        assert!(dir.path().join(format!("{:064}", 1)).exists());
    }

    #[test]
    fn reads_and_repeat_puts_refresh_lru_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let old = b"often-reused".to_vec();
        let old_sha = sha256_hex(&old);
        stash_write(dir.path(), &old_sha, &old).unwrap();
        filetime::set_file_mtime(
            dir.path().join(&old_sha),
            filetime::FileTime::from_unix_time(1_000, 0),
        )
        .unwrap();
        let newer = b"one-off".to_vec();
        let newer_sha = sha256_hex(&newer);
        stash_write(dir.path(), &newer_sha, &newer).unwrap();
        filetime::set_file_mtime(
            dir.path().join(&newer_sha),
            filetime::FileTime::from_unix_time(2_000, 0),
        )
        .unwrap();

        // Reading the old entry marks it recently used…
        assert!(stash_read(dir.path(), &old_sha).unwrap().is_some());
        // …so pruning to one entry keeps IT, not the newer-but-idle one.
        prune_oldest(dir.path(), 1);
        assert!(dir.path().join(&old_sha).exists());
        assert!(!dir.path().join(&newer_sha).exists());
    }
}
