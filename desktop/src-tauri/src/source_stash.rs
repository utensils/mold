//! Local stash of img2img source images, keyed by the SHA-256 of the exact
//! bytes that shipped in the request — the same hash the server records as
//! `OutputMetadata::source_image_sha256`. Reuse settings reads the stash
//! first (covers uploads and canvas-fitted sources that exist nowhere else),
//! then falls back to a gallery filename match. Content-addressed writes are
//! idempotent; the stash is bounded by pruning the oldest entries.

use std::path::{Path, PathBuf};

const STASH_DIR: &str = "source-stash";
/// Upper bound on stashed sources; oldest-by-mtime pruned past this. At the
/// 1–10 MB typical source size this caps the stash around 64–640 MB.
const MAX_STASH_FILES: usize = 64;

/// The hash doubles as the filename — accept only exact lowercase-hex
/// SHA-256 so a malicious value can't traverse out of the stash dir.
fn is_valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
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

/// Remove oldest-by-mtime entries until at most `keep` remain.
fn prune_oldest(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let meta = entry.metadata().ok()?;
            if !meta.is_file() {
                return None;
            }
            Some((
                meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                entry.path(),
            ))
        })
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|(mtime, _)| *mtime);
    for (_, path) in files.iter().take(files.len() - keep) {
        let _ = std::fs::remove_file(path);
    }
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
        let path = dir.join(&sha256);
        // Content-addressed: an existing entry is already the same bytes.
        if !path.exists() {
            std::fs::write(&path, bytes).map_err(|error| error.to_string())?;
        }
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
    tauri::async_runtime::spawn_blocking(move || {
        use base64::Engine;
        let path = dir.join(&sha256);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(
                base64::engine::general_purpose::STANDARD.encode(bytes),
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.to_string()),
        }
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
}
