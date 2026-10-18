//! Fixed, bounded, crash-safe Library copy/sync provenance. No path crosses IPC.
use std::io::{Read, Write};
use std::path::Path;
use tauri::Manager;
const LIMIT: u64 = 8 * 1024 * 1024;
static WRITER: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn validate(value: &serde_json::Value) -> Result<(), String> {
    if value["version"].as_u64() != Some(1)
        || !value["receipts"].is_object()
        || !value["keep"].is_object()
    {
        return Err("Unsupported Library sync journal; existing records were preserved.".into());
    }
    Ok(())
}
fn read(path: &Path) -> Result<Option<serde_json::Value>, String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Could not inspect Library sync journal: {error}")),
    };
    if !metadata.is_file() || metadata.len() > LIMIT {
        return Err("Library sync journal is not a bounded regular file.".into());
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| file.take(LIMIT + 1).read_to_end(&mut bytes))
        .map_err(|error| format!("Could not read Library sync journal: {error}"))?;
    if bytes.len() as u64 > LIMIT {
        return Err("Library sync journal exceeds its size limit.".into());
    }
    let value = serde_json::from_slice(&bytes)
        .map_err(|_| "Library sync journal is damaged; existing records were preserved.")?;
    validate(&value)?;
    Ok(Some(value))
}
fn write(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    validate(value)?;
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    if bytes.len() as u64 > LIMIT {
        return Err("Library sync journal exceeds its size limit.".into());
    }
    let parent = path
        .parent()
        .ok_or("Library app-data directory is unavailable.")?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    // Preserve invalid/foreign prior records rather than overwriting them.
    let _prior = read(path)?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    temporary
        .write_all(&bytes)
        .map_err(|error| error.to_string())?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    temporary.persist(path).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(())
}
#[tauri::command]
pub async fn library_sync_read(app: tauri::AppHandle) -> Result<Option<serde_json::Value>, String> {
    let path = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("library-sync-v1.json");
    tauri::async_runtime::spawn_blocking(move || {
        let _writer = WRITER.lock().map_err(|_| "Library sync journal is busy.")?;
        read(&path)
    })
    .await
    .map_err(|error| format!("Could not read Library sync journal: {error}"))?
}
#[tauri::command]
pub async fn library_sync_write(
    app: tauri::AppHandle,
    document: serde_json::Value,
) -> Result<(), String> {
    let path = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("library-sync-v1.json");
    tauri::async_runtime::spawn_blocking(move || {
        let _writer = WRITER.lock().map_err(|_| "Library sync journal is busy.")?;
        write(&path, &document)
    })
    .await
    .map_err(|error| format!("Could not write Library sync journal: {error}"))?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_and_invalid_update_preserves_prior_journal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library-sync-v1.json");
        assert_eq!(read(&path).unwrap(), None);
        let document = serde_json::json!({"version":1,"receipts":{},"keep":{"destination":true}});
        write(&path, &document).unwrap();
        assert_eq!(read(&path).unwrap(), Some(document.clone()));
        assert!(write(&path, &serde_json::json!({"version":2})).is_err());
        assert_eq!(read(&path).unwrap(), Some(document));
    }
    #[test]
    fn corrupt_or_oversized_journal_is_never_reset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("library-sync-v1.json");
        std::fs::write(&path, b"damaged").unwrap();
        let document = serde_json::json!({"version":1,"receipts":{},"keep":{}});
        assert!(write(&path, &document).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"damaged");
        std::fs::File::create(&path)
            .unwrap()
            .set_len(LIMIT + 1)
            .unwrap();
        assert!(read(&path).is_err());
    }
}
