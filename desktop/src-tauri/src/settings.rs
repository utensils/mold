//! App-local settings, stored as `settings.json` under the Tauri
//! `app_data_dir`. These are window/app preferences only — engine
//! configuration stays in mold's own config stores (config.toml + mold.db).

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectionMode {
    #[default]
    Local,
    Remote,
    Off,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct AppSettings {
    pub mode: ConnectionMode,
    pub remote_url: Option<String>,
    pub last_route: Option<String>,
}

/// Load settings from `path`. A missing or unreadable file yields defaults —
/// settings must never block app startup.
pub fn load(path: &Path) -> AppSettings {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|e| {
            tracing::warn!("settings.json is invalid ({e}); using defaults");
            AppSettings::default()
        }),
        Err(_) => AppSettings::default(),
    }
}

/// Persist settings atomically (write to a sibling temp file, then rename).
pub fn save(path: &Path, settings: &AppSettings) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(settings)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_in(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("settings.json")
    }

    #[test]
    fn round_trips_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        let settings = AppSettings {
            mode: ConnectionMode::Remote,
            remote_url: Some("http://studio.local:7680".into()),
            last_route: Some("/gallery".into()),
        };
        save(&path, &settings).unwrap();
        assert_eq!(load(&path), settings);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load(&path_in(&dir)), AppSettings::default());
        assert_eq!(AppSettings::default().mode, ConnectionMode::Local);
    }

    #[test]
    fn corrupt_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, "not json {").unwrap();
        assert_eq!(load(&path), AppSettings::default());
    }

    #[test]
    fn unknown_fields_are_ignored_for_forward_compat() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"local","futureField":42}"#).unwrap();
        assert_eq!(load(&path), AppSettings::default());
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/settings.json");
        save(&path, &AppSettings::default()).unwrap();
        assert!(path.exists());
    }
}
