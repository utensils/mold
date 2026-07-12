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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Theme {
    #[default]
    System,
    Dark,
    Light,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ThemeFamily {
    #[default]
    Safelight,
    Mold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum UpdateChannel {
    #[default]
    Stable,
    Nightly,
}

fn default_true() -> bool {
    true
}

fn legacy_theme_family() -> ThemeFamily {
    ThemeFamily::Safelight
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppSettings {
    pub mode: ConnectionMode,
    pub remote_url: Option<String>,
    /// Legacy plaintext slot — new saves go to the Keychain (secrets.rs);
    /// kept for reading old settings.json files.
    pub remote_api_key: Option<String>,
    pub last_route: Option<String>,
    /// Environment applied to the embedded engine at start (Performance knobs).
    pub engine_env: std::collections::HashMap<String, String>,
    pub theme: Theme,
    #[serde(default = "legacy_theme_family")]
    pub theme_family: ThemeFamily,
    #[serde(default = "default_true")]
    pub notifications: bool,
    #[serde(default = "default_true")]
    pub dock_badge: bool,
    pub restore_last_route: bool,
    pub runpod_include_hf_token: bool,
    pub runpod_network_volume_id: Option<String>,
    /// Whole-webview scale, stored as a percentage (80-130).
    pub ui_scale_percent: u16,
    /// Signed desktop release stream. Stable is deliberately the migration
    /// default; opting into main-branch builds must always be explicit.
    pub update_channel: UpdateChannel,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            mode: ConnectionMode::default(),
            remote_url: None,
            remote_api_key: None,
            last_route: None,
            engine_env: Default::default(),
            theme: Theme::default(),
            theme_family: ThemeFamily::Mold,
            notifications: true,
            dock_badge: true,
            restore_last_route: false,
            runpod_include_hf_token: false,
            runpod_network_volume_id: None,
            ui_scale_percent: 100,
            update_channel: UpdateChannel::Stable,
        }
    }
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
            remote_api_key: Some("k".into()),
            last_route: Some("/gallery".into()),
            engine_env: [("MOLD_VAE_TILED".to_string(), "force".to_string())]
                .into_iter()
                .collect(),
            theme: Theme::Light,
            theme_family: ThemeFamily::Mold,
            notifications: false,
            dock_badge: true,
            restore_last_route: true,
            runpod_include_hf_token: true,
            runpod_network_volume_id: Some("nv-models".into()),
            ui_scale_percent: 120,
            update_channel: UpdateChannel::Nightly,
        };
        save(&path, &settings).unwrap();
        assert_eq!(load(&path), settings);
    }

    #[test]
    fn legacy_settings_json_gets_pref_defaults() {
        // Files written before the prefs existed must load with notifications
        // and the dock badge ON, not silently disabled.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"local","lastRoute":"/generate"}"#).unwrap();
        let loaded = load(&path);
        assert!(loaded.notifications);
        assert!(loaded.dock_badge);
        assert_eq!(loaded.theme, Theme::System);
        assert_eq!(loaded.theme_family, ThemeFamily::Safelight);
        assert!(loaded.engine_env.is_empty());
        assert!(!loaded.runpod_include_hf_token);
        assert_eq!(loaded.runpod_network_volume_id, None);
        assert_eq!(loaded.ui_scale_percent, 100);
        assert_eq!(loaded.update_channel, UpdateChannel::Stable);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load(&path_in(&dir)), AppSettings::default());
        assert_eq!(AppSettings::default().mode, ConnectionMode::Local);
        assert_eq!(AppSettings::default().theme, Theme::System);
        assert_eq!(AppSettings::default().theme_family, ThemeFamily::Mold);
        assert_eq!(AppSettings::default().update_channel, UpdateChannel::Stable);
    }

    #[test]
    fn update_channel_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        let settings = AppSettings {
            update_channel: UpdateChannel::Nightly,
            ..AppSettings::default()
        };

        save(&path, &settings).unwrap();

        assert_eq!(load(&path).update_channel, UpdateChannel::Nightly);
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
        let expected = AppSettings {
            theme_family: ThemeFamily::Safelight,
            ..AppSettings::default()
        };
        assert_eq!(load(&path), expected);
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/settings.json");
        save(&path, &AppSettings::default()).unwrap();
        assert!(path.exists());
    }
}
