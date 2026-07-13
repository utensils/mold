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

/// Most-recently-used remote hosts the app remembers. Keys live in the
/// secret store under `remote-api-key.<id>`, never here.
pub const MAX_SAVED_HOSTS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedHost {
    /// Slug derived from the URL (`connection::host_id`), stable across saves.
    pub id: String,
    /// Optional friendly name (e.g. the mDNS instance name).
    #[serde(default)]
    pub name: Option<String>,
    pub url: String,
    /// Unix epoch milliseconds of the last successful connection.
    #[serde(default)]
    pub last_used_ms: Option<u64>,
}

/// Move `url` to the front of the MRU list (inserting if new), stamping
/// `last_used_ms`. The list is kept most-recent-first and capped.
pub fn upsert_saved_host(
    hosts: &mut Vec<SavedHost>,
    id: &str,
    url: &str,
    name: Option<String>,
    now_ms: u64,
) {
    // A fresh name wins, but a nameless reconnect (typed hostname, boot-time
    // restore) must not wipe the friendly name a discovery scan gave us.
    let existing_name = hosts
        .iter()
        .find(|h| h.id == id)
        .and_then(|h| h.name.clone());
    hosts.retain(|h| h.id != id);
    hosts.insert(
        0,
        SavedHost {
            id: id.to_string(),
            name: name.or(existing_name),
            url: url.to_string(),
            last_used_ms: Some(now_ms),
        },
    );
    hosts.truncate(MAX_SAVED_HOSTS);
}

/// Add `id` to the boot-reconnect set (idempotent). Every host in use —
/// whether picked as the primary ("Use this host") or added as an extra —
/// goes through this, so the whole set is restored on the next launch.
pub fn remember_connected_host(settings: &mut AppSettings, id: &str) {
    if !settings.connected_host_ids.iter().any(|h| h == id) {
        settings.connected_host_ids.push(id.to_string());
    }
}

/// Drop `id` from the saved list — and when it is also the persisted primary
/// remote, clear that preference too, otherwise the next launch would
/// reconnect via `remote_url` and re-save the host, making Forget a no-op
/// for the active host. Returns true when the primary preference was cleared
/// (the caller must also clear the shared `remote-api-key` secret).
pub fn forget_host(settings: &mut AppSettings, id: &str) -> bool {
    settings.saved_hosts.retain(|h| h.id != id);
    // The boot-reconnect set must not resurrect a forgotten host.
    settings.connected_host_ids.retain(|h| h != id);
    let was_primary = settings
        .remote_url
        .as_deref()
        .is_some_and(|url| crate::connection::host_id(url) == id);
    if was_primary {
        settings.remote_url = None;
        settings.remote_api_key = None;
        settings.mode = ConnectionMode::Local;
    }
    was_primary
}

fn legacy_theme_family() -> ThemeFamily {
    ThemeFamily::Safelight
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppSettings {
    pub mode: ConnectionMode,
    pub remote_url: Option<String>,
    /// Legacy plaintext slot — new saves go to the secret store (secrets.rs);
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
    /// Remote hosts the app has connected to, most recent first.
    pub saved_hosts: Vec<SavedHost>,
    /// Additional hosts (beyond the primary connection) to reconnect at boot.
    pub connected_host_ids: Vec<String>,
    /// Sticky generation-target host id; `None` routes automatically.
    pub generate_target_host: Option<String>,
    /// Also save generations from remote hosts into this Mac's gallery.
    #[serde(default = "default_true")]
    pub save_remote_outputs: bool,
    /// Persisted sidebar width in px; `None` uses the panel default.
    pub nav_rail_width: Option<u32>,
    /// Persisted Generate-inspector width in px; `None` uses the panel default.
    pub generate_params_width: Option<u32>,
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
            saved_hosts: Vec::new(),
            connected_host_ids: Vec::new(),
            generate_target_host: None,
            save_remote_outputs: true,
            nav_rail_width: None,
            generate_params_width: None,
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
            saved_hosts: vec![SavedHost {
                id: "hal9000-7680".into(),
                name: Some("hal9000".into()),
                url: "http://hal9000:7680".into(),
                last_used_ms: Some(1_700_000_000_000),
            }],
            connected_host_ids: vec!["hal9000-7680".into()],
            generate_target_host: Some("hal9000-7680".into()),
            save_remote_outputs: false,
            nav_rail_width: Some(240),
            generate_params_width: Some(360),
        };
        save(&path, &settings).unwrap();
        assert_eq!(load(&path), settings);
    }

    #[test]
    fn legacy_settings_json_defaults_to_no_panel_widths() {
        // Files written before the resizable panels existed must load with
        // both widths unset so the UI falls back to each panel's default.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"local","uiScalePercent":110}"#).unwrap();
        let loaded = load(&path);
        assert_eq!(loaded.nav_rail_width, None);
        assert_eq!(loaded.generate_params_width, None);
    }

    #[test]
    fn legacy_settings_json_defaults_to_saving_remote_outputs() {
        // Files written before the pref existed must load with the local
        // save ON — silently losing remote prints is the worse failure.
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"remote"}"#).unwrap();
        assert!(load(&path).save_remote_outputs);
    }

    #[test]
    fn legacy_settings_json_defaults_to_auto_routing_and_no_extra_hosts() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"local"}"#).unwrap();
        let loaded = load(&path);
        assert!(loaded.connected_host_ids.is_empty());
        assert_eq!(loaded.generate_target_host, None);
    }

    #[test]
    fn legacy_settings_json_defaults_to_no_saved_hosts() {
        let dir = tempfile::tempdir().unwrap();
        let path = path_in(&dir);
        std::fs::write(&path, r#"{"mode":"remote","remoteUrl":"http://h:1"}"#).unwrap();
        assert!(load(&path).saved_hosts.is_empty());
    }

    #[test]
    fn remember_connected_host_is_idempotent() {
        let mut settings = AppSettings::default();
        remember_connected_host(&mut settings, "hal9000-7680");
        remember_connected_host(&mut settings, "hal9000-7680");
        remember_connected_host(&mut settings, "studio-local-7680");
        assert_eq!(
            settings.connected_host_ids,
            vec!["hal9000-7680", "studio-local-7680"]
        );
    }

    #[test]
    fn forget_host_removes_the_boot_reconnect_entry() {
        let mut settings = AppSettings {
            connected_host_ids: vec!["hal9000-7680".into(), "studio-local-7680".into()],
            ..AppSettings::default()
        };
        forget_host(&mut settings, "hal9000-7680");
        // A forgotten host must not resurrect as an extra on the next boot.
        assert_eq!(settings.connected_host_ids, vec!["studio-local-7680"]);
    }

    #[test]
    fn forget_host_clears_the_primary_remote_preference_too() {
        let mut settings = AppSettings {
            mode: ConnectionMode::Remote,
            remote_url: Some("http://hal9000:7680".into()),
            remote_api_key: Some("legacy".into()),
            saved_hosts: vec![SavedHost {
                id: "hal9000-7680".into(),
                name: None,
                url: "http://hal9000:7680".into(),
                last_used_ms: Some(1),
            }],
            ..AppSettings::default()
        };
        // Forgetting the ACTIVE host must clear the reconnect preference —
        // otherwise the next launch re-saves it and Forget is a no-op.
        assert!(forget_host(&mut settings, "hal9000-7680"));
        assert!(settings.saved_hosts.is_empty());
        assert_eq!(settings.remote_url, None);
        assert_eq!(settings.remote_api_key, None);
        assert_eq!(settings.mode, ConnectionMode::Local);
    }

    #[test]
    fn forget_host_leaves_the_primary_alone_for_other_hosts() {
        let mut settings = AppSettings {
            mode: ConnectionMode::Remote,
            remote_url: Some("http://hal9000:7680".into()),
            saved_hosts: vec![SavedHost {
                id: "studio-local-7680".into(),
                name: None,
                url: "http://studio.local:7680".into(),
                last_used_ms: Some(1),
            }],
            ..AppSettings::default()
        };
        assert!(!forget_host(&mut settings, "studio-local-7680"));
        assert!(settings.saved_hosts.is_empty());
        assert_eq!(settings.remote_url.as_deref(), Some("http://hal9000:7680"));
        assert_eq!(settings.mode, ConnectionMode::Remote);
    }

    #[test]
    fn upsert_saved_host_keeps_the_friendly_name_across_nameless_reconnects() {
        let mut hosts = Vec::new();
        upsert_saved_host(
            &mut hosts,
            "hal9000-7680",
            "http://hal9000:7680",
            Some("hal9000".into()),
            1,
        );
        // Boot-time reconnect passes no name; the mDNS name must survive.
        upsert_saved_host(&mut hosts, "hal9000-7680", "http://hal9000:7680", None, 2);
        assert_eq!(hosts[0].name.as_deref(), Some("hal9000"));
        // A fresh explicit name still wins.
        upsert_saved_host(
            &mut hosts,
            "hal9000-7680",
            "http://hal9000:7680",
            Some("renamed".into()),
            3,
        );
        assert_eq!(hosts[0].name.as_deref(), Some("renamed"));
    }

    #[test]
    fn upsert_saved_host_moves_to_front_and_caps() {
        let mut hosts = Vec::new();
        upsert_saved_host(&mut hosts, "a", "http://a:7680", None, 1);
        upsert_saved_host(&mut hosts, "b", "http://b:7680", Some("bee".into()), 2);
        assert_eq!(hosts[0].id, "b");
        assert_eq!(hosts[1].id, "a");

        // Re-connecting to `a` moves it back to the front with a fresh stamp,
        // without duplicating the entry.
        upsert_saved_host(&mut hosts, "a", "http://a:7680", None, 3);
        assert_eq!(hosts.len(), 2);
        assert_eq!(hosts[0].id, "a");
        assert_eq!(hosts[0].last_used_ms, Some(3));

        for i in 0..MAX_SAVED_HOSTS + 3 {
            let id = format!("h{i}");
            upsert_saved_host(&mut hosts, &id, "http://x:7680", None, 10 + i as u64);
        }
        assert_eq!(hosts.len(), MAX_SAVED_HOSTS);
        assert_eq!(hosts[0].id, format!("h{}", MAX_SAVED_HOSTS + 2));
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
